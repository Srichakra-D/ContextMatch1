from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .data import REFERENCE_DATE, read_jsonl, write_json, write_jsonl
from .models import (
    CandidateIntegrity,
    IntegrityFinding,
    IntegrityStatus,
)

SKILL_DURATION_TOLERANCE_MONTHS = 2
AMBIGUOUS_ROLE_TEXT_TECHNOLOGIES = {
    "Go",
    "React",
    "Rust",
    "Spark",
    "Flask",
}


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_knowledge_base(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    knowledge_base = json.loads(source.read_text(encoding="utf-8"))
    required = {"metadata", "companies", "technologies", "certifications"}
    if set(knowledge_base) != required:
        raise ValueError(
            "knowledge base must contain metadata, companies, technologies, "
            "and certifications"
        )
    return knowledge_base, _file_sha256(source)


def load_integrity_report(
    path: str | Path,
) -> dict[str, CandidateIntegrity]:
    adapter = TypeAdapter(list[CandidateIntegrity])
    records = adapter.validate_python(read_jsonl(path))
    result = {record.candidate_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("integrity report contains duplicate candidate IDs")
    return result


def _fact_date(entry: dict[str, Any]) -> date:
    value = entry["date"]
    year = int(value[:4])
    month = int(value[5:7]) if len(value) >= 7 else 1
    day = int(value[8:10]) if len(value) >= 10 else 1
    return date(year, month, day)


def _strictly_before(candidate_date: date, entry: dict[str, Any]) -> bool:
    fact = _fact_date(entry)
    if entry["date_precision"] == "year":
        return candidate_date.year < fact.year
    if entry["date_precision"] == "month":
        return (candidate_date.year, candidate_date.month) < (
            fact.year,
            fact.month,
        )
    return candidate_date < fact


def _finding(
    *,
    rule: str,
    severity: str,
    message: str,
    entity_type: str | None = None,
    entity_name: str | None = None,
    fact: dict[str, Any] | None = None,
) -> IntegrityFinding:
    return IntegrityFinding(
        rule=rule,
        severity=severity,
        message=message,
        entity_type=entity_type,
        entity_name=entity_name,
        fact_date=fact.get("date") if fact else None,
        fact_date_precision=fact.get("date_precision") if fact else None,
        sources=fact.get("sources", []) if fact else [],
    )


def _matches_entity(text: str, names: list[str]) -> bool:
    return any(
        re.search(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", text)
        for name in names
        if len(name) >= 2
    )


def scan_candidate_integrity(
    candidate: dict[str, Any],
    knowledge_base: dict[str, Any],
    knowledge_base_sha256: str,
) -> CandidateIntegrity:
    findings: list[IntegrityFinding] = []
    profile = candidate["profile"]
    careers = candidate["career_history"]
    current = [role for role in careers if role.get("is_current")]

    if len(current) != 1:
        findings.append(
            _finding(
                rule="current_role_count",
                severity="verified_failure",
                message=f"Expected exactly one current role; found {len(current)}.",
            )
        )
    elif (
        current[0].get("company") != profile.get("current_company")
        or current[0].get("title") != profile.get("current_title")
        or current[0].get("end_date") is not None
    ):
        findings.append(
            _finding(
                rule="current_role_profile_mismatch",
                severity="verified_failure",
                message="Current career role contradicts profile company/title.",
            )
        )

    technology_entries = knowledge_base["technologies"]
    for role in careers:
        try:
            start = date.fromisoformat(role["start_date"])
            end = (
                REFERENCE_DATE
                if role.get("end_date") is None
                else date.fromisoformat(role["end_date"])
            )
        except (KeyError, TypeError, ValueError):
            findings.append(
                _finding(
                    rule="invalid_role_dates",
                    severity="verified_failure",
                    message=f"Invalid dates in role at {role.get('company', 'unknown')}.",
                )
            )
            continue

        if start > end or start > REFERENCE_DATE or end > REFERENCE_DATE:
            findings.append(
                _finding(
                    rule="impossible_role_dates",
                    severity="verified_failure",
                    message=(
                        f"Role at {role.get('company', 'unknown')} has impossible "
                        f"dates {start} to {end}."
                    ),
                )
            )

        stated = role.get("duration_months")
        calculated = _months_between(start, end)
        if isinstance(stated, int) and abs(stated - calculated) > 2:
            findings.append(
                _finding(
                    rule="role_duration_mismatch",
                    severity="verified_failure",
                    message=(
                        f"Role at {role.get('company', 'unknown')} states {stated} "
                        f"months but dates imply approximately {calculated}."
                    ),
                )
            )

        company = role.get("company")
        company_fact = knowledge_base["companies"].get(company)
        if (
            company_fact
            and company_fact["status"] == "verified"
            and _strictly_before(start, company_fact)
        ):
            findings.append(
                _finding(
                    rule="employment_before_company_start",
                    severity="verified_failure",
                    message=(
                        f"Employment at {company} starts {start}, before its "
                        f"verified earliest start {company_fact['date']}."
                    ),
                    entity_type="company",
                    entity_name=company,
                    fact=company_fact,
                )
            )

        description = (role.get("description") or "").casefold()
        for technology_name, technology_fact in technology_entries.items():
            if technology_fact["status"] != "verified":
                continue
            if technology_name in AMBIGUOUS_ROLE_TEXT_TECHNOLOGIES:
                continue
            names = [technology_name, *technology_fact.get("aliases", [])]
            if _matches_entity(description, names) and _strictly_before(
                end, technology_fact
            ):
                findings.append(
                    _finding(
                        rule="technology_claim_before_release",
                        severity="verified_failure",
                        message=(
                            f"Role ending {end} explicitly claims {technology_name}, "
                            f"released {technology_fact['date']}."
                        ),
                        entity_type="technology",
                        entity_name=technology_name,
                        fact=technology_fact,
                    )
                )

    zero_usage_expert = [
        skill.get("name", "unknown")
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert"
        and skill.get("duration_months") == 0
    ]
    if zero_usage_expert:
        findings.append(
            _finding(
                rule="expert_skill_zero_usage",
                severity="verified_failure",
                message=(
                    "Expert proficiency with zero usage duration: "
                    + ", ".join(zero_usage_expert)
                    + "."
                ),
            )
        )

    for skill in candidate.get("skills", []):
        technology_fact = technology_entries.get(skill.get("name"))
        duration = skill.get("duration_months")
        if (
            not technology_fact
            or technology_fact["status"] != "verified"
            or not isinstance(duration, int)
        ):
            continue
        release = _fact_date(technology_fact)
        possible = (
            _months_between(release, REFERENCE_DATE)
            + SKILL_DURATION_TOLERANCE_MONTHS
        )
        if duration > possible:
            findings.append(
                _finding(
                    rule="skill_duration_exceeds_technology_age",
                    severity="suspicious",
                    message=(
                        f"{skill['name']} claims {duration} months of use, while "
                        f"approximately {possible} months are possible since "
                        f"{technology_fact['date']}."
                    ),
                    entity_type="technology",
                    entity_name=skill["name"],
                    fact=technology_fact,
                )
            )

    for certification in candidate.get("certifications", []):
        key = f"{certification['name']}|{certification['issuer']}"
        certification_fact = knowledge_base["certifications"].get(key)
        if (
            certification_fact
            and certification_fact["status"] == "verified"
            and certification["year"] < int(certification_fact["date"][:4])
        ):
            findings.append(
                _finding(
                    rule="certification_before_launch",
                    severity="suspicious",
                    message=(
                        f"{certification['name']} is dated {certification['year']}, "
                        f"before verified availability in "
                        f"{certification_fact['date']}; year-level synthetic "
                        "metadata is retained for audit only."
                    ),
                    entity_type="certification",
                    entity_name=key,
                    fact=certification_fact,
                )
            )

    if any(item.severity == "verified_failure" for item in findings):
        status = IntegrityStatus.VERIFIED_FAILURE
    elif findings:
        status = IntegrityStatus.SUSPICIOUS
    else:
        status = IntegrityStatus.CLEAN

    return CandidateIntegrity(
        candidate_id=candidate["candidate_id"],
        status=status,
        findings=findings,
        knowledge_base_schema_version=knowledge_base["metadata"]["schema_version"],
        knowledge_base_sha256=knowledge_base_sha256,
    )


def scan_candidates(
    candidates: list[dict[str, Any]],
    knowledge_base: dict[str, Any],
    knowledge_base_sha256: str,
) -> list[CandidateIntegrity]:
    return [
        scan_candidate_integrity(candidate, knowledge_base, knowledge_base_sha256)
        for candidate in candidates
    ]


def verified_integrity_issues(candidate: dict[str, Any]) -> list[str]:
    """Compatibility helper for deterministic checks that need no knowledge base."""
    issues: list[str] = []
    profile = candidate["profile"]
    careers = candidate["career_history"]
    current = [role for role in careers if role.get("is_current")]
    if len(current) != 1:
        issues.append(f"expected exactly one current role; found {len(current)}")
    elif (
        current[0].get("company") != profile.get("current_company")
        or current[0].get("title") != profile.get("current_title")
        or current[0].get("end_date") is not None
    ):
        issues.append("current role contradicts profile current company/title")
    for role in careers:
        try:
            start = date.fromisoformat(role["start_date"])
            end = (
                REFERENCE_DATE
                if role.get("end_date") is None
                else date.fromisoformat(role["end_date"])
            )
        except (KeyError, TypeError, ValueError):
            issues.append(f"invalid dates in role at {role.get('company', 'unknown')}")
            continue
        stated = role.get("duration_months")
        calculated = _months_between(start, end)
        if isinstance(stated, int) and abs(stated - calculated) > 2:
            issues.append(
                f"role duration contradiction at {role.get('company', 'unknown')}: "
                f"stated {stated}, calculated approximately {calculated} months"
            )
    zero_usage = [
        skill.get("name", "unknown")
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert"
        and skill.get("duration_months") == 0
    ]
    if zero_usage:
        issues.append(
            "expert proficiency with zero usage duration: " + ", ".join(zero_usage)
        )
    return issues


def write_integrity_outputs(
    output_dir: str | Path, records: list[CandidateIntegrity]
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    verified = [
        item for item in records if item.status == IntegrityStatus.VERIFIED_FAILURE
    ]
    suspicious = [
        item for item in records if item.status == IntegrityStatus.SUSPICIOUS
    ]
    counts = {
        status.value: sum(item.status == status for item in records)
        for status in IntegrityStatus
    }
    rule_counts: dict[str, int] = {}
    for record in records:
        for finding in record.findings:
            rule_counts[finding.rule] = rule_counts.get(finding.rule, 0) + 1
    summary = {
        "candidate_count": len(records),
        "status_counts": counts,
        "finding_counts_by_rule": dict(sorted(rule_counts.items())),
        "knowledge_base_schema_version": (
            records[0].knowledge_base_schema_version if records else None
        ),
        "knowledge_base_sha256": (
            records[0].knowledge_base_sha256 if records else None
        ),
    }
    write_jsonl(destination / "integrity_report.jsonl", records)
    write_jsonl(destination / "verified_failures.jsonl", verified)
    write_jsonl(destination / "suspicious_candidates.jsonl", suspicious)
    write_json(destination / "summary.json", summary)
    return summary
