from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator, FormatChecker

CANDIDATE_ID = re.compile(r"^CAND_[0-9]{7}$")
REFERENCE_DATE = date(2026, 6, 20)


def read_candidates(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{source} must contain a JSON array or JSONL records")
    return records


def validate_candidates(
    candidates: Iterable[dict[str, Any]],
    schema_path: str | Path | None = None,
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    validator = None
    if schema_path:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        validator = Draft7Validator(schema, format_checker=FormatChecker())

    for index, candidate in enumerate(candidates):
        label = f"record {index + 1}"
        if not isinstance(candidate, dict):
            errors.append(f"{label}: expected object")
            continue
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not CANDIDATE_ID.fullmatch(cid):
            errors.append(f"{label}: invalid candidate_id {cid!r}")
        elif cid in seen:
            errors.append(f"{label}: duplicate candidate_id {cid}")
        else:
            seen.add(cid)

        if validator:
            for issue in sorted(
                validator.iter_errors(candidate), key=lambda item: list(item.path)
            ):
                location = ".".join(str(part) for part in issue.path) or "<root>"
                errors.append(f"{cid or label} {location}: {issue.message}")
    return errors


def candidate_by_id(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in candidates}


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    return {
        "candidate_id": candidate["candidate_id"],
        "profile": {
            key: profile.get(key)
            for key in (
                "headline",
                "summary",
                "location",
                "country",
                "years_of_experience",
                "current_title",
                "current_company",
                "current_company_size",
                "current_industry",
            )
        },
        "career_history": candidate["career_history"],
        "education": candidate.get("education", []),
        "skills": candidate.get("skills", []),
        "certifications": candidate.get("certifications", []),
        "redrob_signals": {
            key: signals.get(key)
            for key in (
                "last_active_date",
                "open_to_work_flag",
                "recruiter_response_rate",
                "avg_response_time_hours",
                "skill_assessment_scores",
                "notice_period_days",
                "preferred_work_mode",
                "willing_to_relocate",
                "github_activity_score",
                "saved_by_recruiters_30d",
                "interview_completion_rate",
                "offer_acceptance_rate",
                "verified_email",
                "verified_phone",
                "linkedin_connected",
            )
        },
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, values: Iterable[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for value in values:
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
