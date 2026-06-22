#!/usr/bin/env python3
"""Validate knowledge_base.json coverage and internal consistency."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

VALID_STATUSES = {"verified", "ambiguous", "unknown", "fictional", "not_dateable"}
VALID_PRECISION = {"year", "month", "day"}
DATE_PATTERNS = {
    "year": re.compile(r"^\d{4}$"),
    "month": re.compile(r"^\d{4}-(0[1-9]|1[0-2])$"),
    "day": re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
}


def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    metadata = data.get("metadata", {})
    expected = {
        "companies": metadata.get("company_count"),
        "technologies": metadata.get("technology_count"),
        "certifications": metadata.get("certification_count"),
    }
    all_aliases: dict[str, str] = {}

    for section, expected_count in expected.items():
        entries = data.get(section)
        if not isinstance(entries, dict):
            errors.append(f"{section}: expected object")
            continue
        if len(entries) != expected_count:
            errors.append(
                f"{section}: metadata says {expected_count}, found {len(entries)}"
            )
        if list(entries) != sorted(entries):
            errors.append(f"{section}: keys must be alphabetically sorted")
        for key, entry in entries.items():
            prefix = f"{section}.{key}"
            status = entry.get("status")
            if status not in VALID_STATUSES:
                errors.append(f"{prefix}: invalid status {status!r}")
            if entry.get("canonical_name") != key:
                errors.append(f"{prefix}: canonical_name must match key")
            if not isinstance(entry.get("occurrences"), int) or entry["occurrences"] < 1:
                errors.append(f"{prefix}: occurrences must be a positive integer")

            fact_date = entry.get("date")
            precision = entry.get("date_precision")
            basis = entry.get("date_basis")
            sources = entry.get("sources")
            notes = entry.get("notes")
            if status in {"verified", "ambiguous"}:
                if precision not in VALID_PRECISION:
                    errors.append(f"{prefix}: dated entry needs valid precision")
                elif not isinstance(fact_date, str) or not DATE_PATTERNS[
                    precision
                ].fullmatch(fact_date):
                    errors.append(
                        f"{prefix}: date {fact_date!r} does not match {precision}"
                    )
                if not basis:
                    errors.append(f"{prefix}: dated entry needs date_basis")
                if not sources:
                    errors.append(f"{prefix}: dated entry needs a source")
            else:
                if any(value is not None for value in (fact_date, precision, basis)):
                    errors.append(f"{prefix}: undated status must have null date fields")

            if status == "verified":
                for source in sources:
                    parsed = urlparse(source.get("url", ""))
                    if parsed.scheme != "https" or not parsed.netloc:
                        errors.append(f"{prefix}: source URL must be HTTPS")
                    for field in (
                        "title",
                        "publisher",
                        "source_type",
                        "retrieved_on",
                    ):
                        if not source.get(field):
                            errors.append(f"{prefix}: source missing {field}")
            if status in {"ambiguous", "unknown", "fictional", "not_dateable"} and not notes:
                errors.append(f"{prefix}: status {status} requires notes")

            for alias in entry.get("aliases", []):
                normalized = alias.casefold()
                if normalized in all_aliases and all_aliases[normalized] != key:
                    errors.append(
                        f"{prefix}: alias {alias!r} also maps to "
                        f"{all_aliases[normalized]!r}"
                    )
                all_aliases[normalized] = key

    if metadata.get("source_candidate_count") != 100000:
        errors.append("metadata.source_candidate_count must be 100000")
    if metadata.get("company_count") != 63:
        errors.append("metadata.company_count must be 63")
    if metadata.get("technology_count") != 133:
        errors.append("metadata.technology_count must be 133")
    if metadata.get("certification_count") != 8:
        errors.append("metadata.certification_count must be 8")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("knowledge_base.json")
    )
    args = parser.parse_args()
    errors = validate(args.input)
    if errors:
        print(f"Knowledge-base validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for section in ("companies", "technologies", "certifications"):
        statuses: dict[str, int] = {}
        for entry in data[section].values():
            statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        print(f"{section}: {len(data[section])} entries {statuses}")
    print("Knowledge-base validation passed.")


if __name__ == "__main__":
    main()
