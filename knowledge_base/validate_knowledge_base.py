#!/usr/bin/env python3
"""Validate the compact actionable historical knowledge base."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

VALID_PRECISIONS = {"year", "month", "day"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_fact(
    prefix: str,
    entry: dict,
    date_field: str,
    errors: list[str],
    *,
    require_patterns: bool = False,
) -> None:
    fact_date = entry.get(date_field)
    if not isinstance(fact_date, str) or not DATE_PATTERN.fullmatch(fact_date):
        errors.append(f"{prefix}: {date_field} must be YYYY-MM-DD")
    if entry.get("precision") not in VALID_PRECISIONS:
        errors.append(f"{prefix}: precision must be year, month, or day")
    source = entry.get("source")
    parsed = urlparse(source or "")
    if parsed.scheme != "https" or not parsed.netloc:
        errors.append(f"{prefix}: source must be an HTTPS URL")
    if not entry.get("source_type"):
        errors.append(f"{prefix}: source_type is required")
    if require_patterns:
        patterns = entry.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            errors.append(f"{prefix}: at least one regex pattern is required")
        else:
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    errors.append(f"{prefix}: invalid pattern {pattern!r}: {error}")


def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_keys = {
        "schema_version",
        "as_of",
        "reference_date",
        "companies",
        "technologies",
        "certifications",
    }
    if set(data) != expected_keys:
        errors.append(
            f"root keys must be exactly {sorted(expected_keys)}"
        )
    if data.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    for field in ("as_of", "reference_date"):
        if not isinstance(data.get(field), str) or not DATE_PATTERN.fullmatch(
            data[field]
        ):
            errors.append(f"{field} must be YYYY-MM-DD")

    for section in ("companies", "technologies", "certifications"):
        entries = data.get(section)
        if not isinstance(entries, dict):
            errors.append(f"{section}: expected object")
            continue
        if list(entries) != sorted(entries):
            errors.append(f"{section}: keys must be alphabetically sorted")

    for name, entry in data.get("companies", {}).items():
        _validate_fact(f"companies.{name}", entry, "founded_date", errors)
    for name, entry in data.get("technologies", {}).items():
        _validate_fact(
            f"technologies.{name}",
            entry,
            "released_date",
            errors,
            require_patterns=True,
        )
    for name, entry in data.get("certifications", {}).items():
        _validate_fact(
            f"certifications.{name}", entry, "available_date", errors
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("knowledge_base.json"))
    args = parser.parse_args()
    errors = validate(args.input)
    if errors:
        print(f"Knowledge-base validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(
        "Knowledge-base validation passed: "
        f"{len(data['companies'])} companies, "
        f"{len(data['technologies'])} technologies, "
        f"{len(data['certifications'])} certifications."
    )


if __name__ == "__main__":
    main()
