#!/usr/bin/env python3
"""Normalize and validate a compact manually researched knowledge base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate_knowledge_base import validate
except ImportError:
    from validate_knowledge_base import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("knowledge_base1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("knowledge_base.json")
    )
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    compact = {
        "schema_version": 2,
        "as_of": source.get("as_of", "2026-06-23"),
        "reference_date": "2026-06-20",
        "companies": {},
        "technologies": {},
        "certifications": source.get("certifications", {}),
    }
    for name, entry in sorted(source.get("companies", {}).items()):
        compact["companies"][name] = {
            **entry,
            "source_type": entry.get("source_type", "external_research"),
        }
    for name, entry in sorted(source.get("technologies", {}).items()):
        compact["technologies"][name] = {
            **entry,
            "precision": entry.get("precision", "day"),
            "source_type": entry.get("source_type", "external_research"),
        }
    errors = validate_data(compact)
    if errors:
        raise SystemExit("\n".join(errors))
    args.output.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote compact knowledge base to {args.output}")


def validate_data(data: dict) -> list[str]:
    temporary = Path("/tmp/contextmatch-knowledge-base.json")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    try:
        return validate(temporary)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
