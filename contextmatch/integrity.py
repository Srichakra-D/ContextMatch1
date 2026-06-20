from __future__ import annotations

from datetime import date
from typing import Any

from .data import REFERENCE_DATE


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def verified_integrity_issues(candidate: dict[str, Any]) -> list[str]:
    """Return high-confidence contradictions suitable for a hard score cap."""
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

    zero_usage_expert = [
        skill.get("name", "unknown")
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert"
        and skill.get("duration_months") == 0
    ]
    if zero_usage_expert:
        issues.append(
            "expert proficiency with zero usage duration: "
            + ", ".join(zero_usage_expert)
        )
    return issues
