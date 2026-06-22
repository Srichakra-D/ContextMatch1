from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import CalibrationReview, IntegrityStatus, ScoredCandidate


def score_stratum(scored: ScoredCandidate) -> str:
    if scored.integrity_status == IntegrityStatus.SUSPICIOUS:
        return "suspicious"
    if scored.assessment.disqualifiers:
        return "disqualified"
    if scored.rubric_score >= 80:
        return "excellent"
    if scored.rubric_score >= 65:
        return "strong"
    if scored.rubric_score >= 45:
        return "borderline"
    return "weak"


def select_calibration_reviews(
    candidates_by_id: dict[str, dict[str, Any]],
    scores: list[ScoredCandidate],
    size: int = 40,
) -> list[CalibrationReview]:
    if size < 6:
        raise ValueError("calibration size must be at least 6")
    buckets: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for scored in sorted(scores, key=lambda item: (-item.rubric_score, item.candidate_id)):
        if scored.integrity_status == IntegrityStatus.VERIFIED_FAILURE:
            continue
        buckets[score_stratum(scored)].append(scored)

    order = [
        "excellent",
        "strong",
        "borderline",
        "weak",
        "disqualified",
        "suspicious",
    ]
    selected: list[ScoredCandidate] = []
    cursor = 0
    while len(selected) < size:
        made_progress = False
        for name in order:
            if cursor < len(buckets[name]):
                selected.append(buckets[name][cursor])
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            break
        cursor += 1

    already = {item.candidate_id for item in selected}
    if len(selected) < size:
        for scored in sorted(scores, key=lambda item: item.candidate_id):
            if (
                scored.integrity_status != IntegrityStatus.VERIFIED_FAILURE
                and scored.candidate_id not in already
            ):
                selected.append(scored)
                if len(selected) == size:
                    break

    return [
        CalibrationReview(
            candidate_id=scored.candidate_id,
            candidate=candidates_by_id[scored.candidate_id],
            draft_assessment=scored.assessment,
            review_status="pending",
            stratum=score_stratum(scored),
        )
        for scored in selected
    ]


def build_anchor_and_holdout(
    reviews: list[CalibrationReview], anchor_count: int = 8
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(reviews) <= anchor_count:
        raise ValueError("review set must be larger than anchor count")
    effective = [(review, review.effective_assessment()) for review in reviews]
    buckets: dict[str, list[tuple[CalibrationReview, Any]]] = defaultdict(list)
    for item in effective:
        buckets[item[0].stratum].append(item)

    anchor_items: list[tuple[CalibrationReview, Any]] = []
    strata = sorted(buckets)
    cursor = 0
    while len(anchor_items) < anchor_count:
        progressed = False
        for stratum in strata:
            if cursor < len(buckets[stratum]):
                anchor_items.append(buckets[stratum][cursor])
                progressed = True
                if len(anchor_items) == anchor_count:
                    break
        if not progressed:
            break
        cursor += 1

    anchor_ids = {review.candidate_id for review, _ in anchor_items}
    anchors = [
        {
            "candidate_id": review.candidate_id,
            "candidate": review.candidate,
            "assessment": assessment.model_dump(mode="json"),
            "stratum": review.stratum,
        }
        for review, assessment in anchor_items
    ]
    holdout = [
        {
            "candidate_id": review.candidate_id,
            "candidate": review.candidate,
            "assessment": assessment.model_dump(mode="json"),
            "stratum": review.stratum,
        }
        for review, assessment in effective
        if review.candidate_id not in anchor_ids
    ]
    return anchors, holdout


def calibration_report(
    expected: list[dict[str, Any]], predicted: list[ScoredCandidate]
) -> dict[str, Any]:
    predicted_by_id = {item.candidate_id: item for item in predicted}
    differences: list[float] = []
    disqualifier_misses: list[str] = []
    rows = []
    for item in expected:
        target = item["assessment"]
        target_total = sum(target["dimensions"].values())
        actual = predicted_by_id[item["candidate_id"]]
        difference = abs(target_total - actual.assessment.dimensions.total)
        differences.append(difference)
        expected_flags = set(target.get("disqualifiers", []))
        actual_flags = {flag.value for flag in actual.assessment.disqualifiers}
        if expected_flags - actual_flags:
            disqualifier_misses.append(item["candidate_id"])
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "expected_total": target_total,
                "predicted_total": actual.assessment.dimensions.total,
                "absolute_error": difference,
            }
        )
    within_ten = sum(value <= 10 for value in differences) / max(len(differences), 1)
    mean_error = sum(differences) / max(len(differences), 1)
    passed = (
        within_ten >= 0.80
        and mean_error <= 8
        and not disqualifier_misses
    )
    return {
        "passed": passed,
        "mean_absolute_error": round(mean_error, 3),
        "fraction_within_10_points": round(within_ten, 3),
        "disqualifier_misses": disqualifier_misses,
        "rows": rows,
    }
