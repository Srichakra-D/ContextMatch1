from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .models import ScoredCandidate


def _write_csv(
    path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_calibration_comparison(
    path: str | Path,
    expected: list[dict[str, Any]],
    predicted: list[ScoredCandidate],
) -> None:
    predicted_by_id = {item.candidate_id: item for item in predicted}
    rows = []
    for item in expected:
        actual = predicted_by_id[item["candidate_id"]]
        expected_assessment = item["assessment"]
        expected_score = sum(expected_assessment["dimensions"].values())
        predicted_score = actual.assessment.dimensions.total
        expected_flags = sorted(expected_assessment.get("disqualifiers", []))
        predicted_flags = sorted(
            flag.value for flag in actual.assessment.disqualifiers
        )
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "stratum": item.get("stratum", ""),
                "reviewed_score": expected_score,
                "predicted_score": predicted_score,
                "absolute_error": abs(expected_score - predicted_score),
                "reviewed_disqualifiers": "|".join(expected_flags),
                "predicted_disqualifiers": "|".join(predicted_flags),
                "disqualifiers_match": expected_flags == predicted_flags,
                "predicted_confidence": actual.assessment.confidence,
                "integrity_status": actual.integrity_status.value,
            }
        )
    _write_csv(
        path,
        [
            "candidate_id",
            "stratum",
            "reviewed_score",
            "predicted_score",
            "absolute_error",
            "reviewed_disqualifiers",
            "predicted_disqualifiers",
            "disqualifiers_match",
            "predicted_confidence",
            "integrity_status",
        ],
        rows,
    )


def write_individual_ranking(
    path: str | Path,
    scores: list[ScoredCandidate],
    initial_snapshot: dict[str, dict[str, Any]],
) -> list[ScoredCandidate]:
    ranked = sorted(
        scores, key=lambda item: (-item.rubric_score, item.candidate_id)
    )
    rows = []
    for rank, item in enumerate(ranked, start=1):
        initial = initial_snapshot[item.candidate_id]
        current = (
            item.adjudicated_assessment
            or item.repeated_assessment
            or item.assessment
        )
        rows.append(
            {
                "stage_2_rank": rank,
                "candidate_id": item.candidate_id,
                "initial_pass_rank": initial["rank"],
                "initial_pass_score": initial["score"],
                "stage_2_individual_score": item.rubric_score,
                "rank_change": initial["rank"] - rank,
                "was_repeated": item.repeated_assessment is not None,
                "was_adjudicated": item.adjudicated_assessment is not None,
                "confidence": current.confidence,
                "applied_cap": (
                    "" if item.applied_cap is None else item.applied_cap
                ),
                "disqualifiers": "|".join(
                    flag.value for flag in current.disqualifiers
                ),
                "integrity_issue_count": len(item.integrity_issues),
                "integrity_status": item.integrity_status.value,
                "integrity_reasons": "|".join(
                    finding.message for finding in item.integrity_findings
                ),
                "knowledge_facts_used": "|".join(
                    sorted(
                        {
                            f"{finding.entity_type}:{finding.entity_name}"
                            for finding in item.integrity_findings
                            if finding.entity_type and finding.entity_name
                        }
                    )
                ),
                "knowledge_base_schema_version": (
                    ""
                    if item.knowledge_base_schema_version is None
                    else item.knowledge_base_schema_version
                ),
            }
        )
    _write_csv(
        path,
        [
            "stage_2_rank",
            "candidate_id",
            "initial_pass_rank",
            "initial_pass_score",
            "stage_2_individual_score",
            "rank_change",
            "was_repeated",
            "was_adjudicated",
            "confidence",
            "applied_cap",
            "disqualifiers",
            "integrity_issue_count",
            "integrity_status",
            "integrity_reasons",
            "knowledge_facts_used",
            "knowledge_base_schema_version",
        ],
        rows,
    )
    return ranked


def write_comparative_ranking(
    path: str | Path,
    final_ranked: list[ScoredCandidate],
    individual_ranked: list[ScoredCandidate],
    *,
    comparative_top_n: int = 150,
) -> None:
    individual_rank = {
        item.candidate_id: rank
        for rank, item in enumerate(individual_ranked, start=1)
    }
    rows = []
    for final_rank, item in enumerate(final_ranked, start=1):
        previous_rank = individual_rank[item.candidate_id]
        rows.append(
            {
                "stage_3_final_rank": final_rank,
                "candidate_id": item.candidate_id,
                "stage_2_individual_rank": previous_rank,
                "stage_2_individual_score": item.rubric_score,
                "comparative_percentile": (
                    ""
                    if item.comparative_percentile is None
                    else round(item.comparative_percentile, 6)
                ),
                "stage_3_final_score": item.final_score,
                "rank_change": previous_rank - final_rank,
                "compared_in_top_group": previous_rank <= comparative_top_n,
                "selected_top_100": final_rank <= 100,
                "integrity_status": item.integrity_status.value,
                "integrity_reasons": "|".join(
                    finding.message for finding in item.integrity_findings
                ),
            }
        )
    _write_csv(
        path,
        [
            "stage_3_final_rank",
            "candidate_id",
            "stage_2_individual_rank",
            "stage_2_individual_score",
            "comparative_percentile",
            "stage_3_final_score",
            "rank_change",
            "compared_in_top_group",
            "selected_top_100",
            "integrity_status",
            "integrity_reasons",
        ],
        rows,
    )
