from __future__ import annotations

from .models import CandidateAssessment, Disqualifier, DimensionScores

RUBRIC_VERSION = "redrob-v1"
EXPERT_ZERO_USAGE_PENALTY = 5.0
MAX_EXPERT_ZERO_USAGE_PENALTY = 10.0

CAPS: dict[Disqualifier, int] = {
    Disqualifier.PURE_RESEARCH: 30,
    Disqualifier.RECENT_LLM_ONLY: 30,
    Disqualifier.NO_RECENT_CODE: 40,
    Disqualifier.SERVICES_ONLY: 40,
    Disqualifier.DOMAIN_MISMATCH: 45,
}


def average_dimensions(
    first: DimensionScores, second: DimensionScores
) -> DimensionScores:
    values = {}
    for key, value in first.model_dump().items():
        values[key] = round((value + getattr(second, key)) / 2)
    return DimensionScores(**values)


def merge_assessments(
    first: CandidateAssessment, second: CandidateAssessment
) -> CandidateAssessment:
    if first.candidate_id != second.candidate_id:
        raise ValueError("cannot merge assessments for different candidates")
    return CandidateAssessment(
        candidate_id=first.candidate_id,
        dimensions=average_dimensions(first.dimensions, second.dimensions),
        evidence=list(dict.fromkeys(first.evidence + second.evidence))[:5],
        concerns=list(dict.fromkeys(first.concerns + second.concerns))[:5],
        disqualifiers=sorted(
            set(first.disqualifiers) | set(second.disqualifiers),
            key=lambda item: item.value,
        ),
        conflicting_evidence=(
            first.conflicting_evidence or second.conflicting_evidence
        ),
        confidence=(first.confidence + second.confidence) / 2,
    )


def calculate_score(
    assessment: CandidateAssessment, integrity_issues: list[str] | None = None
) -> tuple[float, int | None]:
    raw = float(assessment.dimensions.total)
    if integrity_issues:
        return 0.0, 0
    caps = [CAPS[flag] for flag in assessment.disqualifiers if flag in CAPS]
    cap = min(caps) if caps else None
    return (min(raw, cap) if cap is not None else raw), cap


def expert_zero_usage_penalty(finding_rules: list[str]) -> float:
    count = sum(rule == "expert_skill_zero_usage" for rule in finding_rules)
    return min(count * EXPERT_ZERO_USAGE_PENALTY, MAX_EXPERT_ZERO_USAGE_PENALTY)
