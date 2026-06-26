from contextmatch.models import CandidateAssessment, DimensionScores, Disqualifier
from contextmatch.rubric import (
    calculate_score,
    expert_zero_usage_penalty,
    merge_assessments,
)


def assessment(score_offset=0, flags=None):
    return CandidateAssessment(
        candidate_id="CAND_0000001",
        dimensions=DimensionScores(
            retrieval_ranking=25 - score_offset,
            evaluation_experimentation=20,
            production_ml_python=15,
            product_shipping_outcomes=15,
            ownership_seniority=10,
            nlp_llm_secondary=5,
            logistics_engagement=10,
        ),
        evidence=["Owned production search ranking."],
        concerns=[],
        disqualifiers=flags or [],
        confidence=0.9,
    )


def test_score_is_dimension_sum():
    assert calculate_score(assessment()) == (100.0, None)


def test_disqualifier_applies_lowest_cap():
    value, cap = calculate_score(
        assessment(
            flags=[Disqualifier.SERVICES_ONLY, Disqualifier.RECENT_LLM_ONLY]
        )
    )
    assert value == 30
    assert cap == 30


def test_integrity_always_caps_at_zero():
    assert calculate_score(assessment(), ["impossible"])[0] == 0


def test_expert_zero_usage_penalty_is_limited():
    assert expert_zero_usage_penalty(["expert_skill_zero_usage"]) == 5
    assert (
        expert_zero_usage_penalty(
            ["expert_skill_zero_usage", "expert_skill_zero_usage", "other"]
        )
        == 10
    )


def test_repeat_merge_averages_dimensions_and_unions_flags():
    first = assessment()
    second = assessment(10, [Disqualifier.NO_RECENT_CODE])
    merged = merge_assessments(first, second)
    assert merged.dimensions.retrieval_ranking == 20
    assert merged.disqualifiers == [Disqualifier.NO_RECENT_CODE]
