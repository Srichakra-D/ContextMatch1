import csv

from contextmatch.models import CandidateAssessment, DimensionScores, ScoredCandidate
from contextmatch.stage_outputs import (
    write_calibration_comparison,
    write_comparative_ranking,
    write_individual_ranking,
)


def make_scored(index: int, score: int) -> ScoredCandidate:
    cid = f"CAND_{index:07d}"
    dimensions = DimensionScores(
        retrieval_ranking=min(score, 25),
        evaluation_experimentation=min(max(score - 25, 0), 20),
        production_ml_python=min(max(score - 45, 0), 15),
        product_shipping_outcomes=min(max(score - 60, 0), 15),
        ownership_seniority=min(max(score - 75, 0), 10),
        nlp_llm_secondary=min(max(score - 85, 0), 5),
        logistics_engagement=min(max(score - 90, 0), 10),
    )
    assessment = CandidateAssessment(
        candidate_id=cid,
        dimensions=dimensions,
        evidence=["Specific production ranking evidence."],
        confidence=0.9,
    )
    return ScoredCandidate(
        candidate_id=cid,
        assessment=assessment,
        rubric_score=assessment.dimensions.total,
        final_score=float(assessment.dimensions.total),
    )


def test_writes_all_stage_comparison_files(tmp_path):
    scores = [make_scored(1, 90), make_scored(2, 80), make_scored(3, 70)]
    expected = [
        {
            "candidate_id": scores[0].candidate_id,
            "stratum": "excellent",
            "assessment": scores[0].assessment.model_dump(mode="json"),
        }
    ]
    calibration_path = tmp_path / "stage1.csv"
    write_calibration_comparison(calibration_path, expected, [scores[0]])

    snapshot = {
        item.candidate_id: {"rank": rank, "score": item.rubric_score}
        for rank, item in enumerate(scores, start=1)
    }
    individual_path = tmp_path / "stage2.csv"
    individual = write_individual_ranking(individual_path, scores, snapshot)

    scores[0].comparative_percentile = 100
    scores[0].final_score = 92
    scores[1].comparative_percentile = 50
    scores[1].final_score = 74
    scores[2].comparative_percentile = 0
    scores[2].final_score = 56
    final = sorted(scores, key=lambda item: -item.final_score)
    comparative_path = tmp_path / "stage3.csv"
    write_comparative_ranking(comparative_path, final, individual)

    with calibration_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["absolute_error"] == "0"
    with individual_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["stage_2_rank"] == "1"
    with comparative_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["stage_3_final_rank"] == "1"
