from contextmatch.models import CandidateAssessment, DimensionScores, ScoredCandidate
from contextmatch.output import validate_submission, write_submission


def make_scored(index):
    cid = f"CAND_{index:07d}"
    assessment = CandidateAssessment(
        candidate_id=cid,
        dimensions=DimensionScores(
            retrieval_ranking=20,
            evaluation_experimentation=15,
            production_ml_python=12,
            product_shipping_outcomes=10,
            ownership_seniority=8,
            nlp_llm_secondary=3,
            logistics_engagement=7,
        ),
        evidence=["Owned production search and ranking infrastructure."],
        confidence=0.9,
    )
    return ScoredCandidate(
        candidate_id=cid,
        assessment=assessment,
        rubric_score=75,
        final_score=101 - index,
    )


def test_writes_and_validates_submission(tmp_path):
    ranked = [make_scored(index) for index in range(1, 101)]
    reasonings = {
        item.candidate_id: (
            f"Candidate {item.candidate_id} owned production search ranking systems "
            "with relevant evaluation and engineering evidence."
        )
        for item in ranked
    }
    path = tmp_path / "team.csv"
    write_submission(path, ranked, reasonings)
    errors = validate_submission(path, set(reasonings))
    assert errors == []


def test_validator_rejects_nonfinite_score(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "candidate_id,rank,score,reasoning\n"
        "CAND_0000001,1,nan,This reasoning contains enough factual words for validation today.\n",
        encoding="utf-8",
    )
    errors = validate_submission(path, {"CAND_0000001"}, expected_rows=1)
    assert any("finite" in error for error in errors)
