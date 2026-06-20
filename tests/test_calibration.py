from contextmatch.calibration import (
    build_anchor_and_holdout,
    calibration_report,
    select_calibration_reviews,
)
from contextmatch.models import (
    CalibrationReview,
    CandidateAssessment,
    DimensionScores,
    ScoredCandidate,
)


def make_assessment(cid, total_band):
    retrieval = min(25, total_band)
    remaining = max(total_band - retrieval, 0)
    dimensions = DimensionScores(
        retrieval_ranking=retrieval,
        evaluation_experimentation=min(20, remaining),
        production_ml_python=0,
        product_shipping_outcomes=0,
        ownership_seniority=0,
        nlp_llm_secondary=0,
        logistics_engagement=0,
    )
    return CandidateAssessment(
        candidate_id=cid,
        dimensions=dimensions,
        evidence=["Specific production evidence."],
        confidence=0.9,
    )


def test_selects_requested_calibration_size(candidate_factory):
    candidates = [candidate_factory(i) for i in range(1, 13)]
    scores = []
    for index, candidate in enumerate(candidates):
        assessment = make_assessment(candidate["candidate_id"], 20 + index * 5)
        scores.append(
            ScoredCandidate(
                candidate_id=candidate["candidate_id"],
                assessment=assessment,
                rubric_score=assessment.dimensions.total,
            )
        )
    reviews = select_calibration_reviews(
        {item["candidate_id"]: item for item in candidates}, scores, size=10
    )
    assert len(reviews) == 10
    assert all(review.review_status == "pending" for review in reviews)


def test_builds_anchors_only_after_review(candidate_factory):
    reviews = []
    for index in range(1, 11):
        candidate = candidate_factory(index)
        assessment = make_assessment(candidate["candidate_id"], 50)
        reviews.append(
            CalibrationReview(
                candidate_id=candidate["candidate_id"],
                candidate=candidate,
                draft_assessment=assessment,
                review_status="approved",
                stratum="strong" if index % 2 else "borderline",
            )
        )
    anchors, holdout = build_anchor_and_holdout(reviews, anchor_count=4)
    assert len(anchors) == 4
    assert len(holdout) == 6


def test_calibration_report_passes_close_predictions(candidate_factory):
    expected = []
    predicted = []
    for index in range(1, 6):
        candidate = candidate_factory(index)
        assessment = make_assessment(candidate["candidate_id"], 50)
        expected.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate": candidate,
                "assessment": assessment.model_dump(mode="json"),
            }
        )
        predicted.append(
            ScoredCandidate(
                candidate_id=candidate["candidate_id"],
                assessment=assessment,
                rubric_score=assessment.dimensions.total,
            )
        )
    assert calibration_report(expected, predicted)["passed"] is True
