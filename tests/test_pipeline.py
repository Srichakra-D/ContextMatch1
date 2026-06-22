import asyncio
import csv
import json

from contextmatch.models import (
    CandidateAssessment,
    CandidateIntegrity,
    ComparisonResult,
    DimensionScores,
    IntegrityStatus,
    ReasoningResult,
)
from contextmatch.output import validate_submission
from contextmatch.pipeline import run_full_pipeline


class FakeClient:
    model = "fake-model"

    async def complete_json(
        self,
        messages,
        result_type,
        *,
        temperature=0,
        max_tokens=700,
        thinking=False,
    ):
        content = messages[-1]["content"]
        if result_type is CandidateAssessment:
            candidate = json.loads(content.split("CANDIDATE:\n", 1)[1])
            return CandidateAssessment(
                candidate_id=candidate["candidate_id"],
                dimensions=DimensionScores(
                    retrieval_ranking=20,
                    evaluation_experimentation=15,
                    production_ml_python=12,
                    product_shipping_outcomes=10,
                    ownership_seniority=8,
                    nlp_llm_secondary=3,
                    logistics_engagement=7,
                ),
                evidence=[
                    "Owned production hybrid retrieval and NDCG evaluation."
                ],
                confidence=0.9,
            )
        if result_type is ComparisonResult:
            candidates = json.loads(content.split("CANDIDATES:\n", 1)[1])
            return ComparisonResult(
                ordered_candidate_ids=sorted(
                    item["candidate_id"] for item in candidates
                )
            )
        if result_type is ReasoningResult:
            payload = json.loads(content)
            cid = payload["candidate"]["candidate_id"]
            return ReasoningResult(
                candidate_id=cid,
                reasoning=(
                    f"{cid} demonstrates production hybrid retrieval ownership "
                    "and relevant NDCG evaluation experience for this role."
                ),
            )
        raise AssertionError(result_type)


def test_full_pipeline_with_fake_model(tmp_path, candidate_factory):
    candidates = [candidate_factory(index) for index in range(1, 101)]
    output = tmp_path / "team.csv"
    artifacts = tmp_path / "artifacts"
    report = asyncio.run(
        run_full_pipeline(
            FakeClient(),
            candidates,
            anchors=[],
            integrity_by_id={
                candidate["candidate_id"]: CandidateIntegrity(
                    candidate_id=candidate["candidate_id"],
                    status=IntegrityStatus.CLEAN,
                    knowledge_base_schema_version=1,
                    knowledge_base_sha256="abc",
                )
                for candidate in candidates
            },
            output_csv=output,
            artifacts_dir=artifacts,
        )
    )
    assert report["initial_scoring"]["candidate_count"] == 100
    assert report["repeat_scoring"]["uncertain_count"] == 31
    assert (artifacts / "final_scores.jsonl").exists()
    assert (artifacts / "stage_2_individual_ranking.csv").exists()
    assert (artifacts / "stage_2_individual_assessments.jsonl").exists()
    assert (artifacts / "stage_3_comparative_ranking.csv").exists()
    assert (artifacts / "stage_3_comparative_assessments.jsonl").exists()
    assert validate_submission(
        output, {candidate["candidate_id"] for candidate in candidates}
    ) == []


def test_verified_failure_cannot_enter_final_output(tmp_path, candidate_factory):
    candidates = [candidate_factory(index) for index in range(1, 102)]
    failed_id = candidates[0]["candidate_id"]
    integrity = {
        candidate["candidate_id"]: CandidateIntegrity(
            candidate_id=candidate["candidate_id"],
            status=(
                IntegrityStatus.VERIFIED_FAILURE
                if candidate["candidate_id"] == failed_id
                else IntegrityStatus.CLEAN
            ),
            knowledge_base_schema_version=1,
            knowledge_base_sha256="abc",
        )
        for candidate in candidates
    }
    output = tmp_path / "team.csv"
    asyncio.run(
        run_full_pipeline(
            FakeClient(),
            candidates,
            anchors=[],
            integrity_by_id=integrity,
            output_csv=output,
            artifacts_dir=tmp_path / "artifacts",
        )
    )
    with output.open(newline="", encoding="utf-8") as handle:
        ids = {row["candidate_id"] for row in csv.DictReader(handle)}
    assert failed_id not in ids
