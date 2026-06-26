from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .data import candidate_by_id, write_json, write_jsonl
from .llm import VLLMClient, gather_with_timing
from .models import (
    CandidateAssessment,
    CandidateIntegrity,
    ComparisonResult,
    DimensionScores,
    IntegrityStatus,
    ReasoningResult,
    ScoredCandidate,
)
from .output import fallback_reasoning, reasoning_is_valid, write_submission
from .prompts import comparison_messages, reasoning_messages, scoring_messages
from .rerank import aggregate_comparisons, make_comparison_groups
from .rubric import (
    RUBRIC_VERSION,
    calculate_score,
    expert_zero_usage_penalty,
    merge_assessments,
)
from .stage_outputs import write_comparative_ranking, write_individual_ranking


def load_anchors(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("anchors file must contain a JSON array")
    return value


def _calculate_score_with_integrity(
    assessment: CandidateAssessment,
    integrity: CandidateIntegrity | None,
    integrity_issues: list[str] | None = None,
) -> tuple[float, int | None]:
    score, cap = calculate_score(assessment, integrity_issues)
    if integrity_issues:
        return score, cap
    penalty = expert_zero_usage_penalty(
        [finding.rule for finding in (integrity.findings if integrity else [])]
    )
    return max(score - penalty, 0.0), cap


async def score_candidates(
    client: VLLMClient,
    candidates: list[dict[str, Any]],
    *,
    anchors: list[dict[str, Any]] | None = None,
    integrity_by_id: dict[str, CandidateIntegrity] | None = None,
    thinking: bool = False,
    extra_instructions: dict[str, str] | None = None,
) -> tuple[list[ScoredCandidate], dict[str, float]]:
    async def score_one(candidate: dict[str, Any]) -> ScoredCandidate:
        cid = candidate["candidate_id"]
        integrity = (integrity_by_id or {}).get(cid)
        if integrity and integrity.status == IntegrityStatus.VERIFIED_FAILURE:
            assessment = CandidateAssessment(
                candidate_id=cid,
                dimensions=DimensionScores(
                    retrieval_ranking=0,
                    evaluation_experimentation=0,
                    production_ml_python=0,
                    product_shipping_outcomes=0,
                    ownership_seniority=0,
                    nlp_llm_secondary=0,
                    logistics_engagement=0,
                ),
                evidence=[
                    "Deterministic integrity verification found an impossible profile claim."
                ],
                concerns=[finding.message for finding in integrity.findings[:5]],
                conflicting_evidence=True,
                confidence=1.0,
            )
            return ScoredCandidate(
                candidate_id=cid,
                assessment=assessment,
                rubric_score=0.0,
                rubric_version=RUBRIC_VERSION,
                model_name="deterministic-integrity",
                applied_cap=0,
                integrity_issues=[
                    finding.message
                    for finding in integrity.findings
                    if finding.severity == "verified_failure"
                ],
                integrity_status=integrity.status,
                integrity_findings=integrity.findings,
                knowledge_base_schema_version=(
                    integrity.knowledge_base_schema_version
                ),
                knowledge_base_sha256=integrity.knowledge_base_sha256,
                final_score=0.0,
            )
        assessment = await client.complete_json(
            scoring_messages(
                candidate,
                anchors,
                (extra_instructions or {}).get(cid),
            ),
            CandidateAssessment,
            temperature=0 if not thinking else 0.1,
            max_tokens=800 if not thinking else 4000,
            thinking=thinking,
        )
        if assessment.candidate_id != cid:
            raise ValueError(
                f"model returned {assessment.candidate_id} while scoring {cid}"
            )
        score, cap = _calculate_score_with_integrity(assessment, integrity)
        return ScoredCandidate(
            candidate_id=cid,
            assessment=assessment,
            rubric_score=score,
            rubric_version=RUBRIC_VERSION,
            model_name=client.model,
            applied_cap=cap,
            integrity_issues=[],
            integrity_status=(
                integrity.status if integrity else IntegrityStatus.CLEAN
            ),
            integrity_findings=integrity.findings if integrity else [],
            knowledge_base_schema_version=(
                integrity.knowledge_base_schema_version if integrity else None
            ),
            knowledge_base_sha256=(
                integrity.knowledge_base_sha256 if integrity else None
            ),
            final_score=score,
        )

    results, elapsed = await gather_with_timing(
        [score_one(candidate) for candidate in candidates]
    )
    verified_failure_count = sum(
        item.integrity_status == IntegrityStatus.VERIFIED_FAILURE
        for item in results
    )
    return results, {
        "candidate_count": len(results),
        "model_candidate_count": len(results) - verified_failure_count,
        "skipped_verified_failures": verified_failure_count,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_candidate": round(elapsed / max(len(results), 1), 4),
    }


async def repeat_uncertain_scores(
    client: VLLMClient,
    candidates_by_id: dict[str, dict[str, Any]],
    scores: list[ScoredCandidate],
    anchors: list[dict[str, Any]],
    integrity_by_id: dict[str, CandidateIntegrity] | None = None,
) -> dict[str, Any]:
    initial_order = sorted(scores, key=lambda item: (-item.rubric_score, item.candidate_id))
    rank_by_id = {
        item.candidate_id: rank for rank, item in enumerate(initial_order, start=1)
    }
    uncertain = [
        item
        for item in scores
        if item.integrity_status != IntegrityStatus.VERIFIED_FAILURE
        and (
            70 <= rank_by_id[item.candidate_id] <= 180
        or item.assessment.confidence < 0.75
        or item.assessment.conflicting_evidence
        )
    ]
    repeated, timing = await score_candidates(
        client,
        [candidates_by_id[item.candidate_id] for item in uncertain],
        anchors=anchors,
        integrity_by_id=integrity_by_id,
    )
    repeated_by_id = {item.candidate_id: item for item in repeated}
    disagreements: list[ScoredCandidate] = []
    for item in uncertain:
        second = repeated_by_id[item.candidate_id].assessment
        first_flags = set(item.assessment.disqualifiers)
        second_flags = set(second.disqualifiers)
        item.repeated_assessment = merge_assessments(item.assessment, second)
        if first_flags != second_flags:
            disagreements.append(item)
        item.rubric_score, item.applied_cap = _calculate_score_with_integrity(
            item.repeated_assessment,
            (integrity_by_id or {}).get(item.candidate_id),
            item.integrity_issues,
        )
        item.final_score = item.rubric_score

    if disagreements:
        instructions = {}
        for item in disagreements:
            instructions[item.candidate_id] = (
                "Adjudicate a disagreement between two prior assessments. Return "
                "the most defensible assessment from the candidate facts.\n"
                f"ASSESSMENT_1={item.assessment.model_dump_json()}\n"
                f"ASSESSMENT_2={item.repeated_assessment.model_dump_json()}"
            )
        adjudicated, adjudication_timing = await score_candidates(
            client,
            [candidates_by_id[item.candidate_id] for item in disagreements],
            anchors=anchors,
            integrity_by_id=integrity_by_id,
            thinking=True,
            extra_instructions=instructions,
        )
        adjudicated_by_id = {item.candidate_id: item for item in adjudicated}
        for item in disagreements:
            final = adjudicated_by_id[item.candidate_id].assessment
            item.adjudicated_assessment = final
            item.rubric_score, item.applied_cap = _calculate_score_with_integrity(
                final,
                (integrity_by_id or {}).get(item.candidate_id),
                item.integrity_issues,
            )
            item.final_score = item.rubric_score
    else:
        adjudication_timing = {"candidate_count": 0, "elapsed_seconds": 0}
    return {
        "repeated": timing,
        "adjudicated": adjudication_timing,
        "uncertain_count": len(uncertain),
        "disagreement_count": len(disagreements),
    }


async def comparative_rerank(
    client: VLLMClient,
    candidates_by_id: dict[str, dict[str, Any]],
    scores: list[ScoredCandidate],
    *,
    top_n: int = 150,
    group_size: int = 10,
    rounds: int = 3,
) -> dict[str, Any]:
    leaders = sorted(
        scores, key=lambda item: (-item.rubric_score, item.candidate_id)
    )
    leaders = [
        item
        for item in leaders
        if item.integrity_status != IntegrityStatus.VERIFIED_FAILURE
    ][:top_n]
    scored_by_id = {item.candidate_id: item for item in leaders}
    groups = make_comparison_groups(
        [item.candidate_id for item in leaders],
        group_size=group_size,
        rounds=rounds,
    )

    async def compare(group: list[str]) -> list[str]:
        result = await client.complete_json(
            comparison_messages(
                [
                    (candidates_by_id[cid], scored_by_id[cid])
                    for cid in group
                ]
            ),
            ComparisonResult,
            temperature=0,
            max_tokens=500,
        )
        if len(result.ordered_candidate_ids) != len(group) or set(
            result.ordered_candidate_ids
        ) != set(group):
            missing = [c for c in group if c not in result.ordered_candidate_ids]
            if missing:
                return result.ordered_candidate_ids + missing
            raise ValueError("comparison response must contain each group ID once")
        return result.ordered_candidate_ids

    ordered_groups, elapsed = await gather_with_timing(
        [compare(group) for group in groups]
    )
    percentiles = aggregate_comparisons(list(scored_by_id), ordered_groups)
    leader_ids = set(scored_by_id)
    for item in scores:
        if item.candidate_id not in leader_ids:
            item.comparative_percentile = 0.0
            item.final_score = 0.80 * item.rubric_score
    for item in leaders:
        item.comparative_percentile = percentiles[item.candidate_id]
        item.final_score = 0.80 * item.rubric_score + 0.20 * item.comparative_percentile
    return {
        "candidate_count": len(leaders),
        "group_count": len(groups),
        "elapsed_seconds": round(elapsed, 3),
    }


async def generate_reasonings(
    client: VLLMClient,
    candidates_by_id: dict[str, dict[str, Any]],
    ranked: list[ScoredCandidate],
    *,
    limit: int = 100,
) -> tuple[dict[str, str], dict[str, Any]]:
    leaders = ranked[:limit]

    async def generate(item: ScoredCandidate) -> ReasoningResult:
        return await client.complete_json(
            reasoning_messages(candidates_by_id[item.candidate_id], item),
            ReasoningResult,
            temperature=0.1,
            max_tokens=140,
        )

    generated, elapsed = await gather_with_timing([generate(item) for item in leaders])
    reasonings: dict[str, str] = {}
    used: set[str] = set()
    fallback_count = 0
    for item, result in zip(leaders, generated):
        text = result.reasoning
        normalized = text.casefold()
        if (
            result.candidate_id != item.candidate_id
            or not reasoning_is_valid(text)
            or normalized in used
        ):
            text = fallback_reasoning(candidates_by_id[item.candidate_id], item)
            normalized = text.casefold()
            fallback_count += 1
        if normalized in used:
            profile = candidates_by_id[item.candidate_id]["profile"]
            text = (
                text.rstrip(".")
                + f" Current role: {profile.get('current_title')} at "
                f"{profile.get('current_company')}."
            )
            if len(text.split()) >= 50:
                text = " ".join(text.split()[:49]).rstrip(".,;:") + "."
            normalized = text.casefold()
        used.add(normalized)
        reasonings[item.candidate_id] = text
    return reasonings, {
        "candidate_count": len(leaders),
        "elapsed_seconds": round(elapsed, 3),
        "fallback_count": fallback_count,
    }


async def run_full_pipeline(
    client: VLLMClient,
    candidates: list[dict[str, Any]],
    *,
    anchors: list[dict[str, Any]],
    integrity_by_id: dict[str, CandidateIntegrity],
    output_csv: str | Path,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    candidates_by_id = candidate_by_id(candidates)
    started = time.perf_counter()

    scores, initial_timing = await score_candidates(
        client,
        candidates,
        anchors=anchors,
        integrity_by_id=integrity_by_id,
    )
    write_jsonl(artifacts / "initial_scores.jsonl", scores)
    initial_ranked = sorted(
        scores, key=lambda item: (-item.rubric_score, item.candidate_id)
    )
    initial_snapshot = {
        item.candidate_id: {"rank": rank, "score": item.rubric_score}
        for rank, item in enumerate(initial_ranked, start=1)
    }

    repeat_report = await repeat_uncertain_scores(
        client,
        candidates_by_id,
        scores,
        anchors,
        integrity_by_id,
    )
    write_jsonl(artifacts / "post_repeat_scores.jsonl", scores)
    individual_ranked = write_individual_ranking(
        artifacts / "stage_2_individual_ranking.csv",
        scores,
        initial_snapshot,
    )
    write_jsonl(
        artifacts / "stage_2_individual_assessments.jsonl",
        individual_ranked,
    )

    rerank_report = await comparative_rerank(client, candidates_by_id, scores)
    for item in scores:
        if item.final_score is not None:
            item.final_score = round(item.final_score, 6)
    ranked = sorted(
        scores,
        key=lambda item: (
            -(item.final_score if item.final_score is not None else item.rubric_score),
            item.candidate_id,
        ),
    )
    if any(
        item.integrity_status == IntegrityStatus.VERIFIED_FAILURE
        for item in ranked[:100]
    ):
        raise ValueError("verified integrity failure entered the provisional top 100")
    write_jsonl(artifacts / "final_scores.jsonl", ranked)
    write_comparative_ranking(
        artifacts / "stage_3_comparative_ranking.csv",
        ranked,
        individual_ranked,
    )
    write_jsonl(
        artifacts / "stage_3_comparative_assessments.jsonl",
        ranked,
    )

    reasonings, reasoning_report = await generate_reasonings(
        client, candidates_by_id, ranked
    )
    write_json(artifacts / "reasonings.json", reasonings)
    write_submission(output_csv, ranked, reasonings)

    report = {
        "model_name": client.model,
        "rubric_version": RUBRIC_VERSION,
        "integrity_status_counts": {
            status.value: sum(
                item.integrity_status == status for item in scores
            )
            for status in IntegrityStatus
        },
        "knowledge_base_schema_version": next(
            (
                item.knowledge_base_schema_version
                for item in scores
                if item.knowledge_base_schema_version is not None
            ),
            None,
        ),
        "knowledge_base_sha256": next(
            (
                item.knowledge_base_sha256
                for item in scores
                if item.knowledge_base_sha256 is not None
            ),
            None,
        ),
        "initial_scoring": initial_timing,
        "repeat_scoring": repeat_report,
        "comparative_rerank": rerank_report,
        "reasoning": reasoning_report,
        "stage_outputs": {
            "stage_2_individual_ranking": str(
                artifacts / "stage_2_individual_ranking.csv"
            ),
            "stage_2_individual_assessments": str(
                artifacts / "stage_2_individual_assessments.jsonl"
            ),
            "stage_3_comparative_ranking": str(
                artifacts / "stage_3_comparative_ranking.csv"
            ),
            "stage_3_comparative_assessments": str(
                artifacts / "stage_3_comparative_assessments.jsonl"
            ),
        },
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_csv": str(output_csv),
    }
    write_json(artifacts / "run_report.json", report)
    return report
