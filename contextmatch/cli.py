from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from .calibration import (
    build_anchor_and_holdout,
    calibration_report,
    select_calibration_reviews,
)
from .data import (
    candidate_by_id,
    read_candidates,
    read_jsonl,
    validate_candidates,
    write_json,
    write_jsonl,
)
from .llm import VLLMClient
from .integrity import (
    load_integrity_report,
    load_knowledge_base,
    scan_candidates,
    write_integrity_outputs,
)
from .models import CalibrationReview, CandidateIntegrity, ScoredCandidate
from .output import validate_submission
from .pipeline import load_anchors, run_full_pipeline, score_candidates
from .stage_outputs import write_calibration_comparison

DEFAULT_SCHEMA = "India_runs_data_and_ai_challenge/candidate_schema.json"
DEFAULT_MODEL = "qwen3-14b-awq"
DEFAULT_KNOWLEDGE_BASE = "knowledge_base.json"


def _load_and_validate(args: argparse.Namespace) -> list[dict]:
    candidates = read_candidates(args.input)
    errors = validate_candidates(candidates, args.schema)
    if args.expected_count and len(candidates) != args.expected_count:
        errors.append(
            f"expected {args.expected_count} candidates, found {len(candidates)}"
        )
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        extra = f"\n... and {len(errors) - 50} more" if len(errors) > 50 else ""
        raise ValueError(f"candidate validation failed:\n{preview}{extra}")
    return candidates


def _client(args: argparse.Namespace) -> VLLMClient:
    return VLLMClient(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
    )


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=0,
        help="Optional exact count check. The default accepts any valid count.",
    )


def _add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)


def _load_scores(path: str) -> list[ScoredCandidate]:
    adapter = TypeAdapter(list[ScoredCandidate])
    return adapter.validate_python(read_jsonl(path))


def _load_integrity_context(
    knowledge_base_path: str,
    integrity_report_path: str,
    candidates: list[dict],
    *,
    allow_extra_records: bool = False,
) -> dict[str, CandidateIntegrity]:
    knowledge_base, knowledge_base_sha256 = load_knowledge_base(
        knowledge_base_path
    )
    report = load_integrity_report(integrity_report_path)
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    missing = candidate_ids - report.keys()
    if missing:
        raise ValueError(
            f"integrity report is missing {len(missing)} candidate IDs"
        )
    extras = report.keys() - candidate_ids
    if extras and not allow_extra_records:
        raise ValueError(
            f"integrity report contains {len(extras)} unexpected candidate IDs"
        )
    for record in report.values():
        if record.knowledge_base_sha256 != knowledge_base_sha256:
            raise ValueError(
                "integrity report was produced with a different knowledge base"
            )
        if (
            record.knowledge_base_schema_version
            != knowledge_base["metadata"]["schema_version"]
        ):
            raise ValueError("integrity report knowledge-base version mismatch")
    return {candidate_id: report[candidate_id] for candidate_id in candidate_ids}


async def _score_command(args: argparse.Namespace) -> None:
    candidates = _load_and_validate(args)
    integrity_by_id = _load_integrity_context(
        args.knowledge_base,
        args.integrity_report,
        candidates,
    )
    client = _client(args)
    try:
        scores, timing = await score_candidates(
            client,
            candidates,
            anchors=load_anchors(args.anchors),
            integrity_by_id=integrity_by_id,
        )
    finally:
        await client.close()
    write_jsonl(args.output, scores)
    if args.report:
        write_json(args.report, timing)
    print(json.dumps(timing, indent=2))


async def _evaluate_command(args: argparse.Namespace) -> None:
    expected = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
    candidates = [item["candidate"] for item in expected]
    integrity_by_id = _load_integrity_context(
        args.knowledge_base,
        args.integrity_report,
        candidates,
        allow_extra_records=True,
    )
    client = _client(args)
    try:
        predicted, timing = await score_candidates(
            client,
            candidates,
            anchors=load_anchors(args.anchors),
            integrity_by_id=integrity_by_id,
        )
    finally:
        await client.close()
    report = calibration_report(expected, predicted)
    report["timing"] = timing
    output_dir = Path(args.output).parent
    write_jsonl(output_dir / "stage_1_calibration_predictions.jsonl", predicted)
    write_calibration_comparison(
        output_dir / "stage_1_calibration_comparison.csv",
        expected,
        predicted,
    )
    report["stage_outputs"] = {
        "predictions": str(
            output_dir / "stage_1_calibration_predictions.jsonl"
        ),
        "comparison": str(
            output_dir / "stage_1_calibration_comparison.csv"
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


async def _run_command(args: argparse.Namespace) -> None:
    candidates = _load_and_validate(args)
    integrity_by_id = _load_integrity_context(
        args.knowledge_base, args.integrity_report, candidates
    )
    anchors = load_anchors(args.anchors)
    if len(anchors) != 8:
        raise ValueError(f"expected exactly 8 calibration anchors, found {len(anchors)}")
    client = _client(args)
    try:
        report = await run_full_pipeline(
            client,
            candidates,
            anchors=anchors,
            integrity_by_id=integrity_by_id,
            output_csv=args.output,
            artifacts_dir=args.artifacts_dir,
        )
    finally:
        await client.close()
    errors = validate_submission(
        args.output,
        {candidate["candidate_id"] for candidate in candidates},
        forbidden_candidate_ids={
            candidate_id
            for candidate_id, integrity in integrity_by_id.items()
            if integrity.status.value == "verified_failure"
        },
    )
    if errors:
        raise ValueError("generated submission failed validation:\n- " + "\n- ".join(errors))
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contextmatch")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-input")
    _add_input_arguments(validate)

    scan = commands.add_parser("scan-integrity")
    _add_input_arguments(scan)
    scan.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    scan.add_argument("--output-dir", required=True)

    score = commands.add_parser("score")
    _add_input_arguments(score)
    _add_client_arguments(score)
    score.add_argument("--anchors")
    score.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    score.add_argument("--integrity-report", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--report")

    select = commands.add_parser("select-calibration")
    _add_input_arguments(select)
    select.add_argument("--scores", required=True)
    select.add_argument("--size", type=int, default=40)
    select.add_argument("--output", required=True)

    build = commands.add_parser("build-calibration")
    build.add_argument("--reviews", required=True)
    build.add_argument("--anchor-count", type=int, default=8)
    build.add_argument("--anchors-output", required=True)
    build.add_argument("--holdout-output", required=True)

    evaluate = commands.add_parser("evaluate-calibration")
    _add_client_arguments(evaluate)
    evaluate.add_argument("--anchors", required=True)
    evaluate.add_argument("--holdout", required=True)
    evaluate.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    evaluate.add_argument("--integrity-report", required=True)
    evaluate.add_argument("--output", required=True)

    run = commands.add_parser("run")
    _add_input_arguments(run)
    _add_client_arguments(run)
    run.add_argument("--anchors", required=True)
    run.add_argument("--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE)
    run.add_argument("--integrity-report", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--artifacts-dir", default="runs/latest")

    validate_output = commands.add_parser("validate-output")
    _add_input_arguments(validate_output)
    validate_output.add_argument("--submission", required=True)
    validate_output.add_argument(
        "--knowledge-base", default=DEFAULT_KNOWLEDGE_BASE
    )
    validate_output.add_argument("--integrity-report", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "validate-input":
            candidates = _load_and_validate(args)
            print(f"Valid input: {len(candidates)} candidates")
        elif args.command == "scan-integrity":
            candidates = _load_and_validate(args)
            knowledge_base, knowledge_base_sha256 = load_knowledge_base(
                args.knowledge_base
            )
            records = scan_candidates(
                candidates, knowledge_base, knowledge_base_sha256
            )
            summary = write_integrity_outputs(args.output_dir, records)
            print(json.dumps(summary, indent=2))
        elif args.command == "score":
            asyncio.run(_score_command(args))
        elif args.command == "select-calibration":
            candidates = _load_and_validate(args)
            reviews = select_calibration_reviews(
                candidate_by_id(candidates),
                _load_scores(args.scores),
                args.size,
            )
            write_json(
                args.output,
                [review.model_dump(mode="json") for review in reviews],
            )
            print(f"Wrote {len(reviews)} pending reviews to {args.output}")
        elif args.command == "build-calibration":
            adapter = TypeAdapter(list[CalibrationReview])
            reviews = adapter.validate_python(
                json.loads(Path(args.reviews).read_text(encoding="utf-8"))
            )
            anchors, holdout = build_anchor_and_holdout(
                reviews, args.anchor_count
            )
            write_json(args.anchors_output, anchors)
            write_json(args.holdout_output, holdout)
            print(
                f"Wrote {len(anchors)} anchors and {len(holdout)} holdout records"
            )
        elif args.command == "evaluate-calibration":
            asyncio.run(_evaluate_command(args))
        elif args.command == "run":
            asyncio.run(_run_command(args))
        elif args.command == "validate-output":
            candidates = _load_and_validate(args)
            integrity_by_id = _load_integrity_context(
                args.knowledge_base, args.integrity_report, candidates
            )
            forbidden_ids = {
                candidate_id
                for candidate_id, integrity in integrity_by_id.items()
                if integrity.status.value == "verified_failure"
            }
            errors = validate_submission(
                args.submission,
                {candidate["candidate_id"] for candidate in candidates},
                forbidden_candidate_ids=forbidden_ids,
            )
            if errors:
                raise ValueError("submission validation failed:\n- " + "\n- ".join(errors))
            print("Submission is valid.")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
