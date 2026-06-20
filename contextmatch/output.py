from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .data import CANDIDATE_ID
from .models import ScoredCandidate

HEADER = ["candidate_id", "rank", "score", "reasoning"]


def reasoning_is_valid(reasoning: str) -> bool:
    words = reasoning.split()
    return 8 <= len(words) < 50


def write_submission(
    path: str | Path,
    ranked: list[ScoredCandidate],
    reasonings: dict[str, str],
    *,
    limit: int = 100,
) -> None:
    if len(ranked) < limit:
        raise ValueError(f"need at least {limit} ranked candidates")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for rank, item in enumerate(ranked[:limit], start=1):
            score = item.final_score
            if score is None or not math.isfinite(score):
                raise ValueError(f"{item.candidate_id}: invalid final score")
            reasoning = reasonings.get(item.candidate_id, "")
            if not reasoning_is_valid(reasoning):
                raise ValueError(
                    f"{item.candidate_id}: reasoning must contain 8-49 words"
                )
            writer.writerow([item.candidate_id, rank, f"{score:.6f}", reasoning])


def validate_submission(
    path: str | Path,
    valid_candidate_ids: set[str],
    *,
    expected_rows: int = 100,
) -> list[str]:
    errors: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return ["submission is empty"]
    if header != HEADER:
        errors.append(f"header must be exactly {','.join(HEADER)}")
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    parsed: list[tuple[int, float, str]] = []
    normalized_reasonings: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        cid = (row.get("candidate_id") or "").strip()
        if not CANDIDATE_ID.fullmatch(cid):
            errors.append(f"row {line_number}: invalid candidate_id")
        elif cid not in valid_candidate_ids:
            errors.append(f"row {line_number}: unknown candidate_id {cid}")
        elif cid in seen_ids:
            errors.append(f"row {line_number}: duplicate candidate_id {cid}")
        seen_ids.add(cid)
        try:
            rank = int(row.get("rank", ""))
            if rank in seen_ranks or not 1 <= rank <= expected_rows:
                raise ValueError
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"row {line_number}: invalid or duplicate rank")
            continue
        try:
            score = float(row.get("score", ""))
            if not math.isfinite(score):
                raise ValueError
        except ValueError:
            errors.append(f"row {line_number}: score must be finite")
            continue
        reasoning = " ".join((row.get("reasoning") or "").split())
        if not reasoning_is_valid(reasoning):
            errors.append(f"row {line_number}: reasoning must contain 8-49 words")
        normalized = reasoning.casefold()
        if normalized in normalized_reasonings:
            errors.append(f"row {line_number}: duplicate reasoning")
        normalized_reasonings.add(normalized)
        parsed.append((rank, score, cid))

    parsed.sort()
    for first, second in zip(parsed, parsed[1:]):
        if first[1] < second[1]:
            errors.append(
                f"scores increase from rank {first[0]} to rank {second[0]}"
            )
        if first[1] == second[1] and first[2] > second[2]:
            errors.append(
                f"score tie at ranks {first[0]}/{second[0]} violates ID ordering"
            )
    expected_ranks = set(range(1, expected_rows + 1))
    if seen_ranks != expected_ranks:
        errors.append("ranks must contain every integer from 1 through 100")
    return errors


def fallback_reasoning(candidate: dict[str, Any], scored: ScoredCandidate) -> str:
    assessment = (
        scored.adjudicated_assessment
        or scored.repeated_assessment
        or scored.assessment
    )
    evidence = assessment.evidence[0].rstrip(".")
    concern = assessment.concerns[0].rstrip(".") if assessment.concerns else ""
    text = evidence + "."
    if concern:
        text += " Concern: " + concern + "."
    words = text.split()
    if len(words) >= 50:
        text = " ".join(words[:49]).rstrip(".,;:") + "."
    if len(text.split()) < 8:
        profile = candidate["profile"]
        text = (
            f"{profile.get('current_title')} with "
            f"{profile.get('years_of_experience')} years of experience. "
            f"{text}"
        )
    return text
