from __future__ import annotations

import random
from collections import defaultdict


def make_comparison_groups(
    candidate_ids: list[str],
    *,
    group_size: int = 10,
    rounds: int = 3,
    seed: int = 20260620,
) -> list[list[str]]:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    groups: list[list[str]] = []
    for round_index in range(rounds):
        shuffled = candidate_ids.copy()
        random.Random(seed + round_index).shuffle(shuffled)
        for start in range(0, len(shuffled), group_size):
            group = shuffled[start : start + group_size]
            if len(group) >= 2:
                groups.append(group)
    return groups


def aggregate_comparisons(
    candidate_ids: list[str], ordered_groups: list[list[str]]
) -> dict[str, float]:
    points: dict[str, list[float]] = defaultdict(list)
    expected = set(candidate_ids)
    for group in ordered_groups:
        if len(group) != len(set(group)) or not set(group) <= expected:
            raise ValueError("comparison result contains duplicate or unknown IDs")
        denominator = max(len(group) - 1, 1)
        for position, candidate_id in enumerate(group):
            points[candidate_id].append((len(group) - 1 - position) / denominator)
    missing = expected - points.keys()
    if missing:
        raise ValueError(f"candidates missing from comparisons: {sorted(missing)}")
    average = {
        candidate_id: sum(values) / len(values)
        for candidate_id, values in points.items()
    }
    ordered = sorted(average, key=lambda cid: (-average[cid], cid))
    denominator = max(len(ordered) - 1, 1)
    return {
        candidate_id: 100 * (len(ordered) - 1 - rank) / denominator
        for rank, candidate_id in enumerate(ordered)
    }
