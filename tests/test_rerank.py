from contextmatch.rerank import aggregate_comparisons, make_comparison_groups


def test_comparison_groups_cover_every_candidate_each_round():
    ids = [f"CAND_{index:07d}" for index in range(1, 21)]
    groups = make_comparison_groups(ids, group_size=5, rounds=3)
    appearances = {cid: 0 for cid in ids}
    for group in groups:
        for cid in group:
            appearances[cid] += 1
    assert set(appearances.values()) == {3}


def test_aggregate_comparisons_returns_percentiles():
    ids = ["A", "B", "C"]
    values = aggregate_comparisons(ids, [["A", "B", "C"], ["A", "C", "B"]])
    assert values["A"] == 100
    assert values["B"] < 100
    assert set(values) == set(ids)


def test_aggregate_rejects_unknown_id():
    try:
        aggregate_comparisons(["A", "B"], [["A", "X"]])
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")
