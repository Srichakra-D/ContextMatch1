from copy import deepcopy

from contextmatch.integrity import verified_integrity_issues


def test_clean_candidate_has_no_integrity_issues(candidate_factory):
    assert verified_integrity_issues(candidate_factory()) == []


def test_detects_duration_but_not_zero_usage_expert(candidate_factory):
    candidate = candidate_factory()
    candidate["career_history"][0]["duration_months"] = 100
    candidate["skills"][0]["duration_months"] = 0
    issues = verified_integrity_issues(candidate)
    assert any("duration contradiction" in issue for issue in issues)
    assert not any("zero usage" in issue for issue in issues)


def test_detects_current_role_mismatch(candidate_factory):
    candidate = candidate_factory()
    candidate["profile"]["current_company"] = "DifferentCo"
    assert any(
        "current role contradicts" in issue
        for issue in verified_integrity_issues(candidate)
    )
