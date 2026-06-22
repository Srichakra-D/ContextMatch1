import asyncio
from copy import deepcopy

from contextmatch.integrity import scan_candidate_integrity
from contextmatch.models import (
    CandidateAssessment,
    DimensionScores,
    IntegrityStatus,
)
from contextmatch.pipeline import score_candidates
from contextmatch.prompts import scoring_messages


def knowledge_base():
    source = {
        "url": "https://example.com/official",
        "title": "Official history",
        "publisher": "Example",
        "source_type": "official_company_page",
        "retrieved_on": "2026-06-22",
    }
    return {
        "metadata": {"schema_version": 1},
        "companies": {
            "ProductCo": {
                "status": "verified",
                "date": "2020",
                "date_precision": "year",
                "sources": [source],
            }
        },
        "technologies": {
            "NewFramework": {
                "status": "verified",
                "date": "2022-06",
                "date_precision": "month",
                "aliases": [],
                "sources": [source],
            }
        },
        "certifications": {
            "New Certificate|Provider": {
                "status": "verified",
                "date": "2023",
                "date_precision": "year",
                "sources": [source],
            }
        },
    }


def test_employment_before_verified_company_is_failure(candidate_factory):
    candidate = candidate_factory()
    candidate["career_history"][0]["start_date"] = "2019-01-01"
    candidate["career_history"][0]["duration_months"] = 89
    result = scan_candidate_integrity(candidate, knowledge_base(), "hash")
    assert result.status == IntegrityStatus.VERIFIED_FAILURE
    assert any(
        item.rule == "employment_before_company_start"
        for item in result.findings
    )


def test_skill_age_is_suspicious_only(candidate_factory):
    candidate = candidate_factory()
    candidate["skills"] = [
        {
            "name": "NewFramework",
            "proficiency": "advanced",
            "endorsements": 1,
            "duration_months": 60,
        }
    ]
    result = scan_candidate_integrity(candidate, knowledge_base(), "hash")
    assert result.status == IntegrityStatus.SUSPICIOUS
    assert result.findings[0].rule == "skill_duration_exceeds_technology_age"


def test_unknown_or_fictional_facts_do_not_penalize(candidate_factory):
    candidate = candidate_factory()
    kb = knowledge_base()
    kb["companies"]["ProductCo"] = {
        "status": "fictional",
        "date": None,
        "date_precision": None,
        "sources": [],
    }
    result = scan_candidate_integrity(candidate, kb, "hash")
    assert result.status == IntegrityStatus.CLEAN


def test_certification_year_mismatch_is_suspicious(candidate_factory):
    candidate = candidate_factory()
    candidate["certifications"] = [
        {"name": "New Certificate", "issuer": "Provider", "year": 2022}
    ]
    result = scan_candidate_integrity(candidate, knowledge_base(), "hash")
    assert result.status == IntegrityStatus.SUSPICIOUS
    assert result.findings[0].rule == "certification_before_launch"


class CountingClient:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def complete_json(self, messages, result_type, **kwargs):
        self.calls += 1
        candidate_id = messages[-1]["content"].split('"candidate_id":"', 1)[1][
            :12
        ]
        return CandidateAssessment(
            candidate_id=candidate_id,
            dimensions=DimensionScores(
                retrieval_ranking=1,
                evaluation_experimentation=1,
                production_ml_python=1,
                product_shipping_outcomes=1,
                ownership_seniority=1,
                nlp_llm_secondary=1,
                logistics_engagement=1,
            ),
            evidence=["Valid profile evidence exists."],
            confidence=0.9,
        )


def test_verified_failure_skips_qwen(candidate_factory):
    candidate = candidate_factory()
    integrity = scan_candidate_integrity(
        {
            **candidate,
            "skills": [
                {
                    "name": "Python",
                    "proficiency": "expert",
                    "endorsements": 1,
                    "duration_months": 0,
                }
            ],
        },
        knowledge_base(),
        "hash",
    )
    client = CountingClient()
    scores, _ = asyncio.run(
        score_candidates(
            client,
            [candidate],
            integrity_by_id={candidate["candidate_id"]: integrity},
        )
    )
    assert client.calls == 0
    assert scores[0].rubric_score == 0
    assert scores[0].integrity_status == IntegrityStatus.VERIFIED_FAILURE


def test_suspicious_findings_are_hidden_from_qwen(candidate_factory):
    candidate = candidate_factory()
    messages = scoring_messages(candidate)
    rendered = "\n".join(message["content"] for message in messages)
    assert "skill_duration_exceeds_technology_age" not in rendered
    assert "integrity_status" not in rendered
