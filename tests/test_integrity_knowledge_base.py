import asyncio
import json

from contextmatch.integrity import load_integrity_report, scan_candidate_integrity
from contextmatch.models import (
    CandidateAssessment,
    CandidateIntegrity,
    DimensionScores,
    IntegrityFinding,
    IntegrityStatus,
)
from contextmatch.pipeline import score_candidates
from contextmatch.prompts import scoring_messages


def knowledge_base():
    return {
        "schema_version": 2,
        "as_of": "2026-06-23",
        "reference_date": "2026-06-20",
        "companies": {
            "ProductCo": {
                "founded_date": "2020-01-01",
                "precision": "year",
                "source": "https://example.com/official",
                "source_type": "official_company_page",
            }
        },
        "technologies": {
            "NewFramework": {
                "released_date": "2022-06-01",
                "precision": "month",
                "patterns": ["\\bNewFramework\\b"],
                "source": "https://example.com/framework",
                "source_type": "official_release",
            }
        },
        "certifications": {
            "New Certificate|Provider": {
                "available_date": "2023-01-01",
                "precision": "year",
                "source": "https://example.com/certification",
                "source_type": "official_provider_page",
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


def test_technology_chronology_is_ignored(candidate_factory):
    candidate = candidate_factory()
    candidate["skills"] = [
        {
            "name": "NewFramework",
            "proficiency": "advanced",
            "endorsements": 1,
            "duration_months": 60,
        }
    ]
    candidate["career_history"].append(
        {
            "company": "ProductCo",
            "title": "Search Engineer",
            "start_date": "2021-01-01",
            "end_date": "2021-12-01",
            "duration_months": 11,
            "is_current": False,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Built production systems using NewFramework.",
        }
    )
    result = scan_candidate_integrity(candidate, knowledge_base(), "hash")
    assert result.status == IntegrityStatus.CLEAN
    assert result.findings == []


def test_unknown_or_fictional_facts_do_not_penalize(candidate_factory):
    candidate = candidate_factory()
    kb = knowledge_base()
    del kb["companies"]["ProductCo"]
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


def test_loaded_integrity_report_uses_current_policy(tmp_path):
    path = tmp_path / "integrity_report.jsonl"
    stale = CandidateIntegrity(
        candidate_id="CAND_0000001",
        status=IntegrityStatus.VERIFIED_FAILURE,
        findings=[
            IntegrityFinding(
                rule="expert_skill_zero_usage",
                severity="verified_failure",
                message="Expert proficiency with zero usage duration: Python.",
            ),
            IntegrityFinding(
                rule="skill_duration_exceeds_technology_age",
                severity="suspicious",
                message="Ignored stale technology chronology finding.",
            ),
        ],
        knowledge_base_schema_version=2,
        knowledge_base_sha256="hash",
    )
    path.write_text(json.dumps(stale.model_dump(mode="json")) + "\n")
    loaded = load_integrity_report(path)["CAND_0000001"]
    assert loaded.status == IntegrityStatus.SUSPICIOUS
    assert [finding.rule for finding in loaded.findings] == [
        "expert_skill_zero_usage"
    ]
    assert loaded.findings[0].severity == "suspicious"


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
    failed = candidate.copy()
    failed["career_history"] = [role.copy() for role in candidate["career_history"]]
    failed["career_history"][0]["start_date"] = "2019-01-01"
    failed["career_history"][0]["duration_months"] = 89
    integrity = scan_candidate_integrity(failed, knowledge_base(), "hash")
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


def test_expert_zero_usage_scores_with_penalty(candidate_factory):
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
    assert client.calls == 1
    assert scores[0].rubric_score == 2
    assert scores[0].integrity_status == IntegrityStatus.SUSPICIOUS
    assert scores[0].integrity_findings[0].rule == "expert_skill_zero_usage"


def test_suspicious_findings_are_hidden_from_qwen(candidate_factory):
    candidate = candidate_factory()
    messages = scoring_messages(candidate)
    rendered = "\n".join(message["content"] for message in messages)
    assert "integrity_status" not in rendered
