from __future__ import annotations

import json
from typing import Any

from .data import compact_candidate
from .models import CandidateAssessment, ScoredCandidate

SYSTEM_RUBRIC = """You are evaluating candidates for Redrob's Senior AI Engineer
founding-team role. Apply the rubric literally and consistently.

ROLE FACTS
- The target is roughly 5-9 years, but strong evidence can justify candidates
outside that band.
- Pune or Noida is preferred. Other major Indian cities and candidates willing
to relocate remain viable. No work-visa sponsorship is offered.
- Required strengths are production embeddings-based retrieval, vector or
hybrid search, strong Python, and rigorous ranking evaluation.
- The company wants a hands-on shipper who can own architecture, work with
product, mentor future hires, and improve recruiter engagement.
- Use 2026-06-20 as the reference date for recency and activity.

SCORING (100 points total)
1. retrieval_ranking, 0-25: demonstrated production ownership of search,
retrieval, recommendation, matching, embeddings, hybrid retrieval, vector
indexes, or learning-to-rank. Skills without career evidence earn little.
2. evaluation_experimentation, 0-20: NDCG/MRR/MAP/recall, relevance labels,
offline-online correlation, A/B testing, feedback loops, or rigorous ranking
evaluation actually used in work.
3. production_ml_python, 0-15: hands-on Python, deployment, serving, monitoring,
latency, drift, index refresh, data pipelines, and operational reliability.
4. product_shipping_outcomes, 0-15: shipped systems used by real users,
measurable outcomes, product-company work, and collaboration with product.
5. ownership_seniority, 0-10: technical judgment, architecture ownership,
mentoring, founding-team adaptability, and continued hands-on coding.
6. nlp_llm_secondary, 0-5: useful NLP, LLM, fine-tuning, MLOps, or distributed
systems evidence. Do not reward fashionable terminology by itself.
7. logistics_engagement, 0-10: India/location fit, relocation, notice period,
recent activity, open-to-work, response rate, and interview reliability.

DISQUALIFIER FLAGS
- pure_research_no_production: research-only career with no production system.
- recent_llm_only: AI work is mainly recent API/LangChain/RAG demos and lacks
substantial earlier production ML.
- no_recent_hands_on_code: senior candidate has not written production code
recently.
- services_only_no_product: entire career is consulting/services with no
credible product-company experience.
- domain_mismatch_cv_speech_robotics: career is primarily CV/speech/robotics
without meaningful NLP, information retrieval, search, or recommendation.

RULES
- Career history is primary evidence. Skills, headline, and summary only
corroborate it.
- Do not infer technologies, outcomes, employers, or experience not stated.
- Be strict: a plausible adjacent candidate is not a strong ranking engineer.
- Evidence and concerns must quote or closely paraphrase candidate facts.
- Return only JSON matching the supplied schema.
"""


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _anchor_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    compact = compact_candidate(candidate)
    compact["profile"]["summary"] = _truncate(
        compact["profile"].get("summary") or "", 250
    )
    compact["career_history"] = [
        {
            **role,
            "description": _truncate(role.get("description") or "", 250),
        }
        for role in compact["career_history"][:3]
    ]
    compact["skills"] = compact["skills"][:8]
    return compact


def scoring_messages(
    candidate: dict[str, Any],
    anchors: list[dict[str, Any]] | None = None,
    extra_instruction: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_RUBRIC}]
    if anchors:
        examples = [
            {
                "candidate": _anchor_candidate(item["candidate"]),
                "correct_assessment": item["assessment"],
            }
            for item in anchors
        ]
        messages.append(
            {
                "role": "system",
                "content": "CALIBRATION EXAMPLES:\n"
                + json.dumps(examples, ensure_ascii=False, separators=(",", ":")),
            }
        )
    instruction = (
        "Assess this candidate. candidate_id in the response must exactly match "
        "the input. Use 1-5 specific evidence items and zero or more honest "
        "concerns."
    )
    if extra_instruction:
        instruction += "\n" + extra_instruction
    messages.append(
        {
            "role": "user",
            "content": instruction
            + "\nCANDIDATE:\n"
            + json.dumps(
                compact_candidate(candidate),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    return messages


def comparison_view(
    candidate: dict[str, Any], scored: ScoredCandidate
) -> dict[str, Any]:
    profile = candidate["profile"]
    assessment = (
        scored.adjudicated_assessment
        or scored.repeated_assessment
        or scored.assessment
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "title": profile.get("current_title"),
        "years_of_experience": profile.get("years_of_experience"),
        "location": profile.get("location"),
        "career": [
            {
                "company": role.get("company"),
                "title": role.get("title"),
                "description": _truncate(role.get("description") or "", 250),
            }
            for role in candidate["career_history"][:4]
        ],
        "dimension_scores": assessment.dimensions.model_dump(),
        "rubric_score": scored.rubric_score,
        "evidence": assessment.evidence,
        "concerns": assessment.concerns,
        "disqualifiers": [item.value for item in assessment.disqualifiers],
    }


def comparison_messages(
    candidates: list[tuple[dict[str, Any], ScoredCandidate]]
) -> list[dict[str, str]]:
    payload = [comparison_view(candidate, scored) for candidate, scored in candidates]
    return [
        {"role": "system", "content": SYSTEM_RUBRIC},
        {
            "role": "user",
            "content": (
                "Order every candidate from strongest to weakest for this exact "
                "role. Use the rubric and career evidence, not presentation style. "
                "Return each supplied candidate_id exactly once and no other IDs.\n"
                "CANDIDATES:\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]


def reasoning_messages(
    candidate: dict[str, Any], scored: ScoredCandidate
) -> list[dict[str, str]]:
    assessment = (
        scored.adjudicated_assessment
        or scored.repeated_assessment
        or scored.assessment
    )
    payload = {
        "candidate": compact_candidate(candidate),
        "final_score": scored.final_score,
        "evidence": assessment.evidence,
        "concerns": assessment.concerns,
    }
    return [
        {
            "role": "system",
            "content": (
                "Write factual hiring-ranking reasoning for the supplied candidate. "
                "Use only supplied profile facts. Write 1-2 sentences and fewer "
                "than 50 words. Mention the strongest connection to the Senior AI "
                "Engineer ranking/search role and one material concern when present. "
                "Avoid generic praise. Return only the requested JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
