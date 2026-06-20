from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Disqualifier(str, Enum):
    PURE_RESEARCH = "pure_research_no_production"
    RECENT_LLM_ONLY = "recent_llm_only"
    NO_RECENT_CODE = "no_recent_hands_on_code"
    SERVICES_ONLY = "services_only_no_product"
    DOMAIN_MISMATCH = "domain_mismatch_cv_speech_robotics"


class DimensionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_ranking: int = Field(ge=0, le=25)
    evaluation_experimentation: int = Field(ge=0, le=20)
    production_ml_python: int = Field(ge=0, le=15)
    product_shipping_outcomes: int = Field(ge=0, le=15)
    ownership_seniority: int = Field(ge=0, le=10)
    nlp_llm_secondary: int = Field(ge=0, le=5)
    logistics_engagement: int = Field(ge=0, le=10)

    @property
    def total(self) -> int:
        return sum(self.model_dump().values())


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^CAND_[0-9]{7}$")
    dimensions: DimensionScores
    evidence: list[str] = Field(min_length=1, max_length=5)
    concerns: list[str] = Field(default_factory=list, max_length=5)
    disqualifiers: list[Disqualifier] = Field(default_factory=list)
    conflicting_evidence: bool = False
    confidence: float = Field(ge=0, le=1)

    @field_validator("evidence", "concerns")
    @classmethod
    def clean_text_items(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value.strip()]
        if len(cleaned) != len(values):
            raise ValueError("evidence and concern entries must be non-empty")
        return cleaned


class ReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def normalize_reasoning(cls, value: str) -> str:
        return " ".join(value.split())


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ordered_candidate_ids: list[str] = Field(min_length=2)


class ScoredCandidate(BaseModel):
    candidate_id: str
    assessment: CandidateAssessment
    rubric_score: float
    rubric_version: str = "redrob-v1"
    model_name: str = ""
    applied_cap: int | None = None
    integrity_issues: list[str] = Field(default_factory=list)
    repeated_assessment: CandidateAssessment | None = None
    adjudicated_assessment: CandidateAssessment | None = None
    comparative_percentile: float | None = None
    final_score: float | None = None


class CalibrationReview(BaseModel):
    candidate_id: str
    candidate: dict[str, Any]
    draft_assessment: CandidateAssessment
    review_status: str = Field(pattern="^(pending|approved|corrected)$")
    corrected_assessment: CandidateAssessment | None = None
    reviewer_notes: str = ""
    stratum: str

    def effective_assessment(self) -> CandidateAssessment:
        if self.review_status == "corrected":
            if self.corrected_assessment is None:
                raise ValueError(
                    f"{self.candidate_id}: corrected review needs corrected_assessment"
                )
            return self.corrected_assessment
        if self.review_status == "approved":
            return self.draft_assessment
        raise ValueError(f"{self.candidate_id}: review is still pending")
