from __future__ import annotations

from copy import deepcopy

import pytest


@pytest.fixture
def candidate_factory():
    def build(index: int = 1):
        cid = f"CAND_{index:07d}"
        return {
            "candidate_id": cid,
            "profile": {
                "anonymized_name": "Test Candidate",
                "headline": "Senior Search Engineer",
                "summary": "Built production search and ranking systems.",
                "location": "Noida, Uttar Pradesh",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "Senior Search Engineer",
                "current_company": "ProductCo",
                "current_company_size": "201-500",
                "current_industry": "Software",
            },
            "career_history": [
                {
                    "company": "ProductCo",
                    "title": "Senior Search Engineer",
                    "start_date": "2025-06-20",
                    "end_date": None,
                    "duration_months": 12,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "201-500",
                    "description": "Owned production hybrid retrieval and NDCG evaluation.",
                }
            ],
            "education": [],
            "skills": [
                {
                    "name": "Python",
                    "proficiency": "expert",
                    "endorsements": 10,
                    "duration_months": 60,
                }
            ],
            "certifications": [],
            "languages": [],
            "redrob_signals": {
                "profile_completeness_score": 90,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-06-19",
                "open_to_work_flag": True,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.9,
                "avg_response_time_hours": 2,
                "skill_assessment_scores": {"Python": 90},
                "connection_count": 50,
                "endorsements_received": 10,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30, "max": 40},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 70,
                "search_appearance_30d": 20,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 1,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

    return build
