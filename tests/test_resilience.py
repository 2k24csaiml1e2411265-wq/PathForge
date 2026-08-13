"""Unit tests for the Career Resilience Score calculation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.resilience import compute_resilience_score, resilience_band, WEIGHT_DEMAND, \
    WEIGHT_TRANSFERABILITY, WEIGHT_AI_COMPLEMENTARITY, WEIGHT_SKILL_COVERAGE


def _sample_jobs_df():
    return pd.DataFrame([
        {"job_id": 1, "job_title": "Data Scientist", "company": "A", "location": "X",
         "description": "d", "skills": "python; sql; machine learning", "seniority": "Mid", "industry": "Tech"},
        {"job_id": 2, "job_title": "ML Engineer", "company": "B", "location": "X",
         "description": "d", "skills": "python; docker; machine learning", "seniority": "Mid", "industry": "Tech"},
        {"job_id": 3, "job_title": "Backend Developer", "company": "C", "location": "X",
         "description": "d", "skills": "python; sql; docker", "seniority": "Mid", "industry": "Tech"},
    ])


def test_weights_sum_to_one():
    total = WEIGHT_DEMAND + WEIGHT_TRANSFERABILITY + WEIGHT_AI_COMPLEMENTARITY + WEIGHT_SKILL_COVERAGE
    assert abs(total - 1.0) < 1e-9


def test_resilience_score_is_bounded_0_to_100():
    jobs_df = _sample_jobs_df()
    result = compute_resilience_score(["python", "sql", "docker"], 80.0, jobs_df)
    assert 0.0 <= result["resilience_score"] <= 100.0


def test_resilience_score_zero_skills_is_low():
    jobs_df = _sample_jobs_df()
    result = compute_resilience_score([], 0.0, jobs_df)
    assert result["resilience_score"] == 0.0


def test_resilience_score_components_present():
    jobs_df = _sample_jobs_df()
    result = compute_resilience_score(["python"], 50.0, jobs_df)
    components = result["components"]
    assert set(components.keys()) == {"demand", "transferability", "ai_complementarity", "skill_coverage"}
    assert components["skill_coverage"] == 50.0


def test_higher_coverage_never_decreases_score_all_else_equal():
    jobs_df = _sample_jobs_df()
    low = compute_resilience_score(["python"], 20.0, jobs_df)
    high = compute_resilience_score(["python"], 90.0, jobs_df)
    assert high["resilience_score"] >= low["resilience_score"]


def test_resilience_band_thresholds():
    assert resilience_band(80) == "Strong"
    assert resilience_band(60) == "Moderate"
    assert resilience_band(40) == "Developing"
    assert resilience_band(10) == "At Risk"
