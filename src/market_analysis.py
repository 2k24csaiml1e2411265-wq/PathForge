"""
market_analysis.py
-------------------
Stage 7 of the pipeline: mine data/jobs.csv to answer "what does the market
actually want?" — skill frequency, a normalized 0-100 demand score, and
per-role relevance, all computed transparently from the dataset (no
black-box model).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from src.skill_extraction import extract_skills_from_job_row
from src.utils import load_jobs_dataframe, get_logger

logger = get_logger("pathforge.market_analysis")


def compute_skill_frequency(jobs_df: pd.DataFrame | None = None) -> Counter:
    """Count how many job postings mention each (normalized) skill."""
    if jobs_df is None:
        jobs_df = load_jobs_dataframe()
    freq: Counter = Counter()
    for skills_field in jobs_df.get("skills", []):
        for skill in set(extract_skills_from_job_row(skills_field)):
            freq[skill] += 1
    return freq


def compute_demand_scores(jobs_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Normalize raw skill frequency into a 0-100 'demand score':

        demand_score(skill) = frequency(skill) / max(frequency) * 100

    A skill mentioned in every single posting scores 100; a skill mentioned
    in none of the postings simply won't appear in the result.
    """
    if jobs_df is None:
        jobs_df = load_jobs_dataframe()
    freq = compute_skill_frequency(jobs_df)
    if not freq:
        return {}
    max_freq = max(freq.values())
    return {skill: round((count / max_freq) * 100, 1) for skill, count in freq.items()}


def compute_role_relevance(jobs_df: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    """For each job_title, compute how important each skill is WITHIN that
    role — i.e. the fraction of postings for that role that mention the
    skill. Returns {job_title: {skill: importance_0_to_100}}.
    """
    if jobs_df is None:
        jobs_df = load_jobs_dataframe()

    role_skill_counts: dict[str, Counter] = defaultdict(Counter)
    role_posting_counts: Counter = Counter()

    for _, row in jobs_df.iterrows():
        title = row.get("job_title", "Unknown") or "Unknown"
        role_posting_counts[title] += 1
        for skill in set(extract_skills_from_job_row(row.get("skills", ""))):
            role_skill_counts[title][skill] += 1

    role_relevance: dict[str, dict[str, float]] = {}
    for title, skill_counts in role_skill_counts.items():
        n_postings = role_posting_counts[title] or 1
        role_relevance[title] = {
            skill: round((count / n_postings) * 100, 1)
            for skill, count in skill_counts.items()
        }
    return role_relevance


def top_demand_skills(jobs_df: pd.DataFrame | None = None, top_n: int = 15) -> list[tuple[str, float]]:
    """Convenience: the N highest-demand skills market-wide, for dashboard
    charts. Returns [(skill, demand_score), ...] sorted descending.
    """
    scores = compute_demand_scores(jobs_df)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def skill_importance_for_role(job_title: str, jobs_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Importance (0-100) of each skill for one specific role."""
    relevance = compute_role_relevance(jobs_df)
    return relevance.get(job_title, {})
