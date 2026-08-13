"""
transferability.py
-------------------
Stage 8 of the pipeline: estimate how "transferable" a skill is — i.e. how
useful it stays if the candidate pivots between related roles — using
plain job-role overlap/frequency counted from data/jobs.csv, per the spec's
instruction to avoid unsupported claims and instead ground this in the
dataset.

    transferability_score(skill) = (# distinct job_titles that use the skill
                                     / total distinct job_titles) * 100

A skill used across many different role titles (e.g. "python" appearing in
Data Scientist, ML Engineer, Data Analyst, and Backend Developer postings)
scores near 100. A skill confined to one niche role scores low.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.skill_extraction import extract_skills_from_job_row
from src.utils import load_jobs_dataframe, get_logger

logger = get_logger("pathforge.transferability")


def compute_skill_role_map(jobs_df: pd.DataFrame | None = None) -> dict[str, set[str]]:
    """Return {skill: {job_title_1, job_title_2, ...}} — the set of
    distinct roles that mention each skill."""
    if jobs_df is None:
        jobs_df = load_jobs_dataframe()

    skill_roles: dict[str, set[str]] = defaultdict(set)
    for _, row in jobs_df.iterrows():
        title = row.get("job_title", "Unknown") or "Unknown"
        for skill in set(extract_skills_from_job_row(row.get("skills", ""))):
            skill_roles[skill].add(title)
    return skill_roles


def compute_transferability_scores(jobs_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Return {skill: transferability_score_0_to_100} for every skill in
    the dataset."""
    if jobs_df is None:
        jobs_df = load_jobs_dataframe()

    skill_roles = compute_skill_role_map(jobs_df)
    total_roles = jobs_df["job_title"].nunique() or 1

    return {
        skill: round((len(roles) / total_roles) * 100, 1)
        for skill, roles in skill_roles.items()
    }


def transferability_for_skill(skill: str, jobs_df: pd.DataFrame | None = None) -> dict:
    """Detailed transferability info for a single skill: score + which
    roles it appears in (useful for an explainable UI tooltip)."""
    skill = skill.strip().lower()
    skill_roles = compute_skill_role_map(jobs_df)
    total_roles = (jobs_df if jobs_df is not None else load_jobs_dataframe())["job_title"].nunique() or 1

    roles = sorted(skill_roles.get(skill, set()))
    score = round((len(roles) / total_roles) * 100, 1) if roles else 0.0
    return {"skill": skill, "transferability_score": score, "roles": roles}


def average_transferability(skills: list[str], jobs_df: pd.DataFrame | None = None) -> float:
    """Average transferability score across a list of skills — used by
    resilience.py as one of the weighted signals."""
    if not skills:
        return 0.0
    scores = compute_transferability_scores(jobs_df)
    values = [scores.get(s.strip().lower(), 0.0) for s in skills]
    return round(sum(values) / len(values), 1) if values else 0.0
