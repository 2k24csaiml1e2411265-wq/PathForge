"""
skill_gap.py
------------
Stage 6 of the pipeline: compare a candidate's current skill set against the
skills required by their target job(s), using a transparent, explainable
formula rather than an opaque ML model.

    skill_coverage_percentage = matched_required_skills / total_required_skills * 100

"critical_gaps" are missing skills that appear across MOST of the retrieved
target jobs (i.e. skills you'd need almost regardless of which specific
posting you land) — computed from how often each missing skill shows up
across the provided list of required-skill sets.
"""
from __future__ import annotations

from collections import Counter

from src.skill_normalization import normalize_skill_list


def analyze_skill_gap(current_skills: list[str], required_skills: list[str]) -> dict:
    """Compare one current-skill set against one required-skill set.

    Returns
    -------
    dict with matched_skills, missing_skills, skill_coverage_percentage
    """
    current_norm = set(normalize_skill_list(current_skills))
    required_norm = normalize_skill_list(required_skills)  # keep order, dedup

    matched = [s for s in required_norm if s in current_norm]
    missing = [s for s in required_norm if s not in current_norm]

    total_required = len(required_norm)
    coverage = round((len(matched) / total_required) * 100, 1) if total_required else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage_percentage": coverage,
    }


def analyze_skill_gap_across_jobs(current_skills: list[str],
                                    job_required_skill_lists: list[list[str]],
                                    critical_gap_threshold: float = 0.5) -> dict:
    """Analyze skill gap against MULTIPLE target jobs at once (e.g. the
    Top-K retrieved jobs from vector_search) and surface "critical gaps" —
    missing skills that recur across a large fraction of those jobs, i.e.
    skills you'd need almost no matter which specific role you land.

    Parameters
    ----------
    current_skills : list[str]
    job_required_skill_lists : list of skill lists, one per target job
    critical_gap_threshold : float in (0, 1]
        A missing skill counts as "critical" if it appears in at least this
        fraction of the target jobs.

    Returns
    -------
    dict with matched_skills, missing_skills, critical_gaps,
    skill_coverage_percentage (all pooled across the target jobs), plus
    per_job breakdown.
    """
    current_norm = set(normalize_skill_list(current_skills))

    # Pool every required skill across all target jobs (de-duplicated,
    # frequency-tracked) so we can compute both overall coverage and which
    # missing skills are "critical" (needed by most target jobs).
    skill_frequency: Counter = Counter()
    n_jobs = len(job_required_skill_lists) or 1
    per_job = []

    for required in job_required_skill_lists:
        result = analyze_skill_gap(current_skills, required)
        per_job.append(result)
        for skill in normalize_skill_list(required):
            skill_frequency[skill] += 1

    pooled_required = list(skill_frequency.keys())
    matched = [s for s in pooled_required if s in current_norm]
    missing = [s for s in pooled_required if s not in current_norm]

    critical_gaps = [
        s for s in missing
        if (skill_frequency[s] / n_jobs) >= critical_gap_threshold
    ]
    # Sort critical gaps by how frequently they're demanded (most first)
    critical_gaps.sort(key=lambda s: skill_frequency[s], reverse=True)

    coverage = round((len(matched) / len(pooled_required)) * 100, 1) if pooled_required else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "critical_gaps": critical_gaps,
        "skill_coverage_percentage": coverage,
        "skill_demand_frequency": dict(skill_frequency),
        "per_job": per_job,
    }
