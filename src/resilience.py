"""
resilience.py
--------------
Stage 10 of the pipeline: combine market demand, skill transferability,
AI-complementarity, and skill coverage into a single, fully transparent
0-100 "Career Resilience Score".

    Resilience Score = Demand              x 0.30
                      + Transferability     x 0.25
                      + AI Complementarity  x 0.25
                      + Skill Coverage      x 0.20

All four inputs are themselves already normalized to a 0-100 scale by
their respective modules (market_analysis, transferability, ai_exposure,
skill_gap), so no additional scaling is needed here — the weighted sum is
already in [0, 100].

Weights are intentionally kept as named constants so they're easy to spot,
justify in a viva, and tune.
"""
from __future__ import annotations

from src.ai_exposure import average_ai_complementarity
from src.market_analysis import compute_demand_scores
from src.transferability import average_transferability

# Named, documented weights — must sum to 1.0
WEIGHT_DEMAND = 0.30
WEIGHT_TRANSFERABILITY = 0.25
WEIGHT_AI_COMPLEMENTARITY = 0.25
WEIGHT_SKILL_COVERAGE = 0.20

assert abs((WEIGHT_DEMAND + WEIGHT_TRANSFERABILITY
            + WEIGHT_AI_COMPLEMENTARITY + WEIGHT_SKILL_COVERAGE) - 1.0) < 1e-9


def _average_demand(skills: list[str], jobs_df=None) -> float:
    if not skills:
        return 0.0
    demand_scores = compute_demand_scores(jobs_df)
    values = [demand_scores.get(s.strip().lower(), 0.0) for s in skills]
    return round(sum(values) / len(values), 1) if values else 0.0


def compute_resilience_score(current_skills: list[str], skill_coverage_percentage: float,
                               jobs_df=None) -> dict:
    """Compute the Career Resilience Score for a candidate.

    Parameters
    ----------
    current_skills : list[str]
        The candidate's normalized current skills.
    skill_coverage_percentage : float
        Output of skill_gap.analyze_skill_gap*(...) — coverage of the
        candidate's target role(s), already 0-100.
    jobs_df : optional DataFrame
        Pass through a pre-loaded jobs DataFrame to avoid re-reading the CSV
        repeatedly when scoring many candidates/roles in a loop.

    Returns
    -------
    dict with the final score plus every component (for a transparent
    breakdown chart / viva explanation).
    """
    demand = _average_demand(current_skills, jobs_df)
    transferability = average_transferability(current_skills, jobs_df)
    ai_complementarity = average_ai_complementarity(current_skills)
    coverage = max(0.0, min(100.0, skill_coverage_percentage))

    score = (
        demand * WEIGHT_DEMAND
        + transferability * WEIGHT_TRANSFERABILITY
        + ai_complementarity * WEIGHT_AI_COMPLEMENTARITY
        + coverage * WEIGHT_SKILL_COVERAGE
    )
    score = round(max(0.0, min(100.0, score)), 1)

    return {
        "resilience_score": score,
        "components": {
            "demand": demand,
            "transferability": transferability,
            "ai_complementarity": ai_complementarity,
            "skill_coverage": coverage,
        },
        "weights": {
            "demand": WEIGHT_DEMAND,
            "transferability": WEIGHT_TRANSFERABILITY,
            "ai_complementarity": WEIGHT_AI_COMPLEMENTARITY,
            "skill_coverage": WEIGHT_SKILL_COVERAGE,
        },
        "formula": (
            "Resilience Score = Demand x 0.30 + Transferability x 0.25 "
            "+ AI Complementarity x 0.25 + Skill Coverage x 0.20"
        ),
    }


def resilience_band(score: float) -> str:
    """Human-readable band for the dashboard (purely presentational)."""
    if score >= 75:
        return "Strong"
    if score >= 55:
        return "Moderate"
    if score >= 35:
        return "Developing"
    return "At Risk"
