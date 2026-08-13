"""
ai_exposure.py
---------------
Stage 9 of the pipeline: classify each skill into one of three AI-exposure
buckets using a configurable rules file (data/ai_exposure_rules.json).

IMPORTANT — this is presented throughout the app as an ANALYTICAL
HEURISTIC / INDICATOR, not a validated or guaranteed prediction of job
displacement. The categories are:

    AI-Substitutable  — current AI tools can largely already do this task
    AI-Augmented       — AI tools accelerate the task; a human still directs it
    AI-Complementary   — relies on judgement/relationships AI can't presently replace

Editing data/ai_exposure_rules.json changes the classification without
touching any code.
"""
from __future__ import annotations

from functools import lru_cache

from src.utils import AI_EXPOSURE_RULES_PATH, load_json, get_logger

logger = get_logger("pathforge.ai_exposure")


@lru_cache(maxsize=1)
def _load_rules() -> dict:
    return load_json(AI_EXPOSURE_RULES_PATH, default={
        "categories": {}, "skill_overrides": {},
        "default_category": "ai_augmented", "default_score": 55,
    })


def classify_skill(skill: str) -> dict:
    """Classify a single skill.

    Returns
    -------
    dict with: skill, category (key), label, score, explanation
    """
    rules = _load_rules()
    skill_norm = skill.strip().lower()

    category_key = rules.get("skill_overrides", {}).get(
        skill_norm, rules.get("default_category", "ai_augmented")
    )
    category_meta = rules.get("categories", {}).get(category_key, {})

    score_range = category_meta.get("score_range", [rules.get("default_score", 55)] * 2)
    score = round(sum(score_range) / 2, 1)

    return {
        "skill": skill_norm,
        "category": category_key,
        "label": category_meta.get("label", category_key),
        "score": score,
        "explanation": category_meta.get(
            "explanation",
            "Heuristic classification — no specific rule defined for this skill; "
            "using the default category as an analytical indicator only."
        ),
    }


def classify_skills(skills: list[str]) -> list[dict]:
    """Classify a list of skills."""
    return [classify_skill(s) for s in skills]


def average_ai_complementarity(skills: list[str]) -> float:
    """Average AI-exposure score across a skill list, used by resilience.py.
    Higher score = more AI-complementary (i.e. more resilient), by design
    of the score ranges in ai_exposure_rules.json.
    """
    if not skills:
        return 0.0
    classifications = classify_skills(skills)
    scores = [c["score"] for c in classifications]
    return round(sum(scores) / len(scores), 1) if scores else 0.0


def summarize_exposure(skills: list[str]) -> dict:
    """Count how many of the given skills fall into each exposure category
    — handy for a pie/bar chart in the Streamlit dashboard."""
    classifications = classify_skills(skills)
    counts: dict[str, int] = {}
    for c in classifications:
        counts[c["label"]] = counts.get(c["label"], 0) + 1
    return counts
