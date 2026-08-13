"""
recommendation_engine.py
-------------------------
Stage 11 of the pipeline: turn the structured analysis (skill gap, market
demand, transferability, resilience) into ranked, explainable
recommendations — which skills to learn, which to strengthen, which
projects to build, and which career directions make sense.

    Priority Score = role_relevance x 0.30
                    + demand         x 0.25
                    + skill_gap      x 0.20   (weight of a *missing* skill)
                    + transferability x 0.15
                    + resilience_value x 0.10

This module produces the STRUCTURED recommendations. gemini_service.py
optionally turns these into natural-language explanations/roadmap text —
but the ranking and the underlying numbers are computed here, deterministically.
"""
from __future__ import annotations

from src.ai_exposure import classify_skill
from src.market_analysis import compute_demand_scores, skill_importance_for_role
from src.transferability import compute_transferability_scores

# Named priority-score weights (sum to 1.0)
W_ROLE_RELEVANCE = 0.30
W_DEMAND = 0.25
W_GAP = 0.20
W_TRANSFERABILITY = 0.15
W_RESILIENCE_VALUE = 0.10

# A small, illustrative project-suggestion bank keyed by skill. This is
# deliberately simple/transparent rather than LLM-generated, so the app
# always has *something* sensible to suggest even with no Gemini key.
PROJECT_SUGGESTIONS: dict[str, list[str]] = {
    "machine learning": ["Build an end-to-end ML pipeline (data -> model -> API) on a Kaggle dataset"],
    "deep learning": ["Train and deploy a CNN or transformer model for an image/text task"],
    "natural language processing": ["Build a resume-vs-job-description skill matcher (like this project!)"],
    "computer vision": ["Build a real-time object-detection demo with OpenCV + a pretrained model"],
    "sql": ["Design and query a normalized database for a small e-commerce app"],
    "docker": ["Containerize an existing project and write a docker-compose setup"],
    "aws": ["Deploy a small web app on AWS (EC2/Lambda) with a CI/CD pipeline"],
    "react": ["Build a full-stack dashboard with a React frontend and a REST API backend"],
    "faiss": ["Build a semantic search engine over a document collection"],
    "llm": ["Build a small RAG (retrieval-augmented generation) chatbot over your own notes"],
    "kubernetes": ["Deploy a multi-service app on a local Kubernetes cluster (minikube)"],
}
DEFAULT_PROJECT_SUGGESTION = "Build a small, complete project that uses {skill} end-to-end and document it publicly (GitHub README + demo)."


def _priority_score(skill: str, role_relevance: float, demand: float,
                      is_gap: bool, transferability: float, resilience_value: float) -> float:
    gap_component = 100.0 if is_gap else 0.0
    score = (
        role_relevance * W_ROLE_RELEVANCE
        + demand * W_DEMAND
        + gap_component * W_GAP
        + transferability * W_TRANSFERABILITY
        + resilience_value * W_RESILIENCE_VALUE
    )
    return round(score, 1)


def recommend_skills_to_learn(missing_skills: list[str], target_role: str | None = None,
                                jobs_df=None, top_n: int = 8) -> list[dict]:
    """Rank missing skills by priority score — what to learn first."""
    demand_scores = compute_demand_scores(jobs_df)
    transferability_scores = compute_transferability_scores(jobs_df)
    role_relevance = skill_importance_for_role(target_role, jobs_df) if target_role else {}

    ranked = []
    for skill in missing_skills:
        demand = demand_scores.get(skill, 0.0)
        transferability = transferability_scores.get(skill, 0.0)
        relevance = role_relevance.get(skill, demand)  # fall back to overall demand
        exposure = classify_skill(skill)
        priority = _priority_score(
            skill, role_relevance=relevance, demand=demand,
            is_gap=True, transferability=transferability,
            resilience_value=exposure["score"],
        )
        ranked.append({
            "skill": skill,
            "priority_score": priority,
            "demand": demand,
            "role_relevance": relevance,
            "transferability": transferability,
            "ai_exposure": exposure["label"],
            "suggested_project": (
                PROJECT_SUGGESTIONS[skill][0] if skill in PROJECT_SUGGESTIONS
                else DEFAULT_PROJECT_SUGGESTION.format(skill=skill)
            ),
        })

    ranked.sort(key=lambda r: r["priority_score"], reverse=True)
    return ranked[:top_n]


def recommend_skills_to_strengthen(matched_skills: list[str], target_role: str | None = None,
                                     jobs_df=None, top_n: int = 5) -> list[dict]:
    """Among skills the candidate ALREADY has, highlight the highest-value
    ones to deepen further (high demand + high transferability skills are
    worth specializing in rather than leaving at a basic level)."""
    demand_scores = compute_demand_scores(jobs_df)
    transferability_scores = compute_transferability_scores(jobs_df)
    role_relevance = skill_importance_for_role(target_role, jobs_df) if target_role else {}

    ranked = []
    for skill in matched_skills:
        demand = demand_scores.get(skill, 0.0)
        transferability = transferability_scores.get(skill, 0.0)
        relevance = role_relevance.get(skill, demand)
        combined = round((demand + transferability + relevance) / 3, 1)
        ranked.append({
            "skill": skill, "demand": demand, "transferability": transferability,
            "role_relevance": relevance, "strengthen_value": combined,
        })

    ranked.sort(key=lambda r: r["strengthen_value"], reverse=True)
    return ranked[:top_n]


def recommend_career_directions(current_skills: list[str], transferability_scores=None,
                                  jobs_df=None, top_n: int = 3) -> list[str]:
    """Very lightweight heuristic: suggest role families where the
    candidate's current skills have the highest average transferability —
    i.e. roles they're already well-positioned to pivot toward."""
    from src.market_analysis import compute_role_relevance
    role_relevance = compute_role_relevance(jobs_df)

    role_fit: dict[str, float] = {}
    for role, skill_map in role_relevance.items():
        overlap = [skill_map.get(s, 0.0) for s in current_skills if s in skill_map]
        if overlap:
            role_fit[role] = round(sum(overlap) / len(skill_map), 1) if skill_map else 0.0

    ranked_roles = sorted(role_fit.items(), key=lambda kv: kv[1], reverse=True)
    return [role for role, _ in ranked_roles[:top_n]]


def build_recommendations(skill_gap_result: dict, target_role: str | None = None,
                            jobs_df=None) -> dict:
    """One-call convenience wrapper the Streamlit app / evaluation script
    can use: takes the output of skill_gap.analyze_skill_gap*(...) and
    returns the full recommendation bundle."""
    missing = skill_gap_result.get("missing_skills", []) or skill_gap_result.get("critical_gaps", [])
    matched = skill_gap_result.get("matched_skills", [])

    return {
        "skills_to_learn": recommend_skills_to_learn(missing, target_role, jobs_df),
        "skills_to_strengthen": recommend_skills_to_strengthen(matched, target_role, jobs_df),
        "career_directions": recommend_career_directions(matched + missing, jobs_df=jobs_df),
    }
