"""
gemini_service.py
------------------
Stage 12 of the pipeline: the LLM explanation/contextualization layer.

IMPORTANT (per spec): Gemini is NEVER responsible for calculating scores.
Every number shown in this app (skill coverage, demand, transferability,
AI exposure, resilience score, priority scores) is computed deterministically
by the earlier pipeline stages (skill_gap.py, market_analysis.py,
transferability.py, ai_exposure.py, resilience.py, recommendation_engine.py).
Gemini only receives those already-computed numbers and writes a readable
narrative around them.

Graceful degradation: if GEMINI_API_KEY is not set, or the API call fails
for any reason (no internet, quota, bad key, ...), generate_career_summary()
and generate_roadmap() automatically fall back to a template-based
generator so the app keeps working end-to-end with zero external calls.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.utils import get_logger

logger = get_logger("pathforge.gemini_service")

load_dotenv()  # picks up .env if present; no-op otherwise

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _get_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None


def is_gemini_available() -> bool:
    """Whether we can even attempt a real Gemini call (key present + SDK
    importable). Does NOT guarantee the network call will succeed."""
    if not _get_api_key():
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def _call_gemini(prompt: str) -> str | None:
    """Low-level call. Returns the generated text, or None on any failure
    (missing key, missing SDK, network error, quota error, ...) so callers
    can fall back cleanly instead of crashing the app.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.info("GEMINI_API_KEY not set — using fallback template generator.")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
        return (response.text or "").strip() or None
    except ImportError:
        logger.warning("google-genai SDK not installed — using fallback template generator.")
        return None
    except Exception as exc:
        logger.warning("Gemini API call failed (%s) — using fallback template generator.", exc)
        return None


def _build_context_prompt(structured_data: dict) -> str:
    current = ", ".join(structured_data.get("current_skills", [])) or "none listed"
    missing = ", ".join(structured_data.get("missing_skills", [])[:10]) or "none"
    top_jobs = structured_data.get("top_jobs", [])
    top_jobs_text = "; ".join(
        f"{j.get('job_title')} at {j.get('company')} ({j.get('similarity', 0):.0%} match)"
        for j in top_jobs[:5]
    ) or "no strong matches found"
    resilience = structured_data.get("resilience_score", "N/A")
    coverage = structured_data.get("skill_coverage_percentage", "N/A")

    return (
        "You are a career advisor writing a short, encouraging, practical summary for a "
        "student's AI-generated career report. Use ONLY the data below — do not invent "
        "numbers or claims. Keep it concise (150-250 words), plain language, no markdown headers.\n\n"
        f"Current skills: {current}\n"
        f"Missing skills for target role: {missing}\n"
        f"Top matching jobs: {top_jobs_text}\n"
        f"Skill coverage: {coverage}%\n"
        f"Career Resilience Score: {resilience}/100\n\n"
        "Write: (1) a short career summary, (2) why the skill gaps matter, "
        "(3) the 3 highest priority skills to learn next and why."
    )


def generate_career_summary(structured_data: dict) -> dict:
    """Returns {"text": str, "source": "gemini" | "fallback"}."""
    prompt = _build_context_prompt(structured_data)
    text = _call_gemini(prompt)
    if text:
        return {"text": text, "source": "gemini"}
    return {"text": _fallback_career_summary(structured_data), "source": "fallback"}


def _fallback_career_summary(data: dict) -> str:
    coverage = data.get("skill_coverage_percentage", 0)
    resilience = data.get("resilience_score", 0)
    missing = data.get("missing_skills", [])[:3]
    missing_text = ", ".join(missing) if missing else "no major gaps"

    return (
        f"Based on the skills you've listed, you currently cover {coverage}% of what target "
        f"roles typically require, and your Career Resilience Score is {resilience}/100. "
        f"This score blends how in-demand your current skills are, how transferable they are "
        f"across related roles, how well they hold up against AI automation, and how closely "
        f"they already match your target role.\n\n"
        f"The most impactful next steps are closing the gap on: {missing_text}. Prioritizing "
        f"these will move both your skill coverage and your resilience score, since they tend "
        f"to be in high demand across the roles you're targeting. Build one small, complete "
        f"project per skill and add it to your portfolio — that demonstrates the skill far "
        f"more convincingly than a bullet point on a resume.\n\n"
        f"(This summary was generated by PathForge's built-in fallback generator because no "
        f"Gemini API key was configured — set GEMINI_API_KEY in your .env file for a richer, "
        f"LLM-generated narrative.)"
    )


def generate_roadmap(structured_data: dict) -> dict:
    """Generate the 5-phase personalized learning roadmap. Falls back to a
    deterministic template built from the ranked skills-to-learn list.
    """
    skills_to_learn = [s["skill"] for s in structured_data.get("skills_to_learn", [])]
    prompt = (
        "Based on this priority-ranked list of skills to learn: "
        f"{', '.join(skills_to_learn[:8]) or 'none identified'}, write a 5-phase learning "
        "roadmap titled exactly: Phase 1 - Foundation, Phase 2 - Skill Building, "
        "Phase 3 - Projects, Phase 4 - Interview Preparation, Phase 5 - Job Applications. "
        "For each phase give 2-4 short bullet points. Keep it concise and practical. "
        "Use plain text bullets (a dash), no markdown headers."
    )
    text = _call_gemini(prompt)
    if text:
        return {"text": text, "source": "gemini"}
    return {"text": _fallback_roadmap(skills_to_learn), "source": "fallback"}


def _fallback_roadmap(skills_to_learn: list[str]) -> str:
    top = skills_to_learn[:6] or ["core fundamentals for your target role"]
    foundation = top[:2] or ["revise core CS/math fundamentals"]
    building = top[2:4] or top[:2]
    return (
        "Phase 1 - Foundation\n"
        f"- Solidify fundamentals behind: {', '.join(foundation)}\n"
        "- Follow one well-reviewed course or official documentation per topic\n\n"
        "Phase 2 - Skill Building\n"
        f"- Hands-on practice with: {', '.join(building) if building else 'your priority skills'}\n"
        "- Reproduce small tutorials, then modify them to solve a problem you pick yourself\n\n"
        "Phase 3 - Projects\n"
        "- Build 1-2 portfolio projects that combine your top missing skills\n"
        "- Publish each with a clear README, demo, and short write-up\n\n"
        "Phase 4 - Interview Preparation\n"
        "- Practice explaining your projects and the trade-offs you made\n"
        "- Review core DSA / system-design / role-specific interview questions\n\n"
        "Phase 5 - Job Applications\n"
        "- Tailor your resume to the target role using the skill-gap analysis\n"
        "- Apply in batches and track responses to iterate on your pitch\n"
    )
