# PathForge — Methodology

## Skill extraction
Dictionary-driven phrase matching (data/skills.json) using a spaCy
PhraseMatcher when available, with an automatic regex fallback. Fully
explainable: every detected skill traces back to an exact alias in the
dictionary.

## Skill Coverage
    Skill Coverage % = matched_required_skills / total_required_skills * 100

## Market Demand
    Demand Score(skill) = frequency(skill) / max_frequency * 100
(computed over all postings in data/jobs.csv)

## Transferability
    Transferability(skill) = distinct_roles_using_skill / total_distinct_roles * 100

## AI Exposure
Configurable heuristic (data/ai_exposure_rules.json) classifying skills as
AI-Substitutable / AI-Augmented / AI-Complementary. Presented as an
analytical indicator only, not a validated prediction.

## Career Resilience Score
    Resilience = Demand x 0.30 + Transferability x 0.25
               + AI Complementarity x 0.25 + Skill Coverage x 0.20

## Recommendation Priority Score
    Priority = Role Relevance x 0.30 + Demand x 0.25 + Gap x 0.20
             + Transferability x 0.15 + Resilience Value x 0.10
