# PathForge — Architecture

## Pipeline
Resume (PDF/TXT) -> document_processing.py -> skill_extraction.py + skill_normalization.py
-> embedding_engine.py -> vector_search.py (FAISS) -> skill_gap.py -> market_analysis.py
-> transferability.py -> ai_exposure.py -> resilience.py -> recommendation_engine.py
-> gemini_service.py (optional LLM narrative layer) -> app.py (Streamlit dashboard)

## Key design principle
Every score (skill coverage, demand, transferability, AI exposure, resilience,
recommendation priority) is computed by deterministic, transparent code in `src/`.
Gemini (gemini_service.py) is ONLY used to turn already-computed numbers into
readable prose — it never calculates a score itself, and the app works fully
without it (template-based fallback).

## Offline-friendly fallbacks
- Embeddings: sentence-transformers (needs internet for model weights) ->
  falls back to scikit-learn TF-IDF if unavailable.
- Vector index: FAISS -> falls back to scikit-learn NearestNeighbors if
  faiss isn't installed.
- LLM narrative: Gemini API -> falls back to a template-based generator if
  no API key / no internet / call fails.

## Module map
See `src/` — one file per pipeline stage, each independently testable
(see `tests/`).
