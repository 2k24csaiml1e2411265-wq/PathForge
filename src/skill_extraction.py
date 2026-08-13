"""
skill_extraction.py
--------------------
Stage 2 of the pipeline: extract skills from free text (resume or job
description).

Design choice (per project spec — "keep scoring/extraction transparent,
avoid unsupported ML where a clear rule-based approach is better"):

  * The PRIMARY extraction mechanism is dictionary-driven phrase matching
    against data/skills.json (canonical names + aliases). This is fast,
    fully explainable in a viva ("here is exactly why 'tensorflow2' was
    detected"), and needs no internet access.
  * spaCy is used to (a) tokenize/lemmatize text for more robust matching
    and (b) build an efficient PhraseMatcher over the skill dictionary when
    the `en_core_web_sm` model is available. If spaCy or the model is not
    installed, extraction gracefully falls back to pure regex matching —
    the app keeps working either way.

Output of extract_skills(): a dict grouped by category, e.g.
    {
        "programming_languages": ["python", "sql"],
        "ml_ai": ["machine learning", "pandas"],
        ...
        "_all": ["python", "sql", "machine learning", "pandas"]   # flat, normalized
    }
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.skill_normalization import build_alias_map, get_skill_category, normalize_skill_list
from src.utils import get_logger

logger = get_logger("pathforge.skill_extraction")

_SPACY_MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def _load_spacy_pipeline():
    """Try to load spaCy + the small English model. Returns (nlp, matcher)
    or (None, None) if unavailable — callers must handle the None case.
    """
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
    except ImportError:
        logger.warning("spaCy is not installed — falling back to regex-only skill extraction.")
        return None, None

    try:
        nlp = spacy.load(_SPACY_MODEL_NAME)
    except OSError:
        logger.warning(
            "spaCy model '%s' is not downloaded (run: python -m spacy download %s) — "
            "falling back to regex-only skill extraction.", _SPACY_MODEL_NAME, _SPACY_MODEL_NAME
        )
        return None, None

    alias_map = build_alias_map()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(alias) for alias in alias_map.keys()]
    matcher.add("SKILLS", patterns)
    return nlp, matcher


def _regex_extract(text: str) -> set[str]:
    """Fallback extractor: word-boundary, case-insensitive search for every
    known alias directly in the raw text. Slower than PhraseMatcher on very
    long documents but requires no dependencies beyond the standard library.
    """
    alias_map = build_alias_map()
    text_lower = f" {text.lower()} "
    found = set()
    for alias, canonical in alias_map.items():
        # Escape regex special chars (aliases may contain '.', '+', '#')
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(canonical)
    return found


def _spacy_extract(text: str) -> set[str]:
    nlp, matcher = _load_spacy_pipeline()
    if nlp is None:
        return _regex_extract(text)

    alias_map = build_alias_map()
    found = set()
    # spaCy has input length limits for very long resumes; chunk defensively
    max_len = nlp.max_length - 1000 if nlp.max_length > 2000 else nlp.max_length
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [""]

    for chunk in chunks:
        doc = nlp(chunk)
        for _, start, end in matcher(doc):
            span_text = doc[start:end].text.lower()
            canonical = alias_map.get(span_text)
            if canonical:
                found.add(canonical)
    return found


def extract_skills(text: str, use_spacy: bool = True) -> dict:
    """Extract and categorize skills from free text.

    Parameters
    ----------
    text : str
        Resume text or job description text.
    use_spacy : bool
        If True (default), try the spaCy PhraseMatcher pipeline first and
        fall back to regex automatically. If False, always use regex.

    Returns
    -------
    dict with one key per skill category plus "_all" (flat, normalized,
    de-duplicated list of every skill found).
    """
    if not text or not text.strip():
        return {"_all": []}

    raw_found = _spacy_extract(text) if use_spacy else _regex_extract(text)
    normalized = normalize_skill_list(sorted(raw_found))

    categorized: dict[str, list[str]] = {}
    for skill in normalized:
        category = get_skill_category(skill)
        categorized.setdefault(category, []).append(skill)

    categorized["_all"] = normalized
    return categorized


def extract_skills_from_job_row(skills_field: str) -> list[str]:
    """Convenience helper: the jobs.csv 'skills' column is already a clean
    semicolon-separated list (ground truth from the dataset), so we just
    normalize it rather than re-running text extraction on it.
    """
    from src.utils import split_skill_list
    return normalize_skill_list(split_skill_list(skills_field))
