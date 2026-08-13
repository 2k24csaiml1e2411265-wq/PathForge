"""
skill_normalization.py
-----------------------
Maps skill "surface forms" (variants) to a single canonical name, using the
configurable dictionary in data/skills.json.

Example
-------
    "scikit learn"  -> "scikit-learn"
    "tensorflow2"    -> "tensorflow"
    "machine learning" -> "machine learning"   (already canonical)

The alias table is built once (module import time) and reused everywhere,
so normalization stays cheap and consistent across the whole pipeline.
"""
from __future__ import annotations

from functools import lru_cache

from src.utils import SKILLS_JSON_PATH, load_json, get_logger

logger = get_logger("pathforge.skill_normalization")


@lru_cache(maxsize=1)
def _load_skill_dictionary() -> dict:
    data = load_json(SKILLS_JSON_PATH, default={})
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def build_alias_map() -> dict[str, str]:
    """Build alias -> canonical lookup, e.g. {"sklearn": "scikit-learn", ...}.
    Canonical names also map to themselves so lookups are uniform.
    """
    skill_dict = _load_skill_dictionary()
    alias_map: dict[str, str] = {}
    for canonical, meta in skill_dict.items():
        canonical_norm = canonical.strip().lower()
        alias_map[canonical_norm] = canonical_norm
        for alias in meta.get("aliases", []):
            alias_map[alias.strip().lower()] = canonical_norm
    return alias_map


def normalize_skill(raw_skill: str) -> str:
    """Normalize a single skill string to its canonical form.
    Unknown skills are returned lowercased/stripped, unchanged otherwise —
    we don't want to silently drop skills the dictionary hasn't seen yet.
    """
    if not raw_skill:
        return ""
    cleaned = raw_skill.strip().lower()
    # collapse internal whitespace/punctuation variants like "node . js"
    cleaned = " ".join(cleaned.split())
    alias_map = build_alias_map()
    return alias_map.get(cleaned, cleaned)


def normalize_skill_list(raw_skills: list[str]) -> list[str]:
    """Normalize a list of skills and de-duplicate while preserving order."""
    seen = set()
    normalized = []
    for s in raw_skills:
        norm = normalize_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def get_skill_category(canonical_skill: str) -> str:
    """Return the category (e.g. 'programming_languages') for a canonical
    skill name, or 'other' if it's not in the dictionary."""
    skill_dict = _load_skill_dictionary()
    meta = skill_dict.get(canonical_skill.strip().lower())
    return meta["category"] if meta else "other"
