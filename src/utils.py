"""
utils.py
--------
Small shared helpers used across the PathForge pipeline: OS-independent
path resolution, JSON/CSV loading with friendly error handling, and a
consistent logger.

Keeping these in one place avoids every module re-implementing
"find the project root" / "load this JSON safely" logic.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (OS-independent — always resolved relative to the project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

JOBS_CSV_PATH = DATA_DIR / "jobs.csv"
SKILLS_JSON_PATH = DATA_DIR / "skills.json"
AI_EXPOSURE_RULES_PATH = DATA_DIR / "ai_exposure_rules.json"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("pathforge.utils")


# ---------------------------------------------------------------------------
# Safe loaders
# ---------------------------------------------------------------------------

def load_json(path: Path | str, default: Any = None) -> Any:
    """Load a JSON file, returning `default` (and logging a warning)
    instead of raising if the file is missing or malformed."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("JSON file not found: %s — using default.", path)
        return default if default is not None else {}
    except json.JSONDecodeError as exc:
        logger.warning("JSON file at %s is malformed (%s) — using default.", path, exc)
        return default if default is not None else {}


def load_jobs_dataframe(path: Path | str = JOBS_CSV_PATH) -> pd.DataFrame:
    """Load the jobs dataset, gracefully handling a missing file by
    returning an empty (but correctly-shaped) DataFrame."""
    columns = ["job_id", "job_title", "company", "location",
               "description", "skills", "seniority", "industry"]
    path = Path(path)
    if not path.exists():
        logger.warning("Jobs dataset not found at %s — returning empty DataFrame. "
                        "Run data/generate_sample_jobs.py or "
                        "data/ingest_real_dataset.py first.", path)
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read jobs dataset (%s) — returning empty DataFrame.", exc)
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df["skills"] = df["skills"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it doesn't exist yet; return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_skill_list(raw: str) -> list[str]:
    """Split a semicolon/comma-separated skills string into a clean list."""
    if not raw:
        return []
    raw = raw.replace(";", ",")
    return [s.strip().lower() for s in raw.split(",") if s.strip()]
