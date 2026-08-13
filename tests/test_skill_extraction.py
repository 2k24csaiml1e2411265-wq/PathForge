"""Unit tests for skill extraction and normalization."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skill_extraction import extract_skills
from src.skill_normalization import normalize_skill, normalize_skill_list


def test_extract_skills_finds_known_skills():
    text = "Proficient in Python, SQL, and Machine Learning. Used Docker for deployment."
    result = extract_skills(text)
    assert "python" in result["_all"]
    assert "sql" in result["_all"]
    assert "machine learning" in result["_all"]
    assert "docker" in result["_all"]


def test_extract_skills_categorizes_correctly():
    result = extract_skills("I know Python and AWS and MySQL.")
    assert "python" in result.get("programming_languages", [])
    assert "aws" in result.get("cloud", [])
    assert "mysql" in result.get("databases", [])


def test_extract_skills_empty_text_returns_empty():
    result = extract_skills("")
    assert result["_all"] == []


def test_extract_skills_no_false_positive_on_unrelated_text():
    result = extract_skills("The quick brown fox jumps over the lazy dog.")
    assert result["_all"] == []


def test_normalize_skill_variants():
    assert normalize_skill("scikit learn") == "scikit-learn"
    assert normalize_skill("tensorflow2") == "tensorflow"
    assert normalize_skill("machine learning") == "machine learning"
    assert normalize_skill("SkLearn") == "scikit-learn"


def test_normalize_skill_list_dedupes_and_preserves_order():
    result = normalize_skill_list(["python", "Python3", "sql", "python"])
    assert result == ["python", "sql"]


def test_extract_skills_word_boundary_avoids_substring_false_positive():
    # "r" is a skill (R language) — make sure it's not falsely triggered
    # inside unrelated words like "for" or "framework".
    result = extract_skills("This is a great framework for building things.")
    assert "r" not in result["_all"]
