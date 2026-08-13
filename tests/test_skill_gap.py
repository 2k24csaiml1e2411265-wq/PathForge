"""Unit tests for the skill-gap engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skill_gap import analyze_skill_gap, analyze_skill_gap_across_jobs


def test_analyze_skill_gap_basic_coverage():
    current = ["python", "sql"]
    required = ["python", "sql", "docker", "aws"]
    result = analyze_skill_gap(current, required)
    assert result["matched_skills"] == ["python", "sql"]
    assert set(result["missing_skills"]) == {"docker", "aws"}
    assert result["skill_coverage_percentage"] == 50.0


def test_analyze_skill_gap_full_coverage():
    current = ["python", "sql", "docker"]
    required = ["python", "sql"]
    result = analyze_skill_gap(current, required)
    assert result["skill_coverage_percentage"] == 100.0
    assert result["missing_skills"] == []


def test_analyze_skill_gap_zero_coverage_no_required_skills():
    result = analyze_skill_gap(["python"], [])
    assert result["skill_coverage_percentage"] == 0.0
    assert result["matched_skills"] == []


def test_analyze_skill_gap_across_jobs_identifies_critical_gaps():
    current = ["python"]
    job_requirements = [
        ["python", "sql", "docker"],
        ["python", "sql", "aws"],
        ["python", "sql", "kubernetes"],
    ]
    result = analyze_skill_gap_across_jobs(current, job_requirements, critical_gap_threshold=0.9)
    # "sql" appears in all 3 target jobs and is missing -> critical gap
    assert "sql" in result["critical_gaps"]
    # docker/aws/kubernetes each appear in only 1/3 of jobs -> not critical at 0.9 threshold
    assert "docker" not in result["critical_gaps"]


def test_analyze_skill_gap_across_jobs_coverage_is_pooled():
    current = ["python", "sql"]
    job_requirements = [["python", "sql"], ["python", "docker"]]
    result = analyze_skill_gap_across_jobs(current, job_requirements)
    # pooled required skills: python, sql, docker (3 total); matched: python, sql (2)
    assert result["skill_coverage_percentage"] == round(2 / 3 * 100, 1)
