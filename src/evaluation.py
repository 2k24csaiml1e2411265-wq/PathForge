"""
evaluation.py
-------------
Standalone evaluation script for PathForge. Run directly:

    python -m src.evaluation

Evaluates, using small, realistic, manually-labeled sample data (no
fabricated results):

  1. Skill extraction precision/recall against a hand-labeled resume snippet
  2. Semantic retrieval Top-K relevance (does the query's own role show up
     in the top results for an obviously-matching profile?)
  3. Skill coverage calculation sanity-check against a hand-computed example
  4. Recommendation consistency (same input -> same ranked output)

Prints a clear metrics report. This is intentionally simple/transparent —
per the project spec, formulas and evaluation should be explainable in a
college viva, not a black box.
"""
from __future__ import annotations

from src.skill_extraction import extract_skills
from src.skill_gap import analyze_skill_gap
from src.recommendation_engine import recommend_skills_to_learn
from src.utils import load_jobs_dataframe, get_logger
from src.embedding_engine import create_embeddings
from src.vector_search import build_faiss_index, search_similar_jobs

logger = get_logger("pathforge.evaluation")


# ---------------------------------------------------------------------------
# 1. Skill extraction precision / recall
# ---------------------------------------------------------------------------

LABELED_SAMPLE_TEXT = (
    "Final-year student skilled in Python, SQL, and Machine Learning. "
    "Built projects using TensorFlow and scikit-learn. Familiar with Docker "
    "and basic AWS deployment. Comfortable presenting work to non-technical "
    "stakeholders."
)
# Manually labeled ground truth for the snippet above (canonical forms)
LABELED_GROUND_TRUTH = {
    "python", "sql", "machine learning", "tensorflow", "scikit-learn",
    "docker", "aws", "presentation",
}


def evaluate_skill_extraction() -> dict:
    result = extract_skills(LABELED_SAMPLE_TEXT)
    predicted = set(result.get("_all", []))

    true_positives = predicted & LABELED_GROUND_TRUTH
    false_positives = predicted - LABELED_GROUND_TRUTH
    false_negatives = LABELED_GROUND_TRUTH - predicted

    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(LABELED_GROUND_TRUTH) if LABELED_GROUND_TRUTH else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
    }


# ---------------------------------------------------------------------------
# 2. Semantic retrieval Top-K relevance
# ---------------------------------------------------------------------------

def evaluate_retrieval(top_k: int = 5) -> dict:
    jobs_df = load_jobs_dataframe()
    if jobs_df.empty:
        return {"error": "jobs.csv is empty — run data/generate_sample_jobs.py first."}

    corpus = (jobs_df["job_title"] + " " + jobs_df["description"] + " " + jobs_df["skills"]).tolist()
    vectors, engine = create_embeddings(corpus)
    index = build_faiss_index(vectors, jobs_df["job_id"].tolist())

    # A query obviously written for a Data Analyst — check it retrieves
    # Data Analyst / adjacent-data-role postings in the top-K.
    query = "I work with SQL, Excel-like data analysis, pandas and statistics to find business insights."
    qvec = engine.encode([query])[0]
    results = search_similar_jobs(index, qvec, top_k=top_k)

    retrieved_titles = []
    hits = 0
    for job_id, score in results:
        row = jobs_df[jobs_df.job_id == job_id].iloc[0]
        retrieved_titles.append((row.job_title, round(score, 3)))
        if row.job_title in ("Data Analyst", "Data Scientist"):
            hits += 1

    top_k_relevance = round(hits / len(results), 3) if results else 0.0
    return {
        "query": query,
        "retrieved": retrieved_titles,
        "top_k_relevance": top_k_relevance,
        "backend": engine.backend_name,
    }


# ---------------------------------------------------------------------------
# 3. Skill coverage sanity check
# ---------------------------------------------------------------------------

def evaluate_skill_coverage() -> dict:
    current = ["python", "sql", "pandas"]
    required = ["python", "sql", "pandas", "machine learning", "docker"]
    # Hand-computed expectation: 3 matched / 5 required = 60.0%
    result = analyze_skill_gap(current, required)
    expected_coverage = 60.0
    passed = abs(result["skill_coverage_percentage"] - expected_coverage) < 1e-6
    return {
        "computed_coverage": result["skill_coverage_percentage"],
        "expected_coverage": expected_coverage,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# 4. Recommendation consistency
# ---------------------------------------------------------------------------

def evaluate_recommendation_consistency() -> dict:
    missing = ["docker", "aws", "kubernetes", "machine learning"]
    run_1 = recommend_skills_to_learn(missing, target_role="Cloud/DevOps Engineer")
    run_2 = recommend_skills_to_learn(missing, target_role="Cloud/DevOps Engineer")
    order_1 = [r["skill"] for r in run_1]
    order_2 = [r["skill"] for r in run_2]
    return {"consistent": order_1 == order_2, "order": order_1}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_all() -> dict:
    report = {
        "skill_extraction": evaluate_skill_extraction(),
        "retrieval": evaluate_retrieval(),
        "skill_coverage": evaluate_skill_coverage(),
        "recommendation_consistency": evaluate_recommendation_consistency(),
    }
    return report


def _print_report(report: dict) -> None:
    print("=" * 70)
    print("PathForge — Evaluation Report")
    print("=" * 70)

    se = report["skill_extraction"]
    print("\n[1] Skill Extraction (precision / recall / F1 on labeled sample)")
    print(f"    Precision: {se['precision']}   Recall: {se['recall']}   F1: {se['f1']}")
    print(f"    False positives: {se['false_positives']}")
    print(f"    False negatives: {se['false_negatives']}")

    rt = report["retrieval"]
    print("\n[2] Semantic Retrieval (Top-K relevance)")
    if "error" in rt:
        print(f"    {rt['error']}")
    else:
        print(f"    Backend: {rt['backend']}")
        print(f"    Top-K relevance: {rt['top_k_relevance']}")
        for title, score in rt["retrieved"]:
            print(f"      - {title}: {score}")

    sc = report["skill_coverage"]
    print("\n[3] Skill Coverage sanity check")
    print(f"    Computed: {sc['computed_coverage']}%  Expected: {sc['expected_coverage']}%  "
          f"Passed: {sc['passed']}")

    rc = report["recommendation_consistency"]
    print("\n[4] Recommendation Consistency (same input -> same ranking)")
    print(f"    Consistent: {rc['consistent']}")
    print(f"    Order: {rc['order']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    _print_report(run_all())
