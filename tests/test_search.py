"""Unit tests for the embedding engine and semantic vector search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.embedding_engine import EmbeddingEngine, create_embeddings
from src.vector_search import build_faiss_index, search_similar_jobs, JobIndex


def test_embedding_engine_falls_back_gracefully():
    # In an offline test environment, sentence-transformers model weights
    # can't be downloaded, so this must not raise — it should silently
    # fall back to the TF-IDF backend.
    engine = EmbeddingEngine()
    assert engine.backend_name  # some backend was selected
    vectors, engine2 = create_embeddings(["python developer", "java developer", "chef"])
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0


def test_embeddings_are_l2_normalized():
    vectors, _ = create_embeddings(["python machine learning", "docker kubernetes aws"])
    norms = np.linalg.norm(vectors, axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-4 or n == 0.0


def test_build_and_search_index_returns_most_similar_first():
    corpus = [
        "python machine learning data science pandas",
        "java spring boot backend rest api",
        "python data science pandas numpy",
        "javascript react frontend css html",
    ]
    job_ids = [101, 102, 103, 104]
    vectors, engine = create_embeddings(corpus)
    index = build_faiss_index(vectors, job_ids)

    query_vec = engine.encode(["python pandas data science"])[0]
    results = search_similar_jobs(index, query_vec, top_k=2)

    assert len(results) == 2
    top_job_id, top_score = results[0]
    # The two python/data-science postings (101, 103) should outrank the
    # unrelated java/js postings.
    assert top_job_id in (101, 103)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)  # best-first ordering


def test_search_on_empty_index_returns_empty_list():
    index = JobIndex()
    results = index.search(np.zeros((1, 8), dtype="float32"), top_k=5)
    assert results == []
