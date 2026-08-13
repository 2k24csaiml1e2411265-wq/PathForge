"""
vector_search.py
-----------------
Stage 4 of the pipeline: build a FAISS index over job-posting embeddings and
retrieve the Top-K jobs most semantically similar to a candidate's profile.

If the `faiss` package isn't installed, falls back to scikit-learn's
NearestNeighbors (cosine) — slower on huge datasets, but correct and
dependency-light, and completely fine at this project's scale
(hundreds-to-low-thousands of job postings).

Required functions (per spec):
    build_faiss_index()
    search_similar_jobs()
    save_index()
    load_index()
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils import MODELS_DIR, get_logger, ensure_dir

logger = get_logger("pathforge.vector_search")

FAISS_INDEX_PATH = MODELS_DIR / "jobs.index"
FAISS_META_PATH = MODELS_DIR / "jobs_meta.pkl"

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss is not installed — falling back to a scikit-learn "
                    "NearestNeighbors index (functionally equivalent at this scale).")


class JobIndex:
    """Thin wrapper that hides whether we're backed by FAISS or sklearn."""

    def __init__(self):
        self._faiss_index = None
        self._sk_index = None
        self._vectors: Optional[np.ndarray] = None
        self.job_ids: list[int] = []
        self.backend_name = "faiss" if _FAISS_AVAILABLE else "sklearn-NearestNeighbors"

    def build(self, vectors: np.ndarray, job_ids: list[int]) -> None:
        self.job_ids = list(job_ids)
        self._vectors = vectors.astype("float32")

        if _FAISS_AVAILABLE:
            dim = vectors.shape[1]
            # Vectors are pre-normalized (see embedding_engine.encode), so
            # inner product search == cosine similarity search.
            index = faiss.IndexFlatIP(dim)
            index.add(self._vectors)
            self._faiss_index = index
        else:
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = min(50, len(vectors)) or 1
            self._sk_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            self._sk_index.fit(self._vectors)

        logger.info("Built %s index over %d job postings.", self.backend_name, len(job_ids))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """Return [(job_id, similarity_score), ...] sorted best-first."""
        if not self.job_ids:
            return []

        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        top_k = min(top_k, len(self.job_ids))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector, top_k)
            results = [(self.job_ids[idx], float(score))
                       for idx, score in zip(indices[0], scores[0]) if idx != -1]
        else:
            distances, indices = self._sk_index.kneighbors(query_vector, n_neighbors=top_k)
            # sklearn cosine "distance" = 1 - cosine_similarity
            results = [(self.job_ids[idx], float(1.0 - dist))
                       for idx, dist in zip(indices[0], distances[0])]

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save(self, index_path: Path | str = FAISS_INDEX_PATH,
              meta_path: Path | str = FAISS_META_PATH) -> None:
        ensure_dir(Path(index_path).parent)
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({
                "job_ids": self.job_ids,
                "vectors": self._vectors,
                "backend_name": self.backend_name,
            }, f)

    def load(self, index_path: Path | str = FAISS_INDEX_PATH,
              meta_path: Path | str = FAISS_META_PATH) -> bool:
        meta_path = Path(meta_path)
        if not meta_path.exists():
            return False
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.job_ids = meta["job_ids"]
        self._vectors = meta["vectors"]

        if _FAISS_AVAILABLE and Path(index_path).exists():
            self._faiss_index = faiss.read_index(str(index_path))
        elif self._vectors is not None:
            # Rebuild whichever backend is active from the saved vectors
            self.build(self._vectors, self.job_ids)
        return True


# ---------------------------------------------------------------------------
# Functional wrappers (match the exact function names required by the spec)
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray, job_ids: list[int]) -> JobIndex:
    index = JobIndex()
    index.build(vectors, job_ids)
    return index


def search_similar_jobs(index: JobIndex, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    return index.search(query_vector, top_k=top_k)


def save_index(index: JobIndex, index_path: Path | str = FAISS_INDEX_PATH,
                meta_path: Path | str = FAISS_META_PATH) -> None:
    index.save(index_path, meta_path)


def load_index(index_path: Path | str = FAISS_INDEX_PATH,
                meta_path: Path | str = FAISS_META_PATH) -> Optional[JobIndex]:
    index = JobIndex()
    return index if index.load(index_path, meta_path) else None
