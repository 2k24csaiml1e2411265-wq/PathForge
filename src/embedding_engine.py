"""
embedding_engine.py
--------------------
Stage 3 of the pipeline: turn text (resumes, job descriptions) into vector
embeddings for semantic search.

Offline-friendly design
------------------------
`sentence-transformers` downloads its model weights from the internet the
first time it runs. On a machine without internet access (or the first run
before the model is cached), that download fails. Rather than crash the
whole app, `EmbeddingEngine` automatically falls back to a TF-IDF vector
space (scikit-learn) — a classic, fully-offline, and perfectly explainable
technique for a college viva ("cosine similarity over TF-IDF vectors").

Whichever backend is active is always reported via `.backend_name`, and the
Streamlit app surfaces it to the user so nothing is silently misleading.

Required functions (per spec): create_embeddings(), build handled in
vector_search.py; save_index()/load_index() also live in vector_search.py
since they operate on the FAISS/NN index, not the raw embeddings.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils import MODELS_DIR, get_logger, ensure_dir

logger = get_logger("pathforge.embedding_engine")

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


class EmbeddingEngine:
    """Wraps either a SentenceTransformer model or a TF-IDF vectorizer
    behind one consistent `.encode(texts) -> np.ndarray` interface.
    """

    def __init__(self, prefer_transformer: bool = True):
        self.backend_name: str = "none"
        self._model = None            # SentenceTransformer instance
        self._vectorizer = None       # sklearn TfidfVectorizer instance
        self.dimension: Optional[int] = None

        if prefer_transformer:
            self._try_load_sentence_transformer()
        if self._model is None:
            self._init_tfidf_backend()

    # -- backend setup -----------------------------------------------------

    def _try_load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            self.dimension = self._model.get_embedding_dimension()
            self.backend_name = f"sentence-transformers ({SENTENCE_TRANSFORMER_MODEL})"
            logger.info("Loaded SentenceTransformer backend: %s", self.backend_name)
        except ImportError:
            logger.warning("sentence-transformers not installed — using TF-IDF fallback.")
            self._model = None
        except Exception as exc:
            # Covers: no internet access to download the model, HF hub errors, etc.
            logger.warning(
                "Could not load SentenceTransformer model '%s' (%s). "
                "This usually means no internet access to download model weights. "
                "Falling back to TF-IDF embeddings.", SENTENCE_TRANSFORMER_MODEL, exc
            )
            self._model = None

    def _init_tfidf_backend(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=4096, stop_words="english", ngram_range=(1, 2)
        )
        self.backend_name = "tfidf (offline fallback)"
        logger.info("Using TF-IDF fallback embedding backend.")

    # -- public API ----------------------------------------------------

    def fit(self, corpus: list[str]) -> None:
        """For the TF-IDF backend, the vectorizer must be *fit* on a
        representative corpus (typically all job descriptions) before use.
        No-op for the SentenceTransformer backend (nothing to fit).
        """
        if self._vectorizer is not None:
            self._vectorizer.fit(corpus)
            self.dimension = len(self._vectorizer.vocabulary_)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into a 2D float32 numpy array of shape
        (len(texts), dimension), L2-normalized so cosine similarity reduces
        to a simple dot product (required for FAISS inner-product search).
        """
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype="float32")

        if self._model is not None:
            vectors = self._model.encode(texts, show_progress_bar=False)
            vectors = np.asarray(vectors, dtype="float32")
        else:
            if self._vectorizer is None or not hasattr(self._vectorizer, "vocabulary_"):
                raise RuntimeError(
                    "TF-IDF vectorizer has not been fit yet. Call .fit(corpus) first "
                    "(create_embeddings() does this automatically for the job corpus)."
                )
            vectors = self._vectorizer.transform(texts).toarray().astype("float32")

        # L2-normalize rows so inner product == cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def save(self, path: Path | str = TFIDF_VECTORIZER_PATH) -> None:
        """Persist the TF-IDF vectorizer (SentenceTransformer models don't
        need saving — they're re-downloaded/loaded from cache by name)."""
        if self._vectorizer is not None:
            ensure_dir(Path(path).parent)
            with open(path, "wb") as f:
                pickle.dump(self._vectorizer, f)

    def load_vectorizer(self, path: Path | str = TFIDF_VECTORIZER_PATH) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self._vectorizer = pickle.load(f)
        self.dimension = len(self._vectorizer.vocabulary_)
        self.backend_name = "tfidf (offline fallback, loaded from disk)"
        return True


def create_embeddings(texts: list[str], engine: Optional[EmbeddingEngine] = None,
                       fit_if_tfidf: bool = True) -> tuple[np.ndarray, EmbeddingEngine]:
    """Convenience function matching the spec's required function name.
    Creates (or reuses) an EmbeddingEngine and returns (vectors, engine).
    """
    if engine is None:
        engine = EmbeddingEngine()
    if fit_if_tfidf and engine._vectorizer is not None and not hasattr(engine._vectorizer, "vocabulary_"):
        engine.fit(texts)
    vectors = engine.encode(texts)
    return vectors, engine
