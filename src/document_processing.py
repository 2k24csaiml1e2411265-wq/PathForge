"""
document_processing.py
-----------------------
Stage 1 of the pipeline: turn an uploaded resume (PDF or TXT) into clean,
normalized plain text ready for NLP skill extraction.

Functions
---------
extract_text_from_pdf(file)   -> str
extract_text_from_txt(file)   -> str
clean_text(text)              -> str
preprocess_resume(file, filename=None) -> str   (high-level entry point)
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

from pypdf import PdfReader

from src.utils import get_logger

logger = get_logger("pathforge.document_processing")

FileLike = Union[str, bytes, io.BytesIO, "io.BufferedReader"]


def extract_text_from_pdf(file: FileLike) -> str:
    """Extract raw text from a PDF file.

    `file` can be a filesystem path (str/Path), raw bytes, or a
    file-like object (e.g. the object Streamlit's file_uploader gives you).
    """
    try:
        if isinstance(file, (str, Path)):
            reader = PdfReader(str(file))
        elif isinstance(file, bytes):
            reader = PdfReader(io.BytesIO(file))
        else:
            # file-like object (has .read()) — e.g. Streamlit UploadedFile
            file.seek(0) if hasattr(file, "seek") else None
            reader = PdfReader(file)

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to extract text from a PDF page: %s", exc)
        return "\n".join(pages_text)
    except Exception as exc:
        logger.error("Could not read PDF: %s", exc)
        return ""


def extract_text_from_txt(file: FileLike) -> str:
    """Extract text from a plain-text resume."""
    try:
        if isinstance(file, (str, Path)):
            return Path(file).read_text(encoding="utf-8", errors="ignore")
        if isinstance(file, bytes):
            return file.decode("utf-8", errors="ignore")
        # file-like object
        file.seek(0) if hasattr(file, "seek") else None
        raw = file.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as exc:
        logger.error("Could not read TXT file: %s", exc)
        return ""


def clean_text(text: str) -> str:
    """Normalize whitespace, strip odd control characters, and lightly
    standardize punctuation so downstream regex/skill matching is reliable.
    This intentionally does NOT lowercase or remove stopwords here — that
    happens per-consumer (skill extraction lowercases internally) so the
    cleaned text is still readable for display in the UI.
    """
    if not text:
        return ""

    # Normalize line endings and remove non-printable control chars
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    # Collapse excessive whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix common PDF-extraction artifacts: hyphenated line breaks (e.g.
    # "Machine-\nLearning" -> "Machine-Learning")
    text = re.sub(r"-\n(?=[a-z])", "-", text)

    return text.strip()


def preprocess_resume(file: FileLike, filename: str | None = None) -> str:
    """High-level entry point used by the Streamlit app: figures out the
    file type from `filename` (or by sniffing PDF bytes) and returns clean
    text ready for skill extraction.
    """
    name = (filename or (file if isinstance(file, str) else "") or "").lower()

    is_pdf = name.endswith(".pdf")
    if not is_pdf and not name.endswith(".txt"):
        # Sniff: PDF files start with "%PDF"
        try:
            head = file.read(4) if hasattr(file, "read") else (
                Path(file).read_bytes()[:4] if isinstance(file, (str, Path)) else file[:4]
            )
            if hasattr(file, "seek"):
                file.seek(0)
            is_pdf = head == b"%PDF"
        except Exception:
            is_pdf = False

    raw_text = extract_text_from_pdf(file) if is_pdf else extract_text_from_txt(file)
    cleaned = clean_text(raw_text)

    if not cleaned:
        logger.warning("preprocess_resume produced empty text for file=%s", filename)

    return cleaned
