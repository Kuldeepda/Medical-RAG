"""Shared utilities: logging, similarity conversion, response helpers."""

import logging
import re
from typing import Any

from backend.config import UNKNOWN_RESPONSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medical_rag")


def distance_to_similarity(distance: float, space: str = "cosine") -> float:
    """
    Convert Chroma distance to similarity score in [0, 1].
    For cosine space with normalized embeddings: similarity = 1 - distance.
    """
    if space == "cosine":
        return max(0.0, min(1.0, 1.0 - float(distance)))
    # L2 fallback heuristic
    return max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))


def is_refusal_answer(text: str) -> bool:
    """Detect if the model returned the mandatory refusal phrase."""
    if not text:
        return True
    normalized = text.strip().lower()
    return UNKNOWN_RESPONSE.lower() in normalized or normalized == "i don't know"


def sanitize_question(question: str, max_length: int = 2000) -> str:
    """Basic input sanitization for user questions."""
    cleaned = re.sub(r"\s+", " ", question.strip())
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


# Common misspellings in clinical queries (word-boundary safe)
_MEDICAL_TYPO_FIXES: dict[str, str] = {
    "asthama": "asthma",
    "ashthma": "asthma",
    "diabeties": "diabetes",
    "diabetis": "diabetes",
    "hypertention": "hypertension",
    "pneumonea": "pneumonia",
    "febrile": "fever",
}


def normalize_medical_query(question: str) -> str:
    """Fix frequent typos so embedding search matches medical terms."""
    text = question.lower()
    for typo, fix in _MEDICAL_TYPO_FIXES.items():
        text = re.sub(rf"\b{re.escape(typo)}\b", fix, text)
    return text


def filter_relevant_chunks(
    chunks: list[dict[str, Any]], min_score: float, max_chunks: int = 2
) -> list[dict[str, Any]]:
    """
    Keep only strong matches and prefer chunks from the top document
    to avoid mixing unrelated PDFs in one answer.
    """
    if not chunks:
        return []

    filtered = [c for c in chunks if c.get("similarity", 0) >= min_score]
    if not filtered:
        return []

    top_source = filtered[0].get("metadata", {}).get("source")
    same_document = [
        c for c in filtered if c.get("metadata", {}).get("source") == top_source
    ]
    pool = same_document if same_document else filtered
    return pool[:max_chunks]


def build_extractive_answer(chunks: list[dict[str, Any]], max_chars: int = 1400) -> str:
    """
    Build an answer using only retrieved chunk text (no LLM).
    Safe for hallucination prevention — nothing is invented.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:2]:
        text = (chunk.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)

    if not parts:
        return UNKNOWN_RESPONSE

    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit(" ", 1)[0] + "…"

    return f"Based on the provided medical documents:\n\n{combined}"


def format_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the LLM prompt with source labels."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        parts.append(
            f"[Source {i}: {source}, Page {page}]\n{chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)
