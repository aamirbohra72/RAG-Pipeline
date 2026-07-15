"""
Cross-encoder re-ranker.

Bi-encoders (mistral-embed) score query and doc independently → fast, approximate.
Cross-encoders read (query, doc) together → slower, much more precise ordering.

Senior pattern: retrieve many with hybrid search, re-rank a shortlist with a cross-encoder.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache
def _load_model():
    settings = get_settings()
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: %s (first load may download weights)", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


def rerank(question: str, chunks: List["RetrievedChunk"], top_k: int) -> List["RetrievedChunk"]:
    """
    Re-score candidates with a cross-encoder and return top_k.
    If disabled or the model fails, returns the hybrid-ranked input truncated to top_k.
    """
    settings = get_settings()
    if not chunks:
        return []

    if not settings.rerank_enabled:
        return chunks[:top_k]

    try:
        model = _load_model()
        pairs = [[question, c.text] for c in chunks]
        scores = model.predict(pairs)

        rescored = []
        for chunk, raw in zip(chunks, scores):
            score = float(raw)
            rescored.append(
                chunk.__class__(
                    text=chunk.text,
                    filename=chunk.filename,
                    page=chunk.page,
                    score=round(score, 4),
                    vector_score=chunk.vector_score,
                    lexical_score=chunk.lexical_score,
                    rerank_score=round(score, 4),
                )
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        logger.info(
            "Re-ranked %s candidates → top %s (best=%.4f)",
            len(rescored),
            top_k,
            rescored[0].score if rescored else 0.0,
        )
        return rescored[:top_k]
    except Exception:
        logger.exception("Cross-encoder re-rank failed; falling back to hybrid order")
        return chunks[:top_k]
