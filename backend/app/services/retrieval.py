"""
Hybrid retrieval + cross-encoder re-rank.

  1. Vector search → candidate pool (user-scoped)
  2. Lexical fusion with vector scores
  3. Cross-encoder re-ranks the fused shortlist → final top_k
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from app.config import get_settings
from app.services.embedding_service import embed_texts
from app.services import vectorstore
from app.services.rerank_service import rerank

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    page: int
    doc_id: str
    score: float
    vector_score: float
    lexical_score: float
    rerank_score: Optional[float] = None
    ingested_at: Optional[str] = None


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


def _lexical_score(question: str, document: str) -> float:
    q = _tokenize(question)
    d = _tokenize(document)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve(
    question: str,
    user_id: str,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    settings = get_settings()
    final_k = top_k if top_k is not None else settings.top_k
    query_embedding = embed_texts([question])[0]

    raw = vectorstore.query_vectors(
        user_id=user_id,
        query_embedding=query_embedding,
        n_results=settings.candidate_pool,
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    if not documents:
        return []

    fused: List[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        vec = _distance_to_similarity(dist)
        lex = _lexical_score(question, doc)
        score = (
            settings.hybrid_vector_weight * vec
            + settings.hybrid_lexical_weight * lex
        )
        ingested = meta.get("created_at")
        fused.append(
            RetrievedChunk(
                text=doc,
                filename=meta["filename"],
                page=int(meta["page"]),
                doc_id=str(meta.get("doc_id") or ""),
                score=round(score, 4),
                vector_score=round(vec, 4),
                lexical_score=round(lex, 4),
                ingested_at=str(ingested) if ingested is not None else None,
            )
        )

    fused.sort(key=lambda c: c.score, reverse=True)
    return rerank(question, fused, final_k)
