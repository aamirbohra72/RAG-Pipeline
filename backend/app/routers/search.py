"""Raw hybrid retrieval endpoint (no LLM generation)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import SearchRequest, SearchResponse, SearchResultItem
from app.services import retrieval, vectorstore
from app.services.auth_service import User, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


def _parse_date_after(value: str) -> datetime:
    try:
        if len(value) <= 10:
            parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(400, "date_after must be ISO-8601 (YYYY-MM-DD or full datetime)") from exc
    return parsed


def _matches_doc_type(filename: str, doc_type: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext == doc_type.lower().strip(".")


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    user: User = Depends(get_current_user),
):
    query = request.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    eligible_doc_ids: set[str] | None = None
    if request.date_after:
        cutoff = _parse_date_after(request.date_after)
        ingest_map = vectorstore.get_document_ingest_dates(user.id)
        eligible_doc_ids = {
            doc_id for doc_id, ingested_at in ingest_map.items() if ingested_at >= cutoff
        }
        if not eligible_doc_ids:
            return SearchResponse(results=[])

    try:
        chunks = retrieval.retrieve(query, user.id, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Search failed for user %s", user.id)
        raise HTTPException(500, f"Search failed: {exc}") from exc

    results: list[SearchResultItem] = []
    for chunk in chunks:
        if request.doc_type and not _matches_doc_type(chunk.filename, request.doc_type):
            continue
        if eligible_doc_ids is not None and chunk.doc_id not in eligible_doc_ids:
            continue
        results.append(
            SearchResultItem(
                content=chunk.text,
                source=chunk.filename,
                doc_id=chunk.doc_id,
                page=chunk.page,
                score=chunk.rerank_score if chunk.rerank_score is not None else chunk.score,
                rerank_score=chunk.rerank_score,
                vector_score=chunk.vector_score,
                lexical_score=chunk.lexical_score,
                citation_metadata={
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "doc_id": chunk.doc_id,
                },
            )
        )

    return SearchResponse(results=results[: request.top_k])
