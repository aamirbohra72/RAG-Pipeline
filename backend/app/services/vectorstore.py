"""
Vector store facade.

VECTOR_BACKEND=pgvector  → Neon Postgres + pgvector (cloud)
VECTOR_BACKEND=chroma    → local ChromaDB (./chroma_db)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import get_settings
from app.services.chunking_service import Chunk

logger = logging.getLogger(__name__)


def _backend():
    settings = get_settings()
    if settings.vector_backend == "pgvector":
        from app.services import pgvector_store as store

        return store
    from app.services import chroma_store as store

    return store


def init_vector_backend() -> None:
    settings = get_settings()
    logger.info("Vector backend: %s", settings.vector_backend)
    if settings.vector_backend == "pgvector":
        from app.services.pgvector_store import init_schema

        init_schema()


def add_document(user_id: str, filename: str, chunks: List[Chunk]) -> dict:
    return _backend().add_document(user_id, filename, chunks)


def query_vectors(
    user_id: str,
    query_embedding: List[float],
    n_results: int,
) -> Dict[str, Any]:
    return _backend().query_vectors(user_id, query_embedding, n_results)


def list_documents(user_id: str) -> List[dict]:
    return _backend().list_documents(user_id)


def delete_document(user_id: str, doc_id: str) -> bool:
    return _backend().delete_document(user_id, doc_id)


def chunk_count(user_id: str | None = None) -> int:
    return _backend().chunk_count(user_id)
