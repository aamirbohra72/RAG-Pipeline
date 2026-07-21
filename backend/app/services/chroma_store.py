"""Local ChromaDB persistence (optional when VECTOR_BACKEND=chroma)."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

from app.dependencies import get_chroma_collection
from app.services.chunking_service import Chunk
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)


def add_document(user_id: str, filename: str, chunks: List[Chunk]) -> dict:
    if not chunks:
        raise ValueError(f"No extractable text chunks for {filename}")

    collection = get_chroma_collection()
    doc_id = str(uuid.uuid4())
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "page": c.page,
            "doc_type": c.doc_type,
            "section": c.section,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(
        "Chroma indexed %s with %s chunks (doc_id=%s user_id=%s)",
        filename,
        len(chunks),
        doc_id,
        user_id,
    )
    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}


def query_vectors(
    user_id: str,
    query_embedding: List[float],
    n_results: int,
) -> Dict[str, Any]:
    collection = get_chroma_collection()
    owned = collection.get(where={"user_id": user_id}, include=[])
    owned_count = len(owned.get("ids") or [])
    if owned_count == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    n = min(n_results, owned_count)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"],
    )


def get_document_ingest_dates(user_id: str) -> dict:
    """Chroma backend does not persist ingest timestamps."""
    return {}


def list_documents(user_id: str) -> List[dict]:
    collection = get_chroma_collection()
    items = collection.get(where={"user_id": user_id}, include=["metadatas"])
    seen: Dict[str, dict] = {}
    for meta in items.get("metadatas") or []:
        if not meta:
            continue
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta["filename"],
                "chunks": 0,
            }
        seen[doc_id]["chunks"] += 1
    return list(seen.values())


def delete_document(user_id: str, doc_id: str) -> bool:
    collection = get_chroma_collection()
    existing = collection.get(
        where={"$and": [{"doc_id": doc_id}, {"user_id": user_id}]},
        include=[],
    )
    ids = existing.get("ids") or []
    if not ids:
        return False

    collection.delete(ids=ids)
    logger.info("Chroma deleted document %s for user %s (%s chunks)", doc_id, user_id, len(ids))
    return True


def chunk_count(user_id: str | None = None) -> int:
    collection = get_chroma_collection()
    if user_id is None:
        return collection.count()
    items = collection.get(where={"user_id": user_id}, include=[])
    return len(items.get("ids") or [])
