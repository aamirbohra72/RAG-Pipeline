"""
Neon Postgres + pgvector persistence.

Same public shape as the Chroma store so retrieval/RAG stay unchanged.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings
from app.services.chunking_service import Chunk
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

# mistral-embed output size
EMBEDDING_DIM = 1024

_pool: Optional[ConnectionPool] = None


def _database_url() -> str:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is required when VECTOR_BACKEND=pgvector. "
            "Add your Neon connection string to backend/.env"
        )
    # channel_binding can break some Windows/psycopg setups; Neon works without it
    return (
        settings.database_url.replace("&channel_binding=require", "")
        .replace("?channel_binding=require&", "?")
        .replace("?channel_binding=require", "")
    )


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        # Neon serverless closes idle SSL connections; recycle and health-check
        # pooled connections to avoid "SSL connection has been closed unexpectedly".
        _pool = ConnectionPool(
            conninfo=_database_url(),
            min_size=1,
            max_size=5,
            max_lifetime=300,
            max_idle=60,
            check=ConnectionPool.check_connection,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
        logger.info("Opened Neon/pgvector connection pool")
    return _pool


def init_schema() -> None:
    """Create extension + table + HNSW index (idempotent)."""
    pool = get_pool()
    with pool.connection() as conn:
        # Extension must exist before register_vector looks up the type OID
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY,
                user_id TEXT NOT NULL,
                doc_id UUID NOT NULL,
                filename TEXT NOT NULL,
                page INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                doc_type TEXT NOT NULL DEFAULT 'pdf',
                section TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # Backfill columns on existing deployments
        conn.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT 'pdf'
            """
        )
        conn.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN IF NOT EXISTS section TEXT NOT NULL DEFAULT ''
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_user_id_idx
            ON document_chunks (user_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_doc_user_idx
            ON document_chunks (doc_id, user_id)
            """
        )
        conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = 'document_chunks_embedding_hnsw'
                ) THEN
                    CREATE INDEX document_chunks_embedding_hnsw
                    ON document_chunks
                    USING hnsw (embedding vector_cosine_ops);
                END IF;
            END $$;
            """
        )
    logger.info("Neon pgvector schema ready (document_chunks, dim=%s)", EMBEDDING_DIM)


def add_document(user_id: str, filename: str, chunks: List[Chunk]) -> dict:
    if not chunks:
        raise ValueError(f"No extractable text chunks for {filename}")

    doc_id = str(uuid.uuid4())
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {len(embeddings[0])}"
        )

    rows = []
    doc_type = chunks[0].doc_type if chunks else "pdf"
    for chunk, emb in zip(chunks, embeddings):
        rows.append(
            (
                str(uuid.uuid4()),
                user_id,
                doc_id,
                filename,
                chunk.page,
                chunk.text,
                emb,
                chunk.doc_type or doc_type,
                chunk.section or "",
            )
        )

    pool = get_pool()
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (id, user_id, doc_id, filename, page, content, embedding,
                     doc_type, section)
                VALUES (%s, %s, %s::uuid, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    logger.info(
        "Neon indexed %s with %s chunks (doc_id=%s user_id=%s)",
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
    """Return Chroma-compatible nested lists for retrieval.py."""
    if n_results <= 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    pool = get_pool()
    with pool.connection() as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT id, content, filename, page, doc_id, doc_type, section,
                   (embedding <=> %s::vector) AS distance
            FROM document_chunks
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, user_id, query_embedding, n_results),
        ).fetchall()

    if not rows:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    ids = [str(r["id"]) for r in rows]
    documents = [r["content"] for r in rows]
    metadatas = [
        {
            "doc_id": str(r["doc_id"]),
            "user_id": user_id,
            "filename": r["filename"],
            "page": int(r["page"]),
            "doc_type": r.get("doc_type") or "pdf",
            "section": r.get("section") or "",
        }
        for r in rows
    ]
    distances = [float(r["distance"]) for r in rows]
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def list_documents(user_id: str) -> List[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT doc_id::text AS doc_id, filename, COUNT(*)::int AS chunks,
                   MIN(created_at) AS ingested_at
            FROM document_chunks
            WHERE user_id = %s
            GROUP BY doc_id, filename
            ORDER BY MIN(created_at) DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "chunks": r["chunks"],
            "ingested_at": r["ingested_at"].isoformat() if r.get("ingested_at") else None,
        }
        for r in rows
    ]


def get_document_ingest_dates(user_id: str) -> Dict[str, Any]:
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT doc_id::text AS doc_id, MIN(created_at) AS ingested_at
            FROM document_chunks
            WHERE user_id = %s
            GROUP BY doc_id
            """,
            (user_id,),
        ).fetchall()
    return {r["doc_id"]: r["ingested_at"] for r in rows}


def delete_document(user_id: str, doc_id: str) -> bool:
    pool = get_pool()
    with pool.connection() as conn:
        result = conn.execute(
            """
            DELETE FROM document_chunks
            WHERE doc_id = %s::uuid AND user_id = %s
            """,
            (doc_id, user_id),
        )
        deleted = result.rowcount or 0
    if deleted:
        logger.info(
            "Neon deleted document %s for user %s (%s chunks)",
            doc_id,
            user_id,
            deleted,
        )
        return True
    return False


def chunk_count(user_id: str | None = None) -> int:
    pool = get_pool()
    with pool.connection() as conn:
        if user_id is None:
            row = conn.execute("SELECT COUNT(*)::int AS n FROM document_chunks").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*)::int AS n FROM document_chunks WHERE user_id = %s",
                (user_id,),
            ).fetchone()
    return int(row["n"] if row else 0)


def ping() -> bool:
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("SELECT 1")
    return True
