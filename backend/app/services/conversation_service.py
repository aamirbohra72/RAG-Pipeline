"""
Persistent multi-turn chat history in Neon Postgres.

Tables: conversations, messages — created idempotently alongside document_chunks.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, List, Optional

from pgvector.psycopg import register_vector

from app.config import get_settings
from app.services.pgvector_store import get_pool

logger = logging.getLogger(__name__)


def init_conversation_schema() -> None:
    """Create conversations + messages tables (idempotent)."""
    pool = get_pool()
    with pool.connection() as conn:
        register_vector(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS conversations_user_id_idx
            ON conversations (user_id, updated_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id UUID PRIMARY KEY,
                conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS messages_conversation_idx
            ON messages (conversation_id, created_at)
            """
        )
    logger.info("Conversation schema ready (conversations, messages)")


def create_conversation(user_id: str) -> str:
    conversation_id = str(uuid.uuid4())
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, user_id)
            VALUES (%s::uuid, %s)
            """,
            (conversation_id, user_id),
        )
    return conversation_id


def _conversation_owned(conversation_id: str, user_id: str) -> bool:
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM conversations
            WHERE id = %s::uuid AND user_id = %s
            """,
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def get_or_create_conversation(
    user_id: str, conversation_id: Optional[str] = None
) -> str:
    if conversation_id:
        if _conversation_owned(conversation_id, user_id):
            return conversation_id
        raise ValueError(f"Conversation {conversation_id} not found for user")
    return create_conversation(user_id)


def load_history(
    conversation_id: str,
    user_id: str,
    max_turns: Optional[int] = None,
) -> List[dict]:
    """Return [{role, content}] for the last N turns (user+assistant pairs)."""
    settings = get_settings()
    limit = max_turns if max_turns is not None else settings.chat_history_turns

    if not _conversation_owned(conversation_id, user_id):
        raise ValueError(f"Conversation {conversation_id} not found for user")

    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = %s::uuid
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (conversation_id, limit),
        ).fetchall()

    # Chronological order for the LLM
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return history


def save_turn(
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    citations: Optional[List[dict]] = None,
) -> None:
    if not _conversation_owned(conversation_id, user_id):
        raise ValueError(f"Conversation {conversation_id} not found for user")

    citations_json = json.dumps(citations) if citations else None
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content)
                VALUES (%s::uuid, %s::uuid, 'user', %s)
                """,
                (str(uuid.uuid4()), conversation_id, user_message),
            )
            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, citations)
                VALUES (%s::uuid, %s::uuid, 'assistant', %s, %s::jsonb)
                """,
                (
                    str(uuid.uuid4()),
                    conversation_id,
                    assistant_message,
                    citations_json,
                ),
            )
            cur.execute(
                """
                UPDATE conversations SET updated_at = NOW()
                WHERE id = %s::uuid
                """,
                (conversation_id,),
            )


def list_conversations(user_id: str, limit: int = 20) -> List[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT c.id::text AS conversation_id, c.created_at, c.updated_at,
                   (SELECT COUNT(*)::int FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
            WHERE c.user_id = %s
            ORDER BY c.updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "conversation_id": r["conversation_id"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
            "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
            "message_count": r["message_count"],
        }
        for r in rows
    ]
