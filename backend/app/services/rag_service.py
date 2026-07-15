"""RAG orchestration: retrieve → prompt → generate (sync + stream)."""

from __future__ import annotations

import logging
from typing import Iterator, List

from app.config import get_settings
from app.dependencies import get_mistral_client
from app.schemas import Source
from app.services.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = (
    "You are a careful document assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources inline like [filename, page N] when you use them."
)


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | Source: {c.filename}, page {c.page} | score={c.score}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    chunks: List[RetrievedChunk],
    history: List[dict] | None = None,
) -> list[dict]:
    context = _build_context(chunks)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    # Keep only a short recent history so the prompt stays focused
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        }
    )
    return messages


def _sources_from_chunks(chunks: List[RetrievedChunk]) -> List[Source]:
    return [
        Source(
            filename=c.filename,
            page=c.page,
            snippet=c.text[:220],
            score=c.score,
            rerank_score=c.rerank_score,
            vector_score=c.vector_score,
            lexical_score=c.lexical_score,
        )
        for c in chunks
    ]


def answer_question(
    question: str,
    user_id: str,
    history: List[dict] | None = None,
) -> tuple[str, List[Source]]:
    chunks = retrieve(question, user_id)
    if not chunks:
        return "No documents have been uploaded yet.", []

    settings = get_settings()
    client = get_mistral_client()
    messages = _build_messages(question, chunks, history)

    logger.info("Generating answer with %s context chunks (user=%s)", len(chunks), user_id)
    response = client.chat.complete(model=settings.chat_model, messages=messages)
    answer = response.choices[0].message.content or ""
    return answer, _sources_from_chunks(chunks)


def stream_answer(
    question: str,
    user_id: str,
    history: List[dict] | None = None,
) -> tuple[List[Source], Iterator[str]]:
    """
    Returns (sources, token_iterator).
    Sources are known up-front after retrieval; tokens stream from Mistral.
    """
    chunks = retrieve(question, user_id)
    if not chunks:
        def empty() -> Iterator[str]:
            yield "No documents have been uploaded yet."

        return [], empty()

    settings = get_settings()
    client = get_mistral_client()
    messages = _build_messages(question, chunks, history)
    sources = _sources_from_chunks(chunks)

    stream = client.chat.stream(model=settings.chat_model, messages=messages)

    def token_iter() -> Iterator[str]:
        for event in stream:
            delta = getattr(event.data.choices[0].delta, "content", None)
            if delta:
                yield delta

    return sources, token_iter()
