"""Business logic for MCP tools (transport-agnostic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_server.api_client import RagApiClient
from mcp_server.conversations import conversation_store
from mcp_server.errors import RagApiError


def _doc_type_from_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext or "unknown"


def _derive_confidence(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "low"
    scores: list[float] = []
    for src in sources:
        for key in ("rerank_score", "score"):
            val = src.get(key)
            if isinstance(val, (int, float)):
                scores.append(float(val))
                break
    if not scores:
        return "medium"
    top = max(scores)
    if top >= 0.75:
        return "high"
    if top >= 0.45:
        return "medium"
    return "low"


async def handle_search_documents(
    client: RagApiClient,
    *,
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
    date_after: str | None = None,
) -> dict[str, Any]:
    data = await client.search_documents(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        date_after=date_after,
    )
    results = []
    for item in data.get("results") or []:
        results.append(
            {
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "doc_id": item.get("doc_id", ""),
                "score": item.get("score"),
                "citation_metadata": item.get("citation_metadata") or {
                    "filename": item.get("source"),
                    "page": item.get("page"),
                    "doc_id": item.get("doc_id"),
                },
            }
        )
    if not results:
        return {
            "results": [],
            "message": "No documents matched the query and filters.",
        }
    return {"results": results}


async def handle_ask_question(
    client: RagApiClient,
    *,
    query: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    history = conversation_store.get_history(conversation_id)
    data = await client.ask_question(query=query, history=history)
    answer = data.get("answer", "")
    sources = data.get("sources") or []
    citations = [
        {
            "doc_id": src.get("doc_id") or "",
            "source": src.get("filename", ""),
            "excerpt": src.get("snippet", ""),
        }
        for src in sources
    ]
    new_conversation_id = conversation_store.append_turn(
        conversation_id,
        user_message=query,
        assistant_message=answer,
    )
    return {
        "answer": answer,
        "citations": citations,
        "confidence": _derive_confidence(sources),
        "conversation_id": new_conversation_id,
    }


async def handle_list_documents(
    client: RagApiClient,
    *,
    doc_type: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    data = await client.list_documents()
    documents = []
    for doc in data.get("documents") or []:
        filename = doc.get("filename", "")
        dtype = _doc_type_from_filename(filename)
        if doc_type and dtype != doc_type.lower().strip("."):
            continue
        documents.append(
            {
                "doc_id": doc.get("doc_id"),
                "title": filename,
                "doc_type": dtype,
                "ingested_at": doc.get("ingested_at"),
                "status": "ready",
            }
        )
    return {"documents": documents[:limit]}


async def handle_get_ingestion_status(
    client: RagApiClient,
    *,
    job_id: str,
) -> dict[str, Any]:
    data = await client.get_ingestion_status(job_id)
    state = data.get("state", "UNKNOWN")
    stage = data.get("stage")
    progress_map = {
        "reading": 0.15,
        "extracting": 0.35,
        "chunking": 0.55,
        "embedding": 0.8,
        "done": 1.0,
        "failed": 0.0,
    }
    progress: float | None = None
    if state == "SUCCESS":
        progress = 1.0
    elif state == "FAILURE":
        progress = 0.0
    elif stage in progress_map:
        progress = progress_map[stage]

    return {
        "status": state,
        "progress": progress,
        "stage": stage,
        "error": data.get("error"),
        "result": data.get("result"),
    }


def format_tool_error(exc: Exception) -> str:
    if isinstance(exc, RagApiError):
        return exc.message
    return f"Tool execution failed: {exc}"
