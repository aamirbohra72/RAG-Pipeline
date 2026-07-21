"""MCP tool registration — transport-agnostic server definition."""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server.api_client import RagApiClient
from mcp_server.config import get_settings
from mcp_server.tool_handlers import (
    format_tool_error,
    handle_ask_question,
    handle_get_ingestion_status,
    handle_list_documents,
    handle_search_documents,
)

mcp = FastMCP(
    "rag-platform",
    instructions=(
        "Tools for querying a production RAG knowledge base backed by hybrid "
        "vector + keyword retrieval with re-ranking and LangGraph generation. "
        "Use search_documents for raw grounded passages without an LLM answer. "
        "Use ask_question for a synthesized answer with citations. "
        "Use list_documents to inspect what is indexed before querying."
    ),
)

_client = RagApiClient()


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


async def _run_tool(coro) -> str:
    try:
        result = await coro
        return _json_result(result)
    except Exception as exc:
        # FastMCP surfaces raised exceptions as tool errors (isError=true).
        raise RuntimeError(format_tool_error(exc)) from exc


@mcp.tool(
    name="search_documents",
    description=(
        "Run hybrid retrieval + cross-encoder re-ranking only — no LLM generation. "
        "Returns raw grounded text passages from the user's knowledge base with "
        "relevance scores and citation metadata (filename, page, doc_id). "
        "Use when you need verbatim source material, quotes, or evidence snippets "
        "to cite yourself rather than a synthesized answer. "
        "Optional filters: doc_type (file extension, e.g. 'pdf') and "
        "date_after (ISO date for documents ingested on/after that date). "
        "Returns an empty results list with a message when nothing matches."
    ),
)
async def search_documents(
    query: str,
    top_k: int = 5,
    doc_type: str | None = None,
    date_after: str | None = None,
) -> str:
    return await _run_tool(
        handle_search_documents(
            _client,
            query=query,
            top_k=top_k,
            doc_type=doc_type,
            date_after=date_after,
        )
    )


@mcp.tool(
    name="ask_question",
    description=(
        "Run the full LangGraph RAG pipeline: retrieve → grade context → "
        "generate an LLM answer grounded in the user's documents. "
        "Returns answer text, citations (source filename + excerpt), a "
        "confidence label (high/medium/low derived from retrieval scores), "
        "and conversation_id for multi-turn follow-ups. "
        "Pass conversation_id from a prior call to include chat history. "
        "Use when the user wants a direct answer, summary, or explanation "
        "rather than raw passages."
    ),
)
async def ask_question(
    query: str,
    conversation_id: str | None = None,
) -> str:
    return await _run_tool(
        handle_ask_question(
            _client,
            query=query,
            conversation_id=conversation_id,
        )
    )


@mcp.tool(
    name="list_documents",
    description=(
        "List documents currently indexed in the authenticated user's knowledge "
        "base. Returns doc_id, title (filename), doc_type (file extension), "
        "ingested_at timestamp when available, and status ('ready'). "
        "Use before search/ask to discover available sources or to confirm "
        "uploads finished indexing. Optional doc_type filter and limit."
    ),
)
async def list_documents(
    doc_type: str | None = None,
    limit: int = 20,
) -> str:
    return await _run_tool(
        handle_list_documents(
            _client,
            doc_type=doc_type,
            limit=limit,
        )
    )


@mcp.tool(
    name="get_ingestion_status",
    description=(
        "Poll async PDF ingestion job status (OCR, chunking, embedding). "
        "Call with job_id returned from POST /upload/async. "
        "Returns status (PENDING, PROGRESS, SUCCESS, FAILURE), numeric progress "
        "when available, processing stage, error message on failure, and result "
        "payload on success."
    ),
)
async def get_ingestion_status(job_id: str) -> str:
    return await _run_tool(
        handle_get_ingestion_status(_client, job_id=job_id)
    )


def create_server() -> FastMCP:
    """Factory for tests and alternate transports."""
    settings = get_settings()
    print(
        f"RAG MCP server configured for {settings.rag_api_base_url} "
        f"(transport={settings.transport})",
        file=sys.stderr,
    )
    return mcp
