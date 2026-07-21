#!/usr/bin/env python3
"""
Sanity-check script: spawn the MCP server over stdio and call each tool once.

Usage (from rag-project/rag-project with deps installed):
  python mcp_server/test_mcp_client.py

Set RAG_API_TOKEN or RAG_USER_JWT in the environment before running.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RAG_API_BASE_URL", "http://localhost:8000")
    if not env.get("RAG_API_TOKEN") and not env.get("RAG_USER_JWT"):
        print(
            "Warning: set RAG_API_TOKEN or RAG_USER_JWT for authenticated calls.",
            file=sys.stderr,
        )
    return env


async def _call_tool(session: ClientSession, name: str, arguments: dict) -> None:
    print(f"\n=== tools/call: {name} ===", file=sys.stderr)
    print(f"arguments: {json.dumps(arguments)}", file=sys.stderr)
    result = await session.call_tool(name, arguments)
    payload = {
        "isError": result.isError,
        "content": [block.model_dump() for block in result.content],
        "structuredContent": getattr(result, "structuredContent", None),
    }
    print(json.dumps(payload, indent=2, default=str))


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=str(ROOT),
        env=_server_env(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(
                "Available tools:",
                [t.name for t in tools.tools],
                file=sys.stderr,
            )

            await _call_tool(
                session,
                "list_documents",
                {"limit": 5},
            )
            await _call_tool(
                session,
                "search_documents",
                {"query": "What topics are covered in the uploaded documents?", "top_k": 3},
            )
            ask_result = await session.call_tool(
                "ask_question",
                {"query": "Summarize the main themes in my knowledge base."},
            )
            print("\n=== tools/call: ask_question ===", file=sys.stderr)
            print(
                json.dumps(
                    {
                        "isError": ask_result.isError,
                        "content": [b.model_dump() for b in ask_result.content],
                    },
                    indent=2,
                    default=str,
                )
            )

            conversation_id = None
            if ask_result.content:
                try:
                    body = json.loads(ask_result.content[0].text)
                    conversation_id = body.get("conversation_id")
                except (json.JSONDecodeError, AttributeError, IndexError):
                    pass

            if conversation_id:
                await _call_tool(
                    session,
                    "ask_question",
                    {
                        "query": "Can you elaborate on the most important point?",
                        "conversation_id": conversation_id,
                    },
                )

            sample_job_id = os.environ.get("SAMPLE_JOB_ID", "00000000-0000-0000-0000-000000000000")
            await _call_tool(
                session,
                "get_ingestion_status",
                {"job_id": sample_job_id},
            )


if __name__ == "__main__":
    asyncio.run(main())
