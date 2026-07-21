"""Transport entrypoints."""

from __future__ import annotations

import sys

from mcp_server.config import get_settings
from mcp_server.server import create_server


def run_stdio() -> None:
    """Run MCP over stdio (default for Claude Desktop / Cursor)."""
    settings = get_settings()
    server = create_server()
    server.run(transport="stdio")


def run_streamable_http(host: str = "127.0.0.1", port: int = 8765) -> None:
    """
    Run MCP over streamable HTTP (for remote / multi-client deployments).

    Requires a compatible MCP SDK build with streamable-http transport support.
    """
    settings = get_settings()
    server = create_server()
    try:
        server.run(transport="streamable-http", host=host, port=port)
    except TypeError as exc:
        print(
            "streamable-http transport is not available in this MCP SDK version. "
            "Upgrade mcp or use stdio transport.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def run() -> None:
    settings = get_settings()
    if settings.transport == "streamable-http":
        run_streamable_http()
    else:
        run_stdio()
