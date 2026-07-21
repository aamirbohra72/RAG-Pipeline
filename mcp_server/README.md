# RAG Platform MCP Server

Thin [Model Context Protocol](https://modelcontextprotocol.io) server that exposes your existing FastAPI RAG backend as MCP tools. All retrieval and generation logic stays in the backend — this package only proxies HTTP calls with JWT auth.

## Prerequisites

1. **FastAPI backend running** (default `http://localhost:8000`):

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **A JWT for authentication.** The backend enforces per-user document isolation via `user_id` in the JWT payload. Mint a token for the user whose knowledge base you want to query:

   ```bash
   curl -s -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"you@example.com","password":"your-password"}' \
     | jq -r .access_token
   ```

   Use that value as `RAG_API_TOKEN` (service account style) or `RAG_USER_JWT` (per-user session).

## Install

```bash
cd mcp_server
pip install -r requirements.txt
```

Or from the parent directory:

```bash
pip install -r mcp_server/requirements.txt
```

## Run locally (stdio)

Stdio is the default transport for Claude Desktop, Cursor, and local dev:

```bash
# From rag-project/rag-project (parent of mcp_server/)
export RAG_API_BASE_URL=http://localhost:8000
export RAG_API_TOKEN=eyJ...   # JWT from /auth/login

python -m mcp_server
```

The process reads JSON-RPC from stdin and writes to stdout. **Do not print to stdout** — use stderr for logs.

### Streamable HTTP (future)

Set `TRANSPORT=streamable-http` to use HTTP transport when your MCP SDK version supports it. Tool logic is unchanged; only `mcp_server/transport.py` selects the runner.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RAG_API_BASE_URL` | No | `http://localhost:8000` | FastAPI base URL (no trailing slash) |
| `RAG_API_TOKEN` | Yes* | — | Service-scoped JWT for backend auth |
| `RAG_USER_JWT` | Yes* | — | Per-session user JWT (overrides service token when `PREFER_USER_JWT=true`) |
| `PREFER_USER_JWT` | No | `true` | Prefer `RAG_USER_JWT` over `RAG_API_TOKEN` when both are set |
| `REQUEST_TIMEOUT_SECONDS` | No | `120` | HTTP timeout for backend calls |
| `TRANSPORT` | No | `stdio` | `stdio` or `streamable-http` |

\* At least one of `RAG_API_TOKEN` or `RAG_USER_JWT` must be set.

### Per-user permissions

The backend scopes all vector queries and document lists to the authenticated user's `user_id`. To query a specific user's knowledge base, set `RAG_USER_JWT` to that user's JWT in the MCP client config. When both tokens are present, `RAG_USER_JWT` wins by default (`PREFER_USER_JWT=true`).

## MCP tools

| Tool | Backend endpoint | Purpose |
|------|------------------|---------|
| `search_documents` | `POST /search` | Hybrid retrieval + re-rank only (no LLM) |
| `ask_question` | `POST /query` | Full LangGraph RAG with citations |
| `list_documents` | `GET /documents` | Indexed documents for the user |
| `get_ingestion_status` | `GET /jobs/{job_id}` | Celery ingest job polling |

## Register in Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json` on Windows (or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-platform": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "D:\\GenAI-RAG\\rag-project\\rag-project",
      "env": {
        "RAG_API_BASE_URL": "http://localhost:8000",
        "RAG_API_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
      }
    }
  }
}
```

Use the absolute path to your `rag-project/rag-project` directory for `cwd`. Restart Claude Desktop after saving.

## Register in Cursor

Add to Cursor MCP settings (`.cursor/mcp.json` in your project or global Cursor MCP config):

```json
{
  "mcpServers": {
    "rag-platform": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "D:\\GenAI-RAG\\rag-project\\rag-project",
      "env": {
        "RAG_API_BASE_URL": "http://localhost:8000",
        "RAG_USER_JWT": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
      }
    }
  }
}
```

For per-user access in a shared Cursor workspace, give each developer their own `RAG_USER_JWT` in their local MCP config (not committed to git).

## Sanity-check with the test client

```bash
cd rag-project/rag-project
export RAG_API_TOKEN=eyJ...
python mcp_server/test_mcp_client.py
```

Optional: set `SAMPLE_JOB_ID` to a real Celery job ID from `POST /upload/async` to test `get_ingestion_status`.

The script spawns the server over stdio, lists tools, and prints raw MCP responses for each tool call.

## Error handling

HTTP errors from FastAPI are translated into short, actionable MCP tool errors:

- **401** → auth token invalid/expired
- **404** → document or job not found
- **429** → rate limited
- **503** → backend/async ingest unavailable
- **5xx** → backend error (no stack traces leaked to the LLM)

Connection and timeout failures are reported separately from HTTP status errors.
