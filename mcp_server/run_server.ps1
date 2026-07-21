# Launch RAG MCP server for Cursor (Windows)
$ErrorActionPreference = "Stop"
Set-Location "D:\GenAI-RAG\rag-project\rag-project"
$env:PYTHONPATH = "D:\GenAI-RAG\rag-project\rag-project"
if (-not $env:RAG_API_BASE_URL) { $env:RAG_API_BASE_URL = "http://127.0.0.1:8000" }
& "D:\GenAI-RAG\.venv\Scripts\python.exe" -m mcp_server
