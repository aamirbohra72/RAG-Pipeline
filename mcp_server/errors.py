"""Translate FastAPI HTTP errors into actionable MCP tool messages."""

from __future__ import annotations

import json
from typing import Any

import httpx


class RagApiError(Exception):
    """Raised when the RAG backend returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _extract_detail(body: Any) -> str | None:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and "msg" in first:
                return str(first["msg"])
            return str(first)
    if isinstance(body, str) and body.strip():
        return body.strip()
    return None


def translate_http_error(response: httpx.Response) -> RagApiError:
    detail: str | None = None
    try:
        payload = response.json()
        detail = _extract_detail(payload)
    except (json.JSONDecodeError, ValueError):
        text = response.text.strip()
        if text:
            detail = text[:500]

    status = response.status_code
    if status == 401:
        return RagApiError(
            "Authentication failed: the API token is missing, invalid, or expired. "
            "Mint a fresh JWT via POST /auth/login and update RAG_API_TOKEN or RAG_USER_JWT.",
            status_code=status,
        )
    if status == 404:
        msg = detail or "The requested resource was not found."
        if "document" in msg.lower():
            return RagApiError(f"No matching documents: {msg}", status_code=status)
        if "job" in msg.lower():
            return RagApiError(f"Ingestion job not found: {msg}", status_code=status)
        return RagApiError(msg, status_code=status)
    if status == 429:
        return RagApiError(
            "Rate limited by the RAG backend. Retry after a short delay.",
            status_code=status,
        )
    if status == 503:
        return RagApiError(
            detail
            or "RAG backend unavailable (async ingest or dependencies may be down).",
            status_code=status,
        )
    if status >= 500:
        return RagApiError(
            "RAG backend error — the service is temporarily unavailable. "
            f"{'Detail: ' + detail if detail else 'Try again shortly.'}",
            status_code=status,
        )
    return RagApiError(detail or f"Request failed with HTTP {status}", status_code=status)


def translate_transport_error(exc: Exception) -> RagApiError:
    if isinstance(exc, httpx.ConnectError):
        return RagApiError(
            "Cannot reach the RAG backend. Ensure FastAPI is running and "
            "RAG_API_BASE_URL is correct."
        )
    if isinstance(exc, httpx.TimeoutException):
        return RagApiError(
            "RAG backend request timed out. The query or ingestion may still be running."
        )
    if isinstance(exc, RagApiError):
        return exc
    return RagApiError(f"Unexpected client error: {exc}")
