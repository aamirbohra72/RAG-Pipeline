"""HTTP client that proxies MCP tool calls to FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from mcp_server.config import Settings, get_settings
from mcp_server.errors import RagApiError, translate_http_error, translate_transport_error


class RagApiClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._token_override: str | None = None

    def set_session_token(self, token: str | None) -> None:
        """Optional per-call user JWT override (e.g. from tool context)."""
        self._token_override = token

    def _auth_headers(self) -> dict[str, str]:
        token = self._token_override
        if not token:
            token = self.settings.resolve_auth_token()
        return {"Authorization": f"Bearer {token}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.settings.rag_api_base_url.rstrip('/')}{path}"
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = await client.request(
                    method,
                    url,
                    headers=self._auth_headers(),
                    json=json,
                    params=params,
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc) from exc

        if response.status_code >= 400:
            raise translate_http_error(response)
        if response.status_code == 204:
            return None
        return response.json()

    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
        date_after: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if doc_type is not None:
            payload["doc_type"] = doc_type
        if date_after is not None:
            payload["date_after"] = date_after
        return await self._request("POST", "/search", json=payload)

    async def ask_question(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": query}
        if history:
            payload["history"] = history
        return await self._request("POST", "/query", json=payload)

    async def list_documents(self) -> dict[str, Any]:
        return await self._request("GET", "/documents")

    async def get_ingestion_status(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")
