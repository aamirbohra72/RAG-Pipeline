"""Environment-driven configuration for the MCP server."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the RAG FastAPI backend (no trailing slash).",
    )
    rag_api_token: Optional[str] = Field(
        default=None,
        description="Service-scoped JWT minted for this MCP server (RAG_API_TOKEN).",
    )
    rag_user_jwt: Optional[str] = Field(
        default=None,
        description=(
            "Optional per-session user JWT (RAG_USER_JWT). When set and "
            "prefer_user_jwt=true, overrides the service token so results "
            "respect that user's document permissions."
        ),
    )
    prefer_user_jwt: bool = Field(
        default=True,
        description="Use RAG_USER_JWT instead of RAG_API_TOKEN when both are set.",
    )
    request_timeout_seconds: float = Field(default=120.0, ge=5.0)
    transport: Literal["stdio", "streamable-http"] = Field(default="stdio")

    def resolve_auth_token(self) -> str:
        if self.prefer_user_jwt and self.rag_user_jwt:
            return self.rag_user_jwt
        if self.rag_api_token:
            return self.rag_api_token
        if self.rag_user_jwt:
            return self.rag_user_jwt
        raise ValueError(
            "No auth token configured. Set RAG_API_TOKEN (service JWT) or "
            "RAG_USER_JWT (per-user JWT) in the MCP server environment."
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
