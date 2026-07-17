"""
LangSmith / observability bootstrap.

Call configure_observability() once at app startup (before any LangChain calls).
When LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is set and tracing is enabled,
retriever + LLM + LangGraph nodes appear in the LangSmith UI.
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings

logger = logging.getLogger(__name__)
_configured = False


def configure_observability() -> dict:
    """
    Configure LangSmith tracing from settings / env.
    Safe to call multiple times; only applies once.
    """
    global _configured
    settings = get_settings()

    api_key = (
        settings.langsmith_api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()

    project = settings.langsmith_project
    tracing = settings.langsmith_tracing and bool(api_key)

    # Official LangSmith env vars (and legacy LANGCHAIN_* for compatibility)
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key

    info = {
        "langsmith_tracing": tracing,
        "langsmith_project": project,
        "langsmith_api_key_set": bool(api_key),
        "langgraph": True,
    }

    if tracing:
        logger.info("LangSmith tracing ON — project=%s", project)
    elif settings.langsmith_tracing and not api_key:
        logger.warning(
            "LANGSMITH_TRACING requested but no API key — tracing disabled. "
            "Set LANGSMITH_API_KEY in backend/.env (https://smith.langchain.com)"
        )
    else:
        logger.info("LangSmith tracing OFF")

    _configured = True
    return info


def observability_info() -> dict:
    settings = get_settings()
    api_key = (
        settings.langsmith_api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()
    tracing = settings.langsmith_tracing and bool(api_key)
    return {
        "langsmith_tracing": tracing,
        "langsmith_project": settings.langsmith_project,
        "langsmith_api_key_set": bool(api_key),
        "orchestration": "langgraph",
    }
