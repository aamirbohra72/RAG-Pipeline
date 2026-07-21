"""
RAG orchestration facade.

Generation path: LangGraph (rewrite → retrieve → generate) + LangSmith tracing.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from app.schemas import Source
from app.services import langgraph_rag


def answer_question(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[str, List[Source], dict]:
    return langgraph_rag.answer_with_langgraph(question, user_id, history)


def stream_answer(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[List[Source], Iterator[str], dict]:
    return langgraph_rag.stream_with_langgraph(question, user_id, history)
