"""
Standalone query rewriting for pronoun-dependent follow-ups.

Uses a small/fast LLM call, traced separately in LangSmith as ``query_rewrite``.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langsmith import traceable

from app.config import get_settings

logger = logging.getLogger(__name__)

_FOLLOWUP_PATTERNS = re.compile(
    r"\b("
    r"it|they|them|those|these|that|this|the second|the first|the third|"
    r"the last|the other|above|below|same|also|what about|how about|"
    r"and what|tell me more|more detail|expand on|elaborate"
    r")\b",
    re.I,
)


def _history_to_messages(history: List[dict]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for turn in history[-4:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _looks_like_followup(question: str, history: List[dict]) -> bool:
    if not history:
        return False
    q = question.strip()
    if len(q.split()) <= 12 and _FOLLOWUP_PATTERNS.search(q):
        return True
    # Short questions with prior context are often follow-ups
    if len(q.split()) <= 6 and history:
        return True
    return False


@lru_cache
def _get_rewrite_llm():
    settings = get_settings()
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=settings.rewrite_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0.0,
        max_retries=2,
        streaming=False,
    )


@traceable(name="query_rewrite", run_type="llm")
def rewrite_query_if_needed(
    question: str,
    history: Optional[List[dict]] = None,
) -> dict:
    """
    Rewrite pronoun-dependent follow-ups into standalone retrieval queries.

    Returns:
        {
            "original_question": str,
            "retrieval_query": str,
            "was_rewritten": bool,
        }
    """
    history = history or []
    original = question.strip()

    if not _looks_like_followup(original, history):
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
        }

    settings = get_settings()
    if not settings.rewrite_enabled:
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
        }

    llm = _get_rewrite_llm()
    recent = history[-4:]
    context_lines = []
    for turn in recent:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            context_lines.append(f"{role}: {content}")

    system = SystemMessage(
        content=(
            "You rewrite follow-up questions into standalone search queries. "
            "The user asked a question that depends on prior conversation context. "
            "Produce ONE self-contained question that includes all necessary entities, "
            "topics, and references from the conversation. "
            "Do not answer the question — only output the rewritten query as plain text."
        )
    )
    human = HumanMessage(
        content=(
            "Recent conversation:\n"
            + "\n".join(context_lines)
            + f"\n\nFollow-up question: {original}\n\nStandalone query:"
        )
    )

    try:
        response = llm.invoke([system, human])
        rewritten = (response.content or "").strip()
        if isinstance(rewritten, list):
            parts = []
            for block in rewritten:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            rewritten = "".join(parts).strip()

        if not rewritten or len(rewritten) < 3:
            rewritten = original
            was_rewritten = False
        else:
            was_rewritten = rewritten.lower() != original.lower()

        if was_rewritten:
            logger.info("Query rewrite: %r → %r", original, rewritten)

        return {
            "original_question": original,
            "retrieval_query": rewritten,
            "was_rewritten": was_rewritten,
        }
    except Exception:
        logger.exception("Query rewrite failed; using original question")
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
        }
