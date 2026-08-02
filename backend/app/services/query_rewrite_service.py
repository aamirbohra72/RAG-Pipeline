"""
Standalone query rewriting for pronoun-dependent follow-ups.

Uses a small/fast LLM call, traced separately in LangSmith as ``query_rewrite``.
Ordinals like "the second one" are resolved against the last assistant list
order and rewritten into a query that names the exact entity.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.config import get_settings

logger = logging.getLogger(__name__)

_FOLLOWUP_PATTERNS = re.compile(
    r"\b("
    r"it|they|them|those|these|that|this|the second|the first|the third|"
    r"the last|the other|above|below|same|also|what about|how about|"
    r"and what|tell me more|more detail|expand on|elaborate|"
    r"first|second|third|fourth|fifth|last|previous|next"
    r")\b",
    re.I,
)

_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
    "sixth": 6,
    "6th": 6,
    "last": -1,
}

# "the second one", "second product", "2nd one", "what about the first?"
_ORDINAL_IN_QUESTION = re.compile(
    r"\b(?:the\s+)?(?P<ord>first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|last)\b"
    r"(?:\s+(?:one|item|product|option|entry|result))?",
    re.I,
)

# Leaf bullets only (not "1. Category:" headers)
_BULLET_ITEM = re.compile(
    r"(?m)^[ \t]*[-•][ \t]+(?:\*\*|__|\*)?"
    r"(?P<name>[A-Z][^:*\n]{2,80}?)"
    r"(?:\*\*|__|\*)?"
    r"(?:\s*[:(\-–—,]|\s*$)"
)

# Bold/italic product-like names anywhere in assistant text
_NAMED_ENTITY = re.compile(
    r"(?:\*\*|__|\*)([A-Z][^(*\n]{2,60}?)(?:\*\*|__|\*)"
)

_CATEGORY_HINTS = re.compile(
    r"(?i)\b(batter(?:y|ies)|systems?|inverters?|software|products?|"
    r"overview|sources?|details?|category|categories)\b"
)


def _last_assistant_message(history: List[dict]) -> str:
    for turn in reversed(history):
        if turn.get("role") == "assistant" and (turn.get("content") or "").strip():
            return turn["content"].strip()
    return ""


def _is_category_label(name: str) -> bool:
    cleaned = name.strip()
    lower = cleaned.lower()
    exact = {
        "residential batteries",
        "residential battery systems",
        "commercial battery",
        "commercial battery systems",
        "solar inverter",
        "solar inverters",
        "grid management software",
        "sources",
        "key details",
    }
    if lower in exact:
        return True
    # Brand/model tokens → keep as product (GridPilot Software, NovaCell Home 10)
    if re.search(
        r"(?i)\b(novacell|novagrid|solarsync|gridpilot|ruffwear|weatherbeeta)\b",
        cleaned,
    ):
        return False
    if re.search(r"\d|\bII\b|\bX\b", cleaned):
        return False
    # Generic headings: "Residential Battery Systems", "Solar Inverters"
    if _CATEGORY_HINTS.search(cleaned) and len(cleaned.split()) <= 4:
        return True
    return False


def _extract_ordered_entities(assistant_text: str) -> List[str]:
    """
    Build an ordered list of concrete entities from the last assistant reply.

    Uses bullet leaves only (``- NovaCell Home 10``), skipping numbered
    category headers (``1. Residential Battery Systems``).
    """
    if not assistant_text:
        return []

    entities: List[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        cleaned = re.sub(r"\s+", " ", name).strip(" .*_:-–—")
        if not cleaned or len(cleaned) < 3:
            return
        if _is_category_label(cleaned):
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        entities.append(cleaned)

    for match in _BULLET_ITEM.finditer(assistant_text):
        _add(match.group("name"))

    # Fallback: bold/italic names if no bullets (still skip categories)
    if len(entities) < 2:
        entities = []
        seen.clear()
        for match in _NAMED_ENTITY.finditer(assistant_text):
            _add(match.group(1))

    return entities


def _resolve_ordinal_entity(question: str, history: List[dict]) -> Optional[str]:
    """If the follow-up uses an ordinal, map it to an entity from the last list."""
    match = _ORDINAL_IN_QUESTION.search(question)
    if not match:
        return None

    ord_word = match.group("ord").lower()
    index = _ORDINAL_WORDS.get(ord_word)
    if index is None:
        return None

    entities = _extract_ordered_entities(_last_assistant_message(history))
    if not entities:
        return None

    if index == -1:
        return entities[-1]
    if 1 <= index <= len(entities):
        return entities[index - 1]
    return None


def _looks_like_followup(question: str, history: List[dict]) -> bool:
    if not history:
        return False
    q = question.strip()
    if len(q.split()) <= 14 and _FOLLOWUP_PATTERNS.search(q):
        return True
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


def _llm_rewrite(original: str, history: List[dict], resolved_entity: Optional[str]) -> str:
    recent = history[-4:]
    context_lines = []
    for turn in recent:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            context_lines.append(f"{role}: {content}")

    last_list = _extract_ordered_entities(_last_assistant_message(history))
    list_hint = ""
    if last_list:
        numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(last_list, start=1))
        list_hint = (
            "\n\nOrdered entities from the LAST assistant reply "
            "(use this order for first/second/third/last):\n"
            f"{numbered}\n"
        )

    entity_hint = ""
    if resolved_entity:
        entity_hint = (
            f"\nResolved ordinal entity: {resolved_entity}\n"
            "The rewritten query MUST include this exact name.\n"
        )

    system = SystemMessage(
        content=(
            "You rewrite follow-up questions into standalone search queries for a RAG system.\n"
            "Rules:\n"
            "1. Output ONE self-contained question only — do not answer it.\n"
            "2. If the user says first/second/third/last/that one, resolve it against the "
            "ordered entity list from the LAST assistant message (leaf items in order).\n"
            "3. The rewritten query MUST contain the exact entity/product/document name — "
            "never leave vague phrases like 'the second product' or 'the second one'.\n"
            "4. Do NOT invent a different ordering (e.g. by launch year) unless the user asked for that.\n"
            "5. Keep the user's intent (details, price, warranty, comparison, etc.).\n"
            "6. Plain text only — no quotes or labels."
        )
    )
    human = HumanMessage(
        content=(
            "Recent conversation:\n"
            + "\n".join(context_lines)
            + list_hint
            + entity_hint
            + f"\nFollow-up question: {original}\n\nStandalone query:"
        )
    )

    response = _get_rewrite_llm().invoke([system, human])
    rewritten = (response.content or "").strip()
    if isinstance(rewritten, list):
        parts = []
        for block in rewritten:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        rewritten = "".join(parts).strip()
    return rewritten


@traceable(name="query_rewrite", run_type="llm")
def rewrite_query_if_needed(
    question: str,
    history: Optional[List[dict]] = None,
) -> dict:
    """
    Rewrite pronoun/ordinal follow-ups into standalone retrieval queries.

    Returns:
        {
            "original_question": str,
            "retrieval_query": str,
            "was_rewritten": bool,
            "resolved_entity": str | None,
        }
    """
    history = history or []
    original = question.strip()

    if not _looks_like_followup(original, history):
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
            "resolved_entity": None,
        }

    settings = get_settings()
    if not settings.rewrite_enabled:
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
            "resolved_entity": None,
        }

    resolved = _resolve_ordinal_entity(original, history)

    # Deterministic fast path: ordinal + known entity → named query without LLM
    if resolved and _ORDINAL_IN_QUESTION.search(original):
        # Preserve user intent words beyond the ordinal phrase
        intent = _ORDINAL_IN_QUESTION.sub("", original).strip(" ?.,!")
        intent_l = intent.lower()
        # Strip leftover "what about" / "how about" / "tell me about"
        intent = re.sub(
            r"^(what about|how about|tell me about|and|also)\s*",
            "",
            intent,
            flags=re.I,
        ).strip(" ?.,!")

        if not intent or intent_l in {
            "what about", "how about", "and", "also", "the", "one", "product",
        }:
            retrieval_query = (
                f"What are the details of {resolved} "
                f"(specifications, capacity, price, launch year, warranty)?"
            )
        else:
            retrieval_query = f"What is {resolved}? {intent}".strip()

        logger.info(
            "Ordinal rewrite (deterministic): %r → %r (entity=%r)",
            original,
            retrieval_query,
            resolved,
        )
        return {
            "original_question": original,
            "retrieval_query": retrieval_query,
            "was_rewritten": True,
            "resolved_entity": resolved,
        }

    try:
        rewritten = _llm_rewrite(original, history, resolved)
        if not rewritten or len(rewritten) < 3:
            rewritten = original
            was_rewritten = False
        else:
            # If we know the entity, force it into the query when the LLM omitted it
            if resolved and resolved.lower() not in rewritten.lower():
                rewritten = f"{rewritten.rstrip(' ?')} regarding {resolved}?"
            was_rewritten = rewritten.lower() != original.lower()

        if was_rewritten:
            logger.info("Query rewrite: %r → %r", original, rewritten)

        return {
            "original_question": original,
            "retrieval_query": rewritten,
            "was_rewritten": was_rewritten,
            "resolved_entity": resolved,
        }
    except Exception:
        logger.exception("Query rewrite failed; using original question")
        if resolved:
            fallback = (
                f"What are the details of {resolved} "
                f"(specifications, capacity, price, launch year, warranty)?"
            )
            return {
                "original_question": original,
                "retrieval_query": fallback,
                "was_rewritten": True,
                "resolved_entity": resolved,
            }
        return {
            "original_question": original,
            "retrieval_query": original,
            "was_rewritten": False,
            "resolved_entity": None,
        }
