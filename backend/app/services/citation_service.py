"""
Citation span extraction for frontend highlighting.

Finds the most relevant substring within a retrieved chunk using sentence-level
lexical overlap with the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class HighlightSpan:
    start: int
    end: int
    text: str


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


def _overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def find_highlight_span(question: str, chunk_text: str) -> HighlightSpan:
    """
    Return character offsets for the best supporting excerpt within chunk_text.
    Falls back to the full chunk when no sentence scores above zero.
    """
    text = chunk_text.strip()
    if not text:
        return HighlightSpan(start=0, end=0, text="")

    q_tokens = _tokenize(question)
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return HighlightSpan(start=0, end=len(text), text=text)

    best_span: HighlightSpan | None = None
    best_score = -1.0
    search_from = 0

    for sentence in sentences:
        score = _overlap_score(q_tokens, sentence)
        idx = text.find(sentence, search_from)
        if idx < 0:
            idx = text.find(sentence)
        if idx < 0:
            continue
        start, end = idx, idx + len(sentence)
        if score > best_score:
            best_score = score
            best_span = HighlightSpan(start=start, end=end, text=sentence)
        search_from = max(search_from, end)

    if best_span is None or best_score <= 0:
        excerpt = text[:500] if len(text) > 500 else text
        return HighlightSpan(start=0, end=len(excerpt), text=excerpt)

    return best_span


def normalize_relevance_score(rerank_score: float | None, hybrid_score: float | None) -> float:
    """Map re-ranker / hybrid score to a 0–1 relevance confidence for the UI."""
    raw = rerank_score if rerank_score is not None else hybrid_score
    if raw is None:
        return 0.0
    # Cross-encoder scores are unbounded; sigmoid gives a stable 0–1 display value
    import math

    return round(1.0 / (1.0 + math.exp(-raw)), 4)
