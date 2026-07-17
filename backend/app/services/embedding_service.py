"""
Embeddings via LangChain MistralAIEmbeddings (batched).

Keeps the same mistral-embed model so existing Chroma vectors stay compatible.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_mistralai import MistralAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings() -> MistralAIEmbeddings:
    settings = get_settings()
    return MistralAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.mistral_api_key,
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts in batches via LangChain → Mistral.
    """
    if not texts:
        return []

    settings = get_settings()
    embeddings = get_embeddings()
    all_vectors: List[List[float]] = []
    batch_size = settings.embed_batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        logger.info(
            "LangChain embed batch %s-%s / %s",
            start,
            start + len(batch),
            len(texts),
        )
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)

    return all_vectors


def embed_query(text: str) -> List[float]:
    return get_embeddings().embed_query(text)
