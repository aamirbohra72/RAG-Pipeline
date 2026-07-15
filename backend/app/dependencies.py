"""Shared infrastructure clients (Mistral + Chroma)."""

from functools import lru_cache

import chromadb
from mistralai.client import Mistral

from app.config import get_settings


@lru_cache
def get_mistral_client() -> Mistral:
    settings = get_settings()
    return Mistral(api_key=settings.mistral_api_key)


@lru_cache
def get_chroma_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
