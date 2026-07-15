"""
Uvicorn entrypoint: uvicorn main:app --reload --port 8000

Senior architecture lives under app/:
  app/config.py          — settings
  app/dependencies.py    — Mistral + Chroma clients
  app/routers/           — HTTP endpoints
  app/services/          — PDF, chunk, embed, vectorstore, hybrid retrieve, RAG
"""

from app.main import app

__all__ = ["app"]
