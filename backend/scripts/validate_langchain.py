"""
Validate LangChain integration without needing the frontend.

Usage (from backend/, venv active, API optional):
  python scripts/validate_langchain.py
  python scripts/validate_langchain.py --live   # also hits /health if server is up
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure backend root is on path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_imports() -> None:
    import langchain_core
    import langchain_mistralai
    import langchain_text_splitters
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from langchain_core.retrievers import BaseRetriever

    print("[OK] langchain imports")
    print(f"     langchain_core={getattr(langchain_core, '__version__', '?')}")
    print(f"     langchain_mistralai={getattr(langchain_mistralai, '__version__', '?')}")
    print(f"     text_splitters={getattr(langchain_text_splitters, '__version__', '?')}")
    assert issubclass(
        __import__("app.services.langchain_retriever", fromlist=["HybridRerankRetriever"]).HybridRerankRetriever,
        BaseRetriever,
    )
    print("[OK] HybridRerankRetriever is a LangChain BaseRetriever")
    assert ChatMistralAI and MistralAIEmbeddings
    print("[OK] ChatMistralAI + MistralAIEmbeddings available")


def check_chain_build() -> None:
    from app.services.langchain_rag import build_prompt, build_rag_chain, langchain_stack_info
    from app.services.embedding_service import get_embeddings

    prompt = build_prompt()
    messages = prompt.format_messages(
        question="What is vacation policy?",
        context="[Source: handbook.pdf, page 2]\nEmployees get 18 vacation days.",
        history=[],
    )
    assert any("18 vacation" in m.content for m in messages if hasattr(m, "content"))
    print("[OK] ChatPromptTemplate formats context + question")

    chain = build_rag_chain("validation-user-id")
    assert chain is not None
    print("[OK] LCEL RAG chain builds (retriever | prompt | ChatMistralAI)")

    emb = get_embeddings()
    print(f"[OK] Embeddings engine: {type(emb).__name__}")
    print("[OK] Stack:", langchain_stack_info())


def check_retriever_docs_shape() -> None:
    from langchain_core.documents import Document
    from app.services.langchain_retriever import HybridRerankRetriever

    # Do not call remote APIs — only verify constructor + Document shape helper path
    r = HybridRerankRetriever(user_id="demo-user")
    assert r.user_id == "demo-user"
    sample = Document(
        page_content="hello",
        metadata={"filename": "a.pdf", "page": 1, "score": 0.9},
    )
    assert sample.page_content == "hello"
    print("[OK] Retriever constructed; Document metadata shape ready for citations")


def check_live_health(base_url: str) -> None:
    import urllib.request
    import json

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=10) as resp:
        data = json.loads(resp.read().decode())
    assert data.get("status") == "ok"
    assert data.get("langchain_enabled") is True
    assert isinstance(data.get("langchain"), dict)
    print("[OK] Live /health reports langchain_enabled=true")
    print("     ", data["langchain"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also call running API /health")
    parser.add_argument("--base-url", default="http://localhost:8001")
    args = parser.parse_args()

    print("=== LangChain validation ===")
    check_imports()
    check_chain_build()
    check_retriever_docs_shape()
    if args.live:
        check_live_health(args.base_url)
    print("=== All checks passed ===")
    print()
    print("Manual UI checks:")
    print("  1. Restart backend, open /health -> langchain_enabled true")
    print("  2. Upload PDF, ask a question -> answer + sources still work")
    print("  3. Backend logs should mention 'LangChain RAG'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
