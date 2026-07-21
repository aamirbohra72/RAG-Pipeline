"""
LangChain custom retriever.

Wraps our hybrid vector+lexical retrieval and cross-encoder re-ranker
as a standard LangChain BaseRetriever so LCEL chains can use it.
"""

from __future__ import annotations

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.services.retrieval import retrieve


class HybridRerankRetriever(BaseRetriever):
    """
    User-scoped retriever:
      Chroma (filtered by user_id) → hybrid fusion → cross-encoder re-rank → Documents
    """

    user_id: str = Field(description="Authenticated user id for document isolation")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        chunks = retrieve(query, self.user_id)
        docs: List[Document] = []
        for c in chunks:
            docs.append(
                Document(
                    page_content=c.text,
                    metadata={
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "page": c.page,
                        "score": c.score,
                        "rerank_score": c.rerank_score,
                        "vector_score": c.vector_score,
                        "lexical_score": c.lexical_score,
                        "user_id": self.user_id,
                    },
                )
            )
        return docs
