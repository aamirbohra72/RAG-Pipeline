"""
LangChain LCEL RAG chain.

Pipeline:
  question → HybridRerankRetriever → format context → ChatPromptTemplate
           → ChatMistralAI → string answer

Streaming uses llm.stream() on the same prompt path.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_mistralai import ChatMistralAI

from app.config import get_settings
from app.schemas import Source
from app.services.langchain_retriever import HybridRerankRetriever

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a careful document assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources inline like [filename, page N] when you use them."
)


@lru_cache
def get_chat_llm(streaming: bool = False) -> ChatMistralAI:
    settings = get_settings()
    return ChatMistralAI(
        model=settings.chat_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0.2,
        max_retries=2,
        streaming=streaming,
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        score = meta.get("score", "")
        parts.append(
            f"[Chunk {i} | Source: {filename}, page {page} | score={score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _history_to_messages(history: Optional[List[dict]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _sources_from_docs(docs: List[Document]) -> List[Source]:
    sources: List[Source] = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            Source(
                filename=str(meta.get("filename", "unknown")),
                page=int(meta.get("page", 0) or 0),
                snippet=doc.page_content[:220],
                score=meta.get("score"),
                rerank_score=meta.get("rerank_score"),
                vector_score=meta.get("vector_score"),
                lexical_score=meta.get("lexical_score"),
            )
        )
    return sources


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTION),
            MessagesPlaceholder("history", optional=True),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )


def build_rag_chain(user_id: str, *, streaming: bool = False) -> Runnable:
    """
    End-to-end LCEL chain.

    Input:  {"question": str, "history": list[dict] (optional)}
    Output: answer string

    Used by validation + for a single-call invoke path.
    """
    retriever = HybridRerankRetriever(user_id=user_id)
    prompt = build_prompt()
    llm = get_chat_llm(streaming=streaming)

    def prepare(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        return {
            "question": question,
            "history": _history_to_messages(inputs.get("history")),
            "context": _format_docs(docs),
        }

    return RunnableLambda(prepare) | prompt | llm | StrOutputParser()


def retrieve_docs(question: str, user_id: str) -> List[Document]:
    return HybridRerankRetriever(user_id=user_id).invoke(question)


def answer_with_langchain(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[str, List[Source]]:
    docs = retrieve_docs(question, user_id)
    if not docs:
        return "No documents have been uploaded yet.", []

    prompt = build_prompt()
    llm = get_chat_llm(streaming=False)
    chain = prompt | llm | StrOutputParser()

    logger.info(
        "LangChain RAG generate: %s docs user=%s engine=ChatMistralAI",
        len(docs),
        user_id,
    )
    answer = chain.invoke(
        {
            "question": question,
            "context": _format_docs(docs),
            "history": _history_to_messages(history),
        }
    )
    return answer, _sources_from_docs(docs)


def stream_with_langchain(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[List[Source], Iterator[str]]:
    docs = retrieve_docs(question, user_id)
    if not docs:
        def empty() -> Iterator[str]:
            yield "No documents have been uploaded yet."

        return [], empty()

    prompt = build_prompt()
    llm = get_chat_llm(streaming=True)
    messages = prompt.format_messages(
        question=question,
        context=_format_docs(docs),
        history=_history_to_messages(history),
    )
    sources = _sources_from_docs(docs)

    logger.info(
        "LangChain RAG stream: %s docs user=%s engine=ChatMistralAI",
        len(docs),
        user_id,
    )

    def token_iter() -> Iterator[str]:
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                # Some versions return content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if text:
                            yield text
                    elif isinstance(block, str):
                        yield block

    return sources, token_iter()


def langchain_stack_info() -> dict:
    import langchain_core
    import langchain_mistralai
    import langchain_text_splitters

    from app.services.langgraph_rag import langgraph_info
    from app.services.observability import observability_info

    info = {
        "langchain_core": getattr(langchain_core, "__version__", "unknown"),
        "langchain_mistralai": getattr(langchain_mistralai, "__version__", "unknown"),
        "langchain_text_splitters": getattr(
            langchain_text_splitters, "__version__", "unknown"
        ),
        "retriever": "HybridRerankRetriever",
        "llm": "ChatMistralAI",
        "embeddings": "MistralAIEmbeddings",
        "orchestration": "langgraph",
    }
    info.update(langgraph_info())
    info.update(observability_info())
    return info

