"""
LangGraph RAG orchestration.

Graph:
  START → rewrite → retrieve → generate → END
           ↘ (no docs) → empty_answer → END

Query rewriting runs before retrieval (LangSmith: query_rewrite).
Each node is traced in LangSmith when tracing is enabled.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from app.schemas import Source
from app.services import langchain_rag as lc
from app.services.langchain_retriever import HybridRerankRetriever
from app.services.query_rewrite_service import rewrite_query_if_needed

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    retrieval_query: str
    query_rewritten: bool
    user_id: str
    history: list
    docs: List[Document]
    context: str
    chat_history: List[BaseMessage]
    answer: str


def _rewrite_node(state: RAGState) -> dict:
    question = state["question"]
    history = state.get("history") or []
    result = rewrite_query_if_needed(question, history)
    return {
        "retrieval_query": result["retrieval_query"],
        "query_rewritten": result["was_rewritten"],
    }


def _retrieve_node(state: RAGState) -> dict:
    retrieval_query = state.get("retrieval_query") or state["question"]
    user_id = state["user_id"]
    retriever = HybridRerankRetriever(user_id=user_id)
    docs = retriever.invoke(retrieval_query)
    history = lc._history_to_messages(state.get("history"))
    return {
        "docs": docs,
        "context": lc._format_docs(docs),
        "chat_history": history,
    }


def _generate_node(state: RAGState) -> dict:
    docs = state.get("docs") or []
    if not docs:
        return {"answer": "No documents have been uploaded yet."}

    prompt = lc.build_prompt()
    llm = lc.get_chat_llm(streaming=False)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": state["question"],
            "context": state.get("context") or "",
            "history": state.get("chat_history") or [],
        }
    )
    return {"answer": answer}


def _route_after_retrieve(state: RAGState) -> str:
    docs = state.get("docs") or []
    return "generate" if docs else "empty"


def _empty_node(state: RAGState) -> dict:
    return {"answer": "No documents have been uploaded yet."}


def build_rag_graph():
    """Compile the LangGraph StateGraph used for sync answers."""
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("empty", _empty_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "empty": "empty"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("empty", END)
    return graph.compile()


_graph = None


def get_rag_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
        logger.info("LangGraph RAG compiled (rewrite → retrieve → generate | empty)")
    return _graph


def answer_with_langgraph(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[str, List[Source], dict]:
    graph = get_rag_graph()
    logger.info("LangGraph RAG invoke user=%s", user_id)
    result = graph.invoke(
        {
            "question": question,
            "user_id": user_id,
            "history": history or [],
        }
    )
    docs = result.get("docs") or []
    answer = result.get("answer") or ""
    meta = {
        "retrieval_query": result.get("retrieval_query") or question,
        "query_rewritten": bool(result.get("query_rewritten")),
    }
    return answer, lc._sources_from_docs(docs, question=question), meta


def _run_rewrite_and_retrieve(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> dict:
    """Shared rewrite + retrieve path for sync and streaming."""
    state: RAGState = {
        "question": question,
        "user_id": user_id,
        "history": history or [],
    }
    rewritten = _rewrite_node(state)
    state.update(rewritten)
    retrieved = _retrieve_node(state)
    state.update(retrieved)
    return state


def stream_with_langgraph(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[List[Source], Iterator[str], dict]:
    """
    Rewrite + retrieve (traced), then stream ChatMistralAI tokens.
    """
    state = _run_rewrite_and_retrieve(question, user_id, history)
    docs = state.get("docs") or []
    meta = {
        "retrieval_query": state.get("retrieval_query") or question,
        "query_rewritten": bool(state.get("query_rewritten")),
    }

    if not docs:
        def empty() -> Iterator[str]:
            yield "No documents have been uploaded yet."

        return [], empty(), meta

    sources = lc._sources_from_docs(docs, question=state.get("retrieval_query") or question)
    prompt = lc.build_prompt()
    llm = lc.get_chat_llm(streaming=True)
    messages = prompt.format_messages(
        question=question,
        context=state.get("context") or "",
        history=state.get("chat_history") or [],
    )

    logger.info("LangGraph RAG stream user=%s docs=%s", user_id, len(docs))

    def token_iter() -> Iterator[str]:
        for chunk in llm.stream(messages):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if text:
                            yield text
                    elif isinstance(block, str):
                        yield block

    return sources, token_iter(), meta


def langgraph_info() -> dict:
    try:
        import langgraph

        version = getattr(langgraph, "__version__", "unknown")
    except Exception:
        version = "unavailable"
    return {
        "langgraph_version": version,
        "graph": "rewrite → retrieve → generate | empty",
        "state": "RAGState",
    }
