"""
LangGraph RAG orchestration.

Graph:
  START → retrieve → generate → END
           ↘ (no docs) → empty_answer → END

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

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    question: str
    user_id: str
    history: list
    docs: List[Document]
    context: str
    chat_history: List[BaseMessage]
    answer: str


def _retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    user_id = state["user_id"]
    retriever = HybridRerankRetriever(user_id=user_id)
    docs = retriever.invoke(question)
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
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("empty", _empty_node)

    graph.add_edge(START, "retrieve")
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
        logger.info("LangGraph RAG compiled (retrieve → generate | empty)")
    return _graph


def answer_with_langgraph(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[str, List[Source]]:
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
    return answer, lc._sources_from_docs(docs)


def stream_with_langgraph(
    question: str,
    user_id: str,
    history: Optional[List[dict]] = None,
) -> tuple[List[Source], Iterator[str]]:
    """
    Retrieve via LangGraph retrieve node (traced), then stream ChatMistralAI tokens
    (also traced by LangSmith).
    """
    # Run only retrieve path by invoking full graph would also generate —
    # so call retrieve node logic directly (same code, still LangChain-traced retriever).
    state: RAGState = {
        "question": question,
        "user_id": user_id,
        "history": history or [],
    }
    retrieved = _retrieve_node(state)
    docs = retrieved.get("docs") or []
    if not docs:
        def empty() -> Iterator[str]:
            yield "No documents have been uploaded yet."

        return [], empty()

    sources = lc._sources_from_docs(docs)
    prompt = lc.build_prompt()
    llm = lc.get_chat_llm(streaming=True)
    messages = prompt.format_messages(
        question=question,
        context=retrieved.get("context") or "",
        history=retrieved.get("chat_history") or [],
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

    return sources, token_iter()


def langgraph_info() -> dict:
    try:
        import langgraph

        version = getattr(langgraph, "__version__", "unknown")
    except Exception:
        version = "unavailable"
    return {
        "langgraph_version": version,
        "graph": "retrieve -> generate | empty",
        "state": "RAGState",
    }
