"""Pydantic input/output schemas for MCP tools."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum passages to return.")
    doc_type: Optional[str] = Field(
        default=None,
        description="Filter by file extension (e.g. 'pdf').",
    )
    date_after: Optional[str] = Field(
        default=None,
        description="ISO-8601 date — only documents ingested on/after this date.",
    )


class AskQuestionInput(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the RAG pipeline.")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Reuse prior turns from an earlier ask_question call.",
    )


class ListDocumentsInput(BaseModel):
    doc_type: Optional[str] = Field(default=None, description="Filter by file extension.")
    limit: int = Field(default=20, ge=1, le=200, description="Max documents to return.")


class GetIngestionStatusInput(BaseModel):
    job_id: str = Field(..., min_length=1, description="Celery job ID from async upload.")


class Citation(BaseModel):
    doc_id: str = ""
    source: str
    excerpt: str


class AskQuestionOutput(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    conversation_id: str


class SearchHit(BaseModel):
    content: str
    source: str
    doc_id: str
    score: Optional[float] = None
    citation_metadata: dict[str, Any] = Field(default_factory=dict)
