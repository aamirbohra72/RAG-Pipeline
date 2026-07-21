from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class Source(BaseModel):
    filename: str
    page: int
    snippet: str
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Optional short chat history for multi-turn (senior RAG pattern)
    history: List[dict] = Field(default_factory=list)
    # [{"role": "user"|"assistant", "content": "..."}]


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    doc_type: Optional[str] = None
    date_after: Optional[str] = None  # ISO-8601 date or datetime


class SearchResultItem(BaseModel):
    content: str
    source: str
    doc_id: str
    page: int
    score: float
    rerank_score: Optional[float] = None
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    citation_metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class UploadItem(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    text_pages: int = 0
    ocr_pages: int = 0


class UploadResponse(BaseModel):
    uploaded: List[UploadItem]


class JobRef(BaseModel):
    job_id: str
    filename: str


class AsyncUploadResponse(BaseModel):
    jobs: List[JobRef]


class JobStatus(BaseModel):
    job_id: str
    filename: Optional[str] = None
    state: str  # PENDING | PROGRESS | SUCCESS | FAILURE
    stage: Optional[str] = None
    result: Optional[UploadItem] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chroma_chunks: int  # indexed chunk count (legacy field name)
    vector_backend: str = "pgvector"
    embedding_model: str
    chat_model: str
    rerank_enabled: bool = True
    ocr_enabled: bool = True
    langchain_enabled: bool = True
    langchain: Optional[dict] = None
    neon_ok: Optional[bool] = None
    langsmith_tracing: Optional[bool] = None
    langgraph_enabled: bool = True
    async_ingest: bool = False
    redis_ok: Optional[bool] = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    email: EmailStr


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
