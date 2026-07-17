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
