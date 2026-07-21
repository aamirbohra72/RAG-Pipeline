"""Application settings loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mistral_api_key: str

    # Auth (generate a long random JWT_SECRET for anything beyond local demo)
    jwt_secret: str = "dev-only-change-me-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days
    users_db_path: str = "./users.db"

    embedding_model: str = "mistral-embed"
    chat_model: str = "mistral-large-latest"

    # Chunking (~500 tokens / ~50 token overlap via char approximation)
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Multi-turn chat: load last N messages from Postgres before each query
    chat_history_turns: int = 6

    # Query rewrite for follow-ups (small/fast model, traced as query_rewrite)
    rewrite_enabled: bool = True
    rewrite_model: str = "mistral-small-latest"

    # Retrieval: pull a wider vector candidate pool, then hybrid + cross-encoder
    top_k: int = 4
    candidate_pool: int = 40
    hybrid_vector_weight: float = 0.65
    hybrid_lexical_weight: float = 0.35
    # Drop chunks below this sigmoid(relevance) before generation (blocks 0% noise)
    min_relevance_score: float = 0.2
    # Prefer diverse source docs in the final top_k (helps when one XLSX floods the pool)
    max_chunks_per_doc: int = 2

    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OCR for scanned PDFs (RapidOCR — no system Tesseract needed)
    ocr_enabled: bool = True
    ocr_min_chars: int = 40  # pages with fewer chars trigger OCR
    ocr_dpi: int = 200

    embed_batch_size: int = 32
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "documents"

    # Vector backend: "pgvector" (Neon) or "chroma" (local)
    vector_backend: str = "pgvector"
    database_url: str | None = None

    # LangSmith monitoring (optional — get key at https://smith.langchain.com)
    langsmith_tracing: bool = True
    langsmith_api_key: str | None = None
    langsmith_project: str = "genai-rag"

    # Async ingest via Celery + Redis (Upstash). rediss:// = TLS.
    redis_url: str | None = None
    async_ingest: bool = True
    upload_dir: str = "./uploads"
    celery_task_time_limit: int = 900  # hard limit per PDF (seconds)

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
