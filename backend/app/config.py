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

    # Retrieval: pull a wider vector candidate pool, then hybrid + cross-encoder
    top_k: int = 4
    candidate_pool: int = 20
    hybrid_vector_weight: float = 0.65
    hybrid_lexical_weight: float = 0.35

    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OCR for scanned PDFs (RapidOCR — no system Tesseract needed)
    ocr_enabled: bool = True
    ocr_min_chars: int = 40  # pages with fewer chars trigger OCR
    ocr_dpi: int = 200

    embed_batch_size: int = 32
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "documents"

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
