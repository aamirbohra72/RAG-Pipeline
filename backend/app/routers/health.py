from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse
from app.services import vectorstore

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        chroma_chunks=vectorstore.chunk_count(),
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        rerank_enabled=settings.rerank_enabled,
        ocr_enabled=settings.ocr_enabled,
    )
