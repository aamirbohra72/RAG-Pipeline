from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse
from app.services import vectorstore
from app.services.langchain_rag import langchain_stack_info
from app.services.observability import observability_info

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    try:
        lc_info = langchain_stack_info()
    except Exception as exc:
        lc_info = {"error": str(exc)}

    obs = observability_info()

    neon_ok = None
    if settings.vector_backend == "pgvector":
        try:
            from app.services.pgvector_store import ping

            neon_ok = ping()
        except Exception:
            neon_ok = False

    redis_ok = None
    if settings.async_ingest and settings.redis_url:
        try:
            import ssl

            import redis

            kwargs = {}
            if settings.redis_url.startswith("rediss://"):
                kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            client = redis.from_url(settings.redis_url, socket_connect_timeout=5, **kwargs)
            redis_ok = bool(client.ping())
        except Exception:
            redis_ok = False

    return HealthResponse(
        status="ok" if neon_ok is not False else "degraded",
        chroma_chunks=vectorstore.chunk_count(),
        vector_backend=settings.vector_backend,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        rerank_enabled=settings.rerank_enabled,
        ocr_enabled=settings.ocr_enabled,
        langchain_enabled=True,
        langchain=lc_info,
        neon_ok=neon_ok,
        langsmith_tracing=obs.get("langsmith_tracing"),
        langgraph_enabled=True,
        async_ingest=bool(settings.async_ingest and settings.redis_url),
        redis_ok=redis_ok,
    )
