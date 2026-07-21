import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import auth, documents, health, jobs, query, search, upload
from app.services.auth_service import init_db
from app.services.observability import configure_observability
from app.services.vectorstore import init_vector_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_observability()
    init_db()
    init_vector_backend()

    application = FastAPI(
        title="RAG Backend (Senior)",
        description=(
            "RAG with JWT auth, LangGraph orchestration, LangSmith tracing, "
            "Neon pgvector, hybrid retrieve + re-rank, OCR, streaming."
        ),
        version="2.5.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s → %s (%.1fms) id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    application.include_router(health.router)
    application.include_router(auth.router)
    application.include_router(upload.router)
    application.include_router(jobs.router)
    application.include_router(query.router)
    application.include_router(search.router)
    application.include_router(documents.router)

    return application


app = create_app()
