import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import (
    AsyncUploadResponse,
    JobRef,
    UploadItem,
    UploadResponse,
)
from app.services import vectorstore
from app.services.auth_service import User, get_current_user
from app.services.document_loaders import supported_extensions
from app.services.ingest_service import ingest_bytes, is_supported

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


def _save_upload(file_bytes: bytes, filename: str) -> Path:
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower() or ".bin"
    path = upload_dir / f"{uuid.uuid4()}{ext}"
    path.write_bytes(file_bytes)
    return path


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
):
    if not files:
        raise HTTPException(400, "At least one file is required")

    uploaded: list[UploadItem] = []

    for file in files:
        filename = file.filename or "unknown"
        if not is_supported(filename):
            allowed = ", ".join(sorted(supported_extensions()))
            raise HTTPException(400, f"{filename} is not supported. Allowed: {allowed}")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(400, f"{filename} is empty")

        try:
            result = ingest_bytes(user.id, filename, file_bytes, vectorstore)
            uploaded.append(UploadItem(**result))
            logger.info(
                "Upload %s: text_pages=%s ocr_pages=%s chunks=%s",
                filename,
                result.get("text_pages", 0),
                result.get("ocr_pages", 0),
                result["chunks"],
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to index %s for user %s", filename, user.id)
            raise HTTPException(500, f"Failed to index {filename}: {exc}") from exc

    return UploadResponse(uploaded=uploaded)


@router.post("/upload/async", response_model=AsyncUploadResponse)
async def upload_documents_async(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
):
    """Enqueue documents for background processing via Celery + Upstash Redis."""
    settings = get_settings()
    if not settings.async_ingest or not settings.redis_url:
        raise HTTPException(503, "Async ingest is disabled. Set REDIS_URL and ASYNC_INGEST=true.")

    if not files:
        raise HTTPException(400, "At least one file is required")

    from app.tasks.ingest_tasks import process_document

    jobs: list[JobRef] = []
    for file in files:
        filename = file.filename or "unknown"
        if not is_supported(filename):
            allowed = ", ".join(sorted(supported_extensions()))
            raise HTTPException(400, f"{filename} is not supported. Allowed: {allowed}")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(400, f"{filename} is empty")

        path = _save_upload(file_bytes, filename)
        task = process_document.delay(user.id, filename, str(path))
        jobs.append(JobRef(job_id=task.id, filename=filename))
        logger.info("Queued ingest job %s for %s (user %s)", task.id, filename, user.id)

    return AsyncUploadResponse(jobs=jobs)
