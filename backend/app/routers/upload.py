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
from app.services.chunking_service import chunk_pages
from app.services.pdf_service import extract_pages

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


def _save_upload(file_bytes: bytes) -> Path:
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4()}.pdf"
    path.write_bytes(file_bytes)
    return path


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
):
    if not files:
        raise HTTPException(400, "At least one PDF is required")

    uploaded: list[UploadItem] = []

    for file in files:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{filename} is not a PDF")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(400, f"{filename} is empty")

        try:
            extraction = extract_pages(file_bytes)
            chunks = chunk_pages(extraction.pages)
            result = vectorstore.add_document(user.id, filename, chunks)
            uploaded.append(
                UploadItem(
                    **result,
                    text_pages=extraction.text_pages,
                    ocr_pages=extraction.ocr_pages,
                )
            )
            logger.info(
                "Upload %s: text_pages=%s ocr_pages=%s chunks=%s",
                filename,
                extraction.text_pages,
                extraction.ocr_pages,
                result["chunks"],
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to index %s for user %s", filename, user.id)
            raise HTTPException(500, f"Failed to index {filename}: {exc}") from exc

    return UploadResponse(uploaded=uploaded)


@router.post("/upload/async", response_model=AsyncUploadResponse)
async def upload_pdfs_async(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
):
    """Enqueue PDFs for background processing via Celery + Upstash Redis.

    Returns immediately with a job id per file; poll GET /jobs/{job_id}.
    """
    settings = get_settings()
    if not settings.async_ingest or not settings.redis_url:
        raise HTTPException(503, "Async ingest is disabled. Set REDIS_URL and ASYNC_INGEST=true.")

    if not files:
        raise HTTPException(400, "At least one PDF is required")

    # Import lazily so the API still boots if Celery/redis aren't installed
    from app.tasks.ingest_tasks import process_pdf

    jobs: list[JobRef] = []
    for file in files:
        filename = file.filename or "unknown.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"{filename} is not a PDF")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(400, f"{filename} is empty")

        path = _save_upload(file_bytes)
        task = process_pdf.delay(user.id, filename, str(path))
        jobs.append(JobRef(job_id=task.id, filename=filename))
        logger.info("Queued ingest job %s for %s (user %s)", task.id, filename, user.id)

    return AsyncUploadResponse(jobs=jobs)
