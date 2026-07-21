"""
Background ingest pipeline: extract → chunk → embed → store.

Supports PDF, Markdown, HTML, CSV, and XLSX via ingest_service.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.celery_app import celery
from app.services import vectorstore
from app.services.ingest_service import ingest_bytes

logger = logging.getLogger(__name__)


def _run_ingest(task, user_id: str, filename: str, file_path: str) -> dict:
    path = Path(file_path)

    def progress(stage: str, **extra) -> None:
        task.update_state(
            state="PROGRESS",
            meta={"stage": stage, "filename": filename, **extra},
        )

    try:
        progress("reading")
        file_bytes = path.read_bytes()
        if not file_bytes:
            raise ValueError(f"{filename} is empty")

        progress("extracting")
        progress("chunking")
        progress("embedding")
        result = ingest_bytes(user_id, filename, file_bytes, vectorstore)

        logger.info(
            "Async ingest %s: text_pages=%s ocr_pages=%s chunks=%s",
            filename,
            result.get("text_pages", 0),
            result.get("ocr_pages", 0),
            result["chunks"],
        )
        return {**result, "filename": filename}
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete temp upload %s", file_path)


@celery.task(bind=True, name="ingest.process_document")
def process_document(self, user_id: str, filename: str, file_path: str) -> dict:
    return _run_ingest(self, user_id, filename, file_path)


@celery.task(bind=True, name="ingest.process_pdf")
def process_pdf(self, user_id: str, filename: str, file_path: str) -> dict:
    """Backward-compatible alias for existing workers."""
    return _run_ingest(self, user_id, filename, file_path)
