"""
Background ingest pipeline: extract (OCR fallback) -> chunk -> embed -> store.

The web process saves the uploaded PDF to disk and enqueues this task with the
file path. The worker reads the file, runs the full pipeline, updates progress
via Celery state, and deletes the temp file when done.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.celery_app import celery
from app.services import vectorstore
from app.services.chunking_service import chunk_pages
from app.services.pdf_service import extract_pages

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="ingest.process_pdf")
def process_pdf(self, user_id: str, filename: str, file_path: str) -> dict:
    """Run the ingest pipeline for a single PDF and return an UploadItem-shaped dict."""
    path = Path(file_path)

    def progress(stage: str, **extra) -> None:
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "filename": filename, **extra},
        )

    try:
        progress("reading")
        file_bytes = path.read_bytes()
        if not file_bytes:
            raise ValueError(f"{filename} is empty")

        progress("extracting")
        extraction = extract_pages(file_bytes)

        progress("chunking", text_pages=extraction.text_pages, ocr_pages=extraction.ocr_pages)
        chunks = chunk_pages(extraction.pages)

        progress("embedding", chunks=len(chunks))
        result = vectorstore.add_document(user_id, filename, chunks)

        logger.info(
            "Async ingest %s: text_pages=%s ocr_pages=%s chunks=%s",
            filename,
            extraction.text_pages,
            extraction.ocr_pages,
            result["chunks"],
        )
        return {
            **result,
            "filename": filename,
            "text_pages": extraction.text_pages,
            "ocr_pages": extraction.ocr_pages,
        }
    finally:
        # Always clean up the temp upload, even on failure
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete temp upload %s", file_path)
