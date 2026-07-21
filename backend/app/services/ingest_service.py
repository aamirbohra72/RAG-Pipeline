"""
Unified ingest pipeline for all supported document types.

PDF: pypdf + RapidOCR fallback
Markdown / HTML / CSV / XLSX: document_loaders → row/section chunking
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.services.chunking_service import chunk_pages, chunk_text_blocks
from app.services.document_loaders import load_document, supported_extensions
from app.services.pdf_service import extract_pages

logger = logging.getLogger(__name__)


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in supported_extensions()


def ingest_bytes(user_id: str, filename: str, file_bytes: bytes, vectorstore) -> dict:
    """
    Extract → chunk → embed → store. Returns UploadItem-shaped dict.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        extraction = extract_pages(file_bytes)
        chunks = chunk_pages(extraction.pages, doc_type="pdf")
        result = vectorstore.add_document(user_id, filename, chunks)
        return {
            **result,
            "text_pages": extraction.text_pages,
            "ocr_pages": extraction.ocr_pages,
        }

    loaded = load_document(file_bytes, filename)
    chunks = chunk_text_blocks(loaded.blocks, doc_type=loaded.doc_type)
    if not chunks:
        raise ValueError(f"No extractable text chunks for {filename}")

    result = vectorstore.add_document(user_id, filename, chunks)
    logger.info(
        "Indexed %s (%s): %s blocks → %s chunks",
        filename,
        loaded.doc_type,
        len(loaded.blocks),
        result["chunks"],
    )
    return {
        **result,
        "text_pages": len(loaded.blocks),
        "ocr_pages": 0,
        "doc_type": loaded.doc_type,
    }
