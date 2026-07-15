"""
PDF text extraction with OCR fallback for scanned / image-only pages.

Strategy:
  1. Try native text layer (pypdf) — fast, accurate for digital PDFs
  2. If a page has almost no text, render it with PyMuPDF and run RapidOCR
     (ONNX — no system Tesseract install required on Windows)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

from pypdf import PdfReader

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    pages: List[Tuple[int, str]] = field(default_factory=list)
    text_pages: int = 0
    ocr_pages: int = 0


@lru_cache
def _get_ocr():
    from rapidocr_onnxruntime import RapidOCR

    logger.info("Loading RapidOCR engine (first load may download ONNX models)")
    return RapidOCR()


def _ocr_page_image(png_bytes: bytes) -> str:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    result, _ = _get_ocr()(np.array(image))
    if not result:
        return ""
    # RapidOCR lines: [box, text, confidence]
    return "\n".join(line[1] for line in result if line and len(line) > 1).strip()


def _ocr_pdf_pages(file_bytes: bytes, page_numbers: List[int]) -> dict[int, str]:
    """OCR selected 1-indexed page numbers. Returns {page_num: text}."""
    import fitz

    settings = get_settings()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    out: dict[int, str] = {}
    zoom = settings.ocr_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    wanted = set(page_numbers)
    for idx in range(len(doc)):
        page_num = idx + 1
        if page_num not in wanted:
            continue
        pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
        text = _ocr_page_image(pix.tobytes("png"))
        out[page_num] = text
        logger.info("OCR page %s → %s chars", page_num, len(text))

    doc.close()
    return out


def extract_pages(file_bytes: bytes) -> ExtractionResult:
    """
    Extract (page_number, text) for every page that yields usable text.
    Uses OCR automatically when the text layer is empty/sparse.
    """
    settings = get_settings()
    reader = PdfReader(io.BytesIO(file_bytes))
    n_pages = len(reader.pages)

    digital: dict[int, str] = {}
    needs_ocr: List[int] = []

    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) >= settings.ocr_min_chars:
            digital[idx] = text
        else:
            needs_ocr.append(idx)

    ocr_map: dict[int, str] = {}
    if needs_ocr and settings.ocr_enabled:
        logger.info(
            "Running OCR on %s/%s sparse/empty pages",
            len(needs_ocr),
            n_pages,
        )
        try:
            ocr_map = _ocr_pdf_pages(file_bytes, needs_ocr)
        except Exception:
            logger.exception("OCR failed; keeping digital text only")
    elif needs_ocr and not settings.ocr_enabled:
        logger.warning(
            "%s pages look scanned but OCR_ENABLED=false — those pages skipped",
            len(needs_ocr),
        )

    pages: List[Tuple[int, str]] = []
    text_pages = 0
    ocr_pages = 0

    for page_num in range(1, n_pages + 1):
        if page_num in digital:
            pages.append((page_num, digital[page_num]))
            text_pages += 1
        elif page_num in ocr_map and ocr_map[page_num].strip():
            pages.append((page_num, ocr_map[page_num].strip()))
            ocr_pages += 1

    return ExtractionResult(pages=pages, text_pages=text_pages, ocr_pages=ocr_pages)
