import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.schemas import UploadItem, UploadResponse
from app.services import vectorstore
from app.services.auth_service import User, get_current_user
from app.services.chunking_service import chunk_pages
from app.services.pdf_service import extract_pages

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


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
