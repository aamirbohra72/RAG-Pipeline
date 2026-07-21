from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    page: int
    doc_type: str = "pdf"
    section: str = ""


def get_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    doc_type: str = "pdf",
    section_prefix: str = "",
) -> List[Chunk]:
    """Split each page independently so page citations stay accurate."""
    splitter = get_splitter()
    chunks: List[Chunk] = []
    for page_num, page_text in pages:
        section = section_prefix or f"page {page_num}"
        for piece in splitter.split_text(page_text):
            cleaned = piece.strip()
            if cleaned:
                chunks.append(
                    Chunk(
                        text=cleaned,
                        page=page_num,
                        doc_type=doc_type,
                        section=section,
                    )
                )
    return chunks


def chunk_text_blocks(
    blocks: list[tuple[int, str, str]],
    *,
    doc_type: str,
) -> List[Chunk]:
    """
    Chunk structured blocks: (page_num, section_name, text).
    Used by Markdown/HTML/CSV loaders.
    """
    splitter = get_splitter()
    chunks: List[Chunk] = []
    for page_num, section, text in blocks:
        for piece in splitter.split_text(text):
            cleaned = piece.strip()
            if cleaned:
                chunks.append(
                    Chunk(
                        text=cleaned,
                        page=page_num,
                        doc_type=doc_type,
                        section=section,
                    )
                )
    return chunks
