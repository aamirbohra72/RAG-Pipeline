from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    page: int


def get_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(pages: list[tuple[int, str]]) -> List[Chunk]:
    """Split each page independently so page citations stay accurate."""
    splitter = get_splitter()
    chunks: List[Chunk] = []
    for page_num, page_text in pages:
        for piece in splitter.split_text(page_text):
            cleaned = piece.strip()
            if cleaned:
                chunks.append(Chunk(text=cleaned, page=page_num))
    return chunks
