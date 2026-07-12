"""
backend/document_loader.py
===========================
Loads raw documents from the `documents/` knowledge base (PDF, DOCX, TXT, CSV),
extracts text (with page numbers where applicable), and splits them into
overlapping chunks suitable for embedding.

Pipeline stage: Documents -> Chunking
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document as DocxDocument

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance metadata."""

    text: str
    source: str          # file name, e.g. "fee_structure.pdf"
    page: int | None = None   # 1-indexed page number, if applicable
    chunk_id: str = field(default="")

    def metadata(self) -> dict:
        return {"source": self.source, "page": self.page, "chunk_id": self.chunk_id}


# --------------------------------------------------------------------------
# Per-format extraction
# --------------------------------------------------------------------------
def _extract_pdf(path: Path) -> list[tuple[str, int]]:
    """Return list of (page_text, page_number)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i))
    return pages


def _extract_docx(path: Path) -> list[tuple[str, int]]:
    doc = DocxDocument(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(full_text, None)] if full_text.strip() else []


def _extract_txt(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, None)] if text.strip() else []


def _extract_csv(path: Path) -> list[tuple[str, int]]:
    """Flatten each CSV row into a readable 'key: value, key: value' sentence."""
    rows_text = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = ", ".join(f"{k.strip()}: {v.strip()}" for k, v in row.items() if k and v)
            if sentence:
                rows_text.append(sentence)
    combined = "\n".join(rows_text)
    return [(combined, None)] if combined.strip() else []


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
    ".csv": _extract_csv,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def load_documents(documents_dir: Path | None = None) -> list[tuple[str, str, int | None]]:
    """
    Load all supported files from the documents directory.
    Returns a list of (raw_text, source_filename, page_number).
    """
    documents_dir = documents_dir or settings.DOCUMENTS_DIR
    results: list[tuple[str, str, int | None]] = []

    if not documents_dir.exists():
        logger.warning("Documents directory %s does not exist.", documents_dir)
        return results

    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        extractor = _EXTRACTORS[path.suffix.lower()]
        try:
            for text, page in extractor(path):
                results.append((text, path.name, page))
        except Exception as exc:  # noqa: BLE001 - log and continue with other files
            logger.error("Failed to parse %s: %s", path.name, exc)

    logger.info("Loaded %d text segments from %s", len(results), documents_dir)
    return results


def chunk_documents(
    raw_segments: list[tuple[str, str, int | None]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split loaded text segments into overlapping chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for text, source, page in raw_segments:
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunk_id = f"{source}:{page or 0}:{i}"
            chunks.append(Chunk(text=piece, source=source, page=page, chunk_id=chunk_id))

    logger.info("Produced %d chunks from %d segments", len(chunks), len(raw_segments))
    return chunks


def build_knowledge_chunks(documents_dir: Path | None = None) -> list[Chunk]:
    """Convenience wrapper: load + chunk in one call."""
    raw = load_documents(documents_dir)
    return chunk_documents(raw)
