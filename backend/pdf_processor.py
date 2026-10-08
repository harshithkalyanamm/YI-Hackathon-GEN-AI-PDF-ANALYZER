"""Local PDF extraction, lightweight section detection, and deterministic chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


SECTION_HEADINGS = {
    "corporate", "sector risk", "bank risk", "climate transition risk",
    "nature transition risk", "other financial/risk information", "other financial information",
}


@dataclass
class Chunk:
    text: str
    sector: str | None
    section: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_heading(line: str) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 8) or len(line) > 80 or line.endswith((".", ";", ",", ":")):
        return False
    return line.isupper() or line.istitle() or line.lower() in SECTION_HEADINGS


def _is_generic_heading(line: str) -> bool:
    return line.lower() in SECTION_HEADINGS


def _word_chunks(text: str, sector: str | None, section: str | None, size: int, overlap: int) -> Iterable[Chunk]:
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        selected = words[start:start + size]
        if not selected:
            break
        yield Chunk(" ".join(selected), sector, section)
        if start + size >= len(words):
            break


class PDFProcessor:
    def __init__(self, chunk_words: int = 180, overlap_words: int = 35) -> None:
        self.chunk_words, self.overlap_words = chunk_words, overlap_words

    def extract_text(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as document:
                text = "\n".join(page.get_text("text") for page in document)
        except (fitz.FileDataError, RuntimeError) as exc:
            raise ValueError("The uploaded file is not a readable PDF.") from exc
        text = "\n".join(_normalise(line) for line in text.splitlines() if _normalise(line))
        if not text:
            raise ValueError("No extractable text was found in the PDF.")
        return text

    def chunk_document(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        sector: str | None = None
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if buffer:
                chunks.extend(_word_chunks(" ".join(buffer), sector, section, self.chunk_words, self.overlap_words))
                buffer = []

        for raw_line in text.splitlines():
            line = _normalise(raw_line)
            if _is_heading(line):
                flush()
                if _is_generic_heading(line):
                    section = line
                else:
                    # A non-generic short heading is treated as a sector label. This is intentionally data-driven.
                    sector = line
                    section = None
                continue
            buffer.append(line)
        flush()
        if not chunks:  # Documents without layout-friendly headings remain searchable.
            chunks = list(_word_chunks(_normalise(text), None, None, self.chunk_words, self.overlap_words))
        return chunks
