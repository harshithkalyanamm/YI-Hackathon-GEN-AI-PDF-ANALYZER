"""Offline structured PDF parsing and section-aware chunking.

PyMuPDF is used deliberately as the lightweight local parser for this MVP. The
structured intermediate representation preserves page, headings and tables, and
leaves a clean boundary for a locally provisioned Docling parser in a later
benchmark without changing retrieval or the API.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


SECTION_HEADINGS = {
    "corporate", "sector risk", "bank risk", "climate transition risk",
    "nature transition risk", "other financial/risk information", "other financial information",
}


@dataclass
class StructuredElement:
    page: int
    text: str
    element_type: str
    sector: str | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredDocument:
    document_id: str
    filename: str
    elements: list[StructuredElement]

    def to_dict(self) -> dict:
        return {"document_id": self.document_id, "filename": self.filename,
                "elements": [element.to_dict() for element in self.elements]}


@dataclass
class Chunk:
    text: str
    sector: str | None
    section: str | None
    document_id: str | None = None
    filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = field(default_factory=list)
    element_type: str = "paragraph"
    evidence_scope: str = "CORPORATE"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_heading(line: str, font_size: float = 0, body_size: float = 0) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 10) or len(line) > 100 or line.endswith((".", ";", ",", ":")):
        return False
    layout_heading = body_size > 0 and font_size >= body_size + 1.0
    chapter_heading = bool(re.match(r"^(chapter|part)\s+[0-9ivxlcdm]+\b", line, re.IGNORECASE))
    return layout_heading or chapter_heading or line.isupper() or line.istitle() or line.lower() in SECTION_HEADINGS


def _is_generic_heading(line: str) -> bool:
    return line.lower() in SECTION_HEADINGS


def _word_chunks(text: str, metadata: dict, size: int, overlap: int) -> Iterable[Chunk]:
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        selected = words[start:start + size]
        if not selected:
            break
        yield Chunk(text=" ".join(selected), **metadata)
        if start + size >= len(words):
            break


class PDFProcessor:
    def __init__(self, chunk_words: int = 180, overlap_words: int = 35) -> None:
        self.chunk_words, self.chunk_overlap_words = chunk_words, overlap_words

    def _page_lines(self, page: fitz.Page) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        lines: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = _normalise("".join(span.get("text", "") for span in spans))
                if text:
                    size = max((float(span.get("size", 0)) for span in spans), default=0)
                    lines.append((text, size, tuple(line["bbox"])))
        return lines

    @staticmethod
    def _tables(page: fitz.Page) -> list[tuple[tuple[float, float, float, float], str]]:
        """Preserve table headers and cells as a searchable local table element."""
        try:
            found = page.find_tables()
        except (AttributeError, RuntimeError):
            return []
        results: list[tuple[tuple[float, float, float, float], str]] = []
        for table in found.tables:
            rows = [[_normalise(cell or "") for cell in row] for row in table.extract()]
            rows = [row for row in rows if any(row)]
            if not rows:
                continue
            headers = [cell or f"Column {index + 1}" for index, cell in enumerate(rows[0])]
            serialised = ["Table: " + " | ".join(headers)]
            for row in rows[1:]:
                serialised.append("; ".join(
                    f"{headers[index]}: {value}" for index, value in enumerate(row) if value
                ))
            results.append((tuple(table.bbox), " ".join(part for part in serialised if part)))
        return results

    def parse_document(self, pdf_path: Path, document_id: str, filename: str | None = None) -> StructuredDocument:
        try:
            document = fitz.open(pdf_path)
        except (fitz.FileDataError, RuntimeError) as exc:
            raise ValueError("The uploaded file is not a readable PDF.") from exc

        elements: list[StructuredElement] = []
        sector: str | None = None
        section: str | None = None
        heading_path: list[str] = []
        try:
            for page_number, page in enumerate(document, start=1):
                lines = self._page_lines(page)
                if not lines:
                    continue
                sizes = sorted(size for _, size, _ in lines if size > 0)
                body_size = sizes[len(sizes) // 2] if sizes else 0
                tables = self._tables(page)

                def in_table(bbox: tuple[float, float, float, float]) -> bool:
                    left, top, right, bottom = bbox
                    x, y = (left + right) / 2, (top + bottom) / 2
                    return any(x0 <= x <= x1 and y0 <= y <= y1 for (x0, y0, x1, y1), _ in tables)

                buffer: list[str] = []

                def flush() -> None:
                    nonlocal buffer
                    if buffer:
                        elements.append(StructuredElement(
                            page_number, " ".join(buffer), "paragraph", sector, section, list(heading_path)
                        ))
                        buffer = []

                for line, size, bbox in lines:
                    if in_table(bbox):
                        continue
                    if _is_heading(line, size, body_size):
                        flush()
                        if _is_generic_heading(line):
                            section = line
                            heading_path = ([sector] if sector else []) + [section]
                        elif sector and line.isupper():
                            # In general documents, a chapter/title is often followed by
                            # uppercase subsection labels (for example, POWERS AND
                            # FUNCTIONS). They are sections of the existing subject,
                            # not a new sector/subject boundary.
                            section = line.title()
                            heading_path = [sector, section]
                        else:
                            sector, section, heading_path = line, None, [line]
                    else:
                        buffer.append(line)
                flush()
                for _, table_text in tables:
                    elements.append(StructuredElement(
                        page_number, table_text, "table", sector, section, list(heading_path)
                    ))
        finally:
            document.close()
        if not elements:
            raise ValueError("No extractable text was found in the PDF.")
        return StructuredDocument(document_id, filename or pdf_path.name, elements)

    def extract_text(self, pdf_path: Path) -> str:
        """Compatibility helper for callers that only need text."""
        try:
            with fitz.open(pdf_path) as document:
                text = "\n".join(page.get_text("text") for page in document)
        except (fitz.FileDataError, RuntimeError) as exc:
            raise ValueError("The uploaded file is not a readable PDF.") from exc
        text = "\n".join(_normalise(line) for line in text.splitlines() if _normalise(line))
        if not text:
            raise ValueError("No extractable text was found in the PDF.")
        return text

    def chunk_structured_document(self, document: StructuredDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for element in document.elements:
            metadata = {
                "sector": element.sector, "section": element.section,
                "document_id": document.document_id, "filename": document.filename,
                "page_start": element.page, "page_end": element.page,
                "heading_path": element.heading_path, "element_type": element.element_type,
                "evidence_scope": "SECTOR" if element.sector else "CORPORATE",
            }
            chunks.extend(_word_chunks(element.text, metadata, self.chunk_words, self.chunk_overlap_words))
        return chunks

    def chunk_document(self, text: str) -> list[Chunk]:
        """Legacy text-only entry point retained for existing callers."""
        lines = [_normalise(line) for line in text.splitlines() if _normalise(line)]
        elements: list[StructuredElement] = []
        sector: str | None = None
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if buffer:
                path = ([sector] if sector else []) + ([section] if section else [])
                elements.append(StructuredElement(1, " ".join(buffer), "paragraph", sector, section, path))
                buffer = []

        for line in lines:
            if _is_heading(line):
                flush()
                if _is_generic_heading(line):
                    section = line
                else:
                    sector, section = line, None
            else:
                buffer.append(line)
        flush()
        if not elements and text.strip():
            elements = [StructuredElement(1, _normalise(text), "paragraph")]
        return self.chunk_structured_document(StructuredDocument("text", "text", elements))
