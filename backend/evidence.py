"""Document-grounded evidence extraction and validation.

Evidence objects remain internal. They create an audit boundary between retrieval
and answer writing without exposing sources or metadata through the public API.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .pdf_processor import Chunk
from .retriever import tokens


@dataclass(frozen=True)
class EvidenceSource:
    document_id: str
    filename: str
    page_start: int
    page_end: int
    sector: str | None
    heading_path: list[str]
    element_type: str
    evidence_scope: str


@dataclass(frozen=True)
class EvidenceFinding:
    finding_id: str
    category: str
    statement: str
    source: EvidenceSource

    def to_dict(self) -> dict:
        return asdict(self)


def _category(request: str) -> str:
    return "_".join(re.findall(r"[A-Za-z0-9]+", request.upper())) or "GENERAL"


class EvidenceExtractor:
    """Extract only sentences physically present in retrieved document chunks."""
    def extract(self, chunks: list[Chunk], request: str) -> list[EvidenceFinding]:
        request_terms = tokens(request)
        findings: list[EvidenceFinding] = []
        seen: set[str] = set()
        for chunk in chunks:
            source = EvidenceSource(
                document_id=chunk.document_id or "",
                filename=chunk.filename or "",
                page_start=chunk.page_start or 0,
                page_end=chunk.page_end or 0,
                sector=chunk.sector,
                heading_path=chunk.heading_path,
                element_type=chunk.element_type,
                evidence_scope=chunk.evidence_scope,
            )
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            heading_terms = tokens(" ".join((*chunk.heading_path, chunk.section or "")))
            # When a query names the current heading (for example, "functions"
            # under "Powers and Functions"), it is asking for that section. Keep
            # the source order instead of over-weighting one generic query term.
            ordered_sentences = sentences if request_terms and request_terms <= heading_terms else sorted(
                sentences, key=lambda item: len(tokens(item) & request_terms), reverse=True
            )
            for sentence in ordered_sentences:
                statement = re.sub(r"\s+", " ", sentence).strip()
                if len(statement) < 20 or statement in seen:
                    continue
                findings.append(EvidenceFinding(
                    finding_id=f"F{len(findings) + 1:03d}", category=_category(request),
                    statement=statement, source=source,
                ))
                seen.add(statement)
        return findings

    @staticmethod
    def validate(findings: list[EvidenceFinding], document_id: str, sector: str) -> list[EvidenceFinding]:
        """Enforce source and scope boundaries before any answer uses evidence."""
        wanted_sector = sector.casefold().strip()
        return [
            finding for finding in findings
            if finding.source.document_id == document_id
            and finding.source.page_start > 0
            and finding.source.page_end >= finding.source.page_start
            and finding.source.evidence_scope == "SECTOR"
            and finding.source.sector is not None
            and finding.source.sector.casefold().strip() == wanted_sector
        ]
