"""Evidence-only extractive answer assembly; no generative model or external calls."""
from __future__ import annotations

import json
from .evidence import EvidenceExtractor, EvidenceFinding

NOT_FOUND = "The requested information could not be found in the uploaded document."


class QueryEngine:
    def __init__(self, processor, retriever, store, top_k: int = 4) -> None:
        self.processor, self.retriever, self.store, self.top_k = processor, retriever, store, top_k
        self.evidence_extractor = EvidenceExtractor()

    def index_document(self, document_id: str) -> None:
        document = self.processor.parse_document(
            self.store.upload_path(document_id), document_id, self.store.original_filename(document_id)
        )
        chunks = self.processor.chunk_structured_document(document)
        if not chunks:
            raise ValueError("The PDF did not contain searchable content.")
        directory = self.store.index_dir(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "structured_document.json").write_text(
            json.dumps(document.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        self.retriever.build(document_id, chunks)

    def retrieve_evidence(self, document_id: str, sector: str, request: str) -> list[EvidenceFinding]:
        try:
            chunks = self.retriever.search(document_id, sector, request, self.top_k)
        except FileNotFoundError:
            return []
        if not chunks:
            return []
        raw_findings = self.evidence_extractor.extract(chunks, request)
        return self.evidence_extractor.validate(raw_findings, document_id, sector)

    def answer(self, document_id: str, sector: str, request: str) -> str:
        findings = self.retrieve_evidence(document_id, sector, request)
        if not findings:
            return NOT_FOUND
        # Two evidence sentences normally answer a focused factual question while
        # avoiding a broad chapter-summary response. The detailed assessment
        # endpoints remain available for multi-finding outputs.
        return " ".join(finding.statement for finding in findings[:2])
