"""Evidence-only extractive answer assembly; no generative model or external calls."""
from __future__ import annotations

import re

from .retriever import tokens

NOT_FOUND = "The requested information could not be found in the uploaded document."


class QueryEngine:
    def __init__(self, processor, retriever, store, top_k: int = 4) -> None:
        self.processor, self.retriever, self.store, self.top_k = processor, retriever, store, top_k

    def index_document(self, document_id: str) -> None:
        text = self.processor.extract_text(self.store.upload_path(document_id))
        self.retriever.build(document_id, self.processor.chunk_document(text))

    def answer(self, document_id: str, sector: str, request: str) -> str:
        try:
            chunks = self.retriever.search(document_id, sector, request, self.top_k)
        except FileNotFoundError:
            return NOT_FOUND
        if not chunks:
            return NOT_FOUND
        request_terms = tokens(request)
        selected: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(tokens(sentence) & request_terms),
                reverse=True,
            )
            for sentence in ranked:
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if len(cleaned) >= 20 and cleaned not in seen:
                    selected.append(cleaned)
                    seen.add(cleaned)
                    if len(selected) == 3:
                        return " ".join(selected)
        return " ".join(selected) if selected else NOT_FOUND
