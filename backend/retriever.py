"""Persisted NumPy cosine-similarity retrieval for small local document sets."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import numpy as np

from .pdf_processor import Chunk

STOP_WORDS = {"a", "an", "the", "are", "what", "is", "of", "for", "and", "or", "to", "in", "on", "with", "about", "please", "summarize", "summary"}
GENERIC_RISK_TERMS = {"risk", "major", "main", "key", "information", "detail", "details"}


def tokens(value: str) -> set[str]:
    result: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", value.lower()):
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]  # enough normalisation for 'risks'/'risk' without another NLP runtime
        if word not in STOP_WORDS and len(word) > 1:
            result.add(word)
    return result


class Retriever(ABC):
    @abstractmethod
    def build(self, document_id: str, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, document_id: str, sector: str, request: str, top_k: int) -> list[Chunk]: ...


class LocalNumpyRetriever(Retriever):
    """Exact, transparent local vector search.

    Document and query vectors are L2-normalized by the embedder, making a matrix
    multiplication equivalent to cosine-similarity search. For the hackathon's
    per-document indexes (hundreds or low thousands of chunks), this is simple,
    deterministic, and more than fast enough on a CPU without native DLLs.
    """
    def __init__(self, store, embedder) -> None:
        self.store, self.embedder = store, embedder
        self._cache: dict[str, tuple[np.ndarray, list[Chunk]]] = {}

    def build(self, document_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("The PDF did not contain searchable content.")
        vectors = self.embedder.embed([self._index_text(chunk) for chunk in chunks])
        directory = self.store.index_dir(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", vectors)
        (directory / "chunks.json").write_text(json.dumps([chunk.to_dict() for chunk in chunks]), encoding="utf-8")
        self._cache[document_id] = (vectors, chunks)

    def _load(self, document_id: str):
        if document_id in self._cache:
            return self._cache[document_id]
        directory = self.store.index_dir(document_id)
        vector_file, chunks_file = directory / "vectors.npy", directory / "chunks.json"
        if not vector_file.is_file() or not chunks_file.is_file():
            raise FileNotFoundError(document_id)
        index = np.load(vector_file)
        chunks = [Chunk(**item) for item in json.loads(chunks_file.read_text(encoding="utf-8"))]
        self._cache[document_id] = (index, chunks)
        return index, chunks

    @staticmethod
    def _index_text(chunk: Chunk) -> str:
        return " ".join(value for value in (chunk.sector, chunk.section, chunk.text) if value)

    def search(self, document_id: str, sector: str, request: str, top_k: int) -> list[Chunk]:
        index, chunks = self._load(document_id)
        query = f"{sector} {request}"
        vector = self.embedder.embed([query])
        limit = min(len(chunks), max(top_k * 4, 12))
        candidate_ids = np.argsort(-(index @ vector[0]))[:limit]
        wanted_sector = sector.casefold().strip()
        request_terms = tokens(request)
        # For named risk types, require their distinguishing words. This prevents a
        # climate-risk section being returned for a missing nature-risk request just
        # because both phrases contain the generic word "risk".
        required_terms = request_terms - GENERIC_RISK_TERMS
        scored: list[tuple[float, Chunk]] = []
        for position in candidate_ids:
            if position < 0:
                continue
            chunk = chunks[int(position)]
            # Explicit sector metadata is a safety boundary: never answer for a different labelled sector.
            if chunk.sector and chunk.sector.casefold().strip() != wanted_sector:
                continue
            haystack = tokens(" ".join(value for value in (chunk.section, chunk.text) if value))
            if required_terms and not required_terms.issubset(haystack):
                continue
            lexical = len(request_terms & haystack) / max(1, len(request_terms))
            heading_bonus = 0.65 if chunk.section and tokens(chunk.section) >= request_terms and request_terms else 0
            score = lexical + heading_bonus
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
