"""Persisted NumPy cosine-similarity retrieval for small local document sets."""
from __future__ import annotations

import json
import math
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


def token_sequence(value: str) -> list[str]:
    """Tokenise consistently with ``tokens`` while preserving term frequency for BM25."""
    sequence: list[str] = []
    for word in re.findall(r"[a-z0-9]+", value.lower()):
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]
        if word not in STOP_WORDS and len(word) > 1:
            sequence.append(word)
    return sequence


def _normalise_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    spread = float(values.max() - values.min())
    return np.ones_like(values) if spread == 0 else (values - values.min()) / spread


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
    def __init__(self, store, embedder, strategy: str = "dense") -> None:
        if strategy not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Retrieval strategy must be dense, bm25, or hybrid.")
        self.store, self.embedder = store, embedder
        self.strategy = strategy
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
        return " ".join(value for value in (*chunk.heading_path, chunk.text) if value)

    @staticmethod
    def _bm25_scores(chunks: list[Chunk], request: str) -> np.ndarray:
        """Small in-process BM25 implementation for exact terminology and acronyms."""
        query_terms = token_sequence(request)
        documents = [token_sequence(LocalNumpyRetriever._index_text(chunk)) for chunk in chunks]
        if not documents or not query_terms:
            return np.zeros(len(documents), dtype="float32")
        document_frequency = {
            term: sum(term in set(document) for document in documents) for term in set(query_terms)
        }
        average_length = sum(map(len, documents)) / len(documents)
        scores = np.zeros(len(documents), dtype="float32")
        k1, b = 1.5, 0.75
        for index, document in enumerate(documents):
            frequencies = {term: document.count(term) for term in set(query_terms)}
            for term, frequency in frequencies.items():
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (len(documents) - document_frequency[term] + 0.5) /
                                             (document_frequency[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(document) / max(1.0, average_length))
                scores[index] += inverse_frequency * frequency * (k1 + 1) / denominator
        return scores

    def search(self, document_id: str, sector: str, request: str, top_k: int) -> list[Chunk]:
        index, chunks = self._load(document_id)
        query = f"{sector} {request}"
        vector = self.embedder.embed([query])
        wanted_sector = sector.casefold().strip()
        # Sector is a hard boundary. Filter the vector matrix before ranking so a
        # similarly worded section from another sector cannot consume candidates.
        labelled = [position for position, chunk in enumerate(chunks) if chunk.sector]
        eligible_ids = [
            position for position, chunk in enumerate(chunks)
            if chunk.sector and chunk.sector.casefold().strip() == wanted_sector
        ]
        if labelled and not eligible_ids:
            return []
        if not labelled:  # retain searchability for a document with no detectable headings
            eligible_ids = list(range(len(chunks)))
        eligible_chunks = [chunks[position] for position in eligible_ids]
        dense_scores = index[eligible_ids] @ vector[0]
        bm25_scores = self._bm25_scores(eligible_chunks, request)
        if self.strategy == "dense":
            combined_scores = dense_scores
        elif self.strategy == "bm25":
            combined_scores = bm25_scores
        else:
            # Weighted fusion remains transparent and is covered by the golden
            # evaluation suite. It can be switched with FINANCIAL_RISK_RETRIEVAL_STRATEGY.
            combined_scores = 0.65 * _normalise_scores(dense_scores) + 0.35 * _normalise_scores(bm25_scores)
        ranked_local_ids = np.argsort(-combined_scores)
        request_terms = tokens(request)
        # For named risk types, require their distinguishing words. This prevents a
        # climate-risk section being returned for a missing nature-risk request just
        # because both phrases contain the generic word "risk".
        required_terms = request_terms - GENERIC_RISK_TERMS
        scored: list[tuple[float, Chunk]] = []
        for local_position in ranked_local_ids:
            chunk = eligible_chunks[int(local_position)]
            # Heading context is evidence metadata as well as retrieval context.
            # A passage under "Powers and Functions" can rely on its chapter title
            # to identify the office it describes.
            haystack = tokens(" ".join(value for value in (*chunk.heading_path, chunk.section, chunk.text) if value))
            if required_terms and not required_terms.issubset(haystack):
                continue
            scored.append((float(combined_scores[int(local_position)]), chunk))
            if len(scored) == top_k:
                break
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]
