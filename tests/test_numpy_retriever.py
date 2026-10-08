"""Regression tests for the persisted, dependency-free NumPy index."""
from pathlib import Path

from backend.document_store import LocalDocumentStore
from backend.pdf_processor import PDFProcessor
from backend.qa_engine import QueryEngine
from backend.retriever import LocalNumpyRetriever
from tests.test_api import DeterministicEmbedder
from tests.test_pdf_processor import write_sample_pdf


def test_vectors_are_persisted_and_reloaded_without_cross_sector_leakage(tmp_path: Path) -> None:
    uploads, indexes = tmp_path / "uploads", tmp_path / "indexes"
    store = LocalDocumentStore(uploads, indexes)
    source = tmp_path / "representative-study.pdf"
    write_sample_pdf(source)
    document_id = store.save_upload(source, source.name)

    processor = PDFProcessor(chunk_words=80, overlap_words=10)
    first_engine = QueryEngine(processor, LocalNumpyRetriever(store, DeterministicEmbedder()), store)
    first_engine.index_document(document_id)

    assert (store.index_dir(document_id) / "vectors.npy").is_file()
    assert not (store.index_dir(document_id) / "index.faiss").exists()

    # A fresh retriever proves the saved NumPy vectors, rather than in-memory state,
    # answer the query. Technology content must not leak into an Energy answer.
    reloaded_engine = QueryEngine(processor, LocalNumpyRetriever(store, DeterministicEmbedder()), store)
    answer = reloaded_engine.answer(document_id, "Energy", "climate transition risk")
    assert "emissions regulation" in answer.lower()
    assert "cyber security" not in answer.lower()
