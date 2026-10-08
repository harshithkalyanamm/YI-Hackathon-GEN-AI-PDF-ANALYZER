import io
from pathlib import Path

import numpy as np

from backend.app import create_app
from backend.config import Settings
from backend.document_store import LocalDocumentStore
from backend.pdf_processor import PDFProcessor
from backend.qa_engine import NOT_FOUND, QueryEngine
from backend.retriever import LocalFAISSRetriever, tokens
from backend.schemas import QueryRequest
from tests.test_pdf_processor import write_sample_pdf


class DeterministicEmbedder:
    """Small test double; production always uses LocalDistilBERTEmbedder."""
    def load(self) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        vocabulary = ["energy", "climate", "transition", "bank", "sector", "technology", "risk", "emissions"]
        matrix = np.array([[float(term in tokens(text)) for term in vocabulary] for text in texts], dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1.0)


def make_engine(tmp_path: Path) -> QueryEngine:
    settings = Settings(tmp_path, tmp_path / "uploads", tmp_path / "indexes", tmp_path / "model")
    settings.ensure_directories()
    store = LocalDocumentStore(settings.uploads_dir, settings.indexes_dir)
    # Exercise the persisted local-index fallback; production prefers FAISS automatically.
    retriever = LocalFAISSRetriever(store, DeterministicEmbedder(), use_faiss=False)
    return QueryEngine(PDFProcessor(chunk_words=80, overlap_words=10), retriever, store)


def run_immediate(coroutine):
    """Run endpoint coroutines that do not suspend on external I/O.

    This avoids the Windows host's broken asyncio runner while still exercising
    FastAPI's registered endpoint functions and their Pydantic response objects.
    """
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Endpoint unexpectedly suspended during an in-memory test.")


class InMemoryPDFUpload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self.file = io.BytesIO(content)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_indexing_and_sector_request_retrieval(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    source = tmp_path / "study.pdf"
    write_sample_pdf(source)
    document_id = engine.store.save_upload(source, "study.pdf")
    engine.index_document(document_id)

    answer = engine.answer(document_id, "Energy", "climate transition risk")

    assert "emissions regulation" in answer.lower()
    assert "credit losses" not in answer.lower()


def test_api_upload_query_and_missing_information(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    source = tmp_path / "study.pdf"
    write_sample_pdf(source)
    app = create_app(engine=engine)

    upload_handler = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/upload")
    query_handler = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/query")
    uploaded = run_immediate(upload_handler(InMemoryPDFUpload("study.pdf", source.read_bytes()), engine))
    assert uploaded.status == "processed"

    response = run_immediate(query_handler(QueryRequest(
        document_id=uploaded.document_id, sector="Energy", request="bank risk"
    ), engine))
    assert set(response.model_dump()) == {"answer"}
    assert "credit losses" in response.answer.lower()

    missing = run_immediate(query_handler(QueryRequest(
        document_id=uploaded.document_id, sector="Energy", request="nature transition risk"
    ), engine))
    assert missing.answer == NOT_FOUND
