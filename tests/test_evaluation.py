from pathlib import Path

from backend.evaluation import RetrievalCase, evaluate_retriever
from tests.test_api import make_engine
from tests.test_representative_document import write_representative_pdf


def test_golden_retrieval_set_measures_grounding_and_sector_isolation(tmp_path: Path) -> None:
    source = tmp_path / "fictional-credit-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name)
    engine.index_document(document_id)

    cases = [
        RetrievalCase("T001", document_id, "Energy", "climate transition risk", "Climate Transition Risk", True),
        RetrievalCase("T002", document_id, "Energy", "bank risk", "Bank Risk", True),
        RetrievalCase("T003", document_id, "Technology", "climate transition risk", "Climate Transition Risk", True),
        RetrievalCase("T004", document_id, "Energy", "biodiversity offset requirement", None, False),
    ]
    report = evaluate_retriever(engine.retriever, cases)

    assert report.recall_at_k == 1.0
    assert report.correct_abstention_rate == 1.0
    assert report.wrong_sector_rate == 0.0
