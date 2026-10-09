from pathlib import Path

import pytest

from backend.assessment_service import AssessmentService, INSUFFICIENT
from tests.test_api import make_engine
from tests.test_representative_document import write_representative_pdf


def test_box_one_uses_only_primary_bes_sector_evidence(tmp_path: Path) -> None:
    source = tmp_path / "bes-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name, "PRIMARY_BES")
    engine.index_document(document_id)

    answer = AssessmentService(engine.store, engine).sector_trends(document_id, "Energy")

    assert answer.startswith("Sector-level findings for Energy:")
    assert "carbon-pricing" in answer.lower()
    assert "data-centre" not in answer.lower()


def test_box_one_rejects_documents_without_primary_bes_scope(tmp_path: Path) -> None:
    source = tmp_path / "general-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name)
    engine.index_document(document_id)

    with pytest.raises(ValueError, match="PRIMARY_BES"):
        AssessmentService(engine.store, engine).sector_trends(document_id, "Energy")


def test_box_one_abstains_when_no_sector_evidence_exists(tmp_path: Path) -> None:
    source = tmp_path / "bes-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name, "PRIMARY_BES")
    engine.index_document(document_id)

    assert AssessmentService(engine.store, engine).sector_trends(document_id, "Agriculture") == INSUFFICIENT
