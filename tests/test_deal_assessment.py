from pathlib import Path

import fitz
import pytest

from backend.assessment_service import APPLICABILITY_UNKNOWN, APPLICABLE_EVIDENCE_FOUND, APPLICABLE_EVIDENCE_NOT_FOUND, NOT_APPLICABLE, AssessmentService
from tests.test_api import make_engine


def write_deal_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(72, 72, 540, 780),
        "Energy\n\nClimate Transition Risk\n"
        "The borrower operates high-emitting energy assets and its investment plan includes "
        "methane-abatement equipment required by the new emissions standard.\n\n"
        "Bank Risk\nThe borrower maintains a debt service reserve account equal to six months of interest payments.",
        fontsize=11,
        lineheight=1.35,
    )
    document.save(path)
    document.close()


def test_box_two_uses_only_current_transaction_document(tmp_path: Path) -> None:
    source = tmp_path / "transaction-001-memo.pdf"
    write_deal_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name, "DEAL_DOCUMENT", "TRANSACTION_001")
    engine.index_document(document_id)

    answer = AssessmentService(engine.store, engine).deal_risk_drivers(
        document_id, "TRANSACTION_001", "Energy", "climate transition risk"
    )

    assert answer.startswith(APPLICABLE_EVIDENCE_FOUND)
    assert "methane-abatement" in answer.lower()
    assert "capital expenditure" not in answer.lower()  # no unsupported financial inference


def test_box_two_rejects_cross_transaction_access_and_abstains(tmp_path: Path) -> None:
    source = tmp_path / "transaction-001-memo.pdf"
    write_deal_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name, "DEAL_DOCUMENT", "TRANSACTION_001")
    engine.index_document(document_id)
    service = AssessmentService(engine.store, engine)

    with pytest.raises(ValueError, match="requested transaction"):
        service.deal_risk_drivers(document_id, "TRANSACTION_002", "Energy", "climate transition risk")
    missing = service.deal_risk_drivers(document_id, "TRANSACTION_001", "Energy", "nature transition risk")
    assert missing.startswith(APPLICABILITY_UNKNOWN)
    applicable_missing = service.deal_risk_drivers(
        document_id, "TRANSACTION_001", "Energy", "nature transition risk", "APPLICABLE"
    )
    assert applicable_missing.startswith(APPLICABLE_EVIDENCE_NOT_FOUND)
    not_applicable = service.deal_risk_drivers(
        document_id, "TRANSACTION_001", "Energy", "nature transition risk", "NOT_APPLICABLE"
    )
    assert not_applicable.startswith(NOT_APPLICABLE)
