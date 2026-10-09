from pathlib import Path

from backend.evidence import EvidenceExtractor, EvidenceFinding, EvidenceSource
from tests.test_api import make_engine
from tests.test_representative_document import write_representative_pdf


def test_evidence_retains_provenance_and_validates_sector_scope(tmp_path: Path) -> None:
    source = tmp_path / "fictional-credit-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name)
    engine.index_document(document_id)

    findings = engine.retrieve_evidence(document_id, "Energy", "climate transition risk")

    assert findings
    finding = findings[0]
    assert "carbon-pricing" in finding.statement.lower()
    assert finding.source.document_id == document_id
    assert finding.source.filename == source.name
    assert finding.source.page_start == finding.source.page_end == 2
    assert finding.source.heading_path == ["Energy", "Climate Transition Risk"]
    assert finding.source.evidence_scope == "SECTOR"


def test_evidence_validator_rejects_wrong_document_and_scope() -> None:
    extractor = EvidenceExtractor()
    source = EvidenceSource("other-document", "other.pdf", 1, 1, "Energy", ["Energy"], "paragraph", "DEAL")
    finding = EvidenceFinding("F001", "BANK_RISK", "Evidence from the wrong scope.", source)

    assert extractor.validate([finding], "expected-document", "Energy") == []
