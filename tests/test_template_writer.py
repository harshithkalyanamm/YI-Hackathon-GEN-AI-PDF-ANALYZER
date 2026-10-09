from backend.evidence import EvidenceFinding, EvidenceSource
from backend.template_writer import INSUFFICIENT, TemplateWriter


def finding(statement: str) -> EvidenceFinding:
    return EvidenceFinding(
        "F001", "RISK", statement,
        EvidenceSource("bes", "bes.pdf", 1, 1, "Energy", ["Energy"], "paragraph", "SECTOR"),
    )


def test_templates_only_join_supplied_evidence() -> None:
    evidence = [finding("Carbon pricing may increase compliance costs."), finding("Supply disruption can affect demand.")]
    output = TemplateWriter.sector_trends("Energy", evidence)

    assert output == "Sector-level findings for Energy: Carbon pricing may increase compliance costs. Supply disruption can affect demand."
    assert TemplateWriter.sector_trends("Energy", []) == INSUFFICIENT
