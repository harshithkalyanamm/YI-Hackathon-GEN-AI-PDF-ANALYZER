"""Realistic-but-fictional financial study test; no confidential data is stored."""
from pathlib import Path

import fitz

from tests.test_api import make_engine


PAGES = (
    """Corporate\n\nEnergy\n\nSector Risk\nEnergy issuers are sensitive to commodity-price volatility, supply disruption and uncertain industrial demand. Debt-funded capital programmes can pressure liquidity when margins weaken.\n\nBank Risk\nLenders with concentrated exposure to upstream energy borrowers may experience higher probability of default and increased impairment charges during a prolonged commodity-price decline.""",
    """Energy\n\nClimate Transition Risk\nA tightening carbon-pricing regime and methane-emissions standards could increase compliance costs for high-emitting assets. Investment in grid modernisation, renewable capacity and emissions-abatement equipment may require material capital expenditure. Declining demand for carbon-intensive products could reduce asset utilisation and long-term profitability.\n\nNature Transition Risk\nWater-stress restrictions near extraction sites may constrain operations where permits require reduced withdrawals.""",
    """Technology\n\nSector Risk\nTechnology firms face rapid product obsolescence, cyber-security incidents and aggressive price competition. These factors can disrupt revenue and compress margins.\n\nClimate Transition Risk\nData-centre electricity demand and supplier emissions reporting may increase operating costs for some technology companies.""",
)


def write_representative_pdf(path: Path) -> None:
    document = fitz.open()
    for text in PAGES:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(54, 54, 541, 788), text, fontsize=11, lineheight=1.35)
    document.save(path)
    document.close()


def test_representative_multi_page_financial_document(tmp_path: Path) -> None:
    source = tmp_path / "fictional-energy-credit-study.pdf"
    write_representative_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name)
    engine.index_document(document_id)

    answer = engine.answer(document_id, "Energy", "climate transition risk")

    assert "carbon-pricing regime" in answer.lower()
    assert "data-centre" not in answer.lower()
