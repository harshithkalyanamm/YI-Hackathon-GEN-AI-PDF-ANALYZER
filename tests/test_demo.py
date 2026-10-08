from pathlib import Path

import fitz

from demo import DEMO_LINES, create_demo_pdf


def test_demo_creates_a_twenty_line_pdf(tmp_path: Path) -> None:
    pdf = create_demo_pdf(tmp_path / "demo.pdf")
    with fitz.open(pdf) as document:
        extracted = document[0].get_text("text")

    assert len(DEMO_LINES) == 20
    assert "Banks with high exposure to energy loans" in extracted
