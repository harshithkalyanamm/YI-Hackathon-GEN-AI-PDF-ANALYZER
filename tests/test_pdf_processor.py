from pathlib import Path

import fitz

from backend.pdf_processor import PDFProcessor


def write_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Corporate\n\nEnergy\n\nClimate Transition Risk\n"
        "Energy companies face tighter emissions regulation, changing demand for fossil fuels, "
        "and capital expenditure for lower-carbon technology. These pressures can increase costs "
        "and reduce long-term profitability.\n\nBank Risk\n"
        "Banks with concentrated energy lending may face credit losses when borrowers cannot adapt.\n\n"
        "Technology\n\nSector Risk\n"
        "Technology firms face competition, rapid product obsolescence and cyber security disruption.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def test_extracts_and_labels_sections(tmp_path: Path) -> None:
    pdf = tmp_path / "sample-study.pdf"
    write_sample_pdf(pdf)
    processor = PDFProcessor(chunk_words=80, overlap_words=10)

    chunks = processor.chunk_document(processor.extract_text(pdf))

    assert any(chunk.sector == "Energy" and chunk.section == "Climate Transition Risk" for chunk in chunks)
    assert any(chunk.sector == "Technology" and chunk.section == "Sector Risk" for chunk in chunks)


def test_structured_parse_preserves_page_and_heading_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "sample-study.pdf"
    write_sample_pdf(pdf)
    processor = PDFProcessor(chunk_words=80, overlap_words=10)

    document = processor.parse_document(pdf, "study-001")
    chunks = processor.chunk_structured_document(document)

    climate = next(chunk for chunk in chunks if chunk.section == "Climate Transition Risk")
    assert climate.document_id == "study-001"
    assert climate.page_start == climate.page_end == 1
    assert climate.heading_path == ["Energy", "Climate Transition Risk"]
    assert climate.evidence_scope == "SECTOR"


def test_structured_parse_preserves_table_headers_and_rows(tmp_path: Path) -> None:
    pdf = tmp_path / "financial-table.pdf"
    document = fitz.open()
    page = document.new_page()
    columns, rows = [72, 230, 390], [100, 130, 160, 190]
    for x in columns:
        page.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page.draw_line((columns[0], y), (columns[-1], y))
    values = [["Metric", "2025"], ["Capital expenditure", "125"], ["Impairment charge", "22"]]
    for row_number, row in enumerate(values):
        for column_number, value in enumerate(row):
            page.insert_text((columns[column_number] + 4, rows[row_number] + 20), value, fontsize=9)
    document.save(pdf)
    document.close()

    parsed = PDFProcessor().parse_document(pdf, "table-study")
    table = next(element for element in parsed.elements if element.element_type == "table")

    assert "Metric | 2025" in table.text
    assert "Capital expenditure" in table.text
