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
