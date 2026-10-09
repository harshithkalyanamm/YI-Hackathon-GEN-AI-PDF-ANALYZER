from pathlib import Path

import fitz

from tests.test_api import make_engine


def write_vice_president_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(54, 54, 541, 788),
        "CHAPTER 19 Vice-President\n\nPOWERS AND FUNCTIONS\n"
        "The Vice-President acts as the ex-officio Chairman of Rajya Sabha. "
        "The Vice-President acts as President when a vacancy occurs in the office of the President. "
        "When the President is absent or unable to discharge functions, the Vice-President discharges those functions. "
        "During this period, the Deputy Chairman performs the duties of the Rajya Sabha chairman.",
        fontsize=11, lineheight=1.35,
    )
    document.save(path)
    document.close()


def test_focused_question_uses_heading_context_and_returns_two_direct_functions(tmp_path: Path) -> None:
    source = tmp_path / "indian-polity-extract.pdf"
    write_vice_president_pdf(source)
    engine = make_engine(tmp_path)
    document_id = engine.store.save_upload(source, source.name)
    engine.index_document(document_id)

    answer = engine.answer(document_id, "CHAPTER 19 Vice-President", "functions")

    assert "ex-officio chairman of rajya sabha" in answer.lower()
    assert "acts as president when a vacancy occurs" in answer.lower()
    assert "deputy chairman" not in answer.lower()
