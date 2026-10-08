"""Create a harmless 20-line PDF and demonstrate the local API end to end.

Start the API first, then run:  python demo.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import fitz


DEMO_LINES = (
    "Corporate Overview",
    "This study reviews financial risks across selected sectors.",
    "All findings are based on current market and regulatory conditions.",
    "",
    "Energy",
    "Sector Risk",
    "Energy companies are exposed to volatile fuel prices and supply disruption.",
    "They may also face changing demand from industrial customers.",
    "",
    "Bank Risk",
    "Banks with high exposure to energy loans may face higher credit losses.",
    "Borrowers may struggle to repay debt when energy prices fall sharply.",
    "This can increase loan-loss provisions and reduce bank profitability.",
    "",
    "Climate Transition Risk",
    "Stricter emissions rules may increase compliance and operating costs.",
    "Energy companies may need substantial investment in lower-carbon technology.",
    "Demand for high-carbon fuels may decline over the long term.",
    "",
    "End of study.",
)


def create_demo_pdf(destination: Path) -> Path:
    """Create the 20-line non-confidential PDF used by this demonstration."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    written = page.insert_textbox(
        fitz.Rect(54, 54, 541, 788), "\n".join(DEMO_LINES), fontsize=11, lineheight=1.35
    )
    if written < 0:
        raise RuntimeError("The demo text did not fit onto its PDF page.")
    document.save(destination)
    document.close()
    return destination


def run_demo(api_url: str, pdf_path: Path) -> None:
    """Call the public API exactly as a consuming application would."""
    with httpx.Client(base_url=api_url.rstrip("/"), timeout=60.0) as client:
        with pdf_path.open("rb") as pdf_file:
            upload = client.post("/upload", files={"file": (pdf_path.name, pdf_file, "application/pdf")})
        upload.raise_for_status()
        document_id = upload.json()["document_id"]
        question = {"document_id": document_id, "sector": "Energy", "request": "bank risk"}
        response = client.post("/query", json=question)
        response.raise_for_status()

    print("Demo PDF created:", pdf_path)
    print("Question:", question)
    print("API response:", response.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Financial Risk AI demo.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Running local API URL")
    parser.add_argument("--output", type=Path, default=Path("demo_energy_study.pdf"), help="Generated sample PDF path")
    parser.add_argument("--create-only", action="store_true", help="Create the PDF but do not call the API")
    args = parser.parse_args()

    pdf_path = create_demo_pdf(args.output)
    if args.create_only:
        print(f"Created 20-line sample PDF: {pdf_path}")
        return
    run_demo(args.api_url, pdf_path)


if __name__ == "__main__":
    main()
