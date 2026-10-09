"""Run an offline golden retrieval evaluation against one already-indexed document.

Example:
  python evaluate.py --document-id YOUR_DOCUMENT_ID --cases evaluation/golden_cases.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app import build_engine
from backend.config import Settings
from backend.evaluation import RetrievalCase, evaluate_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local financial-document retrieval.")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/golden_cases.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [RetrievalCase(document_id=args.document_id, **{
        key: value for key, value in item.items() if key != "description"
    }) for item in raw_cases]
    engine = build_engine(Settings.from_environment())
    report = evaluate_retriever(engine.retriever, cases, args.top_k)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
