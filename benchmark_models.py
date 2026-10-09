"""Offline benchmark for approved, already-staged retrieval models.

No packages or model files are downloaded. Each supplied directory must contain
the model artifacts before this command is run.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from backend.app import build_engine
from backend.config import Settings
from backend.document_store import LocalDocumentStore
from backend.evaluation import RetrievalCase, evaluate_retriever
from backend.model_manager import LocalDistilBERTEmbedder
from backend.preflight import check_transformer_model
from backend.retriever import LocalNumpyRetriever


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=LOCAL_MODEL_DIRECTORY.")
    label, directory = value.split("=", 1)
    return label, Path(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark staged local retrieval encoders.")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--cases", type=Path, default=Path("evaluation/golden_cases.json"))
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="Repeat as LABEL=LOCAL_MODEL_DIRECTORY")
    parser.add_argument("--strategy", choices=("dense", "bm25", "hybrid"), default="dense")
    args = parser.parse_args()

    settings = Settings.from_environment()
    baseline = build_engine(settings)
    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [RetrievalCase(document_id=args.document_id, **{
        key: value for key, value in item.items() if key != "description"
    }) for item in raw_cases]
    # Load stored chunks once. Each candidate computes new local embeddings from
    # the same material; no document leaves this process.
    _, chunks = baseline.retriever._load(args.document_id)
    results: dict[str, dict] = {}
    for label, model_dir in args.model:
        check = check_transformer_model(model_dir)
        if not check.ready:
            results[label] = {"ready": False, "detail": check.detail}
            continue
        with tempfile.TemporaryDirectory(prefix="financial-risk-benchmark-") as temporary:
            scratch = Path(temporary)
            # Candidate vectors are deliberately disposable: an evaluation must
            # never replace the production index for the uploaded PDF.
            benchmark_store = LocalDocumentStore(scratch / "uploads", scratch / "indexes")
            embedder = LocalDistilBERTEmbedder(model_dir)
            retriever = LocalNumpyRetriever(benchmark_store, embedder, args.strategy)
            retriever.build(args.document_id, chunks)
            results[label] = {"ready": True, **evaluate_retriever(retriever, cases).to_dict()}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
