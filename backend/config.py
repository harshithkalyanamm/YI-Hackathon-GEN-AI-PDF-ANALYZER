"""Configuration for the local-only financial document service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    uploads_dir: Path
    indexes_dir: Path
    model_dir: Path
    max_upload_bytes: int = 30 * 1024 * 1024
    chunk_words: int = 180
    chunk_overlap_words: int = 35
    top_k: int = 4
    retrieval_strategy: str = "dense"

    @classmethod
    def from_environment(cls) -> "Settings":
        root = Path(os.getenv("FINANCIAL_RISK_ROOT", Path(__file__).resolve().parents[1]))
        data_dir = Path(os.getenv("FINANCIAL_RISK_DATA_DIR", root / "data"))
        return cls(
            project_root=root,
            uploads_dir=data_dir / "uploads",
            indexes_dir=data_dir / "indexes",
            model_dir=Path(os.getenv("FINANCIAL_RISK_MODEL_DIR", root / "models" / "distilbert-base-uncased")),
            retrieval_strategy=os.getenv("FINANCIAL_RISK_RETRIEVAL_STRATEGY", "dense").lower(),
        )

    def ensure_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
