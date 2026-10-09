"""Offline readiness checks for locally staged parser/model benchmark artifacts."""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    kind: str
    ready: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_transformer_model(model_dir: Path) -> ArtifactCheck:
    required = ("config.json",)
    missing = [name for name in required if not (model_dir / name).is_file()]
    # A weights file can be either safe-tensors or a PyTorch binary.
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
    if not weights:
        missing.append("model weights (*.safetensors or pytorch_model*.bin)")
    return ArtifactCheck(str(model_dir), "transformer_model", not missing,
                         "ready" if not missing else "missing: " + ", ".join(missing))


def check_docling() -> ArtifactCheck:
    installed = importlib.util.find_spec("docling") is not None
    return ArtifactCheck("docling", "local_parser", installed,
                         "installed; configure it only after a benchmark" if installed else
                         "not installed (optional; PyMuPDF remains active)")
