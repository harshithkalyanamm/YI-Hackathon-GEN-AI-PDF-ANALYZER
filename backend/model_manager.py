"""One process-wide, local DistilBERT embedding model. No hosted inference is used."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalDistilBERTEmbedder:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self._torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not (self.model_dir / "config.json").is_file():
            raise RuntimeError(
                f"Local model missing at {self.model_dir}. Run the model setup command in README before starting the API."
            )
        # Delayed imports keep non-model management commands and test doubles lightweight.
        # The modules are still loaded exactly once when this production embedder starts.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        assert self.tokenizer is not None and self.model is not None and self._torch is not None
        rows: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = self.tokenizer(texts[start:start + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
                output = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                rows.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(rows)
