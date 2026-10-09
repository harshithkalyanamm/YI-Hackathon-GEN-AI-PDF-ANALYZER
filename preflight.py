"""Report whether optional local benchmark artifacts are ready; performs no downloads."""
from __future__ import annotations

import json
from pathlib import Path

from backend.config import Settings
from backend.preflight import check_docling, check_transformer_model


def main() -> None:
    settings = Settings.from_environment()
    print(json.dumps({
        "default_retrieval_model": check_transformer_model(settings.model_dir).to_dict(),
        "docling": check_docling().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
