"""Storage boundary. Replace LocalDocumentStore with Azure storage without touching API logic."""
from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4


class DocumentStore(ABC):
    @abstractmethod
    def save_upload(self, source: Path, original_filename: str) -> str: ...

    @abstractmethod
    def upload_path(self, document_id: str) -> Path: ...

    @abstractmethod
    def index_dir(self, document_id: str) -> Path: ...

    @abstractmethod
    def exists(self, document_id: str) -> bool: ...


class LocalDocumentStore(DocumentStore):
    def __init__(self, uploads_dir: Path, indexes_dir: Path) -> None:
        self.uploads_dir, self.indexes_dir = uploads_dir, indexes_dir
        uploads_dir.mkdir(parents=True, exist_ok=True)
        indexes_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, source: Path, original_filename: str) -> str:
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_filename).stem).strip("-") or "document"
        document_id = f"{safe_stem[:48]}-{uuid4().hex[:12]}"
        shutil.copyfile(source, self.upload_path(document_id))
        return document_id

    def upload_path(self, document_id: str) -> Path:
        return self.uploads_dir / f"{document_id}.pdf"

    def index_dir(self, document_id: str) -> Path:
        return self.indexes_dir / document_id

    def exists(self, document_id: str) -> bool:
        directory = self.index_dir(document_id)
        has_index = (directory / "index.faiss").is_file() or (directory / "vectors.npy").is_file()
        return self.upload_path(document_id).is_file() and has_index
