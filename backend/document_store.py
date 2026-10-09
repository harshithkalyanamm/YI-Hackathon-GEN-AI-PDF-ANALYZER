"""Storage boundary. Replace LocalDocumentStore with Azure storage without touching API logic."""
from __future__ import annotations

import re
import shutil
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class DocumentMetadata:
    original_filename: str
    document_type: str = "GENERAL"
    transaction_id: str | None = None


class DocumentStore(ABC):
    @abstractmethod
    def save_upload(self, source: Path, original_filename: str, document_type: str = "GENERAL",
                    transaction_id: str | None = None) -> str: ...

    @abstractmethod
    def upload_path(self, document_id: str) -> Path: ...

    @abstractmethod
    def index_dir(self, document_id: str) -> Path: ...

    @abstractmethod
    def exists(self, document_id: str) -> bool: ...

    @abstractmethod
    def original_filename(self, document_id: str) -> str: ...

    @abstractmethod
    def metadata(self, document_id: str) -> DocumentMetadata: ...


class LocalDocumentStore(DocumentStore):
    def __init__(self, uploads_dir: Path, indexes_dir: Path) -> None:
        self.uploads_dir, self.indexes_dir = uploads_dir, indexes_dir
        uploads_dir.mkdir(parents=True, exist_ok=True)
        indexes_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, source: Path, original_filename: str, document_type: str = "GENERAL",
                    transaction_id: str | None = None) -> str:
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_filename).stem).strip("-") or "document"
        document_id = f"{safe_stem[:48]}-{uuid4().hex[:12]}"
        shutil.copyfile(source, self.upload_path(document_id))
        directory = self.index_dir(document_id)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = DocumentMetadata(original_filename, document_type.upper(), transaction_id)
        (directory / "document.json").write_text(json.dumps(metadata.__dict__), encoding="utf-8")
        return document_id

    def upload_path(self, document_id: str) -> Path:
        return self.uploads_dir / f"{document_id}.pdf"

    def index_dir(self, document_id: str) -> Path:
        return self.indexes_dir / document_id

    def exists(self, document_id: str) -> bool:
        directory = self.index_dir(document_id)
        return self.upload_path(document_id).is_file() and (directory / "vectors.npy").is_file()

    def original_filename(self, document_id: str) -> str:
        return self.metadata(document_id).original_filename

    def metadata(self, document_id: str) -> DocumentMetadata:
        metadata_file = self.index_dir(document_id) / "document.json"
        if metadata_file.is_file():
            values = json.loads(metadata_file.read_text(encoding="utf-8"))
            return DocumentMetadata(
                original_filename=values.get("original_filename", self.upload_path(document_id).name),
                document_type=values.get("document_type", "GENERAL"),
                transaction_id=values.get("transaction_id"),
            )
        return DocumentMetadata(self.upload_path(document_id).name)
