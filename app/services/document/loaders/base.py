import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.services.document.models import Document, DocumentMetadata


def compute_checksum(content_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw bytes for deterministic content deduplication."""
    return hashlib.sha256(content_bytes).hexdigest()


class BaseDocumentLoader(ABC):
    """Abstract base class for document format extractors."""

    @abstractmethod
    def load_bytes(
        self,
        content_bytes: bytes,
        source_name: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Parse raw document bytes and return a structured Document model."""
        pass

    def load_file(
        self,
        file_path: str | Path,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Read a file from disk and parse into a Document model."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        content_bytes = path.read_bytes()
        return self.load_bytes(
            content_bytes=content_bytes,
            source_name=path.name,
            custom_metadata=custom_metadata,
        )

    def _create_metadata(
        self,
        source_name: str,
        file_type: str,
        content_bytes: bytes,
        extracted_text: str,
        page_count: int | None = None,
        custom_metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> DocumentMetadata:
        """Helper to construct standardized DocumentMetadata extracting known domain attributes."""
        words = extracted_text.split()
        meta_dict = custom_metadata.copy() if custom_metadata else {}
        doc_type = meta_dict.get("document_type")
        dept = meta_dict.get("department")
        doc_date = meta_dict.get("date")
        author = meta_dict.get("author")
        raw_tags = meta_dict.get("tags", [])
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags]
        else:
            tags = []

        return DocumentMetadata(
            doc_id=doc_id or str(uuid.uuid4()),
            source=source_name,
            file_type=file_type.lower().lstrip("."),
            file_size_bytes=len(content_bytes),
            checksum=compute_checksum(content_bytes),
            page_count=page_count,
            character_count=len(extracted_text),
            word_count=len(words),
            document_type=doc_type,
            department=dept,
            date=str(doc_date) if doc_date is not None else None,
            author=author,
            tags=tags,
            custom_metadata=meta_dict,
        )
