from pathlib import Path
from typing import Any

from app.services.document.loaders.base import BaseDocumentLoader
from app.services.document.loaders.pdf import PDFDocumentLoader
from app.services.document.loaders.text import TextDocumentLoader
from app.services.document.models import Document


class DocumentLoaderFactory:
    """Factory selecting the appropriate document loader based on file extension."""

    @staticmethod
    def get_loader(filename_or_path: str | Path) -> BaseDocumentLoader:
        """Inspect file extension and return matching document loader."""
        ext = Path(filename_or_path).suffix.lower()
        if ext == ".pdf":
            return PDFDocumentLoader()
        return TextDocumentLoader()

    @classmethod
    def load_bytes(
        cls,
        content_bytes: bytes,
        source_name: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Resolve loader by source name and extract Document."""
        loader = cls.get_loader(source_name)
        return loader.load_bytes(
            content_bytes=content_bytes,
            source_name=source_name,
            custom_metadata=custom_metadata,
        )

    @classmethod
    def load_file(
        cls,
        file_path: str | Path,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Resolve loader by path and extract Document from disk."""
        loader = cls.get_loader(file_path)
        return loader.load_file(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )
