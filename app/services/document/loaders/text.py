from pathlib import Path
from typing import Any

from app.services.document.loaders.base import BaseDocumentLoader
from app.services.document.models import Document


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for extracting plain text and markdown documents."""

    def load_bytes(
        self,
        content_bytes: bytes,
        source_name: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        # Decode with utf-8 first, fallback to latin-1
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1", errors="replace")

        ext = Path(source_name).suffix.lower() or ".txt"
        metadata = self._create_metadata(
            source_name=source_name,
            file_type=ext,
            content_bytes=content_bytes,
            extracted_text=text,
            page_count=1,
            custom_metadata=custom_metadata,
        )

        return Document(
            doc_id=metadata.doc_id,
            content=text,
            metadata=metadata,
        )
