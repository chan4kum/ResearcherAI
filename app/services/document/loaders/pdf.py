import io
from typing import Any

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None  # type: ignore[assignment]

from app.services.document.loaders.base import BaseDocumentLoader
from app.services.document.models import Document


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for extracting text and metadata from PDF files using pypdf."""

    def load_bytes(
        self,
        content_bytes: bytes,
        source_name: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Document:
        if pypdf is None:
            raise ImportError(
                "The 'pypdf' package is required to load and parse PDF files. "
                "Install it via 'pip install pypdf' or activate your project virtual environment."
            )

        stream = io.BytesIO(content_bytes)
        reader = pypdf.PdfReader(stream)

        page_texts: list[str] = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            page_texts.append(extracted.strip())

        full_text = "\n\n".join(page_texts).strip()

        metadata = self._create_metadata(
            source_name=source_name,
            file_type="pdf",
            content_bytes=content_bytes,
            extracted_text=full_text,
            page_count=len(reader.pages),
            custom_metadata=custom_metadata,
        )

        return Document(
            doc_id=metadata.doc_id,
            content=full_text,
            metadata=metadata,
        )
