"""Document ingestion package for text, PDF, and local document extraction."""

from app.services.document.chunker import DocumentChunker, display_document_chunks
from app.services.document.loaders.base import BaseDocumentLoader, compute_checksum
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.loaders.pdf import PDFDocumentLoader
from app.services.document.loaders.text import TextDocumentLoader
from app.services.document.models import (
    ChunkingConfig,
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmbeddedChunk,
    IngestResult,
    IngestStatus,
    MetadataFilter,
    normalize_metadata_filter,
)
from app.services.document.store import DocumentStore

__all__ = [
    "BaseDocumentLoader",
    "ChunkMetadata",
    "ChunkingConfig",
    "Document",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentLoaderFactory",
    "DocumentMetadata",
    "DocumentStore",
    "EmbeddedChunk",
    "IngestResult",
    "IngestStatus",
    "MetadataFilter",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "compute_checksum",
    "display_document_chunks",
    "normalize_metadata_filter",
]
