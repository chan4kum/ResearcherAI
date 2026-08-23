import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IngestStatus(StrEnum):
    """Status indicating the outcome of a document ingestion attempt."""

    INGESTED = "ingested"
    SKIPPED_DUPLICATE = "skipped_duplicate"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Metadata tracking provenance, format, size, checksum, domain categorization, and metrics."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(description="Original filename or path of the document")
    file_type: str = Field(description="File extension or format (e.g. 'txt', 'pdf', 'md')")
    file_size_bytes: int = Field(default=0, description="Raw content size in bytes")
    checksum: str = Field(description="SHA-256 hash of raw document content for deduplication")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    page_count: int | None = Field(default=None, description="Total pages if paginated (e.g. PDF)")
    character_count: int = Field(default=0, description="Total extracted characters")
    word_count: int = Field(default=0, description="Total extracted word count")

    # Milestone 10: Metadata-Aware Attributes
    document_type: str | None = Field(
        default=None,
        description="Semantic document type classification (e.g. 'quality_report', 'spec', 'memo')",
    )
    department: str | None = Field(
        default=None,
        description="Originating department or business unit (e.g. 'qa', 'engineering', 'finance')",
    )
    date: str | None = Field(
        default=None,
        description="Document date string (e.g. '2026-08-23')",
    )
    author: str | None = Field(
        default=None,
        description="Document author or creator",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Domain and categorization tags",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata tags",
    )


class MetadataFilter(BaseModel):
    """Structured criteria for filtering retrieved documents and chunks by metadata."""

    source: str | None = Field(default=None, description="Exact or case-insensitive source match")
    document_type: str | None = Field(
        default=None, description="Filter by document type (e.g. 'quality_report')"
    )
    department: str | None = Field(
        default=None, description="Filter by department (e.g. 'engineering')"
    )
    date: str | None = Field(default=None, description="Filter by date string")
    author: str | None = Field(default=None, description="Filter by author name")
    tags: list[str] | None = Field(
        default=None,
        description="Filter by tags (matches if chunk/document contains any of the requested tags)",
    )
    custom_metadata: dict[str, Any] | None = Field(
        default=None, description="Filter by custom key-value metadata pairs"
    )

    def matches(self, meta: Any) -> bool:
        """Evaluate whether a ChunkMetadata or DocumentMetadata satisfies filter criteria."""
        if not isinstance(meta, (ChunkMetadata, DocumentMetadata)):
            return True

        # 1. Source filter
        if self.source is not None:
            if meta.source.lower().strip() != self.source.lower().strip():
                return False

        # 2. Document type filter
        if self.document_type is not None:
            doc_type = getattr(meta, "document_type", None) or meta.custom_metadata.get(
                "document_type"
            )
            if (
                doc_type is None
                or str(doc_type).lower().strip() != self.document_type.lower().strip()
            ):
                return False

        # 3. Department filter
        if self.department is not None:
            dept = getattr(meta, "department", None) or meta.custom_metadata.get("department")
            if dept is None or str(dept).lower().strip() != self.department.lower().strip():
                return False

        # 4. Date filter
        if self.date is not None:
            d = getattr(meta, "date", None) or meta.custom_metadata.get("date")
            if d is None or str(d).lower().strip() != self.date.lower().strip():
                return False

        # 5. Author filter
        if self.author is not None:
            auth = getattr(meta, "author", None) or meta.custom_metadata.get("author")
            if auth is None or str(auth).lower().strip() != self.author.lower().strip():
                return False

        # 6. Tags filter (matches if any requested tag is present in meta tags or custom_metadata)
        if self.tags is not None and len(self.tags) > 0:
            target_tags = [t.lower().strip() for t in getattr(meta, "tags", [])]
            if isinstance(meta.custom_metadata.get("tags"), list):
                target_tags.extend(
                    [str(t).lower().strip() for t in meta.custom_metadata["tags"]]
                )
            required_tags = [t.lower().strip() for t in self.tags]
            if not any(req in target_tags for req in required_tags):
                return False

        # 7. Custom metadata key-value filters
        if self.custom_metadata:
            for k, v in self.custom_metadata.items():
                if k not in meta.custom_metadata or meta.custom_metadata[k] != v:
                    return False

        return True


def normalize_metadata_filter(
    filters: MetadataFilter | dict[str, Any] | None,
) -> MetadataFilter | None:
    """Safely convert a dict or MetadataFilter instance into a validated MetadataFilter object."""
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        return filters
    if isinstance(filters, dict):
        if not filters:
            return None
        known_keys = {
            "source",
            "document_type",
            "department",
            "date",
            "author",
            "tags",
            "custom_metadata",
        }
        known_kwargs: dict[str, Any] = {}
        custom_kwargs: dict[str, Any] = {}
        for k, v in filters.items():
            if k in known_keys:
                known_kwargs[k] = v
            else:
                custom_kwargs[k] = v
        if custom_kwargs:
            merged_custom = known_kwargs.get("custom_metadata", {}) or {}
            merged_custom.update(custom_kwargs)
            known_kwargs["custom_metadata"] = merged_custom
        return MetadataFilter(**known_kwargs)
    raise ValueError(
        f"Invalid filter format: expected dict or MetadataFilter, got {type(filters).__name__}"
    )


class Document(BaseModel):
    """Structured representation of an ingested document with content and metadata."""

    doc_id: str
    content: str
    metadata: DocumentMetadata


class IngestResult(BaseModel):
    """Structured response detailing the result of an ingestion operation."""

    doc_id: str
    status: IngestStatus
    source: str
    checksum: str
    metadata: DocumentMetadata | None = None
    message: str = ""


class ChunkingConfig(BaseModel):
    """Configuration parameters for document text chunking."""

    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Target maximum number of characters per chunk",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Number of overlapping characters between consecutive chunks",
    )

    def validate_overlap(self) -> None:
        """Ensure overlap is strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )


class ChunkMetadata(BaseModel):
    """Metadata tracking chunk position, domain attributes, and parent properties."""

    chunk_id: str = Field(
        description="Deterministic chunk identifier (e.g., {doc_id}_chunk_{index})"
    )
    doc_id: str = Field(description="Parent document identifier")
    index: int = Field(description="0-indexed position of the chunk within the document")
    start_char: int = Field(description="Start character offset in parent document")
    end_char: int = Field(description="End character offset in parent document")
    character_count: int = Field(description="Length of chunk content in characters")
    word_count: int = Field(description="Word count of chunk content")
    source: str = Field(description="Parent document source filename")
    file_type: str = Field(description="Parent document file extension")
    checksum: str = Field(description="Parent document SHA-256 checksum")

    # Milestone 10: Metadata-Aware Attributes Preserved from Document
    document_type: str | None = Field(default=None, description="Semantic document type")
    department: str | None = Field(default=None, description="Originating department")
    date: str | None = Field(default=None, description="Document date string")
    author: str | None = Field(default=None, description="Document author")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved custom metadata from parent document",
    )


class DocumentChunk(BaseModel):
    """A single chunk unit of a document with preserved metadata."""

    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document identifier")
    content: str = Field(description="Text content of the chunk")
    metadata: ChunkMetadata = Field(description="Chunk metadata envelope")
    embedding: list[float] | None = Field(default=None, description="Optional vector embedding")


class EmbeddedChunk(BaseModel):
    """DocumentChunk enriched with vector embedding representation."""

    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document identifier")
    content: str = Field(description="Text content of the chunk")
    embedding: list[float] = Field(description="Dense vector embedding representation")
    metadata: ChunkMetadata = Field(description="Chunk metadata envelope")


