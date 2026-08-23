from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.services.rag.models import Citation


class SourceType(StrEnum):
    """Categorical source classification."""

    INTERNAL_VECTOR = "internal_vector"
    KEYWORD = "keyword"
    WEB_SEARCH = "web_search"
    STRUCTURED_DB = "structured_db"


class SourceResult(BaseModel):
    """Standardized retrieval result returned across all heterogeneous knowledge sources."""

    source: str = Field(
        description="Name or identifier of the retrieval source providing the result"
    )
    source_type: SourceType = Field(
        description="Categorical source type (internal_vector, keyword, web_search, structured_db)"
    )
    content: str = Field(
        description="Primary textual content or record snippet extracted from the source"
    )
    relevance: float = Field(
        default=0.0,
        description="Relevance or similarity score between 0.0 and 1.0",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata attributes and properties",
    )
    citation: Citation = Field(
        description="Structured citation provenance object for ground-truth verification"
    )
