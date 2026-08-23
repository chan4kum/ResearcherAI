"""Retrieval Source Abstraction package providing unified access across
heterogeneous knowledge sources.
"""

from app.services.rag.sources.base import BaseRetrievalSource
from app.services.rag.sources.keyword_source import KeywordSearchSource
from app.services.rag.sources.models import SourceResult, SourceType
from app.services.rag.sources.registry import RetrievalSourceRegistry
from app.services.rag.sources.structured_source import (
    StructuredDatabasePlaceholderSource,
)
from app.services.rag.sources.vector_source import VectorDatabaseSource
from app.services.rag.sources.web_source import WebSearchPlaceholderSource

__all__ = [
    "BaseRetrievalSource",
    "KeywordSearchSource",
    "RetrievalSourceRegistry",
    "SourceResult",
    "SourceType",
    "StructuredDatabasePlaceholderSource",
    "VectorDatabaseSource",
    "WebSearchPlaceholderSource",
]
