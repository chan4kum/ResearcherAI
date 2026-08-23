from abc import ABC, abstractmethod
from typing import Any

from app.services.document.models import MetadataFilter
from app.services.rag.sources.models import SourceResult, SourceType


class BaseRetrievalSource(ABC):
    """Abstract base interface defining the contract for all retrieval sources."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Categorical source type."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable identifier of the retrieval source."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> list[SourceResult]:
        """Search the underlying source and return standardized SourceResult items."""
        pass
