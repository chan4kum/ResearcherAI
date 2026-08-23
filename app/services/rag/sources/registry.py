import asyncio
from typing import Any

from app.core.logging import get_logger
from app.services.document.models import MetadataFilter
from app.services.rag.sources.base import BaseRetrievalSource
from app.services.rag.sources.models import SourceResult, SourceType

logger = get_logger("app.services.rag.sources.registry")


class RetrievalSourceRegistry:
    """Registry managing heterogeneous retrieval sources and multi-source query dispatch."""

    def __init__(self) -> None:
        self._sources_by_name: dict[str, BaseRetrievalSource] = {}
        self._sources_by_type: dict[SourceType, list[BaseRetrievalSource]] = {
            st: [] for st in SourceType
        }

    def register(self, source: BaseRetrievalSource) -> None:
        """Register a retrieval source instance."""
        self._sources_by_name[source.source_name] = source
        if source not in self._sources_by_type[source.source_type]:
            self._sources_by_type[source.source_type].append(source)
        logger.info(
            "retrieval_source_registered",
            source_name=source.source_name,
            source_type=source.source_type.value,
        )

    def get_source(self, name: str) -> BaseRetrievalSource | None:
        """Retrieve source by its unique name."""
        return self._sources_by_name.get(name)

    def get_sources_by_type(self, source_type: SourceType | str) -> list[BaseRetrievalSource]:
        """Retrieve all sources matching a given SourceType."""
        st = SourceType(source_type) if isinstance(source_type, str) else source_type
        return self._sources_by_type.get(st, [])

    def list_sources(self) -> list[BaseRetrievalSource]:
        """List all currently registered retrieval sources."""
        return list(self._sources_by_name.values())

    async def search_all(
        self,
        query: str,
        top_k_per_source: int = 3,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> dict[str, list[SourceResult]]:
        """Query all registered sources concurrently."""
        tasks = [
            source.search(
                query=query,
                top_k=top_k_per_source,
                filters=filters,
                min_relevance=min_relevance,
            )
            for source in self._sources_by_name.values()
        ]
        results = await asyncio.gather(*tasks)
        return {
            source.source_name: res
            for source, res in zip(self._sources_by_name.values(), results)
        }

    async def search_sources(
        self,
        query: str,
        source_names: list[str],
        top_k: int = 5,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> list[SourceResult]:
        """Query a selected subset of sources and combine results."""
        target_sources = [
            self._sources_by_name[name]
            for name in source_names
            if name in self._sources_by_name
        ]
        if not target_sources:
            return []

        tasks = [
            source.search(
                query=query,
                top_k=top_k,
                filters=filters,
                min_relevance=min_relevance,
            )
            for source in target_sources
        ]
        source_results = await asyncio.gather(*tasks)
        flattened = [item for sublist in source_results for item in sublist]
        flattened.sort(key=lambda r: r.relevance, reverse=True)
        return flattened[:top_k]
