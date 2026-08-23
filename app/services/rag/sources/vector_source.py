from typing import Any

from app.core.logging import get_logger
from app.db.repository import BaseVectorRepository
from app.services.document.models import MetadataFilter
from app.services.embedding.service import EmbeddingService
from app.services.rag.retriever import VectorRetriever
from app.services.rag.sources.base import BaseRetrievalSource
from app.services.rag.sources.models import SourceResult, SourceType

logger = get_logger("app.services.rag.sources.vector")


class VectorDatabaseSource(BaseRetrievalSource):
    """Retrieval source querying dense vector indices in PostgreSQL pgvector or in-memory."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: BaseVectorRepository,
        source_name: str = "internal_vector_db",
    ) -> None:
        self._retriever = VectorRetriever(
            embedding_service=embedding_service,
            vector_repository=vector_repository,
        )
        self._source_name = source_name

    @property
    def source_type(self) -> SourceType:
        return SourceType.INTERNAL_VECTOR

    @property
    def source_name(self) -> str:
        return self._source_name

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> list[SourceResult]:
        """Query vector database and return standardized SourceResult list."""
        citations = await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            min_similarity=min_relevance,
            filters=filters,
        )

        results: list[SourceResult] = []
        for citation in citations:
            meta = dict(citation.metadata)
            if citation.document_type is not None:
                meta["document_type"] = citation.document_type
            if citation.department is not None:
                meta["department"] = citation.department
            if citation.author is not None:
                meta["author"] = citation.author
            if citation.date is not None:
                meta["date"] = citation.date
            if citation.tags:
                meta["tags"] = citation.tags
            res = SourceResult(
                source=self.source_name,
                source_type=self.source_type,
                content=citation.content,
                relevance=round(citation.similarity, 4),
                metadata=meta,
                citation=citation,
            )
            results.append(res)

        logger.info(
            "vector_source_search_completed",
            source=self.source_name,
            query=query[:80],
            results_count=len(results),
        )
        return results
