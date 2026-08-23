import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repository import BaseVectorRepository
from app.services.document.models import MetadataFilter, normalize_metadata_filter
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.bm25 import BM25Index
from app.services.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.services.rag.hyde import HyDEGenerator, HyDEResult
from app.services.rag.models import Citation

logger = get_logger("app.services.rag.retriever")


class BaseRetriever(ABC):
    """Abstract interface for knowledge retrieval components with optional metadata filtering."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Retrieve relevant knowledge chunks and convert to structured citations."""
        pass


class VectorRetriever(BaseRetriever):
    """Dense vector similarity retriever utilizing EmbeddingService and BaseVectorRepository."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: BaseVectorRepository,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Generate query vector embedding and retrieve top-K matching chunks."""
        clean_query = query.strip()
        if not clean_query:
            return []

        filter_obj = normalize_metadata_filter(filters)

        # 1. Embed query
        query_embedding = await self._embedding_service.embed_text(clean_query)

        # 2. Vector search in database with metadata filtering
        chunk_results = await self._vector_repository.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filter_obj,
        )

        # 3. Transform to Citations with verified provenance and domain metadata
        citations: list[Citation] = []
        for chunk, similarity in chunk_results:
            citation = Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.metadata.source,
                file_type=chunk.metadata.file_type,
                chunk_index=chunk.metadata.index,
                content=chunk.content,
                similarity=similarity,
                document_type=chunk.metadata.document_type,
                department=chunk.metadata.department,
                date=chunk.metadata.date,
                author=chunk.metadata.author,
                tags=chunk.metadata.tags,
                metadata=chunk.metadata.custom_metadata,
            )
            citations.append(citation)

        logger.info(
            "vector_retrieval_completed",
            query=clean_query[:80],
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filter_obj.model_dump(exclude_none=True) if filter_obj else None,
            results_found=len(citations),
        )
        return citations


class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings (HyDE) retriever.

    Flow:
    User Query -> LLM generates hypothetical passage -> Embed passage -> Vector search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: BaseVectorRepository,
        hyde_generator: HyDEGenerator | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_repository = vector_repository
        self._settings = settings or get_settings()
        self._hyde_generator = hyde_generator or HyDEGenerator(
            llm_service=llm_service, settings=self._settings
        )
        self._last_result: HyDEResult | None = None

    @property
    def hyde_generator(self) -> HyDEGenerator:
        return self._hyde_generator

    @property
    def last_result(self) -> HyDEResult | None:
        return self._last_result

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Generate hypothetical passage, embed it, and search real documents."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Generate hypothetical passage
        hypothetical_doc = await self._hyde_generator.generate(clean_query)

        # 2. Embed hypothetical passage
        hypo_embedding = await self._embedding_service.embed_text(hypothetical_doc)

        filter_obj = normalize_metadata_filter(filters)

        # 3. Vector search in real documents using hypothetical embedding
        chunk_results = await self._vector_repository.search_similar_chunks(
            query_embedding=hypo_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filter_obj,
        )

        # 4. Map to Citations
        citations: list[Citation] = []
        for chunk, similarity in chunk_results:
            citation = Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.metadata.source,
                file_type=chunk.metadata.file_type,
                chunk_index=chunk.metadata.index,
                content=chunk.content,
                similarity=similarity,
                document_type=chunk.metadata.document_type,
                department=chunk.metadata.department,
                date=chunk.metadata.date,
                author=chunk.metadata.author,
                tags=chunk.metadata.tags,
                metadata={
                    **chunk.metadata.custom_metadata,
                    "retrieval_strategy": "hyde",
                },
            )
            citations.append(citation)

        self._last_result = HyDEResult(
            original_query=clean_query,
            hypothetical_document=hypothetical_doc,
            strategy="hyde",
            retrieved_documents_count=len(citations),
        )

        logger.info(
            "hyde_retrieval_completed",
            query=clean_query[:80],
            hypothetical_doc_len=len(hypothetical_doc),
            results_found=len(citations),
        )
        return citations


class BM25Retriever(BaseRetriever):
    """Lexical keyword retriever utilizing the Okapi BM25 ranking algorithm."""

    def __init__(
        self,
        vector_repository: BaseVectorRepository,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._vector_repository = vector_repository
        self._k1 = k1
        self._b = b

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Score candidate chunks against query terms using the Okapi BM25 formula."""
        clean_query = query.strip()
        if not clean_query:
            return []

        filter_obj = normalize_metadata_filter(filters)

        # 1. Fetch candidate chunks from repository
        chunks = await self._vector_repository.list_chunks(filters=filter_obj)
        if not chunks:
            return []

        # 2. Build in-memory index on active candidates
        index = BM25Index(k1=self._k1, b=self._b)
        index.build_index(chunks=chunks, filters=filter_obj)

        # 3. Score chunks
        scored_chunks = index.score(query=clean_query, top_k=top_k, min_score=min_similarity)

        # 4. Map to Citations
        citations: list[Citation] = []
        for chunk, score in scored_chunks:
            citation = Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.metadata.source,
                file_type=chunk.metadata.file_type,
                chunk_index=chunk.metadata.index,
                content=chunk.content,
                similarity=score,
                document_type=chunk.metadata.document_type,
                department=chunk.metadata.department,
                date=chunk.metadata.date,
                author=chunk.metadata.author,
                tags=chunk.metadata.tags,
                metadata=chunk.metadata.custom_metadata,
            )
            citations.append(citation)

        logger.info(
            "bm25_retrieval_completed",
            query=clean_query[:80],
            top_k=top_k,
            min_score=min_similarity,
            results_found=len(citations),
        )
        return citations


class HybridRetriever(BaseRetriever):
    """Hybrid retriever merging dense semantic search and sparse lexical BM25 search."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        fusion_strategy: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._fusion_strategy = fusion_strategy.lower().strip()
        self._rrf_k = rrf_k
        self._alpha = alpha

    @property
    def vector_retriever(self) -> VectorRetriever:
        return self._vector_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        return self._bm25_retriever

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Execute dense and sparse searches concurrently and fuse ranked result lists."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # Retrieve a broader candidate pool (2x top_k) from each modality for optimal fusion
        candidate_k = max(top_k * 2, 10)

        dense_task = self._vector_retriever.retrieve(
            query=clean_query,
            top_k=candidate_k,
            min_similarity=min_similarity,
            filters=filters,
        )
        sparse_task = self._bm25_retriever.retrieve(
            query=clean_query,
            top_k=candidate_k,
            min_similarity=0.0,
            filters=filters,
        )

        dense_citations, sparse_citations = await asyncio.gather(dense_task, sparse_task)

        if self._fusion_strategy == "weighted":
            fused = weighted_score_fusion(
                dense_citations=dense_citations,
                sparse_citations=sparse_citations,
                alpha=self._alpha,
                top_k=top_k,
            )
        else:  # default to RRF (Reciprocal Rank Fusion)
            fused = reciprocal_rank_fusion(
                dense_citations=dense_citations,
                sparse_citations=sparse_citations,
                k=self._rrf_k,
                top_k=top_k,
            )

        logger.info(
            "hybrid_retrieval_completed",
            query=clean_query[:80],
            fusion_strategy=self._fusion_strategy,
            dense_candidates=len(dense_citations),
            sparse_candidates=len(sparse_citations),
            final_fused_count=len(fused),
        )
        return fused


def create_retriever(
    embedding_service: EmbeddingService,
    vector_repository: BaseVectorRepository,
    mode: str = "hybrid",
    strategy: str = "normal",
    settings: Settings | None = None,
    fusion_strategy: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
    llm_service: LLMService | None = None,
) -> BaseRetriever:
    """Factory creating an interchangeable Retriever implementation."""
    current_settings = settings or get_settings()
    resolved_mode = mode.lower().strip()
    resolved_strategy = strategy.lower().strip()

    if resolved_strategy == "hyde" or resolved_mode == "hyde":
        return HyDERetriever(
            embedding_service=embedding_service,
            vector_repository=vector_repository,
            llm_service=llm_service,
            settings=current_settings,
        )

    vector_retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_repository=vector_repository,
    )
    bm25_retriever = BM25Retriever(
        vector_repository=vector_repository,
    )

    if resolved_mode in ("semantic", "vector"):
        return vector_retriever
    elif resolved_mode in ("keyword", "bm25", "lexical"):
        return bm25_retriever
    elif resolved_mode == "hybrid":
        resolved_k = rrf_k or getattr(current_settings, "hybrid_rrf_k", 60)
        resolved_alpha = (
            alpha if alpha is not None else getattr(current_settings, "hybrid_alpha", 0.5)
        )
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy=fusion_strategy,
            rrf_k=resolved_k,
            alpha=resolved_alpha,
        )
    else:
        raise ValueError(
            f"Unsupported retrieval mode: '{mode}'. "
            "Must be 'hybrid', 'semantic', 'keyword', or 'hyde'."
        )
