import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.logging import get_logger
from app.services.rag.bm25 import tokenize
from app.services.rag.models import Citation

logger = get_logger("app.services.rag.reranker")


class RerankMeasurement(BaseModel):
    """Telemetry capturing rank position shift and score delta for a candidate chunk."""

    chunk_id: str = Field(description="Unique chunk identifier")
    source: str = Field(description="Source document name")
    initial_rank: int = Field(description="1-based ordinal rank before reranking")
    reranked_rank: int = Field(description="1-based ordinal rank after reranking")
    initial_score: float = Field(description="Similarity or RRF score before reranking")
    rerank_score: float = Field(description="Cross-encoder / relevance score assigned by reranker")
    rank_delta: int = Field(
        description="Position shift: positive = moved up in rank, negative = moved down"
    )


class RerankSummary(BaseModel):
    """Summary metrics of the reranking stage execution."""

    provider: str = Field(description="Reranker provider identifier (mock, cross-encoder)")
    candidates_count: int = Field(description="Number of candidates passed to reranker (Top N)")
    returned_count: int = Field(description="Number of results selected after reranking (Top K)")
    duration_ms: float = Field(description="Reranking latency in milliseconds")
    measurements: list[RerankMeasurement] = Field(
        default_factory=list,
        description="Detailed rank transition telemetry for each candidate",
    )


class BaseReranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        citations: list[Citation],
        top_k: int = 5,
    ) -> tuple[list[Citation], RerankSummary]:
        """Re-score candidate citations against query and return top-K ranked citations."""
        pass


class MockReranker(BaseReranker):
    """Deterministic mock reranker for testing and offline evaluation.

    Calculates cross-attention surrogate scores by evaluating:
    1. Exact query term containment frequency in chunk content.
    2. Query term containment in document metadata (source, tags, document_type).
    3. Position-weighted term matching.
    """

    def __init__(
        self,
        provider_name: str = "mock",
        simulate_failure: bool = False,
    ) -> None:
        self._provider_name = provider_name
        self._simulate_failure = simulate_failure

    async def rerank(
        self,
        query: str,
        citations: list[Citation],
        top_k: int = 5,
    ) -> tuple[list[Citation], RerankSummary]:
        start_time = time.perf_counter()

        if self._simulate_failure:
            raise RuntimeError("Simulated Reranker failure")

        if not citations:
            return [], RerankSummary(
                provider=self._provider_name,
                candidates_count=0,
                returned_count=0,
                duration_ms=0.0,
                measurements=[],
            )

        query_tokens = set(tokenize(query))
        scored_candidates: list[tuple[Citation, float, int]] = []

        # 1. Score each candidate based on term density and provenance
        for initial_rank, cite in enumerate(citations, start=1):
            content_tokens = tokenize(cite.content)
            matches = sum(1 for t in content_tokens if t in query_tokens)
            coverage = matches / (len(query_tokens) or 1.0)

            # Metadata relevance boost
            meta_boost = 0.0
            meta_str = f"{cite.source} {cite.document_type or ''} {' '.join(cite.tags)}".lower()
            for q_tok in query_tokens:
                if q_tok in meta_str:
                    meta_boost += 0.25

            # Deterministic cross-encoder surrogate score [0.0 - 1.0+]
            rerank_score = round(coverage * 0.7 + meta_boost + 0.1 * cite.similarity, 4)
            scored_candidates.append((cite, rerank_score, initial_rank))

        # 2. Sort candidates by rerank_score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Construct measurements and updated citations
        reranked_citations: list[Citation] = []
        measurements: list[RerankMeasurement] = []

        for new_rank, (cite, score, old_rank) in enumerate(scored_candidates[:top_k], start=1):
            delta = old_rank - new_rank  # positive if item moved up (e.g. rank 4 -> rank 1 => +3)
            measurement = RerankMeasurement(
                chunk_id=cite.chunk_id,
                source=cite.source,
                initial_rank=old_rank,
                reranked_rank=new_rank,
                initial_score=cite.similarity,
                rerank_score=score,
                rank_delta=delta,
            )
            measurements.append(measurement)

            updated_citation = cite.model_copy(
                update={
                    "similarity": score,
                    "initial_rank": old_rank,
                    "rerank_score": score,
                }
            )
            reranked_citations.append(updated_citation)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        summary = RerankSummary(
            provider=self._provider_name,
            candidates_count=len(citations),
            returned_count=len(reranked_citations),
            duration_ms=duration_ms,
            measurements=measurements,
        )

        logger.info(
            "rerank_completed",
            candidates=len(citations),
            top_k=len(reranked_citations),
            duration_ms=duration_ms,
        )
        return reranked_citations, summary


def create_reranker(
    provider: str = "mock",
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseReranker:
    """Factory returning configured BaseReranker implementation."""
    resolved_provider = provider.lower().strip()
    if resolved_provider in ("mock", "fake"):
        return MockReranker(provider_name=resolved_provider, **kwargs)
    else:
        raise ValueError(f"Unsupported reranker provider: '{provider}'")
