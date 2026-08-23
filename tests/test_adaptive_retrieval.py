import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.rag.adaptive import (
    AdaptiveRetriever,
    EvidenceEvaluation,
    EvidenceEvaluator,
    EvidenceSufficiencyStatus,
)
from app.services.rag.models import Citation
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent
from app.services.rag.router import RetrievalRouter
from app.services.rag.routing import RoutingDecision, SourceDestination
from app.services.rag.sources import (
    RetrievalSourceRegistry,
    SourceResult,
    SourceType,
    VectorDatabaseSource,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_evidence_evaluator_sufficient_case() -> None:
    """Verify evidence evaluator marks high-relevance, full-coverage evidence as sufficient."""
    evaluator = EvidenceEvaluator(min_relevance_threshold=0.4, min_coverage_threshold=0.5)

    analysis = QueryAnalysis(
        original_query="What does Boeing flutter damper procedure require?",
        intent=QueryIntent.FACTUAL,
        entities=[],
        confidence_score=0.9,
    )
    routing = RoutingDecision(
        query=analysis.original_query,
        intent=analysis.intent,
        selected_sources=[SourceDestination.INTERNAL_DOCUMENTS],
        reason="Targeting internal technical documentation.",
        confidence=0.95,
    )

    citation = Citation(
        chunk_id="chk_1",
        doc_id="doc_1",
        source="boeing_doc.txt",
        file_type="txt",
        chunk_index=0,
        content="Boeing flutter damper procedure requires ultrasonic torque inspection.",
        similarity=0.88,
    )
    result = SourceResult(
        source="internal_kb",
        source_type=SourceType.INTERNAL_VECTOR,
        content=citation.content,
        relevance=0.88,
        citation=citation,
    )

    evaluation = evaluator.evaluate(
        query=analysis.original_query,
        analysis=analysis,
        routing=routing,
        results=[result],
    )

    assert isinstance(evaluation, EvidenceEvaluation)
    assert evaluation.is_sufficient is True
    assert evaluation.status == EvidenceSufficiencyStatus.SUFFICIENT
    assert evaluation.relevance_score >= 0.8
    assert evaluation.confidence >= 0.6


@pytest.mark.asyncio
async def test_evidence_evaluator_insufficient_empty_case() -> None:
    """Verify empty evidence results in needs_more_retrieval."""
    evaluator = EvidenceEvaluator()

    analysis = QueryAnalysis(
        original_query="What are the details of the secret prototype?",
        intent=QueryIntent.FACTUAL,
        entities=[],
        confidence_score=0.9,
    )
    routing = RoutingDecision(
        query=analysis.original_query,
        intent=analysis.intent,
        selected_sources=[SourceDestination.INTERNAL_DOCUMENTS],
        reason="Targeting internal docs.",
        confidence=0.9,
    )

    evaluation = evaluator.evaluate(
        query=analysis.original_query,
        analysis=analysis,
        routing=routing,
        results=[],
    )

    assert evaluation.is_sufficient is False
    assert evaluation.status == EvidenceSufficiencyStatus.NEEDS_MORE_RETRIEVAL
    assert evaluation.evidence_quantity == 0
    assert "No evidence retrieved" in evaluation.reason


@pytest.mark.asyncio
async def test_adaptive_retriever_sufficient_pipeline(settings: Settings) -> None:
    """Verify end-to-end adaptive retrieval generates answers when evidence is sufficient."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()
    emb_service = EmbeddingService(settings=settings)

    content = "Internal Boeing standard: Ultrasonic flutter damper torque verification is required."
    emb = await emb_service.embed_text(content)
    meta = ChunkMetadata(
        chunk_id="chk_adaptive_1",
        doc_id="doc_adaptive_1",
        index=0,
        start_char=0,
        end_char=80,
        character_count=80,
        word_count=10,
        source="quality_manual.txt",
        file_type="txt",
        checksum="c_ad_1",
        document_type="manual",
    )
    repo._chunks["chk_adaptive_1"] = EmbeddedChunk(
        chunk_id="chk_adaptive_1",
        doc_id="doc_adaptive_1",
        content=content,
        embedding=emb,
        metadata=meta,
    )

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    registry.register(src_vector)

    router = RetrievalRouter(registry=registry, settings=settings)
    adaptive = AdaptiveRetriever(router=router, settings=settings)

    result = await adaptive.retrieve_adaptively(
        query="What is the internal Boeing requirement for flutter damper?",
        max_rounds=2,
        generate_answer=True,
    )

    assert result.status == EvidenceSufficiencyStatus.SUFFICIENT
    assert result.rounds_executed == 1
    assert len(result.results) >= 1
    assert result.answer is not None
    assert len(result.answer) > 5


@pytest.mark.asyncio
async def test_adaptive_retriever_insufficient_bounded_loop(settings: Settings) -> None:
    """Verify adaptive retriever respects max_rounds limits when evidence is missing."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()  # Empty repository

    emb_service = EmbeddingService(settings=settings)
    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    registry.register(src_vector)

    router = RetrievalRouter(registry=registry, settings=settings)
    adaptive = AdaptiveRetriever(router=router, settings=settings)

    result = await adaptive.retrieve_adaptively(
        query="What is the proprietary engine spec for Project Titan 99?",
        max_rounds=2,
        generate_answer=True,
    )

    assert result.status == EvidenceSufficiencyStatus.NEEDS_MORE_RETRIEVAL
    assert result.rounds_executed == 2
    assert result.max_rounds == 2
    assert result.evaluation.is_sufficient is False
    assert result.answer is None


@pytest.mark.asyncio
async def test_adaptive_retrieve_endpoint() -> None:
    """Verify POST /api/v1/rag/adaptive-retrieve returns structured 200 response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "query": "What happened in NVIDIA's latest earnings report?",
            "max_rounds": 2,
            "generate_answer": True,
        }
        response = await client.post("/api/v1/rag/adaptive-retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert data["status"] in ("sufficient", "needs_more_retrieval")
        assert "evaluation" in data
        assert data["rounds_executed"] >= 1


@pytest.mark.asyncio
async def test_adaptive_retrieve_endpoint_validation_error() -> None:
    """Verify POST /api/v1/rag/adaptive-retrieve returns 422 for empty query."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {"query": ""}
        response = await client.post("/api/v1/rag/adaptive-retrieve", json=payload)
        assert response.status_code == 422
