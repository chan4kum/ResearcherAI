import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.rag.router import RetrievalRouter
from app.services.rag.routing import RoutingDecision, SourceDestination
from app.services.rag.sources import (
    RetrievalSourceRegistry,
    StructuredDatabasePlaceholderSource,
    VectorDatabaseSource,
    WebSearchPlaceholderSource,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_route_internal_policy_query(settings: Settings) -> None:
    """Verify internal policy query is routed exclusively to internal documents."""
    router = RetrievalRouter(settings=settings)
    query = "What does our internal policy say regarding travel reimbursement?"

    decision = await router.route(query)
    assert isinstance(decision, RoutingDecision)
    assert decision.query == query
    assert decision.selected_sources == [SourceDestination.INTERNAL_DOCUMENTS]
    assert "internal" in decision.reason.lower() or "policy" in decision.reason.lower()
    assert decision.confidence > 0.8


@pytest.mark.asyncio
async def test_route_external_market_query(settings: Settings) -> None:
    """Verify external query is routed to public web sources."""
    router = RetrievalRouter(settings=settings)
    query = "What happened in NVIDIA's latest earnings announcement and stock price?"

    decision = await router.route(query)
    assert isinstance(decision, RoutingDecision)
    assert decision.selected_sources == [SourceDestination.EXTERNAL_WEB]
    assert "external" in decision.reason.lower() or "market" in decision.reason.lower()


@pytest.mark.asyncio
async def test_route_comparative_internal_and_external_query(settings: Settings) -> None:
    """Verify compound query comparing internal and external data routes to multiple sources."""
    router = RetrievalRouter(settings=settings)
    query = "Compare our internal sales numbers with public market information."

    decision = await router.route(query)
    assert isinstance(decision, RoutingDecision)
    assert SourceDestination.INTERNAL_DOCUMENTS in decision.selected_sources
    assert SourceDestination.EXTERNAL_WEB in decision.selected_sources
    reason_low = decision.reason.lower()
    assert "comparative" in reason_low or "cross-referencing" in reason_low


@pytest.mark.asyncio
async def test_route_structured_database_query(settings: Settings) -> None:
    """Verify structured tabular queries route to structured database."""
    router = RetrievalRouter(settings=settings)
    query = "Show aircraft maintenance log and inspection records for tail number N779XX."

    decision = await router.route(query)
    assert isinstance(decision, RoutingDecision)
    assert decision.selected_sources == [SourceDestination.STRUCTURED_DATABASE]
    assert "tabular" in decision.reason.lower() or "maintenance" in decision.reason.lower()


@pytest.mark.asyncio
async def test_route_empty_query_raises_value_error(settings: Settings) -> None:
    """Verify empty or whitespace query raises ValueError."""
    router = RetrievalRouter(settings=settings)
    with pytest.raises(ValueError, match="cannot be empty"):
        await router.route("   ")


@pytest.mark.asyncio
async def test_route_and_retrieve_end_to_end(settings: Settings) -> None:
    """Verify router selects sources and dispatches search via registry."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()
    emb_service = EmbeddingService(settings=settings)

    # Populate vector repo
    meta = ChunkMetadata(
        chunk_id="chk_internal_1",
        doc_id="doc_internal_1",
        index=0,
        start_char=0,
        end_char=80,
        character_count=80,
        word_count=10,
        source="internal_quality_guide.txt",
        file_type="txt",
        checksum="c1",
        document_type="guideline",
    )
    content = "Internal Boeing quality standard: Flutter damper inspection is mandatory."
    emb = await emb_service.embed_text(content)
    repo._chunks["chk_internal_1"] = EmbeddedChunk(
        chunk_id="chk_internal_1",
        doc_id="doc_internal_1",
        content=content,
        embedding=emb,
        metadata=meta,
    )

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    src_web = WebSearchPlaceholderSource(source_name="web_engine")
    src_db = StructuredDatabasePlaceholderSource(source_name="relational_db")

    registry.register(src_vector)
    registry.register(src_web)
    registry.register(src_db)

    router = RetrievalRouter(registry=registry, settings=settings)

    # Query internal
    decision, results = await router.route_and_retrieve(
        query="What does our internal quality policy say about flutter damper?",
        top_k_per_source=2,
        min_relevance=-1.0,
    )
    assert decision.selected_sources == [SourceDestination.INTERNAL_DOCUMENTS]
    assert len(results) >= 1
    assert any(r.source == "vector_db" for r in results)


@pytest.mark.asyncio
async def test_route_query_endpoint(settings: Settings) -> None:
    """Verify POST /api/v1/rag/route-query returns structured routing response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {"query": "What happened in NVIDIA's latest earnings report?"}
        response = await client.post("/api/v1/rag/route-query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert "external_web" in data["selected_sources"]
        assert len(data["reason"]) > 5
        assert data["confidence"] > 0.5


@pytest.mark.asyncio
async def test_route_query_endpoint_validation_error() -> None:
    """Verify POST /api/v1/rag/route-query returns 422 for empty query."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {"query": ""}
        response = await client.post("/api/v1/rag/route-query", json=payload)
        assert response.status_code == 422
