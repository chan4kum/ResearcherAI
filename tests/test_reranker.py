import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import create_app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.models import Citation
from app.services.rag.reranker import (
    BaseReranker,
    MockReranker,
    RerankMeasurement,
    create_reranker,
)
from app.services.rag.retriever import VectorRetriever
from app.services.rag.service import RAGService
from httpx import ASGITransport, AsyncClient


def make_citation(
    chunk_id: str,
    doc_id: str,
    source: str,
    content: str,
    similarity: float,
    document_type: str | None = None,
    tags: list[str] | None = None,
) -> Citation:
    """Helper creating Citation object for reranker testing."""
    return Citation(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=source,
        file_type="txt",
        chunk_index=0,
        content=content,
        similarity=similarity,
        document_type=document_type,
        tags=tags or [],
        metadata={},
    )


@pytest.mark.asyncio
async def test_mock_reranker_reorders_candidates_and_calculates_rank_delta() -> None:
    """Verify that MockReranker promotes high-relevance candidates and records rank deltas."""
    reranker = MockReranker()

    # Candidate 1: High initial similarity (0.95), but generic content
    c1 = make_citation(
        chunk_id="c1",
        doc_id="d1",
        source="general_manual.txt",
        content="General aircraft operational maintenance instructions and safety guidelines.",
        similarity=0.95,
        document_type="manual",
    )

    # Candidate 2: Moderate similarity (0.80), partial relevance
    c2 = make_citation(
        chunk_id="c2",
        doc_id="d2",
        source="engine_spec.txt",
        content="Turbofan engine core temperature telemetry monitors bypass pressure.",
        similarity=0.80,
        document_type="spec",
    )

    # Candidate 3: Lower initial similarity (0.65), but exact match for query terms and metadata
    c3 = make_citation(
        chunk_id="c3",
        doc_id="d3",
        source="titanium_compressor_failure_audit.txt",
        content=(
            "Titanium high-pressure compressor rotor blade fatigue "
            "micro-crack failure analysis."
        ),
        similarity=0.65,
        document_type="failure_audit",
        tags=["titanium", "compressor", "fatigue"],
    )

    query = "titanium compressor blade fatigue failure audit"
    candidates = [c1, c2, c3]

    reranked, summary = await reranker.rerank(query=query, citations=candidates, top_k=2)

    # Candidate 3 should be promoted from Initial Rank 3 to Reranked Rank 1
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c3"
    assert reranked[0].initial_rank == 3
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score

    # Check telemetry
    assert summary.candidates_count == 3
    assert summary.returned_count == 2
    assert len(summary.measurements) == 2

    m0: RerankMeasurement = summary.measurements[0]
    assert m0.chunk_id == "c3"
    assert m0.initial_rank == 3
    assert m0.reranked_rank == 1
    assert m0.rank_delta == 2  # Moved up 2 positions


@pytest.mark.asyncio
async def test_mock_reranker_empty_candidates() -> None:
    """Reranker handles empty candidate lists gracefully."""
    reranker = MockReranker()
    reranked, summary = await reranker.rerank(query="anything", citations=[], top_k=5)
    assert reranked == []
    assert summary.candidates_count == 0
    assert summary.returned_count == 0
    assert summary.measurements == []


@pytest.mark.asyncio
async def test_mock_reranker_simulated_failure() -> None:
    """Reranker raises RuntimeError when simulate_failure=True."""
    reranker = MockReranker(simulate_failure=True)
    c1 = make_citation("c1", "d1", "f1.txt", "Some text", 0.9)
    with pytest.raises(RuntimeError, match="Simulated Reranker failure"):
        await reranker.rerank(query="test", citations=[c1], top_k=1)


def test_create_reranker_factory() -> None:
    """Test create_reranker factory and validation errors."""
    reranker = create_reranker(provider="mock")
    assert isinstance(reranker, BaseReranker)
    assert isinstance(reranker, MockReranker)

    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        create_reranker(provider="unsupported_provider_xyz")


@pytest.mark.asyncio
async def test_rag_service_with_reranking_enabled_vs_disabled(
    settings: Settings,
) -> None:
    """Verify RAGService behavior with rerank=True vs rerank=False."""
    repo = InMemoryVectorRepository()

    # Populate two chunks
    meta1 = ChunkMetadata(
        chunk_id="chunk_1",
        doc_id="doc_1",
        index=0,
        start_char=0,
        end_char=50,
        character_count=50,
        word_count=8,
        source="fuel_system.txt",
        file_type="txt",
        checksum="chk1",
    )
    meta2 = ChunkMetadata(
        chunk_id="chunk_2",
        doc_id="doc_2",
        index=0,
        start_char=0,
        end_char=50,
        character_count=50,
        word_count=8,
        source="hydraulic_spec.txt",
        file_type="txt",
        checksum="chk2",
    )

    chunk1 = EmbeddedChunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        content="Aircraft auxiliary power unit fuel feed shutoff valve.",
        embedding=[1.0] + [0.0] * 1535,
        metadata=meta1,
    )
    chunk2 = EmbeddedChunk(
        chunk_id="chunk_2",
        doc_id="doc_2",
        content="Hydraulic landing gear extension actuator pressure thresholds.",
        embedding=[0.0, 1.0] + [0.0] * 1534,
        metadata=meta2,
    )

    repo._chunks["chunk_1"] = chunk1
    repo._chunks["chunk_2"] = chunk2

    retriever = VectorRetriever(
        embedding_service=EmbeddingService(settings=settings),
        vector_repository=repo,
    )
    rag_service = RAGService(
        retriever=retriever,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )

    # 1. RAG query with rerank=False
    resp_no_rerank = await rag_service.answer(
        question="Hydraulic landing gear pressure",
        top_k=2,
        min_similarity=-1.0,
        rerank=False,
    )
    assert len(resp_no_rerank.citations) > 0

    # 2. RAG query with rerank=True
    resp_rerank = await rag_service.answer(
        question="Hydraulic landing gear pressure",
        top_k=2,
        min_similarity=-1.0,
        rerank=True,
        top_n=5,
    )
    assert len(resp_rerank.citations) > 0
    assert resp_rerank.citations[0].rerank_score is not None
    assert resp_rerank.citations[0].initial_rank is not None


@pytest.mark.asyncio
async def test_rag_query_endpoint_with_reranking_telemetry(
    settings: Settings,
) -> None:
    """Verify POST /api/v1/rag/query executes with rerank=True and returns ranking metrics."""
    app = create_app(settings=settings)

    # Ingest document
    doc_service = app.state.document_service
    res = doc_service.ingest_bytes(
        content_bytes=b"Boeing 777 primary flight computer quadruple redundant fiber channel bus.",
        source_name="flight_computer_arch.txt",
        custom_metadata={"document_type": "avionics_spec"},
    )
    await doc_service.embed_and_index_document(res.doc_id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "question": "What is the primary flight computer bus architecture?",
            "top_k": 2,
            "min_similarity": -1.0,
            "rerank": True,
            "top_n": 5,
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["reranked"] is True
        assert data["rerank_metrics"] is not None
        assert len(data["rerank_metrics"]) > 0

        metric = data["rerank_metrics"][0]
        assert "chunk_id" in metric
        assert "initial_rank" in metric
        assert "reranked_rank" in metric
        assert "rank_delta" in metric
        assert "rerank_score" in metric

        assert len(data["citations"]) > 0
        assert data["citations"][0]["rerank_score"] is not None
