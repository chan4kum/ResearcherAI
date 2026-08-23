import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import create_app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.evaluator import EvaluationReason, RetrievalEvaluator
from app.services.rag.retriever import VectorRetriever
from app.services.rag.service import RAGService
from httpx import ASGITransport, AsyncClient


def populate_repository_with_test_chunks(repo: InMemoryVectorRepository) -> None:
    """Helper to populate mock vector repository with test chunks."""
    meta1 = ChunkMetadata(
        chunk_id="chunk_boeing_1",
        doc_id="doc_boeing",
        index=0,
        start_char=0,
        end_char=100,
        character_count=100,
        word_count=15,
        source="boeing_production_report.txt",
        file_type="txt",
        checksum="chk_b",
    )
    meta2 = ChunkMetadata(
        chunk_id="chunk_airbus_1",
        doc_id="doc_airbus",
        index=0,
        start_char=0,
        end_char=100,
        character_count=100,
        word_count=15,
        source="airbus_production_report.txt",
        file_type="txt",
        checksum="chk_a",
    )

    chunk1 = EmbeddedChunk(
        chunk_id="chunk_boeing_1",
        doc_id="doc_boeing",
        content=(
            "Boeing 777X and 737 MAX production delays were caused by "
            "fuselage structural supplier quality audits and regulatory recertification."
        ),
        embedding=[1.0] + [0.0] * 1535,
        metadata=meta1,
    )
    chunk2 = EmbeddedChunk(
        chunk_id="chunk_airbus_1",
        doc_id="doc_airbus",
        content=(
            "Airbus A321neo deliveries experienced bottlenecks due to "
            "Pratt & Whitney turbofan engine delivery delays and cabin interior customizations."
        ),
        embedding=[0.0, 1.0] + [0.0] * 1534,
        metadata=meta2,
    )

    repo._chunks["chunk_boeing_1"] = chunk1
    repo._chunks["chunk_airbus_1"] = chunk2


@pytest.mark.asyncio
async def test_successful_first_retrieval(settings: Settings) -> None:
    """Verify that a highly-relevant query is evaluated as sufficient on attempt 1."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)

    retriever = VectorRetriever(
        embedding_service=EmbeddingService(settings=settings),
        vector_repository=repo,
    )
    rag_service = RAGService(
        retriever=retriever,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )

    query = (
        "Boeing 777X and 737 MAX production delays were caused by "
        "fuselage structural supplier quality audits and regulatory recertification."
    )
    response = await rag_service.answer(
        question=query,
        min_similarity=-1.0,
        enable_rewriting=True,
        max_attempts=3,
    )

    assert response.retrieved_chunks_count > 0
    raw_rewrites = response.metadata.get("query_rewriting", [])
    assert len(raw_rewrites) == 1
    assert raw_rewrites[0]["is_sufficient"] is True
    assert raw_rewrites[0]["attempt"] == 1


@pytest.mark.asyncio
async def test_successful_second_retrieval_after_rewrite(settings: Settings) -> None:
    """Verify that an initially deficient retrieval triggers rewrite and succeeds on attempt 2."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)

    retriever = VectorRetriever(
        embedding_service=EmbeddingService(settings=settings),
        vector_repository=repo,
    )
    rag_service = RAGService(
        retriever=retriever,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )

    # Initial query missing 'Boeing' explicitly, but asking about delays
    query = (
        "What were the main reasons for production delays and how did those compare with Airbus?"
    )
    response = await rag_service.answer(
        question=query,
        min_similarity=-1.0,
        enable_rewriting=True,
        max_attempts=3,
    )

    raw_rewrites = response.metadata.get("query_rewriting", [])
    assert len(raw_rewrites) >= 1
    assert "query_rewriting" in response.metadata
    assert len(response.citations) > 0


@pytest.mark.asyncio
async def test_maximum_retries_prevent_infinite_loops(settings: Settings) -> None:
    """Verify loop strictly halts at max_attempts without cycling indefinitely."""
    repo = InMemoryVectorRepository()
    # Empty repository so it will never be sufficient
    retriever = VectorRetriever(
        embedding_service=EmbeddingService(settings=settings),
        vector_repository=repo,
    )
    rag_service = RAGService(
        retriever=retriever,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )

    max_tries = 3
    response = await rag_service.answer(
        question="Unobtainium hyperdrive warp core quantum specifications",
        min_similarity=-1.0,
        enable_rewriting=True,
        max_attempts=max_tries,
    )

    raw_rewrites = response.metadata.get("query_rewriting", [])
    assert len(raw_rewrites) <= max_tries
    # All attempts must be recorded
    for attempt in raw_rewrites:
        assert attempt["is_sufficient"] is False


@pytest.mark.asyncio
async def test_no_result_case_graceful_handling(settings: Settings) -> None:
    """Verify that an empty store evaluates as insufficient_evidence and completes cleanly."""
    evaluator = RetrievalEvaluator()
    analyzer = QueryAnalyzer(settings=settings)

    query = "Nonexistent technology telemetry"
    analysis = await analyzer.analyze(query)
    eval_res = evaluator.evaluate(query=query, analysis=analysis, citations=[])

    assert eval_res.is_sufficient is False
    assert EvaluationReason.INSUFFICIENT_EVIDENCE in eval_res.reasons
    assert eval_res.feedback_prompt is not None


@pytest.mark.asyncio
async def test_rag_query_endpoint_with_rewrite_telemetry(settings: Settings) -> None:
    """Verify POST /api/v1/rag/query with enable_rewriting=True returns rewrite_history."""
    app = create_app(settings=settings)

    # Ingest document
    doc_service = app.state.document_service
    res = doc_service.ingest_bytes(
        content_bytes=(
            b"Boeing 777X wing flutter damper maintenance procedure bulletin SB-2026-X99. "
            b"All titanium fittings must undergo ultrasonic inspection."
        ),
        source_name="boeing_flutter_damper.txt",
        custom_metadata={"document_type": "maintenance_bulletin"},
    )
    await doc_service.embed_and_index_document(res.doc_id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "question": "What is the maintenance bulletin for Boeing 777X flutter damper?",
            "enable_rewriting": True,
            "max_attempts": 3,
            "min_similarity": -1.0,
            "top_k": 2,
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["rewrite_history"] is not None
        assert len(data["rewrite_history"]) >= 1
        first_attempt = data["rewrite_history"][0]
        assert "attempt" in first_attempt
        assert "query" in first_attempt
        assert "top_score" in first_attempt
        assert "is_sufficient" in first_attempt
        assert len(data["citations"]) > 0
