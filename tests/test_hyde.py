import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import create_app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.hyde import HyDEGenerator
from app.services.rag.retriever import HyDERetriever, VectorRetriever, create_retriever
from app.services.rag.service import RAGService
from httpx import ASGITransport, AsyncClient


def populate_repository_with_test_chunks(repo: InMemoryVectorRepository) -> None:
    """Populate in-memory repository with domain-specific engineering documentation."""
    meta1 = ChunkMetadata(
        chunk_id="chunk_boeing_flutter",
        doc_id="doc_boeing_flutter",
        index=0,
        start_char=0,
        end_char=120,
        character_count=120,
        word_count=18,
        source="boeing_flutter_bulletin.txt",
        file_type="txt",
        checksum="chk_bf",
    )
    chunk1 = EmbeddedChunk(
        chunk_id="chunk_boeing_flutter",
        doc_id="doc_boeing_flutter",
        content=(
            "Boeing 777X wing flutter damper maintenance procedure bulletin SB-2026-X99. "
            "All titanium hydraulic fittings must undergo periodic ultrasonic inspection "
            "and torsional rigidity compliance testing."
        ),
        embedding=[1.0] + [0.0] * 1535,
        metadata=meta1,
    )
    repo._chunks["chunk_boeing_flutter"] = chunk1


@pytest.mark.asyncio
async def test_hyde_generator_produces_hypothetical_document(settings: Settings) -> None:
    """Verify HyDEGenerator synthesizes a detailed technical document passage."""
    llm = LLMService(settings=settings)
    generator = HyDEGenerator(llm_service=llm, settings=settings)

    query = "How to inspect Boeing wing flutter dampers?"
    hypo_doc = await generator.generate(query)

    assert hypo_doc is not None
    assert len(hypo_doc) > 30
    assert "Boeing" in hypo_doc or "Technical" in hypo_doc or "flutter" in hypo_doc


@pytest.mark.asyncio
async def test_hyde_generator_empty_query_raises_error(settings: Settings) -> None:
    """Verify empty query raises ValueError."""
    generator = HyDEGenerator(settings=settings)
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await generator.generate("   ")


@pytest.mark.asyncio
async def test_hyde_retriever_generates_and_retrieves(settings: Settings) -> None:
    """Verify HyDERetriever embeds hypothetical passage and retrieves matching real chunks."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)

    embedding_service = EmbeddingService(settings=settings)
    hyde_retriever = HyDERetriever(
        embedding_service=embedding_service,
        vector_repository=repo,
        settings=settings,
    )

    query = "wing flutter damper inspection procedure"
    citations = await hyde_retriever.retrieve(query=query, top_k=2, min_similarity=-1.0)

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_boeing_flutter"
    assert hyde_retriever.last_result is not None
    assert hyde_retriever.last_result.original_query == query
    assert hyde_retriever.last_result.strategy == "hyde"
    assert len(hyde_retriever.last_result.hypothetical_document) > 0


@pytest.mark.asyncio
async def test_rag_service_strategy_switching(settings: Settings) -> None:
    """Verify strategy='normal' vs strategy='hyde' selection in RAGService."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)

    embedding_service = EmbeddingService(settings=settings)
    vector_retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_repository=repo,
    )
    rag_service = RAGService(
        retriever=vector_retriever,
        settings=settings,
    )

    # 1. Normal strategy
    resp_normal = await rag_service.answer(
        question="flutter damper",
        strategy="normal",
        min_similarity=-1.0,
    )
    assert resp_normal.strategy == "normal"
    assert resp_normal.hypothetical_document is None

    # 2. HyDE strategy
    resp_hyde = await rag_service.answer(
        question="flutter damper",
        strategy="hyde",
        min_similarity=-1.0,
    )
    assert resp_hyde.strategy == "hyde"
    assert resp_hyde.hypothetical_document is not None
    assert "hyde" in resp_hyde.metadata


@pytest.mark.asyncio
async def test_create_retriever_factory_hyde_support(settings: Settings) -> None:
    """Verify create_retriever factory supports strategy='hyde' and mode='hyde'."""
    repo = InMemoryVectorRepository()
    emb = EmbeddingService(settings=settings)

    retriever_by_strat = create_retriever(
        embedding_service=emb,
        vector_repository=repo,
        strategy="hyde",
        settings=settings,
    )
    assert isinstance(retriever_by_strat, HyDERetriever)

    retriever_by_mode = create_retriever(
        embedding_service=emb,
        vector_repository=repo,
        mode="hyde",
        settings=settings,
    )
    assert isinstance(retriever_by_mode, HyDERetriever)


@pytest.mark.asyncio
async def test_rag_query_endpoint_with_hyde_strategy(settings: Settings) -> None:
    """Verify POST /api/v1/rag/query with strategy='hyde' returns hypothetical_document."""
    app = create_app(settings=settings)

    # Ingest test document
    doc_service = app.state.document_service
    res = doc_service.ingest_bytes(
        content_bytes=(
            b"Titanium structural flutter damper bulletin SB-2026-X99 specifies "
            b"strict ultrasonic testing schedules for commercial airframes."
        ),
        source_name="titanium_flutter.txt",
        custom_metadata={"document_type": "maintenance_bulletin"},
    )
    await doc_service.embed_and_index_document(res.doc_id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "question": "flutter damper bulletin",
            "strategy": "hyde",
            "min_similarity": -1.0,
            "top_k": 2,
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["strategy"] == "hyde"
        assert data["hypothetical_document"] is not None
        assert len(data["hypothetical_document"]) > 0
        assert len(data["citations"]) > 0
