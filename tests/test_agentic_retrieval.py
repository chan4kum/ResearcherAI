import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import app
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.rag.agentic_retrieval import (
    AgenticRetrievalEngine,
    AgenticRetrievalResult,
    RetrievalTraceStore,
)
from app.services.rag.router import RetrievalRouter
from app.services.rag.sources import (
    KeywordSearchSource,
    RetrievalSourceRegistry,
    StructuredDatabasePlaceholderSource,
    VectorDatabaseSource,
    WebSearchPlaceholderSource,
)
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def test_trace_store(tmp_path: pytest.TempPathFactory) -> RetrievalTraceStore:
    """Fixture providing isolated temporary trace store."""
    return RetrievalTraceStore(storage_dir=tmp_path)


@pytest.mark.asyncio
async def test_no_retrieval_needed_for_greeting(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify conversational queries bypass retrieval and terminate with NO_RETRIEVAL_NEEDED."""
    engine = AgenticRetrievalEngine(trace_store=test_trace_store, settings=settings)
    result = await engine.execute(query="Hello there, how are you?")

    assert isinstance(result, AgenticRetrievalResult)
    assert result.is_sufficient is True
    assert result.trace.total_iterations == 0
    assert result.trace.total_tool_calls == 0
    assert result.trace.termination_reason == "NO_RETRIEVAL_NEEDED"
    assert len(result.citations) == 0
    assert len(result.answer) > 0


@pytest.mark.asyncio
async def test_single_pass_sufficient_retrieval(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify single-pass retrieval satisfies sufficiency and generates citations."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()
    emb_service = EmbeddingService(settings=settings)

    content = (
        "What is the Boeing 777X flutter damper inspection protocol? "
        "Boeing 777X wing flutter damper protocol."
    )
    emb = await emb_service.embed_text(content)
    meta = ChunkMetadata(
        chunk_id="chk_agentic_1",
        doc_id="doc_agentic_1",
        index=0,
        start_char=0,
        end_char=80,
        character_count=80,
        word_count=10,
        source="boeing_sb.txt",
        file_type="txt",
        checksum="c1",
        document_type="bulletin",
    )
    repo._chunks["chk_agentic_1"] = EmbeddedChunk(
        chunk_id="chk_agentic_1",
        doc_id="doc_agentic_1",
        content=content,
        embedding=emb,
        metadata=meta,
    )

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    src_keyword = KeywordSearchSource(repo, source_name="keyword_db")
    registry.register(src_vector)
    registry.register(src_keyword)

    router = RetrievalRouter(registry=registry, settings=settings)
    engine = AgenticRetrievalEngine(router=router, trace_store=test_trace_store, settings=settings)

    result = await engine.execute(
        query="What is the Boeing 777X flutter damper inspection protocol?",
        max_iterations=3,
    )

    assert result.is_sufficient is True
    assert result.trace.total_iterations == 1
    assert result.trace.termination_reason == "EVIDENCE_SUFFICIENT"
    assert len(result.citations) >= 1
    assert any("Boeing 777X" in c.content for c in result.citations)


@pytest.mark.asyncio
async def test_hard_limit_max_iterations(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify loop halts at max_iterations when no evidence is found."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()  # Empty repo
    emb_service = EmbeddingService(settings=settings)

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    registry.register(src_vector)

    router = RetrievalRouter(registry=registry, settings=settings)
    engine = AgenticRetrievalEngine(router=router, trace_store=test_trace_store, settings=settings)

    result = await engine.execute(
        query="What is the top secret formula for Project WarpDrive 999?",
        max_iterations=2,
    )

    assert result.is_sufficient is False
    assert result.trace.total_iterations == 2
    assert result.trace.termination_reason == "MAX_ITERATIONS_REACHED"
    assert len(result.citations) == 0


@pytest.mark.asyncio
async def test_hard_limit_max_tool_calls(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify loop respects max_tool_calls limit."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()
    emb_service = EmbeddingService(settings=settings)

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    src_web = WebSearchPlaceholderSource(source_name="web_engine")
    src_db = StructuredDatabasePlaceholderSource(source_name="sql_db")
    registry.register(src_vector)
    registry.register(src_web)
    registry.register(src_db)

    router = RetrievalRouter(registry=registry, settings=settings)
    engine = AgenticRetrievalEngine(router=router, trace_store=test_trace_store, settings=settings)

    result = await engine.execute(
        query="Compare our internal numbers with public market information.",
        max_iterations=5,
        max_tool_calls=1,
    )

    assert result.trace.termination_reason in ("MAX_TOOL_CALLS_EXCEEDED", "EVIDENCE_SUFFICIENT")
    assert result.trace.total_tool_calls >= 1


@pytest.mark.asyncio
async def test_hard_limit_max_retrieved_documents(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify loop caps accumulated documents at max_retrieved_documents."""
    registry = RetrievalSourceRegistry()
    src_web = WebSearchPlaceholderSource(source_name="web_engine")
    registry.register(src_web)

    router = RetrievalRouter(registry=registry, settings=settings)
    engine = AgenticRetrievalEngine(router=router, trace_store=test_trace_store, settings=settings)

    result = await engine.execute(
        query="What happened in latest aerospace industry news?",
        max_retrieved_documents=1,
    )

    assert len(result.citations) <= 1
    assert result.trace.total_documents_retrieved <= 1


@pytest.mark.asyncio
async def test_trace_persistence_and_lookup(
    settings: Settings,
    test_trace_store: RetrievalTraceStore,
) -> None:
    """Verify trace is saved and can be retrieved by session_id."""
    engine = AgenticRetrievalEngine(trace_store=test_trace_store, settings=settings)
    result = await engine.execute(query="Hello there!")

    session_id = result.trace.session_id
    stored_trace = test_trace_store.get_trace(session_id)
    assert stored_trace is not None
    assert stored_trace.session_id == session_id
    assert stored_trace.termination_reason == "NO_RETRIEVAL_NEEDED"
    assert len(stored_trace.steps) >= 1


@pytest.mark.asyncio
async def test_agentic_retrieval_endpoints() -> None:
    """Verify POST /api/v1/rag/agentic-retrieve and GET /api/v1/rag/traces/{id}."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. Execute agentic retrieval
        payload = {
            "query": "What happened in NVIDIA's latest earnings report?",
            "max_iterations": 2,
            "max_tool_calls": 4,
            "max_retrieved_documents": 10,
        }
        res = await client.post("/api/v1/rag/agentic-retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert "answer" in data
        assert "steps" in data
        assert len(data["steps"]) >= 1

        session_id = data["session_id"]

        # 2. Fetch trace by session_id
        trace_res = await client.get(f"/api/v1/rag/traces/{session_id}")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()
        assert trace_data["session_id"] == session_id
        assert trace_data["original_query"] == payload["query"]

        # 3. Non-existent trace returns 404
        not_found_res = await client.get("/api/v1/rag/traces/non_existent_session_999")
        assert not_found_res.status_code == 404
