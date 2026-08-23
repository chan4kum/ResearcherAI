import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.services.document.models import ChunkMetadata, EmbeddedChunk
from app.services.embedding.service import EmbeddingService
from app.services.rag.sources import (
    KeywordSearchSource,
    RetrievalSourceRegistry,
    SourceResult,
    SourceType,
    StructuredDatabasePlaceholderSource,
    VectorDatabaseSource,
    WebSearchPlaceholderSource,
)


def populate_repository_with_test_chunks(repo: InMemoryVectorRepository) -> None:
    """Helper to populate in-memory repository with test engineering documents."""
    meta1 = ChunkMetadata(
        chunk_id="chunk_vector_1",
        doc_id="doc_vector_1",
        index=0,
        start_char=0,
        end_char=100,
        character_count=100,
        word_count=15,
        source="boeing_flutter_manual.txt",
        file_type="txt",
        checksum="chk_vec1",
        document_type="technical_manual",
        department="aerodynamics",
    )
    chunk1 = EmbeddedChunk(
        chunk_id="chunk_vector_1",
        doc_id="doc_vector_1",
        content=(
            "Boeing 777X wing flutter damper maintenance procedure bulletin SB-2026-X99. "
            "Ultrasonic inspection and torque verification are mandatory."
        ),
        embedding=[1.0] + [0.0] * 1535,
        metadata=meta1,
    )
    repo._chunks["chunk_vector_1"] = chunk1


@pytest.mark.asyncio
async def test_vector_database_source(settings: Settings) -> None:
    """Verify VectorDatabaseSource returns standardized SourceResult with citation."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)
    emb_service = EmbeddingService(settings=settings)

    source = VectorDatabaseSource(
        embedding_service=emb_service,
        vector_repository=repo,
        source_name="internal_vector_kb",
    )

    assert source.source_type == SourceType.INTERNAL_VECTOR
    assert source.source_name == "internal_vector_kb"

    results = await source.search(query="flutter damper maintenance", top_k=2, min_relevance=-1.0)
    assert len(results) == 1
    res = results[0]

    assert isinstance(res, SourceResult)
    assert res.source == "internal_vector_kb"
    assert res.source_type == SourceType.INTERNAL_VECTOR
    assert "Boeing 777X" in res.content
    assert res.relevance is not None
    assert res.metadata["document_type"] == "technical_manual"
    assert res.citation.chunk_id == "chunk_vector_1"
    assert res.citation.source == "boeing_flutter_manual.txt"


@pytest.mark.asyncio
async def test_keyword_search_source() -> None:
    """Verify KeywordSearchSource performs BM25 search and returns SourceResult."""
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)

    source = KeywordSearchSource(
        vector_repository=repo,
        source_name="bm25_search",
    )

    assert source.source_type == SourceType.KEYWORD
    assert source.source_name == "bm25_search"

    results = await source.search(query="ultrasonic inspection mandatory", top_k=2)
    assert len(results) == 1
    res = results[0]

    assert isinstance(res, SourceResult)
    assert res.source == "bm25_search"
    assert res.source_type == SourceType.KEYWORD
    assert "ultrasonic" in res.content.lower()
    assert res.relevance > 0.0
    assert res.citation.doc_id == "doc_vector_1"


@pytest.mark.asyncio
async def test_web_search_placeholder_source() -> None:
    """Verify WebSearchPlaceholderSource returns simulated web search results."""
    source = WebSearchPlaceholderSource(source_name="global_web_search")

    assert source.source_type == SourceType.WEB_SEARCH
    assert source.source_name == "global_web_search"

    results = await source.search(query="Aviation Safety Bureau Boeing 777X", top_k=2)
    assert len(results) >= 1
    res = results[0]

    assert isinstance(res, SourceResult)
    assert res.source == "global_web_search"
    assert res.source_type == SourceType.WEB_SEARCH
    assert "aviation-safety.org" in res.metadata["url"]
    assert res.citation.source.startswith("https://")
    assert res.citation.file_type == "html"
    assert len(res.content) > 20


@pytest.mark.asyncio
async def test_structured_database_placeholder_source() -> None:
    """Verify StructuredDatabasePlaceholderSource returns formatted tabular rows."""
    source = StructuredDatabasePlaceholderSource(source_name="relational_maintenance_db")

    assert source.source_type == SourceType.STRUCTURED_DB
    assert source.source_name == "relational_maintenance_db"

    results = await source.search(query="Boeing 777-9 flutter damper actuator", top_k=2)
    assert len(results) >= 1
    res = results[0]

    assert isinstance(res, SourceResult)
    assert res.source == "relational_maintenance_db"
    assert res.source_type == SourceType.STRUCTURED_DB
    assert "Table: aircraft_maintenance_log" in res.content
    assert res.metadata["table"] == "aircraft_maintenance_log"
    assert res.metadata["primary_key"] == "LOG-2026-777X-001"
    assert res.citation.source.startswith("sql://")


@pytest.mark.asyncio
async def test_retrieval_source_registry(settings: Settings) -> None:
    """Verify RetrievalSourceRegistry registers, lists, and queries heterogeneous sources."""
    registry = RetrievalSourceRegistry()
    repo = InMemoryVectorRepository()
    populate_repository_with_test_chunks(repo)
    emb_service = EmbeddingService(settings=settings)

    src_vector = VectorDatabaseSource(emb_service, repo, source_name="vector_db")
    src_keyword = KeywordSearchSource(repo, source_name="keyword_idx")
    src_web = WebSearchPlaceholderSource(source_name="web_engine")
    src_db = StructuredDatabasePlaceholderSource(source_name="sql_db")

    registry.register(src_vector)
    registry.register(src_keyword)
    registry.register(src_web)
    registry.register(src_db)

    assert len(registry.list_sources()) == 4
    assert registry.get_source("vector_db") is src_vector
    assert len(registry.get_sources_by_type(SourceType.WEB_SEARCH)) == 1

    # Search all sources
    all_results = await registry.search_all(query="flutter damper inspection", top_k_per_source=2)
    assert "vector_db" in all_results
    assert "keyword_idx" in all_results
    assert "web_engine" in all_results
    assert "sql_db" in all_results

    # Search selected sources
    selected_results = await registry.search_sources(
        query="flutter damper",
        source_names=["vector_db", "web_engine"],
        top_k=5,
    )
    assert len(selected_results) >= 2
    sources_found = {r.source for r in selected_results}
    assert "vector_db" in sources_found
