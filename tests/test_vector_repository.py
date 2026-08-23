from unittest.mock import AsyncMock, MagicMock

import pytest
from app.config import Settings
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.db.repository import (
    InMemoryVectorRepository,
    PgVectorRepository,
    create_vector_repository,
)
from app.db.session import DatabaseManager
from app.services.document.models import (
    ChunkMetadata,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
)


@pytest.mark.asyncio
async def test_in_memory_vector_repository_crud_and_search() -> None:
    """Verify InMemoryVectorRepository stores documents, chunks, and performs similarity search."""
    repo = InMemoryVectorRepository()

    # 1. Store document
    doc_meta = DocumentMetadata(
        doc_id="doc_100",
        source="semiconductor.txt",
        file_type="txt",
        checksum="chk_100",
        character_count=200,
        word_count=30,
    )
    doc = Document(doc_id="doc_100", content="Semiconductor lithography.", metadata=doc_meta)
    await repo.store_document(doc)

    retrieved = await repo.get_document("doc_100")
    assert retrieved is not None
    assert retrieved.doc_id == "doc_100"

    docs = await repo.list_documents()
    assert len(docs) == 1

    # 2. Store embedded chunks
    chunk_meta1 = ChunkMetadata(
        chunk_id="chunk_1",
        doc_id="doc_100",
        index=0,
        start_char=0,
        end_char=20,
        character_count=20,
        word_count=3,
        source="semiconductor.txt",
        file_type="txt",
        checksum="chk_100",
    )
    chunk1 = EmbeddedChunk(
        chunk_id="chunk_1",
        doc_id="doc_100",
        content="Semiconductor lithography.",
        embedding=[1.0, 0.0, 0.0],
        metadata=chunk_meta1,
    )

    chunk_meta2 = ChunkMetadata(
        chunk_id="chunk_2",
        doc_id="doc_100",
        index=1,
        start_char=20,
        end_char=40,
        character_count=20,
        word_count=3,
        source="semiconductor.txt",
        file_type="txt",
        checksum="chk_100",
    )
    chunk2 = EmbeddedChunk(
        chunk_id="chunk_2",
        doc_id="doc_100",
        content="Chemical vapor deposition.",
        embedding=[0.0, 1.0, 0.0],
        metadata=chunk_meta2,
    )

    await repo.store_chunks([chunk1, chunk2])

    # 3. Search similar to [1.0, 0.0, 0.0]
    query_vec = [1.0, 0.0, 0.0]
    results = await repo.search_similar_chunks(query_vec, top_k=2)

    assert len(results) == 2
    # First result should be chunk_1 with similarity 1.0
    assert results[0][0].chunk_id == "chunk_1"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0
    # Second result should be chunk_2 with similarity 0.0
    assert results[1][0].chunk_id == "chunk_2"
    assert pytest.approx(results[1][1], abs=1e-4) == 0.0

    # 4. Search with min_similarity threshold
    filtered_results = await repo.search_similar_chunks(query_vec, top_k=2, min_similarity=0.5)
    assert len(filtered_results) == 1
    assert filtered_results[0][0].chunk_id == "chunk_1"

    # 5. Delete document
    deleted = await repo.delete_document("doc_100")
    assert deleted is True
    assert await repo.get_document("doc_100") is None
    # Associated chunks should also be deleted
    assert await repo.search_similar_chunks(query_vec) == []


def test_create_vector_repository_factory() -> None:
    """Verify create_vector_repository returns correct implementation based on settings."""
    settings_mem = Settings(vector_repository_type="in_memory")
    repo_mem = create_vector_repository(settings_mem)
    assert isinstance(repo_mem, InMemoryVectorRepository)

    settings_pg = Settings(
        vector_repository_type="postgres",
        database_url="postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_db",
    )
    repo_pg = create_vector_repository(settings_pg)
    assert isinstance(repo_pg, PgVectorRepository)


@pytest.mark.asyncio
async def test_pgvector_repository_mocked_session() -> None:
    """Verify PgVectorRepository executes expected session queries."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_db_manager = MagicMock(spec=DatabaseManager)
    mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

    repo = PgVectorRepository(mock_db_manager)

    # 1. Test store_document
    doc_meta = DocumentMetadata(
        doc_id="pg_doc_1",
        source="pg_test.txt",
        file_type="txt",
        checksum="pg_chk",
        character_count=50,
        word_count=8,
    )
    doc = Document(doc_id="pg_doc_1", content="Postgres vector test.", metadata=doc_meta)

    # Mock execute result for existing record check
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    await repo.store_document(doc)
    mock_session.add.assert_called_once()

    # 2. Test search_similar_chunks
    chunk_rec = DocumentChunkRecord(
        id="pg_chunk_1",
        doc_id="pg_doc_1",
        chunk_index=0,
        content="Sample content",
        start_char=0,
        end_char=14,
        character_count=14,
        word_count=2,
        embedding=[0.5, 0.5],
        document=DocumentRecord(
            id="pg_doc_1",
            source="pg_test.txt",
            file_type="txt",
            checksum="pg_chk",
            content="Postgres vector test.",
            character_count=50,
            word_count=8,
        ),
    )
    # Cosine distance = 0.1 => similarity = 0.9
    mock_search_res = MagicMock()
    mock_search_res.all.return_value = [(chunk_rec, 0.1)]
    mock_session.execute.return_value = mock_search_res

    results = await repo.search_similar_chunks([0.5, 0.5], top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "pg_chunk_1"
    assert pytest.approx(results[0][1], abs=1e-4) == 0.9
