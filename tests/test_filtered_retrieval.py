from typing import Any

import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.services.document.models import (
    ChunkMetadata,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
    MetadataFilter,
    normalize_metadata_filter,
)
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.retriever import VectorRetriever
from app.services.rag.service import RAGService


def create_test_chunk(
    doc_id: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
    source: str,
    document_type: str | None = None,
    department: str | None = None,
    date: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk with rich domain metadata."""
    meta = ChunkMetadata(
        chunk_id=f"{doc_id}_chunk_{chunk_index}",
        doc_id=doc_id,
        index=chunk_index,
        start_char=0,
        end_char=len(content),
        character_count=len(content),
        word_count=len(content.split()),
        source=source,
        file_type="txt",
        checksum=f"chk_{doc_id}_{chunk_index}",
        document_type=document_type,
        department=department,
        date=date,
        author=author,
        tags=tags or [],
        custom_metadata=custom_metadata or {},
    )
    return EmbeddedChunk(
        chunk_id=meta.chunk_id,
        doc_id=doc_id,
        content=content,
        embedding=embedding,
        metadata=meta,
    )


@pytest.fixture
def mock_embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(settings=settings)


@pytest.fixture
def populated_vector_repo(settings: Settings) -> InMemoryVectorRepository:
    """Populate an in-memory repository with diverse domain documents for filtered testing."""
    repo = InMemoryVectorRepository()

    # Doc 1: Boeing Quality Report
    doc1 = Document(
        doc_id="doc_boeing_qa",
        content=(
            "Boeing 737 fuselage inspection detected minor rivet "
            "alignment variance in section 41."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_boeing_qa",
            source="boeing_qa_audit_2026.txt",
            file_type="txt",
            checksum="chk_boeing_qa",
            character_count=85,
            word_count=12,
            document_type="quality_report",
            department="QA",
            date="2026-08-01",
            author="John Inspector",
            tags=["boeing", "aviation", "quality", "audit"],
            custom_metadata={"inspection_facility": "Renton"},
        ),
    )

    # Doc 2: Boeing Financial Report
    doc2 = Document(
        doc_id="doc_boeing_fin",
        content=(
            "Boeing Q2 fiscal outlook projects commercial aircraft delivery "
            "ramp-up and cash flow rebound."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_boeing_fin",
            source="boeing_financial_q2.txt",
            file_type="txt",
            checksum="chk_boeing_fin",
            character_count=93,
            word_count=13,
            document_type="financial_report",
            department="Finance",
            date="2026-08-15",
            author="Sarah CFO",
            tags=["boeing", "financial", "earnings"],
            custom_metadata={"market": "NYSE"},
        ),
    )

    # Doc 3: Airbus Technical Spec
    doc3 = Document(
        doc_id="doc_airbus_spec",
        content=(
            "Airbus A350 composite wing spar aerodynamic stress tolerances "
            "and carbon fiber curing specs."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_airbus_spec",
            source="airbus_a350_spec.txt",
            file_type="txt",
            checksum="chk_airbus_spec",
            character_count=92,
            word_count=13,
            document_type="engineering_spec",
            department="Engineering",
            date="2026-07-20",
            author="Pierre Engineer",
            tags=["airbus", "composite", "aerospace"],
            custom_metadata={"certification": "EASA"},
        ),
    )

    # Generate synthetic 1536-dim orthogonal embeddings for predictable testing
    vec_qa = [1.0] + [0.0] * 1535
    vec_fin = [0.0, 1.0] + [0.0] * 1534
    vec_spec = [0.0, 0.0, 1.0] + [0.0] * 1533

    chunk1 = create_test_chunk(
        doc_id=doc1.doc_id,
        chunk_index=0,
        content=doc1.content,
        embedding=vec_qa,
        source=doc1.metadata.source,
        document_type=doc1.metadata.document_type,
        department=doc1.metadata.department,
        date=doc1.metadata.date,
        author=doc1.metadata.author,
        tags=doc1.metadata.tags,
        custom_metadata=doc1.metadata.custom_metadata,
    )

    chunk2 = create_test_chunk(
        doc_id=doc2.doc_id,
        chunk_index=0,
        content=doc2.content,
        embedding=vec_fin,
        source=doc2.metadata.source,
        document_type=doc2.metadata.document_type,
        department=doc2.metadata.department,
        date=doc2.metadata.date,
        author=doc2.metadata.author,
        tags=doc2.metadata.tags,
        custom_metadata=doc2.metadata.custom_metadata,
    )

    chunk3 = create_test_chunk(
        doc_id=doc3.doc_id,
        chunk_index=0,
        content=doc3.content,
        embedding=vec_spec,
        source=doc3.metadata.source,
        document_type=doc3.metadata.document_type,
        department=doc3.metadata.department,
        date=doc3.metadata.date,
        author=doc3.metadata.author,
        tags=doc3.metadata.tags,
        custom_metadata=doc3.metadata.custom_metadata,
    )

    repo._documents[doc1.doc_id] = doc1
    repo._documents[doc2.doc_id] = doc2
    repo._documents[doc3.doc_id] = doc3

    repo._chunks[chunk1.chunk_id] = chunk1
    repo._chunks[chunk2.chunk_id] = chunk2
    repo._chunks[chunk3.chunk_id] = chunk3

    return repo


@pytest.mark.asyncio
async def test_unrestricted_search(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
) -> None:
    """Unrestricted search (no filters) retrieves across all documents regardless of metadata."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    citations = await retriever.retrieve(
        query="Boeing inspection and finances",
        top_k=5,
        min_similarity=-1.0,
        filters=None,
    )
    assert len(citations) == 3
    doc_types = {c.document_type for c in citations}
    assert "quality_report" in doc_types
    assert "financial_report" in doc_types
    assert "engineering_spec" in doc_types


@pytest.mark.asyncio
async def test_filtered_search_by_document_type(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
) -> None:
    """Filtering by document_type restricts results strictly to matching quality reports."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    citations = await retriever.retrieve(
        query="Boeing reports",
        top_k=5,
        min_similarity=-1.0,
        filters={"document_type": "quality_report"},
    )
    assert len(citations) == 1
    assert citations[0].doc_id == "doc_boeing_qa"
    assert citations[0].document_type == "quality_report"
    assert citations[0].department == "QA"
    assert "fuselage inspection" in citations[0].content


@pytest.mark.asyncio
async def test_filtered_search_by_department(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
) -> None:
    """Filtering by department = 'Finance' returns only financial documents."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    citations = await retriever.retrieve(
        query="Boeing outlook",
        top_k=5,
        min_similarity=-1.0,
        filters={"department": "Finance"},
    )
    assert len(citations) == 1
    assert citations[0].doc_id == "doc_boeing_fin"
    assert citations[0].department == "Finance"
    assert "fiscal outlook" in citations[0].content


@pytest.mark.asyncio
async def test_filtered_search_by_tags(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
) -> None:
    """Filtering by tags matches chunks containing any of the requested tags."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    citations = await retriever.retrieve(
        query="aerospace composites",
        top_k=5,
        min_similarity=-1.0,
        filters={"tags": ["aerospace"]},
    )
    assert len(citations) == 1
    assert citations[0].doc_id == "doc_airbus_spec"
    assert "aerospace" in citations[0].tags


@pytest.mark.asyncio
async def test_filtered_search_no_matching_documents(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
    settings: Settings,
) -> None:
    """When a filter matches zero documents, RAG returns 0 chunks and a clean grounded response."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    rag_service = RAGService(
        retriever=retriever,
        llm_service=LLMService(settings=settings),
        settings=settings,
    )

    response = await rag_service.answer(
        question="What does the Boeing quality report say?",
        top_k=5,
        min_similarity=-1.0,
        filters={"document_type": "non_existent_category"},
    )
    assert response.retrieved_chunks_count == 0
    assert len(response.citations) == 0
    assert "do not contain information" in response.answer


@pytest.mark.asyncio
async def test_invalid_filter_types(
    mock_embedding_service: EmbeddingService,
    populated_vector_repo: InMemoryVectorRepository,
) -> None:
    """Passing invalid filter types raises a descriptive ValueError."""
    retriever = VectorRetriever(
        embedding_service=mock_embedding_service,
        vector_repository=populated_vector_repo,
    )
    with pytest.raises(ValueError, match="Invalid filter format"):
        await retriever.retrieve(
            query="test",
            filters="not_a_dict_or_filter_object",  # type: ignore[arg-type]
        )


def test_metadata_filter_matching_logic() -> None:
    """Direct unit testing of MetadataFilter matching rules."""
    meta = ChunkMetadata(
        chunk_id="chunk_test_1",
        doc_id="doc_1",
        index=0,
        start_char=0,
        end_char=50,
        character_count=50,
        word_count=8,
        source="test_report.pdf",
        file_type="pdf",
        checksum="chk_1",
        document_type="quality_report",
        department="QA",
        date="2026-08-20",
        author="Alice QA",
        tags=["audit", "iso9001"],
        custom_metadata={"facility": "Plant_A"},
    )

    # 1. Matching single field
    f1 = MetadataFilter(document_type="quality_report")
    assert f1.matches(meta) is True

    # 2. Non-matching single field
    f2 = MetadataFilter(document_type="memo")
    assert f2.matches(meta) is False

    # 3. Matching multi-field
    f3 = MetadataFilter(document_type="quality_report", department="qa", author="alice qa")
    assert f3.matches(meta) is True

    # 4. Matching tag
    f4 = MetadataFilter(tags=["audit"])
    assert f4.matches(meta) is True

    # 5. Non-matching tag
    f5 = MetadataFilter(tags=["finance", "tax"])
    assert f5.matches(meta) is False

    # 6. Matching custom metadata
    f6 = MetadataFilter(custom_metadata={"facility": "Plant_A"})
    assert f6.matches(meta) is True

    # 7. Non-matching custom metadata
    f7 = MetadataFilter(custom_metadata={"facility": "Plant_B"})
    assert f7.matches(meta) is False


def test_normalize_metadata_filter() -> None:
    """Test converting raw dicts and objects to MetadataFilter instances."""
    assert normalize_metadata_filter(None) is None

    filter_dict = {
        "document_type": "quality_report",
        "department": "Engineering",
        "custom_key": "custom_val",
    }
    normalized = normalize_metadata_filter(filter_dict)
    assert normalized is not None
    assert normalized.document_type == "quality_report"
    assert normalized.department == "Engineering"
    assert normalized.custom_metadata == {"custom_key": "custom_val"}
