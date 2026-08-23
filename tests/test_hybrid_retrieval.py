from typing import Any

import pytest
from app.config import Settings
from app.db.repository import InMemoryVectorRepository
from app.main import create_app
from app.services.document.models import (
    ChunkMetadata,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
)
from app.services.embedding.service import EmbeddingService
from app.services.rag.bm25 import BM25Index, tokenize
from app.services.rag.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.services.rag.retriever import (
    BM25Retriever,
    HybridRetriever,
    VectorRetriever,
    create_retriever,
)
from httpx import ASGITransport, AsyncClient


def create_test_chunk(
    doc_id: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
    source: str,
    document_type: str | None = None,
    department: str | None = None,
    tags: list[str] | None = None,
    custom_metadata: dict[str, Any] | None = None,
) -> EmbeddedChunk:
    """Helper to construct an EmbeddedChunk with domain metadata."""
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
def hybrid_test_repo(settings: Settings) -> InMemoryVectorRepository:
    """Populate repository with documents tailored for semantic vs lexical comparison."""
    repo = InMemoryVectorRepository()

    # Doc A: Rich semantic context about aircraft safety without mentioning specific code
    doc_a = Document(
        doc_id="doc_semantic_safety",
        content=(
            "Modern aviation safety protocols require rigorous ultrasonic airframe "
            "scanning to identify micro-fractures in pressurized passenger cabins."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_semantic_safety",
            source="safety_guidelines.txt",
            file_type="txt",
            checksum="chk_a",
            character_count=145,
            word_count=18,
            document_type="safety_protocol",
            department="Aviation Safety",
            tags=["safety", "airframe", "ultrasonic"],
        ),
    )

    # Doc B: Technical bulletin with exact product serial number and error code
    doc_b = Document(
        doc_id="doc_lexical_code",
        content=(
            "Service Bulletin SB-2026-X99: Rectification procedure for hydraulic pump "
            "actuator fault code ERR-7701 on aircraft model B777X-PRO."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_lexical_code",
            source="service_bulletin_x99.txt",
            file_type="txt",
            checksum="chk_b",
            character_count=138,
            word_count=17,
            document_type="service_bulletin",
            department="Maintenance",
            tags=["bulletin", "hydraulic", "maintenance"],
        ),
    )

    # Doc C: Hybrid document combining conceptual safety discussion with specific code
    doc_c = Document(
        doc_id="doc_hybrid_composite",
        content=(
            "Composite wing spar aerodynamic structural integrity testing under FAA-CERT-4491 "
            "exceeded standard tensile resistance thresholds."
        ),
        metadata=DocumentMetadata(
            doc_id="doc_hybrid_composite",
            source="composite_testing.txt",
            file_type="txt",
            checksum="chk_c",
            character_count=146,
            word_count=17,
            document_type="engineering_report",
            department="Engineering",
            tags=["composite", "testing", "faa"],
        ),
    )

    # Synthetic 1536-dim orthogonal embeddings for predictable testing
    vec_a = [1.0] + [0.0] * 1535
    vec_b = [0.0, 1.0] + [0.0] * 1534
    vec_c = [0.5, 0.5] + [0.0] * 1534

    chunk_a = create_test_chunk(
        doc_id=doc_a.doc_id,
        chunk_index=0,
        content=doc_a.content,
        embedding=vec_a,
        source=doc_a.metadata.source,
        document_type=doc_a.metadata.document_type,
        department=doc_a.metadata.department,
        tags=doc_a.metadata.tags,
    )

    chunk_b = create_test_chunk(
        doc_id=doc_b.doc_id,
        chunk_index=0,
        content=doc_b.content,
        embedding=vec_b,
        source=doc_b.metadata.source,
        document_type=doc_b.metadata.document_type,
        department=doc_b.metadata.department,
        tags=doc_b.metadata.tags,
    )

    chunk_c = create_test_chunk(
        doc_id=doc_c.doc_id,
        chunk_index=0,
        content=doc_c.content,
        embedding=vec_c,
        source=doc_c.metadata.source,
        document_type=doc_c.metadata.document_type,
        department=doc_c.metadata.department,
        tags=doc_c.metadata.tags,
    )

    repo._documents[doc_a.doc_id] = doc_a
    repo._documents[doc_b.doc_id] = doc_b
    repo._documents[doc_c.doc_id] = doc_c

    repo._chunks[chunk_a.chunk_id] = chunk_a
    repo._chunks[chunk_b.chunk_id] = chunk_b
    repo._chunks[chunk_c.chunk_id] = chunk_c

    return repo


def test_bm25_tokenization() -> None:
    """Test BM25 word tokenization and normalization."""
    tokens = tokenize("Service Bulletin SB-2026-X99: Fault code ERR-7701!")
    assert "service" in tokens
    assert "bulletin" in tokens
    assert "sb-2026-x99" in tokens or "sb" in tokens
    assert "err-7701" in tokens or "err" in tokens


def test_bm25_index_scoring() -> None:
    """Test direct Okapi BM25 index building and term scoring."""
    index = BM25Index(k1=1.5, b=0.75)
    c1 = create_test_chunk(
        "d1", 0, "Python asynchronous programming with asyncio", [0.0] * 1536, "f1.txt"
    )
    c2 = create_test_chunk(
        "d2", 0, "Database indexing with postgres and pgvector", [0.0] * 1536, "f2.txt"
    )

    index.build_index([c1, c2])
    assert index.corpus_size == 2

    # Query for Python asyncio
    results = index.score("asyncio programming", top_k=5)
    assert len(results) == 1
    assert results[0][0].chunk_id == "d1_chunk_0"
    assert results[0][1] > 0.0

    # Query for PostgreSQL
    results_db = index.score("postgres pgvector", top_k=5)
    assert len(results_db) == 1
    assert results_db[0][0].chunk_id == "d2_chunk_0"


@pytest.mark.asyncio
async def test_semantic_only_retrieval(
    settings: Settings,
    hybrid_test_repo: InMemoryVectorRepository,
) -> None:
    """Semantic retriever excels at matching conceptual questions without exact terms."""
    embedding_service = EmbeddingService(settings=settings)
    vector_retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_repository=hybrid_test_repo,
    )

    # Conceptual search
    citations = await vector_retriever.retrieve(
        query="airplane structural integrity and passenger safety",
        top_k=3,
        min_similarity=-1.0,
    )
    assert len(citations) > 0
    assert all(c.similarity is not None for c in citations)


@pytest.mark.asyncio
async def test_keyword_only_retrieval(
    hybrid_test_repo: InMemoryVectorRepository,
) -> None:
    """Keyword BM25 retriever excels at pinpointing exact alphanumeric codes and IDs."""
    bm25_retriever = BM25Retriever(vector_repository=hybrid_test_repo)

    # Search by exact technical code
    citations = await bm25_retriever.retrieve(
        query="ERR-7701 SB-2026-X99",
        top_k=3,
    )
    assert len(citations) >= 1
    assert citations[0].doc_id == "doc_lexical_code"
    assert "ERR-7701" in citations[0].content


@pytest.mark.asyncio
async def test_hybrid_retrieval_reciprocal_rank_fusion(
    settings: Settings,
    hybrid_test_repo: InMemoryVectorRepository,
) -> None:
    """Hybrid retrieval fuses semantic and keyword results into a single ranked list."""
    embedding_service = EmbeddingService(settings=settings)
    vector_retriever = VectorRetriever(
        embedding_service=embedding_service,
        vector_repository=hybrid_test_repo,
    )
    bm25_retriever = BM25Retriever(vector_repository=hybrid_test_repo)
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fusion_strategy="rrf",
        rrf_k=60,
    )

    # Query containing both conceptual theme and exact certification code
    citations = await hybrid_retriever.retrieve(
        query="aerodynamic structural integrity FAA-CERT-4491",
        top_k=3,
    )
    assert len(citations) > 0
    # The hybrid doc should be ranked top due to mutual reinforcement from both retrievers
    assert citations[0].doc_id == "doc_hybrid_composite"


def test_reciprocal_rank_fusion_logic() -> None:
    """Unit test RRF combination with synthetic citations."""
    c1 = create_test_chunk("d1", 0, "Chunk 1", [0.0] * 1536, "f1.txt")
    c2 = create_test_chunk("d2", 0, "Chunk 2", [0.0] * 1536, "f2.txt")
    c3 = create_test_chunk("d3", 0, "Chunk 3", [0.0] * 1536, "f3.txt")

    from app.services.rag.models import Citation

    cite1 = Citation(
        chunk_id=c1.chunk_id,
        doc_id=c1.doc_id,
        source="f1.txt",
        file_type="txt",
        chunk_index=0,
        content=c1.content,
        similarity=0.9,
    )
    cite2 = Citation(
        chunk_id=c2.chunk_id,
        doc_id=c2.doc_id,
        source="f2.txt",
        file_type="txt",
        chunk_index=0,
        content=c2.content,
        similarity=0.8,
    )
    cite3 = Citation(
        chunk_id=c3.chunk_id,
        doc_id=c3.doc_id,
        source="f3.txt",
        file_type="txt",
        chunk_index=0,
        content=c3.content,
        similarity=0.7,
    )

    # List 1 ranks: [cite1, cite2]
    # List 2 ranks: [cite2, cite3]
    fused = reciprocal_rank_fusion(
        dense_citations=[cite1, cite2],
        sparse_citations=[cite2, cite3],
        k=60,
        top_k=3,
    )
    assert len(fused) == 3
    # cite2 appeared in both lists, so its combined RRF score is highest
    assert fused[0].chunk_id == cite2.chunk_id


def test_weighted_score_fusion_logic() -> None:
    """Unit test weighted score fusion with synthetic citations."""
    from app.services.rag.models import Citation

    cite1 = Citation(
        chunk_id="c1",
        doc_id="d1",
        source="f1.txt",
        file_type="txt",
        chunk_index=0,
        content="C1",
        similarity=1.0,
    )
    cite2 = Citation(
        chunk_id="c2",
        doc_id="d2",
        source="f2.txt",
        file_type="txt",
        chunk_index=0,
        content="C2",
        similarity=0.2,
    )

    fused = weighted_score_fusion(
        dense_citations=[cite1],
        sparse_citations=[cite2],
        alpha=0.5,
        top_k=2,
    )
    assert len(fused) == 2


def test_create_retriever_factory(settings: Settings) -> None:
    """Test create_retriever factory instantiation for all supported modes."""
    embedding_service = EmbeddingService(settings=settings)
    repo = InMemoryVectorRepository()

    ret_hybrid = create_retriever(
        embedding_service=embedding_service,
        vector_repository=repo,
        mode="hybrid",
    )
    assert isinstance(ret_hybrid, HybridRetriever)

    ret_vector = create_retriever(
        embedding_service=embedding_service,
        vector_repository=repo,
        mode="semantic",
    )
    assert isinstance(ret_vector, VectorRetriever)

    ret_bm25 = create_retriever(
        embedding_service=embedding_service,
        vector_repository=repo,
        mode="keyword",
    )
    assert isinstance(ret_bm25, BM25Retriever)

    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        create_retriever(
            embedding_service=embedding_service,
            vector_repository=repo,
            mode="invalid_strategy",
        )


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_metadata_filter(
    settings: Settings,
    hybrid_test_repo: InMemoryVectorRepository,
) -> None:
    """Verify hybrid retrieval strictly respects domain metadata filters."""
    retriever = create_retriever(
        embedding_service=EmbeddingService(settings=settings),
        vector_repository=hybrid_test_repo,
        mode="hybrid",
    )

    # Filter for department="Maintenance"
    citations = await retriever.retrieve(
        query="structural integrity",
        filters={"department": "Maintenance"},
    )
    assert len(citations) == 1
    assert citations[0].department == "Maintenance"
    assert citations[0].doc_id == "doc_lexical_code"


@pytest.mark.asyncio
async def test_rag_query_endpoint_modes(settings: Settings) -> None:
    """Verify POST /api/v1/rag/query executes with mode='hybrid', 'semantic', 'keyword'."""
    app = create_app(settings=settings)

    # Ingest document
    doc_service = app.state.document_service
    res = doc_service.ingest_bytes(
        content_bytes=b"NASA Artemis spacecraft avionics telemetry protocol AP-2026-X.",
        source_name="artemis_telemetry.txt",
        custom_metadata={"document_type": "avionics_spec"},
    )
    await doc_service.embed_and_index_document(res.doc_id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        for mode in ("hybrid", "semantic", "keyword"):
            payload = {
                "question": "What is the avionics telemetry protocol AP-2026-X?",
                "mode": mode,
                "top_k": 2,
                "min_similarity": -1.0,
            }
            resp = await client.post("/api/v1/rag/query", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["retrieval_mode"] == mode
            assert len(data["citations"]) > 0
            assert "answer" in data
