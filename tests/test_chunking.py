import pytest
from app.services.document.chunker import DocumentChunker, display_document_chunks
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.models import ChunkingConfig, Document, DocumentMetadata
from httpx import AsyncClient


def test_chunking_config_validation() -> None:
    """Verify validation raises error if overlap is equal to or greater than chunk size."""
    valid_config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    valid_config.validate_overlap()  # should not raise

    invalid_config1 = ChunkingConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="must be strictly less than"):
        invalid_config1.validate_overlap()

    invalid_config2 = ChunkingConfig(chunk_size=100, chunk_overlap=150)
    with pytest.raises(ValueError, match="must be strictly less than"):
        invalid_config2.validate_overlap()


def test_chunks_are_generated() -> None:
    """Verify chunker partitions text into multiple chunks when length exceeds chunk_size."""
    chunker = DocumentChunker()
    content = "A" * 1200
    meta = DocumentMetadata(
        doc_id="test_doc_1",
        source="large.txt",
        file_type="txt",
        checksum="dummy_checksum",
        character_count=1200,
        word_count=1,
    )
    doc = Document(doc_id="test_doc_1", content=content, metadata=meta)

    config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(doc, config=config)

    # With length 1200, chunk_size 500, step 400:
    # Chunk 0: 0-500
    # Chunk 1: 400-900
    # Chunk 2: 800-1200
    assert len(chunks) == 3
    assert chunks[0].metadata.index == 0
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char == 500
    assert len(chunks[0].content) == 500

    assert chunks[1].metadata.index == 1
    assert chunks[1].metadata.start_char == 400
    assert chunks[1].metadata.end_char == 900
    assert len(chunks[1].content) == 500

    assert chunks[2].metadata.index == 2
    assert chunks[2].metadata.start_char == 800
    assert chunks[2].metadata.end_char == 1200
    assert len(chunks[2].content) == 400


def test_chunk_overlap_content_match() -> None:
    """Verify consecutive chunks correctly share overlapping text content."""
    chunker = DocumentChunker()
    # 26 letters repeated 10 times = 260 characters
    alphabet_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    meta = DocumentMetadata(
        doc_id="alpha_doc",
        source="alphabet.txt",
        file_type="txt",
        checksum="alpha_sum",
        character_count=len(alphabet_text),
    )
    doc = Document(doc_id="alpha_doc", content=alphabet_text, metadata=meta)

    config = ChunkingConfig(chunk_size=100, chunk_overlap=30)
    chunks = chunker.chunk_document(doc, config=config)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        # Suffix of chunk i of length 30 must match prefix of chunk i+1 of length 30
        assert c1.content[-30:] == c2.content[:30]


def test_chunk_metadata_preservation() -> None:
    """Verify all parent metadata, custom metadata, and IDs are preserved across chunks."""
    chunker = DocumentChunker()
    doc_id = "doc_semiconductor_99"
    sentence = (
        "Semiconductor manufacturing involves multiple stages of "
        "photolithography and etching. "
    )
    content = sentence * 10
    meta = DocumentMetadata(
        doc_id=doc_id,
        source="semiconductor_guide.pdf",
        file_type="pdf",
        checksum="sha256_checksum_value",
        custom_metadata={"author": "Dr. Miller", "department": "Hardware", "confidential": True},
        character_count=len(content),
    )
    doc = Document(doc_id=doc_id, content=content, metadata=meta)

    config = ChunkingConfig(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(doc, config=config)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.doc_id == doc_id
        assert chunk.metadata.doc_id == doc_id
        assert chunk.metadata.chunk_id == f"{doc_id}_chunk_{idx}"
        assert chunk.metadata.source == "semiconductor_guide.pdf"
        assert chunk.metadata.file_type == "pdf"
        assert chunk.metadata.checksum == "sha256_checksum_value"
        assert chunk.metadata.custom_metadata == {
            "author": "Dr. Miller",
            "department": "Hardware",
            "confidential": True,
        }
        assert chunk.metadata.character_count == len(chunk.content)
        assert chunk.metadata.word_count == len(chunk.content.split())


def test_empty_document_handling() -> None:
    """Verify empty or whitespace-only documents return an empty chunk list."""
    chunker = DocumentChunker()
    meta = DocumentMetadata(
        doc_id="empty_doc",
        source="empty.txt",
        file_type="txt",
        checksum="empty_sum",
    )

    doc_empty = Document(doc_id="empty_doc", content="", metadata=meta)
    assert chunker.chunk_document(doc_empty) == []

    doc_spaces = Document(doc_id="empty_doc", content="   \n\n\t  ", metadata=meta)
    assert chunker.chunk_document(doc_spaces) == []


def test_very_small_document_handling() -> None:
    """Verify document smaller than chunk_size produces exactly 1 chunk with full content."""
    chunker = DocumentChunker()
    small_text = "Brief summary note."
    meta = DocumentMetadata(
        doc_id="small_doc",
        source="short.txt",
        file_type="txt",
        checksum="short_sum",
        character_count=len(small_text),
    )
    doc = Document(doc_id="small_doc", content=small_text, metadata=meta)

    config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(doc, config=config)

    assert len(chunks) == 1
    assert chunks[0].content == small_text
    assert chunks[0].metadata.index == 0
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char == len(small_text)
    assert chunks[0].metadata.chunk_id == "small_doc_chunk_0"


def test_display_document_chunks_visualizer() -> None:
    """Verify display_document_chunks outputs formatted ASCII diagram."""
    chunker = DocumentChunker()
    content = (
        "First paragraph about distributed systems.\n"
        "Second paragraph about consensus protocols."
    )
    doc = DocumentLoaderFactory.load_bytes(content.encode("utf-8"), "distributed.txt")
    chunks = chunker.chunk_document(doc, ChunkingConfig(chunk_size=40, chunk_overlap=10))

    diagram = display_document_chunks(doc, chunks)
    assert "DOCUMENT: [" in diagram
    assert "Source: distributed.txt" in diagram
    assert "→ CHUNKS (Total:" in diagram
    assert f"[{doc.doc_id}_chunk_0]" in diagram


@pytest.mark.asyncio
async def test_chunk_document_api_endpoints(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/{doc_id}/chunk and POST /api/v1/documents/chunk-text."""
    # 1. Ingest a document
    text = "Machine learning models require data preprocessing, training, evaluation, and serving."
    ingest_res = await client.post(
        "/api/v1/documents/ingest-text",
        json={"content": text, "source_name": "ml_guide.txt"},
    )
    assert ingest_res.status_code == 200
    doc_id = ingest_res.json()["doc_id"]

    # 2. Chunk the ingested document via API
    chunk_res = await client.post(
        f"/api/v1/documents/{doc_id}/chunk",
        json={"chunk_size": 30, "chunk_overlap": 5},
    )
    assert chunk_res.status_code == 200
    data = chunk_res.json()
    assert data["doc_id"] == doc_id
    assert data["total_chunks"] > 1
    assert len(data["chunks"]) == data["total_chunks"]
    assert data["preview_diagram"] is not None
    assert "DOCUMENT:" in data["preview_diagram"]

    # 3. Chunk ad-hoc text directly
    adhoc_res = await client.post(
        "/api/v1/documents/chunk-text",
        json={
            "text": "Ad-hoc text string for instant chunking without persistence.",
            "chunk_size": 25,
            "chunk_overlap": 5,
        },
    )
    assert adhoc_res.status_code == 200
    adhoc_data = adhoc_res.json()
    assert adhoc_data["total_chunks"] >= 2
