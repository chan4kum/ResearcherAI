import pytest
from app.services.document.chunker import DocumentChunker
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.models import ChunkingConfig
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding.service import EmbeddingService


@pytest.mark.asyncio
async def test_embedding_service_embed_single_and_batch() -> None:
    """Verify EmbeddingService embeds single string and text batches."""
    provider = MockEmbeddingProvider(dimensions=64)
    service = EmbeddingService(provider=provider)

    single_vec = await service.embed_text("Semiconductor manufacturing")
    assert len(single_vec) == 64

    batch_resp = await service.embed_texts(["Wafer", "Etching", "Photolithography"])
    assert batch_resp.dimensions == 64
    assert len(batch_resp.embeddings) == 3
    assert batch_resp.embeddings[0] != batch_resp.embeddings[1]


@pytest.mark.asyncio
async def test_embedding_service_embed_chunks() -> None:
    """Verify embed_chunks generates EmbeddedChunk list with preserved metadata."""
    content = "Photolithography transfers circuit patterns onto silicon wafers using light."
    doc = DocumentLoaderFactory.load_bytes(content.encode("utf-8"), "lithography.txt")
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc, ChunkingConfig(chunk_size=40, chunk_overlap=10))

    assert len(chunks) >= 2

    provider = MockEmbeddingProvider(dimensions=128)
    service = EmbeddingService(provider=provider)

    embedded_chunks = await service.embed_chunks(chunks)
    assert len(embedded_chunks) == len(chunks)

    for i, ec in enumerate(embedded_chunks):
        assert ec.chunk_id == chunks[i].chunk_id
        assert ec.doc_id == doc.doc_id
        assert len(ec.embedding) == 128
        assert ec.metadata.source == "lithography.txt"
        assert ec.content == chunks[i].content
        # Check attached to source chunk
        assert chunks[i].embedding == ec.embedding


def test_embedding_service_similarity_calculation() -> None:
    """Verify similarity computation method on EmbeddingService."""
    service = EmbeddingService(provider=MockEmbeddingProvider(dimensions=10))
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    v3 = [0.0, 1.0]

    assert pytest.approx(service.compute_similarity(v1, v2)) == 1.0
    assert pytest.approx(service.compute_similarity(v1, v3)) == 0.0
