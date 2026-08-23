import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_post_embeddings_generate(client: AsyncClient) -> None:
    """Verify POST /api/v1/embeddings/generate returns embeddings batch."""
    payload = {"texts": ["Quantum mechanics", "General relativity", "Standard Model"]}
    response = await client.post("/api/v1/embeddings/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_embeddings"] == 3
    assert len(data["embeddings"]) == 3
    assert data["dimensions"] > 0
    assert isinstance(data["embeddings"][0], list)


@pytest.mark.asyncio
async def test_post_embeddings_similarity(client: AsyncClient) -> None:
    """Verify POST /api/v1/embeddings/similarity computes cosine similarity."""
    payload = {
        "text_a": "Supervised machine learning algorithms",
        "text_b": "Supervised machine learning algorithms",
    }
    response = await client.post("/api/v1/embeddings/similarity", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["text_a"] == payload["text_a"]
    assert data["text_b"] == payload["text_b"]
    assert pytest.approx(data["cosine_similarity"], abs=1e-4) == 1.0


@pytest.mark.asyncio
async def test_post_document_embed_endpoint(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/{doc_id}/embed runs Document -> Chunk -> Embed pipeline."""
    # 1. Ingest document
    upload_res = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"Silicon crystal growth and wafer production.",
                "text/plain",
            )
        },
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]

    # 2. Embed document
    embed_res = await client.post(
        f"/api/v1/documents/{doc_id}/embed",
        json={"chunk_size": 30, "chunk_overlap": 5},
    )
    assert embed_res.status_code == 200
    data = embed_res.json()
    assert data["doc_id"] == doc_id
    assert data["total_chunks"] >= 1
    assert data["dimensions"] > 0
    assert len(data["chunks"]) == data["total_chunks"]

    # Check embedded chunk structure
    first_chunk = data["chunks"][0]
    assert "embedding" in first_chunk
    assert len(first_chunk["embedding"]) == data["dimensions"]
    assert first_chunk["doc_id"] == doc_id
    assert "metadata" in first_chunk


@pytest.mark.asyncio
async def test_post_document_embed_not_found(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/{non_existent}/embed returns 404."""
    response = await client.post("/api/v1/documents/non-existent-doc-id/embed")
    assert response.status_code == 404
