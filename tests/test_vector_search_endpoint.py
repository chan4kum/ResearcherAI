import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_post_document_search_endpoint(client: AsyncClient) -> None:
    """Verify Document -> Chunk -> Embed -> Index -> Search full pipeline."""
    # 1. Ingest document
    content = (
        "Photolithography transfers geometric shapes on a mask to a thin layer "
        "of radiation-sensitive chemical on the surface of a semiconductor wafer."
    )
    upload_res = await client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "photolithography.txt",
                content.encode("utf-8"),
                "text/plain",
            )
        },
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]

    # 2. Embed and Index into Vector DB
    embed_res = await client.post(
        f"/api/v1/documents/{doc_id}/embed",
        json={"chunk_size": 60, "chunk_overlap": 10},
    )
    assert embed_res.status_code == 200
    assert embed_res.json()["total_chunks"] >= 2

    # 3. Perform Vector Similarity Search
    search_res = await client.post(
        "/api/v1/documents/search",
        json={
            "query": "semiconductor wafer masking",
            "top_k": 5,
            "min_similarity": -1.0,
        },
    )
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["query"] == "semiconductor wafer masking"
    assert data["total_results"] >= 1
    assert len(data["results"]) == data["total_results"]

    first_item = data["results"][0]
    assert "chunk_id" in first_item
    assert first_item["doc_id"] == doc_id
    assert "content" in first_item
    assert "similarity" in first_item
    assert "metadata" in first_item
    assert first_item["metadata"]["source"] == "photolithography.txt"


@pytest.mark.asyncio
async def test_post_document_search_validation_error(client: AsyncClient) -> None:
    """Verify validation error on empty query."""
    res = await client.post(
        "/api/v1/documents/search",
        json={"query": ""},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_post_sync_kb_endpoint(client: AsyncClient) -> None:
    """Verify synchronizing a directory containing documents into the vector database."""
    res = await client.post(
        "/api/v1/documents/sync-kb",
        json={"kb_dir": "tests/fixtures", "chunk_size": 80, "chunk_overlap": 15},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_files"] >= 1
    assert data["indexed_documents"] >= 1
    assert data["total_indexed_chunks"] >= 1

    # Search across newly synced fixture knowledge base
    search_res = await client.post(
        "/api/v1/documents/search",
        json={"query": "Semiconductor manufacturing", "top_k": 3},
    )
    assert search_res.status_code == 200
    assert search_res.json()["total_results"] >= 1

