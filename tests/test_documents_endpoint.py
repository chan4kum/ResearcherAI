from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_text_document_endpoint(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/upload accepts text file and returns metadata."""
    file_content = b"Knowledge management architecture for enterprise agents."
    files = {"file": ("enterprise.txt", file_content, "text/plain")}

    response = await client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ingested"
    assert data["source"] == "enterprise.txt"
    assert data["doc_id"] != ""
    assert data["word_count"] == 6
    assert data["character_count"] == len(file_content)

    doc_id = data["doc_id"]

    # Verify document is retrievable via GET /api/v1/documents/{doc_id}
    get_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["content"] == file_content.decode("utf-8")


@pytest.mark.asyncio
async def test_upload_real_pdf_document_endpoint(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/upload accepts real PDF file and extracts text."""
    pdf_bytes = Path("tests/fixtures/sample.pdf").read_bytes()
    files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}

    response = await client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ingested"
    assert data["source"] == "sample.pdf"
    assert data["page_count"] == 1
    assert data["word_count"] > 0


@pytest.mark.asyncio
async def test_upload_duplicate_document_endpoint(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/upload marks duplicate content as skipped_duplicate."""
    content = b"Identical deduplication test content."
    files1 = {"file": ("original.txt", content, "text/plain")}

    # First upload
    res1 = await client.post("/api/v1/documents/upload", files=files1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "ingested"

    # Second upload with same bytes (even with different filename)
    files2 = {"file": ("duplicate_name.txt", content, "text/plain")}
    res2 = await client.post("/api/v1/documents/upload", files=files2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "skipped_duplicate"
    assert data2["doc_id"] == data1["doc_id"]
    assert data2["checksum"] == data1["checksum"]


@pytest.mark.asyncio
async def test_ingest_text_endpoint(client: AsyncClient) -> None:
    """Verify POST /api/v1/documents/ingest-text ingests raw string payloads."""
    payload = {
        "content": "Raw string content without file upload.",
        "source_name": "api_direct.txt",
        "custom_metadata": {"author": "engineer"},
    }
    response = await client.post("/api/v1/documents/ingest-text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ingested"
    assert data["source"] == "api_direct.txt"


@pytest.mark.asyncio
async def test_list_and_delete_document_endpoints(client: AsyncClient) -> None:
    """Verify GET /api/v1/documents and DELETE /api/v1/documents/{doc_id}."""
    payload = {"content": "Content to be deleted.", "source_name": "temp.txt"}
    create_res = await client.post("/api/v1/documents/ingest-text", json=payload)
    doc_id = create_res.json()["doc_id"]

    # List documents
    list_res = await client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1

    # Delete document
    del_res = await client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify 404 after deletion
    get_after = await client.get(f"/api/v1/documents/{doc_id}")
    assert get_after.status_code == 404
