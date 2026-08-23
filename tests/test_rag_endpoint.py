import pytest
from app.config import Settings
from app.main import create_app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app_instance(settings: Settings):
    return create_app(settings=settings)


@pytest.mark.asyncio
async def test_post_rag_query_success(app_instance) -> None:
    # 1. Ingest and index a known document into application state
    doc_service = app_instance.state.document_service
    res = doc_service.ingest_file("tests/fixtures/sample.txt")
    await doc_service.embed_and_index_document(
        res.doc_id, chunk_size=120, chunk_overlap=20
    )

    # 2. Query RAG API
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        payload = {
            "question": "What is semiconductor fabrication?",
            "top_k": 3,
            "min_similarity": -1.0,
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["question"] == "What is semiconductor fabrication?"
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert data["retrieved_chunks_count"] > 0
        assert len(data["citations"]) > 0
        assert data["citations"][0]["source"] == "sample.txt"
        assert data["citations"][0]["file_type"] == "txt"
        assert "similarity" in data["citations"][0]
        assert data["model"] is not None
        assert data["provider"] is not None
        assert "total_tokens" in data["metadata"]


@pytest.mark.asyncio
async def test_post_rag_query_empty_question_validation_error(app_instance) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/rag/query", json={"question": ""})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_post_rag_query_custom_system_prompt(app_instance) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        payload = {
            "question": "Explain photolithography in detail",
            "top_k": 2,
            "system_prompt": "You are a senior hardware engineer answering technical inquiries.",
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "Explain photolithography in detail"
        assert "answer" in data


@pytest.mark.asyncio
async def test_post_rag_query_with_matching_metadata_filter(app_instance) -> None:
    # 1. Ingest document with custom domain metadata
    doc_service = app_instance.state.document_service
    res = doc_service.ingest_bytes(
        content_bytes=b"Boeing 787 composite wing stress report certified by FAA.",
        source_name="boeing_787_stress.txt",
        custom_metadata={
            "document_type": "quality_report",
            "department": "Aviation QA",
            "tags": ["boeing", "composite"],
        },
    )
    await doc_service.embed_and_index_document(res.doc_id, chunk_size=200, chunk_overlap=20)

    # 2. Query with matching filter
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        payload = {
            "question": "What does the Boeing quality report say?",
            "top_k": 3,
            "min_similarity": -1.0,
            "filters": {"document_type": "quality_report", "department": "Aviation QA"},
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieved_chunks_count"] > 0
        assert len(data["citations"]) > 0
        assert data["citations"][0]["document_type"] == "quality_report"
        assert data["citations"][0]["department"] == "Aviation QA"
        assert "boeing_787_stress.txt" in data["citations"][0]["source"]


@pytest.mark.asyncio
async def test_post_rag_query_with_no_matching_metadata_filter(app_instance) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as client:
        payload = {
            "question": "What does the Boeing quality report say?",
            "top_k": 3,
            "min_similarity": -1.0,
            "filters": {"document_type": "non_existent_department"},
        }
        resp = await client.post("/api/v1/rag/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieved_chunks_count"] == 0
        assert len(data["citations"]) == 0
        assert "do not contain information" in data["answer"]

