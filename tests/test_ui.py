"""
tests/test_ui.py — Frontend & UI Integration Tests

Validates:
1. Root SPA route (GET /) serves index.html with valid semantic markup.
2. Static CSS (GET /static/css/style.css) is accessible and valid.
3. Static JS (GET /static/js/app.js) is accessible and valid.
4. UI semantic elements: Research input, example prompt chips, results view, progress stepper, sources list, history drawer, document upload modal.
5. End-to-end user research request flow via the underlying API client contract.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root_ui_serves_html():
    """Verify that GET / returns the Single Page Application index.html."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        body = response.text
        assert "<!DOCTYPE html>" in body
        assert "Agentic Research" in body
        assert "researchInput" in body
        assert "btnSubmitResearch" in body
        assert "landingState" in body
        assert "resultsState" in body
        assert "progressCard" in body
        assert "historyDrawer" in body
        assert "documentModal" in body


@pytest.mark.asyncio
async def test_static_css_accessible():
    """Verify that stylesheet is served properly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/css/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        css_body = response.text
        assert "--bg-app" in css_body
        assert ".search-box-wrapper" in css_body
        assert ".progress-steps" in css_body


@pytest.mark.asyncio
async def test_static_js_accessible():
    """Verify that client JS application is served properly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/static/js/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")
        js_body = response.text
        assert "handleStartResearch" in js_body
        assert "simulateProgressSteps" in js_body
        assert "renderMarkdown" in js_body


@pytest.mark.asyncio
async def test_ui_research_flow_rag():
    """Simulate a complete user research request flow from the UI to the backend."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "question": "What are microservice design patterns?",
            "strategy": "normal",
            "top_k": 5,
            "rerank": True,
        }
        response = await client.post("/api/v1/rag/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "citations" in data
        assert isinstance(data["citations"], list)


@pytest.mark.asyncio
async def test_ui_agent_task_flow():
    """Simulate a user research request using Agent & Tools mode from the UI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "task": "Compute 1024 divided by 8 and summarize.",
        }
        response = await client.post("/api/v1/tasks", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "task" in data
        assert "plan" in data
        assert len(data["plan"]) > 0


@pytest.mark.asyncio
async def test_ui_document_upload_flow():
    """Simulate user document drag-and-drop ingestion from the UI."""
    import uuid

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        unique_id = uuid.uuid4().hex[:8]
        files = {
            "file": (f"doc_{unique_id}.txt", f"Document content {unique_id} for agent testing.".encode(), "text/plain")
        }
        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert data["status"] in ("ingested", "skipped_duplicate")
        assert data["word_count"] > 0
