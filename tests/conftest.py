from collections.abc import AsyncIterator

import pytest
from app.config import Settings
from app.main import create_app
from app.services.document.service import DocumentService
from app.services.document.store import DocumentStore
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def settings() -> Settings:
    """Provide isolated test settings with in-memory storage."""
    return Settings(
        app_name="Enterprise Agentic Test Platform",
        app_version="0.1.0-test",
        app_env="test",
        debug=True,
        log_level="DEBUG",
        storage_dir="",
        vector_repository_type="in_memory",
        database_url="",
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """Provide an asynchronous HTTP test client bound to the FastAPI application."""
    doc_service = DocumentService(store=DocumentStore(storage_dir=None), settings=settings)
    test_app = create_app(settings=settings, document_service=doc_service)
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client

