import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_v1_health_returns_200(client: AsyncClient) -> None:
    """Verify that versioned GET /api/v1/health returns HTTP 200."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0-test"
    assert data["environment"] == "test"


@pytest.mark.asyncio
async def test_get_v1_info_returns_200(client: AsyncClient) -> None:
    """Verify that GET /api/v1/info returns platform metadata."""
    response = await client.get("/api/v1/info")
    assert response.status_code == 200

    data = response.json()
    assert data["app_name"] == "Enterprise Agentic Test Platform"
    assert data["version"] == "0.1.0-test"
    assert data["environment"] == "test"
    assert data["api_version"] == "/api/v1"
