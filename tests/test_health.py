import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_root_health_returns_200(client: AsyncClient) -> None:
    """Verify that root GET /health returns HTTP 200 and expected payload."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0-test"
    assert data["environment"] == "test"

    # Verify correlation header and timing header
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


@pytest.mark.asyncio
async def test_custom_request_id_propagated(client: AsyncClient) -> None:
    """Verify that a client-provided x-request-id is propagated in the response."""
    custom_id = "test-custom-request-id-12345"
    response = await client.get("/health", headers={"x-request-id": custom_id})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == custom_id
