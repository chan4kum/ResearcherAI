import pytest
from app.core.errors import BadRequestException, NotFoundException, register_exception_handlers
from app.core.middleware import register_middlewares
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_not_found_error_format(client: AsyncClient) -> None:
    """Verify 404 returns structured JSON error response."""
    response = await client.get("/non-existent-endpoint")
    assert response.status_code == 404

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
    assert "Not Found" in data["error"]["message"]
    assert "request_id" in data["error"]


@pytest.mark.asyncio
async def test_custom_app_exceptions() -> None:
    """Verify custom application exceptions return proper code, status and details."""
    app = FastAPI()
    register_middlewares(app)
    register_exception_handlers(app)

    @app.get("/trigger-not-found")
    async def trigger_not_found() -> None:
        raise NotFoundException(message="Custom knowledge item not found", details={"id": 42})

    @app.get("/trigger-bad-request")
    async def trigger_bad_request() -> None:
        raise BadRequestException(message="Invalid parameters provided")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res_404 = await client.get("/trigger-not-found")
        assert res_404.status_code == 404
        body_404 = res_404.json()["error"]
        assert body_404["code"] == "NOT_FOUND"
        assert body_404["message"] == "Custom knowledge item not found"
        assert body_404["details"] == {"id": 42}

        res_400 = await client.get("/trigger-bad-request")
        assert res_400.status_code == 400
        body_400 = res_400.json()["error"]
        assert body_400["code"] == "BAD_REQUEST"
        assert body_400["message"] == "Invalid parameters provided"
