import json
from io import StringIO
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from app.core.logging import (
    configure_logging,
    get_logger,
    log_agent_stage,
    redact_sensitive_value,
    sanitize_documents_processor,
)
from app.main import app


def test_redact_sensitive_value_keys() -> None:
    """Test that dictionary keys matching credential patterns are masked."""
    sensitive_data = {
        "api_key": "sk-proj-secret-token-12345",
        "user_password": "SuperSecretPassword123!",
        "database_url": "postgresql+asyncpg://postgres:my_secret_pass@db:5432/main",
        "auth_token": "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret",
        "safe_query": "What is quantum computing?",
        "nested": {
            "secret_key": "nested-secret-value",
            "model_name": "gpt-4o-mini",
        },
    }

    cleaned = {k: redact_sensitive_value(k, v) for k, v in sensitive_data.items()}

    assert cleaned["api_key"] == "***REDACTED***"
    assert cleaned["user_password"] == "***REDACTED***"
    assert cleaned["database_url"] == "***REDACTED***"
    assert cleaned["auth_token"] == "***REDACTED***"
    assert cleaned["safe_query"] == "What is quantum computing?"
    assert cleaned["nested"]["secret_key"] == "***REDACTED***"
    assert cleaned["nested"]["model_name"] == "gpt-4o-mini"


def test_sanitize_documents_processor_truncates_large_chunks() -> None:
    """Test that oversized document content is truncated in log events."""
    huge_text = "A" * 500
    event = {
        "event": "document_retrieved",
        "doc_id": "doc-001",
        "document_content": huge_text,
        "score": 0.95,
    }

    sanitized = sanitize_documents_processor(None, "info", event)
    assert len(sanitized["document_content"]) < 200
    assert "[truncated 500 chars]" in sanitized["document_content"]
    assert sanitized["doc_id"] == "doc-001"
    assert sanitized["score"] == 0.95


def test_log_agent_stage_emits_structured_telemetry() -> None:
    """Test log_agent_stage helper compiles complete observability attributes."""
    mock_logger = MagicMock()

    log_agent_stage(
        mock_logger,
        "agent_stage_execution",
        request_id="req-uuid-1234",
        agent_stage="synthesis",
        retrieval_stage="hybrid_fusion",
        duration_ms=45.67,
        tool_calls=["calculator", "web_search"],
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=150,
        completion_tokens=80,
        total_tokens=230,
        user_metadata={"role": "researcher", "tenant": "enterprise_a"},
        error=None,
    )

    mock_logger.info.assert_called_once()
    call_args, call_kwargs = mock_logger.info.call_args
    assert call_args[0] == "agent_stage_execution"
    assert call_kwargs["request_id"] == "req-uuid-1234"
    assert call_kwargs["agent_stage"] == "synthesis"
    assert call_kwargs["retrieval_stage"] == "hybrid_fusion"
    assert call_kwargs["duration_ms"] == 45.67
    assert call_kwargs["tool_calls"] == ["calculator", "web_search"]
    assert call_kwargs["model_info"] == {"model": "gpt-4o-mini", "provider": "openai"}
    assert call_kwargs["token_info"] == {
        "prompt_tokens": 150,
        "completion_tokens": 80,
        "total_tokens": 230,
    }
    assert call_kwargs["user_metadata"] == {"role": "researcher", "tenant": "enterprise_a"}


def test_log_agent_stage_error_routing() -> None:
    """Test that errors trigger error log level."""
    mock_logger = MagicMock()

    log_agent_stage(
        mock_logger,
        "agent_stage_failed",
        request_id="req-err-999",
        agent_stage="tool_execution",
        error="Tool timeout after 5000ms",
    )

    mock_logger.error.assert_called_once()
    call_args, call_kwargs = mock_logger.error.call_args
    assert call_args[0] == "agent_stage_failed"
    assert call_kwargs["error"] == "Tool timeout after 5000ms"
    assert call_kwargs["agent_stage"] == "tool_execution"


@pytest.mark.asyncio
async def test_http_request_logging_contextvars_binding() -> None:
    """Test that FastAPI requests bind correlation IDs and log structured events."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        custom_request_id = "test-observability-req-001"
        response = await client.get("/health", headers={"x-request-id": custom_request_id})

        assert response.status_code == 200
        assert response.headers["x-request-id"] == custom_request_id
        assert "x-process-time-ms" in response.headers
        assert "x-pod-name" in response.headers
