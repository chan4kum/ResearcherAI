import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import (
    AGENT_EXECUTIONS_TOTAL,
    AGENT_TOOL_CALLS_TOTAL,
    HTTP_REQUESTS_TOTAL,
    LLM_CALLS_TOTAL,
    LLM_ESTIMATED_COST_DOLLARS_TOTAL,
    LLM_TOKENS_TOTAL,
    RETRIEVAL_EXECUTIONS_TOTAL,
    RETRIEVAL_ITERATIONS_TOTAL,
    calculate_estimated_cost,
    get_prometheus_metrics,
    record_llm_metrics,
)
from app.main import app


def test_calculate_estimated_cost() -> None:
    """Test token cost estimation logic."""
    cost_mini = calculate_estimated_cost("gpt-4o-mini", prompt_tokens=10_000, completion_tokens=5_000)
    assert cost_mini > 0
    # 10k * $0.15/1M = $0.0015; 5k * $0.60/1M = $0.003 -> total $0.0045
    assert round(cost_mini, 4) == 0.0045


def test_record_llm_metrics() -> None:
    """Test recording LLM invocations and token usage."""
    record_llm_metrics(
        model="gpt-4o-mini",
        provider="mock",
        status="success",
        duration_sec=0.12,
        prompt_tokens=100,
        completion_tokens=50,
    )

    content, media_type = get_prometheus_metrics()
    text = content.decode("utf-8")

    assert "llm_calls_total" in text
    assert 'model="gpt-4o-mini"' in text
    assert 'provider="mock"' in text
    assert "llm_tokens_total" in text
    assert "llm_estimated_cost_dollars_total" in text


@pytest.mark.asyncio
async def test_get_metrics_endpoint() -> None:
    """Test that GET /metrics returns standard Prometheus text format."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Generate some HTTP traffic
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200

        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "text/plain" in metrics_resp.headers["content-type"]

        body = metrics_resp.text
        assert "http_requests_total" in body
        assert "http_request_duration_seconds" in body
        assert "agent_executions_total" in body
        assert "agent_tool_calls_total" in body
        assert "retrieval_executions_total" in body
        assert "retrieval_iterations_total" in body


@pytest.mark.asyncio
async def test_agent_and_rag_metrics_increment() -> None:
    """Test that executing an agent task and RAG query updates metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        task_resp = await client.post(
            "/api/v1/tasks",
            json={"task": "Calculate 15 * 6"},
        )
        assert task_resp.status_code == 200

        metrics_resp = await client.get("/metrics")
        body = metrics_resp.text

        assert 'agent_executions_total{model="gpt-4o-mini",provider="mock",status="completed"}' in body or 'agent_executions_total' in body
        assert 'agent_tool_calls_total{status="success",tool_name="calculator"}' in body or 'agent_tool_calls_total' in body
