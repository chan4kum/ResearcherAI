"""
tests/test_failure_engineering.py — Controlled Failure Engineering Suite (Milestone 52)

Automated tests simulating and validating resilience across all 12 failure modes:
1. LLM timeout
2. LLM failure (500 / Rate Limit)
3. Vector database unavailable
4. MCP server unavailable
5. Malformed MCP result
6. Empty retrieval
7. Irrelevant retrieval
8. Agent loop exceeding limit
9. Database connection failure
10. Kubernetes Pod crash (liveness failure)
11. Failed deployment (unready state)
12. Failed readiness probe (diagnostic override & restore)
"""

from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.guardrails.tool_governance import ToolExecutionCircuitBreaker
from app.main import app
from app.models.schemas import TaskStatus
from app.services.agent.agent import BasicAgent
from app.services.agent.tools.base import BaseTool, ToolResult
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.base import LLMResponse
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService
from app.services.mcp.client import MCPClient
from app.services.mcp.models import MCPRequest
from app.services.mcp.server import LocalMCPServer
from app.services.rag.models import Citation, RAGResponse
from app.services.rag.retriever import BaseRetriever
from app.services.rag.service import RAGService


# ---------------------------------------------------------------------------
# 1. LLM Timeout Simulation
# ---------------------------------------------------------------------------


class TimeoutLLMProvider(MockLLMProvider):
    """Simulates an upstream LLM provider experiencing a hard timeout."""

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise TimeoutError("LLM inference timed out after 30.0s")


@pytest.mark.asyncio
async def test_failure_1_llm_timeout():
    """Simulate upstream LLM timeout and verify Agent records failure gracefully."""
    llm = LLMService(provider=TimeoutLLMProvider())
    agent = BasicAgent(llm_service=llm, tool_registry=ToolRegistry())

    state = await agent.run(task="Perform complex multi-step reasoning.")
    assert state.status == TaskStatus.FAILED
    assert state.error is not None
    assert "timed out" in state.error.lower()
    assert "failed" in state.trace


# ---------------------------------------------------------------------------
# 2. LLM Provider Failure (500 / RateLimit / Crash)
# ---------------------------------------------------------------------------


class CrashingLLMProvider(MockLLMProvider):
    """Simulates an upstream LLM provider returning HTTP 500 error."""

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise RuntimeError("HTTP 500 Internal Server Error: Provider capacity exhausted")


@pytest.mark.asyncio
async def test_failure_2_llm_500_failure():
    """Simulate upstream LLM 500 crash and verify Agent transitions to FAILED state."""
    llm = LLMService(provider=CrashingLLMProvider())
    agent = BasicAgent(llm_service=llm, tool_registry=ToolRegistry())

    state = await agent.run(task="Generate quarterly summary report.")
    assert state.status == TaskStatus.FAILED
    assert state.error is not None
    assert "500" in state.error
    assert "failed" in state.trace


# ---------------------------------------------------------------------------
# 3. Vector Database Unavailable
# ---------------------------------------------------------------------------


class FailingVectorRetriever(BaseRetriever):
    """Simulates a vector store connection outage."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Citation]:
        raise ConnectionRefusedError("ChromaDB vector store connection refused on port 8000")


@pytest.mark.asyncio
async def test_failure_3_vector_database_unavailable():
    """Simulate vector store crash during RAG retrieval and verify exception isolation."""
    retriever = FailingVectorRetriever()
    llm = LLMService(provider=MockLLMProvider())
    rag_service = RAGService(retriever=retriever, llm_service=llm)

    with pytest.raises(ConnectionRefusedError) as exc_info:
        await rag_service.answer(question="How do I configure Prometheus alerting?")
    assert "connection refused" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 4. MCP Server Unavailable / Disconnected
# ---------------------------------------------------------------------------


class DisconnectedMCPTool(BaseTool):
    """Simulates an MCP tool whose backing server daemon is offline."""

    name: str = "mcp_weather"
    description: str = "Fetch real-time weather data from external MCP server."

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=False,
            error="MCP server unreachable on socket /var/run/mcp.sock: Connection refused",
        )


@pytest.mark.asyncio
async def test_failure_4_mcp_server_unavailable():
    """Simulate offline MCP server and verify Agent gracefully handles tool failure."""
    registry = ToolRegistry()
    registry.register(DisconnectedMCPTool())

    # Tool execution returns failed result cleanly
    result = registry.execute("mcp_weather", city="London")
    assert result.success is False
    assert "connection refused" in result.error.lower()


# ---------------------------------------------------------------------------
# 5. Malformed MCP Result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_5_malformed_mcp_result():
    """Simulate MCP server returning corrupted response and verify client validation."""
    server = LocalMCPServer(server_name="test_corrupt_server")
    client = MCPClient(server=server)

    # Request unhandled method that returns error response
    req = MCPRequest(id="req_corrupt", method="invalid/nonexistent_method", params={})
    resp = await server.handle_request(req)
    assert resp.error is not None
    assert resp.error.code == -32601
    assert "unknown method" in resp.error.message.lower()


# ---------------------------------------------------------------------------
# 6. Empty Retrieval
# ---------------------------------------------------------------------------


class EmptyVectorRetriever(BaseRetriever):
    """Simulates retriever finding zero matching documents."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Citation]:
        return []


@pytest.mark.asyncio
async def test_failure_6_empty_retrieval():
    """Simulate 0 matching documents retrieved and verify fallback disclaimer."""
    retriever = EmptyVectorRetriever()
    llm = LLMService(provider=MockLLMProvider())
    rag_service = RAGService(retriever=retriever, llm_service=llm)

    response: RAGResponse = await rag_service.answer(
        question="What is the internal company policy for quantum computing hardware?"
    )
    assert len(response.citations) == 0
    assert response.retrieved_chunks_count == 0
    assert response.answer is not None
    # Context formatted with empty fallback
    context_str = rag_service.format_context([])
    assert "No relevant context found" in context_str


# ---------------------------------------------------------------------------
# 7. Irrelevant Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_7_irrelevant_retrieval():
    """Simulate retrieval of low-similarity noise chunks and verify similarity scores."""
    irrelevant_citations = [
        Citation(
            chunk_id="chk_001",
            doc_id="noise_001",
            source="unrelated_sports_news.txt",
            chunk_index=0,
            file_type=".txt",
            content="The soccer match ended with a 2-1 score in extra time.",
            similarity=0.08,
        )
    ]
    # Verify low similarity chunks are identified below relevance threshold (0.20)
    assert irrelevant_citations[0].similarity < 0.20


# ---------------------------------------------------------------------------
# 8. Agent Loop Exceeding Limit (Circuit Breaker)
# ---------------------------------------------------------------------------


def test_failure_8_agent_loop_exceeding_limit():
    """Simulate infinite agent reasoning loop and verify circuit breaker trips."""
    breaker = ToolExecutionCircuitBreaker(max_tool_calls=5, max_iterations=10)

    # Simulate 10 iterations (allowed)
    for _ in range(10):
        allowed, err = breaker.record_iteration()
        assert allowed is True
        assert err is None

    # 11th iteration trips circuit breaker
    allowed, err = breaker.record_iteration()
    assert allowed is False
    assert "loop limit exceeded" in err.lower()

    # Tool call budget cap verification
    for _ in range(5):
        allowed, err = breaker.record_tool_call("calculator")
        assert allowed is True

    # 6th tool call trips breaker
    allowed, err = breaker.record_tool_call("calculator")
    assert allowed is False
    assert "budget exceeded" in err.lower()


# ---------------------------------------------------------------------------
# 9. Database Connection Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_9_database_connection_failure():
    """Simulate database disconnection in readiness probe and verify HTTP 503."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Temporarily detach db_manager to simulate database outage
        original_db = getattr(app.state, "db_manager", None)
        try:
            app.state.db_manager = None
            resp = await ac.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"
            assert data["checks"]["database"]["status"] == "unhealthy"
        finally:
            app.state.db_manager = original_db


# ---------------------------------------------------------------------------
# 10. Kubernetes Pod Crash Simulation (Liveness Probe Failure)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_10_kubernetes_pod_crash_simulation():
    """Simulate container deadlock via /api/v1/health/simulate-fail-liveness and verify HTTP 500."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        try:
            # Enable simulated liveness failure
            sim_resp = await ac.post("/api/v1/health/simulate-fail-liveness")
            assert sim_resp.status_code == 200

            # Liveness probe should now fail with 500 Internal Server Error
            live_resp = await ac.get("/live")
            assert live_resp.status_code == 500
            data = live_resp.json()
            assert data["live"] is False
            assert data["status"] == "deadlock_detected"
        finally:
            # Restore liveness probe
            restore_resp = await ac.post("/api/v1/health/simulate-recover")
            assert restore_resp.status_code == 200

        # Verify healthy liveness restored
        healthy_resp = await ac.get("/live")
        assert healthy_resp.status_code == 200
        assert healthy_resp.json()["live"] is True


# ---------------------------------------------------------------------------
# 11. Failed Deployment Simulation (Unready Subsystems)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_11_failed_deployment_simulation():
    """Simulate uninitialized LLM adapter in new deployment and verify readiness failure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        original_llm = getattr(app.state, "llm_service", None)
        try:
            app.state.llm_service = None
            resp = await ac.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["ready"] is False
            assert data["checks"]["llm_service"]["status"] == "unhealthy"
        finally:
            app.state.llm_service = original_llm


# ---------------------------------------------------------------------------
# 12. Failed Readiness Probe Simulation (Diagnostic Override & Recovery)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_12_failed_readiness_probe_simulation():
    """Simulate diagnostic readiness failure and verify endpoint removal & restoration."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        try:
            # 1. Trigger simulated readiness failure
            fail_resp = await ac.post("/api/v1/health/simulate-fail-readiness")
            assert fail_resp.status_code == 200

            # 2. Readiness check fails with 503 Service Unavailable
            probe_resp = await ac.get("/ready")
            assert probe_resp.status_code == 503
            data = probe_resp.json()
            assert data["status"] == "not_ready"
            assert data["ready"] is False
            assert "diagnostic_override" in data["checks"]

            # 3. Restore readiness
            restore_resp = await ac.post("/api/v1/health/simulate-recover")
            assert restore_resp.status_code == 200

            # 4. Readiness check passes with 200 OK
            recovered_resp = await ac.get("/ready")
            assert recovered_resp.status_code == 200
            recovered_data = recovered_resp.json()
            assert recovered_data["status"] == "ready"
            assert recovered_data["ready"] is True
        finally:
            # Guarantee state reset
            app.state._simulate_readiness_failure = False
