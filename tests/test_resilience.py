import asyncio

import pytest
from app.config import Settings
from app.core.errors import BudgetExceededException, CircuitBreakerOpenException
from app.core.resilience.circuit_breaker import CircuitBreaker, CircuitState
from app.core.resilience.cost_guardrail import CostBudgetTracker
from app.core.resilience.rate_limiter import SlidingWindowRateLimiter
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.registry import ToolRegistry


def test_rate_limiter_blocks_excessive_requests(settings: Settings) -> None:
    """Verify rate limiter blocks burst requests and returns 429 with Retry-After."""
    custom_limiter = SlidingWindowRateLimiter(
        requests_per_minute=5,
        burst_limit=2,
    )
    # 1. First 2 requests within burst limit
    allowed1, rem1, _ = custom_limiter.check_rate_limit("client_test_1")
    assert allowed1 is True
    assert rem1 == 4

    allowed2, rem2, _ = custom_limiter.check_rate_limit("client_test_1")
    assert allowed2 is True
    assert rem2 == 3

    # 3. Third request within same second exceeds burst limit (2)
    allowed3, rem3, retry_after = custom_limiter.check_rate_limit("client_test_1")
    assert allowed3 is False
    assert rem3 == 0
    assert retry_after >= 1.0


def test_max_tool_calls_budget_enforcement() -> None:
    """Verify ToolRegistry stops tool executions when CostBudgetTracker tool budget is reached."""
    tracker = CostBudgetTracker(max_tool_calls=2)
    calc_tool = CalculatorTool()
    registry = ToolRegistry(tools=[calc_tool], budget_tracker=tracker)

    # 1. First execution -> succeeds
    res1 = registry.execute("calculator", expression="10 + 10")
    assert res1.success is True
    assert res1.output in (20, "20")
    assert tracker.tool_calls == 1

    # 2. Second execution -> succeeds
    res2 = registry.execute("calculator", expression="5 * 5")
    assert res2.success is True
    assert res2.output in (25, "25")
    assert tracker.tool_calls == 2

    # 3. Third execution -> blocked by budget cap
    res3 = registry.execute("calculator", expression="100 / 4")
    assert res3.success is False
    assert "maximum tool call budget of 2 exceeded" in res3.error.lower()
    assert tracker.tool_calls == 2


def test_max_research_iterations_cap() -> None:
    """Verify CostBudgetTracker raises BudgetExceededException on limit breach."""
    tracker = CostBudgetTracker(max_research_iterations=3)

    tracker.record_research_iteration(1)
    tracker.record_research_iteration(2)
    tracker.record_research_iteration(3)
    assert tracker.research_iterations == 3

    with pytest.raises(BudgetExceededException, match="Maximum research iteration limit"):
        tracker.record_research_iteration(4)


@pytest.mark.asyncio
async def test_circuit_breaker_tripping_and_fast_fail() -> None:
    """Verify circuit breaker trips to OPEN after consecutive failures and fast-fails."""
    breaker = CircuitBreaker(
        name="mock-llm-service",
        failure_threshold=2,
        recovery_timeout_seconds=0.3,
    )
    assert breaker.state == CircuitState.CLOSED

    async def failing_operation() -> None:
        raise ConnectionResetError("Remote model endpoint connection dropped")

    # 1. First failure -> still CLOSED
    with pytest.raises(ConnectionResetError):
        await breaker.call_async(failing_operation)
    assert breaker.state == CircuitState.CLOSED

    # 2. Second failure -> trips to OPEN
    with pytest.raises(ConnectionResetError):
        await breaker.call_async(failing_operation)
    assert breaker.state == CircuitState.OPEN

    # 3. Third call -> fast-fails with CircuitBreakerOpenException without invoking func
    with pytest.raises(CircuitBreakerOpenException, match="temporarily unavailable"):
        await breaker.call_async(failing_operation)


@pytest.mark.asyncio
async def test_circuit_breaker_recovery_flow() -> None:
    """Verify circuit breaker transitions OPEN -> HALF_OPEN -> CLOSED on successful trial calls."""
    breaker = CircuitBreaker(
        name="recoverable-service",
        failure_threshold=1,
        recovery_timeout_seconds=0.2,
        half_open_success_threshold=2,
    )

    async def failing_call() -> None:
        raise RuntimeError("Service failure")

    async def healthy_call() -> str:
        return "healthy_result"

    # Trip to OPEN
    with pytest.raises(RuntimeError):
        await breaker.call_async(failing_call)
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout to transition to HALF_OPEN
    await asyncio.sleep(0.25)
    assert breaker.state == CircuitState.HALF_OPEN

    # 1. First trial call in HALF_OPEN -> succeeds
    r1 = await breaker.call_async(healthy_call)
    assert r1 == "healthy_result"
    assert breaker.state == CircuitState.HALF_OPEN

    # 2. Second trial call in HALF_OPEN -> reaches threshold (2), fully recovers to CLOSED
    r2 = await breaker.call_async(healthy_call)
    assert r2 == "healthy_result"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_request_cancellation_handling() -> None:
    """Verify async agent task cancellation cancels child coroutines cleanly."""
    cancellation_observed = False

    async def long_running_research_agent() -> str:
        nonlocal cancellation_observed
        try:
            await asyncio.sleep(2.0)
            return "finished"
        except asyncio.CancelledError:
            cancellation_observed = True
            raise

    task = asyncio.create_task(long_running_research_agent())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancellation_observed is True
