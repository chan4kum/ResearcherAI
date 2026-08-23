import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from app.core.errors import CircuitBreakerOpenException
from app.core.logging import get_logger

logger = get_logger("app.core.resilience.circuit_breaker")

T = TypeVar("T")


class CircuitState(StrEnum):
    """Operational states of a Circuit Breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Enterprise circuit breaker protecting against cascading downstream dependency failures."""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 10.0,
        half_open_success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._consecutive_success_count = 0
        self._last_state_change_time = time.monotonic()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current evaluated circuit state with automatic half-open timeout evaluation."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_state_change_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._consecutive_success_count = 0
                    logger.info(
                        "circuit_breaker_transition_half_open",
                        circuit=self.name,
                        elapsed_seconds=round(elapsed, 2),
                    )
            return self._state

    def _record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_success_count += 1
                if self._consecutive_success_count >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._consecutive_success_count = 0
                    self._last_state_change_time = time.monotonic()
                    logger.info("circuit_breaker_recovered_closed", circuit=self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change_time = time.monotonic()
                    logger.error(
                        "circuit_breaker_tripped_open",
                        circuit=self.name,
                        failures=self._failure_count,
                        threshold=self.failure_threshold,
                        error=str(exc),
                    )
            elif self._state == CircuitState.HALF_OPEN:
                # Immediate transition back to OPEN on any trial failure
                self._state = CircuitState.OPEN
                self._last_state_change_time = time.monotonic()
                logger.warning(
                    "circuit_breaker_trial_failed_reopened",
                    circuit=self.name,
                    error=str(exc),
                )

    async def call_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async callable under circuit breaker protection."""
        current_st = self.state
        if current_st == CircuitState.OPEN:
            logger.warning("circuit_breaker_fast_fail", circuit=self.name, state=current_st)
            raise CircuitBreakerOpenException(
                message=f"Service '{self.name}' is temporarily unavailable (circuit open).",
                details={"circuit": self.name, "state": current_st.value},
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure(exc)
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous callable under circuit breaker protection."""
        current_st = self.state
        if current_st == CircuitState.OPEN:
            logger.warning("circuit_breaker_fast_fail", circuit=self.name, state=current_st)
            raise CircuitBreakerOpenException(
                message=f"Service '{self.name}' is temporarily unavailable (circuit open).",
                details={"circuit": self.name, "state": current_st.value},
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure(exc)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_success_count = 0
            self._last_state_change_time = time.monotonic()
            logger.info("circuit_breaker_manual_reset", circuit=self.name)
