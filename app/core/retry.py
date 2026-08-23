import asyncio
import functools
import random
from collections.abc import Callable
from typing import Any, TypeVar

from app.core.logging import get_logger

logger = get_logger("app.core.retry")

T = TypeVar("T")


def calculate_backoff(
    attempt: int,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 5.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional random jitter."""
    delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    return delay


async def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 5.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    operation_name: str = "operation",
    **kwargs: Any,
) -> Any:
    """Execute an async callable with exponential backoff retries."""
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                logger.error(
                    "retry_attempts_exhausted",
                    operation=operation_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                raise

            delay = calculate_backoff(
                attempt=attempt,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                jitter=jitter,
            )
            logger.warning(
                "retry_transient_failure",
                operation=operation_name,
                attempt=attempt,
                next_delay_seconds=round(delay, 3),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc


def async_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 5.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    operation_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to retry async functions on transient failure."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        op_name = operation_name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await execute_with_retry(
                fn,
                *args,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                operation_name=op_name,
                **kwargs,
            )

        return wrapper

    return decorator
