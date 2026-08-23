import logging
import re
import sys
from typing import Any

import structlog

# Regex patterns for identifying sensitive credentials in strings
SENSITIVE_KEY_PATTERNS = re.compile(
    r"(api_?key|secret|password|passwd|auth|token|bearer|database_url|db_password|private_key|credential)",
    re.IGNORECASE,
)
BEARER_AUTH_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE)
BASIC_AUTH_PATTERN = re.compile(r"(Basic\s+)[A-Za-z0-9\+\/]+=*", re.IGNORECASE)
DB_URL_PASS_PATTERN = re.compile(r"(://[^:]+:)[^@]+(@)", re.IGNORECASE)


def redact_sensitive_value(key: str, value: Any) -> Any:
    """Recursively redact sensitive values matching credential patterns."""
    if isinstance(value, dict):
        return {k: redact_sensitive_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_value(key, item) for item in value]
    if isinstance(value, str):
        if SENSITIVE_KEY_PATTERNS.search(key):
            return "***REDACTED***"
        # Mask embedded bearer/basic tokens or DB connection string passwords
        scrubbed = BEARER_AUTH_PATTERN.sub(r"\1***REDACTED***", value)
        scrubbed = BASIC_AUTH_PATTERN.sub(r"\1***REDACTED***", scrubbed)
        scrubbed = DB_URL_PASS_PATTERN.sub(r"\1***REDACTED***\2", scrubbed)
        return scrubbed
    if SENSITIVE_KEY_PATTERNS.search(key):
        return "***REDACTED***"
    return value


def redact_secrets_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that masks sensitive credentials across all log keys."""
    return {k: redact_sensitive_value(k, v) for k, v in event_dict.items()}


def sanitize_documents_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that truncates verbose document contents to protect privacy."""
    doc_keys = {"document_content", "raw_text", "raw_content", "chunk_content", "full_text"}
    for key in list(event_dict.keys()):
        if key in doc_keys and isinstance(event_dict[key], str):
            val: str = event_dict[key]
            if len(val) > 120:
                event_dict[key] = f"{val[:100]}... [truncated {len(val)} chars]"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging with security scrubbing and contextvars."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets_processor,
        sanitize_documents_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def log_agent_stage(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    *,
    request_id: str | None = None,
    agent_stage: str | None = None,
    retrieval_stage: str | None = None,
    duration_ms: float | None = None,
    tool_calls: list[str] | list[dict[str, Any]] | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    error: str | None = None,
    user_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Helper to emit consistent, structured observability events across agent workflows."""
    payload: dict[str, Any] = {}
    if request_id:
        payload["request_id"] = request_id
    if agent_stage:
        payload["agent_stage"] = agent_stage
    if retrieval_stage:
        payload["retrieval_stage"] = retrieval_stage
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 2)
    if tool_calls is not None:
        payload["tool_calls"] = tool_calls
    if model or provider:
        payload["model_info"] = {"model": model, "provider": provider}
    if total_tokens is not None or prompt_tokens is not None:
        payload["token_info"] = {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
        }
    if error:
        payload["error"] = error
    if user_metadata:
        payload["user_metadata"] = user_metadata

    payload.update(kwargs)

    if error:
        logger.error(event, **payload)
    else:
        logger.info(event, **payload)


