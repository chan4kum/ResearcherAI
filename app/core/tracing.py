"""
app/core/tracing.py — OpenTelemetry Distributed Tracing for Agentic Platform

Architecture:
  API Request → Agent → Planner → Retrieval → Vector Search → MCP/Tool → LLM → Response

Design decisions:
- ConsoleSpanExporter for local verification (no collector required).
- OTLP gRPC exporter enabled via OTEL_EXPORTER_OTLP_ENDPOINT env var when set.
- Resource attributes pinned to service.name / service.version.
- NO sensitive prompt/document content in span attributes. Only metadata.
- Span errors propagate OTEL StatusCode.ERROR with exception recording.
"""

import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import NonRecordingSpan, Span, StatusCode

from app.core.logging import get_logger

logger = get_logger("app.core.tracing")

_tracer_provider: TracerProvider | None = None

SERVICE_NAME = "agentic-platform"
SERVICE_VERSION = "1.0.0"

# Attributes that must NEVER appear in spans (security)
_REDACTED_ATTR_PREFIXES = ("prompt", "document", "content", "password", "token", "secret", "key")


def _is_safe_attribute(name: str) -> bool:
    """Return False for any attribute name that could leak sensitive content."""
    lower = name.lower()
    return not any(lower.startswith(p) for p in _REDACTED_ATTR_PREFIXES)


def configure_tracing() -> TracerProvider:
    """Initialize the global OpenTelemetry TracerProvider.

    Exporters:
    - ConsoleSpanExporter: Always active for local verification.
    - OTLPSpanExporter: Active when OTEL_EXPORTER_OTLP_ENDPOINT is set.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "deployment.environment": os.environ.get("ENVIRONMENT", "local"),
        }
    )

    provider = TracerProvider(resource=resource)

    # Console exporter — always active for local visibility
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(console_exporter))

    # OTLP gRPC exporter — active when collector endpoint configured
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("otel_otlp_exporter_configured", endpoint=otlp_endpoint)
    else:
        logger.info(
            "otel_console_exporter_active",
            hint="Set OTEL_EXPORTER_OTLP_ENDPOINT to send traces to a collector.",
        )

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    logger.info("opentelemetry_tracing_initialized", service=SERVICE_NAME)
    return provider


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer bound to the global provider."""
    return trace.get_tracer(name, SERVICE_VERSION)


def safe_span_attributes(attributes: dict) -> dict:
    """Filter out any attribute names that could expose sensitive content."""
    return {k: v for k, v in attributes.items() if _is_safe_attribute(k)}


@contextmanager
def agent_span(
    name: str,
    *,
    task_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    strategy: str | None = None,
    tool_name: str | None = None,
    extra: dict | None = None,
) -> Generator[Span, None, None]:
    """Context manager creating a named span with safe metadata attributes.

    Redacts: prompt text, document content, raw LLM output, credentials.
    Records: IDs, model names, strategies, tool names, counts.
    """
    tracer = get_tracer("app.agent")
    attrs: dict = {}
    if task_id:
        attrs["agent.task_id"] = task_id
    if model:
        attrs["llm.model"] = model
    if provider:
        attrs["llm.provider"] = provider
    if strategy:
        attrs["retrieval.strategy"] = strategy
    if tool_name:
        attrs["tool.name"] = tool_name
    if extra:
        for k, v in extra.items():
            if _is_safe_attribute(k):
                attrs[k] = v

    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def get_current_trace_id() -> str:
    """Return hex trace ID of the active span for log correlation."""
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return "no-trace"


def get_current_span_id() -> str:
    """Return hex span ID of the active span for log correlation."""
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.span_id, "016x")
    return "no-span"
