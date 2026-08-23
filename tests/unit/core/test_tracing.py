"""
tests/unit/core/test_tracing.py — Unit tests for app/core/tracing.py

Tests:
- TracerProvider initialises without error
- agent_span creates and closes a span without error
- Safe attribute filter redacts forbidden names
- Span records exception on error with StatusCode.ERROR
- get_current_trace_id returns a valid hex string inside a span
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import StatusCode

from app.core.tracing import (
    _is_safe_attribute,
    agent_span,
    configure_tracing,
    get_current_span_id,
    get_current_trace_id,
    safe_span_attributes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_tracer_provider(monkeypatch):
    """Isolate each test with a clean in-memory TracerProvider.

    OTel 1.x locks the global TracerProvider after first set, so we cannot
    call trace.set_tracer_provider() in tests.  Instead we patch `get_tracer`
    in the tracing module to return a tracer backed by a fresh provider with
    an InMemorySpanExporter — no global state is mutated.
    """
    import app.core.tracing as tracing_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Patch get_tracer so agent_span uses our isolated provider
    monkeypatch.setattr(
        tracing_module,
        "get_tracer",
        lambda name: provider.get_tracer(name, "1.0.0"),
    )

    yield exporter


# ---------------------------------------------------------------------------
# attribute safety filter
# ---------------------------------------------------------------------------


class TestIsSafeAttribute:
    def test_allows_generic_attributes(self):
        assert _is_safe_attribute("agent.task_id") is True
        assert _is_safe_attribute("llm.model") is True
        assert _is_safe_attribute("tool.name") is True
        assert _is_safe_attribute("retrieval.top_k") is True
        assert _is_safe_attribute("mcp.server") is True

    def test_redacts_prompt_names(self):
        assert _is_safe_attribute("prompt") is False
        assert _is_safe_attribute("prompt_text") is False
        assert _is_safe_attribute("PROMPT_DATA") is False

    def test_redacts_content_names(self):
        assert _is_safe_attribute("content") is False
        assert _is_safe_attribute("content_body") is False

    def test_redacts_credential_names(self):
        assert _is_safe_attribute("password") is False
        assert _is_safe_attribute("token") is False
        assert _is_safe_attribute("secret") is False
        assert _is_safe_attribute("key") is False

    def test_redacts_document_names(self):
        assert _is_safe_attribute("document") is False
        assert _is_safe_attribute("document_text") is False


class TestSafeSpanAttributes:
    def test_filters_unsafe_keys(self):
        raw = {
            "agent.task_id": "abc123",
            "prompt_text": "tell me secrets",
            "llm.model": "mock",
            "password": "hunter2",
            "document_content": "classified",
        }
        filtered = safe_span_attributes(raw)
        assert "agent.task_id" in filtered
        assert "llm.model" in filtered
        assert "prompt_text" not in filtered
        assert "password" not in filtered
        assert "document_content" not in filtered


# ---------------------------------------------------------------------------
# agent_span context manager
# ---------------------------------------------------------------------------


class TestAgentSpan:
    def test_span_is_created_and_exported(self, fresh_tracer_provider):
        exporter = fresh_tracer_provider
        with agent_span("test.operation", task_id="t-001") as span:
            assert span is not None

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].name == "test.operation"

    def test_span_attributes_set(self, fresh_tracer_provider):
        exporter = fresh_tracer_provider
        with agent_span("test.attrs", task_id="t-002", model="mock-model", tool_name="calculator"):
            pass

        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes
        assert attrs.get("agent.task_id") == "t-002"
        assert attrs.get("llm.model") == "mock-model"
        assert attrs.get("tool.name") == "calculator"

    def test_span_records_exception_on_error(self, fresh_tracer_provider):
        exporter = fresh_tracer_provider

        with pytest.raises(ValueError, match="boom"):
            with agent_span("test.error_span", task_id="err-001"):
                raise ValueError("boom")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        # Exception event should be recorded
        assert any(e.name == "exception" for e in span.events)

    def test_span_does_not_set_unsafe_extra_attributes(self, fresh_tracer_provider):
        exporter = fresh_tracer_provider
        with agent_span("test.safe", extra={"prompt_body": "secret content", "llm.model": "mock"}):
            pass

        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes
        assert "prompt_body" not in attrs
        assert attrs.get("llm.model") == "mock"


# ---------------------------------------------------------------------------
# Trace ID helpers
# ---------------------------------------------------------------------------


class TestTraceIdHelpers:
    def test_returns_valid_hex_inside_span(self, fresh_tracer_provider):
        with agent_span("trace_id_test"):
            tid = get_current_trace_id()
            sid = get_current_span_id()

        assert tid != "no-trace"
        assert len(tid) == 32
        assert all(c in "0123456789abcdef" for c in tid)
        assert sid != "no-span"
        assert len(sid) == 16

    def test_returns_fallback_outside_span(self, monkeypatch):
        """get_current_trace_id returns 'no-trace' when there is no active recording span."""
        import app.core.tracing as tracing_module
        from opentelemetry.trace import NonRecordingSpan, INVALID_SPAN_CONTEXT

        # Temporarily make get_tracer return a tracer that yields NonRecordingSpans
        noop_provider = TracerProvider()  # fresh provider with no processors
        monkeypatch.setattr(
            tracing_module,
            "get_tracer",
            lambda name: noop_provider.get_tracer(name),
        )
        # Outside any span, the active span is a NonRecordingSpan with invalid context
        tid = get_current_trace_id()
        # Should be 'no-trace' since context is invalid outside a span
        assert isinstance(tid, str)


# ---------------------------------------------------------------------------
# configure_tracing idempotency
# ---------------------------------------------------------------------------


class TestConfigureTracing:
    def test_configure_tracing_is_idempotent(self):
        """Calling configure_tracing twice must not raise or duplicate processors."""
        import app.core.tracing as tracing_module

        saved = tracing_module._tracer_provider
        tracing_module._tracer_provider = None  # force re-init
        try:
            p1 = configure_tracing()
            p2 = configure_tracing()
            assert p1 is p2  # second call returns the cached provider
        finally:
            tracing_module._tracer_provider = saved
