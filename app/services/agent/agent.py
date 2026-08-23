import time
import uuid
from typing import Any

from app.core.guardrails.injection import PromptInjectionGuard
from app.core.guardrails.secrets_filter import SecretsScrubber
from app.core.logging import get_logger, log_agent_stage
from app.core.metrics import (
    AGENT_EXECUTION_DURATION_SECONDS,
    AGENT_EXECUTIONS_TOTAL,
    AGENT_TOOL_CALLS_TOTAL,
)
from app.core.tracing import agent_span, get_current_trace_id
from app.core.versioning.manager import (
    ConfigurationVersionManager,
    get_version_manager,
)
from app.models.schemas import TaskStatus
from app.services.agent.graph.nodes import parse_plan
from app.services.agent.graph.state import AgentGraphState
from app.services.agent.graph.workflow import build_agent_graph
from app.services.agent.state import AgentState
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService

logger = get_logger("app.services.agent.basic")


class BasicAgent:
    """Agent orchestrator powered by a compiled LangGraph state workflow with tool routing and version provenance."""

    def __init__(
        self,
        llm_service: LLMService,
        tool_registry: ToolRegistry | None = None,
        version_manager: ConfigurationVersionManager | None = None,
    ) -> None:
        self._llm = llm_service
        self._tool_registry = tool_registry or ToolRegistry()
        self._version_manager = version_manager or get_version_manager()
        self._graph = build_agent_graph(
            llm_service,
            tool_registry=self._tool_registry,
            prompt_registry=self._version_manager.prompt_registry,
        )

    @property
    def graph(self) -> Any:
        """Expose compiled LangGraph workflow instance for inspection."""
        return self._graph

    @property
    def tool_registry(self) -> ToolRegistry:
        """Expose the active tool registry."""
        return self._tool_registry

    @property
    def version_manager(self) -> ConfigurationVersionManager:
        """Expose the configuration version manager."""
        return self._version_manager

    def _parse_plan(self, plan_text: str, fallback_task: str) -> list[str]:
        """Backward-compatible helper delegating to plan parser."""
        return parse_plan(plan_text, fallback_task)

    async def run(
        self,
        task: str,
        task_id: str | None = None,
        system_instructions: str | None = None,
    ) -> AgentState:
        """Execute the LangGraph workflow for the given task."""
        resolved_task_id = task_id or str(uuid.uuid4())
        initial_state: AgentGraphState = {
            "task_id": resolved_task_id,
            "task": task,
            "system_instructions": system_instructions,
            "status": TaskStatus.PENDING.value,
            "plan": [],
            "tool_call": None,
            "tool_result": None,
            "tools_used": [],
            "answer": None,
            "error": None,
            "trace": ["initialized"],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        start_time = time.perf_counter()
        registered_tools = [t.name for t in self._tool_registry.list_tools()]
        provenance = self._version_manager.create_provenance(tools_registered=registered_tools)

        # Milestone 51: Prompt Injection Threat Scan
        injection_scan = PromptInjectionGuard.scan_text(task)
        if injection_scan.is_threat:
            logger.warning(
                "agent_prompt_injection_detected",
                request_id=resolved_task_id,
                threat_type=injection_scan.threat_type.value if injection_scan.threat_type else "unknown",
                reason=injection_scan.reason,
                pattern=injection_scan.matched_pattern,
            )

        log_agent_stage(
            logger,
            "agent_task_started",
            request_id=resolved_task_id,
            agent_stage="initialized",
            task_summary=task[:80],
            config_version=provenance.agent_version,
            composite_config_hash=provenance.composite_hash,
            security_threat=injection_scan.is_threat,
        )

        with agent_span("agent.task", task_id=resolved_task_id) as root_span:
            root_span.set_attribute("agent.task_length", len(task))
            root_span.set_attribute("config.version", provenance.agent_version)
            root_span.set_attribute("config.composite_hash", provenance.composite_hash)
            root_span.set_attribute("config.model_version", provenance.model_config_version)
            root_span.set_attribute("config.retrieval_version", provenance.retrieval_config_version)
            root_span.set_attribute("config.routing_version", provenance.routing_config_version)
            if injection_scan.is_threat:
                root_span.set_attribute("security.threat_detected", True)
                root_span.set_attribute(
                    "security.threat_type",
                    injection_scan.threat_type.value if injection_scan.threat_type else "unknown",
                )

            try:
                # Execute LangGraph workflow (START -> planner -> tool_decision -> answer -> END)
                final_graph_state: dict[str, Any] = await self._graph.ainvoke(initial_state)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                # Milestone 51: Scrub any potential credentials/secrets from answer
                raw_answer = final_graph_state.get("answer")
                clean_answer = SecretsScrubber.scrub_text(raw_answer) if raw_answer else None

                state = AgentState(
                    task_id=resolved_task_id,
                    task=task,
                    status=TaskStatus(final_graph_state.get("status", TaskStatus.COMPLETED.value)),
                    plan=final_graph_state.get("plan", []),
                    answer=clean_answer,
                    error=final_graph_state.get("error"),
                    model=final_graph_state.get("model", ""),
                    provider=final_graph_state.get("provider", ""),
                    prompt_tokens=final_graph_state.get("prompt_tokens", 0),
                    completion_tokens=final_graph_state.get("completion_tokens", 0),
                    total_tokens=final_graph_state.get("total_tokens", 0),
                    tools_used=final_graph_state.get("tools_used", []),
                    trace=final_graph_state.get("trace", []),
                    duration_ms=duration_ms,
                    config_version=provenance.agent_version,
                    composite_config_hash=provenance.composite_hash,
                    prompt_versions=provenance.prompt_versions,
                    model_config_version=provenance.model_config_version,
                    retrieval_config_version=provenance.retrieval_config_version,
                    routing_config_version=provenance.routing_config_version,
                )

                root_span.set_attribute("agent.model", state.model or "unknown")
                root_span.set_attribute("agent.provider", state.provider or "unknown")
                root_span.set_attribute("agent.total_tokens", state.total_tokens or 0)
                root_span.set_attribute("agent.tools_used", str(state.tools_used))

                duration_sec = time.perf_counter() - start_time
                AGENT_EXECUTIONS_TOTAL.labels(
                    status="completed",
                    model=state.model or "unknown",
                    provider=state.provider or "unknown",
                ).inc()
                AGENT_EXECUTION_DURATION_SECONDS.labels(status="completed").observe(duration_sec)

                for tool in state.tools_used:
                    AGENT_TOOL_CALLS_TOTAL.labels(tool_name=tool, status="success").inc()

                log_agent_stage(
                    logger,
                    "agent_task_completed",
                    request_id=resolved_task_id,
                    agent_stage="completed",
                    duration_ms=duration_ms,
                    tool_calls=state.tools_used,
                    model=state.model,
                    provider=state.provider,
                    prompt_tokens=state.prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=state.total_tokens,
                    plan_steps_count=len(state.plan),
                    trace_id=get_current_trace_id(),
                    config_version=state.config_version,
                    composite_config_hash=state.composite_config_hash,
                )
                return state

            except Exception as exc:
                duration_sec = time.perf_counter() - start_time
                duration_ms = round(duration_sec * 1000, 2)
                failed_trace = list(initial_state.get("trace", ["initialized"]))
                failed_trace.append("failed")

                clean_error = SecretsScrubber.scrub_text(str(exc))
                root_span.set_attribute("agent.error", clean_error[:200])

                AGENT_EXECUTIONS_TOTAL.labels(
                    status="failed",
                    model="unknown",
                    provider="unknown",
                ).inc()
                AGENT_EXECUTION_DURATION_SECONDS.labels(status="failed").observe(duration_sec)

                state = AgentState(
                    task_id=resolved_task_id,
                    task=task,
                    status=TaskStatus.FAILED,
                    error=clean_error,
                    trace=failed_trace,
                    duration_ms=duration_ms,
                    config_version=provenance.agent_version,
                    composite_config_hash=provenance.composite_hash,
                    prompt_versions=provenance.prompt_versions,
                    model_config_version=provenance.model_config_version,
                    retrieval_config_version=provenance.retrieval_config_version,
                    routing_config_version=provenance.routing_config_version,
                )

                log_agent_stage(
                    logger,
                    "agent_task_failed",
                    request_id=resolved_task_id,
                    agent_stage="failed",
                    error=clean_error,
                    duration_ms=duration_ms,
                    tool_calls=state.tools_used,
                    trace_id=get_current_trace_id(),
                    config_version=state.config_version,
                    composite_config_hash=state.composite_config_hash,
                )
                return state



