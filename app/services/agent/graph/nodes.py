import json
import re
from typing import Any

from app.core.guardrails.secrets_filter import SecretsScrubber
from app.core.guardrails.tool_governance import sanitize_tool_argument
from app.core.logging import get_logger
from app.core.tracing import agent_span
from app.core.versioning.prompts import PromptRegistry, get_prompt_registry
from app.models.schemas import TaskStatus
from app.services.agent.graph.state import AgentGraphState
from app.services.agent.tools.base import ToolCall
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService

logger = get_logger("app.services.agent.graph.nodes")


def parse_plan(plan_text: str, fallback_task: str) -> list[str]:
    """Extract a clean, structured list of steps from raw planning text."""
    steps: list[str] = []
    for line in plan_text.strip().splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Strip numbering (e.g., '1.', '1)', 'Step 1:') or bullet points ('-', '*')
        stripped = re.sub(
            r"^(\d+[\.\)]|step\s*\d+:|[-*•])\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            steps.append(stripped)

    if not steps:
        steps = [f"Directly analyze and resolve task: {fallback_task[:120]}"]
    return steps


def _extract_json_block(text: str) -> str:
    """Extract clean JSON substring from LLM response text."""
    # Find markdown code block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Otherwise find first outer brace pair
    match_braces = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match_braces:
        return match_braces.group(1).strip()
    return text.strip()


def create_planner_node(
    llm_service: LLMService,
    prompt_registry: PromptRegistry | None = None,
) -> Any:
    """Factory returning an asynchronous LangGraph node function for the Planner stage."""
    registry = prompt_registry or get_prompt_registry()

    async def planner_node(state: AgentGraphState) -> dict[str, Any]:
        task = state.get("task", "")
        system_instructions = state.get("system_instructions")
        task_id = state.get("task_id", "")
        trace = list(state.get("trace", []))
        trace.append("planning")

        logger.info("planner_node_start", task_id=task_id, task_len=len(task))

        # Retrieve versioned prompt template
        prompt_version = registry.get("planner")
        system_prompt = prompt_version.template
        if system_instructions:
            system_prompt = f"{system_prompt}\nGuidance: {system_instructions}"

        with agent_span("agent.planner", task_id=task_id) as span:
            response = await llm_service.chat(
                message=f"Task to plan:\n{task}",
                system_prompt=system_prompt,
                temperature=0.2,
            )

            plan_steps = parse_plan(response.content, task)
            trace.append("planned")

            span.set_attribute("plan.steps_count", len(plan_steps))
            span.set_attribute("llm.model", response.model or "unknown")
            span.set_attribute("llm.prompt_tokens", response.prompt_tokens or 0)
            span.set_attribute("llm.completion_tokens", response.completion_tokens or 0)

        logger.info("planner_node_completed", task_id=task_id, plan_steps=len(plan_steps))

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        total_tokens = response.total_tokens or (prompt_tokens + completion_tokens)

        return {
            "status": TaskStatus.PLANNING.value,
            "plan": plan_steps,
            "model": response.model,
            "provider": response.provider,
            "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": state.get("total_tokens", 0) + total_tokens,
            "trace": trace,
        }

    return planner_node


def create_tool_decision_node(
    llm_service: LLMService,
    tool_registry: ToolRegistry,
    prompt_registry: PromptRegistry | None = None,
) -> Any:
    """Factory returning node function that decides if a tool should be invoked."""
    registry = prompt_registry or get_prompt_registry()

    async def tool_decision_node(state: AgentGraphState) -> dict[str, Any]:
        task = state.get("task", "")
        plan = state.get("plan", [])
        task_id = state.get("task_id", "")
        trace = list(state.get("trace", []))
        trace.append("tool_decision")

        tools_desc = tool_registry.format_tools_description()
        prompt_version = registry.get("tool_decider")
        system_prompt = prompt_version.template.format(tools_desc=tools_desc)

        formatted_plan = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        user_prompt = (
            f"User Task: {task}\n\n"
            f"Plan:\n{formatted_plan}\n\n"
            "Decide if a tool is required. Return JSON decision only."
        )

        with agent_span("agent.tool_decision", task_id=task_id) as span:
            response = await llm_service.chat(
                message=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            tool_call_dict: dict[str, Any] | None = None
            try:
                cleaned_json = _extract_json_block(response.content)
                parsed = json.loads(cleaned_json)
                tool_name = parsed.get("tool_name")
                if tool_name and str(tool_name).lower() not in ("none", "null"):
                    tool_args = parsed.get("tool_args", {})
                    tool_call = ToolCall(tool_name=str(tool_name), tool_args=tool_args)
                    tool_call_dict = tool_call.model_dump()
                    span.set_attribute("tool.name", str(tool_name))
                    span.set_attribute("tool.decision", "selected")
                else:
                    span.set_attribute("tool.decision", "no_tool_needed")
            except Exception as err:
                logger.warning(
                    "tool_decision_parse_failed",
                    task_id=task_id,
                    error=str(err),
                )
                span.set_attribute("tool.decision", "parse_failed")
                tool_call_dict = None

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        total_tokens = response.total_tokens or (prompt_tokens + completion_tokens)

        logger.info(
            "tool_decision_completed",
            task_id=task_id,
            tool_call=tool_call_dict,
        )

        return {
            "tool_call": tool_call_dict,
            "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": state.get("total_tokens", 0) + total_tokens,
            "trace": trace,
        }

    return tool_decision_node


def create_tool_executor_node(tool_registry: ToolRegistry) -> Any:
    """Factory returning node function that executes the decided tool."""

    async def tool_node(state: AgentGraphState) -> dict[str, Any]:
        task_id = state.get("task_id", "")
        tool_call = state.get("tool_call") or {}
        tool_name = tool_call.get("tool_name", "")
        tool_args = sanitize_tool_argument(tool_call.get("tool_args", {}))
        trace = list(state.get("trace", []))
        tools_used = list(state.get("tools_used", []))

        trace.append("tool_executing")
        logger.info("tool_node_executing", task_id=task_id, tool_name=tool_name)

        with agent_span("agent.tool_execution", task_id=task_id, tool_name=tool_name) as span:
            result = tool_registry.execute(tool_name=tool_name, **tool_args)
            # Scrub secrets from observation output before feeding back to LLM
            if result.output is not None:
                result.output = SecretsScrubber.scrub_data(result.output)
            if result.error is not None:
                result.error = SecretsScrubber.scrub_text(result.error)

            span.set_attribute("tool.success", result.success)
            if not result.success and result.error:
                span.set_attribute("tool.error", result.error[:200])

        trace.append("tool_executed")

        if tool_name and tool_name not in tools_used:
            tools_used.append(tool_name)

        logger.info(
            "tool_node_completed",
            task_id=task_id,
            tool_name=tool_name,
            success=result.success,
        )

        return {
            "tool_result": result.model_dump(),
            "tools_used": tools_used,
            "trace": trace,
        }

    return tool_node


def create_answer_node(
    llm_service: LLMService,
    prompt_registry: PromptRegistry | None = None,
) -> Any:
    """Factory returning an asynchronous LangGraph node function for the Answer Agent stage."""
    registry = prompt_registry or get_prompt_registry()

    async def answer_node(state: AgentGraphState) -> dict[str, Any]:
        task = state.get("task", "")
        plan = state.get("plan", [])
        tool_result = state.get("tool_result")
        system_instructions = state.get("system_instructions")
        task_id = state.get("task_id", "")
        trace = list(state.get("trace", []))
        trace.append("executing")

        logger.info(
            "answer_node_start",
            task_id=task_id,
            plan_steps=len(plan),
            has_tool_result=bool(tool_result),
        )

        formatted_plan = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        prompt_version = registry.get("executor")
        system_prompt = prompt_version.template
        if system_instructions:
            system_prompt = f"{system_prompt}\nSpecific constraints: {system_instructions}"

        prompt_parts = [
            f"User Task:\n{task}",
            f"Execution Plan:\n{formatted_plan}",
        ]

        if tool_result:
            tool_name = tool_result.get("tool_name", "unknown")
            success = tool_result.get("success", False)
            output = tool_result.get("output")
            error = tool_result.get("error")
            prompt_parts.append(
                f"Tool Observation (`{tool_name}`):\n"
                f"- Status: {'SUCCESS' if success else 'FAILED'}\n"
                f"- Output: {output}\n"
                f"- Error: {error}"
            )

        prompt_parts.append("Please generate the complete, high-quality solution.")
        prompt = "\n\n".join(prompt_parts)

        with agent_span(
            "agent.answer_synthesis",
            task_id=task_id,
            model=None,
            extra={"plan.steps_count": len(plan), "has_tool_result": bool(tool_result)},
        ) as span:
            response = await llm_service.chat(
                message=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
            )
            span.set_attribute("llm.model", response.model or "unknown")
            span.set_attribute("llm.provider", response.provider or "unknown")
            span.set_attribute("llm.prompt_tokens", response.prompt_tokens or 0)
            span.set_attribute("llm.completion_tokens", response.completion_tokens or 0)
            span.set_attribute("llm.total_tokens", response.total_tokens or 0)

        trace.append("executed")
        trace.append("completed")

        logger.info("answer_node_completed", task_id=task_id)

        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        total_tokens = response.total_tokens or (prompt_tokens + completion_tokens)

        return {
            "status": TaskStatus.COMPLETED.value,
            "answer": response.content,
            "model": response.model,
            "provider": response.provider,
            "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + completion_tokens,
            "total_tokens": state.get("total_tokens", 0) + total_tokens,
            "trace": trace,
        }

    return answer_node
