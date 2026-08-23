from typing import Any

from app.services.llm.base import BaseLLMProvider, LLMError, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for local testing and offline execution."""

    def __init__(
        self,
        default_response: str | None = None,
        custom_responses: dict[str, str] | None = None,
        model_name: str = "mock-model-v1",
        should_fail: bool = False,
        failure_message: str = "Simulated mock LLM failure",
    ) -> None:
        self._default_response = default_response
        self._custom_responses = custom_responses or {}
        self._model_name = model_name
        self._should_fail = should_fail
        self._failure_message = failure_message
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._should_fail:
            raise LLMError(self._failure_message)

        # Check custom prompt responses
        matched_custom = next(
            (resp for key, resp in self._custom_responses.items() if key.lower() in prompt.lower()),
            None,
        )

        if matched_custom is not None:
            content = matched_custom
        elif self._default_response is not None:
            content = self._default_response
        elif system_prompt and "tool_decider" in system_prompt.lower():
            lower_prompt = prompt.lower()
            math_keys = [
                "calculate", "calc", "*", "+", "/", "math", "sqrt", "25 * 4",
                "multiplied", "divided", "power", "compute", "minus", "plus", "times",
            ]
            info_keys = [
                "app_info", "app info", "metadata", "version", "system info",
                "what application", "capabilities",
            ]
            if any(k in lower_prompt for k in math_keys):
                expr = "25 * 4"
                if "expression:" in lower_prompt:
                    expr = prompt.split("expression:", 1)[1].strip().split("\n")[0]
                elif "25 multiplied by 48" in lower_prompt:
                    expr = "25 * 48"
                elif "1024 divided by 8" in lower_prompt:
                    expr = "1024 / 8"
                elif "2 to the power of 10" in lower_prompt:
                    expr = "2 ** 10"
                elif "(100 + 50) * 3 - 75" in lower_prompt:
                    expr = "(100 + 50) * 3 - 75"
                content = f'{{"tool_name": "calculator", "tool_args": {{"expression": "{expr}"}}}}'
            elif any(k in lower_prompt for k in info_keys):
                content = '{"tool_name": "app_info", "tool_args": {}}'
            else:
                content = '{"tool_name": null, "tool_args": {}}'
        elif system_prompt and "planner" in system_prompt.lower():
            content = (
                "1. Analyze the core requirements of the task\n"
                "2. Formulate a structured step-by-step approach\n"
                "3. Synthesize and format the final answer"
            )
        elif "tool observation (`" in prompt.lower():
            # Extract output from tool observation block
            output_val = "completed successfully"
            for line in prompt.split("\n"):
                stripped = line.strip()
                if stripped.startswith("- Output:"):
                    output_val = stripped.split(":", 1)[1].strip()
                    break
            content = (
                f"The task was executed using the tool. "
                f"The calculation result is {output_val}. "
                f"Therefore, the final answer is {output_val}."
            )
        elif "no relevant context found" in prompt.lower():
            content = (
                "The provided knowledge base documents do not contain "
                "information to answer this question."
            )
        elif "[citation 1]" in prompt.lower():
            lower_prompt = prompt.lower()
            if "germany" in lower_prompt and "semiconductor" in lower_prompt:
                content = (
                    "### Moving from AI to the Semiconductor Industry in Germany\n\n"
                    "Germany is Europe's primary semiconductor powerhouse, driven by **Silicon Saxony** (Dresden), "
                    "Munich, and Stuttgart. Major industry leaders including **Infineon, Bosch, NXP, GlobalFoundries, "
                    "Siemens EDA, and TSMC (ESMC Dresden)** actively hire AI/ML engineers.\n\n"
                    "#### 1. High-Demand AI & Semiconductor Roles\n"
                    "- **AI for EDA & Chip Design**: Applying ML to physical layout, routing optimization, and timing closure (Synopsys, Cadence, Siemens EDA).\n"
                    "- **Edge AI & Neuromorphic Accelerators**: Designing low-power NPU/TPU architectures and DSPs for automotive and industrial IoT.\n"
                    "- **Wafer Fab Yield Optimization**: Computer vision and time-series anomaly detection for cleanroom semiconductor manufacturing.\n\n"
                    "#### 2. Transition Roadmap\n"
                    "1. **Learn Hardware Fundamentals**: Focus on computer architecture, Verilog/VHDL, and MLIR/TVM compiler frameworks.\n"
                    "2. **Target German Tech Hubs**: Dresden (Silicon Saxony), Munich (Infineon HQ), and Stuttgart (Automotive semiconductor R&D).\n"
                    "3. **Visa Pathways**: Leverage the German EU Blue Card or Opportunity Card (*Chancenkarte*) for STEM specialists."
                )
            else:
                # Extract content from all citation blocks to create a comprehensive grounded answer
                blocks = prompt.split("[Citation ")
                chunks: list[str] = []
                for b in blocks[1:]:
                    raw_lines = [line.strip() for line in b.split("\n")]
                    # Line 0 is the citation header (e.g. '1] (Source: doc, Chunk: 0, ...)'), content is in subsequent lines
                    content_lines = [
                        l for l in raw_lines[1:]
                        if l
                        and not l.startswith("---")
                        and not l.startswith("Question:")
                        and not l.startswith("Answer based")
                        and not l.startswith("Context Information")
                        and not l.startswith("(")
                    ]
                    if content_lines:
                        chunks.append(content_lines[0])
                joined_context = " ".join(chunks[:3]) if chunks else "the referenced knowledge base material."
                content = f"Based on [Citation 1], {joined_context}"
        else:
            # Deterministic mock response for testing that incorporates query context
            clean_query = prompt.strip()
            content = (
                f"Mock LLM response for query: {clean_query}. "
                f"The system processed the request and verified all components."
            )

        return LLMResponse(
            content=content,
            model=self._model_name,
            provider=self.provider_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            total_tokens=len(prompt.split()) + len(content.split()),
        )
