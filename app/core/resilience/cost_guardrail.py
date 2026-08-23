import threading
from typing import Any

from app.core.errors import BudgetExceededException
from app.core.logging import get_logger

logger = get_logger("app.core.resilience.cost_guardrail")


class CostBudgetTracker:
    """Session/task budget tracker enforcing caps on tool executions and research iterations."""

    def __init__(
        self,
        max_tool_calls: int = 15,
        max_research_iterations: int = 5,
        max_llm_invocations: int = 25,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_research_iterations = max_research_iterations
        self.max_llm_invocations = max_llm_invocations

        self._tool_calls = 0
        self._research_iterations = 0
        self._llm_invocations = 0
        self._lock = threading.Lock()

    @property
    def tool_calls(self) -> int:
        """Cumulative tool execution count."""
        with self._lock:
            return self._tool_calls

    @property
    def research_iterations(self) -> int:
        """Cumulative research iteration count."""
        with self._lock:
            return self._research_iterations

    def record_tool_call(self, tool_name: str) -> None:
        """Record and validate tool invocation against budget caps."""
        with self._lock:
            if self._tool_calls >= self.max_tool_calls:
                logger.error(
                    "cost_guardrail_tool_budget_exceeded",
                    tool=tool_name,
                    count=self._tool_calls,
                    max_allowed=self.max_tool_calls,
                )
                raise BudgetExceededException(
                    message=(
                        f"Execution halted: Maximum tool call budget of {self.max_tool_calls} "
                        f"exceeded (attempted '{tool_name}')."
                    ),
                    details={
                        "max_tool_calls": self.max_tool_calls,
                        "current_count": self._tool_calls,
                        "tool": tool_name,
                    },
                )
            self._tool_calls += 1

    def record_research_iteration(self, iteration_index: int) -> None:
        """Record and validate research iteration against budget caps."""
        with self._lock:
            if iteration_index > self.max_research_iterations:
                logger.error(
                    "cost_guardrail_research_iteration_exceeded",
                    iteration=iteration_index,
                    max_allowed=self.max_research_iterations,
                )
                raise BudgetExceededException(
                    message=(
                        f"Execution halted: Maximum research iteration limit of "
                        f"{self.max_research_iterations} reached."
                    ),
                    details={
                        "max_research_iterations": self.max_research_iterations,
                        "attempted_iteration": iteration_index,
                    },
                )
            self._research_iterations = max(self._research_iterations, iteration_index)

    def record_llm_invocation(self) -> None:
        """Record and validate LLM invocation against budget caps."""
        with self._lock:
            if self._llm_invocations >= self.max_llm_invocations:
                logger.error(
                    "cost_guardrail_llm_invocation_exceeded",
                    count=self._llm_invocations,
                    max_allowed=self.max_llm_invocations,
                )
                raise BudgetExceededException(
                    message=(
                        f"Execution halted: Maximum LLM invocation limit of "
                        f"{self.max_llm_invocations} reached."
                    ),
                    details={
                        "max_llm_invocations": self.max_llm_invocations,
                        "current_count": self._llm_invocations,
                    },
                )
            self._llm_invocations += 1

    def get_summary(self) -> dict[str, Any]:
        """Return diagnostic metrics of resource consumption."""
        with self._lock:
            return {
                "tool_calls": f"{self._tool_calls}/{self.max_tool_calls}",
                "research_iterations": (
                    f"{self._research_iterations}/{self.max_research_iterations}"
                ),
                "llm_invocations": f"{self._llm_invocations}/{self.max_llm_invocations}",
            }
