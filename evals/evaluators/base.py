"""
evals/evaluators/base.py — Abstract base evaluator interface

Every evaluator scores ONE dimension of system quality and returns a
structured EvalResult with a score, pass/fail, and a diagnostic reason.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvalVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"   # Not applicable for this case type


@dataclass
class EvalResult:
    """Structured result from one evaluator on one test case."""

    dimension: str
    """Which evaluation dimension this result covers."""

    verdict: EvalVerdict
    """Pass / Fail / Skip."""

    score: float
    """Normalized score 0.0 – 1.0. 1.0 = perfect."""

    reason: str
    """Human-readable explanation of the verdict."""

    details: dict[str, Any] = field(default_factory=dict)
    """Extra diagnostic data (not surfaced in summary)."""


class BaseEvaluator(ABC):
    """Abstract interface for a single-dimension evaluator."""

    @property
    @abstractmethod
    def dimension(self) -> str:
        """Name of the dimension this evaluator measures."""

    @abstractmethod
    def evaluate(self, case: Any, response: Any) -> EvalResult:
        """Score the response against the case expectations."""
