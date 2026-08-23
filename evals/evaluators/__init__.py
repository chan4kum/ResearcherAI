"""evals/evaluators/__init__.py"""
from evals.evaluators.agent_success import AgentSuccessEvaluator
from evals.evaluators.answer_quality import AnswerQualityEvaluator
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict
from evals.evaluators.citation_correctness import CitationCorrectnessEvaluator
from evals.evaluators.groundedness import GroundednessEvaluator
from evals.evaluators.retrieval_relevance import RetrievalRelevanceEvaluator
from evals.evaluators.tool_selection import ToolSelectionEvaluator

ALL_EVALUATORS: list[BaseEvaluator] = [
    RetrievalRelevanceEvaluator(),
    CitationCorrectnessEvaluator(),
    GroundednessEvaluator(),
    AnswerQualityEvaluator(),
    AgentSuccessEvaluator(),
    ToolSelectionEvaluator(),
]

__all__ = [
    "ALL_EVALUATORS",
    "BaseEvaluator",
    "EvalResult",
    "EvalVerdict",
    "RetrievalRelevanceEvaluator",
    "CitationCorrectnessEvaluator",
    "GroundednessEvaluator",
    "AnswerQualityEvaluator",
    "AgentSuccessEvaluator",
    "ToolSelectionEvaluator",
]
