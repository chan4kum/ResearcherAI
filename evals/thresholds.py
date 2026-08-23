"""
evals/thresholds.py — Regression threshold definitions and configuration

Defines quality thresholds for continuous CI evaluation:
- Absolute minimum passing scores (non-perfect to account for model variance)
- Maximum allowed regression degradation against golden baseline
- Path triggers for targeted evaluation on code changes
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityThresholds:
    """Quality thresholds enforced in CI pipeline."""

    min_overall_score: float = 0.85
    """Minimum acceptable overall suite score (0.0 – 1.0). Default: 85%."""

    min_dimension_scores: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval_relevance": 0.80,
            "citation_correctness": 0.85,
            "groundedness": 0.80,
            "answer_quality": 0.75,
            "agent_success": 0.85,
            "tool_selection": 0.80,
        }
    )
    """Minimum score required for each individual evaluation dimension."""

    max_allowed_drop: float = 0.05
    """Maximum allowed overall score drop vs golden baseline. Default: 5% (0.05)."""

    max_dimension_drop: float = 0.10
    """Maximum allowed score drop for any individual dimension vs baseline. Default: 10% (0.10)."""


# Path triggers: File patterns that must trigger LLM evaluation in CI
EVAL_TRIGGER_PATTERNS: list[str] = [
    # Prompts & system instructions
    "app/services/agent/prompts/**",
    "app/services/rag/service.py",
    # Models & LLM provider implementations
    "app/services/llm/**",
    # Retrievers & vector search algorithms
    "app/services/rag/retriever.py",
    "app/services/rag/bm25.py",
    "app/services/rag/fusion.py",
    # Agent logic & LangGraph execution nodes
    "app/services/agent/**",
    # Routing & adaptive retrieval
    "app/services/rag/router.py",
    "app/services/rag/routing.py",
    "app/services/rag/adaptive.py",
    # HyDE (Hypothetical Document Embeddings)
    "app/services/rag/hyde.py",
    # Query rewriting & analysis
    "app/services/rag/rewriter.py",
    "app/services/rag/analyzer.py",
    "app/services/rag/query_analysis.py",
    # Evaluators and evaluation dataset
    "evals/**",
]


DEFAULT_THRESHOLDS = QualityThresholds()
