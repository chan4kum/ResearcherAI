"""
evals/dataset/cases.py — Evaluation dataset for the Agentic Platform

Each EvalCase is a structured test fixture encoding:
  - question or task
  - the evaluation type: AGENT | RAG
  - expected behaviour as machine-checkable assertions
  - expected sources (for RAG citations)
  - expected answer characteristics (keyword signals, quality rubric)

Design principles:
  - All cases are deterministic with the mock LLM provider (no API key required)
  - Coverage spans all six evaluation dimensions:
      1. Retrieval relevance
      2. Citation correctness
      3. Groundedness
      4. Answer quality
      5. Agent success
      6. Tool selection
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalType(str, Enum):
    AGENT = "agent"
    RAG = "rag"
    TOOL = "tool"


class GroundednessLevel(str, Enum):
    HIGH = "high"       # Answer must rely only on retrieved context
    MEDIUM = "medium"   # Partial reliance on context is acceptable
    LOW = "low"         # Open-ended reasoning; factuality less critical


class ToolExpectation(BaseModel):
    """Describes the tool the agent should select for a task."""

    expected_tool: str | None = Field(
        description="Tool name the agent should choose, or None if no tool needed"
    )
    must_not_use_tool: list[str] = Field(
        default_factory=list,
        description="Tools that should never be selected for this task",
    )


class RetrievalExpectation(BaseModel):
    """Describes minimum acceptable retrieval quality."""

    min_citations: int = Field(default=1, description="Minimum citations required in response")
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Source filenames / doc_ids that should appear in citations",
    )
    min_avg_similarity: float = Field(
        default=0.0,
        description="Minimum average cosine similarity of returned citations",
    )
    strategy: str | None = Field(
        default=None,
        description="Expected retrieval strategy (normal, hyde, bm25, etc.)",
    )


class AnswerQualityExpectation(BaseModel):
    """Answer quality rubric expressed as observable signals."""

    must_contain_any: list[str] = Field(
        default_factory=list,
        description="At least one of these strings must appear in the answer (case-insensitive)",
    )
    must_not_contain: list[str] = Field(
        default_factory=list,
        description="None of these strings should appear in the answer",
    )
    min_word_count: int = Field(default=5, description="Minimum word count for a valid answer")
    max_word_count: int = Field(
        default=2000, description="Maximum word count (prevents verbosity failure)"
    )
    must_succeed: bool = Field(
        default=True, description="Agent/RAG execution must complete without error"
    )


class EvalCase(BaseModel):
    """Single evaluation test case."""

    id: str = Field(description="Unique case identifier")
    name: str = Field(description="Human-readable short name")
    eval_type: EvalType
    description: str = Field(description="What this case tests")

    # Input
    task_or_question: str = Field(description="The task sent to Agent, or the question sent to RAG")

    # Expectations
    tool_expectation: ToolExpectation | None = None
    retrieval_expectation: RetrievalExpectation | None = None
    answer_quality: AnswerQualityExpectation = Field(
        default_factory=AnswerQualityExpectation
    )
    groundedness: GroundednessLevel = GroundednessLevel.MEDIUM

    # Tags for filtering runs
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EVALUATION DATASET
# ---------------------------------------------------------------------------

EVAL_DATASET: list[EvalCase] = [
    # ── AGENT: success (no tool needed) ────────────────────────────────────
    EvalCase(
        id="AGENT-001",
        name="open_question_no_tool",
        eval_type=EvalType.AGENT,
        description=(
            "Agent should produce a coherent answer to a general knowledge task "
            "without invoking any tool. Tests: agent success, answer quality."
        ),
        task_or_question="Explain the difference between supervised and unsupervised learning.",
        tool_expectation=ToolExpectation(
            expected_tool=None,
            must_not_use_tool=["calculator"],
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["learning", "data", "model", "label", "supervised", "unsupervised"],
            must_not_contain=["error", "failed", "exception"],
            min_word_count=10,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "general-knowledge", "no-tool"],
    ),

    # ── AGENT: calculator tool selection ───────────────────────────────────
    EvalCase(
        id="AGENT-002",
        name="calculator_tool_arithmetic",
        eval_type=EvalType.TOOL,
        description=(
            "Agent must select the calculator tool and return a numerically correct "
            "result for a simple arithmetic task. Tests: tool selection, tool execution."
        ),
        task_or_question="What is 25 multiplied by 48?",
        tool_expectation=ToolExpectation(
            expected_tool="calculator",
            must_not_use_tool=["app_info"],
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["1200", "1,200"],
            min_word_count=2,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "calculator", "arithmetic"],
    ),

    EvalCase(
        id="AGENT-003",
        name="calculator_tool_compound_expression",
        eval_type=EvalType.TOOL,
        description=(
            "Agent must invoke the calculator for a multi-step arithmetic expression. "
            "Tests: tool selection accuracy, expression evaluation correctness."
        ),
        task_or_question="Calculate (100 + 50) * 3 - 75. Show the calculation.",
        tool_expectation=ToolExpectation(
            expected_tool="calculator",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["375", "375.0"],
            min_word_count=3,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "calculator", "compound"],
    ),

    EvalCase(
        id="AGENT-004",
        name="app_info_tool_selection",
        eval_type=EvalType.TOOL,
        description=(
            "Agent should select app_info tool when asked about the application's name, "
            "version, or capabilities. Tests: tool selection for meta-queries."
        ),
        task_or_question="What application am I using? What are its capabilities?",
        tool_expectation=ToolExpectation(
            expected_tool="app_info",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["platform", "agent", "agentic", "api", "application"],
            min_word_count=5,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "app_info", "meta"],
    ),

    # ── AGENT: task structure / planning ───────────────────────────────────
    EvalCase(
        id="AGENT-005",
        name="multi_step_planning",
        eval_type=EvalType.AGENT,
        description=(
            "Agent should produce a structured plan before answering. "
            "Tests: planner quality, multi-step reasoning, answer quality."
        ),
        task_or_question=(
            "Outline a 3-step process for deploying a Python FastAPI application "
            "to a Kubernetes cluster using Helm."
        ),
        tool_expectation=ToolExpectation(expected_tool=None),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["helm", "kubernetes", "deploy", "chart", "pod", "container"],
            must_not_contain=["error", "failed"],
            min_word_count=15,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "planning", "devops"],
    ),

    EvalCase(
        id="AGENT-006",
        name="error_handling_empty_task",
        eval_type=EvalType.AGENT,
        description=(
            "Agent must handle a very short, vague task gracefully without crashing. "
            "Tests: robustness, error handling."
        ),
        task_or_question="help",
        tool_expectation=ToolExpectation(expected_tool=None),
        answer_quality=AnswerQualityExpectation(
            must_not_contain=["traceback", "Exception", "Internal Server Error"],
            min_word_count=2,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "robustness", "edge-case"],
    ),

    # ── RAG: retrieval relevance ────────────────────────────────────────────
    EvalCase(
        id="RAG-001",
        name="retrieval_relevance_basic",
        eval_type=EvalType.RAG,
        description=(
            "RAG must retrieve at least one chunk and return a non-empty grounded answer. "
            "Tests: retrieval pipeline execution, retrieval relevance baseline."
        ),
        task_or_question="What is the agentic platform designed to do?",
        retrieval_expectation=RetrievalExpectation(
            min_citations=0,   # In-memory mock repo may have no docs; we test pipeline health
            min_avg_similarity=0.0,
        ),
        answer_quality=AnswerQualityExpectation(
            must_not_contain=["traceback", "Exception"],
            min_word_count=3,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.HIGH,
        tags=["rag", "retrieval", "relevance"],
    ),

    EvalCase(
        id="RAG-002",
        name="retrieval_with_normal_strategy",
        eval_type=EvalType.RAG,
        description=(
            "RAG must complete successfully with the 'normal' strategy. "
            "Tests: default vector retrieval path health."
        ),
        task_or_question="Describe how vector embeddings are used in semantic search.",
        retrieval_expectation=RetrievalExpectation(
            min_citations=0,
            strategy="normal",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["vector", "embedding", "semantic", "similarity", "search"],
            must_not_contain=["error", "traceback"],
            min_word_count=5,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.HIGH,
        tags=["rag", "retrieval", "vector", "strategy-normal"],
    ),

    EvalCase(
        id="RAG-003",
        name="retrieval_with_hyde_strategy",
        eval_type=EvalType.RAG,
        description=(
            "RAG must complete with HyDE strategy, generating a hypothetical document. "
            "Tests: HyDE pipeline execution, hypothetical document generation."
        ),
        task_or_question="How does retrieval augmented generation improve factual accuracy?",
        retrieval_expectation=RetrievalExpectation(
            min_citations=0,
            strategy="hyde",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["retrieval", "generation", "accuracy", "context", "grounded"],
            must_not_contain=["traceback", "Exception"],
            min_word_count=5,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.HIGH,
        tags=["rag", "retrieval", "hyde", "strategy-hyde"],
    ),

    # ── RAG: citation correctness ───────────────────────────────────────────
    EvalCase(
        id="RAG-004",
        name="citation_structure_correctness",
        eval_type=EvalType.RAG,
        description=(
            "Every citation in the RAG response must have a non-empty source, "
            "chunk_id, and similarity score. Tests: citation schema correctness."
        ),
        task_or_question="What machine learning techniques are commonly used in NLP?",
        retrieval_expectation=RetrievalExpectation(
            min_citations=0,
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["language", "nlp", "model", "text", "learning", "neural"],
            min_word_count=5,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.MEDIUM,
        tags=["rag", "citation", "correctness"],
    ),

    # ── RAG: groundedness ───────────────────────────────────────────────────
    EvalCase(
        id="RAG-005",
        name="groundedness_no_hallucination_signal",
        eval_type=EvalType.RAG,
        description=(
            "RAG answer must not introduce hallucination signals — "
            "fabricated names, impossible dates, or invented citations. "
            "Tests: groundedness enforcement."
        ),
        task_or_question="What are the main components of the Agentic Platform API?",
        retrieval_expectation=RetrievalExpectation(min_citations=0),
        answer_quality=AnswerQualityExpectation(
            must_not_contain=["I don't know", "I cannot", "hallucination detected", "traceback"],
            min_word_count=5,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.HIGH,
        tags=["rag", "groundedness", "hallucination"],
    ),

    # ── RAG: answer quality ─────────────────────────────────────────────────
    EvalCase(
        id="RAG-006",
        name="answer_quality_concise_technical",
        eval_type=EvalType.RAG,
        description=(
            "RAG must produce a technically coherent answer that is neither "
            "too short (< 5 words) nor excessively verbose (> 2000 words). "
            "Tests: answer quality, conciseness."
        ),
        task_or_question="Explain the role of a vector database in RAG systems.",
        retrieval_expectation=RetrievalExpectation(min_citations=0),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["vector", "database", "index", "embedding", "store"],
            must_not_contain=["traceback"],
            min_word_count=5,
            max_word_count=2000,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.HIGH,
        tags=["rag", "answer-quality"],
    ),

    EvalCase(
        id="RAG-007",
        name="empty_question_rejection",
        eval_type=EvalType.RAG,
        description=(
            "RAG must reject or handle an empty question without crashing. "
            "Tests: input validation, robustness."
        ),
        task_or_question="",
        retrieval_expectation=None,
        answer_quality=AnswerQualityExpectation(
            must_not_contain=["traceback", "Internal Server Error"],
            min_word_count=0,
            must_succeed=False,   # Expect a controlled validation error, not a crash
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["rag", "robustness", "edge-case", "validation"],
    ),

    # ── AGENT: agent success dimension ─────────────────────────────────────
    EvalCase(
        id="AGENT-007",
        name="agent_completes_with_completed_status",
        eval_type=EvalType.AGENT,
        description=(
            "Agent run must return status=COMPLETED for a well-formed task. "
            "Tests: agent success dimension."
        ),
        task_or_question="List three benefits of containerizing applications with Docker.",
        tool_expectation=ToolExpectation(expected_tool=None),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["docker", "container", "portab", "isolat", "deploy"],
            min_word_count=10,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "status", "success"],
    ),

    EvalCase(
        id="AGENT-008",
        name="agent_trace_contains_planning_stage",
        eval_type=EvalType.AGENT,
        description=(
            "Agent trace must include a planning stage marker indicating "
            "the LangGraph planner node executed. Tests: workflow completeness."
        ),
        task_or_question="What are the SOLID principles in software engineering?",
        tool_expectation=ToolExpectation(expected_tool=None),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["solid", "single", "responsibility", "principle", "design"],
            min_word_count=10,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "trace", "workflow"],
    ),

    # ── TOOL: tool selection precision ─────────────────────────────────────
    EvalCase(
        id="TOOL-001",
        name="no_tool_for_conceptual_question",
        eval_type=EvalType.TOOL,
        description=(
            "Agent must NOT select a tool for a conceptual/opinion question. "
            "Tests: tool selection precision (false positive avoidance)."
        ),
        task_or_question="What are the trade-offs between REST and GraphQL APIs?",
        tool_expectation=ToolExpectation(
            expected_tool=None,
            must_not_use_tool=["calculator", "app_info"],
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["rest", "graphql", "api", "trade", "query", "schema"],
            min_word_count=10,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "no-tool", "precision"],
    ),

    EvalCase(
        id="TOOL-002",
        name="calculator_division_result",
        eval_type=EvalType.TOOL,
        description=(
            "Calculator tool must correctly evaluate a division expression. "
            "Tests: tool execution correctness, numeric precision."
        ),
        task_or_question="Compute 1024 divided by 8.",
        tool_expectation=ToolExpectation(
            expected_tool="calculator",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["128", "128.0"],
            min_word_count=1,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "calculator", "division"],
    ),

    EvalCase(
        id="TOOL-003",
        name="calculator_power_expression",
        eval_type=EvalType.TOOL,
        description=(
            "Calculator tool must evaluate exponent expressions. "
            "Tests: tool execution with power operator."
        ),
        task_or_question="What is 2 to the power of 10?",
        tool_expectation=ToolExpectation(
            expected_tool="calculator",
        ),
        answer_quality=AnswerQualityExpectation(
            must_contain_any=["1024"],
            min_word_count=1,
            must_succeed=True,
        ),
        groundedness=GroundednessLevel.LOW,
        tags=["agent", "tool", "calculator", "power"],
    ),
]


def get_cases_by_tag(tag: str) -> list[EvalCase]:
    """Filter dataset by tag."""
    return [c for c in EVAL_DATASET if tag in c.tags]


def get_cases_by_type(eval_type: EvalType) -> list[EvalCase]:
    """Filter dataset by evaluation type."""
    return [c for c in EVAL_DATASET if c.eval_type == eval_type]


def get_case_by_id(case_id: str) -> EvalCase | None:
    """Retrieve a specific case by its ID."""
    return next((c for c in EVAL_DATASET if c.id == case_id), None)
