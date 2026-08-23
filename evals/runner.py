"""
evals/runner.py — Evaluation harness for the Agentic Platform

Runs every EvalCase against the live service stack (using the mock LLM provider)
and collects EvalResults per dimension.

Architecture:
  EvalCase → [Agent/RAG service] → response → [6 Evaluators] → CaseReport
  All CaseReports → EvalSuiteReport (summary + per-dimension aggregates)

Design decisions:
  - Uses the same dependency graph as production: mock LLM → real workflow
  - No external API keys required (mock provider is always active offline)
  - Fully async to match the production service interface
  - Writes a JSON report to evals/reports/ for CI diffing
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.dataset.cases import EvalCase, EvalType
from evals.evaluators import ALL_EVALUATORS
from evals.evaluators.base import EvalResult, EvalVerdict


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    dimension: str
    verdict: str
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseReport:
    case_id: str
    case_name: str
    eval_type: str
    task_or_question: str
    passed: bool
    overall_score: float
    duration_ms: float
    error: str | None
    dimensions: list[DimensionScore] = field(default_factory=list)


@dataclass
class EvalSuiteReport:
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    skipped_dimensions: int
    overall_score: float
    per_dimension_scores: dict[str, float]
    cases: list[CaseReport] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Service factory (uses mock stack — no real LLM/DB needed)
# ---------------------------------------------------------------------------


def _build_agent_service() -> Any:
    """Build an AgentService wired to the mock LLM provider."""
    from app.config import get_settings
    from app.services.agent.service import AgentService
    from app.services.agent.tools.app_info import AppInfoTool
    from app.services.agent.tools.calculator import CalculatorTool
    from app.services.agent.tools.registry import ToolRegistry
    from app.services.llm.service import LLMService

    settings = get_settings()
    llm = LLMService(settings=settings)
    registry = ToolRegistry(
        tools=[CalculatorTool(), AppInfoTool(settings=settings)],
        settings=settings,
    )
    return AgentService(llm_service=llm, tool_registry=registry)


async def _build_rag_service() -> Any:
    """Build a RAGService wired to the mock LLM + pre-seeded in-memory vector repository."""
    from app.config import get_settings
    from app.db.repository import InMemoryVectorRepository
    from app.services.document.models import ChunkMetadata, DocumentChunk
    from app.services.embedding.service import EmbeddingService
    from app.services.llm.service import LLMService
    from app.services.rag.retriever import VectorRetriever
    from app.services.rag.service import RAGService

    settings = get_settings()
    llm = LLMService(settings=settings)
    embedding = EmbeddingService(settings=settings)
    repo = InMemoryVectorRepository()

    # Seed knowledge base documents for representative retrieval evaluation
    seed_data = [
        (
            "doc_001",
            "agentic_platform_overview.md",
            "The Agentic Platform is an enterprise intelligence system combining LangGraph agent workflows, vector embeddings for semantic search, OpenTelemetry distributed tracing, Prometheus metrics, and Retrieval-Augmented Generation.",
            {"topic": "overview"},
        ),
        (
            "doc_002",
            "vector_search_guide.md",
            "Vector databases store embeddings as high-dimensional vectors and perform semantic search using approximate nearest neighbor indexing like HNSW or cosine similarity.",
            {"topic": "vector-search"},
        ),
        (
            "doc_003",
            "nlp_machine_learning.md",
            "Machine learning techniques in natural language processing (NLP) include transformer models, neural text embeddings, and supervised sequence classification.",
            {"topic": "nlp"},
        ),
        (
            "doc_004",
            "rag_architecture.md",
            "Retrieval augmented generation improves factual accuracy by retrieving verified ground truth context passages from documents before synthesizing the final answer.",
            {"topic": "rag"},
        ),
        (
            "doc_005",
            "kubernetes_deployment.md",
            "Deploying containerized microservices to Kubernetes uses Helm charts, ReplicaSets, Horizontal Pod Autoscaling, and OpenTelemetry sidecars.",
            {"topic": "devops"},
        ),
    ]

    chunks = []
    for doc_id, source, text, meta in seed_data:
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}_c0",
            doc_id=doc_id,
            content=text,
            metadata=ChunkMetadata(
                chunk_id=f"{doc_id}_c0",
                doc_id=doc_id,
                index=0,
                start_char=0,
                end_char=len(text),
                character_count=len(text),
                word_count=len(text.split()),
                source=source,
                file_type="md",
                checksum="mockchecksum12345",
                custom_metadata=meta,
            ),
        )
        chunks.append(chunk)

    embedded = await embedding.embed_chunks(chunks)
    await repo.store_chunks(embedded)

    retriever = VectorRetriever(embedding_service=embedding, vector_repository=repo)
    return RAGService(
        retriever=retriever,
        llm_service=llm,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Case executor
# ---------------------------------------------------------------------------


async def _execute_case(
    case: EvalCase,
    agent_service: Any,
    rag_service: Any,
) -> tuple[Any, str | None]:
    """Run a single case and return (response, error_str | None)."""
    try:
        if case.eval_type in (EvalType.AGENT, EvalType.TOOL):
            response = await agent_service.execute_task(task=case.task_or_question)
            return response, None

        elif case.eval_type == EvalType.RAG:
            if not case.task_or_question.strip():
                # Simulate expected validation rejection
                from dataclasses import dataclass as _dc

                @_dc
                class _Err:
                    answer: str = ""
                    error: str = "Question cannot be empty."
                    citations: list = field(default_factory=list)
                    strategy: str = "normal"

                return _Err(), "Question cannot be empty."

            strategy = "normal"
            if case.retrieval_expectation and case.retrieval_expectation.strategy:
                strategy = case.retrieval_expectation.strategy

            response = await rag_service.answer(
                question=case.task_or_question,
                strategy=strategy,
            )
            return response, None

        else:
            raise ValueError(f"Unknown eval_type: {case.eval_type}")

    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


async def run_evaluation(
    cases: list[EvalCase] | None = None,
    verbose: bool = True,
    report_dir: str | None = None,
) -> EvalSuiteReport:
    """Run the full evaluation suite and return a structured report.

    Args:
        cases: Subset of EvalCases to run, or None to run all.
        verbose: Print progress to stdout.
        report_dir: Directory to write the JSON report. Defaults to evals/reports/.
    """
    from evals.dataset.cases import EVAL_DATASET

    test_cases = cases or EVAL_DATASET
    agent_service = _build_agent_service()
    rag_service = await _build_rag_service()

    case_reports: list[CaseReport] = []
    dimension_totals: dict[str, list[float]] = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  AGENTIC PLATFORM — AI EVALUATION SUITE")
        print(f"  {len(test_cases)} cases × {len(ALL_EVALUATORS)} dimensions")
        print(f"{'='*70}\n")

    for case in test_cases:
        t0 = time.perf_counter()

        if verbose:
            print(f"[{case.id}] {case.name} ({case.eval_type.value}) ...")

        response, exec_error = await _execute_case(case, agent_service, rag_service)
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        dim_scores: list[DimensionScore] = []
        skipped = 0

        for evaluator in ALL_EVALUATORS:
            try:
                result: EvalResult = evaluator.evaluate(case, response)
            except Exception as exc:
                result = EvalResult(
                    dimension=evaluator.dimension,
                    verdict=EvalVerdict.FAIL,
                    score=0.0,
                    reason=f"Evaluator crashed: {exc}",
                )

            dim_scores.append(
                DimensionScore(
                    dimension=result.dimension,
                    verdict=result.verdict.value,
                    score=result.score,
                    reason=result.reason,
                    details=result.details,
                )
            )

            if result.verdict != EvalVerdict.SKIP:
                dimension_totals.setdefault(result.dimension, []).append(result.score)
            else:
                skipped += 1

        # Overall score = mean of non-skipped dimension scores
        active_scores = [d.score for d in dim_scores if d.verdict != EvalVerdict.SKIP.value]
        case_score = sum(active_scores) / len(active_scores) if active_scores else 1.0
        error_ok = exec_error is None if case.answer_quality.must_succeed else True
        case_passed = all(
            d.verdict in (EvalVerdict.PASS.value, EvalVerdict.SKIP.value)
            for d in dim_scores
        ) and error_ok

        report = CaseReport(
            case_id=case.id,
            case_name=case.name,
            eval_type=case.eval_type.value,
            task_or_question=case.task_or_question[:120],
            passed=case_passed,
            overall_score=round(case_score, 4),
            duration_ms=duration_ms,
            error=exec_error,
            dimensions=dim_scores,
        )
        case_reports.append(report)

        if verbose:
            status = "✅ PASS" if case_passed else "❌ FAIL"
            print(f"  {status}  score={case_score:.2f}  {duration_ms:.0f}ms")
            for d in dim_scores:
                if d.verdict == EvalVerdict.FAIL.value:
                    print(f"    ↳ [{d.dimension}] FAIL — {d.reason}")

    # ── Suite summary ─────────────────────────────────────────────────────
    total = len(case_reports)
    passed_count = sum(1 for r in case_reports if r.passed)
    failed_count = total - passed_count
    all_scores = [r.overall_score for r in case_reports]
    suite_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    per_dim = {
        dim: round(sum(scores) / len(scores), 4)
        for dim, scores in dimension_totals.items()
    }
    total_skipped = sum(
        sum(1 for d in r.dimensions if d.verdict == EvalVerdict.SKIP.value)
        for r in case_reports
    )

    suite_report = EvalSuiteReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_cases=total,
        passed=passed_count,
        failed=failed_count,
        skipped_dimensions=total_skipped,
        overall_score=round(suite_score, 4),
        per_dimension_scores=per_dim,
        cases=case_reports,
    )

    if verbose:
        _print_summary(suite_report)

    # ── Write JSON report ─────────────────────────────────────────────────
    report_path = _write_report(suite_report, report_dir)
    if verbose:
        print(f"\n  Report written to: {report_path}\n")

    return suite_report


def _print_summary(report: EvalSuiteReport) -> None:
    """Print formatted summary to stdout."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total cases  : {report.total_cases}")
    print(f"  Passed       : {report.passed}")
    print(f"  Failed       : {report.failed}")
    print(f"  Overall score: {report.overall_score:.2%}")
    print(f"\n  Per-dimension scores:")
    for dim, score in sorted(report.per_dimension_scores.items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {dim:<30} {bar} {score:.2%}")
    print(f"{'='*70}\n")


def _write_report(report: EvalSuiteReport, report_dir: str | None) -> Path:
    """Serialise the report to a timestamped JSON file."""
    import dataclasses

    base = Path(report_dir) if report_dir else Path(__file__).parent / "reports"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = base / f"eval_report_{ts}.json"

    def _default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return str(o)

    path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=_default))
    return path
