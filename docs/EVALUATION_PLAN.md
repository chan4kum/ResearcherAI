# Evaluation Plan

This document defines how to evaluate ResearcherAI before publishing performance, quality, latency, or cost claims. No benchmark results are recorded here because they have not been run and verified in this PR.

## Goals

Evaluate whether the system retrieves relevant evidence, preserves citation metadata, resists common prompt-injection patterns, and remains observable during normal and failure paths.

## Evaluation Areas

### Retrieval Quality

Measure retrieval quality with a fixed question set and expected evidence references.

Recommended metrics:

- Recall@K for expected chunks or documents.
- Precision@K for retrieved chunks judged relevant.
- Mean reciprocal rank for first relevant result.
- Source coverage for questions requiring multiple sources.

Required artifacts before claiming results:

- Versioned evaluation dataset.
- Exact commit SHA.
- Configuration used for retrieval mode, top_k, reranking, HyDE, and rewriting.
- Raw outputs and judging rubric.

### Answer Grounding

Evaluate generated answers separately from retrieval.

Recommended checks:

- Every citation marker in the answer maps to returned citation metadata.
- Factual claims are supported by retrieved context.
- Unsupported claims are flagged rather than presented as facts.
- Missing evidence leads to uncertainty instead of hallucinated detail.

### Safety and Guardrails

Test prompt-injection and unsafe-document scenarios against `app/core/guardrails/`.

Recommended cases:

- Direct instruction override attempts.
- System prompt extraction attempts.
- Chat-template delimiter injection.
- Indirect prompt injection inside retrieved text.
- Secrets-like strings in model output.
- Unsafe URL patterns for SSRF-oriented checks.

### Observability

Verify that normal and failure paths emit useful operational signals.

Recommended checks:

- HTTP request counters and latency histograms are exposed on `/metrics`.
- LLM token and estimated cost counters are recorded when token usage is available.
- Retrieval metrics record strategy and status.
- Tracing spans are emitted for agent/RAG operations where instrumentation is configured.

### Resilience and Cost Controls

Exercise failure paths rather than only happy paths.

Recommended cases:

- Upstream LLM error.
- Empty retrieval results.
- Reranker failure.
- Rate-limit threshold exceeded.
- Cost budget threshold exceeded.
- Database unavailable during readiness check.

## Reporting Template

When evaluation is run, publish results in this format:

```md
## Evaluation Run

- Date:
- Commit SHA:
- Dataset version:
- Environment:
- Retrieval config:
- LLM provider/model:
- Number of questions:

| Metric | Result | Notes |
|---|---:|---|
| Recall@5 | TBD | |
| Precision@5 | TBD | |
| MRR | TBD | |
| Citation validity | TBD | |
| Unsupported-claim rate | TBD | |
```

## Non-Goals

This plan does not claim production traffic, customer adoption, security certification, or benchmark superiority over other systems.
