# ADR 0001: Retrieval and Agent Architecture

## Status

Accepted for current reference implementation.

## Context

ResearcherAI needs to demonstrate a credible research-assistant architecture without hiding the distinction between implemented behavior and unmeasured production claims. The codebase combines RAG, document ingestion, agent workflow orchestration, safety checks, and observability.

## Decision

Use FastAPI as the service boundary, a database-backed vector repository for document chunks, configurable retrieval strategies, and a compact LangGraph workflow for agentic task handling.

The RAG path supports:

- Dense vector retrieval.
- BM25 keyword retrieval.
- Hybrid fusion.
- HyDE-based retrieval.
- Optional reranking through a provider interface.
- Citation metadata returned through API schemas.

The agent path uses a LangGraph graph with planner, tool-decision, optional tool-execution, and answer nodes.

## Consequences

Benefits:

- The codebase shows real AI platform components rather than a README-only concept.
- Retrieval behavior can be evaluated independently from answer generation.
- Safety and observability hooks are visible in code and can be tested.
- The graph topology is small enough for reviewers to understand quickly.

Tradeoffs:

- The current reranker implementation is an offline deterministic implementation, not a production cross-encoder provider.
- Production performance, answer quality, and security posture still require measured evaluation.
- Deployment claims must stay scoped to assets actually present in the repository.

## Recruiter-Facing Interpretation

This project is best presented as evidence of LLM application architecture, RAG implementation, guardrail design, and operational thinking. It should not be presented as a proven production system without evaluation reports and deployment evidence.
