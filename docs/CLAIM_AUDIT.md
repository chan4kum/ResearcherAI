# Claim Audit

This audit documents what the public README can claim based on the current repository contents. It is intentionally conservative: unsupported metrics, external production claims, and unimplemented infrastructure claims should not appear in recruiter-facing copy.

## Supported Claims

| Claim | Evidence |
|---|---|
| FastAPI backend | `app/main.py`, `app/api/v1/router.py`, `app/api/v1/endpoints/` |
| RAG endpoint with citation metadata | `app/api/v1/endpoints/rag.py`, `app/models/schemas.py`, `app/services/rag/models.py` |
| Dense vector retrieval | `app/services/rag/retriever.py`, `app/db/repository.py` |
| BM25 keyword retrieval | `app/services/rag/retriever.py`, `app/services/rag/bm25.py` |
| Hybrid retrieval and fusion | `app/services/rag/retriever.py`, `app/services/rag/fusion.py` |
| HyDE retrieval strategy | `app/services/rag/hyde.py`, `app/services/rag/retriever.py` |
| Reranking interface with deterministic offline implementation | `app/services/rag/reranker.py` |
| LangGraph agent workflow | `app/services/agent/graph/workflow.py`, `app/services/agent/graph/nodes.py` |
| Tool routing and execution | `app/services/agent/tools/`, `app/services/agent/graph/workflow.py` |
| Prompt injection and retrieved-context safety helpers | `app/core/guardrails/injection.py`, `app/core/guardrails/document_safety.py` |
| Secrets filtering and URL/tool governance helpers | `app/core/guardrails/secrets_filter.py`, `app/core/guardrails/ssrf.py`, `app/core/guardrails/tool_governance.py` |
| Metrics and tracing hooks | `app/core/metrics.py`, `app/core/tracing.py`, `app/core/middleware.py` |
| Resilience and cost controls | `app/core/resilience/`, `app/core/retry.py` |
| Database-backed vector schema | `app/db/models.py`, `app/db/migrations/versions/0001_initial_vector_schema.py` |
| CI and tests | `.github/workflows/ci.yml`, `tests/` |

## Claims That Need Qualification

| Previous style of claim | Safer replacement |
|---|---|
| Enterprise-grade autonomous research platform | Production-oriented research assistant/reference implementation |
| Every single fact is strictly tied to citations | Responses include citation metadata for retrieved context; answer grounding still requires evaluation |
| Cross-encoder reranker | Reranking interface with deterministic offline reranker unless a real cross-encoder provider is added |
| 398 tests passed | Test suite exists; cite current CI only when a specific run is verified |
| Terraform and Helm production deployment | Docker/CI assets exist; do not claim Terraform/Helm unless those directories are present |
| Production performance or reliability | No external production SLOs or benchmark results are currently documented |

## Recruiter-Facing Rule

Use this project to demonstrate architecture, implementation breadth, and safety/observability thinking. Do not claim production adoption, benchmarked accuracy, latency, or security guarantees until those are measured and documented.
