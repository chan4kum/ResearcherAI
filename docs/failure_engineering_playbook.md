# Enterprise Failure Engineering & Resilience Playbook

This document details the controlled failure engineering scenarios, automated detection mechanisms, recovery strategies, user impact analyses, and observability signals for all 12 platform failure modes.

---

## Failure Matrix Summary

| # | Failure Mode | Subsystem | Detection Mechanism | Recovery Mechanism | User Impact | Observability Signal |
|:---|:---|:---|:---|:---|:---|:---|
| **1** | **LLM Timeout** | LLM Adapter / Provider | `asyncio.TimeoutError` caught in `LLMService` | Exponential backoff retry (up to 2 attempts) $\rightarrow$ graceful fallback | Request latency increases; friendly 504 / fallback answer | `llm_errors_total{type="timeout"}` + span `error=true` |
| **2** | **LLM 500 / Crash** | LLM Provider | HTTP 500 / `RuntimeError` in provider client | Exception caught; `TaskStatus.FAILED` recorded with sanitized error | Informative error message ("Service temporarily unavailable") | `llm_call_errors_total{status="500"}` + `agent_task_failed` log |
| **3** | **Vector DB Unavailable** | ChromaDB / pgvector | `ConnectionRefusedError` in `BaseVectorRepository` | Circuit breaker trips; degraded search fallback or clear KB unavailable note | Informative response with parametric knowledge fallback | `vector_db_errors_total` + span `retrieval.error` |
| **4** | **MCP Server Offline** | MCP Protocol Transport | Connection timeout / refused in `MCPClient` | Tool execution marked `success=False`; agent reasons with degraded context | Agent answers without real-time external tool data | `mcp_tool_execution_errors_total` + log `mcp_server_unreachable` |
| **5** | **Malformed MCP Output** | MCP Protocol Payload | Pydantic validation error or JSON decode error | Output sanitized; `ToolExecutionResult(success=False)` returned | Agent proceeds with execution without crashing | `mcp_malformed_response_total` + log `mcp_response_parse_failed` |
| **6** | **Empty Retrieval** | RAG Search Engine | `len(citations) == 0` check in `RAGService` | Informative fallback ("No relevant context found in knowledge base") | User gets clear disclaimer instead of hallucinated facts | `rag_empty_retrieval_total` + span `retrieval.empty=true` |
| **7** | **Irrelevant Retrieval** | RAG Reranker / Evaluator | Similarity score $< 0.20$ threshold filter | Chunks pruned; query rewriter triggers reformulation | Prevents hallucination from irrelevant noise | `rag_low_similarity_filtered_total` + log `retrieval_pruned` |
| **8** | **Agent Loop Exceeded** | LangGraph State Graph | `ToolExecutionCircuitBreaker` (max 5 tools, max 10 steps) | Execution halted immediately; task status set to `FAILED` | Fast fail prevents hanging request or runaway bill | `agent_circuit_breaker_tripped_total` + log `loop_limit_exceeded` |
| **9** | **DB Connection Failure** | Relational DB / Session | `OperationalError` during query execution | Connection pool retry, rollback, and 503 response | 503 Service Unavailable with `Retry-After` header | `db_connection_pool_errors_total` + `/health/ready` returns 503 |
| **10** | **K8s Pod Crash (OOM)** | Kubernetes Node / Pod | Kubelet process exit code (137 OOMKilled / SIGSEGV) | ReplicaSet restarts container (`restartPolicy: Always`) | In-flight request gets 502; next request routes to healthy pod | `kube_pod_container_status_restarts_total` + `OOMKilling` event |
| **11** | **Failed Deployment** | Kubernetes Rollout | RollingUpdate detects failing readiness probe (30s) | Deployment rollout halts; automated `helm rollback` | Zero downtime for end users; old ReplicaSet serves traffic | `kube_deployment_status_replicas_unavailable > 0` |
| **12** | **Failed Readiness Probe** | K8s Ingress & Endpoints | `/health/ready` returns HTTP 503 | Kubelet removes pod IP from Endpoints pool | Zero user impact; traffic routed only to ready pods | `kube_endpoint_address_available` drops + 503 metric |

---

## Detailed Failure Mode Runbooks

### 1. LLM Timeout
- **Failure**: Upstream LLM provider latency exceeds configured client timeout ($30.0\text{s}$) due to network congestion or server overload.
- **Detection**: `asyncio.wait_for` raises `asyncio.TimeoutError` inside `LLMService.generate()`.
- **Recovery**: Automatic single retry with exponential backoff ($1.5\times$). If failure persists, `LLMService` raises structured `LLMTimeoutError`, setting `TaskStatus.FAILED`.
- **User Impact**: In-flight request experiences high latency ($>30\text{s}$) and receives a friendly message: *"The language model request timed out. Please try again shortly."*
- **Observability Signal**:
  - OpenTelemetry span: `llm.request` with attribute `error.type: TimeoutError`.
  - Prometheus Metric: `llm_errors_total{type="timeout", model="gpt-4o-mini"}`.
  - Log: `{"event": "llm_request_timeout", "timeout_seconds": 30.0, "level": "error"}`.

---

### 2. LLM Provider Failure (500 / Rate Limit)
- **Failure**: Upstream provider returns HTTP 500 Internal Server Error, 429 Too Many Requests, or terminates the connection unexpectedly.
- **Detection**: Provider adapter catches `httpx.HTTPStatusError` or `RuntimeError`.
- **Recovery**: Exception is intercepted by `BasicAgent` exception boundary; error message is scrubbed by `SecretsScrubber` to prevent token leakage; task status is marked `TaskStatus.FAILED`.
- **User Impact**: Client receives HTTP 500/502 JSON error response with safe error text.
- **Observability Signal**:
  - Prometheus Metric: `llm_calls_total{status="failed", provider="mock"}`.
  - Structured Log: `{"event": "agent_task_failed", "error": "LLM 500 Internal Server Error", "agent_stage": "failed"}`.

---

### 3. Vector Database Unavailable
- **Failure**: Vector database process (ChromaDB / PostgreSQL pgvector) crashes or network partition prevents vector search.
- **Detection**: `BaseVectorRepository.search()` raises `ConnectionRefusedError` or socket timeout.
- **Recovery**: `RAGService` catches vector store exception and falls back to hybrid text search or returns an informational message that document retrieval is temporarily unavailable.
- **User Impact**: Queries depending on indexed documents receive an explicit notice rather than an unhandled 500 server crash.
- **Observability Signal**:
  - OpenTelemetry span: `rag.retrieval` with attribute `retrieval.fallback: true`.
  - Prometheus Metric: `retrieval_executions_total{status="failed"}`.

---

### 4. MCP Server Unavailable / Disconnected
- **Failure**: External Model Context Protocol (MCP) server daemon is unreachable on its configured socket or port.
- **Detection**: `MCPClient.call_tool()` fails with connection timeout or connection refused error.
- **Recovery**: Tool wrapper catches connection exception and returns `ToolExecutionResult(success=False, error="MCP server unreachable")`. The Agent’s LangGraph planner receives the failed observation and continues to generate a fallback answer.
- **User Impact**: Agent completes task using internal capabilities while noting that external live tool data was unavailable.
- **Observability Signal**:
  - OpenTelemetry span: `agent.tool_execution` with `tool.success: false`.
  - Structured Log: `{"event": "tool_node_completed", "tool_name": "mcp_search", "success": false}`.

---

### 5. Malformed MCP Result
- **Failure**: An external MCP tool returns corrupted JSON or schema-incompatible payload.
- **Detection**: `json.loads` or Pydantic validation fails inside `MCPClient` response deserializer.
- **Recovery**: Deserialization error is captured; raw payload is safely truncated and logged; `ToolExecutionResult(success=False, error="Malformed MCP output")` is passed to the Agent.
- **User Impact**: Request completes safely without unhandled exception; agent explains parsing difficulty.
- **Observability Signal**:
  - Structured Log: `{"event": "mcp_response_parse_failed", "level": "warning"}`.

---

### 6. Empty Retrieval
- **Failure**: User queries a topic that has zero matches in the vector database or query keywords are completely absent.
- **Detection**: `RAGService.answer()` receives `len(citations) == 0`.
- **Recovery**: `RAGService` formats a default prompt indicating no context was discovered. The model generates a polite response with parametric knowledge and explicit disclaimer.
- **User Impact**: Zero hallucinations. User is informed that no authoritative knowledge base chunks matched their request.
- **Observability Signal**:
  - OpenTelemetry span attribute: `retrieval.empty: true`.
  - Prometheus Metric: `retrieval_citations_count{count="0"}`.

---

### 7. Irrelevant Retrieval
- **Failure**: Retriever returns document chunks with low similarity scores ($<0.20$), introducing distracting noise into prompt.
- **Detection**: `RetrievalEvaluator` and `Reranker` score all candidate chunks below similarity threshold.
- **Recovery**: Low-scoring chunks are automatically pruned; query rewriter attempts alternative phrasing or RAG engine informs the user of low confidence.
- **User Impact**: User receives factual, high-precision answers rather than hallucinations driven by noise.
- **Observability Signal**:
  - Structured Log: `{"event": "retrieval_chunks_pruned_low_confidence", "pruned_count": 3}`.

---

### 8. Agent Loop Exceeding Limits
- **Failure**: Agent enters an infinite reasoning loop or attempts recursive tool executions due to ambiguous instructions.
- **Detection**: `ToolExecutionCircuitBreaker` counts tool invocations and state graph transitions; triggers when `tool_calls > 5` or `iterations > 10`.
- **Recovery**: Circuit breaker raises limit exception, terminating graph execution immediately and setting `TaskStatus.FAILED`.
- **User Impact**: Eliminates hanging requests and protects user/tenant against runaway token costs.
- **Observability Signal**:
  - Prometheus Metric: `agent_circuit_breaker_tripped_total`.
  - Structured Log: `{"event": "agent_loop_limit_exceeded", "iterations": 11, "max": 10}`.

---

### 9. Database Connection Failure
- **Failure**: PostgreSQL connection pool exhaustion or network disconnection to persistent store.
- **Detection**: SQLAlchemy / DB session raises `OperationalError("Connection pool exhausted")`.
- **Recovery**: Transaction rollback is executed; connection pool is recycled; API returns HTTP 503 with `Retry-After: 5`.
- **User Impact**: Clean 503 response alerting client to retry after brief interval.
- **Observability Signal**:
  - `/health/ready` check returns HTTP 503 with `checks.database.status: "unhealthy"`.
  - Prometheus Metric: `db_connection_pool_errors_total`.

---

### 10. Kubernetes Pod Crash
- **Failure**: Pod process terminates due to unhandled SIGSEGV, hardware failure, or OOMKilled (Exit Code 137).
- **Detection**: Kubelet container runtime detects process exit (`CrashLoopBackOff` / `OOMKilled`).
- **Recovery**: Kubelet restarts container per `restartPolicy: Always`. Ingress routes subsequent traffic to surviving replica pods in the ReplicaSet.
- **User Impact**: In-flight HTTP request on failed container receives 502/504; subsequent requests succeed on healthy replicas.
- **Observability Signal**:
  - Prometheus Metric: `kube_pod_container_status_restarts_total > 0`.
  - Kubernetes Event: `OOMKilling` or `BackOff`.

---

### 11. Failed Deployment Rollout
- **Failure**: Broken container image or bad configuration deployed via Helm / `kubectl apply`.
- **Detection**: Kubernetes `RollingUpdate` strategy checks new Pod readiness probes. Probes fail for `initialDelaySeconds + periodSeconds * failureThreshold`.
- **Recovery**: Rollout stalls automatically; old ReplicaSet continues serving 100% of production traffic. Automated pipeline executes `helm rollback`.
- **User Impact**: Zero downtime. Production traffic never reaches unready pods.
- **Observability Signal**:
  - Kubernetes Metric: `kube_deployment_status_replicas_unavailable > 0`.
  - Prometheus Alert: `DeploymentRolloutStalled`.

---

### 12. Failed Readiness Probe
- **Failure**: Internal dependency (vector DB / memory / LLM client) degrades on a specific pod.
- **Detection**: `/health/ready` returns HTTP 503 (`status: not_ready`).
- **Recovery**: Kubelet marks pod `Ready: False` and removes its IP from the Service Endpoints routing pool.
- **User Impact**: Clients experience zero failures; traffic is routed exclusively to fully operational pods.
- **Observability Signal**:
  - Kubernetes Metric: `kube_endpoint_address_available` decrements.
  - Health probe log: `readiness_probe_failed_http_503`.
