# Enterprise Agentic Platform — Grafana Operational Observability Runbook

## Overview

The **Enterprise Agentic Platform Production Dashboard** (`monitoring/grafana/agentic_platform_dashboard.json`) provides real-time operational telemetry across three core tiers:
1. **Platform Ingress & HTTP Service Level Objectives (SLOs)**
2. **Agentic Graph & Multi-Pass RAG Retrieval**
3. **Upstream LLM Latencies, Token Velocity & Cost Governance (FinOps)**

---

## Dashboard Panels & Operational Triage Guide

### Row 1: Platform Ingress & HTTP SLOs

#### Panel 1: HTTP Request Rate (QPS by Endpoint)
- **PromQL**: `sum(rate(http_requests_total[2m])) by (method, endpoint)`
- **Operational Question Answered**:
  > *"Is our API experiencing unexpected traffic spikes, load surges, or sudden dropped traffic across specific endpoints?"*
- **SRE Triage Action**:
  - Spike on `POST /api/v1/tasks` indicates batch agent invocation or upstream client retry storm.
  - Drop to 0 on `GET /health` indicates ingress routing failure, DNS outage, or load balancer health-check failure.

#### Panel 2: HTTP 5xx Error Rate (%)
- **PromQL**: `(sum(rate(http_requests_total{status_code=~"5.."}[2m])) / (sum(rate(http_requests_total[2m])) > 0)) * 100`
- **Operational Question Answered**:
  > *"What percentage of user requests are failing with 5xx HTTP errors, and are we violating our 99.9% availability SLA?"*
- **SRE Triage Action**:
  - `> 0.1%`: Triggers Warning alert.
  - `> 1.0%`: Triggers Critical PagerDuty alert. Check pod logs for unhandled exceptions, database connection pool exhaustion, or circuit breaker trips.

#### Panel 3: HTTP Request Latency (p50 / p95 / p99)
- **PromQL**:
  - p99: `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
  - p95: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
  - p50: `histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
- **Operational Question Answered**:
  > *"Are user-facing requests meeting responsiveness SLAs, and what is the tail latency experienced by the 99th percentile of users?"*
- **SRE Triage Action**:
  - High p99 divergence from p50 indicates head-of-line blocking, heavy multi-step agent queries, or cold start container latency.

---

### Row 2: Agent Graph & Multi-Pass Retrieval

#### Panel 4: Agent Graph Execution Time (p50 / p95)
- **PromQL**: `histogram_quantile(0.95, sum(rate(agent_execution_duration_seconds_bucket[5m])) by (le))`
- **Operational Question Answered**:
  > *"How long are multi-step agent reasoning graphs taking from initialization to final answer synthesis?"*
- **AI Ops Triage Action**:
  - Sudden latency increases indicate agent graphs taking more iterations than planned, excessive tool calls, or slow reranker models.

#### Panel 5: Retrieval Iterations & Rewrite Attempts
- **PromQL**: `sum(rate(retrieval_iterations_total[5m])) by (strategy, iteration)`
- **Operational Question Answered**:
  > *"Is the self-evaluator triggering excessive query rewrites (indicative of low retrieval quality or knowledge base drift)?"*
- **AI Ops Triage Action**:
  - High volume of `Attempt 2` and `Attempt 3` indicates user queries are failing initial vector search relevance thresholds, prompting investigation into embeddings indexing or knowledge base coverage gaps.

#### Panel 6: Agent Tool Calls & Success vs Failure
- **PromQL**: `sum(rate(agent_tool_calls_total[5m])) by (tool_name, status)`
- **Operational Question Answered**:
  > *"Which tools are called most frequently, and are third-party or MCP tools failing or timing out?"*
- **AI Ops Triage Action**:
  - Spikes in `status="failed"` for a specific tool (e.g. `web_search` or `calculator`) triggers immediate fallback routing or MCP server restart.

---

### Row 3: Upstream LLM Performance & Cost Governance

#### Panel 7: Upstream LLM Latency (p50 / p95)
- **PromQL**: `histogram_quantile(0.95, sum(rate(llm_call_duration_seconds_bucket[5m])) by (le, model, provider))`
- **Operational Question Answered**:
  > *"Is upstream model provider latency (OpenAI, Anthropic, Bedrock) degrading or causing downstream timeouts?"*
- **AI Ops Triage Action**:
  - High upstream latency isolated to a single provider triggers automatic provider fallback (e.g. switching traffic from OpenAI to Anthropic/Bedrock).

#### Panel 8: Token Consumption Velocity (Prompt vs Completion)
- **PromQL**: `sum(rate(llm_tokens_total[5m])) by (model, token_type)`
- **Operational Question Answered**:
  > *"Are prompt templates exploding in size, or are model completions inflating context windows and exhausting quotas?"*
- **FinOps Triage Action**:
  - Rapid growth in `prompt` tokens indicates prompt injection, bloated system prompts, or redundant document concatenation in RAG.

#### Panel 9: Estimated Hourly LLM Burn Rate ($/hr)
- **PromQL**: `sum(rate(llm_estimated_cost_dollars_total[1h])) by (model, provider) * 3600`
- **Operational Question Answered**:
  > *"What is our current hourly LLM financial burn rate, and are runaway agent loops causing budget overruns?"*
- **FinOps Triage Action**:
  - Provides real-time financial governance to enforce organizational cost limits before monthly provider billing cycles close.
