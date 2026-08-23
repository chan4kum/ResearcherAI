# Target Enterprise Agentic AI Architecture Specification

This specification documents the complete, approved enterprise architecture implemented across the platform, mapping the runtime agentic reasoning flows, multi-source retrieval loops, data persistence layers, production delivery pipelines, observability stack, and cloud infrastructure.

---

## 1. Agent Reasoning & Retrieval Core Architecture

```
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │   FastAPI   │
                    │ API Gateway │
                    └──────┬──────┘
                           │
                           ▼
                ┌────────────────────┐
                │ Agent Orchestrator │
                │     LangGraph      │
                └─────────┬──────────┘
                          │
             ┌────────────┼─────────────┐
             │            │             │
             ▼            ▼             ▼
        Query Agent   Research Agent  Tool Agent
             │            │             │
             └────────────┼─────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Retrieval Planner │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │  Source Router    │
                └─────────┬─────────┘
                          │
          ┌───────────────┼─────────────────┐
          │               │                 │
          ▼               ▼                 ▼
      Vector RAG      Hybrid Search     MCP Sources
          │               │                 │
          │               │                 ├── Tools
          │               │                 ├── Resources
          │               │                 └── External systems
          │               │
          └───────────────┼─────────────────┘
                          ▼
                  Evidence Evaluator
                          │
                    Sufficient?
                    /         \
                  No           Yes
                  │             │
                  ▼             │
          Query Rewriting       │
                  │             │
               HyDE             │
                  │             │
          Retrieve Again        │
                  │             │
                  └──────┬──────┘
                         ▼
                  Research Evidence
                         │
                         ▼
                  Answer Generator
                         │
                         ▼
                    Self-Critic
                         │
                  ┌──────┴──────┐
                  │             │
               Improve        Pass
                  │             │
                  └──────┬──────┘
                         ▼
                    Verification
                         │
                  ┌──────┴──────┐
                  │             │
                Fail           Pass
                  │             │
                  ▼             ▼
             Re-research    Final Answer
                                │
                                ▼
                         Cited Response
```

### Component Details
1. **FastAPI API Gateway** ([`app/main.py`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/main.py)):
   - Token authentication, RBAC authorization, rate limiting, request validation, structured logging, distributed tracing.
2. **LangGraph Agent Orchestrator** ([`app/services/agent/graph/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/agent/graph/)):
   - Dynamic multi-stage state graph (`planner` $\rightarrow$ `tool_decision` $\rightarrow$ `tool_executor` $\rightarrow$ `answer_generator`).
3. **Specialist Agents**:
   - **Query Agent**: Analyzes user intent, query complexity, and required response formatting.
   - **Research Agent** ([`app/services/rag/research/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/rag/research/)): Multi-step research loop with iterative query planning.
   - **Tool Agent**: Dispatches local mathematical and system tools along with remote Model Context Protocol (MCP) server tools.
4. **Retrieval Planning & Source Routing** ([`app/services/rag/router.py`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/rag/router.py)):
   - Routes queries across Vector DB, BM25 keyword search, Hybrid reciprocal rank fusion (RRF), and external MCP servers.
5. **Evidence Evaluator & Adaptive Retrieval Loop** ([`app/services/rag/adaptive.py`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/rag/adaptive.py)):
   - Evaluates retrieved citation relevance and sufficiency.
   - If insufficient: Triggers query rewriting and Hypothetical Document Embeddings (HyDE) for up to 3 bounded iterations.
6. **Answer Generation, Self-Critic, & Verification** ([`app/services/rag/critic/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/rag/critic/) and [`verification/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/services/rag/verification/)):
   - Assesses generated claims against cited source chunks for groundedness, hallucination mitigation, and citation precision.

---

## 2. Multi-Tiered Data Layer

```
                 DATA LAYER

        ┌─────────────────────────┐
        │ PostgreSQL              │
        │                         │
        │ Documents               │
        │ Chunks                  │
        │ Metadata                │
        │ Research state          │
        │ Evaluation data         │
        └───────────┬─────────────┘
                    │
                    ▼
                 pgvector

        ┌─────────────────────────┐
        │ Redis                   │
        │                         │
        │ Cache                   │
        │ Job state               │
        │ Rate limiting           │
        └─────────────────────────┘

        ┌─────────────────────────┐
        │ Object Storage          │
        │                         │
        │ PDFs                    │
        │ Documents               │
        │ Artifacts               │
        └─────────────────────────┘
```

- **PostgreSQL / pgvector** ([`app/db/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/db/)):
  - Stores relational document metadata, chunk text, vector embeddings (1536-dim HNSW indexing), session state, and evaluation logs.
- **Redis Cache & State**:
  - Ephemeral semantic query caching, embedding caches, distributed rate limiting, and async job queues.
- **Object Storage (S3 / GCS)**:
  - Persistent immutable storage for raw PDF/Markdown/JSON uploads and generated evaluation reports.

---

## 3. Production CI/CD & Delivery Pipeline

```
                         GitHub
                            │
                            ▼
                     GitHub Actions
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
              Tests      Security    Evaluation
                │
                ▼
            Docker Build
                │
                ▼
               ECR
                │
                ▼
          Kubernetes / EKS
                │
          ┌─────┴─────┐
          ▼           ▼
        API Pods    Worker Pods
          │           │
          └─────┬─────┘
                │
          ┌─────┴───────────────┐
          ▼                     ▼
       PostgreSQL             Redis
          │
       pgvector
```

- **GitHub Actions CI** ([`.github/workflows/ci.yml`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/.github/workflows/ci.yml)):
  - 3-stage validation pipeline:
    1. `test`: Runs 387+ pytest test cases across unit, integration, and security domains.
    2. `security-scan`: Scans for vulnerabilities, secrets, and SSRF guardrails.
    3. `llm-regression-testing`: Runs 18-case AI evaluation suite against quality thresholds ($\ge 85\%$).
- **Container Registry & EKS Orchestration** ([`helm/agentic-platform/`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/helm/agentic-platform/)):
  - Docker container packaging, ECR push, and Helm deployment to Kubernetes/EKS clusters with HPA autoscaling.

---

## 4. Full-Stack Observability

```
OpenTelemetry
      │
      ├── Traces  ──> Jaeger / AWS X-Ray
      ├── Metrics ──> Prometheus ──> Grafana Dashboards
      └── Logs    ──> Structured JSON / CloudWatch
```

- **OpenTelemetry Tracing** ([`app/core/tracing.py`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/core/tracing.py)):
  - End-to-end span propagation across `api.request` $\rightarrow$ `agent.task` $\rightarrow$ `agent.planner` $\rightarrow$ `retrieval.query` $\rightarrow$ `agent.tool_execution` $\rightarrow$ `llm.request`.
- **Prometheus Metrics** ([`app/core/metrics.py`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/app/core/metrics.py)):
  - Tracks HTTP requests, latency histograms, error rates, token consumption, retrieval iterations, and tool call frequencies.
- **Grafana Dashboard** ([`dashboards/grafana_agentic_ai_dashboard.json`](file:///Users/chandankumar/Desktop/Devops/Agentic_AI_complete/dashboards/grafana_agentic_ai_dashboard.json)):
  - 10 operational panels monitoring throughput, P95/P99 latency, cost, and tool health.

---

## 5. Infrastructure as Code (Terraform)

```
Terraform
    │
    ├── VPC (Public/Private Subnets, NAT Gateways, Security Groups)
    ├── EKS Cluster (Managed Node Groups, Cluster Autoscaler)
    ├── IAM Roles & IRSA (Least-privilege policies for S3, ECR, CloudWatch)
    ├── ECR Repository (Immutable image tags, vulnerability scanning)
    ├── RDS PostgreSQL (pgvector extension, automated backups, multi-AZ)
    └── Networking (Route53, ACM TLS certificates, ALB Ingress Controller)
```
