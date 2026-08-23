# Enterprise Agentic Research Platform — Deployment Guide

This guide details step-by-step instructions for deploying both the frontend UI and backend agentic services across **Local (Docker Compose)** and **Production (Kubernetes / EKS)** environments.

---

## 1. Quick Local Deployment (Docker Compose)

The fastest way to deploy the entire stack locally (PostgreSQL with `pgvector` + Unified Research Web App & API) is with Docker Compose.

### Step 1: Run the Deployment Script
```bash
./scripts/deploy_local.sh
```
*Or execute directly with Docker Compose:*
```bash
docker compose up -d --build
```

### Step 2: Access Endpoints
- **Research Web App (Frontend UI)**: [`http://localhost:8000/`](http://localhost:8000/)
- **Swagger Interactive API Documentation**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ReDoc Documentation**: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)
- **Prometheus Metrics**: [`http://localhost:8000/metrics`](http://localhost:8000/metrics)
- **System Readiness Check**: [`http://localhost:8000/ready`](http://localhost:8000/ready)

### Step 3: Tear Down Local Stack
```bash
docker compose down -v
```

---

## 2. Production Deployment (Kubernetes / EKS)

The application is fully containerized and includes production-grade Kubernetes manifests and Helm charts.

### Option A: Declarative Kubernetes Manifests (`k8s/`)

1. **Deploy all manifests**:
   ```bash
   ./scripts/deploy_k8s.sh agentic-platform false
   ```
   *Or apply individual files manually:*
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secret.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/hpa.yaml
   ```

2. **Verify Rollout & Pod Health**:
   ```bash
   kubectl rollout status deployment/agentic-api-deployment -n agentic-platform
   kubectl get pods,svc,hpa -n agentic-platform
   ```

---

### Option B: Production Helm Chart (`helm/agentic-platform/`)

1. **Deploy with EKS values profile**:
   ```bash
   ./scripts/deploy_k8s.sh agentic-platform true
   ```
   *Or with helm command:*
   ```bash
   helm upgrade --install agentic-platform ./helm/agentic-platform \
       --namespace agentic-platform \
       --create-namespace \
       -f ./helm/agentic-platform/values-eks.yaml
   ```

2. **Retrieve Production LoadBalancer / Ingress URL**:
   ```bash
   kubectl get svc agentic-platform -n agentic-platform -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
   ```

---

## 3. Production Configuration Variables

| Key | Description | Default |
|:---|:---|:---:|
| `APP_ENV` | Application environment (`production`, `development`) | `production` |
| `HOST` | Bind host IP | `0.0.0.0` |
| `PORT` | Listening HTTP port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `VECTOR_REPOSITORY_TYPE` | Vector store backend (`pgvector`, `memory`, `auto`) | `auto` |
| `LLM_PROVIDER` | Active LLM provider (`openai`, `mock`) | `mock` |
| `EMBEDDING_PROVIDER` | Active embedding provider (`openai`, `mock`) | `mock` |
| `RATE_LIMIT_ENABLED` | Enable sliding window rate limiter | `true` |
| `SECURITY_ENABLED` | Enable prompt injection & SSRF guardrails | `true` |

---

## 4. Verification & Smoke Testing

### A. Health & Readiness Probe
```bash
curl -i http://localhost:8000/ready
```

### B. Submit Research Query (RAG with Citations)
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are microservice design patterns?",
    "strategy": "normal",
    "top_k": 3,
    "rerank": true
  }'
```

### C. Execute Tool Agent Task
```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compute 1024 divided by 8 and explain the result."
  }'
```
