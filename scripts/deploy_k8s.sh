#!/usr/bin/env bash
# ==============================================================================
# Enterprise Agentic Research Platform — Kubernetes / Production Deployment
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAMESPACE="${1:-agentic-platform}"
USE_HELM="${2:-false}"

echo "======================================================================"
echo "🚀 Deploying Enterprise Agentic Platform to Kubernetes"
echo "   Namespace: ${NAMESPACE}"
echo "======================================================================"

cd "${ROOT_DIR}"

if [ "${USE_HELM}" = "true" ]; then
    echo "📦 Deploying via Helm Chart (helm/agentic-platform)..."
    helm upgrade --install agentic-platform ./helm/agentic-platform \
        --namespace "${NAMESPACE}" \
        --create-namespace \
        -f ./helm/agentic-platform/values-eks.yaml
else
    echo "📦 Applying declarative Kubernetes manifests (k8s/)..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
fi

echo "⏳ Waiting for deployment rollout to complete..."
kubectl rollout status deployment/agentic-api-deployment -n "${NAMESPACE}" --timeout=120s

echo ""
echo "======================================================================"
echo "🎉 Production Kubernetes Rollout Succeeded!"
echo "======================================================================"
kubectl get pods,svc,hpa -n "${NAMESPACE}"
echo "======================================================================"
