#!/usr/bin/env bash
# ==============================================================================
# Enterprise Agentic Research Platform — Local Docker Compose Deployment
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo "🚀 Deploying Enterprise Agentic Platform (Local Docker Compose)"
echo "======================================================================"

cd "${ROOT_DIR}"

# 1. Check Docker & Compose Availability
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH."
    exit 1
fi

echo "📦 Building and starting containers..."
docker compose up -d --build

echo "⏳ Waiting for service readiness..."
for i in {1..30}; do
    if curl -s -f "http://localhost:8000/ready" > /dev/null 2>&1; then
        echo "✅ Platform is ready and healthy!"
        break
    fi
    echo "   ... waiting for API container (${i}/30)"
    sleep 2
done

echo ""
echo "======================================================================"
echo "🎉 Deployment Complete!"
echo "======================================================================"
echo "🌐 Research UI & Web App: http://localhost:8000/"
echo "📖 Swagger API Docs:      http://localhost:8000/docs"
echo "📊 Prometheus Metrics:    http://localhost:8000/metrics"
echo "🩺 Readiness Check:       http://localhost:8000/ready"
echo "======================================================================"
