# ==============================================================================
# Multi-Stage Dockerfile for Enterprise Agentic Research & Knowledge Platform
# Stage 1: Build virtual environment and wheels
# Stage 2: Minimal, security-hardened production runtime with non-root user
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package dependency definition
COPY pyproject.toml README.md ./

# Install core dependencies into virtualenv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ------------------------------------------------------------------------------
# Stage 2: Final Production Runner
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

LABEL maintainer="Enterprise Agentic AI Team" \
      description="Production Docker Image for Enterprise Agentic Research Platform" \
      version="0.1.0"

# Create dedicated non-root user and group (UID/GID: 10001)
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /app -s /sbin/nologin -c "Unprivileged App User" appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application source code and knowledge base
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser KB/ /app/KB/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

# Ensure local data directory exists and is owned by appuser
RUN mkdir -p /app/.data/documents && \
    chown -R appuser:appuser /app

# Set production environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_ENV=production

# Switch to non-root execution
USER appuser

# Expose HTTP API port
EXPOSE 8000

# Native health check using standard library urllib
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Launch ASGI production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
