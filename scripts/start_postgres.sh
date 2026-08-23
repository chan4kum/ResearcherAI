#!/usr/bin/env bash
set -e

echo "Starting PostgreSQL with pgvector container..."
docker compose up -d postgres

echo "Waiting for PostgreSQL to become healthy..."
until docker compose exec postgres pg_isready -U postgres -d agentic_db; do
  sleep 1
done

echo "Running database migrations..."
alembic upgrade head

echo "PostgreSQL with pgvector is ready!"
