# Enterprise Agentic Research & Knowledge Platform

> **Milestone 09 — Basic RAG (Retrieval-Augmented Generation)**  
> A clean, modern Python foundation featuring an isolated LLM service layer, local safe tools, stateful LangGraph workflows, a Document Ingestion Engine with deduplication, a Deterministic Document Chunking Engine, an Embedding Service Abstraction, a Vector Database Subsystem (PostgreSQL with pgvector + in-memory fallback), and a **Basic RAG Subsystem** (`VectorRetriever`, `RAGService`, and verified citation metadata).

---

## Features

- **Basic RAG Subsystem (`VectorRetriever`, `RAGService`)**:
  - **VectorRetriever**: Query embedding and $K$-nearest neighbor similarity search across document chunks.
  - **Context Augmentation**: Formats cited context blocks with similarity scores and provenance tags (`[Citation 1] (Source: ...)`).
  - **Grounded Answer Synthesis**: Enforces strict factual grounding in retrieved chunks with provenance citations.
  - **Citation Metadata**: Every answer reports referenced chunks, doc IDs, sources, character offsets, and token consumption.
- **5-Stage Pipeline + RAG**:
  $$\text{Question} \longrightarrow \text{Embedding} \longrightarrow \text{Vector Search} \longrightarrow \text{Retrieved Chunks} \longrightarrow \text{LLM} \longrightarrow \text{Grounded Answer + Citations}$$
- **REST API Endpoints**:
  - `POST /api/v1/rag/query`: Ask questions and receive answers strictly grounded in the knowledge base with citation metadata.
  - `POST /api/v1/documents/search`: Vector similarity search returning top-$K$ matching chunks ordered by cosine similarity.
  - `POST /api/v1/documents/sync-kb`: Batch directory synchronization for knowledge base folders.
  - `POST /api/v1/documents/upload`: Text & PDF file upload with SHA-256 deduplication.
  - `POST /api/v1/documents/{doc_id}/chunk`: Document chunking with parameter overrides.
  - `POST /api/v1/embeddings/generate`: Batch vector embedding generation.
  - `POST /api/v1/embeddings/similarity`: Semantic similarity calculation between two texts.
- **Testing**: 112 async test cases executing 100% offline with `pytest` and `pytest-asyncio`.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+ (Python 3.12+ recommended)
- Docker & Docker Compose (optional for live PostgreSQL with pgvector)

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode with development tools
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
```

### 3. Start PostgreSQL with pgvector (Optional for live DB)
```bash
# Start PostgreSQL pgvector container via Docker Compose
docker compose up -d

# Or run the startup script
./scripts/start_postgres.sh
```

### 4. Run the Application
```bash
# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Verify Endpoints

```bash
# 1. Health Check
curl -i http://localhost:8000/health

# 2. Upload and Ingest a Document
curl -i -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@tests/fixtures/sample.txt"

# 3. Chunk, Embed, and Index into Vector DB (replace <DOC_ID>)
curl -i -X POST http://localhost:8000/api/v1/documents/<DOC_ID>/embed \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 100, "chunk_overlap": 20}'

# 4. Perform Semantic Similarity Search across Ingested Chunks
curl -i -X POST http://localhost:8000/api/v1/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query": "semiconductor photolithography wafer fabrication", "top_k": 3, "min_similarity": -1.0}'

# 5. Query Basic RAG for Grounded Answer with Citations
curl -i -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key stages of semiconductor manufacturing?", "top_k": 3}'
```

---

## Run Tests & Linting

```bash
# Run complete test suite (runs 100% offline)
pytest -v

# Run linter
ruff check .

# Run static type checker
mypy app
```

---

## Project Structure

```
app/
  main.py               # FastAPI application factory, lifespan & dependency wiring
  config.py             # Application configuration (DB, Embeddings, LLM)
  core/                 # Errors, logging, and middleware
  db/                   # Vector Database Subsystem
    base.py             # SQLAlchemy DeclarativeBase
    models.py           # DocumentRecord, DocumentChunkRecord with Vector(1536)
    session.py          # DatabaseManager & async engine sessionmaker
    repository.py       # BaseVectorRepository, PgVectorRepository, InMemoryVectorRepository
    migrations/         # Alembic migration scripts and env.py
  api/
    router.py           # Main API router
    v1/                 # API Version 1
      router.py
      endpoints/
        documents.py    # Document upload, chunking, embedding, vector search
        embeddings.py   # Embedding generation & similarity routes
        tasks.py        # Agent task execution (LangGraph)
        chat.py         # LLM chat completions
        health.py       # Health check route
        info.py         # System info route
  models/
    schemas.py          # Pydantic schemas (VectorSearch, Document, Chunk, Embedding, Task)
  services/
    document/           # Document ingestion, chunking, and repository orchestration
      models.py         # Document, ChunkMetadata, DocumentChunk, EmbeddedChunk
      chunker.py        # DocumentChunker & display_document_chunks visualizer
      store.py          # DocumentStore with SHA-256 deduplication
      service.py        # DocumentService orchestrator
      loaders/          # TextDocumentLoader, PDFDocumentLoader, Factory
    embedding/          # Embedding subsystem (Mock & OpenAI providers)
    agent/              # Agent service & LangGraph workflow
    llm/                # Isolated LLM service abstraction layer
scripts/
  start_postgres.sh     # Script to start PostgreSQL with pgvector container
  display_chunks.py     # CLI Debugging visualizer (document -> chunks)
docker-compose.yml      # Local development PostgreSQL + pgvector service
alembic.ini             # Alembic configuration
tests/                  # Async pytest test suite (101 tests)
tests/fixtures/         # Real test sample fixtures (sample.txt, sample.pdf, sample.md)
docs/                   # Architecture and technical design docs
```
