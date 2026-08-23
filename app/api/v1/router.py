from fastapi import APIRouter

from app.api.v1.endpoints import auth, chat, documents, embeddings, health, info, rag, tasks

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health & Diagnostics"])
api_v1_router.include_router(auth.router, prefix="/auth", tags=["Authentication & RBAC"])
api_v1_router.include_router(info.router, tags=["System Info"])
api_v1_router.include_router(chat.router, tags=["LLM Chat"])
api_v1_router.include_router(tasks.router, tags=["Agent Tasks"])
api_v1_router.include_router(documents.router, prefix="/documents", tags=["Document Ingestion"])
api_v1_router.include_router(embeddings.router, tags=["Embeddings"])
api_v1_router.include_router(rag.router, prefix="/rag", tags=["RAG (Knowledge Retrieval)"])


