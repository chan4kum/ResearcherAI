"""
app/api/dependencies.py — FastAPI Dependency Injection Providers

Provides clean, typed, and testable dependency injection providers for domain services.
Falls back gracefully to `request.app.state` singletons or constructs cached instances.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.core.versioning.manager import ConfigurationVersionManager, get_version_manager
from app.db.repository import BaseVectorRepository, create_vector_repository
from app.services.agent.service import AgentService
from app.services.agent.tools.registry import ToolRegistry
from app.services.document.service import DocumentService
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.rag.retriever import HybridRetriever, VectorRetriever
from app.services.rag.service import RAGService


def get_settings_dep(request: Request) -> Settings:
    """Retrieve application settings from app state or environment."""
    return getattr(request.app.state, "settings", None) or get_settings()


def get_llm_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> LLMService:
    """Resolve active LLMService singleton or factory instance."""
    service: LLMService | None = getattr(request.app.state, "llm_service", None)
    if service is None:
        service = LLMService(settings=settings)
        request.app.state.llm_service = service
    return service


def get_embedding_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> EmbeddingService:
    """Resolve active EmbeddingService singleton."""
    service: EmbeddingService | None = getattr(request.app.state, "embedding_service", None)
    if service is None:
        service = EmbeddingService(settings=settings)
        request.app.state.embedding_service = service
    return service


def get_vector_repository(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> BaseVectorRepository:
    """Resolve active BaseVectorRepository singleton."""
    repo: BaseVectorRepository | None = getattr(request.app.state, "vector_repository", None)
    if repo is None:
        repo = create_vector_repository(settings)
        request.app.state.vector_repository = repo
    return repo


def get_document_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    vector_repo: BaseVectorRepository = Depends(get_vector_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> DocumentService:
    """Resolve active DocumentService singleton."""
    service: DocumentService | None = getattr(request.app.state, "document_service", None)
    if service is None:
        service = DocumentService(
            vector_repository=vector_repo,
            embedding_service=embedding_service,
            settings=settings,
        )
        request.app.state.document_service = service
    return service


def get_tool_registry(
    request: Request,
) -> ToolRegistry:
    """Resolve active ToolRegistry singleton."""
    registry: ToolRegistry | None = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        registry = ToolRegistry()
        request.app.state.tool_registry = registry
    return registry


def get_version_manager_dep(
    settings: Settings = Depends(get_settings_dep),
) -> ConfigurationVersionManager:
    """Resolve active ConfigurationVersionManager."""
    return get_version_manager(settings)


def get_agent_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    llm_service: LLMService = Depends(get_llm_service),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    version_manager: ConfigurationVersionManager = Depends(get_version_manager_dep),
) -> AgentService:
    """Resolve active AgentService singleton."""
    service: AgentService | None = getattr(request.app.state, "agent_service", None)
    if service is None:
        service = AgentService(
            llm_service=llm_service,
            tool_registry=tool_registry,
            version_manager=version_manager,
            settings=settings,
        )
        request.app.state.agent_service = service
    return service


def get_rag_service(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    llm_service: LLMService = Depends(get_llm_service),
    vector_repo: BaseVectorRepository = Depends(get_vector_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    document_service: DocumentService = Depends(get_document_service),
) -> RAGService:
    """Resolve active RAGService singleton."""
    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        vector_retriever = VectorRetriever(
            embedding_service=embedding_service,
            vector_repository=vector_repo,
        )
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            document_store=document_service.store,
            settings=settings,
        )
        service = RAGService(
            retriever=hybrid_retriever,
            llm_service=llm_service,
            settings=settings,
        )
        request.app.state.rag_service = service
    return service
