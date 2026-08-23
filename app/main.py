import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.api.v1.endpoints.health import get_health, get_liveness, get_readiness
from app.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import register_middlewares
from app.core.tracing import configure_tracing
from app.db.repository import BaseVectorRepository, create_vector_repository
from app.db.session import DatabaseManager
from app.models.schemas import HealthResponse, LivenessResponse, ReadinessResponse
from app.services.agent.service import AgentService
from app.services.agent.tools.app_info import AppInfoTool
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.document.service import DocumentService
from app.services.embedding.service import EmbeddingService
from app.services.llm.service import LLMService
from app.services.mcp import (
    LocalMCPServer,
    MCPDiscoveryManager,
    MCPSafetyPolicy,
    MCPToolAdapter,
)
from app.services.rag.adaptive import AdaptiveRetriever
from app.services.rag.agentic_retrieval.engine import AgenticRetrievalEngine
from app.services.rag.agentic_retrieval.trace_store import RetrievalTraceStore
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.critic.agent import CriticAgent
from app.services.rag.critic.engine import SelfCorrectionEngine
from app.services.rag.loop.orchestrator import AgenticResearchOrchestrator
from app.services.rag.research.executor import MultiStepResearchExecutor
from app.services.rag.research.planner import MultiStepResearchPlanner
from app.services.rag.retriever import create_retriever
from app.services.rag.router import RetrievalRouter
from app.services.rag.service import RAGService
from app.services.rag.sources import (
    KeywordSearchSource,
    RetrievalSourceRegistry,
    StructuredDatabasePlaceholderSource,
    VectorDatabaseSource,
    WebSearchPlaceholderSource,
)
from app.services.rag.verification.verifier import AnswerVerifier

logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycles."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    configure_tracing()  # Initialize OpenTelemetry tracing
    app.state.startup_time = time.perf_counter()
    logger.info(
        "app_startup",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        vector_repo=settings.vector_repository_type,
    )
    db_manager: DatabaseManager | None = getattr(app.state, "db_manager", None)
    if db_manager and db_manager.is_configured:
        await db_manager.init_db()

    yield

    logger.info("app_shutdown_initiated")
    if db_manager and db_manager.is_configured:
        await db_manager.close()
    logger.info("app_shutdown_completed")


def create_app(
    settings: Settings | None = None,
    llm_service: LLMService | None = None,
    agent_service: AgentService | None = None,
    document_service: DocumentService | None = None,
    embedding_service: EmbeddingService | None = None,
    db_manager: DatabaseManager | None = None,
    vector_repository: BaseVectorRepository | None = None,
    rag_service: RAGService | None = None,
) -> FastAPI:
    """FastAPI application factory."""
    current_settings = settings or get_settings()

    app = FastAPI(
        title=current_settings.app_name,
        version=current_settings.app_version,
        description="Foundational backend for Enterprise Agentic Research & Knowledge Platform.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Store settings and services in application state for easy access
    app.state.settings = current_settings
    resolved_db_manager = db_manager or DatabaseManager(settings=current_settings)
    app.state.db_manager = resolved_db_manager

    resolved_llm_service = llm_service or LLMService(settings=current_settings)
    app.state.llm_service = resolved_llm_service

    # Local MCP Server hosting migrated Calculator tool
    local_mcp_server = LocalMCPServer(server_name="in-process-math-server")
    local_mcp_server.register_tool(CalculatorTool())

    # MCP Safety Policy & Discovery Manager
    mcp_safety_policy = MCPSafetyPolicy(
        allowed_servers=["in-process-math-server", "local-mcp-server"],
        allowed_tools=["calculator", "app_info"],
        timeout_seconds=10.0,
        max_invocations=50,
        enforce_whitelist=True,
    )
    mcp_discovery_manager = MCPDiscoveryManager(safety_policy=mcp_safety_policy)
    resolved_mcp_client = mcp_discovery_manager.register_server(local_mcp_server)

    app.state.local_mcp_server = local_mcp_server
    app.state.mcp_client = resolved_mcp_client
    app.state.mcp_safety_policy = mcp_safety_policy
    app.state.mcp_discovery_manager = mcp_discovery_manager

    from app.services.agent.tools.context7 import Context7Tool
    from app.services.agent.tools.web_search import WebSearchTool

    # Agent ToolRegistry with internal tools (app_info, web_search, context7) and dynamic MCP tool registration
    agent_tool_registry = ToolRegistry(
        tools=[
            AppInfoTool(settings=current_settings),
            WebSearchTool(api_key=current_settings.tavily_api_key),
            Context7Tool(api_key=current_settings.contex7_api_key),
            MCPToolAdapter(
                mcp_client=resolved_mcp_client,
                tool_definition=local_mcp_server.list_tool_definitions()[0],
                safety_policy=mcp_safety_policy,
                tracker=mcp_discovery_manager.tracker,
            ),
        ],
        settings=current_settings,
    )
    app.state.tool_registry = agent_tool_registry
    app.state.agent_service = agent_service or AgentService(
        llm_service=resolved_llm_service,
        tool_registry=agent_tool_registry,
        settings=current_settings,
    )
    resolved_embedding_service = embedding_service or EmbeddingService(
        settings=current_settings
    )
    app.state.embedding_service = resolved_embedding_service

    resolved_vector_repo = vector_repository or create_vector_repository(
        settings=current_settings, db_manager=resolved_db_manager
    )
    app.state.vector_repository = resolved_vector_repo

    app.state.document_service = document_service or DocumentService(
        embedding_service=resolved_embedding_service,
        vector_repository=resolved_vector_repo,
        settings=current_settings,
    )

    resolved_retriever = create_retriever(
        embedding_service=resolved_embedding_service,
        vector_repository=resolved_vector_repo,
        mode=current_settings.default_retrieval_mode,
        settings=current_settings,
    )
    app.state.rag_service = rag_service or RAGService(
        retriever=resolved_retriever,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )

    # Source Registry & Concrete Retrieval Sources
    source_registry = RetrievalSourceRegistry()
    source_registry.register(
        VectorDatabaseSource(
            embedding_service=resolved_embedding_service,
            vector_repository=resolved_vector_repo,
            source_name="internal_vector_db",
        )
    )
    source_registry.register(
        KeywordSearchSource(
            vector_repository=resolved_vector_repo,
            source_name="keyword_search",
        )
    )
    source_registry.register(
        WebSearchPlaceholderSource(
            source_name="web_search_engine",
            api_key=current_settings.tavily_api_key,
        )
    )
    source_registry.register(
        StructuredDatabasePlaceholderSource(source_name="structured_sql_db")
    )
    app.state.retrieval_source_registry = source_registry

    # Query Analyzer & Retrieval Router
    resolved_query_analyzer = QueryAnalyzer(
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.query_analyzer = resolved_query_analyzer

    resolved_router = RetrievalRouter(
        query_analyzer=resolved_query_analyzer,
        llm_service=resolved_llm_service,
        registry=source_registry,
        settings=current_settings,
    )
    app.state.retrieval_router = resolved_router

    # Adaptive Retriever & Agentic Retrieval Engine
    resolved_adaptive_retriever = AdaptiveRetriever(
        router=resolved_router,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.adaptive_retriever = resolved_adaptive_retriever

    resolved_trace_store = RetrievalTraceStore()
    app.state.retrieval_trace_store = resolved_trace_store

    resolved_agentic_engine = AgenticRetrievalEngine(
        router=resolved_router,
        llm_service=resolved_llm_service,
        trace_store=resolved_trace_store,
        settings=current_settings,
    )
    app.state.agentic_retrieval_engine = resolved_agentic_engine

    resolved_research_planner = MultiStepResearchPlanner(
        query_analyzer=resolved_query_analyzer,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.research_planner = resolved_research_planner

    resolved_research_executor = MultiStepResearchExecutor(
        planner=resolved_research_planner,
        retrieval_router=resolved_router,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.research_executor = resolved_research_executor

    resolved_critic_agent = CriticAgent(
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.critic_agent = resolved_critic_agent

    resolved_self_correction_engine = SelfCorrectionEngine(
        critic_agent=resolved_critic_agent,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.self_correction_engine = resolved_self_correction_engine

    resolved_answer_verifier = AnswerVerifier(
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.answer_verifier = resolved_answer_verifier

    resolved_orchestrator = AgenticResearchOrchestrator(
        query_analyzer=resolved_query_analyzer,
        research_planner=resolved_research_planner,
        research_executor=resolved_research_executor,
        retrieval_router=resolved_router,
        self_correction_engine=resolved_self_correction_engine,
        answer_verifier=resolved_answer_verifier,
        llm_service=resolved_llm_service,
        settings=current_settings,
    )
    app.state.agentic_research_orchestrator = resolved_orchestrator

    # Register Middlewares
    register_middlewares(app, current_settings)

    # Register Global Exception Handlers
    register_exception_handlers(app)

    # Root health, readiness, and liveness endpoints
    app.add_api_route(
        "/health",
        get_health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["Health & Diagnostics"],
        summary="Root health check (backward compatible)",
    )
    app.add_api_route(
        "/ready",
        get_readiness,
        methods=["GET"],
        response_model=ReadinessResponse,
        tags=["Health & Diagnostics"],
        summary="Root readiness check",
    )
    app.add_api_route(
        "/live",
        get_liveness,
        methods=["GET"],
        response_model=LivenessResponse,
        tags=["Health & Diagnostics"],
        summary="Root liveness check",
    )

    # Prometheus Application Metrics
    @app.get(
        "/metrics",
        tags=["Health & Diagnostics"],
        summary="Prometheus Application Metrics",
        include_in_schema=True,
    )
    async def metrics() -> Response:
        from fastapi import Response
        from app.core.metrics import get_prometheus_metrics

        content, media_type = get_prometheus_metrics()
        return Response(content=content, media_type=media_type)

    # Include Versioned API Routes
    app.include_router(api_router)

    # Mount Static UI Assets and Root SPA Route
    import os
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def root_ui() -> FileResponse:
            """Serve the lightweight end-user research UI."""
            index_path = os.path.join(static_dir, "index.html")
            return FileResponse(index_path, media_type="text/html")

    return app


app = create_app()


def run() -> None:
    """Launch the server with Uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
