from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ResearcherAI — Autonomous Deep Research Platform"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    api_v1_prefix: str = "/api/v1"

    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000

    # LLM Settings
    llm_provider: str = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # OpenAI Specific Settings
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Google Gemini Settings
    gemini_api_key: str | None = None

    # External Search & Context APIs
    tavily_api_key: str | None = None
    contex7_api_key: str | None = None

    # Document Ingestion & Chunking Settings
    storage_dir: str = ".data/documents"
    knowledge_base_dir: str = "KB"
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50

    # Embedding Settings
    embedding_provider: str = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Database & Vector Storage Settings
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_db"
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    vector_repository_type: str = "auto"

    # Retrieval & RAG Settings
    default_retrieval_mode: str = "hybrid"
    hybrid_fusion_strategy: str = "rrf"
    hybrid_rrf_k: int = 60
    hybrid_alpha: float = 0.5
    enable_reranking: bool = False
    reranker_provider: str = "mock"
    reranker_top_n: int = 10
    enable_query_rewriting: bool = False
    max_retrieval_attempts: int = 3
    min_retrieval_relevance_threshold: float = 0.01
    min_entity_coverage_threshold: float = 0.5
    default_retrieval_strategy: str = "normal"
    enable_hyde: bool = False

    # Production Server & Resilience Settings
    request_timeout_seconds: float = 30.0
    graceful_shutdown_timeout_seconds: float = 10.0
    api_retry_max_attempts: int = 3
    api_retry_backoff_factor: float = 1.5
    api_retry_initial_delay: float = 0.2
    api_retry_max_delay: float = 5.0
    cors_allowed_origins: list[str] = ["*"]

    # API Security & Authentication Settings
    security_enabled: bool = True
    api_secret_key: str = "agentic-ai-platform-super-secret-key-2026"
    auth_tokens: dict[str, dict[str, Any]] = {
        "admin-token-secret-123": {
            "user_id": "usr_admin_001",
            "username": "admin",
            "roles": ["admin"],
        },
        "researcher-token-secret-456": {
            "user_id": "usr_res_001",
            "username": "dr_researcher",
            "roles": ["researcher"],
        },
        "user-token-secret-789": {
            "user_id": "usr_std_001",
            "username": "regular_user",
            "roles": ["user"],
        },
        "viewer-token-secret-000": {
            "user_id": "usr_view_001",
            "username": "guest_viewer",
            "roles": ["viewer"],
        },
    }

    # Resilience & Cost Guardrail Settings
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20
    max_research_iterations_cap: int = 5
    max_tool_calls_cap: int = 15
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: float = 10.0
    circuit_breaker_half_open_success_threshold: int = 2

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in {"development", "dev", "local"}

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful in testing)."""
    get_settings.cache_clear()
