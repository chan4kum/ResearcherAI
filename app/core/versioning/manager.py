"""
app/core/versioning/manager.py — Central Configuration Version Manager

Coordinates explicit versioning and cryptographic fingerprinting across:
- Prompt templates
- Model configurations
- Retrieval and vector search parameters
- Routing and decision logic
- Agent composite runtime state
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.core.versioning.models import (
    AgentConfigVersion,
    ConfigurationProvenance,
    ModelConfigVersion,
    RetrievalConfigVersion,
    RoutingConfigVersion,
)
from app.core.versioning.prompts import PromptRegistry, get_prompt_registry


class ConfigurationVersionManager:
    """Central manager providing configuration version tracking and execution provenance."""

    def __init__(
        self,
        settings: Settings | None = None,
        prompt_registry: PromptRegistry | None = None,
        agent_version: str = "1.0.0",
    ) -> None:
        self._settings = settings or get_settings()
        self._prompt_registry = prompt_registry or get_prompt_registry()
        self._agent_version = agent_version

        # Initialize versioned component configurations from settings
        self._model_config = self._init_model_config()
        self._retrieval_config = self._init_retrieval_config()
        self._routing_config = self._init_routing_config()

    @property
    def prompt_registry(self) -> PromptRegistry:
        return self._prompt_registry

    @property
    def model_config(self) -> ModelConfigVersion:
        return self._model_config

    @property
    def retrieval_config(self) -> RetrievalConfigVersion:
        return self._retrieval_config

    @property
    def routing_config(self) -> RoutingConfigVersion:
        return self._routing_config

    @property
    def agent_version(self) -> str:
        return self._agent_version

    def _init_model_config(self) -> ModelConfigVersion:
        """Derive active ModelConfigVersion from environment settings."""
        return ModelConfigVersion(
            version="1.0.0",
            provider=self._settings.llm_provider,
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            timeout_seconds=self._settings.llm_timeout_seconds,
            max_retries=self._settings.llm_max_retries,
        )

    def _init_retrieval_config(self) -> RetrievalConfigVersion:
        """Derive active RetrievalConfigVersion from environment settings."""
        return RetrievalConfigVersion(
            version="1.0.0",
            retrieval_mode=self._settings.default_retrieval_mode,
            fusion_strategy=self._settings.hybrid_fusion_strategy,
            top_k=5,
            similarity_threshold=self._settings.min_retrieval_relevance_threshold,
            enable_reranking=self._settings.enable_reranking,
            reranker_provider=self._settings.reranker_provider,
            reranker_top_n=self._settings.reranker_top_n,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
        )

    def _init_routing_config(self) -> RoutingConfigVersion:
        """Derive active RoutingConfigVersion from environment settings."""
        return RoutingConfigVersion(
            version="1.0.0",
            default_strategy=self._settings.default_retrieval_strategy,
            enable_hyde=self._settings.enable_hyde,
            enable_query_rewriting=self._settings.enable_query_rewriting,
            max_rewriting_attempts=self._settings.max_retrieval_attempts,
            fallback_strategy="normal",
        )

    def set_model_config(self, config: ModelConfigVersion) -> None:
        """Update active ModelConfigVersion."""
        self._model_config = config

    def set_retrieval_config(self, config: RetrievalConfigVersion) -> None:
        """Update active RetrievalConfigVersion."""
        self._retrieval_config = config

    def set_routing_config(self, config: RoutingConfigVersion) -> None:
        """Update active RoutingConfigVersion."""
        self._routing_config = config

    def get_active_agent_config(
        self,
        tools_registered: list[str] | None = None,
        name: str = "agent_core",
    ) -> AgentConfigVersion:
        """Construct the unified composite AgentConfigVersion snapshot for an execution."""
        return AgentConfigVersion(
            agent_version=self._agent_version,
            name=name,
            prompt_versions=self._prompt_registry.get_active_versions(),
            model_config_version=self._model_config.version,
            retrieval_config_version=self._retrieval_config.version,
            routing_config_version=self._routing_config.version,
            tools_registered=tools_registered or [],
        )

    def create_provenance(
        self,
        tools_registered: list[str] | None = None,
        name: str = "agent_core",
    ) -> ConfigurationProvenance:
        """Generate a cryptographic ConfigurationProvenance audit record for telemetry/tracing."""
        cfg = self.get_active_agent_config(tools_registered=tools_registered, name=name)
        return ConfigurationProvenance(
            composite_hash=cfg.composite_hash,
            agent_version=cfg.agent_version,
            prompt_versions=cfg.prompt_versions,
            model_config_version=cfg.model_config_version,
            retrieval_config_version=cfg.retrieval_config_version,
            routing_config_version=cfg.routing_config_version,
        )

    def get_snapshot_summary(self) -> dict[str, object]:
        """Return human-readable summary of all active versions and hashes."""
        cfg = self.get_active_agent_config()
        return {
            "agent_version": self._agent_version,
            "composite_hash": cfg.composite_hash,
            "prompt_versions": self._prompt_registry.get_active_versions(),
            "model": {
                "version": self._model_config.version,
                "provider": self._model_config.provider,
                "model": self._model_config.model,
                "hash": self._model_config.config_hash,
            },
            "retrieval": {
                "version": self._retrieval_config.version,
                "mode": self._retrieval_config.retrieval_mode,
                "hash": self._retrieval_config.config_hash,
            },
            "routing": {
                "version": self._routing_config.version,
                "strategy": self._routing_config.default_strategy,
                "hash": self._routing_config.config_hash,
            },
        }


_GLOBAL_VERSION_MANAGER: ConfigurationVersionManager | None = None


def get_version_manager(settings: Settings | None = None) -> ConfigurationVersionManager:
    """Retrieve the global ConfigurationVersionManager singleton."""
    global _GLOBAL_VERSION_MANAGER
    if _GLOBAL_VERSION_MANAGER is None:
        _GLOBAL_VERSION_MANAGER = ConfigurationVersionManager(settings=settings)
    return _GLOBAL_VERSION_MANAGER
