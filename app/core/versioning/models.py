"""
app/core/versioning/models.py — Data models for explicit version management

Provides structured, immutable schemas for:
- Prompts (version, template, hash, metadata)
- Model configurations (provider, model, sampling parameters, hash)
- Retrieval configurations (mode, top_k, similarity, reranking, embeddings, hash)
- Routing configurations (strategy, rewriting, fallback, hash)
- Agent composite configurations (complete reproducible execution state snapshot)
- Configuration provenance tracking for execution audits
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def compute_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 fingerprint for a dict or string."""
    if isinstance(data, str):
        content = data.encode("utf-8")
    elif isinstance(data, dict):
        content = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    elif hasattr(data, "model_dump"):
        content = json.dumps(data.model_dump(mode="json"), sort_keys=True, default=str).encode("utf-8")
    else:
        content = str(data).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class PromptVersion(BaseModel):
    """Immutable representation of a versioned prompt template."""

    name: str = Field(description="Logical prompt name (e.g., 'planner', 'tool_decider')")
    version: str = Field(description="Semantic version string (e.g., '1.0.0')")
    template: str = Field(description="Raw template string or instructions")
    description: str = Field(default="", description="Description of the prompt intent and behavior")
    variables: list[str] = Field(
        default_factory=list,
        description="List of expected template substitution variable names",
    )
    prompt_hash: str = Field(
        default="",
        description="Deterministic SHA-256 fingerprint of the prompt template string",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp when this prompt version was created",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.prompt_hash:
            self.prompt_hash = compute_sha256(self.template.strip())

    def format(self, **kwargs: Any) -> str:
        """Format the prompt template with provided keyword arguments safely."""
        return self.template.format(**kwargs)


class ModelConfigVersion(BaseModel):
    """Versioned LLM provider and inference parameter configuration."""

    version: str = Field(default="1.0.0", description="Configuration version")
    provider: str = Field(default="mock", description="LLM provider adapter (mock, openai, anthropic)")
    model: str = Field(default="gpt-4o-mini", description="Base model name or endpoint identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Max generation tokens")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Request timeout")
    max_retries: int = Field(default=2, ge=0, description="Max retry attempts")
    config_hash: str = Field(default="", description="Deterministic SHA-256 hash of configuration")

    def model_post_init(self, __context: Any) -> None:
        if not self.config_hash:
            payload = {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            }
            self.config_hash = compute_sha256(payload)


class RetrievalConfigVersion(BaseModel):
    """Versioned retrieval and vector search parameter configuration."""

    version: str = Field(default="1.0.0", description="Configuration version")
    retrieval_mode: str = Field(default="hybrid", description="Retrieval mode (vector, keyword, hybrid)")
    fusion_strategy: str = Field(default="rrf", description="Fusion algorithm (rrf, weighted)")
    top_k: int = Field(default=5, ge=1, description="Number of candidate chunks retrieved")
    similarity_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Min similarity score")
    enable_reranking: bool = Field(default=False, description="Whether neural reranking is active")
    reranker_provider: str = Field(default="mock", description="Reranker provider adapter")
    reranker_top_n: int = Field(default=10, ge=1, description="Number of candidates to rerank")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    embedding_dimensions: int = Field(default=1536, ge=1, description="Embedding vector dimensionality")
    config_hash: str = Field(default="", description="Deterministic SHA-256 hash of configuration")

    def model_post_init(self, __context: Any) -> None:
        if not self.config_hash:
            payload = {
                "retrieval_mode": self.retrieval_mode,
                "fusion_strategy": self.fusion_strategy,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "enable_reranking": self.enable_reranking,
                "reranker_provider": self.reranker_provider,
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
            }
            self.config_hash = compute_sha256(payload)


class RoutingConfigVersion(BaseModel):
    """Versioned decision routing and query processing strategy configuration."""

    version: str = Field(default="1.0.0", description="Configuration version")
    default_strategy: str = Field(default="normal", description="Default strategy (normal, hyde, adaptive)")
    enable_hyde: bool = Field(default=False, description="Whether HyDE strategy is enabled")
    enable_query_rewriting: bool = Field(default=False, description="Whether query rewriting is active")
    max_rewriting_attempts: int = Field(default=3, ge=1, description="Max rewriting attempts on low score")
    fallback_strategy: str = Field(default="normal", description="Fallback routing path on failure")
    config_hash: str = Field(default="", description="Deterministic SHA-256 hash of configuration")

    def model_post_init(self, __context: Any) -> None:
        if not self.config_hash:
            payload = {
                "default_strategy": self.default_strategy,
                "enable_hyde": self.enable_hyde,
                "enable_query_rewriting": self.enable_query_rewriting,
                "max_rewriting_attempts": self.max_rewriting_attempts,
                "fallback_strategy": self.fallback_strategy,
            }
            self.config_hash = compute_sha256(payload)


class AgentConfigVersion(BaseModel):
    """Composite snapshot encapsulating all component versions for an Agent execution."""

    agent_version: str = Field(default="1.0.0", description="Overall Agent system version")
    name: str = Field(default="agent_core", description="Logical agent configuration name")
    prompt_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of prompt component name to its exact active version string",
    )
    model_config_version: str = Field(default="1.0.0", description="Active ModelConfigVersion")
    retrieval_config_version: str = Field(default="1.0.0", description="Active RetrievalConfigVersion")
    routing_config_version: str = Field(default="1.0.0", description="Active RoutingConfigVersion")
    tools_registered: list[str] = Field(
        default_factory=list,
        description="List of registered tools active during this execution",
    )
    composite_hash: str = Field(
        default="",
        description="Comprehensive SHA-256 fingerprint binding all component configurations",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.composite_hash:
            payload = {
                "agent_version": self.agent_version,
                "name": self.name,
                "prompt_versions": self.prompt_versions,
                "model_config_version": self.model_config_version,
                "retrieval_config_version": self.retrieval_config_version,
                "routing_config_version": self.routing_config_version,
                "tools_registered": sorted(self.tools_registered),
            }
            self.composite_hash = compute_sha256(payload)


class ConfigurationProvenance(BaseModel):
    """Audit record capturing the full provenance of an Agent execution."""

    composite_hash: str = Field(description="SHA-256 composite fingerprint of all settings")
    agent_version: str = Field(description="Agent configuration version")
    prompt_versions: dict[str, str] = Field(description="Prompt version map")
    model_config_version: str = Field(description="Model configuration version")
    retrieval_config_version: str = Field(description="Retrieval configuration version")
    routing_config_version: str = Field(description="Routing configuration version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Execution snapshot timestamp",
    )
