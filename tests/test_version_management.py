"""
tests/test_version_management.py — Unit and integration tests for Model/Prompt Version Management (Milestone 50)
"""

import pytest

from app.config import Settings
from app.core.versioning.manager import (
    ConfigurationVersionManager,
    get_version_manager,
)
from app.core.versioning.models import (
    AgentConfigVersion,
    ConfigurationProvenance,
    ModelConfigVersion,
    PromptVersion,
    RetrievalConfigVersion,
    RoutingConfigVersion,
    compute_sha256,
)
from app.core.versioning.prompts import (
    PromptNotFoundError,
    PromptRegistry,
    get_prompt_registry,
)
from app.models.schemas import TaskStatus
from app.services.agent.agent import BasicAgent
from app.services.agent.service import AgentService
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService


# ---------------------------------------------------------------------------
# 1. PromptVersion & PromptRegistry Unit Tests
# ---------------------------------------------------------------------------


class TestPromptRegistry:
    def test_prompt_version_computes_sha256_hash(self):
        prompt = PromptVersion(
            name="test_prompt",
            version="1.0.0",
            template="You are a helpful assistant.",
            description="Test prompt",
        )
        assert prompt.prompt_hash != ""
        assert len(prompt.prompt_hash) == 64
        # Hash is deterministic
        assert prompt.prompt_hash == compute_sha256("You are a helpful assistant.")

    def test_prompt_registry_canonical_prompts_loaded(self):
        registry = PromptRegistry()
        active = registry.get_active_versions()
        assert "planner" in active
        assert "tool_decider" in active
        assert "executor" in active
        assert "rag_synthesizer" in active
        assert "query_rewriter" in active
        assert "hyde_generator" in active

    def test_prompt_registry_get_default_and_specific_version(self):
        registry = PromptRegistry()
        # Default planner is 1.0.0
        p_default = registry.get("planner")
        assert p_default.version == "1.0.0"

        # Explicit version 1.1.0
        p_v110 = registry.get("planner", version="1.1.0")
        assert p_v110.version == "1.1.0"
        assert p_default.prompt_hash != p_v110.prompt_hash

    def test_prompt_registry_not_found_raises_error(self):
        registry = PromptRegistry()
        with pytest.raises(PromptNotFoundError, match="Prompt 'nonexistent' is not registered"):
            registry.get("nonexistent")

        with pytest.raises(PromptNotFoundError, match="Version '9.9.9' for prompt 'planner' not found"):
            registry.get("planner", version="9.9.9")

    def test_prompt_registry_custom_registration_and_switch_default(self):
        registry = PromptRegistry()
        custom_prompt = PromptVersion(
            name="custom_planner",
            version="2.0.0",
            template="Custom Planner Template: {guidance}",
        )
        registry.register(custom_prompt, set_as_default=True)
        assert registry.get("custom_planner").version == "2.0.0"

        # Register v2.1.0 without setting as default
        custom_v21 = PromptVersion(
            name="custom_planner",
            version="2.1.0",
            template="Custom Planner v2.1: {guidance}",
        )
        registry.register(custom_v21, set_as_default=False)
        assert registry.get("custom_planner").version == "2.0.0"

        # Explicitly switch default
        registry.set_default_version("custom_planner", "2.1.0")
        assert registry.get("custom_planner").version == "2.1.0"

    def test_list_prompts_filtering(self):
        registry = PromptRegistry()
        all_prompts = registry.list_prompts()
        assert len(all_prompts) >= 6

        planner_prompts = registry.list_prompts(name="planner")
        assert len(planner_prompts) >= 2
        assert all(p.name == "planner" for p in planner_prompts)

        empty = registry.list_prompts(name="unknown_prompt")
        assert empty == []


# ---------------------------------------------------------------------------
# 2. Configuration Models and Cryptographic Fingerprints
# ---------------------------------------------------------------------------


class TestConfigurationModels:
    def test_model_config_version_hash(self):
        m1 = ModelConfigVersion(provider="mock", model="gpt-4o-mini", temperature=0.7)
        m2 = ModelConfigVersion(provider="mock", model="gpt-4o-mini", temperature=0.7)
        m3 = ModelConfigVersion(provider="openai", model="gpt-4o", temperature=0.0)

        assert m1.config_hash == m2.config_hash
        assert m1.config_hash != m3.config_hash

    def test_retrieval_config_version_hash(self):
        r1 = RetrievalConfigVersion(retrieval_mode="hybrid", top_k=5)
        r2 = RetrievalConfigVersion(retrieval_mode="vector", top_k=10)

        assert r1.config_hash != r2.config_hash
        assert len(r1.config_hash) == 64

    def test_routing_config_version_hash(self):
        rt1 = RoutingConfigVersion(default_strategy="normal", enable_hyde=False)
        rt2 = RoutingConfigVersion(default_strategy="hyde", enable_hyde=True)

        assert rt1.config_hash != rt2.config_hash

    def test_agent_config_version_composite_hash(self):
        agent_cfg = AgentConfigVersion(
            agent_version="1.0.0",
            name="agent_core",
            prompt_versions={"planner": "1.0.0", "tool_decider": "1.0.0"},
            model_config_version="1.0.0",
            retrieval_config_version="1.0.0",
            routing_config_version="1.0.0",
            tools_registered=["calculator", "app_info"],
        )
        assert agent_cfg.composite_hash != ""
        assert len(agent_cfg.composite_hash) == 64


# ---------------------------------------------------------------------------
# 3. ConfigurationVersionManager Tests
# ---------------------------------------------------------------------------


class TestConfigurationVersionManager:
    def test_manager_initialization_from_settings(self):
        settings = Settings(llm_provider="mock", llm_model="gpt-4o-mini")
        manager = ConfigurationVersionManager(settings=settings)

        assert manager.model_config.provider == "mock"
        assert manager.model_config.model == "gpt-4o-mini"
        assert manager.retrieval_config.version == "1.0.0"
        assert manager.routing_config.version == "1.0.0"

    def test_create_provenance_record(self):
        manager = ConfigurationVersionManager()
        provenance = manager.create_provenance(tools_registered=["calculator"])

        assert isinstance(provenance, ConfigurationProvenance)
        assert provenance.agent_version == "1.0.0"
        assert "planner" in provenance.prompt_versions
        assert len(provenance.composite_hash) == 64

    def test_get_snapshot_summary(self):
        manager = ConfigurationVersionManager()
        summary = manager.get_snapshot_summary()

        assert "agent_version" in summary
        assert "composite_hash" in summary
        assert "prompt_versions" in summary
        assert "model" in summary
        assert "retrieval" in summary
        assert "routing" in summary


# ---------------------------------------------------------------------------
# 4. End-to-End Agent Execution Traceability Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentExecutionTraceability:
    async def test_agent_execution_records_provenance_in_state_and_response(self):
        llm = LLMService(provider=MockLLMProvider(model_name="mock-v1"))
        tool_reg = ToolRegistry()
        version_mgr = ConfigurationVersionManager()

        agent = BasicAgent(
            llm_service=llm,
            tool_registry=tool_reg,
            version_manager=version_mgr,
        )

        state = await agent.run(task="Calculate 42 plus 8")
        assert state.status == TaskStatus.COMPLETED

        # Check state provenance
        assert state.config_version == "1.0.0"
        assert state.composite_config_hash != ""
        assert "planner" in state.prompt_versions
        assert "tool_decider" in state.prompt_versions
        assert "executor" in state.prompt_versions
        assert state.model_config_version == "1.0.0"
        assert state.retrieval_config_version == "1.0.0"
        assert state.routing_config_version == "1.0.0"

        # Check TaskResponse serialization contract
        response = state.to_response()
        assert response.metadata.config_version == "1.0.0"
        assert response.metadata.composite_config_hash == state.composite_config_hash
        assert response.metadata.prompt_versions == state.prompt_versions
        assert response.metadata.model_config_version == "1.0.0"
        assert response.metadata.retrieval_config_version == "1.0.0"
        assert response.metadata.routing_config_version == "1.0.0"

    async def test_agent_service_propagates_version_provenance(self):
        llm = LLMService(provider=MockLLMProvider(model_name="mock-v1"))
        service = AgentService(llm_service=llm)

        response = await service.execute_task(task="Explain recursion in programming.")
        assert response.status == TaskStatus.COMPLETED
        assert response.metadata.config_version == "1.0.0"
        assert response.metadata.composite_config_hash != ""
        assert len(response.metadata.prompt_versions) >= 3

    async def test_dynamic_prompt_version_override(self):
        llm = LLMService(provider=MockLLMProvider(model_name="mock-v1"))
        registry = PromptRegistry()
        registry.set_default_version("planner", "1.1.0")

        version_mgr = ConfigurationVersionManager(prompt_registry=registry)
        agent = BasicAgent(
            llm_service=llm,
            version_manager=version_mgr,
        )

        state = await agent.run(task="Design a resilient architecture")
        assert state.prompt_versions["planner"] == "1.1.0"
        response = state.to_response()
        assert response.metadata.prompt_versions["planner"] == "1.1.0"

    async def test_composite_hash_sensitive_to_model_config_update(self):
        version_mgr = ConfigurationVersionManager()
        prov1 = version_mgr.create_provenance()

        # Update model config
        version_mgr.set_model_config(
            ModelConfigVersion(version="2.0.0", model="claude-3-5-sonnet", provider="anthropic")
        )
        prov2 = version_mgr.create_provenance()

        assert prov1.composite_hash != prov2.composite_hash
        assert prov2.model_config_version == "2.0.0"


# ---------------------------------------------------------------------------
# 5. RAG Execution Provenance Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRAGExecutionTraceability:
    async def test_rag_service_records_version_provenance(self):
        from app.db.repository import InMemoryVectorRepository
        from app.services.embedding.service import EmbeddingService
        from app.services.rag.retriever import VectorRetriever
        from app.services.rag.service import RAGService

        repo = InMemoryVectorRepository()
        emb_service = EmbeddingService()
        retriever = VectorRetriever(embedding_service=emb_service, vector_repository=repo)
        llm = LLMService(provider=MockLLMProvider(model_name="mock-v1"))
        rag_service = RAGService(retriever=retriever, llm_service=llm)

        response = await rag_service.answer(question="What is the system architecture?")
        assert response.answer is not None
        assert "version_provenance" in response.metadata
        provenance = response.metadata["version_provenance"]
        assert provenance["prompt_version"] == "1.0.0"
        assert provenance["retrieval_config_version"] == "1.0.0"
        assert provenance["routing_config_version"] == "1.0.0"
        assert provenance["model_config_version"] == "1.0.0"
        assert provenance["config_hash"] != ""
