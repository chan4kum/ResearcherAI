from app.config import Settings, get_settings
from app.core.versioning.manager import (
    ConfigurationVersionManager,
    get_version_manager,
)
from app.models.schemas import TaskResponse
from app.services.agent.agent import BasicAgent
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService


class AgentService:
    """Domain service managing Agent execution and task lifecycle."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        tool_registry: ToolRegistry | None = None,
        version_manager: ConfigurationVersionManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._tool_registry = tool_registry or ToolRegistry(settings=self._settings)
        self._version_manager = version_manager or get_version_manager(settings=self._settings)
        self._agent = BasicAgent(
            llm_service=self._llm_service,
            tool_registry=self._tool_registry,
            version_manager=self._version_manager,
        )

    @property
    def agent(self) -> BasicAgent:
        return self._agent

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def version_manager(self) -> ConfigurationVersionManager:
        return self._version_manager

    async def execute_task(
        self,
        task: str,
        task_id: str | None = None,
        system_instructions: str | None = None,
    ) -> TaskResponse:
        """Run the agent on a user task and return a structured TaskResponse."""
        state = await self._agent.run(
            task=task,
            task_id=task_id,
            system_instructions=system_instructions,
        )
        return state.to_response()
