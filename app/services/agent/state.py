import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.models.schemas import ExecutionMetadata, TaskResponse, TaskStatus


class AgentState(BaseModel):
    """Encapsulates the lifecycle, execution progress, and telemetry of an Agent task."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    status: TaskStatus = TaskStatus.PENDING
    plan: list[str] = Field(default_factory=list)
    answer: str | None = None
    error: str | None = None

    # Telemetry and metadata
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tools_used: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0

    # Milestone 50: Configuration and prompt version provenance
    config_version: str = "1.0.0"
    composite_config_hash: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    model_config_version: str = "1.0.0"
    retrieval_config_version: str = "1.0.0"
    routing_config_version: str = "1.0.0"

    def add_trace(self, step: str) -> None:
        """Append an execution stage marker to the trace."""
        self.trace.append(step)
        self.updated_at = datetime.now(UTC)

    def record_usage(
        self,
        model: str,
        provider: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Accumulate token metrics and record active model/provider."""
        self.model = model
        self.provider = provider
        if prompt_tokens:
            self.prompt_tokens += prompt_tokens
        if completion_tokens:
            self.completion_tokens += completion_tokens
        if total_tokens:
            self.total_tokens += total_tokens

    def to_response(self) -> TaskResponse:
        """Convert the internal AgentState into the external TaskResponse contract."""
        return TaskResponse(
            task_id=self.task_id,
            task=self.task,
            status=self.status,
            plan=self.plan,
            answer=self.answer,
            error=self.error,
            metadata=ExecutionMetadata(
                model=self.model,
                provider=self.provider,
                duration_ms=self.duration_ms,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens,
                tools_used=self.tools_used,
                trace=self.trace,
                config_version=self.config_version,
                composite_config_hash=self.composite_config_hash,
                prompt_versions=self.prompt_versions,
                model_config_version=self.model_config_version,
                retrieval_config_version=self.retrieval_config_version,
                routing_config_version=self.routing_config_version,
            ),
        )
