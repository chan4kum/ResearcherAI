from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_agent_service
from app.models.schemas import TaskRequest, TaskResponse
from app.services.agent.service import AgentService

router = APIRouter()


@router.post(
    "/tasks",
    response_model=TaskResponse,
    summary="Submit an agent task",
    description=(
        "Submit a goal or task to the AI agent. The agent formulates a plan, "
        "executes it, and returns structured results and telemetry."
    ),
)
async def create_task(
    payload: TaskRequest,
    request: Request,
    agent_service: AgentService = Depends(get_agent_service),
) -> TaskResponse:
    """Receive a task, determine plan of action, execute solution, and return structured results."""
    request_id = getattr(request.state, "request_id", None)
    return await agent_service.execute_task(
        task=payload.task,
        task_id=request_id,
        system_instructions=payload.system_instructions,
    )
