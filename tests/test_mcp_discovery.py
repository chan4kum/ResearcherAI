import time
from typing import Any

import pytest
from app.config import Settings
from app.models.schemas import TaskStatus
from app.services.agent.agent import BasicAgent
from app.services.agent.tools.app_info import AppInfoTool
from app.services.agent.tools.base import BaseTool, ToolResult
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService
from app.services.mcp import (
    LocalMCPServer,
    MCPDiscoveryManager,
    MCPInvocationTracker,
    MCPSafetyPolicy,
)


class SlowMockTool(BaseTool):
    """Mock tool that intentionally sleeps to test execution timeout guardrails."""

    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "A mock tool that simulates a long-running execution."

    def execute(self, **kwargs: Any) -> ToolResult:
        time.sleep(1.0)
        return ToolResult(tool_name=self.name, success=True, output="Finished")


class RestrictedMockTool(BaseTool):
    """Mock tool testing permission restrictions."""

    @property
    def name(self) -> str:
        return "restricted_tool"

    @property
    def description(self) -> str:
        return "Restricted administrative tool."

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output="Admin operation")


@pytest.mark.asyncio
async def test_dynamic_discovery_across_multiple_servers(settings: Settings) -> None:
    """Verify dynamic discovery across multiple distinct MCP servers."""
    server1 = LocalMCPServer(server_name="math-server")
    server1.register_tool(CalculatorTool())

    server2 = LocalMCPServer(server_name="info-server")
    server2.register_tool(AppInfoTool(settings=settings))

    policy = MCPSafetyPolicy(
        allowed_servers=["math-server", "info-server"],
        allowed_tools=["calculator", "app_info"],
    )
    discovery = MCPDiscoveryManager(safety_policy=policy)
    discovery.register_server(server1)
    discovery.register_server(server2)

    discovered = await discovery.discover_capabilities()
    assert len(discovered) == 2

    discovered_names = [tool.name for _, tool in discovered]
    assert "calculator" in discovered_names
    assert "app_info" in discovered_names


@pytest.mark.asyncio
async def test_server_whitelist_enforcement(settings: Settings) -> None:
    """Verify discovery filters out unapproved MCP servers."""
    trusted_server = LocalMCPServer(server_name="trusted-server")
    trusted_server.register_tool(CalculatorTool())

    untrusted_server = LocalMCPServer(server_name="untrusted-server")
    untrusted_server.register_tool(AppInfoTool(settings=settings))

    policy = MCPSafetyPolicy(
        allowed_servers=["trusted-server"],
        allowed_tools=["*"],
        enforce_whitelist=True,
    )
    discovery = MCPDiscoveryManager(safety_policy=policy)
    discovery.register_server(trusted_server)
    discovery.register_server(untrusted_server)

    discovered = await discovery.discover_capabilities()
    assert len(discovered) == 1
    assert discovered[0][1].name == "calculator"


@pytest.mark.asyncio
async def test_tool_whitelist_enforcement() -> None:
    """Verify discovery filters out unauthorized tools even on approved servers."""
    server = LocalMCPServer(server_name="mixed-server")
    server.register_tool(CalculatorTool())
    server.register_tool(RestrictedMockTool())

    policy = MCPSafetyPolicy(
        allowed_servers=["mixed-server"],
        allowed_tools=["calculator"],  # RestrictedMockTool not allowed
        enforce_whitelist=True,
    )
    discovery = MCPDiscoveryManager(safety_policy=policy)
    discovery.register_server(server)

    discovered = await discovery.discover_capabilities()
    assert len(discovered) == 1
    assert discovered[0][1].name == "calculator"


@pytest.mark.asyncio
async def test_invocation_limit_enforcement() -> None:
    """Verify execution is blocked when tool invocation limit is exceeded."""
    server = LocalMCPServer(server_name="calc-server")
    server.register_tool(CalculatorTool())

    policy = MCPSafetyPolicy(
        allowed_servers=["calc-server"],
        allowed_tools=["calculator"],
        max_invocations=2,
    )
    tracker = MCPInvocationTracker(max_invocations=2)
    discovery = MCPDiscoveryManager(safety_policy=policy, tracker=tracker)
    discovery.register_server(server)

    registry = ToolRegistry(tools=[])
    await discovery.sync_to_tool_registry(registry)

    # 1. First invocation -> success
    res1 = registry.execute("calculator", expression="2 + 2")
    assert res1.success is True
    assert res1.output == "4"

    # 2. Second invocation -> success
    res2 = registry.execute("calculator", expression="5 * 5")
    assert res2.success is True
    assert res2.output == "25"

    # 3. Third invocation -> blocked by invocation cap
    res3 = registry.execute("calculator", expression="10 / 2")
    assert res3.success is False
    assert "invocation limit exceeded" in res3.error.lower()


@pytest.mark.asyncio
async def test_timeout_enforcement() -> None:
    """Verify long-running or hanging MCP tools are aborted by timeout guardrails."""
    server = LocalMCPServer(server_name="slow-server")
    server.register_tool(SlowMockTool())

    policy = MCPSafetyPolicy(
        allowed_servers=["slow-server"],
        allowed_tools=["slow_tool"],
        timeout_seconds=0.1,  # Short timeout
    )
    discovery = MCPDiscoveryManager(safety_policy=policy)
    discovery.register_server(server)

    registry = ToolRegistry(tools=[])
    await discovery.sync_to_tool_registry(registry)

    res = registry.execute("slow_tool")
    assert res.success is False
    assert "timed out" in res.error.lower()


@pytest.mark.asyncio
async def test_agent_dynamic_tool_discovery_and_selection(settings: Settings) -> None:
    """Verify BasicAgent dynamically selects and invokes discovered MCP tools."""
    server = LocalMCPServer(server_name="dynamic-agent-server")
    server.register_tool(CalculatorTool())

    policy = MCPSafetyPolicy(
        allowed_servers=["dynamic-agent-server"],
        allowed_tools=["calculator"],
        timeout_seconds=10.0,
        max_invocations=10,
    )
    discovery = MCPDiscoveryManager(safety_policy=policy)
    discovery.register_server(server)

    registry = ToolRegistry(tools=[AppInfoTool(settings=settings)], settings=settings)
    synced_tools = await discovery.sync_to_tool_registry(registry)
    assert "calculator" in synced_tools

    llm_service = LLMService(settings=settings)
    agent = BasicAgent(llm_service=llm_service, tool_registry=registry)

    task = "Please compute 360 / 6 using the calculator."
    state = await agent.run(task=task)

    assert state.status == TaskStatus.COMPLETED
    assert "calculator" in state.tools_used
    assert state.answer is not None
    assert state.error is None
