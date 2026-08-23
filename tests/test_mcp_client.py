import pytest
from app.config import Settings
from app.models.schemas import TaskStatus
from app.services.agent.agent import BasicAgent
from app.services.agent.tools.app_info import AppInfoTool
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService
from app.services.mcp import (
    LocalMCPServer,
    MCPClient,
    MCPToolAdapter,
    MCPToolDefinition,
)


@pytest.mark.asyncio
async def test_mcp_server_initialize_and_list_tools() -> None:
    """Verify MCP Server JSON-RPC handshake and tool discovery."""
    server = LocalMCPServer(server_name="test-math-server", version="1.0.0")
    server.register_tool(CalculatorTool())

    client = MCPClient(server=server)

    init_res = await client.initialize()
    assert init_res["serverInfo"]["name"] == "test-math-server"
    assert init_res["protocolVersion"] == "2024-11-05"

    tools = await client.list_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], MCPToolDefinition)
    assert tools[0].name == "calculator"
    assert "properties" in tools[0].inputSchema


@pytest.mark.asyncio
async def test_mcp_client_call_tool() -> None:
    """Verify direct tool invocation over MCP client via JSON-RPC."""
    server = LocalMCPServer(server_name="test-math-server")
    server.register_tool(CalculatorTool())

    client = MCPClient(server=server)
    res = await client.call_tool(
        tool_name="calculator",
        arguments={"expression": "15 * 6"},
    )

    assert not res.isError
    assert len(res.content) == 1
    assert res.content[0].text == "90"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_integration() -> None:
    """Verify MCPToolAdapter wraps an MCP tool into the native BaseTool interface."""
    server = LocalMCPServer(server_name="test-math-server")
    server.register_tool(CalculatorTool())

    client = MCPClient(server=server)
    tool_defs = await client.list_tools()
    adapter = MCPToolAdapter(mcp_client=client, tool_definition=tool_defs[0])

    assert adapter.name == "calculator"
    assert "mathematical" in adapter.description.lower()
    assert "expression" in adapter.parameters_schema.get("properties", {})

    exec_res = adapter.execute(expression="250 / 5")
    assert exec_res.success is True
    assert exec_res.output in ("50", "50.0")


@pytest.mark.asyncio
async def test_agent_invokes_mcp_tool_end_to_end(settings: Settings) -> None:
    """Verify BasicAgent invokes migrated calculator capability hosted on MCP Server."""
    # 1. Start Local MCP Server and register migrated Calculator tool
    server = LocalMCPServer(server_name="agent-mcp-server")
    server.register_tool(CalculatorTool())

    # 2. Connect MCP Client
    client = MCPClient(server=server)
    tool_defs = await client.list_tools()

    # 3. Create ToolRegistry with Internal Tool (AppInfo) and Migrated MCP Tool (Calculator)
    registry = ToolRegistry(
        tools=[
            AppInfoTool(settings=settings),
            MCPToolAdapter(mcp_client=client, tool_definition=tool_defs[0]),
        ],
        settings=settings,
    )

    # 4. Instantiate Agent and execute task requiring calculator
    llm_service = LLMService(settings=settings)
    agent = BasicAgent(llm_service=llm_service, tool_registry=registry)

    task = "Please calculate 480 / 12 using the calculator tool."
    state = await agent.run(task=task)

    assert state.status == TaskStatus.COMPLETED
    assert "calculator" in state.tools_used
    assert state.answer is not None
    assert state.error is None


@pytest.mark.asyncio
async def test_internal_and_mcp_tool_coexistence(settings: Settings) -> None:
    """Verify internal native tools and MCP tools both function in the same registry."""
    server = LocalMCPServer(server_name="coexistence-server")
    server.register_tool(CalculatorTool())

    client = MCPClient(server=server)
    tool_defs = await client.list_tools()

    registry = ToolRegistry(
        tools=[
            AppInfoTool(settings=settings),
            MCPToolAdapter(mcp_client=client, tool_definition=tool_defs[0]),
        ],
        settings=settings,
    )

    # 1. Call internal tool
    internal_res = registry.execute("app_info")
    assert internal_res.success is True
    assert "app_name" in internal_res.output

    # 2. Call MCP tool
    mcp_res = registry.execute("calculator", expression="100 - 37")
    assert mcp_res.success is True
    assert mcp_res.output == "63"


@pytest.mark.asyncio
async def test_mcp_client_unknown_tool_error() -> None:
    """Verify MCP client handles unknown tool calls gracefully with error outcome."""
    server = LocalMCPServer(server_name="test-server")
    client = MCPClient(server=server)

    res = await client.call_tool(
        tool_name="non_existent_tool",
        arguments={},
    )
    assert res.isError is True
