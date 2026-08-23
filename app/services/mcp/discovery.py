from app.core.logging import get_logger
from app.services.agent.tools.registry import ToolRegistry
from app.services.mcp.adapter import MCPToolAdapter
from app.services.mcp.client import MCPClient
from app.services.mcp.models import MCPToolDefinition
from app.services.mcp.safety import MCPInvocationTracker, MCPSafetyPolicy
from app.services.mcp.server import LocalMCPServer

logger = get_logger("app.services.mcp.discovery")


class MCPDiscoveryManager:
    """Manages dynamic discovery, capability inspection, and safety filtering across MCP servers."""

    def __init__(
        self,
        safety_policy: MCPSafetyPolicy | None = None,
        tracker: MCPInvocationTracker | None = None,
    ) -> None:
        self._policy = safety_policy or MCPSafetyPolicy()
        self._tracker = tracker or MCPInvocationTracker(
            max_invocations=self._policy.max_invocations
        )
        self._clients: dict[str, MCPClient] = {}

    @property
    def safety_policy(self) -> MCPSafetyPolicy:
        """Active safety governance policy."""
        return self._policy

    @property
    def tracker(self) -> MCPInvocationTracker:
        """Active invocation tracker."""
        return self._tracker

    def register_server(self, server_or_client: LocalMCPServer | MCPClient) -> MCPClient:
        """Register an MCP server or client for capability discovery."""
        if isinstance(server_or_client, LocalMCPServer):
            client = MCPClient(server=server_or_client)
        else:
            client = server_or_client

        self._clients[client.server_name.lower()] = client
        logger.info("mcp_server_registered_for_discovery", server=client.server_name)
        return client

    async def discover_capabilities(
        self, server_name: str | None = None
    ) -> list[tuple[MCPClient, MCPToolDefinition]]:
        """Query MCP servers dynamically and return tools passing safety policy filters."""
        discovered: list[tuple[MCPClient, MCPToolDefinition]] = []

        target_clients = (
            [self._clients[server_name.lower()]]
            if server_name and server_name.lower() in self._clients
            else list(self._clients.values())
        )

        for client in target_clients:
            # 1. Server Whitelist Validation
            if not self._policy.is_server_allowed(client.server_name):
                logger.warning(
                    "mcp_server_discovery_skipped_policy",
                    server=client.server_name,
                    reason="Server not in allowed_servers whitelist",
                )
                continue

            try:
                tools = await client.list_tools()
                for tool in tools:
                    # 2. Tool Whitelist Validation
                    if not self._policy.is_tool_allowed(tool.name):
                        logger.info(
                            "mcp_tool_filtered_out_by_policy",
                            server=client.server_name,
                            tool=tool.name,
                        )
                        continue

                    discovered.append((client, tool))
                    logger.debug(
                        "mcp_capability_discovered",
                        server=client.server_name,
                        tool=tool.name,
                    )
            except Exception as exc:
                logger.error(
                    "mcp_server_discovery_failed",
                    server=client.server_name,
                    error=str(exc),
                )

        return discovered

    async def sync_to_tool_registry(self, tool_registry: ToolRegistry) -> list[str]:
        """Discover authorized MCP tools dynamically and mount them into the Agent ToolRegistry."""
        discovered = await self.discover_capabilities()
        registered_tool_names: list[str] = []

        for client, tool_def in discovered:
            adapter = MCPToolAdapter(
                mcp_client=client,
                tool_definition=tool_def,
                safety_policy=self._policy,
                tracker=self._tracker,
            )
            tool_registry.register(adapter)
            registered_tool_names.append(tool_def.name)

        logger.info(
            "mcp_capabilities_synced_to_registry",
            count=len(registered_tool_names),
            tools=registered_tool_names,
        )
        return registered_tool_names
