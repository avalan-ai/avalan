from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from contextlib import AsyncExitStack


class McpCallTool(Tool):
    """Call an MCP server tool using the MCP client.

    Args:
        uri: Base URI of the MCP server.
        name: Name of the tool to invoke.
        arguments: Arguments to send to the tool.
    """

    _client_params: dict[str, object]
    _call_params: dict[str, object]

    def __init__(
        self,
        *,
        client_params: dict[str, object] | None = None,
        call_params: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.__name__ = "call"
        self._client_params = client_params or {}
        self._call_params = call_params or {}

    async def __call__(
        self,
        uri: str,
        name: str,
        arguments: dict[str, object] | None,
        *,
        context: ToolCallContext,
    ) -> list[object]:
        from mcp import Client

        assert uri
        assert name

        async with Client(uri, **self._client_params) as client:
            return await client.call_tool(
                name, arguments or {}, **self._call_params
            )


class McpToolSet(ToolSet):
    """Tool set providing MCP client functionality."""

    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = "mcp",
    ) -> None:
        tools = [McpCallTool()]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
