import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolFilter,
    ToolManagerSettings,
)
from avalan.tool.manager import ToolManager
from avalan.tool.mcp import McpCallTool, McpToolSet


class McpCallToolTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addCleanup(patch.stopall)
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.call_tool.return_value = ["result"]
        self.client = client
        self.Client = MagicMock(return_value=client)
        mcp_mod = ModuleType("mcp")
        mcp_mod.Client = self.Client
        patch.dict(sys.modules, {"mcp": mcp_mod}).start()
        self.tool = McpCallTool()

    async def test_call_with_arguments(self):
        context = ToolCallContext()
        result = await self.tool(
            "http://host", "calc", {"a": 1}, context=context
        )
        self.assertEqual(result, ["result"])
        self.Client.assert_called_once_with("http://host")
        self.client.call_tool.assert_awaited_once_with("calc", {"a": 1})

    async def test_call_without_arguments(self):
        context = ToolCallContext()
        await self.tool("http://host", "calc", None, context=context)
        self.client.call_tool.assert_awaited_once_with("calc", {})

    async def test_passes_client_and_call_params(self):
        tool = McpCallTool(
            client_params={"token": "x"}, call_params={"timeout": 1}
        )
        context = ToolCallContext()
        await tool("http://host", "calc", None, context=context)
        self.Client.assert_called_once_with("http://host", token="x")
        self.client.call_tool.assert_awaited_once_with("calc", {}, timeout=1)

    async def test_manager_filters_match_mcp_tool_not_remote_name(self):
        filtered_names: list[str] = []

        def filter_call(call: ToolCall, context: ToolCallContext):
            filtered_names.append(call.name)
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={
                        "uri": "http://host",
                        "name": "blocked",
                        "arguments": {},
                    },
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["mcp.call"],
            available_toolsets=[McpToolSet()],
            settings=ToolManagerSettings(
                filters=[ToolFilter(func=filter_call, namespace="math")]
            ),
        )
        call = ToolCall(
            id="call-1",
            name="mcp.call",
            arguments={
                "uri": "http://host",
                "name": "math.calculator",
                "arguments": {"a": 1},
            },
        )

        result = await manager(call, context=ToolCallContext())

        self.assertEqual(result.result, ["result"])
        self.assertEqual(filtered_names, [])
        self.client.call_tool.assert_awaited_once_with(
            "math.calculator", {"a": 1}
        )


class McpToolSetTestCase(TestCase):
    def test_default_namespace(self):
        toolset = McpToolSet()
        self.assertEqual(toolset.namespace, "mcp")
        self.assertEqual(len(toolset.tools), 1)
        self.assertEqual(toolset.tools[0].__name__, "call")
