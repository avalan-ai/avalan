from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from avalan.entities import ToolCallContext
from avalan.tool import Tool


class DummyTool(Tool):
    async def __call__(self, *, context: ToolCallContext) -> str:
        return "ok"


class ToolContextTest(IsolatedAsyncioTestCase):
    async def test_aenter_returns_self(self):
        tool = DummyTool()
        result = await tool.__aenter__()
        self.assertIs(result, tool)

    async def test_aexit_delegates_stack(self):
        tool = DummyTool()
        stack = AsyncMock(spec=AsyncExitStack)
        tool._exit_stack = stack
        stack.__aexit__.return_value = False
        result = await tool.__aexit__(None, None, None)
        stack.__aexit__.assert_awaited_once_with(None, None, None)
        self.assertFalse(result)

    async def test_aexit_without_stack(self):
        tool = DummyTool()
        tool._exit_stack = None
        result = await tool.__aexit__(None, None, None)
        self.assertTrue(result)
