from avalan.entities import ToolCallContext
from avalan.tool.browser import BrowserTool, BrowserToolSet, BrowserToolSettings
from contextlib import AsyncExitStack
import types
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch, call


class BrowserToolCallTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.page = MagicMock()
        self.page.goto = AsyncMock()
        self.page.content = AsyncMock(return_value="<html>hi</html>")
        self.page.close = AsyncMock()

        self.browser = MagicMock()
        self.browser.new_page = AsyncMock(return_value=self.page)
        self.browser.close = AsyncMock()

        self.browser_type = MagicMock()
        self.browser_type.launch = AsyncMock(return_value=self.browser)

        self.client = MagicMock()
        self.client.firefox = self.browser_type

        result = types.SimpleNamespace(text_content="parsed")
        converter = MagicMock(return_value=result)

        self.mark_patch = patch(
            "avalan.tool.browser.MarkItDown",
            return_value=MagicMock(convert=converter),
        )
        self.mark_patch.start()
        self.tool = BrowserTool(BrowserToolSettings(), self.client)

    async def asyncTearDown(self):
        self.mark_patch.stop()


class BrowserToolSetTestCase(IsolatedAsyncioTestCase):
    async def test_init_and_enter(self):
        dummy_stack = AsyncMock(spec=AsyncExitStack)
        dummy_stack.enter_async_context = AsyncMock(return_value="client2")
        dummy_stack.__aexit__ = AsyncMock(return_value=False)

        dummy_tool = MagicMock()
        dummy_tool.with_client = MagicMock(return_value=dummy_tool)

        async def dummy_aenter(self):
            return self

        with (
            patch(
                "avalan.tool.browser.async_playwright", return_value="client1"
            ),
            patch("avalan.tool.browser.BrowserTool", return_value=dummy_tool),
            patch("avalan.tool.browser.ToolSet.__aenter__", dummy_aenter),
        ):
            toolset = BrowserToolSet(settings=BrowserToolSettings(), exit_stack=dummy_stack, namespace="b")
            self.assertEqual(toolset.namespace, "b")
            self.assertEqual(toolset._client, "client1")
            self.assertEqual(toolset.tools, [dummy_tool])

            result = await toolset.__aenter__()

        self.assertIs(result, toolset)
        dummy_stack.enter_async_context.assert_awaited_once_with("client1")
        dummy_tool.with_client.assert_called_once_with("client2")


class BrowserToolWithClientTestCase(TestCase):
    def test_with_client(self):
        client1 = MagicMock()
        client2 = MagicMock()
        tool = BrowserTool(BrowserToolSettings(), client1)
        result = tool.with_client(client2)
        self.assertIs(result, tool)
        self.assertIs(tool._client, client2)
