import types
from contextlib import AsyncExitStack
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch, call

from avalan.tool.browser import BrowserTool, BrowserToolSet


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
        self.tool = BrowserTool(self.client)

    async def asyncTearDown(self):
        self.mark_patch.stop()

    async def test_call_and_exit(self):
        result1 = await self.tool("http://a")
        result2 = await self.tool("http://b")

        self.assertEqual(result1, "parsed")
        self.assertEqual(result2, "parsed")
        self.browser_type.launch.assert_awaited_once()
        self.browser.new_page.assert_awaited_once()
        self.page.goto.assert_has_awaits([call("http://a"), call("http://b")])
        self.page.content.assert_awaited()

        await self.tool.__aexit__(None, None, None)

        self.page.close.assert_awaited_once()
        self.browser.close.assert_awaited_once()


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
            toolset = BrowserToolSet(exit_stack=dummy_stack, namespace="b")
            self.assertEqual(toolset.namespace, "b")
            self.assertEqual(toolset._client, "client1")
            self.assertEqual(toolset.tools, [dummy_tool])

            result = await toolset.__aenter__()

        self.assertIs(result, toolset)
        dummy_stack.enter_async_context.assert_awaited_once_with("client1")
        dummy_tool.with_client.assert_called_once_with("client2")
