from avalan.entities import (
    Message,
    MessageRole,
    TextPartition,
    ToolCallContext,
)
from avalan.tool.browser import (
    BrowserTool,
    BrowserToolSet,
    BrowserToolSettings,
)
from contextlib import AsyncExitStack
import types
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from io import StringIO
import numpy as np


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
            toolset = BrowserToolSet(
                settings=BrowserToolSettings(),
                exit_stack=dummy_stack,
                namespace="b",
            )
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


class BrowserToolCallSearchTestCase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = MagicMock()
        self.mark_patch = patch("avalan.tool.browser.MarkItDown")
        self.mark_patch.start()

    async def asyncTearDown(self):
        self.mark_patch.stop()

    async def test_call_search_with_partitioner(self):
        partitions = [
            TextPartition(
                data="a", embeddings=np.array([0.1, 0.2]), total_tokens=1
            ),
            TextPartition(
                data="b", embeddings=np.array([0.2, 0.3]), total_tokens=1
            ),
        ]

        class DummyPartitioner:
            def __init__(self) -> None:
                self.call_mock = AsyncMock(return_value=partitions)
                self.sentence_model = AsyncMock(
                    return_value=np.array([[0.1, 0.2]])
                )

            async def __call__(self, text: str):
                return await self.call_mock(text)

        partitioner = DummyPartitioner()

        settings = BrowserToolSettings(search=True)
        tool = BrowserTool(settings, self.client, partitioner=partitioner)
        tool._read = AsyncMock(return_value="html")

        index = MagicMock()
        index.add = MagicMock()
        index.search = MagicMock(
            return_value=(np.array([[0.05]]), np.array([[0]]))
        )

        with patch(
            "avalan.tool.browser.IndexFlatL2", return_value=index
        ) as idx_patch:
            ctx = ToolCallContext(
                input=Message(role=MessageRole.USER, content="q")
            )
            result = await tool("http://t", context=ctx)

        partitioner.call_mock.assert_awaited_once_with("html")
        partitioner.sentence_model.assert_awaited_once_with(["q"])
        idx_patch.assert_called_once_with(partitions[0].embeddings.shape[0])
        index.add.assert_called_once()
        index.search.assert_called_once()
        self.assertEqual(result, "a\nb")


class BrowserToolCallDebugTestCase(IsolatedAsyncioTestCase):
    async def test_call_debug_source(self):
        debug_io = StringIO("debug")
        settings = BrowserToolSettings(
            debug=True, debug_url="http://t", debug_source=debug_io
        )
        tool = BrowserTool(settings, MagicMock())
        result = await tool("http://t", context=ToolCallContext())
        self.assertEqual(result, "debug")


class BrowserToolReadTestCase(IsolatedAsyncioTestCase):
    async def test_read_launches_browser_and_page(self):
        settings = BrowserToolSettings()
        page = MagicMock()
        page.goto = AsyncMock(
            return_value=MagicMock(
                headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        page.content = AsyncMock(return_value="<html>hi</html>")
        page.close = AsyncMock()

        browser = MagicMock()
        browser.new_page = AsyncMock(return_value=page)
        browser.close = AsyncMock()

        browser_type = MagicMock()
        browser_type.launch = AsyncMock(return_value=browser)

        client = MagicMock()
        client.firefox = browser_type

        tool = BrowserTool(settings, client)
        tool._md.convert_stream = MagicMock(
            return_value=types.SimpleNamespace(text_content="parsed")
        )

        content = await tool._read("http://t")

        browser_type.launch.assert_awaited_once()
        browser.new_page.assert_awaited_once()
        page.goto.assert_awaited_once_with("http://t")
        page.content.assert_awaited_once()
        self.assertEqual(content, "parsed")

    async def test_aexit_closes_resources(self):
        tool = BrowserTool(BrowserToolSettings(), MagicMock())
        tool._page = AsyncMock()
        tool._browser = AsyncMock()
        with patch(
            "avalan.tool.browser.Tool.__aexit__",
            AsyncMock(return_value=True),
        ) as base_exit:
            result = await tool.__aexit__(None, None, None)
        tool._page.close.assert_awaited_once()
        tool._browser.close.assert_awaited_once()
        base_exit.assert_awaited_once_with(None, None, None)
        self.assertTrue(result)
