from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, patch

from avalan.entities import (
    GenericProxyConfig,
    ToolCallContext,
    WebshareProxyConfig,
)
from avalan.tool.youtube import YouTubeToolSet, YouTubeTranscriptTool


class YouTubeToolSetTestCase(TestCase):
    def test_init(self):
        toolset = YouTubeToolSet()
        self.assertEqual(len(toolset.tools), 1)
        self.assertIsInstance(toolset.tools[0], YouTubeTranscriptTool)


class YouTubeTranscriptToolTestCase(IsolatedAsyncioTestCase):
    async def test_call_no_proxy(self):
        dummy_api = SimpleNamespace(
            YouTubeTranscriptApi=SimpleNamespace(
                get_transcript=MagicMock(
                    return_value=[{"text": "a"}, {"text": "b"}]
                )
            )
        )
        with patch(
            "avalan.tool.youtube.import_module", return_value=dummy_api
        ) as imp:
            tool = YouTubeTranscriptTool()
            result = await tool(
                "id", languages=["en"], context=ToolCallContext()
            )
            imp.assert_called_once_with("youtube_transcript_api")
            dummy_api.YouTubeTranscriptApi.get_transcript.assert_called_once_with(
                "id", languages=["en"], proxies=None
            )
            self.assertEqual(result, ["a", "b"])

    async def test_call_with_generic_proxy(self):
        dummy_api = SimpleNamespace(
            YouTubeTranscriptApi=SimpleNamespace(
                get_transcript=MagicMock(return_value=[])
            )
        )
        with patch(
            "avalan.tool.youtube.import_module", return_value=dummy_api
        ):
            proxy = GenericProxyConfig(
                scheme="http", host="h", port=1, username="u", password="p"
            )
            tool = YouTubeTranscriptTool(proxy=proxy)
            await tool("id2", context=ToolCallContext())
            url = "http://u:p@h:1"
            dummy_api.YouTubeTranscriptApi.get_transcript.assert_called_once_with(
                "id2", languages=None, proxies={"http": url, "https": url}
            )

    async def test_call_with_webshare_proxy(self):
        dummy_api = SimpleNamespace(
            YouTubeTranscriptApi=SimpleNamespace(
                get_transcript=MagicMock(return_value=[])
            )
        )
        with patch(
            "avalan.tool.youtube.import_module", return_value=dummy_api
        ):
            proxy = WebshareProxyConfig(
                host="wh", port=2, username="w", password="pw"
            )
            tool = YouTubeTranscriptTool(proxy=proxy)
            await tool("id3", context=ToolCallContext())
            url = "http://w:pw@wh:2"
            dummy_api.YouTubeTranscriptApi.get_transcript.assert_called_once_with(
                "id3", languages=None, proxies={"http": url, "https": url}
            )


if __name__ == "__main__":
    main()
