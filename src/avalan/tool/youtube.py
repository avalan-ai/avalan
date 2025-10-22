from ..compat import override
from ..entities import (
    GenericProxyConfig,
    ToolCallContext,
    WebshareProxyConfig,
)
from . import Tool, ToolSet

from contextlib import AsyncExitStack
from importlib import import_module
from typing import Iterable


class YouTubeTranscriptTool(Tool):
    """Return the transcript of a YouTube video.

    Args:
        video_id: Identifier of the video to fetch.
        languages: Preferred transcript languages in priority order.

    Returns:
        Transcript text chunks for the requested video.
    """

    _proxy: GenericProxyConfig | WebshareProxyConfig | None

    def __init__(
        self,
        *,
        proxy: GenericProxyConfig | WebshareProxyConfig | None = None,
    ) -> None:
        super().__init__()
        self._proxy = proxy
        self.__name__ = "transcript"

    async def __call__(
        self,
        video_id: str,
        *,
        context: ToolCallContext,
        languages: Iterable[str] | None = None,
    ) -> list[str]:
        proxies = None
        if self._proxy:
            proxy_cfg = (
                self._proxy.to_generic()
                if isinstance(self._proxy, WebshareProxyConfig)
                else self._proxy
            )
            proxies = proxy_cfg.to_dict()
        transcript_api = import_module(
            "youtube_transcript_api"
        ).YouTubeTranscriptApi
        transcript = transcript_api.get_transcript(
            video_id,
            languages=list(languages) if languages else None,
            proxies=proxies,
        )
        return [chunk.get("text", "") for chunk in transcript]


class YouTubeToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        proxy: GenericProxyConfig | WebshareProxyConfig | None = None,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ) -> None:
        tools = [YouTubeTranscriptTool(proxy=proxy)]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
