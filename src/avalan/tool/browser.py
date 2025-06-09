from . import Tool, ToolSet
from ..compat import override
from ..entities import Message, MessageRole, ToolCallContext
from ..filters import Partitioner
from contextlib import AsyncExitStack
from dataclasses import dataclass
from email.message import EmailMessage
from faiss import IndexFlatL2
from io import BytesIO, TextIOBase
from markitdown import MarkItDown
from numpy import vstack
from playwright.async_api import (
    async_playwright,
    Browser,
    Page,
    PlaywrightContextManager,
)
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class BrowserToolSettings(dict):
    engine: Literal["chromium", "firefox", "webkit"] = "firefox"
    search: bool = False
    search_context: int | None = 10
    search_k: int = 1
    debug: bool = False
    debug_url: str | None = None
    debug_source: TextIOBase | None = None
    slowdown: int | None = None
    devtools: bool = False
    chromium_sandbox: bool = True
    viewport_width: int = 1024
    viewport_height: int = 768
    scale_factor: float = 1.0
    is_mobile: bool = False
    has_touch: bool = False
    java_script_enabled: bool = True

    def __post_init__(self):
        self["debug"] = self.debug
        self["debug_url"] = self.debug_url
        self["debug_source"] = self.debug_source
        self["engine"] = self.engine
        self["search"] = self.search
        self["search_context"] = self.search_context
        self["search_k"] = self.search_k
        self["slowdown"] = self.slowdown
        self["devtools"] = self.devtools
        self["chromium_sandbox"] = self.chromium_sandbox
        self["viewport_width"] = self.viewport_width
        self["viewport_height"] = self.viewport_height
        self["scale_factor"] = self.scale_factor
        self["is_mobile"] = self.is_mobile
        self["has_touch"] = self.has_touch
        self["java_script_enabled"] = self.java_script_enabled


class BrowserTool(Tool):
    """
    Uses a web browser to access the internet and get the contents of a URL.

    Args:
        url: URL of a page to open.

    Returns:
        Contents of the given URL page.
    """

    _client: PlaywrightContextManager
    _settings: BrowserToolSettings
    _browser: Browser | None = None
    _page: Page | None = None
    _md: MarkItDown | None = None
    _partitioner: Partitioner | None = None

    def __init__(
        self,
        settings: BrowserToolSettings,
        client: PlaywrightContextManager,
        partitioner: Partitioner | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._client = client
        self._md = MarkItDown()
        self._partitioner = partitioner
        self.__name__ = "open"

    async def __call__(self, url: str, *, context: ToolCallContext) -> str:
        content = await self._read(url)

        if (
            self._settings.search
            and self._partitioner
            and context.input
            and isinstance(context.input, Message)
            and context.input.role == MessageRole.USER
        ):
            query = context.input.content
            sentence_model = self._partitioner.sentence_model
            knowledge_partitions = await self._partitioner(content)

            knowledge_embeddings = [
                kp.embeddings for kp in knowledge_partitions
            ]
            query_embeddings = await sentence_model([query])

            assert knowledge_embeddings and query_embeddings.any()

            knowledge_stack = vstack(knowledge_embeddings).astype(
                "float32", copy=False
            )
            query_stack = vstack(query_embeddings).astype(
                "float32", copy=False
            )

            index = IndexFlatL2(knowledge_embeddings[0].shape[0])
            index.add(knowledge_stack)

            distances, ids = index.search(query_stack, self._settings.search_k)

            matches: list[tuple[int, int, float]] = [
                (q_id, kn_id, float(dist))
                for q_id, (dist_row, id_row) in enumerate(zip(distances, ids))
                for dist, kn_id in zip(dist_row, id_row)
            ]
            # smallest distance first
            matches.sort(key=lambda t: t[2])

            for q_id, kn_id, l2_distance in matches:
                knowledge_chunk = (
                    knowledge_partitions[kn_id].data
                    if knowledge_partitions
                    else query if kn_id == 0 else None
                )
                if not knowledge_chunk:
                    continue

                knowledge_match = (
                    "\n".join(
                        [
                            kp.data
                            for kp in knowledge_partitions[
                                max(
                                    kn_id - self._settings.search_context, 0
                                ) : min(
                                    kn_id + self._settings.search_context + 1,
                                    len(knowledge_partitions),
                                )
                            ]
                        ]
                    )
                    if self._settings.search_context
                    else knowledge_chunk
                )

                content = knowledge_match

        return content

    async def _read(self, url: str) -> str:
        if (
            self._settings.debug
            and self._settings.debug_url
            and url == self._settings.debug_url
            and self._settings.debug_source
        ):
            assert isinstance(self._settings.debug_source, TextIOBase)
            content = self._settings.debug_source.read()
            return content

        if not self._browser:
            browser_type = (
                self._client.chromium
                if self._settings.engine == "chromium"
                else (
                    self._client.firefox
                    if self._settings.engine == "firefox"
                    else (
                        self._client.webkit
                        if self._settings.engine == "webkit"
                        else None
                    )
                )
            )
            assert browser_type
            self._browser = await browser_type.launch(
                headless=True,
                slow_mo=self._settings.slowdown,
                devtools=self._settings.devtools,
                executable_path=None,
                args=[],
                env={},
                timeout=0,
                firefox_user_prefs={},
                chromium_sandbox=self._settings.chromium_sandbox,
            )

        if not self._page:
            self._page = await self._browser.new_page(
                viewport={
                    "width": self._settings.viewport_width,
                    "height": self._settings.viewport_height,
                },
                screen=None,
                device_scale_factor=self._settings.scale_factor,
                is_mobile=self._settings.is_mobile,
                has_touch=self._settings.has_touch,
                java_script_enabled=self._settings.java_script_enabled,
                locale=None,
                timezone_id=None,
                geolocation=None,
                permissions=[],
                extra_http_headers={},
                ignore_https_errors=False,
                bypass_csp=False,
                offline=False,
                http_credentials=None,
                storage_state=None,
                accept_downloads=True,
                base_url=None,
                color_scheme=None,
            )

        response = await self._page.goto(url)
        contents: str = await self._page.content()
        content_type_header = response.headers.get("content-type", None)
        assert content_type_header

        m = EmailMessage()
        m["content-type"] = content_type_header
        maintype = m.get_content_maintype() or "text"
        assert maintype == "text"
        encoding = (m.get_param("charset") or "utf-8").lower()
        mime_type = m.get_content_type()
        byte_stream = BytesIO(contents.encode(encoding))
        result = self._md.convert_stream(byte_stream, mime_type=mime_type)
        content = result.text_content
        return content

    def with_client(self, client: PlaywrightContextManager) -> "BrowserTool":
        self._client = client
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        if self._page:
            await self._page.close()
        if self._browser:
            await self._browser.close()

        return await super().__aexit__(exc_type, exc_value, traceback)


class BrowserToolSet(ToolSet):
    _client: PlaywrightContextManager | None = None

    @override
    def __init__(
        self,
        settings: BrowserToolSettings,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
        partitioner: Partitioner | None = None,
    ):
        assert settings

        if not exit_stack:
            exit_stack = AsyncExitStack()

        self._client = async_playwright()

        tools = [BrowserTool(settings, self._client, partitioner=partitioner)]
        return super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )

    @override
    async def __aenter__(self) -> "BrowserToolSet":
        self._client = await self._exit_stack.enter_async_context(self._client)
        for i, tool in enumerate(self._tools):
            self._tools[i] = tool.with_client(self._client)
        return await super().__aenter__()
