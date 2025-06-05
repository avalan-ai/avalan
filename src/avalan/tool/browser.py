from . import Tool, ToolSet
from ..compat import override
from contextlib import AsyncExitStack
from io import BytesIO
from markitdown import MarkItDown
from playwright.async_api import async_playwright, Browser, Page, PlaywrightContextManager


class BrowserTool(Tool):
    """
    Uses a web browser to access the internet.

    Args:
        url: URL of a page to open

    Returns:
        Contents of the given URL page.
    """

    _client: PlaywrightContextManager
    _browser: Browser | None = None
    _page: Page | None = None

    def __init__(self, client: PlaywrightContextManager) -> None:
        super().__init__()
        self._client = client
        self.__name__ = "open"

    async def __call__(self, url: str) -> str:
        if not self._browser:
            browser_type = self._client.firefox
            self._browser = await browser_type.launch(
                headless=False,
                slow_mo=2000, # slows down operations by the given number of milliseconds
                devtools=False,
                executable_path=None,
                args=[],
                env={},
                timeout=0,
                firefox_user_prefs={},
                chromium_sandbox=True
            )

        if not self._page:
            self._page = await self._browser.new_page(
                viewport={"width": 1024, "height": 768},
                screen=None,
                device_scale_factor=1.0,
                is_mobile=False,
                has_touch=False,
                java_script_enabled=True,
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
                color_scheme=None
            )

        await self._page.goto(url)
        contents: str = await self._page.content()
        md = MarkItDown()
        byte_stream = BytesIO(contents.encode("utf-8"))
        result = md.convert(byte_stream, extension="html")

        return result.text_content

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

        return await super().__aexit__(
            exc_type, exc_value, traceback
        )


class BrowserToolSet(ToolSet):
    _client: PlaywrightContextManager | None = None

    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None
    ):
        if not exit_stack:
            exit_stack = AsyncExitStack()

        self._client = async_playwright()

        tools = [
            BrowserTool(self._client)
        ]
        return super().__init__(
            exit_stack=exit_stack,
            namespace=namespace,
            tools=tools
        )

    @override
    async def __aenter__(self) -> "BrowserToolSet":
        self._client = await self._exit_stack.enter_async_context(self._client)
        for i, tool in enumerate(self._tools):
            self._tools[i] = tool.with_client(self._client)
        return await super().__aenter__()

