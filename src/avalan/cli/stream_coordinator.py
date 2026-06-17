from ..entities import ToolCall
from . import confirm_tool_call
from .display import CliStreamDisplayConfig
from .stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenterItem,
    CliStreamRenderableFrame,
    StreamFrameRole,
)

from asyncio import Lock, to_thread
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import TracebackType
from typing import Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live

_FRAME_ROLE_ORDER: tuple[StreamFrameRole, ...] = (
    "events",
    "tools",
    "stats",
    "stream",
    "answer",
)


class CliStreamLive(Protocol):
    """Represent the Rich live methods owned by the coordinator."""

    auto_refresh: bool

    def __enter__(self) -> "CliStreamLive": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def refresh(self) -> None: ...

    def update(self, renderable: RenderableType) -> None: ...


class CliStreamLiveFactory(Protocol):
    """Create a Rich live instance for the coordinator."""

    def __call__(
        self,
        renderable: RenderableType | None,
        *,
        console: Console,
        refresh_per_second: int,
        screen: bool,
    ) -> CliStreamLive: ...


RecordFilenameFactory = Callable[[], str]


class ToolConfirmationPrompt(Protocol):
    """Prompt for one tool call confirmation."""

    def __call__(
        self,
        console: Console,
        call: ToolCall,
        *,
        tty_path: str,
    ) -> str: ...


class CliStreamCoordinator:
    """Coordinate one CLI streaming output lifecycle."""

    def __init__(
        self,
        console: Console,
        display_config: CliStreamDisplayConfig,
        *,
        live_factory: CliStreamLiveFactory | None = None,
        record_filename_factory: RecordFilenameFactory | None = None,
    ) -> None:
        assert isinstance(display_config, CliStreamDisplayConfig)
        self._console = console
        self._display_config = display_config
        self._live_factory = live_factory or _default_live_factory
        self._record_filename_factory = (
            record_filename_factory or stream_recording_filename
        )
        self._live: CliStreamLive | None = None
        self._live_auto_refresh: bool | None = None
        self._role_renderables: dict[StreamFrameRole, RenderableType] = {}
        self._pending_flush = False
        self._pause_depth = 0
        self._closed = False
        self._lock = Lock()

    async def __aenter__(self) -> "CliStreamCoordinator":
        assert not self._closed
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        await self.aclose(flush=exc_value is None)
        return None

    async def handle_item(self, item: CliStreamPresenterItem) -> None:
        """Render or print one stream presenter item."""
        async with self._lock:
            if isinstance(item, CliStreamAnswerTextChunk):
                await self._print_answer_chunk(item)
                return
            if isinstance(item, CliStreamRenderableFrame):
                await self._render_frame(item)
                return

            await self._aclose(flush=False)
            raise AssertionError("unsupported CLI stream presenter item")

    async def render_frame(self, frame: CliStreamRenderableFrame) -> None:
        """Render one live frame through the single owner."""
        async with self._lock:
            await self._render_frame(frame)

    async def _render_frame(self, frame: CliStreamRenderableFrame) -> None:
        assert isinstance(frame, CliStreamRenderableFrame)
        assert not self._closed
        if not self._display_config.live_enabled:
            return

        self._role_renderables[frame.role] = frame.renderable
        self._pending_flush = True
        if self._pause_depth > 0:
            return

        try:
            await self._flush_pending()
        except BaseException:
            await self._aclose(flush=False)
            raise

    async def print_answer_chunk(
        self,
        chunk: CliStreamAnswerTextChunk,
    ) -> None:
        """Print one answer text chunk without starting live rendering."""
        async with self._lock:
            await self._print_answer_chunk(chunk)

    async def _print_answer_chunk(
        self,
        chunk: CliStreamAnswerTextChunk,
    ) -> None:
        assert isinstance(chunk, CliStreamAnswerTextChunk)
        assert not self._closed
        self._console.print(chunk.text, end="")

    async def pause(self) -> None:
        """Pause live rendering for a prompt."""
        async with self._lock:
            assert not self._closed
            self._pause_depth += 1
            if self._pause_depth == 1:
                self._pause_live_refresh()

    async def resume(self) -> None:
        """Resume live rendering and flush queued frames."""
        async with self._lock:
            await self._resume()

    async def flush(self) -> None:
        """Flush the latest queued live frame."""
        async with self._lock:
            assert not self._closed
            if self._pause_depth > 0:
                return
            await self._flush_pending()

    async def _resume(self) -> None:
        assert not self._closed
        if self._pause_depth == 0:
            return
        self._pause_depth -= 1
        if self._pause_depth > 0:
            return

        self._resume_live_refresh()
        try:
            await self._flush_pending()
        except BaseException:
            await self._aclose(flush=False)
            raise

    async def confirm_tool_call(
        self,
        call: ToolCall,
        *,
        tty_path: str = "/dev/tty",
        prompt: ToolConfirmationPrompt = confirm_tool_call,
    ) -> str:
        """Prompt for one tool confirmation while live rendering is paused."""
        assert isinstance(call, ToolCall)
        assert isinstance(tty_path, str)
        assert callable(prompt)

        async with self.paused():
            return await to_thread(
                prompt,
                self._console,
                call,
                tty_path=tty_path,
            )

    @asynccontextmanager
    async def paused(self) -> AsyncIterator[None]:
        """Pause live rendering while awaiting a prompt result."""
        await self.pause()
        try:
            yield
        except BaseException:
            await self.aclose(flush=False)
            raise
        finally:
            if not self._closed:
                await self.resume()

    async def aclose(self, *, flush: bool = True) -> None:
        """Close the live owner, optionally flushing a final frame."""
        async with self._lock:
            await self._aclose(flush=flush)

    async def _aclose(self, *, flush: bool = True) -> None:
        if self._closed:
            return

        try:
            if flush:
                self._pause_depth = 0
                self._resume_live_refresh()
                await self._flush_pending()
        finally:
            self._closed = True
            self._close_live()

    async def _flush_pending(self) -> None:
        if not self._pending_flush or not self._role_renderables:
            return

        renderable = self._current_renderable()
        live = self._ensure_live()
        live.update(renderable)
        self._pending_flush = False
        if self._display_config.record_enabled:
            self._console.save_svg(
                self._record_filename_factory(),
                clear=True,
            )

    def _current_renderable(self) -> RenderableType:
        renderables = [
            self._role_renderables[role]
            for role in _FRAME_ROLE_ORDER
            if role in self._role_renderables
        ]
        assert renderables
        if len(renderables) == 1:
            return renderables[0]
        return Group(*renderables)

    def _ensure_live(self) -> CliStreamLive:
        if self._live is not None:
            return self._live

        live = self._live_factory(
            None,
            console=self._console,
            refresh_per_second=self._display_config.refresh_per_second,
            screen=self._display_config.record_enabled,
        )
        self._live = live.__enter__()
        return self._live

    def _pause_live_refresh(self) -> None:
        if self._live is None or self._live_auto_refresh is not None:
            return

        self._live_auto_refresh = self._live.auto_refresh
        self._live.auto_refresh = False
        self._live.refresh()

    def _resume_live_refresh(self) -> None:
        if self._live is None or self._live_auto_refresh is None:
            return

        self._live.auto_refresh = self._live_auto_refresh
        self._live_auto_refresh = None
        self._live.refresh()

    def _close_live(self) -> None:
        live = self._live
        if live is None:
            return

        self._live = None
        self._live_auto_refresh = None
        live.__exit__(None, None, None)


CliStreamOutputCoordinator = CliStreamCoordinator


def stream_recording_filename() -> str:
    """Return the default SVG recording filename."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d%H%M%S")
    ms = now.microsecond // 1000
    return f"avalan-screenshot-{ts}-{ms:03d}.svg"


def _default_live_factory(
    renderable: RenderableType | None,
    *,
    console: Console,
    refresh_per_second: int,
    screen: bool,
) -> CliStreamLive:
    return Live(
        renderable,
        console=console,
        refresh_per_second=refresh_per_second,
        screen=screen,
    )
