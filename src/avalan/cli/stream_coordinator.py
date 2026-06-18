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
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from types import TracebackType
from typing import Protocol, TypeAlias

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner

_FRAME_ROLE_ORDER: tuple[StreamFrameRole, ...] = (
    "events",
    "tools",
    "stats",
    "stream",
    "answer",
)


@dataclass(frozen=True, slots=True)
class _PromptPauseIdle:
    """Represent a coordinator with no active tool prompt."""


@dataclass(frozen=True, slots=True)
class _PromptPauseActive:
    """Represent a coordinator paused for one tool prompt."""


_PromptPauseState: TypeAlias = _PromptPauseIdle | _PromptPauseActive


@dataclass(frozen=True, slots=True)
class _LiveRefreshRunning:
    """Represent live refresh running with no saved state."""


@dataclass(frozen=True, slots=True)
class _LiveRefreshPaused:
    """Represent live refresh paused with its previous setting."""

    auto_refresh: bool


_LiveRefreshState: TypeAlias = _LiveRefreshRunning | _LiveRefreshPaused
_PROMPT_PAUSE_IDLE = _PromptPauseIdle()
_LIVE_REFRESH_RUNNING = _LiveRefreshRunning()


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
CliStreamClock = Callable[[], float]


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
        diagnostic_console: Console | None = None,
        live_factory: CliStreamLiveFactory | None = None,
        record_filename_factory: RecordFilenameFactory | None = None,
        clock: CliStreamClock | None = None,
    ) -> None:
        assert isinstance(display_config, CliStreamDisplayConfig)
        assert diagnostic_console is None or callable(
            getattr(diagnostic_console, "print", None)
        )
        assert clock is None or callable(clock)
        self._console = console
        self._diagnostic_console = diagnostic_console
        self._display_config = display_config
        self._live_factory = live_factory or _default_live_factory
        self._record_filename_factory = (
            record_filename_factory or stream_recording_filename
        )
        self._clock = clock or perf_counter
        self._flush_interval = 1 / display_config.refresh_per_second
        self._last_flush_at: float | None = None
        self._live: CliStreamLive | None = None
        self._live_refresh: _LiveRefreshState = _LIVE_REFRESH_RUNNING
        self._role_renderables: dict[StreamFrameRole, RenderableType] = {}
        self._stderr_role_renderables: dict[StreamFrameRole, str] = {}
        self._pending_flush = False
        self._manual_pause_depth = 0
        self._prompt_pause: _PromptPauseState = _PROMPT_PAUSE_IDLE
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
        if self._display_config.diagnostic_channel == "none":
            return
        if self._display_config.diagnostic_channel == "stderr":
            self._render_stderr_frame(frame)
            return

        self._role_renderables[frame.role] = frame.renderable
        self._pending_flush = True
        if self._is_paused():
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
        """Pause live rendering manually."""
        async with self._lock:
            assert not self._closed
            was_paused = self._is_paused()
            self._manual_pause_depth += 1
            if not was_paused:
                self._pause_live_refresh()

    async def resume(self) -> None:
        """Resume live rendering and flush queued frames."""
        async with self._lock:
            await self._resume()

    async def flush(self) -> None:
        """Flush the latest queued live frame."""
        async with self._lock:
            assert not self._closed
            if self._is_paused():
                return
            await self._flush_pending(force=True)

    async def _resume(self) -> None:
        assert not self._closed
        if self._manual_pause_depth == 0:
            return
        self._manual_pause_depth -= 1
        if self._is_paused():
            return

        self._resume_live_refresh()
        try:
            await self._flush_pending(force=True)
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

        async with self._tool_prompt_paused():
            return await to_thread(
                prompt,
                self._console,
                call,
                tty_path=tty_path,
            )

    @asynccontextmanager
    async def paused(self) -> AsyncIterator[None]:
        """Pause live rendering within a manual async context."""
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

        should_flush = flush
        try:
            if should_flush:
                self._manual_pause_depth = 0
                self._prompt_pause = _PROMPT_PAUSE_IDLE
                self._resume_live_refresh()
                await self._flush_pending(force=True)
        finally:
            self._closed = True
            self._restore_live_refresh(refresh=False)
            self._clear_pause_state()
            self._close_live()

    async def _flush_pending(self, *, force: bool = False) -> None:
        assert not self._is_paused()
        if not self._pending_flush or not self._role_renderables:
            return
        if not force and not self._flush_gate_due():
            return

        renderable = self._current_renderable()
        live = self._ensure_live()
        live.update(renderable)
        self._last_flush_at = self._clock()
        self._pending_flush = False
        if self._display_config.record_enabled:
            self._console.save_svg(
                self._record_filename_factory(),
                clear=True,
            )

    def _flush_gate_due(self) -> bool:
        if self._last_flush_at is None:
            return True
        return self._clock() - self._last_flush_at >= self._flush_interval

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

    def _render_stderr_frame(self, frame: CliStreamRenderableFrame) -> None:
        key = _stderr_renderable_key(frame.renderable)
        if not key:
            self._stderr_role_renderables.pop(frame.role, None)
            return
        if self._stderr_role_renderables.get(frame.role) == key:
            return

        self._stderr_role_renderables[frame.role] = key
        self._ensure_diagnostic_console().print(frame.renderable)

    def _ensure_diagnostic_console(self) -> Console:
        if self._diagnostic_console is None:
            self._diagnostic_console = Console(
                stderr=True,
                force_terminal=False,
            )
        return self._diagnostic_console

    @asynccontextmanager
    async def _tool_prompt_paused(self) -> AsyncIterator[None]:
        await self._pause_for_prompt()
        try:
            yield
        except BaseException:
            await self.aclose(flush=False)
            raise
        else:
            try:
                await self._resume_prompt()
            except BaseException:
                await self.aclose(flush=False)
                raise

    async def _pause_for_prompt(self) -> None:
        async with self._lock:
            assert not self._closed
            self._start_prompt_pause()

    async def _resume_prompt(self) -> None:
        async with self._lock:
            if self._closed:
                return
            assert isinstance(self._prompt_pause, _PromptPauseActive)
            self._prompt_pause = _PROMPT_PAUSE_IDLE
            if self._is_paused():
                return

            self._resume_live_refresh()
            try:
                await self._flush_pending(force=True)
            except BaseException:
                await self._aclose(flush=False)
                raise

    def _start_prompt_pause(self) -> None:
        assert isinstance(self._prompt_pause, _PromptPauseIdle)
        was_paused = self._is_paused()
        self._prompt_pause = _PromptPauseActive()
        if not was_paused:
            self._pause_live_refresh()

    def _is_paused(self) -> bool:
        return self._manual_pause_depth > 0 or self._is_prompt_paused()

    def _is_prompt_paused(self) -> bool:
        return isinstance(self._prompt_pause, _PromptPauseActive)

    def _pause_live_refresh(self) -> None:
        assert isinstance(self._live_refresh, _LiveRefreshRunning)
        if self._live is None:
            return

        self._live_refresh = _LiveRefreshPaused(
            auto_refresh=self._live.auto_refresh
        )
        self._live.auto_refresh = False
        self._live.refresh()

    def _resume_live_refresh(self) -> None:
        self._restore_live_refresh(refresh=True)

    def _restore_live_refresh(self, *, refresh: bool) -> None:
        state = self._live_refresh
        if isinstance(state, _LiveRefreshRunning):
            return

        if self._live is not None:
            self._live.auto_refresh = state.auto_refresh
            if refresh:
                self._live.refresh()
        self._live_refresh = _LIVE_REFRESH_RUNNING

    def _clear_pause_state(self) -> None:
        self._manual_pause_depth = 0
        self._prompt_pause = _PROMPT_PAUSE_IDLE
        self._live_refresh = _LIVE_REFRESH_RUNNING

    def _close_live(self) -> None:
        live = self._live
        if live is None:
            return

        self._live = None
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


def _stderr_renderable_key(renderable: RenderableType) -> str:
    if isinstance(renderable, Group):
        return "\n".join(
            key
            for key in (
                _stderr_renderable_key(child)
                for child in renderable.renderables
            )
            if key
        )
    if isinstance(renderable, Spinner):
        return str(renderable.text or "")
    return str(renderable)
