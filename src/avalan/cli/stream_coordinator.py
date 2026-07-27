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
from inspect import iscoroutinefunction
from time import perf_counter
from types import TracebackType
from typing import Protocol, TypeAlias, TypeGuard

from rich.console import (
    Console,
    ConsoleOptions,
    Group,
    RenderableType,
    RenderResult,
)
from rich.live import Live
from rich.segment import Segment
from rich.spinner import Spinner
from rich.text import Text

_FRAME_ROLE_ORDER: tuple[StreamFrameRole, ...] = (
    "events",
    "reasoning",
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
class _LiveRunning:
    """Represent an active or not-yet-created live display."""


@dataclass(frozen=True, slots=True)
class _LiveSuspended:
    """Retain live configuration until a new terminal owner is created."""

    auto_refresh: bool


_LiveState: TypeAlias = _LiveRunning | _LiveSuspended
_PROMPT_PAUSE_IDLE = _PromptPauseIdle()
_LIVE_RUNNING = _LiveRunning()


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


class CliStreamPresentationLock(Protocol):
    """Provide exclusive async context-manager presentation access."""

    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _TailOverflowRenderable(Group):
    """Render the newest rows when live output exceeds the terminal."""

    def __init__(self, renderable: RenderableType) -> None:
        if isinstance(renderable, Group):
            super().__init__(*renderable.renderables, fit=renderable.fit)
        else:
            super().__init__(renderable)
        self.renderable = renderable
        self._show_all = False

    def __str__(self) -> str:
        return _stderr_renderable_text(self.renderable)

    def show_all(self) -> None:
        """Disable tail cropping for the final live render."""
        self._show_all = True

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        lines = console.render_lines(self.renderable, options, pad=False)
        maximum_height = options.size.height
        if not self._show_all and len(lines) > maximum_height:
            if maximum_height == 1:
                lines = lines[-1:]
            else:
                overflow = Text(
                    "...",
                    overflow="crop",
                    justify="center",
                    end="",
                    style="live.ellipsis",
                )
                lines = [
                    list(console.render(overflow, options)),
                    *lines[-(maximum_height - 1) :],
                ]

        for index, line in enumerate(lines):
            yield from line
            if index < len(lines) - 1:
                yield Segment.line()


class _TailOverflowLive:
    """Show the live tail while preserving the complete final render."""

    def __init__(
        self,
        live: CliStreamLive,
        renderable: _TailOverflowRenderable | None,
    ) -> None:
        self._live = live
        self._renderable = renderable

    @property
    def auto_refresh(self) -> bool:
        """Return whether the delegated live display refreshes itself."""
        return self._live.auto_refresh

    @auto_refresh.setter
    def auto_refresh(self, value: bool) -> None:
        """Set whether the delegated live display refreshes itself."""
        self._live.auto_refresh = value

    def __enter__(self) -> "_TailOverflowLive":
        self._live = self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._renderable is not None:
            self._renderable.show_all()
        return self._live.__exit__(exc_type, exc_value, traceback)

    def refresh(self) -> None:
        """Refresh the delegated live display."""
        self._live.refresh()

    def update(self, renderable: RenderableType) -> None:
        """Display the newest rows and retain the complete renderable."""
        self._renderable = _TailOverflowRenderable(renderable)
        self._live.update(self._renderable)


RecordFilenameFactory = Callable[[], str]
CliStreamClock = Callable[[], float]
CliStreamLiveCommitCallback = Callable[[], None]


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
        presentation_lock: CliStreamPresentationLock | None = None,
        live_commit_callback: CliStreamLiveCommitCallback | None = None,
    ) -> None:
        assert isinstance(display_config, CliStreamDisplayConfig)
        assert diagnostic_console is None or callable(
            getattr(diagnostic_console, "print", None)
        )
        assert clock is None or callable(clock)
        assert presentation_lock is None or _is_presentation_lock(
            presentation_lock
        )
        assert live_commit_callback is None or callable(live_commit_callback)
        self._console = console
        self._diagnostic_console = diagnostic_console
        self._display_config = display_config
        self._live_factory = live_factory or _default_live_factory
        self._record_filename_factory = (
            record_filename_factory or stream_recording_filename
        )
        self._clock = clock or perf_counter
        self._presentation_lock = presentation_lock
        self._live_commit_callback = live_commit_callback
        self._flush_interval = 1 / display_config.refresh_per_second
        self._last_flush_at: float | None = None
        self._live: CliStreamLive | None = None
        self._live_state: _LiveState = _LIVE_RUNNING
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
        if self._pending_flush and not self._is_paused():
            await self._flush_pending(force=True)
        self._close_live(commit=True)
        self._role_renderables.clear()
        self._pending_flush = False
        self._console.print(chunk.text, end="")

    async def pause(self) -> None:
        """Pause live rendering manually."""
        async with self._presentation_boundary():
            async with self._lock:
                assert not self._closed
                was_paused = self._is_paused()
                self._manual_pause_depth += 1
                if not was_paused:
                    try:
                        self._suspend_live()
                    except BaseException:
                        await self._aclose(flush=False)
                        raise

    async def resume(self) -> None:
        """Resume live rendering and flush queued frames."""
        async with self._presentation_boundary():
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

        try:
            await self._flush_pending(force=True)
            if self._live is not None:
                self._live.refresh()
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
                await self._flush_pending(force=True)
                self._live_state = _LIVE_RUNNING
        finally:
            self._closed = True
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
        state = self._live_state
        if isinstance(state, _LiveSuspended):
            live.auto_refresh = state.auto_refresh
        self._live = live.__enter__()
        self._live_state = _LIVE_RUNNING
        return self._live

    def _render_stderr_frame(self, frame: CliStreamRenderableFrame) -> None:
        key = _stderr_renderable_key(frame.renderable)
        if frame.stderr_append:
            has_content = key is None or bool(key)
            if has_content:
                self._ensure_diagnostic_console().print(
                    frame.renderable,
                    end="",
                )
            return
        if key == "":
            self._stderr_role_renderables.pop(frame.role, None)
            return
        if (
            key is not None
            and self._stderr_role_renderables.get(frame.role) == key
        ):
            return

        if key is None:
            self._stderr_role_renderables.pop(frame.role, None)
        else:
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
        async with self._presentation_boundary():
            async with self._lock:
                assert not self._closed
                try:
                    self._start_prompt_pause()
                except BaseException:
                    await self._aclose(flush=False)
                    raise

    async def _resume_prompt(self) -> None:
        async with self._presentation_boundary():
            async with self._lock:
                if self._closed:
                    return
                assert isinstance(self._prompt_pause, _PromptPauseActive)
                self._prompt_pause = _PROMPT_PAUSE_IDLE
                if self._is_paused():
                    return

                try:
                    await self._flush_pending(force=True)
                    if self._live is not None:
                        self._live.refresh()
                except BaseException:
                    await self._aclose(flush=False)
                    raise

    def _start_prompt_pause(self) -> None:
        assert isinstance(self._prompt_pause, _PromptPauseIdle)
        was_paused = self._is_paused()
        self._prompt_pause = _PromptPauseActive()
        if not was_paused:
            self._suspend_live()

    def _is_paused(self) -> bool:
        return self._manual_pause_depth > 0 or self._is_prompt_paused()

    def _is_prompt_paused(self) -> bool:
        return isinstance(self._prompt_pause, _PromptPauseActive)

    def _suspend_live(self) -> None:
        """Withdraw Rich terminal ownership while a prompt reads input."""
        if self._live is None:
            return

        assert isinstance(self._live_state, _LiveRunning)
        live = self._live
        self._live = None
        self._live_state = _LiveSuspended(auto_refresh=live.auto_refresh)
        update_succeeded = False
        try:
            if self._pending_flush and self._role_renderables:
                live.update(self._current_renderable())
            update_succeeded = True
        finally:
            try:
                live.__exit__(None, None, None)
                if update_succeeded and self._live_commit_callback is not None:
                    self._live_commit_callback()
            finally:
                self._role_renderables.clear()
                self._pending_flush = False
                self._last_flush_at = None

    @asynccontextmanager
    async def _presentation_boundary(self) -> AsyncIterator[None]:
        lock = self._presentation_lock
        if lock is None:
            yield
            return
        async with lock:
            yield

    def _clear_pause_state(self) -> None:
        self._manual_pause_depth = 0
        self._prompt_pause = _PROMPT_PAUSE_IDLE
        self._live_state = _LIVE_RUNNING

    def _close_live(self, *, commit: bool = False) -> None:
        assert isinstance(commit, bool)
        live = self._live
        if live is None:
            return

        self._live = None
        live.__exit__(None, None, None)
        if commit and self._live_commit_callback is not None:
            self._live_commit_callback()


CliStreamOutputCoordinator = CliStreamCoordinator


def _is_presentation_lock(
    value: object,
) -> TypeGuard[CliStreamPresentationLock]:
    value_type = type(value)
    return iscoroutinefunction(
        getattr(value_type, "__aenter__", None)
    ) and iscoroutinefunction(getattr(value_type, "__aexit__", None))


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
    live_renderable = (
        _TailOverflowRenderable(renderable) if renderable is not None else None
    )
    return _TailOverflowLive(
        Live(
            live_renderable,
            console=console,
            refresh_per_second=refresh_per_second,
            screen=screen,
        ),
        live_renderable,
    )


def _stderr_renderable_key(renderable: RenderableType) -> str | None:
    if isinstance(renderable, Group):
        keys = tuple(
            _stderr_renderable_key(child) for child in renderable.renderables
        )
        if any(key is None for key in keys):
            return None
        return "\n".join(key for key in keys if key)
    if isinstance(renderable, Spinner):
        return str(renderable.text or "")
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, str):
        return renderable
    return None


def _stderr_renderable_text(renderable: RenderableType) -> str:
    key = _stderr_renderable_key(renderable)
    return str(renderable) if key is None else key
