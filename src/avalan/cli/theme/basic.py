from ...agent.orchestrator import Orchestrator
from ...entities import Model, User
from ...event import EventStats
from ..display_safety import safe_text as _safe_text
from . import Theme
from . import tool_status_icon as _theme_tool_status_icon
from . import tool_status_style as _theme_tool_status_style
from .stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenter,
    CliStreamPresenterItem,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
    StreamFrameRole,
)

from collections.abc import AsyncGenerator, Mapping, Sequence
from json import JSONDecodeError, loads
from logging import Logger
from re import IGNORECASE, MULTILINE, Pattern, compile
from time import perf_counter

from rich import box
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

_BASIC_SUMMARY_LIMIT = 160
_BASIC_TOOL_RUNNING_THRESHOLD_SECONDS = 1.0
_BASIC_TOOL_DYNAMIC_MAX_SECONDS = 3600.0
_BASIC_TOOL_RUNNING_STYLE = "cyan"
_BASIC_XML_TOOL_BLOCK_PATTERN = compile(
    r"<(?P<tag>tool_call|tool|function_call|function|invoke|tool_code)\b"
    r"[^>]*(?:/>|>[\s\S]*?</(?P=tag)>)",
    IGNORECASE,
)
_BASIC_OPEN_XML_TOOL_PATTERN = compile(
    r"<(?:tool_call\b|tool(?:\s|>)|function_call\b|function\s|invoke\b|"
    r"tool_code\b)[\s\S]*$",
    IGNORECASE,
)
_BASIC_DSML_TOOL_CALLS_PATTERN = compile(
    r"<(?:\uff5c)?DSML(?:\uff5c|:)tool_calls\b[^>]*>"
    r"[\s\S]*?</(?:\uff5c)?DSML(?:\uff5c|:)tool_calls>",
    IGNORECASE,
)
_BASIC_DSML_INVOKE_PATTERN = compile(
    r"<(?:\uff5c)?DSML(?:\uff5c|:)invoke\b[^>]*>"
    r"[\s\S]*?</(?:\uff5c)?DSML(?:\uff5c|:)invoke>",
    IGNORECASE,
)
_BASIC_OPEN_DSML_PATTERN = compile(
    r"<(?:\uff5c)?DSML(?:\uff5c|:)"
    r"(?:tool_calls|invoke|parameter)\b[\s\S]*$",
    IGNORECASE,
)
_BASIC_HARMONY_INTERNAL_PATTERN = compile(
    r"(?:<\|start\|>assistant)?"
    r"<\|channel\|>(?:analysis|commentary)[\s\S]*?"
    r"(?:<\|end\|>|<\|call\|>|"
    r"(?=(?:<\|start\|>assistant)?<\|channel\|>final"
    r"<\|message\|>)|$)",
    IGNORECASE,
)
_BASIC_HARMONY_FINAL_PATTERN = compile(
    r"(?:<\|start\|>assistant)?"
    r"<\|channel\|>final<\|message\|>(?P<message>[\s\S]*?)"
    r"(?:<\|end\|>|$)",
    IGNORECASE,
)
_BASIC_HARMONY_TOKEN_PATTERN = compile(r"<\|[^|]+?\|>")
_BASIC_REACT_LINE_PATTERN = compile(
    r"^[ \t]*(?:Thought|Action|Action Input|Observation)\s*:.*(?:\r?\n|$)",
    IGNORECASE | MULTILINE,
)
_BASIC_TOOL_FENCE_PATTERN = compile(
    r"```(?:tool_call|tool|function_call|invoke)\b[\s\S]*?```",
    IGNORECASE,
)
_BASIC_PROTOCOL_PREFIX_MARKERS = (
    "<tool_call",
    "<tool",
    "<function_call",
    "<function",
    "<invoke",
    "<tool_code",
    "<dsml",
    "<\uff5cdsml",
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|call|>",
    "<|end|>",
    "```tool_call",
    "```tool",
    "```function_call",
    "```invoke",
)
_BASIC_REACT_LINE_MARKERS = (
    "thought:",
    "action:",
    "action input:",
    "observation:",
)
_BASIC_ALWAYS_HIDDEN_TOOL_EVENT_TYPES = frozenset(
    {
        "tool_model_run",
        "tool_model_response",
    }
)
_BASIC_CANONICAL_DUPLICATE_TOOL_EVENT_TYPES = frozenset(
    {
        "tool_execute",
        "tool_result",
    }
)


class BasicTheme(Theme):
    """Provide a low-clutter theme using common display defaults."""

    @property
    def default_display_tools(self) -> bool:
        return True

    @property
    def prefix_stream_answers(self) -> bool:
        return True

    @property
    def icons(self) -> dict[str, str]:
        return {
            "avalan": ":heavy_large_circle:",
            "agent_output": ":robot:",
            "user_input": ":speaking_head:",
        }

    def agent(
        self,
        agent: Orchestrator,
        *args: object,
        models: list[Model | str],
        cans_access: bool | None = None,
        can_access: bool | None = None,
    ) -> RenderableType:
        _ = args, cans_access, can_access
        return Panel(
            _basic_session_header(self, agent, models),
            box=box.SQUARE,
            padding=(0, 1),
        )

    def stream_presenter(
        self,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
        answer_prefix: str | None = None,
    ) -> CliStreamPresenter:
        return BasicStreamPresenter(
            logger,
            event_stats=event_stats,
            answer_prefix=answer_prefix,
        )

    def welcome(
        self,
        url: str,
        name: str,
        version: str,
        license: str,
        user: User | None,
    ) -> RenderableType:
        _ = url, user
        self._welcome_name = _safe_text(name)
        self._welcome_version = _safe_text(version)
        self._welcome_license = _safe_text(license)
        return Group()


def _basic_session_header(
    theme: BasicTheme,
    agent: Orchestrator,
    models: list[Model | str],
) -> str:
    version = getattr(theme, "_welcome_version", "")
    license = getattr(theme, "_welcome_license", "MIT")
    app_name = getattr(theme, "_welcome_name", "avalan")
    version_text = f" {_safe_text(version)}" if version else ""
    app = (
        f"{_safe_text(app_name)}{version_text} - {_safe_text(license)} License"
    )
    model_ids = ", ".join(
        _safe_text(model.id if isinstance(model, Model) else model)
        for model in models
    )
    agent_label = _safe_text(
        getattr(agent, "name", None)
        or getattr(agent, "id", None)
        or theme._("Agent")
    )
    details = ", ".join(
        part
        for part in (
            model_ids or theme._("none"),
            _basic_agent_memories(agent, theme),
        )
        if part
    )
    return (
        theme.icons["avalan"]
        + " "
        + app
        + " :sparkles: "
        + agent_label
        + " ("
        + details
        + ")"
    )


def _basic_agent_memories(agent: Orchestrator, theme: BasicTheme) -> str:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return theme._("stateless")

    parts = [
        (
            theme._("short-term message")
            if getattr(memory, "has_recent_message", False)
            else None
        ),
        (
            theme._("long-term message")
            if getattr(memory, "has_permanent_message", False)
            else None
        ),
    ]
    memory_text = ", ".join(part for part in parts if part)
    return memory_text or theme._("stateless")


class BasicStreamPresenter:
    """Present immutable stream snapshots with Basic answer-first output."""

    requires_completion_snapshot: bool = True
    supports_stderr_diagnostics: bool = True

    def __init__(
        self,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
        answer_prefix: str | None = None,
    ) -> None:
        assert isinstance(logger, Logger)
        assert event_stats is None or isinstance(event_stats, EventStats)
        assert answer_prefix is None or isinstance(answer_prefix, str)
        self._answer_presenter = _BasicAnswerPresenter()
        self._answer_prefix = answer_prefix
        self._answer_prefix_emitted = False
        self._answer_separator_emitted = False
        self._executed_tool_frame_seen = False
        self._event_stats = event_stats
        self._logger = logger
        self._final_newline_emitted = False
        self._last_visible_answer_text = ""
        self._visible_roles: set[StreamFrameRole] = set()

    def reset(self) -> None:
        """Forget emitted answer text, newline, and visible frames."""
        self._answer_presenter.reset()
        self._answer_prefix_emitted = False
        self._answer_separator_emitted = False
        self._executed_tool_frame_seen = False
        self._final_newline_emitted = False
        self._last_visible_answer_text = ""
        self._visible_roles.clear()

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncGenerator[CliStreamPresenterItem, None]:
        """Yield Basic answer chunks before optional diagnostic frames."""
        assert isinstance(request, CliStreamPresenterRequest)
        _ = self._event_stats, self._logger
        async for chunk in self._answer_presenter.present(
            _basic_answer_request(request)
        ):
            if (
                self._answer_prefix
                and not self._answer_prefix_emitted
                and not request.display_config.answer_stdout_only
            ):
                if self._needs_answer_separator(request):
                    self._answer_separator_emitted = True
                    yield CliStreamAnswerTextChunk(text="\n")
                self._answer_prefix_emitted = True
                yield CliStreamAnswerTextChunk(text=self._answer_prefix)
            self._last_visible_answer_text += chunk.text
            yield chunk

        newline = self._terminal_newline(request)
        if newline is not None:
            yield newline

        if request.mode == "answer":
            return

        if (
            request.display_config.diagnostic_channel == "live"
            and self._last_visible_answer_text
            and not _basic_terminal_completed(request)
        ):
            return

        if (
            request.display_config.diagnostic_channel == "live"
            and self._last_visible_answer_text
        ):
            role_renderables: tuple[
                tuple[StreamFrameRole, RenderableType | None],
                ...,
            ] = (("stats", _basic_stats_frame(request)),)
        else:
            role_renderables = (
                ("tools", _basic_tool_frame(request)),
                ("events", _basic_event_frame(request)),
                ("stats", _basic_stats_frame(request)),
            )
        for role, renderable in role_renderables:
            if (
                role == "tools"
                and renderable is not None
                and _basic_has_executed_tool_frame(request)
            ):
                self._executed_tool_frame_seen = True
            frame = self._role_frame(role, renderable)
            if frame is not None:
                yield frame

    def _needs_answer_separator(
        self,
        request: CliStreamPresenterRequest,
    ) -> bool:
        return (
            not self._answer_separator_emitted
            and self._executed_tool_frame_seen
            and request.display_config.diagnostic_channel == "live"
        )

    def _terminal_newline(
        self,
        request: CliStreamPresenterRequest,
    ) -> CliStreamAnswerTextChunk | None:
        if self._final_newline_emitted:
            return None
        if not _basic_terminal_completed(request):
            return None
        if not self._last_visible_answer_text:
            return None
        if self._last_visible_answer_text.endswith("\n"):
            return None

        self._final_newline_emitted = True
        return CliStreamAnswerTextChunk(text="\n")

    def _role_frame(
        self,
        role: StreamFrameRole,
        renderable: RenderableType | None,
    ) -> CliStreamRenderableFrame | None:
        if renderable is not None:
            self._visible_roles.add(role)
            return CliStreamRenderableFrame(renderable=renderable, role=role)
        if role not in self._visible_roles:
            return None
        self._visible_roles.remove(role)
        return CliStreamRenderableFrame(renderable="", role=role)


def _basic_answer_request(
    request: CliStreamPresenterRequest,
) -> CliStreamPresenterRequest:
    return CliStreamPresenterRequest(
        snapshot=request.snapshot,
        display_config=request.display_config,
        context=request.context,
        mode="answer",
    )


def _basic_terminal_completed(request: CliStreamPresenterRequest) -> bool:
    terminal = request.snapshot.terminal
    return terminal.completed and terminal.outcome in (None, "completed")


def _basic_tool_frame(
    request: CliStreamPresenterRequest,
) -> RenderableType | None:
    if not request.display_config.show_tools:
        return None

    snapshot = request.snapshot
    result_tool_call_ids = {
        result.tool_call_id for result in snapshot.tool_results
    }
    canonical_tool_call_ids = _basic_canonical_tool_call_ids(request)
    history_lines = [
        *(
            _basic_completed_tool_line(
                tool.name,
                tool.status,
                tool.elapsed_seconds,
            )
            for tool in snapshot.completed_tools
            if tool.tool_call_id not in result_tool_call_ids
        ),
        *(
            _basic_tool_result_line(
                result.name,
                result.status,
                result.result_summary,
                result.elapsed_seconds,
            )
            for result in snapshot.tool_results
        ),
        *(
            _basic_tool_diagnostic_line(
                diagnostic.requested_name or diagnostic.canonical_name,
                diagnostic.code,
                diagnostic.message,
            )
            for diagnostic in snapshot.tool_diagnostics
        ),
        *(
            _basic_tool_event_line(
                event.event_type,
                event.name or event.tool_call_id,
                event.payload_summary,
            )
            for event in snapshot.tool_events
            if _basic_should_show_tool_event(event, canonical_tool_call_ids)
        ),
    ]
    limit = request.display_config.display_tools_events
    if limit is not None:
        history_lines = history_lines[-limit:] if limit else []

    active_renderables = [
        _basic_active_tool_renderable(
            tool.name,
            started_at=tool.started_at,
            updated_at=tool.updated_at,
            spinner=request.display_config.diagnostic_channel == "live",
        )
        for tool in snapshot.active_tools
    ]
    history_text = _basic_frame_text(history_lines)
    renderables: list[RenderableType] = [
        *(
            active_renderable
            for active_renderable in active_renderables
            if active_renderable
        ),
    ]
    if history_text:
        renderables.append(history_text)
    if not renderables:
        return None
    if len(renderables) == 1:
        return renderables[0]
    return Group(*renderables)


def _basic_event_frame(
    request: CliStreamPresenterRequest,
) -> str | None:
    if not request.display_config.show_events:
        return None

    lines = [
        _basic_event_line(
            event.event_type,
            event.payload_summary or event.observability_summary,
        )
        for event in request.snapshot.events
    ]
    return _basic_frame_text(lines)


def _basic_stats_frame(
    request: CliStreamPresenterRequest,
) -> str | None:
    if not request.display_config.show_stats or not _basic_terminal_completed(
        request
    ):
        return None

    snapshot = request.snapshot
    token_counts = snapshot.token_counts
    timing = snapshot.timing
    lines = [
        _basic_stats_counts_line(
            token_counts.input_tokens,
            token_counts.cached_input_tokens,
            token_counts.output_tokens,
            token_counts.reasoning_usage_tokens,
            token_counts.total_tokens,
        ),
        _basic_stats_elapsed_line(timing.elapsed_seconds),
        *(
            _basic_usage_line(usage.kind, usage.usage_summary)
            for usage in snapshot.usage_summaries
        ),
    ]
    return _basic_frame_text(lines)


def _basic_active_tool_line(
    name: str,
    *,
    started_at: float | None,
    updated_at: float | None,
) -> str:
    tool_name = _basic_markup_summary(name)
    if updated_at is None:
        return _basic_tool_progress_line("Starting", tool_name)
    if started_at is not None:
        elapsed_seconds = max(updated_at - started_at, 0.0)
        if elapsed_seconds < _BASIC_TOOL_RUNNING_THRESHOLD_SECONDS:
            return _basic_tool_progress_line("Starting", tool_name)
        elapsed = _basic_tool_elapsed_text(elapsed_seconds)
        if elapsed:
            return _basic_tool_progress_line("Running", tool_name, elapsed)
    return _basic_tool_progress_line("Running", tool_name)


def _basic_tool_progress_line(
    stage: str,
    tool_name: str,
    elapsed: str | None = None,
) -> str:
    suffix = f" for {escape(elapsed)}" if elapsed else ""
    return (
        f"[{_BASIC_TOOL_RUNNING_STYLE}]{escape(stage)} tool "
        f"{tool_name}{suffix}.[/{_BASIC_TOOL_RUNNING_STYLE}]"
    )


class _BasicActiveToolSpinner(Spinner):
    """Render active tool text that ages while Rich refreshes."""

    def __init__(
        self,
        name: str,
        *,
        started_at: float | None,
        updated_at: float | None,
    ) -> None:
        self._tool_name = name
        self._started_at = started_at
        self._updated_at = updated_at
        self._created_at = perf_counter()
        line = self._line()
        super().__init__(
            "point",
            text=Text.from_markup(line),
            style=(
                _BASIC_TOOL_RUNNING_STYLE
                if _basic_is_tool_progress_line(line)
                else None
            ),
        )

    def render(self, time: float) -> RenderableType:
        """Render the spinner with current elapsed tool text."""
        line = self._line()
        self.text = Text.from_markup(line)
        self.style = (
            _BASIC_TOOL_RUNNING_STYLE
            if _basic_is_tool_progress_line(line)
            else None
        )
        return super().render(time)

    def _line(self) -> str:
        return _basic_active_tool_line(
            self._tool_name,
            started_at=self._started_at,
            updated_at=self._current_updated_at(),
        )

    def _current_updated_at(self) -> float | None:
        if self._updated_at is not None:
            return self._updated_at
        if self._started_at is None:
            return None

        updated_at = perf_counter()
        if (
            updated_at - self._created_at
            < _BASIC_TOOL_RUNNING_THRESHOLD_SECONDS
        ):
            return None
        elapsed_seconds = updated_at - self._started_at
        if (
            elapsed_seconds < _BASIC_TOOL_RUNNING_THRESHOLD_SECONDS
            or elapsed_seconds > _BASIC_TOOL_DYNAMIC_MAX_SECONDS
        ):
            return None
        return updated_at


def _basic_is_tool_progress_line(line: str) -> bool:
    return "Starting tool " in line or "Running tool " in line


def _basic_active_tool_renderable(
    name: str,
    *,
    started_at: float | None,
    updated_at: float | None,
    spinner: bool,
) -> RenderableType:
    line = _basic_active_tool_line(
        name,
        started_at=started_at,
        updated_at=updated_at,
    )
    if not spinner:
        return line
    return _BasicActiveToolSpinner(
        name,
        started_at=started_at,
        updated_at=updated_at,
    )


def _basic_completed_tool_line(
    name: str,
    status: str,
    elapsed_seconds: float | None,
) -> str:
    elapsed = _basic_tool_elapsed_text(elapsed_seconds)
    elapsed_text = f" ({escape(elapsed)})" if elapsed else ""
    status_style = _theme_tool_status_style(status)
    status_icon = _theme_tool_status_icon(status)
    return (
        f"[{status_style}]{status_icon} Executed tool "
        f"{_basic_markup_summary(name)}{elapsed_text}: "
        f"{_basic_markup_summary(status)}.[/{status_style}]"
    )


def _basic_tool_result_line(
    name: str,
    status: str,
    result_summary: str,
    elapsed_seconds: float | None,
) -> str:
    status_style = _theme_tool_status_style(status)
    status_icon = _theme_tool_status_icon(status)
    elapsed = _basic_tool_elapsed_text(elapsed_seconds)
    elapsed_text = f" ({escape(elapsed)})" if elapsed else ""
    return (
        f"[{status_style}]{status_icon} Executed tool "
        f"{_basic_markup_summary(name)}{elapsed_text}: "
        f"{_basic_markup_summary(_basic_tool_result_summary(result_summary))}"
        f"[/{status_style}]"
    )


def _basic_tool_diagnostic_line(
    name: str | None,
    code: str,
    message: str,
) -> str:
    subject = f" {_basic_summary(name)}" if name else ""
    return (
        f"tool{subject} diagnostic {_basic_summary(code)}: "
        f"{_basic_summary(message)}"
    )


def _basic_tool_event_line(
    event_type: str,
    name: str | None,
    payload_summary: str | None,
) -> str:
    subject = _basic_summary(name or payload_summary or "")
    suffix = f": {subject}" if subject else ""
    return f"tool event {_basic_summary(event_type)}{suffix}"


def _basic_has_executed_tool_frame(
    request: CliStreamPresenterRequest,
) -> bool:
    if not request.display_config.show_tools:
        return False

    snapshot = request.snapshot
    result_tool_call_ids = {
        result.tool_call_id for result in snapshot.tool_results
    }
    canonical_tool_call_ids = _basic_canonical_tool_call_ids(request)
    history_entries = [
        *(
            True
            for tool in snapshot.completed_tools
            if tool.tool_call_id not in result_tool_call_ids
        ),
        *(True for _ in snapshot.tool_results),
        *(False for _ in snapshot.tool_diagnostics),
        *(
            False
            for event in snapshot.tool_events
            if _basic_should_show_tool_event(event, canonical_tool_call_ids)
        ),
    ]
    limit = request.display_config.display_tools_events
    if limit is not None:
        history_entries = history_entries[-limit:] if limit else []
    return any(history_entries)


def _basic_canonical_tool_call_ids(
    request: CliStreamPresenterRequest,
) -> set[str]:
    snapshot = request.snapshot
    return {
        *(
            tool.tool_call_id
            for tool in snapshot.completed_tools
            if tool.tool_call_id
        ),
        *(
            result.tool_call_id
            for result in snapshot.tool_results
            if result.tool_call_id
        ),
    }


def _basic_should_show_tool_event(
    event: object,
    canonical_tool_call_ids: set[str],
) -> bool:
    event_type = getattr(event, "event_type")
    tool_call_id = getattr(event, "tool_call_id")
    if event_type in _BASIC_ALWAYS_HIDDEN_TOOL_EVENT_TYPES:
        return False
    if event_type not in _BASIC_CANONICAL_DUPLICATE_TOOL_EVENT_TYPES:
        return True
    return not canonical_tool_call_ids or tool_call_id not in (
        canonical_tool_call_ids | {None}
    )


def _basic_event_line(
    event_type: str,
    summary: str | None,
) -> str:
    suffix = f": {_basic_summary(summary)}" if summary else ""
    return f"event {_basic_summary(event_type)}{suffix}"


def _basic_stats_counts_line(
    input_tokens: int | None,
    cached_input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    total_tokens: int,
) -> str:
    parts = [
        _basic_count("in", input_tokens),
        _basic_count("cached", cached_input_tokens),
        _basic_count("out", output_tokens),
        _basic_count("reasoning", reasoning_tokens),
        _basic_count("total", total_tokens),
    ]
    return "tokens " + ", ".join(part for part in parts if part)


def _basic_stats_elapsed_line(elapsed_seconds: float | None) -> str | None:
    if elapsed_seconds is None:
        return None
    return f"elapsed {elapsed_seconds:.2f}s"


def _basic_tool_elapsed_text(elapsed_seconds: float | None) -> str | None:
    if elapsed_seconds is None:
        return None

    seconds = max(0.0, elapsed_seconds)
    if seconds < 1.0:
        milliseconds = max(1, round(seconds * 1000))
        return f"{milliseconds}ms"
    if seconds < 10.0:
        return f"{seconds:.1f}".rstrip("0").rstrip(".") + "s"
    if seconds < 60.0:
        return f"{round(seconds)}s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        remaining_seconds = round(seconds % 60)
        return f"{minutes}m {remaining_seconds:02d}s"

    hours = int(seconds // 3600)
    remaining_minutes = round((seconds % 3600) / 60)
    return f"{hours}h {remaining_minutes:02d}m"


def _basic_usage_line(kind: str | None, usage_summary: str) -> str:
    label = _basic_summary(kind or "usage")
    return f"usage {label}: {_basic_summary(usage_summary)}"


def _basic_tool_result_summary(result_summary: str) -> str:
    try:
        parsed = loads(result_summary)
    except (JSONDecodeError, TypeError, ValueError):
        return result_summary
    if isinstance(parsed, Mapping) and "result" in parsed:
        return str(parsed["result"])
    return result_summary


def _basic_count(label: str, value: int | None) -> str | None:
    if value is None:
        return None
    return f"{label}={value}"


def _basic_frame_text(lines: Sequence[str | None]) -> str | None:
    text = "\n".join(line for line in lines if line)
    return text or None


def _basic_summary(value: str | None) -> str:
    if not value:
        return ""
    text = " ".join(value.split())
    if len(text) <= _BASIC_SUMMARY_LIMIT:
        return text
    return f"{text[: _BASIC_SUMMARY_LIMIT - 3]}..."


def _basic_markup_summary(value: str | None) -> str:
    return escape(_basic_summary(value))


class _BasicAnswerPresenter:
    """Emit Basic answer text after suppressing protocol-shaped chunks."""

    def __init__(self) -> None:
        self._emitted_answer_text = ""
        self._emitted_visible_answer_text = ""

    def reset(self) -> None:
        """Forget previously emitted raw and visible answer text."""
        self._emitted_answer_text = ""
        self._emitted_visible_answer_text = ""

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncGenerator[CliStreamAnswerTextChunk, None]:
        """Yield unseen, user-visible answer text."""
        assert isinstance(request, CliStreamPresenterRequest)
        if request.mode != "answer":
            raise AssertionError("answer presenter requires answer mode")
        answer_text = request.snapshot.answer_text
        if answer_text == self._emitted_answer_text:
            return
        if not answer_text.startswith(self._emitted_answer_text):
            raise AssertionError("answer snapshots must grow monotonically")

        visible_answer_text = _basic_visible_answer_text(
            answer_text,
            terminal_completed=_basic_terminal_completed(request),
        )
        self._emitted_answer_text = answer_text
        if visible_answer_text == self._emitted_visible_answer_text:
            return

        if visible_answer_text.startswith(self._emitted_visible_answer_text):
            text = visible_answer_text[
                len(self._emitted_visible_answer_text) :
            ]
        else:
            text = visible_answer_text
        self._emitted_visible_answer_text = visible_answer_text
        if text:
            yield CliStreamAnswerTextChunk(text=text)


def _basic_visible_answer_text(
    text: str,
    *,
    terminal_completed: bool,
) -> str:
    assert isinstance(text, str)
    assert isinstance(terminal_completed, bool)
    if not text:
        return ""

    filtered = text
    for pattern in (
        _BASIC_TOOL_FENCE_PATTERN,
        _BASIC_DSML_TOOL_CALLS_PATTERN,
        _BASIC_DSML_INVOKE_PATTERN,
        _BASIC_XML_TOOL_BLOCK_PATTERN,
        _BASIC_HARMONY_INTERNAL_PATTERN,
    ):
        filtered = pattern.sub("", filtered)
    filtered = _BASIC_HARMONY_FINAL_PATTERN.sub(
        lambda match: match.group("message"),
        filtered,
    )
    filtered = _BASIC_HARMONY_TOKEN_PATTERN.sub("", filtered)
    filtered = _BASIC_REACT_LINE_PATTERN.sub("", filtered)
    filtered = _basic_remove_json_tool_lines(filtered)
    for pattern in (
        _BASIC_OPEN_DSML_PATTERN,
        _BASIC_OPEN_XML_TOOL_PATTERN,
        _basic_open_harmony_pattern(filtered),
    ):
        filtered = pattern.sub("", filtered)
    if _basic_json_tool_answer(filtered):
        return ""
    if not terminal_completed:
        filtered = _basic_withhold_protocol_prefix_suffix(filtered)
    return filtered


def _basic_withhold_protocol_prefix_suffix(text: str) -> str:
    length = _basic_protocol_prefix_suffix_length(text)
    if length == 0:
        return text
    return text[:-length]


def _basic_protocol_prefix_suffix_length(text: str) -> int:
    if not text:
        return 0

    lowered = text.lower()
    maximum = 0
    marker_length = max(
        len(marker) for marker in _BASIC_PROTOCOL_PREFIX_MARKERS
    )
    for length in range(1, min(marker_length, len(text)) + 1):
        suffix = lowered[-length:]
        if suffix[0] not in "<`":
            continue
        if any(
            marker.startswith(suffix)
            for marker in _BASIC_PROTOCOL_PREFIX_MARKERS
        ):
            maximum = length

    react_length = _basic_react_prefix_suffix_length(text)
    return max(maximum, react_length)


def _basic_react_prefix_suffix_length(text: str) -> int:
    line_start = max(text.rfind("\n"), text.rfind("\r")) + 1
    candidate = text[line_start:].lstrip(" \t").lower()
    if not candidate:
        return 0
    if any(
        marker.startswith(candidate) for marker in _BASIC_REACT_LINE_MARKERS
    ):
        return len(text) - line_start
    return 0


def _basic_open_harmony_pattern(text: str) -> Pattern[str]:
    if "<|channel|>" not in text and "<|start|>" not in text:
        return compile(r"$^")
    return compile(
        r"(?:<\|start\|>assistant)?"
        r"<\|channel\|>(?:analysis|commentary)[\s\S]*$",
        IGNORECASE,
    )


def _basic_remove_json_tool_lines(text: str) -> str:
    if not text:
        return ""
    kept_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if _basic_json_tool_answer(line):
            continue
        kept_lines.append(line)
    return "".join(kept_lines)


def _basic_json_tool_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        value = loads(stripped)
    except JSONDecodeError:
        return _basic_partial_json_tool_answer(stripped)
    return _basic_json_tool_payload(value)


def _basic_partial_json_tool_answer(text: str) -> bool:
    if not text.startswith("{"):
        return False
    lowered = text.lower()
    if '"tool_calls"' in lowered or '"function_call"' in lowered:
        return True
    if '"arguments"' not in lowered:
        return False
    return '"tool"' in lowered or '"name"' in lowered


def _basic_json_tool_payload(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = {str(key) for key in value}
    if "tool_calls" in keys or "function_call" in keys:
        return True
    if value.get("type") in {"tool_call", "function_call"}:
        return True
    return "arguments" in keys and bool({"tool", "name"} & keys)
