from ...agent.orchestrator import Orchestrator
from ...entities import Model, User
from ...event import EventStats
from ...tool.display import ToolDisplayProjection
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
    pretty_json_answer_text,
    structured_answer_started,
)
from .tool_projection import (
    projection_outcome,
    projection_status,
    projection_summary_markup,
)

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
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
_BASIC_DATABASE_SQL_PREVIEW_LIMIT = 72
_BASIC_TOOL_RUNNING_THRESHOLD_SECONDS = 1.0
_BASIC_TOOL_DYNAMIC_MAX_SECONDS = 3600.0
_BASIC_TOOL_RUNNING_STYLE = "cyan"
_BASIC_TOOL_ARGUMENT_STYLE = "dim"
_BASIC_TOOL_ELAPSED_STYLE = "bold"
_BASIC_TOOL_NAME_STYLE = "bold"
_BASIC_TOOL_SUCCESS_STATUSES = frozenset({"completed", "result"})
_BASIC_DATABASE_ACTIVE_VERBS = {
    "count": "Counting",
    "inspect": "Inspecting",
    "keys": "Inspecting",
    "relationships": "Inspecting",
    "plan": "Explaining",
    "run": "Running",
    "sample": "Sampling",
    "size": "Measuring",
    "tables": "Listing",
    "tasks": "Listing",
    "kill": "Cancelling",
    "locks": "Inspecting",
}
_BASIC_DATABASE_COMPLETED_VERBS = {
    "count": "Counted",
    "inspect": "Inspected",
    "keys": "Inspected",
    "relationships": "Inspected",
    "plan": "Explained",
    "run": "Ran",
    "sample": "Sampled",
    "size": "Measured",
    "tables": "Listed",
    "tasks": "Listed",
    "kill": "Cancelled",
    "locks": "Inspected",
}
_BASIC_DATABASE_ACTION_OPERATIONS = {
    "count": "count",
    "inspect": "inspect",
    "explain": "plan",
    "query": "run",
    "sample": "sample",
    "measure": "size",
    "list": "tables",
    "cancel": "kill",
}
_BASIC_DATABASE_TARGET_DESCRIPTIONS = {
    "count": ("rows in table", True),
    "inspect": ("tables", True),
    "keys": ("keys for table", True),
    "relationships": ("relationships for table", True),
    "sample": ("rows from table", True),
    "size": ("table", True),
    "tables": ("tables", False),
    "tasks": ("tasks", False),
    "kill": ("task", True),
    "locks": ("locks", False),
}
_BASIC_DATABASE_IDENTITY_DETAIL_LABELS = frozenset(
    {
        "database",
        "database_name",
        "db",
        "db_name",
    }
)
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
        "tool_diagnostic",
        "tool_progress",
    }
)
_BASIC_CANONICAL_DUPLICATE_TOOL_EVENT_TYPES = frozenset(
    {
        "tool_execute",
        "tool_result",
    }
)


@dataclass(frozen=True, slots=True)
class _BasicToolLineEntry:
    key: str
    line: str
    executed: bool = False


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
        self._stderr_tool_line_keys: set[str] = set()
        self._active_model_continuations: dict[
            str, tuple[float | None, float | None]
        ] = {}
        self._completed_model_continuation_keys: set[str] = set()

    def reset(self) -> None:
        """Forget emitted answer text, newline, and visible frames."""
        self._answer_presenter.reset()
        self._answer_prefix_emitted = False
        self._answer_separator_emitted = False
        self._executed_tool_frame_seen = False
        self._final_newline_emitted = False
        self._last_visible_answer_text = ""
        self._visible_roles.clear()
        self._stderr_tool_line_keys.clear()
        self._active_model_continuations.clear()
        self._completed_model_continuation_keys.clear()

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncGenerator[CliStreamPresenterItem, None]:
        """Yield Basic answer chunks before optional diagnostic frames."""
        assert isinstance(request, CliStreamPresenterRequest)
        _ = self._event_stats, self._logger
        pre_answer_frame = self._pre_answer_tool_frame(request)
        if pre_answer_frame is not None:
            yield pre_answer_frame
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
        elif request.display_config.diagnostic_channel == "stderr":
            role_renderables = (
                ("tools", self._stderr_tool_frame(request)),
                ("events", _basic_event_frame(request)),
                ("stats", _basic_stats_frame(request)),
            )
        else:
            role_renderables = (
                ("tools", self._tool_frame(request)),
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

    def _tool_frame(
        self,
        request: CliStreamPresenterRequest,
    ) -> RenderableType | None:
        return _basic_tool_frame(
            request,
            completed_model_entries=self._completed_model_entries(request),
        )

    def _stderr_tool_frame(
        self,
        request: CliStreamPresenterRequest,
    ) -> RenderableType | None:
        entries = _basic_tool_entries(
            request,
            include_active=True,
            completed_model_entries=self._completed_model_entries(request),
        )
        new_lines: list[str] = []
        for entry in entries:
            if entry.key in self._stderr_tool_line_keys:
                continue
            self._stderr_tool_line_keys.add(entry.key)
            new_lines.append(entry.line)
        return _basic_frame_text(new_lines)

    def _pre_answer_tool_frame(
        self,
        request: CliStreamPresenterRequest,
    ) -> CliStreamRenderableFrame | None:
        if request.mode == "answer":
            return None
        if request.display_config.diagnostic_channel != "live":
            return None
        if not request.display_config.show_tools:
            return None
        visible_answer_text = _basic_visible_answer_text(
            request.snapshot.answer_text,
            terminal_completed=_basic_terminal_completed(request),
        )
        if structured_answer_started(
            visible_answer_text
        ) and not _basic_terminal_completed(request):
            return None
        if (
            not visible_answer_text
            or visible_answer_text == self._last_visible_answer_text
        ):
            return None
        has_progress = bool(
            self._active_model_continuations
            or request.snapshot.active_model_continuations
            or request.snapshot.active_tools
        )
        entries = self._completed_model_entries(request)
        if not has_progress and not entries:
            return None
        renderable = _basic_tool_frame(
            request,
            include_active_tool_progress=False,
            include_model_progress=False,
        )
        return self._role_frame("tools", renderable)

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

    def _completed_model_entries(
        self,
        request: CliStreamPresenterRequest,
    ) -> tuple[_BasicToolLineEntry, ...]:
        active: dict[str, tuple[float | None, float | None]] = {}
        for continuation in request.snapshot.active_model_continuations:
            active[continuation.model_continuation_id] = (
                continuation.started_at,
                continuation.updated_at,
            )

        completed: list[_BasicToolLineEntry] = []
        for (
            continuation_id,
            timestamps,
        ) in self._active_model_continuations.items():
            if continuation_id in active:
                continue
            key = f"model:{continuation_id}:completed"
            if key in self._completed_model_continuation_keys:
                continue
            self._completed_model_continuation_keys.add(key)
            finished_at = (
                timestamps[1]
                if timestamps[1] is not None
                else perf_counter() if timestamps[0] is not None else None
            )
            completed.append(
                _BasicToolLineEntry(
                    key=key,
                    line=_basic_completed_model_line(
                        started_at=timestamps[0],
                        updated_at=finished_at,
                    ),
                )
            )
        self._active_model_continuations = active
        return tuple(completed)


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
    *,
    completed_model_entries: tuple[_BasicToolLineEntry, ...] = (),
    include_active_tool_progress: bool = True,
    include_model_progress: bool = True,
) -> RenderableType | None:
    if not request.display_config.show_tools:
        return None

    history_lines = [
        entry.line
        for entry in _basic_tool_entries(request, include_active=False)
    ]
    history_lines.extend(entry.line for entry in completed_model_entries)
    active_model_renderables = (
        [
            _basic_active_model_renderable(
                started_at=continuation.started_at,
                updated_at=continuation.updated_at,
                spinner=request.display_config.diagnostic_channel == "live",
            )
            for continuation in request.snapshot.active_model_continuations
        ]
        if include_model_progress
        else []
    )
    active_tool_renderables = (
        [
            _basic_active_tool_renderable(
                tool,
                started_at=tool.started_at,
                updated_at=tool.updated_at,
                spinner=request.display_config.diagnostic_channel == "live",
            )
            for tool in request.snapshot.active_tools
        ]
        if include_active_tool_progress
        else []
    )
    history_text = _basic_frame_text(history_lines)
    renderables: list[RenderableType] = []
    if history_text:
        renderables.append(history_text)
    renderables.extend(
        active_renderable
        for active_renderable in active_model_renderables
        if active_renderable
    )
    renderables.extend(
        active_renderable
        for active_renderable in active_tool_renderables
        if active_renderable
    )
    if not renderables:
        return None
    if len(renderables) == 1:
        return renderables[0]
    return Group(*renderables)


def _basic_tool_entries(
    request: CliStreamPresenterRequest,
    *,
    include_active: bool,
    completed_model_entries: tuple[_BasicToolLineEntry, ...] = (),
) -> tuple[_BasicToolLineEntry, ...]:
    if not request.display_config.show_tools:
        return ()

    snapshot = request.snapshot
    result_tool_call_ids = {
        result.tool_call_id for result in snapshot.tool_results
    }
    canonical_tool_call_ids = _basic_canonical_tool_call_ids(request)
    terminal_entries = (
        (_basic_terminal_error_entry(request),)
        if _basic_terminal_error(request)
        else (
            (_basic_empty_answer_entry(),)
            if _basic_terminal_empty_answer(request)
            else ()
        )
    )
    history_entries = [
        *(
            _BasicToolLineEntry(
                key=_basic_completed_tool_key(tool),
                line=_basic_completed_tool_line(
                    tool.name,
                    tool.status,
                    tool.elapsed_seconds,
                    display_projection=tool.display_projection,
                ),
                executed=True,
            )
            for tool in snapshot.completed_tools
            if tool.tool_call_id not in result_tool_call_ids
        ),
        *(
            _BasicToolLineEntry(
                key=_basic_tool_result_key(result),
                line=_basic_tool_result_line(
                    result.name,
                    result.status,
                    result.result_summary,
                    result.elapsed_seconds,
                    display_projection=result.display_projection,
                ),
                executed=True,
            )
            for result in snapshot.tool_results
        ),
        *(
            _BasicToolLineEntry(
                key=_basic_tool_diagnostic_key(diagnostic),
                line=_basic_tool_diagnostic_line(
                    diagnostic.requested_name or diagnostic.canonical_name,
                    diagnostic.code,
                    diagnostic.message,
                ),
            )
            for diagnostic in snapshot.tool_diagnostics
        ),
        *(
            _BasicToolLineEntry(
                key=_basic_tool_event_key(event),
                line=_basic_tool_event_line(
                    event.event_type,
                    event.name or event.tool_call_id,
                    event.payload_summary,
                ),
            )
            for event in snapshot.tool_events
            if _basic_should_show_tool_event(event, canonical_tool_call_ids)
        ),
        *completed_model_entries,
    ]
    limit = request.display_config.display_tools_events
    if limit is not None:
        history_entries = history_entries[-limit:] if limit else []
    history_entries.extend(terminal_entries)
    if not include_active:
        return tuple(history_entries)

    active_entries = [
        *(
            _BasicToolLineEntry(
                key=f"model:{continuation.model_continuation_id}",
                line=_basic_active_model_line(
                    started_at=continuation.started_at,
                    updated_at=continuation.updated_at,
                ),
            )
            for continuation in snapshot.active_model_continuations
        ),
        *(
            _BasicToolLineEntry(
                key=f"tool:{tool.tool_call_id}",
                line=_basic_active_tool_line(
                    tool.name,
                    started_at=tool.started_at,
                    updated_at=tool.updated_at,
                    display_projection=tool.display_projection,
                ),
            )
            for tool in snapshot.active_tools
        ),
    ]
    return (*history_entries, *active_entries)


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


def _basic_completed_tool_key(tool: object) -> str:
    return (
        "completed:"
        f"{getattr(tool, 'tool_call_id', '')}:"
        f"{getattr(tool, 'sequence', '')}:"
        f"{getattr(tool, 'status', '')}"
    )


def _basic_tool_result_key(result: object) -> str:
    return (
        "result:"
        f"{getattr(result, 'tool_call_id', '')}:"
        f"{getattr(result, 'sequence', '')}:"
        f"{getattr(result, 'status', '')}"
    )


def _basic_tool_diagnostic_key(diagnostic: object) -> str:
    return (
        "diagnostic:"
        f"{getattr(diagnostic, 'diagnostic_id', '')}:"
        f"{getattr(diagnostic, 'sequence', '')}"
    )


def _basic_tool_event_key(event: object) -> str:
    return (
        "event:"
        f"{getattr(event, 'event_type', '')}:"
        f"{getattr(event, 'tool_call_id', '')}:"
        f"{getattr(event, 'sequence', '')}:"
        f"{getattr(event, 'name', '')}:"
        f"{getattr(event, 'payload_summary', '')}"
    )


def _basic_empty_answer_entry() -> _BasicToolLineEntry:
    return _BasicToolLineEntry(
        key="terminal:no_answer",
        line="\n[yellow]⚠️ No final answer emitted.[/yellow]",
    )


def _basic_terminal_error_entry(
    request: CliStreamPresenterRequest,
) -> _BasicToolLineEntry:
    summary = request.snapshot.terminal.error_summary or "stream errored"
    return _BasicToolLineEntry(
        key="terminal:error",
        line=(
            "\n[red]✖ Model stream error: "
            f"{_basic_markup_summary(summary)}[/red]"
        ),
    )


def _basic_terminal_error(request: CliStreamPresenterRequest) -> bool:
    terminal = request.snapshot.terminal
    return terminal.completed and terminal.outcome == "errored"


def _basic_terminal_empty_answer(
    request: CliStreamPresenterRequest,
) -> bool:
    return (
        _basic_terminal_completed(request)
        and not _basic_visible_answer_text(
            request.snapshot.answer_text,
            terminal_completed=True,
        ).strip()
    )


def _basic_active_model_line(
    *,
    started_at: float | None,
    updated_at: float | None,
) -> str:
    elapsed = None
    if started_at is not None and updated_at is not None:
        elapsed = _basic_tool_elapsed_text(max(updated_at - started_at, 0.0))
    suffix = f" for {escape(elapsed)}" if elapsed else ""
    return (
        f"[{_BASIC_TOOL_RUNNING_STYLE}]Thinking{suffix}..."
        f"[/{_BASIC_TOOL_RUNNING_STYLE}]"
    )


def _basic_completed_model_line(
    *,
    started_at: float | None,
    updated_at: float | None,
) -> str:
    elapsed = None
    if started_at is not None and updated_at is not None:
        elapsed = _basic_tool_elapsed_text(max(updated_at - started_at, 0.0))
    suffix = f" for {escape(elapsed)}" if elapsed else ""
    return (
        f"[{_BASIC_TOOL_RUNNING_STYLE}]Thought{suffix}."
        f"[/{_BASIC_TOOL_RUNNING_STYLE}]"
    )


class _BasicActiveModelSpinner(Spinner):
    """Render active model continuation text while Rich refreshes."""

    def __init__(
        self,
        *,
        started_at: float | None,
        updated_at: float | None,
    ) -> None:
        self._started_at = started_at
        self._updated_at = updated_at
        super().__init__(
            "point",
            text=Text.from_markup(self._line()),
            style=_BASIC_TOOL_RUNNING_STYLE,
        )

    def render(self, time: float) -> RenderableType:
        """Render the spinner with current elapsed model text."""
        self.text = Text.from_markup(self._line())
        self.style = _BASIC_TOOL_RUNNING_STYLE
        return super().render(time)

    def _line(self) -> str:
        return _basic_active_model_line(
            started_at=self._started_at,
            updated_at=self._current_updated_at(),
        )

    def _current_updated_at(self) -> float | None:
        if self._updated_at is not None:
            return self._updated_at
        if self._started_at is None:
            return None
        return perf_counter()


def _basic_active_model_renderable(
    *,
    started_at: float | None,
    updated_at: float | None,
    spinner: bool,
) -> RenderableType:
    if not spinner:
        return _basic_active_model_line(
            started_at=started_at,
            updated_at=updated_at,
        )
    return _BasicActiveModelSpinner(
        started_at=started_at,
        updated_at=updated_at,
    )


def _basic_active_tool_line(
    name: str,
    *,
    started_at: float | None,
    updated_at: float | None,
    display_projection: ToolDisplayProjection | None = None,
) -> str:
    database_phrase = _basic_database_phrase_markup(
        display_projection,
        completed=False,
    )
    if database_phrase:
        return _basic_database_progress_line(
            database_phrase,
            started_at=started_at,
            updated_at=updated_at,
        )
    tool_name = _basic_active_tool_name(name, display_projection)
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


def _basic_active_tool_name(
    name: str,
    display_projection: ToolDisplayProjection | None,
) -> str:
    if display_projection is None:
        return _basic_tool_subject_markup(name)
    return _basic_projection_subject_markup(display_projection)


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
        display_projection: ToolDisplayProjection | None = None,
    ) -> None:
        self._tool_name = name
        self._display_projection = display_projection
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
            display_projection=self._display_projection,
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
    tool: object,
    *,
    started_at: float | None,
    updated_at: float | None,
    spinner: bool,
) -> RenderableType:
    if isinstance(tool, str):
        name = tool
        display_projection = None
    else:
        name = getattr(tool, "name")
        assert isinstance(name, str)
        display_projection = getattr(tool, "display_projection", None)
        assert display_projection is None or isinstance(
            display_projection,
            ToolDisplayProjection,
        )
    line = _basic_active_tool_line(
        name,
        started_at=started_at,
        updated_at=updated_at,
        display_projection=display_projection,
    )
    if not spinner:
        return line
    return _BasicActiveToolSpinner(
        name,
        started_at=started_at,
        updated_at=updated_at,
        display_projection=display_projection,
    )


def _basic_completed_tool_line(
    name: str,
    status: str,
    elapsed_seconds: float | None,
    *,
    display_projection: ToolDisplayProjection | None = None,
) -> str:
    display_status = (
        status
        if display_projection is None
        else projection_status(display_projection, status) or status
    )
    status_style = _theme_tool_status_style(display_status)
    status_icon = _theme_tool_status_icon(display_status)
    tool_name = (
        _basic_tool_subject_markup(name)
        if display_projection is None
        else _basic_projection_subject_markup(display_projection)
    )
    database_phrase = _basic_database_phrase_markup(
        display_projection,
        completed=display_status in _BASIC_TOOL_SUCCESS_STATUSES,
    )
    summary = _basic_tool_terminal_markup(
        status,
        display_status,
        display_projection=display_projection,
    )
    summary_text = f": {summary}" if summary else ""
    elapsed_text = _basic_elapsed_suffix(elapsed_seconds)
    return (
        f"[{status_style}]{status_icon} "
        f"{database_phrase or f'Executed tool {tool_name}'}"
        f"{elapsed_text}{summary_text}"
        f"[/{status_style}]"
    )


def _basic_tool_result_line(
    name: str,
    status: str,
    result_summary: str,
    elapsed_seconds: float | None,
    *,
    display_projection: ToolDisplayProjection | None = None,
) -> str:
    display_status = (
        status
        if display_projection is None
        else projection_status(display_projection, status) or status
    )
    status_style = _theme_tool_status_style(display_status)
    status_icon = _theme_tool_status_icon(display_status)
    elapsed_text = _basic_elapsed_suffix(elapsed_seconds)
    tool_name = (
        _basic_tool_subject_markup(name)
        if display_projection is None
        else _basic_projection_subject_markup(display_projection)
    )
    database_phrase = _basic_database_phrase_markup(
        display_projection,
        completed=display_status in _BASIC_TOOL_SUCCESS_STATUSES,
    )
    summary = (
        _basic_markup_summary(_basic_tool_result_summary(result_summary))
        if display_projection is None
        else _basic_tool_terminal_markup(
            status,
            display_status,
            display_projection=display_projection,
        )
    )
    summary_text = f": {summary}" if summary else ""
    return (
        f"[{status_style}]{status_icon} "
        f"{database_phrase or f'Executed tool {tool_name}'}"
        f"{elapsed_text}{summary_text}"
        f"[/{status_style}]"
    )


def _basic_database_progress_line(
    phrase: str,
    *,
    started_at: float | None,
    updated_at: float | None,
) -> str:
    if updated_at is None:
        return (
            f"[{_BASIC_TOOL_RUNNING_STYLE}]{phrase}..."
            f"[/{_BASIC_TOOL_RUNNING_STYLE}]"
        )
    if started_at is not None:
        elapsed_seconds = max(updated_at - started_at, 0.0)
        if elapsed_seconds >= _BASIC_TOOL_RUNNING_THRESHOLD_SECONDS:
            elapsed = _basic_tool_elapsed_text(elapsed_seconds)
            if elapsed:
                return (
                    f"[{_BASIC_TOOL_RUNNING_STYLE}]{phrase} for "
                    f"{escape(elapsed)}.[/{_BASIC_TOOL_RUNNING_STYLE}]"
                )
    return (
        f"[{_BASIC_TOOL_RUNNING_STYLE}]{phrase}.[/{_BASIC_TOOL_RUNNING_STYLE}]"
    )


def _basic_database_phrase_markup(
    projection: ToolDisplayProjection | None,
    *,
    completed: bool,
) -> str | None:
    if projection is None:
        return None
    operation = _basic_database_operation(projection)
    if operation is None:
        return None
    verbs = (
        _BASIC_DATABASE_COMPLETED_VERBS
        if completed
        else _BASIC_DATABASE_ACTIVE_VERBS
    )
    verb = verbs.get(operation)
    if not verb:
        return None
    phrase = _basic_database_specific_phrase_markup(
        operation,
        projection,
        completed=completed,
        verb=verb,
    )
    if phrase:
        return phrase
    return " ".join(
        part
        for part in (
            _basic_database_bold_markup(verb),
            _basic_database_target_markup(operation, projection),
            _basic_database_scope_markup(projection),
        )
        if part
    )


def _basic_database_operation(
    projection: ToolDisplayProjection,
) -> str | None:
    assert isinstance(projection, ToolDisplayProjection)
    label = projection.label or ""
    if label.startswith("database."):
        return label.rsplit(".", 1)[1]
    operation_detail = _basic_projection_detail_value(
        projection,
        "operation",
    )
    if operation_detail:
        return operation_detail
    if projection.scope != "database":
        return None
    if projection.action == "list" and projection.target == "tasks":
        return "tasks"
    return _BASIC_DATABASE_ACTION_OPERATIONS.get(projection.action)


def _basic_database_specific_phrase_markup(
    operation: str,
    projection: ToolDisplayProjection,
    *,
    completed: bool,
    verb: str,
) -> str | None:
    if operation == "run":
        return _basic_database_run_phrase_markup(
            projection,
            completed=completed,
            verb=verb,
        )
    if operation == "inspect":
        return _basic_database_inspect_phrase_markup(projection, verb=verb)
    if operation == "tables":
        return _basic_database_tables_phrase_markup(
            projection,
            completed=completed,
            verb=verb,
        )
    return None


def _basic_database_run_phrase_markup(
    projection: ToolDisplayProjection,
    *,
    completed: bool,
    verb: str,
) -> str:
    if completed and _basic_database_sql_verb(projection) == "SELECT":
        return _basic_database_select_phrase_markup(projection)

    phrase = " ".join(
        part
        for part in (
            _basic_database_bold_markup(f"{verb} SQL"),
            _basic_database_dim_markup("statement"),
            _basic_database_sql_preview_markup(projection),
            _basic_database_scope_markup(projection),
        )
        if part
    )
    row_count = _basic_projection_integer_value(projection, "rows")
    if completed and row_count is not None:
        rows = _basic_count_label(row_count, "row")
        phrase = f"{phrase}: {_basic_database_bold_markup(rows)}"
    return phrase


def _basic_database_select_phrase_markup(
    projection: ToolDisplayProjection,
) -> str:
    phrase = " ".join(
        part
        for part in (
            _basic_database_bold_markup("Executed query"),
            _basic_database_sql_preview_markup(projection),
            _basic_database_scope_markup(projection),
        )
        if part
    )
    row_count = _basic_projection_integer_value(projection, "rows")
    if row_count == 0:
        return f"{phrase}: {_basic_markup_summary('no results')}."
    if row_count is not None:
        rows = _basic_count_label(row_count, "row")
        return (
            f"{phrase}: {_basic_database_bold_markup(rows)} "
            f"{_basic_markup_summary('found')}."
        )
    return phrase


def _basic_database_inspect_phrase_markup(
    projection: ToolDisplayProjection,
    *,
    verb: str,
) -> str:
    table_names = _basic_database_table_names(projection)
    table_count = _basic_database_table_count(projection, table_names)
    if table_count == 0:
        table_names = None
    subject = (
        f"{verb} {_basic_count_label(table_count, 'table')}"
        if table_count is not None
        else f"{verb} tables"
    )
    table_phrase = _basic_database_bold_markup(subject)
    if table_names:
        table_phrase = f"{table_phrase}: {_basic_markup_summary(table_names)}"
    return " ".join(
        part
        for part in (
            table_phrase,
            _basic_database_scope_markup(projection, preposition="from"),
        )
        if part
    )


def _basic_database_tables_phrase_markup(
    projection: ToolDisplayProjection,
    *,
    completed: bool,
    verb: str,
) -> str:
    table_count = _basic_projection_integer_value(projection, "tables")
    if completed and table_count is not None:
        subject = _basic_database_bold_markup(
            f"{verb} {_basic_count_label(table_count, 'table')}"
        )
    else:
        subject = (
            f"{_basic_database_bold_markup(verb)} "
            f"{_basic_database_dim_markup('tables')}"
        )
    return " ".join(
        part
        for part in (
            subject,
            _basic_database_scope_markup(projection),
        )
        if part
    )


def _basic_database_target_markup(
    operation: str,
    projection: ToolDisplayProjection,
) -> str:
    if operation in {"plan", "run"}:
        return _basic_database_statement_markup(projection)
    description, include_target = _BASIC_DATABASE_TARGET_DESCRIPTIONS.get(
        operation,
        (projection.target or "database", False),
    )
    target = projection.target
    if include_target and target and target != description:
        return (
            f"{_basic_database_dim_markup(description)} "
            f"{_basic_database_bold_markup(target)}"
        )
    return _basic_database_dim_markup(description)


def _basic_database_statement_markup(
    projection: ToolDisplayProjection,
) -> str:
    sql_verb = _basic_database_sql_verb(projection)
    if not sql_verb:
        return _basic_database_dim_markup("SQL statement")
    return (
        f"{_basic_database_bold_markup(sql_verb)} "
        f"{_basic_database_dim_markup('statement')}"
    )


def _basic_database_sql_preview_markup(
    projection: ToolDisplayProjection,
) -> str | None:
    sql = _basic_projection_detail_value(projection, "sql")
    if not sql:
        return None
    if len(sql) > _BASIC_DATABASE_SQL_PREVIEW_LIMIT:
        sql = f"{sql[: _BASIC_DATABASE_SQL_PREVIEW_LIMIT - 4].rstrip()} ..."
    return _basic_markup_summary(sql)


def _basic_database_scope_markup(
    projection: ToolDisplayProjection,
    *,
    preposition: str = "in",
) -> str:
    database_name = _basic_database_identity(projection)
    if not database_name:
        return _basic_database_dim_markup(f"{preposition} database")
    return (
        f"{_basic_database_dim_markup(f'{preposition} database')} "
        f"{_basic_markup_summary(database_name)}"
    )


def _basic_database_identity(
    projection: ToolDisplayProjection,
) -> str | None:
    for detail in projection.details:
        if detail.label not in _BASIC_DATABASE_IDENTITY_DETAIL_LABELS:
            continue
        if detail.redacted or detail.value is None:
            continue
        value = _basic_summary(str(detail.value))
        if value:
            return value
    return None


def _basic_database_sql_verb(
    projection: ToolDisplayProjection,
) -> str | None:
    sql_command = _basic_projection_detail_value(projection, "sql_command")
    if sql_command:
        return sql_command.upper()
    sql = None
    if projection.preview and projection.preview.label == "sql":
        sql = projection.preview.content
    if not sql:
        sql = _basic_projection_detail_value(projection, "sql")
    if not sql:
        return None
    first = sql.lstrip().split(maxsplit=1)[0].strip('([`"').upper()
    return first or None


def _basic_database_table_names(
    projection: ToolDisplayProjection,
) -> str | None:
    tables = _basic_projection_detail_value(projection, "tables")
    if tables:
        return tables
    target = _basic_summary(projection.target)
    if not target or target in {"database", "tables"}:
        return None
    return target


def _basic_database_table_count(
    projection: ToolDisplayProjection,
    table_names: str | None,
) -> int | None:
    count = _basic_projection_integer_value(projection, "tables")
    if count is not None:
        return count
    if not table_names:
        return None
    return len(
        [
            table_name
            for table_name in table_names.split(", ")
            if table_name and table_name != "..."
        ]
    )


def _basic_projection_detail_value(
    projection: ToolDisplayProjection,
    label: str,
) -> str | None:
    for detail in projection.details:
        if detail.label != label:
            continue
        if detail.redacted or detail.value is None:
            return None
        return _basic_summary(str(detail.value))
    return None


def _basic_projection_integer_value(
    projection: ToolDisplayProjection,
    label: str,
) -> int | None:
    metric = projection.metrics.get(label)
    value = _basic_integer_value(metric)
    if value is not None:
        return value
    for detail in projection.details:
        if detail.label != label or detail.redacted:
            continue
        value = _basic_integer_value(detail.value)
        if value is not None:
            return value
    return None


def _basic_integer_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _basic_count_label(value: int, singular: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {singular}{suffix}"


def _basic_database_bold_markup(value: str) -> str:
    return (
        f"[{_BASIC_TOOL_NAME_STYLE}]{_basic_markup_summary(value)}"
        f"[/{_BASIC_TOOL_NAME_STYLE}]"
    )


def _basic_database_dim_markup(value: str) -> str:
    return (
        f"[{_BASIC_TOOL_ARGUMENT_STYLE}]{_basic_markup_summary(value)}"
        f"[/{_BASIC_TOOL_ARGUMENT_STYLE}]"
    )


def _basic_tool_subject_markup(subject: str) -> str:
    summary = _basic_summary(subject)
    if not summary:
        return ""
    name, separator, arguments = summary.partition(" ")
    if not separator:
        return (
            f"[{_BASIC_TOOL_NAME_STYLE}]"
            f"{escape(name)}"
            f"[/{_BASIC_TOOL_NAME_STYLE}]"
        )
    return (
        f"[{_BASIC_TOOL_NAME_STYLE}]"
        f"{escape(name)}"
        f"[/{_BASIC_TOOL_NAME_STYLE}]"
        f" [{_BASIC_TOOL_ARGUMENT_STYLE}]"
        f"{escape(arguments)}"
        f"[/{_BASIC_TOOL_ARGUMENT_STYLE}]"
    )


def _basic_projection_subject_markup(
    projection: ToolDisplayProjection,
) -> str:
    assert isinstance(projection, ToolDisplayProjection)
    action = projection.action
    target = projection.target
    scope = projection.scope
    if action == "run" and target:
        subject = target
    else:
        subject = action
        if target:
            subject = f"{subject} {target}"
    if scope and not _basic_is_default_tool_scope(scope) and scope != target:
        subject = f"{subject} in {scope}"
    return _basic_tool_subject_markup(subject)


def _basic_is_default_tool_scope(scope: str) -> bool:
    assert isinstance(scope, str)
    return scope.strip() in {"", "."}


def _basic_elapsed_suffix(elapsed_seconds: float | None) -> str:
    elapsed = _basic_tool_elapsed_text(elapsed_seconds)
    if not elapsed:
        return ""
    return (
        f" [{_BASIC_TOOL_ELAPSED_STYLE}]·"
        f" {escape(elapsed)}[/{_BASIC_TOOL_ELAPSED_STYLE}]"
    )


def _basic_tool_terminal_markup(
    status: str,
    display_status: str,
    *,
    display_projection: ToolDisplayProjection | None,
) -> str | None:
    if display_status in _BASIC_TOOL_SUCCESS_STATUSES:
        return None
    if display_projection is None:
        return _basic_markup_summary(status)
    status_text = _basic_tool_status_outcome_markup(
        display_status,
        projection_outcome(display_projection),
    )
    summary = projection_summary_markup(display_projection)
    terminal = " - ".join(part for part in (status_text, summary) if part)
    return terminal or _basic_markup_summary(status)


def _basic_tool_status_outcome_markup(
    status: str | None,
    outcome: str | None,
) -> str | None:
    if not status and not outcome:
        return None
    if status and outcome and status != outcome:
        return (
            f"{_basic_markup_summary(status)} {_basic_markup_summary(outcome)}"
        )
    return _basic_markup_summary(outcome or status)


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
    return any(
        entry.executed
        for entry in _basic_tool_entries(request, include_active=False)
    )


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
        if (
            not self._emitted_visible_answer_text
            and structured_answer_started(visible_answer_text)
            and not _basic_terminal_completed(request)
        ):
            return
        if (
            _basic_terminal_completed(request)
            and not self._emitted_visible_answer_text
        ):
            visible_answer_text = (
                pretty_json_answer_text(visible_answer_text)
                or visible_answer_text
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
