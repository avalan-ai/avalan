from ...event import EventStats
from . import Theme
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

from rich.console import Group, RenderableType
from rich.spinner import Spinner

_BASIC_SUMMARY_LIMIT = 160
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


class BasicTheme(Theme):
    """Provide a low-clutter theme using common display defaults."""

    def stream_presenter(
        self,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
    ) -> CliStreamPresenter:
        return BasicStreamPresenter(logger, event_stats=event_stats)


class BasicStreamPresenter:
    """Present immutable stream snapshots with Basic answer-first output."""

    requires_completion_snapshot: bool = True
    supports_stderr_diagnostics: bool = True

    def __init__(
        self,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
    ) -> None:
        assert isinstance(logger, Logger)
        assert event_stats is None or isinstance(event_stats, EventStats)
        self._answer_presenter = _BasicAnswerPresenter()
        self._event_stats = event_stats
        self._logger = logger
        self._final_newline_emitted = False
        self._last_visible_answer_text = ""
        self._visible_roles: set[StreamFrameRole] = set()

    def reset(self) -> None:
        """Forget emitted answer text, newline, and visible frames."""
        self._answer_presenter.reset()
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
            self._last_visible_answer_text += chunk.text
            yield chunk

        newline = self._terminal_newline(request)
        if newline is not None:
            yield newline

        if request.mode == "answer":
            return

        role_renderables: tuple[
            tuple[StreamFrameRole, RenderableType | None],
            ...,
        ] = (
            ("tools", _basic_tool_frame(request)),
            ("events", _basic_event_frame(request)),
            ("stats", _basic_stats_frame(request)),
        )
        for role, renderable in role_renderables:
            frame = self._role_frame(role, renderable)
            if frame is not None:
                yield frame

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
    history_lines = [
        *(
            _basic_completed_tool_line(tool.name, tool.status)
            for tool in snapshot.completed_tools
        ),
        *(
            _basic_tool_result_line(
                result.name,
                result.status,
                result.result_summary,
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
        ),
    ]
    limit = request.display_config.display_tools_events
    if limit is not None:
        history_lines = history_lines[-limit:] if limit else []

    active_renderables = [
        _basic_active_tool_renderable(
            tool.name,
            tool.arguments_summary,
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
    arguments_summary: str | None,
) -> str:
    suffix = (
        f": {_basic_summary(arguments_summary)}" if arguments_summary else ""
    )
    return f"tool {_basic_summary(name)} running{suffix}"


def _basic_active_tool_renderable(
    name: str,
    arguments_summary: str | None,
    *,
    spinner: bool,
) -> RenderableType:
    line = _basic_active_tool_line(name, arguments_summary)
    if not spinner:
        return line
    return Spinner("dots", text=line)


def _basic_completed_tool_line(name: str, status: str) -> str:
    return f"tool {_basic_summary(name)} {status}"


def _basic_tool_result_line(
    name: str,
    status: str,
    result_summary: str,
) -> str:
    return (
        f"tool {_basic_summary(name)} {status}: "
        f"{_basic_summary(result_summary)}"
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


def _basic_usage_line(kind: str | None, usage_summary: str) -> str:
    label = _basic_summary(kind or "usage")
    return f"usage {label}: {_basic_summary(usage_summary)}"


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
