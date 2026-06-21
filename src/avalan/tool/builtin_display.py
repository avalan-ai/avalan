from ..entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
)
from ..memory.permanent import Memory, PermanentMemoryStore
from .display import (
    REDACTED_DISPLAY_VALUE,
    ToolDisplayDetail,
    ToolDisplayPreview,
    ToolDisplayProjection,
)

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeAlias, TypeGuard
from urllib.parse import urlsplit, urlunsplit

DisplayScalar: TypeAlias = None | bool | int | float | str

_DISPLAY_LIST_LIMIT = 8


def project_calculator_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="calculate",
            scope="math",
            completed_summary="Calculation completed.",
            result_projector=_calculator_result_projection,
        )
    arguments = _arguments(call)
    expression = _string_argument(arguments, "expression")
    return ToolDisplayProjection(
        action="calculate",
        label=call.name,
        target=expression or "expression",
        scope="math",
        summary="Calculate arithmetic expression.",
        details=(
            _detail("expression", expression),
            _detail("expression_chars", len(expression or "")),
        ),
    )


def project_mcp_call_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="call",
            scope="MCP",
            completed_summary="MCP tool call completed.",
            result_projector=_mcp_result_projection,
        )
    arguments = _arguments(call)
    uri, uri_redacted = _safe_url(_string_argument(arguments, "uri"))
    tool_name = _string_argument(arguments, "name")
    tool_arguments = arguments.get("arguments")
    argument_count = (
        len(tool_arguments) if isinstance(tool_arguments, dict) else 0
    )
    argument_keys = (
        _joined_limited_strings(tool_arguments.keys())
        if isinstance(tool_arguments, dict)
        else None
    )
    return ToolDisplayProjection(
        action="call",
        label=call.name,
        target=tool_name or "MCP tool",
        scope=uri or "MCP server",
        summary="Call an MCP server tool.",
        details=(
            _detail("uri", uri, redacted=uri_redacted),
            _detail("tool", tool_name),
            _detail("argument_count", argument_count),
            _detail("argument_keys", argument_keys),
        ),
        metrics={"arguments": argument_count},
        redacted=uri_redacted,
    )


def project_browser_open_tool_display(
    *,
    call: ToolCall,
    settings: Mapping[str, object] | None = None,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="open",
            scope="browser",
            completed_summary="Browser page opened.",
            result_projector=_text_result_projection,
        )
    arguments = _arguments(call)
    url, url_redacted = _safe_url(_string_argument(arguments, "url"))
    settings = settings or {}
    viewport = _viewport(settings)
    return ToolDisplayProjection(
        action="open",
        label=call.name,
        target=url or "URL",
        scope="browser",
        summary="Open a browser page.",
        details=(
            _detail("url", url, redacted=url_redacted),
            _detail("engine", _string_mapping_value(settings, "engine")),
            _detail("viewport", viewport),
            _detail("search", _bool_mapping_value(settings, "search")),
        ),
        redacted=url_redacted,
    )


def project_code_run_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="run",
            scope="restricted Python",
            completed_summary="Python code completed.",
            result_projector=_text_result_projection,
        )
    arguments = _arguments(call)
    code = _string_argument(arguments, "code")
    return ToolDisplayProjection(
        action="run",
        label=call.name,
        target="run",
        scope="restricted Python",
        summary="Run restricted Python code.",
        details=(
            _detail("language", "python"),
            _detail("code_chars", len(code or "")),
            _detail("arguments", max(len(arguments) - 1, 0)),
        ),
        metrics={"code_chars": len(code or "")},
    )


def project_ast_grep_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action=_ast_grep_action(_arguments(call)),
            scope="code",
            completed_summary="Code search completed.",
            result_projector=_text_result_projection,
        )
    arguments = _arguments(call)
    pattern = _string_argument(arguments, "pattern")
    paths, paths_redacted = _safe_paths(arguments.get("paths"))
    rewrite = _string_argument(arguments, "rewrite")
    return ToolDisplayProjection(
        action=_ast_grep_action(arguments),
        label=call.name,
        target=pattern or "code pattern",
        scope=paths or "workspace",
        summary="Search or rewrite code with ast-grep.",
        details=(
            _detail("pattern", pattern),
            _detail("language", _string_argument(arguments, "lang")),
            _detail("paths", paths, redacted=paths_redacted),
            _detail("rewrite", rewrite is not None),
            _detail("rewrite_chars", len(rewrite or "")),
        ),
        redacted=paths_redacted,
    )


def project_search_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="search",
            scope="internet",
            completed_summary="Search completed.",
            result_projector=_text_result_projection,
        )
    arguments = _arguments(call)
    query = _string_argument(arguments, "query")
    engine = _string_argument(arguments, "engine")
    return ToolDisplayProjection(
        action="search",
        label=call.name,
        target=query or "query",
        scope=engine or "search engine",
        summary="Search for real-time information.",
        details=(
            _detail("query", query),
            _detail("engine", engine),
        ),
    )


def project_memory_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    operation = _memory_operation(call.name)
    if outcome is not None:
        return _memory_terminal_projection(
            call=call,
            outcome=outcome,
            operation=operation,
        )
    arguments = _arguments(call)
    namespace = _string_argument(arguments, "namespace")
    search = _string_argument(arguments, "search")
    action = "list" if operation in {"list", "stores"} else "search"
    target = (
        "memory stores"
        if operation == "stores"
        else namespace or search or "memory"
    )
    scope = (
        "permanent memory stores"
        if operation == "stores"
        else (
            "messages"
            if operation == "message.read"
            else namespace or "permanent memory"
        )
    )
    details = [
        _detail("operation", operation),
        _detail("namespace", namespace),
        _detail("search", search),
    ]
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=target,
        scope=scope,
        summary=_memory_summary(operation),
        details=tuple(details),
    )


def project_graph_tool_display(
    *,
    call: ToolCall,
    settings: Mapping[str, object] | None = None,
    outcome: ToolCallOutcome | None = None,
) -> ToolDisplayProjection:
    assert isinstance(call, ToolCall)
    chart_type = _graph_chart_type(call.name)
    if outcome is not None:
        return _terminal_projection(
            call=call,
            outcome=outcome,
            action="render",
            scope="graph",
            completed_summary="Graph rendered.",
            result_projector=_graph_result_projection,
        )
    arguments = _arguments(call)
    title = _string_argument(arguments, "title")
    output_format = _string_argument(arguments, "output_format") or "png"
    point_count = _graph_point_count(chart_type, arguments)
    series_count = _graph_series_count(chart_type, arguments)
    file_path, file_redacted = _safe_path(
        _string_mapping_value(settings or {}, "file")
    )
    return ToolDisplayProjection(
        action="render",
        label=call.name,
        target=title or f"{chart_type} chart",
        scope="graph",
        summary=f"Render {chart_type} chart.",
        details=(
            _detail("chart_type", chart_type),
            _detail("title", title),
            _detail("format", output_format),
            _detail("width", _number_argument(arguments, "width")),
            _detail("height", _number_argument(arguments, "height")),
            _detail("dpi", _int_argument(arguments, "dpi")),
            _detail("file", file_path, redacted=file_redacted),
        ),
        metrics={
            "points": point_count,
            "series": series_count,
        },
        redacted=file_redacted,
    )


def _terminal_projection(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome,
    action: str,
    scope: str,
    completed_summary: str,
    result_projector: "ResultProjector",
) -> ToolDisplayProjection:
    if isinstance(outcome, ToolCallResult):
        return result_projector(
            call,
            outcome.result,
            action,
            scope,
            completed_summary,
        )
    if isinstance(outcome, ToolCallError):
        return _error_projection(call, outcome, action=action, scope=scope)
    assert isinstance(outcome, ToolCallDiagnostic)
    return _diagnostic_projection(call, outcome, action=action, scope=scope)


ResultProjector: TypeAlias = Callable[
    [ToolCall, object, str, str, str],
    ToolDisplayProjection,
]


def _calculator_result_projection(
    call: ToolCall,
    result: object,
    action: str,
    scope: str,
    completed_summary: str,
) -> ToolDisplayProjection:
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=_string_argument(_arguments(call), "expression")
        or "expression",
        scope=scope,
        summary=completed_summary,
        status="completed",
        outcome="result",
        details=(
            _detail("result", str(result) if result is not None else None),
        ),
        preview=(
            ToolDisplayPreview(label="result", content=str(result))
            if result is not None
            else None
        ),
    )


def _mcp_result_projection(
    call: ToolCall,
    result: object,
    action: str,
    scope: str,
    completed_summary: str,
) -> ToolDisplayProjection:
    count = _result_count(result)
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=_string_argument(_arguments(call), "name") or "MCP tool",
        scope=scope,
        summary=completed_summary,
        status="completed",
        outcome="result",
        details=(
            _detail("result_type", type(result).__name__),
            _detail("items", count),
        ),
        metrics={"items": count},
    )


def _text_result_projection(
    call: ToolCall,
    result: object,
    action: str,
    scope: str,
    completed_summary: str,
) -> ToolDisplayProjection:
    text_chars = len(result) if isinstance(result, str) else None
    text_lines = result.count("\n") + 1 if isinstance(result, str) else None
    count = _result_count(result)
    target, target_redacted = _result_target(call)
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=target,
        scope=scope,
        summary=completed_summary,
        status="completed",
        outcome="result",
        details=(
            _detail("result_type", type(result).__name__),
            _detail("items", count),
            _detail("text_chars", text_chars),
            _detail("text_lines", text_lines),
        ),
        metrics=_compact_metrics(
            {
                "items": count,
                "text_chars": text_chars,
                "text_lines": text_lines,
            }
        ),
        redacted=target_redacted,
    )


def _memory_terminal_projection(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome,
    operation: str,
) -> ToolDisplayProjection:
    if isinstance(outcome, ToolCallError):
        return _error_projection(
            call,
            outcome,
            action="list" if operation in {"list", "stores"} else "search",
            scope="memory",
        )
    if isinstance(outcome, ToolCallDiagnostic):
        return _diagnostic_projection(
            call,
            outcome,
            action="skip",
            scope="memory",
        )
    assert isinstance(outcome, ToolCallResult)
    result = outcome.result
    metrics, details, summary, result_outcome = _memory_result_facts(
        operation,
        result,
    )
    return ToolDisplayProjection(
        action="list" if operation in {"list", "stores"} else "search",
        label=call.name,
        target=_memory_result_target(call, operation),
        scope="memory",
        summary=summary,
        status="completed",
        outcome=result_outcome,
        details=(
            _detail("operation", operation),
            *details,
        ),
        metrics=metrics,
    )


def _graph_result_projection(
    call: ToolCall,
    result: object,
    action: str,
    scope: str,
    completed_summary: str,
) -> ToolDisplayProjection:
    if not isinstance(result, Mapping):
        return _text_result_projection(
            call,
            result,
            action,
            scope,
            completed_summary,
        )
    file_path, file_redacted = _safe_path(
        _string_mapping_value(result, "file")
    )
    chart_type = _string_mapping_value(result, "chart_type")
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=_string_mapping_value(result, "title") or chart_type or "graph",
        scope=scope,
        summary=completed_summary,
        status="completed",
        outcome=_string_mapping_value(result, "format") or "rendered",
        details=(
            _detail("chart_type", chart_type),
            _detail("format", _string_mapping_value(result, "format")),
            _detail("mime_type", _string_mapping_value(result, "mime_type")),
            _detail("width", _number_mapping_value(result, "width")),
            _detail("height", _number_mapping_value(result, "height")),
            _detail("dpi", _int_mapping_value(result, "dpi")),
            _detail("file", file_path, redacted=file_redacted),
        ),
        metrics=_compact_metrics(
            {
                "points": _int_mapping_value(result, "points"),
                "series": _sequence_count(result.get("series")),
            }
        ),
        redacted=file_redacted,
    )


def _error_projection(
    call: ToolCall,
    error: ToolCallError,
    *,
    action: str,
    scope: str,
) -> ToolDisplayProjection:
    target, target_redacted = _result_target(call)
    return ToolDisplayProjection(
        action=action,
        label=call.name,
        target=target,
        scope=scope,
        summary="Tool call failed.",
        status="error",
        outcome=error.error_type,
        severity="error",
        details=(_detail("error_type", error.error_type),),
        redacted=target_redacted,
    )


def _diagnostic_projection(
    call: ToolCall,
    diagnostic: ToolCallDiagnostic,
    *,
    action: str,
    scope: str,
) -> ToolDisplayProjection:
    target, target_redacted = _result_target(call)
    return ToolDisplayProjection(
        action="skip" if action != "skip" else action,
        label=call.name,
        target=target,
        scope=scope,
        summary="Tool call was not executed.",
        status=diagnostic.status.value,
        outcome=diagnostic.code.value,
        severity="warning",
        details=(
            _detail("stage", diagnostic.stage.value),
            _detail("retryable", diagnostic.retryable),
        ),
        redacted=target_redacted,
    )


def _memory_result_facts(
    operation: str,
    result: object,
) -> tuple[
    dict[str, DisplayScalar],
    tuple[ToolDisplayDetail, ...],
    str,
    str,
]:
    if operation == "message.read":
        found = result != "NOT_FOUND"
        text_chars = len(result) if isinstance(result, str) and found else 0
        return (
            {"matches": 1 if found else 0, "text_chars": text_chars},
            (
                _detail("matches", 1 if found else 0),
                _detail("text_chars", text_chars),
            ),
            (
                "Message memory matched."
                if found
                else "No matching message found."
            ),
            "result" if found else "not_found",
        )
    if operation == "stores":
        stores = (
            [item for item in result if isinstance(item, PermanentMemoryStore)]
            if _is_sequence(result)
            else []
        )
        namespaces = _joined_limited_strings(
            store.namespace for store in stores
        )
        return (
            {"stores": len(stores)},
            (
                _detail("stores", len(stores)),
                _detail("namespaces", namespaces),
            ),
            "Memory stores listed.",
            "result",
        )
    if operation == "list":
        memories = (
            [item for item in result if isinstance(item, Memory)]
            if _is_sequence(result)
            else []
        )
        partitions = sum(memory.partitions for memory in memories)
        types = _joined_limited_strings(
            memory.type.value for memory in memories
        )
        return (
            {"memories": len(memories), "partitions": partitions},
            (
                _detail("memories", len(memories)),
                _detail("partitions", partitions),
                _detail("types", types),
            ),
            "Memories listed.",
            "result",
        )
    count = _result_count(result)
    partition_text_chars: int | None = None
    if _is_sequence(result):
        partition_text_chars = 0
        for item in result:
            if isinstance(item, str):
                partition_text_chars += len(item)
    return (
        _compact_metrics(
            {
                "matches": count,
                "text_chars": partition_text_chars,
            }
        ),
        (
            _detail("matches", count),
            _detail("text_chars", partition_text_chars),
        ),
        "Permanent memory search completed.",
        "result",
    )


def _memory_result_target(call: ToolCall, operation: str) -> str:
    arguments = _arguments(call)
    if operation == "stores":
        return "memory stores"
    return (
        _string_argument(arguments, "namespace")
        or _string_argument(arguments, "search")
        or "memory"
    )


def _result_target(call: ToolCall) -> tuple[str, bool]:
    arguments = _arguments(call)
    for key in ("expression", "name", "url", "query", "namespace", "title"):
        value = _string_argument(arguments, key)
        if value:
            if key == "url":
                safe_url, redacted = _safe_url(value)
                return safe_url or "URL", redacted
            return value, False
    return call.name, False


def _arguments(call: ToolCall) -> Mapping[str, object]:
    return call.arguments if isinstance(call.arguments, Mapping) else {}


def _detail(
    label: str,
    value: DisplayScalar,
    *,
    redacted: bool = False,
) -> ToolDisplayDetail:
    if redacted:
        value = REDACTED_DISPLAY_VALUE
    return ToolDisplayDetail(label=label, value=value, redacted=redacted)


def _compact_metrics(
    metrics: Mapping[str, DisplayScalar],
) -> dict[str, DisplayScalar]:
    return {key: value for key, value in metrics.items() if value is not None}


def _string_argument(
    arguments: Mapping[str, object],
    key: str,
) -> str | None:
    value = arguments.get(key)
    return value if isinstance(value, str) else None


def _number_argument(
    arguments: Mapping[str, object],
    key: str,
) -> int | float | None:
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


def _int_argument(
    arguments: Mapping[str, object],
    key: str,
) -> int | None:
    value = arguments.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _string_mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _bool_mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> bool | None:
    value = mapping.get(key)
    return value if isinstance(value, bool) else None


def _number_mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> int | float | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


def _int_mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> int | None:
    value = mapping.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _safe_url(value: str | None) -> tuple[str | None, bool]:
    if not value:
        return None, False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return REDACTED_DISPLAY_VALUE, True
    redacted = bool(
        parsed.username or parsed.password or parsed.query or parsed.fragment
    )
    if parsed.netloc:
        host = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
        try:
            port = parsed.port
        except ValueError:
            port = None
            redacted = True
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{port}" if port else host
        safe = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        return safe, redacted
    if "?" in value or "#" in value:
        return value.split("?", 1)[0].split("#", 1)[0], True
    return value, False


def _safe_path(value: str | None) -> tuple[str | None, bool]:
    if not value:
        return None, False
    if (
        value.startswith(("/", "~"))
        or "\\" in value
        or "$" in value
        or ".." in value.split("/")
    ):
        return REDACTED_DISPLAY_VALUE, True
    return value, False


def _safe_paths(value: object) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, str):
        return _safe_path(value)
    if not _is_sequence(value):
        return None, False
    redacted = False
    paths: list[str] = []
    for index, item in enumerate(value):
        if index >= _DISPLAY_LIST_LIMIT:
            paths.append("...")
            break
        if not isinstance(item, str):
            continue
        path, path_redacted = _safe_path(item)
        redacted = redacted or path_redacted
        if path:
            paths.append(path)
    return ", ".join(paths) if paths else None, redacted


def _joined_limited_strings(values: object) -> str | None:
    if isinstance(values, str | bytes | bytearray):
        return None
    if not isinstance(values, Iterable):
        return None
    strings: list[str] = []
    for index, value in enumerate(values):
        if index >= _DISPLAY_LIST_LIMIT:
            strings.append("...")
            break
        if isinstance(value, str):
            strings.append(value)
    return ", ".join(strings) if strings else None


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _sequence_count(value: object) -> int | None:
    return len(value) if _is_sequence(value) else None


def _result_count(result: object) -> int:
    if _is_sequence(result):
        return len(result)
    if isinstance(result, Mapping):
        return len(result)
    return 1 if result is not None else 0


def _viewport(settings: Mapping[str, object]) -> str | None:
    width = _int_mapping_value(settings, "viewport_width")
    height = _int_mapping_value(settings, "viewport_height")
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def _ast_grep_action(arguments: Mapping[str, object]) -> str:
    return "rewrite" if _string_argument(arguments, "rewrite") else "search"


def _memory_operation(name: str) -> str:
    if name.endswith("message.read"):
        return "message.read"
    return name.rsplit(".", 1)[-1]


def _memory_summary(operation: str) -> str:
    match operation:
        case "message.read":
            return "Search previous messages."
        case "read":
            return "Search permanent memory."
        case "list":
            return "List permanent memories."
        case "stores":
            return "List permanent memory stores."
        case _:
            return "Use memory tool."


def _graph_chart_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _graph_series_count(
    chart_type: str,
    arguments: Mapping[str, object],
) -> int:
    if chart_type in {"bar", "line"}:
        series = arguments.get("series")
        if isinstance(series, Mapping):
            return len(series)
        return 1 if _is_sequence(arguments.get("values")) else 0
    return 1


def _graph_point_count(
    chart_type: str,
    arguments: Mapping[str, object],
) -> int:
    match chart_type:
        case "pie":
            return _sequence_count(arguments.get("values")) or 0
        case "bar":
            categories = _sequence_count(arguments.get("categories")) or 0
            return categories * _graph_series_count(chart_type, arguments)
        case "line":
            x_labels = _sequence_count(arguments.get("x_labels")) or 0
            return x_labels * _graph_series_count(chart_type, arguments)
        case "scatter":
            return min(
                _sequence_count(arguments.get("x")) or 0,
                _sequence_count(arguments.get("y")) or 0,
            )
        case "histogram":
            return _sequence_count(arguments.get("values")) or 0
        case _:
            return 0
