from ...entities import ToolCall, ToolCallOutcome, ToolCallResult
from ..display import (
    REDACTED_DISPLAY_VALUE,
    ToolDisplayDetail,
    ToolDisplayPreview,
    ToolDisplayProjection,
)
from .commands.helpers import is_denied_display_path
from .entities import (
    ExecutionResult,
    GeneratedFile,
    PathOperand,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellCompositionResult,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellFormattedCompositionResult,
    ShellFormattedResult,
)
from .registry import SHELL_COMMAND_DEFINITIONS

from collections.abc import Mapping, Sequence
from shlex import join as shell_join

_REQUEST_ACTIONS = {
    "rg": "search",
    "head": "read",
    "tail": "read",
    "ls": "list",
    "cat": "read",
    "nl": "number",
    "file": "identify",
    "find": "find",
    "wc": "count",
    "awk": "select",
    "sed": "select",
    "jq": "transform",
    "pdfinfo": "inspect",
    "pdftotext": "extract",
    "pdftoppm": "rasterize",
    "tesseract": "recognize",
}
_REQUEST_SUMMARIES = {
    "rg": "Search for a pattern.",
    "head": "Read the first part of a file.",
    "tail": "Read the last part of a file.",
    "ls": "List a path.",
    "cat": "Read a file.",
    "nl": "Number file lines.",
    "file": "Identify file types.",
    "find": "Find workspace entries.",
    "wc": "Count file content.",
    "awk": "Select fields or lines.",
    "sed": "Select line ranges or patterns.",
    "jq": "Transform JSON.",
    "pdfinfo": "Inspect PDF metadata.",
    "pdftotext": "Extract PDF text.",
    "pdftoppm": "Rasterize PDF pages.",
    "tesseract": "Recognize image text.",
}
_PATH_TARGET_COMMANDS = frozenset(
    (
        "head",
        "tail",
        "ls",
        "cat",
        "nl",
        "file",
        "wc",
        "awk",
        "pdfinfo",
        "pdftotext",
        "pdftoppm",
        "tesseract",
    )
)
_GENERATED_OUTPUT_PATH_MARKERS = (
    "/tmp/",
    "/private/tmp/",
    "/var/folders/",
    "/private/var/folders/",
    "avalan-shell-",
)


def project_shell_tool_display(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome | None = None,
    request: ShellCommandRequest | ShellCompositionRequest | None = None,
) -> ToolDisplayProjection | None:
    assert isinstance(call, ToolCall)
    if outcome is not None:
        composition_result = _composition_result_from_outcome(outcome)
        if composition_result is not None:
            return project_shell_composition_result(composition_result)
        result = _execution_result_from_outcome(outcome)
        if result is None:
            return None
        return project_shell_execution_result(result)
    if request is None:
        return None
    if isinstance(request, ShellCompositionRequest):
        return project_shell_composition_request(request)
    return project_shell_command_request(request)


def project_shell_command_request(
    request: ShellCommandRequest,
) -> ToolDisplayProjection:
    assert isinstance(request, ShellCommandRequest)
    action = _REQUEST_ACTIONS.get(request.command, "run")
    target, target_redacted = _request_target(request)
    scope, scope_redacted = _request_scope(request)
    output_details = _request_output_details(request)
    details = (
        *_request_details(request),
        *_request_limit_details(request),
        *output_details,
    )
    metrics = _request_metrics(request)
    return ToolDisplayProjection(
        action=action,
        label=request.tool_name,
        target=target,
        scope=scope,
        summary=_REQUEST_SUMMARIES.get(request.command, "Run a command."),
        details=details,
        metrics=metrics,
        redacted=target_redacted or scope_redacted,
    )


def project_shell_composition_request(
    request: ShellCompositionRequest,
) -> ToolDisplayProjection:
    assert isinstance(request, ShellCompositionRequest)
    target, target_redacted = _composition_request_stage_chain(request.steps)
    scope, scope_redacted = _composition_request_scope(request.steps)
    details = (
        _detail("mode", request.mode),
        _detail("stage chain", target, redacted=target_redacted),
        *(_composition_scope_details(scope, scope_redacted)),
        *(_composition_request_limit_details(request)),
    )
    return ToolDisplayProjection(
        action="pipeline",
        label="shell.pipeline",
        target=target,
        scope=scope,
        summary="Run a structured shell pipeline.",
        details=details,
        metrics=_composition_request_metrics(request),
        redacted=target_redacted or scope_redacted,
    )


def project_shell_execution_result(
    result: ExecutionResult,
) -> ToolDisplayProjection:
    assert isinstance(result, ExecutionResult)
    generated_file = (
        result.generated_files[0] if result.generated_files else None
    )
    target, target_redacted = _result_target(result)
    scope, scope_redacted = _result_scope(result)
    details = (
        *_result_fact_details(result),
        *_generated_file_details(generated_file),
    )
    return ToolDisplayProjection(
        action="run",
        label=result.tool_name,
        target=target,
        scope=scope,
        summary=_result_summary(result),
        status=result.status.value,
        outcome=result.status.value,
        severity=_result_severity(result.status),
        details=details,
        metrics=_result_metrics(result),
        preview=_generated_file_preview(generated_file),
        redacted=target_redacted or scope_redacted,
    )


def project_shell_composition_result(
    result: ShellCompositionResult,
) -> ToolDisplayProjection:
    assert isinstance(result, ShellCompositionResult)
    target = _composition_result_stage_chain(result.steps)
    scope, scope_redacted = _composition_result_scope(result.steps)
    details = [
        _detail("status", result.status.value),
        _detail("mode", result.mode),
        _detail("stage statuses", _composition_stage_statuses(result.steps)),
        _detail("duration ms", result.duration_ms),
        _detail("stdout bytes", result.stdout_bytes),
        _detail("stdout truncated", result.stdout_truncated),
        _detail("stderr bytes", result.stderr_bytes),
        _detail("stderr truncated", result.stderr_truncated),
    ]
    if result.error_code is not None:
        details.append(_detail("error code", result.error_code.value))
    if result.error_message is not None:
        details.append(_detail("error message", result.error_message))
    return ToolDisplayProjection(
        action="pipeline",
        label="shell.pipeline",
        target=target,
        scope=scope,
        summary=_composition_result_summary(result),
        status=result.status.value,
        outcome=result.status.value,
        severity=_result_severity(result.status),
        details=details,
        metrics=_composition_result_metrics(result),
        redacted=scope_redacted,
    )


def _execution_result_from_outcome(
    outcome: ToolCallOutcome,
) -> ExecutionResult | None:
    if not isinstance(outcome, ToolCallResult):
        return None
    result = outcome.result
    if isinstance(result, ShellFormattedResult):
        return result.execution_result
    execution_result = getattr(result, "execution_result", None)
    if isinstance(execution_result, ExecutionResult):
        return execution_result
    return None


def _composition_result_from_outcome(
    outcome: ToolCallOutcome,
) -> ShellCompositionResult | None:
    if not isinstance(outcome, ToolCallResult):
        return None
    result = outcome.result
    if isinstance(result, ShellFormattedCompositionResult):
        return result.composition_result
    composition_result = getattr(result, "composition_result", None)
    if isinstance(composition_result, ShellCompositionResult):
        return composition_result
    return None


def _composition_request_stage_chain(
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[str, bool]:
    redacted = False
    commands: list[str] = []
    for step in steps:
        if step.command in SHELL_COMMAND_DEFINITIONS:
            commands.append(step.command)
            continue
        commands.append(REDACTED_DISPLAY_VALUE)
        redacted = True
    return " | ".join(commands), redacted


def _composition_result_stage_chain(
    steps: tuple[ShellExecutionStepResult, ...],
) -> str:
    return " | ".join(step.command for step in steps)


def _composition_request_scope(
    steps: tuple[ShellCommandStepRequest, ...],
) -> tuple[str, bool]:
    cwd_values = tuple(step.cwd or "." for step in steps)
    return _composition_common_scope(cwd_values)


def _composition_result_scope(
    steps: tuple[ShellExecutionStepResult, ...],
) -> tuple[str, bool]:
    cwd_values: list[str] = []
    for step in steps:
        cwd = step.metadata.get("display_cwd")
        if isinstance(cwd, str) and cwd:
            cwd_values.append(cwd)
    return _composition_common_scope(tuple(cwd_values) or (".",))


def _composition_common_scope(values: tuple[str, ...]) -> tuple[str, bool]:
    redacted = False
    display_values: list[str] = []
    for value in values:
        display_value, value_redacted = _safe_path(value)
        display_values.append(display_value or REDACTED_DISPLAY_VALUE)
        redacted = redacted or value_redacted
    unique_values = tuple(dict.fromkeys(display_values))
    if len(unique_values) == 1:
        return unique_values[0], redacted
    return "mixed cwd", redacted


def _composition_scope_details(
    scope: str,
    scope_redacted: bool,
) -> tuple[ToolDisplayDetail, ...]:
    return (_detail("cwd", scope, redacted=scope_redacted),)


def _composition_request_limit_details(
    request: ShellCompositionRequest,
) -> tuple[ToolDisplayDetail, ...]:
    caps = _composition_caps(
        timeout_seconds=request.timeout_seconds,
        max_stdout_bytes=request.max_stdout_bytes,
        max_stderr_bytes=request.max_stderr_bytes,
        max_intermediate_bytes=request.max_intermediate_bytes,
    )
    return (_detail("caps", caps),) if caps is not None else ()


def _composition_caps(
    *,
    timeout_seconds: float | None,
    max_stdout_bytes: int | None,
    max_stderr_bytes: int | None,
    max_intermediate_bytes: int | None,
) -> str | None:
    parts = [
        _composition_cap("timeout", timeout_seconds),
        _composition_cap("stdout", max_stdout_bytes),
        _composition_cap("stderr", max_stderr_bytes),
        _composition_cap("intermediate", max_intermediate_bytes),
    ]
    caps = ", ".join(part for part in parts if part is not None)
    return caps or None


def _composition_cap(name: str, value: int | float | None) -> str | None:
    if value is None:
        return None
    return f"{name}={value}"


def _composition_request_metrics(
    request: ShellCompositionRequest,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {"stage_count": len(request.steps)}
    if request.timeout_seconds is not None:
        metrics["timeout_seconds"] = request.timeout_seconds
    if request.max_stdout_bytes is not None:
        metrics["max_stdout_bytes"] = request.max_stdout_bytes
    if request.max_stderr_bytes is not None:
        metrics["max_stderr_bytes"] = request.max_stderr_bytes
    if request.max_intermediate_bytes is not None:
        metrics["max_intermediate_bytes"] = request.max_intermediate_bytes
    return metrics


def _composition_result_metrics(
    result: ShellCompositionResult,
) -> dict[str, int]:
    return {
        "stage_count": len(result.steps),
        "duration_ms": result.duration_ms,
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "failed_stage_count": sum(
            1
            for step in result.steps
            if step.status
            not in {
                ShellExecutionStatus.COMPLETED,
                ShellExecutionStatus.NO_MATCHES,
            }
        ),
    }


def _composition_stage_statuses(
    steps: tuple[ShellExecutionStepResult, ...],
) -> str:
    return ", ".join(f"{step.id}:{step.status.value}" for step in steps)


def _composition_result_summary(result: ShellCompositionResult) -> str:
    if result.status is ShellExecutionStatus.COMPLETED:
        return (
            f"Pipeline completed: {_composition_stage_statuses(result.steps)}."
        )
    failed_step = _first_non_success_stage(result.steps)
    if failed_step is not None:
        message = failed_step.error_message or result.error_message
        statuses = _composition_stage_statuses(result.steps)
        if message:
            return (
                "Pipeline failed at "
                f"{failed_step.id}: {failed_step.status.value} ({message}). "
                f"Stages: {statuses}."
            )
        return (
            "Pipeline failed at "
            f"{failed_step.id}: {failed_step.status.value}. "
            f"Stages: {statuses}."
        )
    if result.error_message is not None:
        return (
            f"Pipeline ended with {result.status.value}: "
            f"{result.error_message}."
        )
    return f"Pipeline ended with {result.status.value}."


def _first_non_success_stage(
    steps: tuple[ShellExecutionStepResult, ...],
) -> ShellExecutionStepResult | None:
    for step in steps:
        if step.status not in {
            ShellExecutionStatus.COMPLETED,
            ShellExecutionStatus.NO_MATCHES,
        }:
            return step
    return None


def _request_target(
    request: ShellCommandRequest,
) -> tuple[str | None, bool]:
    match request.command:
        case "rg":
            return _string_option(request.options, "pattern"), False
        case "jq":
            return _string_option(request.options, "filter"), False
        case "find":
            name = _string_option(request.options, "name")
            if name is not None:
                return _safe_path(name)
            return _paths_value(request.paths, default="workspace")
        case "sed":
            patterns = _sequence_option(request.options, "patterns")
            if patterns is not None:
                return patterns, False
            line_ranges = _sequence_option(request.options, "line_ranges")
            if line_ranges is not None:
                return line_ranges, False
            return _paths_value(request.paths, default="files")
        case "awk":
            pattern = _string_option(request.options, "pattern")
            if pattern is not None:
                return pattern, False
            fields = _sequence_option(request.options, "fields")
            if fields is not None:
                return fields, False
            return _paths_value(request.paths, default="files")
        case command if command in _PATH_TARGET_COMMANDS:
            return _paths_value(request.paths, default=".")
        case _:
            return request.command, False


def _request_scope(
    request: ShellCommandRequest,
) -> tuple[str | None, bool]:
    if request.command in {"rg", "find"} and not request.paths:
        return "workspace", False
    if request.command == "ls" and not request.paths:
        return "current directory", False
    return _paths_value(request.paths, default=None)


def _request_details(
    request: ShellCommandRequest,
) -> tuple[ToolDisplayDetail, ...]:
    details = [_detail("command", request.command)]
    paths, paths_redacted = _paths_value(request.paths, default=None)
    if paths is not None:
        details.append(_detail("paths", paths, redacted=paths_redacted))
    match request.command:
        case "rg":
            _append_option(details, request.options, "pattern")
            _append_option(details, request.options, "case")
            _append_option(details, request.options, "fixed_strings")
            _append_option(details, request.options, "context_lines")
            _append_option(details, request.options, "max_depth")
            _append_globs(details, request.options)
        case "head":
            _append_option(details, request.options, "lines")
            _append_option(details, request.options, "byte_count")
        case "tail":
            _append_option(details, request.options, "lines")
            _append_option(details, request.options, "start_line")
            _append_option(details, request.options, "byte_count")
            _append_option(details, request.options, "start_byte")
        case "file":
            _append_option(details, request.options, "brief")
            _append_option(details, request.options, "mime_type")
        case "nl":
            _append_option(details, request.options, "body_numbering")
            _append_option(details, request.options, "number_format")
            _append_option(details, request.options, "number_separator")
            _append_option(details, request.options, "starting_line_number")
            _append_option(details, request.options, "line_increment")
            _append_option(details, request.options, "number_width")
        case "find":
            _append_option(details, request.options, "entry_type")
            _append_safe_path_option(details, request.options, "name")
            _append_option(details, request.options, "min_depth")
            _append_option(details, request.options, "max_depth")
        case "wc":
            counts = _enabled_names(
                request.options,
                ("lines", "words", "count_bytes"),
            )
            if counts is not None:
                details.append(_detail("counts", counts))
        case "awk":
            _append_option(details, request.options, "fields")
            _append_option(details, request.options, "field_separator")
            _append_option(details, request.options, "pattern")
            _append_line_range(details, request.options)
        case "sed":
            _append_option(details, request.options, "line_ranges")
            _append_option(details, request.options, "patterns")
            _append_line_range(details, request.options)
        case "jq":
            _append_option(details, request.options, "filter")
            _append_enabled_options(
                details,
                request.options,
                ("raw_output", "compact", "slurp", "sort_keys"),
            )
        case "pdfinfo":
            _append_page_range(details, request.options)
            _append_option(details, request.options, "boxes")
            _append_option(details, request.options, "iso_dates")
        case "pdftotext":
            _append_page_range(details, request.options)
            _append_option(details, request.options, "layout")
            _append_option(details, request.options, "no_page_breaks")
        case "pdftoppm":
            _append_page_range(details, request.options)
            _append_option(details, request.options, "dpi")
            _append_option(details, request.options, "grayscale")
            _append_option(details, request.options, "format")
        case "tesseract":
            _append_option(details, request.options, "languages")
            _append_option(details, request.options, "psm")
            _append_option(details, request.options, "oem")
            _append_option(details, request.options, "dpi")
            _append_option(details, request.options, "output_format")
    cwd, cwd_redacted = _safe_path(request.cwd)
    if cwd is not None:
        details.append(_detail("cwd", cwd, redacted=cwd_redacted))
    return tuple(details)


def _request_limit_details(
    request: ShellCommandRequest,
) -> tuple[ToolDisplayDetail, ...]:
    details: list[ToolDisplayDetail] = []
    for label, value in (
        ("timeout seconds", request.timeout_seconds),
        ("max stdout bytes", request.max_stdout_bytes),
        ("max stderr bytes", request.max_stderr_bytes),
    ):
        if value is not None:
            details.append(_detail(label, value))
    return tuple(details)


def _request_output_details(
    request: ShellCommandRequest,
) -> tuple[ToolDisplayDetail, ...]:
    command_definition = SHELL_COMMAND_DEFINITIONS.get(request.command)
    if command_definition is None:
        return ()
    try:
        media_type, output_kind = command_definition.output_contract(request)
    except Exception:
        return ()
    return (
        _detail("output kind", output_kind.value),
        _detail("media type", media_type),
    )


def _request_metrics(
    request: ShellCommandRequest,
) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {
        "path_count": len(request.paths),
    }
    if request.timeout_seconds is not None:
        metrics["timeout_seconds"] = request.timeout_seconds
    if request.max_stdout_bytes is not None:
        metrics["max_stdout_bytes"] = request.max_stdout_bytes
    if request.max_stderr_bytes is not None:
        metrics["max_stderr_bytes"] = request.max_stderr_bytes
    for name in (
        "lines",
        "start_line",
        "end_line",
        "first_page",
        "last_page",
        "dpi",
        "psm",
        "oem",
        "max_depth",
        "starting_line_number",
        "line_increment",
        "number_width",
    ):
        value = request.options.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            metrics[name] = value
    return metrics


def _result_fact_details(
    result: ExecutionResult,
) -> tuple[ToolDisplayDetail, ...]:
    details = [
        _detail("status", result.status.value),
        _detail("exit code", result.exit_code),
        _detail("duration ms", result.duration_ms),
        _detail("output kind", result.output_kind.value),
        _detail("media type", result.stdout_media_type),
        _detail("stdout bytes", result.stdout_bytes),
        _detail("stdout truncated", result.stdout_truncated),
        _detail("stderr bytes", result.stderr_bytes),
        _detail("stderr truncated", result.stderr_truncated),
    ]
    if result.error_code is not None:
        details.append(_detail("error code", result.error_code.value))
    if result.error_message is not None:
        details.append(_detail("error message", result.error_message))
    return tuple(details)


def _result_metrics(
    result: ExecutionResult,
) -> dict[str, int]:
    metrics = {
        "duration_ms": result.duration_ms,
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "generated_file_count": len(result.generated_files),
    }
    if result.exit_code is not None:
        metrics["exit_code"] = result.exit_code
    if result.generated_files:
        metrics["generated_total_bytes"] = sum(
            generated_file.bytes for generated_file in result.generated_files
        )
    return metrics


def _generated_file_details(
    generated_file: GeneratedFile | None,
) -> tuple[ToolDisplayDetail, ...]:
    if generated_file is None:
        return ()
    display_path, redacted = _safe_generated_path(
        generated_file.display_path,
    )
    return (
        _detail("generated output", display_path, redacted=redacted),
        _detail("generated media type", generated_file.media_type),
    )


def _generated_file_preview(
    generated_file: GeneratedFile | None,
) -> ToolDisplayPreview | None:
    if generated_file is None:
        return None
    display_path, redacted = _safe_generated_path(
        generated_file.display_path,
    )
    return ToolDisplayPreview(
        content=display_path,
        label="generated output",
        media_type=generated_file.media_type,
        redacted=redacted,
        truncated=generated_file.truncated,
    )


def _result_target(result: ExecutionResult) -> tuple[str, bool]:
    if not result.display_argv:
        return result.command, False
    arguments: list[str] = []
    redacted = False
    for argument in result.display_argv:
        display_argument, argument_redacted = _safe_command_argument(argument)
        arguments.append(display_argument)
        redacted = redacted or argument_redacted
    return shell_join(tuple(arguments)), redacted


def _result_scope(result: ExecutionResult) -> tuple[str, bool]:
    scope, redacted = _safe_path(result.display_cwd)
    return scope or ".", redacted


def _result_summary(result: ExecutionResult) -> str:
    match result.status:
        case ShellExecutionStatus.COMPLETED:
            return f"{result.command} completed."
        case ShellExecutionStatus.NO_MATCHES:
            return f"{result.command} found no matches."
        case ShellExecutionStatus.POLICY_DENIED:
            if result.error_message is not None:
                return (
                    f"{result.command} was denied by policy: "
                    f"{result.error_message}."
                )
            return f"{result.command} was denied by policy."
        case ShellExecutionStatus.NONZERO_EXIT:
            exit_code = result.exit_code
            if exit_code is not None:
                return f"{result.command} exited with status {exit_code}."
            return f"{result.command} exited with a non-zero status."
        case ShellExecutionStatus.TIMEOUT:
            return f"{result.command} timed out."
        case ShellExecutionStatus.COMMAND_UNAVAILABLE:
            return f"{result.command} is unavailable."
        case _:
            return f"{result.command} ended with {result.status.value}."


def _result_severity(status: ShellExecutionStatus) -> str:
    if status in {
        ShellExecutionStatus.COMPLETED,
        ShellExecutionStatus.NO_MATCHES,
    }:
        return "info"
    if status is ShellExecutionStatus.POLICY_DENIED:
        return "warning"
    return "error"


def _paths_value(
    paths: Sequence[PathOperand],
    *,
    default: str | None,
) -> tuple[str | None, bool]:
    if not paths:
        return default, False
    values: list[str] = []
    redacted = False
    for path in paths:
        value, value_redacted = _safe_path(path.path)
        values.append(value or REDACTED_DISPLAY_VALUE)
        redacted = redacted or value_redacted
    if len(values) <= 3:
        return ", ".join(values), redacted
    return f"{', '.join(values[:3])}, ...", redacted


def _safe_path(path: str | None) -> tuple[str | None, bool]:
    if path is None:
        return None, False
    assert isinstance(path, str)
    if _is_unsafe_display_path(path) or is_denied_display_path(path):
        return REDACTED_DISPLAY_VALUE, True
    return path, False


def _is_unsafe_display_path(path: str) -> bool:
    assert isinstance(path, str)
    if path.startswith(("/", "~", "$", "%")):
        return True
    if len(path) > 1 and path[1] == ":":
        return True
    if path == ".." or path.startswith("../") or path.endswith("/.."):
        return True
    if "/../" in path or "\\" in path or "$" in path:
        return True
    return path.endswith("%")


def _safe_generated_path(path: str) -> tuple[str, bool]:
    assert isinstance(path, str)
    if (
        _is_unsafe_display_path(path)
        or is_denied_display_path(path)
        or any(marker in path for marker in _GENERATED_OUTPUT_PATH_MARKERS)
    ):
        return REDACTED_DISPLAY_VALUE, True
    return path, False


def _safe_command_argument(argument: str) -> tuple[str, bool]:
    assert isinstance(argument, str)
    if _is_unsafe_command_argument(argument) or any(
        marker in argument for marker in _GENERATED_OUTPUT_PATH_MARKERS
    ):
        return REDACTED_DISPLAY_VALUE, True
    return argument, False


def _is_unsafe_command_argument(argument: str) -> bool:
    assert isinstance(argument, str)
    if _has_unsafe_control_character(argument):
        return True
    return any(
        _is_unsafe_command_path_candidate(candidate)
        or is_denied_display_path(candidate)
        for candidate in _command_path_candidates(argument)
    )


def _has_unsafe_control_character(value: str) -> bool:
    assert isinstance(value, str)
    return any(
        (ord(character) < 32 and character != "\t")
        or 127 <= ord(character) < 160
        for character in value
    )


def _command_path_candidates(argument: str) -> tuple[str, ...]:
    candidates = [argument]
    if argument.startswith("!"):
        candidates.append(argument[1:])
    for separator in ("=", ","):
        if separator in argument:
            candidates.extend(
                candidate
                for candidate in argument.split(separator)
                if candidate
            )
    return tuple(
        _strip_display_path_quotes(candidate) for candidate in candidates
    )


def _strip_display_path_quotes(value: str) -> str:
    return value.strip("'\"")


def _is_unsafe_command_path_candidate(value: str) -> bool:
    assert isinstance(value, str)
    if not value:
        return False
    if value.startswith(("/", "~")):
        return True
    if len(value) > 1 and value[1] == ":":
        return True
    if value == ".." or value.startswith("../") or value.endswith("/.."):
        return True
    if "/../" in value:
        return True
    if value.startswith("$") and ("/" in value or "\\" in value):
        return True
    if value.startswith("%") and ("/" in value or "\\" in value):
        return True
    return value.startswith("\\\\")


def _append_option(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
    name: str,
) -> None:
    if name not in options:
        return
    value = _display_option_value(options[name])
    if value is None:
        return
    details.append(_detail(_label(name), value))


def _append_safe_path_option(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
    name: str,
) -> None:
    value = _string_option(options, name)
    if value is None:
        return
    display_value, redacted = _safe_path(value)
    details.append(_detail(_label(name), display_value, redacted=redacted))


def _append_enabled_options(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
    names: tuple[str, ...],
) -> None:
    enabled = _enabled_names(options, names)
    if enabled is not None:
        details.append(_detail("enabled options", enabled))


def _append_line_range(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
) -> None:
    start_line = options.get("start_line")
    end_line = options.get("end_line")
    if start_line is None and end_line is None:
        return
    if end_line is None:
        details.append(_detail("start line", start_line))
        return
    details.append(_detail("line range", f"{start_line or 1}-{end_line}"))


def _append_page_range(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
) -> None:
    first_page = options.get("first_page")
    last_page = options.get("last_page")
    if first_page is None and last_page is None:
        return
    if last_page is None:
        details.append(_detail("first page", first_page))
        return
    details.append(_detail("page range", f"{first_page or 1}-{last_page}"))


def _append_globs(
    details: list[ToolDisplayDetail],
    options: Mapping[str, object],
) -> None:
    value = options.get("globs")
    if value is None or value == ():
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        details.append(_detail("globs", str(value)))
        return
    globs: list[str] = []
    redacted = False
    for item in value:
        glob = str(item)
        display_path = glob[1:] if glob.startswith("!") else glob
        if _is_unsafe_display_path(display_path) or is_denied_display_path(
            display_path
        ):
            globs.append(REDACTED_DISPLAY_VALUE)
            redacted = True
        else:
            globs.append(glob)
    details.append(_detail("globs", ", ".join(globs), redacted=redacted))


def _enabled_names(
    options: Mapping[str, object],
    names: tuple[str, ...],
) -> str | None:
    enabled = tuple(_label(name) for name in names if options.get(name))
    if not enabled:
        return None
    return ", ".join(enabled)


def _string_option(
    options: Mapping[str, object],
    name: str,
) -> str | None:
    value = options.get(name)
    if isinstance(value, str) and value:
        return value
    return None


def _sequence_option(
    options: Mapping[str, object],
    name: str,
) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    display_value = _display_option_value(value)
    return display_value if isinstance(display_value, str) else None


def _display_option_value(value: object) -> None | bool | int | float | str:
    if value is None or value == () or value == []:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    ):
        return ", ".join(str(item) for item in value)
    return str(value)


def _label(name: str) -> str:
    assert isinstance(name, str)
    return name.replace("_", " ")


def _detail(
    label: str,
    value: object,
    *,
    redacted: bool = False,
) -> ToolDisplayDetail:
    return ToolDisplayDetail(
        label=label,
        value=_display_option_value(value),
        redacted=redacted,
    )
