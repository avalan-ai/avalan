from ...cli.theme import Theme
from ...filesystem import DEFAULT_TEXT_ENCODING, read_text, write_text
from ...flow import (
    FlowDefinition,
    FlowDefinitionCompileResult,
    FlowDefinitionLoader,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowExecutionUpdate,
    FlowExecutor,
    FlowGraphInspection,
    FlowGraphInspectionResult,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowLoadResult,
    FlowLoopPolicy,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryPolicy,
    FlowSourceSpan,
    FlowStateStore,
    FlowTaskExecutor,
    FlowTimeoutPolicy,
    FlowToolResolver,
    FlowView,
    FlowViewClassDefinition,
    FlowViewComment,
    FlowViewDirection,
    FlowViewEdge,
    FlowViewGroup,
    FlowViewImportMode,
    FlowViewLinkStyle,
    FlowViewNode,
    FlowViewStyle,
    Node,
    PgsqlFlowStateStore,
    compare_flow_topology,
    compile_flow_file,
    default_flow_node_registry,
    flow_input_binding,
    inspect_flow_graph_file,
    parse_mermaid_view,
    render_flow_view,
    skeleton_from_mermaid_view,
)
from ...task import (
    TaskClientUnsupportedOperationError,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskObservedEvent,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRunState,
    TaskSanitizedEventObserver,
    TaskSchemaResolutionError,
    TaskTargetType,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    privacy_policy_with_defaults,
    resolve_schema_ref,
    validate_task_input,
    validate_task_output,
)
from ...task.store import TaskStoreConflictError, TaskStoreNotFoundError
from ...tool import ToolSet
from ...tool.browser import (
    HAS_BROWSER_DEPENDENCIES,
    BrowserToolSet,
    BrowserToolSettings,
)
from ...tool.code import HAS_CODE_DEPENDENCIES, CodeToolSet
from ...tool.database.settings import DatabaseToolSettings
from ...tool.database.toolset import DatabaseToolSet
from ...tool.graph import HAS_GRAPH_DEPENDENCIES, GraphToolSet
from ...tool.graph_settings import GraphToolSettings
from ...tool.manager import ToolManager
from ...tool.math import MathToolSet
from .agent import get_tool_settings
from .task import (
    TaskCliInputError,
    _format_task_cli_value,
    _print_issues,
    _print_missing_inspection_store,
    _print_task_cli_input_error,
    _print_task_command_error,
    _print_task_execution_error,
    _print_task_run_failure,
    _run_awaitable,
    _task_cli_after_sequence,
    _task_cli_client_context,
    _task_cli_inspection_client_context,
    _task_command_metadata,
    _task_diagnostic_console,
    _task_output_is_structured,
    _task_pgsql_database,
    _task_run_json_output,
    _task_run_quiet,
    _task_run_structured_output_requested,
    _task_store_dsn,
    _task_store_schema,
    _validate_task_run_output_path,
    _write_task_run_structured_output,
    task_cli_input,
)

from argparse import Namespace
from asyncio import to_thread
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from json import JSONDecodeError, dumps, loads
from logging import Logger
from os import strerror
from pathlib import Path
from time import monotonic
from uuid import uuid4

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live

_AVALAN_INPUT_TYPE_SCHEMA_KEY = "x-avalan-input-type"
_AVALAN_MIME_TYPES_SCHEMA_KEY = "x-avalan-mime-types"


def flow_validate(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Validate a flow definition without executing it."""
    _ = theme
    return _run_awaitable(_flow_validate(args, console))


async def _flow_validate(args: Namespace, console: Console) -> bool:
    load_result = await _flow_load_validation_result(args.flow, console, args)
    if load_result is None:
        return False
    ok = load_result.ok
    diagnostics = load_result.diagnostics
    if _flow_json_output(args):
        _print_flow_json_result(console, ok=ok, diagnostics=diagnostics)
    elif ok:
        assert load_result.definition is not None
        console.print(
            "Flow definition is valid: "
            f"{load_result.definition.name} "
            f"{_flow_definition_identity(load_result.definition)}",
            markup=False,
        )
    else:
        _print_flow_diagnostics(
            console,
            "Flow definition is invalid.",
            diagnostics,
        )
    return ok


def flow_compile(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Compile a flow definition to strict canonical TOML."""
    _ = theme
    return _run_awaitable(_flow_compile(args, console))


async def _flow_compile(args: Namespace, console: Console) -> bool:
    result = await compile_flow_file(
        args.flow,
        encoding=_flow_text_encoding(args),
    )
    if not result.ok:
        if _flow_json_output(args):
            _print_flow_compile_json_result(console, result)
        else:
            title = (
                "Flow definition could not be read."
                if _flow_result_has_diagnostic(result, "file.read")
                else "Flow definition could not be compiled."
            )
            _print_flow_diagnostics(
                console,
                title,
                result.all_diagnostics,
            )
        return False
    assert result.definition is not None
    assert result.canonical_source is not None
    if args.check:
        if _flow_json_output(args):
            _print_flow_compile_json_result(console, result)
        else:
            console.print(
                "Flow definition compiles: "
                f"{result.definition.name} "
                f"{_flow_definition_identity(result.definition)}",
                markup=False,
            )
        return True
    if args.output is not None:
        try:
            await _flow_write_text_atomic(
                Path(args.output),
                result.canonical_source,
                encoding=_flow_text_encoding(args),
            )
        except (OSError, UnicodeEncodeError) as exc:
            diagnostic = _flow_file_write_diagnostic(exc)
            if _flow_json_output(args):
                _print_flow_json_result(
                    console,
                    ok=False,
                    diagnostics=(diagnostic,),
                )
            else:
                _print_flow_diagnostics(
                    console,
                    "Compiled flow could not be written.",
                    (diagnostic,),
                )
            return False
        if _flow_json_output(args):
            _print_flow_compile_json_result(console, result)
        else:
            console.print("Compiled flow written.", markup=False)
        return True
    if _flow_json_output(args):
        _print_flow_compile_json_result(console, result)
    else:
        console.print(result.canonical_source, markup=False, end="")
    return True


def flow_mermaid(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Run non-executing Mermaid flow authoring commands."""
    _ = theme
    return _run_awaitable(_flow_mermaid(args, console))


async def _flow_mermaid(args: Namespace, console: Console) -> bool:
    command = args.flow_mermaid_command
    match command:
        case "compare":
            return await _flow_mermaid_compare(args, console)
        case "parse":
            return await _flow_mermaid_parse(args, console)
        case "render":
            return await _flow_mermaid_render(args, console)
        case "skeleton":
            return await _flow_mermaid_skeleton(args, console)
    raise AssertionError("unsupported Mermaid flow command")


def flow_graph(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Run static graph authoring commands."""
    _ = theme
    return _run_awaitable(_flow_graph(args, console))


async def _flow_graph(args: Namespace, console: Console) -> bool:
    command = args.flow_graph_command
    match command:
        case "inspect":
            return await _flow_graph_inspect(args, console)
    raise AssertionError("unsupported graph flow command")


def flow_run(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: object | None = None,
    logger: Logger | None = None,
) -> bool:
    """Run a native flow definition."""
    return _run_awaitable(
        _flow_run(args, console, theme=theme, hub=hub, logger=logger)
    )


def flow_inspect(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Inspect a durable flow run."""
    _ = theme
    return _run_awaitable(_flow_inspect(args, console))


def flow_trace(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Export a sanitized flow trace."""
    _ = theme
    return _run_awaitable(_flow_trace(args, console))


def flow_cancel(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Request cancellation for a durable flow run."""
    _ = theme
    return _run_awaitable(_flow_cancel(args, console))


def flow_resume(
    args: Namespace,
    console: Console,
    theme: Theme,
) -> bool:
    """Resume a paused strict flow from durable state."""
    _ = theme
    return _run_awaitable(_flow_resume(args, console))


async def _flow_mermaid_parse(args: Namespace, console: Console) -> bool:
    source = await _flow_read_text(
        args.diagram,
        console,
        args,
        "Mermaid diagram",
    )
    if source is None:
        return False
    result = parse_mermaid_view(
        source,
        import_mode=_flow_import_mode(args),
    )
    ok = result.ok
    if _flow_json_output(args):
        values = {"view": _flow_view_public_dict(result.view)} if ok else {}
        _print_flow_json_result(
            console,
            ok=ok,
            diagnostics=result.diagnostics,
            **values,
        )
    elif ok:
        console.print(
            "Mermaid diagram parsed: "
            f"{len(result.view.nodes)} nodes, {len(result.view.edges)} edges.",
            markup=False,
        )
        if result.diagnostics:
            _print_flow_diagnostics(
                console,
                "Mermaid diagnostics.",
                result.diagnostics,
            )
    else:
        _print_flow_diagnostics(
            console,
            "Mermaid diagram is invalid.",
            result.diagnostics,
        )
    return ok


async def _flow_mermaid_render(args: Namespace, console: Console) -> bool:
    source = await _flow_read_text(
        args.diagram,
        console,
        args,
        "Mermaid diagram",
    )
    if source is None:
        return False
    parsed = parse_mermaid_view(
        source,
        import_mode=_flow_import_mode(args),
    )
    if not parsed.ok:
        if _flow_json_output(args):
            _print_flow_json_result(
                console,
                ok=False,
                diagnostics=parsed.diagnostics,
            )
        else:
            _print_flow_diagnostics(
                console,
                "Mermaid diagram is invalid.",
                parsed.diagnostics,
            )
        return False
    rendered = render_flow_view(parsed.view)
    diagnostics = parsed.diagnostics + rendered.diagnostics
    ok = rendered.ok
    if _flow_json_output(args):
        _print_flow_json_result(
            console,
            ok=ok,
            diagnostics=diagnostics,
            source=rendered.source,
        )
    elif ok:
        console.print(rendered.source, markup=False)
    else:
        _print_flow_diagnostics(
            console,
            "Mermaid diagram could not be rendered.",
            diagnostics,
        )
    return ok


async def _flow_mermaid_compare(args: Namespace, console: Console) -> bool:
    source = await _flow_read_text(
        args.diagram,
        console,
        args,
        "Mermaid diagram",
    )
    if source is None:
        return False
    parsed = parse_mermaid_view(
        source,
        import_mode=_flow_import_mode(args),
    )
    load_result = await _flow_load_validation_result(args.flow, console, args)
    if load_result is None:
        return False
    diagnostics: tuple[FlowDiagnostic, ...]
    definition = load_result.definition
    if parsed.ok and load_result.ok and definition is not None:
        comparison = compare_flow_topology(parsed.view, definition)
        diagnostics = comparison.diagnostics
        ok = comparison.ok
    else:
        diagnostics = parsed.diagnostics + load_result.diagnostics
        ok = _flow_diagnostics_ok(diagnostics)
    if _flow_json_output(args):
        _print_flow_json_result(console, ok=ok, diagnostics=diagnostics)
    elif ok:
        console.print("Flow topology matches.", markup=False)
        if diagnostics:
            _print_flow_diagnostics(
                console,
                "Flow topology diagnostics.",
                diagnostics,
            )
    else:
        _print_flow_diagnostics(
            console,
            "Flow topology does not match.",
            diagnostics,
        )
    return ok


async def _flow_mermaid_skeleton(args: Namespace, console: Console) -> bool:
    source = await _flow_read_text(
        args.diagram,
        console,
        args,
        "Mermaid diagram",
    )
    if source is None:
        return False
    parsed = parse_mermaid_view(
        source,
        import_mode=_flow_import_mode(args),
    )
    if not parsed.ok:
        if _flow_json_output(args):
            _print_flow_json_result(
                console,
                ok=False,
                diagnostics=parsed.diagnostics,
            )
        else:
            _print_flow_diagnostics(
                console,
                "Mermaid diagram is invalid.",
                parsed.diagnostics,
            )
        return False
    result = skeleton_from_mermaid_view(
        parsed.view,
        name=args.name,
        version=args.version,
        revision=args.revision,
    )
    if _flow_json_output(args):
        _print_flow_json_result(
            console,
            ok=result.ok,
            diagnostics=result.diagnostics,
            definition=_flow_definition_public_dict(result.definition),
        )
    elif result.ok:
        console.print(_flow_definition_toml(result.definition), markup=False)
    else:
        _print_flow_diagnostics(
            console,
            "Flow skeleton could not be created.",
            result.diagnostics,
        )
    return result.ok


async def _flow_graph_inspect(args: Namespace, console: Console) -> bool:
    result = await inspect_flow_graph_file(
        args.flow,
        encoding=_flow_text_encoding(args),
    )
    if _flow_json_output(args):
        _print_flow_graph_inspect_json_result(console, result)
    elif result.ok:
        assert result.inspection is not None
        _print_flow_graph_inspection(console, result.inspection)
    else:
        title = (
            "Flow graph could not be read."
            if _flow_result_has_diagnostic(result, "file.read")
            else "Flow graph could not be inspected."
        )
        _print_flow_diagnostics(
            console,
            title,
            _flow_graph_inspect_diagnostics(result),
        )
    return result.ok


async def _flow_load_validation_result(
    path: str | Path,
    console: Console,
    args: Namespace,
) -> FlowLoadResult | None:
    try:
        return await FlowDefinitionLoader(
            encoding=_flow_text_encoding(args)
        ).load_validation_result(Path(path))
    except (OSError, UnicodeDecodeError) as exc:
        diagnostic = _flow_file_read_diagnostic(exc)
        if _flow_json_output(args):
            _print_flow_json_result(
                console,
                ok=False,
                diagnostics=(diagnostic,),
            )
        else:
            _print_flow_diagnostics(
                console,
                "Flow definition could not be read.",
                (diagnostic,),
            )
        return None


def _flow_text_encoding(args: Namespace) -> str:
    encoding = getattr(args, "encoding", DEFAULT_TEXT_ENCODING)
    if encoding is None:
        return DEFAULT_TEXT_ENCODING
    assert isinstance(encoding, str), "encoding must be a string"
    return encoding


async def _flow_read_text(
    path: str | Path,
    console: Console,
    args: Namespace,
    noun: str,
) -> str | None:
    try:
        return await read_text(path, encoding=_flow_text_encoding(args))
    except (OSError, UnicodeDecodeError) as exc:
        diagnostic = _flow_file_read_diagnostic(exc)
        if _flow_json_output(args):
            _print_flow_json_result(
                console,
                ok=False,
                diagnostics=(diagnostic,),
            )
        else:
            _print_flow_diagnostics(
                console,
                f"{noun} could not be read.",
                (diagnostic,),
            )
        return None


def _flow_file_read_diagnostic(
    exc: OSError | UnicodeDecodeError,
) -> FlowDiagnostic:
    message = _flow_file_error_message(exc, "Unable to read file.")
    return FlowDiagnostic(
        code="file.read",
        path="file",
        category=FlowDiagnosticCategory.PRIVACY,
        severity=FlowDiagnosticSeverity.ERROR,
        message=message,
        hint="Use a readable local file.",
    )


def _flow_file_write_diagnostic(
    exc: OSError | UnicodeEncodeError,
) -> FlowDiagnostic:
    message = _flow_file_error_message(exc, "Unable to write file.")
    return FlowDiagnostic(
        code="file.write",
        path="file",
        category=FlowDiagnosticCategory.PRIVACY,
        severity=FlowDiagnosticSeverity.ERROR,
        message=message,
        hint="Use a writable local output path.",
    )


def _flow_file_error_message(
    exc: OSError | UnicodeDecodeError | UnicodeEncodeError,
    fallback: str,
) -> str:
    if isinstance(exc, OSError):
        return strerror(exc.errno) if exc.errno else fallback
    return str(exc) or fallback


def _flow_import_mode(args: Namespace) -> FlowViewImportMode:
    return FlowViewImportMode(args.mode)


def _flow_json_output(args: Namespace) -> bool:
    return bool(getattr(args, "flow_json", False))


def _flow_diagnostics_ok(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> bool:
    return not any(
        diagnostic.severity == FlowDiagnosticSeverity.ERROR
        for diagnostic in diagnostics
    )


def _flow_result_has_diagnostic(
    result: FlowDefinitionCompileResult | FlowGraphInspectionResult,
    code: str,
) -> bool:
    assert isinstance(
        result,
        FlowDefinitionCompileResult | FlowGraphInspectionResult,
    )
    assert isinstance(code, str) and code.strip()
    return any(
        diagnostic["code"] == code for diagnostic in result.public_diagnostics
    )


def _print_flow_json_result(
    console: Console,
    *,
    ok: bool,
    diagnostics: tuple[FlowDiagnostic, ...],
    **values: object,
) -> None:
    payload: dict[str, object] = {
        "ok": ok,
        "diagnostics": tuple(
            diagnostic.as_public_dict() for diagnostic in diagnostics
        ),
    }
    payload.update(values)
    console.print(
        dumps(
            _flow_public_value(payload),
            sort_keys=True,
            separators=(",", ":"),
        ),
        markup=False,
        soft_wrap=True,
    )


def _print_flow_compile_json_result(
    console: Console,
    result: FlowDefinitionCompileResult,
) -> None:
    console.print(
        dumps(
            _flow_public_value(result.as_public_dict()),
            sort_keys=True,
            separators=(",", ":"),
        ),
        markup=False,
        soft_wrap=True,
    )


def _print_flow_graph_inspect_json_result(
    console: Console,
    result: FlowGraphInspectionResult,
) -> None:
    console.print(
        dumps(
            _flow_public_value(result.as_public_dict()),
            sort_keys=True,
            separators=(",", ":"),
        ),
        markup=False,
        soft_wrap=True,
    )


def _print_flow_diagnostics(
    console: Console,
    title: str,
    diagnostics: tuple[FlowDiagnostic, ...],
) -> None:
    console.print(title, markup=False)
    for diagnostic in diagnostics:
        value = diagnostic.as_public_dict()
        location = _flow_diagnostic_location(value)
        console.print(
            f"{value['severity']} {value['code']}{location}",
            markup=False,
        )
        console.print(str(value["message"]), markup=False)
        hint = value.get("hint")
        if hint is not None:
            console.print(f"hint {hint}", markup=False)


def _print_flow_graph_inspection(
    console: Console,
    inspection: FlowGraphInspection,
) -> None:
    public = inspection.as_public_dict()
    nodes = public["nodes"]
    edges = public["edges"]
    bindings = public["bindings"]
    generated_edges = public["generated_edges"]
    assert isinstance(nodes, list | tuple)
    assert isinstance(edges, list | tuple)
    assert isinstance(bindings, list | tuple)
    assert isinstance(generated_edges, list | tuple)
    node_counts = _flow_count_public_values(nodes, "classification")
    edge_counts = _flow_count_public_values(edges, "classification")
    binding_counts = _flow_count_public_values(bindings, "state")
    console.print(
        f"Flow graph inspection: {public['schema_version']}.",
        markup=False,
    )
    console.print(
        "nodes "
        f"actual={node_counts.get('actual', 0)} "
        f"decorative={node_counts.get('decorative', 0)}",
        markup=False,
    )
    console.print(
        "edges "
        f"executable={edge_counts.get('executable', 0)} "
        f"decorative={edge_counts.get('decorative', 0)}",
        markup=False,
    )
    console.print(
        "bindings "
        f"bound={binding_counts.get('bound', 0)} "
        f"unbound={binding_counts.get('unbound', 0)} "
        f"missing={binding_counts.get('missing', 0)} "
        f"decorative={binding_counts.get('decorative', 0)} "
        f"rejected={binding_counts.get('rejected', 0)}",
        markup=False,
    )
    console.print(f"generated_edges {len(generated_edges)}", markup=False)
    if inspection.diagnostics:
        _print_flow_diagnostics(
            console,
            "Flow graph diagnostics.",
            inspection.diagnostics,
        )


def _flow_count_public_values(
    values: list[object] | tuple[object, ...],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        item = value.get(field)
        if not isinstance(item, str):
            continue
        counts[item] = counts.get(item, 0) + 1
    return counts


def _flow_graph_inspect_diagnostics(
    result: FlowGraphInspectionResult,
) -> tuple[FlowDiagnostic, ...]:
    return result.all_diagnostics


def _flow_diagnostic_location(value: Mapping[str, object]) -> str:
    path = value.get("path")
    if isinstance(path, str):
        return f" {path}"
    span = value.get("source_span")
    if not isinstance(span, Mapping):
        return ""
    line = span.get("start_line")
    column = span.get("start_column")
    if isinstance(line, int) and isinstance(column, int):
        return f" line {line}, column {column}"
    return ""


def _flow_definition_identity(definition: FlowDefinition) -> str:
    if definition.version is not None:
        return definition.version
    if definition.revision is not None:
        return definition.revision
    return "unversioned"


def _flow_view_public_dict(view: FlowView) -> dict[str, object]:
    return {
        "import_mode": view.import_mode.value,
        "direction": (
            view.direction.value if view.direction is not None else None
        ),
        "nodes": tuple(
            _flow_view_node_public_dict(node) for node in view.nodes
        ),
        "edges": tuple(
            _flow_view_edge_public_dict(edge) for edge in view.edges
        ),
        "groups": tuple(
            _flow_view_group_public_dict(group) for group in view.groups
        ),
        "class_definitions": tuple(
            _flow_view_class_public_dict(class_definition)
            for class_definition in view.class_definitions
        ),
        "styles": tuple(
            _flow_view_style_public_dict(style) for style in view.styles
        ),
        "link_styles": tuple(
            _flow_view_link_style_public_dict(link_style)
            for link_style in view.link_styles
        ),
        "comments": tuple(
            _flow_view_comment_public_dict(comment)
            for comment in view.comments
        ),
        "metadata": view.metadata,
    }


def _flow_view_node_public_dict(node: FlowViewNode) -> dict[str, object]:
    return _flow_drop_none(
        {
            "id": node.id,
            "label": node.label,
            "shape": node.shape.value,
            "classes": node.classes,
            "style": node.style,
            "metadata": node.metadata,
            "source_span": _flow_source_span_public_dict(node.source_span),
            "implicit": node.implicit,
            "group": node.group,
        }
    )


def _flow_view_edge_public_dict(edge: FlowViewEdge) -> dict[str, object]:
    return _flow_drop_none(
        {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "style": edge.style.value,
            "classes": edge.classes,
            "metadata": edge.metadata,
            "source_span": _flow_source_span_public_dict(edge.source_span),
            "bidirectional": edge.bidirectional,
        }
    )


def _flow_view_group_public_dict(group: FlowViewGroup) -> dict[str, object]:
    return _flow_drop_none(
        {
            "id": group.id,
            "label": group.label,
            "parent": group.parent,
            "direction": (
                group.direction.value if group.direction is not None else None
            ),
            "nodes": group.nodes,
            "groups": group.groups,
            "classes": group.classes,
            "style": group.style,
            "metadata": group.metadata,
            "source_span": _flow_source_span_public_dict(group.source_span),
        }
    )


def _flow_view_class_public_dict(
    class_definition: FlowViewClassDefinition,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "name": class_definition.name,
            "properties": class_definition.properties,
            "source_span": _flow_source_span_public_dict(
                class_definition.source_span
            ),
        }
    )


def _flow_view_style_public_dict(style: FlowViewStyle) -> dict[str, object]:
    return _flow_drop_none(
        {
            "target": style.target,
            "properties": style.properties,
            "source_span": _flow_source_span_public_dict(style.source_span),
        }
    )


def _flow_view_link_style_public_dict(
    link_style: FlowViewLinkStyle,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "edge": link_style.edge,
            "edge_index": link_style.edge_index,
            "properties": link_style.properties,
            "source_span": _flow_source_span_public_dict(
                link_style.source_span
            ),
        }
    )


def _flow_view_comment_public_dict(
    comment: FlowViewComment,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "text": comment.text,
            "source_span": _flow_source_span_public_dict(comment.source_span),
        }
    )


def _flow_definition_public_dict(
    definition: FlowDefinition,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "name": definition.name,
            "version": definition.version,
            "revision": definition.revision,
            "description": definition.description,
            "inputs": tuple(
                _flow_input_public_dict(input_definition)
                for input_definition in definition.inputs
            ),
            "outputs": tuple(
                _flow_output_public_dict(output_definition)
                for output_definition in definition.outputs
            ),
            "entry": _flow_entry_public_dict(definition.entry_behavior),
            "output_behavior": _flow_output_behavior_public_dict(
                definition.output_behavior
            ),
            "nodes": tuple(
                _flow_node_public_dict(node) for node in definition.nodes
            ),
            "edges": tuple(
                _flow_edge_public_dict(edge) for edge in definition.edges
            ),
            "tags": definition.tags,
            "variables": definition.variables,
        }
    )


def _flow_input_public_dict(
    input_definition: FlowInputDefinition,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "name": input_definition.name,
            "type": input_definition.type.value,
            "mime_types": input_definition.mime_types,
            "schema": input_definition.schema,
            "schema_ref": input_definition.schema_ref,
        }
    )


def _flow_output_public_dict(
    output_definition: FlowOutputDefinition,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "name": output_definition.name,
            "type": output_definition.type.value,
            "schema": output_definition.schema,
            "schema_ref": output_definition.schema_ref,
        }
    )


def _flow_entry_public_dict(
    entry_behavior: FlowEntryBehavior | None,
) -> dict[str, object] | None:
    if entry_behavior is None:
        return None
    return {"type": entry_behavior.type.value, "node": entry_behavior.node}


def _flow_output_behavior_public_dict(
    output_behavior: FlowOutputBehavior | None,
) -> dict[str, object] | None:
    if output_behavior is None:
        return None
    return {
        "type": output_behavior.type.value,
        "outputs": output_behavior.outputs,
    }


def _flow_node_public_dict(node: FlowNodeDefinition) -> dict[str, object]:
    return _flow_drop_none(
        {
            "name": node.name,
            "type": node.type,
            "ref": node.ref,
            "input": node.input,
            "output": node.output,
            "join_policy": _flow_join_policy_public_dict(node.join_policy),
            "retry_policy": _flow_retry_policy_public_dict(node.retry_policy),
            "timeout_policy": _flow_timeout_policy_public_dict(
                node.timeout_policy
            ),
            "loop_policy": _flow_loop_policy_public_dict(node.loop_policy),
            "mappings": tuple(
                _flow_mapping_public_dict(mapping) for mapping in node.mappings
            ),
            "config": node.config,
        }
    )


def _flow_edge_public_dict(edge: FlowEdgeDefinition) -> dict[str, object]:
    return _flow_drop_none(
        {
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "kind": edge.kind.value,
            "priority": edge.priority,
            "default": edge.default,
            "routing_policy": edge.routing_policy.value,
        }
    )


def _flow_mapping_public_dict(
    mapping: FlowInputMapping,
) -> dict[str, object]:
    return _flow_drop_none(
        {
            "target": mapping.target,
            "type": mapping.kind.value,
            "source": mapping.source,
            "sources": mapping.sources,
            "fields": mapping.fields,
            "items": mapping.items,
        }
    )


def _flow_join_policy_public_dict(
    policy: FlowJoinPolicy | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    return _flow_drop_none(
        {
            "type": policy.type.value,
            "quorum": policy.quorum,
            "optional_inputs": policy.optional_inputs,
        }
    )


def _flow_retry_policy_public_dict(
    policy: FlowRetryPolicy | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    return _flow_drop_none(
        {
            "max_attempts": policy.max_attempts,
            "backoff": policy.backoff.value,
            "initial_delay_seconds": policy.initial_delay_seconds,
            "max_delay_seconds": policy.max_delay_seconds,
            "retryable_categories": policy.retryable_categories,
            "non_retryable_categories": policy.non_retryable_categories,
            "exhausted_route": policy.exhausted_route,
        }
    )


def _flow_timeout_policy_public_dict(
    policy: FlowTimeoutPolicy | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    return _flow_drop_none({"per_attempt_seconds": policy.per_attempt_seconds})


def _flow_loop_policy_public_dict(
    policy: FlowLoopPolicy | None,
) -> dict[str, object] | None:
    if policy is None:
        return None
    return _flow_drop_none(
        {
            "max_iterations": policy.max_iterations,
            "max_elapsed_seconds": policy.max_elapsed_seconds,
            "output_selector": policy.output_selector,
            "limit_route": policy.limit_route,
        }
    )


def _flow_source_span_public_dict(
    source_span: FlowSourceSpan | None,
) -> object:
    if source_span is None:
        return None
    return source_span.as_public_dict()


def _flow_drop_none(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if item is not None}


def _flow_public_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _flow_public_value(item) for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_flow_public_value(item) for item in value]
    return value


def _flow_definition_toml(definition: FlowDefinition) -> str:
    lines = ["[flow]", f"name = {_toml_value(definition.name)}"]
    if definition.version is not None:
        lines.append(f"version = {_toml_value(definition.version)}")
    if definition.revision is not None:
        lines.append(f"revision = {_toml_value(definition.revision)}")
    if definition.description is not None:
        lines.append(f"description = {_toml_value(definition.description)}")
    if definition.tags:
        lines.append(f"tags = {_toml_value(definition.tags)}")
    if definition.variables:
        lines.append("")
        lines.append("[variables]")
        lines.extend(_toml_mapping_lines(definition.variables))
    for node in definition.nodes:
        lines.append("")
        lines.append(f"[nodes.{_toml_key(node.name)}]")
        lines.append(f"type = {_toml_value(node.type)}")
        if node.config:
            lines.append("")
            lines.append(f"[nodes.{_toml_key(node.name)}.config]")
            lines.extend(_toml_mapping_lines(node.config))
    for edge in definition.edges:
        lines.append("")
        lines.append("[[edges]]")
        lines.append(f"source = {_toml_value(edge.source)}")
        lines.append(f"target = {_toml_value(edge.target)}")
        if edge.label is not None:
            lines.append(f"label = {_toml_value(edge.label)}")
    return "\n".join(lines) + "\n"


def _toml_mapping_lines(value: Mapping[str, object]) -> list[str]:
    return [
        f"{_toml_key(key)} = {_toml_value(item)}"
        for key, item in sorted(value.items())
    ]


def _toml_key(value: str) -> str:
    return dumps(value)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return dumps(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Enum):
        return _toml_value(value.value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{_toml_key(key)} = {_toml_value(item)}"
            for key, item in sorted(value.items())
        )
        return "{" + items + "}"
    raise AssertionError("unsupported TOML value")


async def _flow_write_text_atomic(
    path: Path,
    content: str,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> None:
    assert isinstance(path, Path)
    assert isinstance(content, str)
    assert isinstance(encoding, str)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        await write_text(temporary_path, content, encoding=encoding)
        await to_thread(temporary_path.replace, path)
    except (OSError, UnicodeEncodeError):
        await to_thread(temporary_path.unlink, missing_ok=True)
        raise


async def _flow_inspect(args: Namespace, console: Console) -> bool:
    task_context = _task_cli_inspection_client_context(args, console)
    if task_context is None:
        return False
    assert task_context.database is not None
    try:
        async with task_context as client:
            executor = FlowTaskExecutor(
                client,
                flow_state_store=PgsqlFlowStateStore(task_context.database),
            )
            inspection = await executor.inspect(
                args.run_id,
                after_sequence=_task_cli_after_sequence(args),
            )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_flow_inspection_error(console, exc)
        return False
    _print_flow_runtime_value(
        args,
        console,
        label="inspect",
        value=inspection.as_public_dict(),
    )
    return True


async def _flow_trace(args: Namespace, console: Console) -> bool:
    task_context = _task_cli_inspection_client_context(args, console)
    if task_context is None:
        return False
    assert task_context.database is not None
    try:
        async with task_context as client:
            executor = FlowTaskExecutor(
                client,
                flow_state_store=PgsqlFlowStateStore(task_context.database),
            )
            trace = await executor.export_trace(
                args.run_id,
                after_sequence=_task_cli_after_sequence(args),
            )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_flow_inspection_error(console, exc)
        return False
    _print_flow_runtime_value(args, console, label="trace", value=trace)
    return True


async def _flow_cancel(args: Namespace, console: Console) -> bool:
    task_context = _task_cli_inspection_client_context(args, console)
    if task_context is None:
        return False
    try:
        async with task_context as client:
            run = await client.cancel(args.run_id)
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskClientUnsupportedOperationError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_flow_command_error(console, exc)
        return False
    _print_flow_runtime_value(
        args,
        console,
        label="cancel",
        value={"run_id": run.run_id, "state": run.state.value},
    )
    return True


async def _flow_resume(args: Namespace, console: Console) -> bool:
    decisions = await _flow_resume_decisions(args, console)
    if decisions is None:
        return False
    load_result = await _flow_load_validation_result(args.flow, console, args)
    if load_result is None or not load_result.ok:
        return False
    assert load_result.definition is not None
    state_context = _flow_state_store_context(args, console)
    if state_context is None:
        return False
    try:
        async with state_context as store:
            record = await store.get_flow_execution(args.run_id)
            result = await FlowExecutor().resume(
                load_result.definition,
                record,
                decisions=decisions,
            )
            if result.result is not None:
                await store.update_flow_execution(
                    args.run_id,
                    FlowExecutionUpdate(
                        trace=result.result.trace,
                        node_outputs=result.result.node_outputs,
                        selected_outputs=result.result.outputs,
                        diagnostics=result.result.diagnostics,
                        pause_tokens=result.result.pause_tokens,
                    ),
                    expected_revision=record.revision,
                )
    except (
        AssertionError,
        ImportError,
        OSError,
        TaskStoreConflictError,
        TaskStoreNotFoundError,
    ) as exc:
        _print_flow_command_error(console, exc)
        return False
    diagnostics = result.diagnostics
    if result.result is not None:
        diagnostics = diagnostics + result.result.diagnostics
    if diagnostics:
        if _flow_json_output(args):
            _print_flow_json_result(
                console,
                ok=result.ok,
                diagnostics=diagnostics,
            )
        else:
            _print_flow_diagnostics(
                console,
                "Flow resume produced diagnostics.",
                diagnostics,
            )
        return result.ok
    _print_flow_runtime_value(
        args,
        console,
        label="resume",
        value=result.outputs,
    )
    return result.ok


async def _flow_run(
    args: Namespace,
    console: Console,
    *,
    theme: Theme,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    diagnostic_console = _task_diagnostic_console(args, console)
    flow_path = Path(args.flow)
    loader = FlowDefinitionLoader(encoding=_flow_text_encoding(args))
    try:
        load_result = await loader.load_validation_result(flow_path)
    except (OSError, UnicodeDecodeError) as exc:
        _print_flow_diagnostics(
            diagnostic_console,
            "Flow definition could not be read.",
            (_flow_file_read_diagnostic(exc),),
        )
        return False
    if load_result.definition is not None and load_result.definition.is_strict:
        return await _flow_run_with_task_context(
            args,
            console,
            theme=theme,
            flow_path=flow_path,
            hub=hub,
            logger=logger,
        )
    if load_result.definition is None:
        if _flow_needs_task_context(load_result.issues):
            return await _flow_run_with_task_context(
                args,
                console,
                theme=theme,
                flow_path=flow_path,
                hub=hub,
                logger=logger,
            )
        _print_issues(
            diagnostic_console,
            "Flow definition could not be loaded.",
            _flow_load_task_issues(load_result.issues),
        )
        return False
    load_result = await loader.load_result(flow_path)
    if load_result.definition is None or load_result.flow is None:
        _print_issues(
            diagnostic_console,
            "Flow definition could not be loaded.",
            _flow_load_task_issues(load_result.issues),
        )
        return False
    definition = load_result.definition
    task_definition = await _flow_task_definition_or_report(
        definition,
        flow_path,
        diagnostic_console,
    )
    if task_definition is None:
        return False
    try:
        flow_input = await task_cli_input(args, task_definition)
    except TaskCliInputError as exc:
        _print_task_cli_input_error(diagnostic_console, exc)
        return False
    input_issues = validate_task_input(task_definition, flow_input.value)
    if input_issues:
        _print_issues(
            diagnostic_console, "Flow input is invalid.", input_issues
        )
        return False
    if not _validate_flow_local_files(flow_input.value, diagnostic_console):
        return False
    if not _validate_task_run_output_path(args, diagnostic_console):
        return False
    try:
        result = await load_result.flow.execute_async(
            initial_node=definition.entrypoint,
            initial_inputs=flow_input_binding(
                definition.input,
                flow_input.value,
            ),
        )
    except BaseException:
        _print_task_command_error(
            diagnostic_console,
            "Flow run failed.",
            "flow.execution",
            "Fix the flow definition or node configuration and retry.",
        )
        return False
    output_issues = validate_task_output(task_definition, result)
    if output_issues:
        _print_issues(
            diagnostic_console,
            "Flow output is invalid.",
            output_issues,
        )
        return False
    if _task_run_structured_output_requested(args):
        output_written = _write_task_run_structured_output(
            args,
            console,
            diagnostic_console,
            result,
        )
        if not output_written:
            return False
    if not _task_run_json_output(args) and not _task_run_quiet(args):
        console.print("Legacy native flow run completed.", markup=False)
        console.print(f"output {_format_task_cli_value(result)}", markup=False)
    return True


async def _flow_run_with_task_context(
    args: Namespace,
    console: Console,
    *,
    theme: Theme,
    flow_path: Path,
    hub: object | None,
    logger: Logger | None,
) -> bool:
    diagnostic_console = _task_diagnostic_console(args, console)
    load_result = await _flow_metadata_loader(
        encoding=_flow_text_encoding(args)
    ).load_validation_result(flow_path)
    if load_result.definition is None:
        _print_issues(
            diagnostic_console,
            "Flow definition could not be loaded.",
            _flow_load_task_issues(load_result.issues),
        )
        return False
    definition = load_result.definition
    task_definition = await _flow_task_definition_or_report(
        definition,
        flow_path,
        diagnostic_console,
    )
    if task_definition is None:
        return False
    try:
        flow_input = await task_cli_input(args, task_definition)
    except TaskCliInputError as exc:
        _print_task_cli_input_error(diagnostic_console, exc)
        return False
    input_issues = validate_task_input(task_definition, flow_input.value)
    if input_issues:
        _print_issues(
            diagnostic_console, "Flow input is invalid.", input_issues
        )
        return False
    if not _validate_flow_local_files(flow_input.value, diagnostic_console):
        return False
    if _task_run_structured_output_requested(
        args
    ) and not _task_output_is_structured(task_definition):
        _print_task_command_error(
            diagnostic_console,
            "Flow run output is not structured.",
            "output.unsupported",
            "Use --json or --output with json, object, or array outputs.",
        )
        return False
    if not _validate_task_run_output_path(args, diagnostic_console):
        return False
    tool_manager = _flow_tool_manager(args)
    resolver_issues = _flow_missing_tool_resolver_issues(
        definition,
        tool_manager,
    )
    if resolver_issues:
        _print_issues(
            diagnostic_console,
            "Flow definition could not be loaded.",
            resolver_issues,
        )
        return False
    monitor = _flow_run_progress_monitor(args, console, theme, definition)
    event_observer: TaskSanitizedEventObserver | None = None
    if monitor is not None:
        event_observer = monitor.observe
    try:
        if monitor is not None:
            monitor.start()
        async with AsyncExitStack() as stack:
            tool_resolver: FlowToolResolver | None = None
            if tool_manager is not None:
                tool_resolver = await stack.enter_async_context(tool_manager)
            client_context = _task_cli_client_context(
                flow_path,
                dsn=None,
                schema=None,
                queue=False,
                ephemeral=True,
                hub=hub,
                logger=logger,
                input_value=flow_input.value,
                flow_tool_resolver=tool_resolver,
                flow_concurrency_limit=getattr(args, "flow_parallel", 1),
                event_observer=event_observer,
            )
            async with client_context as client:
                result = await client.run(
                    task_definition,
                    input_value=flow_input.value,
                    metadata=_task_command_metadata(ephemeral=True),
                )
                if (
                    monitor is not None
                    and result.run.state == TaskRunState.SUCCEEDED
                ):
                    monitor.complete()
    except (AssertionError, ImportError, OSError, TaskValidationError) as exc:
        _print_task_execution_error(diagnostic_console, exc)
        return False
    finally:
        if monitor is not None:
            monitor.stop()
    if result.run.state != TaskRunState.SUCCEEDED:
        _print_task_run_failure(diagnostic_console, result)
        return False
    if _task_run_structured_output_requested(args):
        output_written = _write_task_run_structured_output(
            args,
            console,
            diagnostic_console,
            result.output,
        )
        if not output_written:
            return False
    if not _task_run_json_output(args) and not _task_run_quiet(args):
        console.print("Flow run completed.", markup=False)
        console.print(
            f"output {_format_task_cli_value(result.output)}",
            markup=False,
        )
    return True


@dataclass(slots=True)
class _FlowRunNodeStats:
    started_at: float | None = None
    elapsed_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    usage_reasoning_tokens: int = 0
    tools_executed: int = 0

    def snapshot(self, now: float) -> dict[str, int | float]:
        elapsed_ms = self.elapsed_ms
        if self.started_at is not None:
            elapsed_ms = max(elapsed_ms, (now - self.started_at) * 1000)
        return {
            "elapsed_ms": elapsed_ms,
            "input_tokens": max(self.input_tokens, self.usage_input_tokens),
            "output_tokens": max(self.output_tokens, self.usage_output_tokens),
            "reasoning_tokens": max(
                self.reasoning_tokens,
                self.usage_reasoning_tokens,
            ),
            "tools_executed": self.tools_executed,
        }


@dataclass(slots=True)
class _FlowRunProgressMonitor:
    console: Console
    theme: Theme
    mermaid_source: str
    node_states: dict[str, str]
    active_nodes: set[str] = field(default_factory=set)
    node_stats: dict[str, _FlowRunNodeStats] = field(default_factory=dict)
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None
    message: str = "Flow run is starting."
    live: Live | None = field(default=None, init=False)

    def start(self) -> None:
        self.live = Live(
            self,
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start(refresh=True)

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def complete(self) -> None:
        self._apply_event("flow_completed", node=None, status=None)
        self.message = self.theme.flow_run_progress_message("flow_completed")
        if self.live is not None:
            self.live.update(self, refresh=True)

    def observe(self, event: TaskObservedEvent) -> None:
        event_type = getattr(event, "event_type", None)
        if not isinstance(event_type, str):
            return
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            payload = {}
        self._apply_stats(event_type, payload)
        if not event_type.startswith("flow_"):
            if self.live is not None:
                self.live.update(self, refresh=True)
            return
        node = _flow_progress_payload_string(payload, "node")
        status = _flow_progress_payload_string(payload, "status")
        attempt = _flow_progress_payload_int(payload, "attempt")
        if attempt is None:
            attempt = _flow_progress_payload_int(payload, "attempts")
        flow_name = _flow_progress_payload_string(payload, "flow_name")
        self._apply_event(event_type, node=node, status=status)
        self.message = self.theme.flow_run_progress_message(
            event_type,
            node=node,
            status=status,
            attempt=attempt,
            flow_name=flow_name,
        )
        if self.live is not None:
            self.live.update(self, refresh=True)

    def render(self) -> RenderableType:
        return self.theme.flow_run_progress(
            self.mermaid_source,
            node_states=self.node_states,
            active_nodes=tuple(sorted(self.active_nodes)),
            message=self.message,
            console_width=self.console.width,
            flow_stats=self._flow_stats_snapshot(),
        )

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        _ = console, options
        yield self.render()

    def _apply_event(
        self,
        event_type: str,
        *,
        node: str | None,
        status: str | None,
    ) -> None:
        if event_type == "flow_completed":
            self.finished_at = monotonic()
            self.active_nodes.clear()
            return
        if node is None or node not in self.node_states:
            return
        node_stats = self._node_stats(node)
        match event_type:
            case "flow_node_started":
                self.node_states[node] = "running"
                self.active_nodes.add(node)
                if node_stats.started_at is None:
                    node_stats.started_at = monotonic()
            case "flow_node_retrying":
                self.node_states[node] = "retrying"
                self.active_nodes.add(node)
                if node_stats.started_at is None:
                    node_stats.started_at = monotonic()
            case "flow_node_completed":
                self.node_states[node] = status or "succeeded"
                self.active_nodes.discard(node)
                self._finish_node_stats(node_stats)
            case "flow_node_failed" | "flow_node_cancelled":
                self.node_states[node] = status or "failed"
                self.active_nodes.discard(node)
                self._finish_node_stats(node_stats)
            case "flow_node_skipped":
                self.node_states[node] = "skipped"
                self.active_nodes.discard(node)
            case "flow_node_paused":
                self.node_states[node] = "paused"
                self.active_nodes.discard(node)
                self._finish_node_stats(node_stats)
            case "flow_node_resumed":
                self.node_states[node] = "resumed"
                self.active_nodes.discard(node)

    def _finish_node_stats(self, node_stats: _FlowRunNodeStats) -> None:
        if node_stats.started_at is not None:
            node_stats.elapsed_ms = max(
                node_stats.elapsed_ms,
                (monotonic() - node_stats.started_at) * 1000,
            )
        node_stats.started_at = None

    def _apply_stats(
        self,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        if event_type.startswith("flow_"):
            self._apply_flow_stats(payload)
            return
        node = _flow_progress_payload_string(payload, "flow_node")
        if node is None and len(self.active_nodes) == 1:
            node = next(iter(self.active_nodes))
        if node is None or node not in self.node_states:
            return
        node_stats = self._node_stats(node)
        match event_type:
            case "input_token_count_after":
                count = _flow_progress_payload_non_negative_int(
                    payload,
                    "count",
                )
                if count is not None:
                    node_stats.input_tokens += count
            case "token_generated":
                token_type = _flow_progress_payload_string(
                    payload,
                    "token_type",
                )
                if token_type == "ReasoningToken":
                    node_stats.reasoning_tokens += 1
                else:
                    node_stats.output_tokens += 1
            case "tool_execute":
                node_stats.tools_executed += 1
            case "usage_observed":
                self._apply_usage_stats(node_stats, payload)

    def _apply_usage_stats(
        self,
        node_stats: _FlowRunNodeStats,
        payload: Mapping[str, object],
    ) -> None:
        input_tokens = _flow_progress_payload_non_negative_int(
            payload,
            "input_tokens",
        )
        if input_tokens is not None:
            node_stats.usage_input_tokens += input_tokens
        output_tokens = _flow_progress_payload_non_negative_int(
            payload,
            "output_tokens",
        )
        if output_tokens is not None:
            node_stats.usage_output_tokens += output_tokens
        reasoning_tokens = _flow_progress_payload_non_negative_int(
            payload,
            "reasoning_tokens",
        )
        if reasoning_tokens is not None:
            node_stats.usage_reasoning_tokens += reasoning_tokens

    def _apply_flow_stats(
        self,
        payload: Mapping[str, object],
    ) -> None:
        node = _flow_progress_payload_string(payload, "node")
        if node is None or node not in self.node_states:
            return
        duration_ms = _flow_progress_payload_non_negative_number(
            payload,
            "duration_ms",
        )
        if duration_ms is None:
            return
        self._node_stats(node).elapsed_ms = duration_ms

    def _flow_stats_snapshot(self) -> dict[str, dict[str, int | float]]:
        now = monotonic()
        snapshots = {
            node: self._node_stats(node).snapshot(now)
            for node in self.node_states
        }
        executed_states = {"succeeded", "completed", "failed", "cancelled"}
        succeeded_states = {"succeeded", "completed"}
        failed_states = {"failed", "cancelled"}
        elapsed_nodes = [
            values["elapsed_ms"]
            for values in snapshots.values()
            if values["elapsed_ms"] > 0
        ]
        total = {
            "elapsed_ms": ((self.finished_at or now) - self.started_at) * 1000,
            "executed_nodes": sum(
                1
                for state in self.node_states.values()
                if state in executed_states
            ),
            "succeeded_nodes": sum(
                1
                for state in self.node_states.values()
                if state in succeeded_states
            ),
            "failed_nodes": sum(
                1
                for state in self.node_states.values()
                if state in failed_states
            ),
            "average_node_ms": (
                sum(elapsed_nodes) / len(elapsed_nodes) if elapsed_nodes else 0
            ),
            "input_tokens": sum(
                values["input_tokens"] for values in snapshots.values()
            ),
            "output_tokens": sum(
                values["output_tokens"] for values in snapshots.values()
            ),
            "reasoning_tokens": sum(
                values["reasoning_tokens"] for values in snapshots.values()
            ),
            "tools_executed": sum(
                values["tools_executed"] for values in snapshots.values()
            ),
        }
        return {"__total__": total, **snapshots}

    def _node_stats(self, node: str) -> _FlowRunNodeStats:
        return self.node_stats.setdefault(node, _FlowRunNodeStats())


def _flow_run_progress_monitor(
    args: Namespace,
    console: Console,
    theme: Theme,
    definition: FlowDefinition,
) -> _FlowRunProgressMonitor | None:
    if _task_run_json_output(args) or _task_run_quiet(args):
        return None
    if not callable(getattr(theme, "flow_run_progress", None)):
        return None
    if not callable(getattr(theme, "flow_run_progress_message", None)):
        return None
    return _FlowRunProgressMonitor(
        console=console,
        theme=theme,
        mermaid_source=_flow_definition_progress_mermaid_source(definition),
        node_states={node.name: "pending" for node in definition.nodes},
        node_stats={
            node.name: _FlowRunNodeStats() for node in definition.nodes
        },
    )


def _flow_definition_progress_mermaid_source(
    definition: FlowDefinition,
) -> str:
    view = FlowView(
        import_mode=FlowViewImportMode.EXECUTABLE,
        direction=FlowViewDirection.LR,
        nodes=tuple(
            FlowViewNode(id=node.name, label=node.name)
            for node in definition.nodes
        ),
        edges=tuple(
            FlowViewEdge(
                id=f"edge_{index}",
                source=edge.source,
                target=edge.target,
                label=edge.label,
            )
            for index, edge in enumerate(definition.edges)
        ),
    )
    rendered = render_flow_view(view)
    if rendered.ok:
        return rendered.source
    return _flow_definition_progress_fallback_source(definition)


def _flow_definition_progress_fallback_source(
    definition: FlowDefinition,
) -> str:
    lines = ["flowchart LR"]
    for node in definition.nodes:
        lines.append(f"  {node.name}")
    for edge in definition.edges:
        lines.append(f"  {edge.source} --> {edge.target}")
    return "\n".join(lines) + "\n"


def _flow_progress_payload_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _flow_progress_payload_int(
    payload: Mapping[str, object],
    key: str,
) -> int | None:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _flow_progress_payload_non_negative_int(
    payload: Mapping[str, object],
    key: str,
) -> int | None:
    value = _flow_progress_payload_int(payload, key)
    if value is None or value < 0:
        return None
    return value


def _flow_progress_payload_non_negative_number(
    payload: Mapping[str, object],
    key: str,
) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value < 0:
        return None
    return float(value)


@dataclass(slots=True)
class _FlowStateStoreContext:
    store: FlowStateStore
    database: object | None = None

    async def __aenter__(self) -> FlowStateStore:
        if self.database is not None:
            open_database = getattr(self.database, "open")
            opened = open_database()
            if hasattr(opened, "__await__"):
                await opened
        return self.store

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        _ = exc_type, exc, traceback
        if self.database is not None:
            close_database = getattr(self.database, "aclose")
            closed = close_database()
            if hasattr(closed, "__await__"):
                await closed
        return None


def _flow_state_store_context(
    args: Namespace,
    console: Console,
) -> _FlowStateStoreContext | None:
    dsn = _task_store_dsn(args)
    if dsn is None:
        _print_missing_inspection_store(console)
        return None
    database = _task_pgsql_database(dsn, _task_store_schema(args))
    return _FlowStateStoreContext(
        store=PgsqlFlowStateStore(database),
        database=database,
    )


async def _flow_resume_decisions(
    args: Namespace,
    console: Console,
) -> Mapping[str, Mapping[str, object]] | None:
    raw_value = getattr(args, "decision_json", None)
    if not isinstance(raw_value, str) or not raw_value.strip():
        _print_task_command_error(
            console,
            "Flow resume decision JSON is required.",
            "flow.resume_decision_missing",
            "Pass --decision-json with a node-to-decision object.",
        )
        return None
    source = raw_value.strip()
    if source.startswith("@"):
        try:
            source = await read_text(
                source[1:],
                encoding=_flow_text_encoding(args),
            )
        except OSError:
            _print_task_command_error(
                console,
                "Flow resume decision JSON could not be read.",
                "file.read",
                "Use a readable local JSON file.",
            )
            return None
    try:
        value = loads(source)
    except JSONDecodeError:
        _print_task_command_error(
            console,
            "Flow resume decision JSON is invalid.",
            "flow.resume_decision_json",
            "Use a JSON object keyed by review node.",
        )
        return None
    if not isinstance(value, Mapping):
        _print_task_command_error(
            console,
            "Flow resume decisions are invalid.",
            "flow.resume_decision_shape",
            "Use a JSON object keyed by review node.",
        )
        return None
    decisions: dict[str, Mapping[str, object]] = {}
    for node, payload in value.items():
        if (
            not isinstance(node, str)
            or not node.strip()
            or not isinstance(payload, Mapping)
        ):
            _print_task_command_error(
                console,
                "Flow resume decisions are invalid.",
                "flow.resume_decision_shape",
                "Map each review node to a decision payload object.",
            )
            return None
        decisions[node] = {str(key): item for key, item in payload.items()}
    return decisions


def _print_flow_runtime_value(
    args: Namespace,
    console: Console,
    *,
    label: str,
    value: object,
) -> None:
    serialized = _format_task_cli_value(value)
    if _flow_json_output(args):
        console.file.write(serialized + "\n")
        console.file.flush()
        return
    console.print(f"{label} {serialized}", markup=False, soft_wrap=True)


def _print_flow_inspection_error(
    console: Console,
    error: BaseException,
) -> None:
    console.print("Flow inspection failed.", markup=False)
    if isinstance(error, TaskStoreNotFoundError):
        console.print("error flow.not_found", markup=False)
        return
    if isinstance(error, ImportError):
        console.print("error dependency.missing", markup=False)
        return
    if isinstance(error, OSError):
        console.print("error io.failure", markup=False)
        return
    console.print("error flow.inspection", markup=False)


def _print_flow_command_error(
    console: Console,
    error: BaseException,
) -> None:
    console.print("Flow command failed.", markup=False)
    if isinstance(error, TaskStoreNotFoundError):
        console.print("error flow.not_found", markup=False)
        return
    if isinstance(error, TaskStoreConflictError):
        console.print("error flow.conflict", markup=False)
        return
    if isinstance(error, TaskClientUnsupportedOperationError):
        console.print(f"error {error.code}", markup=False)
        return
    if isinstance(error, ImportError):
        console.print("error dependency.missing", markup=False)
        return
    if isinstance(error, OSError):
        console.print("error io.failure", markup=False)
        return
    console.print("error flow.command", markup=False)


def _flow_missing_tool_resolver_issues(
    definition: FlowDefinition,
    resolver: FlowToolResolver | None,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(definition, FlowDefinition)
    if resolver is not None:
        return ()
    return tuple(
        TaskValidationIssue(
            code="flow.unsupported_node_type",
            path=f"nodes.{node.name}.type",
            message="Flow node type is not supported by this runtime.",
            hint="Use --tool or --tools to enable strict flow tool nodes.",
            category=TaskValidationCategory.UNSUPPORTED,
        )
        for node in definition.nodes
        if node.type == "tool"
    )


def _flow_tool_manager(args: Namespace) -> ToolManager | None:
    enabled_tools = _flow_enabled_tools(args)
    if not enabled_tools:
        return None
    available_toolsets: list[ToolSet] = [MathToolSet(namespace="math")]
    if HAS_GRAPH_DEPENDENCIES:
        available_toolsets.append(
            GraphToolSet(
                settings=get_tool_settings(
                    args,
                    prefix="graph",
                    settings_cls=GraphToolSettings,
                )
                or GraphToolSettings(),
                namespace="graph",
            )
        )
    if HAS_CODE_DEPENDENCIES:
        available_toolsets.append(CodeToolSet(namespace="code"))
    if HAS_BROWSER_DEPENDENCIES:
        available_toolsets.append(
            BrowserToolSet(
                settings=get_tool_settings(
                    args,
                    prefix="browser",
                    settings_cls=BrowserToolSettings,
                )
                or BrowserToolSettings(),
                namespace="browser",
            )
        )
    database_settings = get_tool_settings(
        args,
        prefix="database",
        settings_cls=DatabaseToolSettings,
    )
    if database_settings is not None:
        available_toolsets.append(
            DatabaseToolSet(settings=database_settings, namespace="database")
        )
    return ToolManager.create_instance(
        available_toolsets=available_toolsets,
        enable_tools=enabled_tools,
    )


def _flow_enabled_tools(args: Namespace) -> list[str]:
    tools = (getattr(args, "tool", None) or []) + (
        getattr(args, "tools", None) or []
    )
    assert isinstance(tools, list)
    for tool in tools:
        assert isinstance(tool, str) and tool.strip()
    return tools


async def _flow_task_definition(
    definition: FlowDefinition,
    flow_path: Path,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(
            name=definition.name, version=definition.version or "1"
        ),
        input=_flow_task_input(definition),
        output=await _flow_task_output(definition),
        execution=TaskExecutionTarget(
            type=TaskTargetType.FLOW,
            ref=flow_path.name,
        ),
        privacy=_flow_task_privacy(definition),
        definition_base=flow_path,
    )


async def _flow_task_definition_or_report(
    definition: FlowDefinition,
    flow_path: Path,
    console: Console,
) -> TaskDefinition | None:
    try:
        return await _flow_task_definition(definition, flow_path)
    except TaskSchemaResolutionError:
        _print_issues(
            console,
            "Flow definition could not be loaded.",
            (_flow_schema_issue(),),
        )
        return None


def _flow_task_input(definition: FlowDefinition) -> TaskInputContract:
    input_definition = _flow_primary_input(definition)
    if input_definition is None:
        return TaskInputContract.object(_flow_task_inputs_schema(definition))
    match input_definition.type:
        case FlowInputType.STRING:
            return TaskInputContract.string()
        case FlowInputType.INTEGER:
            return TaskInputContract.integer()
        case FlowInputType.NUMBER:
            return TaskInputContract.number()
        case FlowInputType.BOOLEAN:
            return TaskInputContract.boolean()
        case FlowInputType.OBJECT:
            return TaskInputContract.object(
                _plain_mapping(input_definition.schema) or {"type": "object"},
            )
        case FlowInputType.ARRAY:
            return TaskInputContract.array(
                _plain_mapping(input_definition.schema) or {"type": "array"}
            )
        case FlowInputType.FILE:
            return TaskInputContract.file(
                mime_types=input_definition.mime_types,
            )
        case FlowInputType.FILE_ARRAY:
            return TaskInputContract.file_array(
                mime_types=input_definition.mime_types,
            )
    raise AssertionError("unsupported flow input type")  # pragma: no cover


def _flow_task_inputs_schema(
    definition: FlowDefinition,
) -> Mapping[str, object]:
    properties: dict[str, object] = {
        input_definition.name: _flow_task_input_property_schema(
            input_definition
        )
        for input_definition in definition.inputs
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }


def _flow_task_input_property_schema(
    input_definition: FlowInputDefinition,
) -> Mapping[str, object]:
    match input_definition.type:
        case FlowInputType.STRING:
            return {"type": "string"}
        case FlowInputType.INTEGER:
            return {"type": "integer"}
        case FlowInputType.NUMBER:
            return {"type": "number"}
        case FlowInputType.BOOLEAN:
            return {"type": "boolean"}
        case FlowInputType.OBJECT:
            return _plain_mapping(input_definition.schema) or {
                "type": "object"
            }
        case FlowInputType.ARRAY:
            return _plain_mapping(input_definition.schema) or {"type": "array"}
        case FlowInputType.FILE:
            return _flow_task_file_property_schema(input_definition)
        case FlowInputType.FILE_ARRAY:
            return {
                "type": "array",
                "items": _flow_task_file_property_schema(input_definition),
                _AVALAN_INPUT_TYPE_SCHEMA_KEY: FlowInputType.FILE_ARRAY.value,
                _AVALAN_MIME_TYPES_SCHEMA_KEY: input_definition.mime_types,
            }
    raise AssertionError("unsupported flow input type")  # pragma: no cover


def _flow_task_file_property_schema(
    input_definition: FlowInputDefinition,
) -> Mapping[str, object]:
    return {
        "type": "object",
        "required": ["source_kind", "reference"],
        "additionalProperties": True,
        "properties": {
            "source_kind": {"type": "string"},
            "reference": {"type": "string"},
            "mime_type": {"type": "string"},
        },
        _AVALAN_INPUT_TYPE_SCHEMA_KEY: FlowInputType.FILE.value,
        _AVALAN_MIME_TYPES_SCHEMA_KEY: input_definition.mime_types,
    }


async def _flow_task_output(definition: FlowDefinition) -> TaskOutputContract:
    output_definition = _flow_primary_output(definition)
    if output_definition is None:
        return TaskOutputContract.json({})
    schema = await _flow_output_schema(definition)
    match output_definition.type:
        case FlowOutputType.TEXT:
            return TaskOutputContract.text()
        case FlowOutputType.JSON:
            return TaskOutputContract.json(schema or {})
        case FlowOutputType.OBJECT:
            return TaskOutputContract.object(schema or {"type": "object"})
        case FlowOutputType.ARRAY:
            return TaskOutputContract.array(schema or {"type": "array"})
        case FlowOutputType.FILE:
            return TaskOutputContract.file()
        case FlowOutputType.FILE_ARRAY:
            return TaskOutputContract.file_array()
    raise AssertionError("unsupported flow output type")  # pragma: no cover


def _flow_task_privacy(definition: FlowDefinition) -> TaskPrivacyPolicy:
    overrides: dict[str, str | int] = {}
    for key, value in definition.privacy_policy.items():
        if key == "raw_retention_days":
            assert isinstance(value, int)
            overrides[key] = value
            continue
        assert isinstance(value, str)
        overrides[key] = value
    return privacy_policy_with_defaults(overrides or None)


async def _flow_output_schema(
    definition: FlowDefinition,
) -> Mapping[str, object] | None:
    output_definition = _flow_primary_output(definition)
    if output_definition is None:
        return None
    schema = _plain_mapping(output_definition.schema)
    if schema is not None or output_definition.schema_ref is None:
        return schema
    resolved = await resolve_schema_ref(
        output_definition.schema_ref,
        schema_base_path=definition.definition_base,
        path="flow.output.schema_ref",
    )
    return _plain_mapping(resolved.schema)


def _flow_primary_input(
    definition: FlowDefinition,
) -> FlowInputDefinition | None:
    if definition.input is not None:
        return definition.input
    if len(definition.inputs) == 1:
        return definition.inputs[0]
    return None


def _flow_primary_output(
    definition: FlowDefinition,
) -> FlowOutputDefinition | None:
    if definition.output is not None:
        return definition.output
    if len(definition.outputs) == 1:
        return definition.outputs[0]
    return None


def _flow_load_task_issues(
    issues: tuple[FlowLoadIssue, ...],
) -> tuple[TaskValidationIssue, ...]:
    return tuple(
        TaskValidationIssue(
            code=issue.code,
            path=issue.path,
            message=issue.message,
            hint=issue.hint,
            category=_flow_task_validation_category(issue.category),
        )
        for issue in issues
    )


def _flow_task_validation_category(
    category: FlowLoadIssueCategory,
) -> TaskValidationCategory:
    match category:
        case FlowLoadIssueCategory.PARSE | FlowLoadIssueCategory.STRUCTURE:
            return TaskValidationCategory.STRUCTURE
        case FlowLoadIssueCategory.VALUE:
            return TaskValidationCategory.VALUE
        case FlowLoadIssueCategory.UNSUPPORTED:
            return TaskValidationCategory.UNSUPPORTED
        case FlowLoadIssueCategory.PRIVACY:
            return TaskValidationCategory.PRIVACY
    raise AssertionError("unsupported flow issue category")  # pragma: no cover


def _flow_needs_task_context(
    issues: tuple[FlowLoadIssue, ...],
) -> bool:
    return any(issue.code == "flow.unsupported_node_type" for issue in issues)


def _flow_schema_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="output.invalid_schema",
        path="flow.output.schema_ref",
        message="Flow output schema reference is invalid.",
        hint="Use a local JSON object schema inside the flow directory.",
        category=TaskValidationCategory.VALUE,
    )


def _flow_metadata_loader(
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowDefinitionLoader:
    registry = default_flow_node_registry()
    for node_type in ("agent", "file_convert", "pdf_to_images", "tool"):
        metadata: FlowNodeMetadata
        if node_type == "agent":
            metadata = FlowNodeMetadata(
                kind=FlowNodeKind.AGENT,
                supports_ref=True,
                async_only=True,
                input_contracts=(
                    FlowNodeContract(name="input", type="any"),
                    FlowNodeContract(
                        name=None,
                        type="object",
                        metadata={"dynamic": True},
                    ),
                ),
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.JSON,
                    metadata={"dynamic": True},
                ),
            )
        elif node_type == "tool":
            metadata = FlowNodeMetadata(
                kind=FlowNodeKind.TOOL,
                supports_ref=True,
                async_only=True,
                input_contract=FlowNodeContract(
                    name="arguments",
                    type=FlowInputType.OBJECT,
                    metadata={"dynamic": True},
                ),
                output_contract=FlowNodeContract(
                    name="result",
                    type=FlowOutputType.JSON,
                    metadata={"dynamic": True},
                ),
                requires_ref=True,
            )
        else:
            metadata = FlowNodeMetadata(
                kind=FlowNodeKind.FILE_CONVERSION,
                async_only=True,
                input_contract=FlowNodeContract(
                    name="files",
                    type=FlowInputType.FILE_ARRAY,
                ),
                output_contract=FlowNodeContract(
                    name="files",
                    type=FlowOutputType.FILE_ARRAY,
                ),
            )
        registry.register(
            node_type,
            _flow_task_context_metadata_node,
            metadata=metadata,
        )
    return FlowDefinitionLoader(registry, encoding=encoding)


def _flow_task_context_metadata_node(definition: FlowNodeDefinition) -> Node:
    async def run(_: dict[str, object]) -> object:
        raise RuntimeError("Flow node requires task execution context.")

    return Node(definition.name, func=run)


def _plain_mapping(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_value(item) for item in value]
    return value


def _validate_flow_local_files(value: object, console: Console) -> bool:
    for descriptor in _flow_local_file_descriptors(value):
        reference = descriptor.get("reference")
        if not isinstance(reference, str) or not Path(reference).is_file():
            _print_task_command_error(
                console,
                "Flow input file could not be read.",
                "input.file_missing",
                "Pass a readable local file path.",
            )
            return False
    return True


def _flow_local_file_descriptors(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    descriptors: list[Mapping[str, object]] = []
    _collect_flow_local_file_descriptors(value, descriptors)
    return tuple(descriptors)


def _collect_flow_local_file_descriptors(
    value: object,
    descriptors: list[Mapping[str, object]],
) -> None:
    if isinstance(value, Mapping):
        if value.get("source_kind") == "local_path":
            descriptors.append(value)
            return
        for item in value.values():
            _collect_flow_local_file_descriptors(item, descriptors)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect_flow_local_file_descriptors(item, descriptors)
