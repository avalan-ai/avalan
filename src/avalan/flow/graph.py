from .condition import FlowCondition
from .definition import (
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowMetadata,
    FlowNodeDefinition,
    FlowRouteMatchPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)
from .mermaid import (
    MermaidAstEdge,
    MermaidAstEdgeStatement,
    MermaidAstNode,
    MermaidAstNodeStatement,
    MermaidAstStatement,
    MermaidAstSubgraph,
    is_executable_mermaid_edge_id,
    parse_mermaid_import,
)
from .view import FlowViewImportMode

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from urllib.parse import urlsplit


class FlowGraphFormat(StrEnum):
    MERMAID = "mermaid"


class FlowGraphSourceKind(StrEnum):
    INLINE = "inline"
    FILE = "file"


class FlowGraphMode(StrEnum):
    EXECUTABLE = "executable"


class FlowGraphNodeClassification(StrEnum):
    ACTUAL = "actual"
    DECORATIVE = "decorative"


class FlowGraphEdgeClassification(StrEnum):
    EXECUTABLE = "executable"
    DECORATIVE = "decorative"


class FlowGraphBindingState(StrEnum):
    BOUND = "bound"
    UNBOUND = "unbound"
    MISSING = "missing"
    DECORATIVE = "decorative"
    REJECTED = "rejected"


class FlowGraphDiagnosticCode(StrEnum):
    MALFORMED_SOURCE = "flow.graph.malformed_source"
    UNSUPPORTED_FORMAT = "flow.graph.unsupported_format"
    UNSUPPORTED_SOURCE = "flow.graph.unsupported_source"
    UNSUPPORTED_MODE = "flow.graph.unsupported_mode"
    MISSING_SOURCE = "flow.graph.missing_source"
    READ_FAILURE = "flow.graph.read_failure"
    PATH_ESCAPE = "flow.graph.path_escape"
    SOURCE_CONFLICT = "flow.graph.source_conflict"
    EDGE_CONFLICT = "flow.graph.edge_conflict"
    MISSING_EDGE_METADATA_TARGET = "flow.graph.missing_edge_metadata"
    DECORATIVE_EDGE_METADATA_TARGET = "flow.graph.decorative_edge_metadata"
    DUPLICATE_EDGE_ID = "flow.graph.duplicate_edge_id"
    INVALID_EDGE_ID = "flow.graph.invalid_edge_id"
    UNSUPPORTED_EXECUTABLE_EDGE = "flow.graph.unsupported_executable_edge"


_FLOW_GRAPH_EDGE_BINDING_FIELDS = frozenset(
    {
        "condition",
        "default",
        "kind",
        "label",
        "priority",
        "routing_policy",
    }
)
_FLOW_GRAPH_DIAGNOSTIC_MESSAGES = MappingProxyType(
    {
        FlowGraphDiagnosticCode.MALFORMED_SOURCE: (
            "Graph source is malformed.",
            "Fix the graph source syntax.",
        ),
        FlowGraphDiagnosticCode.UNSUPPORTED_FORMAT: (
            "Graph format is not supported.",
            "Use a supported graph format.",
        ),
        FlowGraphDiagnosticCode.UNSUPPORTED_SOURCE: (
            "Graph source type is not supported.",
            "Use an inline diagram or local file path.",
        ),
        FlowGraphDiagnosticCode.UNSUPPORTED_MODE: (
            "Graph mode is not supported.",
            "Use executable graph mode.",
        ),
        FlowGraphDiagnosticCode.MISSING_SOURCE: (
            "Graph source is missing.",
            "Provide exactly one graph source.",
        ),
        FlowGraphDiagnosticCode.READ_FAILURE: (
            "Graph source could not be read.",
            "Check that the graph file is available to the loader.",
        ),
        FlowGraphDiagnosticCode.PATH_ESCAPE: (
            "Graph source path is outside the allowed base.",
            "Use a graph path inside the flow definition directory.",
        ),
        FlowGraphDiagnosticCode.SOURCE_CONFLICT: (
            "Graph source is ambiguous.",
            "Provide exactly one graph source.",
        ),
        FlowGraphDiagnosticCode.EDGE_CONFLICT: (
            "Graph edges conflict with strict edges.",
            "Use either graph authoring or strict edge definitions.",
        ),
        FlowGraphDiagnosticCode.MISSING_EDGE_METADATA_TARGET: (
            "Graph edge metadata targets a missing edge.",
            "Bind graph edge metadata to an explicit Mermaid edge ID.",
        ),
        FlowGraphDiagnosticCode.DECORATIVE_EDGE_METADATA_TARGET: (
            "Graph edge metadata targets a decorative edge.",
            "Bind graph edge metadata only to executable graph edges.",
        ),
        FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID: (
            "Graph edge ID is duplicated.",
            "Use unique explicit Mermaid edge IDs.",
        ),
        FlowGraphDiagnosticCode.INVALID_EDGE_ID: (
            "Graph edge ID is invalid.",
            "Use a TOML-key-safe Mermaid edge ID.",
        ),
        FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE: (
            "Graph edge is not supported for execution.",
            "Use explicit directed graph edges for executable routes.",
        ),
    }
)
_FLOW_GRAPH_PARSE_CODES = frozenset(
    {
        FlowGraphDiagnosticCode.MALFORMED_SOURCE,
        FlowGraphDiagnosticCode.READ_FAILURE,
    }
)
_FLOW_GRAPH_UNSUPPORTED_CODES = frozenset(
    {
        FlowGraphDiagnosticCode.UNSUPPORTED_FORMAT,
        FlowGraphDiagnosticCode.UNSUPPORTED_SOURCE,
        FlowGraphDiagnosticCode.UNSUPPORTED_MODE,
        FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
    }
)
_FLOW_GRAPH_INSPECTABLE_IMPORT_FAILURE_CODES = frozenset(
    {
        FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID,
        FlowGraphDiagnosticCode.INVALID_EDGE_ID,
        FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
    }
)
_MERMAID_IMPORT_GRAPH_DIAGNOSTIC_CODES = MappingProxyType(
    {
        "flow.mermaid.security.ambiguous_shorthand": (
            FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE
        ),
        "flow.mermaid.security.bidirectional_edge": (
            FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE
        ),
        "flow.mermaid.security.duplicate_edge_id": (
            FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID
        ),
        "flow.mermaid.security.invalid_edge_id": (
            FlowGraphDiagnosticCode.INVALID_EDGE_ID
        ),
    }
)
_MERMAID_IMPORT_GRAPH_DIAGNOSTIC_PATHS = MappingProxyType(
    {
        FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID: "graph.edges",
        FlowGraphDiagnosticCode.INVALID_EDGE_ID: "graph.edges",
        FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE: "graph.edges",
    }
)


def _empty_mapping() -> FlowMetadata:
    return MappingProxyType({})


def _empty_edge_bindings() -> Mapping[str, "FlowGraphEdgeBinding"]:
    return MappingProxyType({})


def _empty_string_tuple() -> tuple[str, ...]:
    return ()


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_diagnostics(value: tuple[FlowDiagnostic, ...]) -> None:
    assert isinstance(value, tuple), "diagnostics must be a tuple"
    for diagnostic in value:
        assert isinstance(diagnostic, FlowDiagnostic)


def _assert_source_span(
    value: FlowSourceSpan | None,
    field_name: str,
) -> None:
    if value is not None:
        assert isinstance(
            value, FlowSourceSpan
        ), f"{field_name} must be a source span"


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


def _assert_safe_diagnostic_path(value: str) -> None:
    _assert_non_empty_string(value, "path")
    unsafe_fragments = (":", "/", "\\", "{", "}", "\n", "\r")
    assert not any(
        fragment in value for fragment in unsafe_fragments
    ), "path must be a safe diagnostic path"


def _assert_tuple_items(
    values: tuple[object, ...],
    field_name: str,
    item_type: type[object],
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(value, item_type)


def _freeze_mapping(value: FlowMetadata, *, field_name: str) -> FlowMetadata:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    frozen: dict[str, object] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        frozen[key] = _freeze_value(item, field_name=field_name)
    return MappingProxyType(frozen)


def _freeze_edge_bindings(
    value: Mapping[str, "FlowGraphEdgeBinding"],
) -> Mapping[str, "FlowGraphEdgeBinding"]:
    assert isinstance(value, Mapping), "edge_bindings must be a mapping"
    frozen: dict[str, FlowGraphEdgeBinding] = {}
    for key, binding in value.items():
        _assert_non_empty_string(key, "edge_bindings key")
        assert isinstance(binding, FlowGraphEdgeBinding)
        assert key == binding.edge_id, "edge binding key must match edge_id"
        frozen[key] = binding
    return MappingProxyType(frozen)


def _freeze_value(value: object, *, field_name: str) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, field_name=field_name)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_value(item, field_name=field_name) for item in value
        )
    return value


def _flow_edge_definition_as_public_dict(
    index: int,
    edge: FlowEdgeDefinition,
) -> dict[str, object]:
    value: dict[str, object] = {
        "index": index,
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind.value,
        "priority": edge.priority,
        "default": edge.default,
        "routing_policy": edge.routing_policy.value,
    }
    if edge.label is not None:
        value["has_label"] = True
    if edge.condition is not None:
        value["has_condition"] = True
    return value


def flow_graph_diagnostic(
    code: FlowGraphDiagnosticCode,
    path: str,
    *,
    source_span: FlowSourceSpan | None = None,
    related_spans: tuple[FlowSourceSpan, ...] = (),
    severity: FlowDiagnosticSeverity = FlowDiagnosticSeverity.ERROR,
) -> FlowDiagnostic:
    assert isinstance(code, FlowGraphDiagnosticCode)
    _assert_safe_diagnostic_path(path)
    _assert_source_span(source_span, "source_span")
    assert isinstance(related_spans, tuple), "related_spans must be a tuple"
    for span in related_spans:
        assert isinstance(span, FlowSourceSpan)
    assert isinstance(severity, FlowDiagnosticSeverity)
    message, hint = _FLOW_GRAPH_DIAGNOSTIC_MESSAGES[code]
    return FlowDiagnostic(
        code=code.value,
        category=FlowDiagnosticCategory.GRAPH_COMPILER,
        path=path,
        source_span=source_span,
        severity=severity,
        message=message,
        hint=hint,
        related_spans=related_spans,
    )


def flow_graph_diagnostic_load_category(
    diagnostic: FlowDiagnostic,
) -> str:
    assert isinstance(diagnostic, FlowDiagnostic)
    assert (
        diagnostic.category == FlowDiagnosticCategory.GRAPH_COMPILER
    ), "diagnostic must be a graph compiler diagnostic"
    code = FlowGraphDiagnosticCode(diagnostic.code)
    if code == FlowGraphDiagnosticCode.PATH_ESCAPE:
        return "privacy"
    if code in _FLOW_GRAPH_PARSE_CODES:
        return "parse"
    if code in _FLOW_GRAPH_UNSUPPORTED_CODES:
        return "unsupported"
    return "value"


def compile_flow_graph(
    source: "FlowGraphSource",
    nodes: tuple[FlowNodeDefinition, ...],
    *,
    edge_bindings: Mapping[str, "FlowGraphEdgeBinding"] | None = None,
) -> "FlowGraphCompileResult":
    assert isinstance(source, FlowGraphSource)
    assert isinstance(nodes, tuple), "nodes must be a tuple"
    for node in nodes:
        assert isinstance(node, FlowNodeDefinition)
    bindings = (
        _freeze_edge_bindings(edge_bindings)
        if edge_bindings is not None
        else _empty_edge_bindings()
    )
    diagram, source_identity, diagnostic = _load_graph_source(source)
    if diagnostic is not None:
        return FlowGraphCompileResult(
            source=source,
            edge_bindings=bindings,
            diagnostics=(diagnostic,),
        )

    assert diagram is not None
    parsed = parse_mermaid_import(
        diagram,
        import_mode=FlowViewImportMode.EXECUTABLE,
        source=source_identity,
    )
    if not parsed.ok:
        diagnostics = _graph_import_diagnostics(parsed.diagnostics)
        if _can_inspect_import_failure(diagnostics):
            _, inspection, diagnostics = _inspect_graph_parse(
                source,
                nodes,
                bindings,
                parsed.parse_result.ast.statements,
                import_diagnostics=diagnostics,
            )
            return FlowGraphCompileResult(
                source=source,
                edge_bindings=bindings,
                inspection=inspection,
                diagnostics=diagnostics,
            )
        return FlowGraphCompileResult(
            source=source,
            edge_bindings=bindings,
            diagnostics=diagnostics,
        )

    edges, inspection, diagnostics = _inspect_graph_parse(
        source,
        nodes,
        bindings,
        parsed.parse_result.ast.statements,
    )
    if diagnostics:
        return FlowGraphCompileResult(
            source=source,
            edge_bindings=bindings,
            inspection=inspection,
            diagnostics=diagnostics,
        )
    return FlowGraphCompileResult(
        source=source,
        edges=edges,
        edge_bindings=bindings,
        inspection=inspection,
    )


def _load_graph_source(
    source: "FlowGraphSource",
) -> tuple[str | None, str | None, FlowDiagnostic | None]:
    match source.source_kind:
        case FlowGraphSourceKind.INLINE:
            assert source.diagram is not None
            return source.diagram, source.source_identity, None
        case FlowGraphSourceKind.FILE:
            return _load_graph_file_source(source)


def _load_graph_file_source(
    source: "FlowGraphSource",
) -> tuple[str | None, str | None, FlowDiagnostic | None]:
    assert source.path is not None
    if _is_untrusted_graph_path(source.path):
        return (
            None,
            None,
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.PATH_ESCAPE,
                "graph.source",
                source_span=source.source_span,
            ),
        )
    if source.base_path is None:
        return (
            None,
            None,
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.READ_FAILURE,
                "graph.source",
                source_span=source.source_span,
            ),
        )
    try:
        base_path = source.base_path.resolve()
        path = (base_path / source.path).resolve()
    except (OSError, RuntimeError):
        return (
            None,
            None,
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.READ_FAILURE,
                "graph.source",
                source_span=source.source_span,
            ),
        )
    if not _is_relative_to(path, base_path):
        return (
            None,
            None,
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.PATH_ESCAPE,
                "graph.source",
                source_span=source.source_span,
            ),
        )
    try:
        return (
            path.read_text(encoding="utf-8"),
            source.source_identity or str(path),
            None,
        )
    except (OSError, UnicodeDecodeError):
        return (
            None,
            None,
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.READ_FAILURE,
                "graph.source",
                source_span=source.source_span,
            ),
        )


def _is_untrusted_graph_path(path: Path) -> bool:
    value = str(path)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if urlsplit(value).scheme or "://" in value or "\\" in value:
        return True
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _is_relative_to(path: Path, base_path: Path) -> bool:
    try:
        path.relative_to(base_path)
    except ValueError:
        return False
    return True


def _first_error_span(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> FlowSourceSpan | None:
    return next(
        (
            diagnostic.source_span
            for diagnostic in diagnostics
            if diagnostic.severity == FlowDiagnosticSeverity.ERROR
        ),
        None,
    )


def _graph_import_diagnostics(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> tuple[FlowDiagnostic, ...]:
    graph_diagnostics: list[FlowDiagnostic] = []
    malformed_diagnostics: list[FlowDiagnostic] = []
    for diagnostic in diagnostics:
        code = _MERMAID_IMPORT_GRAPH_DIAGNOSTIC_CODES.get(diagnostic.code)
        if code is None:
            malformed_diagnostics.append(diagnostic)
            continue
        graph_diagnostics.append(
            flow_graph_diagnostic(
                code,
                _MERMAID_IMPORT_GRAPH_DIAGNOSTIC_PATHS[code],
                source_span=diagnostic.source_span,
                related_spans=diagnostic.related_spans,
            )
        )

    if malformed_diagnostics:
        graph_diagnostics.append(
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.MALFORMED_SOURCE,
                "graph.source",
                source_span=_first_error_span(tuple(malformed_diagnostics)),
            )
        )
    return tuple(graph_diagnostics)


def _can_inspect_import_failure(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> bool:
    return bool(diagnostics) and all(
        FlowGraphDiagnosticCode(diagnostic.code)
        in _FLOW_GRAPH_INSPECTABLE_IMPORT_FAILURE_CODES
        for diagnostic in diagnostics
    )


def _inspect_graph_parse(
    source: "FlowGraphSource",
    nodes: tuple[FlowNodeDefinition, ...],
    edge_bindings: Mapping[str, "FlowGraphEdgeBinding"],
    statements: tuple[MermaidAstStatement, ...],
    *,
    import_diagnostics: tuple[FlowDiagnostic, ...] = (),
) -> tuple[
    tuple[FlowEdgeDefinition, ...],
    "FlowGraphInspection",
    tuple[FlowDiagnostic, ...],
]:
    strict_node_names = {node.name for node in nodes}
    node_inspections = _classify_mermaid_nodes(
        _mermaid_nodes(statements),
        strict_node_names,
    )
    edge_inspections = _classify_mermaid_edges(
        _mermaid_edges(statements),
        strict_node_names,
    )
    edges, binding_inspections, diagnostics = _bind_graph_edges(
        edge_inspections,
        edge_bindings,
    )
    if import_diagnostics:
        diagnostics = import_diagnostics + _binding_diagnostics(
            edge_bindings, binding_inspections
        )
    inspection = FlowGraphInspection(
        source=source,
        diagnostics=diagnostics,
        nodes=node_inspections,
        edges=edge_inspections,
        bindings=binding_inspections,
        generated_edges=edges,
    )
    return edges, inspection, diagnostics


def _mermaid_edges(
    statements: tuple[MermaidAstStatement, ...],
) -> tuple[MermaidAstEdge, ...]:
    edges: list[MermaidAstEdge] = []
    for statement in statements:
        if isinstance(statement, MermaidAstEdgeStatement):
            edges.extend(statement.edges)
        if isinstance(statement, MermaidAstSubgraph):
            edges.extend(_mermaid_edges(statement.statements))
    return tuple(edges)


def _mermaid_nodes(
    statements: tuple[MermaidAstStatement, ...],
) -> tuple[MermaidAstNode, ...]:
    nodes: list[MermaidAstNode] = []
    for statement in statements:
        if isinstance(statement, MermaidAstNodeStatement):
            nodes.append(statement.node)
        if isinstance(statement, MermaidAstEdgeStatement):
            nodes.extend(statement.nodes)
        if isinstance(statement, MermaidAstSubgraph):
            nodes.extend(_mermaid_nodes(statement.statements))
    return tuple(nodes)


def _classify_mermaid_nodes(
    nodes: tuple[MermaidAstNode, ...],
    strict_node_names: set[str],
) -> tuple["FlowGraphNodeInspection", ...]:
    inspections: list[FlowGraphNodeInspection] = []
    seen: set[str] = set()
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        if node.id in strict_node_names:
            inspections.append(
                FlowGraphNodeInspection(
                    id=node.id,
                    classification=FlowGraphNodeClassification.ACTUAL,
                    strict_node=node.id,
                    source_span=node.source_span,
                )
            )
        else:
            inspections.append(
                FlowGraphNodeInspection(
                    id=node.id,
                    classification=FlowGraphNodeClassification.DECORATIVE,
                    source_span=node.source_span,
                )
            )
    return tuple(inspections)


def _classify_mermaid_edges(
    edges: tuple[MermaidAstEdge, ...],
    strict_node_names: set[str],
) -> tuple["FlowGraphEdgeInspection", ...]:
    inspections: list[FlowGraphEdgeInspection] = []
    for index, edge in enumerate(edges):
        if (
            edge.source in strict_node_names
            and edge.target in strict_node_names
        ):
            classification = FlowGraphEdgeClassification.EXECUTABLE
        else:
            classification = FlowGraphEdgeClassification.DECORATIVE
        inspections.append(
            FlowGraphEdgeInspection(
                index=index,
                source=edge.source,
                target=edge.target,
                classification=classification,
                edge_id=edge.explicit_id,
                source_span=edge.source_span,
                bidirectional=_is_bidirectional_mermaid_arrow(edge.arrow),
            )
        )
    return tuple(inspections)


def _is_bidirectional_mermaid_arrow(value: str) -> bool:
    return value.startswith("<") and value.endswith(">")


def _bind_graph_edges(
    edge_inspections: tuple["FlowGraphEdgeInspection", ...],
    edge_bindings: Mapping[str, "FlowGraphEdgeBinding"],
) -> tuple[
    tuple[FlowEdgeDefinition, ...],
    tuple["FlowGraphBindingInspection", ...],
    tuple[FlowDiagnostic, ...],
]:
    executable_edges_by_id: dict[str, FlowGraphEdgeInspection] = {}
    decorative_edge_ids: set[str] = set()
    rejected_edge_diagnostic_codes = _rejected_graph_edge_diagnostic_codes(
        edge_inspections
    )
    rejected_edge_ids = frozenset(rejected_edge_diagnostic_codes)
    for edge in edge_inspections:
        if edge.edge_id is None:
            continue
        if edge.edge_id in rejected_edge_ids:
            continue
        if edge.classification == FlowGraphEdgeClassification.EXECUTABLE:
            executable_edges_by_id[edge.edge_id] = edge
        else:
            decorative_edge_ids.add(edge.edge_id)

    binding_inspections = _binding_inspections(
        edge_bindings,
        executable_edges_by_id=executable_edges_by_id,
        decorative_edge_ids=decorative_edge_ids,
        rejected_edge_diagnostic_codes=rejected_edge_diagnostic_codes,
    )
    diagnostics = (
        _edge_id_diagnostics(edge_inspections)
        + _executable_edge_diagnostics(edge_inspections)
        + _binding_diagnostics(edge_bindings, binding_inspections)
    )
    generated_edges = tuple(
        _graph_edge_definition(
            edge,
            edge_bindings.get(edge.edge_id) if edge.edge_id else None,
        )
        for edge in edge_inspections
        if _is_supported_executable_graph_edge(edge, rejected_edge_ids)
    )
    return generated_edges, binding_inspections, diagnostics


def _rejected_graph_edge_diagnostic_codes(
    edge_inspections: tuple["FlowGraphEdgeInspection", ...],
) -> Mapping[str, tuple[str, ...]]:
    counts: dict[str, int] = {}
    rejected: dict[str, list[str]] = {}
    for edge in edge_inspections:
        if edge.edge_id is None:
            continue
        counts[edge.edge_id] = counts.get(edge.edge_id, 0) + 1
        if not is_executable_mermaid_edge_id(edge.edge_id):
            _add_rejected_edge_diagnostic_code(
                rejected,
                edge.edge_id,
                FlowGraphDiagnosticCode.INVALID_EDGE_ID,
            )
        if (
            edge.classification == FlowGraphEdgeClassification.EXECUTABLE
            and edge.bidirectional
        ):
            _add_rejected_edge_diagnostic_code(
                rejected,
                edge.edge_id,
                FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
            )
    for edge_id, count in counts.items():
        if count > 1:
            _add_rejected_edge_diagnostic_code(
                rejected,
                edge_id,
                FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID,
            )
    return MappingProxyType(
        {edge_id: tuple(codes) for edge_id, codes in rejected.items()}
    )


def _add_rejected_edge_diagnostic_code(
    diagnostics: dict[str, list[str]],
    edge_id: str,
    code: FlowGraphDiagnosticCode,
) -> None:
    codes = diagnostics.setdefault(edge_id, [])
    if code.value not in codes:
        codes.append(code.value)


def _edge_id_diagnostics(
    edge_inspections: tuple["FlowGraphEdgeInspection", ...],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    seen: dict[str, FlowSourceSpan | None] = {}
    invalid_seen: set[str] = set()
    for edge in edge_inspections:
        if edge.edge_id is None:
            continue
        if not is_executable_mermaid_edge_id(edge.edge_id):
            if edge.edge_id not in invalid_seen:
                diagnostics.append(
                    flow_graph_diagnostic(
                        FlowGraphDiagnosticCode.INVALID_EDGE_ID,
                        "graph.edges",
                        source_span=edge.source_span,
                    )
                )
                invalid_seen.add(edge.edge_id)
        if edge.edge_id in seen:
            first_span = seen[edge.edge_id]
            diagnostics.append(
                flow_graph_diagnostic(
                    FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID,
                    "graph.edges",
                    source_span=edge.source_span,
                    related_spans=(first_span,) if first_span else (),
                )
            )
            continue
        seen[edge.edge_id] = edge.source_span
    return tuple(diagnostics)


def _executable_edge_diagnostics(
    edge_inspections: tuple["FlowGraphEdgeInspection", ...],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for edge in edge_inspections:
        if edge.classification != FlowGraphEdgeClassification.EXECUTABLE:
            continue
        if edge.bidirectional:
            diagnostics.append(
                flow_graph_diagnostic(
                    FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
                    "graph.edges",
                    source_span=edge.source_span,
                )
            )
            continue
        if edge.edge_id is not None:
            continue
        diagnostics.append(
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
                "graph.edges",
                source_span=edge.source_span,
            )
        )
    return tuple(diagnostics)


def _is_supported_executable_graph_edge(
    edge: "FlowGraphEdgeInspection",
    rejected_edge_ids: frozenset[str],
) -> bool:
    if edge.classification != FlowGraphEdgeClassification.EXECUTABLE:
        return False
    if edge.edge_id is None or edge.edge_id in rejected_edge_ids:
        return False
    return not edge.bidirectional


def _binding_inspections(
    edge_bindings: Mapping[str, "FlowGraphEdgeBinding"],
    *,
    executable_edges_by_id: Mapping[str, "FlowGraphEdgeInspection"],
    decorative_edge_ids: set[str],
    rejected_edge_diagnostic_codes: Mapping[str, tuple[str, ...]],
) -> tuple["FlowGraphBindingInspection", ...]:
    inspections: list[FlowGraphBindingInspection] = []
    for edge_id, binding in edge_bindings.items():
        diagnostic_codes: tuple[str, ...] = ()
        if not is_executable_mermaid_edge_id(edge_id):
            state = FlowGraphBindingState.REJECTED
            diagnostic_codes = (FlowGraphDiagnosticCode.INVALID_EDGE_ID.value,)
        elif edge_id in executable_edges_by_id:
            state = FlowGraphBindingState.BOUND
        elif edge_id in rejected_edge_diagnostic_codes:
            state = FlowGraphBindingState.REJECTED
            diagnostic_codes = rejected_edge_diagnostic_codes[edge_id]
        elif edge_id in decorative_edge_ids:
            state = FlowGraphBindingState.DECORATIVE
            diagnostic_codes = (
                FlowGraphDiagnosticCode.DECORATIVE_EDGE_METADATA_TARGET.value,
            )
        else:
            state = FlowGraphBindingState.MISSING
            diagnostic_codes = (
                FlowGraphDiagnosticCode.MISSING_EDGE_METADATA_TARGET.value,
            )
        inspections.append(
            FlowGraphBindingInspection(
                edge_id=edge_id,
                state=state,
                metadata_fields=_binding_metadata_fields(binding),
                diagnostic_codes=diagnostic_codes,
                source_span=binding.source_span,
            )
        )

    bound_edge_ids = set(edge_bindings)
    for edge in executable_edges_by_id.values():
        assert edge.edge_id is not None
        if edge.edge_id in bound_edge_ids:
            continue
        inspections.append(
            FlowGraphBindingInspection(
                edge_id=edge.edge_id,
                state=FlowGraphBindingState.UNBOUND,
            )
        )
    return tuple(inspections)


def _binding_metadata_fields(
    binding: "FlowGraphEdgeBinding",
) -> tuple[str, ...]:
    fields = set(binding.metadata)
    if binding.label is not None:
        fields.add("label")
    if binding.kind is not None:
        fields.add("kind")
    if binding.condition is not None:
        fields.add("condition")
    if binding.priority is not None:
        fields.add("priority")
    if binding.default is not None:
        fields.add("default")
    if binding.routing_policy is not None:
        fields.add("routing_policy")
    return tuple(fields)


def _binding_diagnostics(
    edge_bindings: Mapping[str, "FlowGraphEdgeBinding"],
    binding_inspections: tuple["FlowGraphBindingInspection", ...],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for inspection in binding_inspections:
        binding = edge_bindings.get(inspection.edge_id)
        if binding is None:
            continue
        match inspection.state:
            case FlowGraphBindingState.MISSING:
                diagnostics.append(
                    flow_graph_diagnostic(
                        FlowGraphDiagnosticCode.MISSING_EDGE_METADATA_TARGET,
                        "graph.edges",
                        source_span=binding.source_span,
                    )
                )
            case FlowGraphBindingState.DECORATIVE:
                diagnostics.append(
                    flow_graph_diagnostic(
                        (
                            FlowGraphDiagnosticCode.DECORATIVE_EDGE_METADATA_TARGET
                        ),
                        "graph.edges",
                        source_span=binding.source_span,
                    )
                )
            case FlowGraphBindingState.REJECTED:
                if is_executable_mermaid_edge_id(inspection.edge_id):
                    continue
                diagnostics.append(
                    flow_graph_diagnostic(
                        FlowGraphDiagnosticCode.INVALID_EDGE_ID,
                        "graph.edges",
                        source_span=binding.source_span,
                    )
                )
            case _:
                continue
    return tuple(diagnostics)


def _graph_edge_definition(
    edge: "FlowGraphEdgeInspection",
    binding: "FlowGraphEdgeBinding | None",
) -> FlowEdgeDefinition:
    if binding is None:
        return FlowEdgeDefinition(source=edge.source, target=edge.target)
    return FlowEdgeDefinition(
        source=edge.source,
        target=edge.target,
        label=binding.label,
        kind=binding.kind or FlowEdgeKind.SUCCESS,
        condition=binding.condition,
        priority=0 if binding.priority is None else binding.priority,
        default=False if binding.default is None else binding.default,
        routing_policy=(
            binding.routing_policy or FlowRouteMatchPolicy.EXCLUSIVE
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphSource:
    format: FlowGraphFormat = FlowGraphFormat.MERMAID
    source_kind: FlowGraphSourceKind
    mode: FlowGraphMode = FlowGraphMode.EXECUTABLE
    diagram: str | None = None
    path: Path | None = None
    base_path: Path | None = None
    source_identity: str | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.format, FlowGraphFormat)
        assert isinstance(self.source_kind, FlowGraphSourceKind)
        assert isinstance(self.mode, FlowGraphMode)
        if self.diagram is not None:
            _assert_non_empty_string(self.diagram, "diagram")
        if self.path is not None:
            assert isinstance(self.path, Path)
        if self.base_path is not None:
            assert isinstance(self.base_path, Path)
        if self.source_identity is not None:
            _assert_non_empty_string(self.source_identity, "source_identity")
        if self.source_span is not None:
            _assert_source_span(self.source_span, "source_span")
        match self.source_kind:
            case FlowGraphSourceKind.INLINE:
                assert (
                    self.diagram is not None
                ), "inline graph source needs diagram"
                assert (
                    self.path is None
                ), "inline graph source cannot include path"
            case FlowGraphSourceKind.FILE:
                assert self.path is not None, "file graph source needs path"
                assert (
                    self.diagram is None
                ), "file graph source cannot include diagram"

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "format": self.format.value,
            "source_kind": self.source_kind.value,
            "mode": self.mode.value,
        }
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphEdgeBinding:
    edge_id: str
    metadata: FlowMetadata = field(default_factory=_empty_mapping)
    label: str | None = None
    kind: FlowEdgeKind | None = None
    condition: FlowCondition | None = None
    priority: int | None = None
    default: bool | None = None
    routing_policy: FlowRouteMatchPolicy | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.edge_id, "edge_id")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )
        for key in self.metadata:
            assert (
                key in _FLOW_GRAPH_EDGE_BINDING_FIELDS
            ), f"metadata.{key} is not allowed"
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        if self.kind is not None:
            assert isinstance(self.kind, FlowEdgeKind)
        if self.condition is not None:
            assert isinstance(self.condition, FlowCondition)
        if self.priority is not None:
            assert isinstance(self.priority, int) and not isinstance(
                self.priority,
                bool,
            )
        if self.default is not None:
            assert isinstance(self.default, bool)
        if self.routing_policy is not None:
            assert isinstance(self.routing_policy, FlowRouteMatchPolicy)
        _assert_source_span(self.source_span, "source_span")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"edge_id": self.edge_id}
        if self.metadata:
            value["metadata_fields"] = tuple(sorted(self.metadata))
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphNodeInspection:
    id: str
    classification: FlowGraphNodeClassification
    strict_node: str | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        assert isinstance(self.classification, FlowGraphNodeClassification)
        if self.strict_node is not None:
            _assert_non_empty_string(self.strict_node, "strict_node")
        if (
            self.classification == FlowGraphNodeClassification.DECORATIVE
            and self.strict_node is not None
        ):
            raise AssertionError(
                "decorative node cannot reference strict node"
            )
        _assert_source_span(self.source_span, "source_span")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "classification": self.classification.value,
        }
        if self.strict_node is not None:
            value["strict_node"] = self.strict_node
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphEdgeInspection:
    index: int
    source: str
    target: str
    classification: FlowGraphEdgeClassification
    edge_id: str | None = None
    source_span: FlowSourceSpan | None = None
    bidirectional: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.index, int) and not isinstance(
            self.index,
            bool,
        )
        assert self.index >= 0, "index must be non-negative"
        _assert_non_empty_string(self.source, "source")
        _assert_non_empty_string(self.target, "target")
        assert isinstance(self.classification, FlowGraphEdgeClassification)
        if self.edge_id is not None:
            _assert_non_empty_string(self.edge_id, "edge_id")
        _assert_source_span(self.source_span, "source_span")
        assert isinstance(self.bidirectional, bool)

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "source": self.source,
            "target": self.target,
            "classification": self.classification.value,
        }
        if self.edge_id is not None:
            value["edge_id"] = self.edge_id
        if self.bidirectional:
            value["bidirectional"] = True
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphBindingInspection:
    edge_id: str
    state: FlowGraphBindingState
    metadata_fields: tuple[str, ...] = field(
        default_factory=_empty_string_tuple
    )
    diagnostic_codes: tuple[str, ...] = field(
        default_factory=_empty_string_tuple
    )
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.edge_id, "edge_id")
        assert isinstance(self.state, FlowGraphBindingState)
        _assert_string_tuple(self.metadata_fields, "metadata_fields")
        fields = tuple(sorted(self.metadata_fields))
        object.__setattr__(self, "metadata_fields", fields)
        _assert_string_tuple(self.diagnostic_codes, "diagnostic_codes")
        _assert_source_span(self.source_span, "source_span")

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "edge_id": self.edge_id,
            "state": self.state.value,
        }
        if self.metadata_fields:
            value["metadata_fields"] = self.metadata_fields
        if self.diagnostic_codes:
            value["diagnostic_codes"] = self.diagnostic_codes
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphInspection:
    schema_version: str = "flow.graph.inspection.v1"
    source: FlowGraphSource | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    nodes: tuple[FlowGraphNodeInspection, ...] = ()
    edges: tuple[FlowGraphEdgeInspection, ...] = ()
    bindings: tuple[FlowGraphBindingInspection, ...] = ()
    generated_edges: tuple[FlowEdgeDefinition, ...] = ()

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.schema_version, "schema_version")
        if self.source is not None:
            assert isinstance(self.source, FlowGraphSource)
        _assert_diagnostics(self.diagnostics)
        _assert_tuple_items(self.nodes, "nodes", FlowGraphNodeInspection)
        _assert_tuple_items(self.edges, "edges", FlowGraphEdgeInspection)
        _assert_tuple_items(
            self.bindings,
            "bindings",
            FlowGraphBindingInspection,
        )
        _assert_tuple_items(
            self.generated_edges,
            "generated_edges",
            FlowEdgeDefinition,
        )
        self._assert_unique_ids(tuple(node.id for node in self.nodes), "nodes")
        self._assert_unique_indexes(
            tuple(edge.index for edge in self.edges),
            "edges",
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "nodes": tuple(node.as_public_dict() for node in self.nodes),
            "edges": tuple(edge.as_public_dict() for edge in self.edges),
            "bindings": tuple(
                binding.as_public_dict() for binding in self.bindings
            ),
            "generated_edges": tuple(
                _flow_edge_definition_as_public_dict(index, edge)
                for index, edge in enumerate(self.generated_edges)
            ),
        }
        if self.source is not None:
            value["source"] = self.source.as_public_dict()
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
        return value

    @staticmethod
    def _assert_unique_ids(values: tuple[str, ...], field_name: str) -> None:
        seen: set[str] = set()
        for value in values:
            assert value not in seen, f"{field_name} must have unique ids"
            seen.add(value)

    @staticmethod
    def _assert_unique_indexes(
        values: tuple[int, ...],
        field_name: str,
    ) -> None:
        seen: set[int] = set()
        for value in values:
            assert value not in seen, f"{field_name} must have unique indexes"
            seen.add(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphCompileResult:
    source: FlowGraphSource | None = None
    edges: tuple[FlowEdgeDefinition, ...] = ()
    edge_bindings: Mapping[str, FlowGraphEdgeBinding] = field(
        default_factory=_empty_edge_bindings
    )
    inspection: FlowGraphInspection | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.source is not None:
            assert isinstance(self.source, FlowGraphSource)
        assert isinstance(self.edges, tuple), "edges must be a tuple"
        for edge in self.edges:
            assert isinstance(edge, FlowEdgeDefinition)
        assert isinstance(
            self.edge_bindings,
            Mapping,
        ), "edge_bindings must be a mapping"
        object.__setattr__(
            self,
            "edge_bindings",
            _freeze_edge_bindings(self.edge_bindings),
        )
        if self.inspection is not None:
            assert isinstance(self.inspection, FlowGraphInspection)
        _assert_diagnostics(self.diagnostics)

    @property
    def ok(self) -> bool:
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "edge_count": len(self.edges),
            "edge_binding_count": len(self.edge_bindings),
        }
        if self.source is not None:
            value["source"] = self.source.as_public_dict()
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
        return value
