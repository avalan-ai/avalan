from .definition import FlowDefinition
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .registry import FlowNodeRegistry, default_flow_node_registry

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

_KNOWN_DEFERRED_NODE_TYPES = frozenset(
    {
        "agent",
        "file_convert",
        "pdf_to_images",
    }
)
_UNTRUSTED_NODE_TYPES = frozenset(
    {
        "callable",
        "function",
        "module",
        "python",
        "python_callable",
    }
)
_RESERVED_FILE_SELECTOR_SOURCES = frozenset(
    {
        "__task_files__",
        "__task_input__",
        "file",
        "files",
    }
)
_UNSUPPORTED_NODE_CODES = frozenset(
    {
        "flow.unknown_node_type",
        "flow.unsupported_node_type",
        "flow.untrusted_callable",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowValidationResult:
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


def validate_flow_definition(
    definition: FlowDefinition,
    registry: FlowNodeRegistry | None = None,
) -> FlowValidationResult:
    assert isinstance(definition, FlowDefinition)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    node_registry = registry or default_flow_node_registry()
    diagnostics: list[FlowDiagnostic] = []
    diagnostics.extend(_validate_node_types(definition, node_registry))
    diagnostics.extend(_validate_graph_contract(definition))
    return FlowValidationResult(diagnostics=tuple(diagnostics))


def flow_validation_diagnostic_load_category(
    diagnostic: FlowDiagnostic,
) -> str:
    assert isinstance(diagnostic, FlowDiagnostic)
    if diagnostic.category == FlowDiagnosticCategory.PRIVACY:
        return "privacy"
    if diagnostic.code in _UNSUPPORTED_NODE_CODES:
        return "unsupported"
    return "value"


def _validate_node_types(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    seen: set[str] = set()
    for node in definition.nodes:
        if node.name in seen:
            diagnostics.append(
                _diagnostic(
                    code="flow.duplicate_node",
                    path=f"nodes.{node.name}",
                    message="Flow node name is declared more than once.",
                    hint="Use unique node names.",
                )
            )
        seen.add(node.name)
        diagnostics.extend(
            _validate_node_type(node.name, node.type, node.ref, registry)
        )
    return tuple(diagnostics)


def _validate_node_type(
    name: str,
    node_type: str,
    ref: str | None,
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    path = f"nodes.{name}.type"
    if node_type in _UNTRUSTED_NODE_TYPES:
        return (
            _diagnostic(
                code="flow.untrusted_callable",
                path=path,
                message="Flow TOML cannot import dynamic callables.",
                hint="Use a registered built-in node type.",
            ),
        )
    diagnostics: list[FlowDiagnostic] = []
    if ref is not None and _is_path_escape(ref):
        diagnostics.append(_path_escape_diagnostic(f"nodes.{name}.ref"))
    if registry.supports(node_type):
        if ref is not None and not registry.supports_ref(node_type):
            diagnostics.append(
                _diagnostic(
                    code="flow.untrusted_callable",
                    path=f"nodes.{name}.ref",
                    message="Built-in flow nodes cannot load external refs.",
                    hint="Remove ref or use a later trusted node type.",
                )
            )
        return tuple(diagnostics)
    if node_type in _KNOWN_DEFERRED_NODE_TYPES:
        diagnostics.append(
            _diagnostic(
                code="flow.unsupported_node_type",
                path=path,
                message="Flow node type is not supported by this runtime.",
                hint="Use a currently registered built-in node type.",
            )
        )
        return tuple(diagnostics)
    diagnostics.append(
        _diagnostic(
            code="flow.unknown_node_type",
            path=path,
            message="Flow node type is unknown.",
            hint="Use a registered built-in node type.",
        )
    )
    return tuple(diagnostics)


def _validate_graph_contract(
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    node_names = {node.name for node in definition.nodes}
    if definition.entrypoint not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint does not reference a node.",
                hint="Set flow.entrypoint to a declared node name.",
            )
        )
    if definition.output_node not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_output_node",
                path="flow.output_node",
                message="Flow output node does not reference a node.",
                hint="Set flow.output_node to a declared node name.",
            )
        )
    for edge in definition.edges:
        if edge.source not in node_names:
            diagnostics.append(_bad_reference_diagnostic("edges.source"))
        if edge.target not in node_names:
            diagnostics.append(_bad_reference_diagnostic("edges.target"))
    if diagnostics:
        return tuple(diagnostics)
    outgoing = {node.name: 0 for node in definition.nodes}
    incoming = {node.name: 0 for node in definition.nodes}
    for edge in definition.edges:
        outgoing[edge.source] += 1
        incoming[edge.target] += 1
    terminals = {name for name, count in outgoing.items() if count == 0}
    start_nodes = {name for name, count in incoming.items() if count == 0}
    if definition.entrypoint not in start_nodes:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint must be a start node.",
                hint="Use a node without inbound edges as the entrypoint.",
            )
        )
    if len(terminals) > 1:
        diagnostics.append(
            _diagnostic(
                code="flow.multiple_outputs",
                path="flow.output_node",
                message="Flow has multiple terminal output nodes.",
                hint="Connect nodes so only one terminal output remains.",
            )
        )
    elif definition.output_node not in terminals:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_output_node",
                path="flow.output_node",
                message="Flow output node must be terminal.",
                hint="Use a node without outbound edges as the output node.",
            )
        )
    diagnostics.extend(_validate_agent_file_selectors(definition))
    if _cycle_nodes(definition):
        diagnostics.append(
            _diagnostic(
                code="flow.cycle",
                path="edges",
                message="Flow graph contains a cycle.",
                hint="Remove the cyclic edge before running the flow.",
            )
        )
    return tuple(diagnostics)


def _validate_agent_file_selectors(
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    node_names = {node.name for node in definition.nodes}
    incoming_sources: dict[str, set[str]] = {
        name: set() for name in node_names
    }
    for edge in definition.edges:
        incoming_sources[edge.target].add(edge.source)
    for node in definition.nodes:
        selector = node.config.get("files_input")
        if selector is None or node.type != "agent":
            continue
        path = f"nodes.{node.name}.config.files_input"
        if not isinstance(selector, str) or not selector.strip():
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_type",
                    path=path,
                    message="Flow field has an invalid type.",
                    hint="Use a dotted node output.",
                )
            )
            continue
        parts = selector.split(".")
        if len(parts) != 2 or any(not part.strip() for part in parts):
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_node",
                    path=path,
                    message="Flow agent file input selector is invalid.",
                    hint="Use a dotted upstream node output selector.",
                )
            )
            continue
        source, _ = parts
        if source in _RESERVED_FILE_SELECTOR_SOURCES:
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_node",
                    path=path,
                    message="Flow agent file input selector is reserved.",
                    hint="Reference a named upstream node output instead.",
                )
            )
            continue
        if source not in node_names:
            diagnostics.append(_bad_reference_diagnostic(path))
            continue
        if source not in incoming_sources[node.name]:
            diagnostics.append(
                _diagnostic(
                    code="flow.bad_reference",
                    path=path,
                    message="Flow agent file input selector is disconnected.",
                    hint="Add an edge from the selected node to this agent.",
                )
            )
    return tuple(diagnostics)


def _cycle_nodes(definition: FlowDefinition) -> frozenset[str]:
    outgoing: dict[str, list[str]] = {
        node.name: [] for node in definition.nodes
    }
    for edge in definition.edges:
        outgoing[edge.source].append(edge.target)
    visited: set[str] = set()
    stack: set[str] = set()
    cycle: set[str] = set()

    def visit(name: str) -> None:
        if name in stack:
            cycle.add(name)
            return
        if name in visited:
            return
        visited.add(name)
        stack.add(name)
        for target in outgoing[name]:
            visit(target)
            if target in cycle:
                cycle.add(name)
        stack.remove(name)

    for node in definition.nodes:
        visit(node.name)
    return frozenset(cycle)


def _is_path_escape(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _path_escape_diagnostic(path: str) -> FlowDiagnostic:
    return _diagnostic(
        code="flow.path_escape",
        path=path,
        message="Flow reference escapes the flow directory.",
        hint="Use a safe relative reference inside the flow directory.",
        category=FlowDiagnosticCategory.PRIVACY,
    )


def _bad_reference_diagnostic(path: str) -> FlowDiagnostic:
    return _diagnostic(
        code="flow.bad_reference",
        path=path,
        message="Flow reference does not match a declared node.",
        hint="Reference an existing node name.",
    )


def _diagnostic(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: FlowDiagnosticCategory = (
        FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION
    ),
) -> FlowDiagnostic:
    return FlowDiagnostic(
        code=code,
        path=path,
        category=category,
        severity=FlowDiagnosticSeverity.ERROR,
        message=message,
        hint=hint,
    )
