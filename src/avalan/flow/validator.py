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
    for edge in definition.edges:
        if edge.source not in node_names:
            diagnostics.append(_bad_reference_diagnostic("edges.source"))
        if edge.target not in node_names:
            diagnostics.append(_bad_reference_diagnostic("edges.target"))
    if diagnostics:
        return tuple(diagnostics)
    if definition.is_strict:
        diagnostics.extend(_validate_strict_contract(definition, node_names))
    else:
        diagnostics.extend(_validate_legacy_graph_contract(definition))
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


def _validate_legacy_graph_contract(
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    node_names = {node.name for node in definition.nodes}
    if definition.entrypoint is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint is required.",
                hint="Set flow.entrypoint to a declared node name.",
            )
        )
    elif definition.entrypoint not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint does not reference a node.",
                hint="Set flow.entrypoint to a declared node name.",
            )
        )
    if definition.output_node is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_output_node",
                path="flow.output_node",
                message="Flow output node is required.",
                hint="Set flow.output_node to a declared node name.",
            )
        )
    elif definition.output_node not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_output_node",
                path="flow.output_node",
                message="Flow output node does not reference a node.",
                hint="Set flow.output_node to a declared node name.",
            )
        )
    if diagnostics:
        return tuple(diagnostics)
    assert definition.entrypoint is not None
    assert definition.output_node is not None
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
    return tuple(diagnostics)


def _validate_strict_contract(
    definition: FlowDefinition,
    node_names: set[str],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    if definition.version is None and definition.revision is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_identity",
                path="flow.version",
                message="Flow version or revision is required.",
                hint="Set flow.version or flow.revision.",
            )
        )
    if definition.input is not None:
        diagnostics.append(
            _diagnostic(
                code="flow.scalar_input_alias",
                path="flow.input",
                message="Flow input alias is not supported.",
                hint="Use declared flow inputs.",
            )
        )
    if definition.output is not None:
        diagnostics.append(
            _diagnostic(
                code="flow.scalar_output_alias",
                path="flow.output",
                message="Flow output alias is not supported.",
                hint="Use declared flow outputs and output behavior.",
            )
        )
    diagnostics.extend(
        _validate_named_contracts(
            names=[
                input_definition.name for input_definition in definition.inputs
            ],
            path="flow.inputs",
            missing_code="flow.missing_inputs",
            duplicate_code="flow.duplicate_input",
            missing_message="Flow requires at least one declared input.",
            duplicate_message="Flow input name is declared more than once.",
            missing_hint="Declare at least one flow input.",
            duplicate_hint="Use unique flow input names.",
        )
    )
    diagnostics.extend(
        _validate_named_contracts(
            names=[
                output_definition.name
                for output_definition in definition.outputs
            ],
            path="flow.outputs",
            missing_code="flow.missing_outputs",
            duplicate_code="flow.duplicate_output",
            missing_message="Flow requires at least one declared output.",
            duplicate_message="Flow output name is declared more than once.",
            missing_hint="Declare at least one flow output.",
            duplicate_hint="Use unique flow output names.",
        )
    )
    if definition.entry_behavior is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_entry_behavior",
                path="flow.entry",
                message="Flow entry behavior is required.",
                hint="Declare the node where execution starts.",
            )
        )
    elif definition.entry_behavior.node not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_entry_node",
                path="flow.entry.node",
                message="Flow entry behavior references an unknown node.",
                hint="Reference a declared node.",
            )
        )
    diagnostics.extend(_validate_output_behavior(definition, node_names))
    return tuple(diagnostics)


def _validate_named_contracts(
    *,
    names: list[str],
    path: str,
    missing_code: str,
    duplicate_code: str,
    missing_message: str,
    duplicate_message: str,
    missing_hint: str,
    duplicate_hint: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    if not names:
        diagnostics.append(
            _diagnostic(
                code=missing_code,
                path=path,
                message=missing_message,
                hint=missing_hint,
            )
        )
        return tuple(diagnostics)
    seen: set[str] = set()
    for name in names:
        if name in seen:
            diagnostics.append(
                _diagnostic(
                    code=duplicate_code,
                    path=f"{path}.{name}",
                    message=duplicate_message,
                    hint=duplicate_hint,
                )
            )
        seen.add(name)
    return tuple(diagnostics)


def _validate_output_behavior(
    definition: FlowDefinition,
    node_names: set[str],
) -> tuple[FlowDiagnostic, ...]:
    if definition.output_behavior is None:
        return (
            _diagnostic(
                code="flow.missing_output_behavior",
                path="flow.output_behavior",
                message="Flow output behavior is required.",
                hint="Map each declared flow output from node outputs.",
            ),
        )
    diagnostics: list[FlowDiagnostic] = []
    declared = {output.name for output in definition.outputs}
    selected = set(definition.output_behavior.outputs)
    for name in sorted(selected - declared):
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_output",
                path=f"flow.output_behavior.outputs.{name}",
                message="Flow output behavior maps an unknown output.",
                hint="Map only declared flow outputs.",
            )
        )
    for name in sorted(declared - selected):
        diagnostics.append(
            _diagnostic(
                code="flow.missing_output_selection",
                path=f"flow.output_behavior.outputs.{name}",
                message="Flow output behavior is missing a declared output.",
                hint="Map every declared flow output.",
            )
        )
    for name, selector in definition.output_behavior.outputs.items():
        parts = selector.split(".", 1)
        if len(parts) != 2 or any(not part.strip() for part in parts):
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_output_selector",
                    path=f"flow.output_behavior.outputs.{name}",
                    message="Flow output behavior selector is invalid.",
                    hint="Use a node output selector.",
                )
            )
            continue
        if parts[0] not in node_names:
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_output_selector_node",
                    path=f"flow.output_behavior.outputs.{name}",
                    message="Flow output behavior references an unknown node.",
                    hint="Reference a declared node output.",
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
