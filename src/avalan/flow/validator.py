from .condition import (
    FlowCondition,
    FlowConditionOperator,
)
from .definition import (
    FlowDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowOutputType,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .registry import FlowNodeRegistry, default_flow_node_registry
from .selector import (
    FlowSelector,
    FlowSelectorError,
    FlowSelectorRoot,
    FlowSelectorStep,
    FlowSelectorStepKind,
    parse_flow_selector,
)

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import cast

_KNOWN_DEFERRED_NODE_TYPES = frozenset(
    {
        "agent",
        "decision",
        "file_convert",
        "human_review",
        "join",
        "notification",
        "pdf_to_images",
        "subflow",
        "validation",
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
_UNSUPPORTED_NODE_CODES = frozenset(
    {
        "flow.unknown_node_type",
        "flow.unsupported_node_type",
        "flow.untrusted_callable",
    }
)
_BOOLEAN_CONDITION_OPERATORS = frozenset(
    {
        FlowConditionOperator.ALL,
        FlowConditionOperator.ANY,
        FlowConditionOperator.NOT,
    }
)
_SELECTOR_CONDITION_OPERATORS = frozenset(
    {
        FlowConditionOperator.EQ,
        FlowConditionOperator.NE,
        FlowConditionOperator.EXISTS,
        FlowConditionOperator.NOT_EXISTS,
        FlowConditionOperator.IS_TYPE,
        FlowConditionOperator.IN,
        FlowConditionOperator.NOT_IN,
        FlowConditionOperator.GT,
        FlowConditionOperator.GTE,
        FlowConditionOperator.LT,
        FlowConditionOperator.LTE,
        FlowConditionOperator.STARTS_WITH,
        FlowConditionOperator.ENDS_WITH,
        FlowConditionOperator.CONTAINS,
        FlowConditionOperator.IS_NULL,
        FlowConditionOperator.NOT_NULL,
    }
)
_COMPARISON_CONDITION_OPERATORS = frozenset(
    {
        FlowConditionOperator.EQ,
        FlowConditionOperator.NE,
        FlowConditionOperator.GT,
        FlowConditionOperator.GTE,
        FlowConditionOperator.LT,
        FlowConditionOperator.LTE,
        FlowConditionOperator.STARTS_WITH,
        FlowConditionOperator.ENDS_WITH,
        FlowConditionOperator.CONTAINS,
    }
)
_NUMERIC_CONDITION_OPERATORS = frozenset(
    {
        FlowConditionOperator.GT,
        FlowConditionOperator.GTE,
        FlowConditionOperator.LT,
        FlowConditionOperator.LTE,
    }
)
_STRING_CONDITION_OPERATORS = frozenset(
    {
        FlowConditionOperator.STARTS_WITH,
        FlowConditionOperator.ENDS_WITH,
        FlowConditionOperator.CONTAINS,
    }
)
_UNSAFE_CONDITION_VALUE_FRAGMENTS = frozenset(
    {
        "$(",
        "${",
        "/",
        "\\",
        "{%",
        "{{",
        "%}",
        "}}",
        "~",
    }
)
_UNSAFE_CONDITION_VALUE_PREFIXES = frozenset(
    {
        "__",
        "env.",
        "environment.",
        "eval(",
        "file.",
        "files.",
        "fs.",
        "import ",
        "network.",
        "runtime.",
        "secret.",
        "secrets.",
        "shell.",
        "task.",
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
    diagnostics.extend(_validate_graph_contract(definition, node_registry))
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
        diagnostics.extend(_validate_node_type(node, registry, definition))
    return tuple(diagnostics)


def _validate_node_type(
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    name = node.name
    node_type = node.type
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
    if node.ref is not None and _is_path_escape(node.ref):
        diagnostics.append(_path_escape_diagnostic(f"nodes.{name}.ref"))
    if registry.supports(node_type):
        if node.ref is not None and not registry.supports_ref(node_type):
            diagnostics.append(
                _diagnostic(
                    code="flow.untrusted_callable",
                    path=f"nodes.{name}.ref",
                    message="Built-in flow nodes cannot load external refs.",
                    hint="Remove ref or use a later trusted node type.",
                )
            )
        diagnostics.extend(
            _validate_registered_node_metadata(
                node,
                registry,
                strict=definition.is_strict,
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


def _validate_registered_node_metadata(
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    *,
    strict: bool,
) -> tuple[FlowDiagnostic, ...]:
    metadata = registry.metadata(node.type)
    assert metadata is not None
    diagnostics: list[FlowDiagnostic] = []
    if strict and metadata.kind is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_node_kind",
                path=f"nodes.{node.name}.type",
                message="Flow node type is missing semantic kind metadata.",
                hint="Register the node type with a semantic node kind.",
            )
        )
    elif metadata.kind is not None:
        diagnostics.extend(_validate_node_kind(node, metadata.kind))
    if metadata.requires_ref and node.ref is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_ref",
                path=f"nodes.{node.name}.ref",
                message="Flow node reference is required.",
                hint="Set ref for the registered node type.",
            )
        )
    for key in metadata.required_config_keys:
        if key not in node.config:
            diagnostics.append(
                _diagnostic(
                    code="flow.missing_node_config",
                    path=f"nodes.{node.name}.config.{key}",
                    message="Flow node configuration is missing.",
                    hint="Add the required node configuration field.",
                )
            )
    return tuple(diagnostics)


def _validate_node_kind(
    node: FlowNodeDefinition,
    kind: FlowNodeKind,
) -> tuple[FlowDiagnostic, ...]:
    if kind != FlowNodeKind.INPUT and node.type == FlowNodeKind.INPUT.value:
        return (
            _diagnostic(
                code="flow.invalid_node_kind",
                path=f"nodes.{node.name}.type",
                message="Flow node type has conflicting semantic kind.",
                hint="Register the node type with matching kind metadata.",
            ),
        )
    return ()


def _validate_graph_contract(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
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
        diagnostics.extend(
            _validate_strict_contract(definition, node_names, registry)
        )
    else:
        diagnostics.extend(_validate_legacy_graph_contract(definition))
    diagnostics.extend(
        _validate_edge_conditions(definition, node_names, registry)
    )
    diagnostics.extend(_validate_agent_file_selectors(definition, registry))
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
    registry: FlowNodeRegistry,
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
    diagnostics.extend(
        _validate_output_behavior(definition, node_names, registry)
    )
    diagnostics.extend(
        _validate_node_mappings(definition, node_names, registry)
    )
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
    registry: FlowNodeRegistry,
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
        path = f"flow.output_behavior.outputs.{name}"
        try:
            parsed = parse_flow_selector(
                selector,
                allowed_roots=frozenset({FlowSelectorRoot.NODE_OUTPUT}),
            )
        except FlowSelectorError as error:
            diagnostics.append(_selector_diagnostic(error, path=path))
            continue
        if parsed.source not in node_names:
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_output_selector_node",
                    path=path,
                    message="Flow output behavior references an unknown node.",
                    hint="Reference a declared node output.",
                )
            )
            continue
        node = definition.node_map[parsed.source]
        assert parsed.output is not None
        if not _supports_output_name(registry, node.type, parsed.output):
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_node_output",
                    path=path,
                    message="Flow output behavior references unknown output.",
                    hint="Reference a declared output for the selected node.",
                )
            )
            continue
        if not _supports_output_path(registry, node.type, parsed):
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_selector_path",
                    path=path,
                    message="Flow output behavior selector path is unknown.",
                    hint="Reference fields declared by the selected output.",
                )
            )
    return tuple(diagnostics)


def _validate_node_mappings(
    definition: FlowDefinition,
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    input_names = {
        input_definition.name for input_definition in definition.inputs
    }
    incoming_sources: dict[str, set[str]] = {
        name: set() for name in node_names
    }
    for edge in definition.edges:
        incoming_sources[edge.target].add(edge.source)
    for node in definition.nodes:
        metadata = registry.metadata(node.type)
        contracts = metadata.input_contracts if metadata is not None else ()
        target_contracts = {
            contract.name: contract
            for contract in contracts
            if contract.name is not None
        }
        required_targets = {
            contract.name
            for contract in contracts
            if contract.name is not None
            and not contract.metadata.get("optional")
            and not contract.metadata.get("dynamic")
        }
        mapped_targets: set[str] = set()
        seen_targets: set[str] = set()
        for mapping in node.mappings:
            path = f"nodes.{node.name}.mapping.{mapping.target}"
            if mapping.target in seen_targets:
                diagnostics.append(
                    _diagnostic(
                        code="flow.duplicate_mapping_target",
                        path=path,
                        message="Flow node mapping target is duplicated.",
                        hint="Map each target input once.",
                    )
                )
            seen_targets.add(mapping.target)
            if target_contracts and mapping.target not in target_contracts:
                diagnostics.append(
                    _diagnostic(
                        code="flow.unknown_mapping_target",
                        path=path,
                        message="Flow node mapping targets an unknown input.",
                        hint="Map only declared node inputs.",
                    )
                )
                continue
            mapped_targets.add(mapping.target)
            target_contract = target_contracts.get(mapping.target)
            diagnostics.extend(
                _validate_mapping(
                    definition,
                    node,
                    mapping,
                    path=path,
                    input_names=input_names,
                    incoming_sources=incoming_sources[node.name],
                    node_names=node_names,
                    registry=registry,
                    target_contract=target_contract,
                )
            )
        for target in sorted(required_targets - mapped_targets):
            diagnostics.append(
                _diagnostic(
                    code="flow.missing_input_mapping",
                    path=f"nodes.{node.name}.mapping.{target}",
                    message="Flow node mapping is missing a required input.",
                    hint="Map every required input for the target node.",
                )
            )
    return tuple(diagnostics)


def _validate_edge_conditions(
    definition: FlowDefinition,
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    input_names = {
        input_definition.name for input_definition in definition.inputs
    }
    if definition.input is not None:
        input_names.add(definition.input.name)
    diagnostics: list[FlowDiagnostic] = []
    for index, edge in enumerate(definition.edges):
        if edge.condition is None:
            continue
        diagnostics.extend(
            _validate_condition(
                definition,
                edge.condition,
                path=f"edges[{index}].condition",
                edge_source=edge.source,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    return tuple(diagnostics)


def _validate_condition(
    definition: FlowDefinition,
    condition: FlowCondition,
    *,
    path: str,
    edge_source: str,
    input_names: set[str],
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    diagnostics.extend(_validate_condition_shape(condition, path=path))
    diagnostics.extend(
        _validate_condition_literal_values(condition, path=path)
    )
    if condition.selector is not None:
        diagnostics.extend(
            _validate_condition_selector(
                definition,
                condition.selector,
                path=f"{path}.selector",
                edge_source=edge_source,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    if condition.value_selector is not None:
        diagnostics.extend(
            _validate_condition_selector(
                definition,
                condition.value_selector,
                path=f"{path}.value_selector",
                edge_source=edge_source,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    for index, child in enumerate(condition.conditions):
        diagnostics.extend(
            _validate_condition(
                definition,
                child,
                path=f"{path}.conditions[{index}]",
                edge_source=edge_source,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    if condition.condition is not None:
        diagnostics.extend(
            _validate_condition(
                definition,
                condition.condition,
                path=f"{path}.condition",
                edge_source=edge_source,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    return tuple(diagnostics)


def _validate_condition_shape(
    condition: FlowCondition,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    operator = condition.operator
    if (
        operator in _SELECTOR_CONDITION_OPERATORS
        and condition.selector is None
    ):
        diagnostics.append(
            _condition_diagnostic(
                code="flow.missing_condition_selector",
                path=f"{path}.selector",
                message="Flow condition is missing a selector.",
                hint="Set a safe selector for this condition.",
            )
        )
    if operator in _COMPARISON_CONDITION_OPERATORS:
        has_literal = condition.value is not None
        if not has_literal and condition.value_selector is None:
            diagnostics.append(
                _condition_diagnostic(
                    code="flow.missing_condition_value",
                    path=f"{path}.value",
                    message="Flow condition is missing a comparison value.",
                    hint="Set value or value_selector.",
                )
            )
    elif (
        operator
        not in {
            FlowConditionOperator.IN,
            FlowConditionOperator.NOT_IN,
        }
        and operator not in _BOOLEAN_CONDITION_OPERATORS
    ):
        if condition.value is not None:
            diagnostics.append(_unsupported_condition_field(path, "value"))
        if condition.value_selector is not None:
            diagnostics.append(
                _unsupported_condition_field(path, "value_selector")
            )
    if operator in _NUMERIC_CONDITION_OPERATORS:
        if condition.value is not None and not _is_condition_number(
            condition.value
        ):
            diagnostics.append(_invalid_condition_value(path))
    if operator in _STRING_CONDITION_OPERATORS:
        if condition.value is not None and not isinstance(
            condition.value,
            str,
        ):
            diagnostics.append(_invalid_condition_value(path))
    if operator == FlowConditionOperator.IS_TYPE:
        if condition.value_type is None:
            diagnostics.append(
                _condition_diagnostic(
                    code="flow.missing_condition_value_type",
                    path=f"{path}.value_type",
                    message="Flow condition is missing a value type.",
                    hint="Set value_type for type checks.",
                )
            )
    elif (
        condition.value_type is not None
        and operator not in _BOOLEAN_CONDITION_OPERATORS
    ):
        diagnostics.append(_unsupported_condition_field(path, "value_type"))
    if operator in {
        FlowConditionOperator.IN,
        FlowConditionOperator.NOT_IN,
    }:
        if (
            not condition.values
            and condition.value_selector is None
            and not isinstance(condition.value, list | tuple)
        ):
            diagnostics.append(
                _condition_diagnostic(
                    code="flow.missing_condition_values",
                    path=f"{path}.values",
                    message="Flow condition is missing membership values.",
                    hint="Set values or value_selector.",
                )
            )
    elif condition.values:
        diagnostics.append(_unsupported_condition_field(path, "values"))
    if operator in {FlowConditionOperator.ALL, FlowConditionOperator.ANY}:
        if not condition.conditions:
            diagnostics.append(
                _condition_diagnostic(
                    code="flow.missing_condition_children",
                    path=f"{path}.conditions",
                    message="Flow condition is missing child conditions.",
                    hint="Add one or more child conditions.",
                )
            )
    elif condition.conditions and operator not in _BOOLEAN_CONDITION_OPERATORS:
        diagnostics.append(_unsupported_condition_field(path, "conditions"))
    if operator == FlowConditionOperator.NOT:
        if condition.condition is None:
            diagnostics.append(
                _condition_diagnostic(
                    code="flow.missing_condition_child",
                    path=f"{path}.condition",
                    message="Flow condition is missing a child condition.",
                    hint="Add a child condition.",
                )
            )
    elif condition.condition is not None:
        diagnostics.append(_unsupported_condition_field(path, "condition"))
    if operator not in _BOOLEAN_CONDITION_OPERATORS:
        return tuple(diagnostics)
    for field_name, value in (
        ("selector", condition.selector),
        ("value", condition.value),
        ("value_selector", condition.value_selector),
        ("value_type", condition.value_type),
    ):
        if value is not None:
            diagnostics.append(_unsupported_condition_field(path, field_name))
    if operator == FlowConditionOperator.NOT and condition.conditions:
        diagnostics.append(_unsupported_condition_field(path, "conditions"))
    return tuple(diagnostics)


def _validate_condition_selector(
    definition: FlowDefinition,
    selector: str,
    *,
    path: str,
    edge_source: str,
    input_names: set[str],
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    try:
        parsed = parse_flow_selector(selector)
    except FlowSelectorError as error:
        return (_selector_diagnostic(error, path=path),)
    if parsed.root == FlowSelectorRoot.FLOW_INPUT:
        if parsed.source in input_names:
            return ()
        return (
            _diagnostic(
                code="flow.unknown_condition_source",
                path=path,
                message="Flow condition references an unknown input.",
                hint="Reference a declared flow input.",
            ),
        )
    if parsed.source not in node_names:
        return (
            _diagnostic(
                code="flow.unknown_condition_source",
                path=path,
                message="Flow condition references an unknown node.",
                hint="Reference the condition edge source node.",
            ),
        )
    if parsed.source != edge_source:
        return (
            _diagnostic(
                code="flow.bad_reference",
                path=path,
                message="Flow condition selector is disconnected.",
                hint="Reference the source node for this edge.",
            ),
        )
    node = definition.node_map[parsed.source]
    assert parsed.output is not None
    if not _supports_output_name(registry, node.type, parsed.output):
        return (
            _diagnostic(
                code="flow.unknown_node_output",
                path=path,
                message="Flow condition references unknown output.",
                hint="Reference a declared output for the edge source node.",
            ),
        )
    if not _supports_output_path(registry, node.type, parsed):
        return (
            _diagnostic(
                code="flow.unknown_selector_path",
                path=path,
                message="Flow condition selector path is unknown.",
                hint="Reference fields declared by the selected output.",
            ),
        )
    return ()


def _validate_condition_literal_values(
    condition: FlowCondition,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for value_path, value in _condition_literals(condition, path=path):
        if _condition_value_is_unsafe(value):
            diagnostics.append(
                _diagnostic(
                    code="flow.unsafe_condition_value",
                    path=value_path,
                    message="Flow condition literal is unsafe.",
                    hint="Use inert literal values only.",
                    category=FlowDiagnosticCategory.PRIVACY,
                )
            )
    return tuple(diagnostics)


def _condition_literals(
    condition: FlowCondition,
    *,
    path: str,
) -> tuple[tuple[str, object], ...]:
    literals: list[tuple[str, object]] = []
    if condition.value is not None:
        literals.append((f"{path}.value", condition.value))
    literals.extend(
        (f"{path}.values[{index}]", value)
        for index, value in enumerate(condition.values)
    )
    return tuple(literals)


def _condition_value_is_unsafe(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _condition_value_is_unsafe(key) or _condition_value_is_unsafe(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_condition_value_is_unsafe(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return any(
        fragment in normalized
        for fragment in _UNSAFE_CONDITION_VALUE_FRAGMENTS
    ) or any(
        normalized.startswith(prefix)
        for prefix in _UNSAFE_CONDITION_VALUE_PREFIXES
    )


def _is_condition_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _invalid_condition_value(path: str) -> FlowDiagnostic:
    return _condition_diagnostic(
        code="flow.invalid_condition_value",
        path=f"{path}.value",
        message="Flow condition value has an invalid type.",
        hint="Use a value compatible with the condition operator.",
    )


def _unsupported_condition_field(
    path: str,
    field_name: str,
) -> FlowDiagnostic:
    return _condition_diagnostic(
        code="flow.unsupported_condition_field",
        path=f"{path}.{field_name}",
        message="Flow condition field is not supported for this operator.",
        hint="Remove fields that do not apply to the condition operator.",
    )


def _condition_diagnostic(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
) -> FlowDiagnostic:
    return _diagnostic(
        code=code,
        path=path,
        message=message,
        hint=hint,
    )


def _validate_mapping(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    mapping: FlowInputMapping,
    *,
    path: str,
    input_names: set[str],
    incoming_sources: set[str],
    node_names: set[str],
    registry: FlowNodeRegistry,
    target_contract: FlowNodeContract | None,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    source_selectors = _mapping_source_selectors(mapping)
    if mapping.kind in {
        FlowMappingKind.SELECT,
        FlowMappingKind.RENAME,
        FlowMappingKind.FILE,
        FlowMappingKind.FILE_ARRAY,
    }:
        diagnostics.extend(
            _validate_single_source_mapping_shape(mapping, path=path)
        )
    elif mapping.kind == FlowMappingKind.OBJECT:
        if not mapping.fields:
            diagnostics.append(
                _diagnostic(
                    code="flow.empty_mapping",
                    path=f"{path}.fields",
                    message="Flow object mapping has no fields.",
                    hint="Add at least one mapped field.",
                )
            )
    elif mapping.kind == FlowMappingKind.ARRAY:
        if not mapping.items:
            diagnostics.append(
                _diagnostic(
                    code="flow.empty_mapping",
                    path=f"{path}.items",
                    message="Flow array mapping has no items.",
                    hint="Add at least one mapped item.",
                )
            )
    elif mapping.kind == FlowMappingKind.MERGE:
        if not mapping.sources:
            diagnostics.append(
                _diagnostic(
                    code="flow.empty_mapping",
                    path=f"{path}.sources",
                    message="Flow merge mapping has no sources.",
                    hint="Add at least one mapped source.",
                )
            )
    for selector_path, selector in source_selectors:
        source_type = _validate_mapping_selector(
            definition,
            selector,
            path=f"{path}.{selector_path}",
            input_names=input_names,
            incoming_sources=incoming_sources,
            node_names=node_names,
            registry=registry,
            diagnostics=diagnostics,
        )
        if source_type is not None and target_contract is not None:
            compatibility = _mapping_type_compatibility(
                mapping,
                source_type=source_type,
                target_contract=target_contract,
            )
            if compatibility is not None:
                diagnostics.append(
                    _diagnostic(
                        code=compatibility,
                        path=f"{path}.{selector_path}",
                        message="Flow mapping source is incompatible.",
                        hint="Use a source that matches the target input.",
                    )
                )
    if target_contract is not None:
        target_type = target_contract.type
        if mapping.kind == FlowMappingKind.OBJECT and target_type not in {
            None,
            FlowInputType.OBJECT,
            "object",
        }:
            diagnostics.append(_mapping_kind_diagnostic(path))
        if mapping.kind == FlowMappingKind.ARRAY and target_type not in {
            None,
            FlowInputType.ARRAY,
            "array",
        }:
            diagnostics.append(_mapping_kind_diagnostic(path))
        if mapping.kind == FlowMappingKind.MERGE and target_type not in {
            None,
            FlowInputType.OBJECT,
            "object",
        }:
            diagnostics.append(_mapping_kind_diagnostic(path))
    return tuple(diagnostics)


def _validate_single_source_mapping_shape(
    mapping: FlowInputMapping,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    if mapping.source is not None:
        return ()
    return (
        _diagnostic(
            code="flow.missing_mapping_source",
            path=f"{path}.source",
            message="Flow node mapping is missing a source.",
            hint="Set a safe selector for this mapping.",
        ),
    )


def _mapping_source_selectors(
    mapping: FlowInputMapping,
) -> tuple[tuple[str, str], ...]:
    selectors: list[tuple[str, str]] = []
    if mapping.source is not None:
        selectors.append(("source", mapping.source))
    selectors.extend(
        (f"sources[{index}]", source)
        for index, source in enumerate(mapping.sources)
    )
    selectors.extend(
        (f"fields.{field}", selector)
        for field, selector in mapping.fields.items()
    )
    selectors.extend(
        (f"items[{index}]", item) for index, item in enumerate(mapping.items)
    )
    return tuple(selectors)


def _validate_mapping_selector(
    definition: FlowDefinition,
    selector: str,
    *,
    path: str,
    input_names: set[str],
    incoming_sources: set[str],
    node_names: set[str],
    registry: FlowNodeRegistry,
    diagnostics: list[FlowDiagnostic],
) -> FlowInputType | FlowOutputType | str | None:
    try:
        parsed = parse_flow_selector(selector)
    except FlowSelectorError as error:
        diagnostics.append(_selector_diagnostic(error, path=path))
        return None
    if parsed.root == FlowSelectorRoot.FLOW_INPUT:
        if parsed.source not in input_names:
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_mapping_source",
                    path=path,
                    message="Flow node mapping references an unknown input.",
                    hint="Reference a declared flow input.",
                )
            )
            return None
        return _flow_input_type(definition, parsed.source)
    if parsed.source not in node_names:
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_mapping_source",
                path=path,
                message="Flow node mapping references an unknown node.",
                hint="Reference a declared upstream node output.",
            )
        )
        return None
    if parsed.source not in incoming_sources:
        diagnostics.append(
            _diagnostic(
                code="flow.bad_reference",
                path=path,
                message="Flow node mapping source is disconnected.",
                hint="Add an edge from the selected source node.",
            )
        )
        return None
    node = definition.node_map[parsed.source]
    assert parsed.output is not None
    if not _supports_output_name(registry, node.type, parsed.output):
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_node_output",
                path=path,
                message="Flow node mapping references unknown output.",
                hint="Reference a declared output for the selected node.",
            )
        )
        return None
    if not _supports_output_path(registry, node.type, parsed):
        diagnostics.append(
            _diagnostic(
                code="flow.unknown_selector_path",
                path=path,
                message="Flow node mapping selector path is unknown.",
                hint="Reference fields declared by the selected output.",
            )
        )
        return None
    return _node_output_type(registry, node.type, parsed.output)


def _flow_input_type(
    definition: FlowDefinition,
    name: str,
) -> FlowInputType | None:
    for input_definition in definition.inputs:
        if input_definition.name == name:
            return input_definition.type
    return None


def _node_output_type(
    registry: FlowNodeRegistry,
    node_type: str,
    output_name: str,
) -> FlowOutputType | str | None:
    metadata = registry.metadata(node_type)
    if metadata is None:
        return None
    for contract in metadata.output_contracts:
        if contract.metadata.get("dynamic"):
            return contract.type
        if contract.name == output_name:
            return contract.type
    return None


def _mapping_type_compatibility(
    mapping: FlowInputMapping,
    *,
    source_type: FlowInputType | FlowOutputType | str,
    target_contract: FlowNodeContract,
) -> str | None:
    target_type = target_contract.type
    if target_type is None or type(source_type) is str:
        return None
    source_enum = cast(FlowInputType | FlowOutputType, source_type)
    if mapping.kind == FlowMappingKind.FILE:
        if _is_file_type(source_enum) and _is_file_type(target_type):
            return None
        return "flow.incompatible_mapping"
    if mapping.kind == FlowMappingKind.FILE_ARRAY:
        if _is_file_array_type(source_enum) and _is_file_array_type(
            target_type,
        ):
            return None
        return "flow.incompatible_mapping"
    if mapping.kind in {FlowMappingKind.SELECT, FlowMappingKind.RENAME}:
        if _contract_types_compatible(source_enum, target_type):
            return None
        return "flow.incompatible_mapping"
    return None


def _contract_types_compatible(
    source_type: FlowInputType | FlowOutputType,
    target_type: FlowInputType | FlowOutputType | str,
) -> bool:
    if type(target_type) is str:
        return True
    target_enum = cast(FlowInputType | FlowOutputType, target_type)
    return _semantic_type(source_type) == _semantic_type(target_enum)


def _semantic_type(value: FlowInputType | FlowOutputType) -> str:
    match value:
        case FlowInputType.FILE | FlowOutputType.FILE:
            return "file"
        case FlowInputType.FILE_ARRAY | FlowOutputType.FILE_ARRAY:
            return "file[]"
        case (
            FlowInputType.OBJECT | FlowOutputType.OBJECT | FlowOutputType.JSON
        ):
            return "object"
        case FlowInputType.ARRAY | FlowOutputType.ARRAY:
            return "array"
        case FlowInputType.STRING | FlowOutputType.TEXT:
            return "string"
        case FlowInputType.INTEGER:
            return "integer"
        case FlowInputType.NUMBER:
            return "number"
        case FlowInputType.BOOLEAN:
            return "boolean"


def _is_file_type(value: FlowInputType | FlowOutputType | str) -> bool:
    return value in {FlowInputType.FILE, FlowOutputType.FILE, "file"}


def _is_file_array_type(value: FlowInputType | FlowOutputType | str) -> bool:
    return value in {
        FlowInputType.FILE_ARRAY,
        FlowOutputType.FILE_ARRAY,
        "file[]",
    }


def _mapping_kind_diagnostic(path: str) -> FlowDiagnostic:
    return _diagnostic(
        code="flow.incompatible_mapping",
        path=path,
        message="Flow mapping kind is incompatible.",
        hint="Use a mapping kind that matches the target input.",
    )


def _supports_output_name(
    registry: FlowNodeRegistry,
    node_type: str,
    output_name: str,
) -> bool:
    metadata = registry.metadata(node_type)
    if metadata is None or not metadata.output_contracts:
        return True
    names: set[str] = set()
    for contract in metadata.output_contracts:
        if contract.metadata.get("dynamic"):
            return True
        if contract.name is not None:
            names.add(contract.name)
    return output_name in names


def _supports_output_path(
    registry: FlowNodeRegistry,
    node_type: str,
    selector: FlowSelector,
) -> bool:
    if not selector.path:
        return True
    assert selector.output is not None
    metadata = registry.metadata(node_type)
    if metadata is None or not metadata.output_contracts:
        return True
    for contract in metadata.output_contracts:
        if contract.name == selector.output:
            return _contract_supports_path(contract, selector.path)
        if contract.metadata.get("dynamic"):
            return True
    return False


def _contract_supports_path(
    contract: FlowNodeContract,
    path: tuple[FlowSelectorStep, ...],
) -> bool:
    if contract.metadata.get("dynamic") or contract.schema_ref is not None:
        return True
    if contract.schema is None:
        return True
    return _schema_supports_path(contract.schema, path)


def _schema_supports_path(
    schema: Mapping[str, object],
    path: tuple[FlowSelectorStep, ...],
) -> bool:
    current = schema
    for step in path:
        schema_type = current.get("type")
        if step.kind == FlowSelectorStepKind.FIELD:
            if schema_type is not None and not _schema_has_type(
                schema_type,
                "object",
            ):
                return False
            properties = current.get("properties")
            if not isinstance(properties, Mapping):
                return True
            field_schema = properties.get(step.value)
            if not isinstance(field_schema, Mapping):
                return False
            current = field_schema
            continue
        if schema_type is not None and not _schema_has_type(
            schema_type,
            "array",
        ):
            return False
        items = current.get("items")
        if not isinstance(items, Mapping):
            return True
        current = items
    return True


def _schema_has_type(schema_type: object, expected: str) -> bool:
    if isinstance(schema_type, str):
        return schema_type == expected
    if isinstance(schema_type, list | tuple):
        return expected in schema_type
    return True


def _validate_agent_file_selectors(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
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
        try:
            parsed = parse_flow_selector(
                selector,
                allowed_roots=frozenset({FlowSelectorRoot.NODE_OUTPUT}),
            )
        except FlowSelectorError as error:
            diagnostics.append(_selector_diagnostic(error, path=path))
            continue
        if parsed.source not in node_names:
            diagnostics.append(_bad_reference_diagnostic(path))
            continue
        source_node = definition.node_map[parsed.source]
        assert parsed.output is not None
        if not _supports_output_name(
            registry, source_node.type, parsed.output
        ):
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_node_output",
                    path=path,
                    message="Flow agent file input selector is unknown.",
                    hint="Reference a declared upstream node output.",
                )
            )
            continue
        if not _supports_output_path(registry, source_node.type, parsed):
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_selector_path",
                    path=path,
                    message="Flow agent file input selector path is unknown.",
                    hint="Reference fields declared by the selected output.",
                )
            )
            continue
        if parsed.source not in incoming_sources[node.name]:
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


def _selector_diagnostic(
    error: FlowSelectorError,
    *,
    path: str,
) -> FlowDiagnostic:
    if error.code == "flow.unsafe_selector":
        return _diagnostic(
            code=error.code,
            path=path,
            message="Flow selector uses an unsafe reference.",
            hint="Use declared flow inputs or node outputs only.",
            category=FlowDiagnosticCategory.PRIVACY,
        )
    if error.code == "flow.reserved_selector":
        return _diagnostic(
            code=error.code,
            path=path,
            message="Flow selector uses a reserved reference.",
            hint="Use declared flow inputs or node outputs only.",
            category=FlowDiagnosticCategory.PRIVACY,
        )
    return _diagnostic(
        code="flow.invalid_output_selector",
        path=path,
        message="Flow selector is invalid.",
        hint="Use a safe dotted selector.",
    )


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
