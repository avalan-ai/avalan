from .condition import (
    FlowCondition,
    FlowConditionOperator,
)
from .definition import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowRouteMatchPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    default_flow_node_registry,
)
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
_HUMAN_REVIEW_REQUIRED_CONFIG_KEYS = frozenset(
    {
        "allowed_decisions",
        "decision_schema",
        "payload_schema",
        "timeout_seconds",
    }
)
_HUMAN_REVIEW_ALLOWED_SCHEMA_TYPES = frozenset({"object"})
_UNSAFE_AUDIT_KEY_FRAGMENTS = frozenset(
    {
        "password",
        "private",
        "prompt",
        "raw",
        "secret",
        "token",
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
    if (
        node.ref is not None
        and _is_path_escape(node.ref)
        and not registry.supports_tool_resolution(node_type)
    ):
        diagnostics.append(_path_escape_diagnostic(f"nodes.{name}.ref"))
    if node_type == FlowNodeKind.HUMAN_REVIEW.value:
        diagnostics.extend(_validate_human_review_node(node))
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
        diagnostics.extend(
            _validate_registered_node_definition(
                definition,
                node,
                registry,
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


def _validate_registered_node_definition(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    return tuple(
        _configuration_error_diagnostic(error)
        for error in registry.validate_node_definition(definition, node)
    )


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


def _validate_human_review_node(
    node: FlowNodeDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for key in sorted(_HUMAN_REVIEW_REQUIRED_CONFIG_KEYS - set(node.config)):
        diagnostics.append(
            _diagnostic(
                code="flow.missing_human_review_config",
                path=f"nodes.{node.name}.config.{key}",
                message="Human review node configuration is missing.",
                hint="Set the required human review configuration field.",
            )
        )
    diagnostics.extend(
        _validate_human_review_schema(
            node,
            key="payload_schema",
            missing_code="flow.missing_human_review_payload_schema",
            invalid_code="flow.invalid_human_review_payload_schema",
        )
    )
    diagnostics.extend(_validate_human_review_decisions(node))
    diagnostics.extend(_validate_human_review_timeout(node))
    diagnostics.extend(_validate_human_review_audit_metadata(node))
    return tuple(diagnostics)


def _validate_human_review_schema(
    node: FlowNodeDefinition,
    *,
    key: str,
    missing_code: str,
    invalid_code: str,
) -> tuple[FlowDiagnostic, ...]:
    path = f"nodes.{node.name}.config.{key}"
    value = node.config.get(key)
    if value is None:
        return (
            _diagnostic(
                code=missing_code,
                path=path,
                message="Human review schema is missing.",
                hint="Set a JSON-schema object for this field.",
            ),
        )
    if not isinstance(value, Mapping):
        return (
            _diagnostic(
                code=invalid_code,
                path=path,
                message="Human review schema is invalid.",
                hint="Use a JSON-schema object.",
            ),
        )
    schema_type = value.get("type")
    if schema_type not in _HUMAN_REVIEW_ALLOWED_SCHEMA_TYPES:
        return (
            _diagnostic(
                code=invalid_code,
                path=f"{path}.type",
                message="Human review schema is invalid.",
                hint="Use an object schema for human review payloads.",
            ),
        )
    return ()


def _validate_human_review_decisions(
    node: FlowNodeDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    path = f"nodes.{node.name}.config.allowed_decisions"
    value = node.config.get("allowed_decisions")
    if value is None:
        return (
            _diagnostic(
                code="flow.missing_human_review_decisions",
                path=path,
                message="Human review decisions are missing.",
                hint="List every allowed decision explicitly.",
            ),
        ) + _validate_human_review_decision_schema(
            node,
            allowed_decisions=set(),
        )
    if not isinstance(value, list | tuple) or isinstance(value, str | bytes):
        return (
            _diagnostic(
                code="flow.invalid_human_review_decisions",
                path=path,
                message="Human review decisions are invalid.",
                hint="Use a non-empty list of decision names.",
            ),
        )
    decisions = tuple(value)
    if not decisions:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_human_review_decisions",
                path=path,
                message="Human review decisions are missing.",
                hint="List every allowed decision explicitly.",
            )
        )
    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        decision_path = f"{path}[{index}]"
        if not _is_human_review_decision_name(decision):
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_human_review_decision",
                    path=decision_path,
                    message="Human review decision is invalid.",
                    hint=(
                        "Use lowercase letters, numbers, underscores, or"
                        " hyphens."
                    ),
                )
            )
            continue
        assert isinstance(decision, str)
        if decision in seen:
            diagnostics.append(
                _diagnostic(
                    code="flow.duplicate_human_review_decision",
                    path=decision_path,
                    message="Human review decision is duplicated.",
                    hint="List each allowed decision once.",
                )
            )
        seen.add(decision)
    diagnostics.extend(
        _validate_human_review_decision_schema(
            node,
            allowed_decisions=seen,
        )
    )
    return tuple(diagnostics)


def _validate_human_review_decision_schema(
    node: FlowNodeDefinition,
    *,
    allowed_decisions: set[str],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics = list(
        _validate_human_review_schema(
            node,
            key="decision_schema",
            missing_code="flow.missing_human_review_decision_schema",
            invalid_code="flow.invalid_human_review_decision_schema",
        )
    )
    if diagnostics:
        return tuple(diagnostics)
    schema = cast(Mapping[str, object], node.config["decision_schema"])
    decision_property = _schema_property(schema, "decision")
    path = f"nodes.{node.name}.config.decision_schema.properties.decision"
    if decision_property is None:
        return (
            _diagnostic(
                code="flow.invalid_human_review_decision_schema",
                path=path,
                message="Human review decision schema is invalid.",
                hint="Declare a decision property with an enum.",
            ),
        )
    enum_value = decision_property.get("enum")
    if (
        not isinstance(enum_value, list | tuple)
        or isinstance(enum_value, str | bytes)
        or not enum_value
    ):
        return (
            _diagnostic(
                code="flow.invalid_human_review_decision_schema",
                path=f"{path}.enum",
                message="Human review decision schema is invalid.",
                hint="Set decision enum to the allowed decisions.",
            ),
        )
    enum_decisions = {item for item in enum_value if isinstance(item, str)}
    if len(enum_decisions) != len(enum_value):
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_human_review_decision_schema",
                path=f"{path}.enum",
                message="Human review decision schema is invalid.",
                hint="Use string enum values only.",
            )
        )
    if allowed_decisions and enum_decisions != allowed_decisions:
        diagnostics.append(
            _diagnostic(
                code="flow.human_review_decision_schema_mismatch",
                path=f"{path}.enum",
                message=(
                    "Human review decision schema does not match allowed"
                    " decisions."
                ),
                hint=(
                    "Keep decision enum values aligned with allowed decisions."
                ),
            )
        )
    return tuple(diagnostics)


def _validate_human_review_timeout(
    node: FlowNodeDefinition,
) -> tuple[FlowDiagnostic, ...]:
    path = f"nodes.{node.name}.config.timeout_seconds"
    value = node.config.get("timeout_seconds")
    if value is None:
        return (
            _diagnostic(
                code="flow.missing_human_review_timeout",
                path=path,
                message="Human review timeout is missing.",
                hint="Set a positive timeout_seconds value.",
            ),
        )
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or value <= 0
    ):
        return (
            _diagnostic(
                code="flow.invalid_human_review_timeout",
                path=path,
                message="Human review timeout is invalid.",
                hint="Use a positive timeout_seconds value.",
            ),
        )
    return ()


def _validate_human_review_audit_metadata(
    node: FlowNodeDefinition,
) -> tuple[FlowDiagnostic, ...]:
    path = f"nodes.{node.name}.config.audit_metadata"
    value = node.config.get("audit_metadata")
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        return (
            _diagnostic(
                code="flow.invalid_human_review_audit_metadata",
                path=path,
                message="Human review audit metadata is invalid.",
                hint="Use a safe object of audit labels.",
            ),
        )
    unsafe_path = _unsafe_audit_metadata_path(value, path=path)
    if unsafe_path is not None:
        return (
            _diagnostic(
                code="flow.unsafe_human_review_audit_metadata",
                path=unsafe_path,
                category=FlowDiagnosticCategory.PRIVACY,
                message="Human review audit metadata is not public-safe.",
                hint="Remove secret, token, raw, prompt, or private fields.",
            ),
        )
    return ()


def _is_human_review_decision_name(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return all(
        character.islower() or character.isdigit() or character in {"_", "-"}
        for character in value
    )


def _schema_property(
    schema: Mapping[str, object],
    name: str,
) -> Mapping[str, object] | None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    property_value = properties.get(name)
    if not isinstance(property_value, Mapping):
        return None
    return property_value


def _unsafe_audit_metadata_path(
    value: Mapping[str, object],
    *,
    path: str,
) -> str | None:
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            return path
        lowered = key.lower()
        if any(
            fragment in lowered for fragment in _UNSAFE_AUDIT_KEY_FRAGMENTS
        ):
            return f"{path}.{key}"
        if isinstance(item, Mapping):
            unsafe_path = _unsafe_audit_metadata_path(
                item,
                path=f"{path}.{key}",
            )
            if unsafe_path is not None:
                return unsafe_path
    return None


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
        diagnostics.extend(_validate_legacy_edge_routing(definition))
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
    diagnostics.extend(_validate_join_policies(definition, registry))
    diagnostics.extend(
        _validate_node_mappings(definition, node_names, registry)
    )
    diagnostics.extend(_validate_tool_nodes(definition, registry))
    diagnostics.extend(
        _validate_node_policies(definition, node_names, registry)
    )
    diagnostics.extend(_validate_edge_routing(definition))
    return tuple(diagnostics)


def _validate_tool_nodes(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for node in definition.nodes:
        if not registry.supports_tool_resolution(node.type):
            continue
        if node.ref is None:
            continue
        try:
            registry.validate_tool_definition(
                node,
                require_explicit_arguments=True,
            )
        except FlowNodeConfigurationError as error:
            diagnostics.append(_configuration_error_diagnostic(error))
    return tuple(diagnostics)


def _validate_legacy_edge_routing(
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for index, edge in enumerate(definition.edges):
        if (
            edge.kind != FlowEdgeKind.SUCCESS
            or edge.priority != 0
            or edge.default
            or edge.routing_policy != FlowRouteMatchPolicy.EXCLUSIVE
        ):
            diagnostics.append(
                _diagnostic(
                    code="flow.unsupported_edge_policy",
                    path=f"edges[{index}]",
                    message="Flow edge routing policy is not supported.",
                    hint="Use strict flow definitions for route policy.",
                )
            )
    return tuple(diagnostics)


def _validate_edge_routing(
    definition: FlowDefinition,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    groups: dict[tuple[str, str], list[tuple[int, FlowEdgeDefinition]]] = {}
    for index, edge in enumerate(definition.edges):
        path = f"edges[{index}]"
        if edge.priority < 0:
            diagnostics.append(
                _diagnostic(
                    code="flow.invalid_route_priority",
                    path=f"{path}.priority",
                    message="Flow edge priority is invalid.",
                    hint="Use zero or a positive integer priority.",
                )
            )
        if edge.default and edge.condition is not None:
            diagnostics.append(
                _diagnostic(
                    code="flow.default_route_condition",
                    path=f"{path}.condition",
                    message="Flow default route cannot have a condition.",
                    hint="Remove the condition or unset default.",
                )
            )
        groups.setdefault((edge.source, edge.kind.value), []).append(
            (
                index,
                edge,
            )
        )
    for (_, _), indexed_edges in groups.items():
        diagnostics.extend(_validate_route_group(indexed_edges))
    return tuple(diagnostics)


def _validate_route_group(
    indexed_edges: list[tuple[int, FlowEdgeDefinition]],
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    default_indexes = [index for index, edge in indexed_edges if edge.default]
    for index in default_indexes[1:]:
        diagnostics.append(
            _diagnostic(
                code="flow.duplicate_default_route",
                path=f"edges[{index}].default",
                message="Flow source has multiple default routes.",
                hint="Keep only one default route for each source and kind.",
            )
        )
    policies = {edge.routing_policy for _, edge in indexed_edges}
    if len(policies) > 1:
        index = indexed_edges[0][0]
        diagnostics.append(
            _diagnostic(
                code="flow.mixed_routing_policy",
                path=f"edges[{index}].routing_policy",
                message="Flow source has mixed route policies.",
                hint="Use one routing policy for each source and kind.",
            )
        )
        return tuple(diagnostics)
    policy = next(iter(policies), FlowRouteMatchPolicy.EXCLUSIVE)
    if policy == FlowRouteMatchPolicy.ALL_MATCHING:
        return tuple(diagnostics)
    priority_indexes: dict[int, list[int]] = {}
    for index, edge in indexed_edges:
        if edge.default:
            continue
        priority_indexes.setdefault(edge.priority, []).append(index)
    for indexes in priority_indexes.values():
        if len(indexes) <= 1:
            continue
        diagnostics.append(
            _diagnostic(
                code="flow.ambiguous_route",
                path=f"edges[{indexes[1]}].priority",
                message="Flow source has ambiguous outgoing routes.",
                hint=(
                    "Use distinct priorities, all_matching, or one "
                    "explicit default route."
                ),
            )
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


def _validate_join_policies(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    inbound_counts = {node.name: 0 for node in definition.nodes}
    for edge in definition.edges:
        inbound_counts[edge.target] += 1
    for node in definition.nodes:
        inbound_count = inbound_counts[node.name]
        path = f"nodes.{node.name}.join_policy"
        if inbound_count > 1 and node.join_policy is None:
            diagnostics.append(
                _diagnostic(
                    code="flow.missing_join_policy",
                    path=path,
                    message="Flow node has multiple inbound paths.",
                    hint="Declare a join policy for fan-in nodes.",
                )
            )
            continue
        if node.join_policy is None:
            continue
        diagnostics.extend(
            _validate_join_policy_shape(
                node.join_policy,
                path=path,
                inbound_count=inbound_count,
            )
        )
        diagnostics.extend(
            _validate_join_policy_optional_inputs(
                node,
                registry,
                path=path,
            )
        )
    return tuple(diagnostics)


def _validate_join_policy_shape(
    policy: FlowJoinPolicy,
    *,
    path: str,
    inbound_count: int,
) -> tuple[FlowDiagnostic, ...]:
    if policy.type == FlowJoinPolicyType.QUORUM:
        if policy.quorum is None:
            return (
                _diagnostic(
                    code="flow.missing_join_quorum",
                    path=f"{path}.quorum",
                    message="Flow quorum join policy is missing quorum.",
                    hint="Set a positive quorum no larger than inbound paths.",
                ),
            )
        if policy.quorum <= 0 or policy.quorum > inbound_count:
            return (
                _diagnostic(
                    code="flow.invalid_join_quorum",
                    path=f"{path}.quorum",
                    message="Flow quorum join policy is invalid.",
                    hint="Use a positive quorum no larger than inbound paths.",
                ),
            )
        return ()
    if policy.quorum is None:
        return ()
    return (
        _diagnostic(
            code="flow.unsupported_join_field",
            path=f"{path}.quorum",
            message="Flow join policy field is not supported.",
            hint="Use quorum only with the quorum join policy.",
        ),
    )


def _validate_join_policy_optional_inputs(
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    assert node.join_policy is not None
    diagnostics: list[FlowDiagnostic] = []
    seen: set[str] = set()
    metadata = registry.metadata(node.type)
    known_inputs = {
        contract.name
        for contract in (metadata.input_contracts if metadata else ())
        if contract.name is not None
    }
    for index, name in enumerate(node.join_policy.optional_inputs):
        input_path = f"{path}.optional_inputs[{index}]"
        if name in seen:
            diagnostics.append(
                _diagnostic(
                    code="flow.duplicate_join_optional_input",
                    path=input_path,
                    message="Flow join optional input is duplicated.",
                    hint="List each optional input once.",
                )
            )
        seen.add(name)
        if name not in known_inputs:
            diagnostics.append(
                _diagnostic(
                    code="flow.unknown_join_optional_input",
                    path=input_path,
                    message="Flow join optional input is unknown.",
                    hint="Reference a declared target input contract.",
                )
            )
    return tuple(diagnostics)


def _validate_node_policies(
    definition: FlowDefinition,
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    input_names = {
        input_definition.name for input_definition in definition.inputs
    }
    for node in definition.nodes:
        if node.retry_policy is not None:
            diagnostics.extend(
                _validate_retry_policy(definition, node, node_names)
            )
        if node.timeout_policy is not None:
            diagnostics.extend(_validate_timeout_policy(node))
        if node.loop_policy is not None:
            diagnostics.extend(
                _validate_loop_policy(
                    definition,
                    node,
                    node.loop_policy,
                    input_names=input_names,
                    node_names=node_names,
                    registry=registry,
                )
            )
    return tuple(diagnostics)


def _validate_retry_policy(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    node_names: set[str],
) -> tuple[FlowDiagnostic, ...]:
    assert node.retry_policy is not None
    policy = node.retry_policy
    path = f"nodes.{node.name}.retry_policy"
    diagnostics: list[FlowDiagnostic] = []
    if policy.max_attempts is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_retry_attempts",
                path=f"{path}.max_attempts",
                message="Flow retry policy is missing max attempts.",
                hint="Set a positive max_attempts value.",
            )
        )
    elif policy.max_attempts <= 0:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_retry_attempts",
                path=f"{path}.max_attempts",
                message="Flow retry policy has invalid max attempts.",
                hint="Use a positive bounded max_attempts value.",
            )
        )
    diagnostics.extend(_validate_retry_backoff(policy, path=path))
    diagnostics.extend(_validate_retry_categories(policy, path=path))
    if policy.exhausted_route is not None:
        diagnostics.extend(
            _validate_policy_route(
                definition,
                node,
                route=policy.exhausted_route,
                path=f"{path}.exhausted_route",
                node_names=node_names,
                unknown_code="flow.unknown_retry_exhaustion_route",
                missing_code="flow.missing_retry_exhaustion_route",
                required_kind=FlowEdgeKind.ERROR,
            )
        )
    return tuple(diagnostics)


def _validate_retry_backoff(
    policy: FlowRetryPolicy,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    if policy.backoff == FlowRetryBackoffStrategy.NONE:
        if policy.initial_delay_seconds is not None:
            diagnostics.append(_unsupported_retry_backoff_field(path))
        if policy.max_delay_seconds is not None:
            diagnostics.append(_unsupported_retry_backoff_field(path))
        return tuple(diagnostics)
    if policy.initial_delay_seconds is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_retry_backoff_delay",
                path=f"{path}.initial_delay_seconds",
                message="Flow retry backoff is missing an initial delay.",
                hint="Set a positive initial_delay_seconds value.",
            )
        )
    elif policy.initial_delay_seconds <= 0:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_retry_backoff_delay",
                path=f"{path}.initial_delay_seconds",
                message="Flow retry backoff delay is invalid.",
                hint="Use a positive initial delay.",
            )
        )
    if policy.max_delay_seconds is not None and policy.max_delay_seconds <= 0:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_retry_max_delay",
                path=f"{path}.max_delay_seconds",
                message="Flow retry maximum delay is invalid.",
                hint="Use a positive maximum delay.",
            )
        )
    if (
        policy.initial_delay_seconds is not None
        and policy.max_delay_seconds is not None
        and policy.max_delay_seconds < policy.initial_delay_seconds
    ):
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_retry_max_delay",
                path=f"{path}.max_delay_seconds",
                message="Flow retry maximum delay is invalid.",
                hint="Use a maximum delay greater than the initial delay.",
            )
        )
    return tuple(diagnostics)


def _unsupported_retry_backoff_field(path: str) -> FlowDiagnostic:
    return _diagnostic(
        code="flow.unsupported_retry_backoff_field",
        path=f"{path}.backoff",
        message="Flow retry backoff field is not supported.",
        hint="Use delay fields only with a backoff strategy.",
    )


def _validate_retry_categories(
    policy: FlowRetryPolicy,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    retryable = set(policy.retryable_categories)
    non_retryable = set(policy.non_retryable_categories)
    if len(retryable) != len(policy.retryable_categories):
        diagnostics.append(
            _diagnostic(
                code="flow.duplicate_retry_category",
                path=f"{path}.retryable_categories",
                message="Flow retry category is duplicated.",
                hint="List each retryable category once.",
            )
        )
    if len(non_retryable) != len(policy.non_retryable_categories):
        diagnostics.append(
            _diagnostic(
                code="flow.duplicate_retry_category",
                path=f"{path}.non_retryable_categories",
                message="Flow retry category is duplicated.",
                hint="List each non-retryable category once.",
            )
        )
    if retryable.intersection(non_retryable):
        diagnostics.append(
            _diagnostic(
                code="flow.conflicting_retry_category",
                path=f"{path}.retryable_categories",
                message="Flow retry category has conflicting behavior.",
                hint=(
                    "Do not list the same category as retryable and "
                    "non-retryable."
                ),
            )
        )
    return tuple(diagnostics)


def _validate_timeout_policy(
    node: FlowNodeDefinition,
) -> tuple[FlowDiagnostic, ...]:
    assert node.timeout_policy is not None
    policy = node.timeout_policy
    path = f"nodes.{node.name}.timeout_policy.per_attempt_seconds"
    if policy.per_attempt_seconds is None:
        return (
            _diagnostic(
                code="flow.missing_timeout",
                path=path,
                message="Flow timeout policy is missing per-attempt timeout.",
                hint="Set a positive per_attempt_seconds value.",
            ),
        )
    if policy.per_attempt_seconds <= 0:
        return (
            _diagnostic(
                code="flow.invalid_timeout",
                path=path,
                message="Flow timeout policy has invalid per-attempt timeout.",
                hint="Use a positive per_attempt_seconds value.",
            ),
        )
    return ()


def _validate_loop_policy(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    policy: FlowLoopPolicy,
    *,
    input_names: set[str],
    node_names: set[str],
    registry: FlowNodeRegistry,
) -> tuple[FlowDiagnostic, ...]:
    path = f"nodes.{node.name}.loop_policy"
    diagnostics: list[FlowDiagnostic] = []
    diagnostics.extend(_validate_loop_bounds(policy, path=path))
    if policy.continue_condition is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_loop_continue_condition",
                path=f"{path}.continue_condition",
                message="Flow loop policy is missing continue condition.",
                hint="Set a declarative continue condition.",
            )
        )
    else:
        diagnostics.extend(
            _validate_condition(
                definition,
                policy.continue_condition,
                path=f"{path}.continue_condition",
                edge_source=node.name,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    if policy.exit_condition is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_loop_exit_condition",
                path=f"{path}.exit_condition",
                message="Flow loop policy is missing exit condition.",
                hint="Set a declarative exit condition.",
            )
        )
    else:
        diagnostics.extend(
            _validate_condition(
                definition,
                policy.exit_condition,
                path=f"{path}.exit_condition",
                edge_source=node.name,
                input_names=input_names,
                node_names=node_names,
                registry=registry,
            )
        )
    if policy.output_selector is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_loop_output",
                path=f"{path}.output_selector",
                message="Flow loop policy is missing normal output.",
                hint="Set a safe node output selector.",
            )
        )
    else:
        diagnostics.extend(
            _validate_loop_output_selector(
                registry,
                node,
                policy.output_selector,
                path=f"{path}.output_selector",
            )
        )
    if policy.limit_route is None:
        diagnostics.append(
            _diagnostic(
                code="flow.missing_loop_limit_route",
                path=f"{path}.limit_route",
                message="Flow loop policy is missing limit route.",
                hint="Set a route for loop limit exhaustion.",
            )
        )
    else:
        diagnostics.extend(
            _validate_policy_route(
                definition,
                node,
                route=policy.limit_route,
                path=f"{path}.limit_route",
                node_names=node_names,
                unknown_code="flow.unknown_loop_limit_route",
                missing_code="flow.missing_loop_limit_route",
            )
        )
    return tuple(diagnostics)


def _validate_loop_bounds(
    policy: FlowLoopPolicy,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    if policy.max_iterations is None and policy.max_elapsed_seconds is None:
        diagnostics.append(
            _diagnostic(
                code="flow.unbounded_loop",
                path=path,
                message="Flow loop policy is unbounded.",
                hint="Set max_iterations or max_elapsed_seconds.",
            )
        )
    if policy.max_iterations is not None and policy.max_iterations <= 0:
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_loop_iterations",
                path=f"{path}.max_iterations",
                message="Flow loop max iterations is invalid.",
                hint="Use a positive max_iterations value.",
            )
        )
    if (
        policy.max_elapsed_seconds is not None
        and policy.max_elapsed_seconds <= 0
    ):
        diagnostics.append(
            _diagnostic(
                code="flow.invalid_loop_elapsed_time",
                path=f"{path}.max_elapsed_seconds",
                message="Flow loop max elapsed time is invalid.",
                hint="Use a positive max_elapsed_seconds value.",
            )
        )
    return tuple(diagnostics)


def _validate_loop_output_selector(
    registry: FlowNodeRegistry,
    node: FlowNodeDefinition,
    selector: str,
    *,
    path: str,
) -> tuple[FlowDiagnostic, ...]:
    try:
        parsed = parse_flow_selector(
            selector,
            allowed_roots=frozenset({FlowSelectorRoot.NODE_OUTPUT}),
        )
    except FlowSelectorError as error:
        return (_selector_diagnostic(error, path=path),)
    if parsed.source != node.name:
        return (
            _diagnostic(
                code="flow.bad_reference",
                path=path,
                message="Flow loop output selector is disconnected.",
                hint="Reference an output from the loop node.",
            ),
        )
    assert parsed.output is not None
    if not _supports_output_name(registry, node.type, parsed.output):
        return (
            _diagnostic(
                code="flow.unknown_node_output",
                path=path,
                message="Flow loop output selector references unknown output.",
                hint="Reference a declared output for the loop node.",
            ),
        )
    if not _supports_output_path(registry, node.type, parsed):
        return (
            _diagnostic(
                code="flow.unknown_selector_path",
                path=path,
                message="Flow loop output selector path is unknown.",
                hint="Reference fields declared by the selected output.",
            ),
        )
    return ()


def _validate_policy_route(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    *,
    route: str,
    path: str,
    node_names: set[str],
    unknown_code: str,
    missing_code: str,
    required_kind: FlowEdgeKind | None = None,
) -> tuple[FlowDiagnostic, ...]:
    if route not in node_names:
        return (
            _diagnostic(
                code=unknown_code,
                path=path,
                message="Flow policy route references an unknown node.",
                hint="Reference a declared node.",
            ),
        )
    for edge in definition.edges:
        if edge.source != node.name or edge.target != route:
            continue
        if required_kind is None or edge.kind == required_kind:
            return ()
    return (
        _diagnostic(
            code=missing_code,
            path=path,
            message="Flow policy route is missing a matching edge.",
            hint="Add an edge from the policy node to the route target.",
        ),
    )


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
            and contract.name not in _join_optional_inputs(node.join_policy)
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


def _join_optional_inputs(
    join_policy: FlowJoinPolicy | None,
) -> frozenset[str]:
    if join_policy is None:
        return frozenset()
    return frozenset(join_policy.optional_inputs)


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


def _configuration_error_diagnostic(
    error: FlowNodeConfigurationError,
) -> FlowDiagnostic:
    return _diagnostic(
        code=error.code,
        path=error.path,
        message=error.message,
        hint=error.hint,
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
