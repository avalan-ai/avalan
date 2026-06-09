from .condition import FlowCondition
from .definition import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputMapping,
    FlowJoinPolicy,
    FlowLoopPolicy,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowRetryPolicy,
    FlowTimeoutPolicy,
)

from collections.abc import Mapping, Sequence
from enum import Enum
from json import dumps
from math import isfinite


def serialize_flow_definition(definition: FlowDefinition) -> str:
    assert isinstance(definition, FlowDefinition)
    writer = _TomlWriter()
    _write_flow_definition(writer, definition)
    return writer.render()


class _TomlWriter:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def table(self, *path: str) -> None:
        self._blank_before_header()
        self._lines.append(f"[{_toml_path(path)}]")

    def array_table(self, *path: str) -> None:
        self._blank_before_header()
        self._lines.append(f"[[{_toml_path(path)}]]")

    def field(self, key: str, value: object) -> None:
        self._lines.append(f"{_toml_key(key)} = {_toml_value(value)}")

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"

    def _blank_before_header(self) -> None:
        if self._lines and self._lines[-1] != "":
            self._lines.append("")


def _write_flow_definition(
    writer: _TomlWriter,
    definition: FlowDefinition,
) -> None:
    writer.table("flow")
    writer.field("name", definition.name)
    _write_optional_fields(
        writer,
        {
            "version": definition.version,
            "revision": definition.revision,
            "description": definition.description,
            "entrypoint": definition.entrypoint,
            "output_node": definition.output_node,
        },
    )
    if definition.tags:
        writer.field("tags", definition.tags)
    if definition.input is not None:
        _write_input_definition(writer, ("flow", "input"), definition.input)
    if definition.output is not None:
        _write_output_definition(
            writer,
            ("flow", "output"),
            definition.output,
        )
    for input_definition in definition.inputs:
        writer.array_table("inputs")
        _write_fields(writer, _input_fields(input_definition))
    for output_definition in definition.outputs:
        writer.array_table("outputs")
        _write_fields(writer, _output_fields(output_definition))
    if definition.entry_behavior is not None:
        _write_entry_behavior(writer, definition.entry_behavior)
    if definition.output_behavior is not None:
        _write_output_behavior(writer, definition.output_behavior)
    _write_metadata_table(
        writer, ("runtime_limits",), definition.runtime_limits
    )
    _write_metadata_table(writer, ("privacy",), definition.privacy_policy)
    _write_metadata_table(
        writer,
        ("observability",),
        definition.observability_policy,
    )
    _write_metadata_table(writer, ("ownership",), definition.ownership)
    _write_metadata_table(writer, ("variables",), definition.variables)
    for node in definition.nodes:
        _write_node_definition(writer, node)
    for edge in definition.edges:
        _write_edge_definition(writer, edge)


def _write_input_definition(
    writer: _TomlWriter,
    path: tuple[str, ...],
    definition: FlowInputDefinition,
) -> None:
    writer.table(*path)
    _write_fields(writer, _input_fields(definition))


def _write_output_definition(
    writer: _TomlWriter,
    path: tuple[str, ...],
    definition: FlowOutputDefinition,
) -> None:
    writer.table(*path)
    _write_fields(writer, _output_fields(definition))


def _input_fields(definition: FlowInputDefinition) -> dict[str, object]:
    fields: dict[str, object] = {
        "name": definition.name,
        "type": definition.type.value,
    }
    if definition.mime_types:
        fields["mime_types"] = definition.mime_types
    if definition.schema is not None:
        fields["schema"] = definition.schema
    if definition.schema_ref is not None:
        fields["schema_ref"] = definition.schema_ref
    return fields


def _output_fields(definition: FlowOutputDefinition) -> dict[str, object]:
    fields: dict[str, object] = {
        "name": definition.name,
        "type": definition.type.value,
    }
    if definition.schema is not None:
        fields["schema"] = definition.schema
    if definition.schema_ref is not None:
        fields["schema_ref"] = definition.schema_ref
    return fields


def _write_entry_behavior(
    writer: _TomlWriter,
    behavior: FlowEntryBehavior,
) -> None:
    writer.table("entry")
    writer.field("type", behavior.type.value)
    writer.field("node", behavior.node)


def _write_output_behavior(
    writer: _TomlWriter,
    behavior: FlowOutputBehavior,
) -> None:
    writer.table("output_behavior")
    writer.field("type", behavior.type.value)
    writer.table("output_behavior", "outputs")
    _write_sorted_mapping(writer, behavior.outputs)


def _write_metadata_table(
    writer: _TomlWriter,
    path: tuple[str, ...],
    value: Mapping[str, object],
) -> None:
    if not value:
        return
    writer.table(*path)
    _write_sorted_mapping(writer, value)


def _write_node_definition(
    writer: _TomlWriter,
    node: FlowNodeDefinition,
) -> None:
    writer.table("nodes", node.name)
    writer.field("type", node.type)
    _write_optional_fields(
        writer,
        {
            "ref": node.ref,
            "input": node.input,
            "output": node.output,
        },
    )
    if node.join_policy is not None:
        _write_join_policy(writer, node.name, node.join_policy)
    if node.retry_policy is not None:
        _write_retry_policy(writer, node.name, node.retry_policy)
    if node.timeout_policy is not None:
        _write_timeout_policy(writer, node.name, node.timeout_policy)
    if node.loop_policy is not None:
        _write_loop_policy(writer, node.name, node.loop_policy)
    for mapping in node.mappings:
        _write_node_mapping(writer, node.name, mapping)
    if node.config:
        writer.table("nodes", node.name, "config")
        _write_sorted_mapping(writer, node.config)


def _write_join_policy(
    writer: _TomlWriter,
    node_name: str,
    policy: FlowJoinPolicy,
) -> None:
    writer.table("nodes", node_name, "join_policy")
    writer.field("type", policy.type.value)
    if policy.quorum is not None:
        writer.field("quorum", policy.quorum)
    if policy.optional_inputs:
        writer.field("optional_inputs", policy.optional_inputs)


def _write_retry_policy(
    writer: _TomlWriter,
    node_name: str,
    policy: FlowRetryPolicy,
) -> None:
    writer.table("nodes", node_name, "retry_policy")
    if policy.max_attempts is not None:
        writer.field("max_attempts", policy.max_attempts)
    writer.field("backoff", policy.backoff.value)
    _write_optional_fields(
        writer,
        {
            "initial_delay_seconds": policy.initial_delay_seconds,
            "max_delay_seconds": policy.max_delay_seconds,
        },
    )
    if policy.retryable_categories:
        writer.field("retryable_categories", policy.retryable_categories)
    if policy.non_retryable_categories:
        writer.field(
            "non_retryable_categories",
            policy.non_retryable_categories,
        )
    if policy.exhausted_route is not None:
        writer.field("exhausted_route", policy.exhausted_route)


def _write_timeout_policy(
    writer: _TomlWriter,
    node_name: str,
    policy: FlowTimeoutPolicy,
) -> None:
    writer.table("nodes", node_name, "timeout_policy")
    if policy.per_attempt_seconds is not None:
        writer.field("per_attempt_seconds", policy.per_attempt_seconds)


def _write_loop_policy(
    writer: _TomlWriter,
    node_name: str,
    policy: FlowLoopPolicy,
) -> None:
    writer.table("nodes", node_name, "loop_policy")
    _write_optional_fields(
        writer,
        {
            "max_iterations": policy.max_iterations,
            "max_elapsed_seconds": policy.max_elapsed_seconds,
            "output_selector": policy.output_selector,
            "limit_route": policy.limit_route,
        },
    )
    if policy.continue_condition is not None:
        writer.field(
            "continue_condition",
            _condition_value(policy.continue_condition),
        )
    if policy.exit_condition is not None:
        writer.field("exit_condition", _condition_value(policy.exit_condition))


def _write_node_mapping(
    writer: _TomlWriter,
    node_name: str,
    mapping: FlowInputMapping,
) -> None:
    writer.table("nodes", node_name, "mapping", mapping.target)
    writer.field("type", mapping.kind.value)
    if mapping.source is not None:
        writer.field("source", mapping.source)
    if mapping.sources:
        writer.field("sources", mapping.sources)
    if mapping.fields:
        writer.field("fields", mapping.fields)
    if mapping.items:
        writer.field("items", mapping.items)


def _write_edge_definition(
    writer: _TomlWriter,
    edge: FlowEdgeDefinition,
) -> None:
    writer.array_table("edges")
    writer.field("source", edge.source)
    writer.field("target", edge.target)
    if edge.label is not None:
        writer.field("label", edge.label)
    writer.field("kind", edge.kind.value)
    writer.field("priority", edge.priority)
    writer.field("default", edge.default)
    writer.field("routing_policy", edge.routing_policy.value)
    if edge.condition is not None:
        writer.field("condition", _condition_value(edge.condition))


def _condition_value(condition: FlowCondition) -> dict[str, object]:
    value: dict[str, object] = {"op": condition.operator.value}
    if condition.selector is not None:
        value["selector"] = condition.selector
    if condition.value is not None:
        value["value"] = condition.value
    if condition.value_selector is not None:
        value["value_selector"] = condition.value_selector
    if condition.values:
        value["values"] = condition.values
    if condition.value_type is not None:
        value["value_type"] = condition.value_type.value
    if condition.conditions:
        value["conditions"] = tuple(
            _condition_value(child) for child in condition.conditions
        )
    if condition.condition is not None:
        value["condition"] = _condition_value(condition.condition)
    return value


def _write_fields(
    writer: _TomlWriter,
    values: Mapping[str, object],
) -> None:
    for key, value in values.items():
        writer.field(key, value)


def _write_optional_fields(
    writer: _TomlWriter,
    values: Mapping[str, object | None],
) -> None:
    for key, value in values.items():
        if value is not None:
            writer.field(key, value)


def _write_sorted_mapping(
    writer: _TomlWriter,
    value: Mapping[str, object],
) -> None:
    for key in sorted(value):
        writer.field(key, value[key])


def _toml_path(path: Sequence[str]) -> str:
    assert path, "path must not be empty"
    return ".".join(_toml_key(part) for part in path)


def _toml_key(value: str) -> str:
    assert isinstance(value, str) and value.strip()
    if (
        value.replace("_", "").replace("-", "").isalnum()
        and not value[0].isdigit()
    ):
        return value
    return dumps(value)


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return dumps(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        assert isfinite(value), "TOML numbers must be finite"
        return repr(value)
    if isinstance(value, Enum):
        return _toml_value(value.value)
    if isinstance(value, Mapping):
        return _toml_inline_table(value)
    if isinstance(value, list | tuple):
        return _toml_array(value)
    raise AssertionError("unsupported TOML value")


def _toml_array(value: Sequence[object]) -> str:
    return "[" + ", ".join(_toml_value(item) for item in value) + "]"


def _toml_inline_table(value: Mapping[object, object]) -> str:
    items: list[str] = []
    for key in sorted(value, key=str):
        assert isinstance(key, str) and key.strip()
        items.append(f"{_toml_key(key)} = {_toml_value(value[key])}")
    return "{" + ", ".join(items) + "}"
