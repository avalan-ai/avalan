from .definition import FlowInputDefinition, FlowNodeDefinition
from .node import Node

from collections.abc import Mapping
from typing import Protocol

FLOW_INPUT_KEY = "__flow_input__"


class FlowNodeFactory(Protocol):
    def __call__(self, definition: FlowNodeDefinition) -> Node: ...


class FlowNodeRegistry:
    def __init__(
        self,
        factories: Mapping[str, FlowNodeFactory] | None = None,
    ) -> None:
        self._factories: dict[str, FlowNodeFactory] = dict(factories or {})

    def register(
        self,
        node_type: str,
        factory: FlowNodeFactory,
    ) -> "FlowNodeRegistry":
        assert isinstance(node_type, str) and node_type.strip()
        assert callable(factory)
        self._factories[node_type] = factory
        return self

    def supports(self, node_type: str) -> bool:
        assert isinstance(node_type, str) and node_type.strip()
        return node_type in self._factories

    def build(self, definition: FlowNodeDefinition) -> Node:
        assert isinstance(definition, FlowNodeDefinition)
        return self._factories[definition.type](definition)


def default_flow_node_registry() -> FlowNodeRegistry:
    return FlowNodeRegistry(
        {
            "constant": _constant_node,
            "echo": _echo_node,
            "input": _input_node,
            "passthrough": _echo_node,
            "select": _select_node,
        }
    )


def flow_input_binding(
    input_definition: FlowInputDefinition | None,
    value: object,
) -> dict[str, object]:
    binding: dict[str, object] = {FLOW_INPUT_KEY: _copy_flow_value(value)}
    if input_definition is None:
        if value is not None:
            binding["value"] = _copy_flow_value(value)
        return binding
    binding[input_definition.name] = _copy_flow_value(value)
    if not isinstance(value, Mapping):
        binding["value"] = _copy_flow_value(value)
    return binding


def _constant_node(definition: FlowNodeDefinition) -> Node:
    def run(_: dict[str, object]) -> object:
        return _copy_flow_value(definition.config.get("value"))

    return Node(definition.name, func=run)


def _echo_node(definition: FlowNodeDefinition) -> Node:
    def run(inputs: dict[str, object]) -> object:
        return _node_input_value(definition, inputs)

    return Node(definition.name, func=run)


def _input_node(definition: FlowNodeDefinition) -> Node:
    def run(inputs: dict[str, object]) -> object:
        return _node_input_value(definition, inputs)

    return Node(definition.name, func=run)


def _select_node(definition: FlowNodeDefinition) -> Node:
    def run(inputs: dict[str, object]) -> object:
        source = _node_input_value(definition, inputs)
        path = definition.config.get("path") or definition.config.get("field")
        assert isinstance(path, str) and path.strip()
        return _select_path(source, path)

    return Node(definition.name, func=run)


def _node_input_value(
    definition: FlowNodeDefinition,
    inputs: Mapping[str, object],
) -> object:
    if definition.input is not None and definition.input in inputs:
        return _copy_flow_value(inputs[definition.input])
    if FLOW_INPUT_KEY in inputs:
        return _copy_flow_value(inputs[FLOW_INPUT_KEY])
    if len(inputs) == 1:
        return _copy_flow_value(next(iter(inputs.values())))
    return _copy_flow_value(dict(inputs))


def _select_path(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdecimal():
            current = current[int(part)]
            continue
        raise KeyError("Flow select node path is missing.")
    return _copy_flow_value(current)


def _copy_flow_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _copy_flow_value(item) for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_copy_flow_value(item) for item in value]
    return value
