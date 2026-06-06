from ..entities import (
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from .definition import (
    FlowInputDefinition,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowOutputType,
)
from .node import Node

from collections.abc import Iterable, Mapping
from typing import Protocol

FLOW_INPUT_KEY = "__flow_input__"
FLOW_TOOL_NODE_TYPE = "tool"


class FlowNodeFactory(Protocol):
    def __call__(self, definition: FlowNodeDefinition) -> Node: ...


class FlowToolResolver(Protocol):
    def list_tools(self) -> list[ToolDescriptor]: ...

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution: ...


class FlowNodeConfigurationError(ValueError):
    def __init__(
        self,
        *,
        code: str,
        path: str,
        message: str,
        hint: str,
    ) -> None:
        assert isinstance(code, str) and code.strip()
        assert isinstance(path, str) and path.strip()
        assert isinstance(message, str) and message.strip()
        assert isinstance(hint, str) and hint.strip()
        self.code = code
        self.path = path
        self.message = message
        self.hint = hint
        super().__init__(code)


class FlowNodeRegistry:
    def __init__(
        self,
        factories: Mapping[str, FlowNodeFactory] | None = None,
        metadata: Mapping[str, FlowNodeMetadata] | None = None,
    ) -> None:
        self._factories: dict[str, FlowNodeFactory] = {}
        self._metadata: dict[str, FlowNodeMetadata] = {}
        node_metadata = metadata or {}
        for node_type, factory in (factories or {}).items():
            self.register(
                node_type,
                factory,
                metadata=node_metadata.get(node_type),
            )

    def register(
        self,
        node_type: str,
        factory: FlowNodeFactory,
        *,
        metadata: FlowNodeMetadata | None = None,
    ) -> "FlowNodeRegistry":
        assert isinstance(node_type, str) and node_type.strip()
        assert callable(factory)
        if metadata is not None:
            assert isinstance(metadata, FlowNodeMetadata)
        self._factories[node_type] = factory
        self._metadata[node_type] = metadata or FlowNodeMetadata()
        return self

    def supports(self, node_type: str) -> bool:
        assert isinstance(node_type, str) and node_type.strip()
        return node_type in self._factories

    def metadata(self, node_type: str) -> FlowNodeMetadata | None:
        assert isinstance(node_type, str) and node_type.strip()
        return self._metadata.get(node_type)

    def supports_ref(self, node_type: str) -> bool:
        assert isinstance(node_type, str) and node_type.strip()
        metadata = self.metadata(node_type)
        return metadata.supports_ref if metadata is not None else False

    def is_async_only(self, node_type: str) -> bool:
        assert isinstance(node_type, str) and node_type.strip()
        metadata = self.metadata(node_type)
        return metadata.async_only if metadata is not None else False

    def input_contract(self, node_type: str) -> FlowNodeContract | None:
        assert isinstance(node_type, str) and node_type.strip()
        metadata = self.metadata(node_type)
        return metadata.input_contract if metadata is not None else None

    def output_contract(self, node_type: str) -> FlowNodeContract | None:
        assert isinstance(node_type, str) and node_type.strip()
        metadata = self.metadata(node_type)
        return metadata.output_contract if metadata is not None else None

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
        },
        {
            "constant": FlowNodeMetadata(
                output_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
            ),
            "echo": FlowNodeMetadata(
                input_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
                output_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
            ),
            "input": FlowNodeMetadata(
                output_contract=FlowNodeContract(
                    name="value",
                    type=FlowOutputType.OBJECT,
                    metadata={"dynamic": True},
                ),
            ),
            "passthrough": FlowNodeMetadata(
                input_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
                output_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
            ),
            "select": FlowNodeMetadata(
                input_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.OBJECT,
                    metadata={"dynamic": True},
                ),
                output_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
            ),
        },
    )


def tool_flow_node_registry(
    resolver: FlowToolResolver,
    *,
    base_registry: FlowNodeRegistry | None = None,
) -> FlowNodeRegistry:
    assert _is_flow_tool_resolver(resolver)
    if base_registry is not None:
        assert isinstance(base_registry, FlowNodeRegistry)
    registry = base_registry or default_flow_node_registry()
    descriptors = {
        descriptor.name: descriptor for descriptor in resolver.list_tools()
    }
    registry.register(
        FLOW_TOOL_NODE_TYPE,
        _tool_node_factory(resolver, descriptors),
        metadata=FlowNodeMetadata(
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
            metadata={"tools": _tool_contracts(descriptors.values())},
        ),
    )
    return registry


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


def _tool_node_factory(
    resolver: FlowToolResolver,
    descriptors: Mapping[str, ToolDescriptor],
) -> FlowNodeFactory:
    def factory(definition: FlowNodeDefinition) -> Node:
        path = f"nodes.{definition.name}.ref"
        requested_name = definition.ref
        if requested_name is None:
            raise FlowNodeConfigurationError(
                code="flow.missing_ref",
                path=path,
                message="Tool flow nodes must declare a tool ref.",
                hint="Set ref to an enabled avalan tool name.",
            )
        if _is_ref_import_like(requested_name):
            raise FlowNodeConfigurationError(
                code="flow.invalid_ref",
                path=path,
                message="Tool flow node ref must be a tool name.",
                hint="Use an enabled avalan tool name, not a path or URI.",
            )
        resolution = resolver.resolve_tool_name(requested_name)
        if resolution.status not in {
            ToolNameResolutionStatus.EXACT,
            ToolNameResolutionStatus.ALIAS,
        }:
            raise FlowNodeConfigurationError(
                code=f"flow.tool_{resolution.status.value}",
                path=path,
                message=(
                    "Tool flow node ref does not resolve to one enabled tool."
                ),
                hint="Use an enabled avalan tool name or unambiguous alias.",
            )
        assert resolution.canonical_name is not None
        descriptor = descriptors[resolution.canonical_name]

        async def run(_: dict[str, object]) -> object:
            raise NotImplementedError(
                "Flow tool node execution is unavailable."
            )

        return Node(
            definition.name,
            label=descriptor.name,
            func=run,
        )

    return factory


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


def _is_flow_tool_resolver(value: object) -> bool:
    return (
        hasattr(value, "list_tools")
        and callable(getattr(value, "list_tools"))
        and hasattr(value, "resolve_tool_name")
        and callable(getattr(value, "resolve_tool_name"))
    )


def _tool_contracts(
    descriptors: Iterable[ToolDescriptor],
) -> dict[str, dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {}
    for descriptor in descriptors:
        assert isinstance(descriptor, ToolDescriptor)
        contract: dict[str, object] = {"aliases": tuple(descriptor.aliases)}
        if descriptor.parameter_schema is not None:
            contract["parameter_schema"] = descriptor.parameter_schema
        if descriptor.return_schema is not None:
            contract["return_schema"] = descriptor.return_schema
        contracts[descriptor.name] = contract
    return contracts


def _is_ref_import_like(ref: str) -> bool:
    return "://" in ref or "/" in ref or "\\" in ref or ":" in ref
