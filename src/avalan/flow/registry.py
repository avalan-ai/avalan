from ..entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
    ToolValue,
)
from ..utils import tool_call_diagnostic_payload, tool_call_error_payload
from .definition import (
    FlowInputDefinition,
    FlowInputType,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowOutputType,
)
from .node import Node

from asyncio import CancelledError
from collections.abc import Awaitable, Iterable, Mapping
from typing import Any, Protocol, cast
from uuid import uuid4

FLOW_INPUT_KEY = "__flow_input__"
FLOW_TOOL_NODE_TYPE = "tool"


class FlowNodeFactory(Protocol):
    def __call__(self, definition: FlowNodeDefinition) -> Node: ...


class FlowToolResolver(Protocol):
    def list_tools(self) -> list[ToolDescriptor]: ...

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution: ...

    def validate_tool_call(
        self, call: ToolCall
    ) -> ToolCallDiagnostic | None: ...

    def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> Awaitable[ToolCallOutcome]: ...


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
                kind=FlowNodeKind.CONSTANT,
                output_contract=FlowNodeContract(
                    name="value",
                    metadata={"dynamic": True},
                ),
            ),
            "echo": FlowNodeMetadata(
                kind=FlowNodeKind.PASS_THROUGH,
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
                kind=FlowNodeKind.INPUT,
                output_contract=FlowNodeContract(
                    name="value",
                    type=FlowOutputType.OBJECT,
                    metadata={"dynamic": True},
                ),
            ),
            "passthrough": FlowNodeMetadata(
                kind=FlowNodeKind.PASS_THROUGH,
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
                kind=FlowNodeKind.SELECT,
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
            kind=FlowNodeKind.TOOL,
            supports_ref=True,
            async_only=True,
            capabilities=(FlowNodeCapability.ASYNC_ONLY,),
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
        _validate_tool_node_bindings(definition, descriptor)
        _validate_tool_node_output_mode(definition)

        async def run(
            inputs: dict[str, object],
            *,
            cancellation_checker: Any = None,
        ) -> object:
            arguments = cast(
                dict[str, ToolValue],
                _tool_node_arguments(definition, descriptor, inputs),
            )
            call = ToolCall(
                id=str(uuid4()),
                name=descriptor.name,
                arguments=arguments,
            )
            diagnostic = resolver.validate_tool_call(call)
            if diagnostic is not None:
                return _tool_node_output(definition, diagnostic)
            outcome = await resolver.execute_call(
                call,
                context=ToolCallContext(
                    cancellation_checker=cancellation_checker,
                    flow_tool_node=True,
                ),
            )
            return _tool_node_output(definition, outcome)

        return Node(
            definition.name,
            label=descriptor.name,
            func=run,
            async_only=True,
            receives_cancellation_checker=True,
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
        and hasattr(value, "validate_tool_call")
        and callable(getattr(value, "validate_tool_call"))
        and hasattr(value, "execute_call")
        and callable(getattr(value, "execute_call"))
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
            contract["input_contract"] = FlowNodeContract(
                name="arguments",
                type=FlowInputType.OBJECT,
                schema=descriptor.parameter_schema,
            )
        if descriptor.return_schema is not None:
            contract["return_schema"] = descriptor.return_schema
            contract["output_contract"] = FlowNodeContract(
                name="result",
                type=FlowOutputType.JSON,
                schema=descriptor.return_schema,
            )
        contracts[descriptor.name] = contract
    return contracts


def _is_ref_import_like(ref: str) -> bool:
    return "://" in ref or "/" in ref or "\\" in ref or ":" in ref


def _validate_tool_node_bindings(
    definition: FlowNodeDefinition,
    descriptor: ToolDescriptor,
) -> None:
    bindings = definition.config.get("arguments")
    if bindings is None:
        return
    path = f"nodes.{definition.name}.config.arguments"
    if not isinstance(bindings, Mapping):
        raise FlowNodeConfigurationError(
            code="flow.invalid_arguments",
            path=path,
            message="Tool node argument bindings are invalid.",
            hint="Use a TOML table mapping parameter names to selectors.",
        )

    parameters = _tool_parameter_names(descriptor)
    required = _tool_required_parameters(descriptor)
    for name, selector in bindings.items():
        if not isinstance(name, str) or name not in parameters:
            raise FlowNodeConfigurationError(
                code="flow.unknown_argument_binding",
                path=f"{path}.{name}",
                message="Tool node argument binding is unknown.",
                hint="Bind only declared tool parameters.",
            )
        if not isinstance(selector, str) or not selector.strip():
            raise FlowNodeConfigurationError(
                code="flow.invalid_argument_selector",
                path=f"{path}.{name}",
                message="Tool node argument selector is invalid.",
                hint="Use a non-empty dotted input selector.",
            )
    missing = sorted(required - set(bindings))
    if missing:
        raise FlowNodeConfigurationError(
            code="flow.missing_argument_binding",
            path=f"{path}.{missing[0]}",
            message="Tool node argument binding is missing.",
            hint="Bind every required tool parameter.",
        )


def _validate_tool_node_output_mode(definition: FlowNodeDefinition) -> None:
    mode = definition.config.get("output_mode", "raw")
    if mode in {"raw", "envelope"}:
        return
    raise FlowNodeConfigurationError(
        code="flow.invalid_output_mode",
        path=f"nodes.{definition.name}.config.output_mode",
        message="Tool node output mode is invalid.",
        hint="Use raw or envelope.",
    )


def _tool_node_output(
    definition: FlowNodeDefinition,
    outcome: ToolCallOutcome,
) -> object:
    if (
        isinstance(outcome, ToolCallDiagnostic)
        and outcome.code is ToolCallDiagnosticCode.CANCELLED
    ):
        raise CancelledError()
    mode = cast(str, definition.config.get("output_mode", "raw"))
    if mode == "envelope":
        return _tool_node_envelope(outcome)
    if isinstance(outcome, ToolCallResult):
        return outcome.result
    if isinstance(outcome, ToolCallDiagnostic):
        if outcome.stage is ToolCallDiagnosticStage.VALIDATE:
            raise FlowNodeConfigurationError(
                code="flow.invalid_arguments",
                path=f"nodes.{definition.name}.config.arguments",
                message="Tool node arguments are invalid.",
                hint="Provide arguments matching the tool parameter schema.",
            )
        raise FlowNodeConfigurationError(
            code="flow.tool_diagnostic",
            path=f"nodes.{definition.name}",
            message="Tool node did not execute.",
            hint="Use envelope output mode to compose diagnostic outcomes.",
        )
    assert isinstance(outcome, ToolCallError)
    raise RuntimeError(
        f"Tool node execution failed: {outcome.error_type}: {outcome.message}"
    )


def _tool_node_envelope(outcome: ToolCallOutcome) -> dict[str, object]:
    if isinstance(outcome, ToolCallResult):
        return {
            "status": "result",
            "call_id": outcome.call.id,
            "canonical_name": outcome.name,
            "result": outcome.result,
            "error": None,
            "diagnostic": None,
        }
    if isinstance(outcome, ToolCallError):
        return {
            "status": "error",
            "call_id": outcome.call.id,
            "canonical_name": outcome.name,
            "result": None,
            "error": tool_call_error_payload(outcome),
            "diagnostic": None,
        }
    return {
        "status": "diagnostic",
        "call_id": outcome.call_id,
        "canonical_name": outcome.canonical_name,
        "result": None,
        "error": None,
        "diagnostic": tool_call_diagnostic_payload(outcome),
    }


def _tool_node_arguments(
    definition: FlowNodeDefinition,
    descriptor: ToolDescriptor,
    inputs: Mapping[str, object],
) -> dict[str, object]:
    bindings = definition.config.get("arguments")
    source = _node_input_value(definition, inputs)
    if bindings is not None:
        assert isinstance(bindings, Mapping)
        return {
            str(name): _select_argument(
                source, str(selector), definition, name
            )
            for name, selector in bindings.items()
            if isinstance(name, str) and isinstance(selector, str)
        }

    properties = _tool_parameter_properties(descriptor)
    parameters = (
        frozenset(properties) if properties is not None else frozenset()
    )
    if properties is not None and not parameters:
        return {}

    if isinstance(source, Mapping):
        matched = {
            name: _copy_flow_value(source[name])
            for name in sorted(parameters)
            if name in source
        }
        if matched:
            return matched

    if len(parameters) == 1:
        parameter = next(iter(parameters))
        return {parameter: source}

    raise FlowNodeConfigurationError(
        code="flow.ambiguous_argument_binding",
        path=f"nodes.{definition.name}.config.arguments",
        message="Tool node arguments are ambiguous.",
        hint="Add explicit argument selectors for this tool node.",
    )


def _select_argument(
    source: object,
    selector: str,
    definition: FlowNodeDefinition,
    name: object,
) -> object:
    try:
        return _select_path(source, selector)
    except (IndexError, KeyError, ValueError):
        raise FlowNodeConfigurationError(
            code="flow.unresolved_argument_selector",
            path=f"nodes.{definition.name}.config.arguments.{name}",
            message="Tool node argument selector cannot be resolved.",
            hint="Reference a value available on the node input.",
        ) from None


def _tool_parameter_names(descriptor: ToolDescriptor) -> frozenset[str]:
    properties = _tool_parameter_properties(descriptor)
    if properties is None:
        return frozenset()
    return frozenset(properties)


def _tool_required_parameters(descriptor: ToolDescriptor) -> frozenset[str]:
    schema = descriptor.parameter_schema
    if schema is None:
        return frozenset()
    required = schema.get("required", [])
    if not isinstance(required, list | tuple):
        return frozenset()
    return frozenset(name for name in required if isinstance(name, str))


def _tool_parameter_properties(
    descriptor: ToolDescriptor,
) -> Mapping[str, object] | None:
    schema = descriptor.parameter_schema
    if schema is None:
        return None
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return None
    return properties
