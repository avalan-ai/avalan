from .condition import FlowCondition

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

FlowMetadata: TypeAlias = Mapping[str, object]


class FlowInputType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    FILE_ARRAY = "file[]"


class FlowOutputType(StrEnum):
    TEXT = "text"
    JSON = "json"
    OBJECT = "object"
    ARRAY = "array"
    FILE = "file"
    FILE_ARRAY = "file[]"


class FlowEntryBehaviorType(StrEnum):
    NODE = "node"


class FlowOutputBehaviorType(StrEnum):
    MAP = "map"


class FlowNodeKind(StrEnum):
    INPUT = "input"
    CONSTANT = "constant"
    PASS_THROUGH = "pass-through"
    SELECT = "select"
    VALIDATION = "validation"
    DECISION = "decision"
    JOIN = "join"
    AGENT = "agent"
    TOOL = "tool"
    FILE_CONVERSION = "file_conversion"
    HUMAN_REVIEW = "human_review"
    NOTIFICATION = "notification"
    SUBFLOW = "subflow"


class FlowNodeCapability(StrEnum):
    DIRECT_ASYNC = "direct_async"
    ASYNC_ONLY = "async_only"
    TASK_BACKED = "task_backed"
    DURABLE_PAUSE = "durable_pause"


class FlowMappingKind(StrEnum):
    SELECT = "select"
    RENAME = "rename"
    OBJECT = "object"
    ARRAY = "array"
    MERGE = "merge"
    FILE = "file"
    FILE_ARRAY = "file[]"


class FlowEdgeKind(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    FINALLY = "finally"
    CANCELLATION = "cancellation"
    PAUSE = "pause"
    RESUME = "resume"


class FlowRouteMatchPolicy(StrEnum):
    EXCLUSIVE = "exclusive"
    ALL_MATCHING = "all_matching"


class FlowJoinPolicyType(StrEnum):
    ALL_SUCCESS = "all_success"
    ALL_DONE = "all_done"
    ANY_SUCCESS = "any_success"
    QUORUM = "quorum"
    FIRST_SUCCESS = "first_success"
    FAIL_FAST = "fail_fast"
    COLLECT = "collect"


def _empty_mapping() -> FlowMetadata:
    return MappingProxyType({})


def _empty_string_mapping() -> Mapping[str, str]:
    return MappingProxyType({})


def _freeze_mapping(value: FlowMetadata) -> FlowMetadata:
    assert isinstance(value, Mapping), "metadata must be a mapping"
    frozen: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str), "metadata keys must be strings"
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


def _assert_contract_tuple(
    values: tuple["FlowNodeContract", ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(value, FlowNodeContract)


def _assert_capability_tuple(
    values: tuple["FlowNodeCapability", ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(value, FlowNodeCapability)


def _freeze_string_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    assert isinstance(value, Mapping), "mapping must be a mapping"
    frozen: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, "mapping key")
        _assert_non_empty_string(item, f"mapping.{key}")
        frozen[key] = item
    return MappingProxyType(frozen)


def _assert_mapping_tuple(
    values: tuple["FlowInputMapping", ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(value, FlowInputMapping)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowInputDefinition:
    name: str
    type: FlowInputType
    mime_types: tuple[str, ...] = ()
    schema: FlowMetadata | None = None
    schema_ref: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        assert isinstance(self.type, FlowInputType)
        _assert_string_tuple(self.mime_types, "mime_types")
        if self.schema is not None:
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        if self.schema_ref is not None:
            _assert_non_empty_string(self.schema_ref, "schema_ref")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowOutputDefinition:
    name: str
    type: FlowOutputType
    schema: FlowMetadata | None = None
    schema_ref: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        assert isinstance(self.type, FlowOutputType)
        if self.schema is not None:
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        if self.schema_ref is not None:
            _assert_non_empty_string(self.schema_ref, "schema_ref")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEntryBehavior:
    node: str
    type: FlowEntryBehaviorType = FlowEntryBehaviorType.NODE

    def __post_init__(self) -> None:
        assert isinstance(self.type, FlowEntryBehaviorType)
        _assert_non_empty_string(self.node, "node")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowOutputBehavior:
    outputs: Mapping[str, str]
    type: FlowOutputBehaviorType = FlowOutputBehaviorType.MAP

    def __post_init__(self) -> None:
        assert isinstance(self.type, FlowOutputBehaviorType)
        object.__setattr__(
            self, "outputs", _freeze_string_mapping(self.outputs)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeContract:
    name: str | None = None
    type: FlowInputType | FlowOutputType | str | None = None
    schema: FlowMetadata | None = None
    schema_ref: str | None = None
    metadata: FlowMetadata = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.name is not None:
            _assert_non_empty_string(self.name, "name")
        if self.type is not None:
            assert isinstance(self.type, FlowInputType | FlowOutputType | str)
            if isinstance(self.type, str):
                _assert_non_empty_string(self.type, "type")
        if self.schema is not None:
            object.__setattr__(self, "schema", _freeze_mapping(self.schema))
        if self.schema_ref is not None:
            _assert_non_empty_string(self.schema_ref, "schema_ref")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowInputMapping:
    target: str
    kind: FlowMappingKind = FlowMappingKind.SELECT
    source: str | None = None
    sources: tuple[str, ...] = ()
    fields: Mapping[str, str] = field(default_factory=_empty_string_mapping)
    items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.target, "target")
        assert isinstance(self.kind, FlowMappingKind)
        if self.source is not None:
            _assert_non_empty_string(self.source, "source")
        _assert_string_tuple(self.sources, "sources")
        object.__setattr__(self, "fields", _freeze_string_mapping(self.fields))
        _assert_string_tuple(self.items, "items")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowJoinPolicy:
    type: FlowJoinPolicyType
    quorum: int | None = None
    optional_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.type, FlowJoinPolicyType)
        if self.quorum is not None:
            assert isinstance(self.quorum, int) and not isinstance(
                self.quorum,
                bool,
            )
        _assert_string_tuple(self.optional_inputs, "optional_inputs")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeMetadata:
    kind: FlowNodeKind | None = None
    supports_ref: bool = False
    async_only: bool = False
    input_contract: FlowNodeContract | None = None
    output_contract: FlowNodeContract | None = None
    input_contracts: tuple[FlowNodeContract, ...] = ()
    output_contracts: tuple[FlowNodeContract, ...] = ()
    capabilities: tuple[FlowNodeCapability, ...] = ()
    requires_ref: bool = False
    required_config_keys: tuple[str, ...] = ()
    metadata: FlowMetadata = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.kind is not None:
            assert isinstance(self.kind, FlowNodeKind)
        assert isinstance(self.supports_ref, bool)
        assert isinstance(self.async_only, bool)
        if self.input_contract is not None:
            assert isinstance(self.input_contract, FlowNodeContract)
        if self.output_contract is not None:
            assert isinstance(self.output_contract, FlowNodeContract)
        _assert_contract_tuple(self.input_contracts, "input_contracts")
        _assert_contract_tuple(self.output_contracts, "output_contracts")
        _assert_capability_tuple(self.capabilities, "capabilities")
        assert isinstance(self.requires_ref, bool)
        _assert_string_tuple(self.required_config_keys, "required_config_keys")
        input_contracts = self.input_contracts
        if self.input_contract is not None and not input_contracts:
            input_contracts = (self.input_contract,)
        output_contracts = self.output_contracts
        if self.output_contract is not None and not output_contracts:
            output_contracts = (self.output_contract,)
        capabilities = self.capabilities
        if (
            self.async_only
            and FlowNodeCapability.ASYNC_ONLY not in capabilities
        ):
            capabilities = capabilities + (FlowNodeCapability.ASYNC_ONLY,)
        if (
            not self.async_only
            and FlowNodeCapability.DIRECT_ASYNC not in capabilities
        ):
            capabilities = capabilities + (FlowNodeCapability.DIRECT_ASYNC,)
        object.__setattr__(self, "input_contracts", input_contracts)
        object.__setattr__(self, "output_contracts", output_contracts)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeDefinition:
    name: str
    type: str
    ref: str | None = None
    input: str | None = None
    output: str | None = None
    join_policy: FlowJoinPolicy | None = None
    mappings: tuple[FlowInputMapping, ...] = ()
    config: FlowMetadata = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        _assert_non_empty_string(self.type, "type")
        if self.ref is not None:
            _assert_non_empty_string(self.ref, "ref")
        if self.input is not None:
            _assert_non_empty_string(self.input, "input")
        if self.output is not None:
            _assert_non_empty_string(self.output, "output")
        if self.join_policy is not None:
            assert isinstance(self.join_policy, FlowJoinPolicy)
        _assert_mapping_tuple(self.mappings, "mappings")
        object.__setattr__(self, "config", _freeze_mapping(self.config))


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEdgeDefinition:
    source: str
    target: str
    label: str | None = None
    kind: FlowEdgeKind = FlowEdgeKind.SUCCESS
    condition: FlowCondition | None = None
    priority: int = 0
    default: bool = False
    routing_policy: FlowRouteMatchPolicy = FlowRouteMatchPolicy.EXCLUSIVE

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.source, "source")
        _assert_non_empty_string(self.target, "target")
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        assert isinstance(self.kind, FlowEdgeKind)
        if self.condition is not None:
            assert isinstance(self.condition, FlowCondition)
        assert isinstance(self.priority, int) and not isinstance(
            self.priority,
            bool,
        )
        assert isinstance(self.default, bool)
        assert isinstance(self.routing_policy, FlowRouteMatchPolicy)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowDefinition:
    name: str
    nodes: tuple[FlowNodeDefinition, ...]
    entrypoint: str | None = None
    output_node: str | None = None
    version: str | None = None
    revision: str | None = None
    description: str | None = None
    input: FlowInputDefinition | None = None
    inputs: tuple[FlowInputDefinition, ...] = ()
    output: FlowOutputDefinition | None = None
    outputs: tuple[FlowOutputDefinition, ...] = ()
    entry_behavior: FlowEntryBehavior | None = None
    output_behavior: FlowOutputBehavior | None = None
    runtime_limits: FlowMetadata = field(default_factory=_empty_mapping)
    privacy_policy: FlowMetadata = field(default_factory=_empty_mapping)
    observability_policy: FlowMetadata = field(default_factory=_empty_mapping)
    tags: tuple[str, ...] = ()
    ownership: FlowMetadata = field(default_factory=_empty_mapping)
    variables: FlowMetadata = field(default_factory=_empty_mapping)
    edges: tuple[FlowEdgeDefinition, ...] = ()
    definition_base: Path | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        if self.entrypoint is not None:
            _assert_non_empty_string(self.entrypoint, "entrypoint")
        if self.output_node is not None:
            _assert_non_empty_string(self.output_node, "output_node")
        if self.version is not None:
            _assert_non_empty_string(self.version, "version")
        if self.revision is not None:
            _assert_non_empty_string(self.revision, "revision")
        if self.description is not None:
            _assert_non_empty_string(self.description, "description")
        if self.input is not None:
            assert isinstance(self.input, FlowInputDefinition)
        assert isinstance(self.inputs, tuple)
        for input_definition in self.inputs:
            assert isinstance(input_definition, FlowInputDefinition)
        if self.output is not None:
            assert isinstance(self.output, FlowOutputDefinition)
        assert isinstance(self.outputs, tuple)
        for output_definition in self.outputs:
            assert isinstance(output_definition, FlowOutputDefinition)
        if self.entry_behavior is not None:
            assert isinstance(self.entry_behavior, FlowEntryBehavior)
        if self.output_behavior is not None:
            assert isinstance(self.output_behavior, FlowOutputBehavior)
        object.__setattr__(
            self, "runtime_limits", _freeze_mapping(self.runtime_limits)
        )
        object.__setattr__(
            self, "privacy_policy", _freeze_mapping(self.privacy_policy)
        )
        object.__setattr__(
            self,
            "observability_policy",
            _freeze_mapping(self.observability_policy),
        )
        _assert_string_tuple(self.tags, "tags")
        object.__setattr__(self, "ownership", _freeze_mapping(self.ownership))
        object.__setattr__(self, "variables", _freeze_mapping(self.variables))
        assert isinstance(self.nodes, tuple)
        for node in self.nodes:
            assert isinstance(node, FlowNodeDefinition)
        assert isinstance(self.edges, tuple)
        for edge in self.edges:
            assert isinstance(edge, FlowEdgeDefinition)
        if self.definition_base is not None:
            assert isinstance(self.definition_base, Path)

    @property
    def node_map(self) -> Mapping[str, FlowNodeDefinition]:
        return MappingProxyType({node.name: node for node in self.nodes})

    @property
    def is_strict(self) -> bool:
        return bool(
            self.revision is not None
            or self.inputs
            or self.outputs
            or self.entry_behavior is not None
            or self.output_behavior is not None
            or self.runtime_limits
            or self.privacy_policy
            or self.observability_policy
            or self.tags
            or self.ownership
        )
