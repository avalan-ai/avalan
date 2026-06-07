from .condition import (
    FlowCondition,
    FlowConditionOperator,
    FlowConditionValueType,
)
from .definition import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowInputDefinition,
    FlowInputMapping,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowOutputDefinition,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowRouteMatchPolicy,
    FlowTimeoutPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .registry import FlowNodeRegistry, default_flow_node_registry
from .selector import FlowSelector, parse_flow_selector
from .validator import validate_flow_definition

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _empty_selector_mapping() -> Mapping[str, FlowSelector]:
    return MappingProxyType({})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            assert isinstance(key, str) and key.strip()
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return _freeze_value(value)  # type: ignore[return-value]


def _freeze_selector_mapping(
    value: Mapping[str, FlowSelector],
) -> Mapping[str, FlowSelector]:
    assert isinstance(value, Mapping)
    frozen: dict[str, FlowSelector] = {}
    for key, item in value.items():
        assert isinstance(key, str) and key.strip()
        assert isinstance(item, FlowSelector)
        frozen[key] = item
    return MappingProxyType(frozen)


def _assert_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowConditionPlan:
    operator: FlowConditionOperator
    selector: FlowSelector | None = None
    value: object | None = None
    value_selector: FlowSelector | None = None
    values: tuple[object, ...] = ()
    value_type: FlowConditionValueType | None = None
    conditions: tuple["FlowConditionPlan", ...] = ()
    condition: "FlowConditionPlan | None" = None

    def __post_init__(self) -> None:
        assert isinstance(self.operator, FlowConditionOperator)
        if self.selector is not None:
            assert isinstance(self.selector, FlowSelector)
        if self.value is not None:
            object.__setattr__(self, "value", _freeze_value(self.value))
        if self.value_selector is not None:
            assert isinstance(self.value_selector, FlowSelector)
        object.__setattr__(
            self,
            "values",
            tuple(_freeze_value(value) for value in self.values),
        )
        if self.value_type is not None:
            assert isinstance(self.value_type, FlowConditionValueType)
        for condition in self.conditions:
            assert isinstance(condition, FlowConditionPlan)
        if self.condition is not None:
            assert isinstance(self.condition, FlowConditionPlan)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowMappingPlan:
    target: str
    kind: FlowMappingKind
    source: FlowSelector | None = None
    sources: tuple[FlowSelector, ...] = ()
    fields: Mapping[str, FlowSelector] = field(
        default_factory=_empty_selector_mapping
    )
    items: tuple[FlowSelector, ...] = ()

    def __post_init__(self) -> None:
        _assert_string(self.target, "target")
        assert isinstance(self.kind, FlowMappingKind)
        if self.source is not None:
            assert isinstance(self.source, FlowSelector)
        for source in self.sources:
            assert isinstance(source, FlowSelector)
        object.__setattr__(
            self,
            "fields",
            _freeze_selector_mapping(self.fields),
        )
        for item in self.items:
            assert isinstance(item, FlowSelector)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowJoinPlan:
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
        for name in self.optional_inputs:
            _assert_string(name, "optional_inputs")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowRetryPlan:
    max_attempts: int
    backoff: FlowRetryBackoffStrategy = FlowRetryBackoffStrategy.NONE
    initial_delay_seconds: int | float | None = None
    max_delay_seconds: int | float | None = None
    retryable_categories: tuple[str, ...] = ()
    non_retryable_categories: tuple[str, ...] = ()
    exhausted_route: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.max_attempts, int) and not isinstance(
            self.max_attempts,
            bool,
        )
        assert self.max_attempts > 0
        assert isinstance(self.backoff, FlowRetryBackoffStrategy)
        if self.initial_delay_seconds is not None:
            assert isinstance(
                self.initial_delay_seconds, int | float
            ) and not isinstance(self.initial_delay_seconds, bool)
        if self.max_delay_seconds is not None:
            assert isinstance(
                self.max_delay_seconds, int | float
            ) and not isinstance(self.max_delay_seconds, bool)
        for category in self.retryable_categories:
            _assert_string(category, "retryable_categories")
        for category in self.non_retryable_categories:
            _assert_string(category, "non_retryable_categories")
        if self.exhausted_route is not None:
            _assert_string(self.exhausted_route, "exhausted_route")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowTimeoutPlan:
    per_attempt_seconds: int | float

    def __post_init__(self) -> None:
        assert isinstance(self.per_attempt_seconds, int | float)
        assert not isinstance(self.per_attempt_seconds, bool)
        assert self.per_attempt_seconds > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowLoopPlan:
    max_iterations: int | None = None
    max_elapsed_seconds: int | float | None = None
    continue_condition: FlowConditionPlan
    exit_condition: FlowConditionPlan
    output_selector: FlowSelector
    limit_route: str

    def __post_init__(self) -> None:
        if self.max_iterations is not None:
            assert isinstance(self.max_iterations, int) and not isinstance(
                self.max_iterations,
                bool,
            )
            assert self.max_iterations > 0
        if self.max_elapsed_seconds is not None:
            assert isinstance(
                self.max_elapsed_seconds, int | float
            ) and not isinstance(self.max_elapsed_seconds, bool)
            assert self.max_elapsed_seconds > 0
        assert isinstance(self.continue_condition, FlowConditionPlan)
        assert isinstance(self.exit_condition, FlowConditionPlan)
        assert isinstance(self.output_selector, FlowSelector)
        _assert_string(self.limit_route, "limit_route")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodePlan:
    name: str
    type: str
    kind: FlowNodeKind
    ref: str | None = None
    input_contracts: tuple[FlowNodeContract, ...] = ()
    output_contracts: tuple[FlowNodeContract, ...] = ()
    capabilities: tuple[FlowNodeCapability, ...] = ()
    mappings: tuple[FlowMappingPlan, ...] = ()
    join: FlowJoinPlan | None = None
    retry: FlowRetryPlan | None = None
    timeout: FlowTimeoutPlan | None = None
    loop: FlowLoopPlan | None = None
    config: Mapping[str, object] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        _assert_string(self.name, "name")
        _assert_string(self.type, "type")
        assert isinstance(self.kind, FlowNodeKind)
        if self.ref is not None:
            _assert_string(self.ref, "ref")
        for contract in self.input_contracts:
            assert isinstance(contract, FlowNodeContract)
        for contract in self.output_contracts:
            assert isinstance(contract, FlowNodeContract)
        for capability in self.capabilities:
            assert isinstance(capability, FlowNodeCapability)
        for mapping in self.mappings:
            assert isinstance(mapping, FlowMappingPlan)
        if self.join is not None:
            assert isinstance(self.join, FlowJoinPlan)
        if self.retry is not None:
            assert isinstance(self.retry, FlowRetryPlan)
        if self.timeout is not None:
            assert isinstance(self.timeout, FlowTimeoutPlan)
        if self.loop is not None:
            assert isinstance(self.loop, FlowLoopPlan)
        object.__setattr__(self, "config", _freeze_mapping(self.config))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEdgePlan:
    index: int
    source: str
    target: str
    kind: FlowEdgeKind
    label: str | None = None
    condition: FlowConditionPlan | None = None
    priority: int = 0
    default: bool = False
    routing_policy: FlowRouteMatchPolicy = FlowRouteMatchPolicy.EXCLUSIVE

    def __post_init__(self) -> None:
        assert isinstance(self.index, int) and self.index >= 0
        _assert_string(self.source, "source")
        _assert_string(self.target, "target")
        assert isinstance(self.kind, FlowEdgeKind)
        if self.label is not None:
            _assert_string(self.label, "label")
        if self.condition is not None:
            assert isinstance(self.condition, FlowConditionPlan)
        assert isinstance(self.priority, int) and not isinstance(
            self.priority,
            bool,
        )
        assert isinstance(self.default, bool)
        assert isinstance(self.routing_policy, FlowRouteMatchPolicy)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowExecutionPlan:
    name: str
    version: str | None
    revision: str | None
    inputs: tuple[FlowInputDefinition, ...]
    outputs: tuple[FlowOutputDefinition, ...]
    entry_node: str
    output_selectors: Mapping[str, FlowSelector]
    nodes: tuple[FlowNodePlan, ...]
    edges: tuple[FlowEdgePlan, ...] = ()

    def __post_init__(self) -> None:
        _assert_string(self.name, "name")
        if self.version is not None:
            _assert_string(self.version, "version")
        if self.revision is not None:
            _assert_string(self.revision, "revision")
        for input_definition in self.inputs:
            assert isinstance(input_definition, FlowInputDefinition)
        for output_definition in self.outputs:
            assert isinstance(output_definition, FlowOutputDefinition)
        _assert_string(self.entry_node, "entry_node")
        object.__setattr__(
            self,
            "output_selectors",
            _freeze_selector_mapping(self.output_selectors),
        )
        for node in self.nodes:
            assert isinstance(node, FlowNodePlan)
        for edge in self.edges:
            assert isinstance(edge, FlowEdgePlan)

    @property
    def node_map(self) -> Mapping[str, FlowNodePlan]:
        return MappingProxyType({node.name: node for node in self.nodes})

    @property
    def edges_by_source(self) -> Mapping[str, tuple[FlowEdgePlan, ...]]:
        grouped: dict[str, list[FlowEdgePlan]] = {}
        for edge in self.edges:
            grouped.setdefault(edge.source, []).append(edge)
        return MappingProxyType(
            {source: tuple(edges) for source, edges in sorted(grouped.items())}
        )

    @property
    def edges_by_target(self) -> Mapping[str, tuple[FlowEdgePlan, ...]]:
        grouped: dict[str, list[FlowEdgePlan]] = {}
        for edge in self.edges:
            grouped.setdefault(edge.target, []).append(edge)
        return MappingProxyType(
            {target: tuple(edges) for target, edges in sorted(grouped.items())}
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowPlanCompileResult:
    plan: FlowExecutionPlan | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.plan is not None:
            assert isinstance(self.plan, FlowExecutionPlan)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.diagnostics

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


def compile_flow_definition(
    definition: FlowDefinition,
    registry: FlowNodeRegistry | None = None,
) -> FlowPlanCompileResult:
    assert isinstance(definition, FlowDefinition)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    node_registry = registry or default_flow_node_registry()
    validation = validate_flow_definition(definition, node_registry)
    if not validation.ok:
        return FlowPlanCompileResult(diagnostics=validation.diagnostics)
    if not definition.is_strict:
        return FlowPlanCompileResult(
            diagnostics=(
                FlowDiagnostic(
                    code="flow.execution.plan_requires_strict_definition",
                    category=FlowDiagnosticCategory.EXECUTION,
                    severity=FlowDiagnosticSeverity.ERROR,
                    path="flow",
                    message=(
                        "Flow execution plans require declared inputs, "
                        "outputs, entry behavior, and output behavior."
                    ),
                    hint="Use a strict flow definition before compiling.",
                ),
            )
        )
    assert definition.entry_behavior is not None
    assert definition.output_behavior is not None
    return FlowPlanCompileResult(
        plan=FlowExecutionPlan(
            name=definition.name,
            version=definition.version,
            revision=definition.revision,
            inputs=definition.inputs,
            outputs=definition.outputs,
            entry_node=definition.entry_behavior.node,
            output_selectors={
                name: parse_flow_selector(selector)
                for name, selector in (
                    definition.output_behavior.outputs.items()
                )
            },
            nodes=tuple(
                _compile_node(node, node_registry) for node in definition.nodes
            ),
            edges=tuple(
                _compile_edge(index, edge)
                for index, edge in enumerate(definition.edges)
            ),
        )
    )


def _compile_node(
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
) -> FlowNodePlan:
    metadata = registry.metadata(node.type)
    assert metadata is not None
    assert metadata.kind is not None
    return FlowNodePlan(
        name=node.name,
        type=node.type,
        kind=metadata.kind,
        ref=node.ref,
        input_contracts=metadata.input_contracts,
        output_contracts=metadata.output_contracts,
        capabilities=metadata.capabilities,
        mappings=tuple(_compile_mapping(mapping) for mapping in node.mappings),
        join=_compile_join(node.join_policy),
        retry=_compile_retry(node.retry_policy),
        timeout=_compile_timeout(node.timeout_policy),
        loop=_compile_loop(node.loop_policy),
        config=node.config,
        metadata=_compile_node_metadata(node, registry, metadata.metadata),
    )


def _compile_node_metadata(
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    compiled = dict(metadata)
    if registry.supports_tool_resolution(node.type):
        descriptor = registry.tool_descriptor(node)
        tool_metadata: dict[str, object] = {
            "canonical_name": descriptor.name,
            "aliases": tuple(descriptor.aliases),
        }
        if descriptor.parameter_schema is not None:
            tool_metadata["parameter_schema"] = descriptor.parameter_schema
        if descriptor.return_schema is not None:
            tool_metadata["return_schema"] = descriptor.return_schema
        compiled["tool"] = tool_metadata
    return compiled


def _compile_mapping(mapping: FlowInputMapping) -> FlowMappingPlan:
    return FlowMappingPlan(
        target=mapping.target,
        kind=mapping.kind,
        source=(
            parse_flow_selector(mapping.source)
            if mapping.source is not None
            else None
        ),
        sources=tuple(
            parse_flow_selector(source) for source in mapping.sources
        ),
        fields={
            name: parse_flow_selector(selector)
            for name, selector in mapping.fields.items()
        },
        items=tuple(parse_flow_selector(item) for item in mapping.items),
    )


def _compile_join(policy: FlowJoinPolicy | None) -> FlowJoinPlan | None:
    if policy is None:
        return None
    return FlowJoinPlan(
        type=policy.type,
        quorum=policy.quorum,
        optional_inputs=policy.optional_inputs,
    )


def _compile_retry(policy: FlowRetryPolicy | None) -> FlowRetryPlan | None:
    if policy is None:
        return None
    assert policy.max_attempts is not None
    return FlowRetryPlan(
        max_attempts=policy.max_attempts,
        backoff=policy.backoff,
        initial_delay_seconds=policy.initial_delay_seconds,
        max_delay_seconds=policy.max_delay_seconds,
        retryable_categories=policy.retryable_categories,
        non_retryable_categories=policy.non_retryable_categories,
        exhausted_route=policy.exhausted_route,
    )


def _compile_timeout(
    policy: FlowTimeoutPolicy | None,
) -> FlowTimeoutPlan | None:
    if policy is None:
        return None
    assert policy.per_attempt_seconds is not None
    return FlowTimeoutPlan(per_attempt_seconds=policy.per_attempt_seconds)


def _compile_loop(policy: FlowLoopPolicy | None) -> FlowLoopPlan | None:
    if policy is None:
        return None
    assert policy.continue_condition is not None
    assert policy.exit_condition is not None
    assert policy.output_selector is not None
    assert policy.limit_route is not None
    return FlowLoopPlan(
        max_iterations=policy.max_iterations,
        max_elapsed_seconds=policy.max_elapsed_seconds,
        continue_condition=_compile_condition(policy.continue_condition),
        exit_condition=_compile_condition(policy.exit_condition),
        output_selector=parse_flow_selector(policy.output_selector),
        limit_route=policy.limit_route,
    )


def _compile_edge(index: int, edge: FlowEdgeDefinition) -> FlowEdgePlan:
    return FlowEdgePlan(
        index=index,
        source=edge.source,
        target=edge.target,
        label=edge.label,
        kind=edge.kind,
        condition=(
            _compile_condition(edge.condition)
            if edge.condition is not None
            else None
        ),
        priority=edge.priority,
        default=edge.default,
        routing_policy=edge.routing_policy,
    )


def _compile_condition(condition: FlowCondition) -> FlowConditionPlan:
    return FlowConditionPlan(
        operator=condition.operator,
        selector=(
            parse_flow_selector(condition.selector)
            if condition.selector is not None
            else None
        ),
        value=condition.value,
        value_selector=(
            parse_flow_selector(condition.value_selector)
            if condition.value_selector is not None
            else None
        ),
        values=condition.values,
        value_type=condition.value_type,
        conditions=tuple(
            _compile_condition(child) for child in condition.conditions
        ),
        condition=(
            _compile_condition(condition.condition)
            if condition.condition is not None
            else None
        ),
    )
