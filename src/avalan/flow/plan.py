from ..container import (
    ContainerAuthorityCaps,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerNormalizedRuntimeEnvelopePlan,
    ContainerPlanRequest,
    ContainerPlanRequestKind,
    ContainerRuntimeEnvelopeKind,
    ContainerSettings,
    normalize_runtime_envelope_plan,
)
from ..isolation import (
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationSettingsSurface,
    trusted_isolation_source,
)
from ..skill import (
    CANONICAL_SKILLS_TOOL_NAMES,
    SKILL_SETTINGS_POLICY_VERSION,
    SkillDiagnosticInfo,
    SkillModelValue,
    SkillRegistry,
    SkillSourceAuthority,
    SkillStatus,
    TrustedSkillSettings,
    merge_skill_settings,
    trusted_skill_settings_fingerprint,
    trusted_skill_settings_identity_dict,
    trusted_skill_source_fingerprint,
    trusted_skill_source_identity_dict,
)
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
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    default_flow_node_registry,
)
from .selector import FlowSelector, parse_flow_selector
from .validator import validate_flow_definition

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from json import dumps
from types import MappingProxyType
from typing import cast

FLOW_RESUME_ISOLATION_METADATA_KEY = "isolation"
FLOW_RESUME_SKILLS_METADATA_KEY = "skills"
_FLOW_RESUME_ISOLATION_METADATA_VERSION = "phase9"
_FLOW_RESUME_SKILLS_METADATA_VERSION = "phase9"


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


def _assert_optional_string(value: str | None, field_name: str) -> None:
    if value is not None:
        _assert_string(value, field_name)


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
class FlowIsolationMetadata:
    mode: IsolationMode | str
    backend: str | None
    profile_registry_id: str | None
    profile_name: str | None
    policy_version: str | None
    scope: str | None
    plan_fingerprint: str
    container_fingerprint: str | None = None

    def __post_init__(self) -> None:
        mode = IsolationMode(self.mode)
        _assert_optional_string(self.backend, "backend")
        _assert_optional_string(
            self.profile_registry_id,
            "profile_registry_id",
        )
        _assert_optional_string(self.profile_name, "profile_name")
        _assert_optional_string(self.policy_version, "policy_version")
        _assert_optional_string(self.scope, "scope")
        _assert_string(self.plan_fingerprint, "plan_fingerprint")
        _assert_optional_string(
            self.container_fingerprint,
            "container_fingerprint",
        )
        object.__setattr__(self, "mode", mode)

    def to_dict(self) -> dict[str, object]:
        mode = cast(IsolationMode, self.mode)
        return {
            "mode": mode.value,
            "backend": self.backend,
            "profile_registry_id": self.profile_registry_id,
            "profile_name": self.profile_name,
            "policy_version": self.policy_version,
            "scope": self.scope,
            "plan_fingerprint": self.plan_fingerprint,
            "container_fingerprint": self.container_fingerprint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowSkillsMetadata:
    settings_fingerprint: str
    source_fingerprint: str
    policy_version: str
    enabled: bool
    bootstrap_enabled: bool
    read_limits: Mapping[str, SkillModelValue]
    index_limits: Mapping[str, SkillModelValue]
    source_limits: Mapping[str, SkillModelValue]
    cursor_limits: Mapping[str, SkillModelValue]
    privacy: Mapping[str, SkillModelValue]
    observability: Mapping[str, SkillModelValue]
    registry_version: str | None = None
    registry_identity: str | None = None
    enabled_tools: tuple[str, ...] = ()
    authority_kinds: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    allowed_skill_ids: tuple[str, ...] = ()
    node: str | None = None
    tool: str | None = None

    def __post_init__(self) -> None:
        _assert_string(self.settings_fingerprint, "settings_fingerprint")
        _assert_string(self.source_fingerprint, "source_fingerprint")
        _assert_string(self.policy_version, "policy_version")
        assert isinstance(self.enabled, bool)
        assert isinstance(self.bootstrap_enabled, bool)
        object.__setattr__(
            self,
            "read_limits",
            _freeze_mapping(self.read_limits),
        )
        object.__setattr__(
            self,
            "index_limits",
            _freeze_mapping(self.index_limits),
        )
        object.__setattr__(
            self,
            "source_limits",
            _freeze_mapping(self.source_limits),
        )
        object.__setattr__(
            self,
            "cursor_limits",
            _freeze_mapping(self.cursor_limits),
        )
        object.__setattr__(self, "privacy", _freeze_mapping(self.privacy))
        object.__setattr__(
            self,
            "observability",
            _freeze_mapping(self.observability),
        )
        _assert_optional_string(self.registry_version, "registry_version")
        _assert_optional_string(self.registry_identity, "registry_identity")
        for tool in self.enabled_tools:
            _assert_string(tool, "enabled_tools")
        for authority_kind in self.authority_kinds:
            _assert_string(authority_kind, "authority_kinds")
        for source_label in self.source_labels:
            _assert_string(source_label, "source_labels")
        for skill_id in self.allowed_skill_ids:
            _assert_string(skill_id, "allowed_skill_ids")
        _assert_optional_string(self.node, "node")
        _assert_optional_string(self.tool, "tool")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "settings_fingerprint": self.settings_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "policy_version": self.policy_version,
            "enabled": self.enabled,
            "bootstrap_enabled": self.bootstrap_enabled,
            "read_limits": self.read_limits,
            "index_limits": self.index_limits,
            "source_limits": self.source_limits,
            "cursor_limits": self.cursor_limits,
            "privacy": self.privacy,
            "observability": self.observability,
            "enabled_tools": self.enabled_tools,
            "authority_kinds": self.authority_kinds,
            "source_labels": self.source_labels,
            "allowed_skill_ids": self.allowed_skill_ids,
        }
        if self.registry_version is not None:
            value["registry_version"] = self.registry_version
        if self.registry_identity is not None:
            value["registry_identity"] = self.registry_identity
        if self.node is not None:
            value["node"] = self.node
        if self.tool is not None:
            value["tool"] = self.tool
        return value


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
    container: ContainerEffectiveSettings | None = None
    isolation: FlowIsolationMetadata | None = None
    skills: TrustedSkillSettings | None = None
    skills_identity: FlowSkillsMetadata | None = None
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
        if self.container is not None:
            assert isinstance(self.container, ContainerEffectiveSettings)
        if self.isolation is not None:
            assert isinstance(self.isolation, FlowIsolationMetadata)
        if self.skills is not None:
            assert isinstance(self.skills, TrustedSkillSettings)
        if self.skills_identity is not None:
            assert isinstance(self.skills_identity, FlowSkillsMetadata)
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
    container: ContainerEffectiveSettings | None = None
    isolation: FlowIsolationMetadata | None = None
    skills: TrustedSkillSettings | None = None
    skills_registry: SkillRegistry | None = None
    skills_identity: FlowSkillsMetadata | None = None
    runtime_envelope: ContainerNormalizedRuntimeEnvelopePlan | None = None

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
        if self.container is not None:
            assert isinstance(self.container, ContainerEffectiveSettings)
        if self.isolation is not None:
            assert isinstance(self.isolation, FlowIsolationMetadata)
        if self.skills is not None:
            assert isinstance(self.skills, TrustedSkillSettings)
        if self.skills_registry is not None:
            assert isinstance(self.skills_registry, SkillRegistry)
        if self.skills_identity is not None:
            assert isinstance(self.skills_identity, FlowSkillsMetadata)
        if self.runtime_envelope is not None:
            assert isinstance(
                self.runtime_envelope,
                ContainerNormalizedRuntimeEnvelopePlan,
            )
        _finalize_flow_isolation_metadata(self)
        _finalize_flow_skills_metadata(self)

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


def flow_resume_isolation_metadata(
    plan: FlowExecutionPlan,
) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    flow_metadata = (
        plan.isolation.to_dict() if plan.isolation is not None else None
    )
    runtime_envelope_metadata = _runtime_envelope_isolation_metadata(
        plan.runtime_envelope
    )
    node_metadata = {
        node.name: node.isolation.to_dict()
        for node in plan.nodes
        if node.isolation is not None
    }
    if (
        flow_metadata is None
        and runtime_envelope_metadata is None
        and not node_metadata
    ):
        return MappingProxyType({})
    return _freeze_mapping(
        {
            FLOW_RESUME_ISOLATION_METADATA_KEY: {
                "version": _FLOW_RESUME_ISOLATION_METADATA_VERSION,
                "flow": flow_metadata,
                "runtime_envelope": runtime_envelope_metadata,
                "nodes": node_metadata,
            }
        }
    )


def flow_resume_skills_metadata(
    plan: FlowExecutionPlan,
) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    flow_metadata = (
        plan.skills_identity.to_dict()
        if plan.skills_identity is not None
        else None
    )
    node_metadata = {
        node.name: node.skills_identity.to_dict()
        for node in plan.nodes
        if node.skills_identity is not None
    }
    if flow_metadata is None and not node_metadata:
        return MappingProxyType({})
    return _freeze_mapping(
        {
            FLOW_RESUME_SKILLS_METADATA_KEY: {
                "version": _FLOW_RESUME_SKILLS_METADATA_VERSION,
                "flow": flow_metadata,
                "nodes": node_metadata,
            }
        }
    )


def flow_resume_metadata(plan: FlowExecutionPlan) -> Mapping[str, object]:
    assert isinstance(plan, FlowExecutionPlan)
    metadata: dict[str, object] = {}
    metadata.update(flow_resume_isolation_metadata(plan))
    metadata.update(flow_resume_skills_metadata(plan))
    return _freeze_mapping(metadata)


def _runtime_envelope_isolation_metadata(
    runtime_envelope: ContainerNormalizedRuntimeEnvelopePlan | None,
) -> dict[str, object] | None:
    if runtime_envelope is None:
        return None
    return {"plan_fingerprint": runtime_envelope.plan_fingerprint}


def _finalize_flow_isolation_metadata(plan: FlowExecutionPlan) -> None:
    expected_plan_isolation = _flow_plan_isolation_metadata(plan)
    if plan.isolation is None:
        object.__setattr__(plan, "isolation", expected_plan_isolation)
    elif expected_plan_isolation is None:
        raise AssertionError("flow plan cannot carry stale isolation metadata")
    else:
        assert (
            plan.isolation.to_dict() == expected_plan_isolation.to_dict()
        ), "flow plan isolation metadata changed"
    finalized_nodes: list[FlowNodePlan] = []
    changed = False
    for node in plan.nodes:
        expected_node_isolation = _flow_node_isolation_metadata(plan, node)
        if node.isolation is None:
            finalized_nodes.append(
                replace(node, isolation=expected_node_isolation)
                if expected_node_isolation is not None
                else node
            )
            changed = changed or expected_node_isolation is not None
            continue
        if expected_node_isolation is None:
            raise AssertionError(
                "flow node cannot carry stale isolation metadata"
            )
        assert (
            node.isolation.to_dict() == expected_node_isolation.to_dict()
        ), "flow node isolation metadata changed"
        finalized_nodes.append(node)
    if changed:
        object.__setattr__(plan, "nodes", tuple(finalized_nodes))


def _finalize_flow_skills_metadata(plan: FlowExecutionPlan) -> None:
    expected_plan_skills = _flow_plan_skills_metadata(plan)
    if plan.skills_identity is None:
        object.__setattr__(plan, "skills_identity", expected_plan_skills)
    elif expected_plan_skills is None:
        raise AssertionError("flow plan cannot carry stale skills metadata")
    else:
        assert (
            plan.skills_identity.to_dict() == expected_plan_skills.to_dict()
        ), "flow plan skills metadata changed"

    finalized_nodes: list[FlowNodePlan] = []
    changed = False
    for node in plan.nodes:
        expected_node_skills = _flow_node_skills_metadata(plan, node)
        if node.skills_identity is None:
            finalized_nodes.append(
                replace(node, skills_identity=expected_node_skills)
                if expected_node_skills is not None
                else node
            )
            changed = changed or expected_node_skills is not None
            continue
        if expected_node_skills is None:
            raise AssertionError(
                "flow node cannot carry stale skills metadata"
            )
        assert (
            node.skills_identity.to_dict() == expected_node_skills.to_dict()
        ), "flow node skills metadata changed"
        finalized_nodes.append(node)
    if changed:
        object.__setattr__(plan, "nodes", tuple(finalized_nodes))


def _flow_plan_skills_metadata(
    plan: FlowExecutionPlan,
) -> FlowSkillsMetadata | None:
    if plan.skills is None:
        return None
    return _skills_metadata(
        plan.skills,
        registry=_plan_skill_registry(plan),
        enabled_tools=_plan_enabled_skills_tools(plan),
    )


def _flow_node_skills_metadata(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
) -> FlowSkillsMetadata | None:
    if node.skills is None:
        return None
    return _skills_metadata(
        node.skills,
        registry=_node_skill_registry(plan, node),
        enabled_tools=_plan_enabled_skills_tools(plan),
        node=node.name,
        tool=_node_tool_name(node),
    )


def _skills_metadata(
    settings: TrustedSkillSettings,
    *,
    registry: SkillRegistry | None,
    enabled_tools: tuple[str, ...],
    node: str | None = None,
    tool: str | None = None,
) -> FlowSkillsMetadata:
    assert isinstance(settings, TrustedSkillSettings)
    if registry is not None:
        assert isinstance(registry, SkillRegistry)
    registry_version = (
        registry.registry_version.as_model_value()
        if registry is not None
        else None
    )
    settings_identity = trusted_skill_settings_identity_dict(settings)
    registry_identity = registry_version or _fingerprint(settings_identity)
    return FlowSkillsMetadata(
        settings_fingerprint=trusted_skill_settings_fingerprint(settings),
        source_fingerprint=trusted_skill_source_fingerprint(settings),
        policy_version=SKILL_SETTINGS_POLICY_VERSION,
        enabled=settings.enabled,
        bootstrap_enabled=settings.bootstrap_enabled,
        read_limits=settings.read_limits.as_model_dict(),
        index_limits=settings.index_limits.as_model_dict(),
        source_limits=settings.source_limits.as_model_dict(),
        cursor_limits=settings.cursor_limits.as_model_dict(),
        privacy=settings.privacy.as_model_dict(),
        observability=settings.observability.as_model_dict(),
        registry_version=registry_version,
        registry_identity=registry_identity,
        enabled_tools=enabled_tools,
        authority_kinds=tuple(
            authority.value for authority in settings.authority_kinds
        ),
        source_labels=tuple(source.label for source in settings.sources),
        allowed_skill_ids=settings.allowed_skill_ids,
        node=node,
        tool=tool,
    )


def _plan_skill_registry(plan: FlowExecutionPlan) -> SkillRegistry | None:
    if plan.skills_registry is not None:
        return plan.skills_registry
    for node in plan.nodes:
        registry = _node_skill_registry(plan, node)
        if registry is not None:
            return registry
    return None


def _node_skill_registry(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
) -> SkillRegistry | None:
    registry = node.metadata.get("skills_registry")
    if isinstance(registry, SkillRegistry):
        return registry
    return plan.skills_registry


def _plan_enabled_skills_tools(
    plan: FlowExecutionPlan,
) -> tuple[str, ...]:
    tool_names: set[str] = set()
    for node in plan.nodes:
        tool_name = _node_tool_name(node)
        if tool_name in CANONICAL_SKILLS_TOOL_NAMES:
            tool_names.add(tool_name)
        for name in _metadata_tool_names(node):
            if name in CANONICAL_SKILLS_TOOL_NAMES:
                tool_names.add(name)
    return tuple(sorted(tool_names))


def _node_tool_name(node: FlowNodePlan) -> str | None:
    tool = node.metadata.get("tool")
    if not isinstance(tool, Mapping):
        return None
    canonical_name = tool.get("canonical_name")
    if isinstance(canonical_name, str) and canonical_name.strip():
        return canonical_name
    return None


def _metadata_tool_names(node: FlowNodePlan) -> tuple[str, ...]:
    tools = node.metadata.get("tools")
    if not isinstance(tools, Mapping):
        return ()
    names: list[str] = []
    for name in tools:
        if isinstance(name, str) and name.strip():
            names.append(name)
    return tuple(names)


def _flow_plan_isolation_metadata(
    plan: FlowExecutionPlan,
) -> FlowIsolationMetadata | None:
    if plan.container is None or not plan.container.enabled:
        return None
    return _container_effective_isolation_metadata(plan.container)


def _flow_node_isolation_metadata(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
) -> FlowIsolationMetadata | None:
    if node.container is None or not node.container.enabled:
        return None
    return _container_effective_isolation_metadata(
        node.container,
        container_fingerprint=flow_node_container_fingerprint(plan, node),
    )


def _container_effective_isolation_metadata(
    settings: ContainerEffectiveSettings,
    *,
    container_fingerprint: str | None = None,
) -> FlowIsolationMetadata:
    isolation = IsolationEffectiveSettings(
        mode=IsolationMode.CONTAINER,
        source=trusted_isolation_source(IsolationSettingsSurface.FLOW_TOML),
        container=settings,
    )
    return FlowIsolationMetadata(
        mode=IsolationMode.CONTAINER,
        backend=_enum_value(settings.backend),
        profile_registry_id=settings.profile_registry_id,
        profile_name=settings.profile_name,
        policy_version=settings.policy_version,
        scope=_enum_value(settings.scope),
        plan_fingerprint=_fingerprint(isolation.canonical_policy_input()),
        container_fingerprint=container_fingerprint,
    )


def _fingerprint(value: Mapping[str, object]) -> str:
    encoded = dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


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


async def compile_flow_definition(
    definition: FlowDefinition,
    registry: FlowNodeRegistry | None = None,
    *,
    skills_settings: TrustedSkillSettings | None = None,
    skills_registry: SkillRegistry | None = None,
) -> FlowPlanCompileResult:
    assert isinstance(definition, FlowDefinition)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    if skills_settings is not None:
        assert isinstance(skills_settings, TrustedSkillSettings)
    if skills_registry is not None:
        assert isinstance(skills_registry, SkillRegistry)
    node_registry = registry or default_flow_node_registry()
    flow_skills = definition.skills or skills_settings
    if flow_skills is not None:
        assert isinstance(flow_skills, TrustedSkillSettings)
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
    try:
        container_defaults = _compile_container_defaults(definition)
        runtime_envelope = _compile_runtime_envelope(
            definition,
            container_defaults,
        )
    except FlowNodeConfigurationError as error:
        return FlowPlanCompileResult(
            diagnostics=(
                FlowDiagnostic(
                    code=error.code,
                    category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
                    severity=FlowDiagnosticSeverity.ERROR,
                    path=error.path,
                    message=error.message,
                    hint=error.hint,
                ),
            )
        )
    nodes: list[FlowNodePlan] = []
    try:
        for node in definition.nodes:
            nodes.append(
                await _compile_node(
                    definition,
                    node,
                    node_registry,
                    container_defaults=container_defaults,
                    flow_skills=flow_skills,
                    skills_registry=skills_registry,
                )
            )
    except FlowNodeConfigurationError as error:
        return FlowPlanCompileResult(
            diagnostics=(
                FlowDiagnostic(
                    code=error.code,
                    category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
                    severity=FlowDiagnosticSeverity.ERROR,
                    path=error.path,
                    message=error.message,
                    hint=error.hint,
                ),
            )
        )
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
            nodes=tuple(nodes),
            edges=tuple(
                _compile_edge(index, edge)
                for index, edge in enumerate(definition.edges)
            ),
            container=container_defaults,
            skills=flow_skills,
            skills_registry=skills_registry,
            runtime_envelope=runtime_envelope,
        )
    )


async def _compile_node(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    *,
    container_defaults: ContainerEffectiveSettings | None,
    flow_skills: TrustedSkillSettings | None,
    skills_registry: SkillRegistry | None,
) -> FlowNodePlan:
    metadata = registry.metadata(node.type)
    assert metadata is not None
    assert metadata.kind is not None
    node_metadata = await _compile_node_metadata(
        definition,
        node,
        registry,
        metadata.metadata,
        skills_registry=skills_registry,
    )
    node_skills = _compile_node_skills(
        node,
        metadata.kind,
        registry,
        node_metadata,
        flow_skills=flow_skills,
        skills_registry=skills_registry,
    )
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
        container=_compile_node_container(
            definition,
            node,
            registry,
            container_defaults,
        ),
        skills=node_skills,
        config=node.config,
        metadata=node_metadata,
    )


def _compile_container_defaults(
    definition: FlowDefinition,
) -> ContainerEffectiveSettings | None:
    if definition.container is None:
        return None
    try:
        return _with_flow_node_scope(
            ContainerAuthorityCaps(settings=definition.container).merge()
        )
    except AssertionError as error:
        raise _container_configuration_error(
            path="runtime.container",
            message="Flow container defaults are invalid.",
            hint=_assertion_hint(error),
        ) from None


def _compile_runtime_envelope(
    definition: FlowDefinition,
    container_defaults: ContainerEffectiveSettings | None,
) -> ContainerNormalizedRuntimeEnvelopePlan | None:
    envelope = definition.runtime_envelope
    if envelope is None:
        return None
    if definition.container is None:
        raise _container_configuration_error(
            path="runtime.container.envelope",
            message="Flow runtime envelope selection has no trusted defaults.",
            hint=(
                "Define runtime.container settings before selecting an "
                "envelope."
            ),
        )
    try:
        effective = definition.container.select_profile(envelope.container)
        request = ContainerPlanRequest(
            request_kind=ContainerPlanRequestKind.RUNTIME_ENVELOPE,
            logical_name=definition.name,
            command="flow-runtime",
            argv=("flow-runtime", definition.name),
            scope=ContainerExecutionScope.RUNTIME_ENVELOPE,
        )
        return normalize_runtime_envelope_plan(
            effective,
            request,
            envelope_kind=ContainerRuntimeEnvelopeKind.FLOW_RUNTIME,
            readiness_timeout_seconds=envelope.readiness_timeout_seconds,
        )
    except AssertionError as error:
        raise _container_configuration_error(
            path="runtime.container.envelope",
            message="Flow runtime envelope selection is invalid.",
            hint=_assertion_hint(error),
        ) from None


def _compile_node_skills(
    node: FlowNodeDefinition,
    kind: FlowNodeKind,
    registry: FlowNodeRegistry,
    metadata: Mapping[str, object],
    *,
    flow_skills: TrustedSkillSettings | None,
    skills_registry: SkillRegistry | None,
) -> TrustedSkillSettings | None:
    assert isinstance(node, FlowNodeDefinition)
    assert isinstance(kind, FlowNodeKind)
    if skills_registry is not None:
        assert isinstance(skills_registry, SkillRegistry)
    node_uses_skills = _node_uses_skills(kind, metadata)
    if node.skills is not None and not node_uses_skills:
        raise FlowNodeConfigurationError(
            code="flow.skills_unsupported_node",
            path=f"nodes.{node.name}.skills",
            message="Flow node skills settings are not supported.",
            hint=(
                "Use skills settings only on agent nodes or skills tool nodes."
            ),
        )
    if not node_uses_skills:
        return None
    inherited = (
        flow_skills
        or _registry_node_skills(registry, node)
        or (skills_registry.settings if skills_registry is not None else None)
    )
    if inherited is None:
        if node.skills is not None:
            raise FlowNodeConfigurationError(
                code="flow.skills_trusted_settings_required",
                path=f"nodes.{node.name}.skills",
                message=(
                    "Flow node skills settings require trusted flow defaults."
                ),
                hint=(
                    "Provide trusted skills defaults from SDK, CLI, or "
                    "deployment configuration."
                ),
            )
        return None
    _assert_registry_settings_allow(
        inherited,
        skills_registry,
        path=f"nodes.{node.name}.skills",
    )
    if node.skills is None:
        _assert_node_registry_settings_allow(
            inherited,
            registry,
            node,
            path=f"nodes.{node.name}.skills",
        )
        return inherited
    result = merge_skill_settings(inherited, node.skills)
    if result.diagnostics:
        raise _skills_configuration_error(
            result.diagnostics[0],
            path=f"nodes.{node.name}.skills",
        )
    _assert_registry_settings_allow(
        result.settings,
        skills_registry,
        path=f"nodes.{node.name}.skills",
    )
    _assert_node_registry_settings_allow(
        result.settings,
        registry,
        node,
        path=f"nodes.{node.name}.skills",
    )
    return result.settings


def _node_uses_skills(
    kind: FlowNodeKind,
    metadata: Mapping[str, object],
) -> bool:
    if kind is FlowNodeKind.AGENT:
        return True
    if kind is not FlowNodeKind.TOOL:
        return False
    tool = metadata.get("tool")
    if not isinstance(tool, Mapping):
        return False
    canonical_name = tool.get("canonical_name")
    return (
        isinstance(canonical_name, str)
        and canonical_name in CANONICAL_SKILLS_TOOL_NAMES
    )


def _registry_node_skills(
    registry: FlowNodeRegistry,
    node: FlowNodeDefinition,
) -> TrustedSkillSettings | None:
    if not registry.supports_tool_resolution(node.type):
        return None
    skill_registry = registry.tool_skill_registry(node.type)
    if skill_registry is None:
        return None
    return skill_registry.settings


def _assert_node_registry_settings_allow(
    settings: TrustedSkillSettings,
    registry: FlowNodeRegistry,
    node: FlowNodeDefinition,
    *,
    path: str,
) -> None:
    if not registry.supports_tool_resolution(node.type):
        return
    skill_registry = registry.tool_skill_registry(node.type)
    if skill_registry is None or skill_registry.settings is None:
        return
    _assert_registry_settings_allow(
        settings,
        skill_registry,
        path=path,
    )


def _assert_registry_settings_allow(
    settings: TrustedSkillSettings,
    registry: SkillRegistry | None,
    *,
    path: str,
) -> None:
    if registry is None:
        return
    if not _registry_actual_sources_allow(registry, settings):
        raise FlowNodeConfigurationError(
            code="flow.skills_policy_invalid",
            path=path,
            message="Flow skills settings are wider than the skills registry.",
            hint=(
                "Use skills flow or node settings that only narrow the "
                "trusted skills registry."
            ),
        )
    if registry.settings is None:
        return
    _assert_settings_not_wider(
        settings,
        registry.settings,
        path=path,
        message="Flow skills settings are wider than the skills registry.",
    )


def _registry_actual_sources_allow(
    registry: SkillRegistry,
    settings: TrustedSkillSettings,
) -> bool:
    if not settings.sources_explicit or not settings.sources:
        return True
    registry_sources = {
        source.label: _registry_source_identity(source)
        for source in registry.sources
    }
    for source in settings.sources:
        if registry_sources.get(source.label) != (
            trusted_skill_source_identity_dict(source)
        ):
            return False
    return True


def _registry_source_identity(source: object) -> Mapping[str, SkillModelValue]:
    source_identity = getattr(source, "source_identity", None)
    if isinstance(source_identity, Mapping) and source_identity:
        return source_identity
    label = getattr(source, "label", None)
    authority = getattr(source, "authority", None)
    status = getattr(source, "status", None)
    if (
        isinstance(label, str)
        and isinstance(authority, SkillSourceAuthority)
        and isinstance(status, SkillStatus)
    ):
        return {
            "label": label,
            "authority": authority.as_model_dict(),
            "enabled": True,
            "allow_hidden_paths": False,
            "status": status.value,
        }
    return {}


def _assert_settings_not_wider(
    requested: TrustedSkillSettings,
    trusted: TrustedSkillSettings,
    *,
    path: str,
    message: str,
) -> None:
    assert isinstance(requested, TrustedSkillSettings)
    assert isinstance(trusted, TrustedSkillSettings)
    if trusted_skill_settings_identity_dict(requested) == (
        trusted_skill_settings_identity_dict(trusted)
    ):
        return
    if _trusted_settings_allow(trusted, requested):
        return
    raise FlowNodeConfigurationError(
        code="flow.skills_policy_invalid",
        path=path,
        message=message,
        hint=(
            "Use skills flow or node settings that only narrow the trusted "
            "skills registry."
        ),
    )


def _trusted_settings_allow(
    trusted: TrustedSkillSettings,
    requested: TrustedSkillSettings,
) -> bool:
    if requested.enabled and not trusted.enabled:
        return False
    if requested.bootstrap_enabled and not trusted.bootstrap_enabled:
        return False
    if not set(requested.authority_kinds).issubset(
        set(trusted.authority_kinds)
    ):
        return False
    if not _trusted_sources_allow(trusted, requested):
        return False
    if not _trusted_skill_ids_allow(
        trusted.allowed_skill_ids,
        requested.allowed_skill_ids,
    ):
        return False
    return (
        trusted.read_limits.allows(requested.read_limits)
        and trusted.index_limits.allows(requested.index_limits)
        and trusted.source_limits.allows(requested.source_limits)
        and trusted.cursor_limits.allows(requested.cursor_limits)
        and trusted.privacy.allows(requested.privacy)
        and trusted.observability.allows(requested.observability)
    )


def _trusted_sources_allow(
    trusted: TrustedSkillSettings,
    requested: TrustedSkillSettings,
) -> bool:
    if not trusted.sources_explicit:
        return not requested.sources_explicit or not requested.sources
    if not requested.sources_explicit:
        return False
    trusted_sources = {
        source.label: trusted_skill_source_identity_dict(source)
        for source in trusted.sources
    }
    for source in requested.sources:
        if trusted_sources.get(source.label) != (
            trusted_skill_source_identity_dict(source)
        ):
            return False
    return True


def _trusted_skill_ids_allow(
    trusted_skill_ids: tuple[str, ...],
    requested_skill_ids: tuple[str, ...],
) -> bool:
    if not trusted_skill_ids:
        return True
    if not requested_skill_ids:
        return False
    return set(requested_skill_ids).issubset(set(trusted_skill_ids))


def _skills_configuration_error(
    diagnostic: SkillDiagnosticInfo,
    *,
    path: str,
) -> FlowNodeConfigurationError:
    assert isinstance(diagnostic, SkillDiagnosticInfo)
    return FlowNodeConfigurationError(
        code=diagnostic.code.value,
        path=_skill_diagnostic_path(diagnostic.path, path=path),
        message=diagnostic.message,
        hint=diagnostic.hint,
    )


def _skill_diagnostic_path(skill_path: str, *, path: str) -> str:
    assert isinstance(skill_path, str) and skill_path.strip()
    suffix = (
        skill_path[len("settings.") :]
        if skill_path.startswith("settings.")
        else skill_path
    )
    return f"{path}.{suffix}"


def _compile_node_container(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    container_defaults: ContainerEffectiveSettings | None,
) -> ContainerEffectiveSettings | None:
    inherited = _registry_tool_container(registry, node)
    if node.container is None:
        try:
            effective = _node_container_cap(
                inherited,
                container_defaults,
            )
            if effective is not None:
                effective = _with_flow_node_scope(effective)
                _validate_node_container_deadline(node, effective)
        except AssertionError as error:
            raise _container_configuration_error(
                path=f"nodes.{node.name}.runtime.container",
                message=(
                    "Node container policy is wider than trusted defaults."
                ),
                hint=_assertion_hint(error),
            ) from None
        return effective
    if inherited is not None:
        try:
            cap = _node_container_cap(inherited, container_defaults)
            assert cap is not None
            effective = ContainerAuthorityCaps(
                settings=_settings_from_effective(cap)
            ).merge((node.container,))
            effective = _with_flow_node_scope(effective)
            _validate_node_container_deadline(node, effective)
            return effective
        except AssertionError as error:
            raise _container_configuration_error(
                path=f"nodes.{node.name}.runtime.container",
                message=(
                    "Node container policy is wider than trusted defaults."
                ),
                hint=_assertion_hint(error),
            ) from None
    if definition.container is None:
        raise _container_configuration_error(
            path=f"nodes.{node.name}.runtime.container",
            message="Node container policy has no trusted flow defaults.",
            hint=(
                "Define runtime.container defaults before node-level "
                "narrowing."
            ),
        )
    try:
        effective = ContainerAuthorityCaps(
            settings=definition.container
        ).merge((node.container,))
        effective = _with_flow_node_scope(effective)
        _validate_node_container_deadline(node, effective)
        return effective
    except AssertionError as error:
        raise _container_configuration_error(
            path=f"nodes.{node.name}.runtime.container",
            message="Node container policy is wider than trusted defaults.",
            hint=_assertion_hint(error),
        ) from None


def _node_container_cap(
    inherited: ContainerEffectiveSettings | None,
    container_defaults: ContainerEffectiveSettings | None,
) -> ContainerEffectiveSettings | None:
    if inherited is None:
        return container_defaults
    if container_defaults is not None:
        _assert_effective_no_wider(container_defaults, inherited)
    return inherited


def _registry_tool_container(
    registry: FlowNodeRegistry,
    node: FlowNodeDefinition,
) -> ContainerEffectiveSettings | None:
    if not registry.supports_tool_resolution(node.type):
        return None
    container = registry.tool_container_settings(node.type)
    if container is not None:
        assert isinstance(container, ContainerEffectiveSettings)
    return container


def _settings_from_effective(
    effective: ContainerEffectiveSettings,
) -> ContainerSettings:
    profile_name = effective.profile_name
    profile = effective.profile
    profiles = {}
    if profile_name is not None:
        assert profile is not None
        profiles[profile_name] = profile
    return ContainerSettings(
        source=effective.source,
        backend=effective.backend,
        default_profile=profile_name,
        allowed_profiles=() if profile_name is None else (profile_name,),
        profiles=profiles,
        profile_registry_id=effective.profile_registry_id,
        policy_version=effective.policy_version,
    )


def _assert_effective_no_wider(
    caps: ContainerEffectiveSettings,
    requested: ContainerEffectiveSettings,
) -> None:
    assert _enum_value(caps.backend) == _enum_value(
        requested.backend
    ), "tool backend cannot widen flow container defaults"
    assert (
        requested.required or not caps.required
    ), "tool required flag cannot weaken flow container defaults"
    if caps.profile is None:
        assert (
            requested.profile is None
        ), "tool profile cannot widen disabled flow container defaults"
        return
    assert requested.profile is not None, "tool profile cannot remove caps"
    _assert_profile_mapping_no_wider(
        caps.profile.to_dict(),
        requested.profile.to_dict(),
    )


def _assert_profile_mapping_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    _assert_mapping_equal(
        _mapping(caps.get("image", {})),
        _mapping(requested.get("image", {})),
        "tool image cannot widen flow container defaults",
    )
    _assert_mapping_equal(
        _mapping(caps.get("workspace", {})),
        _mapping(requested.get("workspace", {})),
        "tool workspace cannot widen flow container defaults",
    )
    _assert_mounts_no_wider(
        _sequence_mapping(caps.get("mounts", ())),
        _sequence_mapping(requested.get("mounts", ())),
    )
    _assert_environment_no_wider(
        _mapping(caps.get("environment", {})),
        _mapping(requested.get("environment", {})),
    )
    _assert_sequences_no_wider(
        _sequence_mapping(caps.get("secrets", ())),
        _sequence_mapping(requested.get("secrets", ())),
        key="name",
        message="tool secrets cannot widen flow container defaults",
    )
    _assert_network_no_wider(
        _mapping(caps.get("network", {})),
        _mapping(requested.get("network", {})),
    )
    _assert_devices_no_wider(
        _mapping(caps.get("devices", {})),
        _mapping(requested.get("devices", {})),
    )
    _assert_resources_no_wider(
        _mapping(caps.get("resources", {})),
        _mapping(requested.get("resources", {})),
    )
    _assert_output_no_wider(
        _mapping(caps.get("output", {})),
        _mapping(requested.get("output", {})),
    )
    _assert_cleanup_no_wider(
        _mapping(caps.get("cleanup", {})),
        _mapping(requested.get("cleanup", {})),
    )
    _assert_mapping_equal(
        _mapping(caps.get("pooling", {})),
        _mapping(requested.get("pooling", {})),
        "tool pooling policy cannot widen flow container defaults",
    )
    _assert_mapping_equal(
        _mapping(caps.get("audit", {})),
        _mapping(requested.get("audit", {})),
        "tool audit policy cannot widen flow container defaults",
    )
    assert _enum_value(caps.get("command_mode")) == _enum_value(
        requested.get("command_mode")
    ), "tool command mode cannot widen flow container defaults"
    assert caps.get("user") == requested.get(
        "user"
    ), "tool user cannot widen flow container defaults"
    assert not (
        caps.get("read_only_rootfs") is True
        and requested.get("read_only_rootfs") is not True
    ), "tool root filesystem cannot widen flow container defaults"
    _assert_escalation_no_wider(
        _mapping(caps.get("escalation", {})),
        _mapping(requested.get("escalation", {})),
    )


def _assert_mounts_no_wider(
    caps: tuple[Mapping[str, object], ...],
    requested: tuple[Mapping[str, object], ...],
) -> None:
    caps_by_target = _mapping_by_key(caps, "target")
    for mount in requested:
        target = _string_value(mount.get("target"))
        assert (
            target in caps_by_target
        ), "tool mounts cannot widen flow container defaults"
        cap = caps_by_target[target]
        assert mount.get("source") == cap.get(
            "source"
        ), "tool mount source cannot widen flow container defaults"
        assert mount.get("mount_type") == cap.get(
            "mount_type"
        ), "tool mount type cannot widen flow container defaults"
        if cap.get("access") == "read":
            assert (
                mount.get("access") == "read"
            ), "tool mount access cannot widen flow container defaults"


def _assert_environment_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    cap_vars = _mapping(caps.get("variables", {}))
    requested_vars = _mapping(requested.get("variables", {}))
    for name, value in requested_vars.items():
        assert (
            name in cap_vars and cap_vars[name] == value
        ), "tool environment cannot widen flow container defaults"
    cap_allowlist = set(_string_tuple(caps.get("allowlist", ())))
    for name in _string_tuple(requested.get("allowlist", ())):
        assert (
            name in cap_allowlist
        ), "tool environment allowlist cannot widen flow container defaults"
    assert not (
        caps.get("inherit_host") is False
        and requested.get("inherit_host") is True
    ), "tool environment cannot inherit host variables"


def _assert_network_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    cap_mode = _string_value(caps.get("mode", "none"))
    requested_mode = _string_value(requested.get("mode", "none"))
    ranks = {"none": 0, "loopback": 1, "allowlist": 2, "full": 3}
    assert (
        ranks[requested_mode] <= ranks[cap_mode]
    ), "tool network cannot widen flow container defaults"
    if cap_mode == "allowlist":
        cap_allowlist = set(_string_tuple(caps.get("egress_allowlist", ())))
        for host in _string_tuple(requested.get("egress_allowlist", ())):
            assert (
                host in cap_allowlist
            ), "tool network allowlist cannot widen flow container defaults"


def _assert_devices_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    cap_devices = set(_string_tuple(caps.get("devices", ())))
    for device in _string_tuple(requested.get("devices", ())):
        assert (
            device in cap_devices
        ), "tool devices cannot widen flow container defaults"


def _assert_resources_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    for key in ("cpu_count", "memory_bytes", "pids", "timeout_seconds"):
        _assert_limit_no_wider(caps.get(key), requested.get(key), key)


def _assert_output_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    for key in ("max_stdout_bytes", "max_stderr_bytes", "max_artifact_bytes"):
        _assert_limit_no_wider(caps.get(key), requested.get(key), key)
    assert not (
        caps.get("allow_artifacts") is False
        and requested.get("allow_artifacts") is True
    ), "tool artifact output cannot widen flow container defaults"


def _assert_cleanup_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    assert caps.get("mode") == requested.get(
        "mode"
    ), "tool cleanup mode cannot widen flow container defaults"
    _assert_limit_no_wider(
        caps.get("grace_seconds"),
        requested.get("grace_seconds"),
        "grace_seconds",
    )


def _assert_escalation_no_wider(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
) -> None:
    ranks = {"deny": 0, "require_review": 1, "preauthorized": 2}
    cap_mode = _string_value(caps.get("mode", "deny"))
    requested_mode = _string_value(requested.get("mode", "deny"))
    assert (
        ranks[requested_mode] <= ranks[cap_mode]
    ), "tool escalation policy cannot widen flow container defaults"


def _assert_sequences_no_wider(
    caps: tuple[Mapping[str, object], ...],
    requested: tuple[Mapping[str, object], ...],
    *,
    key: str,
    message: str,
) -> None:
    caps_by_key = _mapping_by_key(caps, key)
    for item in requested:
        item_key = _string_value(item.get(key))
        assert (
            item_key in caps_by_key and item == caps_by_key[item_key]
        ), message


def _assert_mapping_equal(
    caps: Mapping[str, object],
    requested: Mapping[str, object],
    message: str,
) -> None:
    assert dict(requested) == dict(caps), message


def _assert_limit_no_wider(
    caps: object,
    requested: object,
    field_name: str,
) -> None:
    if caps is None:
        return
    assert isinstance(caps, int) and not isinstance(caps, bool)
    assert isinstance(requested, int) and not isinstance(requested, bool)
    assert requested <= caps, f"tool {field_name} cannot widen flow defaults"


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def _sequence_mapping(value: object) -> tuple[Mapping[str, object], ...]:
    assert isinstance(value, list | tuple)
    result: list[Mapping[str, object]] = []
    for item in value:
        assert isinstance(item, Mapping)
        result.append(item)
    return tuple(result)


def _mapping_by_key(
    values: tuple[Mapping[str, object], ...],
    key: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for value in values:
        key_value = _string_value(value.get(key))
        assert key_value not in result
        result[key_value] = value
    return result


def _string_tuple(value: object) -> tuple[str, ...]:
    assert isinstance(value, list | tuple)
    result: list[str] = []
    for item in value:
        result.append(_string_value(item))
    return tuple(result)


def _string_value(value: object) -> str:
    assert isinstance(value, str) and value.strip()
    return value


def _enum_value(value: object) -> str:
    if hasattr(value, "value"):
        enum_value = getattr(value, "value")
        assert isinstance(enum_value, str)
        return enum_value
    return _string_value(value)


def _with_flow_node_scope(
    effective: ContainerEffectiveSettings,
) -> ContainerEffectiveSettings:
    assert isinstance(effective, ContainerEffectiveSettings)
    return replace(
        effective,
        scope=ContainerExecutionScope.DURABLE_WORKFLOW,
    )


def _validate_node_container_deadline(
    node: FlowNodeDefinition,
    effective: ContainerEffectiveSettings,
) -> None:
    if node.timeout_policy is None:
        return
    if node.timeout_policy.per_attempt_seconds is None:
        return
    if effective.profile is None:
        return
    timeout_seconds = effective.profile.resources.timeout_seconds
    if timeout_seconds is None:
        return
    assert (
        timeout_seconds <= node.timeout_policy.per_attempt_seconds
    ), "container timeout cannot exceed node timeout"


def _container_configuration_error(
    *,
    path: str,
    message: str,
    hint: str,
) -> FlowNodeConfigurationError:
    return FlowNodeConfigurationError(
        code="flow.container_policy_invalid",
        path=path,
        message=message,
        hint=hint,
    )


def _assertion_hint(error: AssertionError) -> str:
    text = str(error).strip()
    if text:
        return text
    return "Use only trusted container defaults and narrower node settings."


def flow_node_container_fingerprint(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
) -> str:
    assert isinstance(plan, FlowExecutionPlan)
    assert isinstance(node, FlowNodePlan)
    payload = {
        "flow": plan.name,
        "version": plan.version,
        "revision": plan.revision,
        "node": node.name,
        "type": node.type,
        "ref": node.ref,
        "config": _json_fingerprint_value(node.config),
    }
    encoded = dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _json_fingerprint_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {
            key: _json_fingerprint_value(item)
            for key, item in sorted(value.items())
            if isinstance(key, str)
        }
    if isinstance(value, list | tuple):
        return [_json_fingerprint_value(item) for item in value]
    return {"type": type(value).__name__}


async def _compile_node_metadata(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    registry: FlowNodeRegistry,
    metadata: Mapping[str, object],
    *,
    skills_registry: SkillRegistry | None,
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
        node_skills_registry = (
            registry.tool_skill_registry(node.type) or skills_registry
        )
        if (
            node_skills_registry is not None
            and descriptor.name in CANONICAL_SKILLS_TOOL_NAMES
        ):
            compiled["skills_registry"] = node_skills_registry
    if registry.supports_subflow_resolution(node.type):
        compiled["subflow"] = await registry.subflow_metadata(
            definition,
            node,
        )
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
