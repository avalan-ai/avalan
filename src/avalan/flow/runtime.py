from ..container import (
    ContainerApprovalRecord,
    ContainerAuditEvent,
    ContainerAuditEventType,
    ContainerAuthorizationDecisionType,
    ContainerEffectiveSettings,
    ContainerPolicy,
    ContainerPolicyContext,
    ContainerPolicyPlan,
    ContainerResultStatus,
    ContainerReviewSurface,
)
from ..event import EventType
from ..model.stream import CanonicalStreamItem
from ..types import assert_non_empty_string
from .condition import FlowConditionOperator, FlowConditionValueType
from .definition import (
    FlowEdgeKind,
    FlowJoinPolicyType,
    FlowMappingKind,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowRetryBackoffStrategy,
    FlowRouteMatchPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .node import CancellationChecker, Node
from .plan import (
    FLOW_RESUME_ISOLATION_METADATA_KEY,
    FlowConditionPlan,
    FlowEdgePlan,
    FlowExecutionPlan,
    FlowMappingPlan,
    FlowNodePlan,
    FlowRetryPlan,
    flow_node_container_fingerprint,
    flow_resume_isolation_metadata,
)
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    default_flow_node_registry,
)
from .selector import FlowSelector, resolve_flow_selector_value
from .state import FlowEdgeState, FlowExecutionTrace, FlowNodeState
from .stream import (
    FlowStreamSession,
    FlowStreamSink,
    canonical_flow_item,
    flow_stream_session,
)

from asyncio import CancelledError, gather, sleep, wait_for
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from importlib import import_module
from inspect import isawaitable
from time import monotonic, time
from types import MappingProxyType, ModuleType
from typing import Protocol, TypeAlias, cast
from uuid import uuid4

_MISSING = object()

FlowPlanNodeRunner: TypeAlias = Callable[
    [FlowNodePlan, Mapping[str, object]],
    object | Awaitable[object],
]
FlowStreamListener: TypeAlias = FlowStreamSink


class FlowRuntimeEnvelopeRunner(Protocol):
    trusted_runtime_envelope_runner: bool

    async def run_flow_runtime_envelope(
        self,
        plan: FlowExecutionPlan,
        *,
        inputs: Mapping[str, object],
        cancellation_checker: CancellationChecker | None,
        event_listener: FlowStreamListener | None,
        stream_session: FlowStreamSession | None,
        concurrency_limit: int,
        resume_trace: FlowExecutionTrace | None,
        resume_node_outputs: Mapping[str, Mapping[str, object]],
        resume_decisions: Mapping[str, Mapping[str, object]],
    ) -> "FlowPlanExecutionResult":
        """Run a flow through a trusted runtime envelope."""


def is_trusted_flow_runtime_envelope_runner(
    runner: object,
) -> bool:
    return getattr(runner, "trusted_runtime_envelope_runner", False) is True


@dataclass(frozen=True, slots=True, kw_only=True)
class _FlowExecutionOptions:
    cancellation_checker: CancellationChecker | None = None
    event_listener: FlowStreamListener | None = None
    stream_session: FlowStreamSession | None = None


_FLOW_EXECUTION_OPTIONS: ContextVar[_FlowExecutionOptions | None] = ContextVar(
    "_FLOW_EXECUTION_OPTIONS", default=None
)


class _JsonSchemaValidator(Protocol):
    def validate(self, instance: object) -> None:
        """Validate a JSON schema instance."""


class _JsonSchemaValidatorClass(Protocol):
    def __call__(self, schema: Mapping[str, object]) -> _JsonSchemaValidator:
        """Build a validator for a JSON schema."""

    def check_schema(self, schema: Mapping[str, object]) -> None:
        """Validate a JSON schema definition."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _JsonSchemaAdapter:
    validator_class: _JsonSchemaValidatorClass
    schema_error: type[Exception]
    validation_error: type[Exception]


@dataclass(frozen=True, slots=True, kw_only=True)
class _NodeRunOutcome:
    node: FlowNodePlan
    state: FlowNodeState
    route_kind: FlowEdgeKind
    attempts: int
    diagnostics: tuple[FlowDiagnostic, ...]
    duration_ms: float | None = None
    failure_category: str | None = None
    exhausted_route: str | None = None
    exhausted_route_kind: FlowEdgeKind | None = None
    pause_token: str | None = None
    output: object = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _FlowEventDraft:
    type: EventType
    payload: Mapping[str, object]
    started: float | None = None
    finished: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.type, EventType)
        assert isinstance(self.payload, Mapping)
        if self.started is not None:
            assert isinstance(self.started, int | float)
            assert not isinstance(self.started, bool)
        if self.finished is not None:
            assert isinstance(self.finished, int | float)
            assert not isinstance(self.finished, bool)


class FlowNodeExecutionError(Exception):
    """Represent a safe flow-local node execution failure."""

    def __init__(
        self,
        *,
        code: str = "flow.execution.node_failed",
        message: str = "Flow node failed.",
        hint: str = "Inspect error routes for this node.",
        failure_category: str = "error",
        route_kind: FlowEdgeKind = FlowEdgeKind.ERROR,
    ) -> None:
        assert isinstance(code, str) and code.strip()
        assert isinstance(message, str) and message.strip()
        assert isinstance(hint, str) and hint.strip()
        assert isinstance(failure_category, str) and failure_category.strip()
        assert isinstance(route_kind, FlowEdgeKind)
        self.code = code
        self.safe_message = message
        self.hint = hint
        self.failure_category = failure_category
        self.route_kind = route_kind
        super().__init__(code)


class FlowNodeRegistryRunner:
    """Run execution plan nodes through a flow node registry."""

    def __init__(self, registry: FlowNodeRegistry | None = None) -> None:
        if registry is not None:
            assert isinstance(registry, FlowNodeRegistry)
        self._registry = registry or default_flow_node_registry()
        self._nodes: dict[str, Node] = {}

    async def __call__(
        self,
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        assert isinstance(node, FlowNodePlan)
        assert isinstance(inputs, Mapping)
        if node.kind == FlowNodeKind.SUBFLOW:
            return await self._run_subflow(node, inputs)
        try:
            runtime_node = self._node(node)
            return await runtime_node.execute_async(dict(inputs))
        except FlowNodeConfigurationError as error:
            raise FlowNodeExecutionError(
                code=error.code,
                message=error.message,
                hint=error.hint,
                failure_category="validation",
            ) from None

    def _node(self, node: FlowNodePlan) -> "Node":
        cached = self._nodes.get(node.name)
        if cached is not None:
            return cached
        runtime_node = self._registry.build(_plan_node_definition(node))
        self._nodes[node.name] = runtime_node
        return runtime_node

    async def _run_subflow(
        self,
        node: FlowNodePlan,
        inputs: Mapping[str, object],
    ) -> object:
        metadata = node.metadata.get("subflow")
        if not isinstance(metadata, Mapping):
            raise _subflow_execution_error()
        plan = metadata.get("plan")
        output_mapping = metadata.get("output_mapping")
        if not isinstance(plan, FlowExecutionPlan) or not isinstance(
            output_mapping,
            Mapping,
        ):
            raise _subflow_execution_error()
        options = _FLOW_EXECUTION_OPTIONS.get()
        result = await execute_flow_plan(
            plan,
            self,
            inputs=inputs,
            cancellation_checker=(
                options.cancellation_checker if options is not None else None
            ),
            event_listener=(
                options.event_listener if options is not None else None
            ),
            stream_session=(
                options.stream_session if options is not None else None
            ),
        )
        if not result.ok:
            raise FlowNodeExecutionError(
                code="flow.execution.subflow_failed",
                message="Subflow node failed.",
                hint="Inspect the referenced flow diagnostics.",
                failure_category="error",
            )
        return {
            str(target): result.outputs[str(source)]
            for target, source in output_mapping.items()
        }


def flow_node_registry_runner(
    registry: FlowNodeRegistry | None = None,
) -> FlowPlanNodeRunner:
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    return FlowNodeRegistryRunner(registry)


def _plan_node_definition(node: FlowNodePlan) -> FlowNodeDefinition:
    assert isinstance(node, FlowNodePlan)
    return FlowNodeDefinition(
        name=node.name,
        type=node.type,
        ref=node.ref,
        config=node.config,
    )


def _subflow_execution_error() -> FlowNodeExecutionError:
    return FlowNodeExecutionError(
        code="flow.execution.subflow_unavailable",
        message="Subflow node is missing a compiled plan.",
        hint="Compile the flow definition before execution.",
        failure_category="validation",
    )


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _empty_node_outputs() -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({})


def _empty_diagnostics() -> tuple[FlowDiagnostic, ...]:
    return ()


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    frozen: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str) and key.strip()
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowRuntimeContext:
    inputs: Mapping[str, object] = field(default_factory=_empty_mapping)
    node_outputs: Mapping[str, Mapping[str, object]] = field(
        default_factory=_empty_node_outputs
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _freeze_mapping(self.inputs))
        frozen_outputs: dict[str, Mapping[str, object]] = {}
        for node_name, outputs in self.node_outputs.items():
            assert isinstance(node_name, str) and node_name.strip()
            assert isinstance(outputs, Mapping)
            frozen_outputs[node_name] = _freeze_mapping(outputs)
        object.__setattr__(
            self,
            "node_outputs",
            MappingProxyType(frozen_outputs),
        )


class FlowRuntimeEvaluationError(ValueError):
    def __init__(self, code: str) -> None:
        assert isinstance(code, str) and code.strip()
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowPlanExecutionResult:
    trace: FlowExecutionTrace
    outputs: Mapping[str, object] = field(default_factory=_empty_mapping)
    diagnostics: tuple[FlowDiagnostic, ...] = field(
        default_factory=_empty_diagnostics
    )
    node_outputs: Mapping[str, Mapping[str, object]] = field(
        default_factory=_empty_node_outputs
    )
    pause_tokens: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", _freeze_mapping(self.outputs))
        assert isinstance(self.trace, FlowExecutionTrace)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)
        frozen_outputs: dict[str, Mapping[str, object]] = {}
        for node_name, outputs in self.node_outputs.items():
            assert isinstance(node_name, str) and node_name.strip()
            assert isinstance(outputs, Mapping)
            frozen_outputs[node_name] = _freeze_mapping(outputs)
        object.__setattr__(
            self,
            "node_outputs",
            MappingProxyType(frozen_outputs),
        )
        frozen_tokens: dict[str, str] = {}
        for node_name, token in self.pause_tokens.items():
            assert isinstance(node_name, str) and node_name.strip()
            assert isinstance(token, str) and token.strip()
            frozen_tokens[node_name] = token
        object.__setattr__(
            self,
            "pause_tokens",
            MappingProxyType(frozen_tokens),
        )

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


async def execute_flow_plan(
    plan: FlowExecutionPlan,
    runner: FlowPlanNodeRunner,
    *,
    inputs: Mapping[str, object] | None = None,
    cancellation_checker: CancellationChecker | None = None,
    event_listener: FlowStreamListener | None = None,
    stream_session: FlowStreamSession | None = None,
    concurrency_limit: int = 1,
    resume_trace: FlowExecutionTrace | None = None,
    resume_node_outputs: Mapping[str, Mapping[str, object]] | None = None,
    resume_decisions: Mapping[str, Mapping[str, object]] | None = None,
    runtime_envelope_runner: FlowRuntimeEnvelopeRunner | None = None,
) -> FlowPlanExecutionResult:
    assert isinstance(plan, FlowExecutionPlan)
    assert callable(runner)
    if runtime_envelope_runner is not None:
        assert hasattr(
            runtime_envelope_runner,
            "run_flow_runtime_envelope",
        )
        assert is_trusted_flow_runtime_envelope_runner(
            runtime_envelope_runner
        ), "runtime envelope runner must be trusted"
    if event_listener is not None:
        assert callable(event_listener)
    if stream_session is not None:
        assert isinstance(stream_session, FlowStreamSession)
    if inputs is not None:
        assert isinstance(inputs, Mapping)
    if resume_trace is not None:
        assert isinstance(resume_trace, FlowExecutionTrace)
    if resume_node_outputs is not None:
        assert isinstance(resume_node_outputs, Mapping)
    if resume_decisions is not None:
        assert isinstance(resume_decisions, Mapping)
    assert isinstance(concurrency_limit, int) and not isinstance(
        concurrency_limit,
        bool,
    )
    assert concurrency_limit > 0
    flow_run_id = _flow_event_id(plan)
    resume_isolation_diagnostics = _resume_isolation_metadata_diagnostics(
        plan,
        resume_trace,
    )
    if resume_isolation_diagnostics:
        return FlowPlanExecutionResult(
            trace=resume_trace or FlowExecutionTrace.from_plan(plan),
            diagnostics=resume_isolation_diagnostics,
            node_outputs=resume_node_outputs or {},
        )
    if plan.runtime_envelope is not None:
        if runtime_envelope_runner is not None:
            return await runtime_envelope_runner.run_flow_runtime_envelope(
                plan,
                inputs=_freeze_mapping(inputs or {}),
                cancellation_checker=cancellation_checker,
                event_listener=event_listener,
                stream_session=stream_session,
                concurrency_limit=concurrency_limit,
                resume_trace=resume_trace,
                resume_node_outputs={
                    node: _freeze_mapping(outputs)
                    for node, outputs in (resume_node_outputs or {}).items()
                },
                resume_decisions={
                    node: _freeze_mapping(payload)
                    for node, payload in (resume_decisions or {}).items()
                },
            )
        return FlowPlanExecutionResult(
            trace=resume_trace or FlowExecutionTrace.from_plan(plan),
            diagnostics=(_runtime_envelope_unavailable_diagnostic(),),
            node_outputs=resume_node_outputs or {},
        )
    if stream_session is None:
        stream_session = flow_stream_session(
            stream_session_id=str(uuid4()),
            run_id=flow_run_id,
            turn_id=flow_run_id,
            cancellation_checker=cancellation_checker,
        )
    cancellation_checker = _stream_session_cancellation_checker(
        stream_session,
        cancellation_checker,
    )
    input_values = _freeze_mapping(inputs or {})
    node_outputs: dict[str, Mapping[str, object]] = {
        node: _freeze_mapping(outputs)
        for node, outputs in (resume_node_outputs or {}).items()
    }
    resume_payloads: dict[str, Mapping[str, object]] = {
        node: _freeze_mapping(payload)
        for node, payload in (resume_decisions or {}).items()
    }
    pause_tokens: dict[str, str] = {}
    diagnostics: list[FlowDiagnostic] = []
    trace = resume_trace or FlowExecutionTrace.from_plan(plan)
    node_map = plan.node_map
    event_drafts: list[_FlowEventDraft] = []
    (
        ready,
        queued,
        processed,
        trace,
        resume_diagnostics,
        container_approvals,
    ) = _initial_runtime_queue(
        plan,
        input_values,
        node_outputs,
        resume_payloads,
        trace,
        event_drafts=event_drafts,
    )
    diagnostics.extend(resume_diagnostics)
    paused = False
    flow_started_at = monotonic()
    await _emit_flow_event(
        event_listener,
        stream_session,
        EventType.FLOW_VALIDATION,
        plan,
        payload={
            "status": "failed" if resume_diagnostics else "succeeded",
            "node_count": len(plan.nodes),
            "edge_count": len(plan.edges),
            "diagnostic_codes": _diagnostic_codes(resume_diagnostics),
        },
    )
    await _emit_flow_event(
        event_listener,
        stream_session,
        EventType.FLOW_STARTED,
        plan,
        payload={"status": "started"},
        started=flow_started_at,
    )
    await _emit_flow_event_drafts(
        event_listener,
        stream_session,
        plan,
        event_drafts,
    )

    try:
        while ready:
            await _check_cancelled(cancellation_checker)
            batch: list[tuple[FlowNodePlan, Mapping[str, object]]] = []
            batch_outcomes: list[_NodeRunOutcome] = []
            batch_context = FlowRuntimeContext(
                inputs=input_values,
                node_outputs=node_outputs,
            )
            while ready and len(batch) < concurrency_limit:
                node_name = ready.popleft()
                queued.discard(node_name)
                node = node_map[node_name]
                started_at = monotonic()
                await _emit_flow_event(
                    event_listener,
                    stream_session,
                    EventType.FLOW_NODE_STARTED,
                    plan,
                    payload={
                        "node": node.name,
                        "status": "started",
                        "attempt": (
                            _node_trace_attempts(
                                trace,
                                node.name,
                            )
                            + 1
                        ),
                    },
                    started=started_at,
                )
                mapped_inputs, mapping_diagnostic = _node_inputs(
                    node,
                    batch_context,
                )
                trace = trace.with_node_state(node.name, FlowNodeState.READY)
                if mapping_diagnostic is not None:
                    batch_outcomes.append(
                        _NodeRunOutcome(
                            node=node,
                            state=FlowNodeState.FAILED,
                            route_kind=FlowEdgeKind.ERROR,
                            attempts=1,
                            diagnostics=(mapping_diagnostic,),
                            duration_ms=_elapsed_ms(started_at),
                        )
                    )
                    continue
                container_outcome = await _container_authorization_outcome(
                    plan,
                    node,
                    approval=container_approvals.get(node.name),
                    started_at=started_at,
                    event_listener=event_listener,
                    stream_session=stream_session,
                )
                if container_outcome is not None:
                    batch_outcomes.append(container_outcome)
                    continue
                pause_outcome = _human_review_pause_outcome(
                    node,
                    started_at,
                )
                if pause_outcome is not None:
                    batch_outcomes.append(pause_outcome)
                    continue
                trace = trace.with_node_state(
                    node.name,
                    FlowNodeState.RUNNING,
                    attempts=1,
                )
                batch.append((node, mapped_inputs))
            if batch:
                batch_outcomes.extend(
                    await gather(
                        *(
                            _run_plan_node(
                                node,
                                mapped_inputs,
                                runner,
                                batch_context,
                                cancellation_checker,
                                plan=plan,
                                event_listener=event_listener,
                                stream_session=stream_session,
                            )
                            for node, mapped_inputs in batch
                        )
                    )
                )
            for outcome in batch_outcomes:
                trace = trace.with_node_state(
                    outcome.node.name,
                    outcome.state,
                    attempts=outcome.attempts,
                    duration_ms=outcome.duration_ms,
                    diagnostics=outcome.diagnostics,
                )
                diagnostics.extend(outcome.diagnostics)
                await _emit_node_outcome_event(
                    event_listener,
                    stream_session,
                    plan,
                    outcome,
                )
                await _emit_container_result_events(
                    event_listener,
                    stream_session,
                    plan,
                    outcome,
                )
                if outcome.route_kind == FlowEdgeKind.SUCCESS:
                    node_outputs[outcome.node.name] = _node_output_mapping(
                        outcome.node,
                        outcome.output,
                    )
                if outcome.pause_token is not None:
                    pause_tokens[outcome.node.name] = outcome.pause_token
                    paused = True
                processed.add(outcome.node.name)
            for outcome in batch_outcomes:
                if outcome.state == FlowNodeState.PAUSED:
                    continue
                route_context = _route_context(
                    input_values,
                    batch_context.node_outputs,
                    outcome,
                )
                route_event_drafts: list[_FlowEventDraft] = []
                routed, trace, route_diagnostics = _route_from_node(
                    plan,
                    outcome.node.name,
                    outcome.route_kind,
                    route_context,
                    trace,
                    forced_target=outcome.exhausted_route,
                    forced_kind=outcome.exhausted_route_kind,
                    event_drafts=route_event_drafts,
                )
                diagnostics.extend(route_diagnostics)
                for target in routed:
                    trace, join_diagnostics = _queue_target_if_ready(
                        plan,
                        target,
                        ready=ready,
                        queued=queued,
                        processed=processed,
                        trace=trace,
                        event_drafts=route_event_drafts,
                    )
                    diagnostics.extend(join_diagnostics)
                await _emit_flow_event_drafts(
                    event_listener,
                    stream_session,
                    plan,
                    route_event_drafts,
                )
            if paused:
                break
    except CancelledError:
        finished_at = monotonic()
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_CANCELLED,
            plan,
            payload={"status": "cancelled"},
            started=flow_started_at,
            finished=finished_at,
        )
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_COMPLETED,
            plan,
            payload={"status": "cancelled"},
            started=flow_started_at,
            finished=finished_at,
        )
        raise

    if paused:
        finished_at = monotonic()
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_COMPLETED,
            plan,
            payload={"status": "paused"},
            started=flow_started_at,
            finished=finished_at,
        )
        return FlowPlanExecutionResult(
            outputs={},
            trace=trace,
            diagnostics=tuple(diagnostics),
            node_outputs=node_outputs,
            pause_tokens=pause_tokens,
        )
    skipped_nodes = tuple(
        node.node
        for node in trace.nodes
        if node.state == FlowNodeState.PENDING
    )
    trace = _mark_unscheduled_nodes_skipped(trace)
    for node_name in skipped_nodes:
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_NODE_SKIPPED,
            plan,
            payload={"node": node_name, "status": "skipped"},
        )
    outputs, output_diagnostics = _select_flow_outputs(
        plan,
        FlowRuntimeContext(inputs=input_values, node_outputs=node_outputs),
    )
    diagnostics.extend(output_diagnostics)
    if output_diagnostics:
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_OUTPUT_SELECTED,
            plan,
            payload={
                "status": "failed",
                "diagnostic_codes": _diagnostic_codes(output_diagnostics),
            },
        )
    else:
        for output_name in outputs:
            await _emit_flow_event(
                event_listener,
                stream_session,
                EventType.FLOW_OUTPUT_SELECTED,
                plan,
                payload={
                    "output_name": output_name,
                    "status": "selected",
                },
            )
    finished_at = monotonic()
    await _emit_flow_event(
        event_listener,
        stream_session,
        EventType.FLOW_COMPLETED,
        plan,
        payload={
            "status": "failed" if diagnostics else "succeeded",
            "diagnostic_codes": _diagnostic_codes(diagnostics),
        },
        started=flow_started_at,
        finished=finished_at,
    )
    return FlowPlanExecutionResult(
        outputs=outputs,
        trace=trace,
        diagnostics=tuple(diagnostics),
        node_outputs=node_outputs,
    )


def _resume_isolation_metadata_diagnostics(
    plan: FlowExecutionPlan,
    resume_trace: FlowExecutionTrace | None,
) -> tuple[FlowDiagnostic, ...]:
    if resume_trace is None:
        return ()
    expected = flow_resume_isolation_metadata(plan)
    actual = resume_trace.metadata
    if not expected:
        if FLOW_RESUME_ISOLATION_METADATA_KEY in actual:
            return (_stale_resume_isolation_diagnostic(),)
        return ()
    expected_isolation = expected.get(FLOW_RESUME_ISOLATION_METADATA_KEY)
    actual_isolation = actual.get(FLOW_RESUME_ISOLATION_METADATA_KEY)
    if not isinstance(expected_isolation, Mapping):
        return ()
    if not isinstance(actual_isolation, Mapping):
        return (_widened_resume_isolation_diagnostic(),)
    expected_nodes = _resume_isolation_node_names(expected_isolation)
    actual_nodes = _resume_isolation_node_names(actual_isolation)
    if expected_nodes is None or actual_nodes is None:
        return (_stale_resume_isolation_diagnostic(),)
    if expected_nodes - actual_nodes:
        return (_widened_resume_isolation_diagnostic(),)
    if actual_isolation != expected_isolation:
        return (_stale_resume_isolation_diagnostic(),)
    return ()


def _resume_isolation_node_names(
    value: Mapping[str, object],
) -> set[str] | None:
    nodes = value.get("nodes")
    if not isinstance(nodes, Mapping):
        return None
    names: set[str] = set()
    for node_name in nodes:
        if not isinstance(node_name, str) or not node_name.strip():
            return None
        names.add(node_name)
    return names


def _widened_resume_isolation_diagnostic() -> FlowDiagnostic:
    return _execution_diagnostic(
        code="flow.execution.isolation_resume_metadata_widened",
        path="trace.metadata.isolation",
        message="Flow resume isolation metadata is missing or narrower.",
        hint=(
            "Resume with metadata captured from the exact paused isolation "
            "plan."
        ),
    )


def _stale_resume_isolation_diagnostic() -> FlowDiagnostic:
    return _execution_diagnostic(
        code="flow.execution.isolation_resume_metadata_stale",
        path="trace.metadata.isolation",
        message="Flow resume isolation metadata does not match this plan.",
        hint=(
            "Recompile the original strict flow or restart after policy "
            "changes."
        ),
    )


def _initial_runtime_queue(
    plan: FlowExecutionPlan,
    inputs: Mapping[str, object],
    node_outputs: dict[str, Mapping[str, object]],
    resume_decisions: Mapping[str, Mapping[str, object]],
    trace: FlowExecutionTrace,
    *,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[
    deque[str],
    set[str],
    set[str],
    FlowExecutionTrace,
    tuple[FlowDiagnostic, ...],
    Mapping[str, ContainerApprovalRecord],
]:
    node_states = {node.node: node.state for node in trace.nodes}
    diagnostics: list[FlowDiagnostic] = []
    resume_routes: dict[str, FlowEdgeKind] = {}
    resumed_container_nodes: set[str] = set()
    container_approvals: dict[str, ContainerApprovalRecord] = {}
    for node_name, payload in resume_decisions.items():
        node = plan.node_map.get(node_name)
        if node is None:
            diagnostics.append(
                _execution_diagnostic(
                    code="flow.execution.unknown_resume_node",
                    path="resume_decisions",
                    message="Flow resume decision references an unknown node.",
                    hint="Resume a paused human review node from this plan.",
                )
            )
            continue
        diagnostic = _validate_resume_decision(node, payload, node_states)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
            trace = trace.with_node_state(
                node.name,
                FlowNodeState.FAILED,
                diagnostics=(diagnostic,),
            )
            _append_node_event_draft(
                event_drafts,
                EventType.FLOW_NODE_FAILED,
                node=node.name,
                status="failed",
                diagnostics=(diagnostic,),
            )
            continue
        if _is_container_review_node(node):
            if payload.get("decision") == "denied":
                diagnostic = _execution_diagnostic(
                    code="flow.execution.container_review_denied",
                    path=f"nodes.{node.name}.runtime.container",
                    message="Container review denied execution.",
                    hint="Change the container profile or approve the review.",
                )
                diagnostics.append(diagnostic)
                trace = trace.with_node_state(
                    node.name,
                    FlowNodeState.FAILED,
                    diagnostics=(diagnostic,),
                )
                _append_node_event_draft(
                    event_drafts,
                    EventType.FLOW_NODE_FAILED,
                    node=node.name,
                    status="failed",
                    diagnostics=(diagnostic,),
                )
                _append_container_event_draft(
                    event_drafts,
                    node=node,
                    event_type=ContainerAuditEventType.REVIEW_DECISION,
                    metadata={"decision": "denied"},
                )
                continue
            approval = _container_approval_from_resume(plan, node, payload)
            if approval is None:
                diagnostic = _execution_diagnostic(
                    code="flow.execution.invalid_container_approval",
                    path=f"nodes.{node.name}.approval",
                    message="Container review approval is invalid.",
                    hint=(
                        "Resume with a durable approval record for the "
                        "exact paused container plan."
                    ),
                )
                diagnostics.append(diagnostic)
                trace = trace.with_node_state(
                    node.name,
                    FlowNodeState.FAILED,
                    diagnostics=(diagnostic,),
                )
                _append_node_event_draft(
                    event_drafts,
                    EventType.FLOW_NODE_FAILED,
                    node=node.name,
                    status="failed",
                    diagnostics=(diagnostic,),
                )
                _append_container_event_draft(
                    event_drafts,
                    node=node,
                    event_type=ContainerAuditEventType.REVIEW_DECISION,
                    metadata={"decision": "invalid"},
                )
                continue
            container_approvals[node.name] = approval
            resumed_container_nodes.add(node.name)
            attempts = max(1, _node_trace_attempts(trace, node.name))
            _append_node_event_draft(
                event_drafts,
                EventType.FLOW_NODE_RESUMED,
                node=node.name,
                status="resumed",
                attempts=attempts,
            )
            _append_container_event_draft(
                event_drafts,
                node=node,
                event_type=ContainerAuditEventType.REVIEW_DECISION,
                metadata={"decision": "approved"},
            )
            continue
        node_outputs[node.name] = _node_output_mapping(node, payload)
        attempts = max(1, _node_trace_attempts(trace, node.name))
        trace = trace.with_node_state(
            node.name,
            FlowNodeState.SUCCEEDED,
            attempts=attempts,
        )
        _append_node_event_draft(
            event_drafts,
            EventType.FLOW_NODE_RESUMED,
            node=node.name,
            status="resumed",
            attempts=attempts,
        )
        resume_routes[node.name] = FlowEdgeKind.RESUME
    processed = {
        node.name
        for node in plan.nodes
        if node_states.get(node.name) == FlowNodeState.SUCCEEDED
        and node.name in node_outputs
    }
    processed.update(resume_routes)
    if not processed:
        if resumed_container_nodes:
            resumed_ready = deque(sorted(resumed_container_nodes))
            return (
                resumed_ready,
                set(resumed_container_nodes),
                set(),
                trace,
                tuple(diagnostics),
                MappingProxyType(container_approvals),
            )
        if resume_decisions:
            return (
                deque(),
                set(),
                set(),
                trace,
                tuple(diagnostics),
                MappingProxyType(container_approvals),
            )
        ready: deque[str] = deque((plan.entry_node,))
        return (
            ready,
            {plan.entry_node},
            set(),
            trace,
            tuple(diagnostics),
            MappingProxyType(container_approvals),
        )

    ready = deque[str]()
    queued: set[str] = set()
    context = FlowRuntimeContext(inputs=inputs, node_outputs=node_outputs)
    for source in sorted(processed, key=_plan_node_order(plan)):
        routed, trace, route_diagnostics = _route_from_node(
            plan,
            source,
            resume_routes.get(source, FlowEdgeKind.SUCCESS),
            context,
            trace,
            event_drafts=event_drafts,
        )
        diagnostics.extend(route_diagnostics)
        for target in routed:
            trace, join_diagnostics = _queue_target_if_ready(
                plan,
                target,
                ready=ready,
                queued=queued,
                processed=processed,
                trace=trace,
                event_drafts=event_drafts,
            )
            diagnostics.extend(join_diagnostics)
    for node_name in sorted(resumed_container_nodes):
        if node_name not in processed and node_name not in queued:
            ready.append(node_name)
            queued.add(node_name)
    return (
        ready,
        queued,
        processed,
        trace,
        tuple(diagnostics),
        MappingProxyType(container_approvals),
    )


def _node_trace_attempts(trace: FlowExecutionTrace, node_name: str) -> int:
    for node in trace.nodes:
        if node.node == node_name:
            return node.attempts
    return 0


def _validate_resume_decision(
    node: FlowNodePlan,
    payload: Mapping[str, object],
    node_states: Mapping[str, FlowNodeState],
) -> FlowDiagnostic | None:
    if _is_container_review_node(node):
        return _validate_container_resume_decision(node, payload, node_states)
    if node.kind != FlowNodeKind.HUMAN_REVIEW:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_node",
            path=f"nodes.{node.name}",
            message="Flow resume decision targets a non-review node.",
            hint="Resume only a paused human review node.",
        )
    if node_states.get(node.name) != FlowNodeState.PAUSED:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_state",
            path=f"nodes.{node.name}",
            message="Flow resume decision targets a node that is not paused.",
            hint="Resume a currently paused human review node.",
        )
    decision = payload.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_decision",
            path=f"nodes.{node.name}.decision",
            message="Flow resume decision is invalid.",
            hint="Provide a safe decision string from the review schema.",
        )
    allowed_decisions = node.config.get("allowed_decisions")
    if (
        isinstance(allowed_decisions, tuple | list)
        and decision not in allowed_decisions
    ):
        return _execution_diagnostic(
            code="flow.execution.unknown_resume_decision",
            path=f"nodes.{node.name}.decision",
            message="Flow resume decision is not allowed.",
            hint="Use one of the configured human review decisions.",
        )
    return _validate_resume_decision_schema(node, payload)


def _validate_container_resume_decision(
    node: FlowNodePlan,
    payload: Mapping[str, object],
    node_states: Mapping[str, FlowNodeState],
) -> FlowDiagnostic | None:
    if node_states.get(node.name) != FlowNodeState.PAUSED:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_state",
            path=f"nodes.{node.name}",
            message="Container review targets a node that is not paused.",
            hint="Resume a currently paused container review node.",
        )
    decision = payload.get("decision")
    if decision not in {"approved", "denied"}:
        return _execution_diagnostic(
            code="flow.execution.invalid_container_review_decision",
            path=f"nodes.{node.name}.decision",
            message="Container review decision is invalid.",
            hint="Use approved or denied.",
        )
    if decision == "approved":
        approval = payload.get("approval")
        if not isinstance(approval, Mapping):
            return _execution_diagnostic(
                code="flow.execution.missing_container_approval",
                path=f"nodes.{node.name}.approval",
                message="Container review approval record is missing.",
                hint=(
                    "Provide a durable approval record for this container "
                    "plan."
                ),
            )
    return None


def _is_container_review_node(node: FlowNodePlan) -> bool:
    return node.container is not None and node.container.enabled


def _container_approval_from_resume(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
    payload: Mapping[str, object],
) -> ContainerApprovalRecord | None:
    decision = payload.get("decision")
    if decision != "approved":
        return None
    approval = payload.get("approval")
    if not isinstance(approval, Mapping):
        return None
    try:
        return _container_approval_from_mapping(approval)
    except AssertionError:
        return None


def _container_approval_from_mapping(
    raw: Mapping[str, object],
) -> ContainerApprovalRecord:
    approved_triggers = raw.get("approved_triggers", ())
    assert isinstance(approved_triggers, list | tuple)
    return ContainerApprovalRecord(
        reviewer_identity=_required_resume_str(raw, "reviewer_identity"),
        review_mode=_required_resume_str(raw, "review_mode"),
        plan_fingerprint=_required_resume_str(raw, "plan_fingerprint"),
        scope_id=_required_resume_str(raw, "scope_id"),
        attempt_id=_optional_resume_str(raw, "attempt_id"),
        profile_name=_required_resume_str(raw, "profile_name"),
        policy_version=_required_resume_str(raw, "policy_version"),
        expires_at_seconds=_required_resume_int(raw, "expires_at_seconds"),
        approved_triggers=tuple(
            _required_sequence_str(trigger, "approved_triggers")
            for trigger in approved_triggers
        ),
    )


def _required_resume_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    assert isinstance(value, str) and value.strip()
    return value


def _optional_resume_str(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    assert isinstance(value, str) and value.strip()
    return value


def _required_resume_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _required_sequence_str(value: object, field_name: str) -> str:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must contain non-empty strings"
    return value


def _validate_resume_decision_schema(
    node: FlowNodePlan,
    payload: Mapping[str, object],
) -> FlowDiagnostic | None:
    schema = node.config.get("decision_schema")
    if schema is None:
        return None
    if not isinstance(schema, Mapping):
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_schema",
            path=f"nodes.{node.name}.config.decision_schema",
            message="Flow resume decision schema is invalid.",
            hint="Use a valid JSON Schema object for review decisions.",
        )
    adapter = _json_schema_adapter()
    if adapter is None:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_schema",
            path=f"nodes.{node.name}.config.decision_schema",
            message="Flow resume decision schema is invalid.",
            hint="Install JSON Schema validation support.",
        )
    schema_data = _json_compatible_mapping(
        cast(Mapping[object, object], schema)
    )
    payload_data = _json_compatible_mapping(
        cast(Mapping[object, object], payload)
    )
    try:
        adapter.validator_class.check_schema(schema_data)
        validator = adapter.validator_class(schema_data)
        validator.validate(payload_data)
    except adapter.schema_error:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_schema",
            path=f"nodes.{node.name}.config.decision_schema",
            message="Flow resume decision schema is invalid.",
            hint="Use a valid JSON Schema object for review decisions.",
        )
    except adapter.validation_error:
        return _execution_diagnostic(
            code="flow.execution.invalid_resume_payload",
            path=f"nodes.{node.name}.decision",
            message="Flow resume decision payload is invalid.",
            hint="Provide a decision payload matching the review schema.",
        )
    return None


def _human_review_pause_outcome(
    node: FlowNodePlan,
    started_at: float,
) -> _NodeRunOutcome | None:
    if node.kind != FlowNodeKind.HUMAN_REVIEW:
        return None
    return _NodeRunOutcome(
        node=node,
        state=FlowNodeState.PAUSED,
        route_kind=FlowEdgeKind.PAUSE,
        attempts=1,
        diagnostics=(),
        duration_ms=_elapsed_ms(started_at),
        pause_token=uuid4().hex,
    )


async def _container_authorization_outcome(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
    *,
    approval: ContainerApprovalRecord | None,
    started_at: float,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> _NodeRunOutcome | None:
    settings = node.container
    if settings is None or not settings.enabled:
        return None
    policy_plan = _container_policy_plan(plan, node)
    policy = ContainerPolicy(policy_version=settings.policy_version)
    try:
        decision = policy.authorize(
            policy_plan,
            approval=approval,
            now_seconds=int(time()),
        )
    except AssertionError:
        diagnostic = _execution_diagnostic(
            code="flow.execution.container_review_invalid",
            path=f"nodes.{node.name}.runtime.container",
            message="Container review approval is invalid.",
            hint=(
                "Resume with an approval for the exact paused container plan."
            ),
        )
        await _emit_container_event(
            event_listener,
            stream_session,
            plan,
            node,
            ContainerAuditEventType.REVIEW_DECISION,
            metadata={"decision": "invalid"},
        )
        return _NodeRunOutcome(
            node=node,
            state=FlowNodeState.FAILED,
            route_kind=FlowEdgeKind.ERROR,
            attempts=1,
            diagnostics=(diagnostic,),
            duration_ms=_elapsed_ms(started_at),
            failure_category="container_policy",
        )
    decision_type = cast(ContainerAuthorizationDecisionType, decision.decision)
    await _emit_container_event(
        event_listener,
        stream_session,
        plan,
        node,
        ContainerAuditEventType.POLICY_EVALUATION,
        metadata={
            "decision": decision_type.value,
            "code": decision.code,
        },
    )
    if decision_type is ContainerAuthorizationDecisionType.ALLOW:
        if approval is not None:
            await _emit_container_event(
                event_listener,
                stream_session,
                plan,
                node,
                ContainerAuditEventType.REVIEW_DECISION,
                metadata={"decision": "approved"},
            )
        await _emit_container_event(
            event_listener,
            stream_session,
            plan,
            node,
            ContainerAuditEventType.BACKEND_SELECTION,
            metadata={"backend": str(settings.backend)},
        )
        await _emit_container_event(
            event_listener,
            stream_session,
            plan,
            node,
            ContainerAuditEventType.CONTAINER_CREATE,
        )
        await _emit_container_event(
            event_listener,
            stream_session,
            plan,
            node,
            ContainerAuditEventType.CONTAINER_START,
        )
        return None
    if decision_type is ContainerAuthorizationDecisionType.REQUIRES_REVIEW:
        await _emit_container_event(
            event_listener,
            stream_session,
            plan,
            node,
            ContainerAuditEventType.REVIEW_REQUEST,
            metadata={"decision": "requires_review"},
        )
        return _NodeRunOutcome(
            node=node,
            state=FlowNodeState.PAUSED,
            route_kind=FlowEdgeKind.PAUSE,
            attempts=1,
            diagnostics=(),
            duration_ms=_elapsed_ms(started_at),
            pause_token=uuid4().hex,
        )
    diagnostic = _execution_diagnostic(
        code="flow.execution.container_policy_denied",
        path=f"nodes.{node.name}.runtime.container",
        message="Container policy denied node execution.",
        hint=decision.explanation,
    )
    await _emit_container_event(
        event_listener,
        stream_session,
        plan,
        node,
        ContainerAuditEventType.DENIAL,
        metadata={"decision": "denied", "code": decision.code},
    )
    return _NodeRunOutcome(
        node=node,
        state=FlowNodeState.FAILED,
        route_kind=FlowEdgeKind.ERROR,
        attempts=1,
        diagnostics=(diagnostic,),
        duration_ms=_elapsed_ms(started_at),
        failure_category="container_policy",
    )


def _container_policy_plan(
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
) -> ContainerPolicyPlan:
    settings = node.container
    assert isinstance(settings, ContainerEffectiveSettings)
    return ContainerPolicyPlan(
        effective_settings=settings,
        context=ContainerPolicyContext(
            surface=ContainerReviewSurface.STRICT_FLOW,
            scope_id=f"{_flow_event_id(plan)}:{node.name}",
        ),
        command_fingerprint=flow_node_container_fingerprint(plan, node),
    )


def _runtime_envelope_unavailable_diagnostic() -> FlowDiagnostic:
    return _execution_diagnostic(
        code="flow.execution.container_runtime_envelope_unavailable",
        path="runtime.container.envelope",
        message=(
            "Flow runtime envelope execution is not available in this runner."
        ),
        hint=(
            "Use an operator runtime that can enforce the flow runtime "
            "envelope."
        ),
    )


def _json_compatible_mapping(
    value: Mapping[object, object],
) -> dict[str, object]:
    return {
        key: _json_compatible_value(item)
        for key, item in value.items()
        if isinstance(key, str) and key.strip()
    }


def _json_compatible_value(value: object) -> object:
    if value is None or isinstance(value, bool | str | int | float):
        return value
    if isinstance(value, Mapping):
        return _json_compatible_mapping(value)
    if isinstance(value, list | tuple):
        return [_json_compatible_value(item) for item in value]
    return {"type": type(value).__name__}


def _json_schema_adapter() -> _JsonSchemaAdapter | None:
    try:
        module = import_module("jsonschema")
    except (ImportError, ValueError):
        return None
    return _json_schema_adapter_from_module(module)


def _json_schema_adapter_from_module(
    module: ModuleType,
) -> _JsonSchemaAdapter | None:
    validator_class = getattr(module, "Draft202012Validator", None)
    schema_error = _exception_class(module, "SchemaError")
    validation_error = _exception_class(module, "ValidationError")
    if (
        validator_class is None
        or schema_error is None
        or validation_error is None
    ):
        return None
    return _JsonSchemaAdapter(
        validator_class=cast(_JsonSchemaValidatorClass, validator_class),
        schema_error=schema_error,
        validation_error=validation_error,
    )


def _exception_class(
    module: ModuleType,
    name: str,
) -> type[Exception] | None:
    value = getattr(module, name, None)
    if isinstance(value, type) and issubclass(value, Exception):
        return value
    return None


def _plan_node_order(plan: FlowExecutionPlan) -> Callable[[str], int]:
    order = {node.name: index for index, node in enumerate(plan.nodes)}
    return lambda node: order[node]


def _route_context(
    inputs: Mapping[str, object],
    previous_node_outputs: Mapping[str, Mapping[str, object]],
    outcome: _NodeRunOutcome,
) -> FlowRuntimeContext:
    if outcome.route_kind != FlowEdgeKind.SUCCESS:
        return FlowRuntimeContext(
            inputs=inputs,
            node_outputs=previous_node_outputs,
        )
    node_outputs = dict(previous_node_outputs)
    node_outputs[outcome.node.name] = _node_output_mapping(
        outcome.node,
        outcome.output,
    )
    return FlowRuntimeContext(inputs=inputs, node_outputs=node_outputs)


def _node_inputs(
    node: FlowNodePlan,
    context: FlowRuntimeContext,
) -> tuple[Mapping[str, object], FlowDiagnostic | None]:
    try:
        return evaluate_flow_node_mappings(node, context), None
    except FlowRuntimeEvaluationError as error:
        return {}, _execution_diagnostic(
            code=error.code,
            path=f"nodes.{node.name}.mappings",
            message="Flow node input mapping failed.",
            hint="Check that every selected runtime value is available.",
        )


async def _run_plan_node(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    context: FlowRuntimeContext,
    cancellation_checker: CancellationChecker | None,
    *,
    plan: FlowExecutionPlan,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> _NodeRunOutcome:
    if node.loop is not None:
        return await _run_loop_plan_node(
            node,
            inputs,
            runner,
            context,
            cancellation_checker,
            plan=plan,
            event_listener=event_listener,
            stream_session=stream_session,
        )
    return await _run_plan_node_attempts(
        node,
        inputs,
        runner,
        cancellation_checker,
        plan=plan,
        event_listener=event_listener,
        stream_session=stream_session,
    )


async def _run_plan_node_attempts(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    cancellation_checker: CancellationChecker | None,
    *,
    plan: FlowExecutionPlan,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> _NodeRunOutcome:
    started_at = monotonic()
    attempts = 0
    first_failure: _NodeRunOutcome | None = None
    while True:
        attempts += 1
        outcome = await _run_plan_node_once(
            node,
            inputs,
            runner,
            cancellation_checker,
            event_listener=event_listener,
            stream_session=stream_session,
        )
        if outcome.route_kind == FlowEdgeKind.SUCCESS:
            return _NodeRunOutcome(
                node=node,
                state=outcome.state,
                route_kind=outcome.route_kind,
                attempts=attempts,
                diagnostics=(),
                duration_ms=_elapsed_ms(started_at),
                output=outcome.output,
            )
        if first_failure is None:
            first_failure = outcome
        if not _should_retry_node(node, outcome, attempts):
            return _NodeRunOutcome(
                node=node,
                state=outcome.state,
                route_kind=outcome.route_kind,
                attempts=attempts,
                diagnostics=first_failure.diagnostics,
                duration_ms=_elapsed_ms(started_at),
                failure_category=outcome.failure_category,
                exhausted_route=(
                    node.retry.exhausted_route
                    if node.retry is not None
                    and attempts >= node.retry.max_attempts
                    else None
                ),
                exhausted_route_kind=(
                    outcome.route_kind
                    if node.retry is not None
                    and attempts >= node.retry.max_attempts
                    else None
                ),
            )
        await _check_cancelled(cancellation_checker)
        delay_seconds = _retry_delay_seconds(node.retry, attempts)
        await _emit_flow_event(
            event_listener,
            stream_session,
            EventType.FLOW_NODE_RETRYING,
            plan,
            payload={
                "node": node.name,
                "status": "retrying",
                "attempt": attempts,
                "retry_delay_seconds": delay_seconds,
                "diagnostic_codes": _diagnostic_codes(outcome.diagnostics),
            },
        )
        if delay_seconds > 0:
            await sleep(delay_seconds)
            await _check_cancelled(cancellation_checker)


async def _run_loop_plan_node(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    base_context: FlowRuntimeContext,
    cancellation_checker: CancellationChecker | None,
    *,
    plan: FlowExecutionPlan,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> _NodeRunOutcome:
    assert node.loop is not None
    started_at = monotonic()
    attempts = 0
    while True:
        outcome = await _run_plan_node_attempts(
            node,
            inputs,
            runner,
            cancellation_checker,
            plan=plan,
            event_listener=event_listener,
            stream_session=stream_session,
        )
        attempts += outcome.attempts
        if outcome.route_kind != FlowEdgeKind.SUCCESS:
            return _NodeRunOutcome(
                node=node,
                state=outcome.state,
                route_kind=outcome.route_kind,
                attempts=attempts,
                diagnostics=outcome.diagnostics,
                duration_ms=_elapsed_ms(started_at),
                failure_category=outcome.failure_category,
                exhausted_route=outcome.exhausted_route,
                exhausted_route_kind=outcome.exhausted_route_kind,
            )
        node_outputs = dict(base_context.node_outputs)
        node_outputs[node.name] = _node_output_mapping(node, outcome.output)
        context = FlowRuntimeContext(
            inputs=base_context.inputs,
            node_outputs=node_outputs,
        )
        loop_event_drafts: list[_FlowEventDraft] = []
        selected_output, diagnostic = _loop_selected_output(
            node,
            context,
            event_drafts=loop_event_drafts,
        )
        await _emit_flow_event_drafts(
            event_listener,
            stream_session,
            plan,
            loop_event_drafts,
        )
        if diagnostic is not None:
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.FAILED,
                route_kind=FlowEdgeKind.ERROR,
                attempts=attempts,
                diagnostics=(diagnostic,),
                duration_ms=_elapsed_ms(started_at),
                failure_category="validation",
            )
        if selected_output is not _MISSING:
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.SUCCEEDED,
                route_kind=FlowEdgeKind.SUCCESS,
                attempts=attempts,
                diagnostics=(),
                duration_ms=_elapsed_ms(started_at),
                output=selected_output,
            )
        if _loop_limit_reached(node, attempts, started_at):
            diagnostic = _execution_diagnostic(
                code="flow.execution.loop_limit_reached",
                path=f"nodes.{node.name}.loop_policy",
                message="Flow loop limit was reached.",
                hint="Inspect the configured loop limit route.",
            )
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.FAILED,
                route_kind=FlowEdgeKind.ERROR,
                attempts=attempts,
                diagnostics=(diagnostic,),
                duration_ms=_elapsed_ms(started_at),
                failure_category="loop_limit",
                exhausted_route=node.loop.limit_route,
            )
        await _check_cancelled(cancellation_checker)


def _loop_selected_output(
    node: FlowNodePlan,
    context: FlowRuntimeContext,
    *,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[object, FlowDiagnostic | None]:
    assert node.loop is not None
    try:
        exit_matched = evaluate_flow_condition_plan(
            node.loop.exit_condition,
            context,
        )
        _append_condition_event_draft(
            event_drafts,
            node=node.name,
            status="matched" if exit_matched else "unmatched",
            matched=exit_matched,
        )
        if exit_matched:
            return (
                evaluate_flow_selector(node.loop.output_selector, context),
                None,
            )
        continue_matched = evaluate_flow_condition_plan(
            node.loop.continue_condition,
            context,
        )
        _append_condition_event_draft(
            event_drafts,
            node=node.name,
            status="matched" if continue_matched else "unmatched",
            matched=continue_matched,
        )
        if continue_matched:
            return _MISSING, None
    except FlowRuntimeEvaluationError as error:
        diagnostic = _execution_diagnostic(
            code=error.code,
            path=f"nodes.{node.name}.loop_policy",
            message="Flow loop condition evaluation failed.",
            hint="Check loop selectors and node output contracts.",
        )
        _append_condition_event_draft(
            event_drafts,
            node=node.name,
            status="failed",
            matched=False,
            diagnostics=(diagnostic,),
        )
        return (
            _MISSING,
            diagnostic,
        )
    return (
        _MISSING,
        _execution_diagnostic(
            code="flow.execution.loop_condition_unmatched",
            path=f"nodes.{node.name}.loop_policy",
            message="Flow loop conditions did not match.",
            hint="Make loop continue and exit conditions exhaustive.",
        ),
    )


def _loop_limit_reached(
    node: FlowNodePlan,
    attempts: int,
    started_at: float,
) -> bool:
    assert node.loop is not None
    if (
        node.loop.max_iterations is not None
        and attempts >= node.loop.max_iterations
    ):
        return True
    return (
        node.loop.max_elapsed_seconds is not None
        and monotonic() - started_at >= node.loop.max_elapsed_seconds
    )


def _elapsed_ms(started_at: float) -> float:
    return (monotonic() - started_at) * 1000


def _elapsed_ms_between(started_at: float, finished_at: float) -> float:
    return (finished_at - started_at) * 1000


async def _emit_flow_event(
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
    event_type: EventType,
    plan: FlowExecutionPlan,
    *,
    payload: Mapping[str, object] | None = None,
    started: float | None = None,
    finished: float | None = None,
) -> None:
    assert isinstance(stream_session, FlowStreamSession)
    if event_listener is None:
        return
    item = canonical_flow_item(
        stream_session=stream_session,
        event_type=event_type,
        payload=_flow_event_payload(plan, payload or {}),
        started=started,
        finished=finished,
    )
    result = event_listener(item)
    if result is not None:
        await result


async def _emit_flow_event_drafts(
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
    plan: FlowExecutionPlan,
    drafts: list[_FlowEventDraft],
) -> None:
    for draft in drafts:
        await _emit_flow_event(
            event_listener,
            stream_session,
            draft.type,
            plan,
            payload=draft.payload,
            started=draft.started,
            finished=draft.finished,
        )


async def _emit_container_event(
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
    plan: FlowExecutionPlan,
    node: FlowNodePlan,
    event_type: ContainerAuditEventType,
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    if node.container is None:
        return
    event = ContainerAuditEvent(
        event_type=event_type,
        scope=node.container.scope,
        profile_name=node.container.profile_name,
        policy_version=node.container.policy_version,
        metadata=metadata or {},
    )
    await _emit_flow_event(
        event_listener,
        stream_session,
        EventType.FLOW_CONTAINER_EVENT,
        plan,
        payload=_container_event_payload(node, event),
    )


async def _emit_container_result_events(
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
    plan: FlowExecutionPlan,
    outcome: _NodeRunOutcome,
) -> None:
    if outcome.node.container is None or outcome.state == FlowNodeState.PAUSED:
        return
    status = (
        ContainerResultStatus.COMPLETED
        if outcome.state == FlowNodeState.SUCCEEDED
        else (
            ContainerResultStatus.CANCELLED
            if outcome.state == FlowNodeState.CANCELLED
            else ContainerResultStatus.FAILED
        )
    )
    await _emit_container_event(
        event_listener,
        stream_session,
        plan,
        outcome.node,
        ContainerAuditEventType.RESULT_RECORDED,
        metadata={"status": status.value},
    )
    await _emit_container_event(
        event_listener,
        stream_session,
        plan,
        outcome.node,
        ContainerAuditEventType.CLEANUP,
        metadata={"status": "completed"},
    )


async def _emit_node_outcome_event(
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
    plan: FlowExecutionPlan,
    outcome: _NodeRunOutcome,
) -> None:
    event_type = _node_outcome_event_type(outcome.state)
    if event_type is None:
        return
    payload: dict[str, object] = {
        "node": outcome.node.name,
        "status": outcome.state.value,
        "attempts": outcome.attempts,
        "route_kind": outcome.route_kind.value,
        "diagnostic_codes": _diagnostic_codes(outcome.diagnostics),
    }
    if outcome.duration_ms is not None:
        payload["duration_ms"] = outcome.duration_ms
    await _emit_flow_event(
        event_listener,
        stream_session,
        event_type,
        plan,
        payload=payload,
    )


def _node_outcome_event_type(
    state: FlowNodeState,
) -> EventType | None:
    match state:
        case FlowNodeState.SUCCEEDED:
            return EventType.FLOW_NODE_COMPLETED
        case FlowNodeState.FAILED:
            return EventType.FLOW_NODE_FAILED
        case FlowNodeState.CANCELLED:
            return EventType.FLOW_NODE_CANCELLED
        case FlowNodeState.PAUSED:
            return EventType.FLOW_NODE_PAUSED
        case _:
            return None


def _flow_event_payload(
    plan: FlowExecutionPlan,
    payload: Mapping[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "flow_id": _flow_event_id(plan),
        "flow_name": plan.name,
    }
    if plan.version is not None:
        value["flow_version"] = plan.version
    if plan.revision is not None:
        value["flow_revision"] = plan.revision
    for key, item in payload.items():
        assert isinstance(key, str) and key.strip()
        value[key] = _flow_event_value(item)
    return value


def _flow_event_id(plan: FlowExecutionPlan) -> str:
    identifier = plan.name
    if plan.version is not None:
        identifier = f"{identifier}@{plan.version}"
    if plan.revision is not None:
        identifier = f"{identifier}#{plan.revision}"
    return identifier


def _flow_event_value(value: object) -> object:
    if value is None or isinstance(value, bool | str | int | float):
        return value
    if isinstance(value, Mapping):
        return {
            key: _flow_event_value(item)
            for key, item in value.items()
            if isinstance(key, str) and key.strip()
        }
    if isinstance(value, list | tuple):
        return tuple(_flow_event_value(item) for item in value)
    return {"type": type(value).__name__}


def _diagnostic_codes(
    diagnostics: tuple[FlowDiagnostic, ...] | list[FlowDiagnostic],
) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in diagnostics)


def _append_node_event_draft(
    drafts: list[_FlowEventDraft] | None,
    event_type: EventType,
    *,
    node: str,
    status: str,
    attempts: int | None = None,
    diagnostics: tuple[FlowDiagnostic, ...] = (),
) -> None:
    if drafts is None:
        return
    payload: dict[str, object] = {
        "node": node,
        "status": status,
        "diagnostic_codes": _diagnostic_codes(diagnostics),
    }
    if attempts is not None:
        payload["attempts"] = attempts
    drafts.append(_FlowEventDraft(type=event_type, payload=payload))


def _append_container_event_draft(
    drafts: list[_FlowEventDraft] | None,
    *,
    node: FlowNodePlan,
    event_type: ContainerAuditEventType,
    metadata: Mapping[str, str] | None = None,
) -> None:
    if drafts is None or node.container is None:
        return
    event = ContainerAuditEvent(
        event_type=event_type,
        scope=node.container.scope,
        profile_name=node.container.profile_name,
        policy_version=node.container.policy_version,
        metadata=metadata or {},
    )
    drafts.append(
        _FlowEventDraft(
            type=EventType.FLOW_CONTAINER_EVENT,
            payload=_container_event_payload(node, event),
        )
    )


def _container_event_payload(
    node: FlowNodePlan,
    event: ContainerAuditEvent,
) -> dict[str, object]:
    serialized = event.to_dict()
    metadata = serialized["metadata"]
    assert isinstance(metadata, Mapping)
    payload: dict[str, object] = {
        "node": node.name,
        "container_event": serialized["event_type"],
        "scope": serialized["scope"],
        "profile_name": serialized["profile_name"],
        "policy_version": serialized["policy_version"],
        "metadata": dict(metadata),
        "status": str(metadata.get("status", serialized["event_type"])),
    }
    decision = metadata.get("decision")
    if isinstance(decision, str) and decision.strip():
        payload["decision"] = decision
    backend = metadata.get("backend")
    if isinstance(backend, str) and backend.strip():
        payload["backend"] = backend
    return payload


def _append_condition_event_draft(
    drafts: list[_FlowEventDraft] | None,
    *,
    status: str,
    matched: bool,
    edge: FlowEdgePlan | None = None,
    node: str | None = None,
    diagnostics: tuple[FlowDiagnostic, ...] = (),
    started: float | None = None,
    finished: float | None = None,
) -> None:
    if drafts is None:
        return
    payload: dict[str, object] = {
        "status": status,
        "matched": matched,
        "diagnostic_codes": _diagnostic_codes(diagnostics),
    }
    if edge is not None:
        payload.update(_edge_event_payload(edge))
    if node is not None:
        payload["node"] = node
    drafts.append(
        _FlowEventDraft(
            type=EventType.FLOW_CONDITION_EVALUATED,
            payload=payload,
            started=started,
            finished=finished,
        )
    )


def _append_edge_eligible_event_draft(
    drafts: list[_FlowEventDraft] | None,
    *,
    edge: FlowEdgePlan,
    status: str,
    eligible: bool,
    diagnostics: tuple[FlowDiagnostic, ...] = (),
    started: float | None = None,
    finished: float | None = None,
) -> None:
    if drafts is None:
        return
    payload = _edge_event_payload(edge)
    payload.update(
        {
            "status": status,
            "eligible": eligible,
            "diagnostic_codes": _diagnostic_codes(diagnostics),
        }
    )
    drafts.append(
        _FlowEventDraft(
            type=EventType.FLOW_EDGE_ELIGIBLE,
            payload=payload,
            started=started,
            finished=finished,
        )
    )


def _append_edge_routed_event_draft(
    drafts: list[_FlowEventDraft] | None,
    *,
    edge: FlowEdgePlan,
    status: str,
) -> None:
    if drafts is None:
        return
    payload = _edge_event_payload(edge)
    payload["status"] = status
    drafts.append(
        _FlowEventDraft(
            type=EventType.FLOW_EDGE_ROUTED,
            payload=payload,
        )
    )


def _append_join_event_draft(
    drafts: list[_FlowEventDraft] | None,
    *,
    node: str,
    ready: bool,
    status: str,
    diagnostics: tuple[FlowDiagnostic, ...] = (),
) -> None:
    if drafts is None:
        return
    drafts.append(
        _FlowEventDraft(
            type=EventType.FLOW_JOIN_READY,
            payload={
                "node": node,
                "ready": ready,
                "status": status,
                "diagnostic_codes": _diagnostic_codes(diagnostics),
            },
        )
    )


def _edge_event_payload(edge: FlowEdgePlan) -> dict[str, object]:
    return {
        "edge_index": edge.index,
        "edge_kind": edge.kind.value,
        "source": edge.source,
        "target": edge.target,
    }


async def _run_plan_node_once(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    cancellation_checker: CancellationChecker | None,
    *,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> _NodeRunOutcome:
    node_event_listener = _node_scoped_stream_listener(
        node.name,
        event_listener,
        stream_session,
    )
    token = _FLOW_EXECUTION_OPTIONS.set(
        _FlowExecutionOptions(
            cancellation_checker=cancellation_checker,
            event_listener=node_event_listener,
            stream_session=stream_session,
        )
    )
    try:
        try:
            output = runner(node, inputs)
            if isawaitable(output):
                if node.timeout is None:
                    output = await output
                else:
                    output = await wait_for(
                        output,
                        timeout=node.timeout.per_attempt_seconds,
                    )
        except CancelledError:
            diagnostic = _execution_diagnostic(
                code="flow.execution.node_cancelled",
                path=f"nodes.{node.name}",
                message="Flow node was cancelled.",
                hint="Inspect cancellation routes for this node.",
            )
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.CANCELLED,
                route_kind=FlowEdgeKind.CANCELLATION,
                attempts=1,
                diagnostics=(diagnostic,),
                failure_category="cancellation",
            )
        except FlowNodeExecutionError as error:
            diagnostic = _execution_diagnostic(
                code=error.code,
                path=f"nodes.{node.name}",
                message=error.safe_message,
                hint=error.hint,
            )
            return _NodeRunOutcome(
                node=node,
                state=(
                    FlowNodeState.CANCELLED
                    if error.route_kind == FlowEdgeKind.CANCELLATION
                    else FlowNodeState.FAILED
                ),
                route_kind=error.route_kind,
                attempts=1,
                diagnostics=(diagnostic,),
                failure_category=error.failure_category,
            )
        except TimeoutError:
            diagnostic = _execution_diagnostic(
                code="flow.execution.node_timeout",
                path=f"nodes.{node.name}",
                message="Flow node timed out.",
                hint="Inspect timeout routes for this node.",
            )
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.FAILED,
                route_kind=FlowEdgeKind.TIMEOUT,
                attempts=1,
                diagnostics=(diagnostic,),
                failure_category="timeout",
            )
        except Exception:
            diagnostic = _execution_diagnostic(
                code="flow.execution.node_failed",
                path=f"nodes.{node.name}",
                message="Flow node failed.",
                hint="Inspect error routes for this node.",
            )
            return _NodeRunOutcome(
                node=node,
                state=FlowNodeState.FAILED,
                route_kind=FlowEdgeKind.ERROR,
                attempts=1,
                diagnostics=(diagnostic,),
                failure_category="error",
            )
        return _NodeRunOutcome(
            node=node,
            state=FlowNodeState.SUCCEEDED,
            route_kind=FlowEdgeKind.SUCCESS,
            attempts=1,
            diagnostics=(),
            output=output,
        )
    finally:
        _FLOW_EXECUTION_OPTIONS.reset(token)


def _node_scoped_stream_listener(
    node_name: str,
    event_listener: FlowStreamListener | None,
    stream_session: FlowStreamSession,
) -> FlowStreamListener | None:
    assert_non_empty_string(node_name, "node_name")
    assert isinstance(stream_session, FlowStreamSession)
    if event_listener is None:
        return None
    assert callable(event_listener)

    def observe(item: CanonicalStreamItem) -> Awaitable[None] | None:
        assert isinstance(item, CanonicalStreamItem)
        assert item.stream_session_id == stream_session.stream_session_id
        assert item.run_id == stream_session.run_id
        assert item.turn_id == stream_session.turn_id
        assert isinstance(item.data, Mapping)
        data = dict(item.data)
        data.setdefault("flow_node", node_name)
        metadata = dict(item.metadata)
        metadata["parent_node_id"] = node_name
        if item.correlation.node_id is not None:
            metadata.setdefault("child_node_id", item.correlation.node_id)
        correlation = replace(
            item.correlation,
            node_id=item.correlation.node_id or node_name,
        )
        return event_listener(
            replace(
                item,
                correlation=correlation,
                data=data,
                metadata=metadata,
            )
        )

    return observe


def _should_retry_node(
    node: FlowNodePlan,
    outcome: _NodeRunOutcome,
    attempts: int,
) -> bool:
    if node.retry is None or attempts >= node.retry.max_attempts:
        return False
    if outcome.route_kind == FlowEdgeKind.CANCELLATION:
        return False
    category = outcome.failure_category or "error"
    if category in node.retry.non_retryable_categories:
        return False
    if node.retry.retryable_categories:
        return category in node.retry.retryable_categories
    return category != "validation"


def _retry_delay_seconds(
    retry: FlowRetryPlan | None,
    failed_attempts: int,
) -> float:
    if retry is None or retry.backoff == FlowRetryBackoffStrategy.NONE:
        return 0
    initial = float(retry.initial_delay_seconds or 0)
    match retry.backoff:
        case FlowRetryBackoffStrategy.CONSTANT:
            delay = initial
        case FlowRetryBackoffStrategy.LINEAR:
            delay = initial * failed_attempts
        case FlowRetryBackoffStrategy.EXPONENTIAL:
            delay = initial * (2 ** (failed_attempts - 1))
        case _:
            raise FlowRuntimeEvaluationError(
                "flow.execution.unsupported_retry_backoff"
            )
    if retry.max_delay_seconds is not None:
        delay = min(delay, float(retry.max_delay_seconds))
    return delay


def _node_output_mapping(
    node: FlowNodePlan,
    output: object,
) -> Mapping[str, object]:
    if _preserve_tool_envelope_output(node, output):
        assert isinstance(output, Mapping)
        return _freeze_mapping(output)
    names = tuple(
        contract.name
        for contract in node.output_contracts
        if contract.name is not None
    )
    if (
        names
        and isinstance(output, Mapping)
        and all(name in output for name in names)
    ):
        return _freeze_mapping({name: output[name] for name in names})
    if len(names) == 1:
        return _freeze_mapping({names[0]: output})
    if isinstance(output, Mapping):
        return _freeze_mapping(output)
    return _freeze_mapping({"result": output})


def _preserve_tool_envelope_output(
    node: FlowNodePlan,
    output: object,
) -> bool:
    if node.kind != FlowNodeKind.TOOL or not isinstance(output, Mapping):
        return False
    return node.config.get("output_mode", "raw") == "envelope"


def _route_from_node(
    plan: FlowExecutionPlan,
    source: str,
    route_kind: FlowEdgeKind,
    context: FlowRuntimeContext,
    trace: FlowExecutionTrace,
    *,
    forced_target: str | None = None,
    forced_kind: FlowEdgeKind | None = None,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[tuple[str, ...], FlowExecutionTrace, tuple[FlowDiagnostic, ...]]:
    diagnostics: list[FlowDiagnostic] = []
    routed: list[str] = []
    selected, trace, route_diagnostics = _select_routes(
        plan,
        source,
        route_kind,
        context,
        trace,
        forced_target=forced_target,
        forced_kind=forced_kind,
        event_drafts=event_drafts,
    )
    routed.extend(edge.target for edge in selected)
    diagnostics.extend(route_diagnostics)
    if route_kind != FlowEdgeKind.SUCCESS and not selected:
        diagnostics.append(
            _execution_diagnostic(
                code="flow.execution.missing_failure_route",
                path=f"nodes.{source}",
                message="Flow node failure has no eligible route.",
                hint="Add an explicit failure route for this node.",
            )
        )
    if route_kind != FlowEdgeKind.FINALLY:
        selected, trace, route_diagnostics = _select_routes(
            plan,
            source,
            FlowEdgeKind.FINALLY,
            context,
            trace,
            event_drafts=event_drafts,
        )
        routed.extend(edge.target for edge in selected)
        diagnostics.extend(route_diagnostics)
    return tuple(routed), trace, tuple(diagnostics)


def _queue_target_if_ready(
    plan: FlowExecutionPlan,
    target: str,
    *,
    ready: deque[str],
    queued: set[str],
    processed: set[str],
    trace: FlowExecutionTrace,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[FlowExecutionTrace, tuple[FlowDiagnostic, ...]]:
    if target in processed or target in queued:
        return trace, ()
    node = plan.node_map[target]
    if node.join is None:
        ready.append(target)
        queued.add(target)
        return trace, ()
    ready_for_join, failed_diagnostic = _join_ready(
        plan,
        target,
        trace,
    )
    if failed_diagnostic is not None:
        processed.add(target)
        _append_join_event_draft(
            event_drafts,
            node=target,
            ready=False,
            status="failed",
            diagnostics=(failed_diagnostic,),
        )
        return (
            trace.with_node_state(
                target,
                FlowNodeState.FAILED,
                diagnostics=(failed_diagnostic,),
            ),
            (failed_diagnostic,),
        )
    if ready_for_join:
        ready.append(target)
        queued.add(target)
    _append_join_event_draft(
        event_drafts,
        node=target,
        ready=ready_for_join,
        status="ready" if ready_for_join else "waiting",
    )
    return trace, ()


def _join_ready(
    plan: FlowExecutionPlan,
    target: str,
    trace: FlowExecutionTrace,
) -> tuple[bool, FlowDiagnostic | None]:
    node = plan.node_map[target]
    assert node.join is not None
    inbound_edges = plan.edges_by_target.get(target, ())
    edge_states = {edge.index: edge.state for edge in trace.edges}
    node_states = {node.node: node.state for node in trace.nodes}
    success_count = 0
    failure_count = 0
    done_count = 0
    for edge in inbound_edges:
        edge_state = edge_states[edge.index]
        source_state = node_states[edge.source]
        if edge_state != FlowEdgeState.PENDING:
            done_count += 1
        if edge_state == FlowEdgeState.TAKEN:
            if source_state == FlowNodeState.SUCCEEDED:
                success_count += 1
            elif source_state in (
                FlowNodeState.FAILED,
                FlowNodeState.CANCELLED,
            ):
                failure_count += 1
        elif edge_state == FlowEdgeState.FAILED:
            failure_count += 1
    inbound_count = len(inbound_edges)
    match node.join.type:
        case FlowJoinPolicyType.ALL_SUCCESS:
            return success_count == inbound_count, None
        case FlowJoinPolicyType.ALL_DONE | FlowJoinPolicyType.COLLECT:
            return done_count == inbound_count, None
        case FlowJoinPolicyType.ANY_SUCCESS | FlowJoinPolicyType.FIRST_SUCCESS:
            return success_count > 0, None
        case FlowJoinPolicyType.QUORUM:
            assert node.join.quorum is not None
            return success_count >= node.join.quorum, None
        case FlowJoinPolicyType.FAIL_FAST:
            if failure_count:
                return False, _execution_diagnostic(
                    code="flow.execution.join_failed",
                    path=f"nodes.{target}.join_policy",
                    message="Flow join failed before all inputs succeeded.",
                    hint="Inspect inbound routes and join policy.",
                )
            return success_count == inbound_count, None
    return False, _execution_diagnostic(
        code="flow.execution.unsupported_join_policy",
        path=f"nodes.{target}.join_policy",
        message="Flow join policy is unsupported.",
        hint="Use a supported join policy for this node.",
    )


def _select_routes(
    plan: FlowExecutionPlan,
    source: str,
    kind: FlowEdgeKind,
    context: FlowRuntimeContext,
    trace: FlowExecutionTrace,
    *,
    forced_target: str | None = None,
    forced_kind: FlowEdgeKind | None = None,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[
    tuple[FlowEdgePlan, ...],
    FlowExecutionTrace,
    tuple[FlowDiagnostic, ...],
]:
    if forced_target is not None:
        return _select_forced_route(
            plan,
            source,
            forced_target,
            kind,
            trace,
            forced_kind=forced_kind,
            event_drafts=event_drafts,
        )
    edges = tuple(
        edge
        for edge in plan.edges_by_source.get(source, ())
        if edge.kind == kind
    )
    if not edges:
        return (), trace, ()
    policy = edges[0].routing_policy
    default_edge = next((edge for edge in edges if edge.default), None)
    matches: list[FlowEdgePlan] = []
    edge_durations: dict[int, float] = {}
    diagnostics: list[FlowDiagnostic] = []
    for edge in sorted(edges, key=lambda item: (item.priority, item.index)):
        if edge.default:
            continue
        started_at = monotonic()
        try:
            matched = _edge_condition_matches(edge, context)
        except FlowRuntimeEvaluationError as error:
            finished_at = monotonic()
            edge_durations[edge.index] = _elapsed_ms_between(
                started_at,
                finished_at,
            )
            diagnostic = _execution_diagnostic(
                code=error.code,
                path=f"edges[{edge.index}].condition",
                message="Flow route condition failed.",
                hint="Check that the route condition selects available data.",
            )
            diagnostics.append(diagnostic)
            trace = trace.with_edge_state(
                edge.index,
                FlowEdgeState.FAILED,
                duration_ms=edge_durations[edge.index],
                diagnostics=(diagnostic,),
            )
            _append_condition_event_draft(
                event_drafts,
                edge=edge,
                status="failed",
                matched=False,
                diagnostics=(diagnostic,),
                started=started_at,
                finished=finished_at,
            )
            _append_edge_eligible_event_draft(
                event_drafts,
                edge=edge,
                status="failed",
                eligible=False,
                diagnostics=(diagnostic,),
                started=started_at,
                finished=finished_at,
            )
            continue
        finished_at = monotonic()
        edge_durations[edge.index] = _elapsed_ms_between(
            started_at,
            finished_at,
        )
        if edge.condition is not None or (
            edge.kind == FlowEdgeKind.RESUME and edge.label is not None
        ):
            _append_condition_event_draft(
                event_drafts,
                edge=edge,
                status="matched" if matched else "unmatched",
                matched=matched,
                started=started_at,
                finished=finished_at,
            )
        _append_edge_eligible_event_draft(
            event_drafts,
            edge=edge,
            status="eligible" if matched else "suppressed",
            eligible=matched,
            started=started_at,
            finished=finished_at,
        )
        if matched:
            matches.append(edge)
        else:
            trace = trace.with_edge_state(
                edge.index,
                FlowEdgeState.SUPPRESSED,
                duration_ms=edge_durations[edge.index],
            )
    if policy == FlowRouteMatchPolicy.ALL_MATCHING:
        selected = tuple(matches)
    else:
        selected = tuple(matches[:1])
    if not selected and default_edge is not None:
        selected = (default_edge,)
    for edge in selected:
        trace = trace.with_edge_state(
            edge.index,
            FlowEdgeState.TAKEN,
            duration_ms=edge_durations.get(edge.index, 0.0),
        )
        if edge.default:
            _append_edge_eligible_event_draft(
                event_drafts,
                edge=edge,
                status="eligible",
                eligible=True,
            )
        _append_edge_routed_event_draft(
            event_drafts,
            edge=edge,
            status="taken",
        )
    selected_indexes = {edge.index for edge in selected}
    suppressed = [
        edge for edge in matches if edge.index not in selected_indexes
    ]
    if default_edge is not None and default_edge.index not in selected_indexes:
        suppressed.append(default_edge)
    for edge in sorted(suppressed, key=lambda item: item.index):
        trace = trace.with_edge_state(
            edge.index,
            FlowEdgeState.SUPPRESSED,
            duration_ms=edge_durations.get(edge.index, 0.0),
        )
        if edge.default:
            _append_edge_eligible_event_draft(
                event_drafts,
                edge=edge,
                status="suppressed",
                eligible=False,
            )
        _append_edge_routed_event_draft(
            event_drafts,
            edge=edge,
            status="suppressed",
        )
    return selected, trace, tuple(diagnostics)


def _select_forced_route(
    plan: FlowExecutionPlan,
    source: str,
    forced_target: str,
    kind: FlowEdgeKind,
    trace: FlowExecutionTrace,
    *,
    forced_kind: FlowEdgeKind | None = None,
    event_drafts: list[_FlowEventDraft] | None = None,
) -> tuple[
    tuple[FlowEdgePlan, ...],
    FlowExecutionTrace,
    tuple[FlowDiagnostic, ...],
]:
    source_edges = plan.edges_by_source.get(source, ())
    selected_kind = forced_kind or kind
    selected = next(
        (
            edge
            for edge in source_edges
            if edge.target == forced_target and edge.kind == selected_kind
        ),
        None,
    )
    if selected is None and forced_kind is None:
        selected = next(
            (edge for edge in source_edges if edge.target == forced_target),
            None,
        )
    if selected is None:
        return (), trace, ()
    started_at = monotonic()
    duration_ms = _elapsed_ms(started_at)
    for edge in source_edges:
        if edge.kind != selected.kind:
            continue
        status = "taken" if edge.index == selected.index else "suppressed"
        if edge.index == selected.index:
            _append_edge_eligible_event_draft(
                event_drafts,
                edge=edge,
                status="eligible",
                eligible=True,
                started=started_at,
            )
        _append_edge_routed_event_draft(
            event_drafts,
            edge=edge,
            status=status,
        )
        trace = trace.with_edge_state(
            edge.index,
            (
                FlowEdgeState.TAKEN
                if edge.index == selected.index
                else FlowEdgeState.SUPPRESSED
            ),
            duration_ms=duration_ms,
        )
    return (selected,), trace, ()


def _edge_condition_matches(
    edge: FlowEdgePlan,
    context: FlowRuntimeContext,
) -> bool:
    if (
        edge.kind == FlowEdgeKind.RESUME
        and edge.label is not None
        and not _resume_label_matches(edge, context)
    ):
        return False
    if edge.condition is None:
        return True
    return evaluate_flow_condition_plan(edge.condition, context)


def _resume_label_matches(
    edge: FlowEdgePlan,
    context: FlowRuntimeContext,
) -> bool:
    source_output = context.node_outputs.get(edge.source)
    if not isinstance(source_output, Mapping):
        return False
    result = source_output.get("result")
    if isinstance(result, Mapping):
        decision = result.get("decision")
    else:
        decision = source_output.get("decision")
    return decision == edge.label


def _mark_unscheduled_nodes_skipped(
    trace: FlowExecutionTrace,
) -> FlowExecutionTrace:
    updated = trace
    for node in trace.nodes:
        if node.state == FlowNodeState.PENDING:
            updated = updated.with_node_state(node.node, FlowNodeState.SKIPPED)
    return updated


def _select_flow_outputs(
    plan: FlowExecutionPlan,
    context: FlowRuntimeContext,
) -> tuple[Mapping[str, object], tuple[FlowDiagnostic, ...]]:
    outputs: dict[str, object] = {}
    diagnostics: list[FlowDiagnostic] = []
    for name, selector in plan.output_selectors.items():
        try:
            outputs[name] = evaluate_flow_selector(selector, context)
        except FlowRuntimeEvaluationError:
            diagnostics.append(
                _execution_diagnostic(
                    code="flow.execution.missing_output",
                    path=f"flow.output_behavior.outputs.{name}",
                    message="Flow output selection failed.",
                    hint="Select an output produced by an executed node.",
                )
            )
    return _freeze_mapping(outputs), tuple(diagnostics)


def _execution_diagnostic(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
) -> FlowDiagnostic:
    return FlowDiagnostic(
        code=code,
        category=FlowDiagnosticCategory.EXECUTION,
        severity=FlowDiagnosticSeverity.ERROR,
        path=path,
        message=message,
        hint=hint,
    )


def _stream_session_cancellation_checker(
    stream_session: FlowStreamSession,
    cancellation_checker: CancellationChecker | None,
) -> CancellationChecker:
    assert isinstance(stream_session, FlowStreamSession)
    if cancellation_checker is not None:
        assert callable(cancellation_checker)
    if (
        cancellation_checker is None
        or cancellation_checker is stream_session.cancellation_checker
    ):
        return stream_session.check_cancelled

    async def check_cancelled() -> None:
        await stream_session.check_cancelled()
        try:
            await cancellation_checker()
        except CancelledError:
            stream_session.cancel()
            raise
        await stream_session.check_cancelled()

    return check_cancelled


async def _check_cancelled(
    cancellation_checker: CancellationChecker | None,
) -> None:
    if cancellation_checker is not None:
        await cancellation_checker()


def evaluate_flow_selector(
    selector: FlowSelector,
    context: FlowRuntimeContext,
) -> object:
    assert isinstance(selector, FlowSelector)
    assert isinstance(context, FlowRuntimeContext)
    return _required_selector_value(selector, context)


def evaluate_flow_node_mappings(
    node: FlowNodePlan,
    context: FlowRuntimeContext,
) -> Mapping[str, object]:
    assert isinstance(node, FlowNodePlan)
    assert isinstance(context, FlowRuntimeContext)
    return evaluate_flow_mappings(node.mappings, context)


def evaluate_flow_mappings(
    mappings: tuple[FlowMappingPlan, ...],
    context: FlowRuntimeContext,
) -> Mapping[str, object]:
    assert isinstance(mappings, tuple)
    assert isinstance(context, FlowRuntimeContext)
    values: dict[str, object] = {}
    for mapping in mappings:
        assert isinstance(mapping, FlowMappingPlan)
        if mapping.target in values:
            raise FlowRuntimeEvaluationError(
                "flow.execution.duplicate_mapping_target"
            )
        values[mapping.target] = _evaluate_mapping(mapping, context)
    return _freeze_mapping(values)


def evaluate_flow_condition_plan(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
) -> bool:
    assert isinstance(condition, FlowConditionPlan)
    assert isinstance(context, FlowRuntimeContext)
    return _evaluate_condition_plan(condition, context)


def _evaluate_mapping(
    mapping: FlowMappingPlan,
    context: FlowRuntimeContext,
) -> object:
    match mapping.kind:
        case (
            FlowMappingKind.SELECT
            | FlowMappingKind.RENAME
            | FlowMappingKind.FILE
            | FlowMappingKind.FILE_ARRAY
        ):
            return _single_source_mapping_value(mapping, context)
        case FlowMappingKind.OBJECT:
            if not mapping.fields:
                raise FlowRuntimeEvaluationError(
                    "flow.execution.empty_mapping"
                )
            return _freeze_mapping(
                {
                    name: _required_selector_value(selector, context)
                    for name, selector in mapping.fields.items()
                }
            )
        case FlowMappingKind.ARRAY:
            if not mapping.items:
                raise FlowRuntimeEvaluationError(
                    "flow.execution.empty_mapping"
                )
            return tuple(
                _required_selector_value(selector, context)
                for selector in mapping.items
            )
        case FlowMappingKind.MERGE:
            if not mapping.sources:
                raise FlowRuntimeEvaluationError(
                    "flow.execution.empty_mapping"
                )
            return _merge_mapping_sources(mapping, context)
        case FlowMappingKind.COALESCE:
            if not mapping.sources:
                raise FlowRuntimeEvaluationError(
                    "flow.execution.empty_mapping"
                )
            return _coalesce_mapping_sources(mapping, context)
    raise FlowRuntimeEvaluationError("flow.execution.unsupported_mapping_kind")


def _single_source_mapping_value(
    mapping: FlowMappingPlan,
    context: FlowRuntimeContext,
) -> object:
    if mapping.source is None:
        raise FlowRuntimeEvaluationError(
            "flow.execution.missing_mapping_source"
        )
    return _required_selector_value(mapping.source, context)


def _merge_mapping_sources(
    mapping: FlowMappingPlan,
    context: FlowRuntimeContext,
) -> Mapping[str, object]:
    merged: dict[str, object] = {}
    for selector in mapping.sources:
        value = _required_selector_value(selector, context)
        if not isinstance(value, Mapping):
            raise FlowRuntimeEvaluationError(
                "flow.execution.merge_requires_object"
            )
        for key, item in value.items():
            assert isinstance(key, str) and key.strip()
            merged[key] = item
    return _freeze_mapping(merged)


def _coalesce_mapping_sources(
    mapping: FlowMappingPlan,
    context: FlowRuntimeContext,
) -> object:
    for selector in mapping.sources:
        value = _selector_value(selector, context, missing_ok=True)
        if value is not _MISSING:
            return value
    raise FlowRuntimeEvaluationError("flow.execution.missing_selector_value")


def _required_selector_value(
    selector: FlowSelector,
    context: FlowRuntimeContext,
) -> object:
    return _selector_value(selector, context, missing_ok=False)


def _selector_value(
    selector: FlowSelector,
    context: FlowRuntimeContext,
    *,
    missing_ok: bool,
) -> object:
    value = resolve_flow_selector_value(
        selector,
        inputs=context.inputs,
        node_outputs=context.node_outputs,
        missing=_MISSING,
    )
    if value is _MISSING and not missing_ok:
        raise FlowRuntimeEvaluationError(
            "flow.execution.missing_selector_value"
        )
    return value


def _evaluate_condition_plan(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
) -> bool:
    match condition.operator:
        case FlowConditionOperator.ALL:
            return all(
                _evaluate_condition_plan(child, context)
                for child in condition.conditions
            )
        case FlowConditionOperator.ANY:
            return any(
                _evaluate_condition_plan(child, context)
                for child in condition.conditions
            )
        case FlowConditionOperator.NOT:
            if condition.condition is None:
                raise FlowRuntimeEvaluationError(
                    "flow.condition_missing_child"
                )
            return not _evaluate_condition_plan(condition.condition, context)
        case FlowConditionOperator.EXISTS:
            return (
                _selected_value(condition, context, missing_ok=True)
                is not _MISSING
            )
        case FlowConditionOperator.NOT_EXISTS:
            return (
                _selected_value(condition, context, missing_ok=True)
                is _MISSING
            )
        case FlowConditionOperator.IS_NULL:
            return _selected_value(condition, context, missing_ok=True) is None
        case FlowConditionOperator.NOT_NULL:
            value = _selected_value(condition, context, missing_ok=True)
            return value is not _MISSING and value is not None
        case FlowConditionOperator.IS_TYPE:
            return _matches_type(
                _selected_value(condition, context),
                condition.value_type,
            )
        case FlowConditionOperator.EQ:
            return _selected_value(condition, context) == _comparison_value(
                condition,
                context,
            )
        case FlowConditionOperator.NE:
            return _selected_value(condition, context) != _comparison_value(
                condition,
                context,
            )
        case FlowConditionOperator.IN:
            return _selected_value(condition, context) in _membership_values(
                condition,
                context,
            )
        case FlowConditionOperator.NOT_IN:
            return _selected_value(
                condition,
                context,
            ) not in _membership_values(condition, context)
        case FlowConditionOperator.STARTS_WITH:
            return _string_condition(
                condition,
                context,
                lambda value, comparison: value.startswith(comparison),
            )
        case FlowConditionOperator.ENDS_WITH:
            return _string_condition(
                condition,
                context,
                lambda value, comparison: value.endswith(comparison),
            )
        case FlowConditionOperator.CONTAINS:
            return _string_condition(
                condition,
                context,
                lambda value, comparison: comparison in value,
            )
        case FlowConditionOperator.GT:
            return _numeric_condition(
                condition,
                context,
                lambda value, comparison: value > comparison,
            )
        case FlowConditionOperator.GTE:
            return _numeric_condition(
                condition,
                context,
                lambda value, comparison: value >= comparison,
            )
        case FlowConditionOperator.LT:
            return _numeric_condition(
                condition,
                context,
                lambda value, comparison: value < comparison,
            )
        case FlowConditionOperator.LTE:
            return _numeric_condition(
                condition,
                context,
                lambda value, comparison: value <= comparison,
            )
    raise FlowRuntimeEvaluationError("flow.condition_unknown_operator")


def _selected_value(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
    *,
    missing_ok: bool = False,
) -> object:
    if condition.selector is None:
        raise FlowRuntimeEvaluationError("flow.condition_missing_selector")
    value = _selector_value(condition.selector, context, missing_ok=True)
    if value is _MISSING and not missing_ok:
        raise FlowRuntimeEvaluationError("flow.condition_missing_value")
    return value


def _comparison_value(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
) -> object:
    if condition.value_selector is not None:
        value = _selector_value(
            condition.value_selector,
            context,
            missing_ok=True,
        )
        if value is _MISSING:
            raise FlowRuntimeEvaluationError("flow.condition_missing_value")
        return value
    return condition.value


def _membership_values(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
) -> tuple[object, ...]:
    if condition.values:
        return condition.values
    value = _comparison_value(condition, context)
    if isinstance(value, list | tuple | frozenset | set):
        return tuple(value)
    return ()


def _string_condition(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
    predicate: Callable[[str, str], bool],
) -> bool:
    value = _selected_value(condition, context)
    comparison = _comparison_value(condition, context)
    if not isinstance(value, str) or not isinstance(comparison, str):
        return False
    return predicate(value, comparison)


def _numeric_condition(
    condition: FlowConditionPlan,
    context: FlowRuntimeContext,
    predicate: Callable[[int | float, int | float], bool],
) -> bool:
    value = _selected_value(condition, context)
    comparison = _comparison_value(condition, context)
    if not _is_number(value) or not _is_number(comparison):
        return False
    assert isinstance(value, int | float)
    assert isinstance(comparison, int | float)
    return predicate(value, comparison)


def _matches_type(
    value: object,
    value_type: FlowConditionValueType | None,
) -> bool:
    match value_type:
        case FlowConditionValueType.STRING:
            return isinstance(value, str)
        case FlowConditionValueType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        case FlowConditionValueType.NUMBER:
            return _is_number(value)
        case FlowConditionValueType.BOOLEAN:
            return isinstance(value, bool)
        case FlowConditionValueType.OBJECT:
            return isinstance(value, Mapping)
        case FlowConditionValueType.ARRAY:
            return isinstance(value, list | tuple)
        case FlowConditionValueType.NULL:
            return value is None
    raise FlowRuntimeEvaluationError("flow.condition_missing_value_type")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
