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
    FlowConditionPlan,
    FlowEdgePlan,
    FlowExecutionPlan,
    FlowMappingPlan,
    FlowNodePlan,
    FlowRetryPlan,
)
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    default_flow_node_registry,
)
from .selector import FlowSelector, resolve_flow_selector_value
from .state import FlowEdgeState, FlowExecutionTrace, FlowNodeState

from asyncio import CancelledError, gather, sleep, wait_for
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from time import monotonic
from types import MappingProxyType
from typing import TypeAlias

_MISSING = object()

FlowPlanNodeRunner: TypeAlias = Callable[
    [FlowNodePlan, Mapping[str, object]],
    object | Awaitable[object],
]


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
    output: object = None


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
        result = await execute_flow_plan(plan, self, inputs=inputs)
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
    concurrency_limit: int = 1,
    resume_trace: FlowExecutionTrace | None = None,
    resume_node_outputs: Mapping[str, Mapping[str, object]] | None = None,
) -> FlowPlanExecutionResult:
    assert isinstance(plan, FlowExecutionPlan)
    assert callable(runner)
    if inputs is not None:
        assert isinstance(inputs, Mapping)
    if resume_trace is not None:
        assert isinstance(resume_trace, FlowExecutionTrace)
    if resume_node_outputs is not None:
        assert isinstance(resume_node_outputs, Mapping)
    assert isinstance(concurrency_limit, int) and not isinstance(
        concurrency_limit,
        bool,
    )
    assert concurrency_limit > 0
    input_values = _freeze_mapping(inputs or {})
    node_outputs: dict[str, Mapping[str, object]] = {
        node: _freeze_mapping(outputs)
        for node, outputs in (resume_node_outputs or {}).items()
    }
    diagnostics: list[FlowDiagnostic] = []
    trace = resume_trace or FlowExecutionTrace.from_plan(plan)
    node_map = plan.node_map
    ready, queued, processed, trace, resume_diagnostics = (
        _initial_runtime_queue(
            plan,
            input_values,
            node_outputs,
            trace,
        )
    )
    diagnostics.extend(resume_diagnostics)

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
            if outcome.route_kind == FlowEdgeKind.SUCCESS:
                node_outputs[outcome.node.name] = _node_output_mapping(
                    outcome.node,
                    outcome.output,
                )
            processed.add(outcome.node.name)
        for outcome in batch_outcomes:
            route_context = _route_context(
                input_values,
                batch_context.node_outputs,
                outcome,
            )
            routed, trace, route_diagnostics = _route_from_node(
                plan,
                outcome.node.name,
                outcome.route_kind,
                route_context,
                trace,
                forced_target=outcome.exhausted_route,
                forced_kind=outcome.exhausted_route_kind,
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
                )
                diagnostics.extend(join_diagnostics)

    trace = _mark_unscheduled_nodes_skipped(trace)
    outputs, output_diagnostics = _select_flow_outputs(
        plan,
        FlowRuntimeContext(inputs=input_values, node_outputs=node_outputs),
    )
    diagnostics.extend(output_diagnostics)
    return FlowPlanExecutionResult(
        outputs=outputs,
        trace=trace,
        diagnostics=tuple(diagnostics),
        node_outputs=node_outputs,
    )


def _initial_runtime_queue(
    plan: FlowExecutionPlan,
    inputs: Mapping[str, object],
    node_outputs: Mapping[str, Mapping[str, object]],
    trace: FlowExecutionTrace,
) -> tuple[
    deque[str],
    set[str],
    set[str],
    FlowExecutionTrace,
    tuple[FlowDiagnostic, ...],
]:
    node_states = {node.node: node.state for node in trace.nodes}
    processed = {
        node.name
        for node in plan.nodes
        if node_states.get(node.name) == FlowNodeState.SUCCEEDED
        and node.name in node_outputs
    }
    if not processed:
        ready: deque[str] = deque((plan.entry_node,))
        return ready, {plan.entry_node}, set(), trace, ()

    ready = deque[str]()
    queued: set[str] = set()
    diagnostics: list[FlowDiagnostic] = []
    context = FlowRuntimeContext(inputs=inputs, node_outputs=node_outputs)
    for source in sorted(processed, key=_plan_node_order(plan)):
        routed, trace, route_diagnostics = _route_from_node(
            plan,
            source,
            FlowEdgeKind.SUCCESS,
            context,
            trace,
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
            )
            diagnostics.extend(join_diagnostics)
    return ready, queued, processed, trace, tuple(diagnostics)


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
) -> _NodeRunOutcome:
    if node.loop is not None:
        return await _run_loop_plan_node(
            node,
            inputs,
            runner,
            context,
            cancellation_checker,
        )
    return await _run_plan_node_attempts(
        node,
        inputs,
        runner,
        cancellation_checker,
    )


async def _run_plan_node_attempts(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    cancellation_checker: CancellationChecker | None,
) -> _NodeRunOutcome:
    started_at = monotonic()
    attempts = 0
    first_failure: _NodeRunOutcome | None = None
    while True:
        attempts += 1
        outcome = await _run_plan_node_once(node, inputs, runner)
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
        if delay_seconds > 0:
            await sleep(delay_seconds)
            await _check_cancelled(cancellation_checker)


async def _run_loop_plan_node(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
    base_context: FlowRuntimeContext,
    cancellation_checker: CancellationChecker | None,
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
        selected_output, diagnostic = _loop_selected_output(node, context)
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
) -> tuple[object, FlowDiagnostic | None]:
    assert node.loop is not None
    try:
        if evaluate_flow_condition_plan(node.loop.exit_condition, context):
            return (
                evaluate_flow_selector(node.loop.output_selector, context),
                None,
            )
        if evaluate_flow_condition_plan(
            node.loop.continue_condition,
            context,
        ):
            return _MISSING, None
    except FlowRuntimeEvaluationError as error:
        return (
            _MISSING,
            _execution_diagnostic(
                code=error.code,
                path=f"nodes.{node.name}.loop_policy",
                message="Flow loop condition evaluation failed.",
                hint="Check loop selectors and node output contracts.",
            ),
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


async def _run_plan_node_once(
    node: FlowNodePlan,
    inputs: Mapping[str, object],
    runner: FlowPlanNodeRunner,
) -> _NodeRunOutcome:
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


def _route_from_node(
    plan: FlowExecutionPlan,
    source: str,
    route_kind: FlowEdgeKind,
    context: FlowRuntimeContext,
    trace: FlowExecutionTrace,
    *,
    forced_target: str | None = None,
    forced_kind: FlowEdgeKind | None = None,
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
            edge_durations[edge.index] = _elapsed_ms(started_at)
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
            continue
        edge_durations[edge.index] = _elapsed_ms(started_at)
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
    return selected, trace, tuple(diagnostics)


def _select_forced_route(
    plan: FlowExecutionPlan,
    source: str,
    forced_target: str,
    kind: FlowEdgeKind,
    trace: FlowExecutionTrace,
    *,
    forced_kind: FlowEdgeKind | None = None,
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
    if edge.condition is None:
        return True
    return evaluate_flow_condition_plan(edge.condition, context)


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
