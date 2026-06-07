from .condition import FlowConditionOperator, FlowConditionValueType
from .definition import FlowEdgeKind, FlowMappingKind, FlowRouteMatchPolicy
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .node import CancellationChecker
from .plan import (
    FlowConditionPlan,
    FlowEdgePlan,
    FlowExecutionPlan,
    FlowMappingPlan,
    FlowNodePlan,
)
from .selector import FlowSelector, resolve_flow_selector_value
from .state import FlowEdgeState, FlowExecutionTrace, FlowNodeState

from asyncio import CancelledError
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from types import MappingProxyType
from typing import TypeAlias

_MISSING = object()

FlowPlanNodeRunner: TypeAlias = Callable[
    [FlowNodePlan, Mapping[str, object]],
    object | Awaitable[object],
]


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
) -> FlowPlanExecutionResult:
    assert isinstance(plan, FlowExecutionPlan)
    assert callable(runner)
    if inputs is not None:
        assert isinstance(inputs, Mapping)
    input_values = _freeze_mapping(inputs or {})
    node_outputs: dict[str, Mapping[str, object]] = {}
    diagnostics: list[FlowDiagnostic] = []
    trace = FlowExecutionTrace.from_plan(plan)
    node_map = plan.node_map
    ready: deque[str] = deque((plan.entry_node,))
    queued: set[str] = {plan.entry_node}
    processed: set[str] = set()

    while ready:
        await _check_cancelled(cancellation_checker)
        node_name = ready.popleft()
        queued.discard(node_name)
        node = node_map[node_name]
        trace = trace.with_node_state(node.name, FlowNodeState.READY)
        context = FlowRuntimeContext(
            inputs=input_values,
            node_outputs=node_outputs,
        )
        mapped_inputs, mapping_diagnostic = _node_inputs(node, context)
        output: object = None
        if mapping_diagnostic is None:
            trace, route_kind, node_diagnostics, output = await _run_plan_node(
                node,
                mapped_inputs,
                runner,
                trace,
            )
            diagnostics.extend(node_diagnostics)
            if route_kind == FlowEdgeKind.SUCCESS:
                node_outputs[node.name] = _node_output_mapping(node, output)
        else:
            route_kind = FlowEdgeKind.ERROR
            diagnostics.append(mapping_diagnostic)
            trace = trace.with_node_state(
                node.name,
                FlowNodeState.FAILED,
                attempts=1,
                diagnostics=(mapping_diagnostic,),
            )
        processed.add(node.name)
        context = FlowRuntimeContext(
            inputs=input_values,
            node_outputs=node_outputs,
        )
        routed, trace, route_diagnostics = _route_from_node(
            plan,
            node.name,
            route_kind,
            context,
            trace,
        )
        diagnostics.extend(route_diagnostics)
        for target in routed:
            if target not in processed and target not in queued:
                ready.append(target)
                queued.add(target)

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
    trace: FlowExecutionTrace,
) -> tuple[
    FlowExecutionTrace,
    FlowEdgeKind,
    tuple[FlowDiagnostic, ...],
    object,
]:
    trace = trace.with_node_state(
        node.name,
        FlowNodeState.RUNNING,
        attempts=1,
    )
    try:
        output = runner(node, inputs)
        if isawaitable(output):
            output = await output
    except CancelledError:
        diagnostic = _execution_diagnostic(
            code="flow.execution.node_cancelled",
            path=f"nodes.{node.name}",
            message="Flow node was cancelled.",
            hint="Inspect cancellation routes for this node.",
        )
        return (
            trace.with_node_state(
                node.name,
                FlowNodeState.CANCELLED,
                attempts=1,
                diagnostics=(diagnostic,),
            ),
            FlowEdgeKind.CANCELLATION,
            (diagnostic,),
            None,
        )
    except TimeoutError:
        diagnostic = _execution_diagnostic(
            code="flow.execution.node_timeout",
            path=f"nodes.{node.name}",
            message="Flow node timed out.",
            hint="Inspect timeout routes for this node.",
        )
        return (
            trace.with_node_state(
                node.name,
                FlowNodeState.FAILED,
                attempts=1,
                diagnostics=(diagnostic,),
            ),
            FlowEdgeKind.TIMEOUT,
            (diagnostic,),
            None,
        )
    except Exception:
        diagnostic = _execution_diagnostic(
            code="flow.execution.node_failed",
            path=f"nodes.{node.name}",
            message="Flow node failed.",
            hint="Inspect error routes for this node.",
        )
        return (
            trace.with_node_state(
                node.name,
                FlowNodeState.FAILED,
                attempts=1,
                diagnostics=(diagnostic,),
            ),
            FlowEdgeKind.ERROR,
            (diagnostic,),
            None,
        )
    return (
        trace.with_node_state(
            node.name,
            FlowNodeState.SUCCEEDED,
            attempts=1,
        ),
        FlowEdgeKind.SUCCESS,
        (),
        output,
    )


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
) -> tuple[tuple[str, ...], FlowExecutionTrace, tuple[FlowDiagnostic, ...]]:
    diagnostics: list[FlowDiagnostic] = []
    routed: list[str] = []
    selected, trace, route_diagnostics = _select_routes(
        plan,
        source,
        route_kind,
        context,
        trace,
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


def _select_routes(
    plan: FlowExecutionPlan,
    source: str,
    kind: FlowEdgeKind,
    context: FlowRuntimeContext,
    trace: FlowExecutionTrace,
) -> tuple[
    tuple[FlowEdgePlan, ...],
    FlowExecutionTrace,
    tuple[FlowDiagnostic, ...],
]:
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
    diagnostics: list[FlowDiagnostic] = []
    for edge in sorted(edges, key=lambda item: (item.priority, item.index)):
        if edge.default:
            continue
        try:
            matched = _edge_condition_matches(edge, context)
        except FlowRuntimeEvaluationError as error:
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
                diagnostics=(diagnostic,),
            )
            continue
        if matched:
            matches.append(edge)
        else:
            trace = trace.with_edge_state(
                edge.index,
                FlowEdgeState.SUPPRESSED,
            )
    if policy == FlowRouteMatchPolicy.ALL_MATCHING:
        selected = tuple(matches)
    else:
        selected = tuple(matches[:1])
    if not selected and default_edge is not None:
        selected = (default_edge,)
    for edge in selected:
        trace = trace.with_edge_state(edge.index, FlowEdgeState.TAKEN)
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
        )
    return selected, trace, tuple(diagnostics)


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
