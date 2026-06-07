from .condition import FlowConditionOperator, FlowConditionValueType
from .definition import FlowMappingKind
from .plan import FlowConditionPlan, FlowMappingPlan, FlowNodePlan
from .selector import FlowSelector, resolve_flow_selector_value

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

_MISSING = object()


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _empty_node_outputs() -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({})


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
