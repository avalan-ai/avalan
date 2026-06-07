from .selector import (
    FlowSelectorRoot,
    FlowSelectorStepKind,
    parse_flow_selector,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

FlowConditionLiteral: TypeAlias = object


class FlowConditionOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IS_TYPE = "is_type"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"
    ALL = "all"
    ANY = "any"
    NOT = "not"


class FlowConditionValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


def _empty_node_outputs() -> Mapping[str, Mapping[str, object]]:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowCondition:
    operator: FlowConditionOperator
    selector: str | None = None
    value: FlowConditionLiteral | None = None
    value_selector: str | None = None
    values: tuple[FlowConditionLiteral, ...] = ()
    value_type: FlowConditionValueType | None = None
    conditions: tuple["FlowCondition", ...] = ()
    condition: "FlowCondition | None" = None

    def __post_init__(self) -> None:
        assert isinstance(self.operator, FlowConditionOperator)
        if self.selector is not None:
            assert isinstance(self.selector, str) and self.selector.strip()
        if self.value is not None:
            object.__setattr__(self, "value", _freeze_value(self.value))
        if self.value_selector is not None:
            assert (
                isinstance(self.value_selector, str)
                and self.value_selector.strip()
            )
        assert isinstance(self.values, tuple)
        object.__setattr__(
            self,
            "values",
            tuple(_freeze_value(value) for value in self.values),
        )
        if self.value_type is not None:
            assert isinstance(self.value_type, FlowConditionValueType)
        assert isinstance(self.conditions, tuple)
        for condition in self.conditions:
            assert isinstance(condition, FlowCondition)
        if self.condition is not None:
            assert isinstance(self.condition, FlowCondition)

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"op": self.operator.value}
        if self.selector is not None:
            value["selector"] = self.selector
        if self.value is not None:
            value["value"] = self.value
        if self.value_selector is not None:
            value["value_selector"] = self.value_selector
        if self.values:
            value["values"] = self.values
        if self.value_type is not None:
            value["value_type"] = self.value_type.value
        if self.conditions:
            value["conditions"] = tuple(
                condition.as_dict() for condition in self.conditions
            )
        if self.condition is not None:
            value["condition"] = self.condition.as_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowConditionEvaluationContext:
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


class FlowConditionEvaluationError(ValueError):
    def __init__(self, code: str) -> None:
        assert isinstance(code, str) and code.strip()
        self.code = code
        super().__init__(code)


_MISSING = object()
_NUMERIC_OPERATORS = {
    FlowConditionOperator.GT,
    FlowConditionOperator.GTE,
    FlowConditionOperator.LT,
    FlowConditionOperator.LTE,
}


def evaluate_flow_condition(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> bool:
    assert isinstance(condition, FlowCondition)
    assert isinstance(context, FlowConditionEvaluationContext)
    return _evaluate_condition(condition, context)


def _evaluate_condition(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> bool:
    match condition.operator:
        case FlowConditionOperator.ALL:
            return all(
                _evaluate_condition(child, context)
                for child in condition.conditions
            )
        case FlowConditionOperator.ANY:
            return any(
                _evaluate_condition(child, context)
                for child in condition.conditions
            )
        case FlowConditionOperator.NOT:
            if condition.condition is None:
                raise FlowConditionEvaluationError(
                    "flow.condition_missing_child"
                )
            return not _evaluate_condition(condition.condition, context)
        case FlowConditionOperator.EXISTS:
            return _selected_value(
                condition, context, missing_ok=True
            ) is not (_MISSING)
        case FlowConditionOperator.NOT_EXISTS:
            return _selected_value(condition, context, missing_ok=True) is (
                _MISSING
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
        case (
            FlowConditionOperator.STARTS_WITH
            | FlowConditionOperator.ENDS_WITH
            | FlowConditionOperator.CONTAINS
        ):
            return _evaluate_string_condition(condition, context)
        case _ if condition.operator in _NUMERIC_OPERATORS:
            return _evaluate_numeric_condition(condition, context)
    raise FlowConditionEvaluationError("flow.condition_unknown_operator")


def _selected_value(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
    *,
    missing_ok: bool = False,
) -> object:
    if condition.selector is None:
        raise FlowConditionEvaluationError("flow.condition_missing_selector")
    value = _resolve_selector(condition.selector, context)
    if value is _MISSING and not missing_ok:
        raise FlowConditionEvaluationError("flow.condition_missing_value")
    return value


def _comparison_value(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> object:
    if condition.value_selector is not None:
        value = _resolve_selector(condition.value_selector, context)
        if value is _MISSING:
            raise FlowConditionEvaluationError("flow.condition_missing_value")
        return value
    return condition.value


def _membership_values(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> tuple[object, ...]:
    if condition.values:
        return condition.values
    value = _comparison_value(condition, context)
    if isinstance(value, list | tuple | frozenset | set):
        return tuple(value)
    return ()


def _evaluate_string_condition(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> bool:
    value = _selected_value(condition, context)
    comparison = _comparison_value(condition, context)
    if not isinstance(value, str) or not isinstance(comparison, str):
        return False
    match condition.operator:
        case FlowConditionOperator.STARTS_WITH:
            return value.startswith(comparison)
        case FlowConditionOperator.ENDS_WITH:
            return value.endswith(comparison)
        case FlowConditionOperator.CONTAINS:
            return comparison in value
    raise FlowConditionEvaluationError("flow.condition_unknown_operator")


def _evaluate_numeric_condition(
    condition: FlowCondition,
    context: FlowConditionEvaluationContext,
) -> bool:
    value = _selected_value(condition, context)
    comparison = _comparison_value(condition, context)
    if not _is_number(value) or not _is_number(comparison):
        return False
    assert isinstance(value, int | float)
    assert isinstance(comparison, int | float)
    match condition.operator:
        case FlowConditionOperator.GT:
            return value > comparison
        case FlowConditionOperator.GTE:
            return value >= comparison
        case FlowConditionOperator.LT:
            return value < comparison
        case FlowConditionOperator.LTE:
            return value <= comparison
    raise FlowConditionEvaluationError("flow.condition_unknown_operator")


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
    raise FlowConditionEvaluationError("flow.condition_missing_value_type")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _resolve_selector(
    selector: str,
    context: FlowConditionEvaluationContext,
) -> object:
    parsed = parse_flow_selector(selector)
    if parsed.root == FlowSelectorRoot.FLOW_INPUT:
        current = context.inputs.get(parsed.source, _MISSING)
    else:
        outputs = context.node_outputs.get(parsed.source)
        if outputs is None or parsed.output is None:
            return _MISSING
        current = outputs.get(parsed.output, _MISSING)
    for step in parsed.path:
        if current is _MISSING:
            return _MISSING
        if step.kind == FlowSelectorStepKind.FIELD:
            if not isinstance(current, Mapping):
                return _MISSING
            current = current.get(step.value, _MISSING)
            continue
        if not isinstance(current, list | tuple):
            return _MISSING
        assert isinstance(step.value, int)
        if step.value >= len(current):
            return _MISSING
        current = current[step.value]
    return current
