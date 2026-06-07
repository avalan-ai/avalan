from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.flow import (
    FlowCondition,
    FlowConditionEvaluationContext,
    FlowConditionEvaluationError,
    FlowConditionOperator,
    FlowConditionValueType,
    evaluate_flow_condition,
)
from avalan.flow import condition as flow_condition


class FlowConditionTestCase(TestCase):
    def setUp(self) -> None:
        self.context = FlowConditionEvaluationContext(
            inputs={
                "payload": {
                    "expected": "ready",
                    "city": "Paris",
                    "enabled": True,
                },
            },
            node_outputs={
                "prepare": {
                    "result": {
                        "status": "ready",
                        "count": 3,
                        "score": 3.5,
                        "tags": ("alpha", "beta"),
                        "nullable": None,
                        "payload": {"nested": "value"},
                    },
                },
            },
        )

    def test_evaluate_flow_condition_operators(self) -> None:
        cases = (
            (
                FlowConditionOperator.EQ,
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="ready",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="done",
                ),
            ),
            (
                FlowConditionOperator.NE,
                FlowCondition(
                    operator=FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="done",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="ready",
                ),
            ),
            (
                FlowConditionOperator.EXISTS,
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="prepare.result.status",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="prepare.result.missing",
                ),
            ),
            (
                FlowConditionOperator.NOT_EXISTS,
                FlowCondition(
                    operator=FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.missing",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.IS_TYPE,
                FlowCondition(
                    operator=FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.count",
                    value_type=FlowConditionValueType.INTEGER,
                ),
                FlowCondition(
                    operator=FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                    value_type=FlowConditionValueType.INTEGER,
                ),
            ),
            (
                FlowConditionOperator.IN,
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
            ),
            (
                FlowConditionOperator.NOT_IN,
                FlowCondition(
                    operator=FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
                FlowCondition(
                    operator=FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
            ),
            (
                FlowConditionOperator.GT,
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=2,
                ),
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.GTE,
                FlowCondition(
                    operator=FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                FlowCondition(
                    operator=FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.LT,
                FlowCondition(
                    operator=FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=4,
                ),
                FlowCondition(
                    operator=FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.LTE,
                FlowCondition(
                    operator=FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                FlowCondition(
                    operator=FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.STARTS_WITH,
                FlowCondition(
                    operator=FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Par",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Lon",
                ),
            ),
            (
                FlowConditionOperator.ENDS_WITH,
                FlowCondition(
                    operator=FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="ris",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="don",
                ),
            ),
            (
                FlowConditionOperator.CONTAINS,
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="ari",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="zzz",
                ),
            ),
            (
                FlowConditionOperator.IS_NULL,
                FlowCondition(
                    operator=FlowConditionOperator.IS_NULL,
                    selector="prepare.result.nullable",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.IS_NULL,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.NOT_NULL,
                FlowCondition(
                    operator=FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.status",
                ),
                FlowCondition(
                    operator=FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.nullable",
                ),
            ),
            (
                FlowConditionOperator.ALL,
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        FlowCondition(
                            operator=FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        FlowCondition(
                            operator=FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.ANY,
                FlowCondition(
                    operator=FlowConditionOperator.ANY,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        FlowCondition(
                            operator=FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                FlowCondition(
                    operator=FlowConditionOperator.ANY,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        FlowCondition(
                            operator=FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.NOT,
                FlowCondition(
                    operator=FlowConditionOperator.NOT,
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="done",
                    ),
                ),
                FlowCondition(
                    operator=FlowConditionOperator.NOT,
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="ready",
                    ),
                ),
            ),
        )

        for operator, true_condition, false_condition in cases:
            with self.subTest(operator=operator.value):
                self.assertTrue(
                    evaluate_flow_condition(true_condition, self.context)
                )
                self.assertFalse(
                    evaluate_flow_condition(false_condition, self.context)
                )

    def test_evaluate_flow_condition_supports_value_selectors(self) -> None:
        condition = FlowCondition(
            operator=FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value_selector="input.payload.expected",
        )

        self.assertTrue(evaluate_flow_condition(condition, self.context))

    def test_evaluate_flow_condition_supports_literal_membership(self) -> None:
        condition = FlowCondition(
            operator=FlowConditionOperator.IN,
            selector="prepare.result.status",
            value=("ready", "done"),
        )
        scalar_condition = FlowCondition(
            operator=FlowConditionOperator.IN,
            selector="prepare.result.status",
            value="ready",
        )

        self.assertTrue(evaluate_flow_condition(condition, self.context))
        self.assertFalse(
            evaluate_flow_condition(scalar_condition, self.context)
        )

    def test_evaluate_flow_condition_checks_all_value_types(self) -> None:
        cases = (
            ("prepare.result.status", FlowConditionValueType.STRING),
            ("prepare.result.score", FlowConditionValueType.NUMBER),
            ("input.payload.enabled", FlowConditionValueType.BOOLEAN),
            ("prepare.result.payload", FlowConditionValueType.OBJECT),
            ("prepare.result.tags", FlowConditionValueType.ARRAY),
            ("prepare.result.nullable", FlowConditionValueType.NULL),
        )

        for selector, value_type in cases:
            with self.subTest(value_type=value_type.value):
                self.assertTrue(
                    evaluate_flow_condition(
                        FlowCondition(
                            operator=FlowConditionOperator.IS_TYPE,
                            selector=selector,
                            value_type=value_type,
                        ),
                        self.context,
                    )
                )

    def test_evaluate_flow_condition_treats_missing_paths_as_absent(
        self,
    ) -> None:
        cases = (
            "missing.result",
            "prepare.result.missing.value",
            "prepare.result.status.name",
            "prepare.result.status[0]",
            "prepare.result.tags[9]",
        )

        for selector in cases:
            with self.subTest(selector=selector):
                self.assertTrue(
                    evaluate_flow_condition(
                        FlowCondition(
                            operator=FlowConditionOperator.NOT_EXISTS,
                            selector=selector,
                        ),
                        self.context,
                    )
                )

        self.assertTrue(
            evaluate_flow_condition(
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="prepare.result.tags[1]",
                    value="beta",
                ),
                self.context,
            )
        )

    def test_evaluate_flow_condition_reports_missing_values(self) -> None:
        cases = (
            (
                FlowCondition(operator=FlowConditionOperator.EQ),
                "flow.condition_missing_selector",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="prepare.result.missing",
                    value="ready",
                ),
                "flow.condition_missing_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value_selector="prepare.result.missing",
                ),
                "flow.condition_missing_value",
            ),
            (
                FlowCondition(operator=FlowConditionOperator.NOT),
                "flow.condition_missing_child",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                ),
                "flow.condition_missing_value_type",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowConditionEvaluationError) as raised:
                    evaluate_flow_condition(condition, self.context)

                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("ready", str(raised.exception))

    def test_evaluate_flow_condition_handles_type_mismatches(self) -> None:
        numeric = FlowCondition(
            operator=FlowConditionOperator.GT,
            selector="prepare.result.status",
            value=3,
        )
        string = FlowCondition(
            operator=FlowConditionOperator.CONTAINS,
            selector="prepare.result.count",
            value="3",
        )

        self.assertFalse(evaluate_flow_condition(numeric, self.context))
        self.assertFalse(evaluate_flow_condition(string, self.context))

    def test_condition_entities_are_frozen_and_serializable(self) -> None:
        raw_value = {"items": ["one"]}
        condition = FlowCondition(
            operator=FlowConditionOperator.IN,
            selector="prepare.result.status",
            value=raw_value,
        )
        raw_value["items"].append("two")

        self.assertEqual(condition.as_dict()["op"], "in")
        self.assertEqual(condition.as_dict()["value"], {"items": ("one",)})
        with self.assertRaises(FrozenInstanceError):
            condition.selector = "prepare.result.other"  # type: ignore[misc]
        with self.assertRaises(AssertionError):
            FlowCondition(
                operator="eq",  # type: ignore[arg-type]
                selector="prepare.result.status",
            )
        with self.assertRaises(AssertionError):
            FlowCondition(
                operator=FlowConditionOperator.EQ,
                selector="",
            )
        with self.assertRaises(AssertionError):
            FlowCondition(
                operator=FlowConditionOperator.EQ,
                values=[1],  # type: ignore[arg-type]
            )

    def test_condition_serializes_nested_fields(self) -> None:
        condition = FlowCondition(
            operator=FlowConditionOperator.NOT,
            condition=FlowCondition(
                operator=FlowConditionOperator.ALL,
                conditions=(
                    FlowCondition(
                        operator=FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value_selector="input.payload.expected",
                    ),
                    FlowCondition(
                        operator=FlowConditionOperator.IS_TYPE,
                        selector="prepare.result.status",
                        value_type=FlowConditionValueType.STRING,
                    ),
                    FlowCondition(
                        operator=FlowConditionOperator.IN,
                        selector="prepare.result.status",
                        values=("ready",),
                    ),
                ),
            ),
        )

        self.assertEqual(
            condition.as_dict(),
            {
                "op": "not",
                "condition": {
                    "op": "all",
                    "conditions": (
                        {
                            "op": "eq",
                            "selector": "prepare.result.status",
                            "value_selector": "input.payload.expected",
                        },
                        {
                            "op": "is_type",
                            "selector": "prepare.result.status",
                            "value_type": "string",
                        },
                        {
                            "op": "in",
                            "selector": "prepare.result.status",
                            "values": ("ready",),
                        },
                    ),
                },
            },
        )

    def test_condition_private_helpers_keep_defensive_errors(self) -> None:
        condition = FlowCondition(
            operator=FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value="ready",
        )
        object.__setattr__(condition, "operator", "unknown")
        with self.assertRaises(FlowConditionEvaluationError) as raised:
            evaluate_flow_condition(condition, self.context)
        self.assertEqual(
            raised.exception.code,
            "flow.condition_unknown_operator",
        )

        string_condition = FlowCondition(
            operator=FlowConditionOperator.CONTAINS,
            selector="prepare.result.status",
            value="ready",
        )
        object.__setattr__(string_condition, "operator", "unknown")
        with self.assertRaises(FlowConditionEvaluationError) as string_error:
            flow_condition._evaluate_string_condition(  # type: ignore[attr-defined]
                string_condition,
                self.context,
            )
        self.assertEqual(
            string_error.exception.code,
            "flow.condition_unknown_operator",
        )

        numeric_condition = FlowCondition(
            operator=FlowConditionOperator.GT,
            selector="prepare.result.count",
            value=1,
        )
        object.__setattr__(numeric_condition, "operator", "unknown")
        with self.assertRaises(FlowConditionEvaluationError) as numeric_error:
            flow_condition._evaluate_numeric_condition(  # type: ignore[attr-defined]
                numeric_condition,
                self.context,
            )
        self.assertEqual(
            numeric_error.exception.code,
            "flow.condition_unknown_operator",
        )

    def test_condition_context_rejects_invalid_values(self) -> None:
        self.assertEqual(FlowConditionEvaluationContext().node_outputs, {})
        with self.assertRaises(AssertionError):
            FlowConditionEvaluationContext(
                node_outputs={"": {}},
            )
        with self.assertRaises(AssertionError):
            FlowConditionEvaluationContext(
                node_outputs={"node": object()},  # type: ignore[dict-item]
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition(
                object(),  # type: ignore[arg-type]
                self.context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition(
                FlowCondition(operator=FlowConditionOperator.EXISTS),
                object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowConditionEvaluationError("")


if __name__ == "__main__":
    main()
