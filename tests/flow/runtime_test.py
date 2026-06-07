from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowConditionOperator,
    FlowConditionPlan,
    FlowConditionValueType,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeKind,
    FlowNodePlan,
    FlowRuntimeContext,
    FlowRuntimeEvaluationError,
    evaluate_flow_condition_plan,
    evaluate_flow_mappings,
    evaluate_flow_node_mappings,
    evaluate_flow_selector,
    parse_flow_selector,
    resolve_flow_selector_value,
)


class FlowRuntimeEvaluationTestCase(TestCase):
    def setUp(self) -> None:
        self.document = {
            "source_kind": "local_path",
            "reference": "/private/customer.pdf",
            "mime_type": "application/pdf",
            "metadata": {"purpose": "review"},
        }
        self.artifact = {
            "source_kind": "artifact",
            "reference": "artifact-1",
            "metadata": {"store": "local"},
        }
        self.context = FlowRuntimeContext(
            inputs={
                "payload": {
                    "customer": {
                        "name": "Ada",
                        "tags": ["vip", "new"],
                    },
                    "city": "Paris",
                    "enabled": True,
                    "expected": "ready",
                    "left": {"shared": "input", "only_input": 1},
                },
                "document": self.document,
                "documents": [self.document, self.artifact],
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
                        "right": {"shared": "node", "only_node": 2},
                    },
                },
            },
        )

    def test_evaluate_flow_selector_resolves_nested_runtime_values(
        self,
    ) -> None:
        self.assertEqual(
            evaluate_flow_selector(
                parse_flow_selector("input.payload.customer.tags[1]"),
                self.context,
            ),
            "new",
        )
        self.assertEqual(
            evaluate_flow_selector(
                parse_flow_selector("prepare.result.payload.nested"),
                self.context,
            ),
            "value",
        )

        sentinel = object()
        missing_selectors = (
            "missing.result.value",
            "prepare.result.missing.value",
            "prepare.result.status.name",
            "prepare.result.status[0]",
            "prepare.result.tags[9]",
        )
        for selector in missing_selectors:
            with self.subTest(selector=selector):
                self.assertIs(
                    resolve_flow_selector_value(
                        parse_flow_selector(selector),
                        inputs=self.context.inputs,
                        node_outputs=self.context.node_outputs,
                        missing=sentinel,
                    ),
                    sentinel,
                )

        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_selector(
                parse_flow_selector("input.missing"),
                self.context,
            )

        self.assertEqual(
            raised.exception.code,
            "flow.execution.missing_selector_value",
        )
        self.assertNotIn("customer.pdf", str(raised.exception))

    def test_runtime_context_freezes_values_and_validates_shape(self) -> None:
        raw_payload = {"items": ["one"]}
        context = FlowRuntimeContext(
            inputs={"payload": raw_payload},
            node_outputs={"node": {"result": raw_payload}},
        )
        raw_payload["items"].append("two")
        payload = cast(Mapping[str, object], context.inputs["payload"])
        node_output = cast(
            Mapping[str, object],
            context.node_outputs["node"]["result"],
        )

        self.assertEqual(payload["items"], ("one",))
        self.assertEqual(node_output["items"], ("one",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], context.inputs)["other"] = "value"
        with self.assertRaises(FrozenInstanceError):
            context.inputs = {}  # type: ignore[misc]
        with self.assertRaises(AssertionError):
            FlowRuntimeContext(inputs={1: "bad"})  # type: ignore[dict-item]
        with self.assertRaises(AssertionError):
            FlowRuntimeContext(node_outputs={"": {}})
        with self.assertRaises(AssertionError):
            FlowRuntimeEvaluationError("")
        with self.assertRaises(AssertionError):
            evaluate_flow_selector(
                "input.payload",  # type: ignore[arg-type]
                context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_selector(
                parse_flow_selector("input.payload"),
                "context",  # type: ignore[arg-type]
            )

    def test_evaluate_flow_node_mappings_supports_all_mapping_kinds(
        self,
    ) -> None:
        node = FlowNodePlan(
            name="mapper",
            type="select",
            kind=FlowNodeKind.SELECT,
            mappings=(
                FlowMappingPlan(
                    target="selected",
                    kind=FlowMappingKind.SELECT,
                    source=parse_flow_selector("input.payload.customer"),
                ),
                FlowMappingPlan(
                    target="renamed",
                    kind=FlowMappingKind.RENAME,
                    source=parse_flow_selector("prepare.result.status"),
                ),
                FlowMappingPlan(
                    target="constructed",
                    kind=FlowMappingKind.OBJECT,
                    fields={
                        "name": parse_flow_selector(
                            "input.payload.customer.name"
                        ),
                        "status": parse_flow_selector("prepare.result.status"),
                    },
                ),
                FlowMappingPlan(
                    target="items",
                    kind=FlowMappingKind.ARRAY,
                    items=(
                        parse_flow_selector("input.payload.city"),
                        parse_flow_selector("prepare.result.count"),
                    ),
                ),
                FlowMappingPlan(
                    target="merged",
                    kind=FlowMappingKind.MERGE,
                    sources=(
                        parse_flow_selector("input.payload.left"),
                        parse_flow_selector("prepare.result.right"),
                    ),
                ),
                FlowMappingPlan(
                    target="document",
                    kind=FlowMappingKind.FILE,
                    source=parse_flow_selector("input.document"),
                ),
                FlowMappingPlan(
                    target="documents",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source=parse_flow_selector("input.documents"),
                ),
            ),
        )

        result = evaluate_flow_node_mappings(node, self.context)
        selected = cast(Mapping[str, object], result["selected"])
        constructed = cast(Mapping[str, object], result["constructed"])
        merged = cast(Mapping[str, object], result["merged"])
        document = cast(Mapping[str, object], result["document"])
        documents = cast(tuple[object, ...], result["documents"])

        self.assertEqual(selected["name"], "Ada")
        self.assertEqual(result["renamed"], "ready")
        self.assertEqual(constructed, {"name": "Ada", "status": "ready"})
        self.assertEqual(result["items"], ("Paris", 3))
        self.assertEqual(
            merged,
            {"shared": "node", "only_input": 1, "only_node": 2},
        )
        self.assertEqual(document["reference"], "/private/customer.pdf")
        self.assertNotIn("path", document)
        self.assertNotIn("bytes", document)
        self.assertEqual(len(documents), 2)
        self.assertEqual(
            cast(Mapping[str, object], documents[1])["reference"],
            "artifact-1",
        )

    def test_evaluate_flow_mappings_rejects_invalid_runtime_values(
        self,
    ) -> None:
        cases = (
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                    ),
                ),
                "flow.execution.missing_mapping_source",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.missing"),
                    ),
                ),
                "flow.execution.missing_selector_value",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.OBJECT,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.ARRAY,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.MERGE,
                    ),
                ),
                "flow.execution.empty_mapping",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.MERGE,
                        sources=(
                            parse_flow_selector("prepare.result.status"),
                        ),
                    ),
                ),
                "flow.execution.merge_requires_object",
            ),
            (
                (
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("input.payload.city"),
                    ),
                    FlowMappingPlan(
                        target="value",
                        kind=FlowMappingKind.SELECT,
                        source=parse_flow_selector("prepare.result.status"),
                    ),
                ),
                "flow.execution.duplicate_mapping_target",
            ),
        )

        for mappings, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowRuntimeEvaluationError) as raised:
                    evaluate_flow_mappings(mappings, self.context)

                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("Paris", str(raised.exception))

        mapping = FlowMappingPlan(
            target="value",
            kind=FlowMappingKind.SELECT,
            source=parse_flow_selector("input.payload.city"),
        )
        object.__setattr__(mapping, "kind", "unsupported")
        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_mappings((mapping,), self.context)
        self.assertEqual(
            raised.exception.code,
            "flow.execution.unsupported_mapping_kind",
        )
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings(
                [mapping],  # type: ignore[arg-type]
                self.context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings((object(),), self.context)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            evaluate_flow_mappings(
                (mapping,),
                "context",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_node_mappings(
                "node",  # type: ignore[arg-type]
                self.context,
            )

    def test_evaluate_flow_condition_plan_operators(self) -> None:
        cases = (
            (
                FlowConditionOperator.EQ,
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="ready",
                ),
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value="done",
                ),
            ),
            (
                FlowConditionOperator.NE,
                self._condition(
                    FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="done",
                ),
                self._condition(
                    FlowConditionOperator.NE,
                    selector="prepare.result.status",
                    value="ready",
                ),
            ),
            (
                FlowConditionOperator.EXISTS,
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.status",
                ),
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.missing",
                ),
            ),
            (
                FlowConditionOperator.NOT_EXISTS,
                self._condition(
                    FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.missing",
                ),
                self._condition(
                    FlowConditionOperator.NOT_EXISTS,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.IS_TYPE,
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.count",
                    value_type=FlowConditionValueType.INTEGER,
                ),
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                    value_type=FlowConditionValueType.INTEGER,
                ),
            ),
            (
                FlowConditionOperator.IN,
                self._condition(
                    FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
                self._condition(
                    FlowConditionOperator.IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
            ),
            (
                FlowConditionOperator.NOT_IN,
                self._condition(
                    FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("done", "failed"),
                ),
                self._condition(
                    FlowConditionOperator.NOT_IN,
                    selector="prepare.result.status",
                    values=("ready", "done"),
                ),
            ),
            (
                FlowConditionOperator.GT,
                self._condition(
                    FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=2,
                ),
                self._condition(
                    FlowConditionOperator.GT,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.GTE,
                self._condition(
                    FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                self._condition(
                    FlowConditionOperator.GTE,
                    selector="prepare.result.count",
                    value=4,
                ),
            ),
            (
                FlowConditionOperator.LT,
                self._condition(
                    FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=4,
                ),
                self._condition(
                    FlowConditionOperator.LT,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.LTE,
                self._condition(
                    FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=3,
                ),
                self._condition(
                    FlowConditionOperator.LTE,
                    selector="prepare.result.count",
                    value=2,
                ),
            ),
            (
                FlowConditionOperator.STARTS_WITH,
                self._condition(
                    FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Par",
                ),
                self._condition(
                    FlowConditionOperator.STARTS_WITH,
                    selector="input.payload.city",
                    value="Lon",
                ),
            ),
            (
                FlowConditionOperator.ENDS_WITH,
                self._condition(
                    FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="ris",
                ),
                self._condition(
                    FlowConditionOperator.ENDS_WITH,
                    selector="input.payload.city",
                    value="don",
                ),
            ),
            (
                FlowConditionOperator.CONTAINS,
                self._condition(
                    FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="ari",
                ),
                self._condition(
                    FlowConditionOperator.CONTAINS,
                    selector="input.payload.city",
                    value="zzz",
                ),
            ),
            (
                FlowConditionOperator.IS_NULL,
                self._condition(
                    FlowConditionOperator.IS_NULL,
                    selector="prepare.result.nullable",
                ),
                self._condition(
                    FlowConditionOperator.IS_NULL,
                    selector="prepare.result.status",
                ),
            ),
            (
                FlowConditionOperator.NOT_NULL,
                self._condition(
                    FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.status",
                ),
                self._condition(
                    FlowConditionOperator.NOT_NULL,
                    selector="prepare.result.nullable",
                ),
            ),
            (
                FlowConditionOperator.ALL,
                self._condition(
                    FlowConditionOperator.ALL,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                self._condition(
                    FlowConditionOperator.ALL,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="ready",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.ANY,
                self._condition(
                    FlowConditionOperator.ANY,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=3,
                        ),
                    ),
                ),
                self._condition(
                    FlowConditionOperator.ANY,
                    conditions=(
                        self._condition(
                            FlowConditionOperator.EQ,
                            selector="prepare.result.status",
                            value="done",
                        ),
                        self._condition(
                            FlowConditionOperator.GT,
                            selector="prepare.result.score",
                            value=4,
                        ),
                    ),
                ),
            ),
            (
                FlowConditionOperator.NOT,
                self._condition(
                    FlowConditionOperator.NOT,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="done",
                    ),
                ),
                self._condition(
                    FlowConditionOperator.NOT,
                    condition=self._condition(
                        FlowConditionOperator.EQ,
                        selector="prepare.result.status",
                        value="ready",
                    ),
                ),
            ),
        )

        for operator, true_condition, false_condition in cases:
            with self.subTest(operator=operator.value):
                self.assertTrue(
                    evaluate_flow_condition_plan(true_condition, self.context)
                )
                self.assertFalse(
                    evaluate_flow_condition_plan(false_condition, self.context)
                )

    def test_evaluate_flow_condition_plan_supports_values_and_types(
        self,
    ) -> None:
        value_selector = self._condition(
            FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value_selector="input.payload.expected",
        )
        literal_membership = self._condition(
            FlowConditionOperator.IN,
            selector="prepare.result.status",
            value=("ready", "done"),
        )
        scalar_membership = self._condition(
            FlowConditionOperator.IN,
            selector="prepare.result.status",
            value="ready",
        )
        type_cases = (
            ("prepare.result.status", FlowConditionValueType.STRING),
            ("prepare.result.count", FlowConditionValueType.INTEGER),
            ("prepare.result.score", FlowConditionValueType.NUMBER),
            ("input.payload.enabled", FlowConditionValueType.BOOLEAN),
            ("prepare.result.payload", FlowConditionValueType.OBJECT),
            ("prepare.result.tags", FlowConditionValueType.ARRAY),
            ("prepare.result.nullable", FlowConditionValueType.NULL),
        )

        self.assertTrue(
            evaluate_flow_condition_plan(value_selector, self.context)
        )
        self.assertTrue(
            evaluate_flow_condition_plan(literal_membership, self.context)
        )
        self.assertFalse(
            evaluate_flow_condition_plan(scalar_membership, self.context)
        )
        self.assertTrue(
            evaluate_flow_condition_plan(
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.tags[1]",
                    value="beta",
                ),
                self.context,
            )
        )
        for selector, value_type in type_cases:
            with self.subTest(value_type=value_type.value):
                self.assertTrue(
                    evaluate_flow_condition_plan(
                        self._condition(
                            FlowConditionOperator.IS_TYPE,
                            selector=selector,
                            value_type=value_type,
                        ),
                        self.context,
                    )
                )

    def test_evaluate_flow_condition_plan_reports_missing_values(
        self,
    ) -> None:
        cases = (
            (
                FlowConditionPlan(operator=FlowConditionOperator.EQ),
                "flow.condition_missing_selector",
            ),
            (
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.missing",
                    value="ready",
                ),
                "flow.condition_missing_value",
            ),
            (
                self._condition(
                    FlowConditionOperator.EQ,
                    selector="prepare.result.status",
                    value_selector="prepare.result.missing",
                ),
                "flow.condition_missing_value",
            ),
            (
                FlowConditionPlan(operator=FlowConditionOperator.NOT),
                "flow.condition_missing_child",
            ),
            (
                self._condition(
                    FlowConditionOperator.IS_TYPE,
                    selector="prepare.result.status",
                ),
                "flow.condition_missing_value_type",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowRuntimeEvaluationError) as raised:
                    evaluate_flow_condition_plan(condition, self.context)

                self.assertEqual(raised.exception.code, code)
                self.assertNotIn("ready", str(raised.exception))

    def test_evaluate_flow_condition_plan_handles_defensive_paths(
        self,
    ) -> None:
        numeric = self._condition(
            FlowConditionOperator.GT,
            selector="prepare.result.status",
            value=3,
        )
        string = self._condition(
            FlowConditionOperator.CONTAINS,
            selector="prepare.result.count",
            value="3",
        )
        unknown = self._condition(
            FlowConditionOperator.EQ,
            selector="prepare.result.status",
            value="ready",
        )
        object.__setattr__(unknown, "operator", "unknown")

        self.assertFalse(evaluate_flow_condition_plan(numeric, self.context))
        self.assertFalse(evaluate_flow_condition_plan(string, self.context))
        with self.assertRaises(FlowRuntimeEvaluationError) as raised:
            evaluate_flow_condition_plan(unknown, self.context)
        self.assertEqual(
            raised.exception.code,
            "flow.condition_unknown_operator",
        )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition_plan(
                "condition",  # type: ignore[arg-type]
                self.context,
            )
        with self.assertRaises(AssertionError):
            evaluate_flow_condition_plan(
                self._condition(
                    FlowConditionOperator.EXISTS,
                    selector="prepare.result.status",
                ),
                "context",  # type: ignore[arg-type]
            )

    def _condition(
        self,
        operator: FlowConditionOperator,
        *,
        selector: str | None = None,
        value: object | None = None,
        value_selector: str | None = None,
        values: tuple[object, ...] = (),
        value_type: FlowConditionValueType | None = None,
        conditions: tuple[FlowConditionPlan, ...] = (),
        condition: FlowConditionPlan | None = None,
    ) -> FlowConditionPlan:
        return FlowConditionPlan(
            operator=operator,
            selector=(
                parse_flow_selector(selector) if selector is not None else None
            ),
            value=value,
            value_selector=(
                parse_flow_selector(value_selector)
                if value_selector is not None
                else None
            ),
            values=values,
            value_type=value_type,
            conditions=conditions,
            condition=condition,
        )


if __name__ == "__main__":
    main()
