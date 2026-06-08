from unittest import TestCase, main

from avalan.flow import (
    FlowSelector,
    FlowSelectorError,
    FlowSelectorRoot,
    FlowSelectorStep,
    FlowSelectorStepKind,
    parse_flow_selector,
)


class FlowSelectorTestCase(TestCase):
    def test_parse_flow_selector_accepts_node_output_paths(self) -> None:
        selector = parse_flow_selector("prepare.result.items[0].name")

        self.assertEqual(
            selector,
            FlowSelector(
                root=FlowSelectorRoot.NODE_OUTPUT,
                source="prepare",
                output="result",
                path=(
                    FlowSelectorStep(
                        kind=FlowSelectorStepKind.FIELD,
                        value="items",
                    ),
                    FlowSelectorStep(
                        kind=FlowSelectorStepKind.INDEX,
                        value=0,
                    ),
                    FlowSelectorStep(
                        kind=FlowSelectorStepKind.FIELD,
                        value="name",
                    ),
                ),
            ),
        )

    def test_parse_flow_selector_accepts_flow_input_paths(self) -> None:
        selector = parse_flow_selector("inputs.payload.customer.tags[1]")

        self.assertEqual(selector.root, FlowSelectorRoot.FLOW_INPUT)
        self.assertEqual(selector.source, "payload")
        self.assertIsNone(selector.output)
        self.assertEqual(
            selector.path,
            (
                FlowSelectorStep(
                    kind=FlowSelectorStepKind.FIELD,
                    value="customer",
                ),
                FlowSelectorStep(
                    kind=FlowSelectorStepKind.FIELD,
                    value="tags",
                ),
                FlowSelectorStep(kind=FlowSelectorStepKind.INDEX, value=1),
            ),
        )

    def test_parse_flow_selector_restricts_allowed_roots(self) -> None:
        with self.assertRaises(FlowSelectorError) as raised:
            parse_flow_selector(
                "inputs.payload",
                allowed_roots=frozenset({FlowSelectorRoot.NODE_OUTPUT}),
            )

        self.assertEqual(raised.exception.code, "flow.invalid_selector")

    def test_parse_flow_selector_rejects_invalid_shapes(self) -> None:
        cases = (
            "",
            "start",
            "start.",
            "start..result",
            "start[0].result",
            "start.result[-1]",
            "start.result[]",
            "inputs",
            "inputs[0].payload",
        )

        for selector in cases:
            with self.subTest(selector=selector):
                with self.assertRaises(FlowSelectorError) as raised:
                    parse_flow_selector(selector)

                self.assertEqual(
                    raised.exception.code, "flow.invalid_selector"
                )

    def test_selector_step_validates_index_values(self) -> None:
        self.assertEqual(
            FlowSelectorStep(kind=FlowSelectorStepKind.INDEX, value=0).value,
            0,
        )
        with self.assertRaises(AssertionError):
            FlowSelectorStep(
                kind=FlowSelectorStepKind.INDEX,
                value=-1,
            )

    def test_parse_flow_selector_rejects_private_or_unsafe_shapes(
        self,
    ) -> None:
        cases = (
            ("env.SECRET", "flow.reserved_selector"),
            ("environment.SECRET", "flow.reserved_selector"),
            ("fs.root", "flow.reserved_selector"),
            ("files.content", "flow.reserved_selector"),
            ("file.content", "flow.reserved_selector"),
            ("network.response", "flow.reserved_selector"),
            ("runtime.node_state", "flow.reserved_selector"),
            ("secret.key", "flow.reserved_selector"),
            ("secrets.key", "flow.reserved_selector"),
            ("task.run_id", "flow.reserved_selector"),
            ("__task_files__.files", "flow.reserved_selector"),
            ("start.__dict__", "flow.reserved_selector"),
            ("start.__class__", "flow.reserved_selector"),
            ("start.result/{{secret}}", "flow.unsafe_selector"),
            ("start.result.${secret}", "flow.unsafe_selector"),
            ("start.result.$(secret)", "flow.unsafe_selector"),
            ("start.result.{% secret %}", "flow.unsafe_selector"),
            ("https://host/value", "flow.unsafe_selector"),
            ("C:\\secret\\value", "flow.unsafe_selector"),
            ("~/secret/value", "flow.unsafe_selector"),
            ("start.result../secret", "flow.unsafe_selector"),
        )

        for selector, code in cases:
            with self.subTest(selector=selector):
                with self.assertRaises(FlowSelectorError) as raised:
                    parse_flow_selector(selector)

                self.assertEqual(raised.exception.code, code)

    def test_selector_entities_reject_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            FlowSelectorStep(
                kind="field",  # type: ignore[arg-type]
                value="name",
            )
        with self.assertRaises(AssertionError):
            FlowSelectorStep(kind=FlowSelectorStepKind.FIELD, value=0)
        with self.assertRaises(AssertionError):
            FlowSelectorStep(kind=FlowSelectorStepKind.INDEX, value=-1)
        with self.assertRaises(AssertionError):
            FlowSelector(
                root="node_output",  # type: ignore[arg-type]
                source="start",
            )
        with self.assertRaises(AssertionError):
            FlowSelector(root=FlowSelectorRoot.NODE_OUTPUT, source="")
        with self.assertRaises(AssertionError):
            FlowSelector(
                root=FlowSelectorRoot.NODE_OUTPUT,
                source="start",
                output="",
            )
        with self.assertRaises(AssertionError):
            FlowSelector(
                root=FlowSelectorRoot.NODE_OUTPUT,
                source="start",
                path=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowSelectorError("")
        with self.assertRaises(AssertionError):
            parse_flow_selector(3)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_flow_selector(
                "start.result",
                allowed_roots={FlowSelectorRoot.NODE_OUTPUT},  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            parse_flow_selector(
                "start.result",
                allowed_roots=frozenset({"node_output"}),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    main()
