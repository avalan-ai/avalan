from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticCodePrefix,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
    all_flow_diagnostic_code_prefixes,
    flow_diagnostic_code_prefixes,
)


class FlowDiagnosticsTestCase(TestCase):
    def test_source_span_serializes_internal_and_public_shapes(self) -> None:
        span = FlowSourceSpan(
            source="/private/customer/flow.toml",
            start_line=2,
            start_column=4,
            end_line=3,
            end_column=9,
        )

        self.assertEqual(
            span.as_dict(),
            {
                "start_line": 2,
                "start_column": 4,
                "end_line": 3,
                "end_column": 9,
                "source": "/private/customer/flow.toml",
            },
        )
        self.assertEqual(
            span.as_public_dict(),
            {
                "start_line": 2,
                "start_column": 4,
                "end_line": 3,
                "end_column": 9,
            },
        )
        self.assertNotIn("customer", str(span.as_public_dict()))
        with self.assertRaises(FrozenInstanceError):
            span.start_line = 1  # type: ignore[misc]

    def test_source_span_serializes_minimal_public_shape(self) -> None:
        self.assertEqual(
            FlowSourceSpan(start_line=1, start_column=1).as_dict(),
            {"start_line": 1, "start_column": 1},
        )

    def test_source_span_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {"start_line": 0, "start_column": 1},
            {"start_line": 1, "start_column": 0},
            {"start_line": 1, "start_column": 1, "source": ""},
            {"start_line": 2, "start_column": 1, "end_line": 1},
            {"start_line": 1, "start_column": 1, "end_column": 0},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowSourceSpan(**case)

    def test_diagnostic_serializes_internal_and_public_shapes(self) -> None:
        source_span = FlowSourceSpan(
            source="/private/customer/flow.toml",
            start_line=4,
            start_column=5,
        )
        related_span = FlowSourceSpan(
            source="/private/customer/flow.toml",
            start_line=7,
            start_column=1,
        )
        diagnostic = FlowDiagnostic(
            code="flow.definition.missing_node",
            path="nodes.start",
            category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
            severity=FlowDiagnosticSeverity.WARNING,
            source_span=source_span,
            message="Flow node is missing.",
            hint="Add the node.",
            related_spans=(related_span,),
        )

        self.assertEqual(
            diagnostic.as_dict(),
            {
                "code": "flow.definition.missing_node",
                "category": "flow_definition_validation",
                "severity": "warning",
                "message": "Flow node is missing.",
                "path": "nodes.start",
                "source_span": {
                    "start_line": 4,
                    "start_column": 5,
                    "source": "/private/customer/flow.toml",
                },
                "hint": "Add the node.",
                "related_spans": (
                    {
                        "start_line": 7,
                        "start_column": 1,
                        "source": "/private/customer/flow.toml",
                    },
                ),
            },
        )
        self.assertEqual(
            diagnostic.as_public_dict(),
            {
                "code": "flow.definition.missing_node",
                "category": "flow_definition_validation",
                "severity": "warning",
                "message": "Flow node is missing.",
                "path": "nodes.start",
                "source_span": {"start_line": 4, "start_column": 5},
                "hint": "Add the node.",
                "related_spans": ({"start_line": 7, "start_column": 1},),
            },
        )
        self.assertNotIn("customer", str(diagnostic.as_public_dict()))

    def test_diagnostic_serializes_minimal_shape(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.execution.failed",
            path="nodes.start",
            category=FlowDiagnosticCategory.EXECUTION,
            message="Flow node failed.",
        )

        self.assertEqual(
            diagnostic.as_dict(),
            {
                "code": "flow.execution.failed",
                "category": "execution",
                "severity": "error",
                "message": "Flow node failed.",
                "path": "nodes.start",
            },
        )

    def test_diagnostic_accepts_source_span_without_path(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.invalid_arrow",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=FlowSourceSpan(start_line=1, start_column=3),
            severity=FlowDiagnosticSeverity.INFO,
            message="Mermaid arrow is not supported.",
        )

        self.assertNotIn("path", diagnostic.as_public_dict())
        self.assertEqual(diagnostic.as_public_dict()["severity"], "info")

    def test_diagnostic_rejects_invalid_values(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        invalid_cases = (
            {
                "code": "",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "message": "",
            },
            {
                "code": "flow.execution.failed",
                "path": "",
                "category": FlowDiagnosticCategory.EXECUTION,
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "category": FlowDiagnosticCategory.EXECUTION,
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": "execution",
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "severity": "error",
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "source_span": object(),
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "hint": "",
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "related_spans": [span],
                "message": "Message.",
            },
            {
                "code": "flow.execution.failed",
                "path": "flow",
                "category": FlowDiagnosticCategory.EXECUTION,
                "related_spans": (object(),),
                "message": "Message.",
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowDiagnostic(**case)

    def test_code_prefixes_are_stable_by_category(self) -> None:
        prefixes = all_flow_diagnostic_code_prefixes()

        self.assertEqual(
            prefixes[FlowDiagnosticCategory.MERMAID_PARSER],
            (FlowDiagnosticCodePrefix.MERMAID_PARSER,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(
                FlowDiagnosticCategory.MERMAID_SECURITY
            ),
            (FlowDiagnosticCodePrefix.MERMAID_SECURITY,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(
                FlowDiagnosticCategory.FLOW_VIEW_BINDING
            ),
            (FlowDiagnosticCodePrefix.FLOW_VIEW_BINDING,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(
                FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION
            ),
            (FlowDiagnosticCodePrefix.FLOW_DEFINITION_VALIDATION,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(FlowDiagnosticCategory.EXECUTION),
            (FlowDiagnosticCodePrefix.EXECUTION,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(FlowDiagnosticCategory.PRIVACY),
            (FlowDiagnosticCodePrefix.PRIVACY,),
        )
        self.assertEqual(
            flow_diagnostic_code_prefixes(
                FlowDiagnosticCategory.TASK_DURABILITY
            ),
            (FlowDiagnosticCodePrefix.TASK_DURABILITY,),
        )
        with self.assertRaises(TypeError):
            prefixes[FlowDiagnosticCategory.EXECUTION] = ()  # type: ignore[index]
        with self.assertRaises(AssertionError):
            flow_diagnostic_code_prefixes("execution")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
