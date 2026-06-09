from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowCondition,
    FlowConditionOperator,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowGraphBindingInspection,
    FlowGraphBindingState,
    FlowGraphCompileResult,
    FlowGraphEdgeBinding,
    FlowGraphEdgeClassification,
    FlowGraphEdgeInspection,
    FlowGraphFormat,
    FlowGraphInspection,
    FlowGraphMode,
    FlowGraphNodeClassification,
    FlowGraphNodeInspection,
    FlowGraphSource,
    FlowGraphSourceKind,
    FlowRouteMatchPolicy,
    FlowSourceSpan,
)


class FlowGraphModelsTestCase(TestCase):
    def test_graph_source_serializes_privacy_safe_public_shape(self) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="graph TD\nprivate_customer_token@--> done",
            source_identity="inline",
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=7,
                start_column=1,
            ),
        )

        self.assertEqual(
            source.as_public_dict(),
            {
                "format": "mermaid",
                "source_kind": "inline",
                "mode": "executable",
                "source_span": {"start_line": 7, "start_column": 1},
            },
        )
        self.assertNotIn(
            "private_customer_token", str(source.as_public_dict())
        )
        self.assertNotIn("/private/customer", str(source.as_public_dict()))
        with self.assertRaises(FrozenInstanceError):
            source.diagram = "changed"  # type: ignore[misc]

    def test_graph_source_accepts_file_source_without_public_path(
        self,
    ) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.FILE,
            path=Path("/private/customer/graph.mmd"),
            base_path=Path("/private/customer"),
        )

        self.assertEqual(
            source.as_public_dict(),
            {
                "format": "mermaid",
                "source_kind": "file",
                "mode": "executable",
            },
        )
        self.assertNotIn("graph.mmd", str(source.as_public_dict()))
        self.assertNotIn("/private/customer", str(source.as_public_dict()))

    def test_graph_source_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {"source_kind": FlowGraphSourceKind.INLINE},
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "diagram": "graph TD",
                "path": Path("graph.mmd"),
            },
            {"source_kind": FlowGraphSourceKind.FILE},
            {
                "source_kind": FlowGraphSourceKind.FILE,
                "diagram": "graph TD",
                "path": Path("graph.mmd"),
            },
            {
                "source_kind": "inline",
                "diagram": "graph TD",
            },
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "format": "mermaid",
                "diagram": "graph TD",
            },
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "mode": "executable",
                "diagram": "graph TD",
            },
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "diagram": "",
            },
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "diagram": "graph TD",
                "source_identity": "",
            },
            {
                "source_kind": FlowGraphSourceKind.INLINE,
                "diagram": "graph TD",
                "source_span": object(),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphSource(**case)

    def test_edge_binding_freezes_metadata_and_serializes_fields(self) -> None:
        metadata_tags = ["primary"]
        condition_metadata = {"op": "exists", "tags": metadata_tags}
        metadata: dict[str, object] = {
            "label": "ok",
            "condition": condition_metadata,
        }
        binding = FlowGraphEdgeBinding(
            edge_id="route_1",
            metadata=metadata,
            label="success",
            kind=FlowEdgeKind.FINALLY,
            condition=FlowCondition(
                operator=FlowConditionOperator.EXISTS,
                selector="start.result",
            ),
            priority=3,
            default=False,
            routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=12,
                start_column=3,
            ),
        )

        condition_metadata["op"] = "changed"
        metadata_tags.append("changed")

        self.assertEqual(binding.edge_id, "route_1")
        self.assertEqual(
            cast(dict[str, object], binding.metadata["condition"])["op"],
            "exists",
        )
        self.assertEqual(
            cast(dict[str, object], binding.metadata["condition"])["tags"],
            ("primary",),
        )
        self.assertEqual(
            binding.as_public_dict(),
            {
                "edge_id": "route_1",
                "metadata_fields": ("condition", "label"),
                "source_span": {"start_line": 12, "start_column": 3},
            },
        )
        self.assertNotIn("success", str(binding.as_public_dict()))
        self.assertNotIn("/private/customer", str(binding.as_public_dict()))
        with self.assertRaises(TypeError):
            binding.metadata["label"] = "changed"  # type: ignore[index]

    def test_edge_binding_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {"edge_id": ""},
            {"edge_id": "route", "metadata": {"source": "start"}},
            {"edge_id": "route", "metadata": {"target": "end"}},
            {"edge_id": "route", "metadata": {"unknown": True}},
            {"edge_id": "route", "metadata": {"": True}},
            {"edge_id": "route", "metadata": []},
            {"edge_id": "route", "label": ""},
            {"edge_id": "route", "kind": "success"},
            {"edge_id": "route", "condition": object()},
            {"edge_id": "route", "priority": True},
            {"edge_id": "route", "default": "false"},
            {"edge_id": "route", "routing_policy": "exclusive"},
            {"edge_id": "route", "source_span": object()},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphEdgeBinding(**case)

    def test_compile_result_freezes_bindings_and_projects_diagnostics(
        self,
    ) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="graph TD\nA route@--> B",
        )
        edge = FlowEdgeDefinition(source="A", target="B")
        binding = FlowGraphEdgeBinding(edge_id="route")
        diagnostic = FlowDiagnostic(
            code="flow.graph.missing_source",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            path="graph",
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=2,
                start_column=1,
            ),
            message="Graph source is missing.",
        )
        result = FlowGraphCompileResult(
            source=source,
            edges=(edge,),
            edge_bindings={"route": binding},
            diagnostics=(diagnostic,),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.public_diagnostics,
            (
                {
                    "code": "flow.graph.missing_source",
                    "category": "graph_compiler",
                    "severity": "error",
                    "message": "Graph source is missing.",
                    "path": "graph",
                    "source_span": {"start_line": 2, "start_column": 1},
                },
            ),
        )
        self.assertEqual(
            result.as_public_dict(),
            {
                "ok": False,
                "edge_count": 1,
                "edge_binding_count": 1,
                "source": {
                    "format": "mermaid",
                    "source_kind": "inline",
                    "mode": "executable",
                },
                "diagnostics": result.public_diagnostics,
            },
        )
        public = str(result.as_public_dict())
        self.assertNotIn("route@-->", public)
        self.assertNotIn("/private/customer", public)
        with self.assertRaises(TypeError):
            result.edge_bindings["route"] = binding  # type: ignore[index]

    def test_compile_result_accepts_warning_only_diagnostics(self) -> None:
        result = FlowGraphCompileResult(
            diagnostics=(
                FlowDiagnostic(
                    code="flow.graph.warning",
                    category=FlowDiagnosticCategory.GRAPH_COMPILER,
                    path="graph",
                    severity=FlowDiagnosticSeverity.WARNING,
                    message="Graph warning.",
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_compile_result_rejects_invalid_values(self) -> None:
        binding = FlowGraphEdgeBinding(edge_id="route")
        invalid_cases = (
            {"source": object()},
            {"edges": [FlowEdgeDefinition(source="A", target="B")]},
            {"edges": (object(),)},
            {"edge_bindings": []},
            {"edge_bindings": {"": binding}},
            {"edge_bindings": {"other": binding}},
            {"edge_bindings": {"route": object()}},
            {"diagnostics": [object()]},
            {"diagnostics": (object(),)},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphCompileResult(**case)

    def test_compile_result_defaults_to_success_public_summary(self) -> None:
        result = FlowGraphCompileResult()

        self.assertTrue(result.ok)
        self.assertEqual(result.public_diagnostics, ())
        self.assertEqual(
            result.as_public_dict(),
            {
                "ok": True,
                "edge_count": 0,
                "edge_binding_count": 0,
            },
        )
        self.assertIs(result.source, None)
        self.assertEqual(result.edges, ())
        self.assertEqual(result.edge_bindings, {})

    def test_node_inspection_serializes_classification_safely(self) -> None:
        actual = FlowGraphNodeInspection(
            id="review",
            classification=FlowGraphNodeClassification.ACTUAL,
            strict_node="review",
            source_span=FlowSourceSpan(
                source="/private/customer/graph.mmd",
                start_line=4,
                start_column=9,
            ),
        )
        decorative = FlowGraphNodeInspection(
            id="private_label_container",
            classification=FlowGraphNodeClassification.DECORATIVE,
        )

        self.assertEqual(
            actual.as_public_dict(),
            {
                "id": "review",
                "classification": "actual",
                "strict_node": "review",
                "source_span": {"start_line": 4, "start_column": 9},
            },
        )
        self.assertEqual(
            decorative.as_public_dict(),
            {
                "id": "private_label_container",
                "classification": "decorative",
            },
        )
        self.assertNotIn(
            "/private/customer",
            str(actual.as_public_dict()),
        )
        with self.assertRaises(FrozenInstanceError):
            actual.strict_node = "changed"  # type: ignore[misc]

    def test_node_inspection_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {
                "id": "",
                "classification": FlowGraphNodeClassification.ACTUAL,
            },
            {"id": "node", "classification": "actual"},
            {
                "id": "node",
                "classification": FlowGraphNodeClassification.ACTUAL,
                "strict_node": "",
            },
            {
                "id": "node",
                "classification": FlowGraphNodeClassification.DECORATIVE,
                "strict_node": "node",
            },
            {
                "id": "node",
                "classification": FlowGraphNodeClassification.ACTUAL,
                "source_span": object(),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphNodeInspection(**case)

    def test_edge_inspection_serializes_classification_safely(self) -> None:
        edge = FlowGraphEdgeInspection(
            index=2,
            source="start",
            target="review",
            classification=FlowGraphEdgeClassification.EXECUTABLE,
            edge_id="route_2",
            source_span=FlowSourceSpan(
                source="/private/customer/graph.mmd",
                start_line=5,
                start_column=3,
            ),
            bidirectional=True,
        )

        self.assertEqual(
            edge.as_public_dict(),
            {
                "index": 2,
                "source": "start",
                "target": "review",
                "classification": "executable",
                "edge_id": "route_2",
                "bidirectional": True,
                "source_span": {"start_line": 5, "start_column": 3},
            },
        )
        self.assertNotIn("/private/customer", str(edge.as_public_dict()))

    def test_edge_inspection_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {
                "index": True,
                "source": "start",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
            },
            {
                "index": -1,
                "source": "start",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
            },
            {
                "index": 0,
                "source": "",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
            },
            {
                "index": 0,
                "source": "start",
                "target": "",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
            },
            {
                "index": 0,
                "source": "start",
                "target": "review",
                "classification": "executable",
            },
            {
                "index": 0,
                "source": "start",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
                "edge_id": "",
            },
            {
                "index": 0,
                "source": "start",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
                "source_span": object(),
            },
            {
                "index": 0,
                "source": "start",
                "target": "review",
                "classification": FlowGraphEdgeClassification.EXECUTABLE,
                "bidirectional": "false",
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphEdgeInspection(**case)

    def test_binding_inspection_sorts_fields_and_serializes_state(
        self,
    ) -> None:
        binding = FlowGraphBindingInspection(
            edge_id="route_1",
            state=FlowGraphBindingState.REJECTED,
            metadata_fields=("routing_policy", "condition"),
            diagnostic_codes=("flow.graph.invalid_edge_id",),
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=8,
                start_column=3,
            ),
        )

        self.assertEqual(
            binding.metadata_fields, ("condition", "routing_policy")
        )
        self.assertEqual(
            binding.as_public_dict(),
            {
                "edge_id": "route_1",
                "state": "rejected",
                "metadata_fields": ("condition", "routing_policy"),
                "diagnostic_codes": ("flow.graph.invalid_edge_id",),
                "source_span": {"start_line": 8, "start_column": 3},
            },
        )
        self.assertNotIn("/private/customer", str(binding.as_public_dict()))

    def test_binding_inspection_rejects_invalid_values(self) -> None:
        invalid_cases = (
            {"edge_id": "", "state": FlowGraphBindingState.BOUND},
            {"edge_id": "route", "state": "bound"},
            {
                "edge_id": "route",
                "state": FlowGraphBindingState.BOUND,
                "metadata_fields": ["label"],
            },
            {
                "edge_id": "route",
                "state": FlowGraphBindingState.BOUND,
                "metadata_fields": ("",),
            },
            {
                "edge_id": "route",
                "state": FlowGraphBindingState.BOUND,
                "diagnostic_codes": ["flow.graph.invalid"],
            },
            {
                "edge_id": "route",
                "state": FlowGraphBindingState.BOUND,
                "diagnostic_codes": ("",),
            },
            {
                "edge_id": "route",
                "state": FlowGraphBindingState.BOUND,
                "source_span": object(),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphBindingInspection(**case)

    def test_graph_inspection_serializes_deterministic_public_shape(
        self,
    ) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="graph TD\nstart route_1@--> review",
            source_identity="/private/customer/flow.toml",
        )
        diagnostic = FlowDiagnostic(
            code="flow.graph.decorative_edge_metadata",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            path="graph.edges.route_2",
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=12,
                start_column=5,
            ),
            message="Graph edge metadata targets a decorative edge.",
        )
        inspection = FlowGraphInspection(
            source=source,
            diagnostics=(diagnostic,),
            nodes=(
                FlowGraphNodeInspection(
                    id="start",
                    classification=FlowGraphNodeClassification.ACTUAL,
                    strict_node="start",
                ),
                FlowGraphNodeInspection(
                    id="note",
                    classification=FlowGraphNodeClassification.DECORATIVE,
                ),
            ),
            edges=(
                FlowGraphEdgeInspection(
                    index=0,
                    source="start",
                    target="review",
                    classification=FlowGraphEdgeClassification.EXECUTABLE,
                    edge_id="route_1",
                ),
            ),
            bindings=(
                FlowGraphBindingInspection(
                    edge_id="route_1",
                    state=FlowGraphBindingState.BOUND,
                    metadata_fields=("label",),
                ),
                FlowGraphBindingInspection(
                    edge_id="route_2",
                    state=FlowGraphBindingState.DECORATIVE,
                    diagnostic_codes=("flow.graph.decorative_edge_metadata",),
                ),
            ),
            generated_edges=(
                FlowEdgeDefinition(
                    source="start",
                    target="review",
                    label="private approval label",
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="private.token",
                    ),
                    priority=4,
                    default=True,
                    routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                ),
            ),
        )

        self.assertEqual(
            inspection.as_public_dict(),
            {
                "schema_version": "flow.graph.inspection.v1",
                "nodes": (
                    {
                        "id": "start",
                        "classification": "actual",
                        "strict_node": "start",
                    },
                    {"id": "note", "classification": "decorative"},
                ),
                "edges": (
                    {
                        "index": 0,
                        "source": "start",
                        "target": "review",
                        "classification": "executable",
                        "edge_id": "route_1",
                    },
                ),
                "bindings": (
                    {
                        "edge_id": "route_1",
                        "state": "bound",
                        "metadata_fields": ("label",),
                    },
                    {
                        "edge_id": "route_2",
                        "state": "decorative",
                        "diagnostic_codes": (
                            "flow.graph.decorative_edge_metadata",
                        ),
                    },
                ),
                "generated_edges": (
                    {
                        "index": 0,
                        "source": "start",
                        "target": "review",
                        "kind": "success",
                        "priority": 4,
                        "default": True,
                        "routing_policy": "all_matching",
                        "has_label": True,
                        "has_condition": True,
                    },
                ),
                "source": {
                    "format": "mermaid",
                    "source_kind": "inline",
                    "mode": "executable",
                },
                "diagnostics": (
                    {
                        "code": "flow.graph.decorative_edge_metadata",
                        "category": "graph_compiler",
                        "severity": "error",
                        "message": (
                            "Graph edge metadata targets a decorative edge."
                        ),
                        "path": "graph.edges.route_2",
                        "source_span": {
                            "start_line": 12,
                            "start_column": 5,
                        },
                    },
                ),
            },
        )
        public = str(inspection.as_public_dict())
        self.assertNotIn("route_1@-->", public)
        self.assertNotIn("/private/customer", public)
        self.assertNotIn("private approval label", public)
        self.assertNotIn("private.token", public)

    def test_graph_inspection_rejects_invalid_values(self) -> None:
        node = FlowGraphNodeInspection(
            id="start",
            classification=FlowGraphNodeClassification.ACTUAL,
        )
        edge = FlowGraphEdgeInspection(
            index=0,
            source="start",
            target="end",
            classification=FlowGraphEdgeClassification.EXECUTABLE,
        )
        binding = FlowGraphBindingInspection(
            edge_id="route",
            state=FlowGraphBindingState.BOUND,
        )
        strict_edge = FlowEdgeDefinition(source="start", target="end")
        diagnostic = FlowDiagnostic(
            code="flow.graph.warning",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            path="graph",
            severity=FlowDiagnosticSeverity.WARNING,
            message="Graph warning.",
        )
        invalid_cases = (
            {"schema_version": ""},
            {"source": object()},
            {"diagnostics": [diagnostic]},
            {"diagnostics": (object(),)},
            {"nodes": [node]},
            {"nodes": (object(),)},
            {"nodes": (node, node)},
            {"edges": [edge]},
            {"edges": (object(),)},
            {"edges": (edge, edge)},
            {"bindings": [binding]},
            {"bindings": (object(),)},
            {"generated_edges": [strict_edge]},
            {"generated_edges": (object(),)},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowGraphInspection(**case)

    def test_graph_inspection_defaults_to_empty_public_shape(self) -> None:
        inspection = FlowGraphInspection()

        self.assertEqual(inspection.public_diagnostics, ())
        self.assertEqual(
            inspection.as_public_dict(),
            {
                "schema_version": "flow.graph.inspection.v1",
                "nodes": (),
                "edges": (),
                "bindings": (),
                "generated_edges": (),
            },
        )

    def test_graph_model_enums_remain_stable(self) -> None:
        self.assertEqual(FlowGraphBindingState.BOUND.value, "bound")
        self.assertEqual(FlowGraphBindingState.UNBOUND.value, "unbound")
        self.assertEqual(FlowGraphBindingState.MISSING.value, "missing")
        self.assertEqual(FlowGraphBindingState.DECORATIVE.value, "decorative")
        self.assertEqual(FlowGraphBindingState.REJECTED.value, "rejected")
        self.assertEqual(FlowGraphEdgeClassification.EXECUTABLE, "executable")
        self.assertEqual(FlowGraphEdgeClassification.DECORATIVE, "decorative")
        self.assertEqual(FlowGraphFormat.MERMAID.value, "mermaid")
        self.assertEqual(FlowGraphSourceKind.INLINE.value, "inline")
        self.assertEqual(FlowGraphSourceKind.FILE.value, "file")
        self.assertEqual(FlowGraphMode.EXECUTABLE.value, "executable")
        self.assertEqual(FlowGraphNodeClassification.ACTUAL.value, "actual")
        self.assertEqual(
            FlowGraphNodeClassification.DECORATIVE.value,
            "decorative",
        )


if __name__ == "__main__":
    main()
