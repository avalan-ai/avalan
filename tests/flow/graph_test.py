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
    FlowGraphCompileResult,
    FlowGraphEdgeBinding,
    FlowGraphFormat,
    FlowGraphMode,
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

    def test_graph_model_enums_remain_stable(self) -> None:
        self.assertEqual(FlowGraphFormat.MERMAID.value, "mermaid")
        self.assertEqual(FlowGraphSourceKind.INLINE.value, "inline")
        self.assertEqual(FlowGraphSourceKind.FILE.value, "file")
        self.assertEqual(FlowGraphMode.EXECUTABLE.value, "executable")


if __name__ == "__main__":
    main()
