from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import TestCase, main
from unittest.mock import patch

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
    FlowGraphDiagnosticCode,
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
    FlowNodeDefinition,
    FlowRouteMatchPolicy,
    FlowSourceSpan,
    compile_flow_graph,
    flow_graph_diagnostic,
    flow_graph_diagnostic_load_category,
)

_GRAPH_DIAGNOSTIC_CASES = (
    (
        FlowGraphDiagnosticCode.MALFORMED_SOURCE,
        "flow.graph.malformed_source",
        "Graph source is malformed.",
        "Fix the graph source syntax.",
        "parse",
    ),
    (
        FlowGraphDiagnosticCode.UNSUPPORTED_FORMAT,
        "flow.graph.unsupported_format",
        "Graph format is not supported.",
        "Use a supported graph format.",
        "unsupported",
    ),
    (
        FlowGraphDiagnosticCode.UNSUPPORTED_SOURCE,
        "flow.graph.unsupported_source",
        "Graph source type is not supported.",
        "Use an inline diagram or local file path.",
        "unsupported",
    ),
    (
        FlowGraphDiagnosticCode.UNSUPPORTED_MODE,
        "flow.graph.unsupported_mode",
        "Graph mode is not supported.",
        "Use executable graph mode.",
        "unsupported",
    ),
    (
        FlowGraphDiagnosticCode.MISSING_SOURCE,
        "flow.graph.missing_source",
        "Graph source is missing.",
        "Provide exactly one graph source.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.READ_FAILURE,
        "flow.graph.read_failure",
        "Graph source could not be read.",
        "Check that the graph file is available to the loader.",
        "parse",
    ),
    (
        FlowGraphDiagnosticCode.PATH_ESCAPE,
        "flow.graph.path_escape",
        "Graph source path is outside the allowed base.",
        "Use a graph path inside the flow definition directory.",
        "privacy",
    ),
    (
        FlowGraphDiagnosticCode.SOURCE_CONFLICT,
        "flow.graph.source_conflict",
        "Graph source is ambiguous.",
        "Provide exactly one graph source.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.EDGE_CONFLICT,
        "flow.graph.edge_conflict",
        "Graph edges conflict with strict edges.",
        "Use either graph authoring or strict edge definitions.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.MISSING_EDGE_METADATA_TARGET,
        "flow.graph.missing_edge_metadata",
        "Graph edge metadata targets a missing edge.",
        "Bind graph edge metadata to an explicit Mermaid edge ID.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.DECORATIVE_EDGE_METADATA_TARGET,
        "flow.graph.decorative_edge_metadata",
        "Graph edge metadata targets a decorative edge.",
        "Bind graph edge metadata only to executable graph edges.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.DUPLICATE_EDGE_ID,
        "flow.graph.duplicate_edge_id",
        "Graph edge ID is duplicated.",
        "Use unique explicit Mermaid edge IDs.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.INVALID_EDGE_ID,
        "flow.graph.invalid_edge_id",
        "Graph edge ID is invalid.",
        "Use a TOML-key-safe Mermaid edge ID.",
        "value",
    ),
    (
        FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE,
        "flow.graph.unsupported_executable_edge",
        "Graph edge is not supported for execution.",
        "Use explicit directed graph edges for executable routes.",
        "unsupported",
    ),
)


class FlowGraphCompilerTestCase(TestCase):
    def test_compile_flow_graph_compiles_inline_mermaid_edges(self) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="\n".join(
                (
                    "flowchart LR",
                    "start route_1@--> review",
                    "review route_2@--> done",
                    "review -.-> note",
                    "subgraph lane[Private customer label]",
                    "done route_3@--> archive",
                    "end",
                )
            ),
            source_identity="/private/customer/flow.toml",
            source_span=FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=9,
                start_column=1,
            ),
        )
        binding = FlowGraphEdgeBinding(edge_id="route_1", label="review")

        result = compile_flow_graph(
            source,
            (
                FlowNodeDefinition(name="start", type="input"),
                FlowNodeDefinition(name="review", type="pass-through"),
                FlowNodeDefinition(name="done", type="pass-through"),
                FlowNodeDefinition(name="archive", type="pass-through"),
            ),
            edge_bindings={"route_1": binding},
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.edge_bindings["route_1"], binding)
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.edges],
            [
                ("start", "review"),
                ("review", "done"),
                ("done", "archive"),
            ],
        )
        self.assertNotIn(
            "Private customer label",
            str(result.as_public_dict()),
        )
        self.assertNotIn("/private/customer", str(result.as_public_dict()))

    def test_compile_flow_graph_reports_malformed_inline_source_safely(
        self,
    ) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="flowchart LR\nstart route@ PrivateCustomerToken",
            source_identity="/private/customer/flow.toml",
        )

        result = compile_flow_graph(
            source,
            (
                FlowNodeDefinition(name="start", type="input"),
                FlowNodeDefinition(name="done", type="pass-through"),
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.graph.malformed_source"],
        )
        self.assertEqual(result.diagnostics[0].source_span.start_line, 2)
        self.assertEqual(result.diagnostics[0].source_span.start_column, 7)
        self.assertNotIn(
            "PrivateCustomerToken", str(result.public_diagnostics)
        )
        self.assertNotIn("/private/customer", str(result.public_diagnostics))

    def test_compile_flow_graph_compiles_file_mermaid_source(self) -> None:
        with TemporaryDirectory() as directory:
            base_path = Path(directory) / "flows"
            graph_directory = base_path / "graphs"
            graph_directory.mkdir(parents=True)
            graph_path = graph_directory / "customer-token.mmd"
            graph_path.write_text(
                "\n".join(
                    (
                        "flowchart LR",
                        "start route_1@--> done",
                        "done route_2@--> archive",
                    )
                ),
                encoding="utf-8",
            )
            source = FlowGraphSource(
                source_kind=FlowGraphSourceKind.FILE,
                path=Path("graphs/customer-token.mmd"),
                base_path=base_path,
            )

            result = compile_flow_graph(
                source,
                (
                    FlowNodeDefinition(name="start", type="input"),
                    FlowNodeDefinition(name="done", type="pass-through"),
                    FlowNodeDefinition(name="archive", type="pass-through"),
                ),
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.edges],
            [("start", "done"), ("done", "archive")],
        )
        self.assertNotIn("customer-token", str(result.as_public_dict()))
        self.assertNotIn(str(graph_path), str(result.as_public_dict()))

    def test_compile_flow_graph_reports_malformed_file_source_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            base_path = Path(directory)
            graph_path = base_path / "customer-token.mmd"
            graph_path.write_text(
                "flowchart LR\nstart route@ PrivateCustomerToken",
                encoding="utf-8",
            )
            source = FlowGraphSource(
                source_kind=FlowGraphSourceKind.FILE,
                path=Path("customer-token.mmd"),
                base_path=base_path,
            )

            result = compile_flow_graph(
                source,
                (
                    FlowNodeDefinition(name="start", type="input"),
                    FlowNodeDefinition(name="done", type="pass-through"),
                ),
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.graph.malformed_source"],
        )
        self.assertIsNotNone(result.diagnostics[0].source_span)
        assert result.diagnostics[0].source_span is not None
        self.assertEqual(result.diagnostics[0].source_span.start_line, 2)
        self.assertEqual(result.diagnostics[0].source_span.start_column, 7)
        self.assertNotIn(
            "PrivateCustomerToken", str(result.public_diagnostics)
        )
        self.assertNotIn("customer-token", str(result.public_diagnostics))
        self.assertNotIn(str(graph_path), str(result.public_diagnostics))

    def test_compile_flow_graph_rejects_unsafe_file_source_paths_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base"
            outside_path = root / "outside"
            base_path.mkdir()
            outside_path.mkdir()
            outside_graph = outside_path / "private-token.mmd"
            outside_graph.write_text("flowchart LR", encoding="utf-8")
            (base_path / "directory-token").mkdir()
            (base_path / "linked-token.mmd").symlink_to(outside_graph)
            source_span = FlowSourceSpan(
                source="/private/customer/flow.toml",
                start_line=4,
                start_column=8,
            )
            cases = (
                (
                    "url",
                    Path("https://example.invalid/private-token.mmd"),
                    "flow.graph.path_escape",
                ),
                ("absolute", outside_graph, "flow.graph.path_escape"),
                (
                    "traversal",
                    Path("../outside/private-token.mmd"),
                    "flow.graph.path_escape",
                ),
                (
                    "symlink",
                    Path("linked-token.mmd"),
                    "flow.graph.path_escape",
                ),
                (
                    "missing",
                    Path("missing-private-token.mmd"),
                    "flow.graph.read_failure",
                ),
                (
                    "directory",
                    Path("directory-token"),
                    "flow.graph.read_failure",
                ),
            )

            for name, path, code in cases:
                with self.subTest(name=name):
                    source = FlowGraphSource(
                        source_kind=FlowGraphSourceKind.FILE,
                        path=path,
                        base_path=base_path,
                        source_span=source_span,
                    )

                    result = compile_flow_graph(source, ())

                    self.assertFalse(result.ok)
                    self.assertEqual(result.edges, ())
                    self.assertEqual(
                        [diagnostic.code for diagnostic in result.diagnostics],
                        [code],
                    )
                    self.assertEqual(
                        result.diagnostics[0].source_span,
                        source_span,
                    )
                    public_text = str(result.public_diagnostics)
                    self.assertNotIn("private-token", public_text)
                    self.assertNotIn("directory-token", public_text)
                    self.assertNotIn("missing-private-token", public_text)
                    self.assertNotIn("https://example.invalid", public_text)
                    self.assertNotIn(str(outside_path), public_text)

    def test_compile_flow_graph_reports_missing_file_base_safely(
        self,
    ) -> None:
        source_span = FlowSourceSpan(
            source="/private/customer/flow.toml",
            start_line=4,
            start_column=8,
        )
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.FILE,
            path=Path("customer-token.mmd"),
            source_span=source_span,
        )

        result = compile_flow_graph(source, ())

        self.assertFalse(result.ok)
        self.assertEqual(result.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.graph.read_failure"],
        )
        self.assertEqual(result.diagnostics[0].source_span, source_span)
        self.assertNotIn("customer-token", str(result.public_diagnostics))
        self.assertNotIn("/private/customer", str(result.public_diagnostics))

    def test_compile_flow_graph_reports_unresolvable_file_path_safely(
        self,
    ) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.FILE,
            path=Path("customer-token.mmd"),
            base_path=Path("/private/customer/base"),
        )

        with patch.object(
            Path,
            "resolve",
            side_effect=OSError("PrivateCustomerToken"),
        ):
            result = compile_flow_graph(source, ())

        self.assertFalse(result.ok)
        self.assertEqual(result.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.graph.read_failure"],
        )
        self.assertNotIn(
            "PrivateCustomerToken", str(result.public_diagnostics)
        )
        self.assertNotIn("customer-token", str(result.public_diagnostics))
        self.assertNotIn("/private/customer", str(result.public_diagnostics))

    def test_compile_flow_graph_reports_unreadable_file_safely(self) -> None:
        with TemporaryDirectory() as directory:
            base_path = Path(directory)
            graph_path = base_path / "customer-token.mmd"
            graph_path.write_text(
                "flowchart LR\nstart route_1@--> done",
                encoding="utf-8",
            )
            source = FlowGraphSource(
                source_kind=FlowGraphSourceKind.FILE,
                path=Path("customer-token.mmd"),
                base_path=base_path,
            )

            with patch.object(
                Path,
                "read_text",
                side_effect=PermissionError("PrivateCustomerToken"),
            ):
                result = compile_flow_graph(source, ())

        self.assertFalse(result.ok)
        self.assertEqual(result.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.graph.read_failure"],
        )
        self.assertNotIn(
            "PrivateCustomerToken", str(result.public_diagnostics)
        )
        self.assertNotIn("customer-token", str(result.public_diagnostics))
        self.assertNotIn(str(graph_path), str(result.public_diagnostics))

    def test_compile_flow_graph_rejects_invalid_arguments(self) -> None:
        source = FlowGraphSource(
            source_kind=FlowGraphSourceKind.INLINE,
            diagram="flowchart LR\nstart route_1@--> done",
        )

        with self.assertRaises(AssertionError):
            compile_flow_graph(object(), ())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_graph(source, [])  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_graph(source, (object(),))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_graph(
                source,
                (),
                edge_bindings=[],  # type: ignore[arg-type]
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
        self.assertEqual(
            tuple(code.value for code in FlowGraphDiagnosticCode),
            tuple(case[1] for case in _GRAPH_DIAGNOSTIC_CASES),
        )

    def test_graph_diagnostic_factory_returns_stable_safe_diagnostics(
        self,
    ) -> None:
        source_span = FlowSourceSpan(
            source="/private/customer/flow.toml",
            start_line=9,
            start_column=4,
        )
        related_span = FlowSourceSpan(
            source="/private/customer/graph.mmd",
            start_line=11,
            start_column=6,
        )

        for (
            code,
            value,
            message,
            hint,
            load_category,
        ) in _GRAPH_DIAGNOSTIC_CASES:
            with self.subTest(code=code):
                diagnostic = flow_graph_diagnostic(
                    code,
                    "graph.edges.route_1",
                    source_span=source_span,
                    related_spans=(related_span,),
                )

                self.assertEqual(diagnostic.code, value)
                self.assertEqual(
                    diagnostic.category,
                    FlowDiagnosticCategory.GRAPH_COMPILER,
                )
                self.assertEqual(diagnostic.severity, "error")
                self.assertEqual(diagnostic.message, message)
                self.assertEqual(diagnostic.hint, hint)
                self.assertEqual(
                    flow_graph_diagnostic_load_category(diagnostic),
                    load_category,
                )
                public = diagnostic.as_public_dict()
                self.assertEqual(public["code"], value)
                self.assertEqual(public["message"], message)
                self.assertEqual(public["hint"], hint)
                public_text = str(public)
                self.assertNotIn("/private/customer", public_text)
                self.assertNotIn("graph TD", public_text)
                self.assertNotIn("https://example.test", public_text)
                self.assertNotIn("SECRET_TOKEN", public_text)

    def test_graph_diagnostic_factory_accepts_warning_severity(self) -> None:
        diagnostic = flow_graph_diagnostic(
            FlowGraphDiagnosticCode.MISSING_SOURCE,
            "graph",
            severity=FlowDiagnosticSeverity.WARNING,
        )

        self.assertEqual(diagnostic.severity, FlowDiagnosticSeverity.WARNING)
        self.assertEqual(diagnostic.as_public_dict()["severity"], "warning")

    def test_graph_diagnostic_factory_rejects_unsafe_inputs(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        invalid_cases = (
            {"code": "flow.graph.missing_source", "path": "graph"},
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "/private/customer/graph.mmd",
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "https://example.test/graph.mmd",
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph.{secret}",
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph\nsource",
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph",
                "source_span": object(),
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph",
                "related_spans": [span],
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph",
                "related_spans": (object(),),
            },
            {
                "code": FlowGraphDiagnosticCode.MISSING_SOURCE,
                "path": "graph",
                "severity": "error",
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    flow_graph_diagnostic(**case)  # type: ignore[arg-type]

    def test_graph_diagnostic_load_category_rejects_non_graph_diagnostics(
        self,
    ) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.definition.invalid_node",
            category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
            path="nodes.start",
            message="Flow node is invalid.",
        )

        with self.assertRaises(AssertionError):
            flow_graph_diagnostic_load_category(diagnostic)

        with self.assertRaises(AssertionError):
            flow_graph_diagnostic_load_category(  # type: ignore[arg-type]
                "diagnostic"
            )

    def test_graph_diagnostic_load_category_rejects_unknown_graph_code(
        self,
    ) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.graph.future_code",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            path="graph",
            message="Graph diagnostic is unknown.",
        )

        with self.assertRaises(ValueError):
            flow_graph_diagnostic_load_category(diagnostic)


if __name__ == "__main__":
    main()
