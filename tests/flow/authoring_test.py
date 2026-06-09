from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import avalan.flow as flow
from avalan.flow import (
    FLOW_VIEW_SKELETON_TAG,
    FlowDefinition,
    FlowDefinitionCompileResult,
    FlowDefinitionSkeletonResult,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowGraphInspectionResult,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowViewImportMode,
    MermaidFlowViewNormalizationResult,
    MermaidRenderResult,
    bind_flow_view,
    compare_flow_topology,
    compile_flow_file,
    compile_flow_source,
    inspect_flow_graph_file,
    inspect_flow_graph_source,
    parse_mermaid_view,
    render_flow_view,
    serialize_flow_definition,
    skeleton_from_mermaid_view,
    validate_flow_definition,
)


class FlowAuthoringTestCase(TestCase):
    def test_parse_mermaid_view_and_render_flow_view_are_stable_apis(
        self,
    ) -> None:
        parsed = parse_mermaid_view("graph TD\nA[Start] --> B[Done]")

        rendered = render_flow_view(parsed.view)

        self.assertIsInstance(parsed, MermaidFlowViewNormalizationResult)
        self.assertTrue(parsed.ok, parsed.public_diagnostics)
        self.assertEqual(parsed.public_diagnostics, ())
        self.assertEqual(
            [(node.id, node.label) for node in parsed.view.nodes],
            [("A", "Start"), ("B", "Done")],
        )
        self.assertIsInstance(rendered, MermaidRenderResult)
        self.assertTrue(rendered.ok, rendered.public_diagnostics)
        self.assertIn("flowchart TD", rendered.source)
        self.assertIn('A["Start"]', rendered.source)

    def test_parse_mermaid_view_reports_executable_import_diagnostics(
        self,
    ) -> None:
        parsed = parse_mermaid_view(
            "graph TD\nA & B --> C",
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/topology.mmd",
        )

        self.assertFalse(parsed.ok)
        self.assertEqual(
            parsed.diagnostics[0].code,
            "flow.mermaid.security.ambiguous_shorthand",
        )
        self.assertNotIn("customer", str(parsed.public_diagnostics))

    def test_bind_and_compare_flow_topology_share_diagnostic_result(
        self,
    ) -> None:
        parsed = parse_mermaid_view("graph TD\nA --> B")
        definition = FlowDefinition(
            name="mismatch",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="C", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="C"),),
        )

        binding = bind_flow_view(parsed.view, definition)
        comparison = compare_flow_topology(parsed.view, definition)

        self.assertFalse(binding.ok)
        self.assertFalse(comparison.ok)
        self.assertEqual(binding.diagnostics, comparison.diagnostics)
        self.assertEqual(
            [diagnostic.code for diagnostic in comparison.diagnostics],
            [
                "flow.view.binding.missing_node",
                "flow.view.binding.extra_node",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_edge",
                "flow.view.binding.extra_edge",
                "flow.view.binding.missing_edge_semantics",
            ],
        )

    def test_skeleton_from_mermaid_view_returns_diagnostic_result(
        self,
    ) -> None:
        parsed = parse_mermaid_view("graph TD\nA --> B")

        result = skeleton_from_mermaid_view(
            parsed.view,
            name="topology",
            version="2026-06-08",
        )

        self.assertIsInstance(result, FlowDefinitionSkeletonResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.public_diagnostics, ())
        self.assertEqual(result.definition.name, "topology")
        self.assertEqual(result.definition.version, "2026-06-08")
        self.assertIn(FLOW_VIEW_SKELETON_TAG, result.definition.tags)
        self.assertEqual(
            [node.name for node in result.definition.nodes],
            ["A", "B"],
        )
        with self.assertRaises(FrozenInstanceError):
            result.diagnostics = ()  # type: ignore[misc]

    def test_skeleton_from_mermaid_view_preserves_view_diagnostics(
        self,
    ) -> None:
        parsed = parse_mermaid_view(
            "graph TD\nA & B --> C",
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/topology.mmd",
        )

        result = skeleton_from_mermaid_view(parsed.view, name="topology")

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics, parsed.view.diagnostics)
        self.assertNotIn("customer", str(result.public_diagnostics))

    def test_validate_flow_definition_remains_public_sdk_import(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="valid",
                version="2026-06-08",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"result": "start.result"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="input"),),
            )
        )

        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertEqual(result.public_diagnostics, ())

    def test_compile_flow_source_returns_canonical_strict_output(self) -> None:
        result = compile_flow_source(
            """
            [flow]
            name = "graph_sdk"
            entrypoint = "start"
            output_node = "finish"

            [graph]
            format = "mermaid"
            source = "inline"
            mode = "executable"
            diagram = '''
            flowchart LR
            start route_1@-->|Private customer route| finish
            start -.-> note["Private customer note"]
            '''

            [graph.edges.route_1]
            label = "approved"

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "echo"
            """,
            source_path="/private/customer/flow.toml",
        )

        self.assertIsInstance(result, FlowDefinitionCompileResult)
        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertTrue(result.authoring_graph)
        assert result.definition is not None
        assert result.canonical_source is not None
        self.assertEqual(result.definition.name, "graph_sdk")
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in result.definition.edges
            ],
            [("start", "finish", "approved")],
        )
        self.assertEqual(
            result.canonical_source,
            serialize_flow_definition(result.definition),
        )
        self.assertIn("[[edges]]", result.canonical_source)
        self.assertNotIn("[graph]", result.canonical_source)
        self.assertNotIn("flowchart", result.canonical_source)
        self.assertNotIn("Private customer", result.canonical_source)
        self.assertIsNotNone(result.graph_inspection)
        public = result.as_public_dict()
        self.assertEqual(
            public["definition"],
            {"name": "graph_sdk", "node_count": 2, "edge_count": 1},
        )
        self.assertEqual(
            public["canonical_source"],
            {"format": "toml", "strict": True},
        )
        self.assertNotIn("Private customer", str(public))
        self.assertNotIn("/private/customer", str(public))

    def test_compile_and_inspect_flow_file_share_graph_projection(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir = root / "graphs"
            graph_dir.mkdir()
            (graph_dir / "private-route.mmd").write_text(
                "\n".join(
                    (
                        "flowchart LR",
                        "start route_1@-->|Private file route| finish",
                    )
                ),
                encoding="utf-8",
            )
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "graph_file_sdk"
                entrypoint = "start"
                output_node = "finish"

                [graph]
                format = "mermaid"
                source = "file"
                mode = "executable"
                path = "graphs/private-route.mmd"

                [nodes.start]
                type = "input"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )

            compiled = compile_flow_file(flow_path)
            inspected = inspect_flow_graph_file(flow_path)

        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        self.assertTrue(inspected.ok, inspected.public_diagnostics)
        assert compiled.graph_inspection is not None
        assert inspected.inspection is not None
        self.assertEqual(
            compiled.graph_inspection.as_public_dict(),
            inspected.inspection.as_public_dict(),
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in compiled.definition.edges],
            [("start", "finish")],
        )
        public = str(inspected.as_public_dict())
        self.assertNotIn("Private file route", public)
        self.assertNotIn("private-route.mmd", public)
        self.assertNotIn(str(flow_path), public)

    def test_compile_flow_source_reports_malformed_toml_safely(self) -> None:
        result = compile_flow_source(
            '[flow]\nname = "private-token"\n[',
            source_path="/private/customer/flow.toml",
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.canonical_source)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.malformed_toml"],
        )
        public = str(result.as_public_dict())
        self.assertNotIn("private-token", public)
        self.assertNotIn("/private/customer", public)

    def test_inspect_flow_graph_source_reports_missing_graph(self) -> None:
        result = inspect_flow_graph_source("""
            [flow]
            name = "strict_sdk"
            entrypoint = "start"
            output_node = "finish"

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"
            """)

        self.assertIsInstance(result, FlowGraphInspectionResult)
        self.assertFalse(result.ok)
        self.assertFalse(result.authoring_graph)
        self.assertIsNone(result.inspection)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.graph.missing_source"],
        )

    def test_inspect_flow_graph_source_reports_graph_failure_safely(
        self,
    ) -> None:
        result = inspect_flow_graph_source(
            """
            [flow]
            name = "invalid_graph_sdk"
            entrypoint = "start"
            output_node = "finish"

            [graph]
            format = "mermaid"
            source = "inline"
            mode = "executable"
            diagram = '''
            flowchart LR
            start -->|Private customer route| finish
            '''

            [nodes.start]
            type = "input"

            [nodes.finish]
            type = "echo"
            """,
            source_path="/private/customer/flow.toml",
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.authoring_graph)
        self.assertIsNotNone(result.inspection)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.graph.unsupported_executable_edge"],
        )
        public = str(result.as_public_dict())
        self.assertNotIn("Private customer route", public)
        self.assertNotIn("/private/customer", public)

    def test_compile_and_inspect_sdk_results_reject_invalid_values(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            FlowDefinitionCompileResult(
                definition=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowDefinitionCompileResult(canonical_source=" ")
        with self.assertRaises(AssertionError):
            FlowDefinitionCompileResult(
                diagnostics=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowDefinitionCompileResult(
                authoring_graph="yes",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowDefinitionCompileResult(
                graph_inspection=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowGraphInspectionResult(
                inspection=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowGraphInspectionResult(
                diagnostics=(object(),),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowGraphInspectionResult(
                authoring_graph="yes",  # type: ignore[arg-type]
            )

    def test_compile_and_inspect_sdk_functions_reject_invalid_arguments(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            compile_flow_source(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_source("", source_path=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_source("", registry=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_file(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            compile_flow_file(
                Path("missing.toml"),
                registry=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            inspect_flow_graph_source(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            inspect_flow_graph_source(
                "",
                source_path=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            inspect_flow_graph_source("", registry=object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            inspect_flow_graph_file(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            inspect_flow_graph_file(
                Path("missing.toml"),
                registry=object(),  # type: ignore[arg-type]
            )

    def test_private_parser_internals_are_not_exported_from_package(
        self,
    ) -> None:
        self.assertFalse(hasattr(flow, "_MermaidTokenizer"))

    def test_mermaid_authoring_apis_do_not_call_flow_parse_helper(
        self,
    ) -> None:
        calls: list[str] = []
        original = flow.Flow.parse_mermaid

        def blocked(_: flow.Flow, source_text: str) -> None:
            calls.append(source_text)
            raise AssertionError("unexpected helper call")

        with patch.object(flow.Flow, "parse_mermaid", blocked):
            parsed = parse_mermaid_view("graph TD\nA --> B")
            direct = flow.parse_mermaid("graph TD\nA --> B")

        self.assertIs(flow.Flow.parse_mermaid, original)
        self.assertTrue(parsed.ok, parsed.public_diagnostics)
        self.assertTrue(direct.ok, direct.public_diagnostics)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    main()
