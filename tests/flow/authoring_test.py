from ast import AST, Attribute, Call, ImportFrom, Name, parse, walk
from asyncio import run as asyncio_run
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
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowGraphDiagnosticCode,
    FlowGraphInspection,
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

_FLOW_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "avalan" / "flow"
)
_FORBIDDEN_ASYNCIO_IMPORTS = frozenset({"run", "to_thread"})
_FORBIDDEN_ASYNCIO_METHODS = frozenset({"run", "to_thread"})
_FORBIDDEN_LOOP_METHODS = frozenset({"run_until_complete"})
_FORBIDDEN_SYNC_FILE_METHODS = frozenset(
    {
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
    }
)


def _flow_async_boundary_violations(
    path: Path,
    source: str,
) -> tuple[str, ...]:
    tree = parse(source, filename=str(path))
    violations: list[str] = []
    for node in walk(tree):
        if isinstance(node, ImportFrom):
            violations.extend(_asyncio_import_violations(path, node))
            continue
        if isinstance(node, Call):
            violations.extend(_call_boundary_violations(path, node))
    return tuple(sorted(violations))


def _asyncio_import_violations(
    path: Path,
    node: ImportFrom,
) -> tuple[str, ...]:
    if node.module != "asyncio":
        return ()
    return tuple(
        f"{_node_location(path, node)} imports asyncio.{alias.name}"
        for alias in node.names
        if alias.name in _FORBIDDEN_ASYNCIO_IMPORTS
    )


def _call_boundary_violations(path: Path, node: Call) -> tuple[str, ...]:
    func = node.func
    if not isinstance(func, Attribute):
        return ()
    violations: list[str] = []
    if func.attr in _FORBIDDEN_SYNC_FILE_METHODS:
        violations.append(f"{_node_location(path, node)} calls {func.attr}")
    if func.attr in _FORBIDDEN_LOOP_METHODS:
        violations.append(f"{_node_location(path, node)} calls {func.attr}")
    if (
        isinstance(func.value, Name)
        and func.value.id == "asyncio"
        and func.attr in _FORBIDDEN_ASYNCIO_METHODS
    ):
        violations.append(
            f"{_node_location(path, node)} calls asyncio.{func.attr}"
        )
    return tuple(violations)


def _node_location(path: Path, node: AST) -> str:
    return f"{path}:{getattr(node, 'lineno', 0)}"


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
        result = asyncio_run(
            compile_flow_source(
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
        )

        self.assertIsInstance(result, FlowDefinitionCompileResult)
        self.assertTrue(result.ok, result.public_diagnostics)
        self.assertTrue(result.authoring_graph)
        assert result.definition is not None
        assert result.canonical_source is not None
        self.assertEqual(result.definition.name, "graph_sdk")
        self.assertIsNone(result.definition.definition_base)
        self.assertNotIn("/private/customer", str(result.definition))
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
        self.assertEqual(public["diagnostics"], ())
        self.assertNotIn("Private customer", str(public))
        self.assertNotIn("/private/customer", str(public))

    def test_compile_flow_source_revalidates_canonical_output_safely(
        self,
    ) -> None:
        with patch(
            "avalan.flow.authoring.serialize_flow_definition",
            return_value="[flow\nprivate = 'customer-token'",
        ):
            result = asyncio_run(
                compile_flow_source(
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
                    '''

                    [nodes.start]
                    type = "input"

                    [nodes.finish]
                    type = "echo"
                    """,
                    source_path="/private/customer/flow.toml",
                )
            )

        self.assertFalse(result.ok)
        self.assertTrue(result.authoring_graph)
        self.assertIsNone(result.definition)
        self.assertIsNone(result.canonical_source)
        self.assertIsNotNone(result.graph_inspection)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            ["flow.malformed_toml"],
        )
        public = str(result.as_public_dict())
        self.assertNotIn("Private customer route", public)
        self.assertNotIn("customer-token", public)
        self.assertNotIn("/private/customer", public)

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

            compiled = asyncio_run(compile_flow_file(flow_path))
            inspected = asyncio_run(inspect_flow_graph_file(flow_path))

        self.assertTrue(compiled.ok, compiled.public_diagnostics)
        self.assertTrue(inspected.ok, inspected.public_diagnostics)
        self.assertEqual(compiled.as_public_dict()["diagnostics"], ())
        self.assertEqual(inspected.as_public_dict()["diagnostics"], ())
        assert compiled.definition is not None
        self.assertIsNone(compiled.definition.definition_base)
        self.assertNotIn(str(root), str(compiled.definition))
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

    def test_compile_flow_file_uses_configurable_graph_encoding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_dir = root / "graphs"
            graph_dir.mkdir()
            (graph_dir / "latin.mmd").write_bytes(
                (
                    "flowchart LR\nstart route_1@-->|Café route| finish\n"
                ).encode("latin-1")
            )
            flow_path = root / "flow.toml"
            flow_path.write_text(
                """
                [flow]
                name = "graph_file_encoding"
                entrypoint = "start"
                output_node = "finish"

                [graph]
                format = "mermaid"
                source = "file"
                mode = "executable"
                path = "graphs/latin.mmd"

                [nodes.start]
                type = "input"

                [nodes.finish]
                type = "echo"
                """,
                encoding="utf-8",
            )

            default_result = asyncio_run(compile_flow_file(flow_path))
            latin_result = asyncio_run(
                compile_flow_file(flow_path, encoding="latin-1")
            )

        self.assertFalse(default_result.ok)
        self.assertEqual(
            [
                diagnostic["code"]
                for diagnostic in default_result.public_diagnostics
            ],
            ["flow.graph.read_failure"],
        )
        self.assertTrue(latin_result.ok, latin_result.public_diagnostics)

    def test_compile_and_inspect_flow_file_report_read_failures_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_path = root / "private-missing.flow.toml"
            invalid_path = root / "private-decode.flow.toml"
            invalid_path.write_bytes(b"[flow]\nname = '\xff'\n")

            missing_compile = asyncio_run(compile_flow_file(missing_path))
            missing_inspect = asyncio_run(
                inspect_flow_graph_file(missing_path)
            )
            decode_compile = asyncio_run(compile_flow_file(invalid_path))
            decode_inspect = asyncio_run(inspect_flow_graph_file(invalid_path))

        for result in (
            missing_compile,
            missing_inspect,
            decode_compile,
            decode_inspect,
        ):
            with self.subTest(result=result):
                self.assertFalse(result.ok)
                self.assertEqual(
                    [
                        diagnostic["code"]
                        for diagnostic in result.public_diagnostics
                    ],
                    ["file.read"],
                )
                public = str(result.as_public_dict())
                self.assertNotIn("private-missing.flow.toml", public)
                self.assertNotIn("private-decode.flow.toml", public)
                self.assertNotIn(str(root), public)

    def test_compile_flow_source_reports_malformed_toml_safely(self) -> None:
        result = asyncio_run(
            compile_flow_source(
                '[flow]\nname = "private-token"\n[',
                source_path="/private/customer/flow.toml",
            )
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
        result = asyncio_run(inspect_flow_graph_source("""
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
                """))

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
        result = asyncio_run(
            inspect_flow_graph_source(
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

    def test_graph_inspection_result_reports_nested_errors(self) -> None:
        diagnostic = FlowDiagnostic(
            code=FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE.value,
            path="graph.edges",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            severity=FlowDiagnosticSeverity.ERROR,
            message="Graph edge is not supported for execution.",
            hint="Use explicit directed graph edges.",
        )
        result = FlowGraphInspectionResult(
            inspection=FlowGraphInspection(diagnostics=(diagnostic,)),
            authoring_graph=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [item["code"] for item in result.public_diagnostics],
            [FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE.value],
        )
        public = result.as_public_dict()
        self.assertEqual(public["diagnostics"], result.public_diagnostics)
        self.assertEqual(public["authoring_graph"], True)
        self.assertIn("inspection", public)

    def test_graph_inspection_result_merges_distinct_errors(self) -> None:
        load_diagnostic = FlowDiagnostic(
            code="flow.definition.invalid",
            path="flow.name",
            category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
            severity=FlowDiagnosticSeverity.ERROR,
            message="Flow definition is invalid.",
            hint="Fix the flow definition.",
        )
        graph_diagnostic = FlowDiagnostic(
            code=FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE.value,
            path="graph.edges",
            category=FlowDiagnosticCategory.GRAPH_COMPILER,
            severity=FlowDiagnosticSeverity.ERROR,
            message="Graph edge is not supported for execution.",
            hint="Use explicit directed graph edges.",
        )
        result = FlowGraphInspectionResult(
            inspection=FlowGraphInspection(diagnostics=(graph_diagnostic,)),
            diagnostics=(load_diagnostic, graph_diagnostic),
            authoring_graph=True,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [item["code"] for item in result.public_diagnostics],
            [
                "flow.definition.invalid",
                FlowGraphDiagnosticCode.UNSUPPORTED_EXECUTABLE_EDGE.value,
            ],
        )
        public = result.as_public_dict()
        self.assertEqual(public["diagnostics"], result.public_diagnostics)
        self.assertIn("inspection", public)

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
            asyncio_run(compile_flow_source(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(
                compile_flow_source("", source_path=object())  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            asyncio_run(
                compile_flow_source("", registry=object())  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            asyncio_run(compile_flow_file(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(
                compile_flow_file(
                    Path("missing.toml"),
                    registry=object(),  # type: ignore[arg-type]
                )
            )
        with self.assertRaises(AssertionError):
            asyncio_run(compile_flow_file(Path("missing.toml"), encoding=""))
        with self.assertRaises(AssertionError):
            asyncio_run(inspect_flow_graph_source(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(
                inspect_flow_graph_source(
                    "",
                    source_path=object(),  # type: ignore[arg-type]
                )
            )
        with self.assertRaises(AssertionError):
            asyncio_run(
                inspect_flow_graph_source("", registry=object())  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            asyncio_run(inspect_flow_graph_file(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(
                inspect_flow_graph_file(
                    Path("missing.toml"),
                    registry=object(),  # type: ignore[arg-type]
                )
            )
        with self.assertRaises(AssertionError):
            asyncio_run(
                inspect_flow_graph_file(Path("missing.toml"), encoding="")
            )

    def test_flow_sdk_internals_keep_async_file_boundaries(self) -> None:
        violations = tuple(
            violation
            for path in sorted(_FLOW_SOURCE_ROOT.rglob("*.py"))
            for violation in _flow_async_boundary_violations(
                path.relative_to(_FLOW_SOURCE_ROOT),
                path.read_text(encoding="utf-8"),
            )
        )

        self.assertEqual(violations, ())

    def test_flow_sdk_boundary_audit_rejects_sync_calls(self) -> None:
        violations = _flow_async_boundary_violations(
            Path("flow.py"),
            """
from asyncio import gather, run, to_thread, wait_for

async def bad(path, loop, task):
    path.read_text()
    path.write_bytes(b"private")
    asyncio.run(task())
    loop.run_until_complete(task())
""",
        )

        self.assertEqual(
            violations,
            (
                "flow.py:2 imports asyncio.run",
                "flow.py:2 imports asyncio.to_thread",
                "flow.py:5 calls read_text",
                "flow.py:6 calls write_bytes",
                "flow.py:7 calls asyncio.run",
                "flow.py:8 calls run_until_complete",
            ),
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
