from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FLOW_VIEW_SKELETON_NODE_TYPE,
    FLOW_VIEW_SKELETON_TAG,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowEntryBehavior,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowView,
    FlowViewBindingResult,
    FlowViewEdge,
    FlowViewImportMode,
    FlowViewNode,
    FlowViewNodeShape,
    bind_flow_view_definition,
    create_flow_definition_skeleton,
    loads_flow_definition_result,
    normalize_mermaid_flow_view,
    render_flow_definition_mermaid,
    render_mermaid_view,
    validate_flow_definition,
)


class FlowViewBindingTestCase(TestCase):
    def test_bind_flow_view_definition_accepts_matching_topology(
        self,
    ) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.EXECUTABLE,
            nodes=(
                FlowViewNode(id="start"),
                FlowViewNode(
                    id="review",
                    shape=FlowViewNodeShape.DIAMOND,
                ),
                FlowViewNode(id="approve"),
                FlowViewNode(id="reject"),
            ),
            edges=(
                FlowViewEdge(
                    id="start_to_review", source="start", target="review"
                ),
                FlowViewEdge(
                    id="review_to_approve",
                    source="review",
                    target="approve",
                    label="Yes",
                ),
                FlowViewEdge(
                    id="review_to_reject",
                    source="review",
                    target="reject",
                    label="No",
                ),
            ),
        )
        definition = FlowDefinition(
            name="review",
            version="2026-06-07",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="result", type=FlowOutputType.OBJECT
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"result": "approve.result"},
            ),
            nodes=(
                FlowNodeDefinition(name="start", type="input"),
                FlowNodeDefinition(name="review", type="pass-through"),
                FlowNodeDefinition(name="approve", type="pass-through"),
                FlowNodeDefinition(name="reject", type="pass-through"),
            ),
            edges=(
                FlowEdgeDefinition(source="start", target="review"),
                FlowEdgeDefinition(source="review", target="approve"),
                FlowEdgeDefinition(source="review", target="reject"),
            ),
        )

        result = bind_flow_view_definition(view, definition)

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(
            definition.node_map["review"].type,
            "pass-through",
        )
        self.assertTrue(
            all(edge.condition is None for edge in definition.edges)
        )

    def test_bind_flow_view_definition_reports_topology_gaps(self) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA --> B\nB --> C",
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/topology.mmd",
        )
        definition = FlowDefinition(
            name="mismatch",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="D", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="D"),),
        )

        binding = bind_flow_view_definition(result.view, definition)

        self.assertFalse(binding.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in binding.diagnostics],
            [
                "flow.view.binding.missing_node",
                "flow.view.binding.extra_node",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.extra_node",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_edge",
                "flow.view.binding.extra_edge",
                "flow.view.binding.missing_edge_semantics",
                "flow.view.binding.extra_edge",
                "flow.view.binding.missing_edge_semantics",
            ],
        )
        self.assertEqual(
            binding.diagnostics[0].category,
            FlowDiagnosticCategory.FLOW_VIEW_BINDING,
        )
        self.assertNotIn("customer", str(binding.public_diagnostics))

    def test_bind_flow_view_definition_counts_duplicate_edges(self) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.PRESENTATION,
            nodes=(FlowViewNode(id="A"), FlowViewNode(id="B")),
            edges=(
                FlowViewEdge(id="first", source="A", target="B"),
                FlowViewEdge(id="second", source="A", target="B"),
            ),
        )
        definition = FlowDefinition(
            name="duplicate_edges",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="B"),),
        )

        binding = bind_flow_view_definition(view, definition)

        self.assertFalse(binding.ok)
        self.assertEqual(
            [
                (diagnostic.code, diagnostic.path)
                for diagnostic in binding.diagnostics
            ],
            [
                ("flow.view.binding.extra_edge", "view.edges.second"),
                (
                    "flow.view.binding.missing_edge_semantics",
                    "view.edges.second",
                ),
            ],
        )

    def test_bind_flow_view_definition_sanitizes_duplicate_edge_spans(
        self,
    ) -> None:
        normalized = normalize_mermaid_flow_view(
            "graph TD\nA --> B\nA --> B",
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/topology.mmd",
        )
        definition = FlowDefinition(
            name="duplicate_edges",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="B"),),
        )

        binding = bind_flow_view_definition(normalized.view, definition)

        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertFalse(binding.ok)
        self.assertEqual(
            [
                (
                    diagnostic.code,
                    diagnostic.path,
                    (
                        diagnostic.source_span.as_dict()
                        if diagnostic.source_span is not None
                        else None
                    ),
                )
                for diagnostic in binding.diagnostics
            ],
            [
                (
                    "flow.view.binding.extra_edge",
                    "view.edges.A_to_B_2",
                    {
                        "start_line": 3,
                        "start_column": 3,
                        "end_line": 3,
                        "end_column": 8,
                        "source": "/private/customer/topology.mmd",
                    },
                ),
                (
                    "flow.view.binding.missing_edge_semantics",
                    "view.edges.A_to_B_2",
                    {
                        "start_line": 3,
                        "start_column": 3,
                        "end_line": 3,
                        "end_column": 8,
                        "source": "/private/customer/topology.mmd",
                    },
                ),
            ],
        )
        public_diagnostics = str(binding.public_diagnostics)
        self.assertNotIn("customer", public_diagnostics)
        self.assertNotIn("topology.mmd", public_diagnostics)
        self.assertNotIn("/private", public_diagnostics)

    def test_bind_flow_view_definition_reports_duplicate_definition_edges(
        self,
    ) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.PRESENTATION,
            nodes=(FlowViewNode(id="A"), FlowViewNode(id="B")),
            edges=(FlowViewEdge(id="first", source="A", target="B"),),
        )
        definition = FlowDefinition(
            name="duplicate_definition_edges",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(
                FlowEdgeDefinition(source="A", target="B"),
                FlowEdgeDefinition(source="A", target="B"),
            ),
        )

        binding = bind_flow_view_definition(view, definition)

        self.assertFalse(binding.ok)
        self.assertEqual(
            [
                (diagnostic.code, diagnostic.path)
                for diagnostic in binding.diagnostics
            ],
            [
                (
                    "flow.view.binding.missing_edge",
                    "definition.edges.A->B",
                ),
            ],
        )

    def test_bind_flow_view_definition_preserves_view_errors(self) -> None:
        normalized = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "graph TD",
                    "A --> B",
                    'click A href "https://example.test/private"',
                )
            ),
            import_mode=FlowViewImportMode.EXECUTABLE,
        )
        definition = FlowDefinition(
            name="unsafe_view",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="B"),),
        )

        binding = bind_flow_view_definition(normalized.view, definition)

        self.assertFalse(normalized.ok)
        self.assertFalse(binding.ok)
        self.assertIn(
            "flow.mermaid.security.unsafe_link_directive",
            [diagnostic.code for diagnostic in binding.diagnostics],
        )
        self.assertNotIn("example.test", str(binding.public_diagnostics))

    def test_bind_flow_view_definition_preserves_view_warnings(
        self,
    ) -> None:
        normalized = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "graph TD",
                    "A --> B",
                    "link A https://example.test/private",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/topology.mmd",
        )
        definition = FlowDefinition(
            name="warned_view",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="B"),),
        )

        binding = bind_flow_view_definition(normalized.view, definition)

        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertTrue(binding.ok, binding.public_diagnostics)
        self.assertEqual(
            [
                (diagnostic.code, diagnostic.severity)
                for diagnostic in binding.diagnostics
            ],
            [
                (
                    "flow.mermaid.security.unsafe_link_directive",
                    FlowDiagnosticSeverity.WARNING,
                ),
                (
                    "flow.mermaid.security.unsafe_external_link",
                    FlowDiagnosticSeverity.WARNING,
                ),
            ],
        )
        self.assertNotIn("example.test", str(binding.public_diagnostics))
        self.assertNotIn("customer", str(binding.public_diagnostics))

    def test_rendered_definition_round_trip_binds_to_semantics(self) -> None:
        definition = FlowDefinition(
            name="rendered",
            version="2026-06-07",
            inputs=(
                FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="result", type=FlowOutputType.OBJECT
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="start"),
            output_behavior=FlowOutputBehavior(
                outputs={"result": "finish.result"},
            ),
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="input",
                    config={"prompt": "private prompt"},
                ),
                FlowNodeDefinition(name="finish", type="pass-through"),
            ),
            edges=(
                FlowEdgeDefinition(
                    source="start",
                    target="finish",
                    label="done",
                ),
            ),
        )

        rendered = render_flow_definition_mermaid(definition)
        normalized = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )
        binding = bind_flow_view_definition(normalized.view, definition)

        self.assertTrue(rendered.ok, rendered.public_diagnostics)
        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertTrue(binding.ok, binding.public_diagnostics)
        self.assertNotIn("input", rendered.source)
        self.assertNotIn("pass-through", rendered.source)
        self.assertNotIn("private prompt", rendered.source)
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in normalized.view.edges
            ],
            [("start", "finish", "done")],
        )

    def test_loaded_definition_round_trip_binds_end_to_end(self) -> None:
        loaded = loads_flow_definition_result("""
            [flow]
            name = "loaded_binding"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.value"

            [nodes.start]
            type = "input"

            [nodes.start.config]
            secret = "private prompt"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"
            label = "done"
            """)

        self.assertTrue(loaded.ok, loaded.issues)
        assert loaded.definition is not None
        validation = validate_flow_definition(loaded.definition)
        rendered = render_flow_definition_mermaid(loaded.definition)
        normalized = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )
        binding = bind_flow_view_definition(
            normalized.view,
            loaded.definition,
        )

        self.assertTrue(validation.ok, validation.public_diagnostics)
        self.assertTrue(rendered.ok, rendered.public_diagnostics)
        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertTrue(binding.ok, binding.public_diagnostics)
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in binding.view.edges
            ],
            [("start", "finish", "done")],
        )
        self.assertNotIn("private prompt", rendered.source)
        self.assertNotIn("private prompt", str(binding.public_diagnostics))

    def test_loaded_definition_binding_reports_drift_without_private_values(
        self,
    ) -> None:
        loaded = loads_flow_definition_result("""
            [flow]
            name = "loaded_drift"
            version = "2026-06-07"

            [[inputs]]
            name = "payload"
            type = "object"

            [[outputs]]
            name = "answer"
            type = "object"

            [entry]
            type = "node"
            node = "start"

            [output_behavior]
            type = "map"

            [output_behavior.outputs]
            answer = "finish.value"

            [nodes.start]
            type = "input"

            [nodes.start.config]
            token = "private-token"

            [nodes.finish]
            type = "echo"

            [[edges]]
            source = "start"
            target = "finish"
            """)

        self.assertTrue(loaded.ok, loaded.issues)
        assert loaded.definition is not None
        normalized = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "flowchart TD",
                    "start --> finish",
                    "finish --> audit",
                )
            ),
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/topology.mmd",
        )
        binding = bind_flow_view_definition(
            normalized.view,
            loaded.definition,
        )

        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertFalse(binding.ok)
        self.assertEqual(
            [
                (diagnostic.code, diagnostic.path)
                for diagnostic in binding.diagnostics
            ],
            [
                ("flow.view.binding.extra_node", "view.nodes.audit"),
                (
                    "flow.view.binding.missing_node_semantics",
                    "view.nodes.audit",
                ),
                ("flow.view.binding.extra_edge", "view.edges.finish_to_audit"),
                (
                    "flow.view.binding.missing_edge_semantics",
                    "view.edges.finish_to_audit",
                ),
            ],
        )
        public_diagnostics = str(binding.public_diagnostics)
        self.assertNotIn("private-token", public_diagnostics)
        self.assertNotIn("customer", public_diagnostics)

    def test_bidirectional_visual_edge_is_not_reverse_semantics(
        self,
    ) -> None:
        normalized = normalize_mermaid_flow_view(
            "graph TD\nA <==> B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )
        definition = FlowDefinition(
            name="two_way",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="B", type="pass-through"),
            ),
            edges=(
                FlowEdgeDefinition(source="A", target="B"),
                FlowEdgeDefinition(source="B", target="A"),
            ),
        )

        binding = bind_flow_view_definition(normalized.view, definition)

        self.assertTrue(normalized.ok, normalized.public_diagnostics)
        self.assertTrue(normalized.view.edges[0].bidirectional)
        self.assertFalse(binding.ok)
        self.assertEqual(
            [
                (diagnostic.code, diagnostic.path)
                for diagnostic in binding.diagnostics
            ],
            [
                (
                    "flow.view.binding.missing_edge",
                    "definition.edges.B->A",
                ),
            ],
        )

    def test_create_flow_definition_skeleton_is_not_executable(self) -> None:
        view = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "flowchart LR",
                    "subgraph lane[Lane]",
                    "A([Start]) -->|Yes| B{Review}",
                    "end",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
        ).view

        skeleton = create_flow_definition_skeleton(
            view,
            name="review_skeleton",
            version="2026-06-07",
            revision="rev-1",
        )
        binding = bind_flow_view_definition(view, skeleton)
        validation = validate_flow_definition(skeleton)

        self.assertEqual(skeleton.tags, (FLOW_VIEW_SKELETON_TAG,))
        self.assertEqual(skeleton.variables["executable"], False)
        self.assertEqual(skeleton.version, "2026-06-07")
        self.assertEqual(skeleton.revision, "rev-1")
        self.assertEqual(skeleton.nodes[0].type, FLOW_VIEW_SKELETON_NODE_TYPE)
        self.assertEqual(skeleton.edges[0].label, "Yes")
        self.assertEqual(
            cast(dict[str, object], skeleton.nodes[0].config)["shape"],
            "stadium",
        )
        self.assertEqual(
            cast(dict[str, object], skeleton.nodes[0].config)["group"],
            "lane",
        )
        self.assertEqual(
            cast(dict[str, object], skeleton.nodes[0].config)["executable"],
            False,
        )
        self.assertFalse(validation.ok)
        self.assertFalse(binding.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in binding.diagnostics],
            [
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_edge_semantics",
            ],
        )

    def test_binding_result_is_frozen_and_validated(self) -> None:
        view = FlowView(import_mode=FlowViewImportMode.PRESENTATION)
        definition = FlowDefinition(name="flow", nodes=())
        diagnostic = FlowDiagnostic(
            code="flow.view.binding.missing_node",
            category=FlowDiagnosticCategory.FLOW_VIEW_BINDING,
            path="definition.nodes.start",
            message="Structured flow node is missing from the view.",
        )
        result = FlowViewBindingResult(
            view=view,
            definition=definition,
            diagnostics=(diagnostic,),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.view.binding.missing_node",
        )
        with self.assertRaises(FrozenInstanceError):
            result.view = view  # type: ignore[misc]

        invalid_cases = (
            {"view": object(), "definition": definition},
            {"view": view, "definition": object()},
            {
                "view": view,
                "definition": definition,
                "diagnostics": [diagnostic],
            },
            {
                "view": view,
                "definition": definition,
                "diagnostics": (object(),),
            },
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewBindingResult(**case)  # type: ignore[arg-type]

    def test_binding_functions_reject_invalid_arguments(self) -> None:
        view = FlowView(import_mode=FlowViewImportMode.PRESENTATION)
        definition = FlowDefinition(name="flow", nodes=())

        with self.assertRaises(AssertionError):
            bind_flow_view_definition(object(), definition)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            bind_flow_view_definition(view, object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            create_flow_definition_skeleton(  # type: ignore[arg-type]
                object(),
                name="flow",
            )
        with self.assertRaises(AssertionError):
            create_flow_definition_skeleton(view, name="")
        with self.assertRaises(AssertionError):
            create_flow_definition_skeleton(view, name="flow", version="")
        with self.assertRaises(AssertionError):
            create_flow_definition_skeleton(view, name="flow", revision="")

    def test_visual_decision_round_trip_remains_non_executable(
        self,
    ) -> None:
        source = "\n".join(
            (
                "flowchart TD",
                "Start([Start]) --> Review{Review}",
                "Review -->|Yes| Approve[Approve]",
                "Review -->|No| Reject[Reject]",
            )
        )
        initial = normalize_mermaid_flow_view(
            source,
            import_mode=FlowViewImportMode.PRESENTATION,
        )
        rendered = render_mermaid_view(initial.view)
        round_trip = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )
        skeleton = create_flow_definition_skeleton(
            round_trip.view,
            name="visual_review",
        )
        binding = bind_flow_view_definition(round_trip.view, skeleton)
        validation = validate_flow_definition(skeleton)

        self.assertTrue(initial.ok, initial.public_diagnostics)
        self.assertTrue(rendered.ok, rendered.public_diagnostics)
        self.assertTrue(round_trip.ok, round_trip.public_diagnostics)
        self.assertEqual(
            round_trip.view.node_map["Review"].shape,
            FlowViewNodeShape.DIAMOND,
        )
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in round_trip.view.edges
            ],
            [
                ("Start", "Review", None),
                ("Review", "Approve", "Yes"),
                ("Review", "Reject", "No"),
            ],
        )
        self.assertEqual(
            {node.type for node in skeleton.nodes},
            {FLOW_VIEW_SKELETON_NODE_TYPE},
        )
        self.assertTrue(
            all(node.config["executable"] is False for node in skeleton.nodes)
        )
        self.assertTrue(all(edge.condition is None for edge in skeleton.edges))
        self.assertFalse(binding.ok)
        self.assertFalse(validation.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in binding.diagnostics],
            [
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_node_semantics",
                "flow.view.binding.missing_edge_semantics",
                "flow.view.binding.missing_edge_semantics",
                "flow.view.binding.missing_edge_semantics",
            ],
        )
        self.assertNotIn("decision", {node.type for node in skeleton.nodes})


if __name__ == "__main__":
    main()
