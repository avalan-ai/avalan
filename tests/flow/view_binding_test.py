from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FLOW_VIEW_SKELETON_NODE_TYPE,
    FLOW_VIEW_SKELETON_TAG,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
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
    normalize_mermaid_flow_view,
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


if __name__ == "__main__":
    main()
