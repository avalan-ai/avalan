from dataclasses import FrozenInstanceError
from unittest import TestCase, main

import avalan.flow as flow
from avalan.flow import (
    FLOW_VIEW_SKELETON_TAG,
    FlowDefinition,
    FlowDefinitionSkeletonResult,
    FlowEdgeDefinition,
    FlowEntryBehavior,
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
    parse_mermaid_view,
    render_flow_view,
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

    def test_private_parser_internals_are_not_exported_from_package(
        self,
    ) -> None:
        self.assertFalse(hasattr(flow, "_MermaidTokenizer"))


if __name__ == "__main__":
    main()
