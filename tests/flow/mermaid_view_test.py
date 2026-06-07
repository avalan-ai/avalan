from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowSourceSpan,
    FlowView,
    FlowViewDirection,
    FlowViewEdgeStyle,
    FlowViewImportMode,
    FlowViewNodeShape,
    MermaidFlowViewNormalizationResult,
    normalize_mermaid_flow_view,
    normalize_mermaid_import_to_flow_view,
    parse_mermaid_import,
)
from avalan.flow.connection import Connection
from avalan.flow.definition import FlowDefinition
from avalan.flow.flow import Flow
from avalan.flow.node import Node


class MermaidFlowViewNormalizerTestCase(TestCase):
    def test_normalize_mermaid_flow_view_preserves_supported_topology(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "flowchart LR",
                    'A(["Start"]) -->|yes| B{Check}',
                    "B -.-> C --> D",
                    "classDef active fill:#fff,stroke:#333",
                    "class A active",
                    "style B fill:#eee,stroke:#111",
                    "linkStyle 1 stroke:#f00",
                    "%% note",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/diagram.mmd",
        )

        view = result.view

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(view.direction, FlowViewDirection.LR)
        self.assertEqual(
            [node.id for node in view.nodes], ["A", "B", "C", "D"]
        )
        self.assertEqual(view.node_map["A"].label, "Start")
        self.assertEqual(view.node_map["A"].shape, FlowViewNodeShape.STADIUM)
        self.assertEqual(view.node_map["A"].classes, ("active",))
        self.assertFalse(view.node_map["A"].implicit)
        self.assertEqual(view.node_map["B"].label, "Check")
        self.assertEqual(view.node_map["B"].shape, FlowViewNodeShape.DIAMOND)
        self.assertEqual(view.node_map["B"].style["fill"], "#eee")
        self.assertTrue(view.node_map["C"].implicit)
        self.assertEqual(
            [(edge.source, edge.target, edge.label) for edge in view.edges],
            [
                ("A", "B", "yes"),
                ("B", "C", None),
                ("C", "D", None),
            ],
        )
        self.assertEqual(view.edges[0].style, FlowViewEdgeStyle.SOLID)
        self.assertEqual(view.edges[1].style, FlowViewEdgeStyle.DOTTED)
        self.assertEqual(view.edges[2].id, "C_to_D")
        self.assertEqual(
            cast(dict[str, object], view.edges[0].metadata["mermaid"])[
                "arrow"
            ],
            "-->",
        )
        self.assertEqual(view.class_definitions[0].name, "active")
        self.assertEqual(
            view.class_definitions[0].properties["stroke"], "#333"
        )
        self.assertEqual(view.styles[0].target, "B")
        self.assertEqual(view.link_styles[0].edge_index, 1)
        self.assertEqual(view.link_styles[0].properties["stroke"], "#f00")
        self.assertEqual(view.comments[0].text, "note")
        self.assertNotIn("customer", str(result.public_diagnostics))
        self.assertNotIsInstance(view, FlowDefinition)
        self.assertNotIsInstance(view, Flow)
        self.assertNotIsInstance(view.nodes[0], Node)
        self.assertNotIsInstance(view.edges[0], Connection)

    def test_normalize_mermaid_flow_view_expands_shorthand_in_source_order(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA & B --> C & D",
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            [node.id for node in result.view.nodes], ["A", "B", "C", "D"]
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.view.edges],
            [
                ("A", "C"),
                ("A", "D"),
                ("B", "C"),
                ("B", "D"),
            ],
        )
        self.assertTrue(all(node.implicit for node in result.view.nodes))
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.security.ambiguous_shorthand",
        )

    def test_normalize_mermaid_flow_view_preserves_nested_groups(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "graph TD",
                    "subgraph outer[Outer]",
                    "subgraph inner[Inner]",
                    "A --> B",
                    "end",
                    "C",
                    "end",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        outer = result.view.group_map["outer"]
        inner = result.view.group_map["inner"]

        self.assertTrue(result.ok)
        self.assertEqual(outer.label, "Outer")
        self.assertEqual(outer.parent, None)
        self.assertEqual(outer.groups, ("inner",))
        self.assertEqual(outer.nodes, ("C",))
        self.assertEqual(inner.label, "Inner")
        self.assertEqual(inner.parent, "outer")
        self.assertEqual(inner.nodes, ("A", "B"))
        self.assertEqual(result.view.node_map["A"].group, "inner")
        self.assertEqual(result.view.node_map["C"].group, "outer")

    def test_normalize_mermaid_flow_view_merges_duplicate_groups_and_shapes(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "graph",
                    "subgraph lane[First]",
                    "A",
                    "end",
                    "subgraph outer[Outer]",
                    "subgraph lane[Second]",
                    "S[[Sub]] --> Db[(Data)]",
                    "Db --> Circle((Circle))",
                    "Circle --> Hex{{Hex}}",
                    "Hex --> Round(Round)",
                    "end",
                    "end",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        lane = result.view.group_map["lane"]

        self.assertTrue(result.ok)
        self.assertEqual(result.view.direction, None)
        self.assertEqual(lane.label, "Second")
        self.assertEqual(lane.parent, "outer")
        self.assertEqual(result.view.group_map["outer"].groups, ("lane",))
        self.assertEqual(
            result.view.node_map["S"].shape, FlowViewNodeShape.SUBROUTINE
        )
        self.assertEqual(
            result.view.node_map["Db"].shape, FlowViewNodeShape.CYLINDER
        )
        self.assertEqual(
            result.view.node_map["Circle"].shape, FlowViewNodeShape.CIRCLE
        )
        self.assertEqual(
            result.view.node_map["Hex"].shape, FlowViewNodeShape.HEXAGON
        )
        self.assertEqual(
            result.view.node_map["Round"].shape, FlowViewNodeShape.ROUNDED
        )

    def test_normalize_mermaid_flow_view_keeps_bidirectional_edges_visual(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA <==> B\nA <==> B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.view.edges), 2)
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.view.edges],
            [("A", "B"), ("A", "B")],
        )
        self.assertTrue(result.view.edges[0].bidirectional)
        self.assertEqual(result.view.edges[0].style, FlowViewEdgeStyle.THICK)
        self.assertEqual(result.view.edges[0].id, "A_to_B")
        self.assertEqual(result.view.edges[1].id, "A_to_B_2")

    def test_normalize_mermaid_flow_view_skips_malformed_directives(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "\n".join(
                (
                    "graph TD",
                    "click A href",
                    "classDef lonely",
                    "class A",
                    "style A",
                    "linkStyle route",
                )
            ),
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.view.class_definitions, ())
        self.assertEqual(result.view.styles, ())
        self.assertEqual(result.view.link_styles, ())
        self.assertIn(
            "flow.mermaid.security.unsafe_link_directive",
            [diagnostic.code for diagnostic in result.diagnostics],
        )
        self.assertIn(
            "flow.mermaid.security.malformed_directive",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    def test_normalize_mermaid_flow_view_keeps_named_link_styles(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA --> B\nlinkStyle route stroke:#0a0",
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.view.link_styles[0].edge, "route")
        self.assertEqual(
            result.view.link_styles[0].properties["stroke"], "#0a0"
        )

    def test_malformed_shorthand_keeps_only_security_warning(
        self,
    ) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA & --> B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.view.edges, ())
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.mermaid.security.ambiguous_shorthand"],
        )

    def test_executable_import_errors_remain_on_normalized_view(self) -> None:
        result = normalize_mermaid_flow_view(
            "graph TD\nA & B --> C",
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.view.has_errors)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.security.ambiguous_shorthand",
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in result.view.edges],
            [("A", "C"), ("B", "C")],
        )

    def test_normalization_result_is_frozen_and_validated(self) -> None:
        import_validation = parse_mermaid_import(
            "graph TD\nA --> B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )
        result = normalize_mermaid_import_to_flow_view(import_validation)
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.invalid",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=FlowSourceSpan(start_line=1, start_column=1),
            message="Invalid Mermaid.",
        )

        manual = MermaidFlowViewNormalizationResult(
            import_validation=import_validation,
            view=result.view,
            diagnostics=(diagnostic,),
        )

        self.assertFalse(manual.ok)
        self.assertEqual(manual.public_diagnostics[0]["code"], diagnostic.code)
        with self.assertRaises(FrozenInstanceError):
            manual.view = FlowView(  # type: ignore[misc]
                import_mode=FlowViewImportMode.PRESENTATION
            )

    def test_normalization_rejects_invalid_arguments(self) -> None:
        import_validation = parse_mermaid_import(
            "graph TD\nA --> B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )
        result = normalize_mermaid_import_to_flow_view(import_validation)

        with self.assertRaises(AssertionError):
            normalize_mermaid_flow_view(  # type: ignore[arg-type]
                1,
                import_mode=FlowViewImportMode.PRESENTATION,
            )
        with self.assertRaises(AssertionError):
            normalize_mermaid_flow_view(
                "graph TD",
                import_mode="presentation",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            normalize_mermaid_flow_view(
                "graph TD",
                import_mode=FlowViewImportMode.PRESENTATION,
                source="",
            )
        with self.assertRaises(AssertionError):
            normalize_mermaid_import_to_flow_view(object())  # type: ignore[arg-type]

        invalid_cases = (
            {"import_validation": object(), "view": result.view},
            {
                "import_validation": import_validation,
                "view": object(),
            },
            {
                "import_validation": import_validation,
                "view": result.view,
                "diagnostics": [object()],
            },
            {
                "import_validation": import_validation,
                "view": result.view,
                "diagnostics": (object(),),
            },
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    MermaidFlowViewNormalizationResult(**case)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
