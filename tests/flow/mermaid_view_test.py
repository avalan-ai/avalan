from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowNodeDefinition,
    FlowSourceSpan,
    FlowView,
    FlowViewClassDefinition,
    FlowViewComment,
    FlowViewDirection,
    FlowViewEdge,
    FlowViewEdgeStyle,
    FlowViewGroup,
    FlowViewImportMode,
    FlowViewLinkStyle,
    FlowViewNode,
    FlowViewNodeShape,
    FlowViewStyle,
    MermaidFlowViewNormalizationResult,
    MermaidRenderResult,
    flow_definition_to_flow_view,
    normalize_mermaid_flow_view,
    normalize_mermaid_import_to_flow_view,
    parse_mermaid_import,
    render_flow_definition_mermaid,
    render_mermaid_view,
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


class MermaidFlowViewRendererTestCase(TestCase):
    def test_render_mermaid_view_round_trips_supported_safe_view(
        self,
    ) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.EXECUTABLE,
            direction=FlowViewDirection.LR,
            nodes=(
                FlowViewNode(
                    id="A",
                    label='Start "safe"',
                    shape=FlowViewNodeShape.STADIUM,
                    classes=("active",),
                    group="lane",
                ),
                FlowViewNode(
                    id="B",
                    label="Review",
                    shape=FlowViewNodeShape.DIAMOND,
                    group="lane",
                ),
                FlowViewNode(
                    id="C",
                    label="Done",
                    shape=FlowViewNodeShape.DOUBLE_CIRCLE,
                ),
            ),
            edges=(
                FlowViewEdge(
                    id="A_to_B",
                    source="A",
                    target="B",
                    label="yes",
                ),
                FlowViewEdge(
                    id="B_to_C",
                    source="B",
                    target="C",
                    style=FlowViewEdgeStyle.DOTTED,
                ),
            ),
            groups=(FlowViewGroup(id="lane", label="Lane", nodes=("A", "B")),),
            class_definitions=(
                FlowViewClassDefinition(
                    name="active",
                    properties={"stroke": "#333"},
                ),
            ),
            styles=(FlowViewStyle(target="B", properties={"fill": "#eee"}),),
            link_styles=(
                FlowViewLinkStyle(edge_index=1, properties={"stroke": "#0a0"}),
            ),
            comments=(FlowViewComment(text="safe comment"),),
        )

        rendered = render_mermaid_view(view)
        reparsed = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertTrue(rendered.ok)
        self.assertTrue(reparsed.ok)
        self.assertEqual(rendered.diagnostics, ())
        self.assertEqual(reparsed.view.direction, FlowViewDirection.LR)
        self.assertEqual(
            [
                (node.id, node.label, node.shape, node.group)
                for node in reparsed.view.nodes
            ],
            [
                ("A", 'Start "safe"', FlowViewNodeShape.STADIUM, "lane"),
                ("B", "Review", FlowViewNodeShape.DIAMOND, "lane"),
                ("C", "Done", FlowViewNodeShape.DOUBLE_CIRCLE, None),
            ],
        )
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label, edge.style)
                for edge in reparsed.view.edges
            ],
            [
                ("A", "B", "yes", FlowViewEdgeStyle.SOLID),
                ("B", "C", None, FlowViewEdgeStyle.DOTTED),
            ],
        )
        self.assertEqual(reparsed.view.group_map["lane"].label, "Lane")
        self.assertEqual(reparsed.view.node_map["A"].classes, ("active",))
        self.assertEqual(reparsed.view.node_map["B"].style["fill"], "#eee")
        self.assertEqual(reparsed.view.class_definitions[0].name, "active")
        self.assertEqual(reparsed.view.link_styles[0].edge_index, 1)
        self.assertEqual(reparsed.view.comments[0].text, "safe comment")

    def test_render_flow_definition_uses_flow_view_without_config_leak(
        self,
    ) -> None:
        definition = FlowDefinition(
            name="flow",
            nodes=(
                FlowNodeDefinition(
                    name="start",
                    type="constant",
                    config={"private_prompt": "do-not-render"},
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

        view = flow_definition_to_flow_view(definition)
        rendered = render_flow_definition_mermaid(definition)
        reparsed = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertEqual(view.import_mode, FlowViewImportMode.EXECUTABLE)
        self.assertEqual([node.id for node in view.nodes], ["start", "finish"])
        self.assertTrue(rendered.ok)
        self.assertNotIn("constant", rendered.source)
        self.assertNotIn("private_prompt", rendered.source)
        self.assertNotIn("do-not-render", rendered.source)
        self.assertEqual(
            [
                (edge.source, edge.target, edge.label)
                for edge in reparsed.view.edges
            ],
            [("start", "finish", "done")],
        )

    def test_render_mermaid_view_escapes_labels_and_reports_bad_ids(
        self,
    ) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.EXECUTABLE,
            nodes=(
                FlowViewNode(id="safe", label='<script>alert("x")</script>'),
                FlowViewNode(id="private token", label="bad"),
            ),
            edges=(
                FlowViewEdge(
                    id="route",
                    source="safe",
                    target="private token",
                    label="x|{{secret}}",
                ),
            ),
            styles=(
                FlowViewStyle(
                    target="safe",
                    properties={"background": "url(https://private.test/a)"},
                ),
            ),
        )

        rendered = render_mermaid_view(view)
        reparsed = normalize_mermaid_flow_view(
            rendered.source,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertFalse(rendered.ok)
        self.assertTrue(reparsed.ok)
        self.assertNotIn("<script>", rendered.source)
        self.assertNotIn("{{secret}}", rendered.source)
        self.assertNotIn("https://private.test", rendered.source)
        self.assertNotIn("private token", str(rendered.public_diagnostics))
        self.assertEqual(
            [diagnostic.code for diagnostic in rendered.diagnostics],
            [
                "flow.mermaid.parser.renderer_invalid_node_identifier",
                "flow.mermaid.parser.renderer_invalid_edge_target",
                "flow.mermaid.parser.renderer_unsafe_style_property",
            ],
        )

    def test_render_mermaid_view_covers_safe_variants_and_warnings(
        self,
    ) -> None:
        view = FlowView(
            import_mode=FlowViewImportMode.EXECUTABLE,
            nodes=(
                FlowViewNode(
                    id="Sub",
                    label="Sub",
                    shape=FlowViewNodeShape.SUBROUTINE,
                    group="child",
                    classes=("bad class", "good"),
                ),
                FlowViewNode(
                    id="Data",
                    label="Data",
                    shape=FlowViewNodeShape.CYLINDER,
                    group="child",
                ),
                FlowViewNode(
                    id="Circle",
                    label="Circle",
                    shape=FlowViewNodeShape.CIRCLE,
                ),
                FlowViewNode(
                    id="Hex",
                    label="Hex",
                    shape=FlowViewNodeShape.HEXAGON,
                ),
                FlowViewNode(
                    id="Round",
                    label="Round",
                    shape=FlowViewNodeShape.ROUNDED,
                ),
                FlowViewNode(id="Rect", label="Rect"),
                FlowViewNode(id="-bad", label="Bad", classes=("good",)),
            ),
            edges=(
                FlowViewEdge(
                    id="invalid_source",
                    source="-bad",
                    target="Sub",
                ),
                FlowViewEdge(
                    id="thick",
                    source="Sub",
                    target="Data",
                    style=FlowViewEdgeStyle.THICK,
                ),
                FlowViewEdge(
                    id="bi_solid",
                    source="Data",
                    target="Circle",
                    bidirectional=True,
                ),
                FlowViewEdge(
                    id="bi_dotted",
                    source="Circle",
                    target="Hex",
                    style=FlowViewEdgeStyle.DOTTED,
                    bidirectional=True,
                ),
                FlowViewEdge(
                    id="bi_thick",
                    source="Hex",
                    target="Round",
                    style=FlowViewEdgeStyle.THICK,
                    bidirectional=True,
                ),
            ),
            groups=(
                FlowViewGroup(
                    id="root",
                    direction=FlowViewDirection.RL,
                    groups=("child",),
                ),
                FlowViewGroup(
                    id="child",
                    parent="root",
                    nodes=("Sub", "Data"),
                ),
                FlowViewGroup(id="bad group"),
            ),
            class_definitions=(
                FlowViewClassDefinition(name="bad class"),
                FlowViewClassDefinition(
                    name="good",
                    properties={"stroke-width": "2px"},
                ),
            ),
            styles=(
                FlowViewStyle(target="bad target", properties={"fill": "red"}),
                FlowViewStyle(
                    target="Rect",
                    properties={"fill": "red;blue"},
                ),
            ),
            link_styles=(
                FlowViewLinkStyle(
                    edge="bad target",
                    properties={"stroke": "#111"},
                ),
            ),
        )

        rendered = render_mermaid_view(view)
        rendered_definition_view = flow_definition_to_flow_view(
            FlowDefinition(
                name="duplicate_edges",
                nodes=(
                    FlowNodeDefinition(name="Sub", type="pass-through"),
                    FlowNodeDefinition(name="Data", type="pass-through"),
                ),
                edges=(
                    FlowEdgeDefinition(source="Sub", target="Data"),
                    FlowEdgeDefinition(source="Sub", target="Data"),
                ),
            )
        )

        self.assertFalse(rendered.ok)
        self.assertEqual(
            [edge.id for edge in rendered_definition_view.edges],
            ["Sub_to_Data", "Sub_to_Data_2"],
        )
        self.assertIn('Sub[["Sub"]]', rendered.source)
        self.assertIn('Data[("Data")]', rendered.source)
        self.assertIn('Circle(("Circle"))', rendered.source)
        self.assertIn('Hex{{"Hex"}}', rendered.source)
        self.assertIn('Round("Round")', rendered.source)
        self.assertIn('Rect["Rect"]', rendered.source)
        self.assertIn("Sub ==> Data", rendered.source)
        self.assertIn("Data <--> Circle", rendered.source)
        self.assertIn("Circle <-.-> Hex", rendered.source)
        self.assertIn("Hex <==> Round", rendered.source)
        self.assertIn("class Sub good", rendered.source)
        self.assertIn("classDef good stroke-width:2px", rendered.source)
        self.assertEqual(
            [diagnostic.code for diagnostic in rendered.diagnostics],
            [
                "flow.mermaid.parser.renderer_unsupported_group_direction",
                "flow.mermaid.parser.renderer_invalid_group_identifier",
                "flow.mermaid.parser.renderer_invalid_node_identifier",
                "flow.mermaid.parser.renderer_invalid_edge_source",
                "flow.mermaid.parser.renderer_invalid_class_identifier",
                "flow.mermaid.parser.renderer_invalid_class_identifier",
                "flow.mermaid.parser.renderer_invalid_style_target",
                "flow.mermaid.parser.renderer_unsafe_style_property",
                "flow.mermaid.parser.renderer_invalid_link_style_target",
            ],
        )

    def test_render_result_is_frozen_and_validated(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.renderer_invalid_node_identifier",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            path="view.nodes",
            message="Flow View node identifier cannot be rendered safely.",
        )
        result = MermaidRenderResult(
            source="flowchart TD\n",
            diagnostics=(diagnostic,),
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.public_diagnostics[0]["code"],
            "flow.mermaid.parser.renderer_invalid_node_identifier",
        )
        with self.assertRaises(FrozenInstanceError):
            result.source = ""  # type: ignore[misc]

        invalid_cases = (
            {"source": 1},
            {"source": "", "diagnostics": [diagnostic]},
            {"source": "", "diagnostics": (object(),)},
        )
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    MermaidRenderResult(**case)  # type: ignore[arg-type]

        with self.assertRaises(AssertionError):
            render_mermaid_view(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            flow_definition_to_flow_view(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            flow_definition_to_flow_view(
                FlowDefinition(name="flow", nodes=()),
                import_mode="executable",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            render_flow_definition_mermaid(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
