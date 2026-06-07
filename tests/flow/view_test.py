from dataclasses import FrozenInstanceError
from typing import cast
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
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
)
from avalan.flow.connection import Connection
from avalan.flow.definition import FlowDefinition
from avalan.flow.flow import Flow
from avalan.flow.node import Node


class FlowViewTestCase(TestCase):
    def test_flow_view_entities_are_frozen_and_copy_nested_mappings(
        self,
    ) -> None:
        span = FlowSourceSpan(
            source="/private/customer/diagram.mmd",
            start_line=2,
            start_column=3,
        )
        metadata = {"source": {"line": 2}, "tags": ["implicit"]}
        style = {"fill": "#fff", "stroke": "#333"}
        class_properties = {"stroke-width": "2px"}
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.unsupported",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=span,
            message="Unsupported syntax.",
        )
        node = FlowViewNode(
            id="review",
            label="Review",
            shape=FlowViewNodeShape.DIAMOND,
            classes=("risk",),
            style=style,
            metadata=metadata,
            source_span=span,
            implicit=True,
            group="checks",
        )
        edge = FlowViewEdge(
            id="review_to_done",
            source="review",
            target="done",
            label="Approved",
            style=FlowViewEdgeStyle.THICK,
            classes=("happy",),
            metadata={"arrow": {"token": "<==>"}},
            source_span=span,
            bidirectional=True,
        )
        group = FlowViewGroup(
            id="checks",
            label="Checks",
            parent="root",
            direction=FlowViewDirection.LR,
            nodes=("review",),
            groups=("nested",),
            classes=("lane",),
            style={"fill": "#eee"},
            metadata={"rank": 1},
            source_span=span,
        )
        class_definition = FlowViewClassDefinition(
            name="risk",
            properties=class_properties,
            source_span=span,
        )
        target_style = FlowViewStyle(
            target="review",
            properties={"color": "#111"},
            source_span=span,
        )
        link_style_by_edge = FlowViewLinkStyle(
            edge="review_to_done",
            properties={"stroke": "#0a0"},
            source_span=span,
        )
        link_style_by_index = FlowViewLinkStyle(
            edge_index=0,
            properties={"stroke-dasharray": "5 5"},
        )
        comment = FlowViewComment(text="", source_span=span)
        view = FlowView(
            import_mode=FlowViewImportMode.PRESENTATION,
            direction=FlowViewDirection.TD,
            nodes=(node,),
            edges=(edge,),
            groups=(group,),
            class_definitions=(class_definition,),
            styles=(target_style,),
            link_styles=(link_style_by_edge, link_style_by_index),
            comments=(comment,),
            diagnostics=(diagnostic,),
            metadata={"format": {"name": "mermaid"}, "tags": ["preview"]},
            source_span=span,
        )

        metadata["source"]["line"] = 9
        metadata["tags"].append("changed")
        style["fill"] = "#000"
        class_properties["stroke-width"] = "4px"

        self.assertEqual(node.id, "review")
        self.assertEqual(node.shape, FlowViewNodeShape.DIAMOND)
        self.assertTrue(node.implicit)
        self.assertEqual(node.group, "checks")
        self.assertEqual(node.style["fill"], "#fff")
        self.assertEqual(
            cast(dict[str, object], node.metadata["source"])["line"],
            2,
        )
        self.assertEqual(node.metadata["tags"], ("implicit",))
        self.assertEqual(edge.style, FlowViewEdgeStyle.THICK)
        self.assertTrue(edge.bidirectional)
        self.assertEqual(group.nodes, ("review",))
        self.assertEqual(group.groups, ("nested",))
        self.assertEqual(group.direction, FlowViewDirection.LR)
        self.assertEqual(class_definition.properties["stroke-width"], "2px")
        self.assertEqual(target_style.properties["color"], "#111")
        self.assertEqual(link_style_by_edge.edge, "review_to_done")
        self.assertEqual(link_style_by_index.edge_index, 0)
        self.assertEqual(comment.text, "")
        self.assertEqual(view.node_map["review"], node)
        self.assertEqual(view.edge_map["review_to_done"], edge)
        self.assertEqual(view.group_map["checks"], group)
        self.assertEqual(view.class_definition_map["risk"], class_definition)
        self.assertTrue(view.has_errors)
        self.assertNotIn("customer", str(view.public_diagnostics))
        self.assertEqual(
            cast(dict[str, object], view.metadata["format"])["name"],
            "mermaid",
        )
        self.assertEqual(view.metadata["tags"], ("preview",))
        self.assertNotIsInstance(view, FlowDefinition)
        self.assertNotIsInstance(view, Flow)
        self.assertNotIsInstance(node, Node)
        self.assertNotIsInstance(edge, Connection)
        with self.assertRaises(FrozenInstanceError):
            view.direction = FlowViewDirection.RL  # type: ignore[misc]

    def test_flow_view_defaults_are_inert_and_warning_only(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.presentation_warning",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=FlowSourceSpan(start_line=1, start_column=1),
            severity=FlowDiagnosticSeverity.WARNING,
            message="Presentation-only feature.",
        )

        view = FlowView(
            import_mode=FlowViewImportMode.EXECUTABLE,
            nodes=(FlowViewNode(id="start"),),
            diagnostics=(diagnostic,),
        )

        self.assertEqual(view.direction, None)
        self.assertEqual(view.nodes[0].label, None)
        self.assertEqual(view.nodes[0].shape, FlowViewNodeShape.RECTANGLE)
        self.assertEqual(view.nodes[0].classes, ())
        self.assertEqual(dict(view.nodes[0].style), {})
        self.assertEqual(dict(view.nodes[0].metadata), {})
        self.assertEqual(view.edges, ())
        self.assertFalse(view.has_errors)
        self.assertEqual(
            view.public_diagnostics,
            (
                {
                    "code": "flow.mermaid.parser.presentation_warning",
                    "category": "mermaid_parser",
                    "severity": "warning",
                    "message": "Presentation-only feature.",
                    "source_span": {"start_line": 1, "start_column": 1},
                },
            ),
        )

    def test_view_enums_are_stable(self) -> None:
        self.assertEqual(FlowViewImportMode.PRESENTATION.value, "presentation")
        self.assertEqual(FlowViewImportMode.EXECUTABLE.value, "executable")
        self.assertEqual(FlowViewDirection.TD.value, "TD")
        self.assertEqual(FlowViewDirection.TB.value, "TB")
        self.assertEqual(FlowViewDirection.BT.value, "BT")
        self.assertEqual(FlowViewDirection.LR.value, "LR")
        self.assertEqual(FlowViewDirection.RL.value, "RL")
        self.assertEqual(FlowViewNodeShape.RECTANGLE.value, "rectangle")
        self.assertEqual(FlowViewNodeShape.ROUNDED.value, "rounded")
        self.assertEqual(FlowViewNodeShape.DIAMOND.value, "diamond")
        self.assertEqual(FlowViewNodeShape.CIRCLE.value, "circle")
        self.assertEqual(
            FlowViewNodeShape.DOUBLE_CIRCLE.value,
            "double_circle",
        )
        self.assertEqual(FlowViewNodeShape.STADIUM.value, "stadium")
        self.assertEqual(FlowViewNodeShape.SUBROUTINE.value, "subroutine")
        self.assertEqual(FlowViewNodeShape.CYLINDER.value, "cylinder")
        self.assertEqual(FlowViewNodeShape.HEXAGON.value, "hexagon")
        self.assertEqual(
            FlowViewNodeShape.PARALLELOGRAM.value,
            "parallelogram",
        )
        self.assertEqual(FlowViewNodeShape.TRAPEZOID.value, "trapezoid")
        self.assertEqual(FlowViewNodeShape.ASYMMETRIC.value, "asymmetric")
        self.assertEqual(FlowViewEdgeStyle.SOLID.value, "solid")
        self.assertEqual(FlowViewEdgeStyle.DOTTED.value, "dotted")
        self.assertEqual(FlowViewEdgeStyle.THICK.value, "thick")

    def test_invalid_view_nodes_raise_assertion_errors(self) -> None:
        invalid_cases = (
            {"id": ""},
            {"id": "node", "label": ""},
            {"id": "node", "shape": "diamond"},
            {"id": "node", "classes": ["risk"]},
            {"id": "node", "classes": ("",)},
            {"id": "node", "style": {"": "red"}},
            {"id": "node", "style": {"fill": ""}},
            {"id": "node", "metadata": []},
            {"id": "node", "metadata": {1: "bad"}},
            {"id": "node", "source_span": object()},
            {"id": "node", "implicit": "yes"},
            {"id": "node", "group": ""},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewNode(**case)  # type: ignore[arg-type]

    def test_invalid_view_edges_raise_assertion_errors(self) -> None:
        invalid_cases = (
            {"id": "", "source": "start", "target": "finish"},
            {"id": "edge", "source": "", "target": "finish"},
            {"id": "edge", "source": "start", "target": ""},
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "label": "",
            },
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "style": "dotted",
            },
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "classes": ["route"],
            },
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "metadata": {1: "bad"},
            },
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "source_span": object(),
            },
            {
                "id": "edge",
                "source": "start",
                "target": "finish",
                "bidirectional": "yes",
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewEdge(**case)  # type: ignore[arg-type]

    def test_invalid_view_groups_raise_assertion_errors(self) -> None:
        invalid_cases = (
            {"id": ""},
            {"id": "group", "label": ""},
            {"id": "group", "parent": ""},
            {"id": "group", "direction": "LR"},
            {"id": "group", "nodes": ["node"]},
            {"id": "group", "nodes": ("",)},
            {"id": "group", "groups": ["child"]},
            {"id": "group", "groups": ("",)},
            {"id": "group", "classes": ["lane"]},
            {"id": "group", "classes": ("",)},
            {"id": "group", "style": {"fill": ""}},
            {"id": "group", "metadata": []},
            {"id": "group", "source_span": object()},
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewGroup(**case)  # type: ignore[arg-type]

    def test_invalid_view_metadata_entities_raise_assertion_errors(
        self,
    ) -> None:
        invalid_class_cases = (
            {"name": ""},
            {"name": "risk", "properties": {"": "red"}},
            {"name": "risk", "source_span": object()},
        )
        invalid_style_cases = (
            {"target": ""},
            {"target": "node", "properties": []},
            {"target": "node", "source_span": object()},
        )
        invalid_link_style_cases = (
            {"properties": {"stroke": "red"}},
            {"edge": "", "properties": {"stroke": "red"}},
            {"edge_index": True, "properties": {"stroke": "red"}},
            {"edge_index": -1, "properties": {"stroke": "red"}},
            {"edge_index": 0, "properties": {"stroke": ""}},
            {"edge_index": 0, "source_span": object(), "properties": {}},
        )
        invalid_comments = (
            {"text": object()},
            {"text": "comment", "source_span": object()},
        )

        for case in invalid_class_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewClassDefinition(**case)  # type: ignore[arg-type]
        for case in invalid_style_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewStyle(**case)  # type: ignore[arg-type]
        for case in invalid_link_style_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewLinkStyle(**case)  # type: ignore[arg-type]
        for case in invalid_comments:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowViewComment(**case)  # type: ignore[arg-type]

    def test_invalid_views_raise_assertion_errors(self) -> None:
        node = FlowViewNode(id="start")
        edge = FlowViewEdge(id="route", source="start", target="finish")
        group = FlowViewGroup(id="group")
        class_definition = FlowViewClassDefinition(name="risk")
        style = FlowViewStyle(target="start")
        link_style = FlowViewLinkStyle(edge_index=0, properties={})
        comment = FlowViewComment(text="comment")
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.invalid",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=FlowSourceSpan(start_line=1, start_column=1),
            message="Invalid diagram.",
        )
        invalid_cases = (
            {"import_mode": "presentation"},
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "direction": "TD",
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "nodes": [node],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "nodes": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "nodes": (node, FlowViewNode(id="start")),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "edges": [edge],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "edges": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "edges": (edge, edge),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "groups": [group],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "groups": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "groups": (group, group),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "class_definitions": [class_definition],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "class_definitions": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "class_definitions": (
                    class_definition,
                    FlowViewClassDefinition(name="risk"),
                ),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "styles": [style],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "styles": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "link_styles": [link_style],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "link_styles": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "comments": [comment],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "comments": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "diagnostics": [diagnostic],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "diagnostics": (object(),),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "metadata": [],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "source_span": object(),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    FlowView(**case)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
