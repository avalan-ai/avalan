from pathlib import Path
from unittest import TestCase, main

from avalan.flow import (
    FlowDefinition,
    FlowDiagnosticSeverity,
    FlowEdgeDefinition,
    FlowNodeDefinition,
    FlowView,
    FlowViewDirection,
    FlowViewEdgeStyle,
    FlowViewImportMode,
    FlowViewNodeShape,
    MermaidDiagramKind,
    MermaidFlowViewNormalizationResult,
    MermaidParseResult,
    MermaidTokenType,
    bind_flow_view_definition,
    normalize_mermaid_flow_view,
    parse_mermaid,
    parse_mermaid_import,
    render_mermaid_view,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "flow"
BINDING_ROOT = FIXTURE_ROOT / "binding"
POSITIVE_ROOT = FIXTURE_ROOT / "mermaid" / "positive"
NEGATIVE_ROOT = FIXTURE_ROOT / "mermaid" / "negative"
SECURITY_ROOT = FIXTURE_ROOT / "mermaid" / "security" / "executable_import"

POSITIVE_FIXTURES = {
    "flowchart_bt_nested_chained.mmd": (),
    "flowchart_lr_shapes_and_styles.mmd": (),
    "flowchart_rl_bidirectional.mmd": (),
    "flowchart_td_quoted_labels.mmd": (),
    "graph_tb_shorthand.mmd": (
        "flow.mermaid.security.ambiguous_shorthand",
        "flow.mermaid.security.ambiguous_shorthand",
    ),
    "graph_td_semicolons.mmd": (),
}

NEGATIVE_FIXTURES = {
    "ambiguous_shorthand_executable.mmd": (
        "flow.mermaid.security.ambiguous_shorthand",
    ),
    "frontmatter_executable.mmd": ("flow.mermaid.security.frontmatter",),
    "html_label.mmd": ("flow.mermaid.security.html_label",),
    "init_directive.mmd": ("flow.mermaid.security.init_directive",),
    "malformed_directive.mmd": ("flow.mermaid.security.malformed_directive",),
    "malformed_subgraph.mmd": ("flow.mermaid.parser.unclosed_subgraph",),
    "script_like_label.mmd": ("flow.mermaid.security.script_like_label",),
    "unknown_directive.mmd": ("flow.mermaid.security.unknown_directive",),
    "unbalanced_subgraph.mmd": ("flow.mermaid.parser.unbalanced_subgraph",),
    "unclosed_edge_label.mmd": ("flow.mermaid.parser.unclosed_edge_label",),
    "unclosed_markdown_label.mmd": (
        "flow.mermaid.parser.unclosed_markdown_label",
    ),
    "unclosed_quoted_label.mmd": (
        "flow.mermaid.parser.unclosed_quoted_label",
    ),
    "unsafe_directives.mmd": (
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_callback_directive",
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_external_link",
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_external_link",
        "flow.mermaid.security.unsafe_callback_directive",
    ),
    "unsupported_diagram_type.mmd": (
        "flow.mermaid.security.unsupported_diagram_type",
    ),
}

SECURITY_FIXTURES = {
    "ambiguous_shorthand.mmd": ("flow.mermaid.security.ambiguous_shorthand",),
    "callback_directive.mmd": (
        "flow.mermaid.security.unsafe_callback_directive",
    ),
    "click_directive.mmd": (
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_callback_directive",
    ),
    "frontmatter.mmd": (
        "flow.mermaid.security.frontmatter",
        "flow.mermaid.security.frontmatter",
    ),
    "href_directive.mmd": (
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_external_link",
    ),
    "html_label.mmd": ("flow.mermaid.security.html_label",),
    "init_directive.mmd": ("flow.mermaid.security.init_directive",),
    "link_directive.mmd": (
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_external_link",
    ),
    "malformed_directive.mmd": ("flow.mermaid.security.malformed_directive",),
    "malformed_subgraph.mmd": (),
    "script_like_label.mmd": ("flow.mermaid.security.script_like_label",),
    "unknown_directive.mmd": ("flow.mermaid.security.unknown_directive",),
    "unsafe_external_link.mmd": (
        "flow.mermaid.security.unsafe_link_directive",
        "flow.mermaid.security.unsafe_external_link",
    ),
    "unsupported_diagram_type.mmd": (
        "flow.mermaid.security.unsupported_diagram_type",
    ),
}


class MermaidConformanceTestCase(TestCase):
    def test_positive_fixtures_parse_normalize_and_round_trip(
        self,
    ) -> None:
        self.assertEqual(set(POSITIVE_FIXTURES), _fixture_names(POSITIVE_ROOT))
        seen = _SeenMermaidSurface()

        for filename, expected_codes in POSITIVE_FIXTURES.items():
            with self.subTest(filename=filename):
                path = POSITIVE_ROOT / filename
                text = _read(path)
                parsed = parse_mermaid(text, source=str(path))
                normalized = normalize_mermaid_flow_view(
                    text,
                    import_mode=FlowViewImportMode.PRESENTATION,
                    source=str(path),
                )
                rendered = render_mermaid_view(normalized.view)
                reparsed = normalize_mermaid_flow_view(
                    rendered.source,
                    import_mode=FlowViewImportMode.EXECUTABLE,
                )

                self.assertTrue(parsed.ok, parsed.public_diagnostics)
                self.assertTrue(normalized.ok, normalized.public_diagnostics)
                self.assertEqual(_codes(normalized), expected_codes)
                self.assertTrue(rendered.ok, rendered.public_diagnostics)
                self.assertTrue(reparsed.ok, reparsed.public_diagnostics)
                self.assertEqual(
                    _view_signature(normalized.view),
                    _view_signature(reparsed.view),
                )
                seen.record(parsed, normalized)

        self.assertEqual(
            seen.diagram_kinds,
            {MermaidDiagramKind.GRAPH, MermaidDiagramKind.FLOWCHART},
        )
        self.assertEqual(seen.directions, set(FlowViewDirection))
        self.assertTrue(
            {
                FlowViewNodeShape.RECTANGLE,
                FlowViewNodeShape.ROUNDED,
                FlowViewNodeShape.DIAMOND,
                FlowViewNodeShape.CIRCLE,
                FlowViewNodeShape.DOUBLE_CIRCLE,
                FlowViewNodeShape.STADIUM,
                FlowViewNodeShape.SUBROUTINE,
                FlowViewNodeShape.CYLINDER,
                FlowViewNodeShape.HEXAGON,
            }.issubset(seen.shapes)
        )
        self.assertEqual(seen.edge_styles, set(FlowViewEdgeStyle))
        self.assertTrue(seen.bidirectional)
        self.assertTrue(seen.chained_edges)
        self.assertTrue(seen.multi_source_target)
        self.assertTrue(seen.nested_groups)
        self.assertTrue(seen.comments)
        self.assertTrue(seen.semicolons)
        self.assertTrue(seen.class_definitions)
        self.assertTrue(seen.classes)
        self.assertTrue(seen.styles)
        self.assertTrue(seen.link_styles)
        self.assertTrue(seen.labeled_edges)
        self.assertTrue(seen.quoted_labels)
        self.assertTrue(seen.unquoted_labels)
        self.assertTrue(seen.markdown_labels)

    def test_negative_fixtures_fail_with_stable_diagnostics(self) -> None:
        self.assertEqual(set(NEGATIVE_FIXTURES), _fixture_names(NEGATIVE_ROOT))

        for filename, expected_codes in NEGATIVE_FIXTURES.items():
            with self.subTest(filename=filename):
                path = NEGATIVE_ROOT / filename
                result = parse_mermaid_import(
                    _read(path),
                    import_mode=FlowViewImportMode.EXECUTABLE,
                    source=str(path),
                )

                self.assertFalse(result.ok)
                self.assertEqual(_codes(result), expected_codes)
                self.assertTrue(
                    all(
                        diagnostic.severity == FlowDiagnosticSeverity.ERROR
                        for diagnostic in result.diagnostics
                    )
                )
                self.assertNotIn(
                    str(NEGATIVE_ROOT), str(result.public_diagnostics)
                )
                self.assertNotIn(
                    "example.test", str(result.public_diagnostics)
                )

    def test_security_fixture_corpus_remains_executable_import_errors(
        self,
    ) -> None:
        self.assertEqual(set(SECURITY_FIXTURES), _fixture_names(SECURITY_ROOT))

        for filename, expected_codes in SECURITY_FIXTURES.items():
            with self.subTest(filename=filename):
                path = SECURITY_ROOT / filename
                result = parse_mermaid_import(
                    _read(path),
                    import_mode=FlowViewImportMode.EXECUTABLE,
                    source=str(path),
                )

                self.assertFalse(result.ok)
                self.assertEqual(_security_codes(result), expected_codes)
                self.assertTrue(
                    all(
                        diagnostic.severity == FlowDiagnosticSeverity.ERROR
                        for diagnostic in result.diagnostics
                        if diagnostic.code.startswith("flow.mermaid.security.")
                    )
                )
                self.assertNotIn(
                    str(SECURITY_ROOT), str(result.public_diagnostics)
                )
                self.assertNotIn(
                    "example.test", str(result.public_diagnostics)
                )

    def test_binding_failure_fixture_reports_missing_semantics(self) -> None:
        self.assertEqual(
            {"topology_mismatch.mmd"}, _fixture_names(BINDING_ROOT)
        )
        view = normalize_mermaid_flow_view(
            _read(BINDING_ROOT / "topology_mismatch.mmd"),
            import_mode=FlowViewImportMode.EXECUTABLE,
            source=str(BINDING_ROOT / "topology_mismatch.mmd"),
        ).view
        definition = FlowDefinition(
            name="mismatch",
            nodes=(
                FlowNodeDefinition(name="A", type="input"),
                FlowNodeDefinition(name="D", type="pass-through"),
            ),
            edges=(FlowEdgeDefinition(source="A", target="D"),),
        )

        result = bind_flow_view_definition(view, definition)

        self.assertFalse(result.ok)
        self.assertEqual(
            _codes(result),
            (
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
            ),
        )
        self.assertNotIn(str(BINDING_ROOT), str(result.public_diagnostics))


class _SeenMermaidSurface:
    def __init__(self) -> None:
        self.diagram_kinds: set[MermaidDiagramKind] = set()
        self.directions: set[FlowViewDirection] = set()
        self.shapes: set[FlowViewNodeShape] = set()
        self.edge_styles: set[FlowViewEdgeStyle] = set()
        self.bidirectional = False
        self.chained_edges = False
        self.multi_source_target = False
        self.nested_groups = False
        self.comments = False
        self.semicolons = False
        self.class_definitions = False
        self.classes = False
        self.styles = False
        self.link_styles = False
        self.labeled_edges = False
        self.quoted_labels = False
        self.unquoted_labels = False
        self.markdown_labels = False

    def record(
        self,
        parsed: MermaidParseResult,
        normalized: MermaidFlowViewNormalizationResult,
    ) -> None:
        if parsed.ast.diagram_kind is not None:
            self.diagram_kinds.add(parsed.ast.diagram_kind)
        if normalized.view.direction is not None:
            self.directions.add(normalized.view.direction)
        self.shapes.update(node.shape for node in normalized.view.nodes)
        self.edge_styles.update(edge.style for edge in normalized.view.edges)
        self.bidirectional = self.bidirectional or any(
            edge.bidirectional for edge in normalized.view.edges
        )
        self.chained_edges = self.chained_edges or any(
            len(statement.edges) > 1
            for statement in parsed.ast.statements
            if hasattr(statement, "edges")
        )
        self.multi_source_target = self.multi_source_target or any(
            diagnostic.code == "flow.mermaid.security.ambiguous_shorthand"
            for diagnostic in normalized.diagnostics
        )
        self.nested_groups = self.nested_groups or any(
            group.parent is not None for group in normalized.view.groups
        )
        self.comments = self.comments or bool(normalized.view.comments)
        self.semicolons = self.semicolons or any(
            token.type == MermaidTokenType.SEMICOLON
            for token in parsed.cst.tokens
        )
        self.class_definitions = self.class_definitions or bool(
            normalized.view.class_definitions
        )
        self.classes = self.classes or any(
            node.classes for node in normalized.view.nodes
        )
        self.styles = self.styles or bool(normalized.view.styles)
        self.link_styles = self.link_styles or bool(
            normalized.view.link_styles
        )
        self.labeled_edges = self.labeled_edges or any(
            edge.label is not None for edge in normalized.view.edges
        )
        self.quoted_labels = self.quoted_labels or any(
            token.type == MermaidTokenType.QUOTED_LABEL
            for token in parsed.cst.tokens
        )
        self.unquoted_labels = (
            self.unquoted_labels or _has_unquoted_mermaid_label(parsed)
        )
        self.markdown_labels = self.markdown_labels or any(
            token.type == MermaidTokenType.MARKDOWN_LABEL
            for token in parsed.cst.tokens
        )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fixture_names(root: Path) -> set[str]:
    return {path.name for path in root.glob("*.mmd")}


def _codes(result: object) -> tuple[str, ...]:
    diagnostics = getattr(result, "diagnostics")
    return tuple(diagnostic.code for diagnostic in diagnostics)


def _security_codes(result: object) -> tuple[str, ...]:
    diagnostics = getattr(result, "diagnostics")
    return tuple(
        diagnostic.code
        for diagnostic in diagnostics
        if diagnostic.code.startswith("flow.mermaid.security.")
    )


def _has_unquoted_mermaid_label(parsed: MermaidParseResult) -> bool:
    tokens = parsed.cst.tokens
    for index, token in enumerate(tokens[1:-1], start=1):
        if token.type != MermaidTokenType.IDENTIFIER:
            continue
        previous_token = tokens[index - 1]
        next_token = tokens[index + 1]
        if (
            previous_token.type == MermaidTokenType.SHAPE_DELIMITER
            and next_token.type == MermaidTokenType.SHAPE_DELIMITER
        ):
            return True
    return False


def _view_signature(view: FlowView) -> dict[str, object]:
    return {
        "direction": view.direction,
        "nodes": tuple(
            (
                node.id,
                node.label,
                node.shape,
                node.classes,
                tuple(sorted(node.style.items())),
                node.group,
            )
            for node in view.nodes
        ),
        "edges": tuple(
            (
                edge.source,
                edge.target,
                edge.label,
                edge.style,
                edge.bidirectional,
            )
            for edge in view.edges
        ),
        "groups": tuple(
            (
                group.id,
                group.label,
                group.parent,
                group.nodes,
                group.groups,
            )
            for group in view.groups
        ),
        "classes": tuple(
            (
                class_definition.name,
                tuple(sorted(class_definition.properties.items())),
            )
            for class_definition in view.class_definitions
        ),
        "styles": tuple(
            (
                style.target,
                tuple(sorted(style.properties.items())),
            )
            for style in view.styles
        ),
        "link_styles": tuple(
            (
                link_style.edge,
                link_style.edge_index,
                tuple(sorted(link_style.properties.items())),
            )
            for link_style in view.link_styles
        ),
        "comments": tuple(comment.text for comment in view.comments),
    }


if __name__ == "__main__":
    main()
