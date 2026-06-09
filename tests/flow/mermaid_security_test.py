from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
    FlowViewImportMode,
    MermaidAst,
    MermaidAstDirective,
    MermaidAstDirectiveKind,
    MermaidAstEdge,
    MermaidAstEdgeStatement,
    MermaidAstNode,
    MermaidCst,
    MermaidImportValidationResult,
    MermaidParseResult,
    parse_mermaid,
    parse_mermaid_import,
    validate_mermaid_import,
)

FIXTURE_ROOT = (
    Path(__file__).parents[1]
    / "fixtures"
    / "flow"
    / "mermaid"
    / "security"
    / "executable_import"
)

EXECUTABLE_FIXTURE_CODES = {
    "ambiguous_shorthand.mmd": "flow.mermaid.security.ambiguous_shorthand",
    "bidirectional_edge.mmd": "flow.mermaid.security.bidirectional_edge",
    "callback_directive.mmd": (
        "flow.mermaid.security.unsafe_callback_directive"
    ),
    "click_directive.mmd": "flow.mermaid.security.unsafe_link_directive",
    "duplicate_edge_id.mmd": "flow.mermaid.security.duplicate_edge_id",
    "frontmatter.mmd": "flow.mermaid.security.frontmatter",
    "href_directive.mmd": "flow.mermaid.security.unsafe_link_directive",
    "html_label.mmd": "flow.mermaid.security.html_label",
    "init_directive.mmd": "flow.mermaid.security.init_directive",
    "invalid_edge_id.mmd": "flow.mermaid.security.invalid_edge_id",
    "link_directive.mmd": "flow.mermaid.security.unsafe_link_directive",
    "malformed_edge_id.mmd": "flow.mermaid.parser.malformed_edge_id",
    "malformed_directive.mmd": "flow.mermaid.security.malformed_directive",
    "malformed_subgraph.mmd": "flow.mermaid.parser.unclosed_subgraph",
    "script_like_label.mmd": "flow.mermaid.security.script_like_label",
    "unknown_directive.mmd": "flow.mermaid.security.unknown_directive",
    "unsafe_external_link.mmd": "flow.mermaid.security.unsafe_external_link",
    "unsupported_diagram_type.mmd": (
        "flow.mermaid.security.unsupported_diagram_type"
    ),
}

PRESENTATION_WARNING_CASES = {
    "ambiguous": (
        "graph TD\nA & B --> C",
        "flow.mermaid.security.ambiguous_shorthand",
    ),
    "html": (
        'graph TD\nA["<b>Review</b>"] --> B',
        "flow.mermaid.security.html_label",
    ),
    "init": (
        "%%{init: {'theme': 'dark'}}%%\ngraph TD\nA --> B",
        "flow.mermaid.security.init_directive",
    ),
    "link": (
        "graph TD\nlink A https://example.test",
        "flow.mermaid.security.unsafe_link_directive",
    ),
    "malformed": (
        "graph TD\nstyle A",
        "flow.mermaid.security.malformed_directive",
    ),
    "script": (
        'graph TD\nA["{{ payload }}"] --> B',
        "flow.mermaid.security.script_like_label",
    ),
    "unknown": (
        "%%{themeVariables: {'danger': true}}%%\ngraph TD\nA --> B",
        "flow.mermaid.security.unknown_directive",
    ),
}


class MermaidImportSecurityTestCase(TestCase):
    def test_executable_import_rejects_security_fixtures(self) -> None:
        self.assertEqual(
            set(EXECUTABLE_FIXTURE_CODES),
            {path.name for path in FIXTURE_ROOT.glob("*.mmd")},
        )
        for filename, expected_code in EXECUTABLE_FIXTURE_CODES.items():
            with self.subTest(filename=filename):
                path = FIXTURE_ROOT / filename
                result = parse_mermaid_import(
                    path.read_text(encoding="utf-8"),
                    import_mode=FlowViewImportMode.EXECUTABLE,
                    source=str(path),
                )

                self.assertFalse(result.ok)
                self.assertIn(expected_code, _codes(result))
                self.assertEqual(
                    _diagnostic(result, expected_code).severity,
                    FlowDiagnosticSeverity.ERROR,
                )
                self.assertNotIn(
                    "example.test", str(result.public_diagnostics)
                )
                self.assertNotIn(
                    str(FIXTURE_ROOT), str(result.public_diagnostics)
                )

    def test_presentation_import_warns_for_inert_constructs(self) -> None:
        for name, (text, expected_code) in PRESENTATION_WARNING_CASES.items():
            with self.subTest(name=name):
                result = parse_mermaid_import(
                    text,
                    import_mode=FlowViewImportMode.PRESENTATION,
                    source=f"/private/customer/{name}.mmd",
                )

                self.assertTrue(result.ok)
                self.assertIn(expected_code, _codes(result))
                self.assertTrue(
                    all(
                        diagnostic.severity == FlowDiagnosticSeverity.WARNING
                        for diagnostic in result.diagnostics
                    )
                )
                self.assertNotIn("customer", str(result.public_diagnostics))
                self.assertNotIn(
                    "example.test", str(result.public_diagnostics)
                )

    def test_safe_visual_metadata_is_valid_in_both_import_modes(self) -> None:
        text = "\n".join(
            (
                "flowchart LR",
                "classDef active fill:#fff,stroke:#333",
                "A route_1@-->|yes| B{Check}",
                "B route-2@-.-> C",
                "class A active",
                "style B fill:#fff,stroke:#333",
                "linkStyle 0 stroke:#f00",
                "%% note",
            )
        )

        for import_mode in FlowViewImportMode:
            with self.subTest(import_mode=import_mode):
                result = parse_mermaid_import(text, import_mode=import_mode)

                self.assertTrue(result.ok)
                self.assertEqual(result.diagnostics, ())
                self.assertIn(
                    MermaidAstDirectiveKind.CLASS_DEF,
                    {
                        statement.kind
                        for statement in result.parse_result.ast.statements
                        if isinstance(statement, MermaidAstDirective)
                    },
                )

    def test_executable_import_rejects_invalid_explicit_edge_ids(
        self,
    ) -> None:
        cases = {
            "dotted": (
                "flowchart LR\nA route.one@--> B",
                (2, 3),
            ),
            "url_like": (
                "flowchart LR\nA https://route@--> B",
                (2, 3),
            ),
            "selector_like": (
                "flowchart LR\nA #route@--> B",
                (2, 3),
            ),
            "path_like": (
                "flowchart LR\nA route/path@--> B",
                (2, 3),
            ),
        }

        for name, (text, position) in cases.items():
            with self.subTest(name=name):
                result = parse_mermaid_import(
                    text,
                    import_mode=FlowViewImportMode.EXECUTABLE,
                    source="/private/customer/diagram.mmd",
                )

                self.assertFalse(result.ok)
                codes = _codes(result)
                self.assertIn(
                    "flow.mermaid.security.invalid_edge_id",
                    codes,
                )
                if name != "url_like":
                    self.assertEqual(
                        codes,
                        ("flow.mermaid.security.invalid_edge_id",),
                    )
                diagnostic = _diagnostic(
                    result,
                    "flow.mermaid.security.invalid_edge_id",
                )
                self.assertEqual(
                    diagnostic.source_span.start_line,
                    position[0],
                )
                self.assertEqual(
                    diagnostic.source_span.start_column,
                    position[1],
                )
                self.assertNotIn("customer", str(result.public_diagnostics))
                self.assertNotIn("route.one", str(result.public_diagnostics))
                self.assertNotIn(
                    "https://route", str(result.public_diagnostics)
                )
                self.assertNotIn("#route", str(result.public_diagnostics))
                self.assertNotIn("route/path", str(result.public_diagnostics))

    def test_executable_import_rejects_duplicate_explicit_edge_ids(
        self,
    ) -> None:
        result = parse_mermaid_import(
            "\n".join(
                (
                    "flowchart LR",
                    "A route_1@--> B",
                    "subgraph lane[Private customer label]",
                    "B route_1@--> C",
                    "end",
                )
            ),
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/diagram.mmd",
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            _codes(result),
            ("flow.mermaid.security.duplicate_edge_id",),
        )
        self.assertEqual(result.diagnostics[0].source_span.start_line, 4)
        self.assertEqual(result.diagnostics[0].source_span.start_column, 3)
        self.assertEqual(len(result.diagnostics[0].related_spans), 1)
        self.assertEqual(
            result.diagnostics[0].related_spans[0].start_line,
            2,
        )
        self.assertNotIn("route_1", str(result.public_diagnostics))
        self.assertNotIn(
            "Private customer label",
            str(result.public_diagnostics),
        )
        self.assertNotIn("customer", str(result.public_diagnostics))

    def test_presentation_import_allows_non_executable_edge_ids(
        self,
    ) -> None:
        result = parse_mermaid_import(
            "flowchart LR\nA route.one@--> B route.one@--> C",
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/diagram.mmd",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_executable_import_rejects_bidirectional_edges(self) -> None:
        result = parse_mermaid_import(
            "\n".join(
                (
                    "flowchart LR",
                    "A private_route@<--> B",
                    "B public_route@--> C",
                )
            ),
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/diagram.mmd",
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            _codes(result),
            ("flow.mermaid.security.bidirectional_edge",),
        )
        self.assertEqual(result.diagnostics[0].source_span.start_line, 2)
        self.assertEqual(result.diagnostics[0].source_span.start_column, 3)
        self.assertNotIn("customer", str(result.public_diagnostics))
        self.assertNotIn("private_route", str(result.public_diagnostics))

    def test_presentation_import_keeps_bidirectional_edges_visual(
        self,
    ) -> None:
        result = parse_mermaid_import(
            "flowchart LR\nA private_route@<--> B",
            import_mode=FlowViewImportMode.PRESENTATION,
            source="/private/customer/diagram.mmd",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_spanless_hand_built_bidirectional_edge_is_ignored_by_security(
        self,
    ) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        parse_result = MermaidParseResult(
            cst=MermaidCst(),
            ast=MermaidAst(
                statements=(
                    MermaidAstEdgeStatement(
                        nodes=(
                            MermaidAstNode(id="A", source_span=span),
                            MermaidAstNode(id="B", source_span=span),
                        ),
                        edges=(
                            MermaidAstEdge(
                                source="A",
                                target="B",
                                arrow="<-->",
                            ),
                        ),
                        source_span=span,
                    ),
                ),
            ),
        )

        result = validate_mermaid_import(
            parse_result,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_malformed_empty_edge_id_marker_remains_parser_error(
        self,
    ) -> None:
        result = parse_mermaid_import(
            "flowchart LR\nA @--> B",
            import_mode=FlowViewImportMode.EXECUTABLE,
            source="/private/customer/diagram.mmd",
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            _codes(result),
            ("flow.mermaid.parser.malformed_edge_id",),
        )
        self.assertNotIn("customer", str(result.public_diagnostics))

    def test_spanless_hand_built_edge_id_is_ignored_by_security(
        self,
    ) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        parse_result = MermaidParseResult(
            cst=MermaidCst(),
            ast=MermaidAst(
                statements=(
                    MermaidAstEdgeStatement(
                        nodes=(
                            MermaidAstNode(id="A", source_span=span),
                            MermaidAstNode(id="B", source_span=span),
                        ),
                        edges=(
                            MermaidAstEdge(
                                source="A",
                                target="B",
                                arrow="-->",
                                explicit_id="route.one",
                            ),
                        ),
                        source_span=span,
                    ),
                ),
            ),
        )

        result = validate_mermaid_import(
            parse_result,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_import_validation_result_is_frozen_and_validated(self) -> None:
        parse_result = parse_mermaid("graph TD\nA --> B")
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.security.invalid",
            category=FlowDiagnosticCategory.MERMAID_SECURITY,
            source_span=FlowSourceSpan(start_line=1, start_column=1),
            severity=FlowDiagnosticSeverity.WARNING,
            message="Invalid Mermaid import.",
        )
        result = MermaidImportValidationResult(
            import_mode=FlowViewImportMode.PRESENTATION,
            parse_result=parse_result,
            diagnostics=(diagnostic,),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.public_diagnostics[0]["code"], diagnostic.code)
        with self.assertRaises(FrozenInstanceError):
            result.import_mode = FlowViewImportMode.EXECUTABLE  # type: ignore[misc]

    def test_import_validation_result_rejects_invalid_values(self) -> None:
        parse_result = parse_mermaid("graph TD\nA --> B")
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.security.invalid",
            category=FlowDiagnosticCategory.MERMAID_SECURITY,
            source_span=FlowSourceSpan(start_line=1, start_column=1),
            message="Invalid Mermaid import.",
        )
        invalid_cases = (
            {
                "import_mode": "presentation",
                "parse_result": parse_result,
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "parse_result": object(),
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "parse_result": parse_result,
                "diagnostics": [diagnostic],
            },
            {
                "import_mode": FlowViewImportMode.PRESENTATION,
                "parse_result": parse_result,
                "diagnostics": (object(),),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    MermaidImportValidationResult(**case)  # type: ignore[arg-type]

    def test_import_validation_functions_reject_invalid_arguments(
        self,
    ) -> None:
        parse_result = parse_mermaid("graph TD\nA --> B")

        with self.assertRaises(AssertionError):
            parse_mermaid_import(1, import_mode=FlowViewImportMode.EXECUTABLE)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_mermaid_import("graph TD", import_mode="executable")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_mermaid_import(
                "graph TD",
                import_mode=FlowViewImportMode.EXECUTABLE,
                source="",
            )
        with self.assertRaises(AssertionError):
            validate_mermaid_import(
                object(),  # type: ignore[arg-type]
                import_mode=FlowViewImportMode.EXECUTABLE,
            )
        with self.assertRaises(AssertionError):
            validate_mermaid_import(
                parse_result,
                import_mode="presentation",  # type: ignore[arg-type]
            )

    def test_validate_mermaid_import_preserves_existing_parse_result(
        self,
    ) -> None:
        parse_result = parse_mermaid("graph TD\nA --> B")

        result = validate_mermaid_import(
            parse_result,
            import_mode=FlowViewImportMode.EXECUTABLE,
        )

        self.assertIs(result.parse_result, parse_result)
        self.assertEqual(result.import_mode, FlowViewImportMode.EXECUTABLE)
        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_genuine_parser_errors_remain_errors_in_presentation_mode(
        self,
    ) -> None:
        result = parse_mermaid_import(
            'graph TD\nA["unterminated',
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.parser.unclosed_quoted_label",
        )

    def test_non_shorthand_parser_errors_stay_errors_in_presentation_mode(
        self,
    ) -> None:
        result = parse_mermaid_import(
            "graph TD\nA B",
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.parser.unsupported_statement",
        )

    def test_path_only_parser_errors_stay_errors_in_presentation_mode(
        self,
    ) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.unsupported_statement",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            path="diagram",
            message="Mermaid statement is not supported.",
        )
        parse_result = MermaidParseResult(
            cst=MermaidCst(),
            ast=MermaidAst(),
            diagnostics=(diagnostic,),
        )

        result = validate_mermaid_import(
            parse_result,
            import_mode=FlowViewImportMode.PRESENTATION,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics, (diagnostic,))


def _codes(result: MermaidImportValidationResult) -> tuple[str, ...]:
    return tuple(diagnostic.code for diagnostic in result.diagnostics)


def _diagnostic(
    result: MermaidImportValidationResult,
    code: str,
) -> FlowDiagnostic:
    matches = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == code
    ]
    assert matches
    return matches[0]


if __name__ == "__main__":
    main()
