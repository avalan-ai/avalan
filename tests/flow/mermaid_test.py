from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
    MermaidToken,
    MermaidTokenizationResult,
    MermaidTokenType,
    tokenize_mermaid,
)


class MermaidTokenizerTestCase(TestCase):
    def test_tokenize_mermaid_recognizes_flowchart_surface(self) -> None:
        result = tokenize_mermaid(
            "\n".join(
                (
                    "flowchart LR",
                    '  A(["Start"]) -->|yes| B{`Check **risk**`};',
                    "  B -.-> C",
                    "  subgraph lane[Ops]",
                    "    class A active",
                    "    style B fill:#fff,stroke:#333",
                    "    linkStyle 0 stroke:#f00",
                    "  end",
                    "  %% keep this comment",
                )
            ),
            source="/private/customer/diagram.mmd",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        types = [token.type for token in result.tokens]
        self.assertIn(MermaidTokenType.FLOWCHART, types)
        self.assertIn(MermaidTokenType.DIRECTION, types)
        self.assertIn(MermaidTokenType.IDENTIFIER, types)
        self.assertIn(MermaidTokenType.SHAPE_DELIMITER, types)
        self.assertIn(MermaidTokenType.QUOTED_LABEL, types)
        self.assertIn(MermaidTokenType.MARKDOWN_LABEL, types)
        self.assertIn(MermaidTokenType.ARROW, types)
        self.assertIn(MermaidTokenType.EDGE_LABEL, types)
        self.assertIn(MermaidTokenType.SEMICOLON, types)
        self.assertIn(MermaidTokenType.SUBGRAPH, types)
        self.assertIn(MermaidTokenType.CLASS_DIRECTIVE, types)
        self.assertIn(MermaidTokenType.STYLE_DIRECTIVE, types)
        self.assertIn(MermaidTokenType.LINK_STYLE_DIRECTIVE, types)
        self.assertIn(MermaidTokenType.END, types)
        self.assertIn(MermaidTokenType.COMMENT, types)
        self.assertIn(MermaidTokenType.COMMA, types)
        self.assertIn(MermaidTokenType.WHITESPACE, types)
        self.assertEqual(result.tokens[0].type, MermaidTokenType.FLOWCHART)
        self.assertEqual(result.tokens[0].source_span.start_line, 1)
        self.assertEqual(result.tokens[0].source_span.start_column, 1)
        self.assertEqual(
            result.tokens[0].source_span.source,
            "/private/customer/diagram.mmd",
        )

    def test_quoted_labels_keep_arrows_comments_directives_and_pipes(
        self,
    ) -> None:
        result = tokenize_mermaid(
            'graph TD\nA["literal --> ; %% | click href"] --> B'
        )

        label = _single_token(result, MermaidTokenType.QUOTED_LABEL)
        self.assertTrue(result.ok)
        self.assertEqual(label.value, '"literal --> ; %% | click href"')
        self.assertNotIn(
            MermaidTokenType.UNSAFE_DIRECTIVE,
            [token.type for token in result.tokens],
        )
        self.assertEqual(
            [token.value for token in result.tokens if token.value == "-->"],
            ["-->"],
        )

    def test_quoted_labels_allow_escaped_quotes(self) -> None:
        result = tokenize_mermaid('graph TD\nA["say \\"go\\""]')

        self.assertTrue(result.ok)
        self.assertEqual(
            _single_token(result, MermaidTokenType.QUOTED_LABEL).value,
            '"say \\"go\\""',
        )

    def test_tokenize_mermaid_recognizes_graph_header_and_edge_variants(
        self,
    ) -> None:
        result = tokenize_mermaid("graph TB\nA <==> B\nB --- C\nC --x D")

        self.assertTrue(result.ok)
        self.assertEqual(
            [
                token.value
                for token in result.tokens
                if token.type == MermaidTokenType.ARROW
            ],
            ["<==>", "---", "--x"],
        )

    def test_tokenize_mermaid_recognizes_unsafe_and_unsupported_directives(
        self,
    ) -> None:
        result = tokenize_mermaid(
            "\n".join(
                (
                    "---",
                    "title: unsafe",
                    "---",
                    "%%{init: {'theme': 'dark'}}%%",
                    "sequenceDiagram",
                    "click A call callback()",
                    "link A https://example.test",
                )
            )
        )

        self.assertTrue(result.ok)
        self.assertGreaterEqual(
            _count_tokens(result, MermaidTokenType.UNSAFE_DIRECTIVE),
            5,
        )
        self.assertEqual(
            _single_token(
                result, MermaidTokenType.UNSUPPORTED_DIRECTIVE
            ).value,
            "sequenceDiagram",
        )

    def test_tokenize_mermaid_reports_unclosed_directive(self) -> None:
        result = tokenize_mermaid("%%{init: {'theme': 'dark'}")

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.parser.unclosed_directive",
        )
        self.assertEqual(
            result.public_diagnostics,
            (
                {
                    "code": "flow.mermaid.parser.unclosed_directive",
                    "category": "mermaid_parser",
                    "severity": "error",
                    "message": "Mermaid directive is not closed.",
                    "source_span": {
                        "start_line": 1,
                        "start_column": 1,
                        "end_line": 1,
                        "end_column": 27,
                    },
                    "hint": "Check Mermaid flowchart syntax near this span.",
                },
            ),
        )

    def test_tokenize_mermaid_reports_unclosed_labels(self) -> None:
        cases = (
            (
                'graph TD\nA["unterminated',
                "flow.mermaid.parser.unclosed_quoted_label",
            ),
            (
                "graph TD\nA[`unterminated",
                "flow.mermaid.parser.unclosed_markdown_label",
            ),
            (
                "graph TD\nA -->|unterminated\nB",
                "flow.mermaid.parser.unclosed_edge_label",
            ),
        )

        for text, code in cases:
            with self.subTest(code=code):
                result = tokenize_mermaid(text)

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)

    def test_tokenize_mermaid_reports_unrecognized_character(self) -> None:
        result = tokenize_mermaid("graph TD\nA @ B")

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.parser.unrecognized_character",
        )
        self.assertEqual(result.diagnostics[0].source_span.start_line, 2)
        self.assertEqual(result.diagnostics[0].source_span.start_column, 3)

    def test_tokenize_mermaid_reports_incomplete_arrow_fragments(
        self,
    ) -> None:
        cases = ("<", "graph TD\nA . B")

        for text in cases:
            with self.subTest(text=text):
                result = tokenize_mermaid(text)

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[0].code,
                    "flow.mermaid.parser.unrecognized_character",
                )

    def test_tokenize_mermaid_accepts_empty_input(self) -> None:
        result = tokenize_mermaid("")

        self.assertTrue(result.ok)
        self.assertEqual(result.tokens, ())
        self.assertEqual(result.diagnostics, ())

    def test_tokenize_mermaid_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            tokenize_mermaid(1)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            tokenize_mermaid("graph TD", source="")

    def test_tokenization_entities_are_frozen_and_validated(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.invalid",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=span,
            severity=FlowDiagnosticSeverity.ERROR,
            message="Invalid Mermaid.",
        )
        token = MermaidToken(
            type=MermaidTokenType.IDENTIFIER,
            value="A",
            source_span=span,
        )
        result = MermaidTokenizationResult(
            tokens=(token,),
            diagnostics=(diagnostic,),
        )
        warning_result = MermaidTokenizationResult(
            diagnostics=(
                FlowDiagnostic(
                    code="flow.mermaid.parser.warning",
                    category=FlowDiagnosticCategory.MERMAID_PARSER,
                    source_span=span,
                    severity=FlowDiagnosticSeverity.WARNING,
                    message="Mermaid warning.",
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertTrue(warning_result.ok)
        self.assertEqual(result.public_diagnostics[0]["code"], diagnostic.code)
        with self.assertRaises(FrozenInstanceError):
            token.value = "B"  # type: ignore[misc]

    def test_tokenization_entities_reject_invalid_values(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        invalid_token_cases = (
            {
                "type": "identifier",
                "value": "A",
                "source_span": span,
            },
            {
                "type": MermaidTokenType.IDENTIFIER,
                "value": 1,
                "source_span": span,
            },
            {
                "type": MermaidTokenType.IDENTIFIER,
                "value": "A",
                "source_span": object(),
            },
        )
        invalid_result_cases = (
            {
                "tokens": [
                    MermaidToken(
                        type=MermaidTokenType.IDENTIFIER,
                        value="A",
                        source_span=span,
                    )
                ]
            },
            {"tokens": (object(),)},
            {
                "diagnostics": [
                    FlowDiagnostic(
                        code="flow.mermaid.parser.invalid",
                        category=FlowDiagnosticCategory.MERMAID_PARSER,
                        source_span=span,
                        message="Invalid Mermaid.",
                    )
                ]
            },
            {"diagnostics": (object(),)},
        )

        for case in invalid_token_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    MermaidToken(**case)  # type: ignore[arg-type]
        for case in invalid_result_cases:
            with self.subTest(case=case):
                with self.assertRaises(AssertionError):
                    MermaidTokenizationResult(**case)  # type: ignore[arg-type]


def _single_token(
    result: MermaidTokenizationResult,
    token_type: MermaidTokenType,
) -> MermaidToken:
    tokens = [token for token in result.tokens if token.type == token_type]
    assert len(tokens) == 1
    return tokens[0]


def _count_tokens(
    result: MermaidTokenizationResult,
    token_type: MermaidTokenType,
) -> int:
    return len([token for token in result.tokens if token.type == token_type])


if __name__ == "__main__":
    main()
