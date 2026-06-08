from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
    MermaidAst,
    MermaidAstComment,
    MermaidAstDirective,
    MermaidAstDirectiveKind,
    MermaidAstEdge,
    MermaidAstEdgeStatement,
    MermaidAstNode,
    MermaidAstNodeStatement,
    MermaidAstSubgraph,
    MermaidCst,
    MermaidCstStatement,
    MermaidDiagramKind,
    MermaidParseResult,
    MermaidToken,
    MermaidTokenizationResult,
    MermaidTokenType,
    parse_mermaid,
    parse_mermaid_tokens,
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
            (
                "graph TD\nA -->|unterminated",
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


class MermaidParserTestCase(TestCase):
    def test_parse_mermaid_builds_cst_and_ast_for_supported_constructs(
        self,
    ) -> None:
        result = parse_mermaid(
            "\n".join(
                (
                    "flowchart LR",
                    '  A(["Start"]) -->|yes| B{`Check`};',
                    "  B -.-> C --> D",
                    "  subgraph lane[Ops]",
                    "    C --> D",
                    "    class C active",
                    "    style D fill:#fff,stroke:#333",
                    "    linkStyle 0 stroke:#f00",
                    "    %% inner note",
                    "  end",
                )
            ),
            source="/private/customer/diagram.mmd",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.ast.diagram_kind, MermaidDiagramKind.FLOWCHART)
        self.assertEqual(result.ast.direction, "LR")
        self.assertGreaterEqual(len(result.cst.statements), 9)
        first_edge = result.ast.statements[0]
        self.assertIsInstance(first_edge, MermaidAstEdgeStatement)
        assert isinstance(first_edge, MermaidAstEdgeStatement)
        self.assertEqual([node.id for node in first_edge.nodes], ["A", "B"])
        self.assertEqual(first_edge.nodes[0].label, "Start")
        self.assertEqual(
            first_edge.nodes[0].shape_tokens, ("(", "[", "]", ")")
        )
        self.assertEqual(first_edge.edges[0].label, "yes")
        chained = result.ast.statements[1]
        self.assertIsInstance(chained, MermaidAstEdgeStatement)
        assert isinstance(chained, MermaidAstEdgeStatement)
        self.assertEqual(
            [(edge.source, edge.target) for edge in chained.edges],
            [("B", "C"), ("C", "D")],
        )
        subgraph = result.ast.statements[2]
        self.assertIsInstance(subgraph, MermaidAstSubgraph)
        assert isinstance(subgraph, MermaidAstSubgraph)
        self.assertEqual(subgraph.id, "lane")
        self.assertEqual(subgraph.label, "Ops")
        self.assertEqual(
            [type(statement) for statement in subgraph.statements],
            [
                MermaidAstEdgeStatement,
                MermaidAstDirective,
                MermaidAstDirective,
                MermaidAstDirective,
                MermaidAstComment,
            ],
        )
        self.assertEqual(result.public_diagnostics, ())

    def test_parse_mermaid_supports_node_only_and_comments(self) -> None:
        result = parse_mermaid("graph TD\n%% keep\nA[`Only node`]")

        self.assertTrue(result.ok)
        self.assertEqual(result.ast.diagram_kind, MermaidDiagramKind.GRAPH)
        self.assertIsInstance(result.ast.statements[0], MermaidAstComment)
        self.assertIsInstance(
            result.ast.statements[1], MermaidAstNodeStatement
        )
        node_statement = result.ast.statements[1]
        assert isinstance(node_statement, MermaidAstNodeStatement)
        self.assertEqual(node_statement.node.id, "A")
        self.assertEqual(node_statement.node.label, "Only node")

    def test_parse_mermaid_preserves_unsupported_construct_diagnostics(
        self,
    ) -> None:
        result = parse_mermaid(
            "\n".join(
                (
                    "sequenceDiagram",
                    "click A call callback()",
                )
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.mermaid.parser.unsupported_construct",
                "flow.mermaid.parser.unsupported_construct",
                "flow.mermaid.parser.missing_header",
            ],
        )
        self.assertEqual(
            [statement.kind for statement in result.ast.statements],
            [
                MermaidAstDirectiveKind.UNSUPPORTED,
                MermaidAstDirectiveKind.UNSAFE,
            ],
        )

    def test_parse_mermaid_reports_unbalanced_subgraphs(self) -> None:
        cases = (
            ("graph TD\nend", "flow.mermaid.parser.unbalanced_subgraph"),
            (
                "graph TD\nsubgraph lane\nA --> B",
                "flow.mermaid.parser.unclosed_subgraph",
            ),
            (
                "graph TD\nsubgraph\nA --> B\nend",
                "flow.mermaid.parser.missing_subgraph_id",
            ),
            ('graph TD\nsubgraph "Ops"\nA --> B\nend', None),
        )

        for text, code in cases:
            with self.subTest(code=code):
                result = parse_mermaid(text)

                if code is None:
                    self.assertTrue(result.ok)
                    subgraph = result.ast.statements[0]
                    self.assertIsInstance(subgraph, MermaidAstSubgraph)
                    assert isinstance(subgraph, MermaidAstSubgraph)
                    self.assertEqual(subgraph.id, "Ops")
                    continue
                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_parse_mermaid_reports_unclosed_label_diagnostics(self) -> None:
        result = parse_mermaid('graph TD\nA["unterminated')

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            "flow.mermaid.parser.unclosed_quoted_label",
        )
        self.assertEqual(result.cst.diagnostics, result.diagnostics[:1])

    def test_parse_mermaid_reports_unknown_statements(self) -> None:
        cases = (
            "graph TD\nA B",
            "graph TD\nTD",
            "graph TD\nA -->",
            "graph TD\nA[One Two]",
            "graph TD\ngraph LR",
        )

        for text in cases:
            with self.subTest(text=text):
                result = parse_mermaid(text)

                self.assertFalse(result.ok)
                self.assertIn(
                    result.diagnostics[0].code,
                    (
                        "flow.mermaid.parser.unsupported_statement",
                        "flow.mermaid.parser.duplicate_header",
                    ),
                )

    def test_parse_mermaid_accepts_empty_input(self) -> None:
        result = parse_mermaid("")

        self.assertTrue(result.ok)
        self.assertEqual(result.cst.tokens, ())
        self.assertEqual(result.cst.statements, ())
        self.assertEqual(result.ast.statements, ())
        self.assertIsNone(result.ast.diagram_kind)

    def test_parse_mermaid_tokens_builds_cst_without_tokenizing_text(
        self,
    ) -> None:
        tokenization = tokenize_mermaid("graph TD\nA --> B")
        cst = parse_mermaid_tokens(tokenization.tokens)

        self.assertTrue(cst.ok)
        self.assertEqual(len(cst.statements), 2)
        self.assertEqual(
            _significant_values(cst.statements[1]),
            ["A", "-->", "B"],
        )

    def test_parse_mermaid_rejects_invalid_arguments(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        token = MermaidToken(
            type=MermaidTokenType.IDENTIFIER,
            value="A",
            source_span=span,
        )

        with self.assertRaises(AssertionError):
            parse_mermaid(1)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_mermaid("graph TD", source="")
        with self.assertRaises(AssertionError):
            parse_mermaid_tokens([token])  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_mermaid_tokens((object(),))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            parse_mermaid_tokens((token,), diagnostics=[])  # type: ignore[arg-type]

    def test_parser_entities_are_frozen_and_validated(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        diagnostic = FlowDiagnostic(
            code="flow.mermaid.parser.invalid",
            category=FlowDiagnosticCategory.MERMAID_PARSER,
            source_span=span,
            message="Invalid Mermaid.",
        )
        token = MermaidToken(
            type=MermaidTokenType.IDENTIFIER,
            value="A",
            source_span=span,
        )
        cst_statement = MermaidCstStatement(tokens=(token,), source_span=span)
        cst = MermaidCst(
            tokens=(token,),
            statements=(cst_statement,),
            diagnostics=(diagnostic,),
        )
        node = MermaidAstNode(id="A", source_span=span)
        node_without_span = MermaidAstNode(id="C")
        node_statement = MermaidAstNodeStatement(node=node, source_span=span)
        edge = MermaidAstEdge(
            source="A",
            target="B",
            arrow="-->",
            source_span=span,
        )
        edge_without_span = MermaidAstEdge(
            source="B",
            target="C",
            arrow="-->",
        )
        edge_statement = MermaidAstEdgeStatement(
            nodes=(node, MermaidAstNode(id="B", source_span=span)),
            edges=(edge,),
            source_span=span,
        )
        directive = MermaidAstDirective(
            kind=MermaidAstDirectiveKind.CLASS,
            arguments=("A", "active"),
            source_span=span,
        )
        comment = MermaidAstComment(text="note", source_span=span)
        subgraph = MermaidAstSubgraph(
            id="lane",
            statements=(edge_statement, directive, comment),
            source_span=span,
        )
        ast = MermaidAst(
            diagram_kind=MermaidDiagramKind.GRAPH,
            direction="TD",
            statements=(node_statement, subgraph),
            diagnostics=(diagnostic,),
            source_span=span,
        )
        result = MermaidParseResult(
            cst=cst,
            ast=ast,
            diagnostics=(diagnostic,),
        )

        self.assertFalse(cst.ok)
        self.assertFalse(ast.ok)
        self.assertFalse(result.ok)
        self.assertEqual(cst.public_diagnostics[0]["code"], diagnostic.code)
        self.assertEqual(ast.public_diagnostics[0]["code"], diagnostic.code)
        self.assertEqual(result.public_diagnostics[0]["code"], diagnostic.code)
        self.assertIsNone(node_without_span.source_span)
        self.assertIsNone(edge_without_span.source_span)
        with self.assertRaises(FrozenInstanceError):
            node.id = "B"  # type: ignore[misc]

    def test_parser_entities_reject_invalid_values(self) -> None:
        span = FlowSourceSpan(start_line=1, start_column=1)
        token = MermaidToken(
            type=MermaidTokenType.IDENTIFIER,
            value="A",
            source_span=span,
        )
        whitespace = MermaidToken(
            type=MermaidTokenType.WHITESPACE,
            value=" ",
            source_span=span,
        )
        node = MermaidAstNode(id="A", source_span=span)
        edge = MermaidAstEdge(
            source="A",
            target="B",
            arrow="-->",
            source_span=span,
        )
        cst = MermaidCst()
        ast = MermaidAst()
        invalid_cases = (
            (MermaidCstStatement, {"tokens": (), "source_span": span}),
            (MermaidCstStatement, {"tokens": [token], "source_span": span}),
            (
                MermaidCstStatement,
                {"tokens": (whitespace,), "source_span": span},
            ),
            (MermaidCst, {"tokens": [token]}),
            (MermaidCst, {"statements": (object(),)}),
            (MermaidAstNode, {"id": ""}),
            (MermaidAstNode, {"id": "A", "label": ""}),
            (MermaidAstNode, {"id": "A", "shape_tokens": ["["]}),
            (MermaidAstNodeStatement, {"node": object(), "source_span": span}),
            (MermaidAstEdge, {"source": "", "target": "B", "arrow": "-->"}),
            (MermaidAstEdge, {"source": "A", "target": "", "arrow": "-->"}),
            (MermaidAstEdge, {"source": "A", "target": "B", "arrow": ""}),
            (
                MermaidAstEdge,
                {"source": "A", "target": "B", "arrow": "-->", "label": ""},
            ),
            (
                MermaidAstEdgeStatement,
                {"nodes": (node,), "edges": (edge,), "source_span": span},
            ),
            (
                MermaidAstEdgeStatement,
                {
                    "nodes": (node, object()),
                    "edges": (edge,),
                    "source_span": span,
                },
            ),
            (
                MermaidAstEdgeStatement,
                {"nodes": (node, node), "edges": (), "source_span": span},
            ),
            (MermaidAstDirective, {"kind": "class", "source_span": span}),
            (
                MermaidAstDirective,
                {
                    "kind": MermaidAstDirectiveKind.CLASS,
                    "arguments": ["A"],
                    "source_span": span,
                },
            ),
            (MermaidAstComment, {"text": 1, "source_span": span}),
            (MermaidAstSubgraph, {"id": "", "source_span": span}),
            (
                MermaidAstSubgraph,
                {"id": "lane", "label": "", "source_span": span},
            ),
            (
                MermaidAstSubgraph,
                {"id": "lane", "statements": (object(),), "source_span": span},
            ),
            (MermaidAst, {"diagram_kind": "graph"}),
            (MermaidAst, {"direction": ""}),
            (MermaidAst, {"statements": (object(),)}),
            (MermaidAst, {"diagnostics": [object()]}),
            (MermaidParseResult, {"cst": object(), "ast": ast}),
            (MermaidParseResult, {"cst": cst, "ast": object()}),
            (
                MermaidParseResult,
                {"cst": cst, "ast": ast, "diagnostics": [object()]},
            ),
        )

        for cls, kwargs in invalid_cases:
            with self.subTest(cls=cls, kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    cls(**kwargs)  # type: ignore[operator]


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


def _significant_values(statement: MermaidCstStatement) -> list[str]:
    return [
        token.value
        for token in statement.tokens
        if token.type
        not in (MermaidTokenType.WHITESPACE, MermaidTokenType.SEMICOLON)
    ]


if __name__ == "__main__":
    main()
