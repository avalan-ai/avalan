from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias


class MermaidTokenType(StrEnum):
    GRAPH = "graph"
    FLOWCHART = "flowchart"
    DIRECTION = "direction"
    IDENTIFIER = "identifier"
    QUOTED_LABEL = "quoted_label"
    MARKDOWN_LABEL = "markdown_label"
    SHAPE_DELIMITER = "shape_delimiter"
    ARROW = "arrow"
    EDGE_LABEL = "edge_label"
    SUBGRAPH = "subgraph"
    END = "end"
    CLASS_DIRECTIVE = "class_directive"
    STYLE_DIRECTIVE = "style_directive"
    LINK_STYLE_DIRECTIVE = "link_style_directive"
    UNSAFE_DIRECTIVE = "unsafe_directive"
    UNSUPPORTED_DIRECTIVE = "unsupported_directive"
    COMMA = "comma"
    SEMICOLON = "semicolon"
    COMMENT = "comment"
    WHITESPACE = "whitespace"


class MermaidDiagramKind(StrEnum):
    GRAPH = "graph"
    FLOWCHART = "flowchart"


class MermaidAstDirectiveKind(StrEnum):
    CLASS = "class"
    STYLE = "style"
    LINK_STYLE = "linkStyle"
    UNSAFE = "unsafe"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidToken:
    type: MermaidTokenType
    value: str
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.type, MermaidTokenType)
        assert isinstance(self.value, str)
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidTokenizationResult:
    tokens: tuple[MermaidToken, ...] = ()
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.tokens, tuple), "tokens must be a tuple"
        for token in self.tokens:
            assert isinstance(token, MermaidToken)
        assert isinstance(
            self.diagnostics, tuple
        ), "diagnostics must be a tuple"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidCstStatement:
    tokens: tuple[MermaidToken, ...]
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.tokens, tuple), "tokens must be a tuple"
        assert self.tokens, "tokens must be non-empty"
        for token in self.tokens:
            assert isinstance(token, MermaidToken)
        assert _significant_tokens(
            self.tokens
        ), "tokens must contain a significant token"
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidCst:
    tokens: tuple[MermaidToken, ...] = ()
    statements: tuple[MermaidCstStatement, ...] = ()
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.tokens, tuple), "tokens must be a tuple"
        for token in self.tokens:
            assert isinstance(token, MermaidToken)
        assert isinstance(self.statements, tuple), "statements must be a tuple"
        for statement in self.statements:
            assert isinstance(statement, MermaidCstStatement)
        assert isinstance(
            self.diagnostics,
            tuple,
        ), "diagnostics must be a tuple"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstNode:
    id: str
    label: str | None = None
    shape_tokens: tuple[str, ...] = ()
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        _assert_string_tuple(self.shape_tokens, "shape_tokens")
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstNodeStatement:
    node: MermaidAstNode
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.node, MermaidAstNode)
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstEdge:
    source: str
    target: str
    arrow: str
    label: str | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.source, "source")
        _assert_non_empty_string(self.target, "target")
        _assert_non_empty_string(self.arrow, "arrow")
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstEdgeStatement:
    nodes: tuple[MermaidAstNode, ...]
    edges: tuple[MermaidAstEdge, ...]
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.nodes, tuple), "nodes must be a tuple"
        assert len(self.nodes) >= 2, "edges require at least two nodes"
        for node in self.nodes:
            assert isinstance(node, MermaidAstNode)
        assert isinstance(self.edges, tuple), "edges must be a tuple"
        assert self.edges, "edges must be non-empty"
        for edge in self.edges:
            assert isinstance(edge, MermaidAstEdge)
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstDirective:
    kind: MermaidAstDirectiveKind
    arguments: tuple[str, ...] = ()
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.kind, MermaidAstDirectiveKind)
        _assert_string_tuple(self.arguments, "arguments")
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstComment:
    text: str
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        assert isinstance(self.text, str), "text must be a string"
        assert isinstance(self.source_span, FlowSourceSpan)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAstSubgraph:
    id: str
    label: str | None = None
    statements: tuple["MermaidAstStatement", ...] = ()
    source_span: FlowSourceSpan

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        _assert_ast_statements(self.statements, "statements")
        assert isinstance(self.source_span, FlowSourceSpan)


MermaidAstStatement: TypeAlias = (
    MermaidAstNodeStatement
    | MermaidAstEdgeStatement
    | MermaidAstDirective
    | MermaidAstComment
    | MermaidAstSubgraph
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidAst:
    diagram_kind: MermaidDiagramKind | None = None
    direction: str | None = None
    statements: tuple[MermaidAstStatement, ...] = ()
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        if self.diagram_kind is not None:
            assert isinstance(self.diagram_kind, MermaidDiagramKind)
        if self.direction is not None:
            _assert_non_empty_string(self.direction, "direction")
        _assert_ast_statements(self.statements, "statements")
        assert isinstance(
            self.diagnostics,
            tuple,
        ), "diagnostics must be a tuple"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)

    @property
    def ok(self) -> bool:
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidParseResult:
    cst: MermaidCst
    ast: MermaidAst
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.cst, MermaidCst)
        assert isinstance(self.ast, MermaidAst)
        assert isinstance(
            self.diagnostics,
            tuple,
        ), "diagnostics must be a tuple"
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


@dataclass(slots=True)
class _MermaidTokenizer:
    text: str
    source: str | None = None
    tokens: list[MermaidToken] = field(default_factory=list)
    diagnostics: list[FlowDiagnostic] = field(default_factory=list)
    index: int = 0
    line: int = 1
    column: int = 1

    def tokenize(self) -> MermaidTokenizationResult:
        while self.index < len(self.text):
            if self._consume_whitespace():
                continue
            if self._consume_comment():
                continue
            if self._consume_init_directive():
                continue
            if self._consume_frontmatter_delimiter():
                continue
            if self._consume_quoted_label():
                continue
            if self._consume_markdown_label():
                continue
            if self._consume_edge_label():
                continue
            if self._consume_arrow():
                continue
            if self._consume_single_character_token():
                continue
            if self._consume_identifier_or_directive():
                continue

            _, start_line, start_column = self._position()
            self._advance()
            self._diagnose(
                "flow.mermaid.parser.unrecognized_character",
                "Unrecognized Mermaid character.",
                start_line,
                start_column,
                self.line,
                self.column,
            )

        return MermaidTokenizationResult(
            tokens=tuple(self.tokens),
            diagnostics=tuple(self.diagnostics),
        )

    def _consume_whitespace(self) -> bool:
        if not self._peek().isspace():
            return False

        start_index, start_line, start_column = self._position()
        while self.index < len(self.text) and self._peek().isspace():
            self._advance()
        self._add_token(
            MermaidTokenType.WHITESPACE,
            start_index,
            start_line,
            start_column,
        )
        return True

    def _consume_comment(self) -> bool:
        if not self.text.startswith("%%", self.index):
            return False
        if self.text.startswith("%%{", self.index):
            return False

        start_index, start_line, start_column = self._position()
        while self.index < len(self.text) and self._peek() != "\n":
            self._advance()
        self._add_token(
            MermaidTokenType.COMMENT,
            start_index,
            start_line,
            start_column,
        )
        return True

    def _consume_init_directive(self) -> bool:
        if not self.text.startswith("%%{", self.index):
            return False

        start_index, start_line, start_column = self._position()
        end = self.text.find("}%%", self.index + 3)
        if end == -1:
            while self.index < len(self.text) and self._peek() != "\n":
                self._advance()
            self._add_token(
                MermaidTokenType.UNSAFE_DIRECTIVE,
                start_index,
                start_line,
                start_column,
            )
            self._diagnose(
                "flow.mermaid.parser.unclosed_directive",
                "Mermaid directive is not closed.",
                start_line,
                start_column,
                self.line,
                self.column,
            )
            return True

        while self.index < end + 3:
            self._advance()
        self._add_token(
            MermaidTokenType.UNSAFE_DIRECTIVE,
            start_index,
            start_line,
            start_column,
        )
        return True

    def _consume_frontmatter_delimiter(self) -> bool:
        if self._at_line_start() and self.text.startswith("---", self.index):
            start_index, start_line, start_column = self._position()
            while self.index < len(self.text) and self._peek() == "-":
                self._advance()
            self._add_token(
                MermaidTokenType.UNSAFE_DIRECTIVE,
                start_index,
                start_line,
                start_column,
            )
            return True
        return False

    def _consume_quoted_label(self) -> bool:
        if self._peek() != '"':
            return False

        start_index, start_line, start_column = self._position()
        self._advance()
        escaped = False
        while self.index < len(self.text):
            character = self._peek()
            self._advance()
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == '"':
                self._add_token(
                    MermaidTokenType.QUOTED_LABEL,
                    start_index,
                    start_line,
                    start_column,
                )
                return True

        self._add_token(
            MermaidTokenType.QUOTED_LABEL,
            start_index,
            start_line,
            start_column,
        )
        self._diagnose(
            "flow.mermaid.parser.unclosed_quoted_label",
            "Quoted Mermaid label is not closed.",
            start_line,
            start_column,
            self.line,
            self.column,
        )
        return True

    def _consume_markdown_label(self) -> bool:
        if self._peek() != "`":
            return False

        start_index, start_line, start_column = self._position()
        self._advance()
        while self.index < len(self.text):
            character = self._peek()
            self._advance()
            if character == "`":
                self._add_token(
                    MermaidTokenType.MARKDOWN_LABEL,
                    start_index,
                    start_line,
                    start_column,
                )
                return True

        self._add_token(
            MermaidTokenType.MARKDOWN_LABEL,
            start_index,
            start_line,
            start_column,
        )
        self._diagnose(
            "flow.mermaid.parser.unclosed_markdown_label",
            "Markdown Mermaid label is not closed.",
            start_line,
            start_column,
            self.line,
            self.column,
        )
        return True

    def _consume_edge_label(self) -> bool:
        if self._peek() != "|":
            return False

        start_index, start_line, start_column = self._position()
        self._advance()
        while self.index < len(self.text):
            character = self._peek()
            self._advance()
            if character == "|":
                self._add_token(
                    MermaidTokenType.EDGE_LABEL,
                    start_index,
                    start_line,
                    start_column,
                )
                return True
            if character == "\n":
                break

        self._add_token(
            MermaidTokenType.EDGE_LABEL,
            start_index,
            start_line,
            start_column,
        )
        self._diagnose(
            "flow.mermaid.parser.unclosed_edge_label",
            "Mermaid edge label is not closed.",
            start_line,
            start_column,
            self.line,
            self.column,
        )
        return True

    def _consume_arrow(self) -> bool:
        start_index, start_line, start_column = self._position()
        character = self._peek()
        if character == "<":
            self._advance()
        if self.index >= len(self.text):
            self._restore(start_index, start_line, start_column)
            return False

        shaft_start = self.index
        while self.index < len(self.text) and self._peek() in "-=.":
            self._advance()
        has_shaft = self.index > shaft_start
        if not has_shaft:
            self._restore(start_index, start_line, start_column)
            return False

        if self.index < len(self.text) and self._peek() in ">ox":
            self._advance()
        if self.index == shaft_start + 1 and self.text[shaft_start] == ".":
            self._restore(start_index, start_line, start_column)
            return False

        self._add_token(
            MermaidTokenType.ARROW,
            start_index,
            start_line,
            start_column,
        )
        return True

    def _consume_single_character_token(self) -> bool:
        token_type = {
            "(": MermaidTokenType.SHAPE_DELIMITER,
            ")": MermaidTokenType.SHAPE_DELIMITER,
            "[": MermaidTokenType.SHAPE_DELIMITER,
            "]": MermaidTokenType.SHAPE_DELIMITER,
            "{": MermaidTokenType.SHAPE_DELIMITER,
            "}": MermaidTokenType.SHAPE_DELIMITER,
            ",": MermaidTokenType.COMMA,
            ";": MermaidTokenType.SEMICOLON,
        }.get(self._peek())
        if token_type is None:
            return False

        start_index, start_line, start_column = self._position()
        self._advance()
        self._add_token(token_type, start_index, start_line, start_column)
        return True

    def _consume_identifier_or_directive(self) -> bool:
        if not _is_identifier_start(self._peek()):
            return False

        start_index, start_line, start_column = self._position()
        while self.index < len(self.text) and _is_identifier_part(
            self._peek()
        ):
            self._advance()
        value = self.text[start_index : self.index]
        self._add_token(
            _token_type_for_identifier(value, self._at_statement_start()),
            start_index,
            start_line,
            start_column,
        )
        return True

    def _add_token(
        self,
        token_type: MermaidTokenType,
        start_index: int,
        start_line: int,
        start_column: int,
    ) -> None:
        self.tokens.append(
            MermaidToken(
                type=token_type,
                value=self.text[start_index : self.index],
                source_span=self._span(
                    start_line,
                    start_column,
                    self.line,
                    self.column,
                ),
            )
        )

    def _diagnose(
        self,
        code: str,
        message: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> None:
        self.diagnostics.append(
            FlowDiagnostic(
                code=code,
                category=FlowDiagnosticCategory.MERMAID_PARSER,
                source_span=self._span(
                    start_line,
                    start_column,
                    end_line,
                    end_column,
                ),
                severity=FlowDiagnosticSeverity.ERROR,
                message=message,
                hint="Check Mermaid flowchart syntax near this span.",
            )
        )

    def _span(
        self,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> FlowSourceSpan:
        return FlowSourceSpan(
            source=self.source,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
        )

    def _advance(self) -> None:
        character = self.text[self.index]
        self.index += 1
        if character == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

    def _peek(self) -> str:
        return self.text[self.index]

    def _position(self) -> tuple[int, int, int]:
        return self.index, self.line, self.column

    def _restore(
        self,
        index: int,
        line: int,
        column: int,
    ) -> None:
        self.index = index
        self.line = line
        self.column = column

    def _at_line_start(self) -> bool:
        return self.column == 1

    def _at_statement_start(self) -> bool:
        for token in reversed(self.tokens):
            if token.type == MermaidTokenType.WHITESPACE:
                if "\n" in token.value:
                    return True
                continue
            return token.type == MermaidTokenType.SEMICOLON
        return True


@dataclass(slots=True)
class _MermaidAstParser:
    cst: MermaidCst
    diagnostics: list[FlowDiagnostic] = field(default_factory=list)
    index: int = 0
    diagram_kind: MermaidDiagramKind | None = None
    direction: str | None = None
    source_span: FlowSourceSpan | None = None

    def parse(self) -> MermaidAst:
        self.diagnostics.extend(self.cst.diagnostics)
        statements = self._parse_statements(stop_at_end=False)
        if self.diagram_kind is None and self.cst.statements:
            first = self.cst.statements[0]
            self._diagnose(
                "flow.mermaid.parser.missing_header",
                "Mermaid flowchart header is required.",
                first.source_span,
            )

        return MermaidAst(
            diagram_kind=self.diagram_kind,
            direction=self.direction,
            statements=statements,
            diagnostics=tuple(self.diagnostics),
            source_span=self.source_span,
        )

    def _parse_statements(
        self,
        *,
        stop_at_end: bool,
    ) -> tuple[MermaidAstStatement, ...]:
        parsed: list[MermaidAstStatement] = []
        while self.index < len(self.cst.statements):
            statement = self.cst.statements[self.index]
            significant = _significant_tokens(statement.tokens)
            first = significant[0]
            if first.type == MermaidTokenType.END:
                if stop_at_end:
                    self.index += 1
                    return tuple(parsed)
                self._diagnose(
                    "flow.mermaid.parser.unbalanced_subgraph",
                    "Subgraph end has no matching subgraph.",
                    first.source_span,
                )
                self.index += 1
                continue

            if first.type in (
                MermaidTokenType.GRAPH,
                MermaidTokenType.FLOWCHART,
            ):
                self._parse_header(statement, significant)
                self.index += 1
                continue

            if first.type == MermaidTokenType.SUBGRAPH:
                parsed.append(self._parse_subgraph(statement, significant))
                continue

            parsed_statement = self._parse_supported_statement(
                statement,
                significant,
            )
            if parsed_statement is not None:
                parsed.extend(parsed_statement)
            self.index += 1

        if stop_at_end:
            self._diagnose(
                "flow.mermaid.parser.unclosed_subgraph",
                "Subgraph is not closed.",
                self.cst.statements[self.index - 1].source_span,
            )
        return tuple(parsed)

    def _parse_header(
        self,
        statement: MermaidCstStatement,
        tokens: tuple[MermaidToken, ...],
    ) -> None:
        if self.diagram_kind is not None:
            self._diagnose(
                "flow.mermaid.parser.duplicate_header",
                "Mermaid flowchart header is already declared.",
                tokens[0].source_span,
            )
            return
        self.diagram_kind = (
            MermaidDiagramKind.GRAPH
            if tokens[0].type == MermaidTokenType.GRAPH
            else MermaidDiagramKind.FLOWCHART
        )
        self.source_span = statement.source_span
        if len(tokens) > 1 and tokens[1].type == MermaidTokenType.DIRECTION:
            self.direction = tokens[1].value

    def _parse_subgraph(
        self,
        statement: MermaidCstStatement,
        tokens: tuple[MermaidToken, ...],
    ) -> MermaidAstSubgraph:
        self.index += 1
        subgraph_id = ""
        label: str | None = None
        if len(tokens) > 1 and tokens[1].type == MermaidTokenType.IDENTIFIER:
            subgraph_id = tokens[1].value
            label = _label_from_tokens(tokens[2:])
        else:
            label = _label_from_tokens(tokens[1:])
            if label is not None:
                subgraph_id = label
        if not subgraph_id:
            subgraph_id = "__invalid_subgraph__"
            self._diagnose(
                "flow.mermaid.parser.missing_subgraph_id",
                "Subgraph declaration requires an identifier.",
                tokens[0].source_span,
            )

        children = self._parse_statements(stop_at_end=True)
        return MermaidAstSubgraph(
            id=subgraph_id,
            label=label,
            statements=children,
            source_span=statement.source_span,
        )

    def _parse_supported_statement(
        self,
        statement: MermaidCstStatement,
        tokens: tuple[MermaidToken, ...],
    ) -> tuple[MermaidAstStatement, ...] | None:
        first = tokens[0]
        if first.type == MermaidTokenType.COMMENT:
            return (
                MermaidAstComment(
                    text=first.value.removeprefix("%%").strip(),
                    source_span=statement.source_span,
                ),
            )
        if first.type == MermaidTokenType.CLASS_DIRECTIVE:
            return (
                self._directive(
                    MermaidAstDirectiveKind.CLASS, statement, tokens
                ),
            )
        if first.type == MermaidTokenType.STYLE_DIRECTIVE:
            return (
                self._directive(
                    MermaidAstDirectiveKind.STYLE, statement, tokens
                ),
            )
        if first.type == MermaidTokenType.LINK_STYLE_DIRECTIVE:
            return (
                self._directive(
                    MermaidAstDirectiveKind.LINK_STYLE,
                    statement,
                    tokens,
                ),
            )
        if first.type == MermaidTokenType.UNSAFE_DIRECTIVE:
            self._diagnose_unsupported(first.source_span)
            return (
                self._directive(
                    MermaidAstDirectiveKind.UNSAFE, statement, tokens
                ),
            )
        if first.type == MermaidTokenType.UNSUPPORTED_DIRECTIVE:
            self._diagnose_unsupported(first.source_span)
            return (
                self._directive(
                    MermaidAstDirectiveKind.UNSUPPORTED,
                    statement,
                    tokens,
                ),
            )
        if first.type != MermaidTokenType.IDENTIFIER:
            self._diagnose_unknown_statement(first.source_span)
            return None

        nodes, edges, position = self._parse_edge_statement(tokens)
        if edges and position == len(tokens):
            return (
                MermaidAstEdgeStatement(
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                    source_span=statement.source_span,
                ),
            )
        if len(nodes) == 1 and position == len(tokens):
            return (
                MermaidAstNodeStatement(
                    node=nodes[0],
                    source_span=statement.source_span,
                ),
            )

        self._diagnose_unknown_statement(first.source_span)
        return None

    def _parse_edge_statement(
        self,
        tokens: tuple[MermaidToken, ...],
    ) -> tuple[list[MermaidAstNode], list[MermaidAstEdge], int]:
        nodes: list[MermaidAstNode] = []
        edges: list[MermaidAstEdge] = []
        node, position = _parse_node(tokens, 0)
        assert node is not None
        nodes.append(node)
        while position < len(tokens):
            arrow_position = position
            arrow = tokens[position]
            if arrow.type != MermaidTokenType.ARROW:
                break
            position += 1
            label: str | None = None
            if (
                position < len(tokens)
                and tokens[position].type == MermaidTokenType.EDGE_LABEL
            ):
                label = _strip_label(tokens[position].value)
                position += 1
            target, next_position = _parse_node(tokens, position)
            if target is None:
                return nodes, edges, arrow_position
            assert target.source_span is not None
            nodes.append(target)
            edges.append(
                MermaidAstEdge(
                    source=nodes[-2].id,
                    target=target.id,
                    arrow=arrow.value,
                    label=label,
                    source_span=_combine_spans(
                        arrow.source_span, target.source_span
                    ),
                )
            )
            position = next_position
        return nodes, edges, position

    def _directive(
        self,
        kind: MermaidAstDirectiveKind,
        statement: MermaidCstStatement,
        tokens: tuple[MermaidToken, ...],
    ) -> MermaidAstDirective:
        return MermaidAstDirective(
            kind=kind,
            arguments=tuple(token.value for token in tokens[1:]),
            source_span=statement.source_span,
        )

    def _diagnose_unknown_statement(self, source_span: FlowSourceSpan) -> None:
        self._diagnose(
            "flow.mermaid.parser.unsupported_statement",
            "Mermaid statement is not supported.",
            source_span,
        )

    def _diagnose_unsupported(self, source_span: FlowSourceSpan) -> None:
        self._diagnose(
            "flow.mermaid.parser.unsupported_construct",
            "Unsupported Mermaid construct is not parsed.",
            source_span,
        )

    def _diagnose(
        self,
        code: str,
        message: str,
        source_span: FlowSourceSpan,
    ) -> None:
        self.diagnostics.append(
            FlowDiagnostic(
                code=code,
                category=FlowDiagnosticCategory.MERMAID_PARSER,
                source_span=source_span,
                severity=FlowDiagnosticSeverity.ERROR,
                message=message,
                hint="Check Mermaid flowchart syntax near this span.",
            )
        )


def parse_mermaid(
    text: str,
    *,
    source: str | None = None,
) -> MermaidParseResult:
    assert isinstance(text, str), "text must be a string"
    if source is not None:
        assert source.strip(), "source must be non-empty"
    tokenization = tokenize_mermaid(text, source=source)
    cst = parse_mermaid_tokens(tokenization.tokens, tokenization.diagnostics)
    ast = _MermaidAstParser(cst=cst).parse()
    diagnostics = tuple(
        (
            *cst.diagnostics,
            *ast.diagnostics[len(cst.diagnostics) :],
        )
    )
    return MermaidParseResult(cst=cst, ast=ast, diagnostics=diagnostics)


def parse_mermaid_tokens(
    tokens: tuple[MermaidToken, ...],
    diagnostics: tuple[FlowDiagnostic, ...] = (),
) -> MermaidCst:
    assert isinstance(tokens, tuple), "tokens must be a tuple"
    for token in tokens:
        assert isinstance(token, MermaidToken)
    assert isinstance(diagnostics, tuple), "diagnostics must be a tuple"
    for diagnostic in diagnostics:
        assert isinstance(diagnostic, FlowDiagnostic)
    return MermaidCst(
        tokens=tokens,
        statements=_split_cst_statements(tokens),
        diagnostics=diagnostics,
    )


def tokenize_mermaid(
    text: str,
    *,
    source: str | None = None,
) -> MermaidTokenizationResult:
    assert isinstance(text, str), "text must be a string"
    if source is not None:
        assert source.strip(), "source must be non-empty"
    return _MermaidTokenizer(text=text, source=source).tokenize()


def _is_identifier_start(value: str) -> bool:
    return value.isalnum() or value in "_#"


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


def _assert_ast_statements(
    values: tuple[MermaidAstStatement, ...],
    field_name: str,
) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        assert isinstance(
            value,
            (
                MermaidAstNodeStatement,
                MermaidAstEdgeStatement,
                MermaidAstDirective,
                MermaidAstComment,
                MermaidAstSubgraph,
            ),
        )


def _split_cst_statements(
    tokens: tuple[MermaidToken, ...],
) -> tuple[MermaidCstStatement, ...]:
    statements: list[MermaidCstStatement] = []
    current: list[MermaidToken] = []
    for token in tokens:
        current.append(token)
        if token.type == MermaidTokenType.SEMICOLON or (
            token.type == MermaidTokenType.WHITESPACE and "\n" in token.value
        ):
            _append_cst_statement(statements, current)
            current = []
    _append_cst_statement(statements, current)
    return tuple(statements)


def _append_cst_statement(
    statements: list[MermaidCstStatement],
    tokens: list[MermaidToken],
) -> None:
    if not any(
        token.type
        not in (
            MermaidTokenType.WHITESPACE,
            MermaidTokenType.SEMICOLON,
        )
        for token in tokens
    ):
        return
    source_span = _combine_spans(tokens[0].source_span, tokens[-1].source_span)
    assert source_span is not None
    statements.append(
        MermaidCstStatement(tokens=tuple(tokens), source_span=source_span)
    )


def _significant_tokens(
    tokens: tuple[MermaidToken, ...],
) -> tuple[MermaidToken, ...]:
    return tuple(
        token
        for token in tokens
        if token.type
        not in (MermaidTokenType.WHITESPACE, MermaidTokenType.SEMICOLON)
    )


def _parse_node(
    tokens: tuple[MermaidToken, ...],
    position: int,
) -> tuple[MermaidAstNode | None, int]:
    if (
        position >= len(tokens)
        or tokens[position].type != MermaidTokenType.IDENTIFIER
    ):
        return None, position

    identifier = tokens[position]
    position += 1
    label: str | None = None
    shape_tokens: list[str] = []
    node_tokens = [identifier]
    while position < len(tokens) and tokens[position].type in (
        MermaidTokenType.SHAPE_DELIMITER,
        MermaidTokenType.QUOTED_LABEL,
        MermaidTokenType.MARKDOWN_LABEL,
        MermaidTokenType.IDENTIFIER,
    ):
        token = tokens[position]
        if token.type == MermaidTokenType.SHAPE_DELIMITER:
            shape_tokens.append(token.value)
        elif token.type in (
            MermaidTokenType.QUOTED_LABEL,
            MermaidTokenType.MARKDOWN_LABEL,
        ):
            label = _strip_label(token.value)
        elif not shape_tokens:
            break
        elif label is None:
            label = token.value
        else:
            break
        node_tokens.append(token)
        position += 1

    source_span = _combine_spans(
        node_tokens[0].source_span,
        node_tokens[-1].source_span,
    )
    return (
        MermaidAstNode(
            id=identifier.value,
            label=label,
            shape_tokens=tuple(shape_tokens),
            source_span=source_span,
        ),
        position,
    )


def _label_from_tokens(tokens: tuple[MermaidToken, ...]) -> str | None:
    for token in tokens:
        if token.type in (
            MermaidTokenType.QUOTED_LABEL,
            MermaidTokenType.MARKDOWN_LABEL,
            MermaidTokenType.EDGE_LABEL,
        ):
            return _strip_label(token.value)
    shape_tokens = _significant_tokens(tokens)
    if len(shape_tokens) >= 3 and shape_tokens[0].value in ("[", "("):
        return "".join(token.value for token in shape_tokens[1:-1])
    return None


def _strip_label(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in '"`|':
        return value[1:-1]
    return value


def _combine_spans(
    start: FlowSourceSpan,
    end: FlowSourceSpan,
) -> FlowSourceSpan:
    return FlowSourceSpan(
        source=start.source,
        start_line=start.start_line,
        start_column=start.start_column,
        end_line=end.end_line,
        end_column=end.end_column,
    )


def _is_identifier_part(value: str) -> bool:
    return value.isalnum() or value in "_-.:/#"


def _token_type_for_identifier(
    value: str,
    at_statement_start: bool,
) -> MermaidTokenType:
    match value:
        case "graph":
            return MermaidTokenType.GRAPH
        case "flowchart":
            return MermaidTokenType.FLOWCHART
        case "TD" | "TB" | "BT" | "LR" | "RL":
            return MermaidTokenType.DIRECTION
        case "subgraph":
            return MermaidTokenType.SUBGRAPH
        case "end":
            return MermaidTokenType.END
        case "class":
            return MermaidTokenType.CLASS_DIRECTIVE
        case "style":
            return MermaidTokenType.STYLE_DIRECTIVE
        case "linkStyle":
            return MermaidTokenType.LINK_STYLE_DIRECTIVE
        case "click" | "link" | "href" | "callback":
            return MermaidTokenType.UNSAFE_DIRECTIVE
        case (
            "accTitle"
            | "accDescr"
            | "architecture"
            | "block"
            | "classDiagram"
            | "erDiagram"
            | "gantt"
            | "gitGraph"
            | "journey"
            | "mindmap"
            | "pie"
            | "requirementDiagram"
            | "sequenceDiagram"
            | "stateDiagram"
            | "timeline"
        ) if at_statement_start:
            return MermaidTokenType.UNSUPPORTED_DIRECTIVE
        case _:
            return MermaidTokenType.IDENTIFIER
