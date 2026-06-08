from .definition import FlowDefinition
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)
from .view import (
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
    FlowViewStyleProperties,
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
    CLASS_DEF_DIRECTIVE = "class_def_directive"
    CLASS_DIRECTIVE = "class_directive"
    STYLE_DIRECTIVE = "style_directive"
    LINK_STYLE_DIRECTIVE = "link_style_directive"
    UNSAFE_DIRECTIVE = "unsafe_directive"
    UNSUPPORTED_DIRECTIVE = "unsupported_directive"
    SHORTHAND_SEPARATOR = "shorthand_separator"
    COMMA = "comma"
    SEMICOLON = "semicolon"
    COMMENT = "comment"
    WHITESPACE = "whitespace"


class MermaidDiagramKind(StrEnum):
    GRAPH = "graph"
    FLOWCHART = "flowchart"


class MermaidAstDirectiveKind(StrEnum):
    CLASS_DEF = "classDef"
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


@dataclass(frozen=True, slots=True, kw_only=True)
class MermaidImportValidationResult:
    import_mode: FlowViewImportMode
    parse_result: MermaidParseResult
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.import_mode, FlowViewImportMode)
        assert isinstance(self.parse_result, MermaidParseResult)
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
class MermaidFlowViewNormalizationResult:
    import_validation: MermaidImportValidationResult
    view: FlowView
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(
            self.import_validation,
            MermaidImportValidationResult,
        )
        assert isinstance(self.view, FlowView)
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
class MermaidRenderResult:
    source: str
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.source, str), "source must be a string"
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
            "&": MermaidTokenType.SHORTHAND_SEPARATOR,
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
        if first.type == MermaidTokenType.CLASS_DEF_DIRECTIVE:
            return (
                self._directive(
                    MermaidAstDirectiveKind.CLASS_DEF, statement, tokens
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
        source_group, position = _parse_node_group(tokens, 0)
        assert source_group
        nodes.extend(source_group)
        current_sources = source_group
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
            target_group, next_position = _parse_node_group(tokens, position)
            if not target_group:
                return nodes, edges, arrow_position
            nodes.extend(target_group)
            for source in current_sources:
                for target in target_group:
                    assert target.source_span is not None
                    edges.append(
                        MermaidAstEdge(
                            source=source.id,
                            target=target.id,
                            arrow=arrow.value,
                            label=label,
                            source_span=_combine_spans(
                                arrow.source_span,
                                target.source_span,
                            ),
                        )
                    )
            current_sources = target_group
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


def parse_mermaid_import(
    text: str,
    *,
    import_mode: FlowViewImportMode,
    source: str | None = None,
) -> MermaidImportValidationResult:
    assert isinstance(text, str), "text must be a string"
    assert isinstance(import_mode, FlowViewImportMode)
    if source is not None:
        assert source.strip(), "source must be non-empty"
    return validate_mermaid_import(
        parse_mermaid(text, source=source),
        import_mode=import_mode,
    )


def normalize_mermaid_flow_view(
    text: str,
    *,
    import_mode: FlowViewImportMode,
    source: str | None = None,
) -> MermaidFlowViewNormalizationResult:
    assert isinstance(text, str), "text must be a string"
    assert isinstance(import_mode, FlowViewImportMode)
    if source is not None:
        assert source.strip(), "source must be non-empty"
    return normalize_mermaid_import_to_flow_view(
        parse_mermaid_import(
            text,
            import_mode=import_mode,
            source=source,
        )
    )


def normalize_mermaid_import_to_flow_view(
    import_validation: MermaidImportValidationResult,
) -> MermaidFlowViewNormalizationResult:
    assert isinstance(import_validation, MermaidImportValidationResult)
    view = _MermaidFlowViewNormalizer(
        import_validation=import_validation,
    ).normalize()
    return MermaidFlowViewNormalizationResult(
        import_validation=import_validation,
        view=view,
        diagnostics=view.diagnostics,
    )


def flow_definition_to_flow_view(
    definition: FlowDefinition,
    *,
    import_mode: FlowViewImportMode = FlowViewImportMode.EXECUTABLE,
) -> FlowView:
    assert isinstance(definition, FlowDefinition)
    assert isinstance(import_mode, FlowViewImportMode)
    edge_ids: set[str] = set()
    return FlowView(
        import_mode=import_mode,
        nodes=tuple(
            FlowViewNode(id=node.name, implicit=False)
            for node in definition.nodes
        ),
        edges=tuple(
            FlowViewEdge(
                id=_next_rendered_edge_id(
                    edge.source,
                    edge.target,
                    edge_ids,
                ),
                source=edge.source,
                target=edge.target,
                label=edge.label,
            )
            for edge in definition.edges
        ),
        metadata={
            "format": {
                "name": "mermaid",
                "diagram_kind": MermaidDiagramKind.FLOWCHART.value,
            },
            "source": "flow_definition",
        },
    )


def render_flow_definition_mermaid(
    definition: FlowDefinition,
) -> MermaidRenderResult:
    assert isinstance(definition, FlowDefinition)
    return render_mermaid_view(flow_definition_to_flow_view(definition))


def render_mermaid_view(view: FlowView) -> MermaidRenderResult:
    assert isinstance(view, FlowView)
    return _MermaidFlowViewRenderer(view=view).render()


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


def validate_mermaid_import(
    parse_result: MermaidParseResult,
    *,
    import_mode: FlowViewImportMode,
) -> MermaidImportValidationResult:
    assert isinstance(parse_result, MermaidParseResult)
    assert isinstance(import_mode, FlowViewImportMode)
    diagnostics = (
        *_import_parser_diagnostics(parse_result, import_mode),
        *_import_security_diagnostics(parse_result, import_mode),
    )
    return MermaidImportValidationResult(
        import_mode=import_mode,
        parse_result=parse_result,
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


@dataclass(slots=True)
class _FlowViewNodeDraft:
    id: str
    label: str | None = None
    shape: FlowViewNodeShape = FlowViewNodeShape.RECTANGLE
    source_span: FlowSourceSpan | None = None
    implicit: bool = True
    group: str | None = None
    classes: list[str] = field(default_factory=list)
    style: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def merge(
        self,
        node: MermaidAstNode,
        *,
        group: str | None,
        implicit: bool,
    ) -> None:
        shape = _shape_from_tokens(node.shape_tokens)
        if node.label is not None:
            self.label = node.label
        if shape != FlowViewNodeShape.RECTANGLE:
            self.shape = shape
        if self.source_span is None or not implicit:
            self.source_span = node.source_span
        self.implicit = self.implicit and implicit
        if self.group is None:
            self.group = group
        if node.shape_tokens:
            self.metadata["mermaid"] = {"shape_tokens": node.shape_tokens}

    def to_node(self) -> FlowViewNode:
        return FlowViewNode(
            id=self.id,
            label=self.label,
            shape=self.shape,
            classes=tuple(self.classes),
            style=self.style,
            metadata=self.metadata,
            source_span=self.source_span,
            implicit=self.implicit,
            group=self.group,
        )


@dataclass(slots=True)
class _FlowViewGroupDraft:
    id: str
    label: str | None = None
    parent: str | None = None
    source_span: FlowSourceSpan | None = None
    nodes: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    style: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_group(self) -> FlowViewGroup:
        return FlowViewGroup(
            id=self.id,
            label=self.label,
            parent=self.parent,
            nodes=tuple(self.nodes),
            groups=tuple(self.groups),
            classes=tuple(self.classes),
            style=self.style,
            metadata=self.metadata,
            source_span=self.source_span,
        )


@dataclass(slots=True)
class _MermaidFlowViewNormalizer:
    import_validation: MermaidImportValidationResult
    nodes: dict[str, _FlowViewNodeDraft] = field(default_factory=dict)
    edges: list[FlowViewEdge] = field(default_factory=list)
    edge_ids: set[str] = field(default_factory=set)
    groups: dict[str, _FlowViewGroupDraft] = field(default_factory=dict)
    class_definitions: list[FlowViewClassDefinition] = field(
        default_factory=list
    )
    styles: list[FlowViewStyle] = field(default_factory=list)
    link_styles: list[FlowViewLinkStyle] = field(default_factory=list)
    comments: list[FlowViewComment] = field(default_factory=list)

    def normalize(self) -> FlowView:
        ast = self.import_validation.parse_result.ast
        self._visit_statements(ast.statements, group=None)
        return FlowView(
            import_mode=self.import_validation.import_mode,
            direction=_view_direction(ast.direction),
            nodes=tuple(draft.to_node() for draft in self.nodes.values()),
            edges=tuple(self.edges),
            groups=tuple(draft.to_group() for draft in self.groups.values()),
            class_definitions=tuple(self.class_definitions),
            styles=tuple(self.styles),
            link_styles=tuple(self.link_styles),
            comments=tuple(self.comments),
            diagnostics=self.import_validation.diagnostics,
            metadata={
                "format": {
                    "name": "mermaid",
                    "diagram_kind": (
                        ast.diagram_kind.value if ast.diagram_kind else None
                    ),
                }
            },
            source_span=ast.source_span,
        )

    def _visit_statements(
        self,
        statements: tuple[MermaidAstStatement, ...],
        *,
        group: str | None,
    ) -> None:
        for statement in statements:
            if isinstance(statement, MermaidAstNodeStatement):
                self._record_node(statement.node, group=group, implicit=False)
            elif isinstance(statement, MermaidAstEdgeStatement):
                self._record_edge_statement(statement, group=group)
            elif isinstance(statement, MermaidAstSubgraph):
                self._record_group(statement, parent=group)
            elif isinstance(statement, MermaidAstDirective):
                self._record_directive(statement)
            else:
                assert isinstance(statement, MermaidAstComment)
                self.comments.append(
                    FlowViewComment(
                        text=statement.text,
                        source_span=statement.source_span,
                    )
                )

    def _record_edge_statement(
        self,
        statement: MermaidAstEdgeStatement,
        *,
        group: str | None,
    ) -> None:
        for node in statement.nodes:
            self._record_node(
                node,
                group=group,
                implicit=not node.label and not node.shape_tokens,
            )
        for edge in statement.edges:
            self._record_edge(edge)

    def _record_group(
        self,
        statement: MermaidAstSubgraph,
        *,
        parent: str | None,
    ) -> None:
        draft = self.groups.get(statement.id)
        if draft is None:
            draft = _FlowViewGroupDraft(
                id=statement.id,
                label=statement.label,
                parent=parent,
                source_span=statement.source_span,
            )
            self.groups[statement.id] = draft
        else:
            if statement.label is not None:
                draft.label = statement.label
            if draft.parent is None:
                draft.parent = parent
        if parent is not None:
            parent_draft = self.groups[parent]
            _append_unique(parent_draft.groups, statement.id)
        self._visit_statements(statement.statements, group=statement.id)

    def _record_directive(self, statement: MermaidAstDirective) -> None:
        match statement.kind:
            case MermaidAstDirectiveKind.CLASS_DEF:
                self._record_class_definition(statement)
            case MermaidAstDirectiveKind.CLASS:
                self._record_class(statement)
            case MermaidAstDirectiveKind.STYLE:
                self._record_style(statement)
            case MermaidAstDirectiveKind.LINK_STYLE:
                self._record_link_style(statement)
            case _:
                return

    def _record_class_definition(
        self,
        statement: MermaidAstDirective,
    ) -> None:
        if len(statement.arguments) < 2:
            return
        self.class_definitions.append(
            FlowViewClassDefinition(
                name=statement.arguments[0],
                properties=_style_properties(statement.arguments[1:]),
                source_span=statement.source_span,
            )
        )

    def _record_class(self, statement: MermaidAstDirective) -> None:
        targets = _class_targets(statement.arguments)
        classes = _class_names(statement.arguments)
        for target in targets:
            node = self.nodes.get(target)
            if node is not None:
                for class_name in classes:
                    _append_unique(node.classes, class_name)

    def _record_style(self, statement: MermaidAstDirective) -> None:
        if len(statement.arguments) < 2:
            return
        target = statement.arguments[0]
        properties = _style_properties(statement.arguments[1:])
        self.styles.append(
            FlowViewStyle(
                target=target,
                properties=properties,
                source_span=statement.source_span,
            )
        )
        node = self.nodes.get(target)
        if node is not None:
            node.style.update(properties)

    def _record_link_style(self, statement: MermaidAstDirective) -> None:
        if len(statement.arguments) < 2:
            return
        target = statement.arguments[0]
        properties = _style_properties(statement.arguments[1:])
        if target.isdigit():
            self.link_styles.append(
                FlowViewLinkStyle(
                    edge_index=int(target),
                    properties=properties,
                    source_span=statement.source_span,
                )
            )
            return
        self.link_styles.append(
            FlowViewLinkStyle(
                edge=target,
                properties=properties,
                source_span=statement.source_span,
            )
        )

    def _record_node(
        self,
        node: MermaidAstNode,
        *,
        group: str | None,
        implicit: bool,
    ) -> None:
        draft = self.nodes.get(node.id)
        if draft is None:
            draft = _FlowViewNodeDraft(id=node.id)
            self.nodes[node.id] = draft
        draft.merge(node, group=group, implicit=implicit)
        if group is not None:
            _append_unique(self.groups[group].nodes, node.id)

    def _record_edge(self, edge: MermaidAstEdge) -> None:
        edge_id = self._next_edge_id(edge.source, edge.target)
        self.edges.append(
            FlowViewEdge(
                id=edge_id,
                source=edge.source,
                target=edge.target,
                label=edge.label,
                style=_edge_style(edge.arrow),
                metadata={"mermaid": {"arrow": edge.arrow}},
                source_span=edge.source_span,
                bidirectional=_is_bidirectional_arrow(edge.arrow),
            )
        )

    def _next_edge_id(self, source: str, target: str) -> str:
        base = f"{_id_fragment(source)}_to_{_id_fragment(target)}"
        candidate = base
        index = 2
        while candidate in self.edge_ids:
            candidate = f"{base}_{index}"
            index += 1
        self.edge_ids.add(candidate)
        return candidate


@dataclass(slots=True)
class _MermaidFlowViewRenderer:
    view: FlowView
    diagnostics: list[FlowDiagnostic] = field(default_factory=list)

    def render(self) -> MermaidRenderResult:
        lines: list[str] = [self._header()]
        rendered_nodes: set[str] = set()
        root_groups = tuple(
            group for group in self.view.groups if group.parent is None
        )
        for group in root_groups:
            self._append_group(lines, group, rendered_nodes, indent="  ")
        for node in self.view.nodes:
            if node.id not in rendered_nodes:
                self._append_node(lines, node, indent="  ")
                rendered_nodes.add(node.id)
        for edge in self.view.edges:
            self._append_edge(lines, edge)
        self._append_class_definitions(lines)
        self._append_node_classes(lines)
        self._append_styles(lines)
        self._append_link_styles(lines)
        self._append_comments(lines)
        return MermaidRenderResult(
            source="\n".join(lines) + "\n",
            diagnostics=tuple(self.diagnostics),
        )

    def _header(self) -> str:
        direction = self.view.direction or FlowViewDirection.TD
        return f"flowchart {direction.value}"

    def _append_group(
        self,
        lines: list[str],
        group: FlowViewGroup,
        rendered_nodes: set[str],
        *,
        indent: str,
    ) -> None:
        if not _is_safe_mermaid_identifier(group.id):
            self._diagnose(
                "invalid_group_identifier",
                "Flow View group identifier cannot be rendered safely.",
                "view.groups",
            )
            return

        lines.append(
            f"{indent}subgraph {group.id}{_render_group_label(group)}"
        )
        if group.direction is not None:
            self._diagnose(
                "unsupported_group_direction",
                "Flow View group direction is not rendered.",
                "view.groups.direction",
                severity=FlowDiagnosticSeverity.WARNING,
            )
        child_indent = f"{indent}  "
        for child_group_id in group.groups:
            child = self.view.group_map.get(child_group_id)
            if child is not None:
                self._append_group(
                    lines,
                    child,
                    rendered_nodes,
                    indent=child_indent,
                )
        for node_id in group.nodes:
            node = self.view.node_map.get(node_id)
            if node is not None and node.id not in rendered_nodes:
                self._append_node(lines, node, indent=child_indent)
                rendered_nodes.add(node.id)
        lines.append(f"{indent}end")

    def _append_node(
        self,
        lines: list[str],
        node: FlowViewNode,
        *,
        indent: str,
    ) -> None:
        if not _is_safe_mermaid_identifier(node.id):
            self._diagnose(
                "invalid_node_identifier",
                "Flow View node identifier cannot be rendered safely.",
                "view.nodes",
            )
            return
        lines.append(f"{indent}{node.id}{_render_node_label(node)}")

    def _append_edge(self, lines: list[str], edge: FlowViewEdge) -> None:
        if not _is_safe_mermaid_identifier(edge.source):
            self._diagnose(
                "invalid_edge_source",
                "Flow View edge source cannot be rendered safely.",
                "view.edges.source",
            )
            return
        if not _is_safe_mermaid_identifier(edge.target):
            self._diagnose(
                "invalid_edge_target",
                "Flow View edge target cannot be rendered safely.",
                "view.edges.target",
            )
            return
        arrow = _render_edge_arrow(edge)
        label = (
            f"|{_escape_edge_label(edge.label)}|"
            if edge.label is not None
            else ""
        )
        lines.append(f"  {edge.source} {arrow}{label} {edge.target}")

    def _append_class_definitions(self, lines: list[str]) -> None:
        for class_definition in self.view.class_definitions:
            if not _is_safe_mermaid_identifier(class_definition.name):
                self._diagnose(
                    "invalid_class_identifier",
                    "Flow View class identifier cannot be rendered safely.",
                    "view.class_definitions",
                )
                continue
            properties = _render_properties(
                class_definition.properties,
                path=f"view.class_definitions.{class_definition.name}",
                diagnostics=self.diagnostics,
            )
            if properties:
                lines.append(
                    f"  classDef {class_definition.name} {properties}"
                )

    def _append_node_classes(self, lines: list[str]) -> None:
        for node in self.view.nodes:
            if not node.classes:
                continue
            if not _is_safe_mermaid_identifier(node.id):
                continue
            class_names = tuple(
                class_name
                for class_name in node.classes
                if _is_safe_mermaid_identifier(class_name)
            )
            if len(class_names) != len(node.classes):
                self._diagnose(
                    "invalid_class_identifier",
                    "Flow View class identifier cannot be rendered safely.",
                    "view.nodes.classes",
                )
            if class_names:
                lines.append(f"  class {node.id} {','.join(class_names)}")

    def _append_styles(self, lines: list[str]) -> None:
        for style in self.view.styles:
            if not _is_safe_mermaid_identifier(style.target):
                self._diagnose(
                    "invalid_style_target",
                    "Flow View style target cannot be rendered safely.",
                    "view.styles",
                )
                continue
            properties = _render_properties(
                style.properties,
                path=f"view.styles.{style.target}",
                diagnostics=self.diagnostics,
            )
            if properties:
                lines.append(f"  style {style.target} {properties}")

    def _append_link_styles(self, lines: list[str]) -> None:
        for link_style in self.view.link_styles:
            target = (
                str(link_style.edge_index)
                if link_style.edge_index is not None
                else link_style.edge
            )
            assert target is not None
            if not (target.isdigit() or _is_safe_mermaid_identifier(target)):
                self._diagnose(
                    "invalid_link_style_target",
                    "Flow View link style target cannot be rendered safely.",
                    "view.link_styles",
                )
                continue
            properties = _render_properties(
                link_style.properties,
                path="view.link_styles",
                diagnostics=self.diagnostics,
            )
            if properties:
                lines.append(f"  linkStyle {target} {properties}")

    def _append_comments(self, lines: list[str]) -> None:
        for comment in self.view.comments:
            lines.append(f"  %% {_escape_comment(comment.text)}")

    def _diagnose(
        self,
        code_suffix: str,
        message: str,
        path: str,
        *,
        severity: FlowDiagnosticSeverity = FlowDiagnosticSeverity.ERROR,
    ) -> None:
        self.diagnostics.append(
            FlowDiagnostic(
                code=f"flow.mermaid.parser.renderer_{code_suffix}",
                category=FlowDiagnosticCategory.MERMAID_PARSER,
                path=path,
                severity=severity,
                message=message,
                hint="Use Mermaid-safe identifiers and labels.",
            )
        )


def _import_parser_diagnostics(
    parse_result: MermaidParseResult,
    import_mode: FlowViewImportMode,
) -> tuple[FlowDiagnostic, ...]:
    unsupported_diagram = _has_token_type(
        parse_result,
        MermaidTokenType.UNSUPPORTED_DIRECTIVE,
    )
    diagnostics: list[FlowDiagnostic] = []
    for diagnostic in parse_result.diagnostics:
        if diagnostic.code == "flow.mermaid.parser.unsupported_construct":
            continue
        if (
            unsupported_diagram
            and diagnostic.code == "flow.mermaid.parser.missing_header"
        ):
            continue
        if (
            import_mode == FlowViewImportMode.PRESENTATION
            and diagnostic.code == "flow.mermaid.parser.unsupported_statement"
            and _diagnostic_statement_has_shorthand(parse_result, diagnostic)
        ):
            continue
        diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _import_security_diagnostics(
    parse_result: MermaidParseResult,
    import_mode: FlowViewImportMode,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for statement in parse_result.cst.statements:
        diagnostics.extend(
            _statement_security_diagnostics(statement, import_mode)
        )
    diagnostics.extend(
        _label_security_diagnostics(
            parse_result.ast.statements,
            import_mode,
        )
    )
    return tuple(diagnostics)


def _statement_security_diagnostics(
    statement: MermaidCstStatement,
    import_mode: FlowViewImportMode,
) -> tuple[FlowDiagnostic, ...]:
    tokens = _significant_tokens(statement.tokens)
    diagnostics: list[FlowDiagnostic] = []
    for token in tokens:
        if token.type == MermaidTokenType.SHORTHAND_SEPARATOR:
            diagnostics.append(
                _import_diagnostic(
                    "ambiguous_shorthand",
                    "Ambiguous Mermaid shorthand is not supported for import.",
                    token.source_span,
                    import_mode,
                )
            )
        if token.type == MermaidTokenType.UNSUPPORTED_DIRECTIVE:
            diagnostics.append(
                _import_diagnostic(
                    "unsupported_diagram_type",
                    "Unsupported Mermaid diagram type is not supported.",
                    token.source_span,
                    import_mode,
                )
            )
        if token.type == MermaidTokenType.UNSAFE_DIRECTIVE:
            code, message = _unsafe_directive_diagnostic(token.value)
            diagnostics.append(
                _import_diagnostic(
                    code,
                    message,
                    token.source_span,
                    import_mode,
                )
            )
        if _is_external_link_token(token.value):
            diagnostics.append(
                _import_diagnostic(
                    "unsafe_external_link",
                    "External Mermaid links are not activated during import.",
                    token.source_span,
                    import_mode,
                )
            )

    if tokens and _directive_is_malformed(tokens):
        diagnostics.append(
            _import_diagnostic(
                "malformed_directive",
                "Mermaid presentation directive is malformed.",
                tokens[0].source_span,
                import_mode,
            )
        )
    return tuple(diagnostics)


def _import_diagnostic(
    code_suffix: str,
    message: str,
    source_span: FlowSourceSpan,
    import_mode: FlowViewImportMode,
) -> FlowDiagnostic:
    severity = (
        FlowDiagnosticSeverity.ERROR
        if import_mode == FlowViewImportMode.EXECUTABLE
        else FlowDiagnosticSeverity.WARNING
    )
    return FlowDiagnostic(
        code=f"flow.mermaid.security.{code_suffix}",
        category=FlowDiagnosticCategory.MERMAID_SECURITY,
        source_span=source_span,
        severity=severity,
        message=message,
        hint="Use a structured flow definition for executable behavior.",
    )


def _label_security_diagnostics(
    statements: tuple[MermaidAstStatement, ...],
    import_mode: FlowViewImportMode,
) -> tuple[FlowDiagnostic, ...]:
    diagnostics: list[FlowDiagnostic] = []
    for statement in statements:
        for node in _statement_nodes(statement):
            diagnostics.extend(
                _label_diagnostics(node.label, node.source_span, import_mode)
            )
        if isinstance(statement, MermaidAstEdgeStatement):
            for edge in statement.edges:
                diagnostics.extend(
                    _label_diagnostics(
                        edge.label,
                        edge.source_span,
                        import_mode,
                    )
                )
        if isinstance(statement, MermaidAstSubgraph):
            diagnostics.extend(
                _label_diagnostics(
                    statement.label,
                    statement.source_span,
                    import_mode,
                )
            )
            diagnostics.extend(
                _label_security_diagnostics(
                    statement.statements,
                    import_mode,
                )
            )
    return tuple(diagnostics)


def _statement_nodes(
    statement: MermaidAstStatement,
) -> tuple[MermaidAstNode, ...]:
    if isinstance(statement, MermaidAstNodeStatement):
        return (statement.node,)
    if isinstance(statement, MermaidAstEdgeStatement):
        return statement.nodes
    return ()


def _label_diagnostics(
    label: str | None,
    source_span: FlowSourceSpan | None,
    import_mode: FlowViewImportMode,
) -> tuple[FlowDiagnostic, ...]:
    if label is None or source_span is None:
        return ()
    if _is_script_like_label(label):
        return (
            _import_diagnostic(
                "script_like_label",
                "Mermaid label contains embedded executable content.",
                source_span,
                import_mode,
            ),
        )
    if _is_html_label(label):
        return (
            _import_diagnostic(
                "html_label",
                "Mermaid HTML labels are not supported for import.",
                source_span,
                import_mode,
            ),
        )
    return ()


def _unsafe_directive_diagnostic(value: str) -> tuple[str, str]:
    stripped = value.strip()
    if stripped == "---":
        return "frontmatter", "Mermaid frontmatter is not supported."
    if stripped.startswith("%%{init"):
        return "init_directive", "Mermaid init directives are not supported."
    if stripped.startswith("%%{"):
        return "unknown_directive", "Mermaid directive is not supported."
    if stripped == "callback":
        return (
            "unsafe_callback_directive",
            "Mermaid callback directives are not activated during import.",
        )
    return (
        "unsafe_link_directive",
        "Mermaid link directives are not activated during import.",
    )


def _directive_is_malformed(tokens: tuple[MermaidToken, ...]) -> bool:
    first = tokens[0]
    if first.type in (
        MermaidTokenType.CLASS_DEF_DIRECTIVE,
        MermaidTokenType.CLASS_DIRECTIVE,
        MermaidTokenType.STYLE_DIRECTIVE,
        MermaidTokenType.LINK_STYLE_DIRECTIVE,
    ):
        return len(tokens) < 3
    return False


def _is_html_label(value: str) -> bool:
    stripped = value.strip().lower()
    return (
        stripped.startswith("<")
        and stripped.endswith(">")
        and any(character.isalpha() for character in stripped[1:-1])
    )


def _is_script_like_label(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "<script",
            "javascript:",
            "onerror=",
            "onload=",
            "{{",
            "{%",
        )
    )


def _is_external_link_token(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "mailto:"))


def _has_token_type(
    parse_result: MermaidParseResult,
    token_type: MermaidTokenType,
) -> bool:
    return any(token.type == token_type for token in parse_result.cst.tokens)


def _diagnostic_statement_has_shorthand(
    parse_result: MermaidParseResult,
    diagnostic: FlowDiagnostic,
) -> bool:
    if diagnostic.source_span is None:
        return False
    for statement in parse_result.cst.statements:
        if (
            statement.source_span.start_line
            <= diagnostic.source_span.start_line
            <= (
                statement.source_span.end_line
                or statement.source_span.start_line
            )
            and any(
                token.type == MermaidTokenType.SHORTHAND_SEPARATOR
                for token in statement.tokens
            )
        ):
            return True
    return False


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


def _parse_node_group(
    tokens: tuple[MermaidToken, ...],
    position: int,
) -> tuple[tuple[MermaidAstNode, ...], int]:
    nodes: list[MermaidAstNode] = []
    node, next_position = _parse_node(tokens, position)
    if node is None:
        return (), position
    nodes.append(node)
    position = next_position
    while (
        position < len(tokens)
        and tokens[position].type == MermaidTokenType.SHORTHAND_SEPARATOR
    ):
        node, next_position = _parse_node(tokens, position + 1)
        if node is None:
            return tuple(nodes), position
        nodes.append(node)
        position = next_position
    return tuple(nodes), position


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
        stripped = value[1:-1]
        if value[0] == '"':
            return stripped.replace('\\"', '"').replace("\\\\", "\\")
        return stripped
    return value


def _view_direction(value: str | None) -> FlowViewDirection | None:
    if value is None:
        return None
    return FlowViewDirection(value)


def _shape_from_tokens(
    tokens: tuple[str, ...],
) -> FlowViewNodeShape:
    match tokens:
        case ("(", "[", "]", ")"):
            return FlowViewNodeShape.STADIUM
        case ("[", "[", "]", "]"):
            return FlowViewNodeShape.SUBROUTINE
        case ("[", "(", ")", "]"):
            return FlowViewNodeShape.CYLINDER
        case ("(", "(", ")", ")"):
            return FlowViewNodeShape.CIRCLE
        case ("(", "(", "(", ")", ")", ")"):
            return FlowViewNodeShape.DOUBLE_CIRCLE
        case ("{", "{", "}", "}"):
            return FlowViewNodeShape.HEXAGON
        case ("(", ")"):
            return FlowViewNodeShape.ROUNDED
        case ("{", "}"):
            return FlowViewNodeShape.DIAMOND
        case _:
            return FlowViewNodeShape.RECTANGLE


def _edge_style(value: str) -> FlowViewEdgeStyle:
    if "=" in value:
        return FlowViewEdgeStyle.THICK
    if "." in value:
        return FlowViewEdgeStyle.DOTTED
    return FlowViewEdgeStyle.SOLID


def _is_bidirectional_arrow(value: str) -> bool:
    return value.startswith("<") and value.endswith(">")


def _style_properties(arguments: tuple[str, ...]) -> dict[str, str]:
    properties: dict[str, str] = {}
    for pair in "".join(arguments).split(","):
        key, separator, value = pair.partition(":")
        if separator and key.strip() and value.strip():
            properties[key.strip()] = value.strip()
    return properties


def _class_targets(arguments: tuple[str, ...]) -> tuple[str, ...]:
    if len(arguments) < 2:
        return ()
    return tuple(argument for argument in arguments[:-1] if argument != ",")


def _class_names(arguments: tuple[str, ...]) -> tuple[str, ...]:
    if len(arguments) < 2:
        return ()
    return tuple(
        name.strip() for name in arguments[-1].split(",") if name.strip()
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _render_group_label(group: FlowViewGroup) -> str:
    if group.label is None:
        return ""
    return f'["{_escape_quoted_label(group.label)}"]'


def _render_node_label(node: FlowViewNode) -> str:
    if node.label is None and node.shape == FlowViewNodeShape.RECTANGLE:
        return ""
    label = _escape_quoted_label(node.label or node.id)
    prefix, suffix = _shape_delimiters(node.shape)
    return f'{prefix}"{label}"{suffix}'


def _render_edge_arrow(edge: FlowViewEdge) -> str:
    if edge.bidirectional:
        match edge.style:
            case FlowViewEdgeStyle.THICK:
                return "<==>"
            case FlowViewEdgeStyle.DOTTED:
                return "<-.->"
            case _:
                return "<-->"
    match edge.style:
        case FlowViewEdgeStyle.THICK:
            return "==>"
        case FlowViewEdgeStyle.DOTTED:
            return "-.->"
        case _:
            return "-->"


def _render_properties(
    properties: FlowViewStyleProperties,
    *,
    path: str,
    diagnostics: list[FlowDiagnostic],
) -> str:
    rendered: list[str] = []
    for key, value in properties.items():
        if _is_safe_style_key(key) and _is_safe_style_value(value):
            rendered.append(f"{key}:{value}")
        else:
            diagnostics.append(
                FlowDiagnostic(
                    code="flow.mermaid.parser.renderer_unsafe_style_property",
                    category=FlowDiagnosticCategory.MERMAID_PARSER,
                    path=path,
                    severity=FlowDiagnosticSeverity.WARNING,
                    message=(
                        "Flow View style property is not rendered because "
                        "it is not Mermaid-safe."
                    ),
                    hint="Use plain CSS presentation values without links.",
                )
            )
    return ",".join(rendered)


def _shape_delimiters(shape: FlowViewNodeShape) -> tuple[str, str]:
    match shape:
        case FlowViewNodeShape.STADIUM:
            return "([", "])"
        case FlowViewNodeShape.SUBROUTINE:
            return "[[", "]]"
        case FlowViewNodeShape.CYLINDER:
            return "[(", ")]"
        case FlowViewNodeShape.CIRCLE:
            return "((", "))"
        case FlowViewNodeShape.DOUBLE_CIRCLE:
            return "(((", ")))"
        case FlowViewNodeShape.HEXAGON:
            return "{{", "}}"
        case FlowViewNodeShape.ROUNDED:
            return "(", ")"
        case FlowViewNodeShape.DIAMOND:
            return "{", "}"
        case _:
            return "[", "]"


def _escape_quoted_label(value: str) -> str:
    return _escape_label_text(value).replace("\\", "\\\\").replace('"', '\\"')


def _escape_edge_label(value: str) -> str:
    return _escape_label_text(value).replace("|", "&#124;")


def _escape_comment(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def _escape_label_text(value: str) -> str:
    escaped = " ".join(value.split())
    replacements = {
        "<": "&lt;",
        ">": "&gt;",
        "javascript:": "javascript&#58;",
        "Javascript:": "Javascript&#58;",
        "JAVASCRIPT:": "JAVASCRIPT&#58;",
        "onerror=": "onerror&#61;",
        "onload=": "onload&#61;",
        "{{": "{ {",
        "{%": "{ %",
    }
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _is_safe_mermaid_identifier(value: str) -> bool:
    if not value or not _is_identifier_start(value[0]):
        return False
    if any(not _is_identifier_part(character) for character in value):
        return False
    return (
        _token_type_for_identifier(value, True) == MermaidTokenType.IDENTIFIER
        and _token_type_for_identifier(value, False)
        == MermaidTokenType.IDENTIFIER
    )


def _is_safe_style_key(value: str) -> bool:
    return bool(value) and all(
        character.isalnum() or character in "-_" for character in value
    )


def _is_safe_style_value(value: str) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in ("url(", "http:", "https:")):
        return False
    if any(character in value for character in "\r\n;{}<>`"):
        return False
    return bool(value.strip())


def _next_rendered_edge_id(
    source: str,
    target: str,
    edge_ids: set[str],
) -> str:
    base = f"{_id_fragment(source)}_to_{_id_fragment(target)}"
    candidate = base
    index = 2
    while candidate in edge_ids:
        candidate = f"{base}_{index}"
        index += 1
    edge_ids.add(candidate)
    return candidate


def _id_fragment(value: str) -> str:
    fragment = "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")
    return fragment or "node"


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
        case "classDef":
            return MermaidTokenType.CLASS_DEF_DIRECTIVE
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
