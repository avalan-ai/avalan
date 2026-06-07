from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)

from dataclasses import dataclass, field
from enum import StrEnum


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
