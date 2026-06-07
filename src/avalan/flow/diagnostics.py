from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class FlowDiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FlowDiagnosticCategory(StrEnum):
    MERMAID_PARSER = "mermaid_parser"
    MERMAID_SECURITY = "mermaid_security"
    FLOW_VIEW_BINDING = "flow_view_binding"
    FLOW_DEFINITION_VALIDATION = "flow_definition_validation"
    EXECUTION = "execution"
    PRIVACY = "privacy"
    TASK_DURABILITY = "task_durability"


class FlowDiagnosticCodePrefix(StrEnum):
    MERMAID_PARSER = "flow.mermaid.parser"
    MERMAID_SECURITY = "flow.mermaid.security"
    FLOW_VIEW_BINDING = "flow.view.binding"
    FLOW_DEFINITION_VALIDATION = "flow.definition"
    EXECUTION = "flow.execution"
    PRIVACY = "flow.privacy"
    TASK_DURABILITY = "flow.task"


_FLOW_DIAGNOSTIC_CODE_PREFIXES = MappingProxyType(
    {
        FlowDiagnosticCategory.MERMAID_PARSER: (
            FlowDiagnosticCodePrefix.MERMAID_PARSER,
        ),
        FlowDiagnosticCategory.MERMAID_SECURITY: (
            FlowDiagnosticCodePrefix.MERMAID_SECURITY,
        ),
        FlowDiagnosticCategory.FLOW_VIEW_BINDING: (
            FlowDiagnosticCodePrefix.FLOW_VIEW_BINDING,
        ),
        FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION: (
            FlowDiagnosticCodePrefix.FLOW_DEFINITION_VALIDATION,
        ),
        FlowDiagnosticCategory.EXECUTION: (
            FlowDiagnosticCodePrefix.EXECUTION,
        ),
        FlowDiagnosticCategory.PRIVACY: (FlowDiagnosticCodePrefix.PRIVACY,),
        FlowDiagnosticCategory.TASK_DURABILITY: (
            FlowDiagnosticCodePrefix.TASK_DURABILITY,
        ),
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowSourceSpan:
    start_line: int
    start_column: int
    source: str | None = None
    end_line: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        assert self.start_line >= 1, "start_line must be positive"
        assert self.start_column >= 1, "start_column must be positive"
        if self.source is not None:
            assert self.source.strip(), "source must be non-empty"
        if self.end_line is not None:
            assert self.end_line >= self.start_line
        if self.end_column is not None:
            assert self.end_column >= 1

    def as_dict(self) -> dict[str, object]:
        value = self.as_public_dict()
        if self.source is not None:
            value["source"] = self.source
        return value

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "start_line": self.start_line,
            "start_column": self.start_column,
        }
        if self.end_line is not None:
            value["end_line"] = self.end_line
        if self.end_column is not None:
            value["end_column"] = self.end_column
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowDiagnostic:
    code: str
    message: str
    category: FlowDiagnosticCategory
    path: str | None = None
    source_span: FlowSourceSpan | None = None
    severity: FlowDiagnosticSeverity = FlowDiagnosticSeverity.ERROR
    hint: str | None = None
    related_spans: tuple[FlowSourceSpan, ...] = ()

    def __post_init__(self) -> None:
        assert self.code.strip(), "code must be non-empty"
        assert self.message.strip(), "message must be non-empty"
        assert isinstance(self.category, FlowDiagnosticCategory)
        assert isinstance(self.severity, FlowDiagnosticSeverity)
        assert (
            self.path is not None or self.source_span is not None
        ), "path or source_span is required"
        if self.path is not None:
            assert self.path.strip(), "path must be non-empty"
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)
        if self.hint is not None:
            assert self.hint.strip(), "hint must be non-empty"
        assert isinstance(self.related_spans, tuple)
        for span in self.related_spans:
            assert isinstance(span, FlowSourceSpan)

    def as_dict(self) -> dict[str, object]:
        return self._as_dict(public=False)

    def as_public_dict(self) -> dict[str, object]:
        return self._as_dict(public=True)

    def _as_dict(self, *, public: bool) -> dict[str, object]:
        value: dict[str, object] = {
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.path is not None:
            value["path"] = self.path
        if self.source_span is not None:
            value["source_span"] = (
                self.source_span.as_public_dict()
                if public
                else self.source_span.as_dict()
            )
        if self.hint is not None:
            value["hint"] = self.hint
        if self.related_spans:
            value["related_spans"] = tuple(
                span.as_public_dict() if public else span.as_dict()
                for span in self.related_spans
            )
        return value


def flow_diagnostic_code_prefixes(
    category: FlowDiagnosticCategory,
) -> tuple[FlowDiagnosticCodePrefix, ...]:
    assert isinstance(category, FlowDiagnosticCategory)
    return _FLOW_DIAGNOSTIC_CODE_PREFIXES[category]


def all_flow_diagnostic_code_prefixes() -> Mapping[
    FlowDiagnosticCategory,
    tuple[FlowDiagnosticCodePrefix, ...],
]:
    return _FLOW_DIAGNOSTIC_CODE_PREFIXES
