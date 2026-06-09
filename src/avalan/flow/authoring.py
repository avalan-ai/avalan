from ..filesystem import DEFAULT_TEXT_ENCODING, assert_text_encoding, read_text
from .binding import (
    FlowViewBindingResult,
    bind_flow_view_definition,
    create_flow_definition_skeleton,
)
from .definition import FlowDefinition
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
)
from .graph import (
    FlowGraphDiagnosticCode,
    FlowGraphInspection,
    flow_graph_diagnostic,
)
from .loader import FlowDefinitionLoader
from .mermaid import (
    MermaidFlowViewNormalizationResult,
    MermaidRenderResult,
    normalize_mermaid_flow_view,
    render_mermaid_view,
)
from .registry import FlowNodeRegistry
from .serializer import serialize_flow_definition
from .view import FlowView, FlowViewImportMode

from dataclasses import dataclass
from os import strerror
from pathlib import Path


def _has_error(diagnostics: tuple[FlowDiagnostic, ...]) -> bool:
    return any(
        diagnostic.severity == FlowDiagnosticSeverity.ERROR
        for diagnostic in diagnostics
    )


def _combine_diagnostics(
    diagnostics: tuple[FlowDiagnostic, ...],
    inspection: FlowGraphInspection | None,
) -> tuple[FlowDiagnostic, ...]:
    if inspection is None:
        return diagnostics
    combined = list(diagnostics)
    seen = {_diagnostic_identity(diagnostic) for diagnostic in combined}
    for diagnostic in inspection.diagnostics:
        identity = _diagnostic_identity(diagnostic)
        if identity in seen:
            continue
        combined.append(diagnostic)
        seen.add(identity)
    return tuple(combined)


def _diagnostic_identity(
    diagnostic: FlowDiagnostic,
) -> tuple[tuple[str, str], ...]:
    assert isinstance(diagnostic, FlowDiagnostic)
    return tuple(
        (key, repr(value))
        for key, value in sorted(diagnostic.as_public_dict().items())
    )


def _file_read_diagnostic(exc: OSError | UnicodeDecodeError) -> FlowDiagnostic:
    if isinstance(exc, OSError):
        message = strerror(exc.errno) if exc.errno else "Unable to read file."
    else:
        message = str(exc) or "Unable to read file."
    return FlowDiagnostic(
        code="file.read",
        path="file",
        category=FlowDiagnosticCategory.PRIVACY,
        severity=FlowDiagnosticSeverity.ERROR,
        message=message,
        hint="Use a readable local file.",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowDefinitionSkeletonResult:
    definition: FlowDefinition
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.definition, FlowDefinition)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)

    @property
    def ok(self) -> bool:
        return not _has_error(self.diagnostics)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowDefinitionCompileResult:
    definition: FlowDefinition | None = None
    canonical_source: str | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    authoring_graph: bool = False
    graph_inspection: FlowGraphInspection | None = None

    def __post_init__(self) -> None:
        if self.definition is not None:
            assert isinstance(self.definition, FlowDefinition)
        if self.canonical_source is not None:
            assert (
                isinstance(self.canonical_source, str)
                and self.canonical_source.strip()
            ), "canonical_source must be a non-empty string"
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)
        assert isinstance(self.authoring_graph, bool)
        if self.graph_inspection is not None:
            assert isinstance(self.graph_inspection, FlowGraphInspection)

    @property
    def ok(self) -> bool:
        return (
            self.definition is not None
            and self.canonical_source is not None
            and not _has_error(self.all_diagnostics)
        )

    @property
    def all_diagnostics(self) -> tuple[FlowDiagnostic, ...]:
        return _combine_diagnostics(self.diagnostics, self.graph_inspection)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.all_diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "authoring_graph": self.authoring_graph,
            "diagnostics": self.public_diagnostics,
        }
        if self.definition is not None:
            value["definition"] = {
                "name": self.definition.name,
                "node_count": len(self.definition.nodes),
                "edge_count": len(self.definition.edges),
            }
        if self.canonical_source is not None:
            value["canonical_source"] = {
                "format": "toml",
                "strict": True,
            }
        if self.graph_inspection is not None:
            value["graph_inspection"] = self.graph_inspection.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphInspectionResult:
    inspection: FlowGraphInspection | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    authoring_graph: bool = False

    def __post_init__(self) -> None:
        if self.inspection is not None:
            assert isinstance(self.inspection, FlowGraphInspection)
        assert isinstance(self.diagnostics, tuple)
        for diagnostic in self.diagnostics:
            assert isinstance(diagnostic, FlowDiagnostic)
        assert isinstance(self.authoring_graph, bool)

    @property
    def ok(self) -> bool:
        return self.inspection is not None and not _has_error(
            self.all_diagnostics
        )

    @property
    def all_diagnostics(self) -> tuple[FlowDiagnostic, ...]:
        return _combine_diagnostics(self.diagnostics, self.inspection)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.all_diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "authoring_graph": self.authoring_graph,
            "diagnostics": self.public_diagnostics,
        }
        if self.inspection is not None:
            value["inspection"] = self.inspection.as_public_dict()
        return value


def parse_mermaid_view(
    source_text: str,
    *,
    import_mode: FlowViewImportMode = FlowViewImportMode.PRESENTATION,
    source: str | None = None,
) -> MermaidFlowViewNormalizationResult:
    assert isinstance(source_text, str), "source_text must be a string"
    assert isinstance(import_mode, FlowViewImportMode)
    if source is not None:
        assert source.strip(), "source must be non-empty"
    return normalize_mermaid_flow_view(
        source_text,
        import_mode=import_mode,
        source=source,
    )


def render_flow_view(view: FlowView) -> MermaidRenderResult:
    assert isinstance(view, FlowView)
    return render_mermaid_view(view)


def bind_flow_view(
    view: FlowView,
    definition: FlowDefinition,
) -> FlowViewBindingResult:
    assert isinstance(view, FlowView)
    assert isinstance(definition, FlowDefinition)
    return bind_flow_view_definition(view, definition)


def compare_flow_topology(
    view: FlowView,
    definition: FlowDefinition,
) -> FlowViewBindingResult:
    assert isinstance(view, FlowView)
    assert isinstance(definition, FlowDefinition)
    return bind_flow_view_definition(view, definition)


async def compile_flow_source(
    source: str,
    *,
    source_path: str | Path | None = None,
    registry: FlowNodeRegistry | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowDefinitionCompileResult:
    assert isinstance(source, str), "source must be a string"
    if source_path is not None:
        assert isinstance(source_path, str | Path)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    assert_text_encoding(encoding)
    result = await FlowDefinitionLoader(
        registry,
        encoding=encoding,
    ).loads_validation_result(
        source,
        source_path=source_path,
        encoding=encoding,
    )
    if result.definition is None:
        return FlowDefinitionCompileResult(
            diagnostics=result.diagnostics,
            authoring_graph=result.authoring_graph,
            graph_inspection=result.graph_inspection,
        )
    canonical_source = serialize_flow_definition(result.definition)
    canonical_result = await FlowDefinitionLoader(
        registry,
        encoding=encoding,
    ).loads_validation_result(
        canonical_source,
        encoding=encoding,
    )
    if not canonical_result.ok:
        return FlowDefinitionCompileResult(
            diagnostics=result.diagnostics + canonical_result.diagnostics,
            authoring_graph=result.authoring_graph,
            graph_inspection=result.graph_inspection,
        )
    canonical_definition = canonical_result.definition
    assert canonical_definition is not None
    return FlowDefinitionCompileResult(
        definition=canonical_definition,
        canonical_source=canonical_source,
        diagnostics=result.diagnostics,
        authoring_graph=result.authoring_graph,
        graph_inspection=result.graph_inspection,
    )


async def compile_flow_file(
    path: str | Path,
    *,
    registry: FlowNodeRegistry | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowDefinitionCompileResult:
    assert isinstance(path, str | Path), "path must be a string or path"
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    assert_text_encoding(encoding)
    source_path = Path(path)
    try:
        source = await read_text(source_path, encoding=encoding)
    except (OSError, UnicodeDecodeError) as exc:
        return FlowDefinitionCompileResult(
            diagnostics=(_file_read_diagnostic(exc),),
        )
    return await compile_flow_source(
        source,
        source_path=source_path,
        registry=registry,
        encoding=encoding,
    )


async def inspect_flow_graph_source(
    source: str,
    *,
    source_path: str | Path | None = None,
    registry: FlowNodeRegistry | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowGraphInspectionResult:
    assert isinstance(source, str), "source must be a string"
    if source_path is not None:
        assert isinstance(source_path, str | Path)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    assert_text_encoding(encoding)
    result = await FlowDefinitionLoader(
        registry,
        encoding=encoding,
    ).loads_validation_result(
        source,
        source_path=source_path,
        encoding=encoding,
    )
    diagnostics = result.diagnostics
    if (
        result.graph_inspection is None
        and not result.authoring_graph
        and not diagnostics
    ):
        diagnostics = (
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.MISSING_SOURCE,
                "graph.source",
            ),
        )
    return FlowGraphInspectionResult(
        inspection=result.graph_inspection,
        diagnostics=diagnostics,
        authoring_graph=result.authoring_graph,
    )


async def inspect_flow_graph_file(
    path: str | Path,
    *,
    registry: FlowNodeRegistry | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowGraphInspectionResult:
    assert isinstance(path, str | Path), "path must be a string or path"
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    assert_text_encoding(encoding)
    source_path = Path(path)
    try:
        source = await read_text(source_path, encoding=encoding)
    except (OSError, UnicodeDecodeError) as exc:
        return FlowGraphInspectionResult(
            diagnostics=(_file_read_diagnostic(exc),),
        )
    return await inspect_flow_graph_source(
        source,
        source_path=source_path,
        registry=registry,
        encoding=encoding,
    )


def skeleton_from_mermaid_view(
    view: FlowView,
    *,
    name: str,
    version: str | None = None,
    revision: str | None = None,
) -> FlowDefinitionSkeletonResult:
    assert isinstance(view, FlowView)
    return FlowDefinitionSkeletonResult(
        definition=create_flow_definition_skeleton(
            view,
            name=name,
            version=version,
            revision=revision,
        ),
        diagnostics=view.diagnostics,
    )
