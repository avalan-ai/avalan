from .binding import (
    FlowViewBindingResult,
    bind_flow_view_definition,
    create_flow_definition_skeleton,
)
from .definition import FlowDefinition
from .diagnostics import FlowDiagnostic, FlowDiagnosticSeverity
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
from pathlib import Path


def _has_error(diagnostics: tuple[FlowDiagnostic, ...]) -> bool:
    return any(
        diagnostic.severity == FlowDiagnosticSeverity.ERROR
        for diagnostic in diagnostics
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
            and not _has_error(self.diagnostics)
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "authoring_graph": self.authoring_graph,
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
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
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
        diagnostics = self.diagnostics
        if self.inspection is not None:
            diagnostics += self.inspection.diagnostics
        return self.inspection is not None and not _has_error(diagnostics)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "authoring_graph": self.authoring_graph,
        }
        if self.inspection is not None:
            value["inspection"] = self.inspection.as_public_dict()
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
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


def compile_flow_source(
    source: str,
    *,
    source_path: str | Path | None = None,
    registry: FlowNodeRegistry | None = None,
) -> FlowDefinitionCompileResult:
    assert isinstance(source, str), "source must be a string"
    if source_path is not None:
        assert isinstance(source_path, str | Path)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    result = FlowDefinitionLoader(registry).loads_validation_result(
        source,
        source_path=source_path,
    )
    if result.definition is None:
        return FlowDefinitionCompileResult(
            diagnostics=result.diagnostics,
            authoring_graph=result.authoring_graph,
            graph_inspection=result.graph_inspection,
        )
    return FlowDefinitionCompileResult(
        definition=result.definition,
        canonical_source=serialize_flow_definition(result.definition),
        diagnostics=result.diagnostics,
        authoring_graph=result.authoring_graph,
        graph_inspection=result.graph_inspection,
    )


def compile_flow_file(
    path: str | Path,
    *,
    registry: FlowNodeRegistry | None = None,
) -> FlowDefinitionCompileResult:
    assert isinstance(path, str | Path), "path must be a string or path"
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    source_path = Path(path)
    return compile_flow_source(
        source_path.read_text(encoding="utf-8"),
        source_path=source_path,
        registry=registry,
    )


def inspect_flow_graph_source(
    source: str,
    *,
    source_path: str | Path | None = None,
    registry: FlowNodeRegistry | None = None,
) -> FlowGraphInspectionResult:
    assert isinstance(source, str), "source must be a string"
    if source_path is not None:
        assert isinstance(source_path, str | Path)
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    result = FlowDefinitionLoader(registry).loads_validation_result(
        source,
        source_path=source_path,
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


def inspect_flow_graph_file(
    path: str | Path,
    *,
    registry: FlowNodeRegistry | None = None,
) -> FlowGraphInspectionResult:
    assert isinstance(path, str | Path), "path must be a string or path"
    if registry is not None:
        assert isinstance(registry, FlowNodeRegistry)
    source_path = Path(path)
    return inspect_flow_graph_source(
        source_path.read_text(encoding="utf-8"),
        source_path=source_path,
        registry=registry,
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
