from .binding import (
    FlowViewBindingResult,
    bind_flow_view_definition,
    create_flow_definition_skeleton,
)
from .definition import FlowDefinition
from .diagnostics import FlowDiagnostic, FlowDiagnosticSeverity
from .mermaid import (
    MermaidFlowViewNormalizationResult,
    MermaidRenderResult,
    normalize_mermaid_flow_view,
    render_mermaid_view,
)
from .view import FlowView, FlowViewImportMode

from dataclasses import dataclass


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
        return not any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )


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
