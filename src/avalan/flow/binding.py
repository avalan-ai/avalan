from .definition import FlowDefinition, FlowEdgeDefinition, FlowNodeDefinition
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)
from .view import FlowView, FlowViewEdge, FlowViewNode

from dataclasses import dataclass

FLOW_VIEW_SKELETON_NODE_TYPE = "flow_view_skeleton"
FLOW_VIEW_SKELETON_TAG = "flow_view_skeleton"


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewBindingResult:
    view: FlowView
    definition: FlowDefinition
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.view, FlowView)
        assert isinstance(self.definition, FlowDefinition)
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


def bind_flow_view_definition(
    view: FlowView,
    definition: FlowDefinition,
) -> FlowViewBindingResult:
    assert isinstance(view, FlowView)
    assert isinstance(definition, FlowDefinition)

    diagnostics = list(view.diagnostics)
    definition_nodes = definition.node_map
    view_nodes = view.node_map

    for definition_node in definition.nodes:
        if definition_node.name not in view_nodes:
            diagnostics.append(
                _diagnostic(
                    "flow.view.binding.missing_node",
                    "Structured flow node is missing from the view.",
                    f"definition.nodes.{definition_node.name}",
                )
            )

    for view_node in view.nodes:
        matching_definition_node = definition_nodes.get(view_node.id)
        if matching_definition_node is None:
            diagnostics.append(
                _diagnostic(
                    "flow.view.binding.extra_node",
                    "Flow View node is not present in the definition.",
                    f"view.nodes.{view_node.id}",
                    source_span=view_node.source_span,
                )
            )
            diagnostics.append(_missing_node_semantics(view_node))
        elif _node_lacks_semantics(matching_definition_node):
            diagnostics.append(_missing_node_semantics(view_node))

    _compare_edges(view, definition, diagnostics)

    return FlowViewBindingResult(
        view=view,
        definition=definition,
        diagnostics=tuple(diagnostics),
    )


def create_flow_definition_skeleton(
    view: FlowView,
    *,
    name: str,
    version: str | None = None,
    revision: str | None = None,
) -> FlowDefinition:
    assert isinstance(view, FlowView)
    assert name.strip(), "name must be a non-empty string"
    if version is not None:
        assert version.strip(), "version must be a non-empty string"
    if revision is not None:
        assert revision.strip(), "revision must be a non-empty string"

    return FlowDefinition(
        name=name,
        version=version,
        revision=revision,
        description="Non-executable topology skeleton.",
        nodes=tuple(_skeleton_node(node) for node in view.nodes),
        edges=tuple(
            FlowEdgeDefinition(
                source=edge.source,
                target=edge.target,
                label=edge.label,
            )
            for edge in view.edges
        ),
        tags=(FLOW_VIEW_SKELETON_TAG,),
        variables={
            "executable": False,
            "source": "flow_view",
        },
    )


def _compare_edges(
    view: FlowView,
    definition: FlowDefinition,
    diagnostics: list[FlowDiagnostic],
) -> None:
    view_edges_by_key = _view_edges_by_key(view.edges)
    definition_edges_by_key = _definition_edges_by_key(definition.edges)
    definition_is_skeleton = _definition_lacks_edge_semantics(definition)
    seen_definition_edges: dict[tuple[str, str], int] = {}
    seen_view_edges: dict[tuple[str, str], int] = {}

    for definition_edge in definition.edges:
        key = (definition_edge.source, definition_edge.target)
        seen_definition_edges[key] = seen_definition_edges.get(key, 0) + 1
        if seen_definition_edges[key] > len(view_edges_by_key.get(key, ())):
            diagnostics.append(
                _diagnostic(
                    "flow.view.binding.missing_edge",
                    "Structured flow edge is missing from the view.",
                    "definition.edges."
                    f"{definition_edge.source}->{definition_edge.target}",
                )
            )

    for view_edge in view.edges:
        key = (view_edge.source, view_edge.target)
        seen_view_edges[key] = seen_view_edges.get(key, 0) + 1
        if seen_view_edges[key] > len(definition_edges_by_key.get(key, ())):
            diagnostics.append(
                _diagnostic(
                    "flow.view.binding.extra_edge",
                    "Flow View edge is not present in the definition.",
                    f"view.edges.{view_edge.id}",
                    source_span=view_edge.source_span,
                )
            )
            diagnostics.append(_missing_edge_semantics(view_edge))
        elif definition_is_skeleton:
            diagnostics.append(_missing_edge_semantics(view_edge))


def _view_edges_by_key(
    edges: tuple[FlowViewEdge, ...],
) -> dict[tuple[str, str], tuple[FlowViewEdge, ...]]:
    grouped: dict[tuple[str, str], list[FlowViewEdge]] = {}
    for edge in edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _definition_edges_by_key(
    edges: tuple[FlowEdgeDefinition, ...],
) -> dict[tuple[str, str], tuple[FlowEdgeDefinition, ...]]:
    grouped: dict[tuple[str, str], list[FlowEdgeDefinition]] = {}
    for edge in edges:
        grouped.setdefault((edge.source, edge.target), []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _skeleton_node(node: FlowViewNode) -> FlowNodeDefinition:
    config: dict[str, object] = {
        "executable": False,
        "shape": node.shape.value,
    }
    if node.label is not None:
        config["label"] = node.label
    if node.group is not None:
        config["group"] = node.group

    return FlowNodeDefinition(
        name=node.id,
        type=FLOW_VIEW_SKELETON_NODE_TYPE,
        config=config,
    )


def _node_lacks_semantics(node: FlowNodeDefinition) -> bool:
    return (
        node.type == FLOW_VIEW_SKELETON_NODE_TYPE
        or node.config.get("executable") is False
    )


def _definition_lacks_edge_semantics(definition: FlowDefinition) -> bool:
    return (
        FLOW_VIEW_SKELETON_TAG in definition.tags
        or definition.variables.get("executable") is False
    )


def _missing_node_semantics(node: FlowViewNode) -> FlowDiagnostic:
    return _diagnostic(
        "flow.view.binding.missing_node_semantics",
        "Flow View node has no executable node semantics.",
        f"view.nodes.{node.id}",
        source_span=node.source_span,
    )


def _missing_edge_semantics(edge: FlowViewEdge) -> FlowDiagnostic:
    return _diagnostic(
        "flow.view.binding.missing_edge_semantics",
        "Flow View edge has no executable edge semantics.",
        f"view.edges.{edge.id}",
        source_span=edge.source_span,
    )


def _diagnostic(
    code: str,
    message: str,
    path: str,
    *,
    source_span: FlowSourceSpan | None = None,
) -> FlowDiagnostic:
    return FlowDiagnostic(
        code=code,
        category=FlowDiagnosticCategory.FLOW_VIEW_BINDING,
        path=path,
        source_span=source_span,
        message=message,
    )
