from .condition import FlowCondition
from .definition import (
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowMetadata,
    FlowRouteMatchPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class FlowGraphFormat(StrEnum):
    MERMAID = "mermaid"


class FlowGraphSourceKind(StrEnum):
    INLINE = "inline"
    FILE = "file"


class FlowGraphMode(StrEnum):
    EXECUTABLE = "executable"


_FLOW_GRAPH_EDGE_BINDING_FIELDS = frozenset(
    {
        "condition",
        "default",
        "kind",
        "label",
        "priority",
        "routing_policy",
    }
)


def _empty_mapping() -> FlowMetadata:
    return MappingProxyType({})


def _empty_edge_bindings() -> Mapping[str, "FlowGraphEdgeBinding"]:
    return MappingProxyType({})


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_diagnostics(value: tuple[FlowDiagnostic, ...]) -> None:
    assert isinstance(value, tuple), "diagnostics must be a tuple"
    for diagnostic in value:
        assert isinstance(diagnostic, FlowDiagnostic)


def _freeze_mapping(value: FlowMetadata, *, field_name: str) -> FlowMetadata:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    frozen: dict[str, object] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, f"{field_name} key")
        frozen[key] = _freeze_value(item, field_name=field_name)
    return MappingProxyType(frozen)


def _freeze_value(value: object, *, field_name: str) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, field_name=field_name)
    if isinstance(value, list | tuple):
        return tuple(
            _freeze_value(item, field_name=field_name) for item in value
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphSource:
    format: FlowGraphFormat = FlowGraphFormat.MERMAID
    source_kind: FlowGraphSourceKind
    mode: FlowGraphMode = FlowGraphMode.EXECUTABLE
    diagram: str | None = None
    path: Path | None = None
    base_path: Path | None = None
    source_identity: str | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.format, FlowGraphFormat)
        assert isinstance(self.source_kind, FlowGraphSourceKind)
        assert isinstance(self.mode, FlowGraphMode)
        if self.diagram is not None:
            _assert_non_empty_string(self.diagram, "diagram")
        if self.path is not None:
            assert isinstance(self.path, Path)
        if self.base_path is not None:
            assert isinstance(self.base_path, Path)
        if self.source_identity is not None:
            _assert_non_empty_string(self.source_identity, "source_identity")
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)
        match self.source_kind:
            case FlowGraphSourceKind.INLINE:
                assert (
                    self.diagram is not None
                ), "inline graph source needs diagram"
                assert (
                    self.path is None
                ), "inline graph source cannot include path"
            case FlowGraphSourceKind.FILE:
                assert self.path is not None, "file graph source needs path"
                assert (
                    self.diagram is None
                ), "file graph source cannot include diagram"

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "format": self.format.value,
            "source_kind": self.source_kind.value,
            "mode": self.mode.value,
        }
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphEdgeBinding:
    edge_id: str
    metadata: FlowMetadata = field(default_factory=_empty_mapping)
    label: str | None = None
    kind: FlowEdgeKind | None = None
    condition: FlowCondition | None = None
    priority: int | None = None
    default: bool | None = None
    routing_policy: FlowRouteMatchPolicy | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.edge_id, "edge_id")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )
        for key in self.metadata:
            assert (
                key in _FLOW_GRAPH_EDGE_BINDING_FIELDS
            ), f"metadata.{key} is not allowed"
        if self.label is not None:
            _assert_non_empty_string(self.label, "label")
        if self.kind is not None:
            assert isinstance(self.kind, FlowEdgeKind)
        if self.condition is not None:
            assert isinstance(self.condition, FlowCondition)
        if self.priority is not None:
            assert isinstance(self.priority, int) and not isinstance(
                self.priority,
                bool,
            )
        if self.default is not None:
            assert isinstance(self.default, bool)
        if self.routing_policy is not None:
            assert isinstance(self.routing_policy, FlowRouteMatchPolicy)
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"edge_id": self.edge_id}
        if self.metadata:
            value["metadata_fields"] = tuple(sorted(self.metadata))
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_public_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowGraphCompileResult:
    source: FlowGraphSource | None = None
    edges: tuple[FlowEdgeDefinition, ...] = ()
    edge_bindings: Mapping[str, FlowGraphEdgeBinding] = field(
        default_factory=_empty_edge_bindings
    )
    diagnostics: tuple[FlowDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if self.source is not None:
            assert isinstance(self.source, FlowGraphSource)
        assert isinstance(self.edges, tuple), "edges must be a tuple"
        for edge in self.edges:
            assert isinstance(edge, FlowEdgeDefinition)
        assert isinstance(
            self.edge_bindings,
            Mapping,
        ), "edge_bindings must be a mapping"
        frozen: dict[str, FlowGraphEdgeBinding] = {}
        for key, binding in self.edge_bindings.items():
            _assert_non_empty_string(key, "edge_bindings key")
            assert isinstance(binding, FlowGraphEdgeBinding)
            assert (
                key == binding.edge_id
            ), "edge binding key must match edge_id"
            frozen[key] = binding
        object.__setattr__(self, "edge_bindings", MappingProxyType(frozen))
        _assert_diagnostics(self.diagnostics)

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

    def as_public_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ok": self.ok,
            "edge_count": len(self.edges),
            "edge_binding_count": len(self.edge_bindings),
        }
        if self.source is not None:
            value["source"] = self.source.as_public_dict()
        if self.diagnostics:
            value["diagnostics"] = self.public_diagnostics
        return value
