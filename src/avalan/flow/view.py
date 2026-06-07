from .diagnostics import FlowDiagnostic, FlowDiagnosticSeverity, FlowSourceSpan

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

FlowViewMetadata: TypeAlias = Mapping[str, object]
FlowViewStyleProperties: TypeAlias = Mapping[str, str]


class FlowViewImportMode(StrEnum):
    PRESENTATION = "presentation"
    EXECUTABLE = "executable"


class FlowViewDirection(StrEnum):
    TD = "TD"
    TB = "TB"
    BT = "BT"
    LR = "LR"
    RL = "RL"


class FlowViewNodeShape(StrEnum):
    RECTANGLE = "rectangle"
    ROUNDED = "rounded"
    DIAMOND = "diamond"
    CIRCLE = "circle"
    DOUBLE_CIRCLE = "double_circle"
    STADIUM = "stadium"
    SUBROUTINE = "subroutine"
    CYLINDER = "cylinder"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallelogram"
    TRAPEZOID = "trapezoid"
    ASYMMETRIC = "asymmetric"


class FlowViewEdgeStyle(StrEnum):
    SOLID = "solid"
    DOTTED = "dotted"
    THICK = "thick"


def _empty_mapping() -> FlowViewMetadata:
    return MappingProxyType({})


def _empty_style_properties() -> FlowViewStyleProperties:
    return MappingProxyType({})


def _freeze_mapping(value: FlowViewMetadata) -> FlowViewMetadata:
    assert isinstance(value, Mapping), "metadata must be a mapping"
    frozen: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str), "metadata keys must be strings"
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_style_properties(
    value: FlowViewStyleProperties,
) -> FlowViewStyleProperties:
    assert isinstance(value, Mapping), "properties must be a mapping"
    frozen: dict[str, str] = {}
    for key, item in value.items():
        _assert_non_empty_string(key, "properties key")
        _assert_non_empty_string(item, f"properties.{key}")
        frozen[key] = item
    return MappingProxyType(frozen)


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert (
        isinstance(value, str) and value.strip()
    ), f"{field_name} must be a non-empty string"


def _assert_optional_non_empty_string(
    value: str | None,
    field_name: str,
) -> None:
    if value is not None:
        _assert_non_empty_string(value, field_name)


def _assert_string_tuple(values: tuple[str, ...], field_name: str) -> None:
    assert isinstance(values, tuple), f"{field_name} must be a tuple"
    for value in values:
        _assert_non_empty_string(value, field_name)


def _assert_optional_source_span(
    value: FlowSourceSpan | None,
    field_name: str,
) -> None:
    if value is not None:
        assert isinstance(
            value, FlowSourceSpan
        ), f"{field_name} must be a source span"


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewClassDefinition:
    name: str
    properties: FlowViewStyleProperties = field(
        default_factory=_empty_style_properties
    )
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        object.__setattr__(
            self,
            "properties",
            _freeze_style_properties(self.properties),
        )
        _assert_optional_source_span(self.source_span, "source_span")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewStyle:
    target: str
    properties: FlowViewStyleProperties = field(
        default_factory=_empty_style_properties
    )
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.target, "target")
        object.__setattr__(
            self,
            "properties",
            _freeze_style_properties(self.properties),
        )
        _assert_optional_source_span(self.source_span, "source_span")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewLinkStyle:
    properties: FlowViewStyleProperties
    edge: str | None = None
    edge_index: int | None = None
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        assert (
            self.edge is not None or self.edge_index is not None
        ), "edge or edge_index is required"
        if self.edge is not None:
            _assert_non_empty_string(self.edge, "edge")
        if self.edge_index is not None:
            assert isinstance(self.edge_index, int) and not isinstance(
                self.edge_index,
                bool,
            )
            assert self.edge_index >= 0, "edge_index must be non-negative"
        object.__setattr__(
            self,
            "properties",
            _freeze_style_properties(self.properties),
        )
        _assert_optional_source_span(self.source_span, "source_span")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewComment:
    text: str
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.text, str), "text must be a string"
        _assert_optional_source_span(self.source_span, "source_span")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewNode:
    id: str
    label: str | None = None
    shape: FlowViewNodeShape = FlowViewNodeShape.RECTANGLE
    classes: tuple[str, ...] = ()
    style: FlowViewStyleProperties = field(
        default_factory=_empty_style_properties
    )
    metadata: FlowViewMetadata = field(default_factory=_empty_mapping)
    source_span: FlowSourceSpan | None = None
    implicit: bool = False
    group: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        _assert_optional_non_empty_string(self.label, "label")
        assert isinstance(self.shape, FlowViewNodeShape)
        _assert_string_tuple(self.classes, "classes")
        object.__setattr__(
            self,
            "style",
            _freeze_style_properties(self.style),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        _assert_optional_source_span(self.source_span, "source_span")
        assert isinstance(self.implicit, bool)
        _assert_optional_non_empty_string(self.group, "group")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewEdge:
    id: str
    source: str
    target: str
    label: str | None = None
    style: FlowViewEdgeStyle = FlowViewEdgeStyle.SOLID
    classes: tuple[str, ...] = ()
    metadata: FlowViewMetadata = field(default_factory=_empty_mapping)
    source_span: FlowSourceSpan | None = None
    bidirectional: bool = False

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        _assert_non_empty_string(self.source, "source")
        _assert_non_empty_string(self.target, "target")
        _assert_optional_non_empty_string(self.label, "label")
        assert isinstance(self.style, FlowViewEdgeStyle)
        _assert_string_tuple(self.classes, "classes")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        _assert_optional_source_span(self.source_span, "source_span")
        assert isinstance(self.bidirectional, bool)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowViewGroup:
    id: str
    label: str | None = None
    parent: str | None = None
    direction: FlowViewDirection | None = None
    nodes: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    style: FlowViewStyleProperties = field(
        default_factory=_empty_style_properties
    )
    metadata: FlowViewMetadata = field(default_factory=_empty_mapping)
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.id, "id")
        _assert_optional_non_empty_string(self.label, "label")
        _assert_optional_non_empty_string(self.parent, "parent")
        if self.direction is not None:
            assert isinstance(self.direction, FlowViewDirection)
        _assert_string_tuple(self.nodes, "nodes")
        _assert_string_tuple(self.groups, "groups")
        _assert_string_tuple(self.classes, "classes")
        object.__setattr__(
            self,
            "style",
            _freeze_style_properties(self.style),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        _assert_optional_source_span(self.source_span, "source_span")


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowView:
    import_mode: FlowViewImportMode
    direction: FlowViewDirection | None = None
    nodes: tuple[FlowViewNode, ...] = ()
    edges: tuple[FlowViewEdge, ...] = ()
    groups: tuple[FlowViewGroup, ...] = ()
    class_definitions: tuple[FlowViewClassDefinition, ...] = ()
    styles: tuple[FlowViewStyle, ...] = ()
    link_styles: tuple[FlowViewLinkStyle, ...] = ()
    comments: tuple[FlowViewComment, ...] = ()
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    metadata: FlowViewMetadata = field(default_factory=_empty_mapping)
    source_span: FlowSourceSpan | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.import_mode, FlowViewImportMode)
        if self.direction is not None:
            assert isinstance(self.direction, FlowViewDirection)
        self._assert_nodes()
        self._assert_edges()
        self._assert_groups()
        self._assert_class_definitions()
        self._assert_tuple_items(self.styles, "styles", FlowViewStyle)
        self._assert_tuple_items(
            self.link_styles,
            "link_styles",
            FlowViewLinkStyle,
        )
        self._assert_tuple_items(self.comments, "comments", FlowViewComment)
        self._assert_tuple_items(
            self.diagnostics,
            "diagnostics",
            FlowDiagnostic,
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        _assert_optional_source_span(self.source_span, "source_span")

    @property
    def node_map(self) -> Mapping[str, FlowViewNode]:
        return MappingProxyType({node.id: node for node in self.nodes})

    @property
    def edge_map(self) -> Mapping[str, FlowViewEdge]:
        return MappingProxyType({edge.id: edge for edge in self.edges})

    @property
    def group_map(self) -> Mapping[str, FlowViewGroup]:
        return MappingProxyType({group.id: group for group in self.groups})

    @property
    def class_definition_map(self) -> Mapping[str, FlowViewClassDefinition]:
        return MappingProxyType(
            {
                class_definition.name: class_definition
                for class_definition in self.class_definitions
            }
        )

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    @property
    def has_errors(self) -> bool:
        return any(
            diagnostic.severity == FlowDiagnosticSeverity.ERROR
            for diagnostic in self.diagnostics
        )

    @staticmethod
    def _assert_tuple_items(
        values: tuple[object, ...],
        field_name: str,
        item_type: type[object],
    ) -> None:
        assert isinstance(values, tuple), f"{field_name} must be a tuple"
        for value in values:
            assert isinstance(value, item_type)

    def _assert_nodes(self) -> None:
        self._assert_tuple_items(self.nodes, "nodes", FlowViewNode)
        self._assert_unique_ids(
            tuple(node.id for node in self.nodes),
            "nodes",
        )

    def _assert_edges(self) -> None:
        self._assert_tuple_items(self.edges, "edges", FlowViewEdge)
        self._assert_unique_ids(
            tuple(edge.id for edge in self.edges),
            "edges",
        )

    def _assert_groups(self) -> None:
        self._assert_tuple_items(self.groups, "groups", FlowViewGroup)
        self._assert_unique_ids(
            tuple(group.id for group in self.groups),
            "groups",
        )

    def _assert_class_definitions(self) -> None:
        self._assert_tuple_items(
            self.class_definitions,
            "class_definitions",
            FlowViewClassDefinition,
        )
        self._assert_unique_ids(
            tuple(
                class_definition.name
                for class_definition in self.class_definitions
            ),
            "class_definitions",
        )

    @staticmethod
    def _assert_unique_ids(values: tuple[str, ...], field_name: str) -> None:
        seen: set[str] = set()
        for value in values:
            assert value not in seen, f"{field_name} must have unique ids"
            seen.add(value)
