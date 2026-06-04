from .definition import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeDefinition,
    FlowOutputDefinition,
    FlowOutputType,
)
from .flow import Flow
from .registry import FlowNodeRegistry, default_flow_node_registry

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from tomllib import TOMLDecodeError, loads
from typing import TypeVar

EnumValue = TypeVar("EnumValue", FlowInputType, FlowOutputType)
RawSection = Mapping[str, object]

_ALLOWED_TOP_LEVEL_SECTIONS = frozenset(
    {
        "edges",
        "flow",
        "input",
        "nodes",
        "output",
        "variables",
    }
)
_ALLOWED_FLOW_FIELDS = frozenset(
    {
        "description",
        "entrypoint",
        "input",
        "name",
        "output",
        "output_node",
        "version",
    }
)
_ALLOWED_INPUT_FIELDS = frozenset(
    {
        "mime_types",
        "name",
        "schema",
        "schema_ref",
        "type",
    }
)
_ALLOWED_OUTPUT_FIELDS = frozenset({"name", "schema", "schema_ref", "type"})
_ALLOWED_NODE_FIELDS = frozenset(
    {
        "config",
        "field",
        "input",
        "output",
        "path",
        "ref",
        "type",
        "value",
    }
)
_KNOWN_DEFERRED_NODE_TYPES = frozenset({"agent"})
_UNTRUSTED_NODE_TYPES = frozenset(
    {
        "callable",
        "function",
        "module",
        "python",
        "python_callable",
    }
)


class FlowLoadIssueCategory(StrEnum):
    PARSE = "parse"
    STRUCTURE = "structure"
    VALUE = "value"
    UNSUPPORTED = "unsupported"
    PRIVACY = "privacy"


class FlowLoadSeverity(StrEnum):
    ERROR = "error"


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowLoadIssue:
    code: str
    path: str
    message: str
    hint: str
    category: FlowLoadIssueCategory
    severity: FlowLoadSeverity = FlowLoadSeverity.ERROR

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowLoadResult:
    definition: FlowDefinition | None
    flow: Flow | None = None
    issues: tuple[FlowLoadIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues and self.definition is not None


class FlowLoadError(ValueError):
    issues: tuple[FlowLoadIssue, ...]

    def __init__(self, issues: tuple[FlowLoadIssue, ...]) -> None:
        assert issues, "issues must not be empty"
        self.issues = issues
        summary = ", ".join(
            f"{issue.code} at {issue.path}" for issue in issues
        )
        super().__init__(f"flow definition could not be loaded: {summary}")


class FlowDefinitionLoader:
    def __init__(self, registry: FlowNodeRegistry | None = None) -> None:
        self._registry = registry or default_flow_node_registry()

    def load(self, path: str | Path) -> FlowDefinition:
        result = self.load_result(path)
        if result.definition is None:
            raise FlowLoadError(result.issues)
        return result.definition

    def load_result(self, path: str | Path) -> FlowLoadResult:
        source_path = Path(path)
        source = source_path.read_text(encoding="utf-8")
        return self.loads_result(source, source_path=source_path)

    def loads(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
    ) -> FlowDefinition:
        result = self.loads_result(source, source_path=source_path)
        if result.definition is None:
            raise FlowLoadError(result.issues)
        return result.definition

    def loads_result(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
    ) -> FlowLoadResult:
        assert isinstance(source, str), "source must be a string"
        try:
            raw = loads(source)
        except TOMLDecodeError:
            return FlowLoadResult(
                definition=None,
                issues=(
                    _issue(
                        code="flow.malformed_toml",
                        path="toml",
                        message="Flow definition TOML is malformed.",
                        hint="Fix the TOML syntax and retry loading.",
                        category=FlowLoadIssueCategory.PARSE,
                    ),
                ),
            )
        return _build_result(
            raw,
            registry=self._registry,
            source_path=source_path,
        )


def load_flow_definition(path: str | Path) -> FlowDefinition:
    return FlowDefinitionLoader().load(path)


def load_flow_definition_result(path: str | Path) -> FlowLoadResult:
    return FlowDefinitionLoader().load_result(path)


def loads_flow_definition(
    source: str,
    *,
    source_path: str | Path | None = None,
) -> FlowDefinition:
    return FlowDefinitionLoader().loads(source, source_path=source_path)


def loads_flow_definition_result(
    source: str,
    *,
    source_path: str | Path | None = None,
) -> FlowLoadResult:
    return FlowDefinitionLoader().loads_result(
        source,
        source_path=source_path,
    )


def build_flow(
    definition: FlowDefinition,
    registry: FlowNodeRegistry | None = None,
) -> Flow:
    assert isinstance(definition, FlowDefinition)
    node_registry = registry or default_flow_node_registry()
    flow = Flow()
    for node_definition in definition.nodes:
        flow.add_node(node_registry.build(node_definition))
    for edge in definition.edges:
        flow.add_connection(edge.source, edge.target, label=edge.label)
    return flow


def _build_result(
    raw: Mapping[str, object],
    *,
    registry: FlowNodeRegistry,
    source_path: str | Path | None,
) -> FlowLoadResult:
    issues: list[FlowLoadIssue] = []
    _validate_top_level_sections(raw, issues)
    flow_raw = _section(raw, "flow", issues, required=True)
    nodes_raw = _section(raw, "nodes", issues, required=True)
    if flow_raw is None or nodes_raw is None:
        return FlowLoadResult(definition=None, issues=tuple(issues))

    _validate_unknown_fields(
        flow_raw,
        allowed=_ALLOWED_FLOW_FIELDS,
        path="flow",
        issues=issues,
    )
    input_raw = _child_section(raw, flow_raw, "input", issues)
    output_raw = _child_section(raw, flow_raw, "output", issues)
    variables_raw = _section(raw, "variables", issues, required=False)
    name = _required_str(flow_raw, "flow.name", "name", issues)
    entrypoint = _required_str(
        flow_raw,
        "flow.entrypoint",
        "entrypoint",
        issues,
    )
    output_node = _required_str(
        flow_raw,
        "flow.output_node",
        "output_node",
        issues,
    )
    version = _optional_str(flow_raw, "flow.version", "version", issues)
    description = _optional_str(
        flow_raw,
        "flow.description",
        "description",
        issues,
    )
    input_definition = _input_definition(input_raw, issues)
    output_definition = _output_definition(output_raw, issues)
    variables = (
        _metadata(variables_raw, "variables", issues)
        if variables_raw is not None
        else {}
    )
    nodes = _node_definitions(nodes_raw, registry, issues)
    edges = _edge_definitions(raw.get("edges"), issues)
    if issues:
        return FlowLoadResult(definition=None, issues=tuple(issues))
    assert name is not None
    assert entrypoint is not None
    assert output_node is not None
    definition = FlowDefinition(
        name=name,
        version=version,
        description=description,
        entrypoint=entrypoint,
        output_node=output_node,
        input=input_definition,
        output=output_definition,
        variables=variables or {},
        nodes=nodes,
        edges=edges,
        definition_base=(
            Path(source_path).parent if source_path is not None else None
        ),
    )
    issues.extend(_validate_graph_contract(definition))
    if issues:
        return FlowLoadResult(definition=None, issues=tuple(issues))
    try:
        flow = build_flow(definition, registry)
    except (AssertionError, KeyError, TypeError, ValueError):
        return FlowLoadResult(
            definition=None,
            issues=(
                _issue(
                    code="flow.invalid_node",
                    path="nodes",
                    message="Flow node configuration is invalid.",
                    hint="Use only supported built-in node configuration.",
                    category=FlowLoadIssueCategory.VALUE,
                ),
            ),
        )
    return FlowLoadResult(definition=definition, flow=flow)


def _validate_top_level_sections(
    raw: Mapping[str, object],
    issues: list[FlowLoadIssue],
) -> None:
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key not in _ALLOWED_TOP_LEVEL_SECTIONS:
            issues.append(_unsupported_section_issue(key))
            continue
        if key == "edges":
            if not isinstance(value, list):
                issues.append(_invalid_section_type("edges"))


def _section(
    raw: Mapping[str, object],
    name: str,
    issues: list[FlowLoadIssue],
    *,
    required: bool,
) -> RawSection | None:
    value = raw.get(name)
    if value is None:
        if required:
            issues.append(_missing_section(name))
        return None
    if not isinstance(value, Mapping):
        if not _has_issue(issues, code="flow.invalid_section", path=name):
            issues.append(_invalid_section_type(name))
        return None
    return value


def _child_section(
    raw: Mapping[str, object],
    flow_raw: RawSection,
    name: str,
    issues: list[FlowLoadIssue],
) -> RawSection | None:
    top_level = raw.get(name)
    nested = flow_raw.get(name)
    if top_level is not None and nested is not None:
        issues.append(
            _issue(
                code="flow.duplicate_section",
                path=name,
                message="Flow section is declared more than once.",
                hint=f"Use either [{name}] or [flow.{name}], not both.",
                category=FlowLoadIssueCategory.STRUCTURE,
            )
        )
        return None
    value = top_level if top_level is not None else nested
    if value is None:
        return None
    if not isinstance(value, Mapping):
        if not _has_issue(issues, code="flow.invalid_section", path=name):
            issues.append(_invalid_section_type(name))
        return None
    return value


def _input_definition(
    raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> FlowInputDefinition | None:
    if raw is None:
        return None
    _validate_unknown_fields(
        raw,
        allowed=_ALLOWED_INPUT_FIELDS,
        path="flow.input",
        issues=issues,
    )
    name = _required_str(raw, "flow.input.name", "name", issues)
    input_type = _enum_value(
        raw,
        "flow.input.type",
        "type",
        FlowInputType,
        issues,
    )
    mime_types = _string_tuple(
        raw,
        "flow.input.mime_types",
        "mime_types",
        issues,
    )
    schema = _optional_mapping(raw, "flow.input.schema", "schema", issues)
    schema_ref = _optional_str(
        raw,
        "flow.input.schema_ref",
        "schema_ref",
        issues,
    )
    if name is None or input_type is None:
        return None
    return FlowInputDefinition(
        name=name,
        type=input_type,
        mime_types=mime_types,
        schema=schema,
        schema_ref=schema_ref,
    )


def _output_definition(
    raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> FlowOutputDefinition | None:
    if raw is None:
        return None
    _validate_unknown_fields(
        raw,
        allowed=_ALLOWED_OUTPUT_FIELDS,
        path="flow.output",
        issues=issues,
    )
    name = _required_str(raw, "flow.output.name", "name", issues)
    output_type = _enum_value(
        raw,
        "flow.output.type",
        "type",
        FlowOutputType,
        issues,
    )
    schema = _optional_mapping(raw, "flow.output.schema", "schema", issues)
    schema_ref = _optional_str(
        raw,
        "flow.output.schema_ref",
        "schema_ref",
        issues,
    )
    if name is None or output_type is None:
        return None
    return FlowOutputDefinition(
        name=name,
        type=output_type,
        schema=schema,
        schema_ref=schema_ref,
    )


def _node_definitions(
    raw: RawSection,
    registry: FlowNodeRegistry,
    issues: list[FlowLoadIssue],
) -> tuple[FlowNodeDefinition, ...]:
    nodes: list[FlowNodeDefinition] = []
    for name, value in raw.items():
        if not isinstance(name, str) or not name.strip():
            issues.append(_invalid_type("nodes", "Use named node tables."))
            continue
        if not isinstance(value, Mapping):
            issues.append(_invalid_section_type(f"nodes.{name}"))
            continue
        _validate_unknown_fields(
            value,
            allowed=_ALLOWED_NODE_FIELDS,
            path=f"nodes.{name}",
            issues=issues,
        )
        node_type = _required_str(
            value,
            f"nodes.{name}.type",
            "type",
            issues,
        )
        ref = _optional_str(value, f"nodes.{name}.ref", "ref", issues)
        input_name = _optional_str(
            value,
            f"nodes.{name}.input",
            "input",
            issues,
        )
        output_name = _optional_str(
            value,
            f"nodes.{name}.output",
            "output",
            issues,
        )
        if node_type is None:
            continue
        _validate_node_type(
            name,
            node_type,
            ref,
            registry,
            issues,
        )
        config = _node_config(value, issues, path=f"nodes.{name}")
        nodes.append(
            FlowNodeDefinition(
                name=name,
                type=node_type,
                ref=ref,
                input=input_name,
                output=output_name,
                config=config,
            )
        )
    return tuple(nodes)


def _edge_definitions(
    value: object,
    issues: list[FlowLoadIssue],
) -> tuple[FlowEdgeDefinition, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        issues.append(_invalid_section_type("edges"))
        return ()
    edges: list[FlowEdgeDefinition] = []
    for index, item in enumerate(value):
        path = f"edges[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_invalid_type(path, "Use TOML edge tables."))
            continue
        source = _required_str(item, f"{path}.source", "source", issues)
        target = _required_str(item, f"{path}.target", "target", issues)
        label = _optional_str(item, f"{path}.label", "label", issues)
        if source is None or target is None:
            continue
        edges.append(
            FlowEdgeDefinition(source=source, target=target, label=label)
        )
    return tuple(edges)


def _node_config(
    raw: RawSection,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> Mapping[str, object]:
    config: dict[str, object] = {}
    nested = raw.get("config")
    if nested is not None:
        if not isinstance(nested, Mapping):
            issues.append(_invalid_type(f"{path}.config", "Use a TOML table."))
        else:
            config.update(_metadata(nested, f"{path}.config", issues) or {})
    for key in ("field", "path", "value"):
        if key in raw:
            config[key] = raw[key]
    return config


def _validate_node_type(
    name: str,
    node_type: str,
    ref: str | None,
    registry: FlowNodeRegistry,
    issues: list[FlowLoadIssue],
) -> None:
    path = f"nodes.{name}.type"
    if node_type in _UNTRUSTED_NODE_TYPES:
        issues.append(
            _issue(
                code="flow.untrusted_callable",
                path=path,
                message="Flow TOML cannot import dynamic callables.",
                hint="Use a registered built-in node type.",
                category=FlowLoadIssueCategory.UNSUPPORTED,
            )
        )
        return
    if ref is not None and _is_path_escape(ref):
        issues.append(_path_escape_issue(f"nodes.{name}.ref"))
    if registry.supports(node_type):
        if ref is not None:
            issues.append(
                _issue(
                    code="flow.untrusted_callable",
                    path=f"nodes.{name}.ref",
                    message="Built-in flow nodes cannot load external refs.",
                    hint="Remove ref or use a later trusted node type.",
                    category=FlowLoadIssueCategory.UNSUPPORTED,
                )
            )
        return
    if node_type in _KNOWN_DEFERRED_NODE_TYPES:
        issues.append(
            _issue(
                code="flow.unsupported_node_type",
                path=path,
                message="Flow node type is not supported by this runtime.",
                hint="Use a currently registered built-in node type.",
                category=FlowLoadIssueCategory.UNSUPPORTED,
            )
        )
        return
    issues.append(
        _issue(
            code="flow.unknown_node_type",
            path=path,
            message="Flow node type is unknown.",
            hint="Use a registered built-in node type.",
            category=FlowLoadIssueCategory.UNSUPPORTED,
        )
    )


def _validate_graph_contract(
    definition: FlowDefinition,
) -> tuple[FlowLoadIssue, ...]:
    issues: list[FlowLoadIssue] = []
    node_names = {node.name for node in definition.nodes}
    if definition.entrypoint not in node_names:
        issues.append(
            _issue(
                code="flow.unknown_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint does not reference a node.",
                hint="Set flow.entrypoint to a declared node name.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    if definition.output_node not in node_names:
        issues.append(
            _issue(
                code="flow.unknown_output_node",
                path="flow.output_node",
                message="Flow output node does not reference a node.",
                hint="Set flow.output_node to a declared node name.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    for edge in definition.edges:
        if edge.source not in node_names:
            issues.append(_bad_reference_issue("edges.source"))
        if edge.target not in node_names:
            issues.append(_bad_reference_issue("edges.target"))
    if issues:
        return tuple(issues)
    outgoing = {node.name: 0 for node in definition.nodes}
    incoming = {node.name: 0 for node in definition.nodes}
    for edge in definition.edges:
        outgoing[edge.source] += 1
        incoming[edge.target] += 1
    terminals = {name for name, count in outgoing.items() if count == 0}
    start_nodes = {name for name, count in incoming.items() if count == 0}
    if definition.entrypoint not in start_nodes:
        issues.append(
            _issue(
                code="flow.invalid_entrypoint",
                path="flow.entrypoint",
                message="Flow entrypoint must be a start node.",
                hint="Use a node without inbound edges as the entrypoint.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    if len(terminals) > 1:
        issues.append(
            _issue(
                code="flow.multiple_outputs",
                path="flow.output_node",
                message="Flow has multiple terminal output nodes.",
                hint="Connect nodes so only one terminal output remains.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    elif definition.output_node not in terminals:
        issues.append(
            _issue(
                code="flow.invalid_output_node",
                path="flow.output_node",
                message="Flow output node must be terminal.",
                hint="Use a node without outbound edges as the output node.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    cycle = _cycle_nodes(definition)
    if cycle:
        issues.append(
            _issue(
                code="flow.cycle",
                path="edges",
                message="Flow graph contains a cycle.",
                hint="Remove the cyclic edge before running the flow.",
                category=FlowLoadIssueCategory.VALUE,
            )
        )
    return tuple(issues)


def _cycle_nodes(definition: FlowDefinition) -> frozenset[str]:
    outgoing: dict[str, list[str]] = {
        node.name: [] for node in definition.nodes
    }
    for edge in definition.edges:
        outgoing[edge.source].append(edge.target)
    visited: set[str] = set()
    stack: set[str] = set()
    cycle: set[str] = set()

    def visit(name: str) -> None:
        if name in stack:
            cycle.add(name)
            return
        if name in visited:
            return
        visited.add(name)
        stack.add(name)
        for target in outgoing[name]:
            visit(target)
            if target in cycle:
                cycle.add(name)
        stack.remove(name)

    for node in definition.nodes:
        visit(node.name)
    return frozenset(cycle)


def _validate_unknown_fields(
    raw: RawSection,
    *,
    allowed: frozenset[str],
    path: str,
    issues: list[FlowLoadIssue],
) -> None:
    for key in raw:
        if isinstance(key, str) and key not in allowed:
            issues.append(
                _issue(
                    code="flow.unsupported_field",
                    path=f"{path}.{key}",
                    message="Flow TOML field is not supported.",
                    hint="Remove unsupported fields from the native flow.",
                    category=FlowLoadIssueCategory.UNSUPPORTED,
                )
            )


def _required_str(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> str | None:
    value = raw.get(field)
    if value is None:
        issues.append(_missing_field(path))
        return None
    if not isinstance(value, str) or not value.strip():
        issues.append(_invalid_type(path, "Use a non-empty string."))
        return None
    return value


def _optional_str(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> str | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        issues.append(_invalid_type(path, "Use a non-empty string."))
        return None
    return value


def _enum_value(
    raw: RawSection,
    path: str,
    field: str,
    enum_type: type[EnumValue],
    issues: list[FlowLoadIssue],
) -> EnumValue | None:
    value = _required_str(raw, path, field, issues)
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError:
        issues.append(
            _issue(
                code="flow.invalid_enum",
                path=path,
                message="Flow field has an unsupported value.",
                hint=(
                    "Use one of: "
                    + ", ".join(member.value for member in enum_type)
                    + "."
                ),
                category=FlowLoadIssueCategory.VALUE,
            )
        )
        return None


def _string_tuple(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> tuple[str, ...]:
    value = raw.get(field, ())
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        issues.append(_invalid_type(path, "Use an array of strings."))
        return ()
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            issues.append(_invalid_type(path, "Use an array of strings."))
            return ()
        strings.append(item)
    return tuple(strings)


def _optional_mapping(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> Mapping[str, object] | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    return _metadata(value, path, issues)


def _metadata(
    raw: Mapping[str, object],
    path: str,
    issues: list[FlowLoadIssue],
) -> Mapping[str, object] | None:
    metadata: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            issues.append(_invalid_type(path, "Use string keys."))
            return None
        metadata[key] = _metadata_value(value)
    return metadata


def _metadata_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, list | tuple):
        return tuple(_metadata_value(item) for item in value)
    return value


def _is_path_escape(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _missing_section(section: str) -> FlowLoadIssue:
    return _issue(
        code="flow.missing_section",
        path=section,
        message="Flow definition is missing a required section.",
        hint=f"Add a [{section}] TOML table.",
        category=FlowLoadIssueCategory.STRUCTURE,
    )


def _missing_field(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.missing_field",
        path=path,
        message="Flow definition is missing a required field.",
        hint="Add the required field to the flow TOML.",
        category=FlowLoadIssueCategory.STRUCTURE,
    )


def _invalid_section_type(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.invalid_section",
        path=path,
        message="Flow definition section is invalid.",
        hint="Use TOML table syntax for flow sections.",
        category=FlowLoadIssueCategory.STRUCTURE,
    )


def _invalid_type(path: str, hint: str) -> FlowLoadIssue:
    return _issue(
        code="flow.invalid_type",
        path=path,
        message="Flow field has an invalid type.",
        hint=hint,
        category=FlowLoadIssueCategory.VALUE,
    )


def _unsupported_section_issue(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.unsupported_section",
        path=path,
        message="Flow TOML section is not supported.",
        hint="Remove unsupported runtime sections from the native flow.",
        category=FlowLoadIssueCategory.UNSUPPORTED,
    )


def _path_escape_issue(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.path_escape",
        path=path,
        message="Flow reference escapes the flow directory.",
        hint="Use a safe relative reference inside the flow directory.",
        category=FlowLoadIssueCategory.PRIVACY,
    )


def _bad_reference_issue(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.bad_reference",
        path=path,
        message="Flow reference does not match a declared node.",
        hint="Reference an existing node name.",
        category=FlowLoadIssueCategory.VALUE,
    )


def _issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: FlowLoadIssueCategory,
) -> FlowLoadIssue:
    return FlowLoadIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
    )


def _has_issue(
    issues: list[FlowLoadIssue],
    *,
    code: str,
    path: str,
) -> bool:
    return any(issue.code == code and issue.path == path for issue in issues)
