from ..container import (
    ContainerDiagnostic,
    ContainerSurface,
    container_syntax_diagnostics,
)
from ..filesystem import (
    DEFAULT_TEXT_ENCODING,
    assert_text_encoding,
    read_text,
)
from .condition import (
    FlowCondition,
    FlowConditionEvaluationContext,
    FlowConditionOperator,
    FlowConditionValueType,
    evaluate_flow_condition,
)
from .definition import (
    FlowDefinition,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowEntryBehaviorType,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeDefinition,
    FlowOutputBehavior,
    FlowOutputBehaviorType,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowRouteMatchPolicy,
    FlowTimeoutPolicy,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)
from .flow import Flow
from .graph import (
    FlowGraphCompileResult,
    FlowGraphDiagnosticCode,
    FlowGraphEdgeBinding,
    FlowGraphFormat,
    FlowGraphInspection,
    FlowGraphMode,
    FlowGraphSource,
    FlowGraphSourceKind,
    compile_flow_graph,
    flow_graph_diagnostic,
    flow_graph_diagnostic_load_category,
)
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    default_flow_node_registry,
)
from .validator import (
    flow_validation_diagnostic_load_category,
    validate_flow_definition,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tomllib import TOMLDecodeError, loads
from typing import Any, TypeVar

EnumValue = TypeVar("EnumValue", bound=StrEnum)
RawSection = Mapping[str, object]

_ALLOWED_TOP_LEVEL_SECTIONS = frozenset(
    {
        "entry",
        "edges",
        "flow",
        "graph",
        "input",
        "inputs",
        "nodes",
        "observability",
        "output",
        "output_behavior",
        "outputs",
        "ownership",
        "privacy",
        "runtime_limits",
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
        "revision",
        "tags",
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
_ALLOWED_ENTRY_FIELDS = frozenset({"node", "type"})
_ALLOWED_OUTPUT_BEHAVIOR_FIELDS = frozenset({"outputs", "type"})
_ALLOWED_NODE_FIELDS = frozenset(
    {
        "config",
        "field",
        "input",
        "join_policy",
        "loop_policy",
        "mapping",
        "output",
        "path",
        "ref",
        "retry_policy",
        "timeout_policy",
        "type",
        "value",
    }
)
_ALLOWED_MAPPING_FIELDS = frozenset(
    {
        "fields",
        "items",
        "source",
        "sources",
        "type",
    }
)
_ALLOWED_JOIN_POLICY_FIELDS = frozenset(
    {
        "optional_inputs",
        "quorum",
        "type",
    }
)
_ALLOWED_RETRY_POLICY_FIELDS = frozenset(
    {
        "backoff",
        "exhausted_route",
        "initial_delay_seconds",
        "max_attempts",
        "max_delay_seconds",
        "non_retryable_categories",
        "retryable_categories",
    }
)
_ALLOWED_TIMEOUT_POLICY_FIELDS = frozenset({"per_attempt_seconds"})
_ALLOWED_LOOP_POLICY_FIELDS = frozenset(
    {
        "continue_condition",
        "exit_condition",
        "limit_route",
        "max_elapsed_seconds",
        "max_iterations",
        "output_selector",
    }
)
_ALLOWED_EDGE_FIELDS = frozenset(
    {
        "condition",
        "default",
        "kind",
        "label",
        "priority",
        "routing_policy",
        "source",
        "target",
    }
)
_ALLOWED_GRAPH_FIELDS = frozenset(
    {
        "diagram",
        "edges",
        "format",
        "mode",
        "path",
        "source",
    }
)
_ALLOWED_GRAPH_EDGE_FIELDS = _ALLOWED_EDGE_FIELDS - {"source", "target"}
_GRAPH_EDGE_METADATA_PATH = "graph.edges.metadata"
_ALLOWED_CONDITION_FIELDS = frozenset(
    {
        "condition",
        "conditions",
        "op",
        "selector",
        "value",
        "value_selector",
        "value_type",
        "values",
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
    source_span: FlowSourceSpan | None = None
    diagnostic_category: FlowDiagnosticCategory | None = None
    severity: FlowLoadSeverity = FlowLoadSeverity.ERROR

    def __post_init__(self) -> None:
        if self.source_span is not None:
            assert isinstance(self.source_span, FlowSourceSpan)
        if self.diagnostic_category is not None:
            assert isinstance(self.diagnostic_category, FlowDiagnosticCategory)

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "code": self.code,
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "hint": self.hint,
        }
        if self.source_span is not None:
            value["source_span"] = self.source_span.as_dict()
        return value

    def to_diagnostic(self) -> FlowDiagnostic:
        return FlowDiagnostic(
            code=self.code,
            path=self.path,
            category=self.diagnostic_category
            or _diagnostic_category(self.category),
            severity=FlowDiagnosticSeverity(self.severity.value),
            message=self.message,
            hint=self.hint,
            source_span=self.source_span,
        )

    def as_public_diagnostic_dict(self) -> dict[str, object]:
        return self.to_diagnostic().as_public_dict()


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowLoadResult:
    definition: FlowDefinition | None
    flow: Flow | None = None
    issues: tuple[FlowLoadIssue, ...] = ()
    authoring_graph: bool = False
    graph_inspection: FlowGraphInspection | None = None

    @property
    def ok(self) -> bool:
        return not self.issues and self.definition is not None

    @property
    def diagnostics(self) -> tuple[FlowDiagnostic, ...]:
        return tuple(issue.to_diagnostic() for issue in self.issues)

    @property
    def public_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            diagnostic.as_public_dict() for diagnostic in self.diagnostics
        )

    def __post_init__(self) -> None:
        assert isinstance(self.authoring_graph, bool)
        if self.graph_inspection is not None:
            assert isinstance(self.graph_inspection, FlowGraphInspection)


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
    def __init__(
        self,
        registry: FlowNodeRegistry | None = None,
        *,
        encoding: str = DEFAULT_TEXT_ENCODING,
    ) -> None:
        assert_text_encoding(encoding)
        self._registry = registry or default_flow_node_registry()
        self._encoding = encoding

    async def load(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> FlowDefinition:
        result = await self.load_result(path, encoding=encoding)
        if result.definition is None:
            raise FlowLoadError(result.issues)
        return result.definition

    async def load_result(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> FlowLoadResult:
        source_path = Path(path)
        text_encoding = self._text_encoding(encoding)
        source = await read_text(source_path, encoding=text_encoding)
        return await self.loads_result(
            source,
            source_path=source_path,
            encoding=text_encoding,
        )

    async def load_validation_result(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
    ) -> FlowLoadResult:
        source_path = Path(path)
        text_encoding = self._text_encoding(encoding)
        source = await read_text(source_path, encoding=text_encoding)
        return await self.loads_validation_result(
            source,
            source_path=source_path,
            encoding=text_encoding,
        )

    async def loads(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
        encoding: str | None = None,
    ) -> FlowDefinition:
        result = await self.loads_result(
            source,
            source_path=source_path,
            encoding=encoding,
        )
        if result.definition is None:
            raise FlowLoadError(result.issues)
        return result.definition

    async def loads_result(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
        encoding: str | None = None,
    ) -> FlowLoadResult:
        assert isinstance(source, str), "source must be a string"
        text_encoding = self._text_encoding(encoding)
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
        return await _build_result(
            raw,
            registry=self._registry,
            source_path=source_path,
            build_runtime=True,
            encoding=text_encoding,
        )

    async def loads_validation_result(
        self,
        source: str,
        *,
        source_path: str | Path | None = None,
        encoding: str | None = None,
    ) -> FlowLoadResult:
        assert isinstance(source, str), "source must be a string"
        text_encoding = self._text_encoding(encoding)
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
        return await _build_result(
            raw,
            registry=self._registry,
            source_path=source_path,
            build_runtime=False,
            encoding=text_encoding,
        )

    def _text_encoding(self, encoding: str | None) -> str:
        if encoding is None:
            return self._encoding
        assert_text_encoding(encoding)
        return encoding


async def load_flow_definition(
    path: str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowDefinition:
    return await FlowDefinitionLoader(encoding=encoding).load(path)


async def load_flow_definition_result(
    path: str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowLoadResult:
    return await FlowDefinitionLoader(encoding=encoding).load_result(path)


async def loads_flow_definition(
    source: str,
    *,
    source_path: str | Path | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowDefinition:
    return await FlowDefinitionLoader(encoding=encoding).loads(
        source,
        source_path=source_path,
    )


async def loads_flow_definition_result(
    source: str,
    *,
    source_path: str | Path | None = None,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> FlowLoadResult:
    return await FlowDefinitionLoader(encoding=encoding).loads_result(
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
        conditions: list[Callable[[Any], bool | Awaitable[bool]]] | None
        conditions = (
            [_edge_condition_callable(definition, node_registry, edge)]
            if edge.condition is not None
            else None
        )
        flow.add_connection(
            edge.source,
            edge.target,
            label=edge.label,
            conditions=conditions,
        )
    return flow


def _edge_condition_callable(
    definition: FlowDefinition,
    registry: FlowNodeRegistry,
    edge: FlowEdgeDefinition,
) -> Callable[[object], bool]:
    assert edge.condition is not None
    source = definition.node_map[edge.source]
    output_names = _condition_output_names(registry, source.type)

    def evaluate(data: object) -> bool:
        context = FlowConditionEvaluationContext(
            node_outputs={
                edge.source: {name: data for name in output_names},
            },
        )
        assert edge.condition is not None
        return evaluate_flow_condition(edge.condition, context)

    return evaluate


def _condition_output_names(
    registry: FlowNodeRegistry,
    node_type: str,
) -> tuple[str, ...]:
    metadata = registry.metadata(node_type)
    names: list[str] = []
    if metadata is not None:
        names.extend(
            contract.name
            for contract in metadata.output_contracts
            if contract.name is not None
        )
    if not names:
        names.extend(("value", "result"))
    return tuple(dict.fromkeys(names))


async def _build_result(
    raw: Mapping[str, object],
    *,
    registry: FlowNodeRegistry,
    source_path: str | Path | None,
    build_runtime: bool,
    encoding: str,
) -> FlowLoadResult:
    issues: list[FlowLoadIssue] = []
    issues.extend(_container_issues(raw))
    if issues:
        return FlowLoadResult(
            definition=None,
            issues=tuple(issues),
            authoring_graph="graph" in raw,
        )
    _validate_top_level_sections(raw, issues)
    flow_raw = _section(raw, "flow", issues, required=True)
    nodes_raw = _section(raw, "nodes", issues, required=True)
    graph_raw = _section(raw, "graph", issues, required=False)
    authoring_graph = "graph" in raw
    _validate_graph_section(graph_raw, issues)
    has_graph_edge_conflict = _validate_graph_edge_conflict(
        raw,
        graph_raw,
        issues,
    )
    if flow_raw is None or nodes_raw is None:
        return FlowLoadResult(
            definition=None,
            issues=tuple(issues),
            authoring_graph=authoring_graph,
        )

    _validate_unknown_fields(
        flow_raw,
        allowed=_ALLOWED_FLOW_FIELDS,
        path="flow",
        issues=issues,
    )
    input_raw = _child_section(raw, flow_raw, "input", issues)
    output_raw = _child_section(raw, flow_raw, "output", issues)
    entry_raw = _section(raw, "entry", issues, required=False)
    output_behavior_raw = _section(
        raw,
        "output_behavior",
        issues,
        required=False,
    )
    runtime_limits_raw = _section(
        raw,
        "runtime_limits",
        issues,
        required=False,
    )
    privacy_raw = _section(raw, "privacy", issues, required=False)
    observability_raw = _section(
        raw,
        "observability",
        issues,
        required=False,
    )
    ownership_raw = _section(raw, "ownership", issues, required=False)
    variables_raw = _section(raw, "variables", issues, required=False)
    name = _required_str(flow_raw, "flow.name", "name", issues)
    entrypoint = _optional_str(
        flow_raw,
        "flow.entrypoint",
        "entrypoint",
        issues,
    )
    output_node = _optional_str(
        flow_raw, "flow.output_node", "output_node", issues
    )
    version = _optional_str(flow_raw, "flow.version", "version", issues)
    revision = _optional_str(flow_raw, "flow.revision", "revision", issues)
    description = _optional_str(
        flow_raw,
        "flow.description",
        "description",
        issues,
    )
    tags = _string_tuple(flow_raw, "flow.tags", "tags", issues)
    input_definition = _input_definition(input_raw, issues)
    input_definitions = _input_definitions(raw.get("inputs"), issues)
    output_definition = _output_definition(output_raw, issues)
    output_definitions = _output_definitions(raw.get("outputs"), issues)
    entry_behavior = _entry_behavior(entry_raw, issues)
    output_behavior = _output_behavior(output_behavior_raw, issues)
    runtime_limits = _optional_metadata(
        runtime_limits_raw,
        "runtime_limits",
        issues,
    )
    privacy_policy = _optional_metadata(privacy_raw, "privacy", issues)
    observability_policy = _optional_metadata(
        observability_raw,
        "observability",
        issues,
    )
    ownership = _optional_metadata(ownership_raw, "ownership", issues)
    variables = (
        _metadata(variables_raw, "variables", issues)
        if variables_raw is not None
        else {}
    )
    nodes = _node_definitions(nodes_raw, issues)
    graph_inspection: FlowGraphInspection | None = None
    definition_base = (
        Path(source_path).parent if source_path is not None else None
    )
    edges = (
        ()
        if has_graph_edge_conflict
        else _edge_definitions(raw.get("edges"), issues)
    )
    is_strict = _uses_strict_definition(raw, flow_raw)
    if not is_strict:
        if entrypoint is None:
            issues.append(_missing_field("flow.entrypoint"))
        if output_node is None:
            issues.append(_missing_field("flow.output_node"))
    if name is None or (
        not is_strict and (entrypoint is None or output_node is None)
    ):
        return FlowLoadResult(
            definition=None,
            issues=tuple(issues),
            authoring_graph=authoring_graph,
        )
    if has_graph_edge_conflict:
        return FlowLoadResult(
            definition=None,
            issues=tuple(issues),
            authoring_graph=authoring_graph,
        )
    if graph_raw is not None:
        graph_result = await _compile_graph_edges(
            graph_raw,
            nodes,
            issues,
            definition_base=definition_base,
            source_path=source_path,
            encoding=encoding,
        )
        if graph_result is None:
            return FlowLoadResult(
                definition=None,
                issues=tuple(issues),
                authoring_graph=authoring_graph,
                graph_inspection=graph_inspection,
            )
        graph_inspection = graph_result.inspection
        if not graph_result.ok:
            return FlowLoadResult(
                definition=None,
                issues=tuple(issues),
                authoring_graph=authoring_graph,
                graph_inspection=graph_inspection,
            )
        edges = graph_result.edges
    definition = FlowDefinition(
        name=name,
        version=version,
        revision=revision,
        description=description,
        entrypoint=entrypoint,
        output_node=output_node,
        input=input_definition,
        inputs=input_definitions,
        output=output_definition,
        outputs=output_definitions,
        entry_behavior=entry_behavior,
        output_behavior=output_behavior,
        runtime_limits=runtime_limits or {},
        privacy_policy=privacy_policy or {},
        observability_policy=observability_policy or {},
        tags=tags,
        ownership=ownership or {},
        variables=variables or {},
        nodes=nodes,
        edges=edges,
        definition_base=definition_base,
    )
    validation_result = validate_flow_definition(definition, registry)
    issues.extend(
        _issue_from_diagnostic(diagnostic)
        for diagnostic in validation_result.diagnostics
    )
    if issues:
        return FlowLoadResult(
            definition=None,
            issues=tuple(issues),
            authoring_graph=authoring_graph,
            graph_inspection=graph_inspection,
        )
    if not build_runtime:
        return FlowLoadResult(
            definition=definition,
            authoring_graph=authoring_graph,
            graph_inspection=graph_inspection,
        )
    try:
        flow = build_flow(definition, registry)
    except FlowNodeConfigurationError as error:
        return FlowLoadResult(
            definition=None,
            issues=(
                _issue(
                    code=error.code,
                    path=error.path,
                    message=error.message,
                    hint=error.hint,
                    category=FlowLoadIssueCategory.VALUE,
                ),
            ),
            authoring_graph=authoring_graph,
            graph_inspection=graph_inspection,
        )
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
            authoring_graph=authoring_graph,
            graph_inspection=graph_inspection,
        )
    return FlowLoadResult(
        definition=definition,
        flow=flow,
        authoring_graph=authoring_graph,
        graph_inspection=graph_inspection,
    )


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
        if key in {"inputs", "outputs"}:
            if not isinstance(value, list):
                issues.append(_invalid_section_type(key))


def _uses_strict_definition(
    raw: Mapping[str, object],
    flow_raw: RawSection,
) -> bool:
    strict_sections = {
        "entry",
        "inputs",
        "observability",
        "output_behavior",
        "outputs",
        "ownership",
        "privacy",
        "runtime_limits",
    }
    return bool(
        strict_sections.intersection(raw)
        or "revision" in flow_raw
        or "tags" in flow_raw
        or _nodes_use_strict_definition(raw.get("nodes"))
    )


def _nodes_use_strict_definition(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(
        isinstance(raw, Mapping)
        and {
            "join_policy",
            "loop_policy",
            "mapping",
            "retry_policy",
            "timeout_policy",
        }.intersection(raw)
        for raw in value.values()
    )


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


def _validate_graph_section(
    raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> None:
    if raw is None:
        return
    _validate_unknown_fields(
        raw,
        allowed=_ALLOWED_GRAPH_FIELDS,
        path="graph",
        issues=issues,
        hint="Remove unsupported fields from the graph authoring table.",
    )
    if "edges" in raw:
        _validate_graph_edges_section(raw["edges"], issues)


def _validate_graph_edge_conflict(
    raw: Mapping[str, object],
    graph_raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> bool:
    if graph_raw is None or not isinstance(raw.get("edges"), list):
        return False
    issues.append(
        _issue_from_diagnostic(
            flow_graph_diagnostic(
                FlowGraphDiagnosticCode.EDGE_CONFLICT,
                "edges",
            )
        )
    )
    return True


def _validate_graph_edges_section(
    value: object,
    issues: list[FlowLoadIssue],
) -> None:
    if not isinstance(value, Mapping):
        issues.append(_invalid_section_type("graph.edges"))
        return
    for edge_id, raw in value.items():
        if not isinstance(edge_id, str) or not edge_id.strip():
            issues.append(
                _invalid_type("graph.edges", "Use named edge tables.")
            )
            continue
        if not isinstance(raw, Mapping):
            issues.append(_invalid_section_type(_GRAPH_EDGE_METADATA_PATH))
            continue
        _validate_unknown_fields(
            raw,
            allowed=_ALLOWED_GRAPH_EDGE_FIELDS,
            path=_GRAPH_EDGE_METADATA_PATH,
            issues=issues,
            hint="Remove unsupported fields from graph edge metadata.",
        )


async def _compile_graph_edges(
    raw: RawSection,
    nodes: tuple[FlowNodeDefinition, ...],
    issues: list[FlowLoadIssue],
    *,
    definition_base: Path | None,
    source_path: str | Path | None,
    encoding: str,
) -> FlowGraphCompileResult | None:
    if _has_graph_issue(issues):
        return None
    source = _graph_source(
        raw,
        issues,
        definition_base=definition_base,
        source_path=source_path,
    )
    edge_bindings = _graph_edge_bindings(raw, issues)
    if source is None or _has_graph_issue(issues):
        return None
    result = await compile_flow_graph(
        source,
        nodes,
        edge_bindings=edge_bindings,
        encoding=encoding,
    )
    issues.extend(
        _issue_from_diagnostic(diagnostic) for diagnostic in result.diagnostics
    )
    return result


def _graph_source(
    raw: RawSection,
    issues: list[FlowLoadIssue],
    *,
    definition_base: Path | None,
    source_path: str | Path | None,
) -> FlowGraphSource | None:
    graph_format = _graph_enum_value(
        raw,
        "graph.format",
        "format",
        FlowGraphFormat,
        FlowGraphDiagnosticCode.UNSUPPORTED_FORMAT,
        issues,
    )
    source_kind = _graph_enum_value(
        raw,
        "graph.source",
        "source",
        FlowGraphSourceKind,
        FlowGraphDiagnosticCode.UNSUPPORTED_SOURCE,
        issues,
    )
    mode = _graph_enum_value(
        raw,
        "graph.mode",
        "mode",
        FlowGraphMode,
        FlowGraphDiagnosticCode.UNSUPPORTED_MODE,
        issues,
    )
    diagram = _optional_str(raw, "graph.diagram", "diagram", issues)
    graph_path = _optional_path(raw, "graph.path", "path", issues)
    if graph_format is None or source_kind is None or mode is None:
        return None
    if _has_graph_issue(issues):
        return None
    if diagram is not None and graph_path is not None:
        issues.append(
            _issue_from_diagnostic(
                flow_graph_diagnostic(
                    FlowGraphDiagnosticCode.SOURCE_CONFLICT,
                    "graph.source",
                )
            )
        )
        return None
    source_identity = str(source_path) if source_path is not None else None
    match source_kind:
        case FlowGraphSourceKind.INLINE:
            if diagram is None:
                if graph_path is not None:
                    issues.append(
                        _issue_from_diagnostic(
                            flow_graph_diagnostic(
                                FlowGraphDiagnosticCode.SOURCE_CONFLICT,
                                "graph.source",
                            )
                        )
                    )
                    return None
                issues.append(_missing_graph_source())
                return None
            return FlowGraphSource(
                format=graph_format,
                source_kind=source_kind,
                mode=mode,
                diagram=diagram,
                source_identity=source_identity,
            )
        case FlowGraphSourceKind.FILE:
            if graph_path is None:
                if diagram is not None:
                    issues.append(
                        _issue_from_diagnostic(
                            flow_graph_diagnostic(
                                FlowGraphDiagnosticCode.SOURCE_CONFLICT,
                                "graph.source",
                            )
                        )
                    )
                    return None
                issues.append(_missing_graph_source())
                return None
            return FlowGraphSource(
                format=graph_format,
                source_kind=source_kind,
                mode=mode,
                path=graph_path,
                base_path=definition_base,
            )


def _graph_edge_bindings(
    raw: RawSection,
    issues: list[FlowLoadIssue],
) -> Mapping[str, FlowGraphEdgeBinding]:
    value = raw.get("edges")
    if value is None or not isinstance(value, Mapping):
        return {}
    bindings: dict[str, FlowGraphEdgeBinding] = {}
    for edge_id, edge_raw in value.items():
        assert isinstance(edge_id, str) and edge_id.strip()
        assert isinstance(edge_raw, Mapping)
        path = _GRAPH_EDGE_METADATA_PATH
        label = _optional_str(edge_raw, f"{path}.label", "label", issues)
        kind = _optional_enum_value(
            edge_raw,
            f"{path}.kind",
            "kind",
            FlowEdgeKind,
            issues,
        )
        condition = _condition_definition(
            edge_raw.get("condition"),
            issues,
            path=f"{path}.condition",
        )
        priority = _optional_int(
            edge_raw,
            f"{path}.priority",
            "priority",
            issues,
        )
        default = _optional_bool(
            edge_raw,
            f"{path}.default",
            "default",
            issues,
        )
        routing_policy = _optional_enum_value(
            edge_raw,
            f"{path}.routing_policy",
            "routing_policy",
            FlowRouteMatchPolicy,
            issues,
        )
        bindings[edge_id] = FlowGraphEdgeBinding(
            edge_id=edge_id,
            metadata={
                key: _metadata_value(value)
                for key, value in edge_raw.items()
                if isinstance(key, str) and key in _ALLOWED_GRAPH_EDGE_FIELDS
            },
            label=label,
            kind=kind,
            condition=condition,
            priority=priority,
            default=default,
            routing_policy=routing_policy,
        )
    return bindings


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


def _input_definitions(
    value: object,
    issues: list[FlowLoadIssue],
) -> tuple[FlowInputDefinition, ...]:
    definitions: list[FlowInputDefinition] = []
    for _, raw in _array_sections(value, "inputs", issues):
        definition = _input_definition(raw, issues)
        if definition is not None:
            definitions.append(definition)
    return tuple(definitions)


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


def _output_definitions(
    value: object,
    issues: list[FlowLoadIssue],
) -> tuple[FlowOutputDefinition, ...]:
    definitions: list[FlowOutputDefinition] = []
    for _, raw in _array_sections(value, "outputs", issues):
        definition = _output_definition(raw, issues)
        if definition is not None:
            definitions.append(definition)
    return tuple(definitions)


def _array_sections(
    value: object,
    path: str,
    issues: list[FlowLoadIssue],
) -> tuple[tuple[int, RawSection], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        return ()
    sections: list[tuple[int, RawSection]] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_invalid_section_type(item_path))
            continue
        sections.append((index, item))
    return tuple(sections)


def _entry_behavior(
    raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> FlowEntryBehavior | None:
    if raw is None:
        return None
    _validate_unknown_fields(
        raw,
        allowed=_ALLOWED_ENTRY_FIELDS,
        path="flow.entry",
        issues=issues,
    )
    entry_type = _enum_value(
        raw,
        "flow.entry.type",
        "type",
        FlowEntryBehaviorType,
        issues,
    )
    node = _required_str(raw, "flow.entry.node", "node", issues)
    if entry_type is None or node is None:
        return None
    return FlowEntryBehavior(type=entry_type, node=node)


def _output_behavior(
    raw: RawSection | None,
    issues: list[FlowLoadIssue],
) -> FlowOutputBehavior | None:
    if raw is None:
        return None
    _validate_unknown_fields(
        raw,
        allowed=_ALLOWED_OUTPUT_BEHAVIOR_FIELDS,
        path="flow.output_behavior",
        issues=issues,
    )
    behavior_type = _enum_value(
        raw,
        "flow.output_behavior.type",
        "type",
        FlowOutputBehaviorType,
        issues,
    )
    outputs = _string_mapping(
        raw,
        "flow.output_behavior.outputs",
        "outputs",
        issues,
    )
    if behavior_type is None or outputs is None:
        return None
    return FlowOutputBehavior(type=behavior_type, outputs=outputs)


def _node_definitions(
    raw: RawSection,
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
        mappings = _node_mappings(
            value.get("mapping"),
            issues,
            path=f"nodes.{name}.mapping",
        )
        join_policy = _join_policy(
            value.get("join_policy"),
            issues,
            path=f"nodes.{name}.join_policy",
        )
        retry_policy = _retry_policy(
            value.get("retry_policy"),
            issues,
            path=f"nodes.{name}.retry_policy",
        )
        timeout_policy = _timeout_policy(
            value.get("timeout_policy"),
            issues,
            path=f"nodes.{name}.timeout_policy",
        )
        loop_policy = _loop_policy(
            value.get("loop_policy"),
            issues,
            path=f"nodes.{name}.loop_policy",
        )
        if node_type is None:
            continue
        config = _node_config(value, issues, path=f"nodes.{name}")
        nodes.append(
            FlowNodeDefinition(
                name=name,
                type=node_type,
                ref=ref,
                input=input_name,
                output=output_name,
                join_policy=join_policy,
                retry_policy=retry_policy,
                timeout_policy=timeout_policy,
                loop_policy=loop_policy,
                mappings=mappings,
                config=config,
            )
        )
    return tuple(nodes)


def _container_issues(
    raw: Mapping[str, object],
) -> tuple[FlowLoadIssue, ...]:
    return tuple(
        _issue_from_container_diagnostic(diagnostic)
        for diagnostic in container_syntax_diagnostics(
            ContainerSurface.FLOW_TOML,
            raw,
        )
    )


def _issue_from_container_diagnostic(
    diagnostic: ContainerDiagnostic,
) -> FlowLoadIssue:
    return _issue(
        code=diagnostic.code.value,
        path=diagnostic.path,
        message=diagnostic.message,
        hint=diagnostic.hint,
        category=FlowLoadIssueCategory.UNSUPPORTED,
    )


def _node_mappings(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> tuple[FlowInputMapping, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return ()
    mappings: list[FlowInputMapping] = []
    for target, raw in value.items():
        target_path = f"{path}.{target}"
        if not isinstance(target, str) or not target.strip():
            issues.append(_invalid_type(path, "Use string keys."))
            continue
        if isinstance(raw, str):
            mappings.append(FlowInputMapping(target=target, source=raw))
            continue
        if not isinstance(raw, Mapping):
            issues.append(_invalid_type(target_path, "Use a TOML table."))
            continue
        _validate_unknown_fields(
            raw,
            allowed=_ALLOWED_MAPPING_FIELDS,
            path=target_path,
            issues=issues,
        )
        kind = _enum_value(
            raw,
            f"{target_path}.type",
            "type",
            FlowMappingKind,
            issues,
        )
        source = _optional_str(raw, f"{target_path}.source", "source", issues)
        sources = _string_tuple(
            raw,
            f"{target_path}.sources",
            "sources",
            issues,
        )
        fields = _optional_string_mapping(
            raw,
            f"{target_path}.fields",
            "fields",
            issues,
        )
        items = _string_tuple(
            raw,
            f"{target_path}.items",
            "items",
            issues,
        )
        if kind is None:
            continue
        mappings.append(
            FlowInputMapping(
                target=target,
                kind=kind,
                source=source,
                sources=sources,
                fields=fields or {},
                items=items,
            )
        )
    return tuple(mappings)


def _join_policy(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowJoinPolicy | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    _validate_unknown_fields(
        value,
        allowed=_ALLOWED_JOIN_POLICY_FIELDS,
        path=path,
        issues=issues,
    )
    policy_type = _enum_value(
        value,
        f"{path}.type",
        "type",
        FlowJoinPolicyType,
        issues,
    )
    quorum = _optional_int(
        value,
        f"{path}.quorum",
        "quorum",
        issues,
    )
    optional_inputs = _string_tuple(
        value,
        f"{path}.optional_inputs",
        "optional_inputs",
        issues,
    )
    if policy_type is None:
        return None
    return FlowJoinPolicy(
        type=policy_type,
        quorum=quorum,
        optional_inputs=optional_inputs,
    )


def _retry_policy(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowRetryPolicy | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    _validate_unknown_fields(
        value,
        allowed=_ALLOWED_RETRY_POLICY_FIELDS,
        path=path,
        issues=issues,
    )
    backoff = _optional_enum_value(
        value,
        f"{path}.backoff",
        "backoff",
        FlowRetryBackoffStrategy,
        issues,
    )
    max_attempts = _optional_int(
        value,
        f"{path}.max_attempts",
        "max_attempts",
        issues,
    )
    initial_delay_seconds = _optional_number(
        value,
        f"{path}.initial_delay_seconds",
        "initial_delay_seconds",
        issues,
    )
    max_delay_seconds = _optional_number(
        value,
        f"{path}.max_delay_seconds",
        "max_delay_seconds",
        issues,
    )
    retryable_categories = _string_tuple(
        value,
        f"{path}.retryable_categories",
        "retryable_categories",
        issues,
    )
    non_retryable_categories = _string_tuple(
        value,
        f"{path}.non_retryable_categories",
        "non_retryable_categories",
        issues,
    )
    exhausted_route = _optional_str(
        value,
        f"{path}.exhausted_route",
        "exhausted_route",
        issues,
    )
    return FlowRetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff or FlowRetryBackoffStrategy.NONE,
        initial_delay_seconds=initial_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        retryable_categories=retryable_categories,
        non_retryable_categories=non_retryable_categories,
        exhausted_route=exhausted_route,
    )


def _timeout_policy(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowTimeoutPolicy | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    _validate_unknown_fields(
        value,
        allowed=_ALLOWED_TIMEOUT_POLICY_FIELDS,
        path=path,
        issues=issues,
    )
    return FlowTimeoutPolicy(
        per_attempt_seconds=_optional_number(
            value,
            f"{path}.per_attempt_seconds",
            "per_attempt_seconds",
            issues,
        ),
    )


def _loop_policy(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowLoopPolicy | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    _validate_unknown_fields(
        value,
        allowed=_ALLOWED_LOOP_POLICY_FIELDS,
        path=path,
        issues=issues,
    )
    continue_condition = _condition_definition(
        value.get("continue_condition"),
        issues,
        path=f"{path}.continue_condition",
    )
    exit_condition = _condition_definition(
        value.get("exit_condition"),
        issues,
        path=f"{path}.exit_condition",
    )
    return FlowLoopPolicy(
        max_iterations=_optional_int(
            value,
            f"{path}.max_iterations",
            "max_iterations",
            issues,
        ),
        max_elapsed_seconds=_optional_number(
            value,
            f"{path}.max_elapsed_seconds",
            "max_elapsed_seconds",
            issues,
        ),
        continue_condition=continue_condition,
        exit_condition=exit_condition,
        output_selector=_optional_str(
            value,
            f"{path}.output_selector",
            "output_selector",
            issues,
        ),
        limit_route=_optional_str(
            value,
            f"{path}.limit_route",
            "limit_route",
            issues,
        ),
    )


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
        _validate_unknown_fields(
            item,
            allowed=_ALLOWED_EDGE_FIELDS,
            path=path,
            issues=issues,
        )
        source = _required_str(item, f"{path}.source", "source", issues)
        target = _required_str(item, f"{path}.target", "target", issues)
        label = _optional_str(item, f"{path}.label", "label", issues)
        kind = _optional_enum_value(
            item,
            f"{path}.kind",
            "kind",
            FlowEdgeKind,
            issues,
        )
        condition = _condition_definition(
            item.get("condition"),
            issues,
            path=f"{path}.condition",
        )
        priority = _optional_int(
            item,
            f"{path}.priority",
            "priority",
            issues,
        )
        default = _optional_bool(
            item,
            f"{path}.default",
            "default",
            issues,
        )
        routing_policy = _optional_enum_value(
            item,
            f"{path}.routing_policy",
            "routing_policy",
            FlowRouteMatchPolicy,
            issues,
        )
        if source is None or target is None:
            continue
        edges.append(
            FlowEdgeDefinition(
                source=source,
                target=target,
                label=label,
                kind=kind or FlowEdgeKind.SUCCESS,
                condition=condition,
                priority=0 if priority is None else priority,
                default=False if default is None else default,
                routing_policy=(
                    routing_policy or FlowRouteMatchPolicy.EXCLUSIVE
                ),
            )
        )
    return tuple(edges)


def _condition_definition(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowCondition | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    _validate_unknown_fields(
        value,
        allowed=_ALLOWED_CONDITION_FIELDS,
        path=path,
        issues=issues,
    )
    operator = _enum_value(
        value,
        f"{path}.op",
        "op",
        FlowConditionOperator,
        issues,
    )
    selector = _optional_str(value, f"{path}.selector", "selector", issues)
    condition_value = (
        _metadata_value(value["value"]) if "value" in value else None
    )
    value_selector = _optional_str(
        value,
        f"{path}.value_selector",
        "value_selector",
        issues,
    )
    values = _condition_values(value, issues, path=path)
    value_type = _optional_condition_value_type(value, issues, path=path)
    conditions = _condition_children(
        value.get("conditions"),
        issues,
        path=f"{path}.conditions",
    )
    condition = _condition_definition(
        value.get("condition"),
        issues,
        path=f"{path}.condition",
    )
    if operator is None:
        return None
    return FlowCondition(
        operator=operator,
        selector=selector,
        value=condition_value,
        value_selector=value_selector,
        values=values,
        value_type=value_type,
        conditions=conditions,
        condition=condition,
    )


def _condition_values(
    raw: RawSection,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> tuple[object, ...]:
    value = raw.get("values")
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        issues.append(_invalid_type(f"{path}.values", "Use an array."))
        return ()
    return tuple(_metadata_value(item) for item in value)


def _optional_condition_value_type(
    raw: RawSection,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> FlowConditionValueType | None:
    if "value_type" not in raw:
        return None
    return _enum_value(
        raw,
        f"{path}.value_type",
        "value_type",
        FlowConditionValueType,
        issues,
    )


def _condition_children(
    value: object,
    issues: list[FlowLoadIssue],
    *,
    path: str,
) -> tuple[FlowCondition, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        issues.append(_invalid_type(path, "Use an array of tables."))
        return ()
    conditions: list[FlowCondition] = []
    for index, item in enumerate(value):
        condition = _condition_definition(
            item,
            issues,
            path=f"{path}[{index}]",
        )
        if condition is not None:
            conditions.append(condition)
    return tuple(conditions)


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


def _validate_unknown_fields(
    raw: RawSection,
    *,
    allowed: frozenset[str],
    path: str,
    issues: list[FlowLoadIssue],
    hint: str = "Remove unsupported fields from the native flow.",
) -> None:
    for key in raw:
        if isinstance(key, str) and key not in allowed:
            issues.append(
                _issue(
                    code="flow.unsupported_field",
                    path=f"{path}.{key}",
                    message="Flow TOML field is not supported.",
                    hint=hint,
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


def _optional_path(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> Path | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        issues.append(_invalid_type(path, "Use a non-empty string."))
        return None
    return Path(value)


def _graph_enum_value(
    raw: RawSection,
    path: str,
    field: str,
    enum_type: type[EnumValue],
    diagnostic_code: FlowGraphDiagnosticCode,
    issues: list[FlowLoadIssue],
) -> EnumValue | None:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        issues.append(
            _issue_from_diagnostic(
                flow_graph_diagnostic(diagnostic_code, path)
            )
        )
        return None
    try:
        return enum_type(value)
    except ValueError:
        issues.append(
            _issue_from_diagnostic(
                flow_graph_diagnostic(diagnostic_code, path)
            )
        )
        return None


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


def _optional_enum_value(
    raw: RawSection,
    path: str,
    field: str,
    enum_type: type[EnumValue],
    issues: list[FlowLoadIssue],
) -> EnumValue | None:
    if field not in raw:
        return None
    return _enum_value(raw, path, field, enum_type, issues)


def _optional_int(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> int | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(_invalid_type(path, "Use an integer."))
        return None
    return value


def _optional_number(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> int | float | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        issues.append(_invalid_type(path, "Use a number."))
        return None
    return value


def _optional_bool(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> bool | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, bool):
        issues.append(_invalid_type(path, "Use true or false."))
        return None
    return value


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


def _string_mapping(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> Mapping[str, str] | None:
    value = raw.get(field)
    if value is None:
        issues.append(_missing_field(path))
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    mapping: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            issues.append(_invalid_type(path, "Use string keys."))
            return None
        if not isinstance(item, str) or not item.strip():
            issues.append(_invalid_type(f"{path}.{key}", "Use a string."))
            return None
        mapping[key] = item
    return mapping


def _optional_string_mapping(
    raw: RawSection,
    path: str,
    field: str,
    issues: list[FlowLoadIssue],
) -> Mapping[str, str] | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        issues.append(_invalid_type(path, "Use a TOML table."))
        return None
    mapping: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            issues.append(_invalid_type(path, "Use string keys."))
            return None
        if not isinstance(item, str) or not item.strip():
            issues.append(_invalid_type(f"{path}.{key}", "Use a string."))
            return None
        mapping[key] = item
    return mapping


def _optional_metadata(
    raw: RawSection | None,
    path: str,
    issues: list[FlowLoadIssue],
) -> Mapping[str, object] | None:
    if raw is None:
        return None
    return _metadata(raw, path, issues)


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


def _missing_graph_source() -> FlowLoadIssue:
    return _issue_from_diagnostic(
        flow_graph_diagnostic(
            FlowGraphDiagnosticCode.MISSING_SOURCE,
            "graph.source",
        )
    )


def _unsupported_section_issue(path: str) -> FlowLoadIssue:
    return _issue(
        code="flow.unsupported_section",
        path=path,
        message="Flow TOML section is not supported.",
        hint="Remove unsupported runtime sections from the native flow.",
        category=FlowLoadIssueCategory.UNSUPPORTED,
    )


def _issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    category: FlowLoadIssueCategory,
    source_span: FlowSourceSpan | None = None,
    diagnostic_category: FlowDiagnosticCategory | None = None,
) -> FlowLoadIssue:
    return FlowLoadIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=category,
        source_span=source_span,
        diagnostic_category=diagnostic_category,
    )


def _issue_from_diagnostic(diagnostic: FlowDiagnostic) -> FlowLoadIssue:
    assert isinstance(diagnostic, FlowDiagnostic)
    assert diagnostic.path is not None
    if diagnostic.category == FlowDiagnosticCategory.GRAPH_COMPILER:
        category = FlowLoadIssueCategory(
            flow_graph_diagnostic_load_category(diagnostic)
        )
        diagnostic_category = diagnostic.category
    else:
        category = FlowLoadIssueCategory(
            flow_validation_diagnostic_load_category(diagnostic)
        )
        diagnostic_category = None
    return _issue(
        code=diagnostic.code,
        path=diagnostic.path,
        message=diagnostic.message,
        hint=diagnostic.hint or "Fix the flow definition.",
        category=category,
        source_span=diagnostic.source_span,
        diagnostic_category=diagnostic_category,
    )


def _diagnostic_category(
    category: FlowLoadIssueCategory,
) -> FlowDiagnosticCategory:
    if category == FlowLoadIssueCategory.PRIVACY:
        return FlowDiagnosticCategory.PRIVACY
    return FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION


def _has_issue(
    issues: list[FlowLoadIssue],
    *,
    code: str,
    path: str,
) -> bool:
    return any(issue.code == code and issue.path == path for issue in issues)


def _has_graph_issue(issues: list[FlowLoadIssue]) -> bool:
    return any(
        issue.path == "graph" or issue.path.startswith("graph.")
        for issue in issues
    )
