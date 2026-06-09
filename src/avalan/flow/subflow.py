from .definition import (
    FlowDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowOutputType,
)
from .loader import FlowDefinitionLoader, FlowLoadIssue
from .node import Node
from .plan import compile_flow_definition
from .registry import (
    FlowNodeConfigurationError,
    FlowNodeRegistry,
    FlowSubflowResolver,
    default_flow_node_registry,
)

from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

FLOW_SUBFLOW_NODE_TYPE = "subflow"


class LocalFlowSubflowResolver:
    def __init__(self, root: str | Path | None = None) -> None:
        if root is not None:
            assert isinstance(root, str | Path)
        self._root = Path(root).resolve() if root is not None else None
        self._cache: dict[Path, Mapping[str, object]] = {}
        self._compiling: set[Path] = set()

    def compile_subflow(
        self,
        ref: str,
        *,
        parent_definition: FlowDefinition,
        node: FlowNodeDefinition,
        registry: FlowNodeRegistry,
    ) -> Mapping[str, object]:
        assert isinstance(ref, str)
        assert isinstance(parent_definition, FlowDefinition)
        assert isinstance(node, FlowNodeDefinition)
        assert isinstance(registry, FlowNodeRegistry)
        output_mapping = _subflow_output_mapping(node)
        path = self._subflow_path(ref, parent_definition, node)
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        if path in self._compiling:
            raise _configuration_error(
                code="flow.subflow_cycle",
                path=f"nodes.{node.name}.ref",
                message="Subflow references form a cycle.",
                hint="Remove the recursive subflow reference.",
            )
        self._compiling.add(path)
        try:
            metadata = self._compile_path(
                path,
                node=node,
                registry=registry,
                output_mapping=output_mapping,
            )
        finally:
            self._compiling.discard(path)
        self._cache[path] = metadata
        return metadata

    def _compile_path(
        self,
        path: Path,
        *,
        node: FlowNodeDefinition,
        registry: FlowNodeRegistry,
        output_mapping: Mapping[str, str],
    ) -> Mapping[str, object]:
        loader = FlowDefinitionLoader(registry)
        load_result = loader.load_validation_result(path)
        if (
            load_result.definition is not None
            and not load_result.authoring_graph
        ):
            load_result = loader.load_result(path)
        if load_result.definition is None:
            assert load_result.issues
            raise _configuration_error(
                code="flow.invalid_subflow",
                path=f"nodes.{node.name}.ref",
                message="Subflow definition cannot be loaded.",
                hint=_invalid_subflow_load_hint(load_result.issues),
            )
        _validate_expectations(load_result.definition, node)
        _validate_input_mapping(load_result.definition, node)
        _validate_output_mapping(load_result.definition, node, output_mapping)
        compile_result = compile_flow_definition(
            load_result.definition,
            registry,
        )
        if compile_result.plan is None or compile_result.diagnostics:
            raise _configuration_error(
                code="flow.invalid_subflow",
                path=f"nodes.{node.name}.ref",
                message="Subflow definition cannot be compiled.",
                hint="Reference a valid strict flow definition.",
            )
        return MappingProxyType(
            {
                "plan": compile_result.plan,
                "output_mapping": output_mapping,
            }
        )

    def _subflow_path(
        self,
        ref: str,
        parent_definition: FlowDefinition,
        node: FlowNodeDefinition,
    ) -> Path:
        if _is_untrusted_ref(ref):
            raise _configuration_error(
                code="flow.path_escape",
                path=f"nodes.{node.name}.ref",
                message="Subflow reference escapes the flow directory.",
                hint=(
                    "Use a safe relative reference inside the flow directory."
                ),
            )
        if Path(ref).suffix != ".toml":
            raise _configuration_error(
                code="flow.invalid_ref",
                path=f"nodes.{node.name}.ref",
                message="Subflow node ref must be a local TOML file.",
                hint="Use a relative .toml file path.",
            )
        base = parent_definition.definition_base or self._root
        if base is None:
            raise _configuration_error(
                code="flow.subflow_untrusted_ref",
                path=f"nodes.{node.name}.ref",
                message="Subflow node has no trusted local base.",
                hint="Load the parent flow from a file or configure a root.",
            )
        base_path = Path(base).resolve()
        path = (base_path / ref).resolve()
        if not _is_relative_to(path, base_path):
            raise _configuration_error(
                code="flow.path_escape",
                path=f"nodes.{node.name}.ref",
                message="Subflow reference escapes the flow directory.",
                hint=(
                    "Use a safe relative reference inside the flow directory."
                ),
            )
        if self._root is not None and not _is_relative_to(path, self._root):
            raise _configuration_error(
                code="flow.path_escape",
                path=f"nodes.{node.name}.ref",
                message="Subflow reference escapes the trusted root.",
                hint="Use a safe relative reference inside the trusted root.",
            )
        if not path.is_file():
            raise _configuration_error(
                code="flow.subflow_not_found",
                path=f"nodes.{node.name}.ref",
                message="Subflow definition file cannot be found.",
                hint="Reference an existing local TOML file.",
            )
        return path


def subflow_node_registry(
    resolver: FlowSubflowResolver | None = None,
    *,
    base_registry: FlowNodeRegistry | None = None,
) -> FlowNodeRegistry:
    if resolver is not None:
        assert hasattr(resolver, "compile_subflow")
    if base_registry is not None:
        assert isinstance(base_registry, FlowNodeRegistry)
    registry = base_registry or default_flow_node_registry()
    subflow_resolver = resolver or LocalFlowSubflowResolver()
    registry.register(
        FLOW_SUBFLOW_NODE_TYPE,
        _subflow_node,
        metadata=FlowNodeMetadata(
            kind=FlowNodeKind.SUBFLOW,
            supports_ref=True,
            async_only=True,
            capabilities=(FlowNodeCapability.ASYNC_ONLY,),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.OBJECT,
                metadata={"dynamic": True},
            ),
            requires_ref=True,
        ),
        validator=_subflow_validator(subflow_resolver, registry),
    )
    registry.register_subflow_resolver(
        FLOW_SUBFLOW_NODE_TYPE,
        subflow_resolver,
    )
    return registry


def _subflow_node(definition: FlowNodeDefinition) -> Node:
    def run(_: dict[str, object]) -> object:
        raise FlowNodeConfigurationError(
            code="flow.execution.subflow_requires_plan",
            path=f"nodes.{definition.name}",
            message="Subflow node requires a compiled execution plan.",
            hint="Run subflow nodes through the strict flow executor.",
        )

    return Node(definition.name, func=run, async_only=True)


def _invalid_subflow_load_hint(
    issues: tuple[FlowLoadIssue, ...],
) -> str:
    assert issues
    issue = issues[0]
    return (
        "Reference a valid flow definition; first load issue is "
        f"{issue.code} at {issue.path}."
    )


def _subflow_validator(
    resolver: FlowSubflowResolver,
    registry: FlowNodeRegistry,
) -> Callable[
    [FlowDefinition, FlowNodeDefinition],
    tuple[FlowNodeConfigurationError, ...],
]:
    assert isinstance(registry, FlowNodeRegistry)

    def validate(
        definition: FlowDefinition,
        node: FlowNodeDefinition,
    ) -> tuple[FlowNodeConfigurationError, ...]:
        if node.ref is None or _is_path_escape_ref(node.ref):
            return ()
        try:
            resolver.compile_subflow(
                node.ref,
                parent_definition=definition,
                node=node,
                registry=registry,
            )
        except FlowNodeConfigurationError as error:
            return (error,)
        return ()

    return validate


def _subflow_output_mapping(
    node: FlowNodeDefinition,
) -> Mapping[str, str]:
    value = node.config.get("output_mapping")
    if not isinstance(value, Mapping) or not value:
        raise _configuration_error(
            code="flow.missing_subflow_output_mapping",
            path=f"nodes.{node.name}.config.output_mapping",
            message="Subflow node output mapping is missing.",
            hint="Map every subflow output into the node result.",
        )
    output_mapping: dict[str, str] = {}
    for target, source in value.items():
        if not isinstance(target, str) or not target.strip():
            raise _configuration_error(
                code="flow.invalid_subflow_output_mapping",
                path=f"nodes.{node.name}.config.output_mapping",
                message="Subflow output mapping target is invalid.",
                hint="Use non-empty result field names.",
            )
        if not isinstance(source, str) or not source.strip():
            raise _configuration_error(
                code="flow.invalid_subflow_output_mapping",
                path=f"nodes.{node.name}.config.output_mapping.{target}",
                message="Subflow output mapping source is invalid.",
                hint="Reference a declared subflow output.",
            )
        output_mapping[target] = source
    return MappingProxyType(output_mapping)


def _validate_expectations(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
) -> None:
    expected_version = _optional_string_config(node, "version")
    expected_revision = _optional_string_config(node, "revision")
    if expected_version is not None and definition.version != expected_version:
        raise _configuration_error(
            code="flow.subflow_version_mismatch",
            path=f"nodes.{node.name}.config.version",
            message="Subflow version does not match the expectation.",
            hint="Update the expected version or the referenced flow.",
        )
    if (
        expected_revision is not None
        and definition.revision != expected_revision
    ):
        raise _configuration_error(
            code="flow.subflow_revision_mismatch",
            path=f"nodes.{node.name}.config.revision",
            message="Subflow revision does not match the expectation.",
            hint="Update the expected revision or the referenced flow.",
        )


def _validate_input_mapping(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
) -> None:
    declared = {
        input_definition.name for input_definition in definition.inputs
    }
    mapped = {mapping.target for mapping in node.mappings}
    for target in sorted(mapped - declared):
        raise _configuration_error(
            code="flow.unknown_subflow_input",
            path=f"nodes.{node.name}.mapping.{target}",
            message="Subflow input mapping targets an unknown input.",
            hint="Map only declared subflow inputs.",
        )
    for target in sorted(declared - mapped):
        raise _configuration_error(
            code="flow.missing_subflow_input_mapping",
            path=f"nodes.{node.name}.mapping.{target}",
            message="Subflow input mapping is missing.",
            hint="Map every declared subflow input.",
        )
    by_name = {
        input_definition.name: input_definition
        for input_definition in definition.inputs
    }
    for mapping in node.mappings:
        _validate_input_mapping_kind(
            node,
            mapping,
            by_name[mapping.target].type,
        )


def _validate_input_mapping_kind(
    node: FlowNodeDefinition,
    mapping: FlowInputMapping,
    input_type: FlowInputType,
) -> None:
    compatible = (
        mapping.kind in {FlowMappingKind.SELECT, FlowMappingKind.RENAME}
        or mapping.kind == FlowMappingKind.COALESCE
        or (
            mapping.kind == FlowMappingKind.OBJECT
            and input_type == FlowInputType.OBJECT
        )
        or (
            mapping.kind == FlowMappingKind.ARRAY
            and input_type == FlowInputType.ARRAY
        )
        or (
            mapping.kind == FlowMappingKind.MERGE
            and input_type == FlowInputType.OBJECT
        )
        or (
            mapping.kind == FlowMappingKind.FILE
            and input_type == FlowInputType.FILE
        )
        or (
            mapping.kind == FlowMappingKind.FILE_ARRAY
            and input_type == FlowInputType.FILE_ARRAY
        )
    )
    if compatible:
        return
    raise _configuration_error(
        code="flow.incompatible_subflow_input_mapping",
        path=f"nodes.{node.name}.mapping.{mapping.target}",
        message="Subflow input mapping kind is incompatible.",
        hint="Use a mapping kind that matches the subflow input type.",
    )


def _validate_output_mapping(
    definition: FlowDefinition,
    node: FlowNodeDefinition,
    output_mapping: Mapping[str, str],
) -> None:
    declared = {
        output_definition.name for output_definition in definition.outputs
    }
    selected = set(output_mapping.values())
    for source in sorted(selected - declared):
        raise _configuration_error(
            code="flow.unknown_subflow_output",
            path=f"nodes.{node.name}.config.output_mapping",
            message="Subflow output mapping references an unknown output.",
            hint="Map only declared subflow outputs.",
        )
    missing = declared - selected
    if missing:
        output = sorted(missing)[0]
        raise _configuration_error(
            code="flow.missing_subflow_output_mapping",
            path=f"nodes.{node.name}.config.output_mapping.{output}",
            message="Subflow output mapping is missing a declared output.",
            hint="Map every declared subflow output.",
        )
    if len(selected) != len(output_mapping):
        raise _configuration_error(
            code="flow.duplicate_subflow_output_mapping",
            path=f"nodes.{node.name}.config.output_mapping",
            message="Subflow output mapping duplicates an output.",
            hint="Map each subflow output once.",
        )


def _optional_string_config(
    node: FlowNodeDefinition,
    key: str,
) -> str | None:
    value = node.config.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    raise _configuration_error(
        code="flow.invalid_subflow_expectation",
        path=f"nodes.{node.name}.config.{key}",
        message="Subflow expectation is invalid.",
        hint="Use a non-empty string expectation.",
    )


def _is_untrusted_ref(ref: str) -> bool:
    return _is_path_escape_ref(ref) or ":" in ref


def _is_path_escape_ref(ref: str) -> bool:
    if "://" in ref or "\\" in ref:
        return True
    posix_path = PurePosixPath(ref)
    windows_path = PureWindowsPath(ref)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _configuration_error(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
) -> FlowNodeConfigurationError:
    return FlowNodeConfigurationError(
        code=code,
        path=path,
        message=message,
        hint=hint,
    )
