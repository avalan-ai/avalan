from ..entities import (
    TOOL_DISPLAY_PROJECTOR_METADATA_KEY,
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolCapabilities,
    ToolDescriptor,
    ToolDescriptorMetadataValue,
    ToolDisplayProjector,
    ToolFilter,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolFormat,
    ToolManagerExecutionMode,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
    ToolProviderArgumentsMode,
    ToolTransformer,
    ToolTransformerResult,
)
from ..interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from ..model.capability import _copy_model_capability_json
from ..skill.bootstrap import skills_bootstrap_prompt
from ..skill.settings import SkillBootstrapPromptSettings
from . import Tool, ToolSet
from .json_schema import get_json_schema
from .name_policy import ToolNamePolicy
from .names import matches_tool_namespace
from .parser import ToolCallParser

from asyncio import CancelledError, wait_for
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from inspect import Parameter, signature
from types import TracebackType
from typing import Any, cast
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class _ToolCandidate:
    canonical_name: str
    tool: Callable[..., Any] | Tool
    aliases: tuple[str, ...]
    namespace: str | None
    schema_prefix: str


@dataclass(frozen=True, slots=True, eq=False)
class _ToolSemanticIdentity:
    canonical_name: str
    tool: Callable[..., Any] | Tool = field(repr=False)
    aliases: tuple[str, ...]
    namespace: str | None
    schema_prefix: str
    descriptor: ToolDescriptor = field(repr=False)
    schema_semantics: object = field(repr=False)
    parameter_schema_semantics: object = field(repr=False)
    return_schema_semantics: object = field(repr=False)
    provider_safe_schema_semantics: object = field(repr=False)
    capabilities: ToolCapabilities
    policy_semantics: object = field(repr=False)
    metadata_semantics: object = field(repr=False)
    display_projector_owner: object | None = field(repr=False)
    display_projector_implementation: object | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ToolRegistration:
    canonical_name: str
    tool: Callable[..., Any]
    aliases: tuple[str, ...]
    namespace: str | None
    descriptor: ToolDescriptor


@dataclass(frozen=True, slots=True)
class _ToolsetRegistrationPlan:
    toolset: ToolSet
    prefix: str | None
    enable_tools: tuple[str, ...] | None
    available_identities: tuple[_ToolSemanticIdentity, ...]
    identities: tuple[_ToolSemanticIdentity, ...]
    registrations: tuple[_ToolRegistration, ...]
    bootstrap_enabled: bool
    bootstrap_prompt_settings: SkillBootstrapPromptSettings | None


@dataclass(frozen=True, slots=True)
class _ToolManagerRegistryState:
    tools: dict[str, Callable[..., Any]]
    aliases: dict[str, list[str]]
    available_aliases: dict[str, list[str]]
    available_tool_names: set[str]
    bootstrap_tool_names: set[str]
    descriptors: dict[str, ToolDescriptor]
    skills_bootstrap_prompt_settings: SkillBootstrapPromptSettings | None
    toolsets: list[ToolSet]


@dataclass(frozen=True, slots=True)
class _ToolsetToolsSnapshot:
    toolset: ToolSet
    tools: list[Callable[..., Any] | Tool | ToolSet]
    contents: tuple[Callable[..., Any] | Tool | ToolSet, ...]


@dataclass(frozen=True, slots=True)
class _AdvertisedToolSources:
    configured: tuple[Callable[..., Any] | Tool | ToolSet, ...]
    available: tuple[Callable[..., Any] | Tool | ToolSet, ...] | None
    advertised: tuple[Callable[..., Any] | Tool | ToolSet, ...] | None
    selected: tuple[Callable[..., Any] | Tool | ToolSet, ...] | None
    custom_selector: bool
    selector_implementation: object = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class _ToolsetInventorySnapshot:
    configured: tuple[_ToolSemanticIdentity, ...]
    available: tuple[_ToolSemanticIdentity, ...] | None
    advertised: tuple[_ToolSemanticIdentity, ...] | None
    selected: tuple[_ToolSemanticIdentity, ...] | None
    custom_selector: bool
    selector_implementation: object = field(repr=False)


class ToolManager:
    _INTERRUPT_CLOSE_TIMEOUT = 0.5

    _parser: ToolCallParser
    _stack: AsyncExitStack
    _aliases: dict[str, list[str]]
    _available_aliases: dict[str, list[str]]
    _available_tool_names: set[str]
    _bootstrap_tool_names: set[str]
    _descriptors: dict[str, ToolDescriptor]
    _registration_plans: list[_ToolsetRegistrationPlan]
    _skills_bootstrap_prompt_settings: SkillBootstrapPromptSettings | None
    _tools: dict[str, Callable[..., Any]] | None = None
    _toolsets: list[ToolSet] | None = None

    @classmethod
    def create_instance(
        cls,
        *,
        available_toolsets: Sequence[ToolSet] | None = None,
        enable_tools: list[str] | None = None,
        settings: ToolManagerSettings | None = None,
    ) -> "ToolManager":
        parser = ToolCallParser(
            eos_token=settings.eos_token if settings else None,
            recovery_formats=settings.recovery_formats if settings else None,
            tool_format=settings.tool_format if settings else None,
            maximum_payload_depth=(
                settings.maximum_parser_payload_depth if settings else None
            ),
            maximum_payload_size=(
                settings.maximum_parser_payload_size if settings else None
            ),
            maximum_text_size=(
                settings.maximum_parser_input_size if settings else None
            ),
        )
        return cls(
            available_toolsets=available_toolsets,
            enable_tools=enable_tools,
            parser=parser,
            settings=settings,
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self._tools)

    @property
    def tools(self) -> list[Callable[..., Any]] | None:
        return list(self._tools.values()) if self._tools else None

    @property
    def tool_format(self) -> ToolFormat | None:
        """Return the tool format configured for this manager."""
        return self._parser.tool_format

    @property
    def parallel_tool_calls(self) -> bool:
        """Return whether parallel tool execution is enabled."""
        return self._settings.parallel_tool_calls

    @property
    def maximum_parallel_tool_calls(self) -> int:
        """Return the configured parallel tool execution limit."""
        return self._settings.maximum_parallel_tool_calls

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for enabled tools."""
        return list(self._descriptors.values())

    def export_model_capability_seed(self) -> dict[str, Any]:
        """Return an isolated callable-free model capability seed."""
        name_policy = self._settings.tool_name_policy
        descriptors: list[dict[str, Any]] = []
        for descriptor in self._descriptors.values():
            function_schema = (
                descriptor.schema.get("function")
                if isinstance(descriptor.schema, dict)
                else None
            )
            description = (
                function_schema.get("description")
                if isinstance(function_schema, dict)
                else None
            )
            if not isinstance(description, str) or not description.strip():
                description = f"Invoke {descriptor.name}."
            parameter_schema = descriptor.parameter_schema or {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            }
            descriptors.append(
                {
                    "canonical_name": descriptor.name,
                    "description": description,
                    "aliases": list(descriptor.aliases),
                    "parameter_schema": _copy_model_capability_json(
                        parameter_schema,
                        f"descriptor {descriptor.name!r} parameter schema",
                    ),
                    "result_schema": (
                        None
                        if descriptor.return_schema is None
                        else _copy_model_capability_json(
                            descriptor.return_schema,
                            f"descriptor {descriptor.name!r} result schema",
                        )
                    ),
                }
            )
        return {
            "version": 1,
            "name_policy": {
                "mode": name_policy.mode.value,
                "prefix": name_policy.prefix,
                "replacement": name_policy.replacement,
                "collapse_replacement": name_policy.collapse_replacement,
                "map": dict(name_policy.map),
                "provider_family": name_policy.provider_family,
            },
            "parser": {
                "tool_format": (
                    None
                    if self.tool_format is None
                    else self.tool_format.value
                ),
                "eos_token": cast(
                    str | None,
                    getattr(self._parser, "_eos_token", None),
                ),
                "recovery_formats": [
                    recovery_format.value
                    for recovery_format in self._parser.recovery_formats
                ],
                "maximum_input_size": self._settings.maximum_parser_input_size,
                "maximum_payload_depth": (
                    self._settings.maximum_parser_payload_depth
                ),
                "maximum_payload_size": (
                    self._settings.maximum_parser_payload_size
                ),
            },
            "descriptors": descriptors,
        }

    def bootstrap_prompt(self) -> str | None:
        """Return compact prompt text for enabled toolsets."""
        return skills_bootstrap_prompt(
            tuple(
                tool_name
                for tool_name in self._descriptors
                if tool_name in self._bootstrap_tool_names
            ),
            settings=self._skills_bootstrap_prompt_settings,
        )

    def describe_tool(self, name: str) -> ToolDescriptor | None:
        """Return the descriptor for an enabled tool."""
        resolution = self.resolve_tool_name(name)
        if resolution.canonical_name is None:
            return None
        return self._descriptors.get(resolution.canonical_name)

    def describe_tool_call(self, call: ToolCall) -> ToolDescriptor | None:
        """Return the descriptor for an executable tool call."""
        assert isinstance(call, ToolCall)
        resolution = self._resolve_call_name(call)
        if resolution.canonical_name is None:
            return None
        return self._descriptors.get(resolution.canonical_name)

    def is_tool_call_parallel_safe(self, call: ToolCall) -> bool:
        """Return whether ``call`` may execute in a parallel fanout."""
        descriptor = self.describe_tool_call(call)
        if descriptor is None:
            return False
        return descriptor.capabilities.parallel_safe

    def resolve_tool_name(self, name: str) -> ToolNameResolution:
        """Resolve a requested tool name against enabled tools."""
        assert isinstance(name, str)
        assert name.strip(), "name must not be empty"
        canonical_request = name.removeprefix("functions.")
        if self._tools and canonical_request in self._tools:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.EXACT,
                canonical_name=canonical_request,
                candidates=[canonical_request],
            )

        if canonical_request in self._available_tool_names:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.DISABLED,
                candidates=[canonical_request],
                diagnostic_code=ToolCallDiagnosticCode.DISABLED_TOOL,
            )

        aliases = self._aliases.get(canonical_request, [])
        if len(aliases) == 1:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.ALIAS,
                canonical_name=aliases[0],
                candidates=aliases,
            )
        if len(aliases) > 1:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.AMBIGUOUS,
                candidates=aliases,
                diagnostic_code=ToolCallDiagnosticCode.AMBIGUOUS_TOOL_NAME,
            )

        if canonical_request in self._available_aliases:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.DISABLED,
                candidates=self._available_aliases[canonical_request],
                diagnostic_code=ToolCallDiagnosticCode.DISABLED_TOOL,
            )

        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
            diagnostic_code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        """Return a diagnostic when a tool call is not executable."""
        resolution = self._resolve_call_name(call)
        if resolution.diagnostic_code is not None:
            return ToolCallDiagnostic(
                id=uuid4(),
                call_id=call.id,
                requested_name=resolution.requested_name,
                canonical_name=resolution.canonical_name,
                code=resolution.diagnostic_code,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message=(
                    f"Tool '{resolution.requested_name}' is "
                    f"{resolution.status.value}."
                ),
                details={"candidates": cast(Any, resolution.candidates)},
            )

        assert resolution.canonical_name is not None
        provider_diagnostic = self._provider_arguments_diagnostic(
            call=call,
            canonical_name=resolution.canonical_name,
        )
        if provider_diagnostic is not None:
            return provider_diagnostic

        arguments = call.arguments if call.arguments is not None else {}
        if not isinstance(arguments, dict):
            return ToolCallDiagnostic(
                id=uuid4(),
                call_id=call.id,
                requested_name=call.name,
                canonical_name=resolution.canonical_name,
                code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                stage=ToolCallDiagnosticStage.VALIDATE,
                message="Tool call arguments must be an object.",
            )

        tool = (
            self._tools.get(resolution.canonical_name) if self._tools else None
        )
        assert tool is not None
        arguments = self._normalize_single_input_argument(tool, arguments)

        diagnostic = self._validate_argument_limits(
            call=call,
            canonical_name=resolution.canonical_name,
            arguments=arguments,
        )
        if diagnostic is not None:
            return diagnostic

        assert resolution.canonical_name is not None
        return self._validate_tool_arguments(
            call=call,
            canonical_name=resolution.canonical_name,
            tool=tool,
            arguments=arguments,
        )

    def __init__(
        self,
        *,
        available_toolsets: Sequence[ToolSet] | None = None,
        enable_tools: list[str] | None = None,
        parser: ToolCallParser,
        settings: ToolManagerSettings | None = None,
    ):
        resolved_settings = settings or ToolManagerSettings()
        toolsets = list(available_toolsets or ())
        stack = AsyncExitStack()
        snapshots = self._snapshot_toolset_tools(toolsets)
        try:
            advertised_inventories = [
                self._toolset_inventory_snapshot(
                    toolset,
                    enable_tools=enable_tools,
                )
                for toolset in toolsets
            ]
            self._validate_model_capability_names(
                [
                    entry
                    for inventory in advertised_inventories
                    for entry in self._snapshot_advertised_names(inventory)
                ],
                resolved_settings,
            )
            verified_advertised_inventories = [
                self._toolset_inventory_snapshot(
                    toolset,
                    enable_tools=enable_tools,
                    validate_inventory=True,
                )
                for toolset in toolsets
            ]
            assert len(advertised_inventories) == len(
                verified_advertised_inventories
            ) and all(
                self._same_inventory_snapshot(original, verified)
                for original, verified in zip(
                    advertised_inventories,
                    verified_advertised_inventories,
                    strict=True,
                )
            ), "advertised tool inventory changed during configuration"
            assert self._toolset_tools_match_snapshots(
                snapshots
            ), "advertised selector inventory must be pure"
            selected_toolsets: list[ToolSet] = []
            for toolset in toolsets:
                selected = (
                    toolset.with_enabled_tools(enable_tools)
                    if enable_tools is not None
                    else toolset
                )
                assert isinstance(
                    selected, ToolSet
                ), "with_enabled_tools must return a ToolSet"
                selected_toolsets.append(selected)
            plans = [
                self._registration_plan(
                    toolset,
                    available_identities=self._advertised_inventory(inventory),
                    expected_identities=self._permitted_inventory(
                        inventory,
                        enable_tools,
                    ),
                    enable_tools=(
                        None if enable_tools is None else tuple(enable_tools)
                    ),
                    settings=resolved_settings,
                )
                for toolset, inventory in zip(
                    selected_toolsets,
                    advertised_inventories,
                    strict=True,
                )
            ]
            state = self._registry_state(plans, resolved_settings)
        except BaseException:
            self._restore_toolset_tools(snapshots)
            raise

        self._parser = parser
        self._settings = resolved_settings
        self._stack = stack
        self._registration_plans = plans
        self._commit_registry_state(state)

    def set_eos_token(self, eos_token: str) -> None:
        self._parser.set_eos_token(eos_token)

    @classmethod
    def _add_aliases(
        cls,
        registry: dict[str, list[str]],
        canonical_name: str,
        aliases: list[str],
    ) -> None:
        for alias in aliases:
            registry.setdefault(alias, []).append(canonical_name)

    @staticmethod
    def _validate_reserved_model_capability_name(
        canonical_name: str,
        aliases: Sequence[str],
    ) -> None:
        if (
            canonical_name == RESERVED_INPUT_CAPABILITY_NAME
            or RESERVED_INPUT_CAPABILITY_NAME in aliases
        ):
            raise ValueError(
                "tool configuration uses reserved model capability name "
                f"{RESERVED_INPUT_CAPABILITY_NAME!r}"
            )

    @classmethod
    def _validate_model_capability_names(
        cls,
        entries: Sequence[tuple[str, Sequence[str]]],
        settings: ToolManagerSettings,
    ) -> None:
        policy = ToolNamePolicy(settings=settings.tool_name_policy)
        for (
            canonical_name,
            provider_name,
        ) in settings.tool_name_policy.map.items():
            cls._validate_reserved_model_capability_name(canonical_name, ())
            cls._validate_reserved_model_capability_name(provider_name, ())

        for canonical_name, aliases in entries:
            cls._validate_reserved_model_capability_name(
                canonical_name,
                aliases,
            )
            for identity in (canonical_name, *aliases):
                if (
                    policy.provider_name(identity)
                    == RESERVED_INPUT_CAPABILITY_NAME
                ):
                    cls._raise_reserved_model_capability_name()

    @staticmethod
    def _raise_reserved_model_capability_name() -> None:
        raise ValueError(
            "tool configuration uses reserved model capability name "
            f"{RESERVED_INPUT_CAPABILITY_NAME!r}"
        )

    @classmethod
    def _tool_aliases(cls, tool: Callable[..., Any] | Tool) -> list[str]:
        aliases = getattr(tool, "aliases", [])
        assert isinstance(aliases, Sequence)
        assert not isinstance(aliases, str)
        return [cls._valid_tool_alias(alias) for alias in aliases]

    @staticmethod
    def _valid_tool_alias(alias: object) -> str:
        assert isinstance(alias, str), "tool aliases must be strings"
        assert alias.strip(), "tool aliases must not be empty"
        return alias

    @staticmethod
    def _valid_tool_name(tool: Callable[..., Any] | Tool) -> str:
        name = getattr(tool, "__name__", tool.__class__.__name__)
        assert isinstance(name, str), "tool names must be strings"
        assert name.strip(), "tool names must not be empty"
        return name

    @staticmethod
    def _inspect_tool_sequence(
        value: object,
        label: str,
    ) -> list[Callable[..., Any] | Tool | ToolSet]:
        assert isinstance(value, Sequence) and not isinstance(
            value, str | bytes
        ), f"{label} must be a sequence"
        tools = list(value)
        for tool in tools:
            assert isinstance(tool, ToolSet) or callable(
                tool
            ), f"{label} must contain only tools or toolsets"
        return cast(list[Callable[..., Any] | Tool | ToolSet], tools)

    @classmethod
    def _advertised_tool_sources(
        cls,
        toolset: ToolSet,
        enable_tools: Sequence[str] | None,
        *,
        validate_inventory: bool,
    ) -> _AdvertisedToolSources:
        missing = object()
        configured_tools = tuple(
            cls._inspect_tool_sequence(toolset.tools, "toolset tools")
        )
        available_tools = getattr(toolset, "available_tools", missing)
        selected_inventory = getattr(
            toolset,
            "available_tools_for_enabled_tools",
            missing,
        )
        advertised_inventory = getattr(
            toolset,
            "advertised_tools_for_enabled_tools",
            missing,
        )
        selected_inventory_getter: Callable[[Sequence[str]], object] | None = (
            None
        )
        advertised_inventory_getter: (
            Callable[[Sequence[str]], object] | None
        ) = None
        if selected_inventory is not missing:
            assert callable(
                selected_inventory
            ), "available_tools_for_enabled_tools must be callable"
            selected_inventory_getter = cast(
                Callable[[Sequence[str]], object],
                selected_inventory,
            )
        if advertised_inventory is not missing:
            assert callable(
                advertised_inventory
            ), "advertised_tools_for_enabled_tools must be callable"
            advertised_inventory_getter = cast(
                Callable[[Sequence[str]], object],
                advertised_inventory,
            )
        available = (
            None
            if available_tools is missing
            else tuple(
                cls._inspect_tool_sequence(
                    available_tools,
                    "available_tools",
                )
            )
        )
        selected = (
            None
            if selected_inventory_getter is None or enable_tools is None
            else tuple(
                cls._inspect_tool_sequence(
                    selected_inventory_getter(enable_tools),
                    "available_tools_for_enabled_tools",
                )
            )
        )
        advertised = (
            None
            if advertised_inventory_getter is None or enable_tools is None
            else tuple(
                cls._inspect_tool_sequence(
                    advertised_inventory_getter(enable_tools),
                    "advertised_tools_for_enabled_tools",
                )
            )
        )
        selector = getattr(toolset, "with_enabled_tools")
        assert callable(selector), "with_enabled_tools must be callable"
        selector_implementation = getattr(selector, "__func__", selector)
        custom_selector = (
            selector_implementation is not ToolSet.with_enabled_tools
        )
        if validate_inventory:
            if selected_inventory_getter is not None:
                assert available is not None, (
                    "available_tools_for_enabled_tools requires a complete "
                    "available_tools inventory"
                )
            if enable_tools is not None and custom_selector:
                assert (
                    available is not None and selected is not None
                ), "custom with_enabled_tools requires complete inventory APIs"
        return _AdvertisedToolSources(
            configured=configured_tools,
            available=available,
            advertised=advertised,
            selected=selected,
            custom_selector=custom_selector,
            selector_implementation=selector_implementation,
        )

    @classmethod
    def _tool_candidates(
        cls,
        toolset: ToolSet,
        prefix: str | None = None,
        *,
        active_toolsets: frozenset[int] = frozenset(),
    ) -> list[_ToolCandidate]:
        toolset_id = id(toolset)
        assert (
            toolset_id not in active_toolsets
        ), "toolset nesting must be acyclic"
        active_toolsets |= {toolset_id}
        namespace = (
            f"{prefix}.{toolset.namespace}"
            if prefix and toolset.namespace
            else prefix or toolset.namespace
        )
        schema_prefix = f"{namespace}." if namespace else ""
        candidates: list[_ToolCandidate] = []
        tools = cls._inspect_tool_sequence(toolset.tools, "toolset tools")
        for tool in tools:
            if isinstance(tool, ToolSet):
                candidates.extend(
                    cls._tool_candidates(
                        tool,
                        namespace,
                        active_toolsets=active_toolsets,
                    )
                )
                continue
            name = cls._valid_tool_name(tool)
            candidates.append(
                _ToolCandidate(
                    canonical_name=f"{schema_prefix}{name}",
                    tool=tool,
                    aliases=tuple(cls._tool_aliases(tool)),
                    namespace=namespace,
                    schema_prefix=schema_prefix,
                )
            )
        return candidates

    @classmethod
    def _tool_semantic_identity(
        cls,
        candidate: _ToolCandidate,
        *,
        include_metadata: bool = True,
    ) -> _ToolSemanticIdentity:
        schema = cls._tool_schema(candidate.tool, candidate.schema_prefix)
        descriptor = cls._tool_descriptor(
            canonical_name=candidate.canonical_name,
            tool=candidate.tool,
            aliases=list(candidate.aliases),
            namespace=candidate.namespace,
            schema=schema,
            include_metadata=include_metadata,
        )
        metadata = dict(descriptor.metadata)
        display_projector = metadata.pop(
            TOOL_DISPLAY_PROJECTOR_METADATA_KEY,
            None,
        )
        assert display_projector is None or callable(display_projector)
        projector_owner, projector_implementation = (
            cls._callable_semantic_parts(display_projector)
        )
        return _ToolSemanticIdentity(
            canonical_name=candidate.canonical_name,
            tool=candidate.tool,
            aliases=candidate.aliases,
            namespace=candidate.namespace,
            schema_prefix=candidate.schema_prefix,
            descriptor=descriptor,
            schema_semantics=cls._freeze_semantic_json(descriptor.schema),
            parameter_schema_semantics=cls._freeze_semantic_json(
                descriptor.parameter_schema
            ),
            return_schema_semantics=cls._freeze_semantic_json(
                descriptor.return_schema
            ),
            provider_safe_schema_semantics=cls._freeze_semantic_json(
                descriptor.provider_safe_schema
            ),
            capabilities=descriptor.capabilities,
            policy_semantics=cls._freeze_semantic_json(descriptor.policy),
            metadata_semantics=cls._freeze_semantic_json(metadata),
            display_projector_owner=projector_owner,
            display_projector_implementation=projector_implementation,
        )

    @staticmethod
    def _callable_semantic_parts(
        value: object | None,
    ) -> tuple[object | None, object | None]:
        if value is None:
            return None, None
        return (
            getattr(value, "__self__", None),
            getattr(value, "__func__", value),
        )

    @classmethod
    def _freeze_semantic_json(cls, value: object) -> object:
        if value is None:
            return ("none",)
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, int):
            return ("int", value)
        if isinstance(value, float):
            return ("float", value)
        if isinstance(value, str):
            return ("str", value)
        if isinstance(value, list):
            return (
                "list",
                tuple(cls._freeze_semantic_json(item) for item in value),
            )
        if isinstance(value, dict):
            for key in value:
                assert isinstance(
                    key, str
                ), "tool descriptor semantics require string mapping keys"
            return (
                "dict",
                tuple(
                    (
                        key,
                        cls._freeze_semantic_json(value[key]),
                    )
                    for key in sorted(value)
                ),
            )
        raise AssertionError(
            "tool descriptor semantics must contain only JSON values"
        )

    @staticmethod
    def _same_tool_identity(
        left: _ToolSemanticIdentity,
        right: _ToolSemanticIdentity,
    ) -> bool:
        return (
            left.tool is right.tool
            and left.canonical_name == right.canonical_name
            and left.aliases == right.aliases
            and left.namespace == right.namespace
            and left.schema_prefix == right.schema_prefix
            and left.schema_semantics == right.schema_semantics
            and left.parameter_schema_semantics
            == right.parameter_schema_semantics
            and left.return_schema_semantics == right.return_schema_semantics
            and left.provider_safe_schema_semantics
            == right.provider_safe_schema_semantics
            and left.capabilities == right.capabilities
            and left.policy_semantics == right.policy_semantics
            and left.metadata_semantics == right.metadata_semantics
            and left.display_projector_owner is right.display_projector_owner
            and left.display_projector_implementation
            is right.display_projector_implementation
        )

    @classmethod
    def _same_tool_inventory(
        cls,
        left: Sequence[_ToolSemanticIdentity],
        right: Sequence[_ToolSemanticIdentity],
    ) -> bool:
        return len(left) == len(right) and all(
            cls._same_tool_identity(original, verified)
            for original, verified in zip(left, right, strict=True)
        )

    @classmethod
    def _same_inventory_snapshot(
        cls,
        left: _ToolsetInventorySnapshot,
        right: _ToolsetInventorySnapshot,
    ) -> bool:
        return (
            cls._same_tool_inventory(left.configured, right.configured)
            and cls._same_optional_tool_inventory(
                left.available,
                right.available,
            )
            and cls._same_optional_tool_inventory(
                left.advertised,
                right.advertised,
            )
            and cls._same_optional_tool_inventory(
                left.selected,
                right.selected,
            )
            and left.custom_selector == right.custom_selector
            and left.selector_implementation is right.selector_implementation
        )

    @classmethod
    def _same_optional_tool_inventory(
        cls,
        left: Sequence[_ToolSemanticIdentity] | None,
        right: Sequence[_ToolSemanticIdentity] | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return cls._same_tool_inventory(left, right)

    @classmethod
    def _source_semantic_identities(
        cls,
        tools: Sequence[Callable[..., Any] | Tool | ToolSet],
        namespace: str | None,
        *,
        partition: str,
        enable_tools: Sequence[str] | None,
        validate_inventory: bool,
        active_toolsets: frozenset[int],
    ) -> tuple[_ToolSemanticIdentity, ...]:
        assert partition in ("configured", "available", "selected")
        schema_prefix = f"{namespace}." if namespace else ""
        identities: list[_ToolSemanticIdentity] = []
        for tool in tools:
            if isinstance(tool, ToolSet):
                nested = cls._toolset_inventory_snapshot(
                    tool,
                    namespace,
                    enable_tools=enable_tools,
                    validate_inventory=validate_inventory,
                    active_toolsets=active_toolsets,
                )
                if partition == "configured":
                    identities.extend(nested.configured)
                elif partition == "available":
                    identities.extend(cls._complete_inventory(nested))
                else:
                    identities.extend(nested.configured)
                continue
            canonical_name = f"{schema_prefix}{cls._valid_tool_name(tool)}"
            identities.append(
                cls._tool_semantic_identity(
                    _ToolCandidate(
                        canonical_name=canonical_name,
                        tool=tool,
                        aliases=tuple(cls._tool_aliases(tool)),
                        namespace=namespace,
                        schema_prefix=schema_prefix,
                    ),
                    include_metadata=(
                        enable_tools is None
                        or any(
                            matches_tool_namespace(canonical_name, enabled)
                            for enabled in enable_tools
                        )
                    ),
                )
            )
        cls._assert_unique_semantic_names(identities, partition)
        return tuple(identities)

    @classmethod
    def _toolset_inventory_snapshot(
        cls,
        toolset: ToolSet,
        prefix: str | None = None,
        *,
        enable_tools: Sequence[str] | None = None,
        validate_inventory: bool = False,
        active_toolsets: frozenset[int] = frozenset(),
    ) -> _ToolsetInventorySnapshot:
        toolset_id = id(toolset)
        assert (
            toolset_id not in active_toolsets
        ), "toolset nesting must be acyclic"
        active_toolsets |= {toolset_id}
        namespace = (
            f"{prefix}.{toolset.namespace}"
            if prefix and toolset.namespace
            else prefix or toolset.namespace
        )
        sources = cls._advertised_tool_sources(
            toolset,
            enable_tools,
            validate_inventory=validate_inventory,
        )
        configured = cls._source_semantic_identities(
            sources.configured,
            namespace,
            partition="configured",
            enable_tools=enable_tools,
            validate_inventory=validate_inventory,
            active_toolsets=active_toolsets,
        )
        available = (
            None
            if sources.available is None
            else cls._source_semantic_identities(
                sources.available,
                namespace,
                partition="available",
                enable_tools=enable_tools,
                validate_inventory=validate_inventory,
                active_toolsets=active_toolsets,
            )
        )
        advertised = (
            None
            if sources.advertised is None
            else cls._source_semantic_identities(
                sources.advertised,
                namespace,
                partition="available",
                enable_tools=enable_tools,
                validate_inventory=validate_inventory,
                active_toolsets=active_toolsets,
            )
        )
        selected = (
            None
            if sources.selected is None
            else cls._source_semantic_identities(
                sources.selected,
                namespace,
                partition="selected",
                enable_tools=enable_tools,
                validate_inventory=validate_inventory,
                active_toolsets=active_toolsets,
            )
        )
        snapshot = _ToolsetInventorySnapshot(
            configured=configured,
            available=available,
            advertised=advertised,
            selected=selected,
            custom_selector=sources.custom_selector,
            selector_implementation=sources.selector_implementation,
        )
        if validate_inventory:
            cls._validate_semantic_inventory_snapshot(
                snapshot,
                enable_tools,
            )
        return snapshot

    @classmethod
    def _validate_semantic_inventory_snapshot(
        cls,
        snapshot: _ToolsetInventorySnapshot,
        enable_tools: Sequence[str] | None,
    ) -> None:
        complete = cls._complete_inventory(snapshot)
        complete_by_name = {
            identity.canonical_name: identity for identity in complete
        }
        for configured in snapshot.configured:
            available = complete_by_name.get(configured.canonical_name)
            assert available is not None and cls._same_tool_identity(
                configured,
                available,
            ), (
                "available_tools inventory must cover configured toolset "
                f"tools or semantics: {configured.canonical_name!r}"
            )
        for advertised in snapshot.advertised or ():
            available = complete_by_name.get(advertised.canonical_name)
            assert available is None or cls._same_tool_identity(
                advertised,
                available,
            ), (
                "advertised tool inventory conflicts with complete inventory "
                f"semantics: {advertised.canonical_name!r}"
            )
        if enable_tools is None:
            return
        expected = cls._requested_inventory(complete, enable_tools)
        planned = (
            snapshot.selected
            if snapshot.selected is not None
            else cls._requested_inventory(snapshot.configured, enable_tools)
        )
        assert cls._same_tool_inventory(
            planned, expected
        ), "selected tool inventory must exactly match permitted selection"

    @staticmethod
    def _complete_inventory(
        snapshot: _ToolsetInventorySnapshot,
    ) -> tuple[_ToolSemanticIdentity, ...]:
        return (
            snapshot.configured
            if snapshot.available is None
            else snapshot.available
        )

    @classmethod
    def _advertised_inventory(
        cls,
        snapshot: _ToolsetInventorySnapshot,
    ) -> tuple[_ToolSemanticIdentity, ...]:
        if snapshot.advertised is None:
            return cls._complete_inventory(snapshot)
        identities = list(snapshot.configured)
        by_name = {
            identity.canonical_name: identity for identity in identities
        }
        for identity in snapshot.advertised:
            previous = by_name.get(identity.canonical_name)
            assert previous is None or cls._same_tool_identity(
                previous,
                identity,
            ), (
                "advertised tool inventory contains inconsistent semantics "
                f"for {identity.canonical_name!r}"
            )
            if previous is None:
                identities.append(identity)
                by_name[identity.canonical_name] = identity
        return tuple(identities)

    @classmethod
    def _permitted_inventory(
        cls,
        snapshot: _ToolsetInventorySnapshot,
        enable_tools: Sequence[str] | None,
    ) -> tuple[_ToolSemanticIdentity, ...]:
        if enable_tools is None:
            return snapshot.configured
        if snapshot.selected is not None:
            return snapshot.selected
        return cls._requested_inventory(snapshot.configured, enable_tools)

    @staticmethod
    def _requested_inventory(
        complete: Sequence[_ToolSemanticIdentity],
        enable_tools: Sequence[str],
    ) -> tuple[_ToolSemanticIdentity, ...]:
        return tuple(
            identity
            for identity in complete
            if any(
                matches_tool_namespace(identity.canonical_name, enabled)
                for enabled in enable_tools
            )
        )

    @staticmethod
    def _assert_unique_semantic_names(
        identities: Sequence[_ToolSemanticIdentity],
        label: str,
    ) -> None:
        names: set[str] = set()
        for identity in identities:
            assert identity.canonical_name not in names, (
                f"{label} tool inventory contains duplicate canonical name "
                f"{identity.canonical_name!r}"
            )
            names.add(identity.canonical_name)

    @staticmethod
    def _semantic_inventory_names(
        identities: Sequence[_ToolSemanticIdentity],
    ) -> list[tuple[str, list[str]]]:
        return [
            (identity.canonical_name, list(identity.aliases))
            for identity in identities
        ]

    @classmethod
    def _snapshot_advertised_names(
        cls,
        snapshot: _ToolsetInventorySnapshot,
    ) -> list[tuple[str, list[str]]]:
        groups = [snapshot.configured]
        if snapshot.available is not None:
            groups.append(snapshot.available)
        if snapshot.advertised is not None:
            groups.append(snapshot.advertised)
        if snapshot.selected is not None:
            groups.append(snapshot.selected)
        return [
            entry
            for group in groups
            for entry in cls._semantic_inventory_names(group)
        ]

    @staticmethod
    def _merge_tool_name_entries(
        groups: Sequence[Sequence[tuple[str, Sequence[str]]]],
    ) -> list[tuple[str, list[str]]]:
        merged: dict[str, tuple[str, ...]] = {}
        for entries in groups:
            for canonical_name, aliases in entries:
                alias_tuple = tuple(aliases)
                previous = merged.get(canonical_name)
                assert previous is None or previous == alias_tuple, (
                    "advertised tool inventory contains inconsistent aliases "
                    f"for {canonical_name!r}"
                )
                if previous is None:
                    merged[canonical_name] = alias_tuple
        return [
            (canonical_name, list(aliases))
            for canonical_name, aliases in merged.items()
        ]

    @classmethod
    def _snapshot_toolset_tools(
        cls,
        toolsets: Sequence[ToolSet],
    ) -> list[_ToolsetToolsSnapshot]:
        snapshots: list[_ToolsetToolsSnapshot] = []
        pending = list(toolsets)
        seen: set[int] = set()
        while pending:
            toolset = pending.pop()
            toolset_id = id(toolset)
            if toolset_id in seen:
                continue
            seen.add(toolset_id)
            tools = toolset._tools
            assert isinstance(tools, list), "toolset tools must be a list"
            contents = tuple(tools)
            snapshots.append(
                _ToolsetToolsSnapshot(
                    toolset=toolset,
                    tools=tools,
                    contents=contents,
                )
            )
            pending.extend(
                tool for tool in contents if isinstance(tool, ToolSet)
            )
        return snapshots

    @staticmethod
    def _restore_toolset_tools(
        snapshots: Sequence[_ToolsetToolsSnapshot],
    ) -> None:
        for snapshot in reversed(snapshots):
            current_tools = snapshot.toolset._tools
            if (
                current_tools is snapshot.tools
                and len(current_tools) == len(snapshot.contents)
                and all(
                    current is original
                    for current, original in zip(
                        current_tools,
                        snapshot.contents,
                        strict=True,
                    )
                )
            ):
                continue
            snapshot.tools[:] = snapshot.contents
            snapshot.toolset._tools = snapshot.tools

    @staticmethod
    def _toolset_tools_match_snapshots(
        snapshots: Sequence[_ToolsetToolsSnapshot],
    ) -> bool:
        return all(
            snapshot.toolset._tools is snapshot.tools
            and len(snapshot.toolset._tools) == len(snapshot.contents)
            and all(
                current is original
                for current, original in zip(
                    snapshot.toolset._tools,
                    snapshot.contents,
                    strict=True,
                )
            )
            for snapshot in snapshots
        )

    @classmethod
    def _registration_plan(
        cls,
        toolset: ToolSet,
        *,
        available_identities: Sequence[_ToolSemanticIdentity],
        expected_identities: Sequence[_ToolSemanticIdentity],
        enable_tools: tuple[str, ...] | None,
        settings: ToolManagerSettings,
        prefix: str | None = None,
    ) -> _ToolsetRegistrationPlan:
        cls._assert_unique_semantic_names(
            available_identities,
            "available",
        )
        identities = tuple(
            cls._tool_semantic_identity(candidate)
            for candidate in cls._tool_candidates(toolset, prefix)
        )
        cls._assert_unique_semantic_names(identities, "enabled")
        assert cls._same_tool_inventory(
            identities, expected_identities
        ), "selected toolset inventory does not match prevalidated selection"
        combined_names = cls._merge_tool_name_entries(
            [
                cls._semantic_inventory_names(available_identities),
                cls._semantic_inventory_names(identities),
            ]
        )
        cls._validate_model_capability_names(combined_names, settings)

        bootstrap_enabled = getattr(toolset, "bootstrap_enabled", True)
        assert isinstance(
            bootstrap_enabled, bool
        ), "bootstrap_enabled must be a boolean"
        bootstrap_prompt_settings = getattr(
            toolset,
            "bootstrap_prompt_settings",
            None,
        )
        assert bootstrap_prompt_settings is None or isinstance(
            bootstrap_prompt_settings,
            SkillBootstrapPromptSettings,
        ), "bootstrap_prompt_settings must be SkillBootstrapPromptSettings"

        registrations: list[_ToolRegistration] = []
        for identity in identities:
            registrations.append(
                _ToolRegistration(
                    canonical_name=identity.canonical_name,
                    tool=cast(Callable[..., Any], identity.tool),
                    aliases=identity.aliases,
                    namespace=identity.namespace,
                    descriptor=identity.descriptor,
                )
            )
        verified_identities = tuple(
            cls._tool_semantic_identity(candidate)
            for candidate in cls._tool_candidates(toolset, prefix)
        )
        assert cls._same_tool_inventory(
            identities,
            verified_identities,
        ), "toolset inventory changed during registration"
        return _ToolsetRegistrationPlan(
            toolset=toolset,
            prefix=prefix,
            enable_tools=enable_tools,
            available_identities=tuple(available_identities),
            identities=identities,
            registrations=tuple(registrations),
            bootstrap_enabled=bootstrap_enabled,
            bootstrap_prompt_settings=bootstrap_prompt_settings,
        )

    @classmethod
    def _registry_state(
        cls,
        plans: Sequence[_ToolsetRegistrationPlan],
        settings: ToolManagerSettings,
    ) -> _ToolManagerRegistryState:
        available_names = cls._merge_tool_name_entries(
            [
                cls._semantic_inventory_names(plan.available_identities)
                for plan in plans
            ]
        )
        cls._validate_model_capability_names(available_names, settings)
        tools: dict[str, Callable[..., Any]] = {}
        aliases: dict[str, list[str]] = {}
        available_aliases: dict[str, list[str]] = {}
        available_tool_names: set[str] = set()
        bootstrap_tool_names: set[str] = set()
        descriptors: dict[str, ToolDescriptor] = {}
        bootstrap_prompt_settings: SkillBootstrapPromptSettings | None = None
        toolsets: list[ToolSet] = []

        for canonical_name, available_tool_aliases in available_names:
            available_tool_names.add(canonical_name)
            cls._add_aliases(
                available_aliases,
                canonical_name,
                available_tool_aliases,
            )
        for plan in plans:
            if plan.registrations and not any(
                registered_toolset is plan.toolset
                for registered_toolset in toolsets
            ):
                toolsets.append(plan.toolset)
            if (
                plan.registrations
                and plan.bootstrap_enabled
                and bootstrap_prompt_settings is None
                and plan.bootstrap_prompt_settings is not None
            ):
                bootstrap_prompt_settings = plan.bootstrap_prompt_settings
            for registration in plan.registrations:
                canonical_name = registration.canonical_name
                if canonical_name in tools:
                    raise ValueError(
                        "duplicate canonical tool registration "
                        f"{canonical_name!r}"
                    )
                tools[canonical_name] = registration.tool
                descriptors[canonical_name] = registration.descriptor
                cls._add_aliases(
                    aliases,
                    canonical_name,
                    list(registration.aliases),
                )
                if plan.bootstrap_enabled:
                    bootstrap_tool_names.add(canonical_name)
        return _ToolManagerRegistryState(
            tools=tools,
            aliases=aliases,
            available_aliases=available_aliases,
            available_tool_names=available_tool_names,
            bootstrap_tool_names=bootstrap_tool_names,
            descriptors=descriptors,
            skills_bootstrap_prompt_settings=bootstrap_prompt_settings,
            toolsets=toolsets,
        )

    def _commit_registry_state(self, state: _ToolManagerRegistryState) -> None:
        self._tools = state.tools
        self._aliases = state.aliases
        self._available_aliases = state.available_aliases
        self._available_tool_names = state.available_tool_names
        self._bootstrap_tool_names = state.bootstrap_tool_names
        self._descriptors = state.descriptors
        self._skills_bootstrap_prompt_settings = (
            state.skills_bootstrap_prompt_settings
        )
        self._toolsets = state.toolsets

    def _register_toolset(
        self, toolset: ToolSet, prefix: str | None = None
    ) -> None:
        assert isinstance(toolset, ToolSet)
        snapshots = self._snapshot_toolset_tools([toolset])
        try:
            plans = list(self._registration_plans)
            registered_index = next(
                (
                    index
                    for index, registered in enumerate(plans)
                    if registered.toolset is toolset
                    and registered.prefix == prefix
                ),
                None,
            )
            previous = (
                None if registered_index is None else plans[registered_index]
            )
            enable_tools = None if previous is None else previous.enable_tools
            inventory = self._toolset_inventory_snapshot(
                toolset,
                prefix,
                enable_tools=enable_tools,
            )
            self._validate_model_capability_names(
                self._snapshot_advertised_names(inventory),
                self._settings,
            )
            verified_inventory = self._toolset_inventory_snapshot(
                toolset,
                prefix,
                enable_tools=enable_tools,
                validate_inventory=True,
            )
            assert self._same_inventory_snapshot(
                inventory,
                verified_inventory,
            ), "advertised tool inventory changed during registration"
            assert self._toolset_tools_match_snapshots(
                snapshots
            ), "advertised selector inventory must be pure"
            plan = self._registration_plan(
                toolset,
                available_identities=(
                    self._reregistration_advertised_inventory(
                        previous,
                        inventory,
                    )
                ),
                expected_identities=self._permitted_inventory(
                    inventory,
                    enable_tools,
                ),
                enable_tools=enable_tools,
                settings=self._settings,
                prefix=prefix,
            )
            if registered_index is None:
                plans.append(plan)
            else:
                assert previous is not None
                self._validate_reregistration_semantics(previous, plan)
                plans[registered_index] = plan
            state = self._registry_state(plans, self._settings)
        except BaseException:
            self._restore_toolset_tools(snapshots)
            raise
        self._registration_plans = plans
        self._commit_registry_state(state)

    @classmethod
    def _reregistration_advertised_inventory(
        cls,
        previous: _ToolsetRegistrationPlan | None,
        snapshot: _ToolsetInventorySnapshot,
    ) -> tuple[_ToolSemanticIdentity, ...]:
        current = cls._advertised_inventory(snapshot)
        if previous is None or previous.enable_tools is None:
            return current

        current_sources = (
            *cls._complete_inventory(snapshot),
            *(snapshot.advertised or ()),
        )
        current_by_name = {
            identity.canonical_name: identity for identity in current_sources
        }
        enabled_names = {
            identity.canonical_name for identity in previous.identities
        }
        identities: list[_ToolSemanticIdentity] = []
        names: set[str] = set()
        for identity in previous.available_identities:
            candidate = current_by_name.get(identity.canonical_name)
            if candidate is None:
                if identity.canonical_name in enabled_names:
                    continue
                candidate = identity
            identities.append(candidate)
            names.add(candidate.canonical_name)
        for identity in current:
            if identity.canonical_name not in names:
                identities.append(identity)
                names.add(identity.canonical_name)
        return tuple(identities)

    @classmethod
    def _validate_reregistration_semantics(
        cls,
        previous: _ToolsetRegistrationPlan,
        replacement: _ToolsetRegistrationPlan,
    ) -> None:
        for original, updated in (
            (previous.available_identities, replacement.available_identities),
            (previous.identities, replacement.identities),
        ):
            updated_by_name = {
                identity.canonical_name: identity for identity in updated
            }
            for identity in original:
                candidate = updated_by_name.get(identity.canonical_name)
                if candidate is None:
                    continue
                assert cls._same_tool_identity(
                    identity, candidate
                ), "re-registration cannot substitute existing tool semantics"

    @staticmethod
    def _tool_schema(
        tool: Callable[..., Any] | Tool, prefix: str
    ) -> dict[str, Any] | None:
        if isinstance(tool, Tool):
            return tool.json_schema(prefix)
        schema = get_json_schema(tool)
        if (
            prefix
            and schema.get("type") == "function"
            and "function" in schema
            and "name" in schema["function"]
        ):
            schema["function"]["name"] = prefix + schema["function"]["name"]
        return schema

    @classmethod
    def _tool_descriptor(
        cls,
        *,
        canonical_name: str,
        tool: Callable[..., Any] | Tool,
        aliases: list[str],
        namespace: str | None,
        schema: dict[str, Any] | None,
        include_metadata: bool = True,
    ) -> ToolDescriptor:
        function_schema = (
            schema.get("function")
            if schema and schema.get("type") == "function"
            else None
        )
        parameters = (
            function_schema.get("parameters")
            if isinstance(function_schema, dict)
            else None
        )
        returns = (
            function_schema.get("return")
            if isinstance(function_schema, dict)
            else None
        )
        return ToolDescriptor(
            name=canonical_name,
            callable=cast(Callable[..., Any], tool),
            aliases=aliases,
            schema=schema,
            parameter_schema=(
                cast(dict[str, Any], parameters)
                if isinstance(parameters, dict)
                else None
            ),
            return_schema=(
                cast(dict[str, Any], returns)
                if isinstance(returns, dict)
                else None
            ),
            namespace=namespace,
            capabilities=cls._tool_capabilities(tool),
            metadata=cls._tool_metadata(tool) if include_metadata else {},
        )

    @classmethod
    def _tool_metadata(
        cls, tool: Callable[..., Any] | Tool
    ) -> dict[str, ToolDescriptorMetadataValue]:
        projector = cls._tool_display_projector(tool)
        if projector is None:
            return {}
        return {TOOL_DISPLAY_PROJECTOR_METADATA_KEY: projector}

    @staticmethod
    def _tool_display_projector(
        tool: Callable[..., Any] | Tool,
    ) -> ToolDisplayProjector | None:
        projector = getattr(tool, TOOL_DISPLAY_PROJECTOR_METADATA_KEY, None)
        if projector is None:
            return None
        assert callable(projector), "tool_display_projector must be callable"
        return cast(ToolDisplayProjector, projector)

    @classmethod
    def _tool_capabilities(
        cls, tool: Callable[..., Any] | Tool
    ) -> ToolCapabilities:
        configured = getattr(tool, "tool_capabilities", None)
        if configured is not None:
            return cls._coerce_tool_capabilities(configured)
        return ToolCapabilities(
            supports_streaming=cls._tool_capability_flag(
                tool,
                "supports_streaming",
                default=False,
            ),
            side_effecting=cls._tool_capability_flag(
                tool,
                "side_effecting",
                default=True,
            ),
            parallel_safe=cls._tool_capability_flag(
                tool,
                "parallel_safe",
                default=False,
            ),
        )

    @staticmethod
    def _coerce_tool_capabilities(value: object) -> ToolCapabilities:
        if isinstance(value, ToolCapabilities):
            return value
        assert isinstance(value, dict), "tool_capabilities must be a mapping"
        supported_keys = {
            "supports_streaming",
            "side_effecting",
            "parallel_safe",
        }
        unknown_keys = sorted(set(value) - supported_keys)
        assert not unknown_keys, "tool_capabilities has unknown keys"
        return ToolCapabilities(
            supports_streaming=value.get("supports_streaming", False),
            side_effecting=value.get("side_effecting", True),
            parallel_safe=value.get("parallel_safe", False),
        )

    @staticmethod
    def _tool_capability_flag(
        tool: Callable[..., Any] | Tool,
        name: str,
        *,
        default: bool,
    ) -> bool:
        value = getattr(tool, name, default)
        assert isinstance(value, bool), f"{name} must be a boolean"
        return value

    async def prepare_call(
        self, call: ToolCall, context: ToolCallContext
    ) -> PreparedToolCall | ToolCallDiagnostic:
        """Return a prepared execution plan or a diagnostic."""
        assert call

        diagnostic = self._guard_diagnostic(call, context)
        if diagnostic is not None:
            return diagnostic

        await _check_cancelled(context)

        prepared = self._prepare_resolved_call(
            call,
            context,
        )
        if isinstance(prepared, ToolCallDiagnostic):
            return prepared

        filtered = await self._apply_filters(
            prepared.call,
            prepared.context,
        )
        if isinstance(filtered, ToolCallDiagnostic):
            return filtered
        call, context = filtered
        await _check_cancelled(context)

        prepared = self._prepare_resolved_call(
            call,
            context,
        )
        if isinstance(prepared, ToolCallDiagnostic):
            return prepared

        diagnostic = self._guard_diagnostic(prepared.call, prepared.context)
        if diagnostic is not None:
            return diagnostic

        validation = self._validate_prepared_call(prepared)
        if validation is not None:
            return validation
        return prepared

    async def execute_prepared_call(
        self, prepared: PreparedToolCall
    ) -> ToolCallResult | ToolCallError:
        """Execute a prepared call without rerunning preparation."""
        assert isinstance(prepared, PreparedToolCall)
        await _check_cancelled(prepared.context)
        return await self._execute_prepared_call(prepared)

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
        *,
        confirm: (
            Callable[
                [ToolCall], Awaitable[str | bool | None] | str | bool | None
            ]
            | None
        ) = None,
    ) -> ToolCallOutcome:
        """Execute a call and return result, error, or diagnostic."""
        assert call

        try:
            prepared = await self.prepare_call(call, context)
        except CancelledError:
            return self._cancelled_diagnostic(call)
        if isinstance(prepared, ToolCallDiagnostic):
            return prepared

        if confirm is not None:
            action = confirm(prepared.call)
            if isinstance(action, Awaitable):
                action = await action
            if action not in (True, "y", "a"):
                return self._diagnostic(
                    call=prepared.call,
                    canonical_name=prepared.call.name,
                    code=ToolCallDiagnosticCode.USER_REJECTED,
                    stage=ToolCallDiagnosticStage.CONFIRM,
                    message="Tool call was rejected before execution.",
                )

        try:
            return await self.execute_prepared_call(prepared)
        except CancelledError:
            return self._cancelled_diagnostic(
                prepared.call,
                canonical_name=prepared.call.name,
            )

    async def __aenter__(self) -> "ToolManager":
        if self._toolsets:
            for i, toolset in enumerate(self._toolsets):
                toolset = await self._stack.enter_async_context(toolset)
                self._toolsets[i] = toolset
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        interrupted_exit = exc_type is not None and issubclass(
            exc_type, (CancelledError, KeyboardInterrupt)
        )
        close = self._stack.__aexit__(exc_type, exc_value, traceback)
        if not interrupted_exit:
            return await close

        try:
            await wait_for(close, timeout=self._INTERRUPT_CLOSE_TIMEOUT)
        except (CancelledError, TimeoutError):
            return False
        except Exception:
            return False
        return False

    async def __call__(
        self, call: ToolCall, context: ToolCallContext
    ) -> ToolCallOutcome | None:
        """Execute a single tool call and return the result."""
        assert call

        if self._settings.execution_mode is ToolManagerExecutionMode.OUTCOMES:
            return await self.execute_call(call, context)

        if self._guard_diagnostic(call, context) is not None:
            return None

        if not self._tools or call.name not in self._tools:
            return None

        await _check_cancelled(context)

        filtered = await self._apply_filters(call, context)
        if isinstance(filtered, ToolCallDiagnostic):
            return None
        call, context = filtered

        await _check_cancelled(context)

        prepared = self._prepare_resolved_call(
            call,
            context,
        )
        if isinstance(prepared, ToolCallDiagnostic):
            return None
        if self._guard_diagnostic(prepared.call, prepared.context) is not None:
            return None
        return await self._execute_prepared_call(prepared)

    def _prepare_resolved_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> PreparedToolCall | ToolCallDiagnostic:
        if call.provider_name is None and not call.name.strip():
            return self._diagnostic(
                call=call,
                code=ToolCallDiagnosticCode.MALFORMED_CALL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Tool call name must not be empty.",
            )

        resolution = self._resolve_call_name(call)
        if resolution.diagnostic_code is not None:
            return self._resolution_diagnostic(call, resolution)
        assert resolution.canonical_name is not None

        provider_diagnostic = self._provider_arguments_diagnostic(
            call=call,
            canonical_name=resolution.canonical_name,
        )
        if provider_diagnostic is not None:
            return provider_diagnostic

        arguments = call.arguments if call.arguments is not None else {}
        if not isinstance(arguments, dict):
            return self._diagnostic(
                call=call,
                canonical_name=resolution.canonical_name,
                code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                stage=ToolCallDiagnosticStage.VALIDATE,
                message="Tool call arguments must be an object.",
            )

        descriptor = self._descriptors[resolution.canonical_name]
        tool = descriptor.callable
        assert tool is not None
        arguments = self._normalize_single_input_argument(tool, arguments)

        diagnostic = self._validate_argument_limits(
            call=call,
            canonical_name=resolution.canonical_name,
            arguments=arguments,
        )
        if diagnostic is not None:
            return diagnostic

        prepared_call = ToolCall(
            id=call.id,
            name=resolution.canonical_name,
            arguments=arguments,
            provider_name=call.provider_name,
            provider_name_encoded=call.provider_name_encoded,
            provider_arguments_malformed=call.provider_arguments_malformed,
        )
        return PreparedToolCall(
            call=prepared_call,
            callable=tool,
            descriptor=descriptor,
            arguments=arguments,
            context=context,
        )

    @classmethod
    def _normalize_single_input_argument(
        cls, tool: Callable[..., Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if set(arguments) != {"input"}:
            return arguments
        parameter_name = cls._single_user_parameter_name(tool)
        if parameter_name is None or parameter_name == "input":
            return arguments
        return {parameter_name: arguments["input"]}

    @staticmethod
    def _single_user_parameter_name(tool: Callable[..., Any]) -> str | None:
        target = tool.__call__ if isinstance(tool, Tool) else tool
        try:
            parameters = signature(target).parameters.values()
        except (TypeError, ValueError):
            return None

        user_parameters: list[str] = []
        for parameter in parameters:
            if parameter.name == "context":
                continue
            if parameter.kind in (
                Parameter.VAR_POSITIONAL,
                Parameter.VAR_KEYWORD,
            ):
                return None
            if parameter.kind in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.KEYWORD_ONLY,
            ):
                user_parameters.append(parameter.name)
        if len(user_parameters) != 1:
            return None
        return user_parameters[0]

    def _guard_diagnostic(
        self, call: ToolCall, context: ToolCallContext
    ) -> ToolCallDiagnostic | None:
        history = context.calls or []

        if self._settings.avoid_repetition and history:
            last = history[-1]
            if last.name == call.name and last.arguments == call.arguments:
                return self._diagnostic(
                    call=call,
                    code=ToolCallDiagnosticCode.REPEATED_CALL,
                    stage=ToolCallDiagnosticStage.GUARD,
                    message="Tool call repeats the previous call.",
                )

        if (
            self._settings.maximum_depth is not None
            and len(history) + 1 > self._settings.maximum_depth
        ):
            return self._diagnostic(
                call=call,
                code=ToolCallDiagnosticCode.MAXIMUM_DEPTH,
                stage=ToolCallDiagnosticStage.GUARD,
                message="Tool call exceeds the maximum depth.",
            )
        return None

    def _validate_prepared_call(
        self, prepared: PreparedToolCall
    ) -> ToolCallDiagnostic | None:
        return self._validate_tool_arguments(
            call=prepared.call,
            canonical_name=prepared.call.name,
            tool=prepared.callable,
            arguments=prepared.arguments,
            context=prepared.context,
        )

    def _resolution_diagnostic(
        self, call: ToolCall, resolution: ToolNameResolution
    ) -> ToolCallDiagnostic:
        assert resolution.diagnostic_code is not None
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=resolution.requested_name,
            canonical_name=resolution.canonical_name,
            code=resolution.diagnostic_code,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message=(
                f"Tool '{resolution.requested_name}' is "
                f"{resolution.status.value}."
            ),
            details={"candidates": cast(Any, resolution.candidates)},
        )

    def _resolve_call_name(self, call: ToolCall) -> ToolNameResolution:
        try:
            return self.resolve_tool_name(call.name)
        except AssertionError:
            return ToolNameResolution(
                requested_name=call.provider_name or call.name or "invalid",
                status=ToolNameResolutionStatus.UNKNOWN,
                diagnostic_code=ToolCallDiagnosticCode.MALFORMED_CALL,
            )

    def _diagnostic(
        self,
        *,
        call: ToolCall,
        code: ToolCallDiagnosticCode,
        stage: ToolCallDiagnosticStage,
        message: str,
        canonical_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ToolCallDiagnostic:
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=call.name if call.name.strip() else None,
            canonical_name=canonical_name,
            code=code,
            stage=stage,
            message=message,
            details=details or {},
        )

    def _cancelled_diagnostic(
        self,
        call: ToolCall,
        *,
        canonical_name: str | None = None,
    ) -> ToolCallDiagnostic:
        return self._diagnostic(
            call=call,
            canonical_name=canonical_name,
            code=ToolCallDiagnosticCode.CANCELLED,
            stage=ToolCallDiagnosticStage.GUARD,
            message="Tool call was cancelled before execution.",
        )

    def _provider_arguments_diagnostic(
        self,
        *,
        call: ToolCall,
        canonical_name: str,
    ) -> ToolCallDiagnostic | None:
        if (
            self._settings.provider_arguments_mode
            is not ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
            or not call.provider_arguments_malformed
        ):
            return None
        return self._diagnostic(
            call=call,
            canonical_name=canonical_name,
            code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
            stage=ToolCallDiagnosticStage.VALIDATE,
            message="Provider tool call arguments are malformed.",
        )

    def _validate_tool_arguments(
        self,
        *,
        call: ToolCall,
        canonical_name: str,
        tool: Callable[..., Any],
        arguments: dict[str, Any],
        context: ToolCallContext | None = None,
    ) -> ToolCallDiagnostic | None:
        descriptor = self._descriptors[canonical_name]
        schema_diagnostic = self._validate_arguments_schema(
            call=call,
            canonical_name=canonical_name,
            arguments=arguments,
            schema=descriptor.parameter_schema,
        )
        if schema_diagnostic is not None:
            return schema_diagnostic

        try:
            if isinstance(tool, Tool):
                signature(tool.__call__).bind(
                    **arguments,
                    context=context or ToolCallContext(),
                )
            else:
                self._bind_callable_arguments(tool, arguments)
        except TypeError as exc:
            return ToolCallDiagnostic(
                id=uuid4(),
                call_id=call.id,
                requested_name=call.name,
                canonical_name=canonical_name,
                code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                stage=ToolCallDiagnosticStage.VALIDATE,
                message=str(exc),
            )
        return None

    def _validate_argument_limits(
        self,
        *,
        call: ToolCall,
        canonical_name: str | None,
        arguments: dict[str, Any],
    ) -> ToolCallDiagnostic | None:
        return ToolCallParser.resource_limit_diagnostic(
            value=arguments,
            maximum_depth=self._settings.maximum_argument_depth,
            maximum_size=self._settings.maximum_argument_size,
            stage=ToolCallDiagnosticStage.VALIDATE,
            call_id=call.id,
            requested_name=call.name,
            canonical_name=canonical_name,
        )

    def _validate_arguments_schema(
        self,
        *,
        call: ToolCall,
        canonical_name: str,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> ToolCallDiagnostic | None:
        if schema is None:
            return None

        error = self._schema_validation_error(arguments, schema, "$")
        if error is None:
            return None
        return self._diagnostic(
            call=call,
            canonical_name=canonical_name,
            code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
            stage=ToolCallDiagnosticStage.VALIDATE,
            message=error,
        )

    @classmethod
    def _schema_validation_error(
        cls,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> str | None:
        if "anyOf" in schema:
            any_of = schema["anyOf"]
            assert isinstance(any_of, list)
            candidate_errors: list[tuple[dict[str, Any], str]] = []
            for candidate in any_of:
                if not isinstance(candidate, dict):
                    continue
                error = cls._schema_validation_error(value, candidate, path)
                if error is None:
                    return None
                candidate_errors.append((candidate, error))
            if isinstance(value, dict):
                for candidate, error in candidate_errors:
                    if _schema_enum_properties_match(value, candidate):
                        return error
                for candidate, error in candidate_errors:
                    if _schema_required_keys_present(value, candidate):
                        return error
            return f"{path} does not match any allowed schema."

        expected_type = schema.get("type")
        if expected_type is not None and not cls._matches_schema_type(
            value,
            expected_type,
        ):
            return (
                f"{path} must be {cls._format_expected_type(expected_type)}."
            )

        if "enum" in schema:
            enum_values = schema["enum"]
            assert isinstance(enum_values, list)
            if value not in enum_values:
                return f"{path} must be one of {enum_values!r}."

        if value is None:
            return None

        schema_type = schema.get("type")
        if schema_type == "string" or (
            isinstance(schema_type, list) and "string" in schema_type
        ):
            return cls._string_schema_validation_error(value, schema, path)
        if schema_type == "object" or (
            isinstance(schema_type, list) and "object" in schema_type
        ):
            return cls._object_schema_validation_error(value, schema, path)
        if schema_type == "array" or (
            isinstance(schema_type, list) and "array" in schema_type
        ):
            return cls._array_schema_validation_error(value, schema, path)
        return None

    @classmethod
    def _object_schema_validation_error(
        cls,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> str | None:
        if not isinstance(value, dict):
            return None

        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        required = schema.get("required", [])
        assert isinstance(required, list)
        for name in required:
            assert isinstance(name, str)
            if name not in value:
                return f"{path}.{name} is required."

        additional_properties = schema.get("additionalProperties", True)
        for name, field_value in value.items():
            field_path = f"{path}.{name}"
            field_schema = properties.get(name)
            if isinstance(field_schema, dict):
                error = cls._schema_validation_error(
                    field_value,
                    field_schema,
                    field_path,
                )
                if error is not None:
                    return error
                continue
            if additional_properties is False:
                return f"{field_path} is not allowed."
            if isinstance(additional_properties, dict):
                error = cls._schema_validation_error(
                    field_value,
                    additional_properties,
                    field_path,
                )
                if error is not None:
                    return error
        return None

    @classmethod
    def _array_schema_validation_error(
        cls,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> str | None:
        if not isinstance(value, list):
            return None

        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return f"{path} must contain at least {min_items} item(s)."

        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            return f"{path} must contain at most {max_items} item(s)."

        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            for index, item_schema in enumerate(prefix_items):
                if index >= len(value) or not isinstance(item_schema, dict):
                    continue
                error = cls._schema_validation_error(
                    value[index],
                    item_schema,
                    f"{path}[{index}]",
                )
                if error is not None:
                    return error
            return None

        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                error = cls._schema_validation_error(
                    item,
                    items,
                    f"{path}[{index}]",
                )
                if error is not None:
                    return error
        return None

    @classmethod
    def _string_schema_validation_error(
        cls,
        value: Any,
        schema: dict[str, Any],
        path: str,
    ) -> str | None:
        if not isinstance(value, str):
            return None

        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            return f"{path} must contain at least {min_length} character(s)."

        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            return f"{path} must contain at most {max_length} character(s)."
        return None

    @classmethod
    def _matches_schema_type(
        cls,
        value: Any,
        expected_type: str | list[str],
    ) -> bool:
        if isinstance(expected_type, list):
            return any(
                cls._matches_schema_type(value, t) for t in expected_type
            )

        match expected_type:
            case "null":
                return value is None
            case "boolean":
                return isinstance(value, bool)
            case "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            case "number":
                return isinstance(value, int | float) and not isinstance(
                    value,
                    bool,
                )
            case "string":
                return isinstance(value, str)
            case "array":
                return isinstance(value, list)
            case "object":
                return isinstance(value, dict)
            case _:
                return True

    @staticmethod
    def _format_expected_type(expected_type: str | list[str]) -> str:
        if isinstance(expected_type, list):
            return " or ".join(expected_type)
        return expected_type

    async def _apply_filters(
        self, call: ToolCall, context: ToolCallContext
    ) -> tuple[ToolCall, ToolCallContext] | ToolCallDiagnostic:
        if not self._settings.filters:
            return call, context

        flow_tool_node = context.flow_tool_node
        for f in self._settings.filters:
            filter_namespace: str | None = None
            if isinstance(f, ToolFilter):
                filter_func = f.func
                filter_namespace = f.namespace
            else:
                filter_func = f
            if not matches_tool_namespace(
                self._filtered_tool_name(call),
                filter_namespace,
            ):
                continue
            modified = filter_func(call, context)
            if isinstance(modified, Awaitable):
                modified = await modified
            if modified is None:
                continue
            if isinstance(modified, ToolFilterResult):
                if modified.status is ToolFilterResultStatus.PASS:
                    continue
                if modified.status is ToolFilterResultStatus.SUPPRESS:
                    return self._diagnostic(
                        call=call,
                        canonical_name=self._diagnostic_canonical_name(call),
                        code=modified.code,
                        stage=ToolCallDiagnosticStage.FILTER,
                        message=(
                            modified.message
                            or "Tool call was suppressed by a filter."
                        ),
                        details=modified.details,
                    )
                assert modified.status is ToolFilterResultStatus.MODIFY
                assert modified.call is not None
                assert modified.context is not None
                modified = (modified.call, modified.context)
            assert isinstance(modified, tuple) and len(modified) == 2
            next_call, next_context = modified
            assert isinstance(next_call, ToolCall)
            assert isinstance(next_context, ToolCallContext)
            next_context = self._preserve_flow_context(
                context,
                next_context,
                flow_tool_node=flow_tool_node,
            )
            flow_tool_node = flow_tool_node or next_context.flow_tool_node
            filtered_name = self._filtered_tool_name(next_call)
            if flow_tool_node and filtered_name != self._filtered_tool_name(
                call
            ):
                return self._diagnostic(
                    call=call,
                    canonical_name=self._diagnostic_canonical_name(call),
                    code=ToolCallDiagnosticCode.FILTER_SUPPRESSED,
                    stage=ToolCallDiagnosticStage.FILTER,
                    message="Tool filter name rewrites are disabled.",
                    details={"filtered_name": filtered_name},
                )
            call, context = next_call, next_context
        return call, context

    def _filtered_tool_name(self, call: ToolCall) -> str:
        try:
            resolution = self.resolve_tool_name(call.name)
        except AssertionError:
            return call.name
        return resolution.canonical_name or resolution.requested_name

    def _diagnostic_canonical_name(self, call: ToolCall) -> str | None:
        resolution = self._resolve_call_name(call)
        return resolution.canonical_name

    @staticmethod
    def _preserve_flow_context(
        current: ToolCallContext,
        next_context: ToolCallContext,
        *,
        flow_tool_node: bool,
    ) -> ToolCallContext:
        if not flow_tool_node and not next_context.flow_tool_node:
            return next_context

        changes: dict[str, Any] = {}
        if not next_context.flow_tool_node:
            changes["flow_tool_node"] = True
        if (
            next_context.cancellation_checker is None
            and current.cancellation_checker is not None
        ):
            changes["cancellation_checker"] = current.cancellation_checker
        if (
            next_context.skills_registry is None
            and current.skills_registry is not None
        ):
            changes["skills_registry"] = current.skills_registry
        if not changes:
            return next_context
        return replace(next_context, **changes)

    async def _execute_prepared_call(
        self, prepared: PreparedToolCall
    ) -> ToolCallResult | ToolCallError:
        call = prepared.call
        context = prepared.context
        tool = prepared.callable
        try:
            result = await self._dispatch_tool(tool, call, context)
            result = self._apply_transformers(call, context, result)

            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                provider_name=call.provider_name,
                provider_name_encoded=call.provider_name_encoded,
                provider_arguments_malformed=(
                    call.provider_arguments_malformed
                ),
                result=result,
            )
        except Exception as exc:
            return ToolCallError(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                provider_name=call.provider_name,
                provider_name_encoded=call.provider_name_encoded,
                provider_arguments_malformed=(
                    call.provider_arguments_malformed
                ),
                error=self._project_error(exc),
                message=str(exc),
            )

    async def _dispatch_tool(
        self,
        tool: Callable[..., Any],
        call: ToolCall,
        context: ToolCallContext,
    ) -> Any:
        is_native_tool = isinstance(tool, Tool)
        arguments = call.arguments or {}
        if is_native_tool and arguments:
            return await tool(**arguments, context=context)
        if is_native_tool:
            return await tool(context=context)
        call_args, call_kwargs = self._bind_callable_arguments(tool, arguments)
        result = tool(*call_args, **call_kwargs)
        if isinstance(result, Awaitable):
            return await result
        return result

    @staticmethod
    def _bind_callable_arguments(
        tool: Callable[..., Any],
        arguments: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        function_signature = signature(tool)
        call_args: list[Any] = []
        call_kwargs = dict(arguments)
        for parameter in function_signature.parameters.values():
            if (
                parameter.kind is Parameter.POSITIONAL_ONLY
                and parameter.name in call_kwargs
            ):
                call_args.append(call_kwargs.pop(parameter.name))
        function_signature.bind(*call_args, **call_kwargs)
        return call_args, call_kwargs

    def _apply_transformers(
        self,
        call: ToolCall,
        context: ToolCallContext,
        result: Any,
    ) -> Any:
        if not self._settings.transformers:
            return result
        for t in self._settings.transformers:
            transformer_namespace: str | None = None
            if isinstance(t, ToolTransformer):
                transformer_func = t.func
                transformer_namespace = t.namespace
            else:
                transformer_func = t
            if not matches_tool_namespace(
                call.name,
                transformer_namespace,
            ):
                continue
            transformed = transformer_func(call, context, result)
            if isinstance(transformed, ToolTransformerResult):
                result = transformed.result
            elif transformed is not None:
                result = transformed
        return result

    @staticmethod
    def _project_error(exc: Exception) -> dict[str, object]:
        return {"type": exc.__class__.__name__}


async def _check_cancelled(context: ToolCallContext) -> None:
    if context.cancellation_checker is not None:
        await context.cancellation_checker()


def _schema_required_keys_present(
    value: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    required = schema.get("required", [])
    assert isinstance(required, list)
    return all(isinstance(key, str) and key in value for key in required)


def _schema_enum_properties_match(
    value: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    properties = schema.get("properties", {})
    assert isinstance(properties, dict)
    matched = False
    for name, field_value in value.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, dict) or "enum" not in field_schema:
            continue
        enum_values = field_schema["enum"]
        assert isinstance(enum_values, list)
        if field_value not in enum_values:
            return False
        matched = True
    return matched
