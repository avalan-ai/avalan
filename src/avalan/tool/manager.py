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
    ToolCallParseOutcome,
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
    ToolNamePolicyMode,
    ToolNameResolution,
    ToolNameResolutionStatus,
    ToolProviderArgumentsMode,
    ToolTransformer,
    ToolTransformerResult,
)
from . import Tool, ToolSet
from .json_schema import get_json_schema
from .name_policy import ToolNamePolicy
from .names import matches_tool_namespace
from .parser import ToolCallParser

from asyncio import CancelledError, wait_for
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import replace
from inspect import Parameter, signature
from types import TracebackType
from typing import Any, cast
from uuid import uuid4


class ToolManager:
    _INTERRUPT_CLOSE_TIMEOUT = 0.5

    _parser: ToolCallParser
    _stack: AsyncExitStack
    _aliases: dict[str, list[str]]
    _available_aliases: dict[str, list[str]]
    _available_tool_names: set[str]
    _descriptors: dict[str, ToolDescriptor]
    _tool_name_policy: ToolNamePolicy
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

    def json_schemas(self) -> list[dict[str, Any]] | None:
        schemas: list[dict[str, Any]] = []
        for toolset in self._toolsets or []:
            toolset_schemas = toolset.json_schemas()
            if toolset_schemas:
                schemas.extend(toolset_schemas)
        return schemas

    def provider_json_schemas(
        self, *, provider_family: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Return enabled tool schemas with provider-facing names."""
        policy = self._tool_name_policy.for_provider(provider_family)
        schemas: list[dict[str, Any]] = []
        for descriptor in self._descriptors.values():
            schema = self._provider_safe_schema(
                descriptor.schema,
                policy=policy,
            )
            if schema is not None:
                schemas.append(schema)
        return schemas or None

    def provider_tool_name(
        self,
        canonical_name: str,
        *,
        provider_family: str | None = None,
    ) -> str:
        """Return the provider-facing name for ``canonical_name``."""
        return self._tool_name_policy.for_provider(
            provider_family
        ).provider_name(canonical_name)

    def canonical_tool_name(
        self,
        provider_name: str,
        *,
        provider_family: str | None = None,
    ) -> str:
        """Return the canonical name for a provider-facing name."""
        return self._tool_name_policy.for_provider(
            provider_family
        ).canonical_name(provider_name)

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for enabled tools."""
        return list(self._descriptors.values())

    def describe_tool(self, name: str) -> ToolDescriptor | None:
        """Return the descriptor for an enabled tool."""
        resolution = self.resolve_tool_name(name)
        if resolution.canonical_name is None:
            return None
        return self._descriptors.get(resolution.canonical_name)

    def describe_tool_call(self, call: ToolCall) -> ToolDescriptor | None:
        """Return the descriptor for an executable tool call."""
        assert isinstance(call, ToolCall)
        try:
            resolution = self._resolve_call_name(call)
        except AssertionError:
            return None
        if resolution.canonical_name is None:
            return None
        return self._descriptors.get(resolution.canonical_name)

    def is_tool_call_parallel_safe(self, call: ToolCall) -> bool:
        """Return whether ``call`` may execute in a parallel fanout."""
        descriptor = self.describe_tool_call(call)
        if descriptor is None:
            return False
        return descriptor.capabilities.parallel_safe

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution:
        """Resolve a requested tool name against enabled tools."""
        assert isinstance(name, str)
        assert name.strip(), "name must not be empty"
        assert isinstance(provider_originated, bool)

        canonical_request = self._canonical_requested_name(
            name,
            provider_originated=provider_originated,
        )
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
        self._parser = parser
        self._settings = settings or ToolManagerSettings()
        self._stack = AsyncExitStack()
        self._aliases = {}
        self._available_aliases = {}
        self._available_tool_names = set()
        self._descriptors = {}

        if available_toolsets:
            for toolset in available_toolsets:
                for name, aliases in self._toolset_tool_names(toolset):
                    self._available_tool_names.add(name)
                    self._add_aliases(self._available_aliases, name, aliases)

        enabled_toolsets = []
        if available_toolsets:
            for toolset in available_toolsets:
                if enable_tools is not None:
                    toolset = toolset.with_enabled_tools(enable_tools)
                if toolset.tools:
                    enabled_toolsets.append(toolset)

        self._tools = {}
        enabled_names: list[str] = []
        for toolset in enabled_toolsets:
            enabled_names.extend(
                name for name, _aliases in self._toolset_tool_names(toolset)
            )
        self._tool_name_policy = ToolNamePolicy(
            settings=self._settings.tool_name_policy
        ).bind(enabled_names)
        if enabled_toolsets:
            for toolset in enabled_toolsets:
                self._register_toolset(toolset)

        self._toolsets = enabled_toolsets

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

    @classmethod
    def _toolset_tool_names(
        cls, toolset: ToolSet, prefix: str | None = None
    ) -> list[tuple[str, list[str]]]:
        namespace_prefix = (
            f"{prefix}.{toolset.namespace}"
            if prefix and toolset.namespace
            else prefix or toolset.namespace
        )
        tool_prefix = f"{namespace_prefix}." if namespace_prefix else ""
        names: list[tuple[str, list[str]]] = []
        for tool in toolset.tools:
            if isinstance(tool, ToolSet):
                names.extend(cls._toolset_tool_names(tool, namespace_prefix))
                continue
            name = getattr(tool, "__name__", tool.__class__.__name__)
            names.append((f"{tool_prefix}{name}", cls._tool_aliases(tool)))
        return names

    def _register_toolset(
        self, toolset: ToolSet, prefix: str | None = None
    ) -> None:
        namespace = (
            f"{prefix}.{toolset.namespace}"
            if prefix and toolset.namespace
            else prefix or toolset.namespace
        )
        tool_prefix = f"{namespace}." if namespace else ""
        for tool in toolset.tools:
            if isinstance(tool, ToolSet):
                self._register_toolset(tool, namespace)
                continue

            assert self._tools is not None
            name = getattr(tool, "__name__", tool.__class__.__name__)
            canonical_name = f"{tool_prefix}{name}"
            self._tools[canonical_name] = cast(Callable[..., Any], tool)
            aliases = self._tool_aliases(tool)
            self._add_aliases(self._aliases, canonical_name, aliases)
            schema = self._tool_schema(tool, tool_prefix)
            self._descriptors[canonical_name] = self._tool_descriptor(
                canonical_name=canonical_name,
                tool=tool,
                aliases=aliases,
                namespace=namespace,
                schema=schema,
                policy=self._tool_name_policy,
            )

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
        policy: ToolNamePolicy | None = None,
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
            provider_safe_schema=cls._provider_safe_schema(
                schema,
                policy=policy or ToolNamePolicy.default(),
            ),
            namespace=namespace,
            capabilities=cls._tool_capabilities(tool),
            metadata=cls._tool_metadata(tool),
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

    @staticmethod
    def _provider_safe_schema(
        schema: dict[str, Any] | None,
        *,
        policy: ToolNamePolicy,
    ) -> dict[str, Any] | None:
        if not schema:
            return None
        provider_schema = deepcopy(schema)
        function = provider_schema.get("function")
        if (
            provider_schema.get("type") == "function"
            and isinstance(function, dict)
            and isinstance(function.get("name"), str)
        ):
            function["name"] = policy.provider_name(function["name"])
        return provider_schema

    def _canonical_requested_name(
        self, name: str, *, provider_originated: bool
    ) -> str:
        requested_name = (
            self._tool_name_policy.canonical_name(name)
            if provider_originated
            else name
        )
        return requested_name.removeprefix("functions.")

    def is_potential_tool_call(self, buffer: str, token_str: str) -> bool:
        """Proxy :meth:`ToolCallParser.is_potential_tool_call`."""
        return self._parser.is_potential_tool_call(buffer, token_str)

    def tool_call_status(
        self, buffer: str, *, final: bool = False
    ) -> ToolCallParser.ToolCallBufferStatus:
        """Proxy :meth:`ToolCallParser.tool_call_status`."""
        return self._parser.tool_call_status(buffer, final=final)

    def parse_calls(self, text: str) -> ToolCallParseOutcome:
        """Return parsed calls and diagnostics for ``text``."""
        outcome = self._parser.parse(text)
        if not outcome.calls:
            return outcome
        return ToolCallParseOutcome(
            calls=[
                self._canonical_provider_originated_call(call)
                for call in outcome.calls
            ],
            diagnostics=outcome.diagnostics,
        )

    def stream_buffer_diagnostics(
        self, buffer: str
    ) -> list[ToolCallDiagnostic]:
        """Return diagnostics for a terminal streaming buffer."""
        return self._parser.stream_buffer_diagnostics(buffer)

    def get_calls(self, text: str) -> list[ToolCall] | None:
        calls = self.parse_calls(text).calls
        return calls or None

    def _canonical_provider_originated_call(self, call: ToolCall) -> ToolCall:
        try:
            canonical_name = self._tool_name_policy.canonical_name(call.name)
        except AssertionError:
            if (
                self._settings.tool_name_policy.mode
                is ToolNamePolicyMode.ENCODED
                and not call.name.startswith(
                    self._settings.tool_name_policy.prefix
                )
            ):
                return call
            return ToolCall(
                id=call.id,
                name="",
                arguments=call.arguments,
                provider_name=call.name,
                provider_name_encoded=call.provider_name_encoded,
                provider_arguments_malformed=(
                    call.provider_arguments_malformed
                ),
            )
        if canonical_name == call.name:
            return call
        return ToolCall(
            id=call.id,
            name=canonical_name,
            arguments=call.arguments,
            provider_name=call.name,
            provider_name_encoded=call.provider_name_encoded,
            provider_arguments_malformed=call.provider_arguments_malformed,
        )

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
        provider_name = call.provider_name
        if provider_name is None:
            return self.resolve_tool_name(call.name)
        try:
            return self.resolve_tool_name(
                provider_name,
                provider_originated=True,
            )
        except AssertionError:
            return ToolNameResolution(
                requested_name=provider_name,
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
            if any(
                isinstance(candidate, dict)
                and cls._schema_validation_error(value, candidate, path)
                is None
                for candidate in any_of
            ):
                return None
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
        if call.provider_name is None:
            try:
                resolution = self.resolve_tool_name(call.name)
            except AssertionError:
                return call.name
            return resolution.canonical_name or resolution.requested_name
        try:
            resolution = self.resolve_tool_name(
                call.provider_name,
                provider_originated=True,
            )
        except AssertionError:
            return call.provider_name
        return resolution.canonical_name or resolution.requested_name

    def _diagnostic_canonical_name(self, call: ToolCall) -> str | None:
        try:
            resolution = self._resolve_call_name(call)
        except AssertionError:
            return None
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
