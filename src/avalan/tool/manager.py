from ..entities import (
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolDescriptor,
    ToolFilter,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolFormat,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
    ToolTransformer,
)
from . import Tool, ToolSet
from .json_schema import get_json_schema
from .names import matches_tool_namespace
from .parser import ToolCallParser

from asyncio import CancelledError, wait_for
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from inspect import signature
from re import compile as compile_regex
from types import TracebackType
from typing import Any, cast
from uuid import uuid4


class ToolManager:
    _INTERRUPT_CLOSE_TIMEOUT = 0.5
    _PROVIDER_TOOL_NAME_PATTERN = compile_regex(r"^[A-Za-z0-9_-]+$")
    _PROVIDER_TOOL_NAME_PREFIX = "avl_"

    _parser: ToolCallParser
    _stack: AsyncExitStack
    _aliases: dict[str, list[str]]
    _available_aliases: dict[str, list[str]]
    _available_tool_names: set[str]
    _descriptors: dict[str, ToolDescriptor]
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
            tool_format=settings.tool_format if settings else None,
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

    def json_schemas(self) -> list[dict[str, Any]] | None:
        schemas: list[dict[str, Any]] = []
        for toolset in self._toolsets or []:
            toolset_schemas = toolset.json_schemas()
            if toolset_schemas:
                schemas.extend(toolset_schemas)
        return schemas

    def list_tools(self) -> list[ToolDescriptor]:
        """Return descriptors for enabled tools."""
        return list(self._descriptors.values())

    def describe_tool(self, name: str) -> ToolDescriptor | None:
        """Return the descriptor for an enabled tool."""
        resolution = self.resolve_tool_name(name)
        if resolution.canonical_name is None:
            return None
        return self._descriptors.get(resolution.canonical_name)

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

        if (
            canonical_request in self._available_tool_names
            or canonical_request in self._available_aliases
        ):
            candidates = self._available_aliases.get(
                canonical_request,
                [canonical_request],
            )
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.DISABLED,
                candidates=candidates,
                diagnostic_code=ToolCallDiagnosticCode.DISABLED_TOOL,
            )

        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
            diagnostic_code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        """Return a diagnostic when a tool call is not executable."""
        resolution = self.resolve_tool_name(call.name)
        if resolution.diagnostic_code is not None:
            return ToolCallDiagnostic(
                id=uuid4(),
                call_id=call.id,
                requested_name=call.name,
                canonical_name=resolution.canonical_name,
                code=resolution.diagnostic_code,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message=f"Tool '{call.name}' is {resolution.status.value}.",
                details={"candidates": cast(Any, resolution.candidates)},
            )

        arguments = call.arguments or {}
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

        assert resolution.canonical_name is not None
        tool = (
            self._tools.get(resolution.canonical_name) if self._tools else None
        )
        assert tool is not None
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
            provider_safe_schema=cls._provider_safe_schema(schema),
            namespace=namespace,
        )

    @classmethod
    def _provider_safe_schema(
        cls, schema: dict[str, Any] | None
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
            function["name"] = cls._encode_provider_tool_name(function["name"])
        return provider_schema

    @classmethod
    def _encode_provider_tool_name(cls, tool_name: str) -> str:
        assert tool_name.strip(), "tool name must not be empty"
        if cls._PROVIDER_TOOL_NAME_PATTERN.fullmatch(
            tool_name
        ) and not tool_name.startswith(cls._PROVIDER_TOOL_NAME_PREFIX):
            return tool_name
        encoded = urlsafe_b64encode(tool_name.encode()).decode().rstrip("=")
        return f"{cls._PROVIDER_TOOL_NAME_PREFIX}{encoded}"

    @classmethod
    def _canonical_requested_name(
        cls, name: str, *, provider_originated: bool
    ) -> str:
        requested_name = (
            cls._decode_provider_tool_name(name)
            if provider_originated
            else name
        )
        return requested_name.removeprefix("functions.")

    @classmethod
    def _decode_provider_tool_name(cls, tool_name: str) -> str:
        assert tool_name.strip(), "tool name must not be empty"
        assert cls._PROVIDER_TOOL_NAME_PATTERN.fullmatch(
            tool_name
        ), "provider tool name is invalid"

        if not tool_name.startswith(cls._PROVIDER_TOOL_NAME_PREFIX):
            return tool_name

        payload = tool_name[len(cls._PROVIDER_TOOL_NAME_PREFIX) :]
        assert payload, "provider tool name is missing encoded content"
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = urlsafe_b64decode(f"{payload}{padding}").decode()
        except (BinasciiError, UnicodeDecodeError) as exc:
            raise AssertionError("provider tool name is malformed") from exc
        assert decoded.strip(), "decoded tool name must not be empty"
        assert (
            cls._encode_provider_tool_name(decoded) == tool_name
        ), "provider tool name is malformed"
        return decoded

    def is_potential_tool_call(self, buffer: str, token_str: str) -> bool:
        """Proxy :meth:`ToolCallParser.is_potential_tool_call`."""
        return self._parser.is_potential_tool_call(buffer, token_str)

    def tool_call_status(
        self, buffer: str
    ) -> ToolCallParser.ToolCallBufferStatus:
        """Proxy :meth:`ToolCallParser.tool_call_status`."""
        return self._parser.tool_call_status(buffer)

    def get_calls(self, text: str) -> list[ToolCall] | None:
        calls = self._parser.parse(text).calls
        return calls or None

    async def prepare_call(
        self, call: ToolCall, context: ToolCallContext
    ) -> PreparedToolCall | ToolCallDiagnostic:
        """Return a prepared execution plan or a diagnostic."""
        assert call

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

        await _check_cancelled(context)

        prepared = self._prepare_resolved_call(
            call,
            context,
            validate=False,
        )
        if isinstance(prepared, ToolCallDiagnostic):
            return prepared

        filtered = self._apply_filters(prepared.call, prepared.context)
        if isinstance(filtered, ToolCallDiagnostic):
            return filtered
        call, context = filtered
        await _check_cancelled(context)

        return self._prepare_resolved_call(call, context)

    async def execute_prepared_call(
        self, prepared: PreparedToolCall
    ) -> ToolCallResult | ToolCallError:
        """Execute a prepared call without rerunning preparation."""
        assert isinstance(prepared, PreparedToolCall)
        await _check_cancelled(prepared.context)
        return await self._execute_prepared_call(prepared)

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
    ) -> ToolCallResult | ToolCallError | None:
        """Execute a single tool call and return the result."""
        assert call

        history = context.calls or []

        if self._settings.avoid_repetition and history:
            last = history[-1]
            if last.name == call.name and last.arguments == call.arguments:
                return None

        if (
            self._settings.maximum_depth is not None
            and len(history) + 1 > self._settings.maximum_depth
        ):
            return None

        if not self._tools or call.name not in self._tools:
            return None

        await _check_cancelled(context)

        filtered = self._apply_filters(call, context)
        if isinstance(filtered, ToolCallDiagnostic):
            return None
        call, context = filtered

        await _check_cancelled(context)

        prepared = self._prepare_resolved_call(
            call,
            context,
            validate=False,
        )
        if isinstance(prepared, ToolCallDiagnostic):
            return None
        return await self._execute_prepared_call(prepared)

    def _prepare_resolved_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
        *,
        validate: bool = True,
    ) -> PreparedToolCall | ToolCallDiagnostic:
        resolution = self.resolve_tool_name(call.name)
        if resolution.diagnostic_code is not None:
            return self._resolution_diagnostic(call, resolution)
        assert resolution.canonical_name is not None

        arguments = call.arguments or {}
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
        prepared_call = ToolCall(
            id=call.id,
            name=resolution.canonical_name,
            arguments=arguments,
        )
        if validate:
            diagnostic = self._validate_tool_arguments(
                call=prepared_call,
                canonical_name=resolution.canonical_name,
                tool=tool,
                arguments=arguments,
                context=context,
            )
            if diagnostic is not None:
                return diagnostic
        return PreparedToolCall(
            call=prepared_call,
            callable=tool,
            descriptor=descriptor,
            arguments=arguments,
            context=context,
        )

    def _resolution_diagnostic(
        self, call: ToolCall, resolution: ToolNameResolution
    ) -> ToolCallDiagnostic:
        assert resolution.diagnostic_code is not None
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=call.name,
            canonical_name=resolution.canonical_name,
            code=resolution.diagnostic_code,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message=f"Tool '{call.name}' is {resolution.status.value}.",
            details={"candidates": cast(Any, resolution.candidates)},
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
            requested_name=call.name,
            canonical_name=canonical_name,
            code=code,
            stage=stage,
            message=message,
            details=details or {},
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
        try:
            if isinstance(tool, Tool):
                signature(tool.__call__).bind(
                    **arguments,
                    context=context or ToolCallContext(),
                )
            else:
                signature(tool).bind(**arguments)
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

    def _apply_filters(
        self, call: ToolCall, context: ToolCallContext
    ) -> tuple[ToolCall, ToolCallContext] | ToolCallDiagnostic:
        if not self._settings.filters:
            return call, context

        for f in self._settings.filters:
            filter_namespace: str | None = None
            if isinstance(f, ToolFilter):
                filter_func = f.func
                filter_namespace = f.namespace
            else:
                filter_func = f
            if not matches_tool_namespace(call.name, filter_namespace):
                continue
            modified = filter_func(call, context)
            if modified is None:
                continue
            if isinstance(modified, ToolFilterResult):
                if modified.status is ToolFilterResultStatus.PASS:
                    continue
                if modified.status is ToolFilterResultStatus.SUPPRESS:
                    return self._diagnostic(
                        call=call,
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
            if context.flow_tool_node and next_call.name != call.name:
                return self._diagnostic(
                    call=call,
                    code=ToolCallDiagnosticCode.FILTER_SUPPRESSED,
                    stage=ToolCallDiagnosticStage.FILTER,
                    message="Tool filter name rewrites are disabled.",
                    details={"filtered_name": next_call.name},
                )
            call, context = next_call, next_context
        return call, context

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
                result=result,
            )
        except Exception as exc:
            return ToolCallError(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                error=exc,
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
        if arguments:
            return await tool(*arguments.values())
        return tool()

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
            if transformed is not None:
                result = transformed
        return result


async def _check_cancelled(context: ToolCallContext) -> None:
    if context.cancellation_checker is not None:
        await context.cancellation_checker()
