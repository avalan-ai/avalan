from ..entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolDescriptor,
    ToolFilter,
    ToolFormat,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
    ToolTransformer,
)
from . import Tool, ToolSet
from .json_schema import get_json_schema
from .parser import ToolCallParser

from asyncio import CancelledError, wait_for
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from inspect import signature
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
    _tools: dict[str, Callable[..., Any]] | None = None
    _toolsets: list[ToolSet] | None = None

    @staticmethod
    def _matches_namespace(tool_name: str, namespace: str | None) -> bool:
        if not namespace:
            return True
        return tool_name == namespace or tool_name.startswith(f"{namespace}.")

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

    def resolve_tool_name(self, name: str) -> ToolNameResolution:
        """Resolve a requested tool name against enabled tools."""
        assert isinstance(name, str)
        assert name.strip(), "name must not be empty"
        if self._tools and name in self._tools:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.EXACT,
                canonical_name=name,
                candidates=[name],
            )

        aliases = self._aliases.get(name, [])
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
            name in self._available_tool_names
            or name in self._available_aliases
        ):
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.DISABLED,
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
        try:
            signature(tool).bind(**arguments)
        except TypeError as exc:
            return ToolCallDiagnostic(
                id=uuid4(),
                call_id=call.id,
                requested_name=call.name,
                canonical_name=resolution.canonical_name,
                code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                stage=ToolCallDiagnosticStage.VALIDATE,
                message=str(exc),
            )
        return None

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
            for i, toolset in enumerate(enabled_toolsets):
                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                for tool in toolset.tools:
                    if isinstance(tool, ToolSet):
                        continue
                    name = getattr(tool, "__name__", tool.__class__.__name__)
                    canonical_name = f"{prefix}{name}"
                    self._tools[canonical_name] = cast(
                        Callable[..., Any], tool
                    )
                    aliases = self._tool_aliases(tool)
                    self._add_aliases(self._aliases, canonical_name, aliases)
                    self._descriptors[canonical_name] = ToolDescriptor(
                        name=canonical_name,
                        aliases=aliases,
                        schema=self._tool_schema(tool, prefix),
                    )

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

        tool = self._tools.get(call.name, None) if self._tools else None

        if not tool:
            return None

        await _check_cancelled(context)

        if self._settings.filters:
            for f in self._settings.filters:
                filter_namespace: str | None = None
                if isinstance(f, ToolFilter):
                    filter_func = f.func
                    filter_namespace = f.namespace
                else:
                    filter_func = f
                if not self._matches_namespace(call.name, filter_namespace):
                    continue
                modified = filter_func(call, context)
                if modified is not None:
                    assert isinstance(modified, tuple) and len(modified) == 2
                    call, context = modified

        await _check_cancelled(context)

        is_native_tool = isinstance(tool, Tool)

        try:
            result = (
                await tool(**call.arguments, context=context)
                if is_native_tool and call.arguments
                else (
                    await tool(context=context)
                    if is_native_tool
                    else (
                        await tool(*call.arguments.values())
                        if call.arguments
                        else tool()
                    )
                )
            )

            if self._settings.transformers:
                for t in self._settings.transformers:
                    transformer_namespace: str | None = None
                    if isinstance(t, ToolTransformer):
                        transformer_func = t.func
                        transformer_namespace = t.namespace
                    else:
                        transformer_func = t
                    if not self._matches_namespace(
                        call.name, transformer_namespace
                    ):
                        continue
                    transformed = transformer_func(call, context, result)
                    if transformed is not None:
                        result = transformed

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


async def _check_cancelled(context: ToolCallContext) -> None:
    if context.cancellation_checker is not None:
        await context.cancellation_checker()
