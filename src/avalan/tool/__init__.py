from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from inspect import Signature, isfunction, signature
from types import TracebackType
from typing import Any, get_type_hints

from transformers.utils import get_json_schema


class Tool(ABC):
    _exit_stack: AsyncExitStack

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()

    def json_schema(self, prefix: str | None = None) -> dict[str, Any]:
        schema = get_json_schema(self)
        if (
            prefix
            and "type" in schema
            and schema["type"] == "function"
            and "function" in schema
            and "name" in schema["function"]
        ):
            schema["function"]["name"] = prefix + schema["function"]["name"]

        return schema

    @staticmethod
    def _get_signature(
        function: Callable[..., Any], exclude_type_names: list[str]
    ) -> Signature:
        function_signature = signature(function)
        parameters = [
            param
            for param in list(function_signature.parameters.values())
            if param.name not in exclude_type_names
        ]
        return Signature(
            parameters=parameters,
            return_annotation=function_signature.return_annotation,
        )

    async def __aenter__(self) -> "Tool":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._exit_stack:
            return await self._exit_stack.__aexit__(
                exc_type, exc_value, traceback
            )
        return True


class ToolSet:
    """Collection of tools sharing an optional namespace."""

    _namespace: str | None
    _exit_stack: AsyncExitStack
    _tools: list[Callable[..., Any] | Tool | "ToolSet"]

    @property
    def namespace(self) -> str | None:
        return self._namespace

    @property
    def tools(self) -> list[Callable[..., Any] | Tool | "ToolSet"]:
        return self._tools

    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
        tools: Sequence[Callable[..., Any] | Tool | "ToolSet"],
    ):
        self._namespace = namespace
        self._exit_stack = exit_stack or AsyncExitStack()
        self._tools = list(tools)

        exclude_type_names = ["self", "context"]

        for i, tool in enumerate(self.tools):
            if (
                not isfunction(tool)
                and callable(tool)
                and isinstance(tool, Tool)
            ):
                type_hints = {
                    type_name: type_type
                    for type_name, type_type in get_type_hints(
                        tool.__call__
                    ).items()
                    if type_name not in exclude_type_names
                }
                tool.__annotations__ = type_hints
                setattr(
                    tool,
                    "__signature__",
                    Tool._get_signature(tool.__call__, exclude_type_names),
                )
                if not tool.__doc__ and tool.__call__.__doc__:
                    tool.__doc__ = tool.__call__.__doc__
                self.tools[i] = tool

    def with_enabled_tools(self, enable_tools: list[str]) -> "ToolSet":
        prefix = f"{self.namespace}." if self.namespace else ""

        tools = []
        for tool in self._tools:
            name = (
                f"{prefix}{getattr(tool, '__name__', tool.__class__.__name__)}"
            )
            for enabled in enable_tools:
                if name == enabled or name.startswith(f"{enabled}."):
                    tools.append(tool)
                    break

        self._tools = tools
        return self

    async def __aenter__(self) -> "ToolSet":
        for tool in self.tools:
            if hasattr(tool, "__aenter__"):
                await self._exit_stack.enter_async_context(tool)
            elif hasattr(tool, "__enter__"):
                self._exit_stack.enter_context(tool)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return await self._exit_stack.__aexit__(exc_type, exc_value, traceback)

    def json_schemas(
        self, prefix: str | None = None
    ) -> list[dict[str, Any]] | None:
        schemas: list[dict[str, Any]] = []
        prefix = (
            f"{prefix}."
            if prefix
            else f"{self.namespace}."
            if self.namespace
            else ""
        )
        for tool in self.tools:
            if isinstance(tool, ToolSet):
                tool_schemas = tool.json_schemas(prefix)
                if tool_schemas:
                    schemas.extend(tool_schemas)
                continue

            schema = (
                tool.json_schema(prefix)
                if isinstance(tool, Tool)
                else get_json_schema(tool)
            )
            if (
                not isinstance(tool, Tool)
                and "type" in schema
                and schema["type"] == "function"
                and "function" in schema
                and "name" in schema["function"]
            ):
                schema["function"]["name"] = (
                    prefix + schema["function"]["name"]
                )
            schemas.append(schema)
        return schemas
