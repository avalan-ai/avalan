from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, ContextDecorator
from inspect import isfunction, signature, Signature
from transformers.utils import get_json_schema
from types import FunctionType
from typing import get_type_hints


class Tool(ABC, ContextDecorator):
    _exit_stack: AsyncExitStack

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()

    def json_schema(self, prefix: str | None = None) -> dict:
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
        function: FunctionType, exclude_type_names: list[str]
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

    async def __aenter__(self) -> "ToolSet":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        if self._exit_stack:
            return await self._exit_stack.__aexit__(
                exc_type, exc_value, traceback
            )
        return True


class ToolSet(ContextDecorator):
    """Collection of tools sharing an optional namespace."""

    _namespace: str | None
    _exit_stack: AsyncExitStack
    _tools: Sequence[Callable]

    @property
    def namespace(self) -> str | None:
        return self._namespace

    @property
    def tools(self) -> Sequence[Callable]:
        return self._tools

    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
        tools: Sequence[Callable | "ToolSet"],
    ):
        self._namespace = namespace
        self._exit_stack = exit_stack or AsyncExitStack()
        self._tools = tools

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
                tool.__signature__ = Tool._get_signature(
                    tool.__call__, exclude_type_names
                )
                if not tool.__doc__ and tool.__call__.__doc__:
                    tool.__doc__ = tool.__call__.__doc__
                self.tools[i] = tool

    def with_enabled_tools(self, enable_tools: list[str]) -> "ToolSet":
        prefix = f"{self.namespace}." if self.namespace else ""
        tools = [
            tool
            for tool in self._tools
            if f"{prefix}{getattr(tool, '__name__', tool.__class__.__name__)}"
            in enable_tools
        ]
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
        traceback: BaseException | None,
    ) -> bool:
        return await self._exit_stack.__aexit__(exc_type, exc_value, traceback)

    def json_schemas(self, prefix: str | None = None) -> list[dict] | None:
        schemas = []
        prefix = (
            f"{prefix}."
            if prefix
            else f"{self.namespace}." if self.namespace else ""
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
