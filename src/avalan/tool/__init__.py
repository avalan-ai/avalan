from abc import ABC
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from inspect import isfunction, signature, Signature
from types import FunctionType
from typing import get_type_hints


class Tool(ABC):
    @staticmethod
    def _get_signature(function: FunctionType) -> Signature:
        return Signature(
            parameters=list(signature(function).parameters.values())[
                1:
            ],  # drop "self"
            return_annotation=signature(function).return_annotation,
        )


@dataclass(frozen=True, kw_only=True)
class ToolSet:
    """Collection of tools sharing an optional namespace."""

    tools: Sequence[Callable]
    namespace: str | None = None
    _stack: AsyncExitStack = field(
        default_factory=AsyncExitStack, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for tool in self.tools:
            if callable(tool) and not isfunction(tool):
                tool.__annotations__ = get_type_hints(tool.__call__)
                tool.__signature__ = Tool._get_signature(tool.__call__)
                if not tool.__doc__ and tool.__call__.__doc__:
                    tool.__doc__ = tool.__call__.__doc__

    async def __aenter__(self) -> "ToolSet":
        for tool in self.tools:
            if hasattr(tool, "__aenter__"):
                await self._stack.enter_async_context(tool)
            elif hasattr(tool, "__enter__"):
                self._stack.enter_context(tool)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        return await self._stack.__aexit__(exc_type, exc_value, traceback)
