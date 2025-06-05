from . import ToolSet
from .browser import BrowserToolSet
from .math import MathToolSet
from .parser import ToolCallParser
from ..entities import ToolCall, ToolCallResult, ToolFormat
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, ContextDecorator
from uuid import uuid4


class ToolManager(ContextDecorator):
    _parser: ToolCallParser
    _stack: AsyncExitStack
    _tools: dict[str, Callable] | None = None
    _toolsets: Sequence[ToolSet] | None = None

    @classmethod
    def create_instance(
        cls,
        *,
        available_toolsets: Sequence[ToolSet] | None = None,
        enable_tools: list[str] | None = None,
        eos_token: str | None = None,
        tool_format: ToolFormat | None = None,
    ):
        if not available_toolsets:
            available_toolsets = [
                BrowserToolSet(namespace="browser"),
                MathToolSet(namespace="math")
            ]

        parser = ToolCallParser(eos_token=eos_token, tool_format=tool_format)
        return cls(
            available_toolsets=available_toolsets,
            enable_tools=enable_tools,
            parser=parser,
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self._tools)

    @property
    def tools(self) -> list[Callable] | None:
        return list(self._tools.values()) if self._tools else None

    def json_schemas(self) -> list[dict] | None:
        schemas = []
        for toolset in self._toolsets:
            schemas.extend(toolset.json_schemas())
        return schemas

    def __init__(
        self,
        *,
        available_toolsets: Sequence[ToolSet] | None = None,
        enable_tools: list[str] | None = None,
        parser: ToolCallParser,
    ):
        self._parser = parser
        self._stack = AsyncExitStack()

        enabled_toolsets = []
        for toolset in available_toolsets:
            if enable_tools:
                toolset = toolset.with_enabled_tools(enable_tools)
            if toolset.tools:
                enabled_toolsets.append(toolset)
        self._toolsets = enabled_toolsets

    def set_eos_token(self, eos_token: str) -> None:
        self._parser.set_eos_token(eos_token)

    def get_calls(self, text: str) -> list[ToolCall] | None:
        return self._parser(text)

    async def __aenter__(self) -> "ToolManager":
        self._tools = {}
        if self._toolsets:
            for i, toolset in enumerate(self._toolsets):
                await self._stack.enter_async_context(toolset)

                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                for tool in toolset.tools:
                    name = getattr(tool, "__name__", tool.__class__.__name__)
                    self._tools[f"{prefix}{name}"] = tool
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        return await self._stack.__aexit__(exc_type, exc_value, traceback)

    async def __call__(self, tool_call: ToolCall) -> ToolCallResult | None:
        """Execute a single tool call and return the result."""
        assert tool_call

        tool = self._tools.get(tool_call.name, None) if self._tools else None

        if not tool:
            return None

        result = (
            await tool(*tool_call.arguments.values())
            if tool_call.arguments
            else tool()
        )

        return ToolCallResult(
            id=uuid4(),
            call=tool_call,
            name=tool_call.name,
            arguments=tool_call.arguments,
            result=result,
        )
