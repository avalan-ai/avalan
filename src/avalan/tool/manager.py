from ..entities import ToolCall, ToolCallResult, ToolFormat
from ..tool import ToolSet
from ..tool.calculator import CalculatorTool
from ..tool.parser import ToolCallParser
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, ContextDecorator
from uuid import uuid4


class ToolManager(ContextDecorator):
    _parser: ToolCallParser
    _tools: dict[str, Callable] | None
    _toolsets: Sequence[ToolSet] | None
    _stack: AsyncExitStack

    @classmethod
    def create_instance(
        cls,
        *args,
        eos_token: str | None = None,
        enable_tools: list[str] | None = None,
        tool_format: ToolFormat | None = None,
        available_toolsets: Sequence[ToolSet] | None = None,
    ):
        enabled_toolsets: list[ToolSet] | None = None

        if not available_toolsets:
            available_toolsets = [ToolSet(tools=[CalculatorTool()])]

        if enable_tools:
            enabled_toolsets = []
            for toolset in available_toolsets:
                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                tools = [
                    tool
                    for tool in toolset.tools
                    if f"{prefix}{getattr(tool, '__name__', tool.__class__.__name__)}"
                    in enable_tools
                ]
                if tools:
                    enabled_toolsets.append(
                        ToolSet(tools=tools, namespace=toolset.namespace)
                    )

        parser = ToolCallParser(eos_token=eos_token, tool_format=tool_format)
        return cls(
            parser=parser,
            toolsets=enabled_toolsets,
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self._tools)

    @property
    def tools(self) -> list[Callable] | None:
        return list(self._tools.values()) if self._tools else None

    def __init__(
        self,
        *args,
        parser: ToolCallParser,
        toolsets: Sequence[ToolSet] | None = None,
    ):
        self._parser = parser
        self._tools = None
        self._toolsets = toolsets
        self._stack = AsyncExitStack()

        if toolsets:
            self._tools = {}
            for toolset in toolsets:
                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                for tool in toolset.tools:
                    name = getattr(tool, "__name__", tool.__class__.__name__)
                    self._tools[f"{prefix}{name}"] = tool

    def set_eos_token(self, eos_token: str) -> None:
        self._parser.set_eos_token(eos_token)

    def get_calls(self, text: str) -> list[ToolCall] | None:
        return self._parser(text)

    async def __aenter__(self) -> "ToolManager":
        if self._toolsets:
            for toolset in self._toolsets:
                await self._stack.enter_async_context(toolset)
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
