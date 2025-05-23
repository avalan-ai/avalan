from ..model.entities import ToolCall, ToolCallResult, ToolFormat
from ..tool import ToolSet
from ..tool.calculator import calculator
from ..tool.parser import ToolCallParser
from types import FunctionType
from typing import Sequence, Tuple

class ToolManager:
    _parser: ToolCallParser
    _tools: dict[str, FunctionType] | None

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
            available_toolsets = [ToolSet(tools=[calculator])]

        if enable_tools:
            enabled_toolsets = []
            for toolset in available_toolsets:
                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                tools = [
                    tool
                    for tool in toolset.tools
                    if f"{prefix}{tool.__name__}" in enable_tools
                ]
                if tools:
                    enabled_toolsets.append(
                        ToolSet(tools=tools, namespace=toolset.namespace)
                    )

        parser = ToolCallParser(
            eos_token=eos_token,
            tool_format=tool_format
        )
        return cls(
            parser=parser,
            toolsets=enabled_toolsets,
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self._tools)

    @property
    def tools(self) -> list[FunctionType] | None:
        return list(self._tools.values()) if self._tools else None

    def __init__(
        self,
        *args,
        parser: ToolCallParser,
        toolsets: Sequence[ToolSet] | None = None,
    ):
        self._parser = parser
        self._tools = None

        if toolsets:
            self._tools = {}
            for toolset in toolsets:
                prefix = f"{toolset.namespace}." if toolset.namespace else ""
                for tool in toolset.tools:
                    self._tools[f"{prefix}{tool.__name__}"] = tool

    def set_eos_token(self, eos_token: str) -> None:
        self._parser.set_eos_token(eos_token)

    def __call__(self, text: str) -> Tuple[
        list[ToolCall] | None,
        list[ToolCallResult] | None
    ]:
        tool_calls = self._parser(text)
        if not tool_calls:
            return None, None

        tool_results: list[ToolCallResult] = []
        for tool_call in tool_calls:
            tool = self._tools.get(tool_call.name, None)
            if tool:
                result = (
                    tool(*tool_call.arguments.values()) if tool_call.arguments
                    else tool()
                )
                tool_results.append(ToolCallResult(
                    call=tool_call,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=result
                ))
        return tool_calls, tool_results
