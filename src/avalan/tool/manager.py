from avalan.model.entities import ToolCall, ToolCallResult, ToolFormat
from avalan.tool.calculator import calculator
from avalan.tool.parser import ToolCallParser
from types import FunctionType
from typing import Optional, Tuple

class ToolManager:
    _parser: ToolCallParser
    _tools: Optional[dict[str,FunctionType]]

    @classmethod
    def create_instance(
        cls,
        *args,
        eos_token: Optional[str]=None,
        enable_tools: Optional[list[str]]=None,
        tool_format: Optional[ToolFormat]=None,
        available_tools: Optional[list[FunctionType]]=None
    ):
        enabled_tools: Optional[list[FunctionType]] = None

        if not available_tools:
            available_tools = [
                calculator
            ]

        if available_tools and enable_tools:
            enabled_tools = [
                tool
                for tool in available_tools
                if tool.__name__ in enable_tools
            ]

        parser = ToolCallParser(
            eos_token=eos_token,
            tool_format=tool_format
        )
        return cls(
            parser=parser,
            tools=enabled_tools
        )

    @property
    def is_empty(self) -> bool:
        return not bool(self._tools)

    @property
    def tools(self) -> Optional[list[FunctionType]]:
        return list(self._tools.values()) if self._tools else None

    def __init__(
        self,
        *args,
        parser: ToolCallParser,
        tools: Optional[list[FunctionType]]=None
    ):
        self._parser = parser
        self._tools = None

        if tools:
            self._tools = {}
            for tool in tools:
                self._tools[tool.__name__] = tool

    def set_eos_token(self, eos_token: str) -> None:
        self._parser.set_eos_token(eos_token)

    def __call__(self, text: str) -> Tuple[
        Optional[list[ToolCall]],
        Optional[list[ToolCallResult]]
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
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=result
                ))
        return tool_calls, tool_results
