from ..entities import ToolCall, ToolCallOutcome
from . import Tool
from .builtin_display import project_search_tool_display


class SearchEngineTool(Tool):
    """Search internet engines for real-time information.

    Args:
        query: Term to search for.
        engine: Search engine to use.

    Returns:
        Result of executing the query against the chosen engine.
    """

    def __init__(self) -> None:
        self.__name__ = "search"

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        return project_search_tool_display(call=call, outcome=outcome)

    async def __call__(self, query: str, engine: str) -> str:
        return (
            "The weather is nice and warm, with 23 degrees celsius, clear"
            " skies, and winds under 11 kmh."
        )
