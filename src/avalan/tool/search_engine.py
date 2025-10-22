from . import Tool


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

    async def __call__(self, query: str, engine: str) -> str:
        return (
            "The weather is nice and warm, with 23 degrees celsius, clear"
            " skies, and winds under 11 kmh."
        )
