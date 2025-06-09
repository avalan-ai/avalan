from . import Tool


class SearchEngineTool(Tool):
    """
    Can search Internet search engines for real time information.

    Args:
        query: Term to search for. Example: "weather today in Buenos Aires"
        engine: Search engine to use. Example: "google"

    Returns:
        Result of searching Google for the given term.
    """

    def __init__(self) -> None:
        self.__name__ = "search"

    async def __call__(self, query: str, engine: str) -> str:
        return (
            "The weather is nice and warm, with 23 degrees celsius, clear"
            " skies, and winds under 11 kmh."
        )
