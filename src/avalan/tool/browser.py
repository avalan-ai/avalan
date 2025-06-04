from . import Tool


class BrowserTool(Tool):
    """
    Uses a web browser to access the internet.

    Args:
        url: URL of a page to open

    Returns:
        Contents of the given URL page.
    """

    def __init__(self) -> None:
        self.__name__ = "open"

    async def __call__(self, url: str) -> str:
        return f"PAGE: {url}. TITLE: \"My digital self, and I\""
