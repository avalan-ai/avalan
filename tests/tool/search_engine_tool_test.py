from avalan.tool.search_engine import SearchEngineTool
from unittest import IsolatedAsyncioTestCase, main


class SearchEngineToolTestCase(IsolatedAsyncioTestCase):
    async def test_call_returns_placeholder(self):
        tool = SearchEngineTool()
        result = await tool("rain", engine="google")
        self.assertEqual(tool.__name__, "search")
        expected = (
            "The weather is nice and warm, with 23 degrees celsius, "
            "clear skies, and winds under 11 kmh."
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    main()
