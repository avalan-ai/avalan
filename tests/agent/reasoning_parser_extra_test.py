from avalan.model.response.parsers.reasoning import ReasoningParser
from unittest import IsolatedAsyncioTestCase


class ReasoningParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_flush_returns_empty(self):
        parser = ReasoningParser()
        await parser.push("<think>")
        await parser.push("a")
        await parser.push("</think>")
        self.assertEqual(await parser.flush(), [])
