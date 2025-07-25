from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.entities import ReasoningSettings
from unittest import IsolatedAsyncioTestCase


class ReasoningParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_flush_returns_empty(self):
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        await parser.push("<think>")
        await parser.push("a")
        await parser.push("</think>")
        self.assertEqual(await parser.flush(), [])

    async def test_set_thinking_affects_state(self):
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        self.assertFalse(parser.is_thinking)
        parser.set_thinking(True)
        self.assertTrue(parser.is_thinking)
        parser.set_thinking(False)
        self.assertFalse(parser.is_thinking)
