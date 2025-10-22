from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings
from avalan.model.response.parsers.reasoning import ReasoningParser


class ReasoningParserExtraTestCase(IsolatedAsyncioTestCase):
    async def test_flush_returns_empty(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        await parser.push("<think>")
        await parser.push("a")
        await parser.push("</think>")
        self.assertEqual(await parser.flush(), [])

    async def test_flush_returns_pending_tokens(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        await parser.push("<")
        await parser.push("thi")
        self.assertEqual(await parser.flush(), ["<", "thi"])

    async def test_set_thinking_affects_state(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        self.assertFalse(parser.is_thinking)
        parser.set_thinking(True)
        self.assertTrue(parser.is_thinking)
        parser.set_thinking(False)
        self.assertFalse(parser.is_thinking)
