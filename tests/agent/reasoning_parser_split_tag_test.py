from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings, ReasoningToken
from avalan.model.response.parsers.reasoning import ReasoningParser


class ReasoningParserSplitTagTestCase(IsolatedAsyncioTestCase):
    async def test_split_start_and_end_tags(self) -> None:
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        outputs = []
        for text in ["<", "think", ">", "a", "b", "<", "/think", ">"]:
            outputs.extend(await parser.push(text))
        self.assertTrue(all(isinstance(t, ReasoningToken) for t in outputs))
        self.assertEqual(
            [t.token for t in outputs],
            ["<", "think", ">", "a", "b", "<", "/think", ">"],
        )
        self.assertFalse(parser.is_thinking)

    async def test_unmatched_partial_tag(self) -> None:
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        outputs = []
        for t in ["<", "unknown", ">"]:
            outputs.extend(await parser.push(t))
        self.assertEqual(outputs, ["<", "unknown", ">"])
