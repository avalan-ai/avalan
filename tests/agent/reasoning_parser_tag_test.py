from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.entities import ReasoningSettings, ReasoningTag, ReasoningToken
from logging import getLogger
from unittest import IsolatedAsyncioTestCase


class ReasoningParserTagTestCase(IsolatedAsyncioTestCase):
    async def test_think_tag(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(tag=ReasoningTag.THINK),
            logger=getLogger(),
        )
        tokens: list[object] = []
        for t in ["a", "<think>", "b", "</think>", "c"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "a")
        self.assertIsInstance(tokens[1], ReasoningToken)
        self.assertEqual(tokens[1].token, "<think>")
        self.assertIsInstance(tokens[2], ReasoningToken)
        self.assertEqual(tokens[2].token, "b")
        self.assertIsInstance(tokens[3], ReasoningToken)
        self.assertEqual(tokens[3].token, "</think>")
        self.assertEqual(tokens[4], "c")

    async def test_channel_tag(self):
        start = "<|channel|>analysis<|message|>"
        end = "<|end|><|start|>assistant<|channel|>final<|message|>"
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(tag=ReasoningTag.CHANNEL),
            logger=getLogger(),
        )
        tokens: list[object] = []
        for t in ["x", start, "y", end, "z"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "x")
        self.assertIsInstance(tokens[1], ReasoningToken)
        self.assertEqual(tokens[1].token, start)
        self.assertIsInstance(tokens[2], ReasoningToken)
        self.assertEqual(tokens[2].token, "y")
        self.assertIsInstance(tokens[3], ReasoningToken)
        self.assertEqual(tokens[3].token, end)
        self.assertEqual(tokens[4], "z")
