from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings, ReasoningToken
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
)


class ReasoningParserTestCase(IsolatedAsyncioTestCase):
    async def test_with_thinking_tags(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        tokens = []
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

    async def test_without_thinking_tags(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        tokens = []
        for t in ["x", "y"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens, ["x", "y"])

    async def test_with_prefixes(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            prefixes=["Thought:"],
            logger=getLogger(),
        )
        tokens = []
        for t in ["Thought:", "d", "e"]:
            tokens.extend(await parser.push(t))
        self.assertIsInstance(tokens[0], ReasoningToken)
        self.assertEqual(tokens[0].token, "Thought:")
        self.assertIsInstance(tokens[1], ReasoningToken)
        self.assertEqual(tokens[1].token, "d")
        self.assertIsInstance(tokens[2], ReasoningToken)
        self.assertEqual(tokens[2].token, "e")

    async def test_without_prefixes(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            prefixes=["Thought:"],
            logger=getLogger(),
        )
        tokens = []
        for t in ["hello", "world"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens, ["hello", "world"])
