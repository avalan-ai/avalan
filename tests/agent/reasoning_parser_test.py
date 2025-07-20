from avalan.agent.orchestrator.response.parsers.reasoning import (
    ReasoningParser,
)
from unittest import IsolatedAsyncioTestCase


class ReasoningParserTestCase(IsolatedAsyncioTestCase):
    async def test_with_thinking_tags(self):
        parser = ReasoningParser()
        tokens = []
        for t in ["a", "<think>", "b", "</think>", "c"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "a")
        self.assertEqual(tokens[1], "<think>")
        self.assertEqual(getattr(tokens[2], "tag", None), "think")
        self.assertEqual(tokens[3], "</think>")
        self.assertEqual(tokens[4], "c")

    async def test_without_thinking_tags(self):
        parser = ReasoningParser()
        tokens = []
        for t in ["x", "y"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens, ["x", "y"])

    async def test_with_prefixes(self):
        parser = ReasoningParser(prefixes=["Thought:"])
        tokens = []
        for t in ["Thought:", "d", "e"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "Thought:")
        self.assertEqual(getattr(tokens[1], "tag", None), "think")
        self.assertEqual(getattr(tokens[2], "tag", None), "think")

    async def test_without_prefixes(self):
        parser = ReasoningParser(prefixes=["Thought:"])
        tokens = []
        for t in ["hello", "world"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens, ["hello", "world"])
