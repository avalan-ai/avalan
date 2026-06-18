from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
)
from avalan.model.stream import StreamItemKind, StreamProviderEvent


def _reasoning_delta_text(item: object) -> str:
    assert isinstance(item, StreamProviderEvent)
    assert item.kind is StreamItemKind.REASONING_DELTA
    assert item.text_delta is not None
    return item.text_delta


def _assert_reasoning_done(item: object) -> None:
    assert isinstance(item, StreamProviderEvent)
    assert item.kind is StreamItemKind.REASONING_DONE


class ReasoningParserTestCase(IsolatedAsyncioTestCase):
    async def test_with_thinking_tags(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        tokens = []
        for t in ["a", "<think>", "b", "</think>", "c"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "a")
        self.assertEqual(_reasoning_delta_text(tokens[1]), "<think>")
        self.assertEqual(_reasoning_delta_text(tokens[2]), "b")
        self.assertEqual(_reasoning_delta_text(tokens[3]), "</think>")
        _assert_reasoning_done(tokens[4])
        self.assertEqual(tokens[5], "c")

    async def test_without_thinking_tags(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
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
        self.assertEqual(
            [_reasoning_delta_text(token) for token in tokens],
            ["Thought:", "d", "e"],
        )

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
