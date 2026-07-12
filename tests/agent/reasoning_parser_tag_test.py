from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings, ReasoningTag
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.stream import (
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamVisibility,
)


def _reasoning_delta_text(item: object) -> str:
    assert isinstance(item, StreamProviderEvent)
    assert item.kind is StreamItemKind.REASONING_DELTA
    assert item.text_delta is not None
    assert (
        item.reasoning_representation
        is StreamReasoningRepresentation.NATIVE_TEXT
    )
    assert item.segment_instance_ordinal == 0
    assert item.visibility is StreamVisibility.PRIVATE
    return item.text_delta


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
        self.assertEqual(_reasoning_delta_text(tokens[1]), "<think>")
        self.assertEqual(_reasoning_delta_text(tokens[2]), "b")
        self.assertEqual(_reasoning_delta_text(tokens[3]), "</think>")
        self.assertEqual(tokens[4], "c")

    async def test_channel_tag(self):
        start = "<|channel|>analysis<|message|>"
        end = "<|end|>"
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(tag=ReasoningTag.CHANNEL),
            logger=getLogger(),
        )
        tokens: list[object] = []
        for t in ["x", start, "y", end, "z"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(tokens[0], "x")
        self.assertEqual(_reasoning_delta_text(tokens[1]), start)
        self.assertEqual(_reasoning_delta_text(tokens[2]), "y")
        self.assertEqual(_reasoning_delta_text(tokens[3]), end)
        self.assertEqual(tokens[4], "z")

    async def test_auto_tag_think(self):
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        tokens: list[object] = []
        for t in ["a", "<think>", "b", "</think>", "c"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(_reasoning_delta_text(tokens[1]), "<think>")
        self.assertEqual(_reasoning_delta_text(tokens[3]), "</think>")
        self.assertEqual(tokens[4], "c")

    async def test_auto_tag_channel(self):
        start = "<|channel|>analysis<|message|>"
        end = "<|end|>"
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
            bos_token="<|startoftext|>",
        )
        tokens: list[object] = []
        for t in ["x", start, "y", end, "z"]:
            tokens.extend(await parser.push(t))
        self.assertEqual(_reasoning_delta_text(tokens[1]), start)
        self.assertEqual(_reasoning_delta_text(tokens[3]), end)
        self.assertEqual(tokens[4], "z")
