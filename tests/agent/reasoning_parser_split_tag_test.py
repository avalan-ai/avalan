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


def _reasoning_delta_texts(items: list[object]) -> list[str]:
    texts: list[str] = []
    for item in items:
        assert isinstance(item, StreamProviderEvent)
        assert item.kind is StreamItemKind.REASONING_DELTA
        assert item.text_delta is not None
        assert (
            item.reasoning_representation
            is StreamReasoningRepresentation.NATIVE_TEXT
        )
        assert item.segment_instance_ordinal == 0
        assert item.visibility is StreamVisibility.PRIVATE
        texts.append(item.text_delta)
    return texts


class ReasoningParserSplitTagTestCase(IsolatedAsyncioTestCase):
    async def test_split_start_and_end_tags(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for text in ["<", "think", ">", "a", "b", "<", "/think", ">"]:
            outputs.extend(await parser.push(text))
        self.assertEqual(
            _reasoning_delta_texts(outputs),
            ["<", "think", ">", "a", "b", "<", "/think", ">"],
        )
        self.assertFalse(parser.is_thinking)

    async def test_split_tags_embedded_in_chunks(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for text in [
            "lead <thi",
            "nk>",
            " private ",
            "</thi",
            "nk> tail",
        ]:
            outputs.extend(await parser.push(text))

        self.assertEqual(outputs[0], "lead ")
        self.assertEqual(outputs[-1], " tail")
        reasoning = outputs[1:-1]
        self.assertEqual(
            _reasoning_delta_texts(reasoning),
            ["<thi", "nk>", " private ", "</thi", "nk>"],
        )
        self.assertFalse(parser.is_thinking)

    async def test_empty_chunk_preserves_pending_embedded_marker(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for text in ["alpha <thi", "", "nk>hidden</think> omega"]:
            outputs.extend(await parser.push(text))

        self.assertEqual(outputs[0], "alpha ")
        self.assertEqual(outputs[-1], " omega")
        reasoning = outputs[1:-1]
        self.assertEqual(
            _reasoning_delta_texts(reasoning),
            ["<thi", "nk>", "hidden", "</think>"],
        )
        self.assertFalse(parser.is_thinking)

    async def test_whitespace_chunk_breaks_pending_embedded_marker(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for text in ["alpha <thi", " ", "nk> visible"]:
            outputs.extend(await parser.push(text))
        outputs.extend(await parser.flush())

        self.assertEqual("".join(outputs), "alpha <thi nk> visible")
        self.assertFalse(
            any(isinstance(token, StreamProviderEvent) for token in outputs)
        )
        self.assertFalse(parser.is_thinking)

    async def test_malformed_embedded_partial_tag_stays_visible(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for text in ["lead <thi", "s tail"]:
            outputs.extend(await parser.push(text))

        self.assertEqual("".join(outputs), "lead <this tail")
        self.assertFalse(
            any(isinstance(token, StreamProviderEvent) for token in outputs)
        )
        self.assertFalse(parser.is_thinking)

    async def test_unmatched_partial_tag(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        for t in ["<", "unknown", ">"]:
            outputs.extend(await parser.push(t))
        self.assertEqual(outputs, ["<", "unknown", ">"])

    async def test_buffer_limit_with_long_prefix(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
        )
        outputs = []
        sequence = ["x"] * 50 + [
            "<",
            "think",
            ">",
            "a",
            "<",
            "/think",
            ">",
            "y",
        ]
        for text in sequence:
            outputs.extend(await parser.push(text))
        self.assertEqual(outputs[:50], ["x"] * 50)
        reasoning = outputs[50:-1]
        self.assertEqual(
            _reasoning_delta_texts(reasoning),
            ["<", "think", ">", "a", "<", "/think", ">"],
        )
        self.assertEqual(outputs[-1], "y")

    async def test_split_channel_start_and_end_tags(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(tag=ReasoningTag.CHANNEL),
            logger=getLogger(),
        )
        start_parts = ["<|channel|>", "analysis", "<|message|>"]
        message = ["Leo", "Messi", "dances", "past", "defenders"]
        end_parts = ["<|", "end", "|>"]
        outputs: list[object] = []
        for text in start_parts + message + end_parts:
            outputs.extend(await parser.push(text))
        self.assertEqual(
            _reasoning_delta_texts(outputs),
            start_parts + message + end_parts,
        )
        self.assertFalse(parser.is_thinking)
