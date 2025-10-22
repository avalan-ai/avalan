from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings, ReasoningTag, ReasoningToken
from avalan.model.response.parsers.reasoning import ReasoningParser


class ReasoningParserSplitTagTestCase(IsolatedAsyncioTestCase):
    async def test_split_start_and_end_tags(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
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
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        outputs = []
        for t in ["<", "unknown", ">"]:
            outputs.extend(await parser.push(t))
        self.assertEqual(outputs, ["<", "unknown", ">"])

    async def test_buffer_limit_with_long_prefix(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
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
        self.assertTrue(all(isinstance(t, ReasoningToken) for t in reasoning))
        self.assertEqual(
            [t.token for t in reasoning],
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
        self.assertTrue(all(isinstance(t, ReasoningToken) for t in outputs))
        self.assertEqual(
            [t.token for t in outputs], start_parts + message + end_parts
        )
        self.assertFalse(parser.is_thinking)
