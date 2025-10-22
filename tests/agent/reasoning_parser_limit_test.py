from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings, ReasoningToken
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
    ReasoningTokenLimitExceeded,
)


class ReasoningParserLimitTestCase(IsolatedAsyncioTestCase):
    async def test_token_limit_switches_to_plain_output(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(max_new_tokens=2),
            logger=getLogger(),
        )
        outputs = []
        for text in ["<think>", "a", "b", "c", "</think>", "d"]:
            outputs.extend(await parser.push(text))
        self.assertIsInstance(outputs[0], ReasoningToken)
        self.assertEqual(outputs[0].token, "<think>")
        self.assertIsInstance(outputs[1], ReasoningToken)
        self.assertEqual(outputs[1].token, "a")
        self.assertEqual(outputs[2], "b")
        self.assertEqual(outputs[3], "c")
        self.assertIsInstance(outputs[4], ReasoningToken)
        self.assertEqual(outputs[4].token, "</think>")
        self.assertEqual(outputs[5], "d")
        self.assertFalse(parser.is_thinking)

    async def test_token_limit_raises_exception(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(
                max_new_tokens=2,
                stop_on_max_new_tokens=True,
            ),
            logger=getLogger(),
        )
        await parser.push("<think>")
        await parser.push("a")
        with self.assertRaises(ReasoningTokenLimitExceeded):
            await parser.push("b")
