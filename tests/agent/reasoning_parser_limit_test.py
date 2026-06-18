from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings
from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
    ReasoningTokenLimitExceeded,
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


class ReasoningParserLimitTestCase(IsolatedAsyncioTestCase):
    async def test_token_limit_switches_to_plain_output(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(max_new_tokens=2),
            logger=getLogger(),
        )
        outputs = []
        for text in ["<think>", "a", "b", "c", "</think>", "d"]:
            outputs.extend(await parser.push(text))
        self.assertEqual(_reasoning_delta_text(outputs[0]), "<think>")
        self.assertEqual(_reasoning_delta_text(outputs[1]), "a")
        self.assertEqual(outputs[2], "b")
        self.assertEqual(outputs[3], "c")
        self.assertEqual(_reasoning_delta_text(outputs[4]), "</think>")
        _assert_reasoning_done(outputs[5])
        self.assertEqual(outputs[6], "d")
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
