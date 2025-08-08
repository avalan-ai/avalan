from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import ReasoningSettings
from avalan.model.response.parsers.reasoning import ReasoningParser


class FakeStartTag:
    def __init__(self, value: str) -> None:
        self.value = value
        self.eq_calls = 0

    def startswith(self, prefix: str) -> bool:
        return True

    def __len__(self) -> int:  # noqa: D401 - delegating
        return len(self.value)

    def __eq__(self, other: object) -> bool:
        self.eq_calls += 1
        if other is self:
            return True
        if isinstance(other, str):
            return self.eq_calls > 1 and self.value == other
        return False


class ReasoningParserEdgeTestCase(IsolatedAsyncioTestCase):
    async def test_pending_buffer_drop_when_exceeds_length(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        parser._pending_tokens = ["<", "thi"]
        parser._pending_str = "<thi"
        tokens = await parser.push("nk>")
        self.assertTrue(parser.is_thinking)
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")
        self.assertEqual([t.token for t in tokens], ["<", "thi", "nk>"])

    async def test_overlong_pending_clears_token(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        tokens = await parser.push("<th")
        self.assertEqual(tokens, [])
        self.assertEqual(parser._pending_tokens, ["<th"])
        self.assertEqual(parser._pending_str, "<th")

    async def test_flush_reasoning_tokens(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        parser.set_thinking(True)
        parser._pending_tokens = ["a", "b"]
        parser._pending_str = "ab"
        tokens = await parser.flush()
        self.assertEqual([t.token for t in tokens], ["a", "b"])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_pending_tokens_trim_when_candidate_exceeds_tag(
        self,
    ) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
            start_tag=FakeStartTag("<think>"),
        )
        await parser.push("<")
        tokens = await parser.push("think>extra")
        self.assertEqual(tokens, [])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_single_token_longer_than_tag_trimmed(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
            start_tag=FakeStartTag("<think>"),
        )
        tokens = await parser.push("<think>extra")
        self.assertEqual(tokens, [])
        self.assertEqual(parser._pending_tokens, [])
        self.assertEqual(parser._pending_str, "")

    async def test_exact_tag_sets_thinking(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
            start_tag=FakeStartTag("<think>"),
        )
        tokens = await parser.push("<think>")
        self.assertTrue(parser.is_thinking)
        self.assertEqual([t.token for t in tokens], ["<think>"])

    async def test_thinking_budget_exhaustion_property(self) -> None:
        parser = ReasoningParser(
            reasoning_settings=ReasoningSettings(),
            logger=getLogger(),
            max_thinking_turns=1,
        )
        await parser.push("<think>")
        await parser.push("</think>")
        self.assertTrue(parser.is_thinking_budget_exhausted)
        tokens = await parser.push("after")
        self.assertEqual(tokens, ["after"])
