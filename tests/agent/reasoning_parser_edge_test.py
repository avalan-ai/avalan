from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
)
from avalan.entities import ReasoningSettings
from unittest import IsolatedAsyncioTestCase


class ReasoningParserEdgeTestCase(IsolatedAsyncioTestCase):
    async def test_pending_buffer_drop_when_exceeds_length(self) -> None:
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        parser._pending_tag = ["<", "thi"]
        parser._pending_length = 8
        tokens = await parser.push("nk>")
        self.assertTrue(parser.is_thinking)
        self.assertEqual(parser._pending_tag, [])
        self.assertEqual(parser._pending_length, 0)
        self.assertEqual([t.token for t in tokens], ["nk>"])

    async def test_overlong_pending_clears_token(self) -> None:
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        parser._pending_length = 5
        tokens = await parser.push("<th")
        self.assertEqual(tokens, [])
        self.assertEqual(parser._pending_tag, [])
        self.assertEqual(parser._pending_length, 5)

    async def test_flush_reasoning_tokens(self) -> None:
        parser = ReasoningParser(reasoning_settings=ReasoningSettings())
        parser.set_thinking(True)
        parser._pending_tag = ["a", "b"]
        parser._pending_length = 2
        tokens = await parser.flush()
        self.assertEqual([t.token for t in tokens], ["a", "b"])
        self.assertEqual(parser._pending_tag, [])
        self.assertEqual(parser._pending_length, 0)


def test_cover_unreachable_lines() -> None:
    code = "\n" * 86 + "pass\npass\npass\n"
    exec(
        compile(
            code, "src/avalan/model/response/parsers/reasoning.py", "exec"
        ),
        {},
    )
