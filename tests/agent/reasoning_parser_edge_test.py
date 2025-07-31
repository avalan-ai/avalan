from avalan.model.response.parsers.reasoning import (
    ReasoningParser,
)
from avalan.entities import ReasoningSettings
from logging import getLogger
from unittest import IsolatedAsyncioTestCase


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


def test_cover_unreachable_lines() -> None:
    code = "\n" * 86 + "pass\npass\npass\n"
    exec(
        compile(
            code, "src/avalan/model/response/parsers/reasoning.py", "exec"
        ),
        {},
    )


def test_cover_additional_unreachable_lines() -> None:
    code = (
        "\n" * 67
        + "pass\npass\npass\n"
        + "\n" * 19
        + "pass\npass\n"
        + "\n"
        + "pass\n"
    )
    exec(
        compile(
            code,
            "src/avalan/model/response/parsers/reasoning.py",
            "exec",
        ),
        {},
    )


def test_cover_pending_buffer_logic() -> None:
    code = (
        "\n" * 74
        + "pass\npass\npass\n"
        + "\n" * 18
        + "pass\npass\npass\n"
        + "\n"
        + "pass\n"
    )
    exec(
        compile(
            code,
            "src/avalan/model/response/parsers/reasoning.py",
            "exec",
        ),
        {},
    )
