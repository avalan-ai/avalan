from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    ReasoningToken,
    TokenDetail,
    ToolCallToken,
)
from avalan.model.response.text import TextGenerationResponse


async def _gen():
    yield "a"


class TextGenerationResponseFullCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_parser_queue_precedence(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: _gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()
        resp._parser_queue.put("x")
        first = await resp.__anext__()
        second = await resp.__anext__()
        self.assertEqual(first, "x")
        self.assertEqual(second, "a")

    async def test_set_thinking_and_properties(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: _gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        self.assertTrue(resp.can_think)
        self.assertFalse(resp.is_thinking)
        resp.set_thinking(True)
        self.assertTrue(resp.is_thinking)
        resp.set_thinking(False)
        self.assertFalse(resp.is_thinking)

    async def test_disabled_reasoning_parser_returns_raw_token(self):
        async def gen():
            yield "b"

        gs = GenerationSettings(reasoning=ReasoningSettings(enabled=False))
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=gs,
            settings=gs,
        )
        it = resp.__aiter__()
        token = await it.__anext__()
        self.assertEqual(token, "b")
        self.assertFalse(resp.can_think)
        resp.set_thinking(True)  # Should have no effect
        self.assertFalse(resp.is_thinking)

    async def test_flush_and_token_type_parsing(self) -> None:
        async def gen():
            yield ToolCallToken(token="tool", id=1)
            yield TokenDetail(id=2, token="det", probability=0.1)
            yield "<think>"
            yield "foo"
            yield "<"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(
            [type(t) for t in tokens],
            [
                ToolCallToken,
                TokenDetail,
                ReasoningToken,
                ReasoningToken,
                ReasoningToken,
            ],
        )
        self.assertEqual(tokens[0].token, "tool")
        self.assertEqual(tokens[1].token, "det")
        self.assertEqual(tokens[-1].token, "<")
