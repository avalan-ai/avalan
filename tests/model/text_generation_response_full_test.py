from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import StreamValidationError


async def _gen():
    yield "a"


class TextGenerationResponseFullCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_count_input_tokens_fallback_paths(self):
        class NoLen:
            def __len__(self):
                raise TypeError("no len")

        self.assertEqual(TextGenerationResponse._count_input_tokens(None), 0)
        self.assertEqual(
            TextGenerationResponse._count_input_tokens({"input_ids": None}), 0
        )
        self.assertEqual(
            TextGenerationResponse._count_input_tokens(NoLen()),
            0,
        )

    async def test_first_input_sequence_edge_paths(self):
        class BadShape:
            shape = object()

            def __getitem__(self, _):
                raise TypeError("bad index")

        self.assertEqual(
            TextGenerationResponse._first_input_sequence([]),
            [],
        )
        bad_shape = BadShape()
        self.assertIs(
            TextGenerationResponse._first_input_sequence(bad_shape), bad_shape
        )

        class OneDimensional:
            shape = (3,)

        one_dimensional = OneDimensional()
        self.assertIs(
            TextGenerationResponse._first_input_sequence(one_dimensional),
            one_dimensional,
        )

    async def test_parser_queue_precedence(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: _gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await iterator.__anext__()

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

    async def test_aiter_preserves_active_iteration_after_manual_thinking(
        self,
    ) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: _gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()
        resp.set_thinking(True)

        self.assertIs(resp.__aiter__(), iterator)
        self.assertTrue(resp.is_thinking)

    async def test_aiter_rejects_reiteration_after_legacy_failure(
        self,
    ) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "</think>"
            yield "tail"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await iterator.__anext__()

        with self.assertRaisesRegex(RuntimeError, "single-use"):
            resp.__aiter__()
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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await it.__anext__()
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in resp]

    async def test_completed_tool_call_bypasses_reasoning_parser(self) -> None:
        call = ToolCall(
            id="call_1",
            name="math.calculator",
            arguments={"expression": "2 + 2"},
        )

        async def gen():
            yield ToolCallToken(token="", call=call)

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in resp]
