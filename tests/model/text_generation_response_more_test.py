from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    ReasoningToken,
)
from avalan.model.response import InvalidJsonResponseException
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import TextGenerationSingleStream


class TextGenerationResponseMoreTestCase(IsolatedAsyncioTestCase):
    async def test_callback_counts_and_to_str(self) -> None:
        async def gen():
            for t in ("a", "b"):
                yield t

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            inputs={"input_ids": [[1, 2]]},
        )
        called = 0

        async def cb():
            nonlocal called
            called += 1

        resp.add_done_callback(cb)
        result = await resp.to_str()
        self.assertEqual(result, "ab")
        self.assertEqual(called, 1)
        self.assertEqual(resp.input_token_count, 2)
        self.assertEqual(resp.output_token_count, 2)
        # calling again should not trigger callback again
        await resp.to_str()
        self.assertEqual(called, 1)

    async def test_single_stream_and_output_count(self) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: TextGenerationSingleStream("ok"),
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )
        result = await resp.to_str()
        self.assertEqual(result, "ok")
        self.assertEqual(resp.output_token_count, 2)

    async def test_to_json_patterns(self) -> None:
        settings = GenerationSettings()
        texts = [
            '```json\n{"a": 1}\n```',
            '```python\n{"a": 1}\n```',
            '{"a": 1}',
        ]
        for text in texts:
            resp = TextGenerationResponse(
                lambda **_: text,
                logger=getLogger(),
                use_async_generator=False,
                generation_settings=settings,
                settings=settings,
            )
            self.assertEqual(await resp.to_json(), '{"a": 1}')

    async def test_to_json_invalid_cases(self) -> None:
        settings = GenerationSettings()
        invalid_block = "```json\n{bad}\n```"
        resp_block = TextGenerationResponse(
            lambda **_: invalid_block,
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )
        with self.assertRaises(InvalidJsonResponseException):
            await resp_block.to_json()

        resp_plain = TextGenerationResponse(
            lambda **_: "no json here",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )
        with self.assertRaises(InvalidJsonResponseException) as ctx:
            await resp_plain.to_json()
        self.assertEqual(str(ctx.exception), "no json here")

    async def test_reasoning_budget_and_limit(self) -> None:
        async def gen_budget():
            for t in ("<think>", "a", "</think>", "b"):
                yield t

        gs = GenerationSettings(reasoning=ReasoningSettings(enabled=True))
        resp = TextGenerationResponse(
            lambda **_: gen_budget(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=gs,
            settings=gs,
        )
        tokens: list = []
        async for t in resp:
            tokens.append(t)
        self.assertEqual(tokens[-1], "b")

        async def gen_limit():
            for t in ("<think>", "a", "b"):
                yield t

        gs_limit = GenerationSettings(
            reasoning=ReasoningSettings(
                max_new_tokens=1, stop_on_max_new_tokens=True
            )
        )
        resp_limit = TextGenerationResponse(
            lambda **_: gen_limit(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=gs_limit,
            settings=gs_limit,
        )
        it = resp_limit.__aiter__()
        first = await it.__anext__()
        self.assertIsInstance(first, ReasoningToken)
        with self.assertRaises(StopAsyncIteration):
            await it.__anext__()
