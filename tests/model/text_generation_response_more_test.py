from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from torch import tensor

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    ReasoningToken,
    ToolCallToken,
)
from avalan.model.response import InvalidJsonResponseException
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    StreamItemKind,
    TextGenerationSingleStream,
    accumulate_canonical_stream_items,
)


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

    async def test_done_callbacks_are_appended_and_run_once(self) -> None:
        async def gen():
            yield "ok"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        calls: list[str] = []

        async def async_callback() -> None:
            calls.append("async")

        def sync_callback() -> None:
            calls.append("sync")

        resp.add_done_callback(async_callback)
        resp.add_done_callback(sync_callback)
        with self.assertRaises(AssertionError):
            resp.add_done_callback(cast(Any, None))

        self.assertEqual(await resp.to_str(), "ok")
        self.assertEqual(calls, ["async", "sync"])

        await resp.to_str()
        self.assertEqual(calls, ["async", "sync"])

    async def test_input_token_count_accepts_tensor_inputs(self) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "ok",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
            inputs=tensor([[1, 2, 3]]),
        )
        self.assertEqual(resp.input_token_count, 3)

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

    async def test_non_stream_canonical_stream_uses_provider_family(
        self,
    ) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "ok",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )

        items = tuple(
            [
                item
                async for item in resp.canonical_stream(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                )
            ]
        )

        self.assertEqual(resp.provider_family, "transformers")
        self.assertEqual(
            {item.provider_family for item in items}, {"transformers"}
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, "ok"
        )

    async def test_canonical_stream_closes_underlying_response_output(
        self,
    ) -> None:
        class PendingOutput:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "PendingOutput":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                return "late"

            async def aclose(self) -> None:
                self.closed = True

        output = PendingOutput()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        stream = resp.canonical_stream(
            stream_session_id="response-stream",
            run_id="response-run",
            turn_id="response-turn",
        )

        item = await stream.__anext__()
        self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
        item = await stream.__anext__()
        self.assertIs(item.kind, StreamItemKind.ANSWER_DELTA)
        await cast(Any, stream).aclose()

        self.assertEqual(output.read_count, 1)
        self.assertTrue(output.closed)

    async def test_canonical_stream_opens_underlying_output_once(self) -> None:
        class Output:
            def __init__(self, value: str) -> None:
                self.value = value
                self.done = False
                self.closed = False

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> str:
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return self.value

            async def aclose(self) -> None:
                self.closed = True

        outputs: list[Output] = []

        def output_fn(**_: object) -> Output:
            output = Output(str(len(outputs) + 1))
            outputs.append(output)
            return output

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        items = tuple(
            [
                item
                async for item in resp.canonical_stream(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                )
            ]
        )

        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].closed)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, "1"
        )

    async def test_aclose_handles_unopened_and_nonclosable_output(
        self,
    ) -> None:
        settings = GenerationSettings()
        unopened = TextGenerationResponse(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        await unopened.aclose()

        class NonClosableOutput:
            def __aiter__(self) -> "NonClosableOutput":
                return self

            async def __anext__(self) -> str:
                return "ok"

        nonclosable = TextGenerationResponse(
            lambda **_: NonClosableOutput(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        nonclosable.__aiter__()
        await nonclosable.aclose()
        await nonclosable.aclose()

    async def test_canonical_stream_preserves_usage_after_close(self) -> None:
        usage = {"input_tokens": 2, "output_tokens": 1}

        class UsageOutput:
            def __init__(self) -> None:
                self._done = False
                self.closed = False
                self.usage: object | None = None

            def __aiter__(self) -> "UsageOutput":
                return self

            async def __anext__(self) -> str:
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                self.usage = usage
                return "ok"

            async def aclose(self) -> None:
                self.closed = True

        output = UsageOutput()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        items = tuple(
            [
                item
                async for item in resp.canonical_stream(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                )
            ]
        )

        self.assertTrue(output.closed)
        self.assertEqual(resp.usage, usage)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, "ok"
        )

    async def test_to_str_uses_answer_channel_accumulation(self) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "</think>"
            yield ToolCallToken(token='{"expression":"2+2"}')
            yield "done"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        result = await resp.to_str()

        self.assertEqual(result, "answer done")
        self.assertEqual(resp.output_token_count, 6)

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

    async def test_reasoning_limit_closes_underlying_output(self) -> None:
        class ReasoningLimitOutput:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "ReasoningLimitOutput":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count == 1:
                    return "<think>"
                return "private"

            async def aclose(self) -> None:
                self.closed = True

        output = ReasoningLimitOutput()
        settings = GenerationSettings(
            reasoning=ReasoningSettings(
                max_new_tokens=1,
                stop_on_max_new_tokens=True,
            )
        )
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        iterator = resp.__aiter__()
        first = await iterator.__anext__()
        self.assertIsInstance(first, ReasoningToken)

        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()

        self.assertEqual(output.read_count, 2)
        self.assertTrue(output.closed)

    async def test_to_str_closes_underlying_output_on_error(self) -> None:
        class FailingOutput:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "FailingOutput":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count == 1:
                    return "before "
                raise RuntimeError("provider failed")

            async def aclose(self) -> None:
                self.closed = True

        output = FailingOutput()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await resp.to_str()

        self.assertEqual(output.read_count, 2)
        self.assertTrue(output.closed)

    async def test_restarted_iteration_resets_response_accumulation(
        self,
    ) -> None:
        async def gen():
            yield "a"
            yield "b"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        iterator = resp.__aiter__()
        self.assertEqual(await iterator.__anext__(), "a")

        tokens = [token async for token in resp]

        self.assertEqual(tokens, ["a", "b"])
        self.assertEqual(await resp.to_str(), "ab")
        self.assertEqual(resp.output_token_count, 2)
