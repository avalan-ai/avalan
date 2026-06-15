from asyncio import CancelledError, Event, create_task, wait_for
from collections.abc import AsyncIterator, Iterator
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from torch import tensor

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    ReasoningToken,
    ToolCall,
    ToolCallToken,
)
from avalan.model.provider import ProviderFamily
from avalan.model.response import InvalidJsonResponseException
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    TextGenerationSingleStream,
    accumulate_canonical_stream_items,
    canonical_item_from_consumer_projection,
    project_canonical_stream_item,
)
from avalan.model.vendor import TextGenerationVendorStream

STREAM_RESPONSE_TEST_TIMEOUT = 1.0


class _HostedProviderSource:
    def __init__(
        self,
        tokens: tuple[str, ...] = ("a",),
        *,
        block_on_read: int | None = None,
    ) -> None:
        self._tokens = tokens
        self._index = 0
        self._block_on_read = block_on_read
        self.read_count = 0
        self.cancel_count = 0
        self.close_count = 0
        self.pull_started = Event()
        self.pull_cancelled = False

    def __aiter__(self) -> "_HostedProviderSource":
        return self

    async def __anext__(self) -> str:
        self.read_count += 1
        if self._block_on_read == self.read_count:
            self.pull_started.set()
            try:
                await Event().wait()
            except CancelledError:
                self.pull_cancelled = True
                raise
        if self._index >= len(self._tokens):
            raise StopAsyncIteration
        token = self._tokens[self._index]
        self._index += 1
        return token

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1


class _HostedVendorStream(TextGenerationVendorStream):
    def __init__(self, source: _HostedProviderSource) -> None:
        async def generator() -> AsyncIterator[str]:
            async for token in source:
                yield token

        super().__init__(
            generator(),
            provider_family=ProviderFamily.OPENAI,
            sources=(source,),
        )


class TextGenerationResponseMoreTestCase(IsolatedAsyncioTestCase):
    @staticmethod
    def _hosted_response(
        source: _HostedProviderSource,
    ) -> TextGenerationResponse:
        return TextGenerationResponse(
            _HostedVendorStream(source),
            logger=getLogger(),
            use_async_generator=True,
        )

    @staticmethod
    def _complete_canonical_items() -> tuple[CanonicalStreamItem, ...]:
        return (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="ok",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={"input_tokens": 0, "output_tokens": 1},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

    async def test_hosted_response_preopen_close_closes_provider(
        self,
    ) -> None:
        source = _HostedProviderSource()
        response = self._hosted_response(source)

        await response.aclose()

        self.assertEqual(source.read_count, 0)
        self.assertEqual(source.cancel_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_hosted_response_preopen_cancel_cancels_provider(
        self,
    ) -> None:
        source = _HostedProviderSource()
        response = self._hosted_response(source)

        await response.cancel()
        await response.aclose()

        self.assertEqual(source.read_count, 0)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)

    async def test_hosted_projection_disconnect_closes_no_read_ahead(
        self,
    ) -> None:
        source = _HostedProviderSource(("a", "b"))
        response = self._hosted_response(source)
        projections = response.consumer_projections(
            stream_session_id="hosted-response-stream",
            run_id="hosted-response-run",
            turn_id="hosted-response-turn",
        )

        started = await anext(projections)
        await projections.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(source.read_count, 1)
        self.assertEqual(source.cancel_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_hosted_projection_cancel_closes_pending_read(
        self,
    ) -> None:
        source = _HostedProviderSource(("a",), block_on_read=2)
        response = self._hosted_response(source)
        projections = response.consumer_projections(
            stream_session_id="hosted-response-stream",
            run_id="hosted-response-run",
            turn_id="hosted-response-turn",
        )

        started = await anext(projections)
        answer = await anext(projections)
        pull = create_task(anext(projections))
        try:
            await wait_for(
                source.pull_started.wait(), STREAM_RESPONSE_TEST_TIMEOUT
            )
            pull.cancel()
            cancelled_items = []
            try:
                cancelled_items.append(
                    await wait_for(pull, STREAM_RESPONSE_TEST_TIMEOUT)
                )
            except CancelledError:
                pass
            if cancelled_items and not any(
                item.kind is StreamItemKind.STREAM_CANCELLED
                for item in cancelled_items
            ):
                cancelled_items.append(
                    await wait_for(
                        anext(projections), STREAM_RESPONSE_TEST_TIMEOUT
                    )
                )
        finally:
            await projections.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertIs(answer.kind, StreamItemKind.ANSWER_DELTA)
        if cancelled_items:
            self.assertIn(
                StreamItemKind.STREAM_CANCELLED,
                [item.kind for item in cancelled_items],
            )
            self.assertIs(
                cancelled_items[-1].terminal_outcome,
                StreamTerminalOutcome.CANCELLED,
            )
        self.assertTrue(source.pull_cancelled)
        self.assertEqual(source.read_count, 2)
        self.assertEqual(source.close_count, 1)

    @staticmethod
    def _errored_canonical_items() -> tuple[CanonicalStreamItem, ...]:
        return (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={
                    "error_type": "RuntimeError",
                    "message": "provider failed",
                },
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )

    @staticmethod
    def _semantic_answer_items() -> tuple[CanonicalStreamItem, ...]:
        tool_call_id = "call-1"
        correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
        return (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="private",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                text_delta='{"x":',
                data={"name": "calculator", "arguments": {"x": 1}},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                text_delta="1}",
                data={"name": "calculator", "arguments": {"x": 1}},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                data={"name": "calculator", "arguments": {"x": 1}},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=6,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=7,
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=8,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
                text_delta="stdout",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=9,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
                data={"result": "stdout"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=10,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="final ",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=11,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=12,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=13,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={"input_tokens": 2, "output_tokens": 2},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=14,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

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

    async def test_done_callback_awaits_generic_awaitable(self) -> None:
        async def gen():
            yield "ok"

        class ProbeAwaitable:
            def __init__(self) -> None:
                self.awaited = False

            def __await__(self) -> Any:
                self.awaited = True
                if False:
                    yield None
                return None

        probe = ProbeAwaitable()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        def callback() -> ProbeAwaitable:
            return probe

        resp.add_done_callback(cast(Any, callback))

        self.assertEqual(await resp.to_str(), "ok")
        self.assertTrue(probe.awaited)

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

    async def test_canonical_stream_accepts_empty_async_output(self) -> None:
        async def gen():
            if False:
                yield "unused"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
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

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, ""
        )
        self.assertEqual(resp.output_token_count, 0)

    async def test_canonical_stream_rejects_legacy_after_semantic_output(
        self,
    ) -> None:
        async def gen():
            yield self._complete_canonical_items()[0]
            yield "legacy"

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
            "legacy stream item after canonical stream item",
        ):
            tuple(
                [
                    item
                    async for item in resp.canonical_stream(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                    )
                ]
            )

    async def test_canonical_stream_reports_semantic_after_legacy_output(
        self,
    ) -> None:
        async def gen():
            yield "legacy"
            yield self._complete_canonical_items()[0]

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
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

        error_item = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_ERRORED
        )
        self.assertIs(items[-1].kind, StreamItemKind.STREAM_CLOSED)
        self.assertIs(
            error_item.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertEqual(
            error_item.data,
            {
                "error_type": "StreamValidationError",
                "message": "canonical stream item after legacy stream item",
            },
        )

    async def test_canonical_stream_finalizer_accepts_sync_close_hook(
        self,
    ) -> None:
        class SyncClosableItems:
            def __init__(self) -> None:
                self.items = iter(
                    TextGenerationResponseMoreTestCase._complete_canonical_items()
                )
                self.closed = False

            def __aiter__(self) -> "SyncClosableItems":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration:
                    raise StopAsyncIteration

            def aclose(self) -> None:
                self.closed = True

        source = SyncClosableItems()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )

        items = tuple(
            [
                item
                async for item in resp._record_canonical_stream_final_text(
                    source
                )
            ]
        )

        self.assertTrue(source.closed)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, "ok"
        )

    async def test_canonical_stream_finalizer_accepts_awaitable_close_hook(
        self,
    ) -> None:
        class ProbeAwaitable:
            def __init__(self) -> None:
                self.awaited = False

            def __await__(self) -> Any:
                self.awaited = True
                if False:
                    yield None
                return None

        class AwaitableClosableItems:
            def __init__(self, probe: ProbeAwaitable) -> None:
                self.items = iter(
                    TextGenerationResponseMoreTestCase._complete_canonical_items()
                )
                self.probe = probe

            def __aiter__(self) -> "AwaitableClosableItems":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration:
                    raise StopAsyncIteration

            def aclose(self) -> ProbeAwaitable:
                return self.probe

        probe = ProbeAwaitable()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )

        items = tuple(
            [
                item
                async for item in resp._record_canonical_stream_final_text(
                    AwaitableClosableItems(probe)
                )
            ]
        )

        self.assertTrue(probe.awaited)
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, "ok"
        )

    async def test_canonical_stream_finalizer_rejects_bad_close_result(
        self,
    ) -> None:
        class BadClosableItems:
            def __init__(self) -> None:
                self.items = iter(
                    TextGenerationResponseMoreTestCase._complete_canonical_items()
                )

            def __aiter__(self) -> "BadClosableItems":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration:
                    raise StopAsyncIteration

            def aclose(self) -> object:
                return object()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "unused",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaises(AssertionError):
            tuple(
                [
                    item
                    async for item in resp._record_canonical_stream_final_text(
                        BadClosableItems()
                    )
                ]
            )

    async def test_consumer_projections_stream_lossless_items(self) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "</think>"
            yield ToolCallToken(
                token='{"expression":"2+2"}',
                call=ToolCall(
                    id="call-1",
                    name="calculator",
                    arguments={"expression": "2+2"},
                ),
            )
            yield "done"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )

        self.assertEqual(
            [item.sequence for item in projections],
            list(range(len(projections))),
        )
        self.assertEqual(
            {item.stream_session_id for item in projections}, {"sdk-stream"}
        )
        self.assertEqual({item.run_id for item in projections}, {"sdk-run"})
        self.assertEqual({item.turn_id for item in projections}, {"sdk-turn"})
        self.assertIn(
            StreamItemKind.REASONING_DELTA, [p.kind for p in projections]
        )
        self.assertIn(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            [p.kind for p in projections],
        )
        self.assertEqual(
            "".join(
                item.text_delta or ""
                for item in projections
                if item.channel is StreamChannel.ANSWER
            ),
            "answer done",
        )
        self.assertEqual(await resp.to_str(), "answer done")

    async def test_consumer_projections_parse_split_reasoning_markers(
        self,
    ) -> None:
        async def gen():
            yield "alpha <thi"
            yield "nk>\n  hidden"
            yield "\n"
            yield "</thi"
            yield "nk> omega"
            yield " <thinking> visible"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )

        self.assertEqual(
            "".join(
                item.text_delta or ""
                for item in projections
                if item.channel is StreamChannel.ANSWER
            ),
            "alpha  omega <thinking> visible",
        )
        self.assertEqual(
            "".join(
                item.text_delta or ""
                for item in projections
                if item.channel is StreamChannel.REASONING
            ),
            "<think>\n  hidden\n</think>",
        )
        self.assertLess(
            next(
                item.sequence
                for item in projections
                if item.kind is StreamItemKind.REASONING_DONE
            ),
            next(
                item.sequence
                for item in projections
                if item.text_delta == " omega"
            ),
        )
        response_text = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )
        self.assertEqual(
            await response_text.to_str(),
            "alpha  omega <thinking> visible",
        )

    async def test_consumer_projections_close_underlying_output(
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
        projections = resp.consumer_projections(
            stream_session_id="sdk-stream",
            run_id="sdk-run",
            turn_id="sdk-turn",
        )

        item = await projections.__anext__()
        self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
        item = await projections.__anext__()
        self.assertIs(item.kind, StreamItemKind.ANSWER_DELTA)
        await cast(Any, projections).aclose()

        self.assertEqual(output.read_count, 1)
        self.assertTrue(output.closed)

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

    async def test_aclose_accepts_sync_close_output(self) -> None:
        class SyncClosableOutput:
            def __init__(self) -> None:
                self.closed = False

            def __aiter__(self) -> "SyncClosableOutput":
                return self

            async def __anext__(self) -> str:
                return "ok"

            def aclose(self) -> None:
                self.closed = True

        output = SyncClosableOutput()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()

        await resp.aclose()
        await resp.aclose()

        self.assertTrue(output.closed)

    async def test_aclose_rejects_bad_sync_close_result(self) -> None:
        class BadSyncCloseOutput:
            def __aiter__(self) -> "BadSyncCloseOutput":
                return self

            async def __anext__(self) -> str:
                return "ok"

            def aclose(self) -> object:
                return object()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: BadSyncCloseOutput(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()

        with self.assertRaises(AssertionError):
            await resp.aclose()

    async def test_aclose_rejects_bad_async_close_result(self) -> None:
        class BadAsyncCloseOutput:
            def __aiter__(self) -> "BadAsyncCloseOutput":
                return self

            async def __anext__(self) -> str:
                return "ok"

            async def aclose(self) -> object:
                return object()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: BadAsyncCloseOutput(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()

        with self.assertRaises(AssertionError):
            await resp.aclose()

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

    async def test_to_str_preserves_split_reasoning_answer_whitespace(
        self,
    ) -> None:
        async def gen():
            for character in "lead <think> private </think> tail":
                yield character

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        tokens = [token async for token in resp]

        self.assertEqual(
            "".join(
                token
                for token in tokens
                if not isinstance(token, ReasoningToken)
            ),
            "lead  tail",
        )
        self.assertEqual(
            "".join(
                token.token
                for token in tokens
                if isinstance(token, ReasoningToken)
            ),
            "<think> private </think>",
        )
        self.assertEqual(await resp.to_str(), "lead  tail")

    async def test_to_str_handles_adjacent_reasoning_sections(
        self,
    ) -> None:
        async def gen():
            for character in "x<think>a</think><think>b</think>y":
                yield character

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        tokens = [token async for token in resp]

        self.assertEqual(
            "".join(
                token
                for token in tokens
                if not isinstance(token, ReasoningToken)
            ),
            "xy",
        )
        self.assertEqual(
            "".join(
                token.token
                for token in tokens
                if isinstance(token, ReasoningToken)
            ),
            "<think>a</think><think>b</think>",
        )
        self.assertEqual(await resp.to_str(), "xy")

    async def test_to_str_preserves_empty_chunk_split_reasoning_marker(
        self,
    ) -> None:
        async def gen():
            yield "alpha <thi"
            yield ""
            yield "nk>hidden</think> omega"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "alpha  omega")

    async def test_consumer_projections_preserve_split_reasoning_whitespace(
        self,
    ) -> None:
        async def gen():
            for character in "lead <think> private </think> tail":
                yield character

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )
        accumulator = accumulate_canonical_stream_items(
            [
                canonical_item_from_consumer_projection(projection)
                for projection in projections
            ]
        )

        self.assertEqual(accumulator.answer_text, "lead  tail")
        self.assertEqual(
            accumulator.reasoning_text,
            "<think> private </think>",
        )

    async def test_consumer_projections_handle_adjacent_reasoning_sections(
        self,
    ) -> None:
        async def gen():
            for character in "x<think>a</think><think>b</think>y":
                yield character

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )
        accumulator = accumulate_canonical_stream_items(
            [
                canonical_item_from_consumer_projection(projection)
                for projection in projections
            ]
        )

        self.assertEqual(accumulator.answer_text, "xy")
        self.assertEqual(
            accumulator.reasoning_text,
            "<think>a</think><think>b</think>",
        )
        self.assertEqual(
            len(
                [
                    projection
                    for projection in projections
                    if projection.kind is StreamItemKind.REASONING_DONE
                ]
            ),
            1,
        )

    async def test_stream_accumulation_and_to_str_match_answer_semantics(
        self,
    ) -> None:
        call = ToolCall(
            id="call-1",
            name="calculator",
            arguments={"x": 1},
        )

        def make_response(shape: str) -> TextGenerationResponse:
            async def gen():
                if shape == "legacy":
                    yield "final "
                    yield "<think>"
                    yield "private"
                    yield "</think>"
                    yield ToolCallToken(token='{"x":1}', call=call)
                    yield "answer"
                    return
                for item in self._semantic_answer_items():
                    if shape == "projection":
                        yield project_canonical_stream_item(item)
                    else:
                        yield item

            settings = GenerationSettings()
            return TextGenerationResponse(
                lambda **_: gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=settings,
                settings=settings,
            )

        for shape in ("legacy", "canonical", "projection"):
            with self.subTest(shape=shape):
                stream_response = make_response(shape)
                _ = [item async for item in stream_response]
                accumulator = stream_response._stream_accumulator
                assert accumulator is not None

                to_str_response = make_response(shape)
                text = await to_str_response.to_str()

                self.assertEqual(accumulator.answer_text, "final answer")
                self.assertEqual(text, accumulator.answer_text)
                self.assertNotIn("private", text)
                self.assertNotIn("stdout", text)
                self.assertNotIn('{"x":1}', text)
                self.assertIn("private", accumulator.reasoning_text)
                self.assertEqual(
                    accumulator.tool_execution_outputs.get("call-1"),
                    "stdout" if shape != "legacy" else None,
                )

    async def test_to_str_preserves_stream_terminal_failure_semantics(
        self,
    ) -> None:
        items = self._errored_canonical_items()

        for projected in (False, True):
            with self.subTest(projected=projected):

                async def gen():
                    for item in items:
                        if projected:
                            yield project_canonical_stream_item(item)
                        else:
                            yield item

                settings = GenerationSettings()
                streamed_response = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                _ = [item async for item in streamed_response]
                accumulator = streamed_response._stream_accumulator
                assert accumulator is not None
                self.assertEqual(accumulator.answer_text, "partial")

                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await streamed_response.to_str()

                to_str_response = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await to_str_response.to_str()

    async def test_async_iteration_finalizes_legacy_canonical_accumulator(
        self,
    ) -> None:
        call = ToolCall(
            id="call_1",
            name="math.calculator",
            arguments={"expression": "2+2"},
        )
        usage = {"input_tokens": 1, "output_tokens": 6}

        class Output:
            def __init__(self) -> None:
                self.usage = usage
                self._tokens: Iterator[str | ToolCallToken] = iter(
                    (
                        "answer ",
                        "<think>",
                        "private",
                        "</think>",
                        ToolCallToken(
                            token='{"expression":"2+2"}',
                            call=call,
                        ),
                        "done",
                    )
                )

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> str | ToolCallToken:
                try:
                    return next(self._tokens)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        output = Output()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        tokens = [token async for token in resp]

        accumulator = resp._stream_accumulator
        assert accumulator is not None
        self.assertEqual(tokens[0], "answer ")
        self.assertIsInstance(tokens[1], ReasoningToken)
        self.assertIsInstance(tokens[4], ToolCallToken)
        self.assertEqual(accumulator.answer_text, "answer done")
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIn("private", accumulator.reasoning_text)
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call_1": '{"expression":"2+2"}'},
        )
        item_count = len(accumulator.items)
        resp._finalize_legacy_stream_accumulator()
        self.assertEqual(len(accumulator.items), item_count)
        self.assertEqual(await resp.to_str(), "answer done")
        self.assertEqual(resp.usage, usage)

    async def test_async_iteration_prefers_canonical_usage_over_provider_usage(
        self,
    ) -> None:
        canonical_usage = {"input_tokens": 2, "output_tokens": 1}
        provider_usage = {"input_tokens": 999, "output_tokens": 999}
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="ok",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage=canonical_usage,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        class Output:
            def __init__(self) -> None:
                self.usage = provider_usage
                self._items: Iterator[CanonicalStreamItem] = iter(items)

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: Output(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        projected = [item async for item in resp]

        self.assertEqual(len(projected), len(items))
        self.assertEqual(resp.usage, canonical_usage)
        self.assertEqual(await resp.to_str(), "ok")

    async def test_to_str_accepts_canonical_output_items(self) -> None:
        async def gen():
            for item in self._complete_canonical_items():
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "ok")
        self.assertEqual(resp.output_token_count, 4)

    async def test_to_str_accepts_projected_output_items(self) -> None:
        async def gen():
            for item in self._complete_canonical_items():
                yield project_canonical_stream_item(item)

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "ok")
        self.assertEqual(resp.output_token_count, 4)

    async def test_to_str_raises_for_semantic_error_terminal(self) -> None:
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={
                    "error_type": "RuntimeError",
                    "message": "provider failed",
                },
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )

        for projected in (False, True):
            with self.subTest(projected=projected):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected:
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                consumed = False

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                resp.add_done_callback(mark_consumed)

                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await resp.to_str()

                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 4)
                self.assertFalse(consumed)

    async def test_to_str_raises_for_semantic_cancelled_terminal(self) -> None:
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        )

        for projected in (False, True):
            with self.subTest(projected=projected):

                async def gen():
                    for item in items:
                        if projected:
                            yield project_canonical_stream_item(item)
                        else:
                            yield item

                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )

                with self.assertRaisesRegex(
                    CancelledError, "stream cancelled"
                ):
                    await resp.to_str()

                self.assertEqual(resp.output_token_count, 4)

    async def test_to_str_uses_semantic_terminal_message_fallbacks(
        self,
    ) -> None:
        cases = (
            ({"error_type": "ProviderError"}, "ProviderError", True),
            ("provider stopped", "provider stopped", False),
        )

        for data, message, preconsume_terminal in cases:
            with self.subTest(message=message):

                async def gen():
                    yield CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=1,
                        kind=StreamItemKind.STREAM_ERRORED,
                        channel=StreamChannel.CONTROL,
                        data=data,
                        terminal_outcome=StreamTerminalOutcome.ERRORED,
                    )

                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )

                if preconsume_terminal:
                    iterator = resp.__aiter__()
                    self.assertIs(
                        (await iterator.__anext__()).kind,
                        StreamItemKind.STREAM_STARTED,
                    )
                    self.assertIs(
                        (await iterator.__anext__()).kind,
                        StreamItemKind.STREAM_ERRORED,
                    )

                with self.assertRaisesRegex(RuntimeError, message):
                    await resp.to_str()

    async def test_canonical_stream_terminal_error_keeps_to_str_failure(
        self,
    ) -> None:
        items = self._errored_canonical_items()

        for projected in (False, True):
            with self.subTest(projected=projected):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected:
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                consumed = False

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                resp.add_done_callback(mark_consumed)

                stream_items = tuple(
                    [
                        item
                        async for item in resp.canonical_stream(
                            stream_session_id="response-stream",
                            run_id="response-run",
                            turn_id="response-turn",
                        )
                    ]
                )

                self.assertIs(
                    stream_items[-1].kind, StreamItemKind.STREAM_ERRORED
                )
                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 4)
                self.assertFalse(consumed)
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await resp.to_str()

    async def test_consumer_projections_legacy_error_keeps_to_str_failure(
        self,
    ) -> None:
        class Output:
            def __init__(self) -> None:
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count == 1:
                    return "partial"
                raise RuntimeError("provider failed")

            async def aclose(self) -> None:
                self.closed = True

        consumed = False

        def mark_consumed() -> None:
            nonlocal consumed
            consumed = True

        output = Output()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.add_done_callback(mark_consumed)

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )

        error_projection = next(
            item
            for item in projections
            if item.kind is StreamItemKind.STREAM_ERRORED
        )
        self.assertIs(
            error_projection.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertTrue(output.closed)
        self.assertFalse(consumed)
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await resp.to_str()

    async def test_async_iteration_terminal_cancel_keeps_to_str_failure(
        self,
    ) -> None:
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
        )

        for projected in (False, True):
            with self.subTest(projected=projected):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected:
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                consumed = False

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                resp.add_done_callback(mark_consumed)
                iterator = resp.__aiter__()

                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_STARTED,
                )
                self.assertEqual(
                    (await iterator.__anext__()).text_delta, "partial"
                )
                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.ANSWER_DONE,
                )
                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_CANCELLED,
                )
                with self.assertRaises(StopAsyncIteration):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)
                self.assertFalse(consumed)
                with self.assertRaisesRegex(
                    CancelledError, "stream cancelled"
                ):
                    await resp.to_str()

    async def test_to_str_after_partial_canonical_iteration_keeps_answer(
        self,
    ) -> None:
        async def gen():
            for item in self._complete_canonical_items():
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        self.assertIs(
            (await iterator.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        self.assertEqual((await iterator.__anext__()).text_delta, "ok")

        self.assertEqual(await resp.to_str(), "ok")

    async def test_to_str_after_partial_projection_iteration_keeps_answer(
        self,
    ) -> None:
        async def gen():
            for item in self._complete_canonical_items():
                yield project_canonical_stream_item(item)

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        self.assertIs(
            (await iterator.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        self.assertEqual((await iterator.__anext__()).text_delta, "ok")

        self.assertEqual(await resp.to_str(), "ok")

    async def test_to_str_rejects_late_semantic_output(self) -> None:
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            )

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
            "semantic stream item emitted after terminal outcome",
        ):
            await resp.to_str()

    async def test_to_str_rejects_late_semantic_output_after_partial_iteration(
        self,
    ) -> None:
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            )

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        self.assertIs(
            (await iterator.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )
        self.assertIs(
            (await iterator.__anext__()).kind,
            StreamItemKind.STREAM_COMPLETED,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await resp.to_str()

    async def test_async_iteration_closes_on_late_semantic_output(
        self,
    ) -> None:
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            ),
        )

        for projected in (False, True):
            with self.subTest(projected=projected):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        return (
                            project_canonical_stream_item(item)
                            if projected
                            else item
                        )

                    async def aclose(self) -> None:
                        self.close_count += 1

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                iterator = resp.__aiter__()

                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_STARTED,
                )
                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_COMPLETED,
                )
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "semantic stream item emitted after terminal outcome",
                ):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 2)

    async def test_async_iteration_rejects_semantic_sequence_discontinuity(
        self,
    ) -> None:
        cases = (
            (
                (
                    CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="gap",
                    ),
                ),
                "lossless consumer stream sequence gap",
            ),
            (
                (
                    CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=2,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="response-stream",
                        run_id="response-run",
                        turn_id="response-turn",
                        sequence=1,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="out of order",
                    ),
                ),
                "stream sequence must increase",
            ),
        )

        for projected, (items, message) in (
            (projected, case) for projected in (False, True) for case in cases
        ):
            with self.subTest(projected=projected, message=message):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected:
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                iterator = resp.__aiter__()

                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_STARTED,
                )
                with self.assertRaisesRegex(StreamValidationError, message):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 1)

    async def test_async_iteration_rejects_mixed_semantic_and_legacy_output(
        self,
    ) -> None:
        first_semantic_item = CanonicalStreamItem(
            stream_session_id="response-stream",
            run_id="response-run",
            turn_id="response-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )

        cases = (
            (
                (first_semantic_item, "legacy"),
                "legacy stream item after canonical stream item",
            ),
            (
                ("legacy", first_semantic_item),
                "canonical stream item after legacy stream item",
            ),
        )

        for projected, (items, message) in (
            (projected, case) for projected in (False, True) for case in cases
        ):
            with self.subTest(projected=projected, message=message):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected and isinstance(item, CanonicalStreamItem):
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                iterator = resp.__aiter__()

                await iterator.__anext__()
                with self.assertRaisesRegex(StreamValidationError, message):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 1)

    async def test_async_iteration_rejects_semantic_output_missing_terminal(
        self,
    ) -> None:
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
        )

        for projected in (False, True):
            with self.subTest(projected=projected):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> object:
                        try:
                            item = next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc
                        if projected:
                            return project_canonical_stream_item(item)
                        return item

                    async def aclose(self) -> None:
                        self.close_count += 1

                consumed = False

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                output = Output()
                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    lambda **_: output,
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                resp.add_done_callback(mark_consumed)
                iterator = resp.__aiter__()

                self.assertIs(
                    (await iterator.__anext__()).kind,
                    StreamItemKind.STREAM_STARTED,
                )
                self.assertEqual(
                    (await iterator.__anext__()).text_delta, "partial"
                )
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "stream missing terminal outcome",
                ):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)
                self.assertEqual(resp.output_token_count, 2)
                self.assertFalse(consumed)

    async def test_to_str_rejects_semantic_after_legacy_output(self) -> None:
        async def gen():
            yield "legacy"
            yield self._complete_canonical_items()[0]

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
            "canonical stream item after legacy stream item",
        ):
            await resp.to_str()

    async def test_to_str_rejects_legacy_after_semantic_output(self) -> None:
        async def gen():
            yield self._complete_canonical_items()[0]
            yield "legacy"

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
            "legacy stream item after canonical stream item",
        ):
            await resp.to_str()

    async def test_consumer_projections_accept_canonical_output_items(
        self,
    ) -> None:
        async def gen():
            for item in self._complete_canonical_items():
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        projections = tuple(
            [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]
        )

        self.assertEqual([item.sequence for item in projections], [0, 1, 2, 3])
        self.assertEqual(
            {item.stream_session_id for item in projections},
            {"response-stream"},
        )
        self.assertEqual(projections[1].text_delta, "ok")
        self.assertIs(
            projections[-1].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_to_str_after_partial_reasoning_iteration_uses_answer_only(
        self,
    ) -> None:
        async def gen():
            yield "answer "
            yield "<think>"
            yield "private"
            yield "</think>"
            yield "done"

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        iterator = resp.__aiter__()

        self.assertEqual(await iterator.__anext__(), "answer ")
        self.assertIsInstance(await iterator.__anext__(), ReasoningToken)
        self.assertIsInstance(await iterator.__anext__(), ReasoningToken)

        self.assertEqual(await resp.to_str(), "answer done")

    async def test_to_str_after_partial_tool_iteration_uses_answer_only(
        self,
    ) -> None:
        async def gen():
            yield "answer "
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
        iterator = resp.__aiter__()

        self.assertEqual(await iterator.__anext__(), "answer ")
        self.assertIsInstance(await iterator.__anext__(), ToolCallToken)

        self.assertEqual(await resp.to_str(), "answer done")

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

    async def test_restarted_iteration_clears_cached_final_text(self) -> None:
        values = iter(("first", "second"))

        def output_fn(**_: object):
            async def gen():
                yield next(values)

            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "first")
        iterator = resp.__aiter__()
        self.assertEqual(await iterator.__anext__(), "second")

        self.assertEqual(await resp.to_str(), "second")
