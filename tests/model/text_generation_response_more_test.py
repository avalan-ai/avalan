from asyncio import CancelledError, Event, create_task, wait_for
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from torch import tensor

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    Token,
    TokenDetail,
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
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
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


class _HostedCanonicalSource:
    def __init__(
        self,
        items: tuple[CanonicalStreamItem, ...],
        *,
        block_on_read: int | None = None,
    ) -> None:
        self._items = items
        self._index = 0
        self._block_on_read = block_on_read
        self.read_count = 0
        self.close_count = 0
        self.pull_started = Event()
        self.pull_cancelled = False

    def __aiter__(self) -> "_HostedCanonicalSource":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        self.read_count += 1
        if self._block_on_read == self.read_count:
            self.pull_started.set()
            try:
                await Event().wait()
            except CancelledError:
                self.pull_cancelled = True
                raise
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    async def aclose(self) -> None:
        self.close_count += 1


class _ExplicitCanonicalOutput:
    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = iter(items)
        self.read_count = 0
        self.close_count = 0

    def __aiter__(self) -> "_ExplicitCanonicalOutput":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        try:
            item = next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        self.read_count += 1
        return item

    async def aclose(self) -> None:
        self.close_count += 1


class _ExplicitCanonicalStreamSource:
    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items
        self.open_count = 0
        self.call_count = 0
        self.outputs: list[_ExplicitCanonicalOutput] = []

    def __call__(
        self, *args: object, **kwargs: object
    ) -> AsyncIterator[CanonicalStreamItem]:
        self.call_count += 1
        return self.canonical_stream(
            stream_session_id="called-stream",
            run_id="called-run",
            turn_id="called-turn",
        )

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        _ = stream_session_id
        _ = run_id
        _ = turn_id
        _ = provider_family
        _ = capabilities
        _ = close_after_terminal
        self.open_count += 1
        output = _ExplicitCanonicalOutput(self._items)
        self.outputs.append(output)
        return output


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
    def _hosted_canonical_response(
        source: _HostedCanonicalSource,
    ) -> TextGenerationResponse:
        return TextGenerationResponse(
            lambda **_: source,
            logger=getLogger(),
            use_async_generator=True,
            provider_family=ProviderFamily.OPENAI,
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

    async def test_hosted_projection_disconnect_closes_after_first_item(
        self,
    ) -> None:
        source = _HostedCanonicalSource(self._complete_canonical_items())
        response = self._hosted_canonical_response(source)
        callbacks = 0

        def mark_consumed() -> None:
            nonlocal callbacks
            callbacks += 1

        response.add_done_callback(mark_consumed)
        projections = response.consumer_projections(
            stream_session_id="hosted-response-stream",
            run_id="hosted-response-run",
            turn_id="hosted-response-turn",
        )

        started = await anext(projections)
        self.assertEqual(callbacks, 0)
        await projections.aclose()
        await response.aclose()

        self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(source.read_count, 1)
        self.assertEqual(source.close_count, 1)
        self.assertEqual(callbacks, 1)

    async def test_hosted_projection_cancel_closes_pending_read(
        self,
    ) -> None:
        source = _HostedCanonicalSource(
            self._complete_canonical_items(), block_on_read=3
        )
        response = self._hosted_canonical_response(source)
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
        self.assertEqual(source.read_count, 3)
        self.assertEqual(source.close_count, 1)

    async def test_base_vendor_stream_response_uses_requested_identity(
        self,
    ) -> None:
        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._complete_canonical_items():
                yield item

        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.HOSTED,
            provider_family=ProviderFamily.OPENAI,
            supports_usage=True,
        )
        response = TextGenerationResponse(
            TextGenerationVendorStream(
                generator(), provider_family=ProviderFamily.OPENAI
            ),
            logger=getLogger(),
            use_async_generator=True,
        )

        items = [
            item
            async for item in response.canonical_stream(
                stream_session_id="requested-stream",
                run_id="requested-run",
                turn_id="requested-turn",
                provider_family=ProviderFamily.OPENAI,
                capabilities=capabilities,
            )
        ]

        self.assertEqual(
            {
                (item.stream_session_id, item.run_id, item.turn_id)
                for item in items
            },
            {("requested-stream", "requested-run", "requested-turn")},
        )
        self.assertEqual(
            items[0].metadata["capabilities"],
            capabilities.to_metadata(),
        )
        self.assertEqual(
            {item.provider_family for item in items},
            {ProviderFamily.OPENAI.value},
        )

    async def test_base_vendor_stream_derives_family_from_capabilities(
        self,
    ) -> None:
        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._complete_canonical_items():
                yield item

        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.HOSTED,
            provider_family=ProviderFamily.OPENAI,
        )
        response = TextGenerationResponse(
            TextGenerationVendorStream(generator()),
            logger=getLogger(),
            use_async_generator=True,
        )

        items = [
            item
            async for item in response.canonical_stream(
                stream_session_id="requested-stream",
                run_id="requested-run",
                turn_id="requested-turn",
                capabilities=capabilities,
            )
        ]

        self.assertEqual(
            {item.provider_family for item in items},
            {ProviderFamily.OPENAI.value},
        )

    async def test_inherited_base_canonical_stream_uses_response_projection(
        self,
    ) -> None:
        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._complete_canonical_items():
                yield item

        class BaseOnlyStream(TextGenerationStream):
            def __init__(self) -> None:
                self._generator = generator()

            def __call__(
                self, *args: object, **kwargs: object
            ) -> AsyncIterator[CanonicalStreamItem]:
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                assert self._generator is not None
                return await self._generator.__anext__()

        response = TextGenerationResponse(
            BaseOnlyStream(),
            logger=getLogger(),
            use_async_generator=True,
        )

        items = [
            item
            async for item in response.canonical_stream(
                stream_session_id="requested-stream",
                run_id="requested-run",
                turn_id="requested-turn",
            )
        ]

        self.assertEqual(
            {
                (item.stream_session_id, item.run_id, item.turn_id)
                for item in items
            },
            {("response-stream", "response-run", "response-turn")},
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text,
            "ok",
        )

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
    def _cancelled_canonical_items() -> tuple[CanonicalStreamItem, ...]:
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
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
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
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
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
            for item in (
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
                    text_delta="a",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="b",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

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
        self.assertEqual(resp.output_token_count, 5)
        # calling again should not trigger callback again
        await resp.to_str()
        self.assertEqual(called, 1)

    async def test_done_callbacks_are_appended_and_run_once(self) -> None:
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
            for item in self._complete_canonical_items():
                yield item

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
        usage = {"input_tokens": 1, "output_tokens": 2}
        resp = TextGenerationResponse(
            lambda **_: TextGenerationSingleStream(
                "ok",
                provider_family="openai",
                usage=usage,
            ),
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
        self.assertEqual(
            accumulate_canonical_stream_items(items).final_usage, usage
        )
        self.assertEqual(resp.usage, usage)
        self.assertEqual(resp.output_token_count, 2)

    async def test_non_stream_canonical_stream_then_to_str_drains_session(
        self,
    ) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: "hello",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )
        stream = resp.canonical_stream(
            stream_session_id="response-stream",
            run_id="response-run",
            turn_id="response-turn",
        )

        try:
            self.assertIs(
                (await anext(stream)).kind,
                StreamItemKind.STREAM_STARTED,
            )
            self.assertEqual((await anext(stream)).text_delta, "hello")
            accumulator = resp._stream_accumulator
            assert accumulator is not None
            self.assertEqual(accumulator.answer_text, "hello")

            self.assertEqual(await resp.to_str(), "hello")
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertIs(
            accumulator.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertTrue(resp._output_closed)
        self.assertEqual(resp.output_token_count, 5)

    async def test_canonical_stream_legacy_rejection_closes_output(
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await stream.__anext__()

        self.assertEqual(output.read_count, 1)
        self.assertTrue(output.closed)

    async def test_canonical_stream_opens_underlying_output_once(self) -> None:
        class Output:
            def __init__(self, value: str) -> None:
                self.items = iter(
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
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta=value,
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
                            usage={"output_tokens": 1},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    )
                )
                self.closed = False

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

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

    async def test_canonical_stream_then_to_str_uses_same_accumulator(
        self,
    ) -> None:
        source = _ExplicitCanonicalStreamSource(
            self._complete_canonical_items()
        )
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            source,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        stream = resp.canonical_stream(
            stream_session_id="requested-stream",
            run_id="requested-run",
            turn_id="requested-turn",
        )

        try:
            self.assertIs(
                (await anext(stream)).kind,
                StreamItemKind.STREAM_STARTED,
            )
            self.assertEqual((await anext(stream)).text_delta, "ok")
            accumulator = resp._stream_accumulator
            assert accumulator is not None
            self.assertEqual(accumulator.answer_text, "ok")

            text = await resp.to_str()
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()

        self.assertEqual(text, "ok")
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(text, accumulator.answer_text)
        self.assertEqual(source.open_count, 1)
        self.assertEqual(source.call_count, 0)
        self.assertEqual(source.outputs[0].read_count, 4)
        self.assertEqual(source.outputs[0].close_count, 1)

    async def test_explicit_canonical_stream_rejects_empty_output(
        self,
    ) -> None:
        source = _ExplicitCanonicalStreamSource(())
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            source,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "stream must contain at least one item",
        ):
            _ = [
                item
                async for item in resp.canonical_stream(
                    stream_session_id="requested-stream",
                    run_id="requested-run",
                    turn_id="requested-turn",
                )
            ]

        self.assertEqual(source.open_count, 1)
        self.assertEqual(source.outputs[0].close_count, 1)

    async def test_overlapping_explicit_canonical_streams_are_rejected(
        self,
    ) -> None:
        source = _ExplicitCanonicalStreamSource(
            self._complete_canonical_items()
        )
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            source,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        first_stream = resp.canonical_stream(
            stream_session_id="first-stream",
            run_id="first-run",
            turn_id="first-turn",
        )

        try:
            self.assertIs(
                (await anext(first_stream)).kind,
                StreamItemKind.STREAM_STARTED,
            )
            first_output = source.outputs[0]

            with self.assertRaisesRegex(RuntimeError, "single-use"):
                resp.canonical_stream(
                    stream_session_id="second-stream",
                    run_id="second-run",
                    turn_id="second-turn",
                )
        finally:
            aclose = getattr(first_stream, "aclose", None)
            if aclose is not None:
                await aclose()

        self.assertEqual(source.open_count, 1)
        self.assertEqual(first_output.close_count, 1)

    async def test_unstarted_explicit_canonical_stream_is_response_owned(
        self,
    ) -> None:
        source = _ExplicitCanonicalStreamSource(
            self._complete_canonical_items()
        )
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            source,
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

        self.assertEqual(source.open_count, 1)
        await resp.aclose()
        self.assertEqual(source.outputs[0].close_count, 1)

        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
        self.assertEqual(source.outputs[0].close_count, 1)

        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertEqual(source.outputs[0].read_count, 0)

    async def test_unstarted_explicit_canonical_stream_close_owns_source(
        self,
    ) -> None:
        source = _ExplicitCanonicalStreamSource(
            self._complete_canonical_items()
        )
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            source,
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
        aclose = getattr(stream, "aclose", None)
        assert aclose is not None
        await aclose()

        self.assertEqual(source.open_count, 1)
        self.assertEqual(source.outputs[0].close_count, 1)

    async def test_unstarted_lazy_canonical_stream_stops_after_close(
        self,
    ) -> None:
        open_count = 0

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._complete_canonical_items():
                yield item

        def output_fn(**_: object) -> AsyncIterator[CanonicalStreamItem]:
            nonlocal open_count
            open_count += 1
            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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
        await resp.aclose()

        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertEqual(open_count, 0)

    async def test_canonical_stream_accepts_empty_async_output(self) -> None:
        usage = {"input_tokens": 1, "output_tokens": 0}

        class EmptyOutput:
            def __init__(self) -> None:
                self.close_count = 0
                self.usage: object | None = None

            def __aiter__(self) -> "EmptyOutput":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.close_count += 1
                self.usage = usage

        settings = GenerationSettings()
        output = EmptyOutput()
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

        self.assertEqual(
            accumulate_canonical_stream_items(items).answer_text, ""
        )
        self.assertEqual(
            accumulate_canonical_stream_items(items).final_usage, usage
        )
        self.assertEqual(output.close_count, 1)
        self.assertEqual(resp.usage, usage)
        self.assertEqual(resp.output_token_count, 0)

    async def test_to_str_accepts_empty_async_output(self) -> None:
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            if False:
                yield self._complete_canonical_items()[0]

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "")
        self.assertEqual(await resp.to_str(), "")
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
            "unsupported legacy SDK response stream item",
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

    async def test_canonical_stream_legacy_rejection_before_semantic_output(
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [
                item
                async for item in resp.canonical_stream(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                )
            ]

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
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._semantic_answer_items():
                yield item

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
            {item.stream_session_id for item in projections},
            {"response-stream"},
        )
        self.assertEqual(
            {item.run_id for item in projections}, {"response-run"}
        )
        self.assertEqual(
            {item.turn_id for item in projections}, {"response-turn"}
        )
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
            "final answer",
        )
        self.assertEqual(await resp.to_str(), "final answer")

    async def test_consumer_projections_then_to_str_keeps_side_channels(
        self,
    ) -> None:
        base_items = self._semantic_answer_items()
        diagnostic = replace(
            base_items[0],
            sequence=1,
            kind=StreamItemKind.STREAM_DIAGNOSTIC,
            text_delta="diagnostic text",
        )
        items = (
            base_items[0],
            diagnostic,
            *(
                replace(item, sequence=item.sequence + 1)
                for item in base_items[1:]
            ),
        )
        usage = {"input_tokens": 2, "output_tokens": 2}
        open_count = 0

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        def output_fn(**_: object) -> AsyncIterator[CanonicalStreamItem]:
            nonlocal open_count
            open_count += 1
            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        text = await resp.to_str()

        self.assertEqual(text, "final answer")
        self.assertEqual(text, accumulator.answer_text)
        self.assertNotIn("private", text)
        self.assertNotIn("stdout", text)
        self.assertNotIn('{"x":1}', text)
        self.assertNotIn("diagnostic", text)
        self.assertIn(
            StreamItemKind.STREAM_DIAGNOSTIC,
            [projection.kind for projection in projections],
        )
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(accumulator.reasoning_text, "private")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"call-1": '{"x":1}'},
        )
        self.assertEqual(
            accumulator.tool_execution_outputs,
            {"call-1": "stdout"},
        )
        self.assertEqual(
            [item.text_delta for item in accumulator.diagnostics],
            ["diagnostic text"],
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertEqual(resp.usage, usage)
        self.assertEqual(open_count, 1)

    async def test_consumer_projections_rejects_local_legacy_reasoning_stream(
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]

        response_text = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
            provider_family="transformers",
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await response_text.to_str()

    async def test_consumer_projections_legacy_rejection_closes_output(
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await projections.__anext__()

        self.assertEqual(output.read_count, 1)
        self.assertTrue(output.closed)

    async def test_default_sdk_legacy_rejection_yields_no_items(
        self,
    ) -> None:
        class LegacyOutput:
            def __init__(self, item: str | Token | TokenDetail) -> None:
                self._item = item
                self.read_count = 0
                self.closed = False

            def __aiter__(self) -> "LegacyOutput":
                return self

            async def __anext__(self) -> str | Token | TokenDetail:
                self.read_count += 1
                if self.read_count > 1:
                    raise AssertionError("default SDK stream read ahead")
                return self._item

            async def aclose(self) -> None:
                self.closed = True

        async def assert_rejection(
            surface: str,
            provider_family: str,
            legacy_item: str | Token | TokenDetail,
        ) -> None:
            output = LegacyOutput(legacy_item)
            settings = GenerationSettings()
            resp = TextGenerationResponse(
                lambda **_: output,
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=settings,
                settings=settings,
                provider_family=provider_family,
            )
            items: list[object] = []

            with self.assertRaisesRegex(
                StreamValidationError,
                "unsupported legacy SDK response stream item",
            ):
                if surface == "canonical_stream":
                    async for item in resp.canonical_stream(
                        stream_session_id="sdk-stream",
                        run_id="sdk-run",
                        turn_id="sdk-turn",
                    ):
                        items.append(item)
                else:
                    async for item in resp.consumer_projections(
                        stream_session_id="sdk-stream",
                        run_id="sdk-run",
                        turn_id="sdk-turn",
                    ):
                        items.append(item)

            self.assertEqual(items, [])
            self.assertEqual(output.read_count, 1)
            self.assertTrue(output.closed)

        legacy_items: tuple[str | Token | TokenDetail, ...] = (
            "legacy",
            Token(token="legacy-token"),
            TokenDetail(
                id=7,
                token="legacy-detail",
                probability=0.5,
                step=1,
            ),
        )
        for surface in ("canonical_stream", "consumer_projections"):
            for provider_family in ("local", "openai"):
                for legacy_item in legacy_items:
                    with self.subTest(
                        surface=surface,
                        provider_family=provider_family,
                        item_type=legacy_item.__class__.__name__,
                    ):
                        await assert_rejection(
                            surface, provider_family, legacy_item
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

    async def test_aclose_retries_after_failed_close_result(self) -> None:
        class RetryCloseOutput:
            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> "RetryCloseOutput":
                return self

            async def __anext__(self) -> str:
                return "ok"

            def aclose(self) -> object | None:
                self.close_count += 1
                if self.close_count == 1:
                    return object()
                return None

        output = RetryCloseOutput()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()

        with self.assertRaises(AssertionError):
            await resp.aclose()
        await resp.aclose()
        await resp.aclose()

        self.assertEqual(output.close_count, 2)

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
                self.items = iter(
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
                            sequence=1,
                            kind=StreamItemKind.USAGE_COMPLETED,
                            channel=StreamChannel.USAGE,
                            usage=usage,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="response-stream",
                            run_id="response-run",
                            turn_id="response-turn",
                            sequence=2,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    )
                )
                self.closed = False
                self.usage: object | None = None

            def __aiter__(self) -> "UsageOutput":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    item = next(self.items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc
                self.usage = usage
                return item

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
            accumulate_canonical_stream_items(items).final_usage, usage
        )

    async def test_to_str_uses_answer_channel_accumulation(self) -> None:
        async def gen():
            for item in self._semantic_answer_items():
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        result = await resp.to_str()

        self.assertEqual(result, "final answer")
        self.assertEqual(resp.output_token_count, 15)

    async def test_to_str_keeps_non_answer_channels_out_of_final_text(
        self,
    ) -> None:
        tool_call_id = "call-1"
        correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
        usage = {"input_tokens": 3, "output_tokens": 2}
        items = (
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                data={"message": "control start"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=1,
                kind=StreamItemKind.STREAM_DIAGNOSTIC,
                channel=StreamChannel.CONTROL,
                text_delta="diagnostic text",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="private reasoning",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                text_delta='{"q":',
                data={"name": "search"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                text_delta='"x"}',
                data={"name": "search"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=6,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                data={"name": "search", "arguments": {"q": "x"}},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=7,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=8,
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=9,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
                text_delta="tool output",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=10,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=correlation,
                data={"result": "tool output"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=11,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="final ",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=12,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=13,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=14,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage=usage,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=15,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                data={"message": "control completed"},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        result = await resp.to_str()
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        self.assertEqual(result, accumulator.answer_text)
        self.assertEqual(result, "final answer")
        self.assertNotIn("diagnostic text", result)
        self.assertNotIn("private reasoning", result)
        self.assertNotIn('{"q":"x"}', result)
        self.assertNotIn("tool output", result)
        self.assertNotIn("control", result)
        self.assertEqual(accumulator.reasoning_text, "private reasoning")
        self.assertEqual(
            accumulator.tool_call_arguments[tool_call_id],
            '{"q":"x"}',
        )
        self.assertEqual(
            accumulator.tool_execution_outputs[tool_call_id],
            "tool output",
        )
        self.assertEqual(
            [item.text_delta for item in accumulator.diagnostics],
            ["diagnostic text"],
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertEqual(resp.usage, usage)
        self.assertEqual(
            [item.kind for item in accumulator.control_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )

    async def test_to_str_preserves_split_reasoning_answer_whitespace(
        self,
    ) -> None:
        async def gen():
            for item in (
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
                    text_delta="lead ",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="<think> private </think>",
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=(
                        StreamReasoningRepresentation.NATIVE_TEXT
                    ),
                    segment_instance_ordinal=0,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta=" tail",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=6,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "lead  tail")
        accumulator = resp._stream_accumulator
        assert accumulator is not None
        self.assertEqual(
            accumulator.reasoning_text, "<think> private </think>"
        )

    async def test_to_str_handles_adjacent_reasoning_sections(
        self,
    ) -> None:
        async def gen():
            for item in (
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
                    text_delta="x",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=2,
                    kind=StreamItemKind.REASONING_DELTA,
                    channel=StreamChannel.REASONING,
                    text_delta="<think>a</think><think>b</think>",
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=(
                        StreamReasoningRepresentation.NATIVE_TEXT
                    ),
                    segment_instance_ordinal=0,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=3,
                    kind=StreamItemKind.REASONING_DONE,
                    channel=StreamChannel.REASONING,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=4,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="y",
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=5,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="response-stream",
                    run_id="response-run",
                    turn_id="response-turn",
                    sequence=6,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertEqual(await resp.to_str(), "xy")
        accumulator = resp._stream_accumulator
        assert accumulator is not None
        self.assertEqual(
            accumulator.reasoning_text,
            "<think>a</think><think>b</think>",
        )

    async def test_to_str_preserves_empty_chunk_split_reasoning_marker(
        self,
    ) -> None:
        async def gen():
            yield "alpha <thi"

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
            await resp.to_str()

    async def test_consumer_projections_preserve_split_reasoning_whitespace(
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
                text_delta="lead ",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="<think>",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta=" private ",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=4,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="</think>",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=5,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=6,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=" tail",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=7,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=8,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

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
                text_delta="x",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="<think>a</think>",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=3,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="<think>b</think>",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=1,
                metadata={"reasoning.segment_boundary": "completed"},
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=4,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=5,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="y",
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=6,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=7,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

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
        projected_items = [
            canonical_item_from_consumer_projection(projection)
            for projection in projections
        ]
        accumulator = accumulate_canonical_stream_items(projected_items)

        self.assertEqual(accumulator.answer_text, "xy")
        self.assertEqual(
            accumulator.reasoning_text,
            "<think>a</think>\n\n<think>b</think>",
        )
        self.assertEqual(
            [
                item.text_delta
                for item in projected_items
                if item.kind is StreamItemKind.REASONING_DELTA
            ],
            ["<think>a</think>", "<think>b</think>"],
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

        legacy_response = make_response("legacy")
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in legacy_response]

        legacy_to_str_response = make_response("legacy")
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await legacy_to_str_response.to_str()

        for shape in ("canonical", "projection"):
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
                    "stdout",
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
                self.assertEqual(
                    await streamed_response.to_str(
                        raise_terminal_exception=False
                    ),
                    "partial",
                )

                to_str_response = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await to_str_response.to_str()

                partial_response = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )
                self.assertEqual(
                    await partial_response.to_str(
                        raise_terminal_exception=False
                    ),
                    "partial",
                )
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await partial_response.to_str()

    async def test_to_str_preserves_input_required_as_non_success(
        self,
    ) -> None:
        correlation = StreamItemCorrelation(
            request_id="request-1",
            continuation_id="continuation-1",
            agent_id="agent-1",
            branch_id="branch-1",
        )
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
                kind=StreamItemKind.INTERACTION_PENDING,
                channel=StreamChannel.INTERACTION,
                correlation=correlation,
            ),
            CanonicalStreamItem(
                stream_session_id="response-stream",
                run_id="response-run",
                turn_id="response-turn",
                sequence=2,
                kind=StreamItemKind.STREAM_INPUT_REQUIRED,
                channel=StreamChannel.CONTROL,
                correlation=correlation,
                terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
            ),
        )

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in items:
                yield item

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        with self.assertRaisesRegex(RuntimeError, "stream input_required"):
            await response.to_str()
        self.assertEqual(
            await response.to_str(raise_terminal_exception=False),
            "",
        )
        self.assertIs(
            response._terminal_failure_outcome,
            StreamTerminalOutcome.INPUT_REQUIRED,
        )

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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in resp]
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

        outputs: list[Output] = []

        def output_fn(**_: object) -> Output:
            output = Output()
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

        projected = [item async for item in resp]
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        self.assertEqual(len(projected), len(items))
        self.assertEqual(resp.usage, canonical_usage)
        self.assertEqual(await resp.to_str(), "ok")
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(accumulator.answer_text, "ok")
        self.assertEqual(len(outputs), 1)

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
                self.assertTrue(consumed)

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

    async def test_close_after_terminal_preserves_failure(self) -> None:
        cases: tuple[
            tuple[tuple[CanonicalStreamItem, ...], type[BaseException], str],
            ...,
        ] = (
            (self._errored_canonical_items(), RuntimeError, "provider failed"),
            (
                self._cancelled_canonical_items(),
                CancelledError,
                "stream cancelled",
            ),
        )

        for items, exception_type, message in cases:
            with self.subTest(exception_type=exception_type.__name__):

                class Output:
                    def __init__(self) -> None:
                        self.items = iter(items)
                        self.close_count = 0

                    def __aiter__(self) -> "Output":
                        return self

                    async def __anext__(self) -> CanonicalStreamItem:
                        try:
                            return next(self.items)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc

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

                item: CanonicalStreamItem | None = None
                for _ in items:
                    item = await iterator.__anext__()

                assert item is not None
                self.assertTrue(item.is_stream_terminal)
                await resp.aclose()

                self.assertEqual(output.close_count, 1)
                self.assertTrue(consumed)
                with self.assertRaisesRegex(exception_type, message):
                    await resp.to_str()

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
                self.assertTrue(consumed)
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await resp.to_str()

    async def test_explicit_canonical_stream_terminal_keeps_to_str_failure(
        self,
    ) -> None:
        cases: tuple[
            tuple[tuple[CanonicalStreamItem, ...], type[BaseException], str],
            ...,
        ] = (
            (self._errored_canonical_items(), RuntimeError, "provider failed"),
            (
                self._cancelled_canonical_items(),
                CancelledError,
                "stream cancelled",
            ),
        )

        for items, exception_type, message in cases:
            with self.subTest(exception_type=exception_type.__name__):
                source = _ExplicitCanonicalStreamSource(items)
                consumed = False

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    source,
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
                            stream_session_id="requested-stream",
                            run_id="requested-run",
                            turn_id="requested-turn",
                        )
                    ]
                )
                accumulator = resp._stream_accumulator
                assert accumulator is not None

                self.assertIs(
                    stream_items[-1].terminal_outcome,
                    accumulator.terminal_outcome,
                )
                self.assertEqual(accumulator.answer_text, "partial")
                self.assertEqual(source.open_count, 1)
                self.assertEqual(source.call_count, 0)
                self.assertTrue(consumed)
                self.assertIsNone(resp._final_text)
                with self.assertRaisesRegex(exception_type, message):
                    await resp.to_str()
                self.assertEqual(source.open_count, 1)

    async def test_finalizer_terminal_paths_use_accumulator_answer(
        self,
    ) -> None:
        settings = GenerationSettings()
        stateful = TextGenerationResponse(
            lambda **_: self._errored_canonical_items(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        stateful._stream_accumulator = accumulate_canonical_stream_items(
            self._errored_canonical_items()
        )

        self.assertEqual(
            await stateful._finalize_stream_accumulation(
                raise_terminal_exception=False
            ),
            "partial",
        )
        self.assertEqual(
            await stateful._finalize_stream_accumulation(
                raise_terminal_exception=False
            ),
            "partial",
        )

        raising = TextGenerationResponse(
            lambda **_: self._errored_canonical_items(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        raising._stream_accumulator = accumulate_canonical_stream_items(
            self._errored_canonical_items()
        )

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await raising._finalize_stream_accumulation(
                raise_terminal_exception=True
            )

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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [
                item
                async for item in resp.consumer_projections(
                    stream_session_id="sdk-stream",
                    run_id="sdk-run",
                    turn_id="sdk-turn",
                )
            ]

        self.assertTrue(output.closed)
        self.assertFalse(consumed)
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await resp.to_str()

    async def test_consumer_projection_terminal_replays_to_str_failure(
        self,
    ) -> None:
        cases: tuple[
            tuple[tuple[CanonicalStreamItem, ...], type[BaseException], str],
            ...,
        ] = (
            (self._errored_canonical_items(), RuntimeError, "provider failed"),
            (
                self._cancelled_canonical_items(),
                CancelledError,
                "stream cancelled",
            ),
        )

        for items, exception_type, message in cases:
            with self.subTest(exception_type=exception_type.__name__):
                open_count = 0
                consumed = False

                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    for item in items:
                        yield item

                def output_fn(
                    **_: object,
                ) -> AsyncIterator[CanonicalStreamItem]:
                    nonlocal open_count
                    open_count += 1
                    return gen()

                def mark_consumed() -> None:
                    nonlocal consumed
                    consumed = True

                settings = GenerationSettings()
                resp = TextGenerationResponse(
                    output_fn,
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
                accumulator = resp._stream_accumulator
                assert accumulator is not None

                self.assertIs(
                    projections[-1].terminal_outcome,
                    accumulator.terminal_outcome,
                )
                self.assertEqual(accumulator.answer_text, "partial")
                self.assertEqual(open_count, 1)
                self.assertTrue(consumed)
                with self.assertRaisesRegex(exception_type, message):
                    await resp.to_str()
                with self.assertRaisesRegex(exception_type, message):
                    await resp.to_str()
                self.assertEqual(open_count, 1)
                self.assertTrue(consumed)

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
                self.assertTrue(consumed)
                with self.assertRaisesRegex(
                    CancelledError, "stream cancelled"
                ):
                    await resp.to_str()

    async def test_to_str_after_partial_canonical_iteration_keeps_answer(
        self,
    ) -> None:
        open_count = 0

        async def gen():
            for item in self._complete_canonical_items():
                yield item

        def output_fn(**_: object) -> AsyncIterator[CanonicalStreamItem]:
            nonlocal open_count
            open_count += 1
            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        self.assertEqual(await resp.to_str(), "ok")
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(accumulator.answer_text, "ok")
        self.assertEqual(open_count, 1)

    async def test_to_str_after_partial_projection_iteration_keeps_answer(
        self,
    ) -> None:
        open_count = 0

        async def gen():
            for item in self._complete_canonical_items():
                yield project_canonical_stream_item(item)

        def output_fn(**_: object) -> AsyncIterator[object]:
            nonlocal open_count
            open_count += 1
            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        self.assertEqual(await resp.to_str(), "ok")
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(accumulator.answer_text, "ok")
        self.assertEqual(open_count, 1)

    async def test_to_str_replays_cached_validation_failure(
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

        class Output:
            def __init__(self) -> None:
                self.items = iter(items)
                self.close_count = 0

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def aclose(self) -> None:
                self.close_count += 1

        outputs: list[Output] = []
        consumed = False

        def output_fn(**_: object) -> Output:
            output = Output()
            outputs.append(output)
            return output

        def mark_consumed() -> None:
            nonlocal consumed
            consumed = True

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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
        self.assertEqual((await iterator.__anext__()).text_delta, "partial")
        accumulator = resp._stream_accumulator
        assert accumulator is not None

        with self.assertRaisesRegex(
            StreamValidationError, "stream missing terminal outcome"
        ) as first_error:
            await resp.to_str()
        with self.assertRaises(StreamValidationError) as second_error:
            await resp.to_str()

        self.assertEqual(
            str(second_error.exception), str(first_error.exception)
        )
        self.assertIs(resp._stream_accumulator, accumulator)
        self.assertEqual(accumulator.answer_text, "partial")
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].close_count, 1)
        self.assertFalse(consumed)

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
                "unsupported legacy SDK response stream item",
            ),
            (
                ("legacy", first_semantic_item),
                "unsupported legacy SDK response stream item",
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

                if isinstance(items[0], CanonicalStreamItem):
                    await iterator.__anext__()
                    self.assertEqual(resp.output_token_count, 1)
                else:
                    self.assertEqual(resp.output_token_count, 0)
                with self.assertRaisesRegex(StreamValidationError, message):
                    await iterator.__anext__()

                self.assertEqual(output.close_count, 1)

    async def test_async_iteration_closes_on_unexpected_record_error(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="response-stream",
            run_id="response-run",
            turn_id="response-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )

        class Output:
            def __init__(self) -> None:
                self.items = iter((item,))
                self.close_count = 0

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self.items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def aclose(self) -> None:
                self.close_count += 1

        class FailingAccumulator:
            def add(self, _: CanonicalStreamItem) -> None:
                raise RuntimeError("accumulator failed")

        output = Output()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()
        resp._stream_accumulator = cast(Any, FailingAccumulator())

        with self.assertRaisesRegex(RuntimeError, "accumulator failed"):
            await resp.__anext__()

        self.assertEqual(output.close_count, 1)

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

    async def test_async_iteration_closes_on_unexpected_stop_validation_error(
        self,
    ) -> None:
        class Output:
            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.close_count += 1

        class FailingAccumulator:
            def validate_complete(self) -> None:
                raise RuntimeError("validate failed")

        output = Output()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()
        resp._stream_accumulator = cast(Any, FailingAccumulator())

        with self.assertRaisesRegex(RuntimeError, "validate failed"):
            await resp.__anext__()

        self.assertEqual(output.close_count, 1)

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
            "unsupported legacy SDK response stream item",
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
            "unsupported legacy SDK response stream item",
        ):
            await resp.to_str()

    async def test_consumer_projections_accept_semantic_output_items(
        self,
    ) -> None:
        for projected in (False, True):
            with self.subTest(projected=projected):

                async def gen():
                    for item in self._complete_canonical_items():
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
                    [item.sequence for item in projections], [0, 1, 2, 3]
                )
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await iterator.__anext__()

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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await iterator.__anext__()

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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            _ = [item async for item in resp]

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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await iterator.__anext__()

        self.assertEqual(output.read_count, 1)
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

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await resp.to_str()

        self.assertEqual(output.read_count, 1)
        self.assertTrue(output.closed)

    async def test_restarted_iteration_continues_active_session(
        self,
    ) -> None:
        open_count = 0

        async def gen():
            for item in self._complete_canonical_items():
                yield item

        def output_fn(**_: object) -> AsyncIterator[CanonicalStreamItem]:
            nonlocal open_count
            open_count += 1
            return gen()

        settings = GenerationSettings()
        resp = TextGenerationResponse(
            output_fn,
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

        tokens = [token async for token in resp]

        self.assertEqual(open_count, 1)
        self.assertEqual(
            [token.kind for token in tokens],
            [
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(await resp.to_str(), "ok")
        self.assertEqual(resp.output_token_count, 4)

    async def test_restarted_iteration_keeps_cached_final_text(self) -> None:
        values = iter(("first", "second"))

        def output_fn(**_: object):
            async def gen():
                text = next(values)
                for item in (
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
                        text_delta=text,
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
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ):
                    yield item

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
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            resp.__aiter__()

        self.assertEqual(await resp.to_str(), "first")

    async def test_session_ownership_edge_paths(self) -> None:
        settings = GenerationSettings()
        non_stream = TextGenerationResponse(
            lambda **_: "ok",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )
        non_stream._claim_stream_session()
        self.assertEqual(await non_stream.to_str(), "ok")

        open_count = 0

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in self._complete_canonical_items():
                yield item

        def output_fn(**_: object) -> AsyncIterator[CanonicalStreamItem]:
            nonlocal open_count
            open_count += 1
            return gen()

        started = TextGenerationResponse(
            output_fn,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        started.__aiter__()
        started._start_stream_output()
        await started.aclose()
        self.assertEqual(open_count, 1)

        active_to_str = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        active_to_str.__aiter__()
        self.assertEqual(await active_to_str.to_str(), "ok")

        closed = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        await closed.aclose()
        await closed.cancel()

        finalized = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        self.assertEqual(await finalized.to_str(), "ok")
        await finalized.cancel()

    async def test_to_str_rejects_claimed_unopened_stream_session(
        self,
    ) -> None:
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: self._complete_canonical_items(),
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

        try:
            with self.assertRaisesRegex(RuntimeError, "single-use"):
                await resp.to_str()
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
            await resp.aclose()

    async def test_to_str_after_close_returns_accumulated_answer(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
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
        await resp.aclose()

        self.assertEqual(await resp.to_str(), "ok")

    async def test_direct_iteration_closes_on_source_exception(self) -> None:
        class Output:
            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                raise RuntimeError("source failed")

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
        resp.__aiter__()

        with self.assertRaisesRegex(RuntimeError, "source failed"):
            await resp.__anext__()

        self.assertEqual(output.close_count, 1)

    async def test_direct_iteration_closes_on_record_cancellation(
        self,
    ) -> None:
        class Output:
            def __init__(self) -> None:
                self.close_count = 0
                items = (
                    TextGenerationResponseMoreTestCase._complete_canonical_items()
                )
                self.item = items[0]

            def __aiter__(self) -> "Output":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                return self.item

            async def aclose(self) -> None:
                self.close_count += 1

        class CancellingAccumulator:
            def add(self, _: CanonicalStreamItem) -> None:
                raise CancelledError()

        output = Output()
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: output,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        resp.__aiter__()
        resp._stream_accumulator = cast(Any, CancellingAccumulator())

        with self.assertRaises(CancelledError):
            await resp.__anext__()

        self.assertEqual(output.close_count, 1)
