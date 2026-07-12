from asyncio import (
    CancelledError,
    create_task,
)
from asyncio import (
    Event as AsyncEvent,
)
from collections.abc import AsyncIterator
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    project_canonical_stream_item,
)
from avalan.server.routers.mcp import MCPResourceStore
from avalan.server.routers.streaming import (
    ProtocolStreamAccumulator,
    ProtocolStreamProjectionState,
    ProtocolStreamRetentionSettings,
    cancellable_stream_iterator,
    canonical_flow_public_metadata,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
    protocol_stream_terminal_snapshot,
    protocol_stream_usage_mappings,
    stream_consumer_iterator,
    stream_iterator,
    stream_terminal_succeeded,
)


class RouterStreamingTestCase(IsolatedAsyncioTestCase):
    async def test_stream_iterator_returns_source_iterator(self) -> None:
        async def gen():
            yield "ok"

        iterator = stream_iterator(gen())

        self.assertEqual(await anext(iterator), "ok")
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_iterator_rejects_non_async_iterable(self) -> None:
        with self.assertRaises(AssertionError):
            stream_iterator(object())

    async def test_stream_consumer_iterator_prefers_projection_api(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.kwargs: dict[str, str] | None = None
                self.raw_iterated = False

            def __aiter__(self) -> AsyncIterator[str]:
                self.raw_iterated = True

                async def gen() -> AsyncIterator[str]:
                    yield "raw"

                return gen()

            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                self.kwargs = {
                    "stream_session_id": stream_session_id,
                    "run_id": run_id,
                    "turn_id": turn_id,
                }

                async def gen() -> AsyncIterator[object]:
                    items = (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="projected",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=4,
                            kind=StreamItemKind.STREAM_CLOSED,
                            channel=StreamChannel.CONTROL,
                        ),
                    )
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        source = Source()

        iterator = stream_consumer_iterator(
            source,
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        projection = await anext(iterator)
        self.assertIs(projection.kind, StreamItemKind.STREAM_STARTED)
        projection = await anext(iterator)
        self.assertEqual(projection.text_delta, "projected")
        self.assertEqual(
            source.kwargs,
            {
                "stream_session_id": "stream",
                "run_id": "run",
                "turn_id": "turn",
            },
        )
        self.assertFalse(source.raw_iterated)
        self.assertIs((await anext(iterator)).kind, StreamItemKind.ANSWER_DONE)
        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_COMPLETED
        )
        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_CLOSED
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_raw_fallback_items(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[str]:
            yield "raw"

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            unsupported_message="unsupported raw fallback",
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported raw fallback",
        ):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_legacy_first_items(
        self,
    ) -> None:
        legacy_rejection_items = (
            "legacy",
            Token(token="legacy"),
            TokenDetail(token="legacy", id=1),
            ReasoningToken(token="legacy"),
            ToolCallToken(
                token="{}",
                call=ToolCall(
                    id="legacy-call",
                    name="legacy",
                    arguments={},
                ),
            ),
            Event(type=EventType.START),
            object(),
        )

        for legacy_rejection_item in legacy_rejection_items:
            with self.subTest(item_type=type(legacy_rejection_item).__name__):

                async def gen() -> AsyncIterator[object]:
                    yield legacy_rejection_item

                iterator = stream_consumer_iterator(
                    gen(),
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    unsupported_message="unsupported shared consumer item",
                )

                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported shared consumer item",
                ):
                    await anext(iterator)

    async def test_stream_consumer_iterator_projects_canonical_async_iterable(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(
                    tool_call_id="tool-1",
                ),
                text_delta="canonical",
                provider_family="fixture",
                provider_event_type="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.USAGE_UPDATE,
                channel=StreamChannel.USAGE,
                usage={"input_tokens": 1},
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                usage={"output_tokens": 1},
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        self.assertIs(
            (await anext(iterator)).kind,
            StreamItemKind.STREAM_STARTED,
        )
        projection = await anext(iterator)
        self.assertIsInstance(projection, StreamConsumerProjection)
        self.assertEqual(projection.stream_session_id, "s")
        self.assertEqual(projection.run_id, "r")
        self.assertEqual(projection.turn_id, "t")
        self.assertEqual(projection.sequence, 1)
        self.assertIs(projection.channel, StreamChannel.ANSWER)
        self.assertIs(projection.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(projection.tool_call_id, "tool-1")
        self.assertEqual(projection.text_delta, "canonical")
        self.assertEqual(projection.provider_family, "fixture")
        self.assertEqual(projection.provider_event_type, "answer")
        self.assertIs((await anext(iterator)).kind, StreamItemKind.ANSWER_DONE)
        usage = await anext(iterator)
        self.assertIs(usage.kind, StreamItemKind.USAGE_UPDATE)
        self.assertEqual(usage.usage, {"input_tokens": 1})
        terminal = await anext(iterator)
        self.assertIs(terminal.kind, StreamItemKind.STREAM_COMPLETED)
        self.assertEqual(terminal.usage, {"output_tokens": 1})
        self.assertIs(
            (await anext(iterator)).kind,
            StreamItemKind.STREAM_CLOSED,
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_consumer_iterator_closes_distinct_iterator(
        self,
    ) -> None:
        class Iterator:
            def __init__(self) -> None:
                self.close_count = 0
                self._items = iter(
                    (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                            usage={},
                        ),
                    )
                )

            def __aiter__(self) -> "Iterator":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def aclose(self) -> None:
                self.close_count += 1

        class Source:
            def __init__(self) -> None:
                self.iterator = Iterator()
                self.close_count = 0

            def __aiter__(self) -> Iterator:
                return self.iterator

            async def aclose(self) -> None:
                self.close_count += 1

        source = Source()
        iterator = stream_consumer_iterator(
            source,
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        items = [item async for item in iterator]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(source.iterator.close_count, 1)
        self.assertEqual(source.close_count, 0)

    async def test_stream_consumer_iterator_closes_direct_generator_on_aclose(
        self,
    ) -> None:
        closed = False

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            nonlocal closed
            try:
                yield CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                )
                yield CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    usage={},
                )
            finally:
                closed = True

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        self.assertIs(
            (await anext(iterator)).kind,
            StreamItemKind.STREAM_STARTED,
        )
        await iterator.aclose()

        self.assertTrue(closed)

    async def test_stream_consumer_iterator_closes_direct_source_on_rejection(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> object:
                return "legacy"

            async def aclose(self) -> None:
                self.close_count += 1

        source = Source()
        iterator = stream_consumer_iterator(
            source,
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
            unsupported_message="unsupported direct source",
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported direct source",
        ):
            await anext(iterator)

        self.assertEqual(source.close_count, 1)

    async def test_stream_consumer_iterator_closes_direct_source_on_cancel(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0
                self.started = AsyncEvent()
                self.pull_cancelled = False

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.pull_cancelled = True
                    raise
                raise StopAsyncIteration

            async def aclose(self) -> None:
                self.close_count += 1

        source = Source()
        iterator = stream_consumer_iterator(
            source,
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )
        task = create_task(anext(iterator))
        await source.started.wait()
        task.cancel()

        with self.assertRaises(CancelledError):
            await task

        self.assertTrue(source.pull_cancelled)
        self.assertEqual(source.close_count, 1)

    async def test_stream_consumer_iterator_leaves_direct_source_open(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.close_count = 0
                self._items = iter(
                    (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                            usage={},
                        ),
                    )
                )

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    return next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

            async def aclose(self) -> None:
                self.close_count += 1

        source = Source()
        iterator = stream_consumer_iterator(
            source,
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        items = [item async for item in iterator]

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        self.assertEqual(source.close_count, 0)

    async def test_stream_consumer_iterator_preserves_terminal_error_data(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={"message": "provider failed", "code": "upstream"},
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_STARTED
        )
        terminal = await anext(iterator)
        self.assertIs(terminal.kind, StreamItemKind.STREAM_ERRORED)
        self.assertEqual(
            terminal.data,
            {"message": "provider failed", "code": "upstream"},
        )
        self.assertIs(
            terminal.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertIs(
            (await anext(iterator)).kind,
            StreamItemKind.STREAM_CLOSED,
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_consumer_iterator_preserves_terminal_cancelled_data(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                data={"reason": "client_disconnect", "retryable": False},
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="fallback-stream",
            run_id="fallback-run",
            turn_id="fallback-turn",
        )

        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_STARTED
        )
        terminal = await anext(iterator)
        self.assertIs(terminal.kind, StreamItemKind.STREAM_CANCELLED)
        self.assertEqual(
            terminal.data,
            {"reason": "client_disconnect", "retryable": False},
        )
        self.assertIs(
            terminal.terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )
        self.assertFalse(stream_terminal_succeeded(terminal))
        snapshot = protocol_stream_terminal_snapshot(terminal)
        self.assertEqual(
            snapshot.data,
            {"reason": "client_disconnect", "retryable": False},
        )
        self.assertIs(snapshot.outcome, StreamTerminalOutcome.CANCELLED)
        self.assertFalse(snapshot.succeeded)
        self.assertIs(
            (await anext(iterator)).kind,
            StreamItemKind.STREAM_CLOSED,
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_bad_projection_api(
        self,
    ) -> None:
        class Source:
            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> object:
                return object()

        with self.assertRaises(AssertionError):
            stream_consumer_iterator(
                Source(),
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
            )

    async def test_stream_consumer_iterator_rejects_non_projection_item(
        self,
    ) -> None:
        class Source:
            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                async def gen() -> AsyncIterator[object]:
                    yield "legacy"

                return gen()

        iterator = stream_consumer_iterator(
            Source(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "consumer projection stream item must be StreamConsumerProjection",
        ):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_projection_sequence_gap(
        self,
    ) -> None:
        class Source:
            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                async def gen() -> AsyncIterator[object]:
                    items = (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="dropped predecessor",
                        ),
                    )
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        iterator = stream_consumer_iterator(
            Source(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_STARTED
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "lossless consumer stream sequence gap",
        ):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_out_of_order_projection(
        self,
    ) -> None:
        class Source:
            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                async def gen() -> AsyncIterator[object]:
                    items = (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="out of order",
                        ),
                    )
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        iterator = stream_consumer_iterator(
            Source(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_STARTED
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "stream sequence must increase",
        ):
            await anext(iterator)

    async def test_stream_consumer_iterator_rejects_late_projection_item(
        self,
    ) -> None:
        class Source:
            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ) -> AsyncIterator[object]:
                async def gen() -> AsyncIterator[object]:
                    items = (
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=1,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="late",
                        ),
                    )
                    for item in items:
                        yield project_canonical_stream_item(item)

                return gen()

        iterator = stream_consumer_iterator(
            Source(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_STARTED
        )
        self.assertIs(
            (await anext(iterator)).kind, StreamItemKind.STREAM_COMPLETED
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await anext(iterator)

    def test_protocol_stream_projection_state_projects_canonical_stream(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )
        items = (
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={"total_tokens": 2},
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        projections = [
            state.project(
                item,
                item.sequence,
                unsupported_message="unsupported stream item",
            )
            for item in items
        ]

        state.validate_complete()
        self.assertTrue(state.has_canonical_items)
        self.assertEqual(state.accumulator.answer_text, "answer")
        self.assertEqual(state.accumulator.final_usage, {"total_tokens": 2})
        self.assertEqual(projections[1].text_delta, "answer")
        terminal = state.terminal_projection()
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.sequence, 4)
        self.assertIs(
            terminal.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )

    def test_protocol_stream_projection_state_accepts_projection_items(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )
        start_item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        canonical_item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="projected",
        )
        state.project(
            project_canonical_stream_item(start_item),
            0,
            unsupported_message="unsupported stream item",
        )
        projection = project_canonical_stream_item(canonical_item)

        result = state.project(
            projection,
            1,
            unsupported_message="unsupported stream item",
        )

        self.assertIs(result, projection)
        self.assertTrue(state.has_canonical_items)
        self.assertEqual(state.accumulator.answer_text, "projected")

    def test_protocol_stream_projection_state_can_skip_accumulation(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            accumulate=False,
        )
        item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="projected",
        )

        projection = state.project(
            item,
            0,
            unsupported_message="unsupported stream item",
        )

        state.validate_complete()
        self.assertTrue(state.has_canonical_items)
        self.assertEqual(projection.text_delta, "projected")
        self.assertEqual(state.accumulator.items, ())
        self.assertIsNone(state.terminal_projection())

    def test_protocol_stream_projection_state_returns_no_terminal_projection(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        self.assertIsNone(state.terminal_projection())

    def test_protocol_stream_terminal_snapshot_preserves_projection_fields(
        self,
    ) -> None:
        terminal = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=4,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data={"message": "failed"},
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            )
        )

        snapshot = protocol_stream_terminal_snapshot(terminal)

        self.assertIs(snapshot.outcome, StreamTerminalOutcome.ERRORED)
        self.assertEqual(snapshot.sequence, 4)
        self.assertEqual(snapshot.data, {"message": "failed"})
        self.assertFalse(snapshot.succeeded)

    def test_protocol_stream_terminal_snapshot_preserves_outcomes(
        self,
    ) -> None:
        completed = protocol_stream_terminal_snapshot(
            StreamTerminalOutcome.COMPLETED
        )
        cancelled = protocol_stream_terminal_snapshot(
            StreamTerminalOutcome.CANCELLED
        )
        missing = protocol_stream_terminal_snapshot(None)

        self.assertIs(completed.outcome, StreamTerminalOutcome.COMPLETED)
        self.assertTrue(completed.succeeded)
        self.assertIsNone(completed.sequence)
        self.assertIs(cancelled.outcome, StreamTerminalOutcome.CANCELLED)
        self.assertFalse(cancelled.succeeded)
        self.assertIsNone(missing.outcome)
        self.assertTrue(missing.succeeded)

    def test_protocol_stream_terminal_snapshot_rejects_non_terminal_projection(
        self,
    ) -> None:
        projection = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
        )

        with self.assertRaises(AssertionError):
            protocol_stream_terminal_snapshot(projection)

    def test_protocol_stream_projection_state_legacy_rejection_first_item(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        with self.assertRaisesRegex(
            StreamValidationError, "unsupported stream item"
        ):
            state.project(
                "legacy",
                0,
                unsupported_message="unsupported stream item",
            )

        self.assertFalse(state.has_canonical_items)

    def test_protocol_stream_projection_state_rejects_mixed_surfaces(
        self,
    ) -> None:
        canonical_item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )
        state.project(
            canonical_item,
            0,
            unsupported_message="unsupported stream item",
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported stream item",
        ):
            state.project(
                "legacy",
                1,
                unsupported_message="unsupported stream item",
            )

    def test_protocol_stream_projection_state_rejects_unsupported_items(
        self,
    ) -> None:
        state = ProtocolStreamProjectionState(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        with self.assertRaisesRegex(
            StreamValidationError, "unsupported stream item for test"
        ):
            state.project(
                Event(type=EventType.START, payload={}),
                0,
                unsupported_message="unsupported stream item for test",
            )

    async def test_cancellable_stream_iterator_yields_until_cancelled(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[str]:
            yield "first"
            yield "second"

        cancel_event = AsyncEvent()
        iterator = cancellable_stream_iterator(gen(), cancel_event)

        self.assertEqual(await anext(iterator), "first")
        cancel_event.set()
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_cancellable_stream_iterator_stops_on_source_exhaustion(
        self,
    ) -> None:
        class Source:
            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> str:
                raise StopAsyncIteration

        iterator = cancellable_stream_iterator(Source(), AsyncEvent())

        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_cancellable_stream_iterator_skips_prescheduled_pull(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.pulled = False

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> str:
                self.pulled = True
                return "late"

        source = Source()
        cancel_event = AsyncEvent()
        cancel_event.set()
        iterator = cancellable_stream_iterator(source, cancel_event)

        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)
        self.assertFalse(source.pulled)

    async def test_cancellable_stream_iterator_interrupts_pending_pull(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.started = AsyncEvent()
                self.cancelled = False

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> str:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.cancelled = True
                    raise
                return "late"

        source = Source()
        cancel_event = AsyncEvent()
        iterator = cancellable_stream_iterator(source, cancel_event)
        pull = create_task(anext(iterator))

        await source.started.wait()
        cancel_event.set()

        with self.assertRaises(StopAsyncIteration):
            await pull
        self.assertTrue(source.cancelled)

    async def test_cancellable_iterator_ends_when_cancel_races_exhaustion(
        self,
    ) -> None:
        cancel_event = AsyncEvent()

        class Source:
            def __init__(self) -> None:
                self.pulled = False

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> str:
                self.pulled = True
                cancel_event.set()
                raise StopAsyncIteration

        source = Source()
        iterator = cancellable_stream_iterator(source, cancel_event)

        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)
        self.assertTrue(source.pulled)

    async def test_cancellable_stream_iterator_cleans_up_on_consumer_cancel(
        self,
    ) -> None:
        class Source:
            def __init__(self) -> None:
                self.started = AsyncEvent()
                self.cancelled = False

            def __aiter__(self) -> "Source":
                return self

            async def __anext__(self) -> str:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.cancelled = True
                    raise
                return "late"

        source = Source()
        cancel_event = AsyncEvent()
        iterator = cancellable_stream_iterator(source, cancel_event)
        pull = create_task(anext(iterator))

        await source.started.wait()
        pull.cancel()

        with self.assertRaises(CancelledError):
            await pull
        self.assertTrue(source.cancelled)

    async def test_cancellable_stream_iterator_rejects_bad_cancel_event(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[str]:
            yield "never"

        iterator = cancellable_stream_iterator(
            gen(),
            object(),  # type: ignore[arg-type]
        )

        with self.assertRaises(AssertionError):
            await anext(iterator)

    async def test_stream_terminal_succeeded_preserves_terminal_state(
        self,
    ) -> None:
        completed = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )
        cancelled = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_CANCELLED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            )
        )
        errored = project_canonical_stream_item(
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            )
        )

        self.assertTrue(stream_terminal_succeeded(None))
        self.assertTrue(
            stream_terminal_succeeded(StreamTerminalOutcome.COMPLETED)
        )
        self.assertTrue(stream_terminal_succeeded(completed))
        self.assertFalse(
            stream_terminal_succeeded(StreamTerminalOutcome.CANCELLED)
        )
        self.assertFalse(
            stream_terminal_succeeded(StreamTerminalOutcome.ERRORED)
        )
        self.assertFalse(stream_terminal_succeeded(cancelled))
        self.assertFalse(stream_terminal_succeeded(errored))
        with self.assertRaises(AssertionError):
            stream_terminal_succeeded(object())  # type: ignore[arg-type]

    async def test_protocol_stream_accumulator_separates_channels(
        self,
    ) -> None:
        accumulator = ProtocolStreamAccumulator()
        for item in (
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
                text_delta='{"a":',
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.TOOL_EXECUTION_STARTED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=7,
                kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
                text_delta="stdout",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=8,
                kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                channel=StreamChannel.TOOL_EXECUTION,
                correlation=StreamItemCorrelation(tool_call_id="tool-1"),
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=9,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=10,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=11,
                kind=StreamItemKind.STREAM_DIAGNOSTIC,
                channel=StreamChannel.CONTROL,
                data={"code": "notice"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=12,
                kind=StreamItemKind.FLOW_EVENT,
                channel=StreamChannel.FLOW,
                correlation=StreamItemCorrelation(
                    flow_run_id="flow-1", node_id="node-1"
                ),
                data={"state": "running"},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=13,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={"input_text_tokens": 1, "output_text_tokens": 2},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=14,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ):
            accumulator.add(item)

        snapshot = accumulator.snapshot()

        self.assertEqual(snapshot.answer_text, "answer")
        self.assertEqual(snapshot.reasoning_text, "plan")
        self.assertEqual(
            snapshot.usage, {"input_text_tokens": 1, "output_text_tokens": 2}
        )
        self.assertIs(
            snapshot.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertTrue(snapshot.terminal_succeeded)
        self.assertEqual(snapshot.tool_call_arguments, {"tool-1": '{"a":'})
        self.assertEqual(snapshot.tool_execution_outputs, {"tool-1": "stdout"})
        self.assertEqual(len(snapshot.diagnostics), 1)
        self.assertEqual(snapshot.diagnostics[0].data, {"code": "notice"})
        self.assertEqual(len(snapshot.flow_items), 1)
        self.assertEqual(
            snapshot.flow_items[0].correlation.flow_run_id, "flow-1"
        )
        self.assertEqual(snapshot.usage_items[0].usage, snapshot.usage)
        self.assertEqual(
            [item.kind for item in snapshot.control_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )
        accumulator.validate_complete()

    async def test_protocol_stream_accumulator_uses_custom_retention_policy(
        self,
    ) -> None:
        retention_policy = StreamRetentionPolicy(
            accumulator_item_limit=1,
            replay_history_item_limit=1,
            metrics_history_item_limit=0,
            flow_history_item_limit=0,
        )
        accumulator = ProtocolStreamAccumulator(
            retention_policy=retention_policy
        )
        completed_data: dict[str, object] = {"message": "finished"}
        final_usage: dict[str, object] = {
            "input_text_tokens": 2,
            "output_text_tokens": 1,
        }
        closed_item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

        for item in (
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                data=completed_data,
                usage=final_usage,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            closed_item,
        ):
            accumulator.add(item)

        snapshot = accumulator.snapshot()

        self.assertEqual(snapshot.answer_text, "answer")
        self.assertEqual(snapshot.usage, final_usage)
        self.assertEqual(snapshot.usage_items, ())
        self.assertEqual(snapshot.control_items, (closed_item,))
        self.assertIs(
            snapshot.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertEqual(snapshot.terminal_snapshot.sequence, 3)
        self.assertEqual(snapshot.terminal_snapshot.data, completed_data)
        self.assertTrue(snapshot.terminal_snapshot.succeeded)
        accumulator.validate_complete()

    async def test_protocol_stream_accumulator_keeps_terminal_error_snapshot(
        self,
    ) -> None:
        accumulator = ProtocolStreamAccumulator(
            retention_policy=StreamRetentionPolicy(
                accumulator_item_limit=1,
                replay_history_item_limit=0,
            )
        )
        error_data: dict[str, object] = {
            "message": "upstream failed",
            "code": "provider",
        }

        for item in (
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                data=error_data,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            ),
        ):
            accumulator.add(item)

        snapshot = accumulator.snapshot()

        self.assertEqual(snapshot.control_items, ())
        self.assertIs(snapshot.terminal_outcome, StreamTerminalOutcome.ERRORED)
        self.assertFalse(snapshot.terminal_succeeded)
        self.assertIs(
            snapshot.terminal_snapshot.outcome, StreamTerminalOutcome.ERRORED
        )
        self.assertEqual(snapshot.terminal_snapshot.sequence, 1)
        self.assertEqual(snapshot.terminal_snapshot.data, error_data)
        self.assertFalse(snapshot.terminal_snapshot.succeeded)
        accumulator.validate_complete()

    def test_canonical_flow_public_metadata_filters_private_fields(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.FLOW_EVENT,
            channel=StreamChannel.FLOW,
            correlation=StreamItemCorrelation(
                flow_run_id="flow-1",
                node_id="node-1",
                parent_sequence=7,
            ),
            data={
                "state": "paused",
                "status": "waiting",
                "private_output": "secret",
            },
            metadata={
                "event_type": "flow_node_paused",
                "state": "paused",
                "status": "waiting",
                "attempt": 2,
                "progress_percent": 50.0,
                "parent_node_id": "parent",
                "child_node_id": "child",
                "private_output": "secret",
                "debug_payload": {"raw": "secret"},
            },
        )

        metadata = canonical_flow_public_metadata(item)

        self.assertEqual(
            metadata,
            {
                "event_type": "flow_node_paused",
                "state": "paused",
                "status": "waiting",
                "attempt": 2,
                "progress_percent": 50.0,
                "parent_node_id": "parent",
                "child_node_id": "child",
            },
        )
        self.assertEqual(metadata["state"], "paused")
        self.assertNotIn("private_output", metadata)
        self.assertNotIn("debug_payload", metadata)

    def test_canonical_flow_public_metadata_rejects_non_flow_items(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )

        with self.assertRaises(AssertionError):
            canonical_flow_public_metadata(item)

    def test_protocol_stream_usage_mappings_preserves_canonical_totals(
        self,
    ) -> None:
        usage = {"input_tokens": 1, "totals": {"input_tokens": 2}}

        self.assertEqual(
            protocol_stream_usage_mappings(usage),
            (usage, usage["totals"]),
        )
        self.assertEqual(protocol_stream_usage_mappings("usage"), ())

    async def test_protocol_stream_accumulator_rejects_duplicate_terminal(
        self,
    ) -> None:
        accumulator = ProtocolStreamAccumulator()
        items = (
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_ERRORED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )

        for item in items[:3]:
            accumulator.add(item)

        with self.assertRaisesRegex(
            StreamValidationError, "duplicate stream terminal item"
        ):
            accumulator.add(items[3])

    async def test_protocol_stream_accumulator_rejects_late_semantic_item(
        self,
    ) -> None:
        accumulator = ProtocolStreamAccumulator()
        for item in (
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.USAGE_COMPLETED,
                channel=StreamChannel.USAGE,
                usage={},
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ):
            accumulator.add(item)

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            accumulator.add(
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=3,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="late",
                )
            )

    def test_protocol_stream_retention_settings_use_policy_limits(
        self,
    ) -> None:
        settings = protocol_stream_retention_settings(
            StreamRetentionPolicy(
                mcp_resource_item_limit=3,
                mcp_resource_text_byte_limit=6,
                a2a_task_record_item_limit=4,
                a2a_task_event_byte_limit=7,
                flow_history_item_limit=5,
            )
        )

        self.assertEqual(settings.resource_item_limit, 3)
        self.assertEqual(settings.resource_text_byte_limit, 6)
        self.assertEqual(settings.task_record_item_limit, 4)
        self.assertEqual(settings.task_event_byte_limit, 7)
        self.assertEqual(settings.flow_history_item_limit, 5)
        self.assertTrue(settings.active_session_lossless)

    def test_protocol_stream_retention_settings_reject_invalid_values(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            protocol_stream_retention_settings(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=-1,
                resource_text_byte_limit=1,
                task_record_item_limit=0,
                task_event_byte_limit=2,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=1,
                task_record_item_limit=0,
                task_event_byte_limit=2,
                flow_history_item_limit=0,
                active_session_lossless=False,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=0,
                task_record_item_limit=0,
                task_event_byte_limit=2,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=-1,
                task_record_item_limit=0,
                task_event_byte_limit=2,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=True,
                task_record_item_limit=0,
                task_event_byte_limit=2,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=1,
                task_record_item_limit=0,
                task_event_byte_limit=1,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                resource_text_byte_limit=1,
                task_record_item_limit=0,
                task_event_byte_limit=True,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(mcp_resource_text_byte_limit=-1)
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(mcp_resource_text_byte_limit=0)
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(mcp_resource_text_byte_limit=True)
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(a2a_task_event_byte_limit=1)
        with self.assertRaises(AssertionError):
            StreamRetentionPolicy(a2a_task_event_byte_limit=True)

    async def test_default_server_stream_retention_surfaces_are_bounded(
        self,
    ) -> None:
        policy = StreamRetentionPolicy()
        settings = protocol_stream_retention_settings()
        event_manager = EventManager(mode=EventManagerMode.SERVER)
        resource_store = MCPResourceStore()

        self.assertEqual(
            settings.resource_item_limit,
            policy.mcp_resource_item_limit,
        )
        self.assertEqual(
            settings.resource_text_byte_limit,
            policy.mcp_resource_text_byte_limit,
        )
        self.assertEqual(
            settings.task_record_item_limit,
            policy.a2a_task_record_item_limit,
        )
        self.assertEqual(
            settings.task_event_byte_limit,
            policy.a2a_task_event_byte_limit,
        )
        self.assertEqual(
            settings.flow_history_item_limit,
            policy.flow_history_item_limit,
        )
        self.assertTrue(settings.active_session_lossless)

        await event_manager.trigger(Event(type=EventType.START))

        self.assertFalse(event_manager.history_config.enabled)
        self.assertEqual(event_manager.history, [])
        self.assertFalse(event_manager.listen_config.enabled)
        self.assertFalse(event_manager.collect_stats)
        self.assertEqual(event_manager.stats.total_triggers, 0)
        self.assertEqual(
            event_manager.default_delivery_config.queue_limit,
            32,
        )

        resource = await resource_store.create(base_path="/mcp")
        for index in range(policy.mcp_resource_item_limit + 2):
            await resource_store.append(resource.id, f"chunk-{index}")

        history = await resource_store.history(resource.id)
        self.assertLessEqual(len(history), policy.mcp_resource_item_limit)
        self.assertEqual(
            "".join(history),
            (await resource_store.get(resource.id)).text,
        )

    async def test_cleanup_stream_sources_closes_unique_sources(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        source = Source()

        await cleanup_stream_sources(source, source, object(), cancelled=True)

        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)

    async def test_cleanup_stream_sources_accepts_sync_methods(self) -> None:
        class Source:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            def cancel(self) -> None:
                self.cancel_count += 1

            def aclose(self) -> None:
                self.close_count += 1

        source = Source()

        await cleanup_stream_sources(source, cancelled=False)

        self.assertEqual(source.cancel_count, 0)
        self.assertEqual(source.close_count, 1)

    async def test_cleanup_stream_sources_rejects_bad_sync_result(
        self,
    ) -> None:
        class Source:
            def aclose(self) -> object:
                return object()

        with self.assertRaises(AssertionError):
            await cleanup_stream_sources(Source())

    async def test_cleanup_stream_sources_rejects_bad_async_result(
        self,
    ) -> None:
        class Source:
            async def aclose(self) -> object:
                return object()

        with self.assertRaises(AssertionError):
            await cleanup_stream_sources(Source())

    async def test_cleanup_stream_sources_closes_after_cancel_error(
        self,
    ) -> None:
        class FailingCancelSource:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1
                raise RuntimeError("cancel failed")

            async def aclose(self) -> None:
                self.close_count += 1

        class Source:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        failing_source = FailingCancelSource()
        source = Source()

        with self.assertRaisesRegex(RuntimeError, "cancel failed"):
            await cleanup_stream_sources(
                failing_source, source, cancelled=True
            )

        self.assertEqual(failing_source.cancel_count, 1)
        self.assertEqual(failing_source.close_count, 1)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)

    async def test_cleanup_stream_sources_reports_multiple_errors(
        self,
    ) -> None:
        class Source:
            def __init__(self, name: str) -> None:
                self.name = name
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1
                raise RuntimeError(f"{self.name} close failed")

        first = Source("first")
        second = Source("second")

        with self.assertRaises(ExceptionGroup) as context:
            await cleanup_stream_sources(first, second)

        self.assertEqual(first.close_count, 1)
        self.assertEqual(second.close_count, 1)
        self.assertEqual(len(context.exception.exceptions), 2)
        self.assertEqual(
            [str(error) for error in context.exception.exceptions],
            ["first close failed", "second close failed"],
        )

    async def test_cleanup_stream_sources_closes_after_cancelled_error(
        self,
    ) -> None:
        class FailingCancelSource:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1
                raise CancelledError()

            async def aclose(self) -> None:
                self.close_count += 1

        class Source:
            def __init__(self) -> None:
                self.cancel_count = 0
                self.close_count = 0

            async def cancel(self) -> None:
                self.cancel_count += 1

            async def aclose(self) -> None:
                self.close_count += 1

        failing_source = FailingCancelSource()
        source = Source()

        with self.assertRaises(CancelledError):
            await cleanup_stream_sources(
                failing_source, source, cancelled=True
            )

        self.assertEqual(failing_source.cancel_count, 1)
        self.assertEqual(failing_source.close_count, 1)
        self.assertEqual(source.cancel_count, 1)
        self.assertEqual(source.close_count, 1)

    async def test_cleanup_stream_sources_reports_mixed_base_errors(
        self,
    ) -> None:
        class CancelledSource:
            def __init__(self) -> None:
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1
                raise CancelledError()

        class FailingSource:
            def __init__(self) -> None:
                self.close_count = 0

            async def aclose(self) -> None:
                self.close_count += 1
                raise RuntimeError("close failed")

        cancelled_source = CancelledSource()
        failing_source = FailingSource()

        with self.assertRaises(BaseExceptionGroup) as context:
            await cleanup_stream_sources(cancelled_source, failing_source)

        self.assertEqual(cancelled_source.close_count, 1)
        self.assertEqual(failing_source.close_count, 1)
        self.assertIsInstance(context.exception.exceptions[0], CancelledError)
        self.assertIsInstance(context.exception.exceptions[1], RuntimeError)

    async def test_cleanup_stream_sources_rejects_bad_cancelled_flag(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            await cleanup_stream_sources(object(), cancelled=1)  # type: ignore[arg-type]
