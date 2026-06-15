from asyncio import (
    CancelledError,
    create_task,
)
from asyncio import (
    Event as AsyncEvent,
)
from collections.abc import AsyncIterator
from unittest import IsolatedAsyncioTestCase

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    project_canonical_stream_item,
)
from avalan.server.routers.streaming import (
    ProtocolStreamAccumulator,
    ProtocolStreamRetentionSettings,
    cancellable_stream_iterator,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
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
                    yield project_canonical_stream_item(
                        CanonicalStreamItem(
                            stream_session_id="s",
                            run_id="r",
                            turn_id="t",
                            sequence=0,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="projected",
                        )
                    )

                return gen()

        source = Source()

        iterator = stream_consumer_iterator(
            source,
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

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
        with self.assertRaises(StopAsyncIteration):
            await anext(iterator)

    async def test_stream_consumer_iterator_falls_back_to_stream_iterator(
        self,
    ) -> None:
        async def gen() -> AsyncIterator[str]:
            yield "raw"

        iterator = stream_consumer_iterator(
            gen(),
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
        )

        self.assertEqual(await anext(iterator), "raw")
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
                a2a_task_record_item_limit=4,
                flow_history_item_limit=5,
            )
        )

        self.assertEqual(settings.resource_item_limit, 3)
        self.assertEqual(settings.task_record_item_limit, 4)
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
                task_record_item_limit=0,
                flow_history_item_limit=0,
                active_session_lossless=True,
            )
        with self.assertRaises(AssertionError):
            ProtocolStreamRetentionSettings(
                resource_item_limit=0,
                task_record_item_limit=0,
                flow_history_item_limit=0,
                active_session_lossless=False,
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
