from asyncio import CancelledError
from collections.abc import AsyncIterator
from unittest import IsolatedAsyncioTestCase

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    project_canonical_stream_item,
)
from avalan.server.routers.streaming import (
    cleanup_stream_sources,
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
