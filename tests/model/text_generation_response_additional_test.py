from asyncio import CancelledError, Event, create_task
from dataclasses import dataclass
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from avalan.entities import GenerationSettings, ReasoningSettings
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)


@dataclass
class Example:
    value: str


class TextGenerationResponseAdditionalTestCase(IsolatedAsyncioTestCase):
    @staticmethod
    def _response() -> TextGenerationResponse:
        settings = GenerationSettings()
        return TextGenerationResponse(
            lambda **_: "ok",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )

    async def test_to_entity(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: '{"value": "ok"}',
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )
        result = await resp.to(Example)
        self.assertEqual(result, Example(value="ok"))

    async def test_disable_reasoning_parser(self):
        async def gen():
            for t in ("<think>", "a", "</think>"):
                yield t

        gs = GenerationSettings(reasoning=ReasoningSettings(enabled=False))
        resp = TextGenerationResponse(
            lambda **_: gen(),
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

    async def test_usage_reads_completed_output_stream(self):
        usage = {"input_tokens": 2, "output_tokens": 1}

        class UsageStream:
            def __init__(self) -> None:
                self._items = iter(
                    (
                        CanonicalStreamItem(
                            stream_session_id="usage-stream",
                            run_id="usage-run",
                            turn_id="usage-turn",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="usage-stream",
                            run_id="usage-run",
                            turn_id="usage-turn",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="ok",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="usage-stream",
                            run_id="usage-run",
                            turn_id="usage-turn",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="usage-stream",
                            run_id="usage-run",
                            turn_id="usage-turn",
                            sequence=3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage=usage,
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    )
                )
                self.usage = None

            def __aiter__(self) -> "UsageStream":
                return self

            async def __anext__(self) -> CanonicalStreamItem:
                try:
                    item = next(self._items)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc
                if item.is_stream_terminal:
                    self.usage = usage
                return item

        class StreamFactory:
            def __init__(self) -> None:
                self.stream = UsageStream()

            def __call__(self, **_: object) -> UsageStream:
                return self.stream

        settings = GenerationSettings()
        factory = StreamFactory()
        resp = TextGenerationResponse(
            factory,
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        self.assertIsNone(resp.usage)
        self.assertEqual(await resp.to_str(), "ok")
        self.assertEqual(resp.usage, usage)

    async def test_cleanup_task_observer_handles_terminal_failures(
        self,
    ) -> None:
        wait_forever = Event()

        async def wait_for_release() -> None:
            await wait_forever.wait()

        cancelled_task = create_task(wait_for_release())
        cancelled_task.cancel()
        with self.assertRaises(CancelledError):
            await cancelled_task
        TextGenerationResponse._observe_cleanup_task(cancelled_task)

        pending_task = create_task(wait_for_release())
        TextGenerationResponse._observe_cleanup_task(pending_task)
        pending_task.cancel()
        with self.assertRaises(CancelledError):
            await pending_task

    async def test_reap_cleanup_tasks_discards_cancelled_stage(self) -> None:
        response = self._response()
        wait_forever = Event()

        async def wait_for_release() -> None:
            await wait_forever.wait()

        cancelled_task = create_task(wait_for_release())
        cancelled_task.cancel()
        with self.assertRaises(CancelledError):
            await cancelled_task
        response._cleanup_tasks["poll"] = cancelled_task

        self.assertEqual(
            response._reap_cleanup_tasks(exclude="aclose"),
            [],
        )
        self.assertNotIn("poll", response._cleanup_tasks)

    def test_cleanup_failure_notes_are_identity_deduplicated(self) -> None:
        primary = RuntimeError("primary")
        cleanup = RuntimeError("cleanup")

        TextGenerationResponse._attach_cleanup_failures(
            primary,
            [primary, cleanup, cleanup],
        )

        self.assertEqual(
            getattr(primary, "__notes__", []),
            ["post-provider cleanup failure: RuntimeError: cleanup"],
        )

    def test_continuation_snapshot_adapter_requires_complete_contract(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "must export, import, and validate reserved calls",
        ):
            TextGenerationResponse(
                lambda **_: "ok",
                logger=getLogger(),
                use_async_generator=False,
                continuation_snapshot_adapter=object(),
            )

    async def test_interrupted_iteration_preserves_cancel_cleanup_failure(
        self,
    ) -> None:
        response = self._response()

        async def cleanup_stage(
            stage: str,
            cleanup: object,
        ) -> None:
            self.assertTrue(callable(cleanup))
            if stage == "cancel":
                raise RuntimeError("cancel cleanup")
            self.assertEqual(stage, "aclose")

        primary = CancelledError()
        with patch.object(
            response,
            "_run_bounded_cleanup_stage",
            new=AsyncMock(side_effect=cleanup_stage),
        ):
            await response._settle_iteration_failure(primary)

        self.assertEqual(
            getattr(primary, "__notes__", []),
            ["post-provider cleanup failure: RuntimeError: cancel cleanup"],
        )
