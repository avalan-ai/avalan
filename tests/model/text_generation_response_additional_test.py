from dataclasses import dataclass
from logging import getLogger
from unittest import IsolatedAsyncioTestCase

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
