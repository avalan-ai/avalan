from collections.abc import AsyncIterable
from unittest import IsolatedAsyncioTestCase

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamTerminalOutcome,
    TextGenerationSingleStream,
    accumulate_canonical_stream_items,
)


async def _collect(
    stream: AsyncIterable[CanonicalStreamItem],
) -> tuple[CanonicalStreamItem, ...]:
    return tuple([item async for item in stream])


class TextGenerationSingleStreamTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_and_reset(self) -> None:
        stream = TextGenerationSingleStream("hello")
        iterator = stream.__aiter__()
        self.assertIs(iterator, stream)
        item = await iterator.__anext__()
        self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
        self.assertEqual(item.sequence, 0)
        self.assertEqual(item.provider_family, None)
        self.assertEqual(
            [item.kind for item in stream.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        for expected_kind in (
            StreamItemKind.ANSWER_DELTA,
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_CLOSED,
        ):
            self.assertIs((await iterator.__anext__()).kind, expected_kind)
        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()

        iterator2 = stream()
        self.assertIs(iterator2, stream)
        self.assertIs(
            (await iterator2.__anext__()).kind,
            StreamItemKind.STREAM_STARTED,
        )

    async def test_async_for_properties_and_usage(self) -> None:
        usage = {"input_tokens": 1}
        stream = TextGenerationSingleStream(
            "foo", provider_family="openai", usage=usage
        )
        items = []
        async for item in stream():
            items.append(item)
        self.assertEqual(items, list(stream.canonical_items))
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()
        iterator = stream.__aiter__()
        self.assertEqual(stream.content, "foo")
        self.assertEqual(stream.provider_family, "openai")
        self.assertEqual(stream.usage, usage)
        self.assertEqual(
            (await iterator.__anext__()).provider_family, "openai"
        )
        self.assertEqual(items[1].text_delta, "foo")
        self.assertIs(
            items[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertIs(items[-2].usage, usage)
        self.assertIs(items[-1].kind, StreamItemKind.STREAM_CLOSED)

    async def test_canonical_stream_uses_requested_identity(self) -> None:
        stream = TextGenerationSingleStream("foo", provider_family="openai")
        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.HOSTED,
            provider_family="openai",
        )

        items = await _collect(
            stream.canonical_stream(
                stream_session_id="requested-stream",
                run_id="requested-run",
                turn_id="requested-turn",
                provider_family="openai",
                capabilities=capabilities,
                close_after_terminal=False,
            )
        )

        self.assertEqual(
            {item.stream_session_id for item in items},
            {"requested-stream"},
        )
        self.assertEqual({item.run_id for item in items}, {"requested-run"})
        self.assertEqual({item.turn_id for item in items}, {"requested-turn"})
        self.assertEqual({item.provider_family for item in items}, {"openai"})
        self.assertEqual(
            items[0].metadata["capabilities"]["backend"],
            "hosted",
        )
        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
            ],
        )

    async def test_accumulation_and_final_text_use_canonical_items(
        self,
    ) -> None:
        stream = TextGenerationSingleStream(
            "answer",
            usage={"output_tokens": 1},
        )

        items = await _collect(stream())
        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.answer_text, "answer")
        self.assertEqual(stream.final_text, "answer")
        self.assertEqual(await stream.to_str(), "answer")
        self.assertEqual(stream.accumulator.answer_text, "answer")
