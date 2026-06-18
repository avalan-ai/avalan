from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import GenerationSettings, Message, MessageRole
from avalan.model.response import InvalidJsonResponseException
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    TextGenerationSingleStream,
    TextGenerationStream,
    accumulate_canonical_stream_items,
)
from avalan.model.vendor import (
    TextGenerationVendor,
    TextGenerationVendorStream,
)


def _canonical_answer_items(text: str) -> tuple[CanonicalStreamItem, ...]:
    return (
        CanonicalStreamItem(
            stream_session_id="model-init-stream",
            run_id="model-init-run",
            turn_id="model-init-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        ),
        CanonicalStreamItem(
            stream_session_id="model-init-stream",
            run_id="model-init-run",
            turn_id="model-init-turn",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=text,
        ),
        CanonicalStreamItem(
            stream_session_id="model-init-stream",
            run_id="model-init-run",
            turn_id="model-init-turn",
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        ),
        CanonicalStreamItem(
            stream_session_id="model-init-stream",
            run_id="model-init-run",
            turn_id="model-init-turn",
            sequence=3,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )


async def _canonical_answer_gen(text: str):
    for item in _canonical_answer_items(text):
        yield item


class TextGenerationResponseTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_and_callback(self):
        consumed = False

        async def done():
            nonlocal consumed
            consumed = True

        resp = TextGenerationResponse(
            lambda **_: _canonical_answer_gen("hi"),
            logger=getLogger(),
            use_async_generator=True,
            inputs={"input_ids": [[1, 2, 3]]},
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        resp.add_done_callback(done)

        tokens = []
        async for token in resp:
            tokens.append(token)
        self.assertEqual(tokens, list(_canonical_answer_items("hi")))
        self.assertTrue(consumed)
        self.assertEqual(resp.input_token_count, 3)
        self.assertEqual(resp.output_token_count, 4)

    async def test_to_json_valid_and_invalid(self):
        resp = TextGenerationResponse(
            lambda **_: 'prefix ```json\n{"a": 1}\n``` suffix',
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        self.assertEqual(await resp.to_json(), '{"a": 1}')

        invalid = TextGenerationResponse(
            lambda **_: '```json {"a": } ```',
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        with self.assertRaises(InvalidJsonResponseException):
            await invalid.to_json()

    async def test_output_token_count_to_str(self):
        resp = TextGenerationResponse(
            lambda **_: _canonical_answer_gen("hi"),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        self.assertEqual(await resp.to_str(), "hi")
        self.assertEqual(resp.output_token_count, 4)

    async def test_output_token_count_to_json(self):
        resp = TextGenerationResponse(
            lambda **_: _canonical_answer_gen('{"a":1}'),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        self.assertEqual(await resp.to_json(), '{"a":1}')
        self.assertEqual(resp.output_token_count, 4)


class StreamVendorTestCase(IsolatedAsyncioTestCase):
    async def test_stream_base_and_vendor_helpers(self):
        async def agen():
            yield "x"

        class DummyStream(TextGenerationStream):
            def __init__(self):
                self._generator = agen()

            def __call__(self, *args, **kwargs):
                return super().__call__(*args, **kwargs)

            async def __anext__(self):
                return await super().__anext__()

        stream = DummyStream()
        self.assertIs(stream.__aiter__(), stream)
        with self.assertRaises(NotImplementedError):
            stream()
        with self.assertRaises(NotImplementedError):
            await stream.__anext__()

        class DummyVendor(TextGenerationVendor):
            async def __call__(self, *args, **kwargs):
                return await super().__call__(*args, **kwargs)

        vendor = DummyVendor()
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.DEVELOPER, content="dev"),
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content="r"),
            Message(role=MessageRole.TOOL, content="t"),
        ]
        with self.assertRaises(NotImplementedError):
            await vendor("m", messages)
        self.assertEqual(vendor._system_prompt(messages), "sys")
        tmpl = vendor._template_messages(
            messages, exclude_roles=[MessageRole.ASSISTANT]
        )
        self.assertEqual(
            tmpl,
            [
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "dev"},
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "t"},
            ],
        )

    async def test_vendor_stream_iteration(self):
        async def agen():
            for item in TextGenerationSingleStream("ab").canonical_items:
                yield item

        stream = TextGenerationVendorStream(agen())
        it = stream()
        self.assertIsNot(it, stream)
        collected = [item async for item in it]

        accumulator = accumulate_canonical_stream_items(collected)

        self.assertEqual(accumulator.answer_text, "ab")

    async def test_vendor_stream_exposes_usage(self):
        usage = {"input_tokens": 1}

        async def agen():
            yield "x"

        class DummyVendorStream(TextGenerationVendorStream):
            async def __anext__(self):
                return await self._generator.__anext__()

        stream = DummyVendorStream(
            agen(),
            provider_family="openai_compatible",
            usage=usage,
        )

        self.assertEqual(stream.provider_family, "openai_compatible")
        self.assertEqual(stream.usage, usage)
