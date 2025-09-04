from avalan.model.response.text import TextGenerationResponse
from avalan.entities import GenerationSettings, Message, MessageRole
from avalan.model.response import InvalidJsonResponseException
from avalan.model.stream import TextGenerationStream
from avalan.model.vendor import (
    TextGenerationVendor,
    TextGenerationVendorStream,
)
from logging import getLogger
from unittest import IsolatedAsyncioTestCase


class TextGenerationResponseTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_and_callback(self):
        async def gen():
            for ch in "hi":
                yield ch

        consumed = False

        async def done():
            nonlocal consumed
            consumed = True

        resp = TextGenerationResponse(
            lambda **_: gen(),
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
        self.assertEqual(tokens, ["h", "i"])
        self.assertTrue(consumed)
        self.assertEqual(resp.input_token_count, 3)
        self.assertEqual(resp.output_token_count, 2)

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
        async def gen():
            for ch in "hi":
                yield ch

        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        self.assertEqual(await resp.to_str(), "hi")
        self.assertEqual(resp.output_token_count, 2)

    async def test_output_token_count_to_json(self):
        async def gen():
            for ch in '{"a":1}':
                yield ch

        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        self.assertEqual(await resp.to_json(), '{"a":1}')
        self.assertEqual(resp.output_token_count, len('{"a":1}'))


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
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "t"},
            ],
        )

    async def test_vendor_stream_iteration(self):
        async def agen():
            for ch in "ab":
                yield ch

        class DummyVendorStream(TextGenerationVendorStream):
            async def __anext__(self):
                return await self._generator.__anext__()

        stream = DummyVendorStream(agen())
        it = stream()
        self.assertIs(it, stream)
        collected = []
        async for token in it:
            collected.append(token)
        self.assertEqual(collected, ["a", "b"])
