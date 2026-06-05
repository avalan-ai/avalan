from dataclasses import dataclass
from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import GenerationSettings, ReasoningSettings
from avalan.model.response.text import TextGenerationResponse
from avalan.model.vendor import TextGenerationVendorStream


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

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(tokens, ["<think>", "a", "</think>"])

    async def test_usage_reads_completed_output_stream(self):
        usage = {"input_tokens": 2, "output_tokens": 1}

        class UsageStream(TextGenerationVendorStream):
            def __init__(self) -> None:
                async def gen():
                    yield "ok"
                    self._usage = usage

                super().__init__(gen())

            async def __anext__(self):
                return await self._generator.__anext__()

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
