from dataclasses import dataclass
from unittest import IsolatedAsyncioTestCase
from avalan.model.response.text import TextGenerationResponse
from avalan.entities import GenerationSettings, ReasoningSettings


@dataclass
class Example:
    value: str


class TextGenerationResponseAdditionalTestCase(IsolatedAsyncioTestCase):
    async def test_to_entity(self):
        settings = GenerationSettings()
        resp = TextGenerationResponse(
            lambda **_: '{"value": "ok"}',
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
            use_async_generator=True,
            generation_settings=gs,
            settings=gs,
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(tokens, ["<think>", "a", "</think>"])
