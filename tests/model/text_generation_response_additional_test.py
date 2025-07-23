from dataclasses import dataclass
from unittest import IsolatedAsyncioTestCase
from avalan.model.response.text import TextGenerationResponse


@dataclass
class Example:
    value: str


class TextGenerationResponseAdditionalTestCase(IsolatedAsyncioTestCase):
    async def test_to_entity(self):
        resp = TextGenerationResponse(
            lambda: '{"value": "ok"}',
            use_async_generator=False,
        )
        result = await resp.to(Example)
        self.assertEqual(result, Example(value="ok"))

    async def test_disable_reasoning_parser(self):
        async def gen():
            for t in ("<think>", "a", "</think>"):
                yield t

        resp = TextGenerationResponse(
            lambda: gen(),
            use_async_generator=True,
            enable_reasoning_parser=False,
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(tokens, ["<think>", "a", "</think>"])
