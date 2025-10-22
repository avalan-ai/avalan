from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
    ReasoningToken,
)
from avalan.model.response.text import TextGenerationResponse


class TextGenerationStopOnLimitTestCase(IsolatedAsyncioTestCase):
    async def test_stop_on_limit(self) -> None:
        async def gen():
            for t in ("<think>", "a", "b", "</think>"):
                yield t

        settings = GenerationSettings(
            reasoning=ReasoningSettings(
                max_new_tokens=2,
                stop_on_max_new_tokens=True,
            )
        )
        resp = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        it = resp.__aiter__()
        first = await it.__anext__()
        second = await it.__anext__()
        self.assertIsInstance(first, ReasoningToken)
        self.assertIsInstance(second, ReasoningToken)
        with self.assertRaises(StopAsyncIteration):
            await it.__anext__()
