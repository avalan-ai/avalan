from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningSettings,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import StreamValidationError


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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await it.__anext__()
