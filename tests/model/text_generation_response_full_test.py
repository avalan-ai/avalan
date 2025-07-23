from avalan.model.response.text import TextGenerationResponse
from unittest import IsolatedAsyncioTestCase


async def _gen():
    yield "a"


class TextGenerationResponseFullCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_parser_queue_precedence(self):
        resp = TextGenerationResponse(lambda: _gen(), use_async_generator=True)
        resp.__aiter__()
        resp._parser_queue.put("x")
        first = await resp.__anext__()
        second = await resp.__anext__()
        self.assertEqual(first, "x")
        self.assertEqual(second, "a")
