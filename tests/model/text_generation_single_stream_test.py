from unittest import IsolatedAsyncioTestCase

from avalan.model.stream import TextGenerationSingleStream


class TextGenerationSingleStreamTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_and_reset(self) -> None:
        stream = TextGenerationSingleStream("hello")
        iterator = stream.__aiter__()
        self.assertIs(iterator, stream)
        self.assertEqual(await iterator.__anext__(), "hello")
        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()

        iterator2 = stream()
        self.assertIs(iterator2, stream)
        self.assertEqual(await iterator2.__anext__(), "hello")
        with self.assertRaises(StopAsyncIteration):
            await iterator2.__anext__()

    async def test_async_for_and_property(self) -> None:
        stream = TextGenerationSingleStream("foo")
        tokens = []
        async for token in stream():
            tokens.append(token)
        self.assertEqual(tokens, ["foo"])
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()
        iterator = stream.__aiter__()
        self.assertEqual(stream.content, "foo")
        self.assertEqual(await iterator.__anext__(), "foo")
