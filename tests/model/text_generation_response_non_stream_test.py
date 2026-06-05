from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import GenerationSettings, Token
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import TextGenerationSingleStream
from avalan.task.usage import usage_observation_from_response


class TextGenerationResponseNonStreamTestCase(IsolatedAsyncioTestCase):
    async def test_str_prefetches_single_stream(self) -> None:
        settings = GenerationSettings()
        usage = {"input_tokens": 2}
        stream = TextGenerationSingleStream(
            "hello",
            provider_family="openai",
            usage=usage,
        )
        response = TextGenerationResponse(
            stream,
            logger=getLogger("response"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(response.usage, usage)
        self.assertEqual(response.provider_family, "openai")
        self.assertFalse(response.is_async_generator)
        self.assertEqual(observation.metadata, {"provider_family": "openai"})
        self.assertEqual(str(response), "hello")
        self.assertEqual(response.output_token_count, len("hello"))
        response._ensure_non_stream_prefetched()

    async def test_to_str_handles_prefilled_buffer(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: Token(token="value"),
            logger=getLogger("response"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        response._buffer.write("prefilled")
        text = await response.to_str()
        self.assertEqual(text, "")

    async def test_prefetch_handles_token_instances(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: Token(token="token-result"),
            logger=getLogger("response-token"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        response._ensure_non_stream_prefetched()
        self.assertEqual(response._prefetched_text, "token-result")
        self.assertEqual(response.output_token_count, len("token-result"))

    async def test_str_converts_generic_result(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: ["generic"],
            logger=getLogger("response-generic"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        self.assertIsNone(response.provider_family)
        self.assertIn("generic", str(response))

    async def test_streaming_str_and_prefetch_guard(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: TextGenerationSingleStream("stream"),
            logger=getLogger("response-stream"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=True,
        )
        response._prefetched_text = "cached"
        response._ensure_non_stream_prefetched()
        self.assertIn("TextGenerationResponse", str(response))

    async def test_streaming_string_output_is_not_replayed_in_to_str(
        self,
    ) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: "hi",
            logger=getLogger("response-stream-string"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=True,
        )
        response.__aiter__()
        first = await response.__anext__()
        self.assertEqual(first, "hi")
        self.assertEqual(await response.to_str(), "hi")
