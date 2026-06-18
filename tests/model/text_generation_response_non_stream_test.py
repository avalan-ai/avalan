from logging import getLogger
from unittest import IsolatedAsyncioTestCase

from avalan.entities import (
    GenerationSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    StreamValidationError,
    TextGenerationSingleStream,
)
from avalan.task.usage import UsageSource, usage_observation_from_response


class TextGenerationResponseNonStreamTestCase(IsolatedAsyncioTestCase):
    @staticmethod
    def _non_stream_response(result: object) -> TextGenerationResponse:
        settings = GenerationSettings()
        return TextGenerationResponse(
            lambda **_: result,
            logger=getLogger("response-legacy-result"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )

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

    async def test_to_json_preserves_returned_single_stream_usage(
        self,
    ) -> None:
        settings = GenerationSettings()
        usage = {
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        }
        response = TextGenerationResponse(
            lambda **_: TextGenerationSingleStream(
                '{"ok": true}',
                provider_family="openai",
                usage=usage,
            ),
            logger=getLogger("response-single-stream-json"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )

        self.assertEqual(await response.to_json(), '{"ok": true}')
        observation = usage_observation_from_response(response)

        self.assertEqual(response.usage, usage)
        self.assertEqual(response.provider_family, "openai")
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.totals.input_tokens, 2)
        self.assertEqual(observation.totals.output_tokens, 1)
        self.assertEqual(observation.totals.total_tokens, 3)
        self.assertEqual(observation.metadata, {"provider_family": "openai"})

    async def test_to_json_drops_malformed_returned_single_stream_usage(
        self,
    ) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: TextGenerationSingleStream(
                '{"ok": false}',
                provider_family="openai",
                usage={
                    "input_tokens": "private prompt",
                    "output_tokens": -1,
                    "total_tokens": True,
                },
            ),
            logger=getLogger("response-single-stream-json-invalid-usage"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )

        self.assertEqual(await response.to_json(), '{"ok": false}')
        observation = usage_observation_from_response(response)

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.source, UsageSource.ESTIMATED)
        self.assertEqual(observation.totals.input_tokens, 0)
        self.assertEqual(observation.totals.output_tokens, 13)
        self.assertIsNone(observation.totals.total_tokens)
        self.assertEqual(observation.metadata, {"provider_family": "openai"})

    async def test_prefetch_rejects_token_instances(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: Token(token="token-result"),
            logger=getLogger("response-token"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            response._ensure_non_stream_prefetched()

    async def test_prefetch_rejects_token_subclasses(self) -> None:
        class CustomToken(Token):
            pass

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: CustomToken(token="token-result"),
            logger=getLogger("response-token-subclass"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            response._ensure_non_stream_prefetched()

    async def test_prefetch_rejects_token_detail_instances(self) -> None:
        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: TokenDetail(
                id=7,
                token="detailed-result",
                probability=0.5,
                step=1,
            ),
            logger=getLogger("response-token-detail"),
            generation_settings=settings,
            settings=settings,
            use_async_generator=False,
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            response._ensure_non_stream_prefetched()

    async def test_non_stream_rejects_reasoning_and_tool_tokens(
        self,
    ) -> None:
        cases = (
            ("reasoning", ReasoningToken("private reasoning")),
            ("tool", ToolCallToken(token='{"name": "calculator"}')),
        )

        for name, result in cases:
            with self.subTest(name=name, surface="prefetch"):
                response = self._non_stream_response(result)
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported legacy SDK response stream item",
                ):
                    response._ensure_non_stream_prefetched()

            with self.subTest(name=name, surface="str"):
                response = self._non_stream_response(result)
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported legacy SDK response stream item",
                ):
                    str(response)

            with self.subTest(name=name, surface="to_str"):
                response = self._non_stream_response(result)
                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported legacy SDK response stream item",
                ):
                    await response.to_str()

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
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await response.__anext__()
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported legacy SDK response stream item",
        ):
            await response.to_str()
