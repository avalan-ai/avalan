from .....entities import GenerationSettings, Message, Token, TokenDetail
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from typing import Any, AsyncGenerator, AsyncIterator, cast

from diffusers import DiffusionPipeline
from google.genai import Client
from google.genai.types import GenerateContentResponse
from transformers import PreTrainedModel


class GoogleStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[GenerateContentResponse]):
        async def generator() -> (
            AsyncGenerator[Token | TokenDetail | str, None]
        ):
            async for chunk in stream:
                text = chunk.text
                if isinstance(text, str):
                    yield text

        super().__init__(generator())


class GoogleClient(TextGenerationVendor):
    _client: Client

    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        contents = [
            m.content if isinstance(m.content, str) else "" for m in messages
        ]

        if use_async_generator:
            stream = await self._client.aio.models.generate_content_stream(
                model=model_id,
                contents=cast(Any, contents),
            )
            return GoogleStream(stream=stream.__aiter__())
        else:
            response = await self._client.aio.models.generate_content(
                model=model_id,
                contents=cast(Any, contents),
            )

            async def single_gen() -> (
                AsyncGenerator[Token | TokenDetail | str, None]
            ):
                yield response.text or ""

            return single_gen()


class GoogleModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.access_token
        return GoogleClient(api_key=self._settings.access_token)
