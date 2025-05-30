from google.genai import Client
from google.genai.types import GenerateContentResponse
from .....compat import override
from .....entities import GenerationSettings, Message, Token, TokenDetail
from .....model import TextGenerationVendor, TextGenerationVendorStream
from .....model.nlp.text.vendor import TextGenerationVendorModel
from .....tool.manager import ToolManager
from transformers import PreTrainedModel
from typing import AsyncIterator


class GoogleStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[GenerateContentResponse]):
        super().__init__(stream)

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        return chunk.text


class GoogleClient(TextGenerationVendor):
    _client: Client

    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    @override
    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        contents = [m.content for m in messages]

        if use_async_generator:
            stream = await self._client.aio.models.generate_content_stream(
                model=model_id,
                contents=contents,
            )
            return GoogleStream(stream=stream.__aiter__())
        else:
            response = await self._client.aio.models.generate_content(
                model=model_id,
                contents=contents,
            )

            async def single_gen():
                yield response.text

            return single_gen()


class GoogleModel(TextGenerationVendorModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert self._settings.access_token
        return GoogleClient(api_key=self._settings.access_token)
