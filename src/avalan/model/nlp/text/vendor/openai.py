from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....model import TextGenerationVendor, TextGenerationVendorStream
from .....model.nlp.text.vendor import TextGenerationVendorModel
from .....tool.manager import ToolManager
from openai import AsyncOpenAI, AsyncStream
from transformers import PreTrainedModel
from typing import AsyncIterator


class OpenAIStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncStream):
        super().__init__(stream.__aiter__())

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        text = chunk.choices[0].delta.content or ""
        return text


class OpenAIClient(TextGenerationVendor):
    _client: AsyncOpenAI

    def __init__(self, api_key: str, base_url: str | None):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

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
        template_messages = self._template_messages(messages)
        client_stream = await self._client.chat.completions.create(
            model=model_id,
            messages=template_messages,
            stream=use_async_generator,
        )
        return OpenAIStream(stream=client_stream)


class OpenAIModel(TextGenerationVendorModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert self._settings.base_url or self._settings.access_token
        return OpenAIClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )
