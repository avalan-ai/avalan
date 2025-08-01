from .....model.stream import TextGenerationSingleStream
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel
from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....tool.manager import ToolManager
from diffusers import DiffusionPipeline
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
        timeout: int | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str] | TextGenerationSingleStream:
        template_messages = self._template_messages(messages)
        client_stream = await self._client.chat.completions.create(
            extra_headers={
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            model=model_id,
            messages=template_messages,
            stream=use_async_generator,
            timeout=timeout,
            response_format=(
                settings.response_format
                if settings and settings.response_format
                else None
            ),
        )

        stream = (
            OpenAIStream(stream=client_stream)
            if use_async_generator
            else TextGenerationSingleStream(
                client_stream.choices[0].message.content
            )
        )

        return stream


class OpenAIModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        return OpenAIClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )
