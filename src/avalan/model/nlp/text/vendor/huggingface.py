from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from typing import Any, AsyncIterator, cast

from diffusers import DiffusionPipeline
from huggingface_hub import AsyncInferenceClient
from transformers import PreTrainedModel


class HuggingfaceStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[Any]) -> None:
        super().__init__(stream)

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        dynamic_chunk = cast(Any, chunk)
        delta = dynamic_chunk.choices[0].delta
        text = getattr(delta, "content", None) or ""
        return text


class HuggingfaceClient(TextGenerationVendor):
    _client: AsyncInferenceClient

    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncInferenceClient(token=api_key, base_url=base_url)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        settings = settings or GenerationSettings()
        template_messages = cast(
            list[dict[str, Any]], self._template_messages(messages)
        )
        stop_strings = (
            settings.stop_strings
            if isinstance(settings.stop_strings, list)
            else [settings.stop_strings]
            if settings.stop_strings
            else None
        )
        response = await self._client.chat_completion(
            model=model_id,
            messages=template_messages,
            temperature=settings.temperature,
            max_tokens=settings.max_new_tokens,
            top_p=settings.top_p,
            stop=stop_strings,
            stream=use_async_generator,
        )
        if use_async_generator:
            return HuggingfaceStream(cast(AsyncIterator[Any], response))
        else:

            async def single_gen() -> AsyncIterator[Token | TokenDetail | str]:
                non_stream_response = cast(Any, response)
                yield non_stream_response.choices[0].message.content or ""

            return single_gen()


class HuggingfaceModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.access_token
        return HuggingfaceClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
