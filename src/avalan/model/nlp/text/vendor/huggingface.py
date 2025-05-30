from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....model import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel
from .....tool.manager import ToolManager
from huggingface_hub import AsyncInferenceClient
from transformers import PreTrainedModel
from typing import AsyncIterator


class HuggingfaceStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator):
        super().__init__(stream.__aiter__())

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None) or ""
        return text


class HuggingfaceClient(TextGenerationVendor):
    _client: AsyncInferenceClient

    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncInferenceClient(token=api_key, base_url=base_url)

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
        settings = settings or GenerationSettings()
        template_messages = self._template_messages(messages)
        response = await self._client.chat_completion(
            model=model_id,
            messages=template_messages,
            temperature=settings.temperature,
            max_tokens=settings.max_new_tokens,
            top_p=settings.top_p,
            stop=settings.stop_strings,
            stream=use_async_generator,
        )
        if use_async_generator:
            return HuggingfaceStream(response)
        else:

            async def single_gen():
                yield response.choices[0].message.content or ""

            return single_gen()


class HuggingfaceModel(TextGenerationVendorModel):
    def _load_model(self) -> PreTrainedModel | TextGenerationVendor:
        assert self._settings.access_token
        return HuggingfaceClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
