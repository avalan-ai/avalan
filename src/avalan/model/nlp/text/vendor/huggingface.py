from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from typing import Any, AsyncIterator

from diffusers import DiffusionPipeline
from huggingface_hub import AsyncInferenceClient
from transformers import PreTrainedModel


class HuggingfaceStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator) -> None:  # type: ignore[type-arg]
        super().__init__(stream.__aiter__())  # type: ignore[arg-type]

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        delta = chunk.choices[0].delta
        text: str = getattr(delta, "content", None) or ""
        return text


class HuggingfaceClient(TextGenerationVendor):
    _client: AsyncInferenceClient

    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncInferenceClient(token=api_key, base_url=base_url)

    @override
    async def __call__(  # type: ignore[override]
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        settings = settings or GenerationSettings()
        template_messages: list[dict[Any, Any]] = [
            {"role": m["role"], "content": m["content"]}
            for m in self._template_messages(messages)
        ]
        stop: list[str] | None = None
        if settings.stop_strings:
            stop = (
                [settings.stop_strings]
                if isinstance(settings.stop_strings, str)
                else settings.stop_strings
            )
        response = await self._client.chat_completion(
            model=model_id,
            messages=template_messages,
            temperature=settings.temperature,
            max_tokens=settings.max_new_tokens,
            top_p=settings.top_p,
            stop=stop,
            stream=use_async_generator,
        )
        if use_async_generator:
            return HuggingfaceStream(response.__aiter__())  # type: ignore[arg-type, union-attr]
        else:

            async def single_gen() -> AsyncIterator[Token | TokenDetail | str]:
                yield response.choices[0].message.content or ""  # type: ignore[union-attr]

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
