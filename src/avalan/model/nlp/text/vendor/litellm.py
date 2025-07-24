from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel
from .....compat import override
from .....entities import GenerationSettings, Message, Token, TokenDetail
from .....tool.manager import ToolManager
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
from typing import AsyncIterator
import litellm


class LiteLLMStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator):
        super().__init__(stream.__aiter__())

    async def __anext__(self) -> Token | TokenDetail | str:
        chunk = await self._generator.__anext__()
        choice = None
        if isinstance(chunk, dict):
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            text = delta.get("content", "")
        else:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", "") if delta else ""
        return text


class LiteLLMClient(TextGenerationVendor):
    _api_key: str | None
    _base_url: str | None

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None
    ):
        self._api_key = api_key
        self._base_url = base_url or "http://localhost:4000"

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
        kwargs = dict(
            model=model_id,
            messages=template_messages,
            api_key=self._api_key,
            stream=use_async_generator,
        )
        if self._base_url:
            kwargs["api_base"] = self._base_url
        result = await litellm.acompletion(**kwargs)
        if use_async_generator:
            return LiteLLMStream(result)

        async def single_gen():
            if isinstance(result, dict):
                yield result["choices"][0]["message"]["content"]
            else:
                yield result.choices[0].message.content or ""

        return single_gen()


class LiteLLMModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return LiteLLMClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
