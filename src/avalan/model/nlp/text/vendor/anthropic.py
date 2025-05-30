from anthropic import AsyncAnthropic
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStopEvent
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
from transformers import PreTrainedModel
from typing import AsyncIterator


class AnthropicStream(TextGenerationVendorStream):
    _stream: AsyncIterator

    def __init__(self, stream: AsyncIterator):
        super().__init__(stream.__aiter__())

    async def __anext__(self) -> Token | TokenDetail | str:
        # We may handle multiple iterations before yielding because
        # Anthropic triggers multiple events besides the deltas
        while True:
            event = await self._generator.__anext__()
            if isinstance(event, RawContentBlockDeltaEvent):
                delta = event.delta
                value = (
                    delta.text
                    if hasattr(delta, "text")
                    else delta.partial_json
                    if hasattr(delta, "partial_json")
                    else delta.thinking
                    if hasattr(delta, "thinking")
                    else None
                )
                if value is not None:
                    return value
            elif isinstance(event, RawMessageStopEvent):
                raise StopAsyncIteration


class AnthropicClient(TextGenerationVendor):
    _client: AsyncAnthropic

    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

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
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(messages, ["system"])
        stream = await self._client.messages.create(
            model=model_id,
            system=system_prompt,
            messages=template_messages,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            stream=use_async_generator,
        )
        return AnthropicStream(stream=stream)


class AnthropicModel(TextGenerationVendorModel):
    def _load_model(self) -> TextGenerationVendor | PreTrainedModel:
        assert self._settings.access_token
        return AnthropicClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
