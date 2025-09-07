from . import TextGenerationVendorModel
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from .....compat import override
from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....tool.manager import ToolManager
from anthropic import AsyncAnthropic
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStopEvent
from contextlib import AsyncExitStack
from diffusers import DiffusionPipeline
from transformers import PreTrainedModel
from typing import AsyncIterator


class AnthropicStream(TextGenerationVendorStream):
    def __init__(self, events: AsyncIterator):
        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            async for event in events:
                if isinstance(event, RawContentBlockDeltaEvent):
                    delta = event.delta
                    value = (
                        delta.text
                        if hasattr(delta, "text")
                        else (
                            delta.partial_json
                            if hasattr(delta, "partial_json")
                            else (
                                delta.thinking
                                if hasattr(delta, "thinking")
                                else None
                            )
                        )
                    )
                    if value is not None:
                        yield value
                elif isinstance(event, RawMessageStopEvent):
                    break

        super().__init__(generator())

    async def __anext__(self) -> Token | TokenDetail | str:
        return await self._generator.__anext__()


class AnthropicClient(TextGenerationVendor):
    _client: AsyncAnthropic
    _exit_stack: AsyncExitStack

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        exit_stack: AsyncExitStack,
    ):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._exit_stack = exit_stack

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
        stream = self._client.messages.stream(
            model=model_id,
            system=system_prompt,
            messages=template_messages,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )
        events = await self._exit_stack.enter_async_context(stream)
        return AnthropicStream(events=events)


class AnthropicModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> TextGenerationVendor | PreTrainedModel | DiffusionPipeline:
        assert self._settings.access_token
        return AnthropicClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
            exit_stack=self._exit_stack,
        )
