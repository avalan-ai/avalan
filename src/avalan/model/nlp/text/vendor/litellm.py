from .....entities import GenerationSettings, Message, Token, TokenDetail
from .....model.stream import TextGenerationSingleStream
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from collections.abc import Mapping
from typing import Any, AsyncGenerator, AsyncIterator

import litellm


class LiteLLMStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[Any]) -> None:
        async def generator() -> (
            AsyncGenerator[Token | TokenDetail | str, None]
        ):
            terminal_usage: object | None = None
            async for chunk in stream:
                usage = LiteLLMClient._field(chunk, "usage")
                if usage is not None:
                    terminal_usage = usage
                text = LiteLLMClient._delta_text(chunk)
                if text is not None:
                    yield text
            self._usage = terminal_usage

        super().__init__(generator(), provider_family="openai_compatible")


class LiteLLMClient(TextGenerationVendor):
    _api_key: str | None
    _base_url: str | None

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None
    ):
        self._api_key = api_key
        self._base_url = base_url or "http://localhost:4000"

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        assert (
            instructions is None
        ), "LiteLLM does not support provider instructions"
        template_messages = self._template_messages(messages)
        kwargs: dict[str, Any] = dict(
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

        return TextGenerationSingleStream(
            LiteLLMClient._message_text(result) or "",
            provider_family="openai_compatible",
            usage=LiteLLMClient._field(result, "usage"),
        )

    @staticmethod
    def _field(value: object, name: str) -> object | None:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _delta_text(chunk: object) -> str | None:
        choices = LiteLLMClient._field(chunk, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        delta = LiteLLMClient._field(choices[0], "delta")
        content = LiteLLMClient._field(delta, "content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _message_text(response: object) -> str | None:
        choices = LiteLLMClient._field(response, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = LiteLLMClient._field(choices[0], "message")
        content = LiteLLMClient._field(message, "content")
        return content if isinstance(content, str) else None


class LiteLLMModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return LiteLLMClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
