from .....entities import (
    GenerationSettings,
    Message,
    Token,
    TokenDetail,
)
from .....model.provider import ProviderFamily
from .....model.stream import TextGenerationSingleStream
from .....tool.manager import ToolManager
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from collections.abc import Mapping
from typing import Any, AsyncGenerator, AsyncIterator, cast

from huggingface_hub import AsyncInferenceClient


class HuggingfaceStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[Any]) -> None:
        async def generator() -> (
            AsyncGenerator[Token | TokenDetail | str, None]
        ):
            terminal_usage: object | None = None
            async for chunk in stream:
                usage = HuggingfaceClient._field(chunk, "usage")
                if usage is not None:
                    terminal_usage = usage
                text = HuggingfaceClient._delta_text(chunk)
                if text is not None:
                    yield text
            self._usage = terminal_usage

        super().__init__(
            generator(),
            provider_family=ProviderFamily.HUGGING_FACE,
            sources=(stream,),
        )


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
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        assert (
            instructions is None
        ), "Hugging Face does not support provider instructions"
        settings = settings or GenerationSettings()
        template_messages = cast(
            list[dict[str, Any]], self._template_messages(messages)
        )
        stop_strings = (
            settings.stop_strings
            if isinstance(settings.stop_strings, list)
            else [settings.stop_strings] if settings.stop_strings else None
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
        non_stream_response = cast(Any, response)
        content = HuggingfaceClient._message_text(non_stream_response)
        return TextGenerationSingleStream(
            content or "",
            provider_family=ProviderFamily.HUGGING_FACE,
            usage=HuggingfaceClient._field(non_stream_response, "usage"),
        )

    @staticmethod
    def _field(value: object, name: str) -> object | None:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _delta_text(chunk: object) -> str | None:
        choices = HuggingfaceClient._field(chunk, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        delta = HuggingfaceClient._field(choices[0], "delta")
        content = HuggingfaceClient._field(delta, "content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _message_text(response: object) -> str | None:
        choices = HuggingfaceClient._field(response, "choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = HuggingfaceClient._field(choices[0], "message")
        content = HuggingfaceClient._field(message, "content")
        return content if isinstance(content, str) else None


class HuggingfaceModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.access_token
        return HuggingfaceClient(
            api_key=self._settings.access_token,
            base_url=self._settings.base_url,
        )
