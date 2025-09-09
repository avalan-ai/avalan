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
    _TEXT_DELTA_EVENTS = {"response.text.delta", "response.output_text.delta"}

    def __init__(self, stream: AsyncStream):
        super().__init__(stream.__aiter__())

    async def __anext__(self) -> Token | TokenDetail | str:
        while True:
            event = await self._generator.__anext__()
            type = getattr(event, "type", None)
            if type and type in self._TEXT_DELTA_EVENTS:
                delta = getattr(event, "delta", None)
                if delta and isinstance(delta, str):
                    return event.delta


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
        kwargs: dict = {
            "extra_headers": {
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            "model": model_id,
            "input": template_messages,
            "stream": use_async_generator,
            "timeout": timeout,
        }
        if settings:
            if settings.max_new_tokens is not None:
                kwargs["max_output_tokens"] = settings.max_new_tokens
            if settings.temperature is not None:
                kwargs["temperature"] = settings.temperature
            if settings.top_p is not None:
                kwargs["top_p"] = settings.top_p
            if settings.stop_strings is not None:
                kwargs["text"] = {"stop": settings.stop_strings}
            if settings.response_format is not None:
                kwargs["response_format"] = settings.response_format
        if tool:
            schemas = OpenAIClient._tool_schemas(tool)
            if schemas:
                kwargs["tools"] = schemas
        client_stream = await self._client.responses.create(**kwargs)

        stream = (
            OpenAIStream(stream=client_stream)
            if use_async_generator
            else TextGenerationSingleStream(
                client_stream.output[0].content[0].text
            )
        )

        return stream

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict] | None:
        schemas = tool.json_schemas()
        return [
            {
                "type": t["type"],
                **t["function"],
                **{
                    "name": TextGenerationVendor.encode_tool_name(
                        t["function"]["name"]
                    )
                }
            }
            for t in tool.json_schemas()
            if t["type"] == "function"
        ] if schemas else None


class OpenAIModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        return OpenAIClient(
            base_url=self._settings.base_url,
            api_key=self._settings.access_token,
        )
