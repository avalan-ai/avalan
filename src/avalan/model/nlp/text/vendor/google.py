from .....entities import (
    GenerationSettings,
    Message,
    MessageRole,
    Token,
    TokenDetail,
)
from .....tool.manager import ToolManager
from ....message import TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import TextGenerationVendorModel

from typing import Any, AsyncGenerator, AsyncIterator, cast

from diffusers import DiffusionPipeline
from google.genai import Client
from google.genai.types import GenerateContentResponse
from transformers import PreTrainedModel


class GoogleStream(TextGenerationVendorStream):
    def __init__(self, stream: AsyncIterator[GenerateContentResponse]):
        async def generator() -> (
            AsyncGenerator[Token | TokenDetail | str, None]
        ):
            async for chunk in stream:
                text = chunk.text
                if isinstance(text, str):
                    yield text

        super().__init__(generator())


class GoogleClient(TextGenerationVendor):
    _client: Client

    def __init__(self, api_key: str):
        self._client = Client(api_key=api_key)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> AsyncIterator[Token | TokenDetail | str]:
        contents = self._template_messages(messages, ["system", "tool"])
        kwargs: dict[str, Any] = {
            "model": model_id,
            "contents": cast(Any, contents),
        }
        config = self._config(messages, settings)
        if config:
            kwargs["config"] = config

        if use_async_generator:
            stream = await self._client.aio.models.generate_content_stream(
                **kwargs,
            )
            return GoogleStream(stream=stream.__aiter__())
        else:
            response = await self._client.aio.models.generate_content(
                **kwargs,
            )

            async def single_gen() -> (
                AsyncGenerator[Token | TokenDetail | str, None]
            ):
                yield response.text or ""

            return single_gen()

    def _config(
        self,
        messages: list[Message],
        settings: GenerationSettings | None,
    ) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        system_prompt = self._system_prompt(messages)
        if system_prompt:
            config["system_instruction"] = system_prompt
        if settings is None:
            return config or None
        if settings.max_new_tokens is not None:
            config["max_output_tokens"] = settings.max_new_tokens
        if settings.temperature is not None:
            config["temperature"] = settings.temperature
        if settings.top_p is not None:
            config["top_p"] = settings.top_p
        if settings.top_k is not None:
            config["top_k"] = settings.top_k
        if settings.stop_strings is not None:
            stop_sequences = (
                [settings.stop_strings]
                if isinstance(settings.stop_strings, str)
                else settings.stop_strings
            )
            config["stop_sequences"] = stop_sequences
        return config or None

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[dict[str, Any]]:
        templated = cast(
            list[dict[str, Any]],
            super()._template_messages(messages, exclude_roles),
        )
        output: list[dict[str, Any]] = []
        for message in templated:
            content = message.get("content")
            output.append(
                {
                    "role": self._message_role(cast(str, message["role"])),
                    "parts": self._parts(content),
                }
            )
        return output

    @staticmethod
    def _message_role(role: str) -> str:
        if role == str(MessageRole.ASSISTANT):
            return "model"
        if role == str(MessageRole.DEVELOPER):
            return str(MessageRole.USER)
        return role

    @staticmethod
    def _parts(content: object) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, list):
            return [
                GoogleClient._part(block)
                for block in content
                if isinstance(block, dict)
            ]
        if isinstance(content, dict):
            return [GoogleClient._part(content)]
        return [{"text": str(content)}]

    @staticmethod
    def _part(block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        match block_type:
            case "file":
                file = block.get("file")
                assert isinstance(file, dict), "File blocks require file data"
                return GoogleClient._file_part(
                    file, default_mime_type="application/pdf"
                )
            case "image_url":
                image = block.get("image_url")
                assert isinstance(
                    image, dict
                ), "Image blocks require image data"
                return GoogleClient._file_part(
                    image, default_mime_type="image/png"
                )
            case "text":
                text = block.get("text")
                assert isinstance(text, str), "Text blocks require text"
                return {"text": text}
            case _:
                return {"text": str(block)}

    @staticmethod
    def _file_part(
        file: dict[str, Any], *, default_mime_type: str
    ) -> dict[str, Any]:
        mime_type_value = file.get("mime_type")
        mime_type = (
            mime_type_value
            if isinstance(mime_type_value, str) and mime_type_value
            else default_mime_type
        )
        display_name = GoogleClient._display_name(file)
        file_uri = GoogleClient._file_uri(file)
        if file_uri:
            file_data: dict[str, Any] = {
                "file_uri": file_uri,
                "mime_type": mime_type,
            }
            if display_name:
                file_data["display_name"] = display_name
            return {"file_data": file_data}

        data = file.get("file_data", file.get("data"))
        assert (
            data is not None
        ), "Google file blocks require file data or file URI"
        inline_data: dict[str, Any] = {
            "data": data,
            "mime_type": mime_type,
        }
        if display_name:
            inline_data["display_name"] = display_name
        return {"inline_data": inline_data}

    @staticmethod
    def _display_name(file: dict[str, Any]) -> str | None:
        for key in ("filename", "title"):
            value = file.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _file_uri(file: dict[str, Any]) -> str | None:
        for key in ("file_url", "url", "uri", "file_id"):
            value = file.get(key)
            if isinstance(value, str) and value:
                return value
        return None


class GoogleModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.access_token
        return GoogleClient(api_key=self._settings.access_token)
