from .....entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningEffort,
)
from .....model.provider import ProviderFamily
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    TextGenerationSingleStream,
    TextGenerationStream,
)
from .....tool.manager import ToolManager
from .....types import LooseJsonValue
from ....message import TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from collections.abc import Mapping
from typing import Any, AsyncIterator, cast

from google.genai import Client
from google.genai.types import GenerateContentResponse


class GoogleStream(TextGenerationVendorStream):
    _stream: AsyncIterator[GenerateContentResponse]

    def __init__(self, stream: AsyncIterator[GenerateContentResponse]):
        self._stream = stream

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            async for item in self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            ):
                yield item

        super().__init__(
            generator(),
            provider_family=ProviderFamily.GOOGLE,
            sources=(stream,),
        )

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        assert self._generator
        return self._generator

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

    def _cleanup_sources(self) -> tuple[object, ...]:
        return self._stream_sources

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: ProviderFamily | str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=ProviderFamily.GOOGLE,
                supports_usage=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        terminal_usage: object | None = None
        terminal_usage_payload: LooseJsonValue | None = None
        async for chunk in self._stream:
            provider_payload = self._provider_payload(chunk)
            usage = GoogleClient._field(chunk, "usage_metadata")
            if usage is None:
                usage = GoogleClient._field(chunk, "usageMetadata")
            if usage is not None:
                terminal_usage = usage
                terminal_usage_payload = provider_payload
            text = GoogleClient._field(chunk, "text")
            if isinstance(text, str):
                yield StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=text,
                    provider_payload=provider_payload,
                    provider_event_type="generate_content.delta",
                )
        if terminal_usage is not None:
            self._usage = terminal_usage
            yield StreamProviderEvent(
                kind=StreamItemKind.USAGE_COMPLETED,
                usage=cast(LooseJsonValue, terminal_usage),
                provider_payload=terminal_usage_payload,
                provider_event_type="generate_content.usage",
            )

    @staticmethod
    def _provider_payload(chunk: object) -> LooseJsonValue | None:
        if isinstance(chunk, Mapping):
            return dict(chunk)
        model_dump = getattr(chunk, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
            if isinstance(payload, Mapping):
                return dict(payload)
        return None


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
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        assert (
            instructions is None
        ), "Google does not support provider instructions"
        contents = self._template_messages(messages, ["system", "tool"])
        kwargs: dict[str, Any] = {
            "model": model_id,
            "contents": cast(Any, contents),
        }
        config = self._config(model_id, messages, settings)
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

            return TextGenerationSingleStream(
                response.text or "",
                provider_family=ProviderFamily.GOOGLE,
                usage=GoogleClient._field(response, "usage_metadata")
                or GoogleClient._field(response, "usageMetadata"),
            )

    def _config(
        self,
        model_id: str,
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
        thinking_config = GoogleClient._thinking_config(model_id, settings)
        if thinking_config:
            config["thinking_config"] = thinking_config
        return config or None

    @staticmethod
    def _thinking_config(
        model_id: str,
        settings: GenerationSettings,
    ) -> dict[str, Any] | None:
        effort = settings.reasoning.effort
        if effort is None or "gemini-3" not in model_id.lower():
            return None

        match effort:
            case ReasoningEffort.NONE:
                thinking_level = ReasoningEffort.MINIMAL.value
            case ReasoningEffort.XHIGH | ReasoningEffort.MAX:
                thinking_level = ReasoningEffort.HIGH.value
            case _:
                thinking_level = effort.value

        return {"thinking_level": thinking_level}

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
    def _field(value: object, attribute: str) -> object | None:
        if isinstance(value, dict):
            return value.get(attribute)
        return getattr(value, attribute, None)

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
