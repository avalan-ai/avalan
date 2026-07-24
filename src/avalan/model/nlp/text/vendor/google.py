from .....entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningEffort,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallResult,
)
from .....model.provider import ProviderFamily
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    TextGenerationNonStreamResult,
    TextGenerationNonStreamToolCall,
    TextGenerationSingleStream,
    TextGenerationStream,
)
from .....types import LooseJsonValue
from .....utils import to_json, tool_call_diagnostic_payload
from ....capability import (
    CorrelatedCapabilityResult,
    ModelCapabilityCatalog,
    ProviderCapabilityCall,
    TaskInputCapabilityCall,
)
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


def _mutable_provider_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _mutable_provider_json(item) for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_mutable_provider_json(item) for item in value]
    return value


class GoogleStream(TextGenerationVendorStream):
    _stream: AsyncIterator[GenerateContentResponse]
    _capability_catalog: ModelCapabilityCatalog | None

    def __init__(
        self,
        stream: AsyncIterator[GenerateContentResponse],
        *,
        capability: ModelCapabilityCatalog | None = None,
    ):
        self._stream = stream
        self._capability_catalog = capability

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
                supports_tool_calls=(
                    self._capability_catalog is not None
                    and not self._capability_catalog.project(
                        ProviderFamily.GOOGLE
                    ).is_empty
                ),
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
            for (
                call_id,
                provider_name,
                arguments,
            ) in GoogleClient._function_calls(chunk):
                if self._capability_catalog is not None:
                    decoded = self._capability_catalog.decode_call(
                        ProviderCapabilityCall(
                            call_id=call_id,
                            provider_name=provider_name,
                            arguments=cast(
                                str | Mapping[str, object], arguments
                            ),
                        ),
                        provider_family=ProviderFamily.GOOGLE,
                    )
                    canonical_name = (
                        decoded.canonical_name
                        if isinstance(decoded, TaskInputCapabilityCall)
                        else decoded.name
                    )
                else:
                    canonical_name = provider_name
                correlation = StreamItemCorrelation(tool_call_id=call_id)
                yield StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    correlation=correlation,
                    text_delta=to_json(arguments),
                    provider_payload=provider_payload,
                    provider_event_type="generate_content.function_call",
                )
                yield StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_READY,
                    correlation=correlation,
                    data={"name": canonical_name},
                    provider_payload=provider_payload,
                    provider_event_type="generate_content.function_call",
                )
                yield StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    correlation=correlation,
                    provider_payload=provider_payload,
                    provider_event_type="generate_content.function_call",
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
    _reasoning_summary_provider = "google"
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
        capability: ModelCapabilityCatalog | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        self._validate_reasoning_summary_request(settings)
        assert (
            instructions is None
        ), "Google does not support provider instructions"
        contents = self._template_messages(
            messages,
            ["system"],
            capability=capability,
        )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "contents": cast(Any, contents),
        }
        config = self._config(
            model_id,
            messages,
            settings,
            capability=capability,
        )
        if config:
            kwargs["config"] = config

        if use_async_generator:
            stream = await self._client.aio.models.generate_content_stream(
                **kwargs,
            )
            return GoogleStream(
                stream=stream.__aiter__(),
                capability=capability,
            )
        else:
            response = await self._client.aio.models.generate_content(
                **kwargs,
            )
            answer_text, calls = GoogleClient._response_parts(
                response,
                capability=capability,
            )
            usage = GoogleClient._field(response, "usage_metadata") or (
                GoogleClient._field(response, "usageMetadata")
            )
            if calls:
                return TextGenerationNonStreamResult.from_provider_parts(
                    answer_text=answer_text,
                    calls=calls,
                    provider_family=ProviderFamily.GOOGLE,
                    usage=usage,
                    answer_event_type="generate_content.text",
                    terminal_event_type="generate_content.completed",
                )
            return TextGenerationSingleStream(
                answer_text,
                provider_family=ProviderFamily.GOOGLE,
                usage=usage,
            )

    def _config(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None,
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> dict[str, Any] | None:
        config: dict[str, Any] = {}
        system_prompt = self._system_prompt(messages)
        if system_prompt:
            config["system_instruction"] = system_prompt
        if capability is not None:
            projection = capability.project(ProviderFamily.GOOGLE)
            if not projection.is_empty:
                declarations: list[dict[str, Any]] = []
                for schema in projection.schemas:
                    if schema.get("type") != "function":
                        continue
                    function = schema.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    declarations.append(
                        {
                            "name": function.get("name", ""),
                            "description": function.get("description", ""),
                            "parameters_json_schema": _mutable_provider_json(
                                function.get("parameters", {})
                            ),
                        }
                    )
                config["tools"] = [{"function_declarations": declarations}]
                if settings and settings.tool_choice is not None:
                    config["tool_config"] = {
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": [
                                projection.tool_choice(settings.tool_choice)
                            ],
                        }
                    }
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
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> list[dict[str, Any]]:
        excluded_roles = set(exclude_roles or [])
        output: list[dict[str, Any]] = []
        for message in messages:
            if str(message.role) in excluded_roles:
                continue
            if message.role == MessageRole.TOOL:
                outcome = (
                    message.tool_call_result
                    or message.tool_call_error
                    or message.tool_call_diagnostic
                )
                if outcome is None:
                    continue
                if isinstance(outcome, ToolCallDiagnostic):
                    if outcome.call_id is None:
                        output.append(
                            {
                                "role": "model",
                                "parts": [
                                    {
                                        "text": to_json(
                                            tool_call_diagnostic_payload(
                                                outcome
                                            )
                                        )
                                    }
                                ],
                            }
                        )
                        continue
                    call_id = str(outcome.call_id)
                    canonical_name = (
                        message.name
                        or outcome.canonical_name
                        or outcome.requested_name
                        or "tool"
                    )
                    result: object = tool_call_diagnostic_payload(outcome)
                else:
                    assert isinstance(
                        outcome,
                        (ToolCallResult, ToolCallError),
                    )
                    call_id = str(outcome.call.id)
                    canonical_name = outcome.call.name
                    result = (
                        outcome.result
                        if isinstance(outcome, ToolCallResult)
                        else {"error": outcome.message}
                    )
                provider_name = TextGenerationVendor.provider_tool_name(
                    canonical_name,
                    capability=capability,
                    provider_family=ProviderFamily.GOOGLE,
                )
                output.append(
                    {
                        "role": str(MessageRole.USER),
                        "parts": [
                            {
                                "function_response": {
                                    "id": call_id,
                                    "name": provider_name,
                                    "response": {"output": result},
                                }
                            }
                        ],
                    }
                )
                continue

            templated = cast(
                list[dict[str, Any]],
                super()._template_messages([message]),
            )[0]
            parts = (
                []
                if message.content is None and message.tool_calls
                else self._parts(templated.get("content"))
            )
            for call in message.tool_calls or []:
                parts.append(
                    {
                        "function_call": {
                            "id": str(call.id) if call.id is not None else "",
                            "name": TextGenerationVendor.provider_tool_name(
                                call.name,
                                capability=capability,
                                provider_family=ProviderFamily.GOOGLE,
                            ),
                            "args": call.arguments or {},
                        }
                    }
                )
            output.append(
                {
                    "role": self._message_role(cast(str, templated["role"])),
                    "parts": parts,
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
    def _function_calls(
        response: object,
    ) -> tuple[tuple[str, str, object], ...]:
        candidates = GoogleClient._field(response, "candidates")
        if not isinstance(candidates, list):
            return ()
        calls: list[tuple[str, str, object]] = []
        for candidate in candidates:
            content = GoogleClient._field(candidate, "content")
            parts = GoogleClient._field(content, "parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                call = GoogleClient._field(part, "function_call")
                if call is None:
                    call = GoogleClient._field(part, "functionCall")
                if call is None:
                    continue
                call_id = GoogleClient._field(call, "id")
                name = GoogleClient._field(call, "name")
                arguments = GoogleClient._field(call, "args")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        "google function call id must be a non-empty string"
                    )
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "google function call name must be a non-empty string"
                    )
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, (str, Mapping)):
                    raise ValueError(
                        "google function call arguments must be an object "
                        "or string"
                    )
                calls.append((call_id, name, arguments))
        return tuple(calls)

    @staticmethod
    def _response_text(
        response: object,
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> str:
        answer_text, calls = GoogleClient._response_parts(
            response,
            capability=capability,
        )
        return answer_text + "".join(
            TextGenerationVendor.build_tool_call_text(
                call.call_id,
                call.name,
                call.arguments,
                tool_name_is_canonical=True,
            )
            for call in calls
        )

    @staticmethod
    def _response_parts(
        response: object,
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> tuple[str, tuple[TextGenerationNonStreamToolCall, ...]]:
        text = GoogleClient._field(response, "text")
        answer_text = text if isinstance(text, str) else ""
        calls: list[TextGenerationNonStreamToolCall] = []
        for call_id, provider_name, arguments in GoogleClient._function_calls(
            response
        ):
            calls.append(
                TextGenerationVendor.non_stream_tool_call(
                    call_id=call_id,
                    provider_name=provider_name,
                    arguments=arguments,
                    capability=capability,
                    provider_family=ProviderFamily.GOOGLE,
                    provider_event_type="generate_content.function_call",
                )
            )
        return answer_text, tuple(calls)

    @staticmethod
    def capability_result_message(
        result: CorrelatedCapabilityResult,
    ) -> dict[str, Any]:
        return {
            "role": str(MessageRole.USER),
            "parts": [
                {
                    "function_response": {
                        "id": str(result.call_id),
                        "name": result.provider_name,
                        "response": result.provider_payload(),
                    }
                }
            ],
        }

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
