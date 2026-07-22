from .....entities import (
    GenerationSettings,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
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
    StreamProviderAdapterError,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentState,
    StreamVisibility,
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
    _decode_text_file_data,
)

from base64 import b64decode
from contextlib import AsyncExitStack
from json import dumps
from re import sub
from typing import Any, AsyncIterator, Mapping, NoReturn, cast

from aioboto3 import Session as Boto3Session


def _mutable_provider_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _mutable_provider_json(item) for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_mutable_provider_json(item) for item in value]
    return value


def _get(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _string(value["text"])
        if "reasoningText" in value:
            return _string(value["reasoningText"])
        if "string" in value:
            return _string(value["string"])
    return None


def _bedrock_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    details = response.get("Error")
    if not isinstance(details, dict):
        return None
    code = details.get("Code")
    return code if isinstance(code, str) else None


def _bedrock_error_message(error: Exception) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        details = response.get("Error")
        if isinstance(details, dict):
            message = details.get("Message")
            if isinstance(message, str):
                return message
    return str(error)


def _geo_inference_prefix(region_name: str | None) -> str | None:
    if region_name is None:
        return None
    if region_name.startswith("us-"):
        return "us."
    if region_name.startswith("eu-"):
        return "eu."
    return None


_BEDROCK_XLSX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
_BEDROCK_DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_BEDROCK_DOCUMENT_FORMATS = {
    "application/msword": "doc",
    "application/pdf": "pdf",
    "application/vnd.ms-excel": "xls",
    _BEDROCK_XLSX_MIME_TYPE: "xlsx",
    _BEDROCK_DOCX_MIME_TYPE: "docx",
    "text/csv": "csv",
    "text/html": "html",
    "text/markdown": "md",
    "text/plain": "txt",
}


class BedrockStream(TextGenerationVendorStream):
    _events: AsyncIterator[Any]
    _canonical_tool_blocks: dict[int, dict[str, Any]]
    _canonical_ready_tool_call_ids: set[str]
    _canonical_done_tool_call_ids: set[str]
    _reasoning_segments: StreamReasoningSegmentState
    _capability_catalog: ModelCapabilityCatalog | None

    def __init__(
        self,
        events: AsyncIterator[Any],
        *,
        capability: ModelCapabilityCatalog | None = None,
    ):
        self._events = events
        self._canonical_tool_blocks = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._reasoning_segments = StreamReasoningSegmentState()
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
            provider_family=ProviderFamily.BEDROCK,
            sources=(events,),
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
        self._canonical_tool_blocks = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._reasoning_segments = StreamReasoningSegmentState()
        return self._provider_canonical_stream(
            self._provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family,
            capabilities=capabilities
            or StreamProviderCapabilities(
                backend=StreamProducerBackend.HOSTED,
                provider_family=self._provider_family,
                supports_reasoning=True,
                supports_tool_calls=True,
                supports_usage=True,
                supports_terminal_events=True,
                supports_cancellation=True,
            ),
            close_after_terminal=close_after_terminal,
        )

    async def _provider_events(self) -> AsyncIterator[StreamProviderEvent]:
        message_stopped = False
        terminal_usage: LooseJsonValue | None = None
        terminal_usage_payload: LooseJsonValue | None = None
        message_stop_payload: LooseJsonValue | None = None

        try:
            async for event in self._events:
                provider_payload = self._provider_payload(event)
                provider_event_type = self._provider_event_type(event)
                metadata = _get(event, "metadata")
                usage = (
                    metadata.get("usage")
                    if isinstance(metadata, dict)
                    else None
                )
                if usage is not None:
                    terminal_usage = cast(LooseJsonValue, usage)
                    terminal_usage_payload = provider_payload
                    continue

                message_stop = _get(event, "messageStop")
                if message_stop is not None or (
                    isinstance(event, dict) and "messageStop" in event
                ):
                    message_stopped = True
                    message_stop_payload = provider_payload
                    continue

                if message_stopped:
                    continue

                try:
                    provider_events = self._provider_events_from_event(event)
                except Exception as exc:
                    raise StreamProviderAdapterError(
                        exc,
                        provider_payload=provider_payload,
                        provider_event_type=provider_event_type,
                    ) from exc
                for provider_event in provider_events:
                    yield provider_event
                    if (
                        provider_event.kind
                        is not StreamItemKind.REASONING_DELTA
                    ):
                        self._reasoning_segments.complete_segment()

            if terminal_usage is not None:
                self._usage = terminal_usage
                yield StreamProviderEvent(
                    kind=StreamItemKind.USAGE_COMPLETED,
                    usage=terminal_usage,
                    provider_payload=terminal_usage_payload,
                    provider_event_type="metadata.usage",
                )
            yield StreamProviderEvent(
                kind=StreamItemKind.STREAM_COMPLETED,
                provider_payload=message_stop_payload,
                provider_event_type=(
                    "messageStop" if message_stop_payload is not None else None
                ),
            )
        finally:
            await self.aclose()

    def _provider_events_from_event(
        self, event: object
    ) -> tuple[StreamProviderEvent, ...]:
        provider_payload = self._provider_payload(event)

        content_start = _get(event, "contentBlockStart")
        if content_start:
            return self._content_block_start_events(
                content_start,
                provider_payload,
            )

        content_delta = _get(event, "contentBlockDelta")
        if content_delta:
            return self._content_block_delta_events(
                content_delta,
                provider_payload,
            )

        content_stop = _get(event, "contentBlockStop")
        if content_stop:
            return self._content_block_stop_events(
                content_stop,
                provider_payload,
            )

        return ()

    def _content_block_start_events(
        self,
        content_start: Mapping[str, Any],
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        block_index = content_start.get("contentBlockIndex")
        if not isinstance(block_index, int):
            raise ValueError("bedrock content block index must be an integer")
        block = content_start.get("contentBlock") or {}
        tool = block.get("toolUse") if isinstance(block, dict) else None
        if not tool:
            return ()
        call_id = self._tool_call_id(tool.get("toolUseId"))
        name = tool.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("bedrock tool call name must be a string")
        self._canonical_tool_blocks[block_index] = {
            "id": call_id,
            "name": name,
            "arguments_seen": False,
            "arguments": "",
        }
        initial = tool.get("input")
        if initial in (None, ""):
            return ()
        return (
            self._tool_argument_delta_event(
                block_index,
                initial,
                provider_payload,
                "contentBlockStart",
            ),
        )

    def _content_block_delta_events(
        self,
        content_delta: Mapping[str, Any],
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        block_index = content_delta.get("contentBlockIndex")
        if not isinstance(block_index, int):
            raise ValueError("bedrock content block index must be an integer")
        delta = content_delta.get("delta") or {}
        text_value = _string(delta.get("text"))
        if text_value:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=text_value,
                    provider_payload=provider_payload,
                    provider_event_type="contentBlockDelta",
                ),
            )
        reasoning_value = _string(delta.get("reasoning"))
        if reasoning_value:
            representation = StreamReasoningRepresentation.NATIVE_TEXT
            correlation = StreamItemCorrelation(
                provider_output_index=block_index
            )
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=reasoning_value,
                    correlation=correlation,
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=representation,
                    segment_instance_ordinal=(
                        self._reasoning_segments.allocate(
                            representation, correlation
                        )
                    ),
                    provider_payload=provider_payload,
                    provider_event_type="contentBlockDelta",
                ),
            )
        tool_delta = delta.get("toolUse")
        if tool_delta:
            fragment = tool_delta.get("input")
            if fragment in (None, ""):
                return ()
            return (
                self._tool_argument_delta_event(
                    block_index,
                    fragment,
                    provider_payload,
                    "contentBlockDelta",
                ),
            )
        return ()

    def _content_block_stop_events(
        self,
        content_stop: Mapping[str, Any],
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        self._reasoning_segments.complete_segment()
        block_index = content_stop.get("contentBlockIndex")
        if not isinstance(block_index, int):
            raise ValueError("bedrock content block index must be an integer")
        block = content_stop.get("contentBlock") or {}
        tool = block.get("toolUse") if isinstance(block, dict) else None
        cached = self._canonical_tool_blocks.pop(block_index, None)
        if cached is None and not tool:
            return ()
        if cached is None:
            assert tool is not None
            cached = {
                "id": self._tool_call_id(tool.get("toolUseId")),
                "name": tool.get("name"),
                "arguments_seen": False,
                "arguments": "",
            }
        if tool:
            name = tool.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise ValueError("bedrock tool call name must be a string")
                cached["name"] = name
            final_input = tool.get("input")
        else:
            final_input = None
        result: list[StreamProviderEvent] = []
        if final_input not in (None, ""):
            result.append(
                self._tool_argument_delta_event(
                    block_index,
                    final_input,
                    provider_payload,
                    "contentBlockStop",
                    state=cached,
                )
            )
        call_id = cast(str, cached["id"])
        result.extend(
            self._mark_tool_ready(
                call_id,
                cached.get("name"),
                cast(str, cached.get("arguments", "")),
                provider_payload,
            )
        )
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                provider_payload=provider_payload,
                provider_event_type="contentBlockStop",
            )
        )
        self._canonical_done_tool_call_ids.add(call_id)
        return tuple(result)

    def _tool_argument_delta_event(
        self,
        block_index: int,
        value: object,
        provider_payload: LooseJsonValue | None,
        provider_event_type: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> StreamProviderEvent:
        tool_block = state or self._canonical_tool_blocks.get(block_index)
        if tool_block is None:
            raise ValueError("bedrock tool call is missing start event")
        call_id = cast(str, tool_block["id"])
        fragment = value if isinstance(value, str) else dumps(value)
        tool_block["arguments_seen"] = True
        tool_block["arguments"] = (
            cast(str, tool_block.get("arguments", "")) + fragment
        )
        return StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation=StreamItemCorrelation(tool_call_id=call_id),
            text_delta=fragment,
            provider_payload=provider_payload,
            provider_event_type=provider_event_type,
        )

    def _mark_tool_ready(
        self,
        call_id: str,
        name: object | None,
        arguments: str,
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        if call_id in self._canonical_done_tool_call_ids:
            raise ValueError("bedrock tool call already completed")
        if call_id in self._canonical_ready_tool_call_ids:
            return ()
        self._canonical_ready_tool_call_ids.add(call_id)
        canonical_name = name
        if isinstance(name, str) and self._capability_catalog is not None:
            decoded = self._capability_catalog.decode_call(
                ProviderCapabilityCall(
                    call_id=call_id,
                    provider_name=name,
                    arguments=arguments or None,
                ),
                provider_family=ProviderFamily.BEDROCK,
            )
            canonical_name = (
                decoded.canonical_name
                if isinstance(decoded, TaskInputCapabilityCall)
                else decoded.name
            )
        elif isinstance(name, str):
            try:
                canonical_name = TextGenerationVendor.canonical_tool_name(name)
            except AssertionError:
                pass
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                data=(
                    {"name": canonical_name}
                    if isinstance(canonical_name, str)
                    else {}
                ),
                provider_payload=provider_payload,
                provider_event_type="contentBlockStop",
            ),
        )

    @staticmethod
    def _tool_call_id(value: object) -> str:
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError("bedrock tool call id must be a non-empty string")

    @staticmethod
    def _provider_payload(event: object) -> LooseJsonValue | None:
        return dict(event) if isinstance(event, dict) else None

    @staticmethod
    def _provider_event_type(event: object) -> str | None:
        if not isinstance(event, dict):
            return None
        for key in (
            "contentBlockStart",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "metadata",
        ):
            if key in event:
                return key
        return None


class BedrockClient(TextGenerationVendor):
    _reasoning_summary_provider = "bedrock"
    _client: Any | None
    _endpoint_url: str | None
    _exit_stack: AsyncExitStack
    _region_name: str | None
    _session: Boto3Session

    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self._session = Boto3Session()
        self._region_name = region_name
        self._endpoint_url = endpoint_url
        self._exit_stack = exit_stack
        self._client = None

    async def _client_instance(self) -> Any:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._region_name:
                kwargs["region_name"] = self._region_name
            if self._endpoint_url:
                kwargs["endpoint_url"] = self._endpoint_url
            self._client = await self._exit_stack.enter_async_context(
                self._session.client("bedrock-runtime", **kwargs)
            )
        return self._client

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
        ), "Amazon Bedrock does not support provider instructions"
        client = await self._client_instance()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(
            messages,
            ["system"],
            capability=capability,
        )
        payload: dict[str, Any] = {
            "modelId": model_id,
            "messages": template_messages,
        }
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]
        inference = self._inference_config(settings)
        if inference:
            payload["inferenceConfig"] = inference
        tool_config = self._tool_config(capability, settings=settings)
        if tool_config:
            payload["toolConfig"] = tool_config

        try:
            if use_async_generator:
                response = await client.converse_stream(**payload)
                stream = (
                    response.get("stream")
                    if isinstance(response, dict)
                    else None
                )
                assert (
                    stream is not None
                ), "Missing stream in Converse response"
                events = (
                    await self._exit_stack.enter_async_context(stream)
                    if hasattr(stream, "__aenter__")
                    else stream
                )
                stream_kwargs: dict[str, Any] = {}
                if capability is not None:
                    stream_kwargs["capability"] = capability
                return BedrockStream(events=events, **stream_kwargs)

            response = await client.converse(**payload)
            usage = (
                response.get("usage") if isinstance(response, dict) else None
            )
            answer_text, calls = self._response_parts(
                response,
                capability=capability,
            )
            if calls:
                return TextGenerationNonStreamResult.from_provider_parts(
                    answer_text=answer_text,
                    calls=calls,
                    provider_family=ProviderFamily.BEDROCK,
                    usage=usage,
                    answer_event_type="converse.text",
                    terminal_event_type="converse.completed",
                )
            return TextGenerationSingleStream(
                answer_text,
                provider_family=ProviderFamily.BEDROCK,
                usage=usage,
            )
        except Exception as error:
            if self._is_invalid_model_identifier_error(error):
                self._raise_invalid_model_identifier(model_id, error)
            if self._is_inference_profile_required_error(error):
                self._raise_inference_profile_required_error(model_id, error)
            if self._is_use_case_details_required_error(error):
                self._raise_use_case_details_required_error(model_id, error)
            if self._is_end_of_life_model_error(error):
                self._raise_end_of_life_model_error(model_id, error)
            raise

    @staticmethod
    def _is_invalid_model_identifier_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ValidationException":
            return False
        return (
            "model identifier is invalid"
            in _bedrock_error_message(error).lower()
        )

    def _raise_invalid_model_identifier(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Invalid Amazon Bedrock model identifier {model_id!r}. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Verify the exact Bedrock foundation-model or "
            "inference-profile ID for your account."
        )
        if model_id.startswith("anthropic.") and not model_id.startswith(
            (
                "us.",
                "eu.",
                "apac.",
            )
        ):
            prefix = _geo_inference_prefix(self._region_name)
            if prefix:
                message += (
                    " Anthropic Bedrock models in this region may require "
                    "a geo-prefixed inference profile ID."
                    f" Try {prefix!r} as the model ID prefix."
                )
            else:
                message += (
                    " Anthropic Bedrock models may require a geo-prefixed "
                    "inference profile such as 'us.anthropic...'."
                )
        raise ValueError(message) from error

    @staticmethod
    def _is_inference_profile_required_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ValidationException":
            return False
        message = _bedrock_error_message(error).lower()
        return (
            "on-demand throughput" in message
            and "inference profile" in message
        )

    def _raise_inference_profile_required_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Amazon Bedrock model identifier {model_id!r} cannot be invoked "
            "directly with on-demand throughput. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += " Use an inference-profile ID or ARN for this model."
        if model_id.startswith("anthropic."):
            prefix = _geo_inference_prefix(self._region_name) or "us."
            regional_profile = prefix + model_id
            global_profile = "global." + model_id
            message += f" Try {regional_profile!r} or {global_profile!r}."
        raise ValueError(message) from error

    @staticmethod
    def _is_use_case_details_required_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ResourceNotFoundException":
            return False
        message = _bedrock_error_message(error).lower()
        return (
            "use case details have not been submitted" in message
            or "fill out the request form" in message
        )

    def _raise_use_case_details_required_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            "Amazon Bedrock blocked access to model identifier "
            f"{model_id!r} because Anthropic use-case details have not "
            "been submitted for this account. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Submit the Anthropic model access form in Amazon Bedrock, "
            "then retry."
        )
        message += (
            " You can verify the current status with "
            "'aws bedrock get-use-case-for-model-access --region "
            f"{self._region_name or 'us-east-1'}'."
        )
        raise ValueError(message) from error

    @staticmethod
    def _is_end_of_life_model_error(error: Exception) -> bool:
        if _bedrock_error_code(error) != "ResourceNotFoundException":
            return False
        return "end of its life" in _bedrock_error_message(error).lower()

    def _raise_end_of_life_model_error(
        self, model_id: str, error: Exception
    ) -> NoReturn:
        message = (
            f"Amazon Bedrock model identifier {model_id!r} is no longer "
            "usable because that model version reached end of life. "
            f"Bedrock replied: {_bedrock_error_message(error)}."
        )
        if self._region_name:
            message += f" Requested region: {self._region_name!r}."
        message += (
            " Use an active inference-profile ID instead of the retired "
            "profile or model version."
        )
        if model_id.startswith(("us.anthropic.", "eu.anthropic.")):
            prefix = model_id.split(".", 1)[0]
            message += (
                " List current options with "
                "'aws bedrock list-inference-profiles --region "
                f"{self._region_name or 'us-east-1'}' and look for active "
                f"{prefix}.anthropic profiles."
            )
        elif model_id.startswith("anthropic."):
            geo_prefix = _geo_inference_prefix(self._region_name)
            if geo_prefix:
                message += (
                    " Anthropic Bedrock models in this region are typically "
                    "invoked through inference profiles."
                    f" Try an active {geo_prefix!r}-prefixed profile."
                )
        raise ValueError(message) from error

    def _inference_config(
        self, settings: GenerationSettings | None
    ) -> dict[str, Any] | None:
        if settings is None:
            return None
        config: dict[str, Any] = {}
        if settings.max_new_tokens is not None:
            config["maxTokens"] = settings.max_new_tokens
        if settings.temperature is not None:
            config["temperature"] = settings.temperature
        if settings.top_p is not None:
            config["topP"] = settings.top_p
        if settings.top_k is not None:
            config["topK"] = settings.top_k
        if settings.stop_strings is not None:
            stop = (
                [settings.stop_strings]
                if isinstance(settings.stop_strings, str)
                else settings.stop_strings
            )
            config["stopSequences"] = stop
        return config or None

    def _tool_config(
        self,
        capability: ModelCapabilityCatalog | None,
        *,
        settings: GenerationSettings | None = None,
    ) -> dict[str, Any] | None:
        schemas = self._tool_schemas(capability) if capability else None
        if not schemas:
            return None
        tool_choice: dict[str, Any]
        if settings is None or settings.tool_choice is None:
            tool_choice = {"auto": {}}
        else:
            assert capability is not None
            provider_name = capability.project(
                ProviderFamily.BEDROCK
            ).tool_choice(settings.tool_choice)
            tool_choice = {"tool": {"name": provider_name}}
        return {"tools": schemas, "toolChoice": tool_choice}

    def _response_text(
        self,
        response: dict[str, Any],
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> str:
        answer_text, calls = self._response_parts(
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

    def _response_parts(
        self,
        response: dict[str, Any],
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> tuple[str, tuple[TextGenerationNonStreamToolCall, ...]]:
        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return "", ()
        parts: list[str] = []
        calls: list[TextGenerationNonStreamToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text_block = block.get("text")
            text_value = _string(text_block)
            if text_value:
                parts.append(text_value)
                continue
            tool_use = block.get("toolUse")
            if not isinstance(tool_use, dict):
                continue
            provider_name = tool_use.get("name")
            call_id = tool_use.get("toolUseId")
            arguments = tool_use.get("input")
            calls.append(
                TextGenerationVendor.non_stream_tool_call(
                    call_id=call_id,
                    provider_name=provider_name,
                    arguments=arguments,
                    capability=capability,
                    provider_family=ProviderFamily.BEDROCK,
                    provider_event_type="converse.tool_use",
                )
            )
        return "".join(parts), tuple(calls)

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> list[dict[str, Any]]:
        templated: list[dict[str, Any]] = []
        for message in messages:
            if exclude_roles and str(message.role) in exclude_roles:
                continue
            if message.role == MessageRole.TOOL:
                result = (
                    message.tool_call_result
                    or message.tool_call_error
                    or message.tool_call_diagnostic
                )
                if (
                    isinstance(result, ToolCallDiagnostic)
                    and result.call_id is None
                ):
                    templated.append(
                        {
                            "role": str(MessageRole.ASSISTANT),
                            "content": [
                                {
                                    "text": to_json(
                                        tool_call_diagnostic_payload(result)
                                    )
                                }
                            ],
                        }
                    )
                    continue
                if result:
                    templated.append(self._tool_result_message(result))
                continue
            templated.append(
                self._format_message(message, capability=capability)
            )
        return templated

    def _format_message(
        self,
        message: Message,
        *,
        capability: ModelCapabilityCatalog | None = None,
    ) -> dict[str, Any]:
        role = str(message.role)
        if role == str(MessageRole.DEVELOPER):
            role = str(MessageRole.USER)
        content_blocks = self._format_content(message.content)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                provider_name = TextGenerationVendor.provider_tool_name(
                    tool_call.name,
                    capability=capability,
                    provider_family=ProviderFamily.BEDROCK,
                )
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": provider_name,
                            "input": tool_call.arguments or [],
                        }
                    }
                )
        return {"role": role, "content": content_blocks}

    def _format_content(
        self, content: str | MessageContent | list[MessageContent] | None
    ) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, MessageContentText):
            return [{"text": content.text}]
        if isinstance(content, MessageContentFile):
            return self._ensure_document_prompt(
                [{"document": self._document_block(content.file)}]
            )
        if isinstance(content, MessageContentImage):
            return [
                {"image": {"source": self._image_source(content.image_url)}}
            ]
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if isinstance(block, MessageContentText):
                    blocks.append({"text": block.text})
                elif isinstance(block, MessageContentFile):
                    blocks.append(
                        {"document": self._document_block(block.file)}
                    )
                elif isinstance(block, MessageContentImage):
                    blocks.append(
                        {
                            "image": {
                                "source": self._image_source(block.image_url)
                            }
                        }
                    )
            return self._ensure_document_prompt(blocks)
        return [{"text": str(content)}]

    @staticmethod
    def _ensure_document_prompt(
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        has_document = any("document" in block for block in blocks)
        has_text = any("text" in block for block in blocks)
        if has_document and not has_text:
            return [{"text": ""}, *blocks]
        return blocks

    @staticmethod
    def _document_block(file: Mapping[str, Any]) -> dict[str, Any]:
        block: dict[str, Any] = {
            "name": BedrockClient._document_name(file),
            "source": BedrockClient._document_source(file),
        }
        document_format = BedrockClient._document_format(file)
        if document_format:
            block["format"] = document_format
        citations = file.get("citations")
        if isinstance(citations, bool):
            block["citations"] = {"enabled": citations}
        context = file.get("context")
        if isinstance(context, str):
            block["context"] = context
        return block

    @staticmethod
    def _document_source(file: Mapping[str, Any]) -> dict[str, Any]:
        mime_type_value = file.get("mime_type")
        mime_type = (
            mime_type_value.lower()
            if isinstance(mime_type_value, str) and mime_type_value
            else None
        )
        data = file.get("file_data", file.get("data"))
        if isinstance(data, (bytes, bytearray)):
            return {"bytes": bytes(data)}
        if isinstance(data, str):
            if mime_type is not None and mime_type.startswith("text/"):
                return {"text": _decode_text_file_data(data)}
            return {"bytes": b64decode(data)}

        file_uri = BedrockClient._file_uri(file)
        assert (
            file_uri is not None
        ), "Bedrock documents require inline data or a file URL"
        assert file_uri.startswith(
            "s3://"
        ), "Bedrock document URLs must use s3:// URIs"
        s3_location: dict[str, Any] = {"uri": file_uri}
        bucket_owner = file.get("bucket_owner")
        if isinstance(bucket_owner, str):
            s3_location["bucketOwner"] = bucket_owner
        return {"s3Location": s3_location}

    @staticmethod
    def _document_name(file: Mapping[str, Any]) -> str:
        name = "Document"
        for key in ("title", "filename"):
            value = file.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip().rsplit(".", 1)[0]
                break
        sanitized = sub(r"[^0-9A-Za-z\s\-\(\)\[\]]", " ", name)
        normalized = sub(r"\s+", " ", sanitized).strip()
        return normalized or "Document"

    @staticmethod
    def _document_format(file: Mapping[str, Any]) -> str | None:
        mime_type_value = file.get("mime_type")
        if isinstance(mime_type_value, str):
            mime_type = mime_type_value.lower()
            document_format = _BEDROCK_DOCUMENT_FORMATS.get(mime_type)
            if document_format:
                return document_format

        for key in ("filename", "title"):
            value = file.get(key)
            if not isinstance(value, str):
                continue
            suffix = value.rsplit(".", 1)
            if len(suffix) != 2:
                continue
            extension = suffix[1].lower()
            if extension in {
                "csv",
                "doc",
                "docx",
                "html",
                "md",
                "pdf",
                "txt",
                "xls",
                "xlsx",
            }:
                return extension
        return None

    @staticmethod
    def _file_uri(file: Mapping[str, Any]) -> str | None:
        for key in ("file_url", "url", "uri"):
            value = file.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _image_source(self, image_url: dict[str, Any]) -> dict[str, Any]:
        if "url" in image_url:
            return {"type": "url", "url": image_url["url"]}
        if "data" in image_url:
            media_type = image_url.get("mime_type", "image/png")
            return {
                "type": "base64",
                "mediaType": media_type,
                "data": image_url["data"],
            }
        return {"type": "url", "url": image_url.get("uri", "")}

    def _tool_result_message(
        self, result: ToolCallResult | ToolCallError | ToolCallDiagnostic
    ) -> dict[str, Any]:
        if isinstance(result, ToolCallDiagnostic):
            assert result.call_id is not None
            return {
                "role": str(MessageRole.USER),
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": str(result.call_id),
                            "content": [
                                {
                                    "text": to_json(
                                        tool_call_diagnostic_payload(result)
                                    )
                                }
                            ],
                            "status": "error",
                        }
                    }
                ],
            }

        content: dict[str, Any] = {
            "toolUseId": result.call.id,
            "content": [
                {
                    "text": to_json(
                        result.result
                        if isinstance(result, ToolCallResult)
                        else result.message
                    )
                }
            ],
            "status": (
                "success" if isinstance(result, ToolCallResult) else "error"
            ),
        }
        if isinstance(result, ToolCallError):
            content["error"] = {
                "name": result.error_type,
                "message": result.message,
            }
        return {
            "role": str(MessageRole.USER),
            "content": [{"toolResult": content}],
        }

    @staticmethod
    def capability_result_message(
        result: CorrelatedCapabilityResult,
    ) -> dict[str, Any]:
        return {
            "role": str(MessageRole.USER),
            "content": [
                {
                    "toolResult": {
                        "toolUseId": str(result.call_id),
                        "content": [{"json": result.provider_payload()}],
                        "status": "success",
                    }
                }
            ],
        }

    @staticmethod
    def _tool_schemas(
        capability: ModelCapabilityCatalog,
    ) -> list[dict[str, Any]] | None:
        schemas = capability.project(ProviderFamily.BEDROCK).schemas
        if not schemas:
            return None
        tools: list[dict[str, Any]] = []
        for schema in schemas:
            if schema.get("type") != "function":
                continue
            function = schema.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name", "")
            tools.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": _mutable_provider_json(
                                function.get("parameters", {})
                            )
                        },
                    }
                }
            )
        return tools or None


class BedrockModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        return BedrockClient(
            exit_stack=self._exit_stack,
            region_name=self._settings.base_url,
            endpoint_url=self._settings.access_token,
        )
