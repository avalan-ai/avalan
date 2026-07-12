from .....entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageRole,
    MessageToolCall,
    ReasoningEffort,
    ToolCall,
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
    TextGenerationSingleStream,
)
from .....tool.manager import ToolManager
from .....types import LooseJsonValue
from .....utils import to_json, tool_call_diagnostic_payload
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
    _decode_text_file_data,
)

from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, cast

from anthropic import APIStatusError, AsyncAnthropic
from anthropic.types import RawContentBlockDeltaEvent, RawMessageStopEvent

_ANTHROPIC_USAGE_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
    "output_tokens",
    "output_tokens_details",
)


def _field(value: object, attribute: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(attribute)
    return getattr(value, attribute, None)


def _anthropic_event_usage(event: object) -> object | None:
    for value in (event, _field(event, "message"), _field(event, "delta")):
        if value is None:
            continue
        usage = _field(value, "usage")
        if usage is not None:
            return usage
    return None


def _usage_mapping(usage: object) -> dict[str, object]:
    if isinstance(usage, Mapping):
        return {
            key: value
            for key, value in usage.items()
            if isinstance(key, str) and value is not None
        }
    return {
        key: value
        for key in _ANTHROPIC_USAGE_KEYS
        if (value := getattr(usage, key, None)) is not None
    }


def _merge_usage(left: object | None, right: object) -> object | None:
    right_value = _usage_mapping(right)
    if not right_value:
        return left
    if left is None:
        return right_value
    return {**_usage_mapping(left), **right_value}


class AnthropicStream(TextGenerationVendorStream):
    _events: AsyncIterator[object]
    _canonical_tool_blocks: dict[int, dict[str, Any]]
    _canonical_ready_tool_call_ids: set[str]
    _canonical_done_tool_call_ids: set[str]
    _reasoning_segments: StreamReasoningSegmentState
    _tool_manager: ToolManager | None

    def __init__(
        self,
        events: AsyncIterator[object],
        *,
        tool: ToolManager | None = None,
    ):
        self._events = events
        self._canonical_tool_blocks = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._reasoning_segments = StreamReasoningSegmentState()
        self._tool_manager = tool

        async def generator() -> AsyncIterator[CanonicalStreamItem]:
            async for item in self.canonical_stream(
                stream_session_id=self._DEFAULT_STREAM_SESSION_ID,
                run_id=self._DEFAULT_RUN_ID,
                turn_id=self._DEFAULT_TURN_ID,
            ):
                yield item

        super().__init__(
            generator(),
            provider_family=ProviderFamily.ANTHROPIC,
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
        cumulative_usage: object | None = None
        usage_provider_payload: LooseJsonValue | None = None
        usage_provider_event_type: str | None = None
        try:
            async for event in self._events:
                event_type = _field(event, "type")
                provider_event_type = (
                    event_type if isinstance(event_type, str) else None
                )
                event_usage = _anthropic_event_usage(event)
                event_provider_payload: LooseJsonValue | None = None
                if event_usage is not None:
                    usage_delta = _usage_mapping(event_usage)
                    merged_usage = _merge_usage(
                        cumulative_usage,
                        event_usage,
                    )
                    if usage_delta:
                        event_provider_payload = self._provider_payload(event)
                        usage_provider_payload = event_provider_payload
                        usage_provider_event_type = provider_event_type
                    cumulative_usage = merged_usage
                if (
                    isinstance(event, RawMessageStopEvent)
                    or event_type == "message_stop"
                ):
                    provider_payload = (
                        event_provider_payload
                        if event_provider_payload is not None
                        else self._provider_payload(event)
                    )
                    if cumulative_usage is not None:
                        self._usage = cumulative_usage
                        yield StreamProviderEvent(
                            kind=StreamItemKind.USAGE_COMPLETED,
                            usage=cast(LooseJsonValue, cumulative_usage),
                            provider_payload=usage_provider_payload,
                            provider_event_type=usage_provider_event_type,
                        )
                    yield StreamProviderEvent(
                        kind=StreamItemKind.STREAM_COMPLETED,
                        provider_payload=provider_payload,
                        provider_event_type=provider_event_type,
                    )
                    break
                try:
                    provider_events = self._provider_events_from_event(event)
                except Exception as exc:
                    raise StreamProviderAdapterError(
                        exc,
                        provider_payload=self._provider_payload(event),
                        provider_event_type=provider_event_type,
                    ) from exc
                for provider_event in provider_events:
                    yield provider_event
                    if (
                        provider_event.kind
                        is not StreamItemKind.REASONING_DELTA
                    ):
                        self._reasoning_segments.complete_segment()
        finally:
            await self.aclose()

    def _provider_events_from_event(
        self, event: object
    ) -> tuple[StreamProviderEvent, ...]:
        event_type = _field(event, "type")
        if event_type is not None and not isinstance(event_type, str):
            raise ValueError("anthropic event type must be a string")
        provider_payload = self._provider_payload(event)

        if event_type == "content_block_start":
            return self._content_block_start_events(
                event, provider_payload, event_type
            )
        if isinstance(event, RawContentBlockDeltaEvent):
            return self._content_block_delta_events(
                event, provider_payload, event_type
            )
        if event_type == "content_block_stop":
            return self._content_block_stop_events(
                event, provider_payload, event_type
            )
        return ()

    def _content_block_start_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        block = _field(event, "content_block")
        if _field(block, "type") != "tool_use":
            return ()
        index = _field(event, "index")
        if not isinstance(index, int):
            raise ValueError("anthropic tool block index must be an integer")
        call_id = self._tool_call_id(_field(block, "id"))
        name = _field(block, "name")
        if name is not None and not isinstance(name, str):
            raise ValueError("anthropic tool call name must be a string")
        self._canonical_tool_blocks[index] = {
            "id": call_id,
            "name": name,
            "arguments_seen": False,
        }
        return ()

    def _content_block_delta_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        delta = getattr(event, "delta")
        thinking = getattr(delta, "thinking", None)
        if isinstance(thinking, str) and thinking:
            representation = StreamReasoningRepresentation.NATIVE_TEXT
            correlation = self._reasoning_correlation(event)
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=thinking,
                    correlation=correlation,
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=representation,
                    segment_instance_ordinal=(
                        self._reasoning_segments.allocate(
                            representation, correlation
                        )
                    ),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )

        partial_json = getattr(delta, "partial_json", None)
        if partial_json is not None:
            if not isinstance(partial_json, str):
                raise ValueError(
                    "anthropic tool call arguments must be a string"
                )
            index = _field(event, "index")
            if not isinstance(index, int):
                raise ValueError(
                    "anthropic tool block index must be an integer"
                )
            block = self._canonical_tool_blocks.get(index)
            if block is None:
                raise ValueError("anthropic tool call is missing start event")
            call_id = cast(str, block["id"])
            block["arguments_seen"] = True
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    correlation=StreamItemCorrelation(tool_call_id=call_id),
                    text_delta=partial_json,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )

        text = getattr(delta, "text", None)
        if isinstance(text, str) and text:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=text,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        return ()

    def _content_block_stop_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        self._reasoning_segments.complete_segment()
        block = _field(event, "content_block")
        index = _field(event, "index")
        if not isinstance(index, int):
            raise ValueError("anthropic tool block index must be an integer")
        cached = self._canonical_tool_blocks.pop(index, None)
        if cached is None and _field(block, "type") != "tool_use":
            return ()
        name = _field(block, "name")
        if cached is not None:
            call_id = cast(str, cached["id"])
            name = name or cached.get("name")
        else:
            call_id = self._tool_call_id(_field(block, "id"))
        if name is not None and not isinstance(name, str):
            raise ValueError("anthropic tool call name must be a string")
        result = list(
            self._tool_argument_from_stop_block(
                block,
                call_id,
                cached,
                provider_payload,
                event_type,
            )
        )
        result.extend(self._mark_tool_ready(call_id, name, provider_payload))
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        self._canonical_done_tool_call_ids.add(call_id)
        return tuple(result)

    @staticmethod
    def _reasoning_correlation(event: object) -> StreamItemCorrelation:
        index = _field(event, "index")
        if index is None:
            return StreamItemCorrelation()
        if type(index) is not int or index < 0:
            raise ValueError(
                "anthropic reasoning block index must be a non-negative "
                "integer"
            )
        return StreamItemCorrelation(provider_output_index=index)

    def _tool_argument_from_stop_block(
        self,
        block: object,
        call_id: str,
        cached: dict[str, Any] | None,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
    ) -> tuple[StreamProviderEvent, ...]:
        if cached is not None and cached.get("arguments_seen"):
            return ()
        tool_input = _field(block, "input")
        if tool_input in (None, ""):
            return ()
        arguments = (
            tool_input if isinstance(tool_input, str) else to_json(tool_input)
        )
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                text_delta=arguments,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _mark_tool_ready(
        self,
        call_id: str,
        name: object | None,
        provider_payload: LooseJsonValue | None,
    ) -> tuple[StreamProviderEvent, ...]:
        if call_id in self._canonical_done_tool_call_ids:
            raise ValueError("anthropic tool call already completed")
        if call_id in self._canonical_ready_tool_call_ids:
            return ()
        self._canonical_ready_tool_call_ids.add(call_id)
        canonical_name = (
            TextGenerationVendor.canonical_tool_name(
                name,
                tool=self._tool_manager,
                provider_family=ProviderFamily.ANTHROPIC,
            )
            if isinstance(name, str)
            else name
        )
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
                provider_event_type="content_block_stop",
            ),
        )

    @staticmethod
    def _tool_call_id(value: object) -> str:
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError("anthropic tool call id must be a non-empty string")

    @staticmethod
    def _provider_payload(event: object) -> LooseJsonValue | None:
        if isinstance(event, Mapping):
            return dict(event)
        model_dump = getattr(event, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
            if isinstance(payload, Mapping):
                return dict(payload)
        return None


class AnthropicClient(TextGenerationVendor):
    _reasoning_summary_provider = "anthropic"
    _RETIRED_MODEL_REPLACEMENTS = {
        "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
        "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    }

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

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationVendorStream:
        self._validate_reasoning_summary_request(settings)
        assert (
            instructions is None
        ), "Anthropic does not support provider instructions"
        settings = settings or GenerationSettings()
        system_prompt = self._system_prompt(messages)
        template_messages = self._template_messages(
            messages,
            ["system"],
            tool=tool,
        )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "system": system_prompt,
            "messages": template_messages,
            "max_tokens": settings.max_new_tokens,
            "tools": AnthropicClient._tool_schemas(tool) if tool else [],
            "tool_choice": {"type": "auto"},
        }
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        extra_headers = AnthropicClient._extra_headers(messages)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        output_config = AnthropicClient._output_config(settings)
        if output_config:
            kwargs["output_config"] = output_config

        try:
            if use_async_generator:
                stream = self._client.messages.stream(**kwargs)
                events = await self._exit_stack.enter_async_context(stream)
                stream_kwargs: dict[str, Any] = {}
                if isinstance(tool, ToolManager):
                    stream_kwargs["tool"] = tool
                return AnthropicStream(events=events, **stream_kwargs)

            response = await self._client.messages.create(**kwargs)
            content = self._non_stream_response_content(response, tool=tool)
            return cast(
                TextGenerationVendorStream,
                TextGenerationSingleStream(
                    content,
                    provider_family=ProviderFamily.ANTHROPIC,
                    usage=getattr(response, "usage", None),
                ),
            )
        except Exception as error:
            AnthropicClient._translate_api_error(model_id, error)
            raise

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
        *,
        tool: ToolManager | None = None,
    ) -> list[TemplateMessage]:
        template_messages: list[dict[str, Any]] = []
        excluded_roles = set(exclude_roles or [])
        tool_call_ids: set[str] = set()
        last_assistant_index: int | None = None
        for message in messages:
            if str(message.role) in excluded_roles:
                continue

            if message.role == MessageRole.TOOL:
                result = (
                    message.tool_call_result
                    or message.tool_call_error
                    or message.tool_call_diagnostic
                )
                if not isinstance(
                    result,
                    (ToolCallResult, ToolCallError, ToolCallDiagnostic),
                ):
                    if not isinstance(message, Message):
                        fallback_messages = self._message_templates(
                            message, [*excluded_roles, "tool"]
                        )
                        for fallback_message in fallback_messages:
                            fallback_message["content"] = (
                                AnthropicClient._content_blocks(
                                    fallback_message.get("content")
                                )
                            )
                        template_messages.extend(fallback_messages)
                    continue
                call: ToolCall | MessageToolCall
                if isinstance(result, ToolCallDiagnostic):
                    if result.call_id is None:
                        template_messages.append(
                            {
                                "role": str(MessageRole.ASSISTANT),
                                "content": [
                                    {
                                        "type": "text",
                                        "text": to_json(
                                            tool_call_diagnostic_payload(
                                                result
                                            )
                                        ),
                                    }
                                ],
                            }
                        )
                        continue
                    call = AnthropicClient._diagnostic_tool_call(
                        message,
                        result,
                    )
                    call_id = str(result.call_id)
                else:
                    call = result.call
                    call_id = str(result.call.id)
                if call_id not in tool_call_ids:
                    tool_use_block = AnthropicClient._tool_use_block(
                        call,
                        tool=tool,
                    )
                    if last_assistant_index is not None:
                        assistant_message = template_messages[
                            last_assistant_index
                        ]
                        content = AnthropicClient._content_blocks(
                            assistant_message.get("content")
                        )
                        content.append(tool_use_block)
                        assistant_message["content"] = content
                    else:
                        template_messages.append(
                            {
                                "role": str(MessageRole.ASSISTANT),
                                "content": [tool_use_block],
                            }
                        )
                        last_assistant_index = len(template_messages) - 1
                    tool_call_ids.add(call_id)
                template_messages.append(
                    AnthropicClient._tool_result_message(result)
                )
                last_assistant_index = None
                continue

            formatted = self._message_template(message)
            if message.tool_calls:
                content = AnthropicClient._content_blocks(
                    formatted.get("content"),
                    empty_when_none=message.content is None,
                )
                for tool_call in message.tool_calls:
                    content.append(
                        AnthropicClient._tool_use_block(
                            tool_call,
                            tool=tool,
                        )
                    )
                    if tool_call.id is not None:
                        tool_call_ids.add(str(tool_call.id))
                formatted["content"] = content
            template_messages.append(formatted)
            last_assistant_index = (
                len(template_messages) - 1
                if message.role == MessageRole.ASSISTANT
                else None
            )

        # @TODO Ensure this doesn't happen from upstream
        if len(template_messages) > 1 and (
            template_messages[0] == template_messages[-1]
        ):
            template_messages.pop()

        return cast(list[TemplateMessage], template_messages)

    def _message_template(self, message: Message) -> dict[str, Any]:
        return self._message_templates(message)[0]

    def _message_templates(
        self,
        message: object,
        exclude_roles: list[TemplateMessageRole | str] | None = None,
    ) -> list[dict[str, Any]]:
        templates = cast(
            list[dict[str, Any]],
            super()._template_messages(
                cast(Any, [message]), cast(Any, exclude_roles)
            ),
        )
        for template in templates:
            content = template.get("content")
            if isinstance(content, list):
                template["content"] = [
                    AnthropicClient._content_block(block)
                    for block in content
                    if isinstance(block, dict)
                ]
        return templates

    @staticmethod
    def _content_blocks(
        content: Any, *, empty_when_none: bool = False
    ) -> list[dict[str, Any]]:
        if isinstance(content, list):
            return [block for block in content if isinstance(block, dict)]
        if empty_when_none:
            return []
        if content is None:
            return []
        return [{"type": "text", "text": str(content)}]

    @staticmethod
    def _tool_use_block(
        call: ToolCall | MessageToolCall,
        *,
        tool: ToolManager | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": str(call.id) if call.id is not None else "",
            "name": TextGenerationVendor.provider_tool_name(
                call.name,
                tool=tool,
                provider_family=ProviderFamily.ANTHROPIC,
            ),
            "input": call.arguments or {},
        }

    @staticmethod
    def _diagnostic_tool_call(
        message: Message,
        diagnostic: ToolCallDiagnostic,
    ) -> MessageToolCall:
        assert diagnostic.call_id is not None
        return MessageToolCall(
            id=str(diagnostic.call_id),
            name=(
                message.name
                or diagnostic.canonical_name
                or diagnostic.requested_name
                or "tool"
            ),
            arguments=cast(Any, message.arguments or {}),
        )

    @staticmethod
    def _tool_result_message(
        result: ToolCallResult | ToolCallError | ToolCallDiagnostic,
    ) -> dict[str, Any]:
        if isinstance(result, ToolCallDiagnostic):
            assert result.call_id is not None
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(result.call_id),
                        "content": to_json(
                            tool_call_diagnostic_payload(result)
                        ),
                        "is_error": True,
                    }
                ],
            }

        tool_result_content: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(result.call.id),
            "content": to_json(
                result.result
                if isinstance(result, ToolCallResult)
                else result.message
            ),
        }
        if isinstance(result, ToolCallError):
            tool_result_content["is_error"] = True
        return {"role": "user", "content": [tool_result_content]}

    @staticmethod
    def _output_config(
        settings: GenerationSettings,
    ) -> dict[str, str] | None:
        effort = settings.reasoning.effort
        if effort is None:
            return None
        match effort:
            case ReasoningEffort.NONE | ReasoningEffort.MINIMAL:
                resolved_effort = ReasoningEffort.LOW
            case ReasoningEffort.XHIGH:
                resolved_effort = ReasoningEffort.MAX
            case _:
                resolved_effort = effort
        return {"effort": resolved_effort.value}

    @staticmethod
    def _content_block(block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        match block_type:
            case "file":
                file = block.get("file")
                assert isinstance(file, dict), "File blocks require file data"
                return AnthropicClient._document_block(file)
            case "image_url":
                image = block.get("image_url")
                assert isinstance(
                    image, dict
                ), "Image blocks require image data"
                return {
                    "type": "image",
                    "source": AnthropicClient._image_source(image),
                }
            case "text":
                return block
            case _:
                return block

    @staticmethod
    def _document_block(file: dict[str, Any]) -> dict[str, Any]:
        source = AnthropicClient._document_source(file)
        block: dict[str, Any] = {"type": "document", "source": source}
        title = file.get("title")
        context = file.get("context")
        citations = file.get("citations")
        if isinstance(title, str):
            block["title"] = title
        if isinstance(context, str):
            block["context"] = context
        if isinstance(citations, bool):
            block["citations"] = {"enabled": citations}
        return block

    @staticmethod
    def _document_source(file: dict[str, Any]) -> dict[str, Any]:
        file_id = file.get("file_id")
        file_url = file.get("file_url") or file.get("url")
        file_data = file.get("file_data") or file.get("data")
        mime_type = file.get("mime_type")
        if isinstance(file_id, str):
            return {"type": "file", "file_id": file_id}
        if isinstance(file_url, str):
            return {"type": "url", "url": file_url}
        if isinstance(file_data, str):
            resolved_mime_type = (
                mime_type if isinstance(mime_type, str) else "application/pdf"
            )
            if resolved_mime_type == "text/plain":
                return {
                    "type": "text",
                    "media_type": resolved_mime_type,
                    "data": _decode_text_file_data(file_data),
                }
            return {
                "type": "base64",
                "media_type": resolved_mime_type,
                "data": file_data,
            }
        raise AssertionError(
            "Anthropic file blocks require file_id, file_url, or file_data"
        )

    @staticmethod
    def _image_source(image: dict[str, Any]) -> dict[str, Any]:
        file_id = image.get("file_id")
        image_url = image.get("url") or image.get("uri")
        image_data = image.get("data")
        mime_type = image.get("mime_type") or "image/png"
        if isinstance(file_id, str):
            return {"type": "file", "file_id": file_id}
        if isinstance(image_url, str):
            return {"type": "url", "url": image_url}
        if isinstance(image_data, str):
            return {
                "type": "base64",
                "media_type": mime_type,
                "data": image_data,
            }
        raise AssertionError(
            "Anthropic image blocks require file_id, url, or data"
        )

    @staticmethod
    def _extra_headers(messages: list[Message]) -> dict[str, str] | None:
        if AnthropicClient._uses_files_api(messages):
            return {"anthropic-beta": "files-api-2025-04-14"}
        return None

    @staticmethod
    def _uses_files_api(messages: list[Message]) -> bool:
        def _content_uses_file_api(
            content: object,
        ) -> bool:
            if isinstance(content, MessageContentFile):
                return isinstance(content.file.get("file_id"), str)
            if isinstance(content, MessageContentImage):
                return isinstance(content.image_url.get("file_id"), str)
            if isinstance(content, list):
                return any(_content_uses_file_api(block) for block in content)
            return False

        return any(
            _content_uses_file_api(message.content) for message in messages
        )

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        provider_ready = isinstance(tool, ToolManager)
        schemas = (
            tool.provider_json_schemas(
                provider_family=ProviderFamily.ANTHROPIC.value
            )
            if provider_ready
            else tool.json_schemas()
        )
        return (
            [
                {
                    "name": (
                        t["function"]["name"]
                        if provider_ready
                        else TextGenerationVendor.encode_tool_name(
                            t["function"]["name"]
                        )
                    ),
                    "description": t["function"]["description"],
                    "input_schema": {
                        **t["function"]["parameters"],
                        "additionalProperties": False,
                    },
                }
                for t in schemas
                if t["type"] == "function"
            ]
            if schemas
            else None
        )

    @staticmethod
    def _non_stream_response_content(
        response: object,
        *,
        tool: ToolManager | None = None,
    ) -> str:
        def _get(value: object, attribute: str) -> object | None:
            if isinstance(value, dict):
                return value.get(attribute)
            return getattr(value, attribute, None)

        parts: list[str] = []
        content_blocks = _get(response, "content")
        if not isinstance(content_blocks, list):
            return "".join(parts)
        for block in content_blocks:
            block_type = _get(block, "type")
            if block_type == "text":
                text = _get(block, "text")
                if isinstance(text, str):
                    parts.append(text)
                continue

            if block_type == "tool_use":
                provider_name = _get(block, "name")
                canonical_name = (
                    TextGenerationVendor.canonical_tool_name(
                        provider_name,
                        tool=tool,
                        provider_family=ProviderFamily.ANTHROPIC,
                    )
                    if isinstance(provider_name, str)
                    and isinstance(tool, ToolManager)
                    else provider_name
                )
                if isinstance(tool, ToolManager):
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            _get(block, "id"),
                            canonical_name,
                            _get(block, "input"),
                            tool_name_is_canonical=isinstance(
                                canonical_name, str
                            ),
                        )
                    )
                else:
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            _get(block, "id"),
                            canonical_name,
                            _get(block, "input"),
                        )
                    )

        return "".join(parts)

    @staticmethod
    def _error_message(error: Exception) -> str:
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            if isinstance(nested, dict):
                message = nested.get("message")
                if isinstance(message, str):
                    return message
        return str(error)

    @staticmethod
    def _is_missing_model_error(error: Exception) -> bool:
        if not isinstance(error, APIStatusError):
            return False
        if error.status_code != 404:
            return False
        return "model" in AnthropicClient._error_message(error).lower()

    @classmethod
    def _translate_api_error(cls, model_id: str, error: Exception) -> None:
        if not cls._is_missing_model_error(error):
            return

        message = (
            f"Anthropic model identifier {model_id!r} was not found. "
            f"Anthropic replied: {cls._error_message(error)}."
        )
        replacement = cls._RETIRED_MODEL_REPLACEMENTS.get(model_id)
        if replacement:
            message += (
                " This model has been retired by Anthropic."
                f" Use {replacement!r} instead."
            )
        else:
            message += (
                " Verify the model identifier against Anthropic's current "
                "models list."
            )
        raise ValueError(message) from error


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
