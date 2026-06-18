from .....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    PromptCacheRetention,
    ReasoningEffort,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallDiagnostic,
    ToolCallResult,
    ToolCallToken,
)
from .....model.provider import ProviderFamily, provider_string_option
from .....model.response.text import TextGenerationResponse
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
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
)

from collections.abc import AsyncIterator, Mapping
from importlib import import_module
from mimetypes import guess_type
from typing import Any, cast
from urllib.parse import urlparse


class _OmitPlaceholder:  # noqa: D101
    pass


Omit: type[Any] = _OmitPlaceholder


class OpenAIStream(TextGenerationVendorStream):
    _TEXT_DELTA_EVENTS = {"response.text.delta", "response.output_text.delta"}
    _TEXT_DONE_EVENTS = {"response.text.done", "response.output_text.done"}
    _REASONING_DELTA_EVENTS = {"response.reasoning_text.delta"}
    _REASONING_DONE_EVENTS = {"response.reasoning_text.done"}
    _TOOL_CALL_ITEM_TYPES = {
        "custom_tool_call",
        "function_call",
        "tool_call",
    }
    _TOOL_ARGUMENT_DELTA_EVENTS = {
        "response.custom_tool_call_input.delta",
        "response.function_call_arguments.delta",
    }
    _TOOL_ARGUMENT_DONE_EVENTS = {
        "response.custom_tool_call_input.done",
        "response.function_call_arguments.done",
    }
    _ERROR_EVENTS = {"response.error", "response.failed", "error"}
    _CANCELLED_EVENTS = {"response.cancelled", "response.canceled"}
    _stream: AsyncIterator[Any]
    _canonical_tool_calls: dict[str, dict[str, str | bool | None]]
    _canonical_ready_tool_call_ids: set[str]
    _canonical_done_tool_call_ids: set[str]

    def __init__(
        self,
        stream: AsyncIterator[Any],
        *,
        provider_family: ProviderFamily | str = ProviderFamily.OPENAI,
    ) -> None:
        self._stream = stream
        self._canonical_tool_calls = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()

        async def generator() -> AsyncIterator[Token | TokenDetail | str]:
            tool_calls: dict[str, dict[str, str | list[str] | None]] = {}
            terminal_usage: object | None = None

            async for event in self._stream:
                etype = OpenAIClient._response_field(event, "type")

                if etype == "response.completed":
                    response = OpenAIClient._response_field(event, "response")
                    usage = OpenAIClient._response_field(response, "usage")
                    if usage is not None:
                        terminal_usage = usage
                    continue

                if etype == "response.output_item.added":
                    item = OpenAIClient._response_field(event, "item")
                    if not self._is_tool_call_item(item):
                        continue
                    try:
                        call_id = self._tool_call_id_from_item(item)
                    except ValueError:
                        continue
                    if call_id is None:
                        continue
                    tool_calls[call_id] = {
                        "name": self._tool_call_name_from_item(item),
                        "args_fragments": [],
                    }
                    continue

                if (
                    etype == "response.custom_tool_call_input.delta"
                    or etype == "response.function_call_arguments.delta"
                ):
                    call_id = self._tool_call_id_from_event(
                        event, required=False
                    )
                    delta = getattr(event, "delta", None)
                    if isinstance(call_id, str) and isinstance(delta, str):
                        tc = tool_calls.setdefault(
                            call_id, {"name": None, "args_fragments": []}
                        )
                        args_fragments = tc["args_fragments"]
                        assert isinstance(args_fragments, list)
                        args_fragments.append(delta)
                        yield ToolCallToken(token=delta)
                    continue

                if etype in self._REASONING_DELTA_EVENTS:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        yield ReasoningToken(token=delta)
                    continue

                if etype in self._TEXT_DELTA_EVENTS:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        yield Token(token=delta)
                    continue

                if etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    call_id = self._tool_call_id_from_item(item)
                    if call_id is None:
                        call_id = self._tool_call_id_from_event(
                            event, required=False
                        )
                    cached = (
                        tool_calls.pop(call_id, None)
                        if isinstance(call_id, str)
                        else None
                    )
                    if cached:
                        name = self._tool_call_name_from_item(item)
                        if name is not None:
                            cached["name"] = name
                        args_fragments = cached["args_fragments"]
                        assert isinstance(args_fragments, list)
                        yield TextGenerationVendor.build_tool_call_token(
                            call_id,
                            cached.get("name"),
                            "".join(args_fragments) or None,
                        )
                    elif (
                        item is not None
                        and getattr(item, "type", None) == "function_call"
                    ):
                        tool_name = getattr(item, "name", None)
                        tool_id = getattr(item, "id", None)

                        if tool_id and tool_name:
                            token = TextGenerationVendor.build_tool_call_token(
                                tool_id,
                                tool_name,
                                getattr(item, "arguments", None),
                            )
                            yield token

                    continue

            if terminal_usage is not None:
                self._usage = terminal_usage

        super().__init__(
            generator(),
            provider_family=provider_family,
            sources=(stream,),
        )

    async def __anext__(self) -> CanonicalStreamItem:
        return await super().__anext__()

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
        self._canonical_tool_calls = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
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
        try:
            async for event in self._stream:
                for provider_event in self._provider_events_from_event(event):
                    yield provider_event
        finally:
            await self.aclose()

    def _provider_events_from_event(
        self, event: object
    ) -> tuple[StreamProviderEvent, ...]:
        event_type_value = OpenAIClient._response_field(event, "type")
        if event_type_value is not None and not isinstance(
            event_type_value, str
        ):
            raise ValueError("response event type must be a string")
        event_type = event_type_value
        provider_payload = self._provider_payload(event)
        error = OpenAIClient._response_field(
            event, "error"
        ) or OpenAIClient._response_field(
            OpenAIClient._response_field(event, "response"), "error"
        )

        if event_type in self._CANCELLED_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_CANCELLED,
                    data=self._response_event_data(event),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._ERROR_EVENTS or error is not None:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data=self._response_error_data(error or event),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type == "response.completed":
            return self._completion_events(event, provider_payload, event_type)
        if event_type in self._TEXT_DELTA_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=self._response_string_field(
                        event, "delta", event_type
                    ),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._TEXT_DONE_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DONE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._REASONING_DELTA_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DELTA,
                    text_delta=self._response_string_field(
                        event, "delta", event_type
                    ),
                    visibility=StreamVisibility.PRIVATE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._REASONING_DONE_EVENTS:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.REASONING_DONE,
                    visibility=StreamVisibility.PRIVATE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type == "response.output_item.added":
            self._record_output_item(event)
            return ()
        if event_type in self._TOOL_ARGUMENT_DELTA_EVENTS:
            return self._tool_argument_delta_events(
                event, provider_payload, event_type
            )
        if event_type in self._TOOL_ARGUMENT_DONE_EVENTS:
            return self._tool_ready_events(event, provider_payload, event_type)
        if event_type == "response.output_item.done":
            return self._tool_done_events(event, provider_payload, event_type)
        return ()

    def _completion_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        response = OpenAIClient._response_field(event, "response")
        usage = OpenAIClient._response_field(response, "usage")
        result: list[StreamProviderEvent] = []
        if usage is not None:
            self._usage = usage
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.USAGE_COMPLETED,
                    usage=cast(LooseJsonValue, usage),
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        result.append(
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_COMPLETED,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        return tuple(result)

    def _record_output_item(self, event: object) -> None:
        item = OpenAIClient._response_field(event, "item")
        if not self._is_tool_call_item(item):
            return
        call_id = self._tool_call_id_from_item(item)
        name = self._tool_call_name_from_item(item)
        if call_id is None:
            return
        self._canonical_tool_calls[call_id] = {
            "name": name,
            "arguments_seen": False,
        }

    def _tool_argument_delta_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        call_id = self._tool_call_id_from_event(event)
        assert call_id is not None
        delta = self._response_string_field(event, "delta", event_type)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        state["arguments_seen"] = True
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                text_delta=delta,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _tool_ready_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        call_id = self._tool_call_id_from_event(event)
        assert call_id is not None
        return self._mark_tool_ready(call_id, provider_payload, event_type)

    def _tool_done_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        item = OpenAIClient._response_field(event, "item")
        if item is not None and not self._is_tool_call_item(item):
            return ()
        call_id = self._tool_call_id_from_item(item)
        if call_id is None:
            call_id = self._tool_call_id_from_event(event, required=False)
        if call_id is None:
            return ()
        if call_id not in self._canonical_tool_calls:
            pending_call_ids = [
                pending_call_id
                for pending_call_id, state in (
                    self._canonical_tool_calls.items()
                )
                if state.get("arguments_seen")
                and pending_call_id not in self._canonical_done_tool_call_ids
            ]
            if pending_call_ids:
                raise StreamValidationError(
                    "response tool call item id "
                    f"{call_id} does not match pending tool call item "
                    f"{pending_call_ids[0]}"
                )
        if call_id in self._canonical_done_tool_call_ids:
            raise ValueError("response tool call already completed")
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {
                "name": self._tool_call_name_from_item(item),
                "arguments_seen": False,
            },
        )
        name = self._tool_call_name_from_item(item)
        if name is not None:
            state["name"] = name

        result = list(
            self._tool_argument_from_done_item(
                item, call_id, provider_payload, event_type
            )
        )
        result.extend(
            self._mark_tool_ready(call_id, provider_payload, event_type)
        )
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

    def _is_tool_call_item(self, item: object) -> bool:
        if item is None:
            return False
        if OpenAIClient._response_field(item, "custom_tool_call") is not None:
            return True
        item_type = OpenAIClient._response_field(item, "type")
        return item_type is None or item_type in self._TOOL_CALL_ITEM_TYPES

    def _tool_argument_from_done_item(
        self,
        item: object,
        call_id: str,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        if state["arguments_seen"]:
            return ()
        arguments = self._tool_call_arguments_from_item(item)
        if arguments is None:
            return ()
        state["arguments_seen"] = True
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
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        if call_id in self._canonical_ready_tool_call_ids:
            return ()
        self._canonical_ready_tool_call_ids.add(call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                correlation=StreamItemCorrelation(tool_call_id=call_id),
                data={"name": state.get("name")},
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _tool_call_id_from_event(
        self, event: object, *, required: bool = True
    ) -> str | None:
        for field_name in ("id", "call_id", "item_id"):
            value = OpenAIClient._response_field(event, field_name)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError(
                "response tool call id must be a non-empty string"
            )
        if required:
            raise ValueError("response tool call id is missing")
        return None

    def _tool_call_id_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        for value in (
            OpenAIClient._response_field(custom, "id"),
            OpenAIClient._response_field(item, "id"),
            OpenAIClient._response_field(item, "call_id"),
        ):
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError(
                "response tool call id must be a non-empty string"
            )
        return None

    def _tool_call_name_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        function = OpenAIClient._response_field(item, "function") or item
        for value in (
            OpenAIClient._response_field(custom, "name"),
            OpenAIClient._response_field(function, "name"),
        ):
            if value is None:
                continue
            if isinstance(value, str):
                return value
            raise ValueError("response tool call name must be a string")
        return None

    def _tool_call_arguments_from_item(self, item: object) -> str | None:
        if item is None:
            return None
        custom = OpenAIClient._response_field(item, "custom_tool_call")
        for value in (
            OpenAIClient._response_field(item, "arguments"),
            OpenAIClient._response_field(custom, "input"),
        ):
            if value is None:
                continue
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                return to_json(value)
            raise ValueError("response tool call arguments must be a string")
        return None

    @staticmethod
    def _response_string_field(
        event: object, field_name: str, event_type: str
    ) -> str:
        value = OpenAIClient._response_field(event, field_name)
        if isinstance(value, str):
            return value
        raise ValueError(f"{event_type} {field_name} must be a string")

    @staticmethod
    def _response_event_data(event: object) -> LooseJsonValue:
        reason = OpenAIClient._response_field(event, "reason")
        if isinstance(reason, str):
            return {"reason": reason}
        return {}

    @staticmethod
    def _response_error_data(error: object) -> LooseJsonValue:
        if isinstance(error, Mapping):
            return {"error": dict(error)}
        message = OpenAIClient._response_field(error, "message")
        if isinstance(message, str):
            return {"error": {"message": message}}
        return {"error": {"message": str(error)}}

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


class OpenAIClient(TextGenerationVendor):
    _DEFAULT_MODEL_ID = "default"
    _client: Any
    _extra_query: dict[str, str] | None
    _is_azure: bool

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        *,
        azure_api_version: str | None = None,
    ):
        global Omit

        self._is_azure = self._is_azure_base_url(base_url)
        self._extra_query = self._azure_extra_query(
            base_url, azure_api_version
        )
        if self._is_azure and api_key is None:
            raise AssertionError(
                "Azure OpenAI Responses requires api-key authentication"
            )

        openai_module = import_module("openai")
        async_openai_type = getattr(openai_module, "AsyncOpenAI")
        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is None:
            assert base_url
            Omit = cast(type[Any], getattr(openai_module, "Omit"))
            client_kwargs.update(
                api_key="",
                default_headers=cast(Any, {"Authorization": Omit()}),
            )
        else:
            client_kwargs["api_key"] = api_key
        self._client = async_openai_type(**client_kwargs)

    async def __call__(
        self,
        model_id: str,
        messages: list[Message],
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        timeout: int | None = None,
        tool: ToolManager | None = None,
        use_async_generator: bool = True,
    ) -> TextGenerationStream:
        template_messages = self._template_messages(messages)
        use_reasoning_profile = self._uses_reasoning_profile(model_id)
        kwargs: dict[str, Any] = {
            "extra_headers": {
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            "model": model_id or self._DEFAULT_MODEL_ID,
            "input": template_messages,
            "store": False,
            "stream": use_async_generator,
            "timeout": timeout,
        }
        if instructions is not None:
            assert isinstance(
                instructions, str
            ), "OpenAI Responses instructions must be a string"
            kwargs["instructions"] = instructions
        if self._extra_query is not None:
            kwargs["extra_query"] = self._extra_query
        if settings:
            if settings.max_new_tokens is not None:
                kwargs["max_output_tokens"] = settings.max_new_tokens
            if settings.temperature is not None and not use_reasoning_profile:
                kwargs["temperature"] = settings.temperature
            if settings.top_p is not None and not use_reasoning_profile:
                kwargs["top_p"] = settings.top_p
            text = OpenAIClient._text_config(settings)
            if text:
                kwargs["text"] = text
            reasoning = OpenAIClient._reasoning_config(settings)
            if reasoning:
                kwargs["reasoning"] = reasoning
            prompt_cache_retention = (
                OpenAIClient._prompt_cache_retention_config(settings)
            )
            if prompt_cache_retention is not None:
                kwargs["prompt_cache_retention"] = prompt_cache_retention
        if tool:
            schemas = OpenAIClient._tool_schemas(tool)
            if schemas:
                kwargs["tools"] = schemas
        client_stream = await self._client.responses.create(**kwargs)

        if use_async_generator:
            return OpenAIStream(
                stream=client_stream,
                provider_family=self._usage_provider_family,
            )

        content = OpenAIClient._non_stream_response_content(client_stream)
        return TextGenerationSingleStream(
            content,
            provider_family=self._usage_provider_family,
            usage=OpenAIClient._response_field(client_stream, "usage"),
        )

    @property
    def _usage_provider_family(self) -> ProviderFamily:
        return (
            ProviderFamily.AZURE_OPENAI
            if self._is_azure
            else ProviderFamily.OPENAI
        )

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
    ) -> list[TemplateMessage] | list[dict[str, Any]]:
        tool_messages = [
            message
            for message in messages
            if message.role == MessageRole.TOOL
            and (
                message.tool_call_result
                or message.tool_call_error
                or message.tool_call_diagnostic
            )
        ]
        do_exclude_roles = [*(exclude_roles or []), "tool"]
        template_messages = super()._template_messages(
            messages, do_exclude_roles
        )
        messages_out = cast(list[dict[str, Any]], template_messages)
        for message in messages_out:
            if "content" not in message:
                continue
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    OpenAIClient._content_block(block)
                    for block in content
                    if isinstance(block, dict)
                ]
        for tool_message in tool_messages:
            outcome = (
                tool_message.tool_call_result
                or tool_message.tool_call_error
                or tool_message.tool_call_diagnostic
            )
            assert outcome is not None

            output: Any
            if isinstance(outcome, ToolCallDiagnostic):
                call_id_value = outcome.call_id
                if call_id_value is None:
                    messages_out.append(
                        {
                            "role": str(MessageRole.ASSISTANT),
                            "content": to_json(
                                tool_call_diagnostic_payload(outcome)
                            ),
                        }
                    )
                    continue
                call_id = str(call_id_value)
                name = (
                    tool_message.name
                    or outcome.canonical_name
                    or outcome.requested_name
                    or "tool"
                )
                arguments = tool_message.arguments
                output = tool_call_diagnostic_payload(outcome)
            else:
                call_id = str(outcome.call.id)
                name = outcome.call.name
                arguments = outcome.call.arguments
                output = (
                    outcome.result
                    if isinstance(outcome, ToolCallResult)
                    else {"error": outcome.message}
                )

            call_message = {
                "type": "function_call",
                "name": TextGenerationVendor.encode_tool_name(name),
                "call_id": call_id,
                "arguments": to_json(arguments),
            }
            messages_out.append(call_message)

            result_message = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": to_json(output),
            }
            messages_out.append(result_message)
        return messages_out

    @staticmethod
    def _reasoning_config(
        settings: GenerationSettings,
    ) -> dict[str, str] | None:
        effort = settings.reasoning.effort
        if effort is None or effort == ReasoningEffort.NONE:
            return None
        assert isinstance(
            effort, ReasoningEffort
        ), "OpenAI Responses reasoning effort is not supported"
        if effort == ReasoningEffort.MAX:
            effort = ReasoningEffort.XHIGH
        return {"effort": effort.value}

    @staticmethod
    def _text_config(settings: GenerationSettings) -> dict[str, Any]:
        text: dict[str, Any] = {}
        if settings.response_format is not None:
            text["format"] = OpenAIClient._response_text_format(
                settings.response_format
            )
        if settings.stop_strings is not None:
            text["stop"] = settings.stop_strings
        return text

    @staticmethod
    def _prompt_cache_retention_config(
        settings: GenerationSettings,
    ) -> str | None:
        retention = settings.prompt_cache_retention
        if retention is None:
            return None
        if isinstance(retention, PromptCacheRetention):
            return retention.value
        assert isinstance(
            retention, str
        ), "OpenAI prompt cache retention must be a string"
        assert retention in {
            item.value for item in PromptCacheRetention
        }, "OpenAI prompt cache retention is not supported"
        return retention

    @staticmethod
    def _response_text_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        assert isinstance(response_format, dict)
        format_type = response_format.get("type")
        match format_type:
            case "text" | "json_object":
                return {"type": format_type}
            case "json_schema":
                return OpenAIClient._json_schema_format(response_format)
            case _:
                raise AssertionError(
                    "OpenAI Responses response format is not supported"
                )

    @staticmethod
    def _json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        has_chat_schema = "json_schema" in response_format
        has_responses_schema = "schema" in response_format
        if has_chat_schema == has_responses_schema:
            raise AssertionError(
                "OpenAI Responses json_schema format is ambiguous"
            )
        if has_chat_schema:
            return OpenAIClient._chat_json_schema_format(response_format)
        return OpenAIClient._responses_json_schema_format(response_format)

    @staticmethod
    def _chat_json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        json_schema = response_format["json_schema"]
        assert isinstance(json_schema, dict)
        schema = json_schema.get("schema")
        assert isinstance(schema, dict)
        name = json_schema.get("name") or schema.get("title") or "response"
        assert isinstance(name, str) and name
        output: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if "strict" in json_schema:
            strict = json_schema["strict"]
            assert isinstance(strict, bool)
            output["strict"] = strict
        return output

    @staticmethod
    def _responses_json_schema_format(
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        schema = response_format["schema"]
        name = response_format.get("name")
        assert isinstance(schema, dict)
        assert isinstance(name, str) and name
        output: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if "strict" in response_format:
            strict = response_format["strict"]
            assert isinstance(strict, bool)
            output["strict"] = strict
        return output

    @staticmethod
    def _is_azure_base_url(base_url: str | None) -> bool:
        if not isinstance(base_url, str):
            return False
        host = urlparse(base_url).hostname or ""
        return host.endswith(".openai.azure.com") or host.endswith(
            ".cognitiveservices.azure.com"
        )

    @staticmethod
    def _azure_extra_query(
        base_url: str | None,
        azure_api_version: str | None,
    ) -> dict[str, str] | None:
        is_azure = OpenAIClient._is_azure_base_url(base_url)
        parsed = urlparse(base_url or "")
        if is_azure and parsed.query:
            raise AssertionError(
                "Azure OpenAI base_url must not include query parameters"
            )
        if azure_api_version is None:
            if is_azure and not parsed.path.rstrip("/").endswith("/openai/v1"):
                raise AssertionError(
                    "Azure OpenAI Responses base_url must end with /openai/v1/"
                )
            return None
        assert isinstance(azure_api_version, str) and azure_api_version.strip()
        if not is_azure:
            raise AssertionError(
                "azure_api_version is only supported for Azure OpenAI"
            )
        return {"api-version": azure_api_version}

    def _uses_reasoning_profile(self, model_id: str) -> bool:
        normalized = (model_id or self._DEFAULT_MODEL_ID).lower()
        return (
            self._is_azure
            or normalized.startswith("gpt-5")
            or (
                len(normalized) > 1
                and normalized[0] == "o"
                and normalized[1].isdigit()
            )
        )

    @staticmethod
    def _content_block(block: dict[str, Any]) -> dict[str, Any]:
        block_type = block.get("type")
        match block_type:
            case "file":
                file = block.get("file")
                assert isinstance(file, dict), "File blocks require file data"
                return OpenAIClient._file_block(file)
            case "image_url":
                image = block.get("image_url")
                assert isinstance(
                    image, dict
                ), "Image blocks require image data"
                return OpenAIClient._image_block(image)
            case "text":
                text = block.get("text")
                assert isinstance(text, str), "Text blocks require text"
                return {"type": "input_text", "text": text}
            case _:
                return block

    @staticmethod
    def _file_block(file: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_file"}
        file_id = file.get("file_id")
        file_url = file.get("file_url") or file.get("url")
        file_data = file.get("file_data") or file.get("data")
        filename = file.get("filename")
        mime_type = file.get("mime_type")

        if isinstance(file_id, str):
            payload["file_id"] = file_id
        elif isinstance(file_url, str):
            payload["file_url"] = file_url
        elif isinstance(file_data, str):
            file_mime_type = (
                mime_type
                if isinstance(mime_type, str)
                else (
                    guess_type(filename)[0]
                    if isinstance(filename, str)
                    else None
                )
            )
            payload["file_data"] = (
                file_data
                if file_data.startswith("data:")
                or not isinstance(file_mime_type, str)
                else f"data:{file_mime_type};base64,{file_data}"
            )
        else:
            raise AssertionError(
                "OpenAI file blocks require file_id, file_url, or file_data"
            )

        if isinstance(filename, str):
            payload["filename"] = filename
        return payload

    @staticmethod
    def _image_block(image: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_image"}
        file_id = image.get("file_id")
        image_url = image.get("url") or image.get("uri")
        image_data = image.get("data")
        mime_type = image.get("mime_type") or "image/png"
        detail = image.get("detail")

        if isinstance(file_id, str):
            payload["file_id"] = file_id
        elif isinstance(image_url, str):
            payload["image_url"] = image_url
        elif isinstance(image_data, str):
            payload["image_url"] = OpenAIClient._image_data_url(
                image_data,
                mime_type,
            )
        else:
            raise AssertionError(
                "OpenAI image blocks require file_id, url, or data"
            )

        if isinstance(detail, str):
            payload["detail"] = detail
        return payload

    @staticmethod
    def _image_data_url(image_data: str, mime_type: object) -> str:
        if image_data.startswith("data:"):
            return image_data
        assert isinstance(
            mime_type, str
        ), "OpenAI image blocks require an image MIME type"
        assert mime_type.startswith(
            "image/"
        ), "OpenAI image blocks require an image MIME type"
        return f"data:{mime_type};base64,{image_data}"

    @staticmethod
    def _tool_schemas(tool: ToolManager) -> list[dict[str, Any]] | None:
        schemas = tool.json_schemas()
        return (
            [
                {
                    "type": t["type"],
                    **t["function"],
                    **{
                        "name": TextGenerationVendor.encode_tool_name(
                            t["function"]["name"]
                        )
                    },
                }
                for t in schemas
                if t["type"] == "function"
            ]
            if schemas
            else None
        )

    @staticmethod
    def _non_stream_response_content(response: object) -> str:
        parts: list[str] = []
        output = OpenAIClient._response_field(response, "output")
        if not isinstance(output, list):
            return "".join(parts)

        for item in output:
            item_type = OpenAIClient._response_field(item, "type")
            contents = OpenAIClient._response_field(item, "content")
            if not isinstance(contents, list):
                contents = []

            if item_type in {None, "message", "output_text"}:
                for content in contents:
                    text = OpenAIClient._response_field(content, "text")
                    if isinstance(text, str):
                        parts.append(text)
                continue

            if item_type in {"tool_call", "function_call"}:
                call = OpenAIClient._response_field(item, "call") or item
                function = (
                    OpenAIClient._response_field(call, "function") or call
                )
                token = TextGenerationVendor.build_tool_call_token(
                    OpenAIClient._response_field(call, "id"),
                    OpenAIClient._response_field(function, "name"),
                    OpenAIClient._response_field(function, "arguments"),
                )
                parts.append(token.token)

        return "".join(parts)

    @staticmethod
    def _response_field(value: object, attribute: str) -> object | None:
        if isinstance(value, dict):
            return value.get(attribute)
        return getattr(value, attribute, None)


class OpenAINonStreamingResponse(TextGenerationResponse):
    _static_response_text: str | None

    def __init__(
        self,
        *args: Any,
        static_response_text: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._static_response_text = static_response_text

    def __str__(self) -> str:
        if self._static_response_text is not None:
            return self._static_response_text

        buffered = self._buffer.getvalue()
        if buffered is not None:
            return buffered

        return object.__repr__(self)

    async def to_str(self) -> str:
        text = await super().to_str()
        self._static_response_text = text
        return text


class OpenAIModel(TextGenerationVendorModel):
    def _load_model(
        self,
    ) -> PreTrainedModel | TextGenerationVendor | DiffusionPipeline:
        assert self._settings.base_url or self._settings.access_token
        azure_api_version = provider_string_option(
            self._settings.provider_options,
            "azure_api_version",
        )
        client_kwargs: dict[str, str | None] = {
            "api_key": self._settings.access_token,
            "base_url": self._settings.base_url,
        }
        if azure_api_version is not None:
            client_kwargs["azure_api_version"] = azure_api_version
        return OpenAIClient(**client_kwargs)

    async def __call__(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        settings: GenerationSettings | None = None,
        *,
        instructions: str | None = None,
        tool: ToolManager | None = None,
    ) -> TextGenerationResponse:
        generation_settings = settings or GenerationSettings()
        messages = self._messages(input, system_prompt, developer_prompt, tool)
        streamer = await self._model(
            self._model_id,
            messages,
            generation_settings,
            instructions=instructions,
            tool=tool,
            use_async_generator=generation_settings.use_async_generator,
        )

        if generation_settings.use_async_generator:
            return TextGenerationResponse(
                streamer,
                logger=self._logger,
                generation_settings=generation_settings,
                settings=generation_settings,
                use_async_generator=True,
            )

        static_text: str | None = None
        if isinstance(streamer, TextGenerationSingleStream):
            content = streamer.content
            static_text = content if isinstance(content, str) else None

        return OpenAINonStreamingResponse(
            streamer,
            logger=self._logger,
            generation_settings=generation_settings,
            settings=generation_settings,
            use_async_generator=False,
            static_response_text=static_text,
        )
