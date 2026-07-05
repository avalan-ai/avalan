from .....entities import (
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    PromptCacheRetention,
    ReasoningEffort,
    ToolCallDiagnostic,
    ToolCallResult,
)
from .....model.provider import ProviderFamily, provider_string_option
from .....model.response.text import TextGenerationResponse
from .....model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProducerBackend,
    StreamProviderAdapterError,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
    is_stream_terminal_kind,
)
from .....tool.manager import ToolManager
from .....types import (
    LooseJsonValue,
    assert_non_negative_int,
    assert_non_negative_number,
)
from .....utils import to_json, tool_call_diagnostic_payload
from ....message import TemplateMessage, TemplateMessageRole
from ....vendor import TextGenerationVendor, TextGenerationVendorStream
from . import (
    DiffusionPipeline,
    PreTrainedModel,
    TextGenerationVendorModel,
)

from asyncio import CancelledError, sleep
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from importlib import import_module
from inspect import isawaitable
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
    _INCOMPLETE_EVENTS = {"response.incomplete"}
    _stream: AsyncIterator[Any]
    _canonical_tool_calls: dict[str, dict[str, str | bool | None]]
    _tool_call_ids_by_item_id: dict[str, str]
    _canonical_ready_tool_call_ids: set[str]
    _canonical_done_tool_call_ids: set[str]
    _answer_text_seen: bool
    _answer_done_seen: bool
    _output_item_sink: Callable[[dict[str, Any]], None] | None
    _stream_factory: Callable[[], Awaitable[AsyncIterator[Any]]] | None
    _stream_retry_delay_seconds: float
    _stream_retries: int
    _tool_manager: ToolManager | None

    def __init__(
        self,
        stream: AsyncIterator[Any],
        *,
        provider_family: ProviderFamily | str = ProviderFamily.OPENAI,
        output_item_sink: Callable[[dict[str, Any]], None] | None = None,
        stream_factory: (
            Callable[[], Awaitable[AsyncIterator[Any]]] | None
        ) = None,
        stream_retry_delay_seconds: float = 0,
        stream_retries: int = 0,
        tool: ToolManager | None = None,
    ) -> None:
        self._stream = stream
        self._canonical_tool_calls = {}
        self._tool_call_ids_by_item_id = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._answer_text_seen = False
        self._answer_done_seen = False
        self._output_item_sink = output_item_sink
        self._stream_factory = stream_factory
        self._stream_retry_delay_seconds = stream_retry_delay_seconds
        self._stream_retries = stream_retries
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
            provider_family=provider_family,
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
        self._canonical_tool_calls = {}
        self._tool_call_ids_by_item_id = {}
        self._canonical_ready_tool_call_ids = set()
        self._canonical_done_tool_call_ids = set()
        self._answer_text_seen = False
        self._answer_done_seen = False
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
            attempts = 0
            while True:
                retry = False
                output_seen = False
                async for event in self._stream:
                    event_type_value = OpenAIClient._response_field(
                        event, "type"
                    )
                    provider_event_type = (
                        event_type_value
                        if isinstance(event_type_value, str)
                        else None
                    )
                    try:
                        provider_events = self._provider_events_from_event(
                            event
                        )
                    except Exception as exc:
                        raise StreamProviderAdapterError(
                            exc,
                            provider_payload=self._provider_payload(event),
                            provider_event_type=provider_event_type,
                        ) from exc
                    if self._should_retry_stream_failure(
                        event,
                        provider_events,
                        output_seen=output_seen,
                        attempts=attempts,
                    ):
                        retry = True
                        break
                    for provider_event in provider_events:
                        if self._is_model_output_event(provider_event):
                            output_seen = True
                        yield provider_event
                if not retry:
                    break
                await self._close_current_stream()
                await self._raise_if_retry_interrupted()
                assert self._stream_factory is not None
                delay = self._stream_retry_delay_seconds * (2**attempts)
                if delay > 0:
                    await sleep(delay)
                await self._raise_if_retry_interrupted()
                attempts += 1
                stream = await self._stream_factory()
                await self._raise_if_retry_interrupted(stream)
                self._stream = stream
                self._stream_sources = (self._stream,)
        finally:
            await self.aclose()

    def _should_retry_stream_failure(
        self,
        event: object,
        provider_events: tuple[StreamProviderEvent, ...],
        *,
        output_seen: bool,
        attempts: int,
    ) -> bool:
        if (
            self._stream_factory is None
            or attempts >= self._stream_retries
            or output_seen
        ):
            return False
        if OpenAIClient._response_field(event, "type") != "response.failed":
            return False
        response = OpenAIClient._response_field(event, "response")
        if OpenAIClient._response_field(response, "error") is not None:
            return False
        output = OpenAIClient._response_field(response, "output")
        if isinstance(output, Sequence) and output:
            return False
        return any(
            provider_event.kind is StreamItemKind.STREAM_ERRORED
            for provider_event in provider_events
        )

    @staticmethod
    def _is_model_output_event(event: StreamProviderEvent) -> bool:
        return event.kind not in {
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_DIAGNOSTIC,
            StreamItemKind.USAGE_COMPLETED,
        } and not is_stream_terminal_kind(event.kind)

    async def _close_current_stream(self) -> None:
        await self._call_stream_source_cleanup(self._stream, "aclose")

    async def _raise_if_retry_interrupted(
        self,
        stream: AsyncIterator[Any] | None = None,
    ) -> None:
        if not (self._stream_cancelled or self._stream_closed):
            return
        if stream is not None:
            await self._call_stream_source_cleanup(stream, "aclose")
        raise CancelledError()

    async def _call_stream_cleanup(self, method_name: str) -> None:
        assert method_name in ("cancel", "aclose")
        errors: list[Exception] = []
        for source in self._cleanup_sources():
            try:
                await self._call_stream_source_cleanup(source, method_name)
            except Exception as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("vendor stream cleanup failed", errors)

    @staticmethod
    async def _call_stream_source_cleanup(
        source: object,
        method_name: str,
    ) -> None:
        method_names = (
            ("cancel", "close", "aclose")
            if method_name == "cancel"
            else ("aclose", "close")
        )
        method = None
        for cleanup_method_name in method_names:
            method = getattr(source, cleanup_method_name, None)
            if method is not None:
                break
        else:
            return
        assert callable(method)
        result = method()
        if isawaitable(result):
            awaited_result = await cast(Awaitable[object], result)
            assert awaited_result is None
        else:
            assert result is None

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
        response = OpenAIClient._response_field(event, "response")
        error = OpenAIClient._response_field(
            event, "error"
        ) or OpenAIClient._response_field(response, "error")

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
            data = (
                self._response_failure_data(response)
                if error is None and event_type == "response.failed"
                else self._response_error_data(error or event)
            )
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data=data,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                ),
            )
        if event_type in self._INCOMPLETE_EVENTS or (
            self._response_is_incomplete(response)
        ):
            return self._incomplete_events(event, provider_payload, event_type)
        if event_type == "response.completed":
            return self._completion_events(event, provider_payload, event_type)
        if event_type in self._TEXT_DELTA_EVENTS:
            self._answer_text_seen = True
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
            text = self._response_optional_string_field(
                event, event_type, "text", "delta"
            )
            if text and not self._answer_text_seen:
                self._answer_text_seen = True
                self._answer_done_seen = True
                return (
                    StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DELTA,
                        text_delta=text,
                        provider_payload=provider_payload,
                        provider_event_type=event_type,
                    ),
                    StreamProviderEvent(
                        kind=StreamItemKind.ANSWER_DONE,
                        provider_payload=provider_payload,
                        provider_event_type=event_type,
                    ),
                )
            if not self._answer_text_seen:
                return ()
            self._answer_done_seen = True
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
            self._record_done_output_item(event)
            return self._tool_done_events(event, provider_payload, event_type)
        return ()

    def _record_done_output_item(self, event: object) -> None:
        if self._output_item_sink is None:
            return
        item = OpenAIClient._response_field(event, "item")
        payload = self._provider_payload(item)
        if not isinstance(payload, dict):
            return
        payload = self._response_input_item_payload(payload)
        item_type = payload.get("type")
        if item_type not in {"function_call", "reasoning"}:
            return
        self._output_item_sink(cast(dict[str, Any], payload))

    @classmethod
    def _response_input_item_payload(
        cls,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "status" or value is None:
                continue
            if (
                payload.get("type") == "reasoning"
                and key == "content"
                and value == []
            ):
                continue
            cleaned[key] = cls._response_input_item_value(value)
        return cleaned

    @classmethod
    def _response_input_item_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls._response_input_item_payload(value)
        if isinstance(value, list):
            return [cls._response_input_item_value(item) for item in value]
        return value

    def _incomplete_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str | None,
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
                kind=StreamItemKind.STREAM_ERRORED,
                data=self._response_incomplete_data(response or event),
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        return tuple(result)

    def _completion_events(
        self,
        event: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        response = OpenAIClient._response_field(event, "response")
        usage = OpenAIClient._response_field(response, "usage")
        result: list[StreamProviderEvent] = []
        if not self._answer_text_seen and not self._answer_done_seen:
            text = self._completed_response_text(response)
            if text:
                self._answer_text_seen = True
                self._answer_done_seen = True
                result.extend(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta=text,
                            provider_payload=provider_payload,
                            provider_event_type=event_type,
                        ),
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DONE,
                            provider_payload=provider_payload,
                            provider_event_type=event_type,
                        ),
                    )
                )
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
        item_id = self._tool_item_id_from_item(item)
        name = self._tool_call_name_from_item(item)
        if call_id is None:
            return
        self._record_tool_call_item_id(item_id, call_id)
        state = self._canonical_tool_calls.setdefault(
            call_id,
            {"name": None, "arguments_seen": False},
        )
        if name is not None:
            state["name"] = name
        if item_id is not None:
            state["protocol_item_id"] = item_id

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
                correlation=self._tool_call_correlation(event, call_id),
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
            return self._message_done_events(
                item, provider_payload, event_type
            )
        call_id = self._tool_call_id_from_item(item)
        item_id = self._tool_item_id_from_item(item)
        item_call_id = OpenAIClient._response_field(item, "call_id")
        if (
            call_id is not None
            and item_id is not None
            and call_id == item_id
            and item_call_id is None
        ):
            call_id = self._tool_call_ids_by_item_id.get(item_id, call_id)
        if call_id is None:
            call_id = self._tool_call_id_from_event(event, required=False)
        if call_id is None:
            return ()
        if item_id is not None and item_id != call_id:
            item_state = self._canonical_tool_calls.get(item_id)
            if item_state is not None and item_state.get("arguments_seen"):
                call_id = item_id
            else:
                self._record_tool_call_item_id(item_id, call_id)
        else:
            self._record_tool_call_item_id(item_id, call_id)
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
                correlation=self._tool_call_correlation(item, call_id),
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        )
        self._canonical_done_tool_call_ids.add(call_id)
        return tuple(result)

    def _message_done_events(
        self,
        item: object,
        provider_payload: LooseJsonValue | None,
        event_type: str,
    ) -> tuple[StreamProviderEvent, ...]:
        if self._answer_text_seen or self._answer_done_seen:
            return ()
        text = self._message_done_text(item)
        if not text:
            return ()

        self._answer_text_seen = True
        result = [
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=text,
                provider_payload=provider_payload,
                provider_event_type=event_type,
            )
        ]
        if not self._answer_done_seen:
            self._answer_done_seen = True
            result.append(
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DONE,
                    provider_payload=provider_payload,
                    provider_event_type=event_type,
                )
            )
        return tuple(result)

    @staticmethod
    def _completed_response_text(response: object) -> str:
        output_text = OpenAIClient._response_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts: list[str] = []
        output = OpenAIClient._response_field(response, "output")
        if isinstance(output, list):
            for item in output:
                item_type = OpenAIClient._response_field(item, "type")
                content = OpenAIClient._response_field(item, "content")
                has_text_content = isinstance(content, list) or isinstance(
                    OpenAIClient._response_field(item, "text"), str
                )
                if item_type in {"message", "output_text"} or (
                    item_type is None and has_text_content
                ):
                    parts.append(OpenAIStream._message_done_text(item))
        return "".join(parts)

    @staticmethod
    def _message_done_text(item: object) -> str:
        direct_text = OpenAIClient._response_field(item, "text")
        if isinstance(direct_text, str) and direct_text:
            return direct_text

        parts: list[str] = []
        contents = OpenAIClient._response_field(item, "content")
        if isinstance(contents, list):
            for content in contents:
                text = OpenAIClient._response_field(content, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

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
                correlation=self._tool_call_correlation(item, call_id),
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
                correlation=self._state_tool_call_correlation(call_id, state),
                data={"name": state.get("name")},
                provider_payload=provider_payload,
                provider_event_type=event_type,
            ),
        )

    def _tool_call_id_from_event(
        self, event: object, *, required: bool = True
    ) -> str | None:
        for field_name in ("call_id", "id", "item_id"):
            value = OpenAIClient._response_field(event, field_name)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                if field_name == "call_id":
                    return value
                return self._tool_call_ids_by_item_id.get(value, value)
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
            OpenAIClient._response_field(item, "call_id"),
            OpenAIClient._response_field(item, "id"),
        ):
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError(
                "response tool call id must be a non-empty string"
            )
        return None

    @staticmethod
    def _tool_item_id_from_item(item: object) -> str | None:
        if item is None:
            return None
        for field_name in ("item_id", "id"):
            value = OpenAIClient._response_field(item, field_name)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value
            raise ValueError("response tool call item id must be a string")
        return None

    def _record_tool_call_item_id(
        self,
        item_id: str | None,
        call_id: str,
    ) -> None:
        assert isinstance(call_id, str)
        assert call_id.strip()
        if item_id is None:
            return
        self._tool_call_ids_by_item_id[item_id] = call_id

    def _tool_call_correlation(
        self,
        source: object,
        call_id: str,
    ) -> StreamItemCorrelation:
        assert isinstance(call_id, str)
        assert call_id.strip()
        item_id = self._tool_item_id_from_item(source)
        return StreamItemCorrelation(
            tool_call_id=call_id,
            protocol_item_id=(
                item_id if item_id is not None and item_id != call_id else None
            ),
        )

    @staticmethod
    def _state_tool_call_correlation(
        call_id: str,
        state: dict[str, str | bool | None],
    ) -> StreamItemCorrelation:
        assert isinstance(call_id, str)
        assert call_id.strip()
        item_id = state.get("protocol_item_id")
        return StreamItemCorrelation(
            tool_call_id=call_id,
            protocol_item_id=(
                item_id
                if isinstance(item_id, str) and item_id != call_id
                else None
            ),
        )

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
                try:
                    return TextGenerationVendor.canonical_tool_name(
                        value,
                        tool=self._tool_manager,
                        provider_family=self._provider_family,
                    )
                except AssertionError:
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
    def _response_optional_string_field(
        event: object,
        event_type: str,
        *field_names: str,
    ) -> str | None:
        for field_name in field_names:
            value = OpenAIClient._response_field(event, field_name)
            if value is None:
                continue
            if isinstance(value, str):
                return value
            raise ValueError(f"{event_type} {field_name} must be a string")
        return None

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
    def _response_failure_data(response: object) -> LooseJsonValue:
        message = OpenAIClient._response_field(response, "message")
        error: dict[str, LooseJsonValue] = {
            "code": "response_failed",
            "message": (
                message
                if isinstance(message, str) and message
                else "response failed"
            ),
        }
        status = OpenAIClient._response_field(response, "status")
        if isinstance(status, str) and status:
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if isinstance(response_id, str) and response_id:
            error["response_id"] = response_id
        return {"error": error}

    @staticmethod
    def _response_is_incomplete(response: object) -> bool:
        status = OpenAIClient._response_field(response, "status")
        return status == "incomplete" or (
            OpenAIClient._response_field(response, "incomplete_details")
            is not None
        )

    @staticmethod
    def _response_incomplete_data(response: object) -> LooseJsonValue:
        details = OpenAIClient._response_field(response, "incomplete_details")
        reason = OpenAIClient._response_field(details, "reason")
        message = "response incomplete"
        error: dict[str, LooseJsonValue] = {
            "code": "response_incomplete",
            "message": message,
        }
        if isinstance(reason, str) and reason:
            error["reason"] = reason
            error["message"] = f"{message}: {reason}"
        status = OpenAIClient._response_field(response, "status")
        if isinstance(status, str) and status:
            error["status"] = status
        response_id = OpenAIClient._response_field(response, "id")
        if isinstance(response_id, str) and response_id:
            error["response_id"] = response_id
        return {"error": error}

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
    _STREAM_RESPONSE_FAILED_RETRIES = 4
    _STREAM_RESPONSE_FAILED_RETRY_DELAY_SECONDS = 1.0
    _client: Any
    _extra_query: dict[str, str] | None
    _is_azure: bool
    _stream_response_failed_retries: int
    _stream_response_failed_retry_delay_seconds: float
    _stateless_response_items: list[dict[str, Any]]

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        *,
        azure_api_version: str | None = None,
        stream_response_failed_retries: int = (
            _STREAM_RESPONSE_FAILED_RETRIES
        ),
        stream_response_failed_retry_delay_seconds: int | float = (
            _STREAM_RESPONSE_FAILED_RETRY_DELAY_SECONDS
        ),
    ):
        global Omit

        self._stream_response_failed_retries = (
            self._normalize_response_failed_retries(
                stream_response_failed_retries
            )
        )
        self._stream_response_failed_retry_delay_seconds = (
            self._normalize_response_failed_retry_delay_seconds(
                stream_response_failed_retry_delay_seconds
            )
        )
        self._is_azure = self._is_azure_base_url(base_url)
        self._extra_query = self._azure_extra_query(
            base_url, azure_api_version
        )
        self._stateless_response_items = []
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
        template_messages = self._template_messages(messages, tool=tool)
        if not self._has_function_call_context(template_messages):
            self._stateless_response_items.clear()
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
                kwargs["include"] = ["reasoning.encrypted_content"]
            prompt_cache_retention = (
                OpenAIClient._prompt_cache_retention_config(settings)
            )
            if prompt_cache_retention is not None:
                kwargs["prompt_cache_retention"] = prompt_cache_retention
        stream_response_failed_retries = OpenAIClient._response_failed_retries(
            settings,
            default=self._stream_response_failed_retries,
        )
        stream_response_failed_retry_delay_seconds = (
            OpenAIClient._response_failed_retry_delay_seconds(
                settings,
                default=(self._stream_response_failed_retry_delay_seconds),
            )
        )
        if tool:
            schemas = OpenAIClient._tool_schemas(tool)
            if schemas:
                kwargs["tools"] = schemas
                if settings and settings.tool_choice is not None:
                    kwargs["tool_choice"] = OpenAIClient._tool_choice(
                        settings.tool_choice,
                        schemas,
                        tool=tool,
                    )

        async def create_response() -> Any:
            return await self._client.responses.create(**kwargs)

        async def stream_factory() -> AsyncIterator[Any]:
            return cast(AsyncIterator[Any], await create_response())

        client_stream = await create_response()

        if use_async_generator:
            stream_kwargs: dict[str, Any] = {}
            if isinstance(tool, ToolManager):
                stream_kwargs["tool"] = tool
            return OpenAIStream(
                stream=cast(AsyncIterator[Any], client_stream),
                provider_family=self._usage_provider_family.value,
                output_item_sink=self._record_stateless_response_item,
                stream_factory=stream_factory,
                stream_retry_delay_seconds=(
                    stream_response_failed_retry_delay_seconds
                ),
                stream_retries=stream_response_failed_retries,
                **stream_kwargs,
            )

        content = OpenAIClient._non_stream_response_content(
            client_stream,
            tool=tool,
            provider_family=self._usage_provider_family,
        )
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

    @staticmethod
    def _response_failed_retries(
        settings: GenerationSettings | None,
        *,
        default: int,
    ) -> int:
        if settings is None or settings.openai_response_failed_retries is None:
            return default
        return OpenAIClient._normalize_response_failed_retries(
            settings.openai_response_failed_retries
        )

    @staticmethod
    def _response_failed_retry_delay_seconds(
        settings: GenerationSettings | None,
        *,
        default: float,
    ) -> float:
        if (
            settings is None
            or settings.openai_response_failed_retry_delay_seconds is None
        ):
            return default
        return OpenAIClient._normalize_response_failed_retry_delay_seconds(
            settings.openai_response_failed_retry_delay_seconds
        )

    @staticmethod
    def _normalize_response_failed_retries(value: object) -> int:
        assert_non_negative_int(value, "openai_response_failed_retries")
        assert isinstance(value, int)
        return value

    @staticmethod
    def _normalize_response_failed_retry_delay_seconds(
        value: object,
    ) -> float:
        assert_non_negative_number(
            value,
            "openai_response_failed_retry_delay_seconds",
        )
        assert isinstance(value, int | float)
        return float(value)

    @staticmethod
    def _provider_response_failed_retries(
        provider_options: Mapping[str, object],
    ) -> int:
        return OpenAIClient._normalize_response_failed_retries(
            provider_options["openai_response_failed_retries"]
        )

    @staticmethod
    def _provider_retry_delay_seconds(
        provider_options: Mapping[str, object],
    ) -> float:
        return OpenAIClient._normalize_response_failed_retry_delay_seconds(
            provider_options["openai_response_failed_retry_delay_seconds"]
        )

    def _template_messages(
        self,
        messages: list[Message],
        exclude_roles: list[TemplateMessageRole] | None = None,
        *,
        tool: ToolManager | None = None,
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
            [
                message
                for message in messages
                if not (
                    message.role == MessageRole.ASSISTANT
                    and message.content is None
                    and message.tool_calls
                )
            ],
            do_exclude_roles,
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
        response_item_messages = self._response_item_tool_messages(
            tool_messages,
            tool=tool,
        )
        if response_item_messages is not None:
            messages_out.extend(response_item_messages)
            return messages_out

        for tool_message in tool_messages:
            messages_out.extend(
                self._synthetic_tool_messages(tool_message, tool=tool)
            )
        return messages_out

    def _response_item_tool_messages(
        self,
        tool_messages: list[Message],
        *,
        tool: ToolManager | None = None,
    ) -> list[dict[str, Any]] | None:
        if not self._stateless_response_items:
            return None

        synthetic_records: list[tuple[str | None, list[dict[str, Any]]]] = []
        outputs_by_call_id: dict[str, dict[str, Any]] = {}
        for tool_message in tool_messages:
            synthetic = self._synthetic_tool_messages(
                tool_message,
                tool=tool,
            )
            call_id: str | None = None
            if len(synthetic) == 2:
                synthetic_call_id = synthetic[0].get("call_id")
                if isinstance(synthetic_call_id, str) and synthetic_call_id:
                    call_id = synthetic_call_id
                    outputs_by_call_id[call_id] = synthetic[1]
            synthetic_records.append((call_id, synthetic))

        if not synthetic_records:
            return []

        messages: list[dict[str, Any]] = []
        matched_call_ids: set[str] = set()
        for response_item in self._stateless_response_items:
            item_type = response_item.get("type")
            if item_type == "reasoning":
                messages.append(deepcopy(response_item))
                continue
            if item_type != "function_call":
                continue
            call_id = response_item.get("call_id")
            if not isinstance(call_id, str):
                continue
            result_message = outputs_by_call_id.get(call_id)
            if result_message is not None:
                messages.append(deepcopy(response_item))
                messages.append(deepcopy(result_message))
                matched_call_ids.add(call_id)

        for call_id, synthetic in synthetic_records:
            if call_id is not None and call_id in matched_call_ids:
                continue
            messages.extend(deepcopy(synthetic))
        return messages

    def _synthetic_tool_messages(
        self,
        tool_message: Message,
        *,
        tool: ToolManager | None = None,
    ) -> list[dict[str, Any]]:
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
                return [
                    {
                        "role": str(MessageRole.ASSISTANT),
                        "content": to_json(
                            tool_call_diagnostic_payload(outcome)
                        ),
                    }
                ]
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

        return [
            {
                "type": "function_call",
                "name": TextGenerationVendor.provider_tool_name(
                    name,
                    tool=tool,
                    provider_family=self._usage_provider_family,
                ),
                "call_id": call_id,
                "arguments": to_json(arguments),
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": to_json(output),
            },
        ]

    def _record_stateless_response_item(self, item: dict[str, Any]) -> None:
        self._stateless_response_items.append(deepcopy(item))

    @staticmethod
    def _has_function_call_context(
        messages: list[TemplateMessage] | list[dict[str, Any]],
    ) -> bool:
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("type") in {
                "function_call",
                "function_call_output",
            }:
                return True
        return False

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
        provider_ready = isinstance(tool, ToolManager)
        schemas = (
            tool.provider_json_schemas(
                provider_family=ProviderFamily.OPENAI.value
            )
            if provider_ready
            else tool.json_schemas()
        )
        return (
            [
                {
                    "type": t["type"],
                    **t["function"],
                    **(
                        {}
                        if provider_ready
                        else {
                            "name": TextGenerationVendor.encode_tool_name(
                                t["function"]["name"]
                            )
                        }
                    ),
                }
                for t in schemas
                if t["type"] == "function"
            ]
            if schemas
            else None
        )

    @staticmethod
    def _tool_choice(
        tool_choice: str,
        schemas: Sequence[Mapping[str, Any]],
        *,
        tool: ToolManager | None = None,
    ) -> dict[str, str]:
        assert tool_choice, "OpenAI tool_choice must be a tool name"
        name = TextGenerationVendor.provider_tool_name(
            tool_choice,
            tool=tool,
            provider_family=ProviderFamily.OPENAI,
        )
        schema_names = {schema.get("name") for schema in schemas}
        assert (
            name in schema_names
        ), "OpenAI tool_choice must match an available tool"
        return {"type": "function", "name": name}

    @staticmethod
    def _non_stream_response_content(
        response: object,
        *,
        tool: ToolManager | None = None,
        provider_family: ProviderFamily | str | None = ProviderFamily.OPENAI,
    ) -> str:
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
                provider_name = OpenAIClient._response_field(function, "name")
                canonical_name = (
                    TextGenerationVendor.canonical_tool_name(
                        provider_name,
                        tool=tool,
                        provider_family=provider_family,
                    )
                    if isinstance(provider_name, str)
                    and isinstance(tool, ToolManager)
                    else provider_name
                )
                call_id = OpenAIClient._response_field(call, "id")
                arguments = OpenAIClient._response_field(function, "arguments")
                if isinstance(tool, ToolManager):
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            call_id,
                            canonical_name,
                            arguments,
                            tool_name_is_canonical=isinstance(
                                canonical_name, str
                            ),
                        )
                    )
                else:
                    parts.append(
                        TextGenerationVendor.build_tool_call_text(
                            call_id,
                            canonical_name,
                            arguments,
                        )
                    )

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
        client_kwargs: dict[str, Any] = {
            "api_key": self._settings.access_token,
            "base_url": self._settings.base_url,
        }
        if azure_api_version is not None:
            client_kwargs["azure_api_version"] = azure_api_version
        provider_options = self._settings.provider_options
        if (
            provider_options is not None
            and "openai_response_failed_retries" in provider_options
        ):
            client_kwargs["stream_response_failed_retries"] = (
                OpenAIClient._provider_response_failed_retries(
                    provider_options
                )
            )
        if (
            provider_options is not None
            and "openai_response_failed_retry_delay_seconds"
            in provider_options
        ):
            retry_delay = OpenAIClient._provider_retry_delay_seconds(
                provider_options
            )
            client_kwargs["stream_response_failed_retry_delay_seconds"] = (
                retry_delay
            )
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
