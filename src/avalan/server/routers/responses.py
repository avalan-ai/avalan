from ...agent.orchestrator import Orchestrator
from ...model.stream import (
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)
from ...server.entities import ResponsesRequest
from ...utils import to_json
from .. import di_get_logger, di_get_orchestrator
from ..remote_container import validate_remote_container_profile_selection
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id
from .streaming import (
    cleanup_stream_sources,
    protocol_stream_terminal_snapshot,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from collections.abc import Mapping
from dataclasses import dataclass
from logging import Logger
from types import MappingProxyType
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

_MAX_COALESCED_DELTA_CHARS = 4096

_RESPONSE_SSE_ITEM_TYPES = {
    "reasoning_text",
    "function_call",
    "custom_tool_call_input",
    "output_text",
}
_RESPONSE_SSE_CONTENT_PART_TYPES = {
    "reasoning_text",
    "input_text",
    "output_text",
}
_RESPONSE_SSE_TOOL_ITEM_TYPES = {
    "function_call",
    "custom_tool_call_input",
}
_RESPONSE_SSE_OUTPUT_INDEX_FIELDS: Mapping[str, int] = MappingProxyType(
    {"output_index": 0}
)
_RESPONSE_SSE_CONTENT_INDEX_FIELDS: Mapping[str, int] = MappingProxyType(
    {
        "output_index": 0,
        "content_index": 0,
    }
)
_RESPONSE_SSE_UNSET = object()


def _response_sse_index_value(
    data: dict[str, Any],
    key: str,
) -> int | None:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


@dataclass(frozen=True, slots=True)
class _ResponsesSSEEvent:
    event: str
    data: dict[str, Any]
    correlation_key: str | None = None
    canonical_channel: StreamChannel | None = None

    def message(self) -> str:
        return sse_message(to_json(self.data), event=self.event)

    def can_coalesce(self, other: "_ResponsesSSEEvent") -> bool:
        assert isinstance(other, _ResponsesSSEEvent)
        self_sequence = self.data.get("sequence_number")
        other_sequence = other.data.get("sequence_number")
        if (
            not isinstance(self_sequence, int)
            or isinstance(self_sequence, bool)
            or not isinstance(other_sequence, int)
            or isinstance(other_sequence, bool)
            or other_sequence != self_sequence + 1
        ):
            return False
        if (
            self.canonical_channel is None
            or other.canonical_channel is None
            or self.canonical_channel is not other.canonical_channel
        ):
            return False
        if (
            _response_sse_index_value(self.data, "output_index") is None
            or _response_sse_index_value(other.data, "output_index") is None
            or _response_sse_index_value(self.data, "content_index") is None
            or _response_sse_index_value(other.data, "content_index") is None
        ):
            return False
        if self.event == "response.tool_execution.output" and (
            self.data.get("data") is not None
            or other.data.get("data") is not None
        ):
            return False
        return (
            self.event == other.event
            and self.correlation_key == other.correlation_key
            and self.data.get("type") == other.data.get("type")
            and self.data.get("output_index") == other.data.get("output_index")
            and self.data.get("content_index")
            == other.data.get("content_index")
            and isinstance(self.data.get("delta"), str)
            and isinstance(other.data.get("delta"), str)
            and self.event
            in {
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.custom_tool_call_input.delta",
                "response.tool_execution.output",
            }
        )

    def coalesce(self, other: "_ResponsesSSEEvent") -> "_ResponsesSSEEvent":
        assert self.can_coalesce(other)
        data = dict(self.data)
        data["delta"] = self.data["delta"] + other.data["delta"]
        data["sequence_number"] = other.data.get(
            "sequence_number", self.data.get("sequence_number")
        )
        return _ResponsesSSEEvent(
            event=self.event,
            data=data,
            correlation_key=self.correlation_key,
            canonical_channel=self.canonical_channel,
        )

    def coalesced_delta_length(self, other: "_ResponsesSSEEvent") -> int:
        assert self.can_coalesce(other)
        return len(self.data["delta"]) + len(other.data["delta"])


@dataclass(frozen=True, slots=True)
class _ResponsesSSEStreamEnvelope:
    response_id: str
    timestamp: int
    model_id: str

    def __post_init__(self) -> None:
        assert isinstance(self.response_id, str)
        assert isinstance(self.timestamp, int)
        assert not isinstance(self.timestamp, bool)
        assert isinstance(self.model_id, str)

    def created_event(self) -> _ResponsesSSEEvent:
        return _ResponsesSSEEvent(
            event="response.created",
            data={
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "created_at": self.timestamp,
                    "model": self.model_id,
                    "type": "response",
                    "status": "in_progress",
                },
            },
        )


@dataclass(frozen=True, slots=True)
class _ResponsesSSEItemState:
    output_item_type: str
    content_part_type: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        assert self.output_item_type in _RESPONSE_SSE_ITEM_TYPES
        if self.content_part_type is not None:
            assert self.content_part_type in _RESPONSE_SSE_CONTENT_PART_TYPES
        if self.tool_call_id is not None:
            assert isinstance(self.tool_call_id, str)
            assert self.tool_call_id.strip()

    @property
    def is_tool_call(self) -> bool:
        return self.output_item_type in _RESPONSE_SSE_TOOL_ITEM_TYPES


@dataclass(slots=True)
class _ResponsesSSEProjectionAdapter:
    state: _ResponsesSSEItemState | None = None

    @property
    def active_tool_call_id(self) -> str | None:
        if self.state is None or not self.state.is_tool_call:
            return None
        return self.state.tool_call_id

    def switch(self, token: StreamConsumerProjection | None) -> list[str]:
        new_state = _response_projection_state(
            token,
            self.active_tool_call_id,
        )
        events = _switch_state(self.state, new_state)
        self.state = new_state
        return events

    def close(self) -> list[str]:
        events = _switch_state(self.state, None)
        self.state = None
        return events


router = APIRouter(tags=["responses"])


@router.post(
    "/responses",
    response_model=None,
    dependencies=[Depends(validate_remote_container_profile_selection)],
)
async def create_response(
    request: ResponsesRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
) -> dict[str, Any] | StreamingResponse:
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages
    model_id = resolve_model_id(orchestrator, request.model)

    response, response_id, timestamp = await orchestrate(
        request, logger, orchestrator
    )

    if request.stream:

        async def generate() -> AsyncIterator[str]:
            adapter = _ResponsesSSEProjectionAdapter()
            stream_envelope = _ResponsesSSEStreamEnvelope(
                response_id=str(response_id),
                timestamp=timestamp,
                model_id=model_id,
            )
            pending_event: _ResponsesSSEEvent | None = None
            terminal_projection: StreamConsumerProjection | None = None
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="responses-sse-stream",
                run_id=str(response_id),
                turn_id="responses-sse-turn",
                unsupported_message=(
                    "unsupported stream item for Responses SSE projection"
                ),
                close_source_on_generator_exit=False,
            )
            cancelled = False

            def enqueue_event(event: _ResponsesSSEEvent) -> list[str]:
                nonlocal pending_event
                if pending_event is None:
                    pending_event = event
                    return []
                if (
                    pending_event.can_coalesce(event)
                    and pending_event.coalesced_delta_length(event)
                    <= _MAX_COALESCED_DELTA_CHARS
                ):
                    pending_event = pending_event.coalesce(event)
                    return []
                messages = [pending_event.message()]
                pending_event = event
                return messages

            def flush_event() -> list[str]:
                nonlocal pending_event
                if pending_event is None:
                    return []
                messages = [pending_event.message()]
                pending_event = None
                return messages

            try:
                yield stream_envelope.created_event().message()

                while True:
                    try:
                        token = await anext(iterator)
                    except StopAsyncIteration:
                        break

                    if token.is_stream_terminal:
                        terminal_projection = token
                    event_sequence = token.sequence

                    events = adapter.switch(token)
                    if events:
                        for event in flush_event():
                            yield event
                    for event in events:
                        yield event

                    for ev in _token_to_sse_events(
                        token,
                        event_sequence,
                        adapter.active_tool_call_id,
                    ):
                        for message in enqueue_event(ev):
                            yield message

                    if adapter.state is None:
                        for event in flush_event():
                            yield event

                for event in flush_event():
                    yield event

                if terminal_projection is None:
                    raise StreamValidationError(
                        "stream missing terminal outcome"
                    )

                events = adapter.close()
                for event in events:
                    yield event

                for ev in _terminal_response_events(terminal_projection):
                    yield ev.message()

                if stream_terminal_succeeded(terminal_projection):
                    await orchestrator.sync_messages()
            except CancelledError:
                cancelled = True
                raise
            finally:
                await cleanup_stream_sources(
                    response,
                    iterator,
                    cancelled=cancelled,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=sse_headers(),
        )

    text = await response.to_str()
    body = {
        "id": str(response_id),
        "created": timestamp,
        "model": model_id,
        "type": "response",
        "output": [{"content": [{"type": "output_text", "text": text}]}],
        "usage": {
            "input_text_tokens": response.input_token_count,
            "output_text_tokens": response.output_token_count,
            "total_tokens": (
                response.input_token_count + response.output_token_count
            ),
        },
    }
    await orchestrator.sync_messages()
    return body


def _terminal_response_events(
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
) -> list[_ResponsesSSEEvent]:
    terminal_snapshot = protocol_stream_terminal_snapshot(terminal)
    terminal_outcome = terminal_snapshot.outcome
    if (
        terminal_outcome is None
        or terminal_outcome is StreamTerminalOutcome.COMPLETED
    ):
        data: dict[str, Any] = {"type": "response.completed"}
        if terminal_snapshot.sequence is not None:
            data["sequence_number"] = terminal_snapshot.sequence
        return [
            _ResponsesSSEEvent(
                event="response.completed",
                data=data,
            )
        ]
    if terminal_outcome is StreamTerminalOutcome.CANCELLED:
        data = {"type": "response.cancelled"}
        if terminal_snapshot.sequence is not None:
            data["sequence_number"] = terminal_snapshot.sequence
        return [
            _ResponsesSSEEvent(
                event="response.cancelled",
                data=data,
            )
        ]
    data = {"type": "response.failed"}
    if terminal_snapshot.sequence is not None:
        data["sequence_number"] = terminal_snapshot.sequence
        if terminal_snapshot.data is not None:
            data["error"] = terminal_snapshot.data
    return [
        _ResponsesSSEEvent(
            event="response.failed",
            data=data,
        )
    ]


def _token_to_sse(token: StreamConsumerProjection, seq: int) -> list[str]:
    assert isinstance(token, StreamConsumerProjection)
    return [event.message() for event in _token_to_sse_events(token, seq)]


def _token_to_sse_events(
    token: StreamConsumerProjection,
    seq: int,
    active_tool_call_id: str | None = None,
) -> list[_ResponsesSSEEvent]:
    assert isinstance(token, StreamConsumerProjection)
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    return _canonical_item_to_sse_events(token, seq, active_tool_call_id)


def _stream_tool_call_protocol_id(
    item: StreamConsumerProjection,
) -> str | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    return item.tool_call_id


def _canonical_item_to_sse(
    item: StreamConsumerProjection, seq: int
) -> list[str]:
    return [
        event.message() for event in _canonical_item_to_sse_events(item, seq)
    ]


def _canonical_item_to_sse_events(
    item: StreamConsumerProjection,
    seq: int,
    active_tool_call_id: str | None = None,
) -> list[_ResponsesSSEEvent]:
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    if item.kind is StreamItemKind.REASONING_DELTA:
        return [
            _ResponsesSSEEvent(
                event="response.reasoning_text.delta",
                data=_response_sse_delta_data(
                    "response.reasoning_text.delta",
                    item.text_delta or "",
                    seq,
                ),
                canonical_channel=item.channel,
            )
        ]
    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        function_call = _projection_function_call_delta(item)
        if function_call is not None:
            return [
                _ResponsesSSEEvent(
                    event="response.function_call_arguments.delta",
                    data=_response_sse_delta_data(
                        "response.function_call_arguments.delta",
                        to_json(function_call),
                        seq,
                        id_value=function_call["id"],
                    ),
                    correlation_key=str(function_call["id"]),
                    canonical_channel=item.channel,
                )
            ]
        protocol_id = (
            _stream_tool_call_protocol_id(item) or active_tool_call_id
        )
        data = _response_sse_delta_data(
            "response.custom_tool_call_input.delta",
            item.text_delta or "",
            seq,
        )
        if protocol_id is not None:
            data["id"] = protocol_id
        return [
            _ResponsesSSEEvent(
                event="response.custom_tool_call_input.delta",
                data=data,
                correlation_key=protocol_id,
                canonical_channel=item.channel,
            )
        ]
    if item.kind is StreamItemKind.ANSWER_DELTA:
        return [
            _ResponsesSSEEvent(
                event="response.output_text.delta",
                data=_response_sse_delta_data(
                    "response.output_text.delta",
                    item.text_delta or "",
                    seq,
                ),
                canonical_channel=item.channel,
            )
        ]
    if item.kind in (
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        event = (
            "response.usage.completed"
            if item.kind is StreamItemKind.USAGE_COMPLETED
            else "response.usage.delta"
        )
        return [
            _ResponsesSSEEvent(
                event=event,
                data={
                    "type": event,
                    "usage": item.usage,
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    if item.kind is StreamItemKind.STREAM_COMPLETED and item.usage is not None:
        return [
            _ResponsesSSEEvent(
                event="response.usage.completed",
                data={
                    "type": "response.usage.completed",
                    "usage": item.usage,
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    if item.kind in {
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    }:
        return [_tool_execution_sse_event(item, seq)]
    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
        return [
            _ResponsesSSEEvent(
                event="response.diagnostic",
                data={
                    "type": "response.diagnostic",
                    "delta": item.text_delta,
                    "data": item.data,
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    return []


def _tool_execution_sse_event(
    item: StreamConsumerProjection, seq: int
) -> _ResponsesSSEEvent:
    event_names = {
        StreamItemKind.TOOL_EXECUTION_STARTED: (
            "response.tool_execution.started"
        ),
        StreamItemKind.TOOL_EXECUTION_OUTPUT: "response.tool_execution.output",
        StreamItemKind.TOOL_EXECUTION_PROGRESS: (
            "response.tool_execution.progress"
        ),
        StreamItemKind.TOOL_EXECUTION_COMPLETED: (
            "response.tool_execution.completed"
        ),
        StreamItemKind.TOOL_EXECUTION_ERROR: "response.tool_execution.error",
        StreamItemKind.TOOL_EXECUTION_CANCELLED: (
            "response.tool_execution.cancelled"
        ),
    }
    event = event_names[item.kind]
    data: dict[str, Any] = {
        "type": event,
        "id": item.tool_call_id,
        "sequence_number": seq,
    }
    if item.text_delta is not None:
        data["delta"] = item.text_delta
    if item.data is not None:
        data["data"] = item.data
    return _ResponsesSSEEvent(
        event=event,
        data=data,
        correlation_key=item.tool_call_id,
        canonical_channel=item.channel,
    )


def _projection_function_call_delta(
    item: StreamConsumerProjection,
) -> dict[str, Any] | None:
    if not isinstance(item.data, dict):
        return None
    name = item.data.get("name")
    if not isinstance(name, str):
        return None
    return {
        "id": item.tool_call_id,
        "name": name,
        "arguments": item.data.get("arguments"),
    }


def _response_sse_delta_data(
    event: str,
    delta: str,
    seq: int,
    *,
    id_value: object = _RESPONSE_SSE_UNSET,
) -> dict[str, Any]:
    assert isinstance(event, str)
    assert event.strip()
    assert isinstance(delta, str)
    assert isinstance(seq, int)
    assert not isinstance(seq, bool)
    data: dict[str, Any] = {"type": event, "delta": delta}
    if id_value is not _RESPONSE_SSE_UNSET:
        data["id"] = id_value
    data.update(_RESPONSE_SSE_CONTENT_INDEX_FIELDS)
    data["sequence_number"] = seq
    return data


def _response_sse_indexed_data(
    event: str,
    *,
    content_index: bool = False,
) -> dict[str, Any]:
    assert isinstance(event, str)
    assert event.strip()
    data: dict[str, Any] = {"type": event}
    data.update(
        _RESPONSE_SSE_CONTENT_INDEX_FIELDS
        if content_index
        else _RESPONSE_SSE_OUTPUT_INDEX_FIELDS
    )
    return data


def _switch_state(
    state: _ResponsesSSEItemState | None,
    new_state: _ResponsesSSEItemState | None,
) -> list[str]:
    events: list[str] = []
    if state != new_state:
        if state is not None and state.output_item_type == "reasoning_text":
            events.append(_reasoning_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())
        elif state is not None and state.output_item_type == "function_call":
            events.append(_function_call_arguments_done(state.tool_call_id))
            events.append(_output_item_done(state.tool_call_id))
        elif (
            state is not None
            and state.output_item_type == "custom_tool_call_input"
        ):
            events.append(_custom_tool_call_input_done(state.tool_call_id))
            events.append(_content_part_done(state.tool_call_id))
            events.append(_output_item_done(state.tool_call_id))
        elif state is not None and state.output_item_type == "output_text":
            events.append(_output_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())

        if new_state is not None:
            events.append(_output_item_added(new_state))
            if new_state.content_part_type is not None:
                events.append(
                    _content_part_added(
                        new_state.content_part_type,
                        new_state.tool_call_id,
                    )
                )

    return events


def _response_projection_state(
    token: StreamConsumerProjection | None,
    active_tool_call_id: str | None = None,
) -> _ResponsesSSEItemState | None:
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    if isinstance(token, StreamConsumerProjection):
        if token.kind is StreamItemKind.REASONING_DELTA:
            return _ResponsesSSEItemState(
                output_item_type="reasoning_text",
                content_part_type="reasoning_text",
            )
        elif token.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            tool_call_id = (
                _stream_tool_call_protocol_id(token) or active_tool_call_id
            )
            if _projection_function_call_delta(token) is not None:
                return _ResponsesSSEItemState(
                    output_item_type="function_call",
                    tool_call_id=tool_call_id,
                )
            return _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id=tool_call_id,
            )
        elif token.kind is StreamItemKind.ANSWER_DELTA:
            return _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="output_text",
            )
        return None
    elif token is None:
        return None
    raise AssertionError("unsupported response stream item")


def _output_item_added(state: _ResponsesSSEItemState) -> str:
    item = {"type": state.output_item_type}
    if state.tool_call_id is not None:
        item["id"] = state.tool_call_id
    data = _response_sse_indexed_data("response.output_item.added")
    data["item"] = item
    return sse_message(
        to_json(data),
        event="response.output_item.added",
    )


def _output_item_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data("response.output_item.done")
    if id is not None:
        data["item"] = {"id": id}
    return sse_message(to_json(data), event="response.output_item.done")


def _function_call_arguments_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data("response.function_call_arguments.done")
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.function_call_arguments.done",
    )


def _reasoning_text_done() -> str:
    return sse_message(
        to_json(
            _response_sse_indexed_data(
                "response.reasoning_text.done",
                content_index=True,
            )
        ),
        event="response.reasoning_text.done",
    )


def _custom_tool_call_input_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data(
        "response.custom_tool_call_input.done",
        content_index=True,
    )
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.custom_tool_call_input.done",
    )


def _output_text_done() -> str:
    return sse_message(
        to_json(
            _response_sse_indexed_data(
                "response.output_text.done",
                content_index=True,
            )
        ),
        event="response.output_text.done",
    )


def _content_part_added(part_type: str, id: str | None = None) -> str:
    part = {"type": part_type}
    if id is not None:
        part["id"] = id
    data = _response_sse_indexed_data(
        "response.content_part.added",
        content_index=True,
    )
    data["part"] = part
    return sse_message(
        to_json(data),
        event="response.content_part.added",
    )


def _content_part_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data(
        "response.content_part.done",
        content_index=True,
    )
    if id is not None:
        data["part"] = {"id": id}
    return sse_message(to_json(data), event="response.content_part.done")
