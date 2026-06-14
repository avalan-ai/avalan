from ...agent.orchestrator import Orchestrator
from ...entities import (
    Token,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallResult,
)
from ...event import Event, EventType
from ...model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    stream_consumer_projection_from_token,
)
from ...server.entities import ResponsesRequest
from ...utils import (
    to_json,
    tool_call_diagnostic_payload,
    tool_call_error_payload,
)
from .. import di_get_logger, di_get_orchestrator
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id
from .streaming import (
    cleanup_stream_sources,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from dataclasses import dataclass
from enum import Enum, auto
from logging import Logger
from typing import Any, AsyncIterator, cast

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse


class ResponseState(Enum):
    REASONING = auto()
    FUNCTION_CALLING = auto()
    CUSTOM_TOOL_CALLING = auto()
    ANSWERING = auto()


_MAX_COALESCED_DELTA_CHARS = 4096


@dataclass(frozen=True, slots=True)
class _ResponsesSSEEvent:
    event: str
    data: dict[str, Any]
    correlation_key: str | None = None

    def message(self) -> str:
        return sse_message(to_json(self.data), event=self.event)

    def can_coalesce(self, other: "_ResponsesSSEEvent") -> bool:
        assert isinstance(other, _ResponsesSSEEvent)
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
        )

    def coalesced_delta_length(self, other: "_ResponsesSSEEvent") -> int:
        assert self.can_coalesce(other)
        return len(self.data["delta"]) + len(other.data["delta"])


router = APIRouter(tags=["responses"])


@router.post("/responses", response_model=None)
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
            seq = 0
            state: ResponseState | None = None
            tool_call_id: str | None = None
            canonical_accumulator: CanonicalStreamAccumulator | None = None
            legacy_stream_seen = False
            pending_event: _ResponsesSSEEvent | None = None
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="responses-sse-stream",
                run_id=response_id,
                turn_id="responses-sse-turn",
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
                sync_messages = True
                yield sse_message(
                    to_json(
                        {
                            "type": "response.created",
                            "response": {
                                "id": str(response_id),
                                "created_at": timestamp,
                                "model": model_id,
                                "type": "response",
                                "status": "in_progress",
                            },
                        }
                    ),
                    event="response.created",
                )

                while True:
                    try:
                        raw_token = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    token: StreamConsumerProjection | Event
                    call_id: str | None = None
                    if isinstance(raw_token, CanonicalStreamItem):
                        if legacy_stream_seen:
                            raise StreamValidationError(
                                "canonical stream item after legacy stream"
                                " item"
                            )
                        if canonical_accumulator is None:
                            canonical_accumulator = (
                                CanonicalStreamAccumulator()
                            )
                        canonical_accumulator.add(raw_token)
                        token = _stream_projection(raw_token, seq)
                        call_id = _stream_tool_call_protocol_id(token)
                    elif isinstance(raw_token, StreamConsumerProjection):
                        if legacy_stream_seen:
                            raise StreamValidationError(
                                "canonical stream item after legacy stream"
                                " item"
                            )
                        if canonical_accumulator is None:
                            canonical_accumulator = (
                                CanonicalStreamAccumulator()
                            )
                        canonical_accumulator.add(
                            canonical_item_from_consumer_projection(raw_token)
                        )
                        token = raw_token
                        call_id = _stream_tool_call_protocol_id(token)
                    elif isinstance(raw_token, Event):
                        token = raw_token
                        if raw_token.type not in (
                            EventType.TOOL_DIAGNOSTIC,
                            EventType.TOOL_PROCESS,
                            EventType.TOOL_RESULT,
                        ):
                            continue
                        item = _tool_call_event_item(raw_token)
                        if item is None:
                            continue
                        item_id = item.get("id")
                        call_id = item_id if isinstance(item_id, str) else None
                    else:
                        if canonical_accumulator is not None:
                            raise StreamValidationError(
                                "legacy stream item after canonical stream"
                                " item"
                            )
                        legacy_stream_seen = True
                        projection = _stream_projection(raw_token, seq)
                        token = projection
                        call_id = _stream_tool_call_protocol_id(projection)
                    event_sequence = (
                        token.sequence
                        if isinstance(token, StreamConsumerProjection)
                        else seq
                    )

                    new_state = _new_state(token)
                    new_tool_call_id = call_id
                    if new_tool_call_id is None and _is_tool_response_state(
                        new_state
                    ):
                        new_tool_call_id = tool_call_id
                    events = _switch_state(
                        state, new_state, tool_call_id, new_tool_call_id
                    )
                    if events:
                        for event in flush_event():
                            yield event
                    state = new_state
                    if _is_tool_response_state(state):
                        if call_id is not None:
                            tool_call_id = call_id
                    else:
                        tool_call_id = None
                    for event in events:
                        yield event

                    for ev in _token_to_sse_events(
                        token, event_sequence, tool_call_id
                    ):
                        for message in enqueue_event(ev):
                            yield message

                    if _new_state(token) is None:
                        for event in flush_event():
                            yield event

                    seq += 1

                for event in flush_event():
                    yield event

                terminal_projection = None
                if canonical_accumulator is not None:
                    canonical_accumulator.validate_complete()
                    terminal_projection = _terminal_projection(
                        canonical_accumulator
                    )
                    sync_messages = stream_terminal_succeeded(
                        terminal_projection
                    )

                events = _switch_state(state, None, tool_call_id, None)
                for event in events:
                    yield event

                for ev in _terminal_response_events(terminal_projection):
                    yield ev.message()

                yield sse_message("{}", event="done")
                if sync_messages:
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
    assert terminal is None or isinstance(
        terminal, (StreamConsumerProjection, StreamTerminalOutcome)
    )
    terminal_outcome = (
        terminal.terminal_outcome
        if isinstance(terminal, StreamConsumerProjection)
        else terminal
    )
    if (
        terminal_outcome is None
        or terminal_outcome is StreamTerminalOutcome.COMPLETED
    ):
        data: dict[str, Any] = {"type": "response.completed"}
        if isinstance(terminal, StreamConsumerProjection):
            assert terminal.is_stream_terminal
            data["sequence_number"] = terminal.sequence
        return [
            _ResponsesSSEEvent(
                event="response.completed",
                data=data,
            )
        ]
    if terminal_outcome is StreamTerminalOutcome.CANCELLED:
        data = {"type": "response.cancelled"}
        if isinstance(terminal, StreamConsumerProjection):
            assert terminal.is_stream_terminal
            data["sequence_number"] = terminal.sequence
        return [
            _ResponsesSSEEvent(
                event="response.cancelled",
                data=data,
            )
        ]
    data = {"type": "response.failed"}
    if isinstance(terminal, StreamConsumerProjection):
        assert terminal.is_stream_terminal
        data["sequence_number"] = terminal.sequence
        if terminal.data is not None:
            data["error"] = terminal.data
    return [
        _ResponsesSSEEvent(
            event="response.failed",
            data=data,
        )
    ]


def _terminal_projection(
    accumulator: CanonicalStreamAccumulator,
) -> StreamConsumerProjection | None:
    assert isinstance(accumulator, CanonicalStreamAccumulator)
    for item in reversed(accumulator.items):
        if item.is_stream_terminal:
            return _stream_projection(item, item.sequence)
    return None


def _token_to_sse(
    token: StreamConsumerProjection | Event,
    seq: int,
) -> list[str]:
    assert isinstance(token, (StreamConsumerProjection, Event))
    return [event.message() for event in _token_to_sse_events(token, seq)]


def _token_to_sse_events(
    token: StreamConsumerProjection | Event,
    seq: int,
    active_tool_call_id: str | None = None,
) -> list[_ResponsesSSEEvent]:
    assert isinstance(token, (StreamConsumerProjection, Event))
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    events: list[_ResponsesSSEEvent] = []

    if isinstance(token, StreamConsumerProjection):
        events.extend(
            _canonical_item_to_sse_events(token, seq, active_tool_call_id)
        )
    elif isinstance(token, Event) and token.type in (
        EventType.TOOL_DIAGNOSTIC,
        EventType.TOOL_PROCESS,
        EventType.TOOL_RESULT,
    ):
        item = _tool_call_event_item(token)
        if item is None:
            return events
        delta_obj = {
            "id": item["id"],
            "name": item["name"],
            "arguments": item.get("arguments"),
        }
        if "diagnostic" in item:
            delta_obj["diagnostic"] = item["diagnostic"]
        elif token.type is EventType.TOOL_RESULT:
            if "error" in item:
                delta_obj["error"] = item.get("error")
            else:
                delta_obj["result"] = (
                    to_json(item["result"])
                    if "result" in item and item["result"] is not None
                    else None
                )

        event_type = (
            "response.tool_call_diagnostic.delta"
            if "diagnostic" in item
            else "response.function_call_arguments.delta"
        )
        events.append(
            _ResponsesSSEEvent(
                event=event_type,
                data={
                    **delta_obj,
                    "type": event_type,
                    "delta": to_json(delta_obj),
                    "id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
                correlation_key=str(item["id"]),
            )
        )
    return events


def _stream_projection(
    token: CanonicalStreamItem | StreamConsumerProjection | Token | str,
    seq: int,
) -> StreamConsumerProjection:
    return stream_consumer_projection_from_token(token, seq)


def _stream_tool_call_protocol_id(
    item: StreamConsumerProjection,
) -> str | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    if item.tool_call_id == "legacy-tool-call":
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
                data={
                    "type": "response.reasoning_text.delta",
                    "delta": item.text_delta or "",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
            )
        ]
    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        function_call = _projection_function_call_delta(item)
        if function_call is not None:
            return [
                _ResponsesSSEEvent(
                    event="response.function_call_arguments.delta",
                    data={
                        "type": "response.function_call_arguments.delta",
                        "delta": to_json(function_call),
                        "id": function_call["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": seq,
                    },
                    correlation_key=str(function_call["id"]),
                )
            ]
        data: dict[str, Any] = {
            "type": "response.custom_tool_call_input.delta",
            "delta": item.text_delta or "",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": seq,
        }
        protocol_id = (
            _stream_tool_call_protocol_id(item) or active_tool_call_id
        )
        if protocol_id is not None:
            data["id"] = protocol_id
        return [
            _ResponsesSSEEvent(
                event="response.custom_tool_call_input.delta",
                data=data,
                correlation_key=protocol_id,
            )
        ]
    if item.kind is StreamItemKind.ANSWER_DELTA:
        return [
            _ResponsesSSEEvent(
                event="response.output_text.delta",
                data={
                    "type": "response.output_text.delta",
                    "delta": item.text_delta or "",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
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


def _switch_state(
    state: ResponseState | None,
    new_state: ResponseState | None,
    current_tool_call_id: str | None,
    new_tool_call_id: str | None,
) -> list[str]:
    events: list[str] = []
    changed = state is not new_state or (
        state is new_state
        and _is_tool_response_state(state)
        and new_tool_call_id is not None
        and current_tool_call_id != new_tool_call_id
    )
    if changed:
        if state is ResponseState.REASONING:
            events.append(_reasoning_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())
        elif state is ResponseState.FUNCTION_CALLING:
            events.append(_function_call_arguments_done(current_tool_call_id))
            events.append(_output_item_done(current_tool_call_id))
        elif state is ResponseState.CUSTOM_TOOL_CALLING:
            events.append(_custom_tool_call_input_done(current_tool_call_id))
            events.append(_content_part_done(current_tool_call_id))
            events.append(_output_item_done(current_tool_call_id))
        elif state is ResponseState.ANSWERING:
            events.append(_output_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())

        if new_state is ResponseState.REASONING:
            events.append(_output_item_added(new_state))
            events.append(_content_part_added("reasoning_text"))
        elif new_state is ResponseState.FUNCTION_CALLING:
            events.append(_output_item_added(new_state, new_tool_call_id))
        elif new_state is ResponseState.CUSTOM_TOOL_CALLING:
            events.append(_output_item_added(new_state, new_tool_call_id))
            events.append(_content_part_added("input_text", new_tool_call_id))
        elif new_state is ResponseState.ANSWERING:
            events.append(_output_item_added(new_state))
            events.append(_content_part_added("output_text"))

    return events


def _new_state(
    token: StreamConsumerProjection | Event | None,
) -> ResponseState | None:
    if isinstance(token, StreamConsumerProjection):
        if token.kind is StreamItemKind.REASONING_DELTA:
            new_state = ResponseState.REASONING
        elif token.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            if _projection_function_call_delta(token) is not None:
                new_state = ResponseState.FUNCTION_CALLING
            else:
                new_state = ResponseState.CUSTOM_TOOL_CALLING
        elif token.kind is StreamItemKind.ANSWER_DELTA:
            new_state = ResponseState.ANSWERING
        else:
            new_state = None
    elif isinstance(token, Event) and token.type in (
        EventType.TOOL_DIAGNOSTIC,
        EventType.TOOL_PROCESS,
        EventType.TOOL_RESULT,
    ):
        new_state = ResponseState.FUNCTION_CALLING
    elif token is None:
        new_state = None
    else:
        raise AssertionError("unsupported response stream item")
    return new_state


def _is_tool_response_state(state: ResponseState | None) -> bool:
    return state in {
        ResponseState.FUNCTION_CALLING,
        ResponseState.CUSTOM_TOOL_CALLING,
    }


def _output_item_added(state: ResponseState, id: str | None = None) -> str:
    item_types = {
        ResponseState.REASONING: "reasoning_text",
        ResponseState.FUNCTION_CALLING: "function_call",
        ResponseState.CUSTOM_TOOL_CALLING: "custom_tool_call_input",
        ResponseState.ANSWERING: "output_text",
    }
    item = {"type": item_types[state]}
    if id is not None:
        item["id"] = id
    return sse_message(
        to_json(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": item,
            }
        ),
        event="response.output_item.added",
    )


def _output_item_done(id: str | None = None) -> str:
    data = {"type": "response.output_item.done", "output_index": 0}
    if id is not None:
        data["item"] = {"id": id}
    return sse_message(to_json(data), event="response.output_item.done")


def _function_call_arguments_done(id: str | None = None) -> str:
    data = {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
    }
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.function_call_arguments.done",
    )


def _reasoning_text_done() -> str:
    return sse_message(
        to_json(
            {
                "type": "response.reasoning_text.done",
                "output_index": 0,
                "content_index": 0,
            }
        ),
        event="response.reasoning_text.done",
    )


def _custom_tool_call_input_done(id: str | None = None) -> str:
    data = {
        "type": "response.custom_tool_call_input.done",
        "output_index": 0,
        "content_index": 0,
    }
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.custom_tool_call_input.done",
    )


def _output_text_done() -> str:
    return sse_message(
        to_json(
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
            }
        ),
        event="response.output_text.done",
    )


def _content_part_added(part_type: str, id: str | None = None) -> str:
    part = {"type": part_type}
    if id is not None:
        part["id"] = id
    return sse_message(
        to_json(
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": part,
            }
        ),
        event="response.content_part.added",
    )


def _content_part_done(id: str | None = None) -> str:
    data = {
        "type": "response.content_part.done",
        "output_index": 0,
        "content_index": 0,
    }
    if id is not None:
        data["part"] = {"id": id}
    return sse_message(to_json(data), event="response.content_part.done")


def _tool_call_event_item(event: Event) -> dict[str, Any] | None:
    payload = cast(Any, event.payload)
    tool_result = (
        payload["result"]
        if event.type == EventType.TOOL_RESULT
        and isinstance(payload, dict)
        and "result" in payload
        else None
    )
    if isinstance(tool_result, ToolCallDiagnostic):
        call = payload.get("call") if isinstance(payload, dict) else None
        return _tool_call_diagnostic_item(
            tool_result,
            call if isinstance(call, ToolCall) else None,
        )
    if event.type is EventType.TOOL_DIAGNOSTIC:
        diagnostic = (
            payload.get("diagnostic") if isinstance(payload, dict) else None
        )
        if not isinstance(diagnostic, ToolCallDiagnostic):
            diagnostics = (
                payload.get("diagnostics")
                if isinstance(payload, dict)
                else None
            )
            if isinstance(diagnostics, list):
                diagnostic = next(
                    (
                        item
                        for item in diagnostics
                        if isinstance(item, ToolCallDiagnostic)
                    ),
                    None,
                )
        if not isinstance(diagnostic, ToolCallDiagnostic):
            return None
        call = payload.get("call") if isinstance(payload, dict) else None
        return _tool_call_diagnostic_item(
            diagnostic,
            call if isinstance(call, ToolCall) else None,
        )
    if tool_result is not None:
        tool_call = tool_result.call
    elif isinstance(payload, list):
        tool_call = payload[0]
    elif isinstance(payload, dict):
        tool_call = payload.get("call") or payload.get(0)
    else:
        tool_call = None
    if tool_call is None:
        return None
    item = {
        "type": "function_call",
        "id": str(tool_call.id),
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }
    if tool_result is not None:
        if isinstance(tool_result, ToolCallError):
            item["error"] = tool_call_error_payload(tool_result)
        elif isinstance(tool_result, ToolCallResult):
            item["result"] = tool_result.result
        else:
            item["result"] = tool_result
    return item


def _tool_call_diagnostic_item(
    diagnostic: ToolCallDiagnostic, call: ToolCall | None
) -> dict[str, Any]:
    diagnostic_payload = {
        "id": str(diagnostic.id),
        **tool_call_diagnostic_payload(diagnostic),
    }
    if diagnostic.call_id is not None:
        diagnostic_payload["call_id"] = str(diagnostic.call_id)
    if diagnostic.started_at is not None:
        diagnostic_payload["started_at"] = diagnostic.started_at.isoformat()
    if diagnostic.finished_at is not None:
        diagnostic_payload["finished_at"] = diagnostic.finished_at.isoformat()
    if diagnostic.duration_ms is not None:
        diagnostic_payload["duration_ms"] = diagnostic.duration_ms
    return {
        "type": "function_call",
        "id": str(call.id if call else diagnostic.call_id or diagnostic.id),
        "name": (
            call.name
            if call
            else diagnostic.canonical_name
            or diagnostic.requested_name
            or "tool"
        ),
        "arguments": call.arguments if call else None,
        "diagnostic": diagnostic_payload,
    }
