from . import orchestrate
from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import (
    ReasoningToken,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    Token,
    TokenDetail,
)
from ...event import Event, EventType
from ...server.entities import ResponsesRequest
from ...utils import to_json
from enum import Enum, auto
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from logging import Logger


class ResponseState(Enum):
    REASONING = auto()
    TOOL_CALLING = auto()
    ANSWERING = auto()


router = APIRouter(tags=["responses"])


@router.post("/responses")
async def create_response(
    request: ResponsesRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
):
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages

    response, response_id, timestamp = await orchestrate(
        request, logger, orchestrator
    )

    if request.stream:

        async def generate():
            seq = 0

            yield _sse(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": str(response_id),
                        "created_at": timestamp,
                        "model": request.model,
                        "type": "response",
                        "status": "in_progress",
                    },
                },
            )

            state: ResponseState | None = None
            tool_call_id: str | None = None

            async for token in response:
                is_event = isinstance(token, Event)
                if is_event and token.type not in (
                    EventType.TOOL_PROCESS,
                    EventType.TOOL_RESULT,
                ):
                    continue

                call_id: str | None = None
                if is_event and token.type in (
                    EventType.TOOL_PROCESS,
                    EventType.TOOL_RESULT,
                ):
                    call_id = _tool_call_event_item(token)["id"]
                elif (
                    isinstance(token, ToolCallToken) and token.call is not None
                ):
                    call_id = str(token.call.id)

                new_state = _new_state(token)
                events = _switch_state(state, new_state, tool_call_id, call_id)
                state = new_state
                if state is ResponseState.TOOL_CALLING:
                    if call_id is not None:
                        tool_call_id = call_id
                else:
                    tool_call_id = None
                for event in events:
                    yield event

                for ev in _token_to_sse(token, seq):
                    yield ev

                seq += 1

            events = _switch_state(state, None, tool_call_id, None)
            for event in events:
                yield event

            yield _sse("response.completed", {"type": "response.completed"})

            yield "event: done\ndata: {}\n\n"
            await orchestrator.sync_messages()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    text = await response.to_str()
    body = {
        "id": str(response_id),
        "created": timestamp,
        "model": request.model,
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


def _token_to_sse(
    token: ReasoningToken | ToolCallToken | Token | TokenDetail | Event | str,
    seq: int,
) -> list[str]:
    events: list[str] = []

    if isinstance(token, ReasoningToken):
        events.append(
            _sse(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "delta": token.token,
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
            )
        )
    elif isinstance(token, Event) and token.type in (
        EventType.TOOL_PROCESS,
        EventType.TOOL_RESULT,
    ):
        item = _tool_call_event_item(token)
        delta_obj = {
            "id": item["id"],
            "name": item["name"],
            "arguments": item.get("arguments"),
        }
        if token.type is EventType.TOOL_RESULT:
            if "error" in item:
                delta_obj["error"] = to_json(item.get("error"))
            else:
                result = item.get("result", None)
                delta_obj["result"] = to_json(result) if result else None

        events.append(
            _sse(
                "response.function_call_arguments.delta",
                {
                    **delta_obj,
                    "type": "response.function_call_arguments.delta",
                    "delta": to_json(delta_obj),
                    "id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
            )
        )
    elif isinstance(token, ToolCallToken):
        if token.call is not None:
            delta_obj = {
                "id": str(token.call.id),
                "name": token.call.name,
                "arguments": token.call.arguments,
            }
            events.append(
                _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "delta": to_json(delta_obj),
                        "id": str(token.call.id),
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": seq,
                    },
                )
            )
        else:
            events.append(
                _sse(
                    "response.custom_tool_call_input.delta",
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "delta": token.token,
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": seq,
                    },
                )
            )
    else:
        events.append(
            _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "delta": (
                        token.token if isinstance(token, Token) else str(token)
                    ),
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": seq,
                },
            )
        )
    return events


def _switch_state(
    state: ResponseState | None,
    new_state: ResponseState | None,
    current_tool_call_id: str | None,
    new_tool_call_id: str | None,
) -> list[str]:
    new_state: ResponseState | None

    events: list[str] = []
    changed = state is not new_state or (
        state is ResponseState.TOOL_CALLING
        and new_state is ResponseState.TOOL_CALLING
        and new_tool_call_id is not None
    )
    if changed:
        if state is ResponseState.REASONING:
            events.append(_reasoning_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())
        elif state is ResponseState.TOOL_CALLING:
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
        elif new_state is ResponseState.TOOL_CALLING:
            events.append(_output_item_added(new_state, new_tool_call_id))
            events.append(_content_part_added("input_text", new_tool_call_id))
        elif new_state is ResponseState.ANSWERING:
            events.append(_output_item_added(new_state))
            events.append(_content_part_added("output_text"))

    return events


def _new_state(
    token: ReasoningToken | ToolCallToken | Token | TokenDetail | str | None,
) -> ResponseState | None:
    if isinstance(token, ReasoningToken):
        new_state = ResponseState.REASONING
    elif isinstance(token, (ToolCallToken, Event)) and (
        not isinstance(token, Event)
        or token.type in (EventType.TOOL_PROCESS, EventType.TOOL_RESULT)
    ):
        new_state = ResponseState.TOOL_CALLING
    elif token is not None:
        new_state = ResponseState.ANSWERING
    else:
        new_state = None
    return new_state


def _output_item_added(state: ResponseState, id: str | None = None) -> str:
    item_types = {
        ResponseState.REASONING: "reasoning_text",
        ResponseState.TOOL_CALLING: "custom_tool_call_input",
        ResponseState.ANSWERING: "output_text",
    }
    item = {"type": item_types[state]}
    if id is not None:
        item["id"] = id
    return _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
        },
    )


def _output_item_done(id: str | None = None) -> str:
    data = {"type": "response.output_item.done", "output_index": 0}
    if id is not None:
        data["item"] = {"id": id}
    return _sse("response.output_item.done", data)


def _reasoning_text_done() -> str:
    return _sse(
        "response.reasoning_text.done",
        {
            "type": "response.reasoning_text.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _custom_tool_call_input_done(id: str | None = None) -> str:
    data = {
        "type": "response.custom_tool_call_input.done",
        "output_index": 0,
        "content_index": 0,
    }
    if id is not None:
        data["id"] = id
    return _sse("response.custom_tool_call_input.done", data)


def _output_text_done() -> str:
    return _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _content_part_added(part_type: str, id: str | None = None) -> str:
    part = {"type": part_type}
    if id is not None:
        part["id"] = id
    return _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": part,
        },
    )


def _content_part_done(id: str | None = None) -> str:
    data = {
        "type": "response.content_part.done",
        "output_index": 0,
        "content_index": 0,
    }
    if id is not None:
        data["part"] = {"id": id}
    return _sse("response.content_part.done", data)


def _tool_call_event_item(event: Event) -> dict:
    tool_result = (
        event.payload["result"]
        if event.type == EventType.TOOL_RESULT and "result" in event.payload
        else None
    )
    tool_call = (
        tool_result.call if tool_result is not None else event.payload[0]
    )
    item = {
        "type": "function_call",
        "id": str(tool_call.id),
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }
    if tool_result is not None:
        if isinstance(tool_result, ToolCallError):
            item["error"] = tool_result.message
        elif isinstance(tool_result, ToolCallResult):
            item["result"] = tool_result.result
        else:
            item["result"] = tool_result
    return item


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\n" + f"data: {to_json(data)}\n\n"
