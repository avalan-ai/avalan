from . import orchestrate
from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import ReasoningToken, ToolCallToken, Token, TokenDetail
from ...event import Event
from ...server.entities import ResponsesRequest
from enum import Enum, auto
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from json import dumps
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

            async for token in response:
                if isinstance(token, Event):
                    continue

                state, events = _switch_state(state, token)
                for event in events:
                    yield event

                yield _token_to_sse(token, seq)

                seq += 1

            _, events = _switch_state(state, None)
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
    token: ReasoningToken | ToolCallToken | Token | TokenDetail | str, seq: int
) -> str:
    result: str | None = None

    if isinstance(token, ReasoningToken):
        result = _sse(
            "response.reasoning_text.delta",
            {
                "type": "response.reasoning_text.delta",
                "delta": token.token,
                "output_index": 0,
                "content_index": 0,
                "sequence_number": seq,
            },
        )
    elif isinstance(token, ToolCallToken):
        result = _sse(
            "response.custom_tool_call_input.delta",
            {
                "type": "response.custom_tool_call_input.delta",
                "delta": token.token,
                "output_index": 0,
                "content_index": 0,
                "sequence_number": seq,
            },
        )
    else:
        result = _sse(
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
    assert result
    return result


def _switch_state(
    state: ResponseState | None,
    token: ReasoningToken | ToolCallToken | Token | TokenDetail | str | None,
) -> tuple[ResponseState | None, list[str]]:
    new_state: ResponseState | None

    if isinstance(token, ReasoningToken):
        new_state = ResponseState.REASONING
    elif isinstance(token, ToolCallToken):
        new_state = ResponseState.TOOL_CALLING
    elif token is not None:
        new_state = ResponseState.ANSWERING
    else:
        new_state = None

    events: list[str] = []
    if state is not new_state:
        if state is ResponseState.REASONING:
            events.append(_reasoning_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())
        elif state is ResponseState.TOOL_CALLING:
            events.append(_custom_tool_call_input_done())
            events.append(_output_item_done())
        elif state is ResponseState.ANSWERING:
            events.append(_output_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())

        if new_state is ResponseState.REASONING:
            events.append(_output_item_added(new_state))
            events.append(_content_part_added("reasoning_text"))
        elif new_state is ResponseState.TOOL_CALLING:
            events.append(_output_item_added(new_state))
        elif new_state is ResponseState.ANSWERING:
            events.append(_output_item_added(new_state))
            events.append(_content_part_added("output_text"))

    return new_state, events


def _output_item_added(state: ResponseState) -> str:
    item_types = {
        ResponseState.REASONING: "reasoning_text",
        ResponseState.TOOL_CALLING: "custom_tool_call_input",
        ResponseState.ANSWERING: "output_text",
    }
    return _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": item_types[state]},
        },
    )


def _output_item_done() -> str:
    return _sse(
        "response.output_item.done",
        {"type": "response.output_item.done", "output_index": 0},
    )


def _reasoning_text_done() -> str:
    return _sse(
        "response.reasoning_text.done",
        {
            "type": "response.reasoning_text.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _custom_tool_call_input_done() -> str:
    return _sse(
        "response.custom_tool_call_input.done",
        {
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _output_text_done() -> str:
    return _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _content_part_added(part_type: str) -> str:
    return _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": part_type},
        },
    )


def _content_part_done() -> str:
    return _sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "output_index": 0,
            "content_index": 0,
        },
    )


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n" + f"data: {dumps(data, separators=(',', ':'))}\n\n"
    )
