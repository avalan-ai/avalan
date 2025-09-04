from . import iter_tokens, orchestrate
from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import ReasoningToken, ToolCallToken
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

            async for token in iter_tokens(response):
                state = _switch_state(state, token)

                if isinstance(token, ReasoningToken):
                    yield _sse(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "delta": token.token,
                            "output_index": 0,
                            "content_index": 0,
                            "sequence_number": seq,  # optional
                        },
                    )
                elif isinstance(token, ToolCallToken):
                    yield _sse(
                        "response.custom_tool_call_input.delta",
                        {
                            "type": "response.custom_tool_call_input.delta",
                            "delta": token.token,
                            "output_index": 0,
                            "content_index": 0,
                            "sequence_number": seq,  # optional
                        },
                    )
                else:
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": token,
                            "output_index": 0,
                            "content_index": 0,
                            "sequence_number": seq,  # optional
                        },
                    )
                seq += 1

            _switch_state(state, None)

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


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n" + f"data: {dumps(data, separators=(',', ':'))}\n\n"
    )


def _switch_state(
    state: ResponseState | None,
    token: str | ReasoningToken | ToolCallToken | None,
) -> ResponseState | None:
    new_state: ResponseState | None = state

    if (
        isinstance(token, ReasoningToken)
        and state is not ResponseState.REASONING
    ):
        new_state = ResponseState.REASONING
    elif (
        isinstance(token, ToolCallToken)
        and state is not ResponseState.TOOL_CALLING
    ):
        new_state = ResponseState.TOOL_CALLING
    elif token is not None and state is not ResponseState.ANSWERING:
        new_state = ResponseState.ANSWERING
    elif token is None:
        new_state = None

    if (
        (state is None and new_state is not None)
        or (state is not None and new_state is None)
        or (new_state != state)
    ):
        if state is ResponseState.REASONING:
            yield _sse(
                "response.reasoning_text.done",
                {
                    "type": "response.reasoning_text.done",
                    "output_index": 0,
                    "content_index": 0,
                },
            )
        elif state is ResponseState.TOOL_CALLING:
            yield _sse(
                "response.custom_tool_call_input.done",
                {
                    "type": "response.custom_tool_call_input.done",
                    "output_index": 0,
                    "content_index": 0,
                },
            )
        elif state is ResponseState.ANSWERING:
            yield _sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                },
            )

    return new_state
