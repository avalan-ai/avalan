from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...server.entities import ResponsesRequest
from ._shared import iter_tokens, orchestrate
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from json import dumps
from logging import Logger

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

            async for token in iter_tokens(response):
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

            yield _sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                },
            )

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
            "input_text_tokens": 0,
            "output_text_tokens": 0,
            "total_tokens": 0,
        },
    }
    await orchestrator.sync_messages()
    return body


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\n" + f"data: {dumps(data, separators=(',', ':'))}\n\n"
    )
