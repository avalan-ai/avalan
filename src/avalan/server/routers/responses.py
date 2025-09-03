import json
from logging import Logger

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...server.entities import ResponsesRequest
from ._shared import iter_tokens, orchestrate

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
            yield (
                "event: response.created\n"
                + f'data: {{"id": "{response_id}", "created": {timestamp},'
                f' "model": "{request.model}"}}\n\n'
            )
            async for token in iter_tokens(response):
                data = json.dumps({"delta": token, "index": 0})
                yield f"event: response.output_text.delta\ndata: {data}\n\n"
            yield "event: response.completed\ndata: {}\n\n"
            yield "event: done\ndata: {}\n\n"
            await orchestrator.sync_messages()

        return StreamingResponse(generate(), media_type="text/event-stream")

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
