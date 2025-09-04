from . import orchestrate
from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import MessageRole, ReasoningToken, ToolCallToken
from ...event import Event
from ...server.entities import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionUsage,
)
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from logging import Logger

router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
):
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages

    logger.info(
        "Processing chat completion request for orchestrator %s",
        str(orchestrator),
    )
    logger.debug(
        "Processing chat completion request with messages %r", request
    )

    response, response_id, timestamp = await orchestrate(
        request, logger, orchestrator
    )

    logger.info(
        "Orchestrator %s responded for chat completion request",
        str(orchestrator),
    )

    # Streaming through SSE (server-sent events with text/event-stream)
    if request.stream:

        async def generate_chunks():
            async for token in response:
                if isinstance(token, Event):
                    continue
                elif isinstance(token, (ReasoningToken, ToolCallToken)):
                    token = token.token

                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token)
                )
                chunk = ChatCompletionChunk(
                    id=str(response_id),
                    created=timestamp,
                    model=request.model,
                    choices=[choice],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"  # SSE data event
            yield "data: [DONE]\n\n"  # end of stream

            await orchestrator.sync_messages()

        logger.debug(
            f"Generating event-stream stream for response {response_id}"
        )

        return StreamingResponse(
            generate_chunks(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non streaming
    text = await response.to_str()
    choices = [
        ChatCompletionChoice(
            index=i,
            message=ChatMessage(role=str(MessageRole.ASSISTANT), content=text),
            finish_reason="stop",
        )
        for i in range(request.n or 1)
    ]
    usage = ChatCompletionUsage()
    response = ChatCompletionResponse(
        id=str(response_id),
        created=timestamp,
        model=request.model,
        choices=choices,
        usage=usage,
    )
    logger.debug(
        "Generated chat completion response #%s %r", response_id, response
    )

    await orchestrator.sync_messages()

    return response
