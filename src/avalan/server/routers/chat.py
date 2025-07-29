from .. import di_get_logger, di_get_orchestrator
from ...agent.orchestrator import Orchestrator
from ...entities import GenerationSettings, Message, MessageRole
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
from time import time

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

    logger.debug("Processing chat completion request with messages %r", request)

    input = [
        Message(role=chat_message.role, content=chat_message.content)
        for chat_message in request.messages
    ]

    logger.debug("Transformed chat completion request to engine input %r", input)

    response_id = (  # generate a pseudo-unique ID
        f"chatcmpl-{int(time() * 1000)}"
    )
    timestamp = int(time())

    settings = GenerationSettings(
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        stop_strings=request.stop,
        top_p=request.top_p,
        # num_return_sequences=request.n
    )

    logger.debug("Calling orchestrator with input %r and settings %r for response %s", input, settings, response_id)

    response = await orchestrator(input, settings=settings)

    # Streaming through SSE (server-sent events with text/event-stream)
    if request.stream:

        async def generate_chunks():
            async for token in response:
                if isinstance(token, Event):
                    continue

                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token)
                )
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=timestamp,
                    model=request.model,
                    choices=[choice],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"  # SSE data event
            yield "data: [DONE]\n\n"  # end of stream

        logger.debug(f"Generating event-stream stream for response {response_id}")

        return StreamingResponse(
            generate_chunks(), media_type="text/event-stream"
        )

    # Non streaming
    message = ChatMessage(
        role=str(MessageRole.ASSISTANT), content=await response.to_str()
    )
    usage = ChatCompletionUsage()
    response = ChatCompletionResponse(
        id=response_id,
        created=timestamp,
        model=request.model,
        choices=[ChatCompletionChoice(message=message, finish_reason="stop")],
        usage=usage,
    )
    logger.debug("Generated chat completion response #%s %r", response_id, response)
    return response
