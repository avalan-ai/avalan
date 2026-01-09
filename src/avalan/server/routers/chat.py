from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from ...event import Event
from ...server.entities import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from .. import di_get_logger, di_get_orchestrator
from ..sse import sse_headers, sse_message
from . import orchestrate

from logging import Logger

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
) -> ChatCompletionResponse | StreamingResponse:
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
                token_text: str
                if isinstance(token, (ReasoningToken, ToolCallToken)):
                    token_text = token.token
                elif isinstance(token, (Token, TokenDetail)):
                    token_text = token.token
                else:
                    token_text = str(token)

                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token_text)
                )
                chunk = ChatCompletionChunk(
                    id=str(response_id),
                    created=timestamp,
                    model=request.model,
                    choices=[choice],
                )
                yield sse_message(chunk.model_dump_json())
            yield sse_message("[DONE]")

            await orchestrator.sync_messages()

        logger.debug(
            f"Generating event-stream stream for response {response_id}"
        )

        return StreamingResponse(
            generate_chunks(),
            media_type="text/event-stream",
            headers=sse_headers(),
        )

    # Non streaming
    text = await response.to_str()
    choices = [
        ChatCompletionChoice(
            index=i,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            finish_reason="stop",
        )
        for i in range(request.n or 1)
    ]
    usage = ChatCompletionUsage()
    chat_response = ChatCompletionResponse(
        id=str(response_id),
        created=timestamp,
        model=request.model,
        choices=choices,
        usage=usage,
    )
    logger.debug(
        "Generated chat completion response #%s %r", response_id, chat_response
    )

    await orchestrator.sync_messages()

    return chat_response
