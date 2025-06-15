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
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from time import time

router = APIRouter(
    prefix="/chat",
    tags=["completions"],
)


def dependency_get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    orchestrator: Orchestrator = Depends(dependency_get_orchestrator),
):
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert request and request.messages

    input = [
        Message(role=chat_message.role, content=chat_message.content)
        for chat_message in request.messages
    ]

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
    return response
