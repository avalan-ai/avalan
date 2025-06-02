from ...agent.orchestrator import Orchestrator
from ...entities import GenerationSettings, Message, MessageRole
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

    # request = model='gpt-4o' messages=[ChatMessage(role='user', content='Explain LLM distillation')] temperature=1.0 top_p=1.0 n=1 stream=True stop=None max_tokens=None presence_penalty=0.0 frequency_penalty=0.0 logit_bias=None user=None

    input = [
        Message(role=chat_message.role, content=chat_message.content)
        for chat_message in request.messages
    ]

    response_id = (
        f"chatcmpl-{int(time() * 1000)}"  # generate a pseudo-unique ID
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
                # OpenAI stream delta chunk
                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token)
                )
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=timestamp,
                    model=request.model,
                    choices=[choice],
                )
                yield f"data: {chunk.json()}\n\n"  # SSE data event
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
