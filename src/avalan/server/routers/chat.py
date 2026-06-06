from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    ReasoningToken,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallToken,
)
from ...event import Event, EventType
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
from ...utils import (
    to_json,
    tool_call_diagnostic_payload,
    tool_call_error_payload,
)
from .. import di_get_logger, di_get_orchestrator
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id

from logging import Logger
from typing import Any, AsyncIterator

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
    model_id = resolve_model_id(orchestrator, request.model)

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

        async def generate_chunks() -> AsyncIterator[str]:
            async for token in response:
                if isinstance(token, Event):
                    token_text = _event_text(token)
                    if not token_text:
                        continue
                elif isinstance(token, (ReasoningToken, ToolCallToken)):
                    token_text = token.token
                else:
                    token_text = str(token)

                choice = ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkChoiceDelta(content=token_text)
                )
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=timestamp,
                    model=model_id,
                    choices=[choice],
                )
                yield sse_message(chunk.model_dump_json())
            yield sse_message("[DONE]")

            await orchestrator.sync_messages()

        logger.debug(
            "Generating event-stream stream for response %s", response_id
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
    final_response = ChatCompletionResponse(
        id=response_id,
        created=timestamp,
        model=model_id,
        choices=choices,
        usage=usage,
    )
    logger.debug(
        "Generated chat completion response #%s %r",
        response_id,
        final_response,
    )

    await orchestrator.sync_messages()

    return final_response


def _event_text(event: Event) -> str:
    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.type is EventType.TOOL_DIAGNOSTIC:
        diagnostic = _payload_diagnostic(payload)
        if diagnostic is None:
            return ""
        return to_json(
            {
                "type": "tool_diagnostic",
                "diagnostic": _diagnostic_payload(diagnostic),
            }
        )
    if event.type is EventType.TOOL_RESULT:
        result = payload.get("result")
        if isinstance(result, ToolCallDiagnostic):
            return to_json(
                {
                    "type": "tool_diagnostic",
                    "diagnostic": _diagnostic_payload(result),
                }
            )
        if isinstance(result, ToolCallError):
            return to_json(
                {
                    "type": "tool_error",
                    "toolCallId": str(result.call.id),
                    "name": result.call.name,
                    "error": tool_call_error_payload(result),
                }
            )
    return ""


def _payload_diagnostic(
    payload: dict[str, Any],
) -> ToolCallDiagnostic | None:
    diagnostic = payload.get("diagnostic")
    if isinstance(diagnostic, ToolCallDiagnostic):
        return diagnostic
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, list):
        return next(
            (
                item
                for item in diagnostics
                if isinstance(item, ToolCallDiagnostic)
            ),
            None,
        )
    return None


def _diagnostic_payload(
    diagnostic: ToolCallDiagnostic,
) -> dict[str, Any]:
    payload = {
        "id": str(diagnostic.id),
        **tool_call_diagnostic_payload(diagnostic),
    }
    if diagnostic.call_id is not None:
        payload["call_id"] = str(diagnostic.call_id)
    return payload
