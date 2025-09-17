from . import orchestrate
from ...agent.orchestrator import Orchestrator
from ...entities import (
    ReasoningToken,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    Token,
    TokenDetail,
)
from ...event import Event, EventType
from ...server.entities import ResponsesRequest
from ...utils import to_json
from asyncio import Event as AsyncEvent
from asyncio import Lock
from contextlib import suppress
from dataclasses import dataclass, replace
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from json import JSONDecodeError, dumps, loads
from logging import Logger
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Iterator
from uuid import UUID, uuid4


RS = "\x1e"

if TYPE_CHECKING:
    from .. import di_get_logger, di_get_orchestrator


@dataclass(slots=True)
class MCPResource:
    """In-memory representation of a streamed MCP resource."""

    id: str
    uri: str
    http_uri: str
    mime_type: str
    text: str
    revision: int
    closed: bool = False


class MCPResourceStore:
    """Thread-safe container for MCP streamed resources."""

    def __init__(self) -> None:
        self._resources: dict[str, MCPResource] = {}
        self._counter = 0
        self._lock = Lock()

    async def create(
        self,
        *,
        base_path: str,
        mime_type: str = "text/plain",
        initial_text: str = "",
    ) -> MCPResource:
        async with self._lock:
            self._counter += 1
            resource_id = f"{self._counter:08x}"
            uri = f"mcp://resources/{resource_id}"
            http_uri = f"{base_path}/resources/{resource_id}"
            resource = MCPResource(
                id=resource_id,
                uri=uri,
                http_uri=http_uri,
                mime_type=mime_type,
                text=initial_text,
                revision=1 if initial_text else 0,
            )
            self._resources[resource_id] = resource
            return replace(resource)

    async def append(self, resource_id: str, text: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            resource.text += text
            resource.revision += 1
            self._resources[resource_id] = resource
            return replace(resource)

    async def close(self, resource_id: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            if not resource.closed:
                resource.closed = True
                resource.revision += 1
                self._resources[resource_id] = resource
            return replace(resource)

    async def get(self, resource_id: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            return replace(resource)

    def _ensure(self, resource_id: str) -> MCPResource:
        if resource_id not in self._resources:
            raise KeyError(resource_id)
        return self._resources[resource_id]


def create_router() -> APIRouter:
    """Construct the MCP HTTP streaming router."""

    from .. import di_get_logger, di_get_orchestrator

    router = APIRouter(tags=["mcp"])

    @router.post("/tools/run")
    async def mcp_run_tool(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> StreamingResponse:
        assert logger and isinstance(logger, Logger)
        assert orchestrator and isinstance(orchestrator, Orchestrator)

        request_id, responses_request, progress_token = await _consume_call_request(
            request
        )

        response, response_uuid, timestamp = await orchestrate(
            responses_request, logger, orchestrator
        )

        cancel_event = AsyncEvent()
        message_iter = _iter_jsonrpc_messages(request)
        watcher = create_task(_watch_for_cancellation(message_iter, cancel_event, logger))

        resource_store = _get_resource_store(request)
        base_path = getattr(request.app.state, "mcp_resource_base_path", "")

        async def stream() -> AsyncGenerator[bytes, None]:
            try:
                async for chunk in _stream_mcp_response(
                    request_id=request_id,
                    request_model=responses_request,
                    response=response,
                    response_id=response_uuid,
                    timestamp=timestamp,
                    progress_token=progress_token,
                    orchestrator=orchestrator,
                    logger=logger,
                    resource_store=resource_store,
                    base_path=base_path,
                    cancel_event=cancel_event,
                ):
                    yield chunk
            finally:
                watcher.cancel()
                with suppress(Exception):
                    await watcher

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(stream(), media_type="application/json", headers=headers)

    @router.get("/resources/{resource_id}")
    async def mcp_get_resource(request: Request, resource_id: str) -> PlainTextResponse:
        store = _get_resource_store(request)
        try:
            resource = await store.get(resource_id)
        except KeyError as exc:  # pragma: no cover - FastAPI handles
            raise HTTPException(status_code=404, detail="Resource not found") from exc
        return PlainTextResponse(resource.text, media_type=resource.mime_type)

    return router


async def _consume_call_request(request: Request) -> tuple[str | int, ResponsesRequest, str]:
    messages = _iter_jsonrpc_messages(request)
    try:
        call_message = await anext(messages)
    except StopAsyncIteration as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Empty MCP request") from exc

    method = call_message.get("method")
    if method not in {"tools/call", "tools/run"}:
        raise HTTPException(status_code=400, detail="Unsupported MCP method")

    params = call_message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    if method == "tools/call":
        name = params.get("name")
        if name not in {"run", "orchestrator.run"}:
            raise HTTPException(status_code=400, detail="Unsupported tool")
        arguments = params.get("arguments")
    else:
        arguments = params

    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="Invalid tool arguments")

    try:
        request_model = ResponsesRequest.model_validate(arguments)
    except Exception as exc:  # pragma: no cover - validation error path
        raise HTTPException(status_code=400, detail="Invalid MCP arguments") from exc

    progress_token = params.get("progressToken") if isinstance(params, dict) else None
    if not progress_token:
        progress_token = str(uuid4())

    if not request_model.stream:
        request_model = request_model.model_copy(update={"stream": True})

    request.state._mcp_message_iter = messages
    return call_message.get("id", str(uuid4())), request_model, progress_token


async def _watch_for_cancellation(
    messages: AsyncIterator[dict[str, Any]],
    cancel_event: AsyncEvent,
    logger: Logger,
) -> None:
    async for message in messages:
        if not isinstance(message, dict):
            continue
        method = message.get("method")
        if method == "notifications/cancelled":
            cancel_event.set()
            logger.debug("Received MCP cancellation notification")
            break


async def _stream_mcp_response(
    *,
    request_id: str | int,
    request_model: ResponsesRequest,
    response: AsyncIterator[
        ReasoningToken | ToolCallToken | Token | TokenDetail | Event | str
    ],
    response_id: UUID,
    timestamp: int,
    progress_token: str,
    orchestrator: Orchestrator,
    logger: Logger,
    resource_store: MCPResourceStore,
    base_path: str,
    cancel_event: AsyncEvent,
) -> AsyncIterator[bytes]:
    answer_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_summaries: dict[str, dict[str, Any]] = {}
    resources: dict[str, MCPResource] = {}
    finished_normally = False

    def emit(message: dict[str, Any]) -> Iterator[bytes]:
        encoded = dumps(message, separators=(",", ":")) + RS
        yield encoded.encode("utf-8")

    try:
        async for item in response:
            if cancel_event.is_set():
                break

            if isinstance(item, ReasoningToken):
                reasoning_chunks.append(item.token)
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {
                        "level": "debug",
                        "message": {
                            "type": "reasoning",
                            "delta": item.token,
                        },
                    },
                }
                for payload in emit(notification):
                    yield payload
                continue

            if isinstance(item, Event) and item.type in (
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
            ):
                async for notification in _tool_event_notifications(
                    event=item,
                    tool_summaries=tool_summaries,
                    resources=resources,
                    resource_store=resource_store,
                    base_path=base_path,
                ):
                    for payload in emit(notification):
                        yield payload
                continue

            if isinstance(item, ToolCallToken):
                notification = _tool_call_token_notification(item)
                if notification is not None:
                    for payload in emit(notification):
                        yield payload
                continue

            text = _token_text(item)
            if text:
                answer_chunks.append(text)
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": progress_token,
                        "progress": {
                            "type": "answer.delta",
                            "delta": text,
                        },
                    },
                }
                for payload in emit(notification):
                    yield payload

        finished_normally = not cancel_event.is_set()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error while streaming MCP response", exc_info=exc)
        cancel_event.set()
        finished_normally = False
        error_message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": "An internal server error occurred.",
            },
        }
        for payload in emit(error_message):
            yield payload
        await orchestrator.sync_messages()
        return

    if cancel_event.is_set():
        await _close_response_iterator(response)
        for resource in resources.values():
            closed = await resource_store.close(resource.id)
            notification = _resource_notification(closed)
            for payload in emit(notification):
                yield payload
        error_message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": "Request cancelled"},
        }
        for payload in emit(error_message):
            yield payload
        await orchestrator.sync_messages()
        return

    if finished_normally:
        completion = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": progress_token,
                "progress": {"type": "answer.completed"},
            },
        }
        for payload in emit(completion):
            yield payload

        answer_text = "".join(answer_chunks)
        reasoning_text = "".join(reasoning_chunks)

        summary = {
            "id": str(response_id),
            "created": timestamp,
            "model": request_model.model,
            "usage": {
                "input_text_tokens": response.input_token_count,
                "output_text_tokens": response.output_token_count,
                "total_tokens": (
                    response.input_token_count + response.output_token_count
                ),
            },
        }
        if reasoning_text:
            summary["reasoning"] = reasoning_text
        if tool_summaries:
            summary["toolCalls"] = list(tool_summaries.values())

        result_message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": (
                    [{"type": "text", "text": answer_text}] if answer_text else []
                ),
                "structuredContent": summary,
            },
        }
        for payload in emit(result_message):
            yield payload

    await orchestrator.sync_messages()


def _token_text(item: Any) -> str:
    if isinstance(item, Token):
        return item.token
    if isinstance(item, TokenDetail):
        return item.token
    if isinstance(item, str):
        return item
    return ""


async def _close_response_iterator(response: Any) -> None:
    iterator = getattr(response, "_response_iterator", None)
    if iterator and hasattr(iterator, "aclose"):
        try:
            await iterator.aclose()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _tool_call_token_notification(token: ToolCallToken) -> dict[str, Any] | None:
    if token.call is None:
        if not token.token:
            return None
        return {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": "info",
                "message": {
                    "type": "tool.input_delta",
                    "delta": token.token,
                },
            },
        }

    delta = {
        "id": str(token.call.id),
        "name": token.call.name,
        "arguments": token.call.arguments,
    }
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": {
                "type": "tool.arguments_delta",
                "toolCallId": str(token.call.id),
                "delta": to_json(delta),
            },
        },
    }


async def _tool_event_notifications(
    *,
    event: Event,
    tool_summaries: dict[str, dict[str, Any]],
    resources: dict[str, MCPResource],
    resource_store: MCPResourceStore,
    base_path: str,
) -> AsyncIterator[dict[str, Any]]:
    item = _tool_call_event_item(event)
    if item is None:
        return

    tool_call_id = item["id"]

    if event.type is EventType.TOOL_PROCESS:
        tool_summaries[tool_call_id] = {
            "id": tool_call_id,
            "name": item.get("name"),
            "arguments": item.get("arguments"),
            "started": event.started,
        }
        yield {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": "info",
                "message": {
                    "type": "tool.call",
                    "toolCallId": tool_call_id,
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                },
            },
        }
        return

    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {"id": tool_call_id, "name": item.get("name"), "arguments": item.get("arguments")},
    )

    payload = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": {
                "type": "tool.result",
                "toolCallId": tool_call_id,
                "name": item.get("name"),
                "arguments": item.get("arguments"),
                "timings": {
                    "started": event.started,
                    "finished": event.finished,
                    "elapsed": event.elapsed,
                },
            },
        },
    }

    message = payload["params"]["message"]

    if "error" in item:
        message["error"] = item["error"]
        tool_summary["error"] = item["error"]
    elif "result" in item:
        message["resultDelta"] = item["result"]
        tool_summary["result"] = item["result"]
        for resource_key, payload in _extract_append_streams(
            tool_call_id, item["result"]
        ).items():
            name, text = payload
            resource = resources.get(resource_key)
            if resource is None:
                resource = await resource_store.create(
                    base_path=base_path, initial_text=text
                )
            else:
                resource = await resource_store.append(resource.id, text)
            resources[resource_key] = resource
            yield _resource_notification(resource)
            tool_summary.setdefault("resources", []).append(
                {"uri": resource.uri, "name": name}
            )

    yield payload


def _resource_notification(resource: MCPResource) -> dict[str, Any]:
    params: dict[str, Any] = {
        "resources": [
            {
                "uri": resource.uri,
                "mimeType": resource.mime_type,
                "revision": resource.revision,
                "httpUri": resource.http_uri,
            }
        ]
    }
    if resource.closed:
        params["resources"][0]["closed"] = True
    else:
        params["resources"][0]["delta"] = {"set": {"text": resource.text}}
    return {"jsonrpc": "2.0", "method": "notifications/resources/updated", "params": params}


def _extract_append_streams(
    tool_call_id: str, result: Any
) -> dict[str, tuple[str, str]]:
    streams: dict[str, tuple[str, str]] = {}
    if isinstance(result, dict):
        for key in ("stdout", "stderr", "logs"):
            value = result.get(key)
            if isinstance(value, str) and value:
                resource_key = f"{tool_call_id}:{key}"
                streams[resource_key] = (key, value)
    return streams


def _tool_call_event_item(event: Event) -> dict[str, Any] | None:
    if not event.payload:
        return None
    if event.type is EventType.TOOL_RESULT:
        tool_result = event.payload.get("result") if isinstance(event.payload, dict) else None
        if isinstance(tool_result, ToolCallError):
            return {
                "id": str(tool_result.call.id),
                "name": tool_result.name,
                "arguments": tool_result.arguments,
                "error": tool_result.message,
            }
        if isinstance(tool_result, ToolCallResult):
            result = (
                tool_result.result
                if isinstance(tool_result.result, (dict, list, str, int, float, bool))
                else to_json(tool_result.result)
            )
            return {
                "id": str(tool_result.call.id),
                "name": tool_result.name,
                "arguments": tool_result.arguments,
                "result": result,
            }
    if isinstance(event.payload, list) and event.payload:
        call = event.payload[0]
    else:
        call = event.payload.get("call") if isinstance(event.payload, dict) else None
    if call is None:
        return None
    return {
        "id": str(call.id),
        "name": call.name,
        "arguments": call.arguments,
    }


def _get_resource_store(request: Request) -> MCPResourceStore:
    store = getattr(request.app.state, "mcp_resource_store", None)
    if store is None:
        store = MCPResourceStore()
        request.app.state.mcp_resource_store = store
    assert isinstance(store, MCPResourceStore)
    return store


async def _iter_jsonrpc_messages(request: Request) -> AsyncGenerator[dict[str, Any], None]:
    if hasattr(request.state, "_mcp_message_iter"):
        iterator = request.state._mcp_message_iter
        delattr(request.state, "_mcp_message_iter")
        async for message in iterator:
            yield message
        return

    buffer = ""
    async for chunk in request.stream():
        if not chunk:
            continue
        buffer += chunk.decode("utf-8")
        while RS in buffer:
            segment, buffer = buffer.split(RS, 1)
            segment = segment.strip()
            if not segment:
                continue
            try:
                yield loads(segment)
            except JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Invalid MCP payload") from exc
    if buffer.strip():
        try:
            yield loads(buffer)
        except JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid MCP payload") from exc
