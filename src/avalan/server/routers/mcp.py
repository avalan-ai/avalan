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
from ...server.entities import ChatCompletionRequest, ChatMessage, MCPToolRequest
from ...utils import to_json
from asyncio import Event as AsyncEvent, Lock, create_task
from contextlib import suppress
from dataclasses import dataclass, replace
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from json import JSONDecodeError, dumps, loads
from logging import Logger
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Final,
    Iterator,
    Literal,
    Protocol,
    TypedDict,
    cast,
)
from uuid import UUID, uuid4

RS: Final[str] = "\x1e"

JSONScalar = None | bool | int | float | str
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]

Method = Literal["initialize", "ping", "tools/list", "tools/call"]
NotificationMethod = Literal[
    "notifications/cancelled", "notifications/message"
]

ResponseItem = (
    ReasoningToken | ToolCallToken | Token | TokenDetail | Event | str
)


class JSONRPCRequest(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    method: str
    params: JSONObject | None


class JSONRPCResult(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    result: JSONObject


class JSONRPCError(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    error: dict[str, JSONValue]


@dataclass(slots=True)
class MCPResource:
    id: str
    uri: str
    http_uri: str
    mime_type: str
    text: str
    revision: int
    closed: bool = False


class MCPResourceStore:
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
    from .. import di_get_logger, di_get_orchestrator

    router = APIRouter(tags=["mcp"])

    @router.post("", response_model=None)
    @router.post("/", response_model=None)
    async def mcp_rpc(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> Response:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, messages = await _expect_jsonrpc_message(
            request, {"initialize", "ping", "tools/list", "tools/call"}
        )
        method = cast(str, message.get("method"))
        if method == "initialize":
            return _handle_initialize_message(
                request, logger, orchestrator, message
            )
        if method == "ping":
            return _handle_ping_message(logger, message)
        if method == "tools/list":
            return _handle_list_tools_message(
                request, logger, orchestrator, message
            )
        if method == "tools/call":
            request_id, responses_request, progress_token = (
                _parse_call_request(request, message, messages)
            )
            return await _start_tool_streaming_response(
                request,
                logger,
                orchestrator,
                request_id,
                responses_request,
                progress_token,
            )

        raise HTTPException(
            status_code=400, detail=f'Unsupported MCP method "{method}"'
        )

    @router.post("/initialize")
    async def mcp_initialize(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, _ = await _expect_jsonrpc_message(request, {"initialize"})
        return _handle_initialize_message(
            request, logger, orchestrator, message
        )

    @router.post("/ping")
    async def mcp_ping(
        request: Request,
        logger: Logger = Depends(di_get_logger),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        message, _ = await _expect_jsonrpc_message(request, {"ping"})
        return _handle_ping_message(logger, message)

    @router.post("/tools/list")
    async def mcp_list_tools(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, _ = await _expect_jsonrpc_message(request, {"tools/list"})
        return _handle_list_tools_message(
            request, logger, orchestrator, message
        )

    @router.get("/resources/{resource_id}")
    async def mcp_get_resource(
        request: Request, resource_id: str
    ) -> PlainTextResponse:
        store = _get_resource_store(request)
        try:
            resource = await store.get(resource_id)
        except KeyError as exc:  # pragma: no cover - FastAPI handles
            raise HTTPException(
                status_code=404, detail="Resource not found"
            ) from exc
        return PlainTextResponse(resource.text, media_type=resource.mime_type)

    return router


async def _consume_call_request(
    request: Request,
) -> tuple[str | int, MCPToolRequest, str]:
    call_message, messages = await _expect_jsonrpc_message(
        request, {"tools/call"}
    )
    return _parse_call_request(request, call_message, messages)


def _parse_call_request(
    request: Request,
    call_message: JSONObject,
    messages: AsyncIterator[JSONObject],
) -> tuple[str | int, MCPToolRequest, str]:
    method = call_message.get("method")
    if method != "tools/call":
        raise HTTPException(
            status_code=400, detail=f'Unsupported MCP method "{method}"'
        )

    params = call_message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    allowed_tool_name = cast(
        str, getattr(request.app.state, "mcp_tool_name", "run")
    )
    arguments = _extract_call_arguments(
        cast(str, method), params, allowed_tool_name=allowed_tool_name
    )
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="Invalid tool arguments")

    try:
        request_model = MCPToolRequest.model_validate(arguments)
    except Exception as exc:  # pragma: no cover - validation error path
        raise HTTPException(
            status_code=400, detail="Invalid MCP arguments"
        ) from exc

    progress_token = cast(str | None, params.get("progressToken"))
    if not progress_token:
        progress_token = str(uuid4())

    request.state._mcp_message_iter = messages
    return (
        cast(str | int, call_message.get("id", str(uuid4()))),
        request_model,
        progress_token,
    )


async def _expect_jsonrpc_message(
    request: Request, allowed_methods: set[Method]
) -> tuple[JSONObject, AsyncIterator[JSONObject]]:
    messages = _iter_jsonrpc_messages(request)
    try:
        message = await anext(messages)
    except (
        StopAsyncIteration
    ) as exc:  # pragma: no cover - defensive validation
        raise HTTPException(
            status_code=400, detail="Empty MCP request"
        ) from exc

    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid MCP payload")

    method = cast(str | None, message.get("method"))
    if method not in allowed_methods:
        raise HTTPException(
            status_code=400, detail=f"Unsupported MCP method {method}"
        )

    return message, messages


def _server_info(request: Request) -> dict[str, str]:
    app = request.app
    name = getattr(app, "title", None) or "avalan"
    version = getattr(app, "version", None)
    if version is None:
        version = getattr(app.state, "version", None)
    if version is None:
        version = "0.0.0"
    return {"name": str(name), "version": str(version)}


def _server_capabilities(orchestrator: Orchestrator) -> dict[str, JSONValue]:
    return {
        "tools": {
            "list": True,
            "call": True,
            "listChanged": False,
        },
        "resources": {
            "subscribe": True,
            "listChanged": False,
        },
    }


class StreamResponse(Protocol):
    input_token_count: int
    output_token_count: int
    _response_iterator: AsyncIterator[ResponseItem] | None

    async def to_str(self) -> str: ...
    def __aiter__(self) -> AsyncIterator[ResponseItem]: ...


MODEL_FALLBACK: Final[str] = "default"


def _default_model_id(orchestrator: Orchestrator) -> str:
    model_ids = getattr(orchestrator, "model_ids", None)
    if model_ids:
        candidates = sorted(str(model_id) for model_id in model_ids)
        if candidates:
            return candidates[0]
    return MODEL_FALLBACK


def _build_chat_request(
    tool_request: MCPToolRequest, orchestrator: Orchestrator
) -> ChatCompletionRequest:
    model_id = _default_model_id(orchestrator)
    return ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatMessage(role="user", content=tool_request.input_string)
        ],
        stream=True,
    )


async def _start_tool_streaming_response(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    request_id: str | int,
    tool_request: MCPToolRequest,
    progress_token: str,
) -> StreamingResponse:
    chat_request = _build_chat_request(tool_request, orchestrator)
    response, response_uuid, timestamp = await orchestrate(
        chat_request, logger, orchestrator
    )
    response_typed = cast(StreamResponse, response)

    cancel_event = AsyncEvent()
    message_iter = _iter_jsonrpc_messages(request)
    watcher = create_task(
        _watch_for_cancellation(message_iter, cancel_event, logger)
    )

    resource_store = _get_resource_store(request)
    base_path = cast(
        str, getattr(request.app.state, "mcp_resource_base_path", "")
    )

    if not chat_request.stream:
        try:
            text = await response_typed.to_str()
        finally:
            watcher.cancel()
            with suppress(Exception):
                await watcher

        summary: dict[str, JSONValue] = {
            "id": str(response_uuid),
            "created": timestamp,
            "model": chat_request.model,
            "usage": {
                "input_text_tokens": getattr(
                    response_typed, "input_token_count", 0
                ),
                "output_text_tokens": getattr(
                    response_typed, "output_token_count", 0
                ),
                "total_tokens": (
                    getattr(response_typed, "input_token_count", 0)
                    + getattr(response_typed, "output_token_count", 0)
                ),
            },
        }
        result_message: JSONRPCResult = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}] if text else [],
                "structuredContent": summary,
            },
        }
        return JSONResponse(result_message)

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in _stream_mcp_response(
                request_id=request_id,
                request_model=chat_request,
                response=response_typed,
                response_id=response_uuid,
                timestamp=timestamp,
                progress_token=progress_token,
                orchestrator=orchestrator,
                logger=logger,
                resource_store=resource_store,
                base_path=base_path,
                cancel_event=cancel_event,
            ):
                yield b"data: " + chunk.rstrip(b"\n") + b"\n\n"
        finally:
            watcher.cancel()
            with suppress(Exception):
                await watcher

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=headers
    )


def _handle_ping_message(
    logger: Logger,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": {},
    }
    logger.debug(
        "Handled MCP ping request", extra={"response_id": response_id}
    )
    return JSONResponse(payload)


def _handle_initialize_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    params_obj: JSONObject = params if isinstance(params, dict) else {}
    protocol_version = str(params_obj.get("protocolVersion") or "1.0.0")

    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": {
            "protocolVersion": protocol_version,
            "capabilities": _server_capabilities(orchestrator),
            "serverInfo": _server_info(request),
        },
    }
    logger.debug(
        "Handled MCP initialize request",
        extra={"response_id": response_id},
    )
    return JSONResponse(payload)


def _handle_list_tools_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    tools = _collect_tool_descriptions(request)
    response_id = cast(str | int, message.get("id", str(uuid4())))
    result: dict[str, JSONValue] = {"tools": tools}
    next_cursor = getattr(request.app.state, "mcp_next_cursor", None)
    if next_cursor:
        result["nextCursor"] = next_cursor
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": result,
    }
    logger.debug(
        "Handled MCP tools list request",
        extra={"response_id": response_id, "tool_count": len(tools)},
    )
    return JSONResponse(payload)


def _collect_tool_descriptions(request: Request) -> list[dict[str, JSONValue]]:
    name = cast(str, getattr(request.app.state, "mcp_tool_name", "run"))
    description = cast(
        str,
        getattr(
            request.app.state,
            "mcp_tool_description",
            "Execute the Avalan orchestrator run endpoint.",
        ),
    )
    return [
        {
            "name": name,
            "description": description,
            "inputSchema": MCPToolRequest.model_json_schema(),
        }
    ]


def _extract_call_arguments(
    method: str, params: JSONObject, *, allowed_tool_name: str
) -> dict[str, JSONValue]:
    if method == "tools/call":
        name = params.get("name")
        if name is None or name != allowed_tool_name:
            raise HTTPException(
                status_code=400, detail=f'Unsupported tool "{name}"'
            )
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            raise HTTPException(
                status_code=400, detail="Invalid tool arguments"
            )
        return cast(dict[str, JSONValue], arguments)

    raise HTTPException(
        status_code=400, detail=f'Unsupported MCP method "{method}"'
    )


async def _watch_for_cancellation(
    messages: AsyncIterator[JSONObject],
    cancel_event: AsyncEvent,
    logger: Logger,
) -> None:
    async for message in messages:
        if not isinstance(message, dict):
            continue
        method = cast(str | None, message.get("method"))
        if method == "notifications/cancelled":
            cancel_event.set()
            logger.debug("Received MCP cancellation notification")
            break


async def _stream_mcp_response(
    *,
    request_id: str | int,
    request_model: ChatCompletionRequest,
    response: StreamResponse,
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
    tool_summaries: dict[str, dict[str, JSONValue]] = {}
    resources: dict[str, MCPResource] = {}
    finished_normally = False

    def emit(message: JSONObject) -> Iterator[bytes]:
        encoded = dumps(message, separators=(",", ":")) + "\n"
        yield encoded.encode("utf-8")

    try:
        async for item in response:
            if cancel_event.is_set():
                break

            if isinstance(item, ReasoningToken):
                reasoning_chunks.append(item.token)
                notification: JSONObject = {
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
                notification: JSONObject = {
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
        error_message: JSONRPCError = {
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
        error_message: JSONRPCError = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": "Request cancelled"},
        }
        for payload in emit(error_message):
            yield payload
        await orchestrator.sync_messages()
        return

    if finished_normally:
        completion: JSONObject = {
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

        summary: dict[str, JSONValue] = {
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

        result_message: JSONRPCResult = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": (
                    [{"type": "text", "text": answer_text}]
                    if answer_text
                    else []
                ),
                "structuredContent": summary,
            },
        }
        for payload in emit(result_message):
            yield payload

    await orchestrator.sync_messages()


def _token_text(item: ResponseItem) -> str:
    if isinstance(item, (Token, TokenDetail)):
        return item.token
    if isinstance(item, str):
        return item
    return ""


async def _close_response_iterator(response: StreamResponse) -> None:
    iterator = getattr(response, "_response_iterator", None)
    if iterator and hasattr(iterator, "aclose"):
        try:
            await cast(AsyncIterator[object], iterator).aclose()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _tool_call_token_notification(
    token: ToolCallToken,
) -> JSONObject | None:
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

    delta: dict[str, JSONValue] = {
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
    tool_summaries: dict[str, dict[str, JSONValue]],
    resources: dict[str, MCPResource],
    resource_store: MCPResourceStore,
    base_path: str,
) -> AsyncIterator[JSONObject]:
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
        {
            "id": tool_call_id,
            "name": item.get("name"),
            "arguments": item.get("arguments"),
        },
    )

    payload: JSONObject = {
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

    message = cast(dict[str, JSONValue], payload["params"]["message"])

    if "error" in item:
        message["error"] = item["error"]
        tool_summary["error"] = item["error"]
    elif "result" in item:
        message["resultDelta"] = item["result"]
        tool_summary["result"] = item["result"]
        for resource_key, payload2 in _extract_append_streams(
            tool_call_id, item["result"]
        ).items():
            name, text = payload2
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
                {
                    "uri": resource.uri,
                    "name": name,
                }
            )

    yield payload


def _resource_notification(resource: MCPResource) -> JSONObject:
    params: dict[str, JSONValue] = {
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
        cast(list[dict[str, JSONValue]], params["resources"])[0][
            "closed"
        ] = True
    else:
        cast(list[dict[str, JSONValue]], params["resources"])[0]["delta"] = {
            "set": {"text": resource.text}
        }
    return {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": params,
    }


def _extract_append_streams(
    tool_call_id: str, result: JSONValue
) -> dict[str, tuple[str, str]]:
    streams: dict[str, tuple[str, str]] = {}
    if isinstance(result, dict):
        for key in ("stdout", "stderr", "logs"):
            value = result.get(key)
            if isinstance(value, str) and value:
                resource_key = f"{tool_call_id}:{key}"
                streams[resource_key] = (key, value)
    return streams


def _tool_call_event_item(event: Event) -> dict[str, JSONValue] | None:
    if not event.payload:
        return None
    if event.type is EventType.TOOL_RESULT:
        tool_result = (
            event.payload.get("result")
            if isinstance(event.payload, dict)
            else None
        )
        if isinstance(tool_result, ToolCallError):
            return {
                "id": str(tool_result.call.id),
                "name": tool_result.name,
                "arguments": tool_result.arguments,
                "error": tool_result.message,
            }
        if isinstance(tool_result, ToolCallResult):
            result: JSONValue = (
                tool_result.result
                if isinstance(
                    tool_result.result, (dict, list, str, int, float, bool)
                )
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
        call = (
            event.payload.get("call")
            if isinstance(event.payload, dict)
            else None
        )
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


async def _iter_jsonrpc_messages(
    request: Request,
) -> AsyncGenerator[JSONObject, None]:
    if hasattr(request.state, "_mcp_message_iter"):
        iterator = cast(
            AsyncIterator[JSONObject], request.state._mcp_message_iter
        )
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
                obj = loads(segment)
            except JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid MCP payload"
                ) from exc
            if not isinstance(obj, dict):
                raise HTTPException(
                    status_code=400, detail="Invalid MCP payload"
                )
            yield cast(JSONObject, obj)
    if buffer.strip():
        try:
            obj2 = loads(buffer)
        except JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid MCP payload"
            ) from exc
        if not isinstance(obj2, dict):
            raise HTTPException(status_code=400, detail="Invalid MCP payload")
        yield cast(JSONObject, obj2)
