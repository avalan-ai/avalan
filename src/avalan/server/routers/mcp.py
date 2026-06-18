from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    Token,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallResult,
)
from ...event import Event, EventType
from ...model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    canonical_item_from_token,
)
from ...server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    MCPToolRequest,
)
from ...types import JsonObject, JsonScalar, LooseJsonValue, MutableJsonValue
from ...utils import (
    to_json,
    tool_call_diagnostic_payload,
    tool_call_error_payload,
)
from ..sse import sse_bytes, sse_headers
from . import (
    MODEL_FALLBACK as DEFAULT_MODEL_FALLBACK,
)
from . import (
    orchestrate,
    resolve_model_id,
)
from .streaming import (
    ProtocolStreamAccumulator,
    ProtocolStreamProjectionState,
    ProtocolStreamSnapshot,
    cancellable_stream_iterator,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
    stream_consumer_iterator,
)

from asyncio import CancelledError, Lock, create_task
from asyncio import Event as AsyncEvent
from contextlib import suppress
from dataclasses import dataclass, field, replace
from json import JSONDecodeError, dumps, loads
from logging import Logger
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Final,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    TypeAlias,
    TypedDict,
    cast,
)
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

RS: Final[str] = "\x1e"
MODEL_FALLBACK: Final[str] = DEFAULT_MODEL_FALLBACK

JSONScalar: TypeAlias = JsonScalar
JSONValue: TypeAlias = MutableJsonValue
JSONObject: TypeAlias = JsonObject

Method = Literal["initialize", "ping", "tools/list", "tools/call"]
NotificationMethod = Literal[
    "notifications/cancelled",
    "notifications/initialized",
    "notifications/message",
]
AllowedMethod = Method | NotificationMethod

ResponseItem = (
    CanonicalStreamItem | StreamConsumerProjection | Token | Event | str
)


def _default_model_id(orchestrator: Orchestrator) -> str:
    return resolve_model_id(orchestrator)


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


class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


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
    def __init__(
        self,
        resource_item_limit: int | None = None,
        resource_limit: int | None = None,
    ) -> None:
        retention_settings = protocol_stream_retention_settings(
            StreamRetentionPolicy()
        )
        if resource_item_limit is None:
            resource_item_limit = retention_settings.resource_item_limit
        if resource_limit is None:
            resource_limit = retention_settings.resource_item_limit
        assert isinstance(resource_item_limit, int)
        assert not isinstance(resource_item_limit, bool)
        assert resource_item_limit >= 0
        assert isinstance(resource_limit, int)
        assert not isinstance(resource_limit, bool)
        assert resource_limit > 0
        self._resources: dict[str, MCPResource] = {}
        self._resource_chunks: dict[str, list[str]] = {}
        self._resource_order: list[str] = []
        self._counter = 0
        self._resource_item_limit = resource_item_limit
        self._resource_limit = resource_limit
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
                text=self._retained_text([initial_text]),
                revision=1 if initial_text else 0,
            )
            self._resources[resource_id] = resource
            self._resource_chunks[resource_id] = (
                self._retained_chunks([initial_text]) if initial_text else []
            )
            self._resource_order.append(resource_id)
            self._enforce_resource_retention(
                protected_resource_ids={resource_id}
            )
            return replace(resource)

    async def append(self, resource_id: str, text: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            chunks = self._resource_chunks.setdefault(resource_id, [])
            chunks.append(text)
            retained_chunks = self._retained_chunks(chunks)
            self._resource_chunks[resource_id] = retained_chunks
            resource.text = self._retained_text(retained_chunks)
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
            self._enforce_resource_retention(protected_resource_ids=set())
            return replace(resource)

    async def close_many(
        self, resource_ids: list[str]
    ) -> tuple[MCPResource, ...]:
        assert isinstance(resource_ids, list)
        for resource_id in resource_ids:
            assert isinstance(resource_id, str)
        protected_resource_ids = set(resource_ids)
        async with self._lock:
            closed_resources: list[MCPResource] = []
            for resource_id in resource_ids:
                resource = self._ensure(resource_id)
                if not resource.closed:
                    resource.closed = True
                    resource.revision += 1
                    self._resources[resource_id] = resource
                closed_resources.append(replace(resource))
            self._enforce_resource_retention(
                protected_resource_ids=protected_resource_ids
            )
            return tuple(closed_resources)

    async def prune_closed(self) -> None:
        async with self._lock:
            self._enforce_resource_retention(protected_resource_ids=set())

    async def get(self, resource_id: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            return replace(resource)

    async def history(self, resource_id: str) -> tuple[str, ...]:
        async with self._lock:
            self._ensure(resource_id)
            return tuple(self._resource_chunks.get(resource_id, ()))

    def _ensure(self, resource_id: str) -> MCPResource:
        if resource_id not in self._resources:
            raise KeyError(resource_id)
        return self._resources[resource_id]

    def _retained_chunks(self, chunks: list[str]) -> list[str]:
        if self._resource_item_limit == 0:
            return []
        if len(chunks) <= self._resource_item_limit:
            return list(chunks)
        return list(chunks[-self._resource_item_limit :])

    def _retained_text(self, chunks: list[str]) -> str:
        return "".join(self._retained_chunks(chunks))

    def _enforce_resource_retention(
        self, *, protected_resource_ids: set[str]
    ) -> None:
        assert isinstance(protected_resource_ids, set)
        while len(self._resources) > self._resource_limit:
            evicted = False
            for resource_id in list(self._resource_order):
                if resource_id in protected_resource_ids:
                    continue
                resource = self._resources.get(resource_id)
                if resource is None:
                    self._resource_order.remove(resource_id)
                    evicted = True
                    break
                if resource.closed:
                    self._resources.pop(resource_id, None)
                    self._resource_chunks.pop(resource_id, None)
                    self._resource_order.remove(resource_id)
                    evicted = True
                    break
            if not evicted:
                return


@dataclass(slots=True)
class _MCPLegacyStreamAdapter:
    stream_session_id: str = "mcp-legacy-stream"
    run_id: str = "mcp-legacy-run"
    turn_id: str = "mcp-legacy-turn"
    sequence: int = 0
    tool_execution_started_ids: set[str] = field(default_factory=set)

    def map(self, item: object) -> tuple[CanonicalStreamItem, ...] | None:
        if isinstance(item, (Token, str)):
            return self._canonical_items_from_token(item)

        if isinstance(item, Event):
            return self.canonical_items_from_event(item)

        return None

    def canonical_items_from_event(
        self,
        event: Event,
    ) -> tuple[CanonicalStreamItem, ...]:
        if event.type not in (
            EventType.TOOL_DIAGNOSTIC,
            EventType.TOOL_PROCESS,
            EventType.TOOL_RESULT,
        ):
            return ()

        item = _tool_call_event_item(event)
        if item is None:
            return ()

        tool_call_id = item.get("id")
        if not isinstance(tool_call_id, str):
            return ()

        result: list[CanonicalStreamItem] = []
        self._append_stream_start(result)

        if event.type is EventType.TOOL_PROCESS:
            self._append_tool_execution_start(
                result, tool_call_id, item, event
            )
            return tuple(result)

        if "diagnostic" in item:
            diagnostic_payload = item["diagnostic"]
            message = (
                diagnostic_payload.get("message")
                if isinstance(diagnostic_payload, dict)
                else None
            )
            result.append(
                self._canonical_item(
                    StreamItemKind.STREAM_DIAGNOSTIC,
                    correlation=StreamItemCorrelation(
                        tool_call_id=tool_call_id
                    ),
                    text_delta=message if isinstance(message, str) else "",
                    data={
                        "toolCallId": tool_call_id,
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                        "diagnostic": diagnostic_payload,
                        "timings": _event_timings(event),
                    },
                )
            )
            return tuple(result)

        if "result" in item:
            self._append_tool_execution_start(
                result, tool_call_id, item, event, synthetic=True
            )
            for category, content in _legacy_tool_execution_output_items(
                tool_call_id, item["result"]
            ):
                result.append(
                    self._canonical_item(
                        StreamItemKind.TOOL_EXECUTION_OUTPUT,
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                        text_delta=content,
                        data={"category": category, "content": content},
                        metadata=_tool_metadata(item),
                    )
                )
            result.append(
                self._canonical_item(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    correlation=StreamItemCorrelation(
                        tool_call_id=tool_call_id
                    ),
                    data={
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                        "result": item["result"],
                        "timings": _event_timings(event),
                    },
                )
            )
            return tuple(result)

        if "error" in item:
            self._append_tool_execution_start(
                result, tool_call_id, item, event, synthetic=True
            )
            result.append(
                self._canonical_item(
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    correlation=StreamItemCorrelation(
                        tool_call_id=tool_call_id
                    ),
                    data={
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                        "error": item["error"],
                        "timings": _event_timings(event),
                    },
                )
            )

        return tuple(result)

    def _append_tool_execution_start(
        self,
        items: list[CanonicalStreamItem],
        tool_call_id: str,
        item: dict[str, JSONValue],
        event: Event,
        *,
        synthetic: bool = False,
    ) -> None:
        if tool_call_id in self.tool_execution_started_ids:
            return
        self.tool_execution_started_ids.add(tool_call_id)
        items.append(
            self._canonical_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
                data={
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "timings": _event_timings(event),
                },
                metadata=(
                    {"mcp.synthetic_tool_execution_start": True}
                    if synthetic
                    else None
                ),
            )
        )

    def _canonical_items_from_token(
        self,
        item: Token | str,
    ) -> tuple[CanonicalStreamItem, ...]:
        result: list[CanonicalStreamItem] = []
        self._append_stream_start(result)
        result.append(
            canonical_item_from_token(
                item,
                self._next_sequence(),
                stream_session_id=self.stream_session_id,
                run_id=self.run_id,
                turn_id=self.turn_id,
            )
        )
        return tuple(result)

    def _append_stream_start(
        self,
        items: list[CanonicalStreamItem],
    ) -> None:
        if self.sequence != 0:
            return
        items.append(
            CanonicalStreamItem(
                stream_session_id=self.stream_session_id,
                run_id=self.run_id,
                turn_id=self.turn_id,
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
        )
        self.sequence = 1

    def _canonical_item(
        self,
        kind: StreamItemKind,
        *,
        correlation: StreamItemCorrelation | None = None,
        text_delta: str | None = None,
        data: LooseJsonValue | None = None,
        metadata: dict[str, LooseJsonValue] | None = None,
    ) -> CanonicalStreamItem:
        channel = (
            StreamChannel.CONTROL
            if kind is StreamItemKind.STREAM_DIAGNOSTIC
            else StreamChannel.TOOL_EXECUTION
        )
        return CanonicalStreamItem(
            stream_session_id=self.stream_session_id,
            run_id=self.run_id,
            turn_id=self.turn_id,
            sequence=self._next_sequence(),
            kind=kind,
            channel=channel,
            correlation=correlation or StreamItemCorrelation(),
            text_delta=text_delta,
            data=data,
            metadata={} if metadata is None else metadata,
        )

    def _next_sequence(self) -> int:
        sequence = self.sequence
        self.sequence += 1
        return sequence


@dataclass(slots=True)
class _MCPStreamProjectionState:
    accumulator: ProtocolStreamAccumulator
    tool_summaries: dict[str, dict[str, JSONValue]]
    resources: dict[str, MCPResource]
    resource_store: MCPResourceStore
    base_path: str
    legacy_adapter: _MCPLegacyStreamAdapter = field(
        default_factory=_MCPLegacyStreamAdapter
    )
    projection_state: ProtocolStreamProjectionState = field(init=False)

    def __post_init__(self) -> None:
        self.projection_state = ProtocolStreamProjectionState(
            stream_session_id=self.legacy_adapter.stream_session_id,
            run_id=self.legacy_adapter.run_id,
            turn_id=self.legacy_adapter.turn_id,
            accumulate=False,
        )

    @property
    def has_canonical_items(self) -> bool:
        return self.projection_state.has_canonical_items

    @property
    def legacy_stream_seen(self) -> bool:
        return self.projection_state.legacy_stream_seen


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
            request,
            {
                "initialize",
                "ping",
                "tools/list",
                "tools/call",
                "notifications/initialized",
            },
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
        if method == "notifications/initialized":
            return _handle_initialized_notification(logger, message)

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

    @router.post("/notifications/initialized")
    async def mcp_initialized_notification(
        request: Request,
        logger: Logger = Depends(di_get_logger),
    ) -> Response:
        assert isinstance(logger, Logger)
        message, _ = await _expect_jsonrpc_message(
            request, {"notifications/initialized"}
        )
        return _handle_initialized_notification(logger, message)

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
        method, params, allowed_tool_name=allowed_tool_name
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
    request: Request, allowed_methods: set[AllowedMethod]
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


def _server_info(request: Request) -> JSONObject:
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


def _build_chat_request(
    tool_request: MCPToolRequest, orchestrator: Orchestrator
) -> ChatCompletionRequest:
    model_id = resolve_model_id(orchestrator)
    return ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatMessage(
                role=MessageRole.USER, content=tool_request.input_string
            )
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
) -> Response:
    chat_request = _build_chat_request(tool_request, orchestrator)
    response, response_uuid, timestamp = await orchestrate(
        chat_request, logger, orchestrator
    )
    response_typed = cast(StreamResponse, response)
    response_uuid_obj = (
        response_uuid
        if isinstance(response_uuid, UUID)
        else UUID(str(response_uuid))
    )

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
                response_id=response_uuid_obj,
                timestamp=timestamp,
                progress_token=progress_token,
                orchestrator=orchestrator,
                logger=logger,
                resource_store=resource_store,
                base_path=base_path,
                cancel_event=cancel_event,
            ):
                yield sse_bytes(chunk)
        finally:
            watcher.cancel()
            with suppress(Exception):
                await watcher

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=sse_headers()
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


def _handle_initialized_notification(
    logger: Logger,
    message: JSONObject,
) -> Response:
    if "id" in message:
        raise HTTPException(
            status_code=400, detail="MCP notifications cannot include an id"
        )

    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    logger.debug("Handled MCP initialized notification")
    return Response(status_code=204)


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
    result: dict[str, JSONValue] = {"tools": cast(JSONValue, tools)}
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
        return arguments

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


async def _cleanup_mcp_stream_sources(
    logger: Logger, *sources: object, cancelled: bool
) -> None:
    try:
        await cleanup_stream_sources(*sources, cancelled=cancelled)
    except BaseExceptionGroup as exc:
        logger.exception("MCP stream source cleanup failed", exc_info=exc)
    except (Exception, CancelledError) as exc:
        logger.exception("MCP stream source cleanup failed", exc_info=exc)


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
    state = _MCPStreamProjectionState(
        accumulator=ProtocolStreamAccumulator(),
        tool_summaries={},
        resources={},
        resource_store=resource_store,
        base_path=base_path,
    )
    finished_normally = False
    response_iterator: AsyncIterator[ResponseItem] | None = None
    stream_error_message: JSONObject | None = None

    def emit(message: JSONObject) -> Iterator[bytes]:
        encoded = (
            dumps(cast(Mapping[str, object], message), separators=(",", ":"))
            + "\n"
        )
        yield encoded.encode("utf-8")

    try:
        response_iterator = stream_consumer_iterator(
            response,
            stream_session_id="mcp-stream",
            run_id=str(response_id),
            turn_id="mcp-turn",
        )
        async for item in cancellable_stream_iterator(
            response_iterator, cancel_event
        ):
            for notification in await _mcp_notifications(
                item, state, progress_token
            ):
                for payload in emit(notification):
                    yield payload

        finished_normally = not cancel_event.is_set()
    except GeneratorExit:
        cancel_event.set()
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=True
        )
        await _close_mcp_resource_notifications(
            resource_store, state.resources
        )
        await resource_store.prune_closed()
        await orchestrator.sync_messages()
        raise
    except CancelledError:
        cancel_event.set()
        finished_normally = False
    except Exception as exc:
        logger.exception("Error while streaming MCP response", exc_info=exc)
        cancel_event.set()
        finished_normally = False
        stream_error_message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": "An internal server error occurred.",
            },
        }

    if cancel_event.is_set():
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=True
        )
        cancel_error_message: JSONObject = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": "Request cancelled"},
        }
        error_message = stream_error_message or cancel_error_message
        terminal_messages = await _collect_terminal_mcp_messages(
            resource_store, state.resources, error_message
        )
        try:
            for message in terminal_messages:
                for payload in emit(message):
                    yield payload
        finally:
            await orchestrator.sync_messages()
        return

    if finished_normally:
        snapshot = state.accumulator.snapshot()
        if state.has_canonical_items:
            try:
                state.accumulator.validate_complete()
            except StreamValidationError as exc:
                logger.exception("Invalid MCP canonical stream", exc_info=exc)
                await _cleanup_mcp_stream_sources(
                    logger, response, response_iterator, cancelled=False
                )
                validation_error_message: JSONObject = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "An internal server error occurred.",
                    },
                }
                terminal_messages = await _collect_terminal_mcp_messages(
                    resource_store,
                    state.resources,
                    validation_error_message,
                )
                try:
                    for message in terminal_messages:
                        for payload in emit(message):
                            yield payload
                finally:
                    await orchestrator.sync_messages()
                return
            if snapshot.terminal_outcome is StreamTerminalOutcome.CANCELLED:
                await _cleanup_mcp_stream_sources(
                    logger, response, response_iterator, cancelled=True
                )
                terminal_cancel_error_message: JSONObject = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": "Request cancelled"},
                }
                terminal_messages = await _collect_terminal_mcp_messages(
                    resource_store,
                    state.resources,
                    terminal_cancel_error_message,
                )
                try:
                    for message in terminal_messages:
                        for payload in emit(message):
                            yield payload
                finally:
                    await orchestrator.sync_messages()
                return
            if snapshot.terminal_outcome is StreamTerminalOutcome.ERRORED:
                await _cleanup_mcp_stream_sources(
                    logger, response, response_iterator, cancelled=False
                )
                error_message = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": _canonical_error_message(snapshot),
                    },
                }
                terminal_messages = await _collect_terminal_mcp_messages(
                    resource_store, state.resources, error_message
                )
                try:
                    for message in terminal_messages:
                        for payload in emit(message):
                            yield payload
                finally:
                    await orchestrator.sync_messages()
                return

        answer_text = snapshot.answer_text
        reasoning_text = snapshot.reasoning_text
        usage = snapshot.usage

        summary: dict[str, JSONValue] = {
            "id": str(response_id),
            "created": timestamp,
            "model": request_model.model,
            "usage": {
                "input_text_tokens": _usage_count(
                    usage, "input_text_tokens", response.input_token_count
                ),
                "output_text_tokens": _usage_count(
                    usage, "output_text_tokens", response.output_token_count
                ),
                "total_tokens": _usage_count(
                    usage,
                    "total_tokens",
                    (response.input_token_count + response.output_token_count),
                ),
            },
        }
        if reasoning_text:
            summary["reasoning"] = reasoning_text
        if state.has_canonical_items:
            _merge_canonical_tool_call_arguments(
                state.tool_summaries, snapshot.tool_call_arguments
            )
        if state.tool_summaries:
            summary["toolCalls"] = list(state.tool_summaries.values())

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
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=False
        )
        terminal_messages = await _collect_terminal_mcp_messages(
            resource_store, state.resources, cast(JSONObject, result_message)
        )
        try:
            if not state.has_canonical_items:
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
            for message in terminal_messages:
                for payload in emit(message):
                    yield payload
        finally:
            await orchestrator.sync_messages()


async def _mcp_notifications(
    item: ResponseItem,
    state: _MCPStreamProjectionState,
    progress_token: str,
) -> list[JSONObject]:
    return await _mcp_stream_item_notifications(item, state, progress_token)


async def _mcp_stream_item_notifications(
    item: ResponseItem,
    state: _MCPStreamProjectionState,
    progress_token: str,
) -> list[JSONObject]:
    sequence = (
        item.sequence
        if isinstance(item, CanonicalStreamItem)
        else state.legacy_adapter.sequence
    )
    projections = state.projection_state.project_many(
        item,
        sequence,
        unsupported_message="unsupported MCP stream item",
    )
    notifications: list[JSONObject] = []
    for projection in projections:
        notifications.extend(
            await _mcp_canonical_stream_item_notifications(
                canonical_item_from_consumer_projection(projection),
                state,
                progress_token,
            )
        )
    return notifications


def _canonical_items_from_mcp_event(
    event: Event,
    state: _MCPStreamProjectionState,
) -> tuple[CanonicalStreamItem, ...]:
    return state.legacy_adapter.canonical_items_from_event(event)


def _legacy_tool_execution_output_items(
    tool_call_id: str,
    result: JSONValue,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, text)
        for name, text in _extract_append_streams(
            tool_call_id, result
        ).values()
    )


def _event_timings(event: Event) -> dict[str, LooseJsonValue]:
    return {
        "started": event.started,
        "finished": event.finished,
        "elapsed": event.elapsed,
    }


def _tool_metadata(item: Mapping[str, JSONValue]) -> dict[str, LooseJsonValue]:
    name = item.get("name")
    return {"tool_name": name} if isinstance(name, str) and name else {}


async def _mcp_canonical_stream_item_notifications(
    item: CanonicalStreamItem,
    state: _MCPStreamProjectionState,
    progress_token: str,
) -> list[JSONObject]:
    notifications: list[JSONObject] = []

    state.accumulator.add(item)
    if item.kind is StreamItemKind.TOOL_CALL_READY:
        _record_canonical_tool_call_ready(item, state.tool_summaries)
        return notifications
    if item.kind is StreamItemKind.REASONING_DELTA:
        reasoning_delta = _canonical_reasoning_delta(item)
        if reasoning_delta:
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {
                        "level": "debug",
                        "message": {
                            "type": "reasoning",
                            "delta": reasoning_delta,
                        },
                    },
                }
            )
        return notifications

    token_notification = _canonical_tool_notification(item)
    if token_notification is not None:
        notifications.append(token_notification)
        return notifications
    tool_execution_notification = _canonical_tool_execution_notification(
        item, state.tool_summaries
    )
    if tool_execution_notification is not None:
        notifications.append(tool_execution_notification)
        return notifications
    async for resource_notification in _canonical_tool_resource_notifications(
        item=item,
        tool_summaries=state.tool_summaries,
        resources=state.resources,
        resource_store=state.resource_store,
        base_path=state.base_path,
    ):
        notifications.append(resource_notification)
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        return notifications
    progress_notification = _canonical_progress_notification(
        item, progress_token
    )
    if progress_notification is not None:
        notifications.append(progress_notification)
        return notifications
    if item.kind is not StreamItemKind.ANSWER_DELTA:
        return notifications

    return notifications


def _record_canonical_tool_call_ready(
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
) -> None:
    assert item.kind is StreamItemKind.TOOL_CALL_READY
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    data = item.data if isinstance(item.data, dict) else {}
    name = data.get("name")
    arguments = data.get("arguments")
    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "name": None,
            "arguments": None,
        },
    )
    if isinstance(name, str) and name:
        tool_summary["name"] = name
    if arguments is not None:
        tool_summary["arguments"] = cast(JSONValue, arguments)


def _canonical_progress_notification(
    item: CanonicalStreamItem,
    progress_token: str,
) -> JSONObject | None:
    if item.kind is StreamItemKind.ANSWER_DELTA:
        delta = item.text_delta or ""
        if not delta:
            return None
        progress: dict[str, JSONValue] = {
            "type": "answer.delta",
            "delta": delta,
        }
    elif item.kind is StreamItemKind.STREAM_COMPLETED:
        progress = {"type": "answer.completed"}
    elif item.kind is StreamItemKind.STREAM_CANCELLED:
        progress = {"type": "stream.cancelled"}
    elif item.kind is StreamItemKind.STREAM_ERRORED:
        progress = {"type": "stream.errored"}
    else:
        return None
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": progress,
        },
    }


def _canonical_error_message(snapshot: ProtocolStreamSnapshot) -> str:
    terminal = next(
        (
            item
            for item in reversed(snapshot.control_items)
            if item.kind is StreamItemKind.STREAM_ERRORED
        ),
        None,
    )
    if terminal is not None and isinstance(terminal.data, dict):
        message = terminal.data.get("message")
        if isinstance(message, str) and message:
            return message
    return "Stream errored."


def _canonical_reasoning_delta(item: CanonicalStreamItem) -> str | None:
    if item.kind is StreamItemKind.REASONING_DELTA:
        return item.text_delta or ""
    if item.kind in (
        StreamItemKind.REASONING_DONE,
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_CLOSED,
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        return ""
    return None


def _canonical_tool_notification(
    item: CanonicalStreamItem,
) -> JSONObject | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    delta = item.text_delta or ""
    tool_call_id = item.correlation.tool_call_id
    data = item.data if isinstance(item.data, dict) else {}
    name = cast(JSONValue, data.get("name"))
    arguments = cast(JSONValue, data.get("arguments"))
    has_call_metadata = "name" in data or "arguments" in data
    if not delta:
        if has_call_metadata and isinstance(tool_call_id, str):
            return _tool_call_notification(
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
            )
        return None
    message: dict[str, JSONValue] = {
        "type": "tool.input_delta",
        "delta": delta,
    }
    if tool_call_id is not None:
        message["toolCallId"] = tool_call_id
    if "name" in data:
        message["name"] = name
    if "arguments" in data:
        message["arguments"] = arguments
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": message,
        },
    }


def _canonical_tool_execution_notification(
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
) -> JSONObject | None:
    if item.kind not in (
        StreamItemKind.STREAM_DIAGNOSTIC,
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
    ):
        return None
    data = item.data if isinstance(item.data, dict) else {}
    tool_call_id = item.correlation.tool_call_id
    if not isinstance(tool_call_id, str):
        return None
    name = cast(JSONValue, data.get("name"))
    arguments = cast(JSONValue, data.get("arguments"))
    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {"id": tool_call_id, "name": name, "arguments": arguments},
    )
    if isinstance(name, str) and name:
        tool_summary["name"] = name
    if arguments is not None:
        tool_summary["arguments"] = arguments

    if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
        timings = cast(JSONValue, data.get("timings"))
        if isinstance(timings, dict):
            tool_summary["started"] = timings.get("started")
        if item.metadata.get("mcp.synthetic_tool_execution_start") is True:
            return None
        return _tool_call_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )
    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
        diagnostic = cast(JSONValue, data.get("diagnostic"))
        tool_summary["diagnostic"] = diagnostic
        return _tool_diagnostic_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
            diagnostic=diagnostic,
            timings=cast(JSONValue, data.get("timings")),
        )

    payload_key = (
        "error"
        if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        else "result"
    )
    payload_value = cast(JSONValue, data.get(payload_key))
    tool_summary[payload_key] = payload_value
    return _tool_result_notification(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments,
        result=payload_value if payload_key == "result" else None,
        error=payload_value if payload_key == "error" else None,
        timings=cast(JSONValue, data.get("timings")),
    )


def _tool_call_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
) -> JSONObject:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": {
                "type": "tool.call",
                "toolCallId": tool_call_id,
                "name": name,
                "arguments": arguments,
            },
        },
    }


def _tool_diagnostic_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
    diagnostic: JSONValue,
    timings: JSONValue,
) -> JSONObject:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "warning",
            "message": {
                "type": "tool.diagnostic",
                "toolCallId": tool_call_id,
                "name": name,
                "arguments": arguments,
                "diagnostic": diagnostic,
                "timings": timings if isinstance(timings, dict) else {},
            },
        },
    }


def _tool_result_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
    result: JSONValue,
    error: JSONValue,
    timings: JSONValue,
) -> JSONObject:
    message: dict[str, JSONValue] = {
        "type": "tool.result",
        "toolCallId": tool_call_id,
        "name": name,
        "arguments": arguments,
        "timings": timings if isinstance(timings, dict) else {},
    }
    if error is not None:
        message["error"] = error
    elif result is not None:
        message["resultDelta"] = result
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "message": message,
        },
    }


async def _canonical_tool_resource_notifications(
    *,
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    resources: dict[str, MCPResource],
    resource_store: MCPResourceStore,
    base_path: str,
) -> AsyncIterator[JSONObject]:
    if item.kind not in (
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
    ):
        return

    tool_call_id = item.correlation.tool_call_id
    if tool_call_id is None:
        return
    data = item.data
    if not isinstance(data, dict):
        return
    category = data.get("category")
    content = _canonical_tool_resource_content(item)
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT and category not in {
        "stdout",
        "stderr",
        "log",
        "logs",
    }:
        return
    if (
        item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS
        and category != "progress"
    ):
        return
    if not content:
        return

    name = "logs" if category == "log" else str(category)
    resource_key = f"{tool_call_id}:{name}"
    resource = resources.get(resource_key)
    if resource is None:
        resource = await resource_store.create(
            base_path=base_path, initial_text=content
        )
    else:
        resource = await resource_store.append(resource.id, content)
    resources[resource_key] = resource

    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "name": _metadata_string(item.metadata, "tool_name"),
            "arguments": None,
        },
    )
    tool_name = _metadata_string(item.metadata, "tool_name")
    if tool_name and not tool_summary.get("name"):
        tool_summary["name"] = tool_name
    existing_resources = tool_summary.setdefault("resources", [])
    if isinstance(existing_resources, list):
        _append_tool_summary_resource(
            existing_resources, uri=resource.uri, name=name
        )

    yield _resource_notification(resource)


def _append_tool_summary_resource(
    resources: list[JSONValue],
    *,
    uri: str,
    name: str,
) -> None:
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if resource.get("uri") == uri and resource.get("name") == name:
            return
    resources.append({"uri": uri, "name": name})


def _canonical_tool_resource_content(item: CanonicalStreamItem) -> str:
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content", item.text_delta)
        return content if isinstance(content, str) else ""
    if item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content")
        if isinstance(content, str):
            return content
        progress = data.get("progress")
        return to_json({"progress": progress}) if progress is not None else ""
    return ""


def _usage_count(
    usage: object | None,
    key: str,
    fallback: int,
) -> int:
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def _metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _merge_canonical_tool_call_arguments(
    tool_summaries: dict[str, dict[str, JSONValue]],
    tool_call_arguments: Mapping[str, str],
) -> None:
    for tool_call_id, arguments in tool_call_arguments.items():
        tool_summary = tool_summaries.setdefault(
            tool_call_id,
            {
                "id": tool_call_id,
                "name": None,
                "arguments": arguments,
            },
        )
        tool_summary["arguments"] = arguments


async def _close_response_iterator(response: StreamResponse) -> None:
    iterator = getattr(response, "_response_iterator", None)
    if iterator and hasattr(iterator, "aclose"):
        try:
            await cast(SupportsAclose, iterator).aclose()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _resource_notification(resource: MCPResource) -> JSONObject:
    resource_payload: dict[str, JSONValue] = {
        "uri": resource.uri,
        "mimeType": resource.mime_type,
        "revision": resource.revision,
        "httpUri": resource.http_uri,
    }
    if resource.closed:
        resource_payload["closed"] = True
    else:
        resource_payload["delta"] = {"set": {"text": resource.text}}
    params: dict[str, JSONValue] = {"resources": [resource_payload]}
    return {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": params,
    }


async def _close_mcp_resource_notifications(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
) -> tuple[JSONObject, ...]:
    notifications: list[JSONObject] = []
    resource_ids = [resource.id for resource in resources.values()]
    for closed in await resource_store.close_many(resource_ids):
        notifications.append(_resource_notification(closed))
    return tuple(notifications)


async def _collect_terminal_mcp_messages(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
    terminal_message: JSONObject,
) -> tuple[JSONObject, ...]:
    messages = list(
        await _close_mcp_resource_notifications(resource_store, resources)
    )
    await resource_store.prune_closed()
    messages.append(terminal_message)
    return tuple(messages)


async def _terminal_mcp_messages(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
    terminal_message: JSONObject,
) -> AsyncIterator[JSONObject]:
    for message in await _collect_terminal_mcp_messages(
        resource_store, resources, terminal_message
    ):
        yield message


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
        if isinstance(tool_result, ToolCallDiagnostic):
            call = (
                event.payload.get("call")
                if isinstance(event.payload, dict)
                else None
            )
            return _tool_call_diagnostic_item(
                tool_result,
                call if isinstance(call, ToolCall) else None,
            )
        if isinstance(tool_result, ToolCallError):
            return {
                "id": str(tool_result.call.id),
                "name": tool_result.name,
                "arguments": cast(JSONValue, tool_result.arguments),
                "error": cast(JSONValue, tool_call_error_payload(tool_result)),
            }
        if isinstance(tool_result, ToolCallResult):
            result: JSONValue = (
                cast(JSONValue, tool_result.result)
                if isinstance(
                    tool_result.result, (dict, list, str, int, float, bool)
                )
                else to_json(tool_result.result)
            )
            return {
                "id": str(tool_result.call.id),
                "name": tool_result.name,
                "arguments": cast(JSONValue, tool_result.arguments),
                "result": result,
            }
    if event.type is EventType.TOOL_DIAGNOSTIC:
        payload = event.payload if isinstance(event.payload, dict) else {}
        diagnostic = _payload_diagnostic(payload)
        if diagnostic is None:
            return None
        call = payload.get("call")
        return _tool_call_diagnostic_item(
            diagnostic,
            call if isinstance(call, ToolCall) else None,
        )
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
        "arguments": cast(JSONValue, call.arguments),
    }


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


def _tool_call_diagnostic_item(
    diagnostic: ToolCallDiagnostic, call: ToolCall | None
) -> dict[str, JSONValue]:
    diagnostic_payload = {
        "id": str(diagnostic.id),
        **tool_call_diagnostic_payload(diagnostic),
    }
    if diagnostic.call_id is not None:
        diagnostic_payload["call_id"] = str(diagnostic.call_id)
    return {
        "id": str(call.id if call else diagnostic.call_id or diagnostic.id),
        "name": (
            call.name
            if call
            else diagnostic.canonical_name
            or diagnostic.requested_name
            or "tool"
        ),
        "arguments": cast(JSONValue, call.arguments if call else None),
        "diagnostic": cast(JSONValue, diagnostic_payload),
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
