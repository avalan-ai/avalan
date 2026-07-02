from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallResult,
    ToolDescriptor,
    ToolValue,
)
from ...model.stream import (
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
)
from ...server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    ContentFile,
    ContentImage,
    ContentText,
    MCPToolRequest,
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionSettings,
    coerce_server_output_redaction_settings,
    sanitize_model_visible_server_protocol_text,
    sanitize_server_protocol_text,
    sanitize_server_protocol_value,
    server_output_redaction_settings_from_state,
)
from ...types import JsonObject, JsonScalar, MutableJsonValue
from ...utils import to_json
from ..container_policy import (
    RemoteContainerRequestError,
    remote_container_policy_from_state,
    validate_remote_container_arguments,
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
    ProtocolStreamSnapshot,
    cancellable_stream_iterator,
    canonical_flow_public_metadata,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
    protocol_stream_usage_mappings,
    stream_consumer_iterator,
)

from asyncio import CancelledError, Lock, create_task
from asyncio import Event as AsyncEvent
from contextlib import suppress
from dataclasses import dataclass, field, replace
from json import JSONDecodeError, dumps, loads
from logging import Logger
from typing import (
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

ResponseItem = CanonicalStreamItem | StreamConsumerProjection


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
        resource_byte_limit: int | None = None,
    ) -> None:
        retention_settings = protocol_stream_retention_settings(
            StreamRetentionPolicy()
        )
        if resource_item_limit is None:
            resource_item_limit = retention_settings.resource_item_limit
        if resource_limit is None:
            resource_limit = retention_settings.resource_item_limit
        if resource_byte_limit is None:
            resource_byte_limit = retention_settings.resource_text_byte_limit
        assert isinstance(resource_item_limit, int)
        assert not isinstance(resource_item_limit, bool)
        assert resource_item_limit >= 0
        assert isinstance(resource_limit, int)
        assert not isinstance(resource_limit, bool)
        assert resource_limit > 0
        assert isinstance(resource_byte_limit, int)
        assert not isinstance(resource_byte_limit, bool)
        assert resource_byte_limit > 0
        self._resources: dict[str, MCPResource] = {}
        self._resource_chunks: dict[str, list[str]] = {}
        self._resource_order: list[str] = []
        self._counter = 0
        self._resource_item_limit = resource_item_limit
        self._resource_limit = resource_limit
        self._resource_byte_limit = resource_byte_limit
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
        retained = list(chunks[-self._resource_item_limit :])
        bounded_reversed: list[str] = []
        remaining_bytes = self._resource_byte_limit

        for chunk in reversed(retained):
            chunk_size = len(chunk.encode("utf-8"))
            if chunk_size <= remaining_bytes:
                bounded_reversed.append(chunk)
                remaining_bytes -= chunk_size
                if remaining_bytes == 0:
                    break
                continue
            bounded = self._utf8_suffix(chunk, remaining_bytes)
            if bounded:
                bounded_reversed.append(bounded)
            break

        bounded_reversed.reverse()
        return bounded_reversed

    def _retained_text(self, chunks: list[str]) -> str:
        return "".join(self._retained_chunks(chunks))

    @staticmethod
    def _utf8_suffix(text: str, byte_limit: int) -> str:
        assert isinstance(byte_limit, int)
        assert not isinstance(byte_limit, bool)
        assert byte_limit > 0
        data = text.encode("utf-8")
        if len(data) <= byte_limit:
            return text

        start = len(data) - byte_limit
        while start < len(data) and (data[start] & 0b1100_0000) == 0b1000_0000:
            start += 1
        return data[start:].decode("utf-8")

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
class _MCPStreamProjectionState:
    accumulator: ProtocolStreamAccumulator
    tool_summaries: dict[str, dict[str, JSONValue]]
    resources: dict[str, MCPResource]
    resource_store: MCPResourceStore
    base_path: str
    output_redaction_settings: ServerOutputRedactionSettings = field(
        default_factory=ServerOutputRedactionSettings
    )
    answer_redactor: ModelVisibleServerProtocolTextRedactor = field(
        default_factory=ModelVisibleServerProtocolTextRedactor
    )
    reasoning_redactor: ModelVisibleServerProtocolTextRedactor = field(
        default_factory=ModelVisibleServerProtocolTextRedactor
    )


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
            if _is_direct_skills_tool_call(orchestrator, message):
                return await _handle_direct_skills_tool_call_message(
                    request, logger, orchestrator, message
                )
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
) -> tuple[str | int, MCPToolRequest, str | int]:
    call_message, messages = await _expect_jsonrpc_message(
        request, {"tools/call"}
    )
    return _parse_call_request(request, call_message, messages)


def _parse_call_request(
    request: Request,
    call_message: JSONObject,
    messages: AsyncIterator[JSONObject],
) -> tuple[str | int, MCPToolRequest, str | int]:
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
        container_request = validate_remote_container_arguments(
            arguments,
            policy=remote_container_policy_from_state(request.app.state),
        )
    except RemoteContainerRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    arguments = cast(dict[str, JSONValue], container_request.arguments)
    if container_request.profile is not None:
        request.state.mcp_container_profile = container_request.profile

    try:
        request_model = MCPToolRequest.model_validate(arguments)
    except Exception as exc:  # pragma: no cover - validation error path
        raise HTTPException(
            status_code=400, detail="Invalid MCP arguments"
        ) from exc

    progress_token = cast(str | int | None, params.get("progressToken"))
    if progress_token is None:
        meta = params.get("_meta")
        if isinstance(meta, dict):
            progress_token = cast(str | int | None, meta.get("progressToken"))
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
    content: str | list[ContentFile | ContentImage | ContentText]
    if tool_request.files:
        content = []
        if tool_request.input_string and tool_request.input_string.strip():
            content.append(
                ContentText(type="text", text=tool_request.input_string)
            )
        content.extend(
            ContentFile(type="file", file=file.as_content_file())
            for file in tool_request.files
        )
    else:
        content = tool_request.input_string or ""
    return ChatCompletionRequest(
        model=model_id,
        messages=[ChatMessage(role=MessageRole.USER, content=content)],
        stream=True,
    )


async def _start_tool_streaming_response(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    request_id: str | int,
    tool_request: MCPToolRequest,
    progress_token: str | int,
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
    output_redaction_settings = server_output_redaction_settings_from_state(
        request.app.state
    )

    if not chat_request.stream:
        try:
            text = sanitize_model_visible_server_protocol_text(
                await response_typed.to_str(),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
                channel="answer",
            )
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
                output_redaction_settings=output_redaction_settings,
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

    tools = _collect_tool_descriptions(request, orchestrator)
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


def _collect_tool_descriptions(
    request: Request,
    orchestrator: Orchestrator | None = None,
) -> list[dict[str, JSONValue]]:
    name = cast(str, getattr(request.app.state, "mcp_tool_name", "run"))
    description = cast(
        str,
        getattr(
            request.app.state,
            "mcp_tool_description",
            "Execute the Avalan orchestrator run endpoint.",
        ),
    )
    tools: list[dict[str, JSONValue]] = [
        {
            "name": name,
            "description": description,
            "inputSchema": MCPToolRequest.model_json_schema(),
        }
    ]
    if orchestrator is None:
        return tools
    tool_manager = getattr(orchestrator, "tool", None)
    list_tools = getattr(tool_manager, "list_tools", None)
    if not callable(list_tools):
        return tools
    for descriptor in list_tools():
        tool_description = _skills_tool_description(descriptor)
        if tool_description is not None:
            tools.append(tool_description)
    return tools


def _skills_tool_description(
    descriptor: ToolDescriptor,
) -> dict[str, JSONValue] | None:
    assert isinstance(descriptor, ToolDescriptor)
    name = descriptor.name
    if not name.startswith("skills."):
        return None
    schema = descriptor.schema or descriptor.provider_safe_schema or {}
    function = schema.get("function") if isinstance(schema, dict) else None
    if not isinstance(function, dict):
        function = {}
    description = function.get("description")
    if not isinstance(description, str):
        description = ""
    input_schema = function.get("parameters")
    if not isinstance(input_schema, dict):
        input_schema = descriptor.parameter_schema
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": description,
        "inputSchema": cast(JSONValue, input_schema),
    }


def _is_direct_skills_tool_call(
    orchestrator: Orchestrator,
    message: JSONObject,
) -> bool:
    params = message.get("params")
    if not isinstance(params, dict):
        return False
    name = params.get("name")
    if not isinstance(name, str) or not name.startswith("skills."):
        return False
    resolution = orchestrator.tool.resolve_tool_name(name)
    return (
        resolution.canonical_name is not None
        and resolution.canonical_name.startswith("skills.")
    )


async def _handle_direct_skills_tool_call_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")
    name = params.get("name")
    if not isinstance(name, str) or not name.startswith("skills."):
        raise HTTPException(
            status_code=400, detail=f'Unsupported tool "{name}"'
        )
    resolution = orchestrator.tool.resolve_tool_name(name)
    canonical_name = resolution.canonical_name
    if canonical_name is None or not canonical_name.startswith("skills."):
        raise HTTPException(
            status_code=400, detail=f'Unsupported tool "{name}"'
        )
    raw_arguments = params.get("arguments")
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise HTTPException(status_code=400, detail="Invalid tool arguments")
    try:
        container_request = validate_remote_container_arguments(
            raw_arguments,
            policy=remote_container_policy_from_state(request.app.state),
        )
    except RemoteContainerRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    arguments = container_request.arguments

    call = ToolCall(
        id=str(uuid4()),
        name=canonical_name,
        arguments=cast(dict[str, ToolValue], arguments),
    )
    context = _direct_tool_call_context(request, orchestrator, call)
    outcome = await orchestrator.tool.execute_call(call, context)
    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload = _direct_tool_call_jsonrpc_result(
        request_id=response_id,
        tool_name=canonical_name,
        outcome=outcome,
        output_redaction_settings=(
            server_output_redaction_settings_from_state(request.app.state)
        ),
    )
    logger.debug(
        "Handled direct MCP skills tool call",
        extra={
            "response_id": response_id,
            "tool_name": canonical_name,
        },
    )
    return JSONResponse(payload)


def _direct_tool_call_context(
    request: Request,
    orchestrator: Orchestrator,
    call: ToolCall,
) -> ToolCallContext:
    context = getattr(request.app.state, "ctx", None)
    participant_id = getattr(context, "participant_id", None)
    agent_id = getattr(orchestrator, "_id", None)
    return ToolCallContext(
        agent_id=agent_id if isinstance(agent_id, UUID) else None,
        participant_id=(
            participant_id if isinstance(participant_id, UUID) else None
        ),
        calls=[call],
    )


def _direct_tool_call_jsonrpc_result(
    *,
    request_id: str | int,
    tool_name: str,
    outcome: ToolCallResult | ToolCallError | ToolCallDiagnostic,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONRPCResult:
    structured = _direct_tool_outcome_structured_content(
        outcome,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
    )
    content_text = dumps(structured, separators=(",", ":"), sort_keys=True)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": content_text}],
            "structuredContent": structured,
        },
    }


def _direct_tool_outcome_structured_content(
    outcome: ToolCallResult | ToolCallError | ToolCallDiagnostic,
    *,
    tool_name: str,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONObject:
    if isinstance(outcome, ToolCallResult):
        return {
            "type": "tool.result",
            "toolCallId": str(outcome.call.id or outcome.id),
            "name": outcome.name,
            "result": cast(
                JSONValue,
                sanitize_server_protocol_value(
                    outcome.result,
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        }
    if isinstance(outcome, ToolCallError):
        return {
            "type": "tool.error",
            "toolCallId": str(outcome.call.id or outcome.id),
            "name": outcome.name,
            "message": sanitize_server_protocol_text(
                outcome.message,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
            "error": cast(
                JSONValue,
                sanitize_server_protocol_value(
                    outcome.error,
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        }
    return {
        "type": "tool.diagnostic",
        "toolCallId": str(outcome.call_id or outcome.id),
        "name": outcome.canonical_name or tool_name,
        "diagnostic": cast(
            JSONValue,
            sanitize_server_protocol_value(
                _tool_diagnostic_payload(
                    outcome,
                    output_redaction_settings=output_redaction_settings,
                ),
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
    }


def _tool_diagnostic_payload(
    diagnostic: ToolCallDiagnostic,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> dict[str, JSONValue]:
    return {
        "id": str(diagnostic.id),
        "call_id": (
            str(diagnostic.call_id) if diagnostic.call_id is not None else None
        ),
        "requested_name": diagnostic.requested_name,
        "canonical_name": diagnostic.canonical_name,
        "status": diagnostic.status.value,
        "code": diagnostic.code.value,
        "stage": diagnostic.stage.value,
        "message": sanitize_server_protocol_text(
            diagnostic.message,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
        "retryable": diagnostic.retryable,
        "details": cast(
            JSONValue,
            sanitize_server_protocol_value(
                diagnostic.details,
                tool_name=diagnostic.canonical_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
    }


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
    progress_token: str | int,
    orchestrator: Orchestrator,
    logger: Logger,
    resource_store: MCPResourceStore,
    base_path: str,
    cancel_event: AsyncEvent,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> AsyncIterator[bytes]:
    output_redaction_settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    state = _MCPStreamProjectionState(
        accumulator=ProtocolStreamAccumulator(),
        tool_summaries={},
        resources={},
        resource_store=resource_store,
        base_path=base_path,
        output_redaction_settings=output_redaction_settings,
        answer_redactor=ModelVisibleServerProtocolTextRedactor(
            output_redaction_settings,
            protocol="mcp",
            channel="answer",
        ),
        reasoning_redactor=ModelVisibleServerProtocolTextRedactor(
            output_redaction_settings,
            protocol="mcp",
            channel="reasoning",
        ),
    )
    finished_normally = False
    response_iterator: AsyncIterator[StreamConsumerProjection] | None = None
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
            unsupported_message="unsupported MCP stream item",
            close_source_on_generator_exit=False,
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

        snapshot = state.accumulator.snapshot()
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
                    "message": _canonical_error_message(
                        snapshot,
                        output_redaction_settings=(
                            state.output_redaction_settings
                        ),
                    ),
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

        answer_text = sanitize_model_visible_server_protocol_text(
            snapshot.answer_text,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
            channel="answer",
        )
        reasoning_text = sanitize_model_visible_server_protocol_text(
            snapshot.reasoning_text,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
            channel="reasoning",
        )
        usage = snapshot.usage

        summary: dict[str, JSONValue] = {
            "id": str(response_id),
            "created": timestamp,
            "model": request_model.model,
            "usage": {
                "input_text_tokens": _usage_count(
                    usage,
                    "input_text_tokens",
                    response.input_token_count,
                    aliases=("input_tokens",),
                ),
                "output_text_tokens": _usage_count(
                    usage,
                    "output_text_tokens",
                    response.output_token_count,
                    aliases=("output_tokens",),
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
        _merge_canonical_tool_call_arguments(
            state.tool_summaries,
            snapshot.tool_call_arguments,
            output_redaction_settings=state.output_redaction_settings,
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
            for message in terminal_messages:
                for payload in emit(message):
                    yield payload
        finally:
            await orchestrator.sync_messages()


async def _mcp_notifications(
    item: StreamConsumerProjection,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    return await _mcp_stream_item_notifications(item, state, progress_token)


async def _mcp_stream_item_notifications(
    item: StreamConsumerProjection,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    return await _mcp_canonical_stream_item_notifications(
        canonical_item_from_consumer_projection(item),
        state,
        progress_token,
    )


async def _mcp_canonical_stream_item_notifications(
    item: CanonicalStreamItem,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    notifications: list[JSONObject] = []

    state.accumulator.add(item)
    if item.is_stream_terminal:
        notifications.extend(
            _mcp_model_text_flush_notifications(
                state,
                progress_token,
                item.sequence,
            )
        )
    if item.kind is StreamItemKind.FLOW_EVENT:
        notifications.append(_canonical_flow_notification(item))
        return notifications
    if item.kind is StreamItemKind.TOOL_CALL_READY:
        _record_canonical_tool_call_ready(
            item,
            state.tool_summaries,
            output_redaction_settings=state.output_redaction_settings,
        )
        return notifications
    if item.kind is StreamItemKind.REASONING_DELTA:
        reasoning_deltas = _canonical_reasoning_deltas(
            item,
            state.reasoning_redactor,
        )
        if reasoning_deltas is None:
            return notifications
        for reasoning_delta in reasoning_deltas:
            if reasoning_delta:
                notifications.append(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {
                            "level": "debug",
                            "data": {
                                "type": "reasoning",
                                "delta": reasoning_delta,
                            },
                        },
                    }
                )
        return notifications

    token_notification = _canonical_tool_notification(
        item,
        output_redaction_settings=state.output_redaction_settings,
    )
    if token_notification is not None:
        notifications.append(token_notification)
        return notifications
    tool_execution_notification = _canonical_tool_execution_notification(
        item,
        state.tool_summaries,
        output_redaction_settings=state.output_redaction_settings,
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
        output_redaction_settings=state.output_redaction_settings,
    ):
        notifications.append(resource_notification)
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        return notifications
    progress_notification = _canonical_progress_notification(
        item,
        progress_token,
        state.answer_redactor,
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
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> None:
    assert item.kind is StreamItemKind.TOOL_CALL_READY
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    data = item.data if isinstance(item.data, dict) else {}
    name = data.get("name")
    arguments = data.get("arguments")
    tool_name = name if isinstance(name, str) else None
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
        tool_summary["arguments"] = cast(
            JSONValue,
            sanitize_server_protocol_value(
                arguments,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )


def _canonical_progress_notification(
    item: CanonicalStreamItem,
    progress_token: str | int,
    redactor: ModelVisibleServerProtocolTextRedactor | None = None,
) -> JSONObject | None:
    if item.kind is StreamItemKind.ANSWER_DELTA:
        deltas = _model_visible_stream_deltas(
            item.text_delta or "",
            redactor,
        )
        if not deltas:
            return None
        message: dict[str, JSONValue] = {
            "type": "answer.delta",
            "delta": "".join(deltas),
        }
    elif item.kind is StreamItemKind.STREAM_COMPLETED:
        message = {"type": "answer.completed"}
    elif item.kind is StreamItemKind.STREAM_CANCELLED:
        message = {"type": "stream.cancelled"}
    elif item.kind is StreamItemKind.STREAM_ERRORED:
        message = {"type": "stream.errored"}
    else:
        return None
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": item.sequence,
            "message": dumps(message, separators=(",", ":")),
        },
    }


def _mcp_model_text_flush_notifications(
    state: _MCPStreamProjectionState,
    progress_token: str | int,
    progress: int,
) -> list[JSONObject]:
    assert isinstance(state, _MCPStreamProjectionState)
    assert isinstance(progress, int) and not isinstance(progress, bool)
    notifications: list[JSONObject] = []
    for reasoning_delta in state.reasoning_redactor.flush():
        if reasoning_delta:
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {
                        "level": "debug",
                        "data": {
                            "type": "reasoning",
                            "delta": reasoning_delta,
                        },
                    },
                }
            )
    for answer_delta in state.answer_redactor.flush():
        if answer_delta:
            message: dict[str, JSONValue] = {
                "type": "answer.delta",
                "delta": answer_delta,
            }
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": progress_token,
                        "progress": progress,
                        "message": dumps(message, separators=(",", ":")),
                    },
                }
            )
    return notifications


def _canonical_error_message(
    snapshot: ProtocolStreamSnapshot,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    terminal = snapshot.terminal_snapshot
    if terminal.outcome is StreamTerminalOutcome.ERRORED and isinstance(
        terminal.data, dict
    ):
        message = terminal.data.get("message")
        if isinstance(message, str) and message:
            return sanitize_server_protocol_text(
                message,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            )
    return "Stream errored."


def _canonical_reasoning_deltas(
    item: CanonicalStreamItem,
    redactor: ModelVisibleServerProtocolTextRedactor | None = None,
) -> tuple[str, ...] | None:
    if item.kind is StreamItemKind.REASONING_DELTA:
        return _model_visible_stream_deltas(
            item.text_delta or "",
            redactor,
        )
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
        return ()
    return None


def _canonical_reasoning_delta(item: CanonicalStreamItem) -> str | None:
    deltas = _canonical_reasoning_deltas(item)
    if deltas is None:
        return None
    return "".join(deltas)


def _model_visible_stream_deltas(
    value: str,
    redactor: ModelVisibleServerProtocolTextRedactor | None,
) -> tuple[str, ...]:
    if redactor is not None:
        return redactor.push(value)
    sanitized = sanitize_model_visible_server_protocol_text(value)
    return (sanitized,) if sanitized else ()


def _canonical_tool_notification(
    item: CanonicalStreamItem,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONObject | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    delta = sanitize_server_protocol_text(
        item.text_delta or "",
        output_redaction_settings=output_redaction_settings,
        protocol="mcp",
    )
    tool_call_id = item.correlation.tool_call_id
    data = item.data if isinstance(item.data, dict) else {}
    tool_name = _protocol_tool_name(item, data)
    name = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("name"),
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    arguments = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("arguments"),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
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
            "data": message,
        },
    }


def _canonical_flow_notification(item: CanonicalStreamItem) -> JSONObject:
    assert item.kind is StreamItemKind.FLOW_EVENT
    metadata = canonical_flow_public_metadata(item)
    message: dict[str, JSONValue] = {
        "type": "flow.event",
        "sequence": item.sequence,
        "metadata": cast(JSONValue, metadata),
    }
    event_type = metadata.get("event_type")
    if isinstance(event_type, str) and event_type:
        message["event"] = event_type
    flow_run_id = item.correlation.flow_run_id
    if flow_run_id is not None:
        message["flowRunId"] = flow_run_id
    node_id = item.correlation.node_id
    if node_id is not None:
        message["nodeId"] = node_id
    parent_sequence = item.correlation.parent_sequence
    if parent_sequence is not None:
        message["parentSequence"] = parent_sequence
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "data": message,
        },
    }


def _canonical_tool_execution_notification(
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
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
    tool_name = _protocol_tool_name(item, data)
    name = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("name"),
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    arguments = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("arguments"),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {"id": tool_call_id, "name": name, "arguments": arguments},
    )
    if isinstance(tool_name, str) and tool_name:
        tool_summary["name"] = name
    if arguments is not None:
        tool_summary["arguments"] = arguments

    if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
        timings = cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("timings"),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )
        if isinstance(timings, dict):
            tool_summary["started"] = timings.get("started")
        return _tool_call_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )
    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
        diagnostic = cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("diagnostic"),
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )
        tool_summary["diagnostic"] = diagnostic
        return _tool_diagnostic_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
            diagnostic=diagnostic,
            timings=cast(
                JSONValue,
                sanitize_server_protocol_value(
                    data.get("timings"),
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        )

    payload_key = (
        "error"
        if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        else "result"
    )
    payload_value = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get(payload_key),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    tool_summary[payload_key] = payload_value
    return _tool_result_notification(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments,
        result=payload_value if payload_key == "result" else None,
        error=payload_value if payload_key == "error" else None,
        timings=cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("timings"),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
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
            "data": {
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
            "data": {
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
            "data": message,
        },
    }


async def _canonical_tool_resource_notifications(
    *,
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    resources: dict[str, MCPResource],
    resource_store: MCPResourceStore,
    base_path: str,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
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
    tool_name = _protocol_tool_name(item, data)
    content = _canonical_tool_resource_content(
        item,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
    )
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
        stored_resource = await resource_store.create(
            base_path=base_path, initial_text=content
        )
        resource = replace(stored_resource, text=content)
    else:
        stored_resource = await resource_store.append(resource.id, content)
        # The shared store is lossy retained history; this per-request copy
        # stays lossless while the active MCP response is being emitted.
        resource = replace(stored_resource, text=resource.text + content)
    resources[resource_key] = resource

    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "name": tool_name,
            "arguments": None,
        },
    )
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


def _canonical_tool_resource_content(
    item: CanonicalStreamItem,
    *,
    tool_name: str | None = None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content", item.text_delta)
        if not isinstance(content, str):
            return ""
        return _mcp_protocol_resource_text(
            content,
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    if item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content")
        if isinstance(content, str):
            return _mcp_protocol_resource_text(
                content,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
        progress = data.get("progress")
        return (
            to_json(
                sanitize_server_protocol_value(
                    {"progress": progress},
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                )
            )
            if progress is not None
            else ""
        )
    return ""


def _mcp_protocol_resource_text(
    value: str,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    if isinstance(tool_name, str) and tool_name.startswith("skills."):
        return to_json(
            sanitize_server_protocol_value(
                {"content": value},
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            )
        )
    return sanitize_server_protocol_text(
        value,
        output_redaction_settings=output_redaction_settings,
        protocol="mcp",
    )


def _usage_count(
    usage: object | None,
    key: str,
    fallback: int,
    *,
    aliases: tuple[str, ...] = (),
) -> int:
    for usage_mapping in protocol_stream_usage_mappings(usage):
        for usage_key in (key, *aliases):
            value = usage_mapping.get(usage_key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return fallback


def _metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _protocol_tool_name(
    item: CanonicalStreamItem,
    data: Mapping[str, object],
) -> str | None:
    name = data.get("name")
    if isinstance(name, str) and name:
        return name
    return _metadata_string(item.metadata, "tool_name")


def _merge_canonical_tool_call_arguments(
    tool_summaries: dict[str, dict[str, JSONValue]],
    tool_call_arguments: Mapping[str, str],
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
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
        raw_tool_name = tool_summary.get("name")
        tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None
        tool_summary["arguments"] = cast(
            JSONValue,
            sanitize_server_protocol_value(
                arguments,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )


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
    params: dict[str, JSONValue] = {
        "uri": resource.uri,
        "resources": [resource_payload],
    }
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
