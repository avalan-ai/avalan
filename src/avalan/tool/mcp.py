from ..compat import override
from ..entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from . import Tool, ToolSet
from .builtin_display import project_mcp_call_tool_display

from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack
from importlib import import_module
from json import JSONDecodeError, dumps, loads
from typing import Protocol, cast
from uuid import uuid4

JSONValue = dict[str, object] | list[object] | str | int | float | bool | None
JSONObject = dict[str, JSONValue]


class _MCPHTTPResponse(Protocol):
    headers: Mapping[str, str]

    async def aread(self) -> bytes: ...

    def aiter_lines(self) -> AsyncIterator[str]: ...

    def raise_for_status(self) -> None: ...


class McpCallTool(Tool):
    """Call an MCP server tool using the MCP client.

    Args:
        uri: Base URI of the MCP server.
        name: Name of the tool to invoke.
        arguments: Arguments to send to the tool.

    Returns:
        Responses returned by the MCP tool invocation.
    """

    _client_params: dict[str, object]
    _call_params: dict[str, object]

    def __init__(
        self,
        *,
        client_params: dict[str, object] | None = None,
        call_params: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.__name__ = "call"
        self._client_params = client_params or {}
        self._call_params = call_params or {}

    def tool_display_projector(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome | None = None,
    ) -> object | None:
        return project_mcp_call_tool_display(call=call, outcome=outcome)

    async def __call__(
        self,
        uri: str,
        name: str,
        arguments: dict[str, object] | None,
        *,
        context: ToolCallContext,
    ) -> dict[str, object]:
        assert uri
        assert name

        return await _call_streamable_http_mcp_tool(
            uri=uri,
            name=name,
            arguments=arguments or {},
            context=context,
            client_params=self._client_params,
            call_params=self._call_params,
        )


class McpToolSet(ToolSet):
    """Tool set providing MCP client functionality."""

    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = "mcp",
    ) -> None:
        tools = [McpCallTool()]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )


async def _call_streamable_http_mcp_tool(
    *,
    uri: str,
    name: str,
    arguments: dict[str, object],
    context: ToolCallContext,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
) -> dict[str, object]:
    httpx_module = import_module("httpx")
    client_factory = getattr(httpx_module, "AsyncClient")

    request_id = call_params.get("request_id") or str(uuid4())
    progress_token = call_params.get("progress_token") or request_id
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": cast(JSONValue, request_id),
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": cast(JSONValue, arguments),
            "progressToken": cast(JSONValue, progress_token),
            "_meta": {"progressToken": cast(JSONValue, progress_token)},
        },
    }
    request_params = {
        key: value
        for key, value in call_params.items()
        if key not in {"progress_token", "request_id"}
    }
    async with client_factory(**_client_options(client_params)) as client:
        async with client.stream(
            "POST",
            uri,
            json=payload,
            **request_params,
        ) as response:
            response.raise_for_status()
            async for message in _iter_mcp_response_messages(response):
                terminal = await _handle_mcp_message(
                    message, request_id=request_id, context=context
                )
                if terminal is not None:
                    return terminal

    raise RuntimeError("MCP response ended without a result")


def _client_options(client_params: Mapping[str, object]) -> dict[str, object]:
    options = dict(client_params)
    raw_headers = options.pop("headers", None)
    headers = (
        dict(cast(Mapping[str, str], raw_headers))
        if isinstance(raw_headers, Mapping)
        else {}
    )
    headers.setdefault("Accept", "application/json, text/event-stream")
    headers.setdefault("Content-Type", "application/json")
    options["headers"] = headers
    return options


async def _iter_mcp_response_messages(
    response: _MCPHTTPResponse,
) -> AsyncIterator[JSONObject]:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        body = await response.aread()
        yield _decode_json_object(body.decode("utf-8"))
        return

    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield _decode_sse_data(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line.startswith("{"):
            yield _decode_json_object(line)
    if data_lines:
        yield _decode_sse_data(data_lines)


def _decode_sse_data(data_lines: list[str]) -> JSONObject:
    return _decode_json_object("\n".join(data_lines))


def _decode_json_object(text: str) -> JSONObject:
    try:
        message = loads(text)
    except JSONDecodeError as exc:
        raise RuntimeError("Invalid MCP JSON-RPC response") from exc
    if not isinstance(message, dict):
        raise RuntimeError("Invalid MCP JSON-RPC response")
    return cast(JSONObject, message)


async def _handle_mcp_message(
    message: JSONObject,
    *,
    request_id: object,
    context: ToolCallContext,
) -> dict[str, object] | None:
    error = message.get("error")
    if isinstance(error, dict) and _matches_request_id(message, request_id):
        code = error.get("code")
        detail = error.get("message")
        raise RuntimeError(f"MCP tool call failed [{code}]: {detail}")

    result = message.get("result")
    if isinstance(result, dict) and _matches_request_id(message, request_id):
        return result

    method = message.get("method")
    if isinstance(method, str):
        await _forward_mcp_notification(method, message, context)
    return None


def _matches_request_id(message: JSONObject, request_id: object) -> bool:
    return message.get("id") == request_id or str(message.get("id")) == str(
        request_id
    )


async def _forward_mcp_notification(
    method: str, message: JSONObject, context: ToolCallContext
) -> None:
    if context.stream_event is None:
        return
    params = message.get("params")
    if not isinstance(params, dict):
        return
    params_json = cast(JSONObject, params)

    if method == "notifications/progress":
        await _forward_mcp_progress(params_json, context)
    elif method == "notifications/message":
        await _forward_mcp_message(params_json, context)
    elif method == "notifications/resources/updated":
        await _forward_mcp_resources(params_json, context)


async def _forward_mcp_progress(
    params: JSONObject, context: ToolCallContext
) -> None:
    payload = _progress_message_payload(params.get("message"))
    if not isinstance(payload, dict):
        return
    event_type = payload.get("type")
    if event_type == "answer.delta":
        delta = payload.get("delta")
        if isinstance(delta, str) and delta:
            await _emit_mcp_stream_event(
                context,
                kind=ToolExecutionStreamKind.STDOUT,
                content=delta,
                metadata={"mcp_method": "notifications/progress"},
            )
        return
    if event_type in {
        "answer.completed",
        "stream.cancelled",
        "stream.errored",
    }:
        await _emit_mcp_stream_event(
            context,
            kind=ToolExecutionStreamKind.PROGRESS,
            content=dumps(payload, separators=(",", ":")),
            progress=1 if event_type == "answer.completed" else None,
            metadata={
                "mcp_method": "notifications/progress",
                "mcp_type": event_type,
            },
        )


def _progress_message_payload(message: object) -> JSONObject | None:
    if isinstance(message, dict):
        return cast(JSONObject, message)
    if not isinstance(message, str):
        return None
    try:
        payload = loads(message)
    except JSONDecodeError:
        return {"type": "progress", "message": message}
    return cast(JSONObject, payload) if isinstance(payload, dict) else None


async def _forward_mcp_message(
    params: JSONObject, context: ToolCallContext
) -> None:
    payload = params.get("data")
    if payload is None:
        payload = params.get("message")
    if payload is None:
        return
    metadata: dict[str, JSONValue] = {"mcp_method": "notifications/message"}
    if isinstance(payload, dict):
        event_type = payload.get("type")
        if isinstance(event_type, str):
            metadata["mcp_type"] = event_type
    content = (
        dumps(payload, separators=(",", ":"))
        if isinstance(payload, (dict, list))
        else str(payload)
    )
    await _emit_mcp_stream_event(
        context,
        kind=ToolExecutionStreamKind.LOG,
        content=content,
        metadata=metadata,
    )


async def _forward_mcp_resources(
    params: JSONObject, context: ToolCallContext
) -> None:
    resources = params.get("resources")
    if not isinstance(resources, list):
        return
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        delta = resource.get("delta")
        if not isinstance(delta, dict):
            continue
        update = delta.get("set")
        if not isinstance(update, dict):
            continue
        text = update.get("text")
        if not isinstance(text, str) or not text:
            continue
        await _emit_mcp_stream_event(
            context,
            kind=ToolExecutionStreamKind.LOG,
            content=text,
            metadata={
                "mcp_method": "notifications/resources/updated",
                "mcp_resource_uri": cast(str, resource.get("uri") or ""),
            },
        )


async def _emit_mcp_stream_event(
    context: ToolCallContext,
    *,
    kind: ToolExecutionStreamKind,
    content: str | None = None,
    progress: int | float | None = None,
    metadata: dict[str, JSONValue] | None = None,
) -> None:
    if context.cancellation_checker is not None:
        await context.cancellation_checker()
    assert context.stream_event is not None
    await context.stream_event(
        ToolExecutionStreamEvent(
            kind=kind,
            content=content,
            progress=progress,
            metadata=metadata or {},
        )
    )
