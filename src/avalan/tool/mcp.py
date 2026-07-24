from ..compat import override
from ..entities import (
    MessageContentFile,
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from . import Tool, ToolSet
from .builtin_display import project_mcp_call_tool_display
from .input_files import input_file_string, iter_input_file_content

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from importlib import import_module
from json import JSONDecodeError, dumps, loads
from typing import Protocol, cast

JSONValue = dict[str, object] | list[object] | str | int | float | bool | None
JSONObject = dict[str, JSONValue]
_MCP_FILE_ARGUMENT_KEYS = {"files", "input_files", "file_descriptors"}


class _MCPDumpable(Protocol):
    """Describe the SDK model serialization used by notification callbacks."""

    def model_dump(self, **kwargs: object) -> object:
        """Return one decoded MCP value."""
        ...


class _McpToolEventSink:
    """Forward decoded MCP notifications to one tool-call stream."""

    def __init__(self, context: ToolCallContext) -> None:
        self._context = context

    async def progress(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        params: dict[str, object] = {"progress": progress}
        if total is not None:
            params["total"] = total
        if message is not None:
            params["message"] = message
        await _forward_mcp_progress(cast(JSONObject, params), self._context)

    async def logging(
        self,
        params: object,
    ) -> None:
        payload = cast(_MCPDumpable, params).model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
        )
        if not isinstance(payload, dict):
            return
        await _forward_mcp_message(
            cast(JSONObject, payload),
            self._context,
        )

    async def message(self, message: object) -> None:
        serializer = getattr(message, "model_dump", None)
        if not callable(serializer):
            return
        payload = serializer(
            by_alias=True,
            mode="json",
            exclude_none=True,
        )
        if not isinstance(payload, dict):
            return
        method = payload.get("method")
        if isinstance(method, str) and method not in {
            "notifications/message",
            "notifications/progress",
        }:
            await _forward_mcp_notification(
                method,
                cast(JSONObject, payload),
                self._context,
            )


class McpCallTool(Tool):
    """Call an MCP server tool using the MCP client.

    Args:
        uri: Base URI of the MCP server.
        name: Name of the tool to invoke.
        arguments: Arguments to send to the tool.
        forward_input_files: Whether to include files attached to this run as
            `input_files` when the remote tool arguments do not already include
            explicit file arguments.

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
        forward_input_files: bool = False,
        *,
        context: ToolCallContext,
    ) -> dict[str, object]:
        assert uri
        assert name
        assert isinstance(forward_input_files, bool)

        return await _call_streamable_http_mcp_tool(
            uri=uri,
            name=name,
            arguments=arguments or {},
            forward_input_files=forward_input_files,
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
    forward_input_files: bool,
    context: ToolCallContext,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
) -> dict[str, object]:
    request_arguments = (
        _arguments_with_context_input_files(arguments, context)
        if forward_input_files
        else arguments
    )
    sink = _McpToolEventSink(context)
    return await _call_initialized_mcp_tool(
        uri=uri,
        name=name,
        arguments=request_arguments,
        context=context,
        client_params=client_params,
        call_params=call_params,
        progress_callback=sink.progress,
        logging_callback=sink.logging,
        message_handler=sink.message,
    )


async def _call_initialized_mcp_tool(
    **kwargs: object,
) -> dict[str, object]:
    """Load the optional MCP SDK only when executing a remote call."""
    try:
        module = import_module("avalan.tool.mcp_session")
    except ModuleNotFoundError as error:
        if error.name == "mcp" or (
            error.name is not None and error.name.startswith("mcp.")
        ):
            raise RuntimeError(
                "MCP calls require the optional 'server' dependencies"
            ) from error
        raise
    call = cast(
        Callable[..., Awaitable[dict[str, object]]],
        getattr(module, "call_initialized_mcp_tool"),
    )
    return await call(**kwargs)


def _arguments_with_context_input_files(
    arguments: dict[str, object], context: ToolCallContext
) -> dict[str, object]:
    if any(key in arguments for key in _MCP_FILE_ARGUMENT_KEYS):
        return arguments

    input_files = _context_input_file_descriptors(context)
    if not input_files:
        return arguments

    request_arguments = dict(arguments)
    request_arguments["input_files"] = input_files
    return request_arguments


def _context_input_file_descriptors(
    context: ToolCallContext,
) -> list[JSONObject]:
    return [
        descriptor
        for descriptor in (
            _mcp_file_descriptor(file_content)
            for file_content in iter_input_file_content(context.input)
        )
        if descriptor is not None
    ]


def _mcp_file_descriptor(
    content: MessageContentFile,
) -> JSONObject | None:
    file = content.file
    filename = input_file_string(
        file, "filename", "fileName", "file_name", "name", "displayName"
    )
    media_type = input_file_string(file, "mime_type", "media_type", "mimeType")
    file_url = input_file_string(file, "file_url", "url", "uri")

    descriptor: JSONObject = {}
    if file_url is not None:
        descriptor["uri"] = file_url
    else:
        file_data = input_file_string(file, "file_data", "data", "base64")
        if file_data is None:
            return None
        descriptor["data"] = file_data

    if media_type is not None:
        descriptor["mimeType"] = media_type
    if filename is not None:
        descriptor["filename"] = filename
    return descriptor


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
