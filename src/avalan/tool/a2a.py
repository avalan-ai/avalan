from ..agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
)
from ..compat import override
from ..entities import (
    Input,
    Message,
    MessageContentFile,
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    normalize_tool_arguments,
)
from ..interaction.a2a import (
    A2A_INPUT_EXTENSION_DESCRIPTION,
    A2A_INPUT_EXTENSION_PARAMS,
    A2A_INPUT_EXTENSION_URI,
    A2AInputRequestMetadata,
    decode_a2a_input_request_metadata,
    encode_a2a_input_resolution_metadata,
)
from ..interaction.a2a_continuation import (
    A2AInputRequiredError,
    A2ARemoteInputContinuation,
    A2AToolContinuationCheckpoint,
    bound_a2a_prior_content,
    project_a2a_input_resolution,
)
from ..interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from ..interaction.entities import (
    AnsweredResolution,
    CancelledResolution,
    DeclinedResolution,
    ExpiredResolution,
    InputResolution,
    RequirementMode,
    SupersededResolution,
    TimedOutResolution,
    UnavailableResolution,
)
from ..interaction.error import InputContractError
from ..types import JsonValue
from . import Tool, ToolSet
from .builtin_display import project_a2a_call_tool_display
from .context import A2AToolSettings
from .input_files import (
    input_file_string,
    iter_input_file_content,
    message_file_content,
)

from asyncio import CancelledError, create_task, shield
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from importlib import import_module
from json import dumps
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

JSONValue = dict[str, object] | list[object] | str | int | float | bool | None
JSONObject = dict[str, JSONValue]

_A2A_HTTPX_CLIENT_PARAM_KEY = "httpx_client"
_FINAL_STATES = {
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_REJECTED",
}
_ERROR_STATES = {
    "TASK_STATE_AUTH_REQUIRED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_FAILED",
    "TASK_STATE_REJECTED",
}
_INPUT_REQUIRED_STATE = "TASK_STATE_INPUT_REQUIRED"


class A2ACallTool(Tool):
    """Call a remote A2A agent skill.

    Args:
        uri: URI of the A2A endpoint.
        name: Name of the remote A2A skill to invoke.
        arguments: Arguments to send to the remote agent.

    Returns:
        Content and structured data returned by the A2A invocation.
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
        return project_a2a_call_tool_display(call=call, outcome=outcome)

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

        return await _call_a2a_agent(
            uri=uri,
            name=name,
            arguments=arguments or {},
            context=context,
            client_params=self._client_params,
            call_params=self._call_params,
        )

    async def resume_input(
        self,
        checkpoint: A2AToolContinuationCheckpoint,
        resolution: InputResolution,
        *,
        context: ToolCallContext,
    ) -> dict[str, object]:
        """Resume one durable same-task A2A input continuation."""
        if type(checkpoint) is not A2AToolContinuationCheckpoint:
            raise TypeError("checkpoint must be an A2A tool continuation")
        uri, name, arguments = _a2a_target(checkpoint.arguments)
        return await _call_a2a_agent(
            uri=uri,
            name=name,
            arguments=arguments,
            continuation=checkpoint.remote,
            resolution=resolution,
            context=context,
            client_params=self._client_params,
            call_params=self._call_params,
        )

    async def cancel_input(
        self,
        checkpoint: A2AToolContinuationCheckpoint,
        *,
        operation_id: str,
        context: ToolCallContext,
    ) -> None:
        """Cancel one durable remote A2A input continuation."""
        if type(checkpoint) is not A2AToolContinuationCheckpoint:
            raise TypeError("checkpoint must be an A2A tool continuation")
        if not isinstance(operation_id, str) or not operation_id:
            raise TypeError("operation_id must be a non-empty string")
        await _cancel_a2a_continuation(
            checkpoint,
            operation_id=operation_id,
            client_params=self._client_params,
            call_params=self._call_params,
        )


class A2AToolSet(ToolSet):
    """Tool set providing A2A client functionality."""

    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = "a2a",
        settings: A2AToolSettings | None = None,
    ) -> None:
        tools = [
            A2ACallTool(
                client_params=(
                    dict(settings.client_params) if settings else None
                ),
                call_params=(dict(settings.call_params) if settings else None),
            )
        ]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )


async def _call_a2a_agent(
    *,
    uri: str,
    name: str,
    arguments: Mapping[str, object],
    context: ToolCallContext,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
    continuation: A2ARemoteInputContinuation | None = None,
    resolution: InputResolution | None = None,
) -> dict[str, object]:
    if (continuation is None) != (resolution is None):
        raise TypeError("A2A continuation and resolution must be paired")
    request_id = str(call_params.get("request_id") or uuid4())
    input_router = _A2AInputRouter(
        uri=uri,
        tool_name=name,
        arguments=arguments,
        context=context,
        ttl_seconds=(
            continuation.ttl_seconds
            if continuation is not None
            else _input_ttl_seconds(call_params)
        ),
    )
    input_extension = continuation is not None or input_router.input_capable
    negotiation = _A2AExtensionNegotiation() if input_extension else None
    a2a_pb2 = import_module("a2a.types.a2a_pb2")
    client_module = import_module("a2a.client")
    constants = import_module("a2a.utils.constants")
    httpx_module = import_module("httpx")
    client_config, owns_httpx_client = _client_config(
        client_module=client_module,
        constants=constants,
        httpx_module=httpx_module,
        client_params=client_params,
        negotiation=negotiation,
    )
    state: _A2AStreamState | None = None
    try:
        async with AsyncExitStack() as stack:
            client = await client_module.create_client(
                _agent_card(
                    a2a_pb2=a2a_pb2,
                    constants=constants,
                    uri=uri,
                    name=name,
                    input_extension=input_extension,
                ),
                client_config=client_config,
            )
            if owns_httpx_client:
                client = await stack.enter_async_context(client)
            if continuation is None:
                request = _send_message_request(
                    a2a_pb2=a2a_pb2,
                    request_id=request_id,
                    name=name,
                    arguments=arguments,
                    context=context,
                    call_params=call_params,
                )
                state = _A2AStreamState(
                    input_extension_requested=input_extension,
                    input_extension_activated=(
                        negotiation.activated
                        if negotiation is not None
                        else None
                    ),
                )
            else:
                assert resolution is not None
                assert negotiation is not None
                request = _input_resolution_request(
                    a2a_pb2=a2a_pb2,
                    resolution=resolution,
                    task_id=continuation.task_id,
                    context_id=continuation.context_id,
                    prior_message_id=continuation.prior_message_id,
                )
                state = _A2AStreamState.from_continuation(
                    continuation,
                    input_extension_activated=negotiation.activated,
                )
                state.begin_continuation()
            call_context = _client_call_context(
                client_module=client_module,
                call_params=call_params,
                input_extension=input_extension,
            )
            json_format = import_module("google.protobuf.json_format")
            try:
                await _consume_a2a_stream(
                    client=client,
                    request=request,
                    call_context=call_context,
                    json_format=json_format,
                    state=state,
                    context=context,
                )
                if continuation is not None and (
                    negotiation is None or not negotiation.activated()
                ):
                    raise RuntimeError("A2A input extension was not activated")
                while (
                    state.error_state is None
                    and state.input_request is not None
                ):
                    resolution = await input_router.resolve(
                        state.input_request,
                        reason=state.input_request_text,
                        remote_task_id=state.task_id,
                        remote_context_id=state.context_id,
                        prior_message_id=state.input_message_id,
                        prior_content=bound_a2a_prior_content(
                            tuple(state.answer_chunks)
                        ),
                        input_cycle_count=state.input_cycle_count,
                    )
                    follow_up = _input_resolution_request(
                        a2a_pb2=a2a_pb2,
                        resolution=resolution,
                        task_id=state.task_id,
                        context_id=state.context_id,
                        prior_message_id=state.input_message_id,
                    )
                    state.begin_continuation()
                    await _consume_a2a_stream(
                        client=client,
                        request=follow_up,
                        call_context=call_context,
                        json_format=json_format,
                        state=state,
                        context=context,
                    )
                    if negotiation is None or not negotiation.activated():
                        raise RuntimeError(
                            "A2A input extension was not activated"
                        )
            except CancelledError as cancellation:
                if state.task_id is not None and state.context_id is not None:
                    cleanup = create_task(
                        _cancel_a2a_task(
                            client=client,
                            task_id=state.task_id,
                            context_id=state.context_id,
                            call_context=call_context,
                            a2a_pb2=a2a_pb2,
                            json_format=json_format,
                        ),
                        name=f"a2a-cancel-{state.task_id}",
                    )
                    try:
                        await shield(cleanup)
                    except BaseException as cleanup_failure:
                        cancellation.add_note(
                            "A2A remote cancellation failed: "
                            f"{cleanup_failure.__class__.__name__}: "
                            f"{cleanup_failure}"
                        )
                raise
    finally:
        if negotiation is not None:
            negotiation.detach()

    assert state is not None
    if state.error_state is not None:
        raise RuntimeError(f"A2A task ended with {state.error_state}")
    if not state.saw_terminal:
        raise RuntimeError("A2A response ended without a terminal event")
    return state.result()


async def _cancel_a2a_task(
    *,
    client: object,
    task_id: str,
    context_id: str,
    call_context: object,
    a2a_pb2: Any,
    json_format: Any,
    operation_id: str | None = None,
) -> None:
    """Best-effort cancel one exact remote task during local cancellation."""
    metadata = import_module("google.protobuf.struct_pb2").Struct()
    metadata.update(
        {
            "contextId": context_id,
            **(
                {"avalanOperationId": operation_id}
                if operation_id is not None
                else {}
            ),
        }
    )
    cancel_task = getattr(client, "cancel_task", None)
    if not callable(cancel_task):
        raise RuntimeError("A2A client cannot cancel the remote task")
    cancelled = await cancel_task(
        a2a_pb2.CancelTaskRequest(
            id=task_id,
            metadata=metadata,
        ),
        context=call_context,
    )
    payload = json_format.MessageToDict(cancelled)
    if not isinstance(payload, Mapping):
        raise RuntimeError("A2A cancellation returned invalid state")
    returned_task_id = _string_member(payload, "id")
    returned_context_id = _string_member(
        payload,
        "contextId",
    ) or _string_member(payload, "context_id")
    if returned_task_id != task_id or returned_context_id != context_id:
        raise RuntimeError("A2A cancellation correlation mismatch")
    status = _object_member(payload, "status")
    state = _string_member(status, "state") if status is not None else None
    if state not in _FINAL_STATES:
        raise RuntimeError("A2A cancellation did not reach a terminal state")


async def _cancel_a2a_continuation(
    checkpoint: A2AToolContinuationCheckpoint,
    *,
    operation_id: str,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
) -> None:
    """Cancel one checkpointed task with a freshly owned client."""
    uri, name, _ = _a2a_target(checkpoint.arguments)
    continuation = checkpoint.remote
    a2a_pb2 = import_module("a2a.types.a2a_pb2")
    client_module = import_module("a2a.client")
    constants = import_module("a2a.utils.constants")
    client_config, owns_client = _client_config(
        client_module=client_module,
        constants=constants,
        httpx_module=import_module("httpx"),
        client_params=client_params,
    )
    async with AsyncExitStack() as stack:
        client = await client_module.create_client(
            _agent_card(
                a2a_pb2=a2a_pb2,
                constants=constants,
                uri=uri,
                name=name,
            ),
            client_config=client_config,
        )
        if owns_client:
            client = await stack.enter_async_context(client)
        await _cancel_a2a_task(
            client=client,
            task_id=continuation.task_id,
            context_id=continuation.context_id,
            call_context=_client_call_context(
                client_module=client_module,
                call_params=call_params,
            ),
            a2a_pb2=a2a_pb2,
            json_format=import_module("google.protobuf.json_format"),
            operation_id=operation_id,
        )


def _client_config(
    *,
    client_module: Any,
    constants: Any,
    httpx_module: Any,
    client_params: Mapping[str, object],
    negotiation: "_A2AExtensionNegotiation | None" = None,
) -> tuple[Any, bool]:
    owns_httpx_client = False
    httpx_client = client_params.get(_A2A_HTTPX_CLIENT_PARAM_KEY)
    if httpx_client is None:
        owns_httpx_client = True
        httpx_client = httpx_module.AsyncClient(
            **_client_options(client_params)
        )
    if negotiation is not None:
        negotiation.attach(httpx_client)
    return (
        client_module.ClientConfig(
            streaming=True,
            httpx_client=httpx_client,
            supported_protocol_bindings=[constants.TransportProtocol.JSONRPC],
            accepted_output_modes=["text/plain", "text/markdown"],
        ),
        owns_httpx_client,
    )


class _A2AExtensionNegotiation:
    """Track the activated input extension from HTTP response headers."""

    def __init__(self) -> None:
        self._activated = False
        self._hooks: list[object] | None = None

    def activated(self) -> bool:
        """Return whether the peer echoed the requested extension."""
        return self._activated

    def attach(self, client: object) -> None:
        """Observe responses from the exact HTTP client used by the SDK."""
        event_hooks = getattr(client, "event_hooks", None)
        if not isinstance(event_hooks, dict):
            raise RuntimeError("A2A client cannot verify extension activation")
        hooks = event_hooks.setdefault("response", [])
        if not isinstance(hooks, list):
            raise RuntimeError("A2A client response hooks are unavailable")
        hooks.append(self._observe)
        self._hooks = hooks

    def detach(self) -> None:
        """Stop observing a caller-owned HTTP client."""
        hooks = self._hooks
        self._hooks = None
        if hooks is not None and self._observe in hooks:
            hooks.remove(self._observe)

    async def _observe(self, response: object) -> None:
        headers = getattr(response, "headers", None)
        request = getattr(response, "request", None)
        request_headers = getattr(request, "headers", None)
        if not isinstance(headers, Mapping) or not isinstance(
            request_headers, Mapping
        ):
            return
        requested = _extension_header_values(
            request_headers.get("A2A-Extensions")
        )
        activated = _extension_header_values(headers.get("A2A-Extensions"))
        if A2A_INPUT_EXTENSION_URI in requested:
            self._activated = A2A_INPUT_EXTENSION_URI in activated


def _extension_header_values(value: object) -> frozenset[str]:
    if not isinstance(value, str):
        return frozenset()
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def _agent_card(
    *,
    a2a_pb2: Any,
    constants: Any,
    uri: str,
    name: str,
    input_extension: bool = False,
) -> Any:
    extensions = []
    if input_extension:
        extensions.append(
            a2a_pb2.AgentExtension(
                uri=A2A_INPUT_EXTENSION_URI,
                description=A2A_INPUT_EXTENSION_DESCRIPTION,
                required=False,
                params=A2A_INPUT_EXTENSION_PARAMS,
            )
        )
    return a2a_pb2.AgentCard(
        name=name,
        description=f"Call the {name} A2A agent.",
        version="1.0.0",
        supported_interfaces=[
            a2a_pb2.AgentInterface(
                url=uri,
                protocol_binding=constants.TransportProtocol.JSONRPC,
                protocol_version=constants.PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=a2a_pb2.AgentCapabilities(
            streaming=True,
            extensions=extensions,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            a2a_pb2.AgentSkill(
                id=name,
                name=name,
                description=f"Call the {name} A2A agent.",
                tags=["a2a", "agent"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )


def _send_message_request(
    *,
    a2a_pb2: Any,
    request_id: str,
    name: str,
    arguments: Mapping[str, object],
    context: ToolCallContext,
    call_params: Mapping[str, object],
) -> Any:
    metadata = _request_metadata(name, arguments, call_params)
    struct_pb2 = import_module("google.protobuf.struct_pb2")
    metadata_struct = struct_pb2.Struct()
    metadata_struct.update(metadata)
    message_id = str(call_params.get("message_id") or request_id)
    return a2a_pb2.SendMessageRequest(
        message=a2a_pb2.Message(
            message_id=message_id,
            role=a2a_pb2.Role.ROLE_USER,
            parts=_message_parts(
                a2a_pb2=a2a_pb2,
                name=name,
                arguments=arguments,
                context=context,
            ),
        ),
        configuration=a2a_pb2.SendMessageConfiguration(
            accepted_output_modes=["text/plain", "text/markdown"],
        ),
        metadata=metadata_struct,
    )


def _input_resolution_request(
    *,
    a2a_pb2: Any,
    resolution: InputResolution,
    task_id: str | None,
    context_id: str | None,
    prior_message_id: str | None,
) -> Any:
    if not task_id or not context_id:
        raise RuntimeError("A2A input request has no task correlation")
    metadata = import_module("google.protobuf.struct_pb2").Struct()
    metadata.update(
        {
            A2A_INPUT_EXTENSION_URI: encode_a2a_input_resolution_metadata(
                resolution
            )
        }
    )
    if isinstance(resolution, AnsweredResolution):
        text = "Input supplied."
    elif isinstance(resolution, DeclinedResolution):
        text = "Input declined."
    else:
        text = "Input cancelled."
    message_id = str(uuid4())
    if message_id in {str(resolution.request_id), prior_message_id}:
        raise RuntimeError("A2A follow-up message identity is not fresh")
    return a2a_pb2.SendMessageRequest(
        message=a2a_pb2.Message(
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
            role=a2a_pb2.Role.ROLE_USER,
            parts=[a2a_pb2.Part(text=text)],
            metadata=metadata,
            extensions=[A2A_INPUT_EXTENSION_URI],
        ),
        configuration=a2a_pb2.SendMessageConfiguration(
            accepted_output_modes=["text/plain", "text/markdown"],
        ),
    )


def _message_text(name: str, arguments: Mapping[str, object]) -> str:
    for key in ("input_string", "message", "input", "prompt"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    if arguments:
        return dumps(arguments, separators=(",", ":"))
    return name


def _message_parts(
    *,
    a2a_pb2: Any,
    name: str,
    arguments: Mapping[str, object],
    context: ToolCallContext,
) -> list[Any]:
    parts = [a2a_pb2.Part(text=_message_text(name, arguments))]
    parts.extend(
        part
        for part in (
            _file_part(a2a_pb2, file_content)
            for file_content in _iter_input_file_content(context.input)
        )
        if part is not None
    )
    return parts


def _file_part(a2a_pb2: Any, content: MessageContentFile) -> Any | None:
    file = content.file
    filename = _file_string(file, "filename", "file_name", "name")
    media_type = _file_string(file, "mime_type", "media_type", "mimeType")
    file_url = _file_string(file, "file_url", "url", "uri")
    if file_url is not None:
        return a2a_pb2.Part(
            url=file_url,
            filename=filename or "",
            media_type=media_type or "",
            metadata={},
        )

    file_data = _file_string(file, "file_data", "data", "base64")
    raw = _decode_file_data(file_data)
    if raw is None:
        return None
    return a2a_pb2.Part(
        raw=raw,
        filename=filename or "",
        media_type=media_type or "",
        metadata={},
    )


def _file_string(file: Mapping[str, object], *keys: str) -> str | None:
    return input_file_string(file, *keys)


def _decode_file_data(value: str | None) -> bytes | None:
    if value is None:
        return None
    payload = value.strip()
    if not payload:
        return None
    if payload.startswith("data:"):
        _prefix, separator, payload = payload.partition(",")
        if not separator:
            return None
        payload = payload.strip()
    try:
        return b64decode(payload, validate=True)
    except (BinasciiError, ValueError):
        return None


def _iter_input_file_content(
    input_value: Input | None,
) -> list[MessageContentFile]:
    return list(iter_input_file_content(input_value))


def _message_file_content(message: Message) -> list[MessageContentFile]:
    return list(message_file_content(message))


def _request_metadata(
    name: str,
    arguments: Mapping[str, object],
    call_params: Mapping[str, object],
) -> dict[str, object]:
    metadata: dict[str, object] = {"skill": name}
    raw_metadata = arguments.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update(cast(Mapping[str, object], raw_metadata))
    raw_call_metadata = call_params.get("metadata")
    if isinstance(raw_call_metadata, Mapping):
        metadata.update(cast(Mapping[str, object], raw_call_metadata))
    metadata.setdefault("arguments", dict(arguments))
    return metadata


def _client_options(client_params: Mapping[str, object]) -> dict[str, object]:
    options = {
        key: value
        for key, value in client_params.items()
        if key != _A2A_HTTPX_CLIENT_PARAM_KEY
    }
    raw_headers = options.pop("headers", None)
    headers = (
        dict(cast(Mapping[str, str], raw_headers))
        if isinstance(raw_headers, Mapping)
        else {}
    )
    headers.setdefault("Accept", "application/json, text/event-stream")
    headers.setdefault("Content-Type", "application/json")
    headers["A2A-Version"] = "1.0"
    options.setdefault("timeout", None)
    options["headers"] = headers
    return options


def _client_call_context(
    *,
    client_module: Any,
    call_params: Mapping[str, object],
    input_extension: bool = False,
) -> Any:
    kwargs: dict[str, object] = {}
    state = call_params.get("state")
    if isinstance(state, Mapping):
        kwargs["state"] = dict(state)
    service_parameters = call_params.get("service_parameters")
    if input_extension:
        if service_parameters is not None and not isinstance(
            service_parameters, Mapping
        ):
            raise ValueError("A2A service parameters must be a mapping")
        parameters = (
            dict(cast(Mapping[str, object], service_parameters))
            if isinstance(service_parameters, Mapping)
            else {}
        )
        advertised = parameters.get("A2A-Extensions")
        extensions = {
            item.strip()
            for item in str(advertised or "").split(",")
            if item.strip()
        }
        extensions.add(A2A_INPUT_EXTENSION_URI)
        parameters["A2A-Extensions"] = ",".join(sorted(extensions))
        kwargs["service_parameters"] = parameters
    elif service_parameters is not None:
        kwargs["service_parameters"] = service_parameters
    timeout = call_params.get("timeout")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
        kwargs["timeout"] = timeout
    return client_module.ClientCallContext(**kwargs)


def _input_ttl_seconds(call_params: Mapping[str, object]) -> int:
    timeout = call_params.get("timeout")
    if (
        isinstance(timeout, int | float)
        and not isinstance(timeout, bool)
        and timeout > 0
    ):
        return min(604_800, max(60, int(timeout)))
    return 300


class _A2AInputRouter:
    """Route one negotiated downstream request through the canonical broker."""

    def __init__(
        self,
        *,
        uri: str,
        tool_name: str,
        arguments: Mapping[str, object],
        context: ToolCallContext,
        ttl_seconds: int,
    ) -> None:
        self._uri = uri
        self._tool_name = tool_name
        self._arguments = normalize_tool_arguments(arguments)
        self._context = context
        self._origin = context.execution_origin
        self._broker = context.interaction_broker
        self._ttl_seconds = ttl_seconds

    @property
    def input_capable(self) -> bool:
        """Return whether this exact call has a live authorized route."""
        execution = self._context.execution
        origin = self._origin
        runtime = (
            execution.interaction_runtime if execution is not None else None
        )
        return (
            execution is not None
            and origin is not None
            and execution.origin == origin
            and isinstance(
                runtime,
                AttachedInteractionRuntime | DurableInteractionRuntime,
            )
            and runtime.actor.principal == origin.principal
            and (
                isinstance(runtime, AttachedInteractionRuntime)
                and self._broker is not None
                or isinstance(runtime, DurableInteractionRuntime)
                and self._context.durable_a2a_input
            )
            and (
                self._context.agent_id is None
                or str(self._context.agent_id) == str(origin.agent_id)
            )
        )

    async def resolve(
        self,
        request: A2AInputRequestMetadata,
        *,
        reason: str,
        remote_task_id: str | None,
        remote_context_id: str | None,
        prior_message_id: str | None,
        prior_content: tuple[str, ...],
        input_cycle_count: int,
    ) -> InputResolution:
        """Return one terminal local resolution bound to the remote request."""
        execution = self._context.execution
        origin = self._origin
        runtime = (
            execution.interaction_runtime if execution is not None else None
        )
        if (
            not self.input_capable
            or execution is None
            or origin is None
            or execution.origin != origin
        ):
            raise RuntimeError("A2A input origin is unavailable")
        normalized_reason = " ".join(reason.split())[:500]
        if isinstance(runtime, DurableInteractionRuntime):
            if (
                remote_task_id is None
                or remote_context_id is None
                or prior_message_id is None
            ):
                raise RuntimeError("A2A durable input route is unavailable")
            raise A2AInputRequiredError(
                A2ARemoteInputContinuation(
                    request=request,
                    request_text=normalized_reason,
                    task_id=remote_task_id,
                    context_id=remote_context_id,
                    prior_message_id=prior_message_id,
                    prior_content=prior_content,
                    ttl_seconds=self._ttl_seconds,
                    input_cycle_count=input_cycle_count,
                )
            )
        broker = self._broker
        if (
            not isinstance(runtime, AttachedInteractionRuntime)
            or broker is None
        ):
            raise RuntimeError("A2A attached input route is unavailable")
        try:
            result = await broker.request(
                InteractionBrokerRequest(
                    actor=runtime.actor,
                    origin=origin,
                    mode=(
                        RequirementMode.REQUIRED
                        if request.required
                        else RequirementMode.ADVISORY
                    ),
                    reason=normalized_reason,
                    questions=request.questions,
                    context_label=_a2a_context_label(
                        self._uri,
                        self._tool_name,
                        remote_task_id,
                        remote_context_id,
                    ),
                    handler=runtime.handler,
                    continuation_ttl_seconds=self._ttl_seconds,
                )
            )
        except CancelledError:
            raise
        except InputContractError as error:
            raise RuntimeError(error.safe_message) from None
        except (RuntimeError, TypeError, ValueError):
            raise RuntimeError("A2A input could not be completed") from None
        try:
            return _remote_input_resolution(result, request)
        except InputContractError as error:
            raise RuntimeError(error.safe_message) from None


def _remote_input_resolution(
    result: InteractionRequestResult,
    request: A2AInputRequestMetadata,
) -> InputResolution:
    if result.delivery is None:
        raise RuntimeError("A2A input is unavailable")
    resolution = result.delivery.record.request.resolution
    if resolution is None:
        raise RuntimeError("A2A input did not reach a terminal state")
    if isinstance(resolution, ExpiredResolution):
        raise RuntimeError("A2A input request expired")
    if isinstance(resolution, SupersededResolution):
        raise RuntimeError("A2A input request was superseded")
    if isinstance(
        resolution,
        AnsweredResolution | DeclinedResolution | CancelledResolution,
    ):
        projected = project_a2a_input_resolution(resolution, request)
        assert projected is not None
        return projected
    if isinstance(resolution, TimedOutResolution):
        raise RuntimeError("A2A input request timed out")
    if isinstance(resolution, UnavailableResolution):
        raise RuntimeError("A2A input is unavailable")
    raise RuntimeError("A2A input did not reach a terminal state")


def _a2a_context_label(
    uri: str,
    tool: str,
    remote_task: object | None,
    remote_context: object | None,
) -> str:
    parts = [
        f"A2A {urlsplit(uri).hostname or 'local'}",
        tool,
        *(
            f"{name} {value}"
            for name, value in (
                ("task", remote_task),
                ("context", remote_context),
            )
            if value is not None
        ),
    ]
    return " · ".join(" ".join(part.split()) for part in parts)[:80]


def _a2a_target(
    arguments: Mapping[str, JsonValue],
) -> tuple[str, str, Mapping[str, object]]:
    return (
        cast(str, arguments["uri"]),
        cast(str, arguments["name"]),
        cast(Mapping[str, object], arguments.get("arguments") or {}),
    )


async def _consume_a2a_stream(
    *,
    client: Any,
    request: Any,
    call_context: Any,
    json_format: Any,
    state: "_A2AStreamState",
    context: ToolCallContext,
) -> None:
    async for stream_response in client.send_message(
        request,
        context=call_context,
    ):
        payload = _stream_response_payload(
            stream_response,
            json_format=json_format,
        )
        await state.process(payload, context)
        if state.input_request is not None or state.error_state is not None:
            break


def _stream_response_payload(
    stream_response: Any, *, json_format: Any | None = None
) -> JSONObject:
    if json_format is None:
        json_format = import_module("google.protobuf.json_format")
    payload = json_format.MessageToDict(stream_response)
    assert isinstance(payload, dict)
    return cast(JSONObject, payload)


class _A2AStreamState:
    task_id: str | None
    context_id: str | None
    final_state: str | None
    error_state: str | None
    saw_terminal: bool
    answer_chunks: list[str]
    artifacts: dict[str, dict[str, object]]
    messages: list[dict[str, object]]
    status_updates: list[dict[str, object]]
    input_request: A2AInputRequestMetadata | None
    input_request_text: str

    def __init__(
        self,
        *,
        input_extension_requested: bool = False,
        input_extension_activated: Callable[[], bool] | None = None,
    ) -> None:
        self.task_id = None
        self.context_id = None
        self.final_state = None
        self.error_state = None
        self.saw_terminal = False
        self.answer_chunks = []
        self.artifacts = {}
        self.messages = []
        self.status_updates = []
        self.input_request = None
        self.input_request_text = ""
        self._input_extension_requested = input_extension_requested
        self._input_extension_activated = (
            input_extension_activated
            if input_extension_activated is not None
            else lambda: input_extension_requested
        )
        self._continued = False
        self._continuation_working = False
        self._input_message_id: str | None = None
        self._prior_input_request_id: str | None = None
        self._input_cycle_count = 0

    @classmethod
    def from_continuation(
        cls,
        continuation: A2ARemoteInputContinuation,
        *,
        input_extension_activated: Callable[[], bool],
    ) -> "_A2AStreamState":
        """Restore the portable stream prefix before a same-task follow-up."""
        if type(continuation) is not A2ARemoteInputContinuation:
            raise TypeError("continuation must be a remote A2A continuation")
        state = cls(
            input_extension_requested=True,
            input_extension_activated=input_extension_activated,
        )
        state.task_id = continuation.task_id
        state.context_id = continuation.context_id
        state.answer_chunks = list(continuation.prior_content)
        state.input_request = continuation.request
        state.input_request_text = continuation.request_text
        state._input_message_id = continuation.prior_message_id
        state._input_cycle_count = continuation.input_cycle_count
        return state

    @property
    def input_message_id(self) -> str | None:
        """Return the correlated remote status-message identity."""
        return self._input_message_id

    @property
    def input_cycle_count(self) -> int:
        """Return the bounded number of remote input requests observed."""
        return self._input_cycle_count

    def begin_continuation(self) -> None:
        """Prepare to consume exactly one same-task continuation."""
        if (
            self.input_request is None
            or not self.task_id
            or not self.context_id
        ):
            raise RuntimeError("A2A input continuation is not available")
        self._prior_input_request_id = str(self.input_request.request_id)
        self.input_request = None
        self.input_request_text = ""
        self.final_state = None
        self.error_state = None
        self.saw_terminal = False
        self._continued = True
        self._continuation_working = False

    async def process(
        self, result: JSONObject, context: ToolCallContext
    ) -> None:
        task = _object_member(result, "task")
        if task is not None:
            self._record_task(task)
            await self._record_task_snapshot(task, context)
            return

        message = _object_member(result, "message")
        if message is not None:
            await self._record_message(message, context)
            return

        status_update = _object_member(result, "statusUpdate")
        if status_update is None:
            status_update = _object_member(result, "status_update")
        if status_update is not None:
            status = _status_payload(status_update)
            self._record_status(status)
            await _emit_status_update(status, context)
            return

        artifact_update = _object_member(result, "artifactUpdate")
        if artifact_update is None:
            artifact_update = _object_member(result, "artifact_update")
        if artifact_update is not None:
            artifact, chunks = self._record_artifact_update(artifact_update)
            if _is_answer_artifact(artifact):
                self.answer_chunks.extend(chunks)
            await _emit_artifact_update(artifact, chunks, context)

    def result(self) -> dict[str, object]:
        answer_text = "".join(self.answer_chunks)
        return {
            "content": (
                [{"type": "text", "text": answer_text}] if answer_text else []
            ),
            "structuredContent": {
                "taskId": self.task_id,
                "contextId": self.context_id,
                "state": self.final_state,
                "artifacts": list(self.artifacts.values()),
                "messages": self.messages,
                "statusUpdates": self.status_updates,
            },
        }

    def _record_task(self, task: Mapping[str, object]) -> None:
        self._bind_correlation(
            _string_member(task, "id"),
            _string_member(task, "contextId")
            or _string_member(task, "context_id"),
        )
        status = _object_member(task, "status")
        if status is not None:
            state = _string_member(status, "state")
            self._record_state(state)
            if state == _INPUT_REQUIRED_STATE:
                message = _object_member(status, "message")
                self._record_input_required(
                    _message_payload(message) if message is not None else None
                )

    async def _record_task_snapshot(
        self, task: Mapping[str, object], context: ToolCallContext
    ) -> None:
        for artifact_data in _mapping_items(task, "artifacts"):
            artifact, chunks = self._record_artifact_update(
                {
                    "taskId": self.task_id,
                    "contextId": self.context_id,
                    "artifact": artifact_data,
                }
            )
            if _is_answer_artifact(artifact):
                self.answer_chunks.extend(chunks)
            await _emit_artifact_update(artifact, chunks, context)
        for message in _mapping_items(task, "history"):
            self.messages.append(_message_payload(message))

    async def _record_message(
        self, message: Mapping[str, object], context: ToolCallContext
    ) -> None:
        if self._continued:
            raise RuntimeError(
                "A2A continuation ended without a terminal task status"
            )
        payload = _message_payload(message)
        self.messages.append(payload)
        self._record_state("TASK_STATE_COMPLETED")
        text = payload.get("text")
        if isinstance(text, str) and text:
            self.answer_chunks.append(text)
            await _emit_message_response(text, context)

    def _record_status(self, status: dict[str, object]) -> None:
        task_id = status.get("taskId")
        context_id = status.get("contextId")
        if self._continued and (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(context_id, str)
            or not context_id
        ):
            raise RuntimeError("A2A continuation correlation is missing")
        self._bind_correlation(
            task_id if isinstance(task_id, str) else None,
            context_id if isinstance(context_id, str) else None,
        )
        state = status.get("state")
        self._record_state(state if isinstance(state, str) else None)
        self.status_updates.append(
            {key: value for key, value in status.items() if key != "message"}
        )
        if state == _INPUT_REQUIRED_STATE:
            message = status.get("message")
            self._record_input_required(
                message if isinstance(message, Mapping) else None
            )

    def _record_state(self, state: str | None) -> None:
        if not state:
            return
        if self._continued and state == "TASK_STATE_WORKING":
            self._continuation_working = True
        if (
            self._continued
            and state == "TASK_STATE_COMPLETED"
            and not self._continuation_working
        ):
            raise RuntimeError(
                "A2A continuation completed before returning to working"
            )
        self.final_state = state
        if state in _FINAL_STATES:
            self.saw_terminal = True
        if state in _ERROR_STATES:
            self.error_state = state

    def _record_input_required(
        self,
        message: Mapping[str, object] | None,
    ) -> None:
        new_request = self.input_request is None
        if new_request and self._continued and not self._continuation_working:
            raise RuntimeError(
                "A2A task requested input before returning to working"
            )
        if not self._input_extension_requested:
            raise RuntimeError(
                "A2A task requested unnegotiated structured input"
            )
        if not self._input_extension_activated():
            raise RuntimeError("A2A input extension was not activated")
        if not self.task_id or not self.context_id or message is None:
            raise RuntimeError("A2A input request has no correlated message")
        if new_request:
            self._input_cycle_count += 1
            if self._input_cycle_count > MAXIMUM_EQUIVALENT_INPUT_REQUESTS:
                raise RuntimeError("A2A input request loop limit reached")
        payload = message
        message_id = payload.get("id")
        if (
            not isinstance(message_id, str)
            or not message_id
            or payload.get("role") != "ROLE_AGENT"
            or payload.get("taskId") != self.task_id
            or payload.get("contextId") != self.context_id
        ):
            raise RuntimeError("A2A input request correlation mismatch")
        extensions = payload.get("extensions")
        if (
            not isinstance(extensions, list)
            or A2A_INPUT_EXTENSION_URI not in extensions
        ):
            raise RuntimeError("A2A input extension was not negotiated")
        parts = payload.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError("A2A input request has no readable fallback")
        text_parts = [
            cast(str, part["text"])
            for part in parts
            if isinstance(part, Mapping)
            and isinstance(part.get("text"), str)
            and cast(str, part["text"]).strip()
        ]
        if not text_parts:
            raise RuntimeError("A2A input request has no readable fallback")
        metadata = payload.get("metadata")
        extension = (
            metadata.get(A2A_INPUT_EXTENSION_URI)
            if isinstance(metadata, Mapping)
            else None
        )
        try:
            request = decode_a2a_input_request_metadata(extension)
        except InputContractError as error:
            raise RuntimeError(error.safe_message) from None
        text = "\n".join(text_parts)
        if message_id == str(request.request_id):
            raise RuntimeError(
                "A2A canonical request identity leaked into messageId"
            )
        if (
            new_request
            and self._continued
            and (
                str(request.request_id) == self._prior_input_request_id
                or message_id == self._input_message_id
            )
        ):
            raise RuntimeError("A2A continuation reused input identity")
        if self.input_request is not None and (
            request != self.input_request
            or text != self.input_request_text
            or message_id != self._input_message_id
        ):
            raise RuntimeError("A2A task changed its pending input request")
        self.input_request = request
        self.input_request_text = text
        self._input_message_id = message_id

    def _bind_correlation(
        self,
        task_id: str | None,
        context_id: str | None,
    ) -> None:
        for current, candidate, label in (
            (self.task_id, task_id, "task"),
            (self.context_id, context_id, "context"),
        ):
            if current and candidate and current != candidate:
                raise RuntimeError(f"A2A {label} correlation mismatch")
        self.task_id = task_id or self.task_id
        self.context_id = context_id or self.context_id

    def _record_artifact_update(
        self, update: Mapping[str, object]
    ) -> tuple[dict[str, object], list[str]]:
        task_id = _string_member(update, "taskId") or _string_member(
            update, "task_id"
        )
        context_id = _string_member(update, "contextId") or _string_member(
            update, "context_id"
        )
        self._bind_correlation(task_id, context_id)

        artifact_data = _object_member(update, "artifact") or {}
        artifact_id = (
            _string_member(artifact_data, "artifactId")
            or _string_member(artifact_data, "artifact_id")
            or _string_member(artifact_data, "id")
            or "artifact"
        )
        metadata = _object_member(artifact_data, "metadata") or {}
        name = _string_member(artifact_data, "name")
        append = bool(update.get("append"))
        last_chunk = bool(update.get("lastChunk") or update.get("last_chunk"))
        chunks = _artifact_text_parts(artifact_data)
        artifact = self.artifacts.setdefault(
            artifact_id,
            {
                "id": artifact_id,
                "name": name,
                "metadata": dict(metadata),
                "text": "",
            },
        )
        if name:
            artifact["name"] = name
        if metadata:
            artifact["metadata"] = dict(metadata)
        if not append:
            artifact["text"] = ""
        artifact["text"] = f"{artifact.get('text', '')}{''.join(chunks)}"
        if last_chunk:
            artifact["completed"] = True
        return artifact, chunks


def _object_member(
    payload: Mapping[str, object], key: str
) -> dict[str, object] | None:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else None


def _mutable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            _mutable_mapping(cast(Mapping[str, object], item))
            if isinstance(item, Mapping)
            else (
                [
                    (
                        _mutable_mapping(cast(Mapping[str, object], nested))
                        if isinstance(nested, Mapping)
                        else nested
                    )
                    for nested in item
                ]
                if isinstance(item, tuple)
                else item
            )
        )
        for key, item in value.items()
    }


def _mapping_items(
    payload: Mapping[str, object], key: str
) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_member(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _message_payload(message: Mapping[str, object]) -> dict[str, object]:
    parts = _parts(message)
    extensions = message.get("extensions")
    return {
        "id": (
            _string_member(message, "messageId")
            or _string_member(message, "message_id")
        ),
        "taskId": (
            _string_member(message, "taskId")
            or _string_member(message, "task_id")
        ),
        "contextId": (
            _string_member(message, "contextId")
            or _string_member(message, "context_id")
        ),
        "role": _string_member(message, "role"),
        "text": "".join(_part_text(part) for part in parts),
        "parts": parts,
        "metadata": _object_member(message, "metadata") or {},
        "extensions": (
            [item for item in extensions if isinstance(item, str)]
            if isinstance(extensions, list)
            else []
        ),
    }


def _status_payload(update: Mapping[str, object]) -> dict[str, object]:
    status = _object_member(update, "status") or {}
    message = _object_member(status, "message")
    return {
        "taskId": (
            _string_member(update, "taskId")
            or _string_member(update, "task_id")
        ),
        "contextId": (
            _string_member(update, "contextId")
            or _string_member(update, "context_id")
        ),
        "state": _string_member(status, "state"),
        "message": _message_payload(message) if message is not None else None,
        "final": bool(update.get("final")),
        "metadata": _object_member(update, "metadata") or {},
    }


def _artifact_text_parts(artifact: Mapping[str, object]) -> list[str]:
    return [
        text
        for text in (_part_text(part) for part in _parts(artifact))
        if text
    ]


def _parts(payload: Mapping[str, object]) -> list[object]:
    parts = payload.get("parts")
    return list(parts) if isinstance(parts, list) else []


def _part_text(part: object) -> str:
    if not isinstance(part, Mapping):
        return ""
    text = part.get("text")
    if isinstance(text, str):
        return text
    data = part.get("data")
    if isinstance(data, str):
        return data
    if isinstance(data, (dict, list)):
        return dumps(data, separators=(",", ":"))
    return ""


def _is_answer_artifact(artifact: Mapping[str, object]) -> bool:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        return artifact.get("id") == "answer"
    return (
        metadata.get("kind") == "answer"
        or metadata.get("channel") == "output"
        or artifact.get("id") == "answer"
    )


async def _emit_status_update(
    status: Mapping[str, object], context: ToolCallContext
) -> None:
    if context.stream_event is None:
        return
    await _emit_a2a_stream_event(
        context,
        kind=ToolExecutionStreamKind.PROGRESS,
        content=dumps(status, separators=(",", ":")),
        progress=1 if status.get("state") == "TASK_STATE_COMPLETED" else None,
        metadata={
            "a2a_type": "status",
            **_event_metadata(status.get("metadata")),
        },
    )


async def _emit_artifact_update(
    artifact: Mapping[str, object],
    chunks: list[str],
    context: ToolCallContext,
) -> None:
    if context.stream_event is None or not chunks:
        return
    metadata = _event_metadata(artifact.get("metadata"))
    await _emit_a2a_stream_event(
        context,
        kind=_stream_kind(metadata),
        content="".join(chunks),
        metadata={
            "a2a_type": "artifact",
            "a2a_artifact_id": str(artifact.get("id") or ""),
            **metadata,
        },
    )


async def _emit_message_response(text: str, context: ToolCallContext) -> None:
    if context.stream_event is None:
        return
    await _emit_a2a_stream_event(
        context,
        kind=ToolExecutionStreamKind.STDOUT,
        content=text,
        progress=1,
        metadata={
            "a2a_type": "message",
            "kind": "answer",
            "channel": "output",
        },
    )


def _event_metadata(value: object) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): cast(JSONValue, item)
        for key, item in value.items()
        if isinstance(key, str)
    }


def _stream_kind(metadata: Mapping[str, object]) -> ToolExecutionStreamKind:
    if metadata.get("kind") == "answer" or metadata.get("channel") == "output":
        return ToolExecutionStreamKind.STDOUT
    if metadata.get("category") == "stderr":
        return ToolExecutionStreamKind.STDERR
    if metadata.get("category") == "stdout":
        return ToolExecutionStreamKind.STDOUT
    return ToolExecutionStreamKind.LOG


async def _emit_a2a_stream_event(
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
