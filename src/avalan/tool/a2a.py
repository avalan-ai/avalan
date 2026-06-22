from ..compat import override
from ..entities import (
    ToolCall,
    ToolCallContext,
    ToolCallOutcome,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from . import Tool, ToolSet
from .builtin_display import project_a2a_call_tool_display

from collections.abc import Mapping
from contextlib import AsyncExitStack
from importlib import import_module
from json import dumps
from typing import Any, cast
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
    "TASK_STATE_CANCELED",
    "TASK_STATE_CANCELLED",
    "TASK_STATE_FAILED",
    "TASK_STATE_REJECTED",
}


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


class A2AToolSet(ToolSet):
    """Tool set providing A2A client functionality."""

    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = "a2a",
    ) -> None:
        tools = [A2ACallTool()]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )


async def _call_a2a_agent(
    *,
    uri: str,
    name: str,
    arguments: dict[str, object],
    context: ToolCallContext,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
) -> dict[str, object]:
    request_id = str(call_params.get("request_id") or uuid4())
    a2a_pb2 = import_module("a2a.types.a2a_pb2")
    client_module = import_module("a2a.client")
    constants = import_module("a2a.utils.constants")
    httpx_module = import_module("httpx")
    client_config, owns_httpx_client = _client_config(
        client_module=client_module,
        constants=constants,
        httpx_module=httpx_module,
        client_params=client_params,
    )
    client = await client_module.create_client(
        _agent_card(
            a2a_pb2=a2a_pb2,
            constants=constants,
            uri=uri,
            name=name,
        ),
        client_config=client_config,
    )
    request = _send_message_request(
        a2a_pb2=a2a_pb2,
        request_id=request_id,
        name=name,
        arguments=arguments,
        call_params=call_params,
    )
    call_context = _client_call_context(
        client_module=client_module,
        call_params=call_params,
    )
    json_format = import_module("google.protobuf.json_format")

    state = _A2AStreamState()
    if owns_httpx_client:
        async with client:
            await _consume_a2a_stream(
                client=client,
                request=request,
                call_context=call_context,
                json_format=json_format,
                state=state,
                context=context,
            )
    else:
        await _consume_a2a_stream(
            client=client,
            request=request,
            call_context=call_context,
            json_format=json_format,
            state=state,
            context=context,
        )

    if state.error_state is not None:
        raise RuntimeError(f"A2A task ended with {state.error_state}")
    if not state.saw_terminal:
        raise RuntimeError("A2A response ended without a terminal event")
    return state.result()


def _client_config(
    *,
    client_module: Any,
    constants: Any,
    httpx_module: Any,
    client_params: Mapping[str, object],
) -> tuple[Any, bool]:
    owns_httpx_client = False
    httpx_client = client_params.get(_A2A_HTTPX_CLIENT_PARAM_KEY)
    if httpx_client is None:
        owns_httpx_client = True
        httpx_client = httpx_module.AsyncClient(
            **_client_options(client_params)
        )
    return (
        client_module.ClientConfig(
            streaming=True,
            httpx_client=httpx_client,
            supported_protocol_bindings=[constants.TransportProtocol.JSONRPC],
            accepted_output_modes=["text/plain", "text/markdown"],
        ),
        owns_httpx_client,
    )


def _agent_card(
    *,
    a2a_pb2: Any,
    constants: Any,
    uri: str,
    name: str,
) -> Any:
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
        capabilities=a2a_pb2.AgentCapabilities(streaming=True),
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
            parts=[a2a_pb2.Part(text=_message_text(name, arguments))],
        ),
        configuration=a2a_pb2.SendMessageConfiguration(
            accepted_output_modes=["text/plain", "text/markdown"],
        ),
        metadata=metadata_struct,
    )


def _message_text(name: str, arguments: Mapping[str, object]) -> str:
    for key in ("input_string", "message", "input", "prompt"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    if arguments:
        return dumps(arguments, separators=(",", ":"))
    return name


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
    headers.setdefault("A2A-Version", "1.0")
    options.setdefault("timeout", None)
    options["headers"] = headers
    return options


def _client_call_context(
    *,
    client_module: Any,
    call_params: Mapping[str, object],
) -> Any:
    kwargs: dict[str, object] = {}
    state = call_params.get("state")
    if isinstance(state, Mapping):
        kwargs["state"] = dict(state)
    service_parameters = call_params.get("service_parameters")
    if service_parameters is not None:
        kwargs["service_parameters"] = service_parameters
    timeout = call_params.get("timeout")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
        kwargs["timeout"] = timeout
    return client_module.ClientCallContext(**kwargs)


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

    def __init__(self) -> None:
        self.task_id = None
        self.context_id = None
        self.final_state = None
        self.error_state = None
        self.saw_terminal = False
        self.answer_chunks = []
        self.artifacts = {}
        self.messages = []
        self.status_updates = []

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
            self.messages.append(_message_payload(message))
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
        self.task_id = _string_member(task, "id") or self.task_id
        self.context_id = (
            _string_member(task, "contextId")
            or _string_member(task, "context_id")
            or self.context_id
        )
        status = _object_member(task, "status")
        if status is not None:
            self._record_state(_string_member(status, "state"))

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

    def _record_status(self, status: dict[str, object]) -> None:
        task_id = status.get("taskId")
        context_id = status.get("contextId")
        if isinstance(task_id, str) and task_id:
            self.task_id = task_id
        if isinstance(context_id, str) and context_id:
            self.context_id = context_id
        state = status.get("state")
        self._record_state(state if isinstance(state, str) else None)
        self.status_updates.append(status)

    def _record_state(self, state: str | None) -> None:
        if not state:
            return
        self.final_state = state
        if state in _FINAL_STATES:
            self.saw_terminal = True
        if state in _ERROR_STATES:
            self.error_state = state

    def _record_artifact_update(
        self, update: Mapping[str, object]
    ) -> tuple[dict[str, object], list[str]]:
        task_id = _string_member(update, "taskId") or _string_member(
            update, "task_id"
        )
        context_id = _string_member(update, "contextId") or _string_member(
            update, "context_id"
        )
        if task_id:
            self.task_id = task_id
        if context_id:
            self.context_id = context_id

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
    return {
        "id": (
            _string_member(message, "messageId")
            or _string_member(message, "message_id")
        ),
        "role": _string_member(message, "role"),
        "text": "".join(_part_text(part) for part in _parts(message)),
        "metadata": _object_member(message, "metadata") or {},
    }


def _status_payload(update: Mapping[str, object]) -> dict[str, object]:
    status = _object_member(update, "status") or {}
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
