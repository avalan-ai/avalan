from ...agent.orchestrator import Orchestrator
from ...entities import MessageRole
from ...model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
)
from ..authority import (
    REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS,
    remote_runtime_authority_key,
)
from ..container_policy import (
    RemoteContainerRequestError,
    RemoteContainerRequestPolicy,
    remote_container_policy_from_state,
    validate_remote_container_arguments,
)
from ..entities import (
    ChatCompletionRequest,
    ChatMessage,
    ContentFile,
    ContentImage,
    ContentText,
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionSettings,
    coerce_server_output_redaction_settings,
    sanitize_server_protocol_text,
    sanitize_server_protocol_value,
    server_output_redaction_settings_from_state,
)
from ..routers import orchestrate, resolve_model_id
from ..routers.streaming import (
    ProtocolReasoningIdentity,
    ProtocolReasoningRedactedText,
    ProtocolReasoningRedactionState,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from json import dumps, loads
from logging import Logger
from typing import Any, cast
from urllib.parse import urljoin

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

A2A_FILE_MODES = [
    "text/plain",
    "text/markdown",
    "text/*",
    "image/png",
    "image/jpeg",
    "image/*",
    "application/json",
    "application/pdf",
    "application/octet-stream",
]
A2A_OUTPUT_MODES = ["text/plain"]
_MISSING = object()
_A2A_CONTENT_KEYS = frozenset({"data", "raw", "text", "url"})
_A2A_FILE_CONTENT_KEYS = frozenset(
    {
        "base64",
        "bytes",
        "data",
        "file_data",
        "file_url",
        "raw",
        "uri",
        "url",
    }
)
_A2A_FILE_METADATA_KEYS = frozenset(
    {
        "displayName",
        "display_name",
        "fileName",
        "file_name",
        "filename",
        "mediaType",
        "media_type",
        "mimeType",
        "mime_type",
        "name",
    }
)
_A2A_TOOL_RESOURCE_CATEGORIES = frozenset(
    {
        "log",
        "logs",
        "progress",
        "stderr",
        "stdout",
    }
)


@dataclass(kw_only=True, slots=True)
class _A2AReasoningArtifactState:
    identity: ProtocolReasoningIdentity
    artifact_id: str
    characters: int = 0
    utf8_bytes: int = 0
    dropped_characters: int = 0
    dropped_utf8_bytes: int = 0
    opened: bool = False
    suppressed: bool = False


def install_a2a_routes(
    app: FastAPI,
    *,
    prefix: str,
    name: str,
    description: str | None,
) -> None:
    """Install A2A SDK v1 routes on ``app``."""
    try:
        _ensure_typing_override()
        a2a_pb2 = import_module("a2a.types.a2a_pb2")
        constants = import_module("a2a.utils.constants")
        route_module = import_module("a2a.server.routes.fastapi_routes")
        jsonrpc_routes_module = import_module(
            "a2a.server.routes.jsonrpc_routes"
        )
        rest_routes_module = import_module("a2a.server.routes.rest_routes")
        response_helpers_module = import_module(
            "a2a.server.request_handlers.response_helpers"
        )
        handler_module = import_module(
            "a2a.server.request_handlers.default_request_handler_v2"
        )
        task_store_module = import_module(
            "a2a.server.tasks.inmemory_task_store"
        )
        responses_module = import_module("starlette.responses")
        routing_module = import_module("starlette.routing")
    except ImportError as exc:
        raise ImportError("A2A router requires the a2a-sdk package") from exc

    card = _build_agent_card(
        a2a_pb2=a2a_pb2,
        constants=constants,
        interface_url=prefix,
        name=name,
        description=description,
    )
    request_handler = handler_module.DefaultRequestHandlerV2(
        agent_executor=AvalanA2AAgentExecutor(app),
        task_store=task_store_module.InMemoryTaskStore(),
        agent_card=card,
    )
    jsonrpc_routes = _validated_a2a_routes(
        _a2a_jsonrpc_routes(
            jsonrpc_routes_module,
            request_handler,
            prefix=prefix,
        ),
        route_class=routing_module.Route,
        jsonrpc=True,
    )
    rest_routes = _validated_a2a_routes(
        rest_routes_module.create_rest_routes(
            request_handler,
            path_prefix=prefix,
            enable_v0_3_compat=False,
        ),
        route_class=routing_module.Route,
    )
    route_module.add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=_agent_card_routes(
            agent_card=card,
            interface_url=prefix,
            agent_card_to_dict=response_helpers_module.agent_card_to_dict,
            json_response=responses_module.JSONResponse,
            route_class=routing_module.Route,
        ),
        jsonrpc_routes=jsonrpc_routes,
        rest_routes=rest_routes,
    )


def _agent_card_routes(
    *,
    agent_card: Any,
    interface_url: str,
    agent_card_to_dict: Any,
    json_response: Any,
    route_class: Any,
) -> list[Any]:
    async def _get_agent_card(request: Any) -> Any:
        card = deepcopy(agent_card)
        absolute_interface_url = _absolute_url(request, interface_url)
        for supported_interface in card.supported_interfaces:
            supported_interface.url = absolute_interface_url
        return json_response(agent_card_to_dict(card))

    return [
        route_class(
            path="/.well-known/agent-card.json",
            endpoint=_get_agent_card,
            methods=["GET"],
        )
    ]


def _absolute_url(request: Any, path: str) -> str:
    return urljoin(str(request.base_url), path.lstrip("/"))


def _a2a_jsonrpc_routes(
    jsonrpc_routes_module: Any,
    request_handler: Any,
    *,
    prefix: str,
) -> list[Any]:
    root_routes = jsonrpc_routes_module.create_jsonrpc_routes(
        request_handler,
        rpc_url=prefix,
        enable_v0_3_compat=False,
    )
    tenant_routes = jsonrpc_routes_module.create_jsonrpc_routes(
        request_handler,
        rpc_url=f"/{{tenant}}{prefix}",
        enable_v0_3_compat=False,
    )
    return [*root_routes, *tenant_routes]


def _validated_a2a_routes(
    routes: Sequence[Any], *, route_class: Any, jsonrpc: bool = False
) -> list[Any]:
    return [
        _validated_a2a_route(route, route_class=route_class, jsonrpc=jsonrpc)
        for route in routes
    ]


def _validated_a2a_route(
    route: Any, *, route_class: Any, jsonrpc: bool = False
) -> Any:
    nested_routes = getattr(route, "routes", None)
    if _is_sequence(nested_routes):
        wrapped_routes = _validated_a2a_routes(
            cast(Sequence[Any], nested_routes),
            route_class=route_class,
            jsonrpc=jsonrpc,
        )
        if isinstance(nested_routes, list):
            nested_routes[:] = wrapped_routes
        return route

    endpoint = getattr(route, "endpoint", None)
    methods = getattr(route, "methods", None)
    path = getattr(route, "path", None)
    if endpoint is None or path is None or methods is None:
        return route
    if "POST" not in methods:
        return route
    return route_class(
        path=path,
        endpoint=_validated_a2a_endpoint(endpoint, jsonrpc=jsonrpc),
        methods=list(methods),
        name=getattr(route, "name", None),
        include_in_schema=getattr(route, "include_in_schema", True),
    )


def _validated_a2a_endpoint(endpoint: Any, *, jsonrpc: bool = False) -> Any:
    async def _endpoint(request: Any) -> Any:
        try:
            payload = await _validate_a2a_json_file_parts(request)
        except HTTPException as exc:
            if jsonrpc:
                return await _a2a_jsonrpc_validation_error_response(
                    request, exc.detail
                )
            raise
        if jsonrpc:
            _inject_a2a_jsonrpc_tenant(request, payload)
        return await endpoint(request)

    return _endpoint


async def _validate_a2a_json_file_parts(request: Any) -> object | None:
    body = await request.body()
    if not body:
        return None
    try:
        payload: object = loads(body)
    except ValueError:
        return None
    try:
        _reject_a2a_remote_runtime_authority(
            payload,
            path="a2a",
            policy=_a2a_request_container_policy(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for part_payload in _a2a_json_part_payloads(payload):
        _validate_a2a_json_part_payload(part_payload)
    return payload


async def _a2a_jsonrpc_validation_error_response(
    request: Any, detail: object
) -> JSONResponse:
    body = await request.body()
    try:
        payload = loads(body)
    except ValueError:
        payload = None
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": _a2a_jsonrpc_request_id(payload),
            "error": {
                "code": -32602,
                "message": "Invalid params",
                "data": detail,
            },
        },
        status_code=200,
    )


def _a2a_jsonrpc_request_id(payload: object) -> str | int | None:
    if not isinstance(payload, Mapping):
        return None
    request_id = payload.get("id")
    if isinstance(request_id, bool):
        return None
    return request_id if isinstance(request_id, str | int) else None


def _inject_a2a_jsonrpc_tenant(request: Any, payload: object | None) -> None:
    if not isinstance(payload, dict):
        return
    path_params = getattr(request, "path_params", {})
    if not isinstance(path_params, Mapping):
        return
    tenant = path_params.get("tenant")
    if not isinstance(tenant, str) or not tenant:
        return
    params = payload.get("params")
    if not isinstance(params, dict):
        return
    params["tenant"] = tenant
    request._body = dumps(payload).encode("utf-8")


def _a2a_json_part_payloads(value: object) -> list[Mapping[object, object]]:
    payloads: list[Mapping[object, object]] = []
    if isinstance(value, Mapping):
        if _a2a_json_part_content_fields(value):
            return payloads
        parts = value.get("parts")
        if _is_sequence(parts):
            for part in cast(Sequence[object], parts):
                if isinstance(part, Mapping):
                    payloads.append(_a2a_json_part_payload(part))
        for key, item in value.items():
            if key in {"data", "metadata", "parts"}:
                continue
            payloads.extend(_a2a_json_part_payloads(item))
    elif _is_sequence(value):
        for item in cast(Sequence[object], value):
            payloads.extend(_a2a_json_part_payloads(item))
    return payloads


def _a2a_json_part_payload(
    part: Mapping[object, object],
) -> Mapping[object, object]:
    root = part.get("root")
    return root if isinstance(root, Mapping) else part


def _validate_a2a_json_part_payload(
    payload: Mapping[object, object],
) -> None:
    content_fields = _a2a_json_part_content_fields(payload)
    if len(content_fields) != 1:
        raise HTTPException(
            status_code=400,
            detail="A2A parts must contain exactly one content field",
        )
    raw_source = _a2a_raw_source(payload)
    if raw_source is not _MISSING:
        _validate_a2a_json_raw_part(raw_source)


def _a2a_json_part_content_fields(
    payload: Mapping[object, object],
) -> list[str]:
    return _a2a_part_content_fields(payload)


def _validate_a2a_json_raw_part(value: object) -> None:
    if _raw_file_data(value) is None:
        raise HTTPException(
            status_code=400,
            detail="A2A raw file parts must be base64 strings",
        )


def _decode_a2a_base64(value: str) -> bytes:
    payload = "".join(value.split())
    padding = "=" * (-len(payload) % 4)
    return b64decode(
        payload + padding,
        altchars=b"-_",
        validate=True,
    )


def _reject_a2a_remote_runtime_authority(
    value: object,
    *,
    path: str,
    policy: RemoteContainerRequestPolicy | None = None,
    part_payload: bool = False,
) -> None:
    if part_payload:
        _reject_a2a_part_remote_runtime_authority(
            value,
            path=path,
            policy=policy,
        )
        return

    fields = _a2a_authority_fields(value)
    if fields is not None:
        for raw_key, item in fields.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if key == "parts" and _is_sequence(item):
                for index, part in enumerate(cast(Sequence[object], item)):
                    _reject_a2a_remote_runtime_authority(
                        _a2a_part_payload(part),
                        path=f"{item_path}[{index}]",
                        policy=policy,
                        part_payload=True,
                    )
                continue
            if _is_allowed_a2a_profile_selector(key, item, policy):
                continue
            if remote_runtime_authority_key(key):
                raise ValueError(_a2a_authority_error(item_path, key))
            _reject_a2a_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
        return

    if _is_sequence(value):
        for index, item in enumerate(cast(Sequence[object], value)):
            _reject_a2a_remote_runtime_authority(
                item,
                path=f"{path}[{index}]",
                policy=policy,
            )


def _reject_a2a_part_remote_runtime_authority(
    value: object,
    *,
    path: str,
    policy: RemoteContainerRequestPolicy | None = None,
) -> None:
    fields = _a2a_authority_fields(value)
    if fields is None:
        return

    for raw_key, item in fields.items():
        key = str(raw_key)
        item_path = f"{path}.{key}"
        if key in _A2A_CONTENT_KEYS:
            _reject_a2a_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
            continue
        if key == "file":
            _reject_a2a_file_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
            continue
        if key == "metadata":
            _reject_a2a_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
            continue
        if _is_allowed_a2a_profile_selector(key, item, policy):
            continue
        if remote_runtime_authority_key(key):
            raise ValueError(_a2a_authority_error(item_path, key))
        _reject_a2a_remote_runtime_authority(
            item,
            path=item_path,
            policy=policy,
        )


def _reject_a2a_file_remote_runtime_authority(
    value: object,
    *,
    path: str,
    policy: RemoteContainerRequestPolicy | None = None,
) -> None:
    fields = _a2a_authority_fields(value)
    if fields is None:
        return

    for raw_key, item in fields.items():
        key = str(raw_key)
        item_path = f"{path}.{key}"
        if key in _A2A_FILE_CONTENT_KEYS:
            _reject_a2a_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
            continue
        if (
            key in _A2A_FILE_METADATA_KEYS
            and not _a2a_metadata_value_needs_scan(item)
        ):
            continue
        if key == "metadata":
            _reject_a2a_remote_runtime_authority(
                item,
                path=item_path,
                policy=policy,
            )
            continue
        if _is_allowed_a2a_profile_selector(key, item, policy):
            continue
        if remote_runtime_authority_key(key):
            raise ValueError(_a2a_authority_error(item_path, key))
        _reject_a2a_remote_runtime_authority(
            item,
            path=item_path,
            policy=policy,
        )


def _a2a_authority_fields(
    value: object,
) -> Mapping[object, object] | None:
    if isinstance(value, Mapping):
        return value

    jsonable = _jsonable_value(value)
    if isinstance(jsonable, Mapping):
        return cast(Mapping[object, object], jsonable)

    try:
        fields = vars(value)
    except TypeError:
        return None
    return {
        key: item
        for key, item in fields.items()
        if isinstance(key, str) and not key.startswith("_")
    }


def _a2a_metadata_value_needs_scan(value: object) -> bool:
    return isinstance(value, Mapping) or _is_sequence(value)


def _a2a_authority_error(path: str, key: str) -> str:
    return (
        "Remote A2A requests cannot provide runtime authority field "
        f"'{key}' at {path}"
    )


def _is_allowed_a2a_profile_selector(
    key: str,
    value: object,
    policy: RemoteContainerRequestPolicy | None,
) -> bool:
    if key not in REMOTE_CONTAINER_PROFILE_SELECTOR_KEYS:
        return False
    if key == "container" and not (
        isinstance(value, Mapping) and set(value) == {"profile"}
    ):
        return False
    if key != "container" and not isinstance(value, str):
        return False
    try:
        validate_remote_container_arguments({key: value}, policy=policy)
    except RemoteContainerRequestError as exc:
        raise ValueError(str(exc)) from exc
    return True


def _a2a_request_container_policy(
    request: object,
) -> RemoteContainerRequestPolicy | None:
    app = getattr(request, "app", None)
    state = getattr(app, "state", None)
    if state is None:
        return None
    return remote_container_policy_from_state(state)


def _ensure_typing_override() -> None:
    typing_module = import_module("typing")
    if hasattr(typing_module, "override"):
        return
    typing_extensions_module = import_module("typing_extensions")
    setattr(typing_module, "override", typing_extensions_module.override)


def _build_agent_card(
    *,
    a2a_pb2: Any,
    constants: Any,
    interface_url: str,
    name: str,
    description: str | None,
) -> Any:
    skill_description = description or "Execute the Avalan agent."
    return a2a_pb2.AgentCard(
        name=name,
        description=skill_description,
        version="1.0.0",
        supported_interfaces=[
            a2a_pb2.AgentInterface(
                url=interface_url,
                protocol_binding=constants.TransportProtocol.JSONRPC,
                protocol_version=constants.PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=a2a_pb2.AgentCapabilities(streaming=True),
        default_input_modes=A2A_FILE_MODES,
        default_output_modes=A2A_OUTPUT_MODES,
        skills=[
            a2a_pb2.AgentSkill(
                id=name,
                name=name,
                description=skill_description,
                tags=["avalan", "agent"],
                input_modes=A2A_FILE_MODES,
                output_modes=A2A_OUTPUT_MODES,
            )
        ],
    )


def _chat_content_from_a2a_context(
    context: Any,
) -> str | list[ContentFile | ContentImage | ContentText]:
    assert context is not None
    message = _a2a_context_message(context)
    content = _chat_content_from_a2a_message(message)
    if content is not None:
        return content

    get_user_input = getattr(context, "get_user_input", None)
    assert callable(get_user_input)
    text = get_user_input()
    assert isinstance(text, str)
    return text


def _a2a_context_message(context: Any) -> object | None:
    message = _field_value(context, "message")
    if message is not _MISSING and _a2a_message_parts(message):
        return message

    current_task = _field_value(context, "current_task")
    if current_task is _MISSING or current_task is None:
        return message if message is not _MISSING else None

    status = _field_value(current_task, "status")
    if status is not _MISSING and status is not None:
        status_message = _field_value(status, "message")
        if status_message is not _MISSING and _a2a_message_parts(
            status_message
        ):
            return status_message

    history = _field_value(current_task, "history")
    if _is_sequence(history):
        messages = list(cast(Sequence[object], history))
        for candidate in reversed(messages):
            if _is_user_a2a_message(candidate) and _a2a_message_parts(
                candidate
            ):
                return candidate
        for candidate in reversed(messages):
            if _a2a_message_parts(candidate):
                return candidate

    return message if message is not _MISSING else None


def _chat_content_from_a2a_message(
    message: object | None,
) -> str | list[ContentFile | ContentImage | ContentText] | None:
    parts = _a2a_message_parts(message)
    if not parts:
        return None

    content: list[ContentFile | ContentImage | ContentText] = []
    for part in parts:
        content_part = _content_from_a2a_part(part)
        if content_part is not None:
            content.append(content_part)
    if not content:
        return None
    text_content = [part for part in content if isinstance(part, ContentText)]
    if len(text_content) == len(content):
        return "\n".join(part.text for part in text_content)
    return content


def _content_from_a2a_part(
    part: object,
) -> ContentFile | ContentImage | ContentText | None:
    payload = _a2a_part_payload(part)
    content_fields = _a2a_part_content_fields(payload)
    if len(content_fields) != 1:
        return None

    content_field = content_fields[0]
    if content_field == "text":
        text = _string_field(payload, "text")
        if text is not None:
            return ContentText(type="text", text=text)
        return None

    if content_field == "raw":
        raw_source = _a2a_raw_source(payload)
        raw_data = _raw_file_data(raw_source)
        if raw_data is not None:
            file_payload = _field_value(payload, "file")
            media_type = _media_type(payload, file_payload)
            if _is_image_media_type(media_type):
                return ContentImage(
                    type="image_url",
                    image_url={
                        "url": f"data:{media_type};base64,{raw_data}",
                    },
                )
            return ContentFile(
                type="file",
                file=_file_metadata(payload, file_payload),
                file_data=raw_data,
                filename=_filename(payload, file_payload),
            )
        return None

    if content_field == "url":
        url = _string_field_value(_a2a_url_source(payload))
        if url is not None:
            file_payload = _field_value(payload, "file")
            media_type = _media_type(payload, file_payload)
            if _is_image_media_type(media_type):
                return ContentImage(type="image_url", image_url={"url": url})
            return ContentFile(
                type="file",
                file=_file_metadata(payload, file_payload),
                file_url=url,
                filename=_filename(payload, file_payload),
            )
        return None

    data = _field_value(payload, "data")
    data_text = _data_part_text(data)
    if data_text is not None:
        return ContentText(type="text", text=data_text)
    return None


def _a2a_part_content_fields(payload: object) -> list[str]:
    fields: list[str] = []
    if _field_value(payload, "text") is not _MISSING:
        fields.append("text")
    if _a2a_raw_source(payload) is not _MISSING:
        fields.append("raw")
    if _a2a_url_source(payload) is not _MISSING:
        fields.append("url")
    if _field_value(payload, "data") is not _MISSING:
        fields.append("data")
    return fields


def _a2a_raw_source(payload: object) -> object:
    raw_source = _field_value(payload, "raw")
    if raw_source is not _MISSING:
        return raw_source

    file_payload = _field_value(payload, "file")
    if file_payload is _MISSING:
        return _MISSING
    return _field_value(
        file_payload, "raw", "bytes", "file_data", "data", "base64"
    )


def _a2a_url_source(payload: object) -> object:
    url = _field_value(payload, "url", "uri", "file_url")
    if url is not _MISSING:
        return url

    file_payload = _field_value(payload, "file")
    if file_payload is _MISSING:
        return _MISSING
    return _field_value(file_payload, "url", "uri", "file_url")


def _string_field_value(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _a2a_part_payload(part: object) -> object:
    root = _field_value(part, "root")
    return root if root is not _MISSING and root is not None else part


def _a2a_message_parts(message: object | None) -> list[object]:
    if message is None:
        return []
    parts = _field_value(message, "parts")
    if not _is_sequence(parts):
        return []
    return list(cast(Sequence[object], parts))


def _is_user_a2a_message(message: object) -> bool:
    role = _field_value(message, "role")
    if role is _MISSING or role is None:
        return True
    role_name = getattr(role, "name", None)
    is_user = _role_value_is_user(role_name)
    if is_user is not None:
        return is_user
    role_value = getattr(role, "value", None)
    is_user = _role_value_is_user(role_value)
    if is_user is not None:
        return is_user
    is_user = _role_value_is_user(role)
    if is_user is not None:
        return is_user
    return str(role).lower().endswith("user")


def _role_value_is_user(value: object) -> bool | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.lower().endswith("user")
    return None


def _raw_file_data(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            decoded = _decode_a2a_base64(stripped)
        except (BinasciiError, ValueError):
            return None
        return b64encode(decoded).decode("ascii")
    if isinstance(value, bytes):
        return b64encode(value).decode("ascii") if value else None
    if isinstance(value, bytearray):
        return b64encode(bytes(value)).decode("ascii") if value else None
    if isinstance(value, memoryview):
        data = value.tobytes()
        return b64encode(data).decode("ascii") if data else None
    nested = _field_value(value, "raw", "bytes", "file_data", "data", "base64")
    if nested is value:
        return None
    return _raw_file_data(nested)


def _file_metadata(*sources: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    filename = _filename(*sources)
    if filename is not None:
        metadata["filename"] = filename
    media_type = _media_type(*sources)
    if media_type is not None:
        metadata["mime_type"] = media_type
    return metadata


def _filename(*sources: object) -> str | None:
    return _first_string(
        sources,
        "filename",
        "file_name",
        "name",
        "display_name",
        "displayName",
    )


def _media_type(*sources: object) -> str | None:
    return _first_string(
        sources,
        "media_type",
        "mediaType",
        "mime_type",
        "mimeType",
    )


def _is_image_media_type(media_type: str | None) -> bool:
    return media_type is not None and media_type.lower().startswith("image/")


def _first_string(sources: tuple[object, ...], *names: str) -> str | None:
    for source in sources:
        value = _string_field(source, *names)
        if value is not None:
            return value
        metadata = _field_value(source, "metadata")
        value = _string_field(metadata, *names)
        if value is not None:
            return value
        json_metadata = _jsonable_value(metadata)
        value = _string_field(json_metadata, *names)
        if value is not None:
            return value
    return None


def _string_field(source: object, *names: str) -> str | None:
    value = _field_value(source, *names)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _data_part_text(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    jsonable = _jsonable_value(value)
    if jsonable is _MISSING:
        return None
    return dumps(jsonable, separators=(",", ":"), sort_keys=True)


def _jsonable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            jsonable_item = _jsonable_value(item)
            if jsonable_item is not _MISSING:
                result[str(key)] = jsonable_item
        return result
    if _is_sequence(value):
        return [
            item
            for item in (
                _jsonable_value(item) for item in cast(Sequence[object], value)
            )
            if item is not _MISSING
        ]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="python", exclude_none=True)
        except TypeError:
            dumped = model_dump()
        return _jsonable_value(dumped)

    if hasattr(value, "DESCRIPTOR"):
        json_format = import_module("google.protobuf.json_format")
        return _jsonable_value(json_format.MessageToDict(value))

    return _MISSING


def _field_value(source: object, *names: str) -> object:
    if source is _MISSING or source is None:
        return _MISSING
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return _MISSING
    has_field = getattr(source, "HasField", None)
    for name in names:
        if callable(has_field):
            try:
                if not has_field(name):
                    continue
            except (TypeError, ValueError):
                # HasField raises for non-protobuf or unsupported names; fall
                # back to getattr probing for dict-like SDK variants.
                pass
        try:
            value = getattr(source, name)
        except AttributeError:
            continue
        if callable(value):
            continue
        return value
    return _MISSING


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview, str)
    )


class AvalanA2AAgentExecutor:
    """Execute Avalan orchestrator calls for A2A SDK routes."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    async def execute(self, context: Any, event_queue: Any) -> None:
        _ensure_typing_override()
        task_id = cast(str, context.task_id)
        context_id = cast(str, context.context_id)
        updater_module = import_module("a2a.server.tasks.task_updater")
        updater = updater_module.TaskUpdater(
            event_queue, task_id=task_id, context_id=context_id
        )
        a2a_pb2 = import_module("a2a.types.a2a_pb2")
        if context.current_task is None:
            await event_queue.enqueue_event(
                a2a_pb2.Task(
                    id=task_id,
                    context_id=context_id,
                    status=a2a_pb2.TaskStatus(
                        state=a2a_pb2.TaskState.TASK_STATE_SUBMITTED
                    ),
                )
            )
        await updater.update_status(
            a2a_pb2.TaskState.TASK_STATE_WORKING,
            metadata={"source": "avalan"},
        )

        response: AsyncIterable[object] | None = None
        iterator: AsyncIterator[object] | None = None
        translator: A2AResponseTranslator | None = None
        try:
            orchestrator = await self._orchestrator()
            request = await self._chat_request(context, orchestrator)
            logger = cast(Logger, self._app.state.logger)
            response, _response_uuid, _timestamp = await orchestrate(
                request, logger, orchestrator
            )
            translator = A2AResponseTranslator(
                updater,
                output_redaction_settings=(
                    server_output_redaction_settings_from_state(
                        self._app.state
                    )
                ),
            )
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="a2a-stream",
                run_id=task_id,
                turn_id=context_id,
                unsupported_message="unsupported A2A stream item",
                close_source_on_generator_exit=False,
            )
            async for item in iterator:
                await translator.process(item)
            await translator.finish()
            if translator.succeeded:
                await orchestrator.sync_messages(response)
        except CancelledError:
            if translator is None:
                await updater.cancel()
            else:
                await translator.abort(StreamTerminalOutcome.CANCELLED)
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=True
                )
            raise
        except Exception:
            if translator is None:
                await updater.failed()
            else:
                await translator.abort(StreamTerminalOutcome.ERRORED)
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=False
                )
            raise
        else:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=False
                )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        _ensure_typing_override()
        task_id = cast(str, context.task_id)
        context_id = cast(str, context.context_id)
        updater_module = import_module("a2a.server.tasks.task_updater")
        updater = updater_module.TaskUpdater(
            event_queue, task_id=task_id, context_id=context_id
        )
        await updater.cancel()

    async def _orchestrator(self) -> Orchestrator:
        server_module = import_module("avalan.server")
        return cast(
            Orchestrator,
            await server_module.di_get_orchestrator_from_app(self._app),
        )

    async def _chat_request(
        self, context: Any, orchestrator: Orchestrator
    ) -> ChatCompletionRequest:
        _reject_a2a_remote_runtime_authority(
            context,
            path="a2a.context",
            policy=remote_container_policy_from_state(self._app.state),
        )
        return ChatCompletionRequest(
            model=resolve_model_id(orchestrator),
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=_chat_content_from_a2a_context(context),
                )
            ],
            stream=True,
        )


class A2AResponseTranslator:
    """Translate canonical Avalan stream items to A2A SDK events."""

    def __init__(
        self,
        updater: Any,
        *,
        output_redaction_settings: ServerOutputRedactionSettings | None = None,
        retention_policy: StreamRetentionPolicy | None = None,
    ) -> None:
        self._updater = updater
        self._a2a_pb2 = import_module("a2a.types.a2a_pb2")
        self._terminal_outcome: StreamTerminalOutcome | None = None
        self._finished = False
        self._locally_closed = False
        self._open_artifacts: set[str] = set()
        output_redaction_settings = coerce_server_output_redaction_settings(
            output_redaction_settings
        )
        self._output_redaction_settings = output_redaction_settings
        self._answer_redactor = ModelVisibleServerProtocolTextRedactor(
            output_redaction_settings,
            protocol="a2a",
            channel="answer",
        )
        self._reasoning_redaction = ProtocolReasoningRedactionState(
            output_redaction_settings,
            protocol="a2a",
        )
        self._pending_model_text_sequences: dict[str, int] = {}
        retention = protocol_stream_retention_settings(retention_policy)
        self._reasoning_segment_limit = retention.a2a_reasoning_segment_limit
        self._reasoning_character_limit = (
            retention.a2a_reasoning_character_limit
        )
        self._reasoning_utf8_byte_limit = (
            retention.a2a_reasoning_text_byte_limit
        )
        self._reasoning_current: _A2AReasoningArtifactState | None = None
        self._reasoning_last_identity: ProtocolReasoningIdentity | None = None
        self._reasoning_suppressed_identity: (
            ProtocolReasoningIdentity | None
        ) = None
        self._reasoning_completed: deque[_A2AReasoningArtifactState] = deque()
        self._reasoning_retained_characters = 0
        self._reasoning_retained_utf8_bytes = 0
        self._reasoning_dropped_artifacts = 0
        self._reasoning_dropped_characters = 0
        self._reasoning_dropped_utf8_bytes = 0
        self._reasoning_continuation_ordinals: dict[str | None, int] = {}
        self._reasoning_next_continuation_ordinal = 0
        self._reasoning_next_segment_ordinals: dict[int, int] = {}
        self._reasoning_continuation_ordinal_hint: int | None = None

    @property
    def succeeded(self) -> bool:
        return not self._locally_closed and stream_terminal_succeeded(
            self._terminal_outcome
        )

    async def process(self, item: object) -> None:
        if isinstance(item, CanonicalStreamItem):
            canonical_item = item
        elif isinstance(item, StreamConsumerProjection):
            canonical_item = canonical_item_from_consumer_projection(item)
        else:
            raise StreamValidationError("unsupported A2A stream item")
        await self._process_canonical_item(canonical_item)

    async def finish(self) -> None:
        if self._finished:
            return
        input_required_state: object | None = None
        if self._terminal_outcome is StreamTerminalOutcome.INPUT_REQUIRED:
            input_required_state = getattr(
                self._a2a_pb2.TaskState,
                "TASK_STATE_INPUT_REQUIRED",
                None,
            )
            if input_required_state is None:
                raise StreamValidationError(
                    "A2A SDK input-required task state is unavailable"
                )
        self._finished = True
        if self._locally_closed and self._terminal_outcome is None:
            return
        await self._flush_model_text(self._terminal_outcome)
        for artifact_id in tuple(self._open_artifacts):
            await self._finish_artifact(artifact_id)
        if self._terminal_outcome is StreamTerminalOutcome.CANCELLED:
            await self._updater.cancel()
        elif self._terminal_outcome is StreamTerminalOutcome.ERRORED:
            await self._updater.failed()
        elif self._terminal_outcome is StreamTerminalOutcome.INPUT_REQUIRED:
            assert input_required_state is not None
            await self._updater.update_status(input_required_state)
        else:
            assert self._terminal_outcome in (
                None,
                StreamTerminalOutcome.COMPLETED,
            )
            await self._updater.complete()

    async def abort(self, outcome: StreamTerminalOutcome) -> None:
        """Finish the current translation with one abnormal task terminal."""
        assert outcome in (
            StreamTerminalOutcome.CANCELLED,
            StreamTerminalOutcome.ERRORED,
        )
        if self._finished:
            return
        self._terminal_outcome = outcome
        await self.finish()

    async def _process_canonical_item(self, item: CanonicalStreamItem) -> None:
        if self._terminal_outcome is not None:
            return
        if item.kind is StreamItemKind.STREAM_CLOSED:
            self._locally_closed = True
            return
        if item.is_stream_terminal:
            self._terminal_outcome = item.terminal_outcome
            return
        if item.kind is not StreamItemKind.REASONING_DELTA:
            await self._close_reasoning_boundary(None)
        if item.kind is StreamItemKind.MODEL_CONTINUATION_STARTED:
            continuation_id = item.correlation.model_continuation_id
            assert continuation_id is not None
            self._reasoning_continuation_ordinal_hint = (
                self._reasoning_continuation_ordinal(continuation_id)
            )
            return
        if item.kind is StreamItemKind.ANSWER_DELTA:
            for text in self._answer_redactor.push(item.text_delta or ""):
                await self._add_text_artifact(
                    artifact_id="answer",
                    text=text,
                    metadata={"kind": "answer", "channel": "output"},
                    name="Answer",
                )
            self._record_model_text_pending(
                "answer",
                item.sequence,
                self._answer_redactor,
            )
            return
        if item.kind is StreamItemKind.REASONING_DELTA:
            await self._process_reasoning_item(item)
            return
        if item.channel in (
            StreamChannel.TOOL_CALL,
            StreamChannel.TOOL_EXECUTION,
        ):
            await self._process_tool_item(item)

    async def _process_reasoning_item(
        self,
        item: CanonicalStreamItem,
    ) -> None:
        identity = ProtocolReasoningIdentity.from_item(item)
        text = item.text_delta
        assert isinstance(text, str) and text
        previous = self._reasoning_last_identity

        if (
            self._reasoning_suppressed_identity is not None
            and identity == self._reasoning_suppressed_identity
        ):
            self._record_reasoning_drop(text)
            return

        if previous is not None and identity != previous:
            admission = self._reasoning_redaction.preview_push(identity, text)
            if admission.marker_reserved:
                if not self._ensure_reasoning_capacity(
                    admission.required_character_count,
                    admission.required_utf8_byte_count,
                    additional_artifacts=0,
                ):
                    await self._reject_reasoning_marker_boundary(
                        item, identity
                    )
                    return
                outputs = self._reasoning_redaction.push(identity, text)
                await self._emit_reasoning_outputs(outputs)
                await self._finish_reasoning_artifact(None)
                self._reasoning_last_identity = None
                self._reasoning_suppressed_identity = None
                self._pending_model_text_sequences.pop("reasoning", None)
                return
            if admission.suppressed:
                outputs = self._reasoning_redaction.push(identity, text)
                await self._emit_reasoning_outputs(outputs)
                await self._finish_reasoning_artifact(None)
                self._reasoning_last_identity = identity
                self._reasoning_suppressed_identity = None
                self._record_reasoning_pending(item.sequence)
                return
            await self._complete_reasoning_redactor(previous)
            await self._finish_reasoning_artifact(None)
            self._reasoning_last_identity = None
            self._reasoning_suppressed_identity = None

        admission = self._reasoning_redaction.preview_push(identity, text)
        if admission.suppressed:
            outputs = self._reasoning_redaction.push(identity, text)
            await self._emit_reasoning_outputs(outputs)
            self._reasoning_last_identity = (
                identity
                if self._reasoning_redaction.identity is not None
                else None
            )
            self._record_reasoning_pending(item.sequence)
            return

        additional_artifacts = int(self._reasoning_current is None)
        if not self._ensure_reasoning_capacity(
            admission.required_character_count,
            admission.required_utf8_byte_count,
            additional_artifacts=additional_artifacts,
        ):
            self._ensure_reasoning_artifact(item.run_id, identity)
            assert self._reasoning_current is not None
            self._reasoning_current.suppressed = True
            self._reasoning_suppressed_identity = identity
            self._reasoning_last_identity = identity
            self._record_reasoning_drop(text)
            self._record_reasoning_pending(item.sequence)
            return

        self._ensure_reasoning_artifact(item.run_id, identity)
        outputs = self._reasoning_redaction.push(identity, text)
        self._reasoning_last_identity = identity
        await self._emit_reasoning_outputs(outputs)
        self._record_reasoning_pending(item.sequence)

    async def _reject_reasoning_marker_boundary(
        self,
        item: CanonicalStreamItem,
        identity: ProtocolReasoningIdentity,
    ) -> None:
        current = self._reasoning_current
        if current is not None:
            current.suppressed = True
            current.dropped_characters += (
                self._reasoning_redaction.pending_character_count
            )
            current.dropped_utf8_bytes += (
                self._reasoning_redaction.pending_utf8_byte_count
            )
            await self._finish_reasoning_artifact(None)
        self._reasoning_redaction = ProtocolReasoningRedactionState(
            self._output_redaction_settings,
            protocol="a2a",
        )
        self._reasoning_last_identity = identity
        self._reasoning_suppressed_identity = identity
        self._pending_model_text_sequences.pop("reasoning", None)
        self._ensure_reasoning_artifact(item.run_id, identity)
        assert item.text_delta is not None
        assert self._reasoning_current is not None
        self._reasoning_current.suppressed = True
        self._record_reasoning_drop(item.text_delta)

    async def _complete_reasoning_redactor(
        self,
        identity: ProtocolReasoningIdentity,
    ) -> None:
        if self._reasoning_redaction.identity is None:
            return
        outputs = self._reasoning_redaction.complete(identity)
        await self._emit_reasoning_outputs(outputs)

    async def _close_reasoning_boundary(
        self,
        outcome: StreamTerminalOutcome | None,
    ) -> None:
        identity = self._reasoning_last_identity
        if identity is not None:
            await self._complete_reasoning_redactor(identity)
        await self._finish_reasoning_artifact(outcome)
        self._reasoning_last_identity = None
        self._reasoning_suppressed_identity = None
        self._pending_model_text_sequences.pop("reasoning", None)

    def _ensure_reasoning_capacity(
        self,
        required_characters: int,
        required_utf8_bytes: int,
        *,
        additional_artifacts: int,
    ) -> bool:
        while self._reasoning_completed and (
            len(self._reasoning_completed)
            + int(self._reasoning_current is not None)
            + additional_artifacts
            > self._reasoning_segment_limit
            or self._reasoning_retained_characters + required_characters
            > self._reasoning_character_limit
            or self._reasoning_retained_utf8_bytes + required_utf8_bytes
            > self._reasoning_utf8_byte_limit
        ):
            dropped = self._reasoning_completed.popleft()
            self._reasoning_retained_characters -= dropped.characters
            self._reasoning_retained_utf8_bytes -= dropped.utf8_bytes
            self._reasoning_dropped_artifacts += 1
            self._reasoning_dropped_characters += dropped.characters
            self._reasoning_dropped_utf8_bytes += dropped.utf8_bytes
        return (
            len(self._reasoning_completed)
            + int(self._reasoning_current is not None)
            + additional_artifacts
            <= self._reasoning_segment_limit
            and self._reasoning_retained_characters + required_characters
            <= self._reasoning_character_limit
            and self._reasoning_retained_utf8_bytes + required_utf8_bytes
            <= self._reasoning_utf8_byte_limit
        )

    def _ensure_reasoning_artifact(
        self,
        run_id: str,
        identity: ProtocolReasoningIdentity,
    ) -> None:
        current = self._reasoning_current
        if current is not None:
            assert current.identity == identity
            return
        if identity.continuation_id is not None:
            continuation_ordinal = self._reasoning_continuation_ordinal(
                identity.continuation_id
            )
            self._reasoning_continuation_ordinal_hint = continuation_ordinal
        elif self._reasoning_continuation_ordinal_hint is not None:
            continuation_ordinal = self._reasoning_continuation_ordinal_hint
        else:
            continuation_ordinal = self._reasoning_continuation_ordinal(None)
        segment_ordinal = self._reasoning_next_segment_ordinals.get(
            continuation_ordinal,
            0,
        )
        self._reasoning_next_segment_ordinals[continuation_ordinal] = (
            segment_ordinal + 1
        )
        self._reasoning_current = _A2AReasoningArtifactState(
            identity=identity,
            artifact_id=(
                f"reasoning-{run_id}-{continuation_ordinal}-{segment_ordinal}"
            ),
        )

    def _reasoning_continuation_ordinal(
        self,
        continuation_id: str | None,
    ) -> int:
        ordinal = self._reasoning_continuation_ordinals.get(continuation_id)
        if ordinal is not None:
            return ordinal
        ordinal = self._reasoning_next_continuation_ordinal
        self._reasoning_next_continuation_ordinal += 1
        self._reasoning_continuation_ordinals[continuation_id] = ordinal
        return ordinal

    async def _emit_reasoning_outputs(
        self,
        outputs: tuple[ProtocolReasoningRedactedText, ...],
    ) -> None:
        for output in outputs:
            current = self._reasoning_current
            assert current is not None
            assert current.identity == output.identity
            await self._add_text_artifact(
                artifact_id=current.artifact_id,
                text=output.text,
                metadata=self._reasoning_metadata(
                    current,
                    status="in_progress",
                    terminal_outcome=None,
                ),
                name="Reasoning",
            )
            characters = len(output.text)
            utf8_bytes = len(output.text.encode("utf-8"))
            current.characters += characters
            current.utf8_bytes += utf8_bytes
            current.opened = True
            self._reasoning_retained_characters += characters
            self._reasoning_retained_utf8_bytes += utf8_bytes

    def _record_reasoning_drop(self, text: str) -> None:
        current = self._reasoning_current
        assert current is not None
        current.dropped_characters += len(text)
        current.dropped_utf8_bytes += len(text.encode("utf-8"))

    def _record_reasoning_pending(self, sequence: int) -> None:
        if self._reasoning_redaction.marker_reserved:
            self._pending_model_text_sequences.setdefault(
                "reasoning",
                sequence,
            )
        else:
            self._pending_model_text_sequences.pop("reasoning", None)

    async def _finish_reasoning_artifact(
        self,
        outcome: StreamTerminalOutcome | None,
    ) -> None:
        current = self._reasoning_current
        if current is None:
            return
        if (
            outcome
            in (
                StreamTerminalOutcome.CANCELLED,
                StreamTerminalOutcome.ERRORED,
                StreamTerminalOutcome.INPUT_REQUIRED,
            )
            or current.suppressed
        ):
            status = "incomplete"
        else:
            status = "completed"
        metadata = self._reasoning_metadata(
            current,
            status=status,
            terminal_outcome=_a2a_reasoning_terminal_outcome(outcome),
        )
        if current.opened:
            await self._finish_artifact(
                current.artifact_id,
                metadata=metadata,
            )
        else:
            await self._updater.add_artifact(
                [],
                artifact_id=current.artifact_id,
                name="Reasoning",
                metadata=metadata,
                append=False,
                last_chunk=True,
            )
        self._reasoning_completed.append(current)
        self._reasoning_current = None
        self._ensure_reasoning_capacity(
            0,
            0,
            additional_artifacts=0,
        )

    def _reasoning_metadata(
        self,
        artifact: _A2AReasoningArtifactState,
        *,
        status: str,
        terminal_outcome: str | None,
    ) -> dict[str, Any]:
        identity = artifact.identity
        metadata: dict[str, Any] = {
            "kind": "reasoning",
            "channel": "reasoning",
            "representation": identity.representation.value,
            "segment_instance_ordinal": identity.segment_instance_ordinal,
            "status": status,
            "terminal_outcome": terminal_outcome,
            "truncation": {
                "truncated": bool(
                    self._reasoning_dropped_artifacts
                    or self._reasoning_dropped_characters
                    or self._reasoning_dropped_utf8_bytes
                    or artifact.dropped_characters
                    or artifact.dropped_utf8_bytes
                ),
                "dropped_artifacts": self._reasoning_dropped_artifacts,
                "dropped_characters": (
                    self._reasoning_dropped_characters
                    + artifact.dropped_characters
                ),
                "dropped_utf8_bytes": (
                    self._reasoning_dropped_utf8_bytes
                    + artifact.dropped_utf8_bytes
                ),
            },
        }
        for key, value in (
            ("provider_item_id", identity.provider_item_id),
            ("output_index", identity.output_index),
            ("summary_index", identity.summary_index),
            ("continuation_id", identity.continuation_id),
        ):
            if value is not None:
                metadata[key] = value
        return metadata

    async def _process_tool_item(self, item: CanonicalStreamItem) -> None:
        tool_call_id = item.correlation.tool_call_id or "tool"
        data = item.data if isinstance(item.data, dict) else {}
        metadata: dict[str, Any] = {
            "kind": "tool",
            "channel": item.channel.value,
            "phase": item.kind.value,
            "tool_call_id": tool_call_id,
        }
        tool_name = data.get("name") or item.metadata.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            metadata["tool_name"] = tool_name
        category = _a2a_tool_item_category(data)
        if category is not None:
            metadata["category"] = category
        tool_name_text = tool_name if isinstance(tool_name, str) else None
        metadata = cast(
            dict[str, Any],
            sanitize_server_protocol_value(
                metadata,
                tool_name=tool_name_text,
                output_redaction_settings=self._output_redaction_settings,
                protocol="a2a",
            ),
        )
        text = _a2a_tool_item_text(
            item,
            data,
            tool_name=tool_name_text,
            output_redaction_settings=self._output_redaction_settings,
        )
        if text:
            await self._add_text_artifact(
                artifact_id=tool_call_id,
                text=text,
                metadata=metadata,
                name=cast(str | None, tool_name),
            )
        await self._updater.update_status(
            self._a2a_pb2.TaskState.TASK_STATE_WORKING,
            metadata=metadata,
        )
        if item.kind in (
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        ):
            await self._finish_artifact(tool_call_id)

    async def _flush_model_text(
        self,
        outcome: StreamTerminalOutcome | None,
    ) -> None:
        for channel, _sequence in sorted(
            self._pending_model_text_sequences.items(),
            key=lambda item: item[1],
        ):
            if channel == "answer":
                for text in self._answer_redactor.flush():
                    await self._add_text_artifact(
                        artifact_id="answer",
                        text=text,
                        metadata={"kind": "answer", "channel": "output"},
                        name="Answer",
                    )
            elif channel == "reasoning":
                await self._close_reasoning_boundary(outcome)
        self._pending_model_text_sequences.clear()
        await self._close_reasoning_boundary(outcome)

    def _record_model_text_pending(
        self,
        channel: str,
        sequence: int,
        redactor: ModelVisibleServerProtocolTextRedactor,
    ) -> None:
        if redactor.has_pending:
            self._pending_model_text_sequences.setdefault(channel, sequence)
        else:
            self._pending_model_text_sequences.pop(channel, None)

    async def _add_text_artifact(
        self,
        *,
        artifact_id: str,
        text: str,
        metadata: dict[str, Any],
        name: str | None,
    ) -> None:
        append = artifact_id in self._open_artifacts
        self._open_artifacts.add(artifact_id)
        await self._updater.add_artifact(
            [self._a2a_pb2.Part(text=text)],
            artifact_id=artifact_id,
            name=name,
            metadata=metadata,
            append=append,
        )

    async def _finish_artifact(
        self,
        artifact_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._updater.add_artifact(
            [],
            artifact_id=artifact_id,
            append=True,
            last_chunk=True,
            **({"metadata": metadata} if metadata is not None else {}),
        )
        self._open_artifacts.discard(artifact_id)


def _a2a_reasoning_terminal_outcome(
    outcome: StreamTerminalOutcome | None,
) -> str | None:
    if outcome is None or outcome is StreamTerminalOutcome.COMPLETED:
        return "completed"
    if outcome is StreamTerminalOutcome.ERRORED:
        return "failed"
    if outcome is StreamTerminalOutcome.CANCELLED:
        return "cancelled"
    assert outcome is StreamTerminalOutcome.INPUT_REQUIRED
    return "input_required"


def _a2a_tool_item_category(data: Mapping[str, object]) -> str | None:
    category = data.get("category")
    if (
        not isinstance(category, str)
        or category not in _A2A_TOOL_RESOURCE_CATEGORIES
    ):
        return None
    return "logs" if category == "log" else category


def _a2a_tool_item_text(
    item: CanonicalStreamItem,
    data: Mapping[str, object],
    *,
    tool_name: str | None = None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    if item.text_delta:
        return _a2a_protocol_text(
            item.text_delta,
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        content = data.get("content")
        if isinstance(content, str):
            return _a2a_protocol_text(
                content,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
        return ""
    if item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS:
        content = data.get("content")
        if isinstance(content, str):
            return _a2a_protocol_text(
                content,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
        progress = data.get("progress")
        if progress is None:
            return ""
        return _a2a_protocol_payload_text(
            {"progress": progress},
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    if item.kind not in (
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    ):
        return ""
    for key in ("result", "error", "message"):
        payload = data.get(key)
        if payload is None:
            continue
        if isinstance(payload, str):
            return _a2a_protocol_text(
                payload,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
        if isinstance(payload, bool | int | float | list | dict):
            return _a2a_protocol_payload_text(
                payload,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
    return _a2a_protocol_text(
        _a2a_public_diagnostic_text(data.get("diagnostic")),
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
    )


def _a2a_protocol_text(
    value: str,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    if _a2a_is_skills_tool(tool_name) and settings.should_redact(
        "skills_tool_content",
        protocol="a2a",
    ):
        sanitized = sanitize_server_protocol_value(
            {"content": value},
            tool_name=tool_name,
            output_redaction_settings=settings,
            protocol="a2a",
        )
        return _a2a_protocol_payload_text(
            sanitized,
            tool_name=None,
            output_redaction_settings=settings,
        )
    return sanitize_server_protocol_text(
        value,
        output_redaction_settings=settings,
        protocol="a2a",
    )


def _a2a_protocol_payload_text(
    value: object,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    sanitized = sanitize_server_protocol_value(
        value,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
        protocol="a2a",
    )
    if isinstance(sanitized, str):
        return sanitized
    return dumps(sanitized, separators=(",", ":"))


def _a2a_is_skills_tool(value: str | None) -> bool:
    return isinstance(value, str) and value.startswith("skills.")


def _a2a_public_diagnostic_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    parts: list[str] = []
    for key in ("code", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if not parts:
        return ""
    return ": ".join(parts)
