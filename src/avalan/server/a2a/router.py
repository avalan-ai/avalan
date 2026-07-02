from ...agent.orchestrator import Orchestrator
from ...entities import MessageRole
from ...model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
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
    sanitize_server_protocol_text,
    sanitize_server_protocol_value,
)
from ..routers import orchestrate, resolve_model_id
from ..routers.streaming import (
    cleanup_stream_sources,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from copy import deepcopy
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
        try:
            orchestrator = await self._orchestrator()
            request = await self._chat_request(context, orchestrator)
            logger = cast(Logger, self._app.state.logger)
            response, _response_uuid, _timestamp = await orchestrate(
                request, logger, orchestrator
            )
            translator = A2AResponseTranslator(updater)
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
                await orchestrator.sync_messages()
        except CancelledError:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=True
                )
            await updater.cancel()
            raise
        except Exception:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=False
                )
            await updater.failed()
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

    def __init__(self, updater: Any) -> None:
        self._updater = updater
        self._a2a_pb2 = import_module("a2a.types.a2a_pb2")
        self._terminal_outcome: StreamTerminalOutcome | None = None
        self._open_artifacts: set[str] = set()
        self._answer_redactor = ModelVisibleServerProtocolTextRedactor()
        self._reasoning_redactor = ModelVisibleServerProtocolTextRedactor()
        self._pending_model_text_sequences: dict[str, int] = {}

    @property
    def succeeded(self) -> bool:
        return stream_terminal_succeeded(self._terminal_outcome)

    async def process(self, item: object) -> None:
        if isinstance(item, CanonicalStreamItem):
            canonical_item = item
        elif isinstance(item, StreamConsumerProjection):
            canonical_item = canonical_item_from_consumer_projection(item)
        else:
            raise StreamValidationError("unsupported A2A stream item")
        await self._process_canonical_item(canonical_item)

    async def finish(self) -> None:
        await self._flush_model_text()
        for artifact_id in tuple(self._open_artifacts):
            await self._finish_artifact(artifact_id)
        if self._terminal_outcome is StreamTerminalOutcome.CANCELLED:
            await self._updater.cancel()
        elif self._terminal_outcome is StreamTerminalOutcome.ERRORED:
            await self._updater.failed()
        else:
            await self._updater.complete()

    async def _process_canonical_item(self, item: CanonicalStreamItem) -> None:
        if item.is_stream_terminal:
            self._terminal_outcome = item.terminal_outcome
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
            for text in self._reasoning_redactor.push(item.text_delta or ""):
                await self._add_text_artifact(
                    artifact_id="reasoning",
                    text=text,
                    metadata={"kind": "reasoning", "channel": "reasoning"},
                    name="Reasoning",
                )
            self._record_model_text_pending(
                "reasoning",
                item.sequence,
                self._reasoning_redactor,
            )
            return
        if item.channel in (
            StreamChannel.TOOL_CALL,
            StreamChannel.TOOL_EXECUTION,
        ):
            await self._process_tool_item(item)

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
            ),
        )
        text = _a2a_tool_item_text(item, data, tool_name=tool_name_text)
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

    async def _flush_model_text(self) -> None:
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
                for text in self._reasoning_redactor.flush():
                    await self._add_text_artifact(
                        artifact_id="reasoning",
                        text=text,
                        metadata={
                            "kind": "reasoning",
                            "channel": "reasoning",
                        },
                        name="Reasoning",
                    )
        self._pending_model_text_sequences.clear()

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

    async def _finish_artifact(self, artifact_id: str) -> None:
        await self._updater.add_artifact(
            [],
            artifact_id=artifact_id,
            append=True,
            last_chunk=True,
        )
        self._open_artifacts.discard(artifact_id)


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
) -> str:
    if item.text_delta:
        return sanitize_server_protocol_text(item.text_delta)
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        content = data.get("content")
        if isinstance(content, str):
            return _a2a_protocol_text(content, tool_name=tool_name)
        return ""
    if item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS:
        content = data.get("content")
        if isinstance(content, str):
            return _a2a_protocol_text(content, tool_name=tool_name)
        progress = data.get("progress")
        if progress is None:
            return ""
        return _a2a_protocol_payload_text(
            {"progress": progress},
            tool_name=tool_name,
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
            return _a2a_protocol_text(payload, tool_name=tool_name)
        if isinstance(payload, bool | int | float | list | dict):
            return _a2a_protocol_payload_text(payload, tool_name=tool_name)
    return sanitize_server_protocol_text(
        _a2a_public_diagnostic_text(data.get("diagnostic"))
    )


def _a2a_protocol_text(value: str, *, tool_name: str | None) -> str:
    if _a2a_is_skills_tool(tool_name):
        sanitized = sanitize_server_protocol_value(
            {"content": value},
            tool_name=tool_name,
        )
        return _a2a_protocol_payload_text(sanitized, tool_name=None)
    return sanitize_server_protocol_text(value)


def _a2a_protocol_payload_text(
    value: object,
    *,
    tool_name: str | None,
) -> str:
    sanitized = sanitize_server_protocol_value(value, tool_name=tool_name)
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
