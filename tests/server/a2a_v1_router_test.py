from asyncio import CancelledError, Event, create_task, sleep
from base64 import b64encode
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from json import loads
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from avalan.interaction.a2a import (
    A2A_INPUT_EXTENSION_URI,
    A2AInputRequestMetadata,
    decode_a2a_input_resolution_metadata,
)
from avalan.interaction.entities import (
    AnsweredResolution,
    CancelledResolution,
    ConfirmationQuestion,
    InputRequestId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    PrincipalScope,
    QuestionId,
    ResolutionStatus,
    TextAnswer,
    TextQuestion,
    UserId,
)
from avalan.interaction.error import InputContractError, InputErrorCode
from avalan.interaction.policy import (
    InteractionActor,
    InteractionPolicy,
    TaskInputCapabilityState,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
)
from avalan.server.a2a import router as a2a_router
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    AvalanA2AAgentExecutor,
    install_a2a_routes,
)
from avalan.server.container_policy import RemoteContainerRequestPolicy
from avalan.server.entities import (
    ContentFile,
    ContentImage,
    ContentText,
    OrchestratorContext,
    ServerOutputRedactionSettings,
)
from avalan.server.interaction import (
    ServerInteractionConfiguration,
    close_server_interactions,
    configure_server_interactions,
)

_MODEL_VISIBLE_REDACTION_SETTINGS = ServerOutputRedactionSettings(enabled=True)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _interaction_configuration(
    policy: InteractionPolicy,
) -> ServerInteractionConfiguration:
    broker = SimpleNamespace(
        policy=policy,
        inspect=AsyncMock(),
        resolve=AsyncMock(),
        cancel=AsyncMock(),
        wait=AsyncMock(),
    )
    configuration = ServerInteractionConfiguration(
        broker=cast(Any, broker),
        principal_resolver=AsyncMock(),
        authorizer=cast(
            Any,
            SimpleNamespace(authorize=AsyncMock()),
        ),
        policy=policy,
    )
    return configuration


def _active_interaction_configuration(
    app: FastAPI,
) -> ServerInteractionConfiguration:
    """Install one live active interaction configuration on an app."""
    configuration = _interaction_configuration(InteractionPolicy())
    configure_server_interactions(app, configuration)
    return configuration


def test_install_a2a_routes_mounts_v1_sdk_routes() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )

    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/.well-known/agent-card.json" in paths
    assert "/a2a" in paths
    assert "/{tenant}/a2a" in paths
    assert "/a2a/message:stream" in paths
    assert "/.well-known/a2a-agent.json" not in paths


def test_agent_card_requires_live_active_interaction_configuration() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    dormant_app = FastAPI()
    install_a2a_routes(
        dormant_app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    active_app = FastAPI()
    configuration = _active_interaction_configuration(active_app)
    install_a2a_routes(
        active_app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )

    dormant = (
        TestClient(dormant_app).get("/.well-known/agent-card.json").json()
    )
    active = TestClient(active_app).get("/.well-known/agent-card.json").json()

    assert active_app.state.interaction_service.configuration is configuration
    assert "extensions" not in dormant["capabilities"]
    assert [
        extension["uri"] for extension in active["capabilities"]["extensions"]
    ] == [A2A_INPUT_EXTENSION_URI]


@pytest.mark.anyio
@pytest.mark.parametrize("deactivation", ("rollback", "removed"))
async def test_a2a_input_capability_updates_at_request_time(
    deactivation: str,
) -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    _active_interaction_configuration(app)
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
        input_extension_required=True,
    )
    client = TestClient(app)
    rpc_body = {
        "jsonrpc": "2.0",
        "id": "dynamic-capability",
        "method": "UnknownMethod",
        "params": {},
    }

    try:
        active_card = client.get("/.well-known/agent-card.json").json()
        active_missing = client.post(
            "/a2a",
            headers={"A2A-Version": "1.0"},
            json=rpc_body,
        )
        active_echo = client.post(
            "/a2a",
            headers={
                "A2A-Version": "1.0",
                "A2A-Extensions": A2A_INPUT_EXTENSION_URI,
            },
            json=rpc_body,
        )

        assert active_card["capabilities"]["extensions"][0]["uri"] == (
            A2A_INPUT_EXTENSION_URI
        )
        assert active_missing.status_code == 400
        assert active_echo.headers["A2A-Extensions"] == (
            A2A_INPUT_EXTENSION_URI
        )

        if deactivation == "rollback":
            configure_server_interactions(
                app,
                _interaction_configuration(
                    InteractionPolicy(
                        capability_state=TaskInputCapabilityState.ROLLBACK
                    )
                ),
            )
        else:
            configure_server_interactions(app, None)

        inactive_card = client.get("/.well-known/agent-card.json").json()
        inactive_missing = client.post(
            "/a2a",
            headers={"A2A-Version": "1.0"},
            json=rpc_body,
        )
        inactive_echo = client.post(
            "/a2a",
            headers={
                "A2A-Version": "1.0",
                "A2A-Extensions": A2A_INPUT_EXTENSION_URI,
            },
            json=rpc_body,
        )

        assert "extensions" not in inactive_card["capabilities"]
        assert inactive_missing.status_code == 200
        assert inactive_missing.json()["error"]["code"] == -32601
        assert "A2A-Extensions" not in inactive_echo.headers
    finally:
        await close_server_interactions(app)


@pytest.mark.anyio
async def test_rollback_keeps_existing_a2a_resolution_enabled() -> None:
    app = FastAPI()
    configuration = _interaction_configuration(
        InteractionPolicy(capability_state=TaskInputCapabilityState.ROLLBACK)
    )
    actor = InteractionActor(principal=PrincipalScope(user_id=UserId("owner")))
    principal_resolver = cast(AsyncMock, configuration.principal_resolver)
    principal_resolver.return_value = actor
    configure_server_interactions(app, configuration)
    executor = AvalanA2AAgentExecutor(app)
    request = object()
    context = SimpleNamespace(
        requested_extensions=(A2A_INPUT_EXTENSION_URI,),
        state={a2a_router._A2A_HTTP_REQUEST_STATE_KEY: request},
    )

    try:
        assert await executor._activated_actor(context) is actor
        principal_resolver.assert_awaited_once_with(request)
    finally:
        await close_server_interactions(app)


@pytest.mark.anyio
async def test_inactive_agent_card_preserves_unrelated_extensions() -> None:
    routes = a2a_router._agent_card_routes(
        interaction_app=FastAPI(),
        agent_card=SimpleNamespace(supported_interfaces=[]),
        interface_url="/a2a",
        agent_card_to_dict=lambda _card: {
            "capabilities": {
                "extensions": [
                    {"uri": A2A_INPUT_EXTENSION_URI},
                    {"uri": "urn:example:other"},
                ]
            }
        },
        json_response=lambda payload: payload,
        route_class=lambda **values: SimpleNamespace(**values),
    )

    payload = await routes[0].endpoint(
        SimpleNamespace(base_url="https://agents.example")
    )

    assert payload["capabilities"]["extensions"] == [
        {"uri": "urn:example:other", "required": False}
    ]


@pytest.mark.anyio
async def test_inactive_endpoint_removes_stale_extension_echo() -> None:
    async def endpoint(_request: object) -> JSONResponse:
        return JSONResponse(
            {"ok": True},
            headers={"A2A-Extensions": A2A_INPUT_EXTENSION_URI},
        )

    response = await a2a_router._validated_a2a_endpoint(endpoint)(
        _BodyRequest(b"")
    )

    assert "A2A-Extensions" not in response.headers


def test_reconfigure_preserves_exact_live_interaction_service() -> None:
    app = FastAPI()
    configuration = _active_interaction_configuration(app)
    service = app.state.interaction_service

    configure_server_interactions(app, configuration)

    assert app.state.interaction_service is service


def test_agent_card_rejects_uninstalled_configuration_shape() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    configured_app = FastAPI()
    configuration = _active_interaction_configuration(configured_app)
    impostor_app = FastAPI()
    impostor_app.state.interaction_service = SimpleNamespace(
        configuration=configuration
    )

    install_a2a_routes(
        impostor_app,
        prefix="/a2a",
        name="run",
        description=None,
    )

    card = TestClient(impostor_app).get("/.well-known/agent-card.json").json()
    assert "extensions" not in card["capabilities"]


def test_build_agent_card_keeps_a2a_skills_metadata_separate() -> None:
    card = a2a_router._build_agent_card(
        a2a_pb2=_FakeA2APb2(),
        constants=_FakeConstants(),
        interface_url="/a2a",
        name="run",
        description="Run the test agent.",
    )

    assert [skill.id for skill in card.skills] == ["run"]
    assert all(skill.id != "skills.read" for skill in card.skills)


def _extension_message_projection() -> dict[str, Any]:
    return {
        "metadata": {A2A_INPUT_EXTENSION_URI: {"kind": "request"}},
        "extensions": [A2A_INPUT_EXTENSION_URI],
    }


def _extension_task_projection() -> dict[str, Any]:
    return {
        "metadata": {A2A_INPUT_EXTENSION_URI: {"kind": "request"}},
        "status": {"message": _extension_message_projection()},
        "history": [_extension_message_projection()],
    }


def _assert_extension_projection(
    value: dict[str, Any],
    *,
    activated: bool,
) -> None:
    assert (A2A_INPUT_EXTENSION_URI in value["metadata"]) is activated
    messages = [value["status"]["message"], *value["history"]]
    for message in messages:
        assert (A2A_INPUT_EXTENSION_URI in message["metadata"]) is activated
        assert (A2A_INPUT_EXTENSION_URI in message["extensions"]) is activated


@pytest.mark.anyio
@pytest.mark.parametrize("activated", (False, True))
async def test_request_handler_scrubs_cancel_and_subscribe_projections(
    activated: bool,
) -> None:
    cancel_projection = _extension_task_projection()
    subscribe_projection = _extension_task_projection()

    async def subscribe(
        params: Any,
        context: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        _ = params, context
        yield subscribe_projection

    delegate = SimpleNamespace(
        on_cancel_task=AsyncMock(return_value=cancel_projection),
        on_subscribe_to_task=subscribe,
    )
    executor = AvalanA2AAgentExecutor(FastAPI())
    handler = a2a_router._A2ARequestHandler(delegate, executor)
    context = SimpleNamespace(
        requested_extensions=((A2A_INPUT_EXTENSION_URI,) if activated else ())
    )

    cancel_result = await handler.on_cancel_task(object(), context)
    results = [
        result
        async for result in handler.on_subscribe_to_task(object(), context)
    ]

    assert cancel_result is cancel_projection
    assert results == [subscribe_projection]
    _assert_extension_projection(cancel_projection, activated=activated)
    _assert_extension_projection(subscribe_projection, activated=activated)


def test_a2a_route_rejects_invalid_raw_base64_before_sdk_parse() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    client = TestClient(app)

    response = client.post(
        "/a2a",
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": "bad-raw",
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": "message-1",
                    "role": "ROLE_USER",
                    "parts": [
                        {
                            "raw": "not base64!",
                            "filename": "bad.bin",
                            "mediaType": "application/octet-stream",
                        }
                    ],
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "bad-raw"
    assert body["error"]["code"] == -32602
    assert body["error"]["message"] == "Invalid params"
    assert body["error"]["data"] == "A2A raw file parts must be base64 strings"


def test_a2a_tenant_jsonrpc_route_rejects_invalid_raw_base64() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    client = TestClient(app)

    response = client.post(
        "/tenant-a/a2a",
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": "tenant-bad-raw",
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": "message-1",
                    "role": "ROLE_USER",
                    "parts": [
                        {
                            "raw": "%%%%",
                            "filename": "bad.bin",
                            "mediaType": "application/octet-stream",
                        }
                    ],
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "tenant-bad-raw"
    assert body["error"]["code"] == -32602
    assert body["error"]["data"] == "A2A raw file parts must be base64 strings"


def test_a2a_jsonrpc_route_rejects_empty_part_before_sdk_parse() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    client = TestClient(app)

    response = client.post(
        "/a2a",
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": "empty-part",
            "method": "SendMessage",
            "params": {
                "message": {
                    "messageId": "message-1",
                    "role": "ROLE_USER",
                    "parts": [{}],
                }
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "empty-part"
    assert body["error"]["code"] == -32602
    assert (
        body["error"]["data"]
        == "A2A parts must contain exactly one content field"
    )


def test_a2a_tenant_rest_route_rejects_invalid_raw_base64() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    client = TestClient(app)

    response = client.post(
        "/tenant-a/a2a/message:send",
        headers={"A2A-Version": "1.0"},
        json={
            "message": {
                "messageId": "message-1",
                "role": "ROLE_USER",
                "parts": [
                    {
                        "raw": "%%%%",
                        "filename": "bad.bin",
                        "mediaType": "application/octet-stream",
                    }
                ],
            }
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "A2A raw file parts must be base64 strings"
    )


def test_agent_card_uses_v1_supported_interfaces() -> None:
    pytest.importorskip("a2a", reason="a2a-sdk is optional locally")
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    client = TestClient(app, base_url="https://agents.example")

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert "url" not in card
    assert card["name"] == "run"
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"] == [
        {
            "url": "https://agents.example/a2a",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
    assert card["skills"][0]["id"] == "run"


def test_agent_card_advertises_text_and_file_modes_without_sdk() -> None:
    card = a2a_router._build_agent_card(
        a2a_pb2=_FakeA2APb2(),
        constants=_FakeConstants(),
        interface_url="/a2a",
        name="run",
        description=None,
    )

    expected = {
        "text/plain",
        "image/png",
        "image/jpeg",
        "application/json",
        "application/pdf",
        "application/octet-stream",
    }

    assert expected <= set(card.default_input_modes)
    assert expected <= set(card.skills[0].input_modes)
    assert card.default_output_modes == ["text/plain"]
    assert card.skills[0].output_modes == ["text/plain"]


def test_typing_override_compat_installs_missing_override(monkeypatch) -> None:
    typing_module = SimpleNamespace()
    override = object()

    def fake_import(name: str):
        if name == "typing":
            return typing_module
        if name == "typing_extensions":
            return SimpleNamespace(override=override)
        raise AssertionError(name)

    monkeypatch.setattr(a2a_router, "import_module", fake_import)

    a2a_router._ensure_typing_override()

    assert typing_module.override is override


def test_install_a2a_routes_reports_missing_sdk(monkeypatch) -> None:
    def fail_import(name: str):
        if name == "a2a.types.a2a_pb2":
            raise ImportError("missing")
        return __import__(name, fromlist=["_"])

    monkeypatch.setattr(a2a_router, "import_module", fail_import)

    with pytest.raises(ImportError, match="A2A router requires"):
        install_a2a_routes(
            FastAPI(),
            prefix="/a2a",
            name="run",
            description=None,
        )


def test_a2a_extension_error_envelopes_are_exact() -> None:
    response_helpers = pytest.importorskip(
        "a2a.server.request_handlers.response_helpers"
    )
    cases = (
        (
            a2a_router._a2a_unavailable(),
            -31910,
            503,
            {
                "code": "avalan.input.unavailable",
                "interaction_state": "unavailable",
            },
        ),
        (
            a2a_router._a2a_contract_error(
                InputContractError(
                    InputErrorCode.SUPERSEDED,
                    "interaction",
                    "input request was superseded",
                )
            ),
            -31913,
            409,
            {
                "code": "avalan.input.superseded",
                "interaction_state": "superseded",
            },
        ),
    )

    for error, code, status, data in cases:
        response = JSONResponse(
            response_helpers.build_error_response("rpc-1", error)
        )

        assert a2a_router._normalize_a2a_error_response(response) == status
        assert loads(bytes(response.body)) == {
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "error": {
                "code": code,
                "message": "Structured input contract result.",
                "data": data,
            },
        }


def test_a2a_resolution_rejects_unknown_and_missing_answer_keys() -> None:
    request = A2AInputRequestMetadata(
        request_id=InputRequestId("request-1"),
        required=True,
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("required"),
                prompt="Continue?",
                required=True,
            ),
            TextQuestion(
                question_id=QuestionId("optional"),
                prompt="Add a note.",
                required=False,
            ),
        ),
    )
    metadata = {
        "kind": "resolution",
        "request_id": "request-1",
        "action": "accept",
        "answers": {"required": True},
    }

    with pytest.raises(
        InputContractError,
        match="answer keys must include every pending question",
    ):
        decode_a2a_input_resolution_metadata(
            metadata,
            request=request,
            resolved_at=datetime(2026, 7, 24, tzinfo=UTC),
        )

    metadata["answers"] = {
        "required": True,
        "optional": "note",
        "unknown": "value",
    }
    with pytest.raises(
        InputContractError,
        match="answer keys must reference pending questions",
    ):
        decode_a2a_input_resolution_metadata(
            metadata,
            request=request,
            resolved_at=datetime(2026, 7, 24, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "question_type",
    (TextQuestion, MultilineTextQuestion),
)
def test_a2a_resolution_allows_empty_optional_text_only(
    question_type: type[TextQuestion] | type[MultilineTextQuestion],
) -> None:
    optional = question_type(
        question_id=QuestionId("text"),
        prompt="Optional text.",
        required=False,
    )
    metadata = {
        "kind": "resolution",
        "request_id": "request-1",
        "action": "accept",
        "answers": {"text": ""},
    }

    resolution = decode_a2a_input_resolution_metadata(
        metadata,
        request=A2AInputRequestMetadata(
            request_id=InputRequestId("request-1"),
            required=False,
            questions=(optional,),
        ),
        resolved_at=datetime(2026, 7, 24, tzinfo=UTC),
    )

    assert isinstance(resolution, AnsweredResolution)
    assert isinstance(resolution.answers[0], TextAnswer | MultilineTextAnswer)
    assert resolution.answers[0].value == ""

    required = question_type(
        question_id=QuestionId("text"),
        prompt="Required text.",
        required=True,
    )
    with pytest.raises(InputContractError, match="non-empty string"):
        decode_a2a_input_resolution_metadata(
            metadata,
            request=A2AInputRequestMetadata(
                request_id=InputRequestId("request-1"),
                required=True,
                questions=(required,),
            ),
            resolved_at=datetime(2026, 7, 24, tzinfo=UTC),
        )


@pytest.mark.anyio
async def test_chat_request_preserves_text_only_a2a_parts() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(text="hello"),
                _FakePart(text="world"),
            ]
        )
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "hello\nworld"
    assert request.tools is None
    assert request.tool_choice is None


@pytest.mark.anyio
async def test_chat_request_rejects_a2a_runtime_authority_metadata() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(
                    text="hello",
                    metadata={
                        "runtime": {
                            "container": {
                                "image": "registry.example/untrusted:latest"
                            }
                        }
                    },
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="runtime authority"):
        await executor._chat_request(context, _ExecutorOrchestrator())


@pytest.mark.anyio
async def test_chat_request_rejects_a2a_skills_authority_metadata() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(
                    text="hello",
                    metadata={
                        "skills": {
                            "sources": [
                                {"root_path": "/Users/me/.codex/skills"}
                            ]
                        }
                    },
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="runtime authority"):
        await executor._chat_request(context, _ExecutorOrchestrator())


@pytest.mark.anyio
async def test_chat_request_rejects_a2a_file_runtime_authority() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                {
                    "file": {
                        "data": "YWJj",
                        "filename": "raw.bin",
                        "mounts": ["/"],
                    }
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="runtime authority"):
        await executor._chat_request(context, _ExecutorOrchestrator())


@pytest.mark.anyio
async def test_chat_request_rejects_a2a_nested_content_authority() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                {"raw": {"base64": "YWJj", "mounts": ["/"]}},
                {
                    "file": {
                        "data": {
                            "base64": "YWJj",
                            "privileged": True,
                        }
                    }
                },
            ]
        )
    )

    with pytest.raises(ValueError, match="runtime authority"):
        await executor._chat_request(context, _ExecutorOrchestrator())


@pytest.mark.anyio
async def test_chat_request_rejects_unexposed_a2a_container_profile() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(
                    text="hello",
                    metadata={"container": {"profile": "workspace-readonly"}},
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="not exposed"):
        await executor._chat_request(context, _ExecutorOrchestrator())


@pytest.mark.anyio
async def test_chat_request_allows_exposed_a2a_container_profile() -> None:
    app = FastAPI()
    app.state.remote_container_policy = RemoteContainerRequestPolicy(
        exposed_profiles=("workspace-readonly",)
    )
    executor = AvalanA2AAgentExecutor(app)
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(
                    text="hello",
                    metadata={"container": {"profile": "workspace-readonly"}},
                )
            ]
        )
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "hello"


@pytest.mark.anyio
async def test_chat_request_builds_multimodal_content_from_a2a_parts() -> None:
    raw_text = b64encode(b"hello").decode("ascii")
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(text="summarize these"),
                _FakePart(
                    raw=b"%PDF-1.7",
                    filename="report.pdf",
                    mediaType="application/pdf",
                ),
                _FakePart(
                    raw=raw_text,
                    metadata={
                        "filename": "note.txt",
                        "media_type": "text/plain",
                    },
                ),
                _FakePart(
                    raw=b"\x89PNG\r\n\x1a\n",
                    filename="inline.png",
                    mediaType="image/png",
                ),
                _FakePart(
                    url="https://files.example/image.png",
                    filename="image.png",
                    media_type="image/png",
                ),
                _FakePart(data={"kind": "metadata", "page": 1}),
            ]
        )
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())
    content = request.messages[0].content

    assert isinstance(content, list)
    assert isinstance(content[0], ContentText)
    assert content[0].text == "summarize these"
    assert isinstance(content[1], ContentFile)
    assert content[1].file_data == b64encode(b"%PDF-1.7").decode("ascii")
    assert content[1].filename == "report.pdf"
    assert content[1].file == {
        "filename": "report.pdf",
        "mime_type": "application/pdf",
    }
    assert isinstance(content[2], ContentFile)
    assert content[2].file_data == raw_text
    assert content[2].file == {
        "filename": "note.txt",
        "mime_type": "text/plain",
    }
    assert isinstance(content[3], ContentImage)
    assert content[3].image_url == {
        "url": "data:image/png;base64,iVBORw0KGgo="
    }
    assert isinstance(content[4], ContentImage)
    assert content[4].image_url == {"url": "https://files.example/image.png"}
    assert isinstance(content[5], ContentText)
    assert content[5].text == '{"kind":"metadata","page":1}'


@pytest.mark.anyio
async def test_chat_request_accepts_nested_a2a_file_payloads() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                {"file": {"data": "YWJj", "filename": "raw.bin"}},
                {"file": {"data": {"base64": "ZA=="}}},
                {"file": {"url": "mcp://resources/1"}},
            ]
        )
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())
    content = request.messages[0].content

    assert isinstance(content, list)
    assert content[0].file_data == "YWJj"
    assert content[0].file == {"filename": "raw.bin"}
    assert content[1].file_data == "ZA=="
    assert content[2].file_url == "mcp://resources/1"


@pytest.mark.anyio
async def test_chat_request_drops_a2a_file_local_path_metadata() -> None:
    a2a_pb2 = pytest.importorskip("a2a.types.a2a_pb2")
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                a2a_pb2.Part(
                    raw=b"%PDF-1.7",
                    filename="report.pdf",
                    media_type="application/pdf",
                    metadata={"local_path": "/workspace/report.pdf"},
                )
            ]
        )
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())
    content = request.messages[0].content

    assert isinstance(content, list)
    assert isinstance(content[0], ContentFile)
    assert content[0].file == {
        "filename": "report.pdf",
        "mime_type": "application/pdf",
    }


@pytest.mark.anyio
async def test_chat_request_uses_current_task_history() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=None,
        current_task=SimpleNamespace(
            history=[
                _FakeMessage([_FakePart(text="old")], role="agent"),
                _FakeMessage(
                    [
                        _FakePart(text="latest"),
                        _FakePart(
                            url="https://files.example/report.pdf",
                            filename="report.pdf",
                            media_type="application/pdf",
                        ),
                    ],
                    role="user",
                ),
            ]
        ),
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())
    content = request.messages[0].content

    assert isinstance(content, list)
    assert content[0].text == "latest"
    assert content[1].file_url == "https://files.example/report.pdf"
    assert content[1].file == {
        "filename": "report.pdf",
        "mime_type": "application/pdf",
    }


@pytest.mark.anyio
async def test_chat_request_uses_numeric_a2a_user_role_history() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=None,
        current_task=SimpleNamespace(
            history=[
                _FakeMessage([_FakePart(text="numeric-user")], role=1),
                _FakeMessage([_FakePart(text="numeric-agent")], role=2),
            ]
        ),
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "numeric-user"


@pytest.mark.anyio
async def test_chat_request_uses_status_message_before_history() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=None,
        current_task=SimpleNamespace(
            status=SimpleNamespace(
                message=_FakeMessage([_FakePart(text="status")])
            ),
            history=[_FakeMessage([_FakePart(text="history")])],
        ),
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "status"


@pytest.mark.anyio
async def test_chat_request_uses_non_user_history_when_needed() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=None,
        current_task=SimpleNamespace(
            history=[
                _FakeMessage([_FakePart(text="older")], role="agent"),
                _FakeMessage([_FakePart(text="newer")], role="agent"),
            ]
        ),
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "newer"


@pytest.mark.anyio
async def test_chat_request_ignores_invalid_file_part_and_falls_back() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage([_FakePart(raw=object())]),
        user_input="fallback",
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "fallback"


@pytest.mark.anyio
async def test_chat_request_rejects_invalid_a2a_oneof_and_raw_base64() -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    context = _ExecutorContext(
        message=_FakeMessage(
            [
                _FakePart(raw="not base64!"),
                _FakePart(raw="YWJj", url="https://files.example/a.txt"),
                _FakePart(text="hello", data={"ignored": True}),
                {"file": {"data": "not base64!"}},
                {"file": {"data": "YWJj", "url": "mcp://resources/1"}},
            ]
        ),
        user_input="fallback",
    )

    request = await executor._chat_request(context, _ExecutorOrchestrator())

    assert request.messages[0].content == "fallback"


def test_a2a_helper_edge_cases(monkeypatch) -> None:
    route_without_endpoint = SimpleNamespace()
    message_without_sequence = SimpleNamespace(parts="not-parts")
    message_without_role = SimpleNamespace()
    enum_role_message = SimpleNamespace(role=SimpleNamespace(name="ROLE_USER"))
    value_role_message = SimpleNamespace(role=SimpleNamespace(value=1))
    object_role_message = SimpleNamespace(role=SimpleNamespace())

    assert (
        a2a_router._validated_a2a_route(
            route_without_endpoint, route_class=object
        )
        is route_without_endpoint
    )
    assert a2a_router._a2a_message_parts(None) == []
    assert a2a_router._a2a_message_parts(message_without_sequence) == []
    assert a2a_router._is_user_a2a_message(message_without_role) is True
    assert a2a_router._is_user_a2a_message(enum_role_message) is True
    assert a2a_router._is_user_a2a_message(value_role_message) is True
    assert a2a_router._is_user_a2a_message(object_role_message) is False
    assert a2a_router._role_value_is_user(True) is None
    assert a2a_router._raw_file_data(bytearray(b"abc")) == "YWJj"
    assert a2a_router._raw_file_data(memoryview(b"abc")) == "YWJj"
    assert a2a_router._content_from_a2a_part({"text": 7}) is None
    assert a2a_router._content_from_a2a_part({"raw": " "}) is None
    assert a2a_router._content_from_a2a_part({"url": []}) is None
    assert a2a_router._content_from_a2a_part({"data": object()}) is None
    assert a2a_router._file_metadata(object()) == {}
    assert (
        a2a_router._first_string(
            (SimpleNamespace(metadata=_ModelDumpMode()),), "value"
        )
        == "mode"
    )
    assert a2a_router._data_part_text(None) is None
    assert a2a_router._data_part_text(object()) is None
    assert a2a_router._field_value(None, "value") is a2a_router._MISSING
    assert (
        a2a_router._field_value({"other": "value"}, "value")
        is a2a_router._MISSING
    )

    self_raw = _SelfRaw()
    assert a2a_router._raw_file_data(self_raw) is None
    assert (
        a2a_router._field_value(_CallableField(), "value")
        is a2a_router._MISSING
    )
    assert (
        a2a_router._field_value(_HasFieldFalse(), "value")
        is a2a_router._MISSING
    )
    assert a2a_router._field_value(_HasFieldRaises(), "value") == "kept"
    assert (
        a2a_router._a2a_context_message(
            SimpleNamespace(message=_FakeMessage([]))
        )
        is not None
    )

    dumped = a2a_router._data_part_text(_ModelDumpFallback())
    assert dumped == '{"value":"fallback"}'
    dumped_with_mode = a2a_router._data_part_text(_ModelDumpMode())
    assert dumped_with_mode == '{"value":"mode"}'
    assert a2a_router._data_part_text(["a", object(), 1]) == '["a",1]'

    fake_json_format = SimpleNamespace(
        MessageToDict=lambda value: {"from": "protobuf"}
    )
    real_import_module = a2a_router.import_module

    def fake_import_module(name: str):
        if name == "google.protobuf.json_format":
            return fake_json_format
        return real_import_module(name)

    monkeypatch.setattr(a2a_router, "import_module", fake_import_module)

    assert a2a_router._data_part_text(_ProtoLike()) == '{"from":"protobuf"}'


@pytest.mark.anyio
async def test_a2a_json_file_part_validator_edge_cases() -> None:
    async def endpoint(request: object) -> str:
        return "ok"

    wrapped = a2a_router._validated_a2a_endpoint(endpoint)
    wrapped_jsonrpc = a2a_router._validated_a2a_endpoint(
        endpoint, jsonrpc=True
    )
    tenant_request = _BodyRequest(
        b'{"params":{"message":{"parts":[{"text":"ok"}]}}}',
        path_params={"tenant": "tenant-a"},
    )

    assert await wrapped(_BodyRequest(b"")) == "ok"
    assert await wrapped(_BodyRequest(b"{invalid")) == "ok"
    assert await wrapped_jsonrpc(tenant_request) == "ok"
    assert loads(await tenant_request.body())["params"]["tenant"] == "tenant-a"
    invalid_json_response = (
        await a2a_router._a2a_jsonrpc_validation_error_response(
            _BodyRequest(b"{invalid"), "bad"
        )
    )
    assert loads(invalid_json_response.body)["id"] is None
    assert a2a_router._a2a_jsonrpc_request_id([]) is None
    assert a2a_router._a2a_jsonrpc_request_id({"id": True}) is None

    request = _BodyRequest(b"{}")
    a2a_router._inject_a2a_jsonrpc_tenant(request, None)
    a2a_router._inject_a2a_jsonrpc_tenant(request, [])
    request.path_params = []
    a2a_router._inject_a2a_jsonrpc_tenant(request, {})
    request.path_params = {}
    a2a_router._inject_a2a_jsonrpc_tenant(request, {"params": {}})
    request.path_params = {"tenant": ""}
    a2a_router._inject_a2a_jsonrpc_tenant(request, {"params": {}})
    request.path_params = {"tenant": "tenant-a"}
    a2a_router._inject_a2a_jsonrpc_tenant(request, {"params": []})
    assert a2a_router._a2a_json_part_payloads({"text": "already-part"}) == []
    assert a2a_router._a2a_json_part_payloads(
        [{"parts": [{"root": {"text": "ok"}}]}]
    ) == [{"text": "ok"}]
    assert a2a_router._a2a_json_part_payloads(
        [{"parts": [{"data": {"parts": [{"raw": "not base64!"}]}}]}]
    ) == [{"data": {"parts": [{"raw": "not base64!"}]}}]
    a2a_router._validate_a2a_json_part_payload({"raw": "-_8"})
    a2a_router._validate_a2a_json_part_payload({"raw": "YWJjZA"})
    a2a_router._validate_a2a_json_part_payload({"raw": "YWJj\nZA=="})
    a2a_router._validate_a2a_json_part_payload(
        {"file": {"data": "YWJj", "filename": "raw.bin"}}
    )
    a2a_router._validate_a2a_json_part_payload(
        {"file": {"data": {"base64": "YWJj"}, "filename": "raw.bin"}}
    )
    assert a2a_router._raw_file_data("-_8") == "+/8="
    assert a2a_router._raw_file_data("YWJj\nZA==") == "YWJjZA=="

    with pytest.raises(a2a_router.HTTPException):
        a2a_router._validate_a2a_json_part_payload(
            {
                "text": "hello",
                "raw": "aGVsbG8=",
            }
        )
    with pytest.raises(a2a_router.HTTPException):
        a2a_router._validate_a2a_json_part_payload({"raw": None})
    with pytest.raises(a2a_router.HTTPException):
        a2a_router._validate_a2a_json_part_payload(
            {"file": {"data": "not base64!"}}
        )
    with pytest.raises(a2a_router.HTTPException):
        a2a_router._validate_a2a_json_part_payload(
            {"file": {"data": {"base64": "not base64!"}}}
        )
    with pytest.raises(a2a_router.HTTPException):
        a2a_router._validate_a2a_json_part_payload({"metadata": {}})


@pytest.mark.anyio
async def test_a2a_json_validator_allows_nested_file_parts() -> None:
    request = _BodyRequest(b"""{
            "params": {
                "message": {
                    "parts": [
                        {
                            "file": {
                                "data": "YWJj",
                                "filename": "raw.bin"
                            }
                        }
                    ]
                }
            }
        }""")

    payload = await a2a_router._validate_a2a_json_file_parts(request)

    assert isinstance(payload, dict)


@pytest.mark.anyio
async def test_a2a_json_validator_rejects_runtime_authority() -> None:
    request = _BodyRequest(b"""{
            "params": {
                "message": {
                    "parts": [
                        {
                            "text": "hello",
                            "metadata": {
                                "container": {
                                    "image": "registry.example/untrusted"
                                }
                            }
                        }
                    ]
                }
            }
        }""")

    with pytest.raises(a2a_router.HTTPException) as exc_info:
        await a2a_router._validate_a2a_json_file_parts(request)

    assert exc_info.value.status_code == 400
    assert "runtime authority" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_a2a_json_validator_rejects_shell_pipeline_authority() -> None:
    request = _BodyRequest(b"""{
            "params": {
                "message": {
                    "parts": [
                        {
                            "text": "hello",
                            "metadata": {
                                "tool": {
                                    "shell": {
                                        "allow_pipelines": true
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }""")

    with pytest.raises(a2a_router.HTTPException) as exc_info:
        await a2a_router._validate_a2a_json_file_parts(request)

    assert exc_info.value.status_code == 400
    assert "runtime authority" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_a2a_json_validator_rejects_nested_content_authority() -> None:
    request = _BodyRequest(b"""{
            "params": {
                "message": {
                    "parts": [
                        {
                            "raw": {
                                "base64": "YWJj",
                                "mounts": ["/"]
                            }
                        },
                        {
                            "file": {
                                "data": {
                                    "base64": "YWJj",
                                    "privileged": true
                                }
                            }
                        }
                    ]
                }
            }
        }""")

    with pytest.raises(a2a_router.HTTPException) as exc_info:
        await a2a_router._validate_a2a_json_file_parts(request)

    assert exc_info.value.status_code == 400
    assert "runtime authority" in str(exc_info.value.detail)


def test_a2a_part_authority_ignores_non_mapping_part_payload() -> None:
    a2a_router._reject_a2a_remote_runtime_authority(
        7,
        path="a2a.parts[0]",
        part_payload=True,
    )


def test_a2a_part_authority_allows_exposed_profile_selector() -> None:
    a2a_router._reject_a2a_remote_runtime_authority(
        {"containerProfile": "workspace-readonly"},
        path="a2a.parts[0]",
        policy=RemoteContainerRequestPolicy(
            exposed_profiles=("workspace-readonly",)
        ),
        part_payload=True,
    )


def test_a2a_part_authority_rejects_direct_runtime_key() -> None:
    for payload in (
        {"allow_pipelines": True},
        {"allowShell": True},
        {"runtime": "container"},
        {"sandboxProfile": "workspace-readonly"},
        {"shell": {"workspace_root": "/private"}},
        {"isolation": {"mode": "sandbox"}},
    ):
        with pytest.raises(ValueError, match="runtime authority"):
            a2a_router._reject_a2a_remote_runtime_authority(
                payload,
                path="a2a.parts[0]",
                part_payload=True,
            )


def test_a2a_part_authority_rejects_nested_content_wrappers() -> None:
    invalid_parts = (
        {"raw": {"base64": "YWJj", "mounts": ["/"]}},
        {"data": [{"base64": "YWJj"}, {"workdir": "/workspace"}]},
        {
            "file": {
                "data": {
                    "base64": "YWJj",
                    "privileged": True,
                }
            }
        },
        {
            "file": {
                "data": "YWJj",
                "filename": {"container": {"image": "untrusted"}},
            }
        },
    )

    for part in invalid_parts:
        with pytest.raises(ValueError, match="runtime authority"):
            a2a_router._reject_a2a_remote_runtime_authority(
                part,
                path="a2a.parts[0]",
                part_payload=True,
            )


def test_a2a_file_authority_allows_safe_metadata_and_profile() -> None:
    a2a_router._reject_a2a_remote_runtime_authority(
        {
            "file": {
                "data": "YWJj",
                "metadata": {"trace_id": "request-1"},
                "containerProfile": "workspace-readonly",
                "attributes": {"tag": "safe"},
            }
        },
        path="a2a.parts[0]",
        policy=RemoteContainerRequestPolicy(
            exposed_profiles=("workspace-readonly",)
        ),
        part_payload=True,
    )


def test_a2a_file_authority_ignores_non_mapping_file_payload() -> None:
    a2a_router._reject_a2a_remote_runtime_authority(
        {"file": 7},
        path="a2a.parts[0]",
        part_payload=True,
    )


def test_a2a_profile_selector_rejects_non_string_alias_value() -> None:
    with pytest.raises(ValueError, match="runtime authority"):
        a2a_router._reject_a2a_remote_runtime_authority(
            {"containerProfile": {"profile": "workspace-readonly"}},
            path="a2a",
            policy=RemoteContainerRequestPolicy(
                exposed_profiles=("workspace-readonly",)
            ),
        )


@pytest.mark.anyio
async def test_executor_passes_a2a_file_parts_to_orchestrate(
    monkeypatch, fake_a2a_imports
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    captured_requests = []

    async def fake_orchestrate(request, *args: object, **kwargs: object):
        captured_requests.append(request)
        return object(), "response-id", 123

    async def fake_cleanup(*args: object, **kwargs: object) -> None:
        return None

    def fake_stream_consumer_iterator(*args: object, **kwargs: object):
        async def iterator():
            yield _item(
                0,
                StreamItemKind.STREAM_COMPLETED,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        return iterator()

    monkeypatch.setattr(a2a_router, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", fake_cleanup)
    monkeypatch.setattr(
        a2a_router,
        "stream_consumer_iterator",
        fake_stream_consumer_iterator,
    )

    await executor.execute(
        _ExecutorContext(
            message=_FakeMessage(
                [
                    _FakePart(text="read"),
                    _FakePart(
                        raw=b"content",
                        filename="file.bin",
                        media_type="application/octet-stream",
                    ),
                ]
            )
        ),
        _FakeEventQueue(),
    )

    content = captured_requests[0].messages[0].content

    assert isinstance(content, list)
    assert content[0].text == "read"
    assert content[1].file_data == b64encode(b"content").decode("ascii")
    assert content[1].file == {
        "filename": "file.bin",
        "mime_type": "application/octet-stream",
    }


@pytest.mark.anyio
async def test_executor_forwards_ctx_output_redaction_settings(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    settings = ServerOutputRedactionSettings(
        enabled=True,
        protocols=frozenset({"a2a"}),
    )
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    app.state.ctx = OrchestratorContext(
        participant_id=None,
        output_redaction_settings=settings,
    )
    executor = AvalanA2AAgentExecutor(app)
    captured_settings: list[ServerOutputRedactionSettings | None] = []

    class CapturingTranslator:
        succeeded = True

        def __init__(
            self,
            updater: object,
            *,
            output_redaction_settings: (
                ServerOutputRedactionSettings | None
            ) = None,
        ) -> None:
            _ = updater
            captured_settings.append(output_redaction_settings)

        async def process(self, item: object) -> None:
            _ = item

        async def finish(self) -> None:
            return None

    async def fake_orchestrate(*args: object, **kwargs: object):
        return object(), "response-id", 123

    async def fake_cleanup(*args: object, **kwargs: object) -> None:
        return None

    def fake_stream_consumer_iterator(*args: object, **kwargs: object):
        async def iterator():
            yield _item(
                0,
                StreamItemKind.STREAM_COMPLETED,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        return iterator()

    monkeypatch.setattr(a2a_router, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", fake_cleanup)
    monkeypatch.setattr(
        a2a_router,
        "stream_consumer_iterator",
        fake_stream_consumer_iterator,
    )
    monkeypatch.setattr(
        a2a_router,
        "A2AResponseTranslator",
        CapturingTranslator,
    )

    await executor.execute(_ExecutorContext(), _FakeEventQueue())

    assert captured_settings == [settings]


@pytest.mark.anyio
async def test_executor_emits_submitted_task_for_new_a2a_task(
    monkeypatch, fake_a2a_imports
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    event_queue = _FakeEventQueue()

    async def fake_orchestrate(request, *args: object, **kwargs: object):
        return object(), "response-id", 123

    async def fake_cleanup(*args: object, **kwargs: object) -> None:
        return None

    def fake_stream_consumer_iterator(*args: object, **kwargs: object):
        async def iterator():
            yield _item(
                0,
                StreamItemKind.STREAM_COMPLETED,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        return iterator()

    monkeypatch.setattr(a2a_router, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", fake_cleanup)
    monkeypatch.setattr(
        a2a_router,
        "stream_consumer_iterator",
        fake_stream_consumer_iterator,
    )

    await executor.execute(
        _ExecutorContext(current_task=None),
        event_queue,
    )

    assert getattr(event_queue.events[0], "id") == "task-1"


@pytest.mark.anyio
async def test_translator_projects_reasoning_tool_and_terminal_states(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(
        updater,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
    )

    await translator.process(
        _item(
            0,
            StreamItemKind.REASONING_DELTA,
            text_delta="plan",
        )
    )
    await translator.process(
        _tool_item(
            1,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            text_delta="live",
            data={"name": "shell.run"},
        )
    )
    await translator.process(
        _tool_item(
            2,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            data={"name": "shell.run"},
        )
    )
    await translator.process(
        _item(
            3,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    await translator.finish()

    assert translator.succeeded is True
    assert updater.artifacts[0]["artifact_id"] == "reasoning-r-0-0"
    assert (
        next(
            artifact
            for artifact in updater.artifacts
            if artifact["artifact_id"] == "call-1"
        )["artifact_id"]
        == "call-1"
    )
    assert updater.artifacts[-1]["last_chunk"] is True
    assert updater.statuses[0]["metadata"]["tool_name"] == "shell.run"
    assert updater.completed == 1


@pytest.mark.anyio
async def test_translator_projects_skills_tool_activity_safely(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(
        updater,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
    )

    await translator.process(
        _tool_item(
            0,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            data={"name": "skills.read", "arguments": {"skill": "demo"}},
            metadata={"tool_name": "skills.read"},
        )
    )
    await translator.process(
        _tool_item(
            1,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            text_delta="private live skill instructions",
            data={},
            metadata={"tool_name": "skills.read"},
        )
    )
    assert len(updater.artifacts) == 1
    live_artifact_text = "".join(
        getattr(part, "text", "") for part in updater.artifacts[0]["parts"]
    )
    assert updater.artifacts[0]["artifact_id"] == "call-1"
    assert "redacted-skill-content" in live_artifact_text
    assert "private live skill instructions" not in live_artifact_text

    await translator.process(
        _tool_item(
            2,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            data={
                "name": "skills.read",
                "result": {
                    "content": "private skill instructions",
                    "path": "/Users/mariano/.codex/skills/demo/SKILL.md",
                },
            },
            metadata={"tool_name": "skills.read"},
        )
    )
    await translator.process(
        _item(
            3,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    await translator.finish()

    artifact_text = "".join(
        getattr(part, "text", "")
        for artifact in updater.artifacts
        for part in artifact["parts"]
    )
    projected = artifact_text + str(updater.artifacts) + str(updater.statuses)

    assert updater.statuses[0]["metadata"]["tool_name"] == "skills.read"
    assert updater.artifacts[0]["artifact_id"] == "call-1"
    assert "redacted-skill-content" in projected
    assert "<host-path>/SKILL.md" in projected
    assert "private live skill instructions" not in projected
    assert "private skill instructions" not in projected
    assert "/Users/mariano" not in projected
    assert updater.completed == 1


@pytest.mark.anyio
async def test_translator_projects_shell_pipeline_stage_streams_safely(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)

    await translator.process(
        _tool_item(
            0,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            data={
                "category": "progress",
                "content": "stage read started",
                "progress": 0.25,
                "metadata": {
                    "private_runtime": "SECRET_RUNTIME",
                    "intermediate_stdout": (
                        "INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK"
                    ),
                },
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            1,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            text_delta="stage warning\n",
            data={
                "category": "stderr",
                "content": "stage warning\n",
                "metadata": {"private_path": "/secret/root"},
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            2,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            text_delta="2\n",
            data={"category": "stdout", "content": "2\n"},
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            3,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            data={
                "name": "shell.pipeline",
                "result": (
                    "tool: shell.pipeline\nstatus: completed\nstdout:\n2\n"
                ),
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _item(
            4,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    await translator.finish()

    artifact_text = "".join(
        part.text
        for artifact in updater.artifacts
        for part in artifact["parts"]
        if hasattr(part, "text")
    )
    projected = str(updater.artifacts) + str(updater.statuses)

    assert "stage read started" in artifact_text
    assert "stage warning\n" in artifact_text
    assert "2\n" in artifact_text
    assert "SECRET_RUNTIME" not in projected
    assert "/secret/root" not in projected
    assert "INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK" not in projected
    assert updater.artifacts[0]["metadata"]["category"] == "progress"
    assert updater.artifacts[1]["metadata"]["category"] == "stderr"
    assert updater.artifacts[2]["metadata"]["category"] == "stdout"


@pytest.mark.anyio
async def test_translator_projects_shell_pipeline_diagnostic_safely(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)

    await translator.process(
        _tool_item(
            0,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            data={"name": "shell.pipeline"},
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            1,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            data={
                "name": "shell.pipeline",
                "diagnostic": {
                    "code": "tool.disabled",
                    "message": "shell.pipeline requires allow_pipelines=true.",
                    "details": {"workspace_root": "/secret/root"},
                },
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _item(
            2,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    await translator.finish()

    artifact_text = "".join(
        part.text
        for artifact in updater.artifacts
        for part in artifact["parts"]
        if hasattr(part, "text")
    )

    assert "tool.disabled" in artifact_text
    assert "allow_pipelines" in artifact_text
    assert "/secret/root" not in artifact_text


@pytest.mark.anyio
async def test_translator_projects_tool_item_fallback_text_branches(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)

    await translator.process(
        _tool_item(
            0,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            text_delta="",
            data={"name": "shell.pipeline", "content": "stdout from data"},
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            1,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            data={"name": "shell.pipeline", "progress": 0.75},
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            2,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            data={
                "name": "shell.pipeline",
                "result": {"ok": True, "count": 2},
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _tool_item(
            3,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            data={
                "name": "shell.pipeline",
                "diagnostic": {"details": {"private": "hidden"}},
            },
            metadata={"tool_name": "shell.pipeline"},
        )
    )
    await translator.process(
        _item(
            4,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    await translator.finish()

    artifact_text = "".join(
        part.text
        for artifact in updater.artifacts
        for part in artifact["parts"]
        if hasattr(part, "text")
    )

    assert "stdout from data" in artifact_text
    assert '{"progress":0.75}' in artifact_text
    assert '{"ok":true,"count":2}' in artifact_text
    assert "hidden" not in artifact_text


@pytest.mark.anyio
async def test_translator_projects_answer_delta(fake_a2a_imports) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)

    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
    )

    assert updater.artifacts[0]["artifact_id"] == "answer"
    assert updater.artifacts[0]["parts"][0].text == "answer"


def test_a2a_tool_text_projection_edge_cases() -> None:
    tool_output = CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=0,
        kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
        channel=StreamChannel.TOOL_EXECUTION,
        correlation=StreamItemCorrelation(tool_call_id="call-1"),
        text_delta="",
        data={"content": {"unexpected": "shape"}},
    )
    tool_progress = CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=1,
        kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
        channel=StreamChannel.TOOL_EXECUTION,
        correlation=StreamItemCorrelation(tool_call_id="call-1"),
        data={},
    )
    skills_output = CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=2,
        kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
        channel=StreamChannel.TOOL_EXECUTION,
        correlation=StreamItemCorrelation(tool_call_id="call-1"),
        text_delta="",
        data={"content": "private skill body"},
    )
    skills_delta = CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=3,
        kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
        channel=StreamChannel.TOOL_EXECUTION,
        correlation=StreamItemCorrelation(tool_call_id="call-1"),
        text_delta="private skill body",
        data={},
    )

    assert a2a_router._a2a_tool_item_text(tool_output, tool_output.data) == ""
    assert (
        a2a_router._a2a_tool_item_text(tool_progress, tool_progress.data) == ""
    )
    assert loads(
        a2a_router._a2a_tool_item_text(
            skills_output,
            skills_output.data,
            tool_name="skills.read",
            output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
        )
    ) == {
        "content": {
            "redacted": True,
            "reason": "<redacted-skill-content>",
        }
    }
    assert loads(
        a2a_router._a2a_tool_item_text(
            skills_delta,
            skills_delta.data,
            tool_name="skills.read",
            output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
        )
    ) == {
        "content": {
            "redacted": True,
            "reason": "<redacted-skill-content>",
        }
    }
    assert (
        a2a_router._a2a_protocol_payload_text(
            "Source: /tmp/skills/demo/SKILL.md",
            tool_name=None,
            output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
        )
        == "Source: <host-path>/SKILL.md"
    )
    assert (
        a2a_router._a2a_tool_item_text(
            skills_output,
            skills_output.data,
            tool_name="skills.read",
        )
        == "private skill body"
    )
    assert (
        a2a_router._a2a_tool_item_text(
            skills_delta,
            skills_delta.data,
            tool_name="skills.read",
        )
        == "private skill body"
    )
    assert (
        a2a_router._a2a_tool_item_text(
            skills_output,
            skills_output.data,
            tool_name="skills.read",
            output_redaction_settings=ServerOutputRedactionSettings(
                enabled=True,
                protocols=frozenset({"mcp"}),
            ),
        )
        == "private skill body"
    )
    assert (
        a2a_router._a2a_tool_item_text(
            skills_delta,
            skills_delta.data,
            tool_name="skills.read",
            output_redaction_settings=ServerOutputRedactionSettings(
                enabled=True,
                protocols=frozenset({"mcp"}),
            ),
        )
        == "private skill body"
    )


@pytest.mark.anyio
async def test_translator_redacts_answer_and_reasoning_skill_echoes(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(
        updater,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
    )

    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="#",
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=" Demo Skill\n\n",
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="Use when answering private operator tasks.\n\n",
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=(
                "Secret answer skill body.\nSource: /tmp/skills/demo/SKILL.md"
            ),
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="#",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta=" Reasoning Skill\n\n",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=6,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="Instructions: keep this skill body hidden.\n\n",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=7,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta=(
                "Secret reasoning skill body.\n"
                "Source: C:/Users/me/skills/demo/SCOPE.md"
            ),
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )

    artifact_text = "".join(
        getattr(part, "text", "")
        for artifact in updater.artifacts
        for part in artifact["parts"]
    )

    assert artifact_text.count("redacted-skill-content") == 2
    assert "# Demo Skill" not in artifact_text
    assert "Use when answering private" not in artifact_text
    assert "# Reasoning Skill" not in artifact_text
    assert "Instructions: keep this skill body hidden" not in artifact_text
    assert "Secret answer skill body" not in artifact_text
    assert "Secret reasoning skill body" not in artifact_text
    assert "/tmp/skills" not in artifact_text
    assert "C:/Users" not in artifact_text


@pytest.mark.anyio
async def test_translator_respects_model_text_channel_filters(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(
        updater,
        output_redaction_settings=ServerOutputRedactionSettings(
            enabled=True,
            channels=frozenset({"reasoning"}),
        ),
    )
    answer_echo = (
        "# Demo Skill\n\n"
        "Use when answering private operator tasks.\n"
        "Secret answer skill body."
    )
    reasoning_echo = (
        "# Reasoning Skill\n\n"
        "Use when reasoning privately.\n"
        "Secret reasoning skill body."
    )

    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=answer_echo,
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta=reasoning_echo,
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )
    await translator.finish()

    artifact_text = "".join(
        getattr(part, "text", "")
        for artifact in updater.artifacts
        for part in artifact["parts"]
    )

    assert "# Demo Skill" in artifact_text
    assert "Secret answer skill body" in artifact_text
    assert "redacted-skill-content" in artifact_text
    assert "Secret reasoning skill body" not in artifact_text


@pytest.mark.anyio
async def test_translator_leaves_answer_skill_echoes_by_default(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)
    skill_echo = (
        "# Demo Skill\n\n"
        "Use when answering private operator tasks.\n"
        "Secret answer skill body.\n"
        "Source: /tmp/skills/demo/SKILL.md"
    )

    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=skill_echo,
        )
    )

    artifact_text = "".join(
        getattr(part, "text", "")
        for artifact in updater.artifacts
        for part in artifact["parts"]
    )

    assert artifact_text == skill_echo


@pytest.mark.anyio
async def test_translator_flushes_buffered_model_text_in_source_order(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(
        updater,
        output_redaction_settings=_MODEL_VISIBLE_REDACTION_SETTINGS,
    )

    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="# Imagegen\n",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
    )
    await translator.process(
        CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="# Browser\n",
        )
    )
    await translator.finish()

    text_artifacts = [
        artifact for artifact in updater.artifacts if artifact["parts"]
    ]

    assert [artifact["artifact_id"] for artifact in text_artifacts] == [
        "reasoning-r-0-0",
        "answer",
    ]
    assert text_artifacts[0]["parts"][0].text == "<redacted-skill-content>"
    assert text_artifacts[1]["parts"][0].text == "# Browser\n"


@pytest.mark.anyio
async def test_translator_handles_projection_cancel_error_and_bad_items(
    fake_a2a_imports,
) -> None:
    cancelled = A2AResponseTranslator(_FakeUpdater())
    await cancelled.process(
        StreamConsumerProjection(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_CANCELLED,
            channel=StreamChannel.CONTROL,
            correlation=StreamItemCorrelation(),
            terminal_outcome=StreamTerminalOutcome.CANCELLED,
        )
    )
    await cancelled.finish()

    errored_updater = _FakeUpdater()
    errored = A2AResponseTranslator(errored_updater)
    await errored.process(
        _item(
            0,
            StreamItemKind.STREAM_ERRORED,
            terminal_outcome=StreamTerminalOutcome.ERRORED,
        )
    )
    await errored.finish()

    bad = A2AResponseTranslator(_FakeUpdater())
    with pytest.raises(Exception, match="unsupported A2A stream item"):
        await bad.process(object())

    assert cancelled.succeeded is False
    assert errored_updater.failed_count == 1


@pytest.mark.parametrize(
    "outcome",
    (
        StreamTerminalOutcome.CANCELLED,
        StreamTerminalOutcome.ERRORED,
    ),
)
@pytest.mark.anyio
async def test_translator_abort_converges_after_finish_failure(
    monkeypatch,
    fake_a2a_imports,
    outcome: StreamTerminalOutcome,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)
    monkeypatch.setattr(
        translator,
        "_flush_model_text",
        AsyncMock(side_effect=RuntimeError("finish failed")),
    )

    with pytest.raises(RuntimeError, match="finish failed"):
        await translator.finish()
    await translator.abort(outcome)
    await translator.abort(outcome)

    assert updater.cancelled == int(outcome is StreamTerminalOutcome.CANCELLED)
    assert updater.failed_count == int(
        outcome is StreamTerminalOutcome.ERRORED
    )


@pytest.mark.anyio
async def test_translator_leaves_input_required_task_nonterminal(
    fake_a2a_imports,
) -> None:
    assert set(StreamTerminalOutcome) == {
        StreamTerminalOutcome.COMPLETED,
        StreamTerminalOutcome.ERRORED,
        StreamTerminalOutcome.CANCELLED,
        StreamTerminalOutcome.INPUT_REQUIRED,
    }
    assert {
        outcome: a2a_router._a2a_reasoning_terminal_outcome(outcome)
        for outcome in StreamTerminalOutcome
    } == {
        StreamTerminalOutcome.COMPLETED: "completed",
        StreamTerminalOutcome.ERRORED: "failed",
        StreamTerminalOutcome.CANCELLED: "cancelled",
        StreamTerminalOutcome.INPUT_REQUIRED: "input_required",
    }
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)
    correlation = StreamItemCorrelation(
        request_id="request-1",
        continuation_id="continuation-1",
        agent_id="agent-1",
        branch_id="branch-1",
    )
    await translator.process(
        _item(
            0,
            StreamItemKind.REASONING_DELTA,
            text_delta="plan",
        )
    )
    await translator.process(
        _item(
            1,
            StreamItemKind.STREAM_INPUT_REQUIRED,
            correlation=correlation,
            terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
        )
    )

    await translator.finish()
    await translator.finish()

    assert translator.succeeded is False
    assert updater.completed == 0
    assert updater.cancelled == 0
    assert updater.failed_count == 0
    assert updater.statuses == [
        {"state": "input_required", "metadata": {}},
    ]
    reasoning_close = next(
        artifact
        for artifact in reversed(updater.artifacts)
        if artifact.get("artifact_id") == "reasoning-r-0-0"
        and artifact.get("last_chunk") is True
    )
    metadata = reasoning_close["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["status"] == "incomplete"
    assert metadata["terminal_outcome"] == "input_required"


@pytest.mark.anyio
async def test_translator_rejects_missing_input_required_task_state(
    fake_a2a_imports,
) -> None:
    del fake_a2a_imports.TaskState.TASK_STATE_INPUT_REQUIRED
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)
    await translator.process(
        _item(
            0,
            StreamItemKind.STREAM_INPUT_REQUIRED,
            correlation=StreamItemCorrelation(
                request_id="request-1",
                continuation_id="continuation-1",
                agent_id="agent-1",
                branch_id="branch-1",
            ),
            terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
        )
    )

    with pytest.raises(
        StreamValidationError,
        match="A2A SDK input-required task state is unavailable",
    ):
        await translator.finish()

    assert updater.completed == 0
    assert updater.cancelled == 0
    assert updater.failed_count == 0


@pytest.mark.anyio
async def test_executor_cancel_and_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
    fake_a2a_imports: Any,
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    context = _ExecutorContext()
    event_queue = _FakeEventQueue()

    fake_a2a_imports.TaskState.TASK_STATE_AUTH_REQUIRED = "auth_required"
    auth_queue = _FakeEventQueue()
    await executor.execute(
        _ExecutorContext(
            current_task=SimpleNamespace(
                status=SimpleNamespace(state="auth_required")
            )
        ),
        auth_queue,
    )
    assert cast(Any, auth_queue.events[0])["state"] == "auth_required"

    async def fail_orchestrate(*args: object, **kwargs: object):
        raise RuntimeError("broken")

    monkeypatch.setattr(a2a_router, "orchestrate", fail_orchestrate)

    with pytest.raises(RuntimeError, match="broken"):
        await executor.execute(context, event_queue)
    await executor.cancel(context, event_queue)

    async def cancel_orchestrate(*args: object, **kwargs: object):
        raise CancelledError

    cancellation_queue = _FakeEventQueue()
    monkeypatch.setattr(a2a_router, "orchestrate", cancel_orchestrate)
    with pytest.raises(CancelledError):
        await executor.execute(context, cancellation_queue)

    assert event_queue.events
    assert [event["kind"] for event in cancellation_queue.events] == [
        "status",
        "cancel",
    ]


@pytest.mark.anyio
async def test_a2a_input_handler_waits_for_exact_registration() -> None:
    handler = a2a_router._A2AInputHandler()
    first_request = cast(Any, SimpleNamespace(request_id="request-1"))
    second_request = cast(Any, SimpleNamespace(request_id="request-2"))
    first_task = create_task(
        handler(cast(Any, SimpleNamespace(request=first_request)))
    )

    assert await handler.request("request-1") is first_request
    second_wait = create_task(handler.request("request-2"))
    await sleep(0)
    assert not second_wait.done()
    second_task = create_task(
        handler(cast(Any, SimpleNamespace(request=second_request)))
    )
    assert await second_wait is second_request

    for task in (first_task, second_task):
        task.cancel()
        with pytest.raises(CancelledError):
            await task


@pytest.mark.anyio
async def test_a2a_input_helper_negative_branches(
    monkeypatch: pytest.MonkeyPatch,
    fake_a2a_imports: Any,
) -> None:
    class _ImmediateEvent:
        def set(self) -> None:
            pass

        async def wait(self) -> None:
            pass

    monkeypatch.setattr(a2a_router, "Event", _ImmediateEvent)
    with pytest.raises(AssertionError, match="cannot finish"):
        await a2a_router._A2AInputHandler()(
            cast(
                Any,
                SimpleNamespace(
                    request=SimpleNamespace(request_id="request-completed")
                ),
            )
        )

    handler = a2a_router._A2AInputHandler()
    registered = Event()
    registered.set()
    handler._registered["missing"] = registered
    with pytest.raises(RuntimeError, match="did not register"):
        await handler.request("missing")
    with pytest.raises(RuntimeError, match="did not register"):
        handler.settlement_event("missing")

    task_store = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(context_id="expected"))
    )
    with pytest.raises(Exception, match="Structured input contract result"):
        await a2a_router._A2ARequestContextBuilder(task_store).build(
            object(),
            task_id="task-1",
            context_id="wrong",
        )

    executor = SimpleNamespace(
        prepare_follow_up=AsyncMock(return_value="replay"),
        deactivate_result=MagicMock(),
    )
    delegate = SimpleNamespace(forwarded="value")
    request_handler = a2a_router._A2ARequestHandler(
        delegate,
        cast(Any, executor),
    )
    assert request_handler.forwarded == "value"
    stream = request_handler.on_message_send_stream(object(), object())
    assert [value async for value in stream] == ["replay"]

    response = a2a_router._required_extension_response(None, jsonrpc=False)
    assert response.status_code == 400
    a2a_router._strip_a2a_input_extension(None)
    for state in (None, {"request_id": True}, {"request_id": object()}):
        assert not a2a_router._a2a_transport_ids(SimpleNamespace(state=state))

    for code in (
        InputErrorCode.ALREADY_RESOLVED,
        InputErrorCode.EXPIRED,
        InputErrorCode.UNAVAILABLE,
        InputErrorCode.FORBIDDEN,
    ):
        a2a_router._a2a_contract_error(
            InputContractError(code, "interaction", "failed")
        )
    for response in (
        SimpleNamespace(body=b"{", headers={}),
        JSONResponse(1),
    ):
        assert a2a_router._normalize_a2a_error_response(response) is None
    with pytest.raises(Exception, match="Structured input contract result"):
        a2a_router._a2a_resolution_idempotency_key("task-1", object())


@pytest.mark.anyio
async def test_follow_up_rejects_ambiguous_and_stale_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scenario(
        *,
        metadata: object = None,
        message_id: object = "message-new",
        seen: set[str] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        executor = AvalanA2AAgentExecutor(FastAPI())
        actor = SimpleNamespace(principal="owner")
        pending = SimpleNamespace(
            context_id="context-1",
            actor=actor,
            seen_message_ids=set() if seen is None else seen,
            seen_transport_ids=set(),
            lock=a2a_router.Lock(),
            request=SimpleNamespace(
                request_id="request-1",
                state=SimpleNamespace(value="created"),
            ),
            continuation_claimed=False,
        )
        executor._pending["task-1"] = pending
        message = SimpleNamespace(
            task_id="task-1",
            message_id=message_id,
            extensions=[A2A_INPUT_EXTENSION_URI],
            metadata=(
                {A2A_INPUT_EXTENSION_URI: {"kind": "resolution"}}
                if metadata is None
                else metadata
            ),
        )
        context = SimpleNamespace(
            state={"method": "send", "request_id": "transport-new"},
            requested_extensions=(A2A_INPUT_EXTENSION_URI,),
        )
        monkeypatch.setattr(
            executor,
            "_activated_actor",
            AsyncMock(return_value=actor),
        )
        monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=actor))
        return executor, pending, SimpleNamespace(message=message), context

    executor, _, params, context = scenario()
    monkeypatch.setattr(
        executor,
        "_activated_actor",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=None))
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor.prepare_follow_up(params, context)

    executor, _, params, context = scenario(message_id=None)
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor.prepare_follow_up(params, context)

    executor, pending, params, context = scenario()

    class _RemovingLock:
        async def __aenter__(self) -> None:
            executor._pending.clear()

        async def __aexit__(self, *args: object) -> None:
            pass

    pending.lock = _RemovingLock()
    terminal = AsyncMock()
    monkeypatch.setattr(executor, "_raise_stored_terminal_error", terminal)
    assert await executor.prepare_follow_up(params, context) is None
    terminal.assert_awaited_once()

    executor, _, params, context = scenario(seen={"message-new"})
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor.prepare_follow_up(params, context)

    metadata_cases: tuple[object, ...] = ("invalid", {})
    for metadata in metadata_cases:
        executor, pending, params, context = scenario(metadata=metadata)
        assert await executor.prepare_follow_up(params, context) is None
        assert context.state[a2a_router._A2A_REFRESH_STATE_KEY] is pending

    executor, pending, params, context = scenario()
    replacement = SimpleNamespace(request_id="request-replacement")
    monkeypatch.setattr(
        a2a_router,
        "decode_a2a_input_resolution_metadata",
        MagicMock(
            side_effect=InputContractError(
                InputErrorCode.EXPIRED,
                "interaction",
                "expired",
            )
        ),
    )
    monkeypatch.setattr(
        executor,
        "_terminal_interaction",
        AsyncMock(return_value=(replacement, ResolutionStatus.EXPIRED)),
    )
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor.prepare_follow_up(params, context)
    assert pending.request is replacement


@pytest.mark.anyio
async def test_executor_cancel_settles_and_cleans_pending_input(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    actor = SimpleNamespace(principal="owner")
    request = SimpleNamespace(request_id="request-1")
    response = object()
    iterator = _CancelledResponse()
    pending = a2a_router._A2APendingInput(
        task_id="task-1",
        context_id="ctx-1",
        actor=cast(Any, actor),
        request=cast(Any, request),
        response=response,
        iterator=iterator,
        translator=MagicMock(),
        orchestrator=cast(Any, MagicMock()),
        activated=True,
        handler=cast(Any, MagicMock()),
        updater=MagicMock(),
    )
    executor._pending["task-1"] = pending
    authorize = AsyncMock()
    applied_request = SimpleNamespace(request_id="request-1")
    apply_resolution = AsyncMock(
        return_value=SimpleNamespace(request=applied_request)
    )
    cleanup = AsyncMock()
    monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=actor))
    monkeypatch.setattr(executor, "_authorize_resolution", authorize)
    monkeypatch.setattr(executor, "_apply_resolution", apply_resolution)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", cleanup)
    queue = _FakeEventQueue()

    await executor.cancel(
        SimpleNamespace(task_id="task-1", context_id="ctx-1"),
        queue,
    )

    assert apply_resolution.await_args is not None
    resolution = apply_resolution.await_args.args[2]
    assert isinstance(resolution, CancelledResolution)
    authorize.assert_awaited_once_with(actor, pending, resolution)
    cleanup.assert_awaited_once_with(response, iterator, cancelled=True)
    assert executor._pending == {}
    assert queue.events == [
        {
            "kind": "status",
            "state": "canceled",
            "metadata": {
                A2A_INPUT_EXTENSION_URI: {
                    "kind": "resolution",
                    "request_id": "request-1",
                }
            },
            "task_id": "task-1",
            "context_id": "ctx-1",
        }
    ]


@pytest.mark.anyio
async def test_executor_cancel_keeps_pending_when_settlement_fails(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    actor = SimpleNamespace(principal="owner")
    response = object()
    iterator = _CancelledResponse()
    pending = a2a_router._A2APendingInput(
        task_id="task-1",
        context_id="ctx-1",
        actor=cast(Any, actor),
        request=cast(Any, SimpleNamespace(request_id="request-1")),
        response=response,
        iterator=iterator,
        translator=MagicMock(),
        orchestrator=cast(Any, MagicMock()),
        activated=True,
        handler=cast(Any, MagicMock()),
        updater=MagicMock(),
    )
    executor._pending["task-1"] = pending
    cleanup = AsyncMock()
    monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=None))
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor.cancel(
            SimpleNamespace(task_id="task-1", context_id="ctx-1"),
            _FakeEventQueue(),
        )
    monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=actor))
    monkeypatch.setattr(executor, "_authorize_resolution", AsyncMock())
    monkeypatch.setattr(
        executor,
        "_apply_resolution",
        AsyncMock(side_effect=RuntimeError("settlement failed")),
    )
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", cleanup)
    queue = _FakeEventQueue()

    with pytest.raises(RuntimeError, match="settlement failed"):
        await executor.cancel(
            SimpleNamespace(task_id="task-1", context_id="ctx-1"),
            queue,
        )

    cleanup.assert_not_awaited()
    assert executor._pending == {"task-1": pending}
    assert queue.events == []


@pytest.mark.anyio
async def test_executor_cancel_removes_terminal_pending_after_cleanup_failure(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    actor = SimpleNamespace(principal="owner")
    response = object()
    iterator = _CancelledResponse()
    pending = a2a_router._A2APendingInput(
        task_id="task-1",
        context_id="ctx-1",
        actor=cast(Any, actor),
        request=cast(Any, SimpleNamespace(request_id="request-1")),
        response=response,
        iterator=iterator,
        translator=MagicMock(),
        orchestrator=cast(Any, MagicMock()),
        activated=True,
        handler=cast(Any, MagicMock()),
        updater=MagicMock(),
    )
    executor._pending["task-1"] = pending
    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(executor, "_actor", AsyncMock(return_value=actor))
    monkeypatch.setattr(executor, "_authorize_resolution", AsyncMock())
    monkeypatch.setattr(
        executor,
        "_apply_resolution",
        AsyncMock(
            return_value=SimpleNamespace(
                request=SimpleNamespace(request_id="request-1")
            )
        ),
    )
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", cleanup)
    executor._task_store = SimpleNamespace(get=AsyncMock(return_value=None))
    queue = _FakeEventQueue()

    with pytest.raises(BaseExceptionGroup) as error_info:
        await executor.cancel(
            SimpleNamespace(task_id="task-1", context_id="ctx-1"),
            queue,
        )

    cleanup.assert_awaited_once_with(response, iterator, cancelled=True)
    assert executor._pending == {}
    assert [str(error) for error in error_info.value.exceptions] == [
        "cleanup failed",
        "A2A task is not stored",
    ]
    assert queue.events == []


@pytest.mark.anyio
async def test_resume_pending_aborts_before_cleanup_on_stream_failure(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    executor = AvalanA2AAgentExecutor(FastAPI())
    translator = SimpleNamespace(
        process=AsyncMock(),
        finish=AsyncMock(),
        abort=AsyncMock(side_effect=RuntimeError("abort failed")),
        succeeded=False,
    )
    response = object()
    iterator = _ErroredResponse()
    pending = a2a_router._A2APendingInput(
        task_id="task-1",
        context_id="ctx-1",
        actor=cast(Any, SimpleNamespace(principal="owner")),
        request=cast(Any, SimpleNamespace(request_id="request-1")),
        response=response,
        iterator=iterator,
        translator=cast(Any, translator),
        orchestrator=cast(Any, MagicMock()),
        activated=True,
        handler=cast(Any, MagicMock()),
        updater=MagicMock(),
    )
    executor._pending["task-1"] = pending
    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", cleanup)
    updater = _FakeUpdater()
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor._resume_pending(
            SimpleNamespace(call_context=SimpleNamespace(state={})),
            updater,
        )
    context = SimpleNamespace(
        call_context=SimpleNamespace(
            state={a2a_router._A2A_RESOLUTION_STATE_KEY: pending}
        )
    )

    with pytest.raises(BaseExceptionGroup) as error_info:
        await executor._resume_pending(context, updater)

    assert [str(error) for error in error_info.value.exceptions] == [
        "stream broken",
        "abort failed",
        "cleanup failed",
    ]
    translator.abort.assert_awaited_once_with(StreamTerminalOutcome.ERRORED)
    cleanup.assert_awaited_once_with(response, iterator, cancelled=False)
    assert executor._pending == {}
    assert updater.statuses[0]["state"] == "working"


@pytest.mark.anyio
async def test_pending_publish_and_successor_failures(
    monkeypatch: pytest.MonkeyPatch,
    fake_a2a_imports: Any,
) -> None:
    pending = SimpleNamespace(task_id="task-1")
    executor = AvalanA2AAgentExecutor(FastAPI())
    executor._pending["task-1"] = object()
    with pytest.raises(RuntimeError, match="already has pending"):
        await executor._publish_pending(
            cast(Any, pending),
            object(),
            object(),
        )

    for previous in (None, object()):
        executor = AvalanA2AAgentExecutor(FastAPI())
        if previous is not None:
            executor._pending["task-1"] = cast(Any, previous)
        monkeypatch.setattr(
            executor,
            "_emit_pending_status",
            AsyncMock(side_effect=RuntimeError("publish failed")),
        )
        with pytest.raises(RuntimeError, match="publish failed"):
            await executor._publish_pending(
                cast(Any, pending),
                object(),
                object(),
                previous=cast(Any, previous),
            )
        assert executor._pending.get("task-1") is previous

    async def one(item: object) -> AsyncIterator[object]:
        yield item

    translator = SimpleNamespace(process=AsyncMock(), abort=AsyncMock())
    current = SimpleNamespace(request_id="request-current")
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", AsyncMock())
    for request_id, handler_request, message in (
        (None, None, "no request correlation"),
        (
            "request-next",
            SimpleNamespace(request_id="request-current"),
            "reused its prior request",
        ),
    ):
        executor = AvalanA2AAgentExecutor(FastAPI())
        continuation = SimpleNamespace(
            task_id="task-1",
            request=current,
            response=object(),
            iterator=one(
                SimpleNamespace(
                    kind=StreamItemKind.INTERACTION_PENDING,
                    correlation=SimpleNamespace(request_id=request_id),
                )
            ),
            translator=translator,
            handler=SimpleNamespace(
                request=AsyncMock(return_value=handler_request)
            ),
        )
        executor._pending["task-1"] = cast(Any, continuation)
        with pytest.raises(StreamValidationError, match=message):
            await executor._continue_pending(
                cast(Any, continuation),
                _FakeUpdater(),
                activated=True,
            )


@pytest.mark.anyio
async def test_interaction_service_negative_branches(
    monkeypatch: pytest.MonkeyPatch,
    fake_a2a_imports: Any,
) -> None:
    fake_a2a_imports.TaskState.TASK_STATE_INPUT_REQUIRED = "input_required"
    pending = SimpleNamespace(
        actor=object(),
        request=SimpleNamespace(
            request_id="request-1",
            origin=object(),
            state_revision=1,
        ),
    )
    empty = AvalanA2AAgentExecutor(FastAPI())
    assert await empty._terminal_interaction(cast(Any, pending)) is None
    assert await empty._stored_task("task-1", object()) is None

    broker = SimpleNamespace(
        inspect=AsyncMock(
            side_effect=(
                SimpleNamespace(status=ResolutionStatus.EXPIRED),
                object(),
            )
        )
    )
    service = SimpleNamespace(
        configuration=SimpleNamespace(
            broker=broker,
            principal_resolver=AsyncMock(
                side_effect=RuntimeError("auth failed")
            ),
        ),
    )
    app = FastAPI()
    app.state.interaction_service = service
    executor = AvalanA2AAgentExecutor(app)
    for name, value in (
        ("ServerInteractionService", SimpleNamespace),
        ("InteractionTerminalMetadata", SimpleNamespace),
        (
            "InteractionCorrelation",
            SimpleNamespace(from_request=MagicMock(return_value=object())),
        ),
        ("ScopedInteractionLookup", lambda **values: values),
    ):
        monkeypatch.setattr(a2a_router, name, value)
    service.configuration.policy = InteractionPolicy(
        capability_state=TaskInputCapabilityState.DORMANT
    )
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor._activated_actor(
            SimpleNamespace(requested_extensions=(A2A_INPUT_EXTENSION_URI,))
        )
    assert await executor._terminal_interaction(cast(Any, pending)) == (
        None,
        ResolutionStatus.EXPIRED,
    )
    assert await executor._terminal_interaction(cast(Any, pending)) is None

    executor._task_store = SimpleNamespace(
        get=AsyncMock(
            side_effect=(
                None,
                SimpleNamespace(status=SimpleNamespace(state="working")),
            )
        )
    )
    results = [
        await executor._working_task(
            "task-1",
            object(),
            cast(Any, pending),
        )
        for _ in range(2)
    ]
    assert results[0] is None and results[1].status.state == "working"

    stored: tuple[dict[str, object], ...] = (
        {A2A_INPUT_EXTENSION_URI: None},
        {A2A_INPUT_EXTENSION_URI: {}},
        {A2A_INPUT_EXTENSION_URI: {"interaction_state": "invalid"}},
    )
    monkeypatch.setattr(
        executor,
        "_stored_task",
        AsyncMock(
            side_effect=tuple(
                SimpleNamespace(metadata=value) for value in stored
            )
        ),
    )
    for _ in stored:
        await executor._raise_stored_terminal_error("task-1", object())

    for state in (
        {},
        {a2a_router._A2A_HTTP_REQUEST_STATE_KEY: object()},
    ):
        assert await executor._actor(SimpleNamespace(state=state)) is None

    service.configuration.authorizer = SimpleNamespace(
        authorize=AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        a2a_router,
        "InteractionRequestAuthorizationTarget",
        lambda **values: SimpleNamespace(**values),
    )
    actor = SimpleNamespace(principal="owner")
    with pytest.raises(Exception, match="Structured input contract result"):
        await executor._authorize_resolution(
            actor, cast(Any, pending), object()
        )

    error = SimpleNamespace(
        code=InputErrorCode.EXPIRED,
        path="interaction",
        message="expired",
    )
    rejected_type = type("_Rejected", (SimpleNamespace,), {})
    service.configuration.broker = SimpleNamespace(
        cancel=AsyncMock(
            side_effect=(
                object(),
                SimpleNamespace(store_result=rejected_type(error=error)),
                SimpleNamespace(store_result=SimpleNamespace(error=error)),
                SimpleNamespace(store_result=SimpleNamespace()),
            )
        )
    )
    for name, value in (
        ("CancelledResolution", SimpleNamespace),
        ("InteractionBrokerResult", SimpleNamespace),
        ("ResolveInteractionRejected", rejected_type),
        ("CancelInteractionCommand", lambda **values: values),
    ):
        monkeypatch.setattr(a2a_router, name, value)
    for _ in range(4):
        with pytest.raises(Exception):
            await executor._apply_resolution(
                actor,
                cast(Any, pending),
                SimpleNamespace(),
            )


@pytest.mark.anyio
async def test_auto_resume_failure_logs_and_cleans_pending(
    monkeypatch,
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    executor = AvalanA2AAgentExecutor(app)
    handler = a2a_router._A2AInputHandler()
    request = cast(Any, SimpleNamespace(request_id="request-1"))
    handler_task = create_task(
        handler(cast(Any, SimpleNamespace(request=request)))
    )
    assert await handler.request("request-1") is request
    handler_task.cancel()
    with pytest.raises(CancelledError):
        await handler_task
    translator = SimpleNamespace(
        abort=AsyncMock(side_effect=RuntimeError("abort failed"))
    )
    response = object()
    iterator = _CancelledResponse()
    pending = a2a_router._A2APendingInput(
        task_id="task-1",
        context_id="ctx-1",
        actor=cast(Any, SimpleNamespace(principal="owner")),
        request=cast(Any, request),
        response=response,
        iterator=iterator,
        translator=cast(Any, translator),
        orchestrator=cast(Any, MagicMock()),
        activated=True,
        handler=handler,
        updater=MagicMock(),
    )
    pending.handler = SimpleNamespace(
        settlement_event=lambda request_id: SimpleNamespace(
            wait=AsyncMock(side_effect=CancelledError)
        )
    )
    with pytest.raises(CancelledError):
        await executor._auto_resume_pending(pending)
    pending.handler = handler
    executor._pending[pending.task_id] = pending
    monkeypatch.setattr(
        executor,
        "_terminal_interaction",
        AsyncMock(return_value=None),
    )
    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", cleanup)

    await executor._auto_resume_pending(pending)

    assert executor._pending == {}
    translator.abort.assert_awaited_once_with(StreamTerminalOutcome.ERRORED)
    cleanup.assert_awaited_once_with(
        response,
        iterator,
        cancelled=False,
    )
    assert app.state.logger.error.call_count == 3


@pytest.mark.anyio
async def test_executor_cleans_response_on_cancellation(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    response = _CancelledResponse()
    cleaned: list[bool] = []

    async def fake_orchestrate(*args: object, **kwargs: object):
        return response, "response-id", 123

    async def fake_cleanup(*args: object, cancelled: bool) -> None:
        cleaned.append(cancelled)

    monkeypatch.setattr(a2a_router, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", fake_cleanup)

    with pytest.raises(CancelledError):
        await executor.execute(_ExecutorContext(), _FakeEventQueue())

    assert cleaned == [True]


@pytest.mark.anyio
async def test_executor_cleans_response_on_stream_error(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    response = _ErroredResponse()
    cleaned: list[bool] = []

    async def fake_orchestrate(*args: object, **kwargs: object):
        return response, "response-id", 123

    async def fake_cleanup(*args: object, cancelled: bool) -> None:
        cleaned.append(cancelled)

    monkeypatch.setattr(a2a_router, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(a2a_router, "cleanup_stream_sources", fake_cleanup)

    with pytest.raises(RuntimeError, match="stream broken"):
        await executor.execute(_ExecutorContext(), _FakeEventQueue())

    assert cleaned == [False]

    async def pending_items() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            kind=StreamItemKind.INTERACTION_PENDING,
            correlation=SimpleNamespace(request_id=None),
        )

    translator = SimpleNamespace(process=AsyncMock(), abort=AsyncMock())
    monkeypatch.setattr(
        executor,
        "_interaction_runtime",
        AsyncMock(return_value=(SimpleNamespace(actor=object()), MagicMock())),
    )
    monkeypatch.setattr(
        a2a_router,
        "stream_consumer_iterator",
        lambda *args, **kwargs: pending_items(),
    )
    monkeypatch.setattr(
        a2a_router,
        "A2AResponseTranslator",
        lambda *args, **kwargs: translator,
    )
    with pytest.raises(StreamValidationError, match="no request correlation"):
        await executor.execute(_ExecutorContext(), _FakeEventQueue())
    translator.abort.assert_awaited_once_with(StreamTerminalOutcome.ERRORED)


@pytest.fixture
def fake_a2a_imports(monkeypatch):
    real_import_module = a2a_router.import_module
    fake_pb2 = _FakeA2APb2()

    def fake_import_module(name: str):
        if name == "a2a.types.a2a_pb2":
            return fake_pb2
        if name == "a2a.server.tasks.task_updater":
            return SimpleNamespace(TaskUpdater=_FakeSdkTaskUpdater)
        return real_import_module(name)

    monkeypatch.setattr(a2a_router, "import_module", fake_import_module)
    return fake_pb2


class _FakeProtoMessage:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _FakeA2APb2:
    AgentCapabilities = _FakeProtoMessage
    AgentCard = _FakeProtoMessage
    AgentExtension = _FakeProtoMessage
    AgentInterface = _FakeProtoMessage
    AgentSkill = _FakeProtoMessage
    Part = _FakeProtoMessage
    Task = _FakeProtoMessage
    TaskStatus = _FakeProtoMessage
    TaskState = SimpleNamespace(
        TASK_STATE_SUBMITTED="submitted",
        TASK_STATE_WORKING="working",
        TASK_STATE_INPUT_REQUIRED="input_required",
        TASK_STATE_CANCELED="canceled",
    )


class _FakeConstants:
    PROTOCOL_VERSION_1_0 = "1.0"
    TransportProtocol = SimpleNamespace(JSONRPC="JSONRPC")


class _FakePart:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _FakeMessage:
    def __init__(self, parts: list[object], *, role: object = "user") -> None:
        self.parts = parts
        self.role = role


class _CallableField:
    def value(self) -> str:
        return "callable"


class _HasFieldFalse:
    value = "hidden"

    def HasField(self, name: str) -> bool:
        return False


class _HasFieldRaises:
    value = "kept"

    def HasField(self, name: str) -> bool:
        raise ValueError(name)


class _ModelDumpFallback:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        if kwargs:
            raise TypeError("mode unsupported")
        return {"value": "fallback"}


class _ModelDumpMode:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        return {"value": "mode"}


class _ProtoLike:
    DESCRIPTOR = object()


class _BodyRequest:
    def __init__(
        self, body: bytes, *, path_params: dict[str, str] | None = None
    ) -> None:
        self._body = body
        self.path_params = path_params or {}

    async def body(self) -> bytes:
        return self._body


class _SelfRaw:
    @property
    def raw(self) -> "_SelfRaw":
        return self


class _FakeUpdater:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, object]] = []
        self.statuses: list[dict[str, object]] = []
        self.completed = 0
        self.cancelled = 0
        self.failed_count = 0

    async def add_artifact(self, parts, **kwargs: object) -> None:
        self.artifacts.append({"parts": parts, **kwargs})

    async def update_status(self, state, metadata=None) -> None:
        self.statuses.append({"state": state, "metadata": metadata or {}})

    async def complete(self) -> None:
        self.completed += 1

    async def cancel(self) -> None:
        self.cancelled += 1

    async def failed(self) -> None:
        self.failed_count += 1


class _FakeSdkTaskUpdater(_FakeUpdater):
    def __init__(
        self,
        event_queue: "_FakeEventQueue",
        *,
        task_id: str,
        context_id: str,
    ) -> None:
        super().__init__()
        self._event_queue = event_queue
        self._task_id = task_id
        self._context_id = context_id

    async def add_artifact(self, parts, **kwargs: object) -> None:
        await super().add_artifact(parts, **kwargs)
        await self._event_queue.enqueue_event(
            {
                "kind": "artifact",
                "parts": parts,
                **kwargs,
            }
        )

    async def update_status(self, state, metadata=None) -> None:
        await super().update_status(state, metadata=metadata)
        await self._event_queue.enqueue_event(
            {
                "kind": "status",
                "state": state,
                "metadata": metadata or {},
                "task_id": self._task_id,
                "context_id": self._context_id,
            }
        )

    async def complete(self) -> None:
        await super().complete()
        await self._event_queue.enqueue_event({"kind": "complete"})

    async def cancel(self) -> None:
        await super().cancel()
        await self._event_queue.enqueue_event({"kind": "cancel"})

    async def failed(self) -> None:
        await super().failed()
        await self._event_queue.enqueue_event({"kind": "failed"})


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


class _ExecutorOrchestrator:
    model_ids = {"test-model"}
    sync_messages = AsyncMock()


_DEFAULT_CURRENT_TASK = SimpleNamespace()


class _ExecutorContext:
    task_id = "task-1"
    context_id = "ctx-1"

    def __init__(
        self,
        *,
        message: object | None = None,
        current_task: object | None = _DEFAULT_CURRENT_TASK,
        user_input: str = "hello",
    ) -> None:
        self.message = message
        self.current_task = current_task
        self._user_input = user_input

    def get_user_input(self) -> str:
        return self._user_input


class _CancelledResponse:
    input_token_count = 0
    output_token_count = 0
    can_think = False
    is_thinking = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise CancelledError

    def set_thinking(self, value: bool) -> None:
        self.is_thinking = value


class _ErroredResponse(_CancelledResponse):
    async def __anext__(self):
        raise RuntimeError("stream broken")


def _item(
    sequence: int,
    kind: StreamItemKind,
    **kwargs: object,
) -> CanonicalStreamItem:
    if kind is StreamItemKind.REASONING_DELTA:
        kwargs.setdefault("visibility", StreamVisibility.PRIVATE)
        kwargs.setdefault(
            "reasoning_representation",
            StreamReasoningRepresentation.NATIVE_TEXT,
        )
        kwargs.setdefault("segment_instance_ordinal", 0)
    return CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=sequence,
        kind=kind,
        channel=(
            StreamChannel.CONTROL
            if kind
            in {
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_INPUT_REQUIRED,
            }
            else StreamChannel.REASONING
        ),
        **kwargs,
    )


def _tool_item(
    sequence: int,
    kind: StreamItemKind,
    **kwargs: object,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=sequence,
        kind=kind,
        channel=StreamChannel.TOOL_EXECUTION,
        correlation=StreamItemCorrelation(tool_call_id="call-1"),
        **kwargs,
    )
