from asyncio import CancelledError
from base64 import b64encode
from json import loads
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.a2a import router as a2a_router
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    AvalanA2AAgentExecutor,
    install_a2a_routes,
)
from avalan.server.container_policy import RemoteContainerRequestPolicy
from avalan.server.entities import ContentFile, ContentImage, ContentText


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
    translator = A2AResponseTranslator(updater)

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
    assert updater.artifacts[0]["artifact_id"] == "reasoning"
    assert updater.artifacts[1]["artifact_id"] == "call-1"
    assert updater.artifacts[-1]["last_chunk"] is True
    assert updater.statuses[0]["metadata"]["tool_name"] == "shell.run"
    assert updater.completed == 1


@pytest.mark.anyio
async def test_translator_projects_skills_tool_activity_safely(
    fake_a2a_imports,
) -> None:
    updater = _FakeUpdater()
    translator = A2AResponseTranslator(updater)

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
            2,
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

    assert a2a_router._a2a_tool_item_text(tool_output, tool_output.data) == ""
    assert (
        a2a_router._a2a_tool_item_text(tool_progress, tool_progress.data) == ""
    )
    assert loads(
        a2a_router._a2a_tool_item_text(
            skills_output,
            skills_output.data,
            tool_name="skills.read",
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
        )
        == "Source: <host-path>/SKILL.md"
    )


@pytest.mark.anyio
async def test_translator_redacts_answer_and_reasoning_skill_echoes(
    fake_a2a_imports,
) -> None:
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


@pytest.mark.anyio
async def test_executor_cancel_and_exception_paths(
    monkeypatch,
    fake_a2a_imports,
) -> None:
    app = FastAPI()
    app.state.logger = MagicMock()
    app.state.orchestrator = _ExecutorOrchestrator()
    executor = AvalanA2AAgentExecutor(app)
    context = _ExecutorContext()
    event_queue = _FakeEventQueue()

    async def fail_orchestrate(*args: object, **kwargs: object):
        raise RuntimeError("broken")

    monkeypatch.setattr(a2a_router, "orchestrate", fail_orchestrate)

    with pytest.raises(RuntimeError, match="broken"):
        await executor.execute(context, event_queue)
    await executor.cancel(context, event_queue)

    assert event_queue.events


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
    AgentInterface = _FakeProtoMessage
    AgentSkill = _FakeProtoMessage
    Part = _FakeProtoMessage
    Task = _FakeProtoMessage
    TaskStatus = _FakeProtoMessage
    TaskState = SimpleNamespace(
        TASK_STATE_SUBMITTED="submitted",
        TASK_STATE_WORKING="working",
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
