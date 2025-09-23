import asyncio
from asyncio import CancelledError
from collections.abc import AsyncGenerator
import importlib
from datetime import datetime, timezone
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

from a2a import types as a2a_types
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
from avalan.entities import (
    MessageRole,
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    A2AStreamEventConverter,
    A2ATaskCreateRequest,
    StreamState,
    _append_unique,
    _artifact_parts_from_payload,
    _build_agent_card,
    _call_identifier,
    _capability_extensions,
    _coerce,
    _coerce_list,
    _collect_jsonrpc_messages,
    _default_model_id,
    _default_skill,
    _enum_value,
    _extract_jsonrpc_instructions,
    _extract_jsonrpc_metadata,
    _filter_payload,
    _input_mode_for_spec,
    _jsonrpc_message_text,
    _jsonrpc_message_to_chat,
    _message_parts_from_payload,
    _normalize_jsonrpc_task_request,
    _normalize_task_request,
    _output_mode_for_spec,
    _role_from_payload,
    _select_jsonrpc_model,
    _skill_from_spec,
    _skill_tags,
    _state_for_item,
    _status_to_state,
    _task_metadata,
    _task_metadata_from_overview,
    _timestamp_to_iso,
    _token_text,
    create_task,
    di_get_task_store,
    router,
)
from avalan.server.a2a.store import TaskStore
from avalan.server.entities import ResponseFormatText


def test_timestamp_role_and_parts_helpers() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    assert _timestamp_to_iso(None) is None
    assert _timestamp_to_iso(ts) == "2024-01-01T00:00:00Z"
    assert _status_to_state("completed").value == "completed"
    assert _status_to_state("unknown").value == "unknown"
    assert _role_from_payload(None).value == "agent"
    assert _role_from_payload("user").value == "user"
    assert _role_from_payload("other").value == "agent"

    content = [
        {"type": "text", "text": "hello"},
        {"type": "data", "value": 1},
        "extra",
    ]
    parts = _message_parts_from_payload(content)
    assert getattr(parts[0].root, "text") == "hello"
    assert getattr(parts[1].root, "data")["type"] == "data"
    assert getattr(parts[2].root, "text") == "extra"
    empty_parts = _message_parts_from_payload("ignored")
    assert getattr(empty_parts[0].root, "text") == ""

    artifact_parts = _artifact_parts_from_payload([
        {"type": "text", "text": "chunk"},
        {"type": "blob", "value": 2},
        "trail",
        3,
    ])
    assert getattr(artifact_parts[0].root, "text") == "chunk"
    assert getattr(artifact_parts[1].root, "data")["type"] == "blob"
    assert getattr(artifact_parts[2].root, "text") == "trail"
    assert getattr(artifact_parts[3].root, "data")["value"] == 3
    default_artifact = _artifact_parts_from_payload(None)
    assert getattr(default_artifact[0].root, "text") == ""

    overview = {
        "metadata": {"jsonrpc_id": "abc", "foo": "bar"},
        "model": "model-x",
        "instructions": "act",
    }
    metadata = _task_metadata_from_overview(overview)
    assert metadata == {"foo": "bar", "model": "model-x", "instructions": "act"}


def test_task_request_conversation_and_metadata() -> None:
    payload = A2ATaskCreateRequest(
        model="model-x",
        messages=[{"role": MessageRole.USER, "content": "Hello"}],
        stream=False,
        metadata={"temperature": 0.9},
        instructions="Follow",
        temperature=0.5,
        top_p=0.7,
        max_tokens=128,
        response_format=ResponseFormatText(type="text"),
    )

    conversation = payload.conversation()
    assert conversation[0].role is MessageRole.SYSTEM
    assert conversation[-1].content == "Hello"

    metadata = _task_metadata(payload)
    assert metadata["temperature"] == 0.9
    assert metadata["top_p"] == 0.7
    assert metadata["max_tokens"] == 128
    assert metadata["response_format"] == {"type": "text"}


def test_di_get_task_store_reuses_same_instance() -> None:
    asyncio.run(_run_di_get_task_store())


async def _run_di_get_task_store() -> None:
    app_state = SimpleNamespace()
    app = SimpleNamespace(state=app_state)
    scope = {
        "type": "http",
        "app": app,
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1234),
        "method": "GET",
        "path": "/",
        "root_path": "",
        "scheme": "http",
        "http_version": "1.1",
    }

    async def receive() -> dict[str, bytes]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive=receive)
    store_one = di_get_task_store(request)
    store_two = di_get_task_store(request)
    assert store_one is store_two


def test_model_selection_and_normalization() -> None:
    orchestrator = SimpleNamespace(model_ids={"b", "a"})
    payload = {
        "jsonrpc": "2.0",
        "id": "rpc-1",
        "method": "message/stream",
        "params": {
            "configuration": {
                "models": ["model-1", ""],
                "instructions": "cfg",
                "extra": True,
            },
            "context": {
                "messages": [{"role": "assistant", "text": "ctx"}],
                "history": [{"role": "user", "text": "old"}],
            },
            "messages": [{"role": "system", "text": "prep"}],
            "conversation": [{"role": "assistant", "text": "prior"}],
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}, "ignore"],
                "metadata": {"source": "test"},
            },
            "metadata": {"trace": "abc"},
            "instructions": "param",
        },
    }

    normalized = _normalize_jsonrpc_task_request(payload, orchestrator)
    assert normalized["model"] == "model-1"
    assert normalized["stream"] is True
    assert normalized["instructions"] == "param"
    assert normalized["metadata"]["jsonrpc_id"] == "rpc-1"
    assert normalized["metadata"]["params_metadata"] == {"trace": "abc"}

    non_jsonrpc = {"model": "plain", "messages": ["hi"]}
    assert _normalize_task_request(non_jsonrpc, orchestrator) == non_jsonrpc

    fallback_orchestrator = SimpleNamespace(model_ids=set())
    assert _default_model_id(fallback_orchestrator) == "default"


def test_normalize_jsonrpc_errors() -> None:
    orchestrator = SimpleNamespace(model_ids=set())
    with pytest.raises(ValueError):
        _normalize_jsonrpc_task_request({}, orchestrator)
    with pytest.raises(ValueError):
        _normalize_jsonrpc_task_request({"params": {"configuration": []}}, orchestrator)
    with pytest.raises(ValueError):
        _normalize_jsonrpc_task_request({"params": {"configuration": {}, "message": []}}, orchestrator)


def test_jsonrpc_helper_functions() -> None:
    params = {
        "context": {"instructions": "ctx", "messages": []},
        "instructions": " from params ",
    }
    assert _extract_jsonrpc_instructions(params, None) == " from params "
    assert _extract_jsonrpc_instructions({"context": {}}, {}) is None

    metadata = _extract_jsonrpc_metadata(
        {"id": 1},
        {"metadata": {"a": 1}},
        {"x": 2},
        {"metadata": {"b": 3}},
    )
    assert metadata == {
        "jsonrpc_id": 1,
        "configuration": {"x": 2},
        "params_metadata": {"a": 1},
        "message_metadata": {"b": 3},
    }

    selection_params = {"models": ["", "pick"], "model": "explicit"}
    orchestrator = SimpleNamespace(model_ids={"z"})
    assert _select_jsonrpc_model(selection_params, None, orchestrator) == "explicit"
    assert (
        _select_jsonrpc_model({}, {"modelIds": ["foo", "bar"]}, orchestrator)
        == "foo"
    )

    collected = _collect_jsonrpc_messages(
        {"conversation": [{"role": "user", "content": "hi"}]},
        {"role": "assistant", "text": "response"},
    )
    assert collected[-1] == {"role": "assistant", "content": "response"}

    message = {
        "parts": [
            {"kind": "text", "text": "part"},
            "skip",
            {"kind": "other", "text": "ignored"},
        ]
    }
    assert _jsonrpc_message_text(message) == "part"
    assert _jsonrpc_message_text({"content": "fallback"}) == "fallback"

    chat = _jsonrpc_message_to_chat({"role": "user", "text": "hi"})
    assert chat == {"role": "user", "content": "hi"}

    assert _enum_value(None) is None
    assert _enum_value(SimpleNamespace(value="x")) == "x"

    tags = _skill_tags(None, "Tool Runner", "Tool Runner")
    assert tags == ["tool", "runner"]
    assert _skill_tags() == ["general"]

    target: list[str] = []
    _append_unique(target, " value ")
    _append_unique(target, "value")
    assert target == ["value"]

    class DummySpec:
        input_type = SimpleNamespace(value="text")
        output_type = SimpleNamespace(value="json")

    assert _input_mode_for_spec(DummySpec()) == "text/plain"
    assert _output_mode_for_spec(DummySpec()) == "application/json"


def test_model_selection_and_message_text_fallbacks() -> None:
    orchestrator = SimpleNamespace(model_ids={"beta", "alpha"})
    assert (
        _select_jsonrpc_model([], "ignored", orchestrator)
        == _default_model_id(orchestrator)
    )
    assert _jsonrpc_message_text({}) == ""


def test_translator_switch_state_paths() -> None:
    asyncio.run(_run_translator_switch_state_paths())


async def _run_translator_switch_state_paths() -> None:
    store = TaskStore()
    await store.create_task(
        "switch",
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator("switch", store)

    created = await translator._switch_state(StreamState.REASONING, None)
    assert created
    assert await translator._switch_state(StreamState.REASONING, None) == []
    completed_reasoning = await translator._switch_state(None, None)
    assert any(event["event"] == "artifact.completed" for event in completed_reasoning)

    tool_events = await translator._switch_state(StreamState.TOOL, "tool-1")
    assert translator._tool_artifact_id == "tool-1"
    assert tool_events
    replaced_tool = await translator._switch_state(StreamState.TOOL, "tool-2")
    assert translator._tool_artifact_id == "tool-2"
    assert any(event["event"] == "artifact.completed" for event in replaced_tool)

    answer_events = await translator._switch_state(StreamState.ANSWER, None)
    assert translator._tool_artifact_id is None
    assert translator._answer_artifact_id == "answer"
    finished = await translator._switch_state(None, None)
    assert translator._answer_artifact_id is None
    assert any(event["event"] == "artifact.completed" for event in finished)


def test_tool_handlers_cover_branches() -> None:
    asyncio.run(_run_tool_handlers_cover_branches())


async def _run_tool_handlers_cover_branches() -> None:
    async def prepare(task_id: str) -> A2AResponseTranslator:
        local_store = TaskStore()
        await local_store.create_task(
            task_id,
            model="model",
            instructions=None,
            input_messages=[],
            metadata={},
        )
        return A2AResponseTranslator(task_id, local_store)

    translator_dict = await prepare("tool-dict")
    call = ToolCall(id="call-1", name="tool", arguments={"value": 1})
    event_dict = Event(
        type=EventType.TOOL_PROCESS,
        payload={"calls": [call, "skip"]},
    )
    process_events = await translator_dict._handle_tool_process(event_dict)
    assert any(event["event"] == "artifact.delta" for event in process_events)

    translator_list = await prepare("tool-list")
    call_none = ToolCall(id="call-2", name="tool", arguments=None)
    event_list = Event(type=EventType.TOOL_PROCESS, payload=[call_none])
    process_list_events = await translator_list._handle_tool_process(event_list)
    assert process_list_events

    translator_result = await prepare("tool-result")
    result_call = ToolCall(id="call-3", name="tool", arguments=None)
    result = ToolCallResult(
        id="res-1", call=result_call, result={"ok": True}, name="tool", arguments=None
    )
    result_event = Event(
        type=EventType.TOOL_RESULT,
        payload={"result": result, "call": result_call},
    )
    result_events = await translator_result._handle_tool_result(result_event)
    assert any(event["event"] == "artifact.completed" for event in result_events)

    translator_error = await prepare("tool-error")
    error_call = ToolCall(id="call-4", name="tool", arguments=None)
    error = ToolCallError(
        id="err-1",
        call=error_call,
        error=RuntimeError("fail"),
        message="fail",
        name="tool",
        arguments=None,
    )
    error_event = Event(
        type=EventType.TOOL_RESULT,
        payload={"result": error, "call": error_call},
    )
    error_events = await translator_error._handle_tool_result(error_event)
    assert any(event["event"] == "artifact.completed" for event in error_events)

    translator_payload = await prepare("tool-payload")
    payload_call = ToolCall(id="call-5", name="tool", arguments=None)
    payload_event = Event(type=EventType.TOOL_RESULT, payload=payload_call)
    payload_events = await translator_payload._handle_tool_result(payload_event)
    assert payload_events

    translator_fallback = await prepare("tool-fallback")
    translator_fallback._tool_artifact_id = "call-6"
    fallback_event = Event(type=EventType.TOOL_RESULT, payload={})
    fallback_events = await translator_fallback._handle_tool_result(fallback_event)
    assert any(event["event"] == "artifact.completed" for event in fallback_events)

    translator_uuid = await prepare("tool-uuid")
    translator_uuid._tool_artifact_id = None
    uuid_event = Event(type=EventType.TOOL_RESULT, payload={})
    uuid_events = await translator_uuid._handle_tool_result(uuid_event)
    assert uuid_events

    translator_token = await prepare("tool-token")
    token_call = ToolCall(id="call-7", name="tool", arguments=None)
    token = ToolCallToken(token="", call=token_call)
    token_events = await translator_token._handle_tool_token(token)
    assert any(event["event"] == "artifact.delta" for event in token_events)


def test_translator_additional_branches() -> None:
    asyncio.run(_run_translator_additional_branches())


async def _run_translator_additional_branches() -> None:
    store = TaskStore()
    await store.create_task(
        "extra", model="model", instructions=None, input_messages=[], metadata={}
    )
    translator = A2AResponseTranslator("extra", store)

    await translator._ensure_reasoning_artifact()
    translator._state = StreamState.REASONING
    non_tool_event = Event(type=EventType.START, payload={})
    non_tool_events = await translator._process_item(non_tool_event)
    assert any(event["event"] == "artifact.completed" for event in non_tool_events)

    tool_call = ToolCall(id="proc", name="processor", arguments={"a": 1})
    process_event = Event(
        type=EventType.TOOL_PROCESS,
        payload={"calls": [tool_call]},
    )
    process_events = await translator._process_item(process_event)
    assert any(event["event"] == "artifact.delta" for event in process_events)

    await translator._process_item("chunk")
    finish_events = await translator._finish()
    assert any(event["event"] == "artifact.completed" for event in finish_events)

    translator_tool = A2AResponseTranslator("extra-tool", store)
    await store.create_task(
        "extra-tool",
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    tool_events = await translator_tool._handle_tool_token(
        ToolCallToken(token="args", call=None)
    )
    assert any(event["event"] == "artifact.delta" for event in tool_events)


def test_event_converter_handles_all_event_types() -> None:
    asyncio.run(_run_event_converter_handles_all_event_types())


async def _run_event_converter_handles_all_event_types() -> None:
    store = TaskStore()
    task_id = "convert"
    initial_events = await store.create_task(
        task_id,
        model="model",
        instructions="Guide",
        input_messages=[],
        metadata={"jsonrpc_id": "rpc"},
    )
    message_id, message_created = await store.ensure_message(
        task_id, role="assistant", channel="out"
    )
    message_delta = await store.add_message_delta(task_id, message_id, "chunk")
    message_complete = await store.complete_message(task_id, message_id)

    artifact_id, artifact_created = await store.ensure_artifact(
        task_id,
        artifact_id="artifact",
        name="Artifact",
        kind="output",
        role="assistant",
        metadata={"role": "assistant"},
    )
    artifact_delta = await store.add_artifact_delta(
        task_id, artifact_id, {"type": "text", "text": "piece"}
    )
    artifact_finish = await store.complete_artifact(task_id, artifact_id)

    status_events = await store.add_status_event(
        task_id,
        status="completed",
        metadata={"phase": "done", "tool_name": "tool"},
    )
    failure_events = await store.fail_task(task_id, "failure")

    converter = A2AStreamEventConverter(task_id, store)
    for event in initial_events:
        converted = await converter.convert(event)
        assert converted["result"]

    for event in message_created + message_delta + message_complete:
        converted = await converter.convert(event)
        assert converted["result"]

    for event in artifact_created + artifact_delta + artifact_finish:
        converted = await converter.convert(event)
        assert converted["result"]

    for event in status_events + failure_events:
        converted = await converter.convert(event)
        assert converted["result"]

    assert await converter.convert({"event": None}) == {"event": None}
    assert await converter.convert({}) == {}


def test_event_converter_fallbacks() -> None:
    asyncio.run(_run_event_converter_fallbacks())


async def _run_event_converter_fallbacks() -> None:
    store = TaskStore()
    task_id = "convert-fallback"
    await store.create_task(
        task_id,
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    converter = A2AStreamEventConverter(task_id, store)

    assert await converter.convert({"event": "unknown"}) == {"event": "unknown"}

    message_event = {
        "event": "message.delta",
        "data": {"message": {}},
    }
    assert await converter.convert(message_event) == message_event


def test_message_and_artifact_result_branches() -> None:
    asyncio.run(_run_message_and_artifact_result_branches())


async def _run_message_and_artifact_result_branches() -> None:
    store = TaskStore()
    task_id = "message-artifact"
    await store.create_task(
        task_id,
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    converter = A2AStreamEventConverter(task_id, store)

    assert (
        await converter._message_result({"event": "message.delta", "data": {}})
        is None
    )
    assert (
        await converter._message_result(
            {"event": "message.delta", "data": {"message": {"id": None}}}
        )
        is None
    )

    artifact_event = {"event": "artifact.delta", "data": {}}
    assert await converter._artifact_result(artifact_event) is None
    assert (
        await converter._artifact_result(
            {"event": "artifact.delta", "data": {"artifact": {}}}
        )
        is None
    )

    artifact_id, _ = await store.ensure_artifact(
        task_id,
        artifact_id="artifact",
        name="Artifact",
        kind="output",
        role="assistant",
    )
    await store.add_artifact_delta(
        task_id, artifact_id, {"type": "text", "text": "one"}
    )
    await store.add_artifact_delta(
        task_id, artifact_id, {"type": "text", "text": "two"}
    )
    complete_events = await store.complete_artifact(task_id, artifact_id)
    complete_event = complete_events[-1]
    artifact_payload = await store.get_artifact(task_id, artifact_id)
    content_length = len(artifact_payload["content"])

    converter._artifact_progress[artifact_id] = 0
    result_new = await converter._artifact_result(complete_event)
    assert result_new.append is True and result_new.last_chunk is True

    converter._artifact_progress[artifact_id] = content_length
    result_repeat = await converter._artifact_result(complete_event)
    assert result_repeat.append is True and result_repeat.last_chunk is True

    empty_id, _ = await store.ensure_artifact(
        task_id,
        artifact_id="empty",
        name=None,
        kind="output",
        role="assistant",
    )
    empty_complete = await store.complete_artifact(task_id, empty_id)
    empty_event = empty_complete[-1]
    converter._artifact_progress[empty_id] = 0
    empty_result = await converter._artifact_result(empty_event)
    assert empty_result.append is True and empty_result.last_chunk is True


def test_agent_card_and_skills() -> None:
    goal = SimpleNamespace(task="Solve", instructions=["Step 1", "Step 2"])
    spec = SimpleNamespace(
        system_prompt="System", developer_prompt="Developer", goal=goal
    )
    spec.input_type = SimpleNamespace(value="text")
    spec.output_type = SimpleNamespace(value="json")
    operation = SimpleNamespace(specification=spec)
    ignored_operation = SimpleNamespace(specification=None)
    orchestrator = SimpleNamespace(
        id=uuid4(),
        name="Agent",
        operations=[ignored_operation, operation],
        model_ids={"m1", "m2"},
    )

    card = _build_agent_card(
        orchestrator, "Execute", "Run things", "https://example.com/tasks"
    )
    assert card["capabilities"]["extensions"]
    assert card["skills"][0]["examples"] == ["Step 1", "Step 2"]

    default_skill = _default_skill("Run", None, orchestrator, ["text"], ["json"], [])
    assert default_skill["input_modes"] == ["text"]

    extensions = _capability_extensions(["instruction"], ["x", "y"])
    assert extensions[0]["uri"].endswith("instructions")


def test_filter_helper_returns_same_payload() -> None:
    assert _filter_payload(SimpleNamespace(), {"a": 1}) == {"a": 1}


def test_skill_and_extension_fallbacks() -> None:
    orchestrator = SimpleNamespace(id=uuid4(), name=None, operations=[])
    spec = SimpleNamespace(goal=None, input_type=None, output_type=None)
    skill = _skill_from_spec(0, spec, None, None, orchestrator)
    assert "Avalan orchestrated agent" in skill["description"]

    default_skill = _default_skill(
        None, None, orchestrator, [], [], ["Example"]
    )
    assert "Avalan orchestrated agent" in default_skill["description"]
    assert default_skill["tags"] == ["general"]

    extensions = _capability_extensions([], {"model-b", "model-a"})
    assert extensions == [
        {
            "uri": "https://avalan.ai/extensions/models",
            "description": "Models available to the orchestrated agent.",
            "params": {"models": ["model-a", "model-b"]},
            "required": False,
        }
    ]


def test_coerce_list_exception_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingEvent:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> None:
            raise ValueError("fail")

    monkeypatch.setattr(a2a_types, "TaskEvent", RaisingEvent, raising=False)
    payload = [{"event": "task.created"}]
    assert _coerce_list("TaskEvent", payload) == payload


def test_filter_payload_filters_fields() -> None:
    class DummyModel:
        model_fields = {"allowed": object()}

    filtered = _filter_payload(DummyModel, {"allowed": 1, "extra": 2})
    assert filtered == {"allowed": 1}


def test_coerce_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_coerce_helpers(monkeypatch))


async def _run_coerce_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    store = TaskStore()
    await store.create_task(
        "coerce",
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    task_payload = await store.get_task("coerce")
    events = await store.get_events("coerce")

    class DummyTask:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(id=payload["id"])

    class DummyEvent:
        @classmethod
        def model_validate(cls, payload: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(event=payload["event"])

    monkeypatch.setattr(a2a_types, "Task", DummyTask, raising=False)
    monkeypatch.setattr(a2a_types, "TaskEvent", DummyEvent, raising=False)

    task = _coerce("Task", task_payload)
    assert task.id == "coerce"
    coerced_events = _coerce_list("TaskEvent", events)
    assert coerced_events[0].event == "task.created"

    assert _coerce("Unknown", task_payload) == task_payload
    assert _coerce_list("Unknown", events) == events


def test_create_task_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_create_task_error_paths(monkeypatch))


async def _run_create_task_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.logger = logging.getLogger("test")

    class DummyOrchestrator:
        def __init__(self) -> None:
            self.id = uuid4()
            self.name = "Agent"
            self.model_ids = {"model"}
            self.synced = False

        async def sync_messages(self) -> None:
            self.synced = True

    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator

    async def orchestrate_stub(*args: object, **kwargs: object):
        async def iterator():
            yield "done"

        return iterator(), uuid4(), 0

    router_module = importlib.import_module("avalan.server.a2a.router")
    assert hasattr(router_module, "orchestrate")
    monkeypatch.setattr(router_module, "orchestrate", orchestrate_stub)

    client = TestClient(app)

    response = client.post("/tasks", data="not json", headers={"content-type": "application/json"})
    assert response.status_code == 400

    response = client.post("/tasks", json=[1, 2, 3])
    assert response.status_code == 400

    response = client.post(
        "/tasks",
        json={"jsonrpc": "2.0", "params": []},
    )
    assert response.status_code == 400

    response = client.post(
        "/tasks",
        json={"model": "model", "messages": []},
    )
    assert response.status_code == 400

    response = client.post(
        "/tasks",
        json={"model": "model", "messages": "invalid"},
    )
    assert response.status_code == 422

    response = client.post(
        "/tasks",
        json={"model": "model", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert response.status_code == 200
    task_payload = response.json()
    assert task_payload["status"] == "completed"

    assert orchestrator.synced is True

    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with client.stream("POST", "/tasks", json=payload) as response_stream:
        assert response_stream.status_code == 200
        list(response_stream.iter_bytes())

    task_id = task_payload["id"]
    events_response = client.get(f"/tasks/{task_id}/events")
    assert events_response.status_code == 200
    events_payload = events_response.json()
    assert events_payload

    artifact_response = client.get(f"/tasks/{task_id}/artifacts/answer")
    assert artifact_response.status_code == 200

    task_response = client.get(f"/tasks/{task_id}")
    assert task_response.status_code == 200


def test_additional_router_helpers() -> None:
    asyncio.run(_run_additional_router_helpers())


async def _run_additional_router_helpers() -> None:
    store = TaskStore()
    await store.create_task(
        "helpers",
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    assert _state_for_item(ReasoningToken("thinking")) is StreamState.REASONING
    assert _state_for_item(Event(type=EventType.TOOL_PROCESS, payload={})) is StreamState.TOOL
    assert _state_for_item(Event(type=EventType.TOOL_RESULT, payload={})) is StreamState.TOOL
    assert _state_for_item(Event(type=EventType.START, payload={})) is None
    assert _state_for_item("text") is StreamState.ANSWER
    assert _state_for_item(Token(token="text")) is StreamState.ANSWER
    assert _state_for_item(123) is None

    call = ToolCall(id="call", name="tool", arguments=None)
    assert _call_identifier(ToolCallToken(token="", call=call)) == "call"
    process_event = Event(type=EventType.TOOL_PROCESS, payload={"calls": [call]})
    assert _call_identifier(process_event) == "call"
    process_list_event = Event(type=EventType.TOOL_PROCESS, payload=[call])
    assert _call_identifier(process_list_event) == "call"
    result_event = Event(
        type=EventType.TOOL_RESULT,
        payload={"result": ToolCallResult(id="r", call=call, result={}, name="tool", arguments=None)},
    )
    assert _call_identifier(result_event) == "call"
    assert _call_identifier(Event(type=EventType.TOOL_RESULT, payload={"call": call})) == "call"
    assert _call_identifier(Event(type=EventType.TOOL_RESULT, payload={})) is None

    assert _token_text("plain") == "plain"
    assert _token_text(Token(token="value")) == "value"
    assert _token_text(Event(type=EventType.START, payload={})) == ""


def test_stream_generator_handles_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_stream_generator_handles_cancelled(monkeypatch))


async def _run_stream_generator_handles_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_module = importlib.import_module("avalan.server.a2a.router")

    class DummyStreamingResponse:
        def __init__(self, iterator: AsyncGenerator[str, None], *args: object, **kwargs: object) -> None:
            self.body_iterator = iterator

    async def orchestrate_stub(*args: object, **kwargs: object):
        async def iterator() -> AsyncGenerator[object, None]:
            if False:
                yield None
            return

        return iterator(), uuid4(), 0

    async def canceling_run_stream(self: A2AResponseTranslator, response: object) -> AsyncGenerator[dict[str, object], None]:
        if False:
            yield {}
        raise CancelledError()

    monkeypatch.setattr(router_module, "StreamingResponse", DummyStreamingResponse)
    monkeypatch.setattr(router_module, "orchestrate", orchestrate_stub)
    monkeypatch.setattr(
        A2AResponseTranslator,
        "run_stream",
        canceling_run_stream,
        raising=False,
    )

    class DummyOrchestrator:
        def __init__(self) -> None:
            self.id = uuid4()
            self.name = "Agent"
            self.model_ids = {"model"}
            self.synced = False

        async def sync_messages(self) -> None:
            self.synced = True

    orchestrator = DummyOrchestrator()
    store = TaskStore()

    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    body = json.dumps(payload).encode()
    first_chunk = True

    async def receive() -> dict[str, object]:
        nonlocal first_chunk
        if first_chunk:
            first_chunk = False
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/tasks",
        "raw_path": b"/tasks",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1234),
        "scheme": "http",
        "root_path": "",
        "app": SimpleNamespace(state=SimpleNamespace()),
    }

    request = Request(scope, receive)
    logger = SimpleNamespace(exception=lambda *args, **kwargs: None)

    response = await create_task(
        request,
        logger=logger,
        orchestrator=orchestrator,
        store=store,
    )
    agen: AsyncGenerator[str, None] | None = getattr(response, "body_iterator", None)
    assert agen is not None

    try:
        events = [await agen.__anext__() for _ in range(4)]
        assert "task.stream.completed" in events[2]
        assert "event: done" in events[3]

        with pytest.raises(CancelledError):
            await agen.__anext__()
    finally:
        await agen.aclose()

    assert orchestrator.synced is True


def test_translator_finish_handles_answer_artifact() -> None:
    asyncio.run(_run_translator_finish_handles_answer_artifact())


async def _run_translator_finish_handles_answer_artifact() -> None:
    store = TaskStore()
    await store.create_task(
        "finish",
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator("finish", store)
    artifact_id, _ = await store.ensure_artifact(
        "finish",
        artifact_id="answer-art",
        name="Answer",
        kind="output",
        role=str(MessageRole.ASSISTANT),
    )
    translator._answer_artifact_id = artifact_id
    events = await translator._finish()
    assert translator._answer_artifact_id is None
    assert any(event["event"] == "artifact.completed" for event in events)


def test_artifact_result_additional_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_artifact_result_additional_cases(monkeypatch))


async def _run_artifact_result_additional_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TaskStore()
    task_id = "artifact-cases"
    await store.create_task(
        task_id,
        model="model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    artifact_id, _ = await store.ensure_artifact(
        task_id,
        artifact_id="artifact",
        name="Artifact",
        kind="output",
        role=str(MessageRole.ASSISTANT),
    )
    converter = A2AStreamEventConverter(task_id, store)

    delta_list_event = {
        "event": "artifact.delta",
        "data": {"artifact": {"id": artifact_id, "payload": [{"v": 1}]}}
    }
    delta_list_result = await converter._artifact_result(delta_list_event)
    assert delta_list_result.artifact.artifact_id == artifact_id

    delta_none_event = {
        "event": "artifact.delta",
        "data": {"artifact": {"id": artifact_id, "payload": None}},
    }
    delta_none_result = await converter._artifact_result(delta_none_event)
    assert delta_none_result.artifact.artifact_id == artifact_id

    async def artifact_with_none(
        task: str, art: str
    ) -> dict[str, object]:
        return {
            "id": art,
            "status": "completed",
            "metadata": {},
            "content": None,
            "role": None,
            "kind": None,
        }

    monkeypatch.setattr(store, "get_artifact", artifact_with_none, raising=False)
    converter._artifact_progress.clear()
    completed_event = {"event": "artifact.completed", "data": {"artifact": {"id": artifact_id}}}
    none_content_result = await converter._artifact_result(completed_event)
    assert none_content_result.append is True
    assert none_content_result.artifact.artifact_id == artifact_id

    async def artifact_with_value(
        task: str, art: str
    ) -> dict[str, object]:
        return {
            "id": art,
            "status": "completed",
            "metadata": {},
            "content": "chunk",
            "role": None,
            "kind": None,
        }

    monkeypatch.setattr(store, "get_artifact", artifact_with_value, raising=False)
    converter._artifact_progress.clear()
    value_content_result = await converter._artifact_result(completed_event)
    assert value_content_result.artifact.artifact_id == artifact_id


def test_coerce_handles_plain_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    class PlainTask:
        def __init__(self, id: str) -> None:
            self.id = id

    class PlainEvent:
        def __init__(self, event: str) -> None:
            self.event = event

    monkeypatch.setattr(a2a_types, "Task", PlainTask, raising=False)
    monkeypatch.setattr(a2a_types, "TaskEvent", PlainEvent, raising=False)

    payload = {"id": "plain"}
    event_payload = [{"event": "task.created"}]

    task = _coerce("Task", payload)
    assert isinstance(task, PlainTask) and task.id == "plain"

    events = _coerce_list("TaskEvent", event_payload)
    assert isinstance(events[0], PlainEvent) and events[0].event == "task.created"
