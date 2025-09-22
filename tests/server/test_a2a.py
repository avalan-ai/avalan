import asyncio
import importlib
import logging
from types import SimpleNamespace
from uuid import uuid4

from a2a import types as a2a_types
from fastapi import FastAPI
from fastapi.testclient import TestClient
from avalan.entities import (
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    A2AStreamEventConverter,
    _di_get_orchestrator,
    router as a2a_router,
    well_known_router,
)
from avalan.server.a2a.store import TaskStore

a2a_router_module = importlib.import_module("avalan.server.a2a.router")


def test_translator_updates_task_store() -> None:
    asyncio.run(_run_translator_flow())


async def _run_translator_flow() -> None:
    store = TaskStore()
    task_id = "task-1"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    translator = A2AResponseTranslator(task_id, store)

    base_call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    tool_result = ToolCallResult(
        id="result-1",
        call=base_call,
        result="ok",
        name=base_call.name,
        arguments=base_call.arguments,
    )

    async def stream():
        yield ReasoningToken("thinking")
        yield ToolCallToken(token="", call=base_call)
        yield Event(
            type=EventType.TOOL_RESULT, payload={"result": tool_result}
        )
        yield Token(token="hello")

    await translator.consume(stream())

    task = await store.get_task(task_id)
    assert task["status"] == "completed"
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert "reasoning" in artifacts
    assert artifacts["reasoning"]["content"][0]["text"] == "thinking"
    assert "answer" in artifacts
    assert artifacts["answer"]["content"][0]["text"] == "hello"
    assert artifacts["answer"]["state"] == "completed"
    tool_artifact = artifacts["call-1"]
    assert tool_artifact["metadata"]["status"] == "success"
    assert any(
        part.get("type") == "data" and part.get("data") == "ok"
        for part in tool_artifact["content"]
    )

    events = await store.get_events(task_id)
    event_names = {event["event"] for event in events}
    assert "artifact.delta" in event_names
    assert "artifact.completed" in event_names
    assert any(
        event["data"].get("metadata", {}).get("phase") == "tool_processing"
        for event in events
        if event["event"] == "task.status.changed"
    )


def test_artifact_delta_parts_are_incremental() -> None:
    asyncio.run(_run_artifact_delta_parts_flow())


async def _run_artifact_delta_parts_flow() -> None:
    store = TaskStore()
    task_id = "task-delta"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    translator = A2AResponseTranslator(task_id, store)
    converter = A2AStreamEventConverter(task_id, store)

    async def stream():
        yield ReasoningToken("first")
        yield ReasoningToken("second")
        yield Token(token="A")
        yield Token(token="B")

    reasoning_chunks: list[str] = []
    answer_chunks: list[str] = []
    async for raw_event in translator.run_stream(stream()):
        converted = await converter.convert(raw_event)
        if not isinstance(converted, dict) or "result" not in converted:
            continue
        response = (
            a2a_types.SendStreamingMessageSuccessResponse.model_validate(
                converted
            )
        )
        result = response.result
        if not (
            isinstance(result, a2a_types.TaskArtifactUpdateEvent)
            and result.append
        ):
            continue
        parts = result.artifact.parts
        assert parts, "expected at least one part"
        part = parts[0]
        text = getattr(part.root, "text", None)
        assert isinstance(text, str)
        if result.artifact.artifact_id == "reasoning":
            reasoning_chunks.append(text)
        elif result.artifact.artifact_id == "answer":
            answer_chunks.append(text)

    assert reasoning_chunks == ["first", "second"]
    assert answer_chunks == ["A", "B"]


def test_agent_card_uses_configured_tool() -> None:
    orchestrator = SimpleNamespace(
        id=uuid4(),
        name="Test Agent",
        operations=[],
        model_ids={"test-model"},
    )

    app = FastAPI()
    app.include_router(a2a_router)
    app.include_router(well_known_router)

    async def orchestrator_override() -> SimpleNamespace:
        return orchestrator

    app.dependency_overrides[_di_get_orchestrator] = orchestrator_override
    app.state.a2a_tool_name = "execute"
    app.state.a2a_tool_description = "Execute the orchestrated agent"

    client = TestClient(app)

    response = client.get("/agent")
    assert response.status_code == 200
    agent_card = response.json()
    assert agent_card["defaultInputModes"] == ["text/plain"]
    assert agent_card["defaultOutputModes"] == ["text/markdown"]
    assert agent_card["url"].endswith("/tasks")

    capabilities = agent_card["capabilities"]
    assert capabilities["streaming"] is True
    assert capabilities["stateTransitionHistory"] is True
    extensions = capabilities.get("extensions")
    assert extensions is not None
    assert any(
        extension["uri"] == "https://avalan.ai/extensions/models"
        for extension in extensions
    )

    skills = agent_card["skills"]
    assert len(skills) == 1
    skill = skills[0]
    assert skill["name"] == "execute"
    assert skill["description"].startswith("Execute the orchestrated agent")
    assert skill["inputModes"] == ["text/plain"]
    assert skill["outputModes"] == ["text/markdown"]
    assert "execute" in skill["tags"]

    well_known = client.get("/.well-known/a2a-agent.json")
    assert well_known.status_code == 200
    assert well_known.json()["skills"] == agent_card["skills"]


def test_create_task_streams_jsonrpc_request(monkeypatch) -> None:
    class StubOrchestrator:
        def __init__(self) -> None:
            self.id = uuid4()
            self.name = "Test Agent"
            self.model_ids = {"test-model"}
            self.synced = False

        async def sync_messages(self) -> None:
            self.synced = True

    async def orchestrate_stub(*_args):
        async def iterator():
            yield "Hello!"

        return iterator(), uuid4(), 0

    app = FastAPI()
    app.include_router(a2a_router)
    app.state.logger = logging.getLogger("test")
    orchestrator = StubOrchestrator()
    app.state.orchestrator = orchestrator

    monkeypatch.setattr(a2a_router_module, "orchestrate", orchestrate_stub)

    client = TestClient(app)
    payload = {
        "id": "msg-123",
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "configuration": {"acceptedOutputModes": ["text/plain"]},
            "message": {
                "kind": "message",
                "messageId": "msg-123",
                "metadata": {},
                "parts": [{"kind": "text", "text": "Ping"}],
                "role": "user",
            },
        },
    }

    with client.stream("POST", "/tasks", json=payload) as response:
        assert response.status_code == 200
        content_type = response.headers["content-type"]
        assert content_type.startswith("text/event-stream")
        body = b"".join(response.iter_bytes())

    text = body.decode("utf-8")
    assert "Hello!" in text
    assert "task.stream.completed" in text
    assert "test-model" in text

    results: list[
        a2a_types.Task
        | a2a_types.Message
        | a2a_types.TaskStatusUpdateEvent
        | a2a_types.TaskArtifactUpdateEvent
    ] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            _, _, data = line.partition("data:")
            payload_text = data.strip()
            if not payload_text:
                continue
            response_model = a2a_types.SendStreamingMessageSuccessResponse.model_validate_json(  # noqa: E501
                payload_text
            )
            results.append(response_model.result)

    assert results
    assert any(isinstance(result, a2a_types.Task) for result in results)
    assert any(
        isinstance(result, a2a_types.TaskArtifactUpdateEvent)
        and result.artifact.artifact_id == "answer"
        for result in results
    )
    assert any(
        isinstance(result, a2a_types.TaskStatusUpdateEvent)
        and result.status.state is a2a_types.TaskState.completed
        for result in results
    )

    assert orchestrator.synced is True
