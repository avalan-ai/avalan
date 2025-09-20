import asyncio
import contextlib
import sys
import types
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest


base_path = Path(__file__).resolve().parents[2] / "src"
ORIGINAL_SYS_PATH = list(sys.path)
sys.path.insert(0, str(base_path))

PREVIOUS_MODULES = {
    "a2a_sdk": sys.modules.get("a2a_sdk"),
    "a2a_sdk.models": sys.modules.get("a2a_sdk.models"),
    "a2a_sdk.models.agent": sys.modules.get("a2a_sdk.models.agent"),
    "a2a_sdk.models.event": sys.modules.get("a2a_sdk.models.event"),
    "a2a_sdk.models.task": sys.modules.get("a2a_sdk.models.task"),
    "avalan": sys.modules.get("avalan"),
    "avalan.server": sys.modules.get("avalan.server"),
    "avalan.agent": sys.modules.get("avalan.agent"),
    "avalan.agent.orchestrator": sys.modules.get("avalan.agent.orchestrator"),
    "avalan.entities": sys.modules.get("avalan.entities"),
}

if "avalan" not in sys.modules:
    package = types.ModuleType("avalan")
    package.__path__ = [str(base_path / "avalan")]
    sys.modules["avalan"] = package

server_package = types.ModuleType("avalan.server")
server_package.__path__ = [str(base_path / "avalan" / "server")]
sys.modules["avalan.server"] = server_package

agent_package = types.ModuleType("avalan.agent")
agent_package.__path__ = [str(base_path / "avalan" / "agent")]
sys.modules["avalan.agent"] = agent_package

orchestrator_stub = types.ModuleType("avalan.agent.orchestrator")


class _Orchestrator:
    pass


orchestrator_stub.Orchestrator = _Orchestrator
sys.modules["avalan.agent.orchestrator"] = orchestrator_stub

entities_stub = types.ModuleType("avalan.entities")


class _Token:
    def __init__(self, token: str) -> None:
        self.token = token


class _TokenDetail(_Token):
    pass


class _ReasoningToken(_Token):
    pass


class _ToolCall:
    def __init__(self, identifier: str, name: str, arguments: str) -> None:
        self.id = identifier
        self.name = name
        self.arguments = arguments


class _ToolCallToken(_Token):
    def __init__(self, token: str, call: _ToolCall | None = None) -> None:
        super().__init__(token)
        self.call = call


class _ToolCallResult:
    def __init__(self, call: _ToolCall, result: Any) -> None:
        self.call = call
        self.result = result
        self.name = call.name
        self.arguments = call.arguments


class _ToolCallError(Exception):
    def __init__(self, message: str, call: _ToolCall | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.call = call
        self.name = call.name if call else ""
        self.arguments = call.arguments if call else ""


entities_stub.Token = _Token
entities_stub.TokenDetail = _TokenDetail
entities_stub.ReasoningToken = _ReasoningToken
entities_stub.ToolCallToken = _ToolCallToken
entities_stub.ToolCallError = _ToolCallError
entities_stub.ToolCallResult = _ToolCallResult
entities_stub.ToolCall = _ToolCall
sys.modules["avalan.entities"] = entities_stub


class _SDKModel:
    def __init__(self, **data: Any) -> None:
        self._data = dict(data)

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "_SDKModel":
        return cls(**payload)

    def model_dump(
        self, *, by_alias: bool = True, exclude_none: bool = True
    ) -> dict[str, Any]:
        if exclude_none:
            return {key: value for key, value in self._data.items() if value is not None}
        return dict(self._data)


class _AgentCard(_SDKModel):
    pass


class _Event(_SDKModel):
    pass


class _Task(_SDKModel):
    pass


class _TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


a2a_sdk_module = types.ModuleType("a2a_sdk")
a2a_sdk_models = types.ModuleType("a2a_sdk.models")
a2a_sdk_agent = types.ModuleType("a2a_sdk.models.agent")
a2a_sdk_event = types.ModuleType("a2a_sdk.models.event")
a2a_sdk_task = types.ModuleType("a2a_sdk.models.task")

a2a_sdk_agent.AgentCard = _AgentCard
a2a_sdk_event.Event = _Event
a2a_sdk_task.Task = _Task
a2a_sdk_task.TaskStatus = _TaskStatus

a2a_sdk_module.models = a2a_sdk_models
a2a_sdk_models.agent = a2a_sdk_agent
a2a_sdk_models.event = a2a_sdk_event
a2a_sdk_models.task = a2a_sdk_task

sys.modules["a2a_sdk"] = a2a_sdk_module
sys.modules["a2a_sdk.models"] = a2a_sdk_models
sys.modules["a2a_sdk.models.agent"] = a2a_sdk_agent
sys.modules["a2a_sdk.models.event"] = a2a_sdk_event
sys.modules["a2a_sdk.models.task"] = a2a_sdk_task

from avalan.entities import ReasoningToken, Token
from avalan.server.a2a.agent import build_agent_card
from avalan.server.a2a.schema import TASK_STATUSES, task_status_value, validate_event
from avalan.server.a2a.store import A2ATaskStore
from avalan.server.a2a.translator import A2ATranslator, event_to_sse


@pytest.fixture(autouse=True, scope="module")
def _restore_modules() -> None:
    yield
    for name, module in PREVIOUS_MODULES.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    sys.path[:] = ORIGINAL_SYS_PATH


def _payload_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    assert isinstance(payload, dict)
    return payload


def test_translator_stream_and_completion() -> None:
    async def _run() -> None:
        store = A2ATaskStore()
        task = store.create_task([{"role": "user", "content": "hi"}], None)

        translator = A2ATranslator(store, task)

        start_events = await translator.start()
        assert task_status_value(task.status) == task_status_value(TASK_STATUSES.running)
        start_payload = _payload_dict(start_events[-1].payload)
        assert start_payload["type"] == "status.changed"

        reasoning_events = await translator.token(ReasoningToken(token="thinking"))
        reasoning_payload = _payload_dict(reasoning_events[-1].payload)
        assert reasoning_payload["delta"]["channel"] == "reasoning"

        output_events = await translator.token(Token(token="hello"))
        output_payload = _payload_dict(output_events[-1].payload)
        assert output_payload["delta"]["channel"] == "output"

        finish_events = await translator.finish()
        assert task_status_value(task.status) == task_status_value(TASK_STATUSES.completed)
        finish_payload = _payload_dict(finish_events[-1].payload)
        assert finish_payload["type"].startswith("task.")
        assert task.output_messages

        sse = event_to_sse(finish_events[-1])
        assert "event:" in sse and "data:" in sse

    asyncio.run(_run())


def test_task_store_subscription() -> None:
    async def _run() -> None:
        store = A2ATaskStore()
        task_id = store.create_task([], None).id
        first = store.append_event(task_id, validate_event({"type": "status.changed"}))

        events: list[Any] = []

        async def consume() -> None:
            async for evt in store.subscribe(task_id):
                events.append(evt)
                if len(events) == 2:
                    break

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.sleep(0)
            second = store.append_event(task_id, validate_event({"type": "status.changed"}))
            await asyncio.wait_for(consumer, timeout=0.2)
        finally:
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

        assert events[0].id == first.id
        assert events[1].id == second.id

    asyncio.run(_run())


def test_agent_card_includes_streaming_capability() -> None:
    orchestrator = types.SimpleNamespace(
        id=uuid4(),
        name=None,
        tool=types.SimpleNamespace(is_empty=True),
        model_ids=set(),
    )

    card = build_agent_card(
        orchestrator,
        name="Avalan Agent",
        version="1.0.0",
        base_url="https://example.com",
        prefix="/a2a",
    )

    assert card["capabilities"]["streaming"] is True
