from asyncio import CancelledError
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_install_a2a_routes_mounts_v1_sdk_routes() -> None:
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
    assert "/a2a/message:stream" in paths
    assert "/.well-known/a2a-agent.json" not in paths


def test_agent_card_uses_v1_supported_interfaces() -> None:
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
async def test_translator_projects_reasoning_tool_and_terminal_states() -> (
    None
):
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
async def test_translator_handles_projection_cancel_error_and_bad_items() -> (
    None
):
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
async def test_executor_cancel_and_exception_paths(monkeypatch) -> None:
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
async def test_executor_cleans_response_on_cancellation(monkeypatch) -> None:
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
async def test_executor_cleans_response_on_stream_error(monkeypatch) -> None:
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


class _FakeEventQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


class _ExecutorOrchestrator:
    model_ids = {"test-model"}
    sync_messages = AsyncMock()


class _ExecutorContext:
    task_id = "task-1"
    context_id = "ctx-1"
    current_task = SimpleNamespace()

    def get_user_input(self) -> str:
        return "hello"


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
