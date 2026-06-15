import asyncio
import importlib
import logging
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from a2a import types as a2a_types
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import raises

from avalan.entities import (
    ReasoningToken,
    Token,
    ToolCall,
    ToolCallResult,
    ToolCallToken,
)
from avalan.event import Event, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    A2AStreamEventConverter,
    _di_get_orchestrator,
    well_known_router,
)
from avalan.server.a2a.router import (
    router as a2a_router,
)
from avalan.server.a2a.store import TaskStore

a2a_router_module = importlib.import_module("avalan.server.a2a.router")


class CleanupTrackingResponse:
    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._index = 0
        self.cancel_count = 0
        self.close_count = 0

    def __aiter__(self) -> "CleanupTrackingResponse":
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1


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


def test_translator_streams_canonical_items() -> None:
    asyncio.run(_run_canonical_translator_flow())


def test_translator_preserves_canonical_tool_ready_payload() -> None:
    asyncio.run(_run_canonical_tool_ready_payload_flow())


def test_translator_ignores_malformed_canonical_tool_ready_payload() -> None:
    asyncio.run(_run_malformed_canonical_tool_ready_payload_flow())


def test_token_text_ignores_canonical_control_item() -> None:
    item = CanonicalStreamItem(
        stream_session_id="s",
        run_id="r",
        turn_id="t",
        sequence=0,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )

    assert a2a_router_module._token_text(item) == ""


async def _run_canonical_translator_flow() -> None:
    store = TaskStore()
    task_id = "task-canonical"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="plan",
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta='{"x"',
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.TOOL_EXECUTION_STARTED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            metadata={"tool_name": "lookup"},
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=6,
            kind=StreamItemKind.TOOL_EXECUTION_OUTPUT,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
            text_delta="result",
            data={"category": "stdout", "content": "result"},
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=7,
            kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=8,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=9,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=10,
            kind=StreamItemKind.USAGE_COMPLETED,
            channel=StreamChannel.USAGE,
            usage={},
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=11,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    text = await translator.consume(stream())
    task = await store.get_task(task_id)
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}

    assert text == "answer"
    assert artifacts["reasoning"]["content"][0]["text"] == "plan"
    assert artifacts["answer"]["content"][0]["text"] == "answer"
    assert artifacts["call-1"]["kind"] == "tool_execution"
    assert artifacts["call-1"]["content"][0]["text"] == '{"x"'
    assert artifacts["call-1"]["content"][1]["text"] == "result"
    assert artifacts["call-1"]["content"][2]["status"] == "completed"
    assert artifacts["call-1"]["metadata"]["channel"] == "tool_execution"
    assert artifacts["call-1"]["metadata"]["tool_name"] == "lookup"
    events = await store.get_events(task_id)
    assert any(
        event["data"].get("metadata", {}).get("phase") == "tool_processing"
        for event in events
        if event["event"] == "task.status.changed"
    )
    assert any(
        event["data"].get("metadata", {}).get("phase")
        == "tool_execution.output"
        for event in events
        if event["event"] == "task.status.changed"
    )


async def _run_canonical_tool_ready_payload_flow() -> None:
    store = TaskStore()
    task_id = "task-ready-payload"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-ready"),
            data={
                "name": "lookup",
                "arguments": {"query": "docs"},
            },
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-ready"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    await translator.consume(stream())
    task = await store.get_task(task_id)
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    artifact = artifacts["call-ready"]

    assert artifact["name"] == "lookup"
    assert artifact["kind"] == "tool_call"
    assert artifact["metadata"] == {
        "channel": "tool_call",
        "tool_call_id": "call-ready",
        "tool_name": "lookup",
        "arguments": {"query": "docs"},
    }
    assert artifact["content"] == [
        {"type": "arguments", "arguments": {"query": "docs"}}
    ]
    assert artifact["state"] == "completed"


async def _run_malformed_canonical_tool_ready_payload_flow() -> None:
    store = TaskStore()
    task_id = "task-ready-malformed"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-ready"),
            data="bad",
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="call-ready"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    await translator.consume(stream())
    task = await store.get_task(task_id)
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    artifact = artifacts["call-ready"]

    assert artifact["name"] is None
    assert artifact["kind"] == "tool_call"
    assert artifact["metadata"] == {
        "channel": "tool_call",
        "tool_call_id": "call-ready",
    }
    assert artifact["content"] == []
    assert artifact["state"] == "completed"


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
    reasoning_last_chunks: list[str] = []
    answer_last_chunks: list[str] = []
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
        if not isinstance(result, a2a_types.TaskArtifactUpdateEvent):
            continue
        parts = result.artifact.parts
        assert parts, "expected at least one part"
        part = parts[0]
        text = getattr(part.root, "text", None)
        assert isinstance(text, str)
        if result.artifact.artifact_id == "reasoning":
            if result.last_chunk:
                reasoning_last_chunks.append(text)
            elif result.append:
                reasoning_chunks.append(text)
        elif result.artifact.artifact_id == "answer":
            if result.last_chunk:
                answer_last_chunks.append(text)
            elif result.append:
                answer_chunks.append(text)

    assert reasoning_chunks == ["first", "second"]
    assert answer_chunks == ["A", "B"]
    assert reasoning_last_chunks == ["second"]
    assert answer_last_chunks == ["B"]


def test_tool_events_emit_status_updates() -> None:
    asyncio.run(_run_tool_status_flow())


async def _run_tool_status_flow() -> None:
    store = TaskStore()
    task_id = "task-tool-status"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )

    translator = A2AResponseTranslator(task_id, store)
    converter = A2AStreamEventConverter(task_id, store)

    base_call = ToolCall(
        id="call-status", name="echo", arguments={"text": "hi"}
    )
    tool_result = ToolCallResult(
        id="result-status",
        call=base_call,
        result="ok",
        name=base_call.name,
        arguments=base_call.arguments,
    )

    async def stream():
        yield ToolCallToken(token="", call=base_call)
        yield Event(
            type=EventType.TOOL_RESULT, payload={"result": tool_result}
        )
        yield Token(token="done")

    status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
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
        if isinstance(result, a2a_types.TaskStatusUpdateEvent):
            status_updates.append(result)

    assert status_updates
    tool_processing = [
        update
        for update in status_updates
        if update.metadata
        and update.metadata.get("phase") == "tool_processing"
    ]
    assert any(
        update.status.state is a2a_types.TaskState.working
        for update in tool_processing
    )
    tool_completed = [
        update
        for update in status_updates
        if update.metadata and update.metadata.get("phase") == "tool_completed"
    ]
    assert tool_completed
    for update in tool_completed:
        assert update.status.state is a2a_types.TaskState.working
        assert update.final is False
        assert update.metadata
        assert update.metadata["tool_status"] == "success"

    final_update = status_updates[-1]
    assert final_update.status.state is a2a_types.TaskState.completed
    assert final_update.final is True


def test_final_status_update_is_marked_final() -> None:
    asyncio.run(_run_final_status_flow())


async def _run_final_status_flow() -> None:
    store = TaskStore()
    task_id = "task-final-status"
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
        yield Token(token="done")

    status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
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
        if isinstance(result, a2a_types.TaskStatusUpdateEvent):
            status_updates.append(result)

    assert status_updates
    final_update = status_updates[-1]
    assert final_update.status.state is a2a_types.TaskState.completed
    assert final_update.final is True


def test_canonical_terminal_drives_task_state() -> None:
    asyncio.run(_run_canonical_terminal_state_flow())


async def _run_canonical_terminal_state_flow() -> None:
    cases = (
        (
            "terminal-completed",
            StreamItemKind.STREAM_COMPLETED,
            StreamTerminalOutcome.COMPLETED,
            None,
            a2a_types.TaskState.completed,
            "completed",
            None,
        ),
        (
            "terminal-errored",
            StreamItemKind.STREAM_ERRORED,
            StreamTerminalOutcome.ERRORED,
            {"message": "provider failed"},
            a2a_types.TaskState.failed,
            "failed",
            "provider failed",
        ),
        (
            "terminal-canceled",
            StreamItemKind.STREAM_CANCELLED,
            StreamTerminalOutcome.CANCELLED,
            None,
            a2a_types.TaskState.canceled,
            "canceled",
            None,
        ),
    )

    for (
        task_id,
        terminal_kind,
        terminal_outcome,
        terminal_data,
        expected_state,
        expected_status,
        expected_error,
    ) in cases:
        store = TaskStore()
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
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=terminal_kind,
                channel=StreamChannel.CONTROL,
                data=terminal_data,
                usage=(
                    {}
                    if terminal_outcome is StreamTerminalOutcome.COMPLETED
                    else None
                ),
                terminal_outcome=terminal_outcome,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
        async for raw_event in translator.run_stream(stream()):
            converted = await converter.convert(raw_event)
            if not isinstance(converted, dict) or "result" not in converted:
                continue
            response = (
                a2a_types.SendStreamingMessageSuccessResponse.model_validate(
                    converted
                )
            )
            if isinstance(response.result, a2a_types.TaskStatusUpdateEvent):
                status_updates.append(response.result)

        overview = await store.get_task_overview(task_id)
        assert overview["status"] == expected_status
        assert overview["error"] == expected_error
        assert overview["completed_at"] is not None
        assert status_updates[-1].status.state is expected_state
        assert status_updates[-1].final is True


def test_canonical_duplicate_terminal_is_rejected() -> None:
    asyncio.run(_run_duplicate_terminal_flow())


async def _run_duplicate_terminal_flow() -> None:
    store = TaskStore()
    task_id = "terminal-duplicate"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.STREAM_ERRORED,
            channel=StreamChannel.CONTROL,
            data={"message": "late"},
            terminal_outcome=StreamTerminalOutcome.ERRORED,
        )

    try:
        await translator.consume(stream())
    except StreamValidationError as exc:
        assert str(exc) == "duplicate stream terminal item"
    else:
        raise AssertionError("expected duplicate terminal to be rejected")


def test_canonical_content_after_terminal_is_rejected() -> None:
    asyncio.run(_run_late_content_terminal_flow())


async def _run_late_content_terminal_flow() -> None:
    store = TaskStore()
    task_id = "terminal-late-content"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.STREAM_CANCELLED,
            channel=StreamChannel.CONTROL,
            terminal_outcome=StreamTerminalOutcome.CANCELLED,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="late",
        )

    try:
        await translator.consume(stream())
    except StreamValidationError as exc:
        assert (
            str(exc) == "semantic stream item emitted after terminal outcome"
        )
    else:
        raise AssertionError("expected late content to be rejected")
    task = await store.get_task(task_id)
    assert task["artifacts"] == []


def test_canonical_tool_completion_keeps_task_working_until_terminal() -> None:
    asyncio.run(_run_canonical_tool_completion_task_lifecycle_flow())


async def _run_canonical_tool_completion_task_lifecycle_flow() -> None:
    store = TaskStore()
    task_id = "tool-completion-is-not-task-completion"
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
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.TOOL_EXECUTION_STARTED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
            channel=StreamChannel.TOOL_EXECUTION,
            correlation=StreamItemCorrelation(tool_call_id="call-1"),
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="done",
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=5,
            kind=StreamItemKind.USAGE_COMPLETED,
            channel=StreamChannel.USAGE,
            usage={},
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=6,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
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
        if isinstance(result, a2a_types.TaskStatusUpdateEvent):
            status_updates.append(result)

    tool_updates = [
        update
        for update in status_updates
        if update.metadata
        and update.metadata.get("phase") == "tool_execution.completed"
    ]
    assert tool_updates
    assert tool_updates[-1].status.state is a2a_types.TaskState.working
    assert tool_updates[-1].final is False
    assert tool_updates[-1].metadata
    assert tool_updates[-1].metadata["tool_execution_status"] == "completed"

    assert status_updates[-1].status.state is a2a_types.TaskState.completed
    assert status_updates[-1].final is True
    task = await store.get_task(task_id)
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["call-1"]["kind"] == "tool_execution"


def test_canonical_stream_missing_terminal_is_rejected() -> None:
    asyncio.run(_run_missing_terminal_flow())


def test_canonical_stream_rejection_closes_open_answer_artifact() -> None:
    asyncio.run(_run_rejected_stream_closes_answer_artifact())


def test_stream_cancellation_closes_open_answer_artifact() -> None:
    asyncio.run(_run_cancelled_stream_closes_answer_artifact())


def test_translator_closes_response_after_success() -> None:
    asyncio.run(_run_translator_closes_response_after_success())


def test_translator_closes_response_after_rejection() -> None:
    asyncio.run(_run_translator_closes_response_after_rejection())


def test_translator_preserves_rejection_when_cleanup_fails() -> None:
    asyncio.run(_run_translator_preserves_rejection_when_cleanup_fails())


def test_translator_cancels_response_after_interruption() -> None:
    asyncio.run(_run_translator_cancels_response_after_interruption())


def test_translator_preserves_cancellation_when_cleanup_fails() -> None:
    asyncio.run(_run_translator_preserves_cancellation_when_cleanup_fails())


def test_translator_cancels_response_when_consumer_closes() -> None:
    asyncio.run(_run_translator_cancels_response_when_consumer_closes())


async def _run_missing_terminal_flow() -> None:
    store = TaskStore()
    task_id = "terminal-missing"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="partial",
        )

    emitted_events: list[dict[str, Any]] = []
    try:
        async for event in translator.run_stream(stream()):
            emitted_events.append(event)
    except StreamValidationError as exc:
        assert str(exc) == "stream missing terminal outcome"
    else:
        raise AssertionError("expected missing terminal to be rejected")
    assert any(
        event["event"] == "artifact.completed"
        and event["data"]["artifact"]["id"] == "answer"
        for event in emitted_events
    )
    task = await store.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error"] == "stream missing terminal outcome"
    assert task["completed_at"] is not None
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"
    events = await store.get_events(task_id)
    assert any(
        event["event"] == "artifact.completed"
        and event["data"]["artifact"]["id"] == "answer"
        for event in events
    )
    assert any(
        event["event"] == "task.failed"
        and event["data"]["error"] == "stream missing terminal outcome"
        for event in events
    )


async def _run_rejected_stream_closes_answer_artifact() -> None:
    store = TaskStore()
    task_id = "mixed-stream"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="partial",
        )
        yield Token(token="legacy")

    emitted_events: list[dict[str, Any]] = []
    try:
        async for event in translator.run_stream(stream()):
            emitted_events.append(event)
    except StreamValidationError as exc:
        assert str(exc) == "legacy stream item after canonical stream item"
    else:
        raise AssertionError("expected mixed stream to be rejected")
    assert any(
        event["event"] == "artifact.completed"
        and event["data"]["artifact"]["id"] == "answer"
        for event in emitted_events
    )
    task = await store.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error"] == "legacy stream item after canonical stream item"
    assert task["completed_at"] is not None
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"


async def _run_cancelled_stream_closes_answer_artifact() -> None:
    store = TaskStore()
    task_id = "stream-cancelled"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)

    async def stream():
        yield Token(token="partial")
        raise asyncio.CancelledError()

    emitted_events: list[dict[str, Any]] = []
    try:
        async for event in translator.run_stream(stream()):
            emitted_events.append(event)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected cancellation to propagate")
    assert any(
        event["event"] == "artifact.completed"
        and event["data"]["artifact"]["id"] == "answer"
        for event in emitted_events
    )
    assert any(
        event["event"] == "task.status.changed"
        and event["data"]["status"] == "canceled"
        for event in emitted_events
    )
    assert await store.cancel_task(task_id) == []
    task = await store.get_task(task_id)
    assert task["status"] == "canceled"
    assert task["error"] is None
    assert task["completed_at"] is not None
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"
    assert artifacts["answer"]["content"][0]["text"] == "partial"


async def _run_translator_closes_response_after_success() -> None:
    store = TaskStore()
    task_id = "response-success"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = CleanupTrackingResponse([Token(token="done")])

    await translator.consume(response)

    assert response.cancel_count == 0
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "completed"


async def _run_translator_closes_response_after_rejection() -> None:
    store = TaskStore()
    task_id = "response-rejected"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = CleanupTrackingResponse(
        [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            Token(token="legacy"),
        ]
    )

    with raises(
        StreamValidationError,
        match="legacy stream item after canonical stream item",
    ):
        async for _ in translator.run_stream(response):
            continue

    assert response.cancel_count == 0
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error"] == "legacy stream item after canonical stream item"


async def _run_translator_preserves_rejection_when_cleanup_fails() -> None:
    class FailingCloseResponse(CleanupTrackingResponse):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("close failed")

    store = TaskStore()
    task_id = "response-rejected-cleanup-failed"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = FailingCloseResponse(
        [
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="partial",
            ),
            Token(token="legacy"),
        ]
    )

    with raises(
        StreamValidationError,
        match="legacy stream item after canonical stream item",
    ) as raised:
        async for _ in translator.run_stream(response):
            continue

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "close failed"
    assert response.cancel_count == 0
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "failed"
    assert task["error"] == "legacy stream item after canonical stream item"
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"


async def _run_translator_cancels_response_after_interruption() -> None:
    store = TaskStore()
    task_id = "response-cancelled"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = CleanupTrackingResponse(
        [
            Token(token="partial"),
            asyncio.CancelledError(),
        ]
    )

    with raises(asyncio.CancelledError):
        async for _ in translator.run_stream(response):
            continue

    assert response.cancel_count == 1
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "canceled"


async def _run_translator_preserves_cancellation_when_cleanup_fails() -> None:
    class FailingCancelResponse(CleanupTrackingResponse):
        async def cancel(self) -> None:
            await super().cancel()
            raise RuntimeError("cancel failed")

    store = TaskStore()
    task_id = "response-cancelled-cleanup-failed"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = FailingCancelResponse(
        [
            Token(token="partial"),
            asyncio.CancelledError(),
        ]
    )

    with raises(asyncio.CancelledError) as raised:
        async for _ in translator.run_stream(response):
            continue

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "cancel failed"
    assert response.cancel_count == 1
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "canceled"
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"
    assert artifacts["answer"]["content"][0]["text"] == "partial"


async def _run_translator_cancels_response_when_consumer_closes() -> None:
    store = TaskStore()
    task_id = "response-closed"
    await store.create_task(
        task_id,
        model="test",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    response = CleanupTrackingResponse(
        [
            Token(token="partial"),
            Token(token="late"),
        ]
    )
    stream = translator.run_stream(response)
    try:
        while True:
            event = await anext(stream)
            if event["event"] == "artifact.delta":
                break
    finally:
        await stream.aclose()

    assert response.cancel_count == 1
    assert response.close_count == 1
    task = await store.get_task(task_id)
    assert task["status"] == "canceled"
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert artifacts["answer"]["state"] == "completed"
    assert artifacts["answer"]["content"][0]["text"] == "partial"


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
