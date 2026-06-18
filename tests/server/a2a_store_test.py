import asyncio
import math
from datetime import datetime, timezone

import pytest

from avalan.model.stream import StreamRetentionPolicy
from avalan.server.a2a.store import (
    TaskArtifact,
    TaskEvent,
    TaskMessage,
    TaskRecord,
    TaskStore,
    TaskStoreRetention,
    _payload_size,
    _trim_event_data_to_bytes,
    _trim_payload_to_bytes,
)


def test_task_entities_payload_serialization() -> None:
    message = TaskMessage(id="msg", role="assistant", channel="output")
    message.append("Hello, ")
    message.append("world!")
    message.complete()

    artifact = TaskArtifact(
        id="art", name=None, kind="log", role="assistant", metadata={}
    )
    artifact.append({"type": "text", "text": "chunk"})
    artifact.complete()

    event = TaskEvent(
        id="evt",
        sequence=1,
        event="custom",
        created_at=datetime.now(tz=timezone.utc).timestamp(),
        data={"payload": True},
    )

    record = TaskRecord(
        id="task",
        status="accepted",
        model="model-x",
        instructions="Do it",
        input_messages=[{"role": "user", "content": "Hi"}],
        metadata={"foo": "bar"},
    )
    record.messages[message.id] = message
    record.message_order.append(message.id)
    record.artifacts[artifact.id] = artifact
    record.artifact_order.append(artifact.id)
    record.events.append(event)

    message_payload = message.to_payload()
    assert message_payload["content"][0]["text"] == "Hello, world!"
    assert math.isclose(message_payload["updated_at"], message.updated_at)

    artifact_payload = artifact.to_payload()
    assert artifact_payload["content"][0]["text"] == "chunk"

    event_payload = event.to_payload("task")
    assert event_payload["task_id"] == "task"

    record_payload = record.to_payload()
    assert record_payload["messages"][0]["id"] == "msg"
    assert record_payload["artifacts"][0]["id"] == "art"


def test_task_store_covers_all_branches() -> None:
    asyncio.run(_exercise_task_store())


def test_task_store_rejects_invalid_retention() -> None:
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_tasks=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_tasks=True)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_tasks="1")  # type: ignore[arg-type]

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_task_age_seconds=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_task_age_seconds=True)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_task_age_seconds="1")  # type: ignore[arg-type]

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_events_per_task=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_events_per_task=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_event_payload_bytes=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_event_payload_bytes=1)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_event_payload_bytes=True)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_event_payload_bytes="2")  # type: ignore[arg-type]

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_messages_per_task=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_messages_per_task=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifacts_per_task=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifacts_per_task=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_message_chunks=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_message_chunks=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_message_bytes=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_message_bytes=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifact_items=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifact_items=True)

    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifact_bytes=0)
    with pytest.raises(AssertionError):
        TaskStoreRetention(max_artifact_bytes=True)

    with pytest.raises(AssertionError):
        TaskStore(retention="bad")  # type: ignore[arg-type]


def test_task_store_default_retention_uses_stream_policy() -> None:
    retention = TaskStoreRetention()
    policy = StreamRetentionPolicy()

    assert retention.max_tasks == policy.a2a_task_record_item_limit
    assert (
        retention.max_event_payload_bytes == policy.a2a_task_event_byte_limit
    )


def test_repeated_default_task_requests_bound_records() -> None:
    asyncio.run(_exercise_repeated_default_task_requests())


async def _exercise_repeated_default_task_requests() -> None:
    retention = TaskStoreRetention()
    store = TaskStore()

    for index in range(retention.max_tasks + 2):
        task_id = f"task-{index}"
        await store.create_task(
            task_id,
            model=None,
            instructions=None,
            input_messages=[],
        )
        await store.complete_task(task_id)

    assert len(store._tasks) == retention.max_tasks
    with pytest.raises(KeyError):
        await store.get_task("task-0")

    latest = await store.get_task(f"task-{retention.max_tasks + 1}")
    assert latest["status"] == "completed"


async def _exercise_task_store() -> None:
    store = TaskStore()

    initial_events = await store.create_task(
        "task",
        model="model-x",
        instructions="Follow the plan",
        input_messages=[{"role": "user", "content": "start"}],
        metadata={"foo": "bar"},
    )
    assert {event["event"] for event in initial_events} == {
        "task.created",
        "task.status.changed",
    }

    no_change_events = await store.set_status("task", "accepted")
    assert no_change_events == []

    progress_events = await store.set_status("task", "in_progress")
    assert progress_events[0]["data"]["status"] == "in_progress"

    status_events = await store.add_status_event(
        "task",
        status="working",
        metadata={"phase": "step", "skip": None},
    )
    status_payload = status_events[0]["data"].get("metadata")
    assert status_payload == {"phase": "step"}

    failure_events = await store.fail_task("task", "boom")
    failure_names = [event["event"] for event in failure_events]
    assert failure_names[-1] == "task.failed"
    failed_overview = await store.get_task_overview("task")
    assert failed_overview["status"] == "failed"
    assert failed_overview["error"] == "boom"
    assert failed_overview["completed_at"] is not None
    assert await store.fail_task("task", "boom") == []

    await store.create_task(
        "pre-failed",
        model=None,
        instructions=None,
        input_messages=[],
    )
    await store.set_status("pre-failed", "failed")
    pre_failed_events = await store.fail_task("pre-failed", "late failure")
    assert pre_failed_events[-1]["event"] == "task.failed"
    pre_failed_overview = await store.get_task_overview("pre-failed")
    assert pre_failed_overview["error"] == "late failure"
    assert pre_failed_overview["completed_at"] is not None

    message_id, message_created_events = await store.ensure_message(
        "task", role="assistant", channel="output"
    )
    assert message_created_events[0]["event"] == "message.created"

    _, duplicate_events = await store.ensure_message(
        "task", message_id=message_id, role="assistant", channel="output"
    )
    assert duplicate_events == []

    delta_events = await store.add_message_delta("task", message_id, "chunk")
    assert delta_events[0]["data"]["message"]["delta"] == "chunk"

    complete_events = await store.complete_message("task", message_id)
    assert complete_events[0]["event"] == "message.completed"

    assert await store.complete_message("task", message_id) == []

    artifact_id, artifact_created_events = await store.ensure_artifact(
        "task",
        artifact_id="artifact-1",
        name=None,
        kind="tool_call",
        role="assistant",
        metadata={"existing": True},
    )
    assert artifact_created_events[0]["event"] == "artifact.created"

    updated_id, reuse_events = await store.ensure_artifact(
        "task",
        artifact_id=artifact_id,
        name="Tool",
        kind="tool_execution",
        role="tool",
        metadata={"extra": "value"},
    )
    assert updated_id == artifact_id
    assert reuse_events == []
    reused_artifact = await store.get_artifact("task", artifact_id)
    assert reused_artifact["name"] == "Tool"
    assert reused_artifact["kind"] == "tool_execution"
    assert reused_artifact["role"] == "tool"
    assert reused_artifact["metadata"] == {
        "existing": True,
        "extra": "value",
    }
    duplicate_id, duplicate_events = await store.ensure_artifact(
        "task",
        artifact_id=artifact_id,
        name="Ignored",
        kind="tool_execution",
        role="tool",
    )
    assert duplicate_id == artifact_id
    assert duplicate_events == []
    reused_artifact = await store.get_artifact("task", artifact_id)
    assert reused_artifact["name"] == "Tool"

    artifact_delta = await store.add_artifact_delta(
        "task", artifact_id, {"type": "text", "text": "result"}
    )
    assert artifact_delta[0]["event"] == "artifact.delta"

    artifact_complete = await store.complete_artifact("task", artifact_id)
    assert artifact_complete[0]["event"] == "artifact.completed"

    assert await store.complete_artifact("task", artifact_id) == []

    task_payload = await store.get_task("task")
    assert task_payload["status"] == "failed"

    events = await store.get_events("task")
    assert events
    last_sequence = events[-1]["sequence"]
    assert await store.get_events("task", after=last_sequence) == []

    artifact_payload = await store.get_artifact("task", artifact_id)
    assert artifact_payload["metadata"]["extra"] == "value"

    message_payload = await store.get_message_payload("task", message_id)
    assert message_payload["content"][0]["text"] == "chunk"

    overview = await store.get_task_overview("task")
    assert overview["error"] == "boom"

    await store.create_task(
        "cancel-task",
        model=None,
        instructions=None,
        input_messages=[],
    )
    cancel_events = await store.cancel_task("cancel-task")
    assert cancel_events[0]["data"]["status"] == "canceled"
    assert await store.cancel_task("cancel-task") == []
    canceled_overview = await store.get_task_overview("cancel-task")
    assert canceled_overview["status"] == "canceled"
    assert canceled_overview["completed_at"] is not None


def test_task_store_bounds_records_and_histories() -> None:
    asyncio.run(_exercise_bounded_store())


def test_retained_event_payload_bytes_do_not_truncate_live_event() -> None:
    asyncio.run(_exercise_event_payload_byte_retention())


async def _exercise_bounded_store() -> None:
    retention = TaskStoreRetention(
        max_tasks=1,
        max_events_per_task=3,
        max_event_payload_bytes=64,
        max_messages_per_task=1,
        max_artifacts_per_task=1,
        max_message_chunks=2,
        max_message_bytes=6,
        max_artifact_items=2,
        max_artifact_bytes=64,
    )
    store = TaskStore(retention=retention)

    await store.create_task(
        "old", model=None, instructions=None, input_messages=[]
    )
    await store.create_task(
        "task", model=None, instructions=None, input_messages=[]
    )
    with pytest.raises(KeyError):
        await store.get_task("old")

    message_one, _ = await store.ensure_message(
        "task", message_id="message-1", role="assistant", channel="output"
    )
    message_two, _ = await store.ensure_message(
        "task", message_id="message-2", role="assistant", channel="output"
    )
    assert message_one == "message-1"
    with pytest.raises(KeyError):
        await store.add_message_delta("task", message_one, "gone")

    await store.add_message_delta("task", message_two, "abcdef")
    await store.add_message_delta("task", message_two, "ghij")
    message = await store.get_message_payload("task", message_two)
    assert message["content"][0]["text"] == "ghij"
    assert len(message["content"][0]["text"].encode("utf-8")) <= 6

    artifact_one, _ = await store.ensure_artifact(
        "task",
        artifact_id="artifact-1",
        name=None,
        kind="tool",
        role="assistant",
    )
    artifact_two, _ = await store.ensure_artifact(
        "task",
        artifact_id="artifact-2",
        name=None,
        kind="tool",
        role="assistant",
    )
    assert artifact_one == "artifact-1"
    with pytest.raises(KeyError):
        await store.add_artifact_delta("task", artifact_one, "gone")

    await store.add_artifact_delta(
        "task", artifact_two, {"type": "text", "text": "one"}
    )
    await store.add_artifact_delta(
        "task", artifact_two, {"type": "text", "text": "two"}
    )
    await store.add_artifact_delta(
        "task", artifact_two, {"type": "text", "text": "three"}
    )
    artifact = await store.get_artifact("task", artifact_two)
    assert len(artifact["content"]) == 1
    assert artifact["content"][0]["text"] == "three"

    await store.set_status("task", "working")
    await store.add_status_event("task", status="still-working")
    events = await store.get_events("task")
    assert len(events) == 3
    assert events == await store.get_events("task", after=0)
    assert all(
        _payload_size(event["data"]) <= retention.max_event_payload_bytes
        for event in events
    )

    task = await store.get_task("task")
    assert [message["id"] for message in task["messages"]] == ["message-2"]
    assert [artifact["id"] for artifact in task["artifacts"]] == ["artifact-2"]


async def _exercise_event_payload_byte_retention() -> None:
    retention = TaskStoreRetention(max_event_payload_bytes=64)
    store = TaskStore(retention=retention)
    large_text = "x" * 512

    await store.create_task(
        "task", model=None, instructions=None, input_messages=[]
    )
    artifact_id, _ = await store.ensure_artifact(
        "task",
        artifact_id="artifact",
        name=None,
        kind="tool",
        role="assistant",
    )

    live_events = await store.add_artifact_delta(
        "task",
        artifact_id,
        {"type": "text", "text": large_text},
    )
    live_event = live_events[0]
    live_payload = live_event["data"]["artifact"]["payload"]
    assert live_payload["text"] == large_text

    stored_events = await store.get_events("task")
    stored_event = next(
        event for event in stored_events if event["event"] == "artifact.delta"
    )
    assert stored_event["id"] == live_event["id"]
    assert stored_event["sequence"] == live_event["sequence"]
    assert stored_event["data"] != live_event["data"]
    assert stored_event["data"]["retention"]["truncated"] is True
    assert _payload_size(stored_event["data"]) <= (
        retention.max_event_payload_bytes
    )

    artifact = await store.get_artifact("task", artifact_id)
    assert artifact["content"][0]["text"] == large_text


def test_task_store_bounds_single_oversized_payloads() -> None:
    asyncio.run(_exercise_byte_bounded_payloads())


def test_task_store_retention_helpers_cover_edge_paths() -> None:
    assert _trim_event_data_to_bytes({"payload": "ok"}, 100) == {
        "payload": "ok"
    }
    assert _trim_event_data_to_bytes({"payload": "x" * 100}, 40) == {
        "retention": {"truncated": True}
    }
    assert _trim_event_data_to_bytes({"payload": "x" * 100}, 2) == {}
    assert _trim_payload_to_bytes("abcdef", 3) == "def"
    assert _trim_payload_to_bytes({"text": "abc"}, 20) == {"text": "abc"}
    asyncio.run(_exercise_count_bounded_payloads())


async def _exercise_count_bounded_payloads() -> None:
    store = TaskStore(
        retention=TaskStoreRetention(
            max_message_chunks=1,
            max_message_bytes=100,
            max_artifact_items=1,
            max_artifact_bytes=1000,
        )
    )

    await store.create_task(
        "task", model=None, instructions=None, input_messages=[]
    )
    message_id, _ = await store.ensure_message(
        "task", role="assistant", channel="output"
    )
    await store.add_message_delta("task", message_id, "one")
    await store.add_message_delta("task", message_id, "two")
    message = await store.get_message_payload("task", message_id)
    assert message["content"][0]["text"] == "two"

    artifact_id, _ = await store.ensure_artifact(
        "task",
        artifact_id="artifact",
        name=None,
        kind="tool",
        role="assistant",
    )
    await store.add_artifact_delta(
        "task", artifact_id, {"type": "text", "text": "one"}
    )
    await store.add_artifact_delta(
        "task", artifact_id, {"type": "text", "text": "two"}
    )
    artifact = await store.get_artifact("task", artifact_id)
    assert artifact["content"] == [{"type": "text", "text": "two"}]


async def _exercise_byte_bounded_payloads() -> None:
    store = TaskStore(
        retention=TaskStoreRetention(
            max_message_bytes=4,
            max_artifact_bytes=8,
        )
    )

    await store.create_task(
        "task", model=None, instructions=None, input_messages=[]
    )
    message_id, _ = await store.ensure_message(
        "task", role="assistant", channel="output"
    )
    await store.add_message_delta("task", message_id, "abcdef")
    message = await store.get_message_payload("task", message_id)
    assert message["content"][0]["text"] == "cdef"

    artifact_id, _ = await store.ensure_artifact(
        "task",
        artifact_id="artifact",
        name=None,
        kind="tool",
        role="assistant",
    )
    await store.add_artifact_delta(
        "task", artifact_id, {"type": "text", "text": "abcdef"}
    )
    artifact = await store.get_artifact("task", artifact_id)
    assert len(str(artifact["content"][0]).encode("utf-8")) <= 8


def test_task_store_prunes_expired_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_exercise_task_ttl(monkeypatch))


async def _exercise_task_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    import avalan.server.a2a.store as store_module

    now = 1000.0
    monkeypatch.setattr(store_module, "_now", lambda: now)
    store = TaskStore(
        retention=TaskStoreRetention(max_tasks=4, max_task_age_seconds=10)
    )

    await store.create_task(
        "old", model=None, instructions=None, input_messages=[]
    )
    now = 1011.0
    with pytest.raises(KeyError):
        await store.get_task("old")

    await store.create_task(
        "new", model=None, instructions=None, input_messages=[]
    )

    assert (await store.get_task_overview("new"))["id"] == "new"
