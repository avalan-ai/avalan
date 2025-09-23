import asyncio
from datetime import datetime, timezone
import math

from avalan.server.a2a.store import (
    TaskArtifact,
    TaskEvent,
    TaskMessage,
    TaskRecord,
    TaskStore,
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
        kind="tool_call",
        role="assistant",
        metadata={"extra": "value"},
    )
    assert updated_id == artifact_id
    assert reuse_events == []

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
