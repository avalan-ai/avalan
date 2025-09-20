"""In-memory bookkeeping for A2A tasks and events."""

from asyncio import Lock
from dataclasses import dataclass, field
from time import time
from typing import Any, Iterable
from uuid import uuid4


def _now() -> float:
    return time()


@dataclass(slots=True)
class TaskMessage:
    """Represents a message emitted while fulfilling a task."""

    id: str
    role: str
    channel: str
    content: list[str] = field(default_factory=list)
    state: str = "in_progress"
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def append(self, chunk: str) -> None:
        self.content.append(chunk)
        self.updated_at = _now()

    def complete(self) -> None:
        self.state = "completed"
        self.updated_at = _now()

    def text(self) -> str:
        return "".join(self.content)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "channel": self.channel,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": [
                {
                    "type": "text",
                    "text": self.text(),
                }
            ],
        }


@dataclass(slots=True)
class TaskArtifact:
    """Stores tool inputs or outputs captured during a task."""

    id: str
    name: str | None
    kind: str
    role: str
    content: list[Any] = field(default_factory=list)
    state: str = "in_progress"
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, item: Any) -> None:
        self.content.append(item)
        self.updated_at = _now()

    def complete(self) -> None:
        self.state = "completed"
        self.updated_at = _now()

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "content": list(self.content),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TaskEvent:
    """Single event recorded for an A2A task."""

    id: str
    sequence: int
    event: str
    created_at: float
    data: dict[str, Any]

    def to_payload(self, task_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": task_id,
            "sequence": self.sequence,
            "event": self.event,
            "created_at": self.created_at,
            "data": self.data,
        }


@dataclass(slots=True)
class TaskRecord:
    """Aggregates the lifecycle of an A2A task."""

    id: str
    status: str
    model: str | None
    instructions: str | None
    input_messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    completed_at: float | None = None
    error: str | None = None
    messages: dict[str, TaskMessage] = field(default_factory=dict)
    message_order: list[str] = field(default_factory=list)
    artifacts: dict[str, TaskArtifact] = field(default_factory=dict)
    artifact_order: list[str] = field(default_factory=list)
    events: list[TaskEvent] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "model": self.model,
            "instructions": self.instructions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": dict(self.metadata),
            "input": list(self.input_messages),
            "messages": [
                self.messages[msg_id].to_payload()
                for msg_id in self.message_order
            ],
            "artifacts": [
                self.artifacts[artifact_id].to_payload()
                for artifact_id in self.artifact_order
            ],
        }


class TaskStore:
    """Thread-safe, in-memory storage for tasks and related events."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._sequence = 0
        self._lock = Lock()

    async def create_task(
        self,
        task_id: str,
        *,
        model: str | None,
        instructions: str | None,
        input_messages: Iterable[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            now = _now()
            record = TaskRecord(
                id=task_id,
                status="accepted",
                model=model,
                instructions=instructions,
                input_messages=list(input_messages),
                metadata=dict(metadata or {}),
                created_at=now,
                updated_at=now,
            )
            self._tasks[task_id] = record
            events = [
                self._append_event(
                    record,
                    "task.created",
                    {
                        "task": {
                            "id": task_id,
                            "model": model,
                            "created_at": now,
                        }
                    },
                ),
                self._append_event(
                    record, "task.status.changed", {"status": record.status}
                ),
            ]
        return [event.to_payload(task_id) for event in events]

    async def set_status(
        self, task_id: str, status: str
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            if record.status == status:
                return []
            record.status = status
            record.updated_at = _now()
            event = self._append_event(
                record, "task.status.changed", {"status": status}
            )
        return [event.to_payload(task_id)]

    async def fail_task(
        self, task_id: str, message: str
    ) -> list[dict[str, Any]]:
        events = await self.set_status(task_id, "failed")
        async with self._lock:
            record = self._tasks[task_id]
            record.error = message
            failure_event = self._append_event(
                record,
                "task.failed",
                {
                    "error": message,
                },
            )
        events.append(failure_event.to_payload(task_id))
        return events

    async def complete_task(self, task_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            record.status = "completed"
            record.completed_at = _now()
            record.updated_at = record.completed_at
            event = self._append_event(
                record,
                "task.status.changed",
                {"status": "completed"},
            )
        return [event.to_payload(task_id)]

    async def ensure_message(
        self,
        task_id: str,
        *,
        message_id: str | None = None,
        role: str,
        channel: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._lock:
            record = self._tasks[task_id]
            identifier = message_id or str(uuid4())
            if identifier in record.messages:
                return identifier, []
            message = TaskMessage(id=identifier, role=role, channel=channel)
            record.messages[identifier] = message
            record.message_order.append(identifier)
            event = self._append_event(
                record,
                "message.created",
                {
                    "message": message.to_payload(),
                },
            )
        return identifier, [event.to_payload(task_id)]

    async def add_message_delta(
        self,
        task_id: str,
        message_id: str,
        delta: str,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            message = record.messages[message_id]
            message.append(delta)
            record.updated_at = message.updated_at
            event = self._append_event(
                record,
                "message.delta",
                {
                    "message": {
                        "id": message.id,
                        "role": message.role,
                        "channel": message.channel,
                        "delta": delta,
                    }
                },
            )
        return [event.to_payload(task_id)]

    async def complete_message(
        self, task_id: str, message_id: str
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            message = record.messages[message_id]
            if message.state == "completed":
                return []
            message.complete()
            record.updated_at = message.updated_at
            event = self._append_event(
                record,
                "message.completed",
                {
                    "message": message.to_payload(),
                },
            )
        return [event.to_payload(task_id)]

    async def ensure_artifact(
        self,
        task_id: str,
        *,
        artifact_id: str,
        name: str | None,
        kind: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        async with self._lock:
            record = self._tasks[task_id]
            if artifact_id in record.artifacts:
                artifact = record.artifacts[artifact_id]
                if name and not artifact.name:
                    artifact.name = name
                if metadata:
                    artifact.metadata.update(metadata)
                return artifact_id, []
            artifact = TaskArtifact(
                id=artifact_id,
                name=name,
                kind=kind,
                role=role,
                metadata=dict(metadata or {}),
            )
            record.artifacts[artifact_id] = artifact
            record.artifact_order.append(artifact_id)
            event = self._append_event(
                record,
                "artifact.created",
                {
                    "artifact": artifact.to_payload(),
                },
            )
        return artifact_id, [event.to_payload(task_id)]

    async def add_artifact_delta(
        self, task_id: str, artifact_id: str, payload: Any
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            artifact = record.artifacts[artifact_id]
            artifact.append(payload)
            record.updated_at = artifact.updated_at
            event = self._append_event(
                record,
                "artifact.delta",
                {
                    "artifact": {
                        "id": artifact.id,
                        "kind": artifact.kind,
                        "payload": payload,
                    }
                },
            )
        return [event.to_payload(task_id)]

    async def complete_artifact(
        self, task_id: str, artifact_id: str
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            artifact = record.artifacts[artifact_id]
            if artifact.state == "completed":
                return []
            artifact.complete()
            record.updated_at = artifact.updated_at
            event = self._append_event(
                record,
                "artifact.completed",
                {
                    "artifact": artifact.to_payload(),
                },
            )
        return [event.to_payload(task_id)]

    async def get_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._tasks[task_id]
            return record.to_payload()

    async def get_events(
        self, task_id: str, *, after: int | None = None
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._tasks[task_id]
            events = [
                event.to_payload(task_id)
                for event in record.events
                if after is None or event.sequence > after
            ]
        return events

    async def get_artifact(
        self, task_id: str, artifact_id: str
    ) -> dict[str, Any]:
        async with self._lock:
            record = self._tasks[task_id]
            artifact = record.artifacts[artifact_id]
            return artifact.to_payload()

    def _append_event(
        self, record: TaskRecord, event: str, data: dict[str, Any]
    ) -> TaskEvent:
        self._sequence += 1
        payload = TaskEvent(
            id=str(uuid4()),
            sequence=self._sequence,
            event=event,
            created_at=_now(),
            data=data,
        )
        record.events.append(payload)
        record.updated_at = payload.created_at
        return payload
