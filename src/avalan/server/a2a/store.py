"""In-memory bookkeeping for A2A tasks and events."""

from asyncio import Lock
from dataclasses import dataclass, field
from json import dumps
from time import time
from typing import Any, Iterable
from uuid import uuid4


def _now() -> float:
    return time()


@dataclass(frozen=True, slots=True)
class TaskStoreRetention:
    """Configure bounded A2A task storage.

    Args:
        max_tasks: Maximum number of task records to retain.
        max_task_age_seconds: Maximum task age, or no age limit.
        max_events_per_task: Maximum event records retained per task.
        max_messages_per_task: Maximum messages retained per task.
        max_artifacts_per_task: Maximum artifacts retained per task.
        max_message_chunks: Maximum text chunks retained per message.
        max_message_bytes: Maximum UTF-8 bytes retained per message.
        max_artifact_items: Maximum payload items retained per artifact.
        max_artifact_bytes: Maximum serialized bytes retained per artifact.

    Returns:
        Retention configuration for the in-memory A2A store.
    """

    max_tasks: int = 1024
    max_task_age_seconds: float | None = None
    max_events_per_task: int = 4096
    max_messages_per_task: int = 256
    max_artifacts_per_task: int = 256
    max_message_chunks: int = 4096
    max_message_bytes: int = 1048576
    max_artifact_items: int = 4096
    max_artifact_bytes: int = 1048576

    def __post_init__(self) -> None:
        assert self.max_tasks > 0
        assert (
            self.max_task_age_seconds is None or self.max_task_age_seconds > 0
        )
        assert self.max_events_per_task > 0
        assert self.max_messages_per_task > 0
        assert self.max_artifacts_per_task > 0
        assert self.max_message_chunks > 0
        assert self.max_message_bytes > 0
        assert self.max_artifact_items > 0
        assert self.max_artifact_bytes > 0


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

    def __init__(self, *, retention: TaskStoreRetention | None = None) -> None:
        if retention is not None:
            assert isinstance(retention, TaskStoreRetention)
        self._tasks: dict[str, TaskRecord] = {}
        self._sequence = 0
        self._lock = Lock()
        self._retention = retention or TaskStoreRetention()

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
            self._prune_expired_tasks()
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
            self._enforce_task_retention()
        return [event.to_payload(task_id) for event in events]

    async def set_status(
        self, task_id: str, status: str
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            if record.status == status:
                return []
            record.status = status
            record.updated_at = _now()
            event = self._append_event(
                record, "task.status.changed", {"status": status}
            )
        return [event.to_payload(task_id)]

    async def add_status_event(
        self,
        task_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            payload: dict[str, Any] = {"status": status}
            if metadata:
                metadata_payload = {
                    key: value
                    for key, value in metadata.items()
                    if value is not None
                }
                if metadata_payload:
                    payload["metadata"] = metadata_payload
            event = self._append_event(
                record,
                "task.status.changed",
                payload,
            )
        return [event.to_payload(task_id)]

    async def fail_task(
        self, task_id: str, message: str
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            if record.status == "failed" and record.error == message:
                return []
            events: list[TaskEvent] = []
            now = _now()
            if record.status != "failed":
                record.status = "failed"
                record.completed_at = now
                record.updated_at = now
                events.append(
                    self._append_event(
                        record,
                        "task.status.changed",
                        {"status": "failed"},
                    )
                )
            elif record.completed_at is None:
                record.completed_at = now
            record.error = message
            record.updated_at = record.completed_at or now
            failure_event = self._append_event(
                record,
                "task.failed",
                {
                    "error": message,
                },
            )
            events.append(failure_event)
        return [event.to_payload(task_id) for event in events]

    async def complete_task(self, task_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            record.status = "completed"
            record.completed_at = _now()
            record.updated_at = record.completed_at
            event = self._append_event(
                record,
                "task.status.changed",
                {"status": "completed"},
            )
        return [event.to_payload(task_id)]

    async def cancel_task(self, task_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            if record.status == "canceled":
                return []
            record.status = "canceled"
            record.completed_at = _now()
            record.updated_at = record.completed_at
            event = self._append_event(
                record,
                "task.status.changed",
                {"status": "canceled"},
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
            record = self._record(task_id)
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
            self._trim_record_messages(record)
        return identifier, [event.to_payload(task_id)]

    async def add_message_delta(
        self,
        task_id: str,
        message_id: str,
        delta: str,
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            message = record.messages[message_id]
            message.append(delta)
            self._trim_message(message)
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
            record = self._record(task_id)
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
            record = self._record(task_id)
            if artifact_id in record.artifacts:
                artifact = record.artifacts[artifact_id]
                if name and not artifact.name:
                    artifact.name = name
                artifact.kind = kind
                artifact.role = role
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
            self._trim_record_artifacts(record)
        return artifact_id, [event.to_payload(task_id)]

    async def add_artifact_delta(
        self, task_id: str, artifact_id: str, payload: Any
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
            artifact = record.artifacts[artifact_id]
            artifact.append(payload)
            self._trim_artifact(artifact)
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
            record = self._record(task_id)
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
            record = self._record(task_id)
            return record.to_payload()

    async def get_events(
        self, task_id: str, *, after: int | None = None
    ) -> list[dict[str, Any]]:
        async with self._lock:
            record = self._record(task_id)
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
            record = self._record(task_id)
            artifact = record.artifacts[artifact_id]
            return artifact.to_payload()

    async def get_message_payload(
        self, task_id: str, message_id: str
    ) -> dict[str, Any]:
        async with self._lock:
            record = self._record(task_id)
            message = record.messages[message_id]
            return message.to_payload()

    async def get_task_overview(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._record(task_id)
            return {
                "id": record.id,
                "status": record.status,
                "model": record.model,
                "instructions": record.instructions,
                "metadata": dict(record.metadata),
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "completed_at": record.completed_at,
                "error": record.error,
            }

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
        self._trim_events(record)
        record.updated_at = payload.created_at
        return payload

    def _record(self, task_id: str) -> TaskRecord:
        self._prune_expired_tasks()
        return self._tasks[task_id]

    def _prune_expired_tasks(self) -> None:
        age = self._retention.max_task_age_seconds
        if age is None:
            return
        cutoff = _now() - age
        expired_ids = [
            task_id
            for task_id, record in self._tasks.items()
            if record.updated_at < cutoff
        ]
        for task_id in expired_ids:
            del self._tasks[task_id]

    def _enforce_task_retention(self) -> None:
        while len(self._tasks) > self._retention.max_tasks:
            oldest_task_id = min(
                self._tasks,
                key=lambda task_id: self._tasks[task_id].updated_at,
            )
            del self._tasks[oldest_task_id]

    def _trim_events(self, record: TaskRecord) -> None:
        overflow = len(record.events) - self._retention.max_events_per_task
        if overflow > 0:
            del record.events[:overflow]

    def _trim_record_messages(self, record: TaskRecord) -> None:
        while (
            len(record.message_order) > self._retention.max_messages_per_task
        ):
            message_id = record.message_order.pop(0)
            record.messages.pop(message_id, None)

    def _trim_record_artifacts(self, record: TaskRecord) -> None:
        while (
            len(record.artifact_order) > self._retention.max_artifacts_per_task
        ):
            artifact_id = record.artifact_order.pop(0)
            record.artifacts.pop(artifact_id, None)

    def _trim_message(self, message: TaskMessage) -> None:
        while len(message.content) > self._retention.max_message_chunks:
            message.content.pop(0)
        while _text_size(message.content) > self._retention.max_message_bytes:
            if len(message.content) == 1:
                message.content[0] = _trim_text_to_bytes(
                    message.content[0], self._retention.max_message_bytes
                )
                break
            message.content.pop(0)

    def _trim_artifact(self, artifact: TaskArtifact) -> None:
        while len(artifact.content) > self._retention.max_artifact_items:
            artifact.content.pop(0)
        while (
            _payload_size(artifact.content)
            > self._retention.max_artifact_bytes
        ):
            if len(artifact.content) == 1:
                artifact.content[0] = _trim_payload_to_bytes(
                    artifact.content[0], self._retention.max_artifact_bytes
                )
                break
            artifact.content.pop(0)


def _text_size(chunks: Iterable[str]) -> int:
    return sum(len(chunk.encode("utf-8")) for chunk in chunks)


def _trim_text_to_bytes(value: str, byte_limit: int) -> str:
    assert isinstance(value, str)
    assert byte_limit > 0
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    return encoded[-byte_limit:].decode("utf-8", errors="ignore")


def _payload_size(value: Any) -> int:
    return len(
        dumps(value, default=str, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    )


def _trim_payload_to_bytes(value: Any, byte_limit: int) -> Any:
    assert byte_limit > 0
    if isinstance(value, str):
        return _trim_text_to_bytes(value, byte_limit)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            trimmed = dict(value)
            trimmed["text"] = _trim_text_to_bytes(text, byte_limit)
            if _payload_size(trimmed) <= byte_limit:
                return trimmed
    return _trim_text_to_bytes(
        dumps(value, default=str, ensure_ascii=False, sort_keys=True),
        byte_limit,
    )
