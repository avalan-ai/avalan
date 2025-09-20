from asyncio import Queue
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any, AsyncIterator
from uuid import uuid4

from .schema import TASK_STATUSES, ensure_task_status

from a2a.types.event import Event
from a2a.types.task import TaskStatus


@dataclass(slots=True)
class A2AEvent:
    id: str
    index: int
    timestamp: float
    payload: Event


@dataclass(slots=True)
class A2ATask:
    id: str
    status: TaskStatus
    input_messages: list[dict[str, Any]]
    metadata: dict[str, Any] | None
    created_at: float
    updated_at: float
    events: list[A2AEvent] = field(default_factory=list)
    output_messages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    message_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    next_index: int = 0


class A2ATaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, A2ATask] = {}
        self._subscribers: dict[str, list[Queue[A2AEvent]]] = defaultdict(list)

    def create_task(
        self, input_messages: list[dict[str, Any]], metadata: dict[str, Any] | None
    ) -> A2ATask:
        timestamp = time()
        task = A2ATask(
            id=str(uuid4()),
            status=TASK_STATUSES.created,
            input_messages=list(input_messages),
            metadata=dict(metadata) if metadata else None,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> A2ATask | None:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus | str) -> A2ATask:
        task = self._ensure(task_id)
        task.status = ensure_task_status(status)
        task.updated_at = time()
        return task

    def append_event(self, task_id: str, payload: Event) -> A2AEvent:
        task = self._ensure(task_id)
        event = A2AEvent(
            id=str(uuid4()),
            index=task.next_index,
            timestamp=time(),
            payload=payload,
        )
        task.events.append(event)
        task.next_index += 1
        task.updated_at = event.timestamp
        for queue in self._subscribers.get(task_id, []):
            queue.put_nowait(event)
        return event

    def upsert_artifact(self, task_id: str, artifact: dict[str, Any]) -> None:
        task = self._ensure(task_id)
        identifier = artifact.get("id")
        if not identifier:
            identifier = str(uuid4())
            artifact = {**artifact, "id": identifier}
        task.artifacts[identifier] = artifact
        task.updated_at = time()

    def finalize_output(
        self,
        task_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        task = self._ensure(task_id)
        task.output_messages = messages
        task.updated_at = time()

    async def subscribe(self, task_id: str) -> AsyncIterator[A2AEvent]:
        task = self._ensure(task_id)
        queue: Queue[A2AEvent] = Queue()
        self._subscribers[task_id].append(queue)

        try:
            for event in task.events:
                yield event
            while True:
                yield await queue.get()
        finally:
            subscribers = self._subscribers.get(task_id)
            if subscribers and queue in subscribers:
                subscribers.remove(queue)
                if not subscribers:
                    self._subscribers.pop(task_id, None)

    def _ensure(self, task_id: str) -> A2ATask:
        if task_id not in self._tasks:
            raise KeyError(task_id)
        return self._tasks[task_id]
