from dataclasses import dataclass
from typing import Any, TypeVar

from a2a_sdk.models.agent import AgentCard
from a2a_sdk.models.event import Event
from a2a_sdk.models.task import Task, TaskStatus


@dataclass(frozen=True, slots=True)
class TaskStatusSet:
    created: TaskStatus
    running: TaskStatus
    completed: TaskStatus
    failed: TaskStatus

    def as_tuple(self) -> tuple[TaskStatus, TaskStatus, TaskStatus, TaskStatus]:
        return (self.created, self.running, self.completed, self.failed)


def _status_value(status: TaskStatus | str) -> str:
    if isinstance(status, TaskStatus):
        return status.value.lower()
    return str(status).lower()


TASK_STATUSES = TaskStatusSet(
    created=TaskStatus.CREATED,
    running=TaskStatus.RUNNING,
    completed=TaskStatus.COMPLETED,
    failed=TaskStatus.FAILED,
)


_STATUS_LOOKUP: dict[str, TaskStatus] = {
    _status_value(status): status for status in TASK_STATUSES.as_tuple()
}


def task_status_value(status: TaskStatus | str) -> str:
    """Return a normalized string representation for a task status."""

    return _status_value(status)


def ensure_task_status(status: TaskStatus | str) -> TaskStatus:
    """Return the canonical task status instance for the provided value."""

    if isinstance(status, TaskStatus):
        return status
    normalized = _status_value(status)
    if normalized in _STATUS_LOOKUP:
        return _STATUS_LOOKUP[normalized]
    return TaskStatus(normalized)


def _convert_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _convert_payload(
            value.model_dump(by_alias=True, exclude_none=True)
        )
    if hasattr(value, "dict"):
        return _convert_payload(value.dict(exclude_none=True))  # type: ignore[attr-defined]
    if isinstance(value, dict):
        return {key: _convert_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert_payload(item) for item in value)
    if isinstance(value, set):
        return [_convert_payload(item) for item in value]
    return value


def _dump(payload: Any) -> dict[str, Any]:
    converted = _convert_payload(payload)
    if isinstance(converted, dict):
        return converted
    if hasattr(converted, "items"):
        return dict(converted.items())  # type: ignore[attr-defined]
    return dict(converted)  # pragma: no cover - defensive


T = TypeVar("T")


def _validate(model: type[T], payload: dict[str, Any]) -> T:
    return model.model_validate(payload)


def validate_agent_card(payload: dict[str, Any]) -> AgentCard:
    """Return an Agent Card instance."""

    return _validate(AgentCard, payload)


def validate_task(payload: dict[str, Any]) -> Task:
    """Return a task instance."""

    return _validate(Task, payload)


def validate_event(payload: dict[str, Any]) -> Event:
    """Return an event instance."""

    return _validate(Event, payload)


def dump_payload(payload: Any) -> dict[str, Any]:
    """Return a JSON-serializable representation of an SDK payload."""

    return _dump(payload)
