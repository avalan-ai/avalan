from ..event import Event
from .event import (
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventStore,
    sanitize_raw_task_event_closed,
)
from .privacy import PrivacySanitizer

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

TaskObservedEvent: TypeAlias = SanitizedTaskEvent | SanitizedTaskEventDraft
TaskSanitizedEventObserver: TypeAlias = Callable[
    [TaskObservedEvent],
    Awaitable[None] | None,
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskEventPipeline:
    store: TaskEventStore
    run_id: str
    sanitizer: PrivacySanitizer
    attempt_id: str | None = None
    capture_events: bool = True
    metrics_observer: TaskSanitizedEventObserver | None = None
    trace_observer: TaskSanitizedEventObserver | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.sanitizer, PrivacySanitizer)
        assert isinstance(self.capture_events, bool)
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if self.metrics_observer is not None:
            assert callable(self.metrics_observer)
        if self.trace_observer is not None:
            assert callable(self.trace_observer)

    async def __call__(self, event: Event) -> None:
        draft = sanitize_raw_task_event_closed(event, self.sanitizer)
        observed: TaskObservedEvent = draft
        if self.capture_events:
            observed = await self.store.append_event(
                self.run_id,
                attempt_id=self.attempt_id,
                category=draft.category,
                event_type=draft.event_type,
                payload=draft.payload,
            )
        await _notify_observer(self.metrics_observer, observed)
        await _notify_observer(self.trace_observer, observed)


async def _notify_observer(
    observer: TaskSanitizedEventObserver | None,
    event: TaskObservedEvent,
) -> None:
    if observer is None:
        return
    result = observer(event)
    if result is not None:
        await result


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
