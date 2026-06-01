from ..event import Event
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from .event import (
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventStore,
    sanitize_raw_task_event_closed,
)
from .privacy import PrivacySanitizer
from .usage import UsageRecord, UsageSource, UsageTotals

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

TaskObservedEvent: TypeAlias = SanitizedTaskEvent | SanitizedTaskEventDraft
TaskSanitizedEventObserver: TypeAlias = Callable[
    [TaskObservedEvent],
    Awaitable[None] | None,
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservabilitySinkHealth:
    name: str
    event_count: int = 0
    usage_count: int = 0
    failure_count: int = 0
    last_failure_code: str | None = None
    children: tuple["ObservabilitySinkHealth", ...] = ()

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        _assert_non_negative_int(self.event_count, "event_count")
        _assert_non_negative_int(self.usage_count, "usage_count")
        _assert_non_negative_int(self.failure_count, "failure_count")
        if self.last_failure_code is not None:
            _assert_non_empty_string(
                self.last_failure_code,
                "last_failure_code",
            )
        assert isinstance(self.children, tuple)
        for child in self.children:
            assert isinstance(child, ObservabilitySinkHealth)

    @property
    def healthy(self) -> bool:
        return self.failure_count == 0 and all(
            child.healthy for child in self.children
        )


class ObservabilitySink(Protocol):
    async def record_event(self, event: TaskObservedEvent) -> None: ...

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None: ...

    def health(self) -> ObservabilitySinkHealth: ...


@dataclass(slots=True, kw_only=True)
class FanoutObservabilitySink(ObservabilitySink):
    sinks: tuple[ObservabilitySink, ...]
    name: str = "fanout"
    _event_count: int = field(default=0, init=False)
    _usage_count: int = field(default=0, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_code: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.name, "name")
        assert isinstance(self.sinks, tuple)
        for sink in self.sinks:
            assert callable(getattr(sink, "record_event", None))
            assert callable(getattr(sink, "record_usage", None))
            assert callable(getattr(sink, "health", None))

    async def record_event(self, event: TaskObservedEvent) -> None:
        self._event_count += 1
        for sink in self.sinks:
            try:
                await sink.record_event(event)
            except Exception as error:
                self._record_failure(error)

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._usage_count += 1
        for sink in self.sinks:
            try:
                await sink.record_usage(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    source=source,
                    totals=totals,
                    metadata=metadata,
                )
            except Exception as error:
                self._record_failure(error)

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=self._event_count,
            usage_count=self._usage_count,
            failure_count=self._failure_count,
            last_failure_code=self._last_failure_code,
            children=tuple(sink.health() for sink in self.sinks),
        )

    def _record_failure(self, error: Exception) -> None:
        self._failure_count += 1
        self._last_failure_code = type(error).__name__


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskEventPipeline:
    store: TaskEventStore
    run_id: str
    sanitizer: PrivacySanitizer
    attempt_id: str | None = None
    capture_events: bool = True
    metrics_observer: TaskSanitizedEventObserver | None = None
    trace_observer: TaskSanitizedEventObserver | None = None
    observability_sink: ObservabilitySink | None = None

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
        if self.observability_sink is not None:
            assert callable(
                getattr(self.observability_sink, "record_event", None)
            )
            assert callable(
                getattr(self.observability_sink, "record_usage", None)
            )
            assert callable(getattr(self.observability_sink, "health", None))

    async def __call__(self, event: Event) -> None:
        draft = sanitize_raw_task_event_closed(event, self.sanitizer)
        observed: TaskObservedEvent = draft
        if self.capture_events:
            try:
                observed = await self.store.append_event(
                    self.run_id,
                    attempt_id=self.attempt_id,
                    category=draft.category,
                    event_type=draft.event_type,
                    payload=draft.payload,
                )
            except Exception:
                observed = draft
        await _notify_observer(self.metrics_observer, observed)
        await _notify_observer(self.trace_observer, observed)
        await record_observability_event(
            self.observability_sink,
            observed,
        )


async def record_observability_event(
    sink: ObservabilitySink | None,
    event: TaskObservedEvent,
) -> None:
    if sink is None:
        return
    try:
        await sink.record_event(event)
    except Exception:
        return


async def record_observability_usage(
    sink: ObservabilitySink | None,
    *,
    run_id: str,
    source: UsageSource,
    totals: UsageTotals,
    attempt_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
    record: UsageRecord | None = None,
) -> None:
    if sink is None:
        return
    try:
        if record is not None:
            record_usage_record = getattr(sink, "record_usage_record", None)
            if callable(record_usage_record):
                await record_usage_record(record)
                return
        await sink.record_usage(
            run_id=run_id,
            attempt_id=attempt_id,
            source=source,
            totals=totals,
            metadata=metadata,
        )
    except Exception:
        return


async def _notify_observer(
    observer: TaskSanitizedEventObserver | None,
    event: TaskObservedEvent,
) -> None:
    if observer is None:
        return
    try:
        result = observer(event)
        if result is not None:
            await result
    except Exception:
        return
