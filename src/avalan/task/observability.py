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
    SanitizedTaskUsageEvent,
    TaskEventStore,
    sanitize_raw_task_event_closed,
)
from .privacy import PrivacySanitizer
from .store import TaskStoreConflictError
from .usage import (
    TaskUsageStore,
    UsageObservation,
    UsageRecord,
    UsageSource,
    UsageTotals,
    stable_usage_id_for_response,
    usage_flow_node,
    usage_observation_entries_from_response,
)

from asyncio import gather
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Protocol, TypeAlias

TaskObservedEvent: TypeAlias = (
    SanitizedTaskEvent | SanitizedTaskEventDraft | SanitizedTaskUsageEvent
)
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
        await gather(
            *(self._record_sink_event(sink, event) for sink in self.sinks)
        )

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
        await gather(
            *(
                self._record_sink_usage(
                    sink,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    source=source,
                    totals=totals,
                    metadata=metadata,
                )
                for sink in self.sinks
            )
        )

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

    async def _record_sink_event(
        self,
        sink: ObservabilitySink,
        event: TaskObservedEvent,
    ) -> None:
        try:
            await sink.record_event(event)
        except Exception as error:
            self._record_failure(error)

    async def _record_sink_usage(
        self,
        sink: ObservabilitySink,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None,
        metadata: Mapping[str, object] | None,
    ) -> None:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskEventPipeline:
    store: TaskEventStore
    run_id: str
    sanitizer: PrivacySanitizer
    attempt_id: str | None = None
    capture_events: bool = True
    event_observer: TaskSanitizedEventObserver | None = None
    metrics_observer: TaskSanitizedEventObserver | None = None
    trace_observer: TaskSanitizedEventObserver | None = None
    observability_sink: ObservabilitySink | None = None
    critical_delivery: bool = False

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.sanitizer, PrivacySanitizer)
        assert isinstance(self.capture_events, bool)
        assert isinstance(self.critical_delivery, bool)
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if self.event_observer is not None:
            assert callable(self.event_observer)
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
            if self.critical_delivery:
                observed = await self.store.append_event(
                    self.run_id,
                    attempt_id=self.attempt_id,
                    category=draft.category,
                    event_type=draft.event_type,
                    payload=draft.payload,
                )
            else:
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
        if self.critical_delivery:
            await gather(
                _notify_observer_strict(self.event_observer, observed),
                _notify_observer_strict(self.metrics_observer, observed),
                _notify_observer_strict(self.trace_observer, observed),
                record_observability_event_strict(
                    self.observability_sink,
                    observed,
                ),
            )
            return
        await gather(
            _notify_observer(self.event_observer, observed),
            _notify_observer(self.metrics_observer, observed),
            _notify_observer(self.trace_observer, observed),
            record_observability_event(
                self.observability_sink,
                observed,
            ),
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


async def record_observability_event_strict(
    sink: ObservabilitySink | None,
    event: TaskObservedEvent,
) -> None:
    if sink is None:
        return
    await sink.record_event(event)


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
    usage_event = SanitizedTaskUsageEvent(
        run_id=run_id,
        attempt_id=attempt_id,
        source=source,
        totals=totals,
        metadata=metadata,
    )
    try:
        await sink.record_event(usage_event)
    except Exception:
        pass
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


async def record_response_usage(
    sink: ObservabilitySink | None,
    *,
    store: TaskUsageStore,
    response: object,
    run_id: str,
    attempt_id: str | None = None,
    segment_id: str | None = None,
    usage_observer: TaskSanitizedEventObserver | None = None,
) -> None:
    entries = usage_observation_entries_from_response(response)
    if not entries:
        return
    flow_node = usage_flow_node(response)
    recorded_usage_ids = await _recorded_usage_ids(
        store,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    record_tasks = []
    for entry in entries:
        usage_id = stable_usage_id_for_response(
            entry.response,
            run_id=run_id,
            attempt_id=attempt_id,
            sequence=entry.sequence,
        )
        if usage_id in recorded_usage_ids:
            continue
        recorded_usage_ids.add(usage_id)
        record_tasks.append(
            _record_response_usage_observation(
                sink,
                store=store,
                run_id=run_id,
                attempt_id=attempt_id,
                segment_id=segment_id,
                usage_id=usage_id,
                observation=entry.observation,
                flow_node=flow_node,
                usage_observer=usage_observer,
            )
        )
    await gather(*record_tasks)


async def observe_response_usage(
    response: object,
    *,
    run_id: str,
    attempt_id: str | None = None,
    segment_id: str | None = None,
    usage_observer: TaskSanitizedEventObserver | None = None,
    observed_usage_ids: set[str] | None = None,
) -> None:
    if usage_observer is None:
        return
    entries = usage_observation_entries_from_response(response)
    if not entries:
        return
    flow_node = usage_flow_node(response)
    observed_tasks = []
    for entry in entries:
        usage_id = stable_usage_id_for_response(
            entry.response,
            run_id=run_id,
            attempt_id=attempt_id,
            sequence=entry.sequence,
        )
        if observed_usage_ids is not None:
            if usage_id in observed_usage_ids:
                continue
            observed_usage_ids.add(usage_id)
        observed_tasks.append(
            _notify_observer(
                usage_observer,
                SanitizedTaskUsageEvent(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    segment_id=segment_id,
                    source=entry.observation.source,
                    totals=entry.observation.totals,
                    metadata=_usage_event_metadata_with_flow_node(
                        entry.observation.metadata,
                        flow_node,
                    ),
                ),
            )
        )
    await gather(*observed_tasks)


async def _recorded_usage_ids(
    store: TaskUsageStore,
    *,
    run_id: str,
    attempt_id: str | None,
) -> set[str]:
    try:
        return {
            record.usage_id
            for record in await store.list_usage(
                run_id,
                attempt_id=attempt_id,
            )
        }
    except Exception:
        return set()


async def _record_response_usage_observation(
    sink: ObservabilitySink | None,
    *,
    store: TaskUsageStore,
    run_id: str,
    attempt_id: str | None,
    segment_id: str | None,
    usage_id: str,
    observation: UsageObservation,
    flow_node: str | None,
    usage_observer: TaskSanitizedEventObserver | None,
) -> None:
    usage_record: UsageRecord | None = None
    try:
        usage_record = await store.append_usage(
            run_id,
            attempt_id=attempt_id,
            segment_id=segment_id,
            usage_id=usage_id,
            source=observation.source,
            totals=observation.totals,
            metadata=observation.metadata,
        )
    except TaskStoreConflictError:
        return
    except Exception:
        pass
    await gather(
        _notify_observer(
            usage_observer,
            SanitizedTaskUsageEvent(
                run_id=run_id,
                attempt_id=attempt_id,
                segment_id=segment_id,
                source=observation.source,
                totals=observation.totals,
                metadata=_usage_event_metadata_with_flow_node(
                    observation.metadata,
                    flow_node,
                ),
            ),
        ),
        record_observability_usage(
            sink,
            run_id=run_id,
            attempt_id=attempt_id,
            source=observation.source,
            totals=observation.totals,
            metadata=observation.metadata,
            record=usage_record,
        ),
    )


def _usage_event_metadata_with_flow_node(
    metadata: Mapping[str, object],
    flow_node: str | None,
) -> Mapping[str, object]:
    if flow_node is None or "flow_node" in metadata:
        return metadata
    return {**metadata, "flow_node": flow_node}


async def _notify_observer(
    observer: TaskSanitizedEventObserver | None,
    event: TaskObservedEvent,
) -> None:
    if observer is None:
        return
    try:
        result = observer(event)
        if isawaitable(result):
            await result
    except Exception:
        return


async def _notify_observer_strict(
    observer: TaskSanitizedEventObserver | None,
    event: TaskObservedEvent,
) -> None:
    if observer is None:
        return
    result = observer(event)
    if isawaitable(result):
        await result
