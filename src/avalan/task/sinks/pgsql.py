from ...pgsql import PgsqlDatabase
from ..event import (
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventCategory,
    TaskEventValue,
)
from ..observability import (
    ObservabilitySink,
    ObservabilitySinkHealth,
    TaskObservedEvent,
)
from ..stores.pgsql import PgsqlTaskStore
from ..usage import UsageRecord, UsageSource, UsageTotals

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


class PgsqlInspectionStore(Protocol):
    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        category: TaskEventCategory,
        payload: TaskEventValue,
        attempt_id: str | None = None,
    ) -> SanitizedTaskEvent: ...

    async def list_events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]: ...

    async def append_usage(
        self,
        run_id: str,
        *,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord: ...

    async def list_usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]: ...

    async def usage_totals(self, run_id: str) -> UsageTotals: ...


@dataclass(slots=True, kw_only=True)
class PgsqlInspectionSink(ObservabilitySink):
    store: PgsqlInspectionStore
    name: str = "pgsql"
    run_id: str | None = None
    attempt_id: str | None = None
    persist_recorded_events: bool = False
    persist_recorded_usage: bool = False
    _event_count: int = field(default=0, init=False)
    _usage_count: int = field(default=0, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_code: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        for method_name in (
            "append_event",
            "list_events",
            "append_usage",
            "list_usage",
            "usage_totals",
        ):
            assert callable(getattr(self.store, method_name, None))
        _assert_non_empty_string(self.name, "name")
        if self.run_id is not None:
            _assert_non_empty_string(self.run_id, "run_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        assert isinstance(self.persist_recorded_events, bool)
        assert isinstance(self.persist_recorded_usage, bool)

    @classmethod
    def from_database(
        cls,
        database: PgsqlDatabase,
        *,
        name: str = "pgsql",
        run_id: str | None = None,
        attempt_id: str | None = None,
        persist_recorded_events: bool = False,
        persist_recorded_usage: bool = False,
    ) -> "PgsqlInspectionSink":
        return cls(
            store=PgsqlTaskStore(database),
            name=name,
            run_id=run_id,
            attempt_id=attempt_id,
            persist_recorded_events=persist_recorded_events,
            persist_recorded_usage=persist_recorded_usage,
        )

    async def record_event(self, event: TaskObservedEvent) -> None:
        assert isinstance(event, SanitizedTaskEvent | SanitizedTaskEventDraft)
        try:
            if (
                isinstance(event, SanitizedTaskEvent)
                and not self.persist_recorded_events
            ):
                self._event_count += 1
                return
            await self.store.append_event(
                self._event_run_id(event),
                attempt_id=_event_attempt_id(event, self.attempt_id),
                event_type=event.event_type,
                category=event.category,
                payload=event.payload,
            )
            self._event_count += 1
        except Exception as error:
            self._record_failure(error)
            raise

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)
        try:
            effective_attempt_id = attempt_id or self.attempt_id
            await self.store.append_usage(
                run_id,
                attempt_id=effective_attempt_id,
                source=source,
                totals=totals,
                metadata=metadata,
            )
            self._usage_count += 1
        except Exception as error:
            self._record_failure(error)
            raise

    async def record_usage_record(self, record: UsageRecord) -> None:
        assert isinstance(record, UsageRecord)
        try:
            if not self.persist_recorded_usage:
                self._usage_count += 1
                return
            await self.store.append_usage(
                record.run_id,
                attempt_id=record.attempt_id,
                source=record.source,
                totals=record.totals,
                metadata=record.metadata,
            )
            self._usage_count += 1
        except Exception as error:
            self._record_failure(error)
            raise

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        if after_sequence is not None:
            _assert_non_negative_int(after_sequence, "after_sequence")
        return await self.store.list_events(
            run_id,
            attempt_id=attempt_id,
            after_sequence=after_sequence,
        )

    async def usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        return await self.store.list_usage(run_id, attempt_id=attempt_id)

    async def totals(self, run_id: str) -> UsageTotals:
        _assert_non_empty_string(run_id, "run_id")
        return await self.store.usage_totals(run_id)

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=self._event_count,
            usage_count=self._usage_count,
            failure_count=self._failure_count,
            last_failure_code=self._last_failure_code,
        )

    def _event_run_id(self, event: TaskObservedEvent) -> str:
        if isinstance(event, SanitizedTaskEvent):
            return event.run_id
        if self.run_id is None:
            raise ValueError("run_id is required for task event drafts")
        return self.run_id

    def _record_failure(self, error: Exception) -> None:
        self._failure_count += 1
        self._last_failure_code = type(error).__name__


def _event_attempt_id(
    event: TaskObservedEvent,
    default_attempt_id: str | None,
) -> str | None:
    if isinstance(event, SanitizedTaskEvent):
        return event.attempt_id
    return default_attempt_id


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _assert_non_negative_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"
