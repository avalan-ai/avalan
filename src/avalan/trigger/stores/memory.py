"""Model locked trigger management without claiming crash durability."""

from ...task.artifacts.ownership_memory import MemoryArtifactOwnership
from ..definition import integer, timestamp
from ..error import TriggerError, TriggerErrorCode
from ..records import (
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerOccurrence,
    TriggerSnapshot,
    TriggerStatus,
)
from ..registration import (
    TriggerMutation,
    TriggerRegistration,
    apply_registration,
    record_failure,
    set_enabled,
)
from ..schedule import SearchLimits
from ..store import HistoryCursor, TriggerDiscoveryCursor, TriggerPage

from asyncio import Lock
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TypeVar
from uuid import uuid4

_Item = TypeVar("_Item")


def _page(
    values: Sequence[_Item],
    cursor: HistoryCursor | None,
    limit: int,
) -> TriggerPage[_Item]:
    integer(limit, 1, 200, "limit")
    offset = cursor.offset if cursor is not None else 0
    end = offset + limit
    return TriggerPage(
        items=tuple(values[offset:end]),
        next_cursor=HistoryCursor(offset=end) if end < len(values) else None,
    )


class InMemoryTriggerStore:
    """Retain management history under one in-process transaction lock.

    Admission composition is a separate capability. This management model
    does not persist task runs or emulate PostgreSQL crash recovery.
    """

    def __init__(
        self,
        owner: OwnerScopeId,
        *,
        clock: Callable[[], datetime] | None = None,
        limits: SearchLimits = SearchLimits(),
    ) -> None:
        assert isinstance(owner, OwnerScopeId)
        self.owner = owner
        self._clock = clock or (lambda: datetime.now(UTC))
        self._limits = limits
        self._lock = Lock()
        self._artifact_ownership: MemoryArtifactOwnership | None = None
        self._current: dict[str, TriggerSnapshot] = {}
        self._definitions: dict[tuple[str, int], TriggerDefinition] = {}
        self._closed: dict[tuple[str, int], datetime] = {}
        self._events: dict[str, tuple[TriggerEvent, ...]] = {}
        self._spans: dict[str, tuple[TriggerCoverageSpan, ...]] = {}
        self._occurrences: dict[str, tuple[TriggerOccurrence, ...]] = {}
        self._registrations: dict[str, tuple[str, TriggerSnapshot]] = {}

    def _require(self, name: str) -> TriggerSnapshot:
        current = self._current.get(name)
        if current is None:
            raise TriggerError(TriggerErrorCode.CONFLICT, "name")
        return current

    def _write(self, mutation: TriggerMutation) -> TriggerSnapshot:
        snapshot = mutation.snapshot
        name = snapshot.definition.name
        self._current[name] = snapshot
        if mutation.definition_created:
            self._definitions[
                (snapshot.state.trigger_id, snapshot.state.revision)
            ] = snapshot.definition
        if mutation.closed_revision is not None:
            assert mutation.event is not None
            self._closed[
                (snapshot.state.trigger_id, mutation.closed_revision)
            ] = mutation.event.recorded_at
        if mutation.superseded is not None:
            self._spans[name] = self._spans.get(name, ()) + (
                mutation.superseded,
            )
        if mutation.event is not None:
            self._events[name] = self._events.get(name, ()) + (mutation.event,)
        return snapshot

    async def apply(
        self,
        registration: TriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerSnapshot:
        async with self._lock:
            current = self._current.get(registration.name)
            mutation = apply_registration(
                self.owner,
                registration,
                current,
                expected_generation=expected_generation,
                decision_time=self._clock(),
                trigger_id=(
                    current.state.trigger_id
                    if current is not None
                    else str(uuid4())
                ),
                limits=self._limits,
            )
            return self._write(mutation)

    async def inspect(self, name: str) -> TriggerSnapshot | None:
        async with self._lock:
            return self._current.get(name)

    async def list(
        self,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerSnapshot]:
        async with self._lock:
            # Insertion order is stable across replacement and control writes.
            return _page(tuple(self._current.values()), cursor, limit)

    async def set_enabled(
        self,
        name: str,
        *,
        enabled: bool,
        expected_generation: int,
    ) -> TriggerSnapshot:
        async with self._lock:
            return self._write(
                set_enabled(
                    self._require(name),
                    enabled=enabled,
                    expected_generation=expected_generation,
                    decision_time=self._clock(),
                )
            )

    async def discover(
        self,
        *,
        decision_time: datetime,
        limit: int = 100,
        after: TriggerDiscoveryCursor | None = None,
    ) -> tuple[TriggerSnapshot, ...]:
        integer(limit, 1, 1000, "discovery.limit")
        assert after is None or isinstance(after, TriggerDiscoveryCursor)
        now = timestamp(decision_time)
        async with self._lock:
            candidates = (
                snapshot
                for snapshot in self._current.values()
                if snapshot.state.status is TriggerStatus.ACTIVE
                and snapshot.state.next_at is not None
                and snapshot.state.next_at <= now
                and (
                    snapshot.state.retry_after is None
                    or snapshot.state.retry_after <= now
                )
            )
            return tuple(
                sorted(
                    candidates,
                    key=lambda snapshot: (
                        after is not None
                        and snapshot.state.last_processed_at
                        > after.round_started_at,
                        after is not None
                        and (
                            snapshot.state.last_processed_at,
                            snapshot.state.trigger_id,
                        )
                        <= (after.last_processed_at, after.trigger_id),
                        snapshot.state.last_processed_at,
                        snapshot.state.trigger_id,
                    ),
                )[:limit]
            )

    async def next_eligible_at(self) -> datetime | None:
        """Return the earliest active cursor after its durable retry delay."""
        async with self._lock:
            return min(
                (
                    max(
                        value.state.next_at,
                        value.state.retry_after or value.state.next_at,
                    )
                    for value in self._current.values()
                    if value.state.status is TriggerStatus.ACTIVE
                    and value.state.next_at is not None
                ),
                default=None,
            )

    async def events(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerEvent]:
        async with self._lock:
            self._require(name)
            return _page(self._events.get(name, ()), cursor, limit)

    async def occurrences(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> TriggerPage[TriggerOccurrence]:
        assert type(newest_first) is bool
        async with self._lock:
            self._require(name)
            return _page(
                tuple(
                    sorted(
                        self._occurrences.get(name, ()),
                        key=lambda item: (item.revision, item.scheduled_at),
                        reverse=newest_first,
                    )
                ),
                cursor,
                limit,
            )

    async def coverage(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerCoverageSpan]:
        async with self._lock:
            self._require(name)
            return _page(self._spans.get(name, ()), cursor, limit)

    async def revision(
        self, name: str, revision: int
    ) -> TriggerDefinition | None:
        integer(revision, 1, 9223372036854775807, "revision")
        async with self._lock:
            current = self._require(name)
            return self._definitions.get((current.state.trigger_id, revision))

    async def fail(
        self,
        name: str,
        *,
        expected_generation: int,
        error_code: TriggerErrorCode,
        attempts: int = 5,
        base_seconds: int = 1,
        max_seconds: int = 60,
        permanent: bool = False,
    ) -> TriggerSnapshot:
        async with self._lock:
            return self._write(
                record_failure(
                    self._require(name),
                    expected_generation=expected_generation,
                    decision_time=self._clock(),
                    error_code=error_code,
                    attempts=attempts,
                    base_seconds=base_seconds,
                    max_seconds=max_seconds,
                    permanent=permanent,
                )
            )


def copy_trigger_state(store: InMemoryTriggerStore) -> InMemoryTriggerStore:
    """Copy mutable state for a combined transaction while locks are held."""
    snapshot = InMemoryTriggerStore(
        store.owner, clock=store._clock, limits=store._limits
    )
    snapshot._artifact_ownership = store._artifact_ownership
    snapshot._current = dict(store._current)
    snapshot._definitions = dict(store._definitions)
    snapshot._closed = dict(store._closed)
    snapshot._events = dict(store._events)
    snapshot._spans = dict(store._spans)
    snapshot._occurrences = dict(store._occurrences)
    snapshot._registrations = dict(store._registrations)
    return snapshot


def publish_trigger_state(
    source: InMemoryTriggerStore, target: InMemoryTriggerStore
) -> None:
    """Publish a validated private snapshot without callbacks or awaits."""
    target._current = source._current
    target._definitions = source._definitions
    target._closed = source._closed
    target._events = source._events
    target._spans = source._spans
    target._occurrences = source._occurrences
    target._registrations = source._registrations
