"""Persist trigger management through one database transaction per mutation."""

from ...pgsql import PgsqlDatabase, PgsqlUnitOfWork
from ..codec import encode_record, record_from_payload
from ..definition import integer, timestamp
from ..error import TriggerError, TriggerErrorCode
from ..records import (
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerOccurrence,
    TriggerSnapshot,
    TriggerState,
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

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from json import dumps
from uuid import uuid4

TRIGGER_SCHEMA_HEAD = "20260907_0002_triggers"


async def check_trigger_schema(unit: PgsqlUnitOfWork) -> None:
    """Reject absent or incompatible schema before any data mutation."""
    await unit.cursor.execute(
        "SELECT to_regclass('avalan_task_alembic_version') AS version_table"
    )
    row = await unit.cursor.fetchone()
    if row is None or row["version_table"] is None:
        raise TriggerError(TriggerErrorCode.SCHEMA_MISMATCH, "store.schema")
    await unit.cursor.execute(
        'SELECT "version_num" FROM "avalan_task_alembic_version"'
    )
    versions = await unit.cursor.fetchall()
    if len(versions) != 1 or versions[0]["version_num"] != TRIGGER_SCHEMA_HEAD:
        raise TriggerError(TriggerErrorCode.SCHEMA_MISMATCH, "store.schema")
    await unit.cursor.execute(
        "SELECT current_setting('transaction_isolation') AS isolation"
    )
    row = await unit.cursor.fetchone()
    if row is None or row["isolation"] != "read committed":
        raise TriggerError(
            TriggerErrorCode.STORE_INCOMPATIBLE, "store.isolation"
        )


async def lock_identity(
    unit: PgsqlUnitOfWork,
    owner: OwnerScopeId,
    identity: str,
) -> None:
    """Fence recovery against an earlier transaction's eventual completion."""
    digest = sha256(
        dumps(
            ["avalan.trigger", owner.value, identity],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    await unit.cursor.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (int.from_bytes(digest[:8], "big", signed=True),),
    )


async def decision_time(unit: PgsqlUnitOfWork) -> datetime:
    """Read the single persisted effective instant after obtaining locks."""
    await unit.cursor.execute("SELECT clock_timestamp() AS decision_time")
    row = await unit.cursor.fetchone()
    if row is None:
        raise TriggerError(TriggerErrorCode.STORE_INCOMPATIBLE, "store.clock")
    return timestamp(row["decision_time"], "store.clock")


def _snapshot(row: Mapping[str, object]) -> TriggerSnapshot:
    definition = record_from_payload(row["definition"])
    state = record_from_payload(row["state"])
    if not isinstance(definition, TriggerDefinition) or not isinstance(
        state, TriggerState
    ):
        raise TriggerError(
            TriggerErrorCode.UNSUPPORTED_VERSION, "store.record"
        )
    return TriggerSnapshot(definition=definition, state=state)


class PgsqlTriggerStore:
    """Bind management to one trusted owner and exact database capability."""

    def __init__(
        self,
        database: PgsqlDatabase,
        owner: OwnerScopeId,
        *,
        limits: SearchLimits = SearchLimits(),
    ) -> None:
        assert isinstance(owner, OwnerScopeId)
        self.database = database
        self.owner = owner
        self._limits = limits

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[PgsqlUnitOfWork]:
        async with self.database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    unit = PgsqlUnitOfWork(
                        connection=connection,
                        cursor=cursor,
                        database=self.database,
                    )
                    await check_trigger_schema(unit)
                    yield unit

    async def preflight(self) -> None:
        async with self.transaction():
            pass

    async def _read(
        self,
        unit: PgsqlUnitOfWork,
        name: str,
        *,
        lock: bool = False,
    ) -> TriggerSnapshot | None:
        query = """
SELECT d.payload AS definition, t.payload AS state
FROM triggers t JOIN trigger_definitions d
    ON (d.owner_scope_id, d.trigger_id, d.revision)
     = (t.owner_scope_id, t.trigger_id, t.revision)
WHERE t.owner_scope_id = %s AND t.name = %s
"""
        if lock:
            query += " FOR UPDATE OF t"
        await unit.cursor.execute(query, (self.owner.value, name))
        row = await unit.cursor.fetchone()
        return _snapshot(row) if row is not None else None

    async def _require(
        self,
        unit: PgsqlUnitOfWork,
        name: str,
    ) -> TriggerSnapshot:
        current = await self._read(unit, name, lock=True)
        if current is None:
            raise TriggerError(TriggerErrorCode.CONFLICT, "name")
        return current

    async def _write(
        self, unit: PgsqlUnitOfWork, mutation: TriggerMutation
    ) -> None:
        if mutation.event is None:
            return
        definition = mutation.snapshot.definition
        state = mutation.snapshot.state
        if mutation.definition_created:
            await unit.cursor.execute(
                """
INSERT INTO trigger_definitions
    (owner_scope_id, trigger_id, revision, task_definition_id,
     payload, registered_at)
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
""",
                (
                    self.owner.value,
                    definition.trigger_id,
                    definition.revision,
                    definition.task_definition_id,
                    encode_record(definition),
                    definition.registered_at,
                ),
            )
        if mutation.closed_revision is not None:
            await unit.cursor.execute(
                """
UPDATE trigger_definitions SET closed_at = %s
WHERE owner_scope_id = %s AND trigger_id = %s AND revision = %s
""",
                (
                    mutation.event.recorded_at,
                    self.owner.value,
                    state.trigger_id,
                    mutation.closed_revision,
                ),
            )
        if mutation.superseded is not None:
            await self._write_span(unit, mutation.superseded)
        parameters = (
            self.owner.value,
            state.trigger_id,
            definition.name,
            state.revision,
            state.generation,
            state.status.value,
            state.next_at,
            state.retry_after,
            state.failure_count,
            state.last_processed_at,
            encode_record(state),
        )
        await unit.cursor.execute(
            """
INSERT INTO triggers
    (owner_scope_id, trigger_id, name, revision, generation, status, next_at,
     retry_after, failure_count, last_processed_at, payload)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (owner_scope_id, trigger_id) DO UPDATE SET
    revision = EXCLUDED.revision, generation = EXCLUDED.generation,
    status = EXCLUDED.status, next_at = EXCLUDED.next_at,
    retry_after = EXCLUDED.retry_after, failure_count = EXCLUDED.failure_count,
    last_processed_at = EXCLUDED.last_processed_at, payload = EXCLUDED.payload
""",
            parameters,
        )
        await self._write_event(unit, mutation.event)

    async def _write_span(
        self, unit: PgsqlUnitOfWork, span: TriggerCoverageSpan
    ) -> None:
        await unit.cursor.execute(
            """
INSERT INTO trigger_coverage_spans
    (owner_scope_id, trigger_id, revision, span_id, first_at, until_at,
     decided_at, disposition, exact_count, payload)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
""",
            (
                span.owner_scope_id.value,
                span.trigger_id,
                span.revision,
                span.span_id,
                span.first_at,
                span.until_at,
                span.decided_at,
                span.disposition.value,
                span.exact_count,
                encode_record(span),
            ),
        )

    async def _write_event(
        self, unit: PgsqlUnitOfWork, event: TriggerEvent
    ) -> None:
        await unit.cursor.execute(
            """
INSERT INTO trigger_events
    (owner_scope_id, trigger_id, revision, event_id, recorded_at, payload)
VALUES (%s, %s, %s, %s, %s, %s::jsonb)
""",
            (
                event.owner_scope_id.value,
                event.trigger_id,
                event.revision,
                event.event_id,
                event.recorded_at,
                encode_record(event),
            ),
        )

    async def apply(
        self,
        registration: TriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerSnapshot:
        async with self.transaction() as unit:
            await lock_identity(unit, self.owner, "name:" + registration.name)
            current = await self._read(unit, registration.name, lock=True)
            mutation = apply_registration(
                self.owner,
                registration,
                current,
                expected_generation=expected_generation,
                decision_time=await decision_time(unit),
                trigger_id=(
                    current.state.trigger_id if current else str(uuid4())
                ),
                limits=self._limits,
            )
            await self._write(unit, mutation)
        return mutation.snapshot

    async def inspect(self, name: str) -> TriggerSnapshot | None:
        async with self.transaction() as unit:
            return await self._read(unit, name)

    async def set_enabled(
        self,
        name: str,
        *,
        enabled: bool,
        expected_generation: int,
    ) -> TriggerSnapshot:
        async with self.transaction() as unit:
            current = await self._require(unit, name)
            mutation = set_enabled(
                current,
                enabled=enabled,
                expected_generation=expected_generation,
                decision_time=await decision_time(unit),
            )
            await self._write(unit, mutation)
        return mutation.snapshot

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
        async with self.transaction() as unit:
            current = await self._require(unit, name)
            mutation = record_failure(
                current,
                expected_generation=expected_generation,
                decision_time=await decision_time(unit),
                error_code=error_code,
                attempts=attempts,
                base_seconds=base_seconds,
                max_seconds=max_seconds,
                permanent=permanent,
            )
            await self._write(unit, mutation)
        return mutation.snapshot

    async def list(
        self,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerSnapshot]:
        integer(limit, 1, 200, "limit")
        offset = cursor.offset if cursor else 0
        async with self.transaction() as unit:
            await unit.cursor.execute(
                """
SELECT d.payload AS definition, t.payload AS state
FROM triggers t JOIN trigger_definitions d
    ON (d.owner_scope_id, d.trigger_id, d.revision)
     = (t.owner_scope_id, t.trigger_id, t.revision)
WHERE t.owner_scope_id = %s ORDER BY t.ordinal LIMIT %s OFFSET %s
""",
                (self.owner.value, limit + 1, offset),
            )
            rows = await unit.cursor.fetchall()
        return TriggerPage(
            items=tuple(_snapshot(row) for row in rows[:limit]),
            next_cursor=(
                HistoryCursor(offset=offset + limit)
                if len(rows) > limit
                else None
            ),
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
        async with self.transaction() as unit:
            await unit.cursor.execute(
                """
SELECT d.payload AS definition, t.payload AS state
FROM triggers t JOIN trigger_definitions d
    ON (d.owner_scope_id, d.trigger_id, d.revision)
     = (t.owner_scope_id, t.trigger_id, t.revision)
WHERE t.owner_scope_id = %s AND t.status = 'active' AND t.next_at <= %s
    AND (t.retry_after IS NULL OR t.retry_after <= %s)
ORDER BY CASE WHEN t.last_processed_at > %s::timestamptz
    THEN 1 ELSE 0 END, CASE WHEN %s::timestamptz IS NOT NULL AND
    (t.last_processed_at, t.trigger_id) <= (%s::timestamptz, %s)
    THEN 1 ELSE 0 END, t.last_processed_at, t.trigger_id LIMIT %s
""",
                (
                    self.owner.value,
                    now,
                    now,
                    after.round_started_at if after is not None else None,
                    after.last_processed_at if after is not None else None,
                    after.last_processed_at if after is not None else None,
                    after.trigger_id if after is not None else None,
                    limit,
                ),
            )
            return tuple(
                _snapshot(row) for row in await unit.cursor.fetchall()
            )

    async def next_eligible_at(self) -> datetime | None:
        """Read the earliest active cursor after its durable retry delay."""
        async with self.transaction() as unit:
            await unit.cursor.execute(
                "SELECT MIN(GREATEST(next_at, retry_after)) AS eligible_at "
                "FROM triggers WHERE owner_scope_id = %s "
                "AND status = 'active' AND next_at IS NOT NULL",
                (self.owner.value,),
            )
            row = await unit.cursor.fetchone()
            if row is None or row["eligible_at"] is None:
                return None
            return timestamp(row["eligible_at"], "store.eligible_at")

    async def events(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerEvent]:
        integer(limit, 1, 200, "limit")
        offset = cursor.offset if cursor else 0
        async with self.transaction() as unit:
            current = await self._read(unit, name)
            if current is None:
                raise TriggerError(TriggerErrorCode.CONFLICT, "name")
            await unit.cursor.execute(
                """
SELECT payload FROM trigger_events
WHERE owner_scope_id = %s AND trigger_id = %s
ORDER BY ordinal LIMIT %s OFFSET %s
""",
                (
                    self.owner.value,
                    current.state.trigger_id,
                    limit + 1,
                    offset,
                ),
            )
            rows = await unit.cursor.fetchall()
        events = []
        for row in rows[:limit]:
            event = record_from_payload(row["payload"])
            if not isinstance(event, TriggerEvent):
                raise TriggerError(
                    TriggerErrorCode.UNSUPPORTED_VERSION, "store.event"
                )
            events.append(event)
        return TriggerPage(
            items=tuple(events),
            next_cursor=(
                HistoryCursor(offset=offset + limit)
                if len(rows) > limit
                else None
            ),
        )

    async def revision(
        self,
        name: str,
        revision: int,
    ) -> TriggerDefinition | None:
        integer(revision, 1, 9223372036854775807, "revision")
        async with self.transaction() as unit:
            current = await self._read(unit, name)
            if current is None:
                raise TriggerError(TriggerErrorCode.CONFLICT, "name")
            await unit.cursor.execute(
                """
SELECT payload FROM trigger_definitions
WHERE owner_scope_id = %s AND trigger_id = %s AND revision = %s
""",
                (self.owner.value, current.state.trigger_id, revision),
            )
            row = await unit.cursor.fetchone()
        if row is None:
            return None
        definition = record_from_payload(row["payload"])
        if not isinstance(definition, TriggerDefinition):
            raise TriggerError(
                TriggerErrorCode.UNSUPPORTED_VERSION, "store.definition"
            )
        return definition

    async def occurrences(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> TriggerPage[TriggerOccurrence]:
        assert type(newest_first) is bool
        order = (
            "revision DESC, scheduled_at DESC"
            if newest_first
            else "revision, scheduled_at"
        )
        integer(limit, 1, 200, "limit")
        offset = cursor.offset if cursor else 0
        async with self.transaction() as unit:
            current = await self._read(unit, name)
            if current is None:
                raise TriggerError(TriggerErrorCode.CONFLICT, "name")
            await unit.cursor.execute(
                f"""
SELECT payload FROM trigger_occurrences
WHERE owner_scope_id = %s AND trigger_id = %s
ORDER BY {order} LIMIT %s OFFSET %s
""",
                (
                    self.owner.value,
                    current.state.trigger_id,
                    limit + 1,
                    offset,
                ),
            )
            rows = await unit.cursor.fetchall()
        items = []
        for row in rows[:limit]:
            item = record_from_payload(row["payload"])
            if not isinstance(item, TriggerOccurrence):
                raise TriggerError(
                    TriggerErrorCode.UNSUPPORTED_VERSION, "store.occurrence"
                )
            items.append(item)
        return TriggerPage(
            items=tuple(items),
            next_cursor=(
                HistoryCursor(offset=offset + limit)
                if len(rows) > limit
                else None
            ),
        )

    async def coverage(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerCoverageSpan]:
        integer(limit, 1, 200, "limit")
        offset = cursor.offset if cursor else 0
        async with self.transaction() as unit:
            current = await self._read(unit, name)
            if current is None:
                raise TriggerError(TriggerErrorCode.CONFLICT, "name")
            await unit.cursor.execute(
                """
SELECT payload FROM trigger_coverage_spans
WHERE owner_scope_id = %s AND trigger_id = %s
ORDER BY ordinal LIMIT %s OFFSET %s
""",
                (
                    self.owner.value,
                    current.state.trigger_id,
                    limit + 1,
                    offset,
                ),
            )
            rows = await unit.cursor.fetchall()
        spans = []
        for row in rows[:limit]:
            span = record_from_payload(row["payload"])
            if not isinstance(span, TriggerCoverageSpan):
                raise TriggerError(
                    TriggerErrorCode.UNSUPPORTED_VERSION, "store.span"
                )
            spans.append(span)
        return TriggerPage(
            items=tuple(spans),
            next_cursor=(
                HistoryCursor(offset=offset + limit)
                if len(rows) > limit
                else None
            ),
        )
