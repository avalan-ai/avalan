from ...pgsql import (
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlRow,
    PgsqlUnitOfWork,
    classify_pgsql_error,
)
from ..queue import (
    TaskQueueConflictError,
    TaskQueueDepth,
    TaskQueueError,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueNotFoundError,
)
from ..state import TaskRunState
from ..store import TaskSnapshotMetadata, freeze_snapshot_metadata

from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from inspect import isawaitable
from json import dumps, loads
from typing import cast
from uuid import uuid4


class PgsqlTaskQueue:
    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        assert hasattr(database, "connection")
        self._database = database
        self._clock = clock or _utc_now
        self._id_factory = id_factory or _uuid_id

    async def open(self) -> None:
        open_database = getattr(self._database, "open", None)
        if open_database is None:
            return
        result = open_database()
        if isawaitable(result):
            await result

    async def aclose(self) -> None:
        aclose = getattr(self._database, "aclose", None)
        if aclose is not None:
            result = aclose()
        else:
            close = getattr(self._database, "close", None)
            result = close() if close is not None else None
        if isawaitable(result):
            await result

    async def __aenter__(self) -> "PgsqlTaskQueue":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        await self.aclose()
        return None

    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem:
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(queue_name, "queue_name")
        _assert_int(priority, "priority")
        if available_at is not None:
            assert isinstance(available_at, datetime)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            run = await _queueable_run_row(unit, run_id, queue_name)
            now = self._now()
            queue_item_id = self._new_id()
            await unit.cursor.execute(
                _INSERT_QUEUE_ITEM_SQL,
                (
                    queue_item_id,
                    run_id,
                    queue_name,
                    TaskQueueItemState.AVAILABLE.value,
                    priority,
                    available_at or now,
                    0,
                    _json(safe_metadata),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskQueueConflictError(
                    "task run already has an active queue job"
                )
            return _queue_item_from_row(row, run_state=_run_state(run))

        return cast(
            TaskQueueItem,
            await self._transaction(
                operation="task_queue_enqueue",
                callback=execute,
            ),
        )

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]:
        _assert_non_empty_string(queue_name, "queue_name")
        _assert_positive_int(limit, "limit")
        checked_at = _ensure_aware_utc(now or self._now())

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _DRAIN_QUEUE_SQL,
                (queue_name, checked_at, limit),
            )
            return tuple(
                _queue_item_from_row(row)
                for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskQueueItem, ...],
            await self._transaction(
                operation="task_queue_drain",
                callback=execute,
            ),
        )

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth:
        _assert_non_empty_string(queue_name, "queue_name")
        checked_at = _ensure_aware_utc(now or self._now())

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _DEPTH_QUEUE_SQL,
                (checked_at, checked_at, queue_name),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                return TaskQueueDepth(
                    queue_name=queue_name,
                    available=0,
                    scheduled=0,
                    claimed=0,
                    dead=0,
                    cancel_requested=0,
                )
            return _queue_depth_from_row(queue_name, row)

        return cast(
            TaskQueueDepth,
            await self._transaction(
                operation="task_queue_depth",
                callback=execute,
            ),
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        _assert_non_empty_string(queue_name, "queue_name")
        checked_at = _ensure_aware_utc(now or self._now())
        depth = await self.depth(queue_name, now=checked_at)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _HEALTH_QUEUE_SQL,
                (checked_at, checked_at, queue_name),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                return TaskQueueHealth(
                    queue_name=queue_name,
                    depth=depth,
                    checked_at=checked_at,
                )
            return TaskQueueHealth(
                queue_name=queue_name,
                depth=depth,
                checked_at=checked_at,
                oldest_available_at=cast(
                    datetime | None,
                    row.get("oldest_available_at"),
                ),
                expired_claims=_row_int(row, "expired_claims", 0),
            )

        return cast(
            TaskQueueHealth,
            await self._transaction(
                operation="task_queue_health",
                callback=execute,
            ),
        )

    async def _transaction(
        self,
        *,
        operation: str,
        callback: Callable[[PgsqlUnitOfWork], Awaitable[object]],
    ) -> object:
        try:
            async with self._database.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        return await callback(
                            PgsqlUnitOfWork(
                                connection=connection,
                                cursor=cursor,
                            )
                        )
        except TaskQueueError:
            raise
        except AssertionError:
            raise
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except PgsqlOperationError as error:
            if error.failure.category == PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise TaskQueueConflictError(str(error)) from None
            raise TaskQueueError(str(error)) from None
        except BaseException as error:
            failure = classify_pgsql_error(error, operation=operation)
            if failure.category == PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise TaskQueueConflictError(
                    "PostgreSQL operation failed: "
                    f"category={failure.category.value}, "
                    f"code={failure.code}, retryable={failure.retryable}"
                ) from None
            raise TaskQueueError(
                "PostgreSQL operation failed: "
                f"category={failure.category.value}, "
                f"code={failure.code}, retryable={failure.retryable}"
            ) from None

    def _new_id(self) -> str:
        value = self._id_factory()
        _assert_non_empty_string(value, "generated id")
        return value

    def _now(self) -> datetime:
        value = self._clock()
        assert isinstance(value, datetime), "clock must return a datetime"
        return _ensure_aware_utc(value)


_SELECT_QUEUEABLE_RUN_SQL = """
SELECT "run_id", "state", "queue_name"
FROM "task_runs"
WHERE "run_id" = %s
"""

_INSERT_QUEUE_ITEM_SQL = """
INSERT INTO "task_queue_items" (
    "queue_item_id",
    "run_id",
    "queue_name",
    "state",
    "priority",
    "available_at",
    "attempts",
    "metadata",
    "created_at",
    "updated_at"
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
ON CONFLICT DO NOTHING
RETURNING *
"""

_DRAIN_QUEUE_SQL = """
SELECT q.*, r."state" AS "run_state"
FROM "task_queue_items" q
JOIN "task_runs" r ON r."run_id" = q."run_id"
WHERE
    q."queue_name" = %s
    AND q."state" = 'available'
    AND q."available_at" <= %s
ORDER BY q."priority" DESC, q."available_at", q."queue_item_id"
LIMIT %s
"""

_DEPTH_QUEUE_SQL = """
SELECT
    COUNT(*) FILTER (
        WHERE q."state" = 'available' AND q."available_at" <= %s
    ) AS "available",
    COUNT(*) FILTER (
        WHERE q."state" = 'available' AND q."available_at" > %s
    ) AS "scheduled",
    COUNT(*) FILTER (WHERE q."state" = 'claimed') AS "claimed",
    COUNT(*) FILTER (WHERE q."state" = 'dead') AS "dead",
    COUNT(*) FILTER (
        WHERE
            q."state" IN ('available', 'claimed')
            AND r."state" = 'cancel_requested'
    ) AS "cancel_requested"
FROM "task_queue_items" q
JOIN "task_runs" r ON r."run_id" = q."run_id"
WHERE q."queue_name" = %s
"""

_HEALTH_QUEUE_SQL = """
SELECT
    MIN(q."available_at") FILTER (
        WHERE q."state" = 'available' AND q."available_at" <= %s
    ) AS "oldest_available_at",
    COUNT(*) FILTER (
        WHERE q."state" = 'claimed' AND q."lease_expires_at" <= %s
    ) AS "expired_claims"
FROM "task_queue_items" q
WHERE q."queue_name" = %s
"""


async def _queueable_run_row(
    unit: PgsqlUnitOfWork,
    run_id: str,
    queue_name: str,
) -> PgsqlRow:
    await unit.cursor.execute(_SELECT_QUEUEABLE_RUN_SQL, (run_id,))
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueNotFoundError("task run was not found")
    if row.get("queue_name") != queue_name:
        raise TaskQueueConflictError("task run targets a different queue")
    run_state = _run_state(row)
    if run_state == TaskRunState.CANCEL_REQUESTED:
        raise TaskQueueConflictError("task run has cancellation requested")
    if run_state not in {TaskRunState.VALIDATED, TaskRunState.QUEUED}:
        raise TaskQueueConflictError("task run is not ready for queueing")
    return row


def _queue_item_from_row(
    row: PgsqlRow,
    *,
    run_state: TaskRunState | None = None,
) -> TaskQueueItem:
    return TaskQueueItem(
        queue_item_id=_row_str(row, "queue_item_id"),
        run_id=_row_str(row, "run_id"),
        queue_name=_row_str(row, "queue_name"),
        state=TaskQueueItemState(_row_str(row, "state")),
        priority=_row_int(row, "priority", 0),
        available_at=_row_datetime(row, "available_at"),
        claimed_at=cast(datetime | None, row.get("claimed_at")),
        lease_expires_at=cast(datetime | None, row.get("lease_expires_at")),
        worker_id=cast(str | None, row.get("worker_id")),
        claim_token=cast(str | None, row.get("claim_token")),
        heartbeat_at=cast(datetime | None, row.get("heartbeat_at")),
        attempts=_row_int(row, "attempts", 0),
        metadata=_loads_metadata(row.get("metadata")),
        created_at=_row_datetime(row, "created_at"),
        updated_at=_row_datetime(row, "updated_at"),
        run_state=run_state or _run_state(row),
    )


def _queue_depth_from_row(
    queue_name: str,
    row: PgsqlRow,
) -> TaskQueueDepth:
    return TaskQueueDepth(
        queue_name=queue_name,
        available=_row_int(row, "available", 0),
        scheduled=_row_int(row, "scheduled", 0),
        claimed=_row_int(row, "claimed", 0),
        dead=_row_int(row, "dead", 0),
        cancel_requested=_row_int(row, "cancel_requested", 0),
    )


def _run_state(row: PgsqlRow) -> TaskRunState:
    return TaskRunState(_row_str(row, "run_state", fallback_key="state"))


def _loads_metadata(value: object) -> TaskSnapshotMetadata:
    if value is None:
        return freeze_snapshot_metadata({})
    if isinstance(value, str):
        loaded = loads(value)
        assert isinstance(loaded, Mapping)
        return freeze_snapshot_metadata(cast(Mapping[str, object], loaded))
    assert isinstance(value, Mapping)
    return freeze_snapshot_metadata(cast(Mapping[str, object], value))


def _json(value: object) -> str:
    return dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _json_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    raise AssertionError("queue metadata must be JSON-compatible")


def _row_str(
    row: PgsqlRow,
    key: str,
    *,
    fallback_key: str | None = None,
) -> str:
    value = row.get(key)
    if value is None and fallback_key is not None:
        value = row.get(fallback_key)
    assert isinstance(value, str)
    return value


def _row_int(row: PgsqlRow, key: str, default: int) -> int:
    value = row.get(key, default)
    assert isinstance(value, int)
    assert not isinstance(value, bool)
    return value


def _row_datetime(row: PgsqlRow, key: str) -> datetime:
    value = row.get(key)
    assert isinstance(value, datetime)
    return value


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _assert_int(value: int, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"


def _assert_positive_int(value: int, field_name: str) -> None:
    _assert_int(value, field_name)
    assert value > 0, f"{field_name} must be positive"


def _uuid_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(UTC)
