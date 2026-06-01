from ...pgsql import (
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlRow,
    PgsqlUnitOfWork,
    classify_pgsql_error,
)
from ..artifact import TaskArtifactRecord
from ..idempotency import (
    TaskIdempotencyIdentity,
    TaskIdempotencyReservation,
    TaskIdempotencyReservationResult,
)
from ..queue import (
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueDepth,
    TaskQueueError,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueNotFoundError,
    TaskQueueRetry,
    TaskQueueSubmission,
)
from ..state import TaskAttemptState, TaskRunState
from ..store import (
    TaskAttempt,
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotMetadata,
    freeze_snapshot_metadata,
)
from ..stores.pgsql import (
    _ASSIGN_RUN_CLAIM_SQL,
    _INSERT_ARTIFACT_SQL,
    _INSERT_ATTEMPT_SQL,
    _INSERT_ATTEMPT_TRANSITION_SQL,
    _INSERT_IDEMPOTENCY_SQL,
    _INSERT_RUN_SQL,
    _INSERT_RUN_TRANSITION_SQL,
    _SELECT_ATTEMPTS_FOR_RUN_SQL,
    _SELECT_IDEMPOTENCY_SQL,
    _UPDATE_RUN_STATE_SQL,
    _artifact_from_row,
    _artifact_provenance_to_payload,
    _artifact_ref_to_payload,
    _artifact_retention_to_payload,
    _attempt_from_row,
    _claim_to_payload,
    _context_to_payload,
    _fetch_definition_row,
    _fetch_run_row,
    _idempotency_from_row,
    _request_to_payload,
    _result_to_payload,
    _run_from_row,
)

from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
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

    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        assert isinstance(request, TaskExecutionRequest)
        _assert_non_empty_string(queue_name, "queue_name")
        _assert_int(priority, "priority")
        if available_at is not None:
            assert isinstance(available_at, datetime)
        if idempotency is not None:
            assert isinstance(idempotency, TaskIdempotencyIdentity)
        if idempotency_expires_at is not None:
            assert isinstance(idempotency_expires_at, datetime)
        assert isinstance(artifacts, tuple)
        for artifact in artifacts:
            assert isinstance(artifact, TaskQueueArtifact)
        safe_run_metadata = freeze_snapshot_metadata(run_metadata)
        safe_queue_metadata = freeze_snapshot_metadata(queue_metadata)
        queued_request = _queued_request(request, queue_name)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            if (
                await _fetch_definition_row(
                    unit,
                    queued_request.definition_id,
                )
                is None
            ):
                raise TaskQueueNotFoundError("task definition was not found")
            if idempotency is not None:
                existing = await _active_idempotency_reservation(
                    unit,
                    idempotency,
                    now=self._now(),
                )
                if existing is not None:
                    row = await _fetch_run_row(unit, existing.run_id)
                    if row is None:
                        raise TaskQueueConflictError(
                            "idempotency reservation target was not found"
                        )
                    return TaskQueueSubmission(
                        run=_run_from_row(row),
                        created=False,
                        idempotency=TaskIdempotencyReservationResult(
                            reservation=existing,
                            created=False,
                        ),
                    )

            run_id = self._new_id()
            now = self._now()
            await unit.cursor.execute(
                _INSERT_RUN_SQL,
                (
                    run_id,
                    queued_request.definition_id,
                    TaskRunState.CREATED.value,
                    queue_name,
                    _json(_request_to_payload(queued_request)),
                    _json(safe_run_metadata),
                    now,
                    now,
                ),
            )
            run_row = await unit.cursor.fetchone()
            if run_row is None:
                raise TaskQueueConflictError("task run already exists")

            artifact_records = []
            for artifact in artifacts:
                artifact_records.append(
                    await _insert_submission_artifact(
                        unit,
                        run_id=run_id,
                        artifact=artifact,
                        now=now,
                    )
                )

            reservation_result = None
            if idempotency is not None:
                reservation_result = await _reserve_idempotency(
                    unit,
                    identity=idempotency,
                    run_id=run_id,
                    expires_at=idempotency_expires_at,
                    now=now,
                )
                if not reservation_result.created:
                    raise TaskQueueConflictError(
                        "idempotency key is already reserved"
                    )

            validated = await _transition_run(
                unit,
                run_id=run_id,
                from_state=TaskRunState.CREATED,
                to_state=TaskRunState.VALIDATED,
                reason="validated",
                transition_id=self._new_id(),
                now=now,
            )
            queued = await _transition_run(
                unit,
                run_id=run_id,
                from_state=validated.state,
                to_state=TaskRunState.QUEUED,
                reason="queued",
                transition_id=self._new_id(),
                now=now,
            )
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
                    _json(safe_queue_metadata),
                    now,
                    now,
                ),
            )
            queue_row = await unit.cursor.fetchone()
            if queue_row is None:
                raise TaskQueueConflictError(
                    "task run already has an active queue job"
                )
            return TaskQueueSubmission(
                run=queued,
                created=True,
                queue_item=_queue_item_from_row(
                    queue_row,
                    run_state=queued.state,
                ),
                idempotency=reservation_result,
                artifacts=tuple(artifact_records),
            )

        return cast(
            TaskQueueSubmission,
            await self._transaction(
                operation="task_queue_enqueue_run",
                callback=execute,
            ),
        )

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

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueClaim | None:
        _assert_non_empty_string(queue_name, "queue_name")
        _assert_non_empty_string(worker_id, "worker_id")
        assert isinstance(lease_expires_at, datetime)
        checked_at = _ensure_aware_utc(now or self._now())
        lease_expires_at = _ensure_aware_utc(lease_expires_at)
        assert lease_expires_at > checked_at
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            claim_token = self._new_id()
            await unit.cursor.execute(
                _CLAIM_QUEUE_ITEM_SQL,
                (
                    queue_name,
                    checked_at,
                    checked_at,
                    lease_expires_at,
                    worker_id,
                    claim_token,
                    checked_at,
                    checked_at,
                ),
            )
            queue_row = await unit.cursor.fetchone()
            if queue_row is None:
                return None
            claim = TaskClaim(
                worker_id=worker_id,
                claim_token=claim_token,
                claimed_at=checked_at,
                lease_expires_at=lease_expires_at,
                heartbeat_at=checked_at,
                metadata=safe_metadata,
            )
            run = await _assign_claimed_run(
                unit,
                run_id=_row_str(queue_row, "run_id"),
                claim=claim,
                transition_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            run, attempt = await _create_claimed_attempt(
                unit,
                run=run,
                attempt_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            return TaskQueueClaim(
                queue_item=_queue_item_from_row(
                    queue_row,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
            )

        return cast(
            TaskQueueClaim | None,
            await self._transaction(
                operation="task_queue_claim",
                callback=execute,
            ),
        )

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem:
        _assert_non_empty_string(queue_item_id, "queue_item_id")
        _assert_non_empty_string(claim_token, "claim_token")
        assert isinstance(lease_expires_at, datetime)
        checked_at = _ensure_aware_utc(now or self._now())
        lease_expires_at = _ensure_aware_utc(lease_expires_at)
        assert lease_expires_at > checked_at

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _HEARTBEAT_QUEUE_ITEM_SQL,
                (
                    checked_at,
                    lease_expires_at,
                    checked_at,
                    queue_item_id,
                    claim_token,
                    checked_at,
                    claim_token,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskQueueConflictError("task queue claim did not match")
            await unit.cursor.execute(
                _HEARTBEAT_RUN_CLAIM_SQL,
                (
                    checked_at.isoformat(),
                    lease_expires_at.isoformat(),
                    checked_at,
                    _row_str(row, "run_id"),
                    claim_token,
                ),
            )
            run_row = await unit.cursor.fetchone()
            if run_row is None:
                raise TaskQueueConflictError("task run claim did not match")
            return _queue_item_from_row(row)

        return cast(
            TaskQueueItem,
            await self._transaction(
                operation="task_queue_heartbeat",
                callback=execute,
            ),
        )

    async def complete(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        result: TaskExecutionResult | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        _assert_non_empty_string(queue_item_id, "queue_item_id")
        _assert_non_empty_string(claim_token, "claim_token")
        assert run_state in {
            TaskRunState.SUCCEEDED,
            TaskRunState.FAILED,
            TaskRunState.CANCELLED,
            TaskRunState.EXPIRED,
        }
        assert attempt_state in {
            TaskAttemptState.SUCCEEDED,
            TaskAttemptState.FAILED,
        }
        assert (
            run_state == TaskRunState.SUCCEEDED
            and attempt_state == TaskAttemptState.SUCCEEDED
        ) or (
            run_state != TaskRunState.SUCCEEDED
            and attempt_state == TaskAttemptState.FAILED
        )
        if result is not None:
            assert isinstance(result, TaskExecutionResult)
        checked_at = _ensure_aware_utc(now or self._now())
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            queue_row = await _fenced_queue_item_row(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                now=checked_at,
            )
            attempt = await _transition_claimed_attempt(
                unit,
                attempt_id=_row_str(queue_row, "last_attempt_id"),
                claim_token=claim_token,
                to_state=attempt_state,
                result=result,
                reason="completed",
                transition_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            run = await _transition_claimed_run(
                unit,
                run_id=_row_str(queue_row, "run_id"),
                claim_token=claim_token,
                to_state=run_state,
                result=result,
                reason="completed",
                transition_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            queue_state = (
                TaskQueueItemState.DONE
                if run_state == TaskRunState.SUCCEEDED
                else TaskQueueItemState.DEAD
            )
            updated_queue = await _complete_queue_item(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                to_state=queue_state,
                now=checked_at,
            )
            return TaskQueueCompletion(
                queue_item=_queue_item_from_row(
                    updated_queue,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
            )

        return cast(
            TaskQueueCompletion,
            await self._transaction(
                operation="task_queue_complete",
                callback=execute,
            ),
        )

    async def retry(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        result: TaskExecutionResult,
        available_at: datetime,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueRetry:
        _assert_non_empty_string(queue_item_id, "queue_item_id")
        _assert_non_empty_string(claim_token, "claim_token")
        assert isinstance(result, TaskExecutionResult)
        assert isinstance(available_at, datetime)
        _assert_positive_int(max_attempts, "max_attempts")
        checked_at = _ensure_aware_utc(now or self._now())
        available_at = _ensure_aware_utc(available_at)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            queue_row = await _fenced_queue_item_row(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                now=checked_at,
            )
            if _row_int(queue_row, "attempts", 0) >= max_attempts:
                raise TaskQueueConflictError("task queue retry limit reached")
            attempt = await _transition_claimed_attempt(
                unit,
                attempt_id=_row_str(queue_row, "last_attempt_id"),
                claim_token=claim_token,
                to_state=TaskAttemptState.FAILED,
                result=result,
                reason="attempt_retry",
                transition_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            run = await _transition_claimed_run(
                unit,
                run_id=_row_str(queue_row, "run_id"),
                claim_token=claim_token,
                to_state=TaskRunState.QUEUED,
                result=None,
                reason="attempt_retry",
                transition_id=self._new_id(),
                now=checked_at,
                metadata=safe_metadata,
            )
            updated_queue = await _retry_queue_item(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                available_at=available_at,
                now=checked_at,
            )
            return TaskQueueRetry(
                queue_item=_queue_item_from_row(
                    updated_queue,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
            )

        return cast(
            TaskQueueRetry,
            await self._transaction(
                operation="task_queue_retry",
                callback=execute,
            ),
        )

    async def abandon_expired(
        self,
        queue_name: str,
        *,
        max_attempts: int,
        limit: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[TaskQueueAbandonment, ...]:
        _assert_non_empty_string(queue_name, "queue_name")
        _assert_positive_int(max_attempts, "max_attempts")
        _assert_positive_int(limit, "limit")
        checked_at = _ensure_aware_utc(now or self._now())
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _SELECT_EXPIRED_CLAIMS_SQL,
                (queue_name, checked_at, limit),
            )
            expired_rows = tuple(await unit.cursor.fetchall())
            abandonments = []
            for queue_row in expired_rows:
                claim_token = _row_str(queue_row, "claim_token")
                attempts = _row_int(queue_row, "attempts", 0)
                retryable = attempts < max_attempts
                attempt = await _transition_claimed_attempt(
                    unit,
                    attempt_id=_row_str(queue_row, "last_attempt_id"),
                    claim_token=claim_token,
                    to_state=TaskAttemptState.ABANDONED,
                    result=None,
                    reason="abandoned",
                    transition_id=self._new_id(),
                    now=checked_at,
                    metadata=safe_metadata,
                )
                run = await _transition_claimed_run(
                    unit,
                    run_id=_row_str(queue_row, "run_id"),
                    claim_token=claim_token,
                    to_state=(
                        TaskRunState.QUEUED
                        if retryable
                        else TaskRunState.FAILED
                    ),
                    result=None,
                    reason="abandoned",
                    transition_id=self._new_id(),
                    now=checked_at,
                    metadata=safe_metadata,
                )
                updated_queue = await _abandon_queue_item(
                    unit,
                    queue_item_id=_row_str(queue_row, "queue_item_id"),
                    claim_token=claim_token,
                    to_state=(
                        TaskQueueItemState.AVAILABLE
                        if retryable
                        else TaskQueueItemState.DEAD
                    ),
                    available_at=checked_at,
                    now=checked_at,
                )
                abandonments.append(
                    TaskQueueAbandonment(
                        queue_item=_queue_item_from_row(
                            updated_queue,
                            run_state=run.state,
                        ),
                        run=run,
                        attempt=attempt,
                    )
                )
            return tuple(abandonments)

        return cast(
            tuple[TaskQueueAbandonment, ...],
            await self._transaction(
                operation="task_queue_abandon_expired",
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

_CLAIM_QUEUE_ITEM_SQL = """
WITH "candidate" AS (
    SELECT q."queue_item_id"
    FROM "task_queue_items" q
    JOIN "task_runs" r ON r."run_id" = q."run_id"
    WHERE
        q."queue_name" = %s
        AND q."state" = 'available'
        AND q."available_at" <= %s
        AND r."state" = 'queued'
    ORDER BY q."priority" DESC, q."available_at", q."queue_item_id"
    LIMIT 1
    FOR UPDATE OF q, r SKIP LOCKED
)
UPDATE "task_queue_items" q
SET "state" = 'claimed',
    "claimed_at" = %s,
    "lease_expires_at" = %s,
    "worker_id" = %s,
    "claim_token" = %s,
    "heartbeat_at" = %s,
    "attempts" = q."attempts" + 1,
    "updated_at" = %s
FROM "candidate" c, "task_runs" r
WHERE q."queue_item_id" = c."queue_item_id"
  AND r."run_id" = q."run_id"
RETURNING q.*, r."state" AS "run_state"
"""

_HEARTBEAT_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items" q
SET "heartbeat_at" = %s,
    "lease_expires_at" = %s,
    "updated_at" = %s
FROM "task_runs" r
WHERE q."run_id" = r."run_id"
  AND q."queue_item_id" = %s
  AND q."state" = 'claimed'
  AND q."claim_token" = %s
  AND q."lease_expires_at" > %s
  AND r."state" = 'claimed'
  AND (r."claim"->>'claim_token') = %s
RETURNING q.*, r."state" AS "run_state"
"""

_HEARTBEAT_RUN_CLAIM_SQL = """
UPDATE "task_runs"
SET "claim" = jsonb_set(
        jsonb_set(
            "claim",
            '{heartbeat_at}',
            to_jsonb(%s::text),
            true
        ),
        '{lease_expires_at}',
        to_jsonb(%s::text),
        true
    ),
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = 'claimed'
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""

_SELECT_FENCED_QUEUE_ITEM_SQL = """
SELECT q.*, r."state" AS "run_state", r."last_attempt_id"
FROM "task_queue_items" q
JOIN "task_runs" r ON r."run_id" = q."run_id"
WHERE q."queue_item_id" = %s
  AND q."state" = 'claimed'
  AND q."claim_token" = %s
  AND q."lease_expires_at" > %s
  AND r."state" = 'claimed'
  AND (r."claim"->>'claim_token') = %s
FOR UPDATE OF q, r
"""

_UPDATE_CLAIMED_ATTEMPT_SQL = """
WITH "candidate" AS (
    SELECT a."attempt_id", a."state" AS "from_state"
    FROM "task_attempts" a
    JOIN "task_runs" r ON r."run_id" = a."run_id"
    WHERE a."attempt_id" = %s
      AND a."state" = ANY(%s)
      AND r."state" = 'claimed'
      AND r."last_attempt_id" = a."attempt_id"
      AND (r."claim"->>'claim_token') = %s
    FOR UPDATE OF a, r
)
UPDATE "task_attempts" a
SET "state" = %s,
    "result" = COALESCE(%s::jsonb, a."result"),
    "updated_at" = %s
FROM "candidate" c
WHERE a."attempt_id" = c."attempt_id"
RETURNING a.*, c."from_state"
"""

_UPDATE_RUN_LAST_ATTEMPT_WITH_CLAIM_SQL = """
UPDATE "task_runs"
SET "last_attempt_id" = %s, "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = 'claimed'
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""

_COMPLETE_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items"
SET "state" = %s,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = 'claimed'
  AND "claim_token" = %s
RETURNING *
"""

_RETRY_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items"
SET "state" = 'available',
    "available_at" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = 'claimed'
  AND "claim_token" = %s
RETURNING *
"""

_SELECT_EXPIRED_CLAIMS_SQL = """
SELECT q.*, r."state" AS "run_state", r."last_attempt_id"
FROM "task_queue_items" q
JOIN "task_runs" r ON r."run_id" = q."run_id"
WHERE q."queue_name" = %s
  AND q."state" = 'claimed'
  AND q."lease_expires_at" <= %s
  AND r."state" = 'claimed'
  AND (r."claim"->>'claim_token') = q."claim_token"
ORDER BY q."lease_expires_at", q."queue_item_id"
LIMIT %s
FOR UPDATE OF q, r SKIP LOCKED
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


async def _assign_claimed_run(
    unit: PgsqlUnitOfWork,
    *,
    run_id: str,
    claim: TaskClaim,
    transition_id: str,
    now: datetime,
    metadata: TaskSnapshotMetadata,
) -> TaskRun:
    await unit.cursor.execute(
        _ASSIGN_RUN_CLAIM_SQL,
        (
            TaskRunState.CLAIMED.value,
            _json(_claim_to_payload(claim)),
            now,
            run_id,
            TaskRunState.QUEUED.value,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task run claim did not match")
    await unit.cursor.execute(
        _INSERT_RUN_TRANSITION_SQL,
        (
            transition_id,
            run_id,
            TaskRunState.QUEUED.value,
            TaskRunState.CLAIMED.value,
            "claimed",
            _json(metadata),
            now,
        ),
    )
    transition_row = await unit.cursor.fetchone()
    if transition_row is None:
        raise TaskQueueConflictError(
            "task run transition could not be recorded"
        )
    return _run_from_row(row)


async def _fenced_queue_item_row(
    unit: PgsqlUnitOfWork,
    *,
    queue_item_id: str,
    claim_token: str,
    now: datetime,
) -> PgsqlRow:
    await unit.cursor.execute(
        _SELECT_FENCED_QUEUE_ITEM_SQL,
        (queue_item_id, claim_token, now, claim_token),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task queue claim did not match")
    return row


async def _transition_claimed_attempt(
    unit: PgsqlUnitOfWork,
    *,
    attempt_id: str,
    claim_token: str,
    to_state: TaskAttemptState,
    result: TaskExecutionResult | None,
    reason: str,
    transition_id: str,
    now: datetime,
    metadata: TaskSnapshotMetadata,
) -> TaskAttempt:
    from_states = (
        [
            TaskAttemptState.CREATED.value,
            TaskAttemptState.RUNNING.value,
        ]
        if to_state == TaskAttemptState.ABANDONED
        else [TaskAttemptState.RUNNING.value]
    )
    await unit.cursor.execute(
        _UPDATE_CLAIMED_ATTEMPT_SQL,
        (
            attempt_id,
            from_states,
            claim_token,
            to_state.value,
            _json(_result_to_payload(result)) if result else None,
            now,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task attempt claim did not match")
    await unit.cursor.execute(
        _INSERT_ATTEMPT_TRANSITION_SQL,
        (
            transition_id,
            attempt_id,
            _row_str(row, "run_id"),
            _row_str(row, "from_state"),
            to_state.value,
            reason,
            _json(metadata),
            now,
        ),
    )
    transition_row = await unit.cursor.fetchone()
    if transition_row is None:
        raise TaskQueueConflictError(
            "task attempt transition could not be recorded"
        )
    return _attempt_from_row(row)


async def _transition_claimed_run(
    unit: PgsqlUnitOfWork,
    *,
    run_id: str,
    claim_token: str,
    to_state: TaskRunState,
    result: TaskExecutionResult | None,
    reason: str,
    transition_id: str,
    now: datetime,
    metadata: TaskSnapshotMetadata,
) -> TaskRun:
    await unit.cursor.execute(
        _UPDATE_RUN_STATE_SQL,
        (
            to_state.value,
            _json(_result_to_payload(result)) if result else None,
            now,
            run_id,
            TaskRunState.CLAIMED.value,
            claim_token,
            claim_token,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task run claim did not match")
    await unit.cursor.execute(
        _INSERT_RUN_TRANSITION_SQL,
        (
            transition_id,
            run_id,
            TaskRunState.CLAIMED.value,
            to_state.value,
            reason,
            _json(metadata),
            now,
        ),
    )
    transition_row = await unit.cursor.fetchone()
    if transition_row is None:
        raise TaskQueueConflictError(
            "task run transition could not be recorded"
        )
    return _run_from_row(row)


async def _complete_queue_item(
    unit: PgsqlUnitOfWork,
    *,
    queue_item_id: str,
    claim_token: str,
    to_state: TaskQueueItemState,
    now: datetime,
) -> PgsqlRow:
    await unit.cursor.execute(
        _COMPLETE_QUEUE_ITEM_SQL,
        (to_state.value, now, queue_item_id, claim_token),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task queue claim did not match")
    return row


async def _retry_queue_item(
    unit: PgsqlUnitOfWork,
    *,
    queue_item_id: str,
    claim_token: str,
    available_at: datetime,
    now: datetime,
) -> PgsqlRow:
    await unit.cursor.execute(
        _RETRY_QUEUE_ITEM_SQL,
        (available_at, now, queue_item_id, claim_token),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task queue claim did not match")
    return row


async def _abandon_queue_item(
    unit: PgsqlUnitOfWork,
    *,
    queue_item_id: str,
    claim_token: str,
    to_state: TaskQueueItemState,
    available_at: datetime,
    now: datetime,
) -> PgsqlRow:
    if to_state == TaskQueueItemState.AVAILABLE:
        return await _retry_queue_item(
            unit,
            queue_item_id=queue_item_id,
            claim_token=claim_token,
            available_at=available_at,
            now=now,
        )
    return await _complete_queue_item(
        unit,
        queue_item_id=queue_item_id,
        claim_token=claim_token,
        to_state=to_state,
        now=now,
    )


async def _create_claimed_attempt(
    unit: PgsqlUnitOfWork,
    *,
    run: TaskRun,
    attempt_id: str,
    now: datetime,
    metadata: TaskSnapshotMetadata,
) -> tuple[TaskRun, TaskAttempt]:
    assert run.claim is not None
    await unit.cursor.execute(_SELECT_ATTEMPTS_FOR_RUN_SQL, (run.run_id,))
    attempt_rows = tuple(await unit.cursor.fetchall())
    for attempt_row in attempt_rows:
        state = TaskAttemptState(_row_str(attempt_row, "state"))
        if state not in {
            TaskAttemptState.SUCCEEDED,
            TaskAttemptState.FAILED,
            TaskAttemptState.ABANDONED,
        }:
            raise TaskQueueConflictError(
                "task run already has an active attempt"
            )
    attempt_number = len(attempt_rows) + 1
    context = TaskExecutionContext(
        run_id=run.run_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        claim=run.claim,
    )
    await unit.cursor.execute(
        _INSERT_ATTEMPT_SQL,
        (
            attempt_id,
            run.run_id,
            attempt_number,
            TaskAttemptState.CREATED.value,
            _json(_context_to_payload(context)),
            _json(metadata),
            now,
            now,
        ),
    )
    inserted_attempt_row = await unit.cursor.fetchone()
    if inserted_attempt_row is None:
        raise TaskQueueConflictError("task attempt already exists")
    await unit.cursor.execute(
        _UPDATE_RUN_LAST_ATTEMPT_WITH_CLAIM_SQL,
        (attempt_id, now, run.run_id, run.claim.claim_token),
    )
    run_row = await unit.cursor.fetchone()
    if run_row is None:
        raise TaskQueueConflictError("task run claim did not match")
    return _run_from_row(run_row), _attempt_from_row(inserted_attempt_row)


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


def _queued_request(
    request: TaskExecutionRequest,
    queue_name: str,
) -> TaskExecutionRequest:
    if request.queue is None:
        return replace(request, queue=queue_name)
    if request.queue != queue_name:
        raise TaskQueueConflictError("task run targets a different queue")
    return request


async def _active_idempotency_reservation(
    unit: PgsqlUnitOfWork,
    identity: TaskIdempotencyIdentity,
    *,
    now: datetime,
) -> TaskIdempotencyReservation | None:
    await unit.cursor.execute(
        _SELECT_IDEMPOTENCY_SQL,
        (identity.identity_key, now),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        return None
    return _idempotency_from_row(row)


async def _reserve_idempotency(
    unit: PgsqlUnitOfWork,
    *,
    identity: TaskIdempotencyIdentity,
    run_id: str,
    expires_at: datetime | None,
    now: datetime,
) -> TaskIdempotencyReservationResult:
    await unit.cursor.execute(
        _INSERT_IDEMPOTENCY_SQL,
        (
            identity.identity_key,
            identity.task_name,
            identity.task_version,
            identity.spec_hash,
            _json(identity.owner_scope.as_dict()),
            identity.strategy.value,
            _json(identity.window.as_dict()) if identity.window else None,
            _json(identity.input.as_dict()) if identity.input else None,
            _json(identity.files.as_dict()) if identity.files else None,
            _json(identity.custom.as_dict()) if identity.custom else None,
            run_id,
            _json({}),
            expires_at,
            now,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is not None:
        return TaskIdempotencyReservationResult(
            reservation=_idempotency_from_row(row),
            created=True,
        )
    existing = await _active_idempotency_reservation(
        unit,
        identity,
        now=now,
    )
    if existing is not None:
        return TaskIdempotencyReservationResult(
            reservation=existing,
            created=False,
        )
    raise TaskQueueConflictError("idempotency key could not be reserved")


async def _insert_submission_artifact(
    unit: PgsqlUnitOfWork,
    *,
    run_id: str,
    artifact: TaskQueueArtifact,
    now: datetime,
) -> TaskArtifactRecord:
    await unit.cursor.execute(
        _INSERT_ARTIFACT_SQL,
        (
            artifact.ref.artifact_id,
            run_id,
            None,
            artifact.purpose.value,
            artifact.state.value,
            _json(_artifact_ref_to_payload(artifact.ref)),
            _json(_artifact_provenance_to_payload(artifact.provenance)),
            _json(_artifact_retention_to_payload(artifact.retention)),
            _json(artifact.metadata),
            now,
            now,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task artifact already exists")
    return _artifact_from_row(row)


async def _transition_run(
    unit: PgsqlUnitOfWork,
    *,
    run_id: str,
    from_state: TaskRunState,
    to_state: TaskRunState,
    reason: str,
    transition_id: str,
    now: datetime,
) -> TaskRun:
    await unit.cursor.execute(
        _UPDATE_RUN_STATE_SQL,
        (
            to_state.value,
            None,
            now,
            run_id,
            from_state.value,
            None,
            None,
        ),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskQueueConflictError("task run state did not match")
    await unit.cursor.execute(
        _INSERT_RUN_TRANSITION_SQL,
        (
            transition_id,
            run_id,
            from_state.value,
            to_state.value,
            reason,
            _json({}),
            now,
        ),
    )
    transition_row = await unit.cursor.fetchone()
    if transition_row is None:
        raise TaskQueueConflictError(
            "task run transition could not be recorded"
        )
    return _run_from_row(row)


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
