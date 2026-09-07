"""Combine memory task writes and queue admission in one rollback snapshot."""

from ..artifact import TaskArtifactRecord
from ..artifacts.ownership_memory import MemoryArtifactUnit
from ..idempotency import TaskIdempotencyReservationResult
from ..queue import TaskQueueItem, TaskQueueItemState
from ..state import TaskRunState
from ..store import TaskRun, TaskStore, TaskStoreConflictError
from ..stores.memory import InMemoryTaskStore
from ..submission import (
    PreparedTaskSubmission,
    TaskSubmissionOutcome,
    TaskSubmissionResult,
    TaskSubmissionWrite,
)

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from uuid import uuid4


def _snapshot(source: InMemoryTaskStore) -> InMemoryTaskStore:
    """Copy every mutable collection while retaining immutable records."""
    target = InMemoryTaskStore(
        clock=source._clock, id_factory=source._id_factory
    )
    target._definitions = dict(source._definitions)
    target._runs = dict(source._runs)
    target._attempts = dict(source._attempts)
    target._attempt_ids_by_run_id = {
        key: list(value)
        for key, value in source._attempt_ids_by_run_id.items()
    }
    target._attempt_segments = dict(source._attempt_segments)
    target._segment_ids_by_attempt_id = {
        key: list(value)
        for key, value in source._segment_ids_by_attempt_id.items()
    }
    target._run_transitions = {
        key: list(value) for key, value in source._run_transitions.items()
    }
    target._attempt_transitions = {
        key: list(value) for key, value in source._attempt_transitions.items()
    }
    target._segment_transitions = {
        key: list(value) for key, value in source._segment_transitions.items()
    }
    target._events_by_run_id = {
        key: list(value) for key, value in source._events_by_run_id.items()
    }
    target._usage_by_run_id = {
        key: list(value) for key, value in source._usage_by_run_id.items()
    }
    target._usage_by_id = dict(source._usage_by_id)
    target._artifacts = dict(source._artifacts)
    target._artifact_ids_by_run_id = {
        key: list(value)
        for key, value in source._artifact_ids_by_run_id.items()
    }
    target._idempotency_by_key = dict(source._idempotency_by_key)
    return target


def _commit(source: InMemoryTaskStore, target: InMemoryTaskStore) -> None:
    """Publish the private snapshot while the target lock remains held."""
    target._definitions = source._definitions
    target._runs = source._runs
    target._attempts = source._attempts
    target._attempt_ids_by_run_id = source._attempt_ids_by_run_id
    target._attempt_segments = source._attempt_segments
    target._segment_ids_by_attempt_id = source._segment_ids_by_attempt_id
    target._run_transitions = source._run_transitions
    target._attempt_transitions = source._attempt_transitions
    target._segment_transitions = source._segment_transitions
    target._events_by_run_id = source._events_by_run_id
    target._usage_by_run_id = source._usage_by_run_id
    target._usage_by_id = source._usage_by_id
    target._artifacts = source._artifacts
    target._artifact_ids_by_run_id = source._artifact_ids_by_run_id
    target._idempotency_by_key = source._idempotency_by_key


@dataclass(slots=True)
class MemoryTaskSubmissionUnit:
    """Retain provisional writes until the memory transaction exits."""

    participant: "MemoryTaskSubmissionParticipant"
    tasks: InMemoryTaskStore
    items: dict[str, TaskQueueItem]
    ledger: dict[
        tuple[str, str], tuple[PreparedTaskSubmission, TaskSubmissionWrite]
    ]
    ownership: MemoryArtifactUnit
    active: bool = True


class MemoryTaskSubmissionParticipant:
    """Support task submission without claiming PostgreSQL or worker leases."""

    def __init__(self, store: InMemoryTaskStore) -> None:
        assert isinstance(store, InMemoryTaskStore)
        self.store = store
        self._items: dict[str, TaskQueueItem] = {}
        self._ledger: dict[
            tuple[str, str], tuple[PreparedTaskSubmission, TaskSubmissionWrite]
        ] = {}

    def validate_submission_store(self, store: TaskStore) -> None:
        if store is not self.store:
            raise TaskStoreConflictError("submission memory store differs")

    async def preflight_submission(self) -> None:
        assert isinstance(self.store, InMemoryTaskStore)

    @asynccontextmanager
    async def submission_transaction(
        self,
    ) -> AsyncIterator[MemoryTaskSubmissionUnit]:
        async with self.store._lock:
            async with (
                self.store._artifact_ownership.transaction() as ownership
            ):
                unit = MemoryTaskSubmissionUnit(
                    participant=self,
                    tasks=_snapshot(self.store),
                    items=dict(self._items),
                    ledger=dict(self._ledger),
                    ownership=ownership,
                )
                try:
                    yield unit
                    _commit(unit.tasks, self.store)
                    self._items = unit.items
                    self._ledger = unit.ledger
                finally:
                    unit.active = False

    async def submit_prepared(
        self,
        prepared: PreparedTaskSubmission,
        *,
        unit_of_work: MemoryTaskSubmissionUnit,
    ) -> TaskSubmissionWrite:
        unit = unit_of_work
        assert unit.active and unit.participant is self
        assert prepared._participant is self
        key = (prepared.owner_scope, prepared.submission_id)
        previous = unit.ledger.get(key)
        if previous is not None:
            if previous[0] != prepared:
                raise TaskStoreConflictError("submission identity changed")
            return previous[1]
        tasks = unit.tasks
        await tasks.get_definition(prepared.execution.definition_id)
        reservation = (
            await tasks.lookup_idempotency_key(prepared.idempotency)
            if prepared.idempotency is not None
            else None
        )
        if reservation is not None:
            run = await tasks.get_run(reservation.run_id)
            write = TaskSubmissionWrite(
                submission_id=prepared.submission_id,
                run=run,
                created=False,
                artifacts=await tasks.list_artifacts(run.run_id),
                queue_item=unit.items.get(run.run_id),
                idempotency=TaskIdempotencyReservationResult(
                    reservation=reservation, created=False
                ),
            )
        else:
            if prepared.run_id in tasks._runs:
                raise TaskStoreConflictError(
                    "submission run identity is occupied"
                )
            now = tasks._now()
            run = TaskRun(
                run_id=prepared.run_id,
                definition_id=prepared.execution.definition_id,
                state=TaskRunState.CREATED,
                request=prepared.execution,
                created_at=now,
                updated_at=now,
                metadata=prepared.run_metadata,
            )
            tasks._runs[run.run_id] = run
            tasks._attempt_ids_by_run_id[run.run_id] = []
            tasks._run_transitions[run.run_id] = []
            tasks._events_by_run_id[run.run_id] = []
            tasks._usage_by_run_id[run.run_id] = []
            tasks._artifact_ids_by_run_id[run.run_id] = []
            idempotency = (
                await tasks.reserve_idempotency_key(
                    prepared.idempotency,
                    run_id=run.run_id,
                    expires_at=prepared.idempotency_expires_at,
                )
                if prepared.idempotency is not None
                else None
            )
            artifacts: list[TaskArtifactRecord] = []
            for artifact in prepared.artifacts:
                artifacts.append(
                    await tasks.append_artifact(
                        run.run_id,
                        purpose=artifact.purpose,
                        ref=artifact.ref,
                        state=artifact.state,
                        provenance=artifact.provenance,
                        retention=artifact.retention,
                        metadata=artifact.metadata,
                    )
                )
            for artifact in prepared.artifacts:
                unit.ownership.attach_submitted_run(artifact.ref)
            run = await tasks.transition_run(
                run.run_id,
                from_states={TaskRunState.CREATED},
                to_state=TaskRunState.VALIDATED,
                reason="validated",
            )
            run = await tasks.transition_run(
                run.run_id,
                from_states={TaskRunState.VALIDATED},
                to_state=TaskRunState.QUEUED,
                reason="queued",
            )
            assert prepared.execution.queue is not None
            item = TaskQueueItem(
                queue_item_id=str(uuid4()),
                run_id=run.run_id,
                queue_name=prepared.execution.queue,
                state=TaskQueueItemState.AVAILABLE,
                priority=prepared.priority,
                available_at=prepared.available_at or now,
                attempts=0,
                created_at=now,
                updated_at=now,
                run_state=run.state,
                metadata=prepared.queue_metadata,
            )
            unit.items[run.run_id] = item
            write = TaskSubmissionWrite(
                submission_id=prepared.submission_id,
                run=run,
                created=True,
                queue_item=item,
                idempotency=idempotency,
                artifacts=tuple(artifacts),
            )
        unit.ledger[key] = (prepared, write)
        return write

    async def reconcile_submission(
        self, prepared: PreparedTaskSubmission
    ) -> TaskSubmissionResult:
        assert prepared._participant is self
        # The same task lock is the completion barrier for an earlier writer.
        async with self.submission_transaction() as unit:
            previous = unit.ledger.get(
                (
                    prepared.owner_scope,
                    prepared.submission_id,
                )
            )
            if previous is not None and previous[0] != prepared:
                raise TaskStoreConflictError("submission identity changed")
            write = previous[1] if previous else None
            if write is not None:
                run = await unit.tasks.get_run(write.run.run_id)
                write = replace(
                    write,
                    run=run,
                    queue_item=(
                        replace(write.queue_item, run_state=run.state)
                        if write.queue_item is not None
                        else None
                    ),
                )
            return TaskSubmissionResult(
                submission_id=prepared.submission_id,
                outcome=(
                    TaskSubmissionOutcome.COMMITTED
                    if write is not None
                    else TaskSubmissionOutcome.NOT_COMMITTED
                ),
                write=write,
                prepared=prepared,
            )
