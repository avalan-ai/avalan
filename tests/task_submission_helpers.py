"""Provide explicit prepared fixtures for submission persistence tests."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncContextManager, Protocol, TypedDict, cast
from uuid import uuid4

from avalan.pgsql import PgsqlUnitOfWork
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.queues.pgsql import PgsqlTaskQueue
from avalan.task.store import (
    TaskExecutionRequest,
    TaskStore,
    freeze_snapshot_metadata,
)
from avalan.task.submission import (
    _PREPARATION_AUTHORITY,
    PreparedTaskSubmission,
    TaskSubmissionArtifact,
    TaskSubmissionOutcome,
    TaskSubmissionResult,
    TaskSubmissionWrite,
)


class SubmissionParticipantFixture(Protocol):
    def submission_transaction(
        self,
    ) -> AsyncContextManager[PgsqlUnitOfWork]: ...

    async def submit_prepared(
        self,
        prepared: PreparedTaskSubmission,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> TaskSubmissionWrite: ...


class SubmissionQueueFixture:
    store: TaskStore
    _submission_writes: dict[str, TaskSubmissionWrite]

    def validate_submission_store(self, store: TaskStore) -> None:
        assert self.store is store

    async def preflight_submission(self) -> None:
        return None

    @asynccontextmanager
    async def submission_transaction(self) -> AsyncIterator[PgsqlUnitOfWork]:
        # The in-memory adapters do not execute SQL. PostgreSQL transaction
        # and rollback guarantees are exercised by the actual queue tests.
        yield cast(PgsqlUnitOfWork, object())

    def record_submission_write(
        self, write: TaskSubmissionWrite
    ) -> TaskSubmissionWrite:
        if not hasattr(self, "_submission_writes"):
            self._submission_writes = {}
        self._submission_writes[write.submission_id] = write
        return write

    async def reconcile_submission(
        self, prepared: PreparedTaskSubmission
    ) -> TaskSubmissionResult:
        write = (
            self._submission_writes.get(prepared.submission_id)
            if hasattr(self, "_submission_writes")
            else None
        )
        return TaskSubmissionResult(
            submission_id=prepared.submission_id,
            outcome=(
                TaskSubmissionOutcome.COMMITTED
                if write
                else TaskSubmissionOutcome.NOT_COMMITTED
            ),
            write=write,
            prepared=prepared,
        )


def prepared_submission_fixture(
    participant: object,
    request: TaskExecutionRequest,
    *,
    queue_name: str,
    priority: int = 0,
    available_at: datetime | None = None,
    idempotency: TaskIdempotencyIdentity | None = None,
    idempotency_expires_at: datetime | None = None,
    artifacts: tuple[TaskSubmissionArtifact, ...] = (),
    run_metadata: Mapping[str, object] | None = None,
    queue_metadata: Mapping[str, object] | None = None,
    run_id: str | None = None,
) -> PreparedTaskSubmission:
    """Construct a fixture for persistence tests, not an SDK submission."""
    assert isinstance(request, TaskExecutionRequest)
    assert request.queue is None or request.queue == queue_name
    return PreparedTaskSubmission(
        submission_id=str(uuid4()),
        run_id=run_id
        or (
            participant._new_id()
            if isinstance(participant, PgsqlTaskQueue)
            else str(uuid4())
        ),
        owner_scope="fixture-owner",
        execution=TaskExecutionRequest(
            definition_id=request.definition_id,
            input_summary=request.input_summary,
            input_payload=request.input_payload,
            file_summaries=request.file_summaries,
            idempotency_key=(
                idempotency.identity_key
                if idempotency
                else request.idempotency_key
            ),
            queue=queue_name,
            metadata=request.metadata,
        ),
        priority=priority,
        available_at=available_at,
        idempotency=idempotency,
        idempotency_expires_at=idempotency_expires_at,
        artifacts=artifacts,
        temporary_artifacts=(),
        run_metadata=freeze_snapshot_metadata(run_metadata),
        queue_metadata=freeze_snapshot_metadata(queue_metadata),
        _authority=_PREPARATION_AUTHORITY,
        _participant=participant,
    )


async def persist_submission_fixture(
    participant: SubmissionParticipantFixture,
    prepared: PreparedTaskSubmission,
) -> TaskSubmissionWrite:
    """Exercise the participant with an explicitly owned test transaction."""
    async with participant.submission_transaction() as unit:
        return await participant.submit_prepared(prepared, unit_of_work=unit)


class FailureEvidenceRow(Protocol):
    """Expose the surface key used by the evidence report collector."""

    def __getitem__(self, key: str) -> object: ...


class TaskFailurePublicResult(TypedDict):
    interaction_state: str
    task_state: str


class TaskFailureEvidence(TypedDict):
    condition_id: str
    surface_id: str
    transition_from: str
    transition_to: str
    public_result_id: str
    public_result: TaskFailurePublicResult
    status_key: str
    status_value: str
    provider_call_count: int
    domain_side_effect_count: int


def task_failure_evidence(value: Mapping[str, object]) -> TaskFailureEvidence:
    """Validate the closed task evidence boundary without changing JSON."""
    strings = (
        "condition_id",
        "surface_id",
        "transition_from",
        "transition_to",
        "public_result_id",
        "status_key",
        "status_value",
    )
    counts = ("provider_call_count", "domain_side_effect_count")
    assert set(value) == {*strings, *counts, "public_result"}
    assert all(isinstance(value[key], str) for key in strings)
    assert all(
        type(value[key]) is int and cast(int, value[key]) >= 0
        for key in counts
    )
    public = value["public_result"]
    assert isinstance(public, dict)
    assert set(public) == {"interaction_state", "task_state"}
    assert all(isinstance(item, str) for item in public.values())
    # The validated record is still the original JSON object for the collector.
    assert isinstance(value, dict)
    return cast(TaskFailureEvidence, value)
