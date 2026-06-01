from ..artifact import (
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
    assert_artifact_state_collection,
    is_terminal_artifact_state,
    is_valid_artifact_transition,
)
from ..definition import TaskDefinition
from ..event import SanitizedTaskEvent, TaskEventCategory, TaskEventValue
from ..idempotency import (
    TaskIdempotencyIdentity,
    TaskIdempotencyReservation,
    TaskIdempotencyReservationResult,
)
from ..state import TaskAttemptState, TaskRunState, is_terminal_run_state
from ..store import (
    TaskAttempt,
    TaskAttemptTransition,
    TaskClaim,
    TaskDefinitionRecord,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskStoreConflictError,
    TaskStoreNotFoundError,
    TaskTransition,
    ensure_attempt_is_mutable,
    ensure_run_is_mutable,
    freeze_snapshot_metadata,
    validate_attempt_transition_request,
    validate_run_transition_request,
)
from ..usage import (
    UsageRecord,
    UsageSource,
    UsageTotals,
    aggregate_usage_totals,
)

from asyncio import Lock
from collections.abc import Callable, Collection, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4


class InMemoryTaskStore:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._id_factory = id_factory or _uuid_id
        self._definitions: dict[str, TaskDefinitionRecord] = {}
        self._runs: dict[str, TaskRun] = {}
        self._attempts: dict[str, TaskAttempt] = {}
        self._attempt_ids_by_run_id: dict[str, list[str]] = {}
        self._run_transitions: dict[str, list[TaskTransition]] = {}
        self._attempt_transitions: dict[str, list[TaskAttemptTransition]] = {}
        self._events_by_run_id: dict[str, list[SanitizedTaskEvent]] = {}
        self._usage_by_run_id: dict[str, list[UsageRecord]] = {}
        self._artifacts: dict[str, TaskArtifactRecord] = {}
        self._artifact_ids_by_run_id: dict[str, list[str]] = {}
        self._idempotency_by_key: dict[str, TaskIdempotencyReservation] = {}
        self._lock = Lock()

    async def register_definition(
        self,
        definition: TaskDefinition,
        *,
        definition_hash: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskDefinitionRecord:
        assert isinstance(definition, TaskDefinition)
        _assert_non_empty_string(definition_hash, "definition_hash")
        async with self._lock:
            existing = self._definitions.get(definition_hash)
            if existing is not None:
                if existing.definition != definition:
                    raise TaskStoreConflictError(
                        "definition hash is already registered"
                    )
                return existing
            record = TaskDefinitionRecord(
                definition_id=definition_hash,
                definition=definition,
                spec_hash=definition_hash,
                created_at=self._now(),
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._definitions[record.definition_id] = record
            return record

    async def get_definition(
        self,
        definition_id: str,
    ) -> TaskDefinitionRecord:
        _assert_non_empty_string(definition_id, "definition_id")
        async with self._lock:
            try:
                return self._definitions[definition_id]
            except KeyError as error:
                raise TaskStoreNotFoundError(
                    "task definition was not found"
                ) from error

    async def create_run(
        self,
        request: TaskExecutionRequest,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun:
        assert isinstance(request, TaskExecutionRequest)
        async with self._lock:
            self._definition_or_raise(request.definition_id)
            created_at = self._now()
            run = TaskRun(
                run_id=self._new_id(),
                definition_id=request.definition_id,
                state=TaskRunState.CREATED,
                request=request,
                created_at=created_at,
                updated_at=created_at,
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._runs[run.run_id] = run
            self._attempt_ids_by_run_id[run.run_id] = []
            self._run_transitions[run.run_id] = []
            self._events_by_run_id[run.run_id] = []
            self._usage_by_run_id[run.run_id] = []
            self._artifact_ids_by_run_id[run.run_id] = []
            return run

    async def get_run(self, run_id: str) -> TaskRun:
        _assert_non_empty_string(run_id, "run_id")
        async with self._lock:
            return self._run_or_raise(run_id)

    async def transition_run(
        self,
        run_id: str,
        *,
        from_states: Collection[TaskRunState],
        to_state: TaskRunState,
        reason: str,
        result: TaskExecutionResult | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun:
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(reason, "reason")
        if result is not None:
            assert isinstance(result, TaskExecutionResult)
        async with self._lock:
            run = self._run_or_raise(run_id)
            self._verify_claim_token(run, claim_token)
            validate_run_transition_request(
                current_state=run.state,
                from_states=from_states,
                to_state=to_state,
            )
            now = self._now()
            transition = TaskTransition(
                transition_id=self._new_id(),
                run_id=run.run_id,
                from_state=run.state,
                to_state=to_state,
                reason=reason,
                created_at=now,
                metadata=freeze_snapshot_metadata(metadata),
            )
            updated = replace(
                run,
                state=to_state,
                claim=None if is_terminal_run_state(to_state) else run.claim,
                updated_at=now,
                result=result if result is not None else run.result,
            )
            self._runs[run.run_id] = updated
            self._run_transitions[run.run_id].append(transition)
            return updated

    async def assign_claim(
        self,
        run_id: str,
        *,
        from_states: Collection[TaskRunState],
        worker_id: str,
        lease_expires_at: datetime,
        reason: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun:
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(worker_id, "worker_id")
        _assert_non_empty_string(reason, "reason")
        assert isinstance(lease_expires_at, datetime)
        async with self._lock:
            run = self._run_or_raise(run_id)
            if run.claim is not None:
                raise TaskStoreConflictError("task run already has a claim")
            validate_run_transition_request(
                current_state=run.state,
                from_states=from_states,
                to_state=TaskRunState.CLAIMED,
            )
            now = self._now()
            claim = TaskClaim(
                worker_id=worker_id,
                claim_token=self._new_id(),
                claimed_at=now,
                lease_expires_at=lease_expires_at,
                heartbeat_at=now,
                metadata=freeze_snapshot_metadata(metadata),
            )
            transition = TaskTransition(
                transition_id=self._new_id(),
                run_id=run.run_id,
                from_state=run.state,
                to_state=TaskRunState.CLAIMED,
                reason=reason,
                created_at=now,
                metadata=freeze_snapshot_metadata(metadata),
            )
            updated = replace(
                run,
                state=TaskRunState.CLAIMED,
                updated_at=now,
                claim=claim,
            )
            self._runs[run.run_id] = updated
            self._run_transitions[run.run_id].append(transition)
            return updated

    async def create_attempt(
        self,
        run_id: str,
        *,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt:
        _assert_non_empty_string(run_id, "run_id")
        async with self._lock:
            run = self._run_or_raise(run_id)
            self._verify_claim_token(run, claim_token)
            ensure_run_is_mutable(run.state)
            self._ensure_no_active_attempt(run_id)
            attempt_ids = self._attempt_ids_by_run_id[run_id]
            attempt_number = len(attempt_ids) + 1
            created_at = self._now()
            attempt_id = self._new_id()
            context = TaskExecutionContext(
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                claim=run.claim,
            )
            attempt = TaskAttempt(
                attempt_id=attempt_id,
                run_id=run_id,
                attempt_number=attempt_number,
                state=TaskAttemptState.CREATED,
                context=context,
                created_at=created_at,
                updated_at=created_at,
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._attempts[attempt_id] = attempt
            attempt_ids.append(attempt_id)
            self._attempt_transitions[attempt_id] = []
            self._runs[run_id] = replace(
                run,
                last_attempt_id=attempt_id,
                updated_at=created_at,
            )
            return attempt

    async def get_attempt(self, attempt_id: str) -> TaskAttempt:
        _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            return self._attempt_or_raise(attempt_id)

    async def list_attempts(self, run_id: str) -> tuple[TaskAttempt, ...]:
        _assert_non_empty_string(run_id, "run_id")
        async with self._lock:
            self._run_or_raise(run_id)
            return tuple(
                self._attempts[attempt_id]
                for attempt_id in self._attempt_ids_by_run_id[run_id]
            )

    async def transition_attempt(
        self,
        attempt_id: str,
        *,
        from_states: Collection[TaskAttemptState],
        to_state: TaskAttemptState,
        reason: str,
        result: TaskExecutionResult | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt:
        _assert_non_empty_string(attempt_id, "attempt_id")
        _assert_non_empty_string(reason, "reason")
        if result is not None:
            assert isinstance(result, TaskExecutionResult)
        async with self._lock:
            attempt = self._attempt_or_raise(attempt_id)
            ensure_attempt_is_mutable(attempt.state)
            run = self._run_or_raise(attempt.run_id)
            self._verify_claim_token(run, claim_token)
            validate_attempt_transition_request(
                current_state=attempt.state,
                from_states=from_states,
                to_state=to_state,
            )
            now = self._now()
            transition = TaskAttemptTransition(
                transition_id=self._new_id(),
                attempt_id=attempt.attempt_id,
                run_id=attempt.run_id,
                from_state=attempt.state,
                to_state=to_state,
                reason=reason,
                created_at=now,
                metadata=freeze_snapshot_metadata(metadata),
            )
            updated = replace(
                attempt,
                state=to_state,
                updated_at=now,
                result=result if result is not None else attempt.result,
            )
            self._attempts[attempt.attempt_id] = updated
            self._attempt_transitions[attempt.attempt_id].append(transition)
            return updated

    async def list_run_transitions(
        self,
        run_id: str,
    ) -> tuple[TaskTransition, ...]:
        _assert_non_empty_string(run_id, "run_id")
        async with self._lock:
            self._run_or_raise(run_id)
            return tuple(self._run_transitions[run_id])

    async def list_attempt_transitions(
        self,
        attempt_id: str,
    ) -> tuple[TaskAttemptTransition, ...]:
        _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            self._attempt_or_raise(attempt_id)
            return tuple(self._attempt_transitions[attempt_id])

    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        category: TaskEventCategory,
        payload: TaskEventValue,
        attempt_id: str | None = None,
    ) -> SanitizedTaskEvent:
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(event_type, "event_type")
        assert isinstance(category, TaskEventCategory)
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is not None:
                attempt = self._attempt_or_raise(attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            event = SanitizedTaskEvent(
                event_id=self._new_id(),
                run_id=run_id,
                attempt_id=attempt_id,
                sequence=len(self._events_by_run_id[run_id]) + 1,
                event_type=event_type,
                category=category,
                payload=payload,
                created_at=self._now(),
            )
            self._events_by_run_id[run_id].append(event)
            return event

    async def list_events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is None:
                return tuple(self._events_by_run_id[run_id])
            attempt = self._attempt_or_raise(attempt_id)
            if attempt.run_id != run_id:
                raise TaskStoreNotFoundError(
                    "task attempt was not found for run"
                )
            return tuple(
                event
                for event in self._events_by_run_id[run_id]
                if event.attempt_id == attempt_id
            )

    async def append_usage(
        self,
        run_id: str,
        *,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord:
        _assert_non_empty_string(run_id, "run_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is not None:
                attempt = self._attempt_or_raise(attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            record = UsageRecord(
                usage_id=self._new_id(),
                run_id=run_id,
                attempt_id=attempt_id,
                sequence=len(self._usage_by_run_id[run_id]) + 1,
                source=source,
                totals=totals,
                created_at=self._now(),
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._usage_by_run_id[run_id].append(record)
            return record

    async def list_usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is None:
                return tuple(self._usage_by_run_id[run_id])
            attempt = self._attempt_or_raise(attempt_id)
            if attempt.run_id != run_id:
                raise TaskStoreNotFoundError(
                    "task attempt was not found for run"
                )
            return tuple(
                record
                for record in self._usage_by_run_id[run_id]
                if record.attempt_id == attempt_id
            )

    async def usage_totals(self, run_id: str) -> UsageTotals:
        _assert_non_empty_string(run_id, "run_id")
        async with self._lock:
            self._run_or_raise(run_id)
            return aggregate_usage_totals(tuple(self._usage_by_run_id[run_id]))

    async def append_artifact(
        self,
        run_id: str,
        *,
        ref: TaskArtifactRef,
        purpose: TaskArtifactPurpose,
        state: TaskArtifactState | None = None,
        attempt_id: str | None = None,
        provenance: TaskArtifactProvenance | None = None,
        retention: TaskArtifactRetention | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRecord:
        _assert_non_empty_string(run_id, "run_id")
        assert isinstance(ref, TaskArtifactRef)
        assert isinstance(purpose, TaskArtifactPurpose)
        artifact_state = state or TaskArtifactState.READY
        assert isinstance(artifact_state, TaskArtifactState)
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        if provenance is not None:
            assert isinstance(provenance, TaskArtifactProvenance)
        if retention is not None:
            assert isinstance(retention, TaskArtifactRetention)
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is not None:
                attempt = self._attempt_or_raise(attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            if ref.artifact_id in self._artifacts:
                raise TaskStoreConflictError("task artifact already exists")
            now = self._now()
            record = TaskArtifactRecord(
                artifact_id=ref.artifact_id,
                run_id=run_id,
                attempt_id=attempt_id,
                purpose=purpose,
                state=artifact_state,
                ref=ref,
                created_at=now,
                updated_at=now,
                provenance=provenance or TaskArtifactProvenance(),
                retention=retention or TaskArtifactRetention(),
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._artifacts[record.artifact_id] = record
            self._artifact_ids_by_run_id[run_id].append(record.artifact_id)
            return record

    async def get_artifact(
        self,
        artifact_id: str,
    ) -> TaskArtifactRecord:
        _assert_non_empty_string(artifact_id, "artifact_id")
        async with self._lock:
            return self._artifact_or_raise(artifact_id)

    async def list_artifacts(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        purpose: TaskArtifactPurpose | None = None,
        state: TaskArtifactState | None = None,
    ) -> tuple[TaskArtifactRecord, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        if purpose is not None:
            assert isinstance(purpose, TaskArtifactPurpose)
        if state is not None:
            assert isinstance(state, TaskArtifactState)
        async with self._lock:
            self._run_or_raise(run_id)
            if attempt_id is not None:
                attempt = self._attempt_or_raise(attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            records = tuple(
                self._artifacts[artifact_id]
                for artifact_id in self._artifact_ids_by_run_id[run_id]
            )
            return tuple(
                record
                for record in records
                if (attempt_id is None or record.attempt_id == attempt_id)
                and (purpose is None or record.purpose == purpose)
                and (state is None or record.state == state)
            )

    async def transition_artifact(
        self,
        artifact_id: str,
        *,
        from_states: Collection[TaskArtifactState],
        to_state: TaskArtifactState,
        reason: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRecord:
        _assert_non_empty_string(artifact_id, "artifact_id")
        _assert_non_empty_string(reason, "reason")
        assert_artifact_state_collection(from_states, "from_states")
        assert isinstance(to_state, TaskArtifactState)
        async with self._lock:
            record = self._artifact_or_raise(artifact_id)
            if is_terminal_artifact_state(record.state):
                raise TaskStoreConflictError(
                    "terminal task artifact cannot be changed"
                )
            if record.state not in from_states:
                raise TaskStoreConflictError(
                    "task artifact state did not match"
                )
            if not is_valid_artifact_transition(record.state, to_state):
                raise TaskStoreConflictError(
                    "task artifact transition is not valid"
                )
            updated_metadata = (
                record.metadata
                if metadata is None
                else freeze_snapshot_metadata(metadata)
            )
            updated = replace(
                record,
                state=to_state,
                updated_at=self._now(),
                metadata=updated_metadata,
            )
            self._artifacts[artifact_id] = updated
            return updated

    async def reserve_idempotency_key(
        self,
        identity: TaskIdempotencyIdentity,
        *,
        run_id: str,
        expires_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskIdempotencyReservationResult:
        assert isinstance(identity, TaskIdempotencyIdentity)
        _assert_non_empty_string(run_id, "run_id")
        if expires_at is not None:
            assert isinstance(expires_at, datetime)
        async with self._lock:
            self._run_or_raise(run_id)
            existing = self._active_idempotency_reservation(identity)
            if existing is not None:
                return TaskIdempotencyReservationResult(
                    reservation=existing,
                    created=False,
                )
            now = self._now()
            reservation = TaskIdempotencyReservation(
                identity=identity,
                run_id=run_id,
                created_at=now,
                expires_at=expires_at,
                metadata=freeze_snapshot_metadata(metadata),
            )
            self._idempotency_by_key[identity.identity_key] = reservation
            return TaskIdempotencyReservationResult(
                reservation=reservation,
                created=True,
            )

    async def lookup_idempotency_key(
        self,
        identity: TaskIdempotencyIdentity,
    ) -> TaskIdempotencyReservation | None:
        assert isinstance(identity, TaskIdempotencyIdentity)
        async with self._lock:
            return self._active_idempotency_reservation(identity)

    def _definition_or_raise(self, definition_id: str) -> TaskDefinitionRecord:
        try:
            return self._definitions[definition_id]
        except KeyError as error:
            raise TaskStoreNotFoundError(
                "task definition was not found"
            ) from error

    def _run_or_raise(self, run_id: str) -> TaskRun:
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise TaskStoreNotFoundError("task run was not found") from error

    def _attempt_or_raise(self, attempt_id: str) -> TaskAttempt:
        try:
            return self._attempts[attempt_id]
        except KeyError as error:
            raise TaskStoreNotFoundError(
                "task attempt was not found"
            ) from error

    def _artifact_or_raise(self, artifact_id: str) -> TaskArtifactRecord:
        try:
            return self._artifacts[artifact_id]
        except KeyError as error:
            raise TaskStoreNotFoundError(
                "task artifact was not found"
            ) from error

    def _active_idempotency_reservation(
        self,
        identity: TaskIdempotencyIdentity,
    ) -> TaskIdempotencyReservation | None:
        reservation = self._idempotency_by_key.get(identity.identity_key)
        if reservation is None:
            return None
        expires_at = reservation.expires_at
        if expires_at is not None and expires_at <= self._now():
            return None
        return reservation

    def _ensure_no_active_attempt(self, run_id: str) -> None:
        for attempt_id in self._attempt_ids_by_run_id[run_id]:
            attempt = self._attempts[attempt_id]
            if attempt.state not in {
                TaskAttemptState.SUCCEEDED,
                TaskAttemptState.FAILED,
                TaskAttemptState.ABANDONED,
            }:
                raise TaskStoreConflictError(
                    "task run already has an active attempt"
                )

    def _verify_claim_token(
        self,
        run: TaskRun,
        claim_token: str | None,
    ) -> None:
        if run.claim is None:
            if claim_token is not None:
                raise TaskStoreConflictError("task claim token did not match")
            return
        if claim_token != run.claim.claim_token:
            raise TaskStoreConflictError("task claim token did not match")

    def _new_id(self) -> str:
        value = self._id_factory()
        _assert_non_empty_string(value, "generated id")
        return value

    def _now(self) -> datetime:
        value = self._clock()
        assert isinstance(value, datetime), "clock must return a datetime"
        return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_id() -> str:
    return uuid4().hex


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
