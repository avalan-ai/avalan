from ...pgsql import (
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlUnitOfWork,
    assert_pgsql_identifier,
    classify_pgsql_error,
)
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ...types import (
    assert_non_negative_int as _assert_non_negative_int,
)
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
from ..definition import (
    IdempotencyMode,
    ObservabilitySinkType,
    PrivacyAction,
    RetryBackoff,
    RunMode,
    TaskArtifactPolicy,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputType,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputType,
    TaskPrivacyPolicy,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskTargetType,
)
from ..event import SanitizedTaskEvent, TaskEventCategory, TaskEventValue
from ..feature_gate import ModuleFinder, TaskFeature, require_features
from ..idempotency import (
    TaskIdempotencyDigest,
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
    TaskSnapshotValue,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
    TaskTransition,
    ensure_attempt_is_mutable,
    ensure_run_is_mutable,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
    validate_attempt_transition_request,
    validate_run_transition_request,
)
from ..usage import (
    UsageRecord,
    UsageSource,
    UsageTotals,
    aggregate_usage_totals,
)

from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from importlib.util import find_spec
from inspect import isawaitable
from json import dumps, loads
from pathlib import Path
from re import fullmatch
from typing import Any, Protocol, cast
from uuid import uuid4

TASK_PGSQL_ALEMBIC_VERSION_TABLE = "avalan_task_alembic_version"
TASK_PGSQL_HEAD_REVISION = "20260530_0001"
TASK_PGSQL_ADVISORY_LOCK_ID = 8_172_673_911_930_301_927
_TASK_PGSQL_REVISION_MODULE = (
    "avalan.task.stores.pgsql_migrations.versions.v20260530_0001_task_schema"
)

ModuleImporter = Callable[[str], object]
TaskPgsqlState = TaskRunState | TaskAttemptState


class PgsqlTaskMigrationError(TaskStoreError):
    pass


class _AlembicConfig(Protocol):
    attributes: dict[str, object]

    def set_main_option(self, name: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlTaskMigrationSettings:
    url: str
    schema: str | None = None
    version_table: str = TASK_PGSQL_ALEMBIC_VERSION_TABLE
    advisory_lock_id: int = TASK_PGSQL_ADVISORY_LOCK_ID
    enabled_features: tuple[TaskFeature, ...] = (
        TaskFeature.POSTGRESQL_MIGRATIONS,
    )
    module_finder: ModuleFinder = find_spec
    module_importer: ModuleImporter = import_module
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.url, "url")
        if self.schema is not None:
            assert_pgsql_identifier(self.schema, "schema")
        assert_pgsql_identifier(self.version_table, "version_table")
        assert isinstance(self.advisory_lock_id, int)
        assert not isinstance(self.advisory_lock_id, bool)
        assert isinstance(self.enabled_features, tuple)
        for feature in self.enabled_features:
            assert isinstance(feature, TaskFeature)
        assert callable(self.module_finder)
        assert callable(self.module_importer)
        assert isinstance(self.attributes, Mapping)


class PgsqlTaskStore:
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

    async def __aenter__(self) -> "PgsqlTaskStore":
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

    async def register_definition(
        self,
        definition: TaskDefinition,
        *,
        definition_hash: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskDefinitionRecord:
        assert isinstance(definition, TaskDefinition)
        _assert_non_empty_string(definition_hash, "definition_hash")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_definition_row(unit, definition_hash)
            if row is not None:
                record = _definition_record_from_row(row)
                if record.definition != definition:
                    raise TaskStoreConflictError(
                        "definition hash is already registered"
                    )
                return record
            now = self._now()
            await unit.cursor.execute(
                _INSERT_DEFINITION_SQL,
                (
                    definition_hash,
                    definition.task.name,
                    definition.task.version,
                    definition_hash,
                    _json(_definition_to_payload(definition)),
                    _json(metadata or {}),
                    now,
                ),
            )
            inserted = await unit.cursor.fetchone()
            if inserted is None:
                row = await _fetch_definition_row(unit, definition_hash)
                if row is None:
                    raise TaskStoreConflictError(
                        "definition hash could not be registered"
                    )
                record = _definition_record_from_row(row)
                if record.definition != definition:
                    raise TaskStoreConflictError(
                        "definition hash is already registered"
                    )
                return record
            return _definition_record_from_row(inserted)

        return cast(
            TaskDefinitionRecord,
            await self._transaction(
                operation="task_definition_register",
                callback=execute,
            ),
        )

    async def get_definition(
        self,
        definition_id: str,
    ) -> TaskDefinitionRecord:
        _assert_non_empty_string(definition_id, "definition_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_definition_row(unit, definition_id)
            if row is None:
                raise TaskStoreNotFoundError("task definition was not found")
            return _definition_record_from_row(row)

        return cast(
            TaskDefinitionRecord,
            await self._transaction(
                operation="task_definition_get",
                callback=execute,
            ),
        )

    async def create_run(
        self,
        request: TaskExecutionRequest,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun:
        assert isinstance(request, TaskExecutionRequest)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            if (
                await _fetch_definition_row(unit, request.definition_id)
                is None
            ):
                raise TaskStoreNotFoundError("task definition was not found")
            run_id = self._new_id()
            now = self._now()
            await unit.cursor.execute(
                _INSERT_RUN_SQL,
                (
                    run_id,
                    request.definition_id,
                    TaskRunState.CREATED.value,
                    request.queue,
                    _json(_request_to_payload(request)),
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError("task run already exists")
            return _run_from_row(row)

        return cast(
            TaskRun,
            await self._transaction(
                operation="task_run_create",
                callback=execute,
            ),
        )

    async def get_run(self, run_id: str) -> TaskRun:
        _assert_non_empty_string(run_id, "run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_run_row(unit, run_id)
            if row is None:
                raise TaskStoreNotFoundError("task run was not found")
            return _run_from_row(row)

        return cast(
            TaskRun,
            await self._transaction(
                operation="task_run_get",
                callback=execute,
            ),
        )

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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            run = await _run_or_raise(unit, run_id)
            _verify_claim_token(run, claim_token)
            validate_run_transition_request(
                current_state=run.state,
                from_states=from_states,
                to_state=to_state,
            )
            now = self._now()
            await unit.cursor.execute(
                _UPDATE_RUN_STATE_SQL,
                (
                    to_state.value,
                    _json(_result_to_payload(result)) if result else None,
                    is_terminal_run_state(to_state),
                    now,
                    run_id,
                    run.state.value,
                    claim_token,
                    claim_token,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError("task run state did not match")
            await unit.cursor.execute(
                _INSERT_RUN_TRANSITION_SQL,
                (
                    self._new_id(),
                    run_id,
                    run.state.value,
                    to_state.value,
                    reason,
                    _json(metadata or {}),
                    now,
                ),
            )
            transition_row = await unit.cursor.fetchone()
            if transition_row is None:
                raise TaskStoreConflictError(
                    "task run transition could not be recorded"
                )
            return _run_from_row(row)

        return cast(
            TaskRun,
            await self._transaction(
                operation="task_run_transition",
                callback=execute,
            ),
        )

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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            run = await _run_or_raise(unit, run_id)
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
            await unit.cursor.execute(
                _ASSIGN_RUN_CLAIM_SQL,
                (
                    TaskRunState.CLAIMED.value,
                    _json(_claim_to_payload(claim)),
                    now,
                    run_id,
                    run.state.value,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError("task run state did not match")
            await unit.cursor.execute(
                _INSERT_RUN_TRANSITION_SQL,
                (
                    self._new_id(),
                    run_id,
                    run.state.value,
                    TaskRunState.CLAIMED.value,
                    reason,
                    _json(metadata or {}),
                    now,
                ),
            )
            transition_row = await unit.cursor.fetchone()
            if transition_row is None:
                raise TaskStoreConflictError(
                    "task run transition could not be recorded"
                )
            return _run_from_row(row)

        return cast(
            TaskRun,
            await self._transaction(
                operation="task_run_claim",
                callback=execute,
            ),
        )

    async def create_attempt(
        self,
        run_id: str,
        *,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt:
        _assert_non_empty_string(run_id, "run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            run = await _run_or_raise(unit, run_id)
            _verify_claim_token(run, claim_token)
            ensure_run_is_mutable(run.state)
            attempts = await _attempt_rows_for_run(unit, run_id)
            for attempt_row in attempts:
                attempt = _attempt_from_row(attempt_row)
                if attempt.state not in {
                    TaskAttemptState.SUCCEEDED,
                    TaskAttemptState.FAILED,
                    TaskAttemptState.ABANDONED,
                }:
                    raise TaskStoreConflictError(
                        "task run already has an active attempt"
                    )
            attempt_number = len(attempts) + 1
            attempt_id = self._new_id()
            now = self._now()
            context = TaskExecutionContext(
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                claim=run.claim,
            )
            await unit.cursor.execute(
                _INSERT_ATTEMPT_SQL,
                (
                    attempt_id,
                    run_id,
                    attempt_number,
                    TaskAttemptState.CREATED.value,
                    _json(_context_to_payload(context)),
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError("task attempt already exists")
            await unit.cursor.execute(
                _UPDATE_RUN_LAST_ATTEMPT_SQL,
                (
                    attempt_id,
                    now,
                    run_id,
                    run.state.value,
                    claim_token,
                    claim_token,
                ),
            )
            run_row = await unit.cursor.fetchone()
            if run_row is None:
                raise TaskStoreConflictError("task run state did not match")
            return _attempt_from_row(row)

        return cast(
            TaskAttempt,
            await self._transaction(
                operation="task_attempt_create",
                callback=execute,
            ),
        )

    async def get_attempt(self, attempt_id: str) -> TaskAttempt:
        _assert_non_empty_string(attempt_id, "attempt_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_attempt_row(unit, attempt_id)
            if row is None:
                raise TaskStoreNotFoundError("task attempt was not found")
            return _attempt_from_row(row)

        return cast(
            TaskAttempt,
            await self._transaction(
                operation="task_attempt_get",
                callback=execute,
            ),
        )

    async def list_attempts(self, run_id: str) -> tuple[TaskAttempt, ...]:
        _assert_non_empty_string(run_id, "run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            rows = await _attempt_rows_for_run(unit, run_id)
            return tuple(_attempt_from_row(row) for row in rows)

        return cast(
            tuple[TaskAttempt, ...],
            await self._transaction(
                operation="task_attempt_list",
                callback=execute,
            ),
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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            attempt = await _attempt_or_raise(unit, attempt_id)
            ensure_attempt_is_mutable(attempt.state)
            run = await _run_or_raise(unit, attempt.run_id)
            _verify_claim_token(run, claim_token)
            validate_attempt_transition_request(
                current_state=attempt.state,
                from_states=from_states,
                to_state=to_state,
            )
            now = self._now()
            await unit.cursor.execute(
                _UPDATE_ATTEMPT_STATE_SQL,
                (
                    to_state.value,
                    _json(_result_to_payload(result)) if result else None,
                    now,
                    attempt_id,
                    attempt.state.value,
                    attempt.run_id,
                    run.state.value,
                    claim_token,
                    claim_token,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task attempt state did not match"
                )
            await unit.cursor.execute(
                _INSERT_ATTEMPT_TRANSITION_SQL,
                (
                    self._new_id(),
                    attempt_id,
                    attempt.run_id,
                    attempt.state.value,
                    to_state.value,
                    reason,
                    _json(metadata or {}),
                    now,
                ),
            )
            transition_row = await unit.cursor.fetchone()
            if transition_row is None:
                raise TaskStoreConflictError(
                    "task attempt transition could not be recorded"
                )
            return _attempt_from_row(row)

        return cast(
            TaskAttempt,
            await self._transaction(
                operation="task_attempt_transition",
                callback=execute,
            ),
        )

    async def list_run_transitions(
        self,
        run_id: str,
    ) -> tuple[TaskTransition, ...]:
        _assert_non_empty_string(run_id, "run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            await unit.cursor.execute(_SELECT_RUN_TRANSITIONS_SQL, (run_id,))
            return tuple(
                _run_transition_from_row(row)
                for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskTransition, ...],
            await self._transaction(
                operation="task_run_transition_list",
                callback=execute,
            ),
        )

    async def list_attempt_transitions(
        self,
        attempt_id: str,
    ) -> tuple[TaskAttemptTransition, ...]:
        _assert_non_empty_string(attempt_id, "attempt_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _attempt_or_raise(unit, attempt_id)
            await unit.cursor.execute(
                _SELECT_ATTEMPT_TRANSITIONS_SQL,
                (attempt_id,),
            )
            return tuple(
                _attempt_transition_from_row(row)
                for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskAttemptTransition, ...],
            await self._transaction(
                operation="task_attempt_transition_list",
                callback=execute,
            ),
        )

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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _lock_run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            await unit.cursor.execute(
                _SELECT_NEXT_EVENT_SEQUENCE_SQL, (run_id,)
            )
            sequence_row = await unit.cursor.fetchone()
            sequence = cast(int, _row_value(sequence_row, "sequence", 1))
            now = self._now()
            await unit.cursor.execute(
                _INSERT_EVENT_SQL,
                (
                    self._new_id(),
                    run_id,
                    attempt_id,
                    sequence,
                    event_type,
                    _json(payload),
                    _json({"category": category.value}),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task event could not be recorded"
                )
            return _event_from_row(row)

        return cast(
            SanitizedTaskEvent,
            await self._transaction(
                operation="task_event_append",
                callback=execute,
            ),
        )

    async def list_events(
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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            await unit.cursor.execute(
                _SELECT_EVENTS_SQL,
                (
                    run_id,
                    attempt_id,
                    attempt_id,
                    after_sequence,
                    after_sequence,
                ),
            )
            return tuple(
                _event_from_row(row) for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[SanitizedTaskEvent, ...],
            await self._transaction(
                operation="task_event_list",
                callback=execute,
            ),
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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _lock_run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            await unit.cursor.execute(
                _SELECT_NEXT_USAGE_SEQUENCE_SQL, (run_id,)
            )
            sequence_row = await unit.cursor.fetchone()
            sequence = cast(int, _row_value(sequence_row, "sequence", 1))
            now = self._now()
            await unit.cursor.execute(
                _INSERT_USAGE_SQL,
                (
                    self._new_id(),
                    run_id,
                    attempt_id,
                    sequence,
                    source.value,
                    totals.input_tokens,
                    totals.output_tokens,
                    totals.total_tokens,
                    totals.cached_input_tokens,
                    totals.cache_creation_input_tokens,
                    totals.reasoning_tokens,
                    _json(metadata or {}),
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task usage could not be recorded"
                )
            return _usage_from_row(row)

        return cast(
            UsageRecord,
            await self._transaction(
                operation="task_usage_append",
                callback=execute,
            ),
        )

    async def list_usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            await unit.cursor.execute(
                _SELECT_USAGE_SQL,
                (run_id, attempt_id, attempt_id),
            )
            return tuple(
                _usage_from_row(row) for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[UsageRecord, ...],
            await self._transaction(
                operation="task_usage_list",
                callback=execute,
            ),
        )

    async def usage_totals(self, run_id: str) -> UsageTotals:
        records = await self.list_usage(run_id)
        return aggregate_usage_totals(records)

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
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            now = self._now()
            await unit.cursor.execute(
                _INSERT_ARTIFACT_SQL,
                (
                    ref.artifact_id,
                    run_id,
                    attempt_id,
                    purpose.value,
                    artifact_state.value,
                    _json(_artifact_ref_to_payload(ref)),
                    _json(
                        _artifact_provenance_to_payload(
                            provenance or TaskArtifactProvenance()
                        )
                    ),
                    _json(
                        _artifact_retention_to_payload(
                            retention or TaskArtifactRetention()
                        )
                    ),
                    _json(safe_metadata),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError("task artifact already exists")
            return _artifact_from_row(row)

        return cast(
            TaskArtifactRecord,
            await self._transaction(
                operation="task_artifact_append",
                callback=execute,
            ),
        )

    async def get_artifact(
        self,
        artifact_id: str,
    ) -> TaskArtifactRecord:
        _assert_non_empty_string(artifact_id, "artifact_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_artifact_row(unit, artifact_id)
            if row is None:
                raise TaskStoreNotFoundError("task artifact was not found")
            return _artifact_from_row(row)

        return cast(
            TaskArtifactRecord,
            await self._transaction(
                operation="task_artifact_get",
                callback=execute,
            ),
        )

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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            await unit.cursor.execute(
                _SELECT_ARTIFACTS_SQL,
                (
                    run_id,
                    attempt_id,
                    attempt_id,
                    purpose.value if purpose else None,
                    purpose.value if purpose else None,
                    state.value if state else None,
                    state.value if state else None,
                ),
            )
            return tuple(
                _artifact_from_row(row) for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskArtifactRecord, ...],
            await self._transaction(
                operation="task_artifact_list",
                callback=execute,
            ),
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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            record = await _artifact_or_raise(unit, artifact_id)
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
            now = self._now()
            await unit.cursor.execute(
                _UPDATE_ARTIFACT_STATE_SQL,
                (
                    to_state.value,
                    _json(metadata) if metadata is not None else None,
                    now,
                    artifact_id,
                    record.state.value,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task artifact state did not match"
                )
            return _artifact_from_row(row)

        return cast(
            TaskArtifactRecord,
            await self._transaction(
                operation="task_artifact_transition",
                callback=execute,
            ),
        )

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

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _run_or_raise(unit, run_id)
            now = self._now()
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
            await unit.cursor.execute(
                _INSERT_IDEMPOTENCY_SQL,
                (
                    identity.identity_key,
                    identity.task_name,
                    identity.task_version,
                    identity.spec_hash,
                    _json(identity.owner_scope.as_dict()),
                    identity.strategy.value,
                    (
                        _json(identity.window.as_dict())
                        if identity.window
                        else None
                    ),
                    (
                        _json(identity.input.as_dict())
                        if identity.input
                        else None
                    ),
                    (
                        _json(identity.files.as_dict())
                        if identity.files
                        else None
                    ),
                    (
                        _json(identity.custom.as_dict())
                        if identity.custom
                        else None
                    ),
                    run_id,
                    _json(metadata or {}),
                    expires_at,
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
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
                raise TaskStoreConflictError(
                    "idempotency key could not be reserved"
                )
            return TaskIdempotencyReservationResult(
                reservation=_idempotency_from_row(row),
                created=True,
            )

        return cast(
            TaskIdempotencyReservationResult,
            await self._transaction(
                operation="task_idempotency_reserve",
                callback=execute,
            ),
        )

    async def lookup_idempotency_key(
        self,
        identity: TaskIdempotencyIdentity,
    ) -> TaskIdempotencyReservation | None:
        assert isinstance(identity, TaskIdempotencyIdentity)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await _active_idempotency_reservation(
                unit,
                identity,
                now=self._now(),
            )

        return cast(
            TaskIdempotencyReservation | None,
            await self._transaction(
                operation="task_idempotency_lookup",
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
        except TaskStoreError:
            raise
        except AssertionError:
            raise
        except (KeyboardInterrupt, SystemExit, CancelledError):
            raise
        except PgsqlOperationError as error:
            if error.failure.category == PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise TaskStoreConflictError(str(error)) from None
            raise TaskStoreError(str(error)) from None
        except BaseException as error:
            failure = classify_pgsql_error(error, operation=operation)
            if failure.category == PgsqlFailureCategory.UNIQUE_CONFLICT:
                raise TaskStoreConflictError(
                    "PostgreSQL operation failed: "
                    f"category={failure.category.value}, "
                    f"code={failure.code}, retryable={failure.retryable}"
                ) from None
            raise TaskStoreError(
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


def task_pgsql_script_location() -> str:
    return str(Path(__file__).with_name("pgsql_migrations"))


def task_pgsql_schema_statements() -> tuple[str, ...]:
    revision = cast(Any, import_module(_TASK_PGSQL_REVISION_MODULE))
    statements = revision.TASK_SCHEMA_STATEMENTS
    assert isinstance(statements, tuple)
    for statement in statements:
        _assert_non_empty_string(statement, "statement")
    return statements


def task_pgsql_state_predicate(
    column_name: str,
    states: Collection[TaskPgsqlState],
    *,
    table_alias: str | None = None,
) -> tuple[str, tuple[object, ...]]:
    assert_pgsql_identifier(column_name, "column_name")
    assert isinstance(states, Collection)
    assert states, "states must not be empty"
    state_values: list[str] = []
    for state in states:
        assert isinstance(state, TaskRunState | TaskAttemptState)
        state_values.append(state.value)
    qualified = _qualified_pgsql_column(column_name, table_alias=table_alias)
    placeholders = ", ".join("%s" for _ in state_values)
    return f"{qualified} IN ({placeholders})", tuple(state_values)


def task_pgsql_claim_token_predicate(
    claim_token_column_name: str,
    claim_token: str | None,
    *,
    table_alias: str | None = None,
) -> tuple[str, tuple[object, ...]]:
    assert_pgsql_identifier(
        claim_token_column_name,
        "claim_token_column_name",
    )
    qualified = _qualified_pgsql_column(
        claim_token_column_name,
        table_alias=table_alias,
    )
    if claim_token is None:
        return f"{qualified} IS NULL", ()
    _assert_non_empty_string(claim_token, "claim_token")
    return f"{qualified} = %s", (claim_token,)


def task_pgsql_upgrade(
    settings: PgsqlTaskMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "upgrade", revision)


def task_pgsql_current(
    settings: PgsqlTaskMigrationSettings,
    *,
    verbose: bool = False,
) -> None:
    assert isinstance(verbose, bool)
    _run_alembic_command(settings, "current", verbose=verbose)


def task_pgsql_check(settings: PgsqlTaskMigrationSettings) -> None:
    _run_alembic_command(settings, "current", check_heads=True)


def task_pgsql_stamp(
    settings: PgsqlTaskMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "stamp", revision)


def task_pgsql_alembic_config(
    settings: PgsqlTaskMigrationSettings,
) -> _AlembicConfig:
    diagnostics = require_features(
        (TaskFeature.POSTGRESQL_MIGRATIONS,),
        enabled_features=settings.enabled_features,
        module_finder=settings.module_finder,
    )
    if diagnostics:
        diagnostic = diagnostics[0]
        raise PgsqlTaskMigrationError(
            f"{diagnostic.code}: {diagnostic.message}"
        )

    config_module = cast(Any, settings.module_importer("alembic.config"))
    config = cast(_AlembicConfig, config_module.Config())
    config.set_main_option("script_location", task_pgsql_script_location())
    config.set_main_option("sqlalchemy.url", settings.url)
    config.set_main_option("version_table", settings.version_table)
    config.set_main_option(
        "task_advisory_lock_id",
        str(settings.advisory_lock_id),
    )
    if settings.schema is not None:
        config.set_main_option("task_schema", settings.schema)
        config.set_main_option("version_table_schema", settings.schema)
    config.attributes.update(dict(settings.attributes))
    return config


def _run_alembic_command(
    settings: PgsqlTaskMigrationSettings,
    command_name: str,
    *args: object,
    **kwargs: object,
) -> None:
    config = task_pgsql_alembic_config(settings)
    command_module = cast(Any, settings.module_importer("alembic.command"))
    command = getattr(command_module, command_name)
    command(config, *args, **kwargs)


def _assert_revision(value: str) -> None:
    _assert_non_empty_string(value, "revision")
    assert fullmatch(r"[A-Za-z0-9_.@+-]+", value)


def _qualified_pgsql_column(
    column_name: str,
    *,
    table_alias: str | None,
) -> str:
    column = f'"{column_name}"'
    if table_alias is None:
        return column
    assert_pgsql_identifier(table_alias, "table_alias")
    return f'"{table_alias}".{column}'


_INSERT_DEFINITION_SQL = """
INSERT INTO "task_definitions" (
    "definition_id", "name", "version", "spec_hash", "definition",
    "metadata", "created_at"
) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
ON CONFLICT ("definition_id") DO NOTHING
RETURNING *
"""
_SELECT_DEFINITION_SQL = """
SELECT * FROM "task_definitions" WHERE "definition_id" = %s
"""
_INSERT_RUN_SQL = """
INSERT INTO "task_runs" (
    "run_id", "definition_id", "state", "queue_name", "request",
    "metadata", "created_at", "updated_at"
) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
ON CONFLICT ("run_id") DO NOTHING
RETURNING *
"""
_SELECT_RUN_SQL = """
SELECT * FROM "task_runs" WHERE "run_id" = %s
"""
_SELECT_RUN_FOR_UPDATE_SQL = """
SELECT * FROM "task_runs" WHERE "run_id" = %s FOR UPDATE
"""
_UPDATE_RUN_STATE_SQL = """
UPDATE "task_runs"
SET "state" = %s,
    "result" = COALESCE(%s::jsonb, "result"),
    "claim" = CASE WHEN %s::boolean THEN NULL ELSE "claim" END,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND (
      (%s IS NULL AND "claim" IS NULL)
      OR ("claim"->>'claim_token') = %s
  )
RETURNING *
"""
_ASSIGN_RUN_CLAIM_SQL = """
UPDATE "task_runs"
SET "state" = %s,
    "claim" = %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND "claim" IS NULL
RETURNING *
"""
_UPDATE_RUN_LAST_ATTEMPT_SQL = """
UPDATE "task_runs"
SET "last_attempt_id" = %s, "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND (
      (%s IS NULL AND "claim" IS NULL)
      OR ("claim"->>'claim_token') = %s
  )
RETURNING *
"""
_INSERT_RUN_TRANSITION_SQL = """
INSERT INTO "task_run_transitions" (
    "transition_id", "run_id", "from_state", "to_state", "reason",
    "metadata", "created_at"
) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT ("transition_id") DO NOTHING
RETURNING *
"""
_SELECT_RUN_TRANSITIONS_SQL = """
SELECT * FROM "task_run_transitions"
WHERE "run_id" = %s
ORDER BY "created_at", "transition_id"
"""
_INSERT_ATTEMPT_SQL = """
INSERT INTO "task_attempts" (
    "attempt_id", "run_id", "attempt_number", "state", "context",
    "metadata", "created_at", "updated_at"
) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
ON CONFLICT ("attempt_id") DO NOTHING
RETURNING *
"""
_SELECT_ATTEMPT_SQL = """
SELECT * FROM "task_attempts" WHERE "attempt_id" = %s
"""
_SELECT_ATTEMPTS_FOR_RUN_SQL = """
SELECT * FROM "task_attempts"
WHERE "run_id" = %s
ORDER BY "attempt_number", "created_at", "attempt_id"
"""
_UPDATE_ATTEMPT_STATE_SQL = """
UPDATE "task_attempts" a
SET "state" = %s,
    "result" = COALESCE(%s::jsonb, a."result"),
    "updated_at" = %s
FROM "task_runs" r
WHERE a."attempt_id" = %s
  AND a."state" = %s
  AND a."run_id" = r."run_id"
  AND r."run_id" = %s
  AND r."state" = %s
  AND (
      (%s IS NULL AND r."claim" IS NULL)
      OR (r."claim"->>'claim_token') = %s
  )
RETURNING a.*
"""
_INSERT_ATTEMPT_TRANSITION_SQL = """
INSERT INTO "task_attempt_transitions" (
    "transition_id", "attempt_id", "run_id", "from_state", "to_state",
    "reason", "metadata", "created_at"
) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT ("transition_id") DO NOTHING
RETURNING *
"""
_SELECT_ATTEMPT_TRANSITIONS_SQL = """
SELECT * FROM "task_attempt_transitions"
WHERE "attempt_id" = %s
ORDER BY "created_at", "transition_id"
"""
_SELECT_NEXT_EVENT_SEQUENCE_SQL = """
SELECT COALESCE(MAX("sequence"), 0) + 1 AS "sequence"
FROM "task_events"
WHERE "run_id" = %s
"""
_INSERT_EVENT_SQL = """
INSERT INTO "task_events" (
    "event_id", "run_id", "attempt_id", "sequence", "event_type",
    "payload", "metadata", "event_time", "created_at"
) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
ON CONFLICT ("event_id") DO NOTHING
RETURNING *
"""
_SELECT_EVENTS_SQL = """
SELECT * FROM "task_events"
WHERE "run_id" = %s
  AND (%s IS NULL OR "attempt_id" = %s)
  AND (%s IS NULL OR "sequence" > %s)
ORDER BY "sequence", "created_at", "event_id"
"""
_SELECT_NEXT_USAGE_SEQUENCE_SQL = """
SELECT COALESCE(MAX("sequence"), 0) + 1 AS "sequence"
FROM "task_usage_records"
WHERE "run_id" = %s
"""
_INSERT_USAGE_SQL = """
INSERT INTO "task_usage_records" (
    "usage_id", "run_id", "attempt_id", "sequence", "source", "prompt_tokens",
    "completion_tokens", "total_tokens", "cached_tokens",
    "cache_creation_input_tokens", "reasoning_tokens", "metadata",
    "created_at"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT ("usage_id") DO NOTHING
RETURNING *
"""
_SELECT_USAGE_SQL = """
SELECT * FROM "task_usage_records"
WHERE "run_id" = %s
  AND (%s IS NULL OR "attempt_id" = %s)
ORDER BY "sequence", "created_at", "usage_id"
"""
_INSERT_ARTIFACT_SQL = """
INSERT INTO "task_artifacts" (
    "artifact_id", "run_id", "attempt_id", "purpose", "state", "ref",
    "provenance", "retention", "metadata", "created_at", "updated_at"
) VALUES (
    %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s
)
ON CONFLICT ("artifact_id") DO NOTHING
RETURNING *
"""
_SELECT_ARTIFACT_SQL = """
SELECT * FROM "task_artifacts" WHERE "artifact_id" = %s
"""
_SELECT_ARTIFACTS_SQL = """
SELECT * FROM "task_artifacts"
WHERE "run_id" = %s
  AND (%s IS NULL OR "attempt_id" = %s)
  AND (%s IS NULL OR "purpose" = %s)
  AND (%s IS NULL OR "state" = %s)
ORDER BY "created_at", "artifact_id"
"""
_UPDATE_ARTIFACT_STATE_SQL = """
UPDATE "task_artifacts"
SET "state" = %s,
    "metadata" = COALESCE(%s::jsonb, "metadata"),
    "updated_at" = %s
WHERE "artifact_id" = %s
  AND "state" = %s
RETURNING *
"""
_INSERT_IDEMPOTENCY_SQL = """
INSERT INTO "task_idempotency_keys" (
    "identity_key", "task_name", "task_version", "spec_hash",
    "owner_scope_hash", "strategy", "window_hash", "input_hash",
    "file_hash", "custom_hash", "run_id", "metadata", "expires_at",
    "created_at"
) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s::jsonb,
          %s::jsonb, %s, %s::jsonb, %s, %s)
ON CONFLICT ("identity_key") DO UPDATE
SET "task_name" = EXCLUDED."task_name",
    "task_version" = EXCLUDED."task_version",
    "spec_hash" = EXCLUDED."spec_hash",
    "owner_scope_hash" = EXCLUDED."owner_scope_hash",
    "strategy" = EXCLUDED."strategy",
    "window_hash" = EXCLUDED."window_hash",
    "input_hash" = EXCLUDED."input_hash",
    "file_hash" = EXCLUDED."file_hash",
    "custom_hash" = EXCLUDED."custom_hash",
    "run_id" = EXCLUDED."run_id",
    "metadata" = EXCLUDED."metadata",
    "expires_at" = EXCLUDED."expires_at",
    "created_at" = EXCLUDED."created_at"
WHERE "task_idempotency_keys"."expires_at" IS NOT NULL
  AND "task_idempotency_keys"."expires_at" <= %s
RETURNING *
"""
_SELECT_IDEMPOTENCY_SQL = """
SELECT * FROM "task_idempotency_keys"
WHERE "identity_key" = %s
  AND ("expires_at" IS NULL OR "expires_at" > %s)
"""


async def _fetch_definition_row(
    unit: PgsqlUnitOfWork,
    definition_id: str,
) -> Mapping[str, object] | None:
    await unit.cursor.execute(_SELECT_DEFINITION_SQL, (definition_id,))
    return await unit.cursor.fetchone()


async def _fetch_run_row(
    unit: PgsqlUnitOfWork,
    run_id: str,
) -> Mapping[str, object] | None:
    await unit.cursor.execute(_SELECT_RUN_SQL, (run_id,))
    return await unit.cursor.fetchone()


async def _run_or_raise(unit: PgsqlUnitOfWork, run_id: str) -> TaskRun:
    row = await _fetch_run_row(unit, run_id)
    if row is None:
        raise TaskStoreNotFoundError("task run was not found")
    return _run_from_row(row)


async def _lock_run_or_raise(unit: PgsqlUnitOfWork, run_id: str) -> TaskRun:
    await unit.cursor.execute(_SELECT_RUN_FOR_UPDATE_SQL, (run_id,))
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskStoreNotFoundError("task run was not found")
    return _run_from_row(row)


async def _fetch_attempt_row(
    unit: PgsqlUnitOfWork,
    attempt_id: str,
) -> Mapping[str, object] | None:
    await unit.cursor.execute(_SELECT_ATTEMPT_SQL, (attempt_id,))
    return await unit.cursor.fetchone()


async def _attempt_or_raise(
    unit: PgsqlUnitOfWork,
    attempt_id: str,
) -> TaskAttempt:
    row = await _fetch_attempt_row(unit, attempt_id)
    if row is None:
        raise TaskStoreNotFoundError("task attempt was not found")
    return _attempt_from_row(row)


async def _attempt_rows_for_run(
    unit: PgsqlUnitOfWork,
    run_id: str,
) -> tuple[Mapping[str, object], ...]:
    await unit.cursor.execute(_SELECT_ATTEMPTS_FOR_RUN_SQL, (run_id,))
    return tuple(await unit.cursor.fetchall())


async def _fetch_artifact_row(
    unit: PgsqlUnitOfWork,
    artifact_id: str,
) -> Mapping[str, object] | None:
    await unit.cursor.execute(_SELECT_ARTIFACT_SQL, (artifact_id,))
    return await unit.cursor.fetchone()


async def _artifact_or_raise(
    unit: PgsqlUnitOfWork,
    artifact_id: str,
) -> TaskArtifactRecord:
    row = await _fetch_artifact_row(unit, artifact_id)
    if row is None:
        raise TaskStoreNotFoundError("task artifact was not found")
    return _artifact_from_row(row)


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


def _definition_record_from_row(
    row: Mapping[str, object],
) -> TaskDefinitionRecord:
    return TaskDefinitionRecord(
        definition_id=cast(str, row["definition_id"]),
        definition=_definition_from_payload(_mapping(row["definition"])),
        spec_hash=cast(str, row["spec_hash"]),
        created_at=_datetime(row["created_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _run_from_row(row: Mapping[str, object]) -> TaskRun:
    return TaskRun(
        run_id=cast(str, row["run_id"]),
        definition_id=cast(str, row["definition_id"]),
        state=TaskRunState(cast(str, row["state"])),
        request=_request_from_payload(_mapping(row["request"])),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        claim=(
            _claim_from_payload(_mapping(row["claim"]))
            if row.get("claim") is not None
            else None
        ),
        last_attempt_id=cast(str | None, row.get("last_attempt_id")),
        result=(
            _result_from_payload(_mapping(row["result"]))
            if row.get("result") is not None
            else None
        ),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _attempt_from_row(row: Mapping[str, object]) -> TaskAttempt:
    return TaskAttempt(
        attempt_id=cast(str, row["attempt_id"]),
        run_id=cast(str, row["run_id"]),
        attempt_number=cast(int, row["attempt_number"]),
        state=TaskAttemptState(cast(str, row["state"])),
        context=_context_from_payload(_mapping(row["context"])),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        result=(
            _result_from_payload(_mapping(row["result"]))
            if row.get("result") is not None
            else None
        ),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _run_transition_from_row(row: Mapping[str, object]) -> TaskTransition:
    return TaskTransition(
        transition_id=cast(str, row["transition_id"]),
        run_id=cast(str, row["run_id"]),
        from_state=TaskRunState(cast(str, row["from_state"])),
        to_state=TaskRunState(cast(str, row["to_state"])),
        reason=cast(str, row["reason"]),
        created_at=_datetime(row["created_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _attempt_transition_from_row(
    row: Mapping[str, object],
) -> TaskAttemptTransition:
    return TaskAttemptTransition(
        transition_id=cast(str, row["transition_id"]),
        attempt_id=cast(str, row["attempt_id"]),
        run_id=cast(str, row["run_id"]),
        from_state=TaskAttemptState(cast(str, row["from_state"])),
        to_state=TaskAttemptState(cast(str, row["to_state"])),
        reason=cast(str, row["reason"]),
        created_at=_datetime(row["created_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _event_from_row(row: Mapping[str, object]) -> SanitizedTaskEvent:
    metadata = _mapping(row.get("metadata", {}))
    return SanitizedTaskEvent(
        event_id=cast(str, row["event_id"]),
        run_id=cast(str, row["run_id"]),
        attempt_id=cast(str | None, row.get("attempt_id")),
        sequence=cast(int, row["sequence"]),
        event_type=cast(str, row["event_type"]),
        category=TaskEventCategory(cast(str, metadata["category"])),
        payload=freeze_snapshot_value(row["payload"]),
        created_at=_datetime(row["created_at"]),
    )


def _usage_from_row(row: Mapping[str, object]) -> UsageRecord:
    metadata = _mapping(row["metadata"])
    return UsageRecord(
        usage_id=cast(str, row["usage_id"]),
        run_id=cast(str, row["run_id"]),
        attempt_id=cast(str | None, row.get("attempt_id")),
        sequence=cast(int, row["sequence"]),
        source=UsageSource(cast(str, row["source"])),
        totals=UsageTotals(
            input_tokens=cast(int | None, row.get("prompt_tokens")),
            cached_input_tokens=cast(int | None, row.get("cached_tokens")),
            cache_creation_input_tokens=cast(
                int | None,
                row.get(
                    "cache_creation_input_tokens",
                    metadata.get("cache_creation_input_tokens"),
                ),
            ),
            output_tokens=cast(int | None, row.get("completion_tokens")),
            reasoning_tokens=cast(
                int | None,
                row.get("reasoning_tokens", metadata.get("reasoning_tokens")),
            ),
            total_tokens=cast(int | None, row.get("total_tokens")),
        ),
        created_at=_datetime(row["created_at"]),
        metadata=freeze_snapshot_metadata(
            {
                key: value
                for key, value in metadata.items()
                if key
                not in {
                    "cache_creation_input_tokens",
                    "reasoning_tokens",
                }
            }
        ),
    )


def _artifact_from_row(row: Mapping[str, object]) -> TaskArtifactRecord:
    return TaskArtifactRecord(
        artifact_id=cast(str, row["artifact_id"]),
        run_id=cast(str, row["run_id"]),
        attempt_id=cast(str | None, row.get("attempt_id")),
        purpose=TaskArtifactPurpose(cast(str, row["purpose"])),
        state=TaskArtifactState(cast(str, row["state"])),
        ref=_artifact_ref_from_payload(_mapping(row["ref"])),
        provenance=_artifact_provenance_from_payload(
            _mapping(row["provenance"])
        ),
        retention=_artifact_retention_from_payload(_mapping(row["retention"])),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _idempotency_from_row(
    row: Mapping[str, object],
) -> TaskIdempotencyReservation:
    return TaskIdempotencyReservation(
        identity=TaskIdempotencyIdentity(
            identity_key=cast(str, row["identity_key"]),
            task_name=cast(str, row["task_name"]),
            task_version=cast(str, row["task_version"]),
            spec_hash=cast(str, row["spec_hash"]),
            owner_scope=_digest_from_payload(
                _mapping(row["owner_scope_hash"])
            ),
            strategy=IdempotencyMode(cast(str, row["strategy"])),
            window=(
                _digest_from_payload(_mapping(row["window_hash"]))
                if row.get("window_hash") is not None
                else None
            ),
            input=(
                _digest_from_payload(_mapping(row["input_hash"]))
                if row.get("input_hash") is not None
                else None
            ),
            files=(
                _digest_from_payload(_mapping(row["file_hash"]))
                if row.get("file_hash") is not None
                else None
            ),
            custom=(
                _digest_from_payload(_mapping(row["custom_hash"]))
                if row.get("custom_hash") is not None
                else None
            ),
        ),
        run_id=cast(str, row["run_id"]),
        created_at=_datetime(row["created_at"]),
        expires_at=(
            _datetime(row["expires_at"])
            if row.get("expires_at") is not None
            else None
        ),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _definition_to_payload(definition: TaskDefinition) -> dict[str, object]:
    return {
        "artifact": {
            "encrypt": definition.artifact.encrypt,
            "max_bytes": definition.artifact.max_bytes,
            "max_count": definition.artifact.max_count,
            "retention_days": definition.artifact.retention_days,
            "storage": definition.artifact.storage,
            "store_bytes": definition.artifact.store_bytes,
        },
        "execution": {
            "ref": definition.execution.ref,
            "type": definition.execution.type.value,
            "variables": _plain(definition.execution.variables),
        },
        "input": {
            "description": definition.input.description,
            "file_conversions": list(definition.input.file_conversions),
            "mime_types": list(definition.input.mime_types),
            "required": definition.input.required,
            "schema": _plain(definition.input.schema),
            "schema_ref": definition.input.schema_ref,
            "type": definition.input.type.value,
        },
        "limits": {
            "artifact_bytes": definition.limits.artifact_bytes,
            "artifact_count": definition.limits.artifact_count,
            "file_bytes": definition.limits.file_bytes,
            "file_count": definition.limits.file_count,
            "input_bytes": definition.limits.input_bytes,
            "output_bytes": definition.limits.output_bytes,
            "total_tokens": definition.limits.total_tokens,
        },
        "observability": {
            "capture_events": definition.observability.capture_events,
            "metrics": definition.observability.metrics,
            "sinks": [sink.value for sink in definition.observability.sinks],
            "trace": definition.observability.trace,
        },
        "output": {
            "description": definition.output.description,
            "schema": _plain(definition.output.schema),
            "schema_ref": definition.output.schema_ref,
            "type": definition.output.type.value,
        },
        "privacy": {
            "errors": definition.privacy.errors.value,
            "events": definition.privacy.events.value,
            "file_bytes": definition.privacy.file_bytes.value,
            "files": definition.privacy.files.value,
            "input": definition.privacy.input.value,
            "output": definition.privacy.output.value,
            "prompt": definition.privacy.prompt.value,
            "raw_retention_days": definition.privacy.raw_retention_days,
            "token_text": definition.privacy.token_text.value,
            "tool_arguments": definition.privacy.tool_arguments.value,
            "tool_results": definition.privacy.tool_results.value,
        },
        "retry": {
            "backoff": definition.retry.backoff.value,
            "jitter": definition.retry.jitter,
            "max_attempts": definition.retry.max_attempts,
            "max_delay_seconds": definition.retry.max_delay_seconds,
        },
        "run": {
            "concurrency": definition.run.concurrency,
            "idempotency": definition.run.idempotency.value,
            "idempotency_key_path": definition.run.idempotency_key_path,
            "mode": definition.run.mode.value,
            "priority": definition.run.priority,
            "queue": definition.run.queue,
            "timeout_seconds": definition.run.timeout_seconds,
        },
        "task": {
            "annotations": _plain(definition.task.annotations),
            "description": definition.task.description,
            "labels": list(definition.task.labels),
            "name": definition.task.name,
            "version": definition.task.version,
        },
    }


def _definition_from_payload(payload: Mapping[str, object]) -> TaskDefinition:
    task = _mapping(payload["task"])
    input_payload = _mapping(payload["input"])
    output_payload = _mapping(payload["output"])
    execution = _mapping(payload["execution"])
    run = _mapping(payload["run"])
    retry = _mapping(payload["retry"])
    privacy = _mapping(payload["privacy"])
    artifact = _mapping(payload["artifact"])
    limits = _mapping(payload["limits"])
    observability = _mapping(payload["observability"])
    return TaskDefinition(
        task=TaskMetadata(
            name=cast(str, task["name"]),
            version=cast(str, task["version"]),
            description=cast(str | None, task.get("description")),
            labels=tuple(cast(list[str], task.get("labels", []))),
            annotations=_mapping(task.get("annotations", {})),
        ),
        input=TaskInputContract(
            type=TaskInputType(cast(str, input_payload["type"])),
            schema=(
                _mapping(input_payload["schema"])
                if input_payload.get("schema") is not None
                else None
            ),
            schema_ref=cast(str | None, input_payload.get("schema_ref")),
            description=cast(str | None, input_payload.get("description")),
            required=cast(bool, input_payload.get("required", True)),
            file_conversions=tuple(
                cast(list[str], input_payload.get("file_conversions", []))
            ),
            mime_types=tuple(
                cast(list[str], input_payload.get("mime_types", []))
            ),
        ),
        output=TaskOutputContract(
            type=TaskOutputType(cast(str, output_payload["type"])),
            schema=(
                _mapping(output_payload["schema"])
                if output_payload.get("schema") is not None
                else None
            ),
            schema_ref=cast(str | None, output_payload.get("schema_ref")),
            description=cast(str | None, output_payload.get("description")),
        ),
        execution=TaskExecutionTarget(
            type=TaskTargetType(cast(str, execution["type"])),
            ref=cast(str, execution["ref"]),
            variables=_mapping(execution.get("variables", {})),
        ),
        run=TaskRunPolicy(
            mode=RunMode(cast(str, run.get("mode", RunMode.DIRECT.value))),
            timeout_seconds=cast(int, run.get("timeout_seconds", 300)),
            idempotency=IdempotencyMode(
                cast(str, run.get("idempotency", IdempotencyMode.NONE.value))
            ),
            queue=cast(str | None, run.get("queue")),
            priority=cast(int | None, run.get("priority")),
            concurrency=cast(int | None, run.get("concurrency")),
            idempotency_key_path=cast(
                str | None,
                run.get("idempotency_key_path"),
            ),
        ),
        retry=TaskRetryPolicy(
            max_attempts=cast(int, retry.get("max_attempts", 1)),
            backoff=RetryBackoff(
                cast(str, retry.get("backoff", RetryBackoff.NONE.value))
            ),
            max_delay_seconds=cast(int | None, retry.get("max_delay_seconds")),
            jitter=cast(bool, retry.get("jitter", False)),
        ),
        privacy=TaskPrivacyPolicy(
            input=PrivacyAction(cast(str, privacy["input"])),
            prompt=PrivacyAction(cast(str, privacy["prompt"])),
            output=PrivacyAction(cast(str, privacy["output"])),
            files=PrivacyAction(cast(str, privacy["files"])),
            file_bytes=PrivacyAction(cast(str, privacy["file_bytes"])),
            token_text=PrivacyAction(cast(str, privacy["token_text"])),
            tool_arguments=PrivacyAction(cast(str, privacy["tool_arguments"])),
            tool_results=PrivacyAction(cast(str, privacy["tool_results"])),
            events=PrivacyAction(cast(str, privacy["events"])),
            errors=PrivacyAction(cast(str, privacy["errors"])),
            raw_retention_days=cast(
                int,
                privacy.get("raw_retention_days", 0),
            ),
        ),
        artifact=TaskArtifactPolicy(
            retention_days=cast(int | None, artifact.get("retention_days")),
            store_bytes=cast(bool, artifact.get("store_bytes", False)),
            storage=cast(str | None, artifact.get("storage")),
            max_count=cast(int | None, artifact.get("max_count")),
            max_bytes=cast(int | None, artifact.get("max_bytes")),
            encrypt=cast(bool, artifact.get("encrypt", True)),
        ),
        limits=TaskLimitsPolicy(
            input_bytes=cast(int | None, limits.get("input_bytes")),
            file_count=cast(int | None, limits.get("file_count")),
            file_bytes=cast(int | None, limits.get("file_bytes")),
            output_bytes=cast(int | None, limits.get("output_bytes")),
            artifact_count=cast(int | None, limits.get("artifact_count")),
            artifact_bytes=cast(int | None, limits.get("artifact_bytes")),
            total_tokens=cast(int | None, limits.get("total_tokens")),
        ),
        observability=TaskObservabilityPolicy(
            sinks=tuple(
                ObservabilitySinkType(sink)
                for sink in cast(list[str], observability["sinks"])
            ),
            metrics=cast(bool, observability.get("metrics", True)),
            trace=cast(bool, observability.get("trace", True)),
            capture_events=cast(
                bool,
                observability.get("capture_events", True),
            ),
        ),
    )


def _request_to_payload(request: TaskExecutionRequest) -> dict[str, object]:
    return {
        "definition_id": request.definition_id,
        "file_summaries": _plain(request.file_summaries),
        "idempotency_key": request.idempotency_key,
        "input_summary": _plain(request.input_summary),
        "metadata": _plain(request.metadata),
        "queue": request.queue,
    }


def _request_from_payload(
    payload: Mapping[str, object],
) -> TaskExecutionRequest:
    file_summaries = cast(
        tuple[TaskSnapshotValue, ...],
        tuple(
            freeze_snapshot_value(item)
            for item in cast(
                list[object],
                payload.get("file_summaries", []),
            )
        ),
    )
    return TaskExecutionRequest(
        definition_id=cast(str, payload["definition_id"]),
        input_summary=freeze_snapshot_value(payload.get("input_summary")),
        file_summaries=file_summaries,
        idempotency_key=cast(str | None, payload.get("idempotency_key")),
        queue=cast(str | None, payload.get("queue")),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _result_to_payload(
    result: TaskExecutionResult | None,
) -> dict[str, object]:
    if result is None:
        return {}
    return {
        "error": _plain(result.error),
        "metadata": _plain(result.metadata),
        "output_summary": _plain(result.output_summary),
    }


def _result_from_payload(payload: Mapping[str, object]) -> TaskExecutionResult:
    return TaskExecutionResult(
        output_summary=freeze_snapshot_value(payload.get("output_summary")),
        error=freeze_snapshot_value(payload.get("error")),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _claim_to_payload(claim: TaskClaim) -> dict[str, object]:
    return {
        "claim_token": claim.claim_token,
        "claimed_at": claim.claimed_at.isoformat(),
        "heartbeat_at": (
            claim.heartbeat_at.isoformat() if claim.heartbeat_at else None
        ),
        "lease_expires_at": claim.lease_expires_at.isoformat(),
        "metadata": _plain(claim.metadata),
        "worker_id": claim.worker_id,
    }


def _claim_from_payload(payload: Mapping[str, object]) -> TaskClaim:
    return TaskClaim(
        worker_id=cast(str, payload["worker_id"]),
        claim_token=cast(str, payload["claim_token"]),
        claimed_at=_datetime(payload["claimed_at"]),
        lease_expires_at=_datetime(payload["lease_expires_at"]),
        heartbeat_at=(
            _datetime(payload["heartbeat_at"])
            if payload.get("heartbeat_at") is not None
            else None
        ),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _context_to_payload(context: TaskExecutionContext) -> dict[str, object]:
    return {
        "attempt_id": context.attempt_id,
        "attempt_number": context.attempt_number,
        "claim": _claim_to_payload(context.claim) if context.claim else None,
        "metadata": _plain(context.metadata),
        "run_id": context.run_id,
    }


def _context_from_payload(
    payload: Mapping[str, object],
) -> TaskExecutionContext:
    return TaskExecutionContext(
        run_id=cast(str, payload["run_id"]),
        attempt_id=cast(str, payload["attempt_id"]),
        attempt_number=cast(int, payload["attempt_number"]),
        claim=(
            _claim_from_payload(_mapping(payload["claim"]))
            if payload.get("claim") is not None
            else None
        ),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _artifact_ref_to_payload(ref: TaskArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": ref.artifact_id,
        "media_type": ref.media_type,
        "metadata": _plain(ref.metadata),
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
        "storage_key": ref.storage_key,
        "store": ref.store,
    }


def _artifact_ref_from_payload(
    payload: Mapping[str, object],
) -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id=cast(str, payload["artifact_id"]),
        store=cast(str, payload["store"]),
        storage_key=cast(str, payload["storage_key"]),
        media_type=cast(str | None, payload.get("media_type")),
        size_bytes=cast(int | None, payload.get("size_bytes")),
        sha256=cast(str | None, payload.get("sha256")),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _artifact_provenance_to_payload(
    provenance: TaskArtifactProvenance,
) -> dict[str, object]:
    return {
        "converter": provenance.converter,
        "metadata": _plain(provenance.metadata),
        "operation": provenance.operation,
        "source_artifact_id": provenance.source_artifact_id,
        "source_attempt_id": provenance.source_attempt_id,
        "source_run_id": provenance.source_run_id,
    }


def _artifact_provenance_from_payload(
    payload: Mapping[str, object],
) -> TaskArtifactProvenance:
    return TaskArtifactProvenance(
        source_artifact_id=cast(str | None, payload.get("source_artifact_id")),
        source_run_id=cast(str | None, payload.get("source_run_id")),
        source_attempt_id=cast(str | None, payload.get("source_attempt_id")),
        operation=cast(str | None, payload.get("operation")),
        converter=cast(str | None, payload.get("converter")),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _artifact_retention_to_payload(
    retention: TaskArtifactRetention,
) -> dict[str, object]:
    return {
        "delete_after_days": retention.delete_after_days,
        "expires_at": (
            retention.expires_at.isoformat() if retention.expires_at else None
        ),
        "metadata": _plain(retention.metadata),
        "retain_metadata": retention.retain_metadata,
    }


def _artifact_retention_from_payload(
    payload: Mapping[str, object],
) -> TaskArtifactRetention:
    return TaskArtifactRetention(
        expires_at=(
            _datetime(payload["expires_at"])
            if payload.get("expires_at") is not None
            else None
        ),
        delete_after_days=cast(int | None, payload.get("delete_after_days")),
        retain_metadata=cast(bool, payload.get("retain_metadata", True)),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _digest_from_payload(
    payload: Mapping[str, object],
) -> TaskIdempotencyDigest:
    return TaskIdempotencyDigest(
        algorithm=cast(str, payload["algorithm"]),
        digest=cast(str, payload["digest"]),
        key_id=cast(str, payload["key_id"]),
    )


def _verify_claim_token(run: TaskRun, claim_token: str | None) -> None:
    if run.claim is None:
        if claim_token is not None:
            raise TaskStoreConflictError("task claim token did not match")
        return
    if claim_token != run.claim.claim_token:
        raise TaskStoreConflictError("task claim token did not match")


def _row_value(
    row: Mapping[str, object] | None,
    key: str,
    default: object,
) -> object:
    if row is None:
        return default
    return row.get(key, default)


def _json(value: object) -> str:
    return dumps(
        _plain(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, str | bytes | bytearray):
        loaded = loads(value)
        assert isinstance(loaded, Mapping)
        return loaded
    assert isinstance(value, Mapping), "row value must be a mapping"
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return _ensure_aware_utc(value)
    assert isinstance(value, str), "datetime value must be a string"
    return _ensure_aware_utc(datetime.fromisoformat(value))


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_id() -> str:
    return uuid4().hex
