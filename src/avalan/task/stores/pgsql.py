from ...pgsql import (
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlUnitOfWork,
    assert_pgsql_identifier,
    classify_pgsql_error,
)
from ...skill import (
    SkillSettingsSurface,
    TrustedSkillSettings,
    UntrustedSkillSettings,
    parse_untrusted_skill_settings_config,
    untrusted_skill_settings_config_dict,
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
    TaskContainerExecutionSettings,
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
from ..error import TaskError
from ..event import (
    SanitizedTaskEvent,
    TaskEventCategory,
    TaskEventValue,
    TaskInteractionEventType,
    task_interaction_event_payload,
)
from ..feature_gate import ModuleFinder, TaskFeature, require_features
from ..idempotency import (
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskIdempotencyReservation,
    TaskIdempotencyReservationResult,
)
from ..queue import (
    TaskQueueCompletion,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueReentry,
    TaskQueueSuspension,
)
from ..settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
    TaskDurableResumeSuccess,
    task_durable_resume_settlement_digest,
)
from ..state import (
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskRunState,
    is_terminal_run_state,
)
from ..store import (
    TaskAttempt,
    TaskAttemptSegment,
    TaskAttemptSegmentTransition,
    TaskAttemptTransition,
    TaskClaim,
    TaskDefinitionRecord,
    TaskExecutionContext,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotValue,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
    TaskTransition,
    allows_cancel_request_without_claim_token,
    ensure_attempt_is_mutable,
    ensure_attempt_segment_is_mutable,
    ensure_run_is_mutable,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
    validate_attempt_segment_transition_request,
    validate_attempt_transition_request,
    validate_run_transition_request,
)
from ..usage import (
    UsageRecord,
    UsageSource,
    UsageTotals,
    aggregate_usage_totals,
    freeze_usage_metadata,
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
from threading import Lock
from typing import Any, Protocol, cast
from uuid import uuid4

TASK_PGSQL_ALEMBIC_VERSION_TABLE = "avalan_task_alembic_version"
TASK_PGSQL_HEAD_REVISION = "20260723_0002"
TASK_PGSQL_ADVISORY_LOCK_ID = 8_172_673_911_930_301_927
_TASK_PGSQL_REVISION_MODULES = (
    "avalan.task.stores.pgsql_migrations.versions.v20260530_0001_task_schema",
    (
        "avalan.task.stores.pgsql_migrations.versions."
        "v20260723_0002_durable_interactions"
    ),
)
_TASK_PGSQL_ALEMBIC_LOCK = Lock()

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

    @property
    def database(self) -> PgsqlDatabase:
        """Return the database used for explicit cross-store transactions."""
        return self._database

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
            if not allows_cancel_request_without_claim_token(
                run,
                to_state,
                claim_token,
            ):
                _verify_claim_token(run, claim_token)
            validate_run_transition_request(
                current_state=run.state,
                from_states=from_states,
                to_state=to_state,
            )
            effective_claim_token: str | None
            if allows_cancel_request_without_claim_token(
                run,
                to_state,
                claim_token,
            ):
                assert run.claim is not None
                effective_claim_token = run.claim.claim_token
            else:
                effective_claim_token = claim_token
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
                    effective_claim_token,
                    effective_claim_token,
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

    async def create_attempt_segment(
        self,
        attempt_id: str,
        *,
        claim_token: str | None = None,
        resumed_from_segment_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttemptSegment:
        _assert_non_empty_string(attempt_id, "attempt_id")
        if resumed_from_segment_id is not None:
            _assert_non_empty_string(
                resumed_from_segment_id,
                "resumed_from_segment_id",
            )

        async def execute(unit: PgsqlUnitOfWork) -> object:
            attempt = await _lock_attempt_or_raise(unit, attempt_id)
            run = await _lock_run_or_raise(unit, attempt.run_id)
            _verify_claim_token(run, claim_token)
            if attempt.state not in {
                TaskAttemptState.CREATED,
                TaskAttemptState.RUNNING,
                TaskAttemptState.SUSPENDED,
            }:
                raise TaskStoreConflictError(
                    "terminal task attempt cannot create a segment"
                )
            await unit.cursor.execute(
                _SELECT_SEGMENTS_FOR_ATTEMPT_SQL,
                (attempt_id,),
            )
            rows = await unit.cursor.fetchall()
            segments = tuple(_segment_from_row(row) for row in rows)
            if any(
                segment.state
                in {
                    TaskAttemptSegmentState.CREATED,
                    TaskAttemptSegmentState.RUNNING,
                }
                for segment in segments
            ):
                raise TaskStoreConflictError(
                    "task attempt already has an active segment"
                )
            previous = None
            if resumed_from_segment_id is not None:
                previous = next(
                    (
                        segment
                        for segment in segments
                        if segment.segment_id == resumed_from_segment_id
                    ),
                    None,
                )
                if (
                    previous is None
                    or previous.state is not TaskAttemptSegmentState.SUSPENDED
                ):
                    raise TaskStoreConflictError(
                        "resumed task segment was not suspended"
                    )
            elif attempt.state is TaskAttemptState.SUSPENDED:
                raise TaskStoreConflictError(
                    "suspended attempt requires a previous segment"
                )
            now = self._now()
            segment = TaskAttemptSegment(
                segment_id=self._new_id(),
                attempt_id=attempt_id,
                run_id=attempt.run_id,
                segment_number=len(segments) + 1,
                state=TaskAttemptSegmentState.CREATED,
                claim=run.claim,
                resumed_from_segment_id=resumed_from_segment_id,
                created_at=now,
                updated_at=now,
                metadata=freeze_snapshot_metadata(metadata),
            )
            await unit.cursor.execute(
                _INSERT_ATTEMPT_SEGMENT_SQL,
                (
                    segment.segment_id,
                    segment.attempt_id,
                    segment.run_id,
                    segment.segment_number,
                    segment.state.value,
                    (
                        _json(_claim_to_payload(segment.claim))
                        if segment.claim is not None
                        else None
                    ),
                    segment.resumed_from_segment_id,
                    _json(segment.metadata),
                    now,
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task attempt segment already exists"
                )
            return _segment_from_row(row)

        return cast(
            TaskAttemptSegment,
            await self._transaction(
                operation="task_attempt_segment_create",
                callback=execute,
            ),
        )

    async def get_attempt_segment(
        self,
        segment_id: str,
    ) -> TaskAttemptSegment:
        _assert_non_empty_string(segment_id, "segment_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await _segment_or_raise(unit, segment_id)

        return cast(
            TaskAttemptSegment,
            await self._transaction(
                operation="task_attempt_segment_get",
                callback=execute,
            ),
        )

    async def list_attempt_segments(
        self,
        attempt_id: str,
    ) -> tuple[TaskAttemptSegment, ...]:
        _assert_non_empty_string(attempt_id, "attempt_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _attempt_or_raise(unit, attempt_id)
            await unit.cursor.execute(
                _SELECT_SEGMENTS_FOR_ATTEMPT_SQL,
                (attempt_id,),
            )
            return tuple(
                _segment_from_row(row) for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskAttemptSegment, ...],
            await self._transaction(
                operation="task_attempt_segment_list",
                callback=execute,
            ),
        )

    async def transition_attempt_segment(
        self,
        segment_id: str,
        *,
        from_states: Collection[TaskAttemptSegmentState],
        to_state: TaskAttemptSegmentState,
        reason: str,
        request_id: str | None = None,
        continuation_id: str | None = None,
        checkpoint_id: str | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttemptSegment:
        _assert_non_empty_string(segment_id, "segment_id")
        _assert_non_empty_string(reason, "reason")
        if request_id is not None:
            _assert_non_empty_string(request_id, "request_id")
        if continuation_id is not None:
            _assert_non_empty_string(continuation_id, "continuation_id")
        if checkpoint_id is not None:
            _assert_non_empty_string(checkpoint_id, "checkpoint_id")
            if request_id is None:
                raise AssertionError(
                    "checkpoint identifiers require interaction correlation"
                )
        if (
            to_state is TaskAttemptSegmentState.SUSPENDED
            and checkpoint_id is None
        ):
            raise AssertionError(
                "suspended segments require a checkpoint identifier"
            )
        if (request_id is None) != (continuation_id is None):
            raise AssertionError(
                "request and continuation identifiers must be paired"
            )

        async def execute(unit: PgsqlUnitOfWork) -> object:
            segment = await _lock_segment_or_raise(unit, segment_id)
            ensure_attempt_segment_is_mutable(segment.state)
            run = await _lock_run_or_raise(unit, segment.run_id)
            _verify_claim_token(run, claim_token)
            validate_attempt_segment_transition_request(
                current_state=segment.state,
                from_states=from_states,
                to_state=to_state,
            )
            if to_state is TaskAttemptSegmentState.SUSPENDED:
                assert request_id is not None
                next_claim = None
            else:
                if request_id is not None:
                    raise TaskStoreConflictError(
                        "only suspended segments retain interactions"
                    )
                next_claim = segment.claim
            now = self._now()
            await unit.cursor.execute(
                _UPDATE_ATTEMPT_SEGMENT_SQL,
                (
                    to_state.value,
                    (
                        _json(_claim_to_payload(next_claim))
                        if next_claim is not None
                        else None
                    ),
                    request_id,
                    continuation_id,
                    checkpoint_id,
                    _json(metadata) if metadata is not None else None,
                    now,
                    segment_id,
                    segment.state.value,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                raise TaskStoreConflictError(
                    "task attempt segment state did not match"
                )
            await unit.cursor.execute(
                _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
                (
                    self._new_id(),
                    segment_id,
                    segment.attempt_id,
                    segment.run_id,
                    segment.state.value,
                    to_state.value,
                    reason,
                    _json(metadata or {}),
                    now,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "task attempt segment transition was not recorded"
                )
            return _segment_from_row(row)

        return cast(
            TaskAttemptSegment,
            await self._transaction(
                operation="task_attempt_segment_transition",
                callback=execute,
            ),
        )

    async def list_attempt_segment_transitions(
        self,
        segment_id: str,
    ) -> tuple[TaskAttemptSegmentTransition, ...]:
        _assert_non_empty_string(segment_id, "segment_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _segment_or_raise(unit, segment_id)
            await unit.cursor.execute(
                _SELECT_ATTEMPT_SEGMENT_TRANSITIONS_SQL,
                (segment_id,),
            )
            return tuple(
                _segment_transition_from_row(row)
                for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskAttemptSegmentTransition, ...],
            await self._transaction(
                operation="task_attempt_segment_transition_list",
                callback=execute,
            ),
        )

    async def suspend_claim(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        segment_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSuspension:
        _assert_non_empty_string(queue_item_id, "queue_item_id")
        _assert_non_empty_string(claim_token, "claim_token")
        _assert_non_empty_string(segment_id, "segment_id")
        _assert_non_empty_string(request_id, "request_id")
        _assert_non_empty_string(continuation_id, "continuation_id")
        if checkpoint_id is not None:
            _assert_non_empty_string(checkpoint_id, "checkpoint_id")
        else:
            raise AssertionError(
                "durable task suspension requires a checkpoint identifier"
            )
        observed_at = self._now() if now is None else _ensure_aware_utc(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await self._suspend_claim_in_unit(
                unit,
                queue_item_id=queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        return cast(
            TaskQueueSuspension,
            await self._transaction(
                operation="task_queue_suspend_claim",
                callback=execute,
            ),
        )

    async def _suspend_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueSuspension:
        """Suspend a claim inside an existing PostgreSQL transaction."""
        interaction_result = _input_required_execution_result(
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
        )
        result_payload = _json(_result_to_payload(interaction_result))
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if (
            queue_row["state"] != TaskQueueItemState.CLAIMED.value
            or queue_row["claim_token"] != claim_token
        ):
            raise TaskStoreConflictError(
                "task queue claim token did not match"
            )
        run = await _lock_run_or_raise(
            unit,
            cast(str, queue_row["run_id"]),
        )
        _verify_claim_token(run, claim_token)
        if run.state is not TaskRunState.RUNNING:
            raise TaskStoreConflictError("task run is not running")
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("running task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        if attempt.state is not TaskAttemptState.RUNNING:
            raise TaskStoreConflictError("task attempt is not running")
        segment = await _lock_segment_or_raise(unit, segment_id)
        if (
            segment.run_id != run.run_id
            or segment.attempt_id != attempt.attempt_id
            or segment.state is not TaskAttemptSegmentState.RUNNING
        ):
            raise TaskStoreConflictError(
                "task attempt segment is not the active running segment"
            )
        await unit.cursor.execute(
            _SELECT_DURABLE_CHECKPOINT_SQL,
            (
                request_id,
                continuation_id,
                run.run_id,
                checkpoint_id,
                checkpoint_id,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "durable interaction checkpoint is not ready"
            )
        await unit.cursor.execute(
            _SUSPEND_RUN_SQL,
            (
                TaskRunState.INPUT_REQUIRED.value,
                result_payload,
                _json(metadata),
                observed_at,
                run.run_id,
                TaskRunState.RUNNING.value,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task run suspension compare-and-swap failed"
            )
        await unit.cursor.execute(
            _SUSPEND_ATTEMPT_SQL,
            (
                TaskAttemptState.SUSPENDED.value,
                result_payload,
                _json(metadata),
                observed_at,
                attempt.attempt_id,
                TaskAttemptState.RUNNING.value,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task attempt suspension compare-and-swap failed"
            )
        await unit.cursor.execute(
            _SUSPEND_SEGMENT_SQL,
            (
                TaskAttemptSegmentState.SUSPENDED.value,
                request_id,
                continuation_id,
                checkpoint_id,
                _json(metadata),
                observed_at,
                segment.segment_id,
                TaskAttemptSegmentState.RUNNING.value,
            ),
        )
        segment_row = await unit.cursor.fetchone()
        if segment_row is None:
            raise TaskStoreConflictError(
                "task segment suspension compare-and-swap failed"
            )
        await unit.cursor.execute(
            _SUSPEND_QUEUE_ITEM_SQL,
            (
                TaskQueueItemState.SUSPENDED.value,
                attempt.attempt_id,
                segment.segment_id,
                request_id,
                continuation_id,
                _json(metadata),
                observed_at,
                queue_item_id,
                TaskQueueItemState.CLAIMED.value,
                claim_token,
            ),
        )
        suspended_queue_row = await unit.cursor.fetchone()
        if suspended_queue_row is None:
            raise TaskStoreConflictError(
                "task queue suspension compare-and-swap failed"
            )
        await _insert_suspension_transitions(
            unit,
            id_factory=self._new_id,
            run=run,
            attempt=attempt,
            segment=segment,
            now=observed_at,
            metadata=metadata,
        )
        await _insert_interaction_event(
            unit,
            id_factory=self._new_id,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            event_type=TaskInteractionEventType.INPUT_REQUIRED,
            request_id=request_id,
            continuation_id=continuation_id,
            segment_id=segment.segment_id,
            now=observed_at,
        )
        suspended_run = _run_from_row(run_row)
        suspended_attempt = _attempt_from_row(attempt_row)
        suspended_segment = _segment_from_row(segment_row)
        return TaskQueueSuspension(
            queue_item=_queue_item_from_row(
                suspended_queue_row,
                run_state=suspended_run.state,
            ),
            run=suspended_run,
            attempt=suspended_attempt,
            segment=suspended_segment,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
        )

    async def requeue_suspended(
        self,
        run_id: str,
        *,
        request_id: str,
        continuation_id: str,
        resolution_revision: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry:
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(request_id, "request_id")
        _assert_non_empty_string(continuation_id, "continuation_id")
        _assert_non_negative_int(resolution_revision, "resolution_revision")
        if resolution_revision == 0:
            raise AssertionError("resolution_revision must be positive")
        observed_at = self._now() if now is None else _ensure_aware_utc(now)
        safe_metadata = freeze_snapshot_metadata(metadata)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            return await self._requeue_suspended_in_unit(
                unit,
                run_id=run_id,
                request_id=request_id,
                continuation_id=continuation_id,
                resolution_revision=resolution_revision,
                observed_at=observed_at,
                metadata=safe_metadata,
            )

        return cast(
            TaskQueueReentry,
            await self._transaction(
                operation="task_queue_requeue_suspended",
                callback=execute,
            ),
        )

    async def _requeue_suspended_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        run_id: str,
        request_id: str,
        continuation_id: str,
        resolution_revision: int,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Requeue suspended work inside an existing transaction."""
        run = await _lock_run_or_raise(unit, run_id)
        await unit.cursor.execute(
            _SELECT_SUSPENDED_QUEUE_FOR_UPDATE_SQL,
            (run_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError(
                "suspended task queue item was not found"
            )
        attempt_id = cast(str | None, queue_row.get("attempt_id"))
        segment_id = cast(str | None, queue_row.get("segment_id"))
        if attempt_id is None or segment_id is None:
            raise TaskStoreConflictError(
                "suspended queue provenance is incomplete"
            )
        attempt = await _lock_attempt_or_raise(unit, attempt_id)
        segment = await _lock_segment_or_raise(unit, segment_id)
        if (
            attempt.state is not TaskAttemptState.SUSPENDED
            or segment.state is not TaskAttemptSegmentState.SUSPENDED
            or segment.request_id != request_id
            or segment.continuation_id != continuation_id
        ):
            raise TaskStoreConflictError(
                "suspended task provenance did not match"
            )
        await unit.cursor.execute(
            _SELECT_ACCEPTED_RESOLUTION_SQL,
            (
                request_id,
                continuation_id,
                run_id,
                resolution_revision,
            ),
        )
        outbox_row = await unit.cursor.fetchone()
        if outbox_row is None:
            raise TaskStoreConflictError(
                "accepted interaction resolution was not found"
            )
        if (
            run.state is TaskRunState.QUEUED
            and queue_row["state"] == TaskQueueItemState.AVAILABLE.value
        ):
            return TaskQueueReentry(
                queue_item=_queue_item_from_row(
                    queue_row,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
                previous_segment=segment,
            )
        if (
            run.state is not TaskRunState.INPUT_REQUIRED
            or queue_row["state"] != TaskQueueItemState.SUSPENDED.value
        ):
            raise TaskStoreConflictError(
                "task is not awaiting interaction input"
            )
        await unit.cursor.execute(
            _REQUEUE_RUN_SQL,
            (
                TaskRunState.QUEUED.value,
                _json(metadata),
                observed_at,
                run_id,
                TaskRunState.INPUT_REQUIRED.value,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task requeue compare-and-swap failed"
            )
        await unit.cursor.execute(
            _CLEAR_SUSPENDED_ATTEMPT_RESULT_SQL,
            (
                observed_at,
                attempt.attempt_id,
                TaskAttemptState.SUSPENDED.value,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task attempt reentry compare-and-swap failed"
            )
        await unit.cursor.execute(
            _REQUEUE_QUEUE_ITEM_SQL,
            (
                TaskQueueItemState.AVAILABLE.value,
                observed_at,
                _json(metadata),
                observed_at,
                queue_row["queue_item_id"],
                TaskQueueItemState.SUSPENDED.value,
            ),
        )
        available_queue_row = await unit.cursor.fetchone()
        if available_queue_row is None:
            raise TaskStoreConflictError(
                "task queue reentry compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                run_id,
                TaskRunState.INPUT_REQUIRED.value,
                TaskRunState.QUEUED.value,
                "interaction resolved",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task requeue transition was not recorded"
            )
        await _insert_interaction_event(
            unit,
            id_factory=self._new_id,
            run_id=run_id,
            attempt_id=attempt.attempt_id,
            event_type=TaskInteractionEventType.INPUT_RESUMED,
            request_id=request_id,
            continuation_id=continuation_id,
            segment_id=segment.segment_id,
            now=observed_at,
        )
        queued_run = _run_from_row(run_row)
        return TaskQueueReentry(
            queue_item=_queue_item_from_row(
                available_queue_row,
                run_state=queued_run.state,
            ),
            run=queued_run,
            attempt=_attempt_from_row(attempt_row),
            previous_segment=segment,
        )

    async def _terminalize_suspended_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        task_run_id: str,
        correlations: tuple[tuple[str, str], ...],
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        event_type: TaskInteractionEventType,
        reason: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
    ) -> TaskQueueCompletion | None:
        """Terminalize or replay one suspended task lifecycle."""
        if event_type in {
            TaskInteractionEventType.INPUT_CANCELLED,
            TaskInteractionEventType.INPUT_SUPERSEDED,
        }:
            if (
                run_state is not TaskRunState.CANCELLED
                or attempt_state is not TaskAttemptState.ABANDONED
            ):
                raise AssertionError(
                    "cancelled input requires cancelled run and "
                    "abandoned attempt"
                )
            error = TaskError.cancellation()
        elif event_type is TaskInteractionEventType.INPUT_EXPIRED:
            if (
                run_state is not TaskRunState.EXPIRED
                or attempt_state is not TaskAttemptState.FAILED
            ):
                raise AssertionError(
                    "expired input requires expired run and failed attempt"
                )
            error = TaskError.timeout()
        else:
            raise AssertionError(
                "event type must terminalize suspended task input"
            )
        if any(
            not request_id or not continuation_id
            for request_id, continuation_id in correlations
        ):
            raise AssertionError("task correlations must be non-empty")
        if len(set(correlations)) != len(correlations):
            raise AssertionError("task correlations must be unique")
        _assert_non_empty_string(task_run_id, "task_run_id")
        _assert_non_empty_string(reason, "reason")
        error_payload: dict[str, TaskSnapshotValue] = {
            "category": error.category.value,
            "code": error.code.value,
            "message": error.message,
            "retryable": error.retryable,
        }
        if error.details:
            error_payload["details"] = error.details
        result = TaskExecutionResult(
            error=error_payload,
            metadata={"interaction_event_type": event_type.value},
        )
        result_payload = _json(_result_to_payload(result))
        run = await _lock_run_or_raise(unit, task_run_id)
        await unit.cursor.execute(
            _SELECT_QUEUE_FOR_RUN_FOR_UPDATE_SQL,
            (task_run_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        attempt_id = cast(str | None, queue_row.get("attempt_id"))
        segment_id = cast(str | None, queue_row.get("segment_id"))
        if (
            run.last_attempt_id is None
            or attempt_id != run.last_attempt_id
            or segment_id is None
        ):
            raise TaskStoreConflictError(
                "suspended task provenance is incomplete"
            )
        attempt = await _lock_attempt_or_raise(unit, attempt_id)
        segment = await _lock_segment_or_raise(unit, segment_id)
        queue_request_id = cast(
            str | None,
            queue_row.get("request_id"),
        )
        queue_continuation_id = cast(
            str | None,
            queue_row.get("continuation_id"),
        )
        if run.state is run_state:
            return _replayed_suspended_task_lifecycle(
                queue_row=queue_row,
                run=run,
                attempt=attempt,
                attempt_state=attempt_state,
                segment=segment,
                result=result,
                correlations=correlations,
            )
        if replay_only:
            return None
        if (
            queue_request_id is None
            or queue_continuation_id is None
            or (queue_request_id, queue_continuation_id) not in correlations
        ):
            raise TaskStoreConflictError(
                "no terminal interaction matched the suspended queue"
            )
        if (
            run.state is not TaskRunState.INPUT_REQUIRED
            or run.claim is not None
            or queue_row["state"] != TaskQueueItemState.SUSPENDED.value
            or queue_row.get("claim_token") is not None
            or queue_row.get("lease_expires_at") is not None
            or attempt.run_id != task_run_id
            or attempt.state is not TaskAttemptState.SUSPENDED
            or attempt.context.claim is not None
            or segment.run_id != task_run_id
            or segment.attempt_id != attempt.attempt_id
            or segment.state is not TaskAttemptSegmentState.SUSPENDED
            or segment.claim is not None
            or segment.request_id != queue_request_id
            or segment.continuation_id != queue_continuation_id
        ):
            raise TaskStoreConflictError(
                "suspended task lifecycle provenance did not match"
            )
        await unit.cursor.execute(
            _UPDATE_ATTEMPT_STATE_SQL,
            (
                attempt_state.value,
                result_payload,
                observed_at,
                attempt.attempt_id,
                TaskAttemptState.SUSPENDED.value,
                task_run_id,
                TaskRunState.INPUT_REQUIRED.value,
                None,
                None,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task attempt lifecycle compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                self._new_id(),
                attempt.attempt_id,
                task_run_id,
                TaskAttemptState.SUSPENDED.value,
                attempt_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task attempt lifecycle transition was not recorded"
            )
        if run_state is TaskRunState.CANCELLED:
            await unit.cursor.execute(
                _UPDATE_RUN_STATE_SQL,
                (
                    TaskRunState.CANCEL_REQUESTED.value,
                    None,
                    False,
                    observed_at,
                    task_run_id,
                    TaskRunState.INPUT_REQUIRED.value,
                    None,
                    None,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "task cancellation request compare-and-swap failed"
                )
            await unit.cursor.execute(
                _INSERT_RUN_TRANSITION_SQL,
                (
                    self._new_id(),
                    task_run_id,
                    TaskRunState.INPUT_REQUIRED.value,
                    TaskRunState.CANCEL_REQUESTED.value,
                    "cancel_requested",
                    _json(metadata),
                    observed_at,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "task cancellation request transition was not recorded"
                )
            run_from_state = TaskRunState.CANCEL_REQUESTED
        else:
            run_from_state = TaskRunState.INPUT_REQUIRED
        await unit.cursor.execute(
            _UPDATE_RUN_STATE_SQL,
            (
                run_state.value,
                result_payload,
                True,
                observed_at,
                task_run_id,
                run_from_state.value,
                None,
                None,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task run lifecycle compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                run_from_state.value,
                run_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task run lifecycle transition was not recorded"
            )
        await unit.cursor.execute(
            _TERMINALIZE_SUSPENDED_QUEUE_SQL,
            (
                TaskQueueItemState.DEAD.value,
                _json(metadata),
                observed_at,
                queue_row["queue_item_id"],
                TaskQueueItemState.SUSPENDED.value,
            ),
        )
        terminal_queue_row = await unit.cursor.fetchone()
        if terminal_queue_row is None:
            raise TaskStoreConflictError(
                "task queue lifecycle compare-and-swap failed"
            )
        await _insert_interaction_event(
            unit,
            id_factory=self._new_id,
            run_id=task_run_id,
            attempt_id=attempt.attempt_id,
            event_type=event_type,
            request_id=queue_request_id,
            continuation_id=queue_continuation_id,
            segment_id=segment.segment_id,
            now=observed_at,
        )
        terminal_run = _run_from_row(run_row)
        return TaskQueueCompletion(
            queue_item=_queue_item_from_row(
                terminal_queue_row,
                run_state=terminal_run.state,
            ),
            run=terminal_run,
            attempt=_attempt_from_row(attempt_row),
        )

    async def _validate_suspended_run_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        task_run_id: str,
    ) -> None:
        """Validate one task can accept another durable interaction."""
        _assert_non_empty_string(task_run_id, "task_run_id")
        run = await _lock_run_or_raise(unit, task_run_id)
        await unit.cursor.execute(
            _SELECT_QUEUE_FOR_RUN_FOR_UPDATE_SQL,
            (task_run_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        attempt_id = cast(str | None, queue_row.get("attempt_id"))
        segment_id = cast(str | None, queue_row.get("segment_id"))
        if (
            run.state is not TaskRunState.INPUT_REQUIRED
            or run.claim is not None
            or run.last_attempt_id is None
            or attempt_id != run.last_attempt_id
            or segment_id is None
            or queue_row["state"] != TaskQueueItemState.SUSPENDED.value
            or queue_row.get("claim_token") is not None
            or queue_row.get("lease_expires_at") is not None
        ):
            raise TaskStoreConflictError(
                "task is not at a durable suspended boundary"
            )
        attempt = await _lock_attempt_or_raise(unit, attempt_id)
        segment = await _lock_segment_or_raise(unit, segment_id)
        if (
            attempt.run_id != task_run_id
            or attempt.state is not TaskAttemptState.SUSPENDED
            or attempt.context.claim is not None
            or segment.run_id != task_run_id
            or segment.attempt_id != attempt_id
            or segment.state is not TaskAttemptSegmentState.SUSPENDED
            or segment.claim is not None
        ):
            raise TaskStoreConflictError(
                "task suspension provenance did not match"
            )

    async def _settle_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        settlement: TaskDurableResumeSettlement,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
        allow_expired_lease: bool = False,
        terminal_run_state: TaskRunState | None = None,
        terminal_reason: str | None = None,
        interaction_event_type: TaskInteractionEventType | None = None,
        interaction_request_id: str | None = None,
        interaction_continuation_id: str | None = None,
    ) -> TaskQueueCompletion:
        """Settle one resumed task claim inside an existing transaction."""
        if type(settlement) is TaskDurableResumeSuccess:
            run_state = TaskRunState.SUCCEEDED
            attempt_state = TaskAttemptState.SUCCEEDED
            segment_state = TaskAttemptSegmentState.SUCCEEDED
            queue_state = TaskQueueItemState.DONE
            reason = "completed"
        elif type(settlement) is TaskDurableResumeFailure:
            run_state = TaskRunState.FAILED
            attempt_state = TaskAttemptState.FAILED
            segment_state = TaskAttemptSegmentState.FAILED
            queue_state = TaskQueueItemState.DEAD
            reason = "execution_failed"
        elif type(settlement) is TaskDurableResumeCancellation:
            run_state = TaskRunState.CANCELLED
            attempt_state = TaskAttemptState.FAILED
            segment_state = TaskAttemptSegmentState.ABANDONED
            queue_state = TaskQueueItemState.DEAD
            reason = "cancelled"
        else:
            raise AssertionError(
                "settlement must be a durable resume success or failure"
            )
        if terminal_run_state is not None:
            if (
                type(settlement) is not TaskDurableResumeFailure
                or terminal_run_state is not TaskRunState.EXPIRED
                or terminal_reason is None
            ):
                raise AssertionError(
                    "only failed resume settlement may expire a task"
                )
            run_state = terminal_run_state
            reason = terminal_reason
        correlation = (
            interaction_request_id,
            interaction_continuation_id,
        )
        if interaction_event_type is None:
            if any(value is not None for value in correlation):
                raise AssertionError(
                    "interaction event correlation requires an event type"
                )
        elif (
            interaction_event_type
            is not TaskInteractionEventType.INPUT_EXPIRED
            or any(value is None for value in correlation)
        ):
            raise AssertionError(
                "expired interaction event requires complete correlation"
            )
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if cast(str, queue_row["run_id"]) != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        segment = await _lock_segment_or_raise(unit, segment_id)
        if (
            attempt.run_id != task_run_id
            or segment.run_id != task_run_id
            or segment.attempt_id != attempt.attempt_id
        ):
            raise TaskStoreConflictError(
                "resumed task settlement provenance did not match"
            )
        cancellation_won = run.state is TaskRunState.CANCEL_REQUESTED or (
            run.state is TaskRunState.CANCELLED
            and attempt.state is TaskAttemptState.ABANDONED
        )
        if cancellation_won:
            run_state = TaskRunState.CANCELLED
            attempt_state = TaskAttemptState.ABANDONED
            segment_state = TaskAttemptSegmentState.ABANDONED
            queue_state = TaskQueueItemState.DEAD
            reason = "cancelled"
            result = _cancelled_task_settlement_result(settlement)
        else:
            result = settlement.result
        result_payload = _json(_result_to_payload(result))
        if run.state is run_state:
            return _replayed_task_settlement(
                queue_row=queue_row,
                queue_state=queue_state,
                run=run,
                attempt=attempt,
                attempt_state=attempt_state,
                segment=segment,
                segment_state=segment_state,
                result=result,
                claim_token=claim_token,
            )
        if replay_only:
            raise TaskStoreConflictError(
                "completed continuation has no matching terminal task"
            )
        if (
            queue_row["state"] != TaskQueueItemState.CLAIMED.value
            or queue_row["claim_token"] != claim_token
        ):
            raise TaskStoreConflictError(
                "task queue claim token did not match"
            )
        lease_expires_at = queue_row.get("lease_expires_at")
        if not isinstance(lease_expires_at, datetime) or (
            lease_expires_at <= observed_at
            if not allow_expired_lease
            else lease_expires_at > observed_at
        ):
            raise TaskStoreConflictError("task queue claim lease expired")
        _verify_claim_token(run, claim_token)
        if run.state not in {
            TaskRunState.RUNNING,
            TaskRunState.CANCEL_REQUESTED,
        }:
            raise TaskStoreConflictError("resumed task run is not running")
        if attempt.state is not TaskAttemptState.RUNNING:
            raise TaskStoreConflictError("resumed task attempt is not running")
        if segment.state is not TaskAttemptSegmentState.RUNNING:
            raise TaskStoreConflictError("resumed task segment is not running")
        if segment.claim is None or segment.claim.claim_token != claim_token:
            raise TaskStoreConflictError(
                "resumed task segment claim did not match"
            )
        await unit.cursor.execute(
            _UPDATE_ATTEMPT_SEGMENT_SQL,
            (
                segment_state.value,
                _json(_claim_to_payload(segment.claim)),
                None,
                None,
                None,
                _json(metadata),
                observed_at,
                segment.segment_id,
                TaskAttemptSegmentState.RUNNING.value,
            ),
        )
        segment_row = await unit.cursor.fetchone()
        if segment_row is None:
            raise TaskStoreConflictError(
                "task segment settlement compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
            (
                self._new_id(),
                segment.segment_id,
                attempt.attempt_id,
                task_run_id,
                TaskAttemptSegmentState.RUNNING.value,
                segment_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task segment settlement transition was not recorded"
            )
        await unit.cursor.execute(
            _UPDATE_ATTEMPT_STATE_SQL,
            (
                attempt_state.value,
                result_payload,
                observed_at,
                attempt.attempt_id,
                TaskAttemptState.RUNNING.value,
                task_run_id,
                run.state.value,
                claim_token,
                claim_token,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task attempt settlement compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                self._new_id(),
                attempt.attempt_id,
                task_run_id,
                TaskAttemptState.RUNNING.value,
                attempt_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task attempt settlement transition was not recorded"
            )
        if run.state is TaskRunState.CANCEL_REQUESTED:
            run_from_state = TaskRunState.CANCEL_REQUESTED
        elif type(settlement) is TaskDurableResumeCancellation:
            await unit.cursor.execute(
                _UPDATE_RUN_STATE_SQL,
                (
                    TaskRunState.CANCEL_REQUESTED.value,
                    None,
                    False,
                    observed_at,
                    task_run_id,
                    TaskRunState.RUNNING.value,
                    claim_token,
                    claim_token,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "task cancellation request compare-and-swap failed"
                )
            await unit.cursor.execute(
                _INSERT_RUN_TRANSITION_SQL,
                (
                    self._new_id(),
                    task_run_id,
                    TaskRunState.RUNNING.value,
                    TaskRunState.CANCEL_REQUESTED.value,
                    "cancel_requested",
                    _json(metadata),
                    observed_at,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "task cancellation request transition was not recorded"
                )
            run_from_state = TaskRunState.CANCEL_REQUESTED
        else:
            run_from_state = TaskRunState.RUNNING
        await unit.cursor.execute(
            _UPDATE_RUN_STATE_SQL,
            (
                run_state.value,
                result_payload,
                True,
                observed_at,
                task_run_id,
                run_from_state.value,
                claim_token,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task run settlement compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                run_from_state.value,
                run_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task run settlement transition was not recorded"
            )
        await unit.cursor.execute(
            _SETTLE_QUEUE_ITEM_SQL,
            (
                queue_state.value,
                observed_at,
                queue_item_id,
                claim_token,
            ),
        )
        settled_queue_row = await unit.cursor.fetchone()
        if settled_queue_row is None:
            raise TaskStoreConflictError(
                "task queue settlement compare-and-swap failed"
            )
        if interaction_event_type is not None and not cancellation_won:
            assert interaction_request_id is not None
            assert interaction_continuation_id is not None
            await _insert_interaction_event(
                unit,
                id_factory=self._new_id,
                run_id=task_run_id,
                attempt_id=attempt.attempt_id,
                event_type=interaction_event_type,
                request_id=interaction_request_id,
                continuation_id=interaction_continuation_id,
                segment_id=segment.segment_id,
                now=observed_at,
            )
        settled_run = _run_from_row(run_row)
        return TaskQueueCompletion(
            queue_item=_queue_item_from_row(
                settled_queue_row,
                run_state=settled_run.state,
            ),
            run=settled_run,
            attempt=_attempt_from_row(attempt_row),
        )

    async def _terminalize_completed_claim_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        settlement: TaskDurableResumeFailure | TaskDurableResumeCancellation,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Terminalize one task after its provider continuation completed."""
        if type(settlement) not in {
            TaskDurableResumeFailure,
            TaskDurableResumeCancellation,
        }:
            raise AssertionError(
                "completed provider task settlement must not succeed"
            )
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if cast(str, queue_row["run_id"]) != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        segment = await _lock_segment_or_raise(unit, segment_id)
        previous_segment_id = cast(
            str | None,
            queue_row.get("segment_id"),
        )
        if previous_segment_id is None:
            raise TaskStoreConflictError(
                "completed provider task has no suspension provenance"
            )
        previous_segment = await _lock_segment_or_raise(
            unit,
            previous_segment_id,
        )
        if (
            queue_row.get("attempt_id") != attempt.attempt_id
            or queue_row.get("request_id") != request_id
            or queue_row.get("continuation_id") != continuation_id
            or attempt.run_id != task_run_id
            or attempt.context.claim is None
            or attempt.context.claim.claim_token != claim_token
            or segment.run_id != task_run_id
            or segment.attempt_id != attempt.attempt_id
            or segment.resumed_from_segment_id != previous_segment.segment_id
            or segment.request_id is not None
            or segment.continuation_id is not None
            or segment.checkpoint_id is not None
            or segment.claim is None
            or segment.claim.claim_token != claim_token
            or previous_segment.run_id != task_run_id
            or previous_segment.attempt_id != attempt.attempt_id
            or previous_segment.state is not TaskAttemptSegmentState.SUSPENDED
            or previous_segment.claim is not None
            or previous_segment.request_id != request_id
            or previous_segment.continuation_id != continuation_id
            or previous_segment.checkpoint_id != checkpoint_id
        ):
            raise TaskStoreConflictError(
                "completed provider task provenance did not match"
            )
        return await self._settle_claim_in_unit(
            unit,
            queue_item_id=queue_item_id,
            claim_token=claim_token,
            segment_id=segment_id,
            task_run_id=task_run_id,
            settlement=settlement,
            observed_at=observed_at,
            metadata=metadata,
        )

    async def _release_claimed_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Release one exact claimed reentry inside a transaction."""
        await unit.cursor.execute(
            _SELECT_RELEASED_CONTINUATION_SQL,
            (
                request_id,
                continuation_id,
                task_run_id,
                checkpoint_id,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "durable continuation is not safe to requeue"
            )
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if cast(str, queue_row["run_id"]) != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        segment_id = cast(str | None, queue_row.get("segment_id"))
        if segment_id is None:
            raise TaskStoreConflictError(
                "claimed reentry has no suspended segment"
            )
        segment = await _lock_segment_or_raise(unit, segment_id)
        _validate_reentry_provenance(
            queue_row=queue_row,
            attempt=attempt,
            segment=segment,
            task_run_id=task_run_id,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
        )
        if (
            run.state is TaskRunState.QUEUED
            and queue_row["state"] == TaskQueueItemState.AVAILABLE.value
        ):
            return TaskQueueReentry(
                queue_item=_queue_item_from_row(
                    queue_row,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
                previous_segment=segment,
            )
        if (
            run.state is not TaskRunState.CLAIMED
            or queue_row["state"] != TaskQueueItemState.CLAIMED.value
            or queue_row.get("claim_token") != claim_token
            or attempt.state is not TaskAttemptState.SUSPENDED
        ):
            raise TaskStoreConflictError("task reentry claim did not match")
        _verify_claim_token(run, claim_token)
        await unit.cursor.execute(
            _RELEASE_REENTRY_ATTEMPT_SQL,
            (
                _json(metadata),
                observed_at,
                attempt.attempt_id,
                task_run_id,
                claim_token,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task reentry attempt release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _RELEASE_REENTRY_RUN_SQL,
            (
                TaskRunState.QUEUED.value,
                _json(metadata),
                observed_at,
                task_run_id,
                TaskRunState.CLAIMED.value,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task reentry run release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _RELEASE_REENTRY_QUEUE_SQL,
            (
                TaskQueueItemState.AVAILABLE.value,
                observed_at,
                _json(metadata),
                observed_at,
                queue_item_id,
                TaskQueueItemState.CLAIMED.value,
                claim_token,
            ),
        )
        queue_result = await unit.cursor.fetchone()
        if queue_result is None:
            raise TaskStoreConflictError(
                "task reentry queue release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                TaskRunState.CLAIMED.value,
                TaskRunState.QUEUED.value,
                "resume_released",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task reentry release transition was not recorded"
            )
        queued_run = _run_from_row(run_row)
        return TaskQueueReentry(
            queue_item=_queue_item_from_row(
                queue_result,
                run_state=queued_run.state,
            ),
            run=queued_run,
            attempt=_attempt_from_row(attempt_row),
            previous_segment=segment,
        )

    async def _fail_claimed_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str | None,
        continuation_id: str | None,
        checkpoint_id: str | None,
        result: TaskExecutionResult,
        reason: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
        replay_only: bool = False,
        terminal_run_state: TaskRunState = TaskRunState.FAILED,
        interaction_event_type: TaskInteractionEventType | None = None,
    ) -> TaskQueueCompletion:
        """Fail one claimed or malformed reentry inside a transaction."""
        if terminal_run_state not in {
            TaskRunState.FAILED,
            TaskRunState.EXPIRED,
        }:
            raise AssertionError("failed reentry must fail or expire its task")
        if interaction_event_type is not None and (
            interaction_event_type
            is not TaskInteractionEventType.INPUT_EXPIRED
            or terminal_run_state is not TaskRunState.EXPIRED
        ):
            raise AssertionError(
                "expired reentry event requires an expired task"
            )
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if cast(str, queue_row["run_id"]) != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        exact_provenance = request_id is not None
        segment: TaskAttemptSegment | None = None
        if exact_provenance:
            assert continuation_id is not None
            assert checkpoint_id is not None
            segment_id = cast(str | None, queue_row.get("segment_id"))
            if segment_id is None:
                raise TaskStoreConflictError(
                    "claimed reentry has no suspended segment"
                )
            segment = await _lock_segment_or_raise(unit, segment_id)
            _validate_reentry_provenance(
                queue_row=queue_row,
                attempt=attempt,
                segment=segment,
                task_run_id=task_run_id,
                request_id=cast(str, request_id),
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
            )
        if run.state is terminal_run_state:
            return _replayed_failed_reentry(
                queue_row=queue_row,
                run=run,
                attempt=attempt,
                result=result,
            )
        if replay_only:
            raise TaskStoreConflictError(
                "invalidated continuation has no matching failed task"
            )
        if (
            run.state is not TaskRunState.CLAIMED
            or queue_row["state"] != TaskQueueItemState.CLAIMED.value
            or queue_row.get("claim_token") != claim_token
            or attempt.state is not TaskAttemptState.SUSPENDED
        ):
            raise TaskStoreConflictError("task reentry claim did not match")
        _verify_claim_token(run, claim_token)
        result_payload = _json(_result_to_payload(result))
        await unit.cursor.execute(
            _FAIL_REENTRY_ATTEMPT_SQL,
            (
                TaskAttemptState.FAILED.value,
                result_payload,
                _json(metadata),
                observed_at,
                attempt.attempt_id,
                task_run_id,
                claim_token,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "task reentry failure attempt compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                self._new_id(),
                attempt.attempt_id,
                task_run_id,
                TaskAttemptState.SUSPENDED.value,
                TaskAttemptState.FAILED.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task reentry failure transition was not recorded"
            )
        await unit.cursor.execute(
            _FAIL_REENTRY_RUN_SQL,
            (
                terminal_run_state.value,
                result_payload,
                _json(metadata),
                observed_at,
                task_run_id,
                TaskRunState.CLAIMED.value,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "task reentry failure run compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                TaskRunState.CLAIMED.value,
                terminal_run_state.value,
                reason,
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task reentry failure run transition was not recorded"
            )
        await unit.cursor.execute(
            _FAIL_REENTRY_QUEUE_SQL,
            (
                TaskQueueItemState.DEAD.value,
                _json(metadata),
                observed_at,
                queue_item_id,
                TaskQueueItemState.CLAIMED.value,
                claim_token,
            ),
        )
        queue_result = await unit.cursor.fetchone()
        if queue_result is None:
            raise TaskStoreConflictError(
                "task reentry failure queue compare-and-swap failed"
            )
        if interaction_event_type is not None:
            if (
                request_id is None
                or continuation_id is None
                or segment is None
            ):
                raise AssertionError(
                    "expired reentry event requires exact provenance"
                )
            await _insert_interaction_event(
                unit,
                id_factory=self._new_id,
                run_id=task_run_id,
                attempt_id=attempt.attempt_id,
                event_type=interaction_event_type,
                request_id=request_id,
                continuation_id=continuation_id,
                segment_id=segment.segment_id,
                now=observed_at,
            )
        failed_run = _run_from_row(run_row)
        return TaskQueueCompletion(
            queue_item=_queue_item_from_row(
                queue_result,
                run_state=failed_run.state,
            ),
            run=failed_run,
            attempt=_attempt_from_row(attempt_row),
        )

    async def _release_running_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueReentry:
        """Release one running pre-dispatch reentry inside a transaction."""
        await unit.cursor.execute(
            _SELECT_RELEASED_CONTINUATION_SQL,
            (
                request_id,
                continuation_id,
                task_run_id,
                checkpoint_id,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "durable continuation is not safe to requeue"
            )
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if cast(str, queue_row["run_id"]) != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        segment = await _lock_segment_or_raise(unit, segment_id)
        if (
            attempt.run_id != task_run_id
            or segment.run_id != task_run_id
            or segment.attempt_id != attempt.attempt_id
        ):
            raise TaskStoreConflictError(
                "running task reentry provenance did not match"
            )
        if (
            run.state is TaskRunState.QUEUED
            and queue_row["state"] == TaskQueueItemState.AVAILABLE.value
        ):
            _validate_released_running_reentry(
                queue_row=queue_row,
                attempt=attempt,
                segment=segment,
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id=checkpoint_id,
            )
            return TaskQueueReentry(
                queue_item=_queue_item_from_row(
                    queue_row,
                    run_state=run.state,
                ),
                run=run,
                attempt=attempt,
                previous_segment=segment,
            )
        if (
            run.state is not TaskRunState.RUNNING
            or queue_row["state"] != TaskQueueItemState.CLAIMED.value
            or queue_row.get("claim_token") != claim_token
            or attempt.state is not TaskAttemptState.RUNNING
            or segment.state is not TaskAttemptSegmentState.RUNNING
            or segment.claim is None
            or segment.claim.claim_token != claim_token
        ):
            raise TaskStoreConflictError(
                "running task reentry claim did not match"
            )
        _verify_claim_token(run, claim_token)
        await unit.cursor.execute(
            _RELEASE_RUNNING_SEGMENT_SQL,
            (
                TaskAttemptSegmentState.SUSPENDED.value,
                request_id,
                continuation_id,
                checkpoint_id,
                _json(metadata),
                observed_at,
                segment_id,
                TaskAttemptSegmentState.RUNNING.value,
            ),
        )
        segment_row = await unit.cursor.fetchone()
        if segment_row is None:
            raise TaskStoreConflictError(
                "running task segment release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
            (
                self._new_id(),
                segment_id,
                attempt.attempt_id,
                task_run_id,
                TaskAttemptSegmentState.RUNNING.value,
                TaskAttemptSegmentState.SUSPENDED.value,
                "resume_released",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "running task segment release transition was not recorded"
            )
        await unit.cursor.execute(
            _RELEASE_RUNNING_ATTEMPT_SQL,
            (
                TaskAttemptState.SUSPENDED.value,
                _json(metadata),
                observed_at,
                attempt.attempt_id,
                task_run_id,
                claim_token,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "running task attempt release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                self._new_id(),
                attempt.attempt_id,
                task_run_id,
                TaskAttemptState.RUNNING.value,
                TaskAttemptState.SUSPENDED.value,
                "resume_released",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "running task attempt release transition was not recorded"
            )
        await unit.cursor.execute(
            _RELEASE_RUNNING_RUN_SQL,
            (
                TaskRunState.QUEUED.value,
                _json(metadata),
                observed_at,
                task_run_id,
                TaskRunState.RUNNING.value,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "running task run release compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                TaskRunState.RUNNING.value,
                TaskRunState.QUEUED.value,
                "resume_released",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "running task run release transition was not recorded"
            )
        await unit.cursor.execute(
            _RELEASE_RUNNING_QUEUE_SQL,
            (
                TaskQueueItemState.AVAILABLE.value,
                observed_at,
                attempt.attempt_id,
                segment_id,
                request_id,
                continuation_id,
                _json(metadata),
                observed_at,
                queue_item_id,
                TaskQueueItemState.CLAIMED.value,
                claim_token,
            ),
        )
        queue_result = await unit.cursor.fetchone()
        if queue_result is None:
            raise TaskStoreConflictError(
                "running task queue release compare-and-swap failed"
            )
        queued_run = _run_from_row(run_row)
        return TaskQueueReentry(
            queue_item=_queue_item_from_row(
                queue_result,
                run_state=queued_run.state,
            ),
            run=queued_run,
            attempt=_attempt_from_row(attempt_row),
            previous_segment=_segment_from_row(segment_row),
        )

    async def _cancel_partial_reentry_in_unit(
        self,
        unit: PgsqlUnitOfWork,
        *,
        queue_item_id: str,
        claim_token: str,
        active_segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        result: TaskExecutionResult,
        observed_at: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Cancel one exact partial resumed startup or replay it."""
        await unit.cursor.execute(
            _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
            (queue_item_id,),
        )
        queue_row = await unit.cursor.fetchone()
        if queue_row is None:
            raise TaskStoreNotFoundError("task queue item was not found")
        if queue_row.get("run_id") != task_run_id:
            raise TaskStoreConflictError(
                "task queue item does not belong to the resumed run"
            )
        run = await _lock_run_or_raise(unit, task_run_id)
        if run.last_attempt_id is None:
            raise TaskStoreConflictError("resumed task has no active attempt")
        attempt = await _lock_attempt_or_raise(
            unit,
            run.last_attempt_id,
        )
        previous_segment_id = queue_row.get("segment_id")
        if not isinstance(previous_segment_id, str) or not previous_segment_id:
            raise TaskStoreConflictError(
                "partial task reentry has no suspension provenance"
            )
        previous_segment = await _lock_segment_or_raise(
            unit,
            previous_segment_id,
        )
        active_segment = await _lock_segment_or_raise(
            unit,
            active_segment_id,
        )
        _validate_partial_reentry_cancellation_provenance(
            queue_row=queue_row,
            attempt=attempt,
            previous_segment=previous_segment,
            active_segment=active_segment,
            task_run_id=task_run_id,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
        )
        cancellation_result = _cancelled_task_settlement_result(
            TaskDurableResumeFailure(result=result)
        )
        if run.state is TaskRunState.CANCELLED:
            return _replayed_partial_reentry_cancellation(
                queue_row=queue_row,
                run=run,
                attempt=attempt,
                previous_segment=previous_segment,
                active_segment=active_segment,
                result=cancellation_result,
                claim_token=claim_token,
            )
        attempt_claim = attempt.context.claim
        if (
            run.state is not TaskRunState.CANCEL_REQUESTED
            or queue_row.get("state") != TaskQueueItemState.CLAIMED.value
            or queue_row.get("claim_token") != claim_token
            or attempt_claim is None
            or attempt_claim.claim_token != claim_token
        ):
            raise TaskStoreConflictError(
                "partial task reentry cancellation claim did not match"
            )
        _verify_claim_token(run, claim_token)
        partial_states = (
            attempt.state,
            active_segment.state,
        )
        if active_segment.segment_id == previous_segment.segment_id:
            if partial_states not in {
                (
                    TaskAttemptState.SUSPENDED,
                    TaskAttemptSegmentState.SUSPENDED,
                ),
                (
                    TaskAttemptState.RUNNING,
                    TaskAttemptSegmentState.SUSPENDED,
                ),
            }:
                raise TaskStoreConflictError(
                    "partial task reentry startup state did not match"
                )
            if active_segment.claim is not None:
                raise TaskStoreConflictError(
                    "suspended task reentry segment retained a claim"
                )
        else:
            segment_claim = active_segment.claim
            if (
                partial_states
                != (
                    TaskAttemptState.RUNNING,
                    TaskAttemptSegmentState.CREATED,
                )
                or segment_claim is None
                or segment_claim.claim_token != claim_token
            ):
                raise TaskStoreConflictError(
                    "partial task reentry startup state did not match"
                )
            await unit.cursor.execute(
                _UPDATE_ATTEMPT_SEGMENT_SQL,
                (
                    TaskAttemptSegmentState.ABANDONED.value,
                    _json(_claim_to_payload(segment_claim)),
                    active_segment.request_id,
                    active_segment.continuation_id,
                    active_segment.checkpoint_id,
                    _json(metadata),
                    observed_at,
                    active_segment.segment_id,
                    TaskAttemptSegmentState.CREATED.value,
                ),
            )
            segment_row = await unit.cursor.fetchone()
            if segment_row is None:
                raise TaskStoreConflictError(
                    "partial task segment cancellation compare-and-swap failed"
                )
            await unit.cursor.execute(
                _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
                (
                    self._new_id(),
                    active_segment.segment_id,
                    attempt.attempt_id,
                    task_run_id,
                    TaskAttemptSegmentState.CREATED.value,
                    TaskAttemptSegmentState.ABANDONED.value,
                    "cancelled",
                    _json(metadata),
                    observed_at,
                ),
            )
            if await unit.cursor.fetchone() is None:
                raise TaskStoreConflictError(
                    "partial task segment cancellation transition was not "
                    "recorded"
                )
        result_payload = _json(_result_to_payload(cancellation_result))
        await unit.cursor.execute(
            _UPDATE_ATTEMPT_STATE_SQL,
            (
                TaskAttemptState.ABANDONED.value,
                result_payload,
                observed_at,
                attempt.attempt_id,
                attempt.state.value,
                task_run_id,
                TaskRunState.CANCEL_REQUESTED.value,
                claim_token,
                claim_token,
            ),
        )
        attempt_row = await unit.cursor.fetchone()
        if attempt_row is None:
            raise TaskStoreConflictError(
                "partial task attempt cancellation compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                self._new_id(),
                attempt.attempt_id,
                task_run_id,
                attempt.state.value,
                TaskAttemptState.ABANDONED.value,
                "cancelled",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "partial task attempt cancellation transition was not recorded"
            )
        await unit.cursor.execute(
            _UPDATE_RUN_STATE_SQL,
            (
                TaskRunState.CANCELLED.value,
                result_payload,
                True,
                observed_at,
                task_run_id,
                TaskRunState.CANCEL_REQUESTED.value,
                claim_token,
                claim_token,
            ),
        )
        run_row = await unit.cursor.fetchone()
        if run_row is None:
            raise TaskStoreConflictError(
                "partial task run cancellation compare-and-swap failed"
            )
        await unit.cursor.execute(
            _INSERT_RUN_TRANSITION_SQL,
            (
                self._new_id(),
                task_run_id,
                TaskRunState.CANCEL_REQUESTED.value,
                TaskRunState.CANCELLED.value,
                "cancelled",
                _json(metadata),
                observed_at,
            ),
        )
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "partial task run cancellation transition was not recorded"
            )
        await unit.cursor.execute(
            _SETTLE_QUEUE_ITEM_SQL,
            (
                TaskQueueItemState.DEAD.value,
                observed_at,
                queue_item_id,
                claim_token,
            ),
        )
        queue_result = await unit.cursor.fetchone()
        if queue_result is None:
            raise TaskStoreConflictError(
                "partial task queue cancellation compare-and-swap failed"
            )
        cancelled_run = _run_from_row(run_row)
        return TaskQueueCompletion(
            queue_item=_queue_item_from_row(
                queue_result,
                run_state=cancelled_run.state,
            ),
            run=cancelled_run,
            attempt=_attempt_from_row(attempt_row),
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
        segment_id: str | None = None,
        usage_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord:
        _assert_non_empty_string(run_id, "run_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        if segment_id is not None:
            _assert_non_empty_string(segment_id, "segment_id")
            if attempt_id is None:
                raise AssertionError("segment usage requires an attempt")
        if usage_id is not None:
            _assert_non_empty_string(usage_id, "usage_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _lock_run_or_raise(unit, run_id)
            if attempt_id is not None:
                attempt = await _attempt_or_raise(unit, attempt_id)
                if attempt.run_id != run_id:
                    raise TaskStoreNotFoundError(
                        "task attempt was not found for run"
                    )
            if segment_id is not None:
                segment = await _segment_or_raise(unit, segment_id)
                if (
                    segment.run_id != run_id
                    or segment.attempt_id != attempt_id
                ):
                    raise TaskStoreNotFoundError(
                        "task attempt segment was not found for run"
                    )
            record_id = usage_id or self._new_id()
            usage_metadata = freeze_usage_metadata(metadata)
            await unit.cursor.execute(
                _SELECT_NEXT_USAGE_SEQUENCE_SQL, (run_id,)
            )
            sequence_row = await unit.cursor.fetchone()
            sequence = cast(int, _row_value(sequence_row, "sequence", 1))
            now = self._now()
            await unit.cursor.execute(
                _INSERT_USAGE_SQL,
                (
                    record_id,
                    run_id,
                    attempt_id,
                    segment_id,
                    sequence,
                    source.value,
                    totals.input_tokens,
                    totals.output_tokens,
                    totals.total_tokens,
                    totals.cached_input_tokens,
                    totals.cache_creation_input_tokens,
                    totals.reasoning_tokens,
                    _json(usage_metadata),
                    now,
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                if usage_id is not None:
                    await unit.cursor.execute(
                        _SELECT_USAGE_BY_ID_SQL,
                        (usage_id,),
                    )
                    row = await unit.cursor.fetchone()
                    if row is not None:
                        record = _usage_from_row(row)
                        if (
                            record.run_id == run_id
                            and record.attempt_id == attempt_id
                            and record.segment_id == segment_id
                        ):
                            return record
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
        source: UsageSource | None = None,
    ) -> tuple[UsageRecord, ...]:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        if source is not None:
            assert isinstance(source, UsageSource)

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
                (
                    run_id,
                    attempt_id,
                    attempt_id,
                    source.value if source is not None else None,
                    source.value if source is not None else None,
                ),
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

    async def usage_totals(
        self,
        run_id: str,
        *,
        source: UsageSource | None = None,
    ) -> UsageTotals:
        records = await self.list_usage(run_id, source=source)
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

    async def list_retention_artifacts(
        self,
        *,
        expired_at: datetime,
        purpose: TaskArtifactPurpose | None = None,
        limit: int = 100,
    ) -> tuple[TaskArtifactRecord, ...]:
        assert isinstance(expired_at, datetime)
        if purpose is not None:
            assert isinstance(purpose, TaskArtifactPurpose)
        _assert_positive_limit(limit)

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _SELECT_RETENTION_ARTIFACTS_SQL,
                (
                    TaskArtifactState.READY.value,
                    purpose.value if purpose else None,
                    purpose.value if purpose else None,
                    expired_at,
                    expired_at,
                    limit,
                ),
            )
            return tuple(
                _artifact_from_row(row) for row in await unit.cursor.fetchall()
            )

        return cast(
            tuple[TaskArtifactRecord, ...],
            await self._transaction(
                operation="task_artifact_retention_list",
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
    statements: list[str] = []
    for module_name in _TASK_PGSQL_REVISION_MODULES:
        revision = cast(Any, import_module(module_name))
        revision_statements = revision.TASK_SCHEMA_STATEMENTS
        assert isinstance(revision_statements, tuple)
        for statement in revision_statements:
            _assert_non_empty_string(statement, "statement")
            statements.append(statement)
    return tuple(statements)


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
    config.set_main_option(
        "sqlalchemy.url",
        _task_pgsql_sqlalchemy_url(settings.url),
    )
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
    with _TASK_PGSQL_ALEMBIC_LOCK:
        command(config, *args, **kwargs)


def _assert_revision(value: str) -> None:
    _assert_non_empty_string(value, "revision")
    assert fullmatch(r"[A-Za-z0-9_.@+-]+", value)


def _task_pgsql_sqlalchemy_url(url: str) -> str:
    _assert_non_empty_string(url, "url")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


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
      (%s::text IS NULL AND "claim" IS NULL)
      OR ("claim"->>'claim_token') = %s::text
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
      (%s::text IS NULL AND "claim" IS NULL)
      OR ("claim"->>'claim_token') = %s::text
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
_SELECT_ATTEMPT_FOR_UPDATE_SQL = """
SELECT * FROM "task_attempts"
WHERE "attempt_id" = %s
FOR UPDATE
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
      (%s::text IS NULL AND r."claim" IS NULL)
      OR (r."claim"->>'claim_token') = %s::text
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
_INSERT_ATTEMPT_SEGMENT_SQL = """
INSERT INTO "task_attempt_segments" (
    "segment_id", "attempt_id", "run_id", "segment_number", "state",
    "claim", "resumed_from_segment_id", "metadata", "created_at", "updated_at"
) VALUES (
    %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s
)
ON CONFLICT ("segment_id") DO NOTHING
RETURNING *
"""
_SELECT_ATTEMPT_SEGMENT_SQL = """
SELECT * FROM "task_attempt_segments"
WHERE "segment_id" = %s
"""
_SELECT_ATTEMPT_SEGMENT_FOR_UPDATE_SQL = """
SELECT * FROM "task_attempt_segments"
WHERE "segment_id" = %s
FOR UPDATE
"""
_SELECT_SEGMENTS_FOR_ATTEMPT_SQL = """
SELECT * FROM "task_attempt_segments"
WHERE "attempt_id" = %s
ORDER BY "segment_number", "created_at", "segment_id"
"""
_UPDATE_ATTEMPT_SEGMENT_SQL = """
UPDATE "task_attempt_segments"
SET
    "state" = %s,
    "claim" = %s::jsonb,
    "request_id" = %s,
    "continuation_id" = %s,
    "checkpoint_id" = %s,
    "metadata" = COALESCE(%s::jsonb, "metadata"),
    "updated_at" = %s
WHERE "segment_id" = %s
  AND "state" = %s
RETURNING *
"""
_INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL = """
INSERT INTO "task_attempt_segment_transitions" (
    "transition_id", "segment_id", "attempt_id", "run_id", "from_state",
    "to_state", "reason", "metadata", "created_at"
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT ("transition_id") DO NOTHING
RETURNING *
"""
_SELECT_ATTEMPT_SEGMENT_TRANSITIONS_SQL = """
SELECT * FROM "task_attempt_segment_transitions"
WHERE "segment_id" = %s
ORDER BY "created_at", "transition_id"
"""
_SELECT_QUEUE_ITEM_FOR_UPDATE_SQL = """
SELECT * FROM "task_queue_items"
WHERE "queue_item_id" = %s
FOR UPDATE
"""
_SELECT_QUEUE_FOR_RUN_FOR_UPDATE_SQL = """
SELECT * FROM "task_queue_items"
WHERE "run_id" = %s
ORDER BY "created_at", "queue_item_id"
LIMIT 1
FOR UPDATE
"""
_SELECT_SUSPENDED_QUEUE_FOR_UPDATE_SQL = """
SELECT * FROM "task_queue_items"
WHERE "run_id" = %s
  AND "state" IN ('suspended', 'available')
ORDER BY "created_at", "queue_item_id"
LIMIT 1
FOR UPDATE
"""
_SELECT_DURABLE_CHECKPOINT_SQL = """
SELECT c."continuation_id"
FROM "interaction_continuations" c
JOIN "interaction_records" r
  ON r."request_id" = c."request_id"
WHERE c."request_id" = %s
  AND c."continuation_id" = %s
  AND c."task_run_id" = %s
  AND (%s::text IS NULL OR c."checkpoint_id" = %s::text)
  AND c."lifecycle_state" = 'pending'
  AND r."request_state" = 'pending'
FOR KEY SHARE OF c, r
"""
_SUSPEND_RUN_SQL = """
UPDATE "task_runs"
SET
    "state" = %s,
    "result" = %s::jsonb,
    "claim" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""
_SUSPEND_ATTEMPT_SQL = """
UPDATE "task_attempts"
SET
    "state" = %s,
    "result" = %s::jsonb,
    "context" = JSONB_SET("context", '{claim}', 'null'::jsonb, TRUE),
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "attempt_id" = %s
  AND "state" = %s
RETURNING *
"""
_SUSPEND_SEGMENT_SQL = """
UPDATE "task_attempt_segments"
SET
    "state" = %s,
    "claim" = NULL,
    "request_id" = %s,
    "continuation_id" = %s,
    "checkpoint_id" = %s,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "segment_id" = %s
  AND "state" = %s
RETURNING *
"""
_SUSPEND_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "attempt_id" = %s,
    "segment_id" = %s,
    "request_id" = %s,
    "continuation_id" = %s,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" = %s
RETURNING *
"""
_SELECT_ACCEPTED_RESOLUTION_SQL = """
SELECT outbox.*
FROM "interaction_resumption_outbox" outbox
JOIN "interaction_continuations" continuation
  ON continuation."continuation_id" = outbox."continuation_id"
JOIN "interaction_records" interaction
  ON interaction."request_id" = outbox."request_id"
WHERE outbox."request_id" = %s
  AND outbox."continuation_id" = %s
  AND outbox."task_run_id" = %s
  AND outbox."resolution_revision" = %s
  AND outbox."status" IN ('pending', 'claimed', 'delivered')
  AND continuation."lifecycle_state" IN (
      'ready',
      'claimed',
      'dispatching',
      'completed'
  )
  AND interaction."request_state" = 'answered'
  AND interaction."state_revision" = outbox."resolution_revision"
FOR UPDATE OF outbox, continuation, interaction
"""
_SELECT_RELEASED_CONTINUATION_SQL = """
SELECT continuation."continuation_id"
FROM "interaction_continuations" continuation
JOIN "interaction_records" interaction
  ON interaction."request_id" = continuation."request_id"
WHERE continuation."request_id" = %s
  AND continuation."continuation_id" = %s
  AND continuation."task_run_id" = %s
  AND continuation."checkpoint_id" = %s
  AND continuation."lifecycle_state" = 'ready'
  AND interaction."request_state" = 'answered'
FOR KEY SHARE OF continuation, interaction
"""
_REQUEUE_RUN_SQL = """
UPDATE "task_runs"
SET
    "state" = %s,
    "result" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND "claim" IS NULL
RETURNING *
"""
_CLEAR_SUSPENDED_ATTEMPT_RESULT_SQL = """
UPDATE "task_attempts"
SET
    "result" = NULL,
    "updated_at" = %s
WHERE "attempt_id" = %s
  AND "state" = %s
RETURNING *
"""
_REQUEUE_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "available_at" = %s,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" IS NULL
RETURNING *
"""
_TERMINALIZE_SUSPENDED_QUEUE_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" IS NULL
RETURNING *
"""
_SETTLE_QUEUE_ITEM_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
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
_RELEASE_REENTRY_ATTEMPT_SQL = """
UPDATE "task_attempts" a
SET
    "context" = JSONB_SET(a."context", '{claim}', 'null'::jsonb, TRUE),
    "metadata" = a."metadata" || %s::jsonb,
    "updated_at" = %s
FROM "task_runs" r
WHERE a."attempt_id" = %s
  AND a."run_id" = %s
  AND a."state" = 'suspended'
  AND r."run_id" = a."run_id"
  AND r."state" = 'claimed'
  AND (r."claim"->>'claim_token') = %s
RETURNING a.*
"""
_RELEASE_REENTRY_RUN_SQL = """
UPDATE "task_runs"
SET
    "state" = %s,
    "result" = NULL,
    "claim" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""
_RELEASE_REENTRY_QUEUE_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "available_at" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" = %s
RETURNING *
"""
_FAIL_REENTRY_ATTEMPT_SQL = """
UPDATE "task_attempts" a
SET
    "state" = %s,
    "result" = %s::jsonb,
    "context" = JSONB_SET(a."context", '{claim}', 'null'::jsonb, TRUE),
    "metadata" = a."metadata" || %s::jsonb,
    "updated_at" = %s
FROM "task_runs" r
WHERE a."attempt_id" = %s
  AND a."run_id" = %s
  AND a."state" = 'suspended'
  AND r."run_id" = a."run_id"
  AND r."state" = 'claimed'
  AND (r."claim"->>'claim_token') = %s
RETURNING a.*
"""
_FAIL_REENTRY_RUN_SQL = """
UPDATE "task_runs"
SET
    "state" = %s,
    "result" = %s::jsonb,
    "claim" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""
_FAIL_REENTRY_QUEUE_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" = %s
RETURNING *
"""
_RELEASE_RUNNING_SEGMENT_SQL = """
UPDATE "task_attempt_segments"
SET
    "state" = %s,
    "claim" = NULL,
    "request_id" = %s,
    "continuation_id" = %s,
    "checkpoint_id" = %s,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "segment_id" = %s
  AND "state" = %s
RETURNING *
"""
_RELEASE_RUNNING_ATTEMPT_SQL = """
UPDATE "task_attempts" a
SET
    "state" = %s,
    "result" = NULL,
    "context" = JSONB_SET(a."context", '{claim}', 'null'::jsonb, TRUE),
    "metadata" = a."metadata" || %s::jsonb,
    "updated_at" = %s
FROM "task_runs" r
WHERE a."attempt_id" = %s
  AND a."run_id" = %s
  AND a."state" = 'running'
  AND r."run_id" = a."run_id"
  AND r."state" = 'running'
  AND (r."claim"->>'claim_token') = %s
RETURNING a.*
"""
_RELEASE_RUNNING_RUN_SQL = """
UPDATE "task_runs"
SET
    "state" = %s,
    "result" = NULL,
    "claim" = NULL,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "run_id" = %s
  AND "state" = %s
  AND ("claim"->>'claim_token') = %s
RETURNING *
"""
_RELEASE_RUNNING_QUEUE_SQL = """
UPDATE "task_queue_items"
SET
    "state" = %s,
    "available_at" = %s,
    "claimed_at" = NULL,
    "lease_expires_at" = NULL,
    "worker_id" = NULL,
    "claim_token" = NULL,
    "heartbeat_at" = NULL,
    "attempt_id" = %s,
    "segment_id" = %s,
    "request_id" = %s,
    "continuation_id" = %s,
    "metadata" = "metadata" || %s::jsonb,
    "updated_at" = %s
WHERE "queue_item_id" = %s
  AND "state" = %s
  AND "claim_token" = %s
RETURNING *
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
  AND (%s::text IS NULL OR "attempt_id" = %s::text)
  AND (%s::integer IS NULL OR "sequence" > %s::integer)
ORDER BY "sequence", "created_at", "event_id"
"""
_SELECT_NEXT_USAGE_SEQUENCE_SQL = """
SELECT COALESCE(MAX("sequence"), 0) + 1 AS "sequence"
FROM "task_usage_records"
WHERE "run_id" = %s
"""
_INSERT_USAGE_SQL = """
INSERT INTO "task_usage_records" (
    "usage_id", "run_id", "attempt_id", "segment_id", "sequence", "source",
    "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens",
    "cache_creation_input_tokens", "reasoning_tokens", "metadata",
    "created_at"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s
)
ON CONFLICT ("usage_id") DO NOTHING
RETURNING *
"""
_SELECT_USAGE_SQL = """
SELECT * FROM "task_usage_records"
WHERE "run_id" = %s
  AND (%s::text IS NULL OR "attempt_id" = %s::text)
  AND (%s::text IS NULL OR "source" = %s::text)
ORDER BY "sequence", "created_at", "usage_id"
"""
_SELECT_USAGE_BY_ID_SQL = """
SELECT * FROM "task_usage_records" WHERE "usage_id" = %s
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
  AND (%s::text IS NULL OR "attempt_id" = %s::text)
  AND (%s::text IS NULL OR "purpose" = %s::text)
  AND (%s::text IS NULL OR "state" = %s::text)
ORDER BY "created_at", "artifact_id"
"""
_SELECT_RETENTION_ARTIFACTS_SQL = """
SELECT * FROM "task_artifacts"
WHERE "state" = %s
  AND (%s::text IS NULL OR "purpose" = %s::text)
  AND (
      (
          "retention" ->> 'expires_at' IS NOT NULL
          AND ("retention" ->> 'expires_at')::timestamptz <= %s
      )
      OR (
          "retention" ->> 'delete_after_days' IS NOT NULL
          AND (
              "created_at"
              + (
                  ("retention" ->> 'delete_after_days')::integer
                  * INTERVAL '1 day'
              )
          ) <= %s
      )
  )
ORDER BY "updated_at", "created_at", "artifact_id"
LIMIT %s
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


async def _lock_attempt_or_raise(
    unit: PgsqlUnitOfWork,
    attempt_id: str,
) -> TaskAttempt:
    await unit.cursor.execute(
        _SELECT_ATTEMPT_FOR_UPDATE_SQL,
        (attempt_id,),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskStoreNotFoundError("task attempt was not found")
    return _attempt_from_row(row)


async def _segment_or_raise(
    unit: PgsqlUnitOfWork,
    segment_id: str,
) -> TaskAttemptSegment:
    await unit.cursor.execute(
        _SELECT_ATTEMPT_SEGMENT_SQL,
        (segment_id,),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskStoreNotFoundError("task attempt segment was not found")
    return _segment_from_row(row)


async def _lock_segment_or_raise(
    unit: PgsqlUnitOfWork,
    segment_id: str,
) -> TaskAttemptSegment:
    await unit.cursor.execute(
        _SELECT_ATTEMPT_SEGMENT_FOR_UPDATE_SQL,
        (segment_id,),
    )
    row = await unit.cursor.fetchone()
    if row is None:
        raise TaskStoreNotFoundError("task attempt segment was not found")
    return _segment_from_row(row)


async def _attempt_rows_for_run(
    unit: PgsqlUnitOfWork,
    run_id: str,
) -> tuple[Mapping[str, object], ...]:
    await unit.cursor.execute(_SELECT_ATTEMPTS_FOR_RUN_SQL, (run_id,))
    return tuple(await unit.cursor.fetchall())


async def _insert_suspension_transitions(
    unit: PgsqlUnitOfWork,
    *,
    id_factory: Callable[[], str],
    run: TaskRun,
    attempt: TaskAttempt,
    segment: TaskAttemptSegment,
    now: datetime,
    metadata: Mapping[str, object],
) -> None:
    statements = (
        (
            _INSERT_RUN_TRANSITION_SQL,
            (
                id_factory(),
                run.run_id,
                run.state.value,
                TaskRunState.INPUT_REQUIRED.value,
                "interaction input required",
                _json(metadata),
                now,
            ),
        ),
        (
            _INSERT_ATTEMPT_TRANSITION_SQL,
            (
                id_factory(),
                attempt.attempt_id,
                attempt.run_id,
                attempt.state.value,
                TaskAttemptState.SUSPENDED.value,
                "interaction input required",
                _json(metadata),
                now,
            ),
        ),
        (
            _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
            (
                id_factory(),
                segment.segment_id,
                segment.attempt_id,
                segment.run_id,
                segment.state.value,
                TaskAttemptSegmentState.SUSPENDED.value,
                "interaction input required",
                _json(metadata),
                now,
            ),
        ),
    )
    for statement, parameters in statements:
        await unit.cursor.execute(statement, parameters)
        if await unit.cursor.fetchone() is None:
            raise TaskStoreConflictError(
                "task suspension transition was not recorded"
            )


async def _insert_interaction_event(
    unit: PgsqlUnitOfWork,
    *,
    id_factory: Callable[[], str],
    run_id: str,
    attempt_id: str,
    event_type: TaskInteractionEventType,
    request_id: str,
    continuation_id: str,
    segment_id: str,
    now: datetime,
) -> None:
    await unit.cursor.execute(_SELECT_NEXT_EVENT_SEQUENCE_SQL, (run_id,))
    sequence_row = await unit.cursor.fetchone()
    sequence = cast(int, _row_value(sequence_row, "sequence", 1))
    await unit.cursor.execute(
        _INSERT_EVENT_SQL,
        (
            id_factory(),
            run_id,
            attempt_id,
            sequence,
            event_type.value,
            _json(
                task_interaction_event_payload(
                    event_type=event_type,
                    request_id=request_id,
                    continuation_id=continuation_id,
                    segment_id=segment_id,
                )
            ),
            _json({"category": TaskEventCategory.INTERACTION.value}),
            now,
            now,
        ),
    )
    if await unit.cursor.fetchone() is None:
        raise TaskStoreConflictError("task interaction event was not recorded")


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
    reservation = _idempotency_from_row(row)
    if reservation.identity != identity:
        raise TaskStoreConflictError(
            "idempotency key is reserved for a different identity"
        )
    return reservation


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


def _segment_from_row(row: Mapping[str, object]) -> TaskAttemptSegment:
    return TaskAttemptSegment(
        segment_id=cast(str, row["segment_id"]),
        attempt_id=cast(str, row["attempt_id"]),
        run_id=cast(str, row["run_id"]),
        segment_number=cast(int, row["segment_number"]),
        state=TaskAttemptSegmentState(cast(str, row["state"])),
        claim=(
            _claim_from_payload(_mapping(row["claim"]))
            if row.get("claim") is not None
            else None
        ),
        resumed_from_segment_id=cast(
            str | None,
            row.get("resumed_from_segment_id"),
        ),
        request_id=cast(str | None, row.get("request_id")),
        continuation_id=cast(
            str | None,
            row.get("continuation_id"),
        ),
        checkpoint_id=cast(str | None, row.get("checkpoint_id")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _segment_transition_from_row(
    row: Mapping[str, object],
) -> TaskAttemptSegmentTransition:
    return TaskAttemptSegmentTransition(
        transition_id=cast(str, row["transition_id"]),
        segment_id=cast(str, row["segment_id"]),
        attempt_id=cast(str, row["attempt_id"]),
        run_id=cast(str, row["run_id"]),
        from_state=TaskAttemptSegmentState(cast(str, row["from_state"])),
        to_state=TaskAttemptSegmentState(cast(str, row["to_state"])),
        reason=cast(str, row["reason"]),
        created_at=_datetime(row["created_at"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
    )


def _validate_reentry_provenance(
    *,
    queue_row: Mapping[str, object],
    attempt: TaskAttempt,
    segment: TaskAttemptSegment,
    task_run_id: str,
    request_id: str,
    continuation_id: str,
    checkpoint_id: str,
) -> None:
    if (
        attempt.run_id != task_run_id
        or segment.run_id != task_run_id
        or segment.attempt_id != attempt.attempt_id
        or segment.state is not TaskAttemptSegmentState.SUSPENDED
        or segment.request_id != request_id
        or segment.continuation_id != continuation_id
        or segment.checkpoint_id != checkpoint_id
        or queue_row.get("request_id") != request_id
        or queue_row.get("continuation_id") != continuation_id
    ):
        raise TaskStoreConflictError(
            "claimed task reentry provenance did not match"
        )


def _validate_partial_reentry_cancellation_provenance(
    *,
    queue_row: Mapping[str, object],
    attempt: TaskAttempt,
    previous_segment: TaskAttemptSegment,
    active_segment: TaskAttemptSegment,
    task_run_id: str,
    request_id: str,
    continuation_id: str,
    checkpoint_id: str,
) -> None:
    if (
        queue_row.get("attempt_id") != attempt.attempt_id
        or queue_row.get("segment_id") != previous_segment.segment_id
        or queue_row.get("request_id") != request_id
        or queue_row.get("continuation_id") != continuation_id
        or attempt.run_id != task_run_id
        or previous_segment.run_id != task_run_id
        or previous_segment.attempt_id != attempt.attempt_id
        or previous_segment.request_id != request_id
        or previous_segment.continuation_id != continuation_id
        or previous_segment.checkpoint_id != checkpoint_id
        or active_segment.run_id != task_run_id
        or active_segment.attempt_id != attempt.attempt_id
    ):
        raise TaskStoreConflictError(
            "partial task reentry cancellation provenance did not match"
        )
    if active_segment.segment_id == previous_segment.segment_id:
        return
    if (
        previous_segment.state is not TaskAttemptSegmentState.SUSPENDED
        or previous_segment.claim is not None
        or active_segment.resumed_from_segment_id
        != previous_segment.segment_id
        or active_segment.request_id is not None
        or active_segment.continuation_id is not None
        or active_segment.checkpoint_id is not None
        or active_segment.segment_number != previous_segment.segment_number + 1
    ):
        raise TaskStoreConflictError(
            "partial task reentry cancellation provenance did not match"
        )


def _replayed_partial_reentry_cancellation(
    *,
    queue_row: Mapping[str, object],
    run: TaskRun,
    attempt: TaskAttempt,
    previous_segment: TaskAttemptSegment,
    active_segment: TaskAttemptSegment,
    result: TaskExecutionResult,
    claim_token: str,
) -> TaskQueueCompletion:
    attempt_claim = attempt.context.claim
    if (
        queue_row.get("state") != TaskQueueItemState.DEAD.value
        or queue_row.get("claimed_at") is not None
        or queue_row.get("lease_expires_at") is not None
        or queue_row.get("worker_id") is not None
        or queue_row.get("claim_token") is not None
        or queue_row.get("heartbeat_at") is not None
        or run.claim is not None
        or run.result != result
        or attempt.state is not TaskAttemptState.ABANDONED
        or attempt.result != result
        or attempt_claim is None
        or attempt_claim.claim_token != claim_token
    ):
        raise TaskStoreConflictError(
            "terminal partial task cancellation does not match the replay"
        )
    if active_segment.segment_id == previous_segment.segment_id:
        if (
            active_segment.state is not TaskAttemptSegmentState.SUSPENDED
            or active_segment.claim is not None
        ):
            raise TaskStoreConflictError(
                "terminal partial task cancellation does not match the replay"
            )
    else:
        segment_claim = active_segment.claim
        if (
            active_segment.state is not TaskAttemptSegmentState.ABANDONED
            or segment_claim is None
            or segment_claim.claim_token != claim_token
        ):
            raise TaskStoreConflictError(
                "terminal partial task cancellation does not match the replay"
            )
    return TaskQueueCompletion(
        queue_item=_queue_item_from_row(
            queue_row,
            run_state=run.state,
        ),
        run=run,
        attempt=attempt,
    )


def _validate_released_running_reentry(
    *,
    queue_row: Mapping[str, object],
    attempt: TaskAttempt,
    segment: TaskAttemptSegment,
    request_id: str,
    continuation_id: str,
    checkpoint_id: str,
) -> None:
    if (
        attempt.state is not TaskAttemptState.SUSPENDED
        or attempt.context.claim is not None
        or segment.state is not TaskAttemptSegmentState.SUSPENDED
        or segment.claim is not None
        or segment.request_id != request_id
        or segment.continuation_id != continuation_id
        or segment.checkpoint_id != checkpoint_id
        or queue_row.get("request_id") != request_id
        or queue_row.get("continuation_id") != continuation_id
        or queue_row.get("segment_id") != segment.segment_id
        or queue_row.get("attempt_id") != attempt.attempt_id
    ):
        raise TaskStoreConflictError(
            "released running task reentry does not match the replay"
        )


def _replayed_failed_reentry(
    *,
    queue_row: Mapping[str, object],
    run: TaskRun,
    attempt: TaskAttempt,
    result: TaskExecutionResult,
) -> TaskQueueCompletion:
    if (
        queue_row["state"] != TaskQueueItemState.DEAD.value
        or queue_row.get("claim_token") is not None
        or queue_row.get("lease_expires_at") is not None
        or run.claim is not None
        or run.result != result
        or attempt.state is not TaskAttemptState.FAILED
        or attempt.result != result
    ):
        raise TaskStoreConflictError(
            "terminal task reentry failure does not match the replay"
        )
    return TaskQueueCompletion(
        queue_item=_queue_item_from_row(
            queue_row,
            run_state=run.state,
        ),
        run=run,
        attempt=attempt,
    )


def _replayed_task_settlement(
    *,
    queue_row: Mapping[str, object],
    queue_state: TaskQueueItemState,
    run: TaskRun,
    attempt: TaskAttempt,
    attempt_state: TaskAttemptState,
    segment: TaskAttemptSegment,
    segment_state: TaskAttemptSegmentState,
    result: TaskExecutionResult,
    claim_token: str,
) -> TaskQueueCompletion:
    attempt_claim = attempt.context.claim
    segment_claim = segment.claim
    if (
        queue_row["state"] != queue_state.value
        or queue_row.get("claim_token") is not None
        or queue_row.get("lease_expires_at") is not None
        or run.claim is not None
        or run.result != result
        or attempt.state is not attempt_state
        or attempt.result != result
        or segment.state is not segment_state
        or attempt_claim is None
        or attempt_claim.claim_token != claim_token
        or segment_claim is None
        or segment_claim.claim_token != claim_token
    ):
        raise TaskStoreConflictError(
            "terminal task settlement does not match the replay"
        )
    return TaskQueueCompletion(
        queue_item=_queue_item_from_row(
            queue_row,
            run_state=run.state,
        ),
        run=run,
        attempt=attempt,
    )


def _cancelled_task_settlement_result(
    settlement: TaskDurableResumeSettlement,
) -> TaskExecutionResult:
    return TaskExecutionResult(
        error=freeze_snapshot_value(TaskError.cancellation().as_dict()),
        metadata={
            "superseded_settlement_digest": (
                task_durable_resume_settlement_digest(settlement)
            )
        },
    )


def _replayed_suspended_task_lifecycle(
    *,
    queue_row: Mapping[str, object],
    run: TaskRun,
    attempt: TaskAttempt,
    attempt_state: TaskAttemptState,
    segment: TaskAttemptSegment,
    result: TaskExecutionResult,
    correlations: tuple[tuple[str, str], ...],
) -> TaskQueueCompletion:
    if (
        queue_row["state"] != TaskQueueItemState.DEAD.value
        or queue_row.get("claim_token") is not None
        or queue_row.get("lease_expires_at") is not None
        or run.claim is not None
        or run.result != result
        or attempt.state is not attempt_state
        or attempt.result != result
        or segment.state is not TaskAttemptSegmentState.SUSPENDED
        or segment.claim is not None
        or queue_row.get("attempt_id") != attempt.attempt_id
        or queue_row.get("segment_id") != segment.segment_id
        or queue_row.get("request_id") != segment.request_id
        or queue_row.get("continuation_id") != segment.continuation_id
        or (
            correlations
            and (
                cast(str, queue_row.get("request_id")),
                cast(str, queue_row.get("continuation_id")),
            )
            not in correlations
        )
    ):
        raise TaskStoreConflictError(
            "terminal suspended task does not match the replay"
        )
    return TaskQueueCompletion(
        queue_item=_queue_item_from_row(
            queue_row,
            run_state=run.state,
        ),
        run=run,
        attempt=attempt,
    )


def _queue_item_from_row(
    row: Mapping[str, object],
    *,
    run_state: TaskRunState,
) -> TaskQueueItem:
    return TaskQueueItem(
        queue_item_id=cast(str, row["queue_item_id"]),
        run_id=cast(str, row["run_id"]),
        queue_name=cast(str, row["queue_name"]),
        state=TaskQueueItemState(cast(str, row["state"])),
        priority=cast(int, row["priority"]),
        available_at=_datetime(row["available_at"]),
        claimed_at=(
            _datetime(row["claimed_at"])
            if row.get("claimed_at") is not None
            else None
        ),
        lease_expires_at=(
            _datetime(row["lease_expires_at"])
            if row.get("lease_expires_at") is not None
            else None
        ),
        worker_id=cast(str | None, row.get("worker_id")),
        claim_token=cast(str | None, row.get("claim_token")),
        heartbeat_at=(
            _datetime(row["heartbeat_at"])
            if row.get("heartbeat_at") is not None
            else None
        ),
        attempts=cast(int, row["attempts"]),
        metadata=freeze_snapshot_metadata(_mapping(row["metadata"])),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        run_state=run_state,
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
        segment_id=cast(str | None, row.get("segment_id")),
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
    payload = {
        "artifact": {
            "encrypt": definition.artifact.encrypt,
            "max_bytes": definition.artifact.max_bytes,
            "max_count": definition.artifact.max_count,
            "retention_days": definition.artifact.retention_days,
            "storage": definition.artifact.storage,
            "store_bytes": definition.artifact.store_bytes,
        },
        "container": definition.container.to_dict(),
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
    if definition.skills_config is not None:
        payload["skills_config"] = untrusted_skill_settings_config_dict(
            definition.skills_config
        )
    if definition.skills_identity is not None:
        payload["skills_identity"] = _plain(definition.skills_identity)
    return payload


def _definition_from_payload(payload: Mapping[str, object]) -> TaskDefinition:
    task = _mapping(payload["task"])
    input_payload = _mapping(payload["input"])
    output_payload = _mapping(payload["output"])
    execution = _mapping(payload["execution"])
    run = _mapping(payload["run"])
    retry = _mapping(payload["retry"])
    privacy = _mapping(payload["privacy"])
    artifact = _mapping(payload["artifact"])
    container = _mapping(payload.get("container", {}))
    limits = _mapping(payload["limits"])
    observability = _mapping(payload["observability"])
    skills_config = (
        _skills_config_from_payload(_mapping(payload["skills_config"]))
        if isinstance(payload.get("skills_config"), Mapping)
        else None
    )
    skills_identity = (
        _mapping(payload["skills_identity"])
        if isinstance(payload.get("skills_identity"), Mapping)
        else None
    )
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
        container=(
            TaskContainerExecutionSettings()
            if not container
            else TaskContainerExecutionSettings.from_dict(container)
        ),
        skills_config=skills_config,
        skills_identity=skills_identity,
    )


def _skills_config_from_payload(
    payload: Mapping[str, object],
) -> UntrustedSkillSettings:
    return parse_untrusted_skill_settings_config(
        payload,
        trusted=TrustedSkillSettings(),
        surface=SkillSettingsSurface.TASK,
        section="skills",
    )


def _request_to_payload(request: TaskExecutionRequest) -> dict[str, object]:
    payload = {
        "definition_id": request.definition_id,
        "file_summaries": _plain(request.file_summaries),
        "idempotency_key": request.idempotency_key,
        "input_summary": _plain(request.input_summary),
        "metadata": _plain(request.metadata),
        "queue": request.queue,
    }
    if request.input_payload is not None:
        payload["input_payload"] = _execution_payload_to_payload(
            request.input_payload
        )
    return payload


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
        input_payload=(
            _execution_payload_from_payload(_mapping(payload["input_payload"]))
            if payload.get("input_payload") is not None
            else None
        ),
        file_summaries=file_summaries,
        idempotency_key=cast(str | None, payload.get("idempotency_key")),
        queue=cast(str | None, payload.get("queue")),
        metadata=freeze_snapshot_metadata(
            _mapping(payload.get("metadata", {}))
        ),
    )


def _execution_payload_to_payload(
    payload: TaskExecutionPayload,
) -> dict[str, object]:
    return {
        "file_values": _plain(payload.file_values),
        "input_value": _plain(payload.input_value),
    }


def _execution_payload_from_payload(
    payload: Mapping[str, object],
) -> TaskExecutionPayload:
    file_values = payload.get("file_values", ())
    assert isinstance(file_values, list | tuple)
    return TaskExecutionPayload(
        file_values=tuple(
            freeze_snapshot_value(value) for value in file_values
        ),
        input_value=freeze_snapshot_value(payload.get("input_value")),
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


def _input_required_execution_result(
    *,
    request_id: str,
    continuation_id: str,
    checkpoint_id: str,
) -> TaskExecutionResult:
    return TaskExecutionResult(
        metadata={
            "interaction": {
                "kind": "input_required",
                "request_id": request_id,
                "continuation_id": continuation_id,
                "checkpoint_id": checkpoint_id,
                "detached_resumption_available": True,
            }
        }
    )


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


def _assert_positive_limit(value: int) -> None:
    assert isinstance(value, int), "limit must be an integer"
    assert not isinstance(value, bool), "limit must be an integer"
    assert value > 0, "limit must be positive"
