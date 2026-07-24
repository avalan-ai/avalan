"""Provide a transactional PostgreSQL test double for interaction stores."""

from asyncio import Lock
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from json import loads
from typing import Any, cast

from avalan.interaction.stores.pgsql import (
    _CHECK_SCHEMA_SQL,
    _CLAIM_OUTBOX_SQL,
    _COMPLETE_OUTBOX_SQL,
    _DEAD_OUTBOX_SQL,
    _DELETE_ORPHANED_BRANCHES_SQL,
    _DELETE_RECORD_SQL,
    _INSERT_CONTINUATION_SQL,
    _INSERT_OUTBOX_SQL,
    _INSERT_RESOLUTION_KEY_SQL,
    _LOCK_RETENTION_SCOPE_SQL,
    _LOCK_STORE_METADATA_SQL,
    _RELEASE_OUTBOX_SQL,
    _SELECT_ACTIVE_CONTINUATIONS_BY_TASK_SQL,
    _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
    _SELECT_BRANCHES_SQL,
    _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
    _SELECT_CONTINUATION_BY_REQUEST_SQL,
    _SELECT_CONTINUATION_FOR_UPDATE_SQL,
    _SELECT_CONTINUATION_SQL,
    _SELECT_EXPIRED_TASK_REENTRY_SQL,
    _SELECT_PENDING_RECORD_COUNT_SQL,
    _SELECT_RECORD_DEADLINE_FOR_UPDATE_SQL,
    _SELECT_RECORDS_SQL,
    _SELECT_RESUMED_TASK_CONTINUATION_FOR_UPDATE_SQL,
    _SELECT_SCOPE_BRANCHES_SQL,
    _SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL,
    _SELECT_SCOPE_RECORDS_SQL,
    _SELECT_SCOPED_RECORD_SQL,
    _SELECT_STORE_METADATA_SQL,
    _SELECT_SWEEP_SQL,
    _SELECT_TASK_BRANCH_CLOSURE_SQL,
    _SELECT_TASK_CONTINUATION_FOR_UPDATE_SQL,
    _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL,
    _SELECT_TASK_RUN_CONTINUATION_FOR_UPDATE_SQL,
    _SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL,
    _SET_REPEATABLE_READ_ONLY_SQL,
    _UPDATE_CONTINUATION_SQL,
    _UPDATE_STORE_METADATA_SQL,
    _UPSERT_BRANCH_SQL,
    _UPSERT_RECORD_SQL,
    INTERACTION_PGSQL_HEAD_REVISION,
)
from avalan.task import (
    EncryptedPrivacyValue,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskKeyPurpose,
    TaskQueueItemState,
    TaskRunState,
)
from avalan.task.stores.pgsql import (
    _CLEAR_SUSPENDED_ATTEMPT_RESULT_SQL,
    _FAIL_REENTRY_ATTEMPT_SQL,
    _FAIL_REENTRY_QUEUE_SQL,
    _FAIL_REENTRY_RUN_SQL,
    _INSERT_ATTEMPT_SEGMENT_SQL,
    _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL,
    _INSERT_ATTEMPT_TRANSITION_SQL,
    _INSERT_EVENT_SQL,
    _INSERT_RUN_TRANSITION_SQL,
    _RELEASE_REENTRY_ATTEMPT_SQL,
    _RELEASE_REENTRY_QUEUE_SQL,
    _RELEASE_REENTRY_RUN_SQL,
    _RELEASE_RUNNING_ATTEMPT_SQL,
    _RELEASE_RUNNING_QUEUE_SQL,
    _RELEASE_RUNNING_RUN_SQL,
    _RELEASE_RUNNING_SEGMENT_SQL,
    _REQUEUE_QUEUE_ITEM_SQL,
    _REQUEUE_RUN_SQL,
    _SELECT_ACCEPTED_RESOLUTION_SQL,
    _SELECT_ATTEMPT_FOR_UPDATE_SQL,
    _SELECT_ATTEMPT_SEGMENT_FOR_UPDATE_SQL,
    _SELECT_ATTEMPT_SQL,
    _SELECT_DURABLE_CHECKPOINT_SQL,
    _SELECT_NEXT_EVENT_SEQUENCE_SQL,
    _SELECT_QUEUE_FOR_RUN_FOR_UPDATE_SQL,
    _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL,
    _SELECT_RELEASED_CONTINUATION_SQL,
    _SELECT_RUN_FOR_UPDATE_SQL,
    _SELECT_RUN_SQL,
    _SELECT_SEGMENTS_FOR_ATTEMPT_SQL,
    _SELECT_SUSPENDED_QUEUE_FOR_UPDATE_SQL,
    _SETTLE_QUEUE_ITEM_SQL,
    _SUSPEND_ATTEMPT_SQL,
    _SUSPEND_QUEUE_ITEM_SQL,
    _SUSPEND_RUN_SQL,
    _SUSPEND_SEGMENT_SQL,
    _TERMINALIZE_SUSPENDED_QUEUE_SQL,
    _UPDATE_ATTEMPT_SEGMENT_SQL,
    _UPDATE_ATTEMPT_STATE_SQL,
    _UPDATE_RUN_STATE_SQL,
)


class FakeInteractionCipher:
    """Reversibly conceal durable payloads without retaining plaintext."""

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        assert purpose is TaskKeyPurpose.RAW_VALUE
        return EncryptedPrivacyValue(
            ciphertext=bytes(byte ^ 0xA5 for byte in value),
            key_id=key_id or "interaction-test",
            algorithm="test-aead",
            metadata=context,
        )

    def decrypt(
        self,
        value: EncryptedPrivacyValue,
        *,
        purpose: TaskKeyPurpose,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        assert purpose is TaskKeyPurpose.RAW_VALUE
        assert context is not None
        return bytes(byte ^ 0xA5 for byte in value.ciphertext)


class FakePgsqlDatabase:
    """Retain durable interaction and task rows with transaction rollback."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.metadata: dict[str, object] = {
            "store_generation": 0,
            "schedule_revision": 0,
        }
        self.records: dict[str, dict[str, object]] = {}
        self.branches: dict[tuple[str, str, str], dict[str, object]] = {}
        self.continuations: dict[str, dict[str, object]] = {}
        self.resolution_keys: dict[tuple[str, str], dict[str, object]] = {}
        self.outbox: dict[str, dict[str, object]] = {}
        self.runs: dict[str, dict[str, object]] = {}
        self.attempts: dict[str, dict[str, object]] = {}
        self.segments: dict[str, dict[str, object]] = {}
        self.queue_items: dict[str, dict[str, object]] = {}
        self.run_transitions: dict[str, dict[str, object]] = {}
        self.attempt_transitions: dict[str, dict[str, object]] = {}
        self.segment_transitions: dict[str, dict[str, object]] = {}
        self.events: dict[str, dict[str, object]] = {}
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.fail_query: str | None = None
        self.fail_after_queries: int | None = None
        self.omit_scope_ownership_presence_result = False
        self.open_count = 0
        self.close_count = 0

    def connection(self) -> "FakeConnectionContext":
        """Return one independently acquired fake connection."""
        return FakeConnectionContext(self)

    async def open(self) -> None:
        """Record a database-open request."""
        self.open_count += 1

    async def aclose(self) -> None:
        """Record a database-close request."""
        self.close_count += 1

    def snapshot(self) -> dict[str, object]:
        """Return rollback state without copying process synchronization."""
        state = {
            "metadata": self.metadata,
            "records": self.records,
            "branches": self.branches,
            "continuations": self.continuations,
            "resolution_keys": self.resolution_keys,
            "outbox": self.outbox,
            "runs": self.runs,
            "attempts": self.attempts,
            "segments": self.segments,
            "queue_items": self.queue_items,
            "run_transitions": self.run_transitions,
            "attempt_transitions": self.attempt_transitions,
            "segment_transitions": self.segment_transitions,
            "events": self.events,
        }
        return deepcopy(state)

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore every persistent fake table after rollback."""
        self.metadata = cast(dict[str, object], snapshot["metadata"])
        self.records = cast(
            dict[str, dict[str, object]],
            snapshot["records"],
        )
        self.branches = cast(
            dict[tuple[str, str, str], dict[str, object]],
            snapshot["branches"],
        )
        self.continuations = cast(
            dict[str, dict[str, object]],
            snapshot["continuations"],
        )
        self.resolution_keys = cast(
            dict[tuple[str, str], dict[str, object]],
            snapshot["resolution_keys"],
        )
        self.outbox = cast(
            dict[str, dict[str, object]],
            snapshot["outbox"],
        )
        self.runs = cast(dict[str, dict[str, object]], snapshot["runs"])
        self.attempts = cast(
            dict[str, dict[str, object]],
            snapshot["attempts"],
        )
        self.segments = cast(
            dict[str, dict[str, object]],
            snapshot["segments"],
        )
        self.queue_items = cast(
            dict[str, dict[str, object]],
            snapshot["queue_items"],
        )
        self.run_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["run_transitions"],
        )
        self.attempt_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["attempt_transitions"],
        )
        self.segment_transitions = cast(
            dict[str, dict[str, object]],
            snapshot["segment_transitions"],
        )
        self.events = cast(
            dict[str, dict[str, object]],
            snapshot["events"],
        )


class FakeConnectionContext:
    """Acquire one connection without owning the transaction lock."""

    def __init__(self, database: FakePgsqlDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeConnection":
        return FakeConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeConnection:
    """Expose cursor and rollback-capable transaction contexts."""

    row_factory: object = None

    def __init__(self, database: FakePgsqlDatabase) -> None:
        self.database = database

    def transaction(self) -> "FakeTransactionContext":
        return FakeTransactionContext(self.database)

    def cursor(self) -> "FakeCursorContext":
        return FakeCursorContext(self.database)

    async def set_autocommit(self, value: bool) -> None:
        assert isinstance(value, bool)


class FakeTransactionContext:
    """Serialize transactions and restore their entry snapshot on failure."""

    def __init__(self, database: FakePgsqlDatabase) -> None:
        self.database = database
        self.snapshot: dict[str, object] | None = None

    async def __aenter__(self) -> "FakeTransactionContext":
        await self.database.lock.acquire()
        self.snapshot = self.database.snapshot()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        try:
            if exc_type is not None:
                assert self.snapshot is not None
                self.database.restore(self.snapshot)
        finally:
            self.database.lock.release()
        return False


class FakeCursorContext:
    """Open one cursor over the currently acquired transaction."""

    def __init__(self, database: FakePgsqlDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FakeCursor":
        return FakeCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeCursor:
    """Interpret only the exact durable-store SQL used by production."""

    def __init__(self, database: FakePgsqlDatabase) -> None:
        self.database = database
        self.row: dict[str, object] | None = None
        self.rows: tuple[dict[str, object], ...] = ()

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | Mapping[str, object] | None = None,
    ) -> None:
        params = (
            tuple(parameters)
            if parameters is not None and not isinstance(parameters, Mapping)
            else ()
        )
        self.row = None
        self.rows = ()
        self.database.executed.append((query, params))
        fail_after_query = self.database.fail_after_queries == 1
        if self.database.fail_after_queries is not None:
            self.database.fail_after_queries -= 1
        if self.database.fail_query is not None and (
            self.database.fail_query in query
        ):
            raise RuntimeError("injected PostgreSQL crash")
        if query == _CHECK_SCHEMA_SQL:
            self.row = {"version_num": INTERACTION_PGSQL_HEAD_REVISION}
        elif query == _SET_REPEATABLE_READ_ONLY_SQL:
            return
        elif query == _LOCK_STORE_METADATA_SQL:
            self.row = dict(self.database.metadata)
        elif query == _SELECT_STORE_METADATA_SQL:
            self.row = dict(self.database.metadata)
        elif query == _UPDATE_STORE_METADATA_SQL:
            self.database.metadata.update(
                store_generation=params[0],
                schedule_revision=params[1],
            )
        elif query == _SELECT_RECORDS_SQL:
            self.rows = tuple(
                deepcopy(self.database.records[key])
                for key in sorted(self.database.records)
            )
        elif query == _SELECT_PENDING_RECORD_COUNT_SQL:
            self.row = {
                "pending_count": sum(
                    row["request_state"] == "pending"
                    for row in self.database.records.values()
                )
            }
        elif query == _SELECT_SCOPED_RECORD_SQL:
            self.rows = tuple(
                deepcopy(row)
                for row in self.database.records.values()
                if (
                    row["request_id"],
                    row["continuation_id"],
                    row["run_id"],
                    row["turn_id"],
                    row["task_id"],
                    row["agent_id"],
                    row["branch_id"],
                    row["model_call_id"],
                    row["scope_identity_digest"],
                )
                == params
            )
        elif query == _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL:
            request_id = params[0]
            assert isinstance(request_id, str)
            record = self.database.records.get(request_id)
            if record is not None and record["continuation_id"] == params[1]:
                self.rows = (deepcopy(record),)
        elif query == _SELECT_SCOPE_RECORDS_SQL:
            self.rows = tuple(
                deepcopy(row)
                for row in sorted(
                    self.database.records.values(),
                    key=lambda item: cast(str, item["request_id"]),
                )
                if row["run_id"] == params[0]
                and row["scope_identity_digest"] == params[1]
            )
        elif query == _SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL:
            if not self.database.omit_scope_ownership_presence_result:
                self.row = self._scope_ownership_presence(params)
        elif query == _SELECT_RECORD_DEADLINE_FOR_UPDATE_SQL:
            record = self.database.records.get(cast(str, params[0]))
            if record is not None:
                self.row = {
                    "absolute_expires_at": record["absolute_expires_at"]
                }
        elif query == _SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL:
            self.rows = tuple(
                {
                    "run_id": record["run_id"],
                    "scope_identity_digest": record["scope_identity_digest"],
                }
                for continuation, record in sorted(
                    (
                        (continuation, record)
                        for continuation in (
                            self.database.continuations.values()
                        )
                        for record in self.database.records.values()
                        if continuation["task_run_id"] == params[0]
                        and continuation["request_id"] == record["request_id"]
                    ),
                    key=lambda item: cast(str, item[1]["request_id"]),
                )
            )
        elif query == _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL:
            self.rows = tuple(
                deepcopy(record)
                for continuation, record in sorted(
                    (
                        (continuation, record)
                        for continuation in (
                            self.database.continuations.values()
                        )
                        for record in self.database.records.values()
                        if continuation["task_run_id"] == params[0]
                        and continuation["request_id"] == record["request_id"]
                    ),
                    key=lambda item: cast(str, item[1]["request_id"]),
                )
            )
        elif query == _SELECT_EXPIRED_TASK_REENTRY_SQL:
            self.row = self._select_expired_task_reentry(params)
        elif query == _SELECT_BRANCHES_SQL:
            self.rows = tuple(
                deepcopy(self.database.branches[key])
                for key in sorted(self.database.branches)
            )
        elif query == _SELECT_SCOPE_BRANCHES_SQL:
            self.rows = tuple(
                deepcopy(row)
                for row in sorted(
                    self.database.branches.values(),
                    key=lambda item: (
                        cast(str, item["run_id"]),
                        cast(str, item["branch_id"]),
                    ),
                )
                if row["run_id"] == params[0]
                and row["scope_identity_digest"] == params[1]
            )
        elif query == _SELECT_TASK_BRANCH_CLOSURE_SQL:
            task_run_id, run_id, scope_identity_digest = params
            assert isinstance(task_run_id, str)
            assert isinstance(run_id, str)
            assert isinstance(scope_identity_digest, str)
            pending: set[tuple[str, str, str]] = set()
            for candidate_continuation in self.database.continuations.values():
                if candidate_continuation["task_run_id"] != task_run_id:
                    continue
                request_id = candidate_continuation["request_id"]
                assert isinstance(request_id, str)
                record = self.database.records.get(request_id)
                if (
                    record is None
                    or record["run_id"] != run_id
                    or record["scope_identity_digest"] != scope_identity_digest
                ):
                    continue
                branch_id = record["branch_id"]
                assert isinstance(branch_id, str)
                pending.add((run_id, branch_id, scope_identity_digest))
            selected: set[tuple[str, str, str]] = set()
            while pending:
                key = pending.pop()
                if key in selected:
                    continue
                branch = self.database.branches.get(key)
                if branch is None:
                    continue
                selected.add(key)
                parent_branch_id = branch["parent_branch_id"]
                assert isinstance(parent_branch_id, str)
                pending.add(
                    (
                        key[0],
                        parent_branch_id,
                        key[2],
                    )
                )
            self.rows = tuple(
                deepcopy(self.database.branches[key])
                for key in sorted(selected)
            )
        elif query == _UPSERT_RECORD_SQL:
            self.row = self._upsert_record(params)
        elif query == _UPSERT_BRANCH_SQL:
            self.row = self._upsert_branch(params)
        elif query == _INSERT_CONTINUATION_SQL:
            self.row = self._insert_continuation(params)
        elif query == _SELECT_CONTINUATION_SQL:
            self.row = self.database.continuations.get(cast(str, params[0]))
        elif query == _SELECT_CONTINUATION_FOR_UPDATE_SQL:
            self.row = self._continuation_with_record(
                self.database.continuations.get(cast(str, params[0]))
            )
        elif query == _SELECT_TASK_RUN_CONTINUATION_FOR_UPDATE_SQL:
            continuation = self.database.continuations.get(
                cast(str, params[0])
            )
            if (
                continuation is not None
                and continuation["task_run_id"] == params[1]
            ):
                self.row = self._continuation_with_record(continuation)
        elif query == _SELECT_TASK_CONTINUATION_FOR_UPDATE_SQL:
            continuation = self.database.continuations.get(
                cast(str, params[0])
            )
            if (
                continuation is not None
                and (
                    continuation["task_run_id"],
                    continuation["request_id"],
                    continuation["checkpoint_id"],
                )
                == params[1:]
            ):
                self.row = self._continuation_with_record(continuation)
        elif query == _SELECT_RESUMED_TASK_CONTINUATION_FOR_UPDATE_SQL:
            continuation = self.database.continuations.get(
                cast(str, params[0])
            )
            active = self.database.segments.get(cast(str, params[2]))
            previous = (
                self.database.segments.get(
                    cast(str, active["resumed_from_segment_id"])
                )
                if active is not None
                and active.get("resumed_from_segment_id") is not None
                else None
            )
            if (
                continuation is not None
                and active is not None
                and previous is not None
                and continuation["task_run_id"] == params[1]
                and active["run_id"] == params[1]
                and previous["run_id"] == params[1]
                and (
                    previous["request_id"],
                    previous["continuation_id"],
                    previous["checkpoint_id"],
                )
                == (
                    continuation["request_id"],
                    continuation["continuation_id"],
                    continuation["checkpoint_id"],
                )
            ):
                self.row = self._continuation_with_record(continuation)
        elif query == _SELECT_ACTIVE_CONTINUATIONS_BY_TASK_SQL:
            self.rows = tuple(
                deepcopy(row)
                for row in sorted(
                    self.database.continuations.values(),
                    key=lambda item: (
                        cast(datetime, item["created_at"]),
                        cast(str, item["continuation_id"]),
                    ),
                )
                if row["task_run_id"] == params[0]
                and row["lifecycle_state"]
                in {"pending", "ready", "claimed", "dispatching"}
            )[:2]
        elif query == _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL:
            self.row = next(
                (
                    row
                    for row in self.database.continuations.values()
                    if row["request_id"] == params[0]
                ),
                None,
            )
        elif query == _SELECT_CONTINUATION_BY_REQUEST_SQL:
            self.row = next(
                (
                    row
                    for row in self.database.continuations.values()
                    if row["request_id"] == params[0]
                ),
                None,
            )
        elif query == _UPDATE_CONTINUATION_SQL:
            self.row = self._update_continuation(params)
        elif query == _INSERT_RESOLUTION_KEY_SQL:
            self._insert_resolution_key(params)
        elif query == _INSERT_OUTBOX_SQL:
            self._insert_outbox(params)
        elif query == _DEAD_OUTBOX_SQL:
            self._dead_outbox(params)
        elif query == _SELECT_SWEEP_SQL:
            self._select_sweep(params)
        elif query == _LOCK_RETENTION_SCOPE_SQL:
            return
        elif query == _DELETE_RECORD_SQL:
            self._delete_record(cast(str, params[0]))
        elif query == _DELETE_ORPHANED_BRANCHES_SQL:
            self._delete_orphaned_branches(
                cast(str, params[0]),
                cast(str, params[1]),
            )
        elif query == _CLAIM_OUTBOX_SQL:
            self._claim_outbox(params)
        elif query == _COMPLETE_OUTBOX_SQL:
            self.row = self._complete_outbox(params)
        elif query == _RELEASE_OUTBOX_SQL:
            self.row = self._release_outbox(params)
        elif query == _SELECT_QUEUE_ITEM_FOR_UPDATE_SQL:
            self.row = self.database.queue_items.get(cast(str, params[0]))
        elif query == _SELECT_RUN_SQL:
            self.row = self.database.runs.get(cast(str, params[0]))
        elif query == _SELECT_ATTEMPT_SQL:
            self.row = self.database.attempts.get(cast(str, params[0]))
        elif query == _SELECT_SEGMENTS_FOR_ATTEMPT_SQL:
            self.rows = tuple(
                row
                for row in self.database.segments.values()
                if row["attempt_id"] == params[0]
            )
        elif query == _SELECT_QUEUE_FOR_RUN_FOR_UPDATE_SQL:
            self.row = next(
                (
                    row
                    for row in self.database.queue_items.values()
                    if row["run_id"] == params[0]
                ),
                None,
            )
        elif query == _SELECT_RUN_FOR_UPDATE_SQL:
            self.row = self.database.runs.get(cast(str, params[0]))
        elif query == _SELECT_ATTEMPT_FOR_UPDATE_SQL:
            self.row = self.database.attempts.get(cast(str, params[0]))
        elif query == _SELECT_ATTEMPT_SEGMENT_FOR_UPDATE_SQL:
            self.row = self.database.segments.get(cast(str, params[0]))
        elif query == _SELECT_DURABLE_CHECKPOINT_SQL:
            self.row = self._select_checkpoint(params)
        elif query == _INSERT_ATTEMPT_SEGMENT_SQL:
            self.row = self._insert_attempt_segment(params)
        elif query == _SUSPEND_RUN_SQL:
            self.row = self._suspend_run(params)
        elif query == _SUSPEND_ATTEMPT_SQL:
            self.row = self._suspend_attempt(params)
        elif query == _SUSPEND_SEGMENT_SQL:
            self.row = self._suspend_segment(params)
        elif query == _SUSPEND_QUEUE_ITEM_SQL:
            self.row = self._suspend_queue(params)
        elif query == _INSERT_RUN_TRANSITION_SQL:
            self.row = self._insert_transition(
                self.database.run_transitions,
                params,
                (
                    "transition_id",
                    "run_id",
                    "from_state",
                    "to_state",
                    "reason",
                    "metadata",
                    "created_at",
                ),
            )
        elif query == _INSERT_ATTEMPT_TRANSITION_SQL:
            self.row = self._insert_transition(
                self.database.attempt_transitions,
                params,
                (
                    "transition_id",
                    "attempt_id",
                    "run_id",
                    "from_state",
                    "to_state",
                    "reason",
                    "metadata",
                    "created_at",
                ),
            )
        elif query == _INSERT_ATTEMPT_SEGMENT_TRANSITION_SQL:
            self.row = self._insert_transition(
                self.database.segment_transitions,
                params,
                (
                    "transition_id",
                    "segment_id",
                    "attempt_id",
                    "run_id",
                    "from_state",
                    "to_state",
                    "reason",
                    "metadata",
                    "created_at",
                ),
            )
        elif query == _SELECT_SUSPENDED_QUEUE_FOR_UPDATE_SQL:
            self.row = next(
                (
                    row
                    for row in self.database.queue_items.values()
                    if row["run_id"] == params[0]
                    and row["state"] in {"suspended", "available"}
                ),
                None,
            )
        elif query == _SELECT_ACCEPTED_RESOLUTION_SQL:
            self.row = self._select_accepted_resolution(params)
        elif query == _SELECT_RELEASED_CONTINUATION_SQL:
            self.row = self._select_released_continuation(params)
        elif query == _REQUEUE_RUN_SQL:
            self.row = self._requeue_run(params)
        elif query == _CLEAR_SUSPENDED_ATTEMPT_RESULT_SQL:
            self.row = self._clear_suspended_attempt_result(params)
        elif query == _REQUEUE_QUEUE_ITEM_SQL:
            self.row = self._requeue_queue(params)
        elif query == _TERMINALIZE_SUSPENDED_QUEUE_SQL:
            self.row = self._terminalize_suspended_queue(params)
        elif query == _UPDATE_ATTEMPT_SEGMENT_SQL:
            self.row = self._settle_segment(params)
        elif query == _UPDATE_ATTEMPT_STATE_SQL:
            self.row = self._settle_attempt(params)
        elif query == _UPDATE_RUN_STATE_SQL:
            self.row = self._settle_run(params)
        elif query == _SETTLE_QUEUE_ITEM_SQL:
            self.row = self._settle_queue(params)
        elif query == _RELEASE_REENTRY_ATTEMPT_SQL:
            self.row = self._release_reentry_attempt(params)
        elif query == _RELEASE_REENTRY_RUN_SQL:
            self.row = self._release_reentry_run(params)
        elif query == _RELEASE_REENTRY_QUEUE_SQL:
            self.row = self._release_reentry_queue(params)
        elif query == _FAIL_REENTRY_ATTEMPT_SQL:
            self.row = self._fail_reentry_attempt(params)
        elif query == _FAIL_REENTRY_RUN_SQL:
            self.row = self._fail_reentry_run(params)
        elif query == _FAIL_REENTRY_QUEUE_SQL:
            self.row = self._fail_reentry_queue(params)
        elif query == _RELEASE_RUNNING_SEGMENT_SQL:
            self.row = self._release_running_segment(params)
        elif query == _RELEASE_RUNNING_ATTEMPT_SQL:
            self.row = self._release_running_attempt(params)
        elif query == _RELEASE_RUNNING_RUN_SQL:
            self.row = self._release_running_run(params)
        elif query == _RELEASE_RUNNING_QUEUE_SQL:
            self.row = self._release_running_queue(params)
        elif query == _SELECT_NEXT_EVENT_SEQUENCE_SQL:
            run_id = params[0]
            self.row = {
                "sequence": (
                    1
                    + max(
                        (
                            cast(int, row["sequence"])
                            for row in self.database.events.values()
                            if row["run_id"] == run_id
                        ),
                        default=0,
                    )
                )
            }
        elif query == _INSERT_EVENT_SQL:
            self.row = self._insert_event(params)
        else:
            raise AssertionError(f"unexpected query: {query}")
        if fail_after_query:
            self.database.fail_after_queries = None
            raise RuntimeError("injected post-query PostgreSQL crash")

    async def executemany(
        self,
        query: str,
        parameters_seq: Sequence[
            tuple[object, ...] | Mapping[str, object] | None
        ],
    ) -> None:
        for parameters in parameters_seq:
            await self.execute(query, parameters)

    async def fetchone(self) -> dict[str, object] | None:
        return deepcopy(self.row)

    async def fetchall(self) -> tuple[dict[str, object], ...]:
        return deepcopy(self.rows)

    async def close(self) -> None:
        return None

    def _scope_ownership_presence(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object]:
        assert len(params) == 20
        run_id = params[0]
        branch_id = params[2]
        include_descendants = params[5]
        turn_id = params[7]
        task_id = params[9]
        agent_id = params[11]
        actor_scope_identity = params[16]
        assert isinstance(run_id, str)
        assert params[1] == run_id
        assert branch_id is None or isinstance(branch_id, str)
        assert params[3] == branch_id
        assert params[4] == run_id
        assert isinstance(include_descendants, bool)
        assert params[6] == run_id
        assert turn_id is None or isinstance(turn_id, str)
        assert params[8] == turn_id
        assert task_id is None or isinstance(task_id, str)
        assert params[10] == task_id
        assert agent_id is None or isinstance(agent_id, str)
        assert params[12] == agent_id
        assert params[13] == branch_id
        assert params[14] == run_id
        assert params[15] == branch_id
        assert isinstance(actor_scope_identity, str)
        assert params[17] == actor_scope_identity
        assert params[18] == actor_scope_identity
        assert params[19] == actor_scope_identity

        owners: set[str] = set()
        for row in self.database.records.values():
            if row["run_id"] != run_id:
                continue
            owner = row["scope_identity_digest"]
            assert isinstance(owner, str)
            owners.add(owner)
        for row in self.database.branches.values():
            if row["run_id"] != run_id:
                continue
            owner = row["scope_identity_digest"]
            assert isinstance(owner, str)
            owners.add(owner)

        allowed: set[tuple[str, str]] = set()
        if branch_id is not None:
            allowed.update((owner, branch_id) for owner in owners)
        if include_descendants:
            changed = True
            while changed:
                changed = False
                for row in self.database.branches.values():
                    if row["run_id"] != run_id:
                        continue
                    owner = row["scope_identity_digest"]
                    child = row["branch_id"]
                    parent = row["parent_branch_id"]
                    assert isinstance(owner, str)
                    assert isinstance(child, str)
                    assert isinstance(parent, str)
                    if (owner, parent) in allowed and (
                        owner,
                        child,
                    ) not in allowed:
                        allowed.add((owner, child))
                        changed = True

        matching_record_owners: set[str] = set()
        for row in self.database.records.values():
            owner = row["scope_identity_digest"]
            record_branch_id = row["branch_id"]
            assert isinstance(owner, str)
            assert isinstance(record_branch_id, str)
            if (
                row["run_id"] != run_id
                or (turn_id is not None and row["turn_id"] != turn_id)
                or (task_id is not None and row["task_id"] != task_id)
                or (agent_id is not None and row["agent_id"] != agent_id)
                or (
                    branch_id is not None
                    and (owner, record_branch_id) not in allowed
                )
            ):
                continue
            matching_record_owners.add(owner)
        matching_branch_owners: set[str] = set()
        if branch_id is not None:
            for row in self.database.branches.values():
                owner = row["scope_identity_digest"]
                registered_branch_id = row["branch_id"]
                assert isinstance(owner, str)
                assert isinstance(registered_branch_id, str)
                if (
                    row["run_id"] == run_id
                    and (owner, registered_branch_id) in allowed
                ):
                    matching_branch_owners.add(owner)
        return {
            "actor_owned_record_match": (
                actor_scope_identity in matching_record_owners
            ),
            "foreign_owned_record_match": any(
                owner != actor_scope_identity
                for owner in matching_record_owners
            ),
            "actor_owned_branch_match": (
                actor_scope_identity in matching_branch_owners
            ),
            "foreign_owned_branch_match": any(
                owner != actor_scope_identity
                for owner in matching_branch_owners
            ),
        }

    def _upsert_record(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        keys = (
            "request_id",
            "continuation_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
            "model_call_id",
            "scope_identity_digest",
            "request_state",
            "state_revision",
            "store_revision",
            "absolute_expires_at",
            "retention_deadline_at",
            "ciphertext",
            "encryption_key_id",
            "encryption_algorithm",
            "encryption_metadata",
            "created_at",
            "updated_at",
        )
        row = dict(zip(keys, params, strict=True))
        row["encryption_metadata"] = loads(
            cast(str, row["encryption_metadata"])
        )
        request_id = cast(str, row["request_id"])
        previous = self.database.records.get(request_id)
        if previous is not None:
            if (
                previous["scope_identity_digest"]
                != row["scope_identity_digest"]
            ):
                return None
            updated = deepcopy(previous)
            for key in (
                "request_state",
                "state_revision",
                "store_revision",
                "absolute_expires_at",
                "retention_deadline_at",
                "ciphertext",
                "encryption_key_id",
                "encryption_algorithm",
                "encryption_metadata",
                "updated_at",
            ):
                updated[key] = row[key]
            row = updated
        self.database.records[request_id] = row
        return {"request_id": request_id}

    def _continuation_with_record(
        self,
        continuation: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if continuation is None:
            return None
        record = self.database.records.get(
            cast(str, continuation["request_id"])
        )
        if record is None:
            return None
        return {
            **continuation,
            "request_absolute_expires_at": record["absolute_expires_at"],
        }

    def _upsert_branch(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        keys = (
            "run_id",
            "branch_id",
            "parent_branch_id",
            "root_branch_id",
            "store_revision",
            "scope_identity_digest",
            "ciphertext",
            "encryption_key_id",
            "encryption_algorithm",
            "encryption_metadata",
        )
        row = dict(zip(keys, params, strict=True))
        row["encryption_metadata"] = loads(
            cast(str, row["encryption_metadata"])
        )
        key = (
            cast(str, row["run_id"]),
            cast(str, row["branch_id"]),
            cast(str, row["scope_identity_digest"]),
        )
        previous = self.database.branches.get(key)
        if previous is not None:
            updated = deepcopy(previous)
            for field_name in (
                "parent_branch_id",
                "root_branch_id",
                "store_revision",
                "ciphertext",
                "encryption_key_id",
                "encryption_algorithm",
                "encryption_metadata",
            ):
                updated[field_name] = row[field_name]
            row = updated
        self.database.branches[key] = row
        return {"run_id": row["run_id"]}

    def _insert_continuation(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        continuation_id = cast(str, params[0])
        request_id = cast(str, params[2])
        if (
            continuation_id in self.database.continuations
            or request_id not in self.database.records
        ):
            return None
        keys = (
            "continuation_id",
            "checkpoint_id",
            "request_id",
            "task_run_id",
            "lifecycle_state",
            "state_revision",
            "store_revision",
            "fencing_token",
            "ciphertext",
            "encryption_key_id",
            "encryption_algorithm",
            "encryption_metadata",
            "expires_at",
            "retention_deadline_at",
            "created_at",
            "updated_at",
        )
        row = dict(zip(keys, params, strict=True))
        row["encryption_metadata"] = loads(
            cast(str, row["encryption_metadata"])
        )
        row.update(
            claim_owner_id=None,
            claim_lease_expires_at=None,
            dispatch_id=None,
            dispatch_started_at=None,
            dispatch_completed_at=None,
            dispatch_ambiguous=False,
            invalid_reason=None,
        )
        self.database.continuations[continuation_id] = row
        return row

    def _update_continuation(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        continuation_id = cast(str, params[16])
        row = self.database.continuations.get(continuation_id)
        if row is None or row["store_revision"] != params[17]:
            return None
        keys = (
            "lifecycle_state",
            "state_revision",
            "store_revision",
            "claim_owner_id",
            "claim_lease_expires_at",
            "fencing_token",
            "dispatch_id",
            "dispatch_started_at",
            "dispatch_completed_at",
            "dispatch_ambiguous",
            "invalid_reason",
            "ciphertext",
            "encryption_key_id",
            "encryption_algorithm",
            "encryption_metadata",
            "updated_at",
        )
        updates = dict(zip(keys, params[:16], strict=True))
        updates["encryption_metadata"] = loads(
            cast(str, updates["encryption_metadata"])
        )
        row.update(updates)
        return row

    def _insert_resolution_key(self, params: tuple[object, ...]) -> None:
        key = (cast(str, params[0]), cast(str, params[1]))
        self.database.resolution_keys.setdefault(
            key,
            {
                "request_id": params[0],
                "idempotency_key": params[1],
                "resolution_digest": params[2],
                "state_revision": params[3],
            },
        )

    def _insert_outbox(self, params: tuple[object, ...]) -> None:
        if any(
            row["continuation_id"] == params[1]
            and row["resolution_revision"] == params[4]
            for row in self.database.outbox.values()
        ):
            return
        row = {
            "outbox_id": params[0],
            "continuation_id": params[1],
            "request_id": params[2],
            "task_run_id": params[3],
            "resolution_revision": params[4],
            "status": "pending",
            "claim_owner_id": None,
            "claim_lease_expires_at": None,
            "fencing_token": 0,
            "attempts": 0,
            "last_error_code": None,
            "available_at": params[5],
            "created_at": params[6],
            "updated_at": params[7],
            "delivered_at": None,
        }
        self.database.outbox[cast(str, row["outbox_id"])] = row

    def _dead_outbox(self, params: tuple[object, ...]) -> None:
        for row in self.database.outbox.values():
            if row["continuation_id"] == params[1] and row["status"] in {
                "pending",
                "claimed",
            }:
                row.update(
                    status="dead",
                    claim_owner_id=None,
                    claim_lease_expires_at=None,
                    updated_at=params[0],
                )

    def _select_sweep(self, params: tuple[object, ...]) -> None:
        expiry = cast(datetime, params[0])
        retention = cast(datetime, params[1])
        limit = cast(int, params[2])
        rows: list[dict[str, object]] = []
        for record in self.database.records.values():
            request_id = cast(str, record["request_id"])
            continuation = next(
                (
                    row
                    for row in self.database.continuations.values()
                    if row["request_id"] == request_id
                ),
                None,
            )
            retention_deadline = cast(
                datetime,
                record["retention_deadline_at"],
            )
            continuation_expired = (
                continuation is not None
                and cast(datetime, continuation["expires_at"]) <= expiry
            )
            if not continuation_expired and retention_deadline > retention:
                continue
            sweep_row = {
                **(deepcopy(continuation) if continuation else {}),
                "continuation_id": (
                    continuation["continuation_id"]
                    if continuation is not None
                    else None
                ),
                "task_run_id": (
                    continuation["task_run_id"]
                    if continuation is not None
                    else None
                ),
                "interaction_request_id": request_id,
                "interaction_run_id": record["run_id"],
                "interaction_scope_identity_digest": record[
                    "scope_identity_digest"
                ],
                "interaction_retention_deadline_at": retention_deadline,
            }
            rows.append(sweep_row)
        self.rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row["interaction_retention_deadline_at"],
                    row.get("continuation_id")
                    or row["interaction_request_id"],
                ),
            )[:limit]
        )

    def _select_expired_task_reentry(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        queue_item_id = cast(str, params[0])
        task_run_id = cast(str, params[1])
        claim_token = cast(str, params[2])
        terminal_claim_token = cast(str, params[3])
        terminal_attempt_claim_token = params[4]
        assert isinstance(terminal_attempt_claim_token, str)
        queue = self.database.queue_items.get(queue_item_id)
        run = self.database.runs.get(task_run_id)
        if queue is None or run is None:
            return None
        run_claim = cast(Mapping[str, object] | None, run["claim"])
        if queue["run_id"] != task_run_id or not (
            (
                queue["state"] == "claimed"
                and queue["claim_token"] == claim_token
                and run_claim is not None
                and run_claim["claim_token"] == claim_token
                and run["state"]
                in {
                    "claimed",
                    "running",
                    "cancel_requested",
                }
            )
            or (queue["state"] == "dead" and run["state"] == "expired")
            or (
                queue["state"] == "dead"
                and run["state"] == "cancelled"
                and run["last_attempt_id"] is not None
                and any(
                    (
                        segment["state"] == "abandoned"
                        and isinstance(segment["claim"], Mapping)
                        and segment["claim"].get("claim_token")
                        == terminal_claim_token
                    )
                    or (
                        segment["segment_id"] == queue["segment_id"]
                        and segment["state"] == "suspended"
                        and segment["claim"] is None
                        and self._task_attempt_claim_token(
                            run["last_attempt_id"]
                        )
                        == terminal_attempt_claim_token
                    )
                    for segment in self.database.segments.values()
                    if segment["attempt_id"] == run["last_attempt_id"]
                )
            )
        ):
            return None
        attempt_id = cast(str | None, run["last_attempt_id"])
        previous_segment_id = cast(str | None, queue["segment_id"])
        if attempt_id is None or previous_segment_id is None:
            return None
        attempt = self.database.attempts.get(attempt_id)
        previous = self.database.segments.get(previous_segment_id)
        segments = tuple(
            row
            for row in self.database.segments.values()
            if row["attempt_id"] == attempt_id
        )
        if attempt is None or previous is None or not segments:
            return None
        active = max(
            segments,
            key=lambda row: cast(int, row["segment_number"]),
        )
        if queue["state"] == "dead" and (
            (
                run["state"] == TaskRunState.EXPIRED.value
                and attempt["state"] != TaskAttemptState.FAILED.value
            )
            or (
                run["state"] == TaskRunState.CANCELLED.value
                and (
                    attempt["state"] != TaskAttemptState.ABANDONED.value
                    or active["state"]
                    not in {
                        TaskAttemptSegmentState.SUSPENDED.value,
                        TaskAttemptSegmentState.ABANDONED.value,
                    }
                )
            )
        ):
            return None
        run_claim_token = (
            run_claim.get("claim_token")
            if isinstance(run_claim, Mapping)
            else None
        )
        attempt_context = attempt.get("context")
        attempt_claim = (
            attempt_context.get("claim")
            if isinstance(attempt_context, Mapping)
            else None
        )
        attempt_claim_token = (
            attempt_claim.get("claim_token")
            if isinstance(attempt_claim, Mapping)
            else None
        )
        previous_claim = previous.get("claim")
        previous_claim_token = (
            previous_claim.get("claim_token")
            if isinstance(previous_claim, Mapping)
            else None
        )
        active_claim = active.get("claim")
        active_claim_token = (
            active_claim.get("claim_token")
            if isinstance(active_claim, Mapping)
            else None
        )
        return {
            **deepcopy(queue),
            "durable_run_state": run["state"],
            "durable_run_claim_token": run_claim_token,
            "durable_attempt_id": attempt["attempt_id"],
            "durable_attempt_state": attempt["state"],
            "durable_attempt_claim_token": attempt_claim_token,
            "previous_segment_id": previous["segment_id"],
            "previous_attempt_id": previous["attempt_id"],
            "previous_run_id": previous["run_id"],
            "previous_segment_number": previous["segment_number"],
            "previous_segment_state": previous["state"],
            "previous_segment_claim_token": previous_claim_token,
            "previous_request_id": previous["request_id"],
            "previous_continuation_id": previous["continuation_id"],
            "previous_checkpoint_id": previous["checkpoint_id"],
            "active_segment_id": active["segment_id"],
            "active_attempt_id": active["attempt_id"],
            "active_run_id": active["run_id"],
            "active_segment_number": active["segment_number"],
            "active_segment_state": active["state"],
            "active_segment_claim_token": active_claim_token,
            "active_resumed_from_segment_id": active[
                "resumed_from_segment_id"
            ],
            "active_request_id": active["request_id"],
            "active_continuation_id": active["continuation_id"],
            "active_checkpoint_id": active["checkpoint_id"],
        }

    def _task_attempt_claim_token(self, attempt_id: object) -> str | None:
        if not isinstance(attempt_id, str):
            return None
        attempt = self.database.attempts.get(attempt_id)
        if attempt is None:
            return None
        context = attempt.get("context")
        if not isinstance(context, Mapping):
            return None
        claim = context.get("claim")
        if not isinstance(claim, Mapping):
            return None
        claim_token = claim.get("claim_token")
        return claim_token if isinstance(claim_token, str) else None

    def _delete_record(self, request_id: str) -> None:
        self.database.records.pop(request_id, None)
        continuation_ids = tuple(
            continuation_id
            for continuation_id, row in self.database.continuations.items()
            if row["request_id"] == request_id
        )
        for continuation_id in continuation_ids:
            self.database.continuations.pop(continuation_id, None)
        self.database.resolution_keys = {
            key: row
            for key, row in self.database.resolution_keys.items()
            if key[0] != request_id
        }
        self.database.outbox = {
            key: row
            for key, row in self.database.outbox.items()
            if row["request_id"] != request_id
        }

    def _delete_orphaned_branches(
        self,
        run_id: str,
        scope_identity_digest: str,
    ) -> None:
        if any(
            row["run_id"] == run_id
            and row["scope_identity_digest"] == scope_identity_digest
            for row in self.database.records.values()
        ):
            return
        self.database.branches = {
            key: row
            for key, row in self.database.branches.items()
            if row["run_id"] != run_id
            or row["scope_identity_digest"] != scope_identity_digest
        }

    def _claim_outbox(self, params: tuple[object, ...]) -> None:
        pending_at = cast(datetime, params[0])
        expired_at = cast(datetime, params[1])
        limit = cast(int, params[2])
        owner, lease, updated_at = params[3:]
        candidates = [
            row
            for row in self.database.outbox.values()
            if (
                row["status"] == "pending"
                and cast(datetime, row["available_at"]) <= pending_at
            )
            or (
                row["status"] == "claimed"
                and cast(datetime, row["claim_lease_expires_at"]) <= expired_at
            )
        ]
        candidates.sort(
            key=lambda row: (row["available_at"], row["outbox_id"])
        )
        for row in candidates[:limit]:
            row.update(
                status="claimed",
                claim_owner_id=owner,
                claim_lease_expires_at=lease,
                fencing_token=cast(int, row["fencing_token"]) + 1,
                attempts=cast(int, row["attempts"]) + 1,
                updated_at=updated_at,
            )
        self.rows = tuple(candidates[:limit])

    def _complete_outbox(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        delivered_at, updated_at, outbox_id, owner, fence = params
        row = self.database.outbox.get(cast(str, outbox_id))
        if (
            row is None
            or row["status"] != "claimed"
            or row["claim_owner_id"] != owner
            or row["fencing_token"] != fence
        ):
            return None
        row.update(
            status="delivered",
            claim_owner_id=None,
            claim_lease_expires_at=None,
            delivered_at=delivered_at,
            updated_at=updated_at,
        )
        return row

    def _release_outbox(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        status, error, updated_at, outbox_id, owner, fence = params
        row = self.database.outbox.get(cast(str, outbox_id))
        if (
            row is None
            or row["status"] != "claimed"
            or row["claim_owner_id"] != owner
            or row["fencing_token"] != fence
        ):
            return None
        row.update(
            status=status,
            claim_owner_id=None,
            claim_lease_expires_at=None,
            last_error_code=error,
            updated_at=updated_at,
        )
        return row

    def _select_checkpoint(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        request_id, continuation_id, task_run_id, checkpoint_id, _ = params
        continuation = self.database.continuations.get(
            cast(str, continuation_id)
        )
        record = self.database.records.get(cast(str, request_id))
        if (
            continuation is None
            or record is None
            or continuation["request_id"] != request_id
            or continuation["task_run_id"] != task_run_id
            or (
                checkpoint_id is not None
                and continuation["checkpoint_id"] != checkpoint_id
            )
            or continuation["lifecycle_state"] != "pending"
            or record["request_state"] != "pending"
        ):
            return None
        return {"continuation_id": continuation_id}

    def _insert_attempt_segment(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            segment_id,
            attempt_id,
            run_id,
            segment_number,
            state,
            claim,
            resumed_from_segment_id,
            metadata,
            created_at,
            updated_at,
        ) = params
        assert isinstance(segment_id, str)
        if segment_id in self.database.segments:
            return None
        assert isinstance(claim, str)
        assert isinstance(metadata, str)
        row = {
            "segment_id": segment_id,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "segment_number": segment_number,
            "state": state,
            "claim": loads(claim),
            "resumed_from_segment_id": resumed_from_segment_id,
            "request_id": None,
            "continuation_id": None,
            "checkpoint_id": None,
            "metadata": loads(metadata),
            "created_at": created_at,
            "updated_at": updated_at,
        }
        self.database.segments[segment_id] = row
        return row

    def _suspend_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, result, metadata, updated_at, run_id, expected, token = params
        row = self.database.runs.get(cast(str, run_id))
        if (
            row is None
            or row["state"] != expected
            or cast(Mapping[str, object], row["claim"]).get("claim_token")
            != token
        ):
            return None
        row.update(
            state=state,
            result=loads(cast(str, result)),
            claim=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _suspend_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, result, metadata, updated_at, attempt_id, expected = params
        row = self.database.attempts.get(cast(str, attempt_id))
        if row is None or row["state"] != expected:
            return None
        context = dict(cast(Mapping[str, object], row["context"]))
        context["claim"] = None
        row.update(
            state=state,
            result=loads(cast(str, result)),
            context=context,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _suspend_segment(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            request_id,
            continuation_id,
            checkpoint_id,
            metadata,
            updated_at,
            segment_id,
            expected,
        ) = params
        row = self.database.segments.get(cast(str, segment_id))
        if row is None or row["state"] != expected:
            return None
        row.update(
            state=state,
            claim=None,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _suspend_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            attempt_id,
            segment_id,
            request_id,
            continuation_id,
            metadata,
            updated_at,
            queue_item_id,
            expected,
            token,
        ) = params
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] != token
        ):
            return None
        row.update(
            state=state,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            attempt_id=attempt_id,
            segment_id=segment_id,
            request_id=request_id,
            continuation_id=continuation_id,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _insert_transition(
        self,
        table: dict[str, dict[str, object]],
        params: tuple[object, ...],
        keys: tuple[str, ...],
    ) -> dict[str, object] | None:
        transition_id = cast(str, params[0])
        if transition_id in table:
            return None
        row = dict(zip(keys, params, strict=True))
        row["metadata"] = loads(cast(str, row["metadata"]))
        table[transition_id] = row
        return row

    def _select_accepted_resolution(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            request_id,
            continuation_id,
            run_id,
            revision,
            interaction_state,
        ) = params
        record = self.database.records.get(cast(str, request_id))
        continuation = self.database.continuations.get(
            cast(str, continuation_id)
        )
        return next(
            (
                row
                for row in self.database.outbox.values()
                if row["request_id"] == request_id
                and row["continuation_id"] == continuation_id
                and row["task_run_id"] == run_id
                and row["resolution_revision"] == revision
                and row["status"] in {"pending", "claimed", "delivered"}
                and continuation is not None
                and continuation["lifecycle_state"]
                in {"ready", "claimed", "dispatching", "completed"}
                and record is not None
                and record["request_state"] == interaction_state
                and record["state_revision"] == revision
            ),
            None,
        )

    def _select_released_continuation(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        request_id, continuation_id, run_id, checkpoint_id = params
        continuation = self.database.continuations.get(
            cast(str, continuation_id)
        )
        record = self.database.records.get(cast(str, request_id))
        if (
            continuation is None
            or record is None
            or continuation["request_id"] != request_id
            or continuation["task_run_id"] != run_id
            or continuation["checkpoint_id"] != checkpoint_id
            or continuation["lifecycle_state"] != "ready"
            or record["request_state"] not in {"answered", "timed_out"}
        ):
            return None
        return {"continuation_id": continuation_id}

    def _requeue_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, run_id, expected = params
        row = self.database.runs.get(cast(str, run_id))
        if row is None or row["state"] != expected or row["claim"] is not None:
            return None
        row.update(
            state=state,
            result=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _clear_suspended_attempt_result(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        updated_at, attempt_id, expected = params
        row = self.database.attempts.get(cast(str, attempt_id))
        if row is None or row["state"] != expected:
            return None
        row.update(result=None, updated_at=updated_at)
        return row

    def _requeue_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, available_at, metadata, updated_at, queue_item_id, expected = (
            params
        )
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] is not None
        ):
            return None
        row.update(
            state=state,
            available_at=available_at,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _terminalize_suspended_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, queue_item_id, expected = params
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] is not None
        ):
            return None
        row.update(
            state=state,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _settle_segment(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            claim,
            request_id,
            continuation_id,
            checkpoint_id,
            metadata,
            updated_at,
            segment_id,
            expected,
        ) = params
        row = self.database.segments.get(cast(str, segment_id))
        if row is None or row["state"] != expected:
            return None
        row.update(
            state=state,
            claim=loads(cast(str, claim)) if claim is not None else None,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            metadata=(
                loads(cast(str, metadata))
                if metadata is not None
                else row["metadata"]
            ),
            updated_at=updated_at,
        )
        return row

    def _settle_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            result,
            updated_at,
            attempt_id,
            expected,
            run_id,
            run_state,
            nullable_token,
            claim_token,
        ) = params
        row = self.database.attempts.get(cast(str, attempt_id))
        run = self.database.runs.get(cast(str, run_id))
        run_claim = (
            cast(Mapping[str, object], run["claim"])
            if run is not None and run["claim"] is not None
            else None
        )
        token_matches = (
            run_claim is None
            if nullable_token is None
            else (
                run_claim is not None
                and run_claim.get("claim_token") == claim_token
            )
        )
        if (
            row is None
            or run is None
            or row["state"] != expected
            or row["run_id"] != run_id
            or run["state"] != run_state
            or not token_matches
        ):
            return None
        row.update(
            state=state,
            result=(
                loads(cast(str, result))
                if result is not None
                else row.get("result")
            ),
            updated_at=updated_at,
        )
        return row

    def _settle_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            result,
            clear_claim,
            updated_at,
            run_id,
            expected,
            nullable_token,
            claim_token,
        ) = params
        row = self.database.runs.get(cast(str, run_id))
        run_claim = (
            cast(Mapping[str, object], row["claim"])
            if row is not None and row["claim"] is not None
            else None
        )
        token_matches = (
            run_claim is None
            if nullable_token is None
            else (
                run_claim is not None
                and run_claim.get("claim_token") == claim_token
            )
        )
        if row is None or row["state"] != expected or not token_matches:
            return None
        row.update(
            state=state,
            result=(
                loads(cast(str, result))
                if result is not None
                else row.get("result")
            ),
            claim=None if clear_claim else row["claim"],
            updated_at=updated_at,
        )
        return row

    def _settle_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, updated_at, queue_item_id, claim_token = params
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != "claimed"
            or row["claim_token"] != claim_token
        ):
            return None
        row.update(
            state=state,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            updated_at=updated_at,
        )
        return row

    def _release_reentry_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        metadata, updated_at, attempt_id, run_id, claim_token = params
        row = self.database.attempts.get(cast(str, attempt_id))
        run = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], run["claim"])
            if run is not None and run["claim"] is not None
            else None
        )
        if (
            row is None
            or run is None
            or row["run_id"] != run_id
            or row["state"] != "suspended"
            or run["state"] != "claimed"
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        context = dict(cast(Mapping[str, object], row["context"]))
        context["claim"] = None
        row.update(
            context=context,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_reentry_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, run_id, expected, claim_token = params
        row = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], row["claim"])
            if row is not None and row["claim"] is not None
            else None
        )
        if (
            row is None
            or row["state"] != expected
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        row.update(
            state=state,
            result=None,
            claim=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_reentry_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            available_at,
            metadata,
            updated_at,
            queue_item_id,
            expected,
            claim_token,
        ) = params
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] != claim_token
        ):
            return None
        row.update(
            state=state,
            available_at=available_at,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _fail_reentry_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            result,
            metadata,
            updated_at,
            attempt_id,
            run_id,
            claim_token,
        ) = params
        row = self.database.attempts.get(cast(str, attempt_id))
        run = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], run["claim"])
            if run is not None and run["claim"] is not None
            else None
        )
        if (
            row is None
            or run is None
            or row["run_id"] != run_id
            or row["state"] != "suspended"
            or run["state"] != "claimed"
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        context = dict(cast(Mapping[str, object], row["context"]))
        context["claim"] = None
        row.update(
            state=state,
            result=loads(cast(str, result)),
            context=context,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _fail_reentry_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            result,
            metadata,
            updated_at,
            run_id,
            expected,
            claim_token,
        ) = params
        row = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], row["claim"])
            if row is not None and row["claim"] is not None
            else None
        )
        if (
            row is None
            or row["state"] != expected
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        row.update(
            state=state,
            result=loads(cast(str, result)),
            claim=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _fail_reentry_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, queue_item_id, expected, claim_token = (
            params
        )
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] != claim_token
        ):
            return None
        row.update(
            state=state,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_running_segment(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            request_id,
            continuation_id,
            checkpoint_id,
            metadata,
            updated_at,
            segment_id,
            expected,
        ) = params
        row = self.database.segments.get(cast(str, segment_id))
        if row is None or row["state"] != expected:
            return None
        row.update(
            state=state,
            claim=None,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_running_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, attempt_id, run_id, claim_token = params
        row = self.database.attempts.get(cast(str, attempt_id))
        run = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], run["claim"])
            if run is not None and run["claim"] is not None
            else None
        )
        if (
            row is None
            or run is None
            or row["run_id"] != run_id
            or row["state"] != "running"
            or run["state"] != "running"
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        context = dict(cast(Mapping[str, object], row["context"]))
        context["claim"] = None
        row.update(
            state=state,
            result=None,
            context=context,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_running_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        state, metadata, updated_at, run_id, expected, claim_token = params
        row = self.database.runs.get(cast(str, run_id))
        claim = (
            cast(Mapping[str, object], row["claim"])
            if row is not None and row["claim"] is not None
            else None
        )
        if (
            row is None
            or row["state"] != expected
            or claim is None
            or claim.get("claim_token") != claim_token
        ):
            return None
        row.update(
            state=state,
            result=None,
            claim=None,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _release_running_queue(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        (
            state,
            available_at,
            attempt_id,
            segment_id,
            request_id,
            continuation_id,
            metadata,
            updated_at,
            queue_item_id,
            expected,
            claim_token,
        ) = params
        row = self.database.queue_items.get(cast(str, queue_item_id))
        if (
            row is None
            or row["state"] != expected
            or row["claim_token"] != claim_token
        ):
            return None
        row.update(
            state=state,
            available_at=available_at,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            attempt_id=attempt_id,
            segment_id=segment_id,
            request_id=request_id,
            continuation_id=continuation_id,
            metadata={
                **cast(Mapping[str, object], row["metadata"]),
                **loads(cast(str, metadata)),
            },
            updated_at=updated_at,
        )
        return row

    def _insert_event(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        keys = (
            "event_id",
            "run_id",
            "attempt_id",
            "sequence",
            "event_type",
            "payload",
            "metadata",
            "event_time",
            "created_at",
        )
        event_id = cast(str, params[0])
        if event_id in self.database.events:
            return None
        row = dict(zip(keys, params, strict=True))
        row["payload"] = loads(cast(str, row["payload"]))
        row["metadata"] = loads(cast(str, row["metadata"]))
        self.database.events[event_id] = row
        return row


class FullFakePgsqlDatabase(FakePgsqlDatabase):
    """Add complete task-store and queue behavior to the interaction fake."""

    def __init__(self) -> None:
        super().__init__()
        self.definitions: dict[str, dict[str, object]] = {}
        self.flow_executions: dict[str, dict[str, object]] = {}
        self.usage: dict[str, dict[str, object]] = {}
        self.artifacts: dict[str, dict[str, object]] = {}
        self.idempotency: dict[str, dict[str, object]] = {}
        self.executed_queries: list[str] = []
        self.before_attempt_update = None
        self.before_last_attempt_update = None
        self.depth_returns_none = False
        self.health_returns_none = False
        self.execute_error: BaseException | None = None
        self.fail_on_query: str | None = None
        self.race_on_idempotency_insert = False
        self.drop_idempotency_insert = False
        self.stale_transition = False
        self.stale_claim_assignment = False
        self.stale_heartbeat_run = False
        self.stale_last_attempt_update = False
        self.stale_complete_queue_item = False
        self.stale_retry_queue_item = False

    def connection(self) -> "FullFakeConnectionContext":
        """Return a connection whose cursor understands every task query."""
        return FullFakeConnectionContext(self)

    def snapshot(self) -> dict[str, object]:
        """Return all interaction, task-store, and queue rows."""
        snapshot = super().snapshot()
        snapshot.update(
            definitions=deepcopy(self.definitions),
            flow_executions=deepcopy(self.flow_executions),
            usage=deepcopy(self.usage),
            artifacts=deepcopy(self.artifacts),
            idempotency=deepcopy(self.idempotency),
        )
        return snapshot

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore all interaction, task-store, and queue rows."""
        super().restore(snapshot)
        self.definitions = cast(
            dict[str, dict[str, object]],
            snapshot["definitions"],
        )
        self.flow_executions = cast(
            dict[str, dict[str, object]],
            snapshot["flow_executions"],
        )
        self.usage = cast(
            dict[str, dict[str, object]],
            snapshot["usage"],
        )
        self.artifacts = cast(
            dict[str, dict[str, object]],
            snapshot["artifacts"],
        )
        self.idempotency = cast(
            dict[str, dict[str, object]],
            snapshot["idempotency"],
        )


class FullFakeConnectionContext:
    """Acquire one full fake connection."""

    def __init__(self, database: FullFakePgsqlDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FullFakeConnection":
        return FullFakeConnection(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FullFakeConnection:
    """Expose the shared transaction and full cursor."""

    row_factory: object = None

    def __init__(self, database: FullFakePgsqlDatabase) -> None:
        self.database = database

    def transaction(self) -> FakeTransactionContext:
        return FakeTransactionContext(self.database)

    def cursor(self) -> "FullFakeCursorContext":
        return FullFakeCursorContext(self.database)

    async def set_autocommit(self, value: bool) -> None:
        assert isinstance(value, bool)


class FullFakeCursorContext:
    """Acquire one cursor over the combined fake tables."""

    def __init__(self, database: FullFakePgsqlDatabase) -> None:
        self.database = database

    async def __aenter__(self) -> "FullFakeCursor":
        return FullFakeCursor(self.database)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FullFakeCursor:
    """Interpret the combined interaction, task-store, and queue SQL."""

    def __init__(self, database: FullFakePgsqlDatabase) -> None:
        self.database = database
        self.row: dict[str, object] | None = None
        self.rows: tuple[dict[str, object], ...] = ()

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | Mapping[str, object] | None = None,
    ) -> None:
        params = (
            tuple(parameters)
            if parameters is not None and not isinstance(parameters, Mapping)
            else ()
        )
        self.row = None
        self.rows = ()
        if self.database.execute_error is not None:
            raise self.database.execute_error
        self.database.executed_queries.append(query)
        if (
            self.database.fail_on_query is not None
            and self.database.fail_on_query in query
        ):
            raise RuntimeError("backend failure includes raw details")
        if query == _INSERT_ATTEMPT_SEGMENT_SQL:
            self.row = self._insert_segment(params)
            return
        if 'UPDATE "task_runs"' in query and '"last_attempt_id"' in query:
            self.row = (
                self._update_claimed_run_last_attempt(params)
                if len(params) == 4
                else self._update_run_last_attempt(params)
            )
            return
        delegate = FakeCursor(self.database)
        try:
            await delegate.execute(query, params)
        except AssertionError as error:
            if not str(error).startswith("unexpected query:"):
                raise
        else:
            self.row = delegate.row
            self.rows = delegate.rows
            return
        if 'SELECT * FROM "task_definitions"' in query:
            self.row = self.database.definitions.get(cast(str, params[0]))
        elif 'INSERT INTO "task_definitions"' in query:
            self.row = self._insert_definition(params)
        elif 'INSERT INTO "task_runs"' in query:
            self.row = self._insert_run(params)
        elif 'UPDATE "task_runs"' in query and '"claim" = %s::jsonb' in query:
            self.row = self._assign_run_claim(params)
        elif (
            'UPDATE "task_runs"' in query
            and '"claim" = CASE' in query
            and '"state" = %s' in query
        ):
            self.row = self._transition_claimed_run(params)
        elif (
            'UPDATE "task_attempts" a' in query
            and '"context" = jsonb_set' in query
        ):
            self.row = self._reuse_claimed_suspended_attempt(params)
        elif 'UPDATE "task_attempts" a' in query:
            self.row = self._transition_claimed_attempt(params)
        elif 'FROM "task_attempts"' in query:
            self.rows = self._attempts_for_run(cast(str, params[0]))
        elif 'INSERT INTO "task_attempts"' in query:
            self.row = self._insert_attempt(params)
        elif 'FROM "task_usage_records"' in query:
            self.rows = self._filtered(
                self.database.usage,
                cast(str, params[0]),
                cast(str | None, params[1]),
                None,
                "sequence",
            )
        elif 'FROM "task_events"' in query:
            self.rows = self._filtered(
                self.database.events,
                cast(str, params[0]),
                cast(str | None, params[1]),
                cast(int | None, params[3]),
                "sequence",
            )
        elif 'SELECT * FROM "task_artifacts"' in query:
            self.rows = self._artifacts_for_run(params)
        elif 'SELECT * FROM "task_idempotency_keys"' in query:
            self.row = self._select_idempotency(params)
        elif 'INSERT INTO "task_idempotency_keys"' in query:
            self.row = self._insert_idempotency(params)
        elif 'INSERT INTO "task_queue_items"' in query:
            self.row = self._insert_queue_item(params)
        elif (
            'SELECT q.*, r."state" AS "run_state", r."last_attempt_id"'
            in query
        ):
            self.row = self._fenced_queue_item(params)
        elif 'UPDATE "task_queue_items" q' in query and "candidate" in query:
            self.row = self._claim_queue_item(params)
        elif 'UPDATE "task_queue_items"' in query and '"state" = %s' in query:
            self.row = self._complete_queue_item(params)
        elif "COUNT(*) FILTER" in query:
            self.row = self._depth(params)
        else:
            raise AssertionError(f"unexpected query: {query}")

    async def fetchone(self) -> dict[str, object] | None:
        return self.row

    async def fetchall(self) -> tuple[dict[str, object], ...]:
        return self.rows

    async def close(self) -> None:
        return None

    def _insert_segment(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        segment_id = cast(str, params[0])
        if segment_id in self.database.segments:
            return None
        keys = (
            "segment_id",
            "attempt_id",
            "run_id",
            "segment_number",
            "state",
            "claim",
            "resumed_from_segment_id",
            "metadata",
            "created_at",
            "updated_at",
        )
        row = dict(zip(keys, params, strict=True))
        row["claim"] = (
            loads(cast(str, row["claim"]))
            if row["claim"] is not None
            else None
        )
        row["metadata"] = loads(cast(str, row["metadata"]))
        row.update(
            request_id=None,
            continuation_id=None,
            checkpoint_id=None,
        )
        self.database.segments[segment_id] = row
        return row

    def _insert_definition(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        definition_id = cast(str, params[0])
        if definition_id in self.database.definitions:
            return None
        row = {
            "definition_id": definition_id,
            "name": params[1],
            "version": params[2],
            "spec_hash": params[3],
            "definition": loads(cast(str, params[4])),
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
        }
        self.database.definitions[definition_id] = row
        return row

    def _insert_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run_id = cast(str, params[0])
        if run_id in self.database.runs:
            return None
        row = {
            "run_id": run_id,
            "definition_id": params[1],
            "state": params[2],
            "queue_name": params[3],
            "request": loads(cast(str, params[4])),
            "metadata": loads(cast(str, params[5])),
            "claim": None,
            "last_attempt_id": None,
            "result": None,
            "created_at": params[6],
            "updated_at": params[7],
        }
        self.database.runs[run_id] = row
        return row

    def _assign_run_claim(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run = self.database.runs.get(cast(str, params[3]))
        if self.database.stale_claim_assignment:
            self.database.stale_claim_assignment = False
            return None
        if (
            run is None
            or run["state"] != params[4]
            or run.get("claim") is not None
        ):
            return None
        run["state"] = params[0]
        run["claim"] = loads(cast(str, params[1]))
        run["updated_at"] = params[2]
        return run

    def _transition_claimed_run(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run = self.database.runs.get(cast(str, params[4]))
        claim = cast(
            dict[str, object] | None,
            run.get("claim") if run is not None else None,
        )
        if self.database.stale_transition:
            self.database.stale_transition = False
            return None
        if (
            run is None
            or run["state"] != params[5]
            or claim is None
            or claim.get("claim_token") != params[6]
        ):
            return None
        run.update(
            state=params[0],
            result=(
                loads(cast(str, params[1]))
                if params[1] is not None
                else run.get("result")
            ),
            claim=None if params[2] else claim,
            updated_at=params[3],
        )
        return run

    def _update_run_last_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        if self.database.before_last_attempt_update is not None:
            self.database.before_last_attempt_update(params)
        run = self.database.runs.get(cast(str, params[2]))
        if run is None or run["state"] != params[3]:
            return None
        claim = cast(dict[str, object] | None, run["claim"])
        if (claim is None and params[4] is not None) or (
            claim is not None and claim["claim_token"] != params[5]
        ):
            return None
        run["last_attempt_id"] = params[0]
        run["updated_at"] = params[1]
        return run

    def _update_claimed_run_last_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run = self.database.runs.get(cast(str, params[2]))
        claim = cast(
            dict[str, object] | None,
            run.get("claim") if run is not None else None,
        )
        if self.database.stale_last_attempt_update:
            self.database.stale_last_attempt_update = False
            return None
        if (
            run is None
            or run["state"] != TaskRunState.CLAIMED.value
            or claim is None
            or claim.get("claim_token") != params[3]
        ):
            return None
        run["last_attempt_id"] = params[0]
        run["updated_at"] = params[1]
        return run

    def _insert_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        attempt_id = cast(str, params[0])
        if attempt_id in self.database.attempts:
            return None
        row = {
            "attempt_id": attempt_id,
            "run_id": params[1],
            "attempt_number": params[2],
            "state": params[3],
            "context": loads(cast(str, params[4])),
            "result": None,
            "metadata": loads(cast(str, params[5])),
            "created_at": params[6],
            "updated_at": params[7],
        }
        self.database.attempts[attempt_id] = row
        return row

    def _transition_claimed_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        attempt = self.database.attempts.get(cast(str, params[0]))
        if attempt is None or attempt["state"] not in cast(
            list[str],
            params[1],
        ):
            return None
        run = self.database.runs[cast(str, attempt["run_id"])]
        claim = cast(dict[str, object] | None, run.get("claim"))
        if (
            run["state"]
            not in {
                TaskRunState.CLAIMED.value,
                TaskRunState.RUNNING.value,
                TaskRunState.CANCEL_REQUESTED.value,
            }
            or run["last_attempt_id"] != attempt["attempt_id"]
            or claim is None
            or claim.get("claim_token") != params[2]
        ):
            return None
        from_state = attempt["state"]
        attempt["state"] = params[3]
        if params[4] is not None:
            attempt["result"] = loads(cast(str, params[4]))
        attempt["updated_at"] = params[5]
        return {**attempt, "from_state": from_state}

    def _reuse_claimed_suspended_attempt(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        attempt = self.database.attempts.get(cast(str, params[2]))
        if (
            attempt is None
            or attempt["state"] != TaskAttemptState.SUSPENDED.value
        ):
            return None
        run = self.database.runs[cast(str, attempt["run_id"])]
        claim = cast(dict[str, object] | None, run.get("claim"))
        if (
            run["state"] != TaskRunState.CLAIMED.value
            or run["last_attempt_id"] != attempt["attempt_id"]
            or claim is None
            or claim.get("claim_token") != params[3]
        ):
            return None
        context = cast(dict[str, object], attempt["context"])
        context["claim"] = loads(cast(str, params[0]))
        attempt["updated_at"] = params[1]
        return attempt

    def _insert_idempotency(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        identity_key = cast(str, params[0])
        if self.database.drop_idempotency_insert:
            self.database.drop_idempotency_insert = False
            return None
        existing = self.database.idempotency.get(identity_key)
        if existing is not None:
            expires_at = cast(datetime | None, existing["expires_at"])
            checked_at = cast(datetime, params[14])
            if expires_at is None or expires_at > checked_at:
                return None
        row = {
            "identity_key": identity_key,
            "task_name": params[1],
            "task_version": params[2],
            "spec_hash": params[3],
            "owner_scope_hash": loads(cast(str, params[4])),
            "strategy": params[5],
            "window_hash": (
                loads(cast(str, params[6])) if params[6] is not None else None
            ),
            "input_hash": (
                loads(cast(str, params[7])) if params[7] is not None else None
            ),
            "file_hash": (
                loads(cast(str, params[8])) if params[8] is not None else None
            ),
            "custom_hash": (
                loads(cast(str, params[9])) if params[9] is not None else None
            ),
            "run_id": params[10],
            "metadata": loads(cast(str, params[11])),
            "expires_at": params[12],
            "created_at": params[13],
        }
        self.database.idempotency[identity_key] = row
        return row

    def _select_idempotency(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = self.database.idempotency.get(cast(str, params[0]))
        if row is None:
            return None
        expires_at = cast(datetime | None, row["expires_at"])
        checked_at = cast(datetime, params[1])
        if expires_at is not None and expires_at <= checked_at:
            return None
        return row

    def _insert_queue_item(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        run_id = cast(str, params[1])
        if any(
            row["run_id"] == run_id
            and row["state"]
            in {
                TaskQueueItemState.AVAILABLE.value,
                TaskQueueItemState.CLAIMED.value,
            }
            for row in self.database.queue_items.values()
        ):
            return None
        row = {
            "queue_item_id": params[0],
            "run_id": run_id,
            "queue_name": params[2],
            "state": params[3],
            "priority": params[4],
            "available_at": params[5],
            "claimed_at": None,
            "lease_expires_at": None,
            "worker_id": None,
            "claim_token": None,
            "heartbeat_at": None,
            "attempts": params[6],
            "metadata": loads(cast(str, params[7])),
            "created_at": params[8],
            "updated_at": params[9],
        }
        self.database.queue_items[cast(str, params[0])] = row
        return row

    def _claim_queue_item(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        queue_name = cast(str, params[0])
        checked_at = cast(datetime, params[1])
        candidates = (
            self._with_run_state(row)
            for row in self.database.queue_items.values()
            if row["queue_name"] == queue_name
            and row["state"] == TaskQueueItemState.AVAILABLE.value
            and cast(datetime, row["available_at"]) <= checked_at
            and self.database.runs[cast(str, row["run_id"])]["state"]
            == TaskRunState.QUEUED.value
        )
        ordered = sorted(
            candidates,
            key=lambda row: (
                -cast(int, row["priority"]),
                cast(datetime, row["available_at"]),
                cast(str, row["queue_item_id"]),
            ),
        )
        if not ordered:
            return None
        row = self.database.queue_items[cast(str, ordered[0]["queue_item_id"])]
        run = self.database.runs[cast(str, row["run_id"])]
        last_attempt_id = cast(str | None, run["last_attempt_id"])
        previous_attempt = (
            self.database.attempts.get(last_attempt_id)
            if last_attempt_id is not None
            else None
        )
        is_reentry = (
            previous_attempt is not None
            and previous_attempt["state"] == TaskAttemptState.SUSPENDED.value
        )
        claim_updates = {
            "state": TaskQueueItemState.CLAIMED.value,
            "claimed_at": params[2],
            "lease_expires_at": params[3],
            "worker_id": params[4],
            "claim_token": params[5],
            "heartbeat_at": params[6],
            "attempts": (
                cast(int, row["attempts"])
                if is_reentry
                else cast(int, row["attempts"]) + 1
            ),
            "updated_at": params[7],
        }
        row.update(claim_updates)
        return {
            **self._with_run_state(row),
            "last_attempt_id": last_attempt_id,
            "is_reentry": is_reentry,
        }

    def _fenced_queue_item(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = self.database.queue_items.get(cast(str, params[0]))
        if row is None:
            return None
        run = self.database.runs[cast(str, row["run_id"])]
        claim = cast(dict[str, object] | None, run.get("claim"))
        if (
            row["state"] != TaskQueueItemState.CLAIMED.value
            or row["claim_token"] != params[1]
            or cast(datetime, row["lease_expires_at"])
            <= cast(datetime, params[2])
            or run["state"]
            not in {
                TaskRunState.CLAIMED.value,
                TaskRunState.RUNNING.value,
                TaskRunState.CANCEL_REQUESTED.value,
            }
            or claim is None
            or claim.get("claim_token") != params[3]
        ):
            return None
        return {
            **self._with_run_state(row),
            "last_attempt_id": run["last_attempt_id"],
        }

    def _complete_queue_item(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        row = self.database.queue_items.get(cast(str, params[2]))
        if self.database.stale_complete_queue_item:
            self.database.stale_complete_queue_item = False
            return None
        if (
            row is None
            or row["state"] != TaskQueueItemState.CLAIMED.value
            or row["claim_token"] != params[3]
        ):
            return None
        row.update(
            state=params[0],
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
            updated_at=params[1],
        )
        return self._with_run_state(row)

    def _depth(
        self,
        params: tuple[object, ...],
    ) -> dict[str, object] | None:
        if self.database.depth_returns_none:
            return None
        checked_at = cast(datetime, params[0])
        queue_name = cast(str, params[2])
        rows = tuple(
            self._with_run_state(row)
            for row in self.database.queue_items.values()
            if row["queue_name"] == queue_name
        )
        return {
            "available": sum(
                1
                for row in rows
                if row["state"] == TaskQueueItemState.AVAILABLE.value
                and cast(datetime, row["available_at"]) <= checked_at
            ),
            "scheduled": sum(
                1
                for row in rows
                if row["state"] == TaskQueueItemState.AVAILABLE.value
                and cast(datetime, row["available_at"]) > checked_at
            ),
            "claimed": sum(
                1
                for row in rows
                if row["state"] == TaskQueueItemState.CLAIMED.value
            ),
            "dead": sum(
                1
                for row in rows
                if row["state"] == TaskQueueItemState.DEAD.value
            ),
            "cancel_requested": sum(
                1
                for row in rows
                if row["state"]
                in {
                    TaskQueueItemState.AVAILABLE.value,
                    TaskQueueItemState.CLAIMED.value,
                }
                and row["run_state"] == TaskRunState.CANCEL_REQUESTED.value
            ),
        }

    def _with_run_state(
        self,
        row: dict[str, object],
    ) -> dict[str, object]:
        run = self.database.runs[cast(str, row["run_id"])]
        return {**row, "run_state": run["state"]}

    def _attempts_for_run(
        self,
        run_id: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            sorted(
                (
                    row
                    for row in self.database.attempts.values()
                    if row["run_id"] == run_id
                ),
                key=lambda row: cast(int, row["attempt_number"]),
            )
        )

    def _filtered(
        self,
        rows: dict[str, dict[str, object]],
        run_id: str,
        attempt_id: str | None,
        after_sequence: int | None,
        order_key: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            sorted(
                (
                    row
                    for row in rows.values()
                    if row["run_id"] == run_id
                    and (attempt_id is None or row["attempt_id"] == attempt_id)
                    and (
                        after_sequence is None
                        or cast(int, row["sequence"]) > after_sequence
                    )
                ),
                key=lambda row: cast(Any, row[order_key]),
            )
        )

    def _artifacts_for_run(
        self,
        params: tuple[object, ...],
    ) -> tuple[dict[str, object], ...]:
        run_id = cast(str, params[0])
        attempt_id = cast(str | None, params[1])
        purpose = cast(str | None, params[3])
        state = cast(str | None, params[5])
        return tuple(
            row
            for row in self.database.artifacts.values()
            if row["run_id"] == run_id
            and (attempt_id is None or row["attempt_id"] == attempt_id)
            and (purpose is None or row["purpose"] == purpose)
            and (state is None or row["state"] == state)
        )
