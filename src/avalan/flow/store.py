from ..pgsql import (
    PgsqlDatabase,
    PgsqlFailureCategory,
    PgsqlOperationError,
    PgsqlUnitOfWork,
    classify_pgsql_error,
)
from ..task.store import (
    TaskSnapshotMetadata,
    TaskSnapshotValue,
    TaskStoreConflictError,
    TaskStoreError,
    TaskStoreNotFoundError,
    empty_snapshot_metadata,
    freeze_snapshot_metadata,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .diagnostics import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowSourceSpan,
)
from .state import (
    FlowEdgeState,
    FlowEdgeTrace,
    FlowExecutionTrace,
    FlowNodeState,
    FlowNodeTrace,
)

from asyncio import CancelledError, Lock
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from json import dumps, loads
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

FlowSnapshotValue: TypeAlias = TaskSnapshotValue
FlowSnapshotMetadata: TypeAlias = TaskSnapshotMetadata


class _TaskRunLookup(Protocol):
    async def get_run(self, run_id: str) -> object: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowNodeAttemptRecord:
    node: str
    attempt: int
    state: FlowNodeState
    duration_ms: int | float | None = None
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    artifact_refs: tuple[FlowSnapshotMetadata, ...] = ()
    metadata: FlowSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.node, "node")
        _assert_positive_int(self.attempt, "attempt")
        assert isinstance(self.state, FlowNodeState)
        _assert_duration(self.duration_ms, "duration_ms")
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(
            self,
            "artifact_refs",
            _freeze_snapshot_mappings(self.artifact_refs, "artifact_refs"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def as_snapshot(self) -> FlowSnapshotMetadata:
        value: dict[str, object] = {
            "node": self.node,
            "attempt": self.attempt,
            "state": self.state.value,
        }
        if self.duration_ms is not None:
            value["duration_ms"] = self.duration_ms
        if self.diagnostics:
            value["diagnostics"] = _diagnostics_to_snapshot(self.diagnostics)
        if self.artifact_refs:
            value["artifact_refs"] = self.artifact_refs
        if self.metadata:
            value["metadata"] = self.metadata
        return freeze_snapshot_metadata(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowExecutionRecord:
    task_run_id: str
    revision: int
    trace: FlowExecutionTrace
    created_at: datetime
    updated_at: datetime
    node_attempts: tuple[FlowNodeAttemptRecord, ...] = ()
    node_outputs: FlowSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )
    selected_outputs: FlowSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )
    loop_counters: Mapping[str, int] = field(default_factory=dict)
    pause_tokens: Mapping[str, str] = field(default_factory=dict)
    diagnostics: tuple[FlowDiagnostic, ...] = ()
    artifact_refs: tuple[FlowSnapshotMetadata, ...] = ()
    metadata: FlowSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.task_run_id, "task_run_id")
        _assert_positive_int(self.revision, "revision")
        assert isinstance(self.trace, FlowExecutionTrace)
        _assert_datetime(self.created_at, "created_at")
        _assert_datetime(self.updated_at, "updated_at")
        assert self.updated_at >= self.created_at
        assert isinstance(self.node_attempts, tuple)
        attempt_keys: set[tuple[str, int]] = set()
        for attempt in self.node_attempts:
            assert isinstance(attempt, FlowNodeAttemptRecord)
            key = (attempt.node, attempt.attempt)
            assert key not in attempt_keys, "node attempts must be unique"
            attempt_keys.add(key)
        object.__setattr__(
            self,
            "node_outputs",
            freeze_snapshot_metadata(self.node_outputs),
        )
        object.__setattr__(
            self,
            "selected_outputs",
            freeze_snapshot_metadata(self.selected_outputs),
        )
        object.__setattr__(
            self,
            "loop_counters",
            _freeze_loop_counters(self.loop_counters),
        )
        object.__setattr__(
            self,
            "pause_tokens",
            _freeze_pause_tokens(self.pause_tokens),
        )
        _assert_diagnostics(self.diagnostics)
        object.__setattr__(
            self,
            "artifact_refs",
            _freeze_snapshot_mappings(self.artifact_refs, "artifact_refs"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def as_snapshot(self) -> FlowSnapshotMetadata:
        value: dict[str, object] = {
            "task_run_id": self.task_run_id,
            "revision": self.revision,
            "trace": flow_trace_to_snapshot(self.trace),
            "node_attempts": tuple(
                attempt.as_snapshot() for attempt in self.node_attempts
            ),
            "node_outputs": self.node_outputs,
            "selected_outputs": self.selected_outputs,
            "loop_counters": self.loop_counters,
            "pause_tokens": self.pause_tokens,
            "diagnostics": _diagnostics_to_snapshot(self.diagnostics),
            "artifact_refs": self.artifact_refs,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return freeze_snapshot_metadata(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowExecutionUpdate:
    trace: FlowExecutionTrace | None = None
    node_attempts: tuple[FlowNodeAttemptRecord, ...] | None = None
    node_outputs: Mapping[str, object] | None = None
    selected_outputs: Mapping[str, object] | None = None
    loop_counters: Mapping[str, int] | None = None
    pause_tokens: Mapping[str, str] | None = None
    diagnostics: tuple[FlowDiagnostic, ...] | None = None
    artifact_refs: tuple[Mapping[str, object], ...] | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.trace is not None:
            assert isinstance(self.trace, FlowExecutionTrace)
        if self.node_attempts is not None:
            assert isinstance(self.node_attempts, tuple)
            for attempt in self.node_attempts:
                assert isinstance(attempt, FlowNodeAttemptRecord)
        if self.node_outputs is not None:
            freeze_snapshot_metadata(self.node_outputs)
        if self.selected_outputs is not None:
            freeze_snapshot_metadata(self.selected_outputs)
        if self.loop_counters is not None:
            _freeze_loop_counters(self.loop_counters)
        if self.pause_tokens is not None:
            _freeze_pause_tokens(self.pause_tokens)
        if self.diagnostics is not None:
            _assert_diagnostics(self.diagnostics)
        if self.artifact_refs is not None:
            _freeze_snapshot_mappings(self.artifact_refs, "artifact_refs")
        if self.metadata is not None:
            freeze_snapshot_metadata(self.metadata)


class FlowStateStore(Protocol):
    async def create_flow_execution(
        self,
        task_run_id: str,
        *,
        trace: FlowExecutionTrace,
        node_attempts: tuple[FlowNodeAttemptRecord, ...] = (),
        node_outputs: Mapping[str, object] | None = None,
        selected_outputs: Mapping[str, object] | None = None,
        loop_counters: Mapping[str, int] | None = None,
        pause_tokens: Mapping[str, str] | None = None,
        diagnostics: tuple[FlowDiagnostic, ...] = (),
        artifact_refs: tuple[Mapping[str, object], ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> FlowExecutionRecord: ...

    async def get_flow_execution(
        self,
        task_run_id: str,
    ) -> FlowExecutionRecord: ...

    async def update_flow_execution(
        self,
        task_run_id: str,
        update: FlowExecutionUpdate,
        *,
        expected_revision: int,
    ) -> FlowExecutionRecord: ...


class InMemoryFlowStateStore:
    def __init__(
        self,
        *,
        task_store: _TaskRunLookup | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if task_store is not None:
            assert hasattr(task_store, "get_run")
        self._task_store = task_store
        self._clock = clock or _utc_now
        self._records: dict[str, FlowExecutionRecord] = {}
        self._lock = Lock()

    async def create_flow_execution(
        self,
        task_run_id: str,
        *,
        trace: FlowExecutionTrace,
        node_attempts: tuple[FlowNodeAttemptRecord, ...] = (),
        node_outputs: Mapping[str, object] | None = None,
        selected_outputs: Mapping[str, object] | None = None,
        loop_counters: Mapping[str, int] | None = None,
        pause_tokens: Mapping[str, str] | None = None,
        diagnostics: tuple[FlowDiagnostic, ...] = (),
        artifact_refs: tuple[Mapping[str, object], ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")
        await self._verify_task_run(task_run_id)
        async with self._lock:
            if task_run_id in self._records:
                raise TaskStoreConflictError("flow execution already exists")
            now = self._now()
            record = _flow_execution_record(
                task_run_id=task_run_id,
                revision=1,
                trace=trace,
                created_at=now,
                updated_at=now,
                node_attempts=node_attempts,
                node_outputs=node_outputs,
                selected_outputs=selected_outputs,
                loop_counters=loop_counters,
                pause_tokens=pause_tokens,
                diagnostics=diagnostics,
                artifact_refs=artifact_refs,
                metadata=metadata,
            )
            self._records[task_run_id] = record
            return record

    async def get_flow_execution(
        self,
        task_run_id: str,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")
        async with self._lock:
            try:
                return self._records[task_run_id]
            except KeyError as error:
                raise TaskStoreNotFoundError(
                    "flow execution was not found"
                ) from error

    async def update_flow_execution(
        self,
        task_run_id: str,
        update: FlowExecutionUpdate,
        *,
        expected_revision: int,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")
        assert isinstance(update, FlowExecutionUpdate)
        _assert_positive_int(expected_revision, "expected_revision")
        async with self._lock:
            current = self._records.get(task_run_id)
            if current is None:
                raise TaskStoreNotFoundError("flow execution was not found")
            if current.revision != expected_revision:
                raise TaskStoreConflictError(
                    "flow execution revision did not match"
                )
            updated = _updated_record(current, update, updated_at=self._now())
            self._records[task_run_id] = updated
            return updated

    async def _verify_task_run(self, task_run_id: str) -> None:
        if self._task_store is None:
            return
        await self._task_store.get_run(task_run_id)

    def _now(self) -> datetime:
        return _ensure_aware_utc(self._clock())


class PgsqlFlowStateStore:
    def __init__(
        self,
        database: PgsqlDatabase,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        assert hasattr(database, "connection")
        self._database = database
        self._clock = clock or _utc_now

    async def create_flow_execution(
        self,
        task_run_id: str,
        *,
        trace: FlowExecutionTrace,
        node_attempts: tuple[FlowNodeAttemptRecord, ...] = (),
        node_outputs: Mapping[str, object] | None = None,
        selected_outputs: Mapping[str, object] | None = None,
        loop_counters: Mapping[str, int] | None = None,
        pause_tokens: Mapping[str, str] | None = None,
        diagnostics: tuple[FlowDiagnostic, ...] = (),
        artifact_refs: tuple[Mapping[str, object], ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await _ensure_task_run_exists(unit, task_run_id)
            now = self._now()
            record = _flow_execution_record(
                task_run_id=task_run_id,
                revision=1,
                trace=trace,
                created_at=now,
                updated_at=now,
                node_attempts=node_attempts,
                node_outputs=node_outputs,
                selected_outputs=selected_outputs,
                loop_counters=loop_counters,
                pause_tokens=pause_tokens,
                diagnostics=diagnostics,
                artifact_refs=artifact_refs,
                metadata=metadata,
            )
            await unit.cursor.execute(
                _INSERT_FLOW_EXECUTION_SQL,
                _insert_flow_execution_params(record),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                existing = await _fetch_flow_execution_row(
                    unit,
                    task_run_id,
                )
                if existing is not None:
                    raise TaskStoreConflictError(
                        "flow execution already exists"
                    )
                raise TaskStoreConflictError(
                    "flow execution could not be created"
                )
            return _flow_execution_from_row(row)

        return cast(
            FlowExecutionRecord,
            await self._transaction(
                operation="flow_execution_create",
                callback=execute,
            ),
        )

    async def get_flow_execution(
        self,
        task_run_id: str,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            row = await _fetch_flow_execution_row(unit, task_run_id)
            if row is None:
                raise TaskStoreNotFoundError("flow execution was not found")
            return _flow_execution_from_row(row)

        return cast(
            FlowExecutionRecord,
            await self._transaction(
                operation="flow_execution_get",
                callback=execute,
            ),
        )

    async def update_flow_execution(
        self,
        task_run_id: str,
        update: FlowExecutionUpdate,
        *,
        expected_revision: int,
    ) -> FlowExecutionRecord:
        _assert_non_empty_string(task_run_id, "task_run_id")
        assert isinstance(update, FlowExecutionUpdate)
        _assert_positive_int(expected_revision, "expected_revision")

        async def execute(unit: PgsqlUnitOfWork) -> object:
            await unit.cursor.execute(
                _UPDATE_FLOW_EXECUTION_SQL,
                _update_flow_execution_params(
                    update,
                    task_run_id=task_run_id,
                    expected_revision=expected_revision,
                    updated_at=self._now(),
                ),
            )
            row = await unit.cursor.fetchone()
            if row is None:
                existing = await _fetch_flow_execution_row(
                    unit,
                    task_run_id,
                )
                if existing is None:
                    raise TaskStoreNotFoundError(
                        "flow execution was not found"
                    )
                raise TaskStoreConflictError(
                    "flow execution revision did not match"
                )
            return _flow_execution_from_row(row)

        return cast(
            FlowExecutionRecord,
            await self._transaction(
                operation="flow_execution_update",
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

    def _now(self) -> datetime:
        return _ensure_aware_utc(self._clock())


def flow_trace_to_snapshot(trace: FlowExecutionTrace) -> FlowSnapshotMetadata:
    assert isinstance(trace, FlowExecutionTrace)
    return freeze_snapshot_metadata(trace.as_public_dict())


def flow_trace_from_snapshot(value: object) -> FlowExecutionTrace:
    payload = _mapping(value, "trace")
    nodes = tuple(
        _node_trace_from_snapshot(item)
        for item in _sequence(payload.get("nodes", ()), "trace.nodes")
    )
    edges = tuple(
        _edge_trace_from_snapshot(item)
        for item in _sequence(payload.get("edges", ()), "trace.edges")
    )
    return FlowExecutionTrace(nodes=nodes, edges=edges)


def flow_execution_record_from_snapshot(
    value: object,
) -> FlowExecutionRecord:
    payload = _mapping(value, "flow_execution")
    return FlowExecutionRecord(
        task_run_id=_string(payload["task_run_id"], "task_run_id"),
        revision=_int(payload["revision"], "revision"),
        trace=flow_trace_from_snapshot(payload["trace"]),
        node_attempts=tuple(
            flow_node_attempt_from_snapshot(item)
            for item in _sequence(
                payload.get("node_attempts", ()),
                "node_attempts",
            )
        ),
        node_outputs=_snapshot_mapping(
            payload.get("node_outputs", {}),
            "node_outputs",
        ),
        selected_outputs=_snapshot_mapping(
            payload.get("selected_outputs", {}),
            "selected_outputs",
        ),
        loop_counters=_loop_counters_from_snapshot(
            payload.get("loop_counters", {})
        ),
        pause_tokens=_pause_tokens_from_snapshot(
            payload.get("pause_tokens", {})
        ),
        diagnostics=_diagnostics_from_snapshot(payload.get("diagnostics", ())),
        artifact_refs=_snapshot_mappings_from_snapshot(
            payload.get("artifact_refs", ()),
            "artifact_refs",
        ),
        metadata=_snapshot_mapping(payload.get("metadata", {}), "metadata"),
        created_at=_datetime(payload["created_at"]),
        updated_at=_datetime(payload["updated_at"]),
    )


def flow_node_attempt_from_snapshot(
    value: object,
) -> FlowNodeAttemptRecord:
    payload = _mapping(value, "node_attempt")
    return FlowNodeAttemptRecord(
        node=_string(payload["node"], "node"),
        attempt=_int(payload["attempt"], "attempt"),
        state=FlowNodeState(_string(payload["state"], "state")),
        duration_ms=_optional_duration(payload.get("duration_ms")),
        diagnostics=_diagnostics_from_snapshot(payload.get("diagnostics", ())),
        artifact_refs=_snapshot_mappings_from_snapshot(
            payload.get("artifact_refs", ()),
            "artifact_refs",
        ),
        metadata=_snapshot_mapping(payload.get("metadata", {}), "metadata"),
    )


def _flow_execution_record(
    *,
    task_run_id: str,
    revision: int,
    trace: FlowExecutionTrace,
    created_at: datetime,
    updated_at: datetime,
    node_attempts: tuple[FlowNodeAttemptRecord, ...],
    node_outputs: Mapping[str, object] | None,
    selected_outputs: Mapping[str, object] | None,
    loop_counters: Mapping[str, int] | None,
    pause_tokens: Mapping[str, str] | None,
    diagnostics: tuple[FlowDiagnostic, ...],
    artifact_refs: tuple[Mapping[str, object], ...],
    metadata: Mapping[str, object] | None,
) -> FlowExecutionRecord:
    return FlowExecutionRecord(
        task_run_id=task_run_id,
        revision=revision,
        trace=trace,
        created_at=created_at,
        updated_at=updated_at,
        node_attempts=node_attempts,
        node_outputs=freeze_snapshot_metadata(node_outputs),
        selected_outputs=freeze_snapshot_metadata(selected_outputs),
        loop_counters=_freeze_loop_counters(loop_counters or {}),
        pause_tokens=_freeze_pause_tokens(pause_tokens or {}),
        diagnostics=diagnostics,
        artifact_refs=_freeze_snapshot_mappings(
            artifact_refs, "artifact_refs"
        ),
        metadata=freeze_snapshot_metadata(metadata),
    )


def _updated_record(
    current: FlowExecutionRecord,
    update: FlowExecutionUpdate,
    *,
    updated_at: datetime,
) -> FlowExecutionRecord:
    return replace(
        current,
        revision=current.revision + 1,
        trace=current.trace if update.trace is None else update.trace,
        node_attempts=(
            current.node_attempts
            if update.node_attempts is None
            else update.node_attempts
        ),
        node_outputs=(
            current.node_outputs
            if update.node_outputs is None
            else freeze_snapshot_metadata(update.node_outputs)
        ),
        selected_outputs=(
            current.selected_outputs
            if update.selected_outputs is None
            else freeze_snapshot_metadata(update.selected_outputs)
        ),
        loop_counters=(
            current.loop_counters
            if update.loop_counters is None
            else _freeze_loop_counters(update.loop_counters)
        ),
        pause_tokens=(
            current.pause_tokens
            if update.pause_tokens is None
            else _freeze_pause_tokens(update.pause_tokens)
        ),
        diagnostics=(
            current.diagnostics
            if update.diagnostics is None
            else update.diagnostics
        ),
        artifact_refs=(
            current.artifact_refs
            if update.artifact_refs is None
            else _freeze_snapshot_mappings(
                update.artifact_refs,
                "artifact_refs",
            )
        ),
        metadata=(
            current.metadata
            if update.metadata is None
            else freeze_snapshot_metadata(update.metadata)
        ),
        updated_at=updated_at,
    )


def _node_trace_from_snapshot(value: object) -> FlowNodeTrace:
    payload = _mapping(value, "node_trace")
    return FlowNodeTrace(
        node=_string(payload["node"], "node"),
        state=FlowNodeState(_string(payload["state"], "state")),
        attempts=_int(payload["attempts"], "attempts"),
        duration_ms=_optional_duration(payload.get("duration_ms")),
        diagnostics=_diagnostics_from_snapshot(payload.get("diagnostics", ())),
    )


def _edge_trace_from_snapshot(value: object) -> FlowEdgeTrace:
    payload = _mapping(value, "edge_trace")
    return FlowEdgeTrace(
        index=_int(payload["index"], "index"),
        source=_string(payload["source"], "source"),
        target=_string(payload["target"], "target"),
        state=FlowEdgeState(_string(payload["state"], "state")),
        duration_ms=_optional_duration(payload.get("duration_ms")),
        diagnostics=_diagnostics_from_snapshot(payload.get("diagnostics", ())),
    )


def _diagnostics_to_snapshot(
    diagnostics: tuple[FlowDiagnostic, ...],
) -> tuple[FlowSnapshotMetadata, ...]:
    _assert_diagnostics(diagnostics)
    return tuple(
        freeze_snapshot_metadata(diagnostic.as_public_dict())
        for diagnostic in diagnostics
    )


def _diagnostics_from_snapshot(
    value: object,
) -> tuple[FlowDiagnostic, ...]:
    return tuple(
        _diagnostic_from_snapshot(item)
        for item in _sequence(value, "diagnostics")
    )


def _diagnostic_from_snapshot(value: object) -> FlowDiagnostic:
    payload = _mapping(value, "diagnostic")
    source_span_value = payload.get("source_span")
    related_spans = tuple(
        _source_span_from_snapshot(item)
        for item in _sequence(payload.get("related_spans", ()), "related")
    )
    return FlowDiagnostic(
        code=_string(payload["code"], "code"),
        category=FlowDiagnosticCategory(
            _string(payload["category"], "category")
        ),
        severity=FlowDiagnosticSeverity(
            _string(payload["severity"], "severity")
        ),
        message=_string(payload["message"], "message"),
        path=(
            _string(payload["path"], "path")
            if payload.get("path") is not None
            else None
        ),
        source_span=(
            _source_span_from_snapshot(source_span_value)
            if source_span_value is not None
            else None
        ),
        hint=(
            _string(payload["hint"], "hint")
            if payload.get("hint") is not None
            else None
        ),
        related_spans=related_spans,
    )


def _source_span_from_snapshot(value: object) -> FlowSourceSpan:
    payload = _mapping(value, "source_span")
    return FlowSourceSpan(
        start_line=_int(payload["start_line"], "start_line"),
        start_column=_int(payload["start_column"], "start_column"),
        end_line=(
            _int(payload["end_line"], "end_line")
            if payload.get("end_line") is not None
            else None
        ),
        end_column=(
            _int(payload["end_column"], "end_column")
            if payload.get("end_column") is not None
            else None
        ),
    )


def _insert_flow_execution_params(
    record: FlowExecutionRecord,
) -> tuple[object, ...]:
    return (
        record.task_run_id,
        record.revision,
        _json(flow_trace_to_snapshot(record.trace)),
        _json(
            tuple(attempt.as_snapshot() for attempt in record.node_attempts)
        ),
        _json(record.node_outputs),
        _json(record.selected_outputs),
        _json(record.loop_counters),
        _json(record.pause_tokens),
        _json(_diagnostics_to_snapshot(record.diagnostics)),
        _json(record.artifact_refs),
        _json(record.metadata),
        record.created_at,
        record.updated_at,
    )


def _update_flow_execution_params(
    update: FlowExecutionUpdate,
    *,
    task_run_id: str,
    expected_revision: int,
    updated_at: datetime,
) -> tuple[object, ...]:
    return (
        (
            _json(flow_trace_to_snapshot(update.trace))
            if update.trace is not None
            else None
        ),
        (
            _json(
                tuple(
                    attempt.as_snapshot() for attempt in update.node_attempts
                )
            )
            if update.node_attempts is not None
            else None
        ),
        (
            _json(freeze_snapshot_metadata(update.selected_outputs))
            if update.selected_outputs is not None
            else None
        ),
        (
            _json(freeze_snapshot_metadata(update.node_outputs))
            if update.node_outputs is not None
            else None
        ),
        (
            _json(_freeze_loop_counters(update.loop_counters))
            if update.loop_counters is not None
            else None
        ),
        (
            _json(_freeze_pause_tokens(update.pause_tokens))
            if update.pause_tokens is not None
            else None
        ),
        (
            _json(_diagnostics_to_snapshot(update.diagnostics))
            if update.diagnostics is not None
            else None
        ),
        (
            _json(
                _freeze_snapshot_mappings(
                    update.artifact_refs,
                    "artifact_refs",
                )
            )
            if update.artifact_refs is not None
            else None
        ),
        (
            _json(freeze_snapshot_metadata(update.metadata))
            if update.metadata is not None
            else None
        ),
        updated_at,
        task_run_id,
        expected_revision,
    )


async def _ensure_task_run_exists(
    unit: PgsqlUnitOfWork,
    task_run_id: str,
) -> None:
    await unit.cursor.execute(_SELECT_TASK_RUN_SQL, (task_run_id,))
    if await unit.cursor.fetchone() is None:
        raise TaskStoreNotFoundError("task run was not found")


async def _fetch_flow_execution_row(
    unit: PgsqlUnitOfWork,
    task_run_id: str,
) -> Mapping[str, object] | None:
    await unit.cursor.execute(_SELECT_FLOW_EXECUTION_SQL, (task_run_id,))
    return await unit.cursor.fetchone()


def _flow_execution_from_row(row: Mapping[str, object]) -> FlowExecutionRecord:
    return FlowExecutionRecord(
        task_run_id=_string(row["task_run_id"], "task_run_id"),
        revision=_int(row["revision"], "revision"),
        trace=flow_trace_from_snapshot(row["trace"]),
        node_attempts=tuple(
            flow_node_attempt_from_snapshot(item)
            for item in _sequence(row["node_attempts"], "node_attempts")
        ),
        node_outputs=_snapshot_mapping(
            row.get("node_outputs", {}),
            "node_outputs",
        ),
        selected_outputs=_snapshot_mapping(
            row["selected_outputs"],
            "selected_outputs",
        ),
        loop_counters=_loop_counters_from_snapshot(row["loop_counters"]),
        pause_tokens=_pause_tokens_from_snapshot(row["pause_tokens"]),
        diagnostics=_diagnostics_from_snapshot(row["diagnostics"]),
        artifact_refs=_snapshot_mappings_from_snapshot(
            row["artifact_refs"],
            "artifact_refs",
        ),
        metadata=_snapshot_mapping(row["metadata"], "metadata"),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _freeze_loop_counters(value: Mapping[str, int]) -> Mapping[str, int]:
    assert isinstance(value, Mapping), "loop_counters must be a mapping"
    counters: dict[str, int] = {}
    for key, counter in value.items():
        _assert_non_empty_string(key, "loop counter key")
        _assert_non_negative_int(counter, "loop counter")
        counters[key] = counter
    return MappingProxyType(counters)


def _freeze_pause_tokens(value: Mapping[str, str]) -> Mapping[str, str]:
    assert isinstance(value, Mapping), "pause_tokens must be a mapping"
    tokens: dict[str, str] = {}
    for key, token in value.items():
        _assert_non_empty_string(key, "pause token key")
        _assert_non_empty_string(token, "pause token")
        tokens[key] = token
    return MappingProxyType(tokens)


def _freeze_snapshot_mappings(
    value: tuple[Mapping[str, object], ...],
    field_name: str,
) -> tuple[FlowSnapshotMetadata, ...]:
    assert isinstance(value, tuple), f"{field_name} must be a tuple"
    return tuple(freeze_snapshot_metadata(item) for item in value)


def _snapshot_mapping(value: object, field_name: str) -> FlowSnapshotMetadata:
    return freeze_snapshot_metadata(_mapping(value, field_name))


def _snapshot_mappings_from_snapshot(
    value: object,
    field_name: str,
) -> tuple[FlowSnapshotMetadata, ...]:
    return tuple(
        _snapshot_mapping(item, field_name)
        for item in _sequence(value, field_name)
    )


def _loop_counters_from_snapshot(value: object) -> Mapping[str, int]:
    return _freeze_loop_counters(
        {
            key: _int(item, "loop counter")
            for key, item in _mapping(value, "loop_counters").items()
        }
    )


def _pause_tokens_from_snapshot(value: object) -> Mapping[str, str]:
    return _freeze_pause_tokens(
        {
            key: _string(item, "pause token")
            for key, item in _mapping(value, "pause_tokens").items()
        }
    )


def _assert_diagnostics(value: tuple[FlowDiagnostic, ...]) -> None:
    assert isinstance(value, tuple), "diagnostics must be a tuple"
    for diagnostic in value:
        assert isinstance(diagnostic, FlowDiagnostic)


def _assert_duration(value: int | float | None, field_name: str) -> None:
    if value is None:
        return
    assert isinstance(value, int | float) and not isinstance(
        value,
        bool,
    ), f"{field_name} must be numeric"
    assert value >= 0, f"{field_name} must be non-negative"


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"


def _assert_non_negative_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"


def _assert_positive_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _optional_duration(value: object) -> int | float | None:
    if value is None:
        return None
    assert isinstance(value, int | float) and not isinstance(
        value,
        bool,
    ), "duration must be numeric"
    return value


def _string(value: object, field_name: str) -> str:
    _assert_non_empty_string(value, field_name)
    return cast(str, value)


def _int(value: object, field_name: str) -> int:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if isinstance(value, str | bytes | bytearray):
        loaded = loads(value)
        assert isinstance(loaded, Mapping), f"{field_name} must be a mapping"
        return loaded
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    return value


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    assert isinstance(value, list | tuple), f"{field_name} must be a sequence"
    return tuple(value)


def _json(value: object) -> str:
    return dumps(
        _plain(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_aware_utc(value: datetime) -> datetime:
    assert isinstance(value, datetime), "datetime value must be a datetime"
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


_SELECT_TASK_RUN_SQL = """
SELECT "run_id" FROM "task_runs" WHERE "run_id" = %s
"""
_INSERT_FLOW_EXECUTION_SQL = """
INSERT INTO "task_flow_executions" (
    "task_run_id", "revision", "trace", "node_attempts",
    "node_outputs", "selected_outputs", "loop_counters", "pause_tokens",
    "diagnostics", "artifact_refs", "metadata", "created_at", "updated_at"
) VALUES (
    %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s
)
ON CONFLICT ("task_run_id") DO NOTHING
RETURNING *
"""
_SELECT_FLOW_EXECUTION_SQL = """
SELECT * FROM "task_flow_executions" WHERE "task_run_id" = %s
"""
_UPDATE_FLOW_EXECUTION_SQL = """
UPDATE "task_flow_executions"
SET "revision" = "revision" + 1,
    "trace" = COALESCE(%s::jsonb, "trace"),
    "node_attempts" = COALESCE(%s::jsonb, "node_attempts"),
    "selected_outputs" = COALESCE(%s::jsonb, "selected_outputs"),
    "node_outputs" = COALESCE(%s::jsonb, "node_outputs"),
    "loop_counters" = COALESCE(%s::jsonb, "loop_counters"),
    "pause_tokens" = COALESCE(%s::jsonb, "pause_tokens"),
    "diagnostics" = COALESCE(%s::jsonb, "diagnostics"),
    "artifact_refs" = COALESCE(%s::jsonb, "artifact_refs"),
    "metadata" = COALESCE(%s::jsonb, "metadata"),
    "updated_at" = %s
WHERE "task_run_id" = %s
  AND "revision" = %s
RETURNING *
"""
