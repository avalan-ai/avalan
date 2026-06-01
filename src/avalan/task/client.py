from ..types import (
    assert_optional_non_negative_int,
    assert_optional_non_negative_number,
    assert_positive_number,
)
from .artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRetention,
)
from .canonical import spec_hash
from .context import TaskInputFile
from .converters import FileConverter
from .definition import RunMode, TaskDefinition
from .event import SanitizedTaskEvent
from .idempotency import task_idempotency_identity
from .materialization import (
    TaskMaterializedFile,
    materialize_task_input_files,
)
from .observability import ObservabilitySink, TaskSanitizedEventObserver
from .privacy import (
    EncryptionProvider,
    HmacProvider,
    PrivacyField,
    PrivacySanitizer,
)
from .queue import TaskQueue, TaskQueueArtifact, TaskQueueSubmission
from .runner import (
    DirectTaskRunner,
    TaskDirectTarget,
    TaskRunResult,
    _input_summary_value,
    _snapshot_value,
)
from .state import TASK_RUN_TERMINAL_STATES, TaskRunState
from .store import (
    TaskAttempt,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotValue,
    TaskStore,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import UsageRecord, UsageTotals
from .validation import (
    TaskValidationError,
    TaskValidationIssue,
    validate_task_definition,
    validate_task_input,
)

from asyncio import sleep as asyncio_sleep
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast


class TaskClientUnsupportedOperationError(RuntimeError):
    code: str
    operation: str

    def __init__(self, *, code: str, operation: str, message: str) -> None:
        self.code = code
        self.operation = operation
        super().__init__(message)


class TaskClientWaitTimeoutError(TimeoutError):
    code: str
    operation: str
    run_id: str

    def __init__(self, *, run_id: str) -> None:
        self.code = "task.wait_timeout"
        self.operation = "wait"
        self.run_id = run_id
        super().__init__("Task run did not reach a terminal state in time.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientValidationResult:
    issues: tuple[TaskValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "valid": self.valid,
                "issues": tuple(issue.as_dict() for issue in self.issues),
            }
        )

    def raise_for_issues(self) -> None:
        if self.issues:
            raise TaskValidationError(self.issues)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientOutput:
    run_id: str
    state: TaskRunState
    output_summary: TaskSnapshotValue = None
    error: TaskSnapshotValue = None

    @property
    def ready(self) -> bool:
        return self.state == TaskRunState.SUCCEEDED

    def as_dict(self) -> TaskSnapshotValue:
        value: dict[str, object] = {
            "run_id": self.run_id,
            "state": self.state.value,
            "ready": self.ready,
        }
        if self.output_summary is not None:
            value["output_summary"] = self.output_summary
        if self.error is not None:
            value["error"] = self.error
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientInspection:
    run: TaskRun
    attempts: tuple[TaskAttempt, ...]
    output: TaskClientOutput
    events: tuple[SanitizedTaskEvent, ...]
    usage: tuple[UsageRecord, ...]
    usage_totals: UsageTotals
    artifacts: tuple[TaskSnapshotValue, ...]

    def as_dict(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "run": _run_inspection_value(self.run),
                "attempts": tuple(
                    _attempt_inspection_value(attempt)
                    for attempt in self.attempts
                ),
                "output": self.output.as_dict(),
                "events": tuple(
                    _event_inspection_value(event) for event in self.events
                ),
                "usage": tuple(
                    _usage_inspection_value(record) for record in self.usage
                ),
                "usage_totals": _usage_totals_value(self.usage_totals),
                "artifacts": self.artifacts,
            }
        )


class TaskClient:
    def __init__(
        self,
        store: TaskStore,
        *,
        target: TaskDirectTarget | TaskTargetRunner,
        queue: TaskQueue | None = None,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        artifact_store: ArtifactStore | None = None,
        file_converters: Mapping[str, FileConverter] | None = None,
        definition_hash: Callable[[TaskDefinition], str] | None = None,
        execution_roots: Iterable[str | Path] = (),
        metrics_event_observer: TaskSanitizedEventObserver | None = None,
        trace_event_observer: TaskSanitizedEventObserver | None = None,
        observability_sink: ObservabilitySink | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._target = _target_runner(target)
        self._queue = queue
        self._hmac_provider = hmac_provider
        self._encryption_provider = encryption_provider
        self._raw_storage_allowed = raw_storage_allowed
        self._artifact_store = artifact_store
        self._file_converters = file_converters
        self._definition_hash = definition_hash
        self._execution_roots = tuple(execution_roots)
        self._metrics_event_observer = metrics_event_observer
        self._trace_event_observer = trace_event_observer
        self._observability_sink = observability_sink
        self._clock = clock or _utc_now
        self._sleep = sleep or asyncio_sleep

    async def validate(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
    ) -> TaskClientValidationResult:
        assert isinstance(definition, TaskDefinition)
        issues: list[TaskValidationIssue] = []
        issues.extend(
            validate_task_definition(
                definition,
                hmac_provider=self._hmac_provider,
                encryption_provider=self._encryption_provider,
                require_configured_keys=True,
                raw_storage_allowed=self._raw_storage_allowed,
                execution_roots=self._execution_roots,
            )
        )
        issues.extend(validate_task_input(definition, input_value))
        issues.extend(
            await self._target.validate_definition(
                definition,
                TaskValidationContext(
                    execution_roots=tuple(
                        Path(root) for root in self._execution_roots
                    ),
                ),
            )
        )
        return TaskClientValidationResult(issues=tuple(issues))

    async def run(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult:
        assert isinstance(definition, TaskDefinition)
        if definition.run.mode != RunMode.DIRECT:
            raise _unsupported_queue_operation("run")
        return await self._runner().run(
            definition,
            input_value=input_value,
            files=files,
            idempotency_key=idempotency_key,
            metadata=metadata,
            expires_at=expires_at,
        )

    async def enqueue(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        available_at: datetime | None = None,
        idempotency_expires_at: datetime | None = None,
        idempotency_window: object = None,
        owner_scope: object = "default",
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(files, tuple)
        if available_at is not None:
            assert isinstance(available_at, datetime)
        if idempotency_expires_at is not None:
            assert isinstance(idempotency_expires_at, datetime)
        if definition.run.mode != RunMode.QUEUE:
            raise _unsupported_queue_operation("enqueue")
        if self._queue is None:
            raise _unsupported_queue_operation("enqueue")
        validation = await self.validate(definition, input_value=input_value)
        validation.raise_for_issues()
        sanitizer = self._sanitizer(definition)
        definition_id = self._definition_hash_value(definition)
        await self._store.register_definition(
            definition,
            definition_hash=definition_id,
        )
        materialized_files = await materialize_task_input_files(
            definition,
            input_value,
            roots=self._execution_roots,
            artifact_store=self._artifact_store,
            hmac_provider=self._hmac_provider,
        )
        input_files = tuple(
            materialized_file.as_input_file()
            for materialized_file in materialized_files
        )
        queued_files = (*files, *input_files)
        input_summary_value = _input_summary_value(definition, input_value)
        idempotency = task_idempotency_identity(
            definition,
            definition_hash=definition_id,
            input_value=input_summary_value,
            files=queued_files,
            owner_scope=owner_scope,
            hmac_provider=cast(HmacProvider, self._hmac_provider),
            window=idempotency_key or idempotency_window,
        )
        return await self._queue.enqueue_run(
            TaskExecutionRequest(
                definition_id=definition_id,
                input_summary=_snapshot_value(
                    sanitizer.sanitize(
                        PrivacyField.INPUT,
                        input_summary_value,
                    )
                ),
                file_summaries=self._file_summaries(
                    sanitizer,
                    queued_files,
                ),
                idempotency_key=(
                    idempotency.identity_key if idempotency else None
                ),
                queue=definition.run.queue,
                metadata=freeze_snapshot_metadata(metadata),
            ),
            queue_name=definition.run.queue or "",
            priority=definition.run.priority or 0,
            available_at=available_at,
            idempotency=idempotency,
            idempotency_expires_at=idempotency_expires_at,
            artifacts=self._queue_artifacts(definition, materialized_files),
            run_metadata={"runner": "queue"},
            queue_metadata=queue_metadata,
        )

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> TaskClientOutput:
        assert isinstance(run_id, str) and run_id.strip()
        assert_optional_non_negative_number(
            timeout_seconds,
            "timeout_seconds",
        )
        assert_positive_number(
            poll_interval_seconds,
            "poll_interval_seconds",
        )
        deadline = (
            self._clock() + timedelta(seconds=timeout_seconds)
            if timeout_seconds is not None
            else None
        )
        while True:
            output = await self.output(run_id)
            if output.state in TASK_RUN_TERMINAL_STATES:
                return output
            if deadline is not None:
                now = self._clock()
                remaining = (deadline - now).total_seconds()
                if remaining <= 0:
                    raise TaskClientWaitTimeoutError(run_id=run_id)
                await self._sleep(min(poll_interval_seconds, remaining))
            else:
                await self._sleep(poll_interval_seconds)

    async def cancel(self, run_id: str) -> TaskRun:
        assert isinstance(run_id, str) and run_id.strip()
        run = await self._store.get_run(run_id)
        if run.state in TASK_RUN_TERMINAL_STATES:
            return run
        if run.state == TaskRunState.CANCEL_REQUESTED:
            return run
        if run.state not in {
            TaskRunState.QUEUED,
            TaskRunState.RUNNING,
        }:
            raise _unsupported_cancel_operation()
        return await self._store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel_requested",
        )

    async def output(self, run_id: str) -> TaskClientOutput:
        run = await self._store.get_run(run_id)
        result = run.result
        return TaskClientOutput(
            run_id=run.run_id,
            state=run.state,
            output_summary=result.output_summary if result else None,
            error=result.error if result else None,
        )

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        assert_optional_non_negative_int(after_sequence, "after_sequence")
        return await self._store.list_events(
            run_id,
            attempt_id=attempt_id,
            after_sequence=after_sequence,
        )

    async def usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]:
        return await self._store.list_usage(run_id, attempt_id=attempt_id)

    async def usage_totals(self, run_id: str) -> UsageTotals:
        return await self._store.usage_totals(run_id)

    async def artifacts(self, run_id: str) -> tuple[TaskSnapshotValue, ...]:
        artifacts = await self._store.list_artifacts(run_id)
        if artifacts:
            return tuple(artifact.summary() for artifact in artifacts)
        run = await self._store.get_run(run_id)
        result = run.result
        if result is None:
            return ()
        references = result.metadata.get("artifacts")
        if references is None:
            return ()
        if isinstance(references, tuple):
            return references
        return (references,)

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> TaskClientInspection:
        return TaskClientInspection(
            run=await self._store.get_run(run_id),
            attempts=await self._store.list_attempts(run_id),
            output=await self.output(run_id),
            events=await self.events(run_id, after_sequence=after_sequence),
            usage=await self.usage(run_id),
            usage_totals=await self.usage_totals(run_id),
            artifacts=await self.artifacts(run_id),
        )

    def _runner(self) -> DirectTaskRunner:
        return DirectTaskRunner(
            self._store,
            target=self._target,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
            artifact_store=self._artifact_store,
            file_converters=self._file_converters,
            definition_hash=self._definition_hash,
            execution_roots=self._execution_roots,
            metrics_event_observer=self._metrics_event_observer,
            trace_event_observer=self._trace_event_observer,
            observability_sink=self._observability_sink,
            clock=self._clock,
            sleep=self._sleep,
        )

    def _definition_hash_value(self, definition: TaskDefinition) -> str:
        return (
            self._definition_hash(definition)
            if self._definition_hash is not None
            else _default_definition_hash(definition)
        )

    def _sanitizer(self, definition: TaskDefinition) -> PrivacySanitizer:
        return PrivacySanitizer(
            definition.privacy,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
        )

    def _file_summaries(
        self,
        sanitizer: PrivacySanitizer,
        files: tuple[TaskInputFile, ...],
    ) -> tuple[TaskSnapshotValue, ...]:
        return tuple(
            _snapshot_value(
                sanitizer.sanitize(PrivacyField.FILES, file.summary())
            )
            for file in files
        )

    def _queue_artifacts(
        self,
        definition: TaskDefinition,
        files: tuple[TaskMaterializedFile, ...],
    ) -> tuple[TaskQueueArtifact, ...]:
        return tuple(
            TaskQueueArtifact(
                ref=file.ref,
                purpose=TaskArtifactPurpose.INPUT,
                provenance=TaskArtifactProvenance(
                    operation="materialization",
                    metadata={
                        "identity": file.identity,
                        "source_kind": file.descriptor.source_kind.value,
                    },
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
                metadata={"identity": file.identity},
            )
            for file in files
        )


def _unsupported_queue_operation(
    operation: str,
) -> TaskClientUnsupportedOperationError:
    return TaskClientUnsupportedOperationError(
        code="task.queue_unsupported",
        operation=operation,
        message=(
            "Queued task operations are not available in this SDK client yet."
        ),
    )


def _unsupported_cancel_operation() -> TaskClientUnsupportedOperationError:
    return TaskClientUnsupportedOperationError(
        code="task.cancel_unavailable",
        operation="cancel",
        message="Task run cannot be cancelled from its current state.",
    )


def _default_definition_hash(definition: TaskDefinition) -> str:
    return spec_hash(definition)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _target_runner(
    target: TaskDirectTarget | TaskTargetRunner,
) -> TaskTargetRunner:
    run = getattr(target, "run", None)
    validate_definition = getattr(target, "validate_definition", None)
    if callable(run) and callable(validate_definition):
        return cast(TaskTargetRunner, target)
    return CallableTaskTargetRunner(cast(TaskDirectTarget, target))


def _run_inspection_value(run: TaskRun) -> dict[str, object]:
    value: dict[str, object] = {
        "run_id": run.run_id,
        "definition_id": run.definition_id,
        "state": run.state.value,
        "created_at": _datetime_value(run.created_at),
        "updated_at": _datetime_value(run.updated_at),
    }
    if run.request.input_summary is not None:
        value["input_summary"] = run.request.input_summary
    if run.request.file_summaries:
        value["file_summaries"] = run.request.file_summaries
    if run.request.queue is not None:
        value["queue"] = run.request.queue
    if run.last_attempt_id is not None:
        value["last_attempt_id"] = run.last_attempt_id
    if run.claim is not None:
        value["claim"] = {
            "worker_id": run.claim.worker_id,
            "claimed_at": _datetime_value(run.claim.claimed_at),
            "lease_expires_at": _datetime_value(run.claim.lease_expires_at),
            "heartbeat_at": (
                _datetime_value(run.claim.heartbeat_at)
                if run.claim.heartbeat_at is not None
                else None
            ),
        }
    if run.result is not None:
        value["result"] = _result_inspection_value(run.result)
    return value


def _attempt_inspection_value(attempt: TaskAttempt) -> dict[str, object]:
    value: dict[str, object] = {
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "attempt_number": attempt.attempt_number,
        "state": attempt.state.value,
        "created_at": _datetime_value(attempt.created_at),
        "updated_at": _datetime_value(attempt.updated_at),
    }
    if attempt.result is not None:
        value["result"] = _result_inspection_value(attempt.result)
    return value


def _result_inspection_value(result: TaskExecutionResult) -> dict[str, object]:
    value: dict[str, object] = {}
    if result.output_summary is not None:
        value["output_summary"] = result.output_summary
    if result.error is not None:
        value["error"] = result.error
    return value


def _event_inspection_value(event: SanitizedTaskEvent) -> dict[str, object]:
    value: dict[str, object] = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "category": event.category.value,
        "created_at": _datetime_value(event.created_at),
    }
    if event.attempt_id is not None:
        value["attempt_id"] = event.attempt_id
    if event.payload is not None:
        value["payload"] = event.payload
    return value


def _usage_inspection_value(record: UsageRecord) -> dict[str, object]:
    value: dict[str, object] = {
        "usage_id": record.usage_id,
        "run_id": record.run_id,
        "sequence": record.sequence,
        "source": record.source.value,
        "totals": _usage_totals_value(record.totals),
        "created_at": _datetime_value(record.created_at),
    }
    if record.attempt_id is not None:
        value["attempt_id"] = record.attempt_id
    if record.metadata:
        value["metadata"] = record.metadata
    return value


def _usage_totals_value(totals: UsageTotals) -> dict[str, object]:
    return {
        "input_tokens": totals.input_tokens,
        "cached_input_tokens": totals.cached_input_tokens,
        "cache_creation_input_tokens": totals.cache_creation_input_tokens,
        "output_tokens": totals.output_tokens,
        "reasoning_tokens": totals.reasoning_tokens,
        "total_tokens": totals.total_tokens,
    }


def _datetime_value(value: datetime) -> str:
    return value.isoformat()
