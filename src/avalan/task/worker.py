from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import ArtifactStore, TaskArtifactPurpose, TaskArtifactState
from .attempt import TaskAttemptPolicy
from .context import TaskInputFile, TaskTargetContext
from .converters import FileConverter
from .converters.registry import default_file_converters
from .definition import ObservabilitySinkType, TaskDefinition, TaskInputType
from .error import TaskError, classify_task_error
from .observability import (
    ObservabilitySink,
    TaskEventPipeline,
    TaskSanitizedEventObserver,
    record_observability_usage,
)
from .privacy import (
    EncryptionProvider,
    HmacProvider,
    PrivacyField,
    PrivacySafeValue,
    PrivacySanitizationError,
    PrivacySanitizer,
    decrypt_encrypted_privacy_value,
)
from .queue import (
    TaskQueue,
    TaskQueueAbandonment,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueRetry,
)
from .runner import (
    _error_summary_with_attempt_policy,
    _output_artifact_retention,
    _output_artifacts_from_output,
    _output_summary_value,
    _sanitize_output_artifact,
    _snapshot_value,
    _task_error_with_attempt_counts,
    task_execution_file_entries_from_value,
    task_input_files_from_materialized,
)
from .state import TaskAttemptState, TaskRunState
from .store import (
    TaskAttempt,
    TaskExecutionResult,
    TaskRun,
    TaskStore,
    TaskStoreConflictError,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import UsageRecord, usage_observation_from_response
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    validate_task_output,
)

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Event,
    TimeoutError,
    create_task,
    sleep,
    wait,
    wait_for,
)
from asyncio import (
    Task as AsyncTask,
)
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4


class TaskWorkerError(RuntimeError):
    pass


class _TaskWorkerShutdownRequested(Exception):
    pass


class TaskQueuedTarget(Protocol):
    def __call__(
        self,
        context: TaskTargetContext,
    ) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskWorkerProcessResult:
    claimed: TaskQueueClaim | None = None
    completion: TaskQueueCompletion | None = None
    retry: TaskQueueRetry | None = None
    abandonment: TaskQueueAbandonment | None = None
    output: object = None
    shutdown_requested: bool = False
    lease_lost: bool = False

    @property
    def processed(self) -> bool:
        return self.claimed is not None


class TaskWorkerShutdown:
    def __init__(self) -> None:
        self._requested = False
        self._event: Event | None = None

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self) -> None:
        self._requested = True
        if self._event is not None:
            self._event.set()

    async def wait(self) -> None:
        if self._requested:
            return
        if self._event is None:
            self._event = Event()
        await self._event.wait()


class TaskWorker:
    def __init__(
        self,
        store: TaskStore,
        queue: TaskQueue,
        *,
        target: TaskQueuedTarget | TaskTargetRunner,
        worker_id: str | None = None,
        queue_name: str = "default",
        lease_seconds: int = 300,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        artifact_store: ArtifactStore | None = None,
        file_converters: Mapping[str, FileConverter] | None = None,
        execution_roots: Iterable[str | Path] = (),
        metrics_event_observer: TaskSanitizedEventObserver | None = None,
        trace_event_observer: TaskSanitizedEventObserver | None = None,
        observability_sink: ObservabilitySink | None = None,
        shutdown: TaskWorkerShutdown | None = None,
        heartbeat_seconds: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        assert hasattr(store, "get_run")
        assert hasattr(queue, "claim")
        self._store = store
        self._queue = queue
        self._target = _target_runner(target)
        self._worker_id = worker_id or _worker_id()
        _assert_non_empty_string(self._worker_id, "worker_id")
        _assert_non_empty_string(queue_name, "queue_name")
        self._queue_name = queue_name
        assert isinstance(lease_seconds, int)
        assert not isinstance(lease_seconds, bool)
        assert lease_seconds > 0
        self._lease_seconds = lease_seconds
        self._hmac_provider = hmac_provider
        self._encryption_provider = encryption_provider
        self._raw_storage_allowed = raw_storage_allowed
        self._artifact_store = artifact_store
        self._file_converters = _file_converters(file_converters)
        self._execution_roots = tuple(execution_roots)
        self._metrics_event_observer = metrics_event_observer
        self._trace_event_observer = trace_event_observer
        self._observability_sink = observability_sink
        if heartbeat_seconds is not None:
            assert isinstance(heartbeat_seconds, int | float)
            assert not isinstance(heartbeat_seconds, bool)
            assert heartbeat_seconds > 0
            assert (
                heartbeat_seconds < lease_seconds
            ), "heartbeat_seconds must be shorter than lease_seconds"
        self._heartbeat_seconds = heartbeat_seconds
        if shutdown is not None:
            assert isinstance(shutdown, TaskWorkerShutdown)
        self._shutdown = shutdown
        self._clock = clock or _utc_now

    async def process_once(self) -> TaskWorkerProcessResult:
        if self._shutdown_requested():
            return TaskWorkerProcessResult(shutdown_requested=True)
        now = self._now()
        claim = await self._queue.claim(
            self._queue_name,
            worker_id=self._worker_id,
            lease_expires_at=now + timedelta(seconds=self._lease_seconds),
            now=now,
            metadata={"worker_id": self._worker_id},
        )
        if claim is None:
            return TaskWorkerProcessResult()
        definition = (
            await self._store.get_definition(claim.run.definition_id)
        ).definition
        sanitizer = self._sanitizer(definition)
        try:
            run, attempt = await self._start_claimed_attempt(claim)
        except TaskStoreConflictError:
            return TaskWorkerProcessResult(claimed=claim, lease_lost=True)
        try:
            await self._validate_target(definition)
            output = await self._execute(
                definition,
                claim=claim,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            )
        except _TaskWorkerShutdownRequested:
            try:
                abandonment = await self._finalize_shutdown(
                    definition,
                    claim=claim,
                )
            except TaskQueueConflictError:
                return TaskWorkerProcessResult(
                    claimed=claim,
                    shutdown_requested=True,
                    lease_lost=True,
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                abandonment=abandonment,
                shutdown_requested=True,
            )
        except TaskQueueConflictError:
            return TaskWorkerProcessResult(claimed=claim, lease_lost=True)
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except BaseException as error:
            try:
                retry = await self._finalize_failure(
                    definition,
                    claim=claim,
                    attempt=attempt,
                    sanitizer=sanitizer,
                    error=error,
                )
            except TaskQueueConflictError:
                return TaskWorkerProcessResult(
                    claimed=claim,
                    lease_lost=True,
                )
            return TaskWorkerProcessResult(claimed=claim, retry=retry)
        try:
            completion = await self._complete_success(
                definition,
                claim=claim,
                attempt=attempt,
                sanitizer=sanitizer,
                output=output,
            )
        except TaskQueueConflictError:
            return TaskWorkerProcessResult(claimed=claim, lease_lost=True)
        return TaskWorkerProcessResult(
            claimed=claim,
            completion=completion,
            output=output,
        )

    async def _execute(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> object:
        context = TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            input_value=self._executable_input_value(definition, run),
            files=await self._input_files(definition, run, attempt),
            metadata=run.request.metadata,
            cancellation_checker=lambda: self._check_cancelled(run.run_id),
            event_listener=self._event_pipeline(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            ),
            usage_observer=(
                (
                    lambda response: self._record_usage(
                        response,
                        definition=definition,
                        run=run,
                        attempt=attempt,
                    )
                )
                if definition.observability.metrics
                else None
            ),
            artifact_store=self._artifact_store,
        )
        await self._check_cancelled(run.run_id)
        output = await self._run_target(
            context,
            claim=claim,
            timeout=definition.run.timeout_seconds,
        )
        await self._check_cancelled(run.run_id)
        issues = validate_task_output(definition, output)
        if issues:
            raise TaskValidationError(issues)
        await self._record_output_artifacts(
            definition,
            output,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
        )
        await self._check_cancelled(run.run_id)
        return output

    def _executable_input_value(
        self,
        definition: TaskDefinition,
        run: TaskRun,
    ) -> object:
        payload = run.request.input_payload
        if payload is None:
            if _queued_input_payload_required(definition, run):
                raise TaskValidationError(
                    (_queue_input_payload_unavailable_issue(),)
                )
            return run.request.input_summary
        if payload.input_value is None:
            if _queued_input_payload_required(definition, run):
                raise TaskValidationError(
                    (_queue_input_payload_unavailable_issue(),)
                )
            return run.request.input_summary
        try:
            return decrypt_encrypted_privacy_value(
                payload.input_value,
                decryption_provider=self._encryption_provider,
            )
        except PrivacySanitizationError as error:
            raise TaskValidationError(
                (_queue_input_payload_unavailable_issue(),)
            ) from error

    async def _run_target(
        self,
        context: TaskTargetContext,
        *,
        claim: TaskQueueClaim,
        timeout: float | None,
    ) -> object:
        if self._shutdown is None and self._heartbeat_seconds is None:
            return await wait_for(self._target.run(context), timeout=timeout)
        target_task = create_task(self._target.run(context))
        heartbeat_task = (
            create_task(self._heartbeat_claim(claim))
            if self._heartbeat_seconds is not None
            else None
        )
        shutdown_task = (
            create_task(self._shutdown.wait())
            if self._shutdown is not None
            else None
        )
        wait_tasks = {target_task}
        if heartbeat_task is not None:
            wait_tasks.add(heartbeat_task)
        if shutdown_task is not None:
            wait_tasks.add(shutdown_task)
        try:
            done, _pending = await wait(
                wait_tasks,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError()  # pragma: no cover
            if heartbeat_task is not None and heartbeat_task in done:
                try:
                    heartbeat_task.result()
                except BaseException:
                    error = TaskQueueConflictError(
                        "task queue heartbeat failed"
                    )
                    raise error from None
                raise _TaskWorkerShutdownRequested()  # pragma: no cover
            if shutdown_task is not None and (
                shutdown_task in done or self._shutdown_requested()
            ):
                raise _TaskWorkerShutdownRequested()
            if target_task in done:
                return await target_task
            raise _TaskWorkerShutdownRequested()  # pragma: no cover
        finally:
            await _cancel_task(target_task)
            if heartbeat_task is not None:
                await _cancel_task(heartbeat_task)
            if shutdown_task is not None:
                await _cancel_task(shutdown_task)

    async def _heartbeat_claim(self, claim: TaskQueueClaim) -> None:
        assert self._heartbeat_seconds is not None
        while True:
            await sleep(self._heartbeat_seconds)
            if self._shutdown_requested():
                return
            now = self._now()
            await self._queue.heartbeat(
                claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                lease_expires_at=(
                    now + timedelta(seconds=self._lease_seconds)
                ),
                now=now,
            )

    async def _start_claimed_attempt(
        self,
        claim: TaskQueueClaim,
    ) -> tuple[TaskRun, TaskAttempt]:
        claim_token = claim.queue_item.claim_token or ""
        run = await self._store.transition_run(
            claim.run.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claim_token,
            metadata={"worker_id": self._worker_id},
        )
        attempt = await self._store.transition_attempt(
            claim.attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claim_token,
            metadata={"worker_id": self._worker_id},
        )
        return run, attempt

    async def _complete_success(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        output: object,
    ) -> TaskQueueCompletion:
        result = TaskExecutionResult(
            output_summary=_snapshot_value(
                sanitizer.sanitize(
                    PrivacyField.OUTPUT,
                    _output_summary_value(definition, output),
                )
            )
        )
        return await self._queue.complete(
            claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            run_state=TaskRunState.SUCCEEDED,
            attempt_state=TaskAttemptState.SUCCEEDED,
            result=result,
            now=self._now(),
            metadata={"worker_id": self._worker_id},
        )

    async def _finalize_failure(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        error: BaseException,
    ) -> TaskQueueRetry | None:
        task_error = classify_task_error(error)
        policy = TaskAttemptPolicy.from_retry_policy(definition.retry)
        decision = policy.decide(
            attempt_number=attempt.attempt_number,
            error=task_error,
        )
        error_summary = self._safe_task_error_summary(
            sanitizer,
            (
                task_error
                if decision.should_retry
                else _task_error_with_attempt_counts(task_error, decision)
            ),
        )
        result = TaskExecutionResult(
            error=_snapshot_value(
                (
                    _error_summary_with_attempt_policy(
                        error_summary,
                        retry_decision=decision,
                    )
                    if decision.should_retry
                    else error_summary
                )
            )
        )
        if decision.should_retry:
            return await self._queue.retry(
                claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                result=result,
                available_at=(
                    self._now()
                    + timedelta(seconds=decision.retry_delay_seconds or 0)
                ),
                max_attempts=policy.max_attempts,
                now=self._now(),
                metadata={"worker_id": self._worker_id},
            )
        run_state = TaskRunState.FAILED
        if isinstance(error, CancelledError):
            run = await self._store.get_run(claim.run.run_id)
            if run.state != TaskRunState.CANCEL_REQUESTED:
                await self._store.transition_run(
                    claim.run.run_id,
                    from_states={TaskRunState.RUNNING},
                    to_state=TaskRunState.CANCEL_REQUESTED,
                    reason="cancel_requested",
                    claim_token=claim.queue_item.claim_token or "",
                    metadata={"worker_id": self._worker_id},
                )
            run_state = TaskRunState.CANCELLED
        await self._queue.complete(
            claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            run_state=run_state,
            attempt_state=TaskAttemptState.FAILED,
            result=result,
            now=self._now(),
            metadata={"worker_id": self._worker_id},
        )
        return None

    async def _finalize_shutdown(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
    ) -> TaskQueueAbandonment:
        policy = TaskAttemptPolicy.from_retry_policy(definition.retry)
        return await self._queue.abandon(
            claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            max_attempts=policy.max_attempts,
            now=self._now(),
            metadata={"worker_id": self._worker_id, "reason": "shutdown"},
        )

    async def _input_files(
        self,
        definition: TaskDefinition,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> tuple[TaskInputFile, ...]:
        payload = run.request.input_payload
        if payload is not None and payload.file_values:
            entries = tuple(
                entry
                for value in payload.file_values
                for entry in task_execution_file_entries_from_value(
                    self._decrypt_file_payload_value(value)
                )
            )
            materialized = tuple(
                entry.materialized_file
                for entry in entries
                if entry.materialized_file is not None
            )
            if not materialized:
                return tuple(entry.file for entry in entries)
            converted = iter(
                await task_input_files_from_materialized(
                    definition,
                    materialized,
                    artifact_store=self._artifact_store,
                    file_converters=self._file_converters,
                    task_store=self._store,
                    run=run,
                    attempt=attempt,
                )
            )
            files: list[TaskInputFile] = []
            for entry in entries:
                files.append(
                    next(converted)
                    if entry.materialized_file is not None
                    else entry.file
                )
            return tuple(files)
        records = await self._store.list_artifacts(
            run.run_id,
            purpose=TaskArtifactPurpose.INPUT,
            state=TaskArtifactState.READY,
        )
        return tuple(
            TaskInputFile(
                logical_path=f"artifact:{record.artifact_id}",
                artifact_ref=record.ref,
                media_type=record.ref.media_type,
                size_bytes=record.ref.size_bytes,
                metadata=record.metadata,
            )
            for record in records
        )

    def _decrypt_file_payload_value(self, value: object) -> object:
        try:
            return decrypt_encrypted_privacy_value(
                value,
                decryption_provider=self._encryption_provider,
            )
        except PrivacySanitizationError as error:
            raise TaskValidationError(
                (
                    TaskValidationIssue(
                        code="queue.file_payload_unavailable",
                        path="request.input_payload.file_values",
                        message=(
                            "Queued task file inputs are unavailable for "
                            "worker execution."
                        ),
                        hint=(
                            "Queue file tasks with encrypted file payload "
                            "storage enabled."
                        ),
                        category=TaskValidationCategory.PRIVACY,
                    ),
                )
            ) from error

    async def _record_output_artifacts(
        self,
        definition: TaskDefinition,
        output: object,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> None:
        for artifact in _output_artifacts_from_output(definition, output):
            safe_artifact = _sanitize_output_artifact(
                artifact,
                sanitizer,
            )
            await self._store.append_artifact(
                run.run_id,
                ref=safe_artifact.ref,
                purpose=TaskArtifactPurpose.OUTPUT,
                state=safe_artifact.state,
                attempt_id=attempt.attempt_id,
                provenance=safe_artifact.provenance,
                retention=_output_artifact_retention(
                    definition,
                    safe_artifact,
                ),
                metadata=safe_artifact.metadata,
            )

    async def _record_usage(
        self,
        response: object,
        *,
        definition: TaskDefinition,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> None:
        observation = usage_observation_from_response(response)
        if observation is None:
            return
        usage_record: UsageRecord | None = None
        try:
            usage_record = await self._store.append_usage(
                run.run_id,
                attempt_id=attempt.attempt_id,
                source=observation.source,
                totals=observation.totals,
            )
        except Exception:
            pass
        await record_observability_usage(
            self._observability_sink_for(definition),
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            source=observation.source,
            totals=observation.totals,
            record=usage_record,
        )

    def _event_pipeline(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> TaskEventPipeline | None:
        metrics_observer = (
            self._metrics_event_observer
            if definition.observability.metrics
            else None
        )
        trace_observer = (
            self._trace_event_observer
            if definition.observability.trace
            else None
        )
        observability_sink = self._observability_sink_for(definition)
        if (
            not definition.observability.capture_events
            and metrics_observer is None
            and trace_observer is None
            and observability_sink is None
        ):
            return None
        return TaskEventPipeline(
            store=self._store,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            sanitizer=sanitizer,
            capture_events=definition.observability.capture_events,
            metrics_observer=metrics_observer,
            trace_observer=trace_observer,
            observability_sink=observability_sink,
        )

    def _observability_sink_for(
        self,
        definition: TaskDefinition,
    ) -> ObservabilitySink | None:
        if definition.observability.sinks == (ObservabilitySinkType.NOOP,):
            return None
        return self._observability_sink

    async def _check_cancelled(self, run_id: str) -> None:
        if self._shutdown_requested():
            raise _TaskWorkerShutdownRequested()
        await self._check_run_cancelled(run_id)

    async def _check_run_cancelled(self, run_id: str) -> None:
        run = await self._store.get_run(run_id)
        if run.state == TaskRunState.CANCEL_REQUESTED:
            raise CancelledError()

    async def _validate_target(self, definition: TaskDefinition) -> None:
        issues = await self._target.validate_definition(
            definition,
            TaskValidationContext(
                execution_roots=tuple(
                    Path(root) for root in self._execution_roots
                ),
            ),
        )
        if issues:
            raise TaskValidationError(issues)

    def _sanitizer(self, definition: TaskDefinition) -> PrivacySanitizer:
        return PrivacySanitizer(
            definition.privacy,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
        )

    def _safe_task_error_summary(
        self,
        sanitizer: PrivacySanitizer,
        error: TaskError,
    ) -> PrivacySafeValue:
        try:
            return sanitizer.sanitize(PrivacyField.ERRORS, error.as_dict())
        except PrivacySanitizationError:
            return {
                "category": error.category.value,
                "code": error.code.value,
                "privacy": "<redacted>",
            }

    def _now(self) -> datetime:
        return self._clock()

    def _shutdown_requested(self) -> bool:
        return self._shutdown is not None and self._shutdown.requested


async def _cancel_task(task: AsyncTask[object]) -> None:
    if task.done():
        return
    task.cancel()
    with suppress(CancelledError):
        await task


def _target_runner(
    target: TaskQueuedTarget | TaskTargetRunner,
) -> TaskTargetRunner:
    run = getattr(target, "run", None)
    validate_definition = getattr(target, "validate_definition", None)
    if callable(run) and callable(validate_definition):
        return cast(TaskTargetRunner, target)
    return CallableTaskTargetRunner(cast(TaskQueuedTarget, target))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _worker_id() -> str:
    return f"worker-{uuid4().hex}"


def _file_converters(
    converters: Mapping[str, FileConverter] | None,
) -> Mapping[str, FileConverter]:
    values: dict[str, FileConverter] = dict(default_file_converters())
    values.update(converters or {})
    return values


def _queued_input_payload_required(
    definition: TaskDefinition,
    run: TaskRun,
) -> bool:
    return (
        run.request.input_summary is not None
        and definition.input.type
        not in {
            TaskInputType.FILE,
            TaskInputType.FILE_ARRAY,
        }
    )


def _queue_input_payload_unavailable_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="queue.input_payload_unavailable",
        path="request.input_payload",
        message="Queued task input is unavailable for worker execution.",
        hint=(
            "Queue scalar and structured task inputs with encrypted payload "
            "storage enabled."
        ),
        category=TaskValidationCategory.PRIVACY,
    )
