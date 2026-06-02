from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import ArtifactStore, TaskArtifactPurpose, TaskArtifactState
from .attempt import TaskAttemptPolicy
from .context import TaskInputFile, TaskTargetContext
from .definition import ObservabilitySinkType, TaskDefinition
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
)
from .queue import (
    TaskQueue,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueRetry,
)
from .runner import (
    _error_summary_with_attempt_policy,
    _output_artifact_retention,
    _output_artifacts_from_output,
    _output_summary_value,
    _snapshot_value,
    _task_error_with_attempt_counts,
)
from .state import TaskAttemptState, TaskRunState
from .store import (
    TaskAttempt,
    TaskExecutionResult,
    TaskRun,
    TaskStore,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import UsageRecord, usage_observation_from_response
from .validation import TaskValidationError, validate_task_output

from asyncio import CancelledError, wait_for
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4


class TaskWorkerError(RuntimeError):
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
    output: object = None

    @property
    def processed(self) -> bool:
        return self.claimed is not None


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
        execution_roots: Iterable[str | Path] = (),
        metrics_event_observer: TaskSanitizedEventObserver | None = None,
        trace_event_observer: TaskSanitizedEventObserver | None = None,
        observability_sink: ObservabilitySink | None = None,
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
        self._execution_roots = tuple(execution_roots)
        self._metrics_event_observer = metrics_event_observer
        self._trace_event_observer = trace_event_observer
        self._observability_sink = observability_sink
        self._clock = clock or _utc_now

    async def process_once(self) -> TaskWorkerProcessResult:
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
        run, attempt = await self._start_claimed_attempt(claim)
        try:
            await self._validate_target(definition)
            output = await self._execute(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            )
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except BaseException as error:
            retry = await self._finalize_failure(
                definition,
                claim=claim,
                attempt=attempt,
                sanitizer=sanitizer,
                error=error,
            )
            return TaskWorkerProcessResult(claimed=claim, retry=retry)
        completion = await self._complete_success(
            definition,
            claim=claim,
            attempt=attempt,
            sanitizer=sanitizer,
            output=output,
        )
        return TaskWorkerProcessResult(
            claimed=claim,
            completion=completion,
            output=output,
        )

    async def _execute(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> object:
        context = TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            input_value=run.request.input_summary,
            files=await self._input_files(run.run_id),
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
        output = await wait_for(
            self._target.run(context),
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
        )
        await self._check_cancelled(run.run_id)
        return output

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

    async def _input_files(self, run_id: str) -> tuple[TaskInputFile, ...]:
        records = await self._store.list_artifacts(
            run_id,
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

    async def _record_output_artifacts(
        self,
        definition: TaskDefinition,
        output: object,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> None:
        for artifact in _output_artifacts_from_output(definition, output):
            await self._store.append_artifact(
                run.run_id,
                ref=artifact.ref,
                purpose=TaskArtifactPurpose.OUTPUT,
                state=artifact.state,
                attempt_id=attempt.attempt_id,
                provenance=artifact.provenance,
                retention=_output_artifact_retention(definition, artifact),
                metadata=artifact.metadata,
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
