from ..agent.continuation import DurableAgentContinuationClaimLease
from ..container import (
    ContainerAsyncBackend,
    ContainerOutputDecisionType,
    ContainerResultStatus,
    run_container_managed_lifecycle,
)
from ..interaction import DurableContinuationResumeState
from ..interaction.error import InputContractError, InputErrorCode
from ..skill import SkillRegistry, TrustedSkillSettings
from ..types import assert_non_empty_string as _assert_non_empty_string
from .artifact import ArtifactStore, TaskArtifactPurpose, TaskArtifactState
from .attempt import TaskAttemptPolicy
from .container import (
    TaskContainerPlans,
    TaskContainerVerificationError,
    task_container_event_payload,
    task_container_input_mount_manifest,
    task_container_lifecycle_run_plan,
    task_container_output_artifacts,
    task_container_output_contract,
    task_container_unsupported_input_mount_path,
    task_container_user_metadata,
    verify_task_container_request,
)
from .context import (
    TaskDurableResumeHandle,
    TaskInputFile,
    TaskTargetContext,
    TaskUsageObservationTracker,
)
from .converters import FileConverter
from .converters.registry import default_file_converters
from .definition import ObservabilitySinkType, TaskDefinition, TaskInputType
from .error import TaskError, classify_task_error
from .event import TaskEventCategory, freeze_task_event_value
from .observability import (
    ObservabilitySink,
    TaskEventPipeline,
    TaskSanitizedEventObserver,
    record_response_usage,
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
    TaskDurableSuspensionCoordinator,
    TaskQueue,
    TaskQueueAbandonment,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueReentry,
    TaskQueueRetry,
    TaskQueueSuspension,
)
from .resume import (
    TaskDurableResumeCoordinator,
    TaskResumeClaimLeaseManager,
)
from .runner import (
    TaskContainerAttemptResult,
    TaskWorkerRuntimeEnvelopeRunner,
    _container_issue,
    _container_validation_issue,
    _error_summary_with_attempt_policy,
    _output_artifact_retention,
    _output_artifacts_from_output,
    _output_summary_value,
    _raise_for_container_backend_selection,
    _sanitize_output_artifact,
    _snapshot_value,
    _task_error_with_attempt_counts,
    is_trusted_task_worker_runtime_envelope_runner,
    task_execution_file_entries_from_value,
    task_input_file_groups_from_materialized,
)
from .settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSuccess,
)
from .skills import (
    TASK_SKILLS_METADATA_KEY,
    revalidate_task_skills_for_worker,
    task_skill_audit_event_publisher,
)
from .state import (
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskRunState,
)
from .store import (
    TaskAttempt,
    TaskAttemptSegment,
    TaskExecutionResult,
    TaskRun,
    TaskStore,
    TaskStoreConflictError,
)
from .target import (
    CallableTaskTargetRunner,
    PreparedTaskDurableResumeTarget,
    TaskDurableResumeTargetPreparer,
    TaskDurableResumeTargetRunner,
    TaskTargetCompleted,
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskTargetSuspended,
    TaskValidationContext,
    completed_task_target_outcome,
    task_target_outcome,
)
from .usage import usage_observations_from_response
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
    Lock,
    TimeoutError,
    create_task,
    current_task,
    gather,
    shield,
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
from typing import NoReturn, Protocol, TypeVar, cast
from uuid import uuid4


class TaskWorkerError(RuntimeError):
    pass


class _TaskDurableResumeRejected(TaskWorkerError):
    pass


class _TaskDurableResumeExpired(_TaskDurableResumeRejected):
    """Carry absolute expiry plus any failed admission cleanup."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.cleanup_error = cleanup_error


class _TaskResumeClaimLeaseBridge(TaskResumeClaimLeaseManager):
    """Bridge one task claim heartbeat into its continuation claim."""

    def __init__(self, claim: TaskQueueClaim) -> None:
        lease_expires_at = claim.queue_item.lease_expires_at
        observed_at = (
            claim.queue_item.heartbeat_at or claim.queue_item.claimed_at
        )
        assert lease_expires_at is not None
        assert observed_at is not None
        self._lease_expires_at = lease_expires_at
        self._observed_at = observed_at
        self._claim_lease: DurableAgentContinuationClaimLease | None = None
        self._lock = Lock()

    async def current_lease_expires_at(self) -> datetime:
        """Return the latest successfully acquired task claim lease."""
        async with self._lock:
            return self._lease_expires_at

    async def bind(
        self,
        claim_lease: DurableAgentContinuationClaimLease,
    ) -> None:
        """Bind and immediately synchronize one continuation claim."""
        if not callable(getattr(claim_lease, "renew", None)):
            raise TypeError(
                "claim_lease must be a durable continuation claim lease"
            )
        async with self._lock:
            if self._claim_lease is not None:
                raise TaskWorkerError(
                    "durable continuation claim lease is already bound"
                )
            self._claim_lease = claim_lease
            await claim_lease.renew(
                self._lease_expires_at,
                now=self._observed_at,
            )

    async def heartbeat(
        self,
        lease_expires_at: datetime,
        *,
        now: datetime,
    ) -> None:
        """Renew a bound continuation after its task heartbeat succeeds."""
        async with self._lock:
            if lease_expires_at < self._lease_expires_at:
                raise TaskWorkerError("task claim lease moved backwards")
            self._lease_expires_at = lease_expires_at
            self._observed_at = now
            if self._claim_lease is not None:
                await self._claim_lease.renew(
                    lease_expires_at,
                    now=now,
                )

    async def unbind(self) -> None:
        """Stop renewing the continuation claim."""
        async with self._lock:
            self._claim_lease = None


class _TaskDurableResumeOwner:
    """Close one admitted durable resume handle exactly once."""

    def __init__(self) -> None:
        self._durable_resume: TaskDurableResumeHandle | None = None
        self._close_task: AsyncTask[None] | None = None
        self._taken = False

    def take(self, durable_resume: TaskDurableResumeHandle | None) -> None:
        """Take ownership of an admitted durable resume handle."""
        if durable_resume is None:
            return
        if self._taken:
            raise TaskWorkerError("durable resume handle is already owned")
        self._taken = True
        self._durable_resume = durable_resume

    async def close(self) -> None:
        """Close the owned durable resume handle exactly once."""
        durable_resume = self._durable_resume
        if durable_resume is None:
            return
        if self._close_task is None:
            self._close_task = create_task(
                _close_durable_resume(durable_resume),
                name="task-durable-resume-handle-close",
            )
        await self._close_task
        self._durable_resume = None


class _TaskClaimCleanupOwner:
    """Own claim unbinding and durable handle closure through cancellation."""

    def __init__(
        self,
        claim_lease_manager: _TaskResumeClaimLeaseBridge,
        resume_owner: _TaskDurableResumeOwner,
        *,
        queue_item_id: str,
    ) -> None:
        self._claim_lease_manager = claim_lease_manager
        self._resume_owner = resume_owner
        self._queue_item_id = queue_item_id
        self._cleanup_task: AsyncTask[None] | None = None

    async def close(self) -> None:
        """Finish exact cleanup despite repeated caller cancellation."""
        if self._cleanup_task is None:
            self._cleanup_task = create_task(
                self._close_owned(),
                name=f"task-claim-cleanup-{self._queue_item_id}",
            )
        await _await_owned_task(self._cleanup_task)

    async def _close_owned(self) -> None:
        errors: list[BaseException] = []
        try:
            await self._claim_lease_manager.unbind()
        except BaseException as error:
            errors.append(error)
        try:
            await self._resume_owner.close()
        except BaseException as error:
            errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "task claim cleanup failed",
                errors,
            )


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
    reentry: TaskQueueReentry | None = None
    suspension: TaskQueueSuspension | None = None
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
        container_backend: ContainerAsyncBackend | None = None,
        worker_runtime_envelope_runner: (
            TaskWorkerRuntimeEnvelopeRunner | None
        ) = None,
        skills_settings: TrustedSkillSettings | None = None,
        skills_registry: SkillRegistry | None = None,
        definition_base: str | Path | None = None,
        shutdown: TaskWorkerShutdown | None = None,
        heartbeat_seconds: float | None = None,
        durable_suspension_coordinator: (
            TaskDurableSuspensionCoordinator | None
        ) = None,
        durable_resume_coordinator: TaskDurableResumeCoordinator | None = None,
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
        if container_backend is not None:
            assert isinstance(container_backend, ContainerAsyncBackend)
        self._container_backend = container_backend
        if worker_runtime_envelope_runner is not None:
            assert callable(worker_runtime_envelope_runner)
            assert is_trusted_task_worker_runtime_envelope_runner(
                worker_runtime_envelope_runner
            ), "worker runtime envelope runner must be trusted"
        self._worker_runtime_envelope_runner = worker_runtime_envelope_runner
        if skills_settings is not None:
            assert isinstance(skills_settings, TrustedSkillSettings)
        if skills_registry is not None:
            assert isinstance(skills_registry, SkillRegistry)
        self._skills_settings = skills_settings
        self._skills_registry = skills_registry
        if definition_base is not None:
            assert isinstance(definition_base, str | Path)
        self._definition_base = definition_base
        if heartbeat_seconds is not None:
            assert isinstance(heartbeat_seconds, int | float)
            assert not isinstance(heartbeat_seconds, bool)
            assert heartbeat_seconds > 0
            assert (
                heartbeat_seconds < lease_seconds
            ), "heartbeat_seconds must be shorter than lease_seconds"
        self._heartbeat_seconds = heartbeat_seconds
        if durable_suspension_coordinator is not None:
            assert callable(
                getattr(
                    durable_suspension_coordinator,
                    "create_and_suspend",
                    None,
                )
            )
        self._durable_suspension_coordinator = durable_suspension_coordinator
        if durable_resume_coordinator is not None:
            assert callable(getattr(durable_resume_coordinator, "admit", None))
        self._durable_resume_coordinator = durable_resume_coordinator
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
        claim_lease_manager = _TaskResumeClaimLeaseBridge(claim)
        heartbeat_seconds = self._claim_heartbeat_seconds(claim)
        heartbeat_task = (
            create_task(
                self._heartbeat_claim(
                    claim,
                    heartbeat_seconds,
                    claim_lease_manager=claim_lease_manager,
                ),
                name=f"task-claim-heartbeat-{claim.queue_item.queue_item_id}",
            )
            if heartbeat_seconds is not None
            else None
        )
        try:
            return await self._process_claim(
                claim,
                heartbeat_task=heartbeat_task,
                claim_lease_manager=claim_lease_manager,
            )
        finally:
            if heartbeat_task is not None:
                heartbeat_cleanup = create_task(
                    _cancel_heartbeat_task(heartbeat_task),
                    name=(
                        "task-claim-heartbeat-cleanup-"
                        f"{claim.queue_item.queue_item_id}"
                    ),
                )
                await _await_owned_task(heartbeat_cleanup)

    async def _process_claim(
        self,
        claim: TaskQueueClaim,
        *,
        heartbeat_task: AsyncTask[None] | None,
        claim_lease_manager: _TaskResumeClaimLeaseBridge,
    ) -> TaskWorkerProcessResult:
        resume_owner = _TaskDurableResumeOwner()
        cleanup_owner = _TaskClaimCleanupOwner(
            claim_lease_manager,
            resume_owner,
            queue_item_id=claim.queue_item.queue_item_id,
        )
        try:
            return await self._process_claim_owned(
                claim,
                heartbeat_task=heartbeat_task,
                claim_lease_manager=claim_lease_manager,
                resume_owner=resume_owner,
            )
        finally:
            await cleanup_owner.close()

    async def _process_claim_owned(
        self,
        claim: TaskQueueClaim,
        *,
        heartbeat_task: AsyncTask[None] | None,
        claim_lease_manager: _TaskResumeClaimLeaseBridge,
        resume_owner: _TaskDurableResumeOwner,
    ) -> TaskWorkerProcessResult:
        definition: TaskDefinition | None = None
        sanitizer: PrivacySanitizer | None = None
        previous_segment: TaskAttemptSegment | None = None
        durable_resume: TaskDurableResumeHandle | None = None
        durable_target: TaskDurableResumeTargetRunner | None = None
        try:
            definition = (
                await self._store.get_definition(claim.run.definition_id)
            ).definition
            sanitizer = self._sanitizer(definition)
            previous_segment = await self._previous_attempt_segment(claim)
            durable_target = self._prepare_durable_resume_target(
                definition,
                previous_segment=previous_segment,
            )
            durable_resume = await self._admit_durable_resume_guarded(
                claim,
                previous_segment,
                heartbeat_task=heartbeat_task,
                claim_lease_manager=claim_lease_manager,
                resume_owner=resume_owner,
            )
            if previous_segment is not None:
                definition = await self._revalidate_skills(
                    definition,
                    run=claim.run,
                    attempt=claim.attempt,
                    sanitizer=sanitizer,
                )
                await self._validate_target(definition)
            self._raise_if_claim_guard_stopped(heartbeat_task)
            if self._shutdown_requested():
                raise _TaskWorkerShutdownRequested()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except BaseException as error:
            if _contains_process_control(error):  # pragma: no cover
                raise
            return await self._converge_claimed_setup_failure(
                claim,
                definition=definition,
                sanitizer=sanitizer,
                previous_segment=previous_segment,
                durable_resume=durable_resume,
                error=error,
            )
        assert definition is not None
        assert sanitizer is not None
        try:
            run, attempt, segment = await self._start_claimed_attempt(
                claim,
                previous_segment=previous_segment,
            )
        except TaskStoreConflictError:
            if durable_resume is not None:
                await durable_resume.release()
            return await self._lease_lost_result(
                claim,
                sanitizer=sanitizer,
            )
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except BaseException as error:
            if _contains_process_control(error):  # pragma: no cover
                raise
            if durable_resume is None:
                raise
            return await self._converge_claimed_setup_failure(
                claim,
                definition=definition,
                sanitizer=sanitizer,
                previous_segment=previous_segment,
                durable_resume=durable_resume,
                error=error,
            )
        try:
            self._raise_if_claim_guard_stopped(heartbeat_task)
            if previous_segment is None:
                definition = await self._revalidate_skills(
                    definition,
                    run=run,
                    attempt=attempt,
                    sanitizer=sanitizer,
                )
                await self._validate_target(definition)
            outcome = await self._execute(
                definition,
                claim=claim,
                run=run,
                attempt=attempt,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=durable_resume,
                durable_target=durable_target,
                heartbeat_task=heartbeat_task,
            )
        except _TaskWorkerShutdownRequested as error:
            if durable_resume is not None:
                try:
                    return await self._converge_durable_failure(
                        claim=claim,
                        segment=segment,
                        previous_segment=previous_segment,
                        sanitizer=sanitizer,
                        durable_resume=durable_resume,
                        error=error,
                    )
                except (TaskQueueConflictError, TaskStoreConflictError):
                    return await self._lease_lost_result(
                        claim,
                        sanitizer=sanitizer,
                        shutdown_requested=True,
                    )
            try:
                abandonment = await self._finalize_shutdown(
                    definition,
                    claim=claim,
                    segment=segment,
                )
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                    shutdown_requested=True,
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                abandonment=abandonment,
                shutdown_requested=True,
            )
        except (TaskQueueConflictError, TaskStoreConflictError) as error:
            if durable_resume is not None:
                try:
                    return await self._converge_durable_failure(
                        claim=claim,
                        segment=segment,
                        previous_segment=previous_segment,
                        sanitizer=sanitizer,
                        durable_resume=durable_resume,
                        error=error,
                    )
                except (TaskQueueConflictError, TaskStoreConflictError):
                    pass
            return await self._lease_lost_result(
                claim,
                sanitizer=sanitizer,
            )
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except _TaskDurableResumeExpired as error:
            return await self._converge_expired_resume(
                claim,
                sanitizer=sanitizer,
                error=error,
            )
        except BaseException as error:
            if _contains_process_control(error):  # pragma: no cover
                raise
            if durable_resume is not None:
                try:
                    return await self._converge_durable_failure(
                        claim=claim,
                        segment=segment,
                        previous_segment=previous_segment,
                        sanitizer=sanitizer,
                        durable_resume=durable_resume,
                        error=error,
                    )
                except (TaskQueueConflictError, TaskStoreConflictError):
                    return await self._lease_lost_result(
                        claim,
                        sanitizer=sanitizer,
                    )
            try:
                retry = await self._finalize_failure(
                    definition,
                    claim=claim,
                    attempt=attempt,
                    segment=segment,
                    sanitizer=sanitizer,
                    error=error,
                )
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            return TaskWorkerProcessResult(claimed=claim, retry=retry)
        outcome = task_target_outcome(outcome)
        if type(outcome) is TaskTargetSuspended:
            try:
                suspension = await self._suspend(
                    claim=claim,
                    segment=segment,
                    outcome=outcome,
                    durable_resume=durable_resume,
                )
            except (TaskQueueConflictError, TaskStoreConflictError) as error:
                if durable_resume is not None:
                    try:
                        return await self._converge_durable_failure(
                            claim=claim,
                            segment=segment,
                            previous_segment=previous_segment,
                            sanitizer=sanitizer,
                            durable_resume=durable_resume,
                            error=error,
                        )
                    except (TaskQueueConflictError, TaskStoreConflictError):
                        pass
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            except (KeyboardInterrupt, SystemExit):  # pragma: no cover
                raise
            except _TaskDurableResumeExpired as error:
                return await self._converge_expired_resume(
                    claim,
                    sanitizer=sanitizer,
                    error=error,
                )
            except BaseException as error:
                if _contains_process_control(error):  # pragma: no cover
                    raise
                if durable_resume is not None:
                    try:
                        return await self._converge_durable_failure(
                            claim=claim,
                            segment=segment,
                            previous_segment=previous_segment,
                            sanitizer=sanitizer,
                            durable_resume=durable_resume,
                            error=error,
                        )
                    except (TaskQueueConflictError, TaskStoreConflictError):
                        return await self._lease_lost_result(
                            claim,
                            sanitizer=sanitizer,
                        )
                try:
                    retry = await self._finalize_failure(
                        definition,
                        claim=claim,
                        attempt=attempt,
                        segment=segment,
                        sanitizer=sanitizer,
                        error=error,
                    )
                except (TaskQueueConflictError, TaskStoreConflictError):
                    return await self._lease_lost_result(
                        claim,
                        sanitizer=sanitizer,
                    )
                return TaskWorkerProcessResult(claimed=claim, retry=retry)
            return TaskWorkerProcessResult(
                claimed=claim,
                suspension=suspension,
            )
        assert type(outcome) is TaskTargetCompleted
        output = outcome.output
        try:
            if durable_resume is None:
                completion = await self._complete_success(
                    definition,
                    claim=claim,
                    attempt=attempt,
                    segment=segment,
                    sanitizer=sanitizer,
                    output=output,
                )
            else:
                completion = await self._settle_durable_success(
                    definition,
                    claim=claim,
                    segment=segment,
                    sanitizer=sanitizer,
                    durable_resume=durable_resume,
                    output=output,
                )
        except (TaskQueueConflictError, TaskStoreConflictError) as error:
            if durable_resume is not None:
                try:
                    return await self._converge_durable_failure(
                        claim=claim,
                        segment=segment,
                        previous_segment=previous_segment,
                        sanitizer=sanitizer,
                        durable_resume=durable_resume,
                        error=error,
                    )
                except (TaskQueueConflictError, TaskStoreConflictError):
                    pass
            return await self._lease_lost_result(
                claim,
                sanitizer=sanitizer,
            )
        return TaskWorkerProcessResult(
            claimed=claim,
            completion=completion,
            output=output,
        )

    async def _converge_claimed_setup_failure(
        self,
        claim: TaskQueueClaim,
        *,
        definition: TaskDefinition | None,
        sanitizer: PrivacySanitizer | None,
        previous_segment: TaskAttemptSegment | None,
        durable_resume: TaskDurableResumeHandle | None,
        error: BaseException,
    ) -> TaskWorkerProcessResult:
        is_reentry = claim.attempt.state is TaskAttemptState.SUSPENDED
        if not is_reentry:
            max_attempts = (
                TaskAttemptPolicy.from_retry_policy(
                    definition.retry
                ).max_attempts
                if definition is not None
                else 1
            )
            try:
                abandonment = await self._queue.abandon(
                    claim.queue_item.queue_item_id,
                    claim_token=claim.queue_item.claim_token or "",
                    max_attempts=max_attempts,
                    now=self._now(),
                    metadata={
                        "worker_id": self._worker_id,
                        "reason": "setup_failed",
                    },
                )
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                abandonment=abandonment,
            )
        coordinator = self._durable_suspension_coordinator
        release_reentry = (
            getattr(coordinator, "release_claimed_reentry", None)
            if coordinator is not None
            else None
        )
        fail_reentry = (
            getattr(coordinator, "fail_claimed_reentry", None)
            if coordinator is not None
            else None
        )
        fail_admitted = (
            getattr(coordinator, "fail_admitted_reentry", None)
            if coordinator is not None
            else None
        )
        task_error = classify_task_error(error)
        deterministic = (
            isinstance(
                error,
                (_TaskDurableResumeRejected, TaskStoreConflictError),
            )
            or not task_error.retryable
        )
        provenance = (
            (
                previous_segment.request_id,
                previous_segment.continuation_id,
                previous_segment.checkpoint_id,
            )
            if previous_segment is not None
            else (None, None, None)
        )
        complete_provenance = all(
            isinstance(value, str) and bool(value) for value in provenance
        )
        if (
            isinstance(error, _TaskDurableResumeExpired)
            and complete_provenance
        ):
            return await self._converge_expired_resume(
                claim,
                sanitizer=sanitizer,
                error=error,
            )
        result = self._claimed_setup_failure_result(
            task_error,
            sanitizer=sanitizer,
        )
        if (
            deterministic
            and durable_resume is not None
            and complete_provenance
        ):
            if not callable(fail_admitted):
                return TaskWorkerProcessResult(
                    claimed=claim,
                    lease_lost=True,
                )
            failure = TaskDurableResumeFailure(result=result)
            rejection = durable_resume.rejection_command_for_settlement(
                failure
            )
            try:
                commit = await fail_admitted(
                    rejection,
                    failure,
                    queue_item_id=claim.queue_item.queue_item_id,
                    claim_token=claim.queue_item.claim_token or "",
                    task_run_id=claim.run.run_id,
                    request_id=cast(str, provenance[0]),
                    continuation_id=cast(str, provenance[1]),
                    checkpoint_id=cast(str, provenance[2]),
                    now=self._now(),
                    metadata={
                        "worker_id": self._worker_id,
                        "reason": "resume_setup_rejected",
                    },
                )
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            completion = commit.completion
            if not isinstance(completion, TaskQueueCompletion):
                raise TaskWorkerError(
                    "durable admission rejection returned invalid state"
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                completion=completion,
            )
        if durable_resume is not None:
            try:
                released = await durable_resume.release_if_pre_dispatch()
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            if not released:
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
        if not deterministic and complete_provenance:
            if not callable(release_reentry):
                return TaskWorkerProcessResult(
                    claimed=claim,
                    lease_lost=True,
                )
            try:
                reentry = await release_reentry(
                    queue_item_id=claim.queue_item.queue_item_id,
                    claim_token=claim.queue_item.claim_token or "",
                    task_run_id=claim.run.run_id,
                    request_id=cast(str, provenance[0]),
                    continuation_id=cast(str, provenance[1]),
                    checkpoint_id=cast(str, provenance[2]),
                    now=self._now(),
                    metadata={
                        "worker_id": self._worker_id,
                        "reason": "resume_setup_released",
                    },
                )
            except (TaskQueueConflictError, TaskStoreConflictError):
                return await self._lease_lost_result(
                    claim,
                    sanitizer=sanitizer,
                )
            if not isinstance(reentry, TaskQueueReentry):
                raise TaskWorkerError(
                    "durable reentry release returned invalid state"
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                reentry=reentry,
                shutdown_requested=isinstance(
                    error,
                    (CancelledError, _TaskWorkerShutdownRequested),
                ),
            )
        if not callable(fail_reentry):
            return TaskWorkerProcessResult(
                claimed=claim,
                lease_lost=True,
            )
        exact = provenance if complete_provenance else (None, None, None)
        try:
            completion = await fail_reentry(
                queue_item_id=claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                task_run_id=claim.run.run_id,
                request_id=exact[0],
                continuation_id=exact[1],
                checkpoint_id=exact[2],
                result=result,
                reason="resume_setup_failed",
                now=self._now(),
                metadata={
                    "worker_id": self._worker_id,
                    "reason": "resume_setup_failed",
                },
            )
        except (TaskQueueConflictError, TaskStoreConflictError):
            return await self._lease_lost_result(
                claim,
                sanitizer=sanitizer,
            )
        if not isinstance(completion, TaskQueueCompletion):
            raise TaskWorkerError(
                "durable reentry failure returned invalid state"
            )
        return TaskWorkerProcessResult(
            claimed=claim,
            completion=completion,
        )

    def _claimed_setup_failure_result(
        self,
        error: TaskError,
        *,
        sanitizer: PrivacySanitizer | None,
    ) -> TaskExecutionResult:
        summary = (
            self._safe_task_error_summary(sanitizer, error)
            if sanitizer is not None
            else cast(PrivacySafeValue, error.as_dict())
        )
        return TaskExecutionResult(error=_snapshot_value(summary))

    async def _revalidate_skills(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> TaskDefinition:
        pipeline = self._event_pipeline(
            definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            critical_delivery=True,
        )
        return await revalidate_task_skills_for_worker(
            definition,
            trusted_settings=self._skills_settings,
            registry=self._skills_registry,
            expected_identity=_task_skills_identity_from_run(run),
            event_manager=task_skill_audit_event_publisher(
                sanitizer=sanitizer,
                raw_event_observer=pipeline,
            ),
            schema_base_path=self._definition_base,
        )

    async def _execute(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        run: TaskRun,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment,
        sanitizer: PrivacySanitizer,
        durable_resume: TaskDurableResumeHandle | None,
        durable_target: TaskDurableResumeTargetRunner | None,
        heartbeat_task: AsyncTask[None] | None,
    ) -> TaskTargetOutcome:
        async def observe_usage(response: object) -> None:
            await self._record_usage(
                response,
                definition=definition,
                run=run,
                attempt=attempt,
                segment=segment,
            )

        usage_observer = (
            observe_usage if definition.observability.metrics else None
        )
        usage_tracker = TaskUsageObservationTracker(
            usage_observer,
            has_observations=lambda response: bool(
                usage_observations_from_response(response)
            ),
        )
        if durable_resume is not None:
            if durable_target is None:
                raise TaskWorkerError("durable resume target was not prepared")
            context = self._target_context(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
                usage_tracker=usage_tracker,
                durable_resume=durable_resume,
            )
            outcome = task_target_outcome(
                await self._run_target(
                    context,
                    claim=claim,
                    timeout=definition.run.timeout_seconds,
                    heartbeat_task=heartbeat_task,
                    durable_target=durable_target,
                )
            )
            await self._check_cancelled(run.run_id)
            if type(outcome) is TaskTargetSuspended:
                return outcome
            assert type(outcome) is TaskTargetCompleted
            output = outcome.output
            await usage_tracker.observe(output)
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
            return completed_task_target_outcome(output)
        files = await self._input_files(definition, run, attempt)
        input_mounts = task_container_input_mount_manifest(
            files,
            allowed_roots=tuple(Path(root) for root in self._execution_roots),
        )
        container_result = await self._run_task_container(
            definition,
            run=run,
            attempt=attempt,
            input_mounts=input_mounts,
            sanitizer=sanitizer,
        )
        if container_result is not None:
            output = container_result.output
            await self._check_cancelled(run.run_id)
            await usage_tracker.observe(output)
            issues = validate_task_output(definition, output)
            if issues:
                raise TaskValidationError(issues)
            if not container_result.output_artifacts_recorded:
                await self._record_output_artifacts(
                    definition,
                    output,
                    run=run,
                    attempt=attempt,
                    sanitizer=sanitizer,
                )
            return completed_task_target_outcome(output)
        context = TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            input_value=self._executable_input_value(definition, run),
            files=files,
            metadata=task_container_user_metadata(run.request.metadata),
            cancellation_checker=lambda: self._check_cancelled(run.run_id),
            event_listener=self._event_pipeline(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            ),
            usage_observer=(
                usage_tracker.observe if usage_observer is not None else None
            ),
            artifact_store=self._artifact_store,
            task_store=self._store,
            durable_resume=durable_resume,
            file_converters=self._file_converters,
        )
        await self._check_cancelled(run.run_id)
        outcome = task_target_outcome(
            await self._run_target(
                context,
                claim=claim,
                timeout=definition.run.timeout_seconds,
                heartbeat_task=heartbeat_task,
            )
        )
        await self._check_cancelled(run.run_id)
        if type(outcome) is TaskTargetSuspended:
            return outcome
        assert type(outcome) is TaskTargetCompleted
        output = outcome.output
        await usage_tracker.observe(output)
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
        return completed_task_target_outcome(output)

    def _target_context(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        usage_tracker: TaskUsageObservationTracker,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetContext:
        return TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            input_value=None,
            files=(),
            metadata=task_container_user_metadata(run.request.metadata),
            cancellation_checker=lambda: self._check_cancelled(run.run_id),
            event_listener=self._event_pipeline(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            ),
            usage_observer=usage_tracker.observe,
            artifact_store=self._artifact_store,
            task_store=self._store,
            durable_resume=durable_resume,
            file_converters=self._file_converters,
        )

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
        heartbeat_task: AsyncTask[None] | None = None,
        durable_target: TaskDurableResumeTargetRunner | None = None,
    ) -> TaskTargetOutcome:
        target_execution = self._target_execution(
            context,
            durable_target=durable_target,
        )
        owns_heartbeat_task = False
        if heartbeat_task is None:
            heartbeat_seconds = self._claim_heartbeat_seconds(claim)
            if heartbeat_seconds is not None:
                heartbeat_task = create_task(
                    self._heartbeat_claim(claim, heartbeat_seconds),
                    name=(
                        "task-claim-heartbeat-"
                        f"{claim.queue_item.queue_item_id}"
                    ),
                )
                owns_heartbeat_task = True
        if self._shutdown is None and heartbeat_task is None:
            try:
                return await wait_for(target_execution, timeout=timeout)
            except TimeoutError:
                durable_resume = context.durable_resume
                if durable_resume is not None:
                    await durable_resume.interrupt_dispatch()
                raise
        target_task = create_task(target_execution)
        shutdown_task = (
            create_task(self._shutdown.wait())
            if self._shutdown is not None
            else None
        )
        wait_tasks: set[AsyncTask[object]] = {
            cast(AsyncTask[object], target_task)
        }
        if heartbeat_task is not None:
            wait_tasks.add(cast(AsyncTask[object], heartbeat_task))
        if shutdown_task is not None:
            wait_tasks.add(cast(AsyncTask[object], shutdown_task))
        try:
            done, _pending = await wait(
                wait_tasks,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                await self._interrupt_durable_target(context, target_task)
                raise TimeoutError()  # pragma: no cover
            if heartbeat_task is not None and heartbeat_task in done:
                interruption_error: BaseException | None = None
                if context.durable_resume is not None:
                    try:
                        await self._interrupt_durable_target(
                            context,
                            target_task,
                        )
                    except BaseException as error:
                        if _contains_process_control(
                            error
                        ):  # pragma: no cover
                            raise
                        interruption_error = error
                self._raise_if_claim_guard_stopped(
                    heartbeat_task,
                    admission_cleanup_error=interruption_error,
                )
                raise _TaskWorkerShutdownRequested()  # pragma: no cover
            if shutdown_task is not None and (
                shutdown_task in done or self._shutdown_requested()
            ):
                if context.durable_resume is not None:
                    await self._interrupt_durable_target(
                        context,
                        target_task,
                    )
                raise _TaskWorkerShutdownRequested()
            if target_task in done:
                return await target_task
            raise _TaskWorkerShutdownRequested()  # pragma: no cover
        finally:
            await _cancel_task(target_task)
            if shutdown_task is not None:
                await _cancel_task(shutdown_task)
            if owns_heartbeat_task and heartbeat_task is not None:
                await _cancel_task(heartbeat_task)

    @staticmethod
    async def _interrupt_durable_target(
        context: TaskTargetContext,
        target_task: AsyncTask[TaskTargetOutcome],
    ) -> None:
        durable_resume = context.durable_resume
        if durable_resume is None:
            return
        await _cancel_task(target_task)
        state = await durable_resume.interrupt_dispatch()
        if state not in {
            DurableContinuationResumeState.RELEASED,
            DurableContinuationResumeState.DISPATCHED,
            DurableContinuationResumeState.AMBIGUOUS,
            DurableContinuationResumeState.COMPLETED,
        }:
            raise TaskWorkerError(
                "durable target interruption did not reach a safe settlement"
            )

    async def _target_execution(
        self,
        context: TaskTargetContext,
        *,
        durable_target: TaskDurableResumeTargetRunner | None,
    ) -> TaskTargetOutcome:
        durable_resume = context.durable_resume
        if durable_resume is None:
            if durable_target is not None:
                raise TaskWorkerError(
                    "durable resume target has no admitted continuation"
                )
            return await self._target.run(context)
        if durable_target is None:
            raise TaskWorkerError("durable resume target was not prepared")
        return await durable_target.resume(context, durable_resume)

    def _prepare_durable_resume_target(
        self,
        definition: TaskDefinition,
        *,
        previous_segment: TaskAttemptSegment | None,
    ) -> TaskDurableResumeTargetRunner | None:
        if previous_segment is None:
            return None
        target_type = definition.execution.type
        if isinstance(self._target, TaskDurableResumeTargetPreparer):
            prepared = self._target.prepare_durable_resume(target_type)
            if type(
                prepared
            ) is PreparedTaskDurableResumeTarget and prepared.is_bound_to(
                self._target, target_type
            ):
                return prepared.runner
        elif (
            isinstance(self._target, TaskDurableResumeTargetRunner)
            and self._target.supports_durable_resume(target_type) is True
        ):
            return self._target
        raise _TaskDurableResumeRejected(
            "task target does not support durable resume"
        )

    def _claim_heartbeat_seconds(
        self,
        claim: TaskQueueClaim,
    ) -> float | None:
        if self._heartbeat_seconds is not None:
            return self._heartbeat_seconds
        if claim.attempt.state is TaskAttemptState.SUSPENDED:
            return self._lease_seconds / 3
        return None

    @staticmethod
    def _raise_if_claim_guard_stopped(
        heartbeat_task: AsyncTask[None] | None,
        *,
        admission_cleanup_error: BaseException | None = None,
    ) -> None:
        if heartbeat_task is None or not heartbeat_task.done():
            return
        TaskWorker._raise_stopped_claim_guard(
            heartbeat_task,
            admission_cleanup_error=admission_cleanup_error,
        )

    @staticmethod
    def _raise_stopped_claim_guard(
        heartbeat_task: AsyncTask[None],
        *,
        admission_cleanup_error: BaseException | None = None,
    ) -> NoReturn:
        """Raise the terminal outcome of one known-completed claim guard."""
        if admission_cleanup_error is not None and _contains_process_control(
            admission_cleanup_error
        ):  # pragma: no cover
            raise admission_cleanup_error
        if heartbeat_task.cancelled():
            if admission_cleanup_error is not None:
                raise admission_cleanup_error
            raise TaskQueueConflictError("task queue heartbeat stopped")
        error = heartbeat_task.exception()
        if error is not None:
            if _contains_process_control(error):  # pragma: no cover
                raise error
            if _is_input_expiry(error):
                cleanup_errors = tuple(
                    candidate
                    for candidate in (
                        _input_expiry_cleanup_error(error),
                        (
                            _input_expiry_cleanup_error(
                                admission_cleanup_error,
                                ignore_cancelled=True,
                            )
                            if admission_cleanup_error is not None
                            else None
                        ),
                    )
                    if candidate is not None
                )
                cleanup_error = (
                    cleanup_errors[0]
                    if len(cleanup_errors) == 1
                    else (
                        BaseExceptionGroup(
                            "durable admission cleanup failed",
                            cleanup_errors,
                        )
                        if cleanup_errors
                        else None
                    )
                )
                raise _TaskDurableResumeExpired(
                    "durable continuation expired before dispatch",
                    cleanup_error=cleanup_error,
                ) from error
            if admission_cleanup_error is not None:
                raise BaseExceptionGroup(
                    "task heartbeat and admission cleanup both failed",
                    (error, admission_cleanup_error),
                )
            raise TaskQueueConflictError(
                "task queue heartbeat failed"
            ) from error
        if admission_cleanup_error is not None:
            raise admission_cleanup_error
        raise _TaskWorkerShutdownRequested()

    async def _heartbeat_claim(
        self,
        claim: TaskQueueClaim,
        heartbeat_seconds: float | None = None,
        *,
        claim_lease_manager: _TaskResumeClaimLeaseBridge | None = None,
    ) -> None:
        heartbeat_seconds = (
            heartbeat_seconds
            if heartbeat_seconds is not None
            else self._claim_heartbeat_seconds(claim)
        )
        assert isinstance(heartbeat_seconds, int | float)
        assert not isinstance(heartbeat_seconds, bool)
        assert heartbeat_seconds > 0
        while True:
            await sleep(heartbeat_seconds)
            if self._shutdown_requested():
                return
            now = self._now()
            item = await self._queue.heartbeat(
                claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                lease_expires_at=(
                    now + timedelta(seconds=self._lease_seconds)
                ),
                now=now,
            )
            if claim_lease_manager is not None:
                lease_expires_at = item.lease_expires_at
                if lease_expires_at is None:
                    raise TaskQueueConflictError(
                        "task queue heartbeat returned no claim lease"
                    )
                await claim_lease_manager.heartbeat(
                    lease_expires_at,
                    now=now,
                )

    async def _start_claimed_attempt(
        self,
        claim: TaskQueueClaim,
        *,
        previous_segment: TaskAttemptSegment | None,
    ) -> tuple[TaskRun, TaskAttempt, TaskAttemptSegment]:
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
            from_states={
                TaskAttemptState.CREATED,
                TaskAttemptState.SUSPENDED,
            },
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claim_token,
            metadata={"worker_id": self._worker_id},
        )
        segment = await self._store.create_attempt_segment(
            attempt.attempt_id,
            claim_token=claim_token,
            resumed_from_segment_id=(
                previous_segment.segment_id
                if previous_segment is not None
                else None
            ),
            metadata={"worker_id": self._worker_id},
        )
        segment = await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.CREATED},
            to_state=TaskAttemptSegmentState.RUNNING,
            reason="started",
            claim_token=claim_token,
            metadata={"worker_id": self._worker_id},
        )
        return run, attempt, segment

    async def _previous_attempt_segment(
        self,
        claim: TaskQueueClaim,
    ) -> TaskAttemptSegment | None:
        segments = await self._store.list_attempt_segments(
            claim.attempt.attempt_id
        )
        previous = segments[-1] if segments else None
        if claim.attempt.state is TaskAttemptState.CREATED:
            if previous is not None:
                raise TaskStoreConflictError(
                    "fresh task attempt already has segment history"
                )
            return None
        if (
            claim.attempt.state is not TaskAttemptState.SUSPENDED
            or previous is None
            or previous.state is not TaskAttemptSegmentState.SUSPENDED
        ):
            raise TaskStoreConflictError(
                "requeued task lacks a suspended prior segment"
            )
        return previous

    async def _admit_durable_resume(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment | None,
        *,
        claim_lease_manager: TaskResumeClaimLeaseManager,
        resume_owner: _TaskDurableResumeOwner,
    ) -> TaskDurableResumeHandle | None:
        coordinator = self._durable_resume_coordinator
        if coordinator is None:
            if previous_segment is not None:
                raise _TaskDurableResumeRejected(
                    "durable task resume coordinator is unavailable"
                )
            return None
        try:
            admission = await coordinator.admit(
                claim,
                previous_segment,
                claim_lease_manager=claim_lease_manager,
            )
        except BaseException as error:
            if _contains_process_control(error):  # pragma: no cover
                raise
            if _is_input_expiry(error):
                raise _TaskDurableResumeExpired(
                    "durable continuation expired before dispatch",
                    cleanup_error=_input_expiry_cleanup_error(error),
                ) from error
            raise
        resume_owner.take(admission)
        if admission is None:
            if previous_segment is not None:
                raise _TaskDurableResumeRejected(
                    "durable task reentry was not admitted"
                )
            return None
        if previous_segment is None:
            raise _TaskDurableResumeRejected(
                "fresh task work cannot carry durable resume admission"
            )
        for method in (
            "dispatch",
            "wait_dispatch_settled",
            "interrupt_dispatch",
            "completed_completion_command",
            "completion_command_for_settlement",
            "completion_command_for_suspension",
            "rejection_command_for_settlement",
            "release",
            "release_if_pre_dispatch",
            "close",
        ):
            if not callable(getattr(admission, method, None)):
                raise _TaskDurableResumeRejected(
                    "durable task resume admission is invalid"
                )
        return admission

    async def _admit_durable_resume_guarded(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment | None,
        *,
        heartbeat_task: AsyncTask[None] | None,
        claim_lease_manager: TaskResumeClaimLeaseManager,
        resume_owner: _TaskDurableResumeOwner,
    ) -> TaskDurableResumeHandle | None:
        """Monitor the task claim while cold admission reconstructs runtime."""
        if heartbeat_task is None:
            return await self._admit_durable_resume(
                claim,
                previous_segment,
                claim_lease_manager=claim_lease_manager,
                resume_owner=resume_owner,
            )
        admission_task = create_task(
            self._admit_durable_resume(
                claim,
                previous_segment,
                claim_lease_manager=claim_lease_manager,
                resume_owner=resume_owner,
            ),
            name=f"durable-resume-admission-{claim.queue_item.queue_item_id}",
        )
        try:
            done, _pending = await wait(
                {
                    cast(AsyncTask[object], admission_task),
                    cast(AsyncTask[object], heartbeat_task),
                },
                return_when=FIRST_COMPLETED,
            )
        except BaseException as error:
            cleanup_error = await _cancel_task_error(admission_task)
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "durable admission wait and cleanup both failed",
                    (error, cleanup_error),
                ) from None
            raise
        if admission_task in done:
            try:
                admission = await admission_task
            except BaseException as error:
                if heartbeat_task.done():
                    self._raise_if_claim_guard_stopped(
                        heartbeat_task,
                        admission_cleanup_error=error,
                    )
                raise
            self._raise_if_claim_guard_stopped(heartbeat_task)
            return admission
        cleanup_error = await _cancel_task_error(admission_task)
        self._raise_stopped_claim_guard(
            heartbeat_task,
            admission_cleanup_error=cleanup_error,
        )

    async def _complete_success(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment,
        sanitizer: PrivacySanitizer,
        output: object,
    ) -> TaskQueueCompletion:
        result = self._success_result(
            definition,
            sanitizer=sanitizer,
            output=output,
        )
        await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.SUCCEEDED,
            reason="completed",
            claim_token=claim.queue_item.claim_token or "",
            metadata={"worker_id": self._worker_id},
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

    async def _settle_durable_success(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        segment: TaskAttemptSegment,
        sanitizer: PrivacySanitizer,
        durable_resume: TaskDurableResumeHandle,
        output: object,
    ) -> TaskQueueCompletion:
        coordinator = self._durable_suspension_coordinator
        settle = (
            getattr(coordinator, "settle_resume", None)
            if coordinator is not None
            else None
        )
        if not callable(settle):
            raise TaskWorkerError(
                "durable task settlement coordinator is unavailable"
            )
        settlement = TaskDurableResumeSuccess(
            result=self._success_result(
                definition,
                sanitizer=sanitizer,
                output=output,
            )
        )
        command = durable_resume.completion_command_for_settlement(settlement)
        commit = await settle(
            command,
            settlement,
            queue_item_id=claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            segment_id=segment.segment_id,
            task_run_id=claim.run.run_id,
            now=self._now(),
            metadata={"worker_id": self._worker_id},
        )
        completion = commit.completion
        if not isinstance(completion, TaskQueueCompletion):
            raise TaskWorkerError(
                "durable settlement coordinator returned invalid state"
            )
        return completion

    def _success_result(
        self,
        definition: TaskDefinition,
        *,
        sanitizer: PrivacySanitizer,
        output: object,
    ) -> TaskExecutionResult:
        return TaskExecutionResult(
            output_summary=_snapshot_value(
                sanitizer.sanitize(
                    PrivacyField.OUTPUT,
                    _output_summary_value(definition, output),
                )
            )
        )

    async def _converge_durable_failure(
        self,
        *,
        claim: TaskQueueClaim,
        segment: TaskAttemptSegment,
        previous_segment: TaskAttemptSegment | None,
        sanitizer: PrivacySanitizer,
        durable_resume: TaskDurableResumeHandle,
        error: BaseException,
    ) -> TaskWorkerProcessResult:
        convergence = create_task(
            self._converge_durable_failure_owned(
                claim=claim,
                segment=segment,
                previous_segment=previous_segment,
                sanitizer=sanitizer,
                durable_resume=durable_resume,
                error=error,
            ),
            name=(
                "task-durable-failure-convergence-"
                f"{claim.queue_item.queue_item_id}"
            ),
        )
        return await _await_durable_failure_convergence(convergence)

    async def _converge_durable_failure_owned(
        self,
        *,
        claim: TaskQueueClaim,
        segment: TaskAttemptSegment,
        previous_segment: TaskAttemptSegment | None,
        sanitizer: PrivacySanitizer,
        durable_resume: TaskDurableResumeHandle,
        error: BaseException,
    ) -> TaskWorkerProcessResult:
        if _contains_cancellation(error):
            await durable_resume.interrupt_dispatch()
        state = await durable_resume.wait_dispatch_settled()
        result = self._durable_failure_result(
            sanitizer,
            error=error,
        )
        coordinator = self._durable_suspension_coordinator
        if state is DurableContinuationResumeState.COMPLETED:
            completion = await self._terminalize_completed_durable_failure(
                claim=claim,
                segment=segment,
                previous_segment=previous_segment,
                durable_resume=durable_resume,
                result=result,
                error=error,
            )
            return TaskWorkerProcessResult(
                claimed=claim,
                completion=completion,
                shutdown_requested=isinstance(
                    error,
                    _TaskWorkerShutdownRequested,
                ),
            )
        if state is DurableContinuationResumeState.DISPATCHED:
            settle = (
                getattr(coordinator, "settle_resume", None)
                if coordinator is not None
                else None
            )
            if not callable(settle):
                raise TaskWorkerError(
                    "durable task settlement coordinator is unavailable"
                )
            settlement = (
                TaskDurableResumeCancellation(result=result)
                if isinstance(error, CancelledError)
                else TaskDurableResumeFailure(result=result)
            )
            command = durable_resume.completion_command_for_settlement(
                settlement
            )
            commit = await settle(
                command,
                settlement,
                queue_item_id=claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                segment_id=segment.segment_id,
                task_run_id=claim.run.run_id,
                now=self._now(),
                metadata={
                    "worker_id": self._worker_id,
                    "resume_dispatch_state": state.value,
                },
            )
            completion = commit.completion
            if not isinstance(completion, TaskQueueCompletion):
                raise TaskWorkerError(
                    "durable settlement coordinator returned invalid state"
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                completion=completion,
                shutdown_requested=isinstance(
                    error,
                    _TaskWorkerShutdownRequested,
                ),
            )
        if state is DurableContinuationResumeState.AMBIGUOUS:
            mark_ambiguous = (
                getattr(coordinator, "mark_resume_ambiguous", None)
                if coordinator is not None
                else None
            )
            if not callable(mark_ambiguous):
                raise TaskWorkerError(
                    "durable ambiguity coordinator is unavailable"
                )
            failure = TaskDurableResumeFailure(
                result=self._durable_ambiguity_result(sanitizer)
            )
            command = durable_resume.completion_command_for_settlement(failure)
            commit = await mark_ambiguous(
                command,
                failure,
                queue_item_id=claim.queue_item.queue_item_id,
                claim_token=claim.queue_item.claim_token or "",
                segment_id=segment.segment_id,
                task_run_id=claim.run.run_id,
                now=self._now(),
                metadata={
                    "worker_id": self._worker_id,
                    "resume_dispatch_state": state.value,
                },
            )
            completion = commit.completion
            if not isinstance(completion, TaskQueueCompletion):
                raise TaskWorkerError(
                    "durable ambiguity coordinator returned invalid state"
                )
            return TaskWorkerProcessResult(
                claimed=claim,
                completion=completion,
                shutdown_requested=isinstance(
                    error,
                    (CancelledError, _TaskWorkerShutdownRequested),
                ),
            )
        if state is DurableContinuationResumeState.ADMITTED:
            released = await durable_resume.release_if_pre_dispatch()
            if not released:
                raise TaskWorkerError(
                    "durable resume dispatch state changed during failure"
                )
            state = DurableContinuationResumeState.RELEASED
        if state is not DurableContinuationResumeState.RELEASED:
            raise TaskWorkerError(
                "durable resume failure did not reach a safe settlement"
            )
        release_running = (
            getattr(coordinator, "release_running_reentry", None)
            if coordinator is not None
            else None
        )
        if not callable(release_running):
            raise TaskWorkerError(
                "durable running reentry release is unavailable"
            )
        if (
            previous_segment is None
            or previous_segment.request_id is None
            or previous_segment.continuation_id is None
            or previous_segment.checkpoint_id is None
        ):
            raise TaskWorkerError(
                "durable running reentry provenance is unavailable"
            )
        reentry = await release_running(
            queue_item_id=claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            segment_id=segment.segment_id,
            task_run_id=claim.run.run_id,
            request_id=previous_segment.request_id,
            continuation_id=previous_segment.continuation_id,
            checkpoint_id=previous_segment.checkpoint_id,
            now=self._now(),
            metadata={
                "worker_id": self._worker_id,
                "resume_dispatch_state": state.value,
            },
        )
        if not isinstance(reentry, TaskQueueReentry):
            raise TaskWorkerError(
                "durable running reentry release returned invalid state"
            )
        return TaskWorkerProcessResult(
            claimed=claim,
            reentry=reentry,
            shutdown_requested=isinstance(
                error,
                (CancelledError, _TaskWorkerShutdownRequested),
            ),
        )

    async def _lease_lost_result(
        self,
        claim: TaskQueueClaim,
        *,
        sanitizer: PrivacySanitizer | None,
        shutdown_requested: bool = False,
    ) -> TaskWorkerProcessResult:
        if claim.attempt.state is not TaskAttemptState.SUSPENDED:
            return TaskWorkerProcessResult(
                claimed=claim,
                shutdown_requested=shutdown_requested,
                lease_lost=True,
            )
        coordinator = self._durable_suspension_coordinator
        reconcile = (
            getattr(coordinator, "reconcile_expired_reentry", None)
            if coordinator is not None
            else None
        )
        if not callable(reconcile):
            return TaskWorkerProcessResult(
                claimed=claim,
                shutdown_requested=shutdown_requested,
                lease_lost=True,
            )
        try:
            commit = await reconcile(
                queue_item_id=claim.queue_item.queue_item_id,
                expected_claim_token=claim.queue_item.claim_token or "",
                task_run_id=claim.run.run_id,
                result=self._durable_ambiguity_result(sanitizer),
                now=self._now(),
                metadata={
                    "worker_id": self._worker_id,
                    "reason": "resume_claim_expired",
                },
            )
        except (TaskQueueConflictError, TaskStoreConflictError):
            return TaskWorkerProcessResult(
                claimed=claim,
                shutdown_requested=shutdown_requested,
                lease_lost=True,
            )
        reentry = getattr(commit, "reentry", None)
        completion = getattr(commit, "completion", None)
        if (
            isinstance(reentry, TaskQueueReentry)
            and completion is None
            and reentry.queue_item.queue_item_id
            == claim.queue_item.queue_item_id
            and reentry.run.run_id == claim.run.run_id
        ):
            return TaskWorkerProcessResult(
                claimed=claim,
                reentry=reentry,
                shutdown_requested=shutdown_requested,
                lease_lost=True,
            )
        if (
            isinstance(completion, TaskQueueCompletion)
            and reentry is None
            and completion.queue_item.queue_item_id
            == claim.queue_item.queue_item_id
            and completion.run.run_id == claim.run.run_id
        ):
            return TaskWorkerProcessResult(
                claimed=claim,
                completion=completion,
                shutdown_requested=shutdown_requested,
                lease_lost=True,
            )
        raise TaskWorkerError(
            "expired durable reentry reconciliation returned invalid state"
        )

    async def _terminalize_completed_durable_failure(
        self,
        *,
        claim: TaskQueueClaim,
        segment: TaskAttemptSegment,
        previous_segment: TaskAttemptSegment | None,
        durable_resume: TaskDurableResumeHandle,
        result: TaskExecutionResult,
        error: BaseException,
    ) -> TaskQueueCompletion:
        """Fail task state without changing completed provider evidence."""
        coordinator = self._durable_suspension_coordinator
        terminalize = (
            getattr(coordinator, "terminalize_completed_resume", None)
            if coordinator is not None
            else None
        )
        if not callable(terminalize):
            raise TaskWorkerError(
                "completed durable task settlement coordinator is unavailable"
            )
        if (
            previous_segment is None
            or previous_segment.request_id is None
            or previous_segment.continuation_id is None
            or previous_segment.checkpoint_id is None
        ):
            raise TaskWorkerError(
                "completed durable task provenance is unavailable"
            )
        settlement = (
            TaskDurableResumeCancellation(result=result)
            if isinstance(error, CancelledError)
            else TaskDurableResumeFailure(result=result)
        )
        completion_command = durable_resume.completed_completion_command()
        metadata = {
            "worker_id": self._worker_id,
            "resume_dispatch_state": (
                DurableContinuationResumeState.COMPLETED.value
            ),
        }
        commit = await terminalize(
            completion_command,
            settlement,
            queue_item_id=claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            segment_id=segment.segment_id,
            task_run_id=claim.run.run_id,
            request_id=previous_segment.request_id,
            checkpoint_id=previous_segment.checkpoint_id,
            now=self._now(),
            metadata=metadata,
        )
        completion = commit.completion
        if not isinstance(completion, TaskQueueCompletion):
            raise TaskWorkerError(
                "completed durable task settlement returned invalid state"
            )
        return completion

    async def _converge_expired_resume(
        self,
        claim: TaskQueueClaim,
        *,
        sanitizer: PrivacySanitizer | None,
        error: _TaskDurableResumeExpired,
    ) -> TaskWorkerProcessResult:
        """Reconcile absolute expiry before surfacing failed cleanup."""
        converged = await self._lease_lost_result(
            claim,
            sanitizer=sanitizer,
        )
        if error.cleanup_error is not None:
            raise error.cleanup_error from error
        return converged

    def _durable_failure_result(
        self,
        sanitizer: PrivacySanitizer,
        *,
        error: BaseException,
    ) -> TaskExecutionResult:
        task_error = classify_task_error(error)
        return TaskExecutionResult(
            error=_snapshot_value(
                self._safe_task_error_summary(sanitizer, task_error)
            )
        )

    def _durable_ambiguity_result(
        self,
        sanitizer: PrivacySanitizer | None,
    ) -> TaskExecutionResult:
        task_error = TaskError.infra()
        summary = (
            self._safe_task_error_summary(sanitizer, task_error)
            if sanitizer is not None
            else cast(PrivacySafeValue, task_error.as_dict())
        )
        return TaskExecutionResult(
            error=_snapshot_value(summary),
            metadata={
                "durable_resume": "provider_dispatch_ambiguous",
            },
        )

    async def _suspend(
        self,
        *,
        claim: TaskQueueClaim,
        segment: TaskAttemptSegment,
        outcome: TaskTargetSuspended,
        durable_resume: TaskDurableResumeHandle | None,
    ) -> TaskQueueSuspension:
        required = outcome.input_required
        durable = outcome.durable
        coordinator = self._durable_suspension_coordinator
        if durable is not None:
            checkpoint_id = outcome.checkpoint_id
            if checkpoint_id is None:
                raise TaskWorkerError(
                    "durable task suspension checkpoint is unavailable"
                )
            if coordinator is None:
                raise TaskWorkerError(
                    "durable task suspension coordinator is unavailable"
                )
            metadata = {
                "worker_id": self._worker_id,
                "detached_resumption_available": (
                    required.detached_resumption_available
                ),
            }
            if durable_resume is None:
                commit = await coordinator.create_and_suspend(
                    durable.command,
                    durable.continuation,
                    queue_item_id=claim.queue_item.queue_item_id,
                    claim_token=claim.queue_item.claim_token or "",
                    segment_id=segment.segment_id,
                    task_run_id=claim.run.run_id,
                    checkpoint_id=checkpoint_id,
                    now=self._now(),
                    metadata=metadata,
                )
            else:
                completion = durable_resume.completion_command_for_suspension(
                    request_id=str(required.request_id),
                    continuation_id=str(required.continuation_id),
                    checkpoint_id=checkpoint_id,
                )
                commit = await coordinator.complete_and_resuspend(
                    completion,
                    durable.command,
                    durable.continuation,
                    queue_item_id=claim.queue_item.queue_item_id,
                    claim_token=claim.queue_item.claim_token or "",
                    segment_id=segment.segment_id,
                    task_run_id=claim.run.run_id,
                    checkpoint_id=checkpoint_id,
                    now=self._now(),
                    metadata=metadata,
                )
            suspension = commit.suspension
            if not isinstance(suspension, TaskQueueSuspension):
                raise TaskWorkerError(
                    "durable suspension coordinator returned invalid state"
                )
            return suspension
        if coordinator is not None or durable_resume is not None:
            raise TaskWorkerError(
                "durable task suspension payload is unavailable"
            )
        return await self._queue.suspend_claim(
            claim.queue_item.queue_item_id,
            claim_token=claim.queue_item.claim_token or "",
            segment_id=segment.segment_id,
            request_id=str(required.request_id),
            continuation_id=str(required.continuation_id),
            checkpoint_id=outcome.checkpoint_id,
            now=self._now(),
            metadata={
                "worker_id": self._worker_id,
                "detached_resumption_available": (
                    required.detached_resumption_available
                ),
            },
        )

    async def _finalize_failure(
        self,
        definition: TaskDefinition,
        *,
        claim: TaskQueueClaim,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment,
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
        await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=(
                TaskAttemptSegmentState.ABANDONED
                if isinstance(error, CancelledError)
                else TaskAttemptSegmentState.FAILED
            ),
            reason=(
                "cancelled"
                if isinstance(error, CancelledError)
                else "execution_failed"
            ),
            claim_token=claim.queue_item.claim_token or "",
            metadata={"worker_id": self._worker_id},
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
        segment: TaskAttemptSegment,
    ) -> TaskQueueAbandonment:
        policy = TaskAttemptPolicy.from_retry_policy(definition.retry)
        await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.ABANDONED,
            reason="shutdown",
            claim_token=claim.queue_item.claim_token or "",
            metadata={"worker_id": self._worker_id},
        )
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
                await task_input_file_groups_from_materialized(
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
                if entry.materialized_file is not None:
                    files.extend(next(converted))
                    continue
                files.append(entry.file)
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
        safe_artifacts = tuple(
            _sanitize_output_artifact(artifact, sanitizer)
            for artifact in _output_artifacts_from_output(definition, output)
        )
        await gather(
            *(
                self._store.append_artifact(
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
                for safe_artifact in safe_artifacts
            )
        )

    async def _record_usage(
        self,
        response: object,
        *,
        definition: TaskDefinition,
        run: TaskRun,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment | None = None,
    ) -> None:
        await record_response_usage(
            self._observability_sink_for(definition),
            store=self._store,
            response=response,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            segment_id=segment.segment_id if segment is not None else None,
        )

    def _event_pipeline(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        critical_delivery: bool = False,
    ) -> TaskEventPipeline | None:
        assert isinstance(critical_delivery, bool)
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
            critical_delivery=critical_delivery,
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
                artifact_store=self._artifact_store,
                task_store=self._store,
                file_converters=self._file_converters,
            ),
        )
        if issues:
            raise TaskValidationError(issues)

    async def _run_task_container(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        input_mounts: tuple[dict[str, object], ...],
        sanitizer: PrivacySanitizer,
    ) -> TaskContainerAttemptResult | None:
        try:
            plans = verify_task_container_request(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=input_mounts,
            )
        except TaskContainerVerificationError as error:
            raise TaskValidationError(
                (_container_validation_issue(error),)
            ) from error
        if not plans.enabled:
            return None
        await self._record_container_event(
            definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            event_type="container_plan_verified",
            plans=plans,
            input_mounts=input_mounts,
            status="verified",
        )
        if plans.worker_envelope is not None:
            if self._worker_runtime_envelope_runner is not None:
                worker_envelope_result = (
                    await self._worker_runtime_envelope_runner(
                        definition,
                        run=run,
                        attempt=attempt,
                        plans=plans,
                        input_mounts=input_mounts,
                    )
                )
                assert isinstance(
                    worker_envelope_result,
                    TaskContainerAttemptResult,
                )
                await self._record_container_event(
                    definition,
                    run=run,
                    attempt=attempt,
                    sanitizer=sanitizer,
                    event_type="container_worker_envelope_completed",
                    plans=plans,
                    input_mounts=input_mounts,
                    status="completed",
                )
                return worker_envelope_result
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.worker_envelope_unsupported",
                        path="container.worker_envelope",
                        message=(
                            "Task worker runtime envelopes require an "
                            "envelope-aware task runtime."
                        ),
                        hint=(
                            "Route this task to a trusted envelope-aware "
                            "worker or remove the worker envelope policy."
                        ),
                    ),
                )
            )
        if plans.attempt is None:
            return None
        if self._container_backend is None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.backend_unavailable",
                        path="container.backend",
                        message=(
                            "Task container execution requires an injected "
                            "container backend."
                        ),
                        hint=(
                            "Run this task with a trusted container-capable "
                            "worker."
                        ),
                    ),
                )
            )
        run_plan = task_container_lifecycle_run_plan(
            plans,
            input_mounts=input_mounts,
        )
        assert run_plan is not None
        await self._check_cancelled(run.run_id)
        unsupported_input_mount_path = (
            task_container_unsupported_input_mount_path(input_mounts)
        )
        if unsupported_input_mount_path is not None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.input_mount_unsupported",
                        path=unsupported_input_mount_path,
                        message=(
                            "Task-attempt container inputs must be "
                            "artifact-backed scoped mounts."
                        ),
                        hint=(
                            "Materialize task inputs to task artifacts before "
                            "running them in a required container."
                        ),
                    ),
                )
            )
        output_contract = task_container_output_contract(definition, plans)
        if output_contract is None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.task_execution_unsupported",
                        path="container.output",
                        message=(
                            "This task output contract cannot be completed "
                            "through the task-attempt container runtime."
                        ),
                        hint=(
                            "Use a file or artifact output contract with "
                            "container artifact output enabled, or run "
                            "without a required task-attempt container."
                        ),
                    ),
                )
            )
        if not output_contract.enabled:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.output_unsupported",
                        path="container.output",
                        message=(
                            "Task-attempt container output artifacts are "
                            "disabled by the trusted container profile."
                        ),
                        hint=(
                            "Enable artifact output in the trusted profile "
                            "or use an output contract that does not require "
                            "container artifact collection."
                        ),
                    ),
                )
            )
        probe = await self._container_backend.probe()
        _raise_for_container_backend_selection(run_plan, probe)
        lifecycle_task = create_task(
            run_container_managed_lifecycle(
                self._container_backend,
                run_plan,
                output_contract=output_contract,
                shutdown_requested=self._shutdown_requested(),
            )
        )
        try:
            while not lifecycle_task.done():
                try:
                    await wait_for(shield(lifecycle_task), timeout=0.1)
                except TimeoutError:
                    await self._check_cancelled(run.run_id)
                    continue
            result = await lifecycle_task
        finally:
            if not lifecycle_task.done():
                await _cancel_task(cast(AsyncTask[object], lifecycle_task))
        status = cast(ContainerResultStatus, result.execution.status)
        await self._record_container_event(
            definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            event_type="container_lifecycle_completed",
            plans=plans,
            input_mounts=input_mounts,
            status=status.value,
        )
        if result.execution.status is not ContainerResultStatus.COMPLETED:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.execution_failed",
                        path="container.result",
                        message=(
                            "Container lifecycle did not complete"
                            " successfully."
                        ),
                        hint=(
                            "Inspect sanitized container events and retry"
                            " safely."
                        ),
                    ),
                )
            )
        if (
            result.output is None
            or result.output.decision is not ContainerOutputDecisionType.ACCEPT
            or not result.output.artifacts
        ):
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.output_unsupported",
                        path="container.output",
                        message=(
                            "Task-attempt container execution did not return "
                            "accepted task output artifacts."
                        ),
                        hint=(
                            "Configure the container backend to copy accepted "
                            "task artifacts for this output contract."
                        ),
                    ),
                )
            )
        try:
            output = await task_container_output_artifacts(
                definition,
                tuple(result.output.artifacts),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=self._artifact_store,
            )
        except TaskContainerVerificationError as error:
            raise TaskValidationError(
                (_container_validation_issue(error),)
            ) from error
        await self._record_output_artifacts(
            definition,
            output,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
        )
        return TaskContainerAttemptResult(
            output=output,
            output_artifacts_recorded=True,
        )

    async def _record_container_event(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        event_type: str,
        plans: TaskContainerPlans,
        input_mounts: tuple[dict[str, object], ...],
        status: str,
    ) -> None:
        if not definition.observability.capture_events:
            return
        payload = freeze_task_event_value(
            sanitizer.sanitize_event(
                event_type,
                task_container_event_payload(
                    status=status,
                    plans=plans,
                    input_mounts=input_mounts,
                ),
            ),
        )
        await self._store.append_event(
            run.run_id,
            event_type=event_type,
            category=TaskEventCategory.UNKNOWN,
            payload=payload,
            attempt_id=attempt.attempt_id,
        )

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


async def _close_durable_resume(
    durable_resume: TaskDurableResumeHandle | None,
) -> None:
    if durable_resume is not None:
        await durable_resume.close()


def _is_input_expiry(error: BaseException) -> bool:
    if isinstance(error, KeyboardInterrupt | SystemExit):
        return False
    if isinstance(error, _TaskDurableResumeExpired):
        return True
    if isinstance(error, InputContractError):
        return error.code is InputErrorCode.EXPIRED
    if isinstance(error, BaseExceptionGroup):
        if any(_contains_process_control(child) for child in error.exceptions):
            return False
        return any(_is_input_expiry(child) for child in error.exceptions)
    return False


def _input_expiry_cleanup_error(
    error: BaseException,
    *,
    ignore_cancelled: bool = False,
) -> BaseException | None:
    residuals = _input_expiry_cleanup_errors(
        error,
        ignore_cancelled=ignore_cancelled,
    )
    if not residuals:
        return None
    if len(residuals) == 1:
        return residuals[0]
    return BaseExceptionGroup(
        "durable expiry cleanup failed",
        residuals,
    )


def _input_expiry_cleanup_errors(
    error: BaseException,
    *,
    ignore_cancelled: bool,
) -> tuple[BaseException, ...]:
    if isinstance(error, _TaskDurableResumeExpired):
        if error.cleanup_error is None:
            return ()
        return _input_expiry_cleanup_errors(
            error.cleanup_error,
            ignore_cancelled=ignore_cancelled,
        )
    if (
        isinstance(error, InputContractError)
        and error.code is InputErrorCode.EXPIRED
    ):
        return ()
    if isinstance(error, BaseExceptionGroup):
        return tuple(
            residual
            for child in error.exceptions
            for residual in _input_expiry_cleanup_errors(
                child,
                ignore_cancelled=ignore_cancelled,
            )
        )
    if ignore_cancelled and isinstance(error, CancelledError):
        return ()
    return (error,)


def _contains_process_control(error: BaseException) -> bool:
    if isinstance(error, KeyboardInterrupt | SystemExit):
        return True
    return isinstance(error, BaseExceptionGroup) and any(
        _contains_process_control(child) for child in error.exceptions
    )


def _contains_cancellation(error: BaseException) -> bool:
    if isinstance(error, CancelledError):
        return True
    return isinstance(error, BaseExceptionGroup) and any(
        _contains_cancellation(child) for child in error.exceptions
    )


async def _await_durable_failure_convergence(
    convergence: AsyncTask[TaskWorkerProcessResult],
) -> TaskWorkerProcessResult:
    return await _await_owned_task(convergence)


_OwnedTaskResult = TypeVar("_OwnedTaskResult")


async def _await_owned_task(
    task: AsyncTask[_OwnedTaskResult],
) -> _OwnedTaskResult:
    """Await an owned child while consuming only caller cancellation."""
    owner = current_task()
    while True:
        try:
            return await shield(task)
        except CancelledError:
            if task.cancelled():
                return task.result()
            if task.done():
                if owner is not None:
                    while owner.cancelling():
                        owner.uncancel()
                return task.result()
            if owner is None:  # pragma: no cover
                raise
            while owner.cancelling():
                owner.uncancel()


async def _cancel_task_error(
    task: AsyncTask[object],
) -> BaseException | None:
    if not task.done():
        task.cancel()
    try:
        await task
    except CancelledError:
        return None
    except BaseException as error:
        return error
    return None


async def _cancel_task(task: AsyncTask[object]) -> None:
    if task.done():
        return
    task.cancel()
    with suppress(CancelledError):
        await task


async def _cancel_heartbeat_task(task: AsyncTask[None]) -> None:
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


def _task_skills_identity_from_run(
    run: TaskRun,
) -> Mapping[str, object] | None:
    value = run.request.metadata.get(TASK_SKILLS_METADATA_KEY)
    if isinstance(value, Mapping):
        return value
    return None


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
