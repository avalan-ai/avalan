from asyncio import (
    CancelledError,
    Event,
    TimeoutError,
    all_tasks,
    create_task,
    current_task,
    shield,
    sleep,
    wait_for,
)
from asyncio import (
    Task as AsyncTask,
)
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch

from avalan.agent.continuation import DurableAgentContinuationClaimLease
from avalan.container import (
    ContainerOutputDecisionType,
    ContainerResultStatus,
)
from avalan.interaction import (
    AgentId,
    BranchId,
    CapabilityRevision,
    ConfirmationQuestion,
    ContinuationClaim,
    ContinuationClaimOwnerId,
    ContinuationCompletionCommand,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationRejectionCommand,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ContinuationStoreRevision,
    CreateInteractionCommand,
    DurableContinuationResumeState,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequestId,
    InputRequiredResult,
    InteractionActor,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PortableContinuation,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    QuestionId,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TurnId,
    create_input_request,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.skill import (
    SkillDiagnosticCode,
    SkillDiagnosticInfo,
    SkillObservabilitySettings,
    SkillReadLimits,
    SkillRegistry,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
)
from avalan.task import (
    EncryptedPrivacyValue,
    ObservabilitySinkHealth,
    ObservabilitySinkType,
    PrivacyAction,
    PrivacySanitizer,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactState,
    TaskAttempt,
    TaskAttemptSegment,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskDefinition,
    TaskDurableExpiredReentryCommit,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputParseError,
    TaskPrivacyPolicy,
    TaskProviderReferenceKind,
    TaskProviderStructuredOutputError,
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueConflictError,
    TaskQueueDepth,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueReentry,
    TaskQueueRetry,
    TaskQueueSubmission,
    TaskQueueSuspension,
    TaskRetryPolicy,
    TaskRun,
    TaskRunPolicy,
    TaskRunState,
    TaskStoreConflictError,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskTargetRunnerRegistry,
    TaskTargetSuspended,
    TaskTargetType,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
    TaskWorkerShutdown,
    UsageSource,
    UsageTotals,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)
from avalan.task import worker as worker_module
from avalan.task.container import TaskContainerVerificationError
from avalan.task.context import (
    TaskDurableResumeHandle,
    TaskEventListener,
    TaskEventListenerRegistration,
)
from avalan.task.error import classify_task_error
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.resume import (
    TaskResumeClaimLeaseManager,
    task_resume_result_digest,
)
from avalan.task.runner import (
    TaskContainerAttemptResult,
    TaskExecutableInputFileEntry,
    task_execution_file_entries_value,
)
from avalan.task.settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
    TaskDurableResumeSuccess,
    task_durable_resume_settlement_digest,
)
from avalan.task.skills import (
    build_task_skill_registry,
    task_definition_with_skills_identity,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.worker import (
    TaskWorkerError,
    TaskWorkerProcessResult,
    _await_owned_task,
    _target_runner,
    _TaskClaimCleanupOwner,
    _TaskDurableResumeOwner,
    _TaskResumeClaimLeaseBridge,
    _TaskWorkerShutdownRequested,
    _utc_now,
    _worker_id,
)


class FakeTarget(TaskTargetRunner):
    def __init__(self, output: object = "done") -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        await context.check_cancelled()
        return completed_task_target_outcome(self.output)


class MultiCallUsageResponse:
    def __init__(self, *responses: object) -> None:
        self.usage_responses = responses


class UsageTextOutput(str):
    _usage: object | None
    _usage_responses: tuple[object, ...]

    def __new__(
        cls,
        value: str,
        *,
        usage: object | None = None,
        usage_responses: tuple[object, ...] = (),
    ) -> "UsageTextOutput":
        output = str.__new__(cls, value)
        output._usage = usage
        output._usage_responses = usage_responses
        return output

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def usage_responses(self) -> tuple[object, ...]:
        return self._usage_responses


class RecordingUsageSink:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.usage_totals: list[UsageTotals] = []

    async def record_event(self, event: object) -> None:
        self.events.append(event)

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.usage_totals.append(totals)

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name="recording",
            event_count=len(self.events),
            usage_count=len(self.usage_totals),
        )

    @property
    def usage_event_count(self) -> int:
        return sum(
            1
            for event in self.events
            if type(event).__name__ == "SanitizedTaskUsageEvent"
        )


class WaitingTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.started = Event()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        self.started.set()
        while True:
            await sleep(0)
            await context.check_cancelled()


class PassiveWaitingTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.started = Event()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        self.started.set()
        while True:
            await sleep(0)


class ShutdownReturningTarget(FakeTarget):
    def __init__(self, shutdown: TaskWorkerShutdown) -> None:
        super().__init__("done")
        self.shutdown = shutdown

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        self.shutdown.request()
        return completed_task_target_outcome("safe output")


class DelayedWaitShutdown(TaskWorkerShutdown):
    async def wait(self) -> None:
        await sleep(1)


class InvalidTarget(FakeTarget):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return (
            TaskValidationIssue(
                code="target.invalid",
                path="execution",
                message="invalid target",
                hint="fix target",
                category=TaskValidationCategory.VALUE,
            ),
        )


class ArtifactOutputTarget(FakeTarget):
    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        return completed_task_target_outcome(
            TaskArtifactRef(
                artifact_id="artifact-output-1",
                store="local",
                storage_key="output.txt",
                media_type="text/plain",
                size_bytes=7,
                metadata={"filename": "private-output.txt"},
            )
        )


class UsageTarget(FakeTarget):
    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        await context.observe_usage(
            SimpleNamespace(input_token_count=3, output_token_count=5)
        )
        return completed_task_target_outcome(self.output)


class SuspendingThenCompletingTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        await context.observe_usage(
            SimpleNamespace(input_token_count=1, output_token_count=2)
        )
        if len(self.contexts) == 1:
            return suspended_task_target_outcome(
                InputRequiredResult(
                    request_id=InputRequestId("request-1"),
                    continuation_id=ContinuationId("continuation-1"),
                    detached_resumption_available=True,
                ),
                checkpoint_id="checkpoint-1",
            )
        return completed_task_target_outcome("safe output")


class DurableSuspendingTarget(TaskTargetRunner):
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.contexts: list[TaskTargetContext] = []
        self.suspension: DurableInteractionSuspension | None = None

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        del definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        self.suspension = _durable_suspension(context, self.now)
        request = self.suspension.command.request
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            checkpoint_id=str(request.continuation_id),
            durable=self.suspension,
        )


class DurableResumableTarget(DurableSuspendingTarget):
    def __init__(
        self,
        now: datetime,
        *,
        error_after_dispatch: BaseException | None = None,
        durable_resume_support: object = True,
    ) -> None:
        super().__init__(now)
        self.error_after_dispatch = error_after_dispatch
        self.durable_resume_support = durable_resume_support
        self.durable_resume_support_calls = 0
        self.resume_contexts: list[TaskTargetContext] = []

    def supports_durable_resume(self, target_type: TaskTargetType) -> bool:
        self.durable_resume_support_calls += 1
        if target_type is not TaskTargetType.AGENT:
            return False
        return cast(bool, self.durable_resume_support)

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self.resume_contexts.append(context)
        output = await durable_resume.dispatch()
        if self.error_after_dispatch is not None:
            raise self.error_after_dispatch
        return completed_task_target_outcome(output)


class DelayedDurableResumableTarget(DurableResumableTarget):
    def __init__(self, now: datetime) -> None:
        super().__init__(now)
        self.resume_started = Event()
        self.resume_proceed = Event()

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self.resume_contexts.append(context)
        self.resume_started.set()
        await self.resume_proceed.wait()
        return completed_task_target_outcome(await durable_resume.dispatch())


class StatefulDurableResumableTarget(DurableResumableTarget):
    def supports_durable_resume(self, target_type: TaskTargetType) -> bool:
        self.durable_resume_support_calls += 1
        if target_type is not TaskTargetType.AGENT:
            return False
        return self.durable_resume_support_calls == 1


class SpoofDurableResumePreparer(DurableResumableTarget):
    def __init__(self, now: datetime) -> None:
        super().__init__(now, durable_resume_support=False)
        self.prepare_calls = 0

    def prepare_durable_resume(self, target_type: TaskTargetType) -> object:
        del target_type
        self.prepare_calls += 1
        return self


class ResuspendingDurableTarget(DurableResumableTarget):
    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self.resume_contexts.append(context)
        await durable_resume.dispatch()
        self.suspension = _durable_suspension(context, self.now)
        request = self.suspension.command.request
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            checkpoint_id="successor-checkpoint",
            durable=self.suspension,
        )


class FakeTaskEventListenerRegistration:
    def close(self) -> None:
        """Close the no-op worker listener registration."""


class FakeDurableResumeHandle:
    def __init__(
        self,
        *,
        output: object = "safe output",
        dispatch_error: BaseException | None = None,
        error_state: DurableContinuationResumeState = (
            DurableContinuationResumeState.AMBIGUOUS
        ),
        dispatch_started: Event | None = None,
        dispatch_proceed: Event | None = None,
    ) -> None:
        self.output = output
        self.dispatch_error = dispatch_error
        self.error_state = error_state
        self.dispatch_started = dispatch_started
        self.dispatch_proceed = dispatch_proceed
        self.state = DurableContinuationResumeState.ADMITTED
        self.dispatch_calls = 0
        self.interrupt_calls = 0
        self.release_calls = 0
        self.close_calls = 0
        self.completed_digest: str | None = None
        self.continuation_id = ContinuationId("continuation-1")
        self.owner_id = ContinuationClaimOwnerId("owner-1")

    def register_event_listener(
        self,
        listener: TaskEventListener,
    ) -> TaskEventListenerRegistration:
        assert callable(listener)
        return FakeTaskEventListenerRegistration()

    async def dispatch(self) -> object:
        self.dispatch_calls += 1
        self.state = DurableContinuationResumeState.DISPATCHING
        if self.dispatch_started is not None:
            self.dispatch_started.set()
        if self.dispatch_proceed is not None:
            await self.dispatch_proceed.wait()
        if self.dispatch_error is not None:
            self.state = self.error_state
            raise self.dispatch_error
        self.state = DurableContinuationResumeState.DISPATCHED
        return self.output

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        return self.state

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        self.interrupt_calls += 1
        if self.state is DurableContinuationResumeState.ADMITTED:
            self.state = DurableContinuationResumeState.RELEASED
        elif self.state is DurableContinuationResumeState.DISPATCHING:
            self.state = DurableContinuationResumeState.AMBIGUOUS
        return self.state

    async def complete_output(self, output: object) -> None:
        self.completed_digest = task_resume_result_digest(output)
        self.state = DurableContinuationResumeState.COMPLETED

    def completion_command_for_output(
        self,
        output: object,
    ) -> ContinuationCompletionCommand:
        del output
        return self._completion_command("0" * 64)

    def completion_command_for_settlement(
        self,
        settlement: TaskDurableResumeSettlement,
    ) -> ContinuationCompletionCommand:
        return self._completion_command(
            task_durable_resume_settlement_digest(settlement)
        )

    def completed_completion_command(
        self,
    ) -> ContinuationCompletionCommand:
        if (
            self.state is not DurableContinuationResumeState.COMPLETED
            or self.completed_digest is None
        ):
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "resume.completed_completion",
                "continuation is not durably completed",
            )
        return self._completion_command(self.completed_digest)

    def completion_command_for_suspension(
        self,
        *,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
    ) -> ContinuationCompletionCommand:
        del request_id, continuation_id, checkpoint_id
        return self._completion_command("1" * 64)

    def rejection_command_for_settlement(
        self,
        failure: TaskDurableResumeFailure,
    ) -> ContinuationRejectionCommand:
        command = self._completion_command(
            task_durable_resume_settlement_digest(failure)
        )
        return ContinuationRejectionCommand(
            continuation_id=command.continuation_id,
            expected_store_revision=command.expected_store_revision,
            owner_id=command.owner_id,
            fencing_token=command.fencing_token,
            result_digest=command.result_digest,
        )

    async def release(self) -> None:
        self.release_calls += 1
        assert self.state is DurableContinuationResumeState.ADMITTED
        self.state = DurableContinuationResumeState.RELEASED

    async def release_if_pre_dispatch(self) -> bool:
        self.release_calls += 1
        if self.state is DurableContinuationResumeState.RELEASED:
            return True
        if self.state is not DurableContinuationResumeState.ADMITTED:
            return False
        self.state = DurableContinuationResumeState.RELEASED
        return True

    async def close(self) -> None:
        self.close_calls += 1

    def _completion_command(
        self,
        digest: str,
    ) -> ContinuationCompletionCommand:
        return ContinuationCompletionCommand(
            continuation_id=self.continuation_id,
            expected_store_revision=ContinuationStoreRevision(2),
            owner_id=self.owner_id,
            fencing_token=ContinuationFencingToken(1),
            result_digest=digest,
        )


class CompletedDigestPinnedDurableResumeHandle(FakeDurableResumeHandle):
    """Model a real admission completed with its provider output digest."""

    def __init__(self, *, output: object = "safe output") -> None:
        super().__init__(output=output)
        self.pinned_digest: str | None = None
        self.settlement_command_calls = 0

    async def dispatch(self) -> object:
        output = await super().dispatch()
        await self.complete_output(output)
        return output

    async def complete_output(self, output: object) -> None:
        self.pinned_digest = task_resume_result_digest(output)
        self.state = DurableContinuationResumeState.COMPLETED

    def completion_command_for_output(
        self,
        output: object,
    ) -> ContinuationCompletionCommand:
        digest = task_resume_result_digest(output)
        if digest != self.pinned_digest:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "resume.result_digest",
                "completion replay changed the result digest",
            )
        return self._completion_command(digest)

    def completion_command_for_settlement(
        self,
        settlement: TaskDurableResumeSettlement,
    ) -> ContinuationCompletionCommand:
        self.settlement_command_calls += 1
        digest = task_durable_resume_settlement_digest(settlement)
        if digest != self.pinned_digest:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "resume.result_digest",
                "completion replay changed the result digest",
            )
        return self._completion_command(digest)

    def completed_completion_command(
        self,
    ) -> ContinuationCompletionCommand:
        if (
            self.state is not DurableContinuationResumeState.COMPLETED
            or self.pinned_digest is None
        ):
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "resume.completed_completion",
                "continuation is not durably completed",
            )
        return self._completion_command(self.pinned_digest)


class ShieldedBlockingDurableResumeHandle(FakeDurableResumeHandle):
    def __init__(self, *, dispatch_wins_interruption: bool = False) -> None:
        super().__init__()
        self.dispatch_wins_interruption = dispatch_wins_interruption
        self.provider_started = Event()
        self.provider_proceed = Event()
        self.dispatch_wait_calls = 0
        self.provider_cancellations = 0
        self.provider_completions = 0
        self._dispatch_task: AsyncTask[object] | None = None

    async def dispatch(self) -> object:
        self.dispatch_calls += 1
        if self._dispatch_task is not None:
            raise AssertionError("provider dispatch was duplicated")
        self._dispatch_task = create_task(self._dispatch_provider())
        return await shield(self._dispatch_task)

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        self.dispatch_wait_calls += 1
        if self.interrupt_calls == 0:
            raise AssertionError(
                "dispatch settlement waited before provider interruption"
            )
        if self.state in {
            DurableContinuationResumeState.ADMITTED,
            DurableContinuationResumeState.DISPATCHING,
        }:
            raise AssertionError("provider dispatch remains unsettled")
        return self.state

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        self.interrupt_calls += 1
        task = self._dispatch_task
        if self.state is DurableContinuationResumeState.ADMITTED:
            self.state = DurableContinuationResumeState.RELEASED
        elif self.state is DurableContinuationResumeState.DISPATCHING:
            assert task is not None
            if self.dispatch_wins_interruption:
                self.provider_proceed.set()
            else:
                task.cancel()
            try:
                await task
            except CancelledError:
                pass
        return self.state

    async def close(self) -> None:
        self.close_calls += 1
        task = self._dispatch_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except CancelledError:
                pass

    async def _dispatch_provider(self) -> object:
        self.state = DurableContinuationResumeState.DISPATCHING
        self.provider_started.set()
        try:
            await self.provider_proceed.wait()
        except CancelledError:
            self.provider_cancellations += 1
            self.state = DurableContinuationResumeState.AMBIGUOUS
            raise
        self.provider_completions += 1
        self.state = DurableContinuationResumeState.DISPATCHED
        return self.output


class GatedFailureConvergenceDurableResumeHandle(FakeDurableResumeHandle):
    def __init__(
        self,
        terminal_state: DurableContinuationResumeState,
    ) -> None:
        super().__init__()
        self.terminal_state = terminal_state
        self.convergence_dispatch_started = Event()
        self.dispatch_blocked = Event()
        self.interrupt_started = Event()
        self.settlement_started = Event()
        self.close_started = Event()
        self.close_proceed = Event()
        self.close_completed = 0
        self.dispatch_wait_calls = 0

    async def dispatch(self) -> object:
        self.dispatch_calls += 1
        self.state = DurableContinuationResumeState.DISPATCHING
        self.convergence_dispatch_started.set()
        await self.dispatch_blocked.wait()
        raise AssertionError("blocked dispatch unexpectedly resumed")

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        self.interrupt_calls += 1
        self.interrupt_started.set()
        self.state = self.terminal_state
        return self.state

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        self.dispatch_wait_calls += 1
        self.settlement_started.set()
        return self.state

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.close_proceed.wait()
        self.close_completed += 1


class FakeDurableContinuationClaimLease:
    def __init__(
        self,
        *,
        expires_at: datetime,
        noop_after_expiry: Callable[[], bool] | None = None,
    ) -> None:
        self.expires_at = expires_at
        self.noop_after_expiry = noop_after_expiry
        self.lease_expires_at: datetime | None = None
        self.renewals: list[tuple[datetime, datetime]] = []
        self.noop_renewals: list[datetime] = []

    async def renew(
        self,
        lease_expires_at: datetime,
        *,
        now: datetime,
    ) -> bool:
        if now >= self.expires_at:
            if self.noop_after_expiry is not None and self.noop_after_expiry():
                self.noop_renewals.append(now)
                return False
            raise InputValidationError(
                InputErrorCode.EXPIRED,
                "continuation",
                "continuation expired",
            )
        effective = min(lease_expires_at, self.expires_at)
        if effective <= now:
            raise RuntimeError("continuation claim lease expired")
        self.lease_expires_at = effective
        self.renewals.append((now, effective))
        return True


class FakeDurableResumeCoordinator:
    def __init__(
        self,
        admission: FakeDurableResumeHandle,
        *,
        claim_lease: FakeDurableContinuationClaimLease | None = None,
        admission_started: Event | None = None,
        admission_proceed: Event | None = None,
    ) -> None:
        self.admission = admission
        self.claim_lease = claim_lease
        self.admission_started = admission_started
        self.admission_proceed = admission_proceed
        self.claim_lease_manager: TaskResumeClaimLeaseManager | None = None
        self.calls: list[tuple[TaskQueueClaim, TaskAttemptSegmentState]] = []

    async def admit(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment | None,
        *,
        claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
    ) -> FakeDurableResumeHandle:
        assert previous_segment is not None
        assert previous_segment.continuation_id is not None
        assert claim.queue_item.claim_token is not None
        self.admission.continuation_id = ContinuationId(
            previous_segment.continuation_id
        )
        self.admission.owner_id = ContinuationClaimOwnerId(
            claim.queue_item.claim_token
        )
        self.calls.append((claim, previous_segment.state))
        if self.claim_lease is not None:
            assert claim_lease_manager is not None
            self.claim_lease_manager = claim_lease_manager
            current_lease = (
                await claim_lease_manager.current_lease_expires_at()
            )
            self.claim_lease.lease_expires_at = current_lease
            await claim_lease_manager.bind(
                cast(
                    DurableAgentContinuationClaimLease,
                    self.claim_lease,
                )
            )
        if self.admission_started is not None:
            self.admission_started.set()
        if self.admission_proceed is not None:
            await self.admission_proceed.wait()
        return self.admission


def _durable_suspension(
    context: TaskTargetContext,
    now: datetime,
) -> DurableInteractionSuspension:
    definition = ExecutionDefinitionRef(
        agent_definition_locator="agent:worker-test",
        agent_definition_revision="agent-r1",
        operation_id="operation-1",
        operation_index=0,
        model_config_reference="model-config-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )
    origin = ExecutionOrigin(
        run_id=RunId(context.execution.run_id),
        turn_id=TurnId("turn-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        model_call_id=ModelCallId("model-call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=definition,
        principal=PrincipalScope(),
    )
    actor = InteractionActor(principal=origin.principal)
    request = create_input_request(
        request_id=InputRequestId("durable-request-1"),
        continuation_id=ContinuationId("durable-continuation-1"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Need one durable answer.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("continue"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=now,
    )
    binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("worker-model"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )
    provider_key = ProviderIdempotencyKey("provider-dispatch-1")
    snapshot = ContinuationSnapshot(
        snapshot_kind="worker.provider-response",
        revision_binding=binding,
        model_call_id=origin.model_call_id,
        provider_idempotency_key=provider_key,
        payload={
            "reserved_capability_call_id": "input-call-1",
            "replay_items": (),
        },
    )
    continuation = PortableContinuation(
        continuation_id=request.continuation_id,
        request_id=request.request_id,
        origin=origin,
        provider_call_id=origin.model_call_id,
        provider_call_correlation_id="input-call-1",
        definition=definition,
        operation_cursor=0,
        generation_settings={},
        transcript=(),
        observations=(),
        revision_binding=binding,
        interaction_count=1,
        tool_loop_count=0,
        stream_sequence=4,
        state_revision=StateRevision(1),
        store_revision=ContinuationStoreRevision(0),
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(days=1),
        claim=ContinuationClaim(),
        fencing_token=ContinuationFencingToken(0),
        provider_snapshot=snapshot,
    )
    return DurableInteractionSuspension(
        command=CreateInteractionCommand(
            actor=actor,
            request=request,
        ),
        continuation=continuation,
    )


class PartiallyObservedUsageWrapperTarget(FakeTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.first_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 4,
                "cached_input_tokens": 1,
                "output_tokens": 6,
                "total_tokens": 10,
                "raw_response_id": "private-first-response",
            },
        )
        self.second_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
                "reasoning_tokens": 3,
                "total_tokens": 12,
                "raw_response_id": "private-second-response",
            },
        )
        self.malformed_response = SimpleNamespace(
            provider_family="openai",
            usage={
                "input_tokens": "private prompt",
                "output_tokens": -1,
                "total_tokens": True,
                "raw_response_body": "private provider body",
            },
        )

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.contexts.append(context)
        await context.observe_usage(self.first_response)
        return completed_task_target_outcome(
            UsageTextOutput(
                "done",
                usage_responses=(
                    self.first_response,
                    self.second_response,
                    self.malformed_response,
                ),
            )
        )


class StaticEncryptionProvider:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        _ = purpose
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted:" + value,
            key_id=key_id or "raw-value",
            algorithm="test-aead",
            metadata=context,
        )

    def decrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        algorithm: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> bytes:
        _ = purpose, key_id, algorithm, context
        prefix = b"encrypted:"
        assert value.startswith(prefix)
        return value[len(prefix) :]


class FakeQueue:
    def __init__(self, store: InMemoryTaskStore, now: datetime) -> None:
        self.store = store
        self.now = now
        self.item: TaskQueueItem | None = None
        self.completed: TaskQueueCompletion | None = None
        self.retried: TaskQueueRetry | None = None
        self.abandoned: TaskQueueAbandonment | None = None
        self.suspended: TaskQueueSuspension | None = None
        self.heartbeats: list[datetime] = []
        self.heartbeat_error: BaseException | None = None
        self.heartbeat_shutdown: TaskWorkerShutdown | None = None
        self.abandon_after_claim = False
        self.require_durable_coordinator = False
        self.durable_commit_active = False
        self.suspend_claim_calls = 0

    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        raise AssertionError("enqueue_run should not be used")

    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem:
        raise AssertionError("enqueue should not be used")

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueClaim | None:
        if (
            self.item is None
            or self.item.state != TaskQueueItemState.AVAILABLE
        ):
            return None
        run = await self.store.assign_claim(
            self.item.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            reason="claimed",
            metadata=metadata,
        )
        claim_token = run.claim.claim_token if run.claim else ""
        if run.last_attempt_id is not None:
            previous_attempt = await self.store.get_attempt(
                run.last_attempt_id
            )
        else:
            previous_attempt = None
        attempt = (
            previous_attempt
            if previous_attempt is not None
            and previous_attempt.state == TaskAttemptState.SUSPENDED
            else await self.store.create_attempt(
                run.run_id,
                claim_token=claim_token,
                metadata=metadata,
            )
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.CLAIMED,
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
            claimed_at=run.claim.claimed_at if run.claim else None,
            lease_expires_at=run.claim.lease_expires_at if run.claim else None,
            worker_id=worker_id,
            claim_token=claim_token,
            heartbeat_at=run.claim.heartbeat_at if run.claim else None,
            metadata=metadata or {},
        )
        claim = TaskQueueClaim(queue_item=self.item, run=run, attempt=attempt)
        if self.abandon_after_claim:
            await self.abandon(
                self.item.queue_item_id,
                claim_token=claim_token,
                max_attempts=2,
                now=now or self.now,
                metadata=metadata,
            )
        return claim

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem:
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        assert self.item.claim_token == claim_token
        heartbeat_at = now or self.now
        self.heartbeats.append(heartbeat_at)
        if self.heartbeat_shutdown is not None:
            self.heartbeat_shutdown.request()
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=self.item.run_id,
            queue_name=self.item.queue_name,
            state=self.item.state,
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=heartbeat_at,
            run_state=self.item.run_state,
            claimed_at=self.item.claimed_at,
            lease_expires_at=lease_expires_at,
            worker_id=self.item.worker_id,
            claim_token=self.item.claim_token,
            heartbeat_at=heartbeat_at,
            metadata=self.item.metadata,
        )
        return self.item

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
        self.suspend_claim_calls += 1
        if self.require_durable_coordinator:
            assert self.durable_commit_active
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        assert self.item.claim_token == claim_token
        current_run = await self.store.get_run(self.item.run_id)
        attempt = await self.store.get_attempt(
            current_run.last_attempt_id or ""
        )
        segment = await self.store.transition_attempt_segment(
            segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.SUSPENDED,
            reason="input_required",
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            claim_token=claim_token,
            metadata=metadata,
        )
        attempt = await self.store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.SUSPENDED,
            reason="input_required",
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.INPUT_REQUIRED,
            reason="input_required",
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=TaskQueueItemState.SUSPENDED,
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
            metadata=metadata or {},
        )
        self.suspended = TaskQueueSuspension(
            queue_item=self.item,
            run=run,
            attempt=attempt,
            segment=segment,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
        )
        return self.suspended

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
        assert self.item is not None
        assert self.item.run_id == run_id
        assert self.item.state == TaskQueueItemState.SUSPENDED
        assert resolution_revision > 0
        run = await self.store.get_run(run_id)
        attempt = await self.store.get_attempt(run.last_attempt_id or "")
        segments = await self.store.list_attempt_segments(attempt.attempt_id)
        previous_segment = segments[-1]
        assert previous_segment.request_id == request_id
        assert previous_segment.continuation_id == continuation_id
        run = await self.store.transition_run(
            run_id,
            from_states={TaskRunState.INPUT_REQUIRED},
            to_state=TaskRunState.QUEUED,
            reason="input_resolved",
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=self.item.priority,
            available_at=now or self.now,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
            metadata=metadata or {},
        )
        return TaskQueueReentry(
            queue_item=self.item,
            run=run,
            attempt=attempt,
            previous_segment=previous_segment,
        )

    async def complete(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        result: TaskExecutionResult | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        assert self.item is not None
        current_run = await self.store.get_run(self.item.run_id)
        attempt_id = current_run.last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=attempt_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={current_run.state},
            to_state=run_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=(
                TaskQueueItemState.DONE
                if run_state == TaskRunState.SUCCEEDED
                else TaskQueueItemState.DEAD
            ),
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.completed = TaskQueueCompletion(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.completed

    async def retry(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        result: TaskExecutionResult,
        available_at: datetime,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueRetry:
        assert self.item is not None
        attempt_id = (
            await self.store.get_run(self.item.run_id)
        ).last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="attempt_retry",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.QUEUED,
            reason="attempt_retry",
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=self.item.priority,
            available_at=available_at,
            attempts=self.item.attempts + 1,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.retried = TaskQueueRetry(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.retried

    async def abandon(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueAbandonment:
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        current_run = await self.store.get_run(self.item.run_id)
        attempt_id = current_run.last_attempt_id
        attempt = await self.store.transition_attempt(
            attempt_id or "",
            from_states={
                TaskAttemptState.CREATED,
                TaskAttemptState.RUNNING,
            },
            to_state=TaskAttemptState.ABANDONED,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        cancel_requested = current_run.state == TaskRunState.CANCEL_REQUESTED
        retryable = (
            attempt.attempt_number < max_attempts and not cancel_requested
        )
        run_state = (
            TaskRunState.CANCELLED
            if cancel_requested
            else TaskRunState.QUEUED if retryable else TaskRunState.FAILED
        )
        run = await self.store.transition_run(
            self.item.run_id,
            from_states={current_run.state},
            to_state=run_state,
            reason="abandoned",
            claim_token=claim_token,
            metadata=metadata,
        )
        if retryable:
            run = replace(run, claim=None)
            self.store._runs[run.run_id] = run
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=run.run_id,
            queue_name=self.item.queue_name,
            state=(
                TaskQueueItemState.AVAILABLE
                if retryable
                else TaskQueueItemState.DEAD
            ),
            priority=self.item.priority,
            available_at=now or self.now,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.now,
            run_state=run.state,
        )
        self.abandoned = TaskQueueAbandonment(
            queue_item=self.item,
            run=run,
            attempt=attempt,
        )
        return self.abandoned

    async def abandon_expired(
        self,
        queue_name: str,
        *,
        max_attempts: int,
        limit: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[TaskQueueAbandonment, ...]:
        return ()

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]:
        return ()

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth:
        return TaskQueueDepth(
            queue_name=queue_name,
            available=0,
            scheduled=0,
            claimed=0,
            dead=0,
            cancel_requested=0,
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        return TaskQueueHealth(
            queue_name=queue_name,
            depth=await self.depth(queue_name),
            checked_at=now or self.now,
        )


@dataclass(frozen=True, slots=True)
class _FakeDurableSuspensionCommit:
    suspension: TaskQueueSuspension


@dataclass(frozen=True, slots=True)
class _FakeDurableSettlementCommit:
    completion: TaskQueueCompletion


class FakeDurableSuspensionCoordinator:
    def __init__(self, queue: FakeQueue) -> None:
        self.queue = queue
        self.calls: list[
            tuple[CreateInteractionCommand, PortableContinuation]
        ] = []
        self.failed_reentries: list[TaskQueueCompletion] = []
        self.rejected_admissions: list[TaskDurableResumeFailure] = []
        self.settlements: list[TaskDurableResumeSettlement] = []
        self.completed_terminalizations: list[
            TaskDurableResumeFailure | TaskDurableResumeCancellation
        ] = []
        self._completed_terminalization_replays: dict[
            str,
            tuple[
                tuple[object, ...],
                _FakeDurableSettlementCommit,
            ],
        ] = {}
        self.ambiguities: list[TaskDurableResumeFailure] = []
        self.claimed_releases: list[TaskQueueReentry] = []
        self.running_releases: list[TaskQueueReentry] = []
        self.expired_commit: TaskDurableExpiredReentryCommit | None = None
        self.expired_calls: list[tuple[str, str, str]] = []
        self.convergence_commit_started: Event | None = None
        self.convergence_commit_proceed: Event | None = None
        self.convergence_commit_calls = 0

    async def _gate_convergence_commit(self) -> None:
        self.convergence_commit_calls += 1
        if self.convergence_commit_started is not None:
            self.convergence_commit_started.set()
        if self.convergence_commit_proceed is not None:
            await self.convergence_commit_proceed.wait()

    async def create_and_suspend(
        self,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        checkpoint_id: str | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSuspensionCommit:
        assert command.request.origin.run_id == RunId(task_run_id)
        self.calls.append((command, continuation))
        self.queue.durable_commit_active = True
        try:
            suspension = await self.queue.suspend_claim(
                queue_item_id,
                claim_token=claim_token,
                segment_id=segment_id,
                request_id=str(command.request.request_id),
                continuation_id=str(command.request.continuation_id),
                checkpoint_id=checkpoint_id,
                now=now,
                metadata=metadata,
            )
        finally:
            self.queue.durable_commit_active = False
        return _FakeDurableSuspensionCommit(suspension=suspension)

    async def complete_and_resuspend(
        self,
        completion: ContinuationCompletionCommand,
        command: CreateInteractionCommand,
        continuation: PortableContinuation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSuspensionCommit:
        del completion
        return await self.create_and_suspend(
            command,
            continuation,
            queue_item_id=queue_item_id,
            claim_token=claim_token,
            segment_id=segment_id,
            task_run_id=task_run_id,
            checkpoint_id=checkpoint_id,
            now=now,
            metadata=metadata,
        )

    async def release_claimed_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry:
        item = self.queue.item
        assert item is not None
        assert item.queue_item_id == queue_item_id
        assert item.run_id == task_run_id
        assert item.claim_token == claim_token
        run = await self.queue.store.get_run(task_run_id)
        attempt = await self.queue.store.get_attempt(run.last_attempt_id or "")
        segments = await self.queue.store.list_attempt_segments(
            attempt.attempt_id
        )
        previous_segment = segments[-1]
        assert previous_segment.request_id == request_id
        assert previous_segment.continuation_id == continuation_id
        assert previous_segment.checkpoint_id == checkpoint_id
        run = replace(
            run,
            state=TaskRunState.QUEUED,
            claim=None,
            updated_at=now or self.queue.now,
            metadata=metadata or {},
        )
        self.queue.store._runs[run.run_id] = run
        self.queue.item = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=run.run_id,
            queue_name=item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=item.priority,
            available_at=now or self.queue.now,
            attempts=item.attempts,
            created_at=item.created_at,
            updated_at=now or self.queue.now,
            run_state=run.state,
            metadata=metadata or {},
        )
        reentry = TaskQueueReentry(
            queue_item=self.queue.item,
            run=run,
            attempt=attempt,
            previous_segment=previous_segment,
        )
        self.claimed_releases.append(reentry)
        return reentry

    async def fail_claimed_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str | None,
        continuation_id: str | None,
        checkpoint_id: str | None,
        result: TaskExecutionResult,
        reason: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        del request_id, continuation_id, checkpoint_id, reason
        item = self.queue.item
        assert item is not None
        assert item.queue_item_id == queue_item_id
        assert item.run_id == task_run_id
        assert item.claim_token == claim_token
        run = await self.queue.store.get_run(task_run_id)
        attempt = await self.queue.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.SUSPENDED},
            to_state=TaskAttemptState.FAILED,
            reason="resume_setup_failed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.queue.store.transition_run(
            task_run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.FAILED,
            reason="resume_setup_failed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.queue.item = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=run.run_id,
            queue_name=item.queue_name,
            state=TaskQueueItemState.DEAD,
            priority=item.priority,
            available_at=item.available_at,
            attempts=item.attempts,
            created_at=item.created_at,
            updated_at=now or self.queue.now,
            run_state=run.state,
            metadata=metadata or {},
        )
        completion = TaskQueueCompletion(
            queue_item=self.queue.item,
            run=run,
            attempt=attempt,
        )
        self.failed_reentries.append(completion)
        return completion

    async def settle_resume(
        self,
        completion: ContinuationCompletionCommand,
        settlement: TaskDurableResumeSettlement,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSettlementCommit:
        await self._gate_convergence_commit()
        del completion
        item = self.queue.item
        assert item is not None
        assert item.queue_item_id == queue_item_id
        assert item.run_id == task_run_id
        segment_state = (
            TaskAttemptSegmentState.SUCCEEDED
            if type(settlement) is TaskDurableResumeSuccess
            else (
                TaskAttemptSegmentState.ABANDONED
                if type(settlement) is TaskDurableResumeCancellation
                else TaskAttemptSegmentState.FAILED
            )
        )
        await self.queue.store.transition_attempt_segment(
            segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=segment_state,
            reason="durable_resume_settled",
            claim_token=claim_token,
            metadata=metadata,
        )
        if type(settlement) is TaskDurableResumeCancellation:
            await self.queue.store.transition_run(
                task_run_id,
                from_states={TaskRunState.RUNNING},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="cancel_requested",
                claim_token=claim_token,
                metadata=metadata,
            )
        task_completion = await self.queue.complete(
            queue_item_id,
            claim_token=claim_token,
            run_state=(
                TaskRunState.SUCCEEDED
                if type(settlement) is TaskDurableResumeSuccess
                else (
                    TaskRunState.CANCELLED
                    if type(settlement) is TaskDurableResumeCancellation
                    else TaskRunState.FAILED
                )
            ),
            attempt_state=(
                TaskAttemptState.SUCCEEDED
                if type(settlement) is TaskDurableResumeSuccess
                else TaskAttemptState.FAILED
            ),
            result=settlement.result,
            now=now,
            metadata=metadata,
        )
        self.settlements.append(settlement)
        return _FakeDurableSettlementCommit(completion=task_completion)

    async def terminalize_completed_resume(
        self,
        completion: ContinuationCompletionCommand,
        settlement: TaskDurableResumeFailure | TaskDurableResumeCancellation,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSettlementCommit:
        await self._gate_convergence_commit()
        identity = (
            completion,
            settlement,
            claim_token,
            segment_id,
            task_run_id,
            request_id,
            checkpoint_id,
        )
        replay = self._completed_terminalization_replays.get(queue_item_id)
        if replay is not None:
            if replay[0] != identity:
                raise TaskQueueConflictError(
                    "completed provider task replay changed identity"
                )
            return replay[1]
        if type(settlement) not in {
            TaskDurableResumeFailure,
            TaskDurableResumeCancellation,
        }:
            raise TaskQueueConflictError(
                "completed provider task cannot settle successfully"
            )
        item = self.queue.item
        if (
            item is None
            or item.queue_item_id != queue_item_id
            or item.run_id != task_run_id
            or item.state is not TaskQueueItemState.CLAIMED
            or item.claim_token != claim_token
            or completion.owner_id != claim_token
        ):
            raise TaskQueueConflictError(
                "completed provider task claim did not match"
            )
        run = await self.queue.store.get_run(task_run_id)
        if (
            run.state is not TaskRunState.RUNNING
            or run.claim is None
            or run.claim.claim_token != claim_token
            or run.last_attempt_id is None
        ):
            raise TaskQueueConflictError(
                "completed provider task run did not match"
            )
        attempt = await self.queue.store.get_attempt(run.last_attempt_id)
        segments = await self.queue.store.list_attempt_segments(
            attempt.attempt_id
        )
        active = next(
            (
                candidate
                for candidate in segments
                if candidate.segment_id == segment_id
            ),
            None,
        )
        previous = (
            next(
                (
                    candidate
                    for candidate in segments
                    if active is not None
                    and candidate.segment_id == active.resumed_from_segment_id
                ),
                None,
            )
            if active is not None
            else None
        )
        if (
            attempt.state is not TaskAttemptState.RUNNING
            or attempt.context.claim is None
            or attempt.context.claim.claim_token != claim_token
            or active is None
            or active.state is not TaskAttemptSegmentState.RUNNING
            or active.claim is None
            or active.claim.claim_token != claim_token
            or previous is None
            or previous.state is not TaskAttemptSegmentState.SUSPENDED
            or previous.request_id != request_id
            or previous.continuation_id != str(completion.continuation_id)
            or previous.checkpoint_id != checkpoint_id
        ):
            raise TaskQueueConflictError(
                "completed provider task provenance did not match"
            )
        store = self.queue.store
        store_attributes = (
            "_runs",
            "_attempts",
            "_attempt_segments",
            "_run_transitions",
            "_attempt_transitions",
            "_segment_transitions",
        )
        store_snapshot = {
            attribute: {
                key: list(value) if isinstance(value, list) else value
                for key, value in getattr(store, attribute).items()
            }
            for attribute in store_attributes
        }
        queue_snapshot = (self.queue.item, self.queue.completed)
        try:
            segment_state = (
                TaskAttemptSegmentState.ABANDONED
                if type(settlement) is TaskDurableResumeCancellation
                else TaskAttemptSegmentState.FAILED
            )
            await store.transition_attempt_segment(
                segment_id,
                from_states={TaskAttemptSegmentState.RUNNING},
                to_state=segment_state,
                reason="completed_provider_processing_failed",
                claim_token=claim_token,
                metadata=metadata,
            )
            run_state = TaskRunState.FAILED
            if type(settlement) is TaskDurableResumeCancellation:
                await store.transition_run(
                    task_run_id,
                    from_states={TaskRunState.RUNNING},
                    to_state=TaskRunState.CANCEL_REQUESTED,
                    reason="cancel_requested",
                    claim_token=claim_token,
                    metadata=metadata,
                )
                run_state = TaskRunState.CANCELLED
            task_completion = await self.queue.complete(
                queue_item_id,
                claim_token=claim_token,
                run_state=run_state,
                attempt_state=TaskAttemptState.FAILED,
                result=settlement.result,
                now=now,
                metadata=metadata,
            )
        except BaseException:
            for attribute, value in store_snapshot.items():
                setattr(store, attribute, value)
            self.queue.item, self.queue.completed = queue_snapshot
            raise
        commit = _FakeDurableSettlementCommit(completion=task_completion)
        self.completed_terminalizations.append(settlement)
        self._completed_terminalization_replays[queue_item_id] = (
            identity,
            commit,
        )
        return commit

    async def fail_admitted_reentry(
        self,
        rejection: ContinuationRejectionCommand,
        failure: TaskDurableResumeFailure,
        *,
        queue_item_id: str,
        claim_token: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSettlementCommit:
        del rejection
        completion = await self.fail_claimed_reentry(
            queue_item_id=queue_item_id,
            claim_token=claim_token,
            task_run_id=task_run_id,
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            result=failure.result,
            reason="task_resume_rejected",
            now=now,
            metadata=metadata,
        )
        self.rejected_admissions.append(failure)
        return _FakeDurableSettlementCommit(completion=completion)

    async def mark_resume_ambiguous(
        self,
        completion: ContinuationCompletionCommand,
        failure: TaskDurableResumeFailure,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> _FakeDurableSettlementCommit:
        commit = await self.settle_resume(
            completion,
            failure,
            queue_item_id=queue_item_id,
            claim_token=claim_token,
            segment_id=segment_id,
            task_run_id=task_run_id,
            now=now,
            metadata=metadata,
        )
        self.ambiguities.append(failure)
        return commit

    async def release_running_reentry(
        self,
        *,
        queue_item_id: str,
        claim_token: str,
        segment_id: str,
        task_run_id: str,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueReentry:
        await self._gate_convergence_commit()
        item = self.queue.item
        assert item is not None
        assert item.queue_item_id == queue_item_id
        assert item.run_id == task_run_id
        run = await self.queue.store.get_run(task_run_id)
        segment = await self.queue.store.transition_attempt_segment(
            segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.SUSPENDED,
            reason="resume_released",
            request_id=request_id,
            continuation_id=continuation_id,
            checkpoint_id=checkpoint_id,
            claim_token=claim_token,
            metadata=metadata,
        )
        attempt = await self.queue.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.SUSPENDED,
            reason="resume_released",
            claim_token=claim_token,
            metadata=metadata,
        )
        run = await self.queue.store.transition_run(
            task_run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.QUEUED,
            reason="resume_released",
            claim_token=claim_token,
            metadata=metadata,
        )
        run = replace(run, claim=None)
        self.queue.store._runs[run.run_id] = run
        self.queue.item = TaskQueueItem(
            queue_item_id=item.queue_item_id,
            run_id=run.run_id,
            queue_name=item.queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=item.priority,
            available_at=now or self.queue.now,
            attempts=item.attempts,
            created_at=item.created_at,
            updated_at=now or self.queue.now,
            run_state=run.state,
            metadata=metadata or {},
        )
        reentry = TaskQueueReentry(
            queue_item=self.queue.item,
            run=run,
            attempt=attempt,
            previous_segment=segment,
        )
        self.running_releases.append(reentry)
        return reentry

    async def reconcile_expired_reentry(
        self,
        *,
        queue_item_id: str,
        expected_claim_token: str,
        task_run_id: str,
        result: TaskExecutionResult,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskDurableExpiredReentryCommit:
        del result, now, metadata
        self.expired_calls.append(
            (
                queue_item_id,
                expected_claim_token,
                task_run_id,
            )
        )
        if self.expired_commit is None:
            raise TaskStoreConflictError(
                "expired durable reentry did not match"
            )
        return self.expired_commit


class TaskDurableResumeSettlementTest(TestCase):
    def test_settlement_variants_reject_invalid_results(self) -> None:
        invalid = cast(TaskExecutionResult, object())
        for settlement_type in (
            TaskDurableResumeSuccess,
            TaskDurableResumeFailure,
            TaskDurableResumeCancellation,
        ):
            with self.subTest(settlement_type=settlement_type.__name__):
                with self.assertRaises(InputValidationError) as raised:
                    settlement_type(result=invalid)
                self.assertIs(
                    raised.exception.code,
                    InputErrorCode.INVALID_TYPE,
                )
                self.assertEqual(
                    raised.exception.path,
                    "task_resume.settlement.result",
                )

    def test_success_rejects_error_result(self) -> None:
        with self.assertRaises(InputValidationError) as raised:
            TaskDurableResumeSuccess(
                result=TaskExecutionResult(error="sanitized failure")
            )

        self.assertIs(
            raised.exception.code,
            InputErrorCode.ILLEGAL_TRANSITION,
        )
        self.assertEqual(
            raised.exception.path,
            "task_resume.settlement.result.error",
        )

    def test_failure_and_cancellation_require_error_only(self) -> None:
        for settlement_type in (
            TaskDurableResumeFailure,
            TaskDurableResumeCancellation,
        ):
            with self.subTest(
                settlement_type=settlement_type.__name__,
                invalid="missing error",
            ):
                with self.assertRaises(InputValidationError) as raised:
                    settlement_type(result=TaskExecutionResult())
                self.assertIs(
                    raised.exception.code,
                    InputErrorCode.ILLEGAL_TRANSITION,
                )
                self.assertEqual(
                    raised.exception.path,
                    "task_resume.settlement.result.error",
                )
            with self.subTest(
                settlement_type=settlement_type.__name__,
                invalid="successful output",
            ):
                with self.assertRaises(InputValidationError) as raised:
                    settlement_type(
                        result=TaskExecutionResult(
                            output_summary="output",
                            error="sanitized failure",
                        )
                    )
                self.assertIs(
                    raised.exception.code,
                    InputErrorCode.ILLEGAL_TRANSITION,
                )
                self.assertEqual(
                    raised.exception.path,
                    "task_resume.settlement.result.output_summary",
                )

    def test_settlement_digest_rejects_unknown_type(self) -> None:
        with self.assertRaises(InputValidationError) as raised:
            task_durable_resume_settlement_digest(
                cast(TaskDurableResumeSettlement, object())
            )

        self.assertIs(raised.exception.code, InputErrorCode.INVALID_TYPE)
        self.assertEqual(raised.exception.path, "task_resume.settlement")


class AwaitOwnedTaskTest(IsolatedAsyncioTestCase):
    async def test_completed_child_clears_pending_owner_cancellation(
        self,
    ) -> None:
        async def completed() -> str:
            return "completed"

        child = create_task(completed())
        await child

        async def deliver_cancellation(task: AsyncTask[object]) -> object:
            self.assertIs(task, child)
            owner = current_task()
            assert owner is not None
            owner.cancel()
            owner.cancel()
            await sleep(0)
            raise AssertionError("cancellation was not delivered")

        with patch.object(
            worker_module,
            "shield",
            deliver_cancellation,
        ):
            result = await _await_owned_task(child)

        self.assertEqual(result, "completed")
        owner = current_task()
        assert owner is not None
        self.assertEqual(owner.cancelling(), 0)

    async def test_pending_self_cancellation_before_entry_is_consumed(
        self,
    ) -> None:
        async def owned_work() -> str:
            await sleep(0)
            return "owned result"

        owned = create_task(owned_work(), name="owned-before-entry")
        owner = current_task()
        assert owner is not None
        self.assertTrue(owner.cancel())

        result = await _await_owned_task(owned)

        self.assertEqual(result, "owned result")
        self.assertTrue(owned.done())
        self.assertFalse(owned.cancelled())
        self.assertEqual(owner.cancelling(), 0)
        self.assertNotIn(owned, all_tasks())

    async def test_repeated_cancellation_while_awaiting_is_consumed(
        self,
    ) -> None:
        started = Event()
        proceed = Event()
        owned_tasks: list[AsyncTask[str]] = []

        async def owned_work() -> str:
            started.set()
            await proceed.wait()
            return "owned result"

        async def owner_work() -> str:
            owned = create_task(owned_work(), name="owned-repeated-cancel")
            owned_tasks.append(owned)
            return await _await_owned_task(owned)

        owner = create_task(owner_work(), name="owned-task-owner")
        await started.wait()
        self.assertTrue(owner.cancel())
        self.assertTrue(owner.cancel())
        await sleep(0)
        self.assertFalse(owner.done())
        self.assertTrue(owner.cancel())
        self.assertTrue(owner.cancel())
        await sleep(0)
        self.assertFalse(owner.done())
        proceed.set()

        result = await wait_for(shield(owner), timeout=1)

        self.assertEqual(result, "owned result")
        self.assertFalse(owner.cancelled())
        self.assertEqual(owner.cancelling(), 0)
        self.assertEqual(len(owned_tasks), 1)
        self.assertTrue(owned_tasks[0].done())
        self.assertFalse(owned_tasks[0].cancelled())
        self.assertNotIn(owned_tasks[0], all_tasks())

    async def test_child_self_cancellation_propagates(self) -> None:
        async def self_cancel() -> None:
            raise CancelledError("owned child cancelled")

        owned = create_task(self_cancel(), name="owned-self-cancel")

        with self.assertRaisesRegex(
            CancelledError,
            "owned child cancelled",
        ):
            await _await_owned_task(owned)

        self.assertTrue(owned.done())
        self.assertTrue(owned.cancelled())
        owner = current_task()
        assert owner is not None
        self.assertEqual(owner.cancelling(), 0)

    async def test_child_runtime_error_propagates(self) -> None:
        error = RuntimeError("owned child failed")

        async def fail() -> None:
            raise error

        owned = create_task(fail(), name="owned-runtime-error")

        with self.assertRaises(RuntimeError) as caught:
            await _await_owned_task(owned)

        self.assertIs(caught.exception, error)
        self.assertTrue(owned.done())
        self.assertFalse(owned.cancelled())

    async def test_child_process_control_group_propagates(self) -> None:
        error = BaseExceptionGroup(
            "owned cleanup failed",
            (
                RuntimeError("cleanup failed"),
                SystemExit(7),
            ),
        )

        async def fail() -> None:
            raise error

        owned = create_task(fail(), name="owned-process-control-error")

        with self.assertRaises(BaseExceptionGroup) as caught:
            await _await_owned_task(owned)

        self.assertIs(caught.exception, error)
        self.assertTrue(owned.done())
        self.assertFalse(owned.cancelled())


class TaskWorkerTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        self.definition = _definition()
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                queue="default",
            )
        )
        self.run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        self.run = await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        self.queue = FakeQueue(self.store, self.now)
        self.queue.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=self.run.run_id,
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=self.run.state,
        )

    async def _prepare_durable_reentry(
        self,
        target: DurableResumableTarget,
        coordinator: FakeDurableSuspensionCoordinator,
    ) -> TaskQueueSuspension:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )
        suspended = await worker.process_once()
        assert suspended.suspension is not None
        suspension = suspended.suspension
        await self.queue.requeue_suspended(
            self.run.run_id,
            request_id=suspension.request_id,
            continuation_id=suspension.continuation_id,
            resolution_revision=2,
            now=self.now,
        )
        return suspension

    async def _use_definition(self, definition: TaskDefinition) -> None:
        self.definition = definition
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                queue="default",
            )
        )
        self.run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        self.run = await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        self.queue = FakeQueue(self.store, self.now)
        self.queue.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=self.run.run_id,
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=self.run.state,
        )

    async def _use_request(self, request: TaskExecutionRequest) -> None:
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(request)
        self.run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        self.run = await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        self.queue = FakeQueue(self.store, self.now)
        self.queue.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=self.run.run_id,
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=self.run.state,
        )

    async def _claim(self) -> TaskQueueClaim:
        claim = await self.queue.claim(
            "default",
            worker_id="worker-1",
            lease_expires_at=self.now.replace(hour=1),
            now=self.now,
            metadata={"worker_id": "worker-1"},
        )
        assert claim is not None
        return claim

    async def test_resume_claim_lease_and_cleanup_owners_fail_closed(
        self,
    ) -> None:
        claim = await self._claim()
        bridge = _TaskResumeClaimLeaseBridge(claim)
        with self.assertRaisesRegex(TypeError, "claim_lease"):
            await bridge.bind(cast(Any, object()))

        lease = FakeDurableContinuationClaimLease(
            expires_at=self.now.replace(hour=2)
        )
        await bridge.bind(cast(Any, lease))
        with self.assertRaisesRegex(TaskWorkerError, "already bound"):
            await bridge.bind(cast(Any, lease))
        with self.assertRaisesRegex(TaskWorkerError, "moved backwards"):
            await bridge.heartbeat(
                self.now.replace(minute=30),
                now=self.now.replace(minute=1),
            )
        await bridge.unbind()

        owner = _TaskDurableResumeOwner()
        await owner.close()
        owner.take(None)
        handle = FakeDurableResumeHandle()
        owner.take(handle)
        with self.assertRaisesRegex(TaskWorkerError, "already owned"):
            owner.take(handle)
        await owner.close()
        await owner.close()
        self.assertEqual(handle.close_calls, 1)

        unbind_failure = RuntimeError("unbind failed")
        close_failure = RuntimeError("close failed")
        one_failure = _TaskClaimCleanupOwner(
            cast(
                Any,
                SimpleNamespace(unbind=AsyncMock(side_effect=unbind_failure)),
            ),
            cast(
                Any,
                SimpleNamespace(close=AsyncMock()),
            ),
            queue_item_id="queue-item-one-error",
        )
        with self.assertRaises(RuntimeError) as single:
            await one_failure.close()
        self.assertIs(single.exception, unbind_failure)

        two_failures = _TaskClaimCleanupOwner(
            cast(
                Any,
                SimpleNamespace(unbind=AsyncMock(side_effect=unbind_failure)),
            ),
            cast(
                Any,
                SimpleNamespace(close=AsyncMock(side_effect=close_failure)),
            ),
            queue_item_id="queue-item-two-errors",
        )
        with self.assertRaises(BaseExceptionGroup) as grouped:
            await two_failures.close()
        self.assertEqual(
            grouped.exception.exceptions,
            (unbind_failure, close_failure),
        )

    async def test_worker_rejects_untrusted_runtime_boundaries(self) -> None:
        with self.assertRaises(AssertionError):
            TaskWorker(
                self.store,
                cast(Any, self.queue),
                target=FakeTarget(),
                container_backend=cast(Any, object()),
            )
        with self.assertRaises(AssertionError):
            TaskWorker(
                self.store,
                cast(Any, self.queue),
                target=FakeTarget(),
                worker_runtime_envelope_runner=cast(Any, object()),
            )

        async def untrusted_runner(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        with self.assertRaisesRegex(
            AssertionError,
            "must be trusted",
        ):
            TaskWorker(
                self.store,
                cast(Any, self.queue),
                target=FakeTarget(),
                worker_runtime_envelope_runner=cast(
                    Any,
                    untrusted_runner,
                ),
            )

        setattr(untrusted_runner, "trusted_runtime_envelope_runner", True)
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_runtime_envelope_runner=cast(Any, untrusted_runner),
        )
        self.assertIs(
            worker._worker_runtime_envelope_runner,
            untrusted_runner,
        )
        with self.assertRaises(AssertionError):
            TaskWorker(
                self.store,
                cast(Any, self.queue),
                target=FakeTarget(),
                definition_base=cast(Any, 1),
            )

    async def test_worker_expiry_and_skill_helpers_cover_terminal_shapes(
        self,
    ) -> None:
        self.assertFalse(worker_module._is_input_expiry(SystemExit(7)))
        self.assertTrue(
            worker_module._is_input_expiry(
                worker_module._TaskDurableResumeExpired("expired")
            )
        )
        self.assertFalse(
            worker_module._is_input_expiry(
                BaseExceptionGroup(
                    "process control",
                    (
                        InputValidationError(
                            InputErrorCode.EXPIRED,
                            "continuation",
                            "expired",
                        ),
                        SystemExit(7),
                    ),
                )
            )
        )
        self.assertEqual(
            worker_module._input_expiry_cleanup_errors(
                worker_module._TaskDurableResumeExpired("expired"),
                ignore_cancelled=False,
            ),
            (),
        )

        run = replace(
            self.run,
            request=replace(
                self.run.request,
                metadata={"skills": {"identity": "trusted"}},
            ),
        )
        self.assertEqual(
            worker_module._task_skills_identity_from_run(run),
            {"identity": "trusted"},
        )
        self.assertIsNone(
            worker_module._task_skills_identity_from_run(self.run)
        )

    async def test_worker_durable_target_helpers_reject_invalid_pairing(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
        )
        claim = await self._claim()
        context = self._target_context(claim)
        durable_target = cast(Any, object())
        with self.assertRaisesRegex(
            TaskWorkerError,
            "has no admitted continuation",
        ):
            await worker._target_execution(
                context,
                durable_target=durable_target,
            )

        handle = FakeDurableResumeHandle(
            error_state=DurableContinuationResumeState.ADMITTED
        )
        durable_context = replace(context, durable_resume=handle)
        with self.assertRaisesRegex(
            TaskWorkerError,
            "was not prepared",
        ):
            await worker._target_execution(
                durable_context,
                durable_target=None,
            )

        async def completed() -> TaskTargetOutcome:
            return completed_task_target_outcome("done")

        target_task = create_task(completed())
        await target_task
        invalid_interrupt = AsyncMock(
            return_value=DurableContinuationResumeState.ADMITTED
        )
        setattr(handle, "interrupt_dispatch", invalid_interrupt)
        with self.assertRaisesRegex(
            TaskWorkerError,
            "safe settlement",
        ):
            await worker._interrupt_durable_target(
                durable_context,
                target_task,
            )
        invalid_interrupt.assert_awaited_once_with()

    async def test_worker_claim_guard_preserves_cleanup_failures(
        self,
    ) -> None:
        async def pending() -> None:
            await sleep(10)

        cancelled = create_task(pending())
        cancelled.cancel()
        with self.assertRaises(CancelledError):
            await cancelled
        cleanup = RuntimeError("cleanup failed")
        with self.assertRaises(RuntimeError) as raised:
            TaskWorker._raise_stopped_claim_guard(
                cancelled,
                admission_cleanup_error=cleanup,
            )
        self.assertIs(raised.exception, cleanup)
        with self.assertRaisesRegex(
            TaskQueueConflictError,
            "heartbeat stopped",
        ):
            TaskWorker._raise_stopped_claim_guard(cancelled)

        heartbeat_error = RuntimeError("heartbeat failed")

        async def fail() -> None:
            raise heartbeat_error

        failed = create_task(fail())
        with self.assertRaises(RuntimeError):
            await failed
        with self.assertRaises(BaseExceptionGroup) as grouped:
            TaskWorker._raise_stopped_claim_guard(
                failed,
                admission_cleanup_error=cleanup,
            )
        self.assertEqual(
            grouped.exception.exceptions,
            (heartbeat_error, cleanup),
        )

        async def complete() -> None:
            return None

        completed = create_task(complete())
        await completed
        with self.assertRaises(RuntimeError) as completed_cleanup:
            TaskWorker._raise_stopped_claim_guard(
                completed,
                admission_cleanup_error=cleanup,
            )
        self.assertIs(completed_cleanup.exception, cleanup)

    async def test_worker_heartbeat_and_reentry_guards_fail_closed(
        self,
    ) -> None:
        claim = await self._claim()
        queue = SimpleNamespace(
            claim=AsyncMock(),
            heartbeat=AsyncMock(
                return_value=SimpleNamespace(lease_expires_at=None)
            ),
        )
        worker = TaskWorker(
            self.store,
            cast(Any, queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        bridge = _TaskResumeClaimLeaseBridge(claim)
        with (
            patch.object(worker_module, "sleep", AsyncMock()),
            self.assertRaisesRegex(
                TaskQueueConflictError,
                "returned no claim lease",
            ),
        ):
            await worker._heartbeat_claim(
                claim,
                heartbeat_seconds=1,
                claim_lease_manager=bridge,
            )

        worker._store = cast(
            Any,
            SimpleNamespace(
                list_attempt_segments=AsyncMock(return_value=(object(),))
            ),
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "already has segment history",
        ):
            await worker._previous_attempt_segment(claim)

        suspended_claim = replace(
            claim,
            attempt=replace(
                claim.attempt,
                state=TaskAttemptState.SUSPENDED,
            ),
        )
        worker._store = cast(
            Any,
            SimpleNamespace(
                list_attempt_segments=AsyncMock(
                    return_value=(
                        SimpleNamespace(state=TaskAttemptSegmentState.RUNNING),
                    )
                )
            ),
        )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "lacks a suspended prior segment",
        ):
            await worker._previous_attempt_segment(suspended_claim)

        resume_owner = _TaskDurableResumeOwner()
        with self.assertRaisesRegex(
            worker_module._TaskDurableResumeRejected,
            "coordinator is unavailable",
        ):
            await worker._admit_durable_resume(
                claim,
                cast(Any, object()),
                claim_lease_manager=bridge,
                resume_owner=resume_owner,
            )

        coordinator = SimpleNamespace(admit=AsyncMock(return_value=None))
        coordinated = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            durable_resume_coordinator=cast(Any, coordinator),
            worker_id="worker-1",
        )
        with self.assertRaisesRegex(
            worker_module._TaskDurableResumeRejected,
            "was not admitted",
        ):
            await coordinated._admit_durable_resume(
                claim,
                cast(Any, object()),
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )

        coordinator.admit = AsyncMock(return_value=FakeDurableResumeHandle())
        with self.assertRaisesRegex(
            worker_module._TaskDurableResumeRejected,
            "fresh task work",
        ):
            await coordinated._admit_durable_resume(
                claim,
                None,
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )

        coordinator.admit = AsyncMock(return_value=object())
        with self.assertRaisesRegex(
            worker_module._TaskDurableResumeRejected,
            "admission is invalid",
        ):
            await coordinated._admit_durable_resume(
                claim,
                cast(Any, object()),
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )

        coordinator.admit = AsyncMock(return_value=None)
        self.assertIsNone(
            await coordinated._admit_durable_resume(
                claim,
                None,
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )
        )

        async def blocked_admission(*_args: object, **_kwargs: object) -> None:
            await Event().wait()

        async def cancel_admission(
            task: AsyncTask[object],
            cleanup_error: BaseException | None,
        ) -> BaseException | None:
            task.cancel()
            try:
                await task
            except CancelledError:
                pass
            return cleanup_error

        wait_error = RuntimeError("admission wait failed")
        cleanup_error = RuntimeError("admission cleanup failed")

        async def cancel_admission_with_error(
            task: AsyncTask[object],
        ) -> BaseException:
            result = await cancel_admission(task, cleanup_error)
            assert result is not None
            return result

        waiting_heartbeat = create_task(Event().wait())
        with (
            patch.object(
                coordinated,
                "_admit_durable_resume",
                new=blocked_admission,
            ),
            patch.object(
                worker_module,
                "wait",
                AsyncMock(side_effect=wait_error),
            ),
            patch.object(
                worker_module,
                "_cancel_task_error",
                new=cancel_admission_with_error,
            ),
            self.assertRaises(BaseExceptionGroup) as grouped,
        ):
            await coordinated._admit_durable_resume_guarded(
                claim,
                None,
                heartbeat_task=waiting_heartbeat,
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )
        waiting_heartbeat.cancel()
        with self.assertRaises(CancelledError):
            await waiting_heartbeat
        self.assertEqual(
            grouped.exception.exceptions,
            (wait_error, cleanup_error),
        )

        async def cancel_admission_cleanly(
            task: AsyncTask[object],
        ) -> None:
            await cancel_admission(task, None)

        waiting_heartbeat = create_task(Event().wait())
        with (
            patch.object(
                coordinated,
                "_admit_durable_resume",
                new=blocked_admission,
            ),
            patch.object(
                worker_module,
                "wait",
                AsyncMock(side_effect=wait_error),
            ),
            patch.object(
                worker_module,
                "_cancel_task_error",
                new=cancel_admission_cleanly,
            ),
            self.assertRaises(RuntimeError) as reraised,
        ):
            await coordinated._admit_durable_resume_guarded(
                claim,
                None,
                heartbeat_task=waiting_heartbeat,
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )
        waiting_heartbeat.cancel()
        with self.assertRaises(CancelledError):
            await waiting_heartbeat
        self.assertIs(reraised.exception, wait_error)

        heartbeat = create_task(sleep(0))
        await heartbeat

        async def fail_admission(
            *_args: object,
            **_kwargs: object,
        ) -> None:
            raise RuntimeError("admission failed")

        with (
            patch.object(
                coordinated,
                "_admit_durable_resume",
                new=fail_admission,
            ),
            patch.object(
                coordinated,
                "_raise_if_claim_guard_stopped",
                side_effect=TaskQueueConflictError("heartbeat stopped"),
            ) as stopped,
            self.assertRaisesRegex(
                TaskQueueConflictError,
                "heartbeat stopped",
            ),
        ):
            await coordinated._admit_durable_resume_guarded(
                claim,
                None,
                heartbeat_task=heartbeat,
                claim_lease_manager=bridge,
                resume_owner=_TaskDurableResumeOwner(),
            )
        stopped.assert_called_once()

    async def test_worker_durable_settlement_guards_reject_invalid_state(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        claim = await self._claim()
        run, attempt, segment = await worker._start_claimed_attempt(
            claim,
            previous_segment=None,
        )
        sanitizer = worker._sanitizer(self.definition)
        handle = FakeDurableResumeHandle()

        with self.assertRaisesRegex(
            TaskWorkerError,
            "settlement coordinator is unavailable",
        ):
            await worker._settle_durable_success(
                self.definition,
                claim=claim,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=handle,
                output="ready",
            )

        settle = AsyncMock(return_value=SimpleNamespace(completion=object()))
        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(settle_resume=settle),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "returned invalid state",
        ):
            await worker._settle_durable_success(
                self.definition,
                claim=claim,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=handle,
                output="ready",
            )

        async def converge(
            state: DurableContinuationResumeState,
            *,
            coordinator: object | None,
            previous: TaskAttemptSegment | None = None,
            release_result: bool | None = None,
        ) -> None:
            durable = FakeDurableResumeHandle()
            durable.state = state
            if release_result is not None:
                setattr(
                    durable,
                    "release_if_pre_dispatch",
                    AsyncMock(return_value=release_result),
                )
            worker._durable_suspension_coordinator = cast(Any, coordinator)
            await worker._converge_durable_failure_owned(
                claim=claim,
                segment=segment,
                previous_segment=previous,
                sanitizer=sanitizer,
                durable_resume=durable,
                error=RuntimeError("private failure"),
            )

        worker._durable_suspension_coordinator = None
        with self.assertRaisesRegex(
            TaskWorkerError,
            "settlement coordinator is unavailable",
        ):
            await converge(
                DurableContinuationResumeState.DISPATCHED,
                coordinator=None,
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "settlement coordinator returned invalid state",
        ):
            await converge(
                DurableContinuationResumeState.DISPATCHED,
                coordinator=SimpleNamespace(
                    settle_resume=AsyncMock(
                        return_value=SimpleNamespace(completion=object())
                    )
                ),
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "ambiguity coordinator is unavailable",
        ):
            await converge(
                DurableContinuationResumeState.AMBIGUOUS,
                coordinator=None,
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "ambiguity coordinator returned invalid state",
        ):
            await converge(
                DurableContinuationResumeState.AMBIGUOUS,
                coordinator=SimpleNamespace(
                    mark_resume_ambiguous=AsyncMock(
                        return_value=SimpleNamespace(completion=object())
                    )
                ),
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "state changed during failure",
        ):
            await converge(
                DurableContinuationResumeState.ADMITTED,
                coordinator=None,
                release_result=False,
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "did not reach a safe settlement",
        ):
            await converge(
                DurableContinuationResumeState.DISPATCHING,
                coordinator=None,
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "running reentry release is unavailable",
        ):
            await converge(
                DurableContinuationResumeState.RELEASED,
                coordinator=None,
            )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "provenance is unavailable",
        ):
            await converge(
                DurableContinuationResumeState.RELEASED,
                coordinator=SimpleNamespace(
                    release_running_reentry=AsyncMock()
                ),
            )
        previous = replace(
            segment,
            state=TaskAttemptSegmentState.SUSPENDED,
            request_id="request",
            continuation_id="continuation",
            checkpoint_id="checkpoint",
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "returned invalid state",
        ):
            await converge(
                DurableContinuationResumeState.RELEASED,
                coordinator=SimpleNamespace(
                    release_running_reentry=AsyncMock(return_value=object())
                ),
                previous=previous,
            )

        self.assertIs(run.state, TaskRunState.RUNNING)
        self.assertIs(attempt.state, TaskAttemptState.RUNNING)

    async def test_worker_reconciliation_and_suspension_guards_fail_closed(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        claim = await self._claim()
        _run, _attempt, segment = await worker._start_claimed_attempt(
            claim,
            previous_segment=None,
        )
        sanitizer = worker._sanitizer(self.definition)
        suspended_claim = replace(
            claim,
            attempt=replace(
                claim.attempt,
                state=TaskAttemptState.SUSPENDED,
            ),
        )

        lease_lost = await worker._lease_lost_result(
            suspended_claim,
            sanitizer=sanitizer,
        )
        self.assertTrue(lease_lost.lease_lost)

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                reconcile_expired_reentry=AsyncMock(return_value=object())
            ),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "reconciliation returned invalid state",
        ):
            await worker._lease_lost_result(
                suspended_claim,
                sanitizer=sanitizer,
            )

        terminalize = AsyncMock(
            return_value=SimpleNamespace(completion=object())
        )
        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(terminalize_completed_resume=terminalize),
        )
        handle = FakeDurableResumeHandle()
        handle.state = DurableContinuationResumeState.COMPLETED
        handle.completed_digest = "0" * 64
        result = TaskExecutionResult(error={"code": "task.failed"})
        with self.assertRaisesRegex(
            TaskWorkerError,
            "provenance is unavailable",
        ):
            await worker._terminalize_completed_durable_failure(
                claim=claim,
                segment=segment,
                previous_segment=None,
                durable_resume=handle,
                result=result,
                error=RuntimeError("private failure"),
            )

        previous = replace(
            segment,
            state=TaskAttemptSegmentState.SUSPENDED,
            request_id="request",
            continuation_id="continuation",
            checkpoint_id="checkpoint",
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "returned invalid state",
        ):
            await worker._terminalize_completed_durable_failure(
                claim=claim,
                segment=segment,
                previous_segment=previous,
                durable_resume=handle,
                result=result,
                error=RuntimeError("private failure"),
            )

        required = InputRequiredResult(
            request_id=InputRequestId("suspend-request"),
            continuation_id=ContinuationId("suspend-continuation"),
            detached_resumption_available=True,
        )
        forged = object.__new__(TaskTargetSuspended)
        object.__setattr__(forged, "input_required", required)
        object.__setattr__(forged, "checkpoint_id", None)
        object.__setattr__(forged, "durable", object())
        with self.assertRaisesRegex(
            TaskWorkerError,
            "checkpoint is unavailable",
        ):
            await worker._suspend(
                claim=claim,
                segment=segment,
                outcome=forged,
                durable_resume=None,
            )

        object.__setattr__(forged, "checkpoint_id", "checkpoint")
        worker._durable_suspension_coordinator = None
        with self.assertRaisesRegex(
            TaskWorkerError,
            "coordinator is unavailable",
        ):
            await worker._suspend(
                claim=claim,
                segment=segment,
                outcome=forged,
                durable_resume=None,
            )

        durable_payload = SimpleNamespace(
            command=object(),
            continuation=object(),
        )
        object.__setattr__(forged, "durable", durable_payload)
        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                create_and_suspend=AsyncMock(
                    return_value=SimpleNamespace(suspension=object())
                )
            ),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "returned invalid state",
        ):
            await worker._suspend(
                claim=claim,
                segment=segment,
                outcome=forged,
                durable_resume=None,
            )

        attached = suspended_task_target_outcome(required)
        with self.assertRaisesRegex(
            TaskWorkerError,
            "payload is unavailable",
        ):
            await worker._suspend(
                claim=claim,
                segment=segment,
                outcome=attached,
                durable_resume=None,
            )

    async def test_worker_setup_failure_matrix_fails_closed(
        self,
    ) -> None:
        claim = await self._claim()
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        sanitizer = worker._sanitizer(self.definition)
        lease_lost = TaskWorkerProcessResult(
            claimed=claim,
            lease_lost=True,
        )
        setattr(
            worker,
            "_lease_lost_result",
            AsyncMock(return_value=lease_lost),
        )
        worker._queue = cast(
            Any,
            SimpleNamespace(
                abandon=AsyncMock(
                    side_effect=TaskQueueConflictError("claim lost")
                )
            ),
        )
        self.assertIs(
            await worker._converge_claimed_setup_failure(
                claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=None,
                durable_resume=None,
                error=RuntimeError("setup failed"),
            ),
            lease_lost,
        )

        suspended_claim = replace(
            claim,
            attempt=replace(
                claim.attempt,
                state=TaskAttemptState.SUSPENDED,
            ),
        )
        previous = TaskAttemptSegment(
            segment_id="previous",
            attempt_id=claim.attempt.attempt_id,
            run_id=claim.run.run_id,
            segment_number=1,
            state=TaskAttemptSegmentState.SUSPENDED,
            created_at=self.now,
            updated_at=self.now,
            request_id="request",
            continuation_id="continuation",
            checkpoint_id="checkpoint",
        )
        deterministic = worker_module._TaskDurableResumeRejected(
            "unsupported resume"
        )

        worker._durable_suspension_coordinator = None
        unavailable = await worker._converge_claimed_setup_failure(
            suspended_claim,
            definition=self.definition,
            sanitizer=sanitizer,
            previous_segment=previous,
            durable_resume=FakeDurableResumeHandle(),
            error=deterministic,
        )
        self.assertTrue(unavailable.lease_lost)

        fail_admitted = AsyncMock(
            side_effect=TaskQueueConflictError("claim lost")
        )
        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(fail_admitted_reentry=fail_admitted),
        )
        self.assertIs(
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=previous,
                durable_resume=FakeDurableResumeHandle(),
                error=deterministic,
            ),
            lease_lost,
        )

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                fail_admitted_reentry=AsyncMock(
                    return_value=SimpleNamespace(completion=object())
                )
            ),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "rejection returned invalid state",
        ):
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=previous,
                durable_resume=FakeDurableResumeHandle(),
                error=deterministic,
            )

        for release in (
            AsyncMock(side_effect=TaskQueueConflictError("claim lost")),
            AsyncMock(return_value=False),
        ):
            handle = FakeDurableResumeHandle()
            setattr(handle, "release_if_pre_dispatch", release)
            worker._durable_suspension_coordinator = None
            self.assertIs(
                await worker._converge_claimed_setup_failure(
                    suspended_claim,
                    definition=self.definition,
                    sanitizer=sanitizer,
                    previous_segment=None,
                    durable_resume=handle,
                    error=RuntimeError("transient setup failure"),
                ),
                lease_lost,
            )

        no_release = await worker._converge_claimed_setup_failure(
            suspended_claim,
            definition=self.definition,
            sanitizer=sanitizer,
            previous_segment=previous,
            durable_resume=None,
            error=RuntimeError("transient setup failure"),
        )
        self.assertTrue(no_release.lease_lost)

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                release_claimed_reentry=AsyncMock(
                    side_effect=TaskQueueConflictError("claim lost")
                )
            ),
        )
        self.assertIs(
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=previous,
                durable_resume=None,
                error=RuntimeError("transient setup failure"),
            ),
            lease_lost,
        )

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                release_claimed_reentry=AsyncMock(return_value=object())
            ),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "release returned invalid state",
        ):
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=previous,
                durable_resume=None,
                error=RuntimeError("transient setup failure"),
            )

        worker._durable_suspension_coordinator = None
        no_failure = await worker._converge_claimed_setup_failure(
            suspended_claim,
            definition=self.definition,
            sanitizer=sanitizer,
            previous_segment=None,
            durable_resume=None,
            error=deterministic,
        )
        self.assertTrue(no_failure.lease_lost)

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                fail_claimed_reentry=AsyncMock(
                    side_effect=TaskQueueConflictError("claim lost")
                )
            ),
        )
        self.assertIs(
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=None,
                durable_resume=None,
                error=deterministic,
            ),
            lease_lost,
        )

        worker._durable_suspension_coordinator = cast(
            Any,
            SimpleNamespace(
                fail_claimed_reentry=AsyncMock(return_value=object())
            ),
        )
        with self.assertRaisesRegex(
            TaskWorkerError,
            "failure returned invalid state",
        ):
            await worker._converge_claimed_setup_failure(
                suspended_claim,
                definition=self.definition,
                sanitizer=sanitizer,
                previous_segment=None,
                durable_resume=None,
                error=deterministic,
            )

    async def test_worker_container_execution_failure_matrix(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        claim = await self._claim()
        run, attempt, _segment = await worker._start_claimed_attempt(
            claim,
            previous_segment=None,
        )
        sanitizer = worker._sanitizer(self.definition)
        record_event = AsyncMock()
        setattr(worker, "_record_container_event", record_event)

        verification_error = TaskContainerVerificationError(
            code="container.invalid",
            path="container",
            message="invalid container plan",
            hint="Use a valid plan.",
        )
        with (
            patch.object(
                worker_module,
                "verify_task_container_request",
                side_effect=verification_error,
            ),
            self.assertRaises(TaskValidationError),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        worker_envelope_plans = SimpleNamespace(
            enabled=True,
            worker_envelope=object(),
            attempt=None,
        )
        with (
            patch.object(
                worker_module,
                "verify_task_container_request",
                return_value=worker_envelope_plans,
            ),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.worker_envelope_unsupported",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        envelope_result = TaskContainerAttemptResult(output="enveloped")
        worker._worker_runtime_envelope_runner = cast(
            Any,
            AsyncMock(return_value=envelope_result),
        )
        with patch.object(
            worker_module,
            "verify_task_container_request",
            return_value=worker_envelope_plans,
        ):
            self.assertIs(
                await worker._run_task_container(
                    self.definition,
                    run=run,
                    attempt=attempt,
                    input_mounts=(),
                    sanitizer=sanitizer,
                ),
                envelope_result,
            )

        no_attempt_plans = SimpleNamespace(
            enabled=True,
            worker_envelope=None,
            attempt=None,
        )
        worker._worker_runtime_envelope_runner = None
        with patch.object(
            worker_module,
            "verify_task_container_request",
            return_value=no_attempt_plans,
        ):
            self.assertIsNone(
                await worker._run_task_container(
                    self.definition,
                    run=run,
                    attempt=attempt,
                    input_mounts=(),
                    sanitizer=sanitizer,
                )
            )

        attempt_plans = SimpleNamespace(
            enabled=True,
            worker_envelope=None,
            attempt=object(),
        )
        with (
            patch.object(
                worker_module,
                "verify_task_container_request",
                return_value=attempt_plans,
            ),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.backend_unavailable",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        backend = SimpleNamespace(probe=AsyncMock(return_value=object()))
        worker._container_backend = cast(Any, backend)

        @contextmanager
        def container_patches(
            *,
            unsupported_path: str | None = None,
            output_contract: object = SimpleNamespace(enabled=True),
        ) -> Iterator[None]:
            with (
                patch.object(
                    worker_module,
                    "verify_task_container_request",
                    return_value=attempt_plans,
                ),
                patch.object(
                    worker_module,
                    "task_container_lifecycle_run_plan",
                    return_value=object(),
                ),
                patch.object(
                    worker,
                    "_check_cancelled",
                    AsyncMock(),
                ),
                patch.object(
                    worker_module,
                    "task_container_unsupported_input_mount_path",
                    return_value=unsupported_path,
                ),
                patch.object(
                    worker_module,
                    "task_container_output_contract",
                    return_value=output_contract,
                ),
                patch.object(
                    worker_module,
                    "_raise_for_container_backend_selection",
                ),
            ):
                yield

        with (
            container_patches(unsupported_path="input_mounts[0]"),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.input_mount_unsupported",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        with (
            container_patches(output_contract=None),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.task_execution_unsupported",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        with (
            container_patches(output_contract=SimpleNamespace(enabled=False)),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.output_unsupported",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        failed_result = SimpleNamespace(
            execution=SimpleNamespace(status=ContainerResultStatus.FAILED),
            output=None,
        )
        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                return_value=failed_result,
            ),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.execution_failed",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        missing_output_result = SimpleNamespace(
            execution=SimpleNamespace(status=ContainerResultStatus.COMPLETED),
            output=None,
        )
        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                return_value=missing_output_result,
            ),
            self.assertRaisesRegex(
                TaskValidationError,
                "container.output_unsupported",
            ),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        accepted_result = SimpleNamespace(
            execution=SimpleNamespace(status=ContainerResultStatus.COMPLETED),
            output=SimpleNamespace(
                decision=ContainerOutputDecisionType.ACCEPT,
                artifacts=(object(),),
            ),
        )
        artifact_error = TaskContainerVerificationError(
            code="container.output.invalid",
            path="container.output",
            message="invalid output artifact",
            hint="Return a valid artifact.",
        )
        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                return_value=accepted_result,
            ),
            patch.object(
                worker_module,
                "task_container_output_artifacts",
                AsyncMock(side_effect=artifact_error),
            ),
            self.assertRaises(TaskValidationError),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                return_value=accepted_result,
            ),
            patch.object(
                worker_module,
                "task_container_output_artifacts",
                AsyncMock(return_value="container output"),
            ),
            patch.object(
                worker,
                "_record_output_artifacts",
                AsyncMock(),
            ),
        ):
            completed = await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )
        self.assertEqual(
            completed,
            TaskContainerAttemptResult(
                output="container output",
                output_artifacts_recorded=True,
            ),
        )

        proceed = Event()
        wait_calls = 0
        cancellation_checks = 0

        async def delayed_lifecycle(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            await proceed.wait()
            return accepted_result

        async def timeout_once(awaitable: Any, *, timeout: float) -> object:
            nonlocal wait_calls
            self.assertEqual(timeout, 0.1)
            wait_calls += 1
            if wait_calls == 1:
                awaitable.cancel()
                raise TimeoutError
            return await awaitable

        async def release_lifecycle(_run_id: str) -> None:
            nonlocal cancellation_checks
            cancellation_checks += 1
            if cancellation_checks == 2:
                proceed.set()

        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                new=delayed_lifecycle,
            ),
            patch.object(worker_module, "wait_for", new=timeout_once),
            patch.object(worker, "_check_cancelled", new=release_lifecycle),
            patch.object(
                worker_module,
                "task_container_output_artifacts",
                AsyncMock(return_value="delayed output"),
            ),
            patch.object(
                worker,
                "_record_output_artifacts",
                AsyncMock(),
            ),
        ):
            delayed = await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )
        self.assertEqual(
            delayed,
            TaskContainerAttemptResult(
                output="delayed output",
                output_artifacts_recorded=True,
            ),
        )
        self.assertEqual(wait_calls, 2)
        self.assertEqual(cancellation_checks, 2)

        never_complete = Event()

        async def blocked_lifecycle(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            await never_complete.wait()
            return accepted_result

        async def fail_wait(awaitable: Any, *, timeout: float) -> object:
            self.assertEqual(timeout, 0.1)
            awaitable.cancel()
            raise RuntimeError("lifecycle wait failed")

        with (
            container_patches(),
            patch.object(
                worker_module,
                "run_container_managed_lifecycle",
                new=blocked_lifecycle,
            ),
            patch.object(worker_module, "wait_for", new=fail_wait),
            self.assertRaisesRegex(RuntimeError, "lifecycle wait failed"),
        ):
            await worker._run_task_container(
                self.definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=sanitizer,
            )

    async def test_worker_container_execution_and_input_file_helpers(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        claim = await self._claim()
        run, attempt, segment = await worker._start_claimed_attempt(
            claim,
            previous_segment=None,
        )
        sanitizer = worker._sanitizer(self.definition)

        with self.assertRaisesRegex(
            TaskWorkerError,
            "durable resume target was not prepared",
        ):
            await worker._execute(
                self.definition,
                claim=claim,
                run=run,
                attempt=attempt,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=FakeDurableResumeHandle(),
                durable_target=None,
                heartbeat_task=None,
            )

        record_output = AsyncMock()
        with (
            patch.object(worker, "_input_files", AsyncMock(return_value=())),
            patch.object(
                worker_module,
                "task_container_input_mount_manifest",
                return_value=(),
            ),
            patch.object(
                worker,
                "_run_task_container",
                AsyncMock(
                    return_value=TaskContainerAttemptResult(
                        output="container output"
                    )
                ),
            ),
            patch.object(worker, "_check_cancelled", AsyncMock()),
            patch.object(worker, "_record_output_artifacts", record_output),
        ):
            outcome = await worker._execute(
                self.definition,
                claim=claim,
                run=run,
                attempt=attempt,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=None,
                durable_target=None,
                heartbeat_task=None,
            )
        self.assertEqual(
            outcome,
            completed_task_target_outcome("container output"),
        )
        record_output.assert_awaited_once()

        materialized = object()
        converted = TaskInputFile(logical_path="artifact:converted")
        plain = TaskInputFile(logical_path="provider:plain")
        entries = (
            SimpleNamespace(
                file=TaskInputFile(logical_path="source:materialized"),
                materialized_file=materialized,
            ),
            SimpleNamespace(file=plain, materialized_file=None),
        )
        payload_run = replace(
            run,
            request=replace(
                run.request,
                input_payload=TaskExecutionPayload(
                    input_value=None,
                    file_values=({"file": "value"},),
                ),
            ),
        )
        convert_files = AsyncMock(return_value=((converted,),))
        with (
            patch.object(
                worker,
                "_decrypt_file_payload_value",
                return_value=object(),
            ),
            patch.object(
                worker_module,
                "task_execution_file_entries_from_value",
                return_value=entries,
            ),
            patch.object(
                worker_module,
                "task_input_file_groups_from_materialized",
                convert_files,
            ),
        ):
            files = await worker._input_files(
                self.definition,
                payload_run,
                attempt,
            )
        self.assertEqual(files, (converted, plain))
        convert_files.assert_awaited_once()

        disabled_definition = replace(
            self.definition,
            observability=TaskObservabilityPolicy.noop(),
        )
        await worker._record_container_event(
            disabled_definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            event_type="container_disabled",
            plans=cast(Any, object()),
            input_mounts=(),
            status="disabled",
        )
        with patch.object(
            worker_module,
            "task_container_event_payload",
            return_value={"status": "completed"},
        ):
            await worker._record_container_event(
                self.definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
                event_type="container_completed",
                plans=cast(Any, object()),
                input_mounts=(),
                status="completed",
            )

    async def _expired_terminal_commit(
        self,
    ) -> TaskDurableExpiredReentryCommit:
        item = self.queue.item
        assert item is not None
        run = await self.store.get_run(self.run.run_id)
        attempt = await self.store.get_attempt(run.last_attempt_id or "")
        result = TaskExecutionResult(error={"code": "expired"})
        return TaskDurableExpiredReentryCommit(
            completion=TaskQueueCompletion(
                queue_item=replace(
                    item,
                    state=TaskQueueItemState.DEAD,
                    run_state=TaskRunState.EXPIRED,
                ),
                run=replace(
                    run,
                    state=TaskRunState.EXPIRED,
                    result=result,
                ),
                attempt=replace(
                    attempt,
                    state=TaskAttemptState.FAILED,
                    result=result,
                ),
            )
        )

    async def _assert_nested_process_control_propagates(
        self,
        process_control: BaseException,
    ) -> BaseExceptionGroup:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        resume_coordinator = FakeDurableResumeCoordinator(
            FakeDurableResumeHandle()
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            heartbeat_seconds=30,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )
        grouped = BaseExceptionGroup(
            "expiry with process control",
            (
                InputValidationError(
                    InputErrorCode.EXPIRED,
                    "continuation",
                    "continuation expired",
                ),
                process_control,
            ),
        )

        with patch.object(
            resume_coordinator,
            "admit",
            side_effect=grouped,
        ):
            with self.assertRaises(BaseExceptionGroup) as caught:
                await worker.process_once()

        self.assertIs(caught.exception, grouped)
        self.assertTrue(
            any(
                isinstance(error, type(process_control))
                for error in caught.exception.exceptions
            )
        )
        self.assertEqual(coordinator.expired_calls, [])
        self.assertEqual(coordinator.rejected_admissions, [])
        self.assertEqual(coordinator.failed_reentries, [])
        self.assertEqual(coordinator.claimed_releases, [])
        self.assertEqual(target.resume_contexts, [])
        return caught.exception

    async def _skills_failure_code(
        self,
        definition: TaskDefinition,
        *,
        skills_settings: TrustedSkillSettings | None = None,
        skills_registry: SkillRegistry | None = None,
    ) -> str:
        await self._use_definition(
            replace(definition, retry=TaskRetryPolicy(max_attempts=1))
        )
        target = FakeTarget("unused")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            skills_settings=skills_settings,
            skills_registry=skills_registry,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        assert self.queue.completed.run.result is not None
        error = self.queue.completed.run.result.error
        assert isinstance(error, Mapping)
        details = error.get("details")
        if isinstance(details, Mapping):
            issues = details.get("issues")
            if isinstance(issues, list | tuple) and issues:
                issue = issues[0]
                if isinstance(issue, Mapping):
                    return cast(str, issue["code"])
        return cast(str, error["code"])

    def _target_context(self, claim: TaskQueueClaim) -> TaskTargetContext:
        async def check_cancelled() -> None:
            return None

        return TaskTargetContext(
            definition=self.definition,
            execution=claim.attempt.context,
            input_value={},
            cancellation_checker=check_cancelled,
        )

    async def test_process_once_completes_claimed_run(self) -> None:
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(
            result.completion.run.result.output_summary,
            {"privacy": "<redacted>"},
        )
        self.assertEqual(
            target.contexts[0].execution.claim.worker_id, "worker-1"
        )
        self.assertIsNone(target.contexts[0].input_value)

    async def test_durable_suspension_uses_atomic_coordinator_first(
        self,
    ) -> None:
        target = DurableSuspendingTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        self.queue.require_durable_coordinator = True
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.suspension)
        assert result.suspension is not None
        self.assertEqual(len(coordinator.calls), 1)
        self.assertEqual(self.queue.suspend_claim_calls, 1)
        assert target.suspension is not None
        command, continuation = coordinator.calls[0]
        self.assertIs(command, target.suspension.command)
        self.assertIs(continuation, target.suspension.continuation)
        self.assertEqual(
            command.request.origin.run_id,
            RunId(self.run.run_id),
        )
        self.assertEqual(
            result.suspension.request_id,
            str(command.request.request_id),
        )
        self.assertEqual(
            result.suspension.continuation_id,
            str(command.request.continuation_id),
        )
        self.assertIsNone(result.suspension.run.claim)
        self.assertIsNone(result.suspension.queue_item.claim_token)
        self.assertIsNone(result.suspension.queue_item.lease_expires_at)
        self.assertEqual(result.suspension.queue_item.attempts, 0)

    async def test_fresh_admission_rejection_closes_returned_handle(
        self,
    ) -> None:
        admission = FakeDurableResumeHandle()

        class FreshAdmissionCoordinator:
            async def admit(
                self,
                claim: TaskQueueClaim,
                previous_segment: TaskAttemptSegment | None,
                *,
                claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
            ) -> FakeDurableResumeHandle:
                del claim, claim_lease_manager
                assert previous_segment is None
                return admission

        target = FakeTarget("must not run")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_resume_coordinator=cast(
                Any,
                FreshAdmissionCoordinator(),
            ),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.abandonment)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(target.contexts, [])

    async def test_durable_reentry_settles_success_without_rerunning_input(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        suspension = await self._prepare_durable_reentry(
            target,
            coordinator,
        )
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.resume_contexts), 1)
        self.assertIsNone(target.resume_contexts[0].input_value)
        self.assertEqual(target.resume_contexts[0].files, ())
        self.assertEqual(len(coordinator.settlements), 1)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeSuccess,
        )
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            segments[1].resumed_from_segment_id,
            suspension.segment.segment_id,
        )

    async def test_durable_reentry_cancellation_settles_cancelled(
        self,
    ) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=CancelledError(),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.CANCELLED)
        self.assertEqual(
            result.completion.attempt.state, TaskAttemptState.FAILED
        )
        self.assertIs(
            type(coordinator.settlements[-1]),
            TaskDurableResumeCancellation,
        )
        self.assertEqual(admission.close_calls, 1)
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.ABANDONED,
        )

    async def test_external_cancellation_releases_predispatch_reentry(
        self,
    ) -> None:
        target = DelayedDurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await target.resume_started.wait()

        self.assertTrue(running.cancel())
        result = await running

        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.reentry)
        assert result.reentry is not None
        self.assertEqual(result.reentry.run.state, TaskRunState.QUEUED)
        self.assertIsNone(result.reentry.run.claim)
        self.assertEqual(
            result.reentry.attempt.state,
            TaskAttemptState.SUSPENDED,
        )
        self.assertEqual(
            result.reentry.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertIsNone(result.reentry.queue_item.claim_token)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.RELEASED,
        )
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.running_releases), 1)

    async def test_external_cancellation_fences_blocked_provider_once(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = ShieldedBlockingDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await admission.provider_started.wait()

        self.assertTrue(running.cancel())
        result = await running

        self.assertTrue(running.done())
        self.assertFalse(running.cancelled())
        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.completion.run.claim)
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertIsNone(result.completion.queue_item.claim_token)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.AMBIGUOUS,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(admission.dispatch_wait_calls, 1)
        self.assertEqual(admission.provider_cancellations, 1)
        self.assertEqual(admission.provider_completions, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.ambiguities), 1)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeFailure,
        )
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.FAILED,
        )

        no_work = await worker.process_once()

        self.assertFalse(no_work.processed)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.resume_contexts), 1)

    async def test_external_cancellation_preserves_dispatch_winning_race(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = ShieldedBlockingDurableResumeHandle(
            dispatch_wins_interruption=True
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await admission.provider_started.wait()

        self.assertTrue(running.cancel())
        result = await running

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.CANCELLED)
        self.assertIsNone(result.completion.run.claim)
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.DISPATCHED,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(admission.dispatch_wait_calls, 1)
        self.assertEqual(admission.provider_cancellations, 0)
        self.assertEqual(admission.provider_completions, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.ambiguities), 0)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeCancellation,
        )
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.ABANDONED,
        )

        no_work = await worker.process_once()

        self.assertFalse(no_work.processed)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)

    async def test_repeat_external_cancellation_converges_failure_once(
        self,
    ) -> None:
        cases = (
            DurableContinuationResumeState.RELEASED,
            DurableContinuationResumeState.DISPATCHED,
            DurableContinuationResumeState.AMBIGUOUS,
        )
        for terminal_state in cases:
            with self.subTest(
                terminal_state=terminal_state.value,
            ):
                await self._use_definition(_definition())
                target = DurableResumableTarget(self.now)
                coordinator = FakeDurableSuspensionCoordinator(self.queue)
                await self._prepare_durable_reentry(target, coordinator)
                admission = GatedFailureConvergenceDurableResumeHandle(
                    terminal_state,
                )
                coordinator.convergence_commit_started = Event()
                coordinator.convergence_commit_proceed = Event()
                unbind_started = Event()
                unbind_proceed = Event()
                unbind_calls = 0
                unbind_completed = 0
                original_unbind = _TaskResumeClaimLeaseBridge.unbind

                async def gated_unbind(
                    bridge: _TaskResumeClaimLeaseBridge,
                ) -> None:
                    nonlocal unbind_calls, unbind_completed
                    unbind_calls += 1
                    unbind_started.set()
                    await unbind_proceed.wait()
                    await original_unbind(bridge)
                    unbind_completed += 1

                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    durable_suspension_coordinator=coordinator,
                    durable_resume_coordinator=FakeDurableResumeCoordinator(
                        admission
                    ),
                    clock=lambda: self.now,
                )
                with patch.object(
                    _TaskResumeClaimLeaseBridge,
                    "unbind",
                    gated_unbind,
                ):
                    running = create_task(worker.process_once())
                    await admission.convergence_dispatch_started.wait()

                    self.assertTrue(running.cancel())
                    assert coordinator.convergence_commit_started is not None
                    await coordinator.convergence_commit_started.wait()
                    self.assertTrue(running.cancel())
                    self.assertFalse(running.done())
                    assert coordinator.convergence_commit_proceed is not None
                    coordinator.convergence_commit_proceed.set()
                    await unbind_started.wait()
                    self.assertTrue(running.cancel())
                    self.assertFalse(running.done())
                    unbind_proceed.set()
                    await admission.close_started.wait()
                    self.assertTrue(running.cancel())
                    self.assertFalse(running.done())
                    admission.close_proceed.set()
                    result = await wait_for(shield(running), timeout=1)

                self.assertTrue(running.done())
                self.assertFalse(running.cancelled())
                self.assertEqual(running.cancelling(), 0)
                self.assertEqual(
                    result.shutdown_requested,
                    terminal_state
                    in {
                        DurableContinuationResumeState.RELEASED,
                        DurableContinuationResumeState.AMBIGUOUS,
                    },
                )
                self.assertEqual(admission.state, terminal_state)
                self.assertEqual(admission.dispatch_calls, 1)
                self.assertEqual(admission.interrupt_calls, 1)
                self.assertEqual(admission.dispatch_wait_calls, 1)
                self.assertEqual(admission.close_calls, 1)
                self.assertEqual(admission.close_completed, 1)
                self.assertEqual(coordinator.convergence_commit_calls, 1)
                self.assertEqual(unbind_calls, 1)
                self.assertEqual(unbind_completed, 1)
                if terminal_state is DurableContinuationResumeState.RELEASED:
                    self.assertIsNotNone(result.reentry)
                    assert result.reentry is not None
                    terminal_queue_item = result.reentry.queue_item
                    terminal_run = result.reentry.run
                    terminal_attempt = result.reentry.attempt
                    expected_segment_state = TaskAttemptSegmentState.SUSPENDED
                    self.assertEqual(len(coordinator.running_releases), 1)
                    self.assertEqual(coordinator.settlements, [])
                    self.assertEqual(coordinator.ambiguities, [])
                else:
                    self.assertIsNotNone(result.completion)
                    assert result.completion is not None
                    terminal_queue_item = result.completion.queue_item
                    terminal_run = result.completion.run
                    terminal_attempt = result.completion.attempt
                    expected_segment_state = (
                        TaskAttemptSegmentState.ABANDONED
                        if terminal_state
                        is DurableContinuationResumeState.DISPATCHED
                        else TaskAttemptSegmentState.FAILED
                    )
                    self.assertEqual(coordinator.running_releases, [])
                    self.assertEqual(len(coordinator.settlements), 1)
                    self.assertEqual(
                        len(coordinator.ambiguities),
                        int(
                            terminal_state
                            is DurableContinuationResumeState.AMBIGUOUS
                        ),
                    )
                self.assertIsNot(
                    terminal_queue_item.state,
                    TaskQueueItemState.CLAIMED,
                )
                self.assertIsNot(terminal_run.state, TaskRunState.RUNNING)
                self.assertIsNot(
                    terminal_attempt.state,
                    TaskAttemptState.RUNNING,
                )
                self.assertIsNone(terminal_run.claim)
                self.assertIsNone(terminal_queue_item.claim_token)
                segments = await self.store.list_attempt_segments(
                    terminal_attempt.attempt_id
                )
                self.assertEqual(
                    segments[-1].state,
                    expected_segment_state,
                )
                self.assertFalse(
                    tuple(
                        task.get_name()
                        for task in all_tasks()
                        if task.get_name().startswith(
                            (
                                "task-durable-failure-convergence-",
                                "task-claim-cleanup-",
                                "task-durable-resume-handle-close",
                            )
                        )
                    )
                )

    async def test_completed_provider_failure_terminalizes_without_replay(
        self,
    ) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=RuntimeError(
                "post-provider processing failed"
            ),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = CompletedDigestPinnedDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        completion = result.completion
        self.assertEqual(completion.run.state, TaskRunState.FAILED)
        self.assertEqual(completion.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(
            completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertIsNone(completion.run.claim)
        self.assertIsNone(completion.queue_item.claim_token)
        self.assertIsNone(completion.queue_item.worker_id)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.COMPLETED,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.settlement_command_calls, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(target.durable_resume_support_calls, 1)
        self.assertEqual(len(target.resume_contexts), 1)
        self.assertEqual(coordinator.settlements, [])
        self.assertEqual(coordinator.ambiguities, [])
        self.assertEqual(coordinator.convergence_commit_calls, 1)
        self.assertEqual(len(coordinator.completed_terminalizations), 1)
        self.assertIs(
            type(coordinator.completed_terminalizations[0]),
            TaskDurableResumeFailure,
        )
        assert completion.run.result is not None
        failure = TaskDurableResumeFailure(result=completion.run.result)
        self.assertNotEqual(
            admission.pinned_digest,
            task_durable_resume_settlement_digest(failure),
        )
        segments = await self.store.list_attempt_segments(
            completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.FAILED,
        )

        no_work = await worker.process_once()

        self.assertFalse(no_work.processed)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.settlement_command_calls, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(coordinator.settlements, [])
        self.assertEqual(len(coordinator.completed_terminalizations), 1)

    async def test_completed_provider_cancellation_is_atomic(self) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=CancelledError(),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = CompletedDigestPinnedDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(
            result.completion.run.state,
            TaskRunState.CANCELLED,
        )
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(len(coordinator.completed_terminalizations), 1)
        self.assertIs(
            type(coordinator.completed_terminalizations[0]),
            TaskDurableResumeCancellation,
        )
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.ABANDONED,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.settlement_command_calls, 0)

    async def test_completed_provider_failure_requires_atomic_coordinator(
        self,
    ) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=RuntimeError("post-provider failure"),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        unavailable = SimpleNamespace(
            create_and_suspend=coordinator.create_and_suspend,
        )
        admission = CompletedDigestPinnedDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=cast(Any, unavailable),
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        with self.assertRaisesRegex(
            TaskWorkerError,
            "completed durable task settlement coordinator is unavailable",
        ):
            await worker.process_once()

        assert self.queue.item is not None
        self.assertEqual(
            self.queue.item.state,
            TaskQueueItemState.CLAIMED,
        )
        run = await self.store.get_run(self.run.run_id)
        assert run.last_attempt_id is not None
        attempt = await self.store.get_attempt(run.last_attempt_id)
        segments = await self.store.list_attempt_segments(attempt.attempt_id)
        self.assertEqual(run.state, TaskRunState.RUNNING)
        self.assertEqual(attempt.state, TaskAttemptState.RUNNING)
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.RUNNING,
        )
        self.assertIsNotNone(run.claim)
        self.assertIsNotNone(self.queue.item.claim_token)

    async def test_cancellation_during_heartbeat_cleanup_preserves_result(
        self,
    ) -> None:
        target = DelayedDurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        heartbeat_started = Event()
        heartbeat_cancel_started = Event()
        heartbeat_cancel_proceed = Event()
        heartbeat_cancel_calls = 0
        heartbeat_cancel_completed = 0

        async def gated_heartbeat(
            claim: TaskQueueClaim,
            heartbeat_seconds: float | None = None,
            *,
            claim_lease_manager: _TaskResumeClaimLeaseBridge | None = None,
        ) -> None:
            nonlocal heartbeat_cancel_calls, heartbeat_cancel_completed
            del claim, heartbeat_seconds, claim_lease_manager
            heartbeat_started.set()
            try:
                await Event().wait()
            except CancelledError:
                heartbeat_cancel_calls += 1
                heartbeat_cancel_started.set()
                await heartbeat_cancel_proceed.wait()
                heartbeat_cancel_completed += 1
                raise

        with patch.object(
            worker,
            "_heartbeat_claim",
            side_effect=gated_heartbeat,
        ):
            running = create_task(worker.process_once())
            await target.resume_started.wait()
            await heartbeat_started.wait()
            target.resume_proceed.set()
            await heartbeat_cancel_started.wait()
            self.assertTrue(running.cancel())
            self.assertFalse(running.done())
            heartbeat_cancel_proceed.set()
            result = await wait_for(shield(running), timeout=1)

        self.assertFalse(running.cancelled())
        self.assertEqual(running.cancelling(), 0)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DONE,
        )
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.SUCCEEDED,
        )
        segments = await self.store.list_attempt_segments(
            result.completion.attempt.attempt_id
        )
        self.assertEqual(
            segments[-1].state,
            TaskAttemptSegmentState.SUCCEEDED,
        )
        self.assertEqual(heartbeat_cancel_calls, 1)
        self.assertEqual(heartbeat_cancel_completed, 1)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.settlements), 1)
        self.assertFalse(
            tuple(
                task.get_name()
                for task in all_tasks()
                if task.get_name().startswith(
                    (
                        "task-claim-heartbeat-",
                        "task-claim-heartbeat-cleanup-",
                    )
                )
            )
        )

    async def test_convergence_child_errors_propagate_after_cleanup(
        self,
    ) -> None:
        for error in (
            CancelledError("convergence self-cancelled"),
            RuntimeError("convergence failed"),
        ):
            with self.subTest(error=type(error).__name__):
                await self._use_definition(_definition())
                target = DurableResumableTarget(self.now)
                coordinator = FakeDurableSuspensionCoordinator(self.queue)
                await self._prepare_durable_reentry(target, coordinator)
                admission = GatedFailureConvergenceDurableResumeHandle(
                    DurableContinuationResumeState.DISPATCHED,
                )
                admission.close_proceed.set()
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    durable_suspension_coordinator=coordinator,
                    durable_resume_coordinator=FakeDurableResumeCoordinator(
                        admission
                    ),
                    clock=lambda: self.now,
                )

                with patch.object(
                    coordinator,
                    "settle_resume",
                    side_effect=error,
                ) as settle_resume:
                    running = create_task(worker.process_once())
                    await admission.convergence_dispatch_started.wait()
                    self.assertTrue(running.cancel())
                    with self.assertRaises(type(error)) as caught:
                        await wait_for(shield(running), timeout=1)

                if isinstance(error, RuntimeError):
                    self.assertIs(caught.exception, error)
                    self.assertFalse(running.cancelled())
                else:
                    self.assertTrue(running.cancelled())
                self.assertEqual(settle_resume.await_count, 1)
                self.assertEqual(admission.dispatch_calls, 1)
                self.assertEqual(admission.interrupt_calls, 1)
                self.assertEqual(admission.dispatch_wait_calls, 1)
                self.assertEqual(admission.close_calls, 1)
                self.assertEqual(admission.close_completed, 1)
                self.assertEqual(coordinator.settlements, [])
                self.assertEqual(coordinator.running_releases, [])
                self.assertEqual(coordinator.ambiguities, [])
                item = self.queue.item
                assert item is not None
                self.assertEqual(item.state, TaskQueueItemState.CLAIMED)
                run = await self.store.get_run(item.run_id)
                attempt = await self.store.get_attempt(
                    run.last_attempt_id or ""
                )
                segments = await self.store.list_attempt_segments(
                    attempt.attempt_id
                )
                self.assertEqual(run.state, TaskRunState.RUNNING)
                self.assertEqual(attempt.state, TaskAttemptState.RUNNING)
                self.assertEqual(
                    segments[-1].state,
                    TaskAttemptSegmentState.RUNNING,
                )
                self.assertFalse(
                    tuple(
                        task.get_name()
                        for task in all_tasks()
                        if task.get_name().startswith(
                            (
                                "task-durable-failure-convergence-",
                                "task-claim-cleanup-",
                                "task-durable-resume-handle-close",
                                "task-claim-heartbeat-",
                            )
                        )
                    )
                )

    async def test_nested_cancellation_interrupts_before_failure_settlement(
        self,
    ) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=BaseExceptionGroup(
                "cancelled cleanup failed",
                (
                    CancelledError(),
                    RuntimeError("runtime cleanup failed"),
                ),
            ),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.DISPATCHED,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.ambiguities), 0)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeFailure,
        )

    async def test_unrelated_cancellation_type_does_not_interrupt_dispatch(
        self,
    ) -> None:
        target = DurableResumableTarget(
            self.now,
            error_after_dispatch=FutureCancelledError(
                "unrelated future cancellation"
            ),
        )
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.DISPATCHED,
        )
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeFailure,
        )

    async def test_durable_reentry_marks_provider_ambiguity_without_retry(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle(
            dispatch_error=OSError("private provider state"),
            error_state=DurableContinuationResumeState.AMBIGUOUS,
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        self.assertIsNone(result.retry)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(result.completion.queue_item.attempts, 0)
        self.assertEqual(len(coordinator.ambiguities), 1)
        self.assertEqual(admission.close_calls, 1)
        assert result.completion.run.result is not None
        self.assertEqual(
            result.completion.run.result.metadata["durable_resume"],
            "provider_dispatch_ambiguous",
        )
        self.assertNotIn(
            "private provider state",
            str(result.completion.run.result),
        )

    async def test_durable_reentry_releases_safe_predispatch_failure(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        suspension = await self._prepare_durable_reentry(
            target,
            coordinator,
        )
        admission = FakeDurableResumeHandle(
            dispatch_error=OSError("safe pre-dispatch failure"),
            error_state=DurableContinuationResumeState.RELEASED,
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.reentry)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        assert result.reentry is not None
        self.assertEqual(result.reentry.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            result.reentry.attempt.state,
            TaskAttemptState.SUSPENDED,
        )
        self.assertEqual(result.reentry.queue_item.attempts, 0)
        self.assertEqual(
            result.reentry.previous_segment.request_id,
            suspension.request_id,
        )
        self.assertEqual(len(coordinator.running_releases), 1)
        self.assertEqual(admission.close_calls, 1)

    async def test_durable_start_conflict_releases_and_closes_admission(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        with patch.object(
            worker,
            "_start_claimed_attempt",
            side_effect=TaskStoreConflictError("stale start"),
        ):
            result = await worker.process_once()

        self.assertTrue(result.lease_lost)
        self.assertEqual(admission.release_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(target.resume_contexts, [])

    async def test_durable_start_retryable_failure_requeues_and_closes(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        with patch.object(
            worker,
            "_start_claimed_attempt",
            side_effect=OSError("cold store unavailable"),
        ):
            result = await worker.process_once()

        self.assertIsNotNone(result.reentry)
        self.assertEqual(len(coordinator.claimed_releases), 1)
        self.assertEqual(admission.release_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(target.resume_contexts, [])

    async def test_durable_start_terminal_failures_close_without_rerun(
        self,
    ) -> None:
        for error in (RuntimeError("broken start"), CancelledError()):
            with self.subTest(error=type(error).__name__):
                await self._use_definition(_definition())
                target = DurableResumableTarget(self.now)
                coordinator = FakeDurableSuspensionCoordinator(self.queue)
                await self._prepare_durable_reentry(target, coordinator)
                admission = FakeDurableResumeHandle()
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    durable_suspension_coordinator=coordinator,
                    durable_resume_coordinator=(
                        FakeDurableResumeCoordinator(admission)
                    ),
                    clock=lambda: self.now,
                )

                with patch.object(
                    worker,
                    "_start_claimed_attempt",
                    side_effect=error,
                ):
                    result = await worker.process_once()

                if isinstance(error, CancelledError):
                    self.assertIsNotNone(result.completion)
                    self.assertEqual(
                        len(coordinator.rejected_admissions),
                        1,
                    )
                    self.assertEqual(admission.release_calls, 0)
                else:
                    self.assertIsNotNone(result.reentry)
                    self.assertEqual(len(coordinator.claimed_releases), 1)
                    self.assertEqual(admission.release_calls, 1)
                self.assertEqual(admission.close_calls, 1)
                self.assertEqual(admission.dispatch_calls, 0)
                self.assertEqual(len(target.contexts), 1)
                self.assertEqual(target.resume_contexts, [])

    async def test_durable_heartbeat_loss_after_start_releases_and_closes(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        started = Event()
        heartbeat_failed = Event()
        original_start = worker._start_claimed_attempt

        async def start_then_observe_heartbeat(
            claim: TaskQueueClaim,
            *,
            previous_segment: TaskAttemptSegment | None,
        ) -> tuple[TaskRun, TaskAttempt, TaskAttemptSegment]:
            result = await original_start(
                claim,
                previous_segment=previous_segment,
            )
            started.set()
            await heartbeat_failed.wait()
            return result

        async def fail_after_start(
            queue_item_id: str,
            *,
            claim_token: str,
            lease_expires_at: datetime,
            now: datetime | None = None,
        ) -> TaskQueueItem:
            del queue_item_id, claim_token, lease_expires_at, now
            await started.wait()
            heartbeat_failed.set()
            raise RuntimeError("heartbeat lost after start")

        with (
            patch.object(
                worker,
                "_start_claimed_attempt",
                new=start_then_observe_heartbeat,
            ),
            patch.object(self.queue, "heartbeat", new=fail_after_start),
        ):
            result = await worker.process_once()

        self.assertIsNotNone(result.reentry)
        self.assertEqual(len(coordinator.running_releases), 1)
        self.assertEqual(admission.release_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(target.resume_contexts, [])

    async def test_durable_output_validation_failure_closes_once(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle(output={"invalid": "text"})
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(coordinator.settlements), 1)
        self.assertIs(
            type(coordinator.settlements[0]),
            TaskDurableResumeFailure,
        )
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.resume_contexts), 1)

    async def test_durable_successor_suspension_closes_once(
        self,
    ) -> None:
        target = ResuspendingDurableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.suspension)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.resume_contexts), 1)

    async def test_durable_shutdown_releases_before_dispatch(self) -> None:
        shutdown = TaskWorkerShutdown()
        target = DelayedDurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await target.resume_started.wait()

        shutdown.request()
        result = await running

        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.reentry)
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.RELEASED,
        )
        self.assertEqual(len(coordinator.running_releases), 1)
        self.assertEqual(coordinator.ambiguities, [])

    async def test_durable_shutdown_fences_inflight_dispatch(self) -> None:
        shutdown = TaskWorkerShutdown()
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        dispatch_started = Event()
        admission = FakeDurableResumeHandle(
            dispatch_started=dispatch_started,
            dispatch_proceed=Event(),
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await dispatch_started.wait()

        shutdown.request()
        result = await running

        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.AMBIGUOUS,
        )
        self.assertEqual(len(coordinator.ambiguities), 1)
        self.assertEqual(coordinator.running_releases, [])

    async def test_durable_heartbeat_loss_interrupts_inflight_dispatch(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        self.queue.heartbeat_error = RuntimeError("private heartbeat")
        dispatch_started = Event()
        admission = FakeDurableResumeHandle(
            dispatch_started=dispatch_started,
            dispatch_proceed=Event(),
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(admission),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertFalse(result.lease_lost)
        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.AMBIGUOUS,
        )
        self.assertEqual(len(coordinator.ambiguities), 1)
        self.assertNotIn("private heartbeat", str(result))

    async def test_durable_reentry_heartbeats_while_cold_admission_blocks(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        resume_coordinator = FakeDurableResumeCoordinator(admission)
        admission_started = Event()
        admission_proceed = Event()
        original_admit = resume_coordinator.admit

        async def blocked_admit(
            claim: TaskQueueClaim,
            previous_segment: TaskAttemptSegment | None,
            *,
            claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
        ) -> FakeDurableResumeHandle:
            admission_started.set()
            await admission_proceed.wait()
            return await original_admit(
                claim,
                previous_segment,
                claim_lease_manager=claim_lease_manager,
            )

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        with patch.object(
            resume_coordinator,
            "admit",
            new=blocked_admit,
        ):
            running = create_task(worker.process_once())
            await admission_started.wait()
            for _ in range(100):
                if self.queue.heartbeats:
                    break
                await sleep(0.001)
            self.assertGreaterEqual(len(self.queue.heartbeats), 1)
            self.assertEqual(target.resume_contexts, [])
            admission_proceed.set()
            result = await running

        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)

    async def test_cold_admission_expiry_cancels_loader_before_reconcile(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        coordinator.expired_commit = await self._expired_terminal_commit()
        deadline = self.now + timedelta(seconds=5)
        admission = FakeDurableResumeHandle()
        admission_started = Event()
        admission_proceed = Event()
        admission_cancelled = Event()
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=deadline,
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            admission,
            claim_lease=claim_lease,
            admission_started=admission_started,
            admission_proceed=admission_proceed,
        )
        original_admit = resume_coordinator.admit

        async def observed_admit(
            claim: TaskQueueClaim,
            previous_segment: TaskAttemptSegment | None,
            *,
            claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
        ) -> FakeDurableResumeHandle:
            try:
                return await original_admit(
                    claim,
                    previous_segment,
                    claim_lease_manager=claim_lease_manager,
                )
            except CancelledError:
                admission_cancelled.set()
                raise

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        with patch.object(
            resume_coordinator,
            "admit",
            new=observed_admit,
        ):
            running = create_task(worker.process_once())
            await admission_started.wait()
            self.now = deadline
            result = await wait_for(running, timeout=1)

        self.assertTrue(admission_cancelled.is_set())
        self.assertFalse(admission_proceed.is_set())
        self.assertIs(result.completion, coordinator.expired_commit.completion)
        self.assertTrue(result.lease_lost)
        self.assertEqual(len(coordinator.expired_calls), 1)
        self.assertEqual(coordinator.rejected_admissions, [])
        self.assertEqual(coordinator.failed_reentries, [])
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(admission.close_calls, 0)
        self.assertEqual(target.resume_contexts, [])

    async def test_cold_expiry_closes_cancellation_racing_admission_once(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        coordinator.expired_commit = await self._expired_terminal_commit()
        deadline = self.now + timedelta(seconds=5)
        admission = FakeDurableResumeHandle()
        admission_started = Event()
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=deadline,
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            admission,
            claim_lease=claim_lease,
        )

        async def late_admit(
            claim: TaskQueueClaim,
            previous_segment: TaskAttemptSegment | None,
            *,
            claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
        ) -> FakeDurableResumeHandle:
            assert previous_segment is not None
            assert claim_lease_manager is not None
            current_lease = (
                await claim_lease_manager.current_lease_expires_at()
            )
            claim_lease.lease_expires_at = current_lease
            await claim_lease_manager.bind(
                cast(
                    DurableAgentContinuationClaimLease,
                    claim_lease,
                )
            )
            admission_started.set()
            try:
                await Event().wait()
            except CancelledError:
                return admission
            raise AssertionError("late admission unexpectedly resumed")

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        with patch.object(
            resume_coordinator,
            "admit",
            new=late_admit,
        ):
            running = create_task(worker.process_once())
            await admission_started.wait()
            self.now = deadline
            result = await wait_for(running, timeout=1)

        self.assertIs(result.completion, coordinator.expired_commit.completion)
        self.assertEqual(len(coordinator.expired_calls), 1)
        self.assertEqual(coordinator.rejected_admissions, [])
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(admission.release_calls, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(target.resume_contexts, [])

    async def test_cold_expiry_reconciles_then_surfaces_cleanup_failure(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        coordinator.expired_commit = await self._expired_terminal_commit()
        deadline = self.now + timedelta(seconds=5)
        admission_started = Event()
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=deadline,
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            FakeDurableResumeHandle(),
            claim_lease=claim_lease,
        )

        async def cleanup_failing_admit(
            claim: TaskQueueClaim,
            previous_segment: TaskAttemptSegment | None,
            *,
            claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
        ) -> FakeDurableResumeHandle:
            assert previous_segment is not None
            assert claim_lease_manager is not None
            await claim_lease_manager.bind(
                cast(
                    DurableAgentContinuationClaimLease,
                    claim_lease,
                )
            )
            admission_started.set()
            try:
                await Event().wait()
            except CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "runtime cleanup failed",
                    (
                        cancellation,
                        InputValidationError(
                            InputErrorCode.EXPIRED,
                            "continuation",
                            "continuation expired",
                        ),
                        RuntimeError("executor close failed"),
                    ),
                ) from None
            raise AssertionError("blocked admission unexpectedly resumed")

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        with patch.object(
            resume_coordinator,
            "admit",
            new=cleanup_failing_admit,
        ):
            running = create_task(worker.process_once())
            await admission_started.wait()
            self.now = deadline
            with self.assertRaisesRegex(
                RuntimeError,
                "executor close failed",
            ):
                await wait_for(running, timeout=1)

        self.assertEqual(len(coordinator.expired_calls), 1)
        self.assertEqual(coordinator.rejected_admissions, [])
        self.assertEqual(coordinator.failed_reentries, [])
        self.assertEqual(target.resume_contexts, [])

    async def test_nested_expiry_keyboard_interrupt_propagates(self) -> None:
        process_control = KeyboardInterrupt()
        caught = await self._assert_nested_process_control_propagates(
            process_control
        )
        self.assertIn(process_control, caught.exceptions)

    async def test_nested_expiry_system_exit_propagates(self) -> None:
        process_control = SystemExit(7)
        caught = await self._assert_nested_process_control_propagates(
            process_control
        )
        self.assertIn(process_control, caught.exceptions)

    async def test_durable_cold_admission_renews_continuation_lease(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        admission_started = Event()
        admission_proceed = Event()
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=self.now + timedelta(minutes=1)
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            admission,
            claim_lease=claim_lease,
            admission_started=admission_started,
            admission_proceed=admission_proceed,
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )
        initial_lease_expires_at = self.now + timedelta(seconds=3)
        running = create_task(worker.process_once())
        await admission_started.wait()

        self.now += timedelta(seconds=4)
        for _ in range(100):
            if (
                claim_lease.lease_expires_at is not None
                and claim_lease.lease_expires_at > initial_lease_expires_at
            ):
                break
            await sleep(0.001)
        self.assertIsNotNone(claim_lease.lease_expires_at)
        assert claim_lease.lease_expires_at is not None
        self.assertGreater(
            claim_lease.lease_expires_at,
            initial_lease_expires_at,
        )
        self.assertGreater(claim_lease.lease_expires_at, self.now)
        admission_proceed.set()
        result = await running

        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.resume_contexts), 1)
        manager = resume_coordinator.claim_lease_manager
        assert manager is not None
        self.assertIsNone(
            cast(Any, manager)._claim_lease,  # noqa: SLF001
        )

    async def test_durable_resume_wait_renews_until_dispatch_fence(
        self,
    ) -> None:
        target = DelayedDurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=self.now + timedelta(minutes=1)
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            admission,
            claim_lease=claim_lease,
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )
        initial_lease_expires_at = self.now + timedelta(seconds=3)
        running = create_task(worker.process_once())
        await target.resume_started.wait()

        self.now += timedelta(seconds=4)
        for _ in range(100):
            if (
                claim_lease.lease_expires_at is not None
                and claim_lease.lease_expires_at > initial_lease_expires_at
            ):
                break
            await sleep(0.001)
        self.assertIsNotNone(claim_lease.lease_expires_at)
        assert claim_lease.lease_expires_at is not None
        self.assertGreater(
            claim_lease.lease_expires_at,
            initial_lease_expires_at,
        )
        self.assertEqual(admission.dispatch_calls, 0)
        target.resume_proceed.set()
        result = await running

        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.resume_contexts), 1)

    async def test_durable_dispatch_survives_heartbeats_past_deadline(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        dispatch_started = Event()
        dispatch_proceed = Event()
        admission = FakeDurableResumeHandle(
            dispatch_started=dispatch_started,
            dispatch_proceed=dispatch_proceed,
        )
        deadline = self.now + timedelta(seconds=5)
        claim_lease = FakeDurableContinuationClaimLease(
            expires_at=deadline,
            noop_after_expiry=lambda: (
                admission.state
                in {
                    DurableContinuationResumeState.DISPATCHING,
                    DurableContinuationResumeState.DISPATCHED,
                }
            ),
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=3,
            heartbeat_seconds=0.001,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(
                admission,
                claim_lease=claim_lease,
            ),
            clock=lambda: self.now,
        )
        running = create_task(worker.process_once())
        await dispatch_started.wait()

        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 0)
        self.now = deadline
        for _ in range(100):
            if claim_lease.noop_renewals:
                break
            await sleep(0.001)
        self.assertTrue(claim_lease.noop_renewals)
        self.assertFalse(running.done())
        self.assertEqual(admission.interrupt_calls, 0)

        dispatch_proceed.set()
        result = await running

        self.assertIsNotNone(result.completion)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 0)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(len(target.resume_contexts), 1)

    async def test_durable_reentry_enables_heartbeat_guard_by_default(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            lease_seconds=9,
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=FakeDurableResumeCoordinator(
                FakeDurableResumeHandle()
            ),
            clock=lambda: self.now,
        )
        claim = await self._claim()

        self.assertEqual(worker._claim_heartbeat_seconds(claim), 3)

    async def test_expired_predispatch_reentry_restores_same_attempt(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        suspension = await self._prepare_durable_reentry(
            target,
            coordinator,
        )
        available_item = self.queue.item
        assert available_item is not None
        available_run = await self.store.get_run(self.run.run_id)
        available_attempt = await self.store.get_attempt(
            available_run.last_attempt_id or ""
        )
        restored = TaskQueueReentry(
            queue_item=available_item,
            run=available_run,
            attempt=available_attempt,
            previous_segment=suspension.segment,
        )
        coordinator.expired_commit = TaskDurableExpiredReentryCommit(
            reentry=restored
        )
        claim = await self._claim()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )

        result = await worker._lease_lost_result(
            claim,
            sanitizer=None,
        )

        self.assertTrue(result.lease_lost)
        self.assertIs(result.reentry, restored)
        self.assertEqual(
            result.reentry.attempt.attempt_id,
            claim.attempt.attempt_id,
        )
        self.assertEqual(
            coordinator.expired_calls,
            [
                (
                    claim.queue_item.queue_item_id,
                    claim.queue_item.claim_token,
                    claim.run.run_id,
                )
            ],
        )
        self.assertEqual(target.resume_contexts, [])

    async def test_expired_postdispatch_reentry_returns_terminal_commit(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        claim = await self._claim()
        result_snapshot = TaskExecutionResult(
            error={"code": "infra"},
            metadata={"durable_resume": "provider_dispatch_ambiguous"},
        )
        terminal_run = replace(
            claim.run,
            state=TaskRunState.FAILED,
            claim=None,
            result=result_snapshot,
        )
        terminal_attempt = replace(
            claim.attempt,
            state=TaskAttemptState.FAILED,
            result=result_snapshot,
        )
        terminal_item = replace(
            claim.queue_item,
            state=TaskQueueItemState.DEAD,
            run_state=TaskRunState.FAILED,
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
        )
        terminal = TaskQueueCompletion(
            queue_item=terminal_item,
            run=terminal_run,
            attempt=terminal_attempt,
        )
        coordinator.expired_commit = TaskDurableExpiredReentryCommit(
            completion=terminal
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )

        result = await worker._lease_lost_result(
            claim,
            sanitizer=None,
        )

        self.assertTrue(result.lease_lost)
        self.assertIs(result.completion, terminal)
        self.assertIsNone(result.reentry)
        self.assertEqual(target.resume_contexts, [])

    async def test_durable_reentry_rejects_non_exact_capability_pre_admission(
        self,
    ) -> None:
        for advertisement in (False, 1, "true"):
            with self.subTest(advertisement=advertisement):
                await self._use_definition(_definition())
                target = DurableResumableTarget(
                    self.now,
                    durable_resume_support=advertisement,
                )
                coordinator = FakeDurableSuspensionCoordinator(self.queue)
                suspension = await self._prepare_durable_reentry(
                    target,
                    coordinator,
                )
                admission = FakeDurableResumeHandle()
                resume_coordinator = FakeDurableResumeCoordinator(admission)
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    durable_suspension_coordinator=coordinator,
                    durable_resume_coordinator=resume_coordinator,
                    clock=lambda: self.now,
                )

                result = await worker.process_once()

                self.assertIsNotNone(result.completion)
                assert result.completion is not None
                self.assertEqual(
                    result.completion.run.state,
                    TaskRunState.FAILED,
                )
                self.assertIsNone(result.completion.run.claim)
                self.assertEqual(
                    result.completion.queue_item.state,
                    TaskQueueItemState.DEAD,
                )
                self.assertIsNone(result.completion.queue_item.claim_token)
                self.assertEqual(
                    target.durable_resume_support_calls,
                    1,
                )
                self.assertEqual(target.resume_contexts, [])
                self.assertEqual(resume_coordinator.calls, [])
                self.assertEqual(admission.dispatch_calls, 0)
                self.assertEqual(admission.release_calls, 0)
                self.assertEqual(admission.close_calls, 0)
                self.assertEqual(coordinator.rejected_admissions, [])
                self.assertEqual(len(coordinator.failed_reentries), 1)
                segments = await self.store.list_attempt_segments(
                    result.completion.attempt.attempt_id
                )
                self.assertEqual(len(segments), 1)
                self.assertEqual(
                    segments[0].segment_id,
                    suspension.segment.segment_id,
                )
                self.assertEqual(
                    segments[0].state,
                    TaskAttemptSegmentState.SUSPENDED,
                )

    async def test_durable_reentry_rejects_spoof_prepared_runner(
        self,
    ) -> None:
        target = SpoofDurableResumePreparer(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        admission = FakeDurableResumeHandle()
        resume_coordinator = FakeDurableResumeCoordinator(admission)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(target.prepare_calls, 1)
        self.assertEqual(target.durable_resume_support_calls, 0)
        self.assertEqual(target.resume_contexts, [])
        self.assertEqual(resume_coordinator.calls, [])
        self.assertEqual(admission.dispatch_calls, 0)
        self.assertEqual(admission.close_calls, 0)
        self.assertEqual(len(coordinator.failed_reentries), 1)

    async def test_registry_prepares_stateful_capability_once(self) -> None:
        target = StatefulDurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        registry = TaskTargetRunnerRegistry(target)
        admission = FakeDurableResumeHandle()
        resume_coordinator = FakeDurableResumeCoordinator(admission)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=registry,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.durable_resume_support_calls, 1)
        self.assertEqual(len(target.resume_contexts), 1)
        self.assertEqual(len(resume_coordinator.calls), 1)
        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.close_calls, 1)
        self.assertEqual(coordinator.rejected_admissions, [])
        self.assertEqual(coordinator.failed_reentries, [])

    async def test_expired_cold_reentry_fails_terminal_without_rerun(
        self,
    ) -> None:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        await self._prepare_durable_reentry(target, coordinator)
        available_item = self.queue.item
        assert available_item is not None
        suspended_run = await self.store.get_run(self.run.run_id)
        suspended_attempt = await self.store.get_attempt(
            suspended_run.last_attempt_id or ""
        )
        expired_result = TaskExecutionResult(error={"code": "expired"})
        terminal = TaskQueueCompletion(
            queue_item=replace(
                available_item,
                state=TaskQueueItemState.DEAD,
                run_state=TaskRunState.EXPIRED,
            ),
            run=replace(
                suspended_run,
                state=TaskRunState.EXPIRED,
                result=expired_result,
            ),
            attempt=replace(
                suspended_attempt,
                state=TaskAttemptState.FAILED,
                result=expired_result,
            ),
        )
        coordinator.expired_commit = TaskDurableExpiredReentryCommit(
            completion=terminal
        )
        resume_coordinator = FakeDurableResumeCoordinator(
            FakeDurableResumeHandle()
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: self.now,
        )

        with patch.object(
            resume_coordinator,
            "admit",
            side_effect=InputValidationError(
                InputErrorCode.EXPIRED,
                "continuation",
                "continuation expired during cold load",
            ),
        ):
            result = await worker.process_once()

        self.assertIs(result.completion, terminal)
        self.assertIsNone(result.reentry)
        self.assertTrue(result.lease_lost)
        self.assertEqual(len(coordinator.expired_calls), 1)
        self.assertEqual(len(coordinator.failed_reentries), 0)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(target.resume_contexts, [])

    async def test_non_durable_reentry_fails_closed_without_rerun(
        self,
    ) -> None:
        target = SuspendingThenCompletingTarget()
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        suspending_worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        suspended = await suspending_worker.process_once()

        self.assertIsNotNone(suspended.suspension)
        self.assertIsNone(suspended.completion)
        self.assertIsNone(suspended.output)
        assert suspended.suspension is not None
        first_attempt = suspended.suspension.attempt
        first_segment = suspended.suspension.segment
        self.assertEqual(first_attempt.attempt_number, 1)
        self.assertEqual(
            first_segment.state,
            TaskAttemptSegmentState.SUSPENDED,
        )
        self.assertIsNone(suspended.suspension.run.claim)
        self.assertIsNone(suspended.suspension.queue_item.claim_token)
        self.assertEqual(
            await self.store.list_artifacts(self.run.run_id),
            (),
        )

        reentry = await self.queue.requeue_suspended(
            self.run.run_id,
            request_id="request-1",
            continuation_id="continuation-1",
            resolution_revision=2,
            now=self.now,
        )
        resuming_worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )
        failed = await resuming_worker.process_once()

        self.assertEqual(
            reentry.attempt.attempt_id,
            first_attempt.attempt_id,
        )
        self.assertIsNotNone(failed.completion)
        assert failed.completion is not None
        self.assertEqual(
            failed.completion.run.state,
            TaskRunState.FAILED,
        )
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(coordinator.failed_reentries), 1)
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].attempt_number, 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.FAILED)
        segments = await self.store.list_attempt_segments(
            attempts[0].attempt_id
        )
        self.assertEqual(segments, (first_segment,))
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.DEAD)
        self.assertEqual(self.queue.item.attempts, 0)
        usage = await self.store.list_usage(self.run.run_id)
        self.assertEqual(
            tuple(record.segment_id for record in usage),
            (first_segment.segment_id,),
        )

    async def test_process_once_revalidates_matching_skills_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            await self._use_definition(await _definition_with_skills(settings))
            target = FakeTarget("safe output")
            worker = TaskWorker(
                self.store,
                cast(object, self.queue),
                target=target,
                worker_id="worker-1",
                skills_settings=settings,
                clock=lambda: self.now,
            )

            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert target.contexts[0].definition.skills_identity is not None
        self.assertEqual(
            target.contexts[0].definition.skills_identity["status"],
            SkillStatus.OK.value,
        )

    async def test_process_once_fails_closed_on_skill_audit_delivery_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(
                root,
                observability=SkillObservabilitySettings(
                    audit_fail_closed=True
                ),
            )
            definition = await _definition_with_skills(settings)
            await self._use_definition(
                replace(definition, retry=TaskRetryPolicy(max_attempts=1))
            )
            target = FakeTarget("safe output")
            worker = TaskWorker(
                self.store,
                cast(object, self.queue),
                target=target,
                worker_id="worker-1",
                skills_settings=settings,
                clock=lambda: self.now,
            )

            with patch.object(
                self.store,
                "append_event",
                side_effect=RuntimeError("private audit store failure"),
            ):
                result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        error = self.queue.completed.run.result.error
        self.assertIsInstance(error, Mapping)
        self.assertNotIn("private audit store failure", str(error))

    async def test_process_once_fails_closed_on_missing_skills_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)

            code = await self._skills_failure_code(definition)

        self.assertEqual(code, "task.skills_registry_missing")

    async def test_process_once_fails_closed_on_stale_skills_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            skill_path = root / "pdf" / "SKILL.md"
            _write_skill(skill_path, body="# Body\nFIRST\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            _write_skill(skill_path, body="# Body\nSECOND\n")

            code = await self._skills_failure_code(
                definition,
                skills_settings=settings,
            )

        self.assertEqual(code, "task.skills_registry_stale")

    async def test_process_once_fails_closed_on_unavailable_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            missing_settings = _trusted_skills(Path(directory) / "missing")

            code = await self._skills_failure_code(
                definition,
                skills_settings=missing_settings,
            )

        self.assertEqual(code, "task.skills_registry_unavailable")

    async def test_process_once_fails_closed_on_widened_registry(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            restricted = _trusted_skills(
                root,
                read_limits=SkillReadLimits(max_lines_per_read=20),
            )
            widened = _trusted_skills(root)
            definition = await _definition_with_skills(restricted)

            code = await self._skills_failure_code(
                definition,
                skills_settings=widened,
            )

        self.assertEqual(code, "task.skills_registry_widened")

    async def test_process_once_fails_closed_on_malformed_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            registry = await _registry_with_status(
                settings,
                SkillStatus.MALFORMED,
            )

            code = await self._skills_failure_code(
                definition,
                skills_registry=registry,
            )

        self.assertEqual(code, "task.skills_registry_malformed")

    async def test_process_once_fails_closed_on_policy_denied_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            _write_skill(root / "pdf" / "SKILL.md", body="# Body\n")
            settings = _trusted_skills(root)
            definition = await _definition_with_skills(settings)
            registry = await _registry_with_status(
                settings,
                SkillStatus.POLICY_DENIED,
            )

            code = await self._skills_failure_code(
                definition,
                skills_registry=registry,
            )

        self.assertEqual(code, "task.skills_registry_policy_denied")

    async def test_process_once_uses_encrypted_execution_payload(self) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    input_value=sanitizer.sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        "private prompt",
                    ),
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        self.assertEqual(target.contexts[0].input_value, "private prompt")
        self.assertNotIn("private prompt", str(result.completion))

    async def test_process_once_uses_encrypted_file_payload(self) -> None:
        await self._use_definition(
            _definition(input_contract=TaskInputContract.file())
        )
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            mime_type="application/pdf",
            owner_scope="tenant-a",
            identity_hmac="hmac-value",
        )
        assert descriptor.provider_reference is not None
        entry = TaskExecutableInputFileEntry(
            file=TaskInputFile(
                logical_path="provider:openai:provider_file_id",
                provider_reference=descriptor.provider_reference,
                media_type="application/pdf",
            )
        )
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_payload=TaskExecutionPayload(
                    file_values=(
                        sanitizer.sanitize_with_action(
                            PrivacyAction.ENCRYPT,
                            task_execution_file_entries_value((entry,))[0],
                        ),
                    ),
                    input_value=None,
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        self.assertEqual(len(target.contexts[0].files), 1)
        file = target.contexts[0].files[0]
        self.assertIsNotNone(file.provider_reference)
        assert file.provider_reference is not None
        self.assertEqual(file.provider_reference.reference, "file-private")
        self.assertNotIn("file-private", str(result.completion))

    async def test_process_once_fails_without_file_payload_decryption(
        self,
    ) -> None:
        await self._use_definition(
            _definition(input_contract=TaskInputContract.file())
        )
        entry = TaskExecutableInputFileEntry(
            file=TaskInputFile(
                logical_path="artifact:input-1",
                artifact_ref=TaskArtifactRef(
                    artifact_id="input-1",
                    store="local",
                    storage_key="private/input-1",
                ),
            )
        )
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_payload=TaskExecutionPayload(
                    file_values=(
                        sanitizer.sanitize_with_action(
                            PrivacyAction.ENCRYPT,
                            task_execution_file_entries_value((entry,))[0],
                        ),
                    ),
                    input_value=None,
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private/input-1", str(self.queue.completed))

    async def test_process_once_fails_without_execution_payload(self) -> None:
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        rendered_result = str(self.queue.completed.run.result)
        self.assertIn("privacy.failure", rendered_result)
        self.assertNotIn("<redacted>", rendered_result)

    async def test_process_once_fails_when_payload_has_no_scalar_input(
        self,
    ) -> None:
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    file_values=(),
                    input_value=None,
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            encryption_provider=StaticEncryptionProvider(),
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("<redacted>", str(self.queue.completed.run.result))

    async def test_process_once_fails_without_decryption_provider(
        self,
    ) -> None:
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(raw_retention_days=1),
            encryption_provider=StaticEncryptionProvider(),
            raw_storage_allowed=True,
        )
        await self._use_request(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<redacted>"},
                input_payload=TaskExecutionPayload(
                    input_value=sanitizer.sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        "private prompt",
                    ),
                ),
                queue="default",
            )
        )
        target = FakeTarget("safe output")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(target.contexts, [])
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private prompt", str(self.queue.completed))

    async def test_process_once_retries_retryable_failures(self) -> None:
        target = FakeTarget()
        target.run = _raise_os_error
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.retry)
        assert result.retry is not None
        self.assertTrue(result.retry.retryable)
        self.assertEqual(result.retry.run.state, TaskRunState.QUEUED)
        self.assertEqual(result.retry.attempt.state, TaskAttemptState.FAILED)
        self.assertNotIn("private", str(result.retry.attempt.result))

    async def test_process_once_completes_terminal_failure(self) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
        target = FakeTarget()
        target.run = _raise_os_error
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertNotIn("private", str(self.queue.completed.run.result))

    async def test_process_once_completes_cancelled_failure(self) -> None:
        target = FakeTarget()
        target.run = _raise_cancelled
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(
            self.queue.completed.run.state,
            TaskRunState.CANCELLED,
        )

    async def test_process_once_records_output_artifacts(self) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.file(),
                artifact=TaskArtifactPolicy.references_only(retention_days=3),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ArtifactOutputTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        records = await self.store.list_artifacts(
            self.run.run_id,
            purpose=TaskArtifactPurpose.OUTPUT,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].state, TaskArtifactState.READY)
        self.assertEqual(records[0].retention.delete_after_days, 3)
        self.assertEqual(records[0].ref.metadata, {"privacy": "<redacted>"})
        self.assertNotIn("private-output", str(records))

    async def test_process_once_records_usage_observations(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=UsageTarget("safe output"),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.completion)
        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 5)

    async def test_process_once_records_returned_output_usage(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(
                UsageTextOutput(
                    "safe output",
                    usage={
                        "input_tokens": 7,
                        "cached_input_tokens": 2,
                        "output_tokens": 4,
                        "reasoning_tokens": 1,
                        "total_tokens": 11,
                        "provider_family": "openai",
                    },
                )
            ),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()
        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertIsNotNone(result.completion)
        self.assertEqual(result.output, "safe output")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 7)
        self.assertEqual(records[0].totals.cached_input_tokens, 2)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertEqual(records[0].totals.reasoning_tokens, 1)
        self.assertEqual(records[0].totals.total_tokens, 11)
        self.assertEqual(records[0].metadata, {"provider_family": "openai"})
        self.assertEqual(totals.total_tokens, 11)

    async def test_process_once_turns_output_validation_into_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.object(
                    schema={"type": "object"}
                ),
                retry=TaskRetryPolicy(max_attempts=1),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget("not an object"),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        assert self.queue.completed.run.result is not None
        self.assertEqual(
            self.queue.completed.run.result.error["code"],
            "output_contract.failed",
        )

    async def test_process_once_records_usage_before_output_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(
                output_contract=TaskOutputContract.object(
                    schema={"type": "object"}
                ),
                retry=TaskRetryPolicy(max_attempts=1),
            )
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(
                UsageTextOutput(
                    "private invalid output",
                    usage={
                        "input_tokens": 2,
                        "cache_creation_input_tokens": 1,
                        "output_tokens": 3,
                        "total_tokens": 5,
                        "provider_family": "openai",
                        "raw_response_id": "private-response-id",
                    },
                )
            ),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()
        records = await self.store.list_usage(self.run.run_id)

        self.assertTrue(result.processed)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.cache_creation_input_tokens, 1)
        self.assertEqual(records[0].totals.total_tokens, 5)
        self.assertNotIn("raw_response_id", records[0].metadata)
        self.assertNotIn(
            "private invalid output",
            str(self.queue.completed.run.result),
        )

    async def test_process_once_classifies_structured_output_failures(
        self,
    ) -> None:
        cases = (
            (
                TaskProviderStructuredOutputError(),
                "provider",
                "provider.structured_output_failed",
            ),
            (
                TaskOutputParseError(),
                "output_contract",
                "output.parse_failed",
            ),
        )

        for error, category, code in cases:
            with self.subTest(code=code):
                await self._use_definition(
                    _definition(retry=TaskRetryPolicy(max_attempts=1))
                )
                target = FakeTarget()

                async def fail(context: TaskTargetContext) -> object:
                    _ = context
                    raise error

                target.run = fail
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=target,
                    worker_id="worker-1",
                    clock=lambda: self.now,
                )

                result = await worker.process_once()

                self.assertTrue(result.processed)
                self.assertIsNotNone(self.queue.completed)
                assert self.queue.completed is not None
                self.assertEqual(
                    self.queue.completed.run.state,
                    TaskRunState.FAILED,
                )
                assert self.queue.completed.run.result is not None
                summary = cast(
                    Mapping[str, object],
                    self.queue.completed.run.result.error,
                )
                self.assertEqual(summary["category"], category)
                self.assertEqual(summary["code"], code)
                self.assertNotIn("private", str(summary))

    async def test_process_once_finalizes_invalid_target(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=InvalidTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNone(result.retry)
        self.assertIsNotNone(self.queue.completed)
        assert self.queue.completed is not None
        self.assertEqual(self.queue.completed.run.state, TaskRunState.FAILED)
        self.assertEqual(
            self.queue.completed.attempt.state,
            TaskAttemptState.FAILED,
        )
        assert self.queue.completed.run.result is not None
        self.assertEqual(
            self.queue.completed.run.result.error["code"],
            "runnable.failed",
        )

    async def test_process_once_skips_claim_after_shutdown_request(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        shutdown.request()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertFalse(result.processed)
        self.assertTrue(result.shutdown_requested)
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)

    async def test_process_once_stops_after_claim_expires_before_start(
        self,
    ) -> None:
        self.queue.abandon_after_claim = True
        target = FakeTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertEqual(target.contexts, [])
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.QUEUED,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.ABANDONED)

    async def test_process_once_sanitizes_start_claim_conflict(
        self,
    ) -> None:
        target = FakeTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with patch.object(
            worker,
            "_start_claimed_attempt",
            side_effect=TaskStoreConflictError("private stale start"),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertEqual(target.contexts, [])
        self.assertNotIn("private stale start", str(result))

    async def test_rejects_heartbeat_interval_not_shorter_than_lease(
        self,
    ) -> None:
        for heartbeat_seconds in (30, 30.0, 31):
            with (
                self.subTest(heartbeat_seconds=heartbeat_seconds),
                self.assertRaisesRegex(
                    AssertionError,
                    "heartbeat_seconds must be shorter than lease_seconds",
                ),
            ):
                TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=FakeTarget(),
                    worker_id="worker-1",
                    lease_seconds=30,
                    heartbeat_seconds=heartbeat_seconds,
                    clock=lambda: self.now,
                )

    async def test_process_once_abandons_active_shutdown_for_reclaim(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()

        class ShutdownTarget(FakeTarget):
            async def run(
                self,
                context: TaskTargetContext,
            ) -> TaskTargetOutcome:
                self.contexts.append(context)
                shutdown.request()
                await context.check_cancelled()
                return completed_task_target_outcome("unreachable")

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertIsNone(self.queue.completed)
        self.assertTrue(result.abandonment.retryable)
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)

    async def test_process_once_shutdown_abandon_can_exhaust_attempts(
        self,
    ) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
        shutdown = TaskWorkerShutdown()

        class ShutdownTarget(FakeTarget):
            async def run(
                self,
                context: TaskTargetContext,
            ) -> TaskTargetOutcome:
                self.contexts.append(context)
                shutdown.request()
                await context.check_cancelled()
                return completed_task_target_outcome("unreachable")

        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertFalse(result.abandonment.retryable)
        self.assertEqual(result.abandonment.run.state, TaskRunState.FAILED)
        self.assertEqual(
            result.abandonment.queue_item.state,
            TaskQueueItemState.DEAD,
        )

    async def test_process_once_abandons_shutdown_after_target_return(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        target = ShutdownReturningTarget(shutdown)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertIsNone(result.completion)
        self.assertIsNone(self.queue.completed)
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertIsNone(result.abandonment.run.result)
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )
        assert self.queue.item is not None
        self.assertEqual(self.queue.item.state, TaskQueueItemState.AVAILABLE)
        self.assertEqual(len(target.contexts), 1)

    async def test_process_once_stops_heartbeat_on_shutdown(self) -> None:
        shutdown = TaskWorkerShutdown()
        self.queue.heartbeat_shutdown = shutdown
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertEqual(len(self.queue.heartbeats), 1)
        self.assertIsNotNone(result.abandonment)
        assert result.abandonment is not None
        self.assertEqual(
            result.abandonment.run.state,
            TaskRunState.QUEUED,
        )
        self.assertEqual(
            result.abandonment.attempt.state,
            TaskAttemptState.ABANDONED,
        )

    async def test_process_once_stops_after_heartbeat_failure(
        self,
    ) -> None:
        await self._use_definition(
            _definition(retry=TaskRetryPolicy(max_attempts=1))
        )
        self.queue.heartbeat_error = RuntimeError("private heartbeat")
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertIsNone(self.queue.completed)
        self.assertIsNone(self.queue.retried)
        self.assertIsNone(self.queue.abandoned)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.RUNNING,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.RUNNING)
        self.assertNotIn("private heartbeat", str(result))

    async def test_process_once_stops_after_heartbeat_claim_conflict(
        self,
    ) -> None:
        self.queue.heartbeat_error = TaskQueueConflictError(
            "private stale claim"
        )
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertIsNone(result.retry)
        self.assertIsNone(result.abandonment)
        self.assertIsNone(self.queue.completed)
        self.assertIsNone(self.queue.retried)
        self.assertIsNone(self.queue.abandoned)
        self.assertEqual(
            (await self.store.get_run(self.run.run_id)).state,
            TaskRunState.RUNNING,
        )
        attempts = await self.store.list_attempts(self.run.run_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].state, TaskAttemptState.RUNNING)
        self.assertNotIn("private stale claim", str(result))

    async def test_process_once_stops_after_shutdown_finalize_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(
                worker,
                "_execute",
                side_effect=_TaskWorkerShutdownRequested(),
            ),
            patch.object(
                worker,
                "_finalize_shutdown",
                side_effect=TaskQueueConflictError("private shutdown claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.shutdown_requested)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.abandonment)
        self.assertNotIn("private shutdown claim", str(result))

    async def test_process_once_stops_after_failure_finalize_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(
                worker,
                "_execute",
                side_effect=RuntimeError("private target failure"),
            ),
            patch.object(
                worker,
                "_finalize_failure",
                side_effect=TaskQueueConflictError("private retry claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.retry)
        self.assertNotIn("private retry claim", str(result))

    async def test_process_once_stops_after_success_complete_conflict(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        with (
            patch.object(
                worker,
                "_execute",
                return_value=completed_task_target_outcome("safe output"),
            ),
            patch.object(
                worker,
                "_complete_success",
                side_effect=TaskQueueConflictError("private complete claim"),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.processed)
        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.completion)
        self.assertNotIn("private complete claim", str(result))

    async def test_run_target_timeout_cancels_target_and_shutdown_watcher(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=WaitingTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with self.assertRaises(TimeoutError):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=0.001,
            )

        self.assertFalse(shutdown.requested)

    async def test_run_target_timeout_fences_durable_dispatch(self) -> None:
        admission = FakeDurableResumeHandle(
            dispatch_started=Event(),
            dispatch_proceed=Event(),
        )
        target = DurableResumableTarget(self.now)
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        claim = await self._claim()
        context = replace(
            self._target_context(claim),
            durable_resume=admission,
        )

        with self.assertRaises(TimeoutError):
            await worker._run_target(
                context,
                claim=claim,
                timeout=0.001,
                durable_target=target,
            )

        self.assertEqual(admission.dispatch_calls, 1)
        self.assertEqual(admission.interrupt_calls, 1)
        self.assertEqual(
            admission.state,
            DurableContinuationResumeState.AMBIGUOUS,
        )

    async def test_run_target_shutdown_cancels_running_target(self) -> None:
        shutdown = TaskWorkerShutdown()
        target = PassiveWaitingTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()
        running = create_task(
            worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )
        )

        await target.started.wait()
        shutdown.request()

        with self.assertRaises(_TaskWorkerShutdownRequested):
            await running

    async def test_run_target_shutdown_wins_before_waiter_completes(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=ShutdownReturningTarget(shutdown),
            worker_id="worker-1",
            shutdown=shutdown,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with (
            patch("avalan.task.worker.wait", new=_target_only_wait),
            self.assertRaises(_TaskWorkerShutdownRequested),
        ):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )

    async def test_run_target_heartbeat_shutdown_cancels_running_target(
        self,
    ) -> None:
        shutdown = DelayedWaitShutdown()
        shutdown.request()
        target = PassiveWaitingTarget()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=target,
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        with self.assertRaises(_TaskWorkerShutdownRequested):
            await worker._run_target(
                self._target_context(claim),
                claim=claim,
                timeout=1,
            )

    async def test_run_target_heartbeat_failure_is_sanitized(
        self,
    ) -> None:
        cases = (
            TaskQueueConflictError("private stale claim"),
            RuntimeError("private heartbeat outage"),
        )
        for heartbeat_error in cases:
            with self.subTest(error=type(heartbeat_error).__name__):
                await self._use_definition(_definition())
                self.queue.heartbeat_error = heartbeat_error
                worker = TaskWorker(
                    self.store,
                    cast(object, self.queue),
                    target=PassiveWaitingTarget(),
                    worker_id="worker-1",
                    heartbeat_seconds=0.001,
                    clock=lambda: self.now,
                )
                claim = await self._claim()

                with self.assertRaises(TaskQueueConflictError) as error:
                    await worker._run_target(
                        self._target_context(claim),
                        claim=claim,
                        timeout=1,
                    )

                self.assertEqual(
                    str(error.exception),
                    "task queue heartbeat failed",
                )

    async def test_run_target_returns_with_heartbeat_enabled(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget("safe output"),
            worker_id="worker-1",
            heartbeat_seconds=30,
            clock=lambda: self.now,
        )
        claim = await self._claim()

        output = await worker._run_target(
            self._target_context(claim),
            claim=claim,
            timeout=1,
        )

        self.assertEqual(
            output,
            completed_task_target_outcome("safe output"),
        )
        self.assertEqual(self.queue.heartbeats, [])

    async def test_heartbeat_claim_returns_after_shutdown_request(
        self,
    ) -> None:
        shutdown = TaskWorkerShutdown()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            shutdown=shutdown,
            heartbeat_seconds=0.001,
            clock=lambda: self.now,
        )
        claim = await self._claim()
        shutdown.request()

        await worker._heartbeat_claim(claim)

        self.assertEqual(self.queue.heartbeats, [])

    async def test_check_cancelled_raises_cancelled_error(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        await self.store.transition_run(
            self.run.run_id,
            from_states={TaskRunState.QUEUED},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel requested",
        )

        with self.assertRaises(CancelledError):
            await worker._check_cancelled(self.run.run_id)

    async def test_helper_branches(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=_callable_target,
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        self.assertIsNone(
            worker._event_pipeline(
                _definition(observability=TaskObservabilityPolicy.noop()),
                run=self.run,
                attempt=await self.store.create_attempt(self.run.run_id),
                sanitizer=PrivacySanitizer(TaskPrivacyPolicy()),
            )
        )
        self.assertIsNone(
            worker._observability_sink_for(
                _definition(
                    observability=TaskObservabilityPolicy(
                        sinks=(ObservabilitySinkType.NOOP,),
                        metrics=False,
                        trace=False,
                        capture_events=False,
                    )
                )
            )
        )
        summary = worker._safe_task_error_summary(
            PrivacySanitizer(
                TaskPrivacyPolicy(errors=PrivacyAction.HASH),
            ),
            classify_task_error(RuntimeError("private")),
        )
        self.assertEqual(summary["privacy"], "<redacted>")
        self.assertIsNotNone(_target_runner(_callable_target))
        self.assertIsInstance(_utc_now(), datetime)
        self.assertTrue(_worker_id().startswith("worker-"))

    async def test_record_usage_handles_missing_and_failed_store_records(
        self,
    ) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        attempt = await self.store.create_attempt(self.run.run_id)

        await worker._record_usage(
            object(),
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )
        with patch.object(
            self.store,
            "append_usage",
            side_effect=RuntimeError("private telemetry failure"),
        ):
            await worker._record_usage(
                SimpleNamespace(input_token_count=1, output_token_count=2),
                definition=self.definition,
                run=self.run,
                attempt=attempt,
            )

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(records, ())

    async def test_record_usage_persists_each_provider_call(self) -> None:
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )
        attempt = await self.store.create_attempt(self.run.run_id)
        response = MultiCallUsageResponse(
            SimpleNamespace(
                usage={
                    "input_tokens": 2,
                    "cached_input_tokens": 1,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "provider_family": "openai",
                }
            ),
            SimpleNamespace(
                usage={
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 6,
                    "reasoning_tokens": 1,
                    "total_tokens": 10,
                    "provider_family": "openai",
                    "raw_response_id": "private-response-id",
                }
            ),
        )

        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(
            [record.source for record in records],
            [UsageSource.EXACT, UsageSource.EXACT],
        )
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 1)
        self.assertNotIn("raw_response_id", records[1].metadata)
        self.assertEqual(totals.input_tokens, 6)
        self.assertEqual(totals.output_tokens, 9)
        self.assertEqual(totals.total_tokens, 15)

    async def test_record_usage_deduplicates_reobserved_response(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )
        attempt = await self.store.create_attempt(self.run.run_id)
        response = SimpleNamespace(input_token_count=2, output_token_count=3)

        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )
        await worker._record_usage(
            response,
            definition=self.definition,
            run=self.run,
            attempt=attempt,
        )

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.output_tokens, 3)
        self.assertEqual(totals.input_tokens, 2)
        self.assertEqual(totals.output_tokens, 3)
        self.assertEqual(len(sink.usage_totals), 1)
        self.assertEqual(sink.usage_event_count, 1)

    async def test_process_once_records_unobserved_returned_wrapper_calls(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=PartiallyObservedUsageWrapperTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )

        result = await worker.process_once()

        records = await self.store.list_usage(self.run.run_id)
        totals = await self.store.usage_totals(self.run.run_id)
        self.assertTrue(result.processed)
        self.assertIsNotNone(result.completion)
        assert result.completion is not None
        self.assertEqual(result.completion.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(records[0].totals.input_tokens, 4)
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.input_tokens, 5)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 3)
        self.assertEqual(totals.input_tokens, 9)
        self.assertEqual(totals.output_tokens, 13)
        self.assertEqual(totals.total_tokens, 22)
        self.assertEqual(len(sink.usage_totals), 2)
        self.assertEqual(sink.usage_event_count, 2)
        self.assertNotIn("private-first-response", str(records))
        self.assertNotIn("private-second-response", str(records))
        self.assertNotIn("private provider body", str(records))

    async def test_record_usage_conflict_does_not_emit_sink_usage(
        self,
    ) -> None:
        sink = RecordingUsageSink()
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
            observability_sink=sink,
        )
        attempt = await self.store.create_attempt(self.run.run_id)

        with patch.object(
            self.store,
            "append_usage",
            side_effect=TaskStoreConflictError("private usage conflict"),
        ):
            await worker._record_usage(
                SimpleNamespace(input_token_count=2, output_token_count=3),
                definition=self.definition,
                run=self.run,
                attempt=attempt,
            )

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(records, ())
        self.assertEqual(sink.usage_totals, [])
        self.assertEqual(sink.usage_event_count, 0)

    async def test_process_once_reports_no_work(self) -> None:
        self.queue.item = None
        worker = TaskWorker(
            self.store,
            cast(object, self.queue),
            target=FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

        result = await worker.process_once()

        self.assertFalse(result.processed)
        self.assertIsNone(result.completion)


def _definition(
    *,
    artifact: TaskArtifactPolicy | None = None,
    input_contract: TaskInputContract | None = None,
    observability: TaskObservabilityPolicy | None = None,
    output_contract: TaskOutputContract | None = None,
    retry: TaskRetryPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="worker_task", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        artifact=artifact or TaskArtifactPolicy.references_only(),
        observability=observability or TaskObservabilityPolicy(),
        run=TaskRunPolicy.queued("default"),
        retry=retry or TaskRetryPolicy(max_attempts=2),
    )


async def _definition_with_skills(
    settings: TrustedSkillSettings,
) -> TaskDefinition:
    return await task_definition_with_skills_identity(
        replace(
            _definition(),
            execution=TaskExecutionTarget.tool("skills"),
            skills=settings,
        ),
    )


def _trusted_skills(
    root: Path,
    *,
    observability: SkillObservabilitySettings | None = None,
    read_limits: SkillReadLimits | None = None,
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label="workspace-main",
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        observability=(
            observability
            if observability is not None
            else SkillObservabilitySettings()
        ),
        read_limits=read_limits or SkillReadLimits(),
    )


async def _registry_with_status(
    settings: TrustedSkillSettings,
    status: SkillStatus,
) -> SkillRegistry:
    assert status in {SkillStatus.MALFORMED, SkillStatus.POLICY_DENIED}
    registry = await build_task_skill_registry(settings)
    diagnostic = SkillDiagnosticInfo(
        code=(
            SkillDiagnosticCode.MANIFEST_MALFORMED
            if status is SkillStatus.MALFORMED
            else SkillDiagnosticCode.POLICY_DENIED
        ),
        status=status,
        message="Registry is not usable.",
        path="skills",
        hint="Use an operator-approved registry.",
    )
    return replace(
        registry,
        diagnostics=(diagnostic,),
    )


def _write_skill(path: Path, *, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "name: pdf\n"
        "description: PDF rendering guidance.\n"
        'tags: ["pdf"]\n'
        "resources: []\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


async def _raise_os_error(context: TaskTargetContext) -> object:
    raise OSError("private backend path")


async def _raise_cancelled(context: TaskTargetContext) -> object:
    raise CancelledError()


async def _target_only_wait(
    tasks: set[AsyncTask[object]],
    *,
    timeout: float | None,
    return_when: object,
) -> tuple[set[AsyncTask[object]], set[AsyncTask[object]]]:
    _ = timeout, return_when
    for _attempt in range(3):
        for task in tasks:
            if task.done() and task.result() == completed_task_target_outcome(
                "safe output"
            ):
                return {task}, tasks - {task}
        await sleep(0)
    raise AssertionError("target task did not finish")


async def _callable_target(context: TaskTargetContext) -> object:
    return "safe output"


if __name__ == "__main__":
    main()
