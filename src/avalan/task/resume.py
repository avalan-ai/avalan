"""Bind requeued task provenance to one durable agent continuation."""

from ..agent.continuation import (
    DurableAgentContinuationAdmission,
    DurableAgentContinuationClaimLease,
    DurableAgentContinuationResumer,
)
from ..interaction.continuation import (
    ContinuationClaimOwnerId,
    ContinuationCompletionCommand,
    ContinuationRejectionCommand,
    DurableContinuationRecord,
    DurableContinuationResumeState,
    derive_continuation_dispatch_id,
)
from ..interaction.entities import InputRequestId, RunId
from ..interaction.error import InputErrorCode, InputValidationError
from ..interaction.policy import InteractionActor
from ..interaction.validation import validate_opaque_id
from ..types import JsonValue
from .context import TaskEventListener, TaskEventListenerRegistration
from .queue import TaskQueueClaim
from .settlement import (
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
    task_durable_resume_settlement_digest,
)
from .state import TaskAttemptSegmentState, TaskAttemptState
from .store import (
    TaskAttemptSegment,
    freeze_snapshot_value,
)

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from inspect import iscoroutinefunction
from json import dumps
from typing import NoReturn, Protocol, cast, final


class TaskContinuationRecordStore(Protocol):
    """Load the unique active continuation bound to one logical task run."""

    async def get_task_continuation_record(
        self,
        task_run_id: str,
    ) -> DurableContinuationRecord:
        """Return the unique task-bound continuation or fail closed."""
        ...


class TaskResumeActorResolver(Protocol):
    """Resolve trusted interaction authority for one claimed task."""

    trusted_task_resume_actor_resolver: bool

    async def resolve_task_resume_actor(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment,
        record: DurableContinuationRecord,
    ) -> InteractionActor:
        """Return the exact actor authorized to resume this task."""
        ...


class TaskResumeClaimLeaseManager(Protocol):
    """Bridge successful task heartbeats into one continuation claim."""

    async def current_lease_expires_at(self) -> datetime:
        """Return the latest successfully acquired task claim lease."""
        ...

    async def bind(
        self,
        claim_lease: DurableAgentContinuationClaimLease,
    ) -> None:
        """Bind the exact continuation claim before cold reconstruction."""
        ...

    async def unbind(self) -> None:
        """Stop renewing the bound continuation claim."""
        ...


@final
class StoredTaskResumeActorResolver:
    """Use the encrypted continuation principal inside a trusted worker."""

    trusted_task_resume_actor_resolver = True

    async def resolve_task_resume_actor(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment,
        record: DurableContinuationRecord,
    ) -> InteractionActor:
        """Return the continuation owner after exact task correlation."""
        del claim, previous_segment
        return InteractionActor(principal=record.continuation.origin.principal)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDurableResumeAdmission:
    """Carry one task-bound claimed continuation into its target runner."""

    record: DurableContinuationRecord
    previous_segment: TaskAttemptSegment
    agent_admission: DurableAgentContinuationAdmission

    def __post_init__(self) -> None:
        if type(self.record) is not DurableContinuationRecord:
            _invalid_type(
                "task_resume.record",
                "a durable continuation record",
            )
        if not isinstance(self.previous_segment, TaskAttemptSegment):
            _invalid_type(
                "task_resume.previous_segment",
                "a task attempt segment",
            )
        if (
            self.previous_segment.state
            is not TaskAttemptSegmentState.SUSPENDED
        ):
            _illegal_transition(
                "task_resume.previous_segment.state",
                "resume provenance must be a suspended segment",
            )
        if type(self.agent_admission) is not DurableAgentContinuationAdmission:
            _invalid_type(
                "task_resume.agent_admission",
                "a durable agent continuation admission",
            )
        continuation = self.record.continuation
        claimed = self.agent_admission.continuation
        if (
            self.previous_segment.request_id != str(continuation.request_id)
            or self.previous_segment.continuation_id
            != str(continuation.continuation_id)
            or self.previous_segment.checkpoint_id != self.record.checkpoint_id
            or claimed.request_id != continuation.request_id
            or claimed.continuation_id != continuation.continuation_id
            or claimed.origin != continuation.origin
        ):
            _correlation_error(
                "task_resume",
                "task suspension does not match the claimed continuation",
            )

    async def dispatch(self) -> object:
        """Resume the fresh runtime without invoking initial task input."""
        return await self.agent_admission.dispatch()

    def register_event_listener(
        self,
        listener: TaskEventListener,
    ) -> TaskEventListenerRegistration:
        """Register one task listener before resumed provider dispatch."""
        return self.agent_admission.register_event_listener(listener)

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        """Wait for owned provider dispatch to settle durably."""
        return await self.agent_admission.wait_dispatch_settled()

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        """Stop owned work at a durable pre- or post-dispatch boundary."""
        return await self.agent_admission.interrupt_dispatch()

    async def complete_output(self, output: object) -> None:
        """Complete continuation metadata after durable task success."""
        await self.agent_admission.complete(task_resume_result_digest(output))

    def completion_command_for_output(
        self,
        output: object,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for atomic task settlement."""
        return self.agent_admission.completion_command(
            task_resume_result_digest(output)
        )

    def completion_command_for_settlement(
        self,
        settlement: TaskDurableResumeSettlement,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for atomic terminal task settlement."""
        return self.agent_admission.completion_command(
            task_durable_resume_settlement_digest(settlement)
        )

    def completed_completion_command(
        self,
    ) -> ContinuationCompletionCommand:
        """Return the provider completion fence without changing its digest."""
        continuation = self.agent_admission.continuation
        if (
            self.agent_admission.state
            is not DurableContinuationResumeState.COMPLETED
            or continuation.completion is None
        ):
            _illegal_transition(
                "task_resume.completed_completion",
                "continuation is not durably completed",
            )
        return self.agent_admission.completion_command(
            continuation.completion.result_digest
        )

    def completion_command_for_suspension(
        self,
        *,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for one atomic successor suspension."""
        request = validate_opaque_id(
            request_id,
            "task_resume.successor.request_id",
        )
        continuation = validate_opaque_id(
            continuation_id,
            "task_resume.successor.continuation_id",
        )
        checkpoint = validate_opaque_id(
            checkpoint_id,
            "task_resume.successor.checkpoint_id",
        )
        return self.agent_admission.completion_command(
            task_resume_result_digest(
                {
                    "kind": "suspended",
                    "request_id": request,
                    "continuation_id": continuation,
                    "checkpoint_id": checkpoint,
                }
            )
        )

    def rejection_command_for_settlement(
        self,
        failure: TaskDurableResumeFailure,
    ) -> ContinuationRejectionCommand:
        """Return a fence for deterministic pre-dispatch task failure."""
        if type(failure) is not TaskDurableResumeFailure:
            _invalid_type(
                "task_resume.failure",
                "a durable resume failure",
            )
        return self.agent_admission.rejection_command(
            task_durable_resume_settlement_digest(failure)
        )

    async def release(self) -> None:
        """Release the claim when the target cannot start dispatch."""
        await self.agent_admission.release()

    async def release_if_pre_dispatch(self) -> bool:
        """Release only when provider dispatch has not taken ownership."""
        return await self.agent_admission.release_if_pre_dispatch()

    async def close(self) -> None:
        """Close resources owned by this task resume admission."""
        await self.agent_admission.close()


@final
class TaskDurableResumeCoordinator:
    """Admit only exactly correlated suspended task reentries."""

    def __init__(
        self,
        record_store: TaskContinuationRecordStore,
        agent_resumer: DurableAgentContinuationResumer,
        *,
        actor_resolver: TaskResumeActorResolver | None = None,
    ) -> None:
        if not iscoroutinefunction(
            getattr(record_store, "get_task_continuation_record", None)
        ):
            _invalid_type(
                "task_resume.record_store",
                "an asynchronous continuation record store",
            )
        if type(agent_resumer) is not DurableAgentContinuationResumer:
            _invalid_type(
                "task_resume.agent_resumer",
                "a durable agent continuation resumer",
            )
        resolved_actor = actor_resolver or StoredTaskResumeActorResolver()
        if getattr(
            resolved_actor,
            "trusted_task_resume_actor_resolver",
            False,
        ) is not True or not iscoroutinefunction(
            getattr(
                resolved_actor,
                "resolve_task_resume_actor",
                None,
            )
        ):
            _invalid_type(
                "task_resume.actor_resolver",
                "a trusted asynchronous task resume actor resolver",
            )
        self._record_store = record_store
        self._agent_resumer = agent_resumer
        self._actor_resolver = resolved_actor

    async def admit(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment | None,
        *,
        claim_lease_manager: TaskResumeClaimLeaseManager | None = None,
    ) -> TaskDurableResumeAdmission | None:
        """Return no admission for fresh work and fence exact reentry."""
        if not isinstance(claim, TaskQueueClaim):
            _invalid_type("task_resume.claim", "a task queue claim")
        if claim.attempt.state is TaskAttemptState.CREATED:
            if previous_segment is not None:
                _correlation_error(
                    "task_resume.previous_segment",
                    "fresh task work cannot carry resume provenance",
                )
            return None
        if claim.attempt.state is not TaskAttemptState.SUSPENDED:
            _illegal_transition(
                "task_resume.claim.attempt.state",
                "only a suspended attempt can resume",
            )
        if not isinstance(previous_segment, TaskAttemptSegment):
            _invalid_type(
                "task_resume.previous_segment",
                "a suspended task attempt segment",
            )
        _validate_previous_segment(claim, previous_segment)
        record = await self._record_store.get_task_continuation_record(
            claim.run.run_id
        )
        _validate_record(claim, previous_segment, record)
        actor = await self._actor_resolver.resolve_task_resume_actor(
            claim,
            previous_segment,
            record,
        )
        if not isinstance(actor, InteractionActor):
            _invalid_type(
                "task_resume.actor",
                "an interaction actor",
            )
        task_claim = claim.run.claim
        assert task_claim is not None
        continuation = record.continuation
        admission = await self._agent_resumer.admit(
            record,
            actor=actor,
            expected_request_id=InputRequestId(
                cast(str, previous_segment.request_id)
            ),
            expected_run_id=RunId(claim.run.run_id),
            expected_checkpoint_id=cast(
                str,
                previous_segment.checkpoint_id,
            ),
            owner_id=ContinuationClaimOwnerId(task_claim.claim_token),
            lease_expires_at=task_claim.lease_expires_at,
            dispatch_id=derive_continuation_dispatch_id(
                continuation.continuation_id
            ),
            lease_expires_at_provider=(
                claim_lease_manager.current_lease_expires_at
                if claim_lease_manager is not None
                else None
            ),
            claim_lease_observer=(
                claim_lease_manager.bind
                if claim_lease_manager is not None
                else None
            ),
        )
        return TaskDurableResumeAdmission(
            record=record,
            previous_segment=previous_segment,
            agent_admission=admission,
        )


def task_resume_result_digest(output: object) -> str:
    """Return a canonical digest for one privacy-safe task result."""
    frozen = freeze_snapshot_value(output)
    encoded = dumps(
        _thaw_snapshot_value(frozen),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _validate_previous_segment(
    claim: TaskQueueClaim,
    segment: TaskAttemptSegment,
) -> None:
    if (
        segment.state is not TaskAttemptSegmentState.SUSPENDED
        or segment.run_id != claim.run.run_id
        or segment.attempt_id != claim.attempt.attempt_id
        or segment.request_id is None
        or segment.continuation_id is None
        or segment.checkpoint_id is None
    ):
        _correlation_error(
            "task_resume.previous_segment",
            "task claim lacks exact suspended continuation provenance",
        )


def _validate_record(
    claim: TaskQueueClaim,
    segment: TaskAttemptSegment,
    record: DurableContinuationRecord,
) -> None:
    if type(record) is not DurableContinuationRecord:
        _invalid_type(
            "task_resume.record",
            "a durable continuation record",
        )
    continuation = record.continuation
    if (
        record.task_run_id != claim.run.run_id
        or record.checkpoint_id != segment.checkpoint_id
        or str(continuation.request_id) != segment.request_id
        or str(continuation.continuation_id) != segment.continuation_id
        or str(continuation.origin.run_id) != claim.run.run_id
    ):
        _correlation_error(
            "task_resume.record",
            "durable record does not match the requeued task",
        )


def _thaw_snapshot_value(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_snapshot_value(item) for item in value]
    return value


def _invalid_type(path: str, expected: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        path,
        f"value must be {expected}",
    )


def _correlation_error(path: str, message: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.CORRELATION_MISMATCH,
        path,
        message,
    )


def _illegal_transition(path: str, message: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.ILLEGAL_TRANSITION,
        path,
        message,
    )
