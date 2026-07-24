"""Test durable PostgreSQL interaction and task transaction boundaries."""

from asyncio import (
    CancelledError,
    Event,
    create_task,
    gather,
    get_running_loop,
    wait_for,
)
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import path as sys_path
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

sys_path.append(str(Path(__file__).parent))

from pgsql_support import (  # noqa: E402
    FakePgsqlDatabase,
    FullFakePgsqlDatabase,
)

from avalan.agent.continuation import DurableAgentContinuationResumer
from avalan.interaction import (
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CapabilityRevision,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationClaimOwnerId,
    ContinuationClaimState,
    ContinuationCompletionCommand,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationRejectionCommand,
    ContinuationRevisionBinding,
    ContinuationRuntimeResolver,
    ContinuationSnapshot,
    ContinuationStoreRevision,
    ControllerId,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputContractError,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchRecord,
    InteractionBranchRegistration,
    InteractionBranchRootLookup,
    InteractionClock,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionNotFoundError,
    InteractionOperation,
    InteractionPolicy,
    InteractionRecord,
    InteractionStoreClosedError,
    InteractionStoreRevision,
    InteractionTime,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    ParticipantId,
    PortableContinuation,
    PresentInteractionCommand,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    QuestionId,
    RecordControllerActivityCommand,
    RegisterInteractionBranchCommand,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolvedContinuationRuntime,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    RunId,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    ScopeSupersessionRejected,
    ScopeSupersessionReplayed,
    SessionId,
    StateRevision,
    StreamSessionId,
    TaskId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TenantId,
    TerminateInputContinuation,
    TurnId,
    UserId,
    WaitForDeadlineChangeCommand,
    WaitForInteractionChangeCommand,
    create_input_request,
)
from avalan.interaction.continuation import (
    derive_continuation_dispatch_id,
    derive_provider_idempotency_key,
)
from avalan.interaction.store import (
    CancelInteractionCommand,
    CreateInteractionApplied,
    CreateInteractionCommand,
    DueInteractionsApplied,
    InteractionBranchRegistrationApplied,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
    TerminalizeInteractionScopeCommand,
    _InteractionAdmissionCleanupDisposition,
    _new_interaction_admission_commands,
)
from avalan.interaction.stores import pgsql as interaction_pgsql
from avalan.interaction.stores.pgsql import (
    _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
    _SELECT_BRANCHES_SQL,
    _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
    _SELECT_CONTINUATION_BY_REQUEST_SQL,
    _SELECT_RECORDS_SQL,
    _SELECT_SCOPE_BRANCHES_SQL,
    _SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL,
    _SELECT_SCOPE_RECORDS_SQL,
    _SELECT_STORE_METADATA_SQL,
    _SELECT_TASK_BRANCH_CLOSURE_SQL,
    _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL,
    _SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL,
    _SET_REPEATABLE_READ_ONLY_SQL,
    _UPSERT_BRANCH_SQL,
    _UPSERT_RECORD_SQL,
    ContinuationDispatchAmbiguousError,
    ContinuationStoreConflictError,
    DurableContinuationLifecycle,
    PgsqlDurableTaskCoordinator,
    PgsqlDurableTaskLifecycle,
    PgsqlInteractionFeatureUnavailableError,
    PgsqlInteractionStore,
    PgsqlInteractionStoreError,
    PgsqlInteractionStoreFactory,
    PgsqlInteractionStorePolicy,
    PgsqlResumptionReconciler,
    ResumptionOutboxRecord,
    _encode_branch,
    _scope_identity_digest,
    require_interaction_pgsql_dependencies,
)
from avalan.pgsql import PgsqlDatabase, PgsqlOperationError, PgsqlUnitOfWork
from avalan.task import (
    EncryptedPrivacyValue,
    TaskAttemptState,
    TaskClaim,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskDefinition,
    TaskError,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInteractionEventType,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueItemState,
    TaskRunPolicy,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreError,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskTargetRunnerRegistry,
    TaskTargetType,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
    completed_task_target_outcome,
    freeze_snapshot_value,
)
from avalan.task.context import TaskDurableResumeHandle
from avalan.task.resume import TaskDurableResumeCoordinator
from avalan.task.settlement import (
    TaskDurableResumeCancellation,
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
    TaskDurableResumeSuccess,
    task_durable_resume_settlement_digest,
)
from avalan.task.state import TaskAttemptSegmentState
from avalan.task.stores import PgsqlTaskStore
from avalan.task.stores import pgsql as task_pgsql

_NOW = datetime(2026, 7, 23, 12, tzinfo=UTC)


async def _unused_task_target(context: TaskTargetContext) -> object:
    _ = context
    return {}


class _Clock(InteractionClock):
    """Return explicitly controlled coherent interaction time."""

    def __init__(self) -> None:
        self.now = _NOW
        self.monotonic = 0.0
        self.changed = Event()

    async def read(self) -> InteractionTime:
        return InteractionTime.from_clock(
            wall_time=self.now,
            monotonic_seconds=self.monotonic,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        while self.monotonic < monotonic_deadline:
            await self.changed.wait()
            self.changed.clear()


class _DeadlineFailureLoader:
    """Cross the absolute deadline during trusted runtime reconstruction."""

    trusted_continuation_runtime_loader = True

    def __init__(self, current: list[datetime], deadline: datetime) -> None:
        self.current = current
        self.deadline = deadline
        self.calls = 0

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        del definition, revision_binding
        self.calls += 1
        self.current[0] = self.deadline
        raise InputValidationError(
            InputErrorCode.EXPIRED,
            "continuation_runtime",
            "continuation expired during cold reconstruction",
        )


class _BlockingDeadlineLoader:
    """Remain in cold reconstruction until the worker cancels the loader."""

    trusted_continuation_runtime_loader = True

    def __init__(self, current: list[datetime], deadline: datetime) -> None:
        self.current = current
        self.deadline = deadline
        self.calls = 0
        self.started = Event()
        self.cancelled = Event()
        self.proceed = Event()

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        del definition, revision_binding
        self.calls += 1
        self.started.set()
        try:
            await self.proceed.wait()
        except CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError(
            "deadline loader was released instead of cancelled"
        )


class _PersistedClaimQueue:
    """Expose one already-claimed fake-PostgreSQL queue row to a worker."""

    def __init__(
        self,
        database: FullFakePgsqlDatabase,
        task_store: PgsqlTaskStore,
        *,
        allow_heartbeats: bool = False,
    ) -> None:
        self.database = database
        self.task_store = task_store
        self.allow_heartbeats = allow_heartbeats
        self.claim_calls = 0
        self.heartbeat_calls = 0

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueClaim | None:
        del queue_name, worker_id, lease_expires_at, now, metadata
        self.claim_calls += 1
        queue_row = self.database.queue_items["queue-item"]
        if queue_row["state"] != TaskQueueItemState.CLAIMED.value:
            return None
        run = await self.task_store.get_run("run")
        assert run.last_attempt_id is not None
        attempt = await self.task_store.get_attempt(run.last_attempt_id)
        return TaskQueueClaim(
            queue_item=task_pgsql._queue_item_from_row(  # noqa: SLF001
                queue_row,
                run_state=run.state,
            ),
            run=run,
            attempt=attempt,
        )

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime,
    ) -> object:
        del lease_expires_at, now
        if not self.allow_heartbeats:
            raise AssertionError(
                "deadline setup must finish before a heartbeat"
            )
        self.heartbeat_calls += 1
        queue_row = self.database.queue_items[queue_item_id]
        assert queue_row["claim_token"] == claim_token
        run = await self.task_store.get_run("run")
        return task_pgsql._queue_item_from_row(  # noqa: SLF001
            queue_row,
            run_state=run.state,
        )


@dataclass(slots=True)
class _DeadlineWorkerHarness:
    database: FullFakePgsqlDatabase
    store: PgsqlInteractionStore
    task_store: PgsqlTaskStore
    task_coordinator: PgsqlDurableTaskCoordinator
    request: InputRequest
    claim: TaskClaim
    loader: _DeadlineFailureLoader | _BlockingDeadlineLoader
    queue: _PersistedClaimQueue
    worker: TaskWorker
    target_calls: list[TaskTargetContext]
    provider_idempotency_key: ProviderIdempotencyKey


class _DeadlineDurableAgentRunner:
    """Advertise one real registry-minted durable AGENT target."""

    def __init__(self, calls: list[TaskTargetContext]) -> None:
        self._calls = calls

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        del definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self._calls.append(context)
        return completed_task_target_outcome("must not run")

    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        return target_type is TaskTargetType.AGENT

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self._calls.append(context)
        return completed_task_target_outcome(await durable_resume.dispatch())


class _Ids(InteractionIdFactory):
    """Mint deterministic opaque interaction identities."""

    def __init__(self) -> None:
        self.value = 0

    def next(self, kind: str) -> str:
        self.value += 1
        return f"{kind}-{self.value}"

    async def new_request_id(self) -> InputRequestId:
        return InputRequestId(self.next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        return ContinuationId(self.next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        return ResolutionIdempotencyKey(self.next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        return ActiveControlLeaseNonce(self.next("lease"))


class _Authorizer:
    """Allow exact echoed authorization operations."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


class _InterleavingAuthorizer(_Authorizer):
    """Resolve once after one persisted snapshot reaches authorization."""

    def __init__(self) -> None:
        self.on_inspect: Callable[[], Awaitable[None]] | None = None

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        if (
            operation is InteractionOperation.INSPECT
            and self.on_inspect is not None
        ):
            callback = self.on_inspect
            self.on_inspect = None
            await callback()
        return await super().authorize(actor, operation, target)


class _Classifier(TaskInputClassifier):
    """Allow exact normalized answer values."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.value = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        self.value += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{self.value}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Cipher:
    """Seal values with a deterministic non-plaintext prefix."""

    def __init__(self) -> None:
        self.fail_encrypt = False
        self.fail_decrypt = False

    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        assert purpose is TaskKeyPurpose.RAW_VALUE
        if self.fail_encrypt:
            raise KeyError("private key error")
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
        if self.fail_decrypt:
            raise KeyError("private key error")
        if value.ciphertext == b"unavailable":
            raise KeyError("row key unavailable")
        return bytes(byte ^ 0xA5 for byte in value.ciphertext)


class _Resumer(InputResumer):
    """Record any process-local delivery that survives a rollback."""

    def __init__(self) -> None:
        self.notifications: list[InputResumptionNotification] = []

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        self.notifications.append(notification)


def _origin(run_id: str = "run") -> ExecutionOrigin:
    definition = ExecutionDefinitionRef(
        agent_definition_locator="agent://durable-test",
        agent_definition_revision="agent-r1",
        operation_id="operation",
        operation_index=1,
        model_config_reference="model-config",
        tool_revision="tools-r1",
        capability_revision="capability-r1",
    )
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("root"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=definition,
        principal=PrincipalScope(user_id=UserId("owner")),
    )


def _request(
    run_id: str = "run",
    *,
    continuation_ttl_seconds: int = 600,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId(f"request-{run_id}"),
        continuation_id=ContinuationId(f"continuation-{run_id}"),
        origin=_origin(run_id),
        mode=RequirementMode.REQUIRED,
        reason="Confirm durable continuation.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=_NOW,
        continuation_ttl_seconds=continuation_ttl_seconds,
    )


def _deadline_worker_definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="deadline-worker", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued("durable"),
    )


def _portable(request: InputRequest) -> PortableContinuation:
    binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("gpt-5"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )
    snapshot = ContinuationSnapshot(
        snapshot_kind="openai.responses.reasoning",
        revision_binding=binding,
        model_call_id=request.origin.model_call_id,
        provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
        payload={
            "reserved_capability_call_id": "input-call",
            "replay_items": (
                {
                    "id": "reasoning-item",
                    "type": "reasoning",
                    "encrypted_content": "provider-ciphertext",
                },
            ),
        },
    )
    return PortableContinuation(
        continuation_id=request.continuation_id,
        request_id=request.request_id,
        origin=request.origin,
        provider_call_id=request.origin.model_call_id,
        provider_call_correlation_id="input-call",
        definition=request.origin.definition,
        operation_cursor=2,
        generation_settings={"temperature": 0},
        transcript=({"role": "user", "content": "opaque"},),
        observations=({"kind": "tool", "completed": True},),
        provider_snapshot=snapshot,
        revision_binding=binding,
        interaction_count=1,
        tool_loop_count=0,
        stream_sequence=3,
        state_revision=StateRevision(0),
        store_revision=ContinuationStoreRevision(0),
        created_at=_NOW,
        updated_at=_NOW,
        expires_at=request.created_at
        + timedelta(seconds=request.continuation_ttl_seconds),
    )


def _create_command(request: InputRequest) -> CreateInteractionCommand:
    return CreateInteractionCommand(
        actor=InteractionActor(principal=request.origin.principal),
        request=request,
    )


def _answer(
    created: CreateInteractionApplied,
    *,
    key: str = "answer-key",
    value: bool = True,
) -> ResolveInteractionCommand:
    record = created.record
    return ResolveInteractionCommand(
        actor=created.command.actor,
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


async def _store(
    database: PgsqlDatabase,
    *,
    cipher: _Cipher | None = None,
    store_policy: PgsqlInteractionStorePolicy | None = None,
    clock: _Clock | None = None,
    authorizer: _Authorizer | None = None,
) -> PgsqlInteractionStore:
    policy = InteractionPolicy()
    return await PgsqlInteractionStoreFactory(
        database,
        policy=policy,
        clock=clock or _Clock(),
        authorizer=authorizer or _Authorizer(),
        id_factory=_Ids(),
        cipher=cipher or _Cipher(),
        classifier=_Classifier(policy),
        store_policy=store_policy,
    ).open()


def _unit(
    *,
    row: object | None = None,
    rows: tuple[object, ...] = (),
) -> PgsqlUnitOfWork:
    """Return one scripted unit of work for private boundary tests."""
    cursor = SimpleNamespace(
        execute=AsyncMock(),
        fetchone=AsyncMock(return_value=row),
        fetchall=AsyncMock(return_value=rows),
    )
    connection = SimpleNamespace(cursor=lambda: None)
    return PgsqlUnitOfWork(
        connection=cast(Any, connection),
        cursor=cast(Any, cursor),
    )


def _seed_running_task(
    database: FakePgsqlDatabase,
    run_id: str,
    *,
    attempt_id: str = "attempt",
    segment_id: str = "segment",
    queue_item_id: str = "queue-item",
    claim_token: str = "claim-token",
) -> None:
    claim = TaskClaim(
        worker_id="worker",
        claim_token=claim_token,
        claimed_at=_NOW - timedelta(seconds=2),
        lease_expires_at=_NOW + timedelta(minutes=1),
        heartbeat_at=_NOW - timedelta(seconds=1),
    )
    claim_payload = task_pgsql._claim_to_payload(claim)
    execution_request = TaskExecutionRequest(
        definition_id="definition",
        queue="durable",
    )
    context = TaskExecutionContext(
        run_id=run_id,
        attempt_id=attempt_id,
        attempt_number=1,
        claim=claim,
    )
    database.runs[run_id] = {
        "run_id": run_id,
        "definition_id": "definition",
        "state": TaskRunState.RUNNING.value,
        "request": task_pgsql._request_to_payload(execution_request),
        "claim": claim_payload,
        "last_attempt_id": attempt_id,
        "result": None,
        "metadata": {},
        "created_at": _NOW - timedelta(minutes=1),
        "updated_at": _NOW,
    }
    database.attempts[attempt_id] = {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "attempt_number": 1,
        "state": TaskAttemptState.RUNNING.value,
        "context": task_pgsql._context_to_payload(context),
        "result": None,
        "metadata": {},
        "created_at": _NOW - timedelta(seconds=30),
        "updated_at": _NOW,
    }
    database.segments[segment_id] = {
        "segment_id": segment_id,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "segment_number": 1,
        "state": TaskAttemptSegmentState.RUNNING.value,
        "claim": claim_payload,
        "resumed_from_segment_id": None,
        "request_id": None,
        "continuation_id": None,
        "metadata": {},
        "created_at": _NOW - timedelta(seconds=20),
        "updated_at": _NOW,
    }
    database.queue_items[queue_item_id] = {
        "queue_item_id": queue_item_id,
        "run_id": run_id,
        "queue_name": "durable",
        "state": TaskQueueItemState.CLAIMED.value,
        "priority": 0,
        "available_at": _NOW - timedelta(minutes=1),
        "claimed_at": claim.claimed_at,
        "lease_expires_at": claim.lease_expires_at,
        "worker_id": claim.worker_id,
        "claim_token": claim.claim_token,
        "heartbeat_at": claim.heartbeat_at,
        "attempts": 1,
        "metadata": {},
        "created_at": _NOW - timedelta(minutes=1),
        "updated_at": _NOW,
    }


def _seed_resumed_running_task(
    database: FakePgsqlDatabase,
    *,
    run_id: str,
    segment_id: str,
) -> None:
    claim = TaskClaim(
        worker_id="resumed-worker",
        claim_token="resumed-claim-token",
        claimed_at=_NOW + timedelta(seconds=4),
        lease_expires_at=_NOW + timedelta(minutes=2),
        heartbeat_at=_NOW + timedelta(seconds=4),
    )
    claim_payload = task_pgsql._claim_to_payload(claim)
    database.runs[run_id].update(
        state=TaskRunState.RUNNING.value,
        claim=claim_payload,
        updated_at=_NOW + timedelta(seconds=4),
    )
    database.attempts["attempt"].update(
        state=TaskAttemptState.RUNNING.value,
        context=task_pgsql._context_to_payload(
            TaskExecutionContext(
                run_id=run_id,
                attempt_id="attempt",
                attempt_number=1,
                claim=claim,
            )
        ),
        updated_at=_NOW + timedelta(seconds=4),
    )
    database.queue_items["queue-item"].update(
        state=TaskQueueItemState.CLAIMED.value,
        claimed_at=claim.claimed_at,
        lease_expires_at=claim.lease_expires_at,
        worker_id=claim.worker_id,
        claim_token=claim.claim_token,
        heartbeat_at=claim.heartbeat_at,
        updated_at=_NOW + timedelta(seconds=4),
    )
    database.segments[segment_id] = {
        "segment_id": segment_id,
        "attempt_id": "attempt",
        "run_id": run_id,
        "segment_number": 2,
        "state": TaskAttemptSegmentState.RUNNING.value,
        "claim": claim_payload,
        "resumed_from_segment_id": "segment",
        "request_id": None,
        "continuation_id": None,
        "checkpoint_id": None,
        "metadata": {},
        "created_at": _NOW + timedelta(seconds=4),
        "updated_at": _NOW + timedelta(seconds=4),
    }


def _expire_reentry_claim(
    database: FakePgsqlDatabase,
    *,
    expires_at: datetime,
) -> None:
    queue = database.queue_items["queue-item"]
    queue["lease_expires_at"] = expires_at
    run_claim = dict(cast(Mapping[str, object], database.runs["run"]["claim"]))
    run_claim["lease_expires_at"] = expires_at.isoformat()
    database.runs["run"]["claim"] = run_claim
    attempt_context = dict(
        cast(Mapping[str, object], database.attempts["attempt"]["context"])
    )
    attempt_claim = dict(cast(Mapping[str, object], attempt_context["claim"]))
    attempt_claim["lease_expires_at"] = expires_at.isoformat()
    attempt_context["claim"] = attempt_claim
    database.attempts["attempt"]["context"] = attempt_context
    for segment in database.segments.values():
        if segment["state"] != TaskAttemptSegmentState.RUNNING.value:
            continue
        segment_claim = dict(cast(Mapping[str, object], segment["claim"]))
        segment_claim["lease_expires_at"] = expires_at.isoformat()
        segment["claim"] = segment_claim


async def _create_suspended_task(
    database: FakePgsqlDatabase,
    store: PgsqlInteractionStore,
    task_store: PgsqlTaskStore,
    *,
    run_id: str,
    suffix: str,
) -> CreateInteractionApplied:
    attempt_id = f"attempt-{suffix}"
    segment_id = f"segment-{suffix}"
    queue_item_id = f"queue-{suffix}"
    claim_token = f"claim-{suffix}"
    _seed_running_task(
        database,
        run_id,
        attempt_id=attempt_id,
        segment_id=segment_id,
        queue_item_id=queue_item_id,
        claim_token=claim_token,
    )
    request = _request(run_id)
    suspended = await PgsqlDurableTaskCoordinator(
        store,
        task_store,
    ).create_and_suspend(
        _create_command(request),
        _portable(request),
        queue_item_id=queue_item_id,
        claim_token=claim_token,
        segment_id=segment_id,
        task_run_id=run_id,
        checkpoint_id=f"checkpoint-{suffix}",
    )
    return suspended.interaction


async def _commit_resolution_before_task_reentry(
    database: FakePgsqlDatabase,
    *,
    request: InputRequest | None = None,
) -> tuple[
    PgsqlInteractionStore,
    PgsqlTaskStore,
    CreateInteractionApplied,
]:
    _seed_running_task(database, "run")
    store = await _store(database)
    task_ids = _Ids()
    task_store = PgsqlTaskStore(
        database,
        clock=lambda: _NOW,
        id_factory=lambda: task_ids.next("task"),
    )
    coordinator = PgsqlDurableTaskCoordinator(store, task_store)
    request = request or _request()
    staged = await coordinator.create_and_suspend(
        _create_command(request),
        _portable(request),
        queue_item_id="queue-item",
        claim_token="claim-token",
        segment_id="segment",
        task_run_id="run",
        checkpoint_id="checkpoint",
    )
    await coordinator.resolve_and_requeue(
        _answer(staged.interaction),
        task_run_id="run",
    )
    return store, task_store, staged.interaction


async def _prepare_resumed_dispatch(
    database: FakePgsqlDatabase,
    *,
    owner_id: str = "resumed-worker",
) -> tuple[PgsqlInteractionStore, PgsqlTaskStore, PortableContinuation]:
    (
        store,
        task_store,
        interaction,
    ) = await _commit_resolution_before_task_reentry(database)
    ready = await store.get_continuation(
        interaction.record.request.continuation_id
    )
    receipt = await store.claim(
        ready.continuation_id,
        expected_store_revision=ready.store_revision,
        owner_id=ContinuationClaimOwnerId(owner_id),
        lease_expires_at=_NOW + timedelta(minutes=2),
        dispatch_id=ContinuationDispatchId("resumed-dispatch"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
        now=_NOW + timedelta(seconds=2),
    )
    dispatching = await store.mark_dispatching(
        ready.continuation_id,
        expected_store_revision=receipt.continuation.store_revision,
        owner_id=ContinuationClaimOwnerId(owner_id),
        fencing_token=receipt.fencing_token,
        now=_NOW + timedelta(seconds=3),
    )
    dispatched = await store.mark_dispatched(
        ready.continuation_id,
        expected_store_revision=dispatching.store_revision,
        owner_id=ContinuationClaimOwnerId(owner_id),
        fencing_token=receipt.fencing_token,
        now=_NOW + timedelta(seconds=4),
    )
    _seed_resumed_running_task(
        database,
        run_id="run",
        segment_id="segment-next",
    )
    return store, task_store, dispatched


async def _prepare_completed_provider(
    database: FakePgsqlDatabase,
) -> tuple[
    PgsqlInteractionStore,
    PgsqlTaskStore,
    PortableContinuation,
    ContinuationCompletionCommand,
]:
    store, task_store, dispatched = await _prepare_resumed_dispatch(
        database,
        owner_id="resumed-claim-token",
    )
    completion = ContinuationCompletionCommand(
        continuation_id=dispatched.continuation_id,
        expected_store_revision=dispatched.store_revision,
        owner_id=ContinuationClaimOwnerId("resumed-claim-token"),
        fencing_token=dispatched.fencing_token,
        result_digest="d" * 64,
    )

    async def complete_provider(unit: object) -> object:
        return await store._complete_continuation_in_unit(
            cast(Any, unit),
            continuation_id=dispatched.continuation_id,
            expected_store_revision=completion.expected_store_revision,
            owner_id=completion.owner_id,
            fencing_token=completion.fencing_token,
            result_digest=completion.result_digest,
            now=_NOW + timedelta(seconds=5),
            expected_task_run_id="run",
        )

    completed = cast(
        PortableContinuation,
        await store._transaction(
            "test_complete_task_provider",
            complete_provider,
        ),
    )
    return store, task_store, completed, completion


@dataclass(frozen=True, slots=True)
class _CancelRequestedSettlementCase:
    database: FakePgsqlDatabase
    coordinator: PgsqlDurableTaskCoordinator
    completion: ContinuationCompletionCommand
    settlement: TaskDurableResumeSettlement
    operation: str


async def _prepare_cancel_requested_settlement_case(
    operation: str,
) -> _CancelRequestedSettlementCase:
    if operation in {
        "completed_provider_failure",
        "completed_provider_cancellation",
    }:
        database = FakePgsqlDatabase()
        store, task_store, _, completion = await _prepare_completed_provider(
            database
        )
        settlement: TaskDurableResumeSettlement
        if operation == "completed_provider_failure":
            settlement = TaskDurableResumeFailure(
                result=TaskExecutionResult(
                    error={"code": "post_provider_processing_failed"}
                )
            )
        else:
            settlement = TaskDurableResumeCancellation(
                result=TaskExecutionResult(
                    error={"code": "post_provider_processing_cancelled"}
                )
            )
    else:
        database = FakePgsqlDatabase()
        store, task_store, dispatched = await _prepare_resumed_dispatch(
            database
        )
        if operation == "success":
            settlement = TaskDurableResumeSuccess(
                result=TaskExecutionResult(output_summary={"answer": 42})
            )
        elif operation == "failure":
            settlement = TaskDurableResumeFailure(
                result=TaskExecutionResult(error={"code": "output_invalid"})
            )
        elif operation == "cancellation":
            settlement = TaskDurableResumeCancellation(
                result=TaskExecutionResult(error={"code": "worker_cancelled"})
            )
        elif operation == "ambiguous":
            settlement = TaskDurableResumeFailure(
                result=TaskExecutionResult(
                    error={"code": "provider_dispatch_ambiguous"}
                )
            )
        else:
            raise AssertionError("unknown cancellation settlement operation")
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=dispatched.fencing_token,
            result_digest=task_durable_resume_settlement_digest(settlement),
        )
    cancelled = await TaskClient(
        task_store,
        target=_unused_task_target,
        clock=lambda: _NOW,
    ).cancel("run")
    assert cancelled.state is TaskRunState.CANCEL_REQUESTED
    return _CancelRequestedSettlementCase(
        database=database,
        coordinator=PgsqlDurableTaskCoordinator(store, task_store),
        completion=completion,
        settlement=settlement,
        operation=operation,
    )


async def _invoke_cancel_requested_settlement_case(
    prepared: _CancelRequestedSettlementCase,
    *,
    completion: ContinuationCompletionCommand,
    settlement: TaskDurableResumeSettlement,
    claim_token: str,
) -> TaskQueueCompletion:
    if prepared.operation == "ambiguous":
        if type(settlement) is not TaskDurableResumeFailure:
            raise AssertionError("ambiguous settlement must fail")
        ambiguity_commit = await prepared.coordinator.mark_resume_ambiguous(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token=claim_token,
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=6),
        )
        return ambiguity_commit.completion
    if prepared.operation in {
        "completed_provider_failure",
        "completed_provider_cancellation",
    }:
        if not isinstance(
            settlement,
            TaskDurableResumeFailure | TaskDurableResumeCancellation,
        ):
            raise AssertionError("completed provider settlement must fail")
        completed_commit = (
            await prepared.coordinator.terminalize_completed_resume(
                completion,
                settlement,
                queue_item_id="queue-item",
                claim_token=claim_token,
                segment_id="segment-next",
                task_run_id="run",
                request_id="request-run",
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=6),
            )
        )
        return completed_commit.completion
    settlement_commit = await prepared.coordinator.settle_resume(
        completion,
        settlement,
        queue_item_id="queue-item",
        claim_token=claim_token,
        segment_id="segment-next",
        task_run_id="run",
        now=_NOW + timedelta(seconds=6),
    )
    return settlement_commit.completion


async def _prepare_claimed_reentry(
    database: FakePgsqlDatabase,
    *,
    request: InputRequest | None = None,
) -> tuple[
    PgsqlInteractionStore,
    PgsqlTaskStore,
    InputRequest,
    TaskClaim,
]:
    (
        store,
        task_store,
        interaction,
    ) = await _commit_resolution_before_task_reentry(
        database,
        request=request,
    )
    request = interaction.record.request
    record_row = database.records[str(request.request_id)]
    await task_store.requeue_suspended(
        "run",
        request_id=str(request.request_id),
        continuation_id=str(request.continuation_id),
        resolution_revision=cast(int, record_row["state_revision"]),
        now=_NOW + timedelta(seconds=2),
    )
    claim = TaskClaim(
        worker_id="resumed-worker",
        claim_token="resumed-claim-token",
        claimed_at=_NOW + timedelta(seconds=3),
        lease_expires_at=_NOW + timedelta(minutes=2),
        heartbeat_at=_NOW + timedelta(seconds=3),
    )
    claim_payload = task_pgsql._claim_to_payload(claim)
    database.runs["run"].update(
        state=TaskRunState.CLAIMED.value,
        claim=claim_payload,
        updated_at=_NOW + timedelta(seconds=3),
    )
    attempt_context = dict(
        cast(Mapping[str, object], database.attempts["attempt"]["context"])
    )
    attempt_context["claim"] = claim_payload
    database.attempts["attempt"].update(
        context=attempt_context,
        updated_at=_NOW + timedelta(seconds=3),
    )
    database.queue_items["queue-item"].update(
        state=TaskQueueItemState.CLAIMED.value,
        claimed_at=claim.claimed_at,
        lease_expires_at=claim.lease_expires_at,
        worker_id=claim.worker_id,
        claim_token=claim.claim_token,
        heartbeat_at=claim.heartbeat_at,
        updated_at=_NOW + timedelta(seconds=3),
    )
    return store, task_store, request, claim


async def _prepare_cancel_requested_startup_boundary(
    boundary: str,
) -> tuple[
    FakePgsqlDatabase,
    PgsqlInteractionStore,
    PgsqlTaskStore,
    InputRequest,
    TaskClaim,
    PortableContinuation,
    str,
]:
    database = FakePgsqlDatabase()
    store, task_store, request, claim = await _prepare_claimed_reentry(
        database
    )
    ready = await store.get_continuation(request.continuation_id)
    receipt = await store.claim(
        request.continuation_id,
        expected_store_revision=ready.store_revision,
        owner_id=ContinuationClaimOwnerId(claim.claim_token),
        lease_expires_at=claim.lease_expires_at,
        dispatch_id=ContinuationDispatchId(f"startup-{boundary}"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
        now=_NOW + timedelta(seconds=4),
    )
    await task_store.transition_run(
        "run",
        from_states={TaskRunState.CLAIMED},
        to_state=TaskRunState.RUNNING,
        reason="started",
        claim_token=claim.claim_token,
    )
    active_segment_id = "segment"
    if boundary in {
        "attempt_running",
        "segment_created",
        "segment_running",
    }:
        await task_store.transition_attempt(
            "attempt",
            from_states={TaskAttemptState.SUSPENDED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claim.claim_token,
        )
    if boundary in {"segment_created", "segment_running"}:
        active_segment = await task_store.create_attempt_segment(
            "attempt",
            claim_token=claim.claim_token,
            resumed_from_segment_id="segment",
        )
        active_segment_id = active_segment.segment_id
    if boundary == "segment_running":
        await task_store.transition_attempt_segment(
            active_segment_id,
            from_states={TaskAttemptSegmentState.CREATED},
            to_state=TaskAttemptSegmentState.RUNNING,
            reason="started",
            claim_token=claim.claim_token,
        )
    if boundary not in {
        "run_running",
        "attempt_running",
        "segment_created",
        "segment_running",
    }:
        raise AssertionError("unknown resumed startup boundary")
    cancelled = await TaskClient(
        task_store,
        target=_unused_task_target,
        clock=lambda: _NOW + timedelta(seconds=5),
    ).cancel("run")
    assert cancelled.state is TaskRunState.CANCEL_REQUESTED
    _expire_reentry_claim(
        database,
        expires_at=_NOW + timedelta(seconds=5),
    )
    return (
        database,
        store,
        task_store,
        request,
        claim,
        receipt.continuation,
        active_segment_id,
    )


async def _insert_corrupt_foreign_scope(
    database: FakePgsqlDatabase,
    store: PgsqlInteractionStore,
    *,
    run_id: str,
    suffix: str,
) -> InputRequest:
    request = _request(f"{run_id}-{suffix}")
    request = replace(
        request,
        origin=replace(
            request.origin,
            run_id=RunId(run_id),
            branch_id=BranchId(f"foreign-root-{suffix}"),
            principal=PrincipalScope(
                user_id=UserId(f"foreign-owner-{suffix}")
            ),
        ),
    )
    created = await store.create_durable(
        _create_command(request),
        _portable(request),
    )
    assert isinstance(created, CreateInteractionApplied)
    database.records[str(request.request_id)]["ciphertext"] = b"\x00"
    database.continuations[str(request.continuation_id)][
        "ciphertext"
    ] = b"\x00"
    return request


def _successor_interaction(
    dispatched: PortableContinuation,
    *,
    suffix: str,
) -> tuple[InputRequest, PortableContinuation]:
    request = replace(
        _request(),
        request_id=InputRequestId(f"request-{suffix}"),
        continuation_id=ContinuationId(f"continuation-{suffix}"),
    )
    base = _portable(request)
    snapshot = base.provider_snapshot
    assert isinstance(snapshot, ContinuationSnapshot)
    continuation = replace(
        base,
        provider_call_correlation_id=f"input-call-{suffix}",
        provider_snapshot=replace(
            snapshot,
            payload={
                "reserved_capability_call_id": f"input-call-{suffix}",
                "replay_items": (
                    {
                        "id": f"reasoning-item-{suffix}",
                        "type": "reasoning",
                        "encrypted_content": f"provider-ciphertext-{suffix}",
                    },
                ),
            },
        ),
        interaction_count=dispatched.interaction_count + 1,
        stream_sequence=dispatched.stream_sequence + 1,
    )
    return request, continuation


async def _deadline_worker_harness(
    *,
    blocked_loader: bool = False,
) -> _DeadlineWorkerHarness:
    database = FullFakePgsqlDatabase()
    request = _request(continuation_ttl_seconds=60)
    dispatch_id = derive_continuation_dispatch_id(request.continuation_id)
    provider_idempotency_key = derive_provider_idempotency_key(
        request.continuation_id,
        dispatch_id,
    )
    portable = _portable(request)
    snapshot = portable.provider_snapshot
    assert type(snapshot) is ContinuationSnapshot
    portable = replace(
        portable,
        provider_snapshot=replace(
            snapshot,
            provider_idempotency_key=provider_idempotency_key,
        ),
    )
    with patch(
        f"{__name__}._portable",
        return_value=portable,
    ):
        store, task_store, request, claim = await _prepare_claimed_reentry(
            database,
            request=request,
        )
    await task_store.register_definition(
        _deadline_worker_definition(),
        definition_hash="definition",
    )
    deadline = request.created_at + timedelta(
        seconds=request.continuation_ttl_seconds
    )
    current = [deadline - timedelta(seconds=1)]
    loader: _DeadlineFailureLoader | _BlockingDeadlineLoader = (
        _BlockingDeadlineLoader(current, deadline)
        if blocked_loader
        else _DeadlineFailureLoader(current, deadline)
    )
    resolver = ContinuationRuntimeResolver(
        loader,
        clock=lambda: current[0],
    )
    resume_coordinator = TaskDurableResumeCoordinator(
        store,
        DurableAgentContinuationResumer(
            store,
            resolver,
            clock=lambda: current[0],
        ),
    )
    task_coordinator = PgsqlDurableTaskCoordinator(store, task_store)
    queue = _PersistedClaimQueue(
        database,
        task_store,
        allow_heartbeats=blocked_loader,
    )
    target_calls: list[TaskTargetContext] = []
    target = TaskTargetRunnerRegistry(
        _DeadlineDurableAgentRunner(target_calls)
    )

    worker = TaskWorker(
        task_store,
        cast(Any, queue),
        target=target,
        worker_id=claim.worker_id,
        queue_name="durable",
        lease_seconds=300,
        heartbeat_seconds=0.001 if blocked_loader else None,
        durable_suspension_coordinator=task_coordinator,
        durable_resume_coordinator=resume_coordinator,
        clock=lambda: current[0],
    )
    return _DeadlineWorkerHarness(
        database=database,
        store=store,
        task_store=task_store,
        task_coordinator=task_coordinator,
        request=request,
        claim=claim,
        loader=loader,
        queue=queue,
        worker=worker,
        target_calls=target_calls,
        provider_idempotency_key=provider_idempotency_key,
    )


class PgsqlInteractionStoreTest(IsolatedAsyncioTestCase):
    async def test_scope_ownership_presence_is_content_free_and_atomic(
        self,
    ) -> None:
        for operation in ("cancel", "supersede"):
            with self.subTest(operation=operation):
                database = FakePgsqlDatabase()
                store = await _store(database)
                mixed_run_id = f"scope-presence-mixed-{operation}"
                foreign = await _insert_corrupt_foreign_scope(
                    database,
                    store,
                    run_id=mixed_run_id,
                    suffix=operation,
                )
                actor_request = _request(mixed_run_id)
                actor_request = replace(
                    actor_request,
                    origin=replace(
                        actor_request.origin,
                        run_id=RunId(mixed_run_id),
                    ),
                )
                actor_created = await store.create(
                    _create_command(actor_request)
                )
                self.assertIsInstance(
                    actor_created,
                    CreateInteractionApplied,
                )
                assert isinstance(actor_created, CreateInteractionApplied)
                actor = actor_created.command.actor

                async def mutate(
                    scope: InteractionExecutionScope,
                ) -> object:
                    if operation == "cancel":
                        return await store.terminalize_scope(
                            TerminalizeInteractionScopeCommand(
                                actor=actor,
                                scope=scope,
                                provenance=AnswerProvenance.HUMAN,
                            )
                        )
                    return await store.supersede_scope(
                        SupersedeInteractionScopeCommand(
                            actor=actor,
                            scope=scope,
                            provenance=AnswerProvenance.HUMAN,
                        )
                    )

                mixed_scope = InteractionExecutionScope(
                    run_id=RunId(mixed_run_id),
                )
                database.fail_query = 'UPDATE "interaction_store_metadata"'
                with self.assertRaises(PgsqlOperationError):
                    await mutate(mixed_scope)
                actor_row = database.records[str(actor_request.request_id)]
                self.assertEqual(
                    actor_row["request_state"],
                    RequestState.PENDING.value,
                )
                self.assertEqual(
                    database.records[str(foreign.request_id)]["ciphertext"],
                    b"\x00",
                )

                database.fail_query = None
                applied = await mutate(mixed_scope)
                if operation == "cancel":
                    self.assertIsInstance(
                        applied,
                        ScopeCancellationApplied,
                    )
                    assert isinstance(applied, ScopeCancellationApplied)
                else:
                    self.assertIsInstance(
                        applied,
                        ScopeSupersessionApplied,
                    )
                    assert isinstance(applied, ScopeSupersessionApplied)
                records = applied.records
                self.assertEqual(
                    tuple(record.request.request_id for record in records),
                    (actor_request.request_id,),
                )

                replayed = await mutate(mixed_scope)
                if operation == "cancel":
                    self.assertIsInstance(
                        replayed,
                        ScopeCancellationReplayed,
                    )
                else:
                    self.assertIsInstance(
                        replayed,
                        ScopeSupersessionReplayed,
                    )

                foreign_only_run_id = f"scope-presence-foreign-{operation}"
                foreign_only = await _insert_corrupt_foreign_scope(
                    database,
                    store,
                    run_id=foreign_only_run_id,
                    suffix=f"only-{operation}",
                )
                foreign_only_row = database.records[
                    str(foreign_only.request_id)
                ]
                foreign_only_row.update(
                    turn_id="foreign-only-turn",
                    task_id="foreign-only-task",
                    agent_id="foreign-only-agent",
                )
                rejected = await mutate(
                    InteractionExecutionScope(
                        run_id=RunId(foreign_only_run_id),
                        turn_id=TurnId("foreign-only-turn"),
                        task_id=TaskId("foreign-only-task"),
                        agent_id=AgentId("foreign-only-agent"),
                        branch_id=foreign_only.origin.branch_id,
                    )
                )
                if operation == "cancel":
                    self.assertIsInstance(
                        rejected,
                        ScopeCancellationRejected,
                    )
                    assert isinstance(rejected, ScopeCancellationRejected)
                else:
                    self.assertIsInstance(
                        rejected,
                        ScopeSupersessionRejected,
                    )
                    assert isinstance(rejected, ScopeSupersessionRejected)
                self.assertEqual(
                    rejected.error.code,
                    InputErrorCode.FORBIDDEN,
                )
                self.assertEqual(
                    database.records[str(foreign_only.request_id)][
                        "ciphertext"
                    ],
                    b"\x00",
                )

                precedence_run_id = (
                    f"scope-presence-record-precedence-{operation}"
                )
                precedence_foreign = await _insert_corrupt_foreign_scope(
                    database,
                    store,
                    run_id=precedence_run_id,
                    suffix=f"precedence-{operation}",
                )
                actor_branch = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=actor,
                        registration=InteractionBranchRegistration(
                            run_id=RunId(precedence_run_id),
                            branch_id=precedence_foreign.origin.branch_id,
                            parent_branch_id=BranchId("actor-root"),
                            principal=actor.principal,
                        ),
                    )
                )
                self.assertIsInstance(
                    actor_branch,
                    InteractionBranchRegistrationApplied,
                )
                precedence_rejected = await mutate(
                    InteractionExecutionScope(
                        run_id=RunId(precedence_run_id),
                        branch_id=precedence_foreign.origin.branch_id,
                    )
                )
                if operation == "cancel":
                    self.assertIsInstance(
                        precedence_rejected,
                        ScopeCancellationRejected,
                    )
                else:
                    self.assertIsInstance(
                        precedence_rejected,
                        ScopeSupersessionRejected,
                    )
                assert isinstance(
                    precedence_rejected,
                    ScopeCancellationRejected | ScopeSupersessionRejected,
                )
                self.assertEqual(
                    precedence_rejected.error.code,
                    InputErrorCode.FORBIDDEN,
                )
                self.assertEqual(
                    database.records[str(precedence_foreign.request_id)][
                        "ciphertext"
                    ],
                    b"\x00",
                )

                requestless_run_id = f"scope-presence-requestless-{operation}"
                requestless_branch_id = BranchId("requestless-branch")
                requestless_principal = PrincipalScope(
                    user_id=UserId(f"requestless-owner-{operation}")
                )
                registered = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=InteractionActor(
                            principal=requestless_principal
                        ),
                        registration=InteractionBranchRegistration(
                            run_id=RunId(requestless_run_id),
                            branch_id=requestless_branch_id,
                            parent_branch_id=BranchId("requestless-root"),
                            principal=requestless_principal,
                        ),
                    )
                )
                self.assertIsInstance(
                    registered,
                    InteractionBranchRegistrationApplied,
                )
                requestless = await mutate(
                    InteractionExecutionScope(
                        run_id=RunId(requestless_run_id),
                        branch_id=requestless_branch_id,
                    )
                )
                if operation == "cancel":
                    self.assertIsInstance(
                        requestless,
                        ScopeCancellationRejected,
                    )
                else:
                    self.assertIsInstance(
                        requestless,
                        ScopeSupersessionRejected,
                    )

                empty = await mutate(
                    InteractionExecutionScope(
                        run_id=RunId(f"scope-presence-empty-{operation}"),
                    )
                )
                if operation == "cancel":
                    self.assertIsInstance(
                        empty,
                        ScopeCancellationReplayed,
                    )
                else:
                    self.assertIsInstance(
                        empty,
                        ScopeSupersessionReplayed,
                    )
                database.omit_scope_ownership_presence_result = True
                with self.assertRaises(
                    PgsqlInteractionStoreError
                ) as missing_presence:
                    await mutate(
                        InteractionExecutionScope(
                            run_id=RunId(
                                f"scope-presence-missing-{operation}"
                            ),
                        )
                    )
                self.assertEqual(
                    missing_presence.exception.code,
                    InputErrorCode.UNAVAILABLE,
                )
                database.omit_scope_ownership_presence_result = False
                self.assertTrue(
                    any(
                        query == _SELECT_SCOPE_OWNERSHIP_PRESENCE_SQL
                        for query, _parameters in database.executed
                    )
                )
                await store.aclose()

    async def test_creation_paths_ignore_corrupt_foreign_same_run_scope(
        self,
    ) -> None:
        public_database = FakePgsqlDatabase()
        public_store = await _store(public_database)
        await _insert_corrupt_foreign_scope(
            public_database,
            public_store,
            run_id="create-isolated",
            suffix="public",
        )
        public_created = await public_store.create(
            _create_command(_request("create-isolated"))
        )
        self.assertIsInstance(public_created, CreateInteractionApplied)

        suspend_database = FakePgsqlDatabase()
        _seed_running_task(suspend_database, "suspend-isolated")
        suspend_store = await _store(suspend_database)
        suspend_task_store = PgsqlTaskStore(
            suspend_database,
            clock=lambda: _NOW,
        )
        await _insert_corrupt_foreign_scope(
            suspend_database,
            suspend_store,
            run_id="suspend-isolated",
            suffix="suspend",
        )
        suspend_request = _request("suspend-isolated")
        suspended = await PgsqlDurableTaskCoordinator(
            suspend_store,
            suspend_task_store,
        ).create_and_suspend(
            _create_command(suspend_request),
            _portable(suspend_request),
            queue_item_id="queue-item",
            claim_token="claim-token",
            segment_id="segment",
            task_run_id="suspend-isolated",
            checkpoint_id="checkpoint",
        )
        self.assertIsInstance(
            suspended.interaction,
            CreateInteractionApplied,
        )

        pending_database = FakePgsqlDatabase()
        pending_store = await _store(pending_database)
        pending_task_store = PgsqlTaskStore(
            pending_database,
            clock=lambda: _NOW,
        )
        await _create_suspended_task(
            pending_database,
            pending_store,
            pending_task_store,
            run_id="pending-isolated",
            suffix="pending-isolated",
        )
        await _insert_corrupt_foreign_scope(
            pending_database,
            pending_store,
            run_id="pending-isolated",
            suffix="pending",
        )
        pending_request = _request("pending-isolated-successor")
        pending_request = replace(
            pending_request,
            origin=replace(
                pending_request.origin,
                run_id=RunId("pending-isolated"),
                branch_id=BranchId("pending-successor-root"),
            ),
        )
        pending = await PgsqlDurableTaskCoordinator(
            pending_store,
            pending_task_store,
        ).create_pending_interaction(
            _create_command(pending_request),
            _portable(pending_request),
            task_run_id="pending-isolated",
            checkpoint_id="checkpoint-pending",
        )
        self.assertIsInstance(pending, CreateInteractionApplied)

        resuspend_database = FakePgsqlDatabase()
        (
            resuspend_store,
            resuspend_task_store,
            dispatched,
        ) = await _prepare_resumed_dispatch(resuspend_database)
        await _insert_corrupt_foreign_scope(
            resuspend_database,
            resuspend_store,
            run_id="run",
            suffix="resuspend",
        )
        successor_request, successor = _successor_interaction(
            dispatched,
            suffix="isolated-successor",
        )
        resuspended = await PgsqlDurableTaskCoordinator(
            resuspend_store,
            resuspend_task_store,
        ).complete_and_resuspend(
            ContinuationCompletionCommand(
                continuation_id=dispatched.continuation_id,
                expected_store_revision=dispatched.store_revision,
                owner_id=ContinuationClaimOwnerId("resumed-worker"),
                fencing_token=dispatched.fencing_token,
                result_digest="c" * 64,
            ),
            _create_command(successor_request),
            successor,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            checkpoint_id="checkpoint-successor",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertIsInstance(
            resuspended.interaction,
            CreateInteractionApplied,
        )

    async def test_exact_task_lookups_hide_foreign_and_missing_rows(
        self,
    ) -> None:
        def signature(
            error: InteractionNotFoundError,
        ) -> tuple[
            InputErrorCode,
            str,
            str,
        ]:
            return error.code, error.path, error.safe_message

        settlement_signatures: list[tuple[InputErrorCode, str, str]] = []
        for absence in ("foreign-corrupt", "missing"):
            with self.subTest(operation="settle", absence=absence):
                database = FakePgsqlDatabase()
                (
                    store,
                    task_store,
                    dispatched,
                ) = await _prepare_resumed_dispatch(database)
                settlement = TaskDurableResumeSuccess(
                    result=TaskExecutionResult(output_summary={"answer": 42})
                )
                completion = ContinuationCompletionCommand(
                    continuation_id=dispatched.continuation_id,
                    expected_store_revision=dispatched.store_revision,
                    owner_id=ContinuationClaimOwnerId("resumed-worker"),
                    fencing_token=dispatched.fencing_token,
                    result_digest=(
                        task_durable_resume_settlement_digest(settlement)
                    ),
                )
                continuation_key = str(dispatched.continuation_id)
                if absence == "foreign-corrupt":
                    database.continuations[continuation_key].update(
                        task_run_id="foreign-run",
                        ciphertext=b"\x00",
                    )
                else:
                    database.continuations.pop(continuation_key)
                before = database.snapshot()

                with self.assertRaises(InteractionNotFoundError) as raised:
                    await PgsqlDurableTaskCoordinator(
                        store,
                        task_store,
                    ).settle_resume(
                        completion,
                        settlement,
                        queue_item_id="queue-item",
                        claim_token="resumed-claim-token",
                        segment_id="segment-next",
                        task_run_id="run",
                        now=_NOW + timedelta(seconds=5),
                    )

                settlement_signatures.append(signature(raised.exception))
                self.assertEqual(database.snapshot(), before)
        self.assertEqual(
            settlement_signatures,
            [settlement_signatures[0]] * 2,
        )

        completed_signatures: list[tuple[InputErrorCode, str, str]] = []
        for absence in ("foreign-corrupt", "missing"):
            with self.subTest(operation="completed", absence=absence):
                database = FakePgsqlDatabase()
                (
                    store,
                    task_store,
                    completed,
                    completion,
                ) = await _prepare_completed_provider(database)
                continuation_key = str(completed.continuation_id)
                if absence == "foreign-corrupt":
                    database.continuations[continuation_key].update(
                        task_run_id="foreign-run",
                        ciphertext=b"\x00",
                    )
                else:
                    database.continuations.pop(continuation_key)
                before = database.snapshot()
                failure = TaskDurableResumeFailure(
                    result=TaskExecutionResult(
                        error={"code": "post_provider_failure"}
                    )
                )

                with self.assertRaises(InteractionNotFoundError) as raised:
                    await PgsqlDurableTaskCoordinator(
                        store,
                        task_store,
                    ).terminalize_completed_resume(
                        completion,
                        failure,
                        queue_item_id="queue-item",
                        claim_token="resumed-claim-token",
                        segment_id="segment-next",
                        task_run_id="run",
                        request_id="request-run",
                        checkpoint_id="checkpoint",
                        now=_NOW + timedelta(seconds=6),
                    )

                completed_signatures.append(signature(raised.exception))
                self.assertEqual(database.snapshot(), before)
        self.assertEqual(
            completed_signatures,
            [completed_signatures[0]] * 2,
        )

        rejection_signatures: list[tuple[InputErrorCode, str, str]] = []
        for absence in ("foreign-corrupt", "missing"):
            with self.subTest(operation="reject", absence=absence):
                database = FakePgsqlDatabase()
                (
                    store,
                    task_store,
                    request,
                    claim,
                ) = await _prepare_claimed_reentry(database)
                ready = await store.get_continuation(request.continuation_id)
                receipt = await store.claim(
                    ready.continuation_id,
                    expected_store_revision=ready.store_revision,
                    owner_id=ContinuationClaimOwnerId(claim.claim_token),
                    lease_expires_at=claim.lease_expires_at,
                    dispatch_id=ContinuationDispatchId(
                        "hidden-rejected-dispatch"
                    ),
                    provider_idempotency_key=ProviderIdempotencyKey(
                        "provider-key"
                    ),
                    now=_NOW + timedelta(seconds=3),
                )
                failure = TaskDurableResumeFailure(
                    result=TaskExecutionResult(
                        error={"code": "resume_setup_rejected"}
                    )
                )
                rejection = ContinuationRejectionCommand(
                    continuation_id=receipt.continuation.continuation_id,
                    expected_store_revision=(
                        receipt.continuation.store_revision
                    ),
                    owner_id=ContinuationClaimOwnerId(claim.claim_token),
                    fencing_token=receipt.fencing_token,
                    result_digest=(
                        task_durable_resume_settlement_digest(failure)
                    ),
                )
                continuation_key = str(request.continuation_id)
                if absence == "foreign-corrupt":
                    database.continuations[continuation_key].update(
                        task_run_id="foreign-run",
                        ciphertext=b"\x00",
                    )
                else:
                    database.continuations.pop(continuation_key)
                before = database.snapshot()

                with self.assertRaises(InteractionNotFoundError) as raised:
                    await PgsqlDurableTaskCoordinator(
                        store,
                        task_store,
                    ).fail_admitted_reentry(
                        rejection,
                        failure,
                        queue_item_id="queue-item",
                        claim_token=claim.claim_token,
                        task_run_id="run",
                        request_id=str(request.request_id),
                        continuation_id=str(request.continuation_id),
                        checkpoint_id="checkpoint",
                        now=_NOW + timedelta(seconds=4),
                    )

                rejection_signatures.append(signature(raised.exception))
                self.assertEqual(database.snapshot(), before)
        self.assertEqual(
            rejection_signatures,
            [rejection_signatures[0]] * 2,
        )

    async def test_coordinator_operations_isolate_corrupt_scopes(
        self,
    ) -> None:
        async def invoke(
            operation: str,
            coordinator: PgsqlDurableTaskCoordinator,
            interaction: CreateInteractionApplied,
            run_id: str,
        ) -> object:
            scope = InteractionExecutionScope(
                run_id=interaction.record.request.origin.run_id,
            )
            if operation == "resolve":
                return await coordinator.resolve_and_requeue(
                    _answer(interaction),
                    task_run_id=run_id,
                )
            if operation == "cancel":
                return await coordinator.cancel_suspended_task(
                    TerminalizeInteractionScopeCommand(
                        actor=interaction.command.actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    ),
                    task_run_id=run_id,
                )
            if operation == "supersede":
                return await coordinator.supersede_suspended_task(
                    SupersedeInteractionScopeCommand(
                        actor=interaction.command.actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    ),
                    task_run_id=run_id,
                )
            assert operation == "trusted_cancel"
            return await coordinator.cancel_input_required_task(
                task_run_id=run_id,
                now=_NOW,
                metadata={},
            )

        expected_states = {
            "resolve": RequestState.ANSWERED,
            "cancel": RequestState.CANCELLED,
            "supersede": RequestState.SUPERSEDED,
            "trusted_cancel": RequestState.CANCELLED,
        }
        completed_operations: list[tuple[str, RequestState]] = []
        for operation, expected_state in expected_states.items():
            with self.subTest(operation=operation, corruption="unrelated"):
                database = FakePgsqlDatabase()
                store = await _store(database)
                task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
                run_id = f"coordinator-isolated-{operation}"
                interaction = await _create_suspended_task(
                    database,
                    store,
                    task_store,
                    run_id=run_id,
                    suffix=run_id,
                )
                unrelated_request = _request(
                    f"coordinator-unrelated-{operation}"
                )
                unrelated_request = replace(
                    unrelated_request,
                    origin=replace(
                        unrelated_request.origin,
                        run_id=RunId(run_id),
                        branch_id=BranchId(f"unrelated-root-{operation}"),
                        principal=PrincipalScope(
                            user_id=UserId(f"unrelated-{operation}")
                        ),
                    ),
                )
                unrelated = await store.create_durable(
                    _create_command(unrelated_request),
                    _portable(unrelated_request),
                )
                self.assertIsInstance(unrelated, CreateInteractionApplied)
                unrelated_row = database.records[
                    str(unrelated_request.request_id)
                ]
                unrelated_row["ciphertext"] = b"\x00"
                unrelated_before = dict(unrelated_row)
                unrelated_continuation_row = database.continuations[
                    str(unrelated_request.continuation_id)
                ]
                unrelated_continuation_row["ciphertext"] = b"\x00"
                unrelated_continuation_before = dict(
                    unrelated_continuation_row
                )

                await invoke(
                    operation,
                    PgsqlDurableTaskCoordinator(store, task_store),
                    interaction,
                    run_id,
                )

                observed_state = RequestState(
                    cast(
                        str,
                        database.records[
                            str(interaction.record.request.request_id)
                        ]["request_state"],
                    )
                )
                self.assertIs(observed_state, expected_state)
                self.assertEqual(unrelated_row, unrelated_before)
                self.assertEqual(
                    unrelated_continuation_row,
                    unrelated_continuation_before,
                )

            with self.subTest(operation=operation, corruption="targeted"):
                database = FakePgsqlDatabase()
                store = await _store(database)
                task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
                run_id = f"coordinator-targeted-{operation}"
                interaction = await _create_suspended_task(
                    database,
                    store,
                    task_store,
                    run_id=run_id,
                    suffix=run_id,
                )
                target_row = database.records[
                    str(interaction.record.request.request_id)
                ]
                target_row["ciphertext"] = b"\x00"
                target_before = dict(target_row)

                with self.assertRaises(PgsqlInteractionStoreError):
                    await invoke(
                        operation,
                        PgsqlDurableTaskCoordinator(store, task_store),
                        interaction,
                        run_id,
                    )

                self.assertEqual(target_row, target_before)
                self.assertEqual(
                    database.runs[run_id]["state"],
                    TaskRunState.INPUT_REQUIRED.value,
                )
            completed_operations.append((operation, observed_state))

        self.assertEqual(
            completed_operations,
            [
                ("resolve", RequestState.ANSWERED),
                ("cancel", RequestState.CANCELLED),
                ("supersede", RequestState.SUPERSEDED),
                ("trusted_cancel", RequestState.CANCELLED),
            ],
        )

    async def test_branch_delta_preserves_full_ancestry_roots(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("branch-root-delta")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)

        async def register(branch_id: str, parent_branch_id: str) -> None:
            result = await store.register_branch(
                RegisterInteractionBranchCommand(
                    actor=created.command.actor,
                    registration=InteractionBranchRegistration(
                        run_id=request.origin.run_id,
                        branch_id=BranchId(branch_id),
                        parent_branch_id=BranchId(parent_branch_id),
                        principal=request.origin.principal,
                    ),
                )
            )
            self.assertTrue(result.store_mutation_applied)

        await register("child", str(request.origin.branch_id))
        await register("sibling", str(request.origin.branch_id))
        stable_before = {
            key: dict(row) for key, row in database.branches.items()
        }
        await register("grandchild", "child")
        await register("great-grandchild", "grandchild")

        expected_root = str(request.origin.branch_id)
        self.assertEqual(
            {
                cast(str, row["branch_id"]): row["root_branch_id"]
                for row in database.branches.values()
            },
            {
                "child": expected_root,
                "sibling": expected_root,
                "grandchild": expected_root,
                "great-grandchild": expected_root,
            },
        )
        for key, row in stable_before.items():
            self.assertEqual(database.branches[key], row)
        root = await store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=created.command.actor,
                run_id=request.origin.run_id,
                branch_id=BranchId("great-grandchild"),
            )
        )
        self.assertIsNotNone(root)
        assert root is not None
        self.assertEqual(root.root_branch_id, request.origin.branch_id)

        database.branches[
            (
                str(request.origin.run_id),
                "grandchild",
                _scope_identity_digest(
                    request.origin.run_id,
                    request.origin.principal,
                ),
            )
        ]["root_branch_id"] = "tampered-root"
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.lookup_branch_root(
                InteractionBranchRootLookup(
                    actor=created.command.actor,
                    run_id=request.origin.run_id,
                    branch_id=BranchId("great-grandchild"),
                )
            )

    def test_scope_identity_digest_is_domain_and_field_exact(self) -> None:
        baseline = _scope_identity_digest(
            RunId("digest-run"),
            PrincipalScope(),
        )
        variants = {
            _scope_identity_digest(
                RunId("digest-run-other"),
                PrincipalScope(),
            ),
            _scope_identity_digest(
                RunId("digest-run"),
                PrincipalScope(user_id=UserId("identity")),
            ),
            _scope_identity_digest(
                RunId("digest-run"),
                PrincipalScope(tenant_id=TenantId("identity")),
            ),
            _scope_identity_digest(
                RunId("digest-run"),
                PrincipalScope(participant_id=ParticipantId("identity")),
            ),
            _scope_identity_digest(
                RunId("digest-run"),
                PrincipalScope(session_id=SessionId("identity")),
            ),
        }
        self.assertEqual(len(variants), 5)
        self.assertNotIn(baseline, variants)
        self.assertRegex(baseline, r"^[0-9a-f]{64}$")
        self.assertEqual(
            baseline,
            "6c5c90dbca8205e30000de259a0f4ea45047dbd7e16003ea21f3b817ecb2ecf9",
        )

    async def test_fake_upserts_preserve_sql_immutable_columns(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("fake-immutable")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        registered = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=created.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=request.origin.run_id,
                    branch_id=BranchId("immutable-child"),
                    parent_branch_id=request.origin.branch_id,
                    principal=request.origin.principal,
                ),
            )
        )
        self.assertTrue(registered.store_mutation_applied)
        record_before = dict(database.records[str(request.request_id)])
        branch_key = (
            str(request.origin.run_id),
            "immutable-child",
            _scope_identity_digest(
                request.origin.run_id,
                request.origin.principal,
            ),
        )
        branch_before = dict(database.branches[branch_key])
        record_params = (
            request.request_id,
            "changed-continuation",
            "changed-run",
            "changed-turn",
            "changed-task",
            "changed-agent",
            "changed-branch",
            "changed-call",
            record_before["scope_identity_digest"],
            RequestState.ANSWERED.value,
            99,
            99,
            record_before["absolute_expires_at"],
            record_before["retention_deadline_at"],
            b"changed-record",
            "changed-key",
            "changed-algorithm",
            "{}",
            _NOW + timedelta(days=1),
            _NOW + timedelta(days=1),
        )
        branch_params = (
            request.origin.run_id,
            BranchId("immutable-child"),
            BranchId("changed-parent"),
            BranchId("changed-root"),
            99,
            branch_before["scope_identity_digest"],
            b"changed-branch",
            "changed-key",
            "changed-algorithm",
            "{}",
        )
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_UPSERT_RECORD_SQL, record_params)
                self.assertIsNotNone(await cursor.fetchone())
                await cursor.execute(_UPSERT_BRANCH_SQL, branch_params)
                self.assertIsNotNone(await cursor.fetchone())

        record_after = database.records[str(request.request_id)]
        for field_name in (
            "request_id",
            "continuation_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
            "model_call_id",
            "scope_identity_digest",
            "created_at",
        ):
            self.assertEqual(
                record_after[field_name],
                record_before[field_name],
            )
        self.assertEqual(
            record_after["request_state"],
            RequestState.ANSWERED.value,
        )
        branch_after = database.branches[branch_key]
        for field_name in (
            "run_id",
            "branch_id",
            "scope_identity_digest",
        ):
            self.assertEqual(
                branch_after[field_name],
                branch_before[field_name],
            )
        self.assertEqual(branch_after["parent_branch_id"], "changed-parent")

        rejected_params = (
            *record_params[:8],
            "0" * 64,
            *record_params[9:],
        )
        rejected_before = dict(record_after)
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_UPSERT_RECORD_SQL, rejected_params)
                self.assertIsNone(await cursor.fetchone())
        self.assertEqual(
            database.records[str(request.request_id)],
            rejected_before,
        )

    async def test_retention_deletes_branches_by_principal_scope(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(
            database,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        shared_run = RunId("retention-shared-run")

        def scoped_request(suffix: str, owner: str) -> InputRequest:
            request = _request(f"retention-{suffix}")
            return replace(
                request,
                origin=replace(
                    request.origin,
                    run_id=shared_run,
                    branch_id=BranchId(f"root-{suffix}"),
                    principal=PrincipalScope(user_id=UserId(owner)),
                ),
            )

        first_request = scoped_request("first", "first-owner")
        second_request = scoped_request("second", "second-owner")
        first = await store.create(_create_command(first_request))
        second = await store.create(_create_command(second_request))
        self.assertIsInstance(first, CreateInteractionApplied)
        self.assertIsInstance(second, CreateInteractionApplied)
        assert isinstance(first, CreateInteractionApplied)
        assert isinstance(second, CreateInteractionApplied)
        for created, suffix in ((first, "first"), (second, "second")):
            registered = await store.register_branch(
                RegisterInteractionBranchCommand(
                    actor=created.command.actor,
                    registration=InteractionBranchRegistration(
                        run_id=shared_run,
                        branch_id=BranchId(f"child-{suffix}"),
                        parent_branch_id=BranchId(f"root-{suffix}"),
                        principal=created.command.actor.principal,
                    ),
                )
            )
            self.assertTrue(registered.store_mutation_applied)

        first_row = database.records[str(first_request.request_id)]
        first_row["ciphertext"] = b"\x00"
        database.records[str(second_request.request_id)][
            "retention_deadline_at"
        ] = _NOW + timedelta(days=10)

        swept = await store.sweep(now=_NOW + timedelta(days=2))

        self.assertEqual(swept.deleted, ())
        self.assertNotIn(str(first_request.request_id), database.records)
        self.assertIn(str(second_request.request_id), database.records)
        self.assertNotIn(
            (
                str(shared_run),
                "child-first",
                _scope_identity_digest(
                    shared_run,
                    first_request.origin.principal,
                ),
            ),
            database.branches,
        )
        self.assertIn(
            (
                str(shared_run),
                "child-second",
                _scope_identity_digest(
                    shared_run,
                    second_request.origin.principal,
                ),
            ),
            database.branches,
        )
        projection = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=second.command.actor,
                correlation=second.record.correlation,
            )
        )
        self.assertEqual(projection, second.record)

    async def test_scoped_operations_isolate_unrelated_corrupt_rows(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)

        def owned_request(
            name: str,
            owner: str,
            *,
            advisory: bool = False,
        ) -> InputRequest:
            request = _request(name)
            return replace(
                request,
                origin=replace(
                    request.origin,
                    principal=PrincipalScope(user_id=UserId(owner)),
                ),
                mode=(
                    RequirementMode.ADVISORY
                    if advisory
                    else RequirementMode.REQUIRED
                ),
                advisory_wait_seconds=60 if advisory else None,
            )

        lookup_request = owned_request("isolated-lookup", "good-owner")
        cancel_request = owned_request("isolated-cancel", "good-owner")
        activity_request = owned_request(
            "isolated-activity",
            "good-owner",
            advisory=True,
        )
        branch_request = owned_request("isolated-branch", "good-owner")
        corrupt_request = owned_request("isolated-corrupt", "bad-owner")
        corrupt_request = replace(
            corrupt_request,
            origin=replace(
                corrupt_request.origin,
                run_id=lookup_request.origin.run_id,
                branch_id=BranchId("other-root"),
            ),
        )
        created_results = tuple(
            [
                await store.create(_create_command(request))
                for request in (
                    lookup_request,
                    cancel_request,
                    activity_request,
                    branch_request,
                    corrupt_request,
                )
            ]
        )
        self.assertTrue(
            all(
                isinstance(result, CreateInteractionApplied)
                for result in created_results
            )
        )
        (
            lookup_created,
            cancel_created,
            activity_created,
            branch_created,
            corrupt_created,
        ) = cast(tuple[CreateInteractionApplied, ...], created_results)
        corrupt_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=corrupt_created.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=corrupt_request.origin.run_id,
                    branch_id=BranchId("corrupt-child"),
                    parent_branch_id=corrupt_request.origin.branch_id,
                    principal=corrupt_request.origin.principal,
                ),
            )
        )
        self.assertTrue(corrupt_branch.store_mutation_applied)

        corrupt_record_row = database.records[str(corrupt_request.request_id)]
        corrupt_record_row["ciphertext"] = b"\x00"
        corrupt_branch_key = (
            str(corrupt_request.origin.run_id),
            "corrupt-child",
            _scope_identity_digest(
                corrupt_request.origin.run_id,
                corrupt_request.origin.principal,
            ),
        )
        database.branches[corrupt_branch_key]["ciphertext"] = b"\x00"
        unrelated_record_before = (
            corrupt_record_row["ciphertext"],
            corrupt_record_row["state_revision"],
            corrupt_record_row["store_revision"],
            corrupt_record_row["updated_at"],
        )
        unrelated_branch_before = dict(database.branches[corrupt_branch_key])

        self.assertIsNotNone((await store.next_deadline()).deadline)
        projection = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=lookup_created.command.actor,
                correlation=lookup_created.record.correlation,
            )
        )
        self.assertEqual(projection, lookup_created.record)
        resolved = await store.resolve(_answer(lookup_created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        cancelled = await store.cancel(
            CancelInteractionCommand(
                actor=cancel_created.command.actor,
                correlation=cancel_created.record.correlation,
                provenance=AnswerProvenance.HUMAN,
            )
        )
        self.assertTrue(cancelled.store_mutation_applied)
        presented = await store.mark_presented(
            PresentInteractionCommand(
                actor=activity_created.command.actor,
                correlation=activity_created.record.correlation,
                expected_store_revision=(
                    activity_created.record.store_revision
                ),
            )
        )
        self.assertTrue(presented.store_mutation_applied)
        activity = await store.record_activity(
            RecordControllerActivityCommand(
                actor=activity_created.command.actor,
                correlation=activity_created.record.correlation,
                evidence=AcquireControllerActivity(
                    request_id=activity_request.request_id,
                    controller_id=ControllerId("isolated-controller"),
                ),
            )
        )
        self.assertTrue(activity.store_mutation_applied)
        branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=branch_created.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=branch_request.origin.run_id,
                    branch_id=BranchId("good-child"),
                    parent_branch_id=branch_request.origin.branch_id,
                    principal=branch_request.origin.principal,
                ),
            )
        )
        self.assertTrue(branch.store_mutation_applied)

        self.assertEqual(
            (
                corrupt_record_row["ciphertext"],
                corrupt_record_row["state_revision"],
                corrupt_record_row["store_revision"],
                corrupt_record_row["updated_at"],
            ),
            unrelated_record_before,
        )
        self.assertEqual(
            database.branches[corrupt_branch_key],
            unrelated_branch_before,
        )
        denied = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=InteractionActor(
                    principal=PrincipalScope(user_id=UserId("wrong-owner"))
                ),
                correlation=lookup_created.record.correlation,
            )
        )
        self.assertIsNone(denied)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=corrupt_created.command.actor,
                    correlation=corrupt_created.record.correlation,
                )
            )
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.lookup_branch_root(
                InteractionBranchRootLookup(
                    actor=corrupt_created.command.actor,
                    run_id=corrupt_request.origin.run_id,
                    branch_id=BranchId("corrupt-child"),
                )
            )

    async def test_deadline_and_retention_skip_corrupt_candidates(
        self,
    ) -> None:
        for corruption in ("utf8", "json", "decrypt"):
            with self.subTest(corruption=corruption):
                database = FakePgsqlDatabase()
                clock = _Clock()
                cipher = _Cipher()
                store = await _store(
                    database,
                    cipher=cipher,
                    clock=clock,
                    store_policy=PgsqlInteractionStorePolicy(retention_days=1),
                )
                good_request = _request(
                    f"deadline-good-{corruption}",
                    continuation_ttl_seconds=60,
                )
                bad_request = replace(
                    _request(
                        f"deadline-bad-{corruption}",
                        continuation_ttl_seconds=60,
                    ),
                    origin=replace(
                        _origin(f"deadline-bad-{corruption}"),
                        principal=PrincipalScope(
                            user_id=UserId(f"bad-owner-{corruption}")
                        ),
                    ),
                )
                good_created = await store.create_durable(
                    _create_command(good_request),
                    _portable(good_request),
                )
                bad_created = await store.create_durable(
                    _create_command(bad_request),
                    _portable(bad_request),
                )
                self.assertIsInstance(
                    good_created,
                    CreateInteractionApplied,
                )
                self.assertIsInstance(
                    bad_created,
                    CreateInteractionApplied,
                )
                bad_record_row = database.records[str(bad_request.request_id)]
                bad_continuation_row = database.continuations[
                    str(bad_request.continuation_id)
                ]
                if corruption == "utf8":
                    corrupt_ciphertext = b"\x00"
                elif corruption == "json":
                    corrupt_ciphertext = cipher.encrypt(
                        b"{",
                        purpose=TaskKeyPurpose.RAW_VALUE,
                    ).ciphertext
                else:
                    corrupt_ciphertext = b"unavailable"
                bad_continuation_row["ciphertext"] = corrupt_ciphertext
                bad_before = (
                    dict(bad_record_row),
                    dict(bad_continuation_row),
                )
                clock.now += timedelta(seconds=60)
                clock.monotonic += 60

                due = await store.terminalize_due(
                    TerminalizeDueInteractionsCommand(maximum_results=10)
                )
                self.assertIsInstance(due, DueInteractionsApplied)
                assert isinstance(due, DueInteractionsApplied)
                self.assertEqual(
                    tuple(record.request.request_id for record in due.records),
                    (good_request.request_id,),
                )
                self.assertEqual(
                    (
                        bad_record_row,
                        bad_continuation_row,
                    ),
                    bad_before,
                )
                self.assertEqual(
                    database.continuations[str(good_request.continuation_id)][
                        "lifecycle_state"
                    ],
                    DurableContinuationLifecycle.INVALIDATED.value,
                )
                self.assertIsNone((await store.next_deadline()).deadline)

                swept = await store.sweep(now=_NOW + timedelta(days=2))
                self.assertEqual(swept.invalidated, ())
                self.assertEqual(
                    set(swept.deleted),
                    {
                        good_request.continuation_id,
                        bad_request.continuation_id,
                    },
                )
                self.assertEqual(database.records, {})

    async def test_deadline_snapshot_uses_nonlocking_continuation_reads(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        clock = _Clock()
        cipher = _Cipher()
        store = await _store(database, cipher=cipher, clock=clock)
        good_request = _request(
            "deadline-read-good",
            continuation_ttl_seconds=120,
        )
        bad_request = _request(
            "deadline-read-bad",
            continuation_ttl_seconds=60,
        )
        good_created = await store.create_durable(
            _create_command(good_request),
            _portable(good_request),
        )
        bad_created = await store.create_durable(
            _create_command(bad_request),
            _portable(bad_request),
        )
        self.assertIsInstance(good_created, CreateInteractionApplied)
        self.assertIsInstance(bad_created, CreateInteractionApplied)
        bad_continuation = database.continuations[
            str(bad_request.continuation_id)
        ]
        bad_continuation["ciphertext"] = b"\x00"

        database.executed.clear()
        snapshot = await store.next_deadline()

        self.assertIsNotNone(snapshot.deadline)
        assert snapshot.deadline is not None
        self.assertEqual(snapshot.deadline.request_id, good_request.request_id)
        self.assertEqual(snapshot.deadline.monotonic_deadline, 120.0)
        queries = tuple(query for query, _ in database.executed)
        self.assertEqual(
            queries,
            (
                _SET_REPEATABLE_READ_ONLY_SQL,
                _SELECT_STORE_METADATA_SQL,
                _SELECT_RECORDS_SQL,
                _SELECT_BRANCHES_SQL,
                _SELECT_CONTINUATION_BY_REQUEST_SQL,
                _SELECT_CONTINUATION_BY_REQUEST_SQL,
            ),
        )
        self.assertNotIn(
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
            queries,
        )

        clock.now += timedelta(seconds=120)
        clock.monotonic += 120
        database.executed.clear()
        due = await store.terminalize_due(
            TerminalizeDueInteractionsCommand(maximum_results=10)
        )
        self.assertIsInstance(due, DueInteractionsApplied)
        assert isinstance(due, DueInteractionsApplied)
        self.assertEqual(
            tuple(record.request.request_id for record in due.records),
            (good_request.request_id,),
        )
        mutation_queries = tuple(query for query, _ in database.executed)
        self.assertIn(
            _SELECT_CONTINUATION_BY_REQUEST_FOR_UPDATE_SQL,
            mutation_queries,
        )
        self.assertNotIn(
            _SELECT_CONTINUATION_BY_REQUEST_SQL,
            mutation_queries,
        )

    async def test_wait_change_returns_projection_from_deciding_snapshot(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        resolver_store = await _store(database)
        request = _request("wait-snapshot")
        created = await resolver_store.create(_create_command(request))
        assert isinstance(created, CreateInteractionApplied)
        authorizer = _InterleavingAuthorizer()
        wait_store = await _store(
            database,
            authorizer=authorizer,
            store_policy=PgsqlInteractionStorePolicy(
                poll_interval_seconds=0.001,
            ),
        )
        resolutions: list[ResolveInteractionApplied] = []

        async def resolve_during_authorization() -> None:
            result = await resolver_store.resolve(_answer(created))
            assert isinstance(result, ResolveInteractionApplied)
            resolutions.append(result)

        poll_calls = 0

        async def poll_outside_transaction(_delay: float) -> None:
            nonlocal poll_calls
            self.assertFalse(database.lock.locked())
            poll_calls += 1

        authorizer.on_inspect = resolve_during_authorization
        with patch(
            "avalan.interaction.stores.pgsql.sleep",
            new=poll_outside_transaction,
        ):
            projection = await wait_store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=created.command.actor,
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )

        self.assertEqual(len(resolutions), 1)
        self.assertIsInstance(projection, InteractionRecord)
        assert isinstance(projection, InteractionRecord)
        self.assertEqual(projection, resolutions[0].record)
        self.assertIsNotNone(projection.request.resolution)
        self.assertGreater(
            projection.store_revision,
            created.record.store_revision,
        )
        self.assertEqual(poll_calls, 1)

    async def test_wait_change_cancellation_and_close_release_snapshot(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("wait-cancel")
        created = await store.create(_create_command(request))
        assert isinstance(created, CreateInteractionApplied)
        poll_started = Event()

        async def blocked_poll(_delay: float) -> None:
            self.assertFalse(database.lock.locked())
            poll_started.set()
            await Event().wait()

        with patch(
            "avalan.interaction.stores.pgsql.sleep",
            new=blocked_poll,
        ):
            waiting = create_task(
                store.wait_for_change(
                    WaitForInteractionChangeCommand(
                        actor=created.command.actor,
                        correlation=created.record.correlation,
                        after_store_revision=created.record.store_revision,
                    )
                )
            )
            await poll_started.wait()
            self.assertTrue(waiting.cancel())
            with self.assertRaises(CancelledError):
                await waiting

        self.assertFalse(database.lock.locked())
        await store.aclose()
        with self.assertRaises(InteractionStoreClosedError):
            await store.wait_for_change(
                WaitForInteractionChangeCommand(
                    actor=created.command.actor,
                    correlation=created.record.correlation,
                    after_store_revision=created.record.store_revision,
                )
            )

    async def test_task_bound_public_mutations_require_coordinator(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        cipher = _Cipher()
        store = await _store(
            database,
            cipher=cipher,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        request = _request("public-task")
        request = replace(
            request,
            origin=replace(
                request.origin,
                task_id=TaskId("task-public"),
            ),
        )
        command = _create_command(request)
        continuation = _portable(request)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.create(command)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.create_durable(
                command,
                continuation,
                task_run_id="public-task",
                checkpoint_id="checkpoint-public",
            )
        self.assertEqual(database.records, {})

        _seed_running_task(database, "public-task")
        task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        suspended = await coordinator.create_and_suspend(
            command,
            continuation,
            queue_item_id="queue-item",
            claim_token="claim-token",
            segment_id="segment",
            task_run_id="public-task",
            checkpoint_id="checkpoint-public",
        )
        durable = await store.get_continuation(request.continuation_id)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.invalidate(
                request.continuation_id,
                expected_store_revision=durable.store_revision,
                reason="standalone mutation",
                now=_NOW + timedelta(seconds=1),
            )
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.sweep(now=_NOW + timedelta(minutes=11))
        self.assertEqual(
            database.records[str(request.request_id)]["request_state"],
            RequestState.PENDING.value,
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.PENDING.value,
        )
        registered = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=command.actor,
                registration=InteractionBranchRegistration(
                    run_id=request.origin.run_id,
                    branch_id=BranchId("public-child"),
                    parent_branch_id=request.origin.branch_id,
                    principal=request.origin.principal,
                ),
            )
        )
        self.assertTrue(registered.store_mutation_applied)

        await coordinator.cancel_input_required_task(
            task_run_id="public-task",
            now=_NOW + timedelta(seconds=2),
            metadata={},
        )
        self.assertEqual(
            suspended.interaction.record.request.request_id,
            request.request_id,
        )
        self.assertTrue(database.branches)
        branch_row = next(iter(database.branches.values()))
        branch_plaintext = cipher.decrypt(
            EncryptedPrivacyValue(
                ciphertext=cast(bytes, branch_row["ciphertext"]),
                key_id=cast(str, branch_row["encryption_key_id"]),
                algorithm=cast(str, branch_row["encryption_algorithm"]),
                metadata=cast(
                    Mapping[str, str],
                    branch_row["encryption_metadata"],
                ),
            ),
            purpose=TaskKeyPurpose.RAW_VALUE,
            context=cast(
                Mapping[str, str],
                branch_row["encryption_metadata"],
            ),
        )
        self.assertIn(b"owner", branch_plaintext)
        swept = await coordinator.sweep_retention(now=_NOW + timedelta(days=2))
        self.assertEqual(swept.deleted, (request.continuation_id,))
        self.assertEqual(database.records, {})
        self.assertEqual(database.continuations, {})
        self.assertEqual(database.branches, {})

    async def test_exact_request_and_continuation_deadline_is_enforced(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("deadline")
        continuation = _portable(request)
        mismatched = replace(
            continuation,
            expires_at=continuation.expires_at + timedelta(seconds=1),
        )
        with self.assertRaises(InputValidationError):
            DurableInteractionSuspension(
                command=_create_command(request),
                continuation=mismatched,
            )
        with self.assertRaises(InputValidationError):
            await store.create_durable(
                _create_command(request),
                mismatched,
            )
        _seed_running_task(database, "deadline")
        with self.assertRaises(InputValidationError):
            await PgsqlDurableTaskCoordinator(
                store,
                PgsqlTaskStore(database, clock=lambda: _NOW),
            ).create_and_suspend(
                _create_command(request),
                mismatched,
                queue_item_id="queue-item",
                claim_token="claim-token",
                segment_id="segment",
                task_run_id="deadline",
                checkpoint_id="checkpoint",
            )
        self.assertEqual(database.records, {})

        created = await store.create_durable(
            _create_command(request),
            continuation,
        )
        assert isinstance(created, CreateInteractionApplied)
        await store.resolve(_answer(created))
        ready = await store.get_continuation(request.continuation_id)
        record_row = database.records[str(request.request_id)]
        record_row["absolute_expires_at"] = _NOW + timedelta(minutes=5)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("deadline-owner"),
                lease_expires_at=_NOW + timedelta(minutes=7),
                dispatch_id=ContinuationDispatchId("deadline-dispatch"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=_NOW + timedelta(minutes=6),
            )
        record_row["absolute_expires_at"] = continuation.expires_at
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("expired-owner"),
                lease_expires_at=_NOW + timedelta(minutes=12),
                dispatch_id=ContinuationDispatchId("expired-dispatch"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=_NOW + timedelta(minutes=11),
            )

    async def test_process_local_bindings_publish_only_after_commit(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("runtime-sql")
        resumer = _Resumer()
        admission, _ = _new_interaction_admission_commands(
            actor=_create_command(request).actor,
            request=request,
            resumer=resumer,
        )
        database.fail_query = 'UPDATE "interaction_store_metadata"'
        with self.assertRaises(Exception):
            await store.create_admission(admission)
        runtime = cast(Any, store)._runtime
        self.assertEqual(runtime.resumers, {})
        self.assertEqual(runtime.admissions, {})
        self.assertEqual(database.records, {})

        database.fail_query = None
        durable_request = _request("runtime-after-persist")
        durable_resumer = _Resumer()
        durable_command = replace(
            _create_command(durable_request),
            resumer=durable_resumer,
        )
        database.fail_query = 'INSERT INTO "interaction_continuations"'
        with self.assertRaises(Exception):
            await store.create_durable(
                durable_command,
                _portable(durable_request),
            )
        self.assertEqual(runtime.resumers, {})
        self.assertEqual(runtime.admissions, {})
        self.assertEqual(database.records, {})

    async def test_admission_cleanup_loads_only_its_bound_record(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)

        baseline_request = _request("cleanup-baseline")
        baseline_resumer = _Resumer()
        baseline_admission, baseline_cleanup = (
            _new_interaction_admission_commands(
                actor=_create_command(baseline_request).actor,
                request=baseline_request,
                resumer=baseline_resumer,
            )
        )
        baseline_created = await store.create_admission(baseline_admission)
        self.assertIsInstance(baseline_created, CreateInteractionApplied)
        baseline = await store.cleanup_admission(baseline_cleanup)
        self.assertIs(
            baseline.disposition,
            _InteractionAdmissionCleanupDisposition.SETTLED,
        )

        unrelated_request = _request("cleanup-malformed-unrelated")
        unrelated = await store.create(
            _create_command(unrelated_request),
        )
        self.assertIsInstance(unrelated, CreateInteractionApplied)
        unrelated_row = database.records[str(unrelated_request.request_id)]
        unrelated_row["ciphertext"] = b"\x00"
        unrelated_before = dict(unrelated_row)

        target_request = _request("cleanup-target")
        target_resumer = _Resumer()
        target_admission, target_cleanup = _new_interaction_admission_commands(
            actor=_create_command(target_request).actor,
            request=target_request,
            resumer=target_resumer,
        )
        target_created = await store.create_admission(target_admission)
        self.assertIsInstance(target_created, CreateInteractionApplied)

        database.executed.clear()
        scoped = await store.cleanup_admission(target_cleanup)

        self.assertEqual(scoped, baseline)
        self.assertEqual(unrelated_row, unrelated_before)
        self.assertEqual(len(target_resumer.notifications), 1)
        self.assertEqual(
            tuple(
                parameters
                for query, parameters in database.executed
                if query == _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL
            ),
            (
                (
                    target_request.request_id,
                    target_request.continuation_id,
                ),
            ),
        )
        queries = tuple(query for query, _ in database.executed)
        self.assertNotIn(_SELECT_RECORDS_SQL, queries)
        self.assertNotIn(_SELECT_BRANCHES_SQL, queries)

        wrong_request = _request("cleanup-wrong-capability")
        _, wrong_cleanup = _new_interaction_admission_commands(
            actor=_create_command(wrong_request).actor,
            request=wrong_request,
            resumer=_Resumer(),
        )
        database.executed.clear()
        wrong = await store.cleanup_admission(wrong_cleanup)
        self.assertIs(
            wrong.disposition,
            _InteractionAdmissionCleanupDisposition.ABSENT,
        )
        wrong_queries = tuple(query for query, _ in database.executed)
        self.assertNotIn(
            _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
            wrong_queries,
        )
        self.assertNotIn(_SELECT_RECORDS_SQL, wrong_queries)
        self.assertNotIn(_SELECT_BRANCHES_SQL, wrong_queries)

        stale_request = _request("cleanup-stale-capability")
        stale_admission, stale_cleanup = _new_interaction_admission_commands(
            actor=_create_command(stale_request).actor,
            request=stale_request,
            resumer=_Resumer(),
        )
        stale_created = await store.create_admission(stale_admission)
        self.assertIsInstance(stale_created, CreateInteractionApplied)
        database.records.pop(str(stale_request.request_id))
        database.executed.clear()
        stale = await store.cleanup_admission(stale_cleanup)
        self.assertIs(
            stale.disposition,
            _InteractionAdmissionCleanupDisposition.ABSENT,
        )
        stale_queries = tuple(query for query, _ in database.executed)
        self.assertIn(
            _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
            stale_queries,
        )
        self.assertNotIn(_SELECT_RECORDS_SQL, stale_queries)
        self.assertNotIn(_SELECT_BRANCHES_SQL, stale_queries)

        database.executed.clear()
        repeated = await store.cleanup_admission(stale_cleanup)
        self.assertIs(
            repeated.disposition,
            _InteractionAdmissionCleanupDisposition.ABSENT,
        )
        repeated_queries = tuple(query for query, _ in database.executed)
        self.assertNotIn(
            _SELECT_ADMISSION_RECORD_FOR_UPDATE_SQL,
            repeated_queries,
        )

    async def test_admission_cleanup_publishes_only_after_commit(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("cleanup-rollback")
        resumer = _Resumer()
        admission, cleanup = _new_interaction_admission_commands(
            actor=_create_command(request).actor,
            request=request,
            resumer=resumer,
        )
        created = await store.create_admission(admission)
        self.assertIsInstance(created, CreateInteractionApplied)
        before = dict(database.records[str(request.request_id)])

        database.fail_query = 'UPDATE "interaction_store_metadata"'
        with self.assertRaises(PgsqlOperationError):
            await store.cleanup_admission(cleanup)

        self.assertEqual(database.records[str(request.request_id)], before)
        self.assertEqual(resumer.notifications, [])

        database.fail_query = None
        settled = await store.cleanup_admission(cleanup)
        self.assertIs(
            settled.disposition,
            _InteractionAdmissionCleanupDisposition.SETTLED,
        )
        self.assertEqual(len(resumer.notifications), 1)

    async def test_durable_continuation_claim_fencing_and_ambiguity(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request()
        continuation = _portable(request)

        created = await store.create_durable(
            _create_command(request),
            continuation,
        )
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        self.assertNotIn(
            request.reason.encode(),
            cast(
                bytes, database.records[str(request.request_id)]["ciphertext"]
            ),
        )
        resolved = await store.resolve(_answer(created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        ready = await store.get_continuation(request.continuation_id)

        claims = await gather(
            store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-a"),
                lease_expires_at=_NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-a"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=_NOW + timedelta(seconds=2),
            ),
            store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-b"),
                lease_expires_at=_NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-b"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=_NOW + timedelta(seconds=2),
            ),
            return_exceptions=True,
        )
        receipts = [
            value for value in claims if not isinstance(value, Exception)
        ]
        failures = [value for value in claims if isinstance(value, Exception)]
        self.assertEqual(len(receipts), 1, claims)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], ContinuationStoreConflictError)
        receipt = receipts[0]
        assert not isinstance(receipt, BaseException)
        claim_revision = receipt.continuation.store_revision
        renewed = await store.renew_claim(
            request.continuation_id,
            expected_store_revision=claim_revision,
            owner_id=cast(
                ContinuationClaimOwnerId,
                receipt.continuation.claim.owner_id,
            ),
            fencing_token=receipt.fencing_token,
            lease_expires_at=_NOW + timedelta(minutes=5),
            now=_NOW + timedelta(minutes=1),
        )
        self.assertTrue(renewed)
        renewed_continuation = await store.get_continuation(
            request.continuation_id
        )
        self.assertEqual(
            renewed_continuation.claim.lease_expires_at,
            _NOW + timedelta(minutes=5),
        )
        self.assertEqual(
            renewed_continuation.store_revision,
            claim_revision,
        )

        dispatching = await store.mark_dispatching(
            request.continuation_id,
            expected_store_revision=claim_revision,
            owner_id=cast(
                ContinuationClaimOwnerId,
                receipt.continuation.claim.owner_id,
            ),
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(minutes=3),
        )
        self.assertFalse(
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=cast(
                    ContinuationClaimOwnerId,
                    receipt.continuation.claim.owner_id,
                ),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=6),
                now=_NOW + timedelta(minutes=3, seconds=10),
            )
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=ContinuationClaimOwnerId("stale-worker"),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=6),
                now=_NOW + timedelta(minutes=3, seconds=10),
            )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=ContinuationStoreRevision(
                    int(claim_revision) - 1
                ),
                owner_id=cast(
                    ContinuationClaimOwnerId,
                    receipt.continuation.claim.owner_id,
                ),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=6),
                now=_NOW + timedelta(minutes=3, seconds=10),
            )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=ContinuationClaimOwnerId("stale-worker"),
                fencing_token=ContinuationFencingToken(
                    int(receipt.fencing_token) + 1
                ),
                lease_expires_at=_NOW + timedelta(minutes=6),
                now=_NOW + timedelta(minutes=3, seconds=10),
            )
        with self.assertRaises(ContinuationDispatchAmbiguousError):
            await store.release(
                request.continuation_id,
                expected_store_revision=dispatching.store_revision,
                owner_id=cast(
                    ContinuationClaimOwnerId,
                    dispatching.claim.owner_id,
                ),
                fencing_token=dispatching.fencing_token,
                now=_NOW + timedelta(minutes=3, seconds=20),
            )
        dispatched = await store.mark_dispatched(
            request.continuation_id,
            expected_store_revision=dispatching.store_revision,
            owner_id=cast(
                ContinuationClaimOwnerId,
                dispatching.claim.owner_id,
            ),
            fencing_token=dispatching.fencing_token,
            now=_NOW + timedelta(minutes=4),
        )
        completed = await store.complete(
            request.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=cast(
                ContinuationClaimOwnerId,
                dispatched.claim.owner_id,
            ),
            fencing_token=dispatched.fencing_token,
            result_digest="a" * 64,
            now=_NOW + timedelta(minutes=5),
        )
        assert completed.completion is not None
        self.assertEqual(completed.completion.result_digest, "a" * 64)
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.COMPLETED.value,
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "claim_owner_id"
            ],
            receipt.continuation.claim.owner_id,
        )
        self.assertFalse(
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=cast(
                    ContinuationClaimOwnerId,
                    receipt.continuation.claim.owner_id,
                ),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=9),
                now=_NOW + timedelta(minutes=6),
            )
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=ContinuationClaimOwnerId("wrong-owner"),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=9),
                now=_NOW + timedelta(minutes=6),
            )

    async def test_released_renewal_requires_prior_claim_owner(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request()
        created = await store.create_durable(
            _create_command(request),
            _portable(request),
        )
        assert isinstance(created, CreateInteractionApplied)
        await store.resolve(_answer(created))
        ready = await store.get_continuation(request.continuation_id)
        owner_id = ContinuationClaimOwnerId("release-owner")
        receipt = await store.claim(
            request.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=owner_id,
            lease_expires_at=_NOW + timedelta(minutes=2),
            dispatch_id=ContinuationDispatchId("release-dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=2),
        )
        claim_revision = receipt.continuation.store_revision
        released = await store.release(
            request.continuation_id,
            expected_store_revision=claim_revision,
            owner_id=owner_id,
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(minutes=1),
        )

        self.assertEqual(
            released.claim.state,
            ContinuationClaimState.FAILED_SAFE_TO_RETRY,
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "claim_owner_id"
            ],
            owner_id,
        )
        self.assertFalse(
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=owner_id,
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=3),
                now=_NOW + timedelta(minutes=1, seconds=1),
            )
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.renew_claim(
                request.continuation_id,
                expected_store_revision=claim_revision,
                owner_id=ContinuationClaimOwnerId("wrong-owner"),
                fencing_token=receipt.fencing_token,
                lease_expires_at=_NOW + timedelta(minutes=3),
                now=_NOW + timedelta(minutes=1, seconds=1),
            )

    async def test_public_complete_rejects_task_bound_continuation(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, _, dispatched = await _prepare_resumed_dispatch(database)

        with self.assertRaises(PgsqlInteractionStoreError):
            await store.complete(
                dispatched.continuation_id,
                expected_store_revision=dispatched.store_revision,
                owner_id=ContinuationClaimOwnerId("resumed-worker"),
                fencing_token=dispatched.fencing_token,
                result_digest="a" * 64,
                now=_NOW + timedelta(seconds=5),
            )

        self.assertEqual(
            await store.get_continuation(dispatched.continuation_id),
            dispatched,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.RUNNING.value,
        )
        self.assertEqual(
            database.queue_items["queue-item"]["state"],
            TaskQueueItemState.CLAIMED.value,
        )

    async def test_atomic_coordinator_rolls_back_and_requeues(self) -> None:
        database = FakePgsqlDatabase()
        _seed_running_task(database, "run")
        interaction_store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        )
        request = _request()
        continuation = _portable(request)

        database.fail_query = 'UPDATE "task_attempts"'
        with self.assertRaises(Exception):
            await coordinator.create_and_suspend(
                _create_command(request),
                continuation,
                queue_item_id="queue-item",
                claim_token="claim-token",
                segment_id="segment",
                task_run_id="run",
                checkpoint_id="checkpoint",
            )
        self.assertEqual(database.records, {})
        self.assertEqual(database.continuations, {})
        self.assertEqual(database.events, {})
        self.assertEqual(database.runs["run"]["state"], "running")
        self.assertEqual(
            database.queue_items["queue-item"]["state"], "claimed"
        )

        database.fail_query = 'INSERT INTO "task_events"'
        with self.assertRaises(Exception):
            await coordinator.create_and_suspend(
                _create_command(request),
                continuation,
                queue_item_id="queue-item",
                claim_token="claim-token",
                segment_id="segment",
                task_run_id="run",
                checkpoint_id="checkpoint",
            )
        self.assertEqual(database.records, {})
        self.assertEqual(database.continuations, {})
        self.assertEqual(database.events, {})
        self.assertEqual(database.runs["run"]["state"], "running")

        database.fail_query = None
        suspended = await coordinator.create_and_suspend(
            _create_command(request),
            continuation,
            queue_item_id="queue-item",
            claim_token="claim-token",
            segment_id="segment",
            task_run_id="run",
            checkpoint_id="checkpoint",
        )
        self.assertEqual(
            suspended.suspension.run.state,
            TaskRunState.INPUT_REQUIRED,
        )
        self.assertIsNone(
            suspended.suspension.queue_item.claim_token,
        )
        durable_record = await interaction_store.get_continuation_record(
            request.continuation_id
        )
        task_record = await interaction_store.get_task_continuation_record(
            "run"
        )
        self.assertEqual(durable_record, task_record)
        self.assertEqual(durable_record.task_run_id, "run")
        self.assertEqual(durable_record.checkpoint_id, "checkpoint")
        self.assertEqual(
            suspended.suspension.run.result,
            suspended.suspension.attempt.result,
        )
        assert suspended.suspension.run.result is not None
        self.assertEqual(
            suspended.suspension.run.result.metadata["interaction"],
            {
                "kind": "input_required",
                "request_id": str(request.request_id),
                "continuation_id": str(request.continuation_id),
                "checkpoint_id": "checkpoint",
                "detached_resumption_available": True,
            },
        )
        self.assertEqual(
            suspended.suspension.segment.checkpoint_id,
            "checkpoint",
        )
        self.assertEqual(
            [row["event_type"] for row in database.events.values()],
            [TaskInteractionEventType.INPUT_REQUIRED.value],
        )

        database.fail_query = 'UPDATE "task_queue_items"\nSET\n    "state"'
        with self.assertRaises(Exception):
            await coordinator.resolve_and_requeue(
                _answer(suspended.interaction),
                task_run_id="run",
            )
        self.assertEqual(
            database.records[str(request.request_id)]["request_state"],
            "pending",
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "lifecycle_state"
            ],
            "pending",
        )
        self.assertEqual(database.outbox, {})
        self.assertEqual(database.runs["run"]["state"], "input_required")

        database.fail_query = None
        reentry = await coordinator.resolve_and_requeue(
            _answer(suspended.interaction),
            task_run_id="run",
        )
        self.assertEqual(reentry.reentry.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            reentry.reentry.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertIsNone(reentry.reentry.run.result)
        self.assertIsNone(reentry.reentry.attempt.result)
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"], "pending"
        )
        self.assertEqual(
            [row["event_type"] for row in database.events.values()],
            [
                TaskInteractionEventType.INPUT_REQUIRED.value,
                TaskInteractionEventType.INPUT_RESUMED.value,
            ],
        )

    async def test_atomic_completion_and_successor_suspension(self) -> None:
        database = FakePgsqlDatabase()
        (
            store,
            task_store,
            interaction,
        ) = await _commit_resolution_before_task_reentry(database)
        previous_request = interaction.record.request
        ready = await store.get_continuation(previous_request.continuation_id)
        receipt = await store.claim(
            ready.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            lease_expires_at=_NOW + timedelta(minutes=2),
            dispatch_id=ContinuationDispatchId("resumed-dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=2),
        )
        dispatching = await store.mark_dispatching(
            ready.continuation_id,
            expected_store_revision=receipt.continuation.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(seconds=3),
        )
        dispatched = await store.mark_dispatched(
            ready.continuation_id,
            expected_store_revision=dispatching.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(seconds=4),
        )
        _seed_resumed_running_task(
            database,
            run_id="run",
            segment_id="segment-next",
        )
        successor_request = replace(
            _request(),
            request_id=InputRequestId("request-next"),
            continuation_id=ContinuationId("continuation-next"),
        )
        base_successor = _portable(successor_request)
        assert base_successor.provider_snapshot is not None
        successor = replace(
            base_successor,
            provider_call_correlation_id="input-call-next",
            provider_snapshot=replace(
                base_successor.provider_snapshot,
                payload={
                    "reserved_capability_call_id": "input-call-next",
                    "replay_items": (
                        {
                            "id": "reasoning-item-next",
                            "type": "reasoning",
                            "encrypted_content": "provider-ciphertext-next",
                        },
                    ),
                },
            ),
            interaction_count=dispatched.interaction_count + 1,
            stream_sequence=dispatched.stream_sequence + 1,
        )
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=receipt.fencing_token,
            result_digest="b" * 64,
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)

        database.fail_query = 'UPDATE "task_runs"\nSET\n    "state"'
        with self.assertRaises(Exception):
            await coordinator.complete_and_resuspend(
                completion,
                _create_command(successor_request),
                successor,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                checkpoint_id="checkpoint-next",
                now=_NOW + timedelta(seconds=5),
            )
        unchanged = await store.get_continuation(dispatched.continuation_id)
        self.assertEqual(unchanged, dispatched)
        with self.assertRaises(InteractionNotFoundError):
            await store.get_continuation(successor.continuation_id)

        database.fail_query = None
        result = await coordinator.complete_and_resuspend(
            completion,
            _create_command(successor_request),
            successor,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            checkpoint_id="checkpoint-next",
            now=_NOW + timedelta(seconds=5),
        )

        self.assertEqual(
            result.completed_continuation.claim.state,
            ContinuationClaimState.COMPLETED,
        )
        assert result.completed_continuation.completion is not None
        self.assertEqual(
            result.completed_continuation.completion.result_digest,
            "b" * 64,
        )
        self.assertEqual(
            result.suspension.run.state,
            TaskRunState.INPUT_REQUIRED,
        )
        self.assertEqual(
            result.suspension.segment.checkpoint_id,
            "checkpoint-next",
        )
        stored_successor = await store.get_task_continuation_record("run")
        self.assertEqual(stored_successor.continuation, successor)
        self.assertEqual(stored_successor.checkpoint_id, "checkpoint-next")
        self.assertEqual(
            [row["event_type"] for row in database.events.values()],
            [
                TaskInteractionEventType.INPUT_REQUIRED.value,
                TaskInteractionEventType.INPUT_RESUMED.value,
                TaskInteractionEventType.INPUT_REQUIRED.value,
            ],
        )

    async def test_cancel_requested_settlement_matrix_converges_atomically(
        self,
    ) -> None:
        operations = (
            "success",
            "failure",
            "cancellation",
            "ambiguous",
            "completed_provider_failure",
            "completed_provider_cancellation",
        )
        for operation in operations:
            with self.subTest(operation=operation):
                prepared = await _prepare_cancel_requested_settlement_case(
                    operation
                )
                database = prepared.database
                baseline = database.snapshot()
                baseline_attempts = database.queue_items["queue-item"][
                    "attempts"
                ]
                baseline_run_transition_ids = set(database.run_transitions)
                baseline_attempt_transition_ids = set(
                    database.attempt_transitions
                )
                baseline_segment_transition_ids = set(
                    database.segment_transitions
                )
                baseline_event_ids = set(database.events)
                continuation_id = str(prepared.completion.continuation_id)
                provider_evidence = dict(
                    database.continuations[continuation_id]
                )

                database.executed.clear()
                await _invoke_cancel_requested_settlement_case(
                    prepared,
                    completion=prepared.completion,
                    settlement=prepared.settlement,
                    claim_token="resumed-claim-token",
                )
                query_count = len(database.executed)
                self.assertGreater(query_count, 0)
                database.restore(baseline)

                for fail_after_query in range(1, query_count + 1):
                    database.fail_after_queries = fail_after_query
                    with self.assertRaises(RuntimeError):
                        await _invoke_cancel_requested_settlement_case(
                            prepared,
                            completion=prepared.completion,
                            settlement=prepared.settlement,
                            claim_token="resumed-claim-token",
                        )
                    self.assertEqual(database.snapshot(), baseline)

                settled = await _invoke_cancel_requested_settlement_case(
                    prepared,
                    completion=prepared.completion,
                    settlement=prepared.settlement,
                    claim_token="resumed-claim-token",
                )
                expected_result = TaskExecutionResult(
                    error=freeze_snapshot_value(
                        TaskError.cancellation().as_dict()
                    ),
                    metadata={
                        "superseded_settlement_digest": (
                            task_durable_resume_settlement_digest(
                                prepared.settlement
                            )
                        )
                    },
                )
                self.assertEqual(settled.run.state, TaskRunState.CANCELLED)
                self.assertEqual(
                    settled.attempt.state,
                    TaskAttemptState.ABANDONED,
                )
                self.assertEqual(
                    settled.queue_item.state,
                    TaskQueueItemState.DEAD,
                )
                self.assertEqual(settled.run.result, expected_result)
                self.assertEqual(settled.attempt.result, expected_result)
                self.assertEqual(
                    settled.queue_item.attempts,
                    baseline_attempts,
                )
                self.assertIsNone(settled.run.claim)
                self.assertIsNone(settled.queue_item.claim_token)
                self.assertIsNone(settled.queue_item.lease_expires_at)
                self.assertEqual(
                    database.segments["segment-next"]["state"],
                    TaskAttemptSegmentState.ABANDONED.value,
                )
                self.assertEqual(
                    {row["status"] for row in database.outbox.values()},
                    {"dead"},
                )
                self.assertEqual(
                    [
                        (row["from_state"], row["to_state"])
                        for transition_id, row in (
                            database.run_transitions.items()
                        )
                        if transition_id not in baseline_run_transition_ids
                    ],
                    [
                        (
                            TaskRunState.CANCEL_REQUESTED.value,
                            TaskRunState.CANCELLED.value,
                        )
                    ],
                )
                self.assertEqual(
                    [
                        (row["from_state"], row["to_state"])
                        for transition_id, row in (
                            database.attempt_transitions.items()
                        )
                        if transition_id not in baseline_attempt_transition_ids
                    ],
                    [
                        (
                            TaskAttemptState.RUNNING.value,
                            TaskAttemptState.ABANDONED.value,
                        )
                    ],
                )
                self.assertEqual(
                    [
                        (row["from_state"], row["to_state"])
                        for transition_id, row in (
                            database.segment_transitions.items()
                        )
                        if transition_id not in baseline_segment_transition_ids
                    ],
                    [
                        (
                            TaskAttemptSegmentState.RUNNING.value,
                            TaskAttemptSegmentState.ABANDONED.value,
                        )
                    ],
                )
                self.assertEqual(set(database.events), baseline_event_ids)
                if operation == "ambiguous" or operation.startswith(
                    "completed_provider_"
                ):
                    self.assertEqual(
                        database.continuations[continuation_id],
                        provider_evidence,
                    )
                else:
                    self.assertEqual(
                        database.continuations[continuation_id][
                            "lifecycle_state"
                        ],
                        DurableContinuationLifecycle.COMPLETED.value,
                    )

                settled_snapshot = database.snapshot()
                replayed = await _invoke_cancel_requested_settlement_case(
                    prepared,
                    completion=prepared.completion,
                    settlement=prepared.settlement,
                    claim_token="resumed-claim-token",
                )
                self.assertEqual(replayed, settled)
                self.assertEqual(database.snapshot(), settled_snapshot)

                with self.assertRaises(
                    (
                        InputValidationError,
                        TaskStoreConflictError,
                    )
                ):
                    await _invoke_cancel_requested_settlement_case(
                        prepared,
                        completion=prepared.completion,
                        settlement=prepared.settlement,
                        claim_token="wrong-claim-token",
                    )
                with self.assertRaises(ContinuationStoreConflictError):
                    await _invoke_cancel_requested_settlement_case(
                        prepared,
                        completion=replace(
                            prepared.completion,
                            fencing_token=ContinuationFencingToken(
                                int(prepared.completion.fencing_token) + 1
                            ),
                        ),
                        settlement=prepared.settlement,
                        claim_token="resumed-claim-token",
                    )
                conflicting = TaskDurableResumeFailure(
                    result=TaskExecutionResult(
                        error={"code": "conflicting_settlement"}
                    )
                )
                conflicting_completion = (
                    prepared.completion
                    if operation.startswith("completed_provider_")
                    else replace(
                        prepared.completion,
                        result_digest=(
                            task_durable_resume_settlement_digest(conflicting)
                        ),
                    )
                )
                with self.assertRaises(
                    (
                        ContinuationStoreConflictError,
                        TaskStoreConflictError,
                    )
                ):
                    await _invoke_cancel_requested_settlement_case(
                        prepared,
                        completion=conflicting_completion,
                        settlement=conflicting,
                        claim_token="resumed-claim-token",
                    )

    async def test_cancel_requested_expired_reentry_replays_after_restart(
        self,
    ) -> None:
        prepared = await _prepare_cancel_requested_settlement_case("failure")
        database = prepared.database
        _expire_reentry_claim(
            database,
            expires_at=_NOW + timedelta(seconds=5),
        )
        continuation_id = str(prepared.completion.continuation_id)
        provider_evidence = dict(database.continuations[continuation_id])
        failure = TaskExecutionResult(
            error={"code": "expired_durable_reentry_claim"}
        )

        recovered = await prepared.coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token="resumed-claim-token",
            task_run_id="run",
            result=failure,
            now=_NOW + timedelta(seconds=6),
            metadata={"source": "lease_sweep"},
        )

        assert recovered.completion is not None
        self.assertEqual(
            recovered.completion.run.state,
            TaskRunState.CANCELLED,
        )
        self.assertEqual(
            recovered.completion.attempt.state,
            TaskAttemptState.ABANDONED,
        )
        self.assertEqual(
            recovered.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.ABANDONED.value,
        )
        self.assertEqual(
            database.continuations[continuation_id],
            provider_evidence,
        )
        self.assertEqual(
            {row["status"] for row in database.outbox.values()},
            {"dead"},
        )
        committed = database.snapshot()

        restarted_store = await _store(database)
        restarted = PgsqlDurableTaskCoordinator(
            restarted_store,
            PgsqlTaskStore(database, clock=lambda: _NOW),
        )
        replayed = await restarted.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token="resumed-claim-token",
            task_run_id="run",
            result=failure,
            now=_NOW + timedelta(seconds=7),
            metadata={"source": "restarted_lease_sweep"},
        )
        self.assertEqual(replayed, recovered)
        self.assertEqual(database.snapshot(), committed)
        with self.assertRaises(TaskStoreConflictError):
            await restarted.reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token="wrong-claim-token",
                task_run_id="run",
                result=failure,
                now=_NOW + timedelta(seconds=7),
            )

    async def test_cancelled_resumed_startup_boundaries_converge_atomically(
        self,
    ) -> None:
        boundaries = (
            "run_running",
            "attempt_running",
            "segment_created",
            "segment_running",
        )
        failure = TaskExecutionResult(
            error={"code": "expired_durable_reentry_claim"}
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                (
                    database,
                    store,
                    task_store,
                    request,
                    claim,
                    claimed_continuation,
                    active_segment_id,
                ) = await _prepare_cancel_requested_startup_boundary(boundary)
                coordinator = PgsqlDurableTaskCoordinator(store, task_store)
                baseline = database.snapshot()
                baseline_events = set(database.events)
                baseline_segment_transitions = set(
                    database.segment_transitions
                )
                previous_evidence = dict(database.segments["segment"])

                database.executed.clear()
                await coordinator.reconcile_expired_reentry(
                    queue_item_id="queue-item",
                    expected_claim_token=claim.claim_token,
                    task_run_id="run",
                    result=failure,
                    now=_NOW + timedelta(seconds=6),
                    metadata={"source": "startup_boundary_sweep"},
                )
                query_count = len(database.executed)
                self.assertGreater(query_count, 0)
                database.restore(baseline)

                for fail_after_query in range(1, query_count + 1):
                    database.fail_after_queries = fail_after_query
                    with self.assertRaises(RuntimeError):
                        await coordinator.reconcile_expired_reentry(
                            queue_item_id="queue-item",
                            expected_claim_token=claim.claim_token,
                            task_run_id="run",
                            result=failure,
                            now=_NOW + timedelta(seconds=6),
                            metadata={"source": "startup_boundary_sweep"},
                        )
                    self.assertEqual(database.snapshot(), baseline)

                recovered = await coordinator.reconcile_expired_reentry(
                    queue_item_id="queue-item",
                    expected_claim_token=claim.claim_token,
                    task_run_id="run",
                    result=failure,
                    now=_NOW + timedelta(seconds=6),
                    metadata={"source": "startup_boundary_sweep"},
                )
                assert recovered.completion is not None
                completion = recovered.completion
                self.assertEqual(
                    completion.run.state,
                    TaskRunState.CANCELLED,
                )
                self.assertEqual(
                    completion.attempt.state,
                    TaskAttemptState.ABANDONED,
                )
                self.assertEqual(
                    completion.queue_item.state,
                    TaskQueueItemState.DEAD,
                )
                self.assertIsNone(completion.run.claim)
                self.assertIsNone(completion.queue_item.claim_token)
                self.assertIsNone(completion.queue_item.lease_expires_at)
                attempt_claim = completion.attempt.context.claim
                assert attempt_claim is not None
                self.assertEqual(
                    attempt_claim.claim_token,
                    claim.claim_token,
                )
                previous_segment = database.segments["segment"]
                self.assertEqual(
                    {
                        key: previous_segment[key]
                        for key in (
                            "state",
                            "claim",
                            "request_id",
                            "continuation_id",
                            "checkpoint_id",
                        )
                    },
                    {
                        key: previous_evidence[key]
                        for key in (
                            "state",
                            "claim",
                            "request_id",
                            "continuation_id",
                            "checkpoint_id",
                        )
                    },
                )
                active_segment = database.segments[active_segment_id]
                if active_segment_id == "segment":
                    self.assertEqual(
                        active_segment["state"],
                        TaskAttemptSegmentState.SUSPENDED.value,
                    )
                    self.assertIsNone(active_segment["claim"])
                else:
                    self.assertEqual(
                        active_segment["state"],
                        TaskAttemptSegmentState.ABANDONED.value,
                    )
                    active_claim = active_segment["claim"]
                    assert isinstance(active_claim, Mapping)
                    self.assertEqual(
                        active_claim["claim_token"],
                        claim.claim_token,
                    )
                continuation = await store.get_continuation(
                    request.continuation_id
                )
                self.assertEqual(
                    continuation.provider_snapshot,
                    claimed_continuation.provider_snapshot,
                )
                self.assertEqual(
                    continuation.transcript,
                    claimed_continuation.transcript,
                )
                self.assertEqual(
                    continuation.observations,
                    claimed_continuation.observations,
                )
                self.assertEqual(
                    continuation.definition,
                    claimed_continuation.definition,
                )
                self.assertEqual(
                    database.continuations[str(request.continuation_id)][
                        "invalid_reason"
                    ],
                    "task_cancelled",
                )
                self.assertIsNone(
                    database.continuations[str(request.continuation_id)][
                        "claim_owner_id"
                    ]
                )
                self.assertEqual(
                    {row["status"] for row in database.outbox.values()},
                    {"dead"},
                )
                self.assertEqual(set(database.events), baseline_events)
                self.assertFalse(
                    any(
                        row["event_type"]
                        == TaskInteractionEventType.INPUT_EXPIRED.value
                        for row in database.events.values()
                    )
                )
                new_segment_transitions = [
                    row
                    for transition_id, row in (
                        database.segment_transitions.items()
                    )
                    if transition_id not in baseline_segment_transitions
                ]
                if active_segment_id == "segment":
                    self.assertEqual(new_segment_transitions, [])
                else:
                    self.assertEqual(len(new_segment_transitions), 1)
                    self.assertEqual(
                        new_segment_transitions[0]["to_state"],
                        TaskAttemptSegmentState.ABANDONED.value,
                    )

                terminal_snapshot = database.snapshot()
                for restart_number in range(3):
                    restarted_store = await _store(database)
                    restarted = PgsqlDurableTaskCoordinator(
                        restarted_store,
                        PgsqlTaskStore(database, clock=lambda: _NOW),
                    )
                    replayed = await restarted.reconcile_expired_reentry(
                        queue_item_id="queue-item",
                        expected_claim_token=claim.claim_token,
                        task_run_id="run",
                        result=failure,
                        now=_NOW + timedelta(seconds=7 + restart_number),
                        metadata={"source": "restarted_sweep"},
                    )
                    self.assertEqual(replayed, recovered)
                    self.assertEqual(database.snapshot(), terminal_snapshot)

    async def test_cancelled_startup_rejects_released_reclaimed_owner(
        self,
    ) -> None:
        (
            database,
            store,
            task_store,
            request,
            claim,
            claimed_continuation,
            _,
        ) = await _prepare_cancel_requested_startup_boundary("segment_running")
        original_owner = claimed_continuation.claim.owner_id
        assert original_owner is not None
        released = await store.release(
            request.continuation_id,
            expected_store_revision=claimed_continuation.store_revision,
            owner_id=original_owner,
            fencing_token=claimed_continuation.fencing_token,
            now=_NOW + timedelta(seconds=5, microseconds=1),
        )
        other_owner = ContinuationClaimOwnerId("other-owner")
        reclaimed = await store.claim(
            request.continuation_id,
            expected_store_revision=released.store_revision,
            owner_id=other_owner,
            lease_expires_at=_NOW + timedelta(minutes=3),
            dispatch_id=ContinuationDispatchId("other-dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=5, microseconds=2),
        )
        self.assertEqual(
            reclaimed.continuation.claim.owner_id,
            other_owner,
        )
        self.assertEqual(
            int(reclaimed.fencing_token),
            int(claimed_continuation.fencing_token) + 1,
        )
        baseline = database.snapshot()

        with self.assertRaises(ContinuationStoreConflictError):
            await PgsqlDurableTaskCoordinator(
                store,
                task_store,
            ).reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token=claim.claim_token,
                task_run_id="run",
                result=TaskExecutionResult(
                    error={"code": "expired_durable_reentry_claim"}
                ),
                now=_NOW + timedelta(seconds=6),
            )

        self.assertEqual(database.snapshot(), baseline)
        self.assertEqual(
            await store.get_continuation(request.continuation_id),
            reclaimed.continuation,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.CANCEL_REQUESTED.value,
        )
        self.assertEqual(
            database.queue_items["queue-item"]["state"],
            TaskQueueItemState.CLAIMED.value,
        )

    async def test_cancelled_startup_requires_exact_claim_columns(
        self,
    ) -> None:
        boundaries = (
            "run_running",
            "attempt_running",
            "segment_created",
            "segment_running",
        )
        for boundary in boundaries:
            for field in (
                "claim_owner_id",
                "fencing_token",
                "store_revision",
            ):
                with self.subTest(boundary=boundary, field=field):
                    (
                        database,
                        store,
                        task_store,
                        request,
                        claim,
                        claimed_continuation,
                        _,
                    ) = await _prepare_cancel_requested_startup_boundary(
                        boundary
                    )
                    continuation_row = database.continuations[
                        str(request.continuation_id)
                    ]
                    if field == "claim_owner_id":
                        continuation_row[field] = "different-owner"
                    elif field == "fencing_token":
                        continuation_row[field] = (
                            int(claimed_continuation.fencing_token) + 1
                        )
                    else:
                        continuation_row[field] = (
                            int(claimed_continuation.store_revision) + 1
                        )
                    baseline = database.snapshot()

                    with self.assertRaises(ContinuationStoreConflictError):
                        await PgsqlDurableTaskCoordinator(
                            store,
                            task_store,
                        ).reconcile_expired_reentry(
                            queue_item_id="queue-item",
                            expected_claim_token=claim.claim_token,
                            task_run_id="run",
                            result=TaskExecutionResult(
                                error={"code": "expired_durable_reentry_claim"}
                            ),
                            now=_NOW + timedelta(seconds=6),
                        )

                    self.assertEqual(database.snapshot(), baseline)

    async def test_partial_startup_rejects_wrong_fences_and_provenance(
        self,
    ) -> None:
        (
            database,
            store,
            task_store,
            request,
            claim,
            _,
            active_segment_id,
        ) = await _prepare_cancel_requested_startup_boundary("segment_created")
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        failure = TaskExecutionResult(
            error={"code": "expired_durable_reentry_claim"}
        )
        baseline = database.snapshot()

        with self.assertRaises(TaskStoreConflictError):
            await coordinator.reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token="wrong-token",
                task_run_id="run",
                result=failure,
                now=_NOW + timedelta(seconds=6),
            )
        with self.assertRaises(TaskStoreConflictError):
            await coordinator.reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token=claim.claim_token,
                task_run_id="wrong-run",
                result=failure,
                now=_NOW + timedelta(seconds=6),
            )

        corruption_cases = (
            ("fence", "continuation", "fencing_token", 99),
            ("request", "segment", "request_id", "wrong-request"),
            (
                "continuation",
                "segment",
                "continuation_id",
                "wrong-continuation",
            ),
            (
                "checkpoint",
                "segment",
                "checkpoint_id",
                "wrong-checkpoint",
            ),
            (
                "provenance",
                active_segment_id,
                "resumed_from_segment_id",
                "wrong-segment",
            ),
            (
                "attempt",
                "queue-item",
                "attempt_id",
                "wrong-attempt",
            ),
        )
        for name, row_id, field, value in corruption_cases:
            with self.subTest(corruption=name):
                database.restore(deepcopy(baseline))
                if row_id == "continuation":
                    database.continuations[str(request.continuation_id)][
                        field
                    ] = value
                elif row_id == "queue-item":
                    database.queue_items[row_id][field] = value
                else:
                    database.segments[row_id][field] = value
                with self.assertRaises(
                    (
                        ContinuationStoreConflictError,
                        InteractionNotFoundError,
                        TaskStoreConflictError,
                    )
                ):
                    await coordinator.reconcile_expired_reentry(
                        queue_item_id="queue-item",
                        expected_claim_token=claim.claim_token,
                        task_run_id="run",
                        result=failure,
                        now=_NOW + timedelta(seconds=6),
                    )

        database.restore(deepcopy(baseline))
        recovered = await coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=claim.claim_token,
            task_run_id="run",
            result=failure,
            now=_NOW + timedelta(seconds=6),
        )
        terminal_snapshot = database.snapshot()
        with self.assertRaises(TaskStoreConflictError):
            await coordinator.reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token=claim.claim_token,
                task_run_id="run",
                result=TaskExecutionResult(
                    error={"code": "wrong_replay_result"}
                ),
                now=_NOW + timedelta(seconds=7),
            )
        self.assertEqual(database.snapshot(), terminal_snapshot)
        replayed = await coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=claim.claim_token,
            task_run_id="run",
            result=failure,
            now=_NOW + timedelta(seconds=7),
        )
        self.assertEqual(replayed, recovered)

    async def test_completed_provider_failure_is_atomic_and_replay_safe(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        (
            store,
            task_store,
            completed,
            provider_completion,
        ) = await _prepare_completed_provider(database)
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(
                error={
                    "category": "validation",
                    "code": "post_provider_processing_failed",
                },
                metadata={"privacy": "sanitized"},
            )
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        baseline = database.snapshot()
        baseline_attempts = database.queue_items["queue-item"]["attempts"]
        assert isinstance(baseline_attempts, int)
        expected_sql_steps = 18

        for fail_after in range(1, expected_sql_steps + 1):
            database.fail_after_queries = fail_after
            with self.assertRaises(RuntimeError):
                await coordinator.terminalize_completed_resume(
                    provider_completion,
                    failure,
                    queue_item_id="queue-item",
                    claim_token="resumed-claim-token",
                    segment_id="segment-next",
                    task_run_id="run",
                    request_id="request-run",
                    checkpoint_id="checkpoint",
                    now=_NOW + timedelta(seconds=6),
                )
            self.assertEqual(database.snapshot(), baseline)

        executed_before = len(database.executed)
        settled = await coordinator.terminalize_completed_resume(
            provider_completion,
            failure,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            request_id="request-run",
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=6),
        )
        self.assertEqual(
            len(database.executed) - executed_before,
            expected_sql_steps,
        )
        self.assertEqual(settled.completed_continuation, completed)
        self.assertEqual(settled.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            settled.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            settled.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(
            settled.completion.queue_item.attempts,
            baseline_attempts,
        )
        self.assertIsNone(settled.completion.run.claim)
        self.assertIsNone(settled.completion.queue_item.claim_token)
        self.assertIsNone(settled.completion.queue_item.lease_expires_at)
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.FAILED.value,
        )
        self.assertEqual(settled.completion.run.result, failure.result)
        transition_counts = (
            len(database.run_transitions),
            len(database.attempt_transitions),
            len(database.segment_transitions),
        )
        continuation_snapshot = database.continuations[
            str(completed.continuation_id)
        ].copy()

        replayed = await coordinator.terminalize_completed_resume(
            provider_completion,
            failure,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            request_id="request-run",
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=7),
            metadata={"replay": True},
        )

        self.assertEqual(replayed, settled)
        self.assertEqual(
            (
                len(database.run_transitions),
                len(database.attempt_transitions),
                len(database.segment_transitions),
            ),
            transition_counts,
        )
        self.assertEqual(
            database.continuations[str(completed.continuation_id)],
            continuation_snapshot,
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await coordinator.terminalize_completed_resume(
                replace(provider_completion, result_digest="e" * 64),
                failure,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                request_id="request-run",
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=8),
            )
        with self.assertRaises(InteractionNotFoundError):
            await coordinator.terminalize_completed_resume(
                provider_completion,
                failure,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                request_id="wrong-request",
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=8),
            )
        conflicting_failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"code": "another_safe_failure"})
        )
        with self.assertRaises(TaskStoreConflictError):
            await coordinator.terminalize_completed_resume(
                provider_completion,
                conflicting_failure,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                request_id="request-run",
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=8),
            )

    async def test_completed_provider_cancellation_preserves_semantics(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        (
            store,
            task_store,
            completed,
            provider_completion,
        ) = await _prepare_completed_provider(database)
        cancellation = TaskDurableResumeCancellation(
            result=TaskExecutionResult(error={"code": "cancelled"})
        )

        settled = await PgsqlDurableTaskCoordinator(
            store,
            task_store,
        ).terminalize_completed_resume(
            provider_completion,
            cancellation,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            request_id="request-run",
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=6),
        )

        self.assertEqual(settled.completed_continuation, completed)
        self.assertEqual(
            settled.completion.run.state,
            TaskRunState.CANCELLED,
        )
        self.assertEqual(
            settled.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            settled.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.ABANDONED.value,
        )
        self.assertIn(
            (
                TaskRunState.RUNNING.value,
                TaskRunState.CANCEL_REQUESTED.value,
            ),
            {
                (
                    row["from_state"],
                    row["to_state"],
                )
                for row in database.run_transitions.values()
            },
        )

    async def test_atomic_terminal_settlement_rolls_back_and_replays(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, dispatched = await _prepare_resumed_dispatch(
            database
        )
        settlement = TaskDurableResumeSuccess(
            result=TaskExecutionResult(
                output_summary={"answer": 42},
                metadata={"kind": "resumed"},
            )
        )
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=dispatched.fencing_token,
            result_digest=task_durable_resume_settlement_digest(settlement),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        database.fail_query = 'SET\n    "state" = %s,\n    "claimed_at" = NULL'
        with self.assertRaises(Exception):
            await coordinator.settle_resume(
                completion,
                settlement,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                now=_NOW + timedelta(seconds=5),
            )
        self.assertEqual(
            await store.get_continuation(dispatched.continuation_id),
            dispatched,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.RUNNING.value,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.RUNNING.value,
        )

        database.fail_query = None
        settled = await coordinator.settle_resume(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(
            settled.completed_continuation.claim.state,
            ContinuationClaimState.COMPLETED,
        )
        self.assertEqual(
            settled.completion.run.state,
            TaskRunState.SUCCEEDED,
        )
        self.assertEqual(
            settled.completion.attempt.state,
            TaskAttemptState.SUCCEEDED,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.SUCCEEDED.value,
        )
        self.assertEqual(
            settled.completion.run.result,
            settlement.result,
        )
        self.assertEqual(
            settled.completion.attempt.result,
            settlement.result,
        )
        transition_counts = (
            len(database.run_transitions),
            len(database.attempt_transitions),
            len(database.segment_transitions),
        )

        replayed = await coordinator.settle_resume(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=6),
        )
        self.assertEqual(replayed, settled)
        self.assertEqual(
            (
                len(database.run_transitions),
                len(database.attempt_transitions),
                len(database.segment_transitions),
            ),
            transition_counts,
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await coordinator.settle_resume(
                replace(
                    completion,
                    owner_id=ContinuationClaimOwnerId("wrong-owner"),
                ),
                settlement,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                now=_NOW + timedelta(seconds=6),
            )

        conflict = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"code": "invalid_output"})
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await coordinator.settle_resume(
                replace(
                    completion,
                    result_digest=task_durable_resume_settlement_digest(
                        conflict
                    ),
                ),
                conflict,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                now=_NOW + timedelta(seconds=7),
            )

    async def test_atomic_terminal_settlement_records_known_failure(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, dispatched = await _prepare_resumed_dispatch(
            database
        )
        settlement = TaskDurableResumeFailure(
            result=TaskExecutionResult(
                error={
                    "category": "validation",
                    "code": "output_invalid",
                }
            )
        )
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=dispatched.fencing_token,
            result_digest=task_durable_resume_settlement_digest(settlement),
        )
        result = await PgsqlDurableTaskCoordinator(
            store,
            task_store,
        ).settle_resume(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=5),
        )

        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.FAILED.value,
        )
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(result.completion.run.result, settlement.result)
        assert result.completed_continuation.completion is not None
        self.assertEqual(
            result.completed_continuation.completion.result_digest,
            completion.result_digest,
        )

    async def test_atomic_terminal_settlement_records_cancellation(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, dispatched = await _prepare_resumed_dispatch(
            database
        )
        settlement = TaskDurableResumeCancellation(
            result=TaskExecutionResult(error={"code": "cancelled"})
        )
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=dispatched.fencing_token,
            result_digest=task_durable_resume_settlement_digest(settlement),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        result = await coordinator.settle_resume(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=5),
        )

        self.assertEqual(
            result.completion.run.state,
            TaskRunState.CANCELLED,
        )
        self.assertEqual(
            result.completion.attempt.state,
            TaskAttemptState.FAILED,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.ABANDONED.value,
        )
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(result.completion.run.result, settlement.result)
        run_states = [
            (
                row["from_state"],
                row["to_state"],
            )
            for row in database.run_transitions.values()
        ]
        self.assertIn(
            (
                TaskRunState.RUNNING.value,
                TaskRunState.CANCEL_REQUESTED.value,
            ),
            run_states,
        )
        self.assertIn(
            (
                TaskRunState.CANCEL_REQUESTED.value,
                TaskRunState.CANCELLED.value,
            ),
            run_states,
        )
        replayed = await coordinator.settle_resume(
            completion,
            settlement,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=6),
        )
        self.assertEqual(replayed, result)

    async def test_ambiguous_resume_fails_task_without_replay(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, dispatched = await _prepare_resumed_dispatch(
            database
        )
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(
                error={"code": "provider_dispatch_ambiguous"}
            )
        )
        completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=ContinuationClaimOwnerId("resumed-worker"),
            fencing_token=dispatched.fencing_token,
            result_digest=task_durable_resume_settlement_digest(failure),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        database.fail_query = 'SET\n    "state" = %s,\n    "claimed_at" = NULL'
        with self.assertRaises(Exception):
            await coordinator.mark_resume_ambiguous(
                completion,
                failure,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                now=_NOW + timedelta(seconds=5),
            )
        self.assertEqual(
            await store.get_continuation(dispatched.continuation_id),
            dispatched,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.RUNNING.value,
        )

        database.fail_query = None
        result = await coordinator.mark_resume_ambiguous(
            completion,
            failure,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(result.ambiguous_continuation, dispatched)
        self.assertIsNone(result.ambiguous_continuation.completion)
        self.assertEqual(
            result.ambiguous_continuation.claim.state,
            ContinuationClaimState.DISPATCHED_AMBIGUOUS,
        )
        self.assertEqual(result.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            result.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        transition_counts = (
            len(database.run_transitions),
            len(database.attempt_transitions),
            len(database.segment_transitions),
        )
        replayed = await coordinator.mark_resume_ambiguous(
            completion,
            failure,
            queue_item_id="queue-item",
            claim_token="resumed-claim-token",
            segment_id="segment-next",
            task_run_id="run",
            now=_NOW + timedelta(seconds=6),
        )
        self.assertEqual(replayed, result)
        self.assertEqual(
            (
                len(database.run_transitions),
                len(database.attempt_transitions),
                len(database.segment_transitions),
            ),
            transition_counts,
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await coordinator.mark_resume_ambiguous(
                replace(
                    completion,
                    owner_id=ContinuationClaimOwnerId("stale-owner"),
                ),
                failure,
                queue_item_id="queue-item",
                claim_token="resumed-claim-token",
                segment_id="segment-next",
                task_run_id="run",
                now=_NOW + timedelta(seconds=7),
            )

    async def test_claimed_reentry_release_and_malformed_failure(
        self,
    ) -> None:
        release_database = FakePgsqlDatabase()
        (
            release_store,
            release_task_store,
            request,
            claim,
        ) = await _prepare_claimed_reentry(release_database)
        release_coordinator = PgsqlDurableTaskCoordinator(
            release_store,
            release_task_store,
        )
        released = await release_coordinator.release_claimed_reentry(
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=4),
        )
        self.assertEqual(released.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            released.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertEqual(
            released.attempt.state,
            TaskAttemptState.SUSPENDED,
        )
        self.assertIsNone(released.attempt.context.claim)
        replayed_release = await release_coordinator.release_claimed_reentry(
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(replayed_release, released)

        failure_database = FakePgsqlDatabase()
        (
            failure_store,
            failure_task_store,
            _,
            failure_claim,
        ) = await _prepare_claimed_reentry(failure_database)
        failure_database.queue_items["queue-item"].update(
            segment_id=None,
            request_id=None,
            continuation_id=None,
        )
        failure = TaskExecutionResult(
            error={"code": "resume_provenance_invalid"}
        )
        failure_coordinator = PgsqlDurableTaskCoordinator(
            failure_store,
            failure_task_store,
        )
        failed = await failure_coordinator.fail_claimed_reentry(
            queue_item_id="queue-item",
            claim_token=failure_claim.claim_token,
            task_run_id="run",
            request_id=None,
            continuation_id=None,
            checkpoint_id=None,
            result=failure,
            reason="resume_provenance_invalid",
            now=_NOW + timedelta(seconds=4),
        )
        self.assertEqual(failed.run.state, TaskRunState.FAILED)
        self.assertEqual(failed.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(failed.queue_item.state, TaskQueueItemState.DEAD)
        self.assertEqual(failed.run.result, failure)
        self.assertIsNone(failed.attempt.context.claim)
        replayed_failure = await failure_coordinator.fail_claimed_reentry(
            queue_item_id="queue-item",
            claim_token=failure_claim.claim_token,
            task_run_id="run",
            request_id=None,
            continuation_id=None,
            checkpoint_id=None,
            result=failure,
            reason="resume_provenance_invalid",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(replayed_failure, failed)

    async def test_running_pre_dispatch_reentry_release_is_replay_safe(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, request, claim = await _prepare_claimed_reentry(
            database
        )
        _seed_resumed_running_task(
            database,
            run_id="run",
            segment_id="segment-next",
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        database.fail_query = (
            'SET\n    "state" = %s,\n    "available_at" = %s,\n'
            '    "claimed_at" = NULL'
        )
        with self.assertRaises(Exception):
            await coordinator.release_running_reentry(
                queue_item_id="queue-item",
                claim_token=claim.claim_token,
                segment_id="segment-next",
                task_run_id="run",
                request_id=str(request.request_id),
                continuation_id=str(request.continuation_id),
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=4),
            )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.RUNNING.value,
        )
        self.assertEqual(
            database.segments["segment-next"]["state"],
            TaskAttemptSegmentState.RUNNING.value,
        )

        database.fail_query = None
        released = await coordinator.release_running_reentry(
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            segment_id="segment-next",
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=4),
        )
        self.assertEqual(released.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            released.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertEqual(
            released.attempt.state,
            TaskAttemptState.SUSPENDED,
        )
        self.assertIsNone(released.attempt.context.claim)
        self.assertEqual(
            released.previous_segment.state,
            TaskAttemptSegmentState.SUSPENDED,
        )
        self.assertEqual(
            released.previous_segment.request_id,
            str(request.request_id),
        )
        self.assertEqual(
            released.previous_segment.continuation_id,
            str(request.continuation_id),
        )
        self.assertEqual(
            released.previous_segment.checkpoint_id,
            "checkpoint",
        )
        self.assertEqual(
            released.queue_item.attempts,
            1,
        )
        replayed = await coordinator.release_running_reentry(
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            segment_id="segment-next",
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(replayed, released)

    async def test_blocked_cold_loader_expires_before_release(self) -> None:
        harness = await _deadline_worker_harness(blocked_loader=True)
        loader = harness.loader
        assert isinstance(loader, _BlockingDeadlineLoader)
        deadline = loader.deadline

        running = create_task(harness.worker.process_once())
        await loader.started.wait()
        loader.current[0] = deadline
        result = await wait_for(running, timeout=1)

        self.assertTrue(loader.cancelled.is_set())
        self.assertFalse(loader.proceed.is_set())
        self.assertGreaterEqual(harness.queue.heartbeat_calls, 1)
        self.assertTrue(result.lease_lost)
        assert result.completion is not None
        completion = result.completion
        self.assertEqual(completion.run.state, TaskRunState.EXPIRED)
        self.assertEqual(completion.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(
            completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        continuation_row = harness.database.continuations[
            str(harness.request.continuation_id)
        ]
        self.assertEqual(
            continuation_row["lifecycle_state"],
            DurableContinuationLifecycle.INVALIDATED.value,
        )
        self.assertEqual(continuation_row["invalid_reason"], "expired")
        self.assertEqual(
            harness.database.records[str(harness.request.request_id)][
                "request_state"
            ],
            RequestState.ANSWERED.value,
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in harness.database.events.values()
            ].count(TaskInteractionEventType.INPUT_EXPIRED.value),
            1,
        )
        self.assertEqual(harness.target_calls, [])

        snapshot = harness.database.snapshot()
        replayed = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "replay"}),
            now=deadline + timedelta(seconds=1),
        )
        self.assertEqual(replayed.completion, completion)
        self.assertEqual(harness.database.snapshot(), snapshot)

    async def test_blocked_cold_expiry_rolls_back_and_replays(self) -> None:
        harness = await _deadline_worker_harness(blocked_loader=True)
        loader = harness.loader
        assert isinstance(loader, _BlockingDeadlineLoader)
        deadline = loader.deadline
        record = await harness.store.get_task_continuation_record("run")
        snapshot = record.continuation.provider_snapshot
        assert type(snapshot) is ContinuationSnapshot
        expected_dispatch_id = derive_continuation_dispatch_id(
            harness.request.continuation_id
        )
        self.assertEqual(
            harness.provider_idempotency_key,
            derive_provider_idempotency_key(
                harness.request.continuation_id,
                expected_dispatch_id,
            ),
        )
        self.assertEqual(
            snapshot.provider_idempotency_key,
            harness.provider_idempotency_key,
        )
        harness.database.fail_query = task_pgsql._FAIL_REENTRY_RUN_SQL

        running = create_task(harness.worker.process_once())
        await loader.started.wait()
        loader.current[0] = deadline
        with self.assertRaises(PgsqlOperationError):
            await wait_for(running, timeout=1)

        self.assertTrue(loader.cancelled.is_set())
        self.assertFalse(loader.proceed.is_set())
        self.assertEqual(
            harness.database.continuations[
                str(harness.request.continuation_id)
            ]["lifecycle_state"],
            DurableContinuationLifecycle.CLAIMED.value,
        )
        self.assertEqual(
            harness.database.runs["run"]["state"],
            TaskRunState.CLAIMED.value,
        )
        self.assertEqual(
            harness.database.attempts["attempt"]["state"],
            TaskAttemptState.SUSPENDED.value,
        )
        self.assertEqual(
            harness.database.queue_items["queue-item"]["state"],
            TaskQueueItemState.CLAIMED.value,
        )
        self.assertFalse(
            any(
                event["event_type"]
                == TaskInteractionEventType.INPUT_EXPIRED.value
                for event in harness.database.events.values()
            )
        )

        harness.database.fail_query = None
        recovered = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "recover"}),
            now=deadline,
        )
        assert recovered.completion is not None
        self.assertEqual(
            recovered.completion.run.state,
            TaskRunState.EXPIRED,
        )
        database_snapshot = harness.database.snapshot()
        replayed = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "replay"}),
            now=deadline + timedelta(seconds=1),
        )
        self.assertEqual(replayed.completion, recovered.completion)
        self.assertEqual(harness.database.snapshot(), database_snapshot)

    async def test_worker_deadline_atomically_expires_claimed_reentry(
        self,
    ) -> None:
        harness = await _deadline_worker_harness()
        deadline = harness.request.created_at + timedelta(
            seconds=harness.request.continuation_ttl_seconds
        )
        self.assertGreater(harness.claim.lease_expires_at, deadline)

        result = await harness.worker.process_once()

        self.assertIsNotNone(result.completion)
        self.assertIsNone(result.reentry)
        self.assertTrue(result.lease_lost)
        assert result.completion is not None
        completion = result.completion
        self.assertEqual(completion.run.state, TaskRunState.EXPIRED)
        self.assertIsNone(completion.run.claim)
        self.assertEqual(completion.attempt.state, TaskAttemptState.FAILED)
        self.assertIsNone(completion.attempt.context.claim)
        self.assertEqual(
            completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertIsNone(completion.queue_item.claim_token)
        self.assertIsNone(completion.queue_item.lease_expires_at)
        assert completion.run.result is not None
        self.assertEqual(completion.run.result, completion.attempt.result)
        self.assertEqual(
            cast(
                Mapping[str, object],
                completion.run.result.error,
            )["code"],
            "timeout.exceeded",
        )
        self.assertEqual(
            completion.run.result.metadata["interaction_event_type"],
            TaskInteractionEventType.INPUT_EXPIRED.value,
        )
        segments = await harness.task_store.list_attempt_segments(
            completion.attempt.attempt_id
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(
            segments[0].state,
            TaskAttemptSegmentState.SUSPENDED,
        )
        self.assertEqual(
            segments[0].request_id,
            str(harness.request.request_id),
        )
        self.assertEqual(
            segments[0].continuation_id,
            str(harness.request.continuation_id),
        )
        self.assertEqual(segments[0].checkpoint_id, "checkpoint")
        continuation_row = harness.database.continuations[
            str(harness.request.continuation_id)
        ]
        self.assertEqual(
            continuation_row["lifecycle_state"],
            DurableContinuationLifecycle.INVALIDATED.value,
        )
        self.assertEqual(continuation_row["invalid_reason"], "expired")
        continuation = await harness.store.get_continuation(
            harness.request.continuation_id
        )
        self.assertIsNone(continuation.claim.owner_id)
        self.assertIsNone(continuation.claim.lease_expires_at)
        self.assertEqual(
            harness.database.records[str(harness.request.request_id)][
                "request_state"
            ],
            RequestState.ANSWERED.value,
        )
        self.assertEqual(
            tuple(harness.database.outbox.values())[0]["status"],
            "dead",
        )
        expired_events = tuple(
            event
            for event in harness.database.events.values()
            if event["event_type"]
            == TaskInteractionEventType.INPUT_EXPIRED.value
        )
        self.assertEqual(len(expired_events), 1)
        self.assertEqual(
            expired_events[0]["payload"],
            {
                "request_id": str(harness.request.request_id),
                "continuation_id": str(harness.request.continuation_id),
                "segment_id": segments[0].segment_id,
            },
        )
        self.assertEqual(harness.loader.calls, 1)
        self.assertEqual(harness.target_calls, [])

        snapshot = harness.database.snapshot()
        replayed = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "replayed"}),
            now=deadline + timedelta(seconds=1),
        )
        self.assertEqual(replayed.completion, completion)
        self.assertEqual(harness.database.snapshot(), snapshot)
        no_work = await harness.worker.process_once()
        self.assertFalse(no_work.processed)
        self.assertEqual(harness.database.snapshot(), snapshot)

    async def test_worker_deadline_reconciliation_rolls_back_atomically(
        self,
    ) -> None:
        harness = await _deadline_worker_harness()
        deadline = harness.request.created_at + timedelta(
            seconds=harness.request.continuation_ttl_seconds
        )
        transition_counts = (
            len(harness.database.run_transitions),
            len(harness.database.attempt_transitions),
            len(harness.database.events),
        )
        harness.database.fail_query = task_pgsql._FAIL_REENTRY_RUN_SQL

        with self.assertRaises(PgsqlOperationError):
            await harness.worker.process_once()

        continuation_row = harness.database.continuations[
            str(harness.request.continuation_id)
        ]
        self.assertEqual(
            continuation_row["lifecycle_state"],
            DurableContinuationLifecycle.CLAIMED.value,
        )
        self.assertEqual(
            harness.database.runs["run"]["state"],
            TaskRunState.CLAIMED.value,
        )
        self.assertEqual(
            harness.database.attempts["attempt"]["state"],
            TaskAttemptState.SUSPENDED.value,
        )
        self.assertEqual(
            harness.database.queue_items["queue-item"]["state"],
            TaskQueueItemState.CLAIMED.value,
        )
        self.assertEqual(
            transition_counts,
            (
                len(harness.database.run_transitions),
                len(harness.database.attempt_transitions),
                len(harness.database.events),
            ),
        )
        self.assertEqual(
            harness.database.records[str(harness.request.request_id)][
                "request_state"
            ],
            RequestState.ANSWERED.value,
        )
        self.assertFalse(
            any(
                event["event_type"]
                == TaskInteractionEventType.INPUT_EXPIRED.value
                for event in harness.database.events.values()
            )
        )

        harness.database.fail_query = None
        recovered = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "recover"}),
            now=deadline,
        )
        assert recovered.completion is not None
        self.assertEqual(
            recovered.completion.run.state,
            TaskRunState.EXPIRED,
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in harness.database.events.values()
            ].count(TaskInteractionEventType.INPUT_EXPIRED.value),
            1,
        )
        snapshot = harness.database.snapshot()
        replayed = await harness.task_coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=harness.claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "replay"}),
            now=deadline + timedelta(seconds=1),
        )
        self.assertEqual(replayed.completion, recovered.completion)
        self.assertEqual(harness.database.snapshot(), snapshot)

    async def test_expired_claimed_reentry_restores_same_attempt_atomically(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, request, claim = await _prepare_claimed_reentry(
            database
        )
        ready = await store.get_continuation(request.continuation_id)
        receipt = await store.claim(
            request.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId(claim.claim_token),
            lease_expires_at=_NOW + timedelta(seconds=5),
            dispatch_id=ContinuationDispatchId("expired-claim"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=4),
        )
        _expire_reentry_claim(
            database,
            expires_at=_NOW + timedelta(seconds=5),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        database.fail_query = (
            'SET\n    "state" = %s,\n    "available_at" = %s,\n'
            '    "claimed_at" = NULL'
        )
        with self.assertRaises(Exception):
            await coordinator.reconcile_expired_reentry(
                queue_item_id="queue-item",
                expected_claim_token=claim.claim_token,
                task_run_id="run",
                result=TaskExecutionResult(
                    error={"code": "expired_resume_claim"}
                ),
                now=_NOW + timedelta(seconds=6),
            )
        self.assertEqual(
            await store.get_continuation(request.continuation_id),
            receipt.continuation,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.CLAIMED.value,
        )
        database.fail_query = None

        committed = await coordinator.reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(error={"code": "expired_resume_claim"}),
            now=_NOW + timedelta(seconds=6),
        )
        assert committed.reentry is not None
        self.assertIsNone(committed.completion)
        self.assertEqual(committed.reentry.run.state, TaskRunState.QUEUED)
        self.assertEqual(
            committed.reentry.queue_item.state,
            TaskQueueItemState.AVAILABLE,
        )
        self.assertEqual(
            committed.reentry.attempt.state,
            TaskAttemptState.SUSPENDED,
        )
        self.assertEqual(committed.reentry.attempt.attempt_id, "attempt")
        self.assertEqual(len(database.attempts), 1)
        self.assertEqual(len(database.segments), 1)
        released = await store.get_continuation(request.continuation_id)
        self.assertEqual(
            released.claim.state,
            ContinuationClaimState.FAILED_SAFE_TO_RETRY,
        )

    async def test_expired_deadline_terminalizes_ready_and_claimed_reentry(
        self,
    ) -> None:
        for lifecycle in ("ready", "claimed"):
            with self.subTest(lifecycle=lifecycle):
                database = FakePgsqlDatabase()
                deadline_request = _request(
                    continuation_ttl_seconds=60,
                )
                (
                    store,
                    task_store,
                    request,
                    claim,
                ) = await _prepare_claimed_reentry(
                    database,
                    request=deadline_request,
                )
                if lifecycle == "claimed":
                    ready = await store.get_continuation(
                        request.continuation_id
                    )
                    await store.claim(
                        request.continuation_id,
                        expected_store_revision=ready.store_revision,
                        owner_id=ContinuationClaimOwnerId(claim.claim_token),
                        lease_expires_at=_NOW + timedelta(seconds=60),
                        dispatch_id=ContinuationDispatchId("deadline-claim"),
                        provider_idempotency_key=(
                            ProviderIdempotencyKey("provider-key")
                        ),
                        now=_NOW + timedelta(seconds=59),
                    )
                _expire_reentry_claim(
                    database,
                    expires_at=_NOW + timedelta(seconds=60),
                )

                committed = await PgsqlDurableTaskCoordinator(
                    store,
                    task_store,
                ).reconcile_expired_reentry(
                    queue_item_id="queue-item",
                    expected_claim_token=claim.claim_token,
                    task_run_id="run",
                    result=TaskExecutionResult(
                        error={"code": "lease_expired"}
                    ),
                    now=_NOW + timedelta(seconds=61),
                )

                self.assertIsNone(committed.reentry)
                assert committed.completion is not None
                self.assertEqual(
                    committed.completion.run.state,
                    TaskRunState.EXPIRED,
                )
                self.assertEqual(
                    committed.completion.attempt.state,
                    TaskAttemptState.FAILED,
                )
                self.assertEqual(
                    committed.completion.queue_item.state,
                    TaskQueueItemState.DEAD,
                )
                run_result = committed.completion.run.result
                assert run_result is not None
                run_error = run_result.error
                assert isinstance(run_error, Mapping)
                self.assertEqual(
                    run_error["code"],
                    "timeout.exceeded",
                )
                self.assertEqual(
                    database.records[str(request.request_id)]["request_state"],
                    RequestState.ANSWERED.value,
                )
                self.assertEqual(
                    database.continuations[str(request.continuation_id)][
                        "lifecycle_state"
                    ],
                    DurableContinuationLifecycle.INVALIDATED.value,
                )
                self.assertEqual(
                    [
                        event["event_type"]
                        for event in database.events.values()
                    ].count(TaskInteractionEventType.INPUT_EXPIRED.value),
                    1,
                )

    async def test_expired_running_reentry_restores_only_before_dispatch(
        self,
    ) -> None:
        pre_dispatch_database = FakePgsqlDatabase()
        (
            pre_store,
            pre_task_store,
            request,
            claim,
        ) = await _prepare_claimed_reentry(pre_dispatch_database)
        ready = await pre_store.get_continuation(request.continuation_id)
        await pre_store.claim(
            request.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId(claim.claim_token),
            lease_expires_at=_NOW + timedelta(seconds=5),
            dispatch_id=ContinuationDispatchId("pre-dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=4),
        )
        _seed_resumed_running_task(
            pre_dispatch_database,
            run_id="run",
            segment_id="segment-next",
        )
        _expire_reentry_claim(
            pre_dispatch_database,
            expires_at=_NOW + timedelta(seconds=5),
        )
        pre_commit = await PgsqlDurableTaskCoordinator(
            pre_store,
            pre_task_store,
        ).reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token=claim.claim_token,
            task_run_id="run",
            result=TaskExecutionResult(
                error={"code": "expired_before_dispatch"}
            ),
            now=_NOW + timedelta(seconds=6),
        )
        assert pre_commit.reentry is not None
        self.assertIsNone(pre_commit.completion)
        self.assertEqual(pre_commit.reentry.attempt.attempt_id, "attempt")
        self.assertEqual(
            pre_commit.reentry.previous_segment.segment_id,
            "segment-next",
        )
        self.assertEqual(len(pre_dispatch_database.attempts), 1)
        self.assertEqual(len(pre_dispatch_database.segments), 2)

        dispatched_database = FakePgsqlDatabase()
        (
            dispatched_store,
            dispatched_task_store,
            dispatched,
        ) = await _prepare_resumed_dispatch(dispatched_database)
        _expire_reentry_claim(
            dispatched_database,
            expires_at=_NOW + timedelta(seconds=5),
        )
        failure = TaskExecutionResult(error={"code": "expired_after_dispatch"})
        dispatched_commit = await PgsqlDurableTaskCoordinator(
            dispatched_store,
            dispatched_task_store,
        ).reconcile_expired_reentry(
            queue_item_id="queue-item",
            expected_claim_token="resumed-claim-token",
            task_run_id="run",
            result=failure,
            now=_NOW + timedelta(seconds=6),
        )
        self.assertIsNone(dispatched_commit.reentry)
        assert dispatched_commit.completion is not None
        self.assertEqual(
            dispatched_commit.completion.run.state,
            TaskRunState.FAILED,
        )
        self.assertEqual(
            dispatched_commit.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(dispatched_commit.completion.run.result, failure)
        self.assertEqual(len(dispatched_database.attempts), 1)
        self.assertEqual(len(dispatched_database.segments), 2)
        self.assertEqual(
            await dispatched_store.get_continuation(
                dispatched.continuation_id
            ),
            dispatched,
        )

    async def test_expired_reentry_rejects_stale_or_unexpired_claim(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, _, claim = await _prepare_claimed_reentry(database)
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        snapshot = database.snapshot()
        for token, now in (
            ("stale-token", _NOW + timedelta(minutes=3)),
            (claim.claim_token, _NOW + timedelta(seconds=4)),
        ):
            with self.subTest(token=token, now=now):
                with self.assertRaises(TaskStoreConflictError):
                    await coordinator.reconcile_expired_reentry(
                        queue_item_id="queue-item",
                        expected_claim_token=token,
                        task_run_id="run",
                        result=TaskExecutionResult(
                            error={"code": "expired_resume_claim"}
                        ),
                        now=now,
                    )
                self.assertEqual(database.snapshot(), snapshot)

    async def test_admitted_reentry_rejection_invalidates_without_orphan(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store, task_store, request, claim = await _prepare_claimed_reentry(
            database
        )
        ready = await store.get_continuation(request.continuation_id)
        receipt = await store.claim(
            ready.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId(claim.claim_token),
            lease_expires_at=claim.lease_expires_at,
            dispatch_id=ContinuationDispatchId("rejected-dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=3),
        )
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"code": "resume_setup_rejected"})
        )
        rejection = ContinuationRejectionCommand(
            continuation_id=receipt.continuation.continuation_id,
            expected_store_revision=receipt.continuation.store_revision,
            owner_id=ContinuationClaimOwnerId(claim.claim_token),
            fencing_token=receipt.fencing_token,
            result_digest=task_durable_resume_settlement_digest(failure),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        database.fail_query = 'SET\n    "state" = %s,\n    "claimed_at" = NULL'
        with self.assertRaises(Exception):
            await coordinator.fail_admitted_reentry(
                rejection,
                failure,
                queue_item_id="queue-item",
                claim_token=claim.claim_token,
                task_run_id="run",
                request_id=str(request.request_id),
                continuation_id=str(request.continuation_id),
                checkpoint_id="checkpoint",
                now=_NOW + timedelta(seconds=4),
            )
        self.assertEqual(
            await store.get_continuation(request.continuation_id),
            receipt.continuation,
        )
        self.assertEqual(
            database.runs["run"]["state"],
            TaskRunState.CLAIMED.value,
        )

        database.fail_query = None
        rejected = await coordinator.fail_admitted_reentry(
            rejection,
            failure,
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=4),
        )
        self.assertEqual(
            rejected.rejected_continuation.claim.state,
            ContinuationClaimState.FAILED_SAFE_TO_RETRY,
        )
        self.assertEqual(rejected.completion.run.state, TaskRunState.FAILED)
        self.assertEqual(
            rejected.completion.queue_item.state,
            TaskQueueItemState.DEAD,
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.INVALIDATED.value,
        )
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"],
            "dead",
        )
        with self.assertRaises(InteractionNotFoundError):
            await store.get_task_continuation_record("run")
        transition_counts = (
            len(database.run_transitions),
            len(database.attempt_transitions),
        )
        replayed = await coordinator.fail_admitted_reentry(
            rejection,
            failure,
            queue_item_id="queue-item",
            claim_token=claim.claim_token,
            task_run_id="run",
            request_id=str(request.request_id),
            continuation_id=str(request.continuation_id),
            checkpoint_id="checkpoint",
            now=_NOW + timedelta(seconds=5),
        )
        self.assertEqual(replayed, rejected)
        self.assertEqual(
            (
                len(database.run_transitions),
                len(database.attempt_transitions),
            ),
            transition_counts,
        )

    async def test_outbox_reconciler_recovers_leases_and_failures(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(
            database,
            store_policy=PgsqlInteractionStorePolicy(
                outbox_lease_seconds=2,
                outbox_max_attempts=2,
            ),
        )
        request = _request("outbox")
        created = await store.create_durable(
            _create_command(request),
            _portable(request),
        )
        assert isinstance(created, CreateInteractionApplied)
        await store.resolve(_answer(created))
        calls: list[str] = []

        async def fail(record: object) -> None:
            calls.append("failed")
            raise RuntimeError("private delivery failure")

        reconciler = PgsqlResumptionReconciler(
            store,
            owner_id=ContinuationClaimOwnerId("outbox-worker"),
            dispatcher=fail,
            clock=lambda: _NOW + timedelta(seconds=2),
        )
        self.assertEqual(await reconciler.run_once(), 1)
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"], "pending"
        )
        self.assertEqual(await reconciler.run_once(), 1)
        self.assertEqual(tuple(database.outbox.values())[0]["status"], "dead")
        self.assertEqual(calls, ["failed", "failed"])

    async def test_outbox_reconciler_requeues_after_preclaim_restart(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        (
            original_store,
            _,
            interaction,
        ) = await _commit_resolution_before_task_reentry(database)
        request = interaction.record.request
        self.assertEqual(database.runs["run"]["state"], "queued")
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"],
            "pending",
        )
        await original_store.aclose()

        restarted_store = await _store(database)
        restarted_task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW + timedelta(seconds=2),
        )
        deliveries: list[str] = []

        async def requeue(record: ResumptionOutboxRecord) -> None:
            assert record.task_run_id == "run"
            deliveries.append(record.outbox_id)
            await restarted_task_store.requeue_suspended(
                record.task_run_id,
                request_id=str(record.request_id),
                continuation_id=str(record.continuation_id),
                resolution_revision=int(record.resolution_revision),
                now=_NOW + timedelta(seconds=2),
            )

        reconciler = PgsqlResumptionReconciler(
            restarted_store,
            owner_id=ContinuationClaimOwnerId("restarted-reconciler"),
            dispatcher=requeue,
            clock=lambda: _NOW + timedelta(seconds=2),
        )
        self.assertEqual(await reconciler.run_once(), 1)
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(database.runs["run"]["state"], "queued")
        self.assertEqual(
            database.queue_items["queue-item"]["state"], "available"
        )
        self.assertEqual(
            database.continuations[str(request.continuation_id)][
                "lifecycle_state"
            ],
            "ready",
        )
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"], "delivered"
        )

    async def test_outbox_reconciler_recovers_expired_claim_and_fences_stale(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        original_store, _, _ = await _commit_resolution_before_task_reentry(
            database
        )
        stale_claims = await original_store.claim_outbox(
            owner_id=ContinuationClaimOwnerId("crashed-reconciler"),
            now=_NOW + timedelta(seconds=2),
        )
        self.assertEqual(len(stale_claims), 1)
        stale_claim = stale_claims[0]
        await original_store.aclose()

        restarted_store = await _store(database)
        restarted_task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW + timedelta(seconds=40),
        )
        delivered_fences: list[int] = []

        async def requeue(record: ResumptionOutboxRecord) -> None:
            assert record.task_run_id is not None
            delivered_fences.append(int(record.fencing_token))
            await restarted_task_store.requeue_suspended(
                record.task_run_id,
                request_id=str(record.request_id),
                continuation_id=str(record.continuation_id),
                resolution_revision=int(record.resolution_revision),
                now=_NOW + timedelta(seconds=40),
            )

        reconciler = PgsqlResumptionReconciler(
            restarted_store,
            owner_id=ContinuationClaimOwnerId("restarted-reconciler"),
            dispatcher=requeue,
            clock=lambda: _NOW + timedelta(seconds=40),
        )
        self.assertEqual(await reconciler.run_once(), 1)
        self.assertEqual(
            delivered_fences,
            [int(stale_claim.fencing_token) + 1],
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await restarted_store.complete_outbox(
                stale_claim,
                owner_id=ContinuationClaimOwnerId("crashed-reconciler"),
                now=_NOW + timedelta(seconds=41),
            )
        with self.assertRaises(ContinuationStoreConflictError):
            await restarted_store.release_outbox(
                stale_claim,
                owner_id=ContinuationClaimOwnerId("crashed-reconciler"),
                error_code="stale",
                now=_NOW + timedelta(seconds=41),
            )
        self.assertEqual(database.runs["run"]["state"], "queued")
        self.assertEqual(
            tuple(database.outbox.values())[0]["status"], "delivered"
        )

    async def test_task_lifecycle_matrix_is_atomic_replayed_and_guarded(
        self,
    ) -> None:
        command: (
            TerminalizeInteractionScopeCommand
            | SupersedeInteractionScopeCommand
            | TerminalizeDueInteractionsCommand
        )
        cases = (
            (
                "cancel",
                RequestState.CANCELLED,
                TaskRunState.CANCELLED,
                TaskAttemptState.ABANDONED,
                TaskInteractionEventType.INPUT_CANCELLED,
            ),
            (
                "supersede",
                RequestState.SUPERSEDED,
                TaskRunState.CANCELLED,
                TaskAttemptState.ABANDONED,
                TaskInteractionEventType.INPUT_SUPERSEDED,
            ),
            (
                "expire",
                RequestState.EXPIRED,
                TaskRunState.EXPIRED,
                TaskAttemptState.FAILED,
                TaskInteractionEventType.INPUT_EXPIRED,
            ),
        )
        for (
            operation,
            request_state,
            run_state,
            attempt_state,
            event_type,
        ) in cases:
            with self.subTest(operation=operation):
                database = FakePgsqlDatabase()
                clock = _Clock()
                store = await _store(database, clock=clock)
                task_store = PgsqlTaskStore(database, clock=lambda: clock.now)
                interaction = await _create_suspended_task(
                    database,
                    store,
                    task_store,
                    run_id=f"run-{operation}",
                    suffix=operation,
                )
                coordinator = PgsqlDurableTaskCoordinator(
                    store,
                    task_store,
                )
                actor = interaction.command.actor
                scope = InteractionExecutionScope(
                    run_id=interaction.record.request.origin.run_id,
                )
                if operation == "cancel":
                    command = TerminalizeInteractionScopeCommand(
                        actor=actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                elif operation == "supersede":
                    command = SupersedeInteractionScopeCommand(
                        actor=actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    )
                else:
                    clock.now += timedelta(minutes=11)
                    clock.monotonic += 660
                    command = TerminalizeDueInteractionsCommand()

                async def direct() -> object:
                    if isinstance(
                        command,
                        TerminalizeInteractionScopeCommand,
                    ):
                        return await store.terminalize_scope(command)
                    if isinstance(
                        command,
                        SupersedeInteractionScopeCommand,
                    ):
                        return await store.supersede_scope(command)
                    return await store.terminalize_due(command)

                async def converge() -> PgsqlDurableTaskLifecycle:
                    if isinstance(
                        command,
                        TerminalizeInteractionScopeCommand,
                    ):
                        return await coordinator.cancel_suspended_task(
                            command,
                            task_run_id=f"run-{operation}",
                            now=clock.now,
                        )
                    if isinstance(
                        command,
                        SupersedeInteractionScopeCommand,
                    ):
                        return await coordinator.supersede_suspended_task(
                            command,
                            task_run_id=f"run-{operation}",
                            now=clock.now,
                        )
                    return await coordinator.expire_suspended_task(
                        command,
                        task_run_id=f"run-{operation}",
                        now=clock.now,
                    )

                with self.assertRaises(PgsqlInteractionStoreError):
                    await direct()
                self.assertEqual(
                    database.records[
                        str(interaction.record.request.request_id)
                    ]["request_state"],
                    RequestState.PENDING.value,
                )

                database.fail_query = (
                    'UPDATE "task_queue_items"\nSET\n'
                    '    "state" = %s,\n    "claimed_at" = NULL'
                )
                with self.assertRaises(Exception):
                    await converge()
                self.assertEqual(
                    database.records[
                        str(interaction.record.request.request_id)
                    ]["request_state"],
                    RequestState.PENDING.value,
                )
                self.assertEqual(
                    database.runs[f"run-{operation}"]["state"],
                    TaskRunState.INPUT_REQUIRED.value,
                )
                self.assertEqual(
                    database.attempts[f"attempt-{operation}"]["state"],
                    TaskAttemptState.SUSPENDED.value,
                )
                self.assertEqual(
                    database.queue_items[f"queue-{operation}"]["state"],
                    TaskQueueItemState.SUSPENDED.value,
                )

                database.fail_query = None
                lifecycle = await converge()
                completion = lifecycle.completion_for(f"run-{operation}")
                assert completion is not None
                records = getattr(lifecycle.interaction, "records", None)
                assert isinstance(records, tuple)
                self.assertEqual(
                    records[0].request.state,
                    request_state,
                )
                self.assertEqual(completion.run.state, run_state)
                self.assertEqual(completion.attempt.state, attempt_state)
                self.assertEqual(
                    completion.queue_item.state,
                    TaskQueueItemState.DEAD,
                )
                self.assertEqual(
                    database.segments[f"segment-{operation}"]["state"],
                    TaskAttemptSegmentState.SUSPENDED.value,
                )
                self.assertEqual(
                    database.continuations[
                        str(interaction.record.request.continuation_id)
                    ]["lifecycle_state"],
                    DurableContinuationLifecycle.INVALIDATED.value,
                )

                replayed = await converge()
                replay_completion = replayed.completion_for(f"run-{operation}")
                self.assertEqual(replay_completion, completion)
                self.assertEqual(
                    [
                        row["event_type"] for row in database.events.values()
                    ].count(event_type.value),
                    1,
                )

    async def test_due_batch_terminalizes_every_task_bound_run(self) -> None:
        database = FakePgsqlDatabase()
        clock = _Clock()
        store = await _store(database, clock=clock)
        task_store = PgsqlTaskStore(database, clock=lambda: clock.now)
        interactions = (
            await _create_suspended_task(
                database,
                store,
                task_store,
                run_id="run-due-a",
                suffix="due-a",
            ),
            await _create_suspended_task(
                database,
                store,
                task_store,
                run_id="run-due-b",
                suffix="due-b",
            ),
        )
        clock.now += timedelta(minutes=11)
        clock.monotonic += 660

        lifecycle = await PgsqlDurableTaskCoordinator(
            store,
            task_store,
        ).expire_suspended_task(
            TerminalizeDueInteractionsCommand(),
            task_run_id="run-due-a",
            now=clock.now,
        )

        self.assertEqual(
            {completion.run.run_id for completion in lifecycle.completions},
            {"run-due-a", "run-due-b"},
        )
        for interaction in interactions:
            run_id = str(interaction.record.request.origin.run_id)
            self.assertEqual(
                database.records[str(interaction.record.request.request_id)][
                    "request_state"
                ],
                RequestState.EXPIRED.value,
            )
            self.assertEqual(
                database.runs[run_id]["state"],
                TaskRunState.EXPIRED.value,
            )
        self.assertEqual(
            {row["event_type"] for row in database.events.values()},
            {
                TaskInteractionEventType.INPUT_REQUIRED.value,
                TaskInteractionEventType.INPUT_EXPIRED.value,
            },
        )

    async def test_scope_lifecycle_collapses_same_run_branch_records(
        self,
    ) -> None:
        for operation in ("cancel", "supersede"):
            with self.subTest(operation=operation):
                database = FakePgsqlDatabase()
                store = await _store(database)
                task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
                current = await _create_suspended_task(
                    database,
                    store,
                    task_store,
                    run_id=f"run-branches-{operation}",
                    suffix=f"branches-{operation}",
                )
                current_request = current.record.request
                child_branch = BranchId(f"child-{operation}")
                registered = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=current.command.actor,
                        registration=InteractionBranchRegistration(
                            run_id=current_request.origin.run_id,
                            branch_id=child_branch,
                            parent_branch_id=(
                                current_request.origin.branch_id
                            ),
                            principal=current_request.origin.principal,
                        ),
                    )
                )
                self.assertTrue(registered.store_mutation_applied)
                child_request = replace(
                    _request(f"run-branches-{operation}"),
                    request_id=InputRequestId(f"request-child-{operation}"),
                    continuation_id=ContinuationId(
                        f"continuation-child-{operation}"
                    ),
                    origin=replace(
                        _origin(f"run-branches-{operation}"),
                        branch_id=child_branch,
                        parent_branch_id=(current_request.origin.branch_id),
                        model_call_id=ModelCallId(
                            f"model-call-child-{operation}"
                        ),
                    ),
                    reason=f"Child request {operation}.",
                )
                coordinator = PgsqlDurableTaskCoordinator(
                    store,
                    task_store,
                )
                child = await coordinator.create_pending_interaction(
                    _create_command(child_request),
                    _portable(child_request),
                    task_run_id=f"run-branches-{operation}",
                    checkpoint_id=f"checkpoint-child-{operation}",
                )
                self.assertIsInstance(child, CreateInteractionApplied)
                scope = InteractionExecutionScope(
                    run_id=current_request.origin.run_id,
                )
                if operation == "cancel":
                    lifecycle = await coordinator.cancel_suspended_task(
                        TerminalizeInteractionScopeCommand(
                            actor=current.command.actor,
                            scope=scope,
                            provenance=AnswerProvenance.HUMAN,
                        ),
                        task_run_id=f"run-branches-{operation}",
                    )
                    expected_state = RequestState.CANCELLED
                else:
                    lifecycle = await coordinator.supersede_suspended_task(
                        SupersedeInteractionScopeCommand(
                            actor=current.command.actor,
                            scope=scope,
                            provenance=AnswerProvenance.HUMAN,
                        ),
                        task_run_id=f"run-branches-{operation}",
                    )
                    expected_state = RequestState.SUPERSEDED

                records = getattr(lifecycle.interaction, "records", None)
                assert isinstance(records, tuple)
                self.assertEqual(len(records), 2)
                self.assertEqual(
                    {record.request.state for record in records},
                    {expected_state},
                )
                self.assertEqual(len(lifecycle.completions), 1)
                completion = lifecycle.completions[0]
                self.assertEqual(
                    completion.run.state,
                    TaskRunState.CANCELLED,
                )
                self.assertEqual(
                    completion.attempt.state,
                    TaskAttemptState.ABANDONED,
                )
                self.assertEqual(
                    database.segments[f"segment-branches-{operation}"][
                        "state"
                    ],
                    TaskAttemptSegmentState.SUSPENDED.value,
                )
                for request in (current_request, child_request):
                    self.assertEqual(
                        database.continuations[str(request.continuation_id)][
                            "lifecycle_state"
                        ],
                        DurableContinuationLifecycle.INVALIDATED.value,
                    )

    async def test_resolution_races_each_terminal_task_lifecycle(
        self,
    ) -> None:
        for operation in ("cancel", "supersede", "expire"):
            for lifecycle_first in (False, True):
                with self.subTest(
                    operation=operation,
                    lifecycle_first=lifecycle_first,
                ):
                    database = FakePgsqlDatabase()
                    clock = _Clock()
                    store = await _store(database, clock=clock)
                    task_store = PgsqlTaskStore(
                        database,
                        clock=lambda: clock.now,
                    )
                    run_id = (
                        f"run-race-{operation}-"
                        f"{'lifecycle' if lifecycle_first else 'answer'}"
                    )
                    interaction = await _create_suspended_task(
                        database,
                        store,
                        task_store,
                        run_id=run_id,
                        suffix=run_id,
                    )
                    coordinator = PgsqlDurableTaskCoordinator(
                        store,
                        task_store,
                    )
                    scope = InteractionExecutionScope(
                        run_id=interaction.record.request.origin.run_id,
                    )
                    if operation == "cancel":
                        lifecycle_call = coordinator.cancel_suspended_task(
                            TerminalizeInteractionScopeCommand(
                                actor=interaction.command.actor,
                                scope=scope,
                                provenance=AnswerProvenance.HUMAN,
                            ),
                            task_run_id=run_id,
                            now=clock.now,
                        )
                        terminal_request_state = RequestState.CANCELLED
                        terminal_run_state = TaskRunState.CANCELLED
                        terminal_event = (
                            TaskInteractionEventType.INPUT_CANCELLED
                        )
                    elif operation == "supersede":
                        lifecycle_call = coordinator.supersede_suspended_task(
                            SupersedeInteractionScopeCommand(
                                actor=interaction.command.actor,
                                scope=scope,
                                provenance=AnswerProvenance.HUMAN,
                            ),
                            task_run_id=run_id,
                            now=clock.now,
                        )
                        terminal_request_state = RequestState.SUPERSEDED
                        terminal_run_state = TaskRunState.CANCELLED
                        terminal_event = (
                            TaskInteractionEventType.INPUT_SUPERSEDED
                        )
                    else:
                        clock.now += timedelta(minutes=11)
                        clock.monotonic += 660
                        lifecycle_call = coordinator.expire_suspended_task(
                            TerminalizeDueInteractionsCommand(),
                            task_run_id=run_id,
                            now=clock.now,
                        )
                        terminal_request_state = RequestState.EXPIRED
                        terminal_run_state = TaskRunState.EXPIRED
                        terminal_event = TaskInteractionEventType.INPUT_EXPIRED
                    answer_call = coordinator.resolve_and_requeue(
                        _answer(
                            interaction,
                            key=f"answer-{run_id}",
                        ),
                        task_run_id=run_id,
                        now=clock.now,
                    )
                    calls = (
                        (lifecycle_call, answer_call)
                        if lifecycle_first
                        else (answer_call, lifecycle_call)
                    )

                    outcomes = await gather(
                        *calls,
                        return_exceptions=True,
                    )

                    request_row = database.records[
                        str(interaction.record.request.request_id)
                    ]
                    continuation_row = database.continuations[
                        str(interaction.record.request.continuation_id)
                    ]
                    run_row = database.runs[run_id]
                    queue_row = database.queue_items[f"queue-{run_id}"]
                    event_types = [
                        event["event_type"]
                        for event in database.events.values()
                    ]
                    if (
                        request_row["request_state"]
                        == RequestState.ANSWERED.value
                    ):
                        self.assertEqual(
                            run_row["state"],
                            TaskRunState.QUEUED.value,
                        )
                        self.assertEqual(
                            queue_row["state"],
                            TaskQueueItemState.AVAILABLE.value,
                        )
                        self.assertEqual(
                            continuation_row["lifecycle_state"],
                            DurableContinuationLifecycle.READY.value,
                        )
                        self.assertEqual(
                            event_types.count(
                                TaskInteractionEventType.INPUT_RESUMED.value
                            ),
                            1,
                        )
                        self.assertEqual(
                            event_types.count(terminal_event.value),
                            0,
                        )
                    else:
                        self.assertEqual(
                            request_row["request_state"],
                            terminal_request_state.value,
                        )
                        self.assertEqual(
                            run_row["state"],
                            terminal_run_state.value,
                        )
                        self.assertEqual(
                            queue_row["state"],
                            TaskQueueItemState.DEAD.value,
                        )
                        self.assertEqual(
                            continuation_row["lifecycle_state"],
                            DurableContinuationLifecycle.INVALIDATED.value,
                        )
                        self.assertEqual(
                            event_types.count(terminal_event.value),
                            1,
                        )
                        self.assertEqual(
                            event_types.count(
                                TaskInteractionEventType.INPUT_RESUMED.value
                            ),
                            0,
                        )
                    self.assertGreaterEqual(
                        sum(
                            not isinstance(outcome, BaseException)
                            for outcome in outcomes
                        ),
                        1,
                    )

    async def test_task_client_cancels_only_exact_bound_branches_atomically(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
        parent = await _create_suspended_task(
            database,
            store,
            task_store,
            run_id="run-client-branches",
            suffix="client-branches",
        )
        parent_request = parent.record.request
        middle_branch = BranchId("client-middle")
        await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=parent.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=parent_request.origin.run_id,
                    branch_id=middle_branch,
                    parent_branch_id=parent_request.origin.branch_id,
                    principal=parent_request.origin.principal,
                ),
            )
        )
        child_branch = BranchId("client-child")
        await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=parent.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=parent_request.origin.run_id,
                    branch_id=child_branch,
                    parent_branch_id=middle_branch,
                    principal=parent_request.origin.principal,
                ),
            )
        )
        child_request = replace(
            _request("run-client-branches"),
            request_id=InputRequestId("request-client-child"),
            continuation_id=ContinuationId("continuation-client-child"),
            origin=replace(
                _origin("run-client-branches"),
                branch_id=child_branch,
                parent_branch_id=middle_branch,
                model_call_id=ModelCallId("model-call-client-child"),
            ),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        child = await coordinator.create_pending_interaction(
            _create_command(child_request),
            _portable(child_request),
            task_run_id="run-client-branches",
            checkpoint_id="checkpoint-client-child",
        )
        healthy_request = replace(
            _request("client-unbound-healthy"),
            origin=replace(
                _origin("run-client-branches"),
                branch_id=BranchId("client-unbound-healthy-root"),
                model_call_id=ModelCallId("model-call-client-unbound-healthy"),
            ),
        )
        healthy = await store.create_durable(
            _create_command(healthy_request),
            _portable(healthy_request),
        )
        self.assertIsInstance(healthy, CreateInteractionApplied)
        corrupt_request = replace(
            _request("client-unbound-corrupt"),
            origin=replace(
                _origin("run-client-branches"),
                branch_id=BranchId("client-unbound-corrupt-root"),
                model_call_id=ModelCallId("model-call-client-unbound-corrupt"),
            ),
        )
        corrupt = await store.create_durable(
            _create_command(corrupt_request),
            _portable(corrupt_request),
        )
        self.assertIsInstance(corrupt, CreateInteractionApplied)
        healthy_branch_id = BranchId("client-unbound-healthy-child")
        healthy_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=healthy.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=healthy_request.origin.run_id,
                    branch_id=healthy_branch_id,
                    parent_branch_id=healthy_request.origin.branch_id,
                    principal=healthy_request.origin.principal,
                ),
            )
        )
        self.assertTrue(healthy_branch.store_mutation_applied)
        corrupt_branch_id = BranchId("client-unbound-corrupt-child")
        corrupt_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=corrupt.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=corrupt_request.origin.run_id,
                    branch_id=corrupt_branch_id,
                    parent_branch_id=corrupt_request.origin.branch_id,
                    principal=corrupt_request.origin.principal,
                ),
            )
        )
        self.assertTrue(corrupt_branch.store_mutation_applied)
        corrupt_record_key = str(corrupt_request.request_id)
        corrupt_continuation_key = str(corrupt_request.continuation_id)
        self.assertIsNone(
            database.continuations[str(healthy_request.continuation_id)][
                "task_run_id"
            ]
        )
        self.assertIsNone(
            database.continuations[corrupt_continuation_key]["task_run_id"]
        )
        corrupt_branch_key = (
            str(corrupt_request.origin.run_id),
            str(corrupt_branch_id),
            _scope_identity_digest(
                corrupt_request.origin.run_id,
                corrupt_request.origin.principal,
            ),
        )
        database.records[corrupt_record_key]["ciphertext"] = b"\x00"
        database.continuations[corrupt_continuation_key][
            "ciphertext"
        ] = b"\x00"
        database.branches[corrupt_branch_key]["ciphertext"] = b"\x00"
        corrupt_record_before = deepcopy(database.records[corrupt_record_key])
        corrupt_continuation_before = deepcopy(
            database.continuations[corrupt_continuation_key]
        )
        branches_before = deepcopy(database.branches)
        sibling = await _create_suspended_task(
            database,
            store,
            task_store,
            run_id="run-client-sibling",
            suffix="client-sibling",
        )
        client = TaskClient(
            task_store,
            target=_unused_task_target,
            durable_lifecycle_coordinator=coordinator,
            clock=lambda: _NOW,
        )
        database.fail_query = (
            'UPDATE "task_queue_items"\nSET\n'
            '    "state" = %s,\n    "claimed_at" = NULL'
        )
        rollback_snapshot = database.snapshot()
        rollback_query_start = len(database.executed)
        with self.assertRaises(Exception):
            await client.cancel("run-client-branches")
        rollback_queries = tuple(
            query
            for query, _parameters in database.executed[rollback_query_start:]
        )
        self.assertEqual(database.snapshot(), rollback_snapshot)

        database.fail_query = None
        cancellation_query_start = len(database.executed)
        cancelled = await client.cancel("run-client-branches")
        cancellation_queries = tuple(
            query
            for query, _parameters in database.executed[
                cancellation_query_start:
            ]
        )
        settled_snapshot = database.snapshot()
        replay_query_start = len(database.executed)
        replayed = await client.cancel("run-client-branches")
        replay_queries = tuple(
            query
            for query, _parameters in database.executed[replay_query_start:]
        )
        self.assertEqual(cancelled, replayed)
        self.assertEqual(database.snapshot(), settled_snapshot)
        self.assertEqual(cancelled.state, TaskRunState.CANCELLED)
        for queries in (
            rollback_queries,
            cancellation_queries,
            replay_queries,
        ):
            self.assertNotIn(_SELECT_RECORDS_SQL, queries)
            self.assertNotIn(_SELECT_SCOPE_RECORDS_SQL, queries)
            self.assertNotIn(_SELECT_BRANCHES_SQL, queries)
            self.assertNotIn(_SELECT_SCOPE_BRANCHES_SQL, queries)
        for queries in (rollback_queries, cancellation_queries):
            self.assertEqual(
                queries.count(_SELECT_TASK_BRANCH_CLOSURE_SQL),
                1,
            )
        self.assertIn(
            _SELECT_TASK_SCOPE_IDENTITIES_FOR_UPDATE_SQL,
            cancellation_queries,
        )
        self.assertEqual(
            cancellation_queries.count(
                _SELECT_TASK_INTERACTIONS_FOR_UPDATE_SQL
            ),
            2,
        )
        for interaction in (parent, child):
            request = interaction.record.request
            self.assertEqual(
                database.records[str(request.request_id)]["request_state"],
                RequestState.CANCELLED.value,
            )
            self.assertEqual(
                database.continuations[str(request.continuation_id)][
                    "lifecycle_state"
                ],
                DurableContinuationLifecycle.INVALIDATED.value,
            )
        self.assertEqual(
            database.records[str(healthy_request.request_id)]["request_state"],
            RequestState.PENDING.value,
        )
        self.assertEqual(
            database.continuations[str(healthy_request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.PENDING.value,
        )
        self.assertEqual(
            database.records[corrupt_record_key],
            corrupt_record_before,
        )
        self.assertEqual(
            database.continuations[corrupt_continuation_key],
            corrupt_continuation_before,
        )
        self.assertEqual(database.branches, branches_before)
        sibling_request = sibling.record.request
        self.assertEqual(
            database.records[str(sibling_request.request_id)]["request_state"],
            RequestState.PENDING.value,
        )
        self.assertEqual(
            database.continuations[str(sibling_request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.PENDING.value,
        )
        self.assertEqual(
            database.runs["run-client-sibling"]["state"],
            TaskRunState.INPUT_REQUIRED.value,
        )
        self.assertEqual(
            [event["event_type"] for event in database.events.values()].count(
                TaskInteractionEventType.INPUT_CANCELLED.value
            ),
            1,
        )

    async def test_trusted_task_cancel_attests_child_only_branch_root(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
        run_id = "run-client-child-only"
        root_request = replace(
            _request("client-child-only-root"),
            origin=replace(
                _origin(run_id),
                branch_id=BranchId("client-child-only-root"),
                model_call_id=ModelCallId("model-call-client-child-only-root"),
            ),
        )
        root = await store.create_durable(
            _create_command(root_request),
            _portable(root_request),
        )
        self.assertIsInstance(root, CreateInteractionApplied)
        assert isinstance(root, CreateInteractionApplied)
        terminal_root = await store.cancel(
            CancelInteractionCommand(
                actor=root.command.actor,
                correlation=root.record.correlation,
                provenance=AnswerProvenance.HUMAN,
            )
        )
        self.assertTrue(terminal_root.store_mutation_applied)
        middle_branch_id = BranchId("client-child-only-middle")
        middle = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=root.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=root_request.origin.run_id,
                    branch_id=middle_branch_id,
                    parent_branch_id=root_request.origin.branch_id,
                    principal=root_request.origin.principal,
                ),
            )
        )
        self.assertTrue(middle.store_mutation_applied)
        child_branch_id = BranchId("client-child-only-child")
        child_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=root.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=root_request.origin.run_id,
                    branch_id=child_branch_id,
                    parent_branch_id=middle_branch_id,
                    principal=root_request.origin.principal,
                ),
            )
        )
        self.assertTrue(child_branch.store_mutation_applied)
        _seed_running_task(
            database,
            run_id,
            attempt_id="attempt-client-child-only",
            segment_id="segment-client-child-only",
            queue_item_id="queue-client-child-only",
            claim_token="claim-client-child-only",
        )
        child_request = replace(
            _request("client-child-only-bound"),
            origin=replace(
                _origin(run_id),
                branch_id=child_branch_id,
                parent_branch_id=middle_branch_id,
                model_call_id=ModelCallId(
                    "model-call-client-child-only-bound"
                ),
            ),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        suspended = await coordinator.create_and_suspend(
            _create_command(child_request),
            _portable(child_request),
            queue_item_id="queue-client-child-only",
            claim_token="claim-client-child-only",
            segment_id="segment-client-child-only",
            task_run_id=run_id,
            checkpoint_id="checkpoint-client-child-only",
        )
        self.assertIsNone(
            database.continuations[str(root_request.continuation_id)][
                "task_run_id"
            ]
        )
        query_start = len(database.executed)

        completion = await coordinator.cancel_input_required_task(
            task_run_id=run_id,
            now=_NOW,
            metadata={},
        )

        queries = tuple(
            query for query, _parameters in database.executed[query_start:]
        )
        self.assertEqual(completion.run.state, TaskRunState.CANCELLED)
        self.assertEqual(
            database.records[str(root_request.request_id)]["request_state"],
            RequestState.CANCELLED.value,
        )
        self.assertEqual(
            database.records[str(child_request.request_id)]["request_state"],
            RequestState.CANCELLED.value,
        )
        self.assertEqual(
            database.continuations[str(child_request.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.INVALIDATED.value,
        )
        self.assertEqual(
            suspended.interaction.record.request.origin,
            child_request.origin,
        )
        self.assertEqual(
            queries.count(_SELECT_TASK_BRANCH_CLOSURE_SQL),
            1,
        )
        self.assertNotIn(_SELECT_RECORDS_SQL, queries)
        self.assertNotIn(_SELECT_SCOPE_RECORDS_SQL, queries)
        self.assertNotIn(_SELECT_BRANCHES_SQL, queries)
        self.assertNotIn(_SELECT_SCOPE_BRANCHES_SQL, queries)

    async def test_trusted_task_cancel_validates_bound_branch_ancestors(
        self,
    ) -> None:
        for corruption in ("ciphertext", "cycle", "missing", "root"):
            with self.subTest(corruption=corruption):
                database = FakePgsqlDatabase()
                store = await _store(database)
                task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
                run_id = f"run-client-ancestor-{corruption}"
                parent = await _create_suspended_task(
                    database,
                    store,
                    task_store,
                    run_id=run_id,
                    suffix=f"client-ancestor-{corruption}",
                )
                parent_request = parent.record.request
                ancestor_branch_id = BranchId(f"client-ancestor-{corruption}")
                registered_ancestor = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=parent.command.actor,
                        registration=InteractionBranchRegistration(
                            run_id=parent_request.origin.run_id,
                            branch_id=ancestor_branch_id,
                            parent_branch_id=(parent_request.origin.branch_id),
                            principal=parent_request.origin.principal,
                        ),
                    )
                )
                self.assertTrue(registered_ancestor.store_mutation_applied)
                self.assertIsInstance(
                    registered_ancestor,
                    InteractionBranchRegistrationApplied,
                )
                assert isinstance(
                    registered_ancestor,
                    InteractionBranchRegistrationApplied,
                )
                child_branch_id = BranchId(f"client-child-{corruption}")
                registered_child = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=parent.command.actor,
                        registration=InteractionBranchRegistration(
                            run_id=parent_request.origin.run_id,
                            branch_id=child_branch_id,
                            parent_branch_id=ancestor_branch_id,
                            principal=parent_request.origin.principal,
                        ),
                    )
                )
                self.assertTrue(registered_child.store_mutation_applied)
                child_request = replace(
                    _request(run_id),
                    request_id=InputRequestId(
                        f"request-client-ancestor-{corruption}"
                    ),
                    continuation_id=ContinuationId(
                        f"continuation-client-ancestor-{corruption}"
                    ),
                    origin=replace(
                        _origin(run_id),
                        branch_id=child_branch_id,
                        parent_branch_id=ancestor_branch_id,
                        model_call_id=ModelCallId(
                            f"model-call-client-ancestor-{corruption}"
                        ),
                    ),
                )
                coordinator = PgsqlDurableTaskCoordinator(
                    store,
                    task_store,
                )
                child = await coordinator.create_pending_interaction(
                    _create_command(child_request),
                    _portable(child_request),
                    task_run_id=run_id,
                    checkpoint_id=f"checkpoint-client-ancestor-{corruption}",
                )
                ancestor_key = (
                    run_id,
                    str(ancestor_branch_id),
                    _scope_identity_digest(
                        parent_request.origin.run_id,
                        parent_request.origin.principal,
                    ),
                )
                child_key = (
                    run_id,
                    str(child_branch_id),
                    ancestor_key[2],
                )
                if corruption == "ciphertext":
                    database.branches[ancestor_key]["ciphertext"] = b"\x00"
                elif corruption == "cycle":
                    cycle_record = replace(
                        registered_ancestor.record,
                        registration=replace(
                            registered_ancestor.record.registration,
                            parent_branch_id=child_branch_id,
                        ),
                    )
                    database.branches[ancestor_key]["parent_branch_id"] = str(
                        child_branch_id
                    )
                    database.branches[ancestor_key]["ciphertext"] = bytes(
                        byte ^ 0xA5 for byte in _encode_branch(cycle_record)
                    )
                elif corruption == "missing":
                    database.branches.pop(child_key)
                else:
                    database.branches[ancestor_key][
                        "root_branch_id"
                    ] = "tampered-root"
                before = database.snapshot()
                query_start = len(database.executed)

                with self.assertRaises(InputContractError):
                    await coordinator.cancel_input_required_task(
                        task_run_id=run_id,
                        now=_NOW,
                        metadata={},
                    )

                queries = tuple(
                    query
                    for query, _parameters in database.executed[query_start:]
                )
                self.assertEqual(database.snapshot(), before)
                self.assertEqual(
                    queries.count(_SELECT_TASK_BRANCH_CLOSURE_SQL),
                    1,
                )
                self.assertNotIn(_SELECT_BRANCHES_SQL, queries)
                self.assertNotIn(_SELECT_SCOPE_BRANCHES_SQL, queries)
                for interaction in (parent, child):
                    request = interaction.record.request
                    self.assertEqual(
                        database.records[str(request.request_id)][
                            "request_state"
                        ],
                        RequestState.PENDING.value,
                    )
                self.assertEqual(
                    database.runs[run_id]["state"],
                    TaskRunState.INPUT_REQUIRED.value,
                )

    async def test_trusted_task_cancel_requires_one_bound_owner(self) -> None:
        def signature(
            error: PgsqlInteractionStoreError,
        ) -> tuple[InputErrorCode, str, str]:
            return error.code, error.path, error.safe_message

        missing_database = FakePgsqlDatabase()
        missing_store = await _store(missing_database)
        missing_task_store = PgsqlTaskStore(
            missing_database,
            clock=lambda: _NOW,
        )
        missing_run_id = "run-client-owner-missing"
        missing = await _create_suspended_task(
            missing_database,
            missing_store,
            missing_task_store,
            run_id=missing_run_id,
            suffix="client-owner-missing",
        )
        missing_database.continuations[
            str(missing.record.request.continuation_id)
        ]["task_run_id"] = None
        missing_snapshot = missing_database.snapshot()
        missing_signatures: list[tuple[InputErrorCode, str, str]] = []
        missing_coordinator = PgsqlDurableTaskCoordinator(
            missing_store,
            missing_task_store,
        )
        for task_run_id in (missing_run_id, "unknown-task-run"):
            with self.assertRaises(PgsqlInteractionStoreError) as raised:
                await missing_coordinator.cancel_input_required_task(
                    task_run_id=task_run_id,
                    now=_NOW,
                    metadata={},
                )
            missing_signatures.append(signature(raised.exception))
            self.assertEqual(
                missing_database.snapshot(),
                missing_snapshot,
            )
        self.assertEqual(
            missing_signatures,
            [missing_signatures[0]] * 2,
        )

        ambiguous_database = FakePgsqlDatabase()
        ambiguous_store = await _store(ambiguous_database)
        ambiguous_task_store = PgsqlTaskStore(
            ambiguous_database,
            clock=lambda: _NOW,
        )
        ambiguous_run_id = "run-client-owner-ambiguous"
        await _create_suspended_task(
            ambiguous_database,
            ambiguous_store,
            ambiguous_task_store,
            run_id=ambiguous_run_id,
            suffix="client-owner-ambiguous",
        )
        foreign_request = replace(
            _request("client-owner-foreign"),
            origin=replace(
                _origin(ambiguous_run_id),
                branch_id=BranchId("client-owner-foreign-root"),
                model_call_id=ModelCallId("model-call-client-owner-foreign"),
                principal=PrincipalScope(user_id=UserId("foreign-owner")),
            ),
        )
        ambiguous_coordinator = PgsqlDurableTaskCoordinator(
            ambiguous_store,
            ambiguous_task_store,
        )
        await ambiguous_coordinator.create_pending_interaction(
            _create_command(foreign_request),
            _portable(foreign_request),
            task_run_id=ambiguous_run_id,
            checkpoint_id="checkpoint-client-owner-foreign",
        )
        ambiguous_snapshot = ambiguous_database.snapshot()

        with self.assertRaises(PgsqlInteractionStoreError) as ambiguous_error:
            await ambiguous_coordinator.cancel_input_required_task(
                task_run_id=ambiguous_run_id,
                now=_NOW,
                metadata={},
            )

        self.assertEqual(
            ambiguous_error.exception.code,
            InputErrorCode.FORBIDDEN,
        )
        self.assertEqual(
            ambiguous_database.snapshot(),
            ambiguous_snapshot,
        )

    async def test_task_client_uses_atomic_durable_cancellation(self) -> None:
        missing_database = FakePgsqlDatabase()
        missing_store = await _store(missing_database)
        missing_task_store = PgsqlTaskStore(
            missing_database,
            clock=lambda: _NOW,
        )
        missing_interaction = await _create_suspended_task(
            missing_database,
            missing_store,
            missing_task_store,
            run_id="run-client-missing",
            suffix="client-missing",
        )
        missing_client = TaskClient(
            missing_task_store,
            target=_unused_task_target,
            clock=lambda: _NOW,
        )

        with self.assertRaises(TaskClientUnsupportedOperationError) as error:
            await missing_client.cancel("run-client-missing")
        self.assertEqual(
            error.exception.code,
            "task.durable_lifecycle_unavailable",
        )
        with self.assertRaises(PgsqlInteractionStoreError):
            await missing_store.resolve(_answer(missing_interaction))
        with self.assertRaises(PgsqlInteractionStoreError):
            await missing_store.cancel(
                CancelInteractionCommand(
                    actor=missing_interaction.command.actor,
                    correlation=missing_interaction.record.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=(
                        missing_interaction.record.request.state_revision
                    ),
                )
            )
        self.assertEqual(
            missing_database.runs["run-client-missing"]["state"],
            TaskRunState.INPUT_REQUIRED.value,
        )
        self.assertEqual(
            missing_database.records[
                str(missing_interaction.record.request.request_id)
            ]["request_state"],
            RequestState.PENDING.value,
        )

        rollback_database = FakePgsqlDatabase()
        rollback_store = await _store(rollback_database)
        rollback_task_store = PgsqlTaskStore(
            rollback_database,
            clock=lambda: _NOW,
        )
        rollback_interaction = await _create_suspended_task(
            rollback_database,
            rollback_store,
            rollback_task_store,
            run_id="run-client-rollback",
            suffix="client-rollback",
        )
        rollback_coordinator = PgsqlDurableTaskCoordinator(
            rollback_store,
            rollback_task_store,
        )
        rollback_client = TaskClient(
            rollback_task_store,
            target=_unused_task_target,
            durable_lifecycle_coordinator=rollback_coordinator,
            clock=lambda: _NOW,
        )
        rollback_database.fail_query = (
            'UPDATE "task_queue_items"\nSET\n'
            '    "state" = %s,\n    "claimed_at" = NULL'
        )
        with self.assertRaises(Exception):
            await rollback_client.cancel("run-client-rollback")
        self.assertEqual(
            rollback_database.runs["run-client-rollback"]["state"],
            TaskRunState.INPUT_REQUIRED.value,
        )
        self.assertEqual(
            rollback_database.records[
                str(rollback_interaction.record.request.request_id)
            ]["request_state"],
            RequestState.PENDING.value,
        )
        self.assertEqual(
            rollback_database.continuations[
                str(rollback_interaction.record.request.continuation_id)
            ]["lifecycle_state"],
            DurableContinuationLifecycle.PENDING.value,
        )

        rollback_database.fail_query = None
        cancelled = await rollback_client.cancel("run-client-rollback")
        replayed = await rollback_client.cancel("run-client-rollback")
        self.assertEqual(cancelled.state, TaskRunState.CANCELLED)
        self.assertEqual(replayed, cancelled)
        self.assertEqual(
            rollback_database.attempts["attempt-client-rollback"]["state"],
            TaskAttemptState.ABANDONED.value,
        )
        self.assertEqual(
            rollback_database.queue_items["queue-client-rollback"]["state"],
            TaskQueueItemState.DEAD.value,
        )
        self.assertEqual(
            rollback_database.records[
                str(rollback_interaction.record.request.request_id)
            ]["request_state"],
            RequestState.CANCELLED.value,
        )
        self.assertEqual(
            rollback_database.continuations[
                str(rollback_interaction.record.request.continuation_id)
            ]["lifecycle_state"],
            DurableContinuationLifecycle.INVALIDATED.value,
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in rollback_database.events.values()
            ].count(TaskInteractionEventType.INPUT_CANCELLED.value),
            1,
        )

    async def test_sweep_deletes_all_encrypted_payload_copies(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(
            database,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        request = _request("sweep")
        created = await store.create_durable(
            _create_command(request),
            _portable(request),
        )
        assert isinstance(created, CreateInteractionApplied)
        expired = await store.sweep(now=_NOW + timedelta(minutes=11))
        self.assertEqual(expired.invalidated, (request.continuation_id,))
        deleted = await store.sweep(now=_NOW + timedelta(days=2))
        self.assertEqual(deleted.deleted, (request.continuation_id,))
        self.assertEqual(database.records, {})
        self.assertEqual(database.continuations, {})
        self.assertEqual(database.outbox, {})
        self.assertEqual(database.resolution_keys, {})

    async def test_sweep_deletes_continuationless_record_and_branch(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        cipher = _Cipher()
        store = await _store(
            database,
            cipher=cipher,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        request = _request("continuationless-retention")
        created = await store.create(_create_command(request))
        assert isinstance(created, CreateInteractionApplied)
        await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=created.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=request.origin.run_id,
                    branch_id=BranchId("retained-child"),
                    parent_branch_id=request.origin.branch_id,
                    principal=request.origin.principal,
                ),
            )
        )
        self.assertEqual(database.continuations, {})
        self.assertEqual(len(database.records), 1)
        self.assertEqual(len(database.branches), 1)
        record_row = database.records[str(request.request_id)]
        branch_row = next(iter(database.branches.values()))
        self.assertNotIn(
            request.reason.encode(),
            cast(bytes, record_row["ciphertext"]),
        )
        branch_plaintext = cipher.decrypt(
            EncryptedPrivacyValue(
                ciphertext=cast(bytes, branch_row["ciphertext"]),
                key_id=cast(str, branch_row["encryption_key_id"]),
                algorithm=cast(str, branch_row["encryption_algorithm"]),
                metadata=cast(
                    Mapping[str, str],
                    branch_row["encryption_metadata"],
                ),
            ),
            purpose=TaskKeyPurpose.RAW_VALUE,
            context=cast(
                Mapping[str, str],
                branch_row["encryption_metadata"],
            ),
        )
        self.assertIn(b"owner", branch_plaintext)

        swept = await store.sweep(now=_NOW + timedelta(days=2))

        self.assertEqual(swept.invalidated, ())
        self.assertEqual(swept.deleted, ())
        self.assertEqual(database.records, {})
        self.assertEqual(database.branches, {})
        self.assertEqual(database.continuations, {})

    async def test_encryption_and_dependency_failures_are_content_safe(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        cipher = _Cipher()
        store = await _store(database, cipher=cipher)
        request = _request("encryption")
        cipher.fail_encrypt = True
        with self.assertRaises(
            PgsqlInteractionFeatureUnavailableError
        ) as error:
            await store.create(_create_command(request))
        self.assertIn("avalan[task-pgsql]", str(error.exception))
        self.assertNotIn(request.reason, str(error.exception))

        with self.assertRaises(PgsqlInteractionFeatureUnavailableError):
            require_interaction_pgsql_dependencies(
                module_finder=lambda _: None
            )

    def test_durable_suspension_rejects_untrusted_or_drifted_values(
        self,
    ) -> None:
        request = _request("suspension-validation")
        command = _create_command(request)
        continuation = _portable(request)
        suspension = DurableInteractionSuspension(
            command=command,
            continuation=continuation,
        )
        self.assertIs(suspension.command, command)
        self.assertIs(suspension.continuation, continuation)

        invalid_values = (
            (
                "command",
                cast(Any, object()),
                continuation,
                "durable_suspension.command",
            ),
            (
                "continuation",
                command,
                cast(Any, object()),
                "durable_suspension.continuation",
            ),
            (
                "resumer",
                replace(command, resumer=_Resumer()),
                continuation,
                "durable_suspension.command.resumer",
            ),
            (
                "request_id",
                command,
                replace(
                    continuation,
                    request_id=InputRequestId("other-request"),
                ),
                "durable_suspension.request_id",
            ),
            (
                "continuation_id",
                command,
                replace(
                    continuation,
                    continuation_id=ContinuationId("other-continuation"),
                ),
                "durable_suspension.continuation_id",
            ),
            (
                "origin",
                command,
                replace(continuation, origin=_origin("other-run")),
                "durable_suspension.origin",
            ),
            (
                "principal",
                replace(
                    command,
                    actor=InteractionActor(
                        principal=PrincipalScope(user_id=UserId("other"))
                    ),
                ),
                continuation,
                "durable_suspension.principal",
            ),
            (
                "expires_at",
                command,
                replace(
                    continuation,
                    expires_at=continuation.expires_at + timedelta(seconds=1),
                ),
                "durable_suspension.expires_at",
            ),
            (
                "persisted_state",
                command,
                replace(
                    continuation,
                    store_revision=ContinuationStoreRevision(1),
                ),
                "durable_suspension.continuation",
            ),
        )
        for (
            name,
            candidate_command,
            candidate_continuation,
            path,
        ) in invalid_values:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    InputValidationError,
                    path,
                ),
            ):
                DurableInteractionSuspension(
                    command=candidate_command,
                    continuation=candidate_continuation,
                )

        drifted_definition = _portable(request)
        object.__setattr__(
            drifted_definition,
            "definition",
            replace(
                drifted_definition.definition,
                operation_id="other-operation",
            ),
        )
        drifted_provider_call = _portable(request)
        object.__setattr__(
            drifted_provider_call,
            "provider_call_id",
            ModelCallId("other-call"),
        )
        for name, candidate, path in (
            (
                "definition",
                drifted_definition,
                "durable_suspension.definition",
            ),
            (
                "provider_call_id",
                drifted_provider_call,
                "durable_suspension.provider_call_id",
            ),
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    InputValidationError,
                    path,
                ),
            ):
                DurableInteractionSuspension(
                    command=command,
                    continuation=candidate,
                )

    def test_pgsql_payload_helpers_reject_every_malformed_shape(self) -> None:
        invalid_calls = (
            (
                interaction_pgsql._decode_record,
                (
                    interaction_pgsql._canonical_bytes(
                        {"idempotency_ledger": {}}
                    ),
                ),
            ),
            (
                interaction_pgsql._decode_record,
                (
                    interaction_pgsql._canonical_bytes(
                        {"idempotency_ledger": [1]}
                    ),
                ),
            ),
            (interaction_pgsql._decode_advisory, (1,)),
            (interaction_pgsql._encode_resolver, (object(),)),
            (interaction_pgsql._decode_resolver, (1,)),
            (
                interaction_pgsql._decode_resolver,
                ({"kind": "unknown"},),
            ),
            (interaction_pgsql._principal_from_payload, (1,)),
            (
                interaction_pgsql._continuation_binding,
                ({"revision_binding": 1},),
            ),
            (interaction_pgsql._object_payload, (b"[]", "payload")),
            (interaction_pgsql._row_str, ({}, "value")),
            (
                interaction_pgsql._row_optional_str,
                ({"value": 1}, "value"),
            ),
            (interaction_pgsql._row_int, ({"value": True}, "value")),
            (interaction_pgsql._row_bool, ({"value": 1}, "value")),
            (interaction_pgsql._row_bytes, ({"value": b""}, "value")),
            (
                interaction_pgsql._row_string_mapping,
                ({"value": 1}, "value"),
            ),
            (
                interaction_pgsql._row_string_mapping,
                ({"value": {1: "value"}}, "value"),
            ),
            (
                interaction_pgsql._mapping_str,
                ({"value": 1}, "value"),
            ),
            (
                interaction_pgsql._mapping_optional_str,
                ({"value": 1}, "value"),
            ),
            (
                interaction_pgsql._mapping_int,
                ({"value": True}, "value"),
            ),
            (
                interaction_pgsql._mapping_optional_int,
                ({"value": True}, "value"),
            ),
            (
                interaction_pgsql._mapping_number,
                ({"value": True}, "value"),
            ),
            (
                interaction_pgsql._mapping_optional_number,
                ({"value": True}, "value"),
            ),
            (
                interaction_pgsql._payload_datetime,
                ({"value": "not-a-date"}, "value"),
            ),
            (
                interaction_pgsql._mapping_optional_datetime,
                ({"value": "not-a-date"}, "value"),
            ),
        )
        for callback, arguments in invalid_calls:
            with (
                self.subTest(callback=callback.__name__, arguments=arguments),
                self.assertRaises(InputContractError),
            ):
                callback(*cast(Any, arguments))

        self.assertEqual(
            interaction_pgsql._row_bytes(
                {"value": memoryview(b"value")},
                "value",
            ),
            b"value",
        )
        self.assertEqual(
            interaction_pgsql._row_string_mapping(
                {"value": '{"key":"value"}'},
                "value",
            ),
            {"key": "value"},
        )
        for callback, arguments in (
            (interaction_pgsql._assert_opaque, ("", "value")),
            (interaction_pgsql._assert_digest, ("bad", "digest")),
            (interaction_pgsql._assert_limit, (0,)),
        ):
            with (
                self.subTest(callback=callback.__name__),
                self.assertRaises(InputValidationError),
            ):
                callback(*cast(Any, arguments))
        with self.assertRaises(PgsqlInteractionStoreError):
            interaction_pgsql._scope_identity_conflict("record")

    def test_pgsql_branch_graph_and_replay_helpers_fail_closed(self) -> None:
        principal = PrincipalScope(user_id=UserId("owner"))
        first = InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("first"),
                parent_branch_id=BranchId("second"),
                principal=principal,
            ),
            store_revision=InteractionStoreRevision(0),
        )
        second = InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("second"),
                parent_branch_id=BranchId("first"),
                principal=principal,
            ),
            store_revision=InteractionStoreRevision(0),
        )
        with self.assertRaises(PgsqlInteractionStoreError):
            interaction_pgsql._branch_roots((first, second))

        root_mismatch = {
            "root_branch_id": "wrong-root",
        }
        self.assertEqual(
            interaction_pgsql._branches_with_valid_row_roots(
                ((root_mismatch, first),),
                tolerate_invalid=True,
            ),
            (),
        )

        continuation = _portable(_request())
        exhausted = replace(
            continuation,
            store_revision=ContinuationStoreRevision(
                interaction_pgsql.MAX_STATE_REVISION
            ),
        )
        with self.assertRaises(PgsqlInteractionStoreError):
            interaction_pgsql._next_continuation_revision(exhausted)

        base_row = {
            "store_revision": int(continuation.store_revision),
            "fencing_token": int(continuation.fencing_token),
            "claim_owner_id": None,
        }
        for row in (
            {
                **base_row,
                "store_revision": int(continuation.store_revision) + 1,
                "lifecycle_state": (
                    DurableContinuationLifecycle.INVALIDATED.value
                ),
                "invalid_reason": "task_cancelled",
            },
            {
                **base_row,
                "lifecycle_state": (
                    DurableContinuationLifecycle.INVALIDATED.value
                ),
                "invalid_reason": "task_cancelled",
            },
            {
                **base_row,
                "lifecycle_state": DurableContinuationLifecycle.READY.value,
            },
        ):
            with (
                self.subTest(row=row),
                self.assertRaises(TaskStoreConflictError),
            ):
                interaction_pgsql._validate_cancelled_continuation_replay(
                    row,
                    continuation,
                )

    async def test_pgsql_persisted_identity_helpers_detect_drift(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("identity")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        record_row = deepcopy(database.records[str(request.request_id)])
        record_row["request_id"] = "wrong-request"
        with self.assertRaises(PgsqlInteractionStoreError):
            interaction_pgsql._validate_record_row_identity(
                record_row,
                created.record,
            )

        branch = InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("child"),
                parent_branch_id=BranchId("root"),
                principal=request.origin.principal,
            ),
            store_revision=InteractionStoreRevision(0),
        )
        digest = _scope_identity_digest(
            branch.registration.run_id,
            branch.registration.principal,
        )
        branch_row = {
            "run_id": "run",
            "branch_id": "wrong-child",
            "parent_branch_id": "root",
            "store_revision": 0,
            "scope_identity_digest": digest,
        }
        with self.assertRaises(PgsqlInteractionStoreError):
            interaction_pgsql._validate_branch_row_identity(
                branch_row,
                branch,
            )

    async def test_pgsql_coordinator_rejects_invalid_public_contracts(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        request = _request()
        command = _create_command(request)
        continuation = _portable(request)
        resumable_command = replace(command, resumer=_Resumer())

        other_database = FakePgsqlDatabase()
        other_task_store = PgsqlTaskStore(
            other_database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("other-task"),
        )
        with self.assertRaisesRegex(ValueError, "share one database"):
            PgsqlDurableTaskCoordinator(store, other_task_store)

        create_and_suspend_cases = (
            (object(), continuation, command, "run"),
            (command, object(), command, "run"),
            (resumable_command, continuation, command, "run"),
            (command, continuation, command, "other-run"),
        )
        for (
            candidate_command,
            candidate_continuation,
            _,
            task_run_id,
        ) in create_and_suspend_cases:
            with (
                self.subTest(
                    method="create_and_suspend",
                    command=candidate_command,
                    continuation=candidate_continuation,
                    task_run_id=task_run_id,
                ),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.create_and_suspend(
                    cast(CreateInteractionCommand, candidate_command),
                    cast(PortableContinuation, candidate_continuation),
                    queue_item_id="queue",
                    claim_token="claim",
                    segment_id="segment",
                    task_run_id=task_run_id,
                    checkpoint_id="checkpoint",
                )

        for candidate_command, candidate_continuation, task_run_id in (
            (object(), continuation, "run"),
            (command, object(), "run"),
            (resumable_command, continuation, "run"),
            (command, continuation, "other-run"),
        ):
            with (
                self.subTest(
                    method="create_pending_interaction",
                    command=candidate_command,
                    continuation=candidate_continuation,
                    task_run_id=task_run_id,
                ),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.create_pending_interaction(
                    cast(CreateInteractionCommand, candidate_command),
                    cast(PortableContinuation, candidate_continuation),
                    task_run_id=task_run_id,
                    checkpoint_id="checkpoint",
                )

        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"code": "failure"})
        )
        digest = task_durable_resume_settlement_digest(failure)
        completion = ContinuationCompletionCommand(
            continuation_id=continuation.continuation_id,
            expected_store_revision=ContinuationStoreRevision(0),
            owner_id=ContinuationClaimOwnerId("claim"),
            fencing_token=ContinuationFencingToken(0),
            result_digest=digest,
        )
        for (
            candidate_completion,
            candidate_command,
            candidate_continuation,
        ) in (
            (object(), command, continuation),
            (completion, object(), continuation),
            (completion, command, object()),
            (completion, resumable_command, continuation),
        ):
            with (
                self.subTest(
                    method="complete_and_resuspend",
                    completion=candidate_completion,
                    command=candidate_command,
                    continuation=candidate_continuation,
                ),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.complete_and_resuspend(
                    cast(ContinuationCompletionCommand, candidate_completion),
                    cast(CreateInteractionCommand, candidate_command),
                    cast(PortableContinuation, candidate_continuation),
                    queue_item_id="queue",
                    claim_token="claim",
                    segment_id="segment",
                    task_run_id="run",
                    checkpoint_id="checkpoint",
                )

        with self.assertRaises(InputValidationError):
            await coordinator.settle_resume(
                cast(ContinuationCompletionCommand, object()),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.settle_resume(
                replace(completion, result_digest="0" * 64),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
            )

        with self.assertRaises(InputValidationError):
            await coordinator.terminalize_completed_resume(
                cast(ContinuationCompletionCommand, object()),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
                request_id="request",
                checkpoint_id="checkpoint",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.terminalize_completed_resume(
                completion,
                cast(TaskDurableResumeFailure, object()),
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
                request_id="request",
                checkpoint_id="checkpoint",
            )

        with self.assertRaises(InputValidationError):
            await coordinator.mark_resume_ambiguous(
                cast(ContinuationCompletionCommand, object()),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.mark_resume_ambiguous(
                completion,
                cast(TaskDurableResumeFailure, object()),
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.mark_resume_ambiguous(
                replace(completion, result_digest="0" * 64),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                segment_id="segment",
                task_run_id="run",
            )

        result = TaskExecutionResult(error={"code": "failure"})
        with self.assertRaises(InputValidationError):
            await coordinator.fail_claimed_reentry(
                queue_item_id="queue",
                claim_token="claim",
                task_run_id="run",
                request_id="request",
                continuation_id=None,
                checkpoint_id=None,
                result=result,
                reason="failure",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.fail_claimed_reentry(
                queue_item_id="queue",
                claim_token="claim",
                task_run_id="run",
                request_id="request",
                continuation_id="continuation",
                checkpoint_id="checkpoint",
                result=cast(TaskExecutionResult, object()),
                reason="failure",
            )

        rejection = ContinuationRejectionCommand(
            continuation_id=continuation.continuation_id,
            expected_store_revision=ContinuationStoreRevision(0),
            owner_id=ContinuationClaimOwnerId("claim"),
            fencing_token=ContinuationFencingToken(0),
            result_digest=digest,
        )
        with self.assertRaises(InputValidationError):
            await coordinator.fail_admitted_reentry(
                cast(ContinuationRejectionCommand, object()),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                task_run_id="run",
                request_id="request",
                continuation_id=str(continuation.continuation_id),
                checkpoint_id="checkpoint",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.fail_admitted_reentry(
                rejection,
                cast(TaskDurableResumeFailure, object()),
                queue_item_id="queue",
                claim_token="claim",
                task_run_id="run",
                request_id="request",
                continuation_id=str(continuation.continuation_id),
                checkpoint_id="checkpoint",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.fail_admitted_reentry(
                replace(rejection, result_digest="0" * 64),
                failure,
                queue_item_id="queue",
                claim_token="claim",
                task_run_id="run",
                request_id="request",
                continuation_id=str(continuation.continuation_id),
                checkpoint_id="checkpoint",
            )
        with self.assertRaises(InputValidationError):
            await coordinator.fail_admitted_reentry(
                rejection,
                failure,
                queue_item_id="queue",
                claim_token="other-claim",
                task_run_id="run",
                request_id="request",
                continuation_id=str(continuation.continuation_id),
                checkpoint_id="checkpoint",
            )

        with self.assertRaises(InputValidationError):
            await coordinator.reconcile_expired_reentry(
                queue_item_id="queue",
                expected_claim_token="claim",
                task_run_id="run",
                result=cast(TaskExecutionResult, object()),
            )

    async def test_pgsql_coordinator_scope_entry_validation(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        request = _request()
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        resolve = _answer(created)
        actor = created.command.actor
        scope = InteractionExecutionScope(run_id=RunId("run"))
        cancel = TerminalizeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
        )
        supersede = SupersedeInteractionScopeCommand(
            actor=actor,
            scope=scope,
            provenance=AnswerProvenance.HUMAN,
        )

        for candidate, task_run_id in (
            (object(), "run"),
            (resolve, "other-run"),
        ):
            with (
                self.subTest(method="resolve", candidate=candidate),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.resolve_and_requeue(
                    cast(ResolveInteractionCommand, candidate),
                    task_run_id=task_run_id,
                )
        for candidate, task_run_id in (
            (object(), "run"),
            (cancel, "other-run"),
        ):
            with (
                self.subTest(method="cancel", candidate=candidate),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.cancel_suspended_task(
                    cast(TerminalizeInteractionScopeCommand, candidate),
                    task_run_id=task_run_id,
                )
        for candidate, task_run_id in (
            (object(), "run"),
            (supersede, "other-run"),
        ):
            with (
                self.subTest(method="supersede", candidate=candidate),
                self.assertRaises(InputValidationError),
            ):
                await coordinator.supersede_suspended_task(
                    cast(SupersedeInteractionScopeCommand, candidate),
                    task_run_id=task_run_id,
                )
        with self.assertRaises(InputValidationError):
            await coordinator.expire_suspended_task(
                cast(TerminalizeDueInteractionsCommand, object()),
                task_run_id="run",
            )

        lifecycle = SimpleNamespace(
            completion_for=lambda _task_run_id: None,
        )
        with (
            patch.object(
                PgsqlDurableTaskCoordinator,
                "_terminalize_task_lifecycle",
                new=AsyncMock(return_value=lifecycle),
            ),
            self.assertRaisesRegex(TaskStoreError, "lost"),
        ):
            await coordinator.cancel_input_required_task(
                task_run_id="run",
                now=_NOW,
                metadata={},
            )

    async def test_pgsql_standalone_durable_creation_rejects_task_binding(
        self,
    ) -> None:
        policy = PgsqlInteractionStorePolicy(encryption_key_id="key-id")
        self.assertEqual(policy.encryption_key_id, "key-id")
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request()
        command = _create_command(request)
        continuation = _portable(request)

        for task_run_id, checkpoint_id, path in (
            ("run", None, "task_run_id"),
            (None, "checkpoint", "checkpoint_id"),
        ):
            with (
                self.subTest(
                    task_run_id=task_run_id,
                    checkpoint_id=checkpoint_id,
                ),
                self.assertRaisesRegex(
                    InputContractError,
                    path,
                ),
            ):
                await store.create_durable(
                    command,
                    continuation,
                    task_run_id=task_run_id,
                    checkpoint_id=checkpoint_id,
                )

        for field_name, drifted in (
            (
                "request_id",
                replace(
                    continuation,
                    request_id=InputRequestId("other-request"),
                ),
            ),
            (
                "continuation_id",
                replace(
                    continuation,
                    continuation_id=ContinuationId("other-continuation"),
                ),
            ),
            (
                "origin",
                replace(
                    continuation,
                    origin=replace(
                        continuation.origin,
                        run_id=RunId("other-run"),
                    ),
                ),
            ),
        ):
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(InputValidationError, field_name),
            ):
                await store.create_durable(command, drifted)

        capacity = interaction_pgsql._PgsqlCreateCapacityError(command)
        with patch.object(
            PgsqlInteractionStore,
            "_run_memory_with_unit",
            new=AsyncMock(side_effect=capacity),
        ):
            self.assertIs(
                await store.create_durable(command, continuation),
                capacity.rejection,
            )

        admission, _ = _new_interaction_admission_commands(
            actor=command.actor,
            request=request,
            resumer=_Resumer(),
        )
        admission_capacity = interaction_pgsql._PgsqlCreateCapacityError(
            admission._command
        )
        with patch.object(
            PgsqlInteractionStore,
            "_run_memory_with_unit",
            new=AsyncMock(side_effect=admission_capacity),
        ):
            self.assertIs(
                await store.create_admission(admission),
                admission_capacity.rejection,
            )

    async def test_pgsql_public_wrappers_preserve_exact_snapshot_scope(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("wrapper")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        command = SimpleNamespace(
            actor=created.command.actor,
            correlation=created.record.correlation,
        )

        with patch.object(
            PgsqlInteractionStore,
            "_run_memory",
            new=AsyncMock(return_value="detached"),
        ) as run_memory:
            self.assertEqual(
                await store.mark_detached(cast(Any, command)),
                "detached",
            )
        self.assertEqual(
            run_memory.await_args.args[0],
            "interaction_mark_detached",
        )

        with patch.object(
            PgsqlInteractionStore,
            "_terminal_memory_operation",
            new=AsyncMock(return_value="defaulted"),
        ) as terminal:
            self.assertEqual(
                await store.resolve_trusted_default(cast(Any, command)),
                "defaulted",
            )
        self.assertEqual(
            terminal.await_args.args[0],
            "interaction_resolve_default",
        )

    async def test_pgsql_claim_and_renewal_reject_invalid_lifecycles(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("claim-guards")
        continuation = _portable(request)
        created = await store.create_durable(
            _create_command(request),
            continuation,
        )
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)

        claim_arguments = {
            "continuation_id": continuation.continuation_id,
            "expected_store_revision": continuation.store_revision,
            "owner_id": ContinuationClaimOwnerId("owner"),
            "lease_expires_at": _NOW + timedelta(minutes=2),
            "dispatch_id": ContinuationDispatchId("dispatch"),
            "provider_idempotency_key": ProviderIdempotencyKey("provider-key"),
            "now": _NOW + timedelta(seconds=1),
        }
        with self.assertRaises(InputValidationError):
            await store.claim(
                **{
                    **claim_arguments,
                    "lease_expires_at": _NOW + timedelta(seconds=1),
                }
            )
        with self.assertRaises(ContinuationStoreConflictError):
            await store.claim(**claim_arguments)

        resolved = await store.resolve(_answer(created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        ready = await store.get_continuation(continuation.continuation_id)
        ready_arguments = {
            **claim_arguments,
            "expected_store_revision": ready.store_revision,
        }
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.claim(
                **{
                    **ready_arguments,
                    "now": ready.expires_at,
                    "lease_expires_at": (
                        ready.expires_at + timedelta(seconds=1)
                    ),
                }
            )
        with self.assertRaises(InputValidationError):
            await store.claim(
                **{
                    **ready_arguments,
                    "lease_expires_at": (
                        ready.expires_at + timedelta(seconds=1)
                    ),
                }
            )

        row = database.continuations[str(continuation.continuation_id)]
        row["lifecycle_state"] = DurableContinuationLifecycle.DISPATCHING.value
        with self.assertRaises(ContinuationDispatchAmbiguousError):
            await store.claim(**ready_arguments)
        database.continuations[str(continuation.continuation_id)][
            "lifecycle_state"
        ] = DurableContinuationLifecycle.READY.value

        receipt = await store.claim(**ready_arguments)
        renew_arguments = {
            "continuation_id": continuation.continuation_id,
            "expected_store_revision": receipt.continuation.store_revision,
            "owner_id": ContinuationClaimOwnerId("owner"),
            "fencing_token": receipt.fencing_token,
            "now": _NOW + timedelta(seconds=2),
        }
        with self.assertRaises(InputValidationError):
            await store.renew_claim(
                **renew_arguments,
                lease_expires_at=_NOW + timedelta(seconds=2),
            )
        with self.assertRaises(InputValidationError):
            await store.renew_claim(
                **renew_arguments,
                lease_expires_at=ready.expires_at + timedelta(seconds=1),
            )
        with (
            patch.object(
                PgsqlInteractionStore,
                "_transaction",
                new=AsyncMock(return_value=object()),
            ),
            self.assertRaises(PgsqlInteractionStoreError),
        ):
            await store.renew_claim(
                **renew_arguments,
                lease_expires_at=_NOW + timedelta(minutes=3),
            )

    async def test_pgsql_invalidation_and_task_lookup_fail_closed(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("invalidate")
        continuation = _portable(request)
        created = await store.create_durable(
            _create_command(request),
            continuation,
        )
        self.assertIsInstance(created, CreateInteractionApplied)
        row = database.continuations[str(continuation.continuation_id)]
        invalidate_arguments = {
            "continuation_id": continuation.continuation_id,
            "expected_store_revision": continuation.store_revision,
            "reason": "cancelled",
            "now": _NOW + timedelta(seconds=1),
        }

        row["task_run_id"] = "run"
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.invalidate(**invalidate_arguments)
        database.continuations[str(continuation.continuation_id)][
            "task_run_id"
        ] = None

        with self.assertRaises(ContinuationStoreConflictError):
            await store.invalidate(
                **{
                    **invalidate_arguments,
                    "expected_store_revision": ContinuationStoreRevision(1),
                }
            )

        database.continuations[str(continuation.continuation_id)][
            "lifecycle_state"
        ] = DurableContinuationLifecycle.COMPLETED.value
        with self.assertRaises(ContinuationStoreConflictError):
            await store.invalidate(**invalidate_arguments)
        database.continuations[str(continuation.continuation_id)][
            "lifecycle_state"
        ] = DurableContinuationLifecycle.DISPATCHING.value
        with self.assertRaises(ContinuationDispatchAmbiguousError):
            await store.invalidate(**invalidate_arguments)
        database.continuations[str(continuation.continuation_id)][
            "lifecycle_state"
        ] = DurableContinuationLifecycle.PENDING.value

        invalidated = await store.invalidate(**invalidate_arguments)
        self.assertEqual(
            invalidated.claim.state,
            ContinuationClaimState.UNCLAIMED,
        )
        self.assertEqual(
            database.continuations[str(continuation.continuation_id)][
                "lifecycle_state"
            ],
            DurableContinuationLifecycle.INVALIDATED.value,
        )

        for suffix in ("first", "second"):
            other_request = _request(f"ambiguous-{suffix}")
            other_continuation = _portable(other_request)
            await store.create_durable(
                _create_command(other_request),
                other_continuation,
            )
            database.continuations[str(other_continuation.continuation_id)][
                "task_run_id"
            ] = "ambiguous-run"
        with self.assertRaises(ContinuationStoreConflictError):
            await store.get_task_continuation_record("ambiguous-run")

    async def test_pgsql_private_transaction_boundaries_preserve_errors(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)

        async def cancel_callback(
            _memory: object,
            _unit: object,
        ) -> object:
            raise CancelledError()

        with self.assertRaises(CancelledError):
            await store._run_memory_with_unit(
                "cancelled_callback",
                cast(Any, cancel_callback),
                mutate=False,
            )

        callback_error = ValueError("callback failed")

        async def fail_callback(
            _memory: object,
            _unit: object,
        ) -> object:
            raise callback_error

        with self.assertRaises(ValueError) as raised:
            await store._run_memory_with_unit(
                "failed_callback",
                cast(Any, fail_callback),
                mutate=False,
            )
        self.assertIs(raised.exception, callback_error)

        async def raise_store_error(_unit: PgsqlUnitOfWork) -> object:
            raise PgsqlInteractionStoreError(
                InputErrorCode.UNAVAILABLE,
                "transaction",
                "expected store error",
            )

        with self.assertRaises(PgsqlInteractionStoreError):
            await store._transaction(
                "store_error",
                raise_store_error,
            )

        metadata_unit = _unit(row=None)

        async def direct_transaction(
            _self: PgsqlInteractionStore,
            _operation: str,
            callback: Callable[[PgsqlUnitOfWork], Awaitable[object]],
            **_kwargs: object,
        ) -> object:
            return await callback(metadata_unit)

        async def unused_read(
            _memory: object,
            _records: object,
        ) -> object:
            raise AssertionError("missing metadata must stop the read")

        with (
            patch.object(
                PgsqlInteractionStore,
                "_transaction",
                new=direct_transaction,
            ),
            self.assertRaises(PgsqlInteractionFeatureUnavailableError),
        ):
            await store._run_memory_read_snapshot(
                "missing_metadata",
                cast(Any, unused_read),
            )

    async def test_pgsql_private_selection_and_capacity_guards(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        mismatch_unit = _unit(
            rows=(
                {
                    "run_id": "other-run",
                    "scope_identity_digest": "d" * 64,
                },
            )
        )
        with self.assertRaises(PgsqlInteractionStoreError):
            await store._resolve_snapshot_selection(
                mismatch_unit,
                interaction_pgsql._trusted_task_snapshot("run"),
            )

        request = _request("private-guards")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        capacity_unit = _unit(row=None)
        with self.assertRaises(PgsqlInteractionFeatureUnavailableError):
            await store._enforce_create_process_capacity(
                capacity_unit,
                created.command,
                created,
            )

        continuation = _portable(request)
        drifted = replace(
            continuation,
            request_id=InputRequestId("other-request"),
        )
        continuation_unit = _unit(row={"continuation_id": "opaque"})
        with patch.object(
            PgsqlInteractionStore,
            "_continuation_from_row",
            return_value=drifted,
        ):
            self.assertEqual(
                await store._records_with_valid_continuations(
                    continuation_unit,
                    (created.record,),
                    for_update=False,
                ),
                (),
            )

    async def test_pgsql_private_continuation_persistence_guards(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("private-continuation")
        continuation = _portable(request)

        with self.assertRaises(InteractionNotFoundError):
            await store._insert_continuation(
                _unit(row=None),
                continuation,
                task_run_id=None,
                checkpoint_id=None,
            )
        with self.assertRaises(InputValidationError):
            await store._insert_continuation(
                _unit(
                    row={
                        "absolute_expires_at": (
                            continuation.expires_at + timedelta(seconds=1)
                        ),
                    }
                ),
                continuation,
                task_run_id=None,
                checkpoint_id=None,
            )

        conflict_unit = _unit()
        conflict_unit.cursor.fetchone.side_effect = (
            {"absolute_expires_at": continuation.expires_at},
            None,
        )
        with self.assertRaises(ContinuationStoreConflictError):
            await store._insert_continuation(
                conflict_unit,
                continuation,
                task_run_id=None,
                checkpoint_id=None,
            )

        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        resolved = await store.resolve(_answer(created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        answered = cast(ResolveInteractionApplied, resolved).record
        for lifecycle, error_type in (
            (DurableContinuationLifecycle.READY, None),
            (
                DurableContinuationLifecycle.INVALIDATED,
                ContinuationStoreConflictError,
            ),
        ):
            unit = _unit(row={"lifecycle_state": lifecycle.value})
            with patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ):
                if error_type is None:
                    await store._ready_continuation(unit, answered)
                else:
                    with self.assertRaises(error_type):
                        await store._ready_continuation(unit, answered)

        for lifecycle, error_type in (
            (DurableContinuationLifecycle.COMPLETED, None),
            (
                DurableContinuationLifecycle.DISPATCHING,
                ContinuationDispatchAmbiguousError,
            ),
        ):
            unit = _unit(row={"lifecycle_state": lifecycle.value})
            if error_type is None:
                await store._invalidate_for_record(unit, answered)
            else:
                with self.assertRaises(error_type):
                    await store._invalidate_for_record(unit, answered)

        stale_unit = _unit(row=None)
        with self.assertRaises(ContinuationStoreConflictError):
            await store._update_continuation(
                stale_unit,
                continuation,
                lifecycle=DurableContinuationLifecycle.PENDING,
                expected_revision=continuation.store_revision,
            )

    async def test_pgsql_private_claim_and_snapshot_guards(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("private-claim")
        continuation = _portable(request)
        created = await store.create_durable(
            _create_command(request),
            continuation,
        )
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        resolved = await store.resolve(_answer(created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        ready = await store.get_continuation(continuation.continuation_id)
        ready_row = deepcopy(
            database.continuations[str(continuation.continuation_id)]
        )

        with self.assertRaises(ContinuationStoreConflictError):
            store._claimed_continuation(
                ready_row,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("owner"),
                fencing_token=ContinuationFencingToken(0),
                now=_NOW + timedelta(seconds=1),
            )

        receipt = await store.claim(
            continuation.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId("owner"),
            lease_expires_at=_NOW + timedelta(minutes=2),
            dispatch_id=ContinuationDispatchId("dispatch"),
            provider_idempotency_key=ProviderIdempotencyKey("provider-key"),
            now=_NOW + timedelta(seconds=1),
        )
        claimed_row = database.continuations[str(continuation.continuation_id)]
        with self.assertRaises(ContinuationStoreConflictError):
            store._claimed_continuation(
                claimed_row,
                expected_store_revision=receipt.continuation.store_revision,
                owner_id=ContinuationClaimOwnerId("owner"),
                fencing_token=receipt.fencing_token,
                now=_NOW + timedelta(minutes=3),
            )

        with self.assertRaises(ContinuationStoreConflictError):
            store._ambiguous_continuation(
                ready_row,
                expected_store_revision=ContinuationStoreRevision(
                    int(ready.store_revision) + 1
                ),
                owner_id=ContinuationClaimOwnerId("owner"),
                fencing_token=ContinuationFencingToken(0),
            )
        with self.assertRaises(ContinuationStoreConflictError):
            store._ambiguous_continuation(
                ready_row,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("owner"),
                fencing_token=ContinuationFencingToken(0),
            )

        selection = interaction_pgsql._InteractionSnapshotSelection(
            run_id=RunId("run"),
            scope_identity_digest="d" * 64,
        )
        empty_unit = _unit(rows=())
        self.assertEqual(
            await store._load_records(empty_unit, selection=selection),
            (),
        )
        self.assertEqual(
            await store._load_branches(empty_unit, selection=selection),
            (),
        )

    async def test_pgsql_private_row_persistence_and_crypto_guards(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        cipher = _Cipher()
        store = await _store(database, cipher=cipher)
        request = _request("private-row")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        with self.assertRaises(PgsqlInteractionStoreError):
            await store._save_records(_unit(row=None), (created.record,))

        branch = InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=RunId("run"),
                branch_id=BranchId("child"),
                parent_branch_id=BranchId("root"),
                principal=request.origin.principal,
            ),
            store_revision=InteractionStoreRevision(0),
        )
        roots = interaction_pgsql._branch_roots((branch,))
        with self.assertRaises(PgsqlInteractionStoreError):
            await store._save_branches(
                _unit(row=None),
                (branch,),
                roots=roots,
            )

        with (
            patch.object(
                PgsqlInteractionStore,
                "_decrypt",
                return_value=b"[]",
            ),
            self.assertRaises(PgsqlInteractionStoreError),
        ):
            store._continuation_from_row({"continuation_id": "continuation"})

        with (
            patch.object(_Cipher, "encrypt", return_value=object()),
            self.assertRaises(PgsqlInteractionFeatureUnavailableError),
        ):
            store._encrypt(b"value", kind="record", identifier="request")

        encrypted_row = {
            "ciphertext": b"value",
            "encryption_key_id": "key",
            "encryption_algorithm": "algorithm",
            "encryption_metadata": {},
        }
        with (
            patch.object(_Cipher, "decrypt", return_value=b""),
            self.assertRaises(PgsqlInteractionStoreError),
        ):
            store._decrypt(
                encrypted_row,
                kind="record",
                identifier="request",
            )

    async def test_pgsql_admission_delivery_contains_failures_and_cancellation(
        self,
    ) -> None:
        outcome = TerminateInputContinuation(
            request_id=InputRequestId("request"),
            status=ResolutionStatus.CANCELLED,
        )
        notification = InputResumptionNotification(
            continuation_id=ContinuationId("continuation"),
            state_revision=StateRevision(1),
            outcome=outcome,
        )

        async def cancelled_resumer(
            _notification: InputResumptionNotification,
        ) -> None:
            raise CancelledError()

        async def failed_resumer(
            _notification: InputResumptionNotification,
        ) -> None:
            raise RuntimeError("delivery failed")

        for resumer in (cancelled_resumer, failed_resumer):
            handoff = get_running_loop().create_future()
            binding = interaction_pgsql._MemoryAdmissionBinding(
                request_id=InputRequestId("request"),
                continuation_id=ContinuationId("continuation"),
                resumer=cast(InputResumer, resumer),
                handoff=handoff,
            )
            with (
                self.subTest(resumer=resumer.__name__),
                patch.object(
                    interaction_pgsql,
                    "_report_resumption_delivery_failure",
                ) as report,
            ):
                await interaction_pgsql._publish_admission_resumption(
                    binding,
                    notification,
                )
            self.assertTrue(handoff.done())
            report.assert_called_once_with()

        started = Event()
        release = Event()

        async def blocked_resumer(
            _notification: InputResumptionNotification,
        ) -> None:
            started.set()
            await release.wait()

        handoff = get_running_loop().create_future()
        binding = interaction_pgsql._MemoryAdmissionBinding(
            request_id=InputRequestId("request"),
            continuation_id=ContinuationId("continuation"),
            resumer=cast(InputResumer, blocked_resumer),
            handoff=handoff,
        )
        delivery = create_task(
            interaction_pgsql._publish_admission_resumption(
                binding,
                notification,
            )
        )
        await started.wait()
        delivery.cancel()
        release.set()
        with self.assertRaises(CancelledError):
            await delivery
        self.assertTrue(handoff.done())

        repeated_release = Event()

        async def repeatedly_blocked_resumer(
            _notification: InputResumptionNotification,
        ) -> None:
            await repeated_release.wait()

        repeated_handoff = get_running_loop().create_future()
        repeated_binding = interaction_pgsql._MemoryAdmissionBinding(
            request_id=InputRequestId("request"),
            continuation_id=ContinuationId("continuation"),
            resumer=cast(InputResumer, repeatedly_blocked_resumer),
            handoff=repeated_handoff,
        )
        original_shield = interaction_pgsql.shield
        shield_calls = 0

        async def repeatedly_cancelled_shield(awaitable: object) -> object:
            nonlocal shield_calls
            shield_calls += 1
            if shield_calls <= 2:
                raise CancelledError()
            repeated_release.set()
            return await original_shield(cast(Awaitable[object], awaitable))

        with (
            patch.object(
                interaction_pgsql,
                "shield",
                new=repeatedly_cancelled_shield,
            ),
            self.assertRaises(CancelledError),
        ):
            await interaction_pgsql._publish_admission_resumption(
                repeated_binding,
                notification,
            )
        self.assertEqual(shield_calls, 3)
        self.assertTrue(repeated_handoff.done())

    async def test_pgsql_reconciler_releases_cancelled_delivery(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        record = ResumptionOutboxRecord(
            outbox_id="outbox",
            continuation_id=ContinuationId("continuation"),
            request_id=InputRequestId("request"),
            task_run_id=None,
            resolution_revision=StateRevision(1),
            status=interaction_pgsql.ResumptionOutboxStatus.CLAIMED,
            fencing_token=ContinuationFencingToken(1),
            attempts=1,
        )

        async def cancel_dispatch(_record: ResumptionOutboxRecord) -> None:
            raise CancelledError()

        reconciler = PgsqlResumptionReconciler(
            store,
            owner_id=ContinuationClaimOwnerId("owner"),
            dispatcher=cancel_dispatch,
            clock=lambda: _NOW,
        )
        with (
            patch.object(
                PgsqlInteractionStore,
                "claim_outbox",
                new=AsyncMock(return_value=(record,)),
            ),
            patch.object(
                PgsqlInteractionStore,
                "release_outbox",
                new=AsyncMock(),
            ) as release_outbox,
            self.assertRaises(CancelledError),
        ):
            await reconciler.run_once()
        self.assertEqual(
            release_outbox.await_args.kwargs["error_code"],
            "cancelled",
        )

    async def test_pgsql_schema_and_lock_helpers_fail_closed(self) -> None:
        database = FakePgsqlDatabase()
        with (
            patch.object(
                interaction_pgsql,
                "_row_str",
                return_value="wrong-revision",
            ),
            self.assertRaises(PgsqlInteractionFeatureUnavailableError),
        ):
            await interaction_pgsql._check_schema(database)

        database.fail_query = "avalan_task_alembic_version"
        with self.assertRaises(PgsqlInteractionFeatureUnavailableError):
            await interaction_pgsql._check_schema(database)

        class _CancelledSchemaDatabase:
            def connection(self) -> object:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            await interaction_pgsql._check_schema(
                cast(PgsqlDatabase, _CancelledSchemaDatabase())
            )

        missing_unit = _unit(row=None)
        for callback, arguments in (
            (interaction_pgsql._lock_store_metadata, (missing_unit,)),
            (
                interaction_pgsql._lock_continuation,
                (missing_unit, ContinuationId("continuation")),
            ),
        ):
            with (
                self.subTest(callback=callback.__name__),
                self.assertRaises(InputContractError),
            ):
                await callback(*cast(Any, arguments))
        with self.assertRaises(InteractionNotFoundError):
            await interaction_pgsql._lock_task_run_continuation(
                missing_unit,
                continuation_id=ContinuationId("continuation"),
                task_run_id="run",
            )

    async def test_pgsql_rejected_task_creates_fail_before_persistence(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        request = _request()
        command = _create_command(request)
        continuation = _portable(request)
        rejection = interaction_pgsql._PgsqlCreateCapacityError(
            command
        ).rejection

        async def reject_persistence(
            _self: PgsqlInteractionStore,
            _operation: str,
            _callback: object,
            *,
            after_persist: Callable[
                [PgsqlUnitOfWork, object],
                Awaitable[None],
            ],
            **_kwargs: object,
        ) -> object:
            await after_persist(_unit(), rejection)
            return rejection

        with patch.object(
            PgsqlInteractionStore,
            "_run_memory_with_unit",
            new=reject_persistence,
        ):
            for operation in ("suspend", "pending"):
                with (
                    self.subTest(operation=operation),
                    self.assertRaises(PgsqlInteractionStoreError),
                ):
                    if operation == "suspend":
                        await coordinator.create_and_suspend(
                            command,
                            continuation,
                            queue_item_id="queue",
                            claim_token="claim",
                            segment_id="segment",
                            task_run_id="run",
                            checkpoint_id="checkpoint",
                        )
                    else:
                        await coordinator.create_pending_interaction(
                            command,
                            continuation,
                            task_run_id="run",
                            checkpoint_id="checkpoint",
                        )

            previous = ContinuationCompletionCommand(
                continuation_id=ContinuationId("previous-continuation"),
                expected_store_revision=ContinuationStoreRevision(0),
                owner_id=ContinuationClaimOwnerId("claim"),
                fencing_token=ContinuationFencingToken(0),
                result_digest="d" * 64,
            )
            with self.assertRaises(PgsqlInteractionStoreError):
                await coordinator.complete_and_resuspend(
                    previous,
                    command,
                    continuation,
                    queue_item_id="queue",
                    claim_token="claim",
                    segment_id="segment",
                    task_run_id="run",
                    checkpoint_id="checkpoint",
                )

    async def test_pgsql_sweep_skips_each_nonactionable_row(self) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        now = _NOW
        completed_cases: list[str] = []

        async def run_row(
            case: str,
            row: Mapping[str, object],
            *,
            continuation: PortableContinuation | BaseException | None = None,
        ) -> None:
            unit = _unit(rows=(row,))

            async def transaction(
                _self: PgsqlInteractionStore,
                _operation: str,
                callback: Callable[[PgsqlUnitOfWork], Awaitable[object]],
                **_kwargs: object,
            ) -> object:
                return await callback(unit)

            patches = [
                patch.object(
                    PgsqlInteractionStore,
                    "_transaction",
                    new=transaction,
                )
            ]
            if isinstance(continuation, BaseException):
                patches.append(
                    patch.object(
                        PgsqlInteractionStore,
                        "_continuation_from_row",
                        side_effect=continuation,
                    )
                )
            elif continuation is not None:
                patches.append(
                    patch.object(
                        PgsqlInteractionStore,
                        "_continuation_from_row",
                        return_value=continuation,
                    )
                )
            with patches[0]:
                if len(patches) == 1:
                    result = await store.sweep(now=now)
                else:
                    with patches[1]:
                        result = await store.sweep(now=now)
            self.assertEqual(result.invalidated, ())
            self.assertEqual(result.deleted, ())
            completed_cases.append(case)

        base = {
            "task_run_id": None,
            "interaction_retention_deadline_at": now + timedelta(days=1),
        }
        await run_row(
            "missing_continuation",
            {**base, "continuation_id": None},
        )
        await run_row(
            "dispatching",
            {
                **base,
                "continuation_id": "dispatching",
                "lifecycle_state": (
                    DurableContinuationLifecycle.DISPATCHING.value
                ),
            },
        )
        await run_row(
            "corrupt",
            {
                **base,
                "continuation_id": "corrupt",
                "lifecycle_state": DurableContinuationLifecycle.READY.value,
            },
            continuation=PgsqlInteractionStoreError(
                InputErrorCode.SNAPSHOT_INVALID,
                "continuation",
                "corrupt",
            ),
        )
        await run_row(
            "not_expired",
            {
                **base,
                "continuation_id": "not-expired",
                "lifecycle_state": DurableContinuationLifecycle.READY.value,
            },
            continuation=_portable(_request("not-expired")),
        )
        self.assertEqual(
            completed_cases,
            [
                "missing_continuation",
                "dispatching",
                "corrupt",
                "not_expired",
            ],
        )

    async def test_pgsql_remaining_public_wrappers_and_poll_guards(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("remaining-wrappers")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)

        capacity = interaction_pgsql._PgsqlCreateCapacityError(created.command)
        with patch.object(
            PgsqlInteractionStore,
            "_run_memory_with_unit",
            new=AsyncMock(side_effect=capacity),
        ):
            self.assertIs(
                await store.create(created.command),
                capacity.rejection,
            )

        scope_command = SimpleNamespace(
            actor=created.command.actor,
            scope=InteractionExecutionScope(
                run_id=RunId("remaining-wrappers")
            ),
        )
        with patch.object(
            PgsqlInteractionStore,
            "_run_memory",
            new=AsyncMock(return_value=()),
        ):
            self.assertEqual(
                await store.list_scoped(cast(Any, scope_command)),
                (),
            )

        terminal_command = SimpleNamespace(
            actor=created.command.actor,
            correlation=created.record.correlation,
        )
        with patch.object(
            PgsqlInteractionStore,
            "_terminal_memory_operation",
            new=AsyncMock(return_value="terminal"),
        ):
            self.assertEqual(
                await store.terminalize(cast(Any, terminal_command)),
                "terminal",
            )

        wait_command = WaitForInteractionChangeCommand(
            actor=created.command.actor,
            correlation=created.record.correlation,
            after_store_revision=created.record.store_revision,
        )
        with (
            patch.object(
                PgsqlInteractionStore,
                "_run_memory_read_snapshot",
                new=AsyncMock(return_value=(None, None)),
            ),
            self.assertRaises(InteractionNotFoundError),
        ):
            await store.wait_for_change(wait_command)

        deadline = SimpleNamespace(schedule_revision=1)
        with patch.object(
            PgsqlInteractionStore,
            "next_deadline",
            new=AsyncMock(return_value=deadline),
        ):
            self.assertIs(
                await store.wait_for_deadline_change(
                    WaitForDeadlineChangeCommand(after_schedule_revision=0)
                ),
                deadline,
            )

        unchanged_deadline = SimpleNamespace(schedule_revision=0)
        next_deadline = AsyncMock(side_effect=(unchanged_deadline, deadline))
        with (
            patch.object(
                PgsqlInteractionStore,
                "next_deadline",
                new=next_deadline,
            ),
            patch.object(
                interaction_pgsql,
                "sleep",
                new=AsyncMock(),
            ) as sleep,
        ):
            self.assertIs(
                await store.wait_for_deadline_change(
                    WaitForDeadlineChangeCommand(after_schedule_revision=0)
                ),
                deadline,
            )
        sleep.assert_awaited_once_with(
            store._store_policy.poll_interval_seconds
        )

        async def invalid_callback(
            _memory: object,
            _unit: object,
        ) -> object:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "callback",
                "expected contract error",
            )

        with self.assertRaises(InputValidationError):
            await store._run_memory_with_unit(
                "contract_callback",
                cast(Any, invalid_callback),
                mutate=False,
            )

    async def test_pgsql_remaining_capacity_and_terminal_persistence_paths(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        request = _request("remaining-persistence")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        rejection = interaction_pgsql._PgsqlCreateCapacityError(
            created.command
        ).rejection
        await store._enforce_create_process_capacity(
            _unit(row=None),
            created.command,
            rejection,
        )
        with self.assertRaises(interaction_pgsql._PgsqlCreateCapacityError):
            await store._enforce_create_process_capacity(
                _unit(
                    row={
                        "pending_count": (
                            store._policy.maximum_pending_interactions_per_process
                        )
                    }
                ),
                created.command,
                created,
            )

        resolved = await store.resolve(_answer(created))
        self.assertIsInstance(resolved, ResolveInteractionApplied)
        answered = cast(ResolveInteractionApplied, resolved).record

        terminal_request = _request("remaining-terminal")
        terminal_created = await store.create(
            _create_command(terminal_request)
        )
        self.assertIsInstance(terminal_created, CreateInteractionApplied)
        assert isinstance(terminal_created, CreateInteractionApplied)
        terminalized = await store.terminalize_scope(
            TerminalizeInteractionScopeCommand(
                actor=terminal_created.command.actor,
                scope=InteractionExecutionScope(
                    run_id=terminal_request.origin.run_id
                ),
                provenance=AnswerProvenance.HUMAN,
            )
        )
        terminal_records = cast(Any, terminalized).records
        self.assertEqual(len(terminal_records), 1)
        non_answered = terminal_records[0]

        result_holder: list[object] = [SimpleNamespace(record=non_answered)]

        async def run_after_persist(
            _self: PgsqlInteractionStore,
            _operation: str,
            _callback: object,
            *,
            after_persist: Callable[
                [PgsqlUnitOfWork, object],
                Awaitable[None],
            ],
            **_kwargs: object,
        ) -> object:
            result = result_holder[0]
            await after_persist(_unit(), result)
            return result

        with (
            patch.object(
                PgsqlInteractionStore,
                "_run_memory_with_unit",
                new=run_after_persist,
            ),
            patch.object(
                PgsqlInteractionStore,
                "_reject_standalone_task_lifecycle",
                new=AsyncMock(),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_invalidate_for_record",
                new=AsyncMock(),
            ) as invalidate,
            patch.object(
                PgsqlInteractionStore,
                "_ready_continuation",
                new=AsyncMock(),
            ) as ready,
            patch.object(
                PgsqlInteractionStore,
                "_sync_resolution_keys",
                new=AsyncMock(),
            ),
        ):
            await store.resolve(_answer(created))
            invalidate.assert_awaited_once()

            result_holder[0] = SimpleNamespace(record=answered)

            async def unused_terminal(_memory: object) -> object:
                return result_holder[0]

            await store._terminal_memory_operation(
                "terminal_answered",
                cast(Any, unused_terminal),
            )
            ready.assert_awaited_once()

    def test_pgsql_remaining_small_helper_positive_paths(self) -> None:
        self.assertEqual(
            interaction_pgsql._mapping_optional_int(
                {"value": 1},
                "value",
            ),
            1,
        )
        origin = replace(
            _origin("branch-run"),
            branch_id=BranchId("child"),
            parent_branch_id=BranchId("root"),
        )
        record = cast(
            InteractionRecord,
            SimpleNamespace(request=SimpleNamespace(origin=origin)),
        )
        branch = InteractionBranchRecord(
            registration=InteractionBranchRegistration(
                run_id=origin.run_id,
                branch_id=origin.branch_id,
                parent_branch_id=cast(BranchId, origin.parent_branch_id),
                principal=origin.principal,
            ),
            store_revision=InteractionStoreRevision(0),
        )
        self.assertEqual(
            interaction_pgsql._records_with_valid_branch_edges(
                (record,),
                (branch,),
            ),
            (record,),
        )

    async def test_pgsql_task_lifecycle_rejects_malformed_results(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        request = _request("lifecycle-result")
        created = await store.create(_create_command(request))
        self.assertIsInstance(created, CreateInteractionApplied)
        assert isinstance(created, CreateInteractionApplied)
        answered_result = await store.resolve(_answer(created))
        self.assertIsInstance(answered_result, ResolveInteractionApplied)
        answered = cast(ResolveInteractionApplied, answered_result).record

        pending_request = _request("lifecycle-pending")
        pending_created = await store.create(_create_command(pending_request))
        self.assertIsInstance(pending_created, CreateInteractionApplied)
        assert isinstance(pending_created, CreateInteractionApplied)
        pending = pending_created.record

        holder: dict[str, object] = {
            "result": SimpleNamespace(records=None),
            "row": None,
        }

        async def run_after_persist(
            _self: PgsqlInteractionStore,
            _operation: str,
            _callback: object,
            *,
            after_persist: Callable[
                [PgsqlUnitOfWork, object],
                Awaitable[None],
            ],
            **_kwargs: object,
        ) -> object:
            result = holder["result"]
            await after_persist(_unit(row=holder["row"]), result)
            return result

        async def unused_lifecycle(
            _memory: object,
            _unit: object,
        ) -> object:
            return holder["result"]

        async def invoke() -> object:
            return await coordinator._terminalize_task_lifecycle(
                "lifecycle_result_guard",
                cast(Any, unused_lifecycle),
                task_run_id="lifecycle-result",
                request_state=RequestState.CANCELLED,
                run_state=TaskRunState.CANCELLED,
                attempt_state=TaskAttemptState.ABANDONED,
                event_type=TaskInteractionEventType.INPUT_CANCELLED,
                reason="cancelled",
                now=_NOW,
                metadata={},
                selection=interaction_pgsql._trusted_task_snapshot(
                    "lifecycle-result"
                ),
            )

        with patch.object(
            PgsqlInteractionStore,
            "_run_memory_with_unit",
            new=run_after_persist,
        ):
            await invoke()
            for result, row in (
                (SimpleNamespace(records=(object(),)), None),
                (SimpleNamespace(records=(pending,)), None),
                (
                    SimpleNamespace(records=(answered,)),
                    {"task_run_id": "other-run"},
                ),
                (
                    SimpleNamespace(records=(answered,)),
                    {"task_run_id": "lifecycle-result"},
                ),
            ):
                holder.update(result=result, row=row)
                with (
                    self.subTest(result=result, row=row),
                    self.assertRaises(PgsqlInteractionStoreError),
                ):
                    await invoke()

    async def test_pgsql_trusted_task_cancellation_rejects_bad_ownership(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        holder: dict[str, object] = {
            "rows": (),
            "records": (),
        }

        async def run_callback(
            _self: PgsqlInteractionStore,
            _operation: str,
            callback: Callable[[object, PgsqlUnitOfWork], Awaitable[object]],
            **_kwargs: object,
        ) -> object:
            memory = SimpleNamespace(terminalize_scope=AsyncMock())
            return await callback(
                cast(Any, memory),
                _unit(rows=cast(tuple[object, ...], holder["rows"])),
            )

        def record_from_row(_self: object, _row: object) -> object:
            records = cast(list[object], holder["records"])
            return records.pop(0)

        async def invoke() -> object:
            return await coordinator.cancel_input_required_task(
                task_run_id="run",
                now=_NOW,
                metadata={},
            )

        with (
            patch.object(
                PgsqlInteractionStore,
                "_run_memory_with_unit",
                new=run_callback,
            ),
            patch.object(
                PgsqlInteractionStore,
                "_record_from_row",
                new=record_from_row,
            ),
        ):
            with self.assertRaises(PgsqlInteractionStoreError):
                await invoke()

            holder["rows"] = ({},)
            holder["records"] = [
                SimpleNamespace(
                    request=SimpleNamespace(
                        origin=SimpleNamespace(
                            run_id=RunId("other-run"),
                            principal=PrincipalScope(user_id=UserId("owner")),
                        )
                    )
                )
            ]
            with self.assertRaises(PgsqlInteractionStoreError):
                await invoke()

            holder["rows"] = ({}, {})
            holder["records"] = [
                SimpleNamespace(
                    request=SimpleNamespace(
                        origin=SimpleNamespace(
                            run_id=RunId("run"),
                            principal=PrincipalScope(user_id=UserId("first")),
                        )
                    )
                ),
                SimpleNamespace(
                    request=SimpleNamespace(
                        origin=SimpleNamespace(
                            run_id=RunId("run"),
                            principal=PrincipalScope(user_id=UserId("second")),
                        )
                    )
                ),
            ]
            with self.assertRaises(PgsqlInteractionStoreError):
                await invoke()

    async def test_pgsql_rejection_helpers_require_exact_claim_state(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        continuation = _portable(_request())
        rejection = ContinuationRejectionCommand(
            continuation_id=continuation.continuation_id,
            expected_store_revision=ContinuationStoreRevision(0),
            owner_id=ContinuationClaimOwnerId("claim"),
            fencing_token=ContinuationFencingToken(0),
            result_digest="d" * 64,
        )
        unit = _unit()

        async def invoke() -> object:
            return await coordinator._reject_admitted_continuation_in_unit(
                unit,
                rejection=rejection,
                task_run_id="run",
                request_id=str(continuation.request_id),
                checkpoint_id="checkpoint",
                now=_NOW,
            )

        with patch.object(
            interaction_pgsql,
            "_lock_task_continuation",
            new=AsyncMock(return_value={"lifecycle_state": "ready"}),
        ):
            with (
                patch.object(
                    PgsqlInteractionStore,
                    "_continuation_from_row",
                    return_value=replace(
                        continuation,
                        origin=replace(
                            continuation.origin,
                            run_id=RunId("other-run"),
                        ),
                    ),
                ),
                self.assertRaises(PgsqlInteractionStoreError),
            ):
                await invoke()

        with (
            patch.object(
                interaction_pgsql,
                "_lock_task_continuation",
                new=AsyncMock(
                    return_value={
                        "lifecycle_state": "invalidated",
                    }
                ),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ),
            self.assertRaises(ContinuationStoreConflictError),
        ):
            await invoke()

        with (
            patch.object(
                interaction_pgsql,
                "_lock_task_continuation",
                new=AsyncMock(
                    return_value={
                        "lifecycle_state": "claimed",
                    }
                ),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ),
            patch.object(
                PgsqlInteractionStore,
                "_claimed_continuation",
                return_value=continuation,
            ),
            self.assertRaises(ContinuationStoreConflictError),
        ):
            await invoke()

    def test_pgsql_cancel_requested_claim_requires_dispatch_evidence(
        self,
    ) -> None:
        continuation = _portable(_request())
        request_id = str(continuation.request_id)
        continuation_id = str(continuation.continuation_id)
        task_row = {
            "previous_segment_id": "segment",
            "active_segment_id": "segment",
            "durable_attempt_state": TaskAttemptState.SUSPENDED.value,
            "active_segment_state": TaskAttemptSegmentState.SUSPENDED.value,
            "state": TaskQueueItemState.CLAIMED.value,
            "run_id": "run",
            "claim_token": "claim",
            "request_id": request_id,
            "continuation_id": continuation_id,
            "durable_run_state": TaskRunState.CANCEL_REQUESTED.value,
            "durable_run_claim_token": "claim",
            "durable_attempt_id": "attempt",
            "attempt_id": "attempt",
            "durable_attempt_claim_token": "claim",
            "segment_id": "segment",
            "previous_attempt_id": "attempt",
            "previous_run_id": "run",
            "previous_segment_state": TaskAttemptSegmentState.SUSPENDED.value,
            "previous_segment_claim_token": None,
            "previous_request_id": request_id,
            "previous_continuation_id": continuation_id,
            "previous_checkpoint_id": "checkpoint",
            "active_attempt_id": "attempt",
            "active_run_id": "run",
            "active_segment_claim_token": None,
            "active_resumed_from_segment_id": None,
            "active_request_id": request_id,
            "active_continuation_id": continuation_id,
            "active_checkpoint_id": "checkpoint",
        }
        continuation_row = {
            "task_run_id": "run",
            "request_id": request_id,
            "continuation_id": continuation_id,
            "checkpoint_id": "checkpoint",
        }
        claimed_without_dispatch = SimpleNamespace(
            claim=SimpleNamespace(
                state=ContinuationClaimState.CLAIMED_PRE_DISPATCH
            ),
            dispatch=None,
        )
        store = cast(
            PgsqlInteractionStore,
            SimpleNamespace(
                _claimed_continuation=lambda *_args, **_kwargs: (
                    claimed_without_dispatch
                )
            ),
        )
        with self.assertRaises(ContinuationStoreConflictError):
            interaction_pgsql._validate_cancel_requested_pre_dispatch_claim(
                store,
                task_row=task_row,
                continuation_row=continuation_row,
                continuation=continuation,
                task_run_id="run",
                request_id=request_id,
                continuation_id=continuation_id,
                checkpoint_id="checkpoint",
                expected_claim_token="claim",
            )

    async def test_pgsql_expired_reentry_rejects_malformed_provenance(
        self,
    ) -> None:
        database = FakePgsqlDatabase()
        store = await _store(database)
        task_ids = _Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
            id_factory=lambda: task_ids.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        continuation = _portable(_request())
        request_id = str(continuation.request_id)
        continuation_id = str(continuation.continuation_id)
        checkpoint_id = "checkpoint"
        provenance: dict[str, object] = {
            "previous_request_id": request_id,
            "previous_continuation_id": continuation_id,
            "previous_checkpoint_id": checkpoint_id,
            "active_segment_id": "segment",
            "durable_run_state": TaskRunState.CANCEL_REQUESTED.value,
            "durable_attempt_state": TaskAttemptState.FAILED.value,
            "active_segment_state": TaskAttemptSegmentState.FAILED.value,
            "lease_expires_at": _NOW - timedelta(seconds=1),
        }
        continuation_row: dict[str, object] = {
            "lifecycle_state": DurableContinuationLifecycle.COMPLETED.value,
            "request_absolute_expires_at": continuation.expires_at,
        }
        unit = _unit(row=provenance)

        async def direct_transaction(
            _self: PgsqlInteractionStore,
            _operation: str,
            callback: Callable[[PgsqlUnitOfWork], Awaitable[object]],
            **_kwargs: object,
        ) -> object:
            return await callback(unit)

        async def invoke(*, now: datetime) -> object:
            return await coordinator.reconcile_expired_reentry(
                queue_item_id="queue",
                expected_claim_token="claim",
                task_run_id="run",
                result=TaskExecutionResult(error={"code": "failure"}),
                now=now,
            )

        with (
            patch.object(
                PgsqlInteractionStore,
                "_transaction",
                new=direct_transaction,
            ),
            patch.object(
                interaction_pgsql,
                "_lock_task_continuation",
                new=AsyncMock(return_value=continuation_row),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ),
            self.assertRaises(TaskStoreConflictError),
        ):
            await invoke(now=_NOW)

        provenance.update(
            durable_run_state=TaskRunState.RUNNING.value,
            durable_attempt_state=TaskAttemptState.RUNNING.value,
            active_segment_state=TaskAttemptSegmentState.RUNNING.value,
        )
        continuation_row["lifecycle_state"] = (
            DurableContinuationLifecycle.READY.value
        )
        with (
            patch.object(
                PgsqlInteractionStore,
                "_transaction",
                new=direct_transaction,
            ),
            patch.object(
                interaction_pgsql,
                "_lock_task_continuation",
                new=AsyncMock(return_value=continuation_row),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ),
            patch.object(
                PgsqlInteractionStore,
                "_update_continuation",
                new=AsyncMock(),
            ),
            patch.object(
                PgsqlTaskStore,
                "_settle_claim_in_unit",
                new=AsyncMock(return_value=object()),
            ),
            self.assertRaises(AssertionError),
        ):
            await invoke(now=continuation.expires_at)

        provenance.update(
            durable_run_state=TaskRunState.FAILED.value,
            durable_attempt_state=TaskAttemptState.FAILED.value,
            active_segment_state=TaskAttemptSegmentState.FAILED.value,
        )
        continuation_row["lifecycle_state"] = (
            DurableContinuationLifecycle.COMPLETED.value
        )
        with (
            patch.object(
                PgsqlInteractionStore,
                "_transaction",
                new=direct_transaction,
            ),
            patch.object(
                interaction_pgsql,
                "_lock_task_continuation",
                new=AsyncMock(return_value=continuation_row),
            ),
            patch.object(
                PgsqlInteractionStore,
                "_continuation_from_row",
                return_value=continuation,
            ),
            patch.object(
                PgsqlTaskStore,
                "_fail_claimed_reentry_in_unit",
                new=AsyncMock(return_value=object()),
            ),
            self.assertRaises(AssertionError),
        ):
            await invoke(now=_NOW)

    def test_pgsql_coordinator_successor_validators_reject_drift(self) -> None:
        request = _request()
        command = _create_command(request)
        continuation = _portable(request)
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(error={"code": "failure"})
        )
        completion = ContinuationCompletionCommand(
            continuation_id=continuation.continuation_id,
            expected_store_revision=ContinuationStoreRevision(0),
            owner_id=ContinuationClaimOwnerId("claim"),
            fencing_token=ContinuationFencingToken(0),
            result_digest=task_durable_resume_settlement_digest(failure),
        )

        with self.assertRaises(InputValidationError):
            PgsqlDurableTaskCoordinator._validate_continuation(
                command,
                replace(
                    continuation,
                    request_id=InputRequestId("other-request"),
                ),
            )
        with self.assertRaises(InputValidationError):
            PgsqlDurableTaskCoordinator._validate_successor(
                completion,
                command=command,
                continuation=continuation,
                task_run_id="other-run",
            )
        with self.assertRaises(InputValidationError):
            PgsqlDurableTaskCoordinator._validate_successor(
                completion,
                command=command,
                continuation=continuation,
                task_run_id="run",
            )
        with self.assertRaises(InputValidationError):
            PgsqlDurableTaskCoordinator._validate_completed_successor(
                continuation,
                continuation,
            )


if __name__ == "__main__":
    from unittest import main

    main()
