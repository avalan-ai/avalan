"""Test claimed and fenced fresh-process agent continuation."""

from asyncio import (
    CancelledError,
    Event,
    all_tasks,
    create_task,
    current_task,
    gather,
    sleep,
)
from asyncio import run as asyncio_run
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ParamSpec, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from avalan.agent import continuation as continuation_module
from avalan.agent import durable_runtime as durable_runtime_module
from avalan.agent.continuation import (
    AgentContinuationEventListener,
    AgentContinuationEventListenerRegistration,
    AgentContinuationResumeCommand,
    DurableAgentContinuationAdmission,
    DurableAgentContinuationClaimLease,
    DurableAgentContinuationResumer,
)
from avalan.agent.execution import AgentExecution, ExecutionCorrelationError
from avalan.agent.loader import OrchestratorLoader
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import Message, MessageRole
from avalan.event.manager import EventManager
from avalan.interaction.broker import InteractionBrokerRequest
from avalan.interaction.codec import (
    canonical_resolution_digest,
    decode_continuation_snapshot,
    encode_continuation_snapshot,
    encode_input_question,
    semantic_request_fingerprint,
)
from avalan.interaction.continuation import (
    ContinuationClaim,
    ContinuationClaimOwnerId,
    ContinuationClaimReceipt,
    ContinuationClaimState,
    ContinuationCompletion,
    ContinuationDispatch,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationRuntimeResolver,
    ContinuationStoreRevision,
    DurableContinuationRecord,
    DurableContinuationResumeState,
    PortableContinuation,
    ResolvedContinuationRuntime,
    derive_provider_idempotency_key,
)
from avalan.interaction.entities import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CapabilityRevision,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    InteractionStoreRevision,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    RunId,
    StateRevision,
    StreamSessionId,
    TaskId,
    TurnId,
    UserId,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.policy import InteractionActor
from avalan.interaction.store import (
    InteractionPresentationState,
    InteractionRecord,
    ResolutionIdempotencyEntry,
)
from avalan.model.capability import (
    ContinuationSnapshotCodecRegistry,
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
)
from avalan.model.nlp.text.vendor import openai as openai_module
from avalan.model.nlp.text.vendor.openai import OpenAIClient
from avalan.model.stream import StreamRetentionPolicy
from avalan.task import (
    TaskAttempt,
    TaskAttemptSegment,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskClaim,
    TaskExecutionContext,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskQueueClaim,
    TaskQueueItem,
    TaskQueueItemState,
    TaskRun,
    TaskRunState,
)
from avalan.task.resume import (
    StoredTaskResumeActorResolver,
    TaskDurableResumeAdmission,
    TaskDurableResumeCoordinator,
    _validate_previous_segment,
    _validate_record,
    task_resume_result_digest,
)
from avalan.task.settlement import TaskDurableResumeFailure
from avalan.types import JsonValue

_NOW = datetime(2026, 7, 23, 12, tzinfo=UTC)
_CLAIMED_AT = _NOW + timedelta(seconds=10)
_LEASE_EXPIRES_AT = _NOW + timedelta(minutes=10)
_RESULT_DIGEST = "a" * 64
_DISPATCH_ID = ContinuationDispatchId("dispatch")
_P = ParamSpec("_P")


def _async_test(
    function: Callable[_P, Coroutine[Any, Any, None]],
) -> Callable[_P, None]:
    @wraps(function)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> None:
        asyncio_run(function(*args, **kwargs))

    return run


class _Adapter:
    def __init__(self) -> None:
        self.imported: list[ContinuationSnapshot] = []
        self.validated: list[ContinuationSnapshot] = []
        self.validation_failure: BaseException | None = None

    def validate_continuation_snapshot_call(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
        expected_provider_name: str,
        expected_arguments: Mapping[str, object],
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )
        assert expected_provider_name == "request_user_input"
        assert expected_arguments["mode"] == RequirementMode.REQUIRED.value
        self.validated.append(snapshot)
        if self.validation_failure is not None:
            raise self.validation_failure

    def import_continuation_snapshot(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )
        self.imported.append(snapshot)


class _EventListenerRegistration:
    def __init__(self) -> None:
        self.close_calls = 0
        self.remove_calls = 0
        self.closed = False
        self.closed_signal = Event()

    def close(self) -> None:
        """Close the no-op test registration."""
        self.close_calls += 1
        if self.closed:
            return
        self.closed = True
        self.remove_calls += 1
        self.closed_signal.set()


class _Executor:
    trusted_agent_continuation_executor = True

    def __init__(self) -> None:
        self.commands: list[AgentContinuationResumeCommand] = []
        self.failure: BaseException | None = None
        self.started: Event | None = None
        self.proceed: Event | None = None
        self.close_calls = 0
        self.close_failure: BaseException | None = None
        self.close_started: Event | None = None
        self.close_proceed: Event | None = None
        self.close_completed: Event | None = None
        self.self_cancel_on_close_completion = False
        self.event_listener_register_calls = 0
        self.event_listener_registration_error: BaseException | None = None
        self.event_listener_registrations: list[_EventListenerRegistration] = (
            []
        )

    def register_event_listener(
        self,
        listener: AgentContinuationEventListener,
    ) -> AgentContinuationEventListenerRegistration:
        assert callable(listener)
        self.event_listener_register_calls += 1
        if self.event_listener_registration_error is not None:
            raise self.event_listener_registration_error
        registration = _EventListenerRegistration()
        self.event_listener_registrations.append(registration)
        return registration

    async def resume_agent_continuation(
        self,
        command: AgentContinuationResumeCommand,
    ) -> object:
        self.commands.append(command)
        if self.started is not None:
            self.started.set()
        if self.proceed is not None:
            await self.proceed.wait()
        if self.failure is not None:
            raise self.failure
        return {"answer": "resumed"}

    async def close_continuation_runtime(self) -> None:
        self.close_calls += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.close_proceed is not None:
            await self.close_proceed.wait()
        if self.close_failure is not None:
            raise self.close_failure
        if self.close_completed is not None:
            self.close_completed.set()
        if self.self_cancel_on_close_completion:
            task = current_task()
            assert task is not None
            task.cancel()


class _Loader:
    trusted_continuation_runtime_loader = True

    def __init__(
        self,
        runtime: ResolvedContinuationRuntime,
    ) -> None:
        self.runtime = runtime
        self.failure: BaseException | None = None
        self.started: Event | None = None
        self.proceed: Event | None = None
        self.calls: list[
            tuple[ExecutionDefinitionRef, ContinuationRevisionBinding]
        ] = []

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        self.calls.append((definition, revision_binding))
        if self.started is not None:
            self.started.set()
        if self.proceed is not None:
            await self.proceed.wait()
        if self.failure is not None:
            raise self.failure
        return self.runtime


ClaimMutator = Callable[[PortableContinuation], PortableContinuation]


class _Store:
    def __init__(
        self,
        record: InteractionRecord,
        continuation: PortableContinuation,
    ) -> None:
        self.interaction_record = record
        self.continuation = continuation
        self.claim_mutator: ClaimMutator | None = None
        self.failures: dict[str, BaseException] = {}
        self.complete_after_failure: BaseException | None = None
        self.calls: list[str] = []
        self.release_returns_no_op = False
        self.release_attempts = 0
        self.release_started: Event | None = None
        self.release_proceed: Event | None = None
        self.release_completed: Event | None = None

    async def lookup_scoped(self, query: object) -> InteractionRecord:
        del query
        self.calls.append("lookup")
        return self.interaction_record

    async def claim(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        lease_expires_at: datetime,
        dispatch_id: ContinuationDispatchId,
        provider_idempotency_key: ProviderIdempotencyKey,
        now: datetime,
    ) -> ContinuationClaimReceipt:
        self._fail("claim")
        assert continuation_id == self.continuation.continuation_id
        assert expected_store_revision == self.continuation.store_revision
        self.calls.append("claim")
        claimed = replace(
            self.continuation,
            claim=ContinuationClaim(
                state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
                attempt=self.continuation.claim.attempt + 1,
            ),
            fencing_token=ContinuationFencingToken(
                int(self.continuation.fencing_token) + 1
            ),
            dispatch=ContinuationDispatch(
                dispatch_id=dispatch_id,
                provider_idempotency_key=provider_idempotency_key,
                marked_at=now,
            ),
            store_revision=ContinuationStoreRevision(
                int(self.continuation.store_revision) + 1
            ),
            updated_at=now,
        )
        if self.claim_mutator is not None:
            claimed = self.claim_mutator(claimed)
        self.continuation = claimed
        return ContinuationClaimReceipt(
            continuation=claimed,
            fencing_token=claimed.fencing_token,
        )

    async def mark_dispatching(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        del continuation_id, owner_id, fencing_token
        self._fail("mark_dispatching")
        assert expected_store_revision == self.continuation.store_revision
        self.calls.append("mark_dispatching")
        self.continuation = replace(
            self.continuation,
            claim=ContinuationClaim(
                state=ContinuationClaimState.DISPATCHED_AMBIGUOUS,
                owner_id=self.continuation.claim.owner_id,
                attempt=self.continuation.claim.attempt,
            ),
            store_revision=ContinuationStoreRevision(
                int(self.continuation.store_revision) + 1
            ),
            updated_at=now,
        )
        return self.continuation

    async def renew_claim(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        lease_expires_at: datetime,
        now: datetime,
    ) -> bool:
        self._fail("renew_claim")
        assert continuation_id == self.continuation.continuation_id
        assert self.continuation.fencing_token == fencing_token
        if (
            self.continuation.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        ):
            return False
        assert expected_store_revision == self.continuation.store_revision
        assert self.continuation.claim.owner_id == owner_id
        assert self.continuation.claim.lease_expires_at is not None
        assert self.continuation.claim.lease_expires_at > now
        assert lease_expires_at <= self.continuation.expires_at
        self.calls.append("renew_claim")
        if lease_expires_at > self.continuation.claim.lease_expires_at:
            self.continuation = replace(
                self.continuation,
                claim=replace(
                    self.continuation.claim,
                    lease_expires_at=lease_expires_at,
                ),
                updated_at=now,
            )
        return True

    async def mark_dispatched(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        del continuation_id, owner_id, fencing_token
        self._fail("mark_dispatched")
        assert expected_store_revision == self.continuation.store_revision
        self.calls.append("mark_dispatched")
        self.continuation = replace(
            self.continuation,
            store_revision=ContinuationStoreRevision(
                int(self.continuation.store_revision) + 1
            ),
            updated_at=now,
        )
        return self.continuation

    async def complete(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        result_digest: str,
        now: datetime,
    ) -> PortableContinuation:
        del continuation_id, owner_id, fencing_token
        self._fail("complete")
        assert expected_store_revision == self.continuation.store_revision
        self.calls.append("complete")
        self.continuation = replace(
            self.continuation,
            claim=ContinuationClaim(
                state=ContinuationClaimState.COMPLETED,
                attempt=self.continuation.claim.attempt,
            ),
            completion=ContinuationCompletion(
                completed_at=now,
                result_digest=result_digest,
            ),
            store_revision=ContinuationStoreRevision(
                int(self.continuation.store_revision) + 1
            ),
            updated_at=now,
        )
        if self.complete_after_failure is not None:
            error = self.complete_after_failure
            self.complete_after_failure = None
            raise error
        return self.continuation

    async def get_continuation(
        self,
        continuation_id: ContinuationId,
    ) -> PortableContinuation:
        self._fail("get_continuation")
        assert continuation_id == self.continuation.continuation_id
        self.calls.append("get_continuation")
        return self.continuation

    async def release(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        del continuation_id, owner_id, fencing_token
        self.release_attempts += 1
        if self.release_started is not None:
            self.release_started.set()
        if self.release_proceed is not None:
            await self.release_proceed.wait()
        self._fail("release")
        assert expected_store_revision == self.continuation.store_revision
        self.calls.append("release")
        if self.release_returns_no_op:
            return self.continuation
        self.continuation = replace(
            self.continuation,
            claim=ContinuationClaim(
                state=ContinuationClaimState.FAILED_SAFE_TO_RETRY,
                attempt=self.continuation.claim.attempt,
            ),
            store_revision=ContinuationStoreRevision(
                int(self.continuation.store_revision) + 1
            ),
            updated_at=now,
        )
        if self.release_completed is not None:
            self.release_completed.set()
        return self.continuation

    def _fail(self, operation: str) -> None:
        failure = self.failures.pop(operation, None)
        if failure is not None:
            raise failure


@dataclass(frozen=True, slots=True)
class _Harness:
    request: InputRequest
    record: DurableContinuationRecord
    store: _Store
    adapter: _Adapter
    executor: _Executor
    loader: _Loader
    resumer: DurableAgentContinuationResumer


def _definition(*, revision: str = "agent-r1") -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="file:///trusted/agent.toml",
        agent_definition_revision=revision,
        operation_id="operation",
        operation_index=0,
        model_config_reference="model-config-r1",
        tool_revision="tools-r1",
        capability_revision="capability-r1",
    )


def _binding() -> ContinuationRevisionBinding:
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("gpt-5"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )


def _openai_client() -> OpenAIClient:
    client = object.__new__(OpenAIClient)
    client._base_url = "https://api.openai.com/v1"  # noqa: SLF001
    client._is_azure = False  # noqa: SLF001
    client._stream_retention_policy = StreamRetentionPolicy()  # noqa: SLF001
    client._replay_owners_by_call_id = {}  # noqa: SLF001
    client._active_replay_owners = {}  # noqa: SLF001
    client._active_replay_streams = {}  # noqa: SLF001
    client._active_replay_call_ids = {}  # noqa: SLF001
    client._ambiguous_replay_call_ids = {}  # noqa: SLF001
    client._replay_association_poisoned = False  # noqa: SLF001
    client._closed = False  # noqa: SLF001
    return client


def _openai_snapshot(
    request: InputRequest,
    binding: ContinuationRevisionBinding,
) -> ContinuationSnapshot:
    client = _openai_client()
    owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        client._stream_retention_policy  # noqa: SLF001
    )
    arguments = {
        "mode": request.mode.value,
        "reason": request.reason,
        "questions": [
            encode_input_question(question) for question in request.questions
        ],
    }
    owner.begin_attempt()
    assert owner.admit(
        {
            "id": "reasoning-current",
            "type": "reasoning",
            "encrypted_content": "encrypted-current-reasoning",
        }
    )
    assert owner.admit(
        {
            "id": "function-current",
            "type": "function_call",
            "call_id": "call-input",
            "name": "request_user_input",
            "arguments": dumps(
                arguments,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    )
    client._activate_replay_owner(owner)  # noqa: SLF001
    return client.export_continuation_snapshot(
        revision_binding=binding,
        model_call_id=request.origin.model_call_id,
        provider_idempotency_key=derive_provider_idempotency_key(
            request.continuation_id,
            _DISPATCH_ID,
        ),
        provider_call_correlation_id="call-input",
    )


def _origin(
    definition: ExecutionDefinitionRef,
    *,
    run_id: str = "task-run",
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("turn"),
        task_id=TaskId(run_id),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=definition,
        principal=PrincipalScope(user_id=UserId("owner")),
    )


def _terminal_request(origin: ExecutionOrigin) -> InputRequest:
    answer = ConfirmationAnswer(
        question_id=QuestionId("confirm"),
        provenance=AnswerProvenance.HUMAN,
        value=True,
    )
    resolution = AnsweredResolution(
        request_id=InputRequestId("request"),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=5),
        answers=(answer,),
    )
    return InputRequest(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Confirm continued execution.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=_NOW,
        continuation_ttl_seconds=3_600,
        state=RequestState.ANSWERED,
        state_revision=StateRevision(2),
        resolution=resolution,
    )


def _portable(
    request: InputRequest,
    binding: ContinuationRevisionBinding,
) -> PortableContinuation:
    snapshot = ContinuationSnapshot(
        snapshot_kind="openai.responses.reasoning",
        revision_binding=binding,
        model_call_id=request.origin.model_call_id,
        provider_idempotency_key=derive_provider_idempotency_key(
            request.continuation_id,
            _DISPATCH_ID,
        ),
        payload={
            "reserved_capability_call_id": "call-input",
            "replay_items": (
                {
                    "id": "reasoning-item",
                    "type": "reasoning",
                    "encrypted_content": "ciphertext",
                },
                {
                    "type": "function_call",
                    "call_id": "call-input",
                    "name": "request_user_input",
                    "arguments": "{}",
                },
            ),
        },
    )
    return PortableContinuation(
        continuation_id=request.continuation_id,
        request_id=request.request_id,
        origin=request.origin,
        provider_call_id=request.origin.model_call_id,
        provider_call_correlation_id="call-input",
        definition=request.origin.definition,
        operation_cursor=0,
        generation_settings={"temperature": 0},
        transcript=({"role": "user", "content": "hello"},),
        observations=(),
        provider_snapshot=snapshot,
        revision_binding=binding,
        interaction_count=1,
        tool_loop_count=0,
        stream_sequence=3,
        state_revision=StateRevision(2),
        store_revision=ContinuationStoreRevision(0),
        created_at=_NOW,
        updated_at=_NOW,
        expires_at=_NOW + timedelta(hours=1),
    )


def _catalog(
    binding: ContinuationRevisionBinding,
) -> ModelCapabilityCatalog:
    registry = ContinuationSnapshotCodecRegistry("resume-test-registry")
    registry.register(
        codec_id="openai-resume-v1",
        revision_binding=binding,
        snapshot_kind="openai.responses.reasoning",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    return ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec_registry=registry,
            continuation_snapshot_codec=registry.reference("openai-resume-v1"),
        ),
        revision_binding=binding,
    )


def _harness() -> _Harness:
    definition = _definition()
    binding = _binding()
    request = _terminal_request(_origin(definition))
    continuation = _portable(request, binding)
    resolution_digest = canonical_resolution_digest(
        cast(AnsweredResolution, request.resolution)
    )
    interaction_record = InteractionRecord(
        request=request,
        semantic_fingerprint=semantic_request_fingerprint(request),
        absolute_expires_at=(
            request.created_at
            + timedelta(seconds=request.continuation_ttl_seconds)
        ),
        presentation=InteractionPresentationState.DETACHED,
        store_revision=InteractionStoreRevision(2),
        resolution_digest=resolution_digest,
        idempotency_ledger=(
            ResolutionIdempotencyEntry(
                key=ResolutionIdempotencyKey("resolution-key"),
                resolution_digest=resolution_digest,
            ),
        ),
        resolved_by=request.origin.principal,
    )
    store = _Store(interaction_record, continuation)
    adapter = _Adapter()
    executor = _Executor()
    runtime = ResolvedContinuationRuntime(
        definition=definition,
        revision_binding=binding,
        runtime=executor,
        operation=object(),
        model=adapter,
        tools=object(),
        capabilities=_catalog(binding),
        credentials_reloaded_from_trusted_config=True,
    )
    loader = _Loader(runtime)
    resolver = ContinuationRuntimeResolver(
        loader,
        clock=lambda: _CLAIMED_AT,
    )
    return _Harness(
        request=request,
        record=DurableContinuationRecord(
            continuation=continuation,
            task_run_id="task-run",
            checkpoint_id="checkpoint",
        ),
        store=store,
        adapter=adapter,
        executor=executor,
        loader=loader,
        resumer=DurableAgentContinuationResumer(
            store,
            resolver,
            clock=lambda: _CLAIMED_AT,
        ),
    )


async def _admit(
    harness: _Harness,
) -> DurableAgentContinuationAdmission:
    return await harness.resumer.admit(
        harness.record,
        actor=InteractionActor(principal=harness.request.origin.principal),
        expected_request_id=harness.request.request_id,
        expected_run_id=harness.request.origin.run_id,
        expected_checkpoint_id="checkpoint",
        owner_id=ContinuationClaimOwnerId("worker-claim"),
        lease_expires_at=_LEASE_EXPIRES_AT,
        dispatch_id=_DISPATCH_ID,
    )


@_async_test
async def test_resume_claims_restores_dispatches_and_completes_once() -> None:
    harness = _harness()

    admission = await _admit(harness)
    first = await admission.dispatch()
    second = await admission.dispatch()
    completed = await admission.complete(_RESULT_DIGEST)
    replayed = await admission.complete(_RESULT_DIGEST)

    assert first == second == {"answer": "resumed"}
    assert completed is replayed
    assert admission.state is DurableContinuationResumeState.COMPLETED
    assert harness.adapter.imported == [
        harness.record.continuation.provider_snapshot
    ]
    assert harness.adapter.validated == harness.adapter.imported
    assert len(harness.executor.commands) == 1
    command = harness.executor.commands[0]
    assert (
        str(command.correlated_result.call_id)
        == harness.record.continuation.provider_call_correlation_id
    )
    assert harness.store.calls == [
        "lookup",
        "claim",
        "mark_dispatching",
        "mark_dispatched",
        "complete",
    ]


@_async_test
async def test_listener_registration_is_exclusive_and_pre_dispatch() -> None:
    harness = _harness()
    admission = await _admit(harness)

    registration = admission.register_event_listener(lambda event: None)

    assert harness.executor.event_listener_register_calls == 1
    assert harness.executor.commands == []
    with pytest.raises(InputValidationError, match="exclusive pre-dispatch"):
        admission.register_event_listener(lambda event: None)
    await admission.dispatch()
    with pytest.raises(InputValidationError, match="exclusive pre-dispatch"):
        admission.register_event_listener(lambda event: None)
    registration.close()
    registration.close()
    concrete = harness.executor.event_listener_registrations[0]
    assert concrete.close_calls == 1
    assert concrete.remove_calls == 1


@_async_test
async def test_listener_registration_failure_precedes_provider_dispatch() -> (
    None
):
    harness = _harness()
    admission = await _admit(harness)
    harness.executor.event_listener_registration_error = RuntimeError(
        "listener registration failed"
    )

    with pytest.raises(RuntimeError, match="listener registration failed"):
        admission.register_event_listener(lambda event: None)

    assert harness.executor.commands == []
    assert "mark_dispatching" not in harness.store.calls
    await admission.release()
    await admission.close()
    assert harness.executor.close_calls == 1


@_async_test
async def test_cancelled_caller_retains_listener_until_dispatch_settles() -> (
    None
):
    harness = _harness()
    harness.executor.started = Event()
    harness.executor.proceed = Event()
    admission = await _admit(harness)
    registration = admission.register_event_listener(lambda event: None)
    dispatch = create_task(admission.dispatch())
    await harness.executor.started.wait()

    dispatch.cancel()
    with pytest.raises(CancelledError):
        await dispatch
    registration.close()

    concrete = harness.executor.event_listener_registrations[0]
    assert concrete.remove_calls == 0
    harness.executor.proceed.set()
    assert (
        await admission.wait_dispatch_settled()
        is DurableContinuationResumeState.DISPATCHED
    )
    await concrete.closed_signal.wait()
    assert concrete.remove_calls == 1
    assert dispatch.done()
    await admission.close()


@_async_test
@pytest.mark.parametrize(
    ("continuation_id", "dispatch_id"),
    (
        ("other-continuation", "dispatch"),
        ("continuation", "other-dispatch"),
    ),
)
async def test_resume_rejects_provider_key_from_mutated_durable_identity(
    continuation_id: str,
    dispatch_id: str,
) -> None:
    harness = _harness()
    continuation = harness.record.continuation
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    changed_snapshot = replace(
        snapshot,
        provider_idempotency_key=derive_provider_idempotency_key(
            ContinuationId(continuation_id),
            ContinuationDispatchId(dispatch_id),
        ),
    )
    changed = replace(
        continuation,
        provider_snapshot=changed_snapshot,
    )
    record = replace(
        harness.record,
        continuation=changed,
    )
    harness.store.continuation = changed

    with pytest.raises(InputValidationError) as raised:
        await harness.resumer.admit(
            record,
            actor=InteractionActor(principal=harness.request.origin.principal),
            expected_request_id=harness.request.request_id,
            expected_run_id=harness.request.origin.run_id,
            expected_checkpoint_id="checkpoint",
            owner_id=ContinuationClaimOwnerId("worker-claim"),
            lease_expires_at=_LEASE_EXPIRES_AT,
            dispatch_id=_DISPATCH_ID,
        )

    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert "provider_idempotency_key" in raised.value.path
    assert harness.store.calls == ["lookup"]
    assert harness.adapter.imported == []
    assert harness.executor.commands == []


@_async_test
async def test_cancelled_loader_owns_release_through_repeat_cancellation() -> (
    None
):
    harness = _harness()
    harness.loader.started = Event()
    harness.loader.proceed = Event()
    harness.store.release_started = Event()
    harness.store.release_proceed = Event()
    harness.store.release_completed = Event()
    running = create_task(_admit(harness))
    await harness.loader.started.wait()

    assert running.cancel("cancel runtime reconstruction")
    await harness.store.release_started.wait()
    assert harness.store.release_attempts == 1
    assert running.cancel("cancel release start")
    assert not running.done()
    harness.store.release_proceed.set()
    await harness.store.release_completed.wait()
    assert running.cancel("cancel release completion")

    with pytest.raises(CancelledError, match="runtime reconstruction"):
        await running

    assert running.cancelled()
    assert harness.store.continuation.claim.state is (
        ContinuationClaimState.FAILED_SAFE_TO_RETRY
    )
    assert harness.store.release_attempts == 1
    assert harness.store.calls.count("release") == 1
    assert harness.executor.close_calls == 0
    assert not tuple(
        task
        for task in all_tasks()
        if task.get_name().startswith("durable-agent-admission-cleanup-")
    )


@_async_test
async def test_near_expiry_claim_binds_before_cold_load_and_clamps() -> None:
    harness = _harness()
    deadline = _CLAIMED_AT + timedelta(seconds=5)
    continuation = replace(
        harness.record.continuation,
        expires_at=deadline,
    )
    harness.store.continuation = continuation
    harness = replace(
        harness,
        record=replace(harness.record, continuation=continuation),
    )
    current = [_CLAIMED_AT]
    harness = replace(
        harness,
        resumer=DurableAgentContinuationResumer(
            harness.store,
            ContinuationRuntimeResolver(
                harness.loader,
                clock=lambda: current[0],
            ),
            clock=lambda: current[0],
        ),
    )
    harness.loader.started = Event()
    harness.loader.proceed = Event()
    observed: list[DurableAgentContinuationClaimLease] = []

    async def current_task_lease() -> datetime:
        return deadline + timedelta(minutes=10)

    async def observe(
        claim_lease: DurableAgentContinuationClaimLease,
    ) -> None:
        observed.append(claim_lease)

    running = create_task(
        harness.resumer.admit(
            harness.record,
            actor=InteractionActor(principal=harness.request.origin.principal),
            expected_request_id=harness.request.request_id,
            expected_run_id=harness.request.origin.run_id,
            expected_checkpoint_id="checkpoint",
            owner_id=ContinuationClaimOwnerId("worker-claim"),
            lease_expires_at=deadline + timedelta(minutes=10),
            dispatch_id=_DISPATCH_ID,
            lease_expires_at_provider=current_task_lease,
            claim_lease_observer=observe,
        )
    )
    await harness.loader.started.wait()

    assert len(observed) == 1
    assert harness.store.continuation.claim.lease_expires_at == deadline
    current[0] = deadline
    harness.loader.proceed.set()
    with pytest.raises(InputValidationError) as raised:
        await running

    assert raised.value.code is InputErrorCode.EXPIRED
    assert harness.store.continuation.claim.state is (
        ContinuationClaimState.FAILED_SAFE_TO_RETRY
    )
    assert harness.store.calls == ["lookup", "claim", "release"]
    assert harness.executor.close_calls == 1


@_async_test
async def test_expired_cold_loader_cleanup_survives_repeat_cancels() -> None:
    harness = _harness()
    deadline = _CLAIMED_AT + timedelta(seconds=5)
    continuation = replace(
        harness.record.continuation,
        expires_at=deadline,
    )
    harness.store.continuation = continuation
    current = [_CLAIMED_AT]
    harness = replace(
        harness,
        record=replace(harness.record, continuation=continuation),
        resumer=DurableAgentContinuationResumer(
            harness.store,
            ContinuationRuntimeResolver(
                harness.loader,
                clock=lambda: current[0],
            ),
            clock=lambda: current[0],
        ),
    )
    harness.loader.started = Event()
    harness.loader.proceed = Event()
    harness.store.release_started = Event()
    harness.store.release_proceed = Event()
    harness.executor.close_started = Event()
    harness.executor.close_proceed = Event()
    harness.executor.close_completed = Event()
    running = create_task(_admit(harness))
    await harness.loader.started.wait()

    current[0] = deadline
    harness.loader.proceed.set()
    await harness.store.release_started.wait()
    assert running.cancel("cancel release start")
    assert not running.done()
    harness.store.release_proceed.set()
    await harness.executor.close_started.wait()
    assert harness.store.continuation.claim.state is (
        ContinuationClaimState.FAILED_SAFE_TO_RETRY
    )
    assert running.cancel("cancel runtime close start")
    assert not running.done()
    harness.executor.close_proceed.set()
    await harness.executor.close_completed.wait()
    assert running.cancel("cancel runtime close completion")

    with pytest.raises(InputValidationError) as raised:
        await running

    assert raised.value.code is InputErrorCode.EXPIRED
    assert raised.value.path == "resume.continuation.expires_at"
    assert not running.cancelled()
    assert harness.store.release_attempts == 1
    assert harness.store.calls.count("release") == 1
    assert harness.executor.close_calls == 1
    assert not tuple(
        task
        for task in all_tasks()
        if task.get_name().startswith("durable-agent-admission-cleanup-")
    )


@_async_test
async def test_dispatch_started_before_deadline_ignores_late_renewal() -> None:
    harness = _harness()
    deadline = _CLAIMED_AT + timedelta(seconds=5)
    continuation = replace(
        harness.record.continuation,
        expires_at=deadline,
    )
    harness.store.continuation = continuation
    harness = replace(
        harness,
        record=replace(harness.record, continuation=continuation),
    )
    current = [_CLAIMED_AT]
    harness = replace(
        harness,
        resumer=DurableAgentContinuationResumer(
            harness.store,
            ContinuationRuntimeResolver(
                harness.loader,
                clock=lambda: current[0],
            ),
            clock=lambda: current[0],
        ),
    )
    observed: list[DurableAgentContinuationClaimLease] = []

    async def observe(
        claim_lease: DurableAgentContinuationClaimLease,
    ) -> None:
        observed.append(claim_lease)

    admission = await harness.resumer.admit(
        harness.record,
        actor=InteractionActor(principal=harness.request.origin.principal),
        expected_request_id=harness.request.request_id,
        expected_run_id=harness.request.origin.run_id,
        expected_checkpoint_id="checkpoint",
        owner_id=ContinuationClaimOwnerId("worker-claim"),
        lease_expires_at=deadline,
        dispatch_id=_DISPATCH_ID,
        claim_lease_observer=observe,
    )
    harness.executor.started = Event()
    harness.executor.proceed = Event()
    dispatch = create_task(admission.dispatch())
    await harness.executor.started.wait()

    assert harness.store.calls[-1] == "mark_dispatching"
    assert len(observed) == 1
    current[0] = deadline
    assert not await observed[0].renew(
        deadline + timedelta(minutes=1),
        now=current[0],
    )
    assert not dispatch.done()

    harness.executor.proceed.set()
    assert await dispatch == {"answer": "resumed"}
    assert len(harness.executor.commands) == 1
    assert harness.store.calls.count("mark_dispatching") == 1
    assert harness.store.calls.count("mark_dispatched") == 1


@_async_test
async def test_resume_command_rejects_tampered_correlated_result_payload() -> (
    None
):
    harness = _harness()
    admission = await _admit(harness)
    command = admission.command

    with pytest.raises(InputValidationError) as raised:
        replace(
            command,
            correlated_result=replace(
                command.correlated_result,
                payload={"tampered": True},
            ),
        )

    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH


@_async_test
async def test_snapshot_call_tampering_rejects_before_provider_dispatch() -> (
    None
):
    harness = _harness()
    harness.adapter.validation_failure = InputValidationError(
        InputErrorCode.CORRELATION_MISMATCH,
        "continuation_snapshot.payload.replay_items",
        "reserved call changed",
    )

    with pytest.raises(InputValidationError) as raised:
        await _admit(harness)

    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert harness.adapter.validated == [
        harness.record.continuation.provider_snapshot
    ]
    assert harness.adapter.imported == []
    assert harness.executor.commands == []
    assert harness.executor.close_calls == 1
    assert harness.store.calls == ["lookup", "claim", "release"]


@_async_test
async def test_setup_failure_reports_stale_release_cleanup() -> None:
    harness = _harness()
    harness.loader.failure = RuntimeError("setup failed")
    harness.store.release_returns_no_op = True

    with pytest.raises(BaseExceptionGroup) as raised:
        await _admit(harness)

    assert len(raised.value.exceptions) == 2
    setup_error, cleanup_error = raised.value.exceptions
    assert isinstance(setup_error, RuntimeError)
    assert str(setup_error) == "setup failed"
    assert isinstance(cleanup_error, InputValidationError)
    assert cleanup_error.code is InputErrorCode.CORRELATION_MISMATCH
    assert cleanup_error.path == "resume.release"
    assert harness.store.continuation.claim.state is (
        ContinuationClaimState.CLAIMED_PRE_DISPATCH
    )
    assert harness.store.calls == ["lookup", "claim", "release"]


@_async_test
@pytest.mark.parametrize(
    ("release_fails", "close_fails"),
    (
        (True, False),
        (False, True),
        (True, True),
    ),
)
async def test_setup_cleanup_aggregates_release_and_close_errors_in_order(
    release_fails: bool,
    close_fails: bool,
) -> None:
    harness = _harness()
    setup_error = RuntimeError("setup failed")
    release_error = RuntimeError("release failed")
    close_error = RuntimeError("close failed")
    harness.adapter.validation_failure = setup_error
    if release_fails:
        harness.store.failures["release"] = release_error
    if close_fails:
        harness.executor.close_failure = close_error

    with pytest.raises(BaseExceptionGroup) as raised:
        await _admit(harness)

    expected_errors = (
        setup_error,
        *((release_error,) if release_fails else ()),
        *((close_error,) if close_fails else ()),
    )
    assert raised.value.exceptions == expected_errors
    assert harness.store.release_attempts == 1
    assert harness.executor.close_calls == 1


@_async_test
async def test_setup_cleanup_preserves_nested_process_control_errors() -> None:
    controls: tuple[tuple[str, BaseException], ...] = (
        ("release", KeyboardInterrupt("interrupt release")),
        ("close", SystemExit(7)),
        (
            "release",
            BaseExceptionGroup(
                "nested release control",
                (
                    RuntimeError("nested release failure"),
                    BaseExceptionGroup(
                        "nested process control",
                        (SystemExit(9),),
                    ),
                ),
            ),
        ),
    )
    for stage, control_error in controls:
        harness = _harness()
        setup_error = RuntimeError(f"setup failed before {stage}")
        harness.adapter.validation_failure = setup_error
        if stage == "release":
            harness.store.failures["release"] = control_error
        else:
            harness.executor.close_failure = control_error

        with pytest.raises(BaseExceptionGroup) as raised:
            await _admit(harness)

        assert raised.value.exceptions == (setup_error, control_error)
        assert harness.store.release_attempts == 1
        assert harness.executor.close_calls == 1

    harness = _harness()
    nested_setup_error = BaseExceptionGroup(
        "nested setup control",
        (
            RuntimeError("setup failed"),
            BaseExceptionGroup(
                "nested process control",
                (KeyboardInterrupt("interrupt setup"),),
            ),
        ),
    )
    release_error = RuntimeError("release failed")
    harness.adapter.validation_failure = nested_setup_error
    harness.store.failures["release"] = release_error

    with pytest.raises(BaseExceptionGroup) as raised:
        await _admit(harness)

    assert raised.value.exceptions == (nested_setup_error, release_error)
    assert harness.store.release_attempts == 1
    assert harness.executor.close_calls == 1


@_async_test
async def test_setup_cleanup_reports_owned_child_self_cancellation() -> None:
    harness = _harness()
    setup_error = RuntimeError("setup failed")
    harness.adapter.validation_failure = setup_error
    harness.executor.self_cancel_on_close_completion = True

    with pytest.raises(BaseExceptionGroup) as raised:
        await _admit(harness)

    assert raised.value.exceptions[0] is setup_error
    assert isinstance(raised.value.exceptions[1], CancelledError)
    assert len(raised.value.exceptions) == 2
    assert harness.store.continuation.claim.state is (
        ContinuationClaimState.FAILED_SAFE_TO_RETRY
    )
    assert harness.store.release_attempts == 1
    assert harness.executor.close_calls == 1
    assert not tuple(
        task
        for task in all_tasks()
        if task.get_name().startswith("durable-agent-admission-cleanup-")
    )


@_async_test
async def test_openai_boundary_tampering_never_invokes_provider() -> None:
    harness = _harness()
    snapshot = _openai_snapshot(harness.request, _binding())
    replay_items = snapshot.payload["replay_items"]
    assert isinstance(replay_items, tuple)
    mutable_items = [
        dict(cast(Mapping[str, object], item)) for item in replay_items
    ]
    current_call = dict(mutable_items[-1])
    current_call["name"] = "other"
    current_call["arguments"] = dumps(
        {
            "mode": harness.request.mode.value,
            "reason": "Tampered reason.",
            "questions": [
                encode_input_question(question)
                for question in harness.request.questions
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    mutable_items[-1] = current_call
    tampered_snapshot = replace(
        snapshot,
        payload={
            **snapshot.payload,
            "replay_items": cast(JsonValue, tuple(mutable_items)),
        },
    )
    continuation = replace(
        harness.record.continuation,
        provider_snapshot=tampered_snapshot,
    )
    harness.store.continuation = continuation
    target = _openai_client()
    harness.loader.runtime = replace(
        harness.loader.runtime,
        model=target,
    )
    harness = replace(
        harness,
        record=replace(
            harness.record,
            continuation=continuation,
        ),
    )

    with pytest.raises(InputValidationError) as raised:
        await _admit(harness)

    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert harness.executor.commands == []
    assert harness.executor.close_calls == 1
    assert target._replay_owners_by_call_id == {}  # noqa: SLF001
    assert harness.store.calls == ["lookup", "claim", "release"]


@_async_test
@pytest.mark.parametrize(
    (
        "task_run_id",
        "checkpoint_id",
        "expected_run_id",
        "expected_checkpoint_id",
        "expected_path",
    ),
    (
        (
            None,
            None,
            "task-run",
            "checkpoint",
            "task_run_id",
        ),
        (
            "task-run",
            "checkpoint",
            "other-run",
            "checkpoint",
            "task_run_id",
        ),
        (
            "task-run",
            "other-checkpoint",
            "task-run",
            "checkpoint",
            "checkpoint_id",
        ),
    ),
)
async def test_resume_rejects_wrong_record_binding_before_claim(
    task_run_id: str | None,
    checkpoint_id: str | None,
    expected_run_id: str,
    expected_checkpoint_id: str,
    expected_path: str,
) -> None:
    harness = _harness()
    record = replace(
        harness.record,
        task_run_id=task_run_id,
        checkpoint_id=checkpoint_id,
    )

    with pytest.raises(InputValidationError) as raised:
        await harness.resumer.admit(
            record,
            actor=InteractionActor(principal=harness.request.origin.principal),
            expected_request_id=harness.request.request_id,
            expected_run_id=RunId(expected_run_id),
            expected_checkpoint_id=expected_checkpoint_id,
            owner_id=ContinuationClaimOwnerId("worker-claim"),
            lease_expires_at=_LEASE_EXPIRES_AT,
            dispatch_id=_DISPATCH_ID,
        )

    assert expected_path in raised.value.path
    assert harness.store.calls == []


@pytest.mark.parametrize(
    "task_run_id",
    (
        None,
        "other-run",
    ),
)
def test_continuation_record_rejects_impossible_task_binding(
    task_run_id: str | None,
) -> None:
    harness = _harness()

    with pytest.raises(InputValidationError):
        replace(harness.record, task_run_id=task_run_id)


def _wrong_owner(claimed: PortableContinuation) -> PortableContinuation:
    return replace(
        claimed,
        claim=replace(
            claimed.claim,
            owner_id=ContinuationClaimOwnerId("other-owner"),
        ),
    )


def _wrong_dispatch(claimed: PortableContinuation) -> PortableContinuation:
    assert claimed.dispatch is not None
    return replace(
        claimed,
        dispatch=replace(
            claimed.dispatch,
            dispatch_id=ContinuationDispatchId("other-dispatch"),
        ),
    )


def _wrong_fence(claimed: PortableContinuation) -> PortableContinuation:
    return replace(
        claimed,
        fencing_token=ContinuationFencingToken(int(claimed.fencing_token) + 1),
    )


def _wrong_correlation(
    claimed: PortableContinuation,
) -> PortableContinuation:
    return replace(
        claimed,
        provider_call_correlation_id="other-call",
    )


def _wrong_definition(
    claimed: PortableContinuation,
) -> PortableContinuation:
    definition = _definition(revision="agent-r2")
    return replace(
        claimed,
        origin=replace(claimed.origin, definition=definition),
        definition=definition,
    )


def _wrong_snapshot(claimed: PortableContinuation) -> PortableContinuation:
    snapshot = claimed.provider_snapshot
    assert snapshot is not None
    changed = replace(
        snapshot,
        provider_idempotency_key=ProviderIdempotencyKey("other-key"),
    )
    dispatch = claimed.dispatch
    assert dispatch is not None
    return replace(
        claimed,
        provider_snapshot=changed,
        dispatch=replace(
            dispatch,
            provider_idempotency_key=changed.provider_idempotency_key,
        ),
    )


@_async_test
@pytest.mark.parametrize(
    "mutator",
    (
        _wrong_owner,
        _wrong_dispatch,
        _wrong_fence,
        _wrong_correlation,
        _wrong_definition,
        _wrong_snapshot,
    ),
)
async def test_resume_rejects_hostile_claim_receipt(
    mutator: ClaimMutator,
) -> None:
    harness = _harness()
    harness.store.claim_mutator = mutator

    with pytest.raises(InputValidationError) as raised:
        await _admit(harness)

    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert harness.adapter.imported == []
    assert harness.executor.commands == []


@_async_test
async def test_dispatch_marker_failure_releases_pre_dispatch_claim() -> None:
    harness = _harness()
    admission = await _admit(harness)
    harness.store.failures["mark_dispatching"] = RuntimeError("database")

    with pytest.raises(RuntimeError, match="database"):
        await admission.dispatch()

    assert admission.state is DurableContinuationResumeState.RELEASED
    assert harness.executor.commands == []
    assert harness.store.calls[-1] == "release"


@_async_test
async def test_release_if_pre_dispatch_is_atomic_and_idempotent() -> None:
    harness = _harness()
    admission = await _admit(harness)

    assert (
        await admission.wait_dispatch_settled()
        is DurableContinuationResumeState.ADMITTED
    )
    assert await admission.release_if_pre_dispatch()
    assert await admission.release_if_pre_dispatch()
    assert admission.state is DurableContinuationResumeState.RELEASED
    assert harness.store.calls.count("release") == 1
    with pytest.raises(InputValidationError):
        await admission.dispatch()


@_async_test
async def test_release_ownership_excludes_racing_dispatch() -> None:
    harness = _harness()
    harness.store.release_started = Event()
    harness.store.release_proceed = Event()
    admission = await _admit(harness)

    release_task = create_task(admission.release())
    await harness.store.release_started.wait()
    with pytest.raises(InputValidationError, match="release ownership"):
        await admission.dispatch()
    harness.store.release_proceed.set()
    released = await release_task

    assert released.claim.state is ContinuationClaimState.FAILED_SAFE_TO_RETRY
    assert admission.state is DurableContinuationResumeState.RELEASED
    assert harness.executor.commands == []


@_async_test
async def test_transient_release_failure_never_enables_dispatch() -> None:
    harness = _harness()
    admission = await _admit(harness)
    harness.store.failures["release"] = RuntimeError("transient")

    with pytest.raises(RuntimeError, match="transient"):
        await admission.release()
    with pytest.raises(InputValidationError, match="release ownership"):
        await admission.dispatch()

    released = await admission.release()
    assert released.claim.state is ContinuationClaimState.FAILED_SAFE_TO_RETRY
    assert admission.state is DurableContinuationResumeState.RELEASED
    assert harness.executor.commands == []


@_async_test
async def test_cancelled_dispatch_is_owned_until_explicit_settlement() -> None:
    harness = _harness()
    harness.executor.started = Event()
    harness.executor.proceed = Event()
    admission = await _admit(harness)

    dispatch_task = create_task(admission.dispatch())
    await harness.executor.started.wait()
    dispatch_task.cancel()
    with pytest.raises(CancelledError):
        await dispatch_task

    assert not await admission.release_if_pre_dispatch()
    assert admission.state is DurableContinuationResumeState.DISPATCHING
    harness.executor.proceed.set()
    settled = await admission.wait_dispatch_settled()

    assert settled is DurableContinuationResumeState.DISPATCHED
    assert len(harness.executor.commands) == 1
    assert harness.store.calls.count("mark_dispatched") == 1


@_async_test
async def test_interrupt_releases_before_dispatch_and_is_idempotent() -> None:
    harness = _harness()
    admission = await _admit(harness)

    first = await admission.interrupt_dispatch()
    second = await admission.interrupt_dispatch()

    assert first is second is DurableContinuationResumeState.RELEASED
    assert harness.store.calls.count("release") == 1
    assert harness.executor.commands == []


@_async_test
async def test_interrupt_fences_in_flight_dispatch_as_ambiguous() -> None:
    harness = _harness()
    harness.executor.started = Event()
    harness.executor.proceed = Event()
    admission = await _admit(harness)
    dispatch_task = create_task(admission.dispatch())
    await harness.executor.started.wait()

    settled = await admission.interrupt_dispatch()
    with pytest.raises(CancelledError):
        await dispatch_task

    assert settled is DurableContinuationResumeState.AMBIGUOUS
    assert admission.continuation.claim.state is (
        ContinuationClaimState.DISPATCHED_AMBIGUOUS
    )
    assert len(harness.executor.commands) == 1
    assert harness.store.calls.count("mark_dispatching") == 1
    assert "mark_dispatched" not in harness.store.calls
    assert "release" not in harness.store.calls


@_async_test
async def test_dispatch_marker_and_release_failure_is_ambiguous() -> None:
    harness = _harness()
    admission = await _admit(harness)
    harness.store.failures["mark_dispatching"] = RuntimeError("database")
    harness.store.failures["release"] = RuntimeError("release")

    with pytest.raises(BaseExceptionGroup) as raised:
        await admission.dispatch()

    assert len(raised.value.exceptions) == 2
    assert admission.state is DurableContinuationResumeState.AMBIGUOUS
    assert harness.executor.commands == []


@_async_test
@pytest.mark.parametrize(
    "operation",
    ("executor", "mark_dispatched"),
)
async def test_post_marker_failure_remains_ambiguous(
    operation: str,
) -> None:
    harness = _harness()
    admission = await _admit(harness)
    if operation == "executor":
        harness.executor.failure = RuntimeError("provider")
    else:
        harness.store.failures["mark_dispatched"] = RuntimeError("database")

    with pytest.raises(RuntimeError):
        await admission.dispatch()

    assert admission.state is DurableContinuationResumeState.AMBIGUOUS
    assert "release" not in harness.store.calls


@_async_test
async def test_completion_failure_is_retryable_after_known_dispatch() -> None:
    harness = _harness()
    admission = await _admit(harness)
    await admission.dispatch()
    harness.store.failures["complete"] = RuntimeError("database")

    with pytest.raises(BaseExceptionGroup):
        await admission.complete(_RESULT_DIGEST)

    assert admission.state is DurableContinuationResumeState.DISPATCHED
    completed = await admission.complete(_RESULT_DIGEST)
    assert completed.completion is not None
    assert (
        admission.state.value == DurableContinuationResumeState.COMPLETED.value
    )


@_async_test
async def test_completion_readback_recovers_lost_success_response() -> None:
    harness = _harness()
    admission = await _admit(harness)
    await admission.dispatch()
    harness.store.complete_after_failure = RuntimeError("lost response")

    completed = await admission.complete(_RESULT_DIGEST)

    assert completed.completion is not None
    assert completed.completion.result_digest == _RESULT_DIGEST
    assert admission.state is DurableContinuationResumeState.COMPLETED


@_async_test
async def test_concurrent_completion_pins_one_digest_during_dispatch() -> None:
    harness = _harness()
    harness.executor.started = Event()
    harness.executor.proceed = Event()
    admission = await _admit(harness)
    dispatch_task = create_task(admission.dispatch())
    await harness.executor.started.wait()

    first_digest = "b" * 64
    second_digest = "c" * 64
    first = create_task(admission.complete(first_digest))
    await sleep(0)
    second = create_task(admission.complete(second_digest))
    await sleep(0)
    harness.executor.proceed.set()

    assert await dispatch_task == {"answer": "resumed"}
    outcomes = await gather(first, second, return_exceptions=True)
    completed = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, PortableContinuation)
    ]
    rejected = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, InputValidationError)
    ]

    assert len(completed) == 1
    assert len(rejected) == 1
    assert rejected[0].code is InputErrorCode.CORRELATION_MISMATCH
    assert rejected[0].path == "resume.result_digest"
    completion = completed[0].completion
    assert completion is not None
    assert completion.result_digest == first_digest
    assert harness.store.continuation.completion == completion
    assert admission.completion_command(first_digest).result_digest == (
        completion.result_digest
    )
    with pytest.raises(InputValidationError):
        admission.completion_command(second_digest)
    assert harness.store.calls.count("complete") == 1


@_async_test
async def test_completion_command_pins_digest_across_owner_clearing() -> None:
    harness = _harness()
    admission = await _admit(harness)
    await admission.dispatch()

    first = admission.completion_command(_RESULT_DIGEST)
    with pytest.raises(InputValidationError):
        admission.completion_command("b" * 64)
    await admission.complete(_RESULT_DIGEST)
    assert admission.continuation.claim.owner_id is None

    replayed = admission.completion_command(_RESULT_DIGEST)
    assert replayed == first
    assert replayed.result_digest == first.result_digest
    assert replayed.owner_id == first.owner_id
    with pytest.raises(InputValidationError):
        admission.completion_command("b" * 64)


@_async_test
async def test_rejection_command_fences_exact_pre_dispatch_admission() -> None:
    harness = _harness()
    admission = await _admit(harness)

    command = admission.rejection_command(_RESULT_DIGEST)

    assert command.continuation_id == admission.continuation.continuation_id
    assert (
        command.expected_store_revision
        == admission.continuation.store_revision
    )
    assert command.owner_id == admission.continuation.claim.owner_id
    assert command.fencing_token == admission.continuation.fencing_token
    assert command.result_digest == _RESULT_DIGEST
    assert admission.state is DurableContinuationResumeState.ADMITTED


@_async_test
async def test_rejection_command_loses_to_release_or_dispatch_ownership() -> (
    None
):
    release_harness = _harness()
    release_admission = await _admit(release_harness)
    release_harness.store.release_started = Event()
    release_harness.store.release_proceed = Event()
    release_task = create_task(release_admission.release())
    await release_harness.store.release_started.wait()

    with pytest.raises(InputValidationError, match="pre-dispatch"):
        release_admission.rejection_command(_RESULT_DIGEST)
    release_harness.store.release_proceed.set()
    await release_task

    dispatch_harness = _harness()
    dispatch_admission = await _admit(dispatch_harness)
    await dispatch_admission.dispatch()

    with pytest.raises(InputValidationError, match="pre-dispatch"):
        dispatch_admission.rejection_command(_RESULT_DIGEST)


@_async_test
@pytest.mark.parametrize(
    ("boundary", "expected_state"),
    (
        ("loader", None),
        ("mark_dispatching", DurableContinuationResumeState.RELEASED),
        ("executor", DurableContinuationResumeState.AMBIGUOUS),
        ("mark_dispatched", DurableContinuationResumeState.AMBIGUOUS),
        ("complete", DurableContinuationResumeState.DISPATCHED),
    ),
)
async def test_cancellation_preserves_boundary_semantics(
    boundary: str,
    expected_state: DurableContinuationResumeState | None,
) -> None:
    harness = _harness()
    if boundary == "loader":
        harness.loader.failure = CancelledError()
        with pytest.raises(CancelledError):
            await _admit(harness)
        assert harness.store.calls[-1] == "release"
        return
    admission = await _admit(harness)
    if boundary == "mark_dispatching":
        harness.store.failures[boundary] = CancelledError()
        with pytest.raises(CancelledError):
            await admission.dispatch()
    elif boundary == "executor":
        harness.executor.failure = CancelledError()
        with pytest.raises(CancelledError):
            await admission.dispatch()
    elif boundary == "mark_dispatched":
        harness.store.failures[boundary] = CancelledError()
        with pytest.raises(CancelledError):
            await admission.dispatch()
    else:
        await admission.dispatch()
        harness.store.failures[boundary] = CancelledError()
        with pytest.raises(BaseExceptionGroup):
            await admission.complete(_RESULT_DIGEST)
    assert admission.state is expected_state


@_async_test
async def test_resume_command_and_admission_reject_every_drifted_component() -> (  # noqa: E501
    None
):
    harness = _harness()
    admission = await _admit(harness)
    command = admission.command
    other_request = _terminal_request(
        _origin(_definition(), run_id="other-task-run")
    )
    cases = (
        (
            "resume.command.continuation",
            {"continuation": cast(Any, object())},
        ),
        (
            "resume.command.continuation.claim.state",
            {"continuation": harness.record.continuation},
        ),
        (
            "resume.command.request",
            {"request": cast(Any, object())},
        ),
        (
            "resume.command.request",
            {"request": other_request},
        ),
        (
            "resume.command.model_result",
            {"model_result": cast(Any, None)},
        ),
        (
            "resume.command.task_input_call",
            {"task_input_call": cast(Any, object())},
        ),
        (
            "resume.command.task_input_call",
            {
                "task_input_call": replace(
                    command.task_input_call,
                    reason="A changed reason.",
                )
            },
        ),
        (
            "resume.command.correlated_result",
            {"correlated_result": cast(Any, object())},
        ),
        (
            "resume.command.correlated_result",
            {
                "correlated_result": replace(
                    command.correlated_result,
                    payload={},
                )
            },
        ),
        (
            "resume.command.resolved_runtime",
            {"resolved_runtime": cast(Any, object())},
        ),
        (
            "resume.command.resolved_runtime",
            {
                "resolved_runtime": replace(
                    command.resolved_runtime,
                    definition=_definition(revision="agent-r2"),
                )
            },
        ),
    )
    for expected_path, changes in cases:
        with pytest.raises(InputValidationError) as raised:
            replace(command, **changes)
        assert raised.value.path == expected_path

    common = {
        "store": harness.store,
        "command": command,
        "owner_id": ContinuationClaimOwnerId("worker-claim"),
        "fencing_token": command.continuation.fencing_token,
        "executor": harness.executor,
        "clock": lambda: _CLAIMED_AT,
    }
    constructor_cases = (
        ("resume.store", {"store": object()}),
        ("resume.admission.command", {"command": object()}),
        (
            "resume.admission.fencing_token",
            {"fencing_token": ContinuationFencingToken(2)},
        ),
        ("resume.runtime.executor", {"executor": object()}),
        ("resume.admission.clock", {"clock": object()}),
    )
    for expected_path, changes in constructor_cases:
        arguments = {**common, **changes}
        with pytest.raises(InputValidationError) as raised:
            DurableAgentContinuationAdmission(**cast(Any, arguments))
        assert raised.value.path == expected_path

    with pytest.raises(InputValidationError) as raised:
        admission.register_event_listener(cast(Any, object()))
    assert raised.value.path == "resume.event_listener"


@_async_test
async def test_admission_wait_interrupt_and_marker_guards_fail_closed() -> (
    None
):
    async def return_result() -> object:
        return {"ok": True}

    async def fail_result() -> object:
        raise RuntimeError("dispatch failed")

    failed_harness = _harness()
    failed_admission = await _admit(failed_harness)
    failed_task = create_task(fail_result())
    await sleep(0)
    failed_admission._dispatch_task = failed_task
    failed_admission._state = DurableContinuationResumeState.AMBIGUOUS
    assert (
        await failed_admission.wait_dispatch_settled()
        is DurableContinuationResumeState.AMBIGUOUS
    )

    pending_harness = _harness()
    pending_admission = await _admit(pending_harness)
    blocker = Event()
    pending_task = create_task(blocker.wait())
    pending_admission._dispatch_task = cast(Any, pending_task)
    waiter = create_task(pending_admission.wait_dispatch_settled())
    await sleep(0)
    waiter.cancel()
    with pytest.raises(CancelledError):
        await waiter
    pending_task.cancel()
    with pytest.raises(CancelledError):
        await pending_task

    unsettled_harness = _harness()
    unsettled_admission = await _admit(unsettled_harness)
    completed_task = create_task(return_result())
    await completed_task
    unsettled_admission._dispatch_task = completed_task
    with pytest.raises(
        InputValidationError,
        match="without durable settlement",
    ):
        await unsettled_admission.wait_dispatch_settled()

    interrupted_harness = _harness()
    interrupted_admission = await _admit(interrupted_harness)
    with patch.object(
        interrupted_admission,
        "release_if_pre_dispatch",
        new=AsyncMock(return_value=False),
    ):
        with pytest.raises(
            InputValidationError,
            match="dispatch changed while interruption",
        ):
            await interrupted_admission.interrupt_dispatch()

    missing_settlement_harness = _harness()
    missing_settlement_admission = await _admit(missing_settlement_harness)
    missing_settlement_admission._state = (
        DurableContinuationResumeState.DISPATCHING
    )
    with pytest.raises(
        InputValidationError,
        match="lacks a durable settlement",
    ):
        await missing_settlement_admission.interrupt_dispatch()

    marker_harness = _harness()
    marker_admission = await _admit(marker_harness)

    async def invalid_marker(*_args: object, **_kwargs: object) -> object:
        return marker_admission.continuation

    marker_harness.store.mark_dispatching = cast(Any, invalid_marker)
    with pytest.raises(InputValidationError, match="dispatch marker"):
        await marker_admission.dispatch()
    assert marker_admission.state is DurableContinuationResumeState.AMBIGUOUS


@_async_test
async def test_admission_completion_rejection_and_release_state_guards() -> (
    None
):
    completed_harness = _harness()
    completed_admission = await _admit(completed_harness)
    await completed_admission.dispatch()
    await completed_admission.complete(_RESULT_DIGEST)
    with pytest.raises(InputValidationError, match="cannot dispatch again"):
        await completed_admission.dispatch()
    assert await completed_admission.release_if_pre_dispatch() is False
    completed_admission._completion_digest = None
    completed_admission._completion_command = None
    with pytest.raises(
        InputValidationError,
        match="another result digest",
    ):
        completed_admission.completion_command("b" * 64)

    ambiguous_harness = _harness()
    ambiguous_admission = await _admit(ambiguous_harness)
    ambiguous_admission._state = DurableContinuationResumeState.AMBIGUOUS
    finished = create_task(asyncio_sleep_result())
    await finished
    ambiguous_admission._dispatch_task = finished
    with pytest.raises(InputValidationError, match="cannot be replayed"):
        await ambiguous_admission.dispatch()
    with pytest.raises(
        InputValidationError,
        match="cannot be completed",
    ):
        await ambiguous_admission.complete(_RESULT_DIGEST)
    with pytest.raises(
        InputValidationError,
        match="lacks a dispatch fence",
    ):
        ambiguous_admission.completion_command(_RESULT_DIGEST)

    admitted_harness = _harness()
    admitted_admission = await _admit(admitted_harness)
    with pytest.raises(
        InputValidationError,
        match="dispatch before completion",
    ):
        await admitted_admission.complete(_RESULT_DIGEST)
    with pytest.raises(InputValidationError, match="not ready for completion"):
        admitted_admission.completion_command(_RESULT_DIGEST)
    with pytest.raises(InputValidationError) as raised:
        admitted_admission.completion_command("not-a-digest")
    assert raised.value.path == "resume.result_digest"

    owner_harness = _harness()
    owner_admission = await _admit(owner_harness)
    owner_admission._state = DurableContinuationResumeState.DISPATCHED
    owner_admission._continuation = replace(
        owner_admission.continuation,
        claim=replace(
            owner_admission.continuation.claim,
            owner_id=ContinuationClaimOwnerId("other-owner"),
        ),
    )
    with pytest.raises(InputValidationError, match="owner changed"):
        owner_admission.completion_command(_RESULT_DIGEST)

    rejection_harness = _harness()
    rejection_admission = await _admit(rejection_harness)
    rejection_admission._continuation = replace(
        rejection_admission.continuation,
        claim=replace(
            rejection_admission.continuation.claim,
            owner_id=ContinuationClaimOwnerId("other-owner"),
        ),
    )
    with pytest.raises(InputValidationError, match="owner changed"):
        rejection_admission.rejection_command(_RESULT_DIGEST)

    released_harness = _harness()
    released_admission = await _admit(released_harness)
    released = await released_admission.release()
    assert await released_admission.release() is released

    dispatched_harness = _harness()
    dispatched_admission = await _admit(dispatched_harness)
    await dispatched_admission.dispatch()
    with pytest.raises(InputValidationError, match="only before dispatch"):
        await dispatched_admission.release()

    owned_harness = _harness()
    owned_admission = await _admit(owned_harness)
    ownership_task = create_task(asyncio_sleep_result())
    await ownership_task
    owned_admission._dispatch_task = ownership_task
    with pytest.raises(
        InputValidationError,
        match="dispatch ownership has already been claimed",
    ):
        await owned_admission.release()


async def asyncio_sleep_result() -> object:
    await sleep(0)
    return {"ok": True}


@_async_test
async def test_claim_lease_and_resumer_boundaries_reject_invalid_values() -> (
    None
):
    harness = _harness()
    admission = await _admit(harness)
    continuation = admission.continuation
    owner_id = ContinuationClaimOwnerId("worker-claim")

    with pytest.raises(InputValidationError) as raised:
        DurableAgentContinuationClaimLease(
            store=harness.store,
            continuation=cast(Any, object()),
            owner_id=owner_id,
            fencing_token=continuation.fencing_token,
        )
    assert raised.value.path == "resume.claim_lease.continuation"
    with pytest.raises(InputValidationError) as raised:
        DurableAgentContinuationClaimLease(
            store=harness.store,
            continuation=continuation,
            owner_id=ContinuationClaimOwnerId("other-owner"),
            fencing_token=continuation.fencing_token,
        )
    assert raised.value.path == "resume.claim_lease"

    lease = DurableAgentContinuationClaimLease(
        store=harness.store,
        continuation=continuation,
        owner_id=owner_id,
        fencing_token=continuation.fencing_token,
    )

    async def invalid_renewal(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        return 1

    harness.store.renew_claim = cast(Any, invalid_renewal)
    with pytest.raises(InputValidationError) as raised:
        await lease.renew(
            _LEASE_EXPIRES_AT,
            now=_CLAIMED_AT,
        )
    assert raised.value.path == "resume.claim_lease.renewal"

    fresh = _harness()
    with pytest.raises(InputValidationError) as raised:
        DurableAgentContinuationResumer(
            fresh.store,
            cast(Any, object()),
        )
    assert raised.value.path == "resume.resolver"
    with pytest.raises(InputValidationError) as raised:
        DurableAgentContinuationResumer(
            fresh.store,
            fresh.resumer._resolver,
            clock=cast(Any, object()),
        )
    assert raised.value.path == "resume.clock"

    common = {
        "record": fresh.record,
        "actor": InteractionActor(principal=fresh.request.origin.principal),
        "expected_request_id": fresh.request.request_id,
        "expected_run_id": fresh.request.origin.run_id,
        "expected_checkpoint_id": "checkpoint",
        "owner_id": ContinuationClaimOwnerId("worker-claim"),
        "lease_expires_at": _LEASE_EXPIRES_AT,
        "dispatch_id": _DISPATCH_ID,
    }
    admission_cases = (
        ("resume.record", {"record": object()}),
        ("resume.actor", {"actor": object()}),
        (
            "resume.lease_expires_at_provider",
            {"lease_expires_at_provider": object()},
        ),
        (
            "resume.claim_lease_observer",
            {"claim_lease_observer": object()},
        ),
    )
    for expected_path, changes in admission_cases:
        arguments = {**common, **changes}
        with pytest.raises(InputValidationError) as raised:
            await fresh.resumer.admit(**cast(Any, arguments))
        assert raised.value.path == expected_path

    snapshotless = _harness()
    snapshotless_record = replace(
        snapshotless.record,
        continuation=replace(
            snapshotless.record.continuation,
            provider_snapshot=None,
        ),
    )
    with pytest.raises(InputValidationError) as raised:
        await snapshotless.resumer.admit(
            snapshotless_record,
            **cast(
                Any,
                {
                    key: value
                    for key, value in common.items()
                    if key != "record"
                },
            ),
        )
    assert raised.value.path == "resume.continuation.provider_snapshot"

    expired = _harness()
    expired_arguments = {
        **common,
        "record": expired.record,
        "lease_expires_at": _CLAIMED_AT,
    }
    with pytest.raises(InputValidationError) as raised:
        await expired.resumer.admit(**cast(Any, expired_arguments))
    assert raised.value.path == "resume.lease_expires_at"


@_async_test
async def test_terminal_projection_and_identity_checks_reject_corruption() -> (
    None
):
    unavailable = _harness()
    unavailable.store.interaction_record = cast(Any, object())
    with pytest.raises(InputValidationError) as raised:
        await unavailable.resumer._terminal_request(
            unavailable.record.continuation,
            actor=InteractionActor(
                principal=unavailable.request.origin.principal
            ),
        )
    assert raised.value.path == "resume.interaction"

    pending = _harness()
    pending_request = replace(
        pending.request,
        state=RequestState.PENDING,
        state_revision=StateRevision(1),
        resolution=None,
    )
    pending_record = replace(pending.store.interaction_record)
    object.__setattr__(pending_record, "request", pending_request)
    pending.store.interaction_record = pending_record
    with pytest.raises(InputValidationError) as raised:
        await pending.resumer._terminal_request(
            pending.record.continuation,
            actor=InteractionActor(principal=pending.request.origin.principal),
        )
    assert raised.value.path == "resume.interaction.state"

    mismatch = _harness()
    mismatch_record = replace(mismatch.store.interaction_record)
    object.__setattr__(
        mismatch_record,
        "request",
        _terminal_request(_origin(_definition(), run_id="other-task-run")),
    )
    mismatch.store.interaction_record = mismatch_record
    with pytest.raises(InputValidationError) as raised:
        await mismatch.resumer._terminal_request(
            mismatch.record.continuation,
            actor=InteractionActor(
                principal=mismatch.request.origin.principal
            ),
        )
    assert raised.value.path == "resume.interaction"

    projection = _harness()
    with patch.object(
        continuation_module,
        "project_resolution_to_model",
        return_value=object(),
    ):
        with pytest.raises(InputValidationError) as raised:
            await projection.resumer._terminal_request(
                projection.record.continuation,
                actor=InteractionActor(
                    principal=projection.request.origin.principal
                ),
            )
    assert raised.value.path == "resume.interaction.resolution"

    identity = _harness()
    actor = InteractionActor(principal=identity.request.origin.principal)
    with pytest.raises(InputValidationError) as raised:
        continuation_module._validate_expected_identity(
            identity.record,
            actor=actor,
            expected_request_id=InputRequestId("other-request"),
            expected_run_id=identity.request.origin.run_id,
            expected_checkpoint_id="checkpoint",
        )
    assert raised.value.path == "resume.expected_request_id"

    corrupt_run_record = replace(identity.record)
    object.__setattr__(
        corrupt_run_record,
        "task_run_id",
        "other-task-run",
    )
    with pytest.raises(InputValidationError) as raised:
        continuation_module._validate_expected_identity(
            corrupt_run_record,
            actor=actor,
            expected_request_id=identity.request.request_id,
            expected_run_id=RunId("other-task-run"),
            expected_checkpoint_id="checkpoint",
        )
    assert raised.value.path == "resume.expected_run_id"

    with pytest.raises(InputValidationError) as raised:
        continuation_module._validate_expected_identity(
            identity.record,
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("other-owner"))
            ),
            expected_request_id=identity.request.request_id,
            expected_run_id=identity.request.origin.run_id,
            expected_checkpoint_id="checkpoint",
        )
    assert raised.value.path == "resume.actor.principal"


@_async_test
async def test_completion_state_races_and_internal_validation_fail_closed() -> (  # noqa: E501
    None
):
    digest_race = _harness()
    digest_admission = await _admit(digest_race)
    digest_admission._completion_digest = "b" * 64
    with pytest.raises(InputValidationError) as raised:
        await digest_admission.complete("c" * 64)
    assert raised.value.path == "resume.result_digest"

    wrong_state = _harness()
    wrong_state_admission = await _admit(wrong_state)
    dispatch_task = create_task(asyncio_sleep_result())
    await dispatch_task
    wrong_state_admission._dispatch_task = dispatch_task
    wrong_state_admission._state = DurableContinuationResumeState.RELEASED
    with pytest.raises(InputValidationError, match="not ready for completion"):
        await wrong_state_admission.complete(_RESULT_DIGEST)

    previous = _harness().record.continuation
    claim_arguments = {
        "owner_id": ContinuationClaimOwnerId("owner"),
        "lease_expires_at": _LEASE_EXPIRES_AT,
        "dispatch_id": _DISPATCH_ID,
        "provider_idempotency_key": derive_provider_idempotency_key(
            previous.continuation_id,
            _DISPATCH_ID,
        ),
        "claimed_at": _CLAIMED_AT,
    }
    with pytest.raises(InputValidationError) as raised:
        continuation_module._validate_claim_receipt(
            previous,
            cast(Any, object()),
            **claim_arguments,
        )
    assert raised.value.path == "resume.claim_receipt"
    with pytest.raises(InputValidationError) as raised:
        continuation_module._validate_portable_payload_unchanged(
            previous,
            cast(Any, object()),
        )
    assert raised.value.path == "resume.continuation"


def test_runtime_catalog_snapshot_and_reserved_call_checks_fail_closed() -> (
    None
):
    harness = _harness()
    continuation = harness.record.continuation
    runtime = harness.loader.runtime

    with pytest.raises(InputValidationError) as raised:
        continuation_module._validated_catalog(
            replace(runtime, capabilities=object()),
            continuation,
        )
    assert raised.value.path == "resume.runtime.capabilities"

    other_binding = replace(
        continuation.revision_binding,
        model_config_revision=ModelConfigRevision("model-r2"),
    )
    with pytest.raises(InputValidationError) as raised:
        continuation_module._validated_catalog(
            replace(runtime, capabilities=_catalog(other_binding)),
            continuation,
        )
    assert raised.value.path == "resume.runtime.capabilities"

    codec_missing_catalog = _catalog(continuation.revision_binding)
    object.__setattr__(
        codec_missing_catalog.support,
        "continuation_snapshot_codec",
        None,
    )
    with (
        patch.object(
            ProviderCapabilitySupport,
            "task_input_advertisement_for",
            return_value=TaskInputCapabilityAdvertisement.DURABLE,
        ),
        pytest.raises(InputValidationError) as raised,
    ):
        continuation_module._validated_catalog(
            replace(runtime, capabilities=codec_missing_catalog),
            continuation,
        )
    assert raised.value.path == "resume.runtime.capabilities"

    roundtrip_catalog = _catalog(continuation.revision_binding)
    registry = roundtrip_catalog.support.continuation_snapshot_codec_registry
    assert registry is not None
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    drifted_snapshot = replace(
        snapshot,
        payload={
            "reserved_capability_call_id": "call-input",
            "replay_items": (),
        },
    )
    with patch.object(
        registry,
        "restore_snapshot",
        return_value=drifted_snapshot,
    ):
        with pytest.raises(InputValidationError) as raised:
            continuation_module._validated_catalog(
                replace(runtime, capabilities=roundtrip_catalog),
                continuation,
            )
    assert raised.value.path == "resume.runtime.capabilities"

    catalog = _catalog(continuation.revision_binding)
    call = continuation_module._task_input_call(
        continuation,
        harness.request,
        catalog,
    )
    with pytest.raises(InputValidationError) as raised:
        continuation_module._restore_provider_snapshot(
            replace(runtime, model=object()),
            continuation,
            call,
        )
    assert raised.value.path == "resume.runtime.model"

    with patch.object(
        ModelCapabilityCatalog,
        "decode_call",
        return_value=object(),
    ):
        with pytest.raises(InputValidationError) as raised:
            continuation_module._task_input_call(
                continuation,
                harness.request,
                catalog,
            )
    assert raised.value.path == "resume.task_input_call"


@_async_test
async def test_portable_stager_and_runtime_ownership_validate_boundaries() -> (
    None
):
    with pytest.raises(TypeError, match="clock must be callable"):
        durable_runtime_module.PortableAgentContinuationStager(
            clock=cast(Any, object())
        )

    harness = _harness()
    request = InteractionBrokerRequest(
        actor=InteractionActor(principal=harness.request.origin.principal),
        origin=harness.request.origin,
        mode=harness.request.mode,
        reason=harness.request.reason,
        questions=harness.request.questions,
        continuation_ttl_seconds=(harness.request.continuation_ttl_seconds),
    )
    execution = MagicMock(spec=AgentExecution)
    response = MagicMock(spec=OrchestratorResponse)
    stager = durable_runtime_module.PortableAgentContinuationStager(
        clock=lambda: _NOW
    )
    call_cases = (
        (
            "request must be an interaction broker request",
            {
                "request": object(),
                "execution": execution,
                "response": response,
                "staging": object(),
            },
        ),
        (
            "execution must be an agent execution",
            {
                "request": request,
                "execution": object(),
                "response": response,
                "staging": object(),
            },
        ),
        (
            "response must be an orchestrator response",
            {
                "request": request,
                "execution": execution,
                "response": object(),
                "staging": object(),
            },
        ),
        (
            "staging must be a durable staging context",
            {
                "request": request,
                "execution": execution,
                "response": response,
                "staging": object(),
            },
        ),
    )
    for message, values in call_cases:
        with pytest.raises(TypeError, match=message):
            await stager(
                cast(Any, values["request"]),
                execution=cast(Any, values["execution"]),
                response=values["response"],
                stream_sequence=0,
                staging=cast(Any, values["staging"]),
            )

    corrupt_staging = object.__new__(
        durable_runtime_module.DurableInteractionStagingContext
    )
    object.__setattr__(
        corrupt_staging,
        "continuation_id",
        harness.request.continuation_id,
    )
    object.__setattr__(
        corrupt_staging,
        "task_input_call",
        cast(Any, object()),
    )
    execution.snapshot = MagicMock(
        ledger=(),
        active_interaction_fingerprint=None,
    )
    with pytest.raises(
        ExecutionCorrelationError,
        match="changed its active interaction",
    ):
        await stager(
            request,
            execution=execution,
            response=response,
            stream_sequence=0,
            staging=corrupt_staging,
        )

    with pytest.raises(TypeError, match="event_manager"):
        durable_runtime_module._TrustedContinuationEventListenerRegistration(
            cast(Any, object()),
            lambda _event: None,
        )
    manager = EventManager()
    with pytest.raises(TypeError, match="listener"):
        durable_runtime_module._TrustedContinuationEventListenerRegistration(
            manager,
            cast(Any, object()),
        )

    with pytest.raises(TypeError, match="async exit stack"):
        durable_runtime_module._TrustedContinuationRuntimeOwnership(
            cast(Any, object())
        )
    ownership = durable_runtime_module._TrustedContinuationRuntimeOwnership(
        AsyncExitStack()
    )
    await ownership.close()
    with pytest.raises(RuntimeError, match="runtime is closing"):
        ownership.register_event_listener(
            manager,
            lambda _event: None,
        )
    await manager.aclose()

    failing_stack = AsyncExitStack()

    async def fail_stack_close() -> None:
        raise RuntimeError("stack close failed")

    class FailingRegistration:
        def close(self) -> None:
            raise RuntimeError("listener close failed")

    failing_stack.push_async_callback(fail_stack_close)
    failing_ownership = (
        durable_runtime_module._TrustedContinuationRuntimeOwnership(
            failing_stack
        )
    )
    failing_ownership._event_listener_registrations.append(
        cast(Any, FailingRegistration())
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        await failing_ownership.close()
    assert [str(error) for error in raised.value.exceptions] == [
        "listener close failed",
        "stack close failed",
    ]


def _durable_runtime_resume_command(
    command: AgentContinuationResumeCommand,
    executor: durable_runtime_module.TrustedAgentContinuationExecutor,
    *,
    active_fingerprint: str,
    counts: tuple[tuple[str, int], ...],
    interaction_count: int,
) -> AgentContinuationResumeCommand:
    transcript = (
        durable_runtime_module._encode_message_record(
            Message(role=MessageRole.USER, content="hello")
        ),
    )
    assistant = durable_runtime_module._encode_message_record(
        Message(role=MessageRole.ASSISTANT, content="input required")
    )
    observation = cast(
        Mapping[str, JsonValue],
        {
            "version": 1,
            "kind": "agent_execution",
            "active_interaction_fingerprint": active_fingerprint,
            "interaction_fingerprint_counts": [
                {"fingerprint": fingerprint, "count": count}
                for fingerprint, count in counts
            ],
            "assistant_message": assistant,
        },
    )
    continuation = replace(
        command.continuation,
        transcript=transcript,
        observations=(observation,),
        interaction_count=interaction_count,
    )
    runtime = replace(command.resolved_runtime, runtime=executor)
    return replace(
        command,
        continuation=continuation,
        resolved_runtime=runtime,
    )


@_async_test
async def test_trusted_executor_rejects_runtime_and_replay_corruption() -> (
    None
):
    ownership = durable_runtime_module._TrustedContinuationRuntimeOwnership(
        AsyncExitStack()
    )
    orchestrator = MagicMock(spec=Orchestrator)
    stager = durable_runtime_module.PortableAgentContinuationStager(
        clock=lambda: _NOW
    )
    constructor_cases = (
        (
            "orchestrator must be a concrete orchestrator",
            {
                "orchestrator": object(),
                "stager": stager,
                "ownership": ownership,
            },
        ),
        (
            "stager must be a portable continuation stager",
            {
                "orchestrator": orchestrator,
                "stager": object(),
                "ownership": ownership,
            },
        ),
        (
            "ownership must be a shared runtime owner",
            {
                "orchestrator": orchestrator,
                "stager": stager,
                "ownership": object(),
            },
        ),
    )
    for message, values in constructor_cases:
        with pytest.raises(TypeError, match=message):
            durable_runtime_module.TrustedAgentContinuationExecutor(
                cast(Any, values["orchestrator"]),
                stager=cast(Any, values["stager"]),
                ownership=cast(Any, values["ownership"]),
            )

    executor = durable_runtime_module.TrustedAgentContinuationExecutor(
        orchestrator,
        stager=stager,
        ownership=ownership,
    )
    with pytest.raises(TypeError, match="agent continuation command"):
        await executor.resume_agent_continuation(cast(Any, object()))

    harness = _harness()
    admission = await _admit(harness)
    with pytest.raises(
        ExecutionCorrelationError,
        match="different continuation runtime",
    ):
        await executor.resume_agent_continuation(admission.command)

    count_mismatch = _durable_runtime_resume_command(
        admission.command,
        executor,
        active_fingerprint="active",
        counts=(("active", 1),),
        interaction_count=2,
    )
    with pytest.raises(
        InputValidationError,
        match="counts do not match",
    ):
        await executor.resume_agent_continuation(count_mismatch)

    missing_active = _durable_runtime_resume_command(
        admission.command,
        executor,
        active_fingerprint="active",
        counts=(("other", 1),),
        interaction_count=1,
    )
    with pytest.raises(
        InputValidationError,
        match="active interaction is absent",
    ):
        await executor.resume_agent_continuation(missing_active)

    duplicate_result = _durable_runtime_resume_command(
        admission.command,
        executor,
        active_fingerprint="active",
        counts=(("active", 2),),
        interaction_count=2,
    )
    reconstructed = MagicMock()
    reconstructed.begin_interaction = AsyncMock()
    reconstructed.abandon_interaction = AsyncMock()
    reconstructed.stage_durable_input_required = AsyncMock()
    reconstructed.record_interaction_result = AsyncMock(return_value=False)
    with patch.object(
        durable_runtime_module,
        "AgentExecution",
        return_value=reconstructed,
    ):
        with pytest.raises(
            ExecutionCorrelationError,
            match="already committed",
        ):
            await executor.resume_agent_continuation(duplicate_result)
    assert reconstructed.begin_interaction.await_count == 2
    reconstructed.abandon_interaction.assert_awaited_once()
    reconstructed.record_interaction_result.assert_awaited_once()
    await ownership.close()


@_async_test
async def test_trusted_runtime_loader_validates_configuration_and_loads() -> (
    None
):
    with TemporaryDirectory() as root_value, TemporaryDirectory() as outside:
        root = Path(root_value)
        trusted_file = root / "agent.toml"
        trusted_file.write_text(
            "[agent]\nrole='assistant'\n",
            encoding="utf-8",
        )
        nested = root / "nested"
        nested.mkdir()
        outside_file = Path(outside) / "agent.toml"
        outside_file.write_text(
            "[agent]\nrole='assistant'\n",
            encoding="utf-8",
        )
        loader = MagicMock(spec=OrchestratorLoader)
        stack = AsyncExitStack()
        valid = {
            "loader": loader,
            "stack": stack,
            "allowed_roots": (root,),
        }
        constructor_cases = (
            ("loader must be an orchestrator loader", {"loader": object()}),
            ("stack must be an async exit stack", {"stack": object()}),
            (
                "allowed_roots must contain existing directories",
                {"allowed_roots": ()},
            ),
            (
                "allowed_roots must contain existing directories",
                {"allowed_roots": (trusted_file,)},
            ),
            (
                "tool_settings must be trusted tool settings",
                {"tool_settings": object()},
            ),
            (
                "disable_memory must be a boolean",
                {"disable_memory": 1},
            ),
            ("uri must be a string or None", {"uri": 1}),
        )
        for message, changes in constructor_cases:
            arguments = {**valid, **changes}
            with pytest.raises((TypeError, ValueError), match=message):
                durable_runtime_module.TrustedAgentContinuationRuntimeLoader(
                    **cast(Any, arguments)
                )

        runtime_loader = (
            durable_runtime_module.TrustedAgentContinuationRuntimeLoader(
                **valid
            )
        )
        with pytest.raises(InputValidationError):
            runtime_loader._trusted_definition_path(
                "https://example.test/agent.toml"
            )
        with pytest.raises(InputValidationError, match="not canonical"):
            runtime_loader._trusted_definition_path(
                f"{root.as_uri()}/nested/../agent.toml"
            )
        with pytest.raises(InputValidationError, match="outside approved"):
            runtime_loader._trusted_definition_path(
                outside_file.resolve(strict=True).as_uri()
            )

        with pytest.raises(TypeError, match="execution definition"):
            await runtime_loader.load_continuation_runtime(
                cast(Any, object()),
                _binding(),
            )
        definition = replace(
            _definition(),
            agent_definition_locator=trusted_file.resolve(
                strict=True
            ).as_uri(),
        )
        with pytest.raises(TypeError, match="revision_binding"):
            await runtime_loader.load_continuation_runtime(
                definition,
                cast(Any, object()),
            )

        admission_loader = MagicMock(spec=OrchestratorLoader)
        admission_loader.from_file = AsyncMock(return_value=object())
        loader.clone_for_stack.return_value = admission_loader
        with pytest.raises(InputValidationError) as raised:
            await runtime_loader.load_continuation_runtime(
                definition,
                _binding(),
            )
        assert raised.value.path == "continuation_runtime.orchestrator"

        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.__aenter__ = AsyncMock(return_value=orchestrator)
        orchestrator.__aexit__ = AsyncMock(return_value=False)
        orchestrator.continuation_execution_contract.return_value = (
            definition,
            _binding(),
            _catalog(_binding()),
        )
        engine_agent = MagicMock()
        engine_agent.engine.model = None
        orchestrator.engine_agent_for_operation.return_value = engine_agent
        admission_loader.from_file = AsyncMock(return_value=orchestrator)
        with pytest.raises(InputValidationError) as raised:
            await runtime_loader.load_continuation_runtime(
                definition,
                _binding(),
            )
        assert raised.value.path == "continuation_runtime.model"
        await stack.aclose()


def test_durable_runtime_decoders_reject_every_malformed_shape() -> None:
    encoded_user = durable_runtime_module._encode_message_record(
        Message(role=MessageRole.USER, content="hello")
    )
    encoded_assistant = durable_runtime_module._encode_message_record(
        Message(role=MessageRole.ASSISTANT, content="input required")
    )
    transcript_cases = (
        ({"version": 1}, "record fields"),
        (
            {**encoded_user, "version": 2},
            "version is unsupported",
        ),
        (
            {**encoded_user, "role": 1},
            "role and data must be strings",
        ),
    )
    for record, message in transcript_cases:
        with pytest.raises(InputValidationError, match=message):
            durable_runtime_module._decode_transcript(
                (cast(Mapping[str, JsonValue], record),)
            )
    with pytest.raises(InputValidationError, match="must not be empty"):
        durable_runtime_module._decode_transcript(())

    base = {
        "version": 1,
        "kind": "agent_execution",
        "active_interaction_fingerprint": "active",
        "interaction_fingerprint_counts": (
            {"fingerprint": "active", "count": 1},
        ),
        "assistant_message": encoded_assistant,
    }
    observation_cases = (
        ((), "exactly one"),
        (({"version": 1},), "fields do not match"),
        (({**base, "version": 2},), "version is unsupported"),
        (
            ({**base, "active_interaction_fingerprint": 1},),
            "values are invalid",
        ),
        (
            (
                {
                    **base,
                    "interaction_fingerprint_counts": ({"count": 1},),
                },
            ),
            "count fields do not match",
        ),
        (
            (
                {
                    **base,
                    "interaction_fingerprint_counts": (
                        {"fingerprint": "active", "count": 0},
                    ),
                },
            ),
            "interaction count is invalid",
        ),
        (
            ({**base, "assistant_message": "invalid"},),
            "assistant message is invalid",
        ),
        (
            ({**base, "assistant_message": encoded_user},),
            "must be an assistant",
        ),
    )
    for observations, message in observation_cases:
        with pytest.raises(InputValidationError, match=message):
            durable_runtime_module._decode_execution_observation(
                cast(Any, observations)
            )


def test_portable_json_conversion_accepts_declared_values_only() -> None:
    class ExampleEnum(Enum):
        VALUE = "value"

    @dataclass
    class ExampleData:
        value: int

    assert durable_runtime_module._json_default(ExampleEnum.VALUE) == "value"
    assert durable_runtime_module._json_default(ExampleData(1)) == {"value": 1}
    with pytest.raises(TypeError, match="not portable JSON"):
        durable_runtime_module._json_default(object())
    with pytest.raises(InputValidationError) as raised:
        durable_runtime_module._portable_json_mapping(
            {"value": object()},
            "settings",
        )
    assert raised.value.code is InputErrorCode.NON_JSON_VALUE
    with pytest.raises(InputValidationError, match="JSON object"):
        durable_runtime_module._portable_json_mapping(
            cast(Any, [1, 2]),
            "settings",
        )


def _task_segment(
    harness: _Harness,
    *,
    state: TaskAttemptSegmentState = TaskAttemptSegmentState.SUSPENDED,
) -> TaskAttemptSegment:
    continuation = harness.record.continuation
    return TaskAttemptSegment(
        segment_id="task-segment",
        attempt_id="task-attempt",
        run_id="task-run",
        segment_number=1,
        state=state,
        created_at=_NOW,
        updated_at=_NOW,
        request_id=str(continuation.request_id),
        continuation_id=str(continuation.continuation_id),
        checkpoint_id=harness.record.checkpoint_id,
    )


def _task_claim(
    *,
    attempt_state: TaskAttemptState,
) -> TaskQueueClaim:
    task_claim = TaskClaim(
        worker_id="worker",
        claim_token="claim-token",
        claimed_at=_CLAIMED_AT,
        lease_expires_at=_LEASE_EXPIRES_AT,
        heartbeat_at=_CLAIMED_AT,
    )
    run = TaskRun(
        run_id="task-run",
        definition_id="definition",
        state=TaskRunState.CLAIMED,
        request=TaskExecutionRequest(definition_id="definition"),
        created_at=_NOW,
        updated_at=_CLAIMED_AT,
        claim=task_claim,
        last_attempt_id="task-attempt",
    )
    attempt = TaskAttempt(
        attempt_id="task-attempt",
        run_id=run.run_id,
        attempt_number=1,
        state=attempt_state,
        context=TaskExecutionContext(
            run_id=run.run_id,
            attempt_id="task-attempt",
            attempt_number=1,
            claim=task_claim,
        ),
        created_at=_NOW,
        updated_at=_CLAIMED_AT,
    )
    queue_item = TaskQueueItem(
        queue_item_id="queue-item",
        run_id=run.run_id,
        queue_name="default",
        state=TaskQueueItemState.CLAIMED,
        priority=0,
        available_at=_NOW,
        attempts=1,
        created_at=_NOW,
        updated_at=_CLAIMED_AT,
        run_state=run.state,
        claimed_at=task_claim.claimed_at,
        lease_expires_at=task_claim.lease_expires_at,
        worker_id=task_claim.worker_id,
        claim_token=task_claim.claim_token,
        heartbeat_at=task_claim.heartbeat_at,
    )
    return TaskQueueClaim(
        queue_item=queue_item,
        run=run,
        attempt=attempt,
    )


class _TaskContinuationRecordStore:
    def __init__(self, record: object) -> None:
        self.record = record

    async def get_task_continuation_record(
        self,
        task_run_id: str,
    ) -> object:
        assert task_run_id == "task-run"
        return self.record


class _InvalidTaskResumeActorResolver:
    trusted_task_resume_actor_resolver = True

    async def resolve_task_resume_actor(
        self,
        claim: TaskQueueClaim,
        previous_segment: TaskAttemptSegment,
        record: DurableContinuationRecord,
    ) -> object:
        del claim, previous_segment, record
        return object()


@_async_test
async def test_task_resume_admission_validation_and_proxy_contract() -> None:
    harness = _harness()
    segment = _task_segment(harness)
    agent_admission = await _admit(harness)
    admission = TaskDurableResumeAdmission(
        record=harness.record,
        previous_segment=segment,
        agent_admission=agent_admission,
    )

    validation_cases = (
        {"record": object()},
        {"previous_segment": object()},
        {
            "previous_segment": replace(
                segment,
                state=TaskAttemptSegmentState.RUNNING,
            )
        },
        {"agent_admission": object()},
        {
            "record": replace(
                harness.record,
                checkpoint_id="other-checkpoint",
            )
        },
    )
    for changes in validation_cases:
        with pytest.raises(InputValidationError):
            replace(admission, **cast(Any, changes))

    def listener(event: object) -> None:
        _ = event

    registration = admission.register_event_listener(listener)
    registration.close()
    assert await admission.dispatch() == {"answer": "resumed"}
    assert (
        await admission.wait_dispatch_settled()
        is DurableContinuationResumeState.DISPATCHED
    )
    command = admission.completion_command_for_output({"answer": "resumed"})
    assert command.result_digest == task_resume_result_digest(
        {"answer": "resumed"}
    )
    failure = TaskDurableResumeFailure(
        result=TaskExecutionResult(error={"code": "task.failed"})
    )

    await admission.complete_output({"answer": "resumed"})
    assert (
        admission.completed_completion_command().result_digest
        == task_resume_result_digest({"answer": "resumed"})
    )
    await admission.close()

    suspension_harness = _harness()
    suspension_admission = TaskDurableResumeAdmission(
        record=suspension_harness.record,
        previous_segment=_task_segment(suspension_harness),
        agent_admission=await _admit(suspension_harness),
    )
    await suspension_admission.dispatch()
    suspended_command = suspension_admission.completion_command_for_suspension(
        request_id="successor-request",
        continuation_id="successor-continuation",
        checkpoint_id="successor-checkpoint",
    )
    assert suspended_command.result_digest == task_resume_result_digest(
        {
            "kind": "suspended",
            "request_id": "successor-request",
            "continuation_id": "successor-continuation",
            "checkpoint_id": "successor-checkpoint",
        }
    )
    await suspension_admission.close()

    settlement_harness = _harness()
    settlement_admission = TaskDurableResumeAdmission(
        record=settlement_harness.record,
        previous_segment=_task_segment(settlement_harness),
        agent_admission=await _admit(settlement_harness),
    )
    await settlement_admission.dispatch()
    settlement_command = (
        settlement_admission.completion_command_for_settlement(failure)
    )
    assert settlement_command.result_digest
    with pytest.raises(InputValidationError):
        settlement_admission.rejection_command_for_settlement(
            cast(Any, object())
        )
    await settlement_admission.close()

    incomplete_harness = _harness()
    incomplete = TaskDurableResumeAdmission(
        record=incomplete_harness.record,
        previous_segment=_task_segment(incomplete_harness),
        agent_admission=await _admit(incomplete_harness),
    )
    with pytest.raises(InputValidationError):
        incomplete.completed_completion_command()
    rejection = incomplete.rejection_command_for_settlement(failure)
    assert rejection.result_digest
    await incomplete.release()
    await incomplete.close()

    interrupt_harness = _harness()
    interrupt = TaskDurableResumeAdmission(
        record=interrupt_harness.record,
        previous_segment=_task_segment(interrupt_harness),
        agent_admission=await _admit(interrupt_harness),
    )
    assert (
        await interrupt.interrupt_dispatch()
        is DurableContinuationResumeState.RELEASED
    )
    await interrupt.close()

    release_harness = _harness()
    releasable = TaskDurableResumeAdmission(
        record=release_harness.record,
        previous_segment=_task_segment(release_harness),
        agent_admission=await _admit(release_harness),
    )
    assert await releasable.release_if_pre_dispatch()
    await releasable.close()


@_async_test
async def test_task_resume_coordinator_rejects_invalid_boundaries() -> None:
    harness = _harness()
    record_store = _TaskContinuationRecordStore(harness.record)

    with pytest.raises(InputValidationError):
        TaskDurableResumeCoordinator(cast(Any, object()), harness.resumer)
    with pytest.raises(InputValidationError):
        TaskDurableResumeCoordinator(record_store, cast(Any, object()))
    with pytest.raises(InputValidationError):
        TaskDurableResumeCoordinator(
            record_store,
            harness.resumer,
            actor_resolver=cast(Any, object()),
        )

    coordinator = TaskDurableResumeCoordinator(
        record_store,
        harness.resumer,
    )
    with pytest.raises(InputValidationError):
        await coordinator.admit(cast(Any, object()), None)

    fresh_claim = _task_claim(attempt_state=TaskAttemptState.CREATED)
    assert await coordinator.admit(fresh_claim, None) is None
    with pytest.raises(InputValidationError):
        await coordinator.admit(fresh_claim, _task_segment(harness))

    suspended_claim = _task_claim(attempt_state=TaskAttemptState.SUSPENDED)
    with pytest.raises(InputValidationError):
        await coordinator.admit(suspended_claim, None)
    invalid_state_claim = _task_claim(attempt_state=TaskAttemptState.SUSPENDED)
    object.__setattr__(
        invalid_state_claim.attempt,
        "state",
        TaskAttemptState.RUNNING,
    )
    with pytest.raises(InputValidationError):
        await coordinator.admit(invalid_state_claim, None)
    mismatched = replace(_task_segment(harness), run_id="other-run")
    with pytest.raises(InputValidationError):
        _validate_previous_segment(suspended_claim, mismatched)
    with pytest.raises(InputValidationError):
        _validate_record(
            suspended_claim,
            _task_segment(harness),
            cast(Any, object()),
        )
    mismatched_record = replace(harness.record)
    object.__setattr__(mismatched_record, "task_run_id", "other-run")
    with pytest.raises(InputValidationError):
        _validate_record(
            suspended_claim,
            _task_segment(harness),
            mismatched_record,
        )

    invalid_actor_coordinator = TaskDurableResumeCoordinator(
        record_store,
        harness.resumer,
        actor_resolver=cast(Any, _InvalidTaskResumeActorResolver()),
    )
    with pytest.raises(InputValidationError):
        await invalid_actor_coordinator.admit(
            suspended_claim,
            _task_segment(harness),
        )

    assert task_resume_result_digest({"nested": ({"value": "ready"},)})
    actor = await StoredTaskResumeActorResolver().resolve_task_resume_actor(
        suspended_claim,
        _task_segment(harness),
        harness.record,
    )
    assert actor.principal == harness.record.continuation.origin.principal
