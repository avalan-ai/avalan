"""Claim, reconstruct, and fence one durable agent continuation."""

from ..event import Event
from ..interaction.codec import (
    encode_input_model_result,
    encode_input_question,
)
from ..interaction.continuation import (
    ContinuationClaimOwnerId,
    ContinuationClaimReceipt,
    ContinuationClaimState,
    ContinuationCompletion,
    ContinuationCompletionCommand,
    ContinuationDispatch,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationRejectionCommand,
    ContinuationRuntimeResolver,
    ContinuationStoreRevision,
    DurableContinuationRecord,
    DurableContinuationResumeState,
    PortableContinuation,
    ResolvedContinuationRuntime,
    derive_provider_idempotency_key,
)
from ..interaction.entities import (
    RESERVED_INPUT_CAPABILITY_NAME,
    ContinuationId,
    InputModelResult,
    InputRequest,
    InputRequestId,
    ProviderIdempotencyKey,
    ResumeInputContinuation,
    RunId,
)
from ..interaction.error import InputErrorCode, InputValidationError
from ..interaction.policy import InteractionActor
from ..interaction.state import project_resolution_to_model
from ..interaction.store import (
    InteractionCorrelation,
    InteractionDisclosureProjection,
    InteractionRecord,
    ScopedInteractionLookup,
)
from ..interaction.validation import (
    validate_aware_datetime,
    validate_opaque_id,
)
from ..model.capability import (
    CorrelatedCapabilityResult,
    ModelCapabilityCatalog,
    ProviderCapabilityCall,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from ..types import JsonValue

from asyncio import CancelledError, Lock, Task, create_task, shield
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import iscoroutinefunction
from re import fullmatch
from typing import NoReturn, Protocol, cast, final

_SHA256_PATTERN = r"[0-9a-f]{64}"
AgentContinuationEventListener = Callable[
    [Event],
    Awaitable[None] | None,
]


class AgentContinuationEventListenerRegistration(Protocol):
    """Own one reconstructed-runtime event listener registration."""

    def close(self) -> None:
        """Remove the reconstructed-runtime listener exactly once."""
        ...


class AgentContinuationExecutor(Protocol):
    """Resume one reconstructed agent without repeating its initial input."""

    trusted_agent_continuation_executor: bool

    def register_event_listener(
        self,
        listener: AgentContinuationEventListener,
    ) -> AgentContinuationEventListenerRegistration:
        """Register one listener before resumed provider dispatch."""
        ...

    async def resume_agent_continuation(
        self,
        command: "AgentContinuationResumeCommand",
    ) -> object:
        """Append one correlated input result and resume provider execution."""
        ...

    async def close_continuation_runtime(self) -> None:
        """Close resources owned by this reconstructed admission."""
        ...


class AgentDurableContinuationStore(Protocol):
    """Expose the durable operations needed by an agent resumer."""

    async def lookup_scoped(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized terminal interaction."""
        ...

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
        """Claim one ready continuation under a fresh fence."""
        ...

    async def mark_dispatching(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Fence replay before the first provider side effect."""
        ...

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
        """Renew one exact pre-dispatch continuation claim."""
        ...

    async def mark_dispatched(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Record that one fenced provider call returned."""
        ...

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
        """Complete one exact fenced continuation."""
        ...

    async def get_continuation(
        self,
        continuation_id: ContinuationId,
    ) -> PortableContinuation:
        """Load the latest durable continuation state."""
        ...

    async def release(
        self,
        continuation_id: ContinuationId,
        *,
        expected_store_revision: ContinuationStoreRevision,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        now: datetime,
    ) -> PortableContinuation:
        """Release only one provably pre-dispatch claim."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentContinuationResumeCommand:
    """Carry one exact reconstructed input result into a fresh runtime."""

    continuation: PortableContinuation
    request: InputRequest
    model_result: InputModelResult
    task_input_call: TaskInputCapabilityCall
    correlated_result: CorrelatedCapabilityResult
    resolved_runtime: ResolvedContinuationRuntime

    def __post_init__(self) -> None:
        continuation = self.continuation
        if type(continuation) is not PortableContinuation:
            _invalid_type(
                "resume.command.continuation",
                "a portable continuation",
            )
        if (
            continuation.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        ):
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "resume.command.continuation.claim.state",
                "resume command requires a pre-dispatch claim",
            )
        if type(self.request) is not InputRequest:
            _invalid_type("resume.command.request", "an input request")
        if (
            self.request.request_id != continuation.request_id
            or self.request.continuation_id != continuation.continuation_id
            or self.request.origin != continuation.origin
        ):
            _correlation_error(
                "resume.command.request",
                "terminal request does not match the continuation",
            )
        outcome = project_resolution_to_model(
            self.request,
            containing_run_exists=True,
        )
        if (
            type(outcome) is not ResumeInputContinuation
            or outcome.result != self.model_result
        ):
            _correlation_error(
                "resume.command.model_result",
                "model result does not match the terminal request",
            )
        if type(self.task_input_call) is not TaskInputCapabilityCall:
            _invalid_type(
                "resume.command.task_input_call",
                "a task-input capability call",
            )
        call = self.task_input_call
        if (
            str(call.call_id) != continuation.provider_call_correlation_id
            or call.mode != self.request.mode
            or call.reason != self.request.reason
            or call.questions != self.request.questions
            or call.advertisement
            is not TaskInputCapabilityAdvertisement.DURABLE
        ):
            _correlation_error(
                "resume.command.task_input_call",
                "reserved task-input call does not match the continuation",
            )
        if type(self.correlated_result) is not CorrelatedCapabilityResult:
            _invalid_type(
                "resume.command.correlated_result",
                "a correlated capability result",
            )
        correlated = self.correlated_result
        expected_payload = CorrelatedCapabilityResult(
            call_id=call.call_id,
            canonical_name=call.canonical_name,
            provider_name=call.provider_name,
            payload=cast(
                Mapping[str, JsonValue],
                encode_input_model_result(self.model_result),
            ),
        ).payload
        if (
            str(correlated.call_id) != str(call.call_id)
            or correlated.canonical_name != call.canonical_name
            or correlated.provider_name != call.provider_name
            or correlated.payload != expected_payload
        ):
            _correlation_error(
                "resume.command.correlated_result",
                "capability result does not match the reserved call",
            )
        runtime = self.resolved_runtime
        if type(runtime) is not ResolvedContinuationRuntime:
            _invalid_type(
                "resume.command.resolved_runtime",
                "a resolved continuation runtime",
            )
        if (
            runtime.definition != continuation.definition
            or runtime.revision_binding != continuation.revision_binding
        ):
            _correlation_error(
                "resume.command.resolved_runtime",
                "fresh runtime does not match the continuation revision",
            )


@final
class _DurableAgentEventListenerRegistration:
    """Defer removal until owned provider dispatch has settled."""

    def __init__(
        self,
        admission: "DurableAgentContinuationAdmission",
        runtime_registration: AgentContinuationEventListenerRegistration,
    ) -> None:
        self._admission = admission
        self._runtime_registration = runtime_registration
        self._closed = False

    def close(self) -> None:
        """Request listener removal exactly once."""
        if self._closed:
            return
        self._closed = True
        self._admission._close_event_listener_after_dispatch(
            self._runtime_registration
        )


@final
class DurableAgentContinuationAdmission:
    """Own one claimed continuation through fenced provider dispatch."""

    def __init__(
        self,
        *,
        store: AgentDurableContinuationStore,
        command: AgentContinuationResumeCommand,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        executor: AgentContinuationExecutor,
        clock: Callable[[], datetime],
    ) -> None:
        _validate_store(store)
        if type(command) is not AgentContinuationResumeCommand:
            _invalid_type("resume.admission.command", "a resume command")
        if command.continuation.fencing_token != fencing_token:
            _correlation_error(
                "resume.admission.fencing_token",
                "admission fence does not match the claimed continuation",
            )
        _validate_executor(executor)
        if not callable(clock):
            _invalid_type("resume.admission.clock", "a clock callable")
        self._store = store
        self._command = command
        self._owner_id = owner_id
        self._fencing_token = fencing_token
        self._executor = executor
        self._clock = clock
        self._continuation = command.continuation
        self._state = DurableContinuationResumeState.ADMITTED
        self._lock = Lock()
        self._dispatch_task: Task[object] | None = None
        self._release_task: Task[PortableContinuation] | None = None
        self._release_owned = False
        self._complete_task: Task[PortableContinuation] | None = None
        self._completion_digest: str | None = None
        self._completion_command: ContinuationCompletionCommand | None = None
        self._event_listener_registration: (
            _DurableAgentEventListenerRegistration | None
        ) = None

    @property
    def command(self) -> AgentContinuationResumeCommand:
        """Return the immutable correlated resume command."""
        return self._command

    @property
    def continuation(self) -> PortableContinuation:
        """Return the latest durable continuation state observed locally."""
        return self._continuation

    @property
    def state(self) -> DurableContinuationResumeState:
        """Return local dispatch progress without exposing provider content."""
        return self._state

    def register_event_listener(
        self,
        listener: AgentContinuationEventListener,
    ) -> AgentContinuationEventListenerRegistration:
        """Register one listener before provider dispatch takes ownership."""
        if not callable(listener):
            _invalid_type(
                "resume.event_listener",
                "an event listener callable",
            )
        if (
            self._state is not DurableContinuationResumeState.ADMITTED
            or self._dispatch_task is not None
            or self._release_owned
            or self._event_listener_registration is not None
        ):
            _illegal_transition(
                "resume.event_listener",
                "listener registration requires exclusive pre-dispatch state",
            )
        runtime_registration = self._executor.register_event_listener(listener)
        registration = _DurableAgentEventListenerRegistration(
            self,
            runtime_registration,
        )
        self._event_listener_registration = registration
        return registration

    def _close_event_listener_after_dispatch(
        self,
        registration: AgentContinuationEventListenerRegistration,
    ) -> None:
        dispatch_task = self._dispatch_task
        if dispatch_task is None or dispatch_task.done():
            registration.close()
            return

        def close_after_dispatch(completed: Task[object]) -> None:
            del completed
            registration.close()

        dispatch_task.add_done_callback(close_after_dispatch)

    async def dispatch(self) -> object:
        """Invoke the resumed provider path at most once in this process."""
        async with self._lock:
            if self._state is DurableContinuationResumeState.RELEASED:
                _illegal_transition(
                    "resume.admission",
                    "released continuation cannot dispatch",
                )
            if self._state is DurableContinuationResumeState.COMPLETED:
                _illegal_transition(
                    "resume.admission",
                    "completed continuation cannot dispatch again",
                )
            if self._state is DurableContinuationResumeState.AMBIGUOUS:
                _unavailable(
                    "resume.admission",
                    "provider dispatch is ambiguous and cannot be replayed",
                )
            if self._release_owned:
                _illegal_transition(
                    "resume.admission",
                    "release ownership excludes provider dispatch",
                )
            if self._dispatch_task is None:
                self._dispatch_task = create_task(
                    self._dispatch_once(),
                    name=(
                        "durable-agent-resume-"
                        f"{self._continuation.continuation_id}"
                    ),
                )
            task = self._dispatch_task
        return await shield(task)

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        """Wait until an owned dispatch reaches a durable local settlement."""
        async with self._lock:
            task = self._dispatch_task
            if task is None:
                return self._state
        try:
            await shield(task)
        except BaseException:
            if not task.done():
                raise
            try:
                task.result()
            except BaseException:
                pass
        state = self._state
        if state in {
            DurableContinuationResumeState.ADMITTED,
            DurableContinuationResumeState.DISPATCHING,
        }:
            _illegal_transition(
                "resume.admission",
                "dispatch task ended without durable settlement",
            )
        return state

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        """Stop owned work at a durable pre- or post-dispatch boundary."""
        async with self._lock:
            task = self._dispatch_task
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            try:
                await shield(task)
            except BaseException:
                pass
            async with self._lock:
                if (
                    self._dispatch_task is task
                    and task.cancelled()
                    and self._state is DurableContinuationResumeState.ADMITTED
                ):
                    self._dispatch_task = None
        if self._state is DurableContinuationResumeState.ADMITTED:
            released = await self.release_if_pre_dispatch()
            if not released:
                _illegal_transition(
                    "resume.admission",
                    "dispatch changed while interruption was settling",
                )
        state = self._state
        if state in {
            DurableContinuationResumeState.ADMITTED,
            DurableContinuationResumeState.DISPATCHING,
        }:
            _illegal_transition(
                "resume.admission",
                "interrupted dispatch lacks a durable settlement",
            )
        return state

    async def complete(self, result_digest: str) -> PortableContinuation:
        """Complete a dispatched continuation after durable task completion."""
        digest = _validate_digest(result_digest, "resume.result_digest")
        dispatch_task: Task[object] | None
        async with self._lock:
            if (
                self._completion_digest is not None
                and self._completion_digest != digest
            ):
                _correlation_error(
                    "resume.result_digest",
                    "completion replay changed the result digest",
                )
            dispatch_task = self._dispatch_task
            if dispatch_task is None:
                _illegal_transition(
                    "resume.admission",
                    "continuation must dispatch before completion",
                )
            assert dispatch_task is not None
        await shield(dispatch_task)
        async with self._lock:
            if (
                self._completion_digest is not None
                and self._completion_digest != digest
            ):
                _correlation_error(
                    "resume.result_digest",
                    "completion replay changed the result digest",
                )
            if self._state is DurableContinuationResumeState.AMBIGUOUS:
                _unavailable(
                    "resume.admission",
                    "ambiguous provider dispatch cannot be completed",
                )
            if self._state is DurableContinuationResumeState.COMPLETED:
                return self._continuation
            if self._state is not DurableContinuationResumeState.DISPATCHED:
                _illegal_transition(
                    "resume.admission",
                    "continuation is not ready for completion",
                )
            self._completion_digest = digest
            if self._completion_command is None:
                self._completion_command = self._make_completion_command(
                    digest
                )
            if self._complete_task is None:
                self._complete_task = create_task(
                    self._complete_once(digest),
                    name=(
                        "durable-agent-complete-"
                        f"{self._continuation.continuation_id}"
                    ),
                )
            task = self._complete_task
        try:
            return await shield(task)
        except BaseException:
            async with self._lock:
                if self._complete_task is task and task.done():
                    self._complete_task = None
            raise

    def completion_command(
        self,
        result_digest: str,
    ) -> ContinuationCompletionCommand:
        """Return a fenced command for completion or successor suspension."""
        digest = _validate_digest(result_digest, "resume.result_digest")
        if (
            self._completion_digest is not None
            and self._completion_digest != digest
        ):
            _correlation_error(
                "resume.result_digest",
                "completion replay changed the result digest",
            )
        if self._completion_command is not None:
            return self._completion_command
        if self._state is DurableContinuationResumeState.COMPLETED:
            completion = self._continuation.completion
            assert completion is not None
            if completion.result_digest != digest:
                _correlation_error(
                    "resume.result_digest",
                    "completed continuation has another result digest",
                )
        elif self._state is DurableContinuationResumeState.AMBIGUOUS:
            if (
                self._continuation.claim.state
                is not ContinuationClaimState.DISPATCHED_AMBIGUOUS
            ):
                _illegal_transition(
                    "resume.admission",
                    "ambiguous continuation lacks a dispatch fence",
                )
        elif self._state is not DurableContinuationResumeState.DISPATCHED:
            _illegal_transition(
                "resume.admission",
                "continuation is not ready for completion",
            )
        if (
            self._state is not DurableContinuationResumeState.COMPLETED
            and self._continuation.claim.owner_id != self._owner_id
        ):
            _correlation_error(
                "resume.admission.owner_id",
                "continuation completion owner changed",
            )
        self._completion_digest = digest
        self._completion_command = self._make_completion_command(digest)
        return self._completion_command

    def _make_completion_command(
        self,
        result_digest: str,
    ) -> ContinuationCompletionCommand:
        """Build the exact first completion fence for stable replay."""
        return ContinuationCompletionCommand(
            continuation_id=self._continuation.continuation_id,
            expected_store_revision=self._continuation.store_revision,
            owner_id=self._owner_id,
            fencing_token=self._fencing_token,
            result_digest=result_digest,
        )

    def rejection_command(
        self,
        result_digest: str,
    ) -> ContinuationRejectionCommand:
        """Return a fence for deterministic pre-dispatch rejection."""
        digest = _validate_digest(result_digest, "resume.result_digest")
        continuation = self._continuation
        if (
            self._state is not DurableContinuationResumeState.ADMITTED
            or continuation.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
            or self._dispatch_task is not None
            or self._release_owned
        ):
            _illegal_transition(
                "resume.admission",
                "only an exclusively owned pre-dispatch admission can reject",
            )
        if continuation.claim.owner_id != self._owner_id:
            _correlation_error(
                "resume.admission.owner_id",
                "continuation rejection owner changed",
            )
        return ContinuationRejectionCommand(
            continuation_id=continuation.continuation_id,
            expected_store_revision=continuation.store_revision,
            owner_id=self._owner_id,
            fencing_token=self._fencing_token,
            result_digest=digest,
        )

    async def release(self) -> PortableContinuation:
        """Release an admitted claim only before provider dispatch starts."""
        async with self._lock:
            if self._state is DurableContinuationResumeState.RELEASED:
                return self._continuation
            if self._state is not DurableContinuationResumeState.ADMITTED:
                _illegal_transition(
                    "resume.admission",
                    "continuation can be released only before dispatch",
                )
            if self._dispatch_task is not None:
                _illegal_transition(
                    "resume.admission",
                    "dispatch ownership has already been claimed",
                )
            self._release_owned = True
            if self._release_task is None:
                self._release_task = create_task(
                    self._release_once(),
                    name=(
                        "durable-agent-release-"
                        f"{self._continuation.continuation_id}"
                    ),
                )
            task = self._release_task
        return await self._await_release(task)

    async def release_if_pre_dispatch(self) -> bool:
        """Release safely when no provider dispatch owns the admission."""
        async with self._lock:
            if self._state is DurableContinuationResumeState.RELEASED:
                return True
            if (
                self._state is not DurableContinuationResumeState.ADMITTED
                or self._dispatch_task is not None
            ):
                return False
            self._release_owned = True
            if self._release_task is None:
                self._release_task = create_task(
                    self._release_once(),
                    name=(
                        "durable-agent-release-"
                        f"{self._continuation.continuation_id}"
                    ),
                )
            task = self._release_task
        await self._await_release(task)
        return True

    async def close(self) -> None:
        """Close resources owned by this continuation admission."""
        await self._executor.close_continuation_runtime()

    async def _await_release(
        self,
        task: Task[PortableContinuation],
    ) -> PortableContinuation:
        try:
            return await shield(task)
        except BaseException:
            async with self._lock:
                if self._release_task is task and task.done():
                    self._release_task = None
            raise

    async def _dispatch_once(self) -> object:
        try:
            marked = await self._store.mark_dispatching(
                self._continuation.continuation_id,
                expected_store_revision=self._continuation.store_revision,
                owner_id=self._owner_id,
                fencing_token=self._fencing_token,
                now=self._now(),
            )
        except BaseException as error:
            try:
                released = await shield(
                    self._store.release(
                        self._continuation.continuation_id,
                        expected_store_revision=(
                            self._continuation.store_revision
                        ),
                        owner_id=self._owner_id,
                        fencing_token=self._fencing_token,
                        now=self._now(),
                    )
                )
                _validate_released_continuation(
                    self._continuation,
                    released,
                    owner_id=self._owner_id,
                    fencing_token=self._fencing_token,
                )
            except BaseException as release_error:
                self._state = DurableContinuationResumeState.AMBIGUOUS
                raise BaseExceptionGroup(
                    "dispatch marker and safe release both failed",
                    (error, release_error),
                ) from None
            self._continuation = released
            self._state = DurableContinuationResumeState.RELEASED
            raise
        try:
            _validate_dispatching_continuation(
                self._continuation,
                marked,
                owner_id=self._owner_id,
                fencing_token=self._fencing_token,
            )
        except BaseException:
            self._state = DurableContinuationResumeState.AMBIGUOUS
            raise
        self._continuation = marked
        self._state = DurableContinuationResumeState.DISPATCHING
        try:
            result = await self._executor.resume_agent_continuation(
                self._command
            )
            dispatched = await self._store.mark_dispatched(
                self._continuation.continuation_id,
                expected_store_revision=self._continuation.store_revision,
                owner_id=self._owner_id,
                fencing_token=self._fencing_token,
                now=self._now(),
            )
            _validate_dispatched_continuation(
                self._continuation,
                dispatched,
                owner_id=self._owner_id,
                fencing_token=self._fencing_token,
            )
        except BaseException:
            self._state = DurableContinuationResumeState.AMBIGUOUS
            raise
        self._continuation = dispatched
        self._state = DurableContinuationResumeState.DISPATCHED
        return result

    async def _complete_once(
        self,
        result_digest: str,
    ) -> PortableContinuation:
        try:
            completed = await self._store.complete(
                self._continuation.continuation_id,
                expected_store_revision=self._continuation.store_revision,
                owner_id=self._owner_id,
                fencing_token=self._fencing_token,
                result_digest=result_digest,
                now=self._now(),
            )
            _validate_completed_continuation(
                self._continuation,
                completed,
                fencing_token=self._fencing_token,
                result_digest=result_digest,
            )
        except BaseException as error:
            try:
                observed = await shield(
                    self._store.get_continuation(
                        self._continuation.continuation_id
                    )
                )
                _validate_completed_continuation(
                    self._continuation,
                    observed,
                    fencing_token=self._fencing_token,
                    result_digest=result_digest,
                )
            except BaseException as verification_error:
                self._state = DurableContinuationResumeState.DISPATCHED
                raise BaseExceptionGroup(
                    "completion write failed without matching "
                    "durable readback",
                    (error, verification_error),
                ) from None
            completed = observed
        self._continuation = completed
        self._state = DurableContinuationResumeState.COMPLETED
        return completed

    async def _release_once(self) -> PortableContinuation:
        released = await self._store.release(
            self._continuation.continuation_id,
            expected_store_revision=self._continuation.store_revision,
            owner_id=self._owner_id,
            fencing_token=self._fencing_token,
            now=self._now(),
        )
        _validate_released_continuation(
            self._continuation,
            released,
            owner_id=self._owner_id,
            fencing_token=self._fencing_token,
        )
        self._continuation = released
        self._state = DurableContinuationResumeState.RELEASED
        return released

    def _now(self) -> datetime:
        return validate_aware_datetime(self._clock(), "resume.clock")


@final
class DurableAgentContinuationClaimLease:
    """Renew one exact claimed continuation during cold reconstruction."""

    def __init__(
        self,
        *,
        store: AgentDurableContinuationStore,
        continuation: PortableContinuation,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
    ) -> None:
        _validate_store(store)
        if type(continuation) is not PortableContinuation:
            _invalid_type("resume.claim_lease.continuation", "a continuation")
        if (
            continuation.claim.state
            is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
            or continuation.claim.owner_id != owner_id
            or continuation.fencing_token != fencing_token
        ):
            _correlation_error(
                "resume.claim_lease",
                "claim lease does not match continuation ownership",
            )
        self._store = store
        self._continuation_id = continuation.continuation_id
        self._expected_store_revision = continuation.store_revision
        self._owner_id = owner_id
        self._fencing_token = fencing_token
        self._expires_at = continuation.expires_at

    async def renew(
        self,
        lease_expires_at: datetime,
        *,
        now: datetime,
    ) -> bool:
        """Renew the fence without outliving the continuation deadline."""
        now = validate_aware_datetime(now, "resume.claim_lease.now")
        requested = validate_aware_datetime(
            lease_expires_at,
            "resume.claim_lease.lease_expires_at",
        )
        effective = min(requested, self._expires_at)
        renewed = await self._store.renew_claim(
            self._continuation_id,
            expected_store_revision=self._expected_store_revision,
            owner_id=self._owner_id,
            fencing_token=self._fencing_token,
            lease_expires_at=effective,
            now=now,
        )
        if type(renewed) is not bool:
            _invalid_type(
                "resume.claim_lease.renewal",
                "a boolean renewal result",
            )
        return renewed


@final
class DurableAgentContinuationResumer:
    """Admit a fresh-runtime continuation before any provider side effect."""

    def __init__(
        self,
        store: AgentDurableContinuationStore,
        resolver: ContinuationRuntimeResolver,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        _validate_store(store)
        if type(resolver) is not ContinuationRuntimeResolver:
            _invalid_type(
                "resume.resolver",
                "a continuation runtime resolver",
            )
        if clock is not None and not callable(clock):
            _invalid_type("resume.clock", "a clock callable")
        self._store = store
        self._resolver = resolver
        self._clock = clock or (lambda: datetime.now(UTC))

    async def admit(
        self,
        record: DurableContinuationRecord,
        *,
        actor: InteractionActor,
        expected_request_id: InputRequestId,
        expected_run_id: RunId,
        expected_checkpoint_id: str,
        owner_id: ContinuationClaimOwnerId,
        lease_expires_at: datetime,
        dispatch_id: ContinuationDispatchId,
        lease_expires_at_provider: (
            Callable[[], Awaitable[datetime]] | None
        ) = None,
        claim_lease_observer: (
            Callable[
                [DurableAgentContinuationClaimLease],
                Awaitable[None],
            ]
            | None
        ) = None,
    ) -> DurableAgentContinuationAdmission:
        """Claim, reconstruct, and validate one continuation for dispatch."""
        if type(record) is not DurableContinuationRecord:
            _invalid_type("resume.record", "a durable continuation record")
        if not isinstance(actor, InteractionActor):
            _invalid_type("resume.actor", "an interaction actor")
        if lease_expires_at_provider is not None and not callable(
            lease_expires_at_provider
        ):
            _invalid_type(
                "resume.lease_expires_at_provider",
                "an asynchronous lease provider",
            )
        if claim_lease_observer is not None and not callable(
            claim_lease_observer
        ):
            _invalid_type(
                "resume.claim_lease_observer",
                "an asynchronous claim lease observer",
            )
        continuation = record.continuation
        _validate_expected_identity(
            record,
            actor=actor,
            expected_request_id=expected_request_id,
            expected_run_id=expected_run_id,
            expected_checkpoint_id=expected_checkpoint_id,
        )
        terminal_request = await self._terminal_request(
            continuation,
            actor=actor,
        )
        snapshot = continuation.provider_snapshot
        if snapshot is None:
            _unavailable(
                "resume.continuation.provider_snapshot",
                "durable provider replay state is unavailable",
            )
        provider_idempotency_key = derive_provider_idempotency_key(
            continuation.continuation_id,
            dispatch_id,
        )
        if snapshot.provider_idempotency_key != provider_idempotency_key:
            _correlation_error(
                "resume.continuation.provider_snapshot."
                "provider_idempotency_key",
                "provider retry key does not match durable dispatch identity",
            )
        if lease_expires_at_provider is not None:
            lease_expires_at = await lease_expires_at_provider()
        lease_expires_at = min(
            validate_aware_datetime(
                lease_expires_at,
                "resume.lease_expires_at",
            ),
            continuation.expires_at,
        )
        now = self._now()
        if lease_expires_at <= now:
            raise InputValidationError(
                InputErrorCode.EXPIRED,
                "resume.lease_expires_at",
                "continuation claim lease expired before admission",
            )
        receipt = await self._store.claim(
            continuation.continuation_id,
            expected_store_revision=continuation.store_revision,
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
            dispatch_id=dispatch_id,
            provider_idempotency_key=provider_idempotency_key,
            now=now,
        )
        _validate_claim_receipt(
            continuation,
            receipt,
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
            dispatch_id=dispatch_id,
            provider_idempotency_key=provider_idempotency_key,
            claimed_at=now,
        )
        claimed = receipt.continuation
        executor: AgentContinuationExecutor | None = None
        try:
            if claim_lease_observer is not None:
                await claim_lease_observer(
                    DurableAgentContinuationClaimLease(
                        store=self._store,
                        continuation=claimed,
                        owner_id=owner_id,
                        fencing_token=receipt.fencing_token,
                    )
                )
            resolved = await self._resolver.resolve(claimed)
            executor = cast(AgentContinuationExecutor, resolved.runtime)
            _validate_executor(executor)
            if self._now() >= claimed.expires_at:
                raise InputValidationError(
                    InputErrorCode.EXPIRED,
                    "resume.continuation.expires_at",
                    "continuation expired during runtime reconstruction",
                )
            catalog = _validated_catalog(resolved, claimed)
            call = _task_input_call(claimed, terminal_request, catalog)
            _restore_provider_snapshot(resolved, claimed, call)
            outcome = project_resolution_to_model(
                terminal_request,
                containing_run_exists=True,
            )
            assert type(outcome) is ResumeInputContinuation
            correlated = catalog.project_result(call, outcome.result)
            command = AgentContinuationResumeCommand(
                continuation=claimed,
                request=terminal_request,
                model_result=outcome.result,
                task_input_call=call,
                correlated_result=correlated,
                resolved_runtime=resolved,
            )
            return DurableAgentContinuationAdmission(
                store=self._store,
                command=command,
                owner_id=owner_id,
                fencing_token=receipt.fencing_token,
                executor=executor,
                clock=self._clock,
            )
        except BaseException as error:
            cleanup = create_task(
                self._cleanup_failed_admission(
                    claimed,
                    owner_id=owner_id,
                    fencing_token=receipt.fencing_token,
                    executor=executor,
                ),
                name=(
                    "durable-agent-admission-cleanup-"
                    f"{claimed.continuation_id}"
                ),
            )
            try:
                convergence_errors = await _await_owned_admission_cleanup(
                    cleanup
                )
            except BaseException as cleanup_error:
                convergence_errors = (cleanup_error,)
            if convergence_errors:
                raise BaseExceptionGroup(
                    "continuation setup convergence failed",
                    (error, *convergence_errors),
                ) from None
            raise

    async def _cleanup_failed_admission(
        self,
        claimed: PortableContinuation,
        *,
        owner_id: ContinuationClaimOwnerId,
        fencing_token: ContinuationFencingToken,
        executor: AgentContinuationExecutor | None,
    ) -> tuple[BaseException, ...]:
        """Release the claim and close every reconstructed runtime resource."""
        errors: list[BaseException] = []
        try:
            released = await self._store.release(
                claimed.continuation_id,
                expected_store_revision=claimed.store_revision,
                owner_id=owner_id,
                fencing_token=fencing_token,
                now=self._now(),
            )
            _validate_released_continuation(
                claimed,
                released,
                owner_id=owner_id,
                fencing_token=fencing_token,
            )
        except BaseException as error:
            errors.append(error)
        if executor is not None:
            try:
                await executor.close_continuation_runtime()
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    async def _terminal_request(
        self,
        continuation: PortableContinuation,
        *,
        actor: InteractionActor,
    ) -> InputRequest:
        origin = continuation.origin
        projection = await self._store.lookup_scoped(
            ScopedInteractionLookup(
                actor=actor,
                correlation=InteractionCorrelation(
                    request_id=continuation.request_id,
                    continuation_id=continuation.continuation_id,
                    run_id=origin.run_id,
                    turn_id=origin.turn_id,
                    task_id=origin.task_id,
                    agent_id=origin.agent_id,
                    branch_id=origin.branch_id,
                    model_call_id=origin.model_call_id,
                ),
            )
        )
        if not isinstance(projection, InteractionRecord):
            _unavailable(
                "resume.interaction",
                "authorized terminal interaction is unavailable",
            )
        request = projection.request
        if request.resolution is None:
            _illegal_transition(
                "resume.interaction.state",
                "durable continuation requires a terminal resolution",
            )
        if (
            request.request_id != continuation.request_id
            or request.continuation_id != continuation.continuation_id
            or request.origin != continuation.origin
        ):
            _correlation_error(
                "resume.interaction",
                "terminal interaction does not match the continuation",
            )
        outcome = project_resolution_to_model(
            request,
            containing_run_exists=True,
        )
        if type(outcome) is not ResumeInputContinuation:
            _illegal_transition(
                "resume.interaction.resolution",
                "terminal interaction cannot resume its containing run",
            )
        return request

    def _now(self) -> datetime:
        return validate_aware_datetime(self._clock(), "resume.clock")


async def _await_owned_admission_cleanup(
    task: Task[tuple[BaseException, ...]],
) -> tuple[BaseException, ...]:
    """Join admission cleanup despite repeated caller cancellation."""
    while True:
        try:
            return await shield(task)
        except CancelledError:
            if task.done():
                return task.result()


def _validate_expected_identity(
    record: DurableContinuationRecord,
    *,
    actor: InteractionActor,
    expected_request_id: InputRequestId,
    expected_run_id: RunId,
    expected_checkpoint_id: str,
) -> None:
    continuation = record.continuation
    checkpoint_id = validate_opaque_id(
        expected_checkpoint_id,
        "resume.expected_checkpoint_id",
    )
    if record.task_run_id is None or record.task_run_id != str(
        expected_run_id
    ):
        _correlation_error(
            "resume.record.task_run_id",
            "durable record is not bound to the requeued task run",
        )
    if record.checkpoint_id != checkpoint_id:
        _correlation_error(
            "resume.record.checkpoint_id",
            "durable record does not match the suspended checkpoint",
        )
    if continuation.request_id != expected_request_id:
        _correlation_error(
            "resume.expected_request_id",
            "requeued request does not match the continuation",
        )
    if continuation.origin.run_id != expected_run_id or (
        continuation.origin.task_id is not None
        and str(continuation.origin.task_id) != str(expected_run_id)
    ):
        _correlation_error(
            "resume.expected_run_id",
            "requeued task run does not match the continuation",
        )
    if actor.principal != continuation.origin.principal:
        _correlation_error(
            "resume.actor.principal",
            "resume actor does not own the continuation",
        )


def _validate_claim_receipt(
    previous: PortableContinuation,
    receipt: ContinuationClaimReceipt,
    *,
    owner_id: ContinuationClaimOwnerId,
    lease_expires_at: datetime,
    dispatch_id: ContinuationDispatchId,
    provider_idempotency_key: ProviderIdempotencyKey,
    claimed_at: datetime,
) -> None:
    if type(receipt) is not ContinuationClaimReceipt:
        _invalid_type(
            "resume.claim_receipt",
            "a continuation claim receipt",
        )
    claimed = receipt.continuation
    _validate_portable_payload_unchanged(previous, claimed)
    dispatch = claimed.dispatch
    if (
        claimed.claim.state is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        or claimed.claim.owner_id != owner_id
        or claimed.claim.lease_expires_at != lease_expires_at
        or claimed.claim.attempt != previous.claim.attempt + 1
        or claimed.fencing_token
        != ContinuationFencingToken(int(previous.fencing_token) + 1)
        or receipt.fencing_token != claimed.fencing_token
        or int(claimed.store_revision) != int(previous.store_revision) + 1
        or claimed.updated_at != claimed_at
        or type(dispatch) is not ContinuationDispatch
        or dispatch.dispatch_id != dispatch_id
        or dispatch.provider_idempotency_key != provider_idempotency_key
        or dispatch.marked_at != claimed_at
        or claimed.completion is not None
    ):
        _correlation_error(
            "resume.claim_receipt",
            "claim receipt changed durable continuation ownership",
        )


def _validate_dispatching_continuation(
    previous: PortableContinuation,
    current: PortableContinuation,
    *,
    owner_id: ContinuationClaimOwnerId,
    fencing_token: ContinuationFencingToken,
) -> None:
    _validate_portable_payload_unchanged(previous, current)
    if (
        current.claim.state is not ContinuationClaimState.DISPATCHED_AMBIGUOUS
        or current.claim.owner_id != owner_id
        or current.claim.lease_expires_at is not None
        or current.claim.attempt != previous.claim.attempt
        or current.fencing_token != fencing_token
        or current.dispatch != previous.dispatch
        or current.completion is not None
        or int(current.store_revision) != int(previous.store_revision) + 1
        or current.updated_at < previous.updated_at
    ):
        _correlation_error(
            "resume.dispatching",
            "dispatch marker changed continuation ownership",
        )


def _validate_dispatched_continuation(
    previous: PortableContinuation,
    current: PortableContinuation,
    *,
    owner_id: ContinuationClaimOwnerId,
    fencing_token: ContinuationFencingToken,
) -> None:
    _validate_dispatching_continuation(
        previous,
        current,
        owner_id=owner_id,
        fencing_token=fencing_token,
    )


def _validate_released_continuation(
    previous: PortableContinuation,
    current: PortableContinuation,
    *,
    owner_id: ContinuationClaimOwnerId,
    fencing_token: ContinuationFencingToken,
) -> None:
    _validate_portable_payload_unchanged(previous, current)
    if (
        previous.claim.state is not ContinuationClaimState.CLAIMED_PRE_DISPATCH
        or previous.claim.owner_id != owner_id
        or previous.fencing_token != fencing_token
        or current.claim.state
        is not ContinuationClaimState.FAILED_SAFE_TO_RETRY
        or current.claim.owner_id is not None
        or current.claim.lease_expires_at is not None
        or current.claim.attempt != previous.claim.attempt
        or current.fencing_token != fencing_token
        or current.dispatch != previous.dispatch
        or current.completion is not None
        or int(current.store_revision) != int(previous.store_revision) + 1
        or current.updated_at < previous.updated_at
    ):
        _correlation_error(
            "resume.release",
            "released continuation changed its durable fence",
        )


def _validate_completed_continuation(
    previous: PortableContinuation,
    current: PortableContinuation,
    *,
    fencing_token: ContinuationFencingToken,
    result_digest: str,
) -> None:
    _validate_portable_payload_unchanged(previous, current)
    completion = current.completion
    if (
        current.claim.state is not ContinuationClaimState.COMPLETED
        or current.claim.owner_id is not None
        or current.claim.lease_expires_at is not None
        or current.claim.attempt != previous.claim.attempt
        or current.fencing_token != fencing_token
        or current.dispatch != previous.dispatch
        or type(completion) is not ContinuationCompletion
        or completion.result_digest != result_digest
        or int(current.store_revision) != int(previous.store_revision) + 1
        or current.updated_at < previous.updated_at
    ):
        _correlation_error(
            "resume.completion",
            "completed continuation changed its durable dispatch",
        )


def _validate_portable_payload_unchanged(
    previous: PortableContinuation,
    current: PortableContinuation,
) -> None:
    if type(current) is not PortableContinuation:
        _invalid_type("resume.continuation", "a portable continuation")
    fields = (
        "version",
        "continuation_id",
        "request_id",
        "origin",
        "provider_call_id",
        "provider_call_correlation_id",
        "definition",
        "operation_cursor",
        "generation_settings",
        "transcript",
        "observations",
        "provider_snapshot",
        "revision_binding",
        "interaction_count",
        "tool_loop_count",
        "stream_sequence",
        "state_revision",
        "created_at",
        "expires_at",
    )
    if any(
        getattr(current, field_name) != getattr(previous, field_name)
        for field_name in fields
    ):
        _correlation_error(
            "resume.continuation",
            "durable continuation payload changed during state transition",
        )


def _validated_catalog(
    runtime: ResolvedContinuationRuntime,
    continuation: PortableContinuation,
) -> ModelCapabilityCatalog:
    if type(runtime.capabilities) is not ModelCapabilityCatalog:
        _unavailable(
            "resume.runtime.capabilities",
            "fresh runtime lacks a trusted model capability catalog",
        )
    catalog = runtime.capabilities
    if (
        catalog.revision_binding != continuation.revision_binding
        or catalog.task_input_advertisement
        is not TaskInputCapabilityAdvertisement.DURABLE
    ):
        _unavailable(
            "resume.runtime.capabilities",
            "fresh runtime does not advertise the exact durable capability",
        )
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    registry = catalog.support.continuation_snapshot_codec_registry
    codec = catalog.support.continuation_snapshot_codec
    if (
        registry is None
        or codec is None
        or not registry.is_registered(codec)
        or not codec.accepts(snapshot)
    ):
        _unavailable(
            "resume.runtime.capabilities",
            "provider snapshot codec is unavailable for the exact revision",
        )
    encoded = registry.export_snapshot(codec, snapshot)
    if (
        registry.restore_snapshot(
            codec,
            encoded,
            continuation.revision_binding,
        )
        != snapshot
    ):
        _unavailable(
            "resume.runtime.capabilities",
            "provider snapshot codec changed replay state",
        )
    return catalog


def _restore_provider_snapshot(
    runtime: ResolvedContinuationRuntime,
    continuation: PortableContinuation,
    call: TaskInputCapabilityCall,
) -> None:
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    adapter = runtime.model
    validate = getattr(adapter, "validate_continuation_snapshot_call", None)
    restore = getattr(adapter, "import_continuation_snapshot", None)
    if not callable(validate) or not callable(restore):
        _unavailable(
            "resume.runtime.model",
            "fresh provider adapter cannot validate and restore "
            "continuation state",
        )
    validate(
        snapshot,
        expected_binding=continuation.revision_binding,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
        expected_provider_name=call.provider_name,
        expected_arguments=call.arguments,
    )
    restore(
        snapshot,
        expected_binding=continuation.revision_binding,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
    )


def _task_input_call(
    continuation: PortableContinuation,
    request: InputRequest,
    catalog: ModelCapabilityCatalog,
) -> TaskInputCapabilityCall:
    provider_family = str(continuation.revision_binding.provider_family)
    provider_name = catalog.provider_name(
        RESERVED_INPUT_CAPABILITY_NAME,
        provider_family=provider_family,
    )
    arguments: dict[str, object] = {
        "mode": request.mode.value,
        "reason": request.reason,
        "questions": [
            encode_input_question(question) for question in request.questions
        ],
    }
    call = catalog.decode_call(
        ProviderCapabilityCall(
            call_id=continuation.provider_call_correlation_id,
            provider_name=provider_name,
            arguments=arguments,
        ),
        provider_family=provider_family,
    )
    if type(call) is not TaskInputCapabilityCall:
        _correlation_error(
            "resume.task_input_call",
            "reserved provider call did not decode as task input",
        )
    return call


def _validate_store(store: object) -> None:
    methods = (
        "lookup_scoped",
        "claim",
        "renew_claim",
        "mark_dispatching",
        "mark_dispatched",
        "complete",
        "get_continuation",
        "release",
    )
    if any(
        not iscoroutinefunction(getattr(store, method, None))
        for method in methods
    ):
        _invalid_type(
            "resume.store",
            "an asynchronous durable continuation store",
        )


def _validate_executor(executor: object) -> None:
    if (
        getattr(executor, "trusted_agent_continuation_executor", False)
        is not True
        or not callable(getattr(executor, "register_event_listener", None))
        or any(
            not iscoroutinefunction(getattr(executor, method, None))
            for method in (
                "resume_agent_continuation",
                "close_continuation_runtime",
            )
        )
    ):
        _unavailable(
            "resume.runtime.executor",
            "fresh runtime lacks a trusted async continuation executor",
        )


def _validate_digest(value: object, path: str) -> str:
    if not isinstance(value, str) or fullmatch(_SHA256_PATTERN, value) is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a lowercase SHA-256 digest",
        )
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


def _unavailable(path: str, message: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.UNAVAILABLE,
        path,
        message,
    )
