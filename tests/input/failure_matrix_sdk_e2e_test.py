"""Exercise every active public SDK failure-matrix cell."""

from asyncio import Event, create_task, run, wait_for
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from sys import path as sys_path
from types import MappingProxyType
from typing import Any, Literal, cast

import pytest

from avalan import (
    AgentRunCancelled,
    AgentRunCompleted,
    AgentRunFailed,
    AgentRunInputRequired,
    AnswerProvenance,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    InputAlreadyResolvedError,
    InputAnswerSubmission,
    InputContinuationRef,
    InputExpiredError,
    InputRequestRef,
    InputResolutionAccepted,
    InputSupersededError,
    InputValidationError,
    QuestionId,
    ResolutionIdempotencyKey,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    TextAnswer,
    resolve_input,
    run_agent,
)
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    ExecutionInputRequiredError,
)
from avalan.interaction.broker import (
    AsyncInteractionBroker,
    InteractionDelivery,
    InteractionRequestResult,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    AnsweredResolution,
    InputRequiredResult,
    InputTimedOutResult,
    PrincipalScope,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    RunId,
    TaskId,
)
from avalan.interaction.error import InputErrorCode
from avalan.interaction.headless import (
    DurableHandoffInputPolicy,
    PredeclaredInputPolicy,
)
from avalan.interaction.policy import InteractionActor, InteractionPolicy
from avalan.interaction.store import (
    CancelInteractionApplied,
    CancelInteractionCommand,
    CreateInteractionApplied,
    InteractionCorrelation,
    InteractionExecutionScope,
    InteractionPresentationApplied,
    InteractionRecord,
    InteractionResolutionResult,
    InteractionStoreReplayed,
    PresentInteractionCommand,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    SupersedeInteractionScopeCommand,
    TerminalizeDueInteractionsCommand,
)
from avalan.sdk import AsyncInputController
from avalan.task import TaskRunState

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))
sys_path.append(str(Path(__file__).parent))

import broker_contract_test as broker_support  # noqa: E402
import failure_matrix_task_e2e_test as task_support  # noqa: E402
import interaction_pgsql_store_test as durable_support  # noqa: E402
import sdk_contract_test as sdk_support  # noqa: E402
from pgsql_support import FakePgsqlDatabase  # noqa: E402

_EVIDENCE_PROPERTY = "failure_matrix_evidence"
_FOUR_SDK_SURFACES = (
    "sdk-attached",
    "sdk-headless-predeclared",
    "sdk-headless-durable",
    "sdk-serverless-durable",
)
_ALL_SDK_SURFACES = (*_FOUR_SDK_SURFACES, "sdk-sessionless")


@dataclass(frozen=True, slots=True)
class _SurfaceObservation:
    """Own the exact dynamic evidence emitted by one SDK surface."""

    condition_id: str
    surface_id: str
    transition_from: RequestState
    transition_to: RequestState
    public_result_id: str
    public_result: Mapping[str, object]
    status_key: str
    status_value: str
    provider_call_count: int
    domain_side_effect_count: int

    def __post_init__(self) -> None:
        assert self.condition_id.startswith("INPUT-F-")
        assert self.surface_id.startswith("sdk-")
        assert isinstance(self.transition_from, RequestState)
        assert isinstance(self.transition_to, RequestState)
        assert self.public_result_id.startswith("sdk.")
        assert self.public_result
        assert self.status_key
        assert self.status_value
        assert self.provider_call_count >= 0
        assert self.domain_side_effect_count >= 0
        object.__setattr__(
            self,
            "public_result",
            MappingProxyType(dict(self.public_result)),
        )

    def evidence(self) -> dict[str, object]:
        """Return this surface-owned observation in verifier wire form."""
        return {
            "condition_id": self.condition_id,
            "surface_id": self.surface_id,
            "transition_from": self.transition_from.value,
            "transition_to": self.transition_to.value,
            "public_result_id": self.public_result_id,
            "public_result": dict(self.public_result),
            "status_key": self.status_key,
            "status_value": self.status_value,
            "provider_call_count": self.provider_call_count,
            "domain_side_effect_count": self.domain_side_effect_count,
        }


@dataclass(slots=True)
class _CausalSurface:
    """Own one request created through its named SDK host boundary."""

    surface_id: str
    request_ref: InputRequestRef | None
    continuation_ref: InputContinuationRef | None
    controller: AsyncInputController | None
    broker: AsyncInteractionBroker
    actor: InteractionActor
    clock: Any
    correlation: InteractionCorrelation
    resolver_commands: list[ResolveInteractionCommand]
    provider_call_count: int
    domain_side_effect_count: int
    resumer: Any | None = None
    durable: Any | None = None

    async def record(self) -> InteractionRecord:
        """Return the exact request owned by this surface."""
        projection = await self.broker.inspect(
            ScopedInteractionLookup(
                actor=self.actor,
                correlation=self.correlation,
            )
        )
        assert isinstance(projection, InteractionRecord)
        return projection

    async def close(self) -> None:
        """Close resources allocated by this exact surface."""
        await self.broker.aclose()
        if self.durable is not None:
            await self.durable.stack.aclose()
            self.durable.temporary.cleanup()


class _CausalAttachedOrchestrator:
    """Create input through the attached runtime supplied by ``run_agent``."""

    def __init__(self, request: object) -> None:
        self.request = request
        self.result: InteractionRequestResult | None = None
        self.provider_call_count = 0
        self.domain_side_effects: list[object] = []

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        self.provider_call_count += 1
        request = replace(cast(Any, self.request), handler=runtime.handler)
        self.result = await runtime.broker.request(request)
        delivery = self.result.delivery
        assert isinstance(delivery, InteractionDelivery)
        return _CompletedResponse()


class _AttachedAdvisoryOrchestrator:
    """Continue the same public run after its broker-owned timeout."""

    def __init__(
        self,
        request: object,
        resumer: broker_support._Resumer,
    ) -> None:
        self.request = request
        self.resumer = resumer
        self.result: InteractionRequestResult | None = None
        self.provider_call_count = 0
        self.domain_side_effects: list[object] = []

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        self.provider_call_count += 1
        request = replace(cast(Any, self.request), handler=runtime.handler)
        self.result = await runtime.broker.request(request)
        delivery = self.result.delivery
        assert isinstance(delivery, InteractionDelivery)
        assert delivery.record.request.state is RequestState.PENDING
        await self.resumer.called.wait()
        settled = await runtime.broker.inspect(
            ScopedInteractionLookup(
                actor=runtime.actor,
                correlation=delivery.correlation,
            )
        )
        assert isinstance(settled, InteractionRecord)
        assert settled.request.state is RequestState.TIMED_OUT
        self.provider_call_count += 1
        return _CompletedResponse()


class _DurableAdvisoryOrchestrator:
    """Continue after the configured durable stager reaches timeout."""

    def __init__(
        self,
        store: Any,
        clock: Any,
        staged: Event,
        correlation: InteractionCorrelation,
        actor: InteractionActor,
    ) -> None:
        self.store = store
        self.clock = clock
        self.staged = staged
        self.correlation = correlation
        self.actor = actor
        self.provider_call_count = 0
        self.record: InteractionRecord | None = None
        self.domain_side_effects: list[object] = []

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, DurableInteractionRuntime)
        self.provider_call_count += 1
        suspension = await cast(Any, runtime.stager)(object())
        assert isinstance(suspension, DurableInteractionSuspension)
        self.staged.set()
        await self.clock.wait_until(2.0)
        due = await self.store.terminalize_due(
            TerminalizeDueInteractionsCommand()
        )
        assert due.store_mutation_applied
        projection = await self.store.lookup_scoped(
            ScopedInteractionLookup(
                actor=self.actor,
                correlation=self.correlation,
            )
        )
        assert isinstance(projection, InteractionRecord)
        assert projection.request.state is RequestState.TIMED_OUT
        self.record = projection
        self.provider_call_count += 1
        return _CompletedResponse()


def _question_for_condition(
    condition_id: str,
) -> ConfirmationQuestion | SingleSelectionQuestion:
    if condition_id == "INPUT-F-05":
        return SingleSelectionQuestion(
            question_id=QuestionId("answer"),
            prompt="Choose one.",
            required=True,
            choices=(
                Choice(
                    value=ChoiceValue("known"),
                    label="Known",
                ),
                Choice(
                    value=ChoiceValue("other-known"),
                    label="Other known",
                ),
            ),
        )
    return ConfirmationQuestion(
        question_id=QuestionId("answer"),
        prompt="Continue?",
        required=True,
    )


def _predeclared_pending_answers(
    question: ConfirmationQuestion | SingleSelectionQuestion,
    condition_id: str,
) -> tuple[object, ...]:
    """Return policy values that causally leave this request unresolved."""
    if condition_id == "INPUT-F-05":
        return (
            SingleSelectionAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.POLICY,
                value=SelectedChoice(value=ChoiceValue("unknown")),
            ),
        )
    if condition_id == "INPUT-F-06":
        return ()
    return (
        TextAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.POLICY,
            value="not a confirmation",
        ),
    )


def _submission_for_condition(
    question: ConfirmationQuestion | SingleSelectionQuestion,
    condition_id: str,
    *,
    value: bool = True,
) -> InputAnswerSubmission:
    answers: tuple[
        ConfirmationAnswer | SingleSelectionAnswer | TextAnswer,
        ...,
    ]
    if condition_id == "INPUT-F-04":
        answers = (
            TextAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value="not a confirmation",
            ),
        )
    elif condition_id == "INPUT-F-05":
        answers = (
            SingleSelectionAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("unknown")),
            ),
        )
    elif condition_id == "INPUT-F-06":
        answers = ()
    else:
        answers = (
            ConfirmationAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=value,
            ),
        )
    return InputAnswerSubmission(
        answers=cast(Any, answers),
        provenance=AnswerProvenance.HUMAN,
    )


async def _open_attached_surface(
    surface_id: str,
    condition_id: str,
    question: ConfirmationQuestion | SingleSelectionQuestion,
    *,
    mode: RequirementMode = RequirementMode.REQUIRED,
    advisory_wait_seconds: int | None = None,
    continuation_ttl_seconds: int = 600,
) -> _CausalSurface:
    """Admit one request through a real attached or predeclared runtime."""
    assert surface_id in {"sdk-attached", "sdk-headless-predeclared"}
    harness = await broker_support._harness()
    resumer = broker_support._Resumer()
    template = broker_support._request(
        None,
        run_id=f"{condition_id}-{surface_id}",
        resumer=resumer,
        mode=mode,
        advisory_wait_seconds=advisory_wait_seconds,
        continuation_ttl_seconds=continuation_ttl_seconds,
        default_value=False if mode is RequirementMode.ADVISORY else None,
        reason=f"Exercise {condition_id} through {surface_id}.",
    )
    template = replace(template, questions=(question,))
    orchestrator = _CausalAttachedOrchestrator(template)
    runtime = AttachedInteractionRuntime(
        broker=harness.broker,
        actor=broker_support._actor(),
        handler=broker_support._DetachedHandler(),
    )
    policy = (
        PredeclaredInputPolicy(
            answers=cast(
                Any,
                _predeclared_pending_answers(question, condition_id),
            )
        )
        if surface_id == "sdk-headless-predeclared"
        else None
    )
    result = await run_agent(
        cast(Any, orchestrator),
        surface_id,
        interaction_runtime=runtime,
        headless_policy=policy,
    )
    assert isinstance(result, AgentRunCompleted)
    requested = orchestrator.result
    assert requested is not None
    delivery = requested.delivery
    assert isinstance(delivery, InteractionDelivery)
    assert delivery.record.request.state is RequestState.PENDING
    if surface_id == "sdk-headless-predeclared":
        assert delivery.handler_attempts == 2
    else:
        assert delivery.handler_attempts == 1
    correlation = delivery.record.correlation
    return _CausalSurface(
        surface_id=surface_id,
        request_ref=None,
        continuation_ref=None,
        controller=None,
        broker=harness.broker,
        actor=broker_support._actor(),
        clock=harness.clock,
        correlation=correlation,
        resolver_commands=[],
        provider_call_count=orchestrator.provider_call_count,
        domain_side_effect_count=len(orchestrator.domain_side_effects),
        resumer=resumer,
    )


async def _open_durable_surface(
    surface_id: str,
    question: ConfirmationQuestion | SingleSelectionQuestion,
    *,
    resolver_strategy: Literal[
        "store",
        "coordinator",
        "coordinator_then_store",
    ],
) -> _CausalSurface:
    """Admit one exact run-bound or task-bound durable SDK request."""
    assert surface_id in {
        "sdk-headless-durable",
        "sdk-serverless-durable",
    }
    suspended = await task_support._durable_failure_harness(question)
    suspension = suspended.target.suspensions[0]
    policy = InteractionPolicy()
    broker = AsyncInteractionBroker(
        store=suspended.interaction_store,
        clock=suspended.clock,
        id_factory=task_support._TestIds(),
        policy=policy,
        classifier=task_support._TestClassifier(policy),
    )
    resolver_commands: list[ResolveInteractionCommand] = []

    async def stager(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return suspension

    async def handoff(received: object) -> object:
        assert received == suspension
        return await task_support._persisted_request(suspended)

    async def waiter(seconds: int) -> None:
        assert seconds == 1

    async def durable_authority(
        correlation: InteractionCorrelation,
    ) -> bool:
        record = (
            await suspended.interaction_store.get_task_continuation_record(
                suspended.run_id
            )
        )
        continuation = record.continuation
        return bool(
            record.task_run_id == suspended.run_id
            and continuation.request_id == correlation.request_id
            and continuation.continuation_id == correlation.continuation_id
            and continuation.origin.run_id == correlation.run_id
        )

    async def durable_resolver(
        command: ResolveInteractionCommand,
    ) -> InteractionResolutionResult:
        resolver_commands.append(command)
        use_coordinator = resolver_strategy == "coordinator" or (
            resolver_strategy == "coordinator_then_store"
            and len(resolver_commands) == 1
        )
        if use_coordinator:
            return (
                await suspended.coordinator.resolve_and_requeue(
                    command,
                    task_run_id=suspended.run_id,
                    now=suspended.clock.now,
                )
            ).resolution
        return await suspended.interaction_store.resolve(command)

    error = ExecutionInputRequiredError(
        InputRequiredResult(
            request_id=suspension.command.request.request_id,
            continuation_id=suspension.command.request.continuation_id,
            detached_resumption_available=True,
        ),
        durable=suspension,
    )
    runtime = DurableInteractionRuntime(
        actor=suspension.command.actor,
        stager=cast(Any, stager),
        run_id=(
            suspended.request.origin.run_id
            if surface_id == "sdk-headless-durable"
            else None
        ),
        task_id=(
            suspended.request.origin.task_id
            if surface_id == "sdk-serverless-durable"
            else None
        ),
    )
    pause = await run_agent(
        cast(Any, sdk_support._InputRequiredOrchestrator(error)),
        surface_id,
        interaction_runtime=runtime,
        headless_policy=DurableHandoffInputPolicy(
            handoff=cast(Any, handoff),
            durable_handoff_wait_seconds=1,
            waiter=waiter,
        ),
    )
    assert isinstance(pause, AgentRunInputRequired)
    assert pause.request_id is not None
    assert pause.continuation_id is not None
    if surface_id == "sdk-headless-durable":
        assert runtime.run_id is not None and runtime.task_id is None
    else:
        assert runtime.task_id is not None and runtime.run_id is None
    provider_call_count = task_support._provider_call_count(
        suspended.model_factory
    )
    inspection = await suspended.client.inspect(suspended.run_id)
    assert provider_call_count == 1
    assert inspection.artifacts == ()
    return _CausalSurface(
        surface_id=surface_id,
        request_ref=pause.request_id,
        continuation_ref=pause.continuation_id,
        controller=AsyncInputController(
            broker=broker,
            actor=suspension.command.actor,
            clock=suspended.clock,
            durable_authority=durable_authority,
            durable_resolver=durable_resolver,
        ),
        broker=broker,
        actor=suspension.command.actor,
        clock=suspended.clock,
        correlation=InteractionCorrelation.from_request(suspended.request),
        resolver_commands=resolver_commands,
        provider_call_count=provider_call_count,
        domain_side_effect_count=len(inspection.artifacts),
        durable=suspended,
    )


async def _open_causal_surface(
    surface_id: str,
    condition_id: str,
    *,
    resolver_strategy: Literal[
        "store",
        "coordinator",
        "coordinator_then_store",
    ] = "store",
    continuation_ttl_seconds: int = 600,
) -> tuple[
    _CausalSurface,
    ConfirmationQuestion | SingleSelectionQuestion,
]:
    """Open one exact surface and return its owned pending request."""
    question = _question_for_condition(condition_id)
    if surface_id in {"sdk-attached", "sdk-headless-predeclared"}:
        return (
            await _open_attached_surface(
                surface_id,
                condition_id,
                question,
                continuation_ttl_seconds=continuation_ttl_seconds,
            ),
            question,
        )
    return (
        await _open_durable_surface(
            surface_id,
            question,
            resolver_strategy=resolver_strategy,
        ),
        question,
    )


async def _run_attached_advisory_surface(
    surface_id: str,
) -> _SurfaceObservation:
    """Run one attached/predeclared invocation through its real timeout."""
    assert surface_id in {"sdk-attached", "sdk-headless-predeclared"}
    harness = await broker_support._harness()
    resumer = broker_support._Resumer()
    question = ConfirmationQuestion(
        question_id=QuestionId("answer"),
        prompt="Continue after the wait?",
        required=False,
        default_value=False,
    )
    template = broker_support._request(
        None,
        run_id=f"INPUT-F-13-{surface_id}",
        resumer=resumer,
        mode=RequirementMode.ADVISORY,
        advisory_wait_seconds=2,
        default_value=False,
        reason=f"Exercise INPUT-F-13 through {surface_id}.",
    )
    template = replace(template, questions=(question,))
    orchestrator = _AttachedAdvisoryOrchestrator(template, resumer)
    runtime = AttachedInteractionRuntime(
        broker=harness.broker,
        actor=broker_support._actor(),
        handler=broker_support._DetachedHandler(),
    )
    policy = (
        PredeclaredInputPolicy(
            answers=(
                TextAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.POLICY,
                    value="not a confirmation",
                ),
            )
        )
        if surface_id == "sdk-headless-predeclared"
        else None
    )
    running = create_task(
        run_agent(
            cast(Any, orchestrator),
            surface_id,
            interaction_runtime=runtime,
            headless_policy=policy,
        )
    )
    try:
        await broker_support._wait_until(
            lambda: 2.0 in harness.clock.wait_calls
        )
        harness.clock.advance(2)
        result = await wait_for(running, timeout=2)
        assert isinstance(result, AgentRunCompleted)
        assert result.to_str() == "continued"
        requested = orchestrator.result
        assert requested is not None
        created = requested.create_result
        assert isinstance(created, CreateInteractionApplied)
        delivery = requested.delivery
        assert isinstance(delivery, InteractionDelivery)
        settled = await broker_support._inspect(
            harness.broker,
            delivery.correlation,
        )
        assert settled.request.state is RequestState.TIMED_OUT
        assert orchestrator.provider_call_count == 2
        assert orchestrator.domain_side_effects == []
        return _SurfaceObservation(
            condition_id="INPUT-F-13",
            surface_id=surface_id,
            transition_from=created.record.request.state,
            transition_to=settled.request.state,
            public_result_id="sdk.completed.v1",
            public_result={
                "kind": result.kind.value,
                "value": result.to_str(),
                "channel": result.channel,
            },
            status_key="result",
            status_value=type(result).__name__,
            provider_call_count=orchestrator.provider_call_count,
            domain_side_effect_count=len(orchestrator.domain_side_effects),
        )
    finally:
        if not running.done():
            running.cancel()
        await harness.broker.aclose()


async def _run_serverless_durable_advisory_surface() -> _SurfaceObservation:
    """Suspend, time out, and resume one real task-bound continuation."""
    suspended = await task_support._durable_failure_harness(
        ConfirmationQuestion(
            question_id=QuestionId("answer"),
            prompt="Continue after the wait?",
            required=False,
            default_value=False,
        ),
        mode=RequirementMode.ADVISORY,
    )
    suspension = suspended.target.suspensions[0]
    correlation = InteractionCorrelation.from_request(suspended.request)
    try:
        assert suspended.suspension.run.state is TaskRunState.INPUT_REQUIRED
        assert suspended.request.mode is RequirementMode.ADVISORY
        assert suspended.request.advisory_wait_seconds is not None
        assert task_support._provider_call_count(suspended.model_factory) == 1
        pending = await suspended.interaction_store.lookup_scoped(
            ScopedInteractionLookup(
                actor=suspension.command.actor,
                correlation=correlation,
            )
        )
        assert isinstance(pending, InteractionRecord)
        presented = await suspended.interaction_store.mark_presented(
            PresentInteractionCommand(
                actor=suspension.command.actor,
                correlation=correlation,
                expected_store_revision=pending.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        deadline = presented.record.request.advisory_deadline
        assert deadline is not None
        suspended.clock.now = deadline + timedelta(microseconds=1)
        suspended.clock.monotonic = (
            float(suspended.request.advisory_wait_seconds) + 0.000001
        )
        timed_out = await suspended.coordinator.timeout_advisory_and_requeue(
            ScopedInteractionLookup(
                actor=suspension.command.actor,
                correlation=correlation,
            ),
            task_run_id=suspended.run_id,
            now=suspended.clock.now,
        )
        assert timed_out.record.request.state is RequestState.TIMED_OUT
        assert timed_out.record.request.request_id == (
            suspended.request.request_id
        )
        assert timed_out.record.request.continuation_id == (
            suspended.request.continuation_id
        )
        assert timed_out.reentry.run.run_id == suspended.run_id
        assert timed_out.reentry.run.state is TaskRunState.QUEUED

        resumed = await task_support._resume_harness(suspended)
        processed = await resumed.worker.process_once()
        assert processed.completion is not None
        assert processed.completion.run.run_id == suspended.run_id
        assert processed.completion.run.state is TaskRunState.SUCCEEDED
        assert len(resumed.executor.commands) == 1
        resume_command = resumed.executor.commands[0]
        assert resume_command.request.request_id == (
            suspended.request.request_id
        )
        assert resume_command.request.continuation_id == (
            suspended.request.continuation_id
        )
        assert resume_command.continuation.origin.run_id == (
            suspended.request.origin.run_id
        )
        assert isinstance(
            resume_command.model_result,
            InputTimedOutResult,
        )
        assert suspended.target.domain_side_effects == ["resumed output"]
        provider_call_count = task_support._provider_call_count(
            suspended.model_factory
        ) + len(resumed.executor.commands)
        assert provider_call_count == 2
        result = AgentRunCompleted(
            value=cast(str, suspended.target.domain_side_effects[0])
        )
        inspection = await suspended.client.inspect(suspended.run_id)
        assert inspection.artifacts == ()
        return _SurfaceObservation(
            condition_id="INPUT-F-13",
            surface_id="sdk-serverless-durable",
            transition_from=pending.request.state,
            transition_to=timed_out.record.request.state,
            public_result_id="sdk.completed.v1",
            public_result={
                "kind": result.kind.value,
                "value": result.to_str(),
                "channel": result.channel,
            },
            status_key="result",
            status_value=type(result).__name__,
            provider_call_count=provider_call_count,
            domain_side_effect_count=len(inspection.artifacts),
        )
    finally:
        await suspended.stack.aclose()
        suspended.temporary.cleanup()


async def _run_durable_advisory_surface(
    surface_id: str,
) -> _SurfaceObservation:
    """Run one durable stager and continue the same invocation on timeout."""
    if surface_id == "sdk-serverless-durable":
        return await _run_serverless_durable_advisory_surface()
    assert surface_id == "sdk-headless-durable"
    database = FakePgsqlDatabase()
    clock = durable_support._Clock()
    store = await durable_support._store(database, clock=clock)
    run_id = f"sdk-f13-{surface_id}"
    base = durable_support._request(run_id)
    request = replace(
        base,
        origin=replace(base.origin, task_id=None),
        mode=RequirementMode.ADVISORY,
        reason=f"Exercise INPUT-F-13 through {surface_id}.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue after the wait?",
                required=False,
                default_value=False,
            ),
        ),
        advisory_wait_seconds=2,
    )
    command = durable_support._create_command(request)
    continuation = durable_support._portable(request)
    suspension = DurableInteractionSuspension(
        command=command,
        continuation=continuation,
    )
    staged = Event()
    stager_calls: list[DurableInteractionSuspension] = []

    async def stager(*args: object, **kwargs: object) -> object:
        del args, kwargs
        created = await store.create_durable(command, continuation)
        assert created.record.request.state is RequestState.PENDING
        presented = await store.mark_presented(
            PresentInteractionCommand(
                actor=command.actor,
                correlation=created.record.correlation,
                expected_store_revision=created.record.store_revision,
            )
        )
        assert isinstance(presented, InteractionPresentationApplied)
        stager_calls.append(suspension)
        return suspension

    runtime = DurableInteractionRuntime(
        actor=command.actor,
        stager=cast(Any, stager),
        run_id=request.origin.run_id,
    )
    correlation = InteractionCorrelation.from_request(request)
    orchestrator = _DurableAdvisoryOrchestrator(
        store,
        clock,
        staged,
        correlation,
        command.actor,
    )
    running = create_task(
        run_agent(
            cast(Any, orchestrator),
            surface_id,
            interaction_runtime=runtime,
        )
    )
    try:
        await wait_for(staged.wait(), timeout=2)
        presented = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=command.actor,
                correlation=correlation,
            )
        )
        assert isinstance(presented, InteractionRecord)
        assert presented.request.advisory_deadline is not None
        clock.now = presented.request.advisory_deadline + timedelta(
            microseconds=1
        )
        clock.monotonic = 2.000001
        clock.changed.set()
        result = await wait_for(running, timeout=2)
        assert isinstance(result, AgentRunCompleted)
        assert result.to_str() == "continued"
        assert stager_calls == [suspension]
        assert orchestrator.record is not None
        assert orchestrator.record.request.state is RequestState.TIMED_OUT
        assert orchestrator.provider_call_count == 2
        assert orchestrator.domain_side_effects == []
        assert runtime.run_id is not None and runtime.task_id is None
        return _SurfaceObservation(
            condition_id="INPUT-F-13",
            surface_id=surface_id,
            transition_from=presented.request.state,
            transition_to=orchestrator.record.request.state,
            public_result_id="sdk.completed.v1",
            public_result={
                "kind": result.kind.value,
                "value": result.to_str(),
                "channel": result.channel,
            },
            status_key="result",
            status_value=type(result).__name__,
            provider_call_count=orchestrator.provider_call_count,
            domain_side_effect_count=len(orchestrator.domain_side_effects),
        )
    finally:
        if not running.done():
            running.cancel()
        await store.aclose()


async def _resolve_causal_surface(
    surface: _CausalSurface,
    submission: InputAnswerSubmission,
    *,
    key: ResolutionIdempotencyKey,
) -> InputResolutionAccepted | InteractionResolutionResult:
    """Resolve through the owning in-band or durable public boundary."""
    if surface.controller is not None:
        assert surface.request_ref is not None
        assert surface.continuation_ref is not None
        return await resolve_input(
            surface.controller,
            surface.request_ref,
            surface.continuation_ref,
            submission,
            idempotency_key=key,
        )
    record = await surface.record()
    return (
        await surface.broker.resolve(
            ResolveInteractionCommand(
                actor=surface.actor,
                correlation=surface.correlation,
                expected_state_revision=record.request.state_revision,
                idempotency_key=key,
                proposed_resolution=AnsweredResolution(
                    request_id=record.request.request_id,
                    provenance=submission.provenance,
                    resolved_at=record.request.created_at,
                    answers=submission.answers,
                ),
            )
        )
    ).store_result


def _record(
    record_property: Callable[[str, object], None],
    observations: tuple[_SurfaceObservation, ...],
) -> None:
    assert observations
    keys = {
        (observation.condition_id, observation.surface_id)
        for observation in observations
    }
    assert len(keys) == len(observations)
    record_property(
        _EVIDENCE_PROPERTY,
        [observation.evidence() for observation in observations],
    )


class _CompletedResponse:
    async def to_str(self) -> str:
        return "continued"


class _DurablePauseOrchestrator:
    """Stage one exact durable suspension before exposing its public pause."""

    def __init__(self) -> None:
        self.provider_call_count = 0
        self.staged: list[DurableInteractionSuspension] = []
        self.domain_side_effects: list[object] = []

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, DurableInteractionRuntime)
        self.provider_call_count += 1
        suspension = await cast(Any, runtime.stager)(object())
        assert isinstance(suspension, DurableInteractionSuspension)
        self.staged.append(suspension)
        raise ExecutionInputRequiredError(
            InputRequiredResult(
                request_id=suspension.command.request.request_id,
                continuation_id=suspension.command.request.continuation_id,
                detached_resumption_available=True,
            ),
            durable=suspension,
        )


async def _durable_pause_for_surface(
    surface_id: str,
) -> _SurfaceObservation:
    """Run one distinct run-bound or task-bound durable SDK admission."""
    assert surface_id in {
        "sdk-headless-durable",
        "sdk-serverless-durable",
    }
    suspension = sdk_support._suspension()
    broker = sdk_support._Broker(sdk_support._pending_record())
    order: list[str] = []
    stager_calls: list[object] = []
    handed_off_requests: list[Any] = []

    async def stager(*args: object, **kwargs: object) -> object:
        assert len(args) == 1
        assert not kwargs
        stager_calls.append(args[0])
        order.append("stage")
        return suspension

    async def handoff(received: object) -> object:
        assert received == suspension
        order.append("persist")
        broker.record = sdk_support._pending_record()
        handed_off_requests.append(broker.record.request)
        return broker.record.request

    async def waiter(seconds: int) -> None:
        assert seconds == 1
        order.append("wait")

    orchestrator = _DurablePauseOrchestrator()
    runtime = DurableInteractionRuntime(
        actor=InteractionActor(principal=PrincipalScope()),
        stager=cast(Any, stager),
        run_id=(
            RunId("headless-durable-run")
            if surface_id == "sdk-headless-durable"
            else None
        ),
        task_id=(
            TaskId("serverless-durable-task")
            if surface_id == "sdk-serverless-durable"
            else None
        ),
    )
    result = await run_agent(
        cast(Any, orchestrator),
        surface_id,
        interaction_runtime=runtime,
        headless_policy=DurableHandoffInputPolicy(
            handoff=cast(Any, handoff),
            durable_handoff_wait_seconds=1,
            waiter=waiter,
        ),
    )
    assert isinstance(result, AgentRunInputRequired)
    assert len(stager_calls) == 1
    assert orchestrator.staged == [suspension]
    assert order == ["stage", "persist", "wait"]
    assert len(handed_off_requests) == 1
    handed_off = handed_off_requests[0]
    assert handed_off.request_id == suspension.command.request.request_id
    assert handed_off.continuation_id == (
        suspension.command.request.continuation_id
    )
    assert result.request.state is handed_off.state
    assert result.request_id is not None
    assert result.continuation_id is not None
    assert orchestrator.provider_call_count == 1
    assert orchestrator.domain_side_effects == []
    if surface_id == "sdk-headless-durable":
        assert runtime.run_id is not None and runtime.task_id is None
    else:
        assert runtime.task_id is not None and runtime.run_id is None
    return _SurfaceObservation(
        condition_id="INPUT-F-12",
        surface_id=surface_id,
        transition_from=handed_off.state,
        transition_to=result.request.state,
        public_result_id="sdk.input_required.v1",
        public_result={
            "kind": result.kind.value,
            "request_id": result.request_id,
            "continuation_id": result.continuation_id,
            "detached_resumption_available": (
                result.detached_resumption_available
            ),
            "channel": result.channel,
        },
        status_key="result",
        status_value=type(result).__name__,
        provider_call_count=orchestrator.provider_call_count,
        domain_side_effect_count=len(orchestrator.domain_side_effects),
    )


class _UnavailableOrchestrator:
    """Drive a configured SDK surface into one real unavailable state."""

    def __init__(
        self,
        harness: broker_support._Harness,
        surface_id: str,
    ) -> None:
        self.harness = harness
        self.surface_id = surface_id
        self.request_result: InteractionRequestResult | None = None
        self.staged: list[DurableInteractionSuspension] = []
        self.provider_calls: list[object] = []
        self.domain_side_effects: list[object] = []

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input
        runtime = kwargs["interaction_runtime"]
        if self.surface_id in {
            "sdk-attached",
            "sdk-headless-predeclared",
        }:
            assert isinstance(runtime, AttachedInteractionRuntime)
        elif self.surface_id in {
            "sdk-headless-durable",
            "sdk-serverless-durable",
        }:
            assert isinstance(runtime, DurableInteractionRuntime)
            if self.surface_id == "sdk-headless-durable":
                assert runtime.run_id is not None and runtime.task_id is None
            else:
                assert runtime.task_id is not None and runtime.run_id is None
            suspension = await cast(Any, runtime.stager)(object())
            assert isinstance(suspension, DurableInteractionSuspension)
            self.staged.append(suspension)
            raise ExecutionInputRequiredError(
                InputRequiredResult(
                    request_id=suspension.command.request.request_id,
                    continuation_id=(
                        suspension.command.request.continuation_id
                    ),
                    detached_resumption_available=True,
                ),
                durable=suspension,
            )
        else:
            assert self.surface_id == "sdk-sessionless"
            assert runtime is None
        broker = (
            runtime.broker
            if isinstance(runtime, AttachedInteractionRuntime)
            else self.harness.broker
        )
        self.request_result = await broker.request(
            broker_support._request(
                None,
                run_id=f"sdk-f01-{self.surface_id}",
                reason=(
                    f"No input-capable host is available on {self.surface_id}."
                ),
            )
        )
        delivery = self.request_result.delivery
        assert delivery is not None
        assert delivery.record.request.state is RequestState.UNAVAILABLE
        resolution = delivery.record.request.resolution
        assert resolution is not None
        assert resolution.status is ResolutionStatus.UNAVAILABLE
        raise InputValidationError(
            InputErrorCode.UNAVAILABLE,
            "interaction",
            "input capability is unavailable",
        )


def test_input_f_01(
    record_property: Callable[[str, object], None],
) -> None:
    """Report unavailable input without invoking a provider or side effect."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _ALL_SDK_SURFACES:
            harness = await broker_support._harness()
            try:
                suspension = sdk_support._suspension()
                unavailable_requests: list[InteractionRequestResult] = []
                stager_calls: list[object] = []
                handoff_calls: list[DurableInteractionSuspension] = []

                async def stager(*args: object, **kwargs: object) -> object:
                    assert len(args) == 1
                    assert not kwargs
                    stager_calls.append(args[0])
                    requested = await harness.broker.request(
                        broker_support._request(
                            None,
                            run_id=f"sdk-f01-{surface_id}",
                            reason=(
                                "No input-capable host is available on "
                                f"{surface_id}."
                            ),
                        )
                    )
                    delivery = requested.delivery
                    assert isinstance(delivery, InteractionDelivery)
                    assert (
                        delivery.record.request.state
                        is RequestState.UNAVAILABLE
                    )
                    unavailable_requests.append(requested)
                    return suspension

                async def handoff(received: object) -> object:
                    assert received == suspension
                    handoff_calls.append(suspension)
                    assert len(unavailable_requests) == 1
                    raise InputValidationError(
                        InputErrorCode.UNAVAILABLE,
                        "interaction",
                        "input capability is unavailable",
                    )

                runtime: object | None
                policy: object | None = None
                if surface_id in {
                    "sdk-attached",
                    "sdk-headless-predeclared",
                }:
                    runtime = AttachedInteractionRuntime(
                        broker=harness.broker,
                        actor=broker_support._actor(),
                        handler=broker_support._DetachedHandler(),
                    )
                    if surface_id == "sdk-headless-predeclared":
                        policy = PredeclaredInputPolicy(
                            answers=(
                                sdk_support._answer(AnswerProvenance.POLICY),
                            )
                        )
                elif surface_id in {
                    "sdk-headless-durable",
                    "sdk-serverless-durable",
                }:
                    runtime = DurableInteractionRuntime(
                        actor=broker_support._actor(),
                        stager=cast(Any, stager),
                        run_id=(
                            RunId("sdk-f01-headless-run")
                            if surface_id == "sdk-headless-durable"
                            else None
                        ),
                        task_id=(
                            TaskId("sdk-f01-serverless-task")
                            if surface_id == "sdk-serverless-durable"
                            else None
                        ),
                    )
                    policy = DurableHandoffInputPolicy(
                        handoff=cast(Any, handoff),
                    )
                else:
                    assert surface_id == "sdk-sessionless"
                    runtime = None
                orchestrator = _UnavailableOrchestrator(
                    harness,
                    surface_id,
                )
                result = await run_agent(
                    cast(Any, orchestrator),
                    surface_id,
                    interaction_runtime=cast(Any, runtime),
                    headless_policy=cast(Any, policy),
                )
                assert isinstance(result, AgentRunFailed)
                assert result.code == "input.unavailable"
                assert result.message == "input capability is unavailable"
                assert not result.retryable
                if surface_id in {
                    "sdk-headless-durable",
                    "sdk-serverless-durable",
                }:
                    assert len(stager_calls) == 1
                    assert orchestrator.staged == [suspension]
                    assert handoff_calls == [suspension]
                    assert len(unavailable_requests) == 1
                    requested = unavailable_requests[0]
                else:
                    assert stager_calls == []
                    assert orchestrator.staged == []
                    assert handoff_calls == []
                    assert unavailable_requests == []
                    requested = orchestrator.request_result
                    assert requested is not None
                created = requested.create_result
                assert isinstance(created, CreateInteractionApplied)
                delivery = requested.delivery
                assert isinstance(delivery, InteractionDelivery)
                assert created.command.request.state is RequestState.CREATED
                assert (
                    delivery.record.request.state is RequestState.UNAVAILABLE
                )
                assert orchestrator.provider_calls == []
                assert orchestrator.domain_side_effects == []
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-01",
                        surface_id=surface_id,
                        transition_from=created.command.request.state,
                        transition_to=delivery.record.request.state,
                        public_result_id="sdk.failed.v1",
                        public_result={
                            "kind": result.kind.value,
                            "code": result.code,
                            "message": result.message,
                            "retryable": result.retryable,
                            "channel": result.channel,
                        },
                        status_key="result",
                        status_value=type(result).__name__,
                        provider_call_count=len(orchestrator.provider_calls),
                        domain_side_effect_count=len(
                            orchestrator.domain_side_effects
                        ),
                    )
                )
            finally:
                await harness.broker.aclose()
        return tuple(observations)

    _record(record_property, run(exercise()))


def _exercise_sdk_validation_rejections(
    condition_id: str,
    code: InputErrorCode,
) -> tuple[_SurfaceObservation, ...]:
    """Exercise validation through each request-owning SDK boundary."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, question = await _open_causal_surface(
                surface_id,
                condition_id,
            )
            try:
                before = await surface.record()
                assert before.request.state is RequestState.PENDING
                submission = _submission_for_condition(
                    question,
                    condition_id,
                )
                if surface.controller is None:
                    rejected = await _resolve_causal_surface(
                        surface,
                        submission,
                        key=ResolutionIdempotencyKey(
                            f"{condition_id}-{surface_id}"
                        ),
                    )
                    assert isinstance(rejected, ResolveInteractionRejected)
                    assert rejected.error.code is code
                    assert not rejected.store_mutation_applied
                    actual_error = rejected.error
                    status_key = "broker_result"
                    status_value = type(rejected).__name__
                else:
                    with pytest.raises(InputValidationError) as rejected_error:
                        await _resolve_causal_surface(
                            surface,
                            submission,
                            key=ResolutionIdempotencyKey(
                                f"{condition_id}-{surface_id}"
                            ),
                        )
                    assert rejected_error.value.code is code
                    assert len(surface.resolver_commands) == 1
                    actual_error = rejected_error.value
                    status_key = "exception"
                    status_value = type(rejected_error.value).__name__
                assert actual_error.code is code
                after = await surface.record()
                assert after.request.state is RequestState.PENDING
                observations.append(
                    _SurfaceObservation(
                        condition_id=condition_id,
                        surface_id=surface.surface_id,
                        transition_from=before.request.state,
                        transition_to=after.request.state,
                        public_result_id="sdk.validation_error.v1",
                        public_result={
                            "error": "input.validation",
                            "interaction_state": after.request.state.value,
                            "channel": "typed",
                        },
                        status_key=status_key,
                        status_value=status_value,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=(
                            surface.domain_side_effect_count
                        ),
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    return run(exercise())


def test_input_f_04(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a mismatched semantic answer type."""
    observations = _exercise_sdk_validation_rejections(
        "INPUT-F-04",
        InputErrorCode.ANSWER_TYPE_MISMATCH,
    )
    _record(record_property, observations)


def test_input_f_05(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject an unknown selection choice."""
    observations = _exercise_sdk_validation_rejections(
        "INPUT-F-05",
        InputErrorCode.UNKNOWN_CHOICE,
    )
    _record(record_property, observations)


def test_input_f_06(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject omission of a required answer."""
    observations = _exercise_sdk_validation_rejections(
        "INPUT-F-06",
        InputErrorCode.MISSING_REQUIRED_ANSWER,
    )
    _record(record_property, observations)


def test_input_f_07(
    record_property: Callable[[str, object], None],
) -> None:
    """Accept a same-key resolution replay idempotently."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, question = await _open_causal_surface(
                surface_id,
                "INPUT-F-07",
                resolver_strategy="coordinator",
            )
            try:
                submission = _submission_for_condition(
                    question,
                    "INPUT-F-07",
                )
                key = ResolutionIdempotencyKey(f"same-key-{surface_id}")
                first = await _resolve_causal_surface(
                    surface,
                    submission,
                    key=key,
                )
                before_replay = await surface.record()
                assert before_replay.request.state is RequestState.ANSWERED
                replay = await _resolve_causal_surface(
                    surface,
                    submission,
                    key=key,
                )
                if surface.controller is None:
                    assert isinstance(first, ResolveInteractionApplied)
                    assert isinstance(replay, InteractionStoreReplayed)
                    public_result = InputResolutionAccepted(
                        interaction_state=cast(
                            Any,
                            replay.record.request.state.value,
                        ),
                        idempotent=True,
                    )
                    status_key = "broker_result"
                else:
                    assert isinstance(first, InputResolutionAccepted)
                    assert isinstance(replay, InputResolutionAccepted)
                    assert not first.idempotent
                    assert replay.idempotent
                    assert len(surface.resolver_commands) == 2
                    public_result = replay
                    status_key = "result"
                after_replay = await surface.record()
                assert after_replay.request.state is RequestState.ANSWERED
                assert (
                    public_result.interaction_state
                    == after_replay.request.state.value
                )
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-07",
                        surface_id=surface.surface_id,
                        transition_from=before_replay.request.state,
                        transition_to=after_replay.request.state,
                        public_result_id="sdk.resolution_accepted.v1",
                        public_result={
                            "kind": public_result.kind,
                            "interaction_state": (
                                public_result.interaction_state
                            ),
                            "idempotent": public_result.idempotent,
                            "channel": public_result.channel,
                        },
                        status_key=status_key,
                        status_value=type(replay).__name__,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=(
                            surface.domain_side_effect_count
                        ),
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_08(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a conflicting second resolution with a public typed error."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, question = await _open_causal_surface(
                surface_id,
                "INPUT-F-08",
                resolver_strategy="coordinator_then_store",
            )
            try:
                key = ResolutionIdempotencyKey(f"conflict-{surface_id}")
                accepted = await _resolve_causal_surface(
                    surface,
                    _submission_for_condition(
                        question,
                        "INPUT-F-08",
                        value=True,
                    ),
                    key=key,
                )
                before_conflict = await surface.record()
                assert before_conflict.request.state is RequestState.ANSWERED
                if surface.controller is None:
                    assert isinstance(accepted, ResolveInteractionApplied)
                    conflict = await _resolve_causal_surface(
                        surface,
                        _submission_for_condition(
                            question,
                            "INPUT-F-08",
                            value=False,
                        ),
                        key=key,
                    )
                    assert isinstance(
                        conflict,
                        ResolveInteractionRejected,
                    )
                    assert conflict.error.code is (
                        InputErrorCode.IDEMPOTENCY_CONFLICT
                    )
                    public_error = InputAlreadyResolvedError()
                    status_key = "broker_result"
                else:
                    assert isinstance(accepted, InputResolutionAccepted)
                    with pytest.raises(
                        InputAlreadyResolvedError
                    ) as conflict_error:
                        await _resolve_causal_surface(
                            surface,
                            _submission_for_condition(
                                question,
                                "INPUT-F-08",
                                value=False,
                            ),
                            key=key,
                        )
                    assert len(surface.resolver_commands) == 2
                    public_error = conflict_error.value
                    conflict = conflict_error.value
                    status_key = "exception"
                after_conflict = await surface.record()
                assert after_conflict.request.state is RequestState.ANSWERED
                assert public_error.code is InputErrorCode.ALREADY_RESOLVED
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-08",
                        surface_id=surface.surface_id,
                        transition_from=before_conflict.request.state,
                        transition_to=after_conflict.request.state,
                        public_result_id="sdk.already_resolved.v1",
                        public_result={
                            "error": public_error.code.value,
                            "interaction_state": (
                                after_conflict.request.state.value
                            ),
                            "channel": "typed",
                        },
                        status_key=status_key,
                        status_value=type(conflict).__name__,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=(
                            surface.domain_side_effect_count
                        ),
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_09(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject resolution after absolute continuation expiry."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, question = await _open_causal_surface(
                surface_id,
                "INPUT-F-09",
                continuation_ttl_seconds=60,
            )
            try:
                before_expiry = await surface.record()
                assert before_expiry.request.state is RequestState.PENDING
                if surface.durable is None:
                    await broker_support._wait_until(
                        lambda: 60.0 in surface.clock.wait_calls
                    )
                    surface.clock.advance(60)
                    assert surface.resumer is not None
                    await surface.resumer.called.wait()
                else:
                    request = surface.durable.request
                    deadline = request.created_at + timedelta(
                        seconds=request.continuation_ttl_seconds
                    )
                    surface.clock.now = deadline
                    surface.clock.monotonic = float(
                        request.continuation_ttl_seconds
                    )
                    lifecycle = (
                        await (
                            surface.durable.coordinator.expire_suspended_task(
                                TerminalizeDueInteractionsCommand(),
                                task_run_id=surface.durable.run_id,
                                now=surface.clock.now,
                            )
                        )
                    )
                    assert (
                        lifecycle.completion_for(surface.durable.run_id)
                        is not None
                    )
                record = await surface.record()
                assert record.request.state is RequestState.EXPIRED
                submission = _submission_for_condition(
                    question,
                    "INPUT-F-09",
                )
                if surface.controller is None:
                    rejected = await _resolve_causal_surface(
                        surface,
                        submission,
                        key=ResolutionIdempotencyKey(f"expired-{surface_id}"),
                    )
                    assert isinstance(rejected, ResolveInteractionRejected)
                    assert not rejected.store_mutation_applied
                    assert (
                        rejected.error.code is InputErrorCode.ALREADY_RESOLVED
                    )
                    public_error = InputExpiredError()
                    status_key = "broker_result"
                    resolution_outcome: object = rejected
                else:
                    with pytest.raises(InputExpiredError) as expired_error:
                        await _resolve_causal_surface(
                            surface,
                            submission,
                            key=ResolutionIdempotencyKey(
                                f"expired-{surface_id}"
                            ),
                        )
                    assert surface.resolver_commands == []
                    public_error = expired_error.value
                    status_key = "exception"
                    resolution_outcome = expired_error.value
                assert public_error.code is InputErrorCode.EXPIRED
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-09",
                        surface_id=surface.surface_id,
                        transition_from=before_expiry.request.state,
                        transition_to=record.request.state,
                        public_result_id="sdk.expired.v1",
                        public_result={
                            "error": public_error.code.value,
                            "interaction_state": record.request.state.value,
                            "channel": "typed",
                        },
                        status_key=status_key,
                        status_value=type(resolution_outcome).__name__,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=(
                            surface.domain_side_effect_count
                        ),
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_10(
    record_property: Callable[[str, object], None],
) -> None:
    """Cancel the exact pending request through its owning host boundary."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, _question = await _open_causal_surface(
                surface_id,
                "INPUT-F-10",
            )
            try:
                pending = await surface.record()
                assert pending.request.state is RequestState.PENDING
                if surface.durable is None:
                    cancelled = await surface.broker.cancel(
                        CancelInteractionCommand(
                            actor=surface.actor,
                            correlation=surface.correlation,
                            provenance=AnswerProvenance.HUMAN,
                            expected_state_revision=(
                                pending.request.state_revision
                            ),
                        )
                    )
                    assert isinstance(
                        cancelled.store_result,
                        CancelInteractionApplied,
                    )
                    status_key = "broker_result"
                    status_value = type(cancelled.store_result).__name__
                    domain_side_effect_count = surface.domain_side_effect_count
                else:
                    cancelled_run = await surface.durable.client.cancel(
                        surface.durable.run_id
                    )
                    assert cancelled_run.state is TaskRunState.CANCELLED
                    inspection = await surface.durable.client.inspect(
                        surface.durable.run_id
                    )
                    assert inspection.run.state is TaskRunState.CANCELLED
                    assert surface.durable.target.resume_contexts == []
                    assert surface.durable.target.domain_side_effects == []
                    assert inspection.artifacts == ()
                    status_key = "task_state"
                    status_value = (
                        f"{type(cancelled_run.state).__name__}."
                        f"{cancelled_run.state.name}"
                    )
                    domain_side_effect_count = len(inspection.artifacts)
                record = await surface.record()
                assert record.request.state is RequestState.CANCELLED
                public_result = AgentRunCancelled()
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-10",
                        surface_id=surface.surface_id,
                        transition_from=pending.request.state,
                        transition_to=record.request.state,
                        public_result_id="sdk.cancelled.v1",
                        public_result={
                            "kind": public_result.kind.value,
                            "channel": public_result.channel,
                        },
                        status_key=status_key,
                        status_value=status_value,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=domain_side_effect_count,
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_11(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject resolution of a superseded originating turn."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            surface, question = await _open_causal_surface(
                surface_id,
                "INPUT-F-11",
            )
            try:
                before_supersession = await surface.record()
                assert (
                    before_supersession.request.state is RequestState.PENDING
                )
                command = SupersedeInteractionScopeCommand(
                    actor=surface.actor,
                    scope=InteractionExecutionScope(
                        run_id=surface.correlation.run_id,
                    ),
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                )
                if surface.durable is None:
                    superseded = await surface.broker.supersede(command)
                    assert isinstance(
                        superseded.store_result,
                        ScopeSupersessionApplied,
                    )
                else:
                    coordinator = surface.durable.coordinator
                    lifecycle = await coordinator.supersede_suspended_task(
                        command,
                        task_run_id=surface.durable.run_id,
                        now=surface.clock.now,
                    )
                    completion = lifecycle.completion_for(
                        surface.durable.run_id
                    )
                    assert completion is not None
                    assert completion.run.state is TaskRunState.CANCELLED
                    inspection = await surface.durable.client.inspect(
                        surface.durable.run_id
                    )
                    assert inspection.artifacts == ()
                    domain_side_effect_count = len(inspection.artifacts)
                record = await surface.record()
                assert record.request.state is RequestState.SUPERSEDED
                submission = _submission_for_condition(
                    question,
                    "INPUT-F-11",
                )
                if surface.controller is None:
                    rejected = await _resolve_causal_surface(
                        surface,
                        submission,
                        key=ResolutionIdempotencyKey(
                            f"superseded-{surface_id}"
                        ),
                    )
                    assert isinstance(rejected, ResolveInteractionRejected)
                    assert not rejected.store_mutation_applied
                    assert (
                        rejected.error.code is InputErrorCode.ALREADY_RESOLVED
                    )
                    public_error = InputSupersededError()
                    status_key = "broker_result"
                    resolution_outcome: object = rejected
                    domain_side_effect_count = surface.domain_side_effect_count
                else:
                    with pytest.raises(
                        InputSupersededError
                    ) as superseded_error:
                        await _resolve_causal_surface(
                            surface,
                            submission,
                            key=ResolutionIdempotencyKey(
                                f"superseded-{surface_id}"
                            ),
                        )
                    assert surface.resolver_commands == []
                    public_error = superseded_error.value
                    status_key = "exception"
                    resolution_outcome = superseded_error.value
                assert public_error.code is InputErrorCode.SUPERSEDED
                observations.append(
                    _SurfaceObservation(
                        condition_id="INPUT-F-11",
                        surface_id=surface.surface_id,
                        transition_from=(before_supersession.request.state),
                        transition_to=record.request.state,
                        public_result_id="sdk.superseded.v1",
                        public_result={
                            "error": public_error.code.value,
                            "interaction_state": record.request.state.value,
                            "channel": "typed",
                        },
                        status_key=status_key,
                        status_value=type(resolution_outcome).__name__,
                        provider_call_count=(surface.provider_call_count),
                        domain_side_effect_count=domain_side_effect_count,
                    )
                )
            finally:
                await surface.close()
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_12(
    record_property: Callable[[str, object], None],
) -> None:
    """Return durable opaque correlation after bounded required waiting."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in (
            "sdk-headless-durable",
            "sdk-serverless-durable",
        ):
            observations.append(await _durable_pause_for_surface(surface_id))
        return tuple(observations)

    _record(record_property, run(exercise()))


def test_input_f_13(
    record_property: Callable[[str, object], None],
) -> None:
    """Continue advisory input after timeout without an invented answer."""

    async def exercise() -> tuple[_SurfaceObservation, ...]:
        observations: list[_SurfaceObservation] = []
        for surface_id in _FOUR_SDK_SURFACES:
            if surface_id in {
                "sdk-attached",
                "sdk-headless-predeclared",
            }:
                observation = await _run_attached_advisory_surface(surface_id)
            else:
                observation = await _run_durable_advisory_surface(surface_id)
            observations.append(observation)
        return tuple(observations)

    observations = run(exercise())
    values = {
        observation.surface_id: observation.public_result["value"]
        for observation in observations
    }
    assert values == {
        "sdk-attached": "continued",
        "sdk-headless-predeclared": "continued",
        "sdk-headless-durable": "continued",
        "sdk-serverless-durable": "resumed output",
    }
    assert values["sdk-serverless-durable"] != values["sdk-attached"]
    assert all(
        observation.provider_call_count == 2
        and observation.domain_side_effect_count == 0
        for observation in observations
    )
    _record(record_property, observations)
