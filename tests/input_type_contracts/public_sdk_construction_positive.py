"""Construct every public input integration from the avalan root."""

from dataclasses import dataclass

from avalan import (
    AgentRunResult,
    AnswerProvenance,
    AttachedInputCancellationHandler,
    AttachedInputContext,
    AttachedInputDetached,
    AttachedInputHandler,
    AttachedInputOutcome,
    DurableInputBridge,
    DurableInputPersistenceAccepted,
    DurableInputPersistenceRequest,
    InputAnswer,
    InputAnswerSubmission,
    InputContinuationRef,
    InputDeclineSubmission,
    InputInspection,
    InputInspectionRequest,
    InputPolicyValueProvider,
    InputQuestion,
    InputRequestRef,
    InputResolutionAccepted,
    InputResolutionRequest,
    InputResolutionResult,
    Message,
    Orchestrator,
    ResolutionIdempotencyKey,
    StateRevision,
    create_attached_input_runtime,
    create_decline_input_policy,
    create_durable_input_integration,
    create_external_controller_input_policy,
    create_policy_value_input_policy,
    create_predeclared_input_policy,
    create_trusted_default_input_policy,
    create_unavailable_input_policy,
    run_agent,
)


async def detach(context: AttachedInputContext) -> AttachedInputOutcome:
    """Detach one semantic request."""
    _ = context
    return AttachedInputDetached()


async def cancelled(context: AttachedInputContext) -> None:
    """Observe cancellation without blocking."""
    _ = context


@dataclass
class PolicyValues:
    """Return policy-owned values through the public provider protocol."""

    answers: tuple[InputAnswer, ...]

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> tuple[InputAnswer, ...]:
        """Return the configured values."""
        _ = context
        return self.answers


class AtomicBridge:
    """Persist and resolve exact opaque durable requests."""

    def __init__(self) -> None:
        self.persisted: DurableInputPersistenceRequest | None = None

    async def persist_input(
        self,
        request: DurableInputPersistenceRequest,
    ) -> DurableInputPersistenceAccepted:
        """Atomically retain one serialized suspension."""
        questions: tuple[InputQuestion, ...] = request.request.questions
        state_revision: StateRevision = request.request.state_revision
        assert questions == request.request.questions
        assert state_revision == request.request.state_revision
        self.persisted = request
        return DurableInputPersistenceAccepted(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            persistence_digest=request.persistence_digest,
        )

    async def inspect_input(
        self,
        request: InputInspectionRequest,
    ) -> InputInspection:
        """Inspect only the exact retained opaque references."""
        persisted = self.persisted
        assert persisted is not None
        assert request.request_id == persisted.request_id
        assert request.continuation_id == persisted.continuation_id
        return InputInspection(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            request=persisted.request,
            detached_resumption_available=True,
        )

    async def resolve_input(
        self,
        request: InputResolutionRequest,
    ) -> InputResolutionResult:
        """Atomically acknowledge one typed resolution and requeue."""
        if isinstance(request.submission, InputAnswerSubmission):
            accepted = InputResolutionAccepted(
                interaction_state="answered",
                idempotent=False,
            )
        else:
            accepted = InputResolutionAccepted(
                interaction_state="declined",
                idempotent=False,
            )
        return InputResolutionResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            resolution=accepted,
        )


HANDLER: AttachedInputHandler = detach
CANCELLATION: AttachedInputCancellationHandler = cancelled
BRIDGE: DurableInputBridge = AtomicBridge()


async def construct_and_drive(
    orchestrator: Orchestrator,
    message: Message,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
    answers: tuple[InputAnswer, ...],
) -> tuple[AgentRunResult[object], AgentRunResult[object]]:
    """Construct attached and durable integrations without internal imports."""
    provider: InputPolicyValueProvider = PolicyValues(answers)
    attached = await create_attached_input_runtime(HANDLER)
    predeclared = create_predeclared_input_policy(
        answers,
        cancellation_handler=CANCELLATION,
    )
    create_policy_value_input_policy(
        provider,
        cancellation_handler=CANCELLATION,
    )
    create_external_controller_input_policy(
        HANDLER,
        cancellation_handler=CANCELLATION,
    )
    create_trusted_default_input_policy(cancellation_handler=CANCELLATION)
    create_decline_input_policy(cancellation_handler=CANCELLATION)
    create_unavailable_input_policy(cancellation_handler=CANCELLATION)
    attached_result = await run_agent(
        orchestrator,
        message,
        interaction_runtime=attached,
        headless_policy=predeclared,
    )

    durable = create_durable_input_integration(
        BRIDGE,
        handoff_wait_seconds=1,
    )
    await durable.controller.inspect_input(request_id, continuation_id)
    await durable.controller.resolve_input(
        request_id,
        continuation_id,
        InputDeclineSubmission(
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER
        ),
        idempotency_key=ResolutionIdempotencyKey("resolve-once"),
    )
    durable_result = await run_agent(
        orchestrator,
        "durable",
        interaction_runtime=durable.runtime,
        headless_policy=durable.headless_policy,
    )
    await attached.aclose()
    await durable.runtime.aclose()
    return attached_result, durable_result
