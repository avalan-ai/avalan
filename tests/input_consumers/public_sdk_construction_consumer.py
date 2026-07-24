"""Drive attached and durable input using only the avalan root API."""

from avalan import (
    AgentRunCompleted,
    AgentRunInputRequired,
    AnswerProvenance,
    AttachedInputContext,
    AttachedInputDetached,
    AttachedInputOutcome,
    DurableInputPersistenceAccepted,
    DurableInputPersistenceRequest,
    InputDeclineSubmission,
    InputInspection,
    InputInspectionRequest,
    InputResolutionAccepted,
    InputResolutionRequest,
    InputResolutionResult,
    Orchestrator,
    ResolutionIdempotencyKey,
    create_attached_input_runtime,
    create_durable_input_integration,
    create_external_controller_input_policy,
    run_agent,
)


async def detach(context: AttachedInputContext) -> AttachedInputOutcome:
    """Detach one public attached request."""
    assert context.validation_error is None
    return AttachedInputDetached()


class AtomicDurableBridge:
    """Act as the atomic persistence and requeue authority."""

    def __init__(self) -> None:
        self.persisted: DurableInputPersistenceRequest | None = None
        self.resolve_calls: list[InputResolutionRequest] = []

    async def persist_input(
        self,
        request: DurableInputPersistenceRequest,
    ) -> DurableInputPersistenceAccepted:
        """Atomically retain the exact serialized suspension."""
        assert self.persisted is None
        assert request.request_payload
        assert request.continuation_payload
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
        """Return only the exact persisted suspension."""
        persisted = self.persisted
        assert persisted is not None
        assert request.request_id == persisted.request_id
        assert request.continuation_id == persisted.continuation_id
        return InputInspection(
            request_id=persisted.request_id,
            continuation_id=persisted.continuation_id,
            request=persisted.request,
            detached_resumption_available=True,
        )

    async def resolve_input(
        self,
        request: InputResolutionRequest,
    ) -> InputResolutionResult:
        """Atomically resolve and requeue the exact continuation."""
        self.resolve_calls.append(request)
        return InputResolutionResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            resolution=InputResolutionAccepted(
                interaction_state="declined",
                idempotent=False,
            ),
        )


async def drive_attached(
    orchestrator: Orchestrator,
) -> AgentRunCompleted[object]:
    """Construct and drive an attached runtime and public controller policy."""
    runtime = await create_attached_input_runtime(detach)
    policy = create_external_controller_input_policy(detach)
    try:
        result = await run_agent(
            orchestrator,
            "attached",
            interaction_runtime=runtime,
            headless_policy=policy,
        )
    finally:
        await runtime.aclose()
    assert isinstance(result, AgentRunCompleted)
    assert isinstance(result.value, str)
    return result


async def drive_durable(
    orchestrator: Orchestrator,
    bridge: AtomicDurableBridge,
) -> AgentRunInputRequired:
    """Construct and drive a durable runtime, handoff, and controller."""
    integration = create_durable_input_integration(
        bridge,
        handoff_wait_seconds=1,
    )
    try:
        result = await run_agent(
            orchestrator,
            "durable",
            interaction_runtime=integration.runtime,
            headless_policy=integration.headless_policy,
        )
        assert isinstance(result, AgentRunInputRequired)
        assert result.request_id is not None
        assert result.continuation_id is not None
        await integration.controller.inspect_input(
            result.request_id,
            result.continuation_id,
        )
        await integration.controller.resolve_input(
            result.request_id,
            result.continuation_id,
            InputDeclineSubmission(
                provenance=AnswerProvenance.EXTERNAL_CONTROLLER
            ),
            idempotency_key=ResolutionIdempotencyKey("resolve-once"),
        )
        return result
    finally:
        await integration.runtime.aclose()
