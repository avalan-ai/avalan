"""Exercise the root-only public attached and durable construction path."""

from asyncio import run
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sys import path as sys_path
from typing import cast

sys_path.append(str(Path(__file__).parents[1]))

from input_consumers.public_sdk_construction_consumer import (  # noqa: E402
    AtomicDurableBridge,
    drive_attached,
    drive_durable,
)

from avalan import AgentRunInputRequired, Orchestrator, RequestState
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    ExecutionInputRequiredError,
)
from avalan.interaction.continuation import (
    ContinuationFencingToken,
    ContinuationStoreRevision,
    PortableContinuation,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    AgentId,
    BranchId,
    CapabilityRevision,
    ConfirmationQuestion,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    QuestionId,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TurnId,
    create_input_request,
)
from avalan.interaction.handler import (
    InputHandlerContext,
    InputHandlerDetached,
)
from avalan.interaction.policy import InteractionActor
from avalan.interaction.store import CreateInteractionCommand

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-public-construction"),
        turn_id=TurnId("turn-public-construction"),
        agent_id=AgentId("agent-public-construction"),
        branch_id=BranchId("branch-public-construction"),
        model_call_id=ModelCallId("call-public-construction"),
        stream_session_id=StreamSessionId("stream-public-construction"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://public-construction",
            agent_definition_revision="agent-r1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        principal=PrincipalScope(),
    )


def _request() -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-public-construction"),
        continuation_id=ContinuationId("continuation-public-construction"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason="Confirm public durable construction.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=_NOW,
    )


def _suspension() -> DurableInteractionSuspension:
    request = _request()
    revision = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("provider"),
        model_id=ModelId("model"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )
    return DurableInteractionSuspension(
        command=CreateInteractionCommand(
            actor=InteractionActor(principal=request.origin.principal),
            request=request,
        ),
        continuation=PortableContinuation(
            continuation_id=request.continuation_id,
            request_id=request.request_id,
            origin=request.origin,
            provider_call_id=request.origin.model_call_id,
            provider_call_correlation_id=str(request.origin.model_call_id),
            definition=request.origin.definition,
            operation_cursor=0,
            generation_settings={},
            transcript=(),
            observations=(),
            revision_binding=revision,
            interaction_count=1,
            tool_loop_count=0,
            stream_sequence=0,
            state_revision=StateRevision(0),
            store_revision=ContinuationStoreRevision(0),
            created_at=_NOW,
            updated_at=_NOW,
            expires_at=_NOW + timedelta(days=1),
            fencing_token=ContinuationFencingToken(0),
        ),
    )


class _AttachedOrchestrator:
    async def __call__(self, input: object, **kwargs: object) -> str:
        assert input == "attached"
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        outcome = await runtime.handler(
            InputHandlerContext(request=_request())
        )
        assert isinstance(outcome, InputHandlerDetached)
        return "attached-complete"


class _DurableOrchestrator:
    def __init__(self, suspension: DurableInteractionSuspension) -> None:
        self.suspension = suspension

    async def __call__(self, input: object, **kwargs: object) -> str:
        assert input == "durable"
        assert isinstance(
            kwargs["interaction_runtime"],
            DurableInteractionRuntime,
        )
        request = self.suspension.command.request
        raise ExecutionInputRequiredError(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            durable=self.suspension,
        )


def test_root_only_consumer_drives_attached_and_atomic_durable_bridge() -> (
    None
):
    """Persist before inspect/resolve and preserve exact opaque correlation."""

    async def exercise() -> None:
        attached = await drive_attached(
            cast(Orchestrator, _AttachedOrchestrator())
        )
        assert attached.value == "attached-complete"

        suspension = _suspension()
        bridge = AtomicDurableBridge()
        durable = await drive_durable(
            cast(Orchestrator, _DurableOrchestrator(suspension)),
            bridge,
        )
        assert isinstance(durable, AgentRunInputRequired)
        assert durable.detached_resumption_available
        assert durable.request.state is RequestState.PENDING
        assert bridge.persisted is not None
        assert bridge.persisted.request.state is RequestState.PENDING
        assert bridge.persisted.request_id == durable.request_id
        assert bridge.persisted.continuation_id == durable.continuation_id
        assert len(bridge.resolve_calls) == 1
        assert (
            bridge.resolve_calls[0].request_id == bridge.persisted.request_id
        )
        assert (
            bridge.resolve_calls[0].continuation_id
            == bridge.persisted.continuation_id
        )

    run(exercise())
