"""Exercise canonical conversation ownership through the public agent SDK."""

from asyncio import create_task, gather, sleep
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from json import dumps
from logging import getLogger
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from phase2_fixtures import (
    binding,
    empty_stateless_plan,
    next_stateless_plan,
    retention,
)

import avalan
import avalan.conversation as conversation
from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    InputType,
    Specification,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.engine import Engine
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run public async integration checks on asyncio."""
    return "asyncio"


class _AgentEngine(EngineAgent):
    """Expose a prompt-free engine for coordinator delegation tests."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        del context
        return {
            "developer_prompt": "configured developer prompt",
            "instructions": "configured instructions",
            "settings": GenerationSettings(temperature=0.25),
            "system_prompt": "configured system prompt",
        }


def _model_response(text: str) -> TextGenerationResponse:
    """Return one real model-manager response for an agent invocation."""
    return TextGenerationResponse(
        lambda: text,
        logger=getLogger(__name__),
        use_async_generator=False,
        generation_settings=GenerationSettings(),
    )


def _memory() -> MagicMock:
    """Return disabled visible memory independent from provider state."""
    memory = MagicMock(spec=MemoryManager)
    memory.has_permanent_message = False
    memory.has_recent_message = False
    memory.permanent_message = None
    memory.recent_message = None
    memory.recent_messages = []
    memory.participant_id = None
    return memory


def _engine(model_id: str = "unused-agent-model") -> MagicMock:
    """Return the capability surface needed before coordinator delegation."""
    engine = MagicMock(spec=Engine)
    engine.model_id = model_id
    engine.model_type = "unused"
    engine.tokenizer = MagicMock(eos_token="<eos>")
    engine.provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
    )
    return engine


def _public_runtime(
    *,
    model_responses: Sequence[TextGenerationResponse] | None = None,
    tool: ToolManager | None = None,
) -> tuple[
    Orchestrator,
    avalan.AgentConversationTurn,
    conversation.InMemoryConversationStore,
    conversation.RunScopedConversationCoordinator,
    AsyncMock,
]:
    """Return one public orchestrator and exact immutable coordinated turn."""
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id="unused-agent-model",
        params={},
    )
    environment = EngineEnvironment(
        engine_uri=engine_uri,
        settings=TransformerEngineSettings(),
    )
    operation = AgentOperation(
        specification=Specification(
            role=None,
            goal=None,
            input_type=InputType.TEXT,
        ),
        environment=environment,
    )
    model_manager = AsyncMock(spec=ModelManager)
    model_manager.side_effect = tuple(
        model_responses
        or (
            _model_response("coordinated public output"),
            _model_response("coordinated public output after restart"),
        )
    )
    memory = _memory()
    tool = tool or ToolManager.create_instance()
    events = EventManager()
    orchestrator = Orchestrator(
        getLogger(__name__),
        model_manager,
        memory,
        tool,
        events,
        operation,
        exit_memory=False,
    )
    engine_agent = _AgentEngine(
        _engine(),
        memory,
        tool,
        events,
        model_manager,
        engine_uri,
        id=orchestrator.id,
    )
    orchestrator._engine_agents[dumps(asdict(environment))] = engine_agent

    agent_id = conversation.ConversationAgentId(str(orchestrator.id))
    scope = conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId("tenant-agent-sdk"),
        principal_id=conversation.AuthorityPrincipalId("principal-agent-sdk"),
        agent_id=agent_id,
        endpoint_id=conversation.AuthorityEndpointId("endpoint-agent-sdk"),
    )
    conversation_id = conversation.ConversationId("conversation-agent-sdk")
    model_slot = conversation.AgentModelSlot("primary")
    topology_path = conversation.parent_agent_topology_path(
        agent_id,
        model_slot,
    )
    binding_seed = replace(
        binding("agent-sdk-seed", agent=str(agent_id)),
        model_or_deployment="unused-agent-model",
    )
    lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=binding_seed,
    )
    lane_binding = replace(binding_seed, lane_id=lane_id)
    provider_lane = conversation.AgentProviderLane(
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id=agent_id,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=lane_binding,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    topology = conversation.AgentLaneTopology(
        conversation_id=conversation_id,
        lanes=(provider_lane,),
    )
    plan = empty_stateless_plan(lane_binding)
    provider_result = conversation.fake_provider_result(
        plan,
        turn=1,
        text="coordinated public output",
    )
    next_plan = next_stateless_plan(lane_binding, provider_result.items)
    next_provider_result = conversation.fake_provider_result(
        next_plan,
        turn=2,
        text="coordinated public output after restart",
    )
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=lane_binding,
                capability_profile=conversation.fake_capability_profile(
                    lane_binding
                ),
                provider_script=(
                    conversation.DeterministicFakeProviderScript(
                        results=(provider_result, next_provider_result)
                    )
                ),
            ),
        ),
    )
    turn = avalan.AgentConversationTurn(
        coordinator=coordinator,
        authority=scope,
        topology=topology,
        lanes=(
            conversation.AgentConversationLane(
                lane_id=lane_id,
                mode=conversation.ConversationMode.STATELESS,
            ),
        ),
        logical_turn_id=conversation.LogicalTurnId("agent-sdk-turn"),
        execution_segment_id=conversation.ExecutionSegmentId(
            "agent-sdk-segment"
        ),
        checkpoint_id=conversation.CheckpointId("agent-sdk-checkpoint"),
        branch_id=conversation.ConversationBranchId("agent-sdk-branch"),
        provisional_response_id=conversation.ProvisionalResponseId(
            "agent-sdk-provisional"
        ),
        public_response_id=conversation.PublicResponseId("agent-sdk-response"),
        idempotency_key=conversation.RequestIdempotencyKey("agent-sdk-key"),
        retention=retention(),
    )
    return orchestrator, turn, store, coordinator, model_manager


@dataclass(frozen=True, slots=True)
class _MultiAgentRuntime:
    """Hold one immutable parent-plus-two-child test invocation."""

    orchestrator: Orchestrator
    turn: avalan.AgentConversationTurn
    store: conversation.InMemoryConversationStore
    coordinator: conversation.RunScopedConversationCoordinator
    model_manager: AsyncMock
    child_orchestrators: tuple[Orchestrator, ...]
    child_bindings: tuple[avalan.AgentConversationChildBinding, ...]
    child_model_managers: tuple[AsyncMock, ...]
    child_effects: list[int]


def _configured_child(
    *,
    agent_id: UUID,
    model_id: str,
    model_responses: Sequence[object],
    tool: ToolManager | None = None,
    controller: conversation.DeterministicFaultController | None = None,
) -> tuple[Orchestrator, AsyncMock]:
    """Return one loaded child orchestrator with its actual engine agent."""
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id=model_id,
        params={},
    )
    environment = EngineEnvironment(
        engine_uri=engine_uri,
        settings=TransformerEngineSettings(),
    )
    operation = AgentOperation(
        specification=Specification(
            role=None,
            goal=None,
            instructions=f"instructions for {model_id}",
            system_prompt=f"system for {model_id}",
            developer_prompt=f"developer for {model_id}",
            settings=GenerationSettings(temperature=0.4),
            input_type=InputType.TEXT,
        ),
        environment=environment,
    )
    model_manager = AsyncMock(spec=ModelManager)
    responses = iter(model_responses)
    first_call = True

    async def invoke_model(call: object) -> object:
        """Return the next child response after an optional async pause."""
        nonlocal first_call
        del call
        if controller is not None and first_call:
            first_call = False
            await controller.reach("provider:dispatch")
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return response

    model_manager.side_effect = invoke_model
    memory = _memory()
    child_tool = tool or ToolManager.create_instance()
    events = EventManager()
    orchestrator = Orchestrator(
        getLogger(__name__),
        model_manager,
        memory,
        child_tool,
        events,
        operation,
        exit_memory=False,
        id=agent_id,
    )
    engine_agent = _AgentEngine(
        _engine(model_id),
        memory,
        child_tool,
        events,
        model_manager,
        engine_uri,
        id=agent_id,
    )
    orchestrator._engine_agents[dumps(asdict(environment))] = engine_agent
    return orchestrator, model_manager


def _topology_lane(
    conversation_id: conversation.ConversationId,
    *,
    owner_kind: conversation.ProviderLaneOwnerKind,
    agent_id: conversation.ConversationAgentId,
    model_slot: str,
    topology_path: conversation.AgentTopologyPath,
    retention_policy: conversation.ChildLaneRetentionPolicy,
    model_id: str,
    parent_lane_id: conversation.ProviderLaneId | None = None,
) -> conversation.AgentProviderLane:
    """Return one lane bound to its exact deterministic topology path."""
    slot = conversation.AgentModelSlot(model_slot)
    seed = replace(
        binding(f"{agent_id}-{model_slot}", agent=str(agent_id)),
        model_or_deployment=model_id,
    )
    lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=owner_kind,
        topology_path=topology_path,
        model_slot=slot,
        binding=seed,
    )
    return conversation.AgentProviderLane(
        owner_kind=owner_kind,
        agent_id=agent_id,
        topology_path=topology_path,
        model_slot=slot,
        binding=replace(seed, lane_id=lane_id),
        retention_policy=retention_policy,
        parent_lane_id=parent_lane_id,
    )


def _private_child_result(
    plan: conversation.StatelessProviderPlan,
    *,
    text: str,
) -> conversation.ProviderResult:
    """Return one visible child message plus private opaque reasoning."""
    base = conversation.fake_provider_result(plan, turn=1, text=text)
    message = replace(
        base.items[0],
        order=conversation.ProviderItemOrder(1),
        provider_index=conversation.ProviderItemIndex(1),
    )
    reasoning = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("child-a-private-reasoning"),
        lane_id=plan.binding.lane_id,
        model_call_id=message.model_call_id,
        kind=conversation.ProviderItemKind.REASONING,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "id": "child-a-private-reasoning",
            "summary": (),
            "type": "reasoning",
        },
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        opaque_state=conversation.OpaqueProviderState(
            _value=b"child-a-private-secret"
        ),
    )
    return replace(base, items=(reasoning, message))


def _next_plan(
    lane_binding: conversation.ProviderLaneBinding,
    previous: conversation.ProviderResult,
) -> conversation.StatelessProviderPlan:
    """Return the exact retained stateless plan for a later turn."""
    return replace(
        empty_stateless_plan(lane_binding),
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=(lane_binding.continuation_codec_version),
            items=previous.items,
        ),
    )


def _multi_agent_runtime(
    *,
    failed_child: bool = False,
    label: str = "",
    reused: _MultiAgentRuntime | None = None,
    child_controller: conversation.DeterministicFaultController | None = None,
) -> _MultiAgentRuntime:
    """Return one frozen parent and exactly two isolated child lanes."""
    if reused is None:
        orchestrator, seed_turn, _, _, model_manager = _public_runtime(
            model_responses=(
                _model_response(f"parent merged child results{label}"),
                _model_response(f"parent continued after restart{label}"),
            )
        )
    else:
        orchestrator = reused.orchestrator
        seed_turn = reused.turn
        model_manager = reused.model_manager

        async def route_parent(call: ModelCall) -> TextGenerationResponse:
            """Return an invocation-specific parent response under reuse."""
            operation_input = call.operation.input
            if type(operation_input) is not list or not operation_input:
                raise AssertionError("parent input must be messages")
            current = operation_input[-1]
            if type(current) is not Message:
                raise AssertionError("parent input must be messages")
            content = current.content
            if type(content) is not str:
                raise AssertionError("parent input must be text")
            if content.startswith("left input"):
                return _model_response("parent merged child results left")
            if content.startswith("right input"):
                return _model_response("parent merged child results right")
            raise AssertionError("unexpected concurrent parent input")

        model_manager.side_effect = route_parent
    parent_agent_id = seed_turn.authority.agent_id
    conversation_id = conversation.ConversationId("multi-agent-conversation")
    parent_path = conversation.parent_agent_topology_path(
        parent_agent_id,
        conversation.AgentModelSlot("parent"),
    )
    parent = _topology_lane(
        conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id=parent_agent_id,
        model_slot="parent",
        topology_path=parent_path,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        model_id="unused-agent-model",
    )
    child_a_uuid = UUID("00000000-0000-0000-0000-00000000000a")
    child_b_uuid = UUID("00000000-0000-0000-0000-00000000000b")
    child_a_id = conversation.ConversationAgentId(str(child_a_uuid))
    child_b_id = conversation.ConversationAgentId(str(child_b_uuid))
    child_a = _topology_lane(
        conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id=child_a_id,
        model_slot="research",
        topology_path=conversation.child_agent_topology_path(
            parent_path,
            child_a_id,
            conversation.AgentModelSlot("research"),
        ),
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        model_id="child-model-a",
        parent_lane_id=parent.lane_id,
    )
    child_b = _topology_lane(
        conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id=child_b_id,
        model_slot="critic",
        topology_path=conversation.child_agent_topology_path(
            parent_path,
            child_b_id,
            conversation.AgentModelSlot("critic"),
        ),
        retention_policy=(
            conversation.ChildLaneRetentionPolicy.DISCARD_TERMINAL
        ),
        model_id="child-model-b",
        parent_lane_id=parent.lane_id,
    )
    topology = conversation.AgentLaneTopology(
        conversation_id=conversation_id,
        lanes=(parent, child_a, child_b),
    )

    parent_plan = empty_stateless_plan(parent.binding)
    parent_first = conversation.fake_provider_result(
        parent_plan,
        turn=1,
        text=f"parent merged child results{label}",
    )
    parent_second = conversation.fake_provider_result(
        _next_plan(parent.binding, parent_first),
        turn=2,
        text=f"parent continued after restart{label}",
    )
    child_a_plan = empty_stateless_plan(child_a.binding)
    child_a_first = _private_child_result(
        child_a_plan,
        text=f"canonical child A{label}",
    )
    child_a_second = conversation.fake_provider_result(
        _next_plan(child_a.binding, child_a_first),
        turn=2,
        text=f"canonical child A after restart{label}",
    )
    child_b_plan = empty_stateless_plan(child_b.binding)
    child_b_first = conversation.fake_provider_result(
        child_b_plan,
        turn=1,
        text=f"canonical child B{label}",
    )
    child_b_second = conversation.fake_provider_result(
        child_b_plan,
        turn=2,
        text=f"canonical child B after restart{label}",
    )
    failed_controller = (
        conversation.DeterministicFaultController(
            (
                conversation.FaultAction(
                    label="provider:dispatch",
                    exception=conversation.ConversationValidationError(),
                ),
            )
        )
        if failed_child
        else None
    )
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            seed_turn.authority
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=tuple(
            conversation.ConversationLaneRuntime(
                binding=lane.binding,
                capability_profile=conversation.fake_capability_profile(
                    lane.binding
                ),
                retention_policy=lane.retention_policy,
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=results,
                    controller=controller,
                ),
            )
            for lane, results, controller in (
                (parent, (parent_first, parent_second), None),
                (
                    child_a,
                    (child_a_first, child_a_second),
                    child_controller,
                ),
                (
                    child_b,
                    (child_b_first, child_b_second),
                    failed_controller,
                ),
            )
        ),
    )
    turn = avalan.AgentConversationTurn(
        coordinator=coordinator,
        authority=seed_turn.authority,
        topology=topology,
        lanes=tuple(
            conversation.AgentConversationLane(
                lane_id=lane.lane_id,
                mode=conversation.ConversationMode.STATELESS,
            )
            for lane in topology.lanes
        ),
        logical_turn_id=conversation.LogicalTurnId("multi-agent-turn-1"),
        execution_segment_id=conversation.ExecutionSegmentId(
            "multi-agent-segment-1"
        ),
        checkpoint_id=conversation.CheckpointId("multi-agent-checkpoint-1"),
        branch_id=conversation.ConversationBranchId("multi-agent-branch"),
        provisional_response_id=conversation.ProvisionalResponseId(
            "multi-agent-provisional-1"
        ),
        public_response_id=conversation.PublicResponseId(
            "multi-agent-response-1"
        ),
        idempotency_key=conversation.RequestIdempotencyKey(
            "multi-agent-key-1"
        ),
        retention=retention(),
    )
    child_effects: list[int] = []

    async def child_private_lookup(value: int) -> str:
        """Return one child-private tool result."""
        child_effects.append(value)
        return f"private-child-result-{value}"

    child_a_tool = ToolManager.create_instance(
        available_toolsets=[ToolSet(tools=[child_private_lookup])],
        enable_tools=["child_private_lookup"],
    )
    child_a_orchestrator, child_a_manager = _configured_child(
        agent_id=child_a_uuid,
        model_id="child-model-a",
        model_responses=(
            (
                conversation.ConversationValidationError()
                if failed_child
                else _model_response(
                    '<tool_call>{"name":"child_private_lookup",'
                    '"arguments":{"value":11}}</tool_call>'
                )
            ),
            _model_response(f"canonical child A{label}"),
            _model_response(f"canonical child A after restart{label}"),
        ),
        tool=child_a_tool,
        controller=child_controller,
    )
    child_b_orchestrator, child_b_manager = _configured_child(
        agent_id=child_b_uuid,
        model_id="child-model-b",
        model_responses=(
            _model_response(f"canonical child B{label}"),
            _model_response(f"canonical child B after restart{label}"),
        ),
    )
    child_orchestrators = (child_a_orchestrator, child_b_orchestrator)
    child_bindings = tuple(
        avalan.AgentConversationChildBinding(
            lane_id=lane.lane_id,
            orchestrator=child,
            operation_index=0,
        )
        for lane, child in zip(
            (child_a, child_b),
            child_orchestrators,
            strict=True,
        )
    )
    return _MultiAgentRuntime(
        orchestrator=orchestrator,
        turn=turn,
        store=store,
        coordinator=coordinator,
        model_manager=model_manager,
        child_orchestrators=child_orchestrators,
        child_bindings=child_bindings,
        child_model_managers=(child_a_manager, child_b_manager),
        child_effects=child_effects,
    )


async def test_normative_agent_contract() -> None:
    """Continue a restarted public agent through canonical ownership."""
    orchestrator, turn, store, coordinator, model_manager = _public_runtime()
    try:
        result = await avalan.run_agent(
            orchestrator,
            "public input",
            conversation_turn=turn,
        )

        assert isinstance(result, avalan.AgentRunCompleted)
        assert result.value == "coordinated public output"
        assert model_manager.await_count == 1
        first_call = model_manager.await_args_list[0].args[0]
        assert first_call.operation.input[0].content == "public input"
        text_parameters = first_call.operation.parameters["text"]
        assert text_parameters.instructions == "configured instructions"
        assert text_parameters.system_prompt == "configured system prompt"
        assert (
            text_parameters.developer_prompt == "configured developer prompt"
        )
        assert first_call.operation.generation_settings.temperature == 0.25
        checkpoint = await store.load(
            conversation.CheckpointId("agent-sdk-checkpoint"),
            turn.authority,
        )
        assert checkpoint.content.lane_topology == (
            turn.topology.checkpoint_topology()
        )
        assert tuple(
            entry.content
            for entry in checkpoint.content.visible_transcript.entries
        ) == ("public input",)
        restarted_turn = replace(
            turn,
            parent=checkpoint,
            logical_turn_id=conversation.LogicalTurnId(
                "agent-sdk-turn-after-restart"
            ),
            execution_segment_id=conversation.ExecutionSegmentId(
                "agent-sdk-segment-after-restart"
            ),
            checkpoint_id=conversation.CheckpointId(
                "agent-sdk-checkpoint-after-restart"
            ),
            provisional_response_id=conversation.ProvisionalResponseId(
                "agent-sdk-provisional-after-restart"
            ),
            public_response_id=conversation.PublicResponseId(
                "agent-sdk-response-after-restart"
            ),
            idempotency_key=conversation.RequestIdempotencyKey(
                "agent-sdk-key-after-restart"
            ),
        )
        second = await avalan.run_agent(
            orchestrator,
            "public input after restart",
            conversation_turn=restarted_turn,
        )

        assert isinstance(second, avalan.AgentRunCompleted)
        assert second.value == "coordinated public output after restart"
        second_checkpoint = await store.load(
            restarted_turn.checkpoint_id,
            restarted_turn.authority,
        )
        assert second_checkpoint.identity.parent_checkpoint_id == (
            checkpoint.identity.checkpoint_id
        )
        assert second_checkpoint.identity.parent_sequence == (
            checkpoint.identity.sequence
        )
        assert second_checkpoint.identity.sequence == 1
        assert tuple(
            entry.content
            for entry in second_checkpoint.content.visible_transcript.entries
        ) == ("public input", "public input after restart")
        second_lane = second_checkpoint.content.lanes[0]
        assert isinstance(
            second_lane,
            conversation.StatelessProviderLaneSnapshot,
        )
        assert tuple(item.order for item in second_lane.ledger.items) == (0, 1)
        diagnostics = coordinator.fake_provider_diagnostics(
            turn.topology.parent_lanes[0].lane_id
        )
        assert diagnostics.plans == ()
        assert diagnostics.remaining_results == 2
        assert store.diagnostics.checkpoints == 2
        assert store.diagnostics.public_responses == 2
        assert store.diagnostics.output_records == 2
        assert store.diagnostics.idempotency_records == 2
        assert model_manager.await_count == 2
        engine_agent = next(iter(orchestrator._engine_agents.values()))
        retained_state = (
            vars(orchestrator),
            vars(engine_agent),
            vars(orchestrator._tool),
        )
        state_names = set(type(turn).__slots__)
        for state in retained_state:
            state_names.update(state)
        assert not any("replay_owner" in name for name in state_names)
        assert not any(
            "ReplayOwner" in type(value).__name__
            or "DirectReplayExecutionState" in type(value).__name__
            for state in retained_state
            for value in state.values()
        )
    finally:
        await orchestrator.__aexit__(None, None, None)


async def test_parent_prepared_call_runs_real_tool_loop_once() -> None:
    """Run configured model, tool, and continuation under one checkpoint."""
    effects: list[int] = []

    async def configured_lookup(value: int) -> str:
        """Return one configured lookup result."""
        effects.append(value)
        return f"lookup-result-{value}"

    tool = ToolManager.create_instance(
        available_toolsets=[ToolSet(tools=[configured_lookup])],
        enable_tools=["configured_lookup"],
    )
    orchestrator, turn, store, coordinator, model_manager = _public_runtime(
        tool=tool,
        model_responses=(
            _model_response(
                '<tool_call>{"name":"configured_lookup",'
                '"arguments":{"value":7}}</tool_call>'
            ),
            _model_response("configured final answer"),
        ),
    )
    try:
        response = await orchestrator(
            "use the configured lookup",
            conversation_turn=turn,
        )
        output = await response.to_str()

        assert output == "configured final answer"
        assert effects == [7]
        assert model_manager.await_count == 2
        first_call, continuation_call = (
            awaited.args[0] for awaited in model_manager.await_args_list
        )
        assert (
            first_call.operation.input[0].content
            == "use the configured lookup"
        )
        continuation_input = continuation_call.operation.input
        assert any(
            getattr(getattr(message, "role", None), "value", None) == "tool"
            and "lookup-result-7" in str(getattr(message, "content", ""))
            for message in continuation_input
        )
        checkpoint = await store.load(turn.checkpoint_id, turn.authority)
        assert checkpoint.content.visible_transcript.entries == (
            conversation.VisibleTranscriptEntry(
                role=conversation.VisibleTranscriptRole.USER,
                content="use the configured lookup",
            ),
        )
        assert len(checkpoint.content.lanes) == 1
        segments = checkpoint.content.execution_segments
        assert tuple(segment.phase for segment in segments) == (
            conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
            conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
            conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        )
        assert segments[0].segment_index == segments[1].segment_index == 0
        assert segments[2].segment_index == 1
        assert len(segments[0].tools) == len(segments[1].tools) == 1
        assert segments[0].tools[0].phase is (
            conversation.ToolExecutionPhase.REQUESTED
        )
        assert segments[1].tools[0].phase is (
            conversation.ToolExecutionPhase.OUTPUT_PERSISTED
        )
        assert segments[0].tools[0].call_id == segments[1].tools[0].call_id
        assert segments[1].tools[0].output_id is not None
        assert segments[2].tools == ()
        assert store.diagnostics.checkpoints == 1
        assert store.diagnostics.output_records == 1
        assert (
            coordinator.fake_provider_diagnostics(
                turn.topology.parent_lanes[0].lane_id
            ).plans
            == ()
        )
        execution = response.execution
        assert execution is not None
        assert tuple(execution.messages) == tuple(
            message
            for message in execution.messages
            if getattr(getattr(message, "role", None), "value", None) != "tool"
        )
    finally:
        await orchestrator.__aexit__(None, None, None)


async def test_parent_tool_effect_failure_fences_unsafe_retry(
    record_property: Callable[[str, object], None],
) -> None:
    """Fence a real tool effect when its model continuation fails."""
    record_property("conversation_acceptance_evidence", "negative")
    effects: list[int] = []

    async def configured_lookup(value: int) -> str:
        """Return one configured lookup result."""
        effects.append(value)
        return f"lookup-result-{value}"

    tool = ToolManager.create_instance(
        available_toolsets=[ToolSet(tools=[configured_lookup])],
        enable_tools=["configured_lookup"],
    )
    first_response = _model_response(
        '<tool_call>{"name":"configured_lookup",'
        '"arguments":{"value":9}}</tool_call>'
    )
    orchestrator, turn, store, coordinator, model_manager = _public_runtime(
        tool=tool,
        model_responses=(first_response, _model_response("unused")),
    )
    assert tuple(descriptor.name for descriptor in tool.list_tools()) == (
        "configured_lookup",
    )
    model_manager.side_effect = (
        first_response,
        RuntimeError("injected continuation failure after tool effect"),
    )
    try:
        result = await avalan.run_agent(
            orchestrator,
            "fail after the configured lookup",
            conversation_turn=turn,
        )

        assert isinstance(result, avalan.AgentRunFailed)
        assert result.code == "agent.execution_failed"
        assert effects == [9]
        assert model_manager.await_count == 2
        continuation_call = model_manager.await_args_list[1].args[0]
        assert any(
            getattr(getattr(message, "role", None), "value", None) == "tool"
            and "lookup-result-9" in str(getattr(message, "content", ""))
            for message in continuation_call.operation.input
        )
        checkpoints = await store.list_checkpoints(
            turn.authority,
            cursor=None,
            limit=10,
        )
        assert checkpoints.checkpoints == ()
        assert store.diagnostics.checkpoints == 0
        assert store.diagnostics.public_responses == 0
        assert store.diagnostics.output_records == 0
        assert store.diagnostics.provisional_responses == 0
        assert store.diagnostics.idempotency_records == 1

        reconciliation = await store.reconcile_ambiguous_dispatch(
            conversation.AmbiguousDispatchReconciliationRequest(
                authority=turn.authority,
                operation=conversation.ConversationOperation.CREATE,
                idempotency_key=turn.idempotency_key,
                resolution=(
                    conversation.AmbiguousDispatchResolution.RETAIN_FENCE
                ),
            )
        )
        assert reconciliation.disposition is (
            conversation.AmbiguousDispatchReconciliationDisposition.FENCE_RETAINED
        )

        retry = await avalan.run_agent(
            orchestrator,
            "fail after the configured lookup",
            conversation_turn=turn,
        )
        assert isinstance(retry, avalan.AgentRunFailed)
        assert effects == [9]
        assert model_manager.await_count == 2
        assert store.diagnostics.checkpoints == 0
        assert store.diagnostics.public_responses == 0
    finally:
        await orchestrator.__aexit__(None, None, None)


async def test_public_agent_rejects_lossy_input_before_dispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject multiple public messages instead of silently dropping one."""
    record_property(
        "conversation_acceptance_evidence", "pre_dispatch_rejection"
    )
    orchestrator, turn, store, coordinator, model_manager = _public_runtime()
    assert model_manager.await_count == 0
    try:
        result = await avalan.run_agent(
            orchestrator,
            ["first", "second"],
            conversation_turn=turn,
        )

        assert isinstance(result, avalan.AgentRunFailed)
        model_manager.assert_not_awaited()
        diagnostics = coordinator.fake_provider_diagnostics(
            turn.topology.parent_lanes[0].lane_id
        )
        assert diagnostics.plans == ()
        assert store.diagnostics.checkpoints == 0
    finally:
        await orchestrator.__aexit__(None, None, None)


async def test_configured_children_run_actual_engines_before_parent() -> None:
    """Invoke two explicitly bound child engines and merge only final text."""
    runtime = _multi_agent_runtime()
    try:
        result = await avalan.run_agent(
            runtime.orchestrator,
            "multi-agent input",
            conversation_turn=runtime.turn,
            conversation_children=runtime.child_bindings,
        )

        assert isinstance(result, avalan.AgentRunCompleted)
        assert result.value == "parent merged child results"
        assert tuple(
            manager.await_count for manager in runtime.child_model_managers
        ) == (2, 1)
        assert runtime.child_effects == [11]
        for manager in runtime.child_model_managers:
            child_call = manager.await_args_list[0].args[0]
            assert child_call.operation.input == [
                Message(
                    role=MessageRole.USER,
                    content="multi-agent input",
                )
            ]
        assert runtime.model_manager.await_count == 1
        parent_await = runtime.model_manager.await_args
        assert parent_await is not None
        parent_call = parent_await.args[0]
        assert (
            parent_call.operation.input[-1].content
            == "multi-agent input\n\nCanonical child results:\n"
            "canonical child A\ncanonical child B"
        )
        assert "private-child-result-11" not in str(
            parent_call.operation.input
        )
        checkpoint = await runtime.store.load(
            runtime.turn.checkpoint_id,
            runtime.turn.authority,
        )
        assert checkpoint.content.lane_topology == (
            runtime.turn.topology.checkpoint_topology()
        )
        assert tuple(
            entry.content
            for entry in checkpoint.content.visible_transcript.entries
        ) == (
            "multi-agent input",
            "canonical child A",
            "canonical child B",
        )
        assert runtime.store.diagnostics.checkpoints == 1
    finally:
        await runtime.coordinator.close()
        for child in runtime.child_orchestrators:
            await child.__aexit__(None, None, None)
        await runtime.orchestrator.__aexit__(None, None, None)


async def test_parent_two_children_persist_isolation_and_restart(
    record_property: Callable[[str, object], None],
) -> None:
    """Merge two exact child outputs and continue from retained state."""
    record_property("conversation_acceptance_evidence", "runtime")
    runtime = _multi_agent_runtime()
    turn = runtime.turn
    parent, child_a, child_b = turn.topology.lanes
    assert len(turn.topology.child_lanes) == len(runtime.child_bindings) == 2
    try:
        first = await avalan.run_agent(
            runtime.orchestrator,
            "multi-agent input",
            conversation_turn=turn,
            conversation_children=runtime.child_bindings,
        )

        assert isinstance(first, avalan.AgentRunCompleted)
        assert first.value == "parent merged child results"
        assert runtime.model_manager.await_count == 1
        assert tuple(
            manager.await_count for manager in runtime.child_model_managers
        ) == (2, 1)
        assert runtime.child_effects == [11]
        first_checkpoint = await runtime.store.load(
            turn.checkpoint_id,
            turn.authority,
        )
        assert first_checkpoint.content.lane_topology == (
            turn.topology.checkpoint_topology()
        )
        retained = {lane.lane_id for lane in first_checkpoint.content.lanes}
        assert parent.lane_id in retained
        assert child_a.lane_id in retained
        assert child_b.lane_id not in retained
        assert tuple(
            entry.content
            for entry in first_checkpoint.content.visible_transcript.entries
        ) == (
            "multi-agent input",
            "canonical child A",
            "canonical child B",
        )

        for lane in turn.topology.lanes:
            assert (
                runtime.coordinator.fake_provider_diagnostics(
                    lane.lane_id
                ).plans
                == ()
            )
        parent_call = runtime.model_manager.await_args_list[0].args[0]
        assert (
            parent_call.operation.input[-1].content
            == "multi-agent input\n\nCanonical child results:\n"
            "canonical child A\ncanonical child B"
        )
        assert "private-child-result-11" not in str(
            parent_call.operation.input
        )
        child_a_snapshot = next(
            lane
            for lane in first_checkpoint.content.lanes
            if lane.lane_id == child_a.lane_id
        )
        assert isinstance(
            child_a_snapshot,
            conversation.StatelessProviderLaneSnapshot,
        )
        assert tuple(item.kind for item in child_a_snapshot.ledger.items) == (
            conversation.ProviderItemKind.FUNCTION_CALL,
            conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
            conversation.ProviderItemKind.MESSAGE,
        )
        restarted_turn = replace(
            turn,
            parent=first_checkpoint,
            logical_turn_id=conversation.LogicalTurnId("multi-agent-turn-2"),
            execution_segment_id=conversation.ExecutionSegmentId(
                "multi-agent-segment-2"
            ),
            checkpoint_id=conversation.CheckpointId(
                "multi-agent-checkpoint-2"
            ),
            provisional_response_id=conversation.ProvisionalResponseId(
                "multi-agent-provisional-2"
            ),
            public_response_id=conversation.PublicResponseId(
                "multi-agent-response-2"
            ),
            idempotency_key=conversation.RequestIdempotencyKey(
                "multi-agent-key-2"
            ),
        )
        second = await avalan.run_agent(
            runtime.orchestrator,
            "input after restart",
            conversation_turn=restarted_turn,
            conversation_children=runtime.child_bindings,
        )

        assert isinstance(second, avalan.AgentRunCompleted)
        assert second.value == "parent continued after restart"
        second_checkpoint = await runtime.store.load(
            restarted_turn.checkpoint_id,
            restarted_turn.authority,
        )
        assert second_checkpoint.identity.parent_checkpoint_id == (
            first_checkpoint.identity.checkpoint_id
        )
        assert second_checkpoint.identity.parent_sequence == (
            first_checkpoint.identity.sequence
        )
        assert second_checkpoint.identity.logical_turn_id == (
            restarted_turn.logical_turn_id
        )
        assert second_checkpoint.content.lane_topology == (
            turn.topology.checkpoint_topology()
        )
        second_retained = {
            lane.lane_id for lane in second_checkpoint.content.lanes
        }
        assert child_a.lane_id in second_retained
        assert child_b.lane_id not in second_retained
        second_child_a = next(
            lane
            for lane in second_checkpoint.content.lanes
            if lane.lane_id == child_a.lane_id
        )
        assert isinstance(
            second_child_a,
            conversation.StatelessProviderLaneSnapshot,
        )
        assert len(second_child_a.ledger.items) == 4
        assert tuple(
            manager.await_count for manager in runtime.child_model_managers
        ) == (3, 2)
        assert runtime.model_manager.await_count == 2
        assert runtime.child_effects == [11]
        child_await = runtime.child_model_managers[0].await_args
        assert child_await is not None
        assert child_await.args[0].operation.input == [
            Message(
                role=MessageRole.USER,
                content="input after restart",
            )
        ]
        assert tuple(
            entry.content
            for entry in second_checkpoint.content.visible_transcript.entries
        ) == (
            "multi-agent input",
            "canonical child A",
            "canonical child B",
            "input after restart",
            "canonical child A after restart",
            "canonical child B after restart",
        )
        for lane in turn.topology.lanes:
            assert (
                runtime.coordinator.fake_provider_diagnostics(
                    lane.lane_id
                ).plans
                == ()
            )
    finally:
        await runtime.coordinator.close()
        for child in runtime.child_orchestrators:
            await child.__aexit__(None, None, None)
        await runtime.orchestrator.__aexit__(None, None, None)


async def test_child_merge_rejects_wrong_provider_and_model_binding(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject missing, duplicate, identity, and model drift pre-dispatch."""
    record_property(
        "conversation_acceptance_evidence", "pre_dispatch_rejection"
    )
    runtime = _multi_agent_runtime()
    child_a = runtime.turn.topology.child_lanes[0]
    wrong_child, wrong_manager = _configured_child(
        agent_id=UUID(str(child_a.agent_id)),
        model_id="wrong-child-model",
        model_responses=(_model_response("must not dispatch"),),
    )
    wrong_model_binding = avalan.AgentConversationChildBinding(
        lane_id=child_a.lane_id,
        orchestrator=wrong_child,
        operation_index=0,
    )
    wrong_agent, wrong_agent_manager = _configured_child(
        agent_id=UUID("00000000-0000-0000-0000-00000000000c"),
        model_id="child-model-a",
        model_responses=(_model_response("must not dispatch"),),
    )
    wrong_agent_binding = avalan.AgentConversationChildBinding(
        lane_id=child_a.lane_id,
        orchestrator=wrong_agent,
        operation_index=0,
    )
    assert (
        wrong_model_binding.lane_id
        == wrong_agent_binding.lane_id
        == child_a.lane_id
    )
    try:
        invalid_bindings = (
            runtime.child_bindings[:1],
            (
                runtime.child_bindings[0],
                runtime.child_bindings[0],
                runtime.child_bindings[1],
            ),
            (
                replace(
                    runtime.child_bindings[0],
                    orchestrator=runtime.child_orchestrators[1],
                ),
                runtime.child_bindings[1],
            ),
            (wrong_agent_binding, runtime.child_bindings[1]),
            (wrong_model_binding, runtime.child_bindings[1]),
        )
        for index, bindings in enumerate(invalid_bindings):
            result = await avalan.run_agent(
                runtime.orchestrator,
                f"binding check {index}",
                conversation_turn=runtime.turn,
                conversation_children=bindings,
            )

            assert isinstance(result, avalan.AgentRunFailed)
        runtime.model_manager.assert_not_awaited()
        wrong_manager.assert_not_awaited()
        wrong_agent_manager.assert_not_awaited()
        assert all(
            manager.await_count == 0
            for manager in runtime.child_model_managers
        )
        assert runtime.store.diagnostics.checkpoints == 0
        for lane in runtime.turn.topology.lanes:
            assert (
                runtime.coordinator.fake_provider_diagnostics(
                    lane.lane_id
                ).plans
                == ()
            )
    finally:
        await runtime.coordinator.close()
        for child in (
            *runtime.child_orchestrators,
            wrong_child,
            wrong_agent,
        ):
            await child.__aexit__(None, None, None)
        await runtime.orchestrator.__aexit__(None, None, None)


async def test_failed_child_never_dispatches_or_publishes_parent(
    record_property: Callable[[str, object], None],
) -> None:
    """Roll back the child phase when one exact child lane fails."""
    record_property("conversation_acceptance_evidence", "negative")
    runtime = _multi_agent_runtime(failed_child=True)
    assert tuple(
        manager.await_count for manager in runtime.child_model_managers
    ) == (0, 0)
    try:
        result = await avalan.run_agent(
            runtime.orchestrator,
            "failed child input",
            conversation_turn=runtime.turn,
            conversation_children=runtime.child_bindings,
        )

        assert isinstance(result, avalan.AgentRunFailed)
        assert runtime.store.diagnostics.checkpoints == 0
        assert runtime.store.diagnostics.public_responses == 0
        assert runtime.store.diagnostics.output_records == 0
        assert tuple(
            manager.await_count for manager in runtime.child_model_managers
        ) == (1, 0)
        assert runtime.child_effects == []
        runtime.model_manager.assert_not_awaited()
        for lane in runtime.turn.topology.lanes:
            assert (
                runtime.coordinator.fake_provider_diagnostics(
                    lane.lane_id
                ).plans
                == ()
            )
    finally:
        await runtime.coordinator.close()
        for child in runtime.child_orchestrators:
            await child.__aexit__(None, None, None)
        await runtime.orchestrator.__aexit__(None, None, None)


async def test_two_child_fanout_keeps_event_loop_heartbeat_live() -> None:
    """Keep the coordinator event loop live during two-child fan-out."""
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:dispatch",
                pause=True,
            ),
        )
    )
    runtime = _multi_agent_runtime(child_controller=controller)
    heartbeat_ticks = 0
    try:
        run_task = create_task(
            avalan.run_agent(
                runtime.orchestrator,
                "heartbeat fan-out",
                conversation_turn=runtime.turn,
                conversation_children=runtime.child_bindings,
            )
        )
        await controller.wait_until_entered("provider:dispatch")
        for _ in range(3):
            await sleep(0)
            heartbeat_ticks += 1
        controller.release("provider:dispatch")
        result = await run_task

        assert heartbeat_ticks == 3
        assert isinstance(result, avalan.AgentRunCompleted)
        assert result.value == "parent merged child results"
        assert tuple(
            manager.await_count for manager in runtime.child_model_managers
        ) == (2, 1)
        for lane in runtime.turn.topology.lanes:
            assert (
                runtime.coordinator.fake_provider_diagnostics(
                    lane.lane_id
                ).plans
                == ()
            )
    finally:
        controller.close()
        await runtime.coordinator.close()
        for child in runtime.child_orchestrators:
            await child.__aexit__(None, None, None)
        await runtime.orchestrator.__aexit__(None, None, None)


async def test_reused_orchestrator_keeps_concurrent_turns_disjoint() -> None:
    """Keep explicit per-invocation turns separate on one reused agent."""
    left = _multi_agent_runtime(label=" left")
    right = _multi_agent_runtime(label=" right", reused=left)
    try:
        left_result, right_result = await gather(
            avalan.run_agent(
                left.orchestrator,
                "left input",
                conversation_turn=left.turn,
                conversation_children=left.child_bindings,
            ),
            avalan.run_agent(
                left.orchestrator,
                "right input",
                conversation_turn=right.turn,
                conversation_children=right.child_bindings,
            ),
        )

        assert isinstance(left_result, avalan.AgentRunCompleted)
        assert isinstance(right_result, avalan.AgentRunCompleted)
        assert left_result.value == "parent merged child results left"
        assert right_result.value == "parent merged child results right"
        left_checkpoint, right_checkpoint = await gather(
            left.store.load(left.turn.checkpoint_id, left.turn.authority),
            right.store.load(right.turn.checkpoint_id, right.turn.authority),
        )
        assert tuple(
            entry.content
            for entry in left_checkpoint.content.visible_transcript.entries
        ) == (
            "left input",
            "canonical child A left",
            "canonical child B left",
        )
        assert tuple(
            entry.content
            for entry in right_checkpoint.content.visible_transcript.entries
        ) == (
            "right input",
            "canonical child A right",
            "canonical child B right",
        )
        assert left.model_manager.await_count == 2
        assert tuple(
            manager.await_count for manager in left.child_model_managers
        ) == (2, 1)
        assert tuple(
            manager.await_count for manager in right.child_model_managers
        ) == (2, 1)
        for runtime in (left, right):
            for lane in runtime.turn.topology.lanes:
                assert (
                    runtime.coordinator.fake_provider_diagnostics(
                        lane.lane_id
                    ).plans
                    == ()
                )
    finally:
        await left.coordinator.close()
        await right.coordinator.close()
        for child in (*left.child_orchestrators, *right.child_orchestrators):
            await child.__aexit__(None, None, None)
        await left.orchestrator.__aexit__(None, None, None)
