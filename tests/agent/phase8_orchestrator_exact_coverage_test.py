"""Close defensive Phase 8 orchestrator coverage gaps."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import engine_agent_test as engine_test_module
import orchestrator_response_contract_coverage_test as response_test_module
import pytest

import avalan.conversation as conversation
from avalan.agent import Specification
from avalan.agent import orchestrator as orchestrator_module
from avalan.agent.conversation_child import AgentConversationChildBinding
from avalan.agent.conversation_trace import (
    AgentProviderResponseTrace,
    AgentToolOutputTrace,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import ExecutionInputRequiredError
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
)
from avalan.event.manager import EventManager
from avalan.interaction.entities import InputRequiredResult
from avalan.model.call import ModelCallContext
from avalan.tool.manager import ToolManager


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 async coverage checks on asyncio."""
    return "asyncio"


def _binding(
    *,
    lane_id: str = "lane-orchestrator-coverage",
    agent_id: str = "agent-orchestrator-coverage",
    model: str = "coverage-model",
) -> conversation.ProviderLaneBinding:
    """Return one exact synthetic provider binding."""
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type="tests.Phase8OrchestratorProvider",
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint="https://provider.test/v1",
        model_or_deployment=model,
        provider_api_revision=conversation.ProviderApiRevision("api-r1"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-r1"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-r1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-r1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-r1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-r1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId(agent_id),
    )


def _plan() -> conversation.StatelessProviderPlan:
    """Return one empty exact stateless plan."""
    binding = _binding()
    return conversation.StatelessProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=binding.continuation_codec_version,
            items=(),
        ),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=None,
        ),
    )


def _trace() -> orchestrator_module._PreparedAgentConversationTrace:
    """Return one isolated prepared trace."""
    return orchestrator_module._PreparedAgentConversationTrace(
        MagicMock(spec=ToolManager)
    )


@pytest.mark.anyio
async def test_prepared_trace_rejects_invalid_sequence_shapes() -> None:
    """Reject invalid trace types, order, pairing, and terminal output."""
    plan = _plan()
    call = ToolCall(id="call-1", name="lookup", arguments={})
    other = ToolCall(id="call-2", name="lookup", arguments={})
    provider = AgentProviderResponseTrace(text="", calls=(call,))
    tool = AgentToolOutputTrace(call=call, outcome=None)

    trace = _trace()
    with pytest.raises(TypeError):
        await trace.record_provider_response(cast(Any, object()))
    with pytest.raises(TypeError):
        await trace.record_tool_output(cast(Any, object()))
    with pytest.raises(conversation.ConversationValidationError):
        trace.complete(plan, "output", invocation_id="invocation")

    trace._events = [tool]
    with pytest.raises(conversation.ConversationValidationError):
        trace.complete(plan, "output", invocation_id="invocation")

    trace._events = [provider]
    with pytest.raises(conversation.ConversationValidationError):
        trace.complete(plan, "output", invocation_id="invocation")

    trace._events = [
        provider,
        AgentToolOutputTrace(call=other, outcome=None),
    ]
    with pytest.raises(conversation.ConversationValidationError):
        trace.complete(plan, "output", invocation_id="invocation")

    trace._events = [AgentProviderResponseTrace(text="different", calls=())]
    with pytest.raises(conversation.ConversationValidationError):
        trace.complete(plan, "output", invocation_id="invocation")


def test_prepared_trace_item_and_tool_guards_fail_closed() -> None:
    """Reject malformed calls and classify every canonical tool outcome."""
    plan = _plan()
    trace = _trace()
    call = ToolCall(id="call-1", name="lookup", arguments={})
    invalid_arguments = ToolCall(
        id="call-invalid",
        name="lookup",
        arguments=cast(Any, []),
    )
    with pytest.raises(conversation.ConversationValidationError):
        trace._call_item(
            plan,
            invalid_arguments,
            invocation_id="invocation",
            response_index=1,
            provider_index=0,
            order=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        trace._requested_tool(plan, call, None)
    trace._tool.describe_tool_call.return_value = None
    with pytest.raises(conversation.ConversationValidationError):
        trace._requested_tool(
            plan,
            call,
            conversation.ProviderCallId("call-1"),
        )

    call_item = trace._call_item(
        plan,
        call,
        invocation_id="invocation",
        response_index=1,
        provider_index=0,
        order=0,
    )
    missing_id = ToolCall(id=None, name="lookup", arguments={})
    with pytest.raises(conversation.ConversationValidationError):
        trace._tool_output_item(
            plan,
            AgentToolOutputTrace(call=missing_id, outcome=None),
            call_item,
            invocation_id="invocation",
            response_index=1,
            order=1,
        )

    error = ToolCallError(
        id="error-1",
        name=call.name,
        arguments=call.arguments,
        call=call,
        error=RuntimeError("private"),
        message="safe error",
    )
    diagnostic = ToolCallDiagnostic(
        id="diagnostic-1",
        call_id=call.id,
        requested_name=call.name,
        code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        stage=ToolCallDiagnosticStage.RESOLVE,
        message="safe diagnostic",
    )
    for outcome, expected in (
        (error, "safe error"),
        (diagnostic, "safe diagnostic"),
        (None, ""),
    ):
        item = trace._tool_output_item(
            plan,
            AgentToolOutputTrace(call=call, outcome=outcome),
            call_item,
            invocation_id="invocation",
            response_index=1,
            order=1,
        )
        assert item.canonical_input["output"] == expected


@pytest.mark.anyio
async def test_prepared_adapter_rejects_invalid_call_and_topology() -> None:
    """Reject invalid adapter input, ownership drift, and absent parent."""
    case = engine_test_module.EngineAgentRunTestCase()
    agent, _engine, _memory, manager = case._make_agent()
    source_context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="input",
    )
    await agent._run(source_context, "input")
    prepared = manager.await_args.args[0]
    adapter = orchestrator_module._PreparedAgentConversationAdapter(
        engine_agent=agent,
        operation=response_test_module._operation(),
        engine_args={},
        event_manager=EventManager(),
        tool=ToolManager.create_instance(),
        tool_confirm=None,
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
        children=(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await adapter.execute(
            cast(Any, object()),
            "input",
            prepared,
            cast(Any, object()),
        )

    turn = object.__new__(conversation.AgentConversationTurn)
    object.__setattr__(
        turn,
        "topology",
        SimpleNamespace(parent_lanes=(), child_lanes=()),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await adapter.execute(turn, "input", prepared, MagicMock())

    exact_context = replace(
        prepared.context,
        conversation_turn=turn,
        conversation_input="input",
        conversation_invocation_adapter=adapter,
    )
    exact_call = replace(prepared, context=exact_context)
    with pytest.raises(conversation.ConversationCapabilityError):
        await adapter.execute(turn, "input", exact_call, MagicMock())


@pytest.mark.anyio
async def test_orchestrator_public_conversation_guards_fail_closed() -> None:
    """Reject invalid public turn and child-binding arguments immediately."""
    orchestrator = object.__new__(Orchestrator)
    with pytest.raises(TypeError):
        await orchestrator(
            "input",
            conversation_turn=cast(Any, object()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await orchestrator(
            "input",
            conversation_children=cast(Any, (object(),)),
        )
    with pytest.raises(conversation.ConversationValidationError):
        orchestrator._resolve_conversation_children(
            None,
            cast(Any, (object(),)),
            cast(Any, object()),
            cast(Any, object()),
        )


def test_parent_call_and_turn_input_reject_lossy_conversion() -> None:
    """Reject malformed child augmentation and loss-prone turn inputs."""
    turn_input = Orchestrator._conversation_turn_input
    with pytest.raises(conversation.ConversationCapabilityError):
        turn_input(["first", "second"])
    assert turn_input(["single"]) == "single"
    assert turn_input("direct") == "direct"
    with pytest.raises(conversation.ConversationCapabilityError):
        turn_input(Message(role=MessageRole.ASSISTANT, content="answer"))
    text = MessageContentText(type="text", text="content text")
    assert (
        turn_input(Message(role=MessageRole.USER, content=text)) == text.text
    )
    assert (
        turn_input(Message(role=MessageRole.USER, content=[text])) == text.text
    )
    unsupported = MessageContentFile(type="file", file={"file_id": "file"})
    with pytest.raises(conversation.ConversationCapabilityError):
        turn_input(Message(role=MessageRole.USER, content=unsupported))
    with pytest.raises(conversation.ConversationCapabilityError):
        turn_input(cast(Any, object()))
    with pytest.raises(conversation.ConversationCapabilityError):
        turn_input("   ")


@pytest.mark.anyio
async def test_parent_call_rejects_invalid_input_and_child_projection() -> (
    None
):
    """Reject non-list model input and malformed canonical child output."""
    case = engine_test_module.EngineAgentRunTestCase()
    agent, _engine, _memory, manager = case._make_agent()
    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="input",
    )
    await agent._run(context, "input")
    prepared = manager.await_args.args[0]
    child = conversation.VisibleTranscriptEntry(
        role=conversation.VisibleTranscriptRole.ASSISTANT,
        content="child output",
    )
    with pytest.raises(conversation.ConversationValidationError):
        orchestrator_module._PreparedAgentConversationAdapter._parent_call_with_children(
            prepared,
            "input",
            (child,),
        )

    user = Message(role=MessageRole.USER, content="input")
    list_call = replace(
        prepared,
        operation=replace(prepared.operation, input=[user]),
        context=replace(prepared.context, input=[user]),
    )
    with pytest.raises(conversation.ConversationValidationError):
        orchestrator_module._PreparedAgentConversationAdapter._parent_call_with_children(
            list_call,
            "different input",
            (child,),
        )


@pytest.mark.anyio
async def test_dispatch_preserves_input_required_boundary() -> None:
    """Propagate input-required control flow without failure settlement."""
    required = ExecutionInputRequiredError(
        InputRequiredResult(
            request_id="request",
            continuation_id="continuation",
            detached_resumption_available=False,
        )
    )

    class SuspendingAgent:
        async def __call__(self, context: object) -> object:
            del context
            raise required

    orchestrator = object.__new__(Orchestrator)
    with pytest.raises(ExecutionInputRequiredError) as raised:
        await orchestrator._dispatch_execution(
            engine_agent=cast(EngineAgent, SuspendingAgent()),
            operation=cast(Any, object()),
            engine_args={},
            execution=cast(Any, object()),
            context=cast(Any, object()),
            messages="input",
            started=0.0,
            tool_confirm=None,
            agent_id=uuid4(),
            participant_id=None,
            session_id=None,
            block_repeated_tool_calls=False,
            maximum_tool_cycles=10,
        )
    assert raised.value is required


def _turn_with_topology(
    parent_lanes: tuple[object, ...],
    child_lanes: tuple[object, ...],
) -> conversation.AgentConversationTurn:
    """Return an exact turn shell for adapter closure tests."""
    turn = object.__new__(conversation.AgentConversationTurn)
    object.__setattr__(
        turn,
        "topology",
        SimpleNamespace(
            parent_lanes=parent_lanes,
            child_lanes=child_lanes,
        ),
    )
    object.__setattr__(turn, "checkpoint_id", "checkpoint-adapter")
    return turn


async def _prepared_model_call() -> tuple[EngineAgent, object]:
    """Return one real engine agent and its prepared model call."""
    case = engine_test_module.EngineAgentRunTestCase()
    agent, _engine, _memory, manager = case._make_agent()
    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="input",
    )
    await agent._run(context, "input")
    return agent, manager.await_args.args[0]


def _adapter_call(
    adapter: orchestrator_module._PreparedAgentConversationAdapter,
    turn: conversation.AgentConversationTurn,
    prepared: Any,
) -> Any:
    """Bind one prepared model call to the exact adapter and turn."""
    context = replace(
        prepared.context,
        conversation_turn=turn,
        conversation_input="input",
        conversation_invocation_adapter=adapter,
    )
    return replace(prepared, context=context)


@pytest.mark.anyio
async def test_child_adapter_rejects_raw_output_and_closes_unowned_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid child output and close a failed pre-wrapper response."""
    parent_engine, prepared = await _prepared_model_call()
    child_binding = _binding(
        lane_id="lane-child",
        agent_id=str(uuid4()),
    )
    parent_lane = SimpleNamespace(
        lane_id=conversation.ProviderLaneId("lane-parent"),
        binding=_binding(lane_id="lane-parent"),
    )
    child_lane = SimpleNamespace(
        lane_id=child_binding.lane_id,
        binding=child_binding,
    )
    tool = ToolManager.create_instance()
    child_orchestrator = SimpleNamespace(
        id=uuid4(),
        tool=tool,
        event_manager=EventManager(),
    )

    class ChildEngine:
        result: object = object()
        acknowledged: list[object] = []

        async def __call__(self, context: object) -> object:
            del context
            return self.result

        def acknowledge_provider_handoff(self, response: object) -> None:
            self.acknowledged.append(response)

    child_engine = ChildEngine()
    resolved_child = orchestrator_module._ResolvedAgentConversationChild(
        lane=cast(Any, child_lane),
        binding=cast(
            Any,
            SimpleNamespace(orchestrator=child_orchestrator),
        ),
        engine_agent=cast(EngineAgent, child_engine),
        operation=response_test_module._operation(),
        engine_args={},
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
    )
    adapter = orchestrator_module._PreparedAgentConversationAdapter(
        engine_agent=parent_engine,
        operation=response_test_module._operation(),
        engine_args={},
        event_manager=EventManager(),
        tool=ToolManager.create_instance(),
        tool_confirm=None,
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
        children=(resolved_child,),
    )
    turn = _turn_with_topology((parent_lane,), (child_lane,))
    model_call = _adapter_call(adapter, turn, prepared)
    child_plan = conversation.StatelessProviderPlan(
        binding=child_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=child_binding.lane_id,
            normalization_version=child_binding.continuation_codec_version,
            items=(),
        ),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=None,
        ),
    )

    async def dispatch_child_only(
        self: conversation.AgentConversationTurn,
        input: str,
        **kwargs: object,
    ) -> Any:
        del self, input
        invocations = cast(Any, kwargs["lane_invocations"])
        return await invocations[0].dispatch(child_plan)

    monkeypatch.setattr(
        conversation.AgentConversationTurn,
        "execute",
        dispatch_child_only,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await adapter.execute(turn, "input", model_call, MagicMock())

    raw = response_test_module._text_response()
    child_engine.result = raw

    class RejectingResponse:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("wrapper rejected")

    monkeypatch.setattr(
        orchestrator_module,
        "OrchestratorResponse",
        RejectingResponse,
    )
    with pytest.raises(RuntimeError, match="wrapper rejected"):
        await adapter.execute(turn, "input", model_call, MagicMock())
    assert child_engine.acknowledged == [raw]
    assert raw.cleanup_complete


@pytest.mark.anyio
async def test_parent_adapter_rejects_missing_children_and_bad_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject incomplete child results, empty parent input, and bad output."""
    parent_engine, prepared = await _prepared_model_call()
    parent_lane = SimpleNamespace(
        lane_id=conversation.ProviderLaneId("lane-parent"),
        binding=_binding(lane_id="lane-parent"),
    )
    child_binding = _binding(lane_id="lane-child", agent_id=str(uuid4()))
    child_lane = SimpleNamespace(
        lane_id=child_binding.lane_id,
        binding=child_binding,
    )
    child = orchestrator_module._ResolvedAgentConversationChild(
        lane=cast(Any, child_lane),
        binding=cast(
            Any,
            SimpleNamespace(
                orchestrator=SimpleNamespace(
                    id=uuid4(),
                    tool=ToolManager.create_instance(),
                    event_manager=EventManager(),
                )
            ),
        ),
        engine_agent=parent_engine,
        operation=response_test_module._operation(),
        engine_args={},
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
    )
    child_adapter = orchestrator_module._PreparedAgentConversationAdapter(
        engine_agent=parent_engine,
        operation=response_test_module._operation(),
        engine_args={},
        event_manager=EventManager(),
        tool=ToolManager.create_instance(),
        tool_confirm=None,
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
        children=(child,),
    )
    child_turn = _turn_with_topology((parent_lane,), (child_lane,))
    child_call = _adapter_call(child_adapter, child_turn, prepared)
    parent_plan = _plan()

    async def parent_only(
        self: conversation.AgentConversationTurn,
        input: str,
        **kwargs: object,
    ) -> Any:
        del self, input
        invocations = cast(Any, kwargs["lane_invocations"])
        return await invocations[-1].dispatch(parent_plan)

    monkeypatch.setattr(
        conversation.AgentConversationTurn,
        "execute",
        parent_only,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await child_adapter.execute(
            child_turn,
            "input",
            child_call,
            MagicMock(),
        )

    adapter = orchestrator_module._PreparedAgentConversationAdapter(
        engine_agent=parent_engine,
        operation=response_test_module._operation(),
        engine_args={},
        event_manager=EventManager(),
        tool=ToolManager.create_instance(),
        tool_confirm=None,
        block_repeated_tool_calls=False,
        maximum_tool_cycles=10,
        children=(),
    )
    turn = _turn_with_topology((parent_lane,), ())
    call = _adapter_call(adapter, turn, prepared)
    empty_call = replace(
        call,
        operation=replace(call.operation, input=None),
        context=replace(call.context, input=None),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await adapter.execute(turn, "input", empty_call, MagicMock())

    async def invalid_model(call: object) -> object:
        del call
        return object()

    with pytest.raises(conversation.ConversationValidationError):
        await adapter.execute(turn, "input", call, invalid_model)

    raw = response_test_module._text_response()

    async def valid_model(call: object) -> object:
        del call
        return raw

    class RejectingResponse:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("parent wrapper rejected")

    monkeypatch.setattr(
        orchestrator_module,
        "OrchestratorResponse",
        RejectingResponse,
    )
    with pytest.raises(RuntimeError, match="parent wrapper rejected"):
        await adapter.execute(turn, "input", call, valid_model)
    assert raw.cleanup_complete


@pytest.mark.anyio
async def test_conversation_child_resolution_guards_fail_closed() -> None:
    """Reject parent topology drift and invalid child engine arguments."""
    orchestrator = object.__new__(Orchestrator)
    turn = _turn_with_topology((), ())
    with pytest.raises(conversation.ConversationBindingDriftError):
        orchestrator._resolve_conversation_children(
            turn,
            (),
            cast(Any, object()),
            cast(Any, object()),
        )

    parent_id = uuid4()
    parent_lane = SimpleNamespace(
        agent_id=conversation.ConversationAgentId(str(parent_id)),
        binding=_binding(
            lane_id="parent-lane",
            agent_id=str(parent_id),
            model="m",
        ),
    )
    turn = _turn_with_topology((parent_lane,), ())
    object.__setattr__(orchestrator, "_id", uuid4())
    parent_engine = SimpleNamespace(
        id=uuid4(),
        engine=SimpleNamespace(model_id="m"),
    )
    parent_operation = replace(
        response_test_module._operation(),
        environment=replace(
            response_test_module._operation().environment,
            engine_uri=replace(
                response_test_module._operation().environment.engine_uri,
                model_id="m",
            ),
        ),
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        orchestrator._resolve_conversation_children(
            turn,
            (),
            cast(Any, parent_engine),
            parent_operation,
        )

    parent_agent, _prepared = await _prepared_model_call()
    child_agent, _prepared = await _prepared_model_call()
    parent_operation = replace(
        response_test_module._operation(),
        environment=replace(
            response_test_module._operation().environment,
            engine_uri=replace(
                response_test_module._operation().environment.engine_uri,
                model_id="m",
            ),
        ),
    )
    child_operation = replace(
        response_test_module._operation(),
        environment=replace(
            response_test_module._operation().environment,
            engine_uri=replace(
                response_test_module._operation().environment.engine_uri,
                model_id="m",
            ),
        ),
    )

    class ChildOrchestrator:
        id = child_agent.id
        operations = [child_operation]
        tool = ToolManager.create_instance()
        event_manager = EventManager()

        def engine_agent_for_operation(
            self, operation_index: int
        ) -> EngineAgent:
            assert operation_index == 0
            return child_agent

        def conversation_engine_args(self) -> object:
            return object()

    child_orchestrator = ChildOrchestrator()
    child_binding = _binding(
        lane_id="child-lane",
        agent_id=str(child_agent.id),
        model="m",
    )
    child_lane = SimpleNamespace(
        lane_id=child_binding.lane_id,
        agent_id=conversation.ConversationAgentId(str(child_agent.id)),
        binding=child_binding,
    )
    child_reference = AgentConversationChildBinding(
        lane_id=child_binding.lane_id,
        orchestrator=child_orchestrator,
    )
    parent_lane = SimpleNamespace(
        lane_id=conversation.ProviderLaneId("parent-lane"),
        agent_id=conversation.ConversationAgentId(str(parent_agent.id)),
        binding=_binding(
            lane_id="parent-lane",
            agent_id=str(parent_agent.id),
            model="m",
        ),
    )
    exact_turn = _turn_with_topology((parent_lane,), (child_lane,))
    object.__setattr__(orchestrator, "_id", parent_agent.id)
    with pytest.raises(conversation.ConversationValidationError):
        orchestrator._resolve_conversation_children(
            exact_turn,
            (child_reference,),
            parent_agent,
            parent_operation,
        )
