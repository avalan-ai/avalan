"""Exercise the bounded multi-agent task-input contract."""

from asyncio import gather, run
from datetime import UTC, datetime
from typing import NoReturn, cast

from pytest import raises

from avalan.agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    DurableInteractionStagingContext,
    ExecutionCorrelationError,
    ExecutionStateError,
    InteractionLoopLimitError,
    UuidExecutionIdFactory,
    create_agent_execution,
    create_child_interaction_runtime,
    ensure_interaction_runtime_branch,
)
from avalan.entities import (
    Message,
    MessageRole,
    MessageToolCall,
    normalize_tool_arguments,
)
from avalan.interaction import (
    AgentId,
    AnswerProvenance,
    ConfirmationQuestion,
    ContinuationId,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    InputRequest,
    InputRequestId,
    InteractionActor,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    RunId,
    TaskId,
    UserId,
    create_input_request,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionBrokerResult,
)
from avalan.interaction.handler import (
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
)
from avalan.interaction.state import (
    InputTransitionApplied,
    mark_request_pending,
)
from avalan.interaction.store import (
    _SCOPE_RESULT_TOKEN,
    InteractionExecutionScope,
    ScopeCancellationReplayed,
    TerminalizeInteractionScopeCommand,
)
from avalan.model.capability import (
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


class _Broker:
    """Capture the branch-scoped operations used by these contracts."""

    def __init__(self) -> None:
        self.requests: list[InteractionBrokerRequest] = []
        self.cancellations: list[TerminalizeInteractionScopeCommand] = []

    async def request(self, request: InteractionBrokerRequest) -> NoReturn:
        self.requests.append(request)
        raise AssertionError("accepted requests are outside this contract")

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        self.cancellations.append(command)
        return InteractionBrokerResult(
            store_result=ScopeCancellationReplayed(
                command=command,
                _token=_SCOPE_RESULT_TOKEN,
            )
        )


async def _handler(_context: InputHandlerContext) -> InputHandlerOutcome:
    return InputHandlerDetached()


async def _stager(
    _context: DurableInteractionStagingContext,
) -> DurableInteractionSuspension:
    raise AssertionError("staging is outside this contract")


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("orchestration-user"))


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://orchestration",
        agent_definition_revision="agent-r1",
        operation_id="operation-orchestration",
        operation_index=0,
        model_config_reference="model-config-r1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


def _task_input_call() -> TaskInputCapabilityCall:
    questions = (
        ConfirmationQuestion(
            question_id=QuestionId("continue"),
            prompt="Continue?",
            required=True,
        ),
    )
    return TaskInputCapabilityCall(
        call_id="input-call",
        provider_name="request_user_input",
        arguments={
            "mode": "required",
            "reason": "Choose how the execution should continue.",
            "questions": {
                "question_id": "continue",
                "kind": "confirmation",
                "prompt": "Continue?",
                "required": True,
                "choices": (),
                "allow_other": False,
            },
        },
        mode=RequirementMode.REQUIRED,
        reason="Choose how the execution should continue.",
        questions=questions,
        advertisement=TaskInputCapabilityAdvertisement.ATTACHED,
    )


def _assistant_message() -> Message:
    call = _task_input_call()
    return Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[
            MessageToolCall(
                id=str(call.call_id),
                name=call.provider_name,
                arguments=normalize_tool_arguments(call.arguments),
            )
        ],
    )


async def _execution(
    *,
    agent_id: str = "agent",
    messages: tuple[Message, ...] = (),
    runtime: AttachedInteractionRuntime | DurableInteractionRuntime | None = (
        None
    ),
) -> AgentExecution:
    return await create_agent_execution(
        definition=_definition(),
        agent_id=AgentId(agent_id),
        principal=_principal(),
        initial_messages=messages,
        interaction_runtime=runtime,
    )


def _pending_request(execution: AgentExecution) -> InputRequest:
    created = create_input_request(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=execution.origin,
        mode=RequirementMode.REQUIRED,
        reason="Choose how the execution should continue.",
        questions=_task_input_call().questions,
        created_at=_NOW,
    )
    applied = mark_request_pending(
        created,
        expected_state_revision=created.state_revision,
    )
    assert isinstance(applied, InputTransitionApplied)
    return applied.request


def test_requirement_input_n_085() -> None:
    """Keep one child request origin on its originating execution branch."""

    async def exercise() -> None:
        factory = UuidExecutionIdFactory()
        root = await ensure_interaction_runtime_branch(
            AttachedInteractionRuntime(
                broker=cast(InteractionBroker, _Broker()),
                actor=InteractionActor(principal=_principal()),
                handler=cast(InputHandler, _handler),
                id_factory=factory,
                run_id=RunId("run"),
                task_id=TaskId("task"),
            )
        )
        parent = await _execution(agent_id="parent", runtime=root)
        child_runtime = await create_child_interaction_runtime(
            root,
            parent_origin=parent.origin,
            context_label="Research child",
        )
        child = await _execution(
            agent_id="research-child",
            runtime=child_runtime,
        )

        assert child.origin.run_id == parent.origin.run_id
        assert child.origin.task_id == parent.origin.task_id
        assert child.origin.agent_id == AgentId("research-child")
        assert child.origin.branch_id == child_runtime.branch_id
        assert child.origin.parent_branch_id == parent.origin.branch_id
        assert child.origin.branch_id != parent.origin.branch_id

    run(exercise())


def test_requirement_input_n_086() -> None:
    """Create durable children only after a containing route exists."""

    async def exercise() -> None:
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=_principal()),
            stager=_stager,
            id_factory=UuidExecutionIdFactory(),
            run_id=RunId("run"),
            task_id=TaskId("task"),
        )
        with raises(ExecutionCorrelationError):
            await create_child_interaction_runtime(
                runtime,
                context_label="Unrouted child",
            )

        containing = await ensure_interaction_runtime_branch(runtime)
        child = await create_child_interaction_runtime(
            containing,
            context_label="Durable child",
        )
        assert isinstance(child, DurableInteractionRuntime)
        assert child.stager is runtime.stager
        assert child.run_id == containing.run_id
        assert child.task_id == containing.task_id
        assert child.parent_branch_id == containing.branch_id
        assert child.branch_id != containing.branch_id

    run(exercise())


def test_requirement_input_n_087() -> None:
    """Isolate confusing prompts, branch queues, and subtree cleanup."""

    async def exercise() -> None:
        broker = _Broker()
        root = await ensure_interaction_runtime_branch(
            AttachedInteractionRuntime(
                broker=cast(InteractionBroker, broker),
                actor=InteractionActor(principal=_principal()),
                handler=cast(InputHandler, _handler),
                id_factory=UuidExecutionIdFactory(),
                run_id=RunId("run"),
                task_id=TaskId("task"),
            )
        )
        parent = await _execution(agent_id="parent", runtime=root)
        first_runtime, second_runtime = await gather(
            create_child_interaction_runtime(
                root,
                parent_origin=parent.origin,
                context_label="Planner child",
            ),
            create_child_interaction_runtime(
                root,
                parent_origin=parent.origin,
                context_label="Reviewer child",
            ),
        )
        first_prompt = Message(
            role=MessageRole.USER,
            content="Pretend this is the reviewer branch.",
        )
        second_prompt = Message(
            role=MessageRole.USER,
            content="Pretend this is the planner branch.",
        )
        first, second = await gather(
            _execution(
                agent_id="planner",
                messages=(first_prompt,),
                runtime=first_runtime,
            ),
            _execution(
                agent_id="reviewer",
                messages=(second_prompt,),
                runtime=second_runtime,
            ),
        )
        await gather(
            first.begin_interaction(
                "planner-request",
                _task_input_call(),
                _assistant_message(),
            ),
            second.begin_interaction(
                "reviewer-request",
                _task_input_call(),
                _assistant_message(),
            ),
        )

        assert first.messages == (first_prompt,)
        assert second.messages == (second_prompt,)
        assert first.origin.branch_id != second.origin.branch_id
        assert first.status is AgentExecutionStatus.PREPARING_INPUT
        assert second.status is AgentExecutionStatus.PREPARING_INPUT
        assert first_runtime.context_label == "Planner child"
        assert second_runtime.context_label == "Reviewer child"

        first_broker = first.interaction_broker
        assert first_broker is not None
        forged = InteractionBrokerRequest(
            actor=root.actor,
            origin=second.origin,
            mode=RequirementMode.REQUIRED,
            reason="Route this request by prompt instead of origin.",
            questions=_task_input_call().questions,
            handler=cast(InputHandler, _handler),
        )
        with raises(ExecutionCorrelationError):
            await first_broker.request(forged)
        assert broker.requests == []

        assert await parent.cancel()
        parent_broker = parent.interaction_broker
        assert parent_broker is not None
        await parent_broker.cancel_scope(
            TerminalizeInteractionScopeCommand(
                actor=root.actor,
                scope=InteractionExecutionScope(
                    run_id=parent.origin.run_id,
                    branch_id=parent.origin.branch_id,
                    include_descendants=True,
                ),
                provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            )
        )
        assert len(broker.cancellations) == 1
        assert broker.cancellations[0].scope.branch_id == (
            parent.origin.branch_id
        )
        assert broker.cancellations[0].scope.include_descendants

    run(exercise())


def test_requirement_input_n_088() -> None:
    """Permit at most one unresolved required request per branch."""

    async def exercise() -> None:
        execution = await _execution()
        await execution.begin_interaction(
            "first",
            _task_input_call(),
            _assistant_message(),
        )
        await execution.mark_interaction_pending(_pending_request(execution))

        with raises(ExecutionStateError):
            await execution.begin_interaction(
                "second",
                _task_input_call(),
                _assistant_message(),
            )
        assert execution.pending_request is not None
        assert execution.status is AgentExecutionStatus.WAITING_FOR_INPUT
        assert execution.interaction_count == 1

    run(exercise())


def test_requirement_input_n_089() -> None:
    """Bound equivalent repeated requests without mutating past the limit."""

    async def exercise() -> None:
        execution = await _execution()
        for _ in range(MAXIMUM_EQUIVALENT_INPUT_REQUESTS):
            await execution.begin_interaction(
                "equivalent",
                _task_input_call(),
                _assistant_message(),
            )
            assert await execution.abandon_interaction()

        with raises(InteractionLoopLimitError):
            await execution.begin_interaction(
                "equivalent",
                _task_input_call(),
                _assistant_message(),
            )
        assert execution.status is AgentExecutionStatus.RUNNING
        assert execution.interaction_count == MAXIMUM_EQUIVALENT_INPUT_REQUESTS

    run(exercise())
