"""Close defensive Phase 8 agent continuation coverage gaps."""

from asyncio import CancelledError, Event, create_task
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import engine_agent_test as engine_test_module
import orchestrator_response_contract_coverage_test as response_test_module
import pytest
from durable_continuation_resume_test import (
    _a2a_checkpoint,
    _admit,
    _conversation_harness,
    _harness,
)
from execution_coverage_regression_test import _created_request, _origin

import avalan.conversation as conversation
from avalan.agent import Specification
from avalan.agent import continuation as continuation_module
from avalan.agent import engine as engine_module
from avalan.agent.continuation import (
    AgentConversationContinuationResolver,
    AgentConversationContinuationResult,
    DurableAgentContinuationResumer,
    ResolvedAgentConversationContinuation,
)
from avalan.agent.continuation_stager import PortableAgentContinuationStager
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    DurableInteractionRuntime,
    ExecutionCorrelationError,
    ExecutionInputRequiredError,
    create_agent_execution,
)
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.conversation import security
from avalan.entities import GenerationSettings, ToolCall
from avalan.interaction.continuation import (
    PortableConversationCheckpointReference,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import InputRequiredResult
from avalan.interaction.error import InputValidationError
from avalan.interaction.policy import InteractionActor
from avalan.model.call import ModelCallContext
from avalan.model.capability import ModelCapabilityCatalog
from avalan.task.durable_agent import (
    DurableAgentTaskHost,
    TaskDurableAgentRuntime,
)
from avalan.task.targets import agent as target_agent_module


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 async coverage checks on asyncio."""
    return "asyncio"


def _invalid() -> Any:
    """Return one deliberately invalid runtime contract value."""
    return object()


@pytest.mark.anyio
async def test_conversation_result_and_resolver_guards_fail_closed() -> None:
    """Reject invalid continuation result, resolver, and apply contracts."""
    with pytest.raises(InputValidationError):
        AgentConversationContinuationResult(
            checkpoint=_invalid(),
            output=None,
        )

    harness = _conversation_harness()
    admission = await _admit(harness)
    command = admission.command
    resolved = command.resolved_conversation
    assert resolved is not None

    with pytest.raises(InputValidationError):
        replace(resolved, checkpoint=_invalid())
    with pytest.raises(InputValidationError):
        replace(resolved, continuation_reference=_invalid())
    with pytest.raises(InputValidationError):
        replace(resolved, apply_result=lambda call, result: result)

    async def invalid_apply(
        call: object,
        result: object,
    ) -> Any:
        del call, result
        return _invalid()

    invalid_resolved = replace(resolved, apply_result=invalid_apply)
    task_input_call = command.task_input_call
    correlated_result = command.correlated_result
    assert task_input_call is not None
    assert correlated_result is not None
    with pytest.raises(InputValidationError):
        await invalid_resolved.apply(task_input_call, correlated_result)

    with pytest.raises(InputValidationError):
        AgentConversationContinuationResolver(
            resolve_continuation=lambda continuation, digest: continuation,
        )

    async def invalid_resolve(
        continuation: object,
        digest: str,
    ) -> Any:
        del continuation, digest
        return _invalid()

    resolver = AgentConversationContinuationResolver(
        resolve_continuation=invalid_resolve,
    )
    with pytest.raises(InputValidationError):
        await resolver.resolve(
            command.continuation,
            command.conversation_continuation_digest or "",
        )


@pytest.mark.anyio
async def test_conversation_resume_authority_guards_fail_closed() -> None:
    """Reject missing, mixed, and mismatched conversation replay authority."""
    harness = _harness()
    with pytest.raises(InputValidationError):
        DurableAgentContinuationResumer(
            harness.store,
            harness.resumer._resolver,
            conversation_resolver=_invalid(),
        )

    missing_resolver = _conversation_harness()
    missing_resolver = replace(
        missing_resolver,
        resumer=DurableAgentContinuationResumer(
            missing_resolver.store,
            missing_resolver.resumer._resolver,
        ),
    )
    with pytest.raises(InputValidationError):
        await _admit(missing_resolver)

    mixed = _harness()
    mixed_continuation = replace(
        mixed.record.continuation,
        version=2,
        conversation_checkpoint_reference=PortableConversationCheckpointReference(
            checkpoint_id="conversation-checkpoint",
            execution_segment_id="conversation-segment",
        ),
    )
    mixed.store.continuation = mixed_continuation
    mixed = replace(
        mixed,
        record=replace(mixed.record, continuation=mixed_continuation),
    )
    with pytest.raises(InputValidationError):
        await _admit(mixed)


@pytest.mark.anyio
async def test_resume_command_rejects_a2a_and_conversation_replay() -> None:
    """Reject a command that mixes downstream A2A and conversation replay."""
    harness = _conversation_harness()
    command = (await _admit(harness)).command
    with pytest.raises(InputValidationError):
        replace(
            command,
            task_input_call=None,
            correlated_result=None,
            a2a_checkpoint=_a2a_checkpoint(command.request),
        )


@pytest.mark.anyio
async def test_resolved_conversation_validation_rejects_all_bad_shapes() -> (
    None
):
    """Reject absent, mixed, unresolved, and mismatched replay state."""
    provider_command = (await _admit(_harness())).command
    with pytest.raises(InputValidationError):
        continuation_module._validate_resolved_conversation(
            provider_command.continuation,
            cast(ResolvedAgentConversationContinuation, _invalid()),
            expected_digest="0" * 64,
        )

    missing = replace(provider_command.continuation, provider_snapshot=None)
    with pytest.raises(InputValidationError):
        continuation_module._validate_resolved_conversation(
            missing,
            None,
            expected_digest=None,
        )

    conversation_command = (await _admit(_conversation_harness())).command
    with pytest.raises(InputValidationError):
        continuation_module._validate_resolved_conversation(
            conversation_command.continuation,
            None,
            expected_digest=None,
        )

    resolved = conversation_command.resolved_conversation
    digest = conversation_command.conversation_continuation_digest
    assert resolved is not None
    assert digest is not None
    mismatched_checkpoint = object.__new__(type(resolved.checkpoint))
    object.__setattr__(
        mismatched_checkpoint,
        "identity",
        replace(
            resolved.checkpoint.identity,
            checkpoint_id=type(resolved.checkpoint.identity.checkpoint_id)(
                "wrong-checkpoint"
            ),
        ),
    )
    object.__setattr__(
        mismatched_checkpoint,
        "kind",
        resolved.checkpoint.kind,
    )
    object.__setattr__(
        mismatched_checkpoint,
        "lifecycle",
        resolved.checkpoint.lifecycle,
    )
    object.__setattr__(
        mismatched_checkpoint,
        "authority",
        resolved.checkpoint.authority,
    )
    with pytest.raises(InputValidationError):
        continuation_module._validate_resolved_conversation(
            conversation_command.continuation,
            replace(resolved, checkpoint=mismatched_checkpoint),
            expected_digest=digest,
        )


def test_execution_conversation_suspension_guards_fail_closed() -> None:
    """Require exact durable state for each conversation suspension marker."""
    created = _created_request(_origin())
    required = InputRequiredResult(
        request_id=created.request_id,
        continuation_id=created.continuation_id,
        detached_resumption_available=True,
    )
    durable = object.__new__(DurableInteractionSuspension)
    object.__setattr__(
        durable,
        "command",
        cast(Any, type("Command", (), {"request": created})()),
    )

    with pytest.raises(ExecutionCorrelationError):
        ExecutionInputRequiredError(
            required,
            request=created,
            checkpoint_id="checkpoint",
        )
    with pytest.raises(ExecutionCorrelationError):
        ExecutionInputRequiredError(
            required,
            request=created,
            checkpoint_id="checkpoint",
            conversation_unit=_invalid(),
        )
    object.__setattr__(
        durable,
        "continuation",
        cast(
            Any,
            type(
                "Continuation",
                (),
                {"conversation_checkpoint_reference": None},
            )(),
        ),
    )
    with pytest.raises(ExecutionCorrelationError):
        ExecutionInputRequiredError(
            required,
            request=created,
            durable=durable,
            checkpoint_id="checkpoint",
            conversation_unit=_invalid(),
        )


@pytest.mark.anyio
async def test_task_conversation_runtime_guards_and_apply_contract() -> None:
    """Reject invalid task runtime state and coordinator output."""
    with pytest.raises(TypeError):
        TaskDurableAgentRuntime(
            store=cast(Any, object()),
            coordinator=cast(Any, object()),
            authority=cast(Any, object()),
        )

    command = (await _admit(_conversation_harness())).command
    resolved = command.resolved_conversation
    digest = command.conversation_continuation_digest
    assert resolved is not None
    assert digest is not None

    class Store:
        def __init__(self) -> None:
            self.checkpoint: object = resolved.checkpoint
            self.reference: object = resolved.continuation_reference

        async def load(self, checkpoint_id: object, authority: object) -> Any:
            del checkpoint_id, authority
            return self.checkpoint

        async def load_continuation_reference(
            self,
            checkpoint_id: object,
            authority: object,
        ) -> Any:
            del checkpoint_id, authority
            return self.reference

    class Coordinator:
        result: object = _invalid()

        async def resume_structured_input(
            self,
            checkpoint: object,
            call: object,
            result: object,
        ) -> Any:
            del checkpoint, call, result
            return self.result

    store = Store()
    coordinator = Coordinator()
    runtime = TaskDurableAgentRuntime(
        store=store,
        coordinator=coordinator,
        authority=resolved.checkpoint.authority,
    )

    with pytest.raises(TypeError):
        await runtime.resolve_continuation(
            _harness().record.continuation,
            digest,
        )

    store.reference = replace(
        resolved.continuation_reference,
        digest=type(resolved.continuation_reference.digest)("0" * 64),
    )
    with pytest.raises(TypeError):
        await runtime.resolve_continuation(command.continuation, digest)

    store.reference = resolved.continuation_reference
    continuation = await runtime.resolve_continuation(
        command.continuation,
        digest,
    )
    task_input_call = command.task_input_call
    correlated_result = command.correlated_result
    assert task_input_call is not None
    assert correlated_result is not None
    with pytest.raises(TypeError):
        await continuation.apply(task_input_call, correlated_result)


def test_task_host_conversation_owner_guards_fail_closed() -> None:
    """Require exactly one typed owner for task conversation replay."""
    continuation_store = SimpleNamespace(
        get_task_continuation_record=lambda: None
    )
    arguments = {
        "orchestrator_loader": cast(Any, object()),
        "stack": cast(Any, object()),
        "allowed_roots": (),
        "continuation_store": continuation_store,
    }
    with pytest.raises(TypeError):
        DurableAgentTaskHost(
            **arguments,
            conversation_resolver=cast(Any, object()),
        )
    with pytest.raises(TypeError):
        DurableAgentTaskHost(
            **arguments,
            conversation_runtime=cast(Any, object()),
        )

    async def resolve(
        continuation: object,
        digest: str,
    ) -> Any:
        del continuation, digest
        return _invalid()

    resolver = AgentConversationContinuationResolver(
        resolve_continuation=resolve,
    )
    runtime = object.__new__(TaskDurableAgentRuntime)
    with pytest.raises(TypeError):
        DurableAgentTaskHost(
            **arguments,
            conversation_resolver=resolver,
            conversation_runtime=runtime,
        )
    with pytest.raises(TypeError):
        DurableAgentTaskHost(
            **arguments,
            conversation_runtime=runtime,
        )


@pytest.mark.anyio
async def test_task_target_rejects_incomplete_conversation_markers() -> None:
    """Reject incomplete conversation and stray non-conversation markers."""
    conversation_command = (await _admit(_conversation_harness())).command
    provider_command = (await _admit(_harness())).command

    def error_for(continuation: object) -> ExecutionInputRequiredError:
        error = ExecutionInputRequiredError.__new__(
            ExecutionInputRequiredError
        )
        error.result = InputRequiredResult(
            request_id=conversation_command.request.request_id,
            continuation_id=conversation_command.request.continuation_id,
            detached_resumption_available=True,
        )
        error.durable = cast(
            DurableInteractionSuspension,
            SimpleNamespace(continuation=continuation),
        )
        error.checkpoint_id = "unexpected-checkpoint"
        error.conversation_unit = None
        return error

    with pytest.raises(ExecutionCorrelationError):
        target_agent_module._suspended_agent_target_outcome(
            error_for(conversation_command.continuation),
            interaction_runtime=None,
            resumed=True,
        )
    with pytest.raises(ExecutionCorrelationError):
        target_agent_module._suspended_agent_target_outcome(
            error_for(provider_command.continuation),
            interaction_runtime=None,
            resumed=True,
        )


def test_engine_durable_response_state_rejects_invalid_values() -> None:
    """Reject untyped settings and negative durable tool-loop counts."""
    with pytest.raises(TypeError):
        engine_module._ConversationDurableResponseState(
            cast(Any, object()),
            0,
        )
    with pytest.raises(TypeError):
        engine_module._ConversationDurableResponseState({}, -1)


@pytest.mark.anyio
async def test_engine_conversation_dispatch_guards_exact_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise direct, adapter, drift, and suspension conversation paths."""
    case = engine_test_module.EngineAgentRunTestCase()
    turn = object.__new__(conversation.AgentConversationTurn)

    async def execute(
        self: conversation.AgentConversationTurn,
        input: str,
        **kwargs: object,
    ) -> object:
        del self, input, kwargs
        return SimpleNamespace(output="conversation-output")

    monkeypatch.setattr(conversation.AgentConversationTurn, "execute", execute)
    agent, _engine, _memory, _manager = case._make_agent()
    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="conversation-input",
        capability=ModelCapabilityCatalog.create(),
        conversation_turn=turn,
        conversation_input="conversation-input",
    )
    await agent._run(context, "conversation-input")

    class InvalidAdapter:
        async def execute(
            self,
            active_turn: object,
            input: str,
            prepared: object,
            invoke_model: Any,
        ) -> object:
            del active_turn, input, prepared
            return await invoke_model(object())

    invalid_context = replace(
        context,
        conversation_invocation_adapter=cast(Any, InvalidAdapter()),
    )
    invalid_agent, _engine, _memory, _manager = case._make_agent()
    with pytest.raises(RuntimeError):
        await invalid_agent._run(invalid_context, "conversation-input")

    class DriftAdapter:
        async def execute(
            self,
            active_turn: object,
            input: str,
            prepared: object,
            invoke_model: Any,
        ) -> object:
            del active_turn, input
            call = cast(Any, prepared)
            drifted_uri = replace(call.engine_uri, model_id="drifted-model")
            return await invoke_model(replace(call, engine_uri=drifted_uri))

    drift_context = replace(
        context,
        conversation_invocation_adapter=cast(Any, DriftAdapter()),
    )
    drift_agent, _engine, _memory, _manager = case._make_agent()
    with pytest.raises(RuntimeError):
        await drift_agent._run(drift_context, "conversation-input")

    boundary = conversation.AgentConversationSuspensionBoundary.__new__(
        conversation.AgentConversationSuspensionBoundary
    )
    RuntimeError.__init__(boundary, "coverage suspension")

    async def suspend(
        self: conversation.AgentConversationTurn,
        input: str,
        **kwargs: object,
    ) -> object:
        del self, input, kwargs
        raise boundary

    async def staged(
        self: object,
        staged_context: object,
        settings: object,
        staged_boundary: object,
    ) -> None:
        del self, staged_context, settings
        assert staged_boundary is boundary

    monkeypatch.setattr(conversation.AgentConversationTurn, "execute", suspend)
    suspended_agent, _engine, _memory, _manager = case._make_agent()
    monkeypatch.setattr(
        type(suspended_agent),
        "_stage_conversation_input_required",
        staged,
    )
    with pytest.raises(AssertionError):
        await suspended_agent._run(context, "conversation-input")


@pytest.mark.anyio
async def test_engine_conversation_staging_guards_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject staging drift and roll back a failed atomic handoff."""
    case = engine_test_module.EngineAgentRunTestCase()
    agent, _engine, _memory, _manager = case._make_agent()
    basic_context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="conversation-input",
    )
    invalid_boundary = cast(
        conversation.AgentConversationSuspensionBoundary,
        object(),
    )
    with pytest.raises(TypeError):
        await agent._stage_conversation_input_required(
            basic_context,
            cast(Any, object()),
            invalid_boundary,
        )
    with pytest.raises(RuntimeError):
        await agent._stage_conversation_input_required(
            basic_context,
            GenerationSettings(),
            invalid_boundary,
        )

    command = (await _admit(_conversation_harness())).command
    decoded = command.task_input_call
    assert decoded is not None
    captured: dict[str, object] = {}
    real_stager = PortableAgentContinuationStager(
        clock=lambda: datetime(2026, 8, 3, 12, tzinfo=UTC)
    )

    async def stager(
        request_spec: object,
        **kwargs: object,
    ) -> DurableInteractionSuspension:
        durable = await real_stager(cast(Any, request_spec), **kwargs)
        captured.update(
            request_spec=request_spec,
            staging=kwargs["staging"],
            durable=durable,
        )
        return durable

    origin = _origin()
    runtime = DurableInteractionRuntime(
        actor=InteractionActor(principal=origin.principal),
        stager=stager,
    )
    execution = await create_agent_execution(
        definition=replace(
            origin.definition,
            capability_revision=str(
                command.continuation.revision_binding.capability_revision
            ),
        ),
        agent_id=origin.agent_id,
        principal=origin.principal,
        initial_messages=(),
        interaction_runtime=runtime,
    )
    turn = object.__new__(conversation.AgentConversationTurn)

    class Capability:
        revision_binding: object = command.continuation.revision_binding
        support = SimpleNamespace(
            provider_family=command.continuation.revision_binding.provider_family
        )
        decoded_result: object = decoded

        def decode_call(self, call: object, **kwargs: object) -> object:
            del call, kwargs
            return self.decoded_result

    capability = Capability()
    context = ModelCallContext(
        specification=Specification(role=None, goal=None),
        input="conversation-input",
        capability=cast(ModelCapabilityCatalog, capability),
        execution=execution,
        conversation_turn=turn,
        conversation_input="conversation-input",
    )
    checkpoint = SimpleNamespace(
        identity=SimpleNamespace(
            checkpoint_id="conversation-checkpoint",
            execution_segment_id="conversation-segment",
        ),
        content=SimpleNamespace(
            execution_segments=(
                SimpleNamespace(
                    phase=SimpleNamespace(value="provider_response")
                ),
            )
        ),
    )
    boundary = SimpleNamespace(
        call=SimpleNamespace(
            canonical_input={
                "name": decoded.provider_name,
                "arguments": "{}",
            },
            call_id=decoded.call_id,
        ),
        request=SimpleNamespace(arguments=decoded.arguments),
        tool=SimpleNamespace(call_id=decoded.call_id),
        checkpoint=checkpoint,
    )

    no_binding = replace(
        context,
        capability=cast(
            ModelCapabilityCatalog,
            SimpleNamespace(revision_binding=None),
        ),
    )
    with pytest.raises(RuntimeError):
        await agent._stage_conversation_input_required(
            no_binding,
            GenerationSettings(),
            cast(Any, boundary),
        )

    malformed = SimpleNamespace(
        request=boundary.request,
        tool=boundary.tool,
        checkpoint=boundary.checkpoint,
        call=SimpleNamespace(canonical_input={}, call_id=None),
    )
    with pytest.raises(ExecutionCorrelationError):
        await agent._stage_conversation_input_required(
            context,
            GenerationSettings(),
            cast(Any, malformed),
        )

    capability.decoded_result = object()
    with pytest.raises(ExecutionCorrelationError):
        await agent._stage_conversation_input_required(
            context,
            GenerationSettings(),
            cast(Any, boundary),
        )
    capability.decoded_result = decoded

    rolled_back = False

    class Unit:
        async def rollback(self) -> None:
            nonlocal rolled_back
            rolled_back = True

    unit = Unit()

    async def stage_suspension(
        self: conversation.AgentConversationTurn,
        staged_checkpoint: object,
        reference: object,
    ) -> object:
        del self, reference
        assert staged_checkpoint is checkpoint
        return unit

    async def fail_required(
        self: AgentExecution,
        request: object,
        result: object,
    ) -> None:
        del self, request, result
        raise RuntimeError("handoff failed")

    monkeypatch.setattr(
        conversation.AgentConversationTurn,
        "stage_structured_input_suspension",
        stage_suspension,
    )
    monkeypatch.setattr(
        AgentExecution,
        "stage_durable_input_required",
        fail_required,
    )
    with pytest.raises(RuntimeError, match="handoff failed"):
        await agent._stage_conversation_input_required(
            context,
            GenerationSettings(),
            cast(Any, boundary),
        )
    assert rolled_back
    assert execution.status is AgentExecutionStatus.RUNNING

    engine_module.EngineAgent._validate_conversation_durable_staging(
        cast(Any, captured["request_spec"]),
        cast(DurableInteractionSuspension, captured["durable"]),
        staging=cast(Any, captured["staging"]),
    )
    with pytest.raises(ExecutionCorrelationError):
        engine_module.EngineAgent._validate_conversation_durable_staging(
            replace(
                cast(Any, captured["request_spec"]),
                reason="changed reason",
            ),
            cast(DurableInteractionSuspension, captured["durable"]),
            staging=cast(Any, captured["staging"]),
        )
    with pytest.raises(TypeError):
        engine_module.EngineAgent._validate_conversation_durable_staging(
            cast(Any, captured["request_spec"]),
            cast(Any, object()),
            staging=cast(Any, captured["staging"]),
        )


@pytest.mark.anyio
async def test_orchestrator_response_trace_sinks_must_be_async() -> None:
    """Reject malformed, synchronous provider and tool trace sinks."""
    base = response_test_module._response()
    with pytest.raises(TypeError):
        OrchestratorResponse(
            base._input,
            base._response,
            base._engine_agent,
            base._operation,
            base._engine_args,
            base._context,
            enable_tool_parsing=False,
            conversation_trace_sink=cast(Any, object()),
        )

    sink = SimpleNamespace(
        record_provider_response=lambda trace: None,
        record_tool_output=lambda trace: None,
    )
    base._conversation_trace_sink = cast(Any, sink)
    with pytest.raises(TypeError):
        await base._record_conversation_provider_response("output", ())
    with pytest.raises(TypeError):
        await base._execute_tool_call_with_lifecycle(
            ToolCall(id="trace-call", name="coverage", arguments={}),
            confirm=False,
            abort_on_reject=False,
            emit_ready=False,
            planned_index=0,
            finish_stream_on_error=False,
        )


@pytest.mark.anyio
async def test_orchestrator_response_awaits_conversation_trace_sinks() -> None:
    """Await accepted trace sinks before provider and tool flow continues."""
    base = response_test_module._response()
    provider_traces: list[object] = []
    tool_traces: list[object] = []

    async def record_provider_response(trace: object) -> None:
        """Retain the provider trace only after its await boundary runs."""
        provider_traces.append(trace)

    async def record_tool_output(trace: object) -> None:
        """Retain the tool trace only after its await boundary runs."""
        tool_traces.append(trace)

    base._conversation_trace_sink = SimpleNamespace(
        record_provider_response=record_provider_response,
        record_tool_output=record_tool_output,
    )

    await base._record_conversation_provider_response("output", ())
    outcome = await base._execute_tool_call_with_lifecycle(
        ToolCall(id="trace-await", name="coverage", arguments={}),
        confirm=False,
        abort_on_reject=False,
        emit_ready=False,
        planned_index=0,
        finish_stream_on_error=False,
    )

    assert outcome.result is None
    assert len(provider_traces) == 1
    assert len(tool_traces) == 1


@pytest.mark.anyio
async def test_interrupt_releases_cancelled_pre_dispatch_task() -> None:
    """Release a claim after cancellation before durable dispatch begins."""
    harness = _harness()
    admission = await _admit(harness)
    entered = Event()

    async def wait_before_durable_dispatch() -> object:
        entered.set()
        await Event().wait()
        return object()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            admission,
            "_dispatch_once",
            wait_before_durable_dispatch,
        )
        dispatch = create_task(admission.dispatch())
        await entered.wait()
        settled = await admission.interrupt_dispatch()

    with pytest.raises(CancelledError):
        await dispatch
    assert admission._dispatch_task is None
    assert (
        settled is continuation_module.DurableContinuationResumeState.RELEASED
    )
    assert harness.store.calls.count("release") == 1


@pytest.mark.anyio
async def test_hardening_rejects_duplicate_validate_plan_admission() -> None:
    """Reject a second plan-validation admission from one coordinator task."""
    scope_digest = conversation.AuthorityDigest("a" * 64)

    class Clock:
        async def now(self) -> datetime:
            """Return a fixed valid time for the operational key ring."""
            return datetime(2026, 8, 25, tzinfo=UTC)

    class Maintenance:
        kind = security.ConversationMaintenanceKind.RETENTION

        async def run(self, *, limit: int) -> int:
            """Run the inert bounded maintenance cycle."""
            assert limit == 1
            return 0

    async def backend_probe() -> security.ConversationBackendHealth:
        """Report a healthy local conversation backend."""
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        )

    async def capability_probe() -> security.ConversationCapabilityHealth:
        """Report every configured capability as locally resolvable."""
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=0,
            resolvable_profiles=0,
        )

    retention = conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.PROCESS_LOCAL,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
    )
    policy = security.resolve_conversation_policy(
        security.ConversationHardeningPolicy(
            default_mode=conversation.ConversationMode.STATELESS,
            allowed_modes=frozenset({conversation.ConversationMode.STATELESS}),
            allowed_reasoning_contexts=frozenset(
                conversation.ReasoningContext
            ),
            compaction=security.ConversationCompactionPolicy(
                allowed_operations=frozenset(conversation.CompactionOperation)
            ),
            backend=security.ConversationCheckpointBackend.PROCESS,
            retention=retention,
            resources=security.ConversationResourcePolicy(),
            checkpoint_keys=security.ConversationKeyRotationPolicy(),
            envelope_keys=security.ConversationKeyRotationPolicy(),
            capability_profiles=(),
            telemetry=security.ConversationTelemetryPolicy(),
        )
    )
    key = security.ConversationOperationalKey(
        key_id="phase-eight-key",
        revision=1,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        purposes=frozenset(security.ConversationKeyPurpose),
        key_bytes=b"k" * 32,
        activated_at=datetime(2026, 8, 25, tzinfo=UTC),
    )
    worker = security.ConversationMaintenanceWorker(
        (Maintenance(),),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.1,
    )
    readiness = security.ConversationReadinessChecker(
        backend_probe=backend_probe,
        key_ring=security.AsyncConversationKeyRing(
            {scope_digest: (key,)},
            clock=Clock(),
        ),
        authority=scope_digest,
        workers=(worker,),
        capability_probe=capability_probe,
        activation=security.ConversationActivationHealth(
            expected_digest=conversation.IntegrityDigest("b" * 64),
            loaded_digest=conversation.IntegrityDigest("b" * 64),
        ),
    )
    hook = security.ConversationHardeningCoordinatorHook(
        policy=policy,
        admission=security.FairConversationAdmissionController(
            policy.resources
        ),
        admission_key=security.ConversationAdmissionKey(
            authority_digest=scope_digest,
            conversation_digest=conversation.IntegrityDigest("c" * 64),
        ),
        readiness=readiness,
        telemetry=security.BoundedConversationTelemetry(max_events=2),
    )

    await hook.start()
    await hook.reach(conversation.CoordinatorAwaitBoundary.VALIDATE_PLAN)
    with pytest.raises(conversation.ConversationValidationError):
        await hook.reach(conversation.CoordinatorAwaitBoundary.VALIDATE_PLAN)
    await hook.reach(conversation.CoordinatorAwaitBoundary.ROLLBACK)
    await hook.close()
