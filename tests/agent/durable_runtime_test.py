"""Exercise portable agent continuation reconstruction across runtimes."""

from asyncio import gather
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from json import dumps
from logging import Logger, getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

from avalan.agent.continuation import AgentContinuationResumeCommand
from avalan.agent.durable_runtime import (
    TrustedAgentContinuationExecutor,
)
from avalan.agent.loader import OrchestratorLoader
from avalan.entities import TransformerEngineSettings
from avalan.event import Event, EventType
from avalan.interaction.codec import encode_input_question
from avalan.interaction.continuation import (
    ContinuationClaim,
    ContinuationClaimOwnerId,
    ContinuationClaimState,
    ContinuationCompletionCommand,
    ContinuationDispatch,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationRejectionCommand,
    ContinuationStoreRevision,
    DurableContinuationResumeState,
    PortableContinuation,
    ResolvedContinuationRuntime,
    decode_portable_continuation,
    encode_portable_continuation,
)
from avalan.interaction.entities import (
    AnsweredResolution,
    AnswerProvenance,
    ConfirmationAnswer,
    InputRequest,
    ProviderConfigRevision,
    RequestState,
    ResumeInputContinuation,
    StateRevision,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.state import project_resolution_to_model
from avalan.model.call import ModelCall
from avalan.model.capability import (
    CapabilityBatchAccepted,
    ModelCapabilityCatalog,
    ProviderCapabilityCall,
    TaskInputCapabilityCall,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
)
from avalan.task.context import (
    TaskDurableResumeHandle,
    TaskEventListener,
    TaskEventListenerRegistration,
    TaskTargetContext,
)
from avalan.task.definition import (
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
)
from avalan.task.durable_agent import DurableAgentTaskHost, _utc_now
from avalan.task.settlement import (
    TaskDurableResumeFailure,
    TaskDurableResumeSettlement,
)
from avalan.task.store import TaskExecutionContext
from avalan.task.target import TaskTargetSuspended
from avalan.task.targets import AgentTaskTargetRunner
from avalan.task.worker import TaskWorker
from avalan.types import LooseJsonValue

_NOW = datetime(2026, 7, 23, 18, tzinfo=UTC)
_OFFSET_NOW = datetime(
    2026,
    7,
    23,
    15,
    tzinfo=timezone(timedelta(hours=-3)),
)
_MODEL_ID = "gpt-5"
_INPUT_ARGUMENTS = {
    "mode": "required",
    "reason": "Need one bounded decision.",
    "questions": [
        {
            "question_id": "continue",
            "kind": "confirmation",
            "prompt": "Continue?",
            "required": True,
            "choices": [],
            "allow_other": False,
        }
    ],
}


class _FalsyClock:
    """Return fixed time while refusing truth-value substitution."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def __bool__(self) -> bool:
        return False


class _UnusedDurableStore:
    """Satisfy host assembly without hiding accidental store access."""

    async def get_task_continuation_record(self, task_run_id: str) -> object:
        del task_run_id
        raise AssertionError("unexpected task continuation lookup")

    async def lookup_scoped(self, query: object) -> object:
        del query
        raise AssertionError("unexpected interaction lookup")

    async def claim(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unexpected continuation claim")

    async def mark_dispatching(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise AssertionError("unexpected dispatch marker")

    async def renew_claim(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise AssertionError("unexpected claim renewal")

    async def mark_dispatched(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise AssertionError("unexpected dispatched marker")

    async def complete(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unexpected continuation completion")

    async def get_continuation(self, continuation_id: object) -> object:
        del continuation_id
        raise AssertionError("unexpected continuation read")

    async def release(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unexpected continuation release")

    async def get_run(self, run_id: str) -> object:
        del run_id
        raise AssertionError("unexpected task run read")


class _UnusedQueue:
    async def claim(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("unexpected queue claim")


class _CountingAgentRunner(AgentTaskTargetRunner):
    """Count original task-target invocation independently of resume."""

    def __init__(
        self,
        loader: OrchestratorLoader,
        *,
        ref_base: Path,
        durable_interaction_runtime_factory: Any,
    ) -> None:
        super().__init__(
            cast(Any, loader),
            ref_base=ref_base,
            durable_interaction_runtime_factory=(
                durable_interaction_runtime_factory
            ),
        )
        self.run_count = 0

    async def run(self, context: TaskTargetContext) -> object:
        self.run_count += 1
        return await super().run(context)


class _ResumeHandle:
    """Dispatch one trusted executor through the public target resume path."""

    def __init__(
        self,
        executor: TrustedAgentContinuationExecutor,
        command: AgentContinuationResumeCommand,
    ) -> None:
        self._executor = executor
        self._command = command
        self.dispatch_count = 0

    def register_event_listener(
        self,
        listener: TaskEventListener,
    ) -> TaskEventListenerRegistration:
        return self._executor.register_event_listener(listener)

    async def dispatch(self) -> object:
        self.dispatch_count += 1
        return await self._executor.resume_agent_continuation(self._command)

    async def wait_dispatch_settled(
        self,
    ) -> DurableContinuationResumeState:
        raise AssertionError("unexpected dispatch settlement wait")

    async def interrupt_dispatch(self) -> DurableContinuationResumeState:
        raise AssertionError("unexpected dispatch interruption")

    async def complete_output(self, output: object) -> None:
        del output
        raise AssertionError("unexpected output completion")

    def completion_command_for_output(
        self,
        output: object,
    ) -> ContinuationCompletionCommand:
        del output
        raise AssertionError("unexpected output completion command")

    def completion_command_for_settlement(
        self,
        settlement: TaskDurableResumeSettlement,
    ) -> ContinuationCompletionCommand:
        del settlement
        raise AssertionError("unexpected settlement completion command")

    def completed_completion_command(self) -> ContinuationCompletionCommand:
        raise AssertionError("unexpected completed command")

    def completion_command_for_suspension(
        self,
        *,
        request_id: str,
        continuation_id: str,
        checkpoint_id: str,
    ) -> ContinuationCompletionCommand:
        del request_id, continuation_id, checkpoint_id
        raise AssertionError("unexpected suspension completion command")

    def rejection_command_for_settlement(
        self,
        failure: TaskDurableResumeFailure,
    ) -> ContinuationRejectionCommand:
        del failure
        raise AssertionError("unexpected rejection command")

    async def release(self) -> None:
        raise AssertionError("unexpected release")

    async def release_if_pre_dispatch(self) -> bool:
        raise AssertionError("unexpected safe release")

    async def close(self) -> None:
        await self._executor.close_continuation_runtime()


class _ColdModelManager:
    """Load exact native adapters and return deterministic provider streams."""

    def __init__(self, *, ask_for_input: bool, call_id_start: int) -> None:
        self.ask_for_input = ask_for_input
        self.call_id_start = call_id_start
        self.calls: list[ModelCall] = []
        self.engines: list[OpenAIModel] = []
        self.closed_engines: list[OpenAIModel] = []
        self.serialized_inputs: list[list[dict[str, Any]]] = []
        self.exit_events: list[str] = []
        self.exited = False
        self.exit_calls = 0

    def __enter__(self) -> "_ColdModelManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> Literal[False]:
        del exc_type, exc_value, traceback
        self.exit_events.append("manager")
        self.exited = True
        self.exit_calls += 1
        return False

    @staticmethod
    def parse_uri(uri: str) -> object:
        return ModelManager.parse_uri(uri)

    def get_engine_settings(
        self,
        engine_uri: object,
        settings: Mapping[str, object] | None = None,
        modality: object | None = None,
    ) -> TransformerEngineSettings:
        del engine_uri, settings, modality
        return TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            access_token="test-only-token",
        )

    def load_engine(
        self,
        engine_uri: object,
        engine_settings: TransformerEngineSettings,
        modality: object,
    ) -> OpenAIModel:
        del modality
        model_id = getattr(engine_uri, "model_id")
        model = OpenAIModel(
            model_id=model_id,
            settings=engine_settings,
            logger=getLogger(__name__),
        )
        model._model = _openai_client()  # noqa: SLF001
        model._exit_stack.push_async_callback(  # noqa: SLF001
            self._record_engine_closed,
            model,
        )
        self.engines.append(model)
        return model

    async def _record_engine_closed(self, model: OpenAIModel) -> None:
        self.exit_events.append("engine")
        self.closed_engines.append(model)

    async def __call__(self, call: ModelCall) -> TextGenerationResponse:
        self.calls.append(call)
        model = cast(OpenAIModel, call.model)
        client = cast(OpenAIClient, model.model)
        call_id = f"input-call-{self.call_id_start + len(self.calls) - 1}"
        if self.ask_for_input:
            owner = client._replay_owner_for_messages(  # noqa: SLF001
                cast(list[Any], call.operation.input)
            )
            owner.begin_attempt()
            self.serialized_inputs.append(
                cast(
                    list[dict[str, Any]],
                    client._template_messages(  # noqa: SLF001
                        cast(list[Any], call.operation.input),
                        capability=cast(
                            ModelCapabilityCatalog,
                            call.context.capability,
                        ),
                        replay_items=owner.replay_items(),
                    ),
                )
            )
            for item in (
                {
                    "id": f"reasoning-{call_id}",
                    "type": "reasoning",
                    "encrypted_content": f"encrypted-{call_id}",
                    "summary": [],
                },
                {
                    "id": f"function-{call_id}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "request_user_input",
                    "arguments": dumps(
                        _INPUT_ARGUMENTS,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            ):
                assert owner.admit(item)
            owner.commit_attempt()
            return _provider_response(
                call,
                call_id=call_id,
                client=client,
                owner=owner,
                ask_for_input=True,
            )
        return _provider_response(
            call,
            call_id=call_id,
            client=client,
            owner=None,
            ask_for_input=False,
        )


class _ManagerFactory:
    """Create isolated model managers for successive fresh processes."""

    def __init__(
        self,
        plans: list[bool],
        *,
        call_id_start: int = 1,
    ) -> None:
        self._plans = plans
        self._call_id_start = call_id_start
        self.instances: list[_ColdModelManager] = []

    def __call__(self, *args: object, **kwargs: object) -> _ColdModelManager:
        del args, kwargs
        if not self._plans:
            raise AssertionError("unexpected orchestrator process")
        manager = _ColdModelManager(
            ask_for_input=self._plans.pop(0),
            call_id_start=self._call_id_start + len(self.instances),
        )
        self.instances.append(manager)
        return manager


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


def _provider_response(
    call: ModelCall,
    *,
    call_id: str,
    client: OpenAIClient,
    owner: object | None,
    ask_for_input: bool,
) -> TextGenerationResponse:
    capability = cast(ModelCapabilityCatalog, call.context.capability)
    provider_name = capability.provider_name(
        "request_user_input",
        provider_family="openai",
    )

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        def item(
            sequence: int,
            kind: StreamItemKind,
            channel: StreamChannel,
            *,
            correlation: StreamItemCorrelation | None = None,
            text_delta: str | None = None,
            data: LooseJsonValue | None = None,
            usage: LooseJsonValue | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id=f"stream-{call_id}",
                run_id=f"run-{call_id}",
                turn_id=f"turn-{call_id}",
                provider_family="openai",
                sequence=sequence,
                kind=kind,
                channel=channel,
                correlation=correlation or StreamItemCorrelation(),
                text_delta=text_delta,
                data=data,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        try:
            yield item(
                0,
                StreamItemKind.STREAM_STARTED,
                StreamChannel.CONTROL,
            )
            if ask_for_input:
                correlation = StreamItemCorrelation(tool_call_id=call_id)
                yield item(
                    1,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    text_delta=dumps(_INPUT_ARGUMENTS),
                    correlation=correlation,
                )
                yield item(
                    2,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamChannel.TOOL_CALL,
                    data={"name": provider_name},
                    correlation=correlation,
                )
                yield item(
                    3,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                    correlation=correlation,
                )
                sequence = 4
            else:
                yield item(
                    1,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="Finished.",
                )
                yield item(
                    2,
                    StreamItemKind.ANSWER_DONE,
                    StreamChannel.ANSWER,
                )
                sequence = 3
            yield item(
                sequence,
                StreamItemKind.STREAM_COMPLETED,
                StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield item(
                sequence + 1,
                StreamItemKind.STREAM_CLOSED,
                StreamChannel.CONTROL,
            )
        finally:
            if owner is not None:
                cast(Any, owner).release()

    return TextGenerationResponse(
        lambda: items(),
        logger=getLogger(__name__),
        use_async_generator=True,
        continuation_snapshot_adapter=client,
    )


def _agent_file(root: Path) -> Path:
    path = root / "agent.toml"
    path.write_text(
        """
[agent]
instructions = "Request confirmation before continuing."

[engine]
uri = "ai://openai/gpt-5"

[run]
maximum_tool_cycles = 4
block_repeated_tool_calls = true
""",
        encoding="utf-8",
    )
    return path


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="cold-runtime", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
    )


def _context(
    definition: TaskDefinition,
    *,
    input_value: object,
    durable_resume: object | None = None,
    event_listener: TaskEventListener | None = None,
) -> TaskTargetContext:
    return TaskTargetContext(
        definition=definition,
        execution=TaskExecutionContext(
            run_id="task-run",
            attempt_id="task-attempt",
            attempt_number=1,
        ),
        input_value=input_value,
        event_listener=event_listener,
        durable_resume=cast(Any, durable_resume),
    )


def _engine_run_event_recorder(
    event_types: list[EventType],
) -> TaskEventListener:
    def record(event: Event) -> None:
        if event.type in {
            EventType.ENGINE_RUN_BEFORE,
            EventType.ENGINE_RUN_AFTER,
        }:
            event_types.append(event.type)

    return record


def _terminal_request(request: InputRequest) -> InputRequest:
    question = request.questions[0]
    resolution = AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            ConfirmationAnswer(
                question_id=question.question_id,
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
        ),
    )
    return replace(
        request,
        state=RequestState.ANSWERED,
        state_revision=StateRevision(2),
        resolution=resolution,
    )


def _claimed_continuation(
    continuation: PortableContinuation,
) -> PortableContinuation:
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    return replace(
        continuation,
        claim=ContinuationClaim(
            state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
            owner_id=ContinuationClaimOwnerId("worker"),
            lease_expires_at=_NOW + timedelta(minutes=5),
            attempt=1,
        ),
        fencing_token=ContinuationFencingToken(1),
        dispatch=ContinuationDispatch(
            dispatch_id=ContinuationDispatchId("dispatch"),
            provider_idempotency_key=snapshot.provider_idempotency_key,
            marked_at=_NOW,
        ),
        store_revision=ContinuationStoreRevision(1),
        updated_at=_NOW,
    )


def _resume_command(
    continuation: PortableContinuation,
    request: InputRequest,
    runtime: ResolvedContinuationRuntime,
) -> AgentContinuationResumeCommand:
    catalog = cast(
        ModelCapabilityCatalog,
        runtime.capabilities,
    )
    provider_family = str(continuation.revision_binding.provider_family)
    provider_name = catalog.provider_name(
        "request_user_input",
        provider_family=provider_family,
    )
    arguments = {
        "mode": request.mode.value,
        "reason": request.reason,
        "questions": [
            encode_input_question(question) for question in request.questions
        ],
    }
    batch = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id=continuation.provider_call_correlation_id,
                provider_name=provider_name,
                arguments=arguments,
            ),
        ),
        provider_family=provider_family,
    )
    assert isinstance(batch, CapabilityBatchAccepted)
    call = cast(TaskInputCapabilityCall, batch.task_input)
    outcome = project_resolution_to_model(
        request,
        containing_run_exists=True,
    )
    assert isinstance(outcome, ResumeInputContinuation)
    model_result = outcome.result
    correlated = catalog.project_result(call, model_result)
    snapshot = continuation.provider_snapshot
    assert snapshot is not None
    adapter = cast(OpenAIClient, runtime.model)
    adapter.validate_continuation_snapshot_call(
        snapshot,
        expected_binding=continuation.revision_binding,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
        expected_provider_name=call.provider_name,
        expected_arguments=call.arguments,
    )
    adapter.import_continuation_snapshot(
        snapshot,
        expected_binding=continuation.revision_binding,
        provider_call_correlation_id=(
            continuation.provider_call_correlation_id
        ),
    )
    return AgentContinuationResumeCommand(
        continuation=cast(Any, continuation),
        request=request,
        model_result=model_result,
        task_input_call=call,
        correlated_result=correlated,
        resolved_runtime=cast(Any, runtime),
    )


class DurableRuntimeTest(IsolatedAsyncioTestCase):
    async def test_real_loader_cold_resume_can_suspend_again(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent_path = _agent_file(root)
            first_stack = AsyncExitStack()
            first_factory = _ManagerFactory([True])
            first_loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=first_stack,
            )
            first_host = DurableAgentTaskHost(
                orchestrator_loader=first_loader,
                stack=first_stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW,
            )
            first_runner = _CountingAgentRunner(
                first_loader,
                ref_base=root,
                durable_interaction_runtime_factory=(
                    first_host.interaction_runtime
                ),
            )
            worker = TaskWorker(
                cast(Any, _UnusedDurableStore()),
                cast(Any, _UnusedQueue()),
                target=first_runner,
                durable_resume_coordinator=first_host.resume_coordinator,
            )
            self.assertIs(
                worker._durable_resume_coordinator,  # noqa: SLF001
                first_host.resume_coordinator,
            )
            self.assertIs(
                cast(  # noqa: SLF001
                    Any,
                    first_runner._durable_interaction_runtime_factory,
                ).__self__,
                first_host,
            )
            initial_engine_run_events: list[EventType] = []

            with patch(
                "avalan.agent.loader.ModelManager",
                side_effect=first_factory,
            ):
                first_outcome = await first_runner.run(
                    _context(
                        _definition(),
                        input_value="original input",
                        event_listener=_engine_run_event_recorder(
                            initial_engine_run_events
                        ),
                    )
                )
            self.assertIsInstance(first_outcome, TaskTargetSuspended)
            first_suspension = cast(
                TaskTargetSuspended,
                first_outcome,
            )
            assert first_suspension.durable is not None
            portable = decode_portable_continuation(
                encode_portable_continuation(
                    first_suspension.durable.continuation
                ),
                expected_binding=(
                    first_suspension.durable.continuation.revision_binding
                ),
            )
            self.assertEqual(first_runner.run_count, 1)
            self.assertEqual(
                portable.definition.agent_definition_locator,
                agent_path.resolve().as_uri(),
            )
            self.assertEqual(portable.tool_loop_count, 0)
            self.assertEqual(
                portable.generation_settings["maximum_tool_cycles"],
                4,
            )
            self.assertEqual(
                initial_engine_run_events,
                [
                    EventType.ENGINE_RUN_BEFORE,
                    EventType.ENGINE_RUN_AFTER,
                ],
            )
            first_manager = first_factory.instances[0]
            self.assertIn(
                first_manager.engines[0],
                first_manager.closed_engines,
            )
            await first_stack.aclose()
            self.assertTrue(first_manager.exited)

            fresh_stack = AsyncExitStack()
            fresh_factory = _ManagerFactory([True], call_id_start=2)
            fresh_loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=fresh_stack,
            )
            fresh_host = DurableAgentTaskHost(
                orchestrator_loader=fresh_loader,
                stack=fresh_stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW,
            )
            fresh_runtime_loader = fresh_host.continuation_runtime_loader
            fresh_runner = _CountingAgentRunner(
                fresh_loader,
                ref_base=root,
                durable_interaction_runtime_factory=(
                    fresh_host.interaction_runtime
                ),
            )
            original_register = (
                OpenAIModel.register_continuation_snapshot_codec
            )
            registrations: list[OpenAIModel] = []

            def tracked_register(
                model: OpenAIModel,
                binding: object,
            ) -> object:
                registrations.append(model)
                return original_register(model, cast(Any, binding))

            with (
                patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=fresh_factory,
                ),
                patch.object(
                    OpenAIModel,
                    "register_continuation_snapshot_codec",
                    new=tracked_register,
                ),
            ):
                runtime = await fresh_runtime_loader.load_continuation_runtime(
                    portable.definition,
                    portable.revision_binding,
                )
            self.assertGreaterEqual(len(registrations), 1)
            executor = cast(
                TrustedAgentContinuationExecutor,
                runtime.runtime,
            )
            fresh_orchestrator = executor._orchestrator  # noqa: SLF001
            self.assertFalse(fresh_orchestrator._exiting)  # noqa: SLF001
            claimed = _claimed_continuation(portable)
            terminal = _terminal_request(
                first_suspension.durable.command.request
            )
            command = _resume_command(claimed, terminal, runtime)
            handle = _ResumeHandle(executor, command)
            failed_listener = _engine_run_event_recorder([])
            event_manager = fresh_orchestrator.event_manager
            with (
                patch.object(
                    event_manager,
                    "add_listener",
                    side_effect=RuntimeError("listener registration failed"),
                ),
                patch.object(
                    event_manager,
                    "remove_listener",
                    wraps=event_manager.remove_listener,
                ) as failed_remove_listener,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "listener registration failed",
                ):
                    await fresh_runner.resume(
                        _context(
                            _definition(),
                            input_value=object(),
                            durable_resume=handle,
                            event_listener=failed_listener,
                        ),
                        handle,
                    )
                failed_remove_listener.assert_called_once_with(failed_listener)
            self.assertEqual(handle.dispatch_count, 0)
            self.assertEqual(fresh_factory.instances[0].calls, [])
            resumed_engine_run_events: list[EventType] = []
            resume_context = _context(
                _definition(),
                input_value=object(),
                durable_resume=handle,
                event_listener=_engine_run_event_recorder(
                    resumed_engine_run_events
                ),
            )
            second_outcome = await fresh_runner.resume(
                resume_context,
                cast(TaskDurableResumeHandle, handle),
            )
            self.assertIsInstance(second_outcome, TaskTargetSuspended)
            successor = cast(TaskTargetSuspended, second_outcome)
            assert successor.durable is not None
            self.assertEqual(fresh_runner.run_count, 0)
            self.assertEqual(handle.dispatch_count, 1)
            self.assertEqual(
                resumed_engine_run_events,
                [
                    EventType.ENGINE_RUN_BEFORE,
                    EventType.ENGINE_RUN_AFTER,
                ],
            )
            self.assertNotEqual(
                successor.durable.continuation.continuation_id,
                portable.continuation_id,
            )
            self.assertGreater(
                len(successor.durable.continuation.transcript),
                len(portable.transcript),
            )
            self.assertEqual(
                successor.durable.continuation.interaction_count,
                portable.interaction_count + 1,
            )
            fresh_manager = fresh_factory.instances[0]
            replay_call_id = str(portable.provider_call_correlation_id)
            replay_input = fresh_manager.serialized_inputs[0]
            original_calls = [
                item
                for item in replay_input
                if item.get("type") == "function_call"
                and item.get("call_id") == replay_call_id
            ]
            correlated_outputs = [
                item
                for item in replay_input
                if item.get("type") == "function_call_output"
                and item.get("call_id") == replay_call_id
            ]
            self.assertEqual(len(original_calls), 1)
            self.assertEqual(
                original_calls[0]["arguments"],
                dumps(
                    _INPUT_ARGUMENTS,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
            self.assertEqual(len(correlated_outputs), 1)
            shutdown_listener = _engine_run_event_recorder([])
            with patch.object(
                event_manager,
                "remove_listener",
                wraps=event_manager.remove_listener,
            ) as remove_listener:
                shutdown_registration = executor.register_event_listener(
                    shutdown_listener
                )
                await gather(handle.close(), fresh_stack.aclose())
                shutdown_registration.close()
                remove_listener.assert_called_once_with(shutdown_listener)
            self.assertTrue(fresh_orchestrator._exiting)  # noqa: SLF001
            self.assertIn(
                fresh_manager.engines[0],
                fresh_manager.closed_engines,
            )
            self.assertEqual(
                fresh_manager.closed_engines.count(fresh_manager.engines[0]),
                1,
            )
            self.assertTrue(fresh_manager.exited)
            self.assertEqual(fresh_manager.exit_calls, 1)
            self.assertEqual(fresh_manager.exit_events, ["engine", "manager"])
            await gather(handle.close(), fresh_stack.aclose())
            self.assertTrue(fresh_orchestrator._exiting)  # noqa: SLF001
            self.assertIn(
                fresh_manager.engines[0],
                fresh_manager.closed_engines,
            )
            self.assertTrue(fresh_manager.exited)
            self.assertEqual(fresh_manager.exit_calls, 1)

    async def test_loader_rejects_locator_revision_and_payload_tamper(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent_path = _agent_file(root)
            outside = root.parent / f"outside-{uuid4()}.toml"
            outside.write_text(agent_path.read_text(), encoding="utf-8")
            stack = AsyncExitStack()
            factory = _ManagerFactory([True])
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            host = DurableAgentTaskHost(
                orchestrator_loader=loader,
                stack=stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW,
            )
            runner = _CountingAgentRunner(
                loader,
                ref_base=root,
                durable_interaction_runtime_factory=host.interaction_runtime,
            )
            with patch(
                "avalan.agent.loader.ModelManager",
                side_effect=factory,
            ):
                outcome = cast(
                    TaskTargetSuspended,
                    await runner.run(
                        _context(_definition(), input_value="original")
                    ),
                )
            assert outcome.durable is not None
            portable = outcome.durable.continuation

            with self.assertRaises(InputValidationError) as outside_error:
                await (
                    host.continuation_runtime_loader.load_continuation_runtime(
                        replace(
                            portable.definition,
                            agent_definition_locator=(
                                outside.resolve().as_uri()
                            ),
                        ),
                        portable.revision_binding,
                    )
                )
            self.assertIs(
                outside_error.exception.code, InputErrorCode.FORBIDDEN
            )

            with self.assertRaises(InputValidationError) as locator_error:
                await (
                    host.continuation_runtime_loader.load_continuation_runtime(
                        replace(
                            portable.definition,
                            agent_definition_locator=(
                                portable.definition.agent_definition_locator
                                + "#tampered"
                            ),
                        ),
                        portable.revision_binding,
                    )
                )
            self.assertIs(
                locator_error.exception.code, InputErrorCode.FORBIDDEN
            )

            transfer_loader_stack = AsyncExitStack()
            transfer_stack = AsyncExitStack()
            transfer_factory = _ManagerFactory([True])
            transfer_loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=transfer_loader_stack,
            )
            transfer_host = DurableAgentTaskHost(
                orchestrator_loader=transfer_loader,
                stack=transfer_stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW,
            )
            transfer_runtime_loader = transfer_host.continuation_runtime_loader
            with (
                patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=transfer_factory,
                ),
                patch.object(
                    transfer_stack,
                    "enter_async_context",
                    side_effect=RuntimeError("transfer failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "transfer failed"),
            ):
                await transfer_runtime_loader.load_continuation_runtime(
                    portable.definition,
                    portable.revision_binding,
                )
            transfer_manager = transfer_factory.instances[0]
            self.assertIn(
                transfer_manager.engines[0],
                transfer_manager.closed_engines,
            )
            self.assertTrue(transfer_manager.exited)
            self.assertEqual(transfer_manager.exit_calls, 1)
            await transfer_stack.aclose()
            await transfer_loader_stack.aclose()

            drift_stack = AsyncExitStack()
            drift_factory = _ManagerFactory([True, True, True])
            drift_loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=drift_stack,
            )
            drift_host = DurableAgentTaskHost(
                orchestrator_loader=drift_loader,
                stack=drift_stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW,
            )
            drift_runtime_loader = drift_host.continuation_runtime_loader
            original_text = agent_path.read_text(encoding="utf-8")
            agent_path.write_text(
                original_text.replace(
                    "Request confirmation before continuing.",
                    "Changed trusted instructions.",
                ),
                encoding="utf-8",
            )
            with patch(
                "avalan.agent.loader.ModelManager",
                side_effect=drift_factory,
            ):
                with self.assertRaises(
                    InputValidationError
                ) as definition_error:
                    await drift_runtime_loader.load_continuation_runtime(
                        portable.definition,
                        portable.revision_binding,
                    )
            self.assertIs(
                definition_error.exception.code,
                InputErrorCode.SNAPSHOT_REVISION_DRIFT,
            )
            self.assertIn(
                drift_factory.instances[0].engines[0],
                drift_factory.instances[0].closed_engines,
            )

            agent_path.write_text(original_text, encoding="utf-8")
            with patch(
                "avalan.agent.loader.ModelManager",
                side_effect=drift_factory,
            ):
                with self.assertRaises(InputValidationError) as binding_error:
                    await drift_runtime_loader.load_continuation_runtime(
                        portable.definition,
                        replace(
                            portable.revision_binding,
                            provider_config_revision=(
                                ProviderConfigRevision("tampered")
                            ),
                        ),
                    )
            self.assertIs(
                binding_error.exception.code,
                InputErrorCode.SNAPSHOT_REVISION_DRIFT,
            )

            with patch(
                "avalan.agent.loader.ModelManager",
                side_effect=drift_factory,
            ):
                runtime = await drift_runtime_loader.load_continuation_runtime(
                    portable.definition,
                    portable.revision_binding,
                )
            claimed = _claimed_continuation(portable)
            terminal = _terminal_request(outcome.durable.command.request)
            command = _resume_command(claimed, terminal, runtime)
            first_record = dict(claimed.transcript[0])
            first_record["data"] = "avalan-message-v1:{"
            tampered = replace(
                claimed,
                transcript=(first_record, *claimed.transcript[1:]),
            )
            executor = cast(
                TrustedAgentContinuationExecutor,
                runtime.runtime,
            )
            with self.assertRaises(InputValidationError) as payload_error:
                await executor.resume_agent_continuation(
                    replace(command, continuation=tampered)
                )
            self.assertIs(
                payload_error.exception.code,
                InputErrorCode.SNAPSHOT_INVALID,
            )
            self.assertEqual(drift_factory.instances[2].calls, [])
            await drift_stack.aclose()
            await stack.aclose()
            outside.unlink()

    async def test_host_clock_controls_request_and_checkpoint_time(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _agent_file(root)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            factory = _ManagerFactory([True])
            host = DurableAgentTaskHost(
                orchestrator_loader=loader,
                stack=stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=_FalsyClock(_OFFSET_NOW),
            )
            runner = _CountingAgentRunner(
                loader,
                ref_base=root,
                durable_interaction_runtime_factory=host.interaction_runtime,
            )
            with (
                patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=factory,
                ),
                patch(
                    "avalan.agent.durable_runtime.datetime"
                ) as wall_datetime,
            ):
                outcome = await runner.run(
                    _context(_definition(), input_value="original")
                )
            self.assertFalse(wall_datetime.now.called)
            self.assertIsInstance(outcome, TaskTargetSuspended)
            suspension = cast(TaskTargetSuspended, outcome)
            assert suspension.durable is not None
            request = suspension.durable.command.request
            continuation = suspension.durable.continuation
            self.assertEqual(request.created_at, _NOW)
            self.assertIs(request.created_at.tzinfo, UTC)
            self.assertEqual(continuation.created_at, request.created_at)
            self.assertEqual(continuation.updated_at, request.created_at)
            self.assertEqual(
                continuation.expires_at,
                request.created_at
                + timedelta(seconds=request.continuation_ttl_seconds),
            )
            await stack.aclose()

    async def test_host_rejects_invalid_and_naive_clocks(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _agent_file(root)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            invalid_clock: Any = object()
            with self.assertRaisesRegex(TypeError, "clock must be callable"):
                DurableAgentTaskHost(
                    orchestrator_loader=loader,
                    stack=stack,
                    allowed_roots=(root,),
                    continuation_store=_UnusedDurableStore(),
                    clock=invalid_clock,
                )

            factory = _ManagerFactory([True])
            host = DurableAgentTaskHost(
                orchestrator_loader=loader,
                stack=stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                clock=lambda: _NOW.replace(tzinfo=None),
            )
            runner = _CountingAgentRunner(
                loader,
                ref_base=root,
                durable_interaction_runtime_factory=host.interaction_runtime,
            )
            with (
                patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=factory,
                ),
                self.assertRaises(InputValidationError) as clock_error,
            ):
                await runner.run(
                    _context(_definition(), input_value="original")
                )
            self.assertIs(
                clock_error.exception.code,
                InputErrorCode.NAIVE_TIMESTAMP,
            )
            self.assertEqual(
                clock_error.exception.path,
                "continuation_stager.clock",
            )
            await stack.aclose()

    async def test_host_validates_assembly_and_runtime_boundaries(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )

            with self.assertRaisesRegex(
                TypeError,
                "continuation_store must expose",
            ):
                DurableAgentTaskHost(
                    orchestrator_loader=loader,
                    stack=stack,
                    allowed_roots=(root,),
                    continuation_store=object(),
                )
            with self.assertRaisesRegex(
                TypeError,
                "actor_resolver must be callable",
            ):
                DurableAgentTaskHost(
                    orchestrator_loader=loader,
                    stack=stack,
                    allowed_roots=(root,),
                    continuation_store=_UnusedDurableStore(),
                    actor_resolver=cast(Any, object()),
                )

            host = DurableAgentTaskHost(
                orchestrator_loader=loader,
                stack=stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
            )
            self.assertIs(host.resume_coordinator, host.resume_coordinator)
            self.assertIs(
                host.continuation_runtime_loader,
                host.continuation_runtime_loader,
            )
            with self.assertRaisesRegex(
                TypeError,
                "context must be a task target context",
            ):
                host.interaction_runtime(cast(Any, object()))

            invalid_actor_host = DurableAgentTaskHost(
                orchestrator_loader=loader,
                stack=stack,
                allowed_roots=(root,),
                continuation_store=_UnusedDurableStore(),
                actor_resolver=lambda context: cast(Any, object()),
            )
            with self.assertRaisesRegex(
                TypeError,
                "invalid actor",
            ):
                invalid_actor_host.interaction_runtime(
                    _context(_definition(), input_value="ready")
                )

            runtime = host.interaction_runtime(
                _context(_definition(), input_value="ready")
            )
            self.assertEqual(runtime.run_id, "task-run")
            self.assertEqual(runtime.task_id, "task-run")
            self.assertIs(_utc_now().tzinfo, UTC)
            await stack.aclose()

    async def test_host_production_clock_defaults_to_utc_now(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _agent_file(root)
            stack = AsyncExitStack()
            loader = OrchestratorLoader(
                hub=MagicMock(spec=HuggingfaceHub),
                logger=MagicMock(spec=Logger),
                participant_id=uuid4(),
                stack=stack,
            )
            factory = _ManagerFactory([True])
            with patch(
                "avalan.task.durable_agent._utc_now",
                return_value=_NOW,
            ) as production_clock:
                host = DurableAgentTaskHost(
                    orchestrator_loader=loader,
                    stack=stack,
                    allowed_roots=(root,),
                    continuation_store=_UnusedDurableStore(),
                )
                runner = _CountingAgentRunner(
                    loader,
                    ref_base=root,
                    durable_interaction_runtime_factory=(
                        host.interaction_runtime
                    ),
                )
                with patch(
                    "avalan.agent.loader.ModelManager",
                    side_effect=factory,
                ):
                    outcome = await runner.run(
                        _context(_definition(), input_value="original")
                    )
            self.assertIsInstance(outcome, TaskTargetSuspended)
            suspension = cast(TaskTargetSuspended, outcome)
            assert suspension.durable is not None
            self.assertEqual(
                suspension.durable.command.request.created_at,
                _NOW,
            )
            self.assertGreaterEqual(production_clock.call_count, 1)
            await stack.aclose()
