from asyncio import CancelledError, Queue
from collections.abc import AsyncIterator
from json import dumps
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallRecoveryFormat,
    ToolCallResult,
    ToolFormat,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.interaction.entities import RESERVED_INPUT_CAPABILITY_NAME
from avalan.model.call import ModelCallContext
from avalan.model.capability import (
    CapabilityBatchRejectionCode,
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
)
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamTerminalOutcome,
    StreamValidationError,
    TextGenerationNonStreamResult,
    local_tool_call_control_frame,
)
from avalan.tool import ToolSet
from avalan.tool.display import ToolDisplayProjection
from avalan.tool.manager import ToolManager


class _DefaultAsyncBoundaryStreamer:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stop_signal = object()
        self.queue: Queue[object] = Queue()

    def put(self, value: object) -> None:
        self.queue.put_nowait(value)

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        self.put(text)
        if stream_end:
            self.put(self.stop_signal)

    def __aiter__(self) -> "_DefaultAsyncBoundaryStreamer":
        return self

    async def __anext__(self) -> str:
        value = await self.queue.get()
        if value is self.stop_signal:
            raise StopAsyncIteration
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, str)
        return value


class _SynchronousGenerationThread:
    def __init__(
        self,
        target: object,
        name: str | None = None,
        daemon: bool | None = None,
    ) -> None:
        assert callable(target)
        self._target = target
        self.name = name
        self.daemon = daemon
        self.ident = 1
        self._alive = False

    def start(self) -> None:
        self._alive = True
        self._target()
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _InterruptCapturingGenerationThread(_SynchronousGenerationThread):
    captured: list[BaseException] = []

    def start(self) -> None:
        self._alive = True
        try:
            self._target()
        except (KeyboardInterrupt, SystemExit) as exc:
            self.captured.append(exc)
        finally:
            self._alive = False


def _default_async_transformers_model() -> TextGenerationModel:
    model = TextGenerationModel(
        "boundary-model",
        TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        ),
        logger=getLogger(),
    )
    model._model = MagicMock()
    model._tokenizer = SimpleNamespace(
        bos_token=None,
        eos_token_id=0,
        chat_template={
            LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}"
        },
        apply_chat_template=lambda *args, **kwargs: [[1]],
    )
    model._tokenize_input = MagicMock(return_value={"input_ids": [[1]]})
    model._log = MagicMock()
    return model


def lookup(query: str) -> str:
    """Look up one value.

    Args:
        query: Value to look up.

    Returns:
        The provided value.
    """
    return query


def _project_lookup_display(*items: object, **kwargs: object) -> object:
    call = kwargs.get("call")
    if not isinstance(call, ToolCall) and items:
        call = items[0]
    assert isinstance(call, ToolCall)
    return ToolDisplayProjection(
        action="lookup",
        target=call.name,
        summary="Look up the requested value.",
    )


setattr(lookup, "tool_display_projector", _project_lookup_display)

_TOOL_SIDE_EFFECT_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_DETECT,
        EventType.TOOL_EXECUTE,
        EventType.TOOL_RESULT,
        EventType.TOOL_DIAGNOSTIC,
        EventType.TOOL_PROGRESS,
        EventType.TOOL_MODEL_RUN,
        EventType.TOOL_MODEL_RESPONSE,
    }
)


class _DummyEngine:
    model_id = "test-model"
    tokenizer = None


def _operation() -> AgentOperation:
    return AgentOperation(
        specification=Specification(role=None, goal=None),
        environment=EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="test-model",
                params={},
            ),
            settings=TransformerEngineSettings(),
        ),
    )


def _manager() -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[ToolSet(namespace="pkg", tools=[lookup])],
        enable_tools=["pkg.lookup"],
        settings=ToolManagerSettings(
            tool_format=ToolFormat.JSON,
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED,
                map={"pkg.lookup": "lookup"},
            ),
            recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK],
        ),
    )


def _special_name_manager() -> tuple[ToolManager, str]:
    namespace = 'pkg."quoted"\\工具'
    canonical_name = f"{namespace}.lookup"
    manager = ToolManager.create_instance(
        available_toolsets=[ToolSet(namespace=namespace, tools=[lookup])],
        enable_tools=[canonical_name],
        settings=ToolManagerSettings(
            tool_format=ToolFormat.JSON,
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.ENCODED,
            ),
            recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK],
        ),
    )
    return manager, canonical_name


def _catalog(manager: ToolManager) -> ModelCapabilityCatalog:
    return ModelCapabilityCatalog.create(
        manager.export_model_capability_seed(),
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            attached_resolution=True,
        ),
    )


def _input_arguments() -> dict[str, object]:
    return {
        "mode": "required",
        "reason": "Need a bounded choice.",
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


def _response(
    *calls: ToolCall,
    provider_family: str = "openai",
) -> TextGenerationResponse:
    async def items() -> AsyncIterator[CanonicalStreamItem]:
        sequence = 0
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
            provider_family=provider_family,
        )
        sequence += 1
        for call in calls:
            assert call.id is not None
            correlation = StreamItemCorrelation(tool_call_id=str(call.id))
            if call.arguments is not None:
                yield CanonicalStreamItem(
                    stream_session_id="provider-stream",
                    run_id="provider-run",
                    turn_id="provider-turn",
                    sequence=sequence,
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    channel=StreamChannel.TOOL_CALL,
                    text_delta=dumps(call.arguments),
                    correlation=correlation,
                    provider_family=provider_family,
                )
                sequence += 1
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": call.name},
                correlation=correlation,
                provider_family=provider_family,
            )
            sequence += 1
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
                provider_family=provider_family,
            )
            sequence += 1
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
            provider_family=provider_family,
        )
        sequence += 1
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
            provider_family=provider_family,
        )

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(),
        use_async_generator=True,
    )


def _lifecycle_response(
    *frames: dict[str, Any],
) -> TextGenerationResponse:
    async def items() -> AsyncIterator[CanonicalStreamItem]:
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        for sequence, frame in enumerate(frames, start=1):
            tool_call_id = frame.get("tool_call_id")
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=sequence,
                kind=cast(StreamItemKind, frame["kind"]),
                channel=StreamChannel.TOOL_CALL,
                text_delta=cast(str | None, frame.get("text_delta")),
                data=cast(Any, frame.get("data")),
                correlation=StreamItemCorrelation(
                    tool_call_id=(
                        tool_call_id if isinstance(tool_call_id, str) else None
                    )
                ),
                provider_family=cast(str | None, frame.get("provider_family")),
            )
        terminal_sequence = len(frames) + 1
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=terminal_sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            stream_session_id="provider-stream",
            run_id="provider-run",
            turn_id="provider-turn",
            sequence=terminal_sequence + 1,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(),
        use_async_generator=True,
    )


def _orchestrator_response(
    response: TextGenerationResponse,
    manager: ToolManager,
    catalog: ModelCapabilityCatalog,
    *,
    confirmation: AsyncMock | None = None,
    enable_tool_parsing: bool = False,
    event_manager: EventManager | None = None,
) -> OrchestratorResponse:
    operation = _operation()
    input_message = Message(role=MessageRole.USER, content="Run the batch.")
    agent = MagicMock(spec=EngineAgent)
    agent.engine = _DummyEngine()
    context = ModelCallContext(
        specification=operation.specification,
        input=input_message,
        capability=catalog,
    )
    return OrchestratorResponse(
        input_message,
        response,
        agent,
        operation,
        {},
        context,
        event_manager=event_manager,
        tool=manager,
        capability=catalog,
        tool_confirm=confirmation,
        enable_tool_parsing=enable_tool_parsing,
    )


def _uncataloged_orchestrator_response(
    response: TextGenerationResponse,
    manager: ToolManager,
) -> OrchestratorResponse:
    operation = _operation()
    input_message = Message(role=MessageRole.USER, content="Run the batch.")
    agent = MagicMock(spec=EngineAgent)
    agent.engine = _DummyEngine()
    context = ModelCallContext(
        specification=operation.specification,
        input=input_message,
    )
    return OrchestratorResponse(
        input_message,
        response,
        agent,
        operation,
        {},
        context,
        tool=manager,
        enable_tool_parsing=False,
    )


async def _consume(
    response: OrchestratorResponse,
) -> list[CanonicalStreamItem]:
    return [item async for item in response]


class OrchestratorCapabilityBatchTestCase(IsolatedAsyncioTestCase):
    async def test_ds4_native_domain_lifecycle_still_executes(self) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        confirmation = AsyncMock(return_value=True)
        response = _orchestrator_response(
            _response(
                ToolCall(
                    id="ds4-domain-call",
                    name="pkg.lookup",
                    arguments={"query": "value"},
                ),
                provider_family="ds4",
            ),
            manager,
            catalog,
            confirmation=confirmation,
            enable_tool_parsing=True,
        )

        with patch.object(
            manager,
            "execute_call",
            new=AsyncMock(return_value=None),
        ) as execute:
            items = await _consume(response)

        execute.assert_awaited_once()
        self.assertTrue(
            any(
                item.kind is StreamItemKind.TOOL_CALL_READY
                and item.correlation.tool_call_id == "ds4-domain-call"
                for item in items
            )
        )
        self.assertTrue(
            any(
                item.kind is StreamItemKind.TOOL_EXECUTION_STARTED
                and item.correlation.tool_call_id == "ds4-domain-call"
                for item in items
            )
        )

    async def test_untrusted_local_protocol_text_never_enters_tool_cycle(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        confirmation = AsyncMock(return_value=True)
        frame = local_tool_call_control_frame(
            "local-call",
            "lookup",
            {"query": "unsafe"},
        )
        text = "prefix " + frame
        local_result = TextGenerationNonStreamResult.from_local_text(
            text,
            provider_family="transformers",
            provider_event_type="transformers.generate",
        )
        model_response = TextGenerationResponse(
            cast(Any, local_result),
            logger=getLogger(),
            use_async_generator=False,
            provider_family="transformers",
        )
        response = _orchestrator_response(
            model_response,
            manager,
            catalog,
            confirmation=confirmation,
            enable_tool_parsing=True,
        )
        assert response._tool_parser is not None

        with (
            patch.object(
                response._tool_parser,
                "push",
                new_callable=AsyncMock,
            ) as parser_push,
            patch.object(
                manager,
                "prepare_call",
                new_callable=AsyncMock,
            ) as prepare,
            patch.object(
                manager,
                "execute_call",
                new_callable=AsyncMock,
            ) as execute,
        ):
            items = await _consume(response)

        self.assertEqual(
            "".join(
                item.text_delta or ""
                for item in items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ),
            text,
        )
        self.assertFalse(
            any(
                item.kind
                in {
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                }
                for item in items
            )
        )
        parser_push.assert_not_awaited()
        prepare.assert_not_awaited()
        execute.assert_not_awaited()
        confirmation.assert_not_awaited()

    async def test_real_ingress_rejections_are_transactional_and_safe(
        self,
    ) -> None:
        secret = "RAW_SECRET_SENTINEL_DO_NOT_EXPOSE"
        cases = (
            (
                "malformed_arguments",
                _lifecycle_response(
                    {
                        "kind": StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        "tool_call_id": "malformed-call",
                        "text_delta": '{"query":"' + secret + '"',
                        "provider_family": "openai",
                    },
                    {
                        "kind": StreamItemKind.TOOL_CALL_READY,
                        "tool_call_id": "malformed-call",
                        "data": {"name": "pkg.lookup"},
                        "provider_family": "openai",
                    },
                    {
                        "kind": StreamItemKind.TOOL_CALL_DONE,
                        "tool_call_id": "malformed-call",
                        "provider_family": "openai",
                    },
                ),
            ),
            (
                "provider_family_conflict",
                _lifecycle_response(
                    {
                        "kind": StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        "tool_call_id": "conflicted-call",
                        "text_delta": dumps({"query": secret}),
                        "provider_family": "openai",
                    },
                    {
                        "kind": StreamItemKind.TOOL_CALL_READY,
                        "tool_call_id": "conflicted-call",
                        "data": {"name": "pkg.lookup"},
                        "provider_family": "anthropic",
                    },
                    {
                        "kind": StreamItemKind.TOOL_CALL_DONE,
                        "tool_call_id": "conflicted-call",
                        "provider_family": "anthropic",
                    },
                ),
            ),
        )
        for name, model_response in cases:
            with self.subTest(name=name):
                manager = _manager()
                catalog = _catalog(manager)
                event_manager = MagicMock(spec=EventManager)
                event_manager.trigger = AsyncMock()
                confirmation = AsyncMock(return_value=True)
                with (
                    patch.object(
                        manager,
                        "describe_tool_call",
                        wraps=manager.describe_tool_call,
                    ) as describe,
                    patch.object(
                        manager, "prepare_call", new_callable=AsyncMock
                    ) as prepare,
                    patch.object(
                        manager, "execute_call", new_callable=AsyncMock
                    ) as execute,
                ):
                    response = _orchestrator_response(
                        model_response,
                        manager,
                        catalog,
                        confirmation=confirmation,
                        event_manager=event_manager,
                    )
                    items = await _consume(response)

                diagnostics = [
                    item
                    for item in items
                    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                ]
                self.assertEqual(len(diagnostics), 1)
                assert isinstance(diagnostics[0].data, dict)
                self.assertEqual(
                    diagnostics[0].data["code"],
                    CapabilityBatchRejectionCode.MALFORMED_CALL.value,
                )
                self.assertNotIn(secret, repr(items))
                self.assertFalse(
                    any(
                        item.kind
                        in {
                            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            StreamItemKind.TOOL_CALL_READY,
                            StreamItemKind.TOOL_CALL_DONE,
                            StreamItemKind.TOOL_EXECUTION_STARTED,
                        }
                        for item in items
                    )
                )
                self.assertTrue(
                    all(
                        call.args[0].type not in _TOOL_SIDE_EFFECT_EVENT_TYPES
                        for call in event_manager.trigger.await_args_list
                    )
                )
                describe.assert_not_called()
                prepare.assert_not_awaited()
                execute.assert_not_awaited()
                confirmation.assert_not_awaited()

    async def test_empty_registry_has_stream_and_to_str_parity(self) -> None:
        source_manager = _manager()
        catalog = _catalog(source_manager)
        empty_manager = ToolManager.create_instance(enable_tools=[])
        call = ToolCall(
            id="provider-input-call",
            name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=cast(Any, _input_arguments()),
        )

        streamed = _orchestrator_response(
            _response(call), empty_manager, catalog
        )
        streamed_items = await _consume(streamed)
        rendered = _orchestrator_response(
            _response(call), empty_manager, catalog
        )
        self.assertEqual(await rendered.to_str(), "")

        for response in (streamed, rendered):
            self.assertIsNotNone(response.task_input_call)
            assert response.task_input_call is not None
            self.assertEqual(
                response.task_input_call.call_id, "provider-input-call"
            )
            self.assertFalse(
                any(
                    item.kind
                    in {
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        StreamItemKind.TOOL_CALL_READY,
                        StreamItemKind.TOOL_CALL_DONE,
                    }
                    for item in response.canonical_items
                )
            )
        self.assertTrue(streamed_items)

    async def test_ready_display_is_deferred_until_domain_acceptance(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        rejected = _orchestrator_response(
            _response(
                ToolCall(
                    id="rejected-call",
                    name="not.advertised",
                    arguments={},
                )
            ),
            manager,
            catalog,
        )
        with (
            patch.object(
                rejected,
                "_tool_call_ready_display_metadata",
                wraps=rejected._tool_call_ready_display_metadata,
            ) as staged_display,
            patch.object(
                rejected,
                "_tool_call_display_projection",
                wraps=rejected._tool_call_display_projection,
            ) as accepted_display,
        ):
            await _consume(rejected)
        staged_display.assert_not_called()
        accepted_display.assert_not_called()

        empty_manager = ToolManager.create_instance(enable_tools=[])
        reserved = _orchestrator_response(
            _response(
                ToolCall(
                    id="reserved-call",
                    name=RESERVED_INPUT_CAPABILITY_NAME,
                    arguments=cast(Any, _input_arguments()),
                )
            ),
            empty_manager,
            catalog,
        )
        with (
            patch.object(
                reserved,
                "_tool_call_ready_display_metadata",
                wraps=reserved._tool_call_ready_display_metadata,
            ) as staged_display,
            patch.object(
                reserved,
                "_tool_call_display_projection",
                wraps=reserved._tool_call_display_projection,
            ) as accepted_display,
        ):
            await _consume(reserved)
        staged_display.assert_not_called()
        accepted_display.assert_not_called()

        accepted = _orchestrator_response(
            _response(
                ToolCall(
                    id="accepted-call",
                    name="pkg.lookup",
                    arguments={"query": "value"},
                )
            ),
            manager,
            catalog,
        )
        with (
            patch.object(
                accepted,
                "_tool_call_ready_display_metadata",
                wraps=accepted._tool_call_ready_display_metadata,
            ) as staged_display,
            patch.object(
                accepted,
                "_tool_call_display_projection",
                wraps=accepted._tool_call_display_projection,
            ) as accepted_display,
            patch.object(
                manager,
                "execute_call",
                new=AsyncMock(return_value=None),
            ),
        ):
            items = await _consume(accepted)
        staged_display.assert_not_called()
        accepted_display.assert_called()
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            cast(
                dict[str, Any],
                ready.metadata["tool_display_projection"],
            )["summary"],
            "Look up the requested value.",
        )

    def test_missing_id_is_rejected_at_canonical_ingress(self) -> None:
        with self.assertRaises(AssertionError):
            CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=0,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": RESERVED_INPUT_CAPABILITY_NAME},
            )

    def test_response_rejects_distinct_context_catalog(self) -> None:
        manager = _manager()
        context_catalog = _catalog(manager)
        other_catalog = _catalog(manager)
        operation = _operation()
        input_message = Message(role=MessageRole.USER, content="Run.")
        agent = MagicMock(spec=EngineAgent)
        agent.engine = _DummyEngine()
        context = ModelCallContext(
            specification=operation.specification,
            input=input_message,
            capability=context_catalog,
        )

        with self.assertRaisesRegex(AssertionError, "must match"):
            OrchestratorResponse(
                input_message,
                _response(),
                agent,
                operation,
                {},
                context,
                tool=manager,
                capability=other_catalog,
                enable_tool_parsing=False,
            )

    async def test_rejected_structured_batches_have_zero_domain_side_effects(
        self,
    ) -> None:
        cases = (
            (
                CapabilityBatchRejectionCode.MIXED_TASK_INPUT_BATCH,
                (
                    ToolCall(
                        id="domain-call",
                        name="pkg.lookup",
                        arguments={"query": "value"},
                    ),
                    ToolCall(
                        id="input-call",
                        name=RESERVED_INPUT_CAPABILITY_NAME,
                        arguments=cast(Any, _input_arguments()),
                    ),
                ),
            ),
            (
                CapabilityBatchRejectionCode.MULTIPLE_TASK_INPUT_CALLS,
                (
                    ToolCall(
                        id="input-one",
                        name=RESERVED_INPUT_CAPABILITY_NAME,
                        arguments=cast(Any, _input_arguments()),
                    ),
                    ToolCall(
                        id="input-two",
                        name=RESERVED_INPUT_CAPABILITY_NAME,
                        arguments=cast(Any, _input_arguments()),
                    ),
                ),
            ),
            (
                CapabilityBatchRejectionCode.MALFORMED_CALL,
                (
                    ToolCall(
                        id="malformed-input",
                        name=RESERVED_INPUT_CAPABILITY_NAME,
                        arguments={"mode": "required"},
                    ),
                ),
            ),
            (
                CapabilityBatchRejectionCode.UNKNOWN_CAPABILITY,
                (
                    ToolCall(
                        id="unknown-call",
                        name="not.advertised",
                        arguments={},
                    ),
                ),
            ),
        )
        self.assertEqual(len(cases), 4)
        for expected_code, calls in cases:
            with self.subTest(code=expected_code.value):
                manager = _manager()
                catalog = _catalog(manager)
                event_manager = MagicMock(spec=EventManager)
                event_manager.trigger = AsyncMock()
                confirmation = AsyncMock(return_value=True)
                with (
                    patch.object(
                        manager,
                        "describe_tool_call",
                        wraps=manager.describe_tool_call,
                    ) as describe,
                    patch.object(
                        manager,
                        "is_tool_call_parallel_safe",
                        wraps=manager.is_tool_call_parallel_safe,
                    ) as parallel_safe,
                    patch.object(
                        manager,
                        "validate_tool_call",
                        wraps=manager.validate_tool_call,
                    ) as validate,
                    patch.object(
                        manager, "prepare_call", new_callable=AsyncMock
                    ) as prepare,
                    patch.object(
                        manager, "execute_call", new_callable=AsyncMock
                    ) as execute,
                ):
                    response = _orchestrator_response(
                        _response(*calls),
                        manager,
                        catalog,
                        confirmation=confirmation,
                        event_manager=event_manager,
                    )
                    items = await _consume(response)

                diagnostics = [
                    cast(dict[str, Any], item.data)
                    for item in items
                    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                    and isinstance(item.data, dict)
                ]
                self.assertIn(
                    expected_code.value, {d["code"] for d in diagnostics}
                )
                self.assertFalse(
                    any(
                        item.kind
                        in {
                            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            StreamItemKind.TOOL_CALL_READY,
                            StreamItemKind.TOOL_CALL_DONE,
                            StreamItemKind.TOOL_EXECUTION_STARTED,
                            StreamItemKind.TOOL_EXECUTION_COMPLETED,
                        }
                        for item in items
                    )
                )
                describe.assert_not_called()
                parallel_safe.assert_not_called()
                validate.assert_not_called()
                prepare.assert_not_awaited()
                execute.assert_not_awaited()
                confirmation.assert_not_awaited()
                self.assertTrue(
                    all(
                        call.args[0].type not in _TOOL_SIDE_EFFECT_EVENT_TYPES
                        for call in event_manager.trigger.await_args_list
                    )
                )
                self.assertEqual(response._attempted_call_signatures, set())
                self.assertEqual(response._tool_cycle_count, 0)
                self.assertEqual(response._consecutive_non_executed_cycles, 0)
                self.assertIsNone(response.task_input_call)

    async def test_missing_id_and_plain_text_reserved_calls_fail_closed(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        confirmation = AsyncMock(return_value=True)
        response = _orchestrator_response(
            _response(),
            manager,
            catalog,
            confirmation=confirmation,
        )
        missing_id = ToolCall(
            id=None,
            name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=cast(Any, _input_arguments()),
        )
        with (
            patch.object(
                response,
                "_response_text_and_calls",
                new=AsyncMock(return_value=("", [missing_id])),
            ),
            patch.object(
                manager, "execute_call", new_callable=AsyncMock
            ) as execute,
        ):
            await response._react(_response())
        self.assertEqual(
            cast(dict[str, Any], response.canonical_items[-1].data)["code"],
            CapabilityBatchRejectionCode.MISSING_CALL_ID.value,
        )
        execute.assert_not_awaited()
        confirmation.assert_not_awaited()
        self.assertIsNone(response.task_input_call)
        self.assertEqual(response._tool_cycle_count, 0)

        plain_response = _orchestrator_response(
            _response(),
            manager,
            catalog,
            confirmation=confirmation,
        )
        plain_output = (
            "<tool_call>"
            + dumps(
                {
                    "id": "synthetic-input",
                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                    "arguments": _input_arguments(),
                }
            )
            + "</tool_call>"
        )
        with patch.object(
            manager, "execute_call", new_callable=AsyncMock
        ) as plain_execute:
            await plain_response._react(
                _response(),
                output=plain_output,
            )
        self.assertEqual(
            cast(dict[str, Any], plain_response.canonical_items[-1].data)[
                "code"
            ],
            CapabilityBatchRejectionCode.NON_STRUCTURED_CALL.value,
        )
        plain_execute.assert_not_awaited()
        confirmation.assert_not_awaited()
        self.assertIsNone(plain_response.task_input_call)
        self.assertEqual(plain_response._tool_cycle_count, 0)

    async def test_valid_reserved_call_preserves_exact_provider_id(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        confirmation = AsyncMock(return_value=True)
        exact_call_id = "provider-call-exact-17"
        with patch.object(
            manager, "execute_call", new_callable=AsyncMock
        ) as execute:
            response = _orchestrator_response(
                _response(
                    ToolCall(
                        id=exact_call_id,
                        name=RESERVED_INPUT_CAPABILITY_NAME,
                        arguments=cast(Any, _input_arguments()),
                    )
                ),
                manager,
                catalog,
                confirmation=confirmation,
            )
            response._reserve_canonical_tool_call_id(exact_call_id)
            await _consume(response)

        execute.assert_not_awaited()
        confirmation.assert_not_awaited()
        self.assertIsNotNone(response.task_input_call)
        assert response.task_input_call is not None
        self.assertEqual(response.task_input_call.call_id, exact_call_id)
        self.assertEqual(response._tool_cycle_count, 0)

    async def test_domain_batch_reaches_manager_as_tool_calls(self) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        confirmation = AsyncMock(return_value=True)
        response = _orchestrator_response(
            _response(
                ToolCall(
                    id="domain-one",
                    name="pkg.lookup",
                    arguments={"query": "one"},
                ),
                ToolCall(
                    id="domain-two",
                    name="pkg.lookup",
                    arguments={"query": "two"},
                ),
            ),
            manager,
            catalog,
            confirmation=confirmation,
        )
        with (
            patch.object(
                manager,
                "describe_tool_call",
                wraps=manager.describe_tool_call,
            ) as describe,
            patch.object(
                manager,
                "execute_call",
                new=AsyncMock(return_value=None),
            ) as execute,
        ):
            items = await _consume(response)

        self.assertEqual(execute.await_count, 2)
        dispatched = [call.args[0] for call in execute.await_args_list]
        self.assertTrue(all(isinstance(call, ToolCall) for call in dispatched))
        self.assertEqual(
            [call.id for call in dispatched], ["domain-one", "domain-two"]
        )
        self.assertEqual(
            [call.name for call in dispatched], ["pkg.lookup", "pkg.lookup"]
        )
        self.assertEqual(
            [call.arguments for call in dispatched],
            [{"query": "one"}, {"query": "two"}],
        )
        self.assertEqual(confirmation.await_count, 2)
        self.assertGreaterEqual(describe.call_count, 2)
        started = [
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED
        ]
        self.assertEqual(len(started), 2)
        self.assertTrue(
            all(
                isinstance(item.metadata, dict)
                and item.metadata["tool_display_projection"]["summary"]
                == "Look up the requested value."
                for item in started
            )
        )
        ready = [
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(len(ready), 2)
        self.assertTrue(
            all(
                cast(
                    dict[str, Any],
                    item.metadata["tool_display_projection"],
                )["summary"]
                == "Look up the requested value."
                for item in ready
            )
        )
        self.assertIsNone(response.task_input_call)

    async def test_domain_call_keeps_canonical_and_provider_ids_separate(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        provider_call_id = "provider-call"
        response = _orchestrator_response(
            _response(
                ToolCall(
                    id=provider_call_id,
                    name="pkg.lookup",
                    arguments={"query": "value"},
                )
            ),
            manager,
            catalog,
        )
        response._reserve_canonical_tool_call_id(provider_call_id)
        agent = cast(MagicMock, response._engine_agent)

        async def continue_model(
            _: ModelCallContext,
        ) -> TextGenerationResponse:
            return _response()

        agent.side_effect = continue_model

        def result(call: ToolCall, *_: object, **__: object) -> ToolCallResult:
            return ToolCallResult(
                id="result",
                name=call.name,
                arguments=call.arguments,
                call=call,
                result="ok",
            )

        with patch.object(
            manager,
            "execute_call",
            new=AsyncMock(side_effect=result),
        ) as execute:
            items = await _consume(response)

        canonical_id = "orchestrator-tool-call-1"
        assert execute.await_args is not None
        dispatched = execute.await_args.args[0]
        self.assertEqual(dispatched.id, canonical_id)
        ready_ids = [
            item.correlation.tool_call_id
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(ready_ids, [canonical_id])

        child_context = agent.call_args.args[0]
        self.assertIs(child_context.capability, catalog)
        self.assertIs(child_context.parent.capability, catalog)
        messages = cast(list[Message], child_context.input)
        provider_message = next(
            message for message in messages if message.tool_calls
        )
        assert provider_message.tool_calls is not None
        self.assertEqual(provider_message.tool_calls[0].id, provider_call_id)

    def test_explicit_catalog_is_injected_into_model_call_context(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        operation = _operation()
        input_message = Message(role=MessageRole.USER, content="Run.")
        agent = MagicMock(spec=EngineAgent)
        agent.engine = _DummyEngine()
        context = ModelCallContext(
            specification=operation.specification,
            input=input_message,
        )

        response = OrchestratorResponse(
            input_message,
            _response(),
            agent,
            operation,
            {},
            context,
            tool=manager,
            capability=catalog,
            enable_tool_parsing=False,
        )

        self.assertIs(response._context.capability, catalog)
        self.assertIs(response._capability_catalog, catalog)

    def test_capability_ingress_failures_remain_private_until_rejection(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)

        parser_response = _orchestrator_response(_response(), manager, catalog)
        parser_response._queue_parser_output(
            SimpleNamespace(kind=StreamItemKind.STREAM_DIAGNOSTIC)
        )
        self.assertTrue(parser_response._staged_tool_batch_invalid)
        self.assertEqual(parser_response._drain_tool_call_batch(), [])
        parser_diagnostic = cast(
            dict[str, Any], parser_response.canonical_items[-1].data
        )
        self.assertEqual(
            parser_diagnostic["code"],
            CapabilityBatchRejectionCode.MALFORMED_CALL.value,
        )

        missing_provider_id = _orchestrator_response(
            _response(), manager, catalog
        )
        self.assertIsNone(
            missing_provider_id._append_canonical_provider_tool_call_item(
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_READY,
                    data={"name": "pkg.lookup"},
                ),
                correlation=StreamItemCorrelation(),
            )
        )
        self.assertEqual(
            missing_provider_id._staged_tool_batch_rejection_code,
            CapabilityBatchRejectionCode.MISSING_CALL_ID,
        )

        missing_arguments = _orchestrator_response(
            _response(), manager, catalog
        )
        self.assertIsNone(
            missing_arguments._append_canonical_provider_tool_call_item(
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                ),
                correlation=StreamItemCorrelation(
                    tool_call_id="missing-arguments"
                ),
            )
        )
        self.assertTrue(missing_arguments._staged_tool_batch_invalid)

        missing_canonical_id = _orchestrator_response(
            _response(), manager, catalog
        )
        self.assertIsNone(
            missing_canonical_id._append_canonical_tool_call_lifecycle_item(
                StreamItemKind.TOOL_CALL_READY,
                data={"name": "pkg.lookup"},
                correlation=StreamItemCorrelation(),
            )
        )
        self.assertEqual(
            missing_canonical_id._staged_tool_batch_rejection_code,
            CapabilityBatchRejectionCode.MISSING_CALL_ID,
        )

        metadata = {"private": "until-accepted"}
        self.assertIs(
            missing_canonical_id._tool_call_ready_display_metadata(
                "call", {"name": "pkg.lookup"}, metadata
            ),
            metadata,
        )
        self.assertFalse(
            any(
                item.kind
                in {
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                }
                for item in (
                    *missing_provider_id.canonical_items,
                    *missing_arguments.canonical_items,
                    *missing_canonical_id.canonical_items,
                )
            )
        )

    def test_legacy_lifecycle_rejects_provider_family_changes(self) -> None:
        response = _uncataloged_orchestrator_response(_response(), _manager())
        correlation = StreamItemCorrelation(tool_call_id="legacy-call")
        response._append_canonical_tool_call_lifecycle_item(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta="{}",
            correlation=correlation,
            provider_family="openai",
        )

        with self.assertRaisesRegex(
            StreamValidationError, "provider family changed"
        ):
            response._append_canonical_tool_call_lifecycle_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta="{}",
                correlation=correlation,
                provider_family="anthropic",
            )

    async def test_cross_provider_complete_batch_is_discarded_atomically(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        response = _orchestrator_response(
            _lifecycle_response(
                {
                    "kind": StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    "tool_call_id": "openai-call",
                    "text_delta": dumps({"query": "one"}),
                    "provider_family": "openai",
                },
                {
                    "kind": StreamItemKind.TOOL_CALL_READY,
                    "tool_call_id": "openai-call",
                    "data": {"name": "lookup"},
                    "provider_family": "openai",
                },
                {
                    "kind": StreamItemKind.TOOL_CALL_DONE,
                    "tool_call_id": "openai-call",
                    "provider_family": "openai",
                },
                {
                    "kind": StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    "tool_call_id": "anthropic-call",
                    "text_delta": dumps({"query": "two"}),
                    "provider_family": "anthropic",
                },
                {
                    "kind": StreamItemKind.TOOL_CALL_READY,
                    "tool_call_id": "anthropic-call",
                    "data": {"name": "lookup"},
                    "provider_family": "anthropic",
                },
                {
                    "kind": StreamItemKind.TOOL_CALL_DONE,
                    "tool_call_id": "anthropic-call",
                    "provider_family": "anthropic",
                },
            ),
            manager,
            catalog,
        )
        with patch.object(
            manager, "execute_call", new_callable=AsyncMock
        ) as execute:
            items = await _consume(response)

        diagnostics = [
            cast(dict[str, Any], item.data)
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            and isinstance(item.data, dict)
        ]
        self.assertEqual(
            [diagnostic["code"] for diagnostic in diagnostics],
            [CapabilityBatchRejectionCode.MALFORMED_CALL.value],
        )
        execute.assert_not_awaited()
        self.assertFalse(response._staged_tool_batch_present)
        self.assertFalse(
            any(
                item.kind
                in {
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                }
                for item in items
            )
        )

    def test_complete_batch_empty_malformed_and_registry_guards(self) -> None:
        manager = _manager()
        catalog = _catalog(manager)

        untouched = _orchestrator_response(_response(), manager, catalog)
        self.assertEqual(untouched._drain_tool_call_batch(), [])

        empty = _orchestrator_response(_response(), manager, catalog)
        empty._staged_tool_batch_present = True
        self.assertEqual(empty._classify_complete_tool_call_batch([]), [])
        self.assertFalse(empty._staged_tool_batch_present)

        malformed = _orchestrator_response(_response(), manager, catalog)
        malformed._staged_tool_batch_present = True
        self.assertIsNone(
            malformed._classify_complete_tool_call_batch(
                [ToolCall(id="malformed", name="", arguments={})]
            )
        )
        malformed_diagnostic = cast(
            dict[str, Any], malformed.canonical_items[-1].data
        )
        self.assertEqual(
            malformed_diagnostic["code"],
            CapabilityBatchRejectionCode.MALFORMED_CALL.value,
        )
        self.assertFalse(malformed._staged_tool_batch_present)

        no_registry = _orchestrator_response(
            _response(),
            ToolManager.create_instance(enable_tools=[]),
            catalog,
        )
        no_registry._staged_tool_batch_present = True
        provider_name = catalog.provider_name(
            "pkg.lookup", provider_family="openai"
        )
        with self.assertRaisesRegex(RuntimeError, "require a tool registry"):
            no_registry._classify_complete_tool_call_batch(
                [
                    ToolCall(
                        id="accepted-without-registry",
                        name="pkg.lookup",
                        arguments={"query": "value"},
                        provider_name=provider_name,
                    )
                ]
            )
        self.assertFalse(no_registry._staged_tool_batch_present)

    def test_anonymous_and_diagnostic_provider_correlation_is_preserved(
        self,
    ) -> None:
        manager = _manager()
        response = _orchestrator_response(
            _response(), manager, _catalog(manager)
        )
        anonymous = ToolCall(id=None, name="pkg.lookup", arguments={})
        decoded = ToolCall(
            id="provider-anonymous",
            name="pkg.lookup",
            arguments={"query": "value"},
        )
        self.assertEqual(
            response._canonical_domain_calls([anonymous], [decoded]),
            [decoded],
        )
        self.assertIs(
            response._provider_facing_tool_call(anonymous), anonymous
        )

        unanchored = ToolCallDiagnostic(
            id="unanchored",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Malformed call.",
        )
        self.assertIs(
            response._provider_facing_tool_outcome(unanchored), unanchored
        )

        anchored = ToolCallDiagnostic(
            id="anchored",
            call_id="canonical-call",
            requested_name="pkg.lookup",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown call.",
        )
        response._provider_tool_call_ids_by_canonical_id["canonical-call"] = (
            "provider-call"
        )
        restored = response._provider_facing_tool_outcome(anchored)
        assert isinstance(restored, ToolCallDiagnostic)
        self.assertEqual(restored.call_id, "provider-call")


class TransformersAsyncOrchestratorBoundaryTestCase(IsolatedAsyncioTestCase):
    async def test_default_async_success_publishes_one_complete_frame(
        self,
    ) -> None:
        manager, canonical_name = _special_name_manager()
        catalog = _catalog(manager)
        call_id = 'call-"quoted"\\path-雪'
        provider_name = catalog.provider_name(
            canonical_name, provider_family="transformers"
        )
        arguments = {"query": "café 雪"}
        result_text = 'result-"quoted"\\path-終'
        frame = local_tool_call_control_frame(
            call_id,
            provider_name,
            arguments,
        )
        model = _default_async_transformers_model()
        confirmation = AsyncMock(return_value=True)

        def generate(
            *args: object,
            streamer: _DefaultAsyncBoundaryStreamer | None = None,
            **kwargs: object,
        ) -> None:
            assert streamer is not None
            for character in frame:
                streamer.put(character)
            streamer.put(streamer.stop_signal)

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                _DefaultAsyncBoundaryStreamer,
            ),
            patch(
                "avalan.model.nlp.text.generation.Thread",
                _SynchronousGenerationThread,
            ),
            patch.object(model, "_generate_output", side_effect=generate),
        ):
            model_response = await model("run", capability=catalog)
            self.assertTrue(model_response.is_async_generator)
            response = _orchestrator_response(
                model_response,
                manager,
                catalog,
                confirmation=confirmation,
                enable_tool_parsing=True,
            )
            agent = cast(MagicMock, response._engine_agent)

            async def continue_model(
                _: ModelCallContext,
            ) -> TextGenerationResponse:
                return _response()

            agent.side_effect = continue_model

            def result(
                call: ToolCall, *_: object, **__: object
            ) -> ToolCallResult:
                return ToolCallResult(
                    id='result-"quoted"\\path-雪',
                    name=call.name,
                    arguments=call.arguments,
                    call=call,
                    result=result_text,
                )

            with patch.object(
                manager,
                "execute_call",
                new=AsyncMock(side_effect=result),
            ) as execute:
                items = await _consume(response)

        execute.assert_awaited_once()
        confirmation.assert_awaited_once()
        assert execute.await_args is not None
        dispatched = execute.await_args.args[0]
        self.assertEqual(dispatched.id, call_id)
        self.assertEqual(dispatched.name, canonical_name)
        self.assertEqual(dispatched.arguments, arguments)
        tool_items = [
            item
            for item in items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        ]
        self.assertEqual(
            {item.correlation.tool_call_id for item in tool_items},
            {call_id},
        )
        child_context = agent.call_args.args[0]
        messages = cast(list[Message], child_context.input)
        assistant_message = next(
            message for message in messages if message.tool_calls
        )
        assert assistant_message.tool_calls is not None
        self.assertEqual(assistant_message.tool_calls[0].id, call_id)
        self.assertEqual(
            assistant_message.tool_calls[0].name,
            canonical_name,
        )
        result_message = next(
            message
            for message in messages
            if message.tool_call_result is not None
        )
        assert result_message.tool_call_result is not None
        self.assertEqual(result_message.tool_call_result.call.id, call_id)
        self.assertEqual(result_message.name, canonical_name)
        self.assertEqual(result_message.content, dumps(result_text))
        self.assertEqual(
            [item.kind for item in items][-2:],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_untrusted_response_discard_drains_all_queued_effects(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        task_input = ToolCall(
            id="trusted-input",
            name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=cast(Any, _input_arguments()),
        )
        response = _orchestrator_response(
            _response(task_input),
            manager,
            catalog,
        )
        await _consume(response)
        self.assertIsNotNone(response.task_input_call)
        queued_calls = (
            ToolCall(
                id="stale-domain-1",
                name="pkg.lookup",
                arguments={"query": "first"},
            ),
            ToolCall(
                id="stale-domain-2",
                name="pkg.lookup",
                arguments={"query": "second"},
            ),
        )
        for call in queued_calls:
            response._calls.put_nowait(call)
        response._staged_tool_batch_present = True
        response._classified_tool_call_object_ids = {
            id(call) for call in queued_calls
        }

        response._discard_untrusted_response_tool_call_batch()

        self.assertTrue(response._calls.empty())
        self.assertIsNone(response.task_input_call)
        self.assertFalse(response._staged_tool_batch_present)
        self.assertEqual(response._classified_tool_call_object_ids, set())

    async def test_default_async_abnormal_terminations_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                "late_worker_runtime_error",
                "worker",
                RuntimeError("late worker failure"),
                StreamItemKind.STREAM_ERRORED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "iterator_failure",
                "iterator",
                RuntimeError("iterator failure"),
                StreamItemKind.STREAM_ERRORED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "asyncio_cancellation",
                "iterator",
                CancelledError("cancelled generation"),
                StreamItemKind.STREAM_CANCELLED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "worker_generator_exit_domain",
                "worker",
                GeneratorExit("generation worker exited"),
                StreamItemKind.STREAM_ERRORED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "worker_generator_exit_control",
                "worker",
                GeneratorExit("generation worker exited"),
                StreamItemKind.STREAM_ERRORED,
                RESERVED_INPUT_CAPABILITY_NAME,
                _input_arguments(),
            ),
            (
                "worker_cancellation_domain",
                "worker",
                CancelledError("generation worker cancelled"),
                StreamItemKind.STREAM_CANCELLED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "worker_cancellation_control",
                "worker",
                CancelledError("generation worker cancelled"),
                StreamItemKind.STREAM_CANCELLED,
                RESERVED_INPUT_CAPABILITY_NAME,
                _input_arguments(),
            ),
            (
                "worker_keyboard_interrupt_domain",
                "worker",
                KeyboardInterrupt("generation worker interrupted"),
                StreamItemKind.STREAM_ERRORED,
                "pkg.lookup",
                {"query": "complete-looking close"},
            ),
            (
                "worker_system_exit_control",
                "worker",
                SystemExit("generation worker exited"),
                StreamItemKind.STREAM_ERRORED,
                RESERVED_INPUT_CAPABILITY_NAME,
                _input_arguments(),
            ),
        )

        for (
            name,
            failure_source,
            failure,
            terminal_kind,
            canonical_name,
            arguments,
        ) in cases:
            with self.subTest(name=name):
                manager = _manager()
                catalog = _catalog(manager)
                call_id = f'{name}-"quoted"\\path-雪'
                provider_name = catalog.provider_name(
                    canonical_name,
                    provider_family="transformers",
                )
                frame = local_tool_call_control_frame(
                    call_id,
                    provider_name,
                    arguments,
                )
                model = _default_async_transformers_model()
                confirmation = AsyncMock(return_value=True)
                process_interrupt = isinstance(
                    failure,
                    KeyboardInterrupt | SystemExit,
                )
                thread_class = (
                    _InterruptCapturingGenerationThread
                    if process_interrupt
                    else _SynchronousGenerationThread
                )
                _InterruptCapturingGenerationThread.captured.clear()

                def generate(
                    *args: object,
                    streamer: _DefaultAsyncBoundaryStreamer | None = None,
                    **kwargs: object,
                ) -> None:
                    assert streamer is not None
                    for character in frame:
                        streamer.put(character)
                    if failure_source == "worker":
                        streamer.put(streamer.stop_signal)
                        raise failure
                    streamer.put(failure)

                with (
                    patch(
                        "avalan.model.nlp.text.generation."
                        "AsyncTextIteratorStreamer",
                        _DefaultAsyncBoundaryStreamer,
                    ),
                    patch(
                        "avalan.model.nlp.text.generation.Thread",
                        thread_class,
                    ),
                    patch.object(
                        model, "_generate_output", side_effect=generate
                    ),
                ):
                    model_response = await model("run", capability=catalog)
                    self.assertTrue(model_response.is_async_generator)
                    response = _orchestrator_response(
                        model_response,
                        manager,
                        catalog,
                        confirmation=confirmation,
                        enable_tool_parsing=True,
                    )
                    agent = cast(MagicMock, response._engine_agent)
                    with (
                        patch.object(
                            manager,
                            "prepare_call",
                            new=AsyncMock(),
                        ) as prepare,
                        patch.object(
                            manager,
                            "execute_call",
                            new=AsyncMock(),
                        ) as execute,
                    ):
                        items = await _consume(response)

                self.assertEqual(
                    "".join(
                        item.text_delta or ""
                        for item in items
                        if item.kind is StreamItemKind.ANSWER_DELTA
                    ),
                    frame,
                )
                self.assertFalse(
                    any(
                        item.kind
                        in {
                            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                            StreamItemKind.TOOL_CALL_READY,
                            StreamItemKind.TOOL_CALL_DONE,
                            StreamItemKind.TOOL_EXECUTION_STARTED,
                            StreamItemKind.STREAM_INPUT_REQUIRED,
                        }
                        for item in items
                    )
                )
                self.assertEqual(
                    [item.kind for item in items][-2:],
                    [terminal_kind, StreamItemKind.STREAM_CLOSED],
                )
                self.assertEqual(
                    sum(item.kind is terminal_kind for item in items), 1
                )
                prepare.assert_not_awaited()
                execute.assert_not_awaited()
                confirmation.assert_not_awaited()
                agent.assert_not_called()
                self.assertIsNone(response.task_input_call)
                self.assertEqual(response._tool_cycle_count, 0)
                self.assertEqual(response._call_history, [])
                if process_interrupt:
                    self.assertEqual(
                        _InterruptCapturingGenerationThread.captured,
                        [failure],
                    )

    async def test_default_async_response_explicit_close_is_idempotent(
        self,
    ) -> None:
        manager = _manager()
        catalog = _catalog(manager)
        frame = local_tool_call_control_frame(
            "close-before-eof",
            catalog.provider_name(
                "pkg.lookup", provider_family="transformers"
            ),
            {"query": "never trusted"},
        )
        model = _default_async_transformers_model()

        def generate(
            *args: object,
            streamer: _DefaultAsyncBoundaryStreamer | None = None,
            **kwargs: object,
        ) -> None:
            assert streamer is not None
            streamer.put(frame)

        with (
            patch(
                "avalan.model.nlp.text.generation.AsyncTextIteratorStreamer",
                _DefaultAsyncBoundaryStreamer,
            ),
            patch(
                "avalan.model.nlp.text.generation.Thread",
                _SynchronousGenerationThread,
            ),
            patch.object(model, "_generate_output", side_effect=generate),
        ):
            model_response = await model("run", capability=catalog)
            iterator = model_response.__aiter__()
            started = await iterator.__anext__()
            self.assertIs(started.kind, StreamItemKind.STREAM_STARTED)
            await model_response.aclose()
            await model_response.aclose()

        with self.assertRaises(StopAsyncIteration):
            await iterator.__anext__()
        self.assertEqual(await model_response.to_str(), "")
