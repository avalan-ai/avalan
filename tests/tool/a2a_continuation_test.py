from asyncio import CancelledError, Event, create_task
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import UUID

from a2a.types import a2a_pb2

from avalan.agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AgentExecution,
    AttachedInteractionRuntime,
    BranchInteractionBroker,
    DurableInteractionRuntime,
)
from avalan.entities import ToolCallContext
from avalan.interaction.a2a import (
    A2A_INPUT_EXTENSION_URI,
    A2AInputRequestMetadata,
)
from avalan.interaction.a2a_continuation import (
    A2AInputRequiredError,
    A2ARemoteInputContinuation,
    A2AToolContinuationCheckpoint,
)
from avalan.interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from avalan.interaction.entities import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelledResolution,
    ConfirmationAnswer,
    ConfirmationQuestion,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    InputRequestId,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RunId,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    UserId,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.policy import InteractionActor
from avalan.tool import a2a as a2a_module
from avalan.tool.a2a import A2ACallTool

_AGENT_ID = "00000000-0000-0000-0000-000000000001"


class _HttpClient:
    def __init__(
        self,
        *,
        echo_extension: bool | tuple[bool, ...] = True,
    ) -> None:
        self.event_hooks: dict[str, list[object]] = {"response": []}
        self.echo_extension = echo_extension


class _SdkClient:
    def __init__(
        self,
        batches: Sequence[Sequence[object]],
        http_client: _HttpClient,
    ) -> None:
        self.batches = batches
        self.http_client = http_client
        self.requests: list[Any] = []
        self.contexts: list[Any] = []
        self.cancel_requests: list[Any] = []
        self.cancel_contexts: list[Any] = []
        self.cancel_state = a2a_pb2.TaskState.TASK_STATE_CANCELED

    async def send_message(
        self,
        request: object,
        *,
        context: object = None,
    ) -> AsyncIterator[object]:
        self.requests.append(request)
        self.contexts.append(context)
        parameters = getattr(context, "service_parameters", None) or {}
        requested = parameters.get("A2A-Extensions", "")
        echo_extension = self.http_client.echo_extension
        echo = (
            echo_extension[len(self.requests) - 1]
            if isinstance(echo_extension, tuple)
            else echo_extension
        )
        response = SimpleNamespace(
            request=SimpleNamespace(
                headers={"A2A-Extensions": requested},
            ),
            headers=({"A2A-Extensions": requested} if echo else {}),
        )
        for hook in tuple(self.http_client.event_hooks["response"]):
            await cast(Any, hook)(response)
        for item in self.batches[len(self.requests) - 1]:
            yield item

    async def cancel_task(
        self,
        request: Any,
        *,
        context: object = None,
    ) -> object:
        self.cancel_requests.append(request)
        self.cancel_contexts.append(context)
        return a2a_pb2.Task(
            id=request.id,
            context_id="remote-context",
            status=a2a_pb2.TaskStatus(state=self.cancel_state),
        )


class _Broker:
    def __init__(self, result: InteractionRequestResult) -> None:
        self.result = result
        self.requests: list[InteractionBrokerRequest] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        return self.result

    async def cancel_scope(self, _command: object) -> object:
        return object()


class _BlockingBroker(_Broker):
    def __init__(self, result: InteractionRequestResult) -> None:
        super().__init__(result)
        self.started = Event()
        self.cancelled = False

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        self.started.set()
        try:
            await Event().wait()
        except CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("blocking broker unexpectedly resumed")


class _BlockingSdkClient(_SdkClient):
    def __init__(self, http_client: _HttpClient) -> None:
        super().__init__([[]], http_client)
        self.started = Event()
        self.cancelled = False

    async def send_message(
        self,
        request: object,
        *,
        context: object = None,
    ) -> AsyncIterator[object]:
        self.requests.append(request)
        self.contexts.append(context)
        parameters = getattr(context, "service_parameters", None) or {}
        requested = parameters.get("A2A-Extensions", "")
        response = SimpleNamespace(
            request=SimpleNamespace(
                headers={"A2A-Extensions": requested},
            ),
            headers={"A2A-Extensions": requested},
        )
        for hook in tuple(self.http_client.event_hooks["response"]):
            await cast(Any, hook)(response)
        self.started.set()
        try:
            await Event().wait()
        except CancelledError:
            self.cancelled = True
            raise
        if False:
            yield object()


async def _handler(_context: object) -> object:
    raise AssertionError("the fake broker owns the terminal result")


async def _stager(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("incapable durable A2A must not stage input")


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-origin"),
        turn_id=TurnId("turn-origin"),
        agent_id=AgentId(_AGENT_ID),
        branch_id=BranchId("branch-origin"),
        model_call_id=ModelCallId("model-origin"),
        stream_session_id=StreamSessionId("stream-origin"),
        task_id=TaskId("local-task"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent.toml",
            agent_definition_revision="revision",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model",
            tool_revision="tools",
            capability_revision="capabilities",
        ),
        principal=PrincipalScope(user_id=UserId("user-origin")),
    )


def _result(resolution: object) -> InteractionRequestResult:
    return cast(
        InteractionRequestResult,
        SimpleNamespace(
            delivery=SimpleNamespace(
                record=SimpleNamespace(
                    request=SimpleNamespace(resolution=resolution)
                )
            )
        ),
    )


def _context(
    resolution: object,
    *,
    broker: _Broker | None = None,
) -> tuple[ToolCallContext, _Broker]:
    origin = _origin()
    broker = broker or _Broker(_result(resolution))
    actor = InteractionActor(principal=origin.principal)
    runtime = AttachedInteractionRuntime(
        broker=cast(Any, broker),
        actor=actor,
        handler=cast(Any, _handler),
    )
    execution = cast(
        AgentExecution,
        SimpleNamespace(origin=origin, interaction_runtime=runtime),
    )
    return (
        ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=execution,
            execution_origin=origin,
            interaction_broker=cast(BranchInteractionBroker, broker),
        ),
        broker,
    )


def _struct(payload: dict[str, object]) -> Any:
    value = import_module("google.protobuf.struct_pb2").Struct()
    value.update(payload)
    return value


def _value(payload: dict[str, object]) -> Any:
    value = import_module("google.protobuf.struct_pb2").Value()
    value.struct_value.update(payload)
    return value


def _input_required(
    *,
    request_id: str = "remote-request",
    message_id: str = "remote-input-message",
) -> a2a_pb2.StreamResponse:
    metadata = {
        "kind": "request",
        "request_id": request_id,
        "required": True,
        "questions": [
            {
                "question_id": "confirm",
                "kind": "confirmation",
                "prompt": "Continue?",
                "required": True,
                "choices": [],
                "allow_other": False,
            }
        ],
    }
    return a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="remote-task",
            context_id="remote-context",
            status=a2a_pb2.TaskStatus(
                state=a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED,
                message=a2a_pb2.Message(
                    message_id=message_id,
                    task_id="remote-task",
                    context_id="remote-context",
                    role=a2a_pb2.Role.ROLE_AGENT,
                    parts=[
                        a2a_pb2.Part(
                            text="Additional input is required.\n1. Continue?"
                        )
                    ],
                    metadata=_struct({A2A_INPUT_EXTENSION_URI: metadata}),
                    extensions=[A2A_INPUT_EXTENSION_URI],
                ),
            ),
        )
    )


def _status(state: Any) -> a2a_pb2.StreamResponse:
    return a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="remote-task",
            context_id="remote-context",
            status=a2a_pb2.TaskStatus(state=state),
        )
    )


def _remote_continuation() -> A2ARemoteInputContinuation:
    request = A2AInputRequestMetadata(
        request_id=InputRequestId("remote-request"),
        required=True,
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
    )
    return A2ARemoteInputContinuation(
        request=request,
        request_text="Additional input is required. 1. Continue?",
        task_id="remote-task",
        context_id="remote-context",
        prior_message_id="remote-input-message",
        prior_content=(),
        ttl_seconds=300,
        input_cycle_count=1,
    )


def _checkpoint(
    remote: A2ARemoteInputContinuation,
    arguments: dict[str, object] | None = None,
) -> A2AToolContinuationCheckpoint:
    return A2AToolContinuationCheckpoint(
        call_id="call-input",
        canonical_name="a2a.call",
        provider_name="a2a.call",
        provider_name_encoded=False,
        arguments={
            "uri": "https://peer.example/a2a",
            "name": "remote.skill",
            "arguments": cast(Any, arguments or {}),
        },
        remote=remote,
        interaction_fingerprint_counts=(("input", 1),),
    )


def _input_message_payload(
    *,
    request_id: str = "remote-request",
    message_id: str = "remote-input-message",
) -> dict[str, object]:
    payload = import_module("google.protobuf.json_format").MessageToDict(
        _input_required(request_id=request_id, message_id=message_id)
    )
    update = cast(Mapping[str, object], payload["statusUpdate"])
    status = cast(Mapping[str, object], update["status"])
    return a2a_module._message_payload(
        cast(Mapping[str, object], status["message"])
    )


def _resolution(
    resolution_type: Any,
    request_id: str = "local-request",
) -> Any:
    return resolution_type(
        request_id=InputRequestId(request_id),
        provenance=AnswerProvenance.POLICY,
        resolved_at=datetime.now(UTC),
    )


class A2AContinuationTestCase(IsolatedAsyncioTestCase):
    async def test_durable_runtime_does_not_advertise_without_bridge(
        self,
    ) -> None:
        origin = _origin()
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=cast(Any, _stager),
        )
        execution = cast(
            AgentExecution,
            SimpleNamespace(origin=origin, interaction_runtime=runtime),
        )
        broker = _Broker(
            _result(
                DeclinedResolution(
                    request_id=InputRequestId("local-request"),
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=datetime.now(UTC),
                )
            )
        )
        context = ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=execution,
            execution_origin=origin,
            interaction_broker=cast(BranchInteractionBroker, broker),
        )
        http_client = _HttpClient()
        client = _SdkClient(
            [[_status(a2a_pb2.TaskState.TASK_STATE_COMPLETED)]],
            http_client,
        )
        cards: list[Any] = []

        async def create_client(
            card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            cards.append(card)
            return client

        with patch("a2a.client.create_client", new=create_client):
            await A2ACallTool(
                client_params={"httpx_client": http_client},
                call_params={"request_id": "initial-request"},
            )(
                "https://peer.example/a2a",
                "remote.skill",
                {},
                context=context,
            )

        self.assertEqual(list(cards[0].capabilities.extensions), [])
        self.assertIsNone(client.contexts[0].service_parameters)
        self.assertEqual(broker.requests, [])

    async def test_durable_runtime_routes_to_typed_checkpoint_callback(
        self,
    ) -> None:
        origin = _origin()
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=cast(Any, _stager),
        )
        execution = cast(
            AgentExecution,
            SimpleNamespace(origin=origin, interaction_runtime=runtime),
        )
        context = ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=execution,
            execution_origin=origin,
            durable_a2a_input=True,
        )
        http_client = _HttpClient()
        client = _SdkClient([[_input_required()]], http_client)
        cards: list[Any] = []

        async def create_client(
            card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            cards.append(card)
            return client

        with patch("a2a.client.create_client", new=create_client):
            with self.assertRaises(A2AInputRequiredError) as raised:
                await A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {"input_string": "Start"},
                    context=context,
                )

        remote = raised.exception.continuation
        self.assertEqual(remote.task_id, "remote-task")
        self.assertEqual(remote.context_id, "remote-context")
        self.assertEqual(
            remote.prior_message_id,
            "remote-input-message",
        )
        self.assertEqual(
            cards[0].capabilities.extensions[0].uri,
            A2A_INPUT_EXTENSION_URI,
        )
        self.assertEqual(len(client.requests), 1)
        resumed_http_client = _HttpClient()
        resumed_client = _SdkClient(
            [
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _status(a2a_pb2.TaskState.TASK_STATE_COMPLETED),
                ]
            ],
            resumed_http_client,
        )

        async def create_resumed_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return resumed_client

        with patch(
            "a2a.client.create_client",
            new=create_resumed_client,
        ):
            result = await A2ACallTool(
                client_params={"httpx_client": resumed_http_client},
            ).resume_input(
                _checkpoint(remote, {"input_string": "Start"}),
                DeclinedResolution(
                    request_id=InputRequestId("remote-request"),
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=datetime.now(UTC),
                ),
                context=ToolCallContext(),
            )

        self.assertEqual(
            cast(dict[str, object], result["structuredContent"])["state"],
            "TASK_STATE_COMPLETED",
        )
        self.assertEqual(len(resumed_client.requests), 1)
        follow_up = resumed_client.requests[0].message
        self.assertEqual(follow_up.task_id, "remote-task")
        self.assertEqual(follow_up.context_id, "remote-context")

    async def test_routes_two_requests_and_continues_same_task(self) -> None:
        resolution = AnsweredResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
            ),
        )
        context, broker = _context(resolution)
        http_client = _HttpClient()
        client = _SdkClient(
            [
                [_input_required()],
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _input_required(
                        request_id="remote-request-two",
                        message_id="remote-input-message-two",
                    ),
                ],
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _status(a2a_pb2.TaskState.TASK_STATE_COMPLETED),
                ],
            ],
            http_client,
        )
        cards: list[Any] = []

        async def create_client(
            card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            cards.append(card)
            self.assertIs(
                getattr(client_config, "httpx_client"),
                http_client,
            )
            return client

        tool = A2ACallTool(
            client_params={"httpx_client": http_client},
            call_params={"request_id": "initial-request", "timeout": 1},
        )
        with (
            patch("a2a.client.create_client", new=create_client),
            patch(
                "avalan.tool.a2a.uuid4",
                side_effect=("follow-up-one", "follow-up-two"),
            ),
        ):
            result = await tool(
                "https://peer.example/a2a",
                "remote.skill",
                {"input_string": "Start"},
                context=context,
            )

        structured = cast(dict[str, object], result["structuredContent"])
        self.assertEqual(
            structured["state"],
            "TASK_STATE_COMPLETED",
        )
        self.assertEqual(len(client.requests), 3)
        for follow_up, message_id, request_id in zip(
            (request.message for request in client.requests[1:]),
            ("follow-up-one", "follow-up-two"),
            ("remote-request", "remote-request-two"),
            strict=True,
        ):
            self.assertEqual(follow_up.task_id, "remote-task")
            self.assertEqual(follow_up.context_id, "remote-context")
            self.assertEqual(follow_up.message_id, message_id)
            self.assertNotEqual(follow_up.message_id, request_id)
            self.assertEqual(
                list(follow_up.extensions),
                [A2A_INPUT_EXTENSION_URI],
            )
            metadata = import_module(
                "google.protobuf.json_format"
            ).MessageToDict(follow_up.metadata)
            self.assertEqual(
                metadata[A2A_INPUT_EXTENSION_URI],
                {
                    "kind": "resolution",
                    "request_id": request_id,
                    "action": "accept",
                    "answers": {"confirm": True},
                },
            )
        self.assertEqual(len(broker.requests), 2)
        routed = broker.requests[0]
        self.assertIs(routed.origin, context.execution_origin)
        self.assertIsNotNone(routed.handler)
        self.assertIn("remote-task", routed.context_label or "")
        self.assertIn("remote-context", routed.context_label or "")
        self.assertEqual(
            client.contexts[0].service_parameters["A2A-Extensions"],
            A2A_INPUT_EXTENSION_URI,
        )
        self.assertEqual(
            cards[0].capabilities.extensions[0].uri,
            A2A_INPUT_EXTENSION_URI,
        )
        self.assertEqual(http_client.event_hooks["response"], [])

    async def test_missing_echo_and_auth_required_never_enter_broker(
        self,
    ) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        for responses, echo, expected in (
            ([[_input_required()]], False, "not activated"),
            (
                [[_status(a2a_pb2.TaskState.TASK_STATE_AUTH_REQUIRED)]],
                True,
                "TASK_STATE_AUTH_REQUIRED",
            ),
        ):
            with self.subTest(expected=expected):
                context, broker = _context(resolution)
                http_client = _HttpClient(echo_extension=echo)
                client = _SdkClient(responses, http_client)

                async def create_client(
                    _card: object,
                    *,
                    client_config: object,
                ) -> _SdkClient:
                    return client

                with patch("a2a.client.create_client", new=create_client):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        await A2ACallTool(
                            client_params={"httpx_client": http_client},
                            call_params={"request_id": "initial-request"},
                        )(
                            "https://peer.example/a2a",
                            "remote.skill",
                            {},
                            context=context,
                        )
                self.assertEqual(broker.requests, [])
                self.assertEqual(len(client.requests), 1)

    async def test_text_only_and_wrong_context_are_never_guessed(
        self,
    ) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        text_only = _input_required()
        text_only.status_update.status.message.ClearField("metadata")
        del text_only.status_update.status.message.extensions[:]
        wrong_context = _input_required()
        wrong_context.status_update.status.message.context_id = "other-context"
        for response, expected in (
            (text_only, "not negotiated"),
            (wrong_context, "correlation mismatch"),
        ):
            with self.subTest(expected=expected):
                context, broker = _context(resolution)
                http_client = _HttpClient()
                client = _SdkClient([[response]], http_client)

                async def create_client(
                    _card: object,
                    *,
                    client_config: object,
                ) -> _SdkClient:
                    return client

                with patch("a2a.client.create_client", new=create_client):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        await A2ACallTool(
                            client_params={"httpx_client": http_client},
                            call_params={"request_id": "initial-request"},
                        )(
                            "https://peer.example/a2a",
                            "remote.skill",
                            {},
                            context=context,
                        )
                self.assertEqual(broker.requests, [])
                self.assertEqual(len(client.requests), 1)

    async def test_input_carrier_accepts_text_with_untrusted_extra_parts(
        self,
    ) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        carrier = _input_required()
        carrier.status_update.status.message.parts.extend(
            [
                a2a_pb2.Part(data=_value({"answer": "untrusted"})),
                a2a_pb2.Part(text="Second readable line."),
            ]
        )
        context, broker = _context(resolution)
        http_client = _HttpClient()
        client = _SdkClient(
            [
                [carrier],
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _status(a2a_pb2.TaskState.TASK_STATE_COMPLETED),
                ],
            ],
            http_client,
        )

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            await A2ACallTool(
                client_params={"httpx_client": http_client},
                call_params={"request_id": "initial-request"},
            )(
                "https://peer.example/a2a",
                "remote.skill",
                {},
                context=context,
            )

        self.assertEqual(len(broker.requests), 1)
        self.assertEqual(
            broker.requests[0].reason,
            "Additional input is required. 1. Continue? Second readable line.",
        )

    async def test_input_carrier_rejects_data_only_parts(self) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        carrier = _input_required()
        del carrier.status_update.status.message.parts[:]
        carrier.status_update.status.message.parts.append(
            a2a_pb2.Part(data=_value({"answer": "untrusted"}))
        )
        context, broker = _context(resolution)
        http_client = _HttpClient()
        client = _SdkClient([[carrier]], http_client)

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            with self.assertRaisesRegex(RuntimeError, "readable fallback"):
                await A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {},
                    context=context,
                )

        self.assertEqual(broker.requests, [])

    async def test_incapable_context_rejects_input_required(self) -> None:
        http_client = _HttpClient()
        client = _SdkClient([[_input_required()]], http_client)
        cards: list[Any] = []

        async def create_client(
            card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            cards.append(card)
            return client

        with patch("a2a.client.create_client", new=create_client):
            with self.assertRaisesRegex(RuntimeError, "unnegotiated"):
                await A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {},
                    context=ToolCallContext(),
                )

        self.assertEqual(list(cards[0].capabilities.extensions), [])
        self.assertIsNone(client.contexts[0].service_parameters)

    async def test_non_answer_timeout_sends_no_cancel_follow_up(self) -> None:
        context, broker = _context(
            TimedOutResolution(
                request_id=InputRequestId("local-request"),
                provenance=AnswerProvenance.POLICY,
                resolved_at=datetime.now(UTC),
            )
        )
        http_client = _HttpClient()
        client = _SdkClient([[_input_required()]], http_client)

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {},
                    context=context,
                )

        self.assertEqual(len(broker.requests), 1)
        self.assertEqual(len(client.requests), 1)

    async def test_continuation_response_must_echo_activation(self) -> None:
        context, broker = _context(
            DeclinedResolution(
                request_id=InputRequestId("local-request"),
                provenance=AnswerProvenance.HUMAN,
                resolved_at=datetime.now(UTC),
            )
        )
        http_client = _HttpClient(echo_extension=(True, False))
        client = _SdkClient(
            [
                [_input_required()],
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _status(a2a_pb2.TaskState.TASK_STATE_COMPLETED),
                ],
            ],
            http_client,
        )

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            with self.assertRaisesRegex(RuntimeError, "not activated"):
                await A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {},
                    context=context,
                )

        self.assertEqual(len(broker.requests), 1)
        self.assertEqual(len(client.requests), 2)

    async def test_continuation_rejects_reentry_and_correlation_drift(
        self,
    ) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        wrong_task = _status(a2a_pb2.TaskState.TASK_STATE_WORKING)
        wrong_task.status_update.task_id = "other-task"
        for final_batch, expected in (
            ([_input_required()], "before returning to working"),
            ([wrong_task], "task correlation mismatch"),
            (
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _input_required(message_id="remote-input-message-two"),
                ],
                "reused input identity",
            ),
            (
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _input_required(
                        request_id="remote-request-two",
                    ),
                ],
                "reused input identity",
            ),
            (
                [_status(a2a_pb2.TaskState.TASK_STATE_COMPLETED)],
                "before returning to working",
            ),
        ):
            with self.subTest(expected=expected):
                context, broker = _context(resolution)
                http_client = _HttpClient()
                client = _SdkClient(
                    [[_input_required()], final_batch],
                    http_client,
                )

                async def create_client(
                    _card: object,
                    *,
                    client_config: object,
                ) -> _SdkClient:
                    return client

                with patch("a2a.client.create_client", new=create_client):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        await A2ACallTool(
                            client_params={"httpx_client": http_client},
                            call_params={"request_id": "initial-request"},
                        )(
                            "https://peer.example/a2a",
                            "remote.skill",
                            {},
                            context=context,
                        )
                self.assertEqual(len(broker.requests), 1)
                self.assertEqual(len(client.requests), 2)

    async def test_parent_cancellation_cancels_pending_broker_wait(
        self,
    ) -> None:
        resolution = DeclinedResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
        )
        blocking = _BlockingBroker(_result(resolution))
        context, broker = _context(resolution, broker=blocking)
        http_client = _HttpClient()
        client = _SdkClient([[_input_required()]], http_client)

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _SdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            pending = create_task(
                A2ACallTool(
                    client_params={"httpx_client": http_client},
                    call_params={"request_id": "initial-request"},
                )(
                    "https://peer.example/a2a",
                    "remote.skill",
                    {},
                    context=context,
                )
            )
            await blocking.started.wait()
            pending.cancel()
            with self.assertRaises(CancelledError):
                await pending

        self.assertIs(broker, blocking)
        self.assertTrue(blocking.cancelled)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len(client.cancel_requests), 1)
        self.assertEqual(client.cancel_requests[0].id, "remote-task")
        cancel_metadata = import_module(
            "google.protobuf.json_format"
        ).MessageToDict(client.cancel_requests[0].metadata)
        self.assertEqual(
            cancel_metadata,
            {"contextId": "remote-context"},
        )
        self.assertEqual(http_client.event_hooks["response"], [])

    async def test_resumed_parent_cancellation_cancels_remote_task(
        self,
    ) -> None:
        http_client = _HttpClient()
        client = _BlockingSdkClient(http_client)

        async def create_client(
            _card: object,
            *,
            client_config: object,
        ) -> _BlockingSdkClient:
            return client

        with patch("a2a.client.create_client", new=create_client):
            pending = create_task(
                A2ACallTool(
                    client_params={"httpx_client": http_client},
                ).resume_input(
                    _checkpoint(_remote_continuation()),
                    DeclinedResolution(
                        request_id=InputRequestId("remote-request"),
                        provenance=AnswerProvenance.HUMAN,
                        resolved_at=datetime.now(UTC),
                    ),
                    context=ToolCallContext(),
                )
            )
            await client.started.wait()
            pending.cancel()
            with self.assertRaises(CancelledError):
                await pending
            client.cancel_state = a2a_pb2.TaskState.TASK_STATE_COMPLETED
            await A2ACallTool(
                client_params={"httpx_client": http_client},
            ).cancel_input(
                _checkpoint(_remote_continuation()),
                operation_id="dispatch-retry-stable",
                context=ToolCallContext(),
            )

        self.assertTrue(client.cancelled)
        self.assertEqual(len(client.cancel_requests), 2)
        self.assertEqual(client.cancel_requests[0].id, "remote-task")
        retry_metadata = import_module(
            "google.protobuf.json_format"
        ).MessageToDict(client.cancel_requests[1].metadata)
        self.assertEqual(
            retry_metadata,
            {
                "contextId": "remote-context",
                "avalanOperationId": "dispatch-retry-stable",
            },
        )
        self.assertEqual(http_client.event_hooks["response"], [])

    async def test_continuation_guards_and_extension_fail_closed_paths(
        self,
    ) -> None:
        resolution = _resolution(DeclinedResolution, "remote-request")
        tool = A2ACallTool()
        with self.assertRaisesRegex(TypeError, "checkpoint"):
            await tool.resume_input(
                cast(Any, object()), resolution, context=ToolCallContext()
            )
        with self.assertRaisesRegex(TypeError, "checkpoint"):
            await tool.cancel_input(
                cast(Any, object()),
                operation_id="operation",
                context=ToolCallContext(),
            )
        with self.assertRaisesRegex(TypeError, "operation_id"):
            await tool.cancel_input(
                _checkpoint(_remote_continuation()),
                operation_id="",
                context=ToolCallContext(),
            )
        with self.assertRaisesRegex(TypeError, "must be paired"):
            await a2a_module._call_a2a_agent(
                uri="https://peer.example/a2a",
                name="remote.skill",
                arguments={},
                continuation=_remote_continuation(),
                context=ToolCallContext(),
                client_params={},
                call_params={},
            )

        negotiation = a2a_module._A2AExtensionNegotiation()
        for client, expected in (
            (object(), "cannot verify"),
            (
                SimpleNamespace(event_hooks={"response": object()}),
                "hooks are unavailable",
            ),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    negotiation.attach(client)
        await negotiation._observe(SimpleNamespace(headers=None, request=None))
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            a2a_module._client_call_context(
                client_module=import_module("a2a.client"),
                call_params={"service_parameters": []},
                input_extension=True,
            )

        with self.assertRaisesRegex(RuntimeError, "no task correlation"):
            a2a_module._input_resolution_request(
                a2a_pb2=a2a_pb2,
                resolution=resolution,
                task_id=None,
                context_id=None,
                prior_message_id=None,
            )
        cancelled = _resolution(CancelledResolution, "remote-request")
        request = a2a_module._input_resolution_request(
            a2a_pb2=a2a_pb2,
            resolution=cancelled,
            task_id="remote-task",
            context_id="remote-context",
            prior_message_id="remote-message",
        )
        self.assertEqual(request.message.parts[0].text, "Input cancelled.")
        with (
            patch.object(a2a_module, "uuid4", return_value="remote-request"),
            self.assertRaisesRegex(RuntimeError, "identity is not fresh"),
        ):
            a2a_module._input_resolution_request(
                a2a_pb2=a2a_pb2,
                resolution=cancelled,
                task_id="remote-task",
                context_id="remote-context",
                prior_message_id=None,
            )

    async def test_cancellation_and_owned_client_failures_are_explicit(
        self,
    ) -> None:
        resolution = _resolution(DeclinedResolution, "remote-request")
        formatter = SimpleNamespace(MessageToDict=lambda value: value)
        cases: tuple[tuple[object, str], ...] = (
            ([], "invalid state"),
            (
                {"id": "other", "contextId": "remote-context"},
                "correlation mismatch",
            ),
            (
                {
                    "id": "remote-task",
                    "contextId": "remote-context",
                    "status": {"state": "TASK_STATE_WORKING"},
                },
                "terminal state",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "cannot cancel"):
            await a2a_module._cancel_a2a_task(
                client=SimpleNamespace(),
                task_id="remote-task",
                context_id="remote-context",
                call_context=object(),
                a2a_pb2=a2a_pb2,
                json_format=formatter,
            )
        for payload, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    await a2a_module._cancel_a2a_task(
                        client=SimpleNamespace(
                            cancel_task=AsyncMock(return_value=payload)
                        ),
                        task_id="remote-task",
                        context_id="remote-context",
                        call_context=object(),
                        a2a_pb2=a2a_pb2,
                        json_format=formatter,
                    )

        http_client = _HttpClient(echo_extension=False)
        client = _SdkClient(
            [
                [
                    _status(a2a_pb2.TaskState.TASK_STATE_WORKING),
                    _status(a2a_pb2.TaskState.TASK_STATE_COMPLETED),
                ]
            ],
            http_client,
        )
        with patch("a2a.client.create_client", AsyncMock(return_value=client)):
            with self.assertRaisesRegex(RuntimeError, "not activated"):
                await A2ACallTool(
                    client_params={"httpx_client": http_client}
                ).resume_input(
                    _checkpoint(_remote_continuation()),
                    resolution,
                    context=ToolCallContext(),
                )

        blocking = _BlockingSdkClient(_HttpClient())
        cast(Any, blocking).cancel_task = AsyncMock(
            side_effect=RuntimeError("cleanup failed")
        )
        with patch(
            "a2a.client.create_client", AsyncMock(return_value=blocking)
        ):
            pending = create_task(
                A2ACallTool(
                    client_params={"httpx_client": blocking.http_client}
                ).resume_input(
                    _checkpoint(_remote_continuation()),
                    resolution,
                    context=ToolCallContext(),
                )
            )
            await blocking.started.wait()
            pending.cancel()
            with self.assertRaises(CancelledError) as raised:
                await pending
        self.assertIn("cleanup failed", " ".join(raised.exception.__notes__))

        class OwnedClient(_SdkClient):
            exited = False

            async def __aenter__(self) -> "OwnedClient":
                return self

            async def __aexit__(self, *_args: object) -> None:
                self.exited = True

        owned = OwnedClient([], _HttpClient())
        with (
            patch("httpx.AsyncClient", return_value=owned.http_client),
            patch("a2a.client.create_client", AsyncMock(return_value=owned)),
        ):
            await A2ACallTool().cancel_input(
                _checkpoint(_remote_continuation()),
                operation_id="operation",
                context=ToolCallContext(),
            )
        self.assertTrue(owned.exited)

    async def test_input_router_reports_every_terminal_failure(self) -> None:
        request = _remote_continuation().request

        def router(context: ToolCallContext) -> Any:
            return a2a_module._A2AInputRouter(
                uri="https://peer.example/a2a",
                tool_name="remote.skill",
                arguments={},
                context=context,
                ttl_seconds=300,
            )

        async def resolve(
            value: Any, task_id: str | None = "remote-task"
        ) -> object:
            return await value.resolve(
                request,
                reason="Input needed.",
                remote_task_id=task_id,
                remote_context_id="remote-context",
                prior_message_id="remote-message",
                prior_content=(),
                input_cycle_count=1,
            )

        with self.assertRaisesRegex(RuntimeError, "origin is unavailable"):
            await resolve(router(ToolCallContext()))

        origin = _origin()
        durable = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=cast(Any, _stager),
        )
        durable_context = ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=cast(
                AgentExecution,
                SimpleNamespace(origin=origin, interaction_runtime=durable),
            ),
            execution_origin=origin,
            durable_a2a_input=True,
        )
        with self.assertRaisesRegex(RuntimeError, "durable.*unavailable"):
            await resolve(router(durable_context), None)

        declined = _resolution(DeclinedResolution)
        context, _ = _context(declined)
        detached = router(context)
        detached._broker = None
        with (
            patch.object(a2a_module._A2AInputRouter, "input_capable", True),
            self.assertRaisesRegex(RuntimeError, "attached.*unavailable"),
        ):
            await resolve(detached)

        for error, expected in (
            (
                InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    "request",
                    "safe broker failure",
                ),
                "safe broker failure",
            ),
            (ValueError("unsafe detail"), "could not be completed"),
        ):
            with self.subTest(expected=expected):
                broker = cast(
                    _Broker,
                    SimpleNamespace(
                        request=AsyncMock(side_effect=error),
                        cancel_scope=AsyncMock(),
                    ),
                )
                failed_context, _ = _context(declined, broker=broker)
                with self.assertRaisesRegex(RuntimeError, expected):
                    await resolve(router(failed_context))

        invalid_answer = AnsweredResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=datetime.now(UTC),
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("wrong"),
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
            ),
        )
        invalid_context, _ = _context(invalid_answer)
        with self.assertRaisesRegex(RuntimeError, "pending questions"):
            await resolve(router(invalid_context))

        outcomes = [
            (
                cast(InteractionRequestResult, SimpleNamespace(delivery=None)),
                "unavailable",
            ),
            (_result(None), "terminal state"),
            (_result(object()), "terminal state"),
        ]
        outcomes.extend(
            (_result(_resolution(kind)), expected)
            for kind, expected in (
                (ExpiredResolution, "expired"),
                (SupersededResolution, "superseded"),
                (UnavailableResolution, "unavailable"),
            )
        )
        for result, expected in outcomes:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    a2a_module._remote_input_resolution(result, request)

    async def test_stream_state_rejects_invalid_continuation_input(
        self,
    ) -> None:
        with self.assertRaisesRegex(TypeError, "remote A2A continuation"):
            a2a_module._A2AStreamState.from_continuation(
                cast(Any, object()),
                input_extension_activated=lambda: True,
            )
        with self.assertRaisesRegex(RuntimeError, "not available"):
            a2a_module._A2AStreamState().begin_continuation()

        message = _input_message_payload()

        def state() -> Any:
            value = a2a_module._A2AStreamState(input_extension_requested=True)
            value.task_id = "remote-task"
            value.context_id = "remote-context"
            return value

        task_state = state()
        task_state._record_task(
            {
                "id": "remote-task",
                "contextId": "remote-context",
                "status": {
                    "state": "TASK_STATE_INPUT_REQUIRED",
                    "message": {**message, "messageId": message["id"]},
                },
            }
        )
        self.assertIsNotNone(task_state.input_request)

        continued = a2a_module._A2AStreamState.from_continuation(
            _remote_continuation(),
            input_extension_activated=lambda: True,
        )
        continued.begin_continuation()
        with self.assertRaisesRegex(RuntimeError, "terminal task status"):
            await continued._record_message({}, ToolCallContext())
        with self.assertRaisesRegex(RuntimeError, "correlation is missing"):
            continued._record_status({"state": "TASK_STATE_WORKING"})

        with self.assertRaisesRegex(RuntimeError, "correlated message"):
            a2a_module._A2AStreamState(
                input_extension_requested=True
            )._record_input_required(message)
        looped = state()
        looped._input_cycle_count = MAXIMUM_EQUIVALENT_INPUT_REQUESTS
        with self.assertRaisesRegex(RuntimeError, "loop limit"):
            looped._record_input_required(message)

        bad_parts = _input_message_payload()
        bad_parts["parts"] = ()
        bad_metadata = _input_message_payload()
        bad_metadata["metadata"] = {A2A_INPUT_EXTENSION_URI: {}}
        leaked = _input_message_payload(message_id="remote-request")
        for payload, expected in (
            (bad_parts, "readable fallback"),
            (bad_metadata, "A2A schema"),
            (leaked, "identity leaked"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    state()._record_input_required(payload)

        pending = state()
        pending._record_input_required(message)
        changed = _input_message_payload()
        changed["parts"] = [{"text": "Changed fallback."}]
        with self.assertRaisesRegex(RuntimeError, "changed its pending"):
            pending._record_input_required(changed)

        self.assertEqual(
            a2a_module._mutable_mapping({"items": ({"value": 1},)}),
            {"items": [{"value": 1}]},
        )
