"""Exercise authenticated structured input through the public server."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    Task,
    create_task,
    ensure_future,
    gather,
    get_running_loop,
    run,
    sleep,
    wait_for,
)
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from json import loads
from logging import getLogger
from typing import Any, cast
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient, Response
from pytest import raises

from avalan.agent.execution import AttachedInteractionRuntime
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import MessageRole
from avalan.interaction import (
    ActiveControlLeaseNonce,
    AgentId,
    AsyncInteractionBroker,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputHandlerContext,
    InputRequest,
    InputRequestId,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBrokerRequest,
    InteractionClock,
    InteractionDisclosure,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionTime,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    RunId,
    SessionId,
    StreamSessionId,
    TaskId,
    TaskInputCapabilityState,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TenantId,
    TextQuestion,
    TextValidationConstraints,
    TurnId,
    UserId,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionBrokerResult,
)
from avalan.interaction.error import (
    InputContractError,
    InputErrorCode,
    InputValidationError,
)
from avalan.interaction.policy import InteractionRequestAuthorizationTarget
from avalan.interaction.state import InputTransitionError
from avalan.interaction.store import (
    CancelInteractionApplied,
    CancelInteractionRejected,
    InteractionCorrelation,
    InteractionRecord,
    InteractionTerminalMetadata,
    ResolveInteractionApplied,
    ResolveInteractionRejected,
    ScopedInteractionLookup,
)
from avalan.interaction.stores import MemoryInteractionStoreFactory
from avalan.model.stream import (
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    stream_channel_for_kind,
)
from avalan.server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    OpenAIRequestExtensions,
    ResponsesRequest,
    ServerOutputRedactionSettings,
    TaskInputExtension,
)
from avalan.server.interaction import (
    TASK_INPUT_EXTENSION,
    ServerDetachedSegment,
    ServerInteractionConfiguration,
    ServerInteractionHandling,
    ServerInteractionLifecycleEvent,
    ServerInteractionRun,
    ServerInteractionService,
    ServerInteractionSurface,
    _http_error_for_code,
    _mapping,
    _request_json,
    _required_string,
    _resolution_command,
    _resume_segment,
    _resume_segment_json,
    _resume_terminal_message,
    _resume_text_message,
    _ServerAttachedInputHandler,
    _ServerHTTPError,
    _store_result,
    _validate_after_store_revision,
    _validate_server_authorization,
    close_server_interactions,
    configure_server_interactions,
    extension_sse_message,
    prepare_openai_interaction_run,
    task_input_extension_from_request,
)
from avalan.server.interaction import (
    router as interaction_router,
)
from avalan.server.routers.chat import (
    _interaction_chat_response,
    create_chat_completion,
)
from avalan.server.routers.chat import (
    router as chat_router,
)
from avalan.server.routers.responses import (
    _DetachedResponsesProjection,
    _ResponsesSSEProjector,
    create_response,
)
from avalan.server.routers.responses import router as responses_router
from avalan.types import LooseJsonValue

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
_OWNER_SCOPE = PrincipalScope(
    user_id=UserId("server-user"),
    tenant_id=TenantId("server-tenant"),
    session_id=SessionId("server-session"),
)
_OTHER_SCOPE = PrincipalScope(
    user_id=UserId("other-user"),
    tenant_id=TenantId("other-tenant"),
    session_id=SessionId("other-session"),
)
_ATTACHED_EXTENSIONS = OpenAIRequestExtensions(
    task_input=TaskInputExtension(version="1", handling="attached")
)
_OWNER = InteractionActor(principal=_OWNER_SCOPE)
_OTHER = InteractionActor(principal=_OTHER_SCOPE)
_EXTENSION_HEADERS = {
    "Authorization": "Bearer owner",
    "Avalan-Extensions": TASK_INPUT_EXTENSION,
}


class _Clock(InteractionClock):
    """Hold deterministic time without firing broker deadlines."""

    def __init__(self) -> None:
        self._waiters: list[Future[None]] = []

    async def read(self) -> InteractionTime:
        """Return one coherent trusted time observation."""
        return InteractionTime.from_clock(
            wall_time=_NOW,
            monotonic_seconds=1.0,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait until broker shutdown cancels the deadline pump."""
        assert monotonic_deadline >= 1.0
        waiter = get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter
        except CancelledError:
            raise
        finally:
            self._waiters.remove(waiter)


class _Ids(InteractionIdFactory):
    """Mint deterministic opaque broker identifiers."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next(self, label: str) -> str:
        self._sequence += 1
        return f"server-{label}-{self._sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return one request identifier."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return one continuation identifier."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return one idempotency key."""
        return ResolutionIdempotencyKey(self._next("idempotency"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return one active-controller lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow the normalized confirmation under the active policy."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self._policy = policy

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return an exact classifier decision for one candidate."""
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self._policy.task_input_classifier_id,
            classification_id="server-classification",
            policy_revision=self._policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Authorize exact operations with full semantic disclosure."""

    def __init__(
        self,
        *,
        allowed: bool = True,
        disclosure: InteractionDisclosure = InteractionDisclosure.FULL,
    ) -> None:
        self.allowed = allowed
        self.disclosure = disclosure

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return an allow decision bound to the supplied target."""
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=self.allowed,
            disclosure=(
                self.disclosure if self.allowed else InteractionDisclosure.NONE
            ),
        )


class _PrincipalResolver:
    """Resolve two test principals from bearer credentials."""

    async def __call__(self, request: Request) -> InteractionActor | None:
        """Return the actor named by the exact bearer token."""
        authorization = request.headers.get("authorization")
        if authorization == "Bearer owner":
            return _OWNER
        if authorization == "Bearer other":
            return _OTHER
        return None


class _FakeProviderResponse:
    """Wrap one async projection stream with Responses token counters."""

    def __init__(
        self,
        iterator: AsyncIterator[StreamConsumerProjection],
    ) -> None:
        self._iterator = iterator
        self.input_token_count = 0
        self.output_token_count = 0

    def __aiter__(self) -> "_FakeProviderResponse":
        return self

    async def __anext__(self) -> StreamConsumerProjection:
        return await anext(self._iterator)

    async def aclose(self) -> None:
        """Close the wrapped source when transport cleanup owns it."""
        close = getattr(self._iterator, "aclose", None)
        if close is not None:
            await close()


class _FaultBroker:
    """Return or raise scripted values at the server trust boundary."""

    def __init__(self, record: InteractionRecord) -> None:
        self.inspect_value: object = record
        self.wait_value: object = record
        self.resolve_value: object = object()
        self.cancel_value: object = object()
        self.inspect_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.resolve_error: Exception | None = None
        self.cancel_error: Exception | None = None

    async def inspect(self, query: object) -> object:
        """Return the scripted inspection projection."""
        _ = query
        if self.inspect_error is not None:
            raise self.inspect_error
        return self.inspect_value

    async def wait(self, command: object) -> object:
        """Return the scripted wait projection."""
        _ = command
        if self.wait_error is not None:
            raise self.wait_error
        return self.wait_value

    async def resolve(self, command: object) -> object:
        """Return the scripted resolution result."""
        _ = command
        if self.resolve_error is not None:
            raise self.resolve_error
        return self.resolve_value

    async def cancel(self, command: object) -> object:
        """Return the scripted cancellation result."""
        _ = command
        if self.cancel_error is not None:
            raise self.cancel_error
        return self.cancel_value

    async def request(self, request: object) -> object:
        """Reject unexpected request creation through this fault broker."""
        _ = request
        raise RuntimeError("fault broker cannot create requests")

    async def cancel_scope(self, command: object) -> object:
        """Reject unexpected scope cancellation through this fault broker."""
        _ = command
        raise RuntimeError("fault broker cannot cancel scopes")


class _RaisingPrincipalResolver:
    """Fail while resolving an HTTP principal."""

    async def __call__(self, request: Request) -> InteractionActor | None:
        """Raise a host authentication failure."""
        _ = request
        raise RuntimeError("authentication backend failed")


class _RaisingAuthorizer:
    """Fail while authorizing an exact interaction operation."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Raise a host authorization failure."""
        _ = actor, operation, target
        raise RuntimeError("authorization backend failed")


class _ImmediateEvent:
    """Return immediately from a normally indefinite handler wait."""

    async def wait(self) -> None:
        """Return immediately."""


async def _empty_projection_stream() -> (
    AsyncIterator[StreamConsumerProjection]
):
    if False:
        yield cast(StreamConsumerProjection, None)


async def _projection_stream(
    *projections: StreamConsumerProjection,
) -> AsyncIterator[StreamConsumerProjection]:
    for projection in projections:
        yield projection


async def _blocking_projection_stream() -> (
    AsyncIterator[StreamConsumerProjection]
):
    await Event().wait()
    if False:
        yield cast(StreamConsumerProjection, None)


async def _failing_projection_stream(
    started: Event | None = None,
) -> AsyncIterator[StreamConsumerProjection]:
    if started is not None:
        started.set()
    try:
        await Event().wait()
    except CancelledError:
        raise RuntimeError("retained read cancellation failed") from None
    if False:
        yield cast(StreamConsumerProjection, None)


def _server_segment(
    iterator: AsyncIterator[StreamConsumerProjection] | None = None,
    *,
    protocol: ServerInteractionSurface = ServerInteractionSurface.CHAT,
) -> ServerDetachedSegment:
    return ServerDetachedSegment(
        iterator=iterator or _empty_projection_stream(),
        response=object(),
        orchestrator=_FakeProviderOrchestrator(),
        protocol=protocol,
        response_id="server-response",
        created=1,
        model_id="server-model",
        output_redaction_settings=ServerOutputRedactionSettings(),
    )


def _unsafe_broker_result(store_result: object) -> InteractionBrokerResult:
    result = object.__new__(InteractionBrokerResult)
    object.__setattr__(result, "store_result", store_result)
    return result


def _unsafe_rejected(
    result_type: (
        type[ResolveInteractionRejected] | type[CancelInteractionRejected]
    ),
    code: InputErrorCode,
) -> ResolveInteractionRejected | CancelInteractionRejected:
    result = object.__new__(result_type)
    object.__setattr__(
        result,
        "error",
        InputTransitionError(code=code, path="request", message="rejected"),
    )
    return result


def _unsafe_applied(
    result_type: (
        type[ResolveInteractionApplied] | type[CancelInteractionApplied]
    ),
    record: InteractionRecord,
) -> ResolveInteractionApplied | CancelInteractionApplied:
    result = object.__new__(result_type)
    object.__setattr__(result, "record", record)
    return result


class _FakeProviderOrchestrator(Orchestrator):
    """Expose one broker-backed confirmation as a fake provider stream."""

    def __init__(self) -> None:
        self.provider_calls = 0
        self.sync_calls = 0
        self.request_ready = Event()
        self.continuation_started = Event()
        self.continuation_gate: Event | None = None
        self.preface_response_text: str | None = None
        self.request: InputRequest | None = None
        self._active_tasks: list[Task[object]] = []

    def __str__(self) -> str:
        return "server-fake-provider"

    async def __call__(
        self,
        messages: object,
        settings: object | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamConsumerProjection]:
        _ = messages, settings
        runtime = kwargs.get("interaction_runtime")
        if runtime is None:
            return _FakeProviderResponse(self._plain_stream())
        assert isinstance(runtime, AttachedInteractionRuntime)
        return _FakeProviderResponse(self._stream(runtime))

    async def sync_messages(self, response: object) -> None:
        """Count one completed outward response synchronization."""
        _ = response
        self.sync_calls += 1

    async def _plain_stream(self) -> AsyncIterator[StreamConsumerProjection]:
        origin = _origin(_OWNER_SCOPE)
        self.provider_calls += 1
        yield _projection(origin, 0, StreamItemKind.STREAM_STARTED)
        yield _projection(
            origin,
            1,
            StreamItemKind.ANSWER_DELTA,
            text_delta="Plain.",
        )
        yield _projection(origin, 2, StreamItemKind.ANSWER_DONE)
        yield _projection(
            origin,
            3,
            StreamItemKind.STREAM_COMPLETED,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield _projection(origin, 4, StreamItemKind.STREAM_CLOSED)

    async def _stream(
        self,
        runtime: AttachedInteractionRuntime,
    ) -> AsyncIterator[StreamConsumerProjection]:
        origin = _origin(runtime.actor.principal)
        self.provider_calls += 1
        yield _projection(origin, 0, StreamItemKind.STREAM_STARTED)
        delivered = Event()

        async def handler(
            context: InputHandlerContext,
        ) -> Any:
            self.request = context.request
            self.request_ready.set()
            delivered.set()
            return await runtime.handler(context)

        request_task = create_task(
            runtime.broker.request(
                InteractionBrokerRequest(
                    actor=runtime.actor,
                    origin=origin,
                    mode=RequirementMode.REQUIRED,
                    reason="Confirm the server continuation.",
                    questions=(
                        ConfirmationQuestion(
                            question_id=QuestionId("continue"),
                            prompt="Continue?",
                            required=True,
                        ),
                        TextQuestion(
                            question_id=QuestionId("note"),
                            prompt="Optional note",
                            required=False,
                            constraints=TextValidationConstraints(
                                minimum_length=1,
                                maximum_length=12,
                            ),
                        ),
                    ),
                    handler=handler,
                    continuation_ttl_seconds=600,
                )
            ),
            name="server-fake-provider-input",
        )
        self._active_tasks.append(request_task)
        await delivered.wait()
        assert self.request is not None
        correlation = _correlation(self.request)
        sequence = 1
        if self.preface_response_text is not None:
            yield _projection(
                origin,
                sequence,
                StreamItemKind.ANSWER_DELTA,
                text_delta=self.preface_response_text,
            )
            sequence += 1
        yield _projection(
            origin,
            sequence,
            StreamItemKind.INTERACTION_CREATED,
            correlation,
        )
        sequence += 1
        yield _projection(
            origin,
            sequence,
            StreamItemKind.INTERACTION_PENDING,
            correlation,
        )
        sequence += 1
        result = await request_task
        assert result.delivery is not None
        terminal_request = result.delivery.record.request
        assert terminal_request.state is RequestState.ANSWERED
        self.continuation_started.set()
        if self.continuation_gate is not None:
            await self.continuation_gate.wait()
        self.provider_calls += 1
        yield _projection(
            origin,
            sequence,
            StreamItemKind.INTERACTION_ANSWERED,
            correlation,
        )
        sequence += 1
        yield _projection(
            origin,
            sequence,
            StreamItemKind.ANSWER_DELTA,
            text_delta="Confirmed.",
        )
        sequence += 1
        yield _projection(origin, sequence, StreamItemKind.ANSWER_DONE)
        sequence += 1
        yield _projection(
            origin,
            sequence,
            StreamItemKind.STREAM_COMPLETED,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield _projection(origin, sequence + 1, StreamItemKind.STREAM_CLOSED)


def _origin(principal: PrincipalScope) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("server-run"),
        turn_id=TurnId("server-turn"),
        task_id=TaskId("server-task"),
        agent_id=AgentId("server-agent"),
        branch_id=BranchId("server-branch"),
        model_call_id=ModelCallId("server-model-call"),
        stream_session_id=StreamSessionId("server-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://server-test",
            agent_definition_revision="revision-1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-1",
            tool_revision="tools-1",
            capability_revision="capabilities-1",
        ),
        principal=principal,
    )


def _correlation(request: InputRequest) -> StreamItemCorrelation:
    origin = request.origin
    return StreamItemCorrelation(
        request_id=request.request_id,
        continuation_id=request.continuation_id,
        task_id=str(origin.task_id) if origin.task_id is not None else None,
        agent_id=origin.agent_id,
        branch_id=origin.branch_id,
        parent_branch_id=origin.parent_branch_id,
    )


def _projection(
    origin: ExecutionOrigin,
    sequence: int,
    kind: StreamItemKind,
    correlation: StreamItemCorrelation | None = None,
    *,
    text_delta: str | None = None,
    usage: LooseJsonValue | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> StreamConsumerProjection:
    return StreamConsumerProjection(
        stream_session_id=str(origin.stream_session_id),
        run_id=str(origin.run_id),
        turn_id=str(origin.turn_id),
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        usage=usage,
        terminal_outcome=terminal_outcome,
    )


async def _open_broker(
    policy: InteractionPolicy | None = None,
) -> AsyncInteractionBroker:
    selected_policy = policy or InteractionPolicy()
    clock = _Clock()
    identifiers = _Ids()
    classifier = _Classifier(selected_policy)
    store = await MemoryInteractionStoreFactory(
        policy=selected_policy,
        clock=clock,
        authorizer=_Authorizer(),
        id_factory=identifiers,
        classifier=classifier,
    ).open()
    return AsyncInteractionBroker(
        store=store,
        clock=clock,
        id_factory=identifiers,
        policy=selected_policy,
        classifier=classifier,
    )


def _app(
    broker: AsyncInteractionBroker,
    orchestrator: _FakeProviderOrchestrator,
    *,
    authorizer: _Authorizer | None = None,
    policy: InteractionPolicy | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.logger = getLogger("server-input-test")
    app.state.orchestrator = orchestrator
    configure_server_interactions(
        app,
        ServerInteractionConfiguration(
            broker=broker,
            principal_resolver=_PrincipalResolver(),
            authorizer=authorizer or _Authorizer(),
            policy=policy or InteractionPolicy(),
        ),
    )
    app.include_router(chat_router, prefix="/v1")
    app.include_router(responses_router, prefix="/v1")
    app.include_router(interaction_router)
    return app


def _completion_payload(
    *,
    stream: bool,
    handling: str,
) -> dict[str, object]:
    return {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Start."}],
        "stream": stream,
        "extensions": {"task_input": {"version": "1", "handling": handling}},
    }


def _responses_payload(
    *,
    stream: bool = True,
    handling: str = "detached",
) -> dict[str, object]:
    return {
        "model": "fake-model",
        "input": [{"role": "user", "content": "Start."}],
        "stream": stream,
        "extensions": {"task_input": {"version": "1", "handling": handling}},
    }


def _resolve_payload(
    request: InputRequest,
    revision: int,
    key: str,
) -> dict[str, object]:
    origin = request.origin
    return {
        "continuation_id": str(request.continuation_id),
        "run_id": str(origin.run_id),
        "turn_id": str(origin.turn_id),
        "task_id": str(origin.task_id) if origin.task_id is not None else None,
        "agent_id": str(origin.agent_id),
        "branch_id": str(origin.branch_id),
        "model_call_id": str(origin.model_call_id),
        "expected_state_revision": revision,
        "idempotency_key": key,
        "status": "answered",
        "answers": [
            {
                "question_id": "continue",
                "kind": "confirmation",
                "provenance": "human",
                "value": True,
            }
        ],
    }


def _resolve_payload_from_observation(
    observation: Mapping[str, Any],
    key: str,
) -> dict[str, object]:
    questions = observation["questions"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    return {
        "continuation_id": observation["continuation_id"],
        "run_id": observation["run_id"],
        "turn_id": observation["turn_id"],
        "task_id": observation["task_id"],
        "agent_id": observation["agent_id"],
        "branch_id": observation["branch_id"],
        "model_call_id": observation["model_call_id"],
        "expected_state_revision": observation["state_revision"],
        "idempotency_key": key,
        "status": "answered",
        "answers": [
            {
                "question_id": question["question_id"],
                "kind": question["kind"],
                "provenance": "human",
                "value": True,
            }
        ],
    }


def _cancel_payload(
    request: InputRequest,
    revision: int,
    key: str,
) -> dict[str, object]:
    payload = _resolve_payload(request, revision, key)
    payload.pop("status")
    payload.pop("answers")
    return payload


def _cancel_payload_from_observation(
    observation: Mapping[str, Any],
    key: str,
) -> dict[str, object]:
    payload = _resolve_payload_from_observation(observation, key)
    payload.pop("status")
    payload.pop("answers")
    return payload


def _sse_events(
    response: Response,
) -> list[tuple[str | None, dict[str, Any] | str]]:
    events: list[tuple[str | None, dict[str, Any] | str]] = []
    event_name: str | None = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            raw = line.removeprefix("data: ")
            if raw == "[DONE]":
                data: dict[str, Any] | str = raw
            else:
                parsed = loads(raw)
                assert isinstance(parsed, dict)
                data = parsed
            events.append((event_name, data))
            event_name = None
    return events


async def _detached_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    authorizer = _Authorizer()
    transport = ASGITransport(
        app=_app(broker, orchestrator, authorizer=authorizer)
    )
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            detached = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(stream=True, handling="detached"),
            )
            assert detached.status_code == 200
            assert detached.headers["avalan-extensions"] == (
                TASK_INPUT_EXTENSION
            )
            events = _sse_events(detached)
            extension_events = [
                event for event in events if event[0] is not None
            ]
            assert [event[0] for event in extension_events] == [
                "input_request.created",
                "input_request.presented",
                "response.input_required",
            ]
            assert all(
                event[0] not in {"response.completed", "response.failed"}
                for event in events
            )
            assert not any(
                isinstance(data, Mapping)
                and data.get("choices")
                and "content" in str(data)
                for _name, data in events
            )
            created_event = extension_events[0][1]
            required_event = extension_events[-1][1]
            assert isinstance(created_event, dict)
            assert isinstance(required_event, dict)
            request_id = created_event["request_id"]
            assert request_id == required_event["request_id"]
            assert orchestrator.provider_calls == 1
            assert orchestrator.sync_calls == 0

            unauthenticated = await client.get(
                f"/v1/input/requests/{request_id}"
            )
            wrong_scope = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer other"},
            )
            assert unauthenticated.status_code == 401
            assert wrong_scope.status_code == 404
            assert "Continue?" not in unauthenticated.text
            assert "Continue?" not in wrong_scope.text
            authorizer.allowed = False
            forbidden = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert forbidden.status_code == 403
            assert "Continue?" not in forbidden.text
            authorizer.allowed = True
            authorizer.disclosure = InteractionDisclosure.TERMINAL_METADATA
            metadata_only = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert metadata_only.status_code == 403
            assert "Continue?" not in metadata_only.text
            authorizer.disclosure = InteractionDisclosure.FULL

            inspected = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert inspected.status_code == 200
            observation = inspected.json()
            assert set(observation) == {
                "request_id",
                "continuation_id",
                "state_revision",
                "run_id",
                "turn_id",
                "agent_id",
                "branch_id",
                "task_id",
                "model_call_id",
                "required",
                "reason",
                "created_at",
                "state",
                "questions",
            }
            assert observation["state"] == "pending"
            assert observation["questions"] == [
                {
                    "question_id": "continue",
                    "kind": "confirmation",
                    "prompt": "Continue?",
                    "required": True,
                    "choices": [],
                    "allow_other": False,
                },
                {
                    "question_id": "note",
                    "kind": "text",
                    "prompt": "Optional note",
                    "required": False,
                    "choices": [],
                    "allow_other": False,
                    "constraints": {
                        "minimum_length": 1,
                        "maximum_length": 12,
                    },
                },
            ]
            good_payload = _resolve_payload_from_observation(
                observation,
                "server-http-key",
            )
            for field_name in (
                "continuation_id",
                "run_id",
                "turn_id",
                "task_id",
                "agent_id",
                "branch_id",
                "model_call_id",
            ):
                mismatched = {
                    **good_payload,
                    field_name: (
                        None if field_name == "task_id" else "wrong-scope"
                    ),
                    "idempotency_key": f"wrong-{field_name}",
                }
                rejected = await client.post(
                    f"/v1/input/requests/{request_id}/resolve",
                    headers={"Authorization": "Bearer owner"},
                    json=mismatched,
                )
                assert rejected.status_code == 422
                missing = dict(good_payload)
                missing.pop(field_name)
                missing["idempotency_key"] = f"missing-{field_name}"
                rejected_missing = await client.post(
                    f"/v1/input/requests/{request_id}/resolve",
                    headers={"Authorization": "Bearer owner"},
                    json=missing,
                )
                assert rejected_missing.status_code == 422
            rejected_nonhuman = await client.post(
                f"/v1/input/requests/{request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json={
                    **good_payload,
                    "idempotency_key": "server-nonhuman-key",
                    "answers": [
                        {
                            "question_id": "continue",
                            "kind": "confirmation",
                            "provenance": "policy",
                            "value": True,
                        }
                    ],
                },
            )
            assert rejected_nonhuman.status_code == 422
            assert orchestrator.provider_calls == 1
            stale_payload = {
                **good_payload,
                "expected_state_revision": observation["state_revision"] + 1,
                "idempotency_key": "server-stale-key",
            }
            stale = await client.post(
                f"/v1/input/requests/{request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=stale_payload,
            )
            assert stale.status_code == 409
            assert stale.json()["code"] == "input.stale_revision"
            assert "Continue?" not in stale.text
            assert orchestrator.provider_calls == 1

            first, second = await gather(
                client.post(
                    f"/v1/input/requests/{request_id}/resolve",
                    headers={"Authorization": "Bearer owner"},
                    json=good_payload,
                ),
                client.post(
                    f"/v1/input/requests/{request_id}/resolve",
                    headers={"Authorization": "Bearer owner"},
                    json={
                        **good_payload,
                        "idempotency_key": "server-racing-key",
                    },
                ),
            )
            assert first.status_code == second.status_code == 200
            assert sorted(
                (
                    first.json()["idempotent"],
                    second.json()["idempotent"],
                )
            ) == [False, True]
            accepted_payload = (
                good_payload
                if not first.json()["idempotent"]
                else {**good_payload, "idempotency_key": "server-racing-key"}
            )
            conflicting = await client.post(
                f"/v1/input/requests/{request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json={
                    **good_payload,
                    "idempotency_key": "server-conflicting-key",
                    "answers": [
                        {
                            "question_id": "continue",
                            "kind": "confirmation",
                            "provenance": "human",
                            "value": False,
                        }
                    ],
                },
            )
            assert conflicting.status_code == 409
            replay = await client.post(
                f"/v1/input/requests/{request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=accepted_payload,
            )
            assert replay.status_code == 200
            assert replay.json() == {
                "kind": "resolution_accepted",
                "interaction_state": "answered",
                "idempotent": True,
                "channel": "json",
            }

            orchestrator.continuation_gate = Event()
            resumed_task = create_task(
                client.get(
                    f"/v1/input/requests/{request_id}/poll",
                    params={"transport": "json"},
                    headers={"Authorization": "Bearer owner"},
                ),
                name="server-resume-owner",
            )
            await wait_for(orchestrator.continuation_started.wait(), timeout=2)
            competing_resume = await client.get(
                f"/v1/input/requests/{request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            assert competing_resume.status_code == 409
            orchestrator.continuation_gate.set()
            resumed = await wait_for(resumed_task, timeout=2)
            assert resumed.status_code == 200
            assert (
                resumed.json()["choices"][0]["message"]["content"]
                == "Confirmed."
            )
            replayed_completion = await client.get(
                f"/v1/input/requests/{request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            assert replayed_completion.json() == resumed.json()
            assert orchestrator.provider_calls == 2
            assert orchestrator.sync_calls == 1
    finally:
        await broker.aclose()


async def _attached_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    transport = ASGITransport(app=_app(broker, orchestrator))
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            response_task = create_task(
                client.post(
                    "/v1/chat/completions",
                    headers=_EXTENSION_HEADERS,
                    json=_completion_payload(
                        stream=True,
                        handling="attached",
                    ),
                ),
                name="server-attached-http",
            )
            await wait_for(orchestrator.request_ready.wait(), timeout=2)
            assert not response_task.done()
            assert orchestrator.request is not None
            request = orchestrator.request
            inspect = await client.get(
                f"/v1/input/requests/{request.request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert inspect.status_code == 200
            resolved = await client.post(
                f"/v1/input/requests/{request.request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=_resolve_payload(
                    request,
                    inspect.json()["state_revision"],
                    "server-attached-key",
                ),
            )
            assert resolved.status_code == 200
            response = await wait_for(response_task, timeout=2)
            event_names = [name for name, _data in _sse_events(response)]
            assert event_names[:3] == [
                "input_request.created",
                "input_request.presented",
                "input_request.resolved",
            ]
            assert _sse_events(response)[-1][1] == "[DONE]"
            assert "Confirmed." in response.text
            assert orchestrator.provider_calls == 2
            assert orchestrator.sync_calls == 1
    finally:
        await broker.aclose()


async def _non_streaming_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    transport = ASGITransport(app=_app(broker, orchestrator))
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            required = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(
                    stream=False,
                    handling="detached",
                ),
            )
            assert required.status_code == 202
            assert set(required.json()) == {
                "status",
                "request_id",
                "continuation_id",
                "detached_resumption_available",
            }
            assert required.json()["status"] == "input_required"
            assert orchestrator.request is not None
            request = orchestrator.request
            inspected = await client.get(
                f"/v1/input/requests/{request.request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            resolved = await client.post(
                f"/v1/input/requests/{request.request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=_resolve_payload(
                    request,
                    inspected.json()["state_revision"],
                    "server-non-stream-key",
                ),
            )
            assert resolved.status_code == 200
            completed = await client.get(
                f"/v1/input/requests/{request.request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            assert completed.status_code == 200
            assert (
                completed.json()["choices"][0]["message"]["content"]
                == "Confirmed."
            )
            assert orchestrator.provider_calls == 2
            assert orchestrator.sync_calls == 1
    finally:
        await broker.aclose()


async def _attached_non_streaming_scenario() -> None:
    for path, payload, expected_text in (
        (
            "/v1/chat/completions",
            _completion_payload(stream=False, handling="attached"),
            ("choices", "message", "content"),
        ),
        (
            "/v1/responses",
            _responses_payload(stream=False, handling="attached"),
            ("output", "content", "text"),
        ),
    ):
        broker = await _open_broker()
        orchestrator = _FakeProviderOrchestrator()
        transport = ASGITransport(app=_app(broker, orchestrator))
        try:
            async with AsyncClient(
                transport=transport,
                base_url="http://server.test",
            ) as client:
                response_task = create_task(
                    client.post(
                        path,
                        headers=_EXTENSION_HEADERS,
                        json=payload,
                    ),
                    name="server-attached-non-stream-http",
                )
                await wait_for(orchestrator.request_ready.wait(), timeout=2)
                assert orchestrator.request is not None
                request = orchestrator.request
                inspected = await client.get(
                    f"/v1/input/requests/{request.request_id}",
                    headers={"Authorization": "Bearer owner"},
                )
                resolved = await client.post(
                    f"/v1/input/requests/{request.request_id}/resolve",
                    headers={"Authorization": "Bearer owner"},
                    json=_resolve_payload(
                        request,
                        inspected.json()["state_revision"],
                        f"attached-{path}-key",
                    ),
                )
                assert resolved.status_code == 200
                completed = await wait_for(response_task, timeout=2)
                assert completed.status_code == 200
                body = completed.json()
                if expected_text[0] == "choices":
                    text = body["choices"][0]["message"]["content"]
                else:
                    text = body["output"][0]["content"][0]["text"]
                assert text == "Confirmed."
                assert completed.headers["avalan-extensions"] == (
                    TASK_INPUT_EXTENSION
                )
        finally:
            await broker.aclose()


async def _responses_non_streaming_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    transport = ASGITransport(app=_app(broker, orchestrator))
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            required = await client.post(
                "/v1/responses",
                headers=_EXTENSION_HEADERS,
                json=_responses_payload(stream=False),
            )
            assert required.status_code == 202
            assert orchestrator.request is not None
            request = orchestrator.request
            inspected = await client.get(
                f"/v1/input/requests/{request.request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            resolved = await client.post(
                f"/v1/input/requests/{request.request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=_resolve_payload(
                    request,
                    inspected.json()["state_revision"],
                    "server-responses-json-key",
                ),
            )
            assert resolved.status_code == 200
            completed = await client.get(
                f"/v1/input/requests/{request.request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            assert completed.status_code == 200
            assert completed.json()["type"] == "response"
            assert (
                completed.json()["output"][0]["content"][0]["text"]
                == "Confirmed."
            )
    finally:
        await broker.aclose()


async def _responses_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    app = _app(broker, orchestrator)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            required = await client.post(
                "/v1/responses",
                headers=_EXTENSION_HEADERS,
                json=_responses_payload(),
            )
            assert required.status_code == 200
            initial_events = _sse_events(required)
            assert [name for name, _data in initial_events] == [
                "response.created",
                "input_request.created",
                "input_request.presented",
                "response.input_required",
            ]
            assert orchestrator.request is not None
            request = orchestrator.request
            inspected = await client.get(
                f"/v1/input/requests/{request.request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            observation = inspected.json()
            resolved = await client.post(
                f"/v1/input/requests/{request.request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=_resolve_payload_from_observation(
                    observation,
                    "server-responses-key",
                ),
            )
            assert resolved.status_code == 200
            resumed = await client.get(
                f"/v1/input/requests/{request.request_id}/poll",
                params={"transport": "stream"},
                headers={
                    "Authorization": "Bearer owner",
                    "Accept": "text/event-stream",
                },
            )
            assert resumed.status_code == 200
            resumed_events = _sse_events(resumed)
            assert [name for name, _data in resumed_events] == [
                "execution.resumed",
                "input_request.resolved",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.usage.completed",
                "response.completed",
            ]
            output_event = resumed_events[4][1]
            assert isinstance(output_event, dict)
            assert output_event["delta"] == "Confirmed."
            assert orchestrator.provider_calls == 2
            assert orchestrator.sync_calls == 1
            orchestrator.preface_response_text = "Preface."
            flushed = await client.post(
                "/v1/responses",
                headers=_EXTENSION_HEADERS,
                json=_responses_payload(),
            )
            flushed_names = [name for name, _data in _sse_events(flushed)]
            assert flushed_names.index("response.output_text.delta") < (
                flushed_names.index("input_request.created")
            )
    finally:
        await close_server_interactions(app)
        await broker.aclose()


async def _negotiation_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    app = _app(broker, orchestrator)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            plain = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "Start."}],
                    "stream": True,
                },
            )
            assert plain.status_code == 200
            assert "avalan-extensions" not in plain.headers
            assert "Plain." in plain.text
            assert orchestrator.provider_calls == 1

            header_only = await client.post(
                "/v1/chat/completions",
                headers={"Avalan-Extensions": TASK_INPUT_EXTENSION},
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "Start."}],
                    "stream": True,
                },
            )
            body_only = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer owner"},
                json=_completion_payload(
                    stream=True,
                    handling="detached",
                ),
            )
            unsupported = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json={
                    **_completion_payload(
                        stream=True,
                        handling="detached",
                    ),
                    "extensions": {
                        "task_input": {
                            "version": "2",
                            "handling": "detached",
                        }
                    },
                },
            )
            unauthenticated = await client.post(
                "/v1/chat/completions",
                headers={"Avalan-Extensions": TASK_INPUT_EXTENSION},
                json=_completion_payload(
                    stream=True,
                    handling="detached",
                ),
            )
            assert header_only.status_code == 503
            assert body_only.status_code == 503
            assert unsupported.status_code == 503
            assert unauthenticated.status_code == 401
            assert orchestrator.provider_calls == 1

            explicitly_unavailable = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(
                    stream=True,
                    handling="unavailable",
                ),
            )
            unsupported_responses = await client.post(
                "/v1/responses",
                headers=_EXTENSION_HEADERS,
                json={
                    **_responses_payload(),
                    "extensions": {
                        "task_input": {
                            "version": "2",
                            "handling": "detached",
                        }
                    },
                },
            )
            assert explicitly_unavailable.status_code == 200
            assert "Plain." in explicitly_unavailable.text
            assert unsupported_responses.status_code == 503
            assert orchestrator.provider_calls == 2

            configure_server_interactions(app, None)
            unavailable = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(
                    stream=True,
                    handling="detached",
                ),
            )
            assert unavailable.status_code == 503
            assert orchestrator.provider_calls == 2
    finally:
        await close_server_interactions(app)
        await broker.aclose()


async def _policy_negotiation_scenario() -> None:
    for capability_state, expected_status, provider_calls in (
        (TaskInputCapabilityState.DORMANT, 503, 0),
        (TaskInputCapabilityState.ROLLBACK, 503, 0),
        (TaskInputCapabilityState.ACTIVE, 200, 1),
    ):
        policy = InteractionPolicy(capability_state=capability_state)
        broker = await _open_broker(policy)
        orchestrator = _FakeProviderOrchestrator()
        app = _app(broker, orchestrator, policy=policy)
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(
                transport=transport,
                base_url="http://server.test",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers=_EXTENSION_HEADERS,
                    json=_completion_payload(
                        stream=True,
                        handling="detached",
                    ),
                )
                assert response.status_code == expected_status
                assert orchestrator.provider_calls == provider_calls
                if capability_state is TaskInputCapabilityState.ROLLBACK:
                    assert policy.resolve_existing
        finally:
            await close_server_interactions(app)
            await broker.aclose()


async def _cancel_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    transport = ASGITransport(app=_app(broker, orchestrator))
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            required = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(stream=True, handling="detached"),
            )
            request_id = next(
                data["request_id"]
                for name, data in _sse_events(required)
                if name == "response.input_required" and isinstance(data, dict)
            )
            assert orchestrator.request is not None
            inspected = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            payload = _cancel_payload_from_observation(
                inspected.json(),
                "server-cancel-key",
            )
            cancelled = await client.post(
                f"/v1/input/requests/{request_id}/cancel",
                headers={"Authorization": "Bearer owner"},
                json=payload,
            )
            assert cancelled.status_code == 200
            assert cancelled.json() == {
                "interaction_state": "cancelled",
                "accepted": True,
                "channel": "json",
            }
            replay = await client.post(
                f"/v1/input/requests/{request_id}/cancel",
                headers={"Authorization": "Bearer owner"},
                json=payload,
            )
            conflicting = await client.post(
                f"/v1/input/requests/{request_id}/cancel",
                headers={"Authorization": "Bearer owner"},
                json={**payload, "idempotency_key": "different-cancel-key"},
            )
            assert replay.status_code == 200
            assert conflicting.status_code == 409
            observed = await client.get(
                f"/v1/input/requests/{request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert observed.json()["state"] == "cancelled"
            assert orchestrator.provider_calls == 1
    finally:
        await broker.aclose()


async def _disconnect_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    app = _app(broker, orchestrator)
    http_request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"authorization", b"Bearer owner"),
                (
                    b"avalan-extensions",
                    TASK_INPUT_EXTENSION.encode(),
                ),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("server.test", 80),
            "root_path": "",
            "app": app,
        }
    )
    request = ChatCompletionRequest(
        model="fake-model",
        messages=[ChatMessage(role=MessageRole.USER, content="Start.")],
        stream=True,
        extensions=_ATTACHED_EXTENSIONS,
    )
    try:
        response = await create_chat_completion(
            request,
            getLogger("server-input-disconnect"),
            orchestrator,
            output_redaction_settings=ServerOutputRedactionSettings(),
            http_request=http_request,
        )
        assert isinstance(response, StreamingResponse)
        body_iterator = response.body_iterator.__aiter__()
        assert "input_request.created" in str(await anext(body_iterator))
        assert "input_request.presented" in str(await anext(body_iterator))
        blocked_read = ensure_future(anext(body_iterator))
        blocked_read.set_name("server-disconnected-read")
        await sleep(0)

        original_install_segment = ServerInteractionRun.install_segment

        async def install_then_report_existing(
            run_: ServerInteractionRun,
            segment: ServerDetachedSegment,
        ) -> object:
            await original_install_segment(run_, segment)
            raise RuntimeError("segment already retained")

        with patch.object(
            ServerInteractionRun,
            "install_segment",
            install_then_report_existing,
        ):
            blocked_read.cancel()
            try:
                await blocked_read
            except CancelledError:
                pass
            else:
                raise AssertionError(
                    "transport read cancellation was swallowed"
                )

        assert orchestrator.request is not None
        pending = orchestrator.request
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            inspected = await client.get(
                f"/v1/input/requests/{pending.request_id}",
                headers={"Authorization": "Bearer owner"},
            )
            assert inspected.json()["state"] == "pending"
            resolved = await client.post(
                f"/v1/input/requests/{pending.request_id}/resolve",
                headers={"Authorization": "Bearer owner"},
                json=_resolve_payload(
                    pending,
                    inspected.json()["state_revision"],
                    "server-disconnect-key",
                ),
            )
            assert resolved.status_code == 200
            resumed = await client.get(
                f"/v1/input/requests/{pending.request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            assert resumed.status_code == 200
            assert (
                resumed.json()["choices"][0]["message"]["content"]
                == "Confirmed."
            )
            assert orchestrator.provider_calls == 2
            assert orchestrator.sync_calls == 1
    finally:
        await broker.aclose()


async def _responses_disconnect_scenario() -> None:
    for stream in (True, False):
        broker = await _open_broker()
        orchestrator = _FakeProviderOrchestrator()
        app = _app(broker, orchestrator)
        http_request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/responses",
                "raw_path": b"/v1/responses",
                "query_string": b"",
                "headers": [
                    (b"authorization", b"Bearer owner"),
                    (
                        b"avalan-extensions",
                        TASK_INPUT_EXTENSION.encode(),
                    ),
                ],
                "client": ("127.0.0.1", 1),
                "server": ("server.test", 80),
                "root_path": "",
                "app": app,
            }
        )
        request = ResponsesRequest(
            model="fake-model",
            input=[ChatMessage(role=MessageRole.USER, content="Start.")],
            stream=stream,
            extensions=_ATTACHED_EXTENSIONS,
        )
        original_install_segment = ServerInteractionRun.install_segment

        async def install_then_report_existing(
            run_: ServerInteractionRun,
            segment: ServerDetachedSegment,
        ) -> object:
            await original_install_segment(run_, segment)
            raise RuntimeError("segment already retained")

        try:
            if stream:
                response = await create_response(
                    request,
                    getLogger("server-responses-disconnect"),
                    orchestrator,
                    output_redaction_settings=(
                        ServerOutputRedactionSettings()
                    ),
                    http_request=http_request,
                )
                assert isinstance(response, StreamingResponse)
                body_iterator = response.body_iterator.__aiter__()
                assert "response.created" in str(await anext(body_iterator))
                assert "input_request.created" in str(
                    await anext(body_iterator)
                )
                assert "input_request.presented" in str(
                    await anext(body_iterator)
                )
                blocked_read = ensure_future(anext(body_iterator))
                blocked_read.set_name("server-responses-stream-disconnect")
                await sleep(0)
                with patch.object(
                    ServerInteractionRun,
                    "install_segment",
                    install_then_report_existing,
                ):
                    blocked_read.cancel()
                    with raises(CancelledError):
                        await blocked_read
            else:
                response_task = create_task(
                    create_response(
                        request,
                        getLogger("server-responses-disconnect"),
                        orchestrator,
                        output_redaction_settings=(
                            ServerOutputRedactionSettings()
                        ),
                        http_request=http_request,
                    ),
                    name="server-responses-json-disconnect",
                )
                await wait_for(orchestrator.request_ready.wait(), timeout=2)
                with patch.object(
                    ServerInteractionRun,
                    "install_segment",
                    install_then_report_existing,
                ):
                    response_task.cancel()
                    with raises(CancelledError):
                        await response_task
            assert orchestrator.request is not None
            pending = await broker.inspect(
                ScopedInteractionLookup(
                    actor=_OWNER,
                    correlation=InteractionCorrelation.from_request(
                        orchestrator.request
                    ),
                )
            )
            assert isinstance(pending, InteractionRecord)
            assert pending.request.state is RequestState.PENDING
        finally:
            await close_server_interactions(app)
            await broker.aclose()


async def _defensive_service_scenario() -> None:
    broker = await _open_broker()
    orchestrator = _FakeProviderOrchestrator()
    authorizer = _Authorizer()
    app = _app(broker, orchestrator, authorizer=authorizer)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://server.test",
        ) as client:
            required = await client.post(
                "/v1/chat/completions",
                headers=_EXTENSION_HEADERS,
                json=_completion_payload(stream=True, handling="detached"),
            )
            assert required.status_code == 200
            assert orchestrator.request is not None
            request = orchestrator.request
            projection = await broker.inspect(
                ScopedInteractionLookup(
                    actor=_OWNER,
                    correlation=InteractionCorrelation.from_request(request),
                )
            )
            assert isinstance(projection, InteractionRecord)
            record = projection
            service = cast(
                ServerInteractionService,
                app.state.interaction_service,
            )
            entry = await service.entry(
                _OWNER,
                str(request.request_id),
                InteractionOperation.INSPECT,
            )

            pending_poll = await client.get(
                f"/v1/input/requests/{request.request_id}/poll",
                params={"transport": "json"},
                headers={"Authorization": "Bearer owner"},
            )
            invalid_poll = await client.get(
                f"/v1/input/requests/{request.request_id}/poll",
                params={"after_store_revision": -1},
                headers={"Authorization": "Bearer owner"},
            )
            malformed_json = await client.post(
                f"/v1/input/requests/{request.request_id}/resolve",
                headers={
                    "Authorization": "Bearer owner",
                    "Content-Type": "application/json",
                },
                content="{",
            )
            assert pending_poll.status_code == 200
            assert invalid_poll.status_code == 422
            assert malformed_json.status_code == 422

            with raises(_ServerHTTPError):
                await service.entry(
                    _OWNER,
                    "",
                    InteractionOperation.INSPECT,
                )
            with raises(_ServerHTTPError):
                await service.entry(
                    _OWNER,
                    cast(str, None),
                    InteractionOperation.INSPECT,
                )
            authorizer.allowed = False
            with raises(_ServerHTTPError):
                await service.entry(
                    _OWNER,
                    str(request.request_id),
                    InteractionOperation.INSPECT,
                )
            authorizer.allowed = True
            with raises(ValueError):
                await service._record_for_entry(
                    entry,
                    cast(InteractionOperation, object()),
                )

            target = InteractionRequestAuthorizationTarget(
                request_id=request.request_id,
                origin=request.origin,
            )
            with raises(_ServerHTTPError):
                _validate_server_authorization(
                    object(),
                    _OWNER,
                    InteractionOperation.INSPECT,
                    target,
                )
            with raises(_ServerHTTPError):
                _validate_server_authorization(
                    InteractionAuthorizationDecision(
                        actor=_OWNER,
                        operation=InteractionOperation.WAIT,
                        target=target,
                        allowed=True,
                        disclosure=InteractionDisclosure.FULL,
                    ),
                    _OWNER,
                    InteractionOperation.INSPECT,
                    target,
                )

            fresh_run = ServerInteractionRun(
                service=service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            with raises(RuntimeError):
                await fresh_run.input_required_event()
            provisional = _server_segment()
            with raises(RuntimeError):
                await fresh_run.install_segment(provisional)
            with raises(RuntimeError):
                await fresh_run.install_segment(_server_segment())
            with raises(RuntimeError):
                await fresh_run.register(request)

            provisional_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=broker,
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_Authorizer(),
                )
            )
            provisional_run = ServerInteractionRun(
                service=provisional_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            provisional_segment = _server_segment(
                _blocking_projection_stream()
            )
            with raises(RuntimeError):
                await provisional_run.install_segment(provisional_segment)
            provisional_reader = create_task(
                provisional_segment.next_projection()
            )
            await sleep(0)
            await wait_for(provisional_service.aclose(), timeout=1)
            with raises(CancelledError):
                await provisional_reader
            assert provisional_segment.closed
            assert provisional_segment.pending_next is None
            assert provisional_reader.done()
            with raises(StopAsyncIteration):
                await provisional_segment.next_projection()
            await provisional_segment.aclose(cancelled=True)
            closed_run = ServerInteractionRun(
                service=provisional_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            closed_segment = _server_segment()
            with raises(RuntimeError):
                await closed_run.install_segment(closed_segment)
            assert closed_segment.closed
            await provisional_service.aclose()
            with raises(RuntimeError):
                await provisional_run.register(request)

            isolated_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=broker,
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_Authorizer(),
                )
            )
            isolated_run = ServerInteractionRun(
                service=isolated_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            with raises(RuntimeError):
                await isolated_run.install_segment(provisional)
            isolated_entry = await isolated_run.register(request)
            assert isolated_entry.segment is provisional
            with raises(RuntimeError):
                await isolated_run.install_segment(_server_segment())
            changed_request = replace(
                request,
                origin=replace(
                    request.origin,
                    run_id=RunId("different-server-run"),
                ),
            )
            with raises(RuntimeError):
                await isolated_run.register(changed_request)
            unhandled = await isolated_run.extension_events(
                _projection(
                    request.origin,
                    10,
                    StreamItemKind.ANSWER_DONE,
                    _correlation(request),
                )
            )
            assert unhandled == ()
            terminal_event = await isolated_run.extension_events(
                _projection(
                    request.origin,
                    11,
                    StreamItemKind.INTERACTION_CANCELLED,
                    _correlation(request),
                )
            )
            assert terminal_event[0]["provenance"] == "external_controller"

            handler = _ServerAttachedInputHandler(isolated_run)
            with patch("avalan.server.interaction.Event", _ImmediateEvent):
                with raises(AssertionError):
                    await handler(InputHandlerContext(request=request))

            raising_authorizer_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=broker,
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_RaisingAuthorizer(),
                )
            )
            raising_authorizer_run = ServerInteractionRun(
                service=raising_authorizer_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            await raising_authorizer_run.register(request)
            with raises(_ServerHTTPError):
                await raising_authorizer_service.entry(
                    _OWNER,
                    str(request.request_id),
                    InteractionOperation.INSPECT,
                )

            fault_broker = _FaultBroker(record)
            fault_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=cast(InteractionBroker, fault_broker),
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_Authorizer(),
                )
            )
            fault_run = ServerInteractionRun(
                service=fault_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.DETACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            fault_entry = await fault_run.register(request)
            contract_error = InputContractError(
                InputErrorCode.NOT_FOUND,
                "test",
                "not found",
            )

            fault_broker.inspect_error = contract_error
            with raises(_ServerHTTPError):
                await fault_service.inspect(_OWNER, str(request.request_id))
            fault_broker.inspect_error = RuntimeError("store failed")
            with raises(_ServerHTTPError):
                await fault_service.inspect(_OWNER, str(request.request_id))
            fault_broker.inspect_error = None
            fault_broker.inspect_value = InteractionTerminalMetadata(
                status=ResolutionStatus.CANCELLED,
                resolved_at=_NOW,
            )
            with raises(_ServerHTTPError):
                await fault_service.inspect(_OWNER, str(request.request_id))
            await fault_service._register(fault_entry)
            fault_broker.inspect_value = object()
            with raises(_ServerHTTPError):
                await fault_service.inspect(_OWNER, str(request.request_id))
            mismatched_record = replace(
                record,
                request=changed_request,
            )
            fault_broker.inspect_value = mismatched_record
            with raises(_ServerHTTPError):
                await fault_service.inspect(_OWNER, str(request.request_id))
            fault_broker.inspect_value = record

            fault_broker.wait_error = contract_error
            with raises(_ServerHTTPError):
                await fault_service.poll(
                    _OWNER,
                    str(request.request_id),
                    int(record.store_revision),
                )
            fault_broker.wait_error = RuntimeError("wait failed")
            with raises(_ServerHTTPError):
                await fault_service.poll(
                    _OWNER,
                    str(request.request_id),
                    int(record.store_revision),
                )
            fault_broker.wait_error = None
            fault_broker.wait_value = object()
            with raises(_ServerHTTPError):
                await fault_service.poll(
                    _OWNER,
                    str(request.request_id),
                    int(record.store_revision),
                )
            fault_broker.wait_value = mismatched_record
            with raises(_ServerHTTPError):
                await fault_service.poll(
                    _OWNER,
                    str(request.request_id),
                    int(record.store_revision),
                )
            fault_broker.wait_value = record
            _poll_entry, polled = await fault_service.poll(
                _OWNER,
                str(request.request_id),
                int(record.store_revision),
            )
            assert polled == record

            resolution_payload = _resolve_payload(
                request,
                int(record.request.state_revision),
                "fault-resolution-key",
            )
            with raises(InputValidationError):
                _resolution_command(
                    _OWNER,
                    record,
                    {**resolution_payload, "answers": object()},
                )
            declined_payload = {
                key: value
                for key, value in resolution_payload.items()
                if key != "answers"
            }
            declined_payload["status"] = "declined"
            assert (
                _resolution_command(
                    _OWNER,
                    record,
                    declined_payload,
                ).proposed_resolution.status
                is ResolutionStatus.DECLINED
            )
            with raises(InputValidationError):
                _resolution_command(
                    _OWNER,
                    record,
                    {**declined_payload, "answers": []},
                )
            with raises(InputValidationError):
                _resolution_command(
                    _OWNER,
                    record,
                    {**declined_payload, "status": "cancelled"},
                )
            with raises(InputValidationError):
                _resolution_command(
                    _OWNER,
                    record,
                    {**resolution_payload, "unknown": True},
                )
            with raises(InputValidationError):
                _resolution_command(
                    _OWNER,
                    record,
                    {
                        **resolution_payload,
                        "expected_state_revision": True,
                    },
                )
            fault_broker.resolve_error = contract_error
            with raises(_ServerHTTPError):
                await fault_service.resolve(
                    _OWNER,
                    str(request.request_id),
                    resolution_payload,
                )
            fault_broker.resolve_error = RuntimeError("resolve failed")
            with raises(_ServerHTTPError):
                await fault_service.resolve(
                    _OWNER,
                    str(request.request_id),
                    resolution_payload,
                )
            fault_broker.resolve_error = None
            fault_broker.resolve_value = _unsafe_broker_result(object())
            with raises(_ServerHTTPError):
                await fault_service.resolve(
                    _OWNER,
                    str(request.request_id),
                    resolution_payload,
                )
            fault_broker.resolve_value = _unsafe_broker_result(
                _unsafe_rejected(
                    ResolveInteractionRejected,
                    InputErrorCode.STALE_REVISION,
                )
            )
            with raises(_ServerHTTPError):
                await fault_service.resolve(
                    _OWNER,
                    str(request.request_id),
                    resolution_payload,
                )
            fault_broker.resolve_value = _unsafe_broker_result(
                _unsafe_applied(ResolveInteractionApplied, mismatched_record)
            )
            with raises(_ServerHTTPError):
                await fault_service.resolve(
                    _OWNER,
                    str(request.request_id),
                    resolution_payload,
                )

            cancel_payload = _cancel_payload(
                request,
                int(record.request.state_revision),
                "fault-cancel-key",
            )
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    {**cancel_payload, "status": "answered"},
                )
            fault_broker.cancel_error = contract_error
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    cancel_payload,
                )
            fault_broker.cancel_error = RuntimeError("cancel failed")
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    cancel_payload,
                )
            fault_broker.cancel_error = None
            fault_broker.cancel_value = _unsafe_broker_result(
                _unsafe_rejected(
                    CancelInteractionRejected,
                    InputErrorCode.STALE_REVISION,
                )
            )
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    cancel_payload,
                )
            fault_broker.cancel_value = _unsafe_broker_result(object())
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    cancel_payload,
                )
            fault_broker.cancel_value = _unsafe_broker_result(
                _unsafe_applied(CancelInteractionApplied, mismatched_record)
            )
            with raises(_ServerHTTPError):
                await fault_service.cancel(
                    _OWNER,
                    str(request.request_id),
                    cancel_payload,
                )

            failed_resolver_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=broker,
                    principal_resolver=_RaisingPrincipalResolver(),
                    authorizer=_Authorizer(),
                )
            )
            with raises(_ServerHTTPError):
                await failed_resolver_service.authenticate(
                    Request(
                        {
                            "type": "http",
                            "method": "GET",
                            "path": "/",
                            "headers": [],
                        }
                    )
                )

            header_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [
                        (
                            b"avalan-extensions",
                            TASK_INPUT_EXTENSION.encode(),
                        ),
                        (b"authorization", b"Bearer owner"),
                    ],
                    "app": app,
                }
            )
            with raises(_ServerHTTPError):
                await prepare_openai_interaction_run(
                    header_request,
                    {
                        "version": "1",
                        "handling": "attached",
                        "extra": True,
                    },
                    surface=ServerInteractionSurface.CHAT,
                )
            with raises(_ServerHTTPError):
                await prepare_openai_interaction_run(
                    header_request,
                    {"version": "1", "handling": 1},
                    surface=ServerInteractionSurface.CHAT,
                )
            assert (
                await prepare_openai_interaction_run(
                    header_request,
                    {"version": "1", "handling": "unavailable"},
                    surface=ServerInteractionSurface.CHAT,
                )
                is None
            )

            async def invalid_json_receive() -> dict[str, object]:
                return {
                    "type": "http.request",
                    "body": b"{",
                    "more_body": False,
                }

            invalid_json_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/",
                    "headers": [],
                },
                receive=invalid_json_receive,
            )
            with raises(_ServerHTTPError):
                await _request_json(invalid_json_request)

            for invalid_configuration in (
                {
                    "broker": broker,
                    "principal_resolver": _PrincipalResolver(),
                    "authorizer": _Authorizer(),
                    "policy": object(),
                },
                {
                    "broker": object(),
                    "principal_resolver": _PrincipalResolver(),
                    "authorizer": _Authorizer(),
                },
                {
                    "broker": broker,
                    "principal_resolver": object(),
                    "authorizer": _Authorizer(),
                },
                {
                    "broker": broker,
                    "principal_resolver": lambda _request: _OWNER,
                    "authorizer": _Authorizer(),
                },
                {
                    "broker": broker,
                    "principal_resolver": _PrincipalResolver(),
                    "authorizer": object(),
                },
            ):
                with raises(TypeError):
                    ServerInteractionConfiguration(
                        **cast(Any, invalid_configuration)
                    )
            with raises(TypeError):
                ServerInteractionConfiguration(
                    broker=broker,
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_Authorizer(),
                    policy=InteractionPolicy(
                        capability_state=TaskInputCapabilityState.ROLLBACK
                    ),
                )
            unavailable_service = ServerInteractionService(
                ServerInteractionConfiguration(
                    broker=cast(InteractionBroker, fault_broker),
                    principal_resolver=_PrincipalResolver(),
                    authorizer=_Authorizer(),
                    policy=InteractionPolicy(
                        capability_state=TaskInputCapabilityState.DORMANT
                    ),
                )
            )
            with raises(_ServerHTTPError):
                await unavailable_service.entry(
                    _OWNER,
                    str(request.request_id),
                    InteractionOperation.RESOLVE,
                )
            with raises(TypeError):
                ServerInteractionService(
                    cast(ServerInteractionConfiguration, object())
                )

            no_terminal_segment = _server_segment(
                _projection_stream(
                    _projection(
                        request.origin,
                        20,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="/",
                    )
                )
            )
            no_terminal_segment.output_redaction_settings = (
                ServerOutputRedactionSettings(enabled=True)
            )
            no_terminal_segment.claim_resume()
            no_terminal_stream = _resume_segment(
                fault_entry,
                record,
                no_terminal_segment,
            )
            assert "execution.resumed" in await anext(no_terminal_stream)
            with raises(RuntimeError):
                while True:
                    await anext(no_terminal_stream)

            completed_segment = _server_segment(
                _projection_stream(
                    _projection(
                        request.origin,
                        20,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="visible/",
                    ),
                    _projection(
                        request.origin,
                        21,
                        StreamItemKind.STREAM_COMPLETED,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                )
            )
            completed_segment.output_redaction_settings = (
                ServerOutputRedactionSettings(enabled=True)
            )
            completed_segment.claim_resume()
            completed_messages = [
                message
                async for message in _resume_segment(
                    fault_entry,
                    record,
                    completed_segment,
                )
            ]
            assert "visible" in "".join(completed_messages)
            assert "[DONE]" in completed_messages[-1]

            disconnected_segment = _server_segment(
                _blocking_projection_stream()
            )
            disconnected_segment.claim_resume()
            disconnected_stream = _resume_segment(
                fault_entry,
                record,
                disconnected_segment,
            )
            await anext(disconnected_stream)
            disconnected_read = ensure_future(anext(disconnected_stream))
            await sleep(0)
            disconnected_read.cancel()
            with raises(CancelledError):
                await disconnected_read
            assert fault_entry.lifecycle[-1].type == "transport.disconnected"
            await disconnected_segment.aclose(cancelled=True)

            retained_read_started = Event()
            orphan_segment = _server_segment(
                _failing_projection_stream(retained_read_started)
            )
            fault_entry.segment = orphan_segment
            abandoned_read = create_task(
                orphan_segment.next_projection(),
                name="server-abandoned-retained-read",
            )
            await wait_for(retained_read_started.wait(), timeout=1)
            retained_read = orphan_segment.pending_next
            assert retained_read is not None
            fault_broker.wait_value = InteractionTerminalMetadata(
                status=ResolutionStatus.CANCELLED,
                resolved_at=_NOW,
            )
            with raises(_ServerHTTPError):
                await wait_for(
                    fault_service.poll(
                        _OWNER,
                        str(request.request_id),
                        int(record.store_revision),
                    ),
                    timeout=1,
                )
            with raises(
                RuntimeError,
                match="retained read cancellation failed",
            ):
                await abandoned_read
            assert orphan_segment.pending_next is None
            assert retained_read.done()

            failed_terminal = _projection(
                request.origin,
                21,
                StreamItemKind.STREAM_CANCELLED,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            )
            failed_json_segment = _server_segment(
                _projection_stream(failed_terminal)
            )
            failed_json_segment.claim_resume()
            with raises(_ServerHTTPError):
                await _resume_segment_json(
                    fault_entry,
                    record,
                    failed_json_segment,
                )

            chat_request = ChatCompletionRequest(
                model="fake-model",
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Start.")
                ],
                stream=False,
            )
            unit_run = ServerInteractionRun(
                service=isolated_service,
                actor=_OWNER,
                handling=ServerInteractionHandling.ATTACHED,
                surface=ServerInteractionSurface.CHAT,
            )
            with raises(StreamValidationError):
                await _interaction_chat_response(
                    request=chat_request,
                    interaction_run=unit_run,
                    response=cast(
                        OrchestratorResponse,
                        _FakeProviderResponse(_empty_projection_stream()),
                    ),
                    response_id="empty-chat-response",
                    timestamp=1,
                    model_id="fake-model",
                    orchestrator=orchestrator,
                    output_redaction_settings=ServerOutputRedactionSettings(),
                )
            with raises(StreamValidationError):
                await _interaction_chat_response(
                    request=chat_request,
                    interaction_run=unit_run,
                    response=cast(
                        OrchestratorResponse,
                        _FakeProviderResponse(
                            _projection_stream(
                                _projection(
                                    request.origin,
                                    0,
                                    StreamItemKind.STREAM_STARTED,
                                ),
                                _projection(
                                    request.origin,
                                    1,
                                    StreamItemKind.STREAM_CANCELLED,
                                    terminal_outcome=(
                                        StreamTerminalOutcome.CANCELLED
                                    ),
                                ),
                            )
                        ),
                    ),
                    response_id="failed-chat-response",
                    timestamp=1,
                    model_id="fake-model",
                    orchestrator=orchestrator,
                    output_redaction_settings=ServerOutputRedactionSettings(),
                )
            blocked_chat = create_task(
                _interaction_chat_response(
                    request=chat_request,
                    interaction_run=unit_run,
                    response=cast(
                        OrchestratorResponse,
                        _FakeProviderResponse(_blocking_projection_stream()),
                    ),
                    response_id="cancelled-chat-response",
                    timestamp=1,
                    model_id="fake-model",
                    orchestrator=orchestrator,
                    output_redaction_settings=ServerOutputRedactionSettings(),
                ),
                name="server-chat-non-stream-cancel",
            )
            await sleep(0)
            blocked_chat.cancel()
            with raises(CancelledError):
                await blocked_chat

            projection_adapter = _DetachedResponsesProjection(
                projector=_ResponsesSSEProjector(
                    "coverage-response",
                    ServerOutputRedactionSettings(),
                ),
                response_id="coverage-response",
                created=1,
                model_id="fake-model",
            )
            completed_terminal = _projection(
                request.origin,
                22,
                StreamItemKind.STREAM_COMPLETED,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            projection_adapter.indexed_output[1] = {"type": "message"}
            with raises(StreamValidationError):
                projection_adapter.json_body(completed_terminal, object())
            projection_adapter.indexed_output.clear()
            failed_terminal = replace(
                _projection(
                    request.origin,
                    22,
                    StreamItemKind.STREAM_ERRORED,
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
                data={"error": {"message": "failed"}},
            )
            failed_body = projection_adapter.json_body(
                failed_terminal,
                object(),
            )
            assert failed_body["error"] == {"message": "failed"}

            responses_json_segment = _server_segment(
                _projection_stream(
                    _projection(
                        request.origin,
                        22,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="response",
                    ),
                    _projection(
                        request.origin,
                        23,
                        StreamItemKind.STREAM_COMPLETED,
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                ),
                protocol=ServerInteractionSurface.RESPONSES,
            )
            responses_json_segment.response = cast(
                OrchestratorResponse,
                _FakeProviderResponse(_empty_projection_stream()),
            )
            responses_json_segment.claim_resume()
            completed_json = await _resume_segment_json(
                fault_entry,
                record,
                responses_json_segment,
            )
            assert loads(bytes(completed_json.body))["type"] == "response"
    finally:
        await close_server_interactions(app)
        await broker.aclose()


def _defensive_value_scenario() -> None:
    event = ServerInteractionLifecycleEvent(
        sequence=0,
        type="input_request.created",
        request_id="request",
        state="pending",
        surface="server",
        validation_code="input.validation",
        idempotent=False,
    )
    assert event.to_dict() == {
        "sequence": 0,
        "type": "input_request.created",
        "request_id": "request",
        "state": "pending",
        "surface": "server",
        "validation_code": "input.validation",
        "idempotent": False,
    }
    for invalid_sequence in (False, -1, 9_007_199_254_740_992):
        with raises(ValueError):
            ServerInteractionLifecycleEvent(
                sequence=invalid_sequence,
                type="event",
                request_id="request",
            )
    for field_name in ("type", "request_id"):
        values = {
            "sequence": 0,
            "type": "event",
            "request_id": "request",
        }
        values[field_name] = ""
        with raises(ValueError):
            ServerInteractionLifecycleEvent(**cast(Any, values))
    for field_name in ("state", "surface", "validation_code"):
        values = {
            "sequence": 0,
            "type": "event",
            "request_id": "request",
            field_name: "",
        }
        with raises(ValueError):
            ServerInteractionLifecycleEvent(**cast(Any, values))
    with raises(ValueError):
        ServerInteractionLifecycleEvent(
            sequence=0,
            type="event",
            request_id="request",
            idempotent=cast(bool, 1),
        )

    segment = _server_segment()
    invalid_fields = (
        ("iterator", object(), TypeError),
        ("orchestrator", object(), TypeError),
        ("protocol", "chat", TypeError),
        ("response_id", "", ValueError),
        ("model_id", "", ValueError),
        ("created", True, TypeError),
        ("choice_count", 0, ValueError),
        ("output_redaction_settings", object(), TypeError),
    )
    for field_name, invalid, error_type in invalid_fields:
        original = getattr(segment, field_name)
        setattr(segment, field_name, invalid)
        with raises(error_type):
            segment.__post_init__()
        setattr(segment, field_name, original)
    segment.responses_projection = cast(Any, object())
    with raises(TypeError):
        segment.__post_init__()
    segment.responses_projection = None
    with raises(RuntimeError):
        segment.release_resume(exhausted=False)

    assert _mapping({"value": 1}) == {"value": 1}
    with raises(InputValidationError):
        _mapping([])
    with raises(InputValidationError):
        _mapping({1: "value"})
    assert _required_string({"key": "value"}, "key", maximum=5) == "value"
    for invalid_string in (None, "", "longer", 1):
        with raises(InputValidationError):
            _required_string({"key": invalid_string}, "key", maximum=5)
    assert _validate_after_store_revision(None) is None
    assert _validate_after_store_revision(1) == 1
    for invalid_revision in (True, -1, 9_007_199_254_740_992):
        with raises(_ServerHTTPError):
            _validate_after_store_revision(invalid_revision)

    with raises(ValueError):
        extension_sse_message({})
    assert "event: input.event" in extension_sse_message(
        {"type": "input.event"}
    )
    assert task_input_extension_from_request(object()) is None

    class MappingExtensions:
        extensions = {"task_input": {"version": "1", "handling": "attached"}}

    class DumpableTaskInput:
        def model_dump(self, *, exclude_none: bool) -> dict[str, str]:
            assert exclude_none
            return {"version": "1", "handling": "attached"}

    class DumpableExtensions:
        extensions = {"task_input": DumpableTaskInput()}

    class InvalidExtensions:
        extensions = object()

    assert task_input_extension_from_request(MappingExtensions()) == {
        "version": "1",
        "handling": "attached",
    }
    assert task_input_extension_from_request(DumpableExtensions()) == {
        "version": "1",
        "handling": "attached",
    }
    with raises(_ServerHTTPError):
        task_input_extension_from_request(InvalidExtensions())

    for code, status in (
        (InputErrorCode.STALE_REVISION, 409),
        (InputErrorCode.EXPIRED, 410),
        (InputErrorCode.ALREADY_RESOLVED, 409),
        (InputErrorCode.IDEMPOTENCY_CONFLICT, 409),
        (InputErrorCode.IDEMPOTENCY_LEDGER_FULL, 409),
        (InputErrorCode.SUPERSEDED, 409),
        (InputErrorCode.NOT_FOUND, 404),
        (InputErrorCode.FORBIDDEN, 403),
        (InputErrorCode.UNAVAILABLE, 503),
        (InputErrorCode.INVALID_FORMAT, 422),
    ):
        assert _http_error_for_code(code).status_code == status
    assert _ServerHTTPError.expired().status_code == 410
    with raises(_ServerHTTPError):
        _store_result(object())

    origin = _origin(_OWNER_SCOPE)
    terminal_projections = {
        outcome: _projection(
            origin,
            index,
            kind,
            terminal_outcome=outcome,
        )
        for index, (outcome, kind) in enumerate(
            (
                (
                    StreamTerminalOutcome.COMPLETED,
                    StreamItemKind.STREAM_COMPLETED,
                ),
                (
                    StreamTerminalOutcome.CANCELLED,
                    StreamItemKind.STREAM_CANCELLED,
                ),
                (
                    StreamTerminalOutcome.ERRORED,
                    StreamItemKind.STREAM_ERRORED,
                ),
            )
        )
    }
    chat_segment = _server_segment()
    responses_segment = _server_segment(
        protocol=ServerInteractionSurface.RESPONSES
    )
    assert "chat.completion.chunk" in _resume_text_message(
        chat_segment,
        "text",
        None,
    )
    assert "sequence_number" in _resume_text_message(
        responses_segment,
        "text",
        _projection(
            origin,
            4,
            StreamItemKind.ANSWER_DELTA,
            text_delta="text",
        ),
    )
    assert "response.completed" in _resume_terminal_message(
        responses_segment,
        terminal_projections[StreamTerminalOutcome.COMPLETED],
    )
    assert "response.cancelled" in _resume_terminal_message(
        responses_segment,
        terminal_projections[StreamTerminalOutcome.CANCELLED],
    )
    assert "response.failed" in _resume_terminal_message(
        responses_segment,
        terminal_projections[StreamTerminalOutcome.ERRORED],
    )
    assert "[DONE]" in _resume_terminal_message(
        chat_segment,
        terminal_projections[StreamTerminalOutcome.COMPLETED],
    )
    assert "chat.completion.cancelled" in _resume_terminal_message(
        chat_segment,
        terminal_projections[StreamTerminalOutcome.CANCELLED],
    )
    assert "chat.completion.failed" in _resume_terminal_message(
        chat_segment,
        terminal_projections[StreamTerminalOutcome.ERRORED],
    )


def test_server_input_lifecycle_events() -> None:
    """Cover attached, detached, cancellation, and reconnect lifecycles."""
    run(_attached_scenario())
    run(_responses_scenario())
    run(_cancel_scenario())
    run(_disconnect_scenario())
    run(_responses_disconnect_scenario())


def test_server_input_authenticated_resolution() -> None:
    """Cover scoped, correlated, idempotent remote resolution."""
    run(_detached_scenario())
    run(_defensive_service_scenario())


def test_server_input_extension_envelopes() -> None:
    """Cover streaming and non-streaming public extension envelopes."""
    run(_non_streaming_scenario())
    run(_attached_non_streaming_scenario())
    run(_responses_non_streaming_scenario())
    run(_negotiation_scenario())


def test_server_input_policy_negotiation() -> None:
    """Reject new HTTP negotiation unless task input is active."""
    run(_policy_negotiation_scenario())


def test_server_input_readable_fallback() -> None:
    """Cover readable fallback text for clients without form rendering."""
    _defensive_value_scenario()
