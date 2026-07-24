"""Exercise the cross-boundary MCP input contract."""

from asyncio import CancelledError, create_task, run, sleep, wait_for
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from logging import getLogger
from socket import AF_INET, SO_REUSEADDR, SOCK_STREAM, SOL_SOCKET, socket
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.session import ElicitationFnT
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel, Field
from uvicorn import Config, Server

from avalan.agent.execution import (
    AgentExecution,
    AttachedInteractionRuntime,
    BranchInteractionBroker,
)
from avalan.entities import ToolCallContext
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    FreeFormOther,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerResolution,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputTransitionApplied,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionDisclosure,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    RunId,
    SessionId,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    TaskId,
    TextAnswer,
    TextQuestion,
    TurnId,
    UserId,
    mark_request_pending,
)
from avalan.interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from avalan.interaction.handler import InputDisconnectReason
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.server.interaction import (
    ServerInteractionConfiguration,
    ServerInteractionService,
)
from avalan.server.mcp_session import (
    MCP_ELICITATION_CREATE_METHOD,
    MCP_PROTOCOL_VERSION,
    MCPFormSessionRegistry,
    mcp_form_other_property_name,
)
from avalan.server.mcp_tasks import (
    MCPTaskController,
    MCPTaskPolicy,
    MCPTaskRequest,
    require_related_task_metadata,
    without_related_task_metadata,
)
from avalan.server.routers import mcp as mcp_router
from avalan.tool import mcp_session as downstream_mcp

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
_OWNER = PrincipalScope(
    user_id=UserId("mcp-owner"),
    session_id=SessionId("mcp-owner-session"),
)
_CHOICES = (
    Choice(value=ChoiceValue("alpha"), label="Alpha"),
    Choice(value=ChoiceValue("beta"), label="Beta"),
)


class _NameForm(BaseModel):
    name: str = Field(title="Name", min_length=1, max_length=20)


def _origin(*, task_id: str = "task") -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        task_id=TaskId(task_id),
        agent_id=AgentId("00000000-0000-0000-0000-000000000001"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://mcp-contract",
            agent_definition_revision="revision",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model",
            tool_revision="tools",
            capability_revision="capabilities",
        ),
        principal=_OWNER,
    )


def _request(
    *questions: InputQuestion,
    request_id: str = "request",
    reason: str = "Additional input is required.",
) -> InputRequest:
    created = InputRequest(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"continuation-{request_id}"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason=reason,
        questions=questions,
        created_at=_NOW,
    )
    pending = mark_request_pending(
        created,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(pending, InputTransitionApplied)
    return pending.request


async def _registry(
    capabilities: object = {"elicitation": {"form": {}}},
    *,
    can_route: bool = True,
    wait_seconds: float = 0.2,
) -> MCPFormSessionRegistry:
    registry = MCPFormSessionRegistry(response_wait_seconds=wait_seconds)
    await registry.initialize(
        session_id="session",
        owner=_OWNER,
        protocol_version=MCP_PROTOCOL_VERSION,
        capabilities=capabilities,
        can_route_and_resume=can_route,
    )
    await registry.mark_initialized("session", _OWNER)
    return registry


async def _round_trip(
    registry: MCPFormSessionRegistry,
    request: InputRequest,
    result: dict[str, object],
    *,
    related_task_id: str | None = None,
) -> tuple[object, object]:
    handler = registry.handler(
        session_id="session",
        owner=_OWNER,
        related_request_id="tool-call",
        related_task_id=related_task_id,
    )
    handled = create_task(handler(InputHandlerContext(request=request)))
    outbound = await registry.next_outbound("session", _OWNER)
    assert outbound is not None
    if related_task_id is not None:
        result["_meta"] = {
            "io.modelcontextprotocol/related-task": {"taskId": related_task_id}
        }
    await registry.dispatch_response(
        "session",
        _OWNER,
        {
            "jsonrpc": "2.0",
            "id": outbound.jsonrpc_id,
            "result": result,
        },
    )
    return outbound, await handled


async def _negotiated_accept() -> None:
    registry = await _registry()
    request = _request(
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        )
    )
    outbound, outcome = await _round_trip(
        registry,
        request,
        {"action": "accept", "content": {"confirm": True}},
    )
    assert outbound.method == MCP_ELICITATION_CREATE_METHOD
    assert outbound.canonical_request_id == str(request.request_id)
    assert outbound.params["mode"] == "form"
    schema = cast(dict[str, object], outbound.params["requestedSchema"])
    assert schema["required"] == ["confirm"]
    assert isinstance(outcome, InputHandlerResolution)
    assert isinstance(outcome.resolution, AnsweredResolution)
    answer = cast(ConfirmationAnswer, outcome.resolution.answers[0])
    assert answer.value is True

    legacy = await _registry({"elicitation": {}})
    assert (await legacy.negotiation("session", _OWNER)).form_available
    legacy_outbound, _ = await _round_trip(
        legacy,
        request,
        {"action": "decline"},
    )
    assert "mode" not in legacy_outbound.params


async def _semantic_outcomes() -> None:
    registry = await _registry({"elicitation": {"form": {}, "url": {}}})
    request = _request(
        MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Notes?",
            required=True,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("choice"),
            prompt="Choose.",
            required=True,
            choices=_CHOICES,
            allow_other=True,
        ),
    )
    other = mcp_form_other_property_name("choice")
    outbound, outcome = await _round_trip(
        registry,
        request,
        {
            "action": "accept",
            "content": {
                "notes": "line one\nline two",
                other: "custom",
            },
        },
    )
    schema = cast(dict[str, object], outbound.params["requestedSchema"])
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    assert properties["choice"]["enum"] == ["alpha", "beta"]
    assert properties[other]["type"] == "string"
    assert isinstance(outcome, InputHandlerResolution)
    assert isinstance(outcome.resolution, AnsweredResolution)
    notes = cast(MultilineTextAnswer, outcome.resolution.answers[0])
    selection = cast(SingleSelectionAnswer, outcome.resolution.answers[1])
    assert notes.value == "line one\nline two"
    assert selection.value == FreeFormOther(text="custom")

    for index, action in enumerate(("decline", "cancel"), start=1):
        _, terminal = await _round_trip(
            registry,
            _request(
                ConfirmationQuestion(
                    question_id=QuestionId("confirm"),
                    prompt="Continue?",
                    required=True,
                ),
                request_id=f"action-{index}",
            ),
            {"action": action},
        )
        if action == "decline":
            assert isinstance(terminal, InputHandlerResolution)
            assert isinstance(terminal.resolution, DeclinedResolution)
        else:
            assert isinstance(terminal, InputHandlerDisconnected)
            assert terminal.reason is InputDisconnectReason.HANDLER_CANCELLED

    unsafe = registry.handler(
        session_id="session",
        owner=_OWNER,
        related_request_id="unsafe-call",
    )
    rejected = await unsafe(
        InputHandlerContext(
            request=_request(
                TextQuestion(
                    question_id=QuestionId("api-key"),
                    prompt="Enter an API key.",
                    required=True,
                ),
                request_id="unsafe",
            )
        )
    )
    assert isinstance(rejected, InputHandlerDisconnected)
    assert rejected.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
    assert (
        await registry.next_outbound("session", _OWNER, timeout_seconds=0.01)
        is None
    )


async def _durable_task_projection() -> None:
    owner = (_OWNER, "session")
    controller = MCPTaskController(
        MCPTaskPolicy(page_size=1, default_ttl_ms=200),
        id_factory=iter(("task-a", "task-b")).__next__,
    )
    creation = await controller.create(
        MCPTaskRequest(requested_ttl_ms=100),
        requestor=owner,
    )
    assert creation.as_dict()["task"]["status"] == "working"
    result = create_task(controller.result("task-a", requestor=owner))
    await sleep(0)
    assert not result.done()
    assert (await creation.handle.transition_input_required())[
        "status"
    ] == "input_required"
    assert (await controller.get("task-a", requestor=owner))[
        "status"
    ] == "input_required"
    await sleep(0)
    assert not result.done()
    assert (await creation.handle.transition_working())["status"] == "working"
    exact = {"content": [{"type": "text", "text": "complete"}]}
    await creation.handle.complete(exact)
    outcome = await result
    assert outcome.result is not None
    require_related_task_metadata(outcome.result, "task-a")
    assert without_related_task_metadata(outcome.result) == exact

    second = await controller.create(MCPTaskRequest(), requestor=owner)
    first_page = await controller.list(requestor=owner)
    assert first_page["nextCursor"] == "task-a"
    second_page = await controller.list(
        requestor=owner,
        cursor=cast(str, first_page["nextCursor"]),
    )
    assert (
        cast(list[dict[str, object]], second_page["tasks"])[0]["taskId"]
        == "task-b"
    )
    assert (await controller.cancel("task-b", requestor=owner))[
        "status"
    ] == "cancelled"
    assert second.handle.cancellation_requested


async def _incapable_fallback() -> None:
    question = ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=True,
    )
    for capabilities, can_route in (
        ({}, True),
        ({"elicitation": {"url": {}}}, True),
        ({"elicitation": {"form": {}}}, False),
    ):
        registry = await _registry(capabilities, can_route=can_route)
        outcome = await registry.handler(
            session_id="session",
            owner=_OWNER,
            related_request_id="call",
        )(InputHandlerContext(request=_request(question)))
        assert isinstance(outcome, InputHandlerDisconnected)
        assert outcome.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
        assert await registry.pending_count("session", _OWNER) == 0


async def _bounded_liveness() -> None:
    registry = await _registry(wait_seconds=0.01)
    request = _request(
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        )
    )
    handled = create_task(
        registry.handler(
            session_id="session",
            owner=_OWNER,
            related_request_id="call",
        )(InputHandlerContext(request=request))
    )
    assert await registry.next_outbound("session", _OWNER) is not None
    outcome = await handled
    assert isinstance(outcome, InputHandlerDisconnected)
    assert outcome.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
    assert await registry.pending_count("session", _OWNER) == 0


class _DownstreamBroker:
    def __init__(self, result: InteractionRequestResult) -> None:
        self.result = result
        self.requests: list[InteractionBrokerRequest] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        return self.result

    async def cancel_scope(self, command: object) -> object:
        del command
        return object()


async def _unused_handler(context: object) -> object:
    del context
    raise AssertionError("the downstream broker owns the interaction")


async def _downstream_round_trip() -> None:
    origin = _origin(task_id="downstream-task")
    resolution = AnsweredResolution(
        request_id=InputRequestId("downstream-request"),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            TextAnswer(
                question_id=QuestionId("name"),
                provenance=AnswerProvenance.HUMAN,
                value="Ada",
            ),
        ),
    )
    result = cast(
        InteractionRequestResult,
        SimpleNamespace(
            delivery=SimpleNamespace(
                record=SimpleNamespace(
                    request=SimpleNamespace(resolution=resolution)
                )
            )
        ),
    )
    broker = _DownstreamBroker(result)
    runtime = AttachedInteractionRuntime(
        broker=cast(Any, broker),
        actor=InteractionActor(principal=origin.principal),
        handler=_unused_handler,
    )
    execution = cast(
        AgentExecution,
        SimpleNamespace(origin=origin, interaction_runtime=runtime),
    )
    context = ToolCallContext(
        agent_id=UUID("00000000-0000-0000-0000-000000000001"),
        execution=execution,
        execution_origin=origin,
        interaction_broker=cast(BranchInteractionBroker, broker),
    )
    router = downstream_mcp._ElicitationRouter(
        uri="memory://downstream",
        tool_name="ask",
        context=context,
        related_task_id=None,
        ttl_seconds=60,
    )
    server = FastMCP("downstream-contract")

    @server.tool()
    async def ask(ctx: Context) -> str:
        elicited = await ctx.elicit(
            message="Who should be greeted?",
            schema=_NameForm,
        )
        return elicited.data.name if elicited.data is not None else "none"

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        server_task = create_task(
            server._mcp_server.run(
                *server_streams,
                server._mcp_server.create_initialization_options(),
                raise_exceptions=True,
            )
        )
        try:
            async with ClientSession(
                *client_streams,
                elicitation_callback=cast(ElicitationFnT, router),
            ) as session:
                initialized = await downstream_mcp._initialize(
                    session,
                    form_capable=True,
                )
                response = await session.call_tool("ask", {})
        finally:
            server_task.cancel()
            with suppress(CancelledError):
                await server_task

    assert initialized.protocolVersion == MCP_PROTOCOL_VERSION
    assert response.content[0].text == "Ada"
    assert len(broker.requests) == 1
    assert broker.requests[0].origin is origin
    assert broker.requests[0].actor.principal == origin.principal


class _InboundBoundary:
    async def __call__(self, request: object) -> InteractionActor | None:
        headers = getattr(request, "headers", {})
        if headers.get("Authorization") != "Bearer mcp-owner":
            return None
        return InteractionActor(principal=_OWNER)

    async def authorize(
        self,
        actor: InteractionActor,
        operation: object,
        target: object,
    ) -> object:
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )

    async def inspect(self, value: object) -> object:
        return value

    async def resolve(self, value: object) -> object:
        return value

    async def cancel(self, value: object) -> object:
        return value

    async def wait(self, value: object) -> object:
        return value

    async def request(self, value: object) -> object:
        return value

    async def cancel_scope(self, value: object) -> object:
        return value


class _InboundOrchestrator:
    tool = None

    def __init__(self) -> None:
        self.scenario = ""
        self.continuations: list[tuple[str, ContinuationId]] = []
        self.requests: list[InputRequest] = []

    async def sync_messages(self, response: object) -> None:
        del response


class _InboundResponse:
    input_token_count = 1
    output_token_count = 1
    _response_iterator = None

    def __init__(
        self,
        runtime: AttachedInteractionRuntime | None,
        orchestrator: _InboundOrchestrator,
    ) -> None:
        self.runtime = runtime
        self.orchestrator = orchestrator

    async def to_str(self) -> str:
        return ""

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        yield CanonicalStreamItem(
            stream_session_id="mcp-public",
            run_id="mcp-public",
            turn_id="mcp-public",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        scenario = self.orchestrator.scenario
        if self.runtime is None:
            answer = "unavailable"
        else:
            question: InputQuestion
            if scenario == "sensitive":
                question = TextQuestion(
                    question_id=QuestionId("api-key"),
                    prompt="Enter an API key.",
                    required=True,
                )
            else:
                question = TextQuestion(
                    question_id=QuestionId("name"),
                    prompt="Name?",
                    required=True,
                )
            request = _request(
                question,
                request_id=f"router-{scenario}",
            )
            self.orchestrator.requests.append(request)
            self.orchestrator.continuations.append(
                (
                    "entered",
                    request.continuation_id,
                )
            )
            outcome = await self.runtime.handler(
                InputHandlerContext(request=request)
            )
            self.orchestrator.continuations.append(
                (
                    "resumed",
                    request.continuation_id,
                )
            )
            if isinstance(outcome, InputHandlerResolution):
                if isinstance(outcome.resolution, AnsweredResolution):
                    value = cast(TextAnswer, outcome.resolution.answers[0])
                    answer = f"continued:{value.value}"
                else:
                    assert isinstance(outcome.resolution, DeclinedResolution)
                    answer = "declined"
            else:
                assert isinstance(outcome, InputHandlerDisconnected)
                answer = (
                    "cancelled"
                    if outcome.reason
                    is InputDisconnectReason.HANDLER_CANCELLED
                    else "unavailable"
                )
        yield CanonicalStreamItem(
            stream_session_id="mcp-public",
            run_id="mcp-public",
            turn_id="mcp-public",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=answer,
        )
        yield CanonicalStreamItem(
            stream_session_id="mcp-public",
            run_id="mcp-public",
            turn_id="mcp-public",
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        yield CanonicalStreamItem(
            stream_session_id="mcp-public",
            run_id="mcp-public",
            turn_id="mcp-public",
            sequence=3,
            kind=StreamItemKind.USAGE_COMPLETED,
            channel=StreamChannel.USAGE,
            usage={},
        )
        yield CanonicalStreamItem(
            stream_session_id="mcp-public",
            run_id="mcp-public",
            turn_id="mcp-public",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )


async def _inbound_router_round_trip() -> None:
    boundary = _InboundBoundary()
    orchestrator = _InboundOrchestrator()
    app = FastAPI()
    app.state.logger = getLogger("mcp-public-e2e")
    app.state.orchestrator = orchestrator
    app.state.interaction_service = ServerInteractionService(
        ServerInteractionConfiguration(
            broker=cast(Any, boundary),
            principal_resolver=boundary,
            authorizer=cast(Any, boundary),
        )
    )
    app.state.mcp_form_session_registry = MCPFormSessionRegistry(
        response_wait_seconds=2,
    )
    app.include_router(mcp_router.create_router(), prefix="/mcp")
    listener = socket(AF_INET, SOCK_STREAM)
    listener.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = cast(tuple[str, int], listener.getsockname())[1]
    server = Server(
        Config(
            app,
            log_level="critical",
            lifespan="off",
        )
    )

    async def orchestrate(
        *args: object,
        **kwargs: object,
    ) -> tuple[_InboundResponse, UUID, int]:
        del args
        runtime = cast(
            AttachedInteractionRuntime | None,
            kwargs["interaction_runtime"],
        )
        return _InboundResponse(runtime, orchestrator), uuid4(), 1

    async def invoke(
        scenario: str,
        action: str | None,
    ) -> tuple[str, int]:
        orchestrator.scenario = scenario
        callbacks = 0

        async def elicit(
            _context: object,
            params: object,
        ) -> mcp_types.ElicitResult:
            nonlocal callbacks
            callbacks += 1
            assert isinstance(params, mcp_types.ElicitRequestFormParams)
            if action == "accept":
                return mcp_types.ElicitResult(
                    action="accept",
                    content={"name": "Ada"},
                )
            assert action in {"decline", "cancel"}
            return mcp_types.ElicitResult(action=action)

        async with AsyncClient(
            headers={"Authorization": "Bearer mcp-owner"},
            timeout=5,
        ) as client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp",
                http_client=client,
            ) as (read_stream, write_stream, _session_id):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=5),
                    elicitation_callback=(
                        cast(ElicitationFnT, elicit)
                        if action is not None
                        else None
                    ),
                ) as session:
                    initialized = await downstream_mcp._initialize(
                        session,
                        form_capable=action is not None,
                    )
                    assert initialized.protocolVersion == MCP_PROTOCOL_VERSION
                    result = await session.call_tool(
                        "run",
                        {"input_string": scenario},
                        read_timeout_seconds=timedelta(seconds=5),
                    )
        return cast(Any, result.content[0]).text, callbacks

    with (
        patch.object(mcp_router, "Orchestrator", _InboundOrchestrator),
        patch.object(mcp_router, "resolve_model_id", return_value="gpt"),
        patch.object(mcp_router, "orchestrate", orchestrate),
    ):
        server_task = create_task(server.serve(sockets=[listener]))
        try:
            for _ in range(200):
                if server.started:
                    break
                await sleep(0.01)
            assert server.started
            assert await invoke("accept", "accept") == ("continued:Ada", 1)
            assert await invoke("decline", "decline") == ("declined", 1)
            assert await invoke("cancel", "cancel") == ("cancelled", 1)
            assert await invoke("incapable", None) == ("unavailable", 0)
            assert await invoke("sensitive", "accept") == ("unavailable", 0)
        finally:
            server.should_exit = True
            await wait_for(server_task, timeout=5)
            listener.close()
            await mcp_router.close_mcp_state(app)

    assert orchestrator.continuations[:2] == [
        ("entered", ContinuationId("continuation-router-accept")),
        ("resumed", ContinuationId("continuation-router-accept")),
    ]
    assert len(orchestrator.requests) == 4


async def _public_projection() -> None:
    await _inbound_router_round_trip()
    await _negotiated_accept()
    await _semantic_outcomes()
    await _incapable_fallback()
    await _downstream_round_trip()
    await _durable_task_projection()
    await _bounded_liveness()


def test_requirement_input_n_075() -> None:
    run(_negotiated_accept())


def test_requirement_input_n_076() -> None:
    run(_semantic_outcomes())


def test_requirement_input_n_077() -> None:
    run(_durable_task_projection())


def test_requirement_input_n_078() -> None:
    run(_incapable_fallback())


def test_requirement_input_n_079() -> None:
    run(_bounded_liveness())


def test_requirement_input_n_080() -> None:
    run(_downstream_round_trip())
