"""Exercise initialized MCP router projection without external services."""

from asyncio import CancelledError, Event, create_task, sleep, wait_for
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from avalan.agent.orchestrator import Orchestrator
from avalan.interaction import (
    AgentId,
    BranchId,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputHandlerContext,
    InputHandlerResolution,
    InputRequest,
    InputRequestId,
    InputTransitionApplied,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionDisclosure,
    InteractionPolicy,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    RunId,
    SessionId,
    StateRevision,
    StreamSessionId,
    TaskId,
    TaskInputCapabilityState,
    TurnId,
    UserId,
    mark_request_pending,
)
from avalan.server.interaction import (
    ServerInteractionConfiguration,
    ServerInteractionService,
)
from avalan.server.mcp_session import (
    MCPFormErrorCode,
    MCPFormSessionError,
)
from avalan.server.mcp_tasks import (
    MCPTaskController,
    MCPTaskPolicy,
    MCPTaskRequest,
    require_related_task_metadata,
)
from avalan.server.routers import mcp as mcp_router


class _Request:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        app: object | None = None,
    ) -> None:
        self.headers = headers or {}
        self.state = SimpleNamespace()
        self.app = app or SimpleNamespace(
            title="Avalan MCP", version="1.0.0", state=SimpleNamespace()
        )
        self._body = body

    async def stream(self) -> AsyncIterator[bytes]:
        yield self._body


class _Broker:
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


class _Resolver:
    async def __call__(self, request: object) -> InteractionActor | None:
        headers = getattr(request, "headers", {})
        if headers.get("Authorization") != "Bearer owner":
            return None
        return InteractionActor(
            principal=PrincipalScope(
                user_id=UserId("owner"),
                session_id=SessionId("owner-session"),
            )
        )


class _AnonymousResolver:
    async def __call__(self, request: object) -> InteractionActor | None:
        del request
        return InteractionActor(principal=PrincipalScope())


class _Authorizer:
    async def authorize(
        self,
        actor: InteractionActor,
        operation: object,
        target: object,
    ) -> InteractionAuthorizationDecision:
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


def _input_request(question: ConfirmationQuestion) -> InputRequest:
    owner = PrincipalScope(
        user_id=UserId("owner"),
        session_id=SessionId("owner-session"),
    )
    request = InputRequest(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=ExecutionOrigin(
            run_id=RunId("run"),
            turn_id=TurnId("turn"),
            task_id=TaskId("task"),
            agent_id=AgentId("agent"),
            branch_id=BranchId("branch"),
            model_call_id=ModelCallId("model-call"),
            stream_session_id=StreamSessionId("stream"),
            definition=ExecutionDefinitionRef(
                agent_definition_locator="agent://mcp",
                agent_definition_revision="revision",
                operation_id="operation",
                operation_index=0,
                model_config_reference="model",
                tool_revision="tools",
                capability_revision="capabilities",
            ),
            principal=owner,
        ),
        mode=RequirementMode.REQUIRED,
        reason="Confirmation is required.",
        questions=(question,),
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )
    pending = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(pending, InputTransitionApplied)
    return pending.request


def _configure(
    request: _Request,
    resolver: object | None = None,
) -> None:
    request.app.state.interaction_service = ServerInteractionService(
        ServerInteractionConfiguration(
            broker=cast(object, _Broker()),
            principal_resolver=cast(Any, resolver or _Resolver()),
            authorizer=_Authorizer(),
        )
    )


def _initialize_message(
    capabilities: object = {"elicitation": {"form": {}}},
    *,
    protocol_version: str = "2025-11-25",
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": "initialize",
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": capabilities,
            "clientInfo": {"name": "test", "version": "1"},
        },
    }


def _route_endpoint(
    *,
    path: str,
    method: str,
) -> Any:
    router = mcp_router.create_router()
    return next(
        cast(Any, route).endpoint
        for route in router.routes
        if cast(Any, route).path == path and method in cast(Any, route).methods
    )


def _wire_request(
    app: object,
    message: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
) -> _Request:
    return _Request(
        app=app,
        headers=headers,
        body=dumps(message, separators=(",", ":")).encode(),
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _initialized_context(
    *,
    capabilities: object = {"elicitation": {"form": {}}},
) -> tuple[_Request, mcp_router._MCPSessionContext]:
    request = _Request(headers={"Authorization": "Bearer owner"})
    _configure(request)
    response = await mcp_router._initialize_mcp_session(
        cast(object, request),
        getLogger("mcp-projection"),
        cast(Orchestrator, object()),
        cast(mcp_router.JSONObject, _initialize_message(capabilities)),
    )
    session_id = response.headers["mcp-session-id"]
    request.headers.update(
        {
            "MCP-Session-Id": session_id,
            "MCP-Protocol-Version": "2025-11-25",
        }
    )
    context = await mcp_router._mcp_session_context(cast(object, request))
    assert context is not None
    await context.registry.mark_initialized(
        context.session_id,
        context.owner,
    )
    context = await mcp_router._mcp_session_context(cast(object, request))
    assert context is not None
    return request, context


@pytest.mark.anyio
async def test_initialized_session_negotiates_tasks_and_form() -> None:
    request, context = await _initialized_context()
    assert context.negotiation.form_available
    assert context.task_requestor is not None
    controller = mcp_router._get_task_controller(cast(object, request))
    capabilities = mcp_router._server_capabilities(
        cast(Orchestrator, object()),
        task_controller=controller,
        task_requestor=context.task_requestor,
    )
    assert capabilities["tasks"] == {
        "list": {},
        "cancel": {},
        "requests": {"tools": {"call": {}}},
    }

    request.state.mcp_task_capable = True
    tools = mcp_router._collect_tool_descriptions(
        cast(object, request),
    )
    assert tools[0]["execution"] == {"taskSupport": "optional"}
    request.state.mcp_task_capable = False
    assert (
        "execution"
        not in mcp_router._collect_tool_descriptions(cast(object, request))[0]
    )

    request.headers["MCP-Protocol-Version"] = "2025-03-26"
    with pytest.raises(MCPFormSessionError):
        await mcp_router._mcp_session_context(cast(object, request))
    request.headers["MCP-Protocol-Version"] = "2025-11-25"
    request.app.state.interaction_service.configuration = (
        ServerInteractionConfiguration(
            broker=cast(object, _Broker()),
            principal_resolver=_Resolver(),
            authorizer=_Authorizer(),
            policy=InteractionPolicy(
                capability_state=TaskInputCapabilityState.DORMANT
            ),
        )
    )
    with pytest.raises(MCPFormSessionError):
        await mcp_router._mcp_session_context(cast(object, request))


@pytest.mark.anyio
async def test_initialize_fails_closed_without_authenticated_identity() -> (
    None
):
    request = _Request()
    _configure(request)
    response = await mcp_router._initialize_mcp_session(
        cast(object, request),
        getLogger("mcp-projection"),
        cast(Orchestrator, object()),
        cast(mcp_router.JSONObject, _initialize_message()),
    )
    payload = loads(response.body)
    assert "mcp-session-id" not in response.headers
    assert "tasks" not in payload["result"]["capabilities"]

    old = _Request(headers={"Authorization": "Bearer owner"})
    _configure(old)
    old_response = await mcp_router._initialize_mcp_session(
        cast(object, old),
        getLogger("mcp-projection"),
        cast(Orchestrator, object()),
        cast(
            mcp_router.JSONObject,
            _initialize_message(protocol_version="2025-06-18"),
        ),
    )
    assert "mcp-session-id" not in old_response.headers
    assert (
        loads(old_response.body)["result"]["protocolVersion"] == "2025-06-18"
    )


@pytest.mark.anyio
async def test_router_dispatches_session_tasks_errors_and_close() -> None:
    rpc = _route_endpoint(path="", method="POST")
    logger = getLogger("mcp-route")
    orchestrator = MagicMock(spec=Orchestrator)
    orchestrator.tool.list_tools.return_value = []

    initialize = _Request(
        headers={"Authorization": "Bearer owner"},
        body=dumps(_initialize_message()).encode(),
    )
    _configure(initialize)
    initialized = await rpc(initialize, logger, orchestrator)
    session_id = initialized.headers["mcp-session-id"]
    app = initialize.app
    headers = {
        "Authorization": "Bearer owner",
        "MCP-Session-Id": session_id,
        "MCP-Protocol-Version": "2025-11-25",
    }
    notified = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert notified.status_code == 204

    for protocol_version in (None, "2025-03-26"):
        protocol_headers = {
            key: value
            for key, value in headers.items()
            if key != "MCP-Protocol-Version"
        }
        if protocol_version is not None:
            protocol_headers["MCP-Protocol-Version"] = protocol_version
        rejected = await rpc(
            _wire_request(
                app,
                {
                    "jsonrpc": "2.0",
                    "id": "version",
                    "method": "ping",
                    "params": {},
                },
                headers=protocol_headers,
            ),
            logger,
            orchestrator,
        )
        error = loads(rejected.body)["error"]
        assert error["code"] == -32602
        assert error["data"] == {"code": "avalan.input.validation"}

    ping = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "version",
                "method": "ping",
                "params": {},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(ping.body) == {
        "jsonrpc": "2.0",
        "id": "version",
        "result": {},
    }

    context_request = _Request(app=app, headers=headers)
    context = await mcp_router._mcp_session_context(
        cast(object, context_request)
    )
    assert context is not None
    handler = context.registry.handler(
        session_id=context.session_id,
        owner=context.owner,
        related_request_id="route-call",
    )
    handled = create_task(
        handler(
            InputHandlerContext(
                request=_input_request(
                    ConfirmationQuestion(
                        question_id=QuestionId("confirm"),
                        prompt="Continue?",
                        required=True,
                    )
                )
            )
        )
    )
    outbound = await context.registry.next_outbound(
        context.session_id,
        context.owner,
    )
    assert outbound is not None
    accepted = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": outbound.jsonrpc_id,
                "result": {
                    "action": "accept",
                    "content": {"confirm": True},
                },
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert accepted.status_code == 202
    assert isinstance(await handled, InputHandlerResolution)

    stale = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": outbound.jsonrpc_id,
                "result": {"action": "decline"},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(stale.body)["error"]["code"] == -32009

    tools = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "tools",
                "method": "tools/list",
                "params": {},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(tools.body)["result"]["tools"][0]["execution"] == {
        "taskSupport": "optional"
    }

    assert context.task_requestor is not None
    controller = mcp_router._get_task_controller(cast(object, context_request))
    await controller.create(
        MCPTaskRequest(),
        requestor=context.task_requestor,
        task_id="route-task",
    )
    task = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "get",
                "method": "tasks/get",
                "params": {"taskId": "route-task"},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(task.body)["result"]["status"] == "working"
    bad_cursor = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "list",
                "method": "tasks/list",
                "params": {"cursor": 1},
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(bad_cursor.body)["error"]["code"] == -32602
    no_session = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "missing",
                "method": "tasks/get",
                "params": {"taskId": "route-task"},
            },
            headers={"Authorization": "Bearer owner"},
        ),
        logger,
        orchestrator,
    )
    assert loads(no_session.body)["error"]["code"] == -32001

    async def task_start(*args: object, **kwargs: object) -> JSONResponse:
        del args, kwargs
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": "call",
                "result": {"task": {"taskId": "stub", "status": "working"}},
            }
        )

    task_call = {
        "jsonrpc": "2.0",
        "id": "call",
        "method": "tools/call",
        "params": {
            "name": "run",
            "arguments": {"input_string": "start"},
            "task": {},
        },
    }
    with patch.object(
        mcp_router,
        "_start_tool_task_response",
        task_start,
    ):
        started = await rpc(
            _wire_request(app, task_call, headers=headers),
            logger,
            orchestrator,
        )
    assert loads(started.body)["result"]["task"]["taskId"] == "stub"

    invalid_call = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "invalid",
                "method": "tools/call",
                "params": [],
            },
            headers=headers,
        ),
        logger,
        orchestrator,
    )
    assert loads(invalid_call.body)["error"]["code"] == -32602
    wrong_actor = await rpc(
        _wire_request(
            app,
            {
                "jsonrpc": "2.0",
                "id": "ping",
                "method": "ping",
                "params": {},
            },
            headers={
                **headers,
                "Authorization": "Bearer wrong",
            },
        ),
        logger,
        orchestrator,
    )
    assert loads(wrong_actor.body)["error"]["code"] == -32001

    mark_error = MCPFormSessionError(
        MCPFormErrorCode.NOT_INITIALIZED,
        -32001,
        "MCP session is not initialized.",
    )
    with patch.object(
        context.registry,
        "mark_initialized",
        AsyncMock(side_effect=mark_error),
    ):
        rejected_notification = await rpc(
            _wire_request(
                app,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                headers=headers,
            ),
            logger,
            orchestrator,
        )
    assert loads(rejected_notification.body)["error"]["code"] == -32001

    with pytest.raises(HTTPException):
        await rpc(
            _wire_request(
                app,
                {
                    "jsonrpc": "2.0",
                    "id": "unknown",
                    "method": "unknown",
                },
                headers=headers,
            ),
            logger,
            orchestrator,
        )

    initialized_endpoint = _route_endpoint(
        path="/notifications/initialized",
        method="POST",
    )
    assert (
        await initialized_endpoint(
            _wire_request(
                app,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                headers=headers,
            ),
            logger,
        )
    ).status_code == 204
    close = _route_endpoint(path="", method="DELETE")
    assert (await close(_Request(app=app, headers=headers))).status_code == 204


@pytest.mark.anyio
async def test_router_accepts_separate_cancellation_notification() -> None:
    request, context = await _initialized_context(capabilities={})
    rpc = _route_endpoint(path="", method="POST")
    logger = getLogger("mcp-cancellation")
    orchestrator = MagicMock(spec=Orchestrator)
    active: dict[str, Event] = {}
    all_started = Event()

    async def orchestrate(
        *args: object,
        **kwargs: object,
    ) -> tuple[object, str, int]:
        del args, kwargs
        return object(), str(uuid4()), 1

    async def stream(**kwargs: object) -> AsyncIterator[bytes]:
        request_id = cast(str, kwargs["request_id"])
        cancel_event = cast(Event, kwargs["cancel_event"])
        active[request_id] = cancel_event
        if len(active) == 3:
            all_started.set()
        await cancel_event.wait()
        yield dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": "Request cancelled"},
            }
        ).encode()

    async def start(request_id: str) -> object:
        return await rpc(
            _wire_request(
                request.app,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "run",
                        "arguments": {"input_string": request_id},
                    },
                },
                headers=request.headers,
            ),
            logger,
            orchestrator,
        )

    async def consume(response: object) -> list[bytes]:
        return [
            chunk
            async for chunk in cast(
                AsyncIterator[bytes],
                getattr(response, "body_iterator"),
            )
        ]

    async def cancel(
        request_id: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> object:
        return await rpc(
            _wire_request(
                request.app,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {
                        "requestId": request_id,
                        "reason": "No longer needed",
                    },
                },
                headers=headers or request.headers,
            ),
            logger,
            orchestrator,
        )

    with (
        patch.object(mcp_router, "orchestrate", orchestrate),
        patch.object(mcp_router, "_stream_mcp_response", stream),
    ):
        first = await start("first")
        second = await start("second")
        disconnected = await start("disconnected")
        first_consumer = create_task(consume(first))
        second_consumer = create_task(consume(second))
        disconnected_consumer = create_task(consume(disconnected))
        await wait_for(all_started.wait(), 1)
        assert set(active) == {"first", "second", "disconnected"}

        assert (await cancel("unknown")).status_code == 202
        assert not any(event.is_set() for event in active.values())

        wrong_owner = await cancel(
            "first",
            headers={
                **request.headers,
                "Authorization": "Bearer wrong",
            },
        )
        assert loads(wrong_owner.body)["error"]["code"] == -32001
        wrong_session = await cancel(
            "first",
            headers={
                **request.headers,
                "MCP-Session-Id": "wrong-session",
            },
        )
        assert loads(wrong_session.body)["error"]["code"] == -32001
        assert not any(event.is_set() for event in active.values())

        assert (await cancel("first")).status_code == 202
        assert active["first"].is_set()
        assert not active["second"].is_set()
        assert not active["disconnected"].is_set()
        assert await first_consumer

        assert (await cancel("first")).status_code == 202
        assert not active["second"].is_set()

        disconnected_consumer.cancel()
        with pytest.raises(CancelledError):
            await disconnected_consumer
        cancellations = getattr(
            request.app.state,
            "mcp_stream_cancellations",
        )
        assert (
            context.owner,
            context.session_id,
            "disconnected",
        ) not in cancellations

        close = _route_endpoint(path="", method="DELETE")
        assert (
            await close(_Request(app=request.app, headers=request.headers))
        ).status_code == 204
        assert await second_consumer
        assert not hasattr(
            request.app.state,
            "mcp_stream_cancellations",
        )


@pytest.mark.anyio
async def test_session_stream_routes_only_its_reverse_request() -> None:
    _, context = await _initialized_context()
    question = ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=True,
    )
    handler = context.registry.handler(
        session_id=context.session_id,
        owner=context.owner,
        related_request_id="call",
    )
    handled = create_task(
        handler(InputHandlerContext(request=_input_request(question)))
    )

    async def source() -> AsyncIterator[bytes]:
        await handled
        yield b'{"jsonrpc":"2.0","id":"call","result":{"content":[]}}\n'

    merged = mcp_router._merge_mcp_session_outbound(
        source(),
        context,
        related_request_id="call",
    )
    reverse = loads((await anext(merged)).decode())
    assert reverse["method"] == "elicitation/create"
    await context.registry.dispatch_response(
        context.session_id,
        context.owner,
        {
            "jsonrpc": "2.0",
            "id": reverse["id"],
            "result": {
                "action": "accept",
                "content": {"confirm": True},
            },
        },
    )
    terminal = loads((await anext(merged)).decode())
    assert terminal["id"] == "call"
    with pytest.raises(StopAsyncIteration):
        await anext(merged)


@pytest.mark.anyio
async def test_task_handlers_preserve_lifecycle_and_exact_result() -> None:
    request, context = await _initialized_context()
    controller = mcp_router._get_task_controller(cast(object, request))
    assert context.task_requestor is not None
    creation = await controller.create(
        MCPTaskRequest(),
        requestor=context.task_requestor,
        task_id="task",
    )

    listed = await mcp_router._handle_task_message(
        cast(object, request),
        {
            "jsonrpc": "2.0",
            "id": "list",
            "method": "tasks/list",
            "params": {"_meta": {"ignored": True}},
        },
        context,
    )
    assert (
        loads(cast(object, listed).body)["result"]["tasks"][0]["taskId"]
        == "task"
    )
    status = await mcp_router._handle_task_message(
        cast(object, request),
        {
            "jsonrpc": "2.0",
            "id": "get",
            "method": "tasks/get",
            "params": {"taskId": "task"},
        },
        context,
    )
    assert loads(cast(object, status).body)["result"]["status"] == "working"

    result_response = mcp_router._mcp_task_result_response(
        controller,
        task_id="task",
        request_id="result",
        requestor=context.task_requestor,
        session=context,
    )
    chunks = create_task(_response_chunks(result_response))
    await sleep(0)
    assert not chunks.done()
    await creation.handle.transition_input_required()
    await sleep(0)
    assert not chunks.done()
    await creation.handle.transition_working()
    await creation.handle.complete(
        {"content": [{"type": "text", "text": "exact"}]}
    )
    messages = _sse_messages(await chunks)
    assert messages[-1]["id"] == "result"
    exact = messages[-1]["result"]
    assert exact["content"] == [{"type": "text", "text": "exact"}]
    require_related_task_metadata(exact, "task")


@pytest.mark.anyio
async def test_task_result_serializes_lookup_errors() -> None:
    _, context = await _initialized_context(capabilities={})
    assert context.task_requestor is not None
    owner = context.task_requestor
    now = [datetime(2026, 7, 24, tzinfo=UTC)]

    missing = MCPTaskController()
    expired = MCPTaskController(
        MCPTaskPolicy(default_ttl_ms=1, maximum_ttl_ms=1),
        clock=lambda: now[0],
        id_factory=lambda: "expired",
    )
    await expired.create(MCPTaskRequest(), requestor=owner)
    now[0] += timedelta(milliseconds=2)
    unauthorized = MCPTaskController(id_factory=lambda: "private")
    await unauthorized.create(MCPTaskRequest(), requestor=owner)

    cases = (
        (missing, "missing", owner, None),
        (expired, "expired", owner, None),
        (
            unauthorized,
            "private",
            (context.owner, "different-session"),
            "authorization",
        ),
    )
    for controller, task_id, requestor, reason in cases:
        response = mcp_router._mcp_task_result_response(
            controller,
            task_id=task_id,
            request_id=f"result-{task_id}",
            requestor=requestor,
            session=context,
        )
        messages = _sse_messages(await _response_chunks(response))
        error = messages[-1]["error"]
        assert error["code"] == -32602
        if reason is not None:
            assert error["data"]["reason"] == reason


async def _response_chunks(response: object) -> list[bytes]:
    chunks: list[bytes] = []
    async for chunk in getattr(response, "body_iterator"):
        chunks.append(
            chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        )
    return chunks


def _sse_messages(chunks: list[bytes]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                messages.append(loads(line[6:]))
    return messages


@pytest.mark.anyio
async def test_task_status_hook_and_consumer_terminal_paths() -> None:
    controller = MCPTaskController()
    owner = ("owner", "session")
    creation = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="task",
    )
    hook = mcp_router._mcp_task_status_hook(creation.handle)
    await hook(
        mcp_router.MCPFormStatusEvent(
            session_id="session",
            request_id="request",
            status=mcp_router.MCPFormStatus.INPUT_REQUIRED,
            related_task_id="task",
        )
    )
    pending = await controller.get("task", requestor=owner)
    assert pending["status"] == "input_required"
    assert pending["_meta"] == {
        "https://avalan.ai/extensions/task-input/v1": {
            "kind": "request",
            "request_id": "request",
        }
    }
    await hook(
        mcp_router.MCPFormStatusEvent(
            session_id="session",
            request_id="request",
            status=mcp_router.MCPFormStatus.ANSWERED,
            related_task_id="task",
        )
    )
    working = await controller.get("task", requestor=owner)
    assert working["status"] == "working"
    assert working["_meta"] == {
        "https://avalan.ai/extensions/task-input/v1": {
            "kind": "resolution",
            "request_id": "request",
        }
    }

    async def result_stream() -> AsyncIterator[bytes]:
        yield b'{"jsonrpc":"2.0","id":"call","result":{"content":[]}}\n'

    await mcp_router._consume_tool_task(
        result_stream(),
        request_id="call",
        handle=creation.handle,
        cancel_event=Event(),
        logger=getLogger("mcp-task"),
    )
    outcome = await controller.result("task", requestor=owner)
    assert outcome.result is not None
    require_related_task_metadata(outcome.result, "task")

    failed = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="failed",
    )

    async def error_stream() -> AsyncIterator[bytes]:
        yield (
            b'{"jsonrpc":"2.0","id":"failed-call","error":'
            b'{"code":-32042,"message":"exact","data":{"safe":true}}}\n'
        )

    await mcp_router._consume_tool_task(
        error_stream(),
        request_id="failed-call",
        handle=failed.handle,
        cancel_event=Event(),
        logger=getLogger("mcp-task"),
    )
    failed_outcome = await controller.result("failed", requestor=owner)
    assert failed_outcome.error == {
        "code": -32042,
        "message": "exact",
        "data": {"safe": True},
    }

    for task_id, status in (
        ("form-cancelled", mcp_router.MCPFormStatus.CANCELLED),
        ("form-unavailable", mcp_router.MCPFormStatus.UNAVAILABLE),
    ):
        terminal = await controller.create(
            MCPTaskRequest(),
            requestor=owner,
            task_id=task_id,
        )
        terminal_hook = mcp_router._mcp_task_status_hook(terminal.handle)
        await terminal_hook(
            mcp_router.MCPFormStatusEvent(
                session_id="session",
                request_id=f"request-{task_id}",
                status=mcp_router.MCPFormStatus.INPUT_REQUIRED,
                related_task_id=task_id,
            )
        )
        await terminal_hook(
            mcp_router.MCPFormStatusEvent(
                session_id="session",
                request_id=f"request-{task_id}",
                status=status,
                related_task_id=task_id,
            )
        )
        assert (await controller.get(task_id, requestor=owner))[
            "status"
        ] == "working"

    raced = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="cancel-race",
    )
    raced_hook = mcp_router._mcp_task_status_hook(raced.handle)
    await raced_hook(
        mcp_router.MCPFormStatusEvent(
            session_id="session",
            request_id="request-cancel-race",
            status=mcp_router.MCPFormStatus.INPUT_REQUIRED,
            related_task_id="cancel-race",
        )
    )
    await controller.cancel("cancel-race", requestor=owner)
    await raced_hook(
        mcp_router.MCPFormStatusEvent(
            session_id="session",
            request_id="request-cancel-race",
            status=mcp_router.MCPFormStatus.UNAVAILABLE,
            related_task_id="cancel-race",
        )
    )
    assert (await controller.get("cancel-race", requestor=owner))[
        "status"
    ] == "cancelled"


@pytest.mark.anyio
async def test_task_start_is_immediate_and_background_exact() -> None:
    request, context = await _initialized_context()
    captured: dict[str, object] = {}
    started = Event()
    release = Event()

    async def orchestrate(
        *args: object,
        **kwargs: object,
    ) -> tuple[object, str, int]:
        del args
        captured["runtime"] = kwargs["interaction_runtime"]
        started.set()
        await release.wait()
        return object(), str(uuid4()), 1

    async def stream(**kwargs: object) -> AsyncIterator[bytes]:
        request_id = kwargs["request_id"]
        yield (
            f'{{"jsonrpc":"2.0","id":"{request_id}",'
            '"result":{"content":[{"type":"text","text":"done"}]}}\n'
        ).encode()

    with (
        patch.object(mcp_router, "orchestrate", orchestrate),
        patch.object(mcp_router, "_stream_mcp_response", stream),
    ):
        response = await wait_for(
            mcp_router._start_tool_task_response(
                cast(object, request),
                getLogger("mcp-task"),
                cast(Orchestrator, object()),
                "call",
                mcp_router.MCPToolRequest(input_string="start"),
                "progress",
                context,
                MCPTaskRequest(),
            ),
            1,
        )
        task = loads(response.body)["result"]["task"]
        assert task["status"] == "working"
        assert not release.is_set()
        await wait_for(started.wait(), 1)
        assert captured["runtime"] is not None
        controller = mcp_router._get_task_controller(cast(object, request))
        assert context.task_requestor is not None
        assert (
            await controller.get(
                task["taskId"],
                requestor=context.task_requestor,
            )
        )["status"] == "working"
        background = request.app.state.mcp_background_tasks
        assert any(
            not pending.done()
            for pending in background[context.task_requestor]
        )
        release.set()
        outcome = await controller.result(
            task["taskId"],
            requestor=context.task_requestor,
        )
    assert outcome.result is not None
    assert outcome.result["content"] == [{"type": "text", "text": "done"}]


@pytest.mark.anyio
async def test_task_cancel_stops_blocking_background_orchestration() -> None:
    request, context = await _initialized_context(capabilities={})
    started = Event()
    cancelled = Event()
    blocker = Event()

    async def orchestrate(
        *args: object,
        **kwargs: object,
    ) -> tuple[object, str, int]:
        del args, kwargs
        started.set()
        try:
            await blocker.wait()
        finally:
            cancelled.set()
        return object(), str(uuid4()), 1

    with patch.object(mcp_router, "orchestrate", orchestrate):
        response = await wait_for(
            mcp_router._start_tool_task_response(
                cast(object, request),
                getLogger("mcp-task-cancel"),
                cast(Orchestrator, object()),
                "call-cancel",
                mcp_router.MCPToolRequest(input_string="start"),
                "progress",
                context,
                MCPTaskRequest(),
            ),
            1,
        )
        task_id = loads(response.body)["result"]["task"]["taskId"]
        await wait_for(started.wait(), 1)
        cancelled_response = await mcp_router._handle_task_message(
            cast(object, request),
            {
                "jsonrpc": "2.0",
                "id": "cancel",
                "method": "tasks/cancel",
                "params": {"taskId": task_id},
            },
            context,
        )
        assert (
            loads(cast(object, cancelled_response).body)["result"]["status"]
            == "cancelled"
        )
        await wait_for(cancelled.wait(), 1)

    controller = mcp_router._get_task_controller(cast(object, request))
    assert context.task_requestor is not None
    outcome = await controller.result(
        task_id,
        requestor=context.task_requestor,
    )
    assert outcome.error == {
        "code": -32000,
        "message": "Request cancelled",
    }
    await sleep(0)
    background = getattr(request.app.state, "mcp_background_tasks", {})
    assert not background.get(context.task_requestor)


@pytest.mark.anyio
async def test_close_mcp_state_cancels_owned_background_work() -> None:
    request, context = await _initialized_context()
    blocker = Event()
    cancel_event = Event()

    async def wait_forever() -> None:
        await blocker.wait()

    mcp_router._register_mcp_cancellation(
        cast(object, request),
        context,
        "stream",
        cancel_event,
    )
    task = create_task(wait_forever())
    assert context.task_requestor is not None
    mcp_router._track_mcp_background_task(
        cast(object, request),
        task,
        context.task_requestor,
    )
    await mcp_router.close_mcp_state(request.app)
    assert task.cancelled()
    assert cancel_event.is_set()
    assert not hasattr(request.app.state, "mcp_form_session_registry")
    assert not hasattr(request.app.state, "mcp_stream_cancellations")
    assert not hasattr(request.app.state, "mcp_task_controller")


@pytest.mark.anyio
async def test_session_rejects_unavailable_and_anonymous_tasks() -> None:
    logger = getLogger("mcp-session-negative")
    orchestrator = MagicMock(spec=Orchestrator)
    orchestrator.tool.list_tools.return_value = []
    rpc = _route_endpoint(path="", method="POST")

    unavailable = await rpc(
        _Request(
            body=dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": "call"},
                }
            ).encode()
        ),
        logger,
        orchestrator,
    )
    assert loads(unavailable.body)["error"]["code"] == -32001
    with pytest.raises(MCPFormSessionError):
        await mcp_router._mcp_session_context(
            cast(object, _Request()),
            required=True,
        )

    anonymous = _Request()
    _configure(anonymous, _AnonymousResolver())
    initialized = await mcp_router._initialize_mcp_session(
        cast(object, anonymous),
        logger,
        orchestrator,
        cast(mcp_router.JSONObject, _initialize_message()),
    )
    anonymous.headers.update(
        {
            "MCP-Session-Id": initialized.headers["mcp-session-id"],
            "MCP-Protocol-Version": "2025-11-25",
        }
    )
    context = await mcp_router._mcp_session_context(cast(object, anonymous))
    assert context is not None
    assert context.task_requestor is None
    with pytest.raises(mcp_router.MCPTaskProtocolError):
        await mcp_router._handle_task_message(
            cast(object, anonymous),
            {
                "jsonrpc": "2.0",
                "id": "list",
                "method": "tasks/list",
                "params": {},
            },
            context,
        )

    failure = MCPFormSessionError(
        MCPFormErrorCode.CAPACITY,
        -32001,
        "MCP form capacity is unavailable.",
    )
    authenticated = _Request(headers={"Authorization": "Bearer owner"})
    _configure(authenticated)
    with patch.object(
        mcp_router.MCPFormSessionRegistry,
        "initialize",
        AsyncMock(side_effect=failure),
    ):
        rejected = await mcp_router._initialize_mcp_session(
            cast(object, authenticated),
            logger,
            orchestrator,
            cast(mcp_router.JSONObject, _initialize_message()),
        )
    assert loads(rejected.body)["error"] == {
        "code": -32001,
        "message": "MCP form capacity is unavailable.",
        "data": {"code": "avalan.input.unavailable"},
    }


@pytest.mark.anyio
async def test_form_capable_tool_stream_uses_session_projection() -> None:
    request, context = await _initialized_context()
    chat_request = mcp_router.ChatCompletionRequest(
        model="test",
        messages=[
            mcp_router.ChatMessage(
                role=mcp_router.MessageRole.USER,
                content="continue",
            )
        ],
        stream=True,
    )

    async def stream(**kwargs: object) -> AsyncIterator[bytes]:
        assert kwargs["request_id"] == "call"
        yield b'{"jsonrpc":"2.0","id":"call","result":{"content":[]}}\n'

    with (
        patch.object(
            mcp_router,
            "orchestrate",
            AsyncMock(return_value=(object(), uuid4(), 1)),
        ),
        patch.object(
            mcp_router,
            "_build_chat_request",
            return_value=chat_request,
        ),
        patch.object(mcp_router, "_stream_mcp_response", side_effect=stream),
    ):
        response = await mcp_router._start_tool_streaming_response(
            cast(object, request),
            getLogger("mcp-session-stream"),
            cast(Orchestrator, object()),
            "call",
            mcp_router.MCPToolRequest(input_string="continue"),
            "progress",
            session=context,
        )
        messages = _sse_messages(await _response_chunks(response))
    assert messages[-1]["id"] == "call"


@pytest.mark.anyio
async def test_task_execution_cancellation_and_failure_edges() -> None:
    request = _Request()
    controller = MCPTaskController()
    owner = ("owner", "session")
    cancelled_creation = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="cancelled",
    )
    operation_started = Event()
    operation_cancelled = Event()

    async def blocking_execution(*args: object, **kwargs: object) -> None:
        del args, kwargs
        operation_started.set()
        try:
            await Event().wait()
        finally:
            operation_cancelled.set()

    with patch.object(
        mcp_router,
        "_execute_tool_task",
        blocking_execution,
    ):
        running = create_task(
            mcp_router._run_tool_task(
                cast(object, request),
                getLogger("mcp-task-cancellation"),
                cast(Orchestrator, object()),
                request_id="call",
                chat_request=cast(
                    mcp_router.ChatCompletionRequest,
                    MagicMock(),
                ),
                progress_token="progress",
                interaction_runtime=None,
                handle=cancelled_creation.handle,
                cancel_event=Event(),
            )
        )
        await wait_for(operation_started.wait(), 1)
        await controller.cancel("cancelled", requestor=owner)
        await wait_for(running, 1)
    assert operation_cancelled.is_set()

    failed_creation = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="failed-execution",
    )
    with patch.object(
        mcp_router,
        "orchestrate",
        AsyncMock(side_effect=RuntimeError("provider failed")),
    ):
        await mcp_router._execute_tool_task(
            cast(object, request),
            getLogger("mcp-task-failure"),
            cast(Orchestrator, object()),
            request_id="call",
            chat_request=cast(
                mcp_router.ChatCompletionRequest,
                MagicMock(),
            ),
            progress_token="progress",
            interaction_runtime=None,
            handle=failed_creation.handle,
            cancel_event=Event(),
        )
    assert (
        await controller.result("failed-execution", requestor=owner)
    ).error == {
        "code": -32603,
        "message": "An internal server error occurred.",
    }

    invalid_handle = MagicMock()
    invalid_handle.transition_working = AsyncMock(
        side_effect=mcp_router.MCPTaskProtocolError(
            code=-32001,
            message="Unexpected task transition.",
            data={"policy": "avalan", "reason": "unexpected"},
        )
    )
    hook = mcp_router._mcp_task_status_hook(invalid_handle)
    with pytest.raises(mcp_router.MCPTaskProtocolError):
        await hook(
            mcp_router.MCPFormStatusEvent(
                session_id="session",
                request_id="request",
                status=mcp_router.MCPFormStatus.ANSWERED,
            )
        )


@pytest.mark.anyio
async def test_task_consumer_failure_and_iterator_edges() -> None:
    controller = MCPTaskController()
    owner = ("owner", "session")

    async def stream(payload: bytes) -> AsyncIterator[bytes]:
        yield payload

    cases = (
        ("missing", b"\n{}\n", "ended without an operation result"),
        (
            "invalid",
            b'{"jsonrpc":"2.0","id":"invalid-call","result":1}\n',
            "returned an invalid operation result",
        ),
        ("malformed", b"not-json\n", "internal server error"),
    )
    for task_id, payload, message_fragment in cases:
        creation = await controller.create(
            MCPTaskRequest(),
            requestor=owner,
            task_id=task_id,
        )
        await mcp_router._consume_tool_task(
            stream(payload),
            request_id=f"{task_id}-call",
            handle=creation.handle,
            cancel_event=Event(),
            logger=getLogger("mcp-task-consumer"),
        )
        outcome = await controller.result(task_id, requestor=owner)
        assert outcome.error is not None
        assert message_fragment in cast(str, outcome.error["message"]).lower()

    cancelled = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="consumer-cancelled",
    )
    cancel_event = Event()

    async def cancelled_stream() -> AsyncIterator[bytes]:
        if False:
            yield b""
        raise CancelledError

    with pytest.raises(CancelledError):
        await mcp_router._consume_tool_task(
            cancelled_stream(),
            request_id="cancelled-call",
            handle=cancelled.handle,
            cancel_event=cancel_event,
            logger=getLogger("mcp-task-consumer"),
        )
    assert cancel_event.is_set()

    class OneChunk:
        def __init__(self) -> None:
            self.chunk: bytes | None = (
                b'{"jsonrpc":"2.0","id":"plain-call","result":{}}\n'
            )

        def __aiter__(self) -> "OneChunk":
            return self

        async def __anext__(self) -> bytes:
            if self.chunk is None:
                raise StopAsyncIteration
            chunk = self.chunk
            self.chunk = None
            return chunk

    plain = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
        task_id="plain",
    )
    await mcp_router._consume_tool_task(
        cast(AsyncIterator[bytes], OneChunk()),
        request_id="plain-call",
        handle=plain.handle,
        cancel_event=Event(),
        logger=getLogger("mcp-task-consumer"),
    )
    plain_outcome = await controller.result("plain", requestor=owner)
    assert plain_outcome.result is not None
    require_related_task_metadata(plain_outcome.result, "plain")


@pytest.mark.anyio
async def test_task_message_negative_and_error_outcome_paths() -> None:
    request, context = await _initialized_context(capabilities={})
    controller = mcp_router._get_task_controller(cast(object, request))
    assert context.task_requestor is not None

    with pytest.raises(mcp_router.MCPTaskProtocolError):
        await mcp_router._handle_task_message(
            cast(object, request),
            {
                "jsonrpc": "2.0",
                "id": "invalid",
                "method": "tasks/get",
                "params": {},
            },
            context,
        )

    creation = await controller.create(
        MCPTaskRequest(),
        requestor=context.task_requestor,
        task_id="result",
    )
    response = await mcp_router._handle_task_message(
        cast(object, request),
        {
            "jsonrpc": "2.0",
            "id": "result-call",
            "method": "tasks/result",
            "params": {"taskId": "result"},
        },
        context,
    )
    chunks = create_task(_response_chunks(response))
    await creation.handle.fail({"code": -32042, "message": "Exact failure."})
    assert _sse_messages(await chunks)[-1]["error"]["code"] == -32042

    with pytest.raises(mcp_router.MCPTaskProtocolError):
        await mcp_router._handle_task_message(
            cast(object, request),
            {
                "jsonrpc": "2.0",
                "id": "unsupported",
                "method": "tasks/unknown",
                "params": {"taskId": "result"},
            },
            context,
        )

    error_message = mcp_router._mcp_task_outcome_message(
        "error",
        mcp_router.MCPTaskOutcome.failure(
            {
                "code": -32043,
                "message": "Preserved.",
            }
        ),
    )
    assert error_message["error"] == {
        "code": -32043,
        "message": "Preserved.",
    }


@pytest.mark.anyio
async def test_background_and_cancellation_registry_edges() -> None:
    owner = ("owner", "session")
    watched = Event()

    async def cancellation_messages() -> AsyncIterator[dict[str, object]]:
        yield {"jsonrpc": "2.0", "id": "ping", "method": "ping"}
        yield {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": "watched"},
        }

    await mcp_router._watch_for_cancellation(
        cast(AsyncIterator[mcp_router.JSONObject], cancellation_messages()),
        watched,
        getLogger("mcp-cancellation-watch"),
        request_id="watched",
    )
    assert watched.is_set()

    await mcp_router._cancel_mcp_background_tasks(object(), owner)
    await mcp_router._cancel_mcp_background_tasks(
        SimpleNamespace(state=SimpleNamespace(mcp_background_tasks=())),
        owner,
    )
    empty = SimpleNamespace(state=SimpleNamespace(mcp_background_tasks={}))
    await mcp_router._cancel_mcp_background_tasks(empty, owner)

    release_first = Event()
    release_second = Event()
    tracked_request = _Request()
    first = create_task(release_first.wait())
    second = create_task(release_second.wait())
    mcp_router._track_mcp_background_task(
        cast(object, tracked_request), first, owner
    )
    mcp_router._track_mcp_background_task(
        cast(object, tracked_request), second, owner
    )
    release_first.set()
    await first
    await sleep(0)
    assert tracked_request.app.state.mcp_background_tasks[owner] == {second}
    release_second.set()
    await second
    await sleep(0)
    assert owner not in tracked_request.app.state.mcp_background_tasks

    orphan_release = Event()
    orphan = create_task(orphan_release.wait())
    mcp_router._track_mcp_background_task(
        cast(object, tracked_request), orphan, owner
    )
    tracked_request.app.state.mcp_background_tasks.pop(owner)
    orphan_release.set()
    await orphan
    await sleep(0)

    cancel_blocker = Event()
    pending = {
        create_task(cancel_blocker.wait()),
        create_task(cancel_blocker.wait()),
    }
    cancel_app = SimpleNamespace(
        state=SimpleNamespace(mcp_background_tasks={owner: pending})
    )
    await mcp_router._cancel_mcp_background_tasks(cancel_app, owner)
    assert all(task.cancelled() for task in pending)

    assert mcp_router._mcp_cancellations(object()) is None
    request, context = await _initialized_context(capabilities={})
    registered = Event()
    mcp_router._register_mcp_cancellation(
        cast(object, request),
        context,
        "registered",
        registered,
    )
    mcp_router._discard_mcp_cancellation(
        cast(object, request),
        context,
        "registered",
        Event(),
    )
    assert hasattr(request.app.state, "mcp_stream_cancellations")
    mcp_router._discard_mcp_cancellation(
        cast(object, request),
        context,
        "registered",
        registered,
    )
    assert not hasattr(request.app.state, "mcp_stream_cancellations")

    first_event = Event()
    second_event = Event()
    other_event = Event()
    other_context = mcp_router._MCPSessionContext(
        session_id="other-session",
        owner=context.owner,
        actor=context.actor,
        service=context.service,
        registry=context.registry,
        negotiation=context.negotiation,
    )
    for session, request_id, event in (
        (context, "first", first_event),
        (context, "second", second_event),
        (other_context, "other", other_event),
    ):
        mcp_router._register_mcp_cancellation(
            cast(object, request),
            session,
            request_id,
            event,
        )
    mcp_router._cancel_mcp_session_requests(request.app, context)
    assert first_event.is_set() and second_event.is_set()
    assert not other_event.is_set()
    assert hasattr(request.app.state, "mcp_stream_cancellations")
    mcp_router._cancel_mcp_session_requests(request.app, other_context)
    assert other_event.is_set()
    assert not hasattr(request.app.state, "mcp_stream_cancellations")

    for message in (
        {
            "jsonrpc": "2.0",
            "id": "invalid",
            "method": "notifications/cancelled",
            "params": {"requestId": "call"},
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": False},
        },
    ):
        with pytest.raises(HTTPException):
            mcp_router._handle_cancelled_notification(
                cast(object, request),
                getLogger("mcp-cancellation-negative"),
                cast(mcp_router.JSONObject, message),
                context,
            )

    await mcp_router.close_mcp_state(object())
    await mcp_router.close_mcp_state(SimpleNamespace(state=SimpleNamespace()))


@pytest.mark.anyio
async def test_outbound_timeout_plain_iterator_and_related_task() -> None:
    _, context = await _initialized_context()
    blocker = Event()

    class Registry:
        def __init__(self) -> None:
            self.calls = 0

        async def next_outbound(
            self,
            *args: object,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            self.calls += 1
            if self.calls == 1:
                return None
            await blocker.wait()
            return None

    class OneChunk:
        def __init__(self) -> None:
            self.chunk: bytes | None = b'{"jsonrpc":"2.0","id":"call"}\n'

        def __aiter__(self) -> "OneChunk":
            return self

        async def __anext__(self) -> bytes:
            if self.chunk is None:
                raise StopAsyncIteration
            await sleep(0)
            chunk = self.chunk
            self.chunk = None
            return chunk

    registry = Registry()
    timeout_context = mcp_router._MCPSessionContext(
        session_id=context.session_id,
        owner=context.owner,
        actor=context.actor,
        service=context.service,
        registry=cast(mcp_router.MCPFormSessionRegistry, registry),
        negotiation=context.negotiation,
    )
    merged = mcp_router._merge_mcp_session_outbound(
        cast(AsyncIterator[bytes], OneChunk()),
        timeout_context,
        related_request_id="call",
    )
    assert await anext(merged) == b'{"jsonrpc":"2.0","id":"call"}\n'
    with pytest.raises(StopAsyncIteration):
        await anext(merged)
    assert registry.calls >= 2

    outbound = mcp_router.MCPFormElicitationOutbound(
        session_id=context.session_id,
        jsonrpc_id="elicitation",
        related_request_id="call",
        canonical_request_id="canonical",
        params={"message": "Continue?", "requestedSchema": {}},
        related_task_id="task",
    )
    message = mcp_router._mcp_outbound_message(outbound)
    require_related_task_metadata(
        cast(dict[str, object], message["params"]),
        "task",
    )
