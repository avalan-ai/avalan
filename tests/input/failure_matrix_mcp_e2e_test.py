"""Exercise public MCP cells in the structured-input failure matrix."""

from asyncio import (
    Event,
    create_task,
    run,
    sleep,
    wait_for,
)
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from json import dumps, loads
from logging import getLogger
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, Request, Response
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.session import ElicitationFnT
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

sys_path.append(str(Path(__file__).parent))

import broker_contract_test as broker_contract  # noqa: E402
import mcp_contract_test as contract  # noqa: E402

from avalan.agent.execution import (  # noqa: E402
    AgentExecution,
    AttachedInteractionRuntime,
    UuidExecutionIdFactory,
)
from avalan.entities import ToolCallContext  # noqa: E402
from avalan.interaction import (  # noqa: E402
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InteractionActor,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionExecutionScope,
    InteractionRequestResult,
    QuestionId,
    RequestState,
    SupersedeInteractionScopeCommand,
    TextAnswer,
)
from avalan.server.interaction import (  # noqa: E402
    ServerInteractionConfiguration,
    ServerInteractionService,
)
from avalan.server.mcp_session import (  # noqa: E402
    MCPFormSessionRegistry,
)
from avalan.server.routers import mcp as mcp_router  # noqa: E402
from avalan.tool.mcp import McpCallTool  # noqa: E402

_EVIDENCE_PROPERTY = "failure_matrix_evidence"
_FORM_SURFACE = "mcp-inbound-elicitation-form"
_TASK_SURFACE = "mcp-inbound-task"
_DOWNSTREAM_SURFACE = "mcp-downstream-elicitation"
_CALL_TOOL_RESULT = "mcp.call_tool_result.v1"
_RecordProperty = Callable[[str, object], None]


@dataclass(frozen=True, slots=True)
class _Observation:
    """Hold evidence obtained from one public boundary invocation."""

    transition: tuple[str, str]
    public_result_id: str
    public_result: Mapping[str, object]
    status: tuple[str, str]


def _record(
    record_property: _RecordProperty,
    condition_id: str,
    surface_id: str,
    observation: _Observation,
) -> None:
    evidence = [
        {
            "condition_id": condition_id,
            "surface_id": surface_id,
            "transition_from": observation.transition[0],
            "transition_to": observation.transition[1],
            "public_result_id": observation.public_result_id,
            "public_result": observation.public_result,
            "status_key": observation.status[0],
            "status_value": observation.status[1],
            "provider_call_count": 0,
            "domain_side_effect_count": 0,
        }
    ]
    record_property(
        _EVIDENCE_PROPERTY,
        dumps(evidence, separators=(",", ":"), sort_keys=True),
    )


def _model_dict(value: BaseModel) -> dict[str, object]:
    projected = value.model_dump(
        by_alias=True,
        mode="json",
        exclude_none=True,
    )
    assert isinstance(projected, dict)
    return cast(dict[str, object], projected)


@dataclass(frozen=True, slots=True)
class _PostedResponse:
    """Capture one response POST and its public HTTP outcome."""

    body: dict[str, object]
    headers: dict[str, str]
    response_body: dict[str, object] | None
    status_code: int


class _TransportCapture:
    """Capture public JSON responses without consuming SSE bodies."""

    def __init__(self) -> None:
        self.callback_posts: list[_PostedResponse] = []
        self.method_responses: list[tuple[str, dict[str, object]]] = []
        self.callback_posted = Event()
        self.callback_error = Event()

    async def response_hook(self, response: Response) -> None:
        request_body = self._request_body(response.request)
        response_body: dict[str, object] | None = None
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            await response.aread()
            decoded = loads(response.content)
            if isinstance(decoded, dict):
                response_body = cast(dict[str, object], decoded)
        method = request_body.get("method")
        if isinstance(method, str) and response_body is not None:
            self.method_responses.append((method, response_body))
        if (
            method is None
            and "id" in request_body
            and ("result" in request_body or "error" in request_body)
        ):
            self.callback_posts.append(
                _PostedResponse(
                    body=request_body,
                    headers=dict(response.request.headers),
                    response_body=response_body,
                    status_code=response.status_code,
                )
            )
            self.callback_posted.set()
            if response_body is not None and "error" in response_body:
                self.callback_error.set()

    def response_for(self, method: str) -> dict[str, object]:
        return next(
            body
            for candidate, body in reversed(self.method_responses)
            if candidate == method
        )

    @staticmethod
    def _request_body(request: Request) -> dict[str, object]:
        try:
            decoded = loads(request.content)
        except (TypeError, ValueError):
            return {}
        return (
            cast(dict[str, object], decoded)
            if isinstance(decoded, dict)
            else {}
        )


def _callback_meta(
    params: mcp_types.ElicitRequestFormParams,
) -> dict[str, object] | None:
    if params.meta is None:
        return None
    return cast(
        dict[str, object],
        params.meta.model_dump(mode="json", exclude_none=True),
    )


async def _raw_response_post(
    client: AsyncClient,
    url: str,
    posted: _PostedResponse,
    body: Mapping[str, object],
) -> dict[str, object] | None:
    headers = {
        name: value
        for name, value in posted.headers.items()
        if name.lower()
        in {
            "accept",
            "authorization",
            "content-type",
            "mcp-protocol-version",
            "mcp-session-id",
        }
    }
    response = await client.post(url, headers=headers, json=dict(body))
    if response.status_code == 202:
        return None
    decoded = response.json()
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


async def _task_create(
    session: ClientSession,
    scenario: str,
) -> mcp_types.CreateTaskResult:
    return await session.send_request(
        mcp_types.ClientRequest(
            mcp_types.CallToolRequest(
                params=mcp_types.CallToolRequestParams(
                    name="run",
                    arguments={"input_string": scenario},
                    task=mcp_types.TaskMetadata(ttl=60_000),
                )
            )
        ),
        mcp_types.CreateTaskResult,
    )


async def _task_get(
    session: ClientSession,
    task_id: str,
) -> mcp_types.GetTaskResult:
    return await session.send_request(
        mcp_types.ClientRequest(
            mcp_types.GetTaskRequest(
                params=mcp_types.GetTaskRequestParams(taskId=task_id)
            )
        ),
        mcp_types.GetTaskResult,
    )


async def _task_cancel(
    session: ClientSession,
    task_id: str,
) -> mcp_types.CancelTaskResult:
    return await session.send_request(
        mcp_types.ClientRequest(
            mcp_types.CancelTaskRequest(
                params=mcp_types.CancelTaskRequestParams(taskId=task_id)
            )
        ),
        mcp_types.CancelTaskResult,
    )


async def _task_result(
    session: ClientSession,
    task_id: str,
) -> mcp_types.GetTaskPayloadResult:
    return await session.send_request(
        mcp_types.ClientRequest(
            mcp_types.GetTaskPayloadRequest(
                params=mcp_types.GetTaskPayloadRequestParams(taskId=task_id)
            )
        ),
        mcp_types.GetTaskPayloadResult,
    )


async def _wait_task_status(
    session: ClientSession,
    task_id: str,
    expected: str,
) -> mcp_types.GetTaskResult:
    for _ in range(200):
        task = await _task_get(session, task_id)
        if task.status == expected:
            return task
        await sleep(0)
    raise AssertionError(f"MCP task did not reach {expected}")


def _answered_state(outcome: InputHandlerOutcome) -> str:
    assert isinstance(outcome, InputHandlerResolution)
    assert isinstance(outcome.resolution, AnsweredResolution)
    return RequestState.ANSWERED.value


async def _inbound_condition(condition_id: str) -> _Observation:
    boundary = contract._InboundBoundary()
    orchestrator = contract._InboundOrchestrator()
    app = FastAPI()
    app.state.logger = getLogger("mcp-matrix-public")
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

    async def orchestrate(
        *args: object,
        **kwargs: object,
    ) -> tuple[contract._InboundResponse, UUID, int]:
        del args
        runtime = cast(
            AttachedInteractionRuntime | None,
            kwargs["interaction_runtime"],
        )
        return (
            contract._InboundResponse(runtime, orchestrator),
            uuid4(),
            1,
        )

    scenarios = {
        "INPUT-F-01": "incapable",
        "INPUT-F-04": "confirmation",
        "INPUT-F-05": "selection",
        "INPUT-F-06": "accept",
        "INPUT-F-07": "confirmation",
        "INPUT-F-08": "confirmation",
        "INPUT-F-10": "accept",
        "INPUT-F-12": "accept",
        "INPUT-F-14": "accept",
        "INPUT-F-15": "ordinary",
    }
    scenario = scenarios[condition_id]
    orchestrator.scenario = scenario
    if condition_id in {"INPUT-F-07", "INPUT-F-14"}:
        orchestrator.after_input_gate = Event()
    invalid_content: dict[str, object] | None = {
        "INPUT-F-04": {"confirm": "yes"},
        "INPUT-F-05": {"choice": "unknown"},
        "INPUT-F-06": {},
    }.get(condition_id)
    valid_content: dict[str, object] = {
        "INPUT-F-04": {"confirm": True},
        "INPUT-F-05": {"choice": "alpha"},
        "INPUT-F-06": {"name": "corrected"},
        "INPUT-F-07": {"confirm": True},
        "INPUT-F-08": {"confirm": True},
        "INPUT-F-10": {"name": "Ada"},
        "INPUT-F-12": {"name": "Ada"},
        "INPUT-F-14": {"name": "Ada"},
    }.get(condition_id, {"name": "Ada"})

    async def elicit(
        _context: object,
        params: object,
    ) -> mcp_types.ElicitResult:
        assert isinstance(params, mcp_types.ElicitRequestFormParams)
        return mcp_types.ElicitResult.model_validate(
            {
                "_meta": _callback_meta(params),
                "action": "accept",
                "content": (
                    cast(dict[str, Any], invalid_content)
                    if invalid_content is not None
                    else cast(dict[str, Any], valid_content)
                ),
            }
        )

    capture = _TransportCapture()
    with (
        patch.object(
            mcp_router, "Orchestrator", contract._InboundOrchestrator
        ),
        patch.object(mcp_router, "resolve_model_id", return_value="gpt"),
        patch.object(mcp_router, "orchestrate", orchestrate),
    ):
        async with contract._loopback_server(
            app,
            lifespan="off",
        ) as base_url:
            url = f"{base_url}/mcp"
            async with AsyncClient(
                headers={"Authorization": "Bearer mcp-owner"},
                event_hooks={"response": [capture.response_hook]},
                timeout=5,
            ) as client:
                async with streamable_http_client(
                    url,
                    http_client=client,
                ) as (read_stream, write_stream, _session_id):
                    callback = (
                        None
                        if condition_id in {"INPUT-F-01", "INPUT-F-15"}
                        else cast(ElicitationFnT, elicit)
                    )
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=5),
                        elicitation_callback=callback,
                    ) as session:
                        initialized = await session.initialize()
                        assert (
                            initialized.protocolVersion
                            == mcp_types.LATEST_PROTOCOL_VERSION
                        )
                        observation = await _invoke_inbound(
                            condition_id,
                            scenario,
                            session,
                            client,
                            url,
                            capture,
                            orchestrator,
                            invalid_content,
                            valid_content,
                        )
    await mcp_router.close_mcp_state(app)
    return observation


async def _invoke_inbound(
    condition_id: str,
    scenario: str,
    session: ClientSession,
    client: AsyncClient,
    url: str,
    capture: _TransportCapture,
    orchestrator: contract._InboundOrchestrator,
    invalid_content: Mapping[str, object] | None,
    valid_content: Mapping[str, object],
) -> _Observation:
    if condition_id in {"INPUT-F-01", "INPUT-F-15"}:
        result = await session.call_tool(
            "run",
            {"input_string": scenario},
            read_timeout_seconds=timedelta(seconds=5),
        )
        public_result = _model_dict(result)
        text = cast(Any, result.content[0]).text
        expected = (
            "unavailable" if condition_id == "INPUT-F-01" else "ordinary"
        )
        assert text == expected
        return _Observation(
            transition=(
                ("created", "unavailable")
                if condition_id == "INPUT-F-01"
                else ("running", "running")
            ),
            public_result_id=_CALL_TOOL_RESULT,
            public_result=public_result,
            status=("result", expected),
        )

    if condition_id in {"INPUT-F-04", "INPUT-F-05", "INPUT-F-06"}:
        assert invalid_content is not None
        call = create_task(
            session.call_tool(
                "run",
                {"input_string": scenario},
                read_timeout_seconds=timedelta(seconds=5),
            )
        )
        await wait_for(capture.callback_error.wait(), timeout=5)
        assert not call.done()
        posted = capture.callback_posts[0]
        assert posted.response_body is not None
        error = cast(dict[str, object], posted.response_body)
        assert cast(dict[str, object], error["error"])["code"] == -32602
        corrected = {
            "jsonrpc": "2.0",
            "id": posted.body["id"],
            "result": {
                "action": "accept",
                "content": dict(valid_content),
            },
        }
        assert await _raw_response_post(client, url, posted, corrected) is None
        await call
        request = orchestrator.requests[-1]
        assert request.state is RequestState.PENDING
        return _Observation(
            transition=(request.state.value, request.state.value),
            public_result_id="mcp.invalid_params_error.v1",
            public_result=error,
            status=("jsonrpc_error", "-32602"),
        )

    if condition_id == "INPUT-F-08":
        await session.call_tool(
            "run",
            {"input_string": scenario},
            read_timeout_seconds=timedelta(seconds=5),
        )
        posted = capture.callback_posts[0]
        original = cast(dict[str, object], posted.body["result"])
        conflict_result = dict(original)
        conflict_result["content"] = {"confirm": False}
        conflict = await _raw_response_post(
            client,
            url,
            posted,
            {
                "jsonrpc": "2.0",
                "id": posted.body["id"],
                "result": conflict_result,
            },
        )
        assert conflict is not None
        assert cast(dict[str, object], conflict["error"])["code"] == -32009
        state = _answered_state(orchestrator.outcomes[-1])
        return _Observation(
            transition=(state, state),
            public_result_id="mcp.conflict_error.v1",
            public_result=conflict,
            status=("jsonrpc_error", "-32009"),
        )

    creation = await _task_create(session, scenario)
    task_id = creation.task.taskId
    pending = await _wait_task_status(
        session,
        task_id,
        "input_required",
    )
    assert pending.status == "input_required"

    if condition_id == "INPUT-F-10":
        cancelled = await _task_cancel(session, task_id)
        assert cancelled.status == "cancelled"
        return _Observation(
            transition=(RequestState.PENDING.value, cancelled.status),
            public_result_id="mcp.task_cancelled.v1",
            public_result=capture.response_for("tasks/cancel"),
            status=("task_status", cancelled.status),
        )

    if condition_id == "INPUT-F-12":
        public_result = capture.response_for("tasks/get")
        await _task_cancel(session, task_id)
        return _Observation(
            transition=(
                RequestState.PENDING.value,
                RequestState.PENDING.value,
            ),
            public_result_id="mcp.task_input_required.v1",
            public_result=public_result,
            status=("task_status", pending.status),
        )

    result_task = create_task(_task_result(session, task_id))
    await wait_for(capture.callback_posted.wait(), timeout=5)
    for _ in range(200):
        if orchestrator.outcomes:
            break
        await sleep(0.01)
    assert orchestrator.outcomes
    working = await _wait_task_status(session, task_id, "working")
    public_result = capture.response_for("tasks/get")

    if condition_id == "INPUT-F-07":
        posted = capture.callback_posts[0]
        assert (
            await _raw_response_post(client, url, posted, posted.body) is None
        )
        state = _answered_state(orchestrator.outcomes[-1])
        observation = _Observation(
            transition=(state, state),
            public_result_id="mcp.task_working.v1",
            public_result=public_result,
            status=("task_status", working.status),
        )
    else:
        assert condition_id == "INPUT-F-14"
        request = orchestrator.requests[-1]
        state = _answered_state(orchestrator.outcomes[-1])
        observation = _Observation(
            transition=(request.state.value, state),
            public_result_id="mcp.task_working.v1",
            public_result=public_result,
            status=("task_status", working.status),
        )
    assert orchestrator.after_input_gate is not None
    orchestrator.after_input_gate.set()
    await wait_for(result_task, timeout=5)
    return observation


class _BrokerProbe:
    """Retain results while delegating every operation to the real broker."""

    def __init__(self, broker: AsyncInteractionBroker) -> None:
        self._broker = broker
        self.policy = broker.policy
        self.results: list[InteractionRequestResult] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        result = await self._broker.request(request)
        self.results.append(result)
        return result

    async def cancel_scope(self, command: object) -> InteractionBrokerResult:
        return await self._broker.cancel_scope(cast(Any, command))

    async def supersede(self, command: object) -> InteractionBrokerResult:
        return await self._broker.supersede(cast(Any, command))


class _DownstreamHandler:
    """Drive one typed broker outcome for a public reverse elicitation."""

    def __init__(
        self,
        condition_id: str,
        clock: broker_contract._Clock,
    ) -> None:
        self.condition_id = condition_id
        self.clock = clock
        self.started = Event()
        self.contexts: list[InputHandlerContext] = []
        self.blocked = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        self.contexts.append(context)
        self.started.set()
        if self.condition_id == "INPUT-F-14":
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=context.request.request_id,
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=self.clock.wall_time,
                    answers=(
                        TextAnswer(
                            question_id=QuestionId("name"),
                            provenance=AnswerProvenance.HUMAN,
                            value="Ada",
                        ),
                    ),
                )
            )
        if self.condition_id == "INPUT-F-10":
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_CANCELLED
            )
        if self.condition_id == "INPUT-F-01":
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_UNAVAILABLE
            )
        await self.blocked.wait()
        raise AssertionError("terminal broker transition did not stop handler")


class _NameForm(BaseModel):
    """Describe the real downstream MCP form."""

    name: str = Field(title="Name", min_length=1, max_length=20)


async def _downstream_condition(condition_id: str) -> _Observation:
    harness = await broker_contract._harness()
    broker = harness.broker
    clock = harness.clock
    probe = _BrokerProbe(broker)
    handler = _DownstreamHandler(condition_id, clock)
    origin = contract._origin(task_id=f"downstream-{condition_id}")
    execution_ids = UuidExecutionIdFactory()
    runtime = AttachedInteractionRuntime(
        broker=cast(Any, probe),
        actor=InteractionActor(principal=origin.principal),
        handler=handler,
        policy=broker.policy,
        id_factory=execution_ids,
        run_id=origin.run_id,
        task_id=origin.task_id,
        branch_id=origin.branch_id,
    )
    execution = AgentExecution(
        origin=origin,
        id_factory=execution_ids,
        initial_messages=(),
        interaction_runtime=runtime,
    )
    branch_broker = execution.interaction_broker
    assert branch_broker is not None
    context = ToolCallContext(
        agent_id=UUID(str(origin.agent_id)),
        execution=execution,
        execution_origin=origin,
        interaction_broker=branch_broker,
    )
    server = FastMCP(
        "mcp-matrix-downstream",
        log_level="CRITICAL",
    )

    @server.tool()
    async def ask(ctx: Context) -> dict[str, object]:
        elicited = await ctx.elicit(
            message="Who should be greeted?",
            schema=_NameForm,
        )
        data = getattr(elicited, "data", None)
        content = data.model_dump(mode="json") if data is not None else None
        return {"action": elicited.action, "content": content}

    app = server.streamable_http_app()
    tool = McpCallTool(call_params={"timeout": 5})
    try:
        async with contract._loopback_server(
            app,
            lifespan="on",
        ) as base_url:
            call = create_task(
                tool(
                    uri=f"{base_url}/mcp",
                    name="ask",
                    arguments={},
                    context=context,
                )
            )
            if condition_id == "INPUT-F-09":
                await wait_for(handler.started.wait(), timeout=5)
                await broker_contract._wait_until(
                    lambda: bool(clock.wait_calls)
                )
                clock.advance(max(clock.wait_calls) - clock.monotonic_seconds)
            elif condition_id == "INPUT-F-11":
                await wait_for(handler.started.wait(), timeout=5)
                await branch_broker.supersede(
                    SupersedeInteractionScopeCommand(
                        actor=runtime.actor,
                        scope=InteractionExecutionScope(
                            run_id=origin.run_id,
                            branch_id=origin.branch_id,
                        ),
                        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    )
                )
            public_result = await wait_for(call, timeout=5)
    finally:
        await broker.aclose()

    assert len(probe.results) == 1
    result = probe.results[0]
    assert result.delivery is not None
    terminal = result.delivery.record.request
    expected_state = {
        "INPUT-F-01": RequestState.UNAVAILABLE,
        "INPUT-F-09": RequestState.EXPIRED,
        "INPUT-F-10": RequestState.CANCELLED,
        "INPUT-F-11": RequestState.SUPERSEDED,
        "INPUT-F-14": RequestState.ANSWERED,
    }[condition_id]
    assert terminal.state is expected_state
    transition_from = (
        result.create_result.command.request.state.value
        if condition_id == "INPUT-F-01"
        else handler.contexts[0].request.state.value
    )
    if condition_id in {"INPUT-F-10", "INPUT-F-14"}:
        raw_structured = public_result.get("structuredContent")
        if isinstance(raw_structured, dict):
            structured = cast(dict[str, object], raw_structured)
        else:
            content_items = cast(
                list[dict[str, object]],
                public_result["content"],
            )
            structured = cast(
                dict[str, object],
                loads(cast(str, content_items[0]["text"])),
            )
        action = cast(str, structured["action"])
        assert action == (
            "cancel" if condition_id == "INPUT-F-10" else "accept"
        )
        status = ("elicitation_action", action)
    else:
        assert public_result["isError"] is True
        status = ("tool_error", "true")
    return _Observation(
        transition=(transition_from, terminal.state.value),
        public_result_id=_CALL_TOOL_RESULT,
        public_result=public_result,
        status=status,
    )


def _assert_condition(
    condition_id: str,
    surface_id: str,
    record_property: _RecordProperty,
) -> None:
    observation = run(
        _downstream_condition(condition_id)
        if surface_id == _DOWNSTREAM_SURFACE
        else _inbound_condition(condition_id)
    )
    _record(record_property, condition_id, surface_id, observation)


@pytest.fixture(params=(_FORM_SURFACE,), ids=(_FORM_SURFACE,))
def form_surface_id(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(params=(_TASK_SURFACE,), ids=(_TASK_SURFACE,))
def task_surface_id(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(params=(_DOWNSTREAM_SURFACE,), ids=(_DOWNSTREAM_SURFACE,))
def downstream_surface_id(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(
    params=(_FORM_SURFACE, _DOWNSTREAM_SURFACE),
    ids=(_FORM_SURFACE, _DOWNSTREAM_SURFACE),
)
def form_or_downstream(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture(
    params=(_TASK_SURFACE, _DOWNSTREAM_SURFACE),
    ids=(_TASK_SURFACE, _DOWNSTREAM_SURFACE),
)
def task_or_downstream(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def test_input_f_01(
    form_or_downstream: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-01", form_or_downstream, record_property)


def test_input_f_04(
    form_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-04", form_surface_id, record_property)


def test_input_f_05(
    form_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-05", form_surface_id, record_property)


def test_input_f_06(
    form_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-06", form_surface_id, record_property)


def test_input_f_07(
    task_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-07", task_surface_id, record_property)


def test_input_f_08(
    form_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-08", form_surface_id, record_property)


def test_input_f_09(
    downstream_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-09", downstream_surface_id, record_property)


def test_input_f_10(
    task_or_downstream: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-10", task_or_downstream, record_property)


def test_input_f_11(
    downstream_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-11", downstream_surface_id, record_property)


def test_input_f_12(
    task_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-12", task_surface_id, record_property)


def test_input_f_14(
    task_or_downstream: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-14", task_or_downstream, record_property)


def test_input_f_15(
    task_surface_id: str,
    record_property: _RecordProperty,
) -> None:
    _assert_condition("INPUT-F-15", task_surface_id, record_property)
