"""Exercise the public A2A structured-input contract."""

from asyncio import run
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from json import loads
from logging import getLogger
from pathlib import Path
from sys import path as sys_path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from avalan.agent.execution import (
    AgentExecution,
    AttachedInteractionRuntime,
    BranchInteractionBroker,
)
from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    FreeFormOther,
    InputRequestId,
    InteractionActor,
    InteractionBrokerRequest,
    InteractionRequestResult,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    QuestionId,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StreamSessionId,
    TaskId,
    TextAnswer,
    TimedOutResolution,
    TurnId,
    UserId,
)
from avalan.interaction import a2a as a2a_module
from avalan.interaction.a2a import (
    A2A_INPUT_EXTENSION_DESCRIPTION,
    A2A_INPUT_EXTENSION_PARAMS,
    A2A_INPUT_EXTENSION_URI,
    A2AInputRequestMetadata,
    a2a_input_request_text,
    decode_a2a_input_request_metadata,
    decode_a2a_input_resolution_metadata,
    encode_a2a_input_request_metadata,
    encode_a2a_input_resolution_metadata,
)
from avalan.interaction.a2a_continuation import (
    A2ARemoteInputContinuation,
    A2AToolContinuationCheckpoint,
    decode_a2a_tool_continuation_observation,
    encode_a2a_tool_continuation_observation,
    project_a2a_input_resolution,
)
from avalan.interaction.error import InputValidationError
from avalan.server.a2a.router import install_a2a_routes
from avalan.tool.a2a import A2ACallTool

sys_path.append(str(Path(__file__).parents[1] / "server"))

import input_interaction_test as server_support  # noqa: E402

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
_OWNER = PrincipalScope(user_id=UserId("a2a-owner"))
_A2A_HEADERS = {
    "A2A-Version": "1.0",
    "A2A-Extensions": A2A_INPUT_EXTENSION_URI,
    "Authorization": "Bearer owner",
}


@dataclass(frozen=True, slots=True)
class _PublicProjection:
    """Store one complete public route/client continuation observation."""

    result: Mapping[str, object]
    events: tuple[str, ...]
    local_requests: tuple[InteractionBrokerRequest, ...]
    provider_calls: int
    sync_calls: int


@dataclass(frozen=True, slots=True)
class _RawContinuation:
    """Store exact wire events from one public A2A continuation."""

    first_events: tuple[dict[str, object], ...]
    second_events: tuple[dict[str, object], ...]
    first_extension_echo: str | None
    second_extension_echo: str | None
    provider_calls: int
    sync_calls: int


class _ClientBroker:
    """Return one deterministic local answer to downstream A2A input."""

    def __init__(self, resolution: object) -> None:
        self.resolution = resolution
        self.requests: list[InteractionBrokerRequest] = []

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Record and answer one downstream input request."""
        self.requests.append(request)
        return cast(
            InteractionRequestResult,
            SimpleNamespace(
                delivery=SimpleNamespace(
                    record=SimpleNamespace(
                        request=SimpleNamespace(resolution=self.resolution)
                    )
                )
            ),
        )

    async def cancel_scope(self, command: object) -> object:
        """Reject unexpected local scope cancellation."""
        _ = command
        raise AssertionError("the successful client path must not cancel")


async def _unused_handler(context: object) -> object:
    """Reject direct handler calls owned by the deterministic broker."""
    _ = context
    raise AssertionError("the deterministic client broker owns resolution")


def _origin(
    *,
    branch_id: str = "nested-branch",
    parent_branch_id: str | None = "parent-branch",
) -> ExecutionOrigin:
    """Return one deterministic nested downstream execution origin."""
    return ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        task_id=TaskId("local-task"),
        agent_id=AgentId("00000000-0000-0000-0000-000000000001"),
        branch_id=BranchId(branch_id),
        parent_branch_id=(
            BranchId(parent_branch_id)
            if parent_branch_id is not None
            else None
        ),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://a2a-contract",
            agent_definition_revision="revision",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model",
            tool_revision="tools",
            capability_revision="capabilities",
        ),
        principal=_OWNER,
    )


def _client_context(
    *,
    stream_event: Callable[[ToolExecutionStreamEvent], Any],
    parent_branch_id: str | None = "parent-branch",
    resolution: object | None = None,
) -> tuple[ToolCallContext, _ClientBroker]:
    """Return one attached downstream context rooted in a nested branch."""
    origin = _origin(parent_branch_id=parent_branch_id)
    if resolution is None:
        resolution = AnsweredResolution(
            request_id=InputRequestId("local-request"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("continue"),
                    provenance=AnswerProvenance.HUMAN,
                    value=True,
                ),
                TextAnswer(
                    question_id=QuestionId("note"),
                    provenance=AnswerProvenance.HUMAN,
                    value="okay",
                ),
            ),
        )
    broker = _ClientBroker(resolution)
    runtime = AttachedInteractionRuntime(
        broker=cast(Any, broker),
        actor=InteractionActor(principal=origin.principal),
        handler=cast(Any, _unused_handler),
    )
    execution = cast(
        AgentExecution,
        SimpleNamespace(origin=origin, interaction_runtime=runtime),
    )
    return (
        ToolCallContext(
            agent_id=UUID(str(origin.agent_id)),
            stream_event=cast(Any, stream_event),
            execution=execution,
            execution_origin=origin,
            interaction_broker=cast(BranchInteractionBroker, broker),
        ),
        broker,
    )


@asynccontextmanager
async def _server(
    *,
    input_extension_required: bool = False,
    broker: Any | None = None,
    provider: Any | None = None,
) -> AsyncIterator[tuple[FastAPI, Any, Any]]:
    """Yield one configured real local A2A route and deterministic provider."""
    broker = broker or await server_support._open_broker()
    provider = provider or server_support._FakeProviderOrchestrator()
    app = server_support._app(broker, provider)
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
        input_extension_required=input_extension_required,
    )
    try:
        yield app, broker, provider
    finally:
        await broker.aclose()


def _send_envelope(
    *,
    rpc_id: str,
    message_id: str,
    task_id: str | None = None,
    context_id: str | None = None,
    request_id: str | None = None,
    answers: Mapping[str, object] | None = None,
    text: str = "Input supplied.",
    method: str = "SendStreamingMessage",
    extensions: tuple[str, ...] = (A2A_INPUT_EXTENSION_URI,),
    extra_resolution: Mapping[str, object] | None = None,
    return_immediately: bool = False,
) -> dict[str, object]:
    """Return one exact JSON-RPC send envelope."""
    message: dict[str, object] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if task_id is not None:
        message["taskId"] = task_id
    if context_id is not None:
        message["contextId"] = context_id
    if request_id is not None:
        resolution: dict[str, object] = {
            "kind": "resolution",
            "request_id": request_id,
            "action": "accept",
            "answers": dict(answers or {"continue": True, "note": "okay"}),
        }
        if extra_resolution is not None:
            resolution.update(extra_resolution)
        message["metadata"] = {A2A_INPUT_EXTENSION_URI: resolution}
        message["extensions"] = list(extensions)
    params: dict[str, object] = {"message": message}
    if return_immediately:
        params["configuration"] = {"returnImmediately": True}
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": method,
        "params": params,
    }


def _sse_events(body: str) -> tuple[dict[str, object], ...]:
    """Decode every JSON object carried by an A2A SSE response."""
    events: list[dict[str, object]] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            value = loads(line.removeprefix("data: "))
            assert isinstance(value, dict)
            events.append(cast(dict[str, object], value))
    assert events
    return tuple(events)


def _result(event: Mapping[str, object]) -> dict[str, object]:
    """Return the JSON-RPC result object from one streamed event."""
    value = event.get("result")
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _task_from_event(event: Mapping[str, object]) -> dict[str, object]:
    """Return the task carried by one streamed event."""
    task = _result(event).get("task")
    assert isinstance(task, dict)
    return cast(dict[str, object], task)


def _status_update(event: Mapping[str, object]) -> dict[str, object]:
    """Return the status update carried by one streamed event."""
    update = _result(event).get("statusUpdate")
    assert isinstance(update, dict)
    return cast(dict[str, object], update)


def _state_from_task(task: Mapping[str, object]) -> str:
    """Return one task state string."""
    status = task.get("status")
    assert isinstance(status, dict)
    state = status.get("state")
    assert isinstance(state, str)
    return state


def _state_from_update(update: Mapping[str, object]) -> str:
    """Return one status-update state string."""
    status = update.get("status")
    assert isinstance(status, dict)
    state = status.get("state")
    assert isinstance(state, str)
    return state


def _input_message(update: Mapping[str, object]) -> dict[str, object]:
    """Return the input-required status message."""
    status = update.get("status")
    assert isinstance(status, dict)
    message = status.get("message")
    assert isinstance(message, dict)
    return cast(dict[str, object], message)


def _extension_payload(message: Mapping[str, object]) -> dict[str, object]:
    """Return the structured task-input extension payload."""
    metadata = message.get("metadata")
    assert isinstance(metadata, dict)
    payload = metadata.get(A2A_INPUT_EXTENSION_URI)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _assert_no_input_extension(task: Mapping[str, object]) -> None:
    """Assert one task projection contains no structured input extension."""
    metadata = task.get("metadata", {})
    assert isinstance(metadata, dict)
    assert A2A_INPUT_EXTENSION_URI not in metadata
    status = task.get("status", {})
    assert isinstance(status, dict)
    messages = list(cast(list[object], task.get("history", [])))
    message = status.get("message")
    if message is not None:
        messages.append(message)
    for value in messages:
        assert isinstance(value, dict)
        assert A2A_INPUT_EXTENSION_URI not in value.get("extensions", [])
        message_metadata = value.get("metadata", {})
        assert isinstance(message_metadata, dict)
        assert A2A_INPUT_EXTENSION_URI not in message_metadata


async def _raw_continuation() -> _RawContinuation:
    """Run one real local JSON-RPC streaming continuation."""
    async with _server() as (app, _broker, provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            first = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-initial",
                    message_id="message-initial",
                ),
            )
            assert first.status_code == 200
            first_events = _sse_events(first.text)
            task = _task_from_event(first_events[0])
            input_update = _status_update(first_events[-1])
            request_id = _extension_payload(_input_message(input_update)).get(
                "request_id"
            )
            task_id = task.get("id")
            context_id = task.get("contextId")
            assert isinstance(request_id, str)
            assert isinstance(task_id, str)
            assert isinstance(context_id, str)
            second = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-follow-up",
                    message_id="message-follow-up",
                    task_id=task_id,
                    context_id=context_id,
                    request_id=request_id,
                ),
            )
            assert second.status_code == 200
            second_events = _sse_events(second.text)
        return _RawContinuation(
            first_events=first_events,
            second_events=second_events,
            first_extension_echo=first.headers.get("A2A-Extensions"),
            second_extension_echo=second.headers.get("A2A-Extensions"),
            provider_calls=provider.provider_calls,
            sync_calls=provider.sync_calls,
        )


async def _request_scoped_activation() -> None:
    """Verify one plain refresh does not inherit prior activation."""
    async with _server() as (app, _broker, _provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            first = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-activated-initial",
                    message_id="message-activated-initial",
                ),
            )
            events = _sse_events(first.text)
            task = _task_from_event(events[0])
            task_id = task.get("id")
            context_id = task.get("contextId")
            assert isinstance(task_id, str)
            assert isinstance(context_id, str)
            plain_rpc_get = await client.post(
                "/a2a",
                headers={
                    "A2A-Version": "1.0",
                    "Authorization": "Bearer owner",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": "rpc-plain-get",
                    "method": "GetTask",
                    "params": {"id": task_id},
                },
            )
            plain_rpc_task = plain_rpc_get.json().get("result")
            assert isinstance(plain_rpc_task, dict)
            _assert_no_input_extension(plain_rpc_task)
            plain_get = await client.get(
                f"/a2a/tasks/{task_id}",
                headers={
                    "A2A-Version": "1.0",
                    "Authorization": "Bearer owner",
                },
            )
            plain_task = plain_get.json()
            assert isinstance(plain_task, dict)
            _assert_no_input_extension(plain_task)
            activated_get = await client.get(
                f"/a2a/tasks/{task_id}",
                headers=_A2A_HEADERS,
            )
            activated_task = activated_get.json()
            assert isinstance(activated_task, dict)
            assert A2A_INPUT_EXTENSION_URI in activated_task["metadata"]
            plain = await client.post(
                "/a2a",
                headers={
                    "A2A-Version": "1.0",
                    "Authorization": "Bearer owner",
                },
                json=_send_envelope(
                    rpc_id="rpc-plain-refresh",
                    message_id="message-plain-refresh",
                    task_id=task_id,
                    context_id=context_id,
                    method="SendMessage",
                    extensions=(),
                ),
            )
            assert "A2A-Extensions" not in plain.headers
            result = plain.json().get("result")
            assert isinstance(result, dict)
            refreshed = result.get("task")
            assert isinstance(refreshed, dict)
            _assert_no_input_extension(refreshed)
            activated_after_refresh = await client.get(
                f"/a2a/tasks/{task_id}",
                headers=_A2A_HEADERS,
            )
            activated_task = activated_after_refresh.json()
            assert isinstance(activated_task, dict)
            assert A2A_INPUT_EXTENSION_URI in activated_task["metadata"]


async def _correlation_observation() -> dict[str, object]:
    """Exercise public rejection of every mismatched follow-up binding."""
    async with _server() as (app, _broker, _provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            first = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-correlation-initial",
                    message_id="message-correlation-initial",
                ),
            )
            first_events = _sse_events(first.text)
            submitted = _task_from_event(first_events[0])
            message = _input_message(_status_update(first_events[-1]))
            request_payload = _extension_payload(message)
            task_id = submitted.get("id")
            context_id = submitted.get("contextId")
            request_id = request_payload.get("request_id")
            assert isinstance(task_id, str)
            assert isinstance(context_id, str)
            assert isinstance(request_id, str)
            stored = await client.get(
                f"/a2a/tasks/{task_id}",
                headers=_A2A_HEADERS,
            )
            task = stored.json()
            assert isinstance(task, dict)
            probes = {
                "task": (
                    _send_envelope(
                        rpc_id="rpc-wrong-task",
                        message_id="message-wrong-task",
                        task_id="wrong-task",
                        context_id=context_id,
                        request_id=request_id,
                        method="SendMessage",
                    ),
                    _A2A_HEADERS,
                ),
                "context": (
                    _send_envelope(
                        rpc_id="rpc-wrong-context",
                        message_id="message-wrong-context",
                        task_id=task_id,
                        context_id="wrong-context",
                        request_id=request_id,
                        method="SendMessage",
                    ),
                    _A2A_HEADERS,
                ),
                "request": (
                    _send_envelope(
                        rpc_id="rpc-wrong-request",
                        message_id="message-wrong-request",
                        task_id=task_id,
                        context_id=context_id,
                        request_id="wrong-request",
                        method="SendMessage",
                    ),
                    _A2A_HEADERS,
                ),
                "principal": (
                    _send_envelope(
                        rpc_id="rpc-wrong-principal",
                        message_id="message-wrong-principal",
                        task_id=task_id,
                        context_id=context_id,
                        request_id=request_id,
                        method="SendMessage",
                    ),
                    {**_A2A_HEADERS, "Authorization": "Bearer other"},
                ),
                "branch": (
                    _send_envelope(
                        rpc_id="rpc-wrong-branch",
                        message_id="message-wrong-branch",
                        task_id=task_id,
                        context_id=context_id,
                        request_id=request_id,
                        method="SendMessage",
                        extra_resolution={"branch_id": "wrong-branch"},
                    ),
                    _A2A_HEADERS,
                ),
                "revision": (
                    _send_envelope(
                        rpc_id="rpc-wrong-revision",
                        message_id="message-wrong-revision",
                        task_id=task_id,
                        context_id=context_id,
                        request_id=request_id,
                        method="SendMessage",
                        extra_resolution={"state_revision": 999},
                    ),
                    _A2A_HEADERS,
                ),
            }
            failures: dict[str, tuple[int, Mapping[str, object]]] = {}
            for name, (envelope, headers) in probes.items():
                response = await client.post(
                    "/a2a",
                    headers=headers,
                    json=envelope,
                )
                body = response.json()
                assert isinstance(body, dict)
                failures[name] = (response.status_code, body)
            authentication = await client.post(
                "/a2a",
                headers={
                    "A2A-Version": "1.0",
                    "A2A-Extensions": A2A_INPUT_EXTENSION_URI,
                },
                json=_send_envelope(
                    rpc_id="rpc-missing-auth",
                    message_id="message-missing-auth",
                    task_id=task_id,
                    context_id=context_id,
                    request_id=request_id,
                    method="SendMessage",
                ),
            )
            completion = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-correlation-valid",
                    message_id="message-correlation-valid",
                    task_id=task_id,
                    context_id=context_id,
                    request_id=request_id,
                ),
            )
            return {
                "task": task,
                "message": message,
                "request": request_payload,
                "failures": failures,
                "authentication": (
                    authentication.status_code,
                    authentication.headers.get("WWW-Authenticate"),
                    authentication.json(),
                ),
                "completion": _sse_events(completion.text),
            }


async def _isolation_observation() -> dict[str, object]:
    """Exercise ambiguous text, fallback, and unavailable-host isolation."""
    async with _server() as (app, _broker, _provider):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="https://a2a.test",
        ) as client:
            first = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-ambiguous-initial",
                    message_id="message-ambiguous-initial",
                ),
            )
            first_events = _sse_events(first.text)
            task = _task_from_event(first_events[0])
            task_id = task.get("id")
            context_id = task.get("contextId")
            assert isinstance(task_id, str)
            assert isinstance(context_id, str)
            ambiguous = await client.post(
                "/a2a",
                headers=_A2A_HEADERS,
                json=_send_envelope(
                    rpc_id="rpc-ambiguous-text",
                    message_id="message-ambiguous-text",
                    task_id=task_id,
                    context_id=context_id,
                    text="yes",
                    method="SendMessage",
                    extensions=(),
                ),
            )
            ambiguous_result = ambiguous.json().get("result")
            assert isinstance(ambiguous_result, dict)
            ambiguous_task = ambiguous_result.get("task")
            assert isinstance(ambiguous_task, dict)
            ambiguous_state = _state_from_task(ambiguous_task)

    async with _server() as (fallback_app, _broker, _provider):
        async with AsyncClient(
            transport=ASGITransport(app=fallback_app),
            base_url="https://fallback.test",
        ) as client:
            fallback = await client.post(
                "/a2a",
                headers={
                    "A2A-Version": "1.0",
                    "Authorization": "Bearer owner",
                },
                json=_send_envelope(
                    rpc_id="rpc-fallback",
                    message_id="message-fallback",
                ),
            )
            fallback_update = _status_update(_sse_events(fallback.text)[-1])
            fallback_state = _state_from_update(fallback_update)
            fallback_message = _input_message(fallback_update)

    auth_app = FastAPI()
    auth_app.state.logger = getLogger("a2a-auth-isolation")
    unavailable_provider = server_support._FakeProviderOrchestrator()
    auth_app.state.orchestrator = unavailable_provider
    install_a2a_routes(
        auth_app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )

    async with AsyncClient(
        transport=ASGITransport(app=auth_app),
        base_url="http://auth.test",
    ) as client:
        unavailable = await client.post(
            "/a2a",
            headers=_A2A_HEADERS,
            json=_send_envelope(
                rpc_id="rpc-unavailable",
                message_id="message-unavailable",
                method="SendMessage",
            ),
        )
    return {
        "ambiguous_state": ambiguous_state,
        "fallback_state": fallback_state,
        "fallback_message": fallback_message,
        "unavailable_status": unavailable.status_code,
        "unavailable_body": unavailable.json(),
        "unavailable_provider_calls": unavailable_provider.provider_calls,
    }


async def _run_public_projection() -> _PublicProjection:
    """Run the real local A2A route through the public downstream tool."""
    async with _server() as (app, _broker, provider):
        events: list[str] = []

        async def observe(event: ToolExecutionStreamEvent) -> None:
            if event.content is not None:
                events.append(event.content)

        context, local_broker = _client_context(stream_event=observe)
        result = await A2ACallTool(
            client_params={
                "transport": ASGITransport(app=app),
                "base_url": "http://testserver",
                "headers": {"Authorization": "Bearer owner"},
            },
            call_params={"request_id": "rpc-client-initial"},
        )(
            "/a2a",
            "run",
            {"input_string": "Start the task."},
            context=context,
        )
        return _PublicProjection(
            result=result,
            events=tuple(events),
            local_requests=tuple(local_broker.requests),
            provider_calls=provider.provider_calls,
            sync_calls=provider.sync_calls,
        )


def _public_projection() -> _PublicProjection:
    """Return one synchronous public route/client observation."""
    return run(_run_public_projection())


def test_requirement_input_n_081() -> None:
    """Project required input as a nonterminal A2A task continuation."""
    observation = run(_raw_continuation())
    first_states = (
        _state_from_task(_task_from_event(observation.first_events[0])),
        *(
            _state_from_update(_status_update(event))
            for event in observation.first_events[1:]
        ),
    )
    second_states = tuple(
        _state_from_update(_status_update(event))
        for event in observation.second_events
        if "statusUpdate" in _result(event)
    )
    assert first_states == (
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_INPUT_REQUIRED",
    )
    assert second_states == (
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
    )
    assert observation.first_extension_echo == A2A_INPUT_EXTENSION_URI
    assert observation.second_extension_echo == A2A_INPUT_EXTENSION_URI
    assert observation.provider_calls == 2
    assert observation.sync_calls == 1


def test_requirement_input_n_082() -> None:
    """Preserve exact task, context, request, and execution correlation."""
    observation = run(_correlation_observation())
    task = cast(Mapping[str, object], observation["task"])
    message = cast(Mapping[str, object], observation["message"])
    request = cast(Mapping[str, object], observation["request"])
    failures = cast(
        Mapping[str, tuple[int, Mapping[str, object]]],
        observation["failures"],
    )
    task_id = task.get("id")
    context_id = task.get("contextId")
    request_id = request.get("request_id")
    assert isinstance(task_id, str)
    assert isinstance(context_id, str)
    assert isinstance(request_id, str)
    assert message.get("taskId") == task_id
    assert message.get("contextId") == context_id
    assert message.get("extensions") == [A2A_INPUT_EXTENSION_URI]
    task_metadata = task.get("metadata")
    assert isinstance(task_metadata, dict)
    assert task_metadata[A2A_INPUT_EXTENSION_URI] == {
        "kind": "request",
        "request_id": request_id,
    }
    for name in ("context", "request", "branch", "revision"):
        status, body = failures[name]
        assert status == 200
        error = body.get("error")
        assert isinstance(error, dict)
        assert error.get("code") == -32602
    wrong_task = failures["task"][1].get("error")
    assert isinstance(wrong_task, dict)
    assert wrong_task.get("code") == -32001
    principal_status, principal_body = failures["principal"]
    assert principal_status == 403
    principal_error = principal_body.get("error")
    assert isinstance(principal_error, dict)
    principal_data = principal_error.get("data")
    assert isinstance(principal_data, dict)
    assert principal_data.get("code") == "avalan.input.unauthorized"
    authentication = cast(
        tuple[int, str | None, Mapping[str, object]],
        observation["authentication"],
    )
    assert authentication[:2] == (401, "Bearer")
    authentication_error = authentication[2].get("error")
    assert isinstance(authentication_error, dict)
    assert authentication_error.get("code") == -32602
    authentication_data = authentication_error.get("data")
    assert isinstance(authentication_data, dict)
    assert (
        authentication_data.get("code")
        == "avalan.input.authentication_required"
    )
    completion = cast(tuple[dict[str, object], ...], observation["completion"])
    assert tuple(
        _state_from_update(_status_update(event))
        for event in completion
        if "statusUpdate" in _result(event)
    ) == ("TASK_STATE_WORKING", "TASK_STATE_COMPLETED")


def test_requirement_input_n_083() -> None:
    """Negotiate structured input while preserving readable fallback."""
    app = FastAPI()
    install_a2a_routes(
        app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
    )
    card = TestClient(app).get("/.well-known/agent-card.json").json()
    assert card["version"] == "1.0.0"
    assert card["capabilities"]["extensions"] == [
        {
            "uri": A2A_INPUT_EXTENSION_URI,
            "description": A2A_INPUT_EXTENSION_DESCRIPTION,
            "required": False,
            "params": A2A_INPUT_EXTENSION_PARAMS,
        }
    ]
    rpc_body = {
        "jsonrpc": "2.0",
        "id": "rpc-negotiation",
        "method": "UnknownMethod",
        "params": {},
    }
    activated = TestClient(app).post(
        "/a2a",
        headers={
            "A2A-Version": "1.0",
            "A2A-Extensions": A2A_INPUT_EXTENSION_URI,
        },
        json=rpc_body,
    )
    assert activated.headers["A2A-Extensions"] == A2A_INPUT_EXTENSION_URI
    optional = TestClient(app).post(
        "/a2a",
        headers={
            "A2A-Version": "1.0",
            "A2A-Extensions": "urn:example:unsupported",
        },
        json=rpc_body,
    )
    assert "A2A-Extensions" not in optional.headers
    required_app = FastAPI()
    install_a2a_routes(
        required_app,
        prefix="/a2a",
        name="run",
        description="Run the test agent.",
        input_extension_required=True,
    )
    required = TestClient(required_app).post(
        "/a2a",
        headers={"A2A-Version": "1.0"},
        json=rpc_body,
    )
    assert required.status_code == 400
    assert required.json()["error"] == {
        "code": -32008,
        "message": "Structured input contract result.",
        "data": {"code": "avalan.input.extension_required"},
    }
    pending = run(_raw_continuation()).first_events[-1]
    message = _input_message(_status_update(pending))
    parts = message.get("parts")
    assert isinstance(parts, list)
    assert parts[0]["text"].startswith("Additional input is required.")
    assert _extension_payload(message)["questions"]
    run(_request_scoped_activation())


def test_requirement_input_n_084() -> None:
    """Keep ambiguous text and authentication outside input resolution."""
    isolation = run(_isolation_observation())
    assert isolation["ambiguous_state"] == "TASK_STATE_INPUT_REQUIRED"
    assert isolation["fallback_state"] == "TASK_STATE_INPUT_REQUIRED"
    fallback = cast(Mapping[str, object], isolation["fallback_message"])
    assert A2A_INPUT_EXTENSION_URI not in cast(
        Mapping[str, object],
        fallback.get("metadata", {}),
    )
    assert A2A_INPUT_EXTENSION_URI not in cast(
        list[object],
        fallback.get("extensions", []),
    )
    assert isolation["unavailable_status"] == 503
    unavailable = cast(Mapping[str, object], isolation["unavailable_body"])
    error = cast(Mapping[str, object], unavailable.get("error"))
    assert error.get("code") == -31910
    assert isolation["unavailable_provider_calls"] == 0
    projection = _public_projection()
    assert len(projection.local_requests) == 1
    origin = projection.local_requests[0].origin
    assert origin.branch_id == BranchId("nested-branch")
    assert origin.parent_branch_id == BranchId("parent-branch")
    structured = projection.result.get("structuredContent")
    assert isinstance(structured, dict)
    assert structured.get("messages") == []
    serialized = repr(projection.result) + "".join(projection.events)
    assert "Bearer owner" not in serialized
    assert "Input supplied." not in serialized


def _transport_questions() -> tuple[
    MultilineTextQuestion,
    SingleSelectionQuestion,
    MultipleSelectionQuestion,
]:
    choices = (
        Choice(value=ChoiceValue("safe"), label="Safe"),
        Choice(value=ChoiceValue("fast"), label="Fast"),
    )
    return (
        MultilineTextQuestion(
            question_id=QuestionId("details"),
            prompt="Details?",
            required=True,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Strategy?",
            required=True,
            choices=choices,
            allow_other=True,
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Checks?",
            required=False,
            choices=choices,
            allow_other=True,
            constraints=SelectionValidationConstraints(maximum=3),
        ),
    )


def _transport_request(*questions: object) -> A2AInputRequestMetadata:
    return A2AInputRequestMetadata(
        request_id=InputRequestId("transport-request"),
        required=True,
        questions=cast(Any, questions or _transport_questions()),
    )


def _transport_remote() -> A2ARemoteInputContinuation:
    return A2ARemoteInputContinuation(
        request=_transport_request(),
        request_text="Additional input is required.",
        task_id="remote-task",
        context_id="remote-context",
        prior_message_id="remote-message",
        prior_content=("prefix",),
        ttl_seconds=300,
        input_cycle_count=1,
    )


def _transport_checkpoint() -> A2AToolContinuationCheckpoint:
    return A2AToolContinuationCheckpoint(
        call_id="transport-call",
        canonical_name="a2a.call",
        provider_name="a2a.call",
        provider_name_encoded=False,
        arguments={
            "uri": "https://peer.example/a2a",
            "name": "remote.skill",
            "arguments": {},
        },
        remote=_transport_remote(),
        interaction_fingerprint_counts=(("input", 1),),
    )


def test_a2a_transport_codec_boundary_matrix() -> None:
    """Exercise each bounded A2A request, answer, and action variant."""
    request = _transport_request()
    encoded = encode_a2a_input_request_metadata(request)
    assert decode_a2a_input_request_metadata(encoded) == request
    assert "Safe" in a2a_input_request_text(request)
    answered = AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(
            MultilineTextAnswer(
                question_id=QuestionId("details"),
                provenance=AnswerProvenance.HUMAN,
                value="line one\nline two",
            ),
            SingleSelectionAnswer(
                question_id=QuestionId("strategy"),
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("safe")),
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("checks"),
                provenance=AnswerProvenance.HUMAN,
                values=(FreeFormOther(text="security"),),
            ),
        ),
    )
    assert isinstance(
        decode_a2a_input_resolution_metadata(
            encode_a2a_input_resolution_metadata(answered),
            request=request,
            resolved_at=_NOW,
        ),
        AnsweredResolution,
    )
    for resolution_type in (DeclinedResolution, CancelledResolution):
        resolution = resolution_type(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        )
        assert isinstance(
            decode_a2a_input_resolution_metadata(
                encode_a2a_input_resolution_metadata(resolution),
                request=request,
                resolved_at=_NOW,
            ),
            resolution_type,
        )

    with pytest.raises(InputValidationError):
        encode_a2a_input_request_metadata(cast(Any, object()))
    with pytest.raises(InputValidationError):
        encode_a2a_input_resolution_metadata(
            TimedOutResolution(
                request_id=request.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=_NOW,
            )
        )
    for value in (
        {**encoded, "kind": "other"},
        {**encoded, "required": 1},
        {**encoded, "questions": {}},
        {**encoded, "questions": []},
    ):
        with pytest.raises(InputValidationError):
            decode_a2a_input_request_metadata(value)

    multiple = encode_a2a_input_request_metadata(
        _transport_request(_transport_questions()[2])
    )
    raw = cast(list[dict[str, object]], multiple["questions"])[0]
    tuple_choices = {
        **raw,
        "choices": tuple(cast(list[object], raw["choices"])),
    }
    assert decode_a2a_input_request_metadata(
        {
            **multiple,
            "questions": [tuple_choices],
        }
    )
    for question in (
        {**raw, "kind": "unsupported"},
        {key: value for key, value in raw.items() if key != "prompt"},
        {**raw, "choices": {}},
        {**raw, "allow_other": "yes"},
        {**raw, "required": "yes"},
    ):
        with pytest.raises(InputValidationError):
            decode_a2a_input_request_metadata(
                {
                    **multiple,
                    "questions": [question],
                }
            )


def test_a2a_transport_answer_failure_matrix() -> None:
    """Reject every unsupported or out-of-bounds answer representation."""

    def decode(question: object, value: object) -> object:
        request = _transport_request(question)
        question_id = str(cast(Any, question).question_id)
        return decode_a2a_input_resolution_metadata(
            {
                "kind": "resolution",
                "request_id": str(request.request_id),
                "action": "accept",
                "answers": {question_id: value},
            },
            request=request,
            resolved_at=_NOW,
        )

    single = _transport_questions()[1]
    assert decode(single, {"kind": "selected_choice", "value": "safe"})
    assert decode(single, {"kind": "free_form_other", "text": "custom"})
    for value in (
        {"kind": "selected_choice", "value": ""},
        {"kind": "selected_choice", "value": "unknown"},
        {"kind": "selected_choice", "value": "safe", "extra": True},
        {"kind": "free_form_other", "text": "x" * 1_001},
        {"kind": "free_form_other", "text": "two\nlines"},
        {"kind": "unknown"},
    ):
        with pytest.raises(InputValidationError):
            decode(single, value)
    no_other = replace(single, allow_other=False)
    with pytest.raises(InputValidationError):
        decode(no_other, {"kind": "free_form_other", "text": "custom"})
    with pytest.raises(InputValidationError):
        decode(_transport_questions()[2], "not-an-array")
    with pytest.raises(InputValidationError):
        decode(_transport_questions()[0], "x" * 10_001)
    request = _transport_request(single)
    for payload in (
        {"kind": "other"},
        {
            "kind": "resolution",
            "request_id": str(request.request_id),
            "action": "unknown",
        },
    ):
        with pytest.raises(InputValidationError):
            decode_a2a_input_resolution_metadata(
                payload,
                request=request,
                resolved_at=_NOW,
            )

    long_answer = MultilineTextAnswer(
        question_id=QuestionId("details"),
        provenance=AnswerProvenance.HUMAN,
        value="x" * 10_001,
    )
    for answer_value in (long_answer, object()):
        with pytest.raises(InputValidationError):
            a2a_module._encode_a2a_answer(answer_value)
    with pytest.raises(InputValidationError):
        a2a_module._encode_a2a_selection(cast(Any, object()))
    for text in ("x" * 1_001, "two\nlines"):
        invalid = object.__new__(FreeFormOther)
        object.__setattr__(invalid, "text", text)
        with pytest.raises(InputValidationError):
            a2a_module._encode_a2a_selection(invalid)


def test_a2a_checkpoint_failure_matrix() -> None:
    """Reject malformed portable checkpoints and unsupported projections."""
    checkpoint = _transport_checkpoint()
    encoded = encode_a2a_tool_continuation_observation(checkpoint)
    assert decode_a2a_tool_continuation_observation((encoded,)) == checkpoint
    assert decode_a2a_tool_continuation_observation(()) is None
    with pytest.raises(TypeError):
        encode_a2a_tool_continuation_observation(cast(Any, object()))
    payloads: tuple[Mapping[str, Any], ...] = (
        {**encoded, "version": 2},
        {**encoded, "interaction_counts": []},
        {**encoded, "call_arguments": []},
    )
    for payload in payloads:
        with pytest.raises(InputValidationError):
            decode_a2a_tool_continuation_observation((payload,))

    remote = checkpoint.remote
    for changes in (
        {"request_text": " "},
        {"request": cast(Any, object())},
        {"ttl_seconds": 59},
        {"prior_content": cast(Any, [])},
    ):
        with pytest.raises(InputValidationError):
            replace(remote, **changes)
    with pytest.raises(InputValidationError):
        replace(checkpoint, canonical_name="")
    with pytest.raises(InputValidationError):
        replace(
            checkpoint,
            arguments={"uri": "", "name": "remote.skill"},
        )
    with pytest.raises(InputValidationError):
        project_a2a_input_resolution(None, remote.request)
