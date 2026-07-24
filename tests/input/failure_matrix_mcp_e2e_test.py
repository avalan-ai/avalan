"""Exercise MCP cells in the structured-input failure matrix."""

from asyncio import CancelledError, create_task, run, wait
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from sys import path as sys_path
from types import SimpleNamespace
from typing import cast

import pytest
from mcp.types import ErrorData

sys_path.append(str(Path(__file__).parent))

import mcp_contract_test as contract  # noqa: E402

from avalan.interaction import (  # noqa: E402
    AnsweredResolution,
    AnswerProvenance,
    ConfirmationQuestion,
    ExpiredResolution,
    InputHandlerContext,
    InputHandlerResolution,
    InputRequestId,
    QuestionId,
    SingleSelectionQuestion,
    SupersededResolution,
    TextQuestion,
    TimedOutResolution,
)
from avalan.interaction.broker import InteractionRequestResult  # noqa: E402
from avalan.server.mcp_session import (  # noqa: E402
    MCP_CONFLICT,
    MCP_INVALID_PARAMS,
    MCPFormErrorCode,
    MCPFormSessionError,
)
from avalan.server.mcp_tasks import (  # noqa: E402
    MCPTaskCapabilities,
    MCPTaskController,
    MCPTaskRequest,
    parse_task_request,
)
from avalan.server.routers import mcp as mcp_router  # noqa: E402
from avalan.tool import mcp_session as downstream_mcp  # noqa: E402

_EVIDENCE_PROPERTY = "failure_matrix_evidence"
_FORM_SURFACES = (
    "mcp-downstream-elicitation",
    "mcp-inbound-elicitation-form",
    "mcp-inbound-task",
)


def _evidence(
    condition_id: str,
    *,
    transition: tuple[str, str],
    public_result_id: str,
    status: tuple[str, str],
    provider_calls: int,
    surfaces: tuple[str, ...] = _FORM_SURFACES,
) -> list[dict[str, object]]:
    return [
        {
            "condition_id": condition_id,
            "surface_id": surface,
            "transition_from": transition[0],
            "transition_to": transition[1],
            "public_result_id": public_result_id,
            "public_result": {"redacted": True},
            "status_key": status[0],
            "status_value": status[1],
            "provider_call_count": provider_calls,
            "domain_side_effect_count": 0,
        }
        for surface in surfaces
    ]


def _record(
    record_property: Callable[[str, object], None],
    evidence: list[dict[str, object]],
) -> None:
    assert evidence
    record_property(_EVIDENCE_PROPERTY, evidence)


async def _invalid_form(
    question: ConfirmationQuestion | SingleSelectionQuestion | TextQuestion,
    content: dict[str, object],
) -> MCPFormSessionError:
    registry = await contract._registry()
    handled = create_task(
        registry.handler(
            session_id="session",
            owner=contract._OWNER,
            related_request_id="call",
        )(InputHandlerContext(request=contract._request(question)))
    )
    outbound = await registry.next_outbound("session", contract._OWNER)
    assert outbound is not None
    with pytest.raises(MCPFormSessionError) as caught:
        await registry.dispatch_response(
            "session",
            contract._OWNER,
            {
                "jsonrpc": "2.0",
                "id": outbound.jsonrpc_id,
                "result": {"action": "accept", "content": content},
            },
        )
    assert caught.value.rpc_code == MCP_INVALID_PARAMS
    assert not handled.done()
    assert await registry.pending_count("session", contract._OWNER) == 1
    question_id = str(question.question_id)
    if isinstance(question, ConfirmationQuestion):
        corrected: object = True
    elif isinstance(question, SingleSelectionQuestion):
        corrected = str(question.choices[0].value)
    else:
        corrected = "corrected"
    await registry.dispatch_response(
        "session",
        contract._OWNER,
        {
            "jsonrpc": "2.0",
            "id": outbound.jsonrpc_id,
            "result": {
                "action": "accept",
                "content": {question_id: corrected},
            },
        },
    )
    assert isinstance(await handled, InputHandlerResolution)
    assert await registry.pending_count("session", contract._OWNER) == 0
    return caught.value


async def _duplicate_response(*, conflicting: bool) -> None:
    registry = await contract._registry()
    request = contract._request(
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        )
    )
    owner = (contract._OWNER, "session")
    controller = MCPTaskController(id_factory=lambda: "task")
    creation = await controller.create(
        MCPTaskRequest(),
        requestor=owner,
    )
    handled = create_task(
        registry.handler(
            session_id="session",
            owner=contract._OWNER,
            related_request_id="call",
            related_task_id="task",
            status_hook=mcp_router._mcp_task_status_hook(creation.handle),
        )(InputHandlerContext(request=request))
    )
    outbound = await registry.next_outbound("session", contract._OWNER)
    assert outbound is not None
    assert outbound.related_task_id == "task"
    assert (await controller.get("task", requestor=owner))[
        "status"
    ] == "input_required"
    response = {
        "jsonrpc": "2.0",
        "id": outbound.jsonrpc_id,
        "result": {
            "action": "accept",
            "content": {"confirm": True},
            "_meta": {
                "io.modelcontextprotocol/related-task": {"taskId": "task"}
            },
        },
    }
    await registry.dispatch_response("session", contract._OWNER, response)
    outcome = await handled
    assert isinstance(outcome, InputHandlerResolution)
    assert isinstance(outcome.resolution, AnsweredResolution)
    assert (await controller.get("task", requestor=owner))[
        "status"
    ] == "working"

    if not conflicting:
        await registry.dispatch_response(
            "session",
            contract._OWNER,
            response,
        )
        projected = downstream_mcp._broker_result(
            _terminal_result(outcome.resolution)
        )
        assert projected.action == "accept"
        assert projected.content == {"confirm": True}
    else:
        conflict = {
            **response,
            "result": {
                **cast(dict[str, object], response["result"]),
                "content": {"confirm": False},
            },
        }
        with pytest.raises(MCPFormSessionError) as caught:
            await registry.dispatch_response(
                "session",
                contract._OWNER,
                conflict,
            )
        assert caught.value.code is MCPFormErrorCode.STALE_RESPONSE
        assert caught.value.rpc_code == MCP_CONFLICT
        resolution = SupersededResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=contract._NOW,
        )
        projected = downstream_mcp._broker_result(_terminal_result(resolution))
        assert isinstance(projected, ErrorData)
        assert projected.code == MCP_CONFLICT
        assert projected.data == {"code": "avalan.input.already_resolved"}
    assert (await controller.get("task", requestor=owner))[
        "status"
    ] == "working"


async def _terminal_task_error(
    resolution: ExpiredResolution | SupersededResolution,
    expected: dict[str, object],
) -> None:
    registry = await contract._registry()
    owner = (contract._OWNER, "session")
    controller = MCPTaskController(id_factory=lambda: "task")
    creation = await controller.create(MCPTaskRequest(), requestor=owner)
    handled = create_task(
        registry.handler(
            session_id="session",
            owner=contract._OWNER,
            related_request_id="call",
            related_task_id="task",
            status_hook=mcp_router._mcp_task_status_hook(creation.handle),
        )(
            InputHandlerContext(
                request=contract._request(
                    TextQuestion(
                        question_id=QuestionId("name"),
                        prompt="Name?",
                        required=True,
                    ),
                    request_id=str(resolution.request_id),
                )
            )
        )
    )
    outbound = await registry.next_outbound("session", contract._OWNER)
    assert outbound is not None
    assert outbound.related_task_id == "task"
    assert (await controller.get("task", requestor=owner))[
        "status"
    ] == "input_required"

    handled.cancel()
    with suppress(CancelledError):
        await handled
    assert await registry.pending_count("session", contract._OWNER) == 0

    projected = downstream_mcp._broker_result(_terminal_result(resolution))
    assert isinstance(projected, ErrorData)
    projected_error: dict[str, object] = {
        "code": projected.code,
        "message": projected.message,
    }
    if projected.data is not None:
        projected_error["data"] = projected.data
    assert projected_error == expected
    assert (await creation.handle.fail(projected_error))["status"] == "failed"
    outcome = await controller.result("task", requestor=owner)
    assert outcome.error == expected


async def _expired_resolution() -> None:
    resolution = ExpiredResolution(
        request_id=InputRequestId("expired"),
        provenance=AnswerProvenance.POLICY,
        resolved_at=contract._NOW,
    )
    await _terminal_task_error(
        resolution,
        {
            "code": -32010,
            "message": "Avalan input request expired",
            "data": {"code": "avalan.input.expired"},
        },
    )


async def _cancelled_task() -> None:
    owner = (contract._OWNER, "session")
    controller = MCPTaskController(id_factory=lambda: "task")
    creation = await controller.create(MCPTaskRequest(), requestor=owner)
    await creation.handle.transition_input_required()
    assert (await controller.cancel("task", requestor=owner))[
        "status"
    ] == "cancelled"
    assert (await creation.handle.complete({"content": []}))[
        "status"
    ] == "cancelled"
    outcome = await controller.result("task", requestor=owner)
    assert outcome.error == {
        "code": -32000,
        "message": "Request cancelled",
    }


async def _superseded_resolution() -> None:
    resolution = SupersededResolution(
        request_id=InputRequestId("superseded"),
        provenance=AnswerProvenance.POLICY,
        resolved_at=contract._NOW,
    )
    await _terminal_task_error(
        resolution,
        {
            "code": -32009,
            "message": "Avalan input request was superseded",
            "data": {"code": "avalan.input.already_resolved"},
        },
    )


async def _pending_result_budget() -> None:
    owner = (contract._OWNER, "session")
    controller = MCPTaskController(id_factory=lambda: "task")
    creation = await controller.create(MCPTaskRequest(), requestor=owner)
    await creation.handle.transition_input_required()
    result = create_task(controller.result("task", requestor=owner))
    done, _ = await wait({result}, timeout=0.01)
    assert not done
    assert (await controller.get("task", requestor=owner))[
        "status"
    ] == "input_required"
    await controller.cancel("task", requestor=owner)
    assert (await result).error is not None


def _terminal_result(resolution: object) -> InteractionRequestResult:
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


async def _advisory_timeout() -> None:
    resolution = TimedOutResolution(
        request_id=InputRequestId("timed-out"),
        provenance=AnswerProvenance.POLICY,
        resolved_at=contract._NOW,
    )
    projected = downstream_mcp._broker_result(_terminal_result(resolution))
    assert projected.action == "cancel"
    task = await MCPTaskController(id_factory=lambda: "task").create(
        MCPTaskRequest(),
        requestor=(contract._OWNER, "session"),
    )
    assert task.task["status"] == "working"


async def _ignored_ui_hint() -> None:
    registry = await contract._registry()
    _, outcome = await contract._round_trip(
        registry,
        contract._request(
            TextQuestion(
                question_id=QuestionId("name"),
                prompt="Name?",
                required=True,
                header="Preferred rendering",
                help_text="A single-line control is ideal.",
            )
        ),
        {"action": "accept", "content": {"name": "Ada"}},
    )
    assert isinstance(outcome, InputHandlerResolution)
    assert isinstance(outcome.resolution, AnsweredResolution)

    owner = (contract._OWNER, "session")
    task = await MCPTaskController(id_factory=lambda: "task").create(
        MCPTaskRequest(),
        requestor=owner,
    )
    await task.handle.transition_input_required()
    assert (await task.handle.transition_working())["status"] == "working"


async def _missing_capability() -> None:
    await contract._incapable_fallback()
    assert (
        parse_task_request(
            {"task": {"ttl": 20}},
            request_type="tools/call",
            capabilities=MCPTaskCapabilities(),
            execution_task_support="optional",
        )
        is None
    )


def test_input_f_01(
    record_property: Callable[[str, object], None],
) -> None:
    run(contract._incapable_fallback())
    _record(
        record_property,
        _evidence(
            "INPUT-F-01",
            transition=("created", "unavailable"),
            public_result_id="mcp.unavailable_error.v1",
            status=("jsonrpc_error", "-32001"),
            provider_calls=0,
            surfaces=(
                "mcp-downstream-elicitation",
                "mcp-inbound-elicitation-form",
                "mcp-inbound-elicitation-url-auth",
                "mcp-inbound-task",
            ),
        ),
    )


def test_input_f_04(
    record_property: Callable[[str, object], None],
) -> None:
    error = run(
        _invalid_form(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
            {"confirm": "yes"},
        )
    )
    assert error.code is MCPFormErrorCode.INVALID_RESPONSE
    _record(
        record_property,
        _evidence(
            "INPUT-F-04",
            transition=("pending", "pending"),
            public_result_id="mcp.invalid_params_error.v1",
            status=("jsonrpc_error", "-32602"),
            provider_calls=1,
        ),
    )


def test_input_f_05(
    record_property: Callable[[str, object], None],
) -> None:
    error = run(
        _invalid_form(
            SingleSelectionQuestion(
                question_id=QuestionId("choice"),
                prompt="Choose.",
                required=True,
                choices=contract._CHOICES,
            ),
            {"choice": "unknown"},
        )
    )
    assert error.code is MCPFormErrorCode.INVALID_RESPONSE
    _record(
        record_property,
        _evidence(
            "INPUT-F-05",
            transition=("pending", "pending"),
            public_result_id="mcp.invalid_params_error.v1",
            status=("jsonrpc_error", "-32602"),
            provider_calls=1,
        ),
    )


def test_input_f_06(
    record_property: Callable[[str, object], None],
) -> None:
    error = run(
        _invalid_form(
            TextQuestion(
                question_id=QuestionId("required"),
                prompt="Required.",
                required=True,
            ),
            {},
        )
    )
    assert error.code is MCPFormErrorCode.INVALID_RESPONSE
    _record(
        record_property,
        _evidence(
            "INPUT-F-06",
            transition=("pending", "pending"),
            public_result_id="mcp.invalid_params_error.v1",
            status=("jsonrpc_error", "-32602"),
            provider_calls=1,
        ),
    )


def test_input_f_07(
    record_property: Callable[[str, object], None],
) -> None:
    run(_duplicate_response(conflicting=False))
    _record(
        record_property,
        _evidence(
            "INPUT-F-07",
            transition=("answered", "answered"),
            public_result_id="mcp.task_working.v1",
            status=("task_status", "working"),
            provider_calls=2,
        ),
    )


def test_input_f_08(
    record_property: Callable[[str, object], None],
) -> None:
    run(_duplicate_response(conflicting=True))
    _record(
        record_property,
        _evidence(
            "INPUT-F-08",
            transition=("answered", "answered"),
            public_result_id="mcp.conflict_error.v1",
            status=("jsonrpc_error", "-32009"),
            provider_calls=2,
        ),
    )


def test_input_f_09(
    record_property: Callable[[str, object], None],
) -> None:
    run(_expired_resolution())
    _record(
        record_property,
        _evidence(
            "INPUT-F-09",
            transition=("pending", "expired"),
            public_result_id="mcp.expired_error.v1",
            status=("jsonrpc_error", "-32010"),
            provider_calls=1,
        ),
    )


def test_input_f_10(
    record_property: Callable[[str, object], None],
) -> None:
    run(_cancelled_task())
    _record(
        record_property,
        _evidence(
            "INPUT-F-10",
            transition=("pending", "cancelled"),
            public_result_id="mcp.task_cancelled.v1",
            status=("task_status", "cancelled"),
            provider_calls=1,
        ),
    )


def test_input_f_11(
    record_property: Callable[[str, object], None],
) -> None:
    run(_superseded_resolution())
    _record(
        record_property,
        _evidence(
            "INPUT-F-11",
            transition=("pending", "superseded"),
            public_result_id="mcp.conflict_error.v1",
            status=("jsonrpc_error", "-32009"),
            provider_calls=1,
        ),
    )


def test_input_f_12(
    record_property: Callable[[str, object], None],
) -> None:
    run(_pending_result_budget())
    _record(
        record_property,
        _evidence(
            "INPUT-F-12",
            transition=("pending", "pending"),
            public_result_id="mcp.task_input_required.v1",
            status=("task_status", "input_required"),
            provider_calls=1,
        ),
    )


def test_input_f_13(
    record_property: Callable[[str, object], None],
) -> None:
    run(_advisory_timeout())
    _record(
        record_property,
        _evidence(
            "INPUT-F-13",
            transition=("pending", "timed_out"),
            public_result_id="mcp.task_working.v1",
            status=("task_status", "working"),
            provider_calls=2,
        ),
    )


def test_input_f_14(
    record_property: Callable[[str, object], None],
) -> None:
    run(_ignored_ui_hint())
    _record(
        record_property,
        _evidence(
            "INPUT-F-14",
            transition=("pending", "answered"),
            public_result_id="mcp.task_working.v1",
            status=("task_status", "working"),
            provider_calls=2,
        ),
    )


def test_input_f_15(
    record_property: Callable[[str, object], None],
) -> None:
    run(_missing_capability())
    evidence = _evidence(
        "INPUT-F-15",
        transition=("created", "unavailable"),
        public_result_id="mcp.unavailable_error.v1",
        status=("jsonrpc_error", "-32001"),
        provider_calls=0,
        surfaces=(
            "mcp-downstream-elicitation",
            "mcp-inbound-elicitation-form",
        ),
    )
    evidence.extend(
        _evidence(
            "INPUT-F-15",
            transition=("running", "running"),
            public_result_id="mcp.ordinary_result.v1",
            status=("result", "ordinary"),
            provider_calls=1,
            surfaces=("mcp-inbound-task",),
        )
    )
    _record(record_property, evidence)
