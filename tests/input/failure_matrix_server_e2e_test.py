"""Exercise executable HTTP cells in the structured-input failure matrix."""

from asyncio import gather, run, sleep
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from json import dumps
from pathlib import Path
from sys import path as sys_path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient, Response

from avalan.interaction import (
    AnswerProvenance,
    AsyncInteractionBroker,
    Choice,
    ChoiceValue,
    InteractionExecutionScope,
    SingleSelectionQuestion,
    SupersedeInteractionScopeCommand,
)
from avalan.server.interaction import (
    close_server_interactions,
    configure_server_interactions,
)

sys_path.append(str(Path(__file__).parents[1] / "server"))
sys_path.append(str(Path(__file__).parent))

import broker_contract_test as broker_support  # noqa: E402
import input_interaction_test as server_support  # noqa: E402

_OWNER_HEADERS = {"Authorization": "Bearer owner"}
_CONFIRMATION_QUESTION = server_support.ConfirmationQuestion
_SURFACE = "http-chat-completions-nonstream"
_CASES = (
    ("INPUT-F-01", _SURFACE),
    ("INPUT-F-01", "http-chat-completions-stream"),
    ("INPUT-F-01", "http-responses-stream"),
    ("INPUT-F-01", "http-responses-nonstream"),
    ("INPUT-F-04", "http-input-resolve"),
    ("INPUT-F-05", "http-input-resolve"),
    ("INPUT-F-06", "http-input-resolve"),
    ("INPUT-F-07", "http-input-resolve"),
    ("INPUT-F-08", "http-input-resolve"),
    ("INPUT-F-09", "http-input-resolve"),
    ("INPUT-F-11", "http-input-resolve"),
    ("INPUT-F-12", _SURFACE),
    ("INPUT-F-12", "http-responses-nonstream"),
    ("INPUT-F-15", _SURFACE),
    ("INPUT-F-15", "http-chat-completions-stream"),
    ("INPUT-F-15", "http-responses-stream"),
    ("INPUT-F-15", "http-responses-nonstream"),
)


def _generation_request(
    surface_id: str,
) -> tuple[str, dict[str, object]]:
    match surface_id:
        case "http-chat-completions-stream":
            return (
                "/v1/chat/completions",
                server_support._completion_payload(
                    stream=True,
                    handling="detached",
                ),
            )
        case "http-chat-completions-nonstream":
            return (
                "/v1/chat/completions",
                server_support._completion_payload(
                    stream=False,
                    handling="detached",
                ),
            )
        case "http-responses-stream":
            return (
                "/v1/responses",
                server_support._responses_payload(
                    stream=True,
                    handling="detached",
                ),
            )
        case "http-responses-nonstream":
            return (
                "/v1/responses",
                server_support._responses_payload(
                    stream=False,
                    handling="detached",
                ),
            )
        case _:
            raise AssertionError(
                f"unsupported HTTP generation surface: {surface_id}"
            )


@asynccontextmanager
async def _pending(
    broker: AsyncInteractionBroker | None = None,
    *,
    surface_id: str = _SURFACE,
) -> AsyncIterator[Any]:
    active_broker = broker or await server_support._open_broker()
    provider = server_support._FakeProviderOrchestrator()
    app = server_support._app(active_broker, provider)
    path, payload = _generation_request(surface_id)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://failure.test",
    ) as client:
        response = await client.post(
            path,
            headers=server_support._EXTENSION_HEADERS,
            json=payload,
        )
        assert response.status_code == 202
        request_id = response.json()["request_id"]
        inspection = await client.get(
            f"/v1/input/requests/{request_id}",
            headers=_OWNER_HEADERS,
        )
        assert inspection.status_code == 200
        try:
            yield SimpleNamespace(
                client=client,
                broker=active_broker,
                provider=provider,
                request_id=request_id,
                required=response,
                body=inspection.json(),
                revision=inspection.json()["state_revision"],
            )
        finally:
            await close_server_interactions(app)
            await active_broker.aclose()
            await gather(*provider._active_tasks, return_exceptions=True)
            assert all(task.done() for task in provider._active_tasks)


def _selection_question(**values: object) -> SingleSelectionQuestion:
    return SingleSelectionQuestion(
        question_id=values["question_id"],
        prompt=str(values["prompt"]),
        required=bool(values["required"]),
        choices=(
            Choice(value=ChoiceValue("known"), label="Known"),
            Choice(value=ChoiceValue("alternate"), label="Alternate"),
        ),
    )


async def _wait(pending: Any, state: str) -> dict[str, Any]:
    for _ in range(100):
        response = await pending.client.get(
            f"/v1/input/requests/{pending.request_id}",
            headers=_OWNER_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        if body["state"] == state:
            return body
        await sleep(0)
    raise AssertionError(f"HTTP interaction did not reach {state}")


async def _post(pending: Any, action: str, payload: object) -> Any:
    return await pending.client.post(
        f"/v1/input/requests/{pending.request_id}/{action}",
        headers=_OWNER_HEADERS,
        json=payload,
    )


async def _poll(pending: Any) -> Any:
    return await pending.client.get(
        f"/v1/input/requests/{pending.request_id}/poll",
        params={"transport": "json"},
        headers=_OWNER_HEADERS,
    )


def _evidence(
    condition_id: str,
    surface_id: str,
    transition: tuple[str, str],
    envelope_id: str,
    response: Response,
    provider_calls: int,
) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "surface_id": surface_id,
        "transition_from": transition[0],
        "transition_to": transition[1],
        "public_result_id": envelope_id,
        "public_result": response.json(),
        "status_key": "http",
        "status_value": str(response.status_code),
        "provider_call_count": provider_calls,
        "domain_side_effect_count": 0,
    }


async def _unavailable(
    condition_id: str, surface_id: str
) -> dict[str, object]:
    broker = await server_support._open_broker()
    provider = server_support._FakeProviderOrchestrator()
    app = server_support._app(broker, provider)
    if condition_id == "INPUT-F-01":
        configure_server_interactions(app, None)
    path, payload = _generation_request(surface_id)
    if condition_id == "INPUT-F-15":
        payload["extensions"] = {
            "task_input": {"version": "2", "handling": "detached"}
        }
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://failure.test",
        ) as client:
            response = await client.post(
                path,
                headers=server_support._EXTENSION_HEADERS,
                json=payload,
            )
            assert response.status_code == 503
            assert response.json()["code"] == "input.unavailable"
            assert provider.provider_calls == 0
            assert not provider._active_tasks
            return _evidence(
                condition_id,
                surface_id,
                ("created", "unavailable"),
                "http.unavailable.v1",
                response,
                provider.provider_calls,
            )
    finally:
        await close_server_interactions(app)
        await broker.aclose()


async def _observe(condition_id: str, surface_id: str) -> dict[str, object]:
    if condition_id in {"INPUT-F-01", "INPUT-F-15"}:
        return await _unavailable(condition_id, surface_id)
    harness = (
        await broker_support._harness()
        if condition_id == "INPUT-F-09"
        else None
    )
    question = (
        _selection_question
        if condition_id == "INPUT-F-05"
        else _CONFIRMATION_QUESTION
    )
    with patch.object(server_support, "ConfirmationQuestion", question):
        async with _pending(
            harness.broker if harness else None,
            surface_id=(
                surface_id if condition_id == "INPUT-F-12" else _SURFACE
            ),
        ) as pending:
            if condition_id in {"INPUT-F-04", "INPUT-F-05", "INPUT-F-06"}:
                payload = server_support._resolve_payload_from_observation(
                    pending.body,
                    f"{condition_id}-invalid",
                )
                if condition_id == "INPUT-F-04":
                    payload["answers"][0]["value"] = "not-a-boolean"
                elif condition_id == "INPUT-F-05":
                    payload["answers"][0]["value"] = {
                        "kind": "selected_choice",
                        "value": "unknown",
                    }
                else:
                    payload["answers"] = []
                response = await _post(pending, "resolve", payload)
                assert response.status_code == 422
                current = await _wait(pending, "pending")
                assert current["state_revision"] == pending.revision
                assert pending.provider.provider_calls == 1
                return _evidence(
                    condition_id,
                    surface_id,
                    ("pending", "pending"),
                    "http.validation_error.v1",
                    response,
                    pending.provider.provider_calls,
                )
            if condition_id in {"INPUT-F-07", "INPUT-F-08"}:
                payload = server_support._resolve_payload_from_observation(
                    pending.body,
                    f"{condition_id}-winner",
                )
                accepted = await _post(pending, "resolve", payload)
                assert accepted.status_code == 200
                repeat_payload = payload
                if condition_id == "INPUT-F-08":
                    repeat_payload = {
                        **payload,
                        "idempotency_key": "conflict",
                        "answers": [
                            {
                                "question_id": "continue",
                                "kind": "confirmation",
                                "provenance": "human",
                                "value": False,
                            }
                        ],
                    }
                repeated = await _post(pending, "resolve", repeat_payload)
                status = 200 if condition_id == "INPUT-F-07" else 409
                assert repeated.status_code == status
                if condition_id == "INPUT-F-07":
                    assert repeated.json()["idempotent"] is True
                else:
                    assert repeated.json()["code"] == "input.already_resolved"
                completed = await _poll(pending)
                replay = await _poll(pending)
                assert replay.json() == completed.json()
                assert pending.provider.provider_calls == 2
                return _evidence(
                    condition_id,
                    surface_id,
                    ("answered", "answered"),
                    (
                        "http.resolution_accepted.v1"
                        if condition_id == "INPUT-F-07"
                        else "http.already_resolved.v1"
                    ),
                    repeated,
                    pending.provider.provider_calls,
                )
            if condition_id == "INPUT-F-09":
                assert harness is not None
                harness.clock.advance(601)
                assert (await _wait(pending, "expired"))["state"] == "expired"
                payload = server_support._resolve_payload_from_observation(
                    pending.body,
                    "late",
                )
                response = await _post(pending, "resolve", payload)
                assert response.status_code == 410
                assert response.json()["code"] == "input.expired"
                assert pending.provider.provider_calls == 1
                return _evidence(
                    condition_id,
                    surface_id,
                    ("pending", "expired"),
                    "http.expired.v1",
                    response,
                    pending.provider.provider_calls,
                )
            if condition_id == "INPUT-F-11":
                request = pending.provider.request
                assert request is not None
                result = await pending.broker.supersede(
                    SupersedeInteractionScopeCommand(
                        actor=server_support._OWNER,
                        scope=InteractionExecutionScope(
                            run_id=request.origin.run_id,
                        ),
                        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    )
                )
                assert result.store_result.store_mutation_applied
                await _wait(pending, "superseded")
                payload = server_support._resolve_payload_from_observation(
                    pending.body,
                    "superseded",
                )
                response = await _post(pending, "resolve", payload)
                assert response.status_code == 409
                assert pending.provider.provider_calls == 1
                return _evidence(
                    condition_id,
                    surface_id,
                    ("pending", "superseded"),
                    "http.already_resolved.v1",
                    response,
                    pending.provider.provider_calls,
                )
            if condition_id == "INPUT-F-12":
                assert pending.body["state"] == "pending"
                assert pending.provider.provider_calls == 1
                return _evidence(
                    condition_id,
                    surface_id,
                    ("pending", "pending"),
                    "http.input_required.v1",
                    pending.required,
                    pending.provider.provider_calls,
                )
            raise AssertionError("condition is not owned by this boundary")


@pytest.mark.parametrize(
    ("condition_id", "surface_id"),
    _CASES,
    ids=tuple("|".join(case) for case in _CASES),
)
def test_server_failure(
    condition_id: str,
    surface_id: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise one exact reachable HTTP condition."""
    evidence = run(_observe(condition_id, surface_id))
    record_property(
        "failure_matrix_evidence", dumps([evidence], sort_keys=True)
    )
