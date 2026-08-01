"""Prove dormant conversation inputs fail before Responses dispatch."""

from collections.abc import Callable
from logging import Logger, getLogger

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from avalan.agent.execution import InteractionRuntime
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.server import (
    di_get_logger,
    di_get_orchestrator,
)
from avalan.server import (
    entities as server_entities,
)
from avalan.server.entities import (
    ResponsesRequest,
    ServerOutputRedactionSettings,
)
from avalan.server.routers import responses as responses_router

_DORMANT_PAYLOADS: tuple[tuple[str, object], ...] = (
    ("agent_lane", "lane-1"),
    ("agent_lanes", ["lane-1"]),
    ("background", True),
    ("branch", "branch-1"),
    ("branch_id", "branch-1"),
    ("checkpoint", "checkpoint-1"),
    ("checkpoint_id", "checkpoint-1"),
    ("compact", True),
    ("compaction", {"compact_threshold": 100}),
    ("context_management", {"compact_threshold": 100}),
    ("continuation", "continuation-1"),
    ("continuation_envelope", "sealed-continuation"),
    ("continuation_id", "continuation-1"),
    ("conversation", {"id": "conversation-1"}),
    ("conversation_envelope", "sealed-conversation"),
    ("conversation_handle", "conversation-1"),
    ("conversation_id", "conversation-1"),
    ("conversation_mode", "stateless"),
    ("envelope", "sealed-state"),
    ("execution_segment_id", "segment-1"),
    ("expected_head_revision", 1),
    ("head", "main"),
    ("head_id", "main"),
    ("head_revision", 1),
    ("idempotency", {"key": "request-1"}),
    ("idempotency_key", "request-1"),
    ("include", ["reasoning.encrypted_content"]),
    ("logical_turn_id", "turn-1"),
    ("model_call_id", "call-1"),
    ("named_head", "main"),
    ("named_head_id", "main"),
    ("named_head_revision", 1),
    ("parent_checkpoint_id", "checkpoint-0"),
    ("previous_response_id", "response-0"),
    ("provider_lane", "stored"),
    ("provider_lane_id", "lane-1"),
    ("provider_storage", "stored"),
    ("provisional_response_id", "provisional-1"),
    ("public_response_id", "response-1"),
    ("reasoning.context", "all_turns"),
    ("reasoning_context", "all_turns"),
    ("request_digest", "digest-1"),
    ("response_id", "response-1"),
    ("served_store", True),
    ("store", True),
    ("structured_input_continuation_id", "continuation-2"),
    ("task_id", "task-1"),
    ("tool_state", {"cycle": 1}),
    ("upstream_response_id", "upstream-1"),
)

_NORMALIZED_ALIASES = (
    ("Previous-Response-ID", "previous_response_id"),
    ("CONVERSATION MODE", "conversation_mode"),
    ("Idempotency.Key", "idempotency_key"),
    ("NamedHeadRevision", "named_head_revision"),
    ("BACKGROUND", "background"),
)


def _client(orchestrator: Orchestrator) -> TestClient:
    app = FastAPI()
    app.include_router(responses_router.router)
    app.dependency_overrides[di_get_logger] = lambda: getLogger()
    app.dependency_overrides[di_get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[
        responses_router._server_output_redaction_settings
    ] = lambda: ServerOutputRedactionSettings()
    return TestClient(app)


@pytest.mark.parametrize("stream", (False, True), ids=("body", "stream"))
@pytest.mark.parametrize(
    ("field", "value"),
    _DORMANT_PAYLOADS,
    ids=tuple(field for field, _value in _DORMANT_PAYLOADS),
)
def test_responses_reject_dormant_conversation_fields_before_dispatch(
    field: str,
    value: object,
    stream: bool,
    record_property: Callable[[str, object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every reserved top-level field for both response transports."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    expected_fields = {
        candidate
        for candidate, _value in _DORMANT_PAYLOADS
        if candidate != "reasoning.context"
    }
    assert expected_fields == set(
        server_entities.DORMANT_CONVERSATION_REQUEST_FIELDS
    )
    dispatch_calls = 0

    async def reject_dispatch(
        request: ResponsesRequest,
        logger: Logger,
        orchestrator: Orchestrator,
        interaction_runtime: InteractionRuntime | None = None,
    ) -> tuple[OrchestratorResponse, str, int]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise AssertionError("dormant conversation input reached dispatch")

    monkeypatch.setattr(responses_router, "orchestrate", reject_dispatch)
    orchestrator = object.__new__(Orchestrator)
    dormant_payload = (
        {"reasoning": {"context": value}}
        if field == "reasoning.context"
        else {field: value}
    )
    response = _client(orchestrator).post(
        "/responses",
        json={
            "input": "hello",
            "model": "probe-model",
            "stream": stream,
            **dormant_payload,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) == 1
    assert detail[0]["type"] == "conversation_continuity_dormant"
    assert detail[0]["ctx"] == {"field": field}
    assert dispatch_calls == 0


@pytest.mark.parametrize("stream", (False, True), ids=("body", "stream"))
@pytest.mark.parametrize(
    ("alias", "field"),
    _NORMALIZED_ALIASES,
    ids=tuple(alias for alias, _field in _NORMALIZED_ALIASES),
)
def test_responses_reject_normalized_conversation_field_aliases(
    alias: str,
    field: str,
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject punctuation, spacing, casing, and camel-case aliases."""
    dispatch_calls = 0

    async def reject_dispatch(
        request: ResponsesRequest,
        logger: Logger,
        orchestrator: Orchestrator,
        interaction_runtime: InteractionRuntime | None = None,
    ) -> tuple[OrchestratorResponse, str, int]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise AssertionError("normalized conversation alias reached dispatch")

    monkeypatch.setattr(responses_router, "orchestrate", reject_dispatch)
    orchestrator = object.__new__(Orchestrator)
    response = _client(orchestrator).post(
        "/responses",
        json={
            "input": "hello",
            "model": "probe-model",
            "stream": stream,
            alias: True,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) == 1
    assert detail[0]["type"] == "conversation_continuity_dormant"
    assert detail[0]["ctx"] == {"field": field}
    assert dispatch_calls == 0


def test_responses_reject_normalized_nested_reasoning_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a normalized alias for the nested reasoning context."""
    dispatch_calls = 0

    async def reject_dispatch(
        request: ResponsesRequest,
        logger: Logger,
        orchestrator: Orchestrator,
        interaction_runtime: InteractionRuntime | None = None,
    ) -> tuple[OrchestratorResponse, str, int]:
        nonlocal dispatch_calls
        dispatch_calls += 1
        raise AssertionError("nested reasoning alias reached dispatch")

    monkeypatch.setattr(responses_router, "orchestrate", reject_dispatch)
    orchestrator = object.__new__(Orchestrator)
    response = _client(orchestrator).post(
        "/responses",
        json={
            "input": "hello",
            "model": "probe-model",
            "reasoning": {"Con-Text": "all_turns"},
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) == 1
    assert detail[0]["type"] == "conversation_continuity_dormant"
    assert detail[0]["ctx"] == {"field": "reasoning.context"}
    assert dispatch_calls == 0


def test_one_shot_responses_preserve_intentional_extensions() -> None:
    """Keep the explicit extension namespace available in dormant mode."""
    payload = {
        "input": "hello",
        "extensions": {
            "task_input": {"version": "1", "handling": "attached"},
            "future_extension": {"value": 1},
        },
    }
    request = server_entities.ResponsesRequest.model_validate(payload)

    assert request.extensions is not None
    assert request.extensions.task_input is not None
    assert request.extensions.task_input.handling == "attached"
    assert request.extensions.model_extra == {"future_extension": {"value": 1}}


@pytest.mark.parametrize(
    "reasoning",
    ({"effort": "high"}, {"summary": "auto"}),
    ids=("effort", "summary"),
)
def test_one_shot_responses_preserve_non_stateful_reasoning(
    reasoning: dict[str, str],
) -> None:
    """Keep existing one-shot reasoning controls compatible."""
    payload = {"input": "hello", "reasoning": reasoning}
    request = ResponsesRequest.model_validate(payload)

    assert request.reasoning is not None
    assert request.reasoning.model_dump(exclude_none=True) == reasoning
