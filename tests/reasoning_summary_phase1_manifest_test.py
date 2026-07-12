"""Test the exact Phase 1 reasoning-summary acceptance inventory."""

from asyncio import run
from contextlib import redirect_stderr
from io import StringIO
from logging import getLogger
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from reasoning_summary_script_loader import load_reasoning_summary_script

from avalan.agent.orchestrator import Orchestrator
from avalan.cli.__main__ import CLI
from avalan.entities import (
    Modality,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
)
from avalan.model.reasoning import ReasoningSummaryCapabilityError
from avalan.server.entities import ReasoningConfig, ResponsesRequest
from avalan.server.routers import orchestrate
from avalan.server.routers import responses as responses_router

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE1_DIMENSION = "phase 1 typed request and capability enforcement"
_PHASE1_NODE_IDS = (
    (
        "tests/reasoning_summary_phase1_manifest_test.py::"
        "test_phase1_acceptance_manifest_pins_exact_catalog"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_reasoning_summary_mode_and_settings_contract"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_engine_normalizes_summary_once_and_preserves_typed_values"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_private_capable_adapter_crosses_shared_modality_choke"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_shared_modality_choke_rejects_before_adapter_invocation"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_request_capability_is_typed_dormant_and_omission_safe"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_direct_model_entries_reject_before_provider_call"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_direct_vendor_clients_reject_before_request_side_effects"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_third_party_vendor_model_identity_is_generic"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_openai_provider_identity_is_client_derived_and_zero_call"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_hosted_provider_omission_keeps_exact_dispatch_shape"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_raw_vendor_omission_keeps_exact_provider_payloads"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_local_provider_omissions_reach_unchanged_dispatch_shapes"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_local_omission_keeps_exact_backend_request_shapes"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_shared_local_omission_invokes_adapter_once"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_private_capable_openai_adapter_forwards_without_fallback_retry"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_explicit_local_provider_error_and_zero_adapter_calls"
    ),
    (
        "tests/reasoning_summary_phase1_test.py::"
        "test_raw_reasoning_summary_preserves_authority_rejection"
    ),
    (
        "tests/server/remote_container_test.py::"
        "RemoteContainerProfileSelectionTestCase::"
        "test_rejects_authority_nested_inside_reasoning_summary"
    ),
    (
        "tests/server/responses_test.py::ResponsesEndpointTestCase::"
        "test_response_endpoint_preserves_summary_authority_rejection"
    ),
    (
        "tests/agent/loader_test.py::LoadJsonOrchestratorVariantsTestCase::"
        "test_run_reasoning_settings_from_file"
    ),
    (
        "tests/agent/loader_test.py::LoadJsonOrchestratorVariantsTestCase::"
        "test_run_reasoning_summary_override_preserves_toml_siblings"
    ),
    (
        "tests/agent/loader_test.py::LoadJsonOrchestratorVariantsTestCase::"
        "test_run_reasoning_summary_file_entry_points_match"
    ),
    (
        "tests/agent/loader_test.py::LoadJsonOrchestratorVariantsTestCase::"
        "test_blueprint_reasoning_literal_round_trips_through_loader"
    ),
    (
        "tests/agent/orchestrator_test.py::OrchestratorCallTestCase::"
        "test_call_applies_presence_aware_generation_options_override"
    ),
    (
        "tests/agent/loader_test.py::LoaderFromFileTestCase::"
        "test_runtime_container_delegates_to_agent_envelope_loader"
    ),
    (
        "tests/agent/loader_test.py::LoaderFromFileTestCase::"
        "test_runtime_container_omission_preserves_legacy_signature"
    ),
    (
        "tests/cli/get_orchestrator_settings_test.py::"
        "GetOrchestratorSettingsTestCase::test_reasoning_settings"
    ),
    (
        "tests/cli/get_orchestrator_settings_test.py::"
        "GetOrchestratorSettingsTestCase::"
        "test_display_reasoning_does_not_request_summary"
    ),
    (
        "tests/cli/agent_test.py::CliAgentInitTestCase::"
        "test_agent_init_run_options_output"
    ),
    (
        "tests/cli/agent_test.py::CliAgentRunTestCase::"
        "test_run_watch_reloads_when_file_changes"
    ),
    (
        "tests/cli/model_test.py::CliModelRunTestCase::"
        "test_model_run_reasoning_effort_cli"
    ),
    (
        "tests/cli/main_test.py::CliCallTestCase::"
        "test_call_maps_unsupported_reasoning_summary_safely"
    ),
    (
        "tests/cli/main_test.py::CliCallTestCase::"
        "test_non_text_reasoning_summary_exits_before_model_loading"
    ),
    (
        "tests/model/modality_registry_test.py::"
        "test_get_operation_from_arguments_maps_summary_only"
    ),
    (
        "tests/model/modality_registry_test.py::"
        "test_non_text_reasoning_summary_rejects_before_handler_dispatch"
    ),
    (
        "tests/reasoning_summary_phase1_manifest_test.py::"
        "test_phase1_cli_acceptance_is_typed_scoped_and_strict"
    ),
    (
        "tests/reasoning_summary_phase1_manifest_test.py::"
        "test_phase1_responses_ingress_is_typed_strict_and_authority_safe"
    ),
    (
        "tests/reasoning_summary_phase1_manifest_test.py::"
        "test_phase1_responses_router_preserves_explicit_field_authority"
    ),
    (
        "tests/reasoning_summary_phase1_manifest_test.py::"
        "test_phase1_responses_unsupported_error_precedes_stream_headers"
    ),
)


def test_phase1_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.dimensions[_PHASE1_DIMENSION] == _PHASE1_NODE_IDS
    assert len(manifest.node_ids) == len(set(manifest.node_ids))


def test_phase1_cli_acceptance_is_typed_scoped_and_strict() -> None:
    parser = CLI._create_parser(
        default_device="cpu",
        cache_dir="/tmp",
        default_locales_path="/tmp",
        default_locale="en_US",
    )
    for mode in ReasoningSummaryMode:
        model = parser.parse_args(
            [
                "model",
                "run",
                "model-id",
                "--reasoning-summary",
                mode.value,
            ]
        )
        agent = parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--run-reasoning-summary",
                mode.value,
            ]
        )
        assert model.reasoning_summary == mode.value
        assert agent.run_reasoning_summary == mode.value

    for command in (
        ["model", "run", "model-id"],
        ["agent", "run", "spec.toml"],
        ["agent", "init"],
    ):
        stderr = StringIO()
        with redirect_stderr(stderr), pytest.raises(SystemExit) as caught:
            parser.parse_args([*command, "--reasoning-summary", "verbose"])
        assert caught.value.code == 2
        assert "invalid choice" in stderr.getvalue()

    stderr = StringIO()
    with redirect_stderr(stderr), pytest.raises(SystemExit) as caught:
        parser.parse_args(
            [
                "model",
                "run",
                "model-id",
                "--reasoning-summary",
                "auto",
                "--no-reasoning",
            ]
        )
    assert caught.value.code == 2
    assert "not allowed with argument" in stderr.getvalue()

    for modality in Modality:
        if modality is Modality.TEXT_GENERATION:
            continue
        stderr = StringIO()
        with redirect_stderr(stderr), pytest.raises(SystemExit) as caught:
            parser.parse_args(
                [
                    "model",
                    "run",
                    "model-id",
                    "--modality",
                    modality.value,
                    "--reasoning-summary",
                    "auto",
                ]
            )
        assert caught.value.code == 2
        assert "requires --modality text_generation" in stderr.getvalue()

    for command in (
        ["agent", "serve", "spec.toml"],
        ["agent", "proxy", "spec.toml"],
        [
            "agent",
            "message",
            "search",
            "spec.toml",
            "--function",
            "l2_distance",
            "--id",
            "agent-id",
            "--participant",
            "participant-id",
            "--session",
            "session-id",
        ],
    ):
        stderr = StringIO()
        with redirect_stderr(stderr), pytest.raises(SystemExit) as caught:
            parser.parse_args([*command, "--reasoning-summary", "auto"])
        assert caught.value.code == 2
        assert "unrecognized arguments" in stderr.getvalue()


def test_phase1_responses_ingress_is_typed_strict_and_authority_safe() -> None:
    for mode in ReasoningSummaryMode:
        request = ResponsesRequest.model_validate(
            {
                "input": "hi",
                "reasoning": {"summary": mode.value},
            }
        )
        assert request.reasoning is not None
        assert request.reasoning.summary is mode
        assert request.reasoning.model_fields_set == {"summary"}

    for invalid in (
        "unknown",
        1,
        True,
        {"value": "auto"},
        ["auto"],
    ):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(
                {
                    "input": "hi",
                    "reasoning": {"summary": invalid},
                }
            )

    with pytest.raises(ValidationError):
        ReasoningConfig.model_validate({"summary": "auto", "unexpected": True})
    for reasoning in (
        {"summary": {"mode": "auto"}},
        {"summary": {"sandboxProfile": "unsafe"}},
        {"summary": "auto", "sandboxProfile": "unsafe"},
    ):
        with pytest.raises(ValidationError, match="runtime authority"):
            ResponsesRequest.model_validate(
                {
                    "input": "hi",
                    "reasoning": reasoning,
                }
            )


class _RecordingOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        _messages: object,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)
        return object()


def test_phase1_responses_router_preserves_explicit_field_authority() -> None:
    logger = getLogger(__name__)
    recorder = _RecordingOrchestrator()
    summary_request = ResponsesRequest(
        input="hi",
        reasoning=ReasoningConfig(
            summary=ReasoningSummaryMode.DETAILED,
        ),
    )
    run(orchestrate(summary_request, logger, cast(Any, recorder)))
    summary_call = recorder.calls[-1]
    summary_settings = cast(Any, summary_call["settings"])
    assert summary_settings.reasoning == ReasoningSettings(
        summary=ReasoningSummaryMode.DETAILED
    )
    assert summary_call["generation_options_override"] == {
        "reasoning": {"summary": ReasoningSummaryMode.DETAILED}
    }

    effort_request = ResponsesRequest(
        input="hi",
        reasoning=ReasoningConfig(effort=ReasoningEffort.HIGH),
    )
    run(orchestrate(effort_request, logger, cast(Any, recorder)))
    effort_call = recorder.calls[-1]
    effort_settings = cast(Any, effort_call["settings"])
    assert effort_settings.reasoning == ReasoningSettings(
        effort=ReasoningEffort.HIGH
    )
    assert effort_call["generation_options_override"] == {
        "reasoning": {"effort": ReasoningEffort.HIGH}
    }

    null_summary_request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "reasoning": {"summary": None},
        }
    )
    run(orchestrate(null_summary_request, logger, cast(Any, recorder)))
    null_summary_call = recorder.calls[-1]
    assert null_summary_call["generation_options_override"] == {
        "reasoning": {"summary": None}
    }

    empty_reasoning_request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "reasoning": {},
        }
    )
    run(orchestrate(empty_reasoning_request, logger, cast(Any, recorder)))
    empty_reasoning_call = recorder.calls[-1]
    assert "generation_options_override" not in empty_reasoning_call

    omitted_request = ResponsesRequest(input="hi")
    run(orchestrate(omitted_request, logger, cast(Any, recorder)))
    omitted_call = recorder.calls[-1]
    omitted_settings = cast(Any, omitted_call["settings"])
    assert omitted_settings.reasoning == ReasoningSettings()
    assert "generation_options_override" not in omitted_call


def test_phase1_responses_unsupported_error_precedes_stream_headers() -> None:
    logger = getLogger(__name__)
    orchestrator = object.__new__(Orchestrator)
    orchestrator._model_ids = {"model"}
    error = ReasoningSummaryCapabilityError(
        provider="bedrock",
        requested_mode=ReasoningSummaryMode.AUTO,
    )
    for stream in (False, True):
        request = ResponsesRequest(
            input="hi",
            stream=stream,
            reasoning=ReasoningConfig(summary=ReasoningSummaryMode.AUTO),
        )
        with (
            patch.object(
                responses_router,
                "orchestrate",
                new=AsyncMock(side_effect=error),
            ) as orchestration,
            pytest.raises(HTTPException) as caught,
        ):
            run(
                responses_router.create_response(
                    request,
                    logger,
                    orchestrator,
                )
            )

        assert caught.value.status_code == 400
        assert cast(Any, caught.value.detail) == {
            "code": "reasoning_summary_unsupported",
            "message": (
                "Provider 'bedrock' does not support reasoning summary "
                "mode 'auto'"
            ),
            "provider": "bedrock",
            "requested_mode": "auto",
        }
        orchestration.assert_awaited_once()
