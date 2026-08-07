"""Verify the live conformance harness without provider network or cost."""

from argparse import Namespace
from asyncio import CancelledError
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import dumps, loads
from pathlib import Path
from sys import modules
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import pytest
from httpx import AsyncClient, Request
from openai.types.responses import (
    CompactedResponse,
    Response,
    ResponseOutputMessage,
)
from openai.types.responses.parsed_response import (
    ParsedResponse,
    ParsedResponseFunctionToolCall,
)


def _load_live_module() -> Any:
    path = (
        Path(__file__).parents[2] / "scripts/conversation_live_conformance.py"
    )
    specification = spec_from_file_location(
        "conversation_live_conformance",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = module_from_spec(specification)
    modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


live = _load_live_module()

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)
FIXTURE_DIRECTORY = Path("tests/fixtures/conversation")
PROVIDER_EVIDENCE = FIXTURE_DIRECTORY / "provider_evidence.phase12.json"
ACTIVATION_MANIFEST = (
    FIXTURE_DIRECTORY / "activation_preflight_manifest.phase12.json"
)
LIVE_RESULTS = FIXTURE_DIRECTORY / "live_conformance_results.phase12.json"


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic harness tests on asyncio only."""
    return "asyncio"


async def _clock() -> datetime:
    return NOW


def _config(
    *,
    family: Any = live.LiveProviderFamily.OPENAI,
    endpoint: str | None = None,
    **changes: object,
) -> Any:
    azure = family is live.LiveProviderFamily.AZURE_OPENAI
    values: dict[str, object] = {
        "provider_family": family,
        "endpoint": (
            endpoint
            or (
                "https://resource.openai.azure.com/openai/v1"
                if azure
                else "https://api.openai.com/v1"
            )
        ),
        "api_form": (
            "azure-openai-v1-preview" if azure else "openai_responses_v1"
        ),
        "provider_api_revision": (
            "azure-openai-v1-preview" if azure else "openapi-2.3.0"
        ),
        "model_or_deployment": "gpt-5",
        "model_or_deployment_revision": "gpt-5",
        "api_key": "sk-live-conformance-fixture",
        "command_authority": True,
        "environment_authority": "authorize-phase12-live-conformance",
        "command_cost_acknowledgement": True,
        "environment_cost_acknowledgement": "accept-phase12-provider-costs",
    }
    values.update(changes)
    return live.LiveConformanceConfig(**cast(Any, values))


def _usage() -> dict[str, object]:
    return {
        "input_tokens": 7,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 10,
    }


def _reasoning(identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "reasoning",
        "summary": [],
        "encrypted_content": f"opaque-{identifier}",
        "status": "completed",
    }


def _function(
    identifier: str,
    call_id: str,
    *,
    value: str = "1943",
) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "function_call",
        "call_id": call_id,
        "name": "phase12_probe",
        "arguments": dumps({"value": value}, separators=(",", ":")),
        "status": "completed",
    }


def _message(identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "phase12 marker",
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _compaction(identifier: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "compaction",
        "encrypted_content": f"opaque-{identifier}",
        "created_by": "provider-compact",
    }


def _input_message(text: str = "phase12 retained input") -> dict[str, object]:
    return {
        "content": [{"text": text, "type": "input_text"}],
        "role": "user",
        "type": "message",
    }


def _response(
    identifier: str,
    output: list[dict[str, object]],
    *,
    context: str | None,
    previous_response_id: str | None = None,
) -> Response:
    return Response.model_validate(
        {
            "id": identifier,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": "gpt-5",
            "output": output,
            "parallel_tool_calls": False,
            "previous_response_id": previous_response_id,
            "reasoning": {"context": context} if context is not None else None,
            "temperature": None,
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "truncation": "disabled",
            "usage": _usage(),
        }
    )


def _compacted_response(
    retained_input: list[dict[str, object]] | None = None,
) -> CompactedResponse:
    retained = retained_input or [_input_message()]
    response = CompactedResponse.model_validate(
        {
            "id": "cmp_phase12",
            "object": "response.compaction",
            "created_at": 1,
            "output": [
                {
                    "id": "cmp_terminal",
                    "type": "compaction",
                    "encrypted_content": "opaque-cmp_terminal",
                },
            ],
            "usage": _usage(),
        }
    )
    retained_messages = [
        ResponseOutputMessage.model_construct(
            id=f"msg_retained_{index}",
            content=cast(Any, deepcopy(message["content"])),
            role=cast(Any, "user"),
            status="completed",
            type="message",
        )
        for index, message in enumerate(retained)
    ]
    response.output[:0] = cast(Any, retained_messages)
    return response


def _parsed_tool_response() -> ParsedResponse[object]:
    response = _response(
        "resp_parsed_tool",
        [
            _reasoning("rs_parsed_tool"),
            _function("fc_parsed_tool", "call_parsed_tool", value="2505"),
        ],
        context="current_turn",
    )
    parsed = ParsedResponse[object].model_validate(
        response.model_dump(mode="json")
    )
    function_call = cast(ParsedResponseFunctionToolCall, parsed.output[1])
    assert isinstance(function_call, ParsedResponseFunctionToolCall)
    function_call.parsed_arguments = {"value": "2505"}
    return parsed


class _Event:
    """Expose one typed-enough stream event discriminator."""

    def __init__(self, event_type: str) -> None:
        self.type = event_type


class _Stream:
    """Yield deterministic event structure and one final response."""

    def __init__(
        self,
        response: Response,
        record_http_request: Callable[[], Awaitable[None]],
    ) -> None:
        self.final_response = ParsedResponse[object].model_validate(
            response.model_dump(mode="json")
        )
        for item in self.final_response.output:
            if isinstance(item, ParsedResponseFunctionToolCall):
                item.parsed_arguments = {"value": "2505"}
        self._events = iter(
            (
                _Event("response.output_item.done"),
                _Event("response.completed"),
            )
        )
        self._record_http_request = record_http_request

    async def __aenter__(self) -> "_Stream":
        await self._record_http_request()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    def __aiter__(self) -> AsyncIterator[_Event]:
        return self

    async def __anext__(self) -> _Event:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def get_final_response(self) -> Response:
        return self.final_response


class _FakeResponses:
    """Return the complete matrix while recording typed request arguments."""

    def __init__(
        self,
        *,
        http_client: AsyncClient,
        requires_explicit_encrypted_content: bool = False,
    ) -> None:
        self.http_client = http_client
        self.requires_explicit_encrypted_content = (
            requires_explicit_encrypted_content
        )
        self.create_calls: list[dict[str, object]] = []
        self.create_results: list[Response] = []
        self.compact_calls: list[dict[str, object]] = []
        self.compact_results: list[CompactedResponse] = []
        self.retrieve_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_results: list[_Stream] = []
        self.retrieve_result: object | None = None
        self.retrieve_results: list[object] = []
        self.delete_failures: list[BaseException] = []
        self.delete_failure: BaseException | None = None
        self.next_http_request_repetitions = 1
        self.next_http_path_override: str | None = None
        self.suppress_next_http_request = False

    async def _record_http_request(self, method: str, path: str) -> None:
        """Invoke configured request hooks without provider material."""
        if self.suppress_next_http_request:
            self.suppress_next_http_request = False
            return
        selected_path = self.next_http_path_override or path
        self.next_http_path_override = None
        repetitions = self.next_http_request_repetitions
        self.next_http_request_repetitions = 1
        for _ in range(repetitions):
            request = Request(
                method, f"https://provider.invalid{selected_path}"
            )
            for hook in self.http_client.event_hooks["request"]:
                await hook(request)

    def _returned_reasoning(
        self,
        identifier: str,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        """Model Azure's documented explicit encrypted-content include."""
        item = _reasoning(identifier)
        if self.requires_explicit_encrypted_content and request.get(
            "include"
        ) != ["reasoning.encrypted_content"]:
            item.pop("encrypted_content")
        return item

    async def create(self, **kwargs: object) -> Response:
        await self._record_http_request("POST", "/v1/responses")
        self.create_calls.append(kwargs)
        index = len(self.create_calls)
        responses = {
            1: _response(
                "resp_stateless_tool",
                [
                    self._returned_reasoning("rs_tool", kwargs),
                    _function("fc_tool", "call_tool"),
                ],
                context="current_turn",
            ),
            2: _response(
                "resp_stateless_all",
                [_message("msg_all")],
                context="all_turns",
            ),
            3: _response(
                "resp_inline",
                [
                    _compaction("cmp_inline"),
                    _message("msg_inline"),
                ],
                context="current_turn",
            ),
            4: _response(
                "resp_compact_replay",
                [_message("msg_compact")],
                context=None,
            ),
            5: _response(
                "resp_stored_first",
                [_message("msg_stored_first")],
                context="current_turn",
            ),
            6: _response(
                "resp_stored_chain",
                [_message("msg_stored_chain")],
                context="all_turns",
                previous_response_id="resp_stored_first",
            ),
        }
        response = responses[index]
        result = (
            ParsedResponse[object].model_validate(
                response.model_dump(mode="json")
            )
            if index == 5
            else response
        )
        self.create_results.append(result)
        return result

    async def compact(self, **kwargs: object) -> CompactedResponse:
        await self._record_http_request("POST", "/v1/responses/compact")
        self.compact_calls.append(kwargs)
        raw_input = kwargs.get("input")
        assert type(raw_input) is list
        result = _compacted_response(
            deepcopy(cast(list[dict[str, object]], raw_input))
        )
        self.compact_results.append(result)
        return result

    async def retrieve(self, response_id: str) -> Response:
        await self._record_http_request(
            "GET",
            f"/v1/responses/{response_id}",
        )
        self.retrieve_calls.append(response_id)
        if self.retrieve_result is not None:
            result = self.retrieve_result
        else:
            response = _response(
                "resp_stored_chain",
                [_message("msg_stored_chain")],
                context="all_turns",
                previous_response_id="resp_stored_first",
            )
            result = ParsedResponse[object].model_validate(
                response.model_dump(mode="json")
            )
        self.retrieve_results.append(result)
        return cast(Response, result)

    async def delete(self, response_id: str) -> None:
        await self._record_http_request(
            "DELETE",
            f"/v1/responses/{response_id}",
        )
        self.delete_calls.append(response_id)
        if self.delete_failures:
            raise self.delete_failures.pop(0)
        if self.delete_failure is not None:
            raise self.delete_failure

    def stream(self, **kwargs: object) -> _Stream:
        self.stream_calls.append(kwargs)
        result = _Stream(
            _response(
                "resp_stream",
                [
                    self._returned_reasoning("rs_stream", kwargs),
                    _function("fc_stream", "call_stream", value="2505"),
                ],
                context="current_turn",
            ),
            lambda: self._record_http_request("POST", "/v1/responses"),
        )
        self.stream_results.append(result)
        return result


class _FakeClient:
    """Expose deterministic Responses resources without network access."""

    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs
        self.max_retries = kwargs.get("max_retries")
        http_client = kwargs.get("http_client")
        assert isinstance(http_client, AsyncClient)
        self.responses = _FakeResponses(
            http_client=http_client,
            requires_explicit_encrypted_content=(
                kwargs.get("base_url")
                == "https://resource.openai.azure.com/openai/v1"
            ),
        )
        self.closed = False
        self.close_failure: BaseException | None = None
        self.instances.append(self)

    async def close(self) -> None:
        self.closed = True
        await cast(AsyncClient, self.arguments["http_client"]).aclose()
        if self.close_failure is not None:
            raise self.close_failure


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")


def _canonicalize(payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("canonical_digest")
    canonical = dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = cast(dict[str, object], payload["canonical_digest"])
    digest["value"] = sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    "family",
    (
        live.LiveProviderFamily.OPENAI,
        live.LiveProviderFamily.AZURE_OPENAI,
    ),
)
async def test_full_fake_sdk_matrix_is_typed_redacted_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    family: Any,
) -> None:
    """Exercise every native case with a deterministic fake SDK client."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    config = _config(family=family)

    receipt = await live.run_live_conformance(
        config,
        transport_factory=live.OpenAISdkLiveConformanceTransport,
        clock=_clock,
    )

    assert receipt.completed_cases == live._EXECUTION_ORDER
    assert receipt.provider_family is family
    payload = receipt.redacted_payload()
    assert payload["opaque_payloads_logged"] is False
    assert payload["production_activation_granted"] is False
    assert "api_key" not in dumps(payload)
    assert config.api_key not in repr(config)
    client = _FakeClient.instances[-1]
    assert client.closed
    assert set(client.arguments) == {
        "api_key",
        "base_url",
        "default_query",
        "http_client",
        "max_retries",
    }
    assert client.arguments["api_key"] == config.api_key
    assert client.arguments["base_url"] == config.endpoint
    assert client.arguments["default_query"] == (
        {"api-version": "preview"}
        if family is live.LiveProviderFamily.AZURE_OPENAI
        else None
    )
    assert isinstance(client.arguments["http_client"], AsyncClient)
    assert client.arguments["max_retries"] == 0
    assert client.max_retries == 0
    assert payload["api_form"] == config.api_form
    assert payload["provider_api_revision"] == config.provider_api_revision
    assert len(cast(str, payload["structural_observations_digest"])) == 64
    assert payload["accounting"] == {
        "logical_operation_count": 11,
        "logical_operation_counts": {
            "compact": 1,
            "create_or_stream": 7,
            "delete": 2,
            "retrieve": 1,
        },
        "http_request_count": 11,
        "http_request_counts": {
            "compact": 1,
            "create_or_stream": 7,
            "delete": 2,
            "retrieve": 1,
            "unexpected": 0,
        },
        "sdk_configured_max_retries": 0,
        "observed_sdk_retry_count": 0,
        "unexpected_request_count": 0,
        "request_path_class_mismatch_count": 0,
        "cleanup_attempted": True,
        "cleanup_completed": True,
        "cleanup_delete_logical_operation_count": 0,
        "cleanup_delete_http_request_count": 0,
        "cleanup_pending_reference_count": 0,
        "client_close_completed": True,
        "successful_matrix_expected_counts_match": True,
    }
    assert "resource.openai.azure.com" not in dumps(payload)
    assert len(client.responses.create_calls) == 6
    assert isinstance(client.responses.create_results[4], ParsedResponse)
    assert type(client.responses.create_results[4]) is not Response
    assert len(client.responses.compact_calls) == 1
    compact_result_types = tuple(
        item.type for item in client.responses.compact_results[0].output
    )
    assert compact_result_types == ("message", "compaction")
    assert client.responses.retrieve_calls == ["resp_stored_chain"]
    assert isinstance(client.responses.retrieve_results[0], ParsedResponse)
    assert client.responses.delete_calls == [
        "resp_stored_chain",
        "resp_stored_first",
    ]
    assert len(client.responses.stream_calls) == 1
    assert isinstance(
        client.responses.stream_results[0].final_response,
        ParsedResponse,
    )
    assert (
        type(client.responses.stream_results[0].final_response) is not Response
    )
    assert isinstance(
        client.responses.stream_results[0].final_response,
        Response,
    )
    encrypted_reasoning_calls = [
        *client.responses.create_calls[:4],
        *client.responses.stream_calls,
    ]
    if family is live.LiveProviderFamily.AZURE_OPENAI:
        assert all(
            call.get("include") == ["reasoning.encrypted_content"]
            for call in encrypted_reasoning_calls
        )
    else:
        assert all(
            call.get("include") is live.omit
            for call in encrypted_reasoning_calls
        )
    assert all(
        "include" not in call for call in client.responses.create_calls[4:]
    )
    assert "include" not in client.responses.compact_calls[0]
    all_calls = [
        *client.responses.create_calls,
        *client.responses.compact_calls,
        *client.responses.stream_calls,
    ]
    assert all("extra_body" not in call for call in all_calls)
    assert client.responses.create_calls[0]["store"] is False
    assert client.responses.create_calls[4]["store"] is True
    assert (
        client.responses.create_calls[5]["previous_response_id"]
        == "resp_stored_first"
    )
    assert client.responses.create_calls[2]["context_management"] == [
        {"type": "compaction", "compact_threshold": 2_048}
    ]
    assert client.responses.create_calls[2]["max_output_tokens"] == 512
    all_turns_replay = cast(
        list[dict[str, object]],
        client.responses.create_calls[1]["input"],
    )
    assert [item["type"] for item in all_turns_replay] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert all("status" not in item for item in all_turns_replay)
    assert all_turns_replay[0]["id"] == "rs_tool"
    assert all_turns_replay[0]["encrypted_content"] == "opaque-rs_tool"
    assert all_turns_replay[1]["id"] == "fc_tool"
    assert all_turns_replay[1]["call_id"] == "call_tool"
    assert all_turns_replay[2]["call_id"] == "call_tool"
    replay_input = cast(
        list[dict[str, object]],
        client.responses.create_calls[3]["input"],
    )
    compact_input = cast(
        list[dict[str, object]],
        client.responses.compact_calls[0]["input"],
    )
    assert replay_input[:1] == compact_input
    assert replay_input[1] == {
        "id": "cmp_terminal",
        "encrypted_content": "opaque-cmp_terminal",
        "type": "compaction",
    }
    retained_dump = (
        client.responses.compact_results[0]
        .output[0]
        .model_dump(
            mode="json",
            exclude_none=True,
            warnings=False,
        )
    )
    assert set(retained_dump) == {"content", "id", "role", "status", "type"}
    assert "id" not in replay_input[0]
    assert "status" not in replay_input[0]


async def test_azure_missing_explicit_encrypted_include_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce Azure omission and stop at the first structural gate."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(
        live,
        "_encrypted_reasoning_include",
        lambda config: live.omit,
    )

    with pytest.raises(
        live.LiveConformanceAssertionError,
        match="live conformance case failed: stateless_current_turn_tool",
    ):
        await live.run_live_conformance(
            _config(family=live.LiveProviderFamily.AZURE_OPENAI),
            transport_factory=live.OpenAISdkLiveConformanceTransport,
            clock=_clock,
        )

    client = _FakeClient.instances[-1]
    assert client.closed
    assert len(client.responses.create_calls) == 1
    assert client.responses.create_calls[0].get("include") is live.omit


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_http_request",
        "repeated_http_request",
        "unexpected_http_path",
        "mismatched_http_path_class",
        "configured_retry_drift",
    ),
)
async def test_generated_transport_accounting_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Reject missing, repeated, unknown, mismatched, or retrying activity."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    client = _FakeClient.instances[-1]
    match mutation:
        case "missing_http_request":
            client.responses.suppress_next_http_request = True
        case "repeated_http_request":
            client.responses.next_http_request_repetitions = 2
        case "unexpected_http_path":
            client.responses.next_http_path_override = "/private-canary"
        case "mismatched_http_path_class":
            client.responses.next_http_path_override = "/v1/responses/compact"
        case "configured_retry_drift":
            transport._accounting.set_sdk_configured_max_retries(1)
        case _:
            raise AssertionError("unhandled accounting mutation")

    with pytest.raises(live.LiveConformanceAccountingError) as captured:
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: transport,
            clock=_clock,
        )

    assert captured.value.__cause__ is None
    assert "private-canary" not in str(captured.value)
    assert client.closed
    snapshot = transport.final_accounting()
    assert snapshot.cleanup_attempted
    assert snapshot.cleanup_completed


async def test_request_hook_retains_no_headers_query_body_ids_or_content() -> (
    None
):
    """Keep the HTTP request hook limited to fixed content-free classes."""
    tracker = live._TransportAccounting()
    tracker.set_sdk_configured_max_retries(0)
    with tracker.logical_operation(live._LiveOperationClass.DELETE):
        await tracker.record_http_request(
            Request(
                "DELETE",
                "https://provider.invalid/v1/responses/private-id"
                "?api-key=private-query",
                headers={"authorization": "private-header"},
                content=b"private-body-content",
            )
        )
    tracker.begin_cleanup()
    tracker.finish_cleanup(
        completed=True,
        pending_reference_count=0,
        client_close_completed=True,
    )

    serialized = dumps(tracker.snapshot().redacted_payload(), sort_keys=True)
    for canary in (
        "private-id",
        "private-query",
        "private-header",
        "private-body-content",
    ):
        assert canary not in serialized
        assert canary not in repr(tracker)


async def test_actual_sdk_client_is_configured_with_zero_retries() -> None:
    """Pin the tracked transport's actual AsyncOpenAI retry configuration."""
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    assert transport._client.max_retries == 0

    await transport.aclose()

    accounting = transport.final_accounting()
    assert accounting.sdk_configured_max_retries == 0
    assert accounting.cleanup_attempted
    assert accounting.cleanup_completed


@pytest.mark.parametrize(
    "changes",
    (
        {"provider_family": cast(Any, "openai_compatible")},
        {"command_authority": False},
        {"environment_authority": "wrong"},
        {"command_cost_acknowledgement": False},
        {"environment_cost_acknowledgement": "wrong"},
        {"api_key": "placeholder"},
        {"api_key": " secret"},
        {"model_or_deployment": ""},
        {"model_or_deployment_revision": "bad revision"},
        {"endpoint": "https://compatible.example/v1"},
    ),
)
def test_preflight_rejects_authority_identity_and_credentials(
    changes: dict[str, object],
) -> None:
    """Fail closed on generic, unauthorized, placeholder, or drifted input."""
    with pytest.raises(live.LiveConformancePreflightError):
        _config(**cast(Any, changes))


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://resource.openai.azure.com/openai/v1",
        "https://resource.openai.azure.com/openai/deployments/name",
        "https://resource.openai.azure.com/openai/v1/",
        "https://resource.openai.azure.com/openai/v1?api-version=preview",
        "https://resource.azure.com/openai/v1",
        "https://user@resource.openai.azure.com/openai/v1",
        "https://resource.openai.azure.com:443/openai/v1",
    ),
)
def test_azure_wrong_api_forms_are_rejected(endpoint: str) -> None:
    """Require the exact documented Azure native v1 base URL."""
    with pytest.raises(live.LiveConformancePreflightError):
        _config(
            family=live.LiveProviderFamily.AZURE_OPENAI,
            endpoint=endpoint,
        )


async def test_partial_matrix_and_stale_fixture_precede_transport_factory(
    tmp_path: Path,
) -> None:
    """Reject partial or stale plans before a client can be constructed."""
    constructed = 0

    def factory(
        config: Any,
    ) -> Any:
        nonlocal constructed
        del config
        constructed += 1
        raise AssertionError("factory must not be reached")

    with pytest.raises(live.LiveConformancePreflightError):
        await live.run_live_conformance(
            _config(),
            cases=live._EXECUTION_ORDER[:-1],
            transport_factory=factory,
            clock=_clock,
        )
    assert constructed == 0

    async def naive_clock() -> datetime:
        return datetime(2026, 8, 3, 12)

    with pytest.raises(live.LiveConformancePreflightError):
        await live.run_live_conformance(
            _config(),
            transport_factory=factory,
            clock=naive_clock,
        )
    assert constructed == 0

    evidence = cast(
        dict[str, object],
        loads(PROVIDER_EVIDENCE.read_text(encoding="utf-8")),
    )
    evidence["accessed_at"] = "2026-01-01"
    for provider in cast(list[dict[str, object]], evidence["providers"]):
        for source in cast(list[dict[str, object]], provider["sources"]):
            source["accessed_at"] = "2026-01-01"
    _canonicalize(evidence)
    stale = tmp_path / PROVIDER_EVIDENCE.name
    _write_json(stale, evidence)
    with pytest.raises(live.LiveConformancePreflightError):
        await live.run_live_conformance(
            _config(),
            provider_evidence_path=stale,
            activation_manifest_path=ACTIVATION_MANIFEST,
            transport_factory=factory,
            clock=_clock,
        )
    assert constructed == 0


def test_evidence_and_inactive_activation_integrity_is_exact(
    tmp_path: Path,
) -> None:
    """Validate links and reject any attempted active production row."""
    evidence_digest, manifest_digest = live.validate_provider_evidence(
        PROVIDER_EVIDENCE,
        ACTIVATION_MANIFEST,
        today=date(2026, 8, 3),
    )
    assert (
        evidence_digest
        == "7076e4ee05f29d7dc759f01af8eb6f657c3ea1eb075de57d8cd4f140f41ad4ef"
    )
    assert (
        manifest_digest
        == "975f79f7a30d304fe0598206fe2ba59ee65911f14bd072585f81c3f800af0d4d"
    )

    manifest = cast(
        dict[str, object],
        loads(ACTIVATION_MANIFEST.read_text(encoding="utf-8")),
    )
    manifest["activation_state"] = "active"
    manifest["production_dispatch_enabled"] = True
    manifest["active_production_rows"] = [{"provider_family": "openai"}]
    _canonicalize(manifest)
    active = tmp_path / ACTIVATION_MANIFEST.name
    _write_json(active, manifest)
    with pytest.raises(live.LiveConformancePreflightError):
        live.validate_provider_evidence(
            PROVIDER_EVIDENCE,
            active,
            today=date(2026, 8, 3),
        )


def test_authorized_native_outcomes_are_redacted_and_nonactivating() -> None:
    """Seal exact live outcomes without treating proof as activation."""
    payload, _ = live._load_json_object(LIVE_RESULTS)
    assert (
        live._validate_canonical_digest(payload)
        == "d3aab6c4e4c83be848a304126c2d933898e499909bc65594959f74bb00c66e44"
    )
    assert payload["provider_families"] == ["openai", "azure_openai"]
    assert (
        payload["authority_scope"] == "explicit_separate_native_provider_runs"
    )
    assert payload["completed_full_matrix_profile_count"] == 1
    assert payload["active_profile_count"] == 0
    assert payload["activation_decision"] == "remain_inactive"

    harness = cast(dict[str, object], payload["harness"])
    current_harness = Path(__file__).parents[2] / cast(str, harness["path"])
    assert (
        sha256(current_harness.read_bytes()).hexdigest()
        == harness["byte_sha256"]
    )
    assert harness["opaque_payloads_logged"] is False
    assert harness["credentials_logged"] is False
    assert harness["byte_sha256"] != harness["reviewed_byte_sha256"]
    assert harness["review_status"] == "pending_post_live_harness_delta_review"

    for link_name in ("provider_evidence", "preflight_manifest"):
        link = cast(dict[str, object], payload[link_name])
        linked_path = FIXTURE_DIRECTORY / cast(str, link["path"])
        assert (
            sha256(linked_path.read_bytes()).hexdigest() == link["byte_sha256"]
        )
        linked, _ = live._load_json_object(linked_path)
        assert (
            live._validate_canonical_digest(linked) == link["canonical_digest"]
        )

    native_openai = cast(dict[str, object], payload["native_openai_attempt"])
    assert native_openai["provider_family"] == "openai"
    assert native_openai["model"] == "gpt-5.6-sol"
    assert native_openai["model_revision"] == "gpt-5.6-sol"
    assert native_openai["total_http_call_count"] == 3
    assert native_openai["live_capability_receipt"] is False
    model_retrieve = cast(dict[str, object], native_openai["model_retrieve"])
    assert model_retrieve == {
        "http_call_count": 1,
        "present": True,
        "returned_model_id": "gpt-5.6-sol",
    }
    matrix_execution = cast(
        dict[str, object], native_openai["matrix_execution"]
    )
    assert (
        matrix_execution["state"]
        == "inactive_account_credit_exhausted_before_inference"
    )
    assert matrix_execution["generation_attempt_count"] == 2
    assert matrix_execution["completed_generation_count"] == 0
    assert matrix_execution["completed_cases"] == []
    assert matrix_execution["failed_case"] == "stateless_current_turn_tool"
    assert matrix_execution["safe_error"] == {
        "class": "RateLimitError",
        "category": "account_credit_quota_blocker",
        "http_status": 429,
        "code": "credit_balance_exhausted",
        "type": "insufficient_quota",
        "param": None,
    }
    assert native_openai["request_context"] == {
        "project_selector_configured": False,
        "organization_selector_configured": False,
        "safe_error_identified_project_or_account": False,
    }
    assert native_openai["side_effects"] == {
        "completed_inference": False,
        "stored_response_created": False,
        "compaction_item_created": False,
        "opaque_provider_content_logged": False,
    }

    expected_azure_profiles = (
        ("gpt-5", "2025-08-07"),
        ("gpt-5-mini", "2025-08-07"),
        ("gpt-5.4-mini", "2026-03-17"),
        ("gpt-5-nano", "2025-08-07"),
        ("gpt-5.6-terra", "2026-07-09"),
        ("gpt-5.6-sol", "2026-07-09"),
    )
    execution_order = cast(list[str], payload["execution_order"])
    azure_openai = cast(dict[str, object], payload["azure_openai_matrix"])
    results = cast(list[dict[str, object]], azure_openai["results"])
    assert (
        tuple(
            (
                cast(str, row["deployment"]),
                cast(str, row["deployment_revision"]),
            )
            for row in results
        )
        == expected_azure_profiles
    )
    for row in results:
        assert cast(str, row["state"]).startswith("inactive_")
        failed_case = row["failed_case"]
        if failed_case is None:
            assert row["completed_cases"] == execution_order
            assert row["safe_error"] is None
            continue
        assert type(failed_case) is str
        failed_index = execution_order.index(failed_case)
        assert row["completed_cases"] == execution_order[:failed_index]
        assert set(cast(dict[str, object], row["safe_error"])) == {
            "class",
            "category",
            "http_status",
            "code",
            "type",
            "param",
        }
    assert cast(dict[str, object], results[0]["safe_error"]) == {
        "class": "BadRequestError",
        "category": "provider_rejection",
        "http_status": 400,
        "code": "unsupported_value",
        "type": "invalid_request_error",
        "param": "reasoning.context",
    }
    terra = results[4]
    assert terra["state"] == "inactive_complete_live_matrix_pending_review"
    assert "live_call_accounting" not in terra
    assert "full_matrix_receipt" not in terra
    receipt = cast(dict[str, object], terra["tracked_cli_receipt"])
    accounting = cast(dict[str, object], receipt["accounting"])
    assert accounting == {
        "cleanup_attempted": True,
        "cleanup_completed": True,
        "cleanup_delete_http_request_count": 0,
        "cleanup_delete_logical_operation_count": 0,
        "cleanup_pending_reference_count": 0,
        "client_close_completed": True,
        "http_request_count": 11,
        "http_request_counts": {
            "compact": 1,
            "create_or_stream": 7,
            "delete": 2,
            "retrieve": 1,
            "unexpected": 0,
        },
        "logical_operation_count": 11,
        "logical_operation_counts": {
            "compact": 1,
            "create_or_stream": 7,
            "delete": 2,
            "retrieve": 1,
        },
        "observed_sdk_retry_count": 0,
        "request_path_class_mismatch_count": 0,
        "sdk_configured_max_retries": 0,
        "successful_matrix_expected_counts_match": True,
        "unexpected_request_count": 0,
    }
    assert receipt["provider_family"] == "azure_openai"
    assert receipt["model_or_deployment"] == "gpt-5.6-terra"
    assert receipt["model_or_deployment_revision"] == "2026-07-09"
    assert receipt["observed_at"] == "2026-08-04T13:16:09.894683+00:00"
    assert receipt["completed_cases"] == execution_order
    assert receipt["opaque_payloads_logged"] is False
    assert receipt["production_activation_granted"] is False
    assert (
        receipt["structural_observations_digest"]
        == "f76c0c145f3775c5e445cb55efb3c9cb5b9293a01695e6850f3764ee6badc5f3"
    )
    serialized = dumps(payload, sort_keys=True)
    for forbidden_field in (
        '"api_key"',
        '"prompt"',
        '"response_id"',
        '"previous_response_id"',
        '"encrypted_content"',
    ):
        assert forbidden_field not in serialized


def _complete_accounting(**changes: object) -> Any:
    """Return one exact content-free complete transport snapshot."""
    values: dict[str, object] = {
        "create_or_stream_logical_operation_count": 7,
        "compact_logical_operation_count": 1,
        "retrieve_logical_operation_count": 1,
        "delete_logical_operation_count": 2,
        "create_or_stream_http_request_count": 7,
        "compact_http_request_count": 1,
        "retrieve_http_request_count": 1,
        "delete_http_request_count": 2,
        "unexpected_http_request_count": 0,
        "sdk_configured_max_retries": 0,
        "observed_sdk_retry_count": 0,
        "request_path_class_mismatch_count": 0,
        "cleanup_attempted": True,
        "cleanup_completed": True,
        "cleanup_delete_logical_operation_count": 0,
        "cleanup_delete_http_request_count": 0,
        "cleanup_pending_reference_count": 0,
        "client_close_completed": True,
    }
    values.update(changes)
    return live.LiveConformanceAccounting(**cast(Any, values))


class _FakeTransport:
    """Return selected structural observations for assertion testing."""

    def __init__(
        self,
        observations: Mapping[Any, Any],
    ) -> None:
        self.observations = observations
        self.closed = False
        self.accounting = _complete_accounting()

    async def execute(
        self,
        case: Any,
    ) -> Any:
        return self.observations[case]

    async def aclose(self) -> None:
        self.closed = True

    def final_accounting(self) -> Any:
        return self.accounting


def _valid_observation(
    case: Any,
) -> Any:
    context = (
        "all_turns"
        if case
        in {
            live.LiveConformanceCase.STATELESS_ALL_TURNS,
            live.LiveConformanceCase.STORED_CHAIN,
        }
        else (
            None
            if case is live.LiveConformanceCase.STANDALONE_COMPACTION
            else "current_turn"
        )
    )
    if case is live.LiveConformanceCase.STORED_RETRIEVE_DELETE:
        return live.LiveConformanceObservation(
            case=case,
            response_status="completed",
            parent_matches=True,
            reasoning_context=None,
            item_kinds=(),
            reported_model_identity="gpt-5",
            retrieved=True,
            deleted=True,
        )
    tool = case in {
        live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
        live.LiveConformanceCase.STREAMING_TOOL,
    }
    replay = case is live.LiveConformanceCase.STATELESS_ALL_TURNS
    tool_evidence = tool or replay
    compact = case in {
        live.LiveConformanceCase.INLINE_COMPACTION,
        live.LiveConformanceCase.STANDALONE_COMPACTION,
    }
    return live.LiveConformanceObservation(
        case=case,
        response_status="completed",
        parent_matches=True,
        reasoning_context=context,
        item_kinds=(
            ("reasoning", "function_call")
            if tool
            else ("compaction", "message") if compact else ("message",)
        ),
        reported_model_identity="gpt-5",
        required_reasoning_items=1 if tool_evidence else 0,
        encrypted_reasoning_items=1 if tool else 0,
        compaction_items=(
            1
            if case
            in {
                live.LiveConformanceCase.INLINE_COMPACTION,
                live.LiveConformanceCase.STANDALONE_COMPACTION,
            }
            else 0
        ),
        opaque_compaction_items=1 if compact else 0,
        tool_calls=1 if tool_evidence else 0,
        required_tool_name_matches=tool_evidence,
        required_tool_arguments_match=tool_evidence,
        tool_correlation_matches=tool_evidence,
        encrypted_reasoning_precedes_tool=tool_evidence,
        preceding_reasoning_replayed=replay,
        stream_event_kinds=(
            ("response.output_item.done", "response.completed")
            if case is live.LiveConformanceCase.STREAMING_TOOL
            else ()
        ),
        complete_output_replayed=(
            case is live.LiveConformanceCase.STATELESS_ALL_TURNS
        ),
        replayed_encrypted_reasoning_items=1 if replay else 0,
        replayed_tool_outputs=1 if replay else 0,
        compact_output_replayed=(
            case is live.LiveConformanceCase.STANDALONE_COMPACTION
        ),
    )


@pytest.mark.parametrize(
    "case",
    (
        live.LiveConformanceCase.STATELESS_ALL_TURNS,
        live.LiveConformanceCase.INLINE_COMPACTION,
        live.LiveConformanceCase.STANDALONE_COMPACTION,
    ),
)
def test_replay_and_compaction_do_not_require_fresh_reasoning(
    case: Any,
) -> None:
    """Accept opaque replay without overclaiming fresh reasoning."""
    observation = _valid_observation(case)
    assert observation.encrypted_reasoning_items == 0
    live._assert_observation(case, observation)


def test_standalone_message_only_followup_is_completed_conformance() -> None:
    """Accept a completed replay response without fresh reasoning output."""
    observation = _valid_observation(
        live.LiveConformanceCase.STANDALONE_COMPACTION
    )
    assert observation.reasoning_context is None
    assert observation.encrypted_reasoning_items == 0
    assert observation.item_kinds == ("compaction", "message")
    live._assert_observation(
        live.LiveConformanceCase.STANDALONE_COMPACTION,
        observation,
    )


def _normalized_stateless_replay_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    output: list[dict[str, object]] = [
        {
            "encrypted_content": "opaque-reasoning",
            "id": "reasoning-ready",
            "summary": [],
            "type": "reasoning",
        },
        {
            "arguments": '{"value":"1943"}',
            "call_id": "call-ready",
            "id": "function-ready",
            "name": "phase12_probe",
            "type": "function_call",
        },
    ]
    tool_outputs: list[dict[str, object]] = [
        {
            "call_id": "call-ready",
            "output": '{"accepted":true}',
            "type": "function_call_output",
        }
    ]
    followup: dict[str, object] = {
        "content": "continue",
        "role": "user",
        "type": "message",
    }
    return output, tool_outputs, followup


def test_complete_stateless_replay_binds_exact_tool_and_reasoning() -> None:
    """Accept exact encrypted reasoning, tool arguments, and correlation."""
    output, tool_outputs, followup = _normalized_stateless_replay_fixture()
    replay = [*output, *tool_outputs, followup]
    assert live._complete_stateless_replay(
        cast(Any, output),
        cast(Any, tool_outputs),
        cast(Any, replay),
    )
    assert live._preceding_encrypted_reasoning_replayed(
        cast(Any, output),
        cast(Any, replay),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_name",
        "wrong_arguments",
        "encrypted_after_tool",
        "mixed_encrypted_and_plain_reasoning",
        "missing_ciphertext",
        "bad_correlation",
    ),
)
def test_complete_stateless_replay_rejects_structural_mutations(
    mutation: str,
) -> None:
    """Reject every semantic mutation of the required tool transition."""
    output, tool_outputs, followup = _normalized_stateless_replay_fixture()
    if mutation == "wrong_name":
        output[1]["name"] = "different_probe"
    elif mutation == "wrong_arguments":
        output[1]["arguments"] = '{"value":"different"}'
    elif mutation == "encrypted_after_tool":
        output[:] = [output[1], output[0]]
    elif mutation == "mixed_encrypted_and_plain_reasoning":
        output.insert(
            1,
            {
                "id": "reasoning-plain",
                "summary": [],
                "type": "reasoning",
            },
        )
    elif mutation == "missing_ciphertext":
        output[0].pop("encrypted_content")
    elif mutation == "bad_correlation":
        tool_outputs[0]["call_id"] = "different-call"
    else:
        raise AssertionError("unhandled test mutation")
    replay = [*output, *tool_outputs, followup]
    assert not live._complete_stateless_replay(
        cast(Any, output),
        cast(Any, tool_outputs),
        cast(Any, replay),
    )


def _normalized_compact_replay_fixture() -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    output: list[dict[str, object]] = [
        {
            "encrypted_content": "opaque-first",
            "id": "cmp-first",
            "type": "compaction",
        },
        {
            "content": [{"text": "retained", "type": "output_text"}],
            "id": "msg-retained",
            "role": "assistant",
            "type": "message",
        },
        {
            "encrypted_content": "opaque-second",
            "id": "cmp-second",
            "type": "compaction",
        },
        {
            "content": [{"text": "trailing", "type": "output_text"}],
            "id": "msg-trailing",
            "role": "assistant",
            "type": "message",
        },
    ]
    followup: dict[str, object] = {
        "content": [{"text": "continue", "type": "input_text"}],
        "role": "user",
        "type": "message",
    }
    return output, followup


def test_complete_compact_replay_preserves_multiple_items_and_order() -> None:
    """Accept every encrypted compact item without position assumptions."""
    output, followup = _normalized_compact_replay_fixture()
    assert output[-1]["type"] == "message"
    assert sum(item["type"] == "compaction" for item in output) == 2
    assert live._complete_compact_replay(
        cast(Any, output),
        cast(Any, [*output, followup]),
    )


@pytest.mark.parametrize("failure", ("pruned", "reordered"))
def test_complete_compact_replay_rejects_loss_or_reordering(
    failure: str,
) -> None:
    """Reject any replay that does not preserve the exact compact output."""
    output, followup = _normalized_compact_replay_fixture()
    replay = (
        [*output[:-1], followup]
        if failure == "pruned"
        else [output[1], output[0], *output[2:], followup]
    )
    assert not live._complete_compact_replay(
        cast(Any, output),
        cast(Any, replay),
    )


@pytest.mark.parametrize("encrypted_content", (None, "", 1))
@pytest.mark.parametrize("compact_index", (0, 2))
def test_complete_compact_replay_rejects_unencrypted_compaction(
    compact_index: int,
    encrypted_content: object,
) -> None:
    """Require every normalized compact item to carry opaque ciphertext."""
    output, followup = _normalized_compact_replay_fixture()
    if encrypted_content is None:
        output[compact_index].pop("encrypted_content")
    else:
        output[compact_index]["encrypted_content"] = encrypted_content
    assert not live._complete_compact_replay(
        cast(Any, output),
        cast(Any, [*output, followup]),
    )


def test_complete_compact_replay_requires_an_opaque_compaction_item() -> None:
    """Reject normalized output containing only non-compaction items."""
    output, followup = _normalized_compact_replay_fixture()
    output = [item for item in output if item["type"] != "compaction"]
    assert not live._complete_compact_replay(
        cast(Any, output),
        cast(Any, [*output, followup]),
    )


@pytest.mark.parametrize(
    ("case", "changes"),
    (
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"case": live.LiveConformanceCase.STORED_CREATE},
        ),
        (
            live.LiveConformanceCase.STORED_CREATE,
            {"response_status": "incomplete"},
        ),
        (
            live.LiveConformanceCase.INLINE_COMPACTION,
            {"response_status": "incomplete"},
        ),
        (
            live.LiveConformanceCase.STANDALONE_COMPACTION,
            {"response_status": "incomplete"},
        ),
        (
            live.LiveConformanceCase.STORED_CHAIN,
            {"parent_matches": False},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"encrypted_reasoning_items": 0},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"required_reasoning_items": 2},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"required_tool_name_matches": False},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"required_tool_arguments_match": False},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"tool_correlation_matches": False},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"encrypted_reasoning_precedes_tool": False},
        ),
        (
            live.LiveConformanceCase.STATELESS_ALL_TURNS,
            {"replayed_encrypted_reasoning_items": 0},
        ),
        (
            live.LiveConformanceCase.STATELESS_ALL_TURNS,
            {"replayed_tool_outputs": 0},
        ),
        (
            live.LiveConformanceCase.STATELESS_ALL_TURNS,
            {"preceding_reasoning_replayed": False},
        ),
        (
            live.LiveConformanceCase.STREAMING_TOOL,
            {"encrypted_reasoning_items": 0},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"item_kinds": ("function_call", "reasoning")},
        ),
        (
            live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL,
            {"tool_calls": 2},
        ),
        (
            live.LiveConformanceCase.STORED_CREATE,
            {"reasoning_context": "all_turns"},
        ),
        (
            live.LiveConformanceCase.STATELESS_ALL_TURNS,
            {"complete_output_replayed": False},
        ),
        (
            live.LiveConformanceCase.INLINE_COMPACTION,
            {"compaction_items": 0},
        ),
        (
            live.LiveConformanceCase.INLINE_COMPACTION,
            {"opaque_compaction_items": 0},
        ),
        (
            live.LiveConformanceCase.STANDALONE_COMPACTION,
            {"compact_output_replayed": False},
        ),
        (
            live.LiveConformanceCase.STANDALONE_COMPACTION,
            {"compaction_items": 0},
        ),
        (
            live.LiveConformanceCase.STANDALONE_COMPACTION,
            {"opaque_compaction_items": 0},
        ),
        (
            live.LiveConformanceCase.STANDALONE_COMPACTION,
            {"reasoning_context": "all_turns"},
        ),
        (
            live.LiveConformanceCase.STREAMING_TOOL,
            {"stream_event_kinds": ("response.output_item.done",)},
        ),
        (
            live.LiveConformanceCase.STREAMING_TOOL,
            {
                "stream_event_kinds": (
                    "response.completed",
                    "response.completed",
                )
            },
        ),
        (
            live.LiveConformanceCase.STORED_RETRIEVE_DELETE,
            {"deleted": False},
        ),
    ),
)
def test_structural_assertions_reject_each_invalid_live_branch(
    case: Any,
    changes: dict[str, object],
) -> None:
    """Reject every malformed structure without retaining provider payloads."""
    observation = replace(
        _valid_observation(case),
        **cast(Any, changes),
    )
    with pytest.raises(live.LiveConformanceAssertionError):
        live._assert_observation(case, observation)


@pytest.mark.parametrize(
    "changes",
    (
        {"required_reasoning_items": 2},
        {"encrypted_reasoning_items": 0},
        {"required_tool_name_matches": False},
        {"required_tool_arguments_match": False},
        {"tool_correlation_matches": False},
        {"encrypted_reasoning_precedes_tool": False},
        {"preceding_reasoning_replayed": True},
        {"opaque_compaction_items": 1},
    ),
)
def test_structural_digest_binds_new_proof_facts(
    changes: dict[str, object],
) -> None:
    """Bind every added non-secret structural fact into the receipt digest."""
    observation = _valid_observation(
        live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL
    )
    accounting = _complete_accounting()
    baseline = live._structural_observations_digest(
        (observation,),
        accounting,
    )
    mutated = replace(observation, **cast(Any, changes))
    assert (
        live._structural_observations_digest((mutated,), accounting)
        != baseline
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"create_or_stream_logical_operation_count": 8},
        {"compact_http_request_count": 2},
        {"sdk_configured_max_retries": 1},
        {"observed_sdk_retry_count": 1},
        {"unexpected_http_request_count": 1},
        {"request_path_class_mismatch_count": 1},
        {"cleanup_attempted": False},
        {"cleanup_completed": False},
        {"cleanup_delete_logical_operation_count": 1},
        {"cleanup_delete_http_request_count": 1},
        {"cleanup_pending_reference_count": 1},
        {"client_close_completed": False},
    ),
)
def test_structural_digest_binds_generated_accounting(
    changes: dict[str, object],
) -> None:
    """Bind every generated accounting fact into the structural digest."""
    observation = _valid_observation(
        live.LiveConformanceCase.STATELESS_CURRENT_TURN_TOOL
    )
    accounting = _complete_accounting()
    baseline = live._structural_observations_digest(
        (observation,),
        accounting,
    )
    mutated = replace(accounting, **cast(Any, changes))
    assert (
        live._structural_observations_digest((observation,), mutated)
        != baseline
    )


def test_provider_reported_model_identity_is_exact() -> None:
    """Reject any provider-reported model identity drift."""
    observation = replace(
        _valid_observation(live.LiveConformanceCase.STORED_CREATE),
        reported_model_identity="different-model-revision",
    )
    with pytest.raises(live.LiveConformanceAssertionError):
        live._assert_provider_identity(_config(), observation)


def test_azure_provider_identity_separates_model_from_revision_pin() -> None:
    """Bind Azure response.model without conflating its revision pin."""
    config = _config(
        family=live.LiveProviderFamily.AZURE_OPENAI,
        model_or_deployment="gpt-5",
        model_or_deployment_revision="2025-08-07",
    )
    observation = _valid_observation(live.LiveConformanceCase.STORED_CREATE)
    live._assert_provider_identity(config, observation)
    with pytest.raises(live.LiveConformanceAssertionError):
        live._assert_provider_identity(
            config,
            replace(observation, reported_model_identity="2025-08-07"),
        )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        ('{"value":"ready"}', True),
        ('{ "value" : "ready" }', True),
        ('{"value":"re\\u0061dy"}', True),
        ('{"value":"ready","value":"ready"}', False),
        ('{"value":"ready","extra":true}', False),
        ('{"value":NaN}', False),
        ('{"value":1}', False),
        ('{"value":', False),
        (None, False),
    ),
)
def test_tool_arguments_match_strict_json_semantics(
    arguments: object,
    expected: bool,
) -> None:
    """Accept JSON formatting variance while rejecting semantic drift."""
    assert live._tool_arguments_match(arguments, "ready") is expected


async def test_execution_failure_canary_is_typed_cause_free_and_redacted() -> (
    None
):
    """Never leak upstream identifiers, URLs, bodies, or exception causes."""
    canary = "upstream-id https://provider.invalid opaque-body secret"

    class LeakingTransport(_FakeTransport):
        async def execute(self, case: Any) -> Any:
            del case
            raise RuntimeError(canary)

    observations = {
        case: _valid_observation(case) for case in live._EXECUTION_ORDER
    }
    with pytest.raises(live.LiveConformanceExecutionError) as captured:
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: LeakingTransport(observations),
            clock=_clock,
        )
    assert captured.value.__cause__ is None
    assert (
        str(captured.value)
        == "live conformance case failed: stateless_current_turn_tool"
    )
    assert canary not in str(captured.value)
    assert "provider.invalid" not in str(captured.value)


async def test_sdk_transport_rejects_unsupported_live_case() -> None:
    """Fail closed when an unrecognized case reaches the SDK transport."""
    transport = object.__new__(live.OpenAISdkLiveConformanceTransport)

    with pytest.raises(
        live.LiveConformanceExecutionError, match="unsupported"
    ):
        await transport.execute(cast(live.LiveConformanceCase, "unknown"))


async def test_runner_closes_transport_after_assertion_failure() -> None:
    """Always invoke cleanup when a structural provider assertion fails."""
    observations = {
        case: _valid_observation(case) for case in live._EXECUTION_ORDER
    }
    observations[live.LiveConformanceCase.INLINE_COMPACTION] = replace(
        observations[live.LiveConformanceCase.INLINE_COMPACTION],
        compaction_items=0,
    )
    transport = _FakeTransport(observations)

    with pytest.raises(live.LiveConformanceAssertionError) as captured:
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: transport,
            clock=_clock,
        )

    assert transport.closed
    assert (
        str(captured.value)
        == "live conformance case failed: inline_compaction"
    )


def test_environment_configuration_requires_exact_provider_variables() -> None:
    """Map only explicit native OpenAI and Azure environment identities."""
    authority = {
        "AVALAN_LIVE_CONFORMANCE_AUTHORITY": (
            "authorize-phase12-live-conformance"
        ),
        "AVALAN_LIVE_CONFORMANCE_COST_ACK": "accept-phase12-provider-costs",
    }
    arguments = Namespace(
        provider="openai",
        authorize_live_provider_conformance=True,
        acknowledge_provider_costs=True,
    )
    openai_config = live._config_from_environment(
        arguments,
        {
            **authority,
            "OPENAI_API_KEY": "sk-explicit",
            "OPENAI_MODEL": "gpt-5",
            "OPENAI_MODEL_REVISION": "revision-exact",
        },
    )
    assert openai_config.endpoint == "https://api.openai.com/v1"

    arguments.provider = "azure_openai"
    azure_config = live._config_from_environment(
        arguments,
        {
            **authority,
            "AZURE_OPENAI_API_KEY": "azure-explicit",
            "AZURE_OPENAI_ENDPOINT": (
                "https://resource.openai.azure.com/openai/v1"
            ),
            "AZURE_OPENAI_DEPLOYMENT": "deployment-exact",
            "AZURE_OPENAI_DEPLOYMENT_REVISION": "revision-exact",
            "AZURE_OPENAI_API_REVISION": "azure-openai-v1-preview",
        },
    )
    assert azure_config.provider_family is live.LiveProviderFamily.AZURE_OPENAI
    assert azure_config.model_or_deployment == "deployment-exact"
    assert azure_config.api_form == "azure-openai-v1-preview"


async def test_protocol_fallbacks_and_order_dependencies_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover protocol fallbacks and reject out-of-order SDK operations."""
    with pytest.raises(NotImplementedError):
        await live.LiveConformanceTransport.execute(
            cast(Any, object()),
            live.LiveConformanceCase.STORED_CREATE,
        )
    with pytest.raises(NotImplementedError):
        await live.LiveConformanceTransport.aclose(cast(Any, object()))
    with pytest.raises(NotImplementedError):
        live.LiveConformanceTransport.final_accounting(cast(Any, object()))

    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    with pytest.raises(live.LiveConformanceAssertionError):
        await transport._stateless_all_turns()
    with pytest.raises(live.LiveConformanceAssertionError):
        await transport._stored_chain()
    with pytest.raises(live.LiveConformanceAssertionError):
        await transport._stored_retrieve_delete()
    await transport.aclose()


async def test_failed_matrix_retries_transient_cleanup_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a transient retained deletion after a structural failure."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    client = _FakeClient.instances[-1]
    canary = "transient-private-response-id"
    transport._cleanup_retries.retain(canary)
    client.responses.delete_failures.append(
        RuntimeError(f"transient failure {canary} secret-body")
    )

    async def invalid_execute(case: Any) -> Any:
        return replace(
            _valid_observation(case),
            required_tool_arguments_match=False,
        )

    monkeypatch.setattr(transport, "execute", invalid_execute)
    with pytest.raises(live.LiveConformanceAssertionError) as captured:
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: transport,
            clock=_clock,
        )
    assert captured.value.__cause__ is None
    assert canary not in str(captured.value)
    assert client.responses.delete_calls == [canary, canary]
    assert transport._cleanup_retries.pending_count == 0
    assert client.closed
    accounting = transport.final_accounting()
    assert accounting.cleanup_attempted
    assert accounting.cleanup_completed
    assert accounting.cleanup_delete_logical_operation_count == 2
    assert accounting.cleanup_delete_http_request_count == 2


async def test_sdk_transport_rejects_untyped_retrieve_and_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep typed lifecycle and cleanup failures explicit."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    transport._stored_first = _response(
        "resp_stored_first",
        [_message("msg_first")],
        context="current_turn",
    )
    transport._stored_chained = _response(
        "resp_stored_chain",
        [_message("msg_chain")],
        context="all_turns",
        previous_response_id="resp_stored_first",
    )
    client = _FakeClient.instances[-1]
    client.responses.retrieve_result = object()
    with pytest.raises(live.LiveConformanceAssertionError):
        await transport._stored_retrieve_delete()

    canary = "resp-secret-upstream-id-and-url-body"
    transport._cleanup_retries.retain(canary)
    client.responses.delete_failure = RuntimeError(
        f"delete failed {canary} https://upstream.invalid secret-body"
    )
    with pytest.raises(live.LiveConformanceCleanupError) as captured:
        await transport.aclose()
    assert captured.value.__cause__ is None
    assert canary not in str(captured.value)
    assert "upstream.invalid" not in str(captured.value)
    assert canary not in repr(transport._cleanup_retries)
    assert transport._cleanup_retries.pending_count == 1
    assert client.closed
    assert client.responses.delete_calls == [canary, canary, canary]
    accounting = transport.final_accounting()
    assert accounting.cleanup_attempted
    assert not accounting.cleanup_completed
    assert accounting.cleanup_delete_logical_operation_count == 3
    assert accounting.cleanup_delete_http_request_count == 3
    assert accounting.cleanup_pending_reference_count == 1


async def test_cleanup_cancellation_still_closes_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the SDK client even when protected deletion is cancelled."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())
    client = _FakeClient.instances[-1]
    transport._cleanup_retries.retain("cancelled-private-response-id")
    client.responses.delete_failure = CancelledError()
    with pytest.raises(CancelledError):
        await transport.aclose()
    assert client.responses.delete_calls == ["cancelled-private-response-id"]
    assert transport._cleanup_retries.pending_count == 1
    assert client.closed
    accounting = transport.final_accounting()
    assert accounting.cleanup_attempted
    assert not accounting.cleanup_completed
    assert accounting.cleanup_delete_logical_operation_count == 1
    assert accounting.cleanup_delete_http_request_count == 1
    assert accounting.cleanup_pending_reference_count == 1


async def test_runner_cancellation_captures_final_post_cleanup_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture final safe accounting after execution cancellation cleanup."""
    _FakeClient.instances.clear()
    monkeypatch.setattr(live, "AsyncOpenAI", _FakeClient)
    transport = live.OpenAISdkLiveConformanceTransport(_config())

    async def cancel_execute(case: Any) -> Any:
        del case
        raise CancelledError()

    monkeypatch.setattr(transport, "execute", cancel_execute)
    with pytest.raises(CancelledError):
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: transport,
            clock=_clock,
        )

    accounting = transport.final_accounting()
    assert accounting.logical_operation_count == 0
    assert accounting.http_request_count == 0
    assert accounting.cleanup_attempted
    assert accounting.cleanup_completed
    assert accounting.client_close_completed


async def test_runner_cleanup_failure_and_receipt_clock_are_explicit() -> None:
    """Propagate cleanup failures and validate the post-run UTC receipt."""
    observations = {
        case: _valid_observation(case) for case in live._EXECUTION_ORDER
    }

    class ClosingFailureTransport(_FakeTransport):
        async def aclose(self) -> None:
            raise RuntimeError("raw-cleanup-canary-upstream-body")

    with pytest.raises(live.LiveConformanceCleanupError) as captured:
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: ClosingFailureTransport(
                observations
            ),
            clock=_clock,
        )
    assert captured.value.__cause__ is None
    assert "raw-cleanup-canary" not in str(captured.value)

    clock_calls = 0

    async def drifting_clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            return NOW
        return datetime(2026, 8, 3, 12)

    with pytest.raises(live.LiveConformancePreflightError):
        await live.run_live_conformance(
            _config(),
            transport_factory=lambda config: _FakeTransport(observations),
            clock=drifting_clock,
        )


def test_fixture_loading_digest_and_field_helpers_fail_closed(
    tmp_path: Path,
) -> None:
    """Reject unreadable, non-object, malformed, and tampered fixtures."""
    missing = tmp_path / "missing.json"
    with pytest.raises(live.LiveConformancePreflightError):
        live._load_json_object(missing)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(live.LiveConformancePreflightError):
        live._load_json_object(malformed)

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    with pytest.raises(live.LiveConformancePreflightError):
        live._load_json_object(sequence)

    payload = cast(
        dict[str, object],
        loads(PROVIDER_EVIDENCE.read_text(encoding="utf-8")),
    )
    malformed_digest = deepcopy(payload)
    cast(dict[str, object], malformed_digest["canonical_digest"])[
        "algorithm"
    ] = "sha1"
    with pytest.raises(live.LiveConformancePreflightError):
        live._validate_canonical_digest(malformed_digest)

    tampered = deepcopy(payload)
    tampered["feature"] = "tampered"
    with pytest.raises(live.LiveConformancePreflightError):
        live._validate_canonical_digest(tampered)

    with pytest.raises(live.LiveConformancePreflightError):
        live._date_field({"accessed_at": 1}, "accessed_at")
    with pytest.raises(live.LiveConformancePreflightError):
        live._date_field({"accessed_at": "not-a-date"}, "accessed_at")
    with pytest.raises(live.LiveConformancePreflightError):
        live._object_field({"sdk": []}, "sdk")


def test_provider_source_validation_rejects_every_malformed_shape() -> None:
    """Reject generic, incomplete, duplicated, and drifted source evidence."""
    evidence = cast(
        dict[str, object],
        loads(PROVIDER_EVIDENCE.read_text(encoding="utf-8")),
    )
    invalid: list[dict[str, object]] = []

    wrong_count = deepcopy(evidence)
    wrong_count["providers"] = []
    invalid.append(wrong_count)

    wrong_row = deepcopy(evidence)
    cast(list[object], wrong_row["providers"])[0] = "openai"
    invalid.append(wrong_row)

    generic = deepcopy(evidence)
    cast(list[dict[str, object]], generic["providers"])[0][
        "provider_family"
    ] = "openai_compatible"
    invalid.append(generic)

    duplicate = deepcopy(evidence)
    cast(list[dict[str, object]], duplicate["providers"])[1][
        "provider_family"
    ] = "openai"
    invalid.append(duplicate)

    wrong_form = deepcopy(evidence)
    cast(list[dict[str, object]], wrong_form["providers"])[0][
        "api_form"
    ] = "compatible"
    invalid.append(wrong_form)

    wrong_encrypted_content_policy = deepcopy(evidence)
    cast(
        list[dict[str, object]],
        wrong_encrypted_content_policy["providers"],
    )[1]["encrypted_content_policy"] = "default_return"
    invalid.append(wrong_encrypted_content_policy)

    wrong_sources = deepcopy(evidence)
    cast(list[dict[str, object]], wrong_sources["providers"])[0][
        "sources"
    ] = {}
    invalid.append(wrong_sources)

    wrong_source = deepcopy(evidence)
    cast(
        list[object],
        cast(list[dict[str, object]], wrong_source["providers"])[0]["sources"],
    )[0] = "source"
    invalid.append(wrong_source)

    incomplete = deepcopy(evidence)
    cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], incomplete["providers"])[0]["sources"],
    )[0]["facts"] = []
    invalid.append(incomplete)

    wrong_urls = deepcopy(evidence)
    cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], wrong_urls["providers"])[0]["sources"],
    )[0]["url"] = "https://developers.openai.com/drift"
    invalid.append(wrong_urls)

    for payload in invalid:
        with pytest.raises(live.LiveConformancePreflightError):
            live._validate_provider_sources(payload, date(2026, 8, 3))


def test_sdk_evidence_validation_rejects_runtime_and_type_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when SDK methods, fields, or typed symbols drift."""
    evidence = cast(
        dict[str, object],
        loads(PROVIDER_EVIDENCE.read_text(encoding="utf-8")),
    )
    runtime_drift = deepcopy(evidence)
    cast(dict[str, object], runtime_drift["sdk"])[
        "installed_version"
    ] = "2.43.0"
    with pytest.raises(live.LiveConformancePreflightError):
        live._validate_sdk_evidence(runtime_drift)

    sdk = cast(dict[str, object], evidence["sdk"])
    typed_fields = cast(list[str], sdk["typed_request_fields"])
    typed_symbols = cast(list[str], sdk["typed_symbols"])
    field_variants = (
        typed_fields[:-1],
        ["conversation", *typed_fields[1:]],
        [*typed_fields, "extra_body"],
        [typed_fields[1], typed_fields[0], *typed_fields[2:]],
    )
    symbol_variants = (
        typed_symbols[:-1],
        ["openai.types.responses.FunctionToolParam", *typed_symbols[1:]],
        [*typed_symbols, "openai.types.responses.FunctionToolParam"],
        [typed_symbols[1], typed_symbols[0], *typed_symbols[2:]],
    )
    for key, variants in (
        ("typed_request_fields", field_variants),
        ("typed_symbols", symbol_variants),
    ):
        for values in variants:
            fixture_drift = deepcopy(evidence)
            cast(dict[str, object], fixture_drift["sdk"])[key] = values
            with pytest.raises(live.LiveConformancePreflightError):
                live._validate_sdk_evidence(fixture_drift)

    def no_parameters(target: object) -> SimpleNamespace:
        del target
        return SimpleNamespace(parameters={})

    with monkeypatch.context() as context:
        context.setattr(live, "signature", no_parameters)
        with pytest.raises(live.LiveConformancePreflightError):
            live._validate_sdk_evidence(evidence)

    def reordered_parameters(target: object) -> SimpleNamespace:
        del target
        return SimpleNamespace(
            parameters=dict.fromkeys(reversed(typed_fields))
        )

    with monkeypatch.context() as context:
        context.setattr(live, "signature", reordered_parameters)
        with pytest.raises(live.LiveConformancePreflightError):
            live._validate_sdk_evidence(evidence)

    with monkeypatch.context() as context:
        context.setattr(live.AsyncResponses, "compact", None)
        with pytest.raises(live.LiveConformancePreflightError):
            live._validate_sdk_evidence(evidence)

    imported_module = live.import_module

    def substituted_symbol(module_name: str) -> object:
        if module_name == "openai.types.responses":
            return SimpleNamespace(CompactedResponse=object())
        return imported_module(module_name)

    with monkeypatch.context() as context:
        context.setattr(live, "import_module", substituted_symbol)
        with pytest.raises(live.LiveConformancePreflightError):
            live._validate_sdk_evidence(evidence)


def test_activation_fixture_rejects_authority_links_reviews_and_matrix(
    tmp_path: Path,
) -> None:
    """Reject activated, unlinked, reviewed, or partial manifest claims."""
    evidence = cast(
        dict[str, object],
        loads(PROVIDER_EVIDENCE.read_text(encoding="utf-8")),
    )
    evidence["production_activation_authority"] = True
    _canonicalize(evidence)
    unauthorized = tmp_path / "provider-unauthorized.json"
    _write_json(unauthorized, evidence)
    with pytest.raises(live.LiveConformancePreflightError):
        live.validate_provider_evidence(
            unauthorized,
            ACTIVATION_MANIFEST,
            today=date(2026, 8, 3),
        )

    source = cast(
        dict[str, object],
        loads(ACTIVATION_MANIFEST.read_text(encoding="utf-8")),
    )
    variants: list[dict[str, object]] = []
    unlinked = deepcopy(source)
    cast(dict[str, object], unlinked["provider_evidence"])["byte_sha256"] = (
        "0" * 64
    )
    variants.append(unlinked)

    reviewed = deepcopy(source)
    cast(dict[str, object], reviewed["review"])["status"] = "complete"
    variants.append(reviewed)

    partial = deepcopy(source)
    partial["required_live_matrix"] = partial["required_live_matrix"][:-1]  # type: ignore[index]
    variants.append(partial)

    for index, manifest in enumerate(variants):
        _canonicalize(manifest)
        path = tmp_path / f"manifest-{index}.json"
        _write_json(path, manifest)
        with pytest.raises(live.LiveConformancePreflightError):
            live.validate_provider_evidence(
                PROVIDER_EVIDENCE,
                path,
                today=date(2026, 8, 3),
            )


@pytest.mark.parametrize(
    "family",
    (
        live.LiveProviderFamily.OPENAI,
        live.LiveProviderFamily.AZURE_OPENAI,
    ),
)
def test_dump_output_strips_only_canonically_matching_parsed_arguments(
    family: Any,
) -> None:
    """Remove the SDK-only parsed field after strict semantic equality."""
    response = _parsed_tool_response()
    function_call = cast(ParsedResponseFunctionToolCall, response.output[1])
    function_call.arguments = '{ "value" : "\\u0032\\u0035\\u0030\\u0035" }'

    output = live._dump_output(response, provider_family=family)

    assert [item["type"] for item in output] == ["reasoning", "function_call"]
    assert "parsed_arguments" not in output[1]
    assert output[1]["arguments"] == function_call.arguments


@pytest.mark.parametrize(
    "mutation",
    (
        "mismatched_parsed_value",
        "parsed_value_not_mapping",
        "invalid_argument_json",
        "duplicate_argument_key",
        "nonfinite_argument_value",
        "unexpected_sdk_field",
    ),
)
def test_dump_output_rejects_unproven_parsed_arguments(mutation: str) -> None:
    """Reject parsed SDK state without exact strict JSON equivalence."""
    response = _parsed_tool_response()
    function_call = cast(ParsedResponseFunctionToolCall, response.output[1])

    match mutation:
        case "mismatched_parsed_value":
            function_call.parsed_arguments = {"value": "different"}
        case "parsed_value_not_mapping":
            function_call.parsed_arguments = ["2505"]
        case "invalid_argument_json":
            function_call.arguments = '{"value":'
        case "duplicate_argument_key":
            function_call.arguments = '{"value":"2505","value":"2505"}'
        case "nonfinite_argument_value":
            function_call.arguments = '{"value":NaN}'
            function_call.parsed_arguments = {"value": float("nan")}
        case "unexpected_sdk_field":
            function_call.__pydantic_extra__ = {"unexpected": True}
        case _:
            raise AssertionError("unhandled mutation")

    with pytest.raises(
        live.LiveConformanceAssertionError,
        match="response output cannot be normalized for exact replay",
    ):
        live._dump_output(
            response,
            provider_family=live.LiveProviderFamily.AZURE_OPENAI,
        )


def test_dump_output_rejects_parsed_argument_exclusion_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if the SDK no longer marks parsed arguments non-API."""
    response = _parsed_tool_response()
    monkeypatch.setattr(
        ParsedResponseFunctionToolCall,
        "__api_exclude__",
        set(),
    )

    with pytest.raises(live.LiveConformanceAssertionError):
        live._dump_output(
            response,
            provider_family=live.LiveProviderFamily.AZURE_OPENAI,
        )


@pytest.mark.parametrize(
    "family",
    (
        live.LiveProviderFamily.OPENAI,
        live.LiveProviderFamily.AZURE_OPENAI,
    ),
)
def test_dump_compacted_output_binds_sdk_misclassified_user_message(
    family: Any,
) -> None:
    """Normalize the observed retained-user shape without broad replay."""
    original = [_input_message()]
    compacted = _compacted_response(original)

    output = live._dump_compacted_output(
        compacted,
        provider_family=family,
        original_input=cast(Any, original),
    )

    assert output == [
        original[0],
        {
            "encrypted_content": "opaque-cmp_terminal",
            "id": "cmp_terminal",
            "type": "compaction",
        },
    ]
    assert "id" not in output[0]
    assert "status" not in output[0]


@pytest.mark.parametrize(
    "mutation",
    (
        "mutated_retained_content",
        "missing_retained_message",
        "extra_retained_message",
        "assistant_role",
        "system_role",
        "unexpected_message_field",
        "empty_content",
        "non_mapping_content_part",
        "missing_content_part_field",
        "extra_content_part_field",
        "wrong_content_part_type",
        "non_string_content_text",
        "bad_retained_order",
        "nonterminal_compaction",
        "duplicate_compaction",
        "missing_compaction",
        "empty_opaque_state",
        "non_string_opaque_state",
        "invalid_message_id",
        "invalid_message_status",
    ),
)
def test_dump_compacted_output_rejects_closed_shape_mutations(
    mutation: str,
) -> None:
    """Reject retained-input drift and malformed terminal compaction."""
    original = (
        [_input_message("first"), _input_message("second")]
        if mutation == "bad_retained_order"
        else [_input_message()]
    )
    compacted = _compacted_response(original)
    retained = cast(Any, compacted.output[0])
    terminal = cast(Any, compacted.output[-1])

    match mutation:
        case "mutated_retained_content":
            retained.content = [{"text": "changed", "type": "input_text"}]
        case "missing_retained_message":
            compacted.output.pop(0)
        case "extra_retained_message":
            extra = _compacted_response([_input_message("extra")]).output[0]
            compacted.output.insert(-1, extra)
        case "assistant_role":
            retained.role = "assistant"
        case "system_role":
            retained.role = "system"
        case "unexpected_message_field":
            retained.phase = "commentary"
        case "empty_content":
            retained.content = []
        case "non_mapping_content_part":
            retained.content = ["invalid"]
        case "missing_content_part_field":
            retained.content = [{"type": "input_text"}]
        case "extra_content_part_field":
            retained.content = [
                {
                    "detail": "auto",
                    "text": "phase12 retained input",
                    "type": "input_text",
                }
            ]
        case "wrong_content_part_type":
            retained.content = [
                {"text": "phase12 retained input", "type": "output_text"}
            ]
        case "non_string_content_text":
            retained.content = [{"text": 1, "type": "input_text"}]
        case "bad_retained_order":
            compacted.output[0], compacted.output[1] = (
                compacted.output[1],
                compacted.output[0],
            )
        case "nonterminal_compaction":
            compacted.output[:] = [terminal, retained]
        case "duplicate_compaction":
            compacted.output.append(terminal.model_copy(deep=True))
        case "missing_compaction":
            compacted.output.pop()
        case "empty_opaque_state":
            terminal.encrypted_content = ""
        case "non_string_opaque_state":
            terminal.encrypted_content = 1
        case "invalid_message_id":
            retained.id = ""
        case "invalid_message_status":
            retained.status = "incomplete"
        case _:
            raise AssertionError("unhandled mutation")

    with pytest.raises(
        live.LiveConformanceAssertionError,
        match="compaction output cannot be normalized for exact replay",
    ):
        live._dump_compacted_output(
            compacted,
            provider_family=live.LiveProviderFamily.AZURE_OPENAI,
            original_input=cast(Any, original),
        )


@pytest.mark.parametrize(
    "original_input",
    (
        [],
        [
            {
                "content": [
                    {"text": "phase12 retained input", "type": "output_text"}
                ],
                "role": "user",
                "type": "message",
            }
        ],
        [
            {
                "content": [
                    {
                        "extra": True,
                        "text": "phase12 retained input",
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
        ],
        [
            {
                "content": [
                    {"text": "phase12 retained input", "type": "input_text"}
                ],
                "role": "assistant",
                "type": "message",
            }
        ],
    ),
)
def test_dump_compacted_output_rejects_invalid_original_binding(
    original_input: list[dict[str, object]],
) -> None:
    """Reject empty or noncanonical original compact-input authority."""
    compacted = _compacted_response([_input_message()])

    with pytest.raises(live.LiveConformanceAssertionError):
        live._dump_compacted_output(
            compacted,
            provider_family=live.LiveProviderFamily.AZURE_OPENAI,
            original_input=cast(Any, original_input),
        )


def test_response_helpers_reject_missing_or_untyped_structure() -> None:
    """Reject invalid reasoning, compaction, tools, and response types."""
    assert not live._reasoning_precedes_tool(("message",))
    assert live._reasoning("current_turn") == {
        "context": "current_turn",
        "effort": "low",
    }
    assert live._reasoning("all_turns") == {
        "context": "all_turns",
        "effort": "low",
    }
    with pytest.raises(live.LiveConformancePreflightError):
        live._reasoning("auto")

    compacted = _compacted_response()
    compacted.output.clear()
    with pytest.raises(live.LiveConformanceAssertionError):
        live._dump_compacted_output(
            compacted,
            provider_family=live.LiveProviderFamily.OPENAI,
            original_input=cast(Any, [_input_message()]),
        )
    with pytest.raises(live.LiveConformanceAssertionError):
        live._dump_compacted_output(
            cast(Any, object()),
            provider_family=live.LiveProviderFamily.OPENAI,
            original_input=cast(Any, [_input_message()]),
        )
    with pytest.raises(live.LiveConformancePreflightError):
        live._provider_family(cast(Any, "openai_compatible"))

    message_only = _response(
        "resp_message",
        [_message("msg_only")],
        context="current_turn",
    )
    with pytest.raises(live.LiveConformanceAssertionError):
        live._tool_outputs(message_only)
    with pytest.raises(live.LiveConformanceAssertionError):
        live._response_observation(
            live.LiveConformanceCase.STORED_CREATE,
            cast(Any, object()),
            expected_parent=None,
        )

    opaque_free_inline = _response(
        "resp-opaque-free-inline",
        [_compaction("cmp-opaque-free-inline")],
        context="current_turn",
    )
    cast(Any, opaque_free_inline.output[0]).encrypted_content = ""
    observation = live._response_observation(
        live.LiveConformanceCase.INLINE_COMPACTION,
        opaque_free_inline,
        expected_parent=None,
    )
    assert observation.compaction_items == 1
    assert observation.opaque_compaction_items == 0
    with pytest.raises(live.LiveConformanceAssertionError):
        live._assert_observation(
            live.LiveConformanceCase.INLINE_COMPACTION,
            observation,
        )


async def test_cli_helpers_are_deterministic_and_do_not_log_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise parser, UTC clock, and redacted command output locally."""
    parser = live._build_parser()
    arguments = parser.parse_args(
        [
            "--provider",
            "openai",
            "--authorize-live-provider-conformance",
            "--acknowledge-provider-costs",
        ]
    )
    assert (await live._utc_clock()).tzinfo is UTC
    config = _config()
    receipt = live.LiveConformanceReceipt(
        provider_family=config.provider_family,
        endpoint_digest="c" * 64,
        api_form=config.api_form,
        provider_api_revision=config.provider_api_revision,
        model_or_deployment=config.model_or_deployment,
        model_or_deployment_revision=config.model_or_deployment_revision,
        model_identity_semantics=(
            "requested_model_and_response_model_revision_exact"
        ),
        observed_at=NOW,
        provider_evidence_digest="a" * 64,
        activation_manifest_digest="b" * 64,
        structural_observations_digest="d" * 64,
        completed_cases=live._EXECUTION_ORDER,
        accounting=_complete_accounting(),
    )

    async def fake_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return receipt

    monkeypatch.setattr(live, "_config_from_environment", lambda *args: config)
    monkeypatch.setattr(live, "run_live_conformance", fake_run)
    assert await live._main(arguments) == 0
    output = capsys.readouterr().out
    assert "sk-live-conformance-fixture" not in output
    assert loads(output)["opaque_payloads_logged"] is False
