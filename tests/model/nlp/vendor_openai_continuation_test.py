"""Test durable OpenAI Responses continuation snapshot ownership."""

from asyncio import run as asyncio_run
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from json import dumps, loads
from subprocess import run
from sys import executable
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, cast, overload
from unittest.mock import patch

import openai
import pytest
from httpx import AsyncClient, MockTransport, Request, Response

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    ToolCall,
    ToolCallResult,
)
from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    CapabilityRevision,
    ContinuationDispatchId,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    decode_continuation_snapshot,
    encode_continuation_snapshot,
)
from avalan.interaction.continuation import derive_provider_idempotency_key
from avalan.interaction.error import (
    InputErrorCode,
    InputSnapshotError,
    InputValidationError,
)
from avalan.model import (
    ContinuationSnapshotCodecRegistry,
    CorrelatedCapabilityResult,
)
from avalan.model.nlp.text.vendor import openai as openai_module
from avalan.model.nlp.text.vendor.openai import (
    OpenAIClient,
    OpenAIModel,
    OpenAIStream,
)
from avalan.model.stream import StreamRetentionPolicy
from avalan.types import JsonValue

_CALL_ID = "call-input"
_PRIOR_CALL_ID = "call-prior-input"
_CONTINUATION_ID = ContinuationId("continuation-input")
_DISPATCH_ID = ContinuationDispatchId("dispatch-input")
_PROVIDER_IDEMPOTENCY_KEY = derive_provider_idempotency_key(
    _CONTINUATION_ID,
    _DISPATCH_ID,
)
_REAL_ASYNC_OPENAI = openai.AsyncOpenAI


class _ValidRequirementModeString(str):
    """Represent a valid-valued non-exact requirement mode string."""


class _OpenAIIdempotencyTransport:
    def __init__(self) -> None:
        self.headers: list[dict[str, str | None]] = []
        self.http_clients: list[AsyncClient] = []
        self.resumed_attempts = 0

    async def handle(self, request: Request) -> Response:
        idempotency_key = request.headers.get("Idempotency-Key")
        self.headers.append(
            {
                "HTTP-Referer": request.headers.get("HTTP-Referer"),
                "Idempotency-Key": idempotency_key,
                "X-Title": request.headers.get("X-Title"),
            }
        )
        if idempotency_key is not None:
            self.resumed_attempts += 1
            if self.resumed_attempts <= 2:
                return Response(
                    500,
                    headers={"retry-after-ms": "0"},
                    json={
                        "error": {
                            "code": "server_error",
                            "message": "retry",
                            "param": None,
                            "type": "server_error",
                        }
                    },
                )
        return Response(
            200,
            json={
                "id": f"response-{len(self.headers)}",
                "created_at": 0.0,
                "model": "gpt-5",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "status": "completed",
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1,
                },
            },
        )

    def client_factory(self, **kwargs: object) -> openai.AsyncOpenAI:
        assert set(kwargs) == {"api_key", "base_url"}
        api_key = kwargs["api_key"]
        base_url = kwargs["base_url"]
        assert isinstance(api_key, str)
        assert base_url is None or isinstance(base_url, str)
        http_client = AsyncClient(transport=MockTransport(self.handle))
        self.http_clients.append(http_client)
        return _REAL_ASYNC_OPENAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )


def _input_arguments(
    *,
    reason: str = "Need input.",
) -> dict[str, object]:
    return {
        "mode": "required",
        "reason": reason,
        "questions": [
            {
                "allow_other": False,
                "choices": [],
                "kind": "confirmation",
                "prompt": "Continue?",
                "question_id": "continue",
                "required": True,
            }
        ],
    }


def _resolved_input_message() -> Message:
    call = ToolCall(
        id=_CALL_ID,
        name=RESERVED_INPUT_CAPABILITY_NAME,
        arguments={},
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
    )
    result = ToolCallResult(
        id=_CALL_ID,
        name=RESERVED_INPUT_CAPABILITY_NAME,
        call=call,
        result={"accepted": True},
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
    )
    return Message(
        role=MessageRole.TOOL,
        tool_call_result=result,
    )


def _binding(**overrides: object) -> ContinuationRevisionBinding:
    values: dict[str, Any] = {
        "provider_family": ProviderFamilyName("openai"),
        "model_id": ModelId("gpt-5"),
        "provider_config_revision": ProviderConfigRevision("provider-r1"),
        "model_config_revision": ModelConfigRevision("model-r1"),
        "capability_revision": CapabilityRevision("capability-r1"),
    }
    values.update(overrides)
    return ContinuationRevisionBinding(**values)


def _client(
    *,
    policy: StreamRetentionPolicy | None = None,
    base_url: str | None = "https://api.openai.com/v1",
) -> OpenAIClient:
    client = object.__new__(OpenAIClient)
    client._base_url = base_url  # noqa: SLF001
    client._is_azure = OpenAIClient._is_azure_base_url(  # noqa: SLF001
        base_url
    )
    client._stream_retention_policy = (  # noqa: SLF001
        policy or StreamRetentionPolicy()
    )
    client._replay_owners_by_call_id = {}  # noqa: SLF001
    client._active_replay_owners = {}  # noqa: SLF001
    client._active_replay_streams = {}  # noqa: SLF001
    client._active_replay_call_ids = {}  # noqa: SLF001
    client._ambiguous_replay_call_ids = {}  # noqa: SLF001
    client._replay_association_poisoned = False  # noqa: SLF001
    client._closed = False  # noqa: SLF001
    return client


def _model(
    *,
    client: OpenAIClient | None = None,
    model_id: str = "gpt-5",
) -> OpenAIModel:
    model = object.__new__(OpenAIModel)
    model._model = client or _client()  # noqa: SLF001
    model._model_id = model_id  # noqa: SLF001
    model._continuation_capability_support = None  # noqa: SLF001
    return model


def _replay_items(
    *,
    reasoning_id: object = "rs_1",
    encrypted_content: object = "ciphertext",
    call_id: object = _CALL_ID,
    function_id: object = "fc_1",
    name: object = "request_user_input",
    arguments: object | None = None,
) -> tuple[dict[str, object], ...]:
    return (
        {
            "id": reasoning_id,
            "type": "reasoning",
            "encrypted_content": encrypted_content,
            "summary": [
                {"type": "summary_text", "text": "Need input."},
            ],
        },
        {
            "id": function_id,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": (
                dumps(
                    _input_arguments(),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if arguments is None
                else arguments
            ),
        },
    )


def _sequential_replay_items(
    *,
    historical_arguments: object | None = None,
    historical_name: object = "request_user_input",
    current_arguments: object | None = None,
    include_historical_reasoning: bool = True,
    include_current_reasoning: bool = True,
    duplicate_current: bool = False,
) -> tuple[dict[str, object], ...]:
    historical = _replay_items(
        reasoning_id="rs_prior",
        encrypted_content="prior-ciphertext",
        call_id=_PRIOR_CALL_ID,
        function_id="fc_prior",
        name=historical_name,
        arguments=(
            dumps(
                _input_arguments(reason="Need earlier input."),
                separators=(",", ":"),
                sort_keys=True,
            )
            if historical_arguments is None
            else historical_arguments
        ),
    )
    current = _replay_items(
        reasoning_id="rs_current",
        encrypted_content="current-ciphertext",
        arguments=current_arguments,
    )
    historical_items = (
        historical if include_historical_reasoning else historical[1:]
    )
    current_items = current if include_current_reasoning else current[1:]
    result = (*historical_items, *current_items)
    if duplicate_current:
        result = (*result, dict(current[1]))
    return result


def _retain(
    client: OpenAIClient,
    items: tuple[dict[str, object], ...] | None = None,
) -> None:
    owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        client._stream_retention_policy  # noqa: SLF001
    )
    owner.begin_attempt()
    for item in items or _replay_items():
        assert owner.admit(cast(dict[str, Any], item))
    owner.commit_attempt()
    client._retain_replay_owner(owner, (_CALL_ID,))  # noqa: SLF001


def _reserved_boundary_manifest(
    items: object,
    *,
    current_call_id: str = _CALL_ID,
) -> tuple[dict[str, object], ...]:
    source_items = (
        items
        if isinstance(items, tuple | list)
        and all(isinstance(item, Mapping) for item in items)
        else _replay_items()
    )
    reasoning: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    for item in source_items:
        assert isinstance(item, Mapping)
        if item.get("type") == "reasoning":
            item_id = item.get("id")
            encrypted_content = item.get("encrypted_content")
            reasoning.append(
                {
                    "item_id": (
                        item_id
                        if type(item_id) is str
                        else "invalid-reasoning"
                    ),
                    "encrypted_content_sha256": (
                        sha256(
                            (
                                encrypted_content
                                if type(encrypted_content) is str
                                else "invalid-reasoning"
                            ).encode("utf-8")
                        ).hexdigest()
                    ),
                }
            )
            continue
        if item.get("type") != "function_call":
            continue
        raw_arguments = item.get("arguments")
        decoded_arguments: object = {}
        try:
            decoded_arguments = (
                loads(raw_arguments) if type(raw_arguments) is str else {}
            )
            canonical_arguments = dumps(
                decoded_arguments,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError):
            canonical_arguments = "{}"
        call_id = item.get("call_id")
        provider_name = item.get("name")
        if (
            call_id == current_call_id
            or provider_name == "request_user_input"
            or (
                isinstance(decoded_arguments, dict)
                and set(decoded_arguments) == {"mode", "reason", "questions"}
            )
        ):
            boundaries.append(
                {
                    "call_id": (
                        call_id if type(call_id) is str else "invalid-call"
                    ),
                    "provider_name": (
                        provider_name
                        if type(provider_name) is str
                        else "invalid-provider"
                    ),
                    "canonical_arguments": canonical_arguments,
                    "reasoning": tuple(reasoning),
                }
            )
        reasoning = []
    return tuple(boundaries)


def _snapshot(
    binding: ContinuationRevisionBinding | None = None,
    *,
    payload: dict[str, object] | None = None,
    snapshot_kind: str = "openai.responses.reasoning",
    include_manifest: bool = True,
    boundary_manifest: tuple[dict[str, object], ...] | None = None,
) -> ContinuationSnapshot:
    values = dict(
        payload
        or {
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": _replay_items(),
        }
    )
    if (
        include_manifest
        and "reserved_capability_call_id" in values
        and "replay_items" in values
        and "reserved_capability_boundaries" not in values
    ):
        current_call_id = values["reserved_capability_call_id"]
        values["reserved_capability_boundaries"] = (
            boundary_manifest
            if boundary_manifest is not None
            else _reserved_boundary_manifest(
                values["replay_items"],
                current_call_id=(
                    current_call_id
                    if type(current_call_id) is str
                    else _CALL_ID
                ),
            )
        )
    snapshot_payload = cast(
        Mapping[str, JsonValue],
        values,
    )
    return ContinuationSnapshot(
        snapshot_kind=snapshot_kind,
        revision_binding=binding or _binding(),
        model_call_id=ModelCallId("model-call"),
        provider_idempotency_key=_PROVIDER_IDEMPOTENCY_KEY,
        payload=snapshot_payload,
    )


async def _capture_resumed_request_headers() -> list[dict[str, str | None]]:
    transport = _OpenAIIdempotencyTransport()
    with patch.object(
        openai,
        "AsyncOpenAI",
        side_effect=transport.client_factory,
    ):
        client = OpenAIClient(api_key="test-key", base_url=None)
        try:
            restored = decode_continuation_snapshot(
                encode_continuation_snapshot(_snapshot()),
                expected_binding=_binding(),
            )
            client.import_continuation_snapshot(
                restored,
                expected_binding=_binding(),
                provider_call_correlation_id=_CALL_ID,
            )
            settings = GenerationSettings(
                openai_max_retries=2,
                use_async_generator=False,
            )
            unrelated = Message(
                role=MessageRole.USER,
                content="Unrelated request.",
            )
            await client(
                "gpt-5",
                [unrelated],
                settings,
                use_async_generator=False,
            )
            await client(
                "gpt-5",
                [_resolved_input_message()],
                settings,
                use_async_generator=False,
            )
            await client(
                "gpt-5",
                [unrelated],
                settings,
                use_async_generator=False,
            )
        finally:
            await client.aclose()
    return transport.headers


def test_imported_provider_key_is_exact_on_every_resumed_retry() -> None:
    assert _snapshot().provider_idempotency_key == _PROVIDER_IDEMPOTENCY_KEY
    headers = asyncio_run(_capture_resumed_request_headers())

    assert [
        request_headers["Idempotency-Key"] for request_headers in headers
    ] == [
        None,
        str(_PROVIDER_IDEMPOTENCY_KEY),
        str(_PROVIDER_IDEMPOTENCY_KEY),
        str(_PROVIDER_IDEMPOTENCY_KEY),
        None,
    ]
    assert all(
        request_headers["X-Title"] == "Avalan"
        and request_headers["HTTP-Referer"]
        == "https://github.com/avalan-ai/avalan"
        for request_headers in headers
    )


def test_openai_snapshot_round_trip_preserves_replay_identity() -> None:
    source = _client()
    _retain(source)
    binding = _binding()

    snapshot = source.export_continuation_snapshot(
        revision_binding=binding,
        model_call_id=ModelCallId("model-call"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-retry"),
        provider_call_correlation_id=_CALL_ID,
    )

    assert snapshot.snapshot_kind == "openai.responses.reasoning"
    items = snapshot.payload["replay_items"]
    assert isinstance(items, tuple)
    first_item = items[0]
    assert isinstance(first_item, Mapping)
    assert first_item["id"] == "rs_1"
    assert first_item["encrypted_content"] == "ciphertext"
    assert snapshot.payload["reserved_capability_call_id"] == _CALL_ID

    registry = ContinuationSnapshotCodecRegistry("openai-registry")
    reference = OpenAIClient.register_continuation_snapshot_codec(
        registry,
        codec_id="openai-responses-v1",
        revision_binding=binding,
    )
    encoded = registry.export_snapshot(reference, snapshot)
    restored = registry.restore_snapshot(reference, encoded, binding)
    assert (
        decode_continuation_snapshot(
            encoded,
            expected_binding=binding,
        )
        == snapshot
    )

    fresh = _client()
    fresh.import_continuation_snapshot(
        restored,
        expected_binding=binding,
        provider_call_correlation_id=_CALL_ID,
    )
    owner = fresh._replay_owners_by_call_id[_CALL_ID]  # noqa: SLF001
    replay_items = owner.replay_items()
    assert replay_items[0]["id"] == "rs_1"
    assert replay_items[0]["encrypted_content"] == "ciphertext"
    assert replay_items[1]["call_id"] == _CALL_ID
    assert "api_key" not in encoded


def test_openai_snapshot_exports_sequential_active_replay_owner() -> None:
    first_source = _client()
    first_owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        first_source._stream_retention_policy  # noqa: SLF001
    )
    first_owner.begin_attempt()
    prior_items = _sequential_replay_items()[:2]
    for item in prior_items:
        assert first_owner.admit(cast(dict[str, Any], item))
    first_source._activate_replay_owner(first_owner)  # noqa: SLF001
    first_snapshot = first_source.export_continuation_snapshot(
        revision_binding=_binding(),
        model_call_id=ModelCallId("first-model-call"),
        provider_idempotency_key=ProviderIdempotencyKey("first-retry"),
        provider_call_correlation_id=_PRIOR_CALL_ID,
    )
    first_source.validate_continuation_snapshot_call(
        first_snapshot,
        expected_binding=_binding(),
        provider_call_correlation_id=_PRIOR_CALL_ID,
        expected_provider_name="request_user_input",
        expected_arguments=cast(
            Mapping[str, JsonValue],
            _input_arguments(reason="Need earlier input."),
        ),
    )

    source = _client()
    source.import_continuation_snapshot(
        first_snapshot,
        expected_binding=_binding(),
        provider_call_correlation_id=_PRIOR_CALL_ID,
    )
    owner = source._replay_owners_by_call_id.pop(  # noqa: SLF001
        _PRIOR_CALL_ID
    )
    source._activate_replay_owner(owner)  # noqa: SLF001
    source._active_replay_call_ids[_PRIOR_CALL_ID] = owner  # noqa: SLF001
    owner.begin_attempt()
    for item in _sequential_replay_items()[2:]:
        assert owner.admit(cast(dict[str, Any], item))

    snapshot = source.export_continuation_snapshot(
        revision_binding=_binding(),
        model_call_id=ModelCallId("model-call"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-retry"),
        provider_call_correlation_id=_CALL_ID,
    )
    source.validate_continuation_snapshot_call(
        snapshot,
        expected_binding=_binding(),
        provider_call_correlation_id=_CALL_ID,
        expected_provider_name="request_user_input",
        expected_arguments=cast(Mapping[str, JsonValue], _input_arguments()),
    )

    assert not owner.released
    owner.release()
    fresh = _client()
    fresh.import_continuation_snapshot(
        snapshot,
        expected_binding=_binding(),
        provider_call_correlation_id=_CALL_ID,
    )
    restored_items = fresh._replay_owners_by_call_id[
        _CALL_ID
    ].replay_items()  # noqa: SLF001
    assert restored_items[0]["encrypted_content"] == "prior-ciphertext"
    assert restored_items[1]["call_id"] == _PRIOR_CALL_ID
    assert restored_items[2]["encrypted_content"] == "current-ciphertext"
    assert restored_items[3]["call_id"] == _CALL_ID
    assert restored_items[3]["name"] == "request_user_input"


def test_native_openai_model_registers_exact_durable_revision() -> None:
    model = _model()
    binding = _binding()

    initial = model.provider_capability_support
    registered = model.register_continuation_snapshot_codec(binding)
    replayed = model.register_continuation_snapshot_codec(binding)

    assert initial.structured_invocation
    assert initial.continuation_snapshot_codec is None
    assert registered is replayed
    assert model.provider_capability_support is registered
    assert registered.structured_invocation
    assert registered.stable_call_ids
    assert registered.correlated_results
    registry = registered.continuation_snapshot_codec_registry
    codec = registered.continuation_snapshot_codec
    assert registry is not None
    assert codec is not None
    assert codec.revision_binding == binding
    assert registry.is_registered(codec)


@pytest.mark.parametrize(
    ("base_url", "native"),
    (
        ("https://api.openai.com/v1", True),
        ("https://compatible.example/v1", False),
        ("http://api.openai.com/v1", False),
    ),
)
def test_durable_registration_uses_sdk_environment_base_url(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    native: bool,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    client = OpenAIClient(api_key="test-key", base_url=None)
    model = _model(client=client)
    try:
        if native:
            support = model.register_continuation_snapshot_codec(_binding())
            assert support.continuation_snapshot_codec is not None
        else:
            with pytest.raises(InputSnapshotError) as raised:
                model.register_continuation_snapshot_codec(_binding())
            assert (
                raised.value.code
                is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE
            )
            assert (
                model.provider_capability_support.continuation_snapshot_codec
                is None
            )
    finally:
        asyncio_run(client.aclose())


@pytest.mark.parametrize(
    "sdk_base_url",
    (
        "https://compatible.example/v1",
        object(),
    ),
)
def test_durable_registration_fails_closed_for_sdk_endpoint(
    sdk_base_url: object,
) -> None:
    client = _client()
    client._client = SimpleNamespace(base_url=sdk_base_url)  # noqa: SLF001
    model = _model(client=client)

    with pytest.raises(InputSnapshotError) as raised:
        model.register_continuation_snapshot_codec(_binding())

    assert raised.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE


def test_durable_registration_requires_an_effective_endpoint() -> None:
    model = _model(client=_client(base_url=None))

    with pytest.raises(InputSnapshotError) as raised:
        model.register_continuation_snapshot_codec(_binding())

    assert raised.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE


def test_native_openai_model_registration_rejects_revision_drift() -> None:
    model = _model()
    model.register_continuation_snapshot_codec(_binding())

    with pytest.raises(InputSnapshotError) as raised:
        model.register_continuation_snapshot_codec(
            _binding(model_config_revision=ModelConfigRevision("model-r2"))
        )

    assert raised.value.code is InputErrorCode.SNAPSHOT_REVISION_DRIFT


@pytest.mark.parametrize(
    ("model_id", "binding", "base_url"),
    (
        ("gpt-4o", _binding(model_id=ModelId("gpt-4o")), None),
        ("gpt-5", _binding(model_id=ModelId("other")), None),
        (
            "gpt-5",
            _binding(),
            "https://compatible.example/v1",
        ),
        (
            "gpt-5",
            _binding(provider_family=ProviderFamilyName("azure_openai")),
            None,
        ),
    ),
)
def test_openai_model_durable_registration_fails_closed(
    model_id: str,
    binding: ContinuationRevisionBinding,
    base_url: str | None,
) -> None:
    model = _model(
        client=_client(
            base_url=(
                "https://api.openai.com/v1" if base_url is None else base_url
            )
        ),
        model_id=model_id,
    )

    with pytest.raises(InputSnapshotError) as raised:
        model.register_continuation_snapshot_codec(binding)

    assert raised.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE
    assert (
        model.provider_capability_support.continuation_snapshot_codec is None
    )


def test_openai_compatible_subclasses_never_register_durable_replay() -> None:
    class CompatibleOpenAIModel(OpenAIModel):
        pass

    class CompatibleOpenAIClient(OpenAIClient):
        pass

    compatible_model = object.__new__(CompatibleOpenAIModel)
    compatible_model._model = _client()  # noqa: SLF001
    compatible_model._model_id = "gpt-5"  # noqa: SLF001
    compatible_model._continuation_capability_support = None  # noqa: SLF001
    compatible_client = cast(
        OpenAIClient,
        object.__new__(CompatibleOpenAIClient),
    )
    compatible_client._base_url = "https://api.openai.com/v1"  # noqa: SLF001
    compatible_client_model = _model(
        client=compatible_client,
    )

    for model in (compatible_model, compatible_client_model):
        assert not model.provider_capability_support.structured_invocation
        with pytest.raises(InputSnapshotError) as raised:
            model.register_continuation_snapshot_codec(_binding())
        assert (
            raised.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE
        )


def test_openai_snapshot_restores_in_fresh_process() -> None:
    source = _client()
    _retain(source)
    binding = _binding()
    snapshot = source.export_continuation_snapshot(
        revision_binding=binding,
        model_call_id=ModelCallId("model-call"),
        provider_idempotency_key=ProviderIdempotencyKey("provider-retry"),
        provider_call_correlation_id=_CALL_ID,
    )
    encoded = encode_continuation_snapshot(snapshot)
    script = dedent("""
        import json
        import sys

        from avalan.interaction import (
            CapabilityRevision,
            ContinuationRevisionBinding,
            ModelConfigRevision,
            ModelId,
            ProviderConfigRevision,
            ProviderFamilyName,
            decode_continuation_snapshot,
        )
        from avalan.model.nlp.text.vendor.openai import OpenAIClient

        binding = ContinuationRevisionBinding(
            provider_family=ProviderFamilyName("openai"),
            model_id=ModelId("gpt-5"),
            provider_config_revision=ProviderConfigRevision("provider-r1"),
            model_config_revision=ModelConfigRevision("model-r1"),
            capability_revision=CapabilityRevision("capability-r1"),
        )
        snapshot = decode_continuation_snapshot(
            sys.argv[1],
            expected_binding=binding,
        )
        client = OpenAIClient(
            api_key="fresh-process-test-key",
            base_url="https://api.openai.com/v1",
        )
        client.import_continuation_snapshot(
            snapshot,
            expected_binding=binding,
            provider_call_correlation_id="call-input",
        )
        owner = client._replay_owners_by_call_id["call-input"]
        items = owner.replay_items()
        print(
            json.dumps(
                {
                    "reasoning_id": items[0]["id"],
                    "encrypted_content": items[0]["encrypted_content"],
                    "reserved_call_id": items[1]["call_id"],
                },
                sort_keys=True,
            )
        )
        """)

    result = run(
        [executable, "-c", script, encoded],
        capture_output=True,
        check=False,
        cwd=".",
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert loads(result.stdout) == {
        "encrypted_content": "ciphertext",
        "reasoning_id": "rs_1",
        "reserved_call_id": _CALL_ID,
    }


def test_openai_snapshot_registration_rejects_invalid_inputs() -> None:
    with pytest.raises(InputValidationError, match="registry"):
        OpenAIClient.register_continuation_snapshot_codec(
            cast(ContinuationSnapshotCodecRegistry, object()),
            codec_id="codec",
            revision_binding=_binding(),
        )
    with pytest.raises(InputSnapshotError) as raised:
        OpenAIClient.register_continuation_snapshot_codec(
            ContinuationSnapshotCodecRegistry("registry"),
            codec_id="codec",
            revision_binding=_binding(
                provider_family=ProviderFamilyName("anthropic")
            ),
        )
    assert raised.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE
    with pytest.raises(InputSnapshotError, match="typed revision"):
        OpenAIClient.register_continuation_snapshot_codec(
            ContinuationSnapshotCodecRegistry("registry"),
            codec_id="codec",
            revision_binding=cast(ContinuationRevisionBinding, object()),
        )


def test_openai_snapshot_export_fails_closed_without_safe_state() -> None:
    binding = _binding()
    client = _client()
    with pytest.raises(InputSnapshotError, match="unavailable or ambiguous"):
        client.export_continuation_snapshot(
            revision_binding=binding,
            model_call_id=ModelCallId("model-call"),
            provider_idempotency_key=ProviderIdempotencyKey("key"),
            provider_call_correlation_id=_CALL_ID,
        )

    _retain(client)
    client._ambiguous_replay_call_ids[_CALL_ID] = None  # noqa: SLF001
    with pytest.raises(InputSnapshotError, match="unavailable or ambiguous"):
        client.export_continuation_snapshot(
            revision_binding=binding,
            model_call_id=ModelCallId("model-call"),
            provider_idempotency_key=ProviderIdempotencyKey("key"),
            provider_call_correlation_id=_CALL_ID,
        )

    client._closed = True  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed"):
        client.export_continuation_snapshot(
            revision_binding=binding,
            model_call_id=ModelCallId("model-call"),
            provider_idempotency_key=ProviderIdempotencyKey("key"),
            provider_call_correlation_id=_CALL_ID,
        )


@pytest.mark.parametrize(
    ("items", "message"),
    (
        (
            _replay_items(reasoning_id=""),
            "reasoning replay identity",
        ),
        (
            _replay_items(encrypted_content=""),
            "reasoning replay identity",
        ),
        (
            _replay_items(call_id="other"),
            "exactly one current reserved call",
        ),
        (
            (_replay_items()[1],),
            "current reserved call lacks fresh encrypted reasoning",
        ),
    ),
)
def test_openai_snapshot_export_rejects_incomplete_replay_state(
    items: tuple[dict[str, object], ...],
    message: str,
) -> None:
    client = _client()
    owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        client._stream_retention_policy  # noqa: SLF001
    )
    owner._items = [dict(item) for item in items]  # noqa: SLF001
    client._replay_owners_by_call_id[_CALL_ID] = owner  # noqa: SLF001

    with pytest.raises(InputSnapshotError, match=message):
        client.export_continuation_snapshot(
            revision_binding=_binding(),
            model_call_id=ModelCallId("model-call"),
            provider_idempotency_key=ProviderIdempotencyKey("key"),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_import_rejects_kind_provider_model_and_revision() -> (
    None
):
    binding = _binding()
    client = _client()
    with pytest.raises(InputSnapshotError, match="continuation snapshot"):
        client.import_continuation_snapshot(
            cast(ContinuationSnapshot, object()),
            expected_binding=binding,
            provider_call_correlation_id=_CALL_ID,
        )
    with pytest.raises(InputSnapshotError) as raised:
        client.import_continuation_snapshot(
            _snapshot(snapshot_kind="other.kind"),
            expected_binding=binding,
            provider_call_correlation_id=_CALL_ID,
        )
    assert raised.value.code is InputErrorCode.SNAPSHOT_UNSUPPORTED

    for actual, expected, code in (
        (
            _binding(provider_family=ProviderFamilyName("azure_openai")),
            binding,
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
        ),
        (
            _binding(model_id=ModelId("other")),
            binding,
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
        ),
        (
            _binding(model_config_revision=ModelConfigRevision("model-r2")),
            binding,
            InputErrorCode.SNAPSHOT_REVISION_DRIFT,
        ),
    ):
        with pytest.raises(InputSnapshotError) as raised:
            client.import_continuation_snapshot(
                _snapshot(actual),
                expected_binding=expected,
                provider_call_correlation_id=_CALL_ID,
            )
        assert raised.value.code is code


@pytest.mark.parametrize(
    "provider_idempotency_key",
    ("provider retry", "provider-rétry"),
)
def test_openai_snapshot_import_rejects_transport_unsafe_provider_key(
    provider_idempotency_key: str,
) -> None:
    snapshot = replace(
        _snapshot(),
        provider_idempotency_key=ProviderIdempotencyKey(
            provider_idempotency_key
        ),
    )

    with pytest.raises(
        InputSnapshotError,
        match="provider_idempotency_key",
    ) as raised:
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )

    assert raised.value.code is InputErrorCode.SNAPSHOT_INVALID


@pytest.mark.parametrize(
    ("snapshot", "call_id", "message"),
    (
        (
            _snapshot(payload={"extra": True}),
            _CALL_ID,
            "payload fields",
        ),
        (
            _snapshot(),
            "other-call",
            "correlation does not match",
        ),
        (
            _snapshot(
                payload={
                    "reserved_capability_call_id": _CALL_ID,
                    "replay_items": "not-items",
                }
            ),
            _CALL_ID,
            "immutable JSON sequence",
        ),
        (
            _snapshot(
                payload={
                    "reserved_capability_call_id": _CALL_ID,
                    "replay_items": (1,),
                }
            ),
            _CALL_ID,
            "must be JSON objects",
        ),
        (
            _snapshot(
                payload={
                    "reserved_capability_call_id": _CALL_ID,
                    "replay_items": _replay_items(call_id="other"),
                }
            ),
            _CALL_ID,
            "current boundary correlation does not match",
        ),
    ),
)
def test_openai_snapshot_import_rejects_invalid_payload(
    snapshot: ContinuationSnapshot,
    call_id: str,
    message: str,
) -> None:
    with pytest.raises(InputSnapshotError, match=message):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=call_id,
        )


@pytest.mark.parametrize("call_id", ("", 1, None))
def test_openai_snapshot_rejects_invalid_call_id(call_id: object) -> None:
    with pytest.raises(InputSnapshotError, match="call ID is invalid"):
        _client().import_continuation_snapshot(
            _snapshot(),
            expected_binding=_binding(),
            provider_call_correlation_id=cast(str, call_id),
        )


def test_openai_snapshot_import_rejects_collision_and_poisoning() -> None:
    for configure in (
        lambda client: client._replay_owners_by_call_id.__setitem__(  # noqa: SLF001
            _CALL_ID,
            openai_module._OpenAIReplayOwner(  # noqa: SLF001
                client._stream_retention_policy  # noqa: SLF001
            ),
        ),
        lambda client: client._ambiguous_replay_call_ids.__setitem__(  # noqa: SLF001
            _CALL_ID,
            None,
        ),
        lambda client: setattr(
            client,
            "_replay_association_poisoned",
            True,
        ),
    ):
        client = _client()
        configure(client)
        with pytest.raises(
            InputSnapshotError,
            match="destination is unavailable",
        ):
            client.import_continuation_snapshot(
                _snapshot(),
                expected_binding=_binding(),
                provider_call_correlation_id=_CALL_ID,
            )


def test_openai_snapshot_import_converts_retention_failure() -> None:
    policy = replace(
        StreamRetentionPolicy(),
        openai_replay_item_limit=1,
    )
    client = _client(policy=policy)

    with pytest.raises(InputSnapshotError, match="could not be restored"):
        client.import_continuation_snapshot(
            _snapshot(),
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )

    assert not client._replay_owners_by_call_id  # noqa: SLF001


def test_openai_snapshot_import_rejects_non_replay_item() -> None:
    items = (*_replay_items(), {"type": "message", "content": "unsafe"})
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": items,
        },
        boundary_manifest=_reserved_boundary_manifest(_replay_items()),
    )

    with pytest.raises(InputSnapshotError, match="final replay item"):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


@pytest.mark.parametrize(
    ("items", "message"),
    (
        (
            _replay_items(call_id=""),
            "function call replay state is incomplete",
        ),
        (
            _replay_items(name="other"),
            "reserved boundary manifest drifted",
        ),
        (
            _replay_items(arguments="{"),
            "arguments are invalid JSON",
        ),
        (
            _replay_items(
                arguments=(
                    '{"mode":"required","mode":"advisory",'
                    '"reason":"Need input.","questions":[]}'
                )
            ),
            "arguments are invalid JSON",
        ),
        (
            _replay_items(arguments="[]"),
            "arguments must be a JSON object",
        ),
        (
            _replay_items(
                arguments=dumps(
                    {
                        "mode": "required",
                        "reason": "Need input.",
                        "questions": [],
                    }
                )
            ),
            "reserved boundary manifest drifted",
        ),
        (
            (
                *_replay_items(),
                _replay_items(
                    function_id="fc_2",
                )[1],
            ),
            "function call replay IDs must be unique",
        ),
    ),
)
def test_openai_snapshot_rejects_tampered_reserved_call(
    items: tuple[dict[str, object], ...],
    message: str,
) -> None:
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": items,
        },
        boundary_manifest=_reserved_boundary_manifest(_replay_items()),
    )

    with pytest.raises(InputSnapshotError, match=message):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


@pytest.mark.parametrize(
    ("items", "message"),
    (
        (
            _sequential_replay_items(
                include_historical_reasoning=False,
            ),
            "reserved boundary manifest drifted",
        ),
        (
            _sequential_replay_items(include_current_reasoning=False),
            "reserved boundary manifest drifted",
        ),
        (
            _sequential_replay_items(duplicate_current=True),
            "function call replay IDs must be unique",
        ),
        (
            _sequential_replay_items(historical_name="other"),
            "reserved boundary manifest drifted",
        ),
        (
            _sequential_replay_items(
                historical_arguments=dumps(
                    {
                        "mode": "required",
                        "reason": "Need earlier input.",
                        "questions": [],
                    }
                )
            ),
            "reserved boundary manifest drifted",
        ),
    ),
)
def test_openai_sequential_snapshot_rejects_boundary_tampering(
    items: tuple[dict[str, object], ...],
    message: str,
) -> None:
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": items,
        },
        boundary_manifest=_reserved_boundary_manifest(
            _sequential_replay_items()
        ),
    )

    with pytest.raises(InputSnapshotError, match=message):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_rejects_trailing_reasoning_after_current_call() -> (
    None
):
    items = (
        *_replay_items(),
        {
            "id": "rs_trailing",
            "type": "reasoning",
            "encrypted_content": "trailing-ciphertext",
        },
    )
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": items,
        },
        boundary_manifest=_reserved_boundary_manifest(_replay_items()),
    )

    with pytest.raises(InputSnapshotError, match="final replay item"):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_rejects_duplicate_historical_call_id() -> None:
    items = list(_sequential_replay_items())
    items.insert(2, dict(items[1]))
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": tuple(items),
        },
        boundary_manifest=_reserved_boundary_manifest(
            _sequential_replay_items()
        ),
    )

    with pytest.raises(InputSnapshotError, match="call replay IDs"):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_rejects_duplicate_reasoning_id() -> None:
    items = list(_sequential_replay_items())
    current_reasoning = dict(items[2])
    current_reasoning["id"] = "rs_prior"
    items[2] = current_reasoning
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": tuple(items),
        },
        boundary_manifest=_reserved_boundary_manifest(
            _sequential_replay_items()
        ),
    )

    with pytest.raises(InputSnapshotError, match="reasoning replay IDs"):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_rejects_combined_historical_call_tampering() -> None:
    items = list(_sequential_replay_items())
    historical_call = dict(items[1])
    historical_call["name"] = "other"
    historical_call["arguments"] = dumps(
        _input_arguments(reason="Tampered historical input."),
        separators=(",", ":"),
        sort_keys=True,
    )
    items[1] = historical_call
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": tuple(items),
        },
        boundary_manifest=_reserved_boundary_manifest(
            _sequential_replay_items()
        ),
    )

    with pytest.raises(InputSnapshotError, match="manifest drifted"):
        _client().import_continuation_snapshot(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )


def test_openai_snapshot_rejects_missing_duplicate_or_reordered_manifest() -> (
    None
):
    items = _sequential_replay_items()
    manifest = _reserved_boundary_manifest(items)
    snapshots = (
        _snapshot(
            payload={
                "reserved_capability_call_id": _CALL_ID,
                "replay_items": items,
            },
            include_manifest=False,
        ),
        _snapshot(
            payload={
                "reserved_capability_call_id": _CALL_ID,
                "replay_items": items,
            },
            boundary_manifest=manifest[1:],
        ),
        _snapshot(
            payload={
                "reserved_capability_call_id": _CALL_ID,
                "replay_items": items,
            },
            boundary_manifest=(*manifest, manifest[-1]),
        ),
        _snapshot(
            payload={
                "reserved_capability_call_id": _CALL_ID,
                "replay_items": items,
            },
            boundary_manifest=tuple(reversed(manifest)),
        ),
    )

    for snapshot in snapshots:
        with pytest.raises(InputSnapshotError):
            _client().import_continuation_snapshot(
                snapshot,
                expected_binding=_binding(),
                provider_call_correlation_id=_CALL_ID,
            )


def test_openai_snapshot_exact_call_validation_rejects_changed_arguments() -> (
    None
):
    changed = _input_arguments(reason="Changed reason.")
    snapshot = _snapshot(
        payload={
            "reserved_capability_call_id": _CALL_ID,
            "replay_items": _sequential_replay_items(
                current_arguments=dumps(
                    changed,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        }
    )
    client = _client()

    with pytest.raises(InputSnapshotError, match="arguments changed"):
        client.validate_continuation_snapshot_call(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
            expected_provider_name="request_user_input",
            expected_arguments=cast(
                Mapping[str, JsonValue],
                _input_arguments(),
            ),
        )


def test_replay_owner_rejects_reuse_and_clears_released_state() -> None:
    boundaries = OpenAIClient._decode_reserved_call_boundaries(
        _reserved_boundary_manifest(_replay_items())
    )
    owner = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        StreamRetentionPolicy()
    )
    owner.restore_reserved_call_boundaries(boundaries)
    with pytest.raises(RuntimeError, match="boundary history"):
        owner.restore_reserved_call_boundaries(boundaries)
    owner.restore_provider_idempotency_key(
        ProviderIdempotencyKey("provider-key")
    )
    with pytest.raises(RuntimeError, match="idempotency key"):
        owner.restore_provider_idempotency_key(
            ProviderIdempotencyKey("other-key")
        )

    owner.release()

    assert owner.reserved_call_boundaries() == ()
    with pytest.raises(RuntimeError, match="boundary history"):
        owner.restore_reserved_call_boundaries(boundaries)
    with pytest.raises(RuntimeError, match="idempotency key"):
        owner.restore_provider_idempotency_key(
            ProviderIdempotencyKey("released-key")
        )
    with pytest.raises(RuntimeError, match="released"):
        owner.take_provider_idempotency_key()


def test_snapshot_import_rejects_non_restorable_item_and_releases_owner() -> (
    None
):
    client = _client()
    original_admit = openai_module._OpenAIReplayOwner.admit  # noqa: SLF001
    captured_owner: openai_module._OpenAIReplayOwner | None = (  # noqa: SLF001
        None
    )
    admit_calls = 0

    def reject_second_item(
        owner: openai_module._OpenAIReplayOwner,  # noqa: SLF001
        item: dict[str, Any],
    ) -> bool:
        nonlocal admit_calls, captured_owner
        if captured_owner is None:
            captured_owner = owner
        else:
            assert captured_owner is owner
        admit_calls += 1
        if admit_calls == 2:
            return False
        return original_admit(owner, item)

    with (
        patch.object(
            openai_module._OpenAIReplayOwner,  # noqa: SLF001
            "admit",
            new=reject_second_item,
        ),
        pytest.raises(InputSnapshotError, match="not restorable"),
    ):
        client.import_continuation_snapshot(
            _snapshot(),
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
        )

    assert admit_calls == 2
    assert captured_owner is not None
    assert captured_owner.released
    assert captured_owner.release_count == 1
    assert captured_owner.item_count == 0
    assert captured_owner.replay_items() == ()
    assert captured_owner.counters == (0, 0, 0, 0)
    assert captured_owner.generic_counters == (0, 0)
    assert captured_owner._provider_idempotency_key is None  # noqa: SLF001
    with pytest.raises(RuntimeError, match="released"):
        captured_owner.take_provider_idempotency_key()
    assert not client._replay_owners_by_call_id  # noqa: SLF001


def test_snapshot_call_validation_rejects_invalid_expected_contract() -> None:
    client = _client()
    snapshot = _snapshot()
    valid_arguments = cast(
        Mapping[str, JsonValue],
        _input_arguments(),
    )

    with pytest.raises(InputSnapshotError, match="capability name"):
        client.validate_continuation_snapshot_call(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
            expected_provider_name="",
            expected_arguments=valid_arguments,
        )
    with pytest.raises(InputSnapshotError, match="arguments are invalid"):
        client.validate_continuation_snapshot_call(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
            expected_provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            expected_arguments=cast(Mapping[str, JsonValue], []),
        )

    original_copy = openai_module._strict_replay_json_copy  # noqa: SLF001
    marker_arguments = cast(
        Mapping[str, JsonValue],
        {"coverage-marker": True},
    )

    def invalidate_expected_arguments(value: object) -> object:
        if value == marker_arguments:
            return []
        return original_copy(value)

    with (
        patch.object(
            openai_module,
            "_strict_replay_json_copy",
            side_effect=invalidate_expected_arguments,
        ),
        pytest.raises(InputSnapshotError, match="arguments are invalid"),
    ):
        client.validate_continuation_snapshot_call(
            snapshot,
            expected_binding=_binding(),
            provider_call_correlation_id=_CALL_ID,
            expected_provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            expected_arguments=marker_arguments,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda _manifest: None, "manifest is invalid"),
        (
            lambda _manifest: ({"unexpected": True},),
            "manifest fields are invalid",
        ),
        (
            lambda manifest: (
                {
                    **manifest[0],
                    "provider_name": "",
                },
            ),
            "manifest is incomplete",
        ),
        (
            lambda manifest: (
                {
                    **manifest[0],
                    "canonical_arguments": dumps(
                        _input_arguments(),
                        indent=2,
                        sort_keys=True,
                    ),
                },
            ),
            "arguments are not canonical",
        ),
        (
            lambda manifest: (
                {
                    **manifest[0],
                    "reasoning": ({"unexpected": True},),
                },
            ),
            "reasoning boundary is invalid",
        ),
        (
            lambda manifest: (
                {
                    **manifest[0],
                    "reasoning": (
                        {
                            **manifest[0]["reasoning"][0],
                            "item_id": "",
                        },
                    ),
                },
            ),
            "reasoning boundary is incomplete",
        ),
    ),
)
def test_reserved_boundary_decoder_rejects_every_malformed_layer(
    mutate: Any,
    message: str,
) -> None:
    manifest = _reserved_boundary_manifest(_replay_items())

    with pytest.raises(InputSnapshotError, match=message):
        OpenAIClient._decode_reserved_call_boundaries(mutate(manifest))


def test_reserved_call_boundary_rejects_nonfinal_and_changed_calls() -> None:
    items = _replay_items()
    with pytest.raises(InputSnapshotError, match="final replay item"):
        OpenAIClient._current_reserved_call_boundary(
            (
                *items,
                {
                    "id": "rs-trailing",
                    "type": "reasoning",
                    "encrypted_content": "trailing-ciphertext",
                },
            ),
            _CALL_ID,
        )
    with pytest.raises(InputSnapshotError, match="capability changed"):
        OpenAIClient._current_reserved_call_boundary(
            _replay_items(name="other"),
            _CALL_ID,
        )


def test_replay_validation_rejects_boundary_order_and_completeness() -> None:
    current_item = _replay_items()[1]
    canonical_arguments = dumps(
        _input_arguments(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    empty_reasoning_boundary = (
        openai_module._OpenAIReservedCallBoundary(  # noqa: SLF001
            call_id=_CALL_ID,
            provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            canonical_arguments=canonical_arguments,
            reasoning=(),
        )
    )
    with pytest.raises(InputSnapshotError, match="lacks encrypted reasoning"):
        OpenAIClient._validate_continuation_replay_items(
            (current_item,),
            _CALL_ID,
            (empty_reasoning_boundary,),
        )

    sequential_items = _sequential_replay_items()
    boundaries = OpenAIClient._decode_reserved_call_boundaries(
        _reserved_boundary_manifest(sequential_items)
    )
    with pytest.raises(InputSnapshotError, match="manifest was reordered"):
        OpenAIClient._validate_continuation_replay_items(
            _replay_items(),
            _CALL_ID,
            boundaries,
        )
    with pytest.raises(InputSnapshotError, match="manifest is incomplete"):
        OpenAIClient._validate_continuation_replay_items(
            sequential_items[:2],
            _CALL_ID,
            boundaries,
        )


@pytest.mark.parametrize(
    ("changed_items", "message"),
    (
        (
            lambda items: (items[0], dict(items[0])),
            "reasoning replay IDs must be unique",
        ),
        (
            lambda items: (items[0], items[1], dict(items[1])),
            "function call replay IDs must be unique",
        ),
    ),
)
def test_replay_validation_rechecks_mutating_sequences(
    changed_items: Any,
    message: str,
) -> None:
    stable_items = _sequential_replay_items()
    boundaries = OpenAIClient._decode_reserved_call_boundaries(
        _reserved_boundary_manifest(stable_items)
    )

    class ChangingReplayItems(Sequence[Mapping[str, object]]):
        def __init__(self) -> None:
            self.iterations = 0

        def __len__(self) -> int:
            return len(stable_items)

        @overload
        def __getitem__(self, index: int) -> Mapping[str, object]: ...

        @overload
        def __getitem__(
            self, index: slice
        ) -> Sequence[Mapping[str, object]]: ...

        def __getitem__(
            self, index: int | slice
        ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
            return stable_items[index]

        def __iter__(self) -> Iterator[Mapping[str, object]]:
            self.iterations += 1
            selected = (
                stable_items
                if self.iterations < 3
                else changed_items(stable_items)
            )
            return iter(selected)

    with pytest.raises(InputSnapshotError, match=message):
        OpenAIClient._validate_continuation_replay_items(
            ChangingReplayItems(),
            _CALL_ID,
            boundaries,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            {"mode": "required", "reason": "Need input."},
            "arguments are incomplete",
        ),
        (
            {
                "mode": 1,
                "reason": "Need input.",
                "questions": _input_arguments()["questions"],
            },
            "mode is invalid",
        ),
        (
            {
                "mode": "unsupported",
                "reason": "Need input.",
                "questions": _input_arguments()["questions"],
            },
            "mode is invalid",
        ),
        (
            {
                "mode": _ValidRequirementModeString("required"),
                "reason": "Need input.",
                "questions": _input_arguments()["questions"],
            },
            "mode is invalid",
        ),
        (
            {
                "mode": "required",
                "reason": "",
                "questions": _input_arguments()["questions"],
            },
            "arguments are invalid",
        ),
        (
            {
                "mode": "required",
                "reason": "Need input.",
                "questions": [{}],
            },
            "questions are invalid",
        ),
    ),
)
def test_reserved_input_arguments_reject_invalid_contracts(
    arguments: Mapping[str, object],
    message: str,
) -> None:
    with pytest.raises(InputSnapshotError, match=message):
        OpenAIClient._validate_reserved_input_arguments(arguments)


def test_replay_owner_lookup_ignores_released_active_owners() -> None:
    client = _client()
    _retain(client)
    retained = client._replay_owners_by_call_id[_CALL_ID]  # noqa: SLF001
    released = openai_module._OpenAIReplayOwner(  # noqa: SLF001
        client._stream_retention_policy  # noqa: SLF001
    )
    released.release()
    client._active_replay_owners[id(released)] = released  # noqa: SLF001

    assert (
        client._continuation_replay_owner(_CALL_ID) is retained  # noqa: SLF001
    )


def test_openai_helpers_reject_invalid_json_and_reasoning_correlation() -> (
    None
):
    with pytest.raises(InputSnapshotError, match="invalid JSON"):
        OpenAIClient._decode_continuation_call_arguments('{"value":NaN}')
    with pytest.raises(ValueError, match="item id"):
        OpenAIStream._reasoning_correlation({"item_id": ""})
    with pytest.raises(ValueError, match="output_index"):
        OpenAIStream._reasoning_correlation({"output_index": -1})

    assert OpenAIClient._is_native_openai_responses_base_url(None)
    assert not OpenAIClient._is_native_openai_responses_base_url(cast(Any, 1))


def test_openai_client_and_model_capability_fallbacks_are_exact() -> None:
    class CompatibleOpenAIClient(OpenAIClient):
        _reasoning_summary_provider = "compatible"

    class CompatibleOpenAIModel(OpenAIModel):
        pass

    CompatibleOpenAIModel.__module__ = "avalan.model.nlp.text.vendor.anthropic"
    native_client = object.__new__(OpenAIClient)
    compatible_client = object.__new__(CompatibleOpenAIClient)
    native_model = object.__new__(OpenAIModel)
    compatible_model = object.__new__(CompatibleOpenAIModel)

    assert native_client.reasoning_summary_request_capability.supported_modes
    assert (
        compatible_client.reasoning_summary_request_capability.supported_modes
        == frozenset()
    )
    assert compatible_client.reasoning_summary_provider == "compatible"
    assert native_model.reasoning_summary_request_capability.supported_modes
    assert (
        compatible_model.reasoning_summary_request_capability.supported_modes
        == frozenset()
    )
    assert compatible_model.reasoning_summary_provider == "anthropic"


def test_openai_model_provider_resolution_uses_live_client_then_settings() -> (
    None
):
    model = object.__new__(OpenAIModel)
    client = SimpleNamespace(_is_azure=True)
    model._model = client  # noqa: SLF001
    assert model.reasoning_summary_provider == "azure_openai"
    client._is_azure = False
    assert model.reasoning_summary_provider == "openai"

    del client._is_azure
    cast(Any, model)._settings = SimpleNamespace(
        base_url="https://resource.openai.azure.com"
    )
    assert model.reasoning_summary_provider == "azure_openai"
    cast(Any, model)._settings = SimpleNamespace(
        base_url="https://api.openai.com/v1"
    )
    assert model.reasoning_summary_provider == "openai"


def test_openai_capability_result_message_is_provider_native() -> None:
    result = CorrelatedCapabilityResult(
        call_id="continuation-call",
        canonical_name=RESERVED_INPUT_CAPABILITY_NAME,
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
        payload={"accepted": True},
    )

    assert OpenAIClient.capability_result_message(result) == {
        "type": "function_call_output",
        "call_id": "continuation-call",
        "output": '{"accepted": true}',
    }
