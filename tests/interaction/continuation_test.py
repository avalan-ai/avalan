"""Test portable continuation validation, codecs, and runtime resolution."""

from asyncio import run as asyncio_run
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps, loads
from types import MappingProxyType
from typing import Any, Callable, cast

import pytest

from avalan.interaction import (
    AgentId,
    BranchId,
    CapabilityRevision,
    ContinuationClaim,
    ContinuationClaimOwnerId,
    ContinuationClaimState,
    ContinuationCompletion,
    ContinuationDispatch,
    ContinuationDispatchId,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationRuntimeResolver,
    ContinuationSnapshot,
    ContinuationStoreRevision,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequestId,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PortableContinuation,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    ResolvedContinuationRuntime,
    RunId,
    StateRevision,
    StreamSessionId,
    TrustedContinuationRuntimeLoader,
    TurnId,
    decode_portable_continuation,
    encode_portable_continuation,
    portable_continuation_digest,
)
from avalan.interaction import continuation as continuation_module
from avalan.interaction.continuation import (
    ContinuationClaimReceipt,
    DurableContinuationRecord,
)
from avalan.interaction.error import (
    InputErrorCode,
    InputSnapshotError,
    InputValidationError,
)

_NOW = datetime(2026, 7, 23, 12, tzinfo=UTC)


def _definition(**overrides: object) -> ExecutionDefinitionRef:
    values: dict[str, Any] = {
        "agent_definition_locator": "agent:test",
        "agent_definition_revision": "agent-revision",
        "operation_id": "operation:test",
        "operation_index": 2,
        "model_config_reference": "model-config:test",
        "tool_revision": "tool-revision",
        "capability_revision": "capability-revision",
    }
    values.update(overrides)
    return ExecutionDefinitionRef(**values)


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


def _origin(
    definition: ExecutionDefinitionRef,
    *,
    model_call_id: str = "model-call",
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId(model_call_id),
        stream_session_id=StreamSessionId("stream"),
        definition=definition,
        principal=PrincipalScope(),
    )


def _provider_snapshot(
    binding: ContinuationRevisionBinding,
    *,
    model_call_id: str = "model-call",
    provider_idempotency_key: str = "provider-retry",
) -> ContinuationSnapshot:
    return ContinuationSnapshot(
        snapshot_kind="openai.responses.reasoning",
        revision_binding=binding,
        model_call_id=ModelCallId(model_call_id),
        provider_idempotency_key=ProviderIdempotencyKey(
            provider_idempotency_key
        ),
        payload={
            "reserved_capability_call_id": "call-input",
            "replay_items": (
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": "ciphertext",
                },
                {
                    "type": "function_call",
                    "call_id": "call-input",
                    "name": "request_user_input",
                    "arguments": "{}",
                },
            ),
        },
    )


def _portable(
    *,
    claim: ContinuationClaim | None = None,
    dispatch: ContinuationDispatch | None = None,
    completion: ContinuationCompletion | None = None,
    provider_snapshot: ContinuationSnapshot | None = None,
    definition: ExecutionDefinitionRef | None = None,
    origin: ExecutionOrigin | None = None,
    binding: ContinuationRevisionBinding | None = None,
    **overrides: object,
) -> PortableContinuation:
    definition = definition or _definition()
    binding = binding or _binding()
    origin = origin or _origin(definition)
    values: dict[str, Any] = {
        "continuation_id": ContinuationId("continuation"),
        "request_id": InputRequestId("request"),
        "origin": origin,
        "provider_call_id": ModelCallId("model-call"),
        "provider_call_correlation_id": "call-input",
        "definition": definition,
        "operation_cursor": 3,
        "generation_settings": {
            "max_tokens": 256,
            "reasoning": {"effort": "high"},
        },
        "transcript": (
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "tool_call_id": "call-input",
                "content": None,
            },
        ),
        "observations": (
            {"kind": "tool_result", "sequence": 4, "completed": True},
        ),
        "provider_snapshot": (
            provider_snapshot
            if provider_snapshot is not None
            else _provider_snapshot(binding)
        ),
        "revision_binding": binding,
        "interaction_count": 1,
        "tool_loop_count": 2,
        "stream_sequence": 5,
        "state_revision": StateRevision(6),
        "store_revision": ContinuationStoreRevision(7),
        "created_at": _NOW,
        "updated_at": _NOW + timedelta(seconds=2),
        "expires_at": _NOW + timedelta(hours=1),
        "claim": claim or ContinuationClaim(),
        "fencing_token": ContinuationFencingToken(0),
        "dispatch": dispatch,
        "completion": completion,
    }
    values.update(overrides)
    return PortableContinuation(**values)


def _dispatch(
    *,
    key: str = "provider-retry",
) -> ContinuationDispatch:
    return ContinuationDispatch(
        dispatch_id=ContinuationDispatchId("dispatch"),
        provider_idempotency_key=ProviderIdempotencyKey(key),
        marked_at=_NOW + timedelta(seconds=1),
    )


def test_durable_provider_key_is_exact_and_pair_bound() -> None:
    continuation_id = ContinuationId("continuation-1")
    dispatch_id = ContinuationDispatchId("dispatch-1")
    expected = (
        "task-input-"
        + sha256(
            b'{"continuation_id":"continuation-1","dispatch_id":"dispatch-1"}'
        ).hexdigest()
    )

    first = continuation_module.derive_provider_idempotency_key(
        continuation_id,
        dispatch_id,
    )
    repeated = continuation_module.derive_provider_idempotency_key(
        continuation_id,
        dispatch_id,
    )

    assert str(first) == expected
    assert repeated == first
    assert (
        continuation_module.derive_provider_idempotency_key(
            ContinuationId("continuation-2"),
            dispatch_id,
        )
        != first
    )
    assert (
        continuation_module.derive_provider_idempotency_key(
            continuation_id,
            ContinuationDispatchId("dispatch-2"),
        )
        != first
    )


def test_continuation_dispatch_id_is_exact_and_stable() -> None:
    continuation_id = ContinuationId("continuation-1")
    expected = (
        "task-resume-"
        + sha256(b'{"continuation_id":"continuation-1"}').hexdigest()
    )

    first = continuation_module.derive_continuation_dispatch_id(
        continuation_id
    )

    assert str(first) == expected
    assert (
        continuation_module.derive_continuation_dispatch_id(continuation_id)
        == first
    )
    assert (
        continuation_module.derive_continuation_dispatch_id(
            ContinuationId("continuation-2")
        )
        != first
    )


def test_portable_continuation_round_trip_is_canonical_and_immutable() -> None:
    continuation = _portable()

    encoded = encode_portable_continuation(continuation)
    restored = decode_portable_continuation(
        encoded.encode(),
        expected_binding=continuation.revision_binding,
    )

    assert restored == continuation
    assert encoded == dumps(
        loads(encoded),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert loads(encoded)["content_sha256"] == portable_continuation_digest(
        continuation
    )
    assert isinstance(restored.generation_settings, MappingProxyType)
    assert isinstance(restored.transcript[0], MappingProxyType)
    assert restored.provider_snapshot is not None
    assert (
        restored.provider_snapshot.payload["replay_items"][0][
            "encrypted_content"
        ]
        == "ciphertext"
    )
    with pytest.raises(TypeError):
        cast(dict[str, object], restored.generation_settings)["new"] = True


def test_terminal_revision_may_be_recorded_after_expiry() -> None:
    """Allow durable sweeps to persist a post-expiry terminal revision."""
    expired_at = _NOW + timedelta(minutes=1)

    continuation = _portable(
        expires_at=expired_at,
        updated_at=expired_at + timedelta(seconds=1),
        store_revision=ContinuationStoreRevision(8),
    )

    assert continuation.updated_at > continuation.expires_at
    assert (
        decode_portable_continuation(
            encode_portable_continuation(continuation),
            expected_binding=continuation.revision_binding,
        )
        == continuation
    )


@pytest.mark.parametrize(
    ("state", "owner", "lease", "dispatch", "completion"),
    (
        (
            ContinuationClaimState.UNCLAIMED,
            None,
            None,
            None,
            None,
        ),
        (
            ContinuationClaimState.CLAIMED_PRE_DISPATCH,
            ContinuationClaimOwnerId("worker"),
            _NOW + timedelta(minutes=1),
            _dispatch(),
            None,
        ),
        (
            ContinuationClaimState.DISPATCHED_AMBIGUOUS,
            ContinuationClaimOwnerId("worker"),
            None,
            _dispatch(),
            None,
        ),
        (
            ContinuationClaimState.FAILED_SAFE_TO_RETRY,
            None,
            None,
            _dispatch(),
            None,
        ),
        (
            ContinuationClaimState.COMPLETED,
            None,
            None,
            _dispatch(),
            ContinuationCompletion(
                completed_at=_NOW + timedelta(seconds=2),
                result_digest="a" * 64,
            ),
        ),
    ),
)
def test_portable_continuation_accepts_every_claim_state(
    state: ContinuationClaimState,
    owner: ContinuationClaimOwnerId | None,
    lease: datetime | None,
    dispatch: ContinuationDispatch | None,
    completion: ContinuationCompletion | None,
) -> None:
    claim = ContinuationClaim(
        state=state,
        owner_id=owner,
        lease_expires_at=lease,
        attempt=1,
    )

    continuation = _portable(
        claim=claim,
        dispatch=dispatch,
        completion=completion,
        fencing_token=ContinuationFencingToken(2),
    )

    assert continuation.claim.state is state
    assert (
        decode_portable_continuation(
            encode_portable_continuation(continuation),
            expected_binding=continuation.revision_binding,
        )
        == continuation
    )


@pytest.mark.parametrize(
    ("kwargs", "path"),
    (
        (
            {
                "state": ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                "owner_id": None,
                "lease_expires_at": _NOW,
            },
            "continuation.claim.owner_id",
        ),
        (
            {
                "state": ContinuationClaimState.UNCLAIMED,
                "owner_id": ContinuationClaimOwnerId("worker"),
            },
            "continuation.claim.owner_id",
        ),
        (
            {
                "state": ContinuationClaimState.DISPATCHED_AMBIGUOUS,
                "owner_id": ContinuationClaimOwnerId("worker"),
                "lease_expires_at": _NOW,
            },
            "continuation.claim.lease_expires_at",
        ),
    ),
)
def test_claim_rejects_owner_or_lease_state_mismatch(
    kwargs: dict[str, Any],
    path: str,
) -> None:
    with pytest.raises(InputValidationError) as raised:
        ContinuationClaim(**kwargs)

    assert raised.value.path == path


@pytest.mark.parametrize(
    ("overrides", "path"),
    (
        ({"version": 2}, "continuation.version"),
        ({"origin": object()}, "continuation.origin"),
        ({"provider_call_correlation_id": ""}, "continuation."),
        ({"definition": object()}, "definition"),
        ({"operation_cursor": True}, "continuation.operation_cursor"),
        ({"generation_settings": ()}, "continuation.generation_settings"),
        ({"transcript": []}, "continuation.transcript"),
        ({"observations": []}, "continuation.observations"),
        ({"revision_binding": object()}, "continuation.revision_binding"),
        ({"interaction_count": -1}, "continuation.interaction_count"),
        ({"state_revision": -1}, "continuation.state_revision"),
        ({"claim": object()}, "continuation.claim"),
        ({"fencing_token": -1}, "continuation.fencing_token"),
        ({"dispatch": object()}, "continuation.dispatch"),
        ({"completion": object()}, "continuation.completion"),
    ),
)
def test_portable_continuation_rejects_invalid_field_types(
    overrides: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(InputValidationError) as raised:
        _portable(**overrides)

    assert raised.value.path.startswith(path)


def test_portable_continuation_rejects_correlation_and_timestamp_drift() -> (
    None
):
    definition = _definition()
    other = _definition(operation_id="operation:other")
    with pytest.raises(InputValidationError, match="definition"):
        _portable(definition=other, origin=_origin(definition))
    with pytest.raises(InputValidationError, match="provider call"):
        _portable(origin=_origin(definition, model_call_id="other"))
    with pytest.raises(InputValidationError, match="timestamps"):
        _portable(updated_at=_NOW - timedelta(seconds=1))
    with pytest.raises(InputValidationError) as raised:
        _portable(
            definition=cast(ExecutionDefinitionRef, object()),
            origin=_origin(definition),
        )
    assert raised.value.path == "continuation.definition"


def test_portable_continuation_rejects_provider_snapshot_drift() -> None:
    binding = _binding()
    drifted = _binding(model_config_revision=ModelConfigRevision("model-r2"))
    with pytest.raises(InputValidationError) as raised:
        _portable(
            binding=binding,
            provider_snapshot=_provider_snapshot(drifted),
        )
    assert raised.value.code is InputErrorCode.SNAPSHOT_REVISION_DRIFT

    with pytest.raises(InputValidationError) as raised:
        _portable(
            binding=binding,
            provider_snapshot=_provider_snapshot(
                binding,
                model_call_id="other",
            ),
        )
    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH

    with pytest.raises(InputValidationError) as raised:
        _portable(
            binding=binding,
            provider_snapshot=cast(ContinuationSnapshot, object()),
        )
    assert raised.value.path == "continuation.provider_snapshot"

    with pytest.raises(InputValidationError) as raised:
        _portable(
            binding=binding,
            claim=ContinuationClaim(
                state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                owner_id=ContinuationClaimOwnerId("worker"),
                lease_expires_at=_NOW + timedelta(minutes=1),
            ),
            dispatch=_dispatch(key="different"),
        )
    assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH


def test_portable_continuation_rejects_live_values_and_secret_keys() -> None:
    with pytest.raises(InputValidationError) as raised:
        _portable(generation_settings={"client": object()})
    assert raised.value.code is InputErrorCode.SNAPSHOT_INVALID

    with pytest.raises(InputValidationError) as raised:
        _portable(observations=({"api_key": "plaintext"},))
    assert raised.value.code is InputErrorCode.SNAPSHOT_SECRET_PROHIBITED


def test_portable_continuation_rejects_claim_metadata_drift() -> None:
    with pytest.raises(InputValidationError, match="dispatch metadata"):
        _portable(
            claim=ContinuationClaim(
                state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                owner_id=ContinuationClaimOwnerId("worker"),
                lease_expires_at=_NOW + timedelta(minutes=1),
            )
        )
    with pytest.raises(InputValidationError, match="completion metadata"):
        _portable(
            completion=ContinuationCompletion(
                completed_at=_NOW + timedelta(seconds=2),
                result_digest="a" * 64,
            )
        )
    with pytest.raises(InputValidationError, match="predates"):
        _portable(
            claim=ContinuationClaim(
                state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
                owner_id=ContinuationClaimOwnerId("worker"),
                lease_expires_at=_NOW + timedelta(minutes=1),
            ),
            dispatch=replace(
                _dispatch(),
                marked_at=_NOW - timedelta(seconds=1),
            ),
        )
    with pytest.raises(InputValidationError, match="outside"):
        _portable(
            claim=ContinuationClaim(
                state=ContinuationClaimState.COMPLETED,
            ),
            dispatch=_dispatch(),
            completion=ContinuationCompletion(
                completed_at=_NOW + timedelta(minutes=2),
                result_digest="a" * 64,
            ),
        )


def _recoded(
    continuation: PortableContinuation,
    mutate: Callable[[dict[str, object]], None],
) -> str:
    payload = loads(encode_portable_continuation(continuation))
    assert isinstance(payload, dict)
    mutate(cast(dict[str, object], payload))
    content = dict(payload)
    content.pop("content_sha256", None)
    payload["content_sha256"] = (
        __import__("hashlib")
        .sha256(
            dumps(
                content,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        .hexdigest()
    )
    return dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_decode_rejects_tampering_schema_and_noncanonical_json() -> None:
    continuation = _portable()
    encoded = encode_portable_continuation(continuation)
    payload = loads(encoded)
    payload["stream_sequence"] = 100
    tampered = dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(InputSnapshotError) as raised:
        decode_portable_continuation(
            tampered,
            expected_binding=continuation.revision_binding,
        )
    assert raised.value.path == "continuation.content_sha256"

    malformed_digest = dict(loads(encoded))
    malformed_digest["content_sha256"] = "not-a-digest"
    with pytest.raises(InputSnapshotError) as raised:
        decode_portable_continuation(
            dumps(
                malformed_digest,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            expected_binding=continuation.revision_binding,
        )
    assert raised.value.path == "continuation.content_sha256"

    missing = dict(loads(encoded))
    del missing["claim"]
    with pytest.raises(InputSnapshotError, match="fields"):
        decode_portable_continuation(
            dumps(missing, separators=(",", ":"), sort_keys=True),
            expected_binding=continuation.revision_binding,
        )

    with pytest.raises(InputSnapshotError, match="canonical"):
        decode_portable_continuation(
            dumps(loads(encoded), indent=2, sort_keys=True),
            expected_binding=continuation.revision_binding,
        )

    duplicate = encoded.replace(
        '{"claim":',
        '{"claim":null,"claim":',
        1,
    )
    with pytest.raises(InputSnapshotError, match="invalid JSON"):
        decode_portable_continuation(
            duplicate,
            expected_binding=continuation.revision_binding,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (object(), "text or UTF-8 bytes"),
        (b"\xff", "must be UTF-8"),
        ("[]", "must be a JSON object"),
        ("{", "invalid JSON"),
    ),
)
def test_decode_rejects_invalid_wire_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(InputSnapshotError, match=message):
        decode_portable_continuation(
            cast(str | bytes, value),
            expected_binding=_binding(),
        )


def test_decode_rejects_unknown_version_provider_model_and_revision() -> None:
    continuation = _portable()

    with pytest.raises(InputSnapshotError) as raised:
        decode_portable_continuation(
            _recoded(
                continuation,
                lambda item: item.__setitem__("version", 2),
            ),
            expected_binding=continuation.revision_binding,
        )
    assert raised.value.code is InputErrorCode.SNAPSHOT_UNSUPPORTED

    for expected, code in (
        (
            _binding(provider_family=ProviderFamilyName("anthropic")),
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
        ),
        (
            _binding(model_id=ModelId("gpt-other")),
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
        ),
        (
            _binding(
                provider_config_revision=ProviderConfigRevision("provider-r2")
            ),
            InputErrorCode.SNAPSHOT_REVISION_DRIFT,
        ),
    ):
        with pytest.raises(InputSnapshotError) as raised:
            decode_portable_continuation(
                encode_portable_continuation(continuation),
                expected_binding=expected,
            )
        assert raised.value.code is code


def test_decode_rejects_invalid_nested_shapes() -> None:
    continuation = _portable()
    mutations = (
        lambda item: item.__setitem__("definition", []),
        lambda item: item.__setitem__("revision_binding", []),
        lambda item: item.__setitem__("generation_settings", []),
        lambda item: item.__setitem__("transcript", {}),
        lambda item: item.__setitem__("claim", []),
        lambda item: item.__setitem__("dispatch", []),
        lambda item: item.__setitem__("completion", []),
        lambda item: item.__setitem__("created_at", "bad"),
        lambda item: item.__setitem__("continuation_id", 1),
        lambda item: item.__setitem__("operation_cursor", "bad"),
        lambda item: item.__setitem__(
            "created_at",
            "2026-07-23T12:00:00",
        ),
        lambda item: item.__setitem__(
            "definition",
            {"unexpected": True},
        ),
        lambda item: item.__setitem__(
            "claim",
            {
                "state": "unknown",
                "owner_id": None,
                "lease_expires_at": None,
                "attempt": 0,
            },
        ),
        lambda item: item.__setitem__(
            "revision_binding",
            {
                "provider_family": "",
                "model_id": "gpt-5",
                "provider_config_revision": "provider-r1",
                "model_config_revision": "model-r1",
                "capability_revision": "capability-r1",
            },
        ),
        lambda item: item.__setitem__(
            "claim",
            {
                "state": "claimed_pre_dispatch",
                "owner_id": None,
                "lease_expires_at": _NOW.isoformat(),
                "attempt": 0,
            },
        ),
    )
    for mutation in mutations:
        with pytest.raises(InputSnapshotError):
            decode_portable_continuation(
                _recoded(continuation, mutation),
                expected_binding=continuation.revision_binding,
            )


def test_codec_rejects_wrong_expected_binding_and_non_json_helpers() -> None:
    continuation = _portable(
        generation_settings={"stop_strings": ("one", "two")}
    )
    encoded = encode_portable_continuation(continuation)
    assert loads(encoded)["generation_settings"]["stop_strings"] == [
        "one",
        "two",
    ]
    with pytest.raises(InputSnapshotError, match="typed revision binding"):
        decode_portable_continuation(
            encoded,
            expected_binding=cast(ContinuationRevisionBinding, object()),
        )
    with pytest.raises(InputSnapshotError, match="non-JSON value"):
        continuation_module._canonical_json(  # noqa: SLF001
            cast(dict[str, object], {"value": object()})
        )


def test_codec_enforces_wire_and_encoded_byte_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuation = _portable()
    monkeypatch.setattr(
        continuation_module,
        "PORTABLE_CONTINUATION_MAX_UTF8_BYTES",
        1,
    )
    with pytest.raises(InputSnapshotError, match="byte bound"):
        encode_portable_continuation(continuation)
    with pytest.raises(InputSnapshotError, match="byte bound"):
        decode_portable_continuation(
            "{}",
            expected_binding=continuation.revision_binding,
        )


class _TrustedLoader:
    trusted_continuation_runtime_loader = True

    def __init__(
        self,
        result: ResolvedContinuationRuntime | object,
        *,
        synchronous: bool = False,
    ) -> None:
        self.result = result
        self.synchronous = synchronous
        self.calls: list[
            tuple[ExecutionDefinitionRef, ContinuationRevisionBinding]
        ] = []

    def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> object:
        self.calls.append((definition, revision_binding))
        if self.synchronous:
            return self.result

        async def load() -> object:
            return self.result

        return load()


def _resolved(
    continuation: PortableContinuation,
    **overrides: object,
) -> ResolvedContinuationRuntime:
    values: dict[str, Any] = {
        "definition": continuation.definition,
        "revision_binding": continuation.revision_binding,
        "runtime": object(),
        "operation": object(),
        "model": object(),
        "tools": object(),
        "capabilities": object(),
        "credentials_reloaded_from_trusted_config": True,
    }
    values.update(overrides)
    return ResolvedContinuationRuntime(**values)


def test_runtime_resolver_loads_fresh_trusted_components() -> None:
    continuation = _portable()
    resolved = _resolved(continuation)
    loader = _TrustedLoader(resolved)
    resolver = ContinuationRuntimeResolver(
        cast(TrustedContinuationRuntimeLoader, loader),
        clock=lambda: _NOW,
    )

    result = asyncio_run(resolver.resolve(continuation))

    assert result is resolved
    assert loader.calls == [
        (continuation.definition, continuation.revision_binding)
    ]
    assert "runtime" not in encode_portable_continuation(continuation)
    assert (
        "credential"
        not in encode_portable_continuation(continuation).casefold()
    )


def test_runtime_resolver_rejects_untrusted_loader_and_invalid_clock() -> None:
    class Untrusted:
        trusted_continuation_runtime_loader = False

        async def load_continuation_runtime(self) -> object:
            return object()

    with pytest.raises(InputValidationError, match="trusted async"):
        ContinuationRuntimeResolver(
            cast(TrustedContinuationRuntimeLoader, Untrusted())
        )
    with pytest.raises(InputValidationError, match="clock callable"):
        ContinuationRuntimeResolver(
            cast(
                TrustedContinuationRuntimeLoader,
                _TrustedLoader(object()),
            ),
            clock=cast(Callable[[], datetime], 1),
        )


def test_runtime_resolver_rejects_invalid_results() -> None:
    continuation = _portable()
    resolved = _resolved(continuation)
    with pytest.raises(InputValidationError) as raised:
        asyncio_run(
            ContinuationRuntimeResolver(
                cast(
                    TrustedContinuationRuntimeLoader,
                    _TrustedLoader(resolved),
                ),
                clock=lambda: continuation.expires_at,
            ).resolve(continuation)
        )
    assert raised.value.code is InputErrorCode.EXPIRED

    with pytest.raises(InputValidationError, match="asynchronous"):
        asyncio_run(
            ContinuationRuntimeResolver(
                cast(
                    TrustedContinuationRuntimeLoader,
                    _TrustedLoader(resolved, synchronous=True),
                ),
                clock=lambda: _NOW,
            ).resolve(continuation)
        )

    with pytest.raises(InputValidationError, match="resolved continuation"):
        asyncio_run(
            ContinuationRuntimeResolver(
                cast(
                    TrustedContinuationRuntimeLoader,
                    _TrustedLoader(object()),
                ),
                clock=lambda: _NOW,
            ).resolve(continuation)
        )

    with pytest.raises(InputValidationError, match="portable continuation"):
        asyncio_run(
            ContinuationRuntimeResolver(
                cast(
                    TrustedContinuationRuntimeLoader,
                    _TrustedLoader(resolved),
                ),
                clock=lambda: _NOW,
            ).resolve(cast(PortableContinuation, object()))
        )


def test_runtime_resolver_rejects_definition_and_binding_drift() -> None:
    continuation = _portable()
    with pytest.raises(InputValidationError) as raised:
        asyncio_run(
            ContinuationRuntimeResolver(
                cast(
                    TrustedContinuationRuntimeLoader,
                    _TrustedLoader(
                        _resolved(
                            continuation,
                            definition=_definition(operation_id="other"),
                        )
                    ),
                ),
                clock=lambda: _NOW,
            ).resolve(continuation)
        )
    assert raised.value.code is InputErrorCode.SNAPSHOT_REVISION_DRIFT

    for binding, code in (
        (
            _binding(model_id=ModelId("other")),
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
        ),
        (
            _binding(model_config_revision=ModelConfigRevision("model-r2")),
            InputErrorCode.SNAPSHOT_REVISION_DRIFT,
        ),
    ):
        with pytest.raises(InputSnapshotError) as raised:
            asyncio_run(
                ContinuationRuntimeResolver(
                    cast(
                        TrustedContinuationRuntimeLoader,
                        _TrustedLoader(
                            _resolved(
                                continuation,
                                revision_binding=binding,
                            )
                        ),
                    ),
                    clock=lambda: _NOW,
                ).resolve(continuation)
            )
        assert raised.value.code is code


def test_resolved_runtime_rejects_missing_components_or_credentials() -> None:
    continuation = _portable()
    for name in ("runtime", "operation", "model", "tools", "capabilities"):
        with pytest.raises(InputValidationError, match="fresh live"):
            _resolved(continuation, **{name: None})
    with pytest.raises(InputValidationError, match="trusted configuration"):
        _resolved(
            continuation,
            credentials_reloaded_from_trusted_config=False,
        )
    with pytest.raises(InputValidationError):
        _resolved(continuation, definition=object())
    with pytest.raises(InputValidationError):
        _resolved(continuation, revision_binding=object())


def test_codec_and_metadata_helpers_reject_invalid_values() -> None:
    continuation = _portable()
    with pytest.raises(InputSnapshotError):
        encode_portable_continuation(cast(PortableContinuation, object()))
    with pytest.raises(InputSnapshotError):
        portable_continuation_digest(cast(PortableContinuation, object()))
    with pytest.raises(InputValidationError):
        ContinuationCompletion(
            completed_at=_NOW,
            result_digest="not-a-digest",
        )
    with pytest.raises(InputValidationError):
        ContinuationDispatch(
            dispatch_id=ContinuationDispatchId(""),
            provider_idempotency_key=ProviderIdempotencyKey("key"),
            marked_at=_NOW,
        )
    with pytest.raises(InputValidationError):
        ContinuationClaim(state=cast(ContinuationClaimState, "unknown"))
    assert portable_continuation_digest(continuation)


def test_claim_receipt_rejects_invalid_or_uncorrelated_claims() -> None:
    owner_id = ContinuationClaimOwnerId("worker")
    claimed = _portable(
        claim=ContinuationClaim(
            state=ContinuationClaimState.CLAIMED_PRE_DISPATCH,
            owner_id=owner_id,
            lease_expires_at=_NOW + timedelta(minutes=10),
            attempt=1,
        ),
        dispatch=_dispatch(),
        fencing_token=ContinuationFencingToken(1),
    )
    receipt = ContinuationClaimReceipt(
        continuation=claimed,
        fencing_token=ContinuationFencingToken(1),
    )
    assert receipt.continuation is claimed

    with pytest.raises(
        InputValidationError,
        match="continuation_claim.continuation",
    ):
        ContinuationClaimReceipt(
            continuation=cast(Any, object()),
            fencing_token=ContinuationFencingToken(1),
        )
    with pytest.raises(
        InputValidationError,
        match="continuation_claim.fencing_token",
    ):
        ContinuationClaimReceipt(
            continuation=claimed,
            fencing_token=ContinuationFencingToken(2),
        )
    with pytest.raises(
        InputValidationError,
        match="claim receipt must contain a pre-dispatch claim",
    ):
        ContinuationClaimReceipt(
            continuation=_portable(),
            fencing_token=ContinuationFencingToken(0),
        )


def test_durable_continuation_record_rejects_nonportable_value() -> None:
    with pytest.raises(
        InputValidationError,
        match="continuation_record.continuation",
    ):
        DurableContinuationRecord(
            continuation=cast(Any, object()),
        )
