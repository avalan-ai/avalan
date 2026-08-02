"""Verify canonical encrypted conversation auxiliary payload codecs."""

from collections.abc import Callable
from dataclasses import replace
from json import dumps, loads
from typing import cast
from unittest.mock import patch

import pytest
from phase2_fixtures import (
    authority,
    binding,
    empty_stateless_plan,
    root_identity,
)

import avalan.conversation as conversation
from avalan.interaction import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    StateRevision,
)


def _output() -> conversation.ProviderLaneOutputCandidate:
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    identity = root_identity("durable-codec")
    receipt = conversation.provider_lane_execution_receipt(
        authority=scope,
        identity=identity,
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    return conversation.ProviderLaneOutputCandidate(
        lane_id=lane_binding.lane_id,
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        execution_receipt=receipt,
    )


def _continuation_reference() -> conversation.PortableContinuationReference:
    definition = ExecutionDefinitionRef(
        agent_definition_locator="agent.toml",
        agent_definition_revision="agent-v1",
        operation_id="operation-1",
        operation_index=0,
        model_config_reference="model-config",
        tool_revision="tools-v1",
        capability_revision=CapabilityRevision("capability-v1"),
    )
    revision_binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("synthetic"),
        model_id=ModelId("model-1"),
        provider_config_revision=ProviderConfigRevision("provider-v1"),
        model_config_revision=ModelConfigRevision("model-v1"),
        capability_revision=CapabilityRevision("capability-v1"),
    )
    return conversation.PortableContinuationReference(
        continuation_id=ContinuationId("continuation-1"),
        state_revision=StateRevision(2),
        digest=conversation.ContinuationDigest("d" * 64),
        definition=definition,
        revision_binding=revision_binding,
    )


def _reservation() -> conversation.ConversationExecutionReservation:
    scope = authority()
    lane_binding = binding()
    return conversation.ConversationExecutionReservation(
        idempotency=conversation.RequestIdempotencyIdentity(
            authority=scope,
            operation=conversation.ConversationOperation.CREATE,
            key=conversation.RequestIdempotencyKey("codec-key"),
            request_digest=conversation.CanonicalRequestDigest(
                "request-digest"
            ),
        ),
        identity=root_identity("durable-codec"),
        lanes=(
            conversation.ProviderLaneExecutionReservation(
                binding=lane_binding,
                mode=conversation.ConversationMode.STATELESS,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            ),
        ),
    )


def _recoded(
    encoded: bytes,
    mutate: Callable[[dict[str, object]], None],
) -> bytes:
    payload = loads(encoded)
    assert isinstance(payload, dict)
    mutate(cast(dict[str, object], payload))
    return dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def test_durable_output_round_trip_is_canonical_and_exact() -> None:
    codec = conversation.DurableConversationCodec()
    output = _output()

    encoded = codec.encode_output(output)
    restored = codec.decode_output(encoded)

    assert restored == output
    assert (
        encoded
        == dumps(
            loads(encoded),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )

    payload = loads(encoded)
    payload["binding"]["lane_id"] = "other-lane"
    recoded = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    with pytest.raises(conversation.ConversationCodecError):
        codec.decode_output(recoded)


def test_continuation_reference_round_trip_and_digests_are_stable() -> None:
    codec = conversation.DurableConversationCodec()
    reference = _continuation_reference()

    encoded = codec.encode_continuation_reference(reference)
    restored = codec.decode_continuation_reference(encoded)

    assert restored == reference
    assert conversation.continuation_definition_digest(
        reference.definition
    ) == conversation.continuation_definition_digest(reference.definition)
    assert conversation.continuation_revision_binding_digest(
        reference.revision_binding
    ) == conversation.continuation_revision_binding_digest(
        reference.revision_binding
    )

    changed = replace(
        reference,
        definition=replace(
            reference.definition,
            operation_id="operation-2",
        ),
    )
    assert conversation.continuation_definition_digest(
        changed.definition
    ) != conversation.continuation_definition_digest(reference.definition)


def test_execution_reservation_digest_binds_identity_lane_and_mode() -> None:
    reservation = _reservation()
    digest = conversation.execution_reservation_digest(reservation)

    assert digest is not None and len(digest) == 64
    assert conversation.execution_reservation_digest(None) is None
    assert (
        conversation.execution_reservation_digest(
            replace(
                reservation,
                identity=replace(
                    reservation.identity,
                    logical_turn_id=conversation.LogicalTurnId("other-turn"),
                ),
            )
        )
        != digest
    )
    assert (
        conversation.execution_reservation_digest(
            replace(
                reservation,
                lanes=(
                    replace(
                        reservation.lanes[0],
                        scope=conversation.ProviderLaneOutputScope.CUMULATIVE,
                    ),
                ),
            )
        )
        != digest
    )


@pytest.mark.parametrize(
    "value",
    (
        b"",
        b"not-json",
        b"[]",
        b'{"kind":"duplicate","kind":"duplicate"}',
        dumps({"kind": "wrong"}).encode(),
    ),
)
def test_durable_codec_rejects_malformed_or_noncanonical_values(
    value: bytes,
) -> None:
    codec = conversation.DurableConversationCodec()

    with pytest.raises(conversation.ConversationError):
        codec.decode_output(value)
    with pytest.raises(conversation.ConversationError):
        codec.decode_continuation_reference(value)


def test_durable_codec_limits_and_invalid_helpers_fail_closed() -> None:
    output = _output()
    tiny = conversation.DurableConversationCodec(
        limits=conversation.DurableConversationCodecLimits(max_bytes=8)
    )
    with pytest.raises(conversation.ConversationLimitError):
        tiny.encode_output(output)
    with pytest.raises(conversation.ConversationLimitError):
        tiny.decode_output(b"123456789")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DurableConversationCodecLimits(max_bytes=0)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DurableConversationCodec(
            limits=cast(conversation.DurableConversationCodecLimits, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.execution_reservation_digest(
            cast(conversation.ConversationExecutionReservation, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.continuation_definition_digest(
            cast(ExecutionDefinitionRef, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.continuation_revision_binding_digest(
            cast(ContinuationRevisionBinding, object())
        )


def test_durable_codec_rejects_invalid_public_values_and_wire_fields() -> None:
    codec = conversation.DurableConversationCodec()
    output = codec.encode_output(_output())
    continuation = codec.encode_continuation_reference(
        _continuation_reference()
    )

    with pytest.raises(conversation.ConversationValidationError):
        codec.encode_output(
            cast(conversation.ProviderLaneOutputCandidate, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        codec.encode_continuation_reference(
            cast(conversation.PortableContinuationReference, object())
        )
    for encoded, decoder in (
        (output, codec.decode_output),
        (continuation, codec.decode_continuation_reference),
    ):
        with pytest.raises(conversation.ConversationCodecError):
            decoder(
                _recoded(
                    encoded,
                    lambda item: item.__setitem__("kind", "wrong"),
                )
            )

    for mutate in (
        lambda item: item.__setitem__("items", {}),
        lambda item: item.__setitem__("mode", ""),
        lambda item: item.__setitem__("version", -1),
    ):
        with pytest.raises(conversation.ConversationCodecError):
            codec.decode_output(_recoded(output, mutate))


def test_durable_codec_wraps_constructor_and_serializer_failures() -> None:
    codec = conversation.DurableConversationCodec()
    encoded = codec.encode_continuation_reference(_continuation_reference())

    def drift_capability(item: dict[str, object]) -> None:
        definition = cast(dict[str, object], item["definition"])
        definition["capability_revision"] = "other-capability"

    with pytest.raises(conversation.ConversationCodecError):
        codec.decode_continuation_reference(
            _recoded(encoded, drift_capability)
        )
    with pytest.raises(conversation.ConversationValidationError):
        codec._encode({"invalid": object()})
    with (
        patch(
            "avalan.conversation.durable_codec.dumps",
            side_effect=TypeError("serializer failure"),
        ),
        pytest.raises(conversation.ConversationCodecError),
    ):
        codec._encode({"valid": "value"})


def test_durable_codec_detects_post_decode_canonicalization_drift() -> None:
    codec = conversation.DurableConversationCodec()
    output = codec.encode_output(_output())
    continuation = codec.encode_continuation_reference(
        _continuation_reference()
    )

    with (
        patch.object(
            conversation.DurableConversationCodec,
            "encode_output",
            return_value=b"drift",
        ),
        pytest.raises(conversation.ConversationCodecError),
    ):
        codec.decode_output(output)
    with (
        patch.object(
            conversation.DurableConversationCodec,
            "encode_continuation_reference",
            return_value=b"drift",
        ),
        pytest.raises(conversation.ConversationCodecError),
    ):
        codec.decode_continuation_reference(continuation)
