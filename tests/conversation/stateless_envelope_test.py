"""Verify bounded caller-held continuation envelopes."""

from base64 import b64decode, urlsafe_b64encode
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

import avalan.conversation as conversation
import avalan.conversation.envelope as envelope_module
from avalan.types import JsonValue
from avalan.utils import to_json

pytestmark = pytest.mark.anyio

_NOW = datetime(2035, 1, 1, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    """Run async envelope checks on asyncio."""
    return "asyncio"


def _authority(
    *,
    tenant: str = "tenant-envelope",
    principal: str = "principal-envelope",
    agent: str = "agent-envelope",
    endpoint: str = "endpoint-envelope",
) -> conversation.AuthorityScope:
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId(tenant),
        principal_id=conversation.AuthorityPrincipalId(principal),
        agent_id=conversation.ConversationAgentId(agent),
        endpoint_id=conversation.AuthorityEndpointId(endpoint),
    )


def _binding(lane_id: str) -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type="tests.StatelessEnvelopeProvider",
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint="https://envelope.provider.test/v1",
        model_or_deployment=f"model-{lane_id}",
        provider_api_revision=conversation.ProviderApiRevision("api-v1"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-v1"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("config-v1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-v1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-v1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-v1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId("agent-envelope"),
    )


def _provider_message(
    binding: conversation.ProviderLaneBinding,
    index: int,
) -> conversation.ProviderItem:
    identifier = f"message-{binding.lane_id}-{index}"
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(identifier),
        lane_id=binding.lane_id,
        model_call_id=conversation.ConversationModelCallId(
            f"call-{binding.lane_id}-{index}"
        ),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(index),
        provider_index=conversation.ProviderItemIndex(index),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input=cast(
            Mapping[str, JsonValue],
            {
                "content": [
                    {
                        "annotations": [],
                        "text": f"safe-{binding.lane_id}",
                        "type": "output_text",
                    }
                ],
                "id": identifier,
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ),
        normalization_version=conversation.ConversationCodecVersion(1),
    )


def _retention() -> conversation.RetentionLimits:
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.TRANSIENT,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=3_600,
    )


def _checkpoint(
    *,
    authority: conversation.AuthorityScope | None = None,
    lanes: tuple[str, ...] = ("lane-parent", "lane-child"),
    sequence: int = 3,
    branch_id: str = "branch-envelope",
) -> conversation.ConversationCheckpoint:
    scope = authority or _authority()
    snapshots = []
    for lane_id in lanes:
        binding = _binding(lane_id)
        snapshots.append(
            conversation.StatelessProviderLaneSnapshot(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(binding.continuation_codec_version),
                    items=(_provider_message(binding, 0),),
                ),
                reasoning=conversation.EffectiveReasoningMetadata(
                    requested=conversation.ReasoningContext.AUTO,
                    effective=None,
                ),
                lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
                retention_policy=(
                    conversation.ChildLaneRetentionPolicy.RETAIN
                ),
            )
        )
    checkpoint = conversation.ConversationCheckpoint(
        identity=conversation.CheckpointIdentity(
            conversation_id=conversation.ConversationId(
                "conversation-envelope"
            ),
            logical_turn_id=conversation.LogicalTurnId("turn-envelope"),
            execution_segment_id=conversation.ExecutionSegmentId(
                "segment-envelope"
            ),
            checkpoint_id=conversation.CheckpointId("checkpoint-envelope"),
            branch_id=conversation.ConversationBranchId(branch_id),
            sequence=conversation.CheckpointSequence(sequence),
            parent_checkpoint_id=(
                conversation.CheckpointId("checkpoint-envelope-parent")
                if sequence
                else None
            ),
            parent_sequence=(
                conversation.CheckpointSequence(sequence - 1)
                if sequence
                else None
            ),
        ),
        kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
        lifecycle=conversation.CheckpointLifecycle.COMMITTED,
        authority=scope,
        content=conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(entries=()),
            lanes=tuple(snapshots),
        ),
        timestamps=conversation.CheckpointTimestamps(
            created_at=_NOW - timedelta(minutes=1),
            committed_at=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        ),
        retention=_retention(),
    )
    return conversation.with_checkpoint_integrity(checkpoint)


def _key(
    key_id: str,
    revision: int,
    status: conversation.ContinuationEnvelopeKeyStatus,
    byte: int,
) -> conversation.ContinuationEnvelopeKey:
    return conversation.ContinuationEnvelopeKey(
        key_id=key_id,
        revision=revision,
        status=status,
        key_bytes=bytes([byte]) * 32,
    )


def _codec(
    authority: conversation.AuthorityScope,
    *keys: conversation.ContinuationEnvelopeKey,
    limits: conversation.ContinuationEnvelopeLimits | None = None,
) -> tuple[
    conversation.ContinuationEnvelopeCodec,
    conversation.InMemoryContinuationEnvelopeKeyResolver,
]:
    resolver = conversation.InMemoryContinuationEnvelopeKeyResolver(
        {conversation.authority_digest(authority): keys}
    )
    return (
        conversation.ContinuationEnvelopeCodec(
            key_resolver=resolver,
            limits=limits or conversation.ContinuationEnvelopeLimits(),
        ),
        resolver,
    )


def _outer(token: conversation.ContinuationEnvelopeToken) -> dict[str, object]:
    value = token.value_for_response().removeprefix(
        conversation.CONTINUATION_ENVELOPE_PREFIX
    )
    encoded = b64decode(value + "=" * (-len(value) % 4), altchars=b"-_")
    result = loads(encoded)
    assert isinstance(result, dict)
    return result


def _token(value: dict[str, object]) -> conversation.ContinuationEnvelopeToken:
    encoded = dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    payload = urlsafe_b64encode(encoded).rstrip(b"=").decode()
    return conversation.ContinuationEnvelopeToken.from_request(
        conversation.CONTINUATION_ENVELOPE_PREFIX + payload,
        max_chars=6_000_000,
    )


async def test_multilane_round_trip_branch_and_redaction() -> None:
    scope = _authority()
    checkpoint = _checkpoint(authority=scope)
    codec, _ = _codec(
        scope,
        _key(
            "key-active",
            2,
            conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            2,
        ),
    )
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    token = await codec.seal(
        checkpoint,
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_parent"),
        issued_at=_NOW,
    )

    assert repr(token) == "ContinuationEnvelopeToken(<redacted>)"
    assert token.value_for_response() not in repr(token)
    with pytest.raises(TypeError):
        str(token)
    with pytest.raises(TypeError):
        f"{token}"

    ordinary = await codec.open(
        token,
        authority=authority,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
        now=_NOW + timedelta(seconds=1),
    )
    assert ordinary.checkpoint == checkpoint
    assert ordinary.public_parent == "resp_parent"
    assert len(ordinary.checkpoint.content.lanes) == 2

    branch = await codec.open(
        token,
        authority=authority,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.EXPLICIT_BRANCH,
            branch_id=conversation.ConversationBranchId("branch-sibling"),
        ),
        now=_NOW + timedelta(seconds=1),
    )
    assert branch.target_branch_id == "branch-sibling"
    assert ordinary.target_branch_id == "branch-envelope"


async def test_rotation_window_and_key_outcomes_are_stable() -> None:
    scope = _authority()
    checkpoint = _checkpoint(authority=scope, lanes=("lane-parent",))
    old = _key(
        "key-old",
        1,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        1,
    )
    codec, resolver = _codec(scope, old)
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    old_token = await codec.seal(
        checkpoint,
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_old"),
        issued_at=_NOW,
    )
    new = _key(
        "key-new",
        2,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        2,
    )
    retiring = replace(
        old,
        status=conversation.ContinuationEnvelopeKeyStatus.RETIRING,
    )
    await resolver.replace_keys(
        conversation.authority_digest(scope),
        (retiring, new),
    )
    opened = await codec.open(
        old_token,
        authority=authority,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
        now=_NOW + timedelta(seconds=1),
    )
    next_token = await codec.seal(
        opened.checkpoint,
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_new"),
        issued_at=_NOW + timedelta(seconds=2),
    )
    assert _outer(next_token)["key_id"] == "key-new"

    for status, error in (
        (
            conversation.ContinuationEnvelopeKeyStatus.RETIRED,
            conversation.ConversationKeyRetiredError,
        ),
        (
            conversation.ContinuationEnvelopeKeyStatus.COMPROMISED,
            conversation.ConversationKeyCompromisedError,
        ),
    ):
        await resolver.replace_keys(
            conversation.authority_digest(scope),
            (replace(old, status=status), new),
        )
        with pytest.raises(error):
            await codec.open(
                old_token,
                authority=authority,
                advance=conversation.ContinuationEnvelopeAdvance(
                    mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
                ),
                now=_NOW + timedelta(seconds=3),
            )

    unknown = _outer(old_token)
    unknown["key_id"] = "key-unknown"
    with pytest.raises(conversation.ConversationKeyMissingError):
        await codec.open(
            _token(unknown),
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW + timedelta(seconds=3),
        )


async def test_key_selection_authenticates_before_lifecycle_policy() -> None:
    """Hide known-key lifecycle state until outer coordinates authenticate."""
    scope = _authority()
    active = _key(
        "key-active",
        2,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        2,
    )
    retired = _key(
        "key-retired",
        1,
        conversation.ContinuationEnvelopeKeyStatus.RETIRED,
        1,
    )
    codec, _ = _codec(scope, retired, active)
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    token = await codec.seal(
        _checkpoint(authority=scope, lanes=("lane-parent",)),
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_active"),
        issued_at=_NOW,
    )
    tampered = _outer(token)
    tampered["key_id"] = retired.key_id
    tampered["key_revision"] = retired.revision

    with pytest.raises(conversation.ConversationCryptoAuthenticationError):
        await codec.open(
            _token(tampered),
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW + timedelta(seconds=1),
        )


def test_opaque_token_blocks_serialization_and_recursion() -> None:
    """Reject generic disclosure and adversarial recursive JSON safely."""
    scope = _authority()
    codec, _ = _codec(
        scope,
        _key(
            "key-active",
            1,
            conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            1,
        ),
    )
    token = conversation.ContinuationEnvelopeToken.from_request(
        "avl_ce1.valid",
        max_chars=100,
    )
    with pytest.raises(TypeError):
        asdict(token)
    with pytest.raises(TypeError):
        to_json(token)
    with pytest.raises(conversation.ConversationLimitError):
        codec._decode_json(b"{}", max_bytes=1)

    adversarial_depth = codec.limits.max_depth + 1
    adversarial = b"[" * adversarial_depth + b"0" + b"]" * adversarial_depth
    with pytest.raises(conversation.ConversationLimitError) as raised:
        codec._decode_json(adversarial)
    assert (
        raised.value.code is conversation.ConversationErrorCode.LIMIT_EXCEEDED
    )
    assert str(raised.value) == "conversation state exceeds a configured limit"
    assert "[" not in str(raised.value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("ciphertext", "AA"),
        ("nonce", "AA"),
        ("authenticated_digest", "0" * 64),
        ("associated_data_digest", "0" * 64),
        ("checkpoint_id", "checkpoint-tampered"),
        ("lane_id", "lane-tampered"),
        ("sequence", 2),
        ("version", 2),
    ),
)
async def test_tampered_outer_fields_fail_closed(
    field: str,
    replacement: object,
) -> None:
    scope = _authority()
    codec, _ = _codec(
        scope,
        _key(
            "key-active",
            1,
            conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            1,
        ),
    )
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    sealed = await codec.seal(
        _checkpoint(authority=scope, lanes=("lane-parent",)),
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_parent"),
        issued_at=_NOW,
    )
    outer = _outer(sealed)
    outer[field] = replacement
    with pytest.raises(conversation.ConversationError):
        await codec.open(
            _token(outer),
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW + timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    "authority",
    (
        _authority(tenant="tenant-other"),
        _authority(principal="principal-other"),
        _authority(agent="agent-other"),
        _authority(endpoint="endpoint-other"),
    ),
)
async def test_cross_authority_and_deployment_replay_is_redacted(
    authority: conversation.AuthorityScope,
) -> None:
    scope = _authority()
    active = _key(
        "key-active",
        1,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        1,
    )
    scope_digest = conversation.authority_digest(scope)
    other_digest = conversation.authority_digest(authority)
    resolver = conversation.InMemoryContinuationEnvelopeKeyResolver(
        {
            scope_digest: (active,),
            other_digest: (active,),
        }
    )
    codec = conversation.ContinuationEnvelopeCodec(key_resolver=resolver)
    trusted = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    token = await codec.seal(
        _checkpoint(authority=scope, lanes=("lane-parent",)),
        authority=trusted,
        public_parent=conversation.PublicResponseId("resp_parent"),
        issued_at=_NOW,
    )
    for attempted in (
        conversation.ContinuationEnvelopeAuthority(
            authority=authority,
            deployment_id="deployment-envelope",
        ),
        conversation.ContinuationEnvelopeAuthority(
            authority=scope,
            deployment_id="deployment-other",
        ),
    ):
        with pytest.raises(conversation.ConversationError) as raised:
            await codec.open(
                token,
                authority=attempted,
                advance=conversation.ContinuationEnvelopeAdvance(
                    mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
                ),
                now=_NOW + timedelta(seconds=1),
            )
        assert token.value_for_response() not in repr(raised.value)


async def test_head_expiry_and_bounded_parser_rejections() -> None:
    scope = _authority()
    limits = conversation.ContinuationEnvelopeLimits(
        max_token_chars=100_000,
        max_plaintext_bytes=100_000,
        max_depth=48,
        max_items=100_000,
        max_string_bytes=50_000,
        ttl_seconds=10,
        clock_skew_seconds=0,
    )
    codec, _ = _codec(
        scope,
        _key(
            "key-active",
            1,
            conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            1,
        ),
        limits=limits,
    )
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    token = await codec.seal(
        _checkpoint(authority=scope, lanes=("lane-parent",)),
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_head"),
        issued_at=_NOW,
        head_id=conversation.NamedHeadId("head-envelope"),
        head_revision=conversation.NamedHeadRevision(4),
    )
    opened = await codec.open(
        token,
        authority=authority,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.NAMED_HEAD,
            head_id=conversation.NamedHeadId("head-envelope"),
            expected_head_revision=conversation.NamedHeadRevision(4),
        ),
        now=_NOW + timedelta(seconds=1),
    )
    assert opened.target_branch_id == "branch-envelope"
    with pytest.raises(conversation.ConversationAuthorizationError):
        await codec.open(
            token,
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.NAMED_HEAD,
                head_id=conversation.NamedHeadId("head-envelope"),
                expected_head_revision=conversation.NamedHeadRevision(3),
            ),
            now=_NOW + timedelta(seconds=1),
        )
    with pytest.raises(conversation.ConversationExpiredError):
        await codec.open(
            token,
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.NAMED_HEAD,
                head_id=conversation.NamedHeadId("head-envelope"),
                expected_head_revision=conversation.NamedHeadRevision(4),
            ),
            now=_NOW + timedelta(seconds=10),
        )
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ContinuationEnvelopeToken.from_request(
            conversation.CONTINUATION_ENVELOPE_PREFIX + "A" * 100_001,
            max_chars=100_000,
        )
    duplicate = b'{"version":1,"version":1}'
    duplicate_value = urlsafe_b64encode(duplicate).rstrip(b"=").decode()
    with pytest.raises(conversation.ConversationCodecError):
        await codec.open(
            conversation.ContinuationEnvelopeToken.from_request(
                conversation.CONTINUATION_ENVELOPE_PREFIX + duplicate_value,
                max_chars=100_000,
            ),
            authority=authority,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW,
        )


async def test_envelope_value_objects_and_key_policy_fail_closed() -> None:
    scope = _authority()
    active = _key(
        "key-active",
        2,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        2,
    )
    retiring = _key(
        "key-retiring",
        1,
        conversation.ContinuationEnvelopeKeyStatus.RETIRING,
        1,
    )
    assert "key_bytes=<redacted>" in repr(active)
    assert (
        retiring.for_read().status is conversation.ConversationKeyStatus.GRACE
    )
    for key, operation, error in (
        (
            retiring,
            retiring.for_write,
            conversation.ConversationKeyPolicyError,
        ),
        (
            replace(
                retiring,
                status=conversation.ContinuationEnvelopeKeyStatus.RETIRED,
            ),
            replace(
                retiring,
                status=conversation.ContinuationEnvelopeKeyStatus.RETIRED,
            ).for_read,
            conversation.ConversationKeyRetiredError,
        ),
        (
            replace(
                retiring,
                status=(
                    conversation.ContinuationEnvelopeKeyStatus.COMPROMISED
                ),
            ),
            replace(
                retiring,
                status=(
                    conversation.ContinuationEnvelopeKeyStatus.COMPROMISED
                ),
            ).for_read,
            conversation.ConversationKeyCompromisedError,
        ),
    ):
        assert key.key_bytes != b""
        with pytest.raises(error):
            operation()

    with pytest.raises(conversation.ConversationKeyPolicyError):
        replace(active, revision=0)
    with pytest.raises(conversation.ConversationKeyPolicyError):
        conversation.InMemoryContinuationEnvelopeKeyResolver({})
    resolver = conversation.InMemoryContinuationEnvelopeKeyResolver(
        {conversation.authority_digest(scope): (active, retiring)}
    )
    with pytest.raises(conversation.ConversationKeyMissingError):
        await resolver.active_key(cast(conversation.AuthorityDigest, "f" * 64))
    with pytest.raises(conversation.ConversationValidationError):
        await resolver.read_key(
            conversation.authority_digest(scope),
            key_id="key-active",
            revision=0,
        )
    for keys in (
        cast(tuple[conversation.ContinuationEnvelopeKey, ...], []),
        (active, active),
        (
            active,
            replace(
                active,
                key_id="key-future-retiring",
                status=conversation.ContinuationEnvelopeKeyStatus.RETIRING,
            ),
        ),
    ):
        with pytest.raises(conversation.ConversationKeyPolicyError):
            await resolver.replace_keys(
                conversation.authority_digest(scope),
                keys,
            )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.ContinuationEnvelopeLimits(max_items=0)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ContinuationEnvelopeLimits(clock_skew_seconds=-1)
    with pytest.raises(conversation.ConversationCodecError):
        conversation.ContinuationEnvelopeToken(_value="not-an-envelope")
    with pytest.raises(conversation.ConversationLimitError):
        conversation.ContinuationEnvelopeToken.from_request(
            conversation.CONTINUATION_ENVELOPE_PREFIX,
            max_chars=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ContinuationEnvelopeAuthority(
            authority=cast(conversation.AuthorityScope, object()),
            deployment_id="deployment-envelope",
        )

    invalid_advances = (
        {"mode": cast(conversation.ParentAdvanceMode, "ordinary_child")},
        {"mode": conversation.ParentAdvanceMode.EXPLICIT_BRANCH},
        {
            "mode": conversation.ParentAdvanceMode.EXPLICIT_BRANCH,
            "branch_id": conversation.ConversationBranchId("branch-new"),
            "expected_head_revision": conversation.NamedHeadRevision(1),
        },
        {"mode": conversation.ParentAdvanceMode.NAMED_HEAD},
        {
            "mode": conversation.ParentAdvanceMode.NAMED_HEAD,
            "head_id": conversation.NamedHeadId("head-one"),
            "expected_head_revision": cast(
                conversation.NamedHeadRevision,
                True,
            ),
        },
        {
            "mode": conversation.ParentAdvanceMode.ORDINARY_CHILD,
            "head_id": conversation.NamedHeadId("head-one"),
        },
    )
    for values in invalid_advances:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ContinuationEnvelopeAdvance(**values)

    codec, _ = _codec(scope, active)
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    token = await codec.seal(
        _checkpoint(authority=scope, lanes=("lane-parent",)),
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_parent"),
        issued_at=_NOW,
    )
    opened = await codec.open(
        token,
        authority=authority,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
        now=_NOW,
    )
    assert "lane_count=1" in repr(opened)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            opened,
            checkpoint=cast(conversation.ConversationCheckpoint, object()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(opened, expires_at=opened.issued_at)


async def test_envelope_codec_defensive_validation_is_total() -> None:
    scope = _authority()
    active = _key(
        "key-active",
        1,
        conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        1,
    )
    codec, resolver = _codec(scope, active)
    authority = conversation.ContinuationEnvelopeAuthority(
        authority=scope,
        deployment_id="deployment-envelope",
    )
    checkpoint = _checkpoint(authority=scope, lanes=("lane-parent",))
    advance = conversation.ContinuationEnvelopeAdvance(
        mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ContinuationEnvelopeCodec(
            key_resolver=cast(
                conversation.ContinuationEnvelopeKeyResolver,
                object(),
            )
        )
    for head_id, head_revision in (
        (conversation.NamedHeadId("head-one"), None),
        (
            conversation.NamedHeadId("head-one"),
            cast(conversation.NamedHeadRevision, -1),
        ),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            await codec.seal(
                checkpoint,
                authority=authority,
                public_parent=conversation.PublicResponseId("resp_parent"),
                issued_at=_NOW,
                head_id=head_id,
                head_revision=head_revision,
            )
    with patch.object(
        resolver,
        "active_key",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(conversation.ConversationKeyPolicyError):
            await codec.seal(
                checkpoint,
                authority=authority,
                public_parent=conversation.PublicResponseId("resp_parent"),
                issued_at=_NOW,
            )
    small_codec, _ = _codec(
        scope,
        active,
        limits=conversation.ContinuationEnvelopeLimits(max_token_chars=1),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await small_codec.seal(
            checkpoint,
            authority=authority,
            public_parent=conversation.PublicResponseId("resp_parent"),
            issued_at=_NOW,
        )
    token = await codec.seal(
        checkpoint,
        authority=authority,
        public_parent=conversation.PublicResponseId("resp_parent"),
        issued_at=_NOW,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await codec.open(
            cast(conversation.ContinuationEnvelopeToken, object()),
            authority=authority,
            advance=advance,
            now=_NOW,
        )
    with patch.object(
        resolver,
        "read_key",
        new=AsyncMock(return_value=object()),
    ):
        with pytest.raises(conversation.ConversationKeyPolicyError):
            await codec.open(
                token,
                authority=authority,
                advance=advance,
                now=_NOW,
            )

    claims = codec._decode_claims(
        codec._encode_claims(
            checkpoint,
            codec.checkpoint_codec.encode(checkpoint),
            authority=authority,
            public_parent=conversation.PublicResponseId("resp_parent"),
            issued_at=_NOW,
            expires_at=_NOW + timedelta(seconds=codec.limits.ttl_seconds),
            head_id=None,
            head_revision=None,
        )
    )
    for replacement_claims in (
        {**claims, "issued_at": "invalid"},
        {**claims, "public_parent": 7},
    ):
        with (
            patch.object(
                codec.__class__,
                "_decode_claims",
                return_value=replacement_claims,
            ),
            patch.object(codec.__class__, "_validate_claim_bindings"),
            patch.object(
                codec.__class__,
                "_validate_advance",
                return_value=checkpoint.identity.branch_id,
            ),
            pytest.raises(conversation.ConversationCodecError),
        ):
            await codec.open(
                token,
                authority=authority,
                advance=advance,
                now=_NOW,
            )

    tiny_codec, _ = _codec(
        scope,
        active,
        limits=conversation.ContinuationEnvelopeLimits(max_plaintext_bytes=1),
    )
    with pytest.raises(conversation.ConversationLimitError):
        tiny_codec._encode_claims(
            checkpoint,
            b"checkpoint",
            authority=authority,
            public_parent=conversation.PublicResponseId("resp_parent"),
            issued_at=_NOW,
            expires_at=_NOW + timedelta(seconds=1),
            head_id=None,
            head_revision=None,
        )
    with patch.object(
        envelope_module,
        "canonical_json_bytes",
        return_value=cast(bytes, "not-bytes"),
    ):
        with pytest.raises(conversation.ConversationCodecError):
            codec._encode_claims(
                checkpoint,
                b"checkpoint",
                authority=authority,
                public_parent=conversation.PublicResponseId("resp_parent"),
                issued_at=_NOW,
                expires_at=_NOW + timedelta(seconds=1),
                head_id=None,
                head_revision=None,
            )

    with pytest.raises(conversation.ConversationLimitError):
        codec._decode_claims(b"")
    raw_claims = cast(
        dict[str, object],
        loads(
            codec._encode_claims(
                checkpoint,
                codec.checkpoint_codec.encode(checkpoint),
                authority=authority,
                public_parent=conversation.PublicResponseId("resp_parent"),
                issued_at=_NOW,
                expires_at=_NOW + timedelta(seconds=codec.limits.ttl_seconds),
                head_id=None,
                head_revision=None,
            )
        ),
    )

    def encoded(value: dict[str, object]) -> bytes:
        return dumps(value, separators=(",", ":"), sort_keys=True).encode()

    missing = dict(raw_claims)
    missing.pop("kind")
    wrong_kind = {**raw_claims, "kind": "wrong"}
    wrong_size = {
        **raw_claims,
        "checkpoint_bytes": cast(int, raw_claims["checkpoint_bytes"]) + 1,
    }
    wrong_lanes = {**raw_claims, "lanes": {}}
    claim_cases: tuple[
        tuple[dict[str, object], type[conversation.ConversationError]], ...
    ] = (
        (missing, conversation.ConversationCodecError),
        (wrong_kind, conversation.ConversationCodecError),
        (wrong_size, conversation.ConversationCryptoAuthenticationError),
        (wrong_lanes, conversation.ConversationCodecError),
    )
    for claim_value, claim_error in claim_cases:
        with pytest.raises(claim_error):
            codec._decode_claims(encoded(claim_value))

    binding_claims = dict(claims)
    binding_cases: tuple[
        tuple[dict[str, object], type[conversation.ConversationError]], ...
    ] = (
        (
            {**binding_claims, "agent_id": "agent-other"},
            conversation.ConversationAuthorizationError,
        ),
        (
            {
                **binding_claims,
                "issued_at": _NOW + timedelta(minutes=1),
            },
            conversation.ConversationAuthorizationError,
        ),
        (
            {
                **binding_claims,
                "expires_at": (
                    cast(datetime, binding_claims["expires_at"])
                    + timedelta(seconds=1)
                ),
            },
            conversation.ConversationCryptoAuthenticationError,
        ),
    )
    for binding_value, binding_error in binding_cases:
        with pytest.raises(binding_error):
            codec._validate_claim_bindings(
                binding_value,
                checkpoint,
                authority=authority,
                scope_digest=conversation.authority_digest(scope),
                now=_NOW,
            )

    advance_cases: tuple[
        tuple[
            dict[str, object],
            conversation.ContinuationEnvelopeAdvance,
            type[conversation.ConversationError],
        ],
        ...,
    ] = (
        (
            {"head_id": "head-one", "head_revision": None},
            advance,
            conversation.ConversationCryptoAuthenticationError,
        ),
        (
            {"head_id": "head-one", "head_revision": 1},
            advance,
            conversation.ConversationAuthorizationError,
        ),
        (
            {"head_id": None, "head_revision": None},
            conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.EXPLICIT_BRANCH,
                branch_id=checkpoint.identity.branch_id,
            ),
            conversation.ConversationAuthorizationError,
        ),
    )
    for head_claims, selected_advance, advance_error in advance_cases:
        with pytest.raises(advance_error):
            codec._validate_advance(
                head_claims,
                checkpoint,
                selected_advance,
            )

    oversized_token = conversation.ContinuationEnvelopeToken(
        _value=conversation.CONTINUATION_ENVELOPE_PREFIX + "A" * 10
    )
    with pytest.raises(conversation.ConversationLimitError):
        small_codec._decode_token(oversized_token)
    outer_cases: tuple[dict[str, object], ...] = (
        {"version": 1},
        {**_outer(token), "sequence": -1},
    )
    for outer in outer_cases:
        with pytest.raises(conversation.ConversationCodecError):
            codec._decode_token(_token(outer))
    empty_checkpoint = cast(
        conversation.ConversationCheckpoint,
        SimpleNamespace(content=SimpleNamespace(lanes=())),
    )
    with pytest.raises(conversation.ConversationValidationError):
        codec._associated_data(
            empty_checkpoint,
            conversation.authority_digest(scope),
            key_id="key-active",
            key_revision=1,
        )
    with pytest.raises(conversation.ConversationAuthorizationError):
        codec._validate_checkpoint(
            cast(conversation.ConversationCheckpoint, object()),
            authority,
        )

    for payload in (b"not-json", b"[]", b'{"value":NaN}'):
        with pytest.raises(conversation.ConversationCodecError):
            codec._decode_json(payload)
    for encoded_value in ("", "with=padding", "%%%"):
        with pytest.raises(conversation.ConversationCodecError):
            codec._base64_decode(encoded_value)
    assert codec._optional_string(None) is None
    assert codec._optional_integer(None) is None
    for operation in (
        lambda: codec._string(1),
        lambda: codec._integer(True),
        lambda: codec._aware_utc(datetime(2035, 1, 1)),
    ):
        with pytest.raises(conversation.ConversationError):
            operation()
