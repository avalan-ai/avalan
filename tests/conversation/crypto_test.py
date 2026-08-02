"""Verify authenticated durable conversation encryption and key policy."""

from dataclasses import replace
from typing import cast

import pytest

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic cryptographic boundaries on asyncio only."""
    return "asyncio"


class _BoundaryHook:
    def __init__(self) -> None:
        self.boundaries: list[conversation.ConversationCryptoBoundary] = []

    async def reach(
        self,
        boundary: conversation.ConversationCryptoBoundary,
    ) -> None:
        self.boundaries.append(boundary)


class _FailingPrimitive:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes,
    ) -> bytes:
        raise self.error

    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes,
    ) -> bytes:
        raise self.error


class _FailingAesGcm:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, key: bytes) -> _FailingPrimitive:
        return _FailingPrimitive(self.error)


class _FailingAeadModule:
    def __init__(self, error: BaseException) -> None:
        self.AESGCM = _FailingAesGcm(error)


def _key(
    *,
    key_id: str = "key-1",
    revision: int = 1,
    status: conversation.ConversationKeyStatus = (
        conversation.ConversationKeyStatus.CURRENT
    ),
    material: bytes = b"k" * 32,
) -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id=key_id,
        revision=revision,
        status=status,
        key_bytes=material,
    )


def _associated_data(
    key: conversation.ConversationDataKey,
    **overrides: object,
) -> conversation.ConversationPayloadAssociatedData:
    values: dict[str, object] = {
        "authority_digest": conversation.AuthorityDigest("a" * 64),
        "checkpoint_id": conversation.CheckpointId("checkpoint-1"),
        "lane_id": conversation.ProviderLaneId("lane-1"),
        "sequence": 2,
        "payload_kind": conversation.ConversationPayloadKind.CHECKPOINT,
        "payload_schema_version": 1,
        "codec_version": conversation.ConversationCodecVersion(1),
        "key_id": key.key_id,
        "key_revision": key.revision,
    }
    values.update(overrides)
    return conversation.ConversationPayloadAssociatedData(**values)


async def test_aes_gcm_round_trip_binds_every_associated_dimension() -> None:
    key = _key()
    associated_data = _associated_data(key)
    hook = _BoundaryHook()
    cipher = conversation.AesGcmConversationCipher(
        nonce_factory=lambda size: bytes(range(size)),
        boundary_hook=hook,
    )

    encrypted = await cipher.encrypt(
        b"private conversation payload",
        key=key,
        associated_data=associated_data,
    )
    restored = await cipher.decrypt(
        encrypted,
        key=key,
        associated_data=associated_data,
    )
    digest = await cipher.authenticated_digest(
        restored,
        key=key,
        associated_data=associated_data,
    )

    assert restored == b"private conversation payload"
    assert encrypted.ciphertext != restored
    assert "private conversation payload" not in repr(encrypted)
    assert "redacted" in repr(encrypted)
    assert encrypted.authenticated_digest == digest
    assert hook.boundaries == [
        conversation.ConversationCryptoBoundary.ENCRYPT_BEFORE,
        conversation.ConversationCryptoBoundary.ENCRYPT_AFTER,
        conversation.ConversationCryptoBoundary.DECRYPT_BEFORE,
        conversation.ConversationCryptoBoundary.DECRYPT_AFTER,
        conversation.ConversationCryptoBoundary.DIGEST_BEFORE,
        conversation.ConversationCryptoBoundary.DIGEST_AFTER,
    ]

    for changed in (
        replace(associated_data, checkpoint_id="checkpoint-2"),
        replace(associated_data, lane_id="lane-2"),
        replace(associated_data, sequence=3),
        replace(
            associated_data,
            payload_kind=conversation.ConversationPayloadKind.LANE_OUTPUT,
        ),
        replace(associated_data, payload_schema_version=2),
        replace(associated_data, codec_version=2),
    ):
        with pytest.raises(conversation.ConversationCryptoAuthenticationError):
            await cipher.decrypt(
                encrypted,
                key=key,
                associated_data=changed,
            )


async def test_cipher_rejects_tampering_limits_and_wrong_key_policy() -> None:
    key = _key()
    associated_data = _associated_data(key)
    cipher = conversation.AesGcmConversationCipher(
        max_payload_bytes=32,
        nonce_factory=lambda size: b"n" * size,
    )
    encrypted = await cipher.encrypt(
        b"bounded",
        key=key,
        associated_data=associated_data,
    )

    for tampered in (
        replace(
            encrypted,
            ciphertext=bytes([encrypted.ciphertext[0] ^ 1])
            + encrypted.ciphertext[1:],
        ),
        replace(encrypted, authenticated_digest="0" * 64),
        replace(encrypted, associated_data_digest="0" * 64),
    ):
        with pytest.raises(conversation.ConversationCryptoAuthenticationError):
            await cipher.decrypt(
                tampered,
                key=key,
                associated_data=associated_data,
            )

    with pytest.raises(conversation.ConversationLimitError):
        await cipher.encrypt(
            b"x" * 33,
            key=key,
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await cipher.encrypt(b"", key=key, associated_data=associated_data)
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await cipher.encrypt(
            b"value",
            key=replace(key, status=conversation.ConversationKeyStatus.GRACE),
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationKeyRetiredError):
        await cipher.decrypt(
            encrypted,
            key=replace(
                key,
                status=conversation.ConversationKeyStatus.RETIRED,
            ),
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await cipher.authenticated_digest(
            b"value",
            key=key,
            associated_data=replace(associated_data, key_revision=2),
        )


async def test_cipher_fails_closed_without_optional_crypto_dependency() -> (
    None
):
    key = _key()
    cipher = conversation.AesGcmConversationCipher(
        module_importer=lambda _: (_ for _ in ()).throw(ImportError()),
    )

    with pytest.raises(conversation.ConversationFeatureUnavailableError):
        await cipher.encrypt(
            b"value",
            key=key,
            associated_data=_associated_data(key),
        )


async def test_key_resolver_rotation_grace_and_retirement_are_exact() -> None:
    authority = conversation.AuthorityDigest("a" * 64)
    first = _key()
    hook = _BoundaryHook()
    resolver = conversation.InMemoryConversationKeyResolver(
        {authority: (first,)},
        boundary_hook=hook,
    )

    assert await resolver.current_write_key(authority) == first
    assert (
        await resolver.read_key(
            authority,
            key_id=first.key_id,
            revision=first.revision,
        )
        == first
    )
    second = _key(key_id="key-2", revision=2, material=b"2" * 32)
    grace = replace(first, status=conversation.ConversationKeyStatus.GRACE)
    await resolver.replace_keys(authority, (grace, second))
    assert await resolver.current_write_key(authority) == second
    assert (
        await resolver.read_key(
            authority,
            key_id=grace.key_id,
            revision=grace.revision,
        )
        == grace
    )
    await resolver.replace_keys(
        authority,
        (
            replace(grace, status=conversation.ConversationKeyStatus.RETIRED),
            second,
        ),
    )
    with pytest.raises(conversation.ConversationKeyRetiredError):
        await resolver.read_key(
            authority,
            key_id=grace.key_id,
            revision=grace.revision,
        )
    with pytest.raises(conversation.ConversationKeyMissingError):
        await resolver.read_key(authority, key_id="absent", revision=1)
    with pytest.raises(conversation.ConversationKeyMissingError):
        await resolver.current_write_key(
            conversation.AuthorityDigest("b" * 64)
        )
    assert hook.boundaries[:4] == [
        conversation.ConversationCryptoBoundary.CURRENT_KEY_BEFORE,
        conversation.ConversationCryptoBoundary.CURRENT_KEY_AFTER,
        conversation.ConversationCryptoBoundary.READ_KEY_BEFORE,
        conversation.ConversationCryptoBoundary.READ_KEY_AFTER,
    ]


def test_crypto_values_reject_invalid_shapes_and_redact_key_material() -> None:
    key = _key()
    assert "kkkk" not in repr(key)
    assert "redacted" in repr(key)

    invalid_keys = (
        {"key_id": "", "key_bytes": b"k" * 32},
        {"revision": 0, "key_bytes": b"k" * 32},
        {"status": "current", "key_bytes": b"k" * 32},
        {"key_bytes": b"short"},
        {"algorithm": "other", "key_bytes": b"k" * 32},
    )
    for overrides in invalid_keys:
        values: dict[str, object] = {
            "key_id": "key",
            "revision": 1,
            "status": conversation.ConversationKeyStatus.CURRENT,
            "key_bytes": b"k" * 32,
        }
        values.update(overrides)
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ConversationDataKey(**values)

    with pytest.raises(conversation.ConversationValidationError):
        _associated_data(key, sequence=-1)
    with pytest.raises(conversation.ConversationValidationError):
        _associated_data(key, payload_kind="checkpoint")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.EncryptedConversationPayload(
            nonce=b"short",
            ciphertext=b"short",
            authenticated_digest="x",
            associated_data_digest="x",
            key_id="key",
            key_revision=1,
            algorithm="aes-256-gcm",
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AesGcmConversationCipher(max_payload_bytes=0)
    with pytest.raises(conversation.ConversationKeyPolicyError):
        conversation.InMemoryConversationKeyResolver(
            cast(
                dict[
                    conversation.AuthorityDigest,
                    tuple[conversation.ConversationDataKey, ...],
                ],
                {},
            )
        )
    with pytest.raises(conversation.ConversationKeyPolicyError):
        conversation.InMemoryConversationKeyResolver(
            {conversation.AuthorityDigest("a" * 64): (key, key)}
        )


async def test_crypto_boundary_and_value_edge_cases_fail_closed() -> None:
    key = _key()
    associated_data = _associated_data(key)
    cipher = conversation.AesGcmConversationCipher(
        max_payload_bytes=32,
        nonce_factory=lambda size: b"n" * size,
    )
    encrypted = await cipher.encrypt(
        b"bounded",
        key=key,
        associated_data=associated_data,
    )

    with pytest.raises(conversation.ConversationValidationError):
        await cipher.boundary_hook.reach(
            cast(conversation.ConversationCryptoBoundary, "invalid")
        )
    for overrides in (
        {"payload_schema_version": 0},
        {"codec_version": 0},
        {"key_revision": 0},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            _associated_data(key, **overrides)
    for overrides in (
        {"authenticated_digest": "x" * 64},
        {"key_revision": 0},
        {"algorithm": "other"},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(encrypted, **overrides)


async def test_cipher_configuration_and_envelope_guards_are_exact() -> None:
    key = _key()
    associated_data = _associated_data(key)
    for overrides in (
        {"nonce_factory": cast(object, None)},
        {"module_importer": cast(object, None)},
        {"boundary_hook": object()},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.AesGcmConversationCipher(**overrides)

    invalid_nonce = conversation.AesGcmConversationCipher(
        nonce_factory=lambda _: b"short"
    )
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await invalid_nonce.encrypt(
            b"value",
            key=key,
            associated_data=associated_data,
        )

    cipher = conversation.AesGcmConversationCipher(
        max_payload_bytes=32,
        nonce_factory=lambda size: b"n" * size,
    )
    encrypted = await cipher.encrypt(
        b"bounded",
        key=key,
        associated_data=associated_data,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await cipher.decrypt(
            cast(conversation.EncryptedConversationPayload, object()),
            key=key,
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationLimitError):
        await cipher.decrypt(
            replace(encrypted, ciphertext=b"x" * 49),
            key=key,
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await cipher.authenticated_digest(
            b"value",
            key=cast(conversation.ConversationDataKey, object()),
            associated_data=associated_data,
        )
    with pytest.raises(conversation.ConversationCryptoAuthenticationError):
        await cipher.decrypt(
            replace(encrypted, key_id="other-key"),
            key=key,
            associated_data=associated_data,
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            RuntimeError("primitive failure"),
            conversation.ConversationCryptoAuthenticationError,
        ),
        (SystemExit("interrupt"), SystemExit),
    ),
)
async def test_cipher_encrypt_preserves_process_control_and_maps_failures(
    error: BaseException,
    expected: type[BaseException],
) -> None:
    key = _key()
    cipher = conversation.AesGcmConversationCipher(
        nonce_factory=lambda size: b"n" * size,
        module_importer=lambda _: _FailingAeadModule(error),
    )

    with pytest.raises(expected):
        await cipher.encrypt(
            b"value",
            key=key,
            associated_data=_associated_data(key),
        )


async def test_cipher_decrypt_preserves_dependency_and_process_failures() -> (
    None
):
    key = _key()
    associated_data = _associated_data(key)
    cipher = conversation.AesGcmConversationCipher(
        nonce_factory=lambda size: b"n" * size
    )
    encrypted = await cipher.encrypt(
        b"value",
        key=key,
        associated_data=associated_data,
    )
    unavailable = conversation.AesGcmConversationCipher(
        module_importer=lambda _: (_ for _ in ()).throw(ImportError())
    )
    with pytest.raises(conversation.ConversationFeatureUnavailableError):
        await unavailable.decrypt(
            encrypted,
            key=key,
            associated_data=associated_data,
        )
    interrupted = conversation.AesGcmConversationCipher(
        module_importer=lambda _: _FailingAeadModule(SystemExit("interrupt"))
    )
    with pytest.raises(SystemExit):
        await interrupted.decrypt(
            encrypted,
            key=key,
            associated_data=associated_data,
        )


async def test_key_resolver_rejects_invalid_rotation_shapes() -> None:
    authority = conversation.AuthorityDigest("a" * 64)
    current = _key(revision=2)
    resolver = conversation.InMemoryConversationKeyResolver(
        {authority: (current,)}
    )

    with pytest.raises(conversation.ConversationKeyPolicyError):
        conversation.InMemoryConversationKeyResolver(
            {
                authority: cast(
                    tuple[conversation.ConversationDataKey, ...],
                    (object(),),
                )
            }
        )
    with pytest.raises(conversation.ConversationValidationError):
        await resolver.read_key(authority, key_id="key-1", revision=0)
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await resolver.replace_keys(
            authority,
            cast(tuple[conversation.ConversationDataKey, ...], (object(),)),
        )
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await resolver.replace_keys(
            authority,
            (
                replace(
                    current,
                    key_id="grace-key",
                    status=conversation.ConversationKeyStatus.GRACE,
                ),
                current,
            ),
        )
