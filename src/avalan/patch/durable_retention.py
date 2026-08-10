"""Encrypt and open dormant durable patch-retention values.

This internal module is not a patch tool, route, or worker.  It provides the
bounded AES-GCM envelope used by authenticated durable test hosts before a
future reviewed retention service is activated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from json import dumps
from secrets import token_bytes
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from avalan.patch.domain import (
    Audience,
    PatchRequestId,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
)
from avalan.patch.durable_store import (
    DurableRequestIdentity,
    DurableRetentionAuthorizer,
    DurableRetentionEnvelopeValidator,
    DurableRetentionKind,
    DurableRetentionRecord,
    DurableStoreError,
    DurableStoreErrorCode,
    EncryptedRetentionValue,
)

_AES_GCM_KEY_BYTES = 32
_AES_GCM_NONCE_BYTES = 12
_AES_GCM_TAG_BYTES = 16
_RETENTION_SCHEMA_VERSION = 1
_MAX_PLAINTEXT_BYTES = 1_048_576 - _AES_GCM_NONCE_BYTES - _AES_GCM_TAG_BYTES


@dataclass(frozen=True, slots=True, repr=False)
class DurableRetentionKey:
    """Carry one in-memory AES-256-GCM retention key without rendering it."""

    key_id: PatchRetentionKeyId
    key_bytes: bytes

    def __post_init__(self) -> None:
        """Require one exact fixed-length durable retention data key."""
        if (
            type(self.key_id) is not PatchRetentionKeyId
            or type(self.key_bytes) is not bytes
            or len(self.key_bytes) != _AES_GCM_KEY_BYTES
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)

    def __repr__(self) -> str:
        """Render key metadata without exposing key material."""
        return (
            "DurableRetentionKey(key_id="
            + repr(self.key_id)
            + ", key_bytes=<redacted>)"
        )


class DurableRetentionKeyResolver(Protocol):
    """Resolve write and exact-version read keys asynchronously."""

    async def active_key(self) -> DurableRetentionKey:
        """Return the active durable retention key for a new value."""

    async def read_key(
        self, key_id: PatchRetentionKeyId
    ) -> DurableRetentionKey:
        """Return the exact durable retention key for one stored envelope."""


@dataclass(frozen=True, slots=True)
class DurableRetentionBinding:
    """Bind a ciphertext to one request, retention record, and purpose."""

    request_id: PatchRequestId
    retention_id: PatchRetentionRecordId
    kind: DurableRetentionKind

    def __post_init__(self) -> None:
        """Require exact persisted coordinates for authenticated data."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.retention_id) is not PatchRetentionRecordId
            or type(self.kind) is not DurableRetentionKind
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)

    def associated_data(self, key_id: PatchRetentionKeyId) -> bytes:
        """Encode the canonical non-secret authenticated retention binding."""
        if type(key_id) is not PatchRetentionKeyId:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return dumps(
            {
                "key_id": key_id.value,
                "kind": self.kind.value,
                "request_id": self.request_id.value,
                "retention_id": self.retention_id.value,
                "version": _RETENTION_SCHEMA_VERSION,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")


@dataclass(frozen=True, slots=True, repr=False)
class DurableEncryptedRetention:
    """Return versioned ciphertext and its retained key identity."""

    key_id: PatchRetentionKeyId
    value: EncryptedRetentionValue

    def __post_init__(self) -> None:
        """Require exact key and opaque ciphertext witnesses."""
        if (
            type(self.key_id) is not PatchRetentionKeyId
            or type(self.value) is not EncryptedRetentionValue
            or self.value.size().value
            <= _AES_GCM_NONCE_BYTES + _AES_GCM_TAG_BYTES
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)


class AesGcmDurableRetentionCipher:
    """Seal and open bounded durable retention values through AES-GCM."""

    def __init__(self, resolver: DurableRetentionKeyResolver) -> None:
        """Bind the async key resolver without resolving a key eagerly."""
        if not callable(getattr(resolver, "active_key", None)) or not callable(
            getattr(resolver, "read_key", None)
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        self._resolver = resolver

    async def seal(
        self, plaintext: bytes, binding: DurableRetentionBinding
    ) -> DurableEncryptedRetention:
        """Encrypt one bounded private retention value under the active key."""
        if (
            type(plaintext) is not bytes
            or len(plaintext) > _MAX_PLAINTEXT_BYTES
            or type(binding) is not DurableRetentionBinding
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
        key = await self._active_key()
        nonce = token_bytes(_AES_GCM_NONCE_BYTES)
        ciphertext = self._aead(key).encrypt(
            nonce,
            plaintext,
            binding.associated_data(key.key_id),
        )
        return DurableEncryptedRetention(
            key.key_id,
            EncryptedRetentionValue(nonce + ciphertext),
        )

    async def open(
        self,
        encrypted: DurableEncryptedRetention,
        binding: DurableRetentionBinding,
    ) -> bytes:
        """Authenticate and decrypt one exact durable retention envelope."""
        if (
            type(encrypted) is not DurableEncryptedRetention
            or type(binding) is not DurableRetentionBinding
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        key = await self._read_key(encrypted.key_id)
        encoded = encrypted.value._ciphertext
        nonce = encoded[:_AES_GCM_NONCE_BYTES]
        ciphertext = encoded[_AES_GCM_NONCE_BYTES:]
        try:
            plaintext = self._aead(key).decrypt(
                nonce,
                ciphertext,
                binding.associated_data(key.key_id),
            )
        except InvalidTag:
            raise DurableStoreError(
                DurableStoreErrorCode.RETENTION_DENIED
            ) from None
        if len(plaintext) > _MAX_PLAINTEXT_BYTES:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_LIMIT)
        return plaintext

    async def _active_key(self) -> DurableRetentionKey:
        """Resolve and validate the active write key without disclosure."""
        try:
            key = await self._resolver.active_key()
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise DurableStoreError(
                DurableStoreErrorCode.RETENTION_DENIED
            ) from None
        if type(key) is not DurableRetentionKey:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return key

    async def _read_key(
        self, key_id: PatchRetentionKeyId
    ) -> DurableRetentionKey:
        """Resolve and bind exactly the stored read-key identifier."""
        try:
            key = await self._resolver.read_key(key_id)
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise DurableStoreError(
                DurableStoreErrorCode.RETENTION_DENIED
            ) from None
        if type(key) is not DurableRetentionKey or key.key_id != key_id:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return key

    def _aead(self, key: DurableRetentionKey) -> AESGCM:
        """Construct the repository-pinned AES-GCM primitive for one key."""
        return AESGCM(key.key_bytes)


class AesGcmDurableRetentionEnvelopeValidator(
    DurableRetentionEnvelopeValidator
):
    """Authenticate stored versioned retention envelopes before persistence."""

    def __init__(self, cipher: AesGcmDurableRetentionCipher) -> None:
        """Bind the exact AEAD cipher selected by trusted retention config."""
        if type(cipher) is not AesGcmDurableRetentionCipher:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        self._cipher = cipher

    async def validate(
        self,
        request_id: PatchRequestId,
        record: DurableRetentionRecord,
    ) -> None:
        """Open and authenticate one exact request-bound retention envelope."""
        if (
            type(request_id) is not PatchRequestId
            or type(record) is not DurableRetentionRecord
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        await self._cipher.open(
            DurableEncryptedRetention(record.key_id, record.value),
            DurableRetentionBinding(
                request_id,
                record.retention_id,
                record.kind,
            ),
        )


class StaticDurableRetentionAuthorizer(DurableRetentionAuthorizer):
    """Return a trusted fixed test-host audience policy for each value kind."""

    def __init__(self, audiences: frozenset[Audience]) -> None:
        """Bind an immutable nonempty authenticated audience policy."""
        if (
            type(audiences) is not frozenset
            or not audiences
            or any(type(item) is not Audience for item in audiences)
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        self._audiences = audiences

    async def audiences_for(
        self,
        identity: DurableRequestIdentity,
        kind: DurableRetentionKind,
    ) -> frozenset[Audience]:
        """Derive configured audiences only after identity authentication."""
        if (
            type(identity) is not DurableRequestIdentity
            or type(kind) is not DurableRetentionKind
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return self._audiences


class InMemoryDurableRetentionKeyResolver:
    """Provide deterministic test-host key rotation without persistence."""

    def __init__(
        self,
        active_key_id: PatchRetentionKeyId,
        keys: Mapping[PatchRetentionKeyId, DurableRetentionKey],
    ) -> None:
        """Bind an immutable exact key map and selected active key."""
        if (
            type(active_key_id) is not PatchRetentionKeyId
            or not isinstance(keys, Mapping)
            or any(
                type(key_id) is not PatchRetentionKeyId
                or type(key) is not DurableRetentionKey
                or key.key_id != key_id
                for key_id, key in keys.items()
            )
            or active_key_id not in keys
        ):
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        self._active_key_id = active_key_id
        self._keys = dict(keys)

    async def active_key(self) -> DurableRetentionKey:
        """Return the explicitly selected active test-host data key."""
        return self._keys[self._active_key_id]

    async def read_key(
        self, key_id: PatchRetentionKeyId
    ) -> DurableRetentionKey:
        """Return an exact retained key or fail without key enumeration."""
        if type(key_id) is not PatchRetentionKeyId or key_id not in self._keys:
            raise DurableStoreError(DurableStoreErrorCode.RETENTION_DENIED)
        return self._keys[key_id]
