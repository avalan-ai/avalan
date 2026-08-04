"""Define bounded async encryption and key-resolution boundaries."""

from .contract import CheckpointId, ProviderLaneId
from .errors import (
    ConversationCryptoAuthenticationError,
    ConversationFeatureUnavailableError,
    ConversationKeyMissingError,
    ConversationKeyPolicyError,
    ConversationKeyRetiredError,
    ConversationLimitError,
    ConversationValidationError,
)
from .value import (
    AuthorityDigest,
    ConversationCodecVersion,
    validate_identifier,
)

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from hmac import digest as hmac_digest
from importlib import import_module
from json import dumps
from secrets import token_bytes
from typing import Protocol, cast, final

CONVERSATION_AEAD_ALGORITHM = "aes-256-gcm"
CONVERSATION_PAYLOAD_SCHEMA_VERSION = 1
_AES_KEY_BYTES = 32
_AES_GCM_NONCE_BYTES = 12
_AES_GCM_TAG_BYTES = 16


class ConversationKeyStatus(StrEnum):
    """Identify whether a data key may write, read, or neither."""

    CURRENT = "current"
    GRACE = "grace"
    RETIRED = "retired"


class ConversationPayloadKind(StrEnum):
    """Identify one closed encrypted durable payload purpose."""

    CHECKPOINT = "checkpoint"
    LANE_OUTPUT = "lane_output"
    CONTINUATION_REFERENCE = "continuation_reference"
    DELETION_TARGET = "deletion_target"


class ConversationCryptoBoundary(StrEnum):
    """Identify deterministic key and AEAD fault-injection boundaries."""

    CURRENT_KEY_BEFORE = "current_key_before"
    CURRENT_KEY_AFTER = "current_key_after"
    READ_KEY_BEFORE = "read_key_before"
    READ_KEY_AFTER = "read_key_after"
    ENCRYPT_BEFORE = "encrypt_before"
    ENCRYPT_AFTER = "encrypt_after"
    DECRYPT_BEFORE = "decrypt_before"
    DECRYPT_AFTER = "decrypt_after"
    DIGEST_BEFORE = "digest_before"
    DIGEST_AFTER = "digest_after"


class ConversationCryptoBoundaryHook(Protocol):
    """Inject deterministic behavior around async cryptographic effects."""

    async def reach(self, boundary: ConversationCryptoBoundary) -> None:
        """Reach one named cryptographic boundary."""
        raise NotImplementedError


@final
class _NoopCryptoBoundaryHook:
    async def reach(self, boundary: ConversationCryptoBoundary) -> None:
        if not isinstance(boundary, ConversationCryptoBoundary):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConversationDataKey:
    """Carry one in-memory data key with explicit lifecycle metadata."""

    key_id: str
    revision: int
    status: ConversationKeyStatus
    key_bytes: bytes
    algorithm: str = CONVERSATION_AEAD_ALGORITHM

    def __post_init__(self) -> None:
        validate_identifier(self.key_id, "key_id")
        if type(self.revision) is not int or self.revision <= 0:
            raise ConversationValidationError()
        if not isinstance(self.status, ConversationKeyStatus):
            raise ConversationValidationError()
        if type(self.key_bytes) is not bytes or len(self.key_bytes) != 32:
            raise ConversationValidationError()
        if self.algorithm != CONVERSATION_AEAD_ALGORITHM:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return key metadata without material."""
        return (
            "ConversationDataKey("
            f"key_id={self.key_id!r}, revision={self.revision!r}, "
            f"status={self.status.value!r}, algorithm={self.algorithm!r}, "
            "key_bytes=<redacted>)"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationPayloadAssociatedData:
    """Bind ciphertext to its complete durable authority and position."""

    authority_digest: AuthorityDigest
    checkpoint_id: CheckpointId
    lane_id: ProviderLaneId
    sequence: int
    payload_kind: ConversationPayloadKind
    payload_schema_version: int
    codec_version: ConversationCodecVersion
    key_id: str
    key_revision: int

    def __post_init__(self) -> None:
        validate_identifier(self.authority_digest, "authority_digest")
        validate_identifier(self.checkpoint_id, "checkpoint_id")
        validate_identifier(self.lane_id, "lane_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ConversationValidationError()
        if not isinstance(self.payload_kind, ConversationPayloadKind):
            raise ConversationValidationError()
        if (
            type(self.payload_schema_version) is not int
            or self.payload_schema_version <= 0
            or type(self.codec_version) is not int
            or self.codec_version <= 0
        ):
            raise ConversationValidationError()
        validate_identifier(self.key_id, "key_id")
        if type(self.key_revision) is not int or self.key_revision <= 0:
            raise ConversationValidationError()

    def encode(self) -> bytes:
        """Return canonical non-secret associated-data bytes."""
        return dumps(
            {
                "authority_digest": self.authority_digest,
                "checkpoint_id": self.checkpoint_id,
                "codec_version": self.codec_version,
                "key_id": self.key_id,
                "key_revision": self.key_revision,
                "lane_id": self.lane_id,
                "payload_kind": self.payload_kind.value,
                "payload_schema_version": self.payload_schema_version,
                "sequence": self.sequence,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EncryptedConversationPayload:
    """Carry authenticated ciphertext and non-secret envelope metadata."""

    nonce: bytes
    ciphertext: bytes
    authenticated_digest: str
    associated_data_digest: str
    key_id: str
    key_revision: int
    algorithm: str

    def __post_init__(self) -> None:
        if (
            type(self.nonce) is not bytes
            or len(self.nonce) != _AES_GCM_NONCE_BYTES
            or type(self.ciphertext) is not bytes
            or len(self.ciphertext) <= _AES_GCM_TAG_BYTES
        ):
            raise ConversationValidationError()
        for value, name in (
            (self.authenticated_digest, "authenticated_digest"),
            (self.associated_data_digest, "associated_data_digest"),
            (self.key_id, "key_id"),
            (self.algorithm, "algorithm"),
        ):
            validate_identifier(value, name)
        for value in (
            self.authenticated_digest,
            self.associated_data_digest,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ConversationValidationError()
        if type(self.key_revision) is not int or self.key_revision <= 0:
            raise ConversationValidationError()
        if self.algorithm != CONVERSATION_AEAD_ALGORITHM:
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return authenticated envelope metadata without encrypted bytes."""
        return (
            "EncryptedConversationPayload("
            f"ciphertext_bytes={len(self.ciphertext)}, "
            f"key_id={self.key_id!r}, key_revision={self.key_revision}, "
            f"algorithm={self.algorithm!r}, encrypted_bytes=<redacted>)"
        )


class ConversationKeyResolver(Protocol):
    """Resolve durable write and grace-read keys asynchronously."""

    async def current_write_key(
        self,
        authority_digest: AuthorityDigest,
    ) -> ConversationDataKey:
        """Return the current write key for one trusted authority."""
        raise NotImplementedError

    async def read_key(
        self,
        authority_digest: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationDataKey:
        """Return one current or grace read key for an exact revision."""
        raise NotImplementedError


class ConversationCipher(Protocol):
    """Encrypt, authenticate, and decrypt bounded payloads asynchronously."""

    async def encrypt(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> EncryptedConversationPayload:
        """Encrypt one bounded payload with exact associated data."""
        raise NotImplementedError

    async def decrypt(
        self,
        payload: EncryptedConversationPayload,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> bytes:
        """Authenticate and decrypt one bounded payload."""
        raise NotImplementedError

    async def authenticated_digest(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> str:
        """Return one authority- and key-scoped content digest."""
        raise NotImplementedError


class _AesGcmPrimitive(Protocol):
    def encrypt(
        self, nonce: bytes, data: bytes, associated_data: bytes
    ) -> bytes: ...

    def decrypt(
        self, nonce: bytes, data: bytes, associated_data: bytes
    ) -> bytes: ...


class _AesGcmType(Protocol):
    def __call__(self, key: bytes) -> _AesGcmPrimitive: ...


class _AeadModule(Protocol):
    AESGCM: _AesGcmType


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AesGcmConversationCipher:
    """Perform bounded in-memory AES-GCM work behind an async boundary."""

    max_payload_bytes: int = 8_388_608
    nonce_factory: Callable[[int], bytes] = token_bytes
    module_importer: Callable[[str], object] = import_module
    boundary_hook: ConversationCryptoBoundaryHook = _NoopCryptoBoundaryHook()

    def __post_init__(self) -> None:
        if (
            type(self.max_payload_bytes) is not int
            or self.max_payload_bytes <= 0
        ):
            raise ConversationValidationError()
        if not callable(self.nonce_factory) or not callable(
            self.module_importer
        ):
            raise ConversationValidationError()
        if not hasattr(self.boundary_hook, "reach"):
            raise ConversationValidationError()

    async def encrypt(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> EncryptedConversationPayload:
        """Encrypt one bounded payload with exact associated data."""
        self._validate_plaintext(plaintext)
        self._validate_write_key(key, associated_data)
        await self.boundary_hook.reach(
            ConversationCryptoBoundary.ENCRYPT_BEFORE
        )
        nonce = self.nonce_factory(_AES_GCM_NONCE_BYTES)
        if type(nonce) is not bytes or len(nonce) != _AES_GCM_NONCE_BYTES:
            raise ConversationKeyPolicyError()
        encoded_ad = associated_data.encode()
        try:
            ciphertext = self._primitive(key).encrypt(
                nonce,
                plaintext,
                encoded_ad,
            )
        except ConversationFeatureUnavailableError:
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ConversationCryptoAuthenticationError() from None
        digest = self._digest_value(plaintext, key, encoded_ad)
        result = EncryptedConversationPayload(
            nonce=nonce,
            ciphertext=ciphertext,
            authenticated_digest=digest,
            associated_data_digest=sha256(encoded_ad).hexdigest(),
            key_id=key.key_id,
            key_revision=key.revision,
            algorithm=key.algorithm,
        )
        await self.boundary_hook.reach(
            ConversationCryptoBoundary.ENCRYPT_AFTER
        )
        return result

    async def decrypt(
        self,
        payload: EncryptedConversationPayload,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> bytes:
        """Authenticate and decrypt one bounded payload."""
        if type(payload) is not EncryptedConversationPayload:
            raise ConversationValidationError()
        self._validate_read_key(key, associated_data, payload)
        if (
            len(payload.ciphertext)
            > self.max_payload_bytes + _AES_GCM_TAG_BYTES
        ):
            raise ConversationLimitError()
        await self.boundary_hook.reach(
            ConversationCryptoBoundary.DECRYPT_BEFORE
        )
        encoded_ad = associated_data.encode()
        if not compare_digest(
            payload.associated_data_digest,
            sha256(encoded_ad).hexdigest(),
        ):
            raise ConversationCryptoAuthenticationError()
        try:
            plaintext = self._primitive(key).decrypt(
                payload.nonce,
                payload.ciphertext,
                encoded_ad,
            )
        except ConversationFeatureUnavailableError:
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ConversationCryptoAuthenticationError() from None
        self._validate_plaintext(plaintext)
        if not compare_digest(
            payload.authenticated_digest,
            self._digest_value(plaintext, key, encoded_ad),
        ):
            raise ConversationCryptoAuthenticationError()
        await self.boundary_hook.reach(
            ConversationCryptoBoundary.DECRYPT_AFTER
        )
        return plaintext

    async def authenticated_digest(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> str:
        """Return one authority- and key-scoped content digest."""
        self._validate_plaintext(plaintext)
        self._validate_key_identity(key, associated_data)
        await self.boundary_hook.reach(
            ConversationCryptoBoundary.DIGEST_BEFORE
        )
        result = self._digest_value(plaintext, key, associated_data.encode())
        await self.boundary_hook.reach(ConversationCryptoBoundary.DIGEST_AFTER)
        return result

    def _primitive(self, key: ConversationDataKey) -> _AesGcmPrimitive:
        try:
            module = cast(
                _AeadModule,
                self.module_importer(
                    "cryptography.hazmat.primitives.ciphers.aead"
                ),
            )
            return module.AESGCM(key.key_bytes)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise ConversationFeatureUnavailableError() from exc

    def _validate_plaintext(self, plaintext: bytes) -> None:
        if type(plaintext) is not bytes or not plaintext:
            raise ConversationValidationError()
        if len(plaintext) > self.max_payload_bytes:
            raise ConversationLimitError()

    @staticmethod
    def _validate_key_identity(
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> None:
        if (
            type(key) is not ConversationDataKey
            or type(associated_data) is not ConversationPayloadAssociatedData
        ):
            raise ConversationValidationError()
        if (
            key.key_id != associated_data.key_id
            or key.revision != associated_data.key_revision
        ):
            raise ConversationKeyPolicyError()

    @classmethod
    def _validate_write_key(
        cls,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> None:
        cls._validate_key_identity(key, associated_data)
        if key.status is not ConversationKeyStatus.CURRENT:
            raise ConversationKeyPolicyError()

    @classmethod
    def _validate_read_key(
        cls,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
        payload: EncryptedConversationPayload,
    ) -> None:
        cls._validate_key_identity(key, associated_data)
        if key.status is ConversationKeyStatus.RETIRED:
            raise ConversationKeyRetiredError()
        if (
            payload.key_id != key.key_id
            or payload.key_revision != key.revision
            or payload.algorithm != key.algorithm
        ):
            raise ConversationCryptoAuthenticationError()

    @staticmethod
    def _digest_value(
        plaintext: bytes,
        key: ConversationDataKey,
        associated_data: bytes,
    ) -> str:
        return hmac_digest(
            key.key_bytes,
            b"avalan.conversation.payload.v1\x00"
            + associated_data
            + b"\x00"
            + plaintext,
            "sha256",
        ).hex()


@final
class InMemoryConversationKeyResolver:
    """Resolve explicitly configured authority-scoped keys for tests."""

    def __init__(
        self,
        keys: Mapping[AuthorityDigest, tuple[ConversationDataKey, ...]],
        *,
        boundary_hook: ConversationCryptoBoundaryHook | None = None,
    ) -> None:
        if not isinstance(keys, Mapping) or not keys:
            raise ConversationKeyPolicyError()
        copied: dict[AuthorityDigest, tuple[ConversationDataKey, ...]] = {}
        for authority, values in keys.items():
            validate_identifier(authority, "authority_digest")
            if (
                type(values) is not tuple
                or not values
                or any(
                    type(value) is not ConversationDataKey for value in values
                )
            ):
                raise ConversationKeyPolicyError()
            self._validate_key_set(values)
            copied[authority] = values
        self._keys = copied
        self._hook = boundary_hook or _NoopCryptoBoundaryHook()

    async def current_write_key(
        self,
        authority_digest: AuthorityDigest,
    ) -> ConversationDataKey:
        """Return the current write key for one trusted authority."""
        validate_identifier(authority_digest, "authority_digest")
        await self._hook.reach(ConversationCryptoBoundary.CURRENT_KEY_BEFORE)
        values = self._keys.get(authority_digest)
        current = (
            tuple(
                value
                for value in values
                if value.status is ConversationKeyStatus.CURRENT
            )
            if values is not None
            else ()
        )
        if len(current) != 1:
            raise ConversationKeyMissingError()
        await self._hook.reach(ConversationCryptoBoundary.CURRENT_KEY_AFTER)
        return current[0]

    async def read_key(
        self,
        authority_digest: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationDataKey:
        """Return one current or grace read key for an exact revision."""
        validate_identifier(authority_digest, "authority_digest")
        validate_identifier(key_id, "key_id")
        if type(revision) is not int or revision <= 0:
            raise ConversationValidationError()
        await self._hook.reach(ConversationCryptoBoundary.READ_KEY_BEFORE)
        key = next(
            (
                value
                for value in self._keys.get(authority_digest, ())
                if value.key_id == key_id and value.revision == revision
            ),
            None,
        )
        if key is None:
            raise ConversationKeyMissingError()
        if key.status is ConversationKeyStatus.RETIRED:
            raise ConversationKeyRetiredError()
        await self._hook.reach(ConversationCryptoBoundary.READ_KEY_AFTER)
        return key

    async def replace_keys(
        self,
        authority_digest: AuthorityDigest,
        keys: tuple[ConversationDataKey, ...],
    ) -> None:
        """Replace one authority's test key policy atomically."""
        validate_identifier(authority_digest, "authority_digest")
        if (
            type(keys) is not tuple
            or not keys
            or any(type(value) is not ConversationDataKey for value in keys)
        ):
            raise ConversationKeyPolicyError()
        self._validate_key_set(keys)
        self._keys[authority_digest] = keys

    @staticmethod
    def _validate_key_set(keys: tuple[ConversationDataKey, ...]) -> None:
        identities = tuple((value.key_id, value.revision) for value in keys)
        if (
            len(identities) != len(set(identities))
            or sum(
                value.status is ConversationKeyStatus.CURRENT for value in keys
            )
            != 1
        ):
            raise ConversationKeyPolicyError()
        current = next(
            value
            for value in keys
            if value.status is ConversationKeyStatus.CURRENT
        )
        if any(
            value.revision >= current.revision
            for value in keys
            if value.status is ConversationKeyStatus.GRACE
        ):
            raise ConversationKeyPolicyError()
