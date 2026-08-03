"""Seal bounded caller-held continuation checkpoints safely."""

from .codec import (
    CHECKPOINT_CODEC_VERSION,
    ConversationCheckpointCodec,
)
from .contract import (
    AuthorityScope,
    CheckpointId,
    ConversationBranchId,
    NamedHeadId,
    NamedHeadRevision,
    ParentAdvanceMode,
    ProviderLaneId,
    PublicResponseId,
)
from .crypto import (
    CONVERSATION_AEAD_ALGORITHM,
    CONVERSATION_PAYLOAD_SCHEMA_VERSION,
    AesGcmConversationCipher,
    ConversationDataKey,
    ConversationKeyStatus,
    ConversationPayloadAssociatedData,
    ConversationPayloadKind,
    EncryptedConversationPayload,
)
from .errors import (
    ConversationAuthorizationError,
    ConversationCodecError,
    ConversationCryptoAuthenticationError,
    ConversationExpiredError,
    ConversationKeyCompromisedError,
    ConversationKeyMissingError,
    ConversationKeyPolicyError,
    ConversationKeyRetiredError,
    ConversationLimitError,
    ConversationValidationError,
)
from .observability import authority_digest
from .state import CheckpointLifecycle, ConversationCheckpoint
from .value import (
    AuthorityDigest,
    JsonLimits,
    canonical_json_bytes,
    freeze_json_value,
    thaw_json_value,
    validate_identifier,
)

from base64 import b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from json import JSONDecodeError, loads
from re import fullmatch
from typing import Protocol, final

CONTINUATION_ENVELOPE_VERSION = 1
CONTINUATION_ENVELOPE_NAMESPACE = "avalan.responses.continuation"
CONTINUATION_ENVELOPE_PREFIX = "avl_ce1."


class ContinuationEnvelopeKeyStatus(StrEnum):
    """Identify one caller-envelope key lifecycle state."""

    ACTIVE = "active"
    RETIRING = "retiring"
    RETIRED = "retired"
    COMPROMISED = "compromised"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContinuationEnvelopeKey:
    """Carry envelope key material with an explicit rotation policy."""

    key_id: str
    revision: int
    status: ContinuationEnvelopeKeyStatus
    key_bytes: bytes

    def __post_init__(self) -> None:
        validate_identifier(self.key_id, "key_id")
        if (
            type(self.revision) is not int
            or self.revision <= 0
            or not isinstance(self.status, ContinuationEnvelopeKeyStatus)
            or type(self.key_bytes) is not bytes
            or len(self.key_bytes) != 32
        ):
            raise ConversationKeyPolicyError()

    def for_write(self) -> ConversationDataKey:
        """Return an active key for the shared asynchronous cipher."""
        if self.status is not ContinuationEnvelopeKeyStatus.ACTIVE:
            raise ConversationKeyPolicyError()
        return self._conversation_key(ConversationKeyStatus.CURRENT)

    def for_read(self) -> ConversationDataKey:
        """Return a current or rotation-window read key."""
        if self.status is ContinuationEnvelopeKeyStatus.RETIRED:
            raise ConversationKeyRetiredError()
        if self.status is ContinuationEnvelopeKeyStatus.COMPROMISED:
            raise ConversationKeyCompromisedError()
        status = (
            ConversationKeyStatus.CURRENT
            if self.status is ContinuationEnvelopeKeyStatus.ACTIVE
            else ConversationKeyStatus.GRACE
        )
        return self._conversation_key(status)

    def for_authentication(self) -> ConversationDataKey:
        """Return key material without revealing lifecycle policy first."""
        return self._conversation_key(ConversationKeyStatus.CURRENT)

    def _conversation_key(
        self,
        status: ConversationKeyStatus,
    ) -> ConversationDataKey:
        return ConversationDataKey(
            key_id=self.key_id,
            revision=self.revision,
            status=status,
            key_bytes=self.key_bytes,
        )

    def __repr__(self) -> str:
        """Return key lifecycle metadata without material."""
        return (
            "ContinuationEnvelopeKey("
            f"key_id={self.key_id!r}, revision={self.revision}, "
            f"status={self.status.value!r}, key_bytes=<redacted>)"
        )


class ContinuationEnvelopeKeyResolver(Protocol):
    """Resolve authority-scoped caller-envelope keys asynchronously."""

    async def active_key(
        self,
        authority_scope_digest: AuthorityDigest,
    ) -> ContinuationEnvelopeKey:
        """Return the only active sealing key."""
        ...

    async def read_key(
        self,
        authority_scope_digest: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ContinuationEnvelopeKey:
        """Return one exact key for opening under rotation policy."""
        ...


@final
class InMemoryContinuationEnvelopeKeyResolver:
    """Resolve explicitly configured envelope keys for embedded runtimes."""

    def __init__(
        self,
        keys: Mapping[AuthorityDigest, tuple[ContinuationEnvelopeKey, ...]],
    ) -> None:
        if not isinstance(keys, Mapping) or not keys:
            raise ConversationKeyPolicyError()
        self._keys: dict[
            AuthorityDigest, tuple[ContinuationEnvelopeKey, ...]
        ] = {}
        for scope_digest, values in keys.items():
            validate_identifier(scope_digest, "authority_scope_digest")
            self._validate_keys(values)
            self._keys[scope_digest] = values

    async def active_key(
        self,
        authority_scope_digest: AuthorityDigest,
    ) -> ContinuationEnvelopeKey:
        """Return the only active sealing key."""
        validate_identifier(
            authority_scope_digest,
            "authority_scope_digest",
        )
        active = tuple(
            key
            for key in self._keys.get(authority_scope_digest, ())
            if key.status is ContinuationEnvelopeKeyStatus.ACTIVE
        )
        if len(active) != 1:
            raise ConversationKeyMissingError()
        return active[0]

    async def read_key(
        self,
        authority_scope_digest: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ContinuationEnvelopeKey:
        """Return one exact key for opening under rotation policy."""
        validate_identifier(
            authority_scope_digest,
            "authority_scope_digest",
        )
        validate_identifier(key_id, "key_id")
        if type(revision) is not int or revision <= 0:
            raise ConversationValidationError()
        key = next(
            (
                candidate
                for candidate in self._keys.get(authority_scope_digest, ())
                if candidate.key_id == key_id
                and candidate.revision == revision
            ),
            None,
        )
        if key is None:
            raise ConversationKeyMissingError()
        return key

    async def replace_keys(
        self,
        authority_scope_digest: AuthorityDigest,
        keys: tuple[ContinuationEnvelopeKey, ...],
    ) -> None:
        """Replace one authority's rotation policy atomically."""
        validate_identifier(
            authority_scope_digest,
            "authority_scope_digest",
        )
        self._validate_keys(keys)
        self._keys[authority_scope_digest] = keys

    @staticmethod
    def _validate_keys(
        keys: tuple[ContinuationEnvelopeKey, ...],
    ) -> None:
        if (
            type(keys) is not tuple
            or not keys
            or any(type(key) is not ContinuationEnvelopeKey for key in keys)
        ):
            raise ConversationKeyPolicyError()
        identities = tuple((key.key_id, key.revision) for key in keys)
        active = tuple(
            key
            for key in keys
            if key.status is ContinuationEnvelopeKeyStatus.ACTIVE
        )
        if len(identities) != len(set(identities)) or len(active) != 1:
            raise ConversationKeyPolicyError()
        if any(
            key.revision >= active[0].revision
            for key in keys
            if key.status is ContinuationEnvelopeKeyStatus.RETIRING
        ):
            raise ConversationKeyPolicyError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationEnvelopeLimits:
    """Bound caller-held token, plaintext, and recursive JSON state."""

    max_token_chars: int = 6_000_000
    max_plaintext_bytes: int = 4_194_304
    max_depth: int = 48
    max_items: int = 100_000
    max_string_bytes: int = 2_097_152
    ttl_seconds: int = 3600
    clock_skew_seconds: int = 30

    def __post_init__(self) -> None:
        for value in (
            self.max_token_chars,
            self.max_plaintext_bytes,
            self.max_depth,
            self.max_items,
            self.max_string_bytes,
            self.ttl_seconds,
        ):
            if type(value) is not int or value <= 0:
                raise ConversationValidationError()
        if (
            type(self.clock_skew_seconds) is not int
            or self.clock_skew_seconds < 0
        ):
            raise ConversationValidationError()

    @property
    def json_limits(self) -> JsonLimits:
        """Return recursive JSON limits for every decoded token value."""
        return JsonLimits(
            max_depth=self.max_depth,
            max_items=self.max_items,
            max_string_bytes=self.max_string_bytes,
        )


@final
class ContinuationEnvelopeToken:
    """Wrap sensitive caller-held state without implicit disclosure."""

    __slots__ = ("__value",)

    def __init__(self, *, _value: str) -> None:
        if (
            type(_value) is not str
            or fullmatch(r"avl_ce1\.[A-Za-z0-9_-]+", _value) is None
        ):
            raise ConversationCodecError()
        self.__value = _value

    @classmethod
    def from_request(
        cls,
        value: str,
        *,
        max_chars: int,
    ) -> "ContinuationEnvelopeToken":
        """Wrap one bounded exact body-extension value."""
        if (
            type(value) is not str
            or type(max_chars) is not int
            or max_chars <= 0
            or not value
            or len(value) > max_chars
            or fullmatch(r"avl_ce1\.[A-Za-z0-9_-]+", value) is None
        ):
            raise ConversationLimitError()
        return cls(_value=value)

    def value_for_response(self) -> str:
        """Reveal the token only for its exact response-extension field."""
        return self.__value

    @property
    def character_count(self) -> int:
        """Return content-free token length accounting."""
        return len(self.__value)

    @property
    def digest(self) -> str:
        """Return a content-safe token digest for replay accounting."""
        return sha256(self.__value.encode("ascii")).hexdigest()

    def __repr__(self) -> str:
        """Return a redacted diagnostic representation."""
        return "ContinuationEnvelopeToken(<redacted>)"

    def __str__(self) -> str:
        """Reject implicit conversion outside the response serializer."""
        raise TypeError("continuation envelope cannot be converted to text")

    def __format__(self, format_spec: str) -> str:
        """Reject format-string disclosure."""
        del format_spec
        raise TypeError("continuation envelope cannot be formatted")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationEnvelopeAuthority:
    """Bind envelope use to trusted served deployment authority."""

    authority: AuthorityScope
    deployment_id: str

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        validate_identifier(self.deployment_id, "deployment_id")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationEnvelopeAdvance:
    """Describe one authenticated immutable-parent or named-head use."""

    mode: ParentAdvanceMode
    branch_id: ConversationBranchId | None = None
    head_id: NamedHeadId | None = None
    expected_head_revision: NamedHeadRevision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ParentAdvanceMode):
            raise ConversationValidationError()
        if self.mode is ParentAdvanceMode.EXPLICIT_BRANCH:
            if self.branch_id is None or self.head_id is not None:
                raise ConversationValidationError()
            validate_identifier(self.branch_id, "branch_id")
            if self.expected_head_revision is not None:
                raise ConversationValidationError()
            return
        if self.mode is ParentAdvanceMode.NAMED_HEAD:
            if self.head_id is None or self.expected_head_revision is None:
                raise ConversationValidationError()
            validate_identifier(self.head_id, "head_id")
            if (
                type(self.expected_head_revision) is not int
                or self.expected_head_revision < 0
                or self.branch_id is not None
            ):
                raise ConversationValidationError()
            return
        if (
            self.mode is not ParentAdvanceMode.ORDINARY_CHILD
            or self.branch_id is not None
            or self.head_id is not None
            or self.expected_head_revision is not None
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OpenedContinuationEnvelope:
    """Return one authorized checkpoint and its requested child policy."""

    checkpoint: ConversationCheckpoint
    public_parent: PublicResponseId
    advance: ContinuationEnvelopeAdvance
    target_branch_id: ConversationBranchId
    token_digest: str
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint) is not ConversationCheckpoint
            or self.checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
            or type(self.advance) is not ContinuationEnvelopeAdvance
        ):
            raise ConversationValidationError()
        validate_identifier(self.public_parent, "public_parent")
        validate_identifier(self.target_branch_id, "target_branch_id")
        validate_identifier(self.token_digest, "token_digest")
        if (
            self.issued_at.utcoffset() is None
            or self.expires_at.utcoffset() is None
            or self.expires_at <= self.issued_at
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only safe opened-envelope accounting."""
        return (
            "OpenedContinuationEnvelope("
            f"lane_count={len(self.checkpoint.content.lanes)}, "
            f"sequence={self.checkpoint.identity.sequence})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _DecodedContinuationToken:
    """Carry encrypted bytes and their authenticated coordinates."""

    payload: EncryptedConversationPayload
    checkpoint_id: CheckpointId
    lane_id: ProviderLaneId
    sequence: int


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationEnvelopeCodec:
    """Encode and open strict version-one encrypted continuation tokens."""

    key_resolver: ContinuationEnvelopeKeyResolver
    limits: ContinuationEnvelopeLimits = ContinuationEnvelopeLimits()
    checkpoint_codec: ConversationCheckpointCodec = (
        ConversationCheckpointCodec()
    )
    cipher: AesGcmConversationCipher = AesGcmConversationCipher()

    def __post_init__(self) -> None:
        if (
            not callable(getattr(self.key_resolver, "active_key", None))
            or not callable(getattr(self.key_resolver, "read_key", None))
            or type(self.limits) is not ContinuationEnvelopeLimits
            or type(self.checkpoint_codec) is not ConversationCheckpointCodec
            or type(self.cipher) is not AesGcmConversationCipher
            or self.cipher.max_payload_bytes < self.limits.max_plaintext_bytes
        ):
            raise ConversationValidationError()

    async def seal(
        self,
        checkpoint: ConversationCheckpoint,
        *,
        authority: ContinuationEnvelopeAuthority,
        public_parent: PublicResponseId,
        issued_at: datetime,
        head_id: NamedHeadId | None = None,
        head_revision: NamedHeadRevision | None = None,
    ) -> ContinuationEnvelopeToken:
        """Seal one committed checkpoint under the active authority key."""
        self._validate_checkpoint(checkpoint, authority)
        validate_identifier(public_parent, "public_parent")
        issued_at = self._aware_utc(issued_at)
        if (head_id is None) != (head_revision is None):
            raise ConversationValidationError()
        if head_id is not None:
            validate_identifier(head_id, "head_id")
            if type(head_revision) is not int or head_revision < 0:
                raise ConversationValidationError()
        scope_digest = authority_digest(authority.authority)
        key = await self.key_resolver.active_key(scope_digest)
        if type(key) is not ContinuationEnvelopeKey:
            raise ConversationKeyPolicyError()
        expires_at = issued_at + timedelta(seconds=self.limits.ttl_seconds)
        checkpoint_bytes = self.checkpoint_codec.encode(checkpoint)
        plaintext = self._encode_claims(
            checkpoint,
            checkpoint_bytes,
            authority=authority,
            public_parent=public_parent,
            issued_at=issued_at,
            expires_at=expires_at,
            head_id=head_id,
            head_revision=head_revision,
        )
        associated_data = self._associated_data(
            checkpoint,
            scope_digest,
            key_id=key.key_id,
            key_revision=key.revision,
        )
        encrypted = await self.cipher.encrypt(
            plaintext,
            key=key.for_write(),
            associated_data=associated_data,
        )
        encoded = self._encode_token(encrypted, associated_data)
        if len(encoded) > self.limits.max_token_chars:
            raise ConversationLimitError()
        return ContinuationEnvelopeToken(_value=encoded)

    async def open(
        self,
        token: ContinuationEnvelopeToken,
        *,
        authority: ContinuationEnvelopeAuthority,
        advance: ContinuationEnvelopeAdvance,
        now: datetime,
    ) -> OpenedContinuationEnvelope:
        """Open and authorize one exact body-extension token."""
        if (
            type(token) is not ContinuationEnvelopeToken
            or type(authority) is not ContinuationEnvelopeAuthority
            or type(advance) is not ContinuationEnvelopeAdvance
        ):
            raise ConversationValidationError()
        now = self._aware_utc(now)
        decoded = self._decode_token(token)
        encrypted = decoded.payload
        scope_digest = authority_digest(authority.authority)
        key = await self.key_resolver.read_key(
            scope_digest,
            key_id=encrypted.key_id,
            revision=encrypted.key_revision,
        )
        if (
            type(key) is not ContinuationEnvelopeKey
            or key.key_id != encrypted.key_id
            or key.revision != encrypted.key_revision
        ):
            raise ConversationKeyPolicyError()
        associated_data = ConversationPayloadAssociatedData(
            authority_digest=scope_digest,
            checkpoint_id=decoded.checkpoint_id,
            lane_id=decoded.lane_id,
            sequence=decoded.sequence,
            payload_kind=ConversationPayloadKind.CONTINUATION_REFERENCE,
            payload_schema_version=CONVERSATION_PAYLOAD_SCHEMA_VERSION,
            codec_version=CHECKPOINT_CODEC_VERSION,
            key_id=key.key_id,
            key_revision=key.revision,
        )
        plaintext = await self.cipher.decrypt(
            encrypted,
            key=key.for_authentication(),
            associated_data=associated_data,
        )
        key.for_read()
        claims = self._decode_claims(plaintext)
        checkpoint = claims["checkpoint"]
        assert type(checkpoint) is ConversationCheckpoint
        self._validate_checkpoint(checkpoint, authority)
        self._validate_claim_bindings(
            claims,
            checkpoint,
            authority=authority,
            scope_digest=scope_digest,
            now=now,
        )
        target_branch = self._validate_advance(
            claims,
            checkpoint,
            advance,
        )
        issued_at = claims["issued_at"]
        expires_at = claims["expires_at"]
        if not isinstance(issued_at, datetime) or not isinstance(
            expires_at,
            datetime,
        ):
            raise ConversationCodecError()
        public_parent = claims["public_parent"]
        if type(public_parent) is not str:
            raise ConversationCodecError()
        return OpenedContinuationEnvelope(
            checkpoint=checkpoint,
            public_parent=PublicResponseId(public_parent),
            advance=advance,
            target_branch_id=target_branch,
            token_digest=token.digest,
            issued_at=issued_at,
            expires_at=expires_at,
        )

    def _encode_claims(
        self,
        checkpoint: ConversationCheckpoint,
        checkpoint_bytes: bytes,
        *,
        authority: ContinuationEnvelopeAuthority,
        public_parent: PublicResponseId,
        issued_at: datetime,
        expires_at: datetime,
        head_id: NamedHeadId | None,
        head_revision: NamedHeadRevision | None,
    ) -> bytes:
        lanes = self._lane_bindings(checkpoint)
        value = {
            "agent_id": str(authority.authority.agent_id),
            "authority_digest": str(authority_digest(authority.authority)),
            "branch_id": str(checkpoint.identity.branch_id),
            "checkpoint": self._base64_encode(checkpoint_bytes),
            "checkpoint_bytes": len(checkpoint_bytes),
            "checkpoint_id": str(checkpoint.identity.checkpoint_id),
            "codec_version": int(CHECKPOINT_CODEC_VERSION),
            "deployment_id": authority.deployment_id,
            "endpoint_id": str(authority.authority.endpoint_id),
            "expires_at": int(expires_at.timestamp()),
            "head_id": str(head_id) if head_id is not None else None,
            "head_revision": (
                int(head_revision) if head_revision is not None else None
            ),
            "issued_at": int(issued_at.timestamp()),
            "kind": CONTINUATION_ENVELOPE_NAMESPACE,
            "lane_count": len(lanes),
            "lanes": lanes,
            "max_authenticated_bytes": self.limits.max_plaintext_bytes,
            "principal_id": str(authority.authority.principal_id),
            "public_parent": str(public_parent),
            "sequence": int(checkpoint.identity.sequence),
            "tenant_id": str(authority.authority.tenant_id),
            "version": CONTINUATION_ENVELOPE_VERSION,
        }
        encoded = canonical_json_bytes(
            freeze_json_value(value, limits=self.limits.json_limits)
        )
        if not encoded or len(encoded) > self.limits.max_plaintext_bytes:
            raise ConversationLimitError()
        if type(encoded) is not bytes:
            raise ConversationCodecError()
        return encoded

    def _decode_claims(self, encoded: bytes) -> dict[str, object]:
        if (
            type(encoded) is not bytes
            or not encoded
            or len(encoded) > self.limits.max_plaintext_bytes
        ):
            raise ConversationLimitError()
        raw = self._decode_json(
            encoded,
            max_bytes=self.limits.max_plaintext_bytes,
        )
        keys = {
            "agent_id",
            "authority_digest",
            "branch_id",
            "checkpoint",
            "checkpoint_bytes",
            "checkpoint_id",
            "codec_version",
            "deployment_id",
            "endpoint_id",
            "expires_at",
            "head_id",
            "head_revision",
            "issued_at",
            "kind",
            "lane_count",
            "lanes",
            "max_authenticated_bytes",
            "principal_id",
            "public_parent",
            "sequence",
            "tenant_id",
            "version",
        }
        if set(raw) != keys:
            raise ConversationCodecError()
        if (
            self._string(raw["kind"]) != CONTINUATION_ENVELOPE_NAMESPACE
            or self._integer(raw["version"]) != CONTINUATION_ENVELOPE_VERSION
            or self._integer(raw["codec_version"]) != CHECKPOINT_CODEC_VERSION
            or self._integer(raw["max_authenticated_bytes"])
            != self.limits.max_plaintext_bytes
        ):
            raise ConversationCodecError()
        checkpoint_bytes = self._base64_decode(self._string(raw["checkpoint"]))
        if len(checkpoint_bytes) != self._integer(raw["checkpoint_bytes"]):
            raise ConversationCryptoAuthenticationError()
        checkpoint = self.checkpoint_codec.decode(checkpoint_bytes)
        issued_at = datetime.fromtimestamp(
            self._integer(raw["issued_at"]),
            tz=UTC,
        )
        expires_at = datetime.fromtimestamp(
            self._integer(raw["expires_at"]),
            tz=UTC,
        )
        lanes = raw["lanes"]
        if not isinstance(lanes, list):
            raise ConversationCodecError()
        result = dict(raw)
        result["checkpoint"] = checkpoint
        result["issued_at"] = issued_at
        result["expires_at"] = expires_at
        result["lanes"] = tuple(lanes)
        return result

    def _validate_claim_bindings(
        self,
        claims: Mapping[str, object],
        checkpoint: ConversationCheckpoint,
        *,
        authority: ContinuationEnvelopeAuthority,
        scope_digest: AuthorityDigest,
        now: datetime,
    ) -> None:
        expected = (
            str(scope_digest),
            str(authority.authority.tenant_id),
            str(authority.authority.principal_id),
            str(authority.authority.agent_id),
            str(authority.authority.endpoint_id),
            authority.deployment_id,
            str(checkpoint.identity.checkpoint_id),
            str(checkpoint.identity.branch_id),
            int(checkpoint.identity.sequence),
            len(checkpoint.content.lanes),
            tuple(self._lane_bindings(checkpoint)),
        )
        actual = (
            self._string(claims["authority_digest"]),
            self._string(claims["tenant_id"]),
            self._string(claims["principal_id"]),
            self._string(claims["agent_id"]),
            self._string(claims["endpoint_id"]),
            self._string(claims["deployment_id"]),
            self._string(claims["checkpoint_id"]),
            self._string(claims["branch_id"]),
            self._integer(claims["sequence"]),
            self._integer(claims["lane_count"]),
            claims["lanes"],
        )
        if actual != expected:
            raise ConversationAuthorizationError()
        issued_at = claims["issued_at"]
        expires_at = claims["expires_at"]
        assert isinstance(issued_at, datetime)
        assert isinstance(expires_at, datetime)
        skew = timedelta(seconds=self.limits.clock_skew_seconds)
        if issued_at > now + skew:
            raise ConversationAuthorizationError()
        if expires_at <= now:
            raise ConversationExpiredError()
        if expires_at - issued_at != timedelta(
            seconds=self.limits.ttl_seconds
        ):
            raise ConversationCryptoAuthenticationError()
        validate_identifier(
            self._string(claims["public_parent"]),
            "public_parent",
        )

    def _validate_advance(
        self,
        claims: Mapping[str, object],
        checkpoint: ConversationCheckpoint,
        advance: ContinuationEnvelopeAdvance,
    ) -> ConversationBranchId:
        head_id = self._optional_string(claims["head_id"])
        head_revision = self._optional_integer(claims["head_revision"])
        if (head_id is None) != (head_revision is None):
            raise ConversationCryptoAuthenticationError()
        if advance.mode is ParentAdvanceMode.ORDINARY_CHILD:
            if head_id is not None:
                raise ConversationAuthorizationError()
            return checkpoint.identity.branch_id
        if advance.mode is ParentAdvanceMode.EXPLICIT_BRANCH:
            assert advance.branch_id is not None
            if (
                head_id is not None
                or advance.branch_id == checkpoint.identity.branch_id
            ):
                raise ConversationAuthorizationError()
            return advance.branch_id
        assert advance.head_id is not None
        assert advance.expected_head_revision is not None
        if (
            head_id != advance.head_id
            or head_revision != advance.expected_head_revision
        ):
            raise ConversationAuthorizationError()
        return checkpoint.identity.branch_id

    def _encode_token(
        self,
        payload: EncryptedConversationPayload,
        associated_data: ConversationPayloadAssociatedData,
    ) -> str:
        value = {
            "algorithm": payload.algorithm,
            "associated_data_digest": payload.associated_data_digest,
            "authenticated_digest": payload.authenticated_digest,
            "ciphertext": self._base64_encode(payload.ciphertext),
            "checkpoint_id": str(associated_data.checkpoint_id),
            "key_id": payload.key_id,
            "key_revision": payload.key_revision,
            "lane_id": str(associated_data.lane_id),
            "nonce": self._base64_encode(payload.nonce),
            "sequence": associated_data.sequence,
            "version": CONTINUATION_ENVELOPE_VERSION,
        }
        encoded = canonical_json_bytes(freeze_json_value(value))
        return CONTINUATION_ENVELOPE_PREFIX + self._base64_encode(encoded)

    def _decode_token(
        self,
        token: ContinuationEnvelopeToken,
    ) -> _DecodedContinuationToken:
        value = token.value_for_response()
        if len(value) > self.limits.max_token_chars:
            raise ConversationLimitError()
        encoded = self._base64_decode(
            value.removeprefix(CONTINUATION_ENVELOPE_PREFIX)
        )
        raw = self._decode_json(
            encoded,
            max_bytes=self.limits.max_token_chars,
        )
        if set(raw) != {
            "algorithm",
            "associated_data_digest",
            "authenticated_digest",
            "ciphertext",
            "checkpoint_id",
            "key_id",
            "key_revision",
            "lane_id",
            "nonce",
            "sequence",
            "version",
        }:
            raise ConversationCodecError()
        if (
            self._integer(raw["version"]) != CONTINUATION_ENVELOPE_VERSION
            or self._string(raw["algorithm"]) != CONVERSATION_AEAD_ALGORITHM
        ):
            raise ConversationCodecError()
        checkpoint_id = CheckpointId(self._string(raw["checkpoint_id"]))
        lane_id = ProviderLaneId(self._string(raw["lane_id"]))
        validate_identifier(checkpoint_id, "checkpoint_id")
        validate_identifier(lane_id, "lane_id")
        sequence = self._integer(raw["sequence"])
        if sequence < 0:
            raise ConversationCodecError()
        return _DecodedContinuationToken(
            payload=EncryptedConversationPayload(
                nonce=self._base64_decode(self._string(raw["nonce"])),
                ciphertext=self._base64_decode(
                    self._string(raw["ciphertext"])
                ),
                authenticated_digest=self._string(raw["authenticated_digest"]),
                associated_data_digest=self._string(
                    raw["associated_data_digest"]
                ),
                key_id=self._string(raw["key_id"]),
                key_revision=self._integer(raw["key_revision"]),
                algorithm=self._string(raw["algorithm"]),
            ),
            checkpoint_id=checkpoint_id,
            lane_id=lane_id,
            sequence=sequence,
        )

    def _associated_data(
        self,
        checkpoint: ConversationCheckpoint,
        scope_digest: AuthorityDigest,
        *,
        key_id: str,
        key_revision: int,
    ) -> ConversationPayloadAssociatedData:
        lanes = checkpoint.content.lanes
        if not lanes:
            raise ConversationValidationError()
        return ConversationPayloadAssociatedData(
            authority_digest=scope_digest,
            checkpoint_id=checkpoint.identity.checkpoint_id,
            lane_id=lanes[0].lane_id,
            sequence=int(checkpoint.identity.sequence),
            payload_kind=ConversationPayloadKind.CONTINUATION_REFERENCE,
            payload_schema_version=CONVERSATION_PAYLOAD_SCHEMA_VERSION,
            codec_version=CHECKPOINT_CODEC_VERSION,
            key_id=key_id,
            key_revision=key_revision,
        )

    @staticmethod
    def _lane_bindings(
        checkpoint: ConversationCheckpoint,
    ) -> list[dict[str, object]]:
        return [
            {
                "capability_revision": str(
                    lane.binding.capability_profile_revision
                ),
                "codec_version": int(lane.binding.continuation_codec_version),
                "configuration_revision": str(
                    lane.binding.model_configuration_revision
                ),
                "execution_revision": str(
                    lane.binding.execution_definition_revision
                ),
                "index": index,
                "lane_id": str(lane.lane_id),
                "provider_binding_digest": str(lane.binding.integrity_digest),
                "tool_revision": str(lane.binding.tool_schema_revision),
            }
            for index, lane in enumerate(checkpoint.content.lanes)
        ]

    @staticmethod
    def _validate_checkpoint(
        checkpoint: ConversationCheckpoint,
        authority: ContinuationEnvelopeAuthority,
    ) -> None:
        if (
            type(checkpoint) is not ConversationCheckpoint
            or checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
            or checkpoint.authority != authority.authority
            or not checkpoint.content.lanes
        ):
            raise ConversationAuthorizationError()

    def _decode_json(
        self,
        encoded: bytes,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, object]:
        selected_max_bytes = (
            self.limits.max_token_chars if max_bytes is None else max_bytes
        )
        if (
            type(encoded) is not bytes
            or type(selected_max_bytes) is not int
            or selected_max_bytes <= 0
            or not encoded
            or len(encoded) > selected_max_bytes
        ):
            raise ConversationLimitError()
        try:
            raw = loads(
                encoded.decode("utf-8"),
                object_pairs_hook=self._unique_object,
                parse_constant=self._reject_constant,
            )
            frozen = freeze_json_value(raw, limits=self.limits.json_limits)
            value = thaw_json_value(frozen)
        except (
            JSONDecodeError,
            OverflowError,
            RecursionError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            raise ConversationCodecError() from error
        if not isinstance(value, dict):
            raise ConversationCodecError()
        return value

    @staticmethod
    def _unique_object(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ConversationCodecError()
            value[key] = item
        return value

    @staticmethod
    def _reject_constant(value: str) -> object:
        del value
        raise ConversationCodecError()

    @staticmethod
    def _base64_encode(value: bytes) -> str:
        return urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _base64_decode(value: str) -> bytes:
        if type(value) is not str or not value or "=" in value:
            raise ConversationCodecError()
        padding = "=" * (-len(value) % 4)
        try:
            return b64decode(
                value + padding,
                altchars=b"-_",
                validate=True,
            )
        except (BinasciiError, ValueError) as error:
            raise ConversationCodecError() from error

    @staticmethod
    def _string(value: object) -> str:
        if type(value) is not str or not value:
            raise ConversationCodecError()
        return value

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if value is None:
            return None
        return ContinuationEnvelopeCodec._string(value)

    @staticmethod
    def _integer(value: object) -> int:
        if type(value) is not int:
            raise ConversationCodecError()
        return value

    @staticmethod
    def _optional_integer(value: object) -> int | None:
        if value is None:
            return None
        return ContinuationEnvelopeCodec._integer(value)

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationValidationError()
        return value.astimezone(UTC)
