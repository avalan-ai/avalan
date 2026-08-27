"""Retain bounded audience records without changing mutation truth.

The service is deliberately inert: callers provide a trusted durable store,
an AES-GCM cipher, and an asynchronous clock.  It neither activates a patch
surface nor changes a journal, result, approval, or outbox record.  Every
operational failure becomes a small independent warning rather than a rewrite
of the associated mutation outcome.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Never, NoReturn, Protocol, final

from avalan.patch.domain import (
    Audience,
    DurationTicks,
    ExpiryTick,
    PatchRetentionRecordId,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    DurableEncryptedRetention,
    DurableRetentionBinding,
)
from avalan.patch.durable_store import (
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionCleanup,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
)


class AudienceRetentionWarning(StrEnum):
    """Name bounded independent retention outcomes."""

    ENCRYPTION_FAILED = "retention_encryption_failed"
    WRITE_FAILED = "retention_write_failed"
    ACCESS_FAILED = "retention_access_failed"
    CLEANUP_FAILED = "retention_cleanup_failed"


class AudienceRetentionError(ValueError):
    """Report invalid retention service construction without content detail."""


class AudienceRetentionClock(Protocol):
    """Read a trusted monotonic retention clock asynchronously."""

    async def now(self) -> ExpiryTick:
        """Return the current monotonic retention tick."""


class AudienceRetentionStore(Protocol):
    """Expose the three retention operations needed by the inert service."""

    async def put_retention(
        self,
        reservation: DurableReservation,
        record: DurableRetentionRecord,
    ) -> None:
        """Persist one encrypted record."""

    async def get_retention_for_audience(
        self,
        access: DurableRetentionAccess,
        retention_id: PatchRetentionRecordId,
        kind: DurableRetentionKind,
        audience: Audience,
        now: ExpiryTick,
    ) -> DurableRetentionRecord:
        """Return one exact audience-kind encrypted record."""

    async def cleanup_retention(
        self, now: ExpiryTick
    ) -> DurableRetentionCleanup:
        """Clean expired encrypted records."""


class AudienceRetentionWriterAuthorizer(Protocol):
    """Authorize one exact audience retention purpose before encryption."""

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Return whether the authenticated request may retain this purpose."""


@dataclass(frozen=True, slots=True)
class AudienceRetentionPolicy:
    """Bound allowed retention purposes, TTL, and terminal cleanup behavior."""

    allowed_kinds: frozenset[DurableRetentionKind]
    ttl: DurationTicks
    delete_on_terminal: bool

    def __post_init__(self) -> None:
        """Require a bounded policy-owned audience retention contract."""
        if (
            type(self.allowed_kinds) is not frozenset
            or not self.allowed_kinds
            or any(
                type(item) is not DurableRetentionKind
                for item in self.allowed_kinds
            )
            or not self.allowed_kinds <= _PHASE_TWELVE_RETENTION_KINDS
            or type(self.ttl) is not DurationTicks
            or self.ttl.value > _MAX_RETENTION_TTL_TICKS
            or type(self.delete_on_terminal) is not bool
        ):
            raise AudienceRetentionError("retention policy is invalid")


@final
@dataclass(frozen=True, slots=True, repr=False, init=False, eq=False)
class AudienceRetentionWriter:
    """Carry one service-issued retention writer authority."""

    _issuer: object
    _store: AudienceRetentionStore
    _reservation: DurableReservation
    _kind: DurableRetentionKind
    _policy: AudienceRetentionPolicy

    def __init__(self, token: Never) -> None:
        """Reject public construction of a writer authority."""
        del token
        raise AudienceRetentionError("retention writer is service-issued")

    def __repr__(self) -> str:
        """Render a stable opaque writer marker."""
        return "AudienceRetentionWriter(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying an exact retention writer authority."""
        raise AudienceRetentionError("retention writer cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject copying an exact retention writer authority."""
        del memo
        raise AudienceRetentionError("retention writer cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing an exact retention writer authority."""
        raise AudienceRetentionError("retention writer cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific writer serialization."""
        del protocol
        raise AudienceRetentionError("retention writer cannot be serialized")


@final
@dataclass(frozen=True, slots=True, repr=False, init=False, eq=False)
class AudienceRetentionReadAuthority:
    """Carry replay-safe exact authority for one audience retention read."""

    _issuer: object
    _store: AudienceRetentionStore
    _reservation: DurableReservation
    _access: DurableRetentionAccess
    _kind: DurableRetentionKind
    _audience: Audience
    _policy: AudienceRetentionPolicy
    _consumed: bool

    def __init__(self, token: Never) -> None:
        """Reject public construction of a retention read authority."""
        del token
        raise AudienceRetentionError(
            "retention read authority is service-issued"
        )

    def __repr__(self) -> str:
        """Render a stable opaque read authority marker."""
        return "AudienceRetentionReadAuthority(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying a request-bound retention read authority."""
        raise AudienceRetentionError(
            "retention read authority cannot be copied"
        )

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject copying a request-bound retention read authority."""
        del memo
        raise AudienceRetentionError(
            "retention read authority cannot be copied"
        )

    def __reduce__(self) -> NoReturn:
        """Reject serializing a request-bound retention read authority."""
        raise AudienceRetentionError(
            "retention read authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific read-authority serialization."""
        del protocol
        raise AudienceRetentionError(
            "retention read authority cannot be serialized"
        )


class DenyAudienceRetentionWriterAuthorizer:
    """Deny every audience retention write until a host authorizes one."""

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Return a closed denial without selecting caller-controlled state."""
        if (
            type(reservation) is not DurableReservation
            or type(kind) is not DurableRetentionKind
        ):
            raise AudienceRetentionError("retention authorization is invalid")
        return False


@dataclass(frozen=True, slots=True, repr=False)
class AudienceRetainedValue:
    """Carry authorized plaintext without exposing it through rendering."""

    _value: bytes

    def __post_init__(self) -> None:
        """Require one immutable bounded plaintext result."""
        if type(self._value) is not bytes or len(self._value) > 1_048_576:
            raise AudienceRetentionError("retained value is invalid")

    def __repr__(self) -> str:
        """Render a stable value-free marker."""
        return "AudienceRetainedValue(<redacted>)"

    def __str__(self) -> str:
        """Render a stable value-free marker."""
        return "<redacted>"

    def read(self) -> bytes:
        """Return plaintext only to the authenticated caller of open."""
        return self._value

    def __copy__(self) -> NoReturn:
        """Reject copies that could widen private-value lifetime."""
        raise AudienceRetentionError("retained value cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could widen private-value lifetime."""
        del memo
        raise AudienceRetentionError("retained value cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing plaintext values."""
        raise AudienceRetentionError("retained value cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific plaintext serialization."""
        del protocol
        raise AudienceRetentionError("retained value cannot be serialized")


@dataclass(frozen=True, slots=True)
class AudienceRetentionWriteReceipt:
    """Report a non-throwing retention attempt without plaintext detail."""

    retention_id: PatchRetentionRecordId | None
    warning: AudienceRetentionWarning | None

    def __post_init__(self) -> None:
        """Require a retention identifier only for successful persistence."""
        if (
            self.retention_id is not None
            and type(self.retention_id) is not PatchRetentionRecordId
        ) or (
            self.warning is not None
            and type(self.warning) is not AudienceRetentionWarning
        ):
            raise AudienceRetentionError("retention write receipt is invalid")
        if (self.retention_id is None) is (self.warning is None):
            raise AudienceRetentionError("retention write receipt is invalid")


@dataclass(frozen=True, slots=True)
class AudienceRetentionReadReceipt:
    """Report authorized plaintext availability without leaking failures."""

    value: AudienceRetainedValue | None
    warning: AudienceRetentionWarning | None

    def __post_init__(self) -> None:
        """Require an exact value-or-warning read result."""
        if (
            self.value is not None
            and type(self.value) is not AudienceRetainedValue
        ) or (
            self.warning is not None
            and type(self.warning) is not AudienceRetentionWarning
        ):
            raise AudienceRetentionError("retention read receipt is invalid")
        if (self.value is None) is (self.warning is None):
            raise AudienceRetentionError("retention read receipt is invalid")


@dataclass(frozen=True, slots=True)
class AudienceRetentionCleanupReceipt:
    """Report cleanup success without content-derived counts or sizes."""

    completed: bool
    warning: AudienceRetentionWarning | None

    def __post_init__(self) -> None:
        """Require one closed cleanup completion result."""
        if (
            type(self.completed) is not bool
            or (
                self.warning is not None
                and type(self.warning) is not AudienceRetentionWarning
            )
            or self.completed is (self.warning is not None)
        ):
            raise AudienceRetentionError(
                "retention cleanup receipt is invalid"
            )


@final
class AudienceRetentionService:
    """Seal, access, and clean audience retention."""

    __slots__ = (
        "_store",
        "_cipher",
        "_clock",
        "_policy",
        "_authorizer",
        "_issuer",
    )

    def __init__(
        self,
        store: AudienceRetentionStore,
        cipher: AesGcmDurableRetentionCipher,
        clock: AudienceRetentionClock,
        policy: AudienceRetentionPolicy,
        authorizer: AudienceRetentionWriterAuthorizer = (
            DenyAudienceRetentionWriterAuthorizer()
        ),
    ) -> None:
        """Bind async retention dependencies without I/O or key resolution."""
        if (
            not callable(getattr(store, "put_retention", None))
            or not callable(getattr(store, "get_retention_for_audience", None))
            or not callable(getattr(store, "cleanup_retention", None))
            or type(cipher) is not AesGcmDurableRetentionCipher
            or not callable(getattr(clock, "now", None))
            or type(policy) is not AudienceRetentionPolicy
            or not callable(getattr(authorizer, "authorize", None))
        ):
            raise AudienceRetentionError("retention service is invalid")
        self._store = store
        self._cipher = cipher
        self._clock = clock
        self._policy = policy
        self._authorizer = authorizer
        self._issuer = object()

    async def issue_writer(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> AudienceRetentionWriter | None:
        """Authorize and issue one exact writer for a policy-owned purpose."""
        if (
            type(reservation) is not DurableReservation
            or type(kind) is not DurableRetentionKind
            or kind not in self._policy.allowed_kinds
        ):
            return None
        try:
            allowed = await self._authorizer.authorize(reservation, kind)
        except Exception:
            return None
        if type(allowed) is not bool or not allowed:
            return None
        return _new_writer(
            self._issuer,
            self._store,
            reservation,
            kind,
            self._policy,
        )

    async def issue_read_authority(
        self,
        reservation: DurableReservation,
        access: DurableRetentionAccess,
        kind: DurableRetentionKind,
    ) -> AudienceRetentionReadAuthority | None:
        """Issue replay-safe authority for one exact retained audience kind."""
        if (
            type(reservation) is not DurableReservation
            or type(access) is not DurableRetentionAccess
            or type(kind) is not DurableRetentionKind
            or kind not in self._policy.allowed_kinds
            or access.request.request_id != reservation.request_id
            or access.request.identity != reservation.identity
        ):
            return None
        try:
            allowed = await self._authorizer.authorize(reservation, kind)
        except Exception:
            return None
        if type(allowed) is not bool or not allowed:
            return None
        return _new_read_authority(
            self._issuer,
            self._store,
            reservation,
            access,
            kind,
            _retention_audience(kind),
            self._policy,
        )

    async def retain(
        self,
        writer: AudienceRetentionWriter,
        value: AudienceRetainedValue,
    ) -> AudienceRetentionWriteReceipt:
        """Encrypt and retain an authority-bound audience artifact."""
        if (
            type(writer) is not AudienceRetentionWriter
            or type(value) is not AudienceRetainedValue
            or writer._issuer is not self._issuer
            or writer._store is not self._store
            or writer._policy is not self._policy
            or writer._kind not in _PHASE_TWELVE_RETENTION_KINDS
            or writer._kind not in self._policy.allowed_kinds
        ):
            return AudienceRetentionWriteReceipt(
                None, AudienceRetentionWarning.WRITE_FAILED
            )
        try:
            allowed = await self._authorizer.authorize(
                writer._reservation, writer._kind
            )
            now = await self._clock.now()
            if type(allowed) is not bool or not allowed:
                raise AudienceRetentionError(
                    "retention authorization is denied"
                )
            if type(now) is not ExpiryTick:
                raise AudienceRetentionError("retention clock is invalid")
            expires_at = ExpiryTick(now.value + self._policy.ttl.value)
        except Exception:
            return AudienceRetentionWriteReceipt(
                None, AudienceRetentionWarning.WRITE_FAILED
            )
        retention_id = PatchRetentionRecordId.new()
        binding = DurableRetentionBinding(
            writer._reservation.request_id, retention_id, writer._kind
        )
        try:
            encrypted = await self._cipher.seal(value.read(), binding)
        except Exception:
            return AudienceRetentionWriteReceipt(
                None, AudienceRetentionWarning.ENCRYPTION_FAILED
            )
        try:
            await self._store.put_retention(
                writer._reservation,
                DurableRetentionRecord(
                    retention_id,
                    writer._kind,
                    encrypted.key_id,
                    encrypted.value,
                    DurableRetentionPolicy(
                        expires_at,
                        self._policy.delete_on_terminal,
                    ),
                ),
            )
        except Exception:
            return AudienceRetentionWriteReceipt(
                None, AudienceRetentionWarning.WRITE_FAILED
            )
        return AudienceRetentionWriteReceipt(retention_id, None)

    async def open(
        self,
        authority: AudienceRetentionReadAuthority,
        retention_id: PatchRetentionRecordId,
    ) -> AudienceRetentionReadReceipt:
        """Open one authorized audience artifact under the trusted clock."""
        try:
            valid_authority = (
                type(authority) is AudienceRetentionReadAuthority
                and type(retention_id) is PatchRetentionRecordId
                and authority._issuer is self._issuer
                and authority._store is self._store
                and authority._policy is self._policy
                and authority._kind in _PHASE_TWELVE_RETENTION_KINDS
                and authority._kind in self._policy.allowed_kinds
                and authority._access.request.request_id
                == authority._reservation.request_id
                and authority._access.request.identity
                == authority._reservation.identity
                and authority._audience is _retention_audience(authority._kind)
                and type(authority._consumed) is bool
                and not authority._consumed
            )
            if valid_authority:
                # This transition has no await before it, so it is atomic for
                # concurrent callers on the event loop.  It remains claimed
                # after cancellation or any later dependency failure.
                object.__setattr__(authority, "_consumed", True)
        except Exception:
            valid_authority = False
        if not valid_authority:
            return AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        try:
            allowed = await self._authorizer.authorize(
                authority._reservation, authority._kind
            )
            now = await self._clock.now()
            if type(allowed) is not bool or not allowed:
                raise AudienceRetentionError(
                    "retention authorization is denied"
                )
            if type(now) is not ExpiryTick:
                raise AudienceRetentionError("retention clock is invalid")
            record = await self._store.get_retention_for_audience(
                authority._access,
                retention_id,
                authority._kind,
                authority._audience,
                now,
            )
            if record.kind is not authority._kind:
                raise AudienceRetentionError("retention kind is invalid")
            plaintext = await self._cipher.open(
                DurableEncryptedRetention(record.key_id, record.value),
                DurableRetentionBinding(
                    authority._reservation.request_id,
                    record.retention_id,
                    record.kind,
                ),
            )
        except Exception:
            return AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        return AudienceRetentionReadReceipt(
            AudienceRetainedValue(plaintext), None
        )

    async def cleanup(self) -> AudienceRetentionCleanupReceipt:
        """Run bounded expiry cleanup with the trusted clock."""
        try:
            now = await self._clock.now()
            if type(now) is not ExpiryTick:
                raise AudienceRetentionError("retention clock is invalid")
            cleanup = await self._store.cleanup_retention(now)
            if type(cleanup) is not DurableRetentionCleanup:
                raise AudienceRetentionError("retention cleanup is invalid")
        except Exception:
            return AudienceRetentionCleanupReceipt(
                False, AudienceRetentionWarning.CLEANUP_FAILED
            )
        return AudienceRetentionCleanupReceipt(True, None)


def _new_writer(
    issuer: object,
    store: AudienceRetentionStore,
    reservation: DurableReservation,
    kind: DurableRetentionKind,
    policy: AudienceRetentionPolicy,
) -> AudienceRetentionWriter:
    """Create one module-private exact writer after host authorization."""
    value = object.__new__(AudienceRetentionWriter)
    object.__setattr__(value, "_issuer", issuer)
    object.__setattr__(value, "_store", store)
    object.__setattr__(value, "_reservation", reservation)
    object.__setattr__(value, "_kind", kind)
    object.__setattr__(value, "_policy", policy)
    return value


def _new_read_authority(
    issuer: object,
    store: AudienceRetentionStore,
    reservation: DurableReservation,
    access: DurableRetentionAccess,
    kind: DurableRetentionKind,
    audience: Audience,
    policy: AudienceRetentionPolicy,
) -> AudienceRetentionReadAuthority:
    """Create module-private replay-safe exact retention read authority."""
    value = object.__new__(AudienceRetentionReadAuthority)
    object.__setattr__(value, "_issuer", issuer)
    object.__setattr__(value, "_store", store)
    object.__setattr__(value, "_reservation", reservation)
    object.__setattr__(value, "_access", access)
    object.__setattr__(value, "_kind", kind)
    object.__setattr__(value, "_audience", audience)
    object.__setattr__(value, "_policy", policy)
    object.__setattr__(value, "_consumed", False)
    return value


def _retention_audience(kind: DurableRetentionKind) -> Audience:
    """Return the sole durable audience authorized for one Phase 12 kind."""
    audiences = {
        DurableRetentionKind.CLI_REVIEW: Audience.OPERATOR,
        DurableRetentionKind.AUDIT_PROJECTION: Audience.AUDIT,
        DurableRetentionKind.METRICS_PROJECTION: Audience.PUBLIC,
        DurableRetentionKind.TELEMETRY_PROJECTION: Audience.PUBLIC,
        DurableRetentionKind.SERVER_READY_PROJECTION: Audience.PUBLIC,
        DurableRetentionKind.DIAGNOSTIC_ASSOCIATION: Audience.OPERATOR,
    }
    try:
        return audiences[kind]
    except KeyError as error:
        raise AudienceRetentionError("retention kind is invalid") from error


_MAX_RETENTION_TTL_TICKS = 86_400
_PHASE_TWELVE_RETENTION_KINDS = frozenset(
    (
        DurableRetentionKind.CLI_REVIEW,
        DurableRetentionKind.AUDIT_PROJECTION,
        DurableRetentionKind.METRICS_PROJECTION,
        DurableRetentionKind.TELEMETRY_PROJECTION,
        DurableRetentionKind.SERVER_READY_PROJECTION,
        DurableRetentionKind.DIAGNOSTIC_ASSOCIATION,
    )
)
