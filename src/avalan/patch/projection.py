"""Construct detached patch projection envelopes at a trusted boundary.

Only trusted host code receives these audience-specific boundaries and their
authority witnesses.  Each boundary derives primitive, digest-bound envelope
data from a sealed source, then releases the source.  Ordinary model, display,
audit, and approver consumers receive only detached canonical ``bytes``
deliveries.  The trusted host adapter verifies receipts and envelopes within
this module; it does not dynamically invoke the data-only ``projection_codec``
module.  Output values have no method, registry, closure, or module path back
to this trusted module.

An arbitrary process participant able to import and introspect trusted modules
is itself within the repository's trusted-host boundary.  The lower-consumer
threat model starts after delivery of a detached envelope and consumer codec.
"""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from json import dumps, loads
from secrets import token_urlsafe
from types import MappingProxyType
from typing import Mapping, NoReturn, TypeAlias, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from avalan.patch.domain import (
    Audience,
    ByteSize,
    PatchPublicCorrelationId,
    PatchResult,
    coarsen_error_code,
)
from avalan.patch.durable_store import DurableTerminalRecord
from avalan.patch.policy import (
    PolicyDisclosure,
    SealedPlan,
    _validate_sealed_plan,
)
from avalan.patch.projection_codec import (
    MAX_PROJECTION_ENVELOPE_BYTES,
    MAX_PROJECTION_RECEIPT_BYTES,
    PROJECTION_DELIVERY_SCHEMA_VERSION,
    PROJECTION_ENVELOPE_SCHEMA_VERSION,
    PROJECTION_RECEIPT_SCHEMA_VERSION,
    ApproverProjectionDelivery,
    ApproverProjectionEnvelope,
    AuditProjectionDelivery,
    AuditProjectionEnvelope,
    ModelProjectionDelivery,
    ModelProjectionEnvelope,
)

_MAX_OUTPUT_BYTES = MAX_PROJECTION_ENVELOPE_BYTES
_ED25519_PUBLIC_KEY_BYTES = 32
_ED25519_SIGNATURE_BYTES = 64

_ROOT_SIGNER = Ed25519PrivateKey.generate()
_HOST_ROOT_PUBLIC_KEY = _ROOT_SIGNER.public_key().public_bytes(
    Encoding.Raw, PublicFormat.Raw
)

ProjectionPayloadValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | tuple["ProjectionPayloadValue", ...]
    | Mapping[str, "ProjectionPayloadValue"]
)
ProjectionPayload: TypeAlias = Mapping[str, ProjectionPayloadValue]
_DecodedJsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["_DecodedJsonValue"]
    | dict[str, "_DecodedJsonValue"]
)
_DecodedJsonObject: TypeAlias = dict[str, _DecodedJsonValue]
_ReceiptFacts: TypeAlias = tuple[bytes, str, str, str, str, str]


class ProjectionError(ValueError):
    """Report an invalid or unauthorized trusted projection artifact."""


@dataclass(frozen=True, slots=True)
class ProjectionOutputLimit:
    """Bound one model envelope's returned diff independently of review."""

    value: ByteSize

    def __post_init__(self) -> None:
        """Require a finite nonzero public-output cap."""
        if (
            type(self.value) is not ByteSize
            or not 1 <= self.value.value <= _MAX_OUTPUT_BYTES
        ):
            raise ProjectionError("projection output limit is invalid")


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class PatchProjectionSource:
    """Bind one sealed plan and matching terminal truth before derivation."""

    plan: SealedPlan
    terminal: DurableTerminalRecord

    @property
    def result(self) -> PatchResult:
        """Return the canonical result held by the trusted terminal record."""
        return self.terminal.result

    def __post_init__(self) -> None:
        """Reject unsealed plans and result substitution across requests."""
        if (
            type(self.plan) is not SealedPlan
            or type(self.terminal) is not DurableTerminalRecord
        ):
            raise ProjectionError("projection source is invalid")
        _validate_sealed_plan(self.plan)
        if (
            self.plan.plan_id != self.terminal.result.plan_id
            or self.plan.binding.request.request_id
            != self.terminal.result.request_id
        ):
            raise ProjectionError(
                "projection source result does not match plan"
            )

    def __repr__(self) -> str:
        """Render a marker that cannot expose sealed canonical content."""
        return "PatchProjectionSource(<redacted>)"

    def __copy__(self) -> NoReturn:
        """Reject copies that could duplicate canonical privileged truth."""
        raise ProjectionError("projection source cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep copies that could duplicate privileged truth."""
        del memo
        raise ProjectionError("projection source cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a source that retains privileged plan content."""
        raise ProjectionError("projection source cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific source serialization."""
        del protocol
        raise ProjectionError("projection source cannot be serialized")


@dataclass(frozen=True, slots=True)
class _TerminalPayload:
    """Store only detached terminal fields permitted in a public envelope."""

    source_digest: str
    terminal_digest: str
    values: ProjectionPayload


@dataclass(frozen=True, slots=True, repr=False)
class _HostProjectionArtifact:
    """Bind one trusted boundary's signed envelope and receipt for delivery."""

    audience: str
    envelope: bytes
    receipt: bytes


@dataclass(frozen=True, slots=True, repr=False)
class _HostVerifiedProjection:
    """Keep one host-authenticated envelope audit and payload private."""

    audience: str
    correlation_id: str
    source_digest: str
    terminal_digest: str
    issuer_id: str
    payload: ProjectionPayload
    envelope: bytes
    receipt: bytes


@dataclass(frozen=True, slots=True, repr=False)
class _ProjectionHostAdapter:
    """Verify signed artifacts with the host-owned root before delivery."""

    _root_public_key: bytes

    def deliver(
        self,
        artifact: _HostProjectionArtifact,
    ) -> _HostVerifiedProjection:
        """Return one host-verified projection from an internal artifact."""
        if type(artifact) is not _HostProjectionArtifact:
            raise ProjectionError("projection host artifact is invalid")
        return _verify_host_artifact(artifact, self._root_public_key)


_HOST_PROJECTION_ADAPTER = _ProjectionHostAdapter(_HOST_ROOT_PUBLIC_KEY)


def _verify_host_artifact(
    artifact: _HostProjectionArtifact,
    root_public_key: bytes,
) -> _HostVerifiedProjection:
    """Verify one internal receipt and envelope before host delivery."""
    receipt = _parse_host_object(
        artifact.receipt,
        MAX_PROJECTION_RECEIPT_BYTES,
        "verification receipt",
    )
    if set(receipt) != {
        "schema_version",
        "audience",
        "correlation_id",
        "source_digest",
        "terminal_digest",
        "issuer_id",
        "public_key",
        "signature",
    }:
        raise ProjectionError(
            "projection verification receipt schema is invalid"
        )
    receipt_audience = _host_text(receipt["audience"], "verification receipt")
    correlation_id = _host_text(
        receipt["correlation_id"], "verification receipt"
    )
    source_digest = _host_text(
        receipt["source_digest"], "verification receipt"
    )
    terminal_digest = _host_text(
        receipt["terminal_digest"], "verification receipt"
    )
    issuer_id = _host_text(receipt["issuer_id"], "verification receipt")
    public_key_hex = _host_text(receipt["public_key"], "verification receipt")
    receipt_signature = _host_text(
        receipt["signature"], "verification receipt"
    )
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != PROJECTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ProjectionError(
            "projection verification receipt schema is invalid"
        )
    _host_validate_header(
        receipt_audience,
        correlation_id,
        source_digest,
        terminal_digest,
        issuer_id,
        "verification receipt",
    )
    if receipt_audience != artifact.audience:
        raise ProjectionError("projection verification receipt is invalid")
    try:
        public_key = bytes.fromhex(public_key_hex)
        if (
            len(public_key) != _ED25519_PUBLIC_KEY_BYTES
            or public_key_hex != public_key.hex()
        ):
            raise ValueError("projection receipt key is not canonical")
        Ed25519PublicKey.from_public_bytes(public_key)
    except ValueError as error:
        raise ProjectionError(
            "projection verification receipt key is invalid"
        ) from error
    _host_verify_signature(
        root_public_key,
        receipt_signature,
        {
            "schema_version": PROJECTION_RECEIPT_SCHEMA_VERSION,
            "audience": receipt_audience,
            "correlation_id": correlation_id,
            "source_digest": source_digest,
            "terminal_digest": terminal_digest,
            "issuer_id": issuer_id,
            "public_key": public_key_hex,
        },
        "verification receipt",
    )
    envelope = _parse_host_object(
        artifact.envelope,
        MAX_PROJECTION_ENVELOPE_BYTES,
        "envelope",
    )
    if set(envelope) != {
        "schema_version",
        "audience",
        "correlation_id",
        "source_digest",
        "terminal_digest",
        "issuer_id",
        "payload",
        "signature",
    }:
        raise ProjectionError("projection envelope schema is invalid")
    audience = _host_text(envelope["audience"], "envelope")
    envelope_correlation_id = _host_text(
        envelope["correlation_id"], "envelope"
    )
    envelope_source_digest = _host_text(envelope["source_digest"], "envelope")
    envelope_terminal_digest = _host_text(
        envelope["terminal_digest"], "envelope"
    )
    envelope_issuer_id = _host_text(envelope["issuer_id"], "envelope")
    envelope_signature = _host_text(envelope["signature"], "envelope")
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != PROJECTION_ENVELOPE_SCHEMA_VERSION
    ):
        raise ProjectionError("projection envelope schema is invalid")
    _host_validate_header(
        audience,
        envelope_correlation_id,
        envelope_source_digest,
        envelope_terminal_digest,
        envelope_issuer_id,
        "envelope",
    )
    if audience != artifact.audience or (
        envelope_correlation_id,
        envelope_source_digest,
        envelope_terminal_digest,
        envelope_issuer_id,
    ) != (correlation_id, source_digest, terminal_digest, issuer_id):
        raise ProjectionError("projection verification receipt is invalid")
    if not isinstance(envelope["payload"], dict):
        raise ProjectionError("projection payload is invalid")
    _host_verify_signature(
        public_key,
        envelope_signature,
        {
            "schema_version": PROJECTION_ENVELOPE_SCHEMA_VERSION,
            "audience": audience,
            "correlation_id": envelope_correlation_id,
            "source_digest": envelope_source_digest,
            "terminal_digest": envelope_terminal_digest,
            "issuer_id": envelope_issuer_id,
            "payload": envelope["payload"],
        },
        "envelope",
    )
    return _HostVerifiedProjection(
        audience,
        correlation_id,
        source_digest,
        terminal_digest,
        issuer_id,
        _host_freeze_payload(envelope["payload"]),
        artifact.envelope,
        artifact.receipt,
    )


def _parse_host_object(
    value: bytes,
    maximum_size: int,
    label: str,
) -> _DecodedJsonObject:
    """Return exactly one canonical host-verified JSON object."""
    if type(value) is not bytes or len(value) > maximum_size:
        raise ProjectionError(f"projection {label} is invalid")
    try:
        decoded = loads(
            value.decode("utf-8"),
            object_pairs_hook=_host_object_without_duplicates,
            parse_constant=_host_reject_nonfinite_number,
        )
        if not isinstance(
            decoded, dict
        ) or value != _canonical_projection_bytes(decoded):
            raise ProjectionError(f"projection {label} is not canonical")
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise ProjectionError(f"projection {label} is invalid") from error
    return decoded


def _host_object_without_duplicates(
    pairs: list[tuple[str, _DecodedJsonValue]],
) -> _DecodedJsonObject:
    """Return one host JSON object while rejecting duplicate keys."""
    result: _DecodedJsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("projection JSON object has duplicate keys")
        result[key] = _host_json_value(value)
    return result


def _host_json_value(value: _DecodedJsonValue) -> _DecodedJsonValue:
    """Return one recursively checked host JSON primitive value."""
    if isinstance(value, list):
        return [_host_json_value(item) for item in value]
    if isinstance(value, dict):
        return _host_json_object(value)
    return value


def _host_json_object(
    value: dict[str, _DecodedJsonValue],
) -> _DecodedJsonObject:
    """Return one string-keyed host JSON object without widening values."""
    return {key: _host_json_value(item) for key, item in value.items()}


def _host_reject_nonfinite_number(value: str) -> NoReturn:
    """Reject non-finite JSON extensions before host canonicalization."""
    del value
    raise ValueError("projection JSON number is not finite")


def _host_text(value: object, label: str) -> str:
    """Return one primitive JSON string or reject one untyped host field."""
    if type(value) is not str:
        raise ProjectionError(f"projection {label} schema is invalid")
    return value


def _host_validate_header(
    audience: str,
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
    label: str,
) -> None:
    """Require bounded host audience, correlation, digest, and issuer facts."""
    if (
        audience not in {"model", "approver", "audit"}
        or not correlation_id.startswith("public_")
        or len(correlation_id) > 128
        or not _is_digest(source_digest)
        or not _is_digest(terminal_digest)
        or not _is_issuer_id(issuer_id)
    ):
        raise ProjectionError(f"projection {label} header is invalid")


def _host_verify_signature(
    public_key: bytes,
    signature: str,
    body: object,
    label: str,
) -> None:
    """Require one exact canonical Ed25519 host signature."""
    try:
        signature_bytes = bytes.fromhex(signature)
        if (
            len(signature_bytes) != _ED25519_SIGNATURE_BYTES
            or signature != signature_bytes.hex()
        ):
            raise ValueError("projection signature is not canonical")
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature_bytes, _canonical_projection_bytes(body)
        )
    except (InvalidSignature, ValueError) as error:
        raise ProjectionError(
            f"projection {label} signature is invalid"
        ) from error


def _host_freeze_payload(
    value: Mapping[str, _DecodedJsonValue],
) -> ProjectionPayload:
    """Return one recursively frozen host-verified primitive payload."""
    payload: dict[str, ProjectionPayloadValue] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise ProjectionError("projection payload is invalid")
        payload[key] = _host_freeze_value(item)
    return MappingProxyType(payload)


def _host_freeze_value(value: object) -> ProjectionPayloadValue:
    """Return one frozen primitive value from a host-verified payload."""
    if value is None or type(value) in {bool, int, float, str}:
        assert value is None or isinstance(value, (bool, int, float, str))
        return value
    if isinstance(value, list):
        return tuple(_host_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return _host_freeze_payload(value)
    raise ProjectionError("projection payload is invalid")


def _host_delivery_bytes(verified: _HostVerifiedProjection) -> bytes:
    """Return detached lower-consumer bytes without internal integrity facts.

    The source and terminal digests bind the root-signed receipt, the
    audience-signed envelope, and any trusted local AAD.  They are not a
    disclosure field: both are content-derived and would otherwise make a
    lower delivery into a source or precondition oracle.
    """
    return _canonical_projection_bytes(
        {
            "schema_version": PROJECTION_DELIVERY_SCHEMA_VERSION,
            "audience": verified.audience,
            "correlation_id": verified.correlation_id,
            "payload": _plain_projection_payload(verified.payload),
        }
    )


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _AudienceAuthority:
    """Bind a trusted boundary's one exact public correlation witness."""

    _issuer: object
    correlation_id: PatchPublicCorrelationId
    source_digest: str
    terminal_digest: str

    def __post_init__(self) -> None:
        """Require opaque correlation and digest facts at trusted setup."""
        if (
            type(self.correlation_id) is not PatchPublicCorrelationId
            or not _is_digest(self.source_digest)
            or not _is_digest(self.terminal_digest)
        ):
            raise ProjectionError("projection authority is invalid")

    def __repr__(self) -> str:
        """Render a source-free authority marker for trusted diagnostics."""
        return f"{type(self).__name__}(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying an exact identity-bound witness."""
        raise ProjectionError("projection authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep-copying an exact identity-bound witness."""
        del memo
        raise ProjectionError("projection authority cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject pickling an exact identity-bound witness."""
        raise ProjectionError("projection authority cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific authority serialization."""
        del protocol
        raise ProjectionError("projection authority cannot be serialized")


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ModelProjectionAuthority(_AudienceAuthority):
    """Authorize only one model envelope from its trusted boundary."""


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ApproverProjectionAuthority(_AudienceAuthority):
    """Authorize only one approver envelope from its trusted boundary."""


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AuditProjectionAuthority(_AudienceAuthority):
    """Authorize only one audit envelope from its trusted boundary."""


class _ProjectionBoundary:
    """Provide common trusted-boundary copy and serialization fences."""

    def __copy__(self) -> NoReturn:
        """Reject copying a trusted construction boundary."""
        raise ProjectionError("projection boundary cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject deep-copying a trusted construction boundary."""
        del memo
        raise ProjectionError("projection boundary cannot be copied")

    def __reduce__(self) -> NoReturn:
        """Reject serializing a trusted construction boundary."""
        raise ProjectionError("projection boundary cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific boundary serialization."""
        del protocol
        raise ProjectionError("projection boundary cannot be serialized")


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ModelProjectionBoundary(_ProjectionBoundary):
    """Construct detached model envelopes from model-safe primitive data."""

    _issuer: object
    _issuer_id: str
    _signing_key: Ed25519PrivateKey
    _authority: ModelProjectionAuthority
    _correlation_id: PatchPublicCorrelationId
    _terminal: _TerminalPayload
    _diff_prefix: bytes | None
    _diff_total_bytes: int
    _metadata_allowed: bool

    def authority(self) -> ModelProjectionAuthority:
        """Return this boundary's exact model-only authority witness."""
        return self._authority

    def project(
        self,
        authority: ModelProjectionAuthority,
        output_limit: ProjectionOutputLimit,
    ) -> ModelProjectionDelivery:
        """Return host-verified detached model delivery bytes."""
        if type(output_limit) is not ProjectionOutputLimit:
            raise ProjectionError("projection output limit is invalid")
        self._require(authority)
        payload = dict(self._terminal.values)
        payload["diff"] = _model_diff_payload(self, output_limit)
        envelope = _encode_model_envelope(
            authority.correlation_id.value,
            self._terminal.source_digest,
            self._terminal.terminal_digest,
            self._issuer_id,
            self._signing_key,
            payload,
        )
        verified = _HOST_PROJECTION_ADAPTER.deliver(
            _HostProjectionArtifact(
                "model",
                envelope,
                _verification_receipt(
                    self._signing_key,
                    "model",
                    self._correlation_id.value,
                    self._terminal.source_digest,
                    self._terminal.terminal_digest,
                    self._issuer_id,
                ),
            )
        )
        return ModelProjectionDelivery(_host_delivery_bytes(verified))

    def _require(self, authority: ModelProjectionAuthority) -> None:
        """Require this exact unmodified model authority instance."""
        if (
            type(authority) is not ModelProjectionAuthority
            or authority is not self._authority
            or authority._issuer is not self._issuer
            or authority.correlation_id is not self._correlation_id
            or authority.source_digest != self._terminal.source_digest
            or authority.terminal_digest != self._terminal.terminal_digest
        ):
            raise ProjectionError("projection authority is not issued here")


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class ApproverProjectionBoundary(_ProjectionBoundary):
    """Construct detached complete-review envelopes for approvers only."""

    _issuer: object
    _issuer_id: str
    _signing_key: Ed25519PrivateKey
    _authority: ApproverProjectionAuthority
    _correlation_id: PatchPublicCorrelationId
    _terminal: _TerminalPayload
    _review: ProjectionPayload | None

    def authority(self) -> ApproverProjectionAuthority:
        """Return this boundary's exact approver-only authority witness."""
        return self._authority

    def project(
        self,
        authority: ApproverProjectionAuthority,
    ) -> ApproverProjectionDelivery:
        """Return host-verified detached approver delivery bytes."""
        self._require(authority)
        if self._review is None:
            raise ProjectionError("complete review disclosure is unavailable")
        payload = dict(self._terminal.values)
        payload["diff"] = _redacted_diff_payload()
        payload["review"] = self._review
        envelope = _encode_approver_envelope(
            authority.correlation_id.value,
            self._terminal.source_digest,
            self._terminal.terminal_digest,
            self._issuer_id,
            self._signing_key,
            payload,
        )
        verified = _HOST_PROJECTION_ADAPTER.deliver(
            _HostProjectionArtifact(
                "approver",
                envelope,
                _verification_receipt(
                    self._signing_key,
                    "approver",
                    self._correlation_id.value,
                    self._terminal.source_digest,
                    self._terminal.terminal_digest,
                    self._issuer_id,
                ),
            )
        )
        return ApproverProjectionDelivery(_host_delivery_bytes(verified))

    def _require(self, authority: ApproverProjectionAuthority) -> None:
        """Require this exact unmodified approver authority instance."""
        if (
            type(authority) is not ApproverProjectionAuthority
            or authority is not self._authority
            or authority._issuer is not self._issuer
            or authority.correlation_id is not self._correlation_id
            or authority.source_digest != self._terminal.source_digest
            or authority.terminal_digest != self._terminal.terminal_digest
        ):
            raise ProjectionError("projection authority is not issued here")


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AuditProjectionBoundary(_ProjectionBoundary):
    """Construct detached content-free audit envelopes only."""

    _issuer: object
    _issuer_id: str
    _signing_key: Ed25519PrivateKey
    _authority: AuditProjectionAuthority
    _correlation_id: PatchPublicCorrelationId
    _terminal: _TerminalPayload

    def authority(self) -> AuditProjectionAuthority:
        """Return this boundary's exact audit-only authority witness."""
        return self._authority

    def project(
        self,
        authority: AuditProjectionAuthority,
    ) -> AuditProjectionDelivery:
        """Return host-verified detached audit delivery bytes."""
        self._require(authority)
        payload = dict(self._terminal.values)
        payload["diff"] = _redacted_diff_payload()
        envelope = _encode_audit_envelope(
            authority.correlation_id.value,
            self._terminal.source_digest,
            self._terminal.terminal_digest,
            self._issuer_id,
            self._signing_key,
            payload,
        )
        verified = _HOST_PROJECTION_ADAPTER.deliver(
            _HostProjectionArtifact(
                "audit",
                envelope,
                _verification_receipt(
                    self._signing_key,
                    "audit",
                    self._correlation_id.value,
                    self._terminal.source_digest,
                    self._terminal.terminal_digest,
                    self._issuer_id,
                ),
            )
        )
        return AuditProjectionDelivery(_host_delivery_bytes(verified))

    def _require(self, authority: AuditProjectionAuthority) -> None:
        """Require this exact unmodified audit authority instance."""
        if (
            type(authority) is not AuditProjectionAuthority
            or authority is not self._authority
            or authority._issuer is not self._issuer
            or authority.correlation_id is not self._correlation_id
            or authority.source_digest != self._terminal.source_digest
            or authority.terminal_digest != self._terminal.terminal_digest
        ):
            raise ProjectionError("projection authority is not issued here")


def create_model_projection_boundary(
    source: PatchProjectionSource,
) -> ModelProjectionBoundary:
    """Derive a trusted model boundary and release canonical source truth."""
    terminal = _terminal_payload(source, Audience.MODEL)
    disclosures = source.plan.binding.final.disclosures
    if PolicyDisclosure.MODEL_DIFF in disclosures:
        diff = source.plan.review.diff.diff._value
        diff_prefix = _utf8_prefix(diff, _MAX_OUTPUT_BYTES)
        diff_total_bytes = len(diff)
    else:
        diff_prefix = None
        diff_total_bytes = 0
    issuer = object()
    issuer_id = _issuer_id()
    signing_key = Ed25519PrivateKey.generate()
    correlation_id = PatchPublicCorrelationId.new()
    authority = ModelProjectionAuthority(
        issuer,
        correlation_id,
        terminal.source_digest,
        terminal.terminal_digest,
    )
    return ModelProjectionBoundary(
        issuer,
        issuer_id,
        signing_key,
        authority,
        correlation_id,
        terminal,
        diff_prefix,
        diff_total_bytes,
        PolicyDisclosure.MODEL_METADATA in disclosures,
    )


def create_approver_projection_boundary(
    source: PatchProjectionSource,
) -> ApproverProjectionBoundary:
    """Derive a trusted approver boundary and release canonical source."""
    terminal = _terminal_payload(source, Audience.APPROVER)
    if (
        PolicyDisclosure.COMPLETE_REVIEW
        in source.plan.binding.final.disclosures
    ):
        review = dict(_review_payload(source.plan.review))
        review["runtime"] = _approver_runtime_payload(source.plan)
    else:
        review = None
    issuer = object()
    issuer_id = _issuer_id()
    signing_key = Ed25519PrivateKey.generate()
    correlation_id = PatchPublicCorrelationId.new()
    authority = ApproverProjectionAuthority(
        issuer,
        correlation_id,
        terminal.source_digest,
        terminal.terminal_digest,
    )
    return ApproverProjectionBoundary(
        issuer,
        issuer_id,
        signing_key,
        authority,
        correlation_id,
        terminal,
        review,
    )


def create_audit_projection_boundary(
    source: PatchProjectionSource,
) -> AuditProjectionBoundary:
    """Derive a trusted audit boundary and release canonical source truth."""
    terminal = _terminal_payload(source, Audience.AUDIT)
    issuer = object()
    issuer_id = _issuer_id()
    signing_key = Ed25519PrivateKey.generate()
    correlation_id = PatchPublicCorrelationId.new()
    authority = AuditProjectionAuthority(
        issuer,
        correlation_id,
        terminal.source_digest,
        terminal.terminal_digest,
    )
    return AuditProjectionBoundary(
        issuer,
        issuer_id,
        signing_key,
        authority,
        correlation_id,
        terminal,
    )


def _encode_model_envelope(
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
    signing_key: Ed25519PrivateKey,
    payload: ProjectionPayload,
) -> ModelProjectionEnvelope:
    """Return one model-typed envelope from trusted primitive payload data."""
    return ModelProjectionEnvelope(
        _encode_envelope(
            "model",
            correlation_id,
            source_digest,
            terminal_digest,
            issuer_id,
            signing_key,
            payload,
        )
    )


def _encode_approver_envelope(
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
    signing_key: Ed25519PrivateKey,
    payload: ProjectionPayload,
) -> ApproverProjectionEnvelope:
    """Return one approver-typed envelope from trusted primitive data."""
    return ApproverProjectionEnvelope(
        _encode_envelope(
            "approver",
            correlation_id,
            source_digest,
            terminal_digest,
            issuer_id,
            signing_key,
            payload,
        )
    )


def _encode_audit_envelope(
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
    signing_key: Ed25519PrivateKey,
    payload: ProjectionPayload,
) -> AuditProjectionEnvelope:
    """Return one audit-typed envelope from trusted primitive payload data."""
    return AuditProjectionEnvelope(
        _encode_envelope(
            "audit",
            correlation_id,
            source_digest,
            terminal_digest,
            issuer_id,
            signing_key,
            payload,
        )
    )


def _encode_envelope(
    audience: str,
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
    signing_key: Ed25519PrivateKey,
    payload: ProjectionPayload,
) -> bytes:
    """Return one signed canonical envelope from trusted boundary data."""
    if (
        audience not in {"model", "approver", "audit"}
        or not correlation_id.startswith("public_")
        or len(correlation_id) > 128
        or not _is_digest(source_digest)
        or not _is_digest(terminal_digest)
        or not _is_issuer_id(issuer_id)
    ):
        raise ProjectionError("projection envelope header is invalid")
    body = {
        "schema_version": 2,
        "audience": audience,
        "correlation_id": correlation_id,
        "source_digest": source_digest,
        "terminal_digest": terminal_digest,
        "issuer_id": issuer_id,
        "payload": _plain_projection_payload(payload),
    }
    signature = signing_key.sign(_canonical_projection_bytes(body)).hex()
    encoded = _canonical_projection_bytes({**body, "signature": signature})
    if len(encoded) > _MAX_OUTPUT_BYTES:
        raise ProjectionError("projection envelope exceeds its output bound")
    return encoded


def _verification_receipt(
    signing_key: Ed25519PrivateKey,
    audience: str,
    correlation_id: str,
    source_digest: str,
    terminal_digest: str,
    issuer_id: str,
) -> bytes:
    """Return one root-authenticated immutable public adapter receipt."""
    body = {
        "schema_version": 1,
        "audience": audience,
        "correlation_id": correlation_id,
        "source_digest": source_digest,
        "terminal_digest": terminal_digest,
        "issuer_id": issuer_id,
        "public_key": (
            signing_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
            .hex()
        ),
    }
    signature = _ROOT_SIGNER.sign(_canonical_projection_bytes(body)).hex()
    return _canonical_projection_bytes({**body, "signature": signature})


def _plain_projection_payload(
    value: ProjectionPayload,
) -> dict[str, ProjectionPayloadValue]:
    """Return JSON primitives for a trusted frozen projection payload."""
    return {key: _plain_projection_value(item) for key, item in value.items()}


def _plain_projection_value(
    value: ProjectionPayloadValue,
) -> ProjectionPayloadValue:
    """Return JSON primitives from one nested trusted payload value."""
    if isinstance(value, Mapping):
        return _plain_projection_payload(value)
    if isinstance(value, tuple):
        return tuple(_plain_projection_value(item) for item in value)
    return value


def _canonical_projection_bytes(value: object) -> bytes:
    """Return trusted canonical JSON bytes for signed projection records."""
    return dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _issuer_id() -> str:
    """Return one bounded random public identifier for a signing boundary."""
    return "issuer_" + token_urlsafe(24)


def _terminal_payload(
    source: PatchProjectionSource,
    audience: Audience,
) -> _TerminalPayload:
    """Validate source and derive detached terminal values and digests."""
    if (
        type(source) is not PatchProjectionSource
        or type(audience) is not Audience
    ):
        raise ProjectionError("projection source is invalid")
    try:
        source.__post_init__()
    except Exception as error:
        raise ProjectionError("projection source is invalid") from error
    result = source.terminal.result
    diagnostic = (
        None
        if result.diagnostic is None
        else coarsen_error_code(result.diagnostic.code, audience).value
    )
    values: dict[str, ProjectionPayloadValue] = {
        "status": result.status.value,
        "mutation_state": result.truth.mutation_state.value,
        "lineage_state": result.truth.lineage_state.value,
        "requested_effect_occurred": (
            result.truth.requested_effect_occurred.value
        ),
        "artifact_state": result.truth.artifact_state.value,
        "workspace_change": result.truth.workspace_change.value,
        "commit_set_exact": result.truth.commit_set_exact,
        "postcondition": result.truth.postcondition.value,
        "diagnostic_code": diagnostic,
    }
    return _TerminalPayload(
        _digest(_canonical_value(source.plan)),
        _digest(_canonical_value(source.terminal)),
        values,
    )


def _model_diff_payload(
    boundary: ModelProjectionBoundary,
    output_limit: ProjectionOutputLimit,
) -> ProjectionPayload:
    """Return one bounded model-safe diff payload from detached bytes."""
    if boundary._diff_prefix is None:
        return _redacted_diff_payload()
    visible = _utf8_prefix(boundary._diff_prefix, output_limit.value.value)
    omitted = boundary._diff_total_bytes - len(visible)
    if omitted:
        return {
            "content": visible.decode("utf-8"),
            "complete": False,
            "truncated": True,
            "redacted": False,
            "reason": "output_limit",
            "omitted_bytes": omitted if boundary._metadata_allowed else None,
        }
    return {
        "content": visible.decode("utf-8"),
        "complete": True,
        "truncated": False,
        "redacted": False,
        "reason": "complete",
        "omitted_bytes": 0 if boundary._metadata_allowed else None,
    }


def _redacted_diff_payload() -> ProjectionPayload:
    """Return the sole content-free audience diff representation."""
    return {
        "content": None,
        "complete": False,
        "truncated": False,
        "redacted": True,
        "reason": "unauthorized",
        "omitted_bytes": None,
    }


def _review_payload(value: object) -> ProjectionPayload:
    """Return detached primitive complete-review payload data for approvers."""
    converted = _review_value(value)
    if not isinstance(converted, dict):
        raise ProjectionError("complete review payload is invalid")
    return converted


def _approver_runtime_payload(plan: SealedPlan) -> ProjectionPayload:
    """Return detached runtime and target facts for approver review only."""
    binding = plan.binding
    handshake = binding.final.handshake
    return {
        "context_kind": binding.context_kind.value,
        "target_implementation": handshake.identity.implementation_id,
        "target_platform": handshake.platform.value,
        "approval_mode": binding.final.approval.mode.value,
    }


def _review_value(value: object) -> ProjectionPayloadValue:
    """Convert one complete-review value to bounded primitive payload data."""
    if value is None or type(value) in {bool, int, float, str}:
        assert value is None or isinstance(value, (bool, int, float, str))
        return value
    if type(value) is bytes:
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Enum):
        return str(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name.removeprefix("_"): _review_value(
                getattr(value, item.name)
            )
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return tuple(_review_value(item) for item in value)
    if isinstance(value, frozenset):
        return tuple(
            sorted(
                (_review_value(item) for item in value),
                key=lambda item: dumps(item, sort_keys=True),
            )
        )
    raise ProjectionError("complete review contains an unsupported value")


def _canonical_value(value: object) -> object:
    """Return stable digest input without retaining source objects."""
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "fields": [
                (item.name, _canonical_value(getattr(value, item.name)))
                for item in fields(value)
            ],
        }
    if isinstance(value, tuple):
        return ["tuple", *(_canonical_value(item) for item in value)]
    if isinstance(value, frozenset):
        return [
            "frozenset",
            *sorted(
                (_canonical_value(item) for item in value),
                key=lambda item: dumps(item, sort_keys=True),
            ),
        ]
    raise ProjectionError("projection source contains an unsupported value")


def _digest(value: object) -> str:
    """Return one lower-case SHA-256 digest for detached canonical facts."""
    return sha256(
        dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_digest(value: str) -> bool:
    """Return whether one value is a lower-case SHA-256 hexadecimal digest."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_issuer_id(value: str) -> bool:
    """Return whether one issuer identifier has the bounded public form."""
    return value.startswith("issuer_") and len(value) <= 128


def _utf8_prefix(value: bytes, limit: int) -> bytes:
    """Return the largest valid UTF-8 prefix that fits one byte limit."""
    candidate = value[:limit]
    while candidate:
        try:
            candidate.decode("utf-8")
            return candidate
        except UnicodeDecodeError:
            candidate = candidate[:-1]
    return candidate
