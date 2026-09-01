"""Define frozen production-source patch activation manifests.

The patch package remains absent by default.  This module binds every future
advertisement to one exact, source-derived profile and keeps deactivation from
rewriting the ownership of an operation that has already started.
"""

from asyncio import Lock
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from hmac import compare_digest
from hmac import digest as hmac_digest
from json import dumps
from re import fullmatch
from secrets import token_bytes

from avalan.patch.domain import (
    ContextKind,
    PatchCommitOwnerId,
    PatchRequestId,
    SequenceNumber,
)
from avalan.patch.durable_store import DurableCommitLease, DurablePatchStore
from avalan.patch.protocols import PatchProtocolSurface
from avalan.patch.toolset import PATCH_APPLY_SCHEMA, PATCH_EDIT_SCHEMA

_COMPONENT_PATTERN = r"[a-z][a-z0-9._:-]{0,127}"
_MANIFEST_VERSION = 1


class PatchActivationError(ValueError):
    """Report a closed patch activation or manifest failure."""


class PatchActivationPlatform(str, Enum):
    """Name the platform receipt fixed in an activation profile."""

    MACOS = "macos"
    LINUX = "linux"


class PatchActivationSurface(str, Enum):
    """Name the public surface covered by an activation profile."""

    JSON_FUNCTION = "json_function"
    MCP = "mcp"
    A2A = "a2a"
    FLOW = "flow"
    TASK = "task"
    MULTI_AGENT = "multi_agent"
    PROVIDER_FREEFORM = "provider_freeform"
    PROVIDER_NATIVE = "provider_native"


class PatchProfileState(str, Enum):
    """Name the immutable evidence and selection state of one profile."""

    INCOMPLETE = "incomplete"
    NOT_SELECTED = "not_selected"
    SELECTED = "selected"


class PatchActivationOperationState(str, Enum):
    """Name the in-flight state whose owner deactivation cannot replace."""

    AWAITING_APPROVAL = "awaiting_approval"
    IN_FLIGHT = "in_flight"
    PARTIAL = "partial"
    SETTLEMENT_PENDING = "settlement_pending"


@dataclass(frozen=True, slots=True)
class PatchActivationLimits:
    """Bound dormant activation registrations and retained operations."""

    max_active_profiles: int = 1
    max_operations_per_profile: int = 64

    def __post_init__(self) -> None:
        """Reject an unbounded or non-integer registry configuration."""
        if (
            type(self.max_active_profiles) is not int
            or self.max_active_profiles < 1
            or type(self.max_operations_per_profile) is not int
            or self.max_operations_per_profile < 1
        ):
            raise PatchActivationError("patch activation limits are invalid")


@dataclass(frozen=True, slots=True)
class PatchProfileComponent:
    """Store one exact normalized non-secret profile component."""

    value: str

    def __post_init__(self) -> None:
        """Reject ambiguous profile component spellings."""
        if (
            type(self.value) is not str
            or fullmatch(_COMPONENT_PATTERN, self.value) is None
        ):
            raise PatchActivationError("patch profile component is invalid")


@dataclass(frozen=True, slots=True)
class PatchActivationProfileKey:
    """Identify the complete non-interchangeable patch profile tuple."""

    context: ContextKind
    platform: PatchActivationPlatform
    filesystem: PatchProfileComponent
    target_implementation: PatchProfileComponent
    target_protocol: PatchProfileComponent
    policy: PatchProfileComponent
    approval: PatchProfileComponent
    persistence: PatchProfileComponent
    surface: PatchActivationSurface
    provider_codec: PatchProfileComponent
    version: PatchProfileComponent

    def __post_init__(self) -> None:
        """Require every activation coordinate to be typed and present."""
        if (
            type(self.context) is not ContextKind
            or type(self.platform) is not PatchActivationPlatform
            or type(self.filesystem) is not PatchProfileComponent
            or type(self.target_implementation) is not PatchProfileComponent
            or type(self.target_protocol) is not PatchProfileComponent
            or type(self.policy) is not PatchProfileComponent
            or type(self.approval) is not PatchProfileComponent
            or type(self.persistence) is not PatchProfileComponent
            or type(self.surface) is not PatchActivationSurface
            or type(self.provider_codec) is not PatchProfileComponent
            or type(self.version) is not PatchProfileComponent
        ):
            raise PatchActivationError(
                "patch activation profile key is invalid"
            )

    @property
    def digest(self) -> str:
        """Return the canonical digest for the complete profile key."""
        return _digest(_profile_key_payload(self))


@dataclass(frozen=True, slots=True)
class PatchProfileProofs:
    """Record independently verified activation properties for one profile."""

    context: bool
    platform: bool
    filesystem: bool
    target: bool
    protocol: bool
    policy: bool
    approval: bool
    persistence: bool
    surface: bool
    provider_codec: bool

    def __post_init__(self) -> None:
        """Reject non-boolean proof values before profile construction."""
        if any(
            type(value) is not bool
            for value in (
                self.context,
                self.platform,
                self.filesystem,
                self.target,
                self.protocol,
                self.policy,
                self.approval,
                self.persistence,
                self.surface,
                self.provider_codec,
            )
        ):
            raise PatchActivationError("patch profile proofs are invalid")

    @property
    def complete(self) -> bool:
        """Return whether every profile coordinate has direct evidence."""
        return all(
            (
                self.context,
                self.platform,
                self.filesystem,
                self.target,
                self.protocol,
                self.policy,
                self.approval,
                self.persistence,
                self.surface,
                self.provider_codec,
            )
        )


@dataclass(frozen=True, slots=True)
class PatchCapabilityProfile:
    """Freeze one exact profile and its non-authorizing capabilities."""

    key: PatchActivationProfileKey
    proofs: PatchProfileProofs
    state: PatchProfileState
    selection_rationale: str
    capability_inventory: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous profile declarations."""
        if (
            type(self.key) is not PatchActivationProfileKey
            or type(self.proofs) is not PatchProfileProofs
            or type(self.state) is not PatchProfileState
            or type(self.selection_rationale) is not str
            or not self.selection_rationale
            or not self.capability_inventory
            or any(
                type(name) is not str
                or name not in {"patch.edit", "patch.apply"}
                for name in self.capability_inventory
            )
            or len(self.capability_inventory)
            != len(set(self.capability_inventory))
        ):
            raise PatchActivationError("patch capability profile is invalid")
        if (
            self.state is PatchProfileState.SELECTED
            and not self.proofs.complete
        ):
            raise PatchActivationError("selected patch profile is incomplete")
        if self.state is PatchProfileState.INCOMPLETE and self.proofs.complete:
            raise PatchActivationError(
                "incomplete patch profile has full proof"
            )

    @property
    def proven(self) -> bool:
        """Return whether the profile can be considered for activation."""
        return self.proofs.complete

    @property
    def digest(self) -> str:
        """Return the canonical digest for this profile and its evidence."""
        return _digest(_profile_payload(self))


@dataclass(frozen=True, slots=True)
class PatchSchemaDescription:
    """Freeze one public schema generated from a production tool source."""

    tool_name: str
    canonical_json: str
    schema_sha256: str

    def __post_init__(self) -> None:
        """Reject an unsealed schema or an unsupported public tool name."""
        if (
            self.tool_name not in {"patch.edit", "patch.apply"}
            or type(self.canonical_json) is not str
            or type(self.schema_sha256) is not str
            or self.schema_sha256 != _digest(self.canonical_json)
        ):
            raise PatchActivationError("patch schema description is invalid")


@dataclass(frozen=True, slots=True)
class PatchProtocolDescription:
    """Describe one protocol capability inventory without authority."""

    protocol_surface: PatchProtocolSurface
    capability_inventory: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a protocol description that widens the source contract."""
        if (
            type(self.protocol_surface) is not PatchProtocolSurface
            or any(
                tool not in {"patch.edit", "patch.apply"}
                for tool in self.capability_inventory
            )
            or len(self.capability_inventory)
            != len(set(self.capability_inventory))
        ):
            raise PatchActivationError("patch protocol description is invalid")


@dataclass(frozen=True, slots=True)
class PatchProductionSource:
    """Name one tracked production symbol used to generate the manifest."""

    module: str
    symbol: str

    def __post_init__(self) -> None:
        """Reject non-production or unaddressable source references."""
        if (
            type(self.module) is not str
            or type(self.symbol) is not str
            or not self.module.startswith("avalan.patch.")
            or fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.symbol) is None
        ):
            raise PatchActivationError("patch production source is invalid")


@dataclass(frozen=True, slots=True)
class PatchProductionManifest:
    """Freeze generated schemas, tools, protocol descriptions, and profiles."""

    schema_version: int
    sources: tuple[PatchProductionSource, ...]
    schemas: tuple[PatchSchemaDescription, ...]
    tool_inventory: tuple[str, ...]
    protocols: tuple[PatchProtocolDescription, ...]
    profiles: tuple[PatchCapabilityProfile, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        """Reject an incomplete, duplicated, or unsealed manifest."""
        profile_keys = tuple(profile.key for profile in self.profiles)
        schema_names = tuple(schema.tool_name for schema in self.schemas)
        if (
            self.schema_version != _MANIFEST_VERSION
            or not self.sources
            or not self.schemas
            or not self.protocols
            or not self.profiles
            or len(self.sources) != len(set(self.sources))
            or len(schema_names) != len(set(schema_names))
            or self.tool_inventory != schema_names
            or len(profile_keys) != len(set(profile_keys))
            or self.manifest_sha256 != _digest(_manifest_payload(self))
        ):
            raise PatchActivationError("patch production manifest is invalid")

    def profile_for(
        self, key: PatchActivationProfileKey
    ) -> PatchCapabilityProfile:
        """Return one exact profile without fallback or partial matching."""
        if type(key) is not PatchActivationProfileKey:
            raise PatchActivationError(
                "patch activation profile is unavailable"
            )
        for profile in self.profiles:
            if profile.key == key:
                return profile
        raise PatchActivationError("patch activation profile is unavailable")


@dataclass(frozen=True, slots=True)
class PatchActivationLease:
    """Record the epoch assigned atomically to newly admissible work."""

    key: PatchActivationProfileKey
    epoch: int
    selected_tools: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject a lease with incomplete key, epoch, or tool inventory."""
        if (
            type(self.key) is not PatchActivationProfileKey
            or type(self.epoch) is not int
            or self.epoch < 1
            or not self.selected_tools
        ):
            raise PatchActivationError("patch activation lease is invalid")


@dataclass(frozen=True, slots=True)
class PatchActivationRuntimeRecord:
    """Bind one profile selection to its authenticated runtime and store."""

    key: PatchActivationProfileKey
    runtime_digest: str
    _store: DurablePatchStore

    def __post_init__(self) -> None:
        """Reject a record without an exact trusted durable-store binding."""
        if (
            type(self.key) is not PatchActivationProfileKey
            or fullmatch(r"[0-9a-f]{64}", self.runtime_digest) is None
            or not _durable_store_contract(self._store)
        ):
            raise PatchActivationError("patch activation runtime is invalid")


def _durable_store_contract(store: object) -> bool:
    """Return whether one selected runtime exposes the durable store API."""
    return all(
        callable(getattr(store, name, None))
        for name in (
            "reserve",
            "persist_plan",
            "claim_commit",
            "renew_lease",
            "bind_worker",
            "mark_worker_reaped",
            "mark_worker_absent",
            "replace_expired_owner",
            "is_current_fence",
            "append_step",
            "append_artifact",
            "suspend",
            "request_cancellation",
            "settle",
            "inspect",
            "inspect_pending",
            "await_terminal",
            "outbox",
        )
    )


@dataclass(frozen=True, slots=True)
class PatchActivationDurableOperation:
    """Freeze the coordinator-issued owner and fence for one operation."""

    request_id: PatchRequestId
    owner: PatchCommitOwnerId
    fence: SequenceNumber

    def __post_init__(self) -> None:
        """Require the exact typed durable coordination values."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.owner) is not PatchCommitOwnerId
            or type(self.fence) is not SequenceNumber
            or self.fence.value == 0
        ):
            raise PatchActivationError("patch activation owner is invalid")


_ACTIVATION_RECEIPT_ISSUER = object()
_ACTIVATION_VERIFIER_ISSUER = object()
_ACTIVATION_AUTHORITY_ISSUER = object()


@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchActivationRuntimeAuthority:
    """Hold verifier-owned authentication material outside public inputs."""

    _key: bytes
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject caller-created activation authorities."""
        del args, kwargs
        raise PatchActivationError(
            "patch activation authority is runtime-owned"
        )

    def _sign(self, payload: bytes) -> bytes:
        """Authenticate one canonical receipt payload for this runtime."""
        if self._issuer is not _ACTIVATION_AUTHORITY_ISSUER:
            raise PatchActivationError("patch activation authority is invalid")
        return hmac_digest(self._key, payload, "sha256")

    def _verify(self, payload: bytes, signature: bytes) -> bool:
        """Verify one receipt without disclosing authority material."""
        return (
            self._issuer is _ACTIVATION_AUTHORITY_ISSUER
            and type(signature) is bytes
            and compare_digest(self._sign(payload), signature)
        )


def _new_activation_authority(
    key: bytes,
) -> PatchActivationRuntimeAuthority:
    """Construct one trusted runtime authority from non-public key material."""
    if type(key) is not bytes or len(key) != 32:
        raise PatchActivationError("patch activation authority is invalid")
    authority = object.__new__(PatchActivationRuntimeAuthority)
    object.__setattr__(authority, "_key", key)
    object.__setattr__(authority, "_issuer", _ACTIVATION_AUTHORITY_ISSUER)
    return authority


_PRODUCTION_ACTIVATION_AUTHORITY = _new_activation_authority(token_bytes(32))


@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchVerifiedActivationReceipt:
    """Bind verifier-authenticated evidence to one runtime store record."""

    manifest_sha256: str
    profile_key: PatchActivationProfileKey
    profile_sha256: str
    runtime_digest: str
    evidence_sha256: str
    _signature: bytes
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject construction outside the sealed verifier factory."""
        del args, kwargs
        raise PatchActivationError(
            "patch activation receipt is verifier-issued"
        )

    def __copy__(self) -> "PatchVerifiedActivationReceipt":
        """Reject copies that could detach a durable verifier binding."""
        raise PatchActivationError("patch activation receipt cannot be copied")

    def __deepcopy__(self, memo: object) -> "PatchVerifiedActivationReceipt":
        """Reject copies that could detach a durable verifier binding."""
        del memo
        raise PatchActivationError("patch activation receipt cannot be copied")


@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchActivationVerifier:
    """Verify sealed source descriptors before issuing activation receipts."""

    _manifest: PatchProductionManifest
    _authority: PatchActivationRuntimeAuthority
    _production: bool
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject caller-made verifiers and arbitrary evidence inputs."""
        del args, kwargs
        raise PatchActivationError(
            "patch activation verifier is factory-issued"
        )

    def _runtime_receipt(
        self, record: PatchActivationRuntimeRecord
    ) -> PatchVerifiedActivationReceipt | None:
        """Issue one profile receipt from a runtime-owned durable record."""
        if self._issuer is not _ACTIVATION_VERIFIER_ISSUER:
            raise PatchActivationError("patch activation verifier is invalid")
        if type(record) is not PatchActivationRuntimeRecord:
            raise PatchActivationError("patch activation receipt is invalid")
        if (
            self._production
            and self._manifest != build_patch_production_manifest()
        ):
            raise PatchActivationError(
                "patch activation manifest is not tracked"
            )
        for profile in self._manifest.profiles:
            if (
                profile.key == record.key
                and profile.state is PatchProfileState.SELECTED
                and profile.proven
            ):
                return _issue_verified_receipt(
                    self._manifest,
                    profile,
                    record,
                    self._authority,
                )
        return None


def build_patch_activation_verifier(
    manifest: PatchProductionManifest,
) -> PatchActivationVerifier:
    """Bind the exact tracked manifest to the receipt-verifier boundary."""
    expected = build_patch_production_manifest()
    if (
        type(manifest) is not PatchProductionManifest
        or manifest.manifest_sha256 != expected.manifest_sha256
        or manifest != expected
    ):
        raise PatchActivationError("patch activation manifest is not tracked")
    return _build_activation_verifier(
        manifest,
        _PRODUCTION_ACTIVATION_AUTHORITY,
        production=True,
    )


def _build_activation_verifier(
    manifest: PatchProductionManifest,
    authority: PatchActivationRuntimeAuthority,
    *,
    production: bool,
) -> PatchActivationVerifier:
    """Bind one trusted authority to a sealed production or test artifact."""
    if (
        type(manifest) is not PatchProductionManifest
        or type(authority) is not PatchActivationRuntimeAuthority
        or type(production) is not bool
    ):
        raise PatchActivationError("patch activation verifier is invalid")
    verifier = object.__new__(PatchActivationVerifier)
    object.__setattr__(verifier, "_manifest", manifest)
    object.__setattr__(verifier, "_authority", authority)
    object.__setattr__(verifier, "_production", production)
    object.__setattr__(verifier, "_issuer", _ACTIVATION_VERIFIER_ISSUER)
    return verifier


def _issue_verified_receipt(
    manifest: PatchProductionManifest,
    profile: PatchCapabilityProfile,
    record: PatchActivationRuntimeRecord,
    authority: PatchActivationRuntimeAuthority,
) -> PatchVerifiedActivationReceipt:
    """Issue one receipt from an exact complete verifier-owned profile."""
    if (
        type(manifest) is not PatchProductionManifest
        or type(profile) is not PatchCapabilityProfile
        or type(record) is not PatchActivationRuntimeRecord
        or type(authority) is not PatchActivationRuntimeAuthority
    ):
        raise PatchActivationError("patch activation receipt is invalid")
    tracked = manifest.profile_for(profile.key)
    if (
        tracked != profile
        or record.key != profile.key
        or profile.state is not PatchProfileState.SELECTED
        or not profile.proven
    ):
        raise PatchActivationError("patch activation profile is unavailable")
    receipt = object.__new__(PatchVerifiedActivationReceipt)
    object.__setattr__(receipt, "manifest_sha256", manifest.manifest_sha256)
    object.__setattr__(receipt, "profile_key", profile.key)
    object.__setattr__(receipt, "profile_sha256", profile.digest)
    object.__setattr__(receipt, "runtime_digest", record.runtime_digest)
    object.__setattr__(
        receipt,
        "evidence_sha256",
        _digest(
            {
                "manifest": manifest.manifest_sha256,
                "profile": profile.digest,
                "runtime": record.runtime_digest,
                "evidence": _production_evidence_digest(manifest, profile),
            }
        ),
    )
    object.__setattr__(
        receipt,
        "_signature",
        authority._sign(_receipt_payload(receipt)),
    )
    object.__setattr__(receipt, "_issuer", _ACTIVATION_RECEIPT_ISSUER)
    return receipt


@dataclass(frozen=True, slots=True)
class PatchActivationOperationBinding:
    """Freeze an operation's original owner and epoch at admission time."""

    request_id: PatchRequestId
    owner: PatchCommitOwnerId
    durable_fence: SequenceNumber
    lease: PatchActivationLease
    state: PatchActivationOperationState

    def __post_init__(self) -> None:
        """Reject a binding that could be substituted across operations."""
        if (
            type(self.request_id) is not PatchRequestId
            or type(self.owner) is not PatchCommitOwnerId
            or type(self.durable_fence) is not SequenceNumber
            or type(self.lease) is not PatchActivationLease
            or type(self.state) is not PatchActivationOperationState
        ):
            raise PatchActivationError("patch activation binding is invalid")


@dataclass(frozen=True, slots=True)
class PatchDeactivationReceipt:
    """State that only future bindings were stopped at one exact epoch."""

    key: PatchActivationProfileKey
    retired_epoch: int | None

    def __post_init__(self) -> None:
        """Reject a malformed deactivation result."""
        if (
            type(self.key) is not PatchActivationProfileKey
            or self.retired_epoch is not None
            and (type(self.retired_epoch) is not int or self.retired_epoch < 1)
        ):
            raise PatchActivationError("patch deactivation receipt is invalid")


def _receipt_matches(
    receipt: PatchVerifiedActivationReceipt,
    manifest: PatchProductionManifest,
    verifier: PatchActivationVerifier,
) -> bool:
    """Return whether one verifier-issued receipt binds this exact manifest."""
    if (
        type(receipt) is not PatchVerifiedActivationReceipt
        or receipt._issuer is not _ACTIVATION_RECEIPT_ISSUER
        or type(receipt.profile_key) is not PatchActivationProfileKey
        or fullmatch(r"[0-9a-f]{64}", receipt.runtime_digest) is None
        or fullmatch(r"[0-9a-f]{64}", receipt.evidence_sha256) is None
        or type(receipt._signature) is not bytes
        or type(verifier) is not PatchActivationVerifier
        or verifier._issuer is not _ACTIVATION_VERIFIER_ISSUER
        or verifier._manifest != manifest
    ):
        return False
    try:
        profile = manifest.profile_for(receipt.profile_key)
    except PatchActivationError:
        return False
    if (
        receipt.manifest_sha256 != manifest.manifest_sha256
        or receipt.profile_sha256 != profile.digest
    ):
        return False
    if profile.state is not PatchProfileState.SELECTED or not profile.proven:
        return False
    return receipt.evidence_sha256 == _digest(
        {
            "manifest": manifest.manifest_sha256,
            "profile": profile.digest,
            "runtime": receipt.runtime_digest,
            "evidence": _production_evidence_digest(manifest, profile),
        }
    ) and verifier._authority._verify(
        _receipt_payload(receipt), receipt._signature
    )


def _receipt_payload(receipt: PatchVerifiedActivationReceipt) -> bytes:
    """Return the complete authority-signed receipt binding payload."""
    return dumps(
        {
            "manifest": receipt.manifest_sha256,
            "profile": receipt.profile_sha256,
            "profile_key": _profile_key_payload(receipt.profile_key),
            "runtime": receipt.runtime_digest,
            "evidence": receipt.evidence_sha256,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class PatchActivationRegistry:
    """Select verified receipts and retain exact durable operation owners."""

    def __init__(
        self,
        manifest: PatchProductionManifest,
        verifier: PatchActivationVerifier,
        limits: PatchActivationLimits = PatchActivationLimits(),
    ) -> None:
        """Initialize a dormant registry from one frozen source manifest."""
        if (
            type(manifest) is not PatchProductionManifest
            or type(verifier) is not PatchActivationVerifier
            or verifier._issuer is not _ACTIVATION_VERIFIER_ISSUER
            or verifier._manifest != manifest
            or type(limits) is not PatchActivationLimits
        ):
            raise PatchActivationError("patch activation manifest is invalid")
        self._manifest = manifest
        self._verifier = verifier
        self._limits = limits
        self._lock = Lock()
        self._epochs: dict[
            PatchActivationProfileKey,
            tuple[int, PatchVerifiedActivationReceipt],
        ] = {}
        self._bindings: dict[
            tuple[PatchActivationProfileKey, PatchRequestId],
            PatchActivationOperationBinding,
        ] = {}
        self._next_epoch = 0

    async def activate(
        self, receipt: PatchVerifiedActivationReceipt
    ) -> PatchActivationLease:
        """Atomically activate one exact verifier-issued durable receipt."""
        if not _receipt_matches(receipt, self._manifest, self._verifier):
            raise PatchActivationError(
                "patch activation receipt is unavailable"
            )
        async with self._lock:
            profile = self._manifest.profile_for(receipt.profile_key)
            if (
                profile.key in self._epochs
                or len(self._epochs) >= self._limits.max_active_profiles
                or any(
                    binding.lease.key == profile.key
                    for binding in self._bindings.values()
                )
            ):
                raise PatchActivationError(
                    "patch activation profile is unavailable"
                )
            self._next_epoch += 1
            self._epochs[profile.key] = (self._next_epoch, receipt)
            return PatchActivationLease(
                key=profile.key,
                epoch=self._next_epoch,
                selected_tools=profile.capability_inventory,
            )

    async def bind_operation(
        self,
        receipt: PatchVerifiedActivationReceipt,
        durable: PatchActivationDurableOperation,
        state: PatchActivationOperationState,
    ) -> PatchActivationOperationBinding:
        """Bind new work to the active epoch without affecting existing work.

        Existing bindings retain their original epoch and owner.
        """
        if (
            not _receipt_matches(receipt, self._manifest, self._verifier)
            or type(durable) is not PatchActivationDurableOperation
            or type(state) is not PatchActivationOperationState
        ):
            raise PatchActivationError("patch activation binding is invalid")
        async with self._lock:
            active = self._epochs.get(receipt.profile_key)
            if active is None or active[1] is not receipt:
                raise PatchActivationError(
                    "patch activation profile is unavailable"
                )
            epoch = active[0]
            binding_key = (receipt.profile_key, durable.request_id)
            existing = self._bindings.get(binding_key)
            if existing is not None:
                if (
                    existing.owner != durable.owner
                    or existing.durable_fence != durable.fence
                    or existing.lease.epoch != epoch
                    or existing.state is not state
                ):
                    raise PatchActivationError(
                        "patch activation binding is unavailable"
                    )
                return existing
            retained = sum(
                binding.lease.key == receipt.profile_key
                for binding in self._bindings.values()
            )
            if retained >= self._limits.max_operations_per_profile:
                raise PatchActivationError(
                    "patch activation profile is unavailable"
                )
            profile = self._manifest.profile_for(receipt.profile_key)
            binding = PatchActivationOperationBinding(
                request_id=durable.request_id,
                owner=durable.owner,
                durable_fence=durable.fence,
                lease=PatchActivationLease(
                    receipt.profile_key,
                    epoch,
                    profile.capability_inventory,
                ),
                state=state,
            )
            self._bindings[binding_key] = binding
            return binding

    async def retain_operation(
        self,
        key: PatchActivationProfileKey,
        durable: PatchActivationDurableOperation,
        state: PatchActivationOperationState,
    ) -> PatchActivationOperationBinding:
        """Retain an already admitted pending or partial durable owner."""
        if (
            type(key) is not PatchActivationProfileKey
            or type(durable) is not PatchActivationDurableOperation
            or state
            not in {
                PatchActivationOperationState.PARTIAL,
                PatchActivationOperationState.SETTLEMENT_PENDING,
            }
        ):
            raise PatchActivationError("patch activation binding is invalid")
        async with self._lock:
            binding_key = (key, durable.request_id)
            binding = self._bindings.get(binding_key)
            if (
                binding is None
                or binding.owner != durable.owner
                or binding.durable_fence != durable.fence
            ):
                raise PatchActivationError(
                    "patch activation binding is unavailable"
                )
            retained = PatchActivationOperationBinding(
                binding.request_id,
                binding.owner,
                binding.durable_fence,
                binding.lease,
                state,
            )
            self._bindings[binding_key] = retained
            return retained

    async def active_binding_count(
        self, key: PatchActivationProfileKey
    ) -> int:
        """Return all retained operations, including deactivated epochs."""
        if type(key) is not PatchActivationProfileKey:
            raise PatchActivationError(
                "patch activation profile is unavailable"
            )
        async with self._lock:
            return sum(
                binding.lease.key == key for binding in self._bindings.values()
            )

    async def advertised_tools(
        self, key: PatchActivationProfileKey
    ) -> tuple[str, ...]:
        """Return tools only for a currently selected exact profile."""
        if type(key) is not PatchActivationProfileKey:
            raise PatchActivationError(
                "patch activation profile is unavailable"
            )
        async with self._lock:
            if key not in self._epochs:
                return ()
            return self._manifest.profile_for(key).capability_inventory

    async def deactivate(
        self, key: PatchActivationProfileKey
    ) -> PatchDeactivationReceipt:
        """Stop future bindings without rewriting issued operation bindings."""
        if type(key) is not PatchActivationProfileKey:
            raise PatchActivationError(
                "patch activation profile is unavailable"
            )
        async with self._lock:
            active = self._epochs.pop(key, None)
            return PatchDeactivationReceipt(
                key,
                None if active is None else active[0],
            )

    async def release_operation(
        self,
        key: PatchActivationProfileKey,
        durable: PatchActivationDurableOperation,
        epoch: int,
    ) -> PatchActivationOperationBinding:
        """Release one settled binding with its original owner and fence."""
        if (
            type(key) is not PatchActivationProfileKey
            or type(durable) is not PatchActivationDurableOperation
            or type(epoch) is not int
            or epoch < 1
        ):
            raise PatchActivationError("patch activation release is invalid")
        async with self._lock:
            binding = self._bindings.get((key, durable.request_id))
            if (
                binding is None
                or binding.owner != durable.owner
                or binding.durable_fence != durable.fence
                or binding.lease.epoch != epoch
            ):
                raise PatchActivationError(
                    "patch activation release is unavailable"
                )
            del self._bindings[(key, durable.request_id)]
            return binding


_ACTIVATION_FACTORY_ISSUER = object()
_ISSUED_ACTIVATION_FACTORIES: list["PatchActivationRuntimeFactory"] = []


@dataclass(frozen=True, slots=True, repr=False, init=False)
class PatchActivationRuntimeFactory:
    """Create activation leases only from a bound runtime record."""

    _manifest: PatchProductionManifest
    _verifier: PatchActivationVerifier
    _issuer: object

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject factories assembled from caller-supplied receipt fields."""
        del args, kwargs
        raise PatchActivationError("patch activation factory is runtime-owned")

    async def activate(
        self, binding: object
    ) -> "PatchActivationRuntime | None":
        """Activate one proven profile from its authenticated store binding."""
        if self._issuer is not _ACTIVATION_FACTORY_ISSUER:
            raise PatchActivationError("patch activation factory is invalid")
        record = _runtime_record(self._manifest, binding)
        if record is None:
            return None
        profile = self._manifest.profile_for(record.key)
        if (
            profile.state is not PatchProfileState.SELECTED
            or not profile.proven
        ):
            return None
        registry = _registry_for_store(
            record._store, self._manifest, self._verifier
        )
        receipt = self._verifier._runtime_receipt(record)
        if receipt is None:
            return None
        try:
            lease = await registry.activate(receipt)
        except PatchActivationError:
            return None
        runtime = PatchActivationRuntime(registry, receipt, lease, record)
        setter = getattr(
            getattr(binding, "service", None), "set_activation_observer", None
        )
        if not callable(setter):
            await runtime.deactivate()
            return None
        try:
            setter(runtime)
        except Exception:
            await runtime.deactivate()
            return None
        return runtime


def is_patch_activation_runtime_factory(value: object) -> bool:
    """Return whether one concrete factory was sealed by this module."""
    return type(value) is PatchActivationRuntimeFactory and any(
        value is factory for factory in _ISSUED_ACTIVATION_FACTORIES
    )


def validates_patch_activation_runtime(
    factory: object,
    binding: object,
    runtime: object,
) -> bool:
    """Return whether one runtime belongs to its sealed factory binding."""
    if (
        type(factory) is not PatchActivationRuntimeFactory
        or not is_patch_activation_runtime_factory(factory)
        or type(runtime) is not PatchActivationRuntime
    ):
        return False
    record = _runtime_record(factory._manifest, binding)
    if record is None or runtime.record != record:
        return False
    return (
        runtime.registry
        is _registry_for_store(
            record._store,
            factory._manifest,
            factory._verifier,
        )
        and runtime.lease.key == record.key
        and runtime.receipt.profile_key == record.key
    )


_RUNTIME_REGISTRIES: dict[tuple[int, str], PatchActivationRegistry] = {}


def _registry_for_store(
    store: DurablePatchStore,
    manifest: PatchProductionManifest,
    verifier: PatchActivationVerifier,
) -> PatchActivationRegistry:
    """Retain one registry across host reconstruction for one durable store."""
    key = (id(store), manifest.manifest_sha256)
    registry = _RUNTIME_REGISTRIES.get(key)
    if registry is None:
        registry = PatchActivationRegistry(manifest, verifier)
        _RUNTIME_REGISTRIES[key] = registry
    return registry


def _runtime_record(
    manifest: PatchProductionManifest,
    binding: object,
) -> PatchActivationRuntimeRecord | None:
    """Derive one exact profile record from a completed runtime handshake."""
    scope = getattr(binding, "scope", None)
    handshake = getattr(binding, "handshake", None)
    coordinator = getattr(binding, "coordinator", None)
    persistence = getattr(binding, "persistence", None)
    policy = getattr(binding, "policy", None)
    store = getattr(coordinator, "durable_store", None)
    if (
        scope is None
        or handshake is None
        or store is None
        or store is not getattr(persistence, "durable_store", None)
        or not isinstance(store, DurablePatchStore)
        or not _durable_store_contract(store)
    ):
        return None
    context = getattr(scope, "context_kind", None)
    platform_value = getattr(
        getattr(handshake, "platform", None), "value", None
    )
    if platform_value == "darwin":
        platform_value = PatchActivationPlatform.MACOS.value
    try:
        platform = PatchActivationPlatform(platform_value)
    except (TypeError, ValueError):
        return None
    profile = next(
        (
            item
            for item in manifest.profiles
            if item.key.context is context and item.key.platform is platform
        ),
        None,
    )
    if profile is None:
        return None
    identity = getattr(scope, "identity", None)
    runtime_digest = _digest(
        {
            "profile": profile.digest,
            "context": getattr(context, "value", None),
            "target": getattr(
                getattr(identity, "target_id", None), "value", None
            ),
            "workspace": getattr(
                getattr(identity, "workspace_id", None), "value", None
            ),
            "domain": getattr(
                getattr(identity, "domain_id", None), "value", None
            ),
            "policy": getattr(
                getattr(policy, "revision", None), "value", None
            ),
            "store_identity": id(store),
        }
    )
    return PatchActivationRuntimeRecord(
        profile.key,
        runtime_digest,
        store,
    )


@dataclass(slots=True, repr=False)
class PatchActivationRuntime:
    """Keep a selected registry lease attached to one ToolSet or SDK host."""

    registry: PatchActivationRegistry
    receipt: PatchVerifiedActivationReceipt
    lease: PatchActivationLease
    record: PatchActivationRuntimeRecord
    _deactivated: bool = False
    _released_operations: set[PatchActivationDurableOperation] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _release_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
    )

    async def bind_durable_commit(self, lease: DurableCommitLease) -> None:
        """Bind a real service-issued owner and fence before an effect."""
        durable = _durable_operation(lease)
        if self._deactivated:
            raise PatchActivationError(
                "patch activation profile is unavailable"
            )
        await self.registry.bind_operation(
            self.receipt,
            durable,
            PatchActivationOperationState.IN_FLIGHT,
        )

    async def retain_durable_commit(self, lease: DurableCommitLease) -> None:
        """Retain a pending or partial service-owned durable operation."""
        await self.registry.retain_operation(
            self.lease.key,
            _durable_operation(lease),
            PatchActivationOperationState.SETTLEMENT_PENDING,
        )

    async def release_durable_commit(self, lease: DurableCommitLease) -> None:
        """Release one terminal owner, tolerating its exact repeat only."""
        durable = _durable_operation(lease)
        async with self._release_lock:
            if durable in self._released_operations:
                return
            await self.registry.release_operation(
                self.lease.key, durable, self.lease.epoch
            )
            self._released_operations.add(durable)

    async def deactivate(self) -> PatchDeactivationReceipt:
        """Stop future admissions while retaining existing durable owners."""
        self._deactivated = True
        return await self.registry.deactivate(self.lease.key)


def _durable_operation(
    lease: DurableCommitLease,
) -> PatchActivationDurableOperation:
    """Project one exact coordinator lease into the activation registry."""
    if type(lease) is not DurableCommitLease:
        raise PatchActivationError("patch activation owner is invalid")
    return PatchActivationDurableOperation(
        lease.request_id, lease.owner_id, lease.fence
    )


def _build_activation_factory(
    manifest: PatchProductionManifest,
    verifier: PatchActivationVerifier,
) -> PatchActivationRuntimeFactory:
    """Seal one source manifest and verifier at the host factory seam."""
    if (
        type(manifest) is not PatchProductionManifest
        or type(verifier) is not PatchActivationVerifier
        or verifier._manifest != manifest
    ):
        raise PatchActivationError("patch activation factory is invalid")
    factory = object.__new__(PatchActivationRuntimeFactory)
    object.__setattr__(factory, "_manifest", manifest)
    object.__setattr__(factory, "_verifier", verifier)
    object.__setattr__(factory, "_issuer", _ACTIVATION_FACTORY_ISSUER)
    _ISSUED_ACTIVATION_FACTORIES.append(factory)
    return factory


def build_patch_runtime_activation_factory() -> PatchActivationRuntimeFactory:
    """Return the fail-closed production activation factory."""
    manifest = build_patch_production_manifest()
    return _build_activation_factory(
        manifest, build_patch_activation_verifier(manifest)
    )


def build_patch_production_manifest() -> PatchProductionManifest:
    """Generate one frozen dormant profile from tracked production symbols."""
    schemas = tuple(
        _schema_description(schema)
        for schema in (PATCH_EDIT_SCHEMA, PATCH_APPLY_SCHEMA)
    )
    profile = PatchCapabilityProfile(
        key=PatchActivationProfileKey(
            context=ContextKind.SANDBOX,
            platform=PatchActivationPlatform.MACOS,
            filesystem=PatchProfileComponent("rooted-no-follow-v1"),
            target_implementation=PatchProfileComponent("sandbox-worker-v1"),
            target_protocol=PatchProfileComponent("sandbox-patch-rpc-v1"),
            policy=PatchProfileComponent("sealed-plan-policy-v1"),
            approval=PatchProfileComponent("durable-plan-approval-v1"),
            persistence=PatchProfileComponent("durable-request-store-v1"),
            surface=PatchActivationSurface.JSON_FUNCTION,
            provider_codec=PatchProfileComponent("json-function-v1"),
            version=PatchProfileComponent("v1"),
        ),
        proofs=PatchProfileProofs(
            context=True,
            platform=False,
            filesystem=True,
            target=True,
            protocol=True,
            policy=True,
            approval=True,
            persistence=True,
            surface=True,
            provider_codec=True,
        ),
        state=PatchProfileState.INCOMPLETE,
        selection_rationale=(
            "The production profile is incomplete until an exact platform "
            "receipt is independently recorded."
        ),
        capability_inventory=tuple(schema.tool_name for schema in schemas),
    )
    sources = (
        PatchProductionSource("avalan.patch.toolset", "PATCH_EDIT_SCHEMA"),
        PatchProductionSource("avalan.patch.toolset", "PATCH_APPLY_SCHEMA"),
        PatchProductionSource("avalan.patch.toolset", "PatchToolSet"),
        PatchProductionSource(
            "avalan.patch.protocols", "PatchProtocolProfile"
        ),
    )
    protocols = tuple(
        PatchProtocolDescription(surface, _protocol_tools(surface))
        for surface in PatchProtocolSurface
    )
    return _manifest(
        sources=sources,
        schemas=schemas,
        protocols=protocols,
        profiles=(profile,),
    )


def render_patch_production_manifest(manifest: PatchProductionManifest) -> str:
    """Render one canonical freeze-check payload without filesystem access."""
    if type(manifest) is not PatchProductionManifest:
        raise PatchActivationError("patch production manifest is invalid")
    return dumps(
        _manifest_wire_payload(manifest),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest(
    *,
    sources: tuple[PatchProductionSource, ...],
    schemas: tuple[PatchSchemaDescription, ...],
    protocols: tuple[PatchProtocolDescription, ...],
    profiles: tuple[PatchCapabilityProfile, ...],
) -> PatchProductionManifest:
    """Construct and seal a manifest after all production sources are known."""
    tool_inventory = tuple(schema.tool_name for schema in schemas)
    return PatchProductionManifest(
        schema_version=_MANIFEST_VERSION,
        sources=sources,
        schemas=schemas,
        tool_inventory=tool_inventory,
        protocols=protocols,
        profiles=profiles,
        manifest_sha256=_digest(
            _manifest_payload_values(
                _MANIFEST_VERSION,
                sources,
                schemas,
                tool_inventory,
                protocols,
                profiles,
            )
        ),
    )


def _schema_description(value: object) -> PatchSchemaDescription:
    """Freeze one exact public tool schema without accepting caller input."""
    canonical_json = dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    match value:
        case {"function": {"name": str(name)}}:
            return PatchSchemaDescription(
                name, canonical_json, _digest(canonical_json)
            )
        case _:
            raise PatchActivationError("patch production schema is invalid")


def _protocol_tools(surface: PatchProtocolSurface) -> tuple[str, ...]:
    """Derive protocol descriptors from the tracked protocol enum source."""
    match surface:
        case PatchProtocolSurface.MCP | PatchProtocolSurface.A2A:
            return ("patch.edit", "patch.apply")
        case (
            PatchProtocolSurface.PROVIDER_FREEFORM
            | PatchProtocolSurface.PROVIDER_NATIVE
        ):
            return ("patch.apply",)
        case (
            PatchProtocolSurface.FLOW
            | PatchProtocolSurface.TASK
            | PatchProtocolSurface.MULTI_AGENT
        ):
            return ()


def _digest(value: object) -> str:
    """Return a stable SHA-256 digest for one canonical JSON-safe value."""
    if type(value) is str:
        encoded = value.encode("utf-8")
    else:
        encoded = dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _profile_key_payload(value: PatchActivationProfileKey) -> dict[str, str]:
    """Return the closed serializable payload for one exact profile key."""
    return {
        "context": value.context.value,
        "platform": value.platform.value,
        "filesystem": value.filesystem.value,
        "target_implementation": value.target_implementation.value,
        "target_protocol": value.target_protocol.value,
        "policy": value.policy.value,
        "approval": value.approval.value,
        "persistence": value.persistence.value,
        "surface": value.surface.value,
        "provider_codec": value.provider_codec.value,
        "version": value.version.value,
    }


def _profile_payload(value: PatchCapabilityProfile) -> object:
    """Return the canonical proof and selection payload for one profile."""
    return {
        "key": _profile_key_payload(value.key),
        "proofs": {
            "context": value.proofs.context,
            "platform": value.proofs.platform,
            "filesystem": value.proofs.filesystem,
            "target": value.proofs.target,
            "protocol": value.proofs.protocol,
            "policy": value.proofs.policy,
            "approval": value.proofs.approval,
            "persistence": value.proofs.persistence,
            "surface": value.proofs.surface,
            "provider_codec": value.proofs.provider_codec,
        },
        "state": value.state.value,
        "selection_rationale": value.selection_rationale,
        "capability_inventory": value.capability_inventory,
    }


def _production_evidence_digest(
    manifest: PatchProductionManifest,
    profile: PatchCapabilityProfile,
) -> str:
    """Return the sealed evidence digest for one tracked profile only."""
    if (
        type(manifest) is not PatchProductionManifest
        or type(profile) is not PatchCapabilityProfile
        or manifest.profile_for(profile.key) != profile
    ):
        raise PatchActivationError("patch activation evidence is invalid")
    return _digest(
        {
            "manifest": manifest.manifest_sha256,
            "profile": profile.digest,
            "profile_key": _profile_key_payload(profile.key),
            "state": profile.state.value,
            "proofs_complete": profile.proven,
        }
    )


def _manifest_payload(value: PatchProductionManifest) -> object:
    """Return the exact sealed portion of one source-derived manifest."""
    return _manifest_payload_values(
        value.schema_version,
        value.sources,
        value.schemas,
        value.tool_inventory,
        value.protocols,
        value.profiles,
    )


def _manifest_payload_values(
    schema_version: int,
    sources: tuple[PatchProductionSource, ...],
    schemas: tuple[PatchSchemaDescription, ...],
    tool_inventory: tuple[str, ...],
    protocols: tuple[PatchProtocolDescription, ...],
    profiles: tuple[PatchCapabilityProfile, ...],
) -> object:
    """Return the sealed payload before appending a manifest digest."""
    return {
        "schema_version": schema_version,
        "sources": [
            {"module": source.module, "symbol": source.symbol}
            for source in sources
        ],
        "schemas": [
            {
                "tool_name": schema.tool_name,
                "canonical_json": schema.canonical_json,
                "schema_sha256": schema.schema_sha256,
            }
            for schema in schemas
        ],
        "tool_inventory": tool_inventory,
        "protocols": [
            {
                "protocol_surface": protocol.protocol_surface.value,
                "capability_inventory": protocol.capability_inventory,
            }
            for protocol in protocols
        ],
        "profiles": [_profile_payload(profile) for profile in profiles],
    }


def _manifest_wire_payload(value: PatchProductionManifest) -> object:
    """Return the public freeze-check payload with its final digest."""
    return {
        "schema_version": value.schema_version,
        "sources": [
            {"module": source.module, "symbol": source.symbol}
            for source in value.sources
        ],
        "schemas": [
            {
                "tool_name": schema.tool_name,
                "canonical_json": schema.canonical_json,
                "schema_sha256": schema.schema_sha256,
            }
            for schema in value.schemas
        ],
        "tool_inventory": value.tool_inventory,
        "protocols": [
            {
                "protocol_surface": protocol.protocol_surface.value,
                "capability_inventory": protocol.capability_inventory,
            }
            for protocol in value.protocols
        ],
        "profiles": [_profile_payload(profile) for profile in value.profiles],
        "manifest_sha256": value.manifest_sha256,
    }
