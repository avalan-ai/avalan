"""Validate and atomically activate native conversation provider evidence."""

from .binding import (
    CapabilityEvidenceState,
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderFamily,
    ProviderLaneBinding,
    ProviderTransport,
    normalize_endpoint,
)
from .errors import (
    ConversationCapabilityError,
    ConversationConflictError,
    ConversationValidationError,
)
from .settings import CompactionOperation, ConversationMode, ReasoningContext
from .value import (
    IntegrityDigest,
    ModelConfigurationRevision,
    ProviderApiRevision,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)

from asyncio import Lock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from itertools import product
from typing import final
from urllib.parse import urlsplit

from packaging.version import InvalidVersion, Version


class ProviderApiForm(StrEnum):
    """Identify one exact native Responses API endpoint form."""

    OPENAI_RESPONSES_V1 = "openai_responses_v1"
    AZURE_OPENAI_V1 = "azure-openai-v1"
    AZURE_OPENAI_V1_PREVIEW = "azure-openai-v1-preview"


AZURE_OPENAI_API_REVISIONS = frozenset(
    {
        ProviderApiRevision(ProviderApiForm.AZURE_OPENAI_V1.value),
        ProviderApiRevision(ProviderApiForm.AZURE_OPENAI_V1_PREVIEW.value),
    }
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationProofSet:
    """Pin deterministic proof identifiers for every activated behavior."""

    transport: tuple[str, ...]
    mode: tuple[str, ...]
    reasoning_context: tuple[str, ...]
    compaction: tuple[str, ...]
    retrieve: tuple[str, ...]
    delete: tuple[str, ...]
    wire: tuple[str, ...]
    public_e2e: tuple[str, ...]
    current_documentation: tuple[str, ...]
    live: tuple[str, ...]

    def __post_init__(self) -> None:
        for values in (
            self.transport,
            self.mode,
            self.reasoning_context,
            self.compaction,
            self.retrieve,
            self.delete,
            self.wire,
            self.public_e2e,
            self.current_documentation,
            self.live,
        ):
            _validate_proof_identifiers(values)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationEvidenceRow:
    """Describe one exact provider behavior cross-product observation."""

    binding_digest: IntegrityDigest
    provider_family: ProviderFamily
    normalized_endpoint: str
    api_form: ProviderApiForm
    provider_api_revision: ProviderApiRevision
    sdk_version: str
    model_or_deployment: str
    model_or_deployment_revision: str
    model_configuration_revision: ModelConfigurationRevision
    transport: ProviderTransport
    mode: ConversationMode
    reasoning_context: ReasoningContext
    compaction_operation: CompactionOperation
    retrieve_supported: bool
    delete_supported: bool
    active: bool
    observed_at: datetime
    valid_until: datetime
    proofs: ActivationProofSet

    def __post_init__(self) -> None:
        _validate_digest(self.binding_digest)
        if (
            not isinstance(self.provider_family, ProviderFamily)
            or not isinstance(self.api_form, ProviderApiForm)
            or not isinstance(self.transport, ProviderTransport)
            or not isinstance(self.mode, ConversationMode)
            or not isinstance(self.reasoning_context, ReasoningContext)
            or not isinstance(
                self.compaction_operation,
                CompactionOperation,
            )
            or type(self.retrieve_supported) is not bool
            or type(self.delete_supported) is not bool
            or type(self.active) is not bool
            or type(self.proofs) is not ActivationProofSet
        ):
            raise ConversationValidationError()
        normalized = normalize_endpoint(self.normalized_endpoint)
        if normalized != self.normalized_endpoint:
            raise ConversationValidationError()
        for value, name in (
            (self.provider_api_revision, "provider_api_revision"),
            (self.model_or_deployment, "model_or_deployment"),
            (
                self.model_or_deployment_revision,
                "model_or_deployment_revision",
            ),
            (
                self.model_configuration_revision,
                "model_configuration_revision",
            ),
        ):
            validate_identifier(value, name)
        _validate_version(self.sdk_version)
        _validate_time_window(self.observed_at, self.valid_until)
        if self.mode not in {
            ConversationMode.STATELESS,
            ConversationMode.STORED,
        } or self.reasoning_context not in {
            ReasoningContext.CURRENT_TURN,
            ReasoningContext.ALL_TURNS,
        }:
            raise ConversationValidationError()
        lifecycle_supported = self.mode is ConversationMode.STORED
        if (
            self.retrieve_supported != lifecycle_supported
            or self.delete_supported != lifecycle_supported
        ):
            raise ConversationValidationError()

    @property
    def cross_product_key(
        self,
    ) -> tuple[
        ProviderTransport,
        ConversationMode,
        ReasoningContext,
        CompactionOperation,
    ]:
        """Return the exact behavior cross-product identity."""
        return (
            self.transport,
            self.mode,
            self.reasoning_context,
            self.compaction_operation,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationManifest:
    """Pin one reviewed, immutable native-provider activation manifest.

    ``review_signature`` is a deployment-pinned content digest. It proves
    exact reviewed bytes when the deployment separately pins the digest; it
    is not a cryptographic signer identity or authenticated signature.
    """

    manifest_id: str
    revision: str
    binding: ProviderLaneBinding
    capability_profile: ConversationCapabilityProfile
    api_form: ProviderApiForm
    minimum_supported_sdk_version: str
    maximum_supported_sdk_version: str
    runtime_sdk_version: str
    model_or_deployment_revision: str
    required_transports: tuple[ProviderTransport, ...]
    required_modes: tuple[ConversationMode, ...]
    required_reasoning_contexts: tuple[ReasoningContext, ...]
    required_compaction_operations: tuple[CompactionOperation, ...]
    evidence: tuple[ActivationEvidenceRow, ...]
    reviewed_by: tuple[str, ...]
    reviewed_at: datetime
    valid_until: datetime
    review_signature: IntegrityDigest = field(init=False)
    integrity_digest: IntegrityDigest = field(init=False)

    def __post_init__(self) -> None:
        validate_identifier(self.manifest_id, "manifest_id")
        validate_identifier(self.revision, "revision")
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.capability_profile)
            is not ConversationCapabilityProfile
            or not isinstance(self.api_form, ProviderApiForm)
        ):
            raise ConversationValidationError()
        _validate_native_provider_form(self.binding, self.api_form)
        self.capability_profile.assert_binding(self.binding)
        if self.capability_profile.test_only:
            raise ConversationValidationError()
        versions = tuple(
            _validate_version(value)
            for value in (
                self.minimum_supported_sdk_version,
                self.maximum_supported_sdk_version,
                self.runtime_sdk_version,
            )
        )
        if not versions[0] <= versions[2] <= versions[1]:
            raise ConversationValidationError()
        if (
            self.binding.sdk_revision
            != f"openai-python-{self.runtime_sdk_version}"
        ):
            raise ConversationValidationError()
        validate_identifier(
            self.model_or_deployment_revision,
            "model_or_deployment_revision",
        )
        _validate_enum_tuple(self.required_transports, ProviderTransport)
        _validate_enum_tuple(self.required_modes, ConversationMode)
        _validate_enum_tuple(
            self.required_reasoning_contexts,
            ReasoningContext,
        )
        _validate_enum_tuple(
            self.required_compaction_operations,
            CompactionOperation,
        )
        if (
            any(
                value
                not in {ConversationMode.STATELESS, ConversationMode.STORED}
                for value in self.required_modes
            )
            or any(
                value
                not in {
                    ReasoningContext.CURRENT_TURN,
                    ReasoningContext.ALL_TURNS,
                }
                for value in self.required_reasoning_contexts
            )
            or self.binding.transport not in self.required_transports
        ):
            raise ConversationValidationError()
        if type(self.evidence) is not tuple or any(
            type(row) is not ActivationEvidenceRow for row in self.evidence
        ):
            raise ConversationValidationError()
        _validate_time_window(self.reviewed_at, self.valid_until)
        _validate_reviewers(self.reviewed_by)
        self._validate_evidence()
        review_signature = IntegrityDigest(
            sha256(self._canonical_bytes(review_signature=None)).hexdigest()
        )
        object.__setattr__(self, "review_signature", review_signature)
        object.__setattr__(
            self,
            "integrity_digest",
            IntegrityDigest(
                sha256(
                    self._canonical_bytes(review_signature=review_signature)
                ).hexdigest()
            ),
        )

    def assert_integrity(self) -> None:
        """Reject a manifest whose immutable digest is inconsistent."""
        expected_signature = sha256(
            self._canonical_bytes(review_signature=None)
        ).hexdigest()
        expected_integrity = sha256(
            self._canonical_bytes(
                review_signature=IntegrityDigest(expected_signature)
            )
        ).hexdigest()
        if not compare_digest(
            self.review_signature,
            expected_signature,
        ) or not compare_digest(self.integrity_digest, expected_integrity):
            raise ConversationValidationError()

    def _validate_evidence(self) -> None:
        expected_keys = set(
            product(
                self.required_transports,
                self.required_modes,
                self.required_reasoning_contexts,
                self.required_compaction_operations,
            )
        )
        actual_keys = tuple(row.cross_product_key for row in self.evidence)
        if (
            set(actual_keys) != expected_keys
            or len(actual_keys) != len(expected_keys)
            or actual_keys
            != tuple(sorted(actual_keys, key=_cross_product_sort))
        ):
            raise ConversationValidationError()
        for row in self.evidence:
            if (
                row.binding_digest != self.binding.integrity_digest
                or row.provider_family is not self.binding.provider_family
                or row.normalized_endpoint != self.binding.normalized_endpoint
                or row.api_form is not self.api_form
                or row.provider_api_revision
                != self.binding.provider_api_revision
                or row.sdk_version != self.runtime_sdk_version
                or row.model_or_deployment != self.binding.model_or_deployment
                or row.model_or_deployment_revision
                != self.model_or_deployment_revision
                or row.model_configuration_revision
                != self.binding.model_configuration_revision
                or row.observed_at > self.reviewed_at
                or row.valid_until < self.valid_until
            ):
                raise ConversationValidationError()
            if row.active:
                self._validate_active_capabilities(row)

    def _validate_active_capabilities(
        self, row: ActivationEvidenceRow
    ) -> None:
        required = {
            (
                ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY
                if row.mode is ConversationMode.STATELESS
                else ConversationCapability.STORED_RESPONSES_CHAINING
            ),
            (
                ConversationCapability.REASONING_CONTEXT_CURRENT_TURN
                if row.reasoning_context is ReasoningContext.CURRENT_TURN
                else ConversationCapability.REASONING_CONTEXT_ALL_TURNS
            ),
        }
        if row.transport is ProviderTransport.STREAMING:
            required.add(ConversationCapability.STREAMING_ITEM_FIDELITY)
        if row.compaction_operation is CompactionOperation.INLINE:
            required.add(ConversationCapability.INLINE_COMPACTION)
        elif row.compaction_operation is CompactionOperation.STANDALONE:
            required.add(ConversationCapability.STANDALONE_COMPACTION)
        if row.mode is ConversationMode.STORED:
            required.update(
                {
                    ConversationCapability.STORED_RESPONSE_RETRIEVAL,
                    ConversationCapability.STORED_RESPONSE_DELETION,
                }
            )
        states = {
            item.capability: item.state
            for item in self.capability_profile.capabilities
        }
        if any(
            states[capability] is not CapabilityEvidenceState.CONFORMANT
            for capability in required
        ):
            raise ConversationValidationError()

    def _canonical_bytes(
        self,
        *,
        review_signature: IntegrityDigest | None,
    ) -> bytes:
        payload = {
            "manifest_id": self.manifest_id,
            "revision": self.revision,
            "binding_digest": self.binding.integrity_digest,
            "capability_profile": {
                "profile_id": self.capability_profile.profile_id,
                "schema_version": self.capability_profile.schema_version,
                "revision": self.capability_profile.revision,
                "binding_alias": self.capability_profile.binding_alias,
                "capabilities": [
                    {
                        "capability": item.capability.value,
                        "state": item.state.value,
                        "evidence_ids": list(item.evidence_ids),
                    }
                    for item in self.capability_profile.capabilities
                ],
                "test_only": self.capability_profile.test_only,
            },
            "api_form": self.api_form.value,
            "minimum_supported_sdk_version": (
                self.minimum_supported_sdk_version
            ),
            "maximum_supported_sdk_version": (
                self.maximum_supported_sdk_version
            ),
            "runtime_sdk_version": self.runtime_sdk_version,
            "model_or_deployment_revision": self.model_or_deployment_revision,
            "required_transports": [
                value.value for value in self.required_transports
            ],
            "required_modes": [value.value for value in self.required_modes],
            "required_reasoning_contexts": [
                value.value for value in self.required_reasoning_contexts
            ],
            "required_compaction_operations": [
                value.value for value in self.required_compaction_operations
            ],
            "evidence": [_row_payload(row) for row in self.evidence],
            "reviewed_by": list(self.reviewed_by),
            "reviewed_at": self.reviewed_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
        }
        if review_signature is not None:
            payload["review_signature"] = review_signature
        return canonical_json_bytes(freeze_json_value(payload))


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationSnapshot:
    """Retain immutable history without granting future dispatch authority."""

    registry_id: str
    generation: int
    active_manifest: ActivationManifest | None
    loaded_manifest_digests: tuple[IntegrityDigest, ...]
    activated_manifest_digests: tuple[IntegrityDigest, ...]
    revoked_manifest_digests: tuple[IntegrityDigest, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.registry_id, "registry_id")
        if type(self.generation) is not int or self.generation < 0:
            raise ConversationValidationError()
        if (
            self.active_manifest is not None
            and type(self.active_manifest) is not ActivationManifest
        ):
            raise ConversationValidationError()
        for values in (
            self.loaded_manifest_digests,
            self.activated_manifest_digests,
            self.revoked_manifest_digests,
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(values))
                or len(values) != len(set(values))
            ):
                raise ConversationValidationError()
            for value in values:
                _validate_digest(value)
        if self.active_manifest is not None and (
            self.active_manifest.integrity_digest
            not in self.loaded_manifest_digests
            or self.active_manifest.integrity_digest
            not in self.activated_manifest_digests
        ):
            raise ConversationValidationError()
        if not set(self.activated_manifest_digests) <= set(
            self.loaded_manifest_digests
        ):
            raise ConversationValidationError()


@final
class AsyncActivationRegistry:
    """Load and atomically switch content-pinned provider manifests."""

    def __init__(
        self,
        *,
        registry_id: str,
        runtime_sdk_version: str,
        trusted_review_signatures: frozenset[IntegrityDigest],
        clock: Callable[[], Awaitable[datetime]],
    ) -> None:
        validate_identifier(registry_id, "registry_id")
        _validate_version(runtime_sdk_version)
        if (
            type(trusted_review_signatures) is not frozenset
            or not trusted_review_signatures
            or not callable(clock)
        ):
            raise ConversationValidationError()
        for signature in trusted_review_signatures:
            _validate_digest(signature)
        self._registry_id = registry_id
        self._runtime_sdk_version = runtime_sdk_version
        self._trusted_review_signatures = trusted_review_signatures
        self._clock = clock
        self._lock = Lock()
        self._generation = 0
        self._loaded: dict[IntegrityDigest, ActivationManifest] = {}
        self._activated: dict[IntegrityDigest, ActivationManifest] = {}
        self._revision_digests: dict[str, IntegrityDigest] = {}
        self._revoked: set[IntegrityDigest] = set()
        self._active: ActivationManifest | None = None

    async def validate(self, manifest: ActivationManifest) -> None:
        """Validate a reviewed manifest against current runtime state."""
        now = await self._now()
        self._validate_at(manifest, now)

    async def load(self, manifest: ActivationManifest) -> ActivationSnapshot:
        """Validate and load a dormant candidate without activating it."""
        now = await self._now()
        self._validate_at(manifest, now)
        async with self._lock:
            prior_digest = self._revision_digests.get(manifest.revision)
            if prior_digest is not None and prior_digest != (
                manifest.integrity_digest
            ):
                raise ConversationConflictError()
            if manifest.integrity_digest not in self._loaded:
                self._loaded[manifest.integrity_digest] = manifest
                self._revision_digests[manifest.revision] = (
                    manifest.integrity_digest
                )
                self._generation += 1
            return self._snapshot_locked()

    async def snapshot(self) -> ActivationSnapshot:
        """Return an immutable point-in-time registry snapshot."""
        async with self._lock:
            return self._snapshot_locked()

    async def apply(
        self,
        manifest_digest: IntegrityDigest,
        *,
        expected_generation: int,
    ) -> ActivationSnapshot:
        """Atomically activate one loaded manifest using generation CAS."""
        _validate_digest(manifest_digest)
        _validate_generation(expected_generation)
        now = await self._now()
        async with self._lock:
            self._require_generation(expected_generation)
            manifest = self._loaded.get(manifest_digest)
            if (
                manifest is None
                or manifest_digest in self._revoked
                or not any(row.active for row in manifest.evidence)
            ):
                raise ConversationCapabilityError()
            self._validate_at(manifest, now)
            self._active = manifest
            self._activated[manifest.integrity_digest] = manifest
            self._generation += 1
            return self._snapshot_locked()

    async def rollback(
        self,
        target: ActivationSnapshot,
        *,
        expected_generation: int,
    ) -> ActivationSnapshot:
        """Atomically restore a prior active manifest or deactivate all."""
        if type(target) is not ActivationSnapshot or (
            target.registry_id != self._registry_id
        ):
            raise ConversationValidationError()
        _validate_generation(expected_generation)
        now = await self._now()
        async with self._lock:
            self._require_generation(expected_generation)
            manifest = target.active_manifest
            if manifest is not None:
                loaded = self._loaded.get(manifest.integrity_digest)
                if (
                    loaded != manifest
                    or self._activated.get(manifest.integrity_digest)
                    != manifest
                    or manifest.integrity_digest in self._revoked
                ):
                    raise ConversationCapabilityError()
                self._validate_at(manifest, now)
            self._active = manifest
            self._generation += 1
            return self._snapshot_locked()

    async def revoke(
        self,
        manifest_digest: IntegrityDigest,
        *,
        expected_generation: int,
    ) -> ActivationSnapshot:
        """Block new dispatch while leaving prior snapshots immutable."""
        _validate_digest(manifest_digest)
        _validate_generation(expected_generation)
        async with self._lock:
            self._require_generation(expected_generation)
            if manifest_digest not in self._loaded:
                raise ConversationValidationError()
            if manifest_digest not in self._revoked:
                self._revoked.add(manifest_digest)
                self._generation += 1
            return self._snapshot_locked()

    async def resolve(
        self,
        binding: ProviderLaneBinding,
        *,
        mode: ConversationMode,
        reasoning_context: ReasoningContext,
        compaction_operation: CompactionOperation,
    ) -> ActivationEvidenceRow:
        """Resolve one active exact row for a new provider dispatch."""
        if (
            type(binding) is not ProviderLaneBinding
            or not isinstance(mode, ConversationMode)
            or not isinstance(reasoning_context, ReasoningContext)
            or not isinstance(compaction_operation, CompactionOperation)
        ):
            raise ConversationValidationError()
        now = await self._now()
        async with self._lock:
            manifest = self._active
            if manifest is None or (
                manifest.integrity_digest in self._revoked
            ):
                raise ConversationCapabilityError()
            manifest.binding.assert_compatible(binding)
            self._validate_at(manifest, now)
            key = (
                binding.transport,
                mode,
                reasoning_context,
                compaction_operation,
            )
            row = next(
                (
                    value
                    for value in manifest.evidence
                    if value.cross_product_key == key
                ),
                None,
            )
            if row is None or not row.active:
                raise ConversationCapabilityError()
            return row

    async def resolve_lifecycle(
        self,
        binding: ProviderLaneBinding,
        *,
        capability: ConversationCapability,
    ) -> ActivationEvidenceRow:
        """Resolve historical stored-state lifecycle compatibility.

        Revocation and rollback stop new provider dispatch, but already
        committed stored responses still need an exact resolver for retrieval
        and deletion during the reviewed compatibility window.
        """
        if type(binding) is not ProviderLaneBinding or capability not in {
            ConversationCapability.STORED_RESPONSE_RETRIEVAL,
            ConversationCapability.STORED_RESPONSE_DELETION,
        }:
            raise ConversationValidationError()
        now = await self._now()
        async with self._lock:
            matches = tuple(
                manifest
                for manifest in self._activated.values()
                if manifest.binding == binding
            )
            if not matches:
                raise ConversationCapabilityError()
            for manifest in sorted(
                matches,
                key=lambda value: (
                    value.reviewed_at,
                    value.integrity_digest,
                ),
                reverse=True,
            ):
                try:
                    self._validate_at(manifest, now)
                except ConversationValidationError:
                    continue
                row = next(
                    (
                        value
                        for value in manifest.evidence
                        if value.active
                        and value.mode is ConversationMode.STORED
                        and (
                            value.retrieve_supported
                            if capability
                            is ConversationCapability.STORED_RESPONSE_RETRIEVAL
                            else value.delete_supported
                        )
                    ),
                    None,
                )
                if row is not None:
                    return row
            raise ConversationCapabilityError()

    async def _now(self) -> datetime:
        now = await self._clock()
        _validate_utc_datetime(now)
        return now

    def _validate_at(
        self,
        manifest: ActivationManifest,
        now: datetime,
    ) -> None:
        if type(manifest) is not ActivationManifest:
            raise ConversationValidationError()
        manifest.assert_integrity()
        if (
            manifest.runtime_sdk_version != self._runtime_sdk_version
            or manifest.review_signature not in self._trusted_review_signatures
            or now < manifest.reviewed_at
            or now >= manifest.valid_until
            or any(
                now < row.observed_at or now >= row.valid_until
                for row in manifest.evidence
            )
        ):
            raise ConversationValidationError()

    def _require_generation(self, expected_generation: int) -> None:
        if expected_generation != self._generation:
            raise ConversationConflictError()

    def _snapshot_locked(self) -> ActivationSnapshot:
        return ActivationSnapshot(
            registry_id=self._registry_id,
            generation=self._generation,
            active_manifest=self._active,
            loaded_manifest_digests=tuple(sorted(self._loaded)),
            activated_manifest_digests=tuple(sorted(self._activated)),
            revoked_manifest_digests=tuple(sorted(self._revoked)),
        )


def _validate_native_provider_form(
    binding: ProviderLaneBinding,
    api_form: ProviderApiForm,
) -> None:
    if binding.provider_family is ProviderFamily.OPENAI:
        if (
            api_form is not ProviderApiForm.OPENAI_RESPONSES_V1
            or binding.normalized_endpoint != "https://api.openai.com/v1"
        ):
            raise ConversationValidationError()
        return
    if binding.provider_family is not ProviderFamily.AZURE_OPENAI or (
        api_form
        not in {
            ProviderApiForm.AZURE_OPENAI_V1,
            ProviderApiForm.AZURE_OPENAI_V1_PREVIEW,
        }
    ):
        raise ConversationValidationError()
    parsed = urlsplit(binding.normalized_endpoint)
    hostname = parsed.hostname
    if (
        parsed.scheme != "https"
        or parsed.port is not None
        or hostname is None
        or not hostname.endswith(".openai.azure.com")
        or hostname == "openai.azure.com"
        or parsed.path != "/openai/v1"
        or binding.azure_resource_identity != hostname
        or binding.provider_api_revision not in AZURE_OPENAI_API_REVISIONS
        or binding.provider_api_revision != api_form.value
    ):
        raise ConversationValidationError()


def _validate_version(value: object) -> Version:
    validate_identifier(value, "sdk_version")
    assert isinstance(value, str)
    try:
        parsed = Version(value)
    except InvalidVersion as exc:
        raise ConversationValidationError() from exc
    if (
        str(parsed) != value
        or len(parsed.release) != 3
        or parsed.epoch != 0
        or parsed.pre is not None
        or parsed.post is not None
        or parsed.dev is not None
        or parsed.local is not None
    ):
        raise ConversationValidationError()
    return parsed


def _validate_digest(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConversationValidationError()


def _validate_utc_datetime(value: object) -> None:
    if (
        not isinstance(value, datetime)
        or value.utcoffset() != timedelta(0)
        or value.isoformat().endswith("+00:00") is False
    ):
        raise ConversationValidationError()


def _validate_time_window(start: datetime, end: datetime) -> None:
    _validate_utc_datetime(start)
    _validate_utc_datetime(end)
    if end <= start:
        raise ConversationValidationError()


def _validate_proof_identifiers(values: object) -> None:
    if (
        type(values) is not tuple
        or not values
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
    ):
        raise ConversationValidationError()
    for value in values:
        validate_identifier(value, "proof_id", max_length=1_024)


def _validate_reviewers(values: object) -> None:
    if (
        type(values) is not tuple
        or not values
        or values != tuple(sorted(values))
        or len(values) != len(set(values))
    ):
        raise ConversationValidationError()
    for value in values:
        validate_identifier(value, "reviewer_id")


def _validate_enum_tuple(values: object, enum_type: type[StrEnum]) -> None:
    if (
        type(values) is not tuple
        or not values
        or any(not isinstance(value, enum_type) for value in values)
        or len(values) != len(set(values))
        or values != tuple(sorted(values, key=lambda value: value.value))
    ):
        raise ConversationValidationError()


def _validate_generation(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ConversationValidationError()


def _cross_product_sort(
    key: tuple[
        ProviderTransport,
        ConversationMode,
        ReasoningContext,
        CompactionOperation,
    ],
) -> tuple[str, str, str, str]:
    return (
        key[0].value,
        key[1].value,
        key[2].value,
        key[3].value,
    )


def _row_payload(row: ActivationEvidenceRow) -> dict[str, object]:
    return {
        "binding_digest": row.binding_digest,
        "provider_family": row.provider_family.value,
        "normalized_endpoint": row.normalized_endpoint,
        "api_form": row.api_form.value,
        "provider_api_revision": row.provider_api_revision,
        "sdk_version": row.sdk_version,
        "model_or_deployment": row.model_or_deployment,
        "model_or_deployment_revision": row.model_or_deployment_revision,
        "model_configuration_revision": row.model_configuration_revision,
        "transport": row.transport.value,
        "mode": row.mode.value,
        "reasoning_context": row.reasoning_context.value,
        "compaction_operation": row.compaction_operation.value,
        "retrieve_supported": row.retrieve_supported,
        "delete_supported": row.delete_supported,
        "active": row.active,
        "observed_at": row.observed_at.isoformat(),
        "valid_until": row.valid_until.isoformat(),
        "proofs": {
            "transport": list(row.proofs.transport),
            "mode": list(row.proofs.mode),
            "reasoning_context": list(row.proofs.reasoning_context),
            "compaction": list(row.proofs.compaction),
            "retrieve": list(row.proofs.retrieve),
            "delete": list(row.proofs.delete),
            "wire": list(row.proofs.wire),
            "public_e2e": list(row.proofs.public_e2e),
            "current_documentation": list(row.proofs.current_documentation),
            "live": list(row.proofs.live),
        },
    }
