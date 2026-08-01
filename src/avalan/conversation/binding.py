"""Bind provider lanes to exact capability and execution identities."""

from .contract import ConversationAgentId, ProviderLaneId
from .errors import (
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationValidationError,
)
from .value import (
    CapabilityProfileId,
    CapabilityProfileRevision,
    ConversationCodecVersion,
    ExecutionDefinitionRevision,
    IntegrityDigest,
    ModelConfigurationRevision,
    ProviderApiRevision,
    ProviderSdkRevision,
    SafeAlias,
    ToolSchemaRevision,
    validate_identifier,
    validate_revision,
)

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import final
from urllib.parse import SplitResult, urlsplit, urlunsplit


class ProviderFamily(StrEnum):
    """Identify one closed provider family."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    SYNTHETIC = "synthetic"


class ProviderTransport(StrEnum):
    """Identify the independently conformed provider transport."""

    NON_STREAMING = "non_streaming"
    STREAMING = "streaming"


class ConversationCapability(StrEnum):
    """Identify one separately evidenced conversation capability."""

    STORED_RESPONSES_CHAINING = "stored_responses_chaining"
    STATELESS_ENCRYPTED_REASONING_REPLAY = (
        "stateless_encrypted_reasoning_replay"
    )
    REASONING_CONTEXT_CURRENT_TURN = "reasoning_context_current_turn"
    REASONING_CONTEXT_ALL_TURNS = "reasoning_context_all_turns"
    INLINE_COMPACTION = "inline_compaction"
    STANDALONE_COMPACTION = "standalone_compaction"
    STREAMING_ITEM_FIDELITY = "streaming_item_fidelity"
    STORED_RESPONSE_RETRIEVAL = "stored_response_retrieval"
    STORED_RESPONSE_DELETION = "stored_response_deletion"


class CapabilityEvidenceState(StrEnum):
    """Identify the evidence state of one exact capability profile."""

    INCAPABLE = "incapable"
    DORMANT = "dormant"
    TEST_ONLY = "test_only"
    CONFORMANT = "conformant"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderLaneBinding:
    """Bind continuation state to every provider and execution revision."""

    lane_id: ProviderLaneId
    adapter_type: str
    provider_family: ProviderFamily
    normalized_endpoint: str
    model_or_deployment: str
    provider_api_revision: ProviderApiRevision
    sdk_revision: ProviderSdkRevision
    model_configuration_revision: ModelConfigurationRevision
    capability_profile_revision: CapabilityProfileRevision
    tool_schema_revision: ToolSchemaRevision
    execution_definition_revision: ExecutionDefinitionRevision
    continuation_codec_version: ConversationCodecVersion
    transport: ProviderTransport
    agent_id: ConversationAgentId
    azure_resource_identity: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.lane_id, "lane_id")
        validate_identifier(self.adapter_type, "adapter_type")
        if not isinstance(self.provider_family, ProviderFamily):
            raise ConversationValidationError()
        normalized = normalize_endpoint(self.normalized_endpoint)
        if normalized != self.normalized_endpoint:
            raise ConversationValidationError()
        validate_identifier(self.model_or_deployment, "model_or_deployment")
        for value, name in (
            (self.provider_api_revision, "provider_api_revision"),
            (self.sdk_revision, "sdk_revision"),
            (
                self.model_configuration_revision,
                "model_configuration_revision",
            ),
            (self.capability_profile_revision, "capability_profile_revision"),
            (self.tool_schema_revision, "tool_schema_revision"),
            (
                self.execution_definition_revision,
                "execution_definition_revision",
            ),
            (self.agent_id, "agent_id"),
        ):
            validate_identifier(value, name)
        validate_revision(
            self.continuation_codec_version,
            "continuation_codec_version",
        )
        if not isinstance(self.transport, ProviderTransport):
            raise ConversationValidationError()
        if self.provider_family is ProviderFamily.AZURE_OPENAI:
            if self.azure_resource_identity is None:
                raise ConversationValidationError()
            validate_identifier(
                self.azure_resource_identity,
                "azure_resource_identity",
            )
            if (
                self.azure_resource_identity
                != self.azure_resource_identity.casefold()
            ):
                raise ConversationValidationError()
        elif self.azure_resource_identity is not None:
            raise ConversationValidationError()

    @property
    def safe_alias(self) -> SafeAlias:
        """Return a stable non-secret alias for observability."""
        digest = sha256(self._canonical_identity().encode("utf-8")).hexdigest()
        return SafeAlias(f"lane-binding-{digest[:16]}")

    @property
    def integrity_digest(self) -> IntegrityDigest:
        """Return the full content-safe digest of the exact binding."""
        return IntegrityDigest(
            sha256(self._canonical_identity().encode("utf-8")).hexdigest()
        )

    def assert_compatible(self, current: "ProviderLaneBinding") -> None:
        """Reject any provider or execution identity drift."""
        if type(current) is not ProviderLaneBinding or current != self:
            raise ConversationBindingDriftError()

    def _canonical_identity(self) -> str:
        values = (
            self.adapter_type,
            self.provider_family.value,
            self.normalized_endpoint,
            self.azure_resource_identity or "",
            self.model_or_deployment,
            self.provider_api_revision,
            self.sdk_revision,
            self.model_configuration_revision,
            self.capability_profile_revision,
            self.tool_schema_revision,
            self.execution_definition_revision,
            str(self.continuation_codec_version),
            self.transport.value,
            self.agent_id,
        )
        return "".join(f"{len(value)}:{value}" for value in values)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityEvidence:
    """Record one capability state and its exact evidence identifiers."""

    capability: ConversationCapability
    state: CapabilityEvidenceState
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.capability, ConversationCapability
        ) or not isinstance(
            self.state,
            CapabilityEvidenceState,
        ):
            raise ConversationValidationError()
        if type(self.evidence_ids) is not tuple:
            raise ConversationValidationError()
        for value in self.evidence_ids:
            validate_identifier(value, "evidence_id")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ConversationValidationError()
        if (
            self.state
            in {
                CapabilityEvidenceState.TEST_ONLY,
                CapabilityEvidenceState.CONFORMANT,
            }
            and not self.evidence_ids
        ):
            raise ConversationValidationError()
        if (
            self.state
            in {
                CapabilityEvidenceState.DORMANT,
                CapabilityEvidenceState.INCAPABLE,
            }
            and self.evidence_ids
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationCapabilityProfile:
    """Describe independently versioned evidence for one exact lane binding."""

    profile_id: CapabilityProfileId
    schema_version: int
    revision: CapabilityProfileRevision
    binding_alias: SafeAlias
    capabilities: tuple[CapabilityEvidence, ...]
    test_only: bool

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, "profile_id")
        validate_revision(self.schema_version, "schema_version")
        if self.schema_version == 0:
            raise ConversationValidationError()
        validate_identifier(self.revision, "revision")
        validate_identifier(self.binding_alias, "binding_alias")
        if (
            type(self.capabilities) is not tuple
            or type(self.test_only) is not bool
        ):
            raise ConversationValidationError()
        names = tuple(evidence.capability for evidence in self.capabilities)
        if len(names) != len(ConversationCapability) or set(names) != set(
            ConversationCapability
        ):
            raise ConversationValidationError()
        if any(
            type(item) is not CapabilityEvidence for item in self.capabilities
        ):
            raise ConversationValidationError()
        if not self.test_only and any(
            item.state is CapabilityEvidenceState.TEST_ONLY
            for item in self.capabilities
        ):
            raise ConversationValidationError()

    def require(self, capability: ConversationCapability) -> None:
        """Reject unsupported, dormant, or non-production capability use."""
        if not isinstance(capability, ConversationCapability):
            raise ConversationValidationError()
        evidence = next(
            item for item in self.capabilities if item.capability is capability
        )
        allowed = (
            evidence.state is CapabilityEvidenceState.CONFORMANT
            or self.test_only
            and evidence.state is CapabilityEvidenceState.TEST_ONLY
        )
        if not allowed:
            raise ConversationCapabilityError()

    def assert_binding(self, binding: ProviderLaneBinding) -> None:
        """Reject a profile attached to a different exact binding."""
        if type(binding) is not ProviderLaneBinding:
            raise ConversationValidationError()
        if binding.safe_alias != self.binding_alias:
            raise ConversationBindingDriftError()
        if binding.capability_profile_revision != self.revision:
            raise ConversationBindingDriftError()


def normalize_endpoint(value: str) -> str:
    """Return a credential-free normalized provider endpoint."""
    validate_identifier(value, "endpoint", max_length=2_048)
    parsed = urlsplit(value)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConversationValidationError()
    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConversationValidationError() from exc
    if port is not None and not (
        scheme == "https" and port == 443 or scheme == "http" and port == 80
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    path = parsed.path.rstrip("/")
    normalized = SplitResult(scheme, netloc, path, "", "")
    return urlunsplit(normalized)
