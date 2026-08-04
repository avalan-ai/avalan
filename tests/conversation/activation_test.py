"""Verify strict native-provider activation and atomic registry behavior."""

from asyncio import gather
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
import pytest
from native_openai_provider_test import (
    _binding as _native_binding,
)
from native_openai_provider_test import (
    _capabilities as _native_capabilities,
)
from native_openai_provider_test import _message as _native_message
from native_openai_provider_test import _plan as _native_plan
from native_openai_provider_test import _profile as _native_profile
from native_openai_provider_test import _response as _native_response
from openai import AsyncOpenAI

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)
OBSERVED = NOW - timedelta(hours=1)
EXPIRES = NOW + timedelta(days=1)


@pytest.fixture
def anyio_backend() -> str:
    """Run activation effects on asyncio only."""
    return "asyncio"


async def _clock() -> datetime:
    return NOW


def _binding(
    *,
    family: conversation.ProviderFamily = conversation.ProviderFamily.OPENAI,
    endpoint: str | None = None,
    api_revision: str | None = None,
    transport: conversation.ProviderTransport = (
        conversation.ProviderTransport.NON_STREAMING
    ),
) -> conversation.ProviderLaneBinding:
    azure = family is conversation.ProviderFamily.AZURE_OPENAI
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId("lane-activation"),
        adapter_type=(
            "avalan.conversation.providers.openai.NativeOpenAIProvider"
        ),
        provider_family=family,
        normalized_endpoint=endpoint
        or (
            "https://resource.openai.azure.com/openai/v1"
            if azure
            else "https://api.openai.com/v1"
        ),
        azure_resource_identity=(
            "resource.openai.azure.com" if azure else None
        ),
        model_or_deployment="deployment-live" if azure else "gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            api_revision
            or ("azure-openai-v1-preview" if azure else "openapi-2.3.0")
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-live-1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-live-1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-live-1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-live-1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=transport,
        agent_id=conversation.ConversationAgentId("agent-activation"),
    )


def _profile(
    binding: conversation.ProviderLaneBinding,
    *,
    incapable: conversation.ConversationCapability | None = None,
    test_only: bool = False,
) -> conversation.ConversationCapabilityProfile:
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId("profile-activation"),
        schema_version=1,
        revision=binding.capability_profile_revision,
        binding_alias=binding.safe_alias,
        capabilities=tuple(
            conversation.CapabilityEvidence(
                capability=capability,
                state=(
                    conversation.CapabilityEvidenceState.INCAPABLE
                    if capability is incapable
                    else (
                        conversation.CapabilityEvidenceState.TEST_ONLY
                        if test_only
                        else conversation.CapabilityEvidenceState.CONFORMANT
                    )
                ),
                evidence_ids=(
                    ()
                    if capability is incapable
                    else (f"capability-proof-{capability.value}",)
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=test_only,
    )


def _proofs(**changes: object) -> conversation.ActivationProofSet:
    values: dict[str, object] = {
        "transport": ("transport-proof",),
        "mode": ("mode-proof",),
        "reasoning_context": ("reasoning-proof",),
        "compaction": ("compaction-proof",),
        "retrieve": ("retrieve-proof",),
        "delete": ("delete-proof",),
        "wire": ("wire-proof",),
        "public_e2e": ("public-e2e-proof",),
        "current_documentation": ("docs-2026-08-03",),
        "live": ("live-proof-2026-08-03",),
    }
    values.update(changes)
    return conversation.ActivationProofSet(**cast(Any, values))


def _row(
    binding: conversation.ProviderLaneBinding,
    *,
    api_form: conversation.ProviderApiForm,
    mode: conversation.ConversationMode,
    reasoning_context: conversation.ReasoningContext,
    compaction_operation: conversation.CompactionOperation,
    active: bool = True,
    valid_until: datetime = EXPIRES,
) -> conversation.ActivationEvidenceRow:
    stored = mode is conversation.ConversationMode.STORED
    return conversation.ActivationEvidenceRow(
        binding_digest=binding.integrity_digest,
        provider_family=binding.provider_family,
        normalized_endpoint=binding.normalized_endpoint,
        api_form=api_form,
        provider_api_revision=binding.provider_api_revision,
        sdk_version="2.42.0",
        model_or_deployment=binding.model_or_deployment,
        model_or_deployment_revision="provider-model-revision-1",
        model_configuration_revision=binding.model_configuration_revision,
        transport=binding.transport,
        mode=mode,
        reasoning_context=reasoning_context,
        compaction_operation=compaction_operation,
        retrieve_supported=stored,
        delete_supported=stored,
        active=active,
        observed_at=OBSERVED,
        valid_until=valid_until,
        proofs=_proofs(),
    )


def _manifest(
    *,
    binding: conversation.ProviderLaneBinding | None = None,
    active: bool = True,
    profile: conversation.ConversationCapabilityProfile | None = None,
    reviewed_at: datetime = NOW,
    valid_until: datetime = EXPIRES,
    revision: str = "activation-revision-1",
) -> conversation.ActivationManifest:
    selected = binding or _binding()
    api_form = _api_form(selected)
    modes = (
        conversation.ConversationMode.STATELESS,
        conversation.ConversationMode.STORED,
    )
    contexts = (
        conversation.ReasoningContext.ALL_TURNS,
        conversation.ReasoningContext.CURRENT_TURN,
    )
    compactions = (
        conversation.CompactionOperation.INLINE,
        conversation.CompactionOperation.NONE,
        conversation.CompactionOperation.STANDALONE,
    )
    rows = tuple(
        sorted(
            (
                _row(
                    selected,
                    api_form=api_form,
                    mode=mode,
                    reasoning_context=context,
                    compaction_operation=compaction,
                    active=active,
                    valid_until=valid_until,
                )
                for mode in modes
                for context in contexts
                for compaction in compactions
            ),
            key=lambda row: tuple(
                value.value for value in row.cross_product_key
            ),
        )
    )
    return conversation.ActivationManifest(
        manifest_id="manifest-native-conversation",
        revision=revision,
        binding=selected,
        capability_profile=profile or _profile(selected),
        api_form=api_form,
        minimum_supported_sdk_version="2.40.0",
        maximum_supported_sdk_version="2.45.0",
        runtime_sdk_version="2.42.0",
        model_or_deployment_revision="provider-model-revision-1",
        required_transports=(selected.transport,),
        required_modes=modes,
        required_reasoning_contexts=contexts,
        required_compaction_operations=compactions,
        evidence=rows,
        reviewed_by=("architecture-review", "security-review"),
        reviewed_at=reviewed_at,
        valid_until=valid_until,
    )


def _registry(
    *trusted_manifests: conversation.ActivationManifest,
    runtime_sdk_version: str = "2.42.0",
    clock: Callable[[], Awaitable[datetime]] = _clock,
) -> conversation.AsyncActivationRegistry:
    manifests = trusted_manifests or (_manifest(),)
    return conversation.AsyncActivationRegistry(
        registry_id="registry-native-conversation",
        runtime_sdk_version=runtime_sdk_version,
        trusted_review_signatures=frozenset(
            manifest.review_signature for manifest in manifests
        ),
        clock=clock,
    )


def _api_form(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ProviderApiForm:
    if binding.provider_family is conversation.ProviderFamily.OPENAI:
        return conversation.ProviderApiForm.OPENAI_RESPONSES_V1
    return conversation.ProviderApiForm(binding.provider_api_revision)


def _production_profile(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ConversationCapabilityProfile:
    source = _native_capabilities(binding)
    return replace(
        source,
        capabilities=tuple(
            replace(
                evidence,
                state=(
                    conversation.CapabilityEvidenceState.CONFORMANT
                    if evidence.state
                    is conversation.CapabilityEvidenceState.TEST_ONLY
                    else evidence.state
                ),
            )
            for evidence in source.capabilities
        ),
        test_only=False,
    )


def _production_stateless_manifest(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ActivationManifest:
    row = _row(
        binding,
        api_form=_api_form(binding),
        mode=conversation.ConversationMode.STATELESS,
        reasoning_context=conversation.ReasoningContext.CURRENT_TURN,
        compaction_operation=conversation.CompactionOperation.NONE,
    )
    return conversation.ActivationManifest(
        manifest_id="manifest-native-stateless",
        revision="activation-stateless-revision-1",
        binding=binding,
        capability_profile=_production_profile(binding),
        api_form=_api_form(binding),
        minimum_supported_sdk_version="2.40.0",
        maximum_supported_sdk_version="2.45.0",
        runtime_sdk_version="2.42.0",
        model_or_deployment_revision="provider-model-revision-1",
        required_transports=(binding.transport,),
        required_modes=(conversation.ConversationMode.STATELESS,),
        required_reasoning_contexts=(
            conversation.ReasoningContext.CURRENT_TURN,
        ),
        required_compaction_operations=(
            conversation.CompactionOperation.NONE,
        ),
        evidence=(row,),
        reviewed_by=("architecture-review", "security-review"),
        reviewed_at=NOW,
        valid_until=EXPIRES,
    )


def _production_stored_profile(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STORED_RESPONSES_CHAINING,
        conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL,
        conversation.ConversationCapability.STORED_RESPONSE_DELETION,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"activation-stored-{binding.lane_id}"
        ),
        schema_version=1,
        revision=binding.capability_profile_revision,
        binding_alias=binding.safe_alias,
        capabilities=tuple(
            conversation.CapabilityEvidence(
                capability=capability,
                state=(
                    conversation.CapabilityEvidenceState.CONFORMANT
                    if capability in supported
                    else conversation.CapabilityEvidenceState.INCAPABLE
                ),
                evidence_ids=(
                    (f"activation-stored-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=False,
    )


def _stored_execution() -> conversation.NativeOpenAIStoredExecution:
    return conversation.NativeOpenAIStoredExecution(
        instructions="Use the activation-bound stored execution.",
        max_output_tokens=512,
        max_tool_calls=4,
        parallel_tool_calls=False,
        temperature=0.2,
        top_p=0.8,
        truncation="disabled",
    )


def _native_stored_binding(
    *,
    lane_id: str,
    azure: bool = False,
) -> conversation.ProviderLaneBinding:
    binding = conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=(
            "avalan.conversation.providers.openai_stored."
            "NativeOpenAIStoredProvider"
        ),
        provider_family=(
            conversation.ProviderFamily.AZURE_OPENAI
            if azure
            else conversation.ProviderFamily.OPENAI
        ),
        normalized_endpoint=(
            "https://resource.openai.azure.com/openai/v1"
            if azure
            else "https://api.openai.com/v1"
        ),
        azure_resource_identity=(
            "resource.openai.azure.com" if azure else None
        ),
        model_or_deployment="deployment-stored" if azure else "gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1" if azure else "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=conversation.ModelConfigurationRevision(
            "model-config-activation-stored"
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision(
                "capability-activation-stored"
            )
        ),
        tool_schema_revision=conversation.ToolSchemaRevision(
            "tools-activation-stored"
        ),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision(
                "execution-activation-stored"
            )
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId("agent-activation-stored"),
    )
    return replace(
        binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=binding,
                execution=_stored_execution(),
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
                    if azure
                    else (
                        conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                    )
                ),
            )
        ),
    )


def _native_stored_profile(
    binding: conversation.ProviderLaneBinding,
) -> conversation.NativeOpenAIStoredProfile:
    return conversation.NativeOpenAIStoredProfile(
        profile_id="activation-stored-provider",
        binding=binding,
        execution=_stored_execution(),
        encrypted_content=(
            conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            if binding.provider_family
            is conversation.ProviderFamily.AZURE_OPENAI
            else conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
        ),
    )


def _production_stored_manifest(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ActivationManifest:
    row = _row(
        binding,
        api_form=_api_form(binding),
        mode=conversation.ConversationMode.STORED,
        reasoning_context=conversation.ReasoningContext.CURRENT_TURN,
        compaction_operation=conversation.CompactionOperation.NONE,
    )
    return conversation.ActivationManifest(
        manifest_id="manifest-native-stored",
        revision="activation-stored-revision-1",
        binding=binding,
        capability_profile=_production_stored_profile(binding),
        api_form=_api_form(binding),
        minimum_supported_sdk_version="2.40.0",
        maximum_supported_sdk_version="2.45.0",
        runtime_sdk_version="2.42.0",
        model_or_deployment_revision="provider-model-revision-1",
        required_transports=(binding.transport,),
        required_modes=(conversation.ConversationMode.STORED,),
        required_reasoning_contexts=(
            conversation.ReasoningContext.CURRENT_TURN,
        ),
        required_compaction_operations=(
            conversation.CompactionOperation.NONE,
        ),
        evidence=(row,),
        reviewed_by=("architecture-review", "security-review"),
        reviewed_at=NOW,
        valid_until=EXPIRES,
    )


def test_manifest_is_immutable_and_digest_is_deterministic() -> None:
    """Keep reviewed manifests immutable and content-addressed."""
    first = _manifest()
    second = _manifest()

    assert first.integrity_digest == second.integrity_digest
    first.assert_integrity()
    with pytest.raises(FrozenInstanceError):
        first.revision = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.evidence[0].active = False  # type: ignore[misc]


def test_manifest_integrity_covers_every_capability_profile_field() -> None:
    """Bind every semantic capability-profile field into reviewed content."""
    source = _manifest().capability_profile
    mutations: tuple[tuple[str, object], ...] = (
        ("profile_id", conversation.CapabilityProfileId("other-profile")),
        ("schema_version", 2),
        (
            "revision",
            conversation.CapabilityProfileRevision("other-revision"),
        ),
        ("binding_alias", conversation.SafeAlias("other-binding-alias")),
        ("capabilities", tuple(reversed(source.capabilities))),
        ("test_only", True),
    )

    for field_name, value in mutations:
        manifest = _manifest()
        object.__setattr__(
            manifest.capability_profile,
            field_name,
            value,
        )
        with pytest.raises(conversation.ConversationValidationError):
            manifest.assert_integrity()


def test_streaming_manifest_requires_streaming_capability() -> None:
    """Exercise separately reviewed streaming transport evidence."""
    binding = _binding(transport=conversation.ProviderTransport.STREAMING)

    manifest = _manifest(binding=binding)

    assert manifest.required_transports == (
        conversation.ProviderTransport.STREAMING,
    )


def test_native_openai_and_azure_forms_are_accepted() -> None:
    """Accept only exact native OpenAI and Azure endpoint forms."""
    openai = _manifest()
    azure = _manifest(
        binding=_binding(family=conversation.ProviderFamily.AZURE_OPENAI)
    )

    assert openai.api_form is conversation.ProviderApiForm.OPENAI_RESPONSES_V1
    assert azure.api_form is (
        conversation.ProviderApiForm.AZURE_OPENAI_V1_PREVIEW
    )


@pytest.mark.parametrize(
    ("binding", "api_form"),
    (
        (
            _binding(
                family=conversation.ProviderFamily.OPENAI_COMPATIBLE,
                endpoint="https://compatible.example/v1",
            ),
            conversation.ProviderApiForm.OPENAI_RESPONSES_V1,
        ),
        (
            _binding(
                family=conversation.ProviderFamily.SYNTHETIC,
                endpoint="https://synthetic.example/v1",
            ),
            conversation.ProviderApiForm.OPENAI_RESPONSES_V1,
        ),
        (
            _binding(endpoint="https://api.openai.com/v2"),
            conversation.ProviderApiForm.OPENAI_RESPONSES_V1,
        ),
        (
            _binding(),
            conversation.ProviderApiForm.AZURE_OPENAI_V1_PREVIEW,
        ),
        (
            _binding(
                family=conversation.ProviderFamily.AZURE_OPENAI,
                endpoint="https://resource.openai.azure.com/v1",
            ),
            conversation.ProviderApiForm.AZURE_OPENAI_V1_PREVIEW,
        ),
        (
            _binding(
                family=conversation.ProviderFamily.AZURE_OPENAI,
                api_revision="2026-01-01",
            ),
            conversation.ProviderApiForm.AZURE_OPENAI_V1_PREVIEW,
        ),
    ),
)
def test_non_native_or_wrong_api_forms_are_rejected(
    binding: conversation.ProviderLaneBinding,
    api_form: conversation.ProviderApiForm,
) -> None:
    """Fail closed on generic-compatible and malformed native identities."""
    source = _manifest()

    with pytest.raises(conversation.ConversationValidationError):
        conversation.ActivationManifest(
            manifest_id=source.manifest_id,
            revision=source.revision,
            binding=binding,
            capability_profile=_profile(binding),
            api_form=api_form,
            minimum_supported_sdk_version="2.40.0",
            maximum_supported_sdk_version="2.45.0",
            runtime_sdk_version="2.42.0",
            model_or_deployment_revision="provider-model-revision-1",
            required_transports=(binding.transport,),
            required_modes=source.required_modes,
            required_reasoning_contexts=source.required_reasoning_contexts,
            required_compaction_operations=(
                source.required_compaction_operations
            ),
            evidence=source.evidence,
            reviewed_by=source.reviewed_by,
            reviewed_at=source.reviewed_at,
            valid_until=source.valid_until,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_supported_sdk_version", "2.42"),
        ("maximum_supported_sdk_version", "not-a-version"),
        ("runtime_sdk_version", "2.99.0"),
        ("runtime_sdk_version", "2.39.0"),
        ("required_modes", (conversation.ConversationMode.OFF,)),
        ("required_reasoning_contexts", (conversation.ReasoningContext.AUTO,)),
        (
            "required_compaction_operations",
            (
                conversation.CompactionOperation.NONE,
                conversation.CompactionOperation.INLINE,
            ),
        ),
        (
            "reviewed_by",
            ("security-review", "architecture-review"),
        ),
    ),
)
def test_manifest_metadata_validation_is_strict(
    field: str,
    value: object,
) -> None:
    """Reject noncanonical SDK ranges, dimensions, and review evidence."""
    source = _manifest()

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, **cast(Any, {field: value}))


@pytest.mark.parametrize(
    "field",
    (
        "transport",
        "mode",
        "reasoning_context",
        "compaction",
        "retrieve",
        "delete",
        "wire",
        "public_e2e",
        "current_documentation",
        "live",
    ),
)
def test_every_proof_category_is_mandatory(field: str) -> None:
    """Require deterministic identifiers for every evidence category."""
    with pytest.raises(conversation.ConversationValidationError):
        _proofs(**{field: ()})


@pytest.mark.parametrize(
    "bad_values",
    (
        ("z-proof", "a-proof"),
        ("same-proof", "same-proof"),
        (" proof",),
    ),
)
def test_proof_identifiers_are_canonical(bad_values: tuple[str, ...]) -> None:
    """Reject unordered, duplicate, and unnormalized proof identifiers."""
    with pytest.raises(conversation.ConversationValidationError):
        _proofs(live=bad_values)


@pytest.mark.parametrize(
    "changes",
    (
        {"mode": conversation.ConversationMode.OFF},
        {"reasoning_context": conversation.ReasoningContext.AUTO},
        {"retrieve_supported": True},
        {"delete_supported": True},
        {"observed_at": EXPIRES},
        {"valid_until": OBSERVED},
        {"sdk_version": "2.42"},
        {"binding_digest": conversation.IntegrityDigest("f" * 63)},
    ),
)
def test_evidence_row_rejects_invalid_behavior(
    changes: dict[str, object],
) -> None:
    """Reject invalid modes, lifecycle claims, time, SDK, and digests."""
    source = _manifest().evidence[0]

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, **cast(Any, changes))


def test_evidence_row_rejects_wrong_types_and_endpoint_normalization() -> None:
    """Reject typed-field substitution and noncanonical endpoints."""
    source = _manifest().evidence[0]

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, active=cast(Any, 1))
    with pytest.raises(conversation.ConversationValidationError):
        replace(source, normalized_endpoint="https://api.openai.com/v1/")


def test_manifest_rejects_partial_duplicate_and_unordered_cross_products() -> (
    None
):
    """Require one row for every declared cross-product combination."""
    source = _manifest()

    for evidence in (
        source.evidence[:-1],
        source.evidence[:-1] + (source.evidence[0],),
        tuple(reversed(source.evidence)),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(source, evidence=evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("binding_digest", conversation.IntegrityDigest("a" * 64)),
        ("normalized_endpoint", "https://api.openai.com/v2"),
        ("provider_api_revision", conversation.ProviderApiRevision("other")),
        ("sdk_version", "2.43.0"),
        ("model_or_deployment", "gpt-other"),
        ("model_or_deployment_revision", "other-revision"),
        (
            "model_configuration_revision",
            conversation.ModelConfigurationRevision("other-config"),
        ),
    ),
)
def test_manifest_rejects_inconsistent_evidence(
    field: str,
    value: object,
) -> None:
    """Reject evidence drift from the exact reviewed provider binding."""
    source = _manifest()
    evidence = (
        replace(
            source.evidence[0],
            **cast(Any, {field: value}),
        ),
    ) + source.evidence[1:]

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, evidence=evidence)


def test_active_rows_require_conformant_capabilities() -> None:
    """Reject active rows backed by incapable or test-only profiles."""
    binding = _binding()
    profile = _profile(
        binding,
        incapable=conversation.ConversationCapability.INLINE_COMPACTION,
    )

    with pytest.raises(conversation.ConversationValidationError):
        _manifest(binding=binding, profile=profile)
    with pytest.raises(conversation.ConversationValidationError):
        _manifest(binding=binding, profile=_profile(binding, test_only=True))


def test_manifest_rejects_wrong_types_sdk_binding_and_evidence_container() -> (
    None
):
    """Reject malformed top-level types, SDK pins, and row containers."""
    source = _manifest()
    drifted_binding = replace(
        source.binding,
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.43.0"),
    )

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, api_form=cast(Any, "openai_responses_v1"))
    with pytest.raises(conversation.ConversationValidationError):
        _manifest(binding=drifted_binding)
    with pytest.raises(conversation.ConversationValidationError):
        replace(source, evidence=cast(Any, list(source.evidence)))


@pytest.mark.parametrize(
    "changes",
    (
        {"generation": -1},
        {"active_manifest": cast(Any, object())},
        {"loaded_manifest_digests": cast(Any, [])},
        {
            "loaded_manifest_digests": (
                conversation.IntegrityDigest("b" * 64),
                conversation.IntegrityDigest("a" * 64),
            )
        },
        {
            "activated_manifest_digests": (
                conversation.IntegrityDigest("a" * 64),
            )
        },
    ),
)
def test_snapshot_rejects_invalid_state(changes: dict[str, object]) -> None:
    """Keep historical activation snapshots exact and deterministic."""
    source = conversation.ActivationSnapshot(
        registry_id="snapshot-registry",
        generation=0,
        active_manifest=None,
        loaded_manifest_digests=(),
        activated_manifest_digests=(),
        revoked_manifest_digests=(),
    )

    with pytest.raises(conversation.ConversationValidationError):
        replace(source, **cast(Any, changes))


def test_snapshot_requires_active_manifest_to_be_loaded() -> None:
    """Reject snapshots that detach active content from loaded history."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ActivationSnapshot(
            registry_id="snapshot-registry",
            generation=1,
            active_manifest=_manifest(),
            loaded_manifest_digests=(),
            activated_manifest_digests=(),
            revoked_manifest_digests=(),
        )


@pytest.mark.parametrize(
    ("signatures", "clock"),
    (
        (frozenset(), _clock),
        (cast(Any, set()), _clock),
        (frozenset({conversation.IntegrityDigest("a" * 63)}), _clock),
        (frozenset({_manifest().review_signature}), cast(Any, object())),
    ),
)
def test_registry_configuration_is_strict(
    signatures: frozenset[conversation.IntegrityDigest],
    clock: Callable[[], Awaitable[datetime]],
) -> None:
    """Reject empty trust, malformed signatures, and non-callable clocks."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AsyncActivationRegistry(
            registry_id="registry-invalid",
            runtime_sdk_version="2.42.0",
            trusted_review_signatures=signatures,
            clock=clock,
        )


async def test_registry_starts_dormant_and_load_does_not_activate() -> None:
    """Keep all provider activation dormant until an explicit CAS apply."""
    registry = _registry()
    initial = await registry.snapshot()

    loaded = await registry.load(_manifest())

    assert initial.generation == 0
    assert initial.active_manifest is None
    assert loaded.generation == 1
    assert loaded.active_manifest is None
    assert len(loaded.loaded_manifest_digests) == 1
    assert loaded.activated_manifest_digests == ()


@pytest.mark.parametrize(
    ("manifest", "trusted"),
    (
        (_manifest(revision="untrusted-revision"), False),
        (
            _manifest(
                reviewed_at=NOW + timedelta(minutes=1),
                valid_until=EXPIRES,
            ),
            True,
        ),
        (_manifest(reviewed_at=OBSERVED, valid_until=NOW), True),
    ),
)
async def test_registry_rejects_untrusted_future_or_stale_manifests(
    manifest: conversation.ActivationManifest,
    trusted: bool,
) -> None:
    """Reject untrusted and temporally invalid activation evidence."""
    registry = _registry(manifest) if trusted else _registry()
    with pytest.raises(conversation.ConversationValidationError):
        await registry.validate(manifest)


async def test_registry_rejects_runtime_sdk_drift_and_bad_clock() -> None:
    """Fail closed on runtime SDK and asynchronous clock drift."""

    async def naive_clock() -> datetime:
        return datetime(2026, 8, 3, 12)

    with pytest.raises(conversation.ConversationValidationError):
        await _registry(runtime_sdk_version="2.43.0").validate(_manifest())
    with pytest.raises(conversation.ConversationValidationError):
        await _registry(clock=naive_clock).validate(_manifest())


async def test_registry_detects_integrity_tampering() -> None:
    """Reject a typed manifest whose retained digest was corrupted."""
    manifest = _manifest()
    object.__setattr__(
        manifest,
        "integrity_digest",
        conversation.IntegrityDigest("0" * 64),
    )

    with pytest.raises(conversation.ConversationValidationError):
        await _registry().validate(manifest)

    signature_manifest = _manifest()
    object.__setattr__(
        signature_manifest,
        "review_signature",
        conversation.IntegrityDigest("1" * 64),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await _registry().validate(signature_manifest)


async def test_validate_rejects_non_manifest_input() -> None:
    """Reject untyped candidates before activation validation."""
    with pytest.raises(conversation.ConversationValidationError):
        await _registry().validate(cast(Any, object()))


async def test_load_is_idempotent_and_revision_collision_fails() -> None:
    """Load identical content once and reject revision reuse with drift."""
    manifest = _manifest()
    changed = _manifest(active=False)
    registry = _registry(manifest, changed)
    first = await registry.load(manifest)
    second = await registry.load(manifest)

    assert second == first
    with pytest.raises(conversation.ConversationConflictError):
        await registry.load(changed)


async def test_apply_and_resolve_exact_active_row() -> None:
    """Activate and resolve only the exact binding behavior row."""
    registry = _registry()
    manifest = _manifest()
    loaded = await registry.load(manifest)
    applied = await registry.apply(
        manifest.integrity_digest,
        expected_generation=loaded.generation,
    )

    row = await registry.resolve(
        manifest.binding,
        mode=conversation.ConversationMode.STORED,
        reasoning_context=conversation.ReasoningContext.ALL_TURNS,
        compaction_operation=conversation.CompactionOperation.STANDALONE,
    )

    assert applied.active_manifest == manifest
    assert applied.activated_manifest_digests == (manifest.integrity_digest,)
    assert row.active
    assert row.retrieve_supported
    assert row.delete_supported


@pytest.mark.parametrize("azure", (False, True))
async def test_native_provider_dispatch_requires_exact_active_registry(
    azure: bool,
) -> None:
    """Guard the actual SDK boundary with off, applied, and revoked state."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            request=request,
            json=_native_response(
                "response-activation",
                [_native_message("message-activation", "active")],
            ),
        )

    binding = _native_binding(
        lane_id=f"lane-production-activation-{azure}",
        azure=azure,
    )
    profile = _production_profile(binding)
    manifest = _production_stateless_manifest(binding)
    registry = _registry(manifest)

    def provider(
        activation_registry: conversation.AsyncActivationRegistry | None,
    ) -> conversation.NativeOpenAIStatelessProvider:
        client = AsyncOpenAI(
            api_key="activation-key",
            base_url=binding.normalized_endpoint,
            default_query=None,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
            max_retries=0,
        )
        return conversation.NativeOpenAIStatelessProvider(
            client=client,
            profile=_native_profile(binding),
            capability_profile=profile,
            activation_registry=activation_registry,
        )

    inactive = provider(None)
    activated = provider(registry)
    plan = _native_plan(
        binding,
        reasoning=conversation.ReasoningContext.CURRENT_TURN,
    )
    try:
        with pytest.raises(conversation.ConversationCapabilityError):
            await inactive.dispatch(plan)
        with pytest.raises(conversation.ConversationCapabilityError):
            await activated.dispatch(plan)
        assert dispatches == 0

        loaded = await registry.load(manifest)
        applied = await registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        )
        result = await activated.dispatch(plan)
        assert tuple(item.kind for item in result.items) == (
            conversation.ProviderItemKind.MESSAGE,
        )
        assert dispatches == 1

        await registry.revoke(
            manifest.integrity_digest,
            expected_generation=applied.generation,
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            await activated.dispatch(plan)
        assert dispatches == 1
    finally:
        await inactive.aclose()
        await activated.aclose()


@pytest.mark.parametrize("azure", (False, True))
async def test_stored_dispatch_and_lifecycle_use_registry_boundaries(
    azure: bool,
) -> None:
    """Block new stored effects while retaining revoked-state deletion."""
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            await request.aread()
            payload = _native_response(
                "private-activation-stored",
                [_native_message("message-activation-stored", "stored")],
            )
            payload["store"] = True
            return httpx.Response(200, request=request, json=payload)
        assert request.method == "DELETE"
        return httpx.Response(204, request=request)

    binding = _native_stored_binding(
        lane_id=f"lane-production-stored-activation-{azure}",
        azure=azure,
    )
    profile = _production_stored_profile(binding)
    manifest = _production_stored_manifest(binding)
    registry = _registry(manifest)

    def provider(
        activation_registry: conversation.AsyncActivationRegistry | None,
    ) -> conversation.NativeOpenAIStoredProvider:
        client = AsyncOpenAI(
            api_key="activation-stored-key",
            base_url=binding.normalized_endpoint,
            default_query=None,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
            max_retries=0,
        )
        return conversation.NativeOpenAIStoredProvider(
            client=client,
            profile=_native_stored_profile(binding),
            capability_profile=profile,
            activation_registry=activation_registry,
        )

    inactive = provider(None)
    activated = provider(registry)
    plan = conversation.FirstStoredProviderPlan(
        binding=binding,
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.CURRENT_TURN,
            effective=None,
        ),
        new_input={"text": "stored activation"},
    )
    upstream_id = conversation.UpstreamResponseId("private-activation-stored")
    try:
        with pytest.raises(conversation.ConversationCapabilityError):
            await inactive.dispatch(plan)
        with pytest.raises(conversation.ConversationCapabilityError):
            await inactive.delete(upstream_id)
        assert methods == []

        loaded = await registry.load(manifest)
        applied = await registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        )
        result = await activated.dispatch(plan)
        assert result.upstream_response_id == upstream_id
        assert methods == ["POST"]

        await registry.revoke(
            manifest.integrity_digest,
            expected_generation=applied.generation,
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            await activated.dispatch(plan)
        deletion = await activated.delete(upstream_id)
        assert deletion.disposition is (
            conversation.UpstreamDeleteDisposition.DELETED
        )
        assert methods == ["POST", "DELETE"]
    finally:
        await inactive.aclose()
        await activated.aclose()


async def test_inactive_rows_and_all_inactive_manifests_do_not_dispatch() -> (
    None
):
    """Never resolve inactive evidence or activate an entirely dormant set."""
    manifest = _manifest(active=False)
    source = _manifest(revision="activation-revision-2")
    inactive = replace(source.evidence[0], active=False)
    mixed = replace(
        source,
        evidence=(inactive,) + source.evidence[1:],
    )
    registry = _registry(manifest, mixed)
    loaded = await registry.load(manifest)

    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        )

    loaded = await registry.load(mixed)
    await registry.apply(
        mixed.integrity_digest,
        expected_generation=loaded.generation,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.resolve(
            mixed.binding,
            mode=inactive.mode,
            reasoning_context=inactive.reasoning_context,
            compaction_operation=inactive.compaction_operation,
        )


async def test_resolve_rejects_binding_drift_and_missing_dimension() -> None:
    """Reject exact-binding drift and behavior outside the manifest."""
    registry = _registry()
    manifest = _manifest()
    loaded = await registry.load(manifest)
    await registry.apply(
        manifest.integrity_digest,
        expected_generation=loaded.generation,
    )

    with pytest.raises(conversation.ConversationBindingDriftError):
        await registry.resolve(
            _binding(transport=conversation.ProviderTransport.STREAMING),
            mode=conversation.ConversationMode.STORED,
            reasoning_context=conversation.ReasoningContext.ALL_TURNS,
            compaction_operation=conversation.CompactionOperation.NONE,
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.resolve(
            manifest.binding,
            mode=conversation.ConversationMode.OFF,
            reasoning_context=conversation.ReasoningContext.ALL_TURNS,
            compaction_operation=conversation.CompactionOperation.NONE,
        )


async def test_generation_cas_makes_concurrent_apply_atomic() -> None:
    """Allow only one concurrent activation for an observed generation."""
    registry = _registry()
    manifest = _manifest()
    loaded = await registry.load(manifest)

    results = await gather(
        registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        ),
        registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        ),
        return_exceptions=True,
    )

    assert (
        sum(
            isinstance(value, conversation.ActivationSnapshot)
            for value in results
        )
        == 1
    )
    assert (
        sum(
            isinstance(value, conversation.ConversationConflictError)
            for value in results
        )
        == 1
    )


async def test_rollback_restores_prior_manifest_or_dormant_state() -> None:
    """Use an exact prior snapshot for atomic rollback and deactivation."""
    first = _manifest()
    second = _manifest(revision="activation-revision-2", active=False)
    second_row = replace(second.evidence[0], active=True)
    second = replace(second, evidence=(second_row,) + second.evidence[1:])
    registry = _registry(first, second)
    dormant = await registry.snapshot()
    loaded = await registry.load(first)
    active_first = await registry.apply(
        first.integrity_digest,
        expected_generation=loaded.generation,
    )
    loaded_second = await registry.load(second)
    active_second = await registry.apply(
        second.integrity_digest,
        expected_generation=loaded_second.generation,
    )

    restored = await registry.rollback(
        active_first,
        expected_generation=active_second.generation,
    )
    deactivated = await registry.rollback(
        dormant,
        expected_generation=restored.generation,
    )

    assert restored.active_manifest == first
    assert deactivated.active_manifest is None


async def test_revocation_blocks_dispatch_but_preserves_prior_snapshot() -> (
    None
):
    """Preserve compatibility evidence while revoking new dispatch."""
    registry = _registry()
    manifest = _manifest()
    loaded = await registry.load(manifest)
    prior = await registry.apply(
        manifest.integrity_digest,
        expected_generation=loaded.generation,
    )
    revoked = await registry.revoke(
        manifest.integrity_digest,
        expected_generation=prior.generation,
    )
    same_revocation = await registry.revoke(
        manifest.integrity_digest,
        expected_generation=revoked.generation,
    )

    assert prior.active_manifest == manifest
    assert prior.revoked_manifest_digests == ()
    assert revoked.active_manifest == manifest
    assert revoked.revoked_manifest_digests == (manifest.integrity_digest,)
    assert same_revocation == revoked
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.resolve(
            manifest.binding,
            mode=conversation.ConversationMode.STATELESS,
            reasoning_context=conversation.ReasoningContext.CURRENT_TURN,
            compaction_operation=conversation.CompactionOperation.NONE,
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.rollback(
            prior,
            expected_generation=revoked.generation,
        )
    retrieved = await registry.resolve_lifecycle(
        manifest.binding,
        capability=(
            conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL
        ),
    )
    deleted = await registry.resolve_lifecycle(
        manifest.binding,
        capability=conversation.ConversationCapability.STORED_RESPONSE_DELETION,
    )
    assert retrieved.mode is conversation.ConversationMode.STORED
    assert deleted.mode is conversation.ConversationMode.STORED


async def test_lifecycle_compatibility_rejects_unknown_or_invalid_scope() -> (
    None
):
    """Keep historical lifecycle resolution exact and time bounded."""
    registry = _registry()
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.resolve_lifecycle(
            _binding(),
            capability=(
                conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await registry.resolve_lifecycle(
            _binding(),
            capability=(conversation.ConversationCapability.INLINE_COMPACTION),
        )


async def test_loaded_manifest_does_not_grant_lifecycle_authority() -> None:
    """Require an actual activation before historical cleanup resolution."""
    manifest = _manifest()
    registry = _registry(manifest)
    loaded = await registry.load(manifest)

    assert loaded.activated_manifest_digests == ()
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.resolve_lifecycle(
            manifest.binding,
            capability=(
                conversation.ConversationCapability.STORED_RESPONSE_DELETION
            ),
        )
    forged = loaded
    object.__setattr__(forged, "active_manifest", manifest)
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.rollback(
            forged,
            expected_generation=loaded.generation,
        )


async def test_lifecycle_compatibility_skips_stale_and_incapable_history() -> (
    None
):
    """Use a later valid resolver and reject history with no stored row."""
    observed = NOW

    async def clock() -> datetime:
        return observed

    stale = _manifest(
        revision="activation-stale-history",
        valid_until=NOW + timedelta(minutes=1),
    )
    valid = _manifest(
        revision="activation-valid-history",
        reviewed_at=OBSERVED,
    )
    registry = _registry(stale, valid, clock=clock)
    loaded_stale = await registry.load(stale)
    active_stale = await registry.apply(
        stale.integrity_digest,
        expected_generation=loaded_stale.generation,
    )
    loaded_valid = await registry.load(valid)
    await registry.apply(
        valid.integrity_digest,
        expected_generation=loaded_valid.generation,
    )
    assert stale.integrity_digest in active_stale.activated_manifest_digests
    observed = NOW + timedelta(minutes=2)
    resolved = await registry.resolve_lifecycle(
        valid.binding,
        capability=conversation.ConversationCapability.STORED_RESPONSE_DELETION,
    )
    assert resolved.mode is conversation.ConversationMode.STORED

    stateless = _production_stateless_manifest(
        _native_binding(lane_id="lane-stateless-history")
    )
    stateless_registry = _registry(stateless)
    loaded_stateless = await stateless_registry.load(stateless)
    await stateless_registry.apply(
        stateless.integrity_digest,
        expected_generation=loaded_stateless.generation,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await stateless_registry.resolve_lifecycle(
            stateless.binding,
            capability=(
                conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL
            ),
        )


async def test_mutations_reject_stale_generation_and_unknown_digest() -> None:
    """Reject stale CAS, unknown activation, and unknown revocation targets."""
    registry = _registry()
    manifest = _manifest()
    loaded = await registry.load(manifest)

    with pytest.raises(conversation.ConversationConflictError):
        await registry.apply(manifest.integrity_digest, expected_generation=0)
    with pytest.raises(conversation.ConversationValidationError):
        await registry.apply(manifest.integrity_digest, expected_generation=-1)
    with pytest.raises(conversation.ConversationCapabilityError):
        await registry.apply(
            conversation.IntegrityDigest("a" * 64),
            expected_generation=loaded.generation,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await registry.revoke(
            conversation.IntegrityDigest("a" * 64),
            expected_generation=loaded.generation,
        )


async def test_resolve_rejects_untyped_arguments() -> None:
    """Reject non-enum dispatch dimensions before consulting activation."""
    registry = _registry()

    with pytest.raises(conversation.ConversationValidationError):
        await registry.resolve(
            _binding(),
            mode=cast(Any, "stored"),
            reasoning_context=conversation.ReasoningContext.ALL_TURNS,
            compaction_operation=conversation.CompactionOperation.NONE,
        )


async def test_foreign_snapshot_cannot_be_used_for_rollback() -> None:
    """Reject rollback snapshots issued by another registry identity."""
    source = await _registry().snapshot()
    trusted = _manifest()
    registry = conversation.AsyncActivationRegistry(
        registry_id="other-registry",
        runtime_sdk_version="2.42.0",
        trusted_review_signatures=frozenset({trusted.review_signature}),
        clock=_clock,
    )

    with pytest.raises(conversation.ConversationValidationError):
        await registry.rollback(source, expected_generation=0)
