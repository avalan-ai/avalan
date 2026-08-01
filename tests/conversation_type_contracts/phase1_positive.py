"""Exercise all valid typed conversation families and async effects."""

from typing import assert_type

from avalan.conversation import (
    AuthorityDigest,
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    CallerHeldState,
    CanonicalRequestDigest,
    CapabilityEvidence,
    CapabilityEvidenceState,
    CapabilityProfileId,
    CapabilityProfileRevision,
    CheckpointCandidate,
    CheckpointId,
    CheckpointSequence,
    CompactionPolicy,
    ContinuationDigest,
    ConversationAgentId,
    ConversationBranchId,
    ConversationCapability,
    ConversationCapabilityProfile,
    ConversationCheckpoint,
    ConversationCodecVersion,
    ConversationCoordinator,
    ConversationHandle,
    ConversationId,
    ConversationMode,
    ConversationModeChangeAuthorization,
    ConversationModeChangeOperation,
    ConversationModeConversion,
    ConversationModelCallId,
    ConversationModeReset,
    ConversationModeTransition,
    ConversationObservation,
    ConversationObserver,
    ConversationParent,
    ConversationProvider,
    ConversationProviderStream,
    ConversationRequestSemantics,
    ConversationResetDisposition,
    ConversationResult,
    ConversationSettings,
    ConversationStore,
    ConversationStreamTerminal,
    ConversationTaskId,
    DisabledCompaction,
    EffectiveReasoningMetadata,
    ExecutionDefinitionRevision,
    ExecutionSegmentId,
    InlineCompaction,
    IntegrityDigest,
    LogicalTurnId,
    ModelConfigurationRevision,
    ModeTransitionAuthority,
    NamedHeadId,
    NamedHeadParent,
    NamedHeadRevision,
    OneShotConversationSettings,
    ProviderApiRevision,
    ProviderCallId,
    ProviderFamily,
    ProviderItem,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    ProviderLaneBinding,
    ProviderLaneId,
    ProviderPlan,
    ProviderResult,
    ProviderSdkRevision,
    ProviderTransport,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    RequestSemanticDigest,
    SafeAlias,
    StandaloneCompactRequest,
    StandaloneCompactResult,
    StatelessConversationHandle,
    StatelessConversationSettings,
    StatelessParent,
    StatelessProviderPlan,
    StoredConversationHandle,
    StoredConversationSettings,
    StoredParent,
    StoredProviderPlan,
    StructuredInputContinuationId,
    ToolSchemaRevision,
    UpstreamResponseId,
    validate_mode_transition_authority,
)

assert_type(ConversationId("conversation"), ConversationId)
assert_type(LogicalTurnId("logical-turn"), LogicalTurnId)
assert_type(ExecutionSegmentId("execution-segment"), ExecutionSegmentId)
assert_type(CheckpointId("checkpoint"), CheckpointId)
assert_type(ConversationBranchId("branch"), ConversationBranchId)
assert_type(NamedHeadId("head"), NamedHeadId)
assert_type(ProviderLaneId("lane"), ProviderLaneId)
assert_type(ConversationModelCallId("model-call"), ConversationModelCallId)
assert_type(PublicResponseId("public-response"), PublicResponseId)
assert_type(ProvisionalResponseId("provisional"), ProvisionalResponseId)
assert_type(UpstreamResponseId("upstream-response"), UpstreamResponseId)
assert_type(ConversationTaskId("task"), ConversationTaskId)
assert_type(ConversationAgentId("agent"), ConversationAgentId)
assert_type(
    StructuredInputContinuationId("continuation"),
    StructuredInputContinuationId,
)
assert_type(AuthorityTenantId("tenant"), AuthorityTenantId)
assert_type(AuthorityPrincipalId("principal"), AuthorityPrincipalId)
assert_type(AuthorityEndpointId("endpoint"), AuthorityEndpointId)
assert_type(RequestIdempotencyKey("idempotency"), RequestIdempotencyKey)
assert_type(CanonicalRequestDigest("canonical"), CanonicalRequestDigest)
assert_type(ContinuationDigest("continuation-digest"), ContinuationDigest)
assert_type(CheckpointSequence(1), CheckpointSequence)
assert_type(NamedHeadRevision(1), NamedHeadRevision)
assert_type(ProviderItemId("item"), ProviderItemId)
assert_type(ProviderCallId("call"), ProviderCallId)
assert_type(CapabilityProfileId("profile"), CapabilityProfileId)
assert_type(
    CapabilityProfileRevision("profile-revision"),
    CapabilityProfileRevision,
)
assert_type(ProviderApiRevision("api-revision"), ProviderApiRevision)
assert_type(ProviderSdkRevision("sdk-revision"), ProviderSdkRevision)
assert_type(
    ModelConfigurationRevision("model-revision"),
    ModelConfigurationRevision,
)
assert_type(ToolSchemaRevision("tool-revision"), ToolSchemaRevision)
assert_type(
    ExecutionDefinitionRevision("definition-revision"),
    ExecutionDefinitionRevision,
)
assert_type(ConversationCodecVersion(1), ConversationCodecVersion)
assert_type(ProviderItemIndex(0), ProviderItemIndex)
assert_type(ProviderItemOrder(0), ProviderItemOrder)
assert_type(IntegrityDigest("integrity"), IntegrityDigest)
assert_type(AuthorityDigest("authority"), AuthorityDigest)
assert_type(RequestSemanticDigest("semantic"), RequestSemanticDigest)
assert_type(SafeAlias("safe-alias"), SafeAlias)

AUTHORITY = AuthorityScope(
    source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
    tenant_id=AuthorityTenantId("tenant"),
    principal_id=AuthorityPrincipalId("principal"),
    agent_id=ConversationAgentId("agent"),
    endpoint_id=AuthorityEndpointId("endpoint"),
)
assert_type(AUTHORITY, AuthorityScope)

BINDING = ProviderLaneBinding(
    lane_id=ProviderLaneId("lane"),
    adapter_type="tests.SyntheticProvider",
    provider_family=ProviderFamily.SYNTHETIC,
    normalized_endpoint="https://api.example.test/v1",
    model_or_deployment="model",
    provider_api_revision=ProviderApiRevision("api-revision"),
    sdk_revision=ProviderSdkRevision("sdk-revision"),
    model_configuration_revision=ModelConfigurationRevision("model-revision"),
    capability_profile_revision=CapabilityProfileRevision("profile-revision"),
    tool_schema_revision=ToolSchemaRevision("tool-revision"),
    execution_definition_revision=ExecutionDefinitionRevision(
        "definition-revision"
    ),
    continuation_codec_version=ConversationCodecVersion(1),
    transport=ProviderTransport.NON_STREAMING,
    agent_id=ConversationAgentId("agent"),
)
assert_type(BINDING, ProviderLaneBinding)

STATELESS_HANDLE = StatelessConversationHandle(
    conversation_id=ConversationId("conversation-stateless"),
    checkpoint_id=CheckpointId("checkpoint-stateless"),
    branch_id=ConversationBranchId("main"),
    envelope=CallerHeldState(_value="authenticated-envelope"),
)
STORED_HANDLE = StoredConversationHandle(
    conversation_id=ConversationId("conversation-stored"),
    checkpoint_id=CheckpointId("checkpoint-stored"),
    branch_id=ConversationBranchId("main"),
    public_response_id=PublicResponseId("response-stored"),
)
assert_type(STATELESS_HANDLE, StatelessConversationHandle)
assert_type(STORED_HANDLE, StoredConversationHandle)
STATELESS_HANDLE_FAMILY: ConversationHandle = STATELESS_HANDLE
STORED_HANDLE_FAMILY: ConversationHandle = STORED_HANDLE
assert_type(STATELESS_HANDLE_FAMILY, ConversationHandle)
assert_type(STORED_HANDLE_FAMILY, ConversationHandle)

STATELESS_PARENT = StatelessParent(handle=STATELESS_HANDLE)
STORED_PARENT = StoredParent(handle=STORED_HANDLE)
assert_type(STATELESS_PARENT, StatelessParent)
assert_type(STORED_PARENT, StoredParent)
STATELESS_PARENT_FAMILY: ConversationParent = STATELESS_PARENT
STORED_PARENT_FAMILY: ConversationParent = STORED_PARENT
assert_type(STATELESS_PARENT_FAMILY, ConversationParent)
assert_type(STORED_PARENT_FAMILY, ConversationParent)

DISABLED_COMPACTION = DisabledCompaction()
INLINE_COMPACTION = InlineCompaction(compact_threshold=512)
assert_type(DISABLED_COMPACTION, DisabledCompaction)
assert_type(INLINE_COMPACTION, InlineCompaction)
COMPACTION: CompactionPolicy = INLINE_COMPACTION
assert_type(COMPACTION, CompactionPolicy)

ONE_SHOT = OneShotConversationSettings()
STATELESS = StatelessConversationSettings(
    parent=STATELESS_PARENT,
    compaction=INLINE_COMPACTION,
)
STORED = StoredConversationSettings(
    provider_storage_disclosed=True,
    parent=STORED_PARENT,
)
assert_type(ONE_SHOT, OneShotConversationSettings)
assert_type(STATELESS, StatelessConversationSettings)
assert_type(STORED, StoredConversationSettings)
ONE_SHOT_FAMILY: ConversationSettings = ONE_SHOT
STATELESS_FAMILY: ConversationSettings = STATELESS
STORED_FAMILY: ConversationSettings = STORED
assert_type(ONE_SHOT_FAMILY, ConversationSettings)
assert_type(STATELESS_FAMILY, ConversationSettings)
assert_type(STORED_FAMILY, ConversationSettings)

RESET_AUTHORIZATION = ConversationModeChangeAuthorization(
    authority=AUTHORITY,
    binding=BINDING,
    checkpoint_id=STATELESS_HANDLE.checkpoint_id,
    parent=STATELESS_PARENT,
    source_mode=ConversationMode.STATELESS,
    target_mode=ConversationMode.STORED,
    operation=ConversationModeChangeOperation.RESET,
)
CONVERSION_AUTHORIZATION = ConversationModeChangeAuthorization(
    authority=AUTHORITY,
    binding=BINDING,
    checkpoint_id=STATELESS_HANDLE.checkpoint_id,
    parent=STATELESS_PARENT,
    source_mode=ConversationMode.STATELESS,
    target_mode=ConversationMode.STORED,
    operation=ConversationModeChangeOperation.CONVERT,
)
RESET = ConversationModeReset(authorization=RESET_AUTHORIZATION)
CONVERSION = ConversationModeConversion(authorization=CONVERSION_AUTHORIZATION)
assert_type(RESET_AUTHORIZATION, ConversationModeChangeAuthorization)
assert_type(CONVERSION_AUTHORIZATION, ConversationModeChangeAuthorization)
assert_type(RESET_AUTHORIZATION, ModeTransitionAuthority)
assert_type(RESET, ConversationModeReset)
assert_type(CONVERSION, ConversationModeConversion)
assert_type(RESET.disposition, ConversationResetDisposition)
RESET_FAMILY: ConversationModeTransition = RESET
CONVERSION_FAMILY: ConversationModeTransition = CONVERSION
assert_type(RESET_FAMILY, ConversationModeTransition)
assert_type(CONVERSION_FAMILY, ConversationModeTransition)
assert_type(
    validate_mode_transition_authority(
        RESET,
        current_checkpoint_id=STATELESS_HANDLE.checkpoint_id,
        current_parent=STATELESS_PARENT,
        current_authority=AUTHORITY,
        current_binding=BINDING,
    ),
    None,
)

REASONING = EffectiveReasoningMetadata(
    requested=STATELESS.reasoning_context,
    effective=None,
)
RESULT = ConversationResult(
    handle=STATELESS_HANDLE,
    reasoning=REASONING,
    checkpoint_digest=IntegrityDigest("checkpoint-digest"),
)
TERMINAL = ConversationStreamTerminal(result=RESULT)
HEAD_PARENT = NamedHeadParent(
    head_id=NamedHeadId("main"),
    expected_revision=NamedHeadRevision(1),
    parent=STATELESS_PARENT,
)
COMPACT_REQUEST = StandaloneCompactRequest(parent=STATELESS_PARENT)
COMPACT_RESULT = StandaloneCompactResult(
    handle=STATELESS_HANDLE,
    canonical_context_digest=IntegrityDigest("context-digest"),
)
assert_type(REASONING, EffectiveReasoningMetadata)
assert_type(RESULT, ConversationResult)
assert_type(TERMINAL, ConversationStreamTerminal)
assert_type(HEAD_PARENT, NamedHeadParent)
assert_type(COMPACT_REQUEST, StandaloneCompactRequest)
assert_type(COMPACT_RESULT, StandaloneCompactResult)


def assert_capability_and_protocol_families(
    binding: ProviderLaneBinding,
    evidence: CapabilityEvidence,
    profile: ConversationCapabilityProfile,
    capability: ConversationCapability,
    evidence_state: CapabilityEvidenceState,
    provider_family: ProviderFamily,
    transport: ProviderTransport,
    coordinator: ConversationCoordinator,
    store: ConversationStore,
    provider: ConversationProvider,
    provider_stream: ConversationProviderStream,
    observer: ConversationObserver,
    checkpoint: ConversationCheckpoint,
    candidate: CheckpointCandidate,
    stateless_plan: StatelessProviderPlan,
    stored_plan: StoredProviderPlan,
    provider_result: ProviderResult,
) -> None:
    """Assert every capability, checkpoint, plan, and protocol family."""
    assert_type(binding, ProviderLaneBinding)
    assert_type(evidence, CapabilityEvidence)
    assert_type(profile, ConversationCapabilityProfile)
    assert_type(capability, ConversationCapability)
    assert_type(evidence_state, CapabilityEvidenceState)
    assert_type(provider_family, ProviderFamily)
    assert_type(transport, ProviderTransport)
    assert_type(coordinator, ConversationCoordinator)
    assert_type(store, ConversationStore)
    assert_type(provider, ConversationProvider)
    assert_type(provider_stream, ConversationProviderStream)
    assert_type(observer, ConversationObserver)
    assert_type(checkpoint, ConversationCheckpoint)
    assert_type(candidate, CheckpointCandidate)
    assert_type(stateless_plan, StatelessProviderPlan)
    assert_type(stored_plan, StoredProviderPlan)
    assert_type(provider_result, ProviderResult)
    stateless_family: ProviderPlan = stateless_plan
    stored_family: ProviderPlan = stored_plan
    assert_type(stateless_family, ProviderPlan)
    assert_type(stored_family, ProviderPlan)


async def exercise_async_effects(
    coordinator: ConversationCoordinator,
    store: ConversationStore,
    provider: ConversationProvider,
    observer: ConversationObserver,
    observation: ConversationObservation,
    request: ConversationRequestSemantics,
    settings: ConversationSettings,
    plan: ProviderPlan,
    candidate: CheckpointCandidate,
    checkpoint_id: CheckpointId,
    authority: AuthorityScope,
) -> tuple[
    ConversationResult,
    ConversationStreamTerminal,
    ConversationCheckpoint,
    ConversationCheckpoint,
    ProviderResult,
    ProviderResult,
    ProviderItem | None,
]:
    """Await every coordinator, store, provider, and observer effect."""
    result = await coordinator.execute(request, settings)
    terminal = await coordinator.stream(request, settings)
    loaded = await store.load(checkpoint_id, authority)
    committed = await store.commit(candidate)
    provider_result = await provider.dispatch(plan)
    provider_stream = await provider.stream(plan)
    last_item: ProviderItem | None = None
    async for item in provider_stream:
        last_item = item
    provider_terminal = await provider_stream.terminal()
    await provider_stream.aclose()
    await observer.publish(observation)
    await store.close()
    return (
        result,
        terminal,
        loaded,
        committed,
        provider_result,
        provider_terminal,
        last_item,
    )
