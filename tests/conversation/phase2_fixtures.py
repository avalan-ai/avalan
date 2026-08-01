"""Build deterministic Phase 2 conversation runtime fixtures."""

from datetime import UTC, datetime

import avalan.conversation as conversation

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)


def authority(
    principal: str = "principal-phase2",
    *,
    tenant: str = "tenant-phase2",
    agent: str = "agent-phase2",
) -> conversation.AuthorityScope:
    """Return one deterministic authenticated authority."""
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId(tenant),
        principal_id=conversation.AuthorityPrincipalId(principal),
        agent_id=conversation.ConversationAgentId(agent),
        endpoint_id=conversation.AuthorityEndpointId("endpoint-phase2"),
    )


def binding(
    lane_id: str = "lane-phase2",
    *,
    streaming: bool = False,
    agent: str = "agent-phase2",
) -> conversation.ProviderLaneBinding:
    """Return one exact synthetic fake-lane binding."""
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=(
            "avalan.conversation.fakes.DeterministicFakeProviderScript"
        ),
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint="https://fake.phase2.test/v1",
        model_or_deployment=f"model-{lane_id}",
        provider_api_revision=conversation.ProviderApiRevision("api-phase2"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-phase2"),
        model_configuration_revision=conversation.ModelConfigurationRevision(
            "model-config-phase2"
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-phase2")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-phase2"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-phase2")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=conversation.ConversationAgentId(agent),
    )


def reasoning(
    requested: conversation.ReasoningContext = (
        conversation.ReasoningContext.AUTO
    ),
) -> conversation.EffectiveReasoningMetadata:
    """Return provider-plan reasoning metadata."""
    return conversation.EffectiveReasoningMetadata(
        requested=requested,
        effective=None,
    )


def empty_stateless_plan(
    lane_binding: conversation.ProviderLaneBinding,
    *,
    requested: conversation.ReasoningContext = (
        conversation.ReasoningContext.AUTO
    ),
) -> conversation.StatelessProviderPlan:
    """Return one first-turn stateless fake plan."""
    return conversation.StatelessProviderPlan(
        binding=lane_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(),
        ),
        reasoning=reasoning(requested),
    )


def next_stateless_plan(
    lane_binding: conversation.ProviderLaneBinding,
    items: tuple[conversation.ProviderItem, ...],
    *,
    requested: conversation.ReasoningContext = (
        conversation.ReasoningContext.AUTO
    ),
) -> conversation.StatelessProviderPlan:
    """Return one child-turn stateless fake plan."""
    return conversation.StatelessProviderPlan(
        binding=lane_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=items,
        ),
        reasoning=reasoning(requested),
    )


def first_stored_plan(
    lane_binding: conversation.ProviderLaneBinding,
) -> conversation.FirstStoredProviderPlan:
    """Return one first-turn provider-stored fake plan."""
    return conversation.FirstStoredProviderPlan(
        binding=lane_binding,
        reasoning=reasoning(),
    )


def retention(
    *,
    stored: bool = False,
    ttl: int = 3_600,
) -> conversation.RetentionLimits:
    """Return bounded process-local retention for fake execution."""
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.PROCESS_LOCAL,
            upstream=(
                conversation.ProviderLaneStorage.STORED
                if stored
                else conversation.ProviderLaneStorage.STATELESS
            ),
            provider_storage_disclosed=stored,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.KNOWN
            if stored
            else conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=ttl,
        known_upstream_ttl_seconds=ttl if stored else None,
    )


def root_identity(
    suffix: str = "1",
    *,
    branch_id: str = "branch-phase2",
) -> conversation.CheckpointIdentity:
    """Return one deterministic root checkpoint identity."""
    return conversation.CheckpointIdentity(
        conversation_id=conversation.ConversationId(f"conversation-{suffix}"),
        logical_turn_id=conversation.LogicalTurnId(f"turn-{suffix}"),
        execution_segment_id=conversation.ExecutionSegmentId(
            f"segment-{suffix}"
        ),
        checkpoint_id=conversation.CheckpointId(f"checkpoint-{suffix}"),
        branch_id=conversation.ConversationBranchId(branch_id),
        sequence=conversation.CheckpointSequence(0),
    )


def child_identity(
    parent: conversation.ConversationCheckpoint,
    suffix: str,
    *,
    branch_id: str | None = None,
) -> conversation.CheckpointIdentity:
    """Return one deterministic child checkpoint identity."""
    return conversation.CheckpointIdentity(
        conversation_id=parent.identity.conversation_id,
        logical_turn_id=conversation.LogicalTurnId(f"turn-{suffix}"),
        execution_segment_id=conversation.ExecutionSegmentId(
            f"segment-{suffix}"
        ),
        checkpoint_id=conversation.CheckpointId(f"checkpoint-{suffix}"),
        branch_id=(
            conversation.ConversationBranchId(branch_id)
            if branch_id is not None
            else parent.identity.branch_id
        ),
        sequence=conversation.CheckpointSequence(parent.identity.sequence + 1),
        parent_checkpoint_id=parent.identity.checkpoint_id,
        parent_sequence=parent.identity.sequence,
    )


def semantics(
    scope: conversation.AuthorityScope,
    *,
    operation: conversation.ConversationOperation,
    mode: conversation.ConversationMode,
    parent_id: conversation.CheckpointId | None = None,
    text: str = "safe-input",
) -> conversation.ConversationRequestSemantics:
    """Return digestable semantic input for one run."""
    return conversation.ConversationRequestSemantics(
        authority=scope,
        operation=operation,
        mode=mode,
        reasoning_context=conversation.ReasoningContext.AUTO,
        semantic_input={"text": text},
        parent_checkpoint_id=parent_id,
    )


def request(
    *,
    scope: conversation.AuthorityScope,
    identity: conversation.CheckpointIdentity,
    advance: conversation.ConversationAdvance,
    lane_ids: tuple[str, ...] = ("lane-phase2",),
    modes: tuple[conversation.ConversationMode, ...] = (
        conversation.ConversationMode.STATELESS,
    ),
    key: str = "key-phase2",
    response_suffix: str = "1",
    stored_retention: bool = False,
    boundary: conversation.ConversationCommitBoundary = (
        conversation.ConversationCommitBoundary.OUTWARD_TURN
    ),
) -> conversation.ConversationRunRequest:
    """Return one exact run request for the selected advance and lanes."""
    parent_id = (
        advance.parent_checkpoint_id
        if not isinstance(advance, conversation.FirstTurnAdvance)
        else None
    )
    operation = (
        conversation.ConversationOperation.CREATE
        if isinstance(
            advance,
            conversation.FirstTurnAdvance | conversation.ResetAdvance,
        )
        else (
            conversation.ConversationOperation.BRANCH
            if isinstance(advance, conversation.ExplicitBranchAdvance)
            else conversation.ConversationOperation.CONTINUE
        )
    )
    mode = modes[0]
    outward = boundary is conversation.ConversationCommitBoundary.OUTWARD_TURN
    return conversation.ConversationRunRequest(
        semantics=semantics(
            scope,
            operation=operation,
            mode=mode,
            parent_id=parent_id,
            text=f"safe-input-{response_suffix}",
        ),
        identity=identity,
        advance=advance,
        lanes=tuple(
            conversation.ConversationLaneRequest(
                lane_id=conversation.ProviderLaneId(lane_id),
                mode=lane_mode,
            )
            for lane_id, lane_mode in zip(lane_ids, modes, strict=True)
        ),
        visible_delta=(
            conversation.VisibleTranscriptEntry(
                role=conversation.VisibleTranscriptRole.USER,
                content=f"visible-{response_suffix}",
            ),
        ),
        retention=retention(stored=stored_retention),
        idempotency_key=conversation.RequestIdempotencyKey(key),
        boundary=boundary,
        provisional_response_id=(
            conversation.ProvisionalResponseId(
                f"provisional-{response_suffix}"
            )
            if outward
            else None
        ),
        public_response_id=(
            conversation.PublicResponseId(f"response-{response_suffix}")
            if outward
            else None
        ),
    )


def coordinator(
    *,
    store: conversation.ConversationStore,
    scope: conversation.AuthorityScope,
    runtimes: tuple[conversation.ConversationLaneRuntime, ...],
    publisher: conversation.DeterministicFakePublisher | None = None,
    observer: conversation.DeterministicFakeObserver | None = None,
    boundary_hook: conversation.FakeCoordinatorBoundaryHook | None = None,
) -> conversation.RunScopedConversationCoordinator:
    """Return one run-scoped coordinator with deterministic fake effects."""
    return conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(NOW),
        publisher=publisher or conversation.DeterministicFakePublisher(),
        observer=observer or conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=runtimes,
        boundary_hook=boundary_hook,
    )
