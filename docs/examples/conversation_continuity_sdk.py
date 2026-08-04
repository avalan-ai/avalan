"""Run deterministic direct conversation operations without network access."""

from asyncio import run
from datetime import UTC, datetime
from json import dumps

import avalan
import avalan.conversation as conversation


def _authority() -> conversation.AuthorityScope:
    """Return the trusted authority for the local example."""
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.FIXED_LOCAL_SINGLE_USER,
        principal_id=conversation.AuthorityPrincipalId(
            "principal-local-example"
        ),
        agent_id=conversation.ConversationAgentId("agent-local-example"),
        endpoint_id=conversation.AuthorityEndpointId("endpoint-local-example"),
        local_single_user_configured=True,
    )


def _binding() -> conversation.ProviderLaneBinding:
    """Return one exact test-only synthetic provider binding."""
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId("lane-local-example"),
        adapter_type=(
            "avalan.conversation.fakes.DeterministicFakeProviderScript"
        ),
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint="https://local.example.invalid/v1",
        model_or_deployment="deterministic-local-model",
        provider_api_revision=conversation.ProviderApiRevision("local-v1"),
        sdk_revision=conversation.ProviderSdkRevision("local-sdk-v1"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("local-model-v1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("local-capabilities-v1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("local-tools-v1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("local-execution-v1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.STREAMING,
        agent_id=conversation.ConversationAgentId("agent-local-example"),
    )


def _reasoning() -> conversation.EffectiveReasoningMetadata:
    """Return requested reasoning metadata for a fake provider plan."""
    return conversation.EffectiveReasoningMetadata(
        requested=conversation.ReasoningContext.CURRENT_TURN,
        effective=None,
    )


def _empty_plan(
    binding: conversation.ProviderLaneBinding,
) -> conversation.StatelessProviderPlan:
    """Return a first-turn stateless provider plan."""
    return conversation.StatelessProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(),
        ),
        reasoning=_reasoning(),
    )


def _continued_plan(
    binding: conversation.ProviderLaneBinding,
    items: tuple[conversation.ProviderItem, ...],
) -> conversation.StatelessProviderPlan:
    """Return a stateless plan containing the complete prior output."""
    return conversation.StatelessProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=items,
        ),
        reasoning=_reasoning(),
    )


def build_local_client() -> avalan.DirectConversationClient:
    """Build a deterministic test-only direct client."""
    authority = _authority()
    binding = _binding()
    first_plan = _empty_plan(binding)
    first = conversation.fake_provider_result(
        first_plan,
        turn=1,
        text="first local answer",
    )
    continued_plan = _continued_plan(binding, first.items)
    results = (
        first,
        conversation.fake_provider_result(
            continued_plan,
            turn=2,
            text="continued local answer",
        ),
        conversation.fake_provider_result(
            continued_plan,
            turn=3,
            text="branch local answer",
        ),
        conversation.fake_compaction_result(
            continued_plan,
            turn=4,
            opaque_state=bytes(32),
        ),
        conversation.fake_provider_result(
            first_plan,
            turn=5,
            text="reset local answer",
        ),
    )
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            authority
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2030, 1, 1, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=binding,
                capability_profile=conversation.fake_capability_profile(
                    binding
                ),
                provider_script=(
                    conversation.DeterministicFakeProviderScript(
                        results=results
                    )
                ),
            ),
        ),
    )
    retention = conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.PROCESS_LOCAL,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=3_600,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=authority,
        lane=binding,
        retention=retention,
        id_namespace="local-example",
    )
    return avalan.DirectConversationClient(runtime)


async def run_example() -> dict[str, object]:
    """Execute streaming, continuation, branch, compact, and reset."""
    client = build_local_client()
    stream = await client.create(
        "Start a deterministic local conversation.",
        avalan.StatelessConversationSettings(
            reasoning_context=avalan.ReasoningContext.CURRENT_TURN
        ),
        stream=True,
    )
    events = [event async for event in stream]
    terminal = events[-1]
    assert isinstance(terminal, avalan.DirectConversationStreamTerminal)
    first = terminal.result
    assert isinstance(first.handle, avalan.StatelessConversationHandle)
    parent = avalan.StatelessParent(handle=first.handle)

    continued = await client.continue_conversation(
        "Continue from the immutable parent.",
        avalan.StatelessConversationSettings(parent=parent),
    )
    branched = await client.branch(
        "Explore an independent child.",
        avalan.StatelessConversationSettings(
            parent=parent,
            branch=avalan.ConversationBranchIntent(
                parent=parent,
                branch_id=conversation.ConversationBranchId(
                    "local-example-branch"
                ),
            ),
        ),
    )
    compacted = await client.compact(
        avalan.StandaloneCompactRequest(parent=parent)
    )
    compact_handle = await client.commit_compact(compacted)
    reset = await client.reset(
        "Start a fresh root and discard prior opaque continuity.",
        avalan.ConversationResetIntent(
            parent=parent,
            target_mode=avalan.ConversationMode.STATELESS,
        ),
        avalan.StatelessConversationSettings(),
    )

    return {
        "stream_event_types": [type(event).__name__ for event in events],
        "terminal_handle_committed": stream.committed_handle == first.handle,
        "continued_output": continued.output,
        "branch_output": branched.output,
        "branch_isolated": branched.handle.branch_id != first.handle.branch_id,
        "compact_item_count": compacted.canonical_context.item_count,
        "compact_committed": (
            compact_handle.checkpoint_id != first.handle.checkpoint_id
        ),
        "reset_output": reset.output,
        "reset_is_fresh_root": (
            reset.handle.conversation_id != first.handle.conversation_id
        ),
    }


if __name__ == "__main__":
    print(dumps(run(run_example()), indent=2, sort_keys=True))
