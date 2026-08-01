"""Exercise valid Phase 0 conversation contract typing."""

from avalan.conversation import (
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    CheckpointId,
    CheckpointIdentity,
    CheckpointSequence,
    ConversationAgentId,
    ConversationBranchId,
    ConversationId,
    ExecutionSegmentId,
    LocalResponseStorage,
    LogicalTurnId,
    ProviderLaneStorage,
    StoragePolicy,
)


def checkpoint_id(value: CheckpointId) -> CheckpointId:
    """Return one statically distinct checkpoint identifier."""
    return value


AUTHORITY = AuthorityScope(
    source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
    tenant_id=AuthorityTenantId("tenant"),
    principal_id=AuthorityPrincipalId("principal"),
    agent_id=ConversationAgentId("agent"),
    endpoint_id=AuthorityEndpointId("endpoint"),
)
CHECKPOINT = CheckpointIdentity(
    conversation_id=ConversationId("conversation"),
    logical_turn_id=LogicalTurnId("turn"),
    execution_segment_id=ExecutionSegmentId("segment"),
    checkpoint_id=CheckpointId("checkpoint"),
    branch_id=ConversationBranchId("branch"),
    sequence=CheckpointSequence(0),
)
POLICY = StoragePolicy(
    local=LocalResponseStorage.PROCESS_LOCAL,
    upstream=ProviderLaneStorage.OFF,
)
TYPED_CHECKPOINT = checkpoint_id(CHECKPOINT.checkpoint_id)
