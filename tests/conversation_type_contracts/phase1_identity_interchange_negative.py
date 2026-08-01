"""Reject interchange across public, private, execution, and revision IDs."""

from avalan.conversation import (
    CheckpointId,
    CheckpointSequence,
    ConversationAgentId,
    ConversationId,
    ConversationModelCallId,
    ConversationTaskId,
    NamedHeadRevision,
    ProviderApiRevision,
    ProviderLaneId,
    ProviderSdkRevision,
    PublicResponseId,
    UpstreamResponseId,
)


def checkpoint_id(value: CheckpointId) -> CheckpointId:
    """Return one checkpoint identifier."""
    return value


def public_response_id(value: PublicResponseId) -> PublicResponseId:
    """Return one outward public response identifier."""
    return value


def upstream_response_id(value: UpstreamResponseId) -> UpstreamResponseId:
    """Return one private upstream response identifier."""
    return value


def conversation_id(value: ConversationId) -> ConversationId:
    """Return one logical conversation identifier."""
    return value


def provider_lane_id(value: ProviderLaneId) -> ProviderLaneId:
    """Return one provider lane identifier."""
    return value


def conversation_task_id(value: ConversationTaskId) -> ConversationTaskId:
    """Return one task identifier."""
    return value


def named_head_revision(value: NamedHeadRevision) -> NamedHeadRevision:
    """Return one named-head revision."""
    return value


def provider_sdk_revision(
    value: ProviderSdkRevision,
) -> ProviderSdkRevision:
    """Return one provider SDK revision."""
    return value


PUBLIC_ID = PublicResponseId("public-response")
UPSTREAM_ID = UpstreamResponseId("upstream-response")
CHECKPOINT_ID = CheckpointId("checkpoint")
MODEL_CALL_ID = ConversationModelCallId("model-call")
AGENT_ID = ConversationAgentId("agent")
CHECKPOINT_SEQUENCE = CheckpointSequence(1)
API_REVISION = ProviderApiRevision("api-revision")

INVALID_CHECKPOINT_ID = checkpoint_id(PUBLIC_ID)
INVALID_UPSTREAM_FROM_CHECKPOINT = upstream_response_id(CHECKPOINT_ID)
INVALID_UPSTREAM_FROM_PUBLIC = upstream_response_id(PUBLIC_ID)
INVALID_PUBLIC_FROM_UPSTREAM = public_response_id(UPSTREAM_ID)
INVALID_CONVERSATION_ID = conversation_id(CHECKPOINT_ID)
INVALID_LANE_ID = provider_lane_id(MODEL_CALL_ID)
INVALID_TASK_ID = conversation_task_id(AGENT_ID)
INVALID_HEAD_REVISION = named_head_revision(CHECKPOINT_SEQUENCE)
INVALID_SDK_REVISION = provider_sdk_revision(API_REVISION)
