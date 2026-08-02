from ..agent import Specification
from ..conversation.binding import ProviderLaneBinding
from ..conversation.contract import AuthorityScope
from ..conversation.errors import (
    ConversationCapabilityError,
    ConversationValidationError,
)
from ..conversation.state import ConversationCheckpoint
from ..entities import EngineUri, Input, Operation
from .capability import ModelCapabilityCatalog

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from ..agent.execution import AgentExecution, BranchInteractionBroker
    from ..conversation.protocols import ConversationCoordinator
    from ..interaction.entities import ExecutionOrigin
    from .engine import Engine


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCallContext:
    specification: Specification
    input: Input | None
    capability: ModelCapabilityCatalog | None = None
    engine_args: dict[str, Any] = field(default_factory=dict)
    parent: "ModelCallContext | None" = None
    root_parent: "ModelCallContext | None" = None
    agent_id: UUID | None = None
    participant_id: UUID | None = None
    session_id: UUID | None = None
    execution: "AgentExecution | None" = None
    execution_origin: "ExecutionOrigin | None" = None
    interaction_broker: "BranchInteractionBroker | None" = None
    conversation_coordinator: "ConversationCoordinator | None" = None
    conversation_authority: AuthorityScope | None = None
    conversation_lane: ProviderLaneBinding | None = None
    conversation_checkpoint: ConversationCheckpoint | None = None

    def __post_init__(self) -> None:
        configured = (
            self.conversation_coordinator is not None,
            self.conversation_authority is not None,
            self.conversation_lane is not None,
            self.conversation_checkpoint is not None,
        )
        if not any(configured):
            return
        if not all(configured[:3]):
            raise ConversationValidationError()
        coordinator = self.conversation_coordinator
        if coordinator is None or any(
            not callable(getattr(coordinator, method, None))
            for method in ("execute", "stream", "stream_with_sink", "compact")
        ):
            raise ConversationValidationError()
        authority = self.conversation_authority
        lane = self.conversation_lane
        if type(authority) is not AuthorityScope:
            raise ConversationValidationError()
        if type(lane) is not ProviderLaneBinding:
            raise ConversationValidationError()
        if lane.agent_id != authority.agent_id:
            raise ConversationValidationError()
        checkpoint = self.conversation_checkpoint
        if checkpoint is None:
            return
        if (
            type(checkpoint) is not ConversationCheckpoint
            or checkpoint.authority != authority
        ):
            raise ConversationValidationError()
        checkpoint_lanes = {
            item.lane_id: item for item in checkpoint.content.lanes
        }
        checkpoint_lane = checkpoint_lanes.get(lane.lane_id)
        if checkpoint_lane is None:
            raise ConversationValidationError()
        checkpoint_lane.binding.assert_compatible(lane)


def validate_native_model_call_context(
    context: ModelCallContext | None,
) -> None:
    """Reject active conversation state before native model dispatch."""
    if context is None:
        return
    if type(context) is not ModelCallContext:
        raise ConversationValidationError()
    if any(
        value is not None
        for value in (
            context.conversation_coordinator,
            context.conversation_authority,
            context.conversation_lane,
            context.conversation_checkpoint,
        )
    ):
        # Native provider lanes remain intentionally dormant in Phase 4.
        raise ConversationCapabilityError()


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCall:
    engine_uri: EngineUri
    model: "Engine"
    operation: Operation
    capability: ModelCapabilityCatalog | None = None
    context: ModelCallContext

    def __post_init__(self) -> None:
        context_capability = self.context.capability
        if self.capability is None:
            if context_capability is not None:
                object.__setattr__(self, "capability", context_capability)
            return
        if context_capability is None:
            object.__setattr__(
                self,
                "context",
                replace(self.context, capability=self.capability),
            )
            return
        assert (
            context_capability is self.capability
        ), "model call and context capabilities must be identical"
