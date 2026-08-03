"""Integrate agent execution with canonical conversation ownership."""

from ..types import JsonValue
from .binding import ProviderLaneBinding
from .contract import (
    CONVERSATION_CONTRACT_VERSION,
    AuthorityScope,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointSequence,
    ChildLaneRetentionPolicy,
    ConversationAgentId,
    ConversationBranchId,
    ConversationId,
    ConversationModelCallId,
    ConversationOperation,
    ConversationSurface,
    ExecutionSegmentId,
    LogicalTurnId,
    PortableContinuationReference,
    ProviderLaneId,
    ProviderLaneOwnerKind,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    RetentionLimits,
    SurfaceDisposition,
)
from .errors import ConversationCapabilityError, ConversationValidationError
from .execution import (
    AgentStructuredInputRequested,
    ProviderExecutionSegmentPhase,
    ProviderToolExecution,
)
from .items import (
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemPhase,
    VisibleTranscriptEntry,
    VisibleTranscriptRole,
)
from .observability import ConversationRequestSemantics
from .protocols import (
    ConversationCoordinator,
    ConversationUnitOfWork,
    ProviderPlan,
    ProviderResult,
    StatelessProviderPlan,
)
from .runtime import (
    AtomicCommitReceipt,
    ConversationCommitBoundary,
    ConversationLaneRequest,
    ConversationRunRequest,
    ExplicitBranchAdvance,
    FirstTurnAdvance,
    NamedHeadAdvance,
    OrdinaryChildAdvance,
)
from .settings import (
    CompactionPolicy,
    ConversationMode,
    DisabledCompaction,
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    ProviderUsage,
    ReasoningContext,
)
from .state import (
    CheckpointLifecycle,
    ConversationCheckpoint,
    ProviderLaneLifecycle,
    ProviderLaneTopology,
    ProviderLaneTopologyEntry,
    validate_checkpoint_parent_kind,
)
from .value import (
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    canonical_json_bytes,
    validate_identifier,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from inspect import isawaitable
from types import MappingProxyType
from typing import NewType, Protocol, cast, final

AgentTopologyPath = NewType("AgentTopologyPath", str)
AgentModelSlot = NewType("AgentModelSlot", str)


_SURFACE_DISPOSITIONS: Mapping[
    ConversationSurface,
    SurfaceDisposition,
] = MappingProxyType(
    {
        ConversationSurface.DIRECT_MODEL_SDK: SurfaceDisposition.ACTIVATED,
        ConversationSurface.AGENT_SDK: SurfaceDisposition.ACTIVATED,
        ConversationSurface.CLI: SurfaceDisposition.DEFERRED,
        ConversationSurface.FLOW: SurfaceDisposition.DEFERRED,
        ConversationSurface.MCP: SurfaceDisposition.DEFERRED,
        ConversationSurface.A2A: SurfaceDisposition.DEFERRED,
        ConversationSurface.SERVED_RESPONSES: SurfaceDisposition.ACTIVATED,
    }
)


def agent_conversation_surface_disposition(
    surface: ConversationSurface,
) -> SurfaceDisposition:
    """Return the explicit Phase 8 continuity disposition for one surface."""
    if not isinstance(surface, ConversationSurface):
        raise ConversationValidationError()
    return _SURFACE_DISPOSITIONS[surface]


def require_agent_conversation_surface(
    surface: ConversationSurface,
) -> None:
    """Reject every continuity surface not explicitly activated."""
    if agent_conversation_surface_disposition(surface) is not (
        SurfaceDisposition.ACTIVATED
    ):
        raise ConversationCapabilityError()


def parent_agent_topology_path(
    agent_id: ConversationAgentId,
    model_slot: AgentModelSlot,
) -> AgentTopologyPath:
    """Return the stable topology path for one parent-agent model lane."""
    validate_identifier(agent_id, "agent_id")
    validate_identifier(model_slot, "model_slot")
    return AgentTopologyPath(f"agent/{agent_id}/{model_slot}")


def child_agent_topology_path(
    parent_path: AgentTopologyPath,
    agent_id: ConversationAgentId,
    model_slot: AgentModelSlot,
) -> AgentTopologyPath:
    """Return the stable topology path for one child-agent model lane."""
    validate_identifier(parent_path, "parent_path")
    validate_identifier(agent_id, "agent_id")
    validate_identifier(model_slot, "model_slot")
    return AgentTopologyPath(f"{parent_path}/child/{agent_id}/{model_slot}")


def direct_model_topology_path(
    model_slot: AgentModelSlot,
) -> AgentTopologyPath:
    """Return the stable topology path for one direct-model lane."""
    validate_identifier(model_slot, "model_slot")
    return AgentTopologyPath(f"direct/{model_slot}")


def derive_agent_provider_lane_id(
    *,
    conversation_id: ConversationId,
    owner_kind: ProviderLaneOwnerKind,
    topology_path: AgentTopologyPath,
    model_slot: AgentModelSlot,
    binding: ProviderLaneBinding,
) -> ProviderLaneId:
    """Derive one deterministic provider lane from frozen identity fields."""
    validate_identifier(conversation_id, "conversation_id")
    if not isinstance(owner_kind, ProviderLaneOwnerKind):
        raise ConversationValidationError()
    validate_identifier(topology_path, "topology_path")
    validate_identifier(model_slot, "model_slot")
    if type(binding) is not ProviderLaneBinding:
        raise ConversationValidationError()
    values = (
        str(CONVERSATION_CONTRACT_VERSION),
        str(conversation_id),
        owner_kind.value,
        str(topology_path),
        str(model_slot),
        binding.provider_family.value,
        binding.normalized_endpoint,
        binding.azure_resource_identity or "",
        binding.model_or_deployment,
        str(binding.model_configuration_revision),
        str(binding.execution_definition_revision),
    )
    encoded = "".join(f"{len(value)}:{value}" for value in values).encode(
        "utf-8"
    )
    return ProviderLaneId(f"lane-v1-{sha256(encoded).hexdigest()}")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentProviderLane:
    """Bind one parent, child, or direct model to an isolated provider lane."""

    owner_kind: ProviderLaneOwnerKind
    agent_id: ConversationAgentId
    topology_path: AgentTopologyPath
    model_slot: AgentModelSlot
    binding: ProviderLaneBinding
    retention_policy: ChildLaneRetentionPolicy
    parent_lane_id: ProviderLaneId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner_kind, ProviderLaneOwnerKind):
            raise ConversationValidationError()
        validate_identifier(self.agent_id, "agent_id")
        validate_identifier(self.topology_path, "topology_path")
        validate_identifier(self.model_slot, "model_slot")
        if (
            type(self.binding) is not ProviderLaneBinding
            or self.binding.agent_id != self.agent_id
            or not isinstance(
                self.retention_policy,
                ChildLaneRetentionPolicy,
            )
        ):
            raise ConversationValidationError()
        child = self.owner_kind is ProviderLaneOwnerKind.CHILD_AGENT
        if child != (self.parent_lane_id is not None):
            raise ConversationValidationError()
        if self.parent_lane_id is not None:
            validate_identifier(self.parent_lane_id, "parent_lane_id")
        expected_prefix = {
            ProviderLaneOwnerKind.DIRECT_MODEL: "direct/",
            ProviderLaneOwnerKind.PARENT_AGENT: f"agent/{self.agent_id}/",
            ProviderLaneOwnerKind.CHILD_AGENT: "agent/",
        }[self.owner_kind]
        if not str(self.topology_path).startswith(expected_prefix):
            raise ConversationValidationError()
        expected_suffix = (
            f"/child/{self.agent_id}/{self.model_slot}"
            if child
            else f"/{self.model_slot}"
        )
        if not str(self.topology_path).endswith(expected_suffix):
            raise ConversationValidationError()

    @property
    def lane_id(self) -> ProviderLaneId:
        """Return the exact lane identity held by the provider binding."""
        return self.binding.lane_id

    def validate_identity(self, conversation_id: ConversationId) -> None:
        """Reject a binding whose lane ID was derived for another topology."""
        expected = derive_agent_provider_lane_id(
            conversation_id=conversation_id,
            owner_kind=self.owner_kind,
            topology_path=self.topology_path,
            model_slot=self.model_slot,
            binding=self.binding,
        )
        if self.lane_id != expected:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentLaneTopology:
    """Record deterministic isolated lanes and child retention policy."""

    conversation_id: ConversationId
    lanes: tuple[AgentProviderLane, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.conversation_id, "conversation_id")
        if (
            type(self.lanes) is not tuple
            or not self.lanes
            or any(type(lane) is not AgentProviderLane for lane in self.lanes)
        ):
            raise ConversationValidationError()
        lane_ids = tuple(lane.lane_id for lane in self.lanes)
        paths = tuple(lane.topology_path for lane in self.lanes)
        if len(lane_ids) != len(set(lane_ids)) or len(paths) != len(
            set(paths)
        ):
            raise ConversationValidationError()
        by_id = {lane.lane_id: lane for lane in self.lanes}
        for lane in self.lanes:
            lane.validate_identity(self.conversation_id)
            if lane.owner_kind is ProviderLaneOwnerKind.CHILD_AGENT:
                assert lane.parent_lane_id is not None
                parent = by_id.get(lane.parent_lane_id)
                if (
                    parent is None
                    or parent.owner_kind
                    is not ProviderLaneOwnerKind.PARENT_AGENT
                    or not str(lane.topology_path).startswith(
                        f"{parent.topology_path}/child/"
                    )
                ):
                    raise ConversationValidationError()

    @property
    def parent_lanes(self) -> tuple[AgentProviderLane, ...]:
        """Return parent lanes in their frozen topology order."""
        return tuple(
            lane
            for lane in self.lanes
            if lane.owner_kind is ProviderLaneOwnerKind.PARENT_AGENT
        )

    @property
    def child_lanes(self) -> tuple[AgentProviderLane, ...]:
        """Return child lanes in their frozen topology order."""
        return tuple(
            lane
            for lane in self.lanes
            if lane.owner_kind is ProviderLaneOwnerKind.CHILD_AGENT
        )

    def outward_child_results(
        self,
        outputs: Mapping[ProviderLaneId, tuple[VisibleTranscriptEntry, ...]],
    ) -> tuple[VisibleTranscriptEntry, ...]:
        """Merge only canonical visible child results in topology order."""
        if not isinstance(outputs, Mapping):
            raise ConversationValidationError()
        child_ids = {lane.lane_id for lane in self.child_lanes}
        if any(lane_id not in child_ids for lane_id in outputs):
            raise ConversationValidationError()
        merged: list[VisibleTranscriptEntry] = []
        for lane in self.child_lanes:
            entries = outputs.get(lane.lane_id, ())
            if type(entries) is not tuple or any(
                type(entry) is not VisibleTranscriptEntry for entry in entries
            ):
                raise ConversationValidationError()
            merged.extend(entries)
        return tuple(merged)

    def checkpoint_topology(self) -> ProviderLaneTopology:
        """Return provider-state-free topology for checkpoint persistence."""
        return ProviderLaneTopology(
            schema_version=1,
            entries=tuple(
                ProviderLaneTopologyEntry(
                    lane_id=lane.lane_id,
                    owner_kind=lane.owner_kind,
                    agent_id=lane.agent_id,
                    topology_path=str(lane.topology_path),
                    model_slot=str(lane.model_slot),
                    retention_policy=lane.retention_policy,
                    binding_digest=lane.binding.integrity_digest,
                    parent_lane_id=lane.parent_lane_id,
                )
                for lane in self.lanes
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentConversationLane:
    """Select exact conversation behavior for one topology lane."""

    lane_id: ProviderLaneId
    mode: ConversationMode
    reasoning_context: ReasoningContext = ReasoningContext.AUTO
    compaction: CompactionPolicy = DisabledCompaction()

    def __post_init__(self) -> None:
        ConversationLaneRequest(
            lane_id=self.lane_id,
            mode=self.mode,
            reasoning_context=self.reasoning_context,
            compaction=self.compaction,
        )

    def request(self) -> ConversationLaneRequest:
        """Return the closed coordinator lane request."""
        return ConversationLaneRequest(
            lane_id=self.lane_id,
            mode=self.mode,
            reasoning_context=self.reasoning_context,
            compaction=self.compaction,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentConversationResult:
    """Return one completed outward agent turn and safe child projection."""

    receipt: AtomicCommitReceipt
    output: str
    child_results: tuple[VisibleTranscriptEntry, ...]

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not AtomicCommitReceipt
            or type(self.output) is not str
            or type(self.child_results) is not tuple
            or any(
                type(entry) is not VisibleTranscriptEntry
                for entry in self.child_results
            )
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AgentConversationExecutionSegmentCandidate:
    """Carry one runtime trace until the coordinator binds its identity."""

    segment_index: int
    phase: ProviderExecutionSegmentPhase
    items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    tools: tuple[ProviderToolExecution, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.segment_index) is not int
            or self.segment_index < 0
            or not isinstance(self.phase, ProviderExecutionSegmentPhase)
            or type(self.items) is not tuple
            or any(type(item) is not ProviderItem for item in self.items)
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or type(self.usage) is not ProviderUsage
            or type(self.tools) is not tuple
            or any(
                type(tool) is not ProviderToolExecution for tool in self.tools
            )
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AgentConversationLaneInvocationResult:
    """Return canonical output and its exact provider/tool boundaries."""

    result: ProviderResult
    segments: tuple[AgentConversationExecutionSegmentCandidate, ...]

    def __post_init__(self) -> None:
        if (
            type(self.result) is not ProviderResult
            or type(self.segments) is not tuple
            or not self.segments
            or any(
                type(segment) is not AgentConversationExecutionSegmentCandidate
                for segment in self.segments
            )
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AgentConversationLaneInvocation:
    """Bind one runtime-only lane dispatcher to an exact provider binding."""

    binding: ProviderLaneBinding
    dispatch: Callable[
        [ProviderPlan],
        Awaitable[AgentConversationLaneInvocationResult],
    ] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not ProviderLaneBinding or not callable(
            self.dispatch
        ):
            raise ConversationValidationError()

    @property
    def lane_id(self) -> ProviderLaneId:
        """Return the exact lane selected by this invocation."""
        return self.binding.lane_id


class AgentConversationInvocationAdapter(Protocol):
    """Execute one prepared agent model call through canonical ownership."""

    async def execute(
        self,
        turn: "AgentConversationTurn",
        input: str,
        prepared_model_call: object,
        invoke_model: Callable[[object], Awaitable[object]],
    ) -> AgentConversationResult:
        """Run one invocation without retaining its callbacks."""
        raise NotImplementedError


def agent_conversation_provider_result(
    plan: ProviderPlan,
    output: str,
    *,
    invocation_id: str,
) -> ProviderResult:
    """Return one canonical result from a completed real agent invocation."""
    if (
        type(plan) is not StatelessProviderPlan
        or type(output) is not str
        or not output.strip()
    ):
        raise ConversationValidationError()
    validate_identifier(invocation_id, "invocation_id")
    suffix = sha256(
        canonical_json_bytes(
            {
                "binding_digest": plan.binding.integrity_digest,
                "invocation_id": invocation_id,
                "ledger_length": len(plan.ledger.items),
            }
        )
    ).hexdigest()
    item_id = ProviderItemId(f"agent-item-{suffix}")
    item = ProviderItem(
        item_id=item_id,
        lane_id=plan.binding.lane_id,
        model_call_id=ConversationModelCallId(f"agent-model-call-{suffix}"),
        kind=ProviderItemKind.MESSAGE,
        order=ProviderItemOrder(len(plan.ledger.items)),
        provider_index=ProviderItemIndex(0),
        phase=ProviderItemPhase.FINAL,
        caller=ProviderItemCaller.PROVIDER,
        canonical_input={
            "content": (
                {
                    "annotations": (),
                    "text": output,
                    "type": "output_text",
                },
            ),
            "id": item_id,
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        normalization_version=plan.binding.continuation_codec_version,
    )
    return ProviderResult(
        items=(item,),
        reasoning=EffectiveReasoningMetadata(
            requested=plan.reasoning.requested,
            effective=EffectiveReasoningContext.CURRENT_TURN,
        ),
        usage=ProviderUsage(),
    )


@final
class AgentConversationSuspensionBoundary(RuntimeError):
    """Carry one staged checkpoint after a durable provider-call fence."""

    request: AgentStructuredInputRequested
    call: ProviderItem
    tool: ProviderToolExecution
    checkpoint: ConversationCheckpoint

    def __init__(
        self,
        *,
        request: AgentStructuredInputRequested,
        call: ProviderItem,
        tool: ProviderToolExecution,
        checkpoint: ConversationCheckpoint,
    ) -> None:
        segments = checkpoint.content.execution_segments
        if (
            type(request) is not AgentStructuredInputRequested
            or type(call) is not ProviderItem
            or type(tool) is not ProviderToolExecution
            or type(checkpoint) is not ConversationCheckpoint
            or checkpoint.kind
            is not CheckpointKind.STRUCTURED_INPUT_SUSPENSION
            or checkpoint.lifecycle is not CheckpointLifecycle.STAGED
            or not segments
            or call.call_id != tool.call_id
            or request.arguments != tool.arguments
            or segments[-1].lane_id != call.lane_id
            or tool not in segments[-1].tools
            or not any(
                lane.lane_id == call.lane_id
                and lane.lifecycle is ProviderLaneLifecycle.SUSPENDED
                for lane in checkpoint.content.lanes
            )
        ):
            raise ConversationValidationError()
        self.request = request
        self.call = call
        self.tool = tool
        self.checkpoint = checkpoint
        super().__init__("agent conversation input required")

    def __repr__(self) -> str:
        """Return content-free suspension metadata."""
        return "AgentConversationSuspensionBoundary(content=<redacted>)"


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AgentConversationTurn:
    """Own one immutable per-invocation coordinated agent turn."""

    coordinator: ConversationCoordinator
    authority: AuthorityScope
    topology: AgentLaneTopology
    lanes: tuple[AgentConversationLane, ...]
    logical_turn_id: LogicalTurnId
    execution_segment_id: ExecutionSegmentId
    checkpoint_id: CheckpointId
    branch_id: ConversationBranchId
    provisional_response_id: ProvisionalResponseId
    public_response_id: PublicResponseId
    idempotency_key: RequestIdempotencyKey
    retention: RetentionLimits
    parent: ConversationCheckpoint | None = None
    advance: ExplicitBranchAdvance | NamedHeadAdvance | None = None

    def __post_init__(self) -> None:
        if (
            not callable(getattr(self.coordinator, "execute", None))
            or type(self.authority) is not AuthorityScope
            or type(self.topology) is not AgentLaneTopology
            or type(self.lanes) is not tuple
            or not self.lanes
            or any(
                type(lane) is not AgentConversationLane for lane in self.lanes
            )
            or type(self.retention) is not RetentionLimits
        ):
            raise ConversationValidationError()
        for value, name in (
            (self.logical_turn_id, "logical_turn_id"),
            (self.execution_segment_id, "execution_segment_id"),
            (self.checkpoint_id, "checkpoint_id"),
            (self.branch_id, "branch_id"),
            (self.provisional_response_id, "provisional_response_id"),
            (self.public_response_id, "public_response_id"),
            (self.idempotency_key, "idempotency_key"),
        ):
            validate_identifier(value, name)
        if self.topology.conversation_id != (
            self.parent.identity.conversation_id
            if self.parent is not None
            else self.topology.conversation_id
        ):
            raise ConversationValidationError()
        lane_ids = tuple(lane.lane_id for lane in self.lanes)
        if len(lane_ids) != len(set(lane_ids)) or set(lane_ids) != {
            lane.lane_id for lane in self.topology.lanes
        }:
            raise ConversationValidationError()
        root_lanes = tuple(
            lane
            for lane in self.topology.lanes
            if lane.owner_kind
            in {
                ProviderLaneOwnerKind.DIRECT_MODEL,
                ProviderLaneOwnerKind.PARENT_AGENT,
            }
        )
        if not root_lanes or any(
            lane.agent_id != self.authority.agent_id for lane in root_lanes
        ):
            raise ConversationValidationError()
        if self.topology.child_lanes and not self.topology.parent_lanes:
            raise ConversationValidationError()
        if self.parent is not None:
            validate_checkpoint_parent_kind(
                CheckpointKind.COMPLETED_OUTWARD_TURN,
                self.parent.kind,
            )
            if (
                type(self.parent) is not ConversationCheckpoint
                or self.parent.lifecycle is not CheckpointLifecycle.COMMITTED
                or self.parent.authority != self.authority
            ):
                raise ConversationValidationError()
            if self.advance is None:
                if self.parent.identity.branch_id != self.branch_id:
                    raise ConversationValidationError()
            elif isinstance(self.advance, ExplicitBranchAdvance):
                if (
                    self.advance.parent_checkpoint_id
                    != self.parent.identity.checkpoint_id
                    or self.advance.branch_id != self.branch_id
                    or self.parent.identity.branch_id == self.branch_id
                ):
                    raise ConversationValidationError()
            elif isinstance(self.advance, NamedHeadAdvance):
                if (
                    self.advance.parent_checkpoint_id
                    != self.parent.identity.checkpoint_id
                    or self.parent.identity.branch_id != self.branch_id
                ):
                    raise ConversationValidationError()
            else:
                raise ConversationValidationError()
            if (
                self.parent.content.lane_topology is not None
                and self.parent.content.lane_topology
                != self.topology.checkpoint_topology()
            ):
                raise ConversationValidationError()
            topology_by_lane = {
                lane.lane_id: lane for lane in self.topology.lanes
            }
            for snapshot in self.parent.content.lanes:
                expected = topology_by_lane.get(snapshot.lane_id)
                if expected is None:
                    raise ConversationValidationError()
                snapshot.binding.assert_compatible(expected.binding)
        elif self.advance is not None:
            raise ConversationValidationError()

    @property
    def conversation_id(self) -> ConversationId:
        """Return the exact conversation identifier for this invocation."""
        return self.topology.conversation_id

    async def execute(
        self,
        input: str,
        *,
        lane_invocations: (
            tuple[AgentConversationLaneInvocation, ...] | None
        ) = (None),
    ) -> AgentConversationResult:
        """Run child lanes, merge safe output, and commit one outward turn."""
        if type(input) is not str or not input or not input.strip():
            raise ConversationValidationError()
        if len(input.encode("utf-8")) > 1_048_576:
            raise ConversationValidationError()
        if lane_invocations is not None:
            return await self._execute_runtime_invocation(
                input,
                lane_invocations,
            )
        child_lanes = self.topology.child_lanes
        child_results: tuple[VisibleTranscriptEntry, ...] = ()
        parent = self.parent
        if child_lanes:
            child_request = self._child_request(input)
            child_receipt = await self.coordinator.execute(child_request)
            child_results = self._safe_child_results(child_receipt)
        outward_request = self._outward_request(
            input,
            parent=parent,
            child_results=child_results,
        )
        receipt = await self.coordinator.execute(outward_request)
        output_entries = tuple(
            entry
            for candidate in receipt.output_candidates
            for entry in candidate.public_output.items
        )
        return AgentConversationResult(
            receipt=receipt,
            output="\n".join(entry.content for entry in output_entries),
            child_results=child_results,
        )

    async def _execute_runtime_invocation(
        self,
        input: str,
        lane_invocations: tuple[AgentConversationLaneInvocation, ...],
    ) -> AgentConversationResult:
        """Commit one callback-produced parent result through coordination."""
        parent_lanes = self.topology.parent_lanes
        topology_lanes = self.topology.lanes
        invocation_by_lane = {
            invocation.lane_id: invocation for invocation in lane_invocations
        }
        if (
            len(parent_lanes) != 1
            or type(lane_invocations) is not tuple
            or len(invocation_by_lane) != len(lane_invocations)
            or set(invocation_by_lane)
            != {lane.lane_id for lane in topology_lanes}
            or any(
                type(invocation) is not AgentConversationLaneInvocation
                or invocation.binding
                != next(
                    lane.binding
                    for lane in topology_lanes
                    if lane.lane_id == invocation.lane_id
                )
                for invocation in lane_invocations
            )
        ):
            raise ConversationCapabilityError()
        operation = getattr(self.coordinator, "execute_agent", None)
        if not callable(operation):
            raise ConversationCapabilityError()
        request = self._outward_request(
            input,
            parent=self.parent,
            child_results=(),
            include_children=True,
        )
        ordered_lanes = (
            *self.topology.child_lanes,
            *parent_lanes,
        )
        request_by_lane = {lane.lane_id: lane for lane in request.lanes}
        request = replace(
            request,
            lanes=tuple(
                request_by_lane[lane.lane_id] for lane in ordered_lanes
            ),
        )
        ordered_invocations = tuple(
            invocation_by_lane[lane.lane_id] for lane in ordered_lanes
        )
        pending = cast(
            Callable[
                [
                    ConversationRunRequest,
                    tuple[AgentConversationLaneInvocation, ...],
                ],
                Awaitable[AtomicCommitReceipt],
            ],
            operation,
        )(request, ordered_invocations)
        if not isawaitable(pending):
            raise ConversationCapabilityError()
        receipt = await pending
        child_ids = {lane.lane_id for lane in self.topology.child_lanes}
        child_results: tuple[VisibleTranscriptEntry, ...] = ()
        if child_ids:
            child_receipt = replace(
                receipt,
                result=None,
                outbox=None,
                output_candidates=tuple(
                    candidate
                    for candidate in receipt.output_candidates
                    if candidate.lane_id in child_ids
                ),
            )
            child_results = self._safe_child_results(child_receipt)
        parent_ids = {lane.lane_id for lane in parent_lanes}
        output_entries = tuple(
            entry
            for candidate in receipt.output_candidates
            if candidate.lane_id in parent_ids
            for entry in candidate.public_output.items
        )
        return AgentConversationResult(
            receipt=receipt,
            output="\n".join(entry.content for entry in output_entries),
            child_results=child_results,
        )

    async def stage_structured_input_suspension(
        self,
        checkpoint: ConversationCheckpoint,
        continuation: PortableContinuationReference,
    ) -> ConversationUnitOfWork:
        """Stage one exact agent suspension for host-level atomic commit."""
        if (
            type(checkpoint) is not ConversationCheckpoint
            or type(continuation) is not PortableContinuationReference
            or checkpoint.kind
            is not CheckpointKind.STRUCTURED_INPUT_SUSPENSION
            or checkpoint.lifecycle is not CheckpointLifecycle.STAGED
            or checkpoint.authority != self.authority
            or checkpoint.identity.conversation_id != self.conversation_id
            or checkpoint.identity.logical_turn_id != self.logical_turn_id
            or checkpoint.content.lane_topology
            != self.topology.checkpoint_topology()
        ):
            raise ConversationValidationError()
        stage = getattr(
            self.coordinator,
            "stage_structured_input_suspension",
            None,
        )
        if not callable(stage):
            raise ConversationCapabilityError()
        operation = cast(
            Callable[
                [ConversationCheckpoint, PortableContinuationReference],
                Awaitable[ConversationUnitOfWork],
            ],
            stage,
        )(checkpoint, continuation)
        if not isawaitable(operation):
            raise ConversationCapabilityError()
        return await operation

    def _child_request(self, input: str) -> ConversationRunRequest:
        seed = canonical_json_bytes(
            {
                "checkpoint_id": self.checkpoint_id,
                "kind": "agent-child-fanout",
            }
        )
        suffix = sha256(seed).hexdigest()
        identity = CheckpointIdentity(
            conversation_id=self.conversation_id,
            logical_turn_id=self.logical_turn_id,
            execution_segment_id=ExecutionSegmentId(
                f"agent-child-segment-{suffix}"
            ),
            checkpoint_id=CheckpointId(f"agent-child-checkpoint-{suffix}"),
            branch_id=self.branch_id,
            sequence=CheckpointSequence(0),
        )
        return self._request(
            input=input,
            identity=identity,
            advance=FirstTurnAdvance(),
            selected={lane.lane_id for lane in self.topology.child_lanes},
            boundary=ConversationCommitBoundary.INTERNAL_SEGMENT,
            visible_delta=(
                VisibleTranscriptEntry(
                    role=VisibleTranscriptRole.USER,
                    content=input,
                ),
            ),
            parent_checkpoint_id=None,
            idempotency_key=RequestIdempotencyKey(
                f"agent-child-{sha256(seed).hexdigest()}"
            ),
        )

    def _outward_request(
        self,
        input: str,
        *,
        parent: ConversationCheckpoint | None,
        child_results: tuple[VisibleTranscriptEntry, ...],
        include_children: bool = False,
    ) -> ConversationRunRequest:
        sequence = 0 if parent is None else parent.identity.sequence + 1
        identity = CheckpointIdentity(
            conversation_id=self.conversation_id,
            logical_turn_id=self.logical_turn_id,
            execution_segment_id=self.execution_segment_id,
            checkpoint_id=self.checkpoint_id,
            branch_id=self.branch_id,
            sequence=CheckpointSequence(sequence),
            parent_checkpoint_id=(
                parent.identity.checkpoint_id if parent is not None else None
            ),
            parent_sequence=(
                parent.identity.sequence if parent is not None else None
            ),
        )
        advance: (
            FirstTurnAdvance
            | OrdinaryChildAdvance
            | ExplicitBranchAdvance
            | NamedHeadAdvance
        ) = (
            FirstTurnAdvance()
            if parent is None
            else (
                self.advance
                or OrdinaryChildAdvance(
                    parent_checkpoint_id=parent.identity.checkpoint_id
                )
            )
        )
        child_text = "\n".join(entry.content for entry in child_results)
        provider_input = (
            input
            if not child_text
            else f"{input}\n\nCanonical child results:\n{child_text}"
        )
        visible_delta = (
            (
                VisibleTranscriptEntry(
                    role=VisibleTranscriptRole.USER,
                    content=input,
                ),
            )
            if parent is None
            or parent.identity.logical_turn_id != self.logical_turn_id
            else ()
        ) + child_results
        selected = {
            lane.lane_id
            for lane in self.topology.lanes
            if include_children
            or lane.owner_kind
            in {
                ProviderLaneOwnerKind.DIRECT_MODEL,
                ProviderLaneOwnerKind.PARENT_AGENT,
            }
        }
        return self._request(
            input=provider_input,
            identity=identity,
            advance=advance,
            selected=selected,
            boundary=ConversationCommitBoundary.OUTWARD_TURN,
            visible_delta=visible_delta,
            parent_checkpoint_id=(
                parent.identity.checkpoint_id if parent is not None else None
            ),
            idempotency_key=self.idempotency_key,
        )

    def _request(
        self,
        *,
        input: str,
        identity: CheckpointIdentity,
        advance: (
            FirstTurnAdvance
            | OrdinaryChildAdvance
            | ExplicitBranchAdvance
            | NamedHeadAdvance
        ),
        selected: set[ProviderLaneId],
        boundary: ConversationCommitBoundary,
        visible_delta: tuple[VisibleTranscriptEntry, ...],
        parent_checkpoint_id: CheckpointId | None,
        idempotency_key: RequestIdempotencyKey,
    ) -> ConversationRunRequest:
        outward = boundary is ConversationCommitBoundary.OUTWARD_TURN
        return ConversationRunRequest(
            semantics=ConversationRequestSemantics(
                authority=self.authority,
                operation=(
                    ConversationOperation.CREATE
                    if parent_checkpoint_id is None
                    else (
                        ConversationOperation.BRANCH
                        if isinstance(advance, ExplicitBranchAdvance)
                        else ConversationOperation.CONTINUE
                    )
                ),
                mode=self._aggregate_mode(selected),
                reasoning_context=ReasoningContext.AUTO,
                semantic_input={"text": input},
                parent_checkpoint_id=parent_checkpoint_id,
            ),
            identity=identity,
            advance=advance,
            lanes=tuple(
                lane.request()
                for lane in self.lanes
                if lane.lane_id in selected
            ),
            visible_delta=visible_delta,
            retention=self.retention,
            idempotency_key=idempotency_key,
            boundary=boundary,
            lane_topology=self.topology.checkpoint_topology(),
            provisional_response_id=(
                self.provisional_response_id if outward else None
            ),
            public_response_id=self.public_response_id if outward else None,
        )

    def _aggregate_mode(
        self,
        selected: set[ProviderLaneId],
    ) -> ConversationMode:
        modes = tuple(
            lane.mode for lane in self.lanes if lane.lane_id in selected
        )
        if not modes:
            raise ConversationCapabilityError()
        return modes[0]

    def _safe_child_results(
        self,
        receipt: AtomicCommitReceipt,
    ) -> tuple[VisibleTranscriptEntry, ...]:
        children = {lane.lane_id: lane for lane in self.topology.child_lanes}
        candidates = receipt.output_candidates
        candidate_ids = tuple(candidate.lane_id for candidate in candidates)
        if (
            len(candidate_ids) != len(set(candidate_ids))
            or set(candidate_ids) != set(children)
            or any(
                candidate.binding != children[candidate.lane_id].binding
                for candidate in candidates
            )
        ):
            raise ConversationValidationError()
        outputs = {
            candidate.lane_id: candidate.public_output.items
            for candidate in candidates
        }
        return self.topology.outward_child_results(outputs)

    def __repr__(self) -> str:
        """Return only content-free per-invocation metadata."""
        return (
            "AgentConversationTurn("
            f"conversation_id={self.conversation_id!r}, "
            f"lane_count={len(self.lanes)}, parent_present="
            f"{self.parent is not None})"
        )


def agent_topology_digest(topology: AgentLaneTopology) -> str:
    """Return a content-safe digest of exact lane ownership and retention."""
    if type(topology) is not AgentLaneTopology:
        raise ConversationValidationError()
    payload: JsonValue = {
        "conversation_id": topology.conversation_id,
        "lanes": tuple(
            {
                "agent_id": lane.agent_id,
                "binding_digest": lane.binding.integrity_digest,
                "lane_id": lane.lane_id,
                "model_slot": lane.model_slot,
                "owner_kind": lane.owner_kind.value,
                "parent_lane_id": lane.parent_lane_id,
                "retention_policy": lane.retention_policy.value,
                "topology_path": lane.topology_path,
            }
            for lane in topology.lanes
        ),
        "version": 1,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()
