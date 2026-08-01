"""Provide deterministic typed fixtures for conversation contract tests."""

from asyncio import Event
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

from avalan.conversation import (
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    CanonicalRequestDigest,
    CheckpointId,
    CheckpointSequence,
    ConversationAgentId,
    ConversationBranchId,
    ConversationId,
    ConversationModelCallId,
    ConversationTaskId,
    ExecutionSegmentId,
    FailureBoundary,
    LogicalTurnId,
    NamedHeadId,
    ProviderLaneId,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    StructuredInputContinuationId,
    UpstreamResponseId,
)


class ManualConversationClock:
    """Provide manually advanced aware wall-clock time."""

    def __init__(self, initial: datetime) -> None:
        assert isinstance(initial, datetime)
        assert initial.utcoffset() is not None
        self._now = initial

    def now(self) -> datetime:
        """Return the current deterministic instant."""
        return self._now

    def advance(self, seconds: int) -> None:
        """Advance the clock without sleeping."""
        assert type(seconds) is int and seconds >= 0
        self._now += timedelta(seconds=seconds)


class ConversationIdFactory:
    """Create deterministic, statically distinct conversation IDs."""

    def __init__(self, prefix: str) -> None:
        assert prefix and prefix == prefix.strip()
        self._prefix = prefix
        self._sequence = 0

    def _next(self, kind: str) -> str:
        assert kind
        self._sequence += 1
        return f"{self._prefix}-{kind}-{self._sequence:04d}"

    def conversation_id(self) -> ConversationId:
        """Return the next conversation identifier."""
        return ConversationId(self._next("conversation"))

    def logical_turn_id(self) -> LogicalTurnId:
        """Return the next logical-turn identifier."""
        return LogicalTurnId(self._next("logical-turn"))

    def execution_segment_id(self) -> ExecutionSegmentId:
        """Return the next execution-segment identifier."""
        return ExecutionSegmentId(self._next("execution-segment"))

    def checkpoint_id(self) -> CheckpointId:
        """Return the next checkpoint identifier."""
        return CheckpointId(self._next("checkpoint"))

    def branch_id(self) -> ConversationBranchId:
        """Return the next branch identifier."""
        return ConversationBranchId(self._next("branch"))

    def named_head_id(self) -> NamedHeadId:
        """Return the next named-head identifier."""
        return NamedHeadId(self._next("named-head"))

    def provider_lane_id(self) -> ProviderLaneId:
        """Return the next provider-lane identifier."""
        return ProviderLaneId(self._next("provider-lane"))

    def model_call_id(self) -> ConversationModelCallId:
        """Return the next model-call identifier."""
        return ConversationModelCallId(self._next("model-call"))

    def provisional_response_id(self) -> ProvisionalResponseId:
        """Return the next provisional response identifier."""
        return ProvisionalResponseId(self._next("provisional-response"))

    def public_response_id(self) -> PublicResponseId:
        """Return the next public response identifier."""
        return PublicResponseId(self._next("public-response"))

    def upstream_response_id(self) -> UpstreamResponseId:
        """Return the next upstream response identifier."""
        return UpstreamResponseId(self._next("upstream-response"))

    def task_id(self) -> ConversationTaskId:
        """Return the next task identifier."""
        return ConversationTaskId(self._next("task"))

    def agent_id(self) -> ConversationAgentId:
        """Return the next agent identifier."""
        return ConversationAgentId(self._next("agent"))

    def structured_input_continuation_id(
        self,
    ) -> StructuredInputContinuationId:
        """Return the next structured-input continuation identifier."""
        return StructuredInputContinuationId(
            self._next("structured-input-continuation")
        )

    def tenant_id(self) -> AuthorityTenantId:
        """Return the next authority tenant identifier."""
        return AuthorityTenantId(self._next("tenant"))

    def principal_id(self) -> AuthorityPrincipalId:
        """Return the next authority principal identifier."""
        return AuthorityPrincipalId(self._next("principal"))

    def endpoint_id(self) -> AuthorityEndpointId:
        """Return the next authority endpoint identifier."""
        return AuthorityEndpointId(self._next("endpoint"))

    def idempotency_key(self) -> RequestIdempotencyKey:
        """Return the next request idempotency key."""
        return RequestIdempotencyKey(self._next("idempotency-key"))

    def request_digest(self) -> CanonicalRequestDigest:
        """Return the next canonical request digest."""
        return CanonicalRequestDigest(self._next("request-digest"))

    def checkpoint_sequence(self) -> CheckpointSequence:
        """Return the current deterministic sequence as a typed revision."""
        return CheckpointSequence(self._sequence)


class AsyncConversationBarrier:
    """Release deterministic competitors after every party arrives."""

    def __init__(self, parties: int) -> None:
        assert type(parties) is int and parties > 0
        self._parties = parties
        self._arrivals: list[str] = []
        self._released = Event()

    @property
    def arrivals(self) -> tuple[str, ...]:
        """Return arrivals in deterministic observation order."""
        return tuple(self._arrivals)

    async def arrive_and_wait(self, participant: str) -> None:
        """Wait until every configured participant has arrived."""
        assert participant and participant == participant.strip()
        if participant in self._arrivals:
            raise ValueError("barrier participant arrived more than once")
        if len(self._arrivals) >= self._parties:
            raise RuntimeError("barrier received too many participants")
        self._arrivals.append(participant)
        if len(self._arrivals) == self._parties:
            self._released.set()
        await self._released.wait()


class InjectedConversationFault(RuntimeError):
    """Represent one deterministic contract-test fault."""

    def __init__(self, boundary: FailureBoundary) -> None:
        assert isinstance(boundary, FailureBoundary)
        self.boundary = boundary
        super().__init__(boundary.value)


class ConversationFaultInjector:
    """Inject explicitly armed failures only at awaited boundaries."""

    def __init__(self, armed: Iterable[FailureBoundary]) -> None:
        self._armed = frozenset(armed)
        assert all(isinstance(item, FailureBoundary) for item in self._armed)
        self._visited: list[FailureBoundary] = []

    @property
    def visited(self) -> tuple[FailureBoundary, ...]:
        """Return boundaries visited in call order."""
        return tuple(self._visited)

    async def reach(self, boundary: FailureBoundary) -> None:
        """Raise when the awaited boundary is armed."""
        assert isinstance(boundary, FailureBoundary)
        self._visited.append(boundary)
        if boundary in self._armed:
            raise InjectedConversationFault(boundary)


@dataclass(frozen=True, slots=True, kw_only=True)
class TestResponseResource:
    """Describe one deterministic response publication fixture."""

    __test__: ClassVar[bool] = False

    name: str
    terminal_publication_allowed: bool

    def __post_init__(self) -> None:
        assert self.name and self.name == self.name.strip()
        assert type(self.terminal_publication_allowed) is bool


def fixture_authority_scope() -> AuthorityScope:
    """Return the canonical deterministic authority scope."""
    return AuthorityScope(
        source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=AuthorityTenantId("tenant-fixture"),
        principal_id=AuthorityPrincipalId("principal-fixture"),
        agent_id=ConversationAgentId("agent-fixture"),
        endpoint_id=AuthorityEndpointId("endpoint-fixture"),
        network_exposed=True,
    )
