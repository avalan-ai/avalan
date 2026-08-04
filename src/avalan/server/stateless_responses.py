"""Coordinate caller-held served Responses continuation state."""

from ..agent.conversation_child import AgentConversationChildBinding
from ..conversation.agent import AgentConversationTurn
from ..conversation.contract import (
    AuthorityScope,
    AuthoritySource,
    CheckpointId,
    LocalResponseStorage,
    NamedHeadId,
    NamedHeadRevision,
    ParentAdvanceMode,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    RetentionLimits,
)
from ..conversation.envelope import (
    ContinuationEnvelopeAdvance,
    ContinuationEnvelopeAuthority,
    ContinuationEnvelopeCodec,
    ContinuationEnvelopeToken,
    OpenedContinuationEnvelope,
)
from ..conversation.errors import (
    ConversationAuthorizationError,
    ConversationCapabilityError,
    ConversationCodecError,
    ConversationConflictError,
    ConversationCryptoAuthenticationError,
    ConversationKeyMissingError,
    ConversationStorageError,
    ConversationValidationError,
)
from ..conversation.observability import authority_digest
from ..conversation.runtime import (
    ExplicitBranchAdvance,
    NamedHeadAdvance,
    StoreCloseDisposition,
    StoreCloseResolution,
)
from ..conversation.settings import (
    ConversationResult,
    ProviderUsage,
    ReasoningContext,
)
from ..conversation.state import (
    CheckpointLifecycle,
    ConversationCheckpoint,
    StatelessProviderLaneSnapshot,
)
from ..conversation.store import StoreNonRetentionAudit
from ..conversation.value import (
    IntegrityDigest,
    canonical_json_bytes,
    freeze_json_value,
    validate_identifier,
)
from ..types import JsonValue
from .responses_lifecycle import (
    PUBLIC_RESPONSE_ID_PATTERN,
    ResponsesAuthorityResolver,
    ResponsesClock,
    StoredResponsesResource,
    validate_public_response_id,
)

from asyncio import Lock
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from inspect import iscoroutinefunction
from re import fullmatch
from secrets import token_hex
from typing import Protocol, final
from uuid import uuid4

from fastapi import FastAPI, Request


def _is_async_callable(value: object) -> bool:
    """Return whether a callable has an asynchronous call boundary."""
    return callable(value) and (
        iscoroutinefunction(value)
        or iscoroutinefunction(getattr(value, "__call__", None))
    )


class StatelessResponseOutcome(StrEnum):
    """Identify one transient request's terminal cleanup outcome."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCONNECTED = "disconnected"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessOperationalAuditRecord:
    """Retain bounded content-free operational accounting only."""

    authority_scope_digest: str
    operation: str
    outcome: StatelessResponseOutcome
    request_bytes: int
    response_bytes: int
    input_items: int
    output_items: int
    provider_lanes: int
    reconstructable_state_count: int

    def __post_init__(self) -> None:
        validate_identifier(
            self.authority_scope_digest,
            "authority_scope_digest",
        )
        validate_identifier(self.operation, "operation")
        if not isinstance(self.outcome, StatelessResponseOutcome):
            raise ConversationValidationError()
        for value in (
            self.request_bytes,
            self.response_bytes,
            self.input_items,
            self.output_items,
            self.provider_lanes,
            self.reconstructable_state_count,
        ):
            if type(value) is not int or value < 0:
                raise ConversationValidationError()


class StatelessResponsesAuditHook(Protocol):
    """Record content-free non-retention evidence asynchronously."""

    async def record(
        self,
        record: StatelessOperationalAuditRecord,
    ) -> None:
        """Record one bounded operational audit event."""
        ...


@final
class InMemoryStatelessResponsesAuditHook:
    """Collect bounded operational records for embedded deployments."""

    def __init__(self) -> None:
        self._records: list[StatelessOperationalAuditRecord] = []
        self._lock = Lock()

    @property
    def records(self) -> tuple[StatelessOperationalAuditRecord, ...]:
        """Return an immutable content-free record snapshot."""
        return tuple(self._records)

    async def record(
        self,
        record: StatelessOperationalAuditRecord,
    ) -> None:
        """Record one bounded operational audit event."""
        if type(record) is not StatelessOperationalAuditRecord:
            raise ConversationValidationError()
        async with self._lock:
            self._records.append(record)


@final
class _NoopStatelessResponsesAuditHook:
    async def record(
        self,
        record: StatelessOperationalAuditRecord,
    ) -> None:
        if type(record) is not StatelessOperationalAuditRecord:
            raise ConversationValidationError()


class StatelessResponsesTransientStore(Protocol):
    """Own one request-local conversation store until terminal cleanup."""

    @property
    def durable(self) -> bool:
        """Return false for request-local state."""
        ...

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> ConversationResult:
        """Retrieve one terminal result before disposal."""
        ...

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Load one terminal checkpoint before disposal."""
        ...

    async def close(self) -> StoreCloseResolution:
        """Dispose every reconstructable state surface."""
        ...

    async def audit_non_retention(self) -> StoreNonRetentionAudit:
        """Return content-free counts after disposal."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessNamedHeadReservation:
    """Reserve one exact content-free named-head advancement."""

    authority_scope_digest: str
    head_id: NamedHeadId
    expected_revision: NamedHeadRevision
    request_digest: str
    lease_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.authority_scope_digest, "authority_scope_digest"),
            (self.head_id, "head_id"),
            (self.request_digest, "request_digest"),
            (self.lease_id, "lease_id"),
        ):
            validate_identifier(value, name)
        if (
            type(self.expected_revision) is not int
            or self.expected_revision < 0
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only safe reservation coordinates."""
        return (
            "StatelessNamedHeadReservation("
            f"head_id={self.head_id!r}, "
            f"expected_revision={self.expected_revision})"
        )


class StatelessNamedHeadLedger(Protocol):
    """Fence named-head advances without retaining conversation state."""

    async def reserve(
        self,
        *,
        authority_scope_digest: str,
        head_id: NamedHeadId,
        expected_revision: NamedHeadRevision,
        request_digest: str,
    ) -> StatelessNamedHeadReservation:
        """Reserve one exact compare-and-swap operation."""
        ...

    async def commit(
        self,
        reservation: StatelessNamedHeadReservation,
        *,
        continuation_digest: str,
    ) -> NamedHeadRevision:
        """Commit one reservation to its next content-free revision."""
        ...

    async def release(
        self,
        reservation: StatelessNamedHeadReservation,
    ) -> None:
        """Release an uncommitted reservation idempotently."""
        ...


@final
class InMemoryStatelessNamedHeadLedger:
    """Fence named heads with digests and revisions only."""

    def __init__(self) -> None:
        self._heads: dict[tuple[str, NamedHeadId], tuple[int, str]] = {}
        self._reservations: dict[
            tuple[str, NamedHeadId], StatelessNamedHeadReservation
        ] = {}
        self._lock = Lock()

    async def reserve(
        self,
        *,
        authority_scope_digest: str,
        head_id: NamedHeadId,
        expected_revision: NamedHeadRevision,
        request_digest: str,
    ) -> StatelessNamedHeadReservation:
        """Reserve one exact compare-and-swap operation."""
        validate_identifier(
            authority_scope_digest,
            "authority_scope_digest",
        )
        validate_identifier(head_id, "head_id")
        validate_identifier(request_digest, "request_digest")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ConversationValidationError()
        key = (authority_scope_digest, head_id)
        async with self._lock:
            current = self._heads.get(key, (0, ""))[0]
            if current != expected_revision or key in self._reservations:
                raise ConversationConflictError()
            reservation = StatelessNamedHeadReservation(
                authority_scope_digest=authority_scope_digest,
                head_id=head_id,
                expected_revision=expected_revision,
                request_digest=request_digest,
                lease_id=f"head-lease-{token_hex(16)}",
            )
            self._reservations[key] = reservation
            return reservation

    async def commit(
        self,
        reservation: StatelessNamedHeadReservation,
        *,
        continuation_digest: str,
    ) -> NamedHeadRevision:
        """Commit one reservation to its next content-free revision."""
        if type(reservation) is not StatelessNamedHeadReservation:
            raise ConversationValidationError()
        validate_identifier(continuation_digest, "continuation_digest")
        key = (reservation.authority_scope_digest, reservation.head_id)
        async with self._lock:
            if self._reservations.get(key) != reservation:
                raise ConversationConflictError()
            current = self._heads.get(key, (0, ""))[0]
            if current != reservation.expected_revision:
                raise ConversationConflictError()
            revision = NamedHeadRevision(current + 1)
            self._heads[key] = (revision, continuation_digest)
            del self._reservations[key]
            return revision

    async def release(
        self,
        reservation: StatelessNamedHeadReservation,
    ) -> None:
        """Release an uncommitted reservation idempotently."""
        if type(reservation) is not StatelessNamedHeadReservation:
            raise ConversationValidationError()
        key = (reservation.authority_scope_digest, reservation.head_id)
        async with self._lock:
            if self._reservations.get(key) == reservation:
                del self._reservations[key]

    async def inspect(
        self,
        authority_scope_digest: str,
        head_id: NamedHeadId,
    ) -> tuple[NamedHeadRevision, str] | None:
        """Return one content-free head revision and token digest."""
        async with self._lock:
            value = self._heads.get((authority_scope_digest, head_id))
            return (
                (NamedHeadRevision(value[0]), value[1])
                if value is not None
                else None
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessResponsesTurnPlan:
    """Carry trusted state for one request-local coordinated agent turn."""

    authority: AuthorityScope
    input_text: str
    public_response_id: PublicResponseId
    provisional_response_id: ProvisionalResponseId
    idempotency_key: RequestIdempotencyKey
    request_fingerprint: str
    retention: RetentionLimits
    reasoning_context: ReasoningContext
    streaming: bool
    advance: ContinuationEnvelopeAdvance
    opened_parent: OpenedContinuationEnvelope | None = None

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        validate_identifier(
            self.input_text,
            "input_text",
            max_length=1_048_576,
        )
        validate_public_response_id(self.public_response_id)
        for value, name in (
            (self.provisional_response_id, "provisional_response_id"),
            (self.idempotency_key, "idempotency_key"),
        ):
            validate_identifier(value, name)
        if (
            fullmatch(r"[0-9a-f]{64}", self.request_fingerprint) is None
            or type(self.retention) is not RetentionLimits
            or self.retention.storage.local
            is not LocalResponseStorage.TRANSIENT
            or not isinstance(self.reasoning_context, ReasoningContext)
            or type(self.streaming) is not bool
            or type(self.advance) is not ContinuationEnvelopeAdvance
            or (
                self.opened_parent is not None
                and type(self.opened_parent) is not OpenedContinuationEnvelope
            )
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only content-free request accounting."""
        return (
            "StatelessResponsesTurnPlan("
            f"streaming={self.streaming}, "
            f"parent_present={self.opened_parent is not None})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedStatelessResponsesTurn:
    """Own one coordinated turn and its request-local store."""

    turn: AgentConversationTurn
    store: StatelessResponsesTransientStore
    children: tuple[AgentConversationChildBinding, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.turn) is not AgentConversationTurn
            or getattr(self.store, "durable", None) is not False
            or type(self.children) is not tuple
            or any(
                type(child) is not AgentConversationChildBinding
                for child in self.children
            )
        ):
            raise ConversationValidationError()
        for name in (
            "retrieve",
            "load",
            "close",
            "audit_non_retention",
        ):
            operation = getattr(self.store, name, None)
            if not callable(operation) or not iscoroutinefunction(operation):
                raise ConversationValidationError()


class StatelessResponsesTurnResolver(Protocol):
    """Build one run-scoped transient turn from trusted policy."""

    async def __call__(
        self,
        plan: StatelessResponsesTurnPlan,
    ) -> PreparedStatelessResponsesTurn:
        """Return one exact prepared transient turn."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessCompactPlan:
    """Carry complete canonical context to one provider-native compact call."""

    authority: AuthorityScope
    model: str
    lane_id: str
    input: tuple[Mapping[str, JsonValue], ...]
    instructions: str | None = None
    checkpoint: ConversationCheckpoint | None = None

    def __post_init__(self) -> None:
        if type(self.authority) is not AuthorityScope:
            raise ConversationValidationError()
        validate_identifier(self.model, "model")
        validate_identifier(self.lane_id, "lane_id")
        if self.instructions is not None and (
            type(self.instructions) is not str or not self.instructions
        ):
            raise ConversationValidationError()
        if type(self.input) is not tuple or not self.input:
            raise ConversationValidationError()
        frozen = tuple(freeze_json_value(item) for item in self.input)
        if any(not isinstance(item, Mapping) for item in frozen):
            raise ConversationValidationError()
        object.__setattr__(self, "input", frozen)
        if self.checkpoint is not None and (
            type(self.checkpoint) is not ConversationCheckpoint
            or self.checkpoint.authority != self.authority
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only safe compact input accounting."""
        return (
            "StatelessCompactPlan("
            f"item_count={len(self.input)}, "
            f"checkpoint_present={self.checkpoint is not None})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessCompactResult:
    """Return provider-canonical compact context without tool execution."""

    id: str
    created_at: int
    output: tuple[Mapping[str, JsonValue], ...]
    usage: ProviderUsage
    checkpoint: ConversationCheckpoint | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.id, "id")
        if type(self.created_at) is not int or self.created_at < 0:
            raise ConversationValidationError()
        if type(self.output) is not tuple or not self.output:
            raise ConversationValidationError()
        frozen = tuple(freeze_json_value(item) for item in self.output)
        if any(not isinstance(item, Mapping) for item in frozen):
            raise ConversationValidationError()
        object.__setattr__(self, "output", frozen)
        if type(self.usage) is not ProviderUsage:
            raise ConversationValidationError()
        if self.checkpoint is not None and (
            type(self.checkpoint) is not ConversationCheckpoint
            or self.checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
        ):
            raise ConversationValidationError()

    def __repr__(self) -> str:
        """Return only safe compact output accounting."""
        return f"StatelessCompactResult(item_count={len(self.output)})"


class StatelessCompactResolver(Protocol):
    """Dispatch one provider-native tool-free compact operation."""

    async def __call__(
        self,
        plan: StatelessCompactPlan,
    ) -> StatelessCompactResult:
        """Return provider-canonical compact output unchanged."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessResponsesPolicy:
    """Freeze served stateless capabilities and authority bindings."""

    agent_id: str
    endpoint_id: str
    deployment_id: str
    retention: RetentionLimits
    compact_lane_id: str
    public_model: str = "default"
    allowed_reasoning_contexts: frozenset[ReasoningContext] = frozenset(
        ReasoningContext
    )
    max_canonical_items: int = 100_000
    max_canonical_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        for value, name in (
            (self.agent_id, "agent_id"),
            (self.endpoint_id, "endpoint_id"),
            (self.deployment_id, "deployment_id"),
            (self.compact_lane_id, "compact_lane_id"),
            (self.public_model, "public_model"),
        ):
            validate_identifier(value, name)
        if (
            type(self.retention) is not RetentionLimits
            or self.retention.storage.local
            is not LocalResponseStorage.TRANSIENT
            or type(self.allowed_reasoning_contexts) is not frozenset
            or not self.allowed_reasoning_contexts
            or any(
                not isinstance(value, ReasoningContext)
                for value in self.allowed_reasoning_contexts
            )
        ):
            raise ConversationValidationError()
        for bound in (self.max_canonical_items, self.max_canonical_bytes):
            if type(bound) is not int or bound <= 0:
                raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessResponsesConfiguration:
    """Configure authenticated served stateless response operations."""

    authority_resolver: ResponsesAuthorityResolver
    turn_resolver: StatelessResponsesTurnResolver
    envelope_codec: ContinuationEnvelopeCodec
    policy: StatelessResponsesPolicy
    compact_resolver: StatelessCompactResolver | None = None
    named_head_ledger: StatelessNamedHeadLedger = field(
        default_factory=InMemoryStatelessNamedHeadLedger
    )
    audit_hook: StatelessResponsesAuditHook = (
        _NoopStatelessResponsesAuditHook()
    )
    clock: ResponsesClock | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.authority_resolver, "authority_resolver"),
            (self.turn_resolver, "turn_resolver"),
        ):
            if not _is_async_callable(value):
                raise TypeError(f"{name} must be asynchronous")
        if (
            type(self.envelope_codec) is not ContinuationEnvelopeCodec
            or type(self.policy) is not StatelessResponsesPolicy
        ):
            raise TypeError("invalid stateless Responses configuration")
        if self.compact_resolver is not None and (
            not _is_async_callable(self.compact_resolver)
        ):
            raise TypeError("compact_resolver must be asynchronous")
        for component, method_name, name in (
            (self.named_head_ledger, "reserve", "named_head_ledger"),
            (self.named_head_ledger, "commit", "named_head_ledger"),
            (self.named_head_ledger, "release", "named_head_ledger"),
            (self.audit_hook, "record", "audit_hook"),
        ):
            method = getattr(component, method_name, None)
            if not callable(method) or not iscoroutinefunction(method):
                raise TypeError(f"{name} must be asynchronous")
        if self.clock is not None:
            method = getattr(self.clock, "now", None)
            if not callable(method) or not iscoroutinefunction(method):
                raise TypeError("clock must be asynchronous")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedStatelessResponse:
    """Carry one transient turn plus optional named-head reservation."""

    plan: StatelessResponsesTurnPlan
    prepared: PreparedStatelessResponsesTurn
    reservation: StatelessNamedHeadReservation | None = None


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessResponseCommit:
    """Return a public body and terminal-only caller-held state."""

    body: Mapping[str, object]
    continuation: ContinuationEnvelopeToken
    audit: StoreNonRetentionAudit

    def __repr__(self) -> str:
        """Return only safe terminal accounting."""
        return (
            "StatelessResponseCommit("
            "reconstructable_state_count="
            f"{self.audit.reconstructable_state_count})"
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StatelessCompactCommit:
    """Return canonical compact output and optional continuation state."""

    result: StatelessCompactResult
    continuation: ContinuationEnvelopeToken | None = None

    def response_body(self) -> dict[str, object]:
        """Return the official-client-compatible compact resource."""
        body: dict[str, object] = {
            "id": self.result.id,
            "created_at": self.result.created_at,
            "object": "response.compaction",
            "output": [dict(item) for item in self.result.output],
            "usage": {
                "input_tokens": self.result.usage.input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": self.result.usage.output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": (
                    self.result.usage.input_tokens
                    + self.result.usage.output_tokens
                ),
            },
        }
        if self.continuation is not None:
            body["extensions"] = continuation_response_extension(
                self.continuation
            )
        return body

    def __repr__(self) -> str:
        """Return only safe compact terminal accounting."""
        return (
            "StatelessCompactCommit("
            f"continuation_present={self.continuation is not None})"
        )


@final
class StatelessResponsesService:
    """Authorize, seal, compact, and dispose served stateless state."""

    def __init__(self, configuration: StatelessResponsesConfiguration) -> None:
        if type(configuration) is not StatelessResponsesConfiguration:
            raise TypeError("invalid stateless Responses configuration")
        self.configuration = configuration

    async def authenticate(self, request: Request) -> AuthorityScope:
        """Resolve strict authenticated served authority."""
        try:
            authority = await self.configuration.authority_resolver(request)
        except Exception:
            authority = None
        policy = self.configuration.policy
        if (
            type(authority) is not AuthorityScope
            or authority.source
            is not AuthoritySource.AUTHENTICATED_SERVER_CONTEXT
            or authority.tenant_id is None
            or not authority.network_exposed
            or str(authority.agent_id) != policy.agent_id
            or str(authority.endpoint_id) != policy.endpoint_id
        ):
            raise ConversationAuthorizationError()
        return authority

    async def now(self) -> datetime:
        """Return one validated aware server time."""
        clock = self.configuration.clock
        value = datetime.now(UTC) if clock is None else await clock.now()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationStorageError()
        return value.astimezone(UTC)

    async def prepare_turn(
        self,
        *,
        authority: AuthorityScope,
        input_text: str,
        request_fingerprint: str,
        reasoning_context: ReasoningContext,
        streaming: bool,
        idempotency_key: str | None,
        continuation_value: ContinuationEnvelopeToken | None,
        advance: ContinuationEnvelopeAdvance,
    ) -> PreparedStatelessResponse:
        """Open optional caller state and prepare one request-local turn."""
        policy = self.configuration.policy
        if reasoning_context not in policy.allowed_reasoning_contexts:
            raise ConversationCapabilityError()
        opened = None
        now = await self.now()
        if continuation_value is not None:
            if type(continuation_value) is not ContinuationEnvelopeToken:
                raise ConversationValidationError()
            opened = await self._open_envelope(
                continuation_value,
                authority=authority,
                advance=advance,
                now=now,
            )
        elif advance.mode is not ParentAdvanceMode.ORDINARY_CHILD:
            if not (
                advance.mode is ParentAdvanceMode.NAMED_HEAD
                and advance.expected_head_revision == 0
            ):
                raise ConversationValidationError()
        suffix = self._response_suffix(
            authority,
            request_fingerprint,
            idempotency_key=idempotency_key,
        )
        plan = StatelessResponsesTurnPlan(
            authority=authority,
            input_text=input_text,
            public_response_id=PublicResponseId(f"resp_avl_{suffix}"),
            provisional_response_id=ProvisionalResponseId(
                f"provisional-avl-{suffix}"
            ),
            idempotency_key=RequestIdempotencyKey(
                idempotency_key or f"stateless-{suffix}"
            ),
            request_fingerprint=request_fingerprint,
            retention=policy.retention,
            reasoning_context=reasoning_context,
            streaming=streaming,
            advance=advance,
            opened_parent=opened,
        )
        reservation = await self._reserve_head(plan)
        prepared: PreparedStatelessResponsesTurn | None = None
        try:
            prepared = await self.configuration.turn_resolver(plan)
            self._validate_prepared(plan, prepared)
        except BaseException:
            if type(prepared) is PreparedStatelessResponsesTurn:
                await self._close_failed(
                    PreparedStatelessResponse(
                        plan=plan,
                        prepared=prepared,
                        reservation=reservation,
                    )
                )
            elif reservation is not None:
                await self.configuration.named_head_ledger.release(reservation)
            raise
        assert prepared is not None
        return PreparedStatelessResponse(
            plan=plan,
            prepared=prepared,
            reservation=reservation,
        )

    async def finalize(
        self,
        prepared: PreparedStatelessResponse,
        *,
        request_bytes: int,
        response_bytes: int,
        input_items: int,
        output_items: int,
    ) -> StatelessResponseCommit:
        """Seal terminal state, dispose plaintext, and return caller state."""
        if type(prepared) is not PreparedStatelessResponse:
            raise ConversationValidationError()
        plan = prepared.plan
        store = prepared.prepared.store
        token: ContinuationEnvelopeToken | None = None
        body: Mapping[str, object] | None = None
        audit: StoreNonRetentionAudit | None = None
        try:
            result = await store.retrieve(
                plan.public_response_id,
                plan.authority,
            )
            checkpoint = await store.load(
                result.handle.checkpoint_id,
                plan.authority,
            )
            if (
                checkpoint.identity.checkpoint_id
                != prepared.prepared.turn.checkpoint_id
            ):
                raise ConversationStorageError()
            resource = StoredResponsesResource(
                result=result,
                checkpoint=checkpoint,
                public_model=self.configuration.policy.public_model,
            )
            next_head_revision = (
                NamedHeadRevision(prepared.reservation.expected_revision + 1)
                if prepared.reservation is not None
                else None
            )
            token = await self.configuration.envelope_codec.seal(
                checkpoint,
                authority=ContinuationEnvelopeAuthority(
                    authority=plan.authority,
                    deployment_id=self.configuration.policy.deployment_id,
                ),
                public_parent=plan.public_response_id,
                issued_at=await self.now(),
                head_id=(
                    prepared.reservation.head_id
                    if prepared.reservation is not None
                    else None
                ),
                head_revision=next_head_revision,
            )
            close = await store.close()
            if close.disposition is not StoreCloseDisposition.CLOSED:
                raise ConversationStorageError()
            audit = await store.audit_non_retention()
            if audit.reconstructable_state_count != 0:
                raise ConversationStorageError()
            if prepared.reservation is not None:
                revision = await self.configuration.named_head_ledger.commit(
                    prepared.reservation,
                    continuation_digest=token.digest,
                )
                if revision != next_head_revision:
                    raise ConversationConflictError()
            body = resource.response_body()
            mutable_body = dict(body)
            mutable_body["extensions"] = continuation_response_extension(token)
            body = mutable_body
        except BaseException:
            await self._close_failed(prepared)
            raise
        assert token is not None and body is not None and audit is not None
        await self._record_audit(
            plan.authority,
            operation="response_create",
            outcome=StatelessResponseOutcome.COMPLETED,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            input_items=input_items,
            output_items=output_items,
            provider_lanes=len(prepared.prepared.turn.lanes),
            retention=audit,
        )
        return StatelessResponseCommit(
            body=body,
            continuation=token,
            audit=audit,
        )

    async def abort(
        self,
        prepared: PreparedStatelessResponse,
        *,
        outcome: StatelessResponseOutcome,
        request_bytes: int,
        input_items: int,
    ) -> None:
        """Dispose uncommitted state without producing a continuation."""
        if (
            type(prepared) is not PreparedStatelessResponse
            or outcome is StatelessResponseOutcome.COMPLETED
        ):
            raise ConversationValidationError()
        audit = await self._close_failed(prepared)
        await self._record_audit(
            prepared.plan.authority,
            operation="response_create",
            outcome=outcome,
            request_bytes=request_bytes,
            response_bytes=0,
            input_items=input_items,
            output_items=0,
            provider_lanes=len(prepared.prepared.turn.lanes),
            retention=audit,
        )

    async def compact(
        self,
        *,
        authority: AuthorityScope,
        model: str,
        instructions: str | None,
        canonical_input: tuple[Mapping[str, JsonValue], ...],
        continuation_value: ContinuationEnvelopeToken | None,
        lane_id: str | None,
    ) -> StatelessCompactCommit:
        """Run one stateless provider-native compact operation."""
        resolver = self.configuration.compact_resolver
        if resolver is None:
            raise ConversationCapabilityError()
        policy = self.configuration.policy
        if model != policy.public_model:
            raise ConversationCapabilityError()
        selected_lane = lane_id or policy.compact_lane_id
        if selected_lane != policy.compact_lane_id:
            raise ConversationCapabilityError()
        checkpoint = None
        if continuation_value is not None:
            if type(continuation_value) is not ContinuationEnvelopeToken:
                raise ConversationValidationError()
            opened = await self._open_envelope(
                continuation_value,
                authority=authority,
                advance=ContinuationEnvelopeAdvance(
                    mode=ParentAdvanceMode.ORDINARY_CHILD
                ),
                now=await self.now(),
            )
            checkpoint = opened.checkpoint
            lanes = tuple(
                item
                for item in checkpoint.content.lanes
                if str(item.lane_id) == selected_lane
            )
            if len(lanes) != 1:
                raise ConversationAuthorizationError()
            lane = lanes[0]
            if not canonical_input:
                if not isinstance(lane, StatelessProviderLaneSnapshot):
                    raise ConversationCapabilityError()
                canonical_input = tuple(
                    item.canonical_input for item in lane.ledger.items
                )
        if not canonical_input:
            raise ConversationValidationError()
        encoded = canonical_json_bytes(
            freeze_json_value(list(canonical_input))
        )
        if (
            len(canonical_input) > policy.max_canonical_items
            or len(encoded) > policy.max_canonical_bytes
        ):
            raise ConversationValidationError()
        result = await resolver(
            StatelessCompactPlan(
                authority=authority,
                model=model,
                lane_id=selected_lane,
                input=canonical_input,
                instructions=instructions,
                checkpoint=checkpoint,
            )
        )
        if type(result) is not StatelessCompactResult:
            raise ConversationStorageError()
        continuation = None
        if result.checkpoint is not None:
            if result.checkpoint.authority != authority:
                raise ConversationAuthorizationError()
            continuation = await self.configuration.envelope_codec.seal(
                result.checkpoint,
                authority=ContinuationEnvelopeAuthority(
                    authority=authority,
                    deployment_id=policy.deployment_id,
                ),
                public_parent=PublicResponseId(result.id),
                issued_at=await self.now(),
            )
        await self._record_audit(
            authority,
            operation="response_compact",
            outcome=StatelessResponseOutcome.COMPLETED,
            request_bytes=len(encoded),
            response_bytes=len(
                canonical_json_bytes(freeze_json_value(list(result.output)))
            ),
            input_items=len(canonical_input),
            output_items=len(result.output),
            provider_lanes=1,
            retention=StoreNonRetentionAudit(
                checkpoints=0,
                provider_ledgers=0,
                public_mappings=0,
                provisional_mappings=0,
                idempotency_records=0,
                named_heads=0,
                queues=0,
                outbox_records=0,
                task_state=0,
                envelope_plaintexts=0,
                temporary_files=0,
            ),
        )
        return StatelessCompactCommit(
            result=result,
            continuation=continuation,
        )

    async def record_standard_terminal(
        self,
        authority: AuthorityScope,
        *,
        outcome: StatelessResponseOutcome,
        request_bytes: int,
        response_bytes: int,
        input_items: int,
        output_items: int,
    ) -> None:
        """Audit canonical caller replay without retained hidden state."""
        await self._record_audit(
            authority,
            operation="canonical_replay",
            outcome=outcome,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            input_items=input_items,
            output_items=output_items,
            provider_lanes=1,
            retention=StoreNonRetentionAudit(
                checkpoints=0,
                provider_ledgers=0,
                public_mappings=0,
                provisional_mappings=0,
                idempotency_records=0,
                named_heads=0,
                queues=0,
                outbox_records=0,
                task_state=0,
                envelope_plaintexts=0,
                temporary_files=0,
            ),
        )

    async def _reserve_head(
        self,
        plan: StatelessResponsesTurnPlan,
    ) -> StatelessNamedHeadReservation | None:
        advance = plan.advance
        if advance.mode is not ParentAdvanceMode.NAMED_HEAD:
            return None
        assert advance.head_id is not None
        assert advance.expected_head_revision is not None
        return await self.configuration.named_head_ledger.reserve(
            authority_scope_digest=str(authority_digest(plan.authority)),
            head_id=advance.head_id,
            expected_revision=advance.expected_head_revision,
            request_digest=plan.request_fingerprint,
        )

    async def _open_envelope(
        self,
        token: ContinuationEnvelopeToken,
        *,
        authority: AuthorityScope,
        advance: ContinuationEnvelopeAdvance,
        now: datetime,
    ) -> OpenedContinuationEnvelope:
        try:
            return await self.configuration.envelope_codec.open(
                token,
                authority=ContinuationEnvelopeAuthority(
                    authority=authority,
                    deployment_id=(self.configuration.policy.deployment_id),
                ),
                advance=advance,
                now=now,
            )
        except (
            ConversationAuthorizationError,
            ConversationCryptoAuthenticationError,
            ConversationKeyMissingError,
        ):
            raise ConversationAuthorizationError() from None
        except ConversationCodecError:
            raise ConversationValidationError() from None

    async def _close_failed(
        self,
        prepared: PreparedStatelessResponse,
    ) -> StoreNonRetentionAudit:
        failures: list[BaseException] = []
        if prepared.reservation is not None:
            try:
                await self.configuration.named_head_ledger.release(
                    prepared.reservation
                )
            except BaseException as error:
                failures.append(error)
        store = prepared.prepared.store
        try:
            close = await store.close()
            if close.disposition is not StoreCloseDisposition.CLOSED:
                failures.append(ConversationStorageError())
        except BaseException as error:
            failures.append(error)
        try:
            audit = await store.audit_non_retention()
        except BaseException as error:
            failures.append(error)
            raise failures[0]
        if audit.reconstructable_state_count != 0:
            raise ConversationStorageError()
        if failures:
            raise failures[0]
        return audit

    async def _record_audit(
        self,
        authority: AuthorityScope,
        *,
        operation: str,
        outcome: StatelessResponseOutcome,
        request_bytes: int,
        response_bytes: int,
        input_items: int,
        output_items: int,
        provider_lanes: int,
        retention: StoreNonRetentionAudit,
    ) -> None:
        await self.configuration.audit_hook.record(
            StatelessOperationalAuditRecord(
                authority_scope_digest=str(authority_digest(authority)),
                operation=operation,
                outcome=outcome,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                input_items=input_items,
                output_items=output_items,
                provider_lanes=provider_lanes,
                reconstructable_state_count=(
                    retention.reconstructable_state_count
                ),
            )
        )

    @staticmethod
    def _response_suffix(
        authority: AuthorityScope,
        request_fingerprint: str,
        *,
        idempotency_key: str | None,
    ) -> str:
        if fullmatch(r"[0-9a-f]{64}", request_fingerprint) is None:
            raise ConversationValidationError()
        if idempotency_key is None:
            return uuid4().hex
        validate_identifier(idempotency_key, "idempotency_key")
        return sha256(
            canonical_json_bytes(
                {
                    "authority": str(authority_digest(authority)),
                    "idempotency_key": idempotency_key,
                    "request_fingerprint": request_fingerprint,
                }
            )
        ).hexdigest()[:32]

    @staticmethod
    def _validate_prepared(
        plan: StatelessResponsesTurnPlan,
        prepared: PreparedStatelessResponsesTurn,
    ) -> None:
        if type(prepared) is not PreparedStatelessResponsesTurn:
            raise ConversationValidationError()
        turn = prepared.turn
        opened = plan.opened_parent
        expected_advance: ExplicitBranchAdvance | NamedHeadAdvance | None = (
            None
        )
        if opened is not None:
            if plan.advance.mode is ParentAdvanceMode.EXPLICIT_BRANCH:
                assert plan.advance.branch_id is not None
                expected_advance = ExplicitBranchAdvance(
                    parent_checkpoint_id=(
                        opened.checkpoint.identity.checkpoint_id
                    ),
                    branch_id=plan.advance.branch_id,
                )
            elif plan.advance.mode is ParentAdvanceMode.NAMED_HEAD:
                assert plan.advance.head_id is not None
                assert plan.advance.expected_head_revision is not None
                expected_advance = NamedHeadAdvance(
                    head_id=plan.advance.head_id,
                    parent_checkpoint_id=(
                        opened.checkpoint.identity.checkpoint_id
                    ),
                    expected_revision=plan.advance.expected_head_revision,
                )
        if (
            turn.authority != plan.authority
            or turn.public_response_id != plan.public_response_id
            or turn.provisional_response_id != plan.provisional_response_id
            or turn.idempotency_key != plan.idempotency_key
            or turn.retention != plan.retention
            or turn.parent
            != (opened.checkpoint if opened is not None else None)
            or turn.advance != expected_advance
            or turn.branch_id
            != (
                opened.target_branch_id
                if opened is not None
                else turn.branch_id
            )
        ):
            raise ConversationValidationError()


def continuation_response_extension(
    token: ContinuationEnvelopeToken,
) -> dict[str, object]:
    """Serialize caller-held state only into its exact response field."""
    if type(token) is not ContinuationEnvelopeToken:
        raise ConversationValidationError()
    return {
        "avalan": {
            "version": "1",
            "conversation": {
                "version": "1",
                "continuation_envelope": token.value_for_response(),
            },
        }
    }


def configure_stateless_responses(
    app: FastAPI,
    configuration: StatelessResponsesConfiguration | None,
) -> None:
    """Install or remove the served stateless service."""
    if configuration is None:
        if hasattr(app.state, "stateless_responses_service"):
            delattr(app.state, "stateless_responses_service")
        return
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )


def canonical_compact_digest(
    result: StatelessCompactResult,
) -> IntegrityDigest:
    """Digest exact provider-canonical compact output."""
    if type(result) is not StatelessCompactResult:
        raise ConversationValidationError()
    return IntegrityDigest(
        sha256(
            canonical_json_bytes(freeze_json_value(list(result.output)))
        ).hexdigest()
    )


def is_public_stateless_response_id(value: str) -> bool:
    """Return whether a value is a syntactically valid Avalan public ID."""
    return (
        type(value) is str
        and fullmatch(PUBLIC_RESPONSE_ID_PATTERN, value) is not None
    )
