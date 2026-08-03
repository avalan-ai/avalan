"""Coordinate authenticated durable Responses lifecycle operations."""

from ..agent.conversation_child import AgentConversationChildBinding
from ..conversation.agent import AgentConversationTurn
from ..conversation.contract import (
    AuthorityScope,
    AuthoritySource,
    CheckpointId,
    CheckpointKind,
    LocalDeletionState,
    LocalResponseStorage,
    ProviderLaneOwnerKind,
    ProvisionalResponseId,
    PublicResponseId,
    RequestIdempotencyKey,
    RetentionLimits,
)
from ..conversation.errors import (
    ConversationAuthorizationError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationConflictError,
    ConversationStorageError,
    ConversationValidationError,
)
from ..conversation.lifecycle import LocalDeletionPreparation
from ..conversation.observability import authority_digest
from ..conversation.runtime import StoreCloseResolution, SweepReceipt
from ..conversation.settings import ConversationResult, ReasoningContext
from ..conversation.state import CheckpointLifecycle, ConversationCheckpoint
from ..conversation.value import canonical_json_bytes, validate_identifier

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from inspect import iscoroutinefunction
from re import fullmatch
from types import MappingProxyType
from typing import Protocol, final
from uuid import uuid4

from fastapi import FastAPI, Request

PUBLIC_RESPONSE_ID_PATTERN = r"resp_avl_[0-9a-f]{32}"


class ServedResponseLifecycle(StrEnum):
    """Identify one outward response publication state."""

    STAGED = "staged"
    COMPLETED_PROVIDER = "completed_provider"
    CHECKPOINT_COMMITTED = "checkpoint_committed"
    PUBLISHED = "published"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"
    EXPIRED = "expired"
    RECONCILIATION_REQUIRED = "reconciliation_required"


SERVED_RESPONSE_TRANSITIONS = MappingProxyType(
    {
        ServedResponseLifecycle.STAGED: frozenset(
            {
                ServedResponseLifecycle.COMPLETED_PROVIDER,
                ServedResponseLifecycle.FAILED,
            }
        ),
        ServedResponseLifecycle.COMPLETED_PROVIDER: frozenset(
            {
                ServedResponseLifecycle.CHECKPOINT_COMMITTED,
                ServedResponseLifecycle.FAILED,
                ServedResponseLifecycle.RECONCILIATION_REQUIRED,
            }
        ),
        ServedResponseLifecycle.CHECKPOINT_COMMITTED: frozenset(
            {
                ServedResponseLifecycle.PUBLISHED,
                ServedResponseLifecycle.RECONCILIATION_REQUIRED,
            }
        ),
        ServedResponseLifecycle.PUBLISHED: frozenset(
            {
                ServedResponseLifecycle.TOMBSTONED,
                ServedResponseLifecycle.EXPIRED,
            }
        ),
        ServedResponseLifecycle.FAILED: frozenset(
            {
                ServedResponseLifecycle.TOMBSTONED,
                ServedResponseLifecycle.EXPIRED,
            }
        ),
        ServedResponseLifecycle.TOMBSTONED: frozenset(),
        ServedResponseLifecycle.EXPIRED: frozenset(),
        ServedResponseLifecycle.RECONCILIATION_REQUIRED: frozenset(
            {
                ServedResponseLifecycle.PUBLISHED,
                ServedResponseLifecycle.TOMBSTONED,
                ServedResponseLifecycle.EXPIRED,
            }
        ),
    }
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ServedResponseLifecycleRecord:
    """Track one content-free response publication transition."""

    public_response_id: PublicResponseId
    state: ServedResponseLifecycle = ServedResponseLifecycle.STAGED

    def __post_init__(self) -> None:
        validate_public_response_id(self.public_response_id)
        if not isinstance(self.state, ServedResponseLifecycle):
            raise ConversationValidationError()

    def transition(
        self,
        target: ServedResponseLifecycle,
    ) -> "ServedResponseLifecycleRecord":
        """Return the next legal lifecycle state."""
        if target not in SERVED_RESPONSE_TRANSITIONS[self.state]:
            raise ConversationValidationError()
        return ServedResponseLifecycleRecord(
            public_response_id=self.public_response_id,
            state=target,
        )


class ResponsesAuthorityResolver(Protocol):
    """Resolve authenticated server authority from one HTTP request."""

    async def __call__(self, request: Request) -> AuthorityScope | None:
        """Return trusted authority or no authenticated principal."""
        ...


class ResponsesClock(Protocol):
    """Read one aware server time asynchronously."""

    async def now(self) -> datetime:
        """Return the current aware instant."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ServedResponsesTurnPlan:
    """Carry trusted state needed to build one coordinated agent turn."""

    authority: AuthorityScope
    input_text: str
    public_response_id: PublicResponseId
    provisional_response_id: ProvisionalResponseId
    idempotency_key: RequestIdempotencyKey
    request_fingerprint: str
    retention: RetentionLimits
    reasoning_context: ReasoningContext
    compact_threshold: int | None
    includes: tuple[str, ...]
    tool_names: tuple[str, ...]
    parent: ConversationCheckpoint | None
    streaming: bool

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
        if fullmatch(r"[0-9a-f]{64}", self.request_fingerprint) is None:
            raise ConversationValidationError()
        if type(self.retention) is not RetentionLimits:
            raise ConversationValidationError()
        if self.retention.storage.local is not LocalResponseStorage.DURABLE:
            raise ConversationValidationError()
        if not isinstance(self.reasoning_context, ReasoningContext):
            raise ConversationValidationError()
        if self.compact_threshold is not None and (
            type(self.compact_threshold) is not int
            or self.compact_threshold <= 0
        ):
            raise ConversationValidationError()
        for values in (self.includes, self.tool_names):
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise ConversationValidationError()
        if self.parent is not None and (
            type(self.parent) is not ConversationCheckpoint
            or self.parent.lifecycle is not CheckpointLifecycle.COMMITTED
            or self.parent.kind is not CheckpointKind.COMPLETED_OUTWARD_TURN
            or self.parent.authority != self.authority
        ):
            raise ConversationValidationError()
        if type(self.streaming) is not bool:
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedServedResponsesTurn:
    """Return one validated turn and its configured child bindings."""

    turn: AgentConversationTurn
    children: tuple[AgentConversationChildBinding, ...] = ()

    def __post_init__(self) -> None:
        if type(self.turn) is not AgentConversationTurn or (
            type(self.children) is not tuple
            or any(
                type(child) is not AgentConversationChildBinding
                for child in self.children
            )
        ):
            raise ConversationValidationError()


class ServedResponsesTurnResolver(Protocol):
    """Build a run-scoped coordinated turn from trusted server policy."""

    async def __call__(
        self,
        plan: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        """Return one exact prepared turn."""
        ...


class ServedResponsesDurableStore(Protocol):
    """Expose the durable store operations used by the served boundary."""

    @property
    def durable(self) -> bool:
        """Return whether state survives process restart."""
        ...

    async def retrieve(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> ConversationResult:
        """Retrieve one authorized committed result."""
        ...

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Load one authorized committed checkpoint."""
        ...

    async def sweep(self, now: datetime, *, limit: int) -> SweepReceipt:
        """Apply bounded expiry policy."""
        ...

    async def prepare_deletion(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
    ) -> LocalDeletionPreparation:
        """Prepare one local-first deletion."""
        ...

    async def tombstone(
        self,
        public_response_id: PublicResponseId,
        authority: AuthorityScope,
        at: datetime,
    ) -> ConversationCheckpoint:
        """Commit one immediate local tombstone."""
        ...

    async def close(self) -> StoreCloseResolution:
        """Close every owned storage resource."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ServedResponsesPolicy:
    """Freeze server-owned Responses continuation capabilities."""

    agent_id: str
    endpoint_id: str
    retention: RetentionLimits
    public_model: str = "default"
    allowed_reasoning_contexts: frozenset[ReasoningContext] = frozenset(
        ReasoningContext
    )
    allowed_includes: frozenset[str] = frozenset()
    allowed_tool_names: frozenset[str] = frozenset()
    min_compact_threshold: int | None = None
    max_compact_threshold: int | None = None
    sweep_limit: int = 100

    def __post_init__(self) -> None:
        validate_identifier(self.agent_id, "agent_id")
        validate_identifier(self.endpoint_id, "endpoint_id")
        validate_identifier(self.public_model, "public_model")
        if type(self.retention) is not RetentionLimits:
            raise ConversationValidationError()
        if self.retention.storage.local is not LocalResponseStorage.DURABLE:
            raise ConversationValidationError()
        if (
            type(self.allowed_reasoning_contexts) is not frozenset
            or not self.allowed_reasoning_contexts
            or any(
                not isinstance(value, ReasoningContext)
                for value in self.allowed_reasoning_contexts
            )
            or type(self.allowed_includes) is not frozenset
            or any(
                type(value) is not str or not value
                for value in self.allowed_includes
            )
            or type(self.allowed_tool_names) is not frozenset
            or any(
                type(value) is not str or not value
                for value in self.allowed_tool_names
            )
        ):
            raise ConversationValidationError()
        bounds = (self.min_compact_threshold, self.max_compact_threshold)
        if (
            any(
                value is not None and (type(value) is not int or value <= 0)
                for value in bounds
            )
            or (
                self.min_compact_threshold is None
                and self.max_compact_threshold is not None
            )
            or (
                self.min_compact_threshold is not None
                and self.max_compact_threshold is None
            )
            or (
                self.min_compact_threshold is not None
                and self.max_compact_threshold is not None
                and self.min_compact_threshold > self.max_compact_threshold
            )
        ):
            raise ConversationValidationError()
        if type(self.sweep_limit) is not int or self.sweep_limit <= 0:
            raise ConversationValidationError()

    def validate_capabilities(
        self,
        *,
        reasoning_context: ReasoningContext,
        includes: tuple[str, ...],
        tool_names: tuple[str, ...],
        compact_threshold: int | None,
    ) -> None:
        """Reject any request that broadens server-owned capability."""
        if (
            reasoning_context not in self.allowed_reasoning_contexts
            or not set(includes) <= self.allowed_includes
            or not set(tool_names) <= self.allowed_tool_names
        ):
            raise ConversationCapabilityError()
        if compact_threshold is None:
            return
        if (
            self.min_compact_threshold is None
            or self.max_compact_threshold is None
            or not self.min_compact_threshold
            <= compact_threshold
            <= self.max_compact_threshold
        ):
            raise ConversationCapabilityError()


@final
class _UtcResponsesClock:
    async def now(self) -> datetime:
        return datetime.now(UTC)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ServedResponsesConfiguration:
    """Configure authenticated durable Responses service integration."""

    store: ServedResponsesDurableStore
    authority_resolver: ResponsesAuthorityResolver
    turn_resolver: ServedResponsesTurnResolver
    policy: ServedResponsesPolicy
    clock: ResponsesClock = _UtcResponsesClock()
    close_store_on_shutdown: bool = False

    def __post_init__(self) -> None:
        if getattr(self.store, "durable", False) is not True:
            raise TypeError("store must be durable")
        for method_name in (
            "retrieve",
            "load",
            "sweep",
            "prepare_deletion",
            "tombstone",
            "close",
        ):
            _require_async_method(self.store, method_name, "store")
        if type(self.policy) is not ServedResponsesPolicy:
            raise TypeError("policy must be a served Responses policy")
        _require_async_callable(self.authority_resolver, "authority_resolver")
        _require_async_callable(self.turn_resolver, "turn_resolver")
        _require_async_method(self.clock, "now", "clock")
        if type(self.close_store_on_shutdown) is not bool:
            raise TypeError("close_store_on_shutdown must be a boolean")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredResponsesResource:
    """Bind one safe public result to its authorized checkpoint."""

    result: ConversationResult
    checkpoint: ConversationCheckpoint
    public_model: str

    def __post_init__(self) -> None:
        if (
            type(self.result) is not ConversationResult
            or type(self.checkpoint) is not ConversationCheckpoint
            or self.result.public_response_id is None
            or self.result.handle.checkpoint_id
            != self.checkpoint.identity.checkpoint_id
            or self.checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
        ):
            raise ConversationValidationError()
        validate_identifier(self.public_model, "public_model")

    def response_body(self) -> dict[str, object]:
        """Return a public projection without private provider state."""
        public_id = self.result.public_response_id
        assert public_id is not None
        topology = self.checkpoint.content.lane_topology
        parent_entries = (
            tuple(
                entry
                for entry in topology.entries
                if entry.owner_kind is ProviderLaneOwnerKind.PARENT_AGENT
            )
            if topology is not None
            else ()
        )
        if len(parent_entries) != 1:
            raise ConversationStorageError()
        parent_outputs = tuple(
            lane
            for lane in self.result.lane_outputs
            if lane.lane_id == parent_entries[0].lane_id
        )
        if len(parent_outputs) != 1:
            raise ConversationStorageError()
        parent_output = parent_outputs[0]
        output: list[dict[str, object]] = []
        for item_index, item in enumerate(parent_output.items):
            item_id = _public_item_id(
                public_id,
                item_index,
                item.content,
            )
            output.append(
                {
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": item.role.value,
                    "content": [
                        {
                            "type": "output_text",
                            "text": item.content,
                            "annotations": [],
                        }
                    ],
                }
            )
        input_tokens = parent_output.usage.input_tokens
        output_tokens = parent_output.usage.output_tokens
        created = int(self.checkpoint.timestamps.created_at.timestamp())
        return {
            "id": str(public_id),
            "object": "response",
            "type": "response",
            "created_at": created,
            "created": created,
            "model": self.public_model,
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": output,
            "metadata": {
                "avalan_lifecycle": ServedResponseLifecycle.PUBLISHED.value,
                "avalan_checkpoint_digest": str(self.result.checkpoint_digest),
            },
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + output_tokens,
                "input_text_tokens": input_tokens,
                "output_text_tokens": output_tokens,
            },
        }


@final
class ServedResponsesService:
    """Resolve, authorize, and publish durable Responses state."""

    def __init__(self, configuration: ServedResponsesConfiguration) -> None:
        if type(configuration) is not ServedResponsesConfiguration:
            raise TypeError(
                "configuration must be a served Responses configuration"
            )
        self.configuration = configuration

    async def authenticate(self, request: Request) -> AuthorityScope:
        """Resolve strict authenticated tenant and principal authority."""
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
        value = await self.configuration.clock.now()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ConversationStorageError()
        return value

    async def sweep(self) -> None:
        """Apply configured retention before each public state operation."""
        await self.configuration.store.sweep(
            await self.now(),
            limit=self.configuration.policy.sweep_limit,
        )

    async def resolve_parent(
        self,
        public_response_id: str,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Resolve one authorized committed public parent checkpoint."""
        validate_public_response_id(public_response_id)
        await self.sweep()
        resource = await self.retrieve(public_response_id, authority)
        checkpoint = resource.checkpoint
        if checkpoint.kind is not CheckpointKind.COMPLETED_OUTWARD_TURN:
            raise ConversationAuthorizationError()
        return checkpoint

    async def prepare_turn(
        self,
        *,
        authority: AuthorityScope,
        input_text: str,
        parent: ConversationCheckpoint | None,
        reasoning_context: ReasoningContext,
        compact_threshold: int | None,
        includes: tuple[str, ...],
        tool_names: tuple[str, ...],
        streaming: bool,
        idempotency_key: str | None,
        request_fingerprint: str,
    ) -> PreparedServedResponsesTurn:
        """Build and validate one policy-bound coordinated agent turn."""
        if fullmatch(r"[0-9a-f]{64}", request_fingerprint) is None:
            raise ConversationValidationError()
        if idempotency_key is None:
            suffix = uuid4().hex
        else:
            validate_identifier(idempotency_key, "idempotency_key")
            suffix = sha256(
                canonical_json_bytes(
                    {
                        "authority": authority_digest(authority),
                        "idempotency_key": idempotency_key,
                        "request_fingerprint": request_fingerprint,
                    }
                )
            ).hexdigest()[:32]
        public_id = PublicResponseId(f"resp_avl_{suffix}")
        provisional_id = ProvisionalResponseId(f"provisional-avl-{suffix}")
        key = RequestIdempotencyKey(
            idempotency_key or f"served-response-{suffix}"
        )
        plan = ServedResponsesTurnPlan(
            authority=authority,
            input_text=input_text,
            public_response_id=public_id,
            provisional_response_id=provisional_id,
            idempotency_key=key,
            request_fingerprint=request_fingerprint,
            retention=self.configuration.policy.retention,
            reasoning_context=reasoning_context,
            compact_threshold=compact_threshold,
            includes=includes,
            tool_names=tool_names,
            parent=parent,
            streaming=streaming,
        )
        prepared = await self.configuration.turn_resolver(plan)
        if type(prepared) is not PreparedServedResponsesTurn:
            raise ConversationValidationError()
        turn = prepared.turn
        if (
            turn.authority != authority
            or turn.public_response_id != public_id
            or turn.provisional_response_id != provisional_id
            or turn.idempotency_key != key
            or turn.retention != self.configuration.policy.retention
            or turn.parent != parent
        ):
            raise ConversationValidationError()
        return prepared

    async def retrieve(
        self,
        public_response_id: str,
        authority: AuthorityScope,
    ) -> StoredResponsesResource:
        """Retrieve one authorized published public response."""
        validate_public_response_id(public_response_id)
        result = await self.configuration.store.retrieve(
            PublicResponseId(public_response_id),
            authority,
        )
        checkpoint = await self.configuration.store.load(
            result.handle.checkpoint_id,
            authority,
        )
        if result.public_response_id != public_response_id:
            raise ConversationAuthorizationError()
        await self._require_live(checkpoint)
        return StoredResponsesResource(
            result=result,
            checkpoint=checkpoint,
            public_model=self.configuration.policy.public_model,
        )

    async def assert_committed(
        self,
        public_response_id: str,
        authority: AuthorityScope,
    ) -> StoredResponsesResource:
        """Return the committed result or a safe commit failure."""
        try:
            return await self.retrieve(public_response_id, authority)
        except ConversationAuthorizationError as error:
            raise ConversationCommitError() from error

    async def tombstone(
        self,
        public_response_id: str,
        authority: AuthorityScope,
    ) -> LocalDeletionState:
        """Apply one idempotent immediate local tombstone."""
        validate_public_response_id(public_response_id)
        await self.sweep()
        identifier = PublicResponseId(public_response_id)
        prepared = await self.configuration.store.prepare_deletion(
            identifier,
            authority,
        )
        if prepared.state is LocalDeletionState.ACTIVE:
            if prepared.checkpoint is None:
                raise ConversationStorageError()
            await self._require_live(prepared.checkpoint)
            try:
                await self.configuration.store.tombstone(
                    identifier,
                    authority,
                    await self.now(),
                )
                return LocalDeletionState.TOMBSTONED
            except (
                ConversationAuthorizationError,
                ConversationConflictError,
            ) as error:
                settled = await self.configuration.store.prepare_deletion(
                    identifier,
                    authority,
                )
                if settled.state in {
                    LocalDeletionState.TOMBSTONED,
                    LocalDeletionState.DELETED,
                }:
                    return settled.state
                raise error
        return prepared.state

    async def _require_live(
        self,
        checkpoint: ConversationCheckpoint,
    ) -> None:
        """Reject an expired target independently of bounded sweeping."""
        expires_at = checkpoint.timestamps.expires_at
        if expires_at is not None and expires_at <= await self.now():
            raise ConversationAuthorizationError()

    async def aclose(self) -> None:
        """Close durable storage only when ownership is explicit."""
        if self.configuration.close_store_on_shutdown:
            await self.configuration.store.close()


def configure_served_responses(
    app: FastAPI,
    configuration: ServedResponsesConfiguration | None,
) -> None:
    """Install or remove the typed durable Responses service."""
    if configuration is None:
        if hasattr(app.state, "served_responses_service"):
            delattr(app.state, "served_responses_service")
        return
    app.state.served_responses_service = ServedResponsesService(configuration)


async def close_served_responses(app: FastAPI) -> None:
    """Close one configured service and release owned durable resources."""
    service = getattr(app.state, "served_responses_service", None)
    if isinstance(service, ServedResponsesService):
        await service.aclose()


def validate_public_response_id(value: object) -> None:
    """Reject malformed and upstream-looking public response identifiers."""
    if (
        not isinstance(value, str)
        or fullmatch(PUBLIC_RESPONSE_ID_PATTERN, value) is None
    ):
        raise ConversationAuthorizationError()


def _public_item_id(
    public_response_id: PublicResponseId,
    item_index: int,
    content: str,
) -> str:
    """Return one stable content-bound public output item identifier."""
    payload = canonical_json_bytes(
        {
            "content": content,
            "item_index": item_index,
            "public_response_id": public_response_id,
        }
    )
    return f"msg_avl_{sha256(payload).hexdigest()[:32]}"


def _require_async_callable(value: object, name: str) -> None:
    """Require one asynchronous callable configuration boundary."""
    if not callable(value) or not iscoroutinefunction(value):
        raise TypeError(f"{name} must be asynchronous")


def _require_async_method(value: object, method_name: str, name: str) -> None:
    """Require one asynchronous method on a configured service."""
    method = getattr(value, method_name, None)
    if not callable(method) or not iscoroutinefunction(method):
        raise TypeError(f"{name}.{method_name} must be asynchronous")
