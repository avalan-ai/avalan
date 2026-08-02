"""Freeze dormant conversation state and storage contracts."""

from ..interaction.entities import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    StateRevision,
)

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import NewType, TypeAlias, final

CONVERSATION_CONTRACT_VERSION = 1

ConversationId = NewType("ConversationId", str)
LogicalTurnId = NewType("LogicalTurnId", str)
ExecutionSegmentId = NewType("ExecutionSegmentId", str)
CheckpointId = NewType("CheckpointId", str)
ConversationBranchId = NewType("ConversationBranchId", str)
NamedHeadId = NewType("NamedHeadId", str)
ProviderLaneId = NewType("ProviderLaneId", str)
ConversationModelCallId = NewType("ConversationModelCallId", str)
PublicResponseId = NewType("PublicResponseId", str)
ProvisionalResponseId = NewType("ProvisionalResponseId", str)
UpstreamResponseId = NewType("UpstreamResponseId", str)
ConversationTaskId = NewType("ConversationTaskId", str)
ConversationAgentId = NewType("ConversationAgentId", str)
StructuredInputContinuationId: TypeAlias = ContinuationId
AuthorityTenantId = NewType("AuthorityTenantId", str)
AuthorityPrincipalId = NewType("AuthorityPrincipalId", str)
AuthorityEndpointId = NewType("AuthorityEndpointId", str)
RequestIdempotencyKey = NewType("RequestIdempotencyKey", str)
CanonicalRequestDigest = NewType("CanonicalRequestDigest", str)
ContinuationDigest = NewType("ContinuationDigest", str)
CheckpointSequence = NewType("CheckpointSequence", int)
NamedHeadRevision = NewType("NamedHeadRevision", int)


class CheckpointKind(StrEnum):
    """Identify one immutable conversation checkpoint boundary."""

    INTERNAL_PROVIDER_BOUNDARY = "internal_provider_boundary"
    STRUCTURED_INPUT_SUSPENSION = "structured_input_suspension"
    COMPLETED_OUTWARD_TURN = "completed_outward_turn"
    STANDALONE_COMPACT_RESULT = "standalone_compact_result"
    TOMBSTONE = "tombstone"
    SUPERSESSION = "supersession"


class CheckpointVisibility(StrEnum):
    """Identify how a checkpoint may leave its owning subsystem."""

    PRIVATE_EXECUTION = "private_execution"
    PUBLIC_RESPONSE = "public_response"
    CALLER_HELD = "caller_held"
    PRIVATE_LIFECYCLE = "private_lifecycle"


class CheckpointCommitState(StrEnum):
    """Identify whether staged checkpoint bytes became authoritative."""

    STAGED = "staged"
    COMMITTED = "committed"
    QUARANTINED = "quarantined"


class PublicResponseIdState(StrEnum):
    """Identify whether a provisional response ID was committed."""

    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    WITHHELD = "withheld"


class PublicResponseMappingState(StrEnum):
    """Identify whether a public ID resolves through Avalan-owned state."""

    ABSENT = "absent"
    PRIVATE_TRANSIENT = "private_transient"
    ADDRESSABLE = "addressable"
    TOMBSTONED = "tombstoned"


class ResponseResourceState(StrEnum):
    """Identify the lifecycle state of an outward response resource."""

    ALLOCATED = "allocated"
    DISPATCHING = "dispatching"
    STREAMING = "streaming"
    INPUT_REQUIRED = "input_required"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"
    DELETED = "deleted"
    EXPIRED = "expired"


class ResponseOperation(StrEnum):
    """Identify a caller operation on a response resource."""

    RETRIEVE = "retrieve"
    CONTINUE = "continue"
    COMPACT = "compact"
    DELETE = "delete"


class ResponseOperationDisposition(StrEnum):
    """Describe the caller-visible result of a response operation."""

    ALLOWED = "allowed"
    STRUCTURED_INPUT_ONLY = "structured_input_only"
    DENIED_STATE = "denied_state"
    NOT_ADDRESSABLE = "not_addressable"
    CONCEALED = "concealed"


class ParentAdvanceMode(StrEnum):
    """Identify how an immutable parent is used for a child checkpoint."""

    ORDINARY_CHILD = "ordinary_child"
    EXPLICIT_BRANCH = "explicit_branch"
    NAMED_HEAD = "named_head"


class NamedHeadAdvanceDisposition(StrEnum):
    """Describe a named-head compare-and-swap decision."""

    ADVANCE = "advance"
    CONFLICT = "conflict"


class ConversationOperation(StrEnum):
    """Identify an operation within an idempotency namespace."""

    CREATE = "create"
    CONTINUE = "continue"
    BRANCH = "branch"
    COMPACT = "compact"
    RETRIEVE = "retrieve"
    DELETE = "delete"


class IdempotencyRecordState(StrEnum):
    """Identify the durable outcome known for an idempotent operation."""

    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    FAILED_NO_DISPATCH = "failed_no_dispatch"
    AMBIGUOUS = "ambiguous"


class IdempotencyDisposition(StrEnum):
    """Describe how a request relates to an idempotency record."""

    EXECUTE = "execute"
    REPLAY_COMMITTED = "replay_committed"
    CONFLICT = "conflict"
    FENCED = "fenced"


class FailureBoundary(StrEnum):
    """Identify a failure or effect boundary relevant to retries."""

    VALIDATION_BEFORE_DISPATCH = "validation_before_dispatch"
    PROVIDER_REJECTION = "provider_rejection"
    KNOWN_NO_DISPATCH_TRANSPORT = "known_no_dispatch_transport"
    AMBIGUOUS_POSSIBLE_DISPATCH = "ambiguous_possible_dispatch"
    FAILURE_BEFORE_OUTPUT = "failure_before_output"
    FAILURE_AFTER_VISIBLE_OUTPUT = "failure_after_visible_output"
    MALFORMED_STREAM_ITEM = "malformed_stream_item"
    TOOL_EFFECT = "tool_effect"
    SUSPENSION = "suspension"
    CHECKPOINT_COMMIT = "checkpoint_commit"
    OUTWARD_PUBLICATION = "outward_publication"


class RetryRule(StrEnum):
    """Identify the only permitted automatic-retry category."""

    NEVER = "never"
    BOUNDED_EFFECT_FREE = "bounded_effect_free"
    FENCED_RECONCILIATION = "fenced_reconciliation"


class LocalResponseStorage(StrEnum):
    """Identify Avalan-owned response retention independently of providers."""

    NONE = "none"
    PROCESS_LOCAL = "process_local"
    TRANSIENT = "transient"
    DURABLE = "durable"


class ProviderLaneStorage(StrEnum):
    """Identify upstream provider-lane continuation storage."""

    OFF = "off"
    STATELESS = "stateless"
    STORED = "stored"


class ProviderLaneOwnerKind(StrEnum):
    """Identify the execution owner used to derive a provider lane ID."""

    DIRECT_MODEL = "direct_model"
    PARENT_AGENT = "parent_agent"
    CHILD_AGENT = "child_agent"


class ChildLaneRetentionPolicy(StrEnum):
    """Identify deterministic child-lane retention after an outward turn."""

    RETAIN = "retain"
    DISCARD_TERMINAL = "discard_terminal"


class UpstreamLifetimeStatus(StrEnum):
    """Distinguish absent, unknown, and known upstream lifetimes."""

    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"
    KNOWN = "known"


class LocalDeletionState(StrEnum):
    """Identify the local-first deletion lifecycle."""

    ACTIVE = "active"
    TOMBSTONED = "tombstoned"
    DELETED = "deleted"


class UpstreamDeletionState(StrEnum):
    """Identify upstream cleanup status without restoring local access."""

    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ConfigurationSource(StrEnum):
    """Identify one layer in the conversation configuration authority."""

    SERVER_POLICY = "server_policy"
    SERVED_AGENT = "served_agent"
    MODEL_PROVIDER = "model_provider"
    REQUEST = "request"
    PROVIDER_DEFAULT = "provider_default"


class AuthoritySource(StrEnum):
    """Identify a trusted source of conversation authority."""

    TRUSTED_HOST_CONTEXT = "trusted_host_context"
    AUTHENTICATED_SERVER_CONTEXT = "authenticated_server_context"
    FIXED_LOCAL_SINGLE_USER = "fixed_local_single_user"


class ConversationSurface(StrEnum):
    """Identify a potential conversation-continuity surface."""

    DIRECT_MODEL_SDK = "direct_model_sdk"
    AGENT_SDK = "agent_sdk"
    CLI = "cli"
    FLOW = "flow"
    MCP = "mcp"
    A2A = "a2a"
    SERVED_RESPONSES = "served_responses"


class SurfaceDisposition(StrEnum):
    """Identify an initial-release surface decision."""

    ACTIVATED = "activated"
    DEFERRED = "deferred"
    INCAPABLE = "incapable"


class MigrationDisposition(StrEnum):
    """Identify a compatibility decision at a persistence boundary."""

    REFERENCE_EXISTING = "reference_existing"
    VERSIONED_MIGRATION = "versioned_migration"
    COMPATIBLE_READ = "compatible_read"
    REJECT_ROLLBACK = "reject_rollback"


def _assert_invariant(condition: bool, *message_parts: str) -> None:
    """Assert one contract invariant with its caller-facing explanation."""
    assert condition, " ".join(message_parts)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorityScope:
    """Bind state to trusted tenant, principal, agent, and endpoint scope."""

    source: AuthoritySource
    principal_id: AuthorityPrincipalId
    agent_id: ConversationAgentId
    endpoint_id: AuthorityEndpointId
    tenant_id: AuthorityTenantId | None = None
    local_single_user_configured: bool = False
    network_exposed: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.source, AuthoritySource)
        _validate_identifier(self.principal_id, "principal_id")
        _validate_identifier(self.agent_id, "agent_id")
        _validate_identifier(self.endpoint_id, "endpoint_id")
        if self.tenant_id is not None:
            _validate_identifier(self.tenant_id, "tenant_id")
        assert type(self.local_single_user_configured) is bool
        assert type(self.network_exposed) is bool
        if self.source is AuthoritySource.AUTHENTICATED_SERVER_CONTEXT:
            _assert_invariant(
                self.tenant_id is not None,
                "authenticated server authority requires a tenant",
            )
        if self.source is AuthoritySource.FIXED_LOCAL_SINGLE_USER:
            _assert_invariant(
                self.tenant_id is None,
                "fixed local single-user authority cannot claim a tenant",
            )
            assert self.local_single_user_configured, (
                "fixed local single-user authority requires explicit"
                " configuration"
            )
            _assert_invariant(
                not self.network_exposed,
                "fixed local single-user authority cannot be network exposed",
            )
        else:
            _assert_invariant(
                not self.local_single_user_configured,
                "local single-user configuration requires",
                "fixed local authority",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointIdentity:
    """Identify one immutable boundary within a logical turn and branch."""

    conversation_id: ConversationId
    logical_turn_id: LogicalTurnId
    execution_segment_id: ExecutionSegmentId
    checkpoint_id: CheckpointId
    branch_id: ConversationBranchId
    sequence: CheckpointSequence
    parent_checkpoint_id: CheckpointId | None = None
    parent_sequence: CheckpointSequence | None = None

    def __post_init__(self) -> None:
        for name in (
            "conversation_id",
            "logical_turn_id",
            "execution_segment_id",
            "checkpoint_id",
            "branch_id",
        ):
            _validate_identifier(getattr(self, name), name)
        _validate_revision(self.sequence, "sequence")
        if self.parent_checkpoint_id is None:
            _assert_invariant(
                self.parent_sequence is None,
                "root checkpoint cannot carry a parent sequence",
            )
            assert self.sequence == 0, "root checkpoint sequence must be zero"
        else:
            _validate_identifier(
                self.parent_checkpoint_id,
                "parent_checkpoint_id",
            )
            parent_sequence = self.parent_sequence
            message = "child checkpoint requires its parent sequence"
            assert parent_sequence is not None, message
            _validate_revision(parent_sequence, "parent_sequence")
            _assert_invariant(
                self.sequence == parent_sequence + 1,
                "child checkpoint sequence must equal parent sequence",
                "plus one",
            )
            _assert_invariant(
                self.parent_checkpoint_id != self.checkpoint_id,
                "checkpoint cannot parent itself",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PortableContinuationReference:
    """Reference durable structured-input state without copying its payload."""

    continuation_id: ContinuationId
    state_revision: StateRevision
    digest: ContinuationDigest
    definition: ExecutionDefinitionRef
    revision_binding: ContinuationRevisionBinding

    def __post_init__(self) -> None:
        _validate_identifier(self.continuation_id, "continuation_id")
        _validate_revision(self.state_revision, "state_revision")
        _validate_identifier(self.digest, "digest")
        assert type(self.definition) is ExecutionDefinitionRef
        assert type(self.revision_binding) is ContinuationRevisionBinding
        assert (
            self.definition.capability_revision
            == self.revision_binding.capability_revision
        ), "continuation capability revisions must match"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RequestIdempotencyIdentity:
    """Scope request deduplication to authority, operation, key, and digest."""

    authority: AuthorityScope
    operation: ConversationOperation
    key: RequestIdempotencyKey
    request_digest: CanonicalRequestDigest

    def __post_init__(self) -> None:
        assert type(self.authority) is AuthorityScope
        assert isinstance(self.operation, ConversationOperation)
        _validate_identifier(self.key, "key")
        _validate_identifier(self.request_digest, "request_digest")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class IdempotencyRecord:
    """Record the known state of one scoped idempotent operation."""

    identity: RequestIdempotencyIdentity
    state: IdempotencyRecordState

    def __post_init__(self) -> None:
        assert type(self.identity) is RequestIdempotencyIdentity
        assert isinstance(self.state, IdempotencyRecordState)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class FailureFence:
    """Freeze retry, parent-preservation, and reconciliation behavior."""

    retry_rule: RetryRule
    fence_duplicate_dispatch: bool
    preserve_parent: bool
    quarantine_completed_upstream: bool
    reconciliation_required: bool

    def __post_init__(self) -> None:
        assert isinstance(self.retry_rule, RetryRule)
        for value in (
            self.fence_duplicate_dispatch,
            self.preserve_parent,
            self.quarantine_completed_upstream,
            self.reconciliation_required,
        ):
            assert type(value) is bool


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoragePolicy:
    """Keep Avalan response storage independent from provider storage."""

    local: LocalResponseStorage
    upstream: ProviderLaneStorage
    provider_storage_disclosed: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.local, LocalResponseStorage)
        assert isinstance(self.upstream, ProviderLaneStorage)
        assert type(self.provider_storage_disclosed) is bool
        assert (
            self.upstream is not ProviderLaneStorage.STORED
            or self.provider_storage_disclosed
        ), "provider-stored continuation requires explicit disclosure"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseStorageContext:
    """Bind independent storage axes to public response addressability."""

    policy: StoragePolicy
    public_mapping: PublicResponseMappingState

    def __post_init__(self) -> None:
        assert type(self.policy) is StoragePolicy
        assert isinstance(self.public_mapping, PublicResponseMappingState)
        if self.public_mapping is PublicResponseMappingState.PRIVATE_TRANSIENT:
            _assert_invariant(
                self.policy.local is LocalResponseStorage.TRANSIENT,
                "private transient mappings require transient",
                "local storage",
            )
        if self.public_mapping in {
            PublicResponseMappingState.ADDRESSABLE,
            PublicResponseMappingState.TOMBSTONED,
        }:
            assert self.policy.local in {
                LocalResponseStorage.PROCESS_LOCAL,
                LocalResponseStorage.DURABLE,
            }, "public mappings require process-local or durable storage"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RetentionLimits:
    """Compute retention from the minimum applicable independent lifetime."""

    storage: StoragePolicy
    upstream_lifetime_status: UpstreamLifetimeStatus
    local_ttl_seconds: int | None = None
    envelope_ttl_seconds: int | None = None
    known_upstream_ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        assert type(self.storage) is StoragePolicy
        assert isinstance(
            self.upstream_lifetime_status, UpstreamLifetimeStatus
        )
        for value in (
            self.local_ttl_seconds,
            self.envelope_ttl_seconds,
            self.known_upstream_ttl_seconds,
        ):
            if value is not None:
                assert type(value) is int and value > 0
        if self.storage.upstream is ProviderLaneStorage.STORED:
            assert self.upstream_lifetime_status in {
                UpstreamLifetimeStatus.UNKNOWN,
                UpstreamLifetimeStatus.KNOWN,
            }, "provider-stored state requires an upstream lifetime status"
        else:
            assert (
                self.upstream_lifetime_status
                is UpstreamLifetimeStatus.NOT_APPLICABLE
            ), "non-stored provider state cannot claim an upstream lifetime"
        if self.upstream_lifetime_status is UpstreamLifetimeStatus.KNOWN:
            _assert_invariant(
                self.known_upstream_ttl_seconds is not None,
                "known upstream lifetime requires a positive TTL",
            )
        else:
            _assert_invariant(
                self.known_upstream_ttl_seconds is None,
                "unknown or inapplicable upstream lifetime cannot carry a TTL",
            )

    @property
    def effective_ttl_seconds(self) -> int | None:
        """Return the minimum lifetime that is known and applicable."""
        applicable = tuple(
            value
            for value in (
                self.local_ttl_seconds,
                self.envelope_ttl_seconds,
                self.known_upstream_ttl_seconds,
            )
            if value is not None
        )
        return min(applicable) if applicable else None


CHECKPOINT_VISIBILITY: Mapping[CheckpointKind, CheckpointVisibility] = (
    MappingProxyType(
        {
            CheckpointKind.INTERNAL_PROVIDER_BOUNDARY: (
                CheckpointVisibility.PRIVATE_EXECUTION
            ),
            CheckpointKind.STRUCTURED_INPUT_SUSPENSION: (
                CheckpointVisibility.PRIVATE_EXECUTION
            ),
            CheckpointKind.COMPLETED_OUTWARD_TURN: (
                CheckpointVisibility.PUBLIC_RESPONSE
            ),
            CheckpointKind.STANDALONE_COMPACT_RESULT: (
                CheckpointVisibility.CALLER_HELD
            ),
            CheckpointKind.TOMBSTONE: CheckpointVisibility.PRIVATE_LIFECYCLE,
            CheckpointKind.SUPERSESSION: (
                CheckpointVisibility.PRIVATE_LIFECYCLE
            ),
        },
    )
)

CHECKPOINT_COMMIT_TRANSITIONS: Mapping[
    CheckpointCommitState, frozenset[CheckpointCommitState]
] = MappingProxyType(
    {
        CheckpointCommitState.STAGED: frozenset(
            {
                CheckpointCommitState.COMMITTED,
                CheckpointCommitState.QUARANTINED,
            },
        ),
        CheckpointCommitState.COMMITTED: frozenset(),
        CheckpointCommitState.QUARANTINED: frozenset(),
    },
)

PUBLIC_RESPONSE_ID_TRANSITIONS: Mapping[
    PublicResponseIdState, frozenset[PublicResponseIdState]
] = MappingProxyType(
    {
        PublicResponseIdState.PROVISIONAL: frozenset(
            {
                PublicResponseIdState.COMMITTED,
                PublicResponseIdState.WITHHELD,
            },
        ),
        PublicResponseIdState.COMMITTED: frozenset(),
        PublicResponseIdState.WITHHELD: frozenset(),
    },
)

RESPONSE_RESOURCE_TRANSITIONS: Mapping[
    ResponseResourceState, frozenset[ResponseResourceState]
] = MappingProxyType(
    {
        ResponseResourceState.ALLOCATED: frozenset(
            {
                ResponseResourceState.DISPATCHING,
                ResponseResourceState.FAILED,
            },
        ),
        ResponseResourceState.DISPATCHING: frozenset(
            {
                ResponseResourceState.STREAMING,
                ResponseResourceState.INPUT_REQUIRED,
                ResponseResourceState.COMMITTING,
                ResponseResourceState.FAILED,
            },
        ),
        ResponseResourceState.STREAMING: frozenset(
            {
                ResponseResourceState.INPUT_REQUIRED,
                ResponseResourceState.COMMITTING,
                ResponseResourceState.FAILED,
            },
        ),
        ResponseResourceState.INPUT_REQUIRED: frozenset(
            {
                ResponseResourceState.DISPATCHING,
                ResponseResourceState.FAILED,
                ResponseResourceState.TOMBSTONED,
                ResponseResourceState.EXPIRED,
            },
        ),
        ResponseResourceState.COMMITTING: frozenset(
            {
                ResponseResourceState.COMPLETED,
                ResponseResourceState.FAILED,
            },
        ),
        ResponseResourceState.COMPLETED: frozenset(
            {
                ResponseResourceState.TOMBSTONED,
                ResponseResourceState.EXPIRED,
            },
        ),
        ResponseResourceState.FAILED: frozenset(
            {
                ResponseResourceState.TOMBSTONED,
                ResponseResourceState.EXPIRED,
            },
        ),
        ResponseResourceState.TOMBSTONED: frozenset(
            {ResponseResourceState.DELETED},
        ),
        ResponseResourceState.DELETED: frozenset(),
        ResponseResourceState.EXPIRED: frozenset(
            {ResponseResourceState.DELETED},
        ),
    },
)

_NOT_ADDRESSABLE_OPERATIONS: Mapping[
    ResponseOperation, ResponseOperationDisposition
] = MappingProxyType(
    {
        operation: ResponseOperationDisposition.NOT_ADDRESSABLE
        for operation in ResponseOperation
    },
)
_CONCEALED_OPERATIONS: Mapping[
    ResponseOperation, ResponseOperationDisposition
] = MappingProxyType(
    {
        operation: ResponseOperationDisposition.CONCEALED
        for operation in ResponseOperation
    },
)
RESPONSE_OPERATION_POLICY: Mapping[
    ResponseResourceState,
    Mapping[ResponseOperation, ResponseOperationDisposition],
] = MappingProxyType(
    {
        ResponseResourceState.ALLOCATED: _NOT_ADDRESSABLE_OPERATIONS,
        ResponseResourceState.DISPATCHING: _NOT_ADDRESSABLE_OPERATIONS,
        ResponseResourceState.STREAMING: _NOT_ADDRESSABLE_OPERATIONS,
        ResponseResourceState.INPUT_REQUIRED: MappingProxyType(
            {
                ResponseOperation.RETRIEVE: (
                    ResponseOperationDisposition.ALLOWED
                ),
                ResponseOperation.CONTINUE: (
                    ResponseOperationDisposition.STRUCTURED_INPUT_ONLY
                ),
                ResponseOperation.COMPACT: (
                    ResponseOperationDisposition.DENIED_STATE
                ),
                ResponseOperation.DELETE: ResponseOperationDisposition.ALLOWED,
            },
        ),
        ResponseResourceState.COMMITTING: _NOT_ADDRESSABLE_OPERATIONS,
        ResponseResourceState.COMPLETED: MappingProxyType(
            {
                operation: ResponseOperationDisposition.ALLOWED
                for operation in ResponseOperation
            },
        ),
        ResponseResourceState.FAILED: MappingProxyType(
            {
                ResponseOperation.RETRIEVE: (
                    ResponseOperationDisposition.ALLOWED
                ),
                ResponseOperation.CONTINUE: (
                    ResponseOperationDisposition.DENIED_STATE
                ),
                ResponseOperation.COMPACT: (
                    ResponseOperationDisposition.DENIED_STATE
                ),
                ResponseOperation.DELETE: ResponseOperationDisposition.ALLOWED,
            },
        ),
        ResponseResourceState.TOMBSTONED: _CONCEALED_OPERATIONS,
        ResponseResourceState.DELETED: _CONCEALED_OPERATIONS,
        ResponseResourceState.EXPIRED: _CONCEALED_OPERATIONS,
    },
)

FAILURE_FENCES: Mapping[FailureBoundary, FailureFence] = MappingProxyType(
    {
        FailureBoundary.VALIDATION_BEFORE_DISPATCH: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=False,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        ),
        FailureBoundary.PROVIDER_REJECTION: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=False,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        ),
        FailureBoundary.KNOWN_NO_DISPATCH_TRANSPORT: FailureFence(
            retry_rule=RetryRule.BOUNDED_EFFECT_FREE,
            fence_duplicate_dispatch=False,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        ),
        FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH: FailureFence(
            retry_rule=RetryRule.FENCED_RECONCILIATION,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=True,
        ),
        FailureBoundary.FAILURE_BEFORE_OUTPUT: FailureFence(
            retry_rule=RetryRule.BOUNDED_EFFECT_FREE,
            fence_duplicate_dispatch=False,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        ),
        FailureBoundary.FAILURE_AFTER_VISIBLE_OUTPUT: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=True,
        ),
        FailureBoundary.MALFORMED_STREAM_ITEM: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=True,
        ),
        FailureBoundary.TOOL_EFFECT: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=True,
        ),
        FailureBoundary.SUSPENSION: FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        ),
        FailureBoundary.CHECKPOINT_COMMIT: FailureFence(
            retry_rule=RetryRule.FENCED_RECONCILIATION,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=True,
            reconciliation_required=True,
        ),
        FailureBoundary.OUTWARD_PUBLICATION: FailureFence(
            retry_rule=RetryRule.FENCED_RECONCILIATION,
            fence_duplicate_dispatch=True,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=True,
        ),
    },
)

LOCAL_DELETION_TRANSITIONS: Mapping[
    LocalDeletionState, frozenset[LocalDeletionState]
] = MappingProxyType(
    {
        LocalDeletionState.ACTIVE: frozenset({LocalDeletionState.TOMBSTONED}),
        LocalDeletionState.TOMBSTONED: frozenset({LocalDeletionState.DELETED}),
        LocalDeletionState.DELETED: frozenset(),
    },
)

UPSTREAM_DELETION_TRANSITIONS: Mapping[
    UpstreamDeletionState, frozenset[UpstreamDeletionState]
] = MappingProxyType(
    {
        UpstreamDeletionState.NOT_APPLICABLE: frozenset(),
        UpstreamDeletionState.PENDING: frozenset(
            {
                UpstreamDeletionState.SUCCEEDED,
                UpstreamDeletionState.FAILED,
                UpstreamDeletionState.UNSUPPORTED,
            },
        ),
        UpstreamDeletionState.SUCCEEDED: frozenset(),
        UpstreamDeletionState.FAILED: frozenset(
            {UpstreamDeletionState.PENDING},
        ),
        UpstreamDeletionState.UNSUPPORTED: frozenset(),
    },
)

CONFIGURATION_PRECEDENCE = (
    ConfigurationSource.SERVER_POLICY,
    ConfigurationSource.SERVED_AGENT,
    ConfigurationSource.MODEL_PROVIDER,
    ConfigurationSource.REQUEST,
    ConfigurationSource.PROVIDER_DEFAULT,
)


def response_transition_allowed(
    source: ResponseResourceState,
    target: ResponseResourceState,
) -> bool:
    """Return whether one response lifecycle mutation is legal."""
    assert isinstance(source, ResponseResourceState)
    assert isinstance(target, ResponseResourceState)
    return target in RESPONSE_RESOURCE_TRANSITIONS[source]


def response_operation_disposition(
    state: ResponseResourceState,
    operation: ResponseOperation,
    public_id_state: PublicResponseIdState,
    storage: ResponseStorageContext,
) -> ResponseOperationDisposition:
    """Return the caller-visible policy for one response operation."""
    assert isinstance(state, ResponseResourceState)
    assert isinstance(operation, ResponseOperation)
    assert isinstance(public_id_state, PublicResponseIdState)
    assert type(storage) is ResponseStorageContext
    if public_id_state is not PublicResponseIdState.COMMITTED:
        return ResponseOperationDisposition.NOT_ADDRESSABLE
    if (
        state
        in {
            ResponseResourceState.TOMBSTONED,
            ResponseResourceState.DELETED,
            ResponseResourceState.EXPIRED,
        }
        or storage.public_mapping is PublicResponseMappingState.TOMBSTONED
    ):
        return ResponseOperationDisposition.CONCEALED
    if storage.public_mapping is not PublicResponseMappingState.ADDRESSABLE:
        return ResponseOperationDisposition.NOT_ADDRESSABLE
    state_disposition = RESPONSE_OPERATION_POLICY[state][operation]
    if state_disposition not in {
        ResponseOperationDisposition.ALLOWED,
        ResponseOperationDisposition.STRUCTURED_INPUT_ONLY,
    }:
        return state_disposition
    if (
        operation is ResponseOperation.CONTINUE
        and storage.policy.upstream is ProviderLaneStorage.OFF
    ):
        return ResponseOperationDisposition.DENIED_STATE
    if (
        operation is ResponseOperation.COMPACT
        and storage.policy.upstream is not ProviderLaneStorage.STATELESS
    ):
        return ResponseOperationDisposition.DENIED_STATE
    return state_disposition


def terminal_publication_allowed(
    state: ResponseResourceState,
    checkpoint_state: CheckpointCommitState,
    public_id_state: PublicResponseIdState,
) -> bool:
    """Return whether a terminal completed event may be published."""
    assert isinstance(state, ResponseResourceState)
    assert isinstance(checkpoint_state, CheckpointCommitState)
    assert isinstance(public_id_state, PublicResponseIdState)
    return (
        state is ResponseResourceState.COMPLETED
        and checkpoint_state is CheckpointCommitState.COMMITTED
        and public_id_state is PublicResponseIdState.COMMITTED
    )


def named_head_advance_disposition(
    expected_revision: NamedHeadRevision,
    current_revision: NamedHeadRevision,
) -> NamedHeadAdvanceDisposition:
    """Return the deterministic named-head compare-and-swap result."""
    _validate_revision(expected_revision, "expected_revision")
    _validate_revision(current_revision, "current_revision")
    if expected_revision == current_revision:
        return NamedHeadAdvanceDisposition.ADVANCE
    return NamedHeadAdvanceDisposition.CONFLICT


def idempotency_disposition(
    request: RequestIdempotencyIdentity,
    existing: IdempotencyRecord | None,
) -> IdempotencyDisposition:
    """Return whether to execute, replay, reject, or fence a request."""
    assert type(request) is RequestIdempotencyIdentity
    if existing is None:
        return IdempotencyDisposition.EXECUTE
    assert type(existing) is IdempotencyRecord
    if (
        request.authority != existing.identity.authority
        or request.operation is not existing.identity.operation
        or request.key != existing.identity.key
    ):
        return IdempotencyDisposition.EXECUTE
    if request.request_digest != existing.identity.request_digest:
        return IdempotencyDisposition.CONFLICT
    if existing.state is IdempotencyRecordState.COMMITTED:
        return IdempotencyDisposition.REPLAY_COMMITTED
    if existing.state is IdempotencyRecordState.FAILED_NO_DISPATCH:
        return IdempotencyDisposition.EXECUTE
    return IdempotencyDisposition.FENCED


def _validate_identifier(value: str, name: str) -> None:
    assert isinstance(value, str), f"{name} must be a string"
    assert value and value == value.strip(), f"{name} must be non-empty"


def _validate_revision(value: int, name: str) -> None:
    _assert_invariant(
        type(value) is int and value >= 0,
        f"{name} must be a non-negative integer",
    )


def capability_revision(
    reference: PortableContinuationReference,
) -> CapabilityRevision:
    """Return the shared capability revision of a continuation reference."""
    assert type(reference) is PortableContinuationReference
    return reference.revision_binding.capability_revision
