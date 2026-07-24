"""Define the authoritative async interaction-store contract."""

from .codec import canonical_resolution_digest, semantic_request_fingerprint
from .entities import (
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancellationScope,
    CancelledResolution,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ControllerId,
    DeadlineScheduleRevision,
    ExpiredResolution,
    FreeFormOther,
    InputCandidateResolution,
    InputRequest,
    InputRequestId,
    InputResolution,
    InteractionStoreGeneration,
    InteractionStoreRevision,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    RunId,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    SupersededResolution,
    TaskId,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    _is_input_candidate_resolution,
)
from .error import InputErrorCode, InputValidationError
from .handler import InputResumer, InputResumerRegistration
from .policy import (
    MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
    AcquireControllerActivity,
    ControllerActivityAction,
    ControllerActivityEvidence,
    DisconnectControllerActivity,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionBranchAuthorizationTarget,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    InteractionTime,
    PulseControllerActivity,
    ReleaseControllerActivity,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    is_controller_activity_evidence,
    validate_resolution_idempotency_key,
)
from .state import (
    InputTransitionApplied,
    InputTransitionError,
    _anchor_request_presentation,
    mark_request_pending,
    resolve_request,
)
from .validation import (
    MAX_STATE_REVISION,
    validate_aware_datetime,
    validate_bool,
    validate_finite_number,
    validate_int,
    validate_opaque_id,
    validate_state_revision,
)

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from hashlib import sha256
from json import dumps
from re import compile as compile_pattern
from typing import Literal, Protocol, TypeAlias, final

_SHA256_PATTERN = compile_pattern(r"^[0-9a-f]{64}$")


class InteractionPresentationState(StrEnum):
    """Identify attached presentation progress independently from lifecycle."""

    QUEUED = "queued"
    PRESENTED = "presented"
    DETACHED = "detached"


class AdvisoryWaitStatus(StrEnum):
    """Identify authoritative advisory inactivity-budget accounting."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    EXHAUSTED = "exhausted"


class InteractionStoreResultKind(StrEnum):
    """Identify one atomic store-operation result."""

    APPLIED = "applied"
    REPLAYED = "replayed"
    REJECTED = "rejected"


class InteractionReplayKind(StrEnum):
    """Identify why a terminal resolution is returned without re-resolving."""

    SAME_KEY = "same_key"
    SEMANTIC_NEW_KEY = "semantic_new_key"


class ResolutionIdempotencyDisposition(StrEnum):
    """Identify the transport-key decision before lifecycle CAS handling."""

    NEW_RESOLUTION = "new_resolution"
    SAME_KEY = "same_key"
    SAME_KEY_CONFLICT = "same_key_conflict"
    SEMANTIC_NEW_KEY = "semantic_new_key"
    TERMINAL_CONFLICT = "terminal_conflict"
    LEDGER_FULL = "ledger_full"


class ResolutionDecisionStage(StrEnum):
    """Identify one authoritative resolution-admission decision stage."""

    AUTHORIZATION = "authorization"
    CORRELATION = "correlation"
    DEADLINE = "deadline"
    IDEMPOTENCY_KEY = "idempotency_key"
    SEMANTIC_REPLAY = "semantic_replay"
    STATE_REVISION = "state_revision"
    VALIDATION = "validation"
    COMMIT = "commit"


RESOLUTION_DECISION_PRECEDENCE: tuple[ResolutionDecisionStage, ...] = (
    ResolutionDecisionStage.AUTHORIZATION,
    ResolutionDecisionStage.CORRELATION,
    ResolutionDecisionStage.DEADLINE,
    ResolutionDecisionStage.IDEMPOTENCY_KEY,
    ResolutionDecisionStage.SEMANTIC_REPLAY,
    ResolutionDecisionStage.STATE_REVISION,
    ResolutionDecisionStage.VALIDATION,
    ResolutionDecisionStage.COMMIT,
)


_SYSTEM_RESOLVER_TOKEN = object()
_ADMISSION_CAPABILITY_TOKEN = object()
_ADMISSION_COMMAND_TOKEN = object()
_ADMISSION_CLEANUP_RESULT_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionSystemResolver:
    """Seal one internal system resolution authority."""

    def __init__(self, *, _token: object) -> None:
        if _token is not _SYSTEM_RESOLVER_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "resolver",
                "system resolution authority is store-internal",
            )


_DEADLINE_RESOLVER = _InteractionSystemResolver(_token=_SYSTEM_RESOLVER_TOKEN)
_TRUSTED_DEFAULT_RESOLVER = _InteractionSystemResolver(
    _token=_SYSTEM_RESOLVER_TOKEN
)
_TRUSTED_POLICY_RESOLVER = _InteractionSystemResolver(
    _token=_SYSTEM_RESOLVER_TOKEN
)
_ADMISSION_CLEANUP_RESOLVER = _InteractionSystemResolver(
    _token=_SYSTEM_RESOLVER_TOKEN
)

InteractionResolutionAuthority: TypeAlias = (
    PrincipalScope | _InteractionSystemResolver
)
PrincipalAuthoredProvenance: TypeAlias = Literal[
    AnswerProvenance.HUMAN,
    AnswerProvenance.EXTERNAL_CONTROLLER,
]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionCorrelation:
    """Correlate one request with its complete logical execution identity."""

    request_id: InputRequestId
    continuation_id: ContinuationId
    run_id: RunId
    turn_id: TurnId
    agent_id: AgentId
    branch_id: BranchId
    model_call_id: ModelCallId
    task_id: TaskId | None = None

    def __post_init__(self) -> None:
        for field_name, constructor in (
            ("request_id", InputRequestId),
            ("continuation_id", ContinuationId),
            ("run_id", RunId),
            ("turn_id", TurnId),
            ("agent_id", AgentId),
            ("branch_id", BranchId),
            ("model_call_id", ModelCallId),
        ):
            object.__setattr__(
                self,
                field_name,
                constructor(
                    validate_opaque_id(
                        getattr(self, field_name),
                        f"correlation.{field_name}",
                    )
                ),
            )
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                TaskId(
                    validate_opaque_id(self.task_id, "correlation.task_id")
                ),
            )

    @classmethod
    def from_request(cls, request: InputRequest) -> "InteractionCorrelation":
        """Return complete correlation from one canonical request."""
        if type(request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request",
                "value must be an input request",
            )
        origin = request.origin
        return cls(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            run_id=origin.run_id,
            turn_id=origin.turn_id,
            task_id=origin.task_id,
            agent_id=origin.agent_id,
            branch_id=origin.branch_id,
            model_call_id=origin.model_call_id,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionExecutionScope:
    """Select one run or a narrower execution subtree."""

    run_id: RunId
    turn_id: TurnId | None = None
    task_id: TaskId | None = None
    agent_id: AgentId | None = None
    branch_id: BranchId | None = None
    include_descendants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            RunId(validate_opaque_id(self.run_id, "scope.run_id")),
        )
        for field_name, constructor in (
            ("turn_id", TurnId),
            ("task_id", TaskId),
            ("agent_id", AgentId),
            ("branch_id", BranchId),
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    constructor(
                        validate_opaque_id(value, f"scope.{field_name}")
                    ),
                )
        validate_bool(self.include_descendants, "scope.include_descendants")
        if self.include_descendants and self.branch_id is None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "scope.include_descendants",
                "descendant selection requires a branch identifier",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRegistration:
    """Register explicit child ancestry before the child creates a request."""

    run_id: RunId
    branch_id: BranchId
    parent_branch_id: BranchId
    principal: PrincipalScope

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            RunId(validate_opaque_id(self.run_id, "branch.run_id")),
        )
        object.__setattr__(
            self,
            "branch_id",
            BranchId(validate_opaque_id(self.branch_id, "branch.branch_id")),
        )
        object.__setattr__(
            self,
            "parent_branch_id",
            BranchId(
                validate_opaque_id(
                    self.parent_branch_id,
                    "branch.parent_branch_id",
                )
            ),
        )
        if self.branch_id == self.parent_branch_id:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "branch.parent_branch_id",
                "parent branch must differ from child branch",
            )
        if not isinstance(self.principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "branch.principal",
                "value must be a principal scope",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RegisterInteractionBranchCommand:
    """Authorize and persist one explicit branch edge."""

    actor: InteractionActor
    registration: InteractionBranchRegistration

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        if not isinstance(
            self.registration,
            InteractionBranchRegistration,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "branch.registration",
                "value must be an interaction branch registration",
            )

    @property
    def authorization_target(self) -> InteractionBranchAuthorizationTarget:
        """Return the exact branch-specific authorization target."""
        return InteractionBranchAuthorizationTarget(
            run_id=self.registration.run_id,
            branch_id=self.registration.branch_id,
            parent_branch_id=self.registration.parent_branch_id,
            principal=self.registration.principal,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRootLookup:
    """Request the authoritative root for one exact persisted branch."""

    actor: InteractionActor
    run_id: RunId
    branch_id: BranchId

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        object.__setattr__(
            self,
            "run_id",
            RunId(validate_opaque_id(self.run_id, "branch_root.run_id")),
        )
        object.__setattr__(
            self,
            "branch_id",
            BranchId(
                validate_opaque_id(self.branch_id, "branch_root.branch_id")
            ),
        )

    @property
    def authorization_target(self) -> InteractionScopeAuthorizationTarget:
        """Return the exact content-free branch authorization target."""
        return InteractionScopeAuthorizationTarget(
            run_id=self.run_id,
            principal=self.actor.principal,
            branch_id=self.branch_id,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRoot:
    """Expose only one authoritative branch-to-root identity mapping."""

    run_id: RunId
    branch_id: BranchId
    root_branch_id: BranchId

    def __post_init__(self) -> None:
        for field_name, constructor in (
            ("run_id", RunId),
            ("branch_id", BranchId),
            ("root_branch_id", BranchId),
        ):
            object.__setattr__(
                self,
                field_name,
                constructor(
                    validate_opaque_id(
                        getattr(self, field_name),
                        f"branch_root.{field_name}",
                    )
                ),
            )
        if self.branch_id == self.root_branch_id:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "branch_root.root_branch_id",
                "a persisted child branch must have a distinct root",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRecord:
    """Persist one explicit branch edge and its metadata revision."""

    registration: InteractionBranchRegistration
    store_revision: InteractionStoreRevision

    def __post_init__(self) -> None:
        if not isinstance(self.registration, InteractionBranchRegistration):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "branch_record.registration",
                "value must be an interaction branch registration",
            )
        object.__setattr__(
            self,
            "store_revision",
            InteractionStoreRevision(
                validate_state_revision(
                    self.store_revision,
                    "branch_record.store_revision",
                )
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolutionIdempotencyEntry:
    """Bind one accepted transport key to canonical semantic content."""

    key: ResolutionIdempotencyKey
    resolution_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            validate_resolution_idempotency_key(
                self.key,
                "idempotency.key",
            ),
        )
        _validate_sha256(
            self.resolution_digest,
            "idempotency.resolution_digest",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AdvisoryWaitState:
    """Persist advisory timing outside semantic request state."""

    status: AdvisoryWaitStatus
    budget_seconds: int
    remaining_seconds: float
    presented_at: datetime | None = None
    running_since_monotonic: float | None = None
    controller_id: ControllerId | None = None
    lease_nonce: ActiveControlLeaseNonce | None = None
    activity_sequence: int | None = None
    lease_expires_at_monotonic: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AdvisoryWaitStatus):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "advisory.status",
                "value must be an advisory wait status",
            )
        budget = validate_int(
            self.budget_seconds,
            "advisory.budget_seconds",
            minimum=1,
            maximum=3_600,
        )
        object.__setattr__(self, "budget_seconds", budget)
        remaining = float(
            validate_finite_number(
                self.remaining_seconds,
                "advisory.remaining_seconds",
            )
        )
        if remaining < 0 or remaining > budget:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "advisory.remaining_seconds",
                "remaining advisory budget is out of bounds",
            )
        object.__setattr__(self, "remaining_seconds", remaining)
        if self.presented_at is not None:
            object.__setattr__(
                self,
                "presented_at",
                validate_aware_datetime(
                    self.presented_at,
                    "advisory.presented_at",
                ),
            )
        for field_name in (
            "running_since_monotonic",
            "lease_expires_at_monotonic",
        ):
            value = getattr(self, field_name)
            if value is not None:
                normalized = float(
                    validate_finite_number(value, f"advisory.{field_name}")
                )
                if normalized < 0:
                    raise InputValidationError(
                        InputErrorCode.OUT_OF_BOUNDS,
                        f"advisory.{field_name}",
                        "monotonic time must be non-negative",
                    )
                object.__setattr__(self, field_name, normalized)
        if self.activity_sequence is not None:
            object.__setattr__(
                self,
                "activity_sequence",
                validate_int(
                    self.activity_sequence,
                    "advisory.activity_sequence",
                    minimum=0,
                    maximum=MAX_STATE_REVISION,
                ),
            )
        if self.controller_id is not None:
            object.__setattr__(
                self,
                "controller_id",
                ControllerId(
                    validate_opaque_id(
                        self.controller_id,
                        "advisory.controller_id",
                    )
                ),
            )
        if self.lease_nonce is not None:
            object.__setattr__(
                self,
                "lease_nonce",
                ActiveControlLeaseNonce(
                    validate_opaque_id(
                        self.lease_nonce,
                        "advisory.lease_nonce",
                    )
                ),
            )
        _validate_advisory_wait_state(self)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionRecord:
    """Store one authoritative request and its operational metadata."""

    request: InputRequest
    semantic_fingerprint: str
    absolute_expires_at: datetime
    presentation: InteractionPresentationState
    store_revision: InteractionStoreRevision
    advisory_wait: AdvisoryWaitState | None = None
    resolution_digest: str | None = None
    idempotency_ledger: tuple[ResolutionIdempotencyEntry, ...] = ()
    resolved_by: InteractionResolutionAuthority | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "record.request",
                "value must be an input request",
            )
        _validate_sha256(
            self.semantic_fingerprint,
            "record.semantic_fingerprint",
        )
        if self.semantic_fingerprint != semantic_request_fingerprint(
            self.request
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "record.semantic_fingerprint",
                "fingerprint does not match the canonical request",
            )
        expires_at = validate_aware_datetime(
            self.absolute_expires_at,
            "record.absolute_expires_at",
        )
        object.__setattr__(self, "absolute_expires_at", expires_at)
        expected_expiry = self.request.created_at + timedelta(
            seconds=self.request.continuation_ttl_seconds
        )
        if expires_at != expected_expiry:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "record.absolute_expires_at",
                "absolute expiry must use the immutable continuation TTL",
            )
        if not isinstance(self.presentation, InteractionPresentationState):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "record.presentation",
                "value must be an interaction presentation state",
            )
        if self.request.state is RequestState.CREATED:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "record.request.state",
                "store records must use the authoritative admission path",
            )
        object.__setattr__(
            self,
            "store_revision",
            InteractionStoreRevision(
                validate_state_revision(
                    self.store_revision,
                    "record.store_revision",
                )
            ),
        )
        _validate_record_advisory(self)
        _validate_record_resolution(self)
        _validate_idempotency_ledger(self)

    @property
    def correlation(self) -> InteractionCorrelation:
        """Return complete logical correlation for this record."""
        return InteractionCorrelation.from_request(self.request)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionTerminalMetadata:
    """Expose only a terminal resolution category and trusted time."""

    status: ResolutionStatus
    resolved_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResolutionStatus):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "metadata.status",
                "value must be a resolution status",
            )
        object.__setattr__(
            self,
            "resolved_at",
            validate_aware_datetime(self.resolved_at, "metadata.resolved_at"),
        )


InteractionDisclosureProjection: TypeAlias = (
    InteractionRecord | InteractionTerminalMetadata
)


def project_authorized_interaction(
    record: InteractionRecord,
    decision: InteractionAuthorizationDecision,
) -> InteractionDisclosureProjection:
    """Project one record through its exact authorization-bound disclosure."""
    _validate_record(record, "record")
    if not isinstance(decision, InteractionAuthorizationDecision):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "authorization",
            "value must be an interaction authorization decision",
        )
    if not isinstance(
        decision.target,
        InteractionRequestAuthorizationTarget,
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "authorization.target",
            "record disclosure requires a request authorization target",
        )
    if decision.operation not in {
        InteractionOperation.INSPECT,
        InteractionOperation.LIST,
        InteractionOperation.WAIT,
    }:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "authorization.operation",
            "operation does not permit record disclosure",
        )
    if (
        decision.target.request_id != record.request.request_id
        or decision.target.origin != record.request.origin
    ):
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "authorization.target",
            "authorization target does not match the interaction record",
        )
    if (
        not decision.allowed
        or decision.disclosure is InteractionDisclosure.NONE
    ):
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "authorization.disclosure",
            "authorization does not permit interaction disclosure",
        )
    if (
        decision.disclosure is InteractionDisclosure.FULL
        and decision.actor.principal == record.request.origin.principal
    ):
        return record
    resolution = record.request.resolution
    if resolution is None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "authorization.disclosure",
            "terminal metadata requires a terminal interaction",
        )
    return InteractionTerminalMetadata(
        status=resolution.status,
        resolved_at=resolution.resolved_at,
    )


def evaluate_resolution_idempotency(
    record: InteractionRecord,
    *,
    key: ResolutionIdempotencyKey,
    resolution_digest: str,
    maximum_keys: int = MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
) -> ResolutionIdempotencyDisposition:
    """Return the bounded transport-key decision before lifecycle CAS."""
    _validate_record(record, "record")
    normalized_key = validate_resolution_idempotency_key(key)
    normalized_digest = _validate_sha256(
        resolution_digest,
        "resolution_digest",
    )
    limit = validate_int(
        maximum_keys,
        "maximum_keys",
        minimum=1,
        maximum=MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
    )
    for entry in record.idempotency_ledger:
        if entry.key == normalized_key:
            if entry.resolution_digest == normalized_digest:
                return ResolutionIdempotencyDisposition.SAME_KEY
            return ResolutionIdempotencyDisposition.SAME_KEY_CONFLICT
    if record.resolution_digest is None:
        return ResolutionIdempotencyDisposition.NEW_RESOLUTION
    if not _is_candidate_resolution_record(record):
        return ResolutionIdempotencyDisposition.TERMINAL_CONFLICT
    if record.resolution_digest != normalized_digest:
        return ResolutionIdempotencyDisposition.TERMINAL_CONFLICT
    if len(record.idempotency_ledger) >= limit:
        return ResolutionIdempotencyDisposition.LEDGER_FULL
    return ResolutionIdempotencyDisposition.SEMANTIC_NEW_KEY


def validate_interaction_admission(
    snapshot_records: tuple[InteractionRecord, ...],
    command: "CreateInteractionCommand",
    policy: InteractionPolicy,
) -> None:
    """Validate capacity from a complete locked lifetime snapshot."""
    if not isinstance(snapshot_records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "snapshot_records",
            "admission snapshot records must be a tuple",
        )
    if not isinstance(command, CreateInteractionCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a create interaction command",
        )
    _validate_policy(policy)
    request_ids: set[InputRequestId] = set()
    for record in snapshot_records:
        _validate_record(record, "snapshot_records")
        if record.request.request_id in request_ids:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "snapshot_records",
                "admission snapshot request identifiers must be unique",
            )
        request_ids.add(record.request.request_id)
    if command.request.request_id in request_ids:
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            "command.request.request_id",
            "request identifier already exists",
        )
    origin = command.request.origin
    unresolved = tuple(
        record
        for record in snapshot_records
        if record.request.state is RequestState.PENDING
        and record.request.resolution is None
    )
    if len(unresolved) >= policy.maximum_pending_interactions_per_process:
        raise InputValidationError(
            InputErrorCode.CAPACITY_EXCEEDED,
            "snapshot_records",
            "process pending-interaction capacity is exhausted",
        )
    unresolved_run = tuple(
        record
        for record in unresolved
        if record.request.origin.run_id == origin.run_id
        and record.request.origin.principal == origin.principal
    )
    if len(unresolved_run) >= policy.maximum_unresolved_interactions_per_run:
        raise InputValidationError(
            InputErrorCode.CAPACITY_EXCEEDED,
            "command.request.origin.run_id",
            "run unresolved-interaction capacity is exhausted",
        )
    if command.request.mode is RequirementMode.REQUIRED:
        unresolved_required_branch = tuple(
            record
            for record in unresolved_run
            if record.request.mode is RequirementMode.REQUIRED
            and record.request.origin.branch_id == origin.branch_id
        )
        if len(unresolved_required_branch) >= (
            policy.maximum_unresolved_required_interactions_per_branch
        ):
            raise InputValidationError(
                InputErrorCode.CAPACITY_EXCEEDED,
                "command.request.origin.branch_id",
                "branch unresolved-required capacity is exhausted",
            )
    fingerprint = semantic_request_fingerprint(command.request)
    equivalent_lifetime = tuple(
        record
        for record in snapshot_records
        if record.request.origin.run_id == origin.run_id
        and record.request.origin.principal == origin.principal
        and record.request.origin.branch_id == origin.branch_id
        and record.semantic_fingerprint == fingerprint
    )
    if len(equivalent_lifetime) >= (
        policy.maximum_equivalent_interactions_per_branch
    ):
        raise InputValidationError(
            InputErrorCode.INTERACTION_LOOP_LIMIT,
            "command.request",
            "logical branch lifetime equivalent-request limit is exhausted",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CreateInteractionCommand:
    """Atomically persist and admit one canonical request for handling."""

    actor: InteractionActor
    request: InputRequest
    resumer: InputResumer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        if type(self.request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "create.request",
                "value must be an input request",
            )
        if self.request.state is not RequestState.CREATED:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "create.request.state",
                "new store requests must begin in created state",
            )
        if self.resumer is not None:
            InputResumerRegistration(
                continuation_id=self.request.continuation_id,
                resumer=self.resumer,
            )

    @property
    def resumer_registration(self) -> InputResumerRegistration | None:
        """Return the atomic non-persisted resumer binding, when present."""
        if self.resumer is None:
            return None
        return InputResumerRegistration(
            continuation_id=self.request.continuation_id,
            resumer=self.resumer,
        )


@final
class _InteractionAdmissionCapability:
    """Identify one broker-bound admission without exposing request data."""

    __slots__ = ("_seal",)

    def __init__(self, *, _token: object) -> None:
        if _token is not _ADMISSION_CAPABILITY_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "admission.capability",
                "admission capabilities are broker-internal",
            )
        self._seal = _token


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionAdmissionCreateCommand:
    """Carry one sealed create command and its cleanup capability."""

    _command: CreateInteractionCommand = field(repr=False)
    _capability: _InteractionAdmissionCapability = field(repr=False)
    _seal: object = field(repr=False)

    def __init__(
        self,
        *,
        command: CreateInteractionCommand,
        capability: _InteractionAdmissionCapability,
        _token: object,
    ) -> None:
        if _token is not _ADMISSION_COMMAND_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "admission.create",
                "bound admission commands are broker-internal",
            )
        object.__setattr__(self, "_command", command)
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _token)


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionAdmissionCleanupCommand:
    """Carry only the opaque authority for one admitted bridge."""

    _capability: _InteractionAdmissionCapability = field(repr=False)
    _seal: object = field(repr=False)

    def __init__(
        self,
        *,
        capability: _InteractionAdmissionCapability,
        _token: object,
    ) -> None:
        if _token is not _ADMISSION_COMMAND_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "admission.cleanup",
                "admission cleanup commands are broker-internal",
            )
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_seal", _token)


class _InteractionAdmissionCleanupDisposition(StrEnum):
    """Identify a content-free admission-cleanup conclusion."""

    ABSENT = "absent"
    TERMINAL = "terminal"
    SETTLED = "settled"


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionAdmissionCleanupResult:
    """Prove cleanup without disclosing the bound interaction."""

    disposition: _InteractionAdmissionCleanupDisposition
    _seal: object = field(repr=False)

    def __init__(
        self,
        *,
        disposition: _InteractionAdmissionCleanupDisposition,
        _token: object,
    ) -> None:
        if _token is not _ADMISSION_CLEANUP_RESULT_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "admission.cleanup_result",
                "admission cleanup results are store-internal",
            )
        if not isinstance(
            disposition,
            _InteractionAdmissionCleanupDisposition,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "admission.cleanup_result.disposition",
                "value must be an admission cleanup disposition",
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "_seal", _token)


def _validate_interaction_admission_capability(
    capability: object,
    path: str,
) -> _InteractionAdmissionCapability:
    """Return one exact constructor-sealed admission capability."""
    if type(capability) is not _InteractionAdmissionCapability:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be an admission capability",
        )
    assert isinstance(capability, _InteractionAdmissionCapability)
    if getattr(capability, "_seal", None) is not _ADMISSION_CAPABILITY_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            path,
            "admission capability seal is invalid",
        )
    return capability


def _validate_interaction_admission_create_command(
    command: object,
) -> _InteractionAdmissionCreateCommand:
    """Return one exact constructor-sealed admission create command."""
    if type(command) is not _InteractionAdmissionCreateCommand:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "admission.create",
            "value must be a sealed admission create command",
        )
    assert isinstance(command, _InteractionAdmissionCreateCommand)
    if getattr(command, "_seal", None) is not _ADMISSION_COMMAND_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "admission.create",
            "admission create command seal is invalid",
        )
    _validate_interaction_admission_capability(
        getattr(command, "_capability", None),
        "admission.create.capability",
    )
    nested_command = getattr(command, "_command", None)
    if type(nested_command) is not CreateInteractionCommand:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "admission.create.command",
            "value must be a create interaction command",
        )
    assert isinstance(nested_command, CreateInteractionCommand)
    if nested_command.resumer is None:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "admission.create.command.resumer",
            "value must bind an input resumer",
        )
    return command


def _validate_interaction_admission_cleanup_command(
    command: object,
) -> _InteractionAdmissionCleanupCommand:
    """Return one exact constructor-sealed admission cleanup command."""
    if type(command) is not _InteractionAdmissionCleanupCommand:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "admission.cleanup",
            "value must be a sealed admission cleanup command",
        )
    assert isinstance(command, _InteractionAdmissionCleanupCommand)
    if getattr(command, "_seal", None) is not _ADMISSION_COMMAND_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "admission.cleanup",
            "admission cleanup command seal is invalid",
        )
    _validate_interaction_admission_capability(
        getattr(command, "_capability", None),
        "admission.cleanup.capability",
    )
    return command


def _validate_interaction_admission_cleanup_result(
    result: object,
    *,
    path: str = "admission.cleanup_result",
) -> _InteractionAdmissionCleanupResult:
    """Return one exact constructor-sealed admission cleanup proof."""
    if type(result) is not _InteractionAdmissionCleanupResult:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a sealed admission cleanup proof",
        )
    assert isinstance(result, _InteractionAdmissionCleanupResult)
    if getattr(result, "_seal", None) is not _ADMISSION_CLEANUP_RESULT_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            path,
            "admission cleanup proof seal is invalid",
        )
    if type(getattr(result, "disposition", None)) is not (
        _InteractionAdmissionCleanupDisposition
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            f"{path}.disposition",
            "value must be an admission cleanup disposition",
        )
    return result


def _new_interaction_admission_commands(
    *,
    actor: InteractionActor,
    request: InputRequest,
    resumer: InputResumer,
) -> tuple[
    _InteractionAdmissionCreateCommand,
    _InteractionAdmissionCleanupCommand,
]:
    """Mint one inseparable broker admission and cleanup capability."""
    command = CreateInteractionCommand(
        actor=actor,
        request=request,
        resumer=resumer,
    )
    capability = _InteractionAdmissionCapability(
        _token=_ADMISSION_CAPABILITY_TOKEN
    )
    return (
        _InteractionAdmissionCreateCommand(
            command=command,
            capability=capability,
            _token=_ADMISSION_COMMAND_TOKEN,
        ),
        _InteractionAdmissionCleanupCommand(
            capability=capability,
            _token=_ADMISSION_COMMAND_TOKEN,
        ),
    )


def _new_interaction_admission_cleanup_result(
    disposition: _InteractionAdmissionCleanupDisposition,
) -> _InteractionAdmissionCleanupResult:
    """Mint one sealed content-free cleanup proof for a concrete store."""
    return _InteractionAdmissionCleanupResult(
        disposition=disposition,
        _token=_ADMISSION_CLEANUP_RESULT_TOKEN,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ScopedInteractionLookup:
    """Request one scope-filtered record without an existence probe."""

    actor: InteractionActor
    correlation: InteractionCorrelation

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ListInteractionsCommand:
    """List records within one authenticated execution scope."""

    actor: InteractionActor
    scope: InteractionExecutionScope

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        if not isinstance(self.scope, InteractionExecutionScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "list.scope",
                "value must be an interaction execution scope",
            )

    @property
    def authorization_target(self) -> InteractionScopeAuthorizationTarget:
        """Return the exact scope target, including for an empty result."""
        return _scope_authorization_target(
            self.scope,
            self.actor.principal,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PresentInteractionCommand:
    """Record actual attached presentation and start advisory inactivity."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_store_revision: InteractionStoreRevision

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "expected_store_revision",
            InteractionStoreRevision(
                validate_state_revision(
                    self.expected_store_revision,
                    "present.expected_store_revision",
                )
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DetachInteractionCommand:
    """Record detached handling through the shared store interface."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_store_revision: InteractionStoreRevision

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "expected_store_revision",
            InteractionStoreRevision(
                validate_state_revision(
                    self.expected_store_revision,
                    "detach.expected_store_revision",
                )
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolveInteractionCommand:
    """Submit one candidate resolution for atomic deadline and CAS handling."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_state_revision: StateRevision
    idempotency_key: ResolutionIdempotencyKey
    proposed_resolution: InputCandidateResolution

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "expected_state_revision",
            StateRevision(
                validate_state_revision(
                    self.expected_state_revision,
                    "resolve.expected_state_revision",
                )
            ),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            validate_resolution_idempotency_key(
                self.idempotency_key,
                "resolve.idempotency_key",
            ),
        )
        if not _is_input_candidate_resolution(self.proposed_resolution):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolve.proposed_resolution",
                "external resolution may only answer or decline",
            )
        if self.proposed_resolution.request_id != self.correlation.request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "resolve.proposed_resolution.request_id",
                "resolution does not match request correlation",
            )
        if self.proposed_resolution.provenance in {
            AnswerProvenance.TRUSTED_DEFAULT,
            AnswerProvenance.POLICY,
        }:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "resolve.proposed_resolution.provenance",
                "external resolution cannot claim trusted provenance",
            )
        if isinstance(self.proposed_resolution, AnsweredResolution) and any(
            answer.provenance
            in {
                AnswerProvenance.TRUSTED_DEFAULT,
                AnswerProvenance.POLICY,
            }
            for answer in self.proposed_resolution.answers
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "resolve.proposed_resolution.answers",
                "external answers cannot claim trusted provenance",
            )

    @property
    def resolution_digest(self) -> str:
        """Return semantic content independent from trusted commit time."""
        return canonical_resolution_digest(self.proposed_resolution)


_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class _TrustedPolicyResolutionCommand:
    """Carry one sealed trusted-host policy resolution."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_state_revision: StateRevision
    idempotency_key: ResolutionIdempotencyKey
    proposed_resolution: InputCandidateResolution
    _authority: object = field(repr=False)

    def __init__(
        self,
        *,
        actor: InteractionActor,
        correlation: InteractionCorrelation,
        expected_state_revision: StateRevision,
        idempotency_key: ResolutionIdempotencyKey,
        proposed_resolution: InputCandidateResolution,
        _token: object,
    ) -> None:
        if _token is not _TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "policy_resolution.authority",
                "policy commands must be minted by the trusted host",
            )
        _validate_actor(actor)
        _validate_correlation(correlation)
        revision = StateRevision(
            validate_state_revision(
                expected_state_revision,
                "policy_resolution.expected_state_revision",
            )
        )
        key = validate_resolution_idempotency_key(
            idempotency_key,
            "policy_resolution.idempotency_key",
        )
        if not _is_input_candidate_resolution(proposed_resolution):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "policy_resolution.proposed_resolution",
                "policy resolution may only answer or decline",
            )
        if proposed_resolution.request_id != correlation.request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "policy_resolution.proposed_resolution.request_id",
                "resolution does not match request correlation",
            )
        if proposed_resolution.provenance is not AnswerProvenance.POLICY:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "policy_resolution.proposed_resolution.provenance",
                "policy resolution requires policy provenance",
            )
        if isinstance(proposed_resolution, AnsweredResolution) and any(
            answer.provenance is not AnswerProvenance.POLICY
            for answer in proposed_resolution.answers
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "policy_resolution.proposed_resolution.answers",
                "policy answers require policy provenance",
            )
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "expected_state_revision", revision)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(
            self,
            "proposed_resolution",
            proposed_resolution,
        )
        object.__setattr__(self, "_authority", _token)

    @property
    def resolution_digest(self) -> str:
        """Return semantic content independent from trusted commit time."""
        return canonical_resolution_digest(self.proposed_resolution)


_CandidateResolutionCommand: TypeAlias = (
    ResolveInteractionCommand | _TrustedPolicyResolutionCommand
)


def _new_trusted_policy_resolution_command(
    *,
    actor: InteractionActor,
    correlation: InteractionCorrelation,
    expected_state_revision: StateRevision,
    idempotency_key: ResolutionIdempotencyKey,
    proposed_resolution: InputCandidateResolution,
) -> _TrustedPolicyResolutionCommand:
    """Mint one sealed store command at the trusted broker boundary."""
    return _TrustedPolicyResolutionCommand(
        actor=actor,
        correlation=correlation,
        expected_state_revision=expected_state_revision,
        idempotency_key=idempotency_key,
        proposed_resolution=proposed_resolution,
        _token=_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
    )


def _validate_trusted_policy_resolution_command(
    command: object,
) -> _TrustedPolicyResolutionCommand:
    """Return one exactly sealed trusted-policy store command."""
    if type(command) is not _TrustedPolicyResolutionCommand:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a trusted policy resolution command",
        )
    assert isinstance(command, _TrustedPolicyResolutionCommand)
    if command._authority is not _TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "policy_resolution.authority",
            "policy command authority is invalid",
        )
    return command


def _validate_candidate_resolution_command(
    command: object,
) -> _CandidateResolutionCommand:
    """Return one valid external or sealed policy candidate command."""
    if isinstance(command, ResolveInteractionCommand):
        return command
    return _validate_trusted_policy_resolution_command(command)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedDefaultResolutionRequest:
    """Request trusted default derivation as one authenticated actor."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_state_revision: StateRevision

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "expected_state_revision",
            StateRevision(
                validate_state_revision(
                    self.expected_state_revision,
                    "trusted_default.expected_state_revision",
                )
            ),
        )


_TRUSTED_DEFAULT_COMMAND_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class TrustedDefaultResolutionCommand:
    """Carry one sealed trusted-host default-resolution authority."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    expected_state_revision: StateRevision
    _authority: object = field(repr=False)

    def __init__(
        self,
        *,
        request: TrustedDefaultResolutionRequest,
        _token: object,
    ) -> None:
        if _token is not _TRUSTED_DEFAULT_COMMAND_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "trusted_default.authority",
                "trusted-default commands must be minted by the trusted host",
            )
        if type(request) is not TrustedDefaultResolutionRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "trusted_default.request",
                "value must be a trusted-default resolution request",
            )
        object.__setattr__(self, "actor", request.actor)
        object.__setattr__(self, "correlation", request.correlation)
        object.__setattr__(
            self,
            "expected_state_revision",
            request.expected_state_revision,
        )
        object.__setattr__(self, "_authority", _token)


def _new_trusted_default_resolution_command(
    request: TrustedDefaultResolutionRequest,
) -> TrustedDefaultResolutionCommand:
    """Mint one store command at the trusted broker boundary."""
    return TrustedDefaultResolutionCommand(
        request=request,
        _token=_TRUSTED_DEFAULT_COMMAND_TOKEN,
    )


def _validate_trusted_default_resolution_command(
    command: object,
) -> TrustedDefaultResolutionCommand:
    """Return one exactly sealed trusted-default store command."""
    if type(command) is not TrustedDefaultResolutionCommand:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a trusted-default resolution command",
        )
    assert isinstance(command, TrustedDefaultResolutionCommand)
    if command._authority is not _TRUSTED_DEFAULT_COMMAND_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "trusted_default.authority",
            "trusted-default command authority is invalid",
        )
    return command


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalizeInteractionCommand:
    """Request unavailable or superseded terminalization."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    status: Literal[
        ResolutionStatus.UNAVAILABLE,
        ResolutionStatus.SUPERSEDED,
    ]
    provenance: PrincipalAuthoredProvenance
    expected_state_revision: StateRevision | None = None

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        _validate_terminalization_fields(
            self.status,
            self.provenance,
        )
        _validate_optional_revision(
            self,
            self.expected_state_revision,
            "terminalize.expected_state_revision",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInteractionCommand:
    """Cancel one direct request with request-local scope."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    provenance: PrincipalAuthoredProvenance
    expected_state_revision: StateRevision | None = None
    status: Literal[ResolutionStatus.CANCELLED] = field(
        init=False,
        default=ResolutionStatus.CANCELLED,
    )
    cancellation_scope: Literal[CancellationScope.REQUEST] = field(
        init=False,
        default=CancellationScope.REQUEST,
    )

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        _validate_principal_authored_provenance(
            self.provenance,
            "cancel.provenance",
        )
        _validate_optional_revision(
            self,
            self.expected_state_revision,
            "cancel.expected_state_revision",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalizeInteractionScopeCommand:
    """Cancel one containing run and its selected execution subtree."""

    actor: InteractionActor
    scope: InteractionExecutionScope
    provenance: PrincipalAuthoredProvenance
    status: Literal[ResolutionStatus.CANCELLED] = field(
        init=False,
        default=ResolutionStatus.CANCELLED,
    )
    cancellation_scope: Literal[CancellationScope.CONTAINING_RUN] = field(
        init=False,
        default=CancellationScope.CONTAINING_RUN,
    )

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        if not isinstance(self.scope, InteractionExecutionScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "terminalize_scope.scope",
                "value must be an interaction execution scope",
            )
        _validate_principal_authored_provenance(
            self.provenance,
            "terminalize_scope.provenance",
        )

    @property
    def authorization_target(self) -> InteractionScopeAuthorizationTarget:
        """Return the exact scope target, even when no records exist."""
        return _scope_authorization_target(
            self.scope,
            self.actor.principal,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class SupersedeInteractionScopeCommand:
    """Supersede one execution subtree atomically."""

    actor: InteractionActor
    scope: InteractionExecutionScope
    provenance: PrincipalAuthoredProvenance
    status: Literal[ResolutionStatus.SUPERSEDED] = field(
        init=False,
        default=ResolutionStatus.SUPERSEDED,
    )

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        if not isinstance(self.scope, InteractionExecutionScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "supersede_scope.scope",
                "value must be an interaction execution scope",
            )
        _validate_principal_authored_provenance(
            self.provenance,
            "supersede_scope.provenance",
        )

    @property
    def authorization_target(self) -> InteractionScopeAuthorizationTarget:
        """Return the exact scope target, even when no records exist."""
        return _scope_authorization_target(
            self.scope,
            self.actor.principal,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RecordControllerActivityCommand:
    """Record authenticated sequenced activity without resetting its budget."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    evidence: ControllerActivityEvidence

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        if not is_controller_activity_evidence(self.evidence):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "activity.evidence",
                "value must be controller activity evidence",
            )
        if self.evidence.request_id != self.correlation.request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "activity.request_id",
                "activity does not match request correlation",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WaitForInteractionChangeCommand:
    """Wait for a newer store revision without a registration race."""

    actor: InteractionActor
    correlation: InteractionCorrelation
    after_store_revision: InteractionStoreRevision

    def __post_init__(self) -> None:
        _validate_actor(self.actor)
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "after_store_revision",
            InteractionStoreRevision(
                validate_state_revision(
                    self.after_store_revision,
                    "wait.after_store_revision",
                )
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionDeadline:
    """Identify the next monotonic deadline for one request."""

    request_id: InputRequestId
    monotonic_deadline: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(
                    self.request_id,
                    "deadline.request_id",
                )
            ),
        )
        value = float(
            validate_finite_number(
                self.monotonic_deadline,
                "deadline.monotonic_deadline",
            )
        )
        if value < 0:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "deadline.monotonic_deadline",
                "monotonic deadline must be non-negative",
            )
        object.__setattr__(self, "monotonic_deadline", value)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionDeadlineSnapshot:
    """Expose the earliest deadline and its change-notification generation."""

    schedule_revision: DeadlineScheduleRevision
    deadline: InteractionDeadline | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schedule_revision",
            DeadlineScheduleRevision(
                validate_state_revision(
                    self.schedule_revision,
                    "deadline.schedule_revision",
                )
            ),
        )
        if self.deadline is not None and not isinstance(
            self.deadline,
            InteractionDeadline,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "deadline.deadline",
                "value must be an interaction deadline",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class WaitForDeadlineChangeCommand:
    """Wait until the deadline schedule generation changes."""

    after_schedule_revision: DeadlineScheduleRevision

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "after_schedule_revision",
            DeadlineScheduleRevision(
                validate_state_revision(
                    self.after_schedule_revision,
                    "deadline.after_schedule_revision",
                )
            ),
        )


def select_next_interaction_deadline(
    records: tuple[InteractionRecord, ...],
    observed_at: InteractionTime,
    schedule_revision: DeadlineScheduleRevision,
) -> InteractionDeadlineSnapshot:
    """Return the deterministic next deadline from one locked snapshot."""
    if not isinstance(records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "records",
            "deadline candidates must be a tuple",
        )
    if type(observed_at) is not InteractionTime:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "observed_at",
            "value must be a sealed trusted clock observation",
        )
    revision = DeadlineScheduleRevision(
        validate_state_revision(
            schedule_revision,
            "deadline.schedule_revision",
        )
    )
    request_ids: set[InputRequestId] = set()
    deadlines: list[InteractionDeadline] = []
    for record in records:
        _validate_record(record, "records")
        request_id = record.request.request_id
        if request_id in request_ids:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "records",
                "deadline candidate request identifiers must be unique",
            )
        request_ids.add(request_id)
        if (
            record.request.state is not RequestState.PENDING
            or record.request.resolution is not None
        ):
            continue
        deadlines.append(
            InteractionDeadline(
                request_id=request_id,
                monotonic_deadline=_effective_interaction_deadline(
                    record,
                    observed_at,
                ),
            )
        )
    deadline = min(
        deadlines,
        key=lambda item: (item.monotonic_deadline, str(item.request_id)),
        default=None,
    )
    return InteractionDeadlineSnapshot(
        schedule_revision=revision,
        deadline=deadline,
    )


def _effective_interaction_deadline(
    record: InteractionRecord,
    observed_at: InteractionTime,
) -> float:
    wall_remaining = (
        record.absolute_expires_at - observed_at.wall_time
    ).total_seconds()
    deadline = observed_at.monotonic_seconds + max(0.0, wall_remaining)
    wait = record.advisory_wait
    if isinstance(wait, AdvisoryWaitState):
        if wait.status is AdvisoryWaitStatus.RUNNING:
            assert wait.running_since_monotonic is not None
            deadline = min(
                deadline,
                wait.running_since_monotonic + wait.remaining_seconds,
            )
        elif wait.status is AdvisoryWaitStatus.PAUSED:
            assert wait.lease_expires_at_monotonic is not None
            deadline = min(deadline, wait.lease_expires_at_monotonic)
    return deadline


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalizeDueInteractionsCommand:
    """Settle every deadline due at the store's trusted clock observation."""

    maximum_results: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_results",
            validate_int(
                self.maximum_results,
                "deadlines.maximum_results",
                minimum=1,
                maximum=1_024,
            ),
        )


_CLASSIFIER_BINDING_TOKEN = object()
_CLASSIFICATION_PROOF_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class _TaskInputClassifierBinding:
    """Hold one opaque store-owned classifier configuration capability."""

    classifier_id: str
    policy_revision: str
    _capability: object = field(repr=False)

    def __init__(
        self,
        *,
        classifier_id: str,
        policy_revision: str,
        _token: object,
    ) -> None:
        if _token is not _CLASSIFIER_BINDING_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "classifier_binding",
                "classifier bindings must be minted by the store broker",
            )
        object.__setattr__(
            self,
            "classifier_id",
            validate_opaque_id(
                classifier_id,
                "classifier_binding.classifier_id",
                maximum_characters=128,
                maximum_bytes=512,
            ),
        )
        object.__setattr__(
            self,
            "policy_revision",
            validate_opaque_id(
                policy_revision,
                "classifier_binding.policy_revision",
                maximum_characters=128,
                maximum_bytes=512,
            ),
        )
        object.__setattr__(self, "_capability", object())


@final
@dataclass(frozen=True, slots=True, init=False)
class _BoundTaskInputClassifications:
    """Seal untrusted classifier outputs to one exact candidate."""

    classifier_id: str
    policy_revision: str
    request_id: InputRequestId
    candidate_digest: str
    classifications: tuple[TaskInputClassification, ...] = field(repr=False)
    _binding_capability: object = field(repr=False)

    def __init__(
        self,
        *,
        binding: _TaskInputClassifierBinding,
        request_id: InputRequestId,
        candidate_digest: str,
        classifications: tuple[TaskInputClassification, ...],
        _token: object,
    ) -> None:
        if _token is not _CLASSIFICATION_PROOF_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "classification_proof",
                "classification proofs must be minted by the store broker",
            )
        object.__setattr__(self, "classifier_id", binding.classifier_id)
        object.__setattr__(self, "policy_revision", binding.policy_revision)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "candidate_digest", candidate_digest)
        object.__setattr__(self, "classifications", classifications)
        object.__setattr__(
            self,
            "_binding_capability",
            binding._capability,
        )


def _new_task_input_classifier_binding(
    *,
    classifier_id: str,
    policy_revision: str,
) -> _TaskInputClassifierBinding:
    """Create one opaque binding owned by a concrete store backing."""
    return _TaskInputClassifierBinding(
        classifier_id=classifier_id,
        policy_revision=policy_revision,
        _token=_CLASSIFIER_BINDING_TOKEN,
    )


InteractionPresentationCommand: TypeAlias = (
    PresentInteractionCommand | DetachInteractionCommand
)
InteractionDeadlineTriggerCommand: TypeAlias = (
    _CandidateResolutionCommand
    | TrustedDefaultResolutionCommand
    | PresentInteractionCommand
    | DetachInteractionCommand
    | RecordControllerActivityCommand
    | CancelInteractionCommand
    | TerminalizeInteractionCommand
)
InteractionLeaseExpiryTriggerCommand: TypeAlias = (
    _CandidateResolutionCommand
    | TrustedDefaultResolutionCommand
    | DetachInteractionCommand
    | RecordControllerActivityCommand
    | CancelInteractionCommand
    | TerminalizeInteractionCommand
)
InteractionExternalMutationCommand: TypeAlias = (
    CreateInteractionCommand
    | PresentInteractionCommand
    | DetachInteractionCommand
    | ResolveInteractionCommand
    | TerminalizeInteractionCommand
    | CancelInteractionCommand
    | TerminalizeInteractionScopeCommand
    | SupersedeInteractionScopeCommand
    | RegisterInteractionBranchCommand
    | RecordControllerActivityCommand
    | TerminalizeDueInteractionsCommand
)
InteractionScopeMutationCommand: TypeAlias = (
    TerminalizeInteractionScopeCommand | SupersedeInteractionScopeCommand
)
_InteractionBranchOwnershipKey: TypeAlias = tuple[
    RunId,
    PrincipalScope,
    BranchId,
]


_STORE_BACKING_TOKEN = object()
_PARTIAL_STORE_BACKING_TOKEN = object()
_STORE_BACKING_MUTATION_TOKEN = object()
_BRANCH_CLOSURE_ATTESTATION_TOKEN = object()
_SCOPE_OWNERSHIP_ATTESTATION_TOKEN = object()
_SCOPE_SELECTION_TOKEN = object()
_SCOPE_RESULT_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionBranchClosureAttestation:
    """Seal authoritative roots for one exact partial branch closure."""

    authoritative_branch_roots: frozenset[_InteractionBranchOwnershipKey]

    def __init__(
        self,
        authoritative_branch_roots: frozenset[_InteractionBranchOwnershipKey],
        *,
        _token: object,
    ) -> None:
        if _token is not _BRANCH_CLOSURE_ATTESTATION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "branch_closure_attestation",
                "branch closure attestations are store-internal",
            )
        object.__setattr__(
            self,
            "authoritative_branch_roots",
            authoritative_branch_roots,
        )


@final
@dataclass(frozen=True, slots=True, init=False)
class _InteractionScopeOwnershipAttestation:
    """Seal content-free ownership presence for one exact scope."""

    scope: InteractionExecutionScope
    principal: PrincipalScope
    actor_owned_record_match: bool
    foreign_owned_record_match: bool
    actor_owned_branch_match: bool
    foreign_owned_branch_match: bool

    def __init__(
        self,
        *,
        scope: InteractionExecutionScope,
        principal: PrincipalScope,
        actor_owned_record_match: bool,
        foreign_owned_record_match: bool,
        actor_owned_branch_match: bool,
        foreign_owned_branch_match: bool,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_OWNERSHIP_ATTESTATION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "scope_ownership_attestation",
                "scope ownership attestations are store-internal",
            )
        if not isinstance(scope, InteractionExecutionScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "scope_ownership_attestation.scope",
                "value must be an interaction execution scope",
            )
        if not isinstance(principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "scope_ownership_attestation.principal",
                "value must be a principal scope",
            )
        validate_bool(
            actor_owned_record_match,
            "scope_ownership_attestation.actor_owned_record_match",
        )
        validate_bool(
            foreign_owned_record_match,
            "scope_ownership_attestation.foreign_owned_record_match",
        )
        validate_bool(
            actor_owned_branch_match,
            "scope_ownership_attestation.actor_owned_branch_match",
        )
        validate_bool(
            foreign_owned_branch_match,
            "scope_ownership_attestation.foreign_owned_branch_match",
        )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "principal", principal)
        object.__setattr__(
            self,
            "actor_owned_record_match",
            actor_owned_record_match,
        )
        object.__setattr__(
            self,
            "foreign_owned_record_match",
            foreign_owned_record_match,
        )
        object.__setattr__(
            self,
            "actor_owned_branch_match",
            actor_owned_branch_match,
        )
        object.__setattr__(
            self,
            "foreign_owned_branch_match",
            foreign_owned_branch_match,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class _InteractionStoreBackingSnapshot:
    """Hold one immutable authoritative backing inventory generation."""

    records: tuple[InteractionRecord, ...] = field(repr=False)
    branch_records: tuple[InteractionBranchRecord, ...] = field(repr=False)
    store_generation: InteractionStoreGeneration
    branch_closure_attestation: _InteractionBranchClosureAttestation | None = (
        field(default=None, repr=False)
    )
    scope_ownership_attestation: (
        _InteractionScopeOwnershipAttestation | None
    ) = field(default=None, repr=False)


@final
@dataclass(frozen=True, slots=True, init=False, eq=False)
class _InteractionStoreBacking:
    """Own one shared concrete store inventory and opaque capability."""

    _capability: object = field(repr=False)
    _snapshot: _InteractionStoreBackingSnapshot = field(repr=False)

    def __init__(
        self,
        *,
        records: tuple[InteractionRecord, ...],
        branch_records: tuple[InteractionBranchRecord, ...],
        store_generation: InteractionStoreGeneration,
        scope_ownership_attestation: (
            _InteractionScopeOwnershipAttestation | None
        ) = None,
        _token: object,
    ) -> None:
        if (
            _token is not _STORE_BACKING_TOKEN
            and _token is not _PARTIAL_STORE_BACKING_TOKEN
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "backing",
                "store backing capabilities are broker-internal",
            )
        normalized_records, normalized_branches = _validate_scope_snapshot(
            records,
            branch_records,
        )
        branch_closure_attestation = (
            _mint_interaction_branch_closure_attestation(
                normalized_records,
                normalized_branches,
            )
            if _token is _PARTIAL_STORE_BACKING_TOKEN
            else None
        )
        if scope_ownership_attestation is not None:
            _validate_interaction_scope_ownership_attestation(
                scope_ownership_attestation
            )
        if (
            branch_closure_attestation is not None
            and scope_ownership_attestation is not None
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "backing",
                "one partial backing cannot combine task and scope "
                "attestations",
            )
        object.__setattr__(self, "_capability", object())
        object.__setattr__(
            self,
            "_snapshot",
            _InteractionStoreBackingSnapshot(
                records=normalized_records,
                branch_records=normalized_branches,
                store_generation=_validate_store_generation(store_generation),
                branch_closure_attestation=branch_closure_attestation,
                scope_ownership_attestation=scope_ownership_attestation,
            ),
        )

    def _advance(
        self,
        *,
        records: tuple[InteractionRecord, ...],
        branch_records: tuple[InteractionBranchRecord, ...],
        _token: object,
    ) -> _InteractionStoreBackingSnapshot:
        """Replace inventory after one exact centralized mutation."""
        if _token is not _STORE_BACKING_MUTATION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "backing",
                "store backing inventory mutations are broker-internal",
            )
        normalized_records, normalized_branches = _validate_scope_snapshot(
            records,
            branch_records,
        )
        branch_closure_attestation = self._snapshot.branch_closure_attestation
        scope_ownership_attestation = (
            self._snapshot.scope_ownership_attestation
        )
        if branch_closure_attestation is not None:
            _validate_interaction_branch_closure_attestation(
                branch_closure_attestation,
                normalized_records,
                normalized_branches,
            )
        next_generation = InteractionStoreGeneration(
            validate_state_revision(
                self._snapshot.store_generation + 1,
                "store_generation",
            )
        )
        snapshot = _InteractionStoreBackingSnapshot(
            records=normalized_records,
            branch_records=normalized_branches,
            store_generation=next_generation,
            branch_closure_attestation=branch_closure_attestation,
            scope_ownership_attestation=scope_ownership_attestation,
        )
        object.__setattr__(self, "_snapshot", snapshot)
        return snapshot


def _new_interaction_store_backing(
    *,
    records: tuple[InteractionRecord, ...] = (),
    branch_records: tuple[InteractionBranchRecord, ...] = (),
    store_generation: InteractionStoreGeneration = InteractionStoreGeneration(
        0
    ),
) -> _InteractionStoreBacking:
    """Create one authoritative inventory shared by concrete-store handles."""
    return _InteractionStoreBacking(
        records=records,
        branch_records=branch_records,
        store_generation=store_generation,
        scope_ownership_attestation=None,
        _token=_STORE_BACKING_TOKEN,
    )


def _new_partial_interaction_store_backing(
    *,
    records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    store_generation: InteractionStoreGeneration,
) -> _InteractionStoreBacking:
    """Create one sealed backing from an exact persisted branch closure."""
    return _InteractionStoreBacking(
        records=records,
        branch_records=branch_records,
        store_generation=store_generation,
        scope_ownership_attestation=None,
        _token=_PARTIAL_STORE_BACKING_TOKEN,
    )


def _new_scoped_interaction_store_backing(
    *,
    records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    store_generation: InteractionStoreGeneration,
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
    actor_owned_record_match: bool,
    foreign_owned_record_match: bool,
    actor_owned_branch_match: bool,
    foreign_owned_branch_match: bool,
) -> _InteractionStoreBacking:
    """Create one actor-only backing with exact global scope presence."""
    attestation = _InteractionScopeOwnershipAttestation(
        scope=scope,
        principal=principal,
        actor_owned_record_match=actor_owned_record_match,
        foreign_owned_record_match=foreign_owned_record_match,
        actor_owned_branch_match=actor_owned_branch_match,
        foreign_owned_branch_match=foreign_owned_branch_match,
        _token=_SCOPE_OWNERSHIP_ATTESTATION_TOKEN,
    )
    return _InteractionStoreBacking(
        records=records,
        branch_records=branch_records,
        store_generation=store_generation,
        scope_ownership_attestation=attestation,
        _token=_STORE_BACKING_TOKEN,
    )


def _snapshot_interaction_store_backing(
    backing: _InteractionStoreBacking,
) -> _InteractionStoreBackingSnapshot:
    """Return the backing's immutable full inventory under the store lock."""
    _validate_store_backing(backing)
    return backing._snapshot


def _insert_interaction_store_backing_record(
    backing: _InteractionStoreBacking,
    record: InteractionRecord,
) -> _InteractionStoreBackingSnapshot:
    """Insert one record into the authoritative backing inventory."""
    snapshot = _snapshot_interaction_store_backing(backing)
    _validate_record(record, "record")
    if any(
        current.request.request_id == record.request.request_id
        for current in snapshot.records
    ):
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            "record.request.request_id",
            "the backing already contains this request identifier",
        )
    return backing._advance(
        records=snapshot.records + (record,),
        branch_records=snapshot.branch_records,
        _token=_STORE_BACKING_MUTATION_TOKEN,
    )


def _replace_interaction_store_backing_records(
    backing: _InteractionStoreBacking,
    previous: tuple[InteractionRecord, ...],
    records: tuple[InteractionRecord, ...],
) -> _InteractionStoreBackingSnapshot:
    """CAS-replace exact records while preserving the full inventory."""
    snapshot = _snapshot_interaction_store_backing(backing)
    if not isinstance(previous, tuple) or not isinstance(records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "records",
            "backing record replacements must be tuples",
        )
    if len(previous) != len(records) or not previous:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "records",
            "backing record replacements must be non-empty and paired",
        )
    current_by_id = {
        item.request.request_id: item for item in snapshot.records
    }
    replacements: dict[InputRequestId, InteractionRecord] = {}
    for expected, replacement in zip(previous, records, strict=True):
        _validate_record(expected, "previous")
        _validate_record(replacement, "records")
        request_id = expected.request.request_id
        if replacement.request.request_id != request_id:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "records",
                "replacement request identifiers must remain unchanged",
            )
        if request_id in replacements:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "previous",
                "backing record replacements must be unique",
            )
        if current_by_id.get(request_id) != expected:
            raise InputValidationError(
                InputErrorCode.STALE_REVISION,
                "previous",
                "backing record replacement does not match current state",
            )
        replacements[request_id] = replacement
    return backing._advance(
        records=tuple(
            replacements.get(item.request.request_id, item)
            for item in snapshot.records
        ),
        branch_records=snapshot.branch_records,
        _token=_STORE_BACKING_MUTATION_TOKEN,
    )


def _insert_interaction_store_backing_branch_record(
    backing: _InteractionStoreBacking,
    branch_record: InteractionBranchRecord,
) -> _InteractionStoreBackingSnapshot:
    """Insert one branch edge into the authoritative backing inventory."""
    snapshot = _snapshot_interaction_store_backing(backing)
    _validate_branch_record(branch_record, "branch_record")
    registration = branch_record.registration
    key = _branch_identity_key(registration)
    if any(
        _branch_identity_key(item.registration) == key
        for item in snapshot.branch_records
    ):
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            "branch_record.registration.branch_id",
            "the backing already contains this run branch",
        )
    return backing._advance(
        records=snapshot.records,
        branch_records=snapshot.branch_records + (branch_record,),
        _token=_STORE_BACKING_MUTATION_TOKEN,
    )


def _replace_interaction_store_backing_branch_record(
    backing: _InteractionStoreBacking,
    previous: InteractionBranchRecord,
    branch_record: InteractionBranchRecord,
) -> _InteractionStoreBackingSnapshot:
    """CAS-replace one exact authoritative branch edge."""
    snapshot = _snapshot_interaction_store_backing(backing)
    _validate_branch_record(previous, "previous")
    _validate_branch_record(branch_record, "branch_record")
    previous_registration = previous.registration
    key = _branch_identity_key(previous_registration)
    registration = branch_record.registration
    if registration.principal != previous_registration.principal:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "branch_record.registration.principal",
            "replacement branch ownership must remain unchanged",
        )
    if registration != previous_registration:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "branch_record.registration",
            "replacement branch registration must remain unchanged",
        )
    found = False
    branch_records: list[InteractionBranchRecord] = []
    for current in snapshot.branch_records:
        current_registration = current.registration
        if _branch_identity_key(current_registration) != key:
            branch_records.append(current)
            continue
        if current != previous:
            raise InputValidationError(
                InputErrorCode.STALE_REVISION,
                "previous",
                "branch replacement does not match current state",
            )
        branch_records.append(branch_record)
        found = True
    if not found:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "previous",
            "branch replacement does not match current state",
        )
    return backing._advance(
        records=snapshot.records,
        branch_records=tuple(branch_records),
        _token=_STORE_BACKING_MUTATION_TOKEN,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class _InteractionScopeTransaction:
    """Seal one complete selection inside a concrete-store transaction."""

    command: InteractionScopeMutationCommand = field(repr=False)
    scope: InteractionExecutionScope = field(repr=False)
    principal: PrincipalScope = field(repr=False)
    store_generation: InteractionStoreGeneration = field(repr=False)
    snapshot_digest: str = field(repr=False)
    selected_records: tuple[InteractionRecord, ...] = field(repr=False)
    _backing_capability: object = field(repr=False)

    def __init__(
        self,
        *,
        command: InteractionScopeMutationCommand,
        scope: InteractionExecutionScope,
        principal: PrincipalScope,
        store_generation: InteractionStoreGeneration,
        snapshot_digest: str,
        selected_records: tuple[InteractionRecord, ...],
        backing: _InteractionStoreBacking,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_SELECTION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "selection",
                "scope transactions must be minted under a store lock",
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "principal", principal)
        object.__setattr__(self, "store_generation", store_generation)
        object.__setattr__(self, "snapshot_digest", snapshot_digest)
        object.__setattr__(self, "selected_records", selected_records)
        object.__setattr__(
            self,
            "_backing_capability",
            backing._capability,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CreateInteractionApplied:
    """Prove one exact create-and-admit command mutation."""

    command: CreateInteractionCommand
    record: InteractionRecord
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command, CreateInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.command",
                "value must be a create interaction command",
            )
        _validate_policy(self.policy)
        expected = _reduce_create_interaction(self.command, self.policy)
        if self.record != expected:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "result.record",
                "create result must exactly admit its command request",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInteractionApplied:
    """Prove one exact request-local cancellation command mutation."""

    command: CancelInteractionCommand
    previous: InteractionRecord
    record: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command, CancelInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.command",
                "value must be a cancel interaction command",
            )
        _validate_temporal_context(self.observed_at, self.policy)
        expected = _reduce_request_cancellation(
            self.previous,
            self.command,
            self.observed_at,
            self.policy,
        )
        _require_exact_result_record(self.record, expected, "cancellation")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalizeInteractionApplied:
    """Prove one exact unavailable or superseded request mutation."""

    command: TerminalizeInteractionCommand
    previous: InteractionRecord
    record: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command, TerminalizeInteractionCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.command",
                "value must be a terminalize interaction command",
            )
        _validate_temporal_context(self.observed_at, self.policy)
        expected = _reduce_request_terminalization(
            self.previous,
            self.command,
            self.observed_at,
            self.policy,
        )
        _require_exact_result_record(self.record, expected, "terminalization")


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ScopeCancellationApplied:
    """Expose only authorized records settled by one cancellation batch."""

    command: TerminalizeInteractionScopeCommand
    previous: tuple[InteractionRecord, ...]
    records: tuple[InteractionRecord, ...]
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __init__(
        self,
        *,
        command: TerminalizeInteractionScopeCommand,
        previous: tuple[InteractionRecord, ...],
        records: tuple[InteractionRecord, ...],
        observed_at: InteractionTime,
        policy: InteractionPolicy,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_RESULT_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result",
                "scope results must be minted inside a store transaction",
            )
        _validate_temporal_context(observed_at, policy)
        expected = _reduce_scope_cancellation(
            previous,
            command,
            observed_at,
            policy,
        )
        _require_exact_result_records(records, expected, "cancellation")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "previous", previous)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "store_mutation_applied", True)
        object.__setattr__(self, "kind", InteractionStoreResultKind.APPLIED)


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ScopeSupersessionApplied:
    """Expose only authorized records settled by one supersession batch."""

    command: SupersedeInteractionScopeCommand
    previous: tuple[InteractionRecord, ...]
    records: tuple[InteractionRecord, ...]
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __init__(
        self,
        *,
        command: SupersedeInteractionScopeCommand,
        previous: tuple[InteractionRecord, ...],
        records: tuple[InteractionRecord, ...],
        observed_at: InteractionTime,
        policy: InteractionPolicy,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_RESULT_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result",
                "scope results must be minted inside a store transaction",
            )
        _validate_temporal_context(observed_at, policy)
        expected = _reduce_scope_supersession(
            previous,
            command,
            observed_at,
            policy,
        )
        _require_exact_result_records(records, expected, "supersession")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "previous", previous)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "store_mutation_applied", True)
        object.__setattr__(self, "kind", InteractionStoreResultKind.APPLIED)


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ScopeCancellationReplayed:
    """Return a content-free empty-scope cancellation result."""

    command: TerminalizeInteractionScopeCommand
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REPLAYED] = field(
        init=False,
        default=InteractionStoreResultKind.REPLAYED,
    )

    def __init__(
        self,
        *,
        command: TerminalizeInteractionScopeCommand,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_RESULT_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result",
                "scope results must be minted inside a store transaction",
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "store_mutation_applied", False)
        object.__setattr__(self, "kind", InteractionStoreResultKind.REPLAYED)

    @property
    def records(self) -> tuple[InteractionRecord, ...]:
        """Return the exact empty no-op result tuple."""
        return ()


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ScopeSupersessionReplayed:
    """Return a content-free empty-scope supersession result."""

    command: SupersedeInteractionScopeCommand
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REPLAYED] = field(
        init=False,
        default=InteractionStoreResultKind.REPLAYED,
    )

    def __init__(
        self,
        *,
        command: SupersedeInteractionScopeCommand,
        _token: object,
    ) -> None:
        if _token is not _SCOPE_RESULT_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result",
                "scope results must be minted inside a store transaction",
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "store_mutation_applied", False)
        object.__setattr__(self, "kind", InteractionStoreResultKind.REPLAYED)

    @property
    def records(self) -> tuple[InteractionRecord, ...]:
        """Return the exact empty no-op result tuple."""
        return ()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionPresentationApplied:
    """Return one committed presentation-only metadata mutation."""

    command: InteractionPresentationCommand
    record: InteractionRecord
    previous: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        validate_interaction_presentation_transition(
            self.previous,
            self.record,
            self.command,
            self.observed_at,
            self.policy,
        )

    @property
    def presentation(self) -> InteractionPresentationState:
        """Return the presentation state selected by the exact command."""
        if isinstance(self.command, PresentInteractionCommand):
            return InteractionPresentationState.PRESENTED
        return InteractionPresentationState.DETACHED


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ControllerActivityApplied:
    """Return one committed active-control lease mutation."""

    command: RecordControllerActivityCommand
    record: InteractionRecord
    previous: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    lease_nonce: ActiveControlLeaseNonce | None = None
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command, RecordControllerActivityCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.command",
                "value must be a controller activity command",
            )
        evidence = self.command.evidence
        if type(evidence) is AcquireControllerActivity:
            nonce = ActiveControlLeaseNonce(
                validate_opaque_id(
                    self.lease_nonce,
                    "result.lease_nonce",
                )
            )
            object.__setattr__(self, "lease_nonce", nonce)
        elif self.lease_nonce is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "result.lease_nonce",
                "only lease acquisition returns a newly minted nonce",
            )
        validate_controller_activity_transition(
            self.previous,
            self.record,
            self.command,
            self.observed_at,
            self.policy,
            lease_nonce=self.lease_nonce,
        )

    @property
    def action(self) -> ControllerActivityAction:
        """Return the exact committed controller action."""
        return self.command.evidence.action

    @property
    def evidence(self) -> ControllerActivityEvidence:
        """Return the controller evidence bound by the exact command."""
        return self.command.evidence


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ControllerLeaseExpiredApplied:
    """Return the lease-expiry mutation that won before controller evidence."""

    record: InteractionRecord
    previous: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    command: InteractionLeaseExpiryTriggerCommand | None = None
    internal_authority: _InteractionSystemResolver | None = None
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        _validate_temporal_context(self.observed_at, self.policy)
        if self.command is None:
            if self.internal_authority is not _DEADLINE_RESOLVER:
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "result.internal_authority",
                    "commandless lease expiry requires sealed authority",
                )
        else:
            if self.internal_authority is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "result.internal_authority",
                    "external lease expiry cannot claim internal authority",
                )
            _validate_lease_expiry_trigger_command(
                self.previous,
                self.command,
            )
        expected = _reduce_expired_controller_lease(
            self.previous,
            self.observed_at,
            self.policy,
        )
        if expected is None or self.record != expected:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "result.record",
                "lease-expiry result must exactly resume the expired lease",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolveInteractionApplied:
    """Return a deadline or candidate resolution lifecycle commit."""

    record: InteractionRecord
    previous: InteractionRecord
    decision_stage: ResolutionDecisionStage
    observed_at: InteractionTime
    policy: InteractionPolicy
    command: InteractionDeadlineTriggerCommand | None = None
    idempotency_key: ResolutionIdempotencyKey | None = None
    classifier_binding: _TaskInputClassifierBinding | None = field(
        default=None,
        repr=False,
    )
    classification_proof: _BoundTaskInputClassifications | None = field(
        default=None,
        repr=False,
    )
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        if self.decision_stage is ResolutionDecisionStage.COMMIT:
            key = validate_resolution_idempotency_key(
                self.idempotency_key,
                "result.idempotency_key",
            )
            object.__setattr__(self, "idempotency_key", key)
        validate_resolution_commit_transition(
            self.previous,
            self.record,
            self.decision_stage,
            self.observed_at,
            self.policy,
            command=self.command,
            idempotency_key=self.idempotency_key,
            classifier_binding=self.classifier_binding,
            classification_proof=self.classification_proof,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedDefaultResolutionApplied:
    """Return one committed request-derived trusted-default resolution."""

    command: TrustedDefaultResolutionCommand
    record: InteractionRecord
    previous: InteractionRecord
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        _validate_temporal_context(self.observed_at, self.policy)
        expected = _reduce_trusted_default_resolution(
            self.previous,
            self.command,
            self.observed_at,
            self.policy,
        )
        _require_exact_result_record(
            self.record,
            expected,
            "trusted-default resolution",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionStoreReplayed:
    """Return a terminal outcome after a same-key or semantic replay."""

    command: _CandidateResolutionCommand
    record: InteractionRecord
    replay_kind: InteractionReplayKind
    previous: InteractionRecord | None = None
    lifecycle_mutation_applied: Literal[False] = field(
        init=False,
        default=False,
    )
    kind: Literal[InteractionStoreResultKind.REPLAYED] = field(
        init=False,
        default=InteractionStoreResultKind.REPLAYED,
    )

    def __post_init__(self) -> None:
        _validate_record(self.record, "result.record")
        _validate_candidate_resolution_command(self.command)
        key = self.command.idempotency_key
        if not _is_candidate_resolution_record(self.record):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "result.record",
                "caller-key replay requires an original candidate commit",
            )
        if not isinstance(self.replay_kind, InteractionReplayKind):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.replay_kind",
                "value must be an interaction replay kind",
            )
        _validate_replay_command(self.record, self.command)
        if self.replay_kind is InteractionReplayKind.SAME_KEY:
            if self.previous is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "result.previous",
                    "same-key replay performs no store mutation",
                )
            if not any(
                entry.key == key
                and entry.resolution_digest == self.command.resolution_digest
                for entry in self.record.idempotency_ledger
            ):
                raise InputValidationError(
                    InputErrorCode.IDEMPOTENCY_CONFLICT,
                    "result.idempotency_key",
                    "same-key replay must identify an accepted binding",
                )
            return
        if self.previous is None:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.previous",
                "new-key semantic replay requires the pre-binding record",
            )
        _validate_semantic_replay_binding(
            self.previous,
            self.record,
            self.command,
        )

    @property
    def idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return the exact transport key from the triggering command."""
        return self.command.idempotency_key

    @property
    def store_mutation_applied(self) -> bool:
        """Return whether replay atomically persisted a new transport key."""
        return self.replay_kind is InteractionReplayKind.SEMANTIC_NEW_KEY

    @property
    def decision_stage(self) -> ResolutionDecisionStage:
        """Return the admission stage that selected this replay."""
        if self.replay_kind is InteractionReplayKind.SAME_KEY:
            return ResolutionDecisionStage.IDEMPOTENCY_KEY
        return ResolutionDecisionStage.SEMANTIC_REPLAY


def apply_semantic_resolution_replay(
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    *,
    maximum_keys: int = MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
) -> InteractionStoreReplayed:
    """Append one exact semantic-replay key without lifecycle mutation."""
    _validate_candidate_resolution_command(command)
    disposition = evaluate_resolution_idempotency(
        previous,
        key=command.idempotency_key,
        resolution_digest=command.resolution_digest,
        maximum_keys=maximum_keys,
    )
    if disposition is not ResolutionIdempotencyDisposition.SEMANTIC_NEW_KEY:
        raise InputValidationError(
            (
                InputErrorCode.IDEMPOTENCY_LEDGER_FULL
                if disposition is ResolutionIdempotencyDisposition.LEDGER_FULL
                else InputErrorCode.IDEMPOTENCY_CONFLICT
            ),
            "command.idempotency_key",
            "command is not an admissible new-key semantic replay",
        )
    record = replace(
        previous,
        store_revision=_next_store_revision(previous.store_revision),
        idempotency_ledger=previous.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=command.idempotency_key,
                resolution_digest=command.resolution_digest,
            ),
        ),
    )
    return InteractionStoreReplayed(
        command=command,
        record=record,
        previous=previous,
        replay_kind=InteractionReplayKind.SEMANTIC_NEW_KEY,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _InteractionStoreRejectedBase:
    """Validate fields shared by operation-specific store rejections."""

    error: InputTransitionError
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REJECTED] = field(
        init=False,
        default=InteractionStoreResultKind.REJECTED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.error, InputTransitionError):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.error",
                "value must be an input transition error",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CreateInteractionRejected(_InteractionStoreRejectedBase):
    """Reject one create interaction command without mutation."""

    command: CreateInteractionCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (CreateInteractionCommand,),
            "create interaction",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInteractionRejected(_InteractionStoreRejectedBase):
    """Reject one request cancellation command without mutation."""

    command: CancelInteractionCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (CancelInteractionCommand,),
            "cancel interaction",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalizeInteractionRejected(_InteractionStoreRejectedBase):
    """Reject one request terminalization command without mutation."""

    command: TerminalizeInteractionCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (TerminalizeInteractionCommand,),
            "terminalize interaction",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeCancellationRejected(_InteractionStoreRejectedBase):
    """Reject one scope cancellation command without mutation."""

    command: TerminalizeInteractionScopeCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (TerminalizeInteractionScopeCommand,),
            "scope cancellation",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeSupersessionRejected(_InteractionStoreRejectedBase):
    """Reject one scope supersession command without mutation."""

    command: SupersedeInteractionScopeCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (SupersedeInteractionScopeCommand,),
            "scope supersession",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DueInteractionsRejected(_InteractionStoreRejectedBase):
    """Reject one due-interaction batch command without mutation."""

    command: TerminalizeDueInteractionsCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (TerminalizeDueInteractionsCommand,),
            "terminalize due interactions",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionPresentationRejected(_InteractionStoreRejectedBase):
    """Reject one presentation command without mutation."""

    command: InteractionPresentationCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (PresentInteractionCommand, DetachInteractionCommand),
            "interaction presentation",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ControllerActivityRejected(_InteractionStoreRejectedBase):
    """Reject one controller activity command without mutation."""

    command: RecordControllerActivityCommand

    def __post_init__(self) -> None:
        _InteractionStoreRejectedBase.__post_init__(self)
        _validate_rejected_command(
            self.command,
            (RecordControllerActivityCommand,),
            "controller activity",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ResolveInteractionRejected:
    """Reject resolution admission at one explicit precedence stage."""

    command: _CandidateResolutionCommand
    error: InputTransitionError
    decision_stage: ResolutionDecisionStage
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REJECTED] = field(
        init=False,
        default=InteractionStoreResultKind.REJECTED,
    )

    def __post_init__(self) -> None:
        _validate_candidate_resolution_command(self.command)
        if not isinstance(self.error, InputTransitionError):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.error",
                "value must be an input transition error",
            )
        if not isinstance(self.decision_stage, ResolutionDecisionStage):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.decision_stage",
                "value must be a resolution decision stage",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRegistrationApplied:
    """Return one newly persisted explicit branch edge."""

    command: RegisterInteractionBranchCommand
    record: InteractionBranchRecord
    store_mutation_applied: Literal[True] = field(init=False, default=True)
    kind: Literal[InteractionStoreResultKind.APPLIED] = field(
        init=False,
        default=InteractionStoreResultKind.APPLIED,
    )

    def __post_init__(self) -> None:
        _validate_branch_record(self.record, "result.record")
        _validate_branch_result_command(self.record, self.command)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRegistrationReplayed:
    """Return an identical pre-existing branch edge without mutation."""

    command: RegisterInteractionBranchCommand
    record: InteractionBranchRecord
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REPLAYED] = field(
        init=False,
        default=InteractionStoreResultKind.REPLAYED,
    )

    def __post_init__(self) -> None:
        _validate_branch_record(self.record, "result.record")
        _validate_branch_result_command(self.record, self.command)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchRegistrationRejected:
    """Reject a conflicting, cyclic, or unauthorized branch edge."""

    command: RegisterInteractionBranchCommand
    error: InputTransitionError
    store_mutation_applied: Literal[False] = field(init=False, default=False)
    kind: Literal[InteractionStoreResultKind.REJECTED] = field(
        init=False,
        default=InteractionStoreResultKind.REJECTED,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.command, RegisterInteractionBranchCommand):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.command",
                "value must be a branch registration command",
            )
        if not isinstance(self.error, InputTransitionError):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "result.error",
                "value must be an input transition error",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DueInteractionsApplied:
    """Return one exact same-observation due-settlement batch."""

    command: TerminalizeDueInteractionsCommand
    previous: tuple[InteractionRecord, ...]
    records: tuple[InteractionRecord, ...]
    observed_at: InteractionTime
    policy: InteractionPolicy
    store_mutation_applied: bool = field(init=False)
    kind: InteractionStoreResultKind = field(init=False)

    def __post_init__(self) -> None:
        _validate_temporal_context(self.observed_at, self.policy)
        expected = _reduce_due_interactions(
            self.previous,
            self.command,
            self.observed_at,
            self.policy,
        )
        _require_exact_result_records(self.records, expected, "due settlement")
        applied = bool(self.records)
        object.__setattr__(self, "store_mutation_applied", applied)
        object.__setattr__(
            self,
            "kind",
            (
                InteractionStoreResultKind.APPLIED
                if applied
                else InteractionStoreResultKind.REPLAYED
            ),
        )


CreateInteractionResult: TypeAlias = (
    CreateInteractionApplied | CreateInteractionRejected
)
CancelInteractionResult: TypeAlias = (
    CancelInteractionApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
    | CancelInteractionRejected
)
TerminalizeInteractionResult: TypeAlias = (
    TerminalizeInteractionApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
    | TerminalizeInteractionRejected
)
ScopeCancellationResult: TypeAlias = (
    ScopeCancellationApplied
    | ScopeCancellationReplayed
    | ScopeCancellationRejected
)
ScopeSupersessionResult: TypeAlias = (
    ScopeSupersessionApplied
    | ScopeSupersessionReplayed
    | ScopeSupersessionRejected
)
DueInteractionsResult: TypeAlias = (
    DueInteractionsApplied | DueInteractionsRejected
)
InteractionPresentationResult: TypeAlias = (
    InteractionPresentationApplied
    | ControllerLeaseExpiredApplied
    | ResolveInteractionApplied
    | InteractionPresentationRejected
)
ControllerActivityResult: TypeAlias = (
    ControllerActivityApplied
    | ControllerLeaseExpiredApplied
    | ResolveInteractionApplied
    | ControllerActivityRejected
)
InteractionResolutionResult: TypeAlias = (
    ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
    | InteractionStoreReplayed
    | ResolveInteractionRejected
)
TrustedDefaultResolutionResult: TypeAlias = (
    TrustedDefaultResolutionApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
)
InteractionBranchRegistrationResult: TypeAlias = (
    InteractionBranchRegistrationApplied
    | InteractionBranchRegistrationReplayed
    | InteractionBranchRegistrationRejected
)


def _present_interaction_record(
    record: InteractionRecord,
    command: PresentInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionPresentationApplied | ResolveInteractionApplied:
    """Apply trusted presentation timing as a metadata-only mutation."""
    return apply_interaction_presentation(
        record,
        command,
        observed_at,
        policy,
    )


class InteractionStore(Protocol):
    """Persist interaction state through strict async atomic operations."""

    async def create(
        self,
        command: CreateInteractionCommand,
    ) -> CreateInteractionResult:
        """Create, admit, and bind any bare resumer atomically.

        The implementation must call validate_interaction_admission with its
        complete locked lifetime snapshot. Terminal records remain part of
        equivalent-request loop accounting. A supplied resumer is registered
        under the request's broker-minted continuation identifier in the same
        commit and is never persisted.
        """
        ...

    async def create_admission(
        self,
        command: _InteractionAdmissionCreateCommand,
    ) -> CreateInteractionResult:
        """Commit one sealed broker admission and cleanup binding atomically.

        Cancellation may interrupt the caller before it observes the applied
        result, but once this coroutine terminates the implementation must not
        perform a later create mutation. The retained cleanup capability must
        then conclusively report absence or clean the committed admission.
        """
        ...

    async def cleanup_admission(
        self,
        command: _InteractionAdmissionCleanupCommand,
    ) -> _InteractionAdmissionCleanupResult:
        """Settle or prove absence of one exact capability-bound admission.

        This operation must not consult external authorization. An exact
        capability match is its sole authority. It atomically preserves any
        due deadline and controller-lease precedence before otherwise settling
        the match as unavailable, extracts its ephemeral resumer, and is
        idempotent after another terminal mutation. A terminal result is not
        conclusive until any bridge previously extracted for that exact
        capability has completed its delivery handoff. An unbound capability
        is indistinguishable from absence and must not probe identifiers. An
        absent result is a point-in-time conclusion only: a caller retaining
        the matching live create operation must join it or successfully close
        the handle before releasing cleanup authority.
        """
        ...

    async def lookup_scoped(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionDisclosureProjection | None:
        """Return one authorized projection or indistinguishable absence."""
        ...

    async def list_scoped(
        self,
        command: ListInteractionsCommand,
    ) -> tuple[InteractionDisclosureProjection, ...]:
        """Return projections visible within one authorized scope target."""
        ...

    async def lookup_branch_root(
        self,
        query: InteractionBranchRootLookup,
    ) -> InteractionBranchRoot | None:
        """Return one authorized root mapping or indistinguishable absence."""
        ...

    async def mark_presented(
        self,
        command: PresentInteractionCommand,
    ) -> InteractionPresentationResult:
        """Record presentation without awaiting external handler work."""
        ...

    async def mark_detached(
        self,
        command: DetachInteractionCommand,
    ) -> InteractionPresentationResult:
        """Record detached handling through a public store mutation."""
        ...

    async def resolve(
        self,
        command: _CandidateResolutionCommand,
    ) -> InteractionResolutionResult:
        """Apply RESOLUTION_DECISION_PRECEDENCE in one atomic operation.

        A semantic new-key replay must use
        apply_semantic_resolution_replay under the same store lock.
        """
        ...

    async def resolve_trusted_default(
        self,
        command: TrustedDefaultResolutionCommand,
    ) -> TrustedDefaultResolutionResult:
        """Derive and commit declared defaults through trusted policy."""
        ...

    async def terminalize(
        self,
        command: TerminalizeInteractionCommand,
    ) -> TerminalizeInteractionResult:
        """Apply one trusted non-answer terminal transition atomically."""
        ...

    async def cancel(
        self,
        command: CancelInteractionCommand,
    ) -> CancelInteractionResult:
        """Cancel one exact request with request-local cancellation scope."""
        ...

    async def terminalize_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> ScopeCancellationResult:
        """Cancel a complete locked-snapshot scope selection atomically.

        The implementation must share one backing across all handles and call
        the private scope-transaction helpers under that backing's lock. The
        helpers read and mutate the complete backing-owned inventory directly.
        """
        ...

    async def supersede_scope(
        self,
        command: SupersedeInteractionScopeCommand,
    ) -> ScopeSupersessionResult:
        """Supersede a complete locked-snapshot scope selection atomically.

        The implementation must share one backing across all handles and call
        the private scope-transaction helpers under that backing's lock. The
        helpers read and mutate the complete backing-owned inventory directly.
        """
        ...

    async def register_branch(
        self,
        command: RegisterInteractionBranchCommand,
    ) -> InteractionBranchRegistrationResult:
        """Register a child edge and reject cycles or ownership drift."""
        ...

    async def record_activity(
        self,
        command: RecordControllerActivityCommand,
    ) -> ControllerActivityResult:
        """Acquire, pulse, release, or disconnect active-control evidence."""
        ...

    async def wait_for_change(
        self,
        command: WaitForInteractionChangeCommand,
    ) -> InteractionDisclosureProjection:
        """Wait for a newer authorized projection.

        Missing and unauthorized targets must both raise the same content-safe
        InteractionNotFoundError. A closed local handle instead raises
        InteractionStoreClosedError.
        """
        ...

    async def next_deadline(self) -> InteractionDeadlineSnapshot:
        """Return select_next_interaction_deadline from one locked snapshot."""
        ...

    async def wait_for_deadline_change(
        self,
        command: WaitForDeadlineChangeCommand,
    ) -> InteractionDeadlineSnapshot:
        """Wake on generation change or raise InteractionStoreClosedError."""
        ...

    async def terminalize_due(
        self,
        command: TerminalizeDueInteractionsCommand,
    ) -> DueInteractionsResult:
        """Settle due advisory and absolute deadlines deterministically."""
        ...

    async def aclose(self) -> None:
        """Idempotently close only this handle and fail its pending waits.

        An operation that completes its atomic commit before close returns its
        committed result. Close that linearizes first causes that operation,
        every later operation, and both waiter variants to raise
        InteractionStoreClosedError. A separately reopened handle cannot close
        this handle; its committed backing-state changes may still wake this
        handle's data waiters through the shared store notification path.
        """
        ...


class InteractionStoreFactory(Protocol):
    """Open independent handles over one configured persisted backing state."""

    async def open(self) -> InteractionStore:
        """Return one open handle with isolated waits and close state."""
        ...


def apply_create_interaction(
    command: CreateInteractionCommand,
    policy: InteractionPolicy,
) -> CreateInteractionApplied:
    """Apply one exact create-and-admit command without backend math."""
    return CreateInteractionApplied(
        command=command,
        record=_reduce_create_interaction(command, policy),
        policy=policy,
    )


def apply_interaction_presentation(
    previous: InteractionRecord,
    command: PresentInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionPresentationApplied | ResolveInteractionApplied:
    """Apply first presentation from one trusted temporal observation."""
    _validate_presentation_command(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    return InteractionPresentationApplied(
        command=command,
        record=_reduce_interaction_presentation(
            previous,
            command,
            observed_at,
            policy,
        ),
        previous=previous,
        observed_at=observed_at,
        policy=policy,
    )


def apply_interaction_detachment(
    previous: InteractionRecord,
    command: DetachInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> (
    InteractionPresentationApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
):
    """Apply detachment after deadline and lease-expiry precedence."""
    _validate_presentation_command_identity(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    _validate_presentation_command(previous, command)
    return InteractionPresentationApplied(
        command=command,
        record=_reduce_interaction_presentation(
            previous,
            command,
            observed_at,
            policy,
        ),
        previous=previous,
        observed_at=observed_at,
        policy=policy,
    )


def apply_controller_activity(
    previous: InteractionRecord,
    command: RecordControllerActivityCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    lease_nonce: ActiveControlLeaseNonce | None = None,
) -> (
    ControllerActivityApplied
    | ControllerLeaseExpiredApplied
    | ResolveInteractionApplied
):
    """Apply activity or the deadline or lease-expiry mutation that wins."""
    _validate_controller_command(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    return ControllerActivityApplied(
        command=command,
        record=_reduce_controller_activity(
            previous,
            command,
            observed_at,
            policy,
            lease_nonce=lease_nonce,
        ),
        previous=previous,
        observed_at=observed_at,
        policy=policy,
        lease_nonce=lease_nonce,
    )


def apply_candidate_resolution(
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    classifier_binding: _TaskInputClassifierBinding | None = None,
    classification_proof: _BoundTaskInputClassifications | None = None,
) -> ResolveInteractionApplied | ControllerLeaseExpiredApplied:
    """Apply a candidate only after deadline and lease-expiry settlement."""
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
            command=command,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    return ResolveInteractionApplied(
        record=_reduce_candidate_resolution(
            previous,
            command,
            observed_at,
            policy,
            classifier_binding,
            classification_proof,
        ),
        previous=previous,
        decision_stage=ResolutionDecisionStage.COMMIT,
        observed_at=observed_at,
        policy=policy,
        command=command,
        idempotency_key=command.idempotency_key,
        classifier_binding=classifier_binding,
        classification_proof=classification_proof,
    )


def apply_trusted_default_resolution(
    previous: InteractionRecord,
    command: TrustedDefaultResolutionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> TrustedDefaultResolutionResult:
    """Derive declared defaults after deadline and lease precedence."""
    _validate_trusted_default_command_identity(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    _validate_trusted_default_command_revision(previous, command)
    return TrustedDefaultResolutionApplied(
        command=command,
        record=_reduce_trusted_default_resolution(
            previous,
            command,
            observed_at,
            policy,
        ),
        previous=previous,
        observed_at=observed_at,
        policy=policy,
    )


def apply_due_interaction(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> ResolveInteractionApplied | ControllerLeaseExpiredApplied | None:
    """Apply one due lifecycle or controller-lease settlement."""
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is None:
        return None
    return ControllerLeaseExpiredApplied(
        record=lease_expiry,
        previous=previous,
        observed_at=observed_at,
        policy=policy,
        internal_authority=_DEADLINE_RESOLVER,
    )


def apply_due_interactions(
    previous: tuple[InteractionRecord, ...],
    command: TerminalizeDueInteractionsCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> DueInteractionsApplied:
    """Apply one bounded due batch from one coherent clock observation."""
    return DueInteractionsApplied(
        command=command,
        previous=previous,
        records=_reduce_due_interactions(
            previous,
            command,
            observed_at,
            policy,
        ),
        observed_at=observed_at,
        policy=policy,
    )


def apply_request_cancellation(
    previous: InteractionRecord,
    command: CancelInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> (
    CancelInteractionApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
):
    """Apply cancellation or the deadline or lease mutation that wins."""
    _validate_request_cancellation_command_identity(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    _validate_request_cancellation_command_revision(previous, command)
    return CancelInteractionApplied(
        command=command,
        previous=previous,
        record=_reduce_request_cancellation(
            previous,
            command,
            observed_at,
            policy,
        ),
        observed_at=observed_at,
        policy=policy,
    )


def apply_request_terminalization(
    previous: InteractionRecord,
    command: TerminalizeInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> (
    TerminalizeInteractionApplied
    | ResolveInteractionApplied
    | ControllerLeaseExpiredApplied
):
    """Apply terminalization or the deadline or lease mutation that wins."""
    _validate_request_terminalization_command_identity(previous, command)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return ResolveInteractionApplied(
            command=command,
            record=deadline,
            previous=previous,
            decision_stage=ResolutionDecisionStage.DEADLINE,
            observed_at=observed_at,
            policy=policy,
        )
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    if lease_expiry is not None:
        return ControllerLeaseExpiredApplied(
            command=command,
            record=lease_expiry,
            previous=previous,
            observed_at=observed_at,
            policy=policy,
        )
    _validate_request_terminalization_command_revision(previous, command)
    return TerminalizeInteractionApplied(
        command=command,
        previous=previous,
        record=_reduce_request_terminalization(
            previous,
            command,
            observed_at,
            policy,
        ),
        observed_at=observed_at,
        policy=policy,
    )


def _begin_scope_transaction(
    backing: _InteractionStoreBacking,
    command: InteractionScopeMutationCommand,
) -> _InteractionScopeTransaction:
    """Begin a complete scope mutation under a concrete-store lock.

    Concrete stores must share one backing and call this helper while holding
    that backing's store lock. The helper reads every record and explicit
    branch edge directly from the backing's authoritative generation. The
    returned private plan cannot cross the public store boundary.
    """
    snapshot = _snapshot_interaction_store_backing(backing)
    if not isinstance(
        command,
        (
            TerminalizeInteractionScopeCommand,
            SupersedeInteractionScopeCommand,
        ),
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a scope mutation command",
        )
    selected = _select_scope_records(
        snapshot.records,
        command.scope,
        snapshot.branch_records,
        command.actor.principal,
    )
    _validate_scope_ownership_presence(
        snapshot,
        command.scope,
        command.actor.principal,
        selected,
    )
    authoritative_branch_roots = (
        frozenset()
        if snapshot.branch_closure_attestation is None
        else (snapshot.branch_closure_attestation.authoritative_branch_roots)
    )
    _validate_scope_ownership(
        snapshot.records,
        snapshot.branch_records,
        authoritative_branch_roots,
        command.scope,
        command.actor.principal,
        selected,
    )
    return _InteractionScopeTransaction(
        command=command,
        scope=command.scope,
        principal=command.actor.principal,
        store_generation=snapshot.store_generation,
        snapshot_digest=_canonical_scope_snapshot_digest(
            snapshot.records,
            snapshot.branch_records,
            authoritative_branch_roots,
            snapshot.scope_ownership_attestation,
        ),
        selected_records=selected,
        backing=backing,
        _token=_SCOPE_SELECTION_TOKEN,
    )


def _apply_scope_cancellation(
    transaction: _InteractionScopeTransaction,
    command: TerminalizeInteractionScopeCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    backing: _InteractionStoreBacking,
) -> ScopeCancellationApplied | ScopeCancellationReplayed:
    """Commit one private complete cancellation transaction."""
    if not isinstance(command, TerminalizeInteractionScopeCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a scope cancellation command",
        )
    _validate_scope_transaction_commit(
        transaction,
        command,
        backing,
    )
    if not transaction.selected_records:
        return ScopeCancellationReplayed(
            command=command,
            _token=_SCOPE_RESULT_TOKEN,
        )
    records = _reduce_scope_cancellation(
        transaction.selected_records,
        command,
        observed_at,
        policy,
    )
    result = ScopeCancellationApplied(
        command=command,
        previous=transaction.selected_records,
        records=records,
        observed_at=observed_at,
        policy=policy,
        _token=_SCOPE_RESULT_TOKEN,
    )
    _replace_interaction_store_backing_records(
        backing,
        transaction.selected_records,
        records,
    )
    return result


def _apply_scope_supersession(
    transaction: _InteractionScopeTransaction,
    command: SupersedeInteractionScopeCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    backing: _InteractionStoreBacking,
) -> ScopeSupersessionApplied | ScopeSupersessionReplayed:
    """Commit one private complete supersession transaction."""
    if not isinstance(command, SupersedeInteractionScopeCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a scope supersession command",
        )
    _validate_scope_transaction_commit(
        transaction,
        command,
        backing,
    )
    if not transaction.selected_records:
        return ScopeSupersessionReplayed(
            command=command,
            _token=_SCOPE_RESULT_TOKEN,
        )
    records = _reduce_scope_supersession(
        transaction.selected_records,
        command,
        observed_at,
        policy,
    )
    result = ScopeSupersessionApplied(
        command=command,
        previous=transaction.selected_records,
        records=records,
        observed_at=observed_at,
        policy=policy,
        _token=_SCOPE_RESULT_TOKEN,
    )
    _replace_interaction_store_backing_records(
        backing,
        transaction.selected_records,
        records,
    )
    return result


def _validate_policy(policy: object) -> None:
    if not isinstance(policy, InteractionPolicy):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "policy",
            "value must be an interaction policy",
        )


def _scope_authorization_target(
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
) -> InteractionScopeAuthorizationTarget:
    return InteractionScopeAuthorizationTarget(
        run_id=scope.run_id,
        turn_id=scope.turn_id,
        task_id=scope.task_id,
        agent_id=scope.agent_id,
        branch_id=scope.branch_id,
        include_descendants=scope.include_descendants,
        principal=principal,
    )


def _task_input_classification_requests(
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    policy: InteractionPolicy,
) -> tuple[TaskInputClassificationRequest, ...]:
    """Return all classifier work for one exact stored candidate."""
    _validate_pending_record(previous)
    _validate_candidate_resolution_command(command)
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "resolve command does not match the stored request",
        )
    _validate_policy(policy)
    return tuple(
        TaskInputClassificationRequest(
            value=value,
            request_id=command.correlation.request_id,
            candidate_digest=command.resolution_digest,
            question_id=question_id,
            semantic_type=semantic_type,
            policy_revision=policy.task_input_policy_revision,
        )
        for value, question_id, semantic_type in _candidate_free_form_values(
            command
        )
    )


def _candidate_free_form_values(
    command: _CandidateResolutionCommand,
) -> tuple[tuple[str, QuestionId, QuestionType], ...]:
    resolution = command.proposed_resolution
    work: list[tuple[str, QuestionId, QuestionType]] = []
    if isinstance(resolution, AnsweredResolution):
        for answer in resolution.answers:
            if isinstance(answer, (TextAnswer, MultilineTextAnswer)):
                value = answer.value
            elif isinstance(answer, SingleSelectionAnswer) and isinstance(
                answer.value,
                FreeFormOther,
            ):
                value = answer.value.text
            elif isinstance(answer, MultipleSelectionAnswer):
                other = next(
                    (
                        value
                        for value in answer.values
                        if isinstance(value, FreeFormOther)
                    ),
                    None,
                )
                if other is None:
                    continue
                value = other.text
            else:
                continue
            work.append((value, answer.question_id, answer.question_type))
    return tuple(work)


def _bind_task_input_classifications(
    binding: _TaskInputClassifierBinding,
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    classifications: tuple[TaskInputClassification, ...],
    policy: InteractionPolicy,
) -> _BoundTaskInputClassifications:
    """Bind untrusted outputs to exact centralized classifier work."""
    _validate_classifier_binding_policy(binding, policy)
    requests = _task_input_classification_requests(
        previous,
        command,
        policy,
    )
    normalized = _validate_untrusted_classification_outputs(
        binding,
        requests,
        classifications,
    )
    return _BoundTaskInputClassifications(
        binding=binding,
        request_id=command.correlation.request_id,
        candidate_digest=command.resolution_digest,
        classifications=normalized,
        _token=_CLASSIFICATION_PROOF_TOKEN,
    )


def _validate_candidate_classifications(
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    policy: InteractionPolicy,
    binding: _TaskInputClassifierBinding | None,
    proof: _BoundTaskInputClassifications | None,
) -> None:
    requests = _task_input_classification_requests(
        previous,
        command,
        policy,
    )
    if not requests:
        if binding is not None or proof is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "classification_proof",
                "candidate does not require task-input classification",
            )
        return
    if type(binding) is not _TaskInputClassifierBinding:
        raise InputValidationError(
            InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE,
            "classifier_binding",
            "free-form input requires the backing classifier binding",
        )
    if type(proof) is not _BoundTaskInputClassifications:
        raise InputValidationError(
            InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE,
            "classification_proof",
            "free-form input requires broker-bound classifications",
        )
    assert isinstance(binding, _TaskInputClassifierBinding)
    assert isinstance(proof, _BoundTaskInputClassifications)
    _validate_classifier_binding_policy(binding, policy)
    if proof._binding_capability is not binding._capability:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "classification_proof",
            "classification proof belongs to another classifier binding",
        )
    if (
        proof.classifier_id != binding.classifier_id
        or proof.policy_revision != binding.policy_revision
        or proof.request_id != command.correlation.request_id
        or proof.candidate_digest != command.resolution_digest
    ):
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "classification_proof",
            "classification proof does not match the current candidate",
        )
    _validate_untrusted_classification_outputs(
        binding,
        requests,
        proof.classifications,
    )


def _validate_classifier_binding_policy(
    binding: object,
    policy: InteractionPolicy,
) -> None:
    _validate_policy(policy)
    if type(binding) is not _TaskInputClassifierBinding:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "classifier_binding",
            "value must be a store-owned classifier binding",
        )
    assert isinstance(binding, _TaskInputClassifierBinding)
    if (
        binding.classifier_id != policy.task_input_classifier_id
        or binding.policy_revision != policy.task_input_policy_revision
    ):
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "classifier_binding",
            "classifier binding is stale for current policy",
        )


def _validate_untrusted_classification_outputs(
    binding: _TaskInputClassifierBinding,
    requests: tuple[TaskInputClassificationRequest, ...],
    classifications: object,
) -> tuple[TaskInputClassification, ...]:
    if not isinstance(classifications, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "classifications",
            "classifier outputs must be a tuple",
        )
    if len(classifications) != len(requests):
        raise InputValidationError(
            InputErrorCode.SECRET_CLASSIFICATION_UNAVAILABLE,
            "classifications",
            "every free-form value requires one classifier output",
        )
    classification_ids: set[str] = set()
    question_ids: set[QuestionId] = set()
    normalized: list[TaskInputClassification] = []
    for request, classification in zip(requests, classifications, strict=True):
        if type(classification) is not TaskInputClassification:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "classifications",
                "values must be task-input classifier outputs",
            )
        assert isinstance(classification, TaskInputClassification)
        if (
            classification.classifier_id != binding.classifier_id
            or classification.policy_revision != binding.policy_revision
        ):
            raise InputValidationError(
                InputErrorCode.STALE_REVISION,
                "classifications",
                "classifier identity or policy revision is not current",
            )
        if (
            classification.request_id != request.request_id
            or classification.candidate_digest != request.candidate_digest
            or classification.question_id != request.question_id
            or classification.semantic_type is not request.semantic_type
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "classifications",
                "classifier output does not match its exact requested value",
            )
        if (
            classification.classification_id in classification_ids
            or classification.question_id in question_ids
        ):
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "classifications",
                "classifier outputs must have unique identities",
            )
        if (
            classification.decision
            is not TaskInputClassificationDecision.ALLOW
        ):
            raise InputValidationError(
                InputErrorCode.PROHIBITED_INPUT,
                "classifications",
                "configured policy rejected submitted free-form input",
            )
        classification_ids.add(classification.classification_id)
        question_ids.add(classification.question_id)
        normalized.append(classification)
    return tuple(normalized)


def _validate_rejected_command(
    command: object,
    expected_types: tuple[type[object], ...],
    operation: str,
) -> None:
    if not isinstance(command, expected_types):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "result.command",
            f"value must be a {operation} command",
        )


def _validate_temporal_context(
    observed_at: object,
    policy: object,
) -> None:
    if type(observed_at) is not InteractionTime:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "observed_at",
            "value must be a sealed trusted clock observation",
        )
    _validate_policy(policy)


def _reduce_create_interaction(
    command: CreateInteractionCommand,
    policy: InteractionPolicy,
) -> InteractionRecord:
    if not isinstance(command, CreateInteractionCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a create interaction command",
        )
    _validate_policy(policy)
    transition = mark_request_pending(
        command.request,
        expected_state_revision=command.request.state_revision,
    )
    if not isinstance(transition, InputTransitionApplied):
        raise InputValidationError(
            transition.error.code,
            transition.error.path,
            transition.error.message,
        )
    request = transition.request
    advisory_wait = None
    if request.mode is RequirementMode.ADVISORY:
        assert request.advisory_wait_seconds is not None
        advisory_wait = AdvisoryWaitState(
            status=AdvisoryWaitStatus.QUEUED,
            budget_seconds=request.advisory_wait_seconds,
            remaining_seconds=request.advisory_wait_seconds,
        )
    return InteractionRecord(
        request=request,
        semantic_fingerprint=semantic_request_fingerprint(request),
        absolute_expires_at=request.created_at
        + timedelta(seconds=request.continuation_ttl_seconds),
        presentation=InteractionPresentationState.QUEUED,
        store_revision=InteractionStoreRevision(1),
        advisory_wait=advisory_wait,
    )


def _reduce_interaction_presentation(
    previous: InteractionRecord,
    command: InteractionPresentationCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord:
    _validate_record(previous, "previous")
    _validate_temporal_context(observed_at, policy)
    _validate_presentation_command(previous, command)
    if observed_at.wall_time < previous.request.created_at:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "observed_at.wall_time",
            "presentation observation predates request creation",
        )
    if _due_resolution_status(previous, observed_at, policy) is not None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "deadline settlement must win before presentation mutation",
        )
    presentation = (
        InteractionPresentationState.PRESENTED
        if isinstance(command, PresentInteractionCommand)
        else InteractionPresentationState.DETACHED
    )
    if (
        previous.request.state is not RequestState.PENDING
        or previous.request.resolution is not None
        or previous.presentation
        not in {
            InteractionPresentationState.QUEUED,
            InteractionPresentationState.PRESENTED,
        }
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "previous",
            "presentation mutations require an unresolved pending request",
        )
    if presentation is InteractionPresentationState.DETACHED:
        return replace(
            previous,
            presentation=presentation,
            store_revision=_next_store_revision(previous.store_revision),
        )
    if previous.presentation is not InteractionPresentationState.QUEUED:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "previous.presentation",
            "only a queued request can record first presentation",
        )
    request = _anchor_request_presentation(
        previous.request,
        observed_at.wall_time,
    )
    advisory_wait = previous.advisory_wait
    if advisory_wait is not None:
        if advisory_wait.status is not AdvisoryWaitStatus.QUEUED:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "previous.advisory_wait.status",
                "first presentation requires queued advisory timing",
            )
        advisory_wait = replace(
            advisory_wait,
            status=AdvisoryWaitStatus.RUNNING,
            presented_at=observed_at.wall_time,
            running_since_monotonic=observed_at.monotonic_seconds,
        )
    return replace(
        previous,
        request=request,
        presentation=presentation,
        store_revision=_next_store_revision(previous.store_revision),
        advisory_wait=advisory_wait,
    )


def _validate_pending_record(previous: InteractionRecord) -> None:
    _validate_record(previous, "previous")
    if (
        previous.request.state is not RequestState.PENDING
        or previous.request.resolution is not None
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "previous.request.state",
            "mutation requires an unresolved pending request",
        )


def _validate_monotonic_progress(
    wait: AdvisoryWaitState,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> None:
    anchor = wait.running_since_monotonic
    if wait.status is AdvisoryWaitStatus.PAUSED:
        assert wait.lease_expires_at_monotonic is not None
        anchor = (
            wait.lease_expires_at_monotonic
            - policy.active_control_lease_seconds
        )
    if anchor is not None and observed_at.monotonic_seconds < anchor:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at.monotonic_seconds",
            "trusted monotonic observation predates stored operational time",
        )


def _due_resolution_status(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> ResolutionStatus | None:
    _validate_pending_record(previous)
    _validate_temporal_context(observed_at, policy)
    wait = previous.advisory_wait
    if isinstance(wait, AdvisoryWaitState):
        _validate_monotonic_progress(wait, observed_at, policy)
    if observed_at.wall_deadline_reached(previous.absolute_expires_at):
        return ResolutionStatus.EXPIRED
    if not isinstance(wait, AdvisoryWaitState):
        return None
    effective_deadline = None
    if wait.status is AdvisoryWaitStatus.RUNNING:
        assert wait.running_since_monotonic is not None
        effective_deadline = (
            wait.running_since_monotonic + wait.remaining_seconds
        )
    elif wait.status is AdvisoryWaitStatus.PAUSED:
        assert wait.lease_expires_at_monotonic is not None
        if observed_at.monotonic_seconds >= wait.lease_expires_at_monotonic:
            effective_deadline = (
                wait.lease_expires_at_monotonic + wait.remaining_seconds
            )
    if (
        effective_deadline is not None
        and observed_at.monotonic_deadline_reached(effective_deadline)
    ):
        return ResolutionStatus.TIMED_OUT
    return None


def _reduce_due_resolution(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord | None:
    status = _due_resolution_status(previous, observed_at, policy)
    if status is None:
        return None
    if status is ResolutionStatus.EXPIRED:
        resolution: InputResolution = ExpiredResolution(
            request_id=previous.request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=observed_at.wall_time,
        )
        advisory_wait = previous.advisory_wait
    else:
        resolution = TimedOutResolution(
            request_id=previous.request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=observed_at.wall_time,
        )
        wait = previous.advisory_wait
        assert isinstance(wait, AdvisoryWaitState)
        advisory_wait = replace(
            wait,
            status=AdvisoryWaitStatus.EXHAUSTED,
            remaining_seconds=0,
            running_since_monotonic=None,
            controller_id=None,
            lease_nonce=None,
            activity_sequence=None,
            lease_expires_at_monotonic=None,
        )
    return _reduce_terminal_resolution(
        previous,
        resolution,
        _DEADLINE_RESOLVER,
        advisory_wait=advisory_wait,
    )


def _reduce_expired_controller_lease(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord | None:
    _validate_pending_record(previous)
    _validate_temporal_context(observed_at, policy)
    wait = previous.advisory_wait
    if not isinstance(wait, AdvisoryWaitState):
        return None
    _validate_monotonic_progress(wait, observed_at, policy)
    if wait.status is not AdvisoryWaitStatus.PAUSED:
        return None
    lease_expiry = wait.lease_expires_at_monotonic
    assert lease_expiry is not None
    if observed_at.monotonic_seconds < lease_expiry:
        return None
    if _due_resolution_status(previous, observed_at, policy) is not None:
        return None
    resumed = replace(
        wait,
        status=AdvisoryWaitStatus.RUNNING,
        running_since_monotonic=lease_expiry,
        controller_id=None,
        lease_nonce=None,
        activity_sequence=None,
        lease_expires_at_monotonic=None,
    )
    return replace(
        previous,
        advisory_wait=resumed,
        store_revision=_next_store_revision(previous.store_revision),
    )


def _reduce_controller_activity(
    previous: InteractionRecord,
    command: RecordControllerActivityCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    lease_nonce: ActiveControlLeaseNonce | None,
) -> InteractionRecord:
    _validate_pending_record(previous)
    _validate_temporal_context(observed_at, policy)
    _validate_controller_command(previous, command)
    evidence = command.evidence
    if (
        previous.request.mode is not RequirementMode.ADVISORY
        or previous.presentation is not InteractionPresentationState.PRESENTED
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "previous",
            "controller activity requires its presented advisory request",
        )
    wait = previous.advisory_wait
    assert isinstance(wait, AdvisoryWaitState)
    _validate_monotonic_progress(wait, observed_at, policy)
    if _due_resolution_status(previous, observed_at, policy) is not None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "deadline settlement must win before controller activity",
        )
    if (
        _reduce_expired_controller_lease(previous, observed_at, policy)
        is not None
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "lease-expiry settlement must win before controller activity",
        )
    if type(evidence) is AcquireControllerActivity:
        if wait.status is not AdvisoryWaitStatus.RUNNING:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "previous.advisory_wait.status",
                "acquire requires a running advisory wait",
            )
        assert wait.running_since_monotonic is not None
        nonce = ActiveControlLeaseNonce(
            validate_opaque_id(lease_nonce, "lease_nonce")
        )
        remaining = wait.remaining_seconds - (
            observed_at.monotonic_seconds - wait.running_since_monotonic
        )
        if remaining <= 0:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "observed_at",
                "advisory timeout must win at its effective deadline",
            )
        updated_wait = replace(
            wait,
            status=AdvisoryWaitStatus.PAUSED,
            remaining_seconds=remaining,
            running_since_monotonic=None,
            controller_id=evidence.controller_id,
            lease_nonce=nonce,
            activity_sequence=0,
            lease_expires_at_monotonic=(
                observed_at.monotonic_seconds
                + policy.active_control_lease_seconds
            ),
        )
    else:
        if lease_nonce is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "lease_nonce",
                "only acquisition returns a newly minted nonce",
            )
        _validate_existing_controller_lease(wait, evidence)
        lease_expiry = wait.lease_expires_at_monotonic
        assert lease_expiry is not None
        if observed_at.monotonic_seconds >= lease_expiry:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "observed_at",
                "controller evidence must arrive before lease expiry",
            )
        if type(evidence) is PulseControllerActivity:
            updated_wait = replace(
                wait,
                activity_sequence=evidence.sequence,
                lease_expires_at_monotonic=(
                    observed_at.monotonic_seconds
                    + policy.active_control_lease_seconds
                ),
            )
        elif type(evidence) in {
            ReleaseControllerActivity,
            DisconnectControllerActivity,
        }:
            updated_wait = replace(
                wait,
                status=AdvisoryWaitStatus.RUNNING,
                running_since_monotonic=observed_at.monotonic_seconds,
                controller_id=None,
                lease_nonce=None,
                activity_sequence=None,
                lease_expires_at_monotonic=None,
            )
        else:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "evidence",
                "value must be supported controller activity evidence",
            )
    return replace(
        previous,
        advisory_wait=updated_wait,
        store_revision=_next_store_revision(previous.store_revision),
    )


def _reduce_candidate_resolution(
    previous: InteractionRecord,
    command: _CandidateResolutionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    classifier_binding: _TaskInputClassifierBinding | None = None,
    classification_proof: _BoundTaskInputClassifications | None = None,
) -> InteractionRecord:
    _validate_pending_record(previous)
    _validate_temporal_context(observed_at, policy)
    _validate_candidate_resolution_command(command)
    if (
        isinstance(command, _TrustedPolicyResolutionCommand)
        and command.actor.principal != previous.request.origin.principal
    ):
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "command.actor",
            "policy resolver differs from the request principal",
        )
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "resolve command does not match the stored request",
        )
    if command.expected_state_revision != previous.request.state_revision:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "expected_state_revision",
            "request revision is stale",
        )
    _validate_candidate_classifications(
        previous,
        command,
        policy,
        classifier_binding,
        classification_proof,
    )
    if _due_resolution_status(previous, observed_at, policy) is not None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "deadline settlement must win before candidate resolution",
        )
    if (
        _reduce_expired_controller_lease(previous, observed_at, policy)
        is not None
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "lease-expiry settlement must win before candidate resolution",
        )
    if len(previous.idempotency_ledger) >= (
        policy.maximum_idempotency_keys_per_request
    ):
        raise InputValidationError(
            InputErrorCode.IDEMPOTENCY_LEDGER_FULL,
            "record.idempotency_ledger",
            "idempotency ledger reached its configured bound",
        )
    trusted = replace(
        command.proposed_resolution,
        resolved_at=observed_at.wall_time,
    )
    transition = resolve_request(
        previous.request,
        trusted,
        expected_state_revision=command.expected_state_revision,
    )
    if not isinstance(transition, InputTransitionApplied):
        raise InputValidationError(
            transition.error.code,
            transition.error.path,
            transition.error.message,
        )
    digest = canonical_resolution_digest(trusted)
    return replace(
        previous,
        request=transition.request,
        store_revision=_next_store_revision(previous.store_revision),
        resolution_digest=digest,
        idempotency_ledger=previous.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=command.idempotency_key,
                resolution_digest=digest,
            ),
        ),
        resolved_by=(
            _TRUSTED_POLICY_RESOLVER
            if isinstance(command, _TrustedPolicyResolutionCommand)
            else command.actor.principal
        ),
    )


def _validate_trusted_default_command_identity(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_pending_record(previous)
    command = _validate_trusted_default_resolution_command(command)
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "trusted-default command does not match the stored request",
        )


def _validate_trusted_default_command_revision(
    previous: InteractionRecord,
    command: TrustedDefaultResolutionCommand,
) -> None:
    if command.expected_state_revision != previous.request.state_revision:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "command.expected_state_revision",
            "request revision is stale",
        )


def _validate_trusted_default_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_trusted_default_command_identity(previous, command)
    assert isinstance(command, TrustedDefaultResolutionCommand)
    _validate_trusted_default_command_revision(previous, command)


def _reduce_trusted_default_resolution(
    previous: InteractionRecord,
    command: TrustedDefaultResolutionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord:
    _validate_trusted_default_command(previous, command)
    _validate_temporal_context(observed_at, policy)
    _require_no_prior_settlement(previous, observed_at, policy)
    trusted = _declared_trusted_default_resolution(
        previous.request,
        observed_at.wall_time,
    )
    transition = resolve_request(
        previous.request,
        trusted,
        expected_state_revision=command.expected_state_revision,
    )
    if not isinstance(transition, InputTransitionApplied):
        raise InputValidationError(
            transition.error.code,
            transition.error.path,
            transition.error.message,
        )
    return replace(
        previous,
        request=transition.request,
        store_revision=_next_store_revision(previous.store_revision),
        resolution_digest=canonical_resolution_digest(trusted),
        resolved_by=_TRUSTED_DEFAULT_RESOLVER,
    )


def _declared_trusted_default_resolution(
    request: InputRequest,
    resolved_at: datetime,
) -> AnsweredResolution:
    answers: list[
        ConfirmationAnswer
        | TextAnswer
        | MultilineTextAnswer
        | SingleSelectionAnswer
        | MultipleSelectionAnswer
    ] = []
    for question in request.questions:
        if isinstance(question, ConfirmationQuestion):
            confirmation_default = question.default_value
            if confirmation_default is None:
                _require_optional_default(question.required)
                continue
            answers.append(
                ConfirmationAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.TRUSTED_DEFAULT,
                    value=confirmation_default,
                )
            )
        elif isinstance(question, TextQuestion):
            text_default = question.default_value
            if text_default is None:
                _require_optional_default(question.required)
                continue
            answers.append(
                TextAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.TRUSTED_DEFAULT,
                    value=text_default,
                )
            )
        elif isinstance(question, MultilineTextQuestion):
            multiline_default = question.default_value
            if multiline_default is None:
                _require_optional_default(question.required)
                continue
            answers.append(
                MultilineTextAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.TRUSTED_DEFAULT,
                    value=multiline_default,
                )
            )
        elif isinstance(question, SingleSelectionQuestion):
            single_default = question.default_value
            if single_default is None:
                _require_optional_default(question.required)
                continue
            answers.append(
                SingleSelectionAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.TRUSTED_DEFAULT,
                    value=SelectedChoice(value=single_default),
                )
            )
        else:
            assert isinstance(question, MultipleSelectionQuestion)
            multiple_default = question.default_value
            if multiple_default is None:
                _require_optional_default(question.required)
                continue
            answers.append(
                MultipleSelectionAnswer(
                    question_id=question.question_id,
                    provenance=AnswerProvenance.TRUSTED_DEFAULT,
                    values=tuple(
                        SelectedChoice(value=value)
                        for value in multiple_default
                    ),
                )
            )
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.TRUSTED_DEFAULT,
        resolved_at=resolved_at,
        answers=tuple(answers),
    )


def _require_optional_default(required: bool) -> None:
    if required:
        raise InputValidationError(
            InputErrorCode.INVALID_DEFAULT,
            "request.questions",
            "required question has no declared trusted default",
        )


def _reduce_terminal_resolution(
    previous: InteractionRecord,
    resolution: InputResolution,
    resolver: InteractionResolutionAuthority,
    *,
    advisory_wait: AdvisoryWaitState | None = None,
) -> InteractionRecord:
    transition = resolve_request(
        previous.request,
        resolution,
        expected_state_revision=previous.request.state_revision,
    )
    if not isinstance(transition, InputTransitionApplied):
        raise InputValidationError(
            transition.error.code,
            transition.error.path,
            transition.error.message,
        )
    return replace(
        previous,
        request=transition.request,
        store_revision=_next_store_revision(previous.store_revision),
        advisory_wait=advisory_wait,
        resolution_digest=canonical_resolution_digest(resolution),
        resolved_by=resolver,
    )


def _reduce_interaction_admission_cleanup(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord:
    """Settle one admission after preserving temporal winner precedence."""
    _validate_pending_record(previous)
    _validate_temporal_context(observed_at, policy)
    deadline = _reduce_due_resolution(previous, observed_at, policy)
    if deadline is not None:
        return deadline
    lease_expiry = _reduce_expired_controller_lease(
        previous,
        observed_at,
        policy,
    )
    settlement_base = lease_expiry or previous
    resolution = UnavailableResolution(
        request_id=settlement_base.request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=observed_at.wall_time,
    )
    return _reduce_terminal_resolution(
        settlement_base,
        resolution,
        _ADMISSION_CLEANUP_RESOLVER,
        advisory_wait=settlement_base.advisory_wait,
    )


def _require_no_prior_settlement(
    previous: InteractionRecord,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> None:
    if (
        _due_resolution_status(previous, observed_at, policy) is not None
        or _reduce_expired_controller_lease(previous, observed_at, policy)
        is not None
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "observed_at",
            "deadline or lease-expiry settlement must apply first",
        )


def _validate_request_cancellation_command_identity(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_pending_record(previous)
    if not isinstance(command, CancelInteractionCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a cancel interaction command",
        )
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "cancellation command does not match the stored request",
        )


def _validate_request_cancellation_command_revision(
    previous: InteractionRecord,
    command: CancelInteractionCommand,
) -> None:
    if (
        command.expected_state_revision is not None
        and command.expected_state_revision != previous.request.state_revision
    ):
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "command.expected_state_revision",
            "request revision is stale",
        )


def _validate_request_cancellation_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_request_cancellation_command_identity(previous, command)
    assert isinstance(command, CancelInteractionCommand)
    _validate_request_cancellation_command_revision(previous, command)


def _validate_request_terminalization_command_identity(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_pending_record(previous)
    if not isinstance(command, TerminalizeInteractionCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a terminalize interaction command",
        )
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "terminalization command does not match the stored request",
        )


def _validate_request_terminalization_command_revision(
    previous: InteractionRecord,
    command: TerminalizeInteractionCommand,
) -> None:
    if (
        command.expected_state_revision is not None
        and command.expected_state_revision != previous.request.state_revision
    ):
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "command.expected_state_revision",
            "request revision is stale",
        )


def _validate_request_terminalization_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_request_terminalization_command_identity(previous, command)
    assert isinstance(command, TerminalizeInteractionCommand)
    _validate_request_terminalization_command_revision(previous, command)


def _reduce_request_cancellation(
    previous: InteractionRecord,
    command: CancelInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord:
    _validate_request_cancellation_command(previous, command)
    _validate_temporal_context(observed_at, policy)
    _require_no_prior_settlement(previous, observed_at, policy)
    resolution = CancelledResolution(
        request_id=previous.request.request_id,
        provenance=command.provenance,
        resolved_at=observed_at.wall_time,
        scope=CancellationScope.REQUEST,
    )
    return _reduce_terminal_resolution(
        previous,
        resolution,
        command.actor.principal,
        advisory_wait=previous.advisory_wait,
    )


def _reduce_request_terminalization(
    previous: InteractionRecord,
    command: TerminalizeInteractionCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> InteractionRecord:
    _validate_request_terminalization_command(previous, command)
    _validate_temporal_context(observed_at, policy)
    _require_no_prior_settlement(previous, observed_at, policy)
    if command.status is ResolutionStatus.UNAVAILABLE:
        resolution: InputResolution = UnavailableResolution(
            request_id=previous.request.request_id,
            provenance=command.provenance,
            resolved_at=observed_at.wall_time,
        )
    else:
        resolution = SupersededResolution(
            request_id=previous.request.request_id,
            provenance=command.provenance,
            resolved_at=observed_at.wall_time,
        )
    return _reduce_terminal_resolution(
        previous,
        resolution,
        command.actor.principal,
        advisory_wait=previous.advisory_wait,
    )


def _validate_store_generation(
    value: object,
) -> InteractionStoreGeneration:
    return InteractionStoreGeneration(
        validate_state_revision(value, "store_generation")
    )


def _record_selection_key(record: InteractionRecord) -> tuple[str, ...]:
    origin = record.request.origin
    return (
        str(origin.run_id),
        str(origin.turn_id),
        "" if origin.task_id is None else str(origin.task_id),
        str(origin.agent_id),
        str(origin.branch_id),
        str(record.request.request_id),
    )


def _branch_selection_key(
    record: InteractionBranchRecord,
) -> tuple[str, ...]:
    registration = record.registration
    principal = registration.principal
    return (
        str(registration.run_id),
        "" if principal.user_id is None else str(principal.user_id),
        "" if principal.tenant_id is None else str(principal.tenant_id),
        (
            ""
            if principal.participant_id is None
            else str(principal.participant_id)
        ),
        "" if principal.session_id is None else str(principal.session_id),
        str(registration.parent_branch_id),
        str(registration.branch_id),
    )


def _branch_ownership_selection_key(
    key: _InteractionBranchOwnershipKey,
) -> tuple[str, ...]:
    """Return a deterministic ordering key for one branch owner."""
    run_id, principal, branch_id = key
    return (
        str(run_id),
        "" if principal.user_id is None else str(principal.user_id),
        "" if principal.tenant_id is None else str(principal.tenant_id),
        (
            ""
            if principal.participant_id is None
            else str(principal.participant_id)
        ),
        "" if principal.session_id is None else str(principal.session_id),
        str(branch_id),
    )


def _branch_identity_key(
    registration: InteractionBranchRegistration,
) -> tuple[RunId, PrincipalScope, BranchId]:
    """Return one principal-scoped branch identity."""
    return (
        registration.run_id,
        registration.principal,
        registration.branch_id,
    )


def _resolve_interaction_branch_root(
    branch_records: tuple[InteractionBranchRecord, ...],
    query: InteractionBranchRootLookup,
) -> InteractionBranchRoot | None:
    """Resolve one same-run, same-principal persisted ancestry chain."""
    if not isinstance(query, InteractionBranchRootLookup):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "query",
            "value must be an interaction branch-root lookup",
        )
    if not isinstance(branch_records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "branch_records",
            "branch-root lookup requires an immutable branch snapshot",
        )
    registrations: dict[
        tuple[RunId, PrincipalScope, BranchId],
        InteractionBranchRegistration,
    ] = {}
    for record in branch_records:
        _validate_branch_record(record, "branch_records")
        registration = record.registration
        key = _branch_identity_key(registration)
        if key in registrations:
            return None
        registrations[key] = registration

    current = query.branch_id
    principal = query.actor.principal
    if (query.run_id, principal, current) not in registrations:
        return None
    seen: set[tuple[RunId, PrincipalScope, BranchId]] = set()
    while True:
        key = (query.run_id, principal, current)
        if key in seen:
            return None
        current_registration = registrations.get(key)
        if current_registration is None:
            return InteractionBranchRoot(
                run_id=query.run_id,
                branch_id=query.branch_id,
                root_branch_id=current,
            )
        seen.add(key)
        current = current_registration.parent_branch_id


def _derive_interaction_branch_closure_roots(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
) -> frozenset[_InteractionBranchOwnershipKey]:
    """Derive only terminal roots proven by one exact branch closure."""
    registrations = {
        _branch_identity_key(record.registration): record.registration
        for record in branch_records
    }
    if len(registrations) != len(branch_records):
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            "branch_records",
            "an exact branch closure cannot contain duplicate edges",
        )
    record_owners = {
        (
            record.request.origin.run_id,
            record.request.origin.principal,
            record.request.origin.branch_id,
        )
        for record in snapshot_records
    }
    visited: set[_InteractionBranchOwnershipKey] = set()
    authoritative_roots: set[_InteractionBranchOwnershipKey] = set()
    for record in snapshot_records:
        origin = record.request.origin
        if origin.parent_branch_id is None:
            continue
        current = (
            origin.run_id,
            origin.principal,
            origin.branch_id,
        )
        direct = registrations.get(current)
        if (
            direct is None
            or direct.parent_branch_id != origin.parent_branch_id
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "branch_records",
                "an exact branch closure lacks its selected record edge",
            )
        seen: set[_InteractionBranchOwnershipKey] = set()
        while current in registrations:
            if current in seen:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "branch_records",
                    "an exact branch closure cannot contain an ancestry cycle",
                )
            seen.add(current)
            visited.add(current)
            registration = registrations[current]
            current = (
                registration.run_id,
                registration.principal,
                registration.parent_branch_id,
            )
        if current not in record_owners:
            authoritative_roots.add(current)
    if visited != set(registrations):
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "branch_records",
            "an exact branch closure contains an unrelated edge",
        )
    return frozenset(authoritative_roots)


def _mint_interaction_branch_closure_attestation(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
) -> _InteractionBranchClosureAttestation:
    """Mint one sealed ownership attestation for an exact branch closure."""
    return _InteractionBranchClosureAttestation(
        _derive_interaction_branch_closure_roots(
            snapshot_records,
            branch_records,
        ),
        _token=_BRANCH_CLOSURE_ATTESTATION_TOKEN,
    )


def _validate_interaction_branch_closure_attestation(
    attestation: _InteractionBranchClosureAttestation,
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
) -> None:
    """Validate that a sealed closure attestation remains exact."""
    if type(attestation) is not _InteractionBranchClosureAttestation:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "branch_closure_attestation",
            "value must be a sealed branch closure attestation",
        )
    if attestation.authoritative_branch_roots != (
        _derive_interaction_branch_closure_roots(
            snapshot_records,
            branch_records,
        )
    ):
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "branch_closure_attestation",
            "branch closure roots changed within the sealed backing",
        )


def _validate_interaction_scope_ownership_attestation(
    attestation: _InteractionScopeOwnershipAttestation,
) -> None:
    """Validate one exact constructor-sealed ownership attestation."""
    if type(attestation) is not _InteractionScopeOwnershipAttestation:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "scope_ownership_attestation",
            "value must be a sealed scope ownership attestation",
        )
    if not isinstance(attestation.scope, InteractionExecutionScope):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "scope_ownership_attestation.scope",
            "value must be an interaction execution scope",
        )
    if not isinstance(attestation.principal, PrincipalScope):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "scope_ownership_attestation.principal",
            "value must be a principal scope",
        )
    validate_bool(
        attestation.actor_owned_record_match,
        "scope_ownership_attestation.actor_owned_record_match",
    )
    validate_bool(
        attestation.foreign_owned_record_match,
        "scope_ownership_attestation.foreign_owned_record_match",
    )
    validate_bool(
        attestation.actor_owned_branch_match,
        "scope_ownership_attestation.actor_owned_branch_match",
    )
    validate_bool(
        attestation.foreign_owned_branch_match,
        "scope_ownership_attestation.foreign_owned_branch_match",
    )


def _validate_scope_snapshot(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
) -> tuple[
    tuple[InteractionRecord, ...],
    tuple[InteractionBranchRecord, ...],
]:
    if not isinstance(snapshot_records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "snapshot_records",
            "scope snapshot records must be a tuple",
        )
    if not isinstance(branch_records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "branch_records",
            "scope branch graph must be a tuple",
        )
    request_ids: set[InputRequestId] = set()
    for record in snapshot_records:
        _validate_record(record, "snapshot_records")
        request_id = record.request.request_id
        if request_id in request_ids:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "snapshot_records",
                "scope snapshot request identifiers must be unique",
            )
        request_ids.add(request_id)
    branch_by_child: dict[
        tuple[RunId, PrincipalScope, BranchId],
        InteractionBranchRegistration,
    ] = {}
    for branch_record in branch_records:
        _validate_branch_record(branch_record, "branch_records")
        registration = branch_record.registration
        key = _branch_identity_key(registration)
        if key in branch_by_child:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "branch_records",
                "each run branch must have exactly one registered parent",
            )
        branch_by_child[key] = registration
    for key in branch_by_child:
        seen: set[tuple[RunId, PrincipalScope, BranchId]] = set()
        current = key
        while current in branch_by_child:
            if current in seen:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "branch_records",
                    "scope branch graph cannot contain an ancestry cycle",
                )
            seen.add(current)
            registration = branch_by_child[current]
            current = (
                registration.run_id,
                registration.principal,
                registration.parent_branch_id,
            )
    for record in snapshot_records:
        origin = record.request.origin
        record_registration = branch_by_child.get(
            (
                origin.run_id,
                origin.principal,
                origin.branch_id,
            )
        )
        if origin.parent_branch_id is None:
            if record_registration is not None:
                raise InputValidationError(
                    InputErrorCode.CORRELATION_MISMATCH,
                    "snapshot_records",
                    "request ancestry differs from its registered branch",
                )
            continue
        if (
            record_registration is None
            or record_registration.parent_branch_id != origin.parent_branch_id
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "snapshot_records",
                "request ancestry lacks its exact same-run branch edge",
            )
    return (
        tuple(sorted(snapshot_records, key=_record_selection_key)),
        tuple(sorted(branch_records, key=_branch_selection_key)),
    )


def _select_scope_records(
    snapshot_records: tuple[InteractionRecord, ...],
    scope: InteractionExecutionScope,
    branch_records: tuple[InteractionBranchRecord, ...],
    principal: PrincipalScope,
) -> tuple[InteractionRecord, ...]:
    if not isinstance(scope, InteractionExecutionScope):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "scope",
            "value must be an interaction execution scope",
        )
    descendants = _scope_descendant_branches(
        scope,
        branch_records,
        principal,
    )
    selected: list[InteractionRecord] = []
    for record in snapshot_records:
        origin = record.request.origin
        if origin.principal != principal or not _record_matches_scope(
            record,
            scope,
            descendants,
            pending_only=True,
        ):
            continue
        selected.append(record)
    return tuple(selected)


def _record_matches_scope(
    record: InteractionRecord,
    scope: InteractionExecutionScope,
    descendants: frozenset[BranchId],
    *,
    pending_only: bool,
) -> bool:
    """Return whether one record matches exact execution-scope semantics."""
    origin = record.request.origin
    return not (
        (
            pending_only
            and (
                record.request.state is not RequestState.PENDING
                or record.request.resolution is not None
            )
        )
        or origin.run_id != scope.run_id
        or (scope.turn_id is not None and origin.turn_id != scope.turn_id)
        or (scope.task_id is not None and origin.task_id != scope.task_id)
        or (scope.agent_id is not None and origin.agent_id != scope.agent_id)
        or (
            scope.branch_id is not None and origin.branch_id not in descendants
        )
    )


def _scope_ownership_presence(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
) -> tuple[bool, bool, bool, bool]:
    """Return content-free record and branch ownership for one scope."""
    descendants_by_principal: dict[PrincipalScope, frozenset[BranchId]] = {}
    actor_owned_record_match = False
    foreign_owned_record_match = False
    actor_owned_branch_match = False
    foreign_owned_branch_match = False
    for record in snapshot_records:
        owner = record.request.origin.principal
        descendants = descendants_by_principal.get(owner)
        if descendants is None:
            descendants = _scope_descendant_branches(
                scope,
                branch_records,
                owner,
            )
            descendants_by_principal[owner] = descendants
        if not _record_matches_scope(
            record,
            scope,
            descendants,
            pending_only=False,
        ):
            continue
        if owner == principal:
            actor_owned_record_match = True
        else:
            foreign_owned_record_match = True
    if scope.branch_id is not None:
        for branch_record in branch_records:
            registration = branch_record.registration
            owner = registration.principal
            descendants = descendants_by_principal.get(owner)
            if descendants is None:
                descendants = _scope_descendant_branches(
                    scope,
                    branch_records,
                    owner,
                )
                descendants_by_principal[owner] = descendants
            if (
                registration.run_id != scope.run_id
                or registration.branch_id not in descendants
            ):
                continue
            if owner == principal:
                actor_owned_branch_match = True
            else:
                foreign_owned_branch_match = True
    return (
        actor_owned_record_match,
        foreign_owned_record_match,
        actor_owned_branch_match,
        foreign_owned_branch_match,
    )


def _validate_scope_ownership_presence(
    snapshot: _InteractionStoreBackingSnapshot,
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
    selected_records: tuple[InteractionRecord, ...],
) -> None:
    """Reject a foreign-only scope while preserving empty-scope replay."""
    attestation = snapshot.scope_ownership_attestation
    if attestation is None:
        (
            actor_owned_record_match,
            foreign_owned_record_match,
            actor_owned_branch_match,
            foreign_owned_branch_match,
        ) = _scope_ownership_presence(
            snapshot.records,
            snapshot.branch_records,
            scope,
            principal,
        )
    else:
        _validate_interaction_scope_ownership_attestation(attestation)
        if attestation.scope != scope or attestation.principal != principal:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "scope_ownership_attestation",
                "scope ownership attestation does not match the command",
            )
        actor_owned_record_match = attestation.actor_owned_record_match
        foreign_owned_record_match = attestation.foreign_owned_record_match
        actor_owned_branch_match = attestation.actor_owned_branch_match
        foreign_owned_branch_match = attestation.foreign_owned_branch_match
    if selected_records and not actor_owned_record_match:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "scope_ownership_attestation",
            "scope ownership presence differs from selected records",
        )
    foreign_only_record_match = (
        not actor_owned_record_match and foreign_owned_record_match
    )
    foreign_only_branch_match = (
        not actor_owned_record_match
        and not foreign_owned_record_match
        and not actor_owned_branch_match
        and foreign_owned_branch_match
    )
    if not selected_records and (
        foreign_only_record_match or foreign_only_branch_match
    ):
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "scope",
            "matching scope ownership belongs to another principal",
        )


def _validate_scope_ownership(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    authoritative_branch_roots: frozenset[_InteractionBranchOwnershipKey],
    scope: InteractionExecutionScope,
    principal: PrincipalScope,
    selected_records: tuple[InteractionRecord, ...],
) -> None:
    owners: set[tuple[RunId, PrincipalScope, BranchId]] = set()
    registrations: dict[
        tuple[RunId, PrincipalScope, BranchId],
        InteractionBranchRegistration,
    ] = {}
    for record in snapshot_records:
        origin = record.request.origin
        owners.add((origin.run_id, origin.principal, origin.branch_id))
    for branch_record in branch_records:
        registration = branch_record.registration
        key = _branch_identity_key(registration)
        registrations[key] = registration
        owners.add(key)
    owners.update(authoritative_branch_roots)
    relevant: set[tuple[RunId, PrincipalScope, BranchId]] = set()
    if scope.branch_id is not None:
        relevant.update(
            (scope.run_id, principal, branch_id)
            for branch_id in _scope_descendant_branches(
                scope,
                branch_records,
                principal,
            )
        )
    for record in selected_records:
        origin = record.request.origin
        if origin.principal != principal:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "selected_records",
                "every selected record must belong to the scope actor",
            )
        current = (origin.run_id, principal, origin.branch_id)
        seen: set[_InteractionBranchOwnershipKey] = set()
        selected_branch = True
        while current not in seen:
            seen.add(current)
            relevant.add(current)
            current_registration = registrations.get(current)
            if current_registration is None:
                if selected_branch and origin.parent_branch_id is not None:
                    raise InputValidationError(
                        InputErrorCode.FORBIDDEN,
                        "branch_records",
                        "selected ancestry lacks its exact owner-bound edge",
                    )
                break
            selected_branch = False
            current = (
                current_registration.run_id,
                current_registration.principal,
                current_registration.parent_branch_id,
            )
    for key in relevant:
        if key not in owners:
            if selected_records:
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "branch_records",
                    "selected ancestry lacks authoritative ownership",
                )


def _validate_scope_transaction_commit(
    transaction: object,
    command: InteractionScopeMutationCommand,
    backing: _InteractionStoreBacking,
) -> None:
    if type(transaction) is not _InteractionScopeTransaction:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "transaction",
            "value must be a sealed interaction scope transaction",
        )
    assert isinstance(transaction, _InteractionScopeTransaction)
    _validate_store_backing(backing)
    if transaction._backing_capability is not backing._capability:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "backing",
            "scope transaction belongs to a different store backing",
        )
    if transaction.command != command:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "transaction.command",
            "scope transaction does not match the exact command",
        )
    snapshot = _snapshot_interaction_store_backing(backing)
    if transaction.store_generation != snapshot.store_generation:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "store_generation",
            "scope transaction is stale for the commit store generation",
        )
    authoritative_branch_roots = (
        frozenset()
        if snapshot.branch_closure_attestation is None
        else (snapshot.branch_closure_attestation.authoritative_branch_roots)
    )
    current_digest = _canonical_scope_snapshot_digest(
        snapshot.records,
        snapshot.branch_records,
        authoritative_branch_roots,
        snapshot.scope_ownership_attestation,
    )
    if transaction.snapshot_digest != current_digest:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "snapshot_records",
            "authoritative scope snapshot changed before commit",
        )
    expected = _select_scope_records(
        snapshot.records,
        command.scope,
        snapshot.branch_records,
        command.actor.principal,
    )
    _validate_scope_ownership_presence(
        snapshot,
        command.scope,
        command.actor.principal,
        expected,
    )
    _validate_scope_ownership(
        snapshot.records,
        snapshot.branch_records,
        authoritative_branch_roots,
        command.scope,
        command.actor.principal,
        expected,
    )
    if (
        transaction.scope != command.scope
        or transaction.principal != command.actor.principal
        or expected != transaction.selected_records
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "transaction",
            "scope transaction differs from its locked full snapshot",
        )


def _validate_store_backing(backing: object) -> None:
    if type(backing) is not _InteractionStoreBacking:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "backing",
            "value must be a store-owned backing capability",
        )


def _canonical_scope_snapshot_digest(
    snapshot_records: tuple[InteractionRecord, ...],
    branch_records: tuple[InteractionBranchRecord, ...],
    authoritative_branch_roots: frozenset[_InteractionBranchOwnershipKey],
    scope_ownership_attestation: _InteractionScopeOwnershipAttestation | None,
) -> str:
    payload = {
        "authoritative_branch_roots": _canonical_scope_snapshot_value(
            tuple(
                sorted(
                    authoritative_branch_roots,
                    key=_branch_ownership_selection_key,
                )
            )
        ),
        "branch_records": _canonical_scope_snapshot_value(branch_records),
        "scope_ownership_attestation": _canonical_scope_snapshot_value(
            scope_ownership_attestation
        ),
        "snapshot_records": _canonical_scope_snapshot_value(snapshot_records),
    }
    encoded = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_scope_snapshot_value(value: object) -> object:
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "fields": [
                [
                    item.name,
                    _canonical_scope_snapshot_value(getattr(value, item.name)),
                ]
                for item in fields(value)
            ],
        }
    if isinstance(value, Mapping):
        items = [
            [
                _canonical_scope_snapshot_value(key),
                _canonical_scope_snapshot_value(value[key]),
            ]
            for key in value
        ]
        items.sort(
            key=lambda item: dumps(
                item[0],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return {"mapping": items}
    if isinstance(value, tuple):
        return {
            "tuple": [_canonical_scope_snapshot_value(item) for item in value]
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "snapshot_records",
        "scope snapshot contains a non-canonical value",
    )


def _scope_descendant_branches(
    scope: InteractionExecutionScope,
    branch_records: tuple[InteractionBranchRecord, ...],
    principal: PrincipalScope,
) -> frozenset[BranchId]:
    if scope.branch_id is None:
        return frozenset()
    allowed = {scope.branch_id}
    if not scope.include_descendants:
        return frozenset(allowed)
    changed = True
    while changed:
        changed = False
        for record in branch_records:
            registration = record.registration
            if (
                registration.run_id == scope.run_id
                and registration.principal == principal
                and registration.parent_branch_id in allowed
                and registration.branch_id not in allowed
            ):
                allowed.add(registration.branch_id)
                changed = True
    return frozenset(allowed)


def _reduce_scope_cancellation(
    previous: tuple[InteractionRecord, ...],
    command: TerminalizeInteractionScopeCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> tuple[InteractionRecord, ...]:
    if not isinstance(command, TerminalizeInteractionScopeCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a scope cancellation command",
        )
    _validate_temporal_context(observed_at, policy)
    if not isinstance(previous, tuple) or not previous:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "previous",
            "scope cancellation requires a non-empty sealed selection",
        )
    records: list[InteractionRecord] = []
    for record in previous:
        _validate_pending_record(record)
        due = _reduce_due_resolution(record, observed_at, policy)
        if due is None:
            due = _reduce_expired_controller_lease(
                record,
                observed_at,
                policy,
            )
        if due is not None:
            records.append(due)
            continue
        resolution = CancelledResolution(
            request_id=record.request.request_id,
            provenance=command.provenance,
            resolved_at=observed_at.wall_time,
            scope=CancellationScope.CONTAINING_RUN,
        )
        records.append(
            _reduce_terminal_resolution(
                record,
                resolution,
                command.actor.principal,
                advisory_wait=record.advisory_wait,
            )
        )
    return tuple(records)


def _reduce_scope_supersession(
    previous: tuple[InteractionRecord, ...],
    command: SupersedeInteractionScopeCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> tuple[InteractionRecord, ...]:
    if not isinstance(command, SupersedeInteractionScopeCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a scope supersession command",
        )
    _validate_temporal_context(observed_at, policy)
    if not isinstance(previous, tuple) or not previous:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "previous",
            "scope supersession requires a non-empty sealed selection",
        )
    records: list[InteractionRecord] = []
    for record in previous:
        _validate_pending_record(record)
        due = _reduce_due_resolution(record, observed_at, policy)
        if due is None:
            due = _reduce_expired_controller_lease(
                record,
                observed_at,
                policy,
            )
        if due is not None:
            records.append(due)
            continue
        resolution = SupersededResolution(
            request_id=record.request.request_id,
            provenance=command.provenance,
            resolved_at=observed_at.wall_time,
        )
        records.append(
            _reduce_terminal_resolution(
                record,
                resolution,
                command.actor.principal,
                advisory_wait=record.advisory_wait,
            )
        )
    return tuple(records)


def _reduce_due_interactions(
    previous: tuple[InteractionRecord, ...],
    command: TerminalizeDueInteractionsCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> tuple[InteractionRecord, ...]:
    if not isinstance(command, TerminalizeDueInteractionsCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a terminalize-due command",
        )
    if not isinstance(previous, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "previous",
            "due-settlement candidates must be a tuple",
        )
    _validate_temporal_context(observed_at, policy)
    request_ids: set[InputRequestId] = set()
    candidates: list[InteractionRecord] = []
    for record in previous:
        _validate_record(record, "previous")
        if record.request.request_id in request_ids:
            raise InputValidationError(
                InputErrorCode.DUPLICATE,
                "previous",
                "due-settlement candidates must be unique",
            )
        request_ids.add(record.request.request_id)
        if record.request.state is RequestState.PENDING:
            candidates.append(record)
    candidates.sort(
        key=lambda record: (
            _effective_interaction_deadline(record, observed_at),
            str(record.request.request_id),
        )
    )
    records: list[InteractionRecord] = []
    for record in candidates:
        due = _reduce_due_resolution(record, observed_at, policy)
        if due is None:
            due = _reduce_expired_controller_lease(
                record,
                observed_at,
                policy,
            )
        if due is not None:
            records.append(due)
        if len(records) == command.maximum_results:
            break
    return tuple(records)


def _require_exact_result_record(
    record: InteractionRecord,
    expected: InteractionRecord,
    operation: str,
) -> None:
    _validate_record(record, "result.record")
    if record != expected:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.record",
            f"{operation} result changed fields outside its exact boundary",
        )


def _require_exact_result_records(
    records: tuple[InteractionRecord, ...],
    expected: tuple[InteractionRecord, ...],
    operation: str,
) -> None:
    if not isinstance(records, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "result.records",
            "scope result records must be a tuple",
        )
    if records != expected:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.records",
            f"scope {operation} must preserve every unrelated field",
        )


def validate_interaction_presentation_transition(
    previous: InteractionRecord,
    record: InteractionRecord,
    command: InteractionPresentationCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
) -> None:
    """Validate one exact presentation-only store mutation."""
    _validate_temporal_context(observed_at, policy)
    expected = _reduce_interaction_presentation(
        previous,
        command,
        observed_at,
        policy,
    )
    if record != expected:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.record",
            "presentation mutated fields outside its exact metadata boundary",
        )


def validate_controller_activity_transition(
    previous: InteractionRecord,
    record: InteractionRecord,
    command: RecordControllerActivityCommand,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    lease_nonce: ActiveControlLeaseNonce | None = None,
) -> None:
    """Validate one exact active-control lease metadata mutation."""
    _validate_temporal_context(observed_at, policy)
    expected = _reduce_controller_activity(
        previous,
        command,
        observed_at,
        policy,
        lease_nonce=lease_nonce,
    )
    if record != expected:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.record",
            "controller activity mutated outside its exact timed envelope",
        )


def validate_resolution_commit_transition(
    previous: InteractionRecord,
    record: InteractionRecord,
    decision_stage: ResolutionDecisionStage,
    observed_at: InteractionTime,
    policy: InteractionPolicy,
    *,
    command: InteractionDeadlineTriggerCommand | None,
    idempotency_key: ResolutionIdempotencyKey | None,
    classifier_binding: _TaskInputClassifierBinding | None = None,
    classification_proof: _BoundTaskInputClassifications | None = None,
) -> None:
    """Validate one exact candidate or deadline resolution commit."""
    _validate_temporal_context(observed_at, policy)
    if decision_stage not in {
        ResolutionDecisionStage.DEADLINE,
        ResolutionDecisionStage.COMMIT,
    }:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "result.decision_stage",
            "an applied resolution must be a deadline or candidate commit",
        )
    if decision_stage is ResolutionDecisionStage.COMMIT:
        command = _validate_candidate_resolution_command(command)
        expected = _reduce_candidate_resolution(
            previous,
            command,
            observed_at,
            policy,
            classifier_binding,
            classification_proof,
        )
        if idempotency_key != command.idempotency_key:
            raise InputValidationError(
                InputErrorCode.IDEMPOTENCY_CONFLICT,
                "result.idempotency_key",
                "candidate result key must match its resolve command",
            )
    else:
        if command is not None:
            _validate_deadline_trigger_command(previous, command)
        if idempotency_key is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "result.idempotency_key",
                "deadline settlement cannot bind a caller transport key",
            )
        if classifier_binding is not None or classification_proof is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "result.classification_proof",
                "deadline settlement cannot bind candidate classification",
            )
        due = _reduce_due_resolution(
            previous,
            observed_at,
            policy,
        )
        if due is None:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "result.record",
                "deadline result requires a due operational deadline",
            )
        expected = due
    if record != expected:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.record",
            "resolution result mutated outside its exact timed envelope",
        )


def _next_store_revision(
    revision: InteractionStoreRevision,
) -> InteractionStoreRevision:
    if revision == MAX_STATE_REVISION:
        raise InputValidationError(
            InputErrorCode.STATE_REVISION_EXHAUSTED,
            "result.previous.store_revision",
            "store revision is exhausted",
        )
    return InteractionStoreRevision(revision + 1)


def _validate_existing_controller_lease(
    previous: AdvisoryWaitState,
    evidence: ControllerActivityEvidence,
) -> None:
    if not isinstance(
        evidence,
        (
            PulseControllerActivity,
            ReleaseControllerActivity,
            DisconnectControllerActivity,
        ),
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "result.evidence",
            "existing lease action requires sequenced evidence",
        )
    if (
        previous.status is not AdvisoryWaitStatus.PAUSED
        or previous.controller_id != evidence.controller_id
        or previous.lease_nonce != evidence.lease_nonce
        or previous.activity_sequence is None
        or previous.activity_sequence == MAX_STATE_REVISION
        or evidence.sequence != previous.activity_sequence + 1
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "result.evidence",
            "controller, nonce, and strict next sequence must match the lease",
        )


def _is_candidate_resolution_record(record: InteractionRecord) -> bool:
    resolution = record.request.resolution
    return (
        _is_input_candidate_resolution(resolution)
        and resolution is not None
        and (
            (
                resolution.provenance is AnswerProvenance.POLICY
                and record.resolved_by is _TRUSTED_POLICY_RESOLVER
            )
            or (
                resolution.provenance
                not in {
                    AnswerProvenance.POLICY,
                    AnswerProvenance.TRUSTED_DEFAULT,
                }
                and isinstance(record.resolved_by, PrincipalScope)
            )
        )
        and record.resolution_digest is not None
        and bool(record.idempotency_ledger)
        and all(
            entry.resolution_digest == record.resolution_digest
            for entry in record.idempotency_ledger
        )
    )


def _validate_sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a lowercase SHA-256 digest",
        )
    return value


def _validate_advisory_wait_state(state: AdvisoryWaitState) -> None:
    valid = False
    control_values = (
        state.controller_id,
        state.lease_nonce,
        state.activity_sequence,
        state.lease_expires_at_monotonic,
    )
    match state.status:
        case AdvisoryWaitStatus.QUEUED:
            valid = (
                state.presented_at is None
                and state.running_since_monotonic is None
                and state.remaining_seconds == state.budget_seconds
                and all(value is None for value in control_values)
            )
        case AdvisoryWaitStatus.RUNNING:
            valid = (
                state.presented_at is not None
                and state.running_since_monotonic is not None
                and state.remaining_seconds > 0
                and all(value is None for value in control_values)
            )
        case AdvisoryWaitStatus.PAUSED:
            valid = (
                state.presented_at is not None
                and state.running_since_monotonic is None
                and state.remaining_seconds > 0
                and all(value is not None for value in control_values)
            )
        case AdvisoryWaitStatus.EXHAUSTED:
            valid = (
                state.presented_at is not None
                and state.running_since_monotonic is None
                and state.remaining_seconds == 0
                and all(value is None for value in control_values)
            )
    if not valid:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "advisory",
            "advisory wait fields do not match their status",
        )


def _validate_record_advisory(record: InteractionRecord) -> None:
    if record.request.mode is RequirementMode.REQUIRED:
        if record.advisory_wait is not None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "record.advisory_wait",
                "required requests cannot store advisory timing",
            )
        return
    if not isinstance(record.advisory_wait, AdvisoryWaitState):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "record.advisory_wait",
            "advisory requests require advisory timing state",
        )
    if (
        record.advisory_wait.budget_seconds
        != record.request.advisory_wait_seconds
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.advisory_wait",
            "advisory timing must match the canonical request budget",
        )
    wait = record.advisory_wait
    queued = (
        wait.status is AdvisoryWaitStatus.QUEUED
        and wait.presented_at is None
        and record.request.advisory_deadline is None
    )
    anchored = (
        wait.status
        in {
            AdvisoryWaitStatus.RUNNING,
            AdvisoryWaitStatus.PAUSED,
            AdvisoryWaitStatus.EXHAUSTED,
        }
        and wait.presented_at is not None
        and record.request.advisory_deadline is not None
    )
    valid_presentation = (
        (record.presentation is InteractionPresentationState.QUEUED and queued)
        or (
            record.presentation is InteractionPresentationState.PRESENTED
            and anchored
        )
        or (
            record.presentation is InteractionPresentationState.DETACHED
            and (queued or anchored)
        )
    )
    if not valid_presentation:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.presentation",
            "presentation and advisory timing states do not agree",
        )
    timed_out = record.request.state is RequestState.TIMED_OUT
    if (wait.status is AdvisoryWaitStatus.EXHAUSTED) != timed_out:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.advisory_wait.status",
            "only advisory timeout can exhaust the inactivity budget",
        )
    if anchored:
        assert wait.presented_at is not None
        expected = wait.presented_at + timedelta(seconds=wait.budget_seconds)
        if record.request.advisory_deadline != expected:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "record.advisory_wait",
                "canonical deadline must anchor to actual presentation",
            )


def _validate_record_resolution(record: InteractionRecord) -> None:
    resolution = record.request.resolution
    if resolution is None:
        if (
            record.resolution_digest is not None
            or record.resolved_by is not None
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "record.resolution",
                "pending records cannot store terminal resolution metadata",
            )
        return
    if record.resolution_digest is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.resolution_digest",
            "terminal records require a semantic resolution digest",
        )
    _validate_sha256(record.resolution_digest, "record.resolution_digest")
    if record.resolution_digest != canonical_resolution_digest(resolution):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.resolution_digest",
            "resolution digest does not match the stored outcome",
        )
    if isinstance(resolution, (TimedOutResolution, ExpiredResolution)):
        if (
            record.resolved_by is not _DEADLINE_RESOLVER
            or resolution.provenance is not AnswerProvenance.POLICY
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "record.resolved_by",
                "deadline outcomes require sealed policy authority",
            )
        if isinstance(resolution, TimedOutResolution):
            if (
                record.request.advisory_deadline is None
                or record.presentation is InteractionPresentationState.QUEUED
            ):
                raise InputValidationError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "record.resolution",
                    "timeout requires a presented advisory request",
                )
        elif resolution.resolved_at < record.absolute_expires_at:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "record.resolution.resolved_at",
                "expiry predates the absolute continuation deadline",
            )
    elif _is_input_candidate_resolution(resolution):
        if resolution.provenance is AnswerProvenance.TRUSTED_DEFAULT:
            if record.resolved_by is not _TRUSTED_DEFAULT_RESOLVER:
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "record.resolved_by",
                    "trusted defaults require sealed policy authority",
                )
            if isinstance(resolution, AnsweredResolution) and any(
                answer.provenance is not AnswerProvenance.TRUSTED_DEFAULT
                for answer in resolution.answers
            ):
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "record.resolution.answers",
                    "trusted-default answers require trusted provenance",
                )
        elif resolution.provenance is AnswerProvenance.POLICY:
            if record.resolved_by is not _TRUSTED_POLICY_RESOLVER:
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "record.resolved_by",
                    "policy candidates require sealed policy authority",
                )
            if isinstance(resolution, AnsweredResolution) and any(
                answer.provenance is not AnswerProvenance.POLICY
                for answer in resolution.answers
            ):
                raise InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "record.resolution.answers",
                    "policy answers require policy provenance",
                )
        elif not isinstance(record.resolved_by, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "record.resolved_by",
                "external candidate outcomes require principal authority",
            )
        elif isinstance(resolution, AnsweredResolution) and any(
            answer.provenance
            in {
                AnswerProvenance.TRUSTED_DEFAULT,
                AnswerProvenance.POLICY,
            }
            for answer in resolution.answers
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "record.resolution.answers",
                "external answers cannot claim trusted provenance",
            )
    elif (
        isinstance(resolution, UnavailableResolution)
        and record.resolved_by is _ADMISSION_CLEANUP_RESOLVER
    ):
        if resolution.provenance is not AnswerProvenance.POLICY:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "record.resolution.provenance",
                "admission cleanup requires sealed policy provenance",
            )
    elif not isinstance(record.resolved_by, PrincipalScope):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "record.resolved_by",
            "non-deadline outcomes require principal authority",
        )


def _validate_idempotency_ledger(record: InteractionRecord) -> None:
    if not isinstance(record.idempotency_ledger, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "record.idempotency_ledger",
            "idempotency ledger must be a tuple",
        )
    if (
        len(record.idempotency_ledger)
        > MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST
    ):
        raise InputValidationError(
            InputErrorCode.IDEMPOTENCY_LEDGER_FULL,
            "record.idempotency_ledger",
            "idempotency ledger exceeds its fixed request bound",
        )
    if not all(
        isinstance(entry, ResolutionIdempotencyEntry)
        for entry in record.idempotency_ledger
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "record.idempotency_ledger",
            "ledger entries must be resolution idempotency entries",
        )
    keys = tuple(entry.key for entry in record.idempotency_ledger)
    if len(keys) != len(set(keys)):
        raise InputValidationError(
            InputErrorCode.DUPLICATE,
            "record.idempotency_ledger",
            "idempotency keys must be unique per request",
        )
    if record.resolution_digest is None and record.idempotency_ledger:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.idempotency_ledger",
            "unresolved records cannot retain accepted resolution keys",
        )
    resolution = record.request.resolution
    external_candidate = (
        resolution is not None
        and _is_input_candidate_resolution(resolution)
        and resolution.provenance is not AnswerProvenance.TRUSTED_DEFAULT
    )
    trusted_default = (
        resolution is not None
        and _is_input_candidate_resolution(resolution)
        and resolution.provenance is AnswerProvenance.TRUSTED_DEFAULT
    )
    if external_candidate and not record.idempotency_ledger:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.idempotency_ledger",
            "candidate outcomes require their accepted caller key binding",
        )
    if (
        resolution is not None
        and (not _is_input_candidate_resolution(resolution) or trusted_default)
        and record.idempotency_ledger
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "record.idempotency_ledger",
            "system, cancellation, and terminalization outcomes reject keys",
        )
    if record.resolution_digest is not None and any(
        entry.resolution_digest != record.resolution_digest
        for entry in record.idempotency_ledger
    ):
        raise InputValidationError(
            InputErrorCode.IDEMPOTENCY_CONFLICT,
            "record.idempotency_ledger",
            "accepted keys must bind to the terminal semantic digest",
        )


def _validate_terminalization_fields(
    status: object,
    provenance: object,
) -> None:
    if status not in {
        ResolutionStatus.UNAVAILABLE,
        ResolutionStatus.SUPERSEDED,
    }:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "terminalize.status",
            "direct terminalization must be unavailable or superseded",
        )
    _validate_principal_authored_provenance(
        provenance,
        "terminalize.provenance",
    )


def _validate_principal_authored_provenance(
    provenance: object,
    path: str,
) -> None:
    if not isinstance(provenance, AnswerProvenance):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be answer provenance",
        )
    if provenance not in {
        AnswerProvenance.HUMAN,
        AnswerProvenance.EXTERNAL_CONTROLLER,
    }:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            path,
            "principal-authored commands cannot claim trusted provenance",
        )


def _validate_optional_revision(
    owner: object,
    value: object,
    path: str,
) -> None:
    if value is not None:
        object.__setattr__(
            owner,
            "expected_state_revision",
            StateRevision(validate_state_revision(value, path)),
        )


def _validate_actor(actor: object) -> None:
    if not isinstance(actor, InteractionActor):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "actor",
            "value must be an interaction actor",
        )


def _validate_presentation_command_identity(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_record(previous, "previous")
    if not isinstance(
        command,
        (PresentInteractionCommand, DetachInteractionCommand),
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a presentation or detachment command",
        )
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "presentation command does not match the stored request",
        )
    if command.actor.principal != previous.request.origin.principal:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "command.actor",
            "presentation actor differs from the request principal",
        )


def _validate_presentation_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_presentation_command_identity(previous, command)
    assert isinstance(
        command,
        (PresentInteractionCommand, DetachInteractionCommand),
    )
    if command.expected_store_revision != previous.store_revision:
        raise InputValidationError(
            InputErrorCode.STALE_REVISION,
            "command.expected_store_revision",
            "interaction store revision is stale",
        )


def _validate_controller_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    _validate_record(previous, "previous")
    if not isinstance(command, RecordControllerActivityCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "command",
            "value must be a controller activity command",
        )
    if command.correlation != previous.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.correlation",
            "controller command does not match the stored request",
        )
    if command.actor.principal != previous.request.origin.principal:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "command.actor",
            "controller actor differs from the request principal",
        )
    if command.evidence.request_id != previous.request.request_id:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "command.evidence.request_id",
            "controller evidence does not match the stored request",
        )


def _validate_deadline_trigger_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    if isinstance(
        command,
        (ResolveInteractionCommand, _TrustedPolicyResolutionCommand),
    ):
        _validate_candidate_resolution_command(command)
        if command.correlation != previous.correlation:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "result.command.correlation",
                "deadline-triggering command does not match the request",
            )
        return
    if isinstance(command, TrustedDefaultResolutionCommand):
        _validate_trusted_default_command_identity(previous, command)
        return
    if isinstance(
        command,
        (PresentInteractionCommand, DetachInteractionCommand),
    ):
        _validate_presentation_command_identity(previous, command)
        return
    if isinstance(command, RecordControllerActivityCommand):
        _validate_controller_command(previous, command)
        return
    if isinstance(command, CancelInteractionCommand):
        _validate_request_cancellation_command_identity(previous, command)
        return
    if isinstance(command, TerminalizeInteractionCommand):
        _validate_request_terminalization_command_identity(previous, command)
        return
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "result.command",
        "value must be a supported deadline-triggering command",
    )


def _validate_lease_expiry_trigger_command(
    previous: InteractionRecord,
    command: object,
) -> None:
    if isinstance(
        command,
        (ResolveInteractionCommand, _TrustedPolicyResolutionCommand),
    ):
        _validate_candidate_resolution_command(command)
        if command.correlation != previous.correlation:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "result.command.correlation",
                "lease-expiry command does not match the request",
            )
        return
    if isinstance(command, TrustedDefaultResolutionCommand):
        _validate_trusted_default_command_identity(previous, command)
        return
    if isinstance(command, DetachInteractionCommand):
        _validate_presentation_command_identity(previous, command)
        return
    if isinstance(command, RecordControllerActivityCommand):
        _validate_controller_command(previous, command)
        return
    if isinstance(command, CancelInteractionCommand):
        _validate_request_cancellation_command_identity(previous, command)
        return
    if isinstance(command, TerminalizeInteractionCommand):
        _validate_request_terminalization_command_identity(previous, command)
        return
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "result.command",
        "value must be a supported lease-expiry command",
    )


def _validate_replay_command(
    record: InteractionRecord,
    command: _CandidateResolutionCommand,
) -> None:
    if command.correlation != record.correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "result.command.correlation",
            "replay command does not match the terminal request",
        )
    if command.resolution_digest != record.resolution_digest:
        raise InputValidationError(
            InputErrorCode.IDEMPOTENCY_CONFLICT,
            "result.command.proposed_resolution",
            "replay command differs from the terminal candidate",
        )
    if isinstance(command, _TrustedPolicyResolutionCommand):
        if record.resolved_by is not _TRUSTED_POLICY_RESOLVER:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result.command.actor",
                "policy replay lacks sealed resolver authority",
            )
        if command.actor.principal != record.request.origin.principal:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "result.command.actor",
                "policy replay actor differs from the request principal",
            )
    elif command.actor.principal != record.resolved_by:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "result.command.actor",
            "replay actor differs from the original candidate resolver",
        )


def _validate_branch_result_command(
    record: InteractionBranchRecord,
    command: object,
) -> None:
    if not isinstance(command, RegisterInteractionBranchCommand):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "result.command",
            "value must be a branch registration command",
        )
    if command.registration != record.registration:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "result.command.registration",
            "branch result does not match the exact registered edge",
        )
    if command.actor.principal != command.registration.principal:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "result.command.actor",
            "branch result actor differs from its authorization target",
        )


def _validate_correlation(correlation: object) -> None:
    if not isinstance(correlation, InteractionCorrelation):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "correlation",
            "value must be interaction correlation",
        )


def _validate_record(value: object, path: str) -> None:
    if not isinstance(value, InteractionRecord):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be an interaction record",
        )


def _validate_branch_record(value: object, path: str) -> None:
    if not isinstance(value, InteractionBranchRecord):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be an interaction branch record",
        )


def _validate_semantic_replay_binding(
    previous: InteractionRecord,
    record: InteractionRecord,
    command: _CandidateResolutionCommand,
) -> None:
    _validate_record(previous, "result.previous")
    if not _is_candidate_resolution_record(previous):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "result.previous",
            "semantic replay requires an original candidate commit",
        )
    _validate_replay_command(previous, command)
    if any(
        entry.key == command.idempotency_key
        for entry in previous.idempotency_ledger
    ):
        raise InputValidationError(
            InputErrorCode.IDEMPOTENCY_CONFLICT,
            "result.command.idempotency_key",
            "semantic new-key replay requires an absent transport key",
        )
    expected = replace(
        previous,
        store_revision=_next_store_revision(previous.store_revision),
        idempotency_ledger=previous.idempotency_ledger
        + (
            ResolutionIdempotencyEntry(
                key=command.idempotency_key,
                resolution_digest=command.resolution_digest,
            ),
        ),
    )
    if record != expected:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "result.record",
            "semantic replay may mutate only its transport-key ledger",
        )
