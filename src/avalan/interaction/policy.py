"""Define trusted async policy boundaries for interaction runtimes."""

from .entities import (
    ActiveControlLeaseNonce,
    AgentId,
    BranchId,
    ContinuationId,
    ControllerId,
    ExecutionOrigin,
    InputRequestId,
    PrincipalScope,
    QuestionId,
    QuestionType,
    ResolutionIdempotencyKey,
    RunId,
    TaskId,
    TurnId,
)
from .error import InputErrorCode, InputValidationError
from .validation import (
    validate_aware_datetime,
    validate_bool,
    validate_finite_number,
    validate_int,
    validate_opaque_id,
)

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias, final

MAX_RESOLUTION_IDEMPOTENCY_KEY_CHARACTERS = 256
MAX_RESOLUTION_IDEMPOTENCY_KEY_BYTES = 1_024
MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST = 32
MAX_UNRESOLVED_INTERACTIONS_PER_RUN = 8
MAX_UNRESOLVED_REQUIRED_INTERACTIONS_PER_BRANCH = 1
MAX_EQUIVALENT_INTERACTIONS_PER_BRANCH = 3
MAX_PENDING_INTERACTIONS_PER_PROCESS = 1_024


class InteractionOperation(StrEnum):
    """Identify one broker operation subject to authorization."""

    CREATE = "create"
    DELIVER = "deliver"
    WAIT = "wait"
    INSPECT = "inspect"
    INSPECT_BRANCH = "inspect_branch"
    LIST = "list"
    RESOLVE = "resolve"
    TRUSTED_DEFAULT = "trusted_default"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_SCOPE = "cancel_scope"
    EXPIRE = "expire"
    SUPERSEDE = "supersede"
    REGISTER_BRANCH = "register_branch"
    RECORD_ACTIVITY = "record_activity"


class InteractionDisclosure(StrEnum):
    """Limit interaction content returned to an authorized actor."""

    NONE = "none"
    TERMINAL_METADATA = "terminal_metadata"
    FULL = "full"


class TaskInputClassificationDecision(StrEnum):
    """Identify the trusted submitted-value classification result."""

    ALLOW = "allow"
    REJECT_SECRET = "reject_secret"


class HandlerLossDisposition(StrEnum):
    """Select deterministic behavior after attached capability disappears."""

    DETACH = "detach"
    UNAVAILABLE = "unavailable"
    CANCEL_REQUEST = "cancel_request"


class DeadlineTiePolicy(StrEnum):
    """Define precedence when a submission arrives exactly at a deadline."""

    DEADLINE_FIRST = "deadline_first"


class InteractionSettlement(StrEnum):
    """Identify the winner of one same-observation settlement race."""

    ABSOLUTE_EXPIRY = "absolute_expiry"
    ADVISORY_TIMEOUT = "advisory_timeout"
    CANDIDATE_RESOLUTION = "candidate_resolution"


INTERACTION_SETTLEMENT_PRECEDENCE: tuple[InteractionSettlement, ...] = (
    InteractionSettlement.ABSOLUTE_EXPIRY,
    InteractionSettlement.ADVISORY_TIMEOUT,
    InteractionSettlement.CANDIDATE_RESOLUTION,
)


class ControllerActivityAction(StrEnum):
    """Identify one authenticated active-control lease operation."""

    ACQUIRE = "acquire"
    PULSE = "pulse"
    RELEASE = "release"
    DISCONNECT = "disconnect"


def select_interaction_settlement(
    *,
    absolute_expiry_due: bool,
    advisory_timeout_due: bool,
    candidate_resolution_present: bool,
) -> InteractionSettlement | None:
    """Return the authoritative winner for one atomic clock observation."""
    for value, settlement in (
        (absolute_expiry_due, InteractionSettlement.ABSOLUTE_EXPIRY),
        (advisory_timeout_due, InteractionSettlement.ADVISORY_TIMEOUT),
        (
            candidate_resolution_present,
            InteractionSettlement.CANDIDATE_RESOLUTION,
        ),
    ):
        validate_bool(value, f"settlement.{settlement.value}")
        if value:
            return settlement
    return None


def validate_resolution_idempotency_key(
    value: object,
    path: str = "idempotency_key",
) -> ResolutionIdempotencyKey:
    """Return one bounded opaque resolution idempotency key."""
    return ResolutionIdempotencyKey(
        validate_opaque_id(
            value,
            path,
            maximum_characters=MAX_RESOLUTION_IDEMPOTENCY_KEY_CHARACTERS,
            maximum_bytes=MAX_RESOLUTION_IDEMPOTENCY_KEY_BYTES,
        )
    )


_CLOCK_OBSERVATION_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class InteractionTime:
    """Carry one internally minted coherent clock observation."""

    wall_time: datetime
    monotonic_seconds: float

    def __init__(
        self,
        *,
        wall_time: datetime,
        monotonic_seconds: float,
        _token: object,
    ) -> None:
        if _token is not _CLOCK_OBSERVATION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "time",
                "clock observations must be minted by the trusted clock",
            )
        object.__setattr__(
            self,
            "wall_time",
            validate_aware_datetime(wall_time, "time.wall_time"),
        )
        monotonic_seconds = validate_finite_number(
            monotonic_seconds,
            "time.monotonic_seconds",
        )
        if monotonic_seconds < 0:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "time.monotonic_seconds",
                "monotonic time must be non-negative",
            )
        object.__setattr__(self, "monotonic_seconds", float(monotonic_seconds))

    @classmethod
    def from_clock(
        cls,
        *,
        wall_time: datetime,
        monotonic_seconds: float,
    ) -> "InteractionTime":
        """Return one observation minted inside a trusted clock adapter."""
        return cls(
            wall_time=wall_time,
            monotonic_seconds=monotonic_seconds,
            _token=_CLOCK_OBSERVATION_TOKEN,
        )

    def monotonic_deadline_reached(self, deadline: float) -> bool:
        """Return whether a monotonic deadline is due, including equality."""
        normalized = float(
            validate_finite_number(deadline, "deadline.monotonic_seconds")
        )
        if normalized < 0:
            raise InputValidationError(
                InputErrorCode.OUT_OF_BOUNDS,
                "deadline.monotonic_seconds",
                "monotonic deadline must be non-negative",
            )
        return self.monotonic_seconds >= normalized

    def wall_deadline_reached(self, deadline: datetime) -> bool:
        """Return whether a wall deadline is due, including equality."""
        normalized = validate_aware_datetime(deadline, "deadline.wall_time")
        return self.wall_time >= normalized


class InteractionClock(Protocol):
    """Provide trusted time and cancellable deadline waiting."""

    async def read(self) -> InteractionTime:
        """Return one coherent trusted clock observation."""
        ...

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait until the monotonic deadline or task cancellation."""
        ...


class InteractionIdFactory(Protocol):
    """Mint runtime-owned opaque interaction identities."""

    async def new_request_id(self) -> InputRequestId:
        """Return a new opaque request identifier."""
        ...

    async def new_continuation_id(self) -> ContinuationId:
        """Return a new opaque continuation identifier."""
        ...

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return a new opaque resolution idempotency key."""
        ...

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return a new server-owned active-control lease nonce."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionActor:
    """Carry principal scope resolved by a trusted host boundary."""

    principal: PrincipalScope

    def __post_init__(self) -> None:
        if not isinstance(self.principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "actor.principal",
                "value must be a principal scope",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionRequestAuthorizationTarget:
    """Expose content-free immutable ownership to an authorizer."""

    request_id: InputRequestId
    origin: ExecutionOrigin

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(
                    self.request_id,
                    "authorization.request_id",
                )
            ),
        )
        if not isinstance(self.origin, ExecutionOrigin):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.origin",
                "value must be an execution origin",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionBranchAuthorizationTarget:
    """Expose one branch-specific registration target to an authorizer."""

    run_id: RunId
    branch_id: BranchId
    parent_branch_id: BranchId
    principal: PrincipalScope

    def __post_init__(self) -> None:
        for field_name, constructor in (
            ("run_id", RunId),
            ("branch_id", BranchId),
            ("parent_branch_id", BranchId),
        ):
            object.__setattr__(
                self,
                field_name,
                constructor(
                    validate_opaque_id(
                        getattr(self, field_name),
                        f"authorization.{field_name}",
                    )
                ),
            )
        if self.branch_id == self.parent_branch_id:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "authorization.parent_branch_id",
                "parent branch must differ from child branch",
            )
        if not isinstance(self.principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.principal",
                "value must be a principal scope",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionScopeAuthorizationTarget:
    """Expose one content-free execution scope to an authorizer."""

    run_id: RunId
    principal: PrincipalScope
    turn_id: TurnId | None = None
    task_id: TaskId | None = None
    agent_id: AgentId | None = None
    branch_id: BranchId | None = None
    include_descendants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            RunId(validate_opaque_id(self.run_id, "authorization.run_id")),
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
                        validate_opaque_id(
                            value,
                            f"authorization.{field_name}",
                        )
                    ),
                )
        if not isinstance(self.principal, PrincipalScope):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.principal",
                "value must be a principal scope",
            )
        validate_bool(
            self.include_descendants,
            "authorization.include_descendants",
        )
        if self.include_descendants and self.branch_id is None:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "authorization.include_descendants",
                "descendant authorization requires a branch identifier",
            )


InteractionAuthorizationTarget: TypeAlias = (
    InteractionRequestAuthorizationTarget
    | InteractionBranchAuthorizationTarget
    | InteractionScopeAuthorizationTarget
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionAuthorizationDecision:
    """Bind permission and maximum disclosure to one exact authorization."""

    actor: InteractionActor
    operation: InteractionOperation
    target: InteractionAuthorizationTarget
    allowed: bool
    disclosure: InteractionDisclosure

    def __post_init__(self) -> None:
        if not isinstance(self.actor, InteractionActor):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.actor",
                "value must be an interaction actor",
            )
        if not isinstance(self.operation, InteractionOperation):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.operation",
                "value must be an interaction operation",
            )
        if not isinstance(
            self.target,
            (
                InteractionRequestAuthorizationTarget,
                InteractionBranchAuthorizationTarget,
                InteractionScopeAuthorizationTarget,
            ),
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "value must be an interaction authorization target",
            )
        if (
            self.operation is InteractionOperation.REGISTER_BRANCH
            and not isinstance(
                self.target,
                InteractionBranchAuthorizationTarget,
            )
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "branch registration requires a branch target",
            )
        if self.operation is InteractionOperation.INSPECT_BRANCH and (
            not isinstance(self.target, InteractionScopeAuthorizationTarget)
            or self.target.branch_id is None
            or self.target.turn_id is not None
            or self.target.task_id is not None
            or self.target.agent_id is not None
            or self.target.include_descendants
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "branch inspection requires an exact branch scope target",
            )
        if (
            self.operation is InteractionOperation.CANCEL_SCOPE
            and not isinstance(
                self.target,
                InteractionScopeAuthorizationTarget,
            )
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "scope mutation requires an execution-scope target",
            )
        if (
            self.operation is InteractionOperation.SUPERSEDE
            and not isinstance(
                self.target,
                (
                    InteractionRequestAuthorizationTarget,
                    InteractionScopeAuthorizationTarget,
                ),
            )
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "supersession requires a request or execution-scope target",
            )
        if self.operation is InteractionOperation.LIST and not isinstance(
            self.target,
            (
                InteractionRequestAuthorizationTarget,
                InteractionScopeAuthorizationTarget,
            ),
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "list authorization requires a scope or request target",
            )
        if self.operation not in {
            InteractionOperation.LIST,
            InteractionOperation.CANCEL_SCOPE,
            InteractionOperation.SUPERSEDE,
            InteractionOperation.REGISTER_BRANCH,
            InteractionOperation.INSPECT_BRANCH,
        } and not isinstance(
            self.target,
            InteractionRequestAuthorizationTarget,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.target",
                "request operation requires a request target",
            )
        validate_bool(self.allowed, "authorization.allowed")
        if not isinstance(self.disclosure, InteractionDisclosure):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "authorization.disclosure",
                "value must be an interaction disclosure",
            )
        if (
            not self.allowed
            and self.disclosure is not InteractionDisclosure.NONE
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "authorization.disclosure",
                "denied authorization cannot disclose interaction data",
            )


class InteractionAuthorizer(Protocol):
    """Authorize one operation after a scope-filtered ownership lookup."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return the operation permission and disclosure limit."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInputClassificationRequest:
    """Carry one normalized submitted text value to trusted policy."""

    value: str = field(repr=False)
    request_id: InputRequestId
    candidate_digest: str
    question_id: QuestionId
    semantic_type: QuestionType
    policy_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "classification.value",
                "value must be normalized text",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(
                    self.request_id,
                    "classification.request_id",
                )
            ),
        )
        _validate_candidate_digest(self.candidate_digest)
        object.__setattr__(
            self,
            "question_id",
            QuestionId(
                validate_opaque_id(
                    self.question_id,
                    "classification.question_id",
                )
            ),
        )
        if not isinstance(self.semantic_type, QuestionType):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "classification.semantic_type",
                "value must be a question type",
            )
        object.__setattr__(
            self,
            "policy_revision",
            validate_opaque_id(
                self.policy_revision,
                "classification.policy_revision",
                maximum_characters=128,
                maximum_bytes=512,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInputClassification:
    """Return an untrusted classifier decision for broker validation."""

    decision: TaskInputClassificationDecision
    classifier_id: str
    classification_id: str
    policy_revision: str
    request_id: InputRequestId
    candidate_digest: str
    question_id: QuestionId
    semantic_type: QuestionType

    def __post_init__(self) -> None:
        if not isinstance(self.decision, TaskInputClassificationDecision):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "classification.decision",
                "value must be a task-input classification decision",
            )
        for field_name, value in (
            ("classifier_id", self.classifier_id),
            ("classification_id", self.classification_id),
            ("policy_revision", self.policy_revision),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_opaque_id(
                    value,
                    f"classification.{field_name}",
                    maximum_characters=128,
                    maximum_bytes=512,
                ),
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(
                    self.request_id,
                    "classification.request_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "candidate_digest",
            _validate_candidate_digest(self.candidate_digest),
        )
        object.__setattr__(
            self,
            "question_id",
            QuestionId(
                validate_opaque_id(
                    self.question_id,
                    "classification.question_id",
                )
            ),
        )
        if not isinstance(self.semantic_type, QuestionType):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "classification.semantic_type",
                "value must be a question type",
            )


class TaskInputClassifier(Protocol):
    """Classify normalized submitted text through trusted host policy."""

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one deterministic untrusted classification output."""
        ...


def _validate_candidate_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "classification.candidate_digest",
            "value must be a lowercase SHA-256 digest",
        )
    return value


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AcquireControllerActivity:
    """Request a new server-issued active-control lease."""

    request_id: InputRequestId
    controller_id: ControllerId
    action: Literal[ControllerActivityAction.ACQUIRE] = field(
        init=False,
        default=ControllerActivityAction.ACQUIRE,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(self.request_id, "activity.request_id")
            ),
        )
        object.__setattr__(
            self,
            "controller_id",
            ControllerId(
                validate_opaque_id(
                    self.controller_id,
                    "activity.controller_id",
                )
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _SequencedControllerActivity:
    """Authenticate one action against an existing active-control lease."""

    request_id: InputRequestId
    controller_id: ControllerId
    lease_nonce: ActiveControlLeaseNonce
    sequence: int

    def __post_init__(self) -> None:
        if not _is_sequenced_controller_activity(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "activity",
                "value must be a supported sequenced activity variant",
            )
        object.__setattr__(
            self,
            "request_id",
            InputRequestId(
                validate_opaque_id(self.request_id, "activity.request_id")
            ),
        )
        object.__setattr__(
            self,
            "controller_id",
            ControllerId(
                validate_opaque_id(
                    self.controller_id,
                    "activity.controller_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "lease_nonce",
            ActiveControlLeaseNonce(
                validate_opaque_id(self.lease_nonce, "activity.lease_nonce")
            ),
        )
        object.__setattr__(
            self,
            "sequence",
            validate_int(
                self.sequence,
                "activity.sequence",
                minimum=1,
                maximum=9_007_199_254_740_991,
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PulseControllerActivity(_SequencedControllerActivity):
    """Record one authenticated monotonic activity pulse."""

    action: Literal[ControllerActivityAction.PULSE] = field(
        init=False,
        default=ControllerActivityAction.PULSE,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseControllerActivity(_SequencedControllerActivity):
    """Release one authenticated active-control lease."""

    action: Literal[ControllerActivityAction.RELEASE] = field(
        init=False,
        default=ControllerActivityAction.RELEASE,
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DisconnectControllerActivity(_SequencedControllerActivity):
    """Record authenticated controller loss for one active lease."""

    action: Literal[ControllerActivityAction.DISCONNECT] = field(
        init=False,
        default=ControllerActivityAction.DISCONNECT,
    )


SequencedControllerActivity: TypeAlias = (
    PulseControllerActivity
    | ReleaseControllerActivity
    | DisconnectControllerActivity
)
ControllerActivityEvidence: TypeAlias = (
    AcquireControllerActivity | SequencedControllerActivity
)


_SEQUENCED_CONTROLLER_ACTIVITY_TYPES: tuple[
    type[_SequencedControllerActivity], ...
] = (
    PulseControllerActivity,
    ReleaseControllerActivity,
    DisconnectControllerActivity,
)


def _is_sequenced_controller_activity(value: object) -> bool:
    return type(value) in _SEQUENCED_CONTROLLER_ACTIVITY_TYPES


def is_controller_activity_evidence(value: object) -> bool:
    """Return whether a value is one supported controller action."""
    return type(value) is AcquireControllerActivity or (
        _is_sequenced_controller_activity(value)
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InteractionPolicy:
    """Freeze runtime limits and deterministic failure behavior."""

    maximum_unresolved_interactions_per_run: int = (
        MAX_UNRESOLVED_INTERACTIONS_PER_RUN
    )
    maximum_unresolved_required_interactions_per_branch: int = (
        MAX_UNRESOLVED_REQUIRED_INTERACTIONS_PER_BRANCH
    )
    maximum_equivalent_interactions_per_branch: int = (
        MAX_EQUIVALENT_INTERACTIONS_PER_BRANCH
    )
    maximum_pending_interactions_per_process: int = (
        MAX_PENDING_INTERACTIONS_PER_PROCESS
    )
    maximum_idempotency_keys_per_request: int = (
        MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST
    )
    active_control_lease_seconds: int = 30
    attached_loss_with_resumer: HandlerLossDisposition = (
        HandlerLossDisposition.DETACH
    )
    attached_loss_without_resumer: HandlerLossDisposition = (
        HandlerLossDisposition.UNAVAILABLE
    )
    deadline_tie: DeadlineTiePolicy = DeadlineTiePolicy.DEADLINE_FIRST
    task_input_classifier_id: str = "task-input-classifier"
    task_input_policy_revision: str = "task-input-policy-v1"

    def __post_init__(self) -> None:
        for field_name, maximum in (
            (
                "maximum_unresolved_interactions_per_run",
                MAX_UNRESOLVED_INTERACTIONS_PER_RUN,
            ),
            (
                "maximum_unresolved_required_interactions_per_branch",
                MAX_UNRESOLVED_REQUIRED_INTERACTIONS_PER_BRANCH,
            ),
            (
                "maximum_equivalent_interactions_per_branch",
                MAX_EQUIVALENT_INTERACTIONS_PER_BRANCH,
            ),
            (
                "maximum_pending_interactions_per_process",
                MAX_PENDING_INTERACTIONS_PER_PROCESS,
            ),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_int(
                    getattr(self, field_name),
                    f"policy.{field_name}",
                    minimum=1,
                    maximum=maximum,
                ),
            )
        object.__setattr__(
            self,
            "maximum_idempotency_keys_per_request",
            validate_int(
                self.maximum_idempotency_keys_per_request,
                "policy.maximum_idempotency_keys_per_request",
                minimum=1,
                maximum=MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
            ),
        )
        object.__setattr__(
            self,
            "active_control_lease_seconds",
            validate_int(
                self.active_control_lease_seconds,
                "policy.active_control_lease_seconds",
                minimum=1,
                maximum=300,
            ),
        )
        for field_name in (
            "task_input_classifier_id",
            "task_input_policy_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_opaque_id(
                    getattr(self, field_name),
                    f"policy.{field_name}",
                    maximum_characters=128,
                    maximum_bytes=512,
                ),
            )
        for field_name in (
            "attached_loss_with_resumer",
            "attached_loss_without_resumer",
        ):
            if not isinstance(
                getattr(self, field_name),
                HandlerLossDisposition,
            ):
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    f"policy.{field_name}",
                    "value must be a handler-loss disposition",
                )
        if (
            self.attached_loss_with_resumer
            is not HandlerLossDisposition.DETACH
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "policy.attached_loss_with_resumer",
                "a registered resumer requires detached handling",
            )
        if self.attached_loss_without_resumer is HandlerLossDisposition.DETACH:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "policy.attached_loss_without_resumer",
                "detached handling requires a registered resumer",
            )
        if self.deadline_tie is not DeadlineTiePolicy.DEADLINE_FIRST:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "policy.deadline_tie",
                "deadline settlement must precede answer acceptance "
                "at equality",
            )
