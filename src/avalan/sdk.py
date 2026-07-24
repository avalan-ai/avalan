"""Expose the typed asynchronous public agent-input SDK."""

from .agent.continuation_stager import PortableAgentContinuationStager
from .agent.execution import (
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    ExecutionInputRequiredError,
    ExecutionTerminatedError,
    InteractionRuntime,
)
from .agent.orchestrator_contract import Orchestrator
from .entities import Input, ReasoningOrchestratorResponse
from .interaction.broker import AsyncInteractionBroker, InteractionBroker
from .interaction.codec import encode_input_request
from .interaction.continuation import encode_portable_continuation
from .interaction.durable import DurableInteractionSuspension
from .interaction.entities import (
    ActiveControlLeaseNonce,
    AnsweredResolution,
    AnswerProvenance,
    ContinuationId,
    DeclinedResolution,
    InputAnswer,
    InputCandidateResolution,
    InputQuestion,
    InputRequest,
    InputRequestId,
    ParticipantId,
    PrincipalScope,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    SessionId,
    StateRevision,
    TenantId,
    UserId,
    _is_input_answer_variant,
    _is_input_question_variant,
)
from .interaction.error import (
    InputAlreadyResolvedError,
    InputAuthorizationError,
    InputContractError,
    InputErrorCode,
    InputExpiredError,
    InputNotFoundError,
    InputSupersededError,
    InputValidationError,
)
from .interaction.handler import (
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerResolution,
    _is_async_callable,
)
from .interaction.headless import (
    DEFAULT_DURABLE_HANDOFF_WAIT_SECONDS,
    DeclineInputPolicy,
    DurableHandoffInputPolicy,
    ExternalControllerInputPolicy,
    HeadlessInputPolicy,
    PolicyValueInputPolicy,
    PredeclaredInputPolicy,
    TrustedDefaultInputPolicy,
    UnavailableInputPolicy,
)
from .interaction.policy import (
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchAuthorizationTarget,
    InteractionClock,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    InteractionTime,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
)
from .interaction.store import (
    InteractionCorrelation,
    InteractionRecord,
    InteractionResolutionResult,
    InteractionStoreReplayed,
    InteractionTerminalMetadata,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopedInteractionLookup,
)
from .interaction.stores.memory import MemoryInteractionStoreFactory

from asyncio import get_running_loop, sleep
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from json import JSONDecodeError, dumps, loads
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    NewType,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    final,
    overload,
    runtime_checkable,
)
from uuid import uuid4

if TYPE_CHECKING:
    from .agent.orchestrator.orchestrators.default import (
        DefaultOrchestrator,
    )
    from .agent.orchestrator.orchestrators.json import (
        JsonOrchestrator,
        JsonOrchestratorOutput,
    )
    from .agent.orchestrator.orchestrators.reasoning.cot import (
        ReasoningOrchestrator,
    )

InputRequestRef = NewType("InputRequestRef", str)
InputContinuationRef = NewType("InputContinuationRef", str)
DurableInputRequestPayload = NewType("DurableInputRequestPayload", str)
DurableInputContinuationPayload = NewType(
    "DurableInputContinuationPayload",
    str,
)

_INPUT_REF_VERSION = 1
_INPUT_REF_PREFIX = "avl-input-v1"
_INPUT_REF_DOMAIN = b"avalan.public-input-ref.v1\x00"
_INPUT_REF_PAYLOAD_KEYS = frozenset(
    {
        "agent_id",
        "branch_id",
        "continuation_id",
        "kind",
        "model_call_id",
        "request_id",
        "run_id",
        "task_id",
        "turn_id",
        "version",
    }
)
_HEADLESS_POLICY_TYPES = (
    PredeclaredInputPolicy,
    PolicyValueInputPolicy,
    ExternalControllerInputPolicy,
    TrustedDefaultInputPolicy,
    DeclineInputPolicy,
    DurableHandoffInputPolicy,
    UnavailableInputPolicy,
)


class AgentRunResultKind(StrEnum):
    """Identify one member of the public asynchronous run-result union."""

    COMPLETED = "completed"
    INPUT_REQUIRED = "input_required"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AttachedInputDisconnectReason(StrEnum):
    """Identify a content-safe attached input channel failure."""

    CONTROL_CHANNEL_CLOSED = "control_channel_closed"
    HANDLER_CANCELLED = "handler_cancelled"
    HANDLER_UNAVAILABLE = "handler_unavailable"


@runtime_checkable
class _AsyncStringResponse(Protocol):
    async def to_str(self) -> str:
        """Return one asynchronously materialized string result."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputRequestView:
    """Expose immutable semantic input without internal execution identity."""

    mode: RequirementMode
    reason: str
    questions: tuple[InputQuestion, ...]
    created_at: datetime
    state: RequestState
    state_revision: StateRevision

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RequirementMode):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.mode",
                "value must be an input requirement mode",
            )
        if not isinstance(self.reason, str):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.reason",
                "value must be a string",
            )
        if not isinstance(self.questions, tuple) or not all(
            _is_input_question_variant(question) for question in self.questions
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.questions",
                "value must be a tuple of typed input questions",
            )
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.created_at",
                "value must be an aware datetime",
            )
        if not isinstance(self.state, RequestState):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.state",
                "value must be an input request state",
            )
        if (
            isinstance(self.state_revision, bool)
            or not isinstance(self.state_revision, int)
            or self.state_revision < 0
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request.state_revision",
                "value must be a non-negative integer",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputPrincipal:
    """Describe a public input principal without execution identity."""

    user_id: str | None = None
    tenant_id: str | None = None
    participant_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        _principal_scope(self)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputValidationFeedback:
    """Expose content-safe correction feedback to an attached handler."""

    code: InputErrorCode
    path: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, InputErrorCode):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.validation_error.code",
                "value must be an input error code",
            )
        for name in ("path", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise InputValidationError(
                    InputErrorCode.INVALID_TYPE,
                    f"handler.validation_error.{name}",
                    "value must be a non-empty string",
                )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AttachedInputContext:
    """Provide semantic input and optional correction feedback."""

    request: InputRequestView
    validation_error: InputValidationFeedback | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not InputRequestView:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.request",
                "value must be an input request view",
            )
        if (
            self.validation_error is not None
            and type(self.validation_error) is not InputValidationFeedback
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.validation_error",
                "value must be input validation feedback",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AttachedInputDetached:
    """Request detached handling without manufacturing an answer."""

    kind: Literal["detached"] = "detached"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AttachedInputDisconnected:
    """Report attached channel loss without implying a decline."""

    reason: AttachedInputDisconnectReason
    kind: Literal["disconnected"] = "disconnected"

    def __post_init__(self) -> None:
        if not isinstance(self.reason, AttachedInputDisconnectReason):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.disconnect_reason",
                "value must be an attached disconnect reason",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputInspection:
    """Return one authorized public input projection."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef
    request: InputRequestView
    detached_resumption_available: bool

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)
        if type(self.request) is not InputRequestView:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "inspection.request",
                "value must be an input request view",
            )
        if type(self.detached_resumption_available) is not bool:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "inspection.detached_resumption_available",
                "value must be a boolean",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputAnswerSubmission:
    """Submit typed answers with externally authored provenance."""

    answers: tuple[InputAnswer, ...]
    provenance: Literal[
        AnswerProvenance.HUMAN,
        AnswerProvenance.EXTERNAL_CONTROLLER,
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.answers, tuple) or not all(
            _is_input_answer_variant(answer) for answer in self.answers
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "submission.answers",
                "answers must be a tuple of typed input answers",
            )
        if self.provenance not in {
            AnswerProvenance.HUMAN,
            AnswerProvenance.EXTERNAL_CONTROLLER,
        }:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "submission.provenance",
                "submission requires externally authored provenance",
            )
        if any(
            answer.provenance is not self.provenance for answer in self.answers
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "submission.answers",
                "answer provenance must match submission provenance",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputDeclineSubmission:
    """Submit one explicit externally controlled decline."""

    provenance: Literal[
        AnswerProvenance.HUMAN,
        AnswerProvenance.EXTERNAL_CONTROLLER,
    ] = AnswerProvenance.EXTERNAL_CONTROLLER

    def __post_init__(self) -> None:
        if self.provenance not in {
            AnswerProvenance.HUMAN,
            AnswerProvenance.EXTERNAL_CONTROLLER,
        }:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "submission.provenance",
                "submission requires externally authored provenance",
            )


InputSubmission: TypeAlias = InputAnswerSubmission | InputDeclineSubmission
AttachedInputOutcome: TypeAlias = (
    InputSubmission | AttachedInputDetached | AttachedInputDisconnected
)


class AttachedInputHandler(Protocol):
    """Handle semantic attached input asynchronously."""

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> AttachedInputOutcome:
        """Return one typed external submission or channel outcome."""
        ...


class AttachedInputCancellationHandler(Protocol):
    """Receive cancellation of one no-longer-needed public input request."""

    async def __call__(self, context: AttachedInputContext) -> None:
        """Handle cancellation of one attached request."""
        ...


class InputPolicyValueProvider(Protocol):
    """Compute typed policy-owned answers asynchronously."""

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> tuple[InputAnswer, ...]:
        """Return policy-provenance answers for one semantic request."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputResolutionAccepted:
    """Acknowledge one accepted or idempotently replayed resolution."""

    interaction_state: Literal["answered", "declined"]
    idempotent: bool
    kind: Literal["resolution_accepted"] = "resolution_accepted"
    channel: Literal["typed"] = "typed"

    def __post_init__(self) -> None:
        if self.interaction_state not in {"answered", "declined"}:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.interaction_state",
                "value must be answered or declined",
            )
        if type(self.idempotent) is not bool:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resolution.idempotent",
                "value must be a boolean",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputInspectionRequest:
    """Request one authorized durable input projection."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputResolutionRequest:
    """Request one atomic durable resolution and continuation requeue."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef
    submission: InputSubmission
    idempotency_key: ResolutionIdempotencyKey

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)
        _validate_submission(self.submission)
        if (
            not isinstance(self.idempotency_key, str)
            or not self.idempotency_key
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "idempotency_key",
                "value must be a non-empty string",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputResolutionResult:
    """Echo one correlated durable resolution bridge result."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef
    resolution: InputResolutionAccepted

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)
        if type(self.resolution) is not InputResolutionAccepted:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "input_bridge.resolution",
                "bridge must return a typed resolution acknowledgement",
            )


class InputControllerBridge(Protocol):
    """Authorize durable inspection and atomic resolution at the host."""

    async def inspect_input(
        self,
        request: InputInspectionRequest,
    ) -> InputInspection:
        """Return the exact authorized projection for opaque references."""
        ...

    async def resolve_input(
        self,
        request: InputResolutionRequest,
    ) -> InputResolutionResult:
        """Atomically resolve and requeue the exact opaque continuation."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInputPersistenceRequest:
    """Carry one serializable suspension to an atomic host bridge."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef
    request: InputRequestView
    request_payload: DurableInputRequestPayload
    continuation_payload: DurableInputContinuationPayload
    persistence_digest: str

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)
        if type(self.request) is not InputRequestView:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_bridge.request",
                "value must be an input request view",
            )
        _validate_persistence_payload(
            self.request_payload,
            "durable_bridge.request_payload",
        )
        _validate_persistence_payload(
            self.continuation_payload,
            "durable_bridge.continuation_payload",
        )
        _validate_sha256(
            self.persistence_digest,
            "durable_bridge.persistence_digest",
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInputPersistenceAccepted:
    """Acknowledge one exact atomic suspension persistence."""

    request_id: InputRequestRef
    continuation_id: InputContinuationRef
    persistence_digest: str

    def __post_init__(self) -> None:
        _decode_correlation_pair(self.request_id, self.continuation_id)
        _validate_sha256(
            self.persistence_digest,
            "durable_bridge.persistence_digest",
        )


class DurableInputBridge(InputControllerBridge, Protocol):
    """Own atomic suspension persistence, inspection, and resolution."""

    async def persist_input(
        self,
        request: DurableInputPersistenceRequest,
    ) -> DurableInputPersistenceAccepted:
        """Atomically persist and schedule one exact durable suspension."""
        ...


class InputControllerClient(Protocol):
    """Expose typed durable input inspection and resolution operations."""

    async def inspect_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InputInspection:
        """Return one authorized public input projection."""
        ...

    async def resolve_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
        submission: InputSubmission,
        *,
        idempotency_key: ResolutionIdempotencyKey,
    ) -> InputResolutionAccepted:
        """Resolve one authorized public input request."""
        ...


@final
class InputController:
    """Inspect and resolve durable input through a root-public bridge."""

    def __init__(self, bridge: InputControllerBridge) -> None:
        _validate_input_controller_bridge(bridge)
        self._bridge = bridge

    async def inspect(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InputInspection:
        """Return one authorized public input projection."""
        return await self.inspect_input(request_id, continuation_id)

    async def inspect_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InputInspection:
        """Return one authorized public input projection."""
        correlation = _decode_correlation_pair(
            request_id,
            continuation_id,
        )
        result = await self._bridge.inspect_input(
            InputInspectionRequest(
                request_id=request_id,
                continuation_id=continuation_id,
            )
        )
        if type(result) is not InputInspection:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "input_bridge.inspection",
                "bridge must return a typed input inspection",
            )
        _require_ref_echo(
            correlation,
            result.request_id,
            result.continuation_id,
            path="input_bridge.inspection",
        )
        return result

    async def resolve(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
        submission: InputSubmission,
        *,
        idempotency_key: ResolutionIdempotencyKey,
    ) -> InputResolutionAccepted:
        """Resolve one authorized public input request."""
        return await self.resolve_input(
            request_id,
            continuation_id,
            submission,
            idempotency_key=idempotency_key,
        )

    async def resolve_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
        submission: InputSubmission,
        *,
        idempotency_key: ResolutionIdempotencyKey,
    ) -> InputResolutionAccepted:
        """Atomically resolve and requeue one exact durable continuation."""
        correlation = _decode_correlation_pair(
            request_id,
            continuation_id,
        )
        _validate_submission(submission)
        result = await self._bridge.resolve_input(
            InputResolutionRequest(
                request_id=request_id,
                continuation_id=continuation_id,
                submission=submission,
                idempotency_key=idempotency_key,
            )
        )
        if type(result) is not InputResolutionResult:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "input_bridge.resolution",
                "bridge must return a typed input resolution result",
            )
        _require_ref_echo(
            correlation,
            result.request_id,
            result.continuation_id,
            path="input_bridge.resolution",
        )
        expected_state = (
            "answered"
            if isinstance(submission, InputAnswerSubmission)
            else "declined"
        )
        if result.resolution.interaction_state != expected_state:
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "input_bridge.resolution.interaction_state",
                "bridge resolution does not match the submitted outcome",
            )
        return result.resolution


def create_input_controller(
    bridge: InputControllerBridge,
) -> InputController:
    """Create one durable controller over a root-public host bridge."""
    return InputController(bridge)


T_co = TypeVar("T_co", covariant=True)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunCompleted(Generic[T_co]):
    """Return one successfully completed agent value."""

    value: T_co
    kind: Literal[AgentRunResultKind.COMPLETED] = AgentRunResultKind.COMPLETED
    channel: Literal["typed"] = "typed"

    def to_str(self) -> str:
        """Return a completed string value."""
        if not isinstance(self.value, str):
            raise TypeError("completed value is not a string")
        return self.value

    def to_json(self) -> object:
        """Return a completed JSON-compatible value."""
        _validate_json_value(self.value)
        return self.value


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunInputRequired:
    """Return durable detached input correlation without raw exceptions."""

    request: InputRequestView
    request_id: InputRequestRef | None
    continuation_id: InputContinuationRef | None
    detached_resumption_available: bool
    kind: Literal[AgentRunResultKind.INPUT_REQUIRED] = (
        AgentRunResultKind.INPUT_REQUIRED
    )
    channel: Literal["typed"] = "typed"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunCancelled:
    """Return a typed cancellation outcome."""

    kind: Literal[AgentRunResultKind.CANCELLED] = AgentRunResultKind.CANCELLED
    channel: Literal["typed"] = "typed"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunFailed:
    """Return a content-safe typed execution failure."""

    code: str
    message: str
    retryable: bool
    kind: Literal[AgentRunResultKind.FAILED] = AgentRunResultKind.FAILED
    channel: Literal["typed"] = "typed"


AgentRunResult: TypeAlias = (
    AgentRunCompleted[T_co]
    | AgentRunInputRequired
    | AgentRunCancelled
    | AgentRunFailed
)


@final
class AgentInteractionRuntime:
    """Own one root-public attached or durable interaction runtime."""

    __slots__ = ("_closer", "_runtime")
    _closer: Callable[[], Awaitable[None]]
    _runtime: object

    def __init__(self) -> None:
        raise TypeError("use an avalan interaction runtime factory")

    async def __aenter__(self) -> "AgentInteractionRuntime":
        """Return this owned interaction runtime."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close resources owned by this interaction runtime."""
        del exc_type, exc, traceback
        await self.aclose()

    async def aclose(self) -> None:
        """Close resources owned by this interaction runtime."""
        await self._closer()


@final
class AgentHeadlessInputPolicy:
    """Carry one root-public headless policy adapter."""

    __slots__ = ("_policy",)
    _policy: object

    def __init__(self) -> None:
        raise TypeError("use an avalan headless policy factory")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInputIntegration:
    """Bundle one durable runtime, handoff policy, and controller."""

    runtime: AgentInteractionRuntime
    headless_policy: AgentHeadlessInputPolicy
    controller: InputController

    def __post_init__(self) -> None:
        if type(self.runtime) is not AgentInteractionRuntime:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_integration.runtime",
                "value must be a factory-created interaction runtime",
            )
        if type(self.headless_policy) is not AgentHeadlessInputPolicy:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_integration.headless_policy",
                "value must be a factory-created headless policy",
            )
        if type(self.controller) is not InputController:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_integration.controller",
                "value must be an input controller",
            )


async def create_attached_input_runtime(
    handler: AttachedInputHandler | None = None,
    *,
    principal: InputPrincipal | None = None,
) -> AgentInteractionRuntime:
    """Create an owned in-process runtime for one async public handler."""
    if handler is not None and not _is_async_callable(handler):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "handler",
            "value must be an async attached input handler",
        )
    scope = _principal_scope(principal or InputPrincipal())
    actor = InteractionActor(principal=scope)
    clock = _SystemInteractionClock()
    id_factory = _UuidInteractionIdFactory()
    policy = InteractionPolicy()
    authorizer = _BoundPrincipalAuthorizer(scope)
    classifier = _AllowTaskInputClassifier(policy)
    store = await MemoryInteractionStoreFactory(
        policy=policy,
        clock=clock,
        authorizer=authorizer,
        id_factory=id_factory,
        classifier=classifier,
    ).open()
    broker = AsyncInteractionBroker(
        store=store,
        clock=clock,
        id_factory=id_factory,
        policy=policy,
        classifier=classifier,
    )
    public_handler = handler or _DetachedAttachedInputHandler()
    runtime = AttachedInteractionRuntime(
        broker=broker,
        actor=actor,
        handler=_AttachedInputHandlerAdapter(public_handler, clock),
    )
    return _new_agent_interaction_runtime(runtime, broker.aclose)


def create_predeclared_input_policy(
    answers: tuple[InputAnswer, ...],
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy backed by immutable trusted-host values."""
    return _new_agent_headless_input_policy(
        PredeclaredInputPolicy(
            answers=answers,
            cancellation_handler=_cancellation_adapter(cancellation_handler),
        )
    )


def create_policy_value_input_policy(
    provider: InputPolicyValueProvider,
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy backed by one async trusted value provider."""
    if not _is_async_callable(provider):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless.provider",
            "value must be an async policy-value provider",
        )
    return _new_agent_headless_input_policy(
        PolicyValueInputPolicy(
            provider=_PolicyValueProviderAdapter(provider),
            cancellation_handler=_cancellation_adapter(cancellation_handler),
        )
    )


def create_external_controller_input_policy(
    controller: AttachedInputHandler,
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy backed by one async external controller."""
    if not _is_async_callable(controller):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless.controller",
            "value must be an async external input controller",
        )
    return _new_agent_headless_input_policy(
        ExternalControllerInputPolicy(
            controller=_AttachedInputHandlerAdapter(
                controller,
                _SystemInteractionClock(),
            ),
            cancellation_handler=_cancellation_adapter(cancellation_handler),
        )
    )


def create_trusted_default_input_policy(
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy that applies only request-declared defaults."""
    return _new_agent_headless_input_policy(
        TrustedDefaultInputPolicy(
            cancellation_handler=_cancellation_adapter(cancellation_handler)
        )
    )


def create_decline_input_policy(
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy that explicitly declines every input request."""
    return _new_agent_headless_input_policy(
        DeclineInputPolicy(
            cancellation_handler=_cancellation_adapter(cancellation_handler)
        )
    )


def create_unavailable_input_policy(
    *,
    cancellation_handler: AttachedInputCancellationHandler | None = None,
) -> AgentHeadlessInputPolicy:
    """Create a policy that reports an unavailable input channel."""
    return _new_agent_headless_input_policy(
        UnavailableInputPolicy(
            cancellation_handler=_cancellation_adapter(cancellation_handler)
        )
    )


def create_durable_input_integration(
    bridge: DurableInputBridge,
    *,
    principal: InputPrincipal | None = None,
    handoff_wait_seconds: int = DEFAULT_DURABLE_HANDOFF_WAIT_SECONDS,
) -> DurableInputIntegration:
    """Create a durable runtime over one atomic external host bridge."""
    _validate_durable_input_bridge(bridge)
    scope = _principal_scope(principal or InputPrincipal())
    runtime = DurableInteractionRuntime(
        actor=InteractionActor(principal=scope),
        stager=PortableAgentContinuationStager(),
    )
    policy = DurableHandoffInputPolicy(
        handoff=_DurableInputBridgeHandoff(bridge),
        durable_handoff_wait_seconds=handoff_wait_seconds,
    )
    return DurableInputIntegration(
        runtime=_new_agent_interaction_runtime(runtime, _noop_close),
        headless_policy=_new_agent_headless_input_policy(policy),
        controller=create_input_controller(bridge),
    )


@final
class AsyncInputController:
    """Inspect and resolve durable input through an authorized broker."""

    def __init__(
        self,
        *,
        broker: InteractionBroker,
        actor: InteractionActor,
        clock: InteractionClock,
        durable_authority: (
            Callable[[InteractionCorrelation], Awaitable[bool]] | None
        ) = None,
        durable_resolver: (
            Callable[
                [ResolveInteractionCommand],
                Awaitable[InteractionResolutionResult],
            ]
            | None
        ) = None,
    ) -> None:
        if durable_authority is not None and not _is_async_callable(
            durable_authority
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_authority",
                "value must be an async durable authority",
            )
        if durable_resolver is not None and not _is_async_callable(
            durable_resolver
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_resolver",
                "value must be an async durable resolver",
            )
        self._broker = broker
        self._actor = actor
        self._clock = clock
        self._durable_authority = durable_authority
        self._durable_resolver = durable_resolver

    async def inspect(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InputInspection:
        """Return one authorized public input projection."""
        return await self.inspect_input(request_id, continuation_id)

    async def inspect_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InputInspection:
        """Return one authorized public input projection."""
        record = await self._record(request_id, continuation_id)
        resumable = (
            False
            if self._durable_authority is None
            else await self._durable_authority(record.correlation)
        )
        if not isinstance(resumable, bool):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_authority",
                "durable authority must return a boolean",
            )
        return _inspection(record, resumable=resumable)

    async def resolve(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
        submission: InputSubmission,
        *,
        idempotency_key: ResolutionIdempotencyKey,
    ) -> InputResolutionAccepted:
        """Resolve one authorized public input request."""
        return await self.resolve_input(
            request_id,
            continuation_id,
            submission,
            idempotency_key=idempotency_key,
        )

    async def resolve_input(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
        submission: InputSubmission,
        *,
        idempotency_key: ResolutionIdempotencyKey,
    ) -> InputResolutionAccepted:
        """Resolve one authorized public input request."""
        if type(submission) not in (
            InputAnswerSubmission,
            InputDeclineSubmission,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "submission",
                "value must be a typed input submission",
            )
        record = await self._record(request_id, continuation_id)
        request = record.request
        _require_pending_resolution_state(request.state)
        await self._require_durable_authority(record.correlation)
        observed_at = await self._clock.read()
        if isinstance(submission, InputAnswerSubmission):
            resolution: InputCandidateResolution = AnsweredResolution(
                request_id=request.request_id,
                provenance=submission.provenance,
                resolved_at=observed_at.wall_time,
                answers=submission.answers,
            )
            interaction_state: Literal["answered", "declined"] = "answered"
        else:
            resolution = DeclinedResolution(
                request_id=request.request_id,
                provenance=submission.provenance,
                resolved_at=observed_at.wall_time,
            )
            interaction_state = "declined"
        command = ResolveInteractionCommand(
            actor=self._actor,
            correlation=record.correlation,
            expected_state_revision=request.state_revision,
            idempotency_key=idempotency_key,
            proposed_resolution=resolution,
        )
        resolver = self._durable_resolver
        if resolver is None:
            raise InputValidationError(
                InputErrorCode.UNAVAILABLE,
                "durable_resolver",
                "durable resolution is unavailable",
            )
        store_result = await resolver(command)
        _validate_resolution_correlation(
            store_result,
            record.correlation,
        )
        if isinstance(store_result, ResolveInteractionApplied):
            _raise_for_non_candidate_state(store_result.record.request.state)
            return InputResolutionAccepted(
                interaction_state=interaction_state,
                idempotent=False,
            )
        if isinstance(store_result, InteractionStoreReplayed):
            _raise_for_non_candidate_state(store_result.record.request.state)
            return InputResolutionAccepted(
                interaction_state=interaction_state,
                idempotent=True,
            )
        if isinstance(store_result, ResolveInteractionRejected):
            _raise_public_transition_error(
                store_result.error.code,
                store_result.error.path,
                store_result.error.message,
            )
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "input_resolver",
            "input resolver returned an unrelated resolution result",
        )

    async def _require_durable_authority(
        self,
        correlation: InteractionCorrelation,
    ) -> None:
        authority = self._durable_authority
        if authority is None:
            raise InputAuthorizationError()
        authorized = await authority(correlation)
        if not isinstance(authorized, bool):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_authority",
                "durable authority must return a boolean",
            )
        if not authorized:
            raise InputAuthorizationError()

    async def _record(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> InteractionRecord:
        correlation = _decode_correlation_pair(
            request_id,
            continuation_id,
        )
        try:
            projection = await self._broker.inspect(
                ScopedInteractionLookup(
                    actor=self._actor,
                    correlation=correlation,
                )
            )
        except InputAuthorizationError:
            raise
        except InputContractError as error:
            if error.code is InputErrorCode.FORBIDDEN:
                raise InputAuthorizationError() from None
            if error.code is InputErrorCode.NOT_FOUND:
                raise InputNotFoundError() from None
            raise
        if projection is None:
            raise InputNotFoundError()
        if isinstance(projection, InteractionTerminalMetadata):
            raise InputNotFoundError()
        if not isinstance(projection, InteractionRecord):
            raise InputNotFoundError()
        if projection.correlation != correlation:
            raise InputNotFoundError()
        return projection


async def inspect_input(
    controller: InputControllerClient,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
) -> InputInspection:
    """Inspect one durable input through the typed controller."""
    return await controller.inspect_input(request_id, continuation_id)


async def resolve_input(
    controller: InputControllerClient,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
    submission: InputSubmission,
    *,
    idempotency_key: ResolutionIdempotencyKey,
) -> InputResolutionAccepted:
    """Resolve one durable input through the typed controller."""
    return await controller.resolve_input(
        request_id,
        continuation_id,
        submission,
        idempotency_key=idempotency_key,
    )


@overload
async def run_agent(
    orchestrator: "JsonOrchestrator",
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime | None = None,
    headless_policy: AgentHeadlessInputPolicy | None = None,
    generation_options_override: Mapping[str, object] | None = None,
    operation_index: int = 0,
) -> "AgentRunResult[JsonOrchestratorOutput]": ...


@overload
async def run_agent(
    orchestrator: "ReasoningOrchestrator",
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime | None = None,
    headless_policy: AgentHeadlessInputPolicy | None = None,
    generation_options_override: Mapping[str, object] | None = None,
    operation_index: int = 0,
) -> "AgentRunResult[ReasoningOrchestratorResponse]": ...


@overload
async def run_agent(
    orchestrator: "DefaultOrchestrator",
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime | None = None,
    headless_policy: AgentHeadlessInputPolicy | None = None,
    generation_options_override: Mapping[str, object] | None = None,
    operation_index: int = 0,
) -> AgentRunResult[str]: ...


@overload
async def run_agent(
    orchestrator: Orchestrator,
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime | None = None,
    headless_policy: AgentHeadlessInputPolicy | None = None,
    generation_options_override: Mapping[str, object] | None = None,
    operation_index: int = 0,
) -> AgentRunResult[object]: ...


async def run_agent(
    orchestrator: Orchestrator,
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime | None = None,
    headless_policy: AgentHeadlessInputPolicy | None = None,
    generation_options_override: Mapping[str, object] | None = None,
    operation_index: int = 0,
) -> AgentRunResult[object]:
    """Run one agent asynchronously and return a strict typed outcome."""
    try:
        internal_runtime = _unwrap_interaction_runtime(interaction_runtime)
        internal_policy = _unwrap_headless_policy(headless_policy)
        runtime = _runtime_for_policy(internal_runtime, internal_policy)
        response: object = await orchestrator(
            input,
            interaction_runtime=runtime,
            generation_options_override=generation_options_override,
            operation_index=operation_index,
        )
        if isinstance(response, str | ReasoningOrchestratorResponse):
            value: object = response
        elif isinstance(response, _AsyncStringResponse):
            value = await response.to_str()
        else:
            value = response
        return AgentRunCompleted(value=value)
    except ExecutionInputRequiredError as error:
        return await _input_required_result(error, internal_policy)
    except ExecutionTerminatedError as error:
        if error.outcome.status is ResolutionStatus.CANCELLED:
            return AgentRunCancelled()
        return AgentRunFailed(
            code=f"input.{error.outcome.status.value}",
            message="agent execution ended while awaiting input",
            retryable=False,
        )
    except InputContractError as error:
        return AgentRunFailed(
            code=error.code.value,
            message=error.safe_message,
            retryable=False,
        )
    except Exception:
        return AgentRunFailed(
            code="agent.execution_failed",
            message="agent execution failed",
            retryable=False,
        )


def _unwrap_interaction_runtime(
    runtime: object | None,
) -> InteractionRuntime | None:
    if runtime is None:
        return None
    if type(runtime) is AgentInteractionRuntime:
        internal = runtime._runtime
        if isinstance(
            internal,
            AttachedInteractionRuntime | DurableInteractionRuntime,
        ):
            return internal
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "interaction_runtime",
            "factory-created interaction runtime is invalid",
        )
    if isinstance(
        runtime,
        AttachedInteractionRuntime | DurableInteractionRuntime,
    ):
        return runtime
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "interaction_runtime",
        "value must be a factory-created interaction runtime",
    )


def _unwrap_headless_policy(
    policy: object | None,
) -> HeadlessInputPolicy | None:
    if policy is None:
        return None
    if type(policy) is AgentHeadlessInputPolicy:
        internal = policy._policy
        if isinstance(internal, _HEADLESS_POLICY_TYPES):
            return internal
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless_policy",
            "factory-created headless policy is invalid",
        )
    if isinstance(policy, _HEADLESS_POLICY_TYPES):
        return policy
    if callable(policy):
        return cast(HeadlessInputPolicy, policy)
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "headless_policy",
        "value must be a factory-created headless policy",
    )


def _runtime_for_policy(
    runtime: InteractionRuntime | None,
    policy: HeadlessInputPolicy | None,
) -> InteractionRuntime | None:
    if policy is None:
        return runtime
    if isinstance(policy, DurableHandoffInputPolicy):
        if runtime is not None and not isinstance(
            runtime,
            DurableInteractionRuntime,
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless_policy",
                "durable handoff requires a durable interaction runtime",
            )
        return runtime
    if not isinstance(runtime, AttachedInteractionRuntime):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "interaction_runtime",
            "attached headless policy requires an attached runtime",
        )
    return replace(runtime, handler=policy)


async def _input_required_result(
    error: ExecutionInputRequiredError,
    policy: HeadlessInputPolicy | None,
) -> AgentRunResult[str]:
    request = error.request
    if request is None:
        return AgentRunFailed(
            code="input.correlation_unavailable",
            message="input correlation is unavailable",
            retryable=False,
        )
    if isinstance(policy, DurableHandoffInputPolicy):
        suspension = error.durable
        if suspension is None:
            return AgentRunFailed(
                code="input.durable_handoff_unavailable",
                message="durable input handoff is unavailable",
                retryable=False,
            )
        try:
            request = await policy.persist(suspension)
            await policy.wait()
        except InputContractError as handoff_error:
            return AgentRunFailed(
                code=handoff_error.code.value,
                message=handoff_error.safe_message,
                retryable=False,
            )
        except Exception:
            return AgentRunFailed(
                code="input.durable_handoff_failed",
                message="durable input handoff failed",
                retryable=True,
            )
        correlation = InteractionCorrelation.from_request(request)
        return AgentRunInputRequired(
            request=_request_view(request),
            request_id=InputRequestRef(
                _encode_correlation_ref("request", correlation)
            ),
            continuation_id=InputContinuationRef(
                _encode_correlation_ref("continuation", correlation)
            ),
            detached_resumption_available=True,
        )
    return AgentRunInputRequired(
        request=_request_view(request),
        request_id=None,
        continuation_id=None,
        detached_resumption_available=False,
    )


def _request_view(request: InputRequest) -> InputRequestView:
    return InputRequestView(
        mode=request.mode,
        reason=request.reason,
        questions=request.questions,
        created_at=request.created_at,
        state=request.state,
        state_revision=request.state_revision,
    )


def _inspection(
    record: InteractionRecord,
    *,
    resumable: bool,
) -> InputInspection:
    correlation = record.correlation
    return InputInspection(
        request_id=InputRequestRef(
            _encode_correlation_ref("request", correlation)
        ),
        continuation_id=InputContinuationRef(
            _encode_correlation_ref("continuation", correlation)
        ),
        request=_request_view(record.request),
        detached_resumption_available=resumable,
    )


def _new_agent_interaction_runtime(
    runtime: InteractionRuntime,
    closer: Callable[[], Awaitable[None]],
) -> AgentInteractionRuntime:
    public_runtime = object.__new__(AgentInteractionRuntime)
    public_runtime._runtime = runtime
    public_runtime._closer = closer
    return public_runtime


def _new_agent_headless_input_policy(
    policy: HeadlessInputPolicy,
) -> AgentHeadlessInputPolicy:
    public_policy = object.__new__(AgentHeadlessInputPolicy)
    public_policy._policy = policy
    return public_policy


async def _noop_close() -> None:
    return None


def _principal_scope(principal: InputPrincipal) -> PrincipalScope:
    if type(principal) is not InputPrincipal:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "principal",
            "value must be an input principal",
        )
    return PrincipalScope(
        user_id=(
            None if principal.user_id is None else UserId(principal.user_id)
        ),
        tenant_id=(
            None
            if principal.tenant_id is None
            else TenantId(principal.tenant_id)
        ),
        participant_id=(
            None
            if principal.participant_id is None
            else ParticipantId(principal.participant_id)
        ),
        session_id=(
            None
            if principal.session_id is None
            else SessionId(principal.session_id)
        ),
    )


@final
class _SystemInteractionClock:
    async def read(self) -> InteractionTime:
        loop = get_running_loop()
        return InteractionTime.from_clock(
            wall_time=datetime.now(UTC),
            monotonic_seconds=loop.time(),
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        delay = max(0.0, monotonic_deadline - get_running_loop().time())
        await sleep(delay)


@final
class _UuidInteractionIdFactory:
    async def new_request_id(self) -> InputRequestId:
        return InputRequestId(f"request-{uuid4()}")

    async def new_continuation_id(self) -> ContinuationId:
        return ContinuationId(f"continuation-{uuid4()}")

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        return ResolutionIdempotencyKey(f"resolution-{uuid4()}")

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        return ActiveControlLeaseNonce(f"lease-{uuid4()}")


@final
class _BoundPrincipalAuthorizer:
    def __init__(self, principal: PrincipalScope) -> None:
        self._principal = principal

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        allowed = (
            actor.principal == self._principal
            and _authorization_target_principal(target) == self._principal
        )
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=allowed,
            disclosure=(
                InteractionDisclosure.FULL
                if allowed
                else InteractionDisclosure.NONE
            ),
        )


def _authorization_target_principal(
    target: InteractionAuthorizationTarget,
) -> PrincipalScope:
    if isinstance(target, InteractionRequestAuthorizationTarget):
        return target.origin.principal
    if isinstance(
        target,
        InteractionBranchAuthorizationTarget
        | InteractionScopeAuthorizationTarget,
    ):
        return target.principal
    raise InputValidationError(
        InputErrorCode.INVALID_TYPE,
        "authorization.target",
        "value must be an interaction authorization target",
    )


@final
class _AllowTaskInputClassifier:
    def __init__(self, policy: InteractionPolicy) -> None:
        self._policy = policy

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self._policy.task_input_classifier_id,
            classification_id=f"classification-{uuid4()}",
            policy_revision=request.policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


@final
class _DetachedAttachedInputHandler:
    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> AttachedInputOutcome:
        del context
        return AttachedInputDetached()


def _public_handler_context(
    context: InputHandlerContext,
) -> AttachedInputContext:
    validation_error = context.validation_error
    return AttachedInputContext(
        request=_request_view(context.request),
        validation_error=(
            None
            if validation_error is None
            else InputValidationFeedback(
                code=validation_error.code,
                path=validation_error.path,
                message=validation_error.message,
            )
        ),
    )


@final
class _InputCancellationAdapter:
    def __init__(self, handler: AttachedInputCancellationHandler) -> None:
        self._handler = handler

    async def __call__(self, context: InputHandlerContext) -> None:
        await self._handler(_public_handler_context(context))


def _cancellation_adapter(
    handler: AttachedInputCancellationHandler | None,
) -> _InputCancellationAdapter | None:
    if handler is None:
        return None
    if not _is_async_callable(handler):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless.cancellation_handler",
            "value must be an async cancellation handler",
        )
    return _InputCancellationAdapter(handler)


@final
class _PolicyValueProviderAdapter:
    def __init__(self, provider: InputPolicyValueProvider) -> None:
        self._provider = provider

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> tuple[InputAnswer, ...]:
        return await self._provider(_public_handler_context(context))


@final
class _AttachedInputHandlerAdapter:
    def __init__(
        self,
        handler: AttachedInputHandler,
        clock: InteractionClock,
    ) -> None:
        self._handler = handler
        self._clock = clock

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> (
        InputHandlerResolution
        | InputHandlerDetached
        | InputHandlerDisconnected
    ):
        outcome = await self._handler(_public_handler_context(context))
        _validate_attached_input_outcome(outcome)
        if isinstance(outcome, InputAnswerSubmission):
            observed_at = await self._clock.read()
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=context.request.request_id,
                    provenance=outcome.provenance,
                    resolved_at=observed_at.wall_time,
                    answers=outcome.answers,
                )
            )
        if isinstance(outcome, InputDeclineSubmission):
            observed_at = await self._clock.read()
            return InputHandlerResolution(
                resolution=DeclinedResolution(
                    request_id=context.request.request_id,
                    provenance=outcome.provenance,
                    resolved_at=observed_at.wall_time,
                )
            )
        if isinstance(outcome, AttachedInputDetached):
            return InputHandlerDetached()
        assert isinstance(outcome, AttachedInputDisconnected)
        return InputHandlerDisconnected(
            reason=InputDisconnectReason(outcome.reason.value)
        )


@final
class _DurableInputBridgeHandoff:
    def __init__(self, bridge: DurableInputBridge) -> None:
        self._bridge = bridge

    async def __call__(
        self,
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        if type(suspension) is not DurableInteractionSuspension:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_bridge.suspension",
                "value must be a durable interaction suspension",
            )
        staged = suspension.command.request
        correlation = InteractionCorrelation.from_request(staged)
        request_id = InputRequestRef(
            _encode_correlation_ref("request", correlation)
        )
        continuation_id = InputContinuationRef(
            _encode_correlation_ref("continuation", correlation)
        )
        request_payload = DurableInputRequestPayload(
            dumps(
                encode_input_request(staged),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        continuation_payload = DurableInputContinuationPayload(
            encode_portable_continuation(suspension.continuation)
        )
        pending = replace(
            staged,
            state=RequestState.PENDING,
            state_revision=StateRevision(int(staged.state_revision) + 1),
        )
        persistence_digest = _persistence_digest(
            request_payload,
            continuation_payload,
        )
        result = await self._bridge.persist_input(
            DurableInputPersistenceRequest(
                request_id=request_id,
                continuation_id=continuation_id,
                request=_request_view(pending),
                request_payload=request_payload,
                continuation_payload=continuation_payload,
                persistence_digest=persistence_digest,
            )
        )
        if type(result) is not DurableInputPersistenceAccepted:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_bridge.persistence",
                "bridge must return a typed persistence acknowledgement",
            )
        _require_ref_echo(
            correlation,
            result.request_id,
            result.continuation_id,
            path="durable_bridge.persistence",
        )
        if not compare_digest(
            result.persistence_digest,
            persistence_digest,
        ):
            raise InputValidationError(
                InputErrorCode.CORRELATION_MISMATCH,
                "durable_bridge.persistence_digest",
                "bridge acknowledgement does not match the suspension",
            )
        return pending


def _validate_attached_input_outcome(
    outcome: object,
) -> AttachedInputOutcome:
    if type(outcome) not in {
        InputAnswerSubmission,
        InputDeclineSubmission,
        AttachedInputDetached,
        AttachedInputDisconnected,
    }:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "handler.outcome",
            "handler must return a typed public input outcome",
        )
    return cast(AttachedInputOutcome, outcome)


def _validate_submission(submission: object) -> InputSubmission:
    if type(submission) not in (
        InputAnswerSubmission,
        InputDeclineSubmission,
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "submission",
            "value must be a typed input submission",
        )
    return cast(InputSubmission, submission)


def _validate_input_controller_bridge(
    bridge: InputControllerBridge,
) -> None:
    if not all(
        _is_async_callable(getattr(bridge, name, None))
        for name in ("inspect_input", "resolve_input")
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "input_bridge",
            "bridge inspection and resolution methods must be async",
        )


def _validate_durable_input_bridge(bridge: DurableInputBridge) -> None:
    _validate_input_controller_bridge(bridge)
    if not _is_async_callable(getattr(bridge, "persist_input", None)):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "durable_bridge",
            "bridge persistence method must be async",
        )


def _require_ref_echo(
    expected: InteractionCorrelation,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
    *,
    path: str,
) -> None:
    actual = _decode_correlation_pair(request_id, continuation_id)
    if actual != expected:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            path,
            "bridge result does not match the requested input references",
        )


def _persistence_digest(
    request_payload: DurableInputRequestPayload,
    continuation_payload: DurableInputContinuationPayload,
) -> str:
    return sha256(
        request_payload.encode("utf-8")
        + b"\x00"
        + continuation_payload.encode("utf-8")
    ).hexdigest()


def _validate_persistence_payload(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a non-empty serialized payload",
        )
    return value


def _validate_sha256(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be a lowercase SHA-256 digest",
        )
    return value


def _encode_correlation_ref(
    kind: Literal["request", "continuation"],
    correlation: InteractionCorrelation,
) -> str:
    payload = {
        "agent_id": str(correlation.agent_id),
        "branch_id": str(correlation.branch_id),
        "continuation_id": str(correlation.continuation_id),
        "kind": kind,
        "model_call_id": str(correlation.model_call_id),
        "request_id": str(correlation.request_id),
        "run_id": str(correlation.run_id),
        "task_id": (
            None if correlation.task_id is None else str(correlation.task_id)
        ),
        "turn_id": str(correlation.turn_id),
        "version": _INPUT_REF_VERSION,
    }
    encoded = urlsafe_b64encode(
        dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).rstrip(b"=")
    checksum = sha256(_INPUT_REF_DOMAIN + encoded).hexdigest()
    return f"{_INPUT_REF_PREFIX}.{encoded.decode('ascii')}.{checksum}"


def _decode_correlation_pair(
    request_ref: InputRequestRef,
    continuation_ref: InputContinuationRef,
) -> InteractionCorrelation:
    request = _decode_correlation_ref(request_ref, "request")
    continuation = _decode_correlation_ref(
        continuation_ref,
        "continuation",
    )
    if request != continuation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "input_refs",
            "request and continuation references do not match",
        )
    return request


def _decode_correlation_ref(
    value: object,
    expected_kind: Literal["request", "continuation"],
) -> InteractionCorrelation:
    if not isinstance(value, str):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "input_ref",
            "input reference must be a string",
        )
    if not value.isascii():
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref",
            "input reference has an unsupported format",
        )
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != _INPUT_REF_PREFIX:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref",
            "input reference has an unsupported format",
        )
    encoded = parts[1].encode("ascii", errors="strict")
    expected_checksum = sha256(_INPUT_REF_DOMAIN + encoded).hexdigest()
    if not compare_digest(parts[2], expected_checksum):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref.checksum",
            "input reference checksum is invalid",
        )
    try:
        padded = encoded + b"=" * (-len(encoded) % 4)
        payload = loads(urlsafe_b64decode(padded).decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError, ValueError):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref.payload",
            "input reference payload is invalid",
        ) from None
    if (
        not isinstance(payload, dict)
        or frozenset(payload) != _INPUT_REF_PAYLOAD_KEYS
        or payload.get("version") != _INPUT_REF_VERSION
        or payload.get("kind") != expected_kind
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref.payload",
            "input reference payload is invalid",
        )
    try:
        return InteractionCorrelation(
            request_id=payload["request_id"],
            continuation_id=payload["continuation_id"],
            run_id=payload["run_id"],
            turn_id=payload["turn_id"],
            task_id=payload["task_id"],
            agent_id=payload["agent_id"],
            branch_id=payload["branch_id"],
            model_call_id=payload["model_call_id"],
        )
    except (InputContractError, TypeError):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "input_ref.payload",
            "input reference payload is invalid",
        ) from None


def _validate_resolution_correlation(
    result: object,
    correlation: InteractionCorrelation,
) -> None:
    result_correlation: InteractionCorrelation | None = None
    if isinstance(
        result,
        ResolveInteractionApplied | InteractionStoreReplayed,
    ):
        result_correlation = result.record.correlation
    elif isinstance(result, ResolveInteractionRejected):
        result_correlation = result.command.correlation
    if result_correlation is not None and result_correlation != correlation:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "input_resolver",
            "input resolver returned an unrelated interaction",
        )


def _raise_for_non_candidate_state(state: RequestState) -> None:
    if state in {RequestState.ANSWERED, RequestState.DECLINED}:
        return
    if state is RequestState.EXPIRED:
        raise InputExpiredError()
    if state is RequestState.SUPERSEDED:
        raise InputSupersededError()
    raise InputAlreadyResolvedError()


def _require_pending_resolution_state(state: RequestState) -> None:
    if state in {
        RequestState.PENDING,
        RequestState.ANSWERED,
        RequestState.DECLINED,
    }:
        return
    if state is RequestState.EXPIRED:
        raise InputExpiredError()
    if state is RequestState.SUPERSEDED:
        raise InputSupersededError()
    raise InputAlreadyResolvedError()


def _raise_public_transition_error(
    code: InputErrorCode,
    path: str,
    message: str,
) -> None:
    if code in {
        InputErrorCode.ALREADY_RESOLVED,
        InputErrorCode.IDEMPOTENCY_CONFLICT,
    }:
        raise InputAlreadyResolvedError()
    if code is InputErrorCode.EXPIRED:
        raise InputExpiredError()
    if code is InputErrorCode.SUPERSEDED:
        raise InputSupersededError()
    if code is InputErrorCode.NOT_FOUND:
        raise InputNotFoundError()
    if code is InputErrorCode.FORBIDDEN:
        raise InputAuthorizationError()
    raise InputValidationError(code, path, message)


def _validate_json_value(value: object) -> None:
    try:
        dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError):
        raise TypeError("completed value is not JSON-compatible") from None
