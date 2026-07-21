"""Apply pure deterministic interaction state transitions."""

from .entities import (
    AnsweredResolution,
    CancellationScope,
    CancelledResolution,
    DeclinedResolution,
    InputAnsweredResult,
    InputCancelledResult,
    InputContinuationOutcome,
    InputDeclinedResult,
    InputModelResult,
    InputRequest,
    InputResolution,
    InputTimedOutResult,
    InputUnavailableResult,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    StateRevision,
    TerminateInputContinuation,
    TimedOutResolution,
    UnavailableResolution,
    _is_input_resolution_variant,
    _validate_resolution_against_request,
)
from .error import InputErrorCode, InputValidationError
from .validation import (
    MAX_STATE_REVISION,
    validate_aware_datetime,
    validate_state_revision,
)

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, TypeAlias, final


class TransitionResultType(StrEnum):
    """Identify whether a pure lifecycle transition was accepted."""

    APPLIED = "applied"
    REJECTED = "rejected"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputTransitionError:
    """Describe a content-safe rejected transition."""

    code: InputErrorCode
    path: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, InputErrorCode):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "error.code",
                "value must be an input error code",
            )
        if not isinstance(self.path, str) or not self.path:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "error.path",
                "value must be a non-empty string",
            )
        if not isinstance(self.message, str) or not self.message:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "error.message",
                "value must be a non-empty string",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputTransitionResultBase:
    """Store the prior request for every transition outcome."""

    previous: InputRequest

    def __post_init__(self) -> None:
        if not _is_input_transition_result_variant(self):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "transition",
                "value must be a supported input transition result variant",
            )
        if type(self.previous) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "previous",
                "value must be an input request",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputTransitionApplied(_InputTransitionResultBase):
    """Return the immutable request produced by a legal transition."""

    request: InputRequest
    mutation_applied: bool
    kind: Literal[TransitionResultType.APPLIED] = field(
        init=False,
        default=TransitionResultType.APPLIED,
    )

    def __post_init__(self) -> None:
        _InputTransitionResultBase.__post_init__(self)
        if type(self.request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "request",
                "value must be an input request",
            )
        if not isinstance(self.mutation_applied, bool):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "mutation_applied",
                "value must be a boolean",
            )
        if self.mutation_applied:
            if self.request.state_revision != self.previous.state_revision + 1:
                raise InputValidationError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "state_revision",
                    "mutation must increment the revision exactly once",
                )
            if not _same_request_contract(self.previous, self.request):
                raise InputValidationError(
                    InputErrorCode.CORRELATION_MISMATCH,
                    "request",
                    "transition must preserve the immutable request contract",
                )
            legal_state_change = (
                self.previous.state is RequestState.CREATED
                and self.request.state is RequestState.PENDING
            ) or (
                self.previous.state is RequestState.PENDING
                and self.request.state
                not in {RequestState.CREATED, RequestState.PENDING}
            )
            if not legal_state_change:
                raise InputValidationError(
                    InputErrorCode.ILLEGAL_TRANSITION,
                    "request.state",
                    "applied transition contains an illegal state change",
                )
        elif self.request is not self.previous:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "request",
                "idempotent replay must preserve the stored request",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputTransitionRejected(_InputTransitionResultBase):
    """Return a typed error while preserving the exact prior request."""

    error: InputTransitionError
    kind: Literal[TransitionResultType.REJECTED] = field(
        init=False,
        default=TransitionResultType.REJECTED,
    )

    def __post_init__(self) -> None:
        _InputTransitionResultBase.__post_init__(self)
        if not isinstance(self.error, InputTransitionError):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "error",
                "value must be an input transition error",
            )


InputTransitionResult: TypeAlias = (
    InputTransitionApplied | InputTransitionRejected
)


_INPUT_TRANSITION_RESULT_VARIANT_TYPES: tuple[
    type[_InputTransitionResultBase], ...
] = (
    InputTransitionApplied,
    InputTransitionRejected,
)


def _is_input_transition_result_variant(value: object) -> bool:
    return type(value) in _INPUT_TRANSITION_RESULT_VARIANT_TYPES


def _same_request_contract(
    previous: InputRequest, request: InputRequest
) -> bool:
    return (
        previous.request_id == request.request_id
        and previous.continuation_id == request.continuation_id
        and previous.origin == request.origin
        and previous.mode is request.mode
        and previous.reason == request.reason
        and previous.questions == request.questions
        and previous.created_at == request.created_at
        and previous.continuation_ttl_seconds
        == request.continuation_ttl_seconds
        and previous.advisory_wait_seconds == request.advisory_wait_seconds
        and (
            previous.state is RequestState.CREATED
            or previous.advisory_deadline == request.advisory_deadline
        )
        and previous.interaction_class is request.interaction_class
    )


def mark_request_pending(
    request: InputRequest,
    *,
    expected_state_revision: StateRevision,
) -> InputTransitionResult:
    """Admit one created request into the authoritative queued state."""
    if type(request) is not InputRequest:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "request",
            "value must be an input request",
        )
    error = _precondition_error(
        request,
        expected_state_revision,
        expected_state=RequestState.CREATED,
    )
    if error is not None:
        return InputTransitionRejected(previous=request, error=error)
    next_revision = _next_revision(request)
    if isinstance(next_revision, InputTransitionError):
        return InputTransitionRejected(previous=request, error=next_revision)
    updated = replace(
        request,
        state=RequestState.PENDING,
        state_revision=next_revision,
    )
    return InputTransitionApplied(
        previous=request,
        request=updated,
        mutation_applied=True,
    )


def resolve_request(
    request: InputRequest,
    resolution: InputResolution,
    *,
    expected_state_revision: StateRevision,
) -> InputTransitionResult:
    """Validate and apply one exactly-once terminal resolution."""
    if type(request) is not InputRequest:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "request",
            "value must be an input request",
        )
    if not _is_input_resolution_variant(resolution):
        return InputTransitionRejected(
            previous=request,
            error=InputTransitionError(
                code=InputErrorCode.INVALID_TYPE,
                path="resolution",
                message="value must be a supported input resolution variant",
            ),
        )
    error = _revision_error(request, expected_state_revision)
    if error is not None:
        return InputTransitionRejected(previous=request, error=error)
    if request.state not in {RequestState.CREATED, RequestState.PENDING}:
        if request.resolution == resolution:
            return InputTransitionApplied(
                previous=request,
                request=request,
                mutation_applied=False,
            )
        return InputTransitionRejected(
            previous=request,
            error=InputTransitionError(
                code=InputErrorCode.ILLEGAL_TRANSITION,
                path="request.state",
                message="request has already resolved",
            ),
        )
    if request.state is not RequestState.PENDING:
        return InputTransitionRejected(
            previous=request,
            error=InputTransitionError(
                code=InputErrorCode.ILLEGAL_TRANSITION,
                path="request.state",
                message="request state does not permit resolution",
            ),
        )
    try:
        _validate_resolution_against_request(request, resolution)
    except InputValidationError as exc:
        return InputTransitionRejected(
            previous=request,
            error=InputTransitionError(
                code=exc.code,
                path=exc.path,
                message=exc.safe_message,
            ),
        )
    next_revision = _next_revision(request)
    if isinstance(next_revision, InputTransitionError):
        return InputTransitionRejected(previous=request, error=next_revision)
    updated = replace(
        request,
        state=RequestState(resolution.status.value),
        state_revision=next_revision,
        resolution=resolution,
    )
    return InputTransitionApplied(
        previous=request,
        request=updated,
        mutation_applied=True,
    )


def project_resolution_to_model(
    request: InputRequest,
    *,
    containing_run_exists: bool,
) -> InputContinuationOutcome:
    """Apply the frozen outcome-to-model continuation matrix."""
    if type(request) is not InputRequest:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "request",
            "value must be an input request",
        )
    if not isinstance(containing_run_exists, bool):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "containing_run_exists",
            "value must be a boolean",
        )
    resolution = request.resolution
    if resolution is not None and not _is_input_resolution_variant(resolution):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "resolution",
            "value must be a supported input resolution variant",
        )
    if resolution is None or request.state in {
        RequestState.CREATED,
        RequestState.PENDING,
    }:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "request.state",
            "request must have a terminal resolution",
        )
    containing_run_cancelled = (
        type(resolution) is CancelledResolution
        and resolution.scope is CancellationScope.CONTAINING_RUN
    )
    if (
        not containing_run_exists
        or containing_run_cancelled
        or resolution.status
        in {
            ResolutionStatus.EXPIRED,
            ResolutionStatus.SUPERSEDED,
        }
    ):
        return TerminateInputContinuation(
            request_id=request.request_id,
            status=resolution.status,
        )
    result = _model_result(resolution)
    return ResumeInputContinuation(
        request_id=request.request_id,
        result=result,
    )


def _precondition_error(
    request: InputRequest,
    expected_state_revision: StateRevision,
    *,
    expected_state: RequestState,
) -> InputTransitionError | None:
    revision_error = _revision_error(request, expected_state_revision)
    if revision_error is not None:
        return revision_error
    if request.state is not expected_state:
        return InputTransitionError(
            code=InputErrorCode.ILLEGAL_TRANSITION,
            path="request.state",
            message="request state does not permit this transition",
        )
    return None


def _revision_error(
    request: InputRequest,
    expected_state_revision: StateRevision,
) -> InputTransitionError | None:
    try:
        revision = validate_state_revision(
            expected_state_revision,
            "expected_state_revision",
        )
    except InputValidationError as exc:
        return InputTransitionError(
            code=exc.code,
            path=exc.path,
            message=exc.safe_message,
        )
    if revision != request.state_revision:
        return InputTransitionError(
            code=InputErrorCode.STALE_REVISION,
            path="expected_state_revision",
            message="request revision is stale",
        )
    return None


def _next_revision(
    request: InputRequest,
) -> StateRevision | InputTransitionError:
    if request.state_revision == MAX_STATE_REVISION:
        return InputTransitionError(
            code=InputErrorCode.STATE_REVISION_EXHAUSTED,
            path="state_revision",
            message="request state revision is exhausted",
        )
    return StateRevision(request.state_revision + 1)


def _anchor_request_presentation(
    request: InputRequest,
    presented_at: datetime,
) -> InputRequest:
    """Anchor advisory timing without changing lifecycle state or revision."""
    if type(request) is not InputRequest:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "request",
            "value must be an input request",
        )
    if request.state is not RequestState.PENDING:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "request.state",
            "only pending requests can record presentation",
        )
    presented = validate_aware_datetime(
        presented_at,
        "presented_at",
    )
    if presented < request.created_at:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "presented_at",
            "presentation timestamp predates request creation",
        )
    if request.mode is RequirementMode.REQUIRED:
        return request
    if request.advisory_deadline is not None:
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "request.advisory_deadline",
            "advisory presentation is already anchored",
        )
    assert request.advisory_wait_seconds is not None
    try:
        deadline = presented + timedelta(seconds=request.advisory_wait_seconds)
    except OverflowError as exc:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            "presented_at",
            "presentation timestamp cannot represent the advisory deadline",
        ) from exc
    return replace(request, advisory_deadline=deadline)


def _model_result(resolution: InputResolution) -> InputModelResult:
    if not _is_input_resolution_variant(resolution):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "resolution",
            "value must be a supported input resolution variant",
        )
    if type(resolution) is AnsweredResolution:
        return InputAnsweredResult(
            request_id=resolution.request_id,
            provenance=resolution.provenance,
            resolved_at=resolution.resolved_at,
            answers=resolution.answers,
        )
    if type(resolution) is DeclinedResolution:
        return InputDeclinedResult(
            request_id=resolution.request_id,
            provenance=resolution.provenance,
            resolved_at=resolution.resolved_at,
        )
    if type(resolution) is CancelledResolution:
        return InputCancelledResult(
            request_id=resolution.request_id,
            provenance=resolution.provenance,
            resolved_at=resolution.resolved_at,
        )
    if type(resolution) is TimedOutResolution:
        return InputTimedOutResult(
            request_id=resolution.request_id,
            provenance=resolution.provenance,
            resolved_at=resolution.resolved_at,
        )
    if type(resolution) is UnavailableResolution:
        return InputUnavailableResult(
            request_id=resolution.request_id,
            provenance=resolution.provenance,
            resolved_at=resolution.resolved_at,
        )
    raise InputValidationError(
        InputErrorCode.ILLEGAL_TRANSITION,
        "resolution.status",
        "terminal resolution cannot be returned to the model",
    )
