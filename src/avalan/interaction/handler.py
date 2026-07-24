"""Define async attached-handler and in-process resumer contracts."""

from .entities import (
    ContinuationId,
    InputCandidateResolution,
    InputContinuationOutcome,
    InputRequest,
    StateRevision,
    _is_input_candidate_resolution,
    _is_input_continuation_outcome_variant,
)
from .error import InputErrorCode, InputValidationError
from .state import InputTransitionError
from .validation import validate_opaque_id, validate_state_revision

from dataclasses import dataclass, field
from enum import StrEnum
from inspect import iscoroutinefunction
from typing import Callable, Literal, Protocol, TypeAlias, cast, final


class InputHandlerResultKind(StrEnum):
    """Identify the result of one attached handler invocation."""

    RESOLUTION = "resolution"
    DETACHED = "detached"
    DISCONNECTED = "disconnected"


class InputDisconnectReason(StrEnum):
    """Identify a content-safe attached-channel disconnect reason."""

    CONTROL_CHANNEL_CLOSED = "control_channel_closed"
    HANDLER_CANCELLED = "handler_cancelled"
    HANDLER_UNAVAILABLE = "handler_unavailable"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputHandlerContext:
    """Provide one immutable request and optional correction feedback."""

    request: InputRequest
    validation_error: InputTransitionError | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not InputRequest:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.request",
                "value must be an input request",
            )
        if self.validation_error is not None and not isinstance(
            self.validation_error, InputTransitionError
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.validation_error",
                "value must be an input transition error",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputHandlerResolution:
    """Return one typed resolution from an attached handler."""

    resolution: InputCandidateResolution
    kind: Literal[InputHandlerResultKind.RESOLUTION] = field(
        init=False,
        default=InputHandlerResultKind.RESOLUTION,
    )

    def __post_init__(self) -> None:
        if not _is_input_candidate_resolution(self.resolution):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.resolution",
                "attached handlers may only answer or decline",
            )


_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN = object()


@final
@dataclass(frozen=True, slots=True, init=False)
class _TrustedInputHandlerResolution:
    """Carry one sealed trusted-host handler decision."""

    resolution: InputCandidateResolution | None
    trusted_default: bool
    _authority: object = field(repr=False)
    kind: Literal[InputHandlerResultKind.RESOLUTION] = field(
        init=False,
        default=InputHandlerResultKind.RESOLUTION,
    )

    def __init__(
        self,
        *,
        resolution: InputCandidateResolution | None,
        trusted_default: bool,
        _token: object,
    ) -> None:
        if _token is not _TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN:
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "handler.trusted_authority",
                "trusted handler outcomes must be minted by the host",
            )
        if type(trusted_default) is not bool:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.trusted_default",
                "value must be a boolean",
            )
        if trusted_default:
            if resolution is not None:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "handler.resolution",
                    "trusted-default outcomes cannot carry a candidate",
                )
        elif not _is_input_candidate_resolution(resolution):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.resolution",
                "trusted policy must carry an answer or decline",
            )
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "trusted_default", trusted_default)
        object.__setattr__(self, "_authority", _token)
        object.__setattr__(
            self,
            "kind",
            InputHandlerResultKind.RESOLUTION,
        )


def _new_trusted_default_input_handler_resolution() -> (
    _TrustedInputHandlerResolution
):
    """Mint one sealed request-derived trusted-default outcome."""
    return _TrustedInputHandlerResolution(
        resolution=None,
        trusted_default=True,
        _token=_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
    )


def _new_trusted_policy_input_handler_resolution(
    resolution: InputCandidateResolution,
) -> _TrustedInputHandlerResolution:
    """Mint one sealed trusted-policy candidate outcome."""
    return _TrustedInputHandlerResolution(
        resolution=resolution,
        trusted_default=False,
        _token=_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
    )


def _validate_trusted_input_handler_resolution(
    outcome: object,
) -> _TrustedInputHandlerResolution:
    """Return one exactly sealed trusted-host handler outcome."""
    if type(outcome) is not _TrustedInputHandlerResolution:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "handler.outcome",
            "value must be a trusted host handler outcome",
        )
    assert isinstance(outcome, _TrustedInputHandlerResolution)
    if outcome._authority is not _TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "handler.trusted_authority",
            "trusted handler outcome authority is invalid",
        )
    return outcome


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputHandlerDetached:
    """Request durable or external handling without fabricating an answer."""

    kind: Literal[InputHandlerResultKind.DETACHED] = field(
        init=False,
        default=InputHandlerResultKind.DETACHED,
    )

    def __post_init__(self) -> None:
        if type(self) is not InputHandlerDetached:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.outcome",
                "value must be a detached handler outcome",
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputHandlerDisconnected:
    """Report channel loss without representing an intentional decline."""

    reason: InputDisconnectReason
    kind: Literal[InputHandlerResultKind.DISCONNECTED] = field(
        init=False,
        default=InputHandlerResultKind.DISCONNECTED,
    )

    def __post_init__(self) -> None:
        if type(self) is not InputHandlerDisconnected:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.outcome",
                "value must be a disconnected handler outcome",
            )
        if not isinstance(self.reason, InputDisconnectReason):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "handler.disconnect_reason",
                "value must be an input disconnect reason",
            )


InputHandlerOutcome: TypeAlias = (
    InputHandlerResolution | InputHandlerDetached | InputHandlerDisconnected
)


class InputHandler(Protocol):
    """Present a request and asynchronously return a typed outcome."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Handle one attached request or correction attempt."""
        ...


_InputHandlerOutcome: TypeAlias = (
    InputHandlerOutcome | _TrustedInputHandlerResolution
)


class _InputHandler(Protocol):
    """Handle input through either public or sealed trusted outcomes."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        """Handle one attached request or trusted policy attempt."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputResumptionNotification:
    """Carry one committed outcome to an in-process continuation waiter."""

    continuation_id: ContinuationId
    state_revision: StateRevision
    outcome: InputContinuationOutcome

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(
                    self.continuation_id,
                    "resumption.continuation_id",
                )
            ),
        )
        object.__setattr__(
            self,
            "state_revision",
            StateRevision(
                validate_state_revision(
                    self.state_revision,
                    "resumption.state_revision",
                )
            ),
        )
        if not _is_input_continuation_outcome_variant(self.outcome):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resumption.outcome",
                "value must be an input continuation outcome",
            )


class InputResumer(Protocol):
    """Receive one committed in-process continuation notification."""

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Notify one in-process continuation waiter."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class InputResumerRegistration:
    """Bind one continuation to a non-persisted in-process resumer."""

    continuation_id: ContinuationId
    resumer: InputResumer

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_id",
            ContinuationId(
                validate_opaque_id(
                    self.continuation_id,
                    "resumer.continuation_id",
                )
            ),
        )
        if not _is_async_callable(self.resumer):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "resumer.callback",
                "value must be an async input resumer",
            )


def _is_async_callable(value: object) -> bool:
    if not callable(value):
        return False
    callback = cast(Callable[..., object], value)
    if iscoroutinefunction(callback):
        return True
    bound_call = getattr(value, "__call__", None)
    return callable(bound_call) and iscoroutinefunction(
        cast(Callable[..., object], bound_call)
    )
