"""Define explicit async policies for headless task-input handling."""

from .durable import DurableInteractionSuspension
from .entities import (
    AnsweredResolution,
    AnswerProvenance,
    DeclinedResolution,
    InputAnswer,
    InputRequest,
    RequestState,
    StateRevision,
    _is_input_answer_variant,
)
from .error import InputErrorCode, InputValidationError
from .handler import (
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    _InputHandlerOutcome,
    _is_async_callable,
    _new_trusted_default_input_handler_resolution,
    _new_trusted_policy_input_handler_resolution,
)
from .validation import validate_int

from asyncio import CancelledError, Future, ensure_future, sleep
from asyncio import wait as wait_for_tasks
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import NoReturn, Protocol, TypeAlias, final

MIN_DURABLE_HANDOFF_WAIT_SECONDS = 1
DEFAULT_DURABLE_HANDOFF_WAIT_SECONDS = 30
MAX_DURABLE_HANDOFF_WAIT_SECONDS = 300


class InputCancellationHandler(Protocol):
    """Receive cancellation of one no-longer-needed input request."""

    async def __call__(self, context: InputHandlerContext) -> None:
        """Handle cancellation of an attached request."""
        ...


class InputPolicyValueProvider(Protocol):
    """Compute typed policy-owned answers asynchronously."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> tuple[InputAnswer, ...]:
        """Return policy-provenance answers for one semantic request."""
        ...


class DurableHandoffWaiter(Protocol):
    """Wait asynchronously before a durable suspension is returned."""

    async def __call__(self, wait_seconds: int) -> None:
        """Wait for at most the configured caller handoff budget."""
        ...


class DurableHandoffHandler(Protocol):
    """Persist one durable suspension through a trusted host boundary."""

    async def __call__(
        self,
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        """Return the exact authoritative pending request."""
        ...


@final
class AsyncioDurableHandoffWaiter:
    """Wait with the active event loop without creating a sync bridge."""

    async def __call__(self, wait_seconds: int) -> None:
        """Wait for the configured caller handoff budget."""
        await sleep(wait_seconds)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CancellationAwarePolicy:
    """Share exact async cancellation delivery across headless handlers."""

    cancellation_handler: (
        Callable[[InputHandlerContext], Awaitable[None]] | None
    ) = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.cancellation_handler is not None and not _is_async_callable(
            self.cancellation_handler
        ):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.cancellation_handler",
                "value must be an async cancellation handler",
            )

    async def _cancelled(self, context: InputHandlerContext) -> None:
        handler = self.cancellation_handler
        if handler is not None:
            await handler(context)

    async def _raise_cancelled(
        self,
        context: InputHandlerContext,
        cancellation: CancelledError,
    ) -> NoReturn:
        try:
            await self._cancelled(context)
        except BaseException as callback_error:
            cancellation.add_note(
                "input cancellation callback failed: "
                f"{type(callback_error).__name__}"
            )
        raise cancellation


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PredeclaredInputPolicy(_CancellationAwarePolicy):
    """Answer from immutable values declared by trusted host policy."""

    answers: tuple[InputAnswer, ...]

    def __post_init__(self) -> None:
        _CancellationAwarePolicy.__post_init__(self)
        _validate_policy_answers(self.answers, "headless.answers")

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        """Return predeclared policy values for one matching request."""
        _raise_validation_feedback(context)
        return _policy_answer_outcome(context, self.answers)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyValueInputPolicy(_CancellationAwarePolicy):
    """Resolve values through one trusted async policy provider."""

    provider: Callable[
        [InputHandlerContext],
        Awaitable[tuple[InputAnswer, ...]],
    ] = field(repr=False)

    def __post_init__(self) -> None:
        _CancellationAwarePolicy.__post_init__(self)
        if not _is_async_callable(self.provider):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.provider",
                "value must be an async policy-value provider",
            )

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        """Return values computed by the trusted async policy provider."""
        try:
            answers = await self.provider(context)
            _validate_policy_answers(answers, "headless.provider.answers")
            return _policy_answer_outcome(context, answers)
        except CancelledError as cancellation:
            await self._raise_cancelled(context, cancellation)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalControllerInputPolicy(_CancellationAwarePolicy):
    """Delegate attached handling to one explicit async controller."""

    controller: Callable[
        [InputHandlerContext],
        Awaitable[InputHandlerOutcome],
    ] = field(repr=False)

    def __post_init__(self) -> None:
        _CancellationAwarePolicy.__post_init__(self)
        if not _is_async_callable(self.controller):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.controller",
                "value must be an async input controller",
            )

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return one typed external-controller outcome."""
        try:
            outcome = await self.controller(context)
            _validate_external_controller_outcome(outcome)
            return outcome
        except CancelledError as cancellation:
            await self._raise_cancelled(context, cancellation)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedDefaultInputPolicy(_CancellationAwarePolicy):
    """Apply only defaults declared by the semantic request."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        """Return the request-derived trusted-default resolution."""
        _raise_validation_feedback(context)
        return _new_trusted_default_input_handler_resolution()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DeclineInputPolicy(_CancellationAwarePolicy):
    """Decline every request through explicit policy provenance."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> _InputHandlerOutcome:
        """Return an explicit policy decline."""
        return _new_trusted_policy_input_handler_resolution(
            DeclinedResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=context.request.created_at,
            )
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableHandoffInputPolicy(_CancellationAwarePolicy):
    """Detach after one bounded caller wait without changing input expiry."""

    handoff: Callable[
        [DurableInteractionSuspension],
        Awaitable[InputRequest],
    ] = field(repr=False)
    durable_handoff_wait_seconds: int = DEFAULT_DURABLE_HANDOFF_WAIT_SECONDS
    waiter: Callable[[int], Awaitable[None]] = field(
        default_factory=AsyncioDurableHandoffWaiter,
        repr=False,
    )

    def __post_init__(self) -> None:
        _CancellationAwarePolicy.__post_init__(self)
        if not _is_async_callable(self.handoff):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.handoff",
                "value must be an async durable-handoff handler",
            )
        object.__setattr__(
            self,
            "durable_handoff_wait_seconds",
            validate_int(
                self.durable_handoff_wait_seconds,
                "headless.durable_handoff_wait_seconds",
                minimum=MIN_DURABLE_HANDOFF_WAIT_SECONDS,
                maximum=MAX_DURABLE_HANDOFF_WAIT_SECONDS,
            ),
        )
        if not _is_async_callable(self.waiter):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.waiter",
                "value must be an async durable-handoff waiter",
            )

    async def persist(
        self,
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        """Persist and return one exact authoritative pending request."""
        if type(suspension) is not DurableInteractionSuspension:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "headless.suspension",
                "value must be a durable interaction suspension",
            )
        request = await self.handoff(suspension)
        return _validate_handoff_request(suspension, request)

    async def wait(self) -> None:
        """Wait only for the bounded synchronous-caller handoff budget."""
        waiter: Future[None] = ensure_future(
            self.waiter(self.durable_handoff_wait_seconds)
        )
        try:
            completed, _pending = await wait_for_tasks(
                (waiter,),
                timeout=self.durable_handoff_wait_seconds,
            )
        except CancelledError:
            _cancel_waiter(waiter)
            raise
        if waiter not in completed:
            _cancel_waiter(waiter)
            return
        await waiter

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Request detached handling without manufacturing an answer."""
        del context
        return InputHandlerDetached()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class UnavailableInputPolicy(_CancellationAwarePolicy):
    """Mark attached headless input unavailable without an answer."""

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return an explicit unavailable control-channel outcome."""
        del context
        return InputHandlerDisconnected(
            reason=InputDisconnectReason.HANDLER_UNAVAILABLE
        )


HeadlessInputPolicy: TypeAlias = (
    PredeclaredInputPolicy
    | PolicyValueInputPolicy
    | ExternalControllerInputPolicy
    | TrustedDefaultInputPolicy
    | DeclineInputPolicy
    | DurableHandoffInputPolicy
    | UnavailableInputPolicy
)


def _cancel_waiter(waiter: Future[None]) -> None:
    waiter.add_done_callback(_consume_waiter_outcome)
    waiter.cancel()


def _consume_waiter_outcome(waiter: Future[None]) -> None:
    if not waiter.cancelled():
        waiter.exception()


def _validate_policy_answers(
    answers: object,
    path: str,
) -> tuple[InputAnswer, ...]:
    if not isinstance(answers, tuple):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "policy answers must be a tuple",
        )
    if not all(_is_input_answer_variant(answer) for answer in answers):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "policy answers must contain typed input answers",
        )
    normalized = answers
    if any(
        answer.provenance is not AnswerProvenance.POLICY
        for answer in normalized
    ):
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            path,
            "policy answers require policy provenance",
        )
    return normalized


def _validate_handoff_request(
    suspension: DurableInteractionSuspension,
    request: object,
) -> InputRequest:
    if type(request) is not InputRequest:
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless.handoff.request",
            "handoff must return an authoritative input request",
        )
    assert isinstance(request, InputRequest)
    staged = suspension.command.request
    expected = replace(
        staged,
        state=RequestState.PENDING,
        state_revision=StateRevision(int(staged.state_revision) + 1),
    )
    if request != expected:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "headless.handoff.request",
            "persisted request does not match the durable suspension",
        )
    return request


def _policy_answer_outcome(
    context: InputHandlerContext,
    answers: tuple[InputAnswer, ...],
) -> _InputHandlerOutcome:
    return _new_trusted_policy_input_handler_resolution(
        AnsweredResolution(
            request_id=context.request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=context.request.created_at,
            answers=answers,
        )
    )


def _raise_validation_feedback(context: InputHandlerContext) -> None:
    error = context.validation_error
    if error is not None:
        raise InputValidationError(
            error.code,
            error.path,
            error.message,
        )


def _validate_external_controller_outcome(
    outcome: InputHandlerOutcome,
) -> None:
    if not isinstance(
        outcome,
        (
            InputHandlerResolution,
            InputHandlerDetached,
            InputHandlerDisconnected,
        ),
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "headless.controller.outcome",
            "controller must return a typed input-handler outcome",
        )
    if not isinstance(outcome, InputHandlerResolution):
        return
    resolution = outcome.resolution
    if resolution.provenance not in {
        AnswerProvenance.HUMAN,
        AnswerProvenance.EXTERNAL_CONTROLLER,
    }:
        raise InputValidationError(
            InputErrorCode.FORBIDDEN,
            "headless.controller.resolution.provenance",
            "external controllers cannot claim trusted provenance",
        )
    if isinstance(resolution, AnsweredResolution) and any(
        answer.provenance != resolution.provenance
        for answer in resolution.answers
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "headless.controller.resolution.answers",
            "answer provenance must match its external resolution",
        )
