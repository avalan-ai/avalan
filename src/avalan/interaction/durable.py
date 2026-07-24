"""Define deferred durable interaction persistence values."""

from .continuation import (
    ContinuationClaimState,
    PortableContinuation,
)
from .error import InputErrorCode, InputValidationError
from .store import CreateInteractionCommand

from dataclasses import dataclass
from datetime import timedelta
from typing import final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInteractionSuspension:
    """Carry one uncommitted request and its portable continuation."""

    command: CreateInteractionCommand
    continuation: PortableContinuation

    def __post_init__(self) -> None:
        if type(self.command) is not CreateInteractionCommand:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_suspension.command",
                "command must create one interaction",
            )
        if type(self.continuation) is not PortableContinuation:
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "durable_suspension.continuation",
                "continuation must be portable",
            )
        if self.command.resumer is not None:
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "durable_suspension.command.resumer",
                "deferred durable interactions cannot retain a live resumer",
            )
        request = self.command.request
        continuation = self.continuation
        if request.request_id != continuation.request_id:
            self._raise_correlation_mismatch("request_id")
        if request.continuation_id != continuation.continuation_id:
            self._raise_correlation_mismatch("continuation_id")
        if request.origin != continuation.origin:
            self._raise_correlation_mismatch("origin")
        if request.origin.definition != continuation.definition:
            self._raise_correlation_mismatch("definition")
        if request.origin.model_call_id != continuation.provider_call_id:
            self._raise_correlation_mismatch("provider_call_id")
        if self.command.actor.principal != request.origin.principal:
            self._raise_correlation_mismatch("principal")
        absolute_expires_at = request.created_at + timedelta(
            seconds=request.continuation_ttl_seconds
        )
        if absolute_expires_at != continuation.expires_at:
            self._raise_correlation_mismatch("expires_at")
        if (
            continuation.claim.state is not ContinuationClaimState.UNCLAIMED
            or int(continuation.store_revision) != 0
            or int(continuation.fencing_token) != 0
            or continuation.dispatch is not None
            or continuation.completion is not None
        ):
            raise InputValidationError(
                InputErrorCode.ILLEGAL_TRANSITION,
                "durable_suspension.continuation",
                "deferred continuation must be unclaimed and unpersisted",
            )

    @staticmethod
    def _raise_correlation_mismatch(field_name: str) -> None:
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            f"durable_suspension.{field_name}",
            "deferred request and continuation do not match",
        )
