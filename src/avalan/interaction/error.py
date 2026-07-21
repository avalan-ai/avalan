"""Define typed failures for the interaction domain."""

from enum import StrEnum


class InputErrorCode(StrEnum):
    """Identify a stable interaction failure category."""

    INVALID_TYPE = "input.invalid_type"
    EMPTY = "input.empty"
    OUT_OF_BOUNDS = "input.out_of_bounds"
    INVALID_FORMAT = "input.invalid_format"
    DUPLICATE = "input.duplicate"
    INVALID_DEFAULT = "input.invalid_default"
    INVALID_RECOMMENDATION = "input.invalid_recommendation"
    UNKNOWN_QUESTION = "input.unknown_question"
    UNKNOWN_CHOICE = "input.unknown_choice"
    ANSWER_TYPE_MISMATCH = "input.answer_type_mismatch"
    MISSING_REQUIRED_ANSWER = "input.missing_required_answer"
    UNEXPECTED_ANSWER = "input.unexpected_answer"
    INVALID_CARDINALITY = "input.invalid_cardinality"
    OTHER_NOT_ALLOWED = "input.other_not_allowed"
    NAIVE_TIMESTAMP = "input.naive_timestamp"
    NON_JSON_VALUE = "input.non_json_value"
    CORRELATION_MISMATCH = "input.correlation_mismatch"
    ILLEGAL_TRANSITION = "input.illegal_transition"
    TIMED_OUT_REQUIRED = "input.timed_out_required"
    STALE_REVISION = "input.stale_revision"
    STATE_REVISION_EXHAUSTED = "input.state_revision_exhausted"
    SNAPSHOT_UNSUPPORTED = "input.continuation_snapshot_unsupported"
    SNAPSHOT_PROVIDER_UNAVAILABLE = "input.continuation_provider_unavailable"
    SNAPSHOT_REVISION_DRIFT = "input.continuation_revision_drift"
    SNAPSHOT_INVALID = "input.continuation_snapshot_invalid"
    SNAPSHOT_SECRET_PROHIBITED = (
        "input.continuation_snapshot_secret_prohibited"
    )
    PROHIBITED_INPUT = "input.secret_prohibited"
    SECRET_CLASSIFICATION_UNAVAILABLE = (
        "input.secret_classification_unavailable"
    )
    AUTHENTICATION_REQUIRED = "input.authentication_required"
    FORBIDDEN = "input.forbidden"
    NOT_FOUND = "input.not_found"
    ALREADY_RESOLVED = "input.already_resolved"
    EXPIRED = "input.expired"
    SUPERSEDED = "input.superseded"
    UNAVAILABLE = "input.unavailable"
    CAPACITY_EXCEEDED = "input.capacity_exceeded"
    INTERACTION_LOOP_LIMIT = "input.interaction_loop_limit"
    IDEMPOTENCY_CONFLICT = "input.idempotency_conflict"
    IDEMPOTENCY_LEDGER_FULL = "input.idempotency_ledger_full"
    STORE_CLOSED = "input.store_closed"


class InputContractError(ValueError):
    """Report a content-safe canonical interaction failure."""

    def __init__(
        self,
        code: InputErrorCode,
        path: str,
        message: str,
    ) -> None:
        assert isinstance(code, InputErrorCode)
        assert isinstance(path, str) and path
        assert isinstance(message, str) and message
        self.code = code
        self.path = path
        self.safe_message = message
        super().__init__(f"{code.value}: {path}: {message}")


class InputValidationError(InputContractError):
    """Report invalid canonical interaction content."""


class InputCodecError(InputValidationError):
    """Report an invalid interaction wire representation."""


class InputSnapshotError(InputCodecError):
    """Report an invalid or unsupported interaction snapshot."""


class InteractionStoreClosedError(InputContractError):
    """Report access through one closed interaction-store handle."""

    def __init__(self) -> None:
        super().__init__(
            InputErrorCode.STORE_CLOSED,
            "store.handle",
            "interaction store handle is closed",
        )


class InteractionNotFoundError(InputContractError):
    """Report indistinguishable missing or unauthorized interaction access."""

    def __init__(self) -> None:
        super().__init__(
            InputErrorCode.NOT_FOUND,
            "interaction",
            "interaction was not found",
        )
