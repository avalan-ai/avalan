"""Define stable, content-safe conversation domain errors."""

from .contract import FailureBoundary

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class ConversationErrorCode(StrEnum):
    """Identify one stable conversation failure category."""

    VALIDATION_FAILED = "conversation_validation_failed"
    CAPABILITY_UNSUPPORTED = "conversation_capability_unsupported"
    BINDING_DRIFT = "conversation_binding_drift"
    CONFLICT = "conversation_conflict"
    INTEGRITY_FAILED = "conversation_integrity_failed"
    EXPIRED = "conversation_expired"
    DELETED = "conversation_deleted"
    STORAGE_FAILED = "conversation_storage_failed"
    DISPATCH_AMBIGUOUS = "conversation_dispatch_ambiguous"
    COMMIT_FAILED = "conversation_state_commit_failed"
    PUBLICATION_FAILED = "conversation_publication_failed"
    AUTHORIZATION_FAILED = "conversation_authorization_failed"
    LIMIT_EXCEEDED = "conversation_limit_exceeded"
    CODEC_FAILED = "conversation_codec_failed"
    TRANSITION_INVALID = "conversation_transition_invalid"


class DurableConversationErrorCode(StrEnum):
    """Identify one stable durable conversation failure category."""

    KEY_MISSING = "conversation_key_missing"
    KEY_RETIRED = "conversation_key_retired"
    KEY_POLICY_INVALID = "conversation_key_policy_invalid"
    CRYPTO_AUTHENTICATION_FAILED = "conversation_crypto_authentication_failed"
    FEATURE_UNAVAILABLE = "conversation_feature_unavailable"
    MIGRATION_REQUIRED = "conversation_migration_required"


_ERROR_MESSAGES: Mapping[
    ConversationErrorCode | DurableConversationErrorCode,
    str,
] = MappingProxyType(
    {
        ConversationErrorCode.VALIDATION_FAILED: (
            "conversation input is invalid"
        ),
        ConversationErrorCode.CAPABILITY_UNSUPPORTED: (
            "the requested conversation capability is unsupported"
        ),
        ConversationErrorCode.BINDING_DRIFT: (
            "the provider-lane binding changed"
        ),
        ConversationErrorCode.CONFLICT: (
            "the conversation operation conflicts with current state"
        ),
        ConversationErrorCode.INTEGRITY_FAILED: (
            "conversation state failed integrity validation"
        ),
        ConversationErrorCode.EXPIRED: "conversation state is unavailable",
        ConversationErrorCode.DELETED: "conversation state is unavailable",
        ConversationErrorCode.STORAGE_FAILED: "conversation storage failed",
        ConversationErrorCode.DISPATCH_AMBIGUOUS: (
            "provider dispatch outcome is ambiguous"
        ),
        ConversationErrorCode.COMMIT_FAILED: (
            "conversation state commit failed"
        ),
        ConversationErrorCode.PUBLICATION_FAILED: (
            "conversation publication failed"
        ),
        ConversationErrorCode.AUTHORIZATION_FAILED: (
            "conversation state is unavailable"
        ),
        ConversationErrorCode.LIMIT_EXCEEDED: (
            "conversation state exceeds a configured limit"
        ),
        ConversationErrorCode.CODEC_FAILED: (
            "encoded conversation state is invalid"
        ),
        ConversationErrorCode.TRANSITION_INVALID: (
            "conversation state transition is invalid"
        ),
        DurableConversationErrorCode.KEY_MISSING: (
            "conversation encryption key is unavailable"
        ),
        DurableConversationErrorCode.KEY_RETIRED: (
            "conversation encryption key is retired"
        ),
        DurableConversationErrorCode.KEY_POLICY_INVALID: (
            "conversation encryption key policy is invalid"
        ),
        DurableConversationErrorCode.CRYPTO_AUTHENTICATION_FAILED: (
            "conversation ciphertext authentication failed"
        ),
        DurableConversationErrorCode.FEATURE_UNAVAILABLE: (
            "durable conversation storage is unavailable"
        ),
        DurableConversationErrorCode.MIGRATION_REQUIRED: (
            "durable conversation storage migration is required"
        ),
    }
)


class ConversationError(RuntimeError):
    """Report one stable error without retaining caller or provider content."""

    def __init__(
        self,
        code: ConversationErrorCode | DurableConversationErrorCode,
        *,
        boundary: FailureBoundary = FailureBoundary.VALIDATION_BEFORE_DISPATCH,
    ) -> None:
        assert isinstance(
            code,
            ConversationErrorCode | DurableConversationErrorCode,
        )
        assert isinstance(boundary, FailureBoundary)
        message = _ERROR_MESSAGES[code]
        self.code = code
        self.boundary = boundary
        self.safe_message = message
        super().__init__(message)

    def __repr__(self) -> str:
        """Return a content-safe diagnostic representation."""
        return (
            f"{type(self).__name__}(code={self.code.value!r}, "
            f"boundary={self.boundary.value!r})"
        )


class ConversationValidationError(ConversationError):
    """Report invalid conversation-domain input."""

    def __init__(self) -> None:
        super().__init__(ConversationErrorCode.VALIDATION_FAILED)


class ConversationCapabilityError(ConversationError):
    """Report an unsupported requested conversation capability."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.CAPABILITY_UNSUPPORTED,
        )


class ConversationBindingDriftError(ConversationError):
    """Report an incompatible provider-lane binding."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.BINDING_DRIFT,
        )


class ConversationConflictError(ConversationError):
    """Report a branch, revision, or idempotency conflict."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.CONFLICT,
        )


class ConversationIntegrityError(ConversationError):
    """Report failed checkpoint or envelope integrity validation."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.INTEGRITY_FAILED,
        )


class ConversationExpiredError(ConversationError):
    """Report expired conversation state."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.EXPIRED,
        )


class ConversationDeletedError(ConversationError):
    """Report deleted or tombstoned conversation state."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.DELETED,
        )


class ConversationStorageError(ConversationError):
    """Report a conversation storage failure."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.STORAGE_FAILED,
            boundary=FailureBoundary.CHECKPOINT_COMMIT,
        )


class ConversationAmbiguousDispatchError(ConversationError):
    """Report a dispatch whose provider-side effect is unknown."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.DISPATCH_AMBIGUOUS,
            boundary=FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH,
        )


class ConversationCommitError(ConversationError):
    """Report failure to commit an authoritative checkpoint."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.COMMIT_FAILED,
            boundary=FailureBoundary.CHECKPOINT_COMMIT,
        )


class ConversationPublicationError(ConversationError):
    """Report outward publication failure after commit."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.PUBLICATION_FAILED,
            boundary=FailureBoundary.OUTWARD_PUBLICATION,
        )


class ConversationAuthorizationError(ConversationError):
    """Conceal absent and unauthorized conversation state uniformly."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.AUTHORIZATION_FAILED,
        )


class ConversationLimitError(ConversationError):
    """Report a bounded conversation-domain limit."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.LIMIT_EXCEEDED,
        )


class ConversationCodecError(ConversationError):
    """Report malformed or unsupported encoded conversation state."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.CODEC_FAILED,
        )


class ConversationTransitionError(ConversationError):
    """Report an illegal immutable state transition."""

    def __init__(self) -> None:
        super().__init__(
            ConversationErrorCode.TRANSITION_INVALID,
        )


class ConversationKeyMissingError(ConversationError):
    """Report a missing durable conversation read or write key."""

    def __init__(self) -> None:
        super().__init__(DurableConversationErrorCode.KEY_MISSING)


class ConversationKeyRetiredError(ConversationError):
    """Report a durable conversation key outside its read grace period."""

    def __init__(self) -> None:
        super().__init__(DurableConversationErrorCode.KEY_RETIRED)


class ConversationKeyPolicyError(ConversationError):
    """Report a fail-closed durable conversation key policy."""

    def __init__(self) -> None:
        super().__init__(DurableConversationErrorCode.KEY_POLICY_INVALID)


class ConversationCryptoAuthenticationError(ConversationError):
    """Report modified ciphertext or associated data uniformly."""

    def __init__(self) -> None:
        super().__init__(
            DurableConversationErrorCode.CRYPTO_AUTHENTICATION_FAILED
        )


class ConversationFeatureUnavailableError(ConversationError):
    """Report unavailable optional durable conversation dependencies."""

    def __init__(self) -> None:
        super().__init__(DurableConversationErrorCode.FEATURE_UNAVAILABLE)


class ConversationMigrationRequiredError(ConversationError):
    """Report an absent or incompatible durable conversation migration."""

    def __init__(self) -> None:
        super().__init__(DurableConversationErrorCode.MIGRATION_REQUIRED)
