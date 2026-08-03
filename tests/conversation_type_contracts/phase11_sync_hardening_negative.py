"""Reject synchronous hardening effects, admission, and key resolution."""

from avalan.conversation.security import (
    AsyncConversationKeyRing,
    ConversationAdmissionKey,
    ConversationAdmissionLease,
    ConversationEffectRunner,
    ConversationKeyPurpose,
    ConversationOperationalKey,
    FairConversationAdmissionController,
)
from avalan.conversation.value import AuthorityDigest


async def value() -> int:
    """Return one typed asynchronous value."""
    return 1


def reject_sync_effect(runner: ConversationEffectRunner) -> int:
    """Reject a provider effect whose coroutine is not awaited."""
    return runner.provider(value())


def reject_sync_admission(
    controller: FairConversationAdmissionController,
    admission: ConversationAdmissionKey,
) -> ConversationAdmissionLease:
    """Reject admission whose coroutine is not awaited."""
    return controller.acquire(admission)


def reject_sync_key(
    key_ring: AsyncConversationKeyRing,
    authority: AuthorityDigest,
) -> ConversationOperationalKey:
    """Reject key resolution whose coroutine is not awaited."""
    return key_ring.resolve_active(
        authority, ConversationKeyPurpose.CHECKPOINT
    )
