"""Prove hardening policy and effects remain strictly asynchronous."""

from typing import assert_type

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


async def prove_phase11_hardening(
    runner: ConversationEffectRunner,
    controller: FairConversationAdmissionController,
    admission: ConversationAdmissionKey,
    key_ring: AsyncConversationKeyRing,
    authority: AuthorityDigest,
) -> tuple[int, ConversationAdmissionLease, ConversationOperationalKey]:
    """Return exact bounded effect, admission, and key types."""

    async def value() -> int:
        return 1

    result = assert_type(await runner.provider(value()), int)
    lease = assert_type(
        await controller.acquire(admission), ConversationAdmissionLease
    )
    key = assert_type(
        await key_ring.resolve_active(
            authority, ConversationKeyPurpose.CHECKPOINT
        ),
        ConversationOperationalKey,
    )
    return result, lease, key
