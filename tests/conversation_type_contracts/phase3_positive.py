"""Prove Phase 3 durable storage and cryptography satisfy public protocols."""

from avalan.conversation import (
    AesGcmConversationCipher,
    ConversationCipher,
    ConversationKeyResolver,
    ConversationStore,
    InMemoryConversationKeyResolver,
    PgsqlConversationStore,
)


def prove_phase3_protocols(
    store: PgsqlConversationStore,
    key_resolver: InMemoryConversationKeyResolver,
    cipher: AesGcmConversationCipher,
) -> tuple[
    ConversationStore,
    ConversationKeyResolver,
    ConversationCipher,
]:
    """Return Phase 3 implementations through their public protocols."""
    store_protocol: ConversationStore = store
    key_resolver_protocol: ConversationKeyResolver = key_resolver
    cipher_protocol: ConversationCipher = cipher
    return store_protocol, key_resolver_protocol, cipher_protocol
