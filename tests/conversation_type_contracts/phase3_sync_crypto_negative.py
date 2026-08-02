"""Reject synchronous substitutes for every Phase 3 crypto effect."""

from avalan.conversation import (
    AuthorityDigest,
    ConversationCipher,
    ConversationDataKey,
    ConversationKeyResolver,
    ConversationPayloadAssociatedData,
    EncryptedConversationPayload,
)


class SyncKeyResolver:
    """Resolve durable keys synchronously."""

    def current_write_key(
        self,
        authority_digest: AuthorityDigest,
    ) -> ConversationDataKey:
        """Return a write key without an awaitable boundary."""
        raise NotImplementedError

    def read_key(
        self,
        authority_digest: AuthorityDigest,
        *,
        key_id: str,
        revision: int,
    ) -> ConversationDataKey:
        """Return a read key without an awaitable boundary."""
        raise NotImplementedError


class SyncCipher:
    """Encrypt and decrypt synchronously."""

    def encrypt(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> EncryptedConversationPayload:
        """Encrypt without an awaitable boundary."""
        raise NotImplementedError

    def decrypt(
        self,
        payload: EncryptedConversationPayload,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> bytes:
        """Decrypt without an awaitable boundary."""
        raise NotImplementedError

    def authenticated_digest(
        self,
        plaintext: bytes,
        *,
        key: ConversationDataKey,
        associated_data: ConversationPayloadAssociatedData,
    ) -> str:
        """Digest without an awaitable boundary."""
        raise NotImplementedError


key_resolver: ConversationKeyResolver = SyncKeyResolver()
cipher: ConversationCipher = SyncCipher()
