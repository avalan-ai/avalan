"""Reject synchronous substitutes for every new Phase 2 effect protocol."""

from datetime import datetime

from avalan.conversation import (
    AuthorityScope,
    ConversationAuthorityResolver,
    ConversationCheckpoint,
    ConversationClock,
    ConversationOutbox,
    ConversationPublisher,
    ConversationRetryWaiter,
    ConversationUnitOfWork,
    OutboxClaimResolution,
    OutboxClaimTarget,
    PublicationIntent,
)


class SyncAuthorityResolver:
    """Resolve authority synchronously."""

    def resolve(self) -> AuthorityScope:
        """Return authority without an awaitable boundary."""
        raise NotImplementedError


class SyncClock:
    """Read time synchronously."""

    def now(self) -> datetime:
        """Return time without an awaitable boundary."""
        raise NotImplementedError


class SyncRetryWaiter:
    """Wait synchronously."""

    def wait(self, attempt: int) -> None:
        """Return without an awaitable boundary."""
        raise NotImplementedError


class SyncPublisher:
    """Publish synchronously."""

    def publish(self, intent: PublicationIntent) -> None:
        """Return without an awaitable boundary."""
        raise NotImplementedError


class SyncOutbox:
    """Claim and settle synchronously."""

    def claim(self, target: OutboxClaimTarget) -> OutboxClaimResolution:
        """Return a closed claim without an awaitable boundary."""
        raise NotImplementedError

    def acknowledge(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Acknowledge without an awaitable boundary."""
        raise NotImplementedError

    def release(
        self,
        target: OutboxClaimTarget,
        owner_token: str,
    ) -> None:
        """Release without an awaitable boundary."""
        raise NotImplementedError


class SyncUnitOfWork:
    """Commit and roll back synchronously."""

    def __aenter__(self) -> "SyncUnitOfWork":
        """Enter without an awaitable boundary."""
        return self

    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit without an awaitable boundary."""
        raise NotImplementedError

    def commit(self) -> ConversationCheckpoint:
        """Commit without an awaitable boundary."""
        raise NotImplementedError

    def rollback(self) -> None:
        """Roll back without an awaitable boundary."""
        raise NotImplementedError


authority: ConversationAuthorityResolver = SyncAuthorityResolver()
clock: ConversationClock = SyncClock()
waiter: ConversationRetryWaiter = SyncRetryWaiter()
publisher: ConversationPublisher = SyncPublisher()
outbox: ConversationOutbox = SyncOutbox()
unit_of_work: ConversationUnitOfWork = SyncUnitOfWork()
