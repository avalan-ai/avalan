"""Reject synchronous implementations of effectful mutation protocols."""

from typing import Protocol

from avalan.patch.domain import PatchResult


class PatchTerminalPublisher(Protocol):
    """Publish a terminal result asynchronously."""

    async def publish(self, result: PatchResult) -> None:
        """Publish one terminal result."""


class SyncPublisher:
    """Expose an intentionally invalid synchronous publisher boundary."""

    def publish(self, result: PatchResult) -> None:
        """Publish a terminal result without an async contract."""
        assert result


publisher: PatchTerminalPublisher = SyncPublisher()
