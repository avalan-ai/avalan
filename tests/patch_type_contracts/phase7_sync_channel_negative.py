"""Reject a synchronous rooted local commit channel implementation."""

from avalan.patch.coordinator import (
    RootedCommitChannel,
    SealedCommitCommand,
    WorkerReport,
)


class SynchronousChannel:
    """Deliberately violate the asynchronous rooted channel protocol."""

    def commit_local(self, command: SealedCommitCommand) -> WorkerReport:
        """Return synchronously for the negative type fixture."""
        raise RuntimeError(command)


channel: RootedCommitChannel = SynchronousChannel()
