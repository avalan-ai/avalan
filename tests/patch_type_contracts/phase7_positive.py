"""Assert the private local commit channel remains typed and asynchronous."""

from typing import assert_type

from avalan.patch.coordinator import (
    CommitWorker,
    RootedLocalCommitWorker,
    SealedCommitCommand,
    WorkerReport,
)
from avalan.patch.local_commit import LocalCommitTarget
from avalan.patch.target import ResolvedMutationScope, TargetHandshake


async def assert_local_commit_types(
    target: LocalCommitTarget,
    scope: ResolvedMutationScope,
    command: SealedCommitCommand,
) -> None:
    """Assert the rooted worker cannot become a synchronous target protocol."""
    assert_type(await target.handshake(scope), TargetHandshake)
    worker = await target.worker(scope)
    assert_type(worker, RootedLocalCommitWorker)
    await assert_commit_worker(worker, command)
    assert_type(await worker.commit(command), WorkerReport)


async def assert_commit_worker(
    worker: CommitWorker, command: SealedCommitCommand
) -> None:
    """Require the local worker to satisfy the shared async protocol."""
    assert_type(await worker.commit(command), WorkerReport)
