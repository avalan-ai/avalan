"""Assert the narrow sandbox mutation channel remains async and typed."""

from typing import assert_type

from avalan.patch.coordinator import (
    CommitWorker,
    RootedSandboxCommitWorker,
    SealedCommitCommand,
    WorkerReport,
)
from avalan.patch.domain import ContextKind
from avalan.patch.sandbox_commit import (
    SandboxCommitTarget,
    SandboxInspectionTarget,
    SandboxPatchRuntime,
    SandboxPatchRuntimeBinder,
    SandboxPatchRuntimeSettings,
    SandboxScopeResolver,
)
from avalan.patch.target import (
    InspectionBatch,
    InspectionRequest,
    ResolvedMutationScope,
    ScopeSelection,
    TargetHandshake,
)
from avalan.patch.toolset import PatchRuntimeBinding


async def assert_sandbox_mutation_types(
    runtime: SandboxPatchRuntime,
    resolver: SandboxScopeResolver,
    target: SandboxCommitTarget,
    inspection: SandboxInspectionTarget,
    scope: ResolvedMutationScope,
    request: InspectionRequest,
    command: SealedCommitCommand,
) -> None:
    """Assert the sandbox target exposes no synchronous commit escape hatch."""
    assert_type(runtime, SandboxPatchRuntime)
    assert_type(
        await resolver.resolve(scope_selection()), ResolvedMutationScope
    )
    assert_type(await target.handshake(scope), TargetHandshake)
    worker = await target.worker(scope)
    assert_type(worker, RootedSandboxCommitWorker)
    assert_type(await worker.commit(command), WorkerReport)
    assert_type(await inspection.inspect(request), InspectionBatch)
    await assert_commit_worker(worker, command)


def scope_selection() -> ScopeSelection:
    """Supply the sandbox-only selection used by the runtime resolver."""
    return ScopeSelection(ContextKind.SANDBOX)


async def assert_commit_worker(
    worker: CommitWorker, command: SealedCommitCommand
) -> None:
    """Require the sandbox worker to satisfy the shared async protocol."""
    assert_type(await worker.commit(command), WorkerReport)


async def assert_sandbox_binder_types(
    settings: SandboxPatchRuntimeSettings,
    binder: SandboxPatchRuntimeBinder,
) -> None:
    """Require the production binder to retain selected runtime types."""
    assert_type(settings.create_runtime(), SandboxPatchRuntime)
    assert_type(await binder.bind(), PatchRuntimeBinding)
