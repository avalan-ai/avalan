"""Assert the narrow persistent-container patch contract is typed."""

from typing import assert_type

from avalan.patch.container_target import (
    ContainerInspectionTarget,
    ContainerPatchImage,
    ContainerPatchRuntime,
    ContainerPatchRuntimeSettings,
    ContainerPatchTarget,
    ContainerPersistentLeaseAuthority,
)
from avalan.patch.coordinator import RootedSandboxCommitWorker
from avalan.patch.target import (
    InspectionBatch,
    InspectionRequest,
    ResolvedMutationScope,
    TargetHandshake,
)


def assert_container_image_type() -> None:
    """Require a pinned image reference before runtime construction."""
    assert_type(ContainerPatchImage("sha256:" + "0" * 64), ContainerPatchImage)


def assert_container_lease_authority_type() -> None:
    """Require host-only persistent lease authority to stay typed."""
    assert_type(
        ContainerPersistentLeaseAuthority.from_bytes(b"0" * 32),
        ContainerPersistentLeaseAuthority,
    )


async def assert_container_runtime_types(
    settings: ContainerPatchRuntimeSettings,
    scope: ResolvedMutationScope,
    request: InspectionRequest,
) -> None:
    """Require service inspection and commit channels to remain async."""
    runtime = settings.create_runtime()
    assert_type(runtime, ContainerPatchRuntime)
    inspection = ContainerInspectionTarget(runtime)
    target = ContainerPatchTarget(runtime)
    assert_type(await inspection.inspect(request), InspectionBatch)
    assert_type(await target.handshake(scope), TargetHandshake)
    assert_type(await target.worker(scope), RootedSandboxCommitWorker)
