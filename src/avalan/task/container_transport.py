"""Mount the admitted command protocol for the existing container lifecycle."""

from ..container import (
    ContainerAsyncBackend,
    ContainerManagedLifecycleResult,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerOutputContract,
    ContainerRunPlan,
    run_container_managed_lifecycle,
)
from .container_invocation import (
    CONTAINER_DEPLOYMENT_ROOT,
    CONTAINER_INVOCATION_PATH,
    container_invocation_bytes,
)
from .deployment import ExecutionDeploymentError
from .deployment_catalog import ExecutionDeploymentBinding
from .store import TaskExecutionContext

from dataclasses import dataclass, replace
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp


@dataclass(frozen=True, slots=True)
class ContainerInvocationStaging:
    """Retain the exact invocation directory until cleanup is confirmed."""

    directory: Path
    context: TaskExecutionContext


class ContainerInvocationOwnership:
    """Retain unsettled mounts until cleanup is confirmed."""

    def __init__(self) -> None:
        self._pending: dict[Path, ContainerInvocationStaging] = {}

    @property
    def pending(self) -> tuple[ContainerInvocationStaging, ...]:
        return tuple(self._pending.values())

    def _acquire(
        self, context: TaskExecutionContext
    ) -> ContainerInvocationStaging:
        staging = ContainerInvocationStaging(
            Path(mkdtemp(prefix="avalan-task-invocation-")), context
        )
        self._pending[staging.directory] = staging
        return staging

    def _settle(self, staging: ContainerInvocationStaging) -> None:
        assert self._pending[staging.directory] is staging
        rmtree(staging.directory)
        del self._pending[staging.directory]


class ContainerInvocationUnsettled(ExecutionDeploymentError):
    """Expose a retained mount after the backend cannot prove cleanup."""

    def __init__(
        self,
        staging: ContainerInvocationStaging,
        result: ContainerManagedLifecycleResult,
    ) -> None:
        self.staging = staging
        self.result = result
        super().__init__("container.cleanup_unknown")


async def run_deployment_container(
    backend: ContainerAsyncBackend,
    plan: ContainerRunPlan,
    *,
    context: TaskExecutionContext,
    input_value: object = None,
    file_delivery: bool = False,
    binding: ExecutionDeploymentBinding | None,
    output_contract: ContainerOutputContract,
    shutdown_requested: bool,
    ownership: ContainerInvocationOwnership,
) -> ContainerManagedLifecycleResult:
    """Retain read-only invocation mounts until lifecycle shutdown settles."""
    if context.trigger is None:
        return await run_container_managed_lifecycle(
            backend,
            plan,
            output_contract=output_contract,
            shutdown_requested=shutdown_requested,
        )
    if binding is None or context.deployment != binding.manifest:
        raise ExecutionDeploymentError("container.binding")
    await binding.verify()
    image = plan.image.reference.split("@", 1)[0]
    if image.rfind(":") > image.rfind("/"):
        image = image.rsplit(":", 1)[0]
    if (
        image + "@" + str(plan.image.digest)
        not in binding.manifest.container_image_digests
    ):
        raise ExecutionDeploymentError("container.image")
    deployment_mount = ContainerMountDeclaration(
        target=CONTAINER_DEPLOYMENT_ROOT,
        mount_type=ContainerMountType.WORKSPACE,
        access=ContainerMountAccess.READ,
        source=str(binding.application_root.resolve(strict=True)),
    )
    mounts = list(plan.mounts)
    for mount in mounts:
        if (
            mount.target == CONTAINER_DEPLOYMENT_ROOT
            and mount != deployment_mount
        ):
            raise ExecutionDeploymentError("container.deployment_mount")
        if mount.target == CONTAINER_INVOCATION_PATH:
            raise ExecutionDeploymentError("container.invocation_mount")
    if deployment_mount not in mounts:
        mounts.append(deployment_mount)
    staging = ownership._acquire(context)
    invocation = staging.directory / "invocation.json"
    try:
        invocation.write_bytes(
            container_invocation_bytes(
                context, input_value=input_value, file_delivery=file_delivery
            )
        )
        invocation.chmod(0o400)
        mounts.append(
            ContainerMountDeclaration(
                target=CONTAINER_INVOCATION_PATH,
                mount_type=ContainerMountType.INPUT,
                access=ContainerMountAccess.READ,
                source=str(invocation),
            )
        )
    except BaseException:
        # No external operation has started, so this allocation is settled.
        ownership._settle(staging)
        raise
    result = await run_container_managed_lifecycle(
        backend,
        replace(plan, mounts=tuple(mounts)),
        output_contract=output_contract,
        shutdown_requested=shutdown_requested,
    )
    if (
        not result.cleanup_completed
        or result.cleanup_uncertain
        or result.orphan_quarantined
    ):
        raise ContainerInvocationUnsettled(staging, result)
    ownership._settle(staging)
    return result
