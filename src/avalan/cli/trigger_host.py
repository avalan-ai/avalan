"""Retain and reconstruct the CLI's verified native execution bindings."""

from ..agent.loader import OrchestratorLoader
from ..task.canonical import spec_hash
from ..task.definition import RunMode, TaskTargetType
from ..task.deployment import (
    DeploymentFileRoot,
    ExecutionDeployment,
    ExecutionDeploymentError,
    execution_deployment_from_payload,
)
from ..task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from ..task.deployment_closure import ExecutionClosure
from ..task.loader import TaskDefinitionLoader
from ..task.target import TaskValidationContext
from ..task.targets.agent import AgentTaskTargetRunner
from ..task.targets.file_flow import FileFlowTaskResolver
from ..task.targets.flow import FlowTaskTargetRunner
from ..task.validation import TaskValidationError

from errno import EEXIST, ENOTEMPTY
from importlib.metadata import version
from json import dumps, loads
from os import chmod
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp


class TriggerCliHost:
    """Use immutable local deployment files and the existing native engines.

    The directory is trusted operator storage and must survive restarts.
    A retained binding is never replaced or garbage-collected by this host.
    """

    def __init__(self, root: Path, loader: OrchestratorLoader) -> None:
        self.root = root.absolute()
        self.loader = loader
        self.runtime = Path(__file__).resolve().parents[1]
        self.bindings: dict[str, ExecutionDeploymentBinding] = {}
        self.catalog = ExecutionDeploymentCatalog(
            allowed_runtime_options=(
                "agent.disable_memory",
                "agent.require_shell_pipeline_opt_in",
                "flow.concurrency_limit",
            )
        )

    def _target(
        self, base: Path, kind: TaskTargetType
    ) -> AgentTaskTargetRunner | FlowTaskTargetRunner:
        agent = AgentTaskTargetRunner(
            self.loader, ref_base=base, require_shell_pipeline_opt_in=True
        )
        if kind is TaskTargetType.AGENT:
            return agent
        if kind is TaskTargetType.FLOW:
            return FlowTaskTargetRunner(
                ref_base=base,
                strict_resolver=FileFlowTaskResolver(base),
                agent_runner=agent,
            )
        raise ExecutionDeploymentError("cli.target")

    async def binding(
        self,
        root: Path,
        task_ref: str,
        *,
        manifest: ExecutionDeployment | None = None,
    ) -> ExecutionDeploymentBinding:
        root = root.resolve(strict=True)
        closure = ExecutionClosure(root, self.runtime)
        files = closure.collect(task_ref)
        task_path = root / task_ref
        definition = await TaskDefinitionLoader().load(task_path)
        if definition.run.mode is not RunMode.QUEUE:
            raise ExecutionDeploymentError("cli.queue_required")
        if definition.container.enabled:
            raise ExecutionDeploymentError("cli.container_host_required")
        if (
            definition.skills is not None
            or definition.skills_config is not None
        ):
            raise ExecutionDeploymentError("cli.skills_host_required")
        if closure.tools or any(
            filters is None or bool(filters)
            for filters in closure.tool_filters
        ):
            raise ExecutionDeploymentError("cli.tools_host_required")
        target = self._target(task_path.parent, definition.execution.type)
        issues = await target.validate_definition(
            definition, TaskValidationContext(execution_roots=(root,))
        )
        if issues:
            raise TaskValidationError(issues)
        if manifest is None:
            manifest = ExecutionDeployment(
                task_ref=task_ref,
                task_hash=await spec_hash(
                    definition, schema_base_path=task_path
                ),
                files=files,
                runtime_version=version("avalan"),
                runtime_options=target.execution_deployment_options(
                    task_path.parent
                ),
            )
        existing = self.bindings.get(manifest.execution_deployment_id)
        if existing is not None:
            if (
                existing.application_root != root
                or existing.manifest != manifest
            ):
                raise ExecutionDeploymentError("binding.replacement")
            return await self.catalog.resolve(manifest.execution_deployment_id)
        binding = ExecutionDeploymentBinding(
            manifest=manifest,
            application_root=root,
            runtime_root=self.runtime,
            definition=definition,
            target=target,
        )
        await self.catalog.register(binding)
        self.bindings[manifest.execution_deployment_id] = binding
        return binding

    async def retain(
        self, source_root: Path, task_ref: str
    ) -> ExecutionDeploymentBinding:
        """Copy verified files once, then bind to their immutable location."""
        # Use a separate transient catalog so the same identity can be bound
        # once to its final persisted root in this host's catalog.
        transient = TriggerCliHost(self.root, self.loader)
        source = await transient.binding(source_root, task_ref)
        identity = source.manifest.execution_deployment_id
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.resolve(strict=True) != self.root:
            raise ExecutionDeploymentError("cli.catalog_root")
        destination = self.root / identity
        if not destination.exists():
            staging = Path(mkdtemp(prefix=".preparing-", dir=self.root))
            try:
                application = staging / "application"
                application.mkdir(mode=0o700)
                for item in source.manifest.files:
                    if item.root is not DeploymentFileRoot.APPLICATION:
                        continue
                    data = ExecutionClosure(
                        source.application_root, self.runtime
                    ).read(item.root, item.path)
                    target = application / item.path
                    target.parent.mkdir(
                        mode=0o700, parents=True, exist_ok=True
                    )
                    target.write_bytes(data)
                    chmod(target, 0o400)
                record = staging / "manifest.json"
                record.write_text(
                    dumps(
                        source.manifest.payload(),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                chmod(record, 0o400)
                await TriggerCliHost(self.root, self.loader).binding(
                    application, task_ref, manifest=source.manifest
                )
                try:
                    staging.rename(destination)
                except OSError as error:
                    # A concurrent host may publish the same immutable ID.
                    # load() below verifies its complete retained binding.
                    if error.errno not in (EEXIST, ENOTEMPTY):
                        raise
            finally:
                if staging.exists():
                    rmtree(staging)
        return await self.load(identity)

    async def load(self, identity: str) -> ExecutionDeploymentBinding:
        from_identity = self.root / identity
        if len(identity) != 64 or any(
            c not in "0123456789abcdef" for c in identity
        ):
            raise ExecutionDeploymentError("cli.deployment_id")
        if from_identity.resolve(strict=True) != from_identity:
            raise ExecutionDeploymentError("cli.catalog_root")
        record = from_identity / "manifest.json"
        if record.is_symlink():
            raise ExecutionDeploymentError("cli.catalog_manifest")
        manifest = execution_deployment_from_payload(loads(record.read_text()))
        if manifest.execution_deployment_id != identity:
            raise ExecutionDeploymentError("cli.deployment_id")
        return await self.binding(
            from_identity / "application", manifest.task_ref, manifest=manifest
        )

    async def restore(self) -> ExecutionDeploymentCatalog:
        """Load retained manifests without depending on process callbacks."""
        if not self.root.exists():
            raise ExecutionDeploymentError("cli.catalog_missing")
        for directory in sorted(self.root.iterdir()):
            if directory.name.startswith(".preparing-"):
                continue
            await self.load(directory.name)
        return self.catalog
