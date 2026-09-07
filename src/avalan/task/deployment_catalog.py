"""Retain and revalidate host execution bindings by immutable manifest ID."""

from ..tool.names import matches_tool_namespace
from .canonical import spec_hash
from .definition import TaskDefinition
from .deployment import (
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from .deployment_closure import ExecutionClosure
from .deployment_requirements import (
    deployment_container_images,
    deployment_skill_manifest_id,
)
from .deployment_target import ExecutionDeploymentTarget
from .loader import TaskDefinitionLoader
from .target import TaskTargetRunner

from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionDeploymentBinding:
    """Retain the host's runner and roots for one immutable deployment.

    The host supplies the runner already configured for these roots. A catalog
    never substitutes a runner or root from a newer registration. File sealing
    covers local definitions; external providers and tools remain host-owned.
    """

    manifest: ExecutionDeployment
    application_root: Path
    runtime_root: Path
    target: TaskTargetRunner
    definition: TaskDefinition

    async def verify(self, *, file_delivery: bool = False) -> None:
        """Read the entire sealed closure before any execution dispatch."""
        if (
            self.runtime_root.resolve(strict=True)
            != Path(__file__).resolve().parents[1]
        ):
            raise ExecutionDeploymentError("runtime.root")
        if not isinstance(self.target, ExecutionDeploymentTarget):
            raise ExecutionDeploymentError("target.binding_capability")
        options = self.target.execution_deployment_options(
            (self.application_root / self.manifest.task_ref).parent.resolve(
                strict=True
            )
        )
        if options != self.manifest.runtime_options:
            raise ExecutionDeploymentError("runtime_options")
        closure = ExecutionClosure(self.application_root, self.runtime_root)
        required = closure.collect(
            self.manifest.task_ref, file_delivery=file_delivery
        )
        declared = {
            (file.root, file.path): file for file in self.manifest.files
        }
        if any(
            declared.get((file.root, file.path)) != file for file in required
        ):
            raise ExecutionDeploymentError("closure")
        for file in self.manifest.files:
            closure.read(file.root, file.path)
            if closure.files[(file.root, file.path)] != file:
                raise ExecutionDeploymentError("file.sha256")
        if self.manifest.runtime_version != version("avalan"):
            raise ExecutionDeploymentError("runtime_version")
        if any(filters is None for filters in closure.tool_filters):
            raise ExecutionDeploymentError("agent.tool_allowlist")
        if not closure.tools.issubset(self.manifest.required_tools):
            raise ExecutionDeploymentError("required_tools")
        # Reloading the sealed task also prevents a persisted definition from
        # being paired with unrelated host files under a matching opaque ID.
        if (
            deployment_container_images(self.definition)
            != self.manifest.container_image_digests
        ):
            raise ExecutionDeploymentError("container_image_digests")
        if (
            await deployment_skill_manifest_id(
                self.definition,
                definition_base=self.application_root / self.manifest.task_ref,
            )
            != self.manifest.skill_manifest_id
        ):
            raise ExecutionDeploymentError("skill_manifest_id")
        loaded = await TaskDefinitionLoader(
            skills_settings=self.definition.skills
        ).load(self.application_root / self.manifest.task_ref)
        # Container settings are a trusted host selection, not a TOML-loaded
        # authority. Their complete canonical policy remains in task_hash.
        loaded = replace(loaded, container=self.definition.container)
        if (
            await spec_hash(
                loaded,
                schema_base_path=self.application_root
                / self.manifest.task_ref,
            )
            != self.manifest.task_hash
            or await spec_hash(
                self.definition,
                schema_base_path=self.application_root
                / self.manifest.task_ref,
            )
            != self.manifest.task_hash
        ):
            raise ExecutionDeploymentError("task_hash")


class ExecutionDeploymentCatalog:
    """Keep admitted deployments available after a host changes its default."""

    def __init__(
        self,
        *,
        available_tools: tuple[str, ...] = (),
        allowed_runtime_options: tuple[str, ...] = (),
        container_invocation_images: tuple[str, ...] = (),
    ) -> None:
        assert available_tools == tuple(sorted(set(available_tools)))
        assert all(
            isinstance(tool, str) and tool.strip() for tool in available_tools
        )
        assert allowed_runtime_options == tuple(
            sorted(set(allowed_runtime_options))
        )
        assert all(
            isinstance(name, str) and name.strip()
            for name in allowed_runtime_options
        )
        assert container_invocation_images == tuple(
            sorted(set(container_invocation_images))
        )
        self._container_invocation_images = container_invocation_images
        self._available_tools = available_tools
        self._allowed_runtime_options = allowed_runtime_options
        self._bindings: dict[str, ExecutionDeploymentBinding] = {}

    async def register(self, binding: ExecutionDeploymentBinding) -> str:
        """Validate a binding before retaining its immutable identity."""
        assert isinstance(binding, ExecutionDeploymentBinding)
        manifest = binding.manifest
        if not set(manifest.container_image_digests).issubset(
            self._container_invocation_images
        ):
            raise ExecutionDeploymentError("container.command_protocol")
        if not set(manifest.required_tools).issubset(self._available_tools):
            raise ExecutionDeploymentError("required_tools")
        if not {option.name for option in manifest.runtime_options}.issubset(
            self._allowed_runtime_options
        ):
            raise ExecutionDeploymentError("runtime_options")
        await binding.verify()
        closure = ExecutionClosure(
            binding.application_root, binding.runtime_root
        )
        closure.collect(manifest.task_ref)
        requested = set(closure.tools)
        for filters in closure.tool_filters:
            assert filters is not None
            for pattern in filters:
                matched = {
                    name
                    for name in self._available_tools
                    if matches_tool_namespace(name, pattern)
                }
                if not matched:
                    raise ExecutionDeploymentError("required_tools")
                requested.update(matched)
        if tuple(sorted(requested)) != manifest.required_tools:
            raise ExecutionDeploymentError("required_tools")
        identity = manifest.execution_deployment_id
        previous = self._bindings.get(identity)
        if previous is not None and previous is not binding:
            # Replacing a live binding under the same ID could redirect an old
            # admitted run even when its nominal file manifest is unchanged.
            raise ExecutionDeploymentError("binding.replacement")
        self._bindings[identity] = binding
        return identity

    async def resolve(self, identity: str) -> ExecutionDeploymentBinding:
        """Verify the retained binding against its immutable roots."""
        binding = self._bindings.get(identity)
        if binding is None:
            raise ExecutionDeploymentError("binding.unavailable")
        await binding.verify()
        return binding
