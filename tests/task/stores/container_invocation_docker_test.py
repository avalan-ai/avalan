"""Exercise the fixed task command and mounted verifier in actual Docker."""

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from json import loads
from os import environ
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from container_worker_deployment_helpers import (
    FixtureCommandDockerBackend,
    exercise_scheduled_container,
)
from task_deployment_helpers import DeploymentProviderLoader

from avalan.container import (
    ContainerBackend,
    ContainerCommandPlan,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputPolicy,
    ContainerProfile,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    ContainerWorkspaceMapping,
    DockerContainerBackend,
    DockerSubprocessRunner,
)
from avalan.task.canonical import spec_hash
from avalan.task.container_transport import (
    ContainerInvocationOwnership,
    ContainerInvocationUnsettled,
    run_deployment_container,
)
from avalan.task.definition import TaskContainerExecutionSettings
from avalan.task.deployment import (
    DeploymentFileRoot,
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from avalan.task.deployment_catalog import (
    ExecutionDeploymentBinding,
    ExecutionDeploymentCatalog,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.provenance import TriggerInvocationContext
from avalan.task.store import TaskExecutionContext
from avalan.task.targets.agent import AgentTaskTargetRunner

IMAGE = (
    "python@sha256:"
    "2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91"
)


class ContainerInvocationDockerTest(IsolatedAsyncioTestCase):
    async def test_actual_command_verifies_mount_before_output_dispatch(
        self,
    ) -> None:
        await self.exercise(queued=False)

    async def test_scheduled_worker_transports_frozen_input_and_output(
        self,
    ) -> None:
        await self.exercise(queued=True)

    async def test_queued_worker_rejects_unapproved_rootful_backend(
        self,
    ) -> None:
        await self.exercise(queued=True, rootful_authorized=False)

    async def test_queued_worker_rejects_metadata_only_output(self) -> None:
        await self.exercise(queued=True, copied_bytes=False)

    async def exercise(
        self,
        *,
        queued: bool,
        rootful_authorized: bool = True,
        copied_bytes: bool = True,
    ) -> None:
        dependency_path = environ.get("AVALAN_TRIGGER_CONTAINER_TEST_DEPS")
        if not dependency_path:
            self.skipTest(
                "set AVALAN_TRIGGER_CONTAINER_TEST_DEPS for the owned Docker"
                " protocol fixture"
            )
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = Path(__file__).resolve().parents[3] / "src"
            (root / "task.toml").write_text(
                '[task]\nname="container"\nversion="1"\n[input]\ntype="string"\n[output]\ntype="text"\n[execution]\ntype="agent"\nref="agent.toml"'
            )
            if queued:
                task = root / "task.toml"
                task.write_text(
                    task.read_text().replace('type="text"', 'type="file"')
                    + '\n[run]\nmode="queue"\nqueue="container"\n'
                    '[privacy]\ninput="encrypt"\nfiles="drop"\n'
                    "raw_retention_days=1\n"
                )
            (root / "agent.toml").write_text(
                '[agent]\nname="container"\ninstructions="retained"\n[engine]\nuri="ai://env:KEY@openai/gpt-4o-mini"\n[tool]\nenable=[]'
            )
            command = root / "avalan-task"
            copyfile(
                Path(__file__).parent.parent
                / "container_invocation_command.py",
                command,
            )
            command.chmod(0o555)
            profile = ContainerProfile.minimal_readonly(
                name="protocol", image_reference=IMAGE
            )
            if queued:
                profile = replace(
                    profile,
                    image=replace(
                        profile.image,
                        platform="linux/arm64",
                        pull_policy=ContainerPullPolicy.NEVER,
                    ),
                    mounts=(
                        ContainerMountDeclaration(
                            source=str(root),
                            target="/workspace",
                            mount_type=ContainerMountType.WORKSPACE,
                        ),
                    ),
                    workspace=ContainerWorkspaceMapping(host_root=str(root)),
                    output=ContainerOutputPolicy(
                        allow_artifacts=True, max_artifact_bytes=4096
                    ),
                )
            settings = ContainerEffectiveSettings(
                backend=ContainerBackend.DOCKER,
                required=True,
                scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                source=ContainerSettingsSource(
                    surface=ContainerSurface.SDK,
                    trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
                ),
                policy_version="protocol-v1",
                profile_registry_id="protocol",
                profile_name="protocol",
                profile=profile,
                allowed_profiles=("protocol",),
            )
            definition = replace(
                await TaskDefinitionLoader().load(root / "task.toml"),
                container=TaskContainerExecutionSettings(attempt=settings),
            )
            target = AgentTaskTargetRunner(
                DeploymentProviderLoader(), ref_base=root
            )
            closure = ExecutionClosure(root, source / "avalan")
            closure.collect("task.toml")
            closure.read(DeploymentFileRoot.APPLICATION, "avalan-task")
            manifest = ExecutionDeployment(
                task_ref="task.toml",
                task_hash=await spec_hash(definition, schema_base_path=root),
                files=tuple(
                    closure.files[key] for key in sorted(closure.files)
                ),
                runtime_version=version("avalan"),
                runtime_options=target.execution_deployment_options(root),
                container_image_digests=(IMAGE,),
            )
            binding = ExecutionDeploymentBinding(
                manifest=manifest,
                application_root=root,
                runtime_root=source / "avalan",
                target=target,
                definition=definition,
            )
            options = tuple(option.name for option in manifest.runtime_options)
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.command_protocol"
            ):
                await ExecutionDeploymentCatalog(
                    allowed_runtime_options=options
                ).register(binding)
            catalog = ExecutionDeploymentCatalog(
                allowed_runtime_options=options,
                container_invocation_images=(IMAGE,),
            )
            await catalog.register(binding)
            now = datetime(2026, 9, 7, tzinfo=UTC)
            context = TaskExecutionContext(
                run_id="container-run",
                attempt_id="container-attempt",
                attempt_number=1,
                deployment=manifest,
                trigger=TriggerInvocationContext(
                    trigger_id="trigger",
                    trigger_revision=2,
                    occurrence_id="slot",
                    scheduled_at=now,
                    dispatched_at=now,
                ),
            )
            mounts = tuple(
                ContainerMountDeclaration(
                    target=destination,
                    source=str(path),
                    mount_type=ContainerMountType.INPUT,
                )
                for path, destination in (
                    (command, "/usr/local/bin/avalan-task"),
                    (
                        source,
                        "/runtime",
                    ),
                    (
                        Path(dependency_path),
                        "/deps",
                    ),
                )
            )
            output_directory = root / "outputs"
            output_directory.mkdir()
            mounts = (
                *mounts,
                ContainerMountDeclaration(
                    target="/outputs",
                    source=str(output_directory),
                    mount_type=ContainerMountType.OUTPUT,
                    access=ContainerMountAccess.WRITE,
                ),
            )
            if queued:
                await exercise_scheduled_container(
                    self,
                    binding,
                    catalog,
                    mounts,
                    output_directory,
                    rootful_authorized=rootful_authorized,
                    copied_bytes=copied_bytes,
                )
                return
            plan = ContainerRunPlan(
                backend=ContainerBackend.DOCKER,
                profile_name="protocol",
                image=ContainerImagePolicy(
                    reference=IMAGE,
                    pull_policy=ContainerPullPolicy.NEVER,
                    platform="linux/arm64",
                ),
                command=ContainerCommandPlan(
                    tool_name="avalan-task",
                    command="avalan-task",
                    argv=("avalan-task", "agent", "agent.toml"),
                    cwd="/workspace",
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                ),
                mounts=mounts,
            )
            ownership = ContainerInvocationOwnership()
            result = await run_deployment_container(
                DockerContainerBackend(),
                plan,
                context=context,
                binding=binding,
                output_contract=ContainerOutputContract(
                    contract_type=ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=4096,
                ),
                shutdown_requested=False,
                ownership=ownership,
            )
            self.assertEqual(
                result.execution.status,
                ContainerResultStatus.COMPLETED,
                result,
            )
            assert result.output is not None
            self.assertEqual(len(result.output.artifacts), 1)
            content = (output_directory / "receipt.json").read_bytes()
            self.assertEqual(
                "sha256:" + sha256(content).hexdigest(),
                result.output.artifacts[0].digest,
            )
            receipt = loads(content)
            self.assertEqual(receipt["occurrence_id"], "slot")
            self.assertEqual(
                receipt["deployment_id"], manifest.execution_deployment_id
            )
            self.assertTrue(receipt["dispatched"])

            (output_directory / "receipt.json").unlink()
            tampered = root / "tampered.toml"
            tampered.write_text(
                '[agent]\ninstructions="replacement must never dispatch"'
            )
            shadow = ContainerMountDeclaration(
                target="/workspace/agent.toml",
                source=str(tampered),
                mount_type=ContainerMountType.INPUT,
            )
            rejected_backend = FixtureCommandDockerBackend(
                (), output_directory
            )
            try:
                rejected = await run_deployment_container(
                    rejected_backend,
                    replace(plan, mounts=(*mounts, shadow)),
                    context=context,
                    binding=binding,
                    output_contract=ContainerOutputContract(
                        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=4096,
                    ),
                    shutdown_requested=False,
                    ownership=ownership,
                )
            except ContainerInvocationUnsettled as error:
                rejected = error.result
                self.assertTrue(rejected.cleanup_uncertain)
                self.assertEqual(ownership.pending, (error.staging,))
                self.assertTrue(
                    (error.staging.directory / "invocation.json").is_file()
                )
            self.assertEqual(
                rejected.execution.status, ContainerResultStatus.FAILED
            )
            self.assertFalse((output_directory / "receipt.json").exists())
            if rejected.output is not None:
                self.assertEqual(rejected.output.artifacts, ())
            assert rejected.stream is not None
            self.assertIn(
                b"task.deployment_mismatch: container.closure",
                b"".join(chunk.content for chunk in rejected.stream.chunks),
            )

            if ownership.pending:
                self.assertEqual(len(rejected_backend.created), 1)
                absent = await DockerSubprocessRunner().run(
                    ("inspect", rejected_backend.created[0].container_id)
                )
                self.assertNotEqual(absent.return_code, 0)
                self.assertIn(b"no such object", absent.stderr.lower())
                # The exact create/remove calls have finished, and independent
                # daemon inspection now proves this test container is absent.
                ownership._settle(ownership.pending[0])
            self.assertEqual(ownership.pending, ())
