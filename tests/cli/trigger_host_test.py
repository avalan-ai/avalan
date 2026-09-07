from contextlib import AsyncExitStack
from dataclasses import replace
from errno import EACCES, EEXIST, ENOTEMPTY
from logging import getLogger
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, PropertyMock, patch
from uuid import uuid4

from trigger_cli_helpers import write_application

from avalan.agent.loader import OrchestratorLoader
from avalan.cli.trigger_host import TriggerCliHost
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.skill import TrustedSkillSettings
from avalan.task.definition import (
    TaskContainerExecutionSettings,
    TaskTargetType,
)
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.loader import TaskDefinitionLoader
from avalan.task.validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)


class TriggerCliHostTest(IsolatedAsyncioTestCase):
    async def test_retained_catalog_reconstructs_old_nested_deployment(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            write_application(source / "nested")
            async with AsyncExitStack() as stack:
                hub = HuggingfaceHub(
                    "unused", str(root / "cache"), getLogger(__name__)
                )
                loader = OrchestratorLoader(
                    hub=hub,
                    logger=getLogger(__name__),
                    participant_id=uuid4(),
                    stack=stack,
                )
                host = TriggerCliHost(root / "catalog", loader)
                old = await host.retain(source, "nested/task.toml")
                (source / "nested/agent.toml").write_text(
                    (source / "nested/agent.toml")
                    .read_text()
                    .replace("old retained", "replacement")
                )
                replacement = await host.retain(source, "nested/task.toml")
                self.assertNotEqual(
                    old.manifest.execution_deployment_id,
                    replacement.manifest.execution_deployment_id,
                )
                restarted = TriggerCliHost(root / "catalog", loader)
                catalog = await restarted.restore()
                restored = await catalog.resolve(
                    old.manifest.execution_deployment_id
                )
                self.assertIn(
                    "old retained",
                    (
                        restored.application_root / "nested/agent.toml"
                    ).read_text(),
                )
                self.assertNotEqual(restored.application_root, source)
                self.assertEqual(
                    (await host.retain(source, "nested/task.toml")).manifest,
                    replacement.manifest,
                )
                agent = restored.application_root / "nested/agent.toml"
                agent.chmod(0o600)
                agent.write_text("tampered")
                with self.assertRaises(ExecutionDeploymentError):
                    await catalog.resolve(old.manifest.execution_deployment_id)

    async def test_host_rejects_unsupported_settings_before_retaining(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            write_application(source)
            async with AsyncExitStack() as stack:
                loader = OrchestratorLoader(
                    hub=HuggingfaceHub(
                        "unused", str(root / "cache"), getLogger(__name__)
                    ),
                    logger=getLogger(__name__),
                    participant_id=uuid4(),
                    stack=stack,
                )
                host = TriggerCliHost(root / "catalog", loader)
                task = source / "task.toml"
                original = task.read_text()
                task.write_text(
                    original.replace('mode="queue"', 'mode="direct"')
                )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "queue_required"
                ):
                    await host.retain(source, "task.toml")
                task.write_text(original)
                definition = await TaskDefinitionLoader().load(task)
                with patch(
                    "avalan.cli.trigger_host.TaskDefinitionLoader.load",
                    AsyncMock(
                        return_value=replace(
                            definition, skills=TrustedSkillSettings()
                        )
                    ),
                ):
                    with self.assertRaisesRegex(
                        ExecutionDeploymentError, "skills_host"
                    ):
                        await host.retain(source, "task.toml")
                with patch.object(
                    TaskContainerExecutionSettings,
                    "enabled",
                    PropertyMock(return_value=True),
                ):
                    with self.assertRaisesRegex(
                        ExecutionDeploymentError, "container_host"
                    ):
                        await host.retain(source, "task.toml")
                agent = source / "agent.toml"
                agent.write_text(
                    agent.read_text().replace("enable=[]", 'enable="search.*"')
                )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "tools_host"
                ):
                    await host.retain(source, "task.toml")
                agent.write_text(
                    agent.read_text().replace('enable="search.*"', "enable=[]")
                )
                issue = TaskValidationIssue(
                    category=TaskValidationCategory.VALUE,
                    code="invalid",
                    path="target",
                    message="private",
                    hint="private",
                )
                with patch(
                    "avalan.cli.trigger_host.AgentTaskTargetRunner.validate_definition",
                    AsyncMock(return_value=(issue,)),
                ):
                    with self.assertRaises(TaskValidationError):
                        await host.retain(source, "task.toml")
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "cli.target"
                ):
                    host._target(source, TaskTargetType.CALLABLE)
                self.assertFalse(host.root.exists())
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "catalog_missing"
                ):
                    await host.restore()

    async def test_catalog_paths_identity_and_competing_publication(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source"
            write_application(source)
            async with AsyncExitStack() as stack:
                loader = OrchestratorLoader(
                    hub=HuggingfaceHub(
                        "unused", str(root / "cache"), getLogger(__name__)
                    ),
                    logger=getLogger(__name__),
                    participant_id=uuid4(),
                    stack=stack,
                )
                for number, error_number in enumerate(
                    (EEXIST, ENOTEMPTY, EACCES)
                ):
                    host = TriggerCliHost(root / str(number), loader)

                    def competing_publish(
                        path: Path, destination: Path
                    ) -> None:
                        if error_number != EACCES:
                            copytree(path, destination)
                        raise OSError(error_number, "publication race")

                    with patch.object(Path, "rename", competing_publish):
                        if error_number == EACCES:
                            with self.assertRaises(PermissionError):
                                await host.retain(source, "task.toml")
                        else:
                            retained = await host.retain(source, "task.toml")
                            self.assertEqual(
                                retained.manifest.task_ref, "task.toml"
                            )
                    self.assertFalse(tuple(host.root.glob(".preparing-*")))
                host = TriggerCliHost(root / "catalog", loader)
                bound = await host.binding(source, "task.toml")
                copied = root / "copied"
                copytree(source, copied)
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "binding.replacement"
                ):
                    await host.binding(copied, "task.toml")
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "deployment_id"
                ):
                    await host.load("../private")
                retained_host = TriggerCliHost(root / "retained", loader)
                retained = await retained_host.retain(source, "task.toml")
                identity = retained.manifest.execution_deployment_id
                (retained_host.root / ".preparing-owned").mkdir()
                await retained_host.restore()
                aliased = root / "aliased"
                aliased.symlink_to(
                    retained_host.root, target_is_directory=True
                )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "catalog_root"
                ):
                    await TriggerCliHost(aliased, loader).retain(
                        source, "task.toml"
                    )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "catalog_root"
                ):
                    await TriggerCliHost(aliased, loader).load(identity)
                record = retained_host.root / identity / "manifest.json"
                backup = root / "manifest.json"
                backup.write_bytes(record.read_bytes())
                record.unlink()
                record.symlink_to(backup)
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "catalog_manifest"
                ):
                    await retained_host.load(identity)
                record.unlink()
                (source / "agent.toml").write_text(
                    (source / "agent.toml")
                    .read_text()
                    .replace("old retained", "new content")
                )
                other = await TriggerCliHost(root / "other", loader).retain(
                    source, "task.toml"
                )
                record.write_bytes(
                    (
                        root
                        / "other"
                        / other.manifest.execution_deployment_id
                        / "manifest.json"
                    ).read_bytes()
                )
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "deployment_id"
                ):
                    await retained_host.load(identity)
                self.assertEqual(bound.manifest, retained.manifest)
