"""Derive skill and image requirements from actual typed host settings."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from avalan.container import (
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerProfile,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
)
from avalan.skill import TrustedSkillSettings
from avalan.task import (
    TaskContainerExecutionSettings,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
)
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.deployment_requirements import (
    deployment_container_images,
    deployment_skill_manifest_id,
)


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="requirements", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
    )


class DeploymentRequirementsTest(IsolatedAsyncioTestCase):
    async def test_skill_policy_identity_is_canonical_and_changes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.toml").write_text("[agent]\n[tool]\nenable=[]")
            task = replace(
                definition(), skills=TrustedSkillSettings(enabled=False)
            )
            first = await deployment_skill_manifest_id(
                task, definition_base=root
            )
            self.assertIsNotNone(first)
            self.assertEqual(
                first,
                await deployment_skill_manifest_id(task, definition_base=root),
            )
            assert task.skills is not None
            changed = replace(
                task, skills=replace(task.skills, bootstrap_enabled=False)
            )
            self.assertNotEqual(
                first,
                await deployment_skill_manifest_id(
                    changed, definition_base=root
                ),
            )

    def test_image_identity_uses_pinned_repository_and_checks_profile(
        self,
    ) -> None:
        settings = ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=True,
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            source=ContainerSettingsSource(
                surface=ContainerSurface.SDK,
                trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
            ),
            policy_version="fixture",
            profile_registry_id="fixture",
            allowed_profiles=("fixture",),
        )
        task = replace(
            definition(),
            container=TaskContainerExecutionSettings(attempt=settings),
        )
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "container.profile"
        ):
            deployment_container_images(task)
        profile = ContainerProfile.minimal_readonly(
            name="fixture",
            image_reference="registry:5000/repository:tag@sha256:" + "a" * 64,
        )
        settings = replace(settings, profile=profile, profile_name="fixture")
        task = replace(
            task,
            container=TaskContainerExecutionSettings(
                attempt=settings, worker_envelope=settings
            ),
        )
        self.assertEqual(
            deployment_container_images(task),
            ("registry:5000/repository@sha256:" + "a" * 64,),
        )
