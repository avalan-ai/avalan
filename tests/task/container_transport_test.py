"""Retain invocation mounts across unresolved lifecycle ownership outcomes."""

from asyncio import CancelledError, Event, create_task
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from deployment_catalog_test import catalog_fixture

from avalan.container import (
    ContainerBackend,
    ContainerCommandPlan,
    ContainerEffectiveSettings,
    ContainerExecutionResult,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerManagedLifecycleResult,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerProfile,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    DockerContainerBackend,
)
from avalan.task.canonical import spec_hash
from avalan.task.container_transport import (
    ContainerInvocationOwnership,
    ContainerInvocationUnsettled,
    run_deployment_container,
)
from avalan.task.definition import TaskContainerExecutionSettings
from avalan.task.deployment import ExecutionDeploymentError
from avalan.task.deployment_catalog import ExecutionDeploymentBinding
from avalan.task.provenance import TriggerInvocationContext
from avalan.task.store import TaskExecutionContext

IMAGE = "example.invalid/runtime@sha256:" + "a" * 64


class ContainerTransportTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        fixture = catalog_fixture()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        original = await fixture.binding("container")
        profile = ContainerProfile.minimal_readonly(
            name="test", image_reference=IMAGE
        )
        settings = ContainerEffectiveSettings(
            backend=ContainerBackend.DOCKER,
            required=True,
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            source=ContainerSettingsSource(
                surface=ContainerSurface.SDK,
                trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
            ),
            policy_version="test",
            profile_registry_id="test",
            profile_name="test",
            profile=profile,
        )
        definition = replace(
            original.definition,
            container=TaskContainerExecutionSettings(attempt=settings),
        )
        manifest = replace(
            original.manifest,
            task_hash=await spec_hash(
                definition, schema_base_path=original.application_root
            ),
            container_image_digests=(IMAGE,),
        )
        self.binding: ExecutionDeploymentBinding | None = replace(
            original, manifest=manifest, definition=definition
        )
        now = datetime(2026, 9, 7, tzinfo=UTC)
        self.context = TaskExecutionContext(
            run_id="run",
            attempt_id="attempt",
            attempt_number=1,
            deployment=manifest,
            trigger=TriggerInvocationContext(
                trigger_id="trigger",
                trigger_revision=1,
                occurrence_id="slot",
                scheduled_at=now,
                dispatched_at=now,
            ),
        )
        self.plan = ContainerRunPlan(
            backend=ContainerBackend.DOCKER,
            profile_name="test",
            image=ContainerImagePolicy(reference=IMAGE),
            command=ContainerCommandPlan(
                tool_name="avalan-task",
                command="avalan-task",
                argv=("avalan-task", "agent", "agent.toml"),
                cwd="/workspace",
                scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
            ),
        )
        self.owner = ContainerInvocationOwnership()
        # Every lifecycle in this file is an explicit no-I/O fixture. These
        # teardown releases do not infer real backend settlement from age.
        self.addCleanup(self.release_fixture_staging)
        self.output = ContainerOutputContract(
            contract_type=ContainerOutputContractType.TASK_ARTIFACT,
            max_bytes=4096,
        )

    def release_fixture_staging(self) -> None:
        for staging in self.owner.pending:
            self.owner._settle(staging)

    async def run_transport(
        self, plan: ContainerRunPlan | None = None
    ) -> ContainerManagedLifecycleResult:
        return await run_deployment_container(
            DockerContainerBackend(),
            plan or self.plan,
            context=self.context,
            binding=self.binding,
            output_contract=self.output,
            shutdown_requested=False,
            ownership=self.owner,
        )

    async def test_uncertain_exception_and_quarantine_preserve_recovery_handle(
        self,
    ) -> None:
        for result in (
            ContainerManagedLifecycleResult(
                execution=ContainerExecutionResult(
                    status=ContainerResultStatus.FAILED
                ),
                cleanup_completed=True,
                cleanup_uncertain=True,
            ),
            ContainerManagedLifecycleResult(
                execution=ContainerExecutionResult(
                    status=ContainerResultStatus.FAILED
                ),
                cleanup_completed=False,
            ),
            ContainerManagedLifecycleResult(
                execution=ContainerExecutionResult(
                    status=ContainerResultStatus.FAILED
                ),
                cleanup_completed=True,
                orphan_quarantined=True,
            ),
        ):
            with self.subTest(result=result):
                with patch(
                    "avalan.task.container_transport.run_container_managed_lifecycle",
                    AsyncMock(return_value=result),
                ):
                    with self.assertRaises(
                        ContainerInvocationUnsettled
                    ) as caught:
                        await self.run_transport()
                self.assertIs(caught.exception.result, result)
                self.assertEqual(
                    self.owner.pending, (caught.exception.staging,)
                )
                self.assertTrue(
                    (
                        caught.exception.staging.directory / "invocation.json"
                    ).is_file()
                )
                self.release_fixture_staging()
        with patch(
            "avalan.task.container_transport.run_container_managed_lifecycle",
            AsyncMock(side_effect=OSError("transport unavailable")),
        ):
            with self.assertRaises(OSError):
                await self.run_transport()
        self.assertEqual(len(self.owner.pending), 1)

    async def test_repeated_cancellation_keeps_owned_mount_until_settled(
        self,
    ) -> None:
        entered, first_cancel, second_cancel = Event(), Event(), Event()

        async def interrupted(
            *args: object, **kwargs: object
        ) -> ContainerManagedLifecycleResult:
            entered.set()
            try:
                await Event().wait()
            except CancelledError:
                first_cancel.set()
                try:
                    await Event().wait()
                except CancelledError:
                    second_cancel.set()
                    raise
            raise AssertionError("unreachable fixture continuation")

        with patch(
            "avalan.task.container_transport.run_container_managed_lifecycle",
            new=interrupted,
        ):
            running = create_task(self.run_transport())
            await entered.wait()
            staging = self.owner.pending[0]
            running.cancel()
            await first_cancel.wait()
            self.assertTrue((staging.directory / "invocation.json").exists())
            running.cancel()
            await second_cancel.wait()
            with self.assertRaises(CancelledError):
                await running
        self.assertEqual(self.owner.pending, (staging,))

    async def test_rejects_wrong_binding_image_mount_and_prewrite_failure(
        self,
    ) -> None:
        original = self.binding
        assert original is not None
        self.binding = None
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "container.binding"
        ):
            await self.run_transport()
        self.binding = original
        with self.assertRaisesRegex(
            ExecutionDeploymentError, "container.image"
        ):
            await self.run_transport(
                replace(
                    self.plan,
                    image=ContainerImagePolicy(
                        reference="example.invalid/runtime:tag@sha256:"
                        + "b" * 64
                    ),
                )
            )
        for target, path in (
            ("/workspace", "container.deployment_mount"),
            ("/run/avalan/task-invocation.json", "container.invocation_mount"),
        ):
            with self.assertRaisesRegex(ExecutionDeploymentError, path):
                await self.run_transport(
                    replace(
                        self.plan,
                        mounts=(
                            ContainerMountDeclaration(
                                target=target,
                                source=str(original.application_root),
                                mount_type=ContainerMountType.INPUT,
                            ),
                        ),
                    )
                )
        with patch.object(
            Path, "write_bytes", side_effect=OSError("staging unavailable")
        ):
            with self.assertRaises(OSError):
                await self.run_transport()
        self.assertEqual(self.owner.pending, ())
