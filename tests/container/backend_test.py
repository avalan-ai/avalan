from asyncio import run as run_async
from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnosticCode,
    ContainerBackendError,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendProbeResult,
    ContainerBackendRuntimeRequirements,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerDeviceClass,
    ContainerDevicePolicy,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputValidationResult,
    ContainerPullPolicy,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerRunPlan,
    select_container_backend,
)

_DIGEST = "7" * 64
_DIGEST_ALT = "8" * 64
_IMAGE = f"ghcr.io/example/backend-tools@sha256:{_DIGEST}"


class ContainerBackendTest(TestCase):
    def test_probe_selection_and_runtime_requirements(self) -> None:
        podman = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(
                    backend=ContainerBackend.PODMAN,
                    rootless=True,
                )
            )
        )
        docker_probe = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(
                        backend=ContainerBackend.DOCKER,
                        rootless=False,
                        vm_isolated=False,
                    )
                )
            ).probe()
        )
        vm_probe = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(
                        backend=ContainerBackend.APPLE_CONTAINER,
                        rootless=False,
                        vm_isolated=True,
                    )
                )
            ).probe()
        )
        podman_probe = run_async(podman.probe())
        marker = ContainerBackendRuntimeRequirements(
            marker="container_runtime",
            environment_variables=("AVALAN_CONTAINER_TESTS",),
            requires_network=False,
            requires_secrets=False,
        )

        explicit = select_container_backend(
            _run_plan(backend=ContainerBackend.PODMAN),
            (docker_probe, podman_probe),
            auto_enabled=False,
        )
        auto = select_container_backend(
            _run_plan(backend=ContainerBackend.AUTO),
            (docker_probe, vm_probe, podman_probe),
            auto_enabled=True,
        )
        empty_auto = select_container_backend(
            _run_plan(backend=ContainerBackend.AUTO),
            (),
            auto_enabled=True,
        )
        auto_disabled = select_container_backend(
            _run_plan(backend=ContainerBackend.AUTO),
            (podman_probe,),
            auto_enabled=False,
        )
        missing_explicit = select_container_backend(
            _run_plan(backend=ContainerBackend.PODMAN),
            (docker_probe,),
            auto_enabled=False,
        )
        rootful_denied = select_container_backend(
            _run_plan(backend=ContainerBackend.DOCKER),
            (docker_probe,),
            auto_enabled=False,
        )
        rootful_allowed = select_container_backend(
            _run_plan(backend=ContainerBackend.DOCKER),
            (docker_probe,),
            auto_enabled=False,
            rootful_authorized=True,
        )

        self.assertTrue(podman_probe.ok)
        self.assertEqual(podman_probe.to_dict()["backend"], "podman")
        self.assertEqual(marker.to_dict()["marker"], "container_runtime")
        self.assertTrue(explicit.ok)
        self.assertEqual(explicit.backend, ContainerBackend.PODMAN)
        self.assertEqual(auto.backend, ContainerBackend.PODMAN)
        self.assertFalse(empty_auto.ok)
        self.assertEqual(
            empty_auto.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertFalse(auto_disabled.ok)
        self.assertEqual(
            auto_disabled.diagnostics[0].code,
            ContainerBackendDiagnosticCode.AUTO_NOT_ENABLED,
        )
        self.assertFalse(missing_explicit.ok)
        self.assertEqual(
            missing_explicit.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertFalse(rootful_denied.ok)
        self.assertEqual(
            rootful_denied.diagnostics[0].code,
            ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
        )
        self.assertTrue(rootful_allowed.ok)

    def test_selection_rejects_unavailable_and_capability_mismatch(
        self,
    ) -> None:
        unavailable = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    available=False,
                )
            ).probe()
        )
        mismatch = select_container_backend(
            _rich_run_plan(),
            (
                unavailable,
                _probe(
                    _capabilities(
                        guest_os="windows",
                        architecture="arm64",
                        platform_emulation=False,
                        pull=False,
                        build=False,
                        network_modes=(ContainerNetworkMode.NONE,),
                        mount_types=(),
                        device_classes=(),
                        resource_limits=False,
                        streaming_attach=False,
                    )
                ),
            ),
            auto_enabled=False,
            rootful_authorized=True,
        )

        codes = {diagnostic.code for diagnostic in mismatch.diagnostics}
        messages = {diagnostic.message for diagnostic in mismatch.diagnostics}

        self.assertFalse(unavailable.ok)
        self.assertFalse(mismatch.ok)
        self.assertIn(
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            codes,
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
            codes,
        )
        self.assertIn("image pull is not supported", messages)
        self.assertIn("image build is not supported", messages)
        self.assertIn("streaming attach is not supported", messages)

    def test_successful_fake_lifecycle_streams_stats_outputs_and_cleanup(
        self,
    ) -> None:
        contract = _output_contract()
        output = ContainerOutputValidationResult(
            decision=ContainerOutputDecisionType.ACCEPT,
            contract=contract,
        )
        fake = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(pull=True),
                resolved_digest=f"sha256:{_DIGEST_ALT}",
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"hello\n",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"warn\n",
                        sequence=1,
                    ),
                ),
                stream_delay_seconds=0.001,
                stats_samples=(
                    ContainerBackendStats(cpu_nanos=1, memory_bytes=2, pids=3),
                ),
                output_result=output,
            )
        )
        result = run_async(
            fake.run(
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                output_contract=contract,
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(result.output, output)
        self.assertEqual(
            [chunk.to_dict()["content"] for chunk in result.stream_chunks],
            ["hello\n", "warn\n"],
        )
        self.assertEqual(result.stats[0].to_dict()["memory_bytes"], 2)
        self.assertEqual(
            result.execution.metadata["cleanup_uncertain"],
            "false",
        )
        self.assertEqual(
            result.to_dict()["cleanup_uncertain"],
            False,
        )
        self.assertEqual(
            fake.operations,
            (
                ContainerBackendOperation.IMAGE_RESOLUTION,
                ContainerBackendOperation.IMAGE_PULL,
                ContainerBackendOperation.CREATE,
                ContainerBackendOperation.ATTACH,
                ContainerBackendOperation.START,
                ContainerBackendOperation.STREAM,
                ContainerBackendOperation.STATS,
                ContainerBackendOperation.WAIT,
                ContainerBackendOperation.INSPECT,
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerBackendOperation.REMOVE,
                ContainerBackendOperation.CLEANUP,
            ),
        )

    def test_policy_denials_for_direct_pull_and_build_operations(self) -> None:
        fake = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        plan = _run_plan()
        image = run_async(fake.resolve_image(plan))

        self.assertTrue(image.ok)
        self.assertEqual(image.to_dict()["digest"], f"sha256:{_DIGEST}")
        with self.assertRaises(ContainerBackendError) as pull_context:
            run_async(fake.pull_image(plan, image))
        with self.assertRaises(ContainerBackendError) as build_context:
            run_async(fake.build_image(plan))

        self.assertEqual(
            pull_context.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.PULL_DENIED,
        )
        self.assertEqual(
            build_context.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BUILD_DENIED,
        )

    def test_lifecycle_denials_and_operation_failures_are_normalized(
        self,
    ) -> None:
        matrix = (
            (
                ContainerBackendOperation.IMAGE_RESOLUTION,
                ContainerBackendDiagnosticCode.IMAGE_DENIED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_PULL,
                ContainerBackendDiagnosticCode.PULL_DENIED,
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_PULL,
                ContainerBackendDiagnosticCode.PULL_FAILED,
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_BUILD,
                ContainerBackendDiagnosticCode.BUILD_FAILED,
                _run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY),
                None,
            ),
            (
                ContainerBackendOperation.CREATE,
                ContainerBackendDiagnosticCode.CREATE_FAILED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.ATTACH,
                ContainerBackendDiagnosticCode.ATTACH_FAILED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.START,
                ContainerBackendDiagnosticCode.START_FAILED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.WAIT,
                ContainerBackendDiagnosticCode.WAIT_FAILED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerBackendDiagnosticCode.COPY_FAILED,
                _run_plan(),
                _output_contract(),
            ),
            (
                ContainerBackendOperation.REMOVE,
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.CLEANUP,
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                _run_plan(),
                None,
            ),
        )

        for operation, code, plan, contract in matrix:
            with self.subTest(operation=operation.value, code=code.value):
                result = run_async(
                    ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(pull=True, build=True),
                            operation_diagnostics={operation: code},
                        )
                    ).run(plan, output_contract=contract)
                )
                codes = {diagnostic.code for diagnostic in result.diagnostics}

                self.assertIn(code, codes)
                self.assertIn(
                    code.value,
                    str(result.diagnostics[0].to_dict()["code"]),
                )
                self.assertEqual(
                    result.execution.status,
                    (
                        ContainerResultStatus.DENIED
                        if code
                        in {
                            ContainerBackendDiagnosticCode.IMAGE_DENIED,
                            ContainerBackendDiagnosticCode.PULL_DENIED,
                        }
                        else ContainerResultStatus.FAILED
                    ),
                )

    def test_build_denial_failure_exit_and_output_rejection_paths(
        self,
    ) -> None:
        build_denied = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(build=True),
                    operation_diagnostics={
                        ContainerBackendOperation.IMAGE_BUILD: (
                            ContainerBackendDiagnosticCode.BUILD_DENIED
                        ),
                    },
                )
            ).run(_run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY))
        )
        failed_exit = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_exit_code=7,
                )
            ).run(_run_plan())
        )
        build_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(build=True),
            )
        )
        build_success = run_async(
            build_backend.run(
                _run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY)
            )
        )
        rejected_output = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    output_result=ContainerOutputValidationResult(
                        decision=ContainerOutputDecisionType.REJECT,
                        contract=_output_contract(),
                    ),
                )
            ).run(_run_plan(), output_contract=_output_contract())
        )

        self.assertEqual(
            build_denied.execution.status,
            ContainerResultStatus.DENIED,
        )
        self.assertEqual(failed_exit.execution.exit_code, 7)
        self.assertEqual(
            failed_exit.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertEqual(
            build_success.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIn(
            ContainerBackendOperation.IMAGE_BUILD,
            build_backend.operations,
        )
        self.assertEqual(
            rejected_output.execution.status,
            ContainerResultStatus.FAILED,
        )

    def test_soft_operation_and_wait_timeout_fail_lifecycle(self) -> None:
        soft_start_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                soft_operation_diagnostics={
                    ContainerBackendOperation.START: (
                        ContainerBackendDiagnosticCode.START_FAILED
                    ),
                },
            )
        )
        wait_timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                wait_timed_out=True,
            )
        )
        soft_image_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                soft_operation_diagnostics={
                    ContainerBackendOperation.IMAGE_RESOLUTION: (
                        ContainerBackendDiagnosticCode.IMAGE_DENIED
                    ),
                },
            )
        )
        soft_pull_denied_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(pull=True),
                soft_operation_diagnostics={
                    ContainerBackendOperation.IMAGE_PULL: (
                        ContainerBackendDiagnosticCode.PULL_DENIED
                    ),
                },
            )
        )

        soft_start = run_async(soft_start_backend.run(_run_plan()))
        wait_timeout = run_async(wait_timeout_backend.run(_run_plan()))
        soft_image = run_async(soft_image_backend.run(_run_plan()))
        soft_pull_denied = run_async(
            soft_pull_denied_backend.run(
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)
            )
        )

        self.assertEqual(
            soft_start.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.START_FAILED,
            {diagnostic.code for diagnostic in soft_start.diagnostics},
        )
        self.assertNotIn(
            ContainerBackendOperation.WAIT,
            soft_start_backend.operations,
        )
        self.assertEqual(
            wait_timeout.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.TIMEOUT,
            {diagnostic.code for diagnostic in wait_timeout.diagnostics},
        )
        self.assertIn(
            ContainerBackendOperation.KILL,
            wait_timeout_backend.operations,
        )
        self.assertEqual(
            soft_image.execution.status,
            ContainerResultStatus.DENIED,
        )
        self.assertEqual(
            soft_pull_denied.execution.status,
            ContainerResultStatus.DENIED,
        )

    def test_cancellation_timeout_stop_kill_and_orphan_quarantine(
        self,
    ) -> None:
        cancelled_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(ContainerBackendOperation.STREAM,),
            )
        )
        timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(ContainerBackendOperation.WAIT,),
            )
        )
        orphan_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cleanup_uncertain=True,
            )
        )
        cancel_kill_failure_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(ContainerBackendOperation.STREAM,),
                operation_diagnostics={
                    ContainerBackendOperation.KILL: (
                        ContainerBackendDiagnosticCode.CLEANUP_FAILED
                    ),
                },
            )
        )
        timeout_kill_failure_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(ContainerBackendOperation.WAIT,),
                operation_diagnostics={
                    ContainerBackendOperation.KILL: (
                        ContainerBackendDiagnosticCode.CLEANUP_FAILED
                    ),
                },
            )
        )
        timeout_soft_orphan_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                wait_timed_out=True,
                soft_operation_diagnostics={
                    ContainerBackendOperation.KILL: (
                        ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
                    ),
                },
            )
        )
        timeout_kill_timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(
                    ContainerBackendOperation.WAIT,
                    ContainerBackendOperation.KILL,
                ),
            )
        )
        cancel_kill_cancel_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(
                    ContainerBackendOperation.STREAM,
                    ContainerBackendOperation.KILL,
                ),
            )
        )
        remove_timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(ContainerBackendOperation.REMOVE,),
            )
        )
        remove_cancel_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(ContainerBackendOperation.REMOVE,),
            )
        )
        cleanup_timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(ContainerBackendOperation.CLEANUP,),
            )
        )
        cleanup_cancel_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(ContainerBackendOperation.CLEANUP,),
            )
        )
        soft_remove_failure_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                soft_operation_diagnostics={
                    ContainerBackendOperation.REMOVE: (
                        ContainerBackendDiagnosticCode.CLEANUP_FAILED
                    ),
                },
            )
        )

        cancelled = run_async(cancelled_backend.run(_run_plan()))
        timed_out = run_async(timeout_backend.run(_run_plan()))
        orphaned = run_async(orphan_backend.run(_run_plan()))
        cancel_kill_failure = run_async(
            cancel_kill_failure_backend.run(_run_plan())
        )
        timeout_kill_failure = run_async(
            timeout_kill_failure_backend.run(_run_plan())
        )
        timeout_soft_orphan = run_async(
            timeout_soft_orphan_backend.run(_run_plan())
        )
        timeout_kill_timeout = run_async(
            timeout_kill_timeout_backend.run(_run_plan())
        )
        cancel_kill_cancel = run_async(
            cancel_kill_cancel_backend.run(_run_plan())
        )
        remove_timeout = run_async(remove_timeout_backend.run(_run_plan()))
        remove_cancel = run_async(remove_cancel_backend.run(_run_plan()))
        cleanup_timeout = run_async(cleanup_timeout_backend.run(_run_plan()))
        cleanup_cancel = run_async(cleanup_cancel_backend.run(_run_plan()))
        soft_remove_failure = run_async(
            soft_remove_failure_backend.run(_run_plan())
        )
        container = run_async(
            ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_capabilities())
            ).create(_run_plan())
        )
        stop_result = run_async(orphan_backend.stop(container))
        kill_result = run_async(orphan_backend.kill(container))

        self.assertEqual(
            cancelled.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertIn(
            ContainerBackendOperation.KILL,
            cancelled_backend.operations,
        )
        self.assertEqual(
            timed_out.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertIn(
            ContainerBackendOperation.KILL,
            timeout_backend.operations,
        )
        self.assertTrue(orphaned.cleanup_uncertain)
        self.assertTrue(orphaned.orphan_quarantined)
        self.assertEqual(
            orphaned.diagnostics[-1].code,
            ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
        )
        self.assertTrue(stop_result.ok)
        self.assertTrue(kill_result.ok)
        self.assertEqual(
            cancel_kill_failure.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertTrue(cancel_kill_failure.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            {
                diagnostic.code
                for diagnostic in cancel_kill_failure.diagnostics
            },
        )
        self.assertEqual(
            timeout_kill_failure.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertTrue(timeout_kill_failure.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            {
                diagnostic.code
                for diagnostic in timeout_kill_failure.diagnostics
            },
        )
        self.assertTrue(timeout_soft_orphan.cleanup_uncertain)
        self.assertTrue(timeout_soft_orphan.orphan_quarantined)
        self.assertTrue(timeout_kill_timeout.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.TIMEOUT,
            {
                diagnostic.code
                for diagnostic in timeout_kill_timeout.diagnostics
                if diagnostic.operation is ContainerBackendOperation.KILL
            },
        )
        self.assertTrue(cancel_kill_cancel.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.CANCELLED,
            {
                diagnostic.code
                for diagnostic in cancel_kill_cancel.diagnostics
                if diagnostic.operation is ContainerBackendOperation.KILL
            },
        )
        for cleanup_failure in (
            remove_timeout,
            remove_cancel,
            cleanup_timeout,
            cleanup_cancel,
            soft_remove_failure,
        ):
            with self.subTest(cleanup_failure=cleanup_failure):
                self.assertEqual(
                    cleanup_failure.execution.status,
                    ContainerResultStatus.FAILED,
                )
                self.assertTrue(cleanup_failure.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.TIMEOUT,
            {
                diagnostic.code
                for diagnostic in remove_timeout.diagnostics
                if diagnostic.operation is ContainerBackendOperation.REMOVE
            },
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.CANCELLED,
            {
                diagnostic.code
                for diagnostic in cleanup_cancel.diagnostics
                if diagnostic.operation is ContainerBackendOperation.CLEANUP
            },
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            {
                diagnostic.code
                for diagnostic in soft_remove_failure.diagnostics
            },
        )

    def test_value_objects_serialize_and_validate(self) -> None:
        operation_result = ContainerBackendOperationResult(
            operation="cleanup",
            metadata={"removed": "true"},
        )
        fake = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        container = run_async(fake.create(_run_plan()))
        inspected = run_async(fake.inspect(container))
        wait_result = run_async(fake.wait(container))

        self.assertTrue(operation_result.ok)
        self.assertEqual(operation_result.to_dict()["operation"], "cleanup")
        self.assertEqual(container.to_dict()["container_id"], "fake-1")
        self.assertEqual(inspected.to_dict()["status"], "exited")
        self.assertTrue(wait_result.ok)
        self.assertEqual(wait_result.to_dict()["exit_code"], 0)
        with self.assertRaises(AssertionError):
            ContainerBackendStats(cpu_nanos=-1)
        with self.assertRaises(AssertionError):
            ContainerBackendRuntimeRequirements(marker="")
        with self.assertRaises(AssertionError):
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=cast(bytes, "bad"),
                sequence=0,
            )


def _capabilities(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    host_os: str = "linux",
    guest_os: str = "linux",
    architecture: str = "amd64",
    platform_emulation: bool = True,
    rootless: bool = True,
    vm_isolated: bool = False,
    pull: bool = True,
    build: bool = True,
    network_modes: tuple[ContainerNetworkMode, ...] = (
        ContainerNetworkMode.NONE,
        ContainerNetworkMode.ALLOWLIST,
    ),
    mount_types: tuple[ContainerMountType, ...] = (
        ContainerMountType.WORKSPACE,
        ContainerMountType.OUTPUT,
    ),
    device_classes: tuple[ContainerDeviceClass, ...] = (
        ContainerDeviceClass.CPU,
    ),
    resource_limits: bool = True,
    streaming_attach: bool = True,
) -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=backend,
        host_os=host_os,
        guest_os=guest_os,
        architecture=architecture,
        platform_emulation=platform_emulation,
        rootless=rootless,
        build=build,
        pull=pull,
        network_modes=network_modes,
        mount_types=mount_types,
        resource_limits=resource_limits,
        device_classes=device_classes,
        per_container_vm_isolation=vm_isolated,
        streaming_attach=streaming_attach,
        stats=True,
    )


def _probe(
    capabilities: ContainerBackendCapabilities,
) -> ContainerBackendProbeResult:
    return run_async(
        ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=capabilities)
        ).probe()
    )


def _run_plan(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    pull_policy: ContainerPullPolicy = ContainerPullPolicy.NEVER,
    build_policy: ContainerBuildPolicy = ContainerBuildPolicy.DISABLED,
    image: ContainerImagePolicy | None = None,
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=backend,
        profile_name="backend-profile",
        image=image
        or ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
        ),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=("echo", "ok"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        network=ContainerNetworkPolicy(),
        policy_version="phase7",
    )


def _rich_run_plan() -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="backend-profile",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=ContainerPullPolicy.IF_MISSING,
            build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
            platform="linux/amd64",
        ),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=("echo", "ok"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        mounts=(
            ContainerMountDeclaration(
                source="out",
                target="/out",
                mount_type=ContainerMountType.OUTPUT,
                access=ContainerMountAccess.WRITE,
            ),
        ),
        network=ContainerNetworkPolicy(
            mode=ContainerNetworkMode.ALLOWLIST,
            egress_allowlist=("api.example.test",),
        ),
        devices=ContainerDevicePolicy(devices=(ContainerDeviceClass.CPU,)),
        resources=ContainerResourceLimits(cpu_count=1),
        policy_version="phase7",
    )


def _output_contract() -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=64,
    )


if __name__ == "__main__":
    main()
