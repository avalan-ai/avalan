from asyncio import CancelledError, StreamReader, create_task, gather, sleep
from asyncio import run as run_async
from collections.abc import Callable, Coroutine, Mapping, Sequence
from os import environ
from pathlib import Path
from time import perf_counter
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendError,
    ContainerBackendOperation,
    ContainerBackendStream,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerLifecycleDeadlines,
    ContainerLifecycleResources,
    ContainerManagedLifecycleResult,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerPullPolicy,
    ContainerResourceLimits,
    ContainerResultStatus,
    ContainerRunPlan,
    DockerCommandResult,
    DockerContainerBackend,
    DockerSubprocessRunner,
    docker_container_capabilities,
    run_container_managed_lifecycle,
    select_container_backend,
)
from avalan.container.docker import (
    _cleanup_failure_code,
    _command_failure_message,
    _docker_info_rootless,
    _first_json_mapping,
    _host_architecture,
    _int_field,
    _local_image_references,
    _memory_usage_bytes,
    _network_argument,
    _quantity_bytes,
    _repo_digests,
    _stats_from_json,
    _timeout_label,
    _wait_exit_code,
)

_DIGEST = "1" * 64
_DIGEST_ALT = "2" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"
_LIVE_E2E = environ.get("AVALAN_CONTAINER_DOCKER_E2E") == "1"
_LIVE_E2E_IMAGE = environ.get("AVALAN_CONTAINER_DOCKER_E2E_IMAGE")


class DockerContainerBackendTest(TestCase):
    def test_probe_reports_supported_capabilities(self) -> None:
        runner = _FakeRunner(info_rootless=True)
        backend = DockerContainerBackend(runner)

        probe = run_async(backend.probe())
        selection = select_container_backend(_run_plan(), (probe,))

        self.assertTrue(probe.ok)
        self.assertTrue(selection.ok)
        self.assertEqual(selection.backend, ContainerBackend.DOCKER)
        self.assertIsInstance(probe.capabilities, ContainerBackendCapabilities)
        assert probe.capabilities is not None
        self.assertEqual(probe.capabilities.backend, ContainerBackend.DOCKER)
        self.assertTrue(probe.capabilities.rootless)
        self.assertTrue(probe.capabilities.pull)
        self.assertFalse(probe.capabilities.build)
        self.assertTrue(probe.capabilities.streaming_attach)
        self.assertTrue(probe.capabilities.stats)
        self.assertNotIn(None, runner.timeouts)

    def test_probe_missing_cli_and_daemon_failure_fail_closed(self) -> None:
        missing = DockerContainerBackend(_FakeRunner(available=False))
        daemon = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("version",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"Cannot connect to the Docker daemon",
                        ),
                    )
                }
            )
        )
        not_found = DockerContainerBackend(
            _FakeRunner(results={("version",): (FileNotFoundError("docker"),)})
        )
        timeout = DockerContainerBackend(
            _FakeRunner(results={("version",): (TimeoutError("slow"),)})
        )
        info_failure = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("info",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"info failed",
                        ),
                    )
                }
            )
        )
        invalid_info = DockerContainerBackend(
            _FakeRunner(results={("info",): (_ok(stdout=b"not json"),)})
        )

        probes = tuple(
            run_async(backend.probe())
            for backend in (
                missing,
                daemon,
                not_found,
                timeout,
                info_failure,
                invalid_info,
            )
        )

        for probe in probes:
            with self.subTest(message=probe.diagnostics[0].message):
                self.assertFalse(probe.available)
                self.assertEqual(
                    probe.diagnostics[0].code,
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                )
                self.assertEqual(
                    probe.diagnostics[0].operation,
                    ContainerBackendOperation.PROBE,
                )

    def test_probe_rootful_socket_requires_authorization(self) -> None:
        probe = run_async(DockerContainerBackend(_FakeRunner()).probe())

        denied = select_container_backend(_run_plan(), (probe,))
        allowed = select_container_backend(
            _run_plan(),
            (probe,),
            rootful_authorized=True,
        )

        self.assertFalse(denied.ok)
        self.assertEqual(
            denied.diagnostics[0].code,
            ContainerBackendDiagnosticCode.ROOTFUL_NOT_AUTHORIZED,
        )
        self.assertTrue(allowed.ok)

    def test_subprocess_runner_uses_docker_cli(self) -> None:
        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _FakeProcess:
            self.assertEqual(args, ("docker", "version"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            return _FakeProcess(returncode=3, stdout=b"out", stderr=b"err")

        with patch(
            "avalan.container.docker.which",
            return_value="/bin/docker",
        ):
            runner = DockerSubprocessRunner()
            self.assertTrue(runner.available())
        with patch(
            "avalan.container.docker.create_subprocess_exec",
            create_process,
        ):
            result = run_async(runner.run(("version",)))

        self.assertEqual(result.return_code, 3)
        self.assertEqual(result.stdout, b"out")
        self.assertEqual(result.stderr, b"err")

    def test_subprocess_runner_bounds_captured_output(self) -> None:
        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _FakeProcess:
            self.assertEqual(args, ("docker", "logs", "cid"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            return _FakeProcess(
                returncode=0,
                stdout=b"x" * 1048580,
                stderr=b"",
            )

        with patch(
            "avalan.container.docker.create_subprocess_exec",
            create_process,
        ):
            result = run_async(DockerSubprocessRunner().run(("logs", "cid")))

        self.assertEqual(len(result.stdout), 1048576)

    def test_subprocess_runner_kills_process_on_cancel(self) -> None:
        processes: list[_CancellingProcess] = []

        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _CancellingProcess:
            self.assertEqual(args, ("docker", "logs", "cid"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            process = _CancellingProcess()
            processes.append(process)
            return process

        with patch(
            "avalan.container.docker.create_subprocess_exec",
            create_process,
        ):
            with self.assertRaises(CancelledError):
                run_async(DockerSubprocessRunner().run(("logs", "cid")))

        self.assertTrue(processes[0].killed)
        self.assertTrue(processes[0].waited)

    def test_subprocess_runner_times_out_and_kills_process(self) -> None:
        processes: list[_HangingProcess] = []

        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _HangingProcess:
            self.assertEqual(args, ("docker", "logs", "cid"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            process = _HangingProcess()
            processes.append(process)
            return process

        with patch(
            "avalan.container.docker.create_subprocess_exec",
            create_process,
        ):
            with self.assertRaises(TimeoutError):
                run_async(
                    DockerSubprocessRunner(
                        default_timeout_seconds=0.001,
                    ).run(("logs", "cid"))
                )

        self.assertTrue(processes[0].killed)
        self.assertTrue(processes[0].waited)

    def test_subprocess_runner_tolerates_racy_process_cleanup(
        self,
    ) -> None:
        processes: list[_RacyCleanupProcess] = []

        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _RacyCleanupProcess:
            self.assertEqual(args, ("docker", "logs", "cid"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            process = _RacyCleanupProcess()
            processes.append(process)
            return process

        with (
            patch(
                "avalan.container.docker.create_subprocess_exec",
                create_process,
            ),
            patch("avalan.container.docker._DOCKER_KILL_GRACE_SECONDS", 0.001),
        ):
            with self.assertRaises(TimeoutError):
                run_async(
                    DockerSubprocessRunner(
                        default_timeout_seconds=0.001,
                    ).run(("logs", "cid"))
                )

        self.assertTrue(processes[0].kill_called)
        self.assertTrue(processes[0].wait_called)
        self.assertEqual(_timeout_label(None), "the configured deadline")

    def test_resolve_pull_and_build_policy(self) -> None:
        backend = DockerContainerBackend(
            _FakeRunner(
                results={("pull",): (_ok(stderr=b"pulled"),)},
            )
        )
        pull_plan = _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)

        image = run_async(backend.resolve_image(pull_plan))
        pull = run_async(backend.pull_image(pull_plan, image))
        with self.assertRaises(ContainerBackendError) as denied:
            run_async(backend.pull_image(_run_plan(), image))
        with self.assertRaises(ContainerBackendError) as build_denied:
            run_async(backend.build_image(pull_plan))

        self.assertTrue(image.ok)
        self.assertTrue(pull.ok)
        self.assertEqual(pull.metadata["return_code"], "0")
        self.assertEqual(
            denied.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.PULL_DENIED,
        )
        self.assertEqual(
            build_denied.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BUILD_DENIED,
        )

    def test_pull_reports_post_pull_image_verification_failure(self) -> None:
        backend = DockerContainerBackend(
            _FakeRunner(
                results={("pull",): (_ok(stderr=b"pulled"),)},
                image_digest=f"sha256:{_DIGEST_ALT}",
            )
        )
        plan = _run_plan(pull_policy=ContainerPullPolicy.ALWAYS)
        image = run_async(backend.resolve_image(plan))

        result = run_async(backend.pull_image(plan, image))

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )

    def test_resolve_allows_missing_image_only_when_pull_is_enabled(
        self,
    ) -> None:
        missing = DockerContainerBackend(_FakeRunner(image_digest=None))
        timeout = DockerContainerBackend(
            _FakeRunner(
                results={("image", "inspect"): (TimeoutError("slow"),)}
            )
        )
        cli_error = DockerContainerBackend(
            _FakeRunner(
                results={("image", "inspect"): (OSError("missing docker"),)}
            )
        )

        denied = run_async(missing.resolve_image(_run_plan()))
        allowed = run_async(
            missing.resolve_image(
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)
            )
        )
        timed_out = run_async(timeout.resolve_image(_run_plan()))
        unavailable = run_async(cli_error.resolve_image(_run_plan()))

        self.assertFalse(denied.ok)
        self.assertEqual(
            denied.diagnostics[0].code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertTrue(allowed.ok)
        self.assertEqual(
            timed_out.diagnostics[0].code,
            ContainerBackendDiagnosticCode.TIMEOUT,
        )
        self.assertEqual(
            unavailable.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_digest_and_platform_mismatches_fail_closed(self) -> None:
        digest_mismatch = DockerContainerBackend(
            _FakeRunner(image_digest=f"sha256:{_DIGEST_ALT}")
        )
        platform_mismatch = DockerContainerBackend(
            _FakeRunner(image_os="linux", image_architecture="arm64")
        )
        os_mismatch = DockerContainerBackend(
            _FakeRunner(image_os="freebsd", image_architecture="amd64")
        )
        missing_digest = DockerContainerBackend(
            _FakeRunner(image_inspect_stdout=b'{"RepoDigests": []}')
        )
        config_id_only = DockerContainerBackend(
            _FakeRunner(
                image_inspect_stdout=(
                    b'{"RepoDigests": [], "Id": "sha256:'
                    + _DIGEST.encode("ascii")
                    + b'", "Os": "linux", "Architecture": "amd64"}'
                )
            )
        )
        missing_platform = DockerContainerBackend(
            _FakeRunner(
                image_inspect_stdout=(
                    b'{"RepoDigests": ["ghcr.io/example/sdk-tools@sha256:'
                    + _DIGEST.encode("ascii")
                    + b'"]}'
                )
            )
        )

        digest = run_async(digest_mismatch.resolve_image(_run_plan()))
        platform = run_async(platform_mismatch.resolve_image(_run_plan()))
        os_platform = run_async(os_mismatch.resolve_image(_run_plan()))
        missing = run_async(missing_digest.resolve_image(_run_plan()))
        config_id = run_async(config_id_only.resolve_image(_run_plan()))
        platformless = run_async(missing_platform.resolve_image(_run_plan()))

        self.assertFalse(digest.ok)
        self.assertEqual(
            digest.diagnostics[0].code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn("digest does not match", digest.diagnostics[0].message)
        self.assertFalse(platform.ok)
        self.assertEqual(
            platform.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertIn("architecture", platform.diagnostics[0].message)
        self.assertFalse(os_platform.ok)
        self.assertIn("image OS", os_platform.diagnostics[0].message)
        self.assertFalse(missing.ok)
        self.assertIn("digest is unavailable", missing.diagnostics[0].message)
        self.assertFalse(config_id.ok)
        self.assertIn(
            "digest is unavailable",
            config_id.diagnostics[0].message,
        )
        self.assertFalse(platformless.ok)
        self.assertEqual(
            platformless.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertIn(
            "platform metadata",
            platformless.diagnostics[0].message,
        )

    def test_create_start_stream_wait_inspect_stats_copy_and_cleanup(
        self,
    ) -> None:
        runner = _FakeRunner(
            logs_stdout=b"out\n",
            logs_stderr=b"err\n",
            wait_exit_code=7,
            copy_entries={"report.txt": b"summary"},
        )
        backend = DockerContainerBackend(runner)
        plan = _run_plan(
            resources=ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=2_097_153,
                pids=64,
                timeout_seconds=30,
            ),
            mounts=(
                ContainerMountDeclaration(
                    source=".",
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                    access=ContainerMountAccess.READ,
                ),
                ContainerMountDeclaration(
                    target="/scratch",
                    mount_type=ContainerMountType.SCRATCH,
                    access=ContainerMountAccess.WRITE,
                ),
            ),
        )

        container = run_async(backend.create(plan))
        attach = run_async(backend.attach(container))
        start = run_async(backend.start(container))
        chunks = run_async(backend.stream(container))
        stats = run_async(backend.stats(container))
        wait = run_async(backend.wait(container))
        inspection = run_async(backend.inspect(container))
        output = run_async(backend.copy_outputs(container, _output_contract()))
        stop = run_async(backend.stop(container))
        kill = run_async(backend.kill(container))
        remove = run_async(backend.remove(container))
        cleanup = run_async(backend.cleanup(container))

        self.assertTrue(attach.ok)
        self.assertTrue(start.ok)
        self.assertEqual(wait.exit_code, 7)
        self.assertEqual(inspection.status, "exited")
        self.assertEqual(inspection.exit_code, 7)
        self.assertEqual(stats[0].memory_bytes, 2 * 1024 * 1024)
        self.assertEqual(output.decision, ContainerOutputDecisionType.ACCEPT)
        self.assertTrue(stop.ok)
        self.assertTrue(kill.ok)
        self.assertTrue(remove.ok)
        self.assertTrue(cleanup.ok)
        self.assertEqual(
            [(chunk.stream, chunk.content) for chunk in chunks],
            [
                (ContainerBackendStream.STDOUT, b"out\n"),
                (ContainerBackendStream.STDERR, b"err\n"),
            ],
        )
        create_args = _first_call(runner, "create")
        self.assertIn("--read-only", create_args)
        self.assertIn("--cap-drop", create_args)
        self.assertIn("ALL", create_args)
        self.assertIn("--security-opt", create_args)
        self.assertIn("no-new-privileges", create_args)
        self.assertIn("--network", create_args)
        self.assertIn("none", create_args)
        self.assertIn("--cpus", create_args)
        self.assertIn("2", create_args)
        self.assertIn("--memory", create_args)
        self.assertIn("2097153", create_args)
        self.assertIn("--pids-limit", create_args)
        self.assertIn("64", create_args)
        self.assertIn("--mount", create_args)
        self.assertIn("--tmpfs", create_args)
        self.assertIn(_IMAGE, create_args)
        self.assertTrue(_has_call(runner, ("logs", "--follow")))
        self.assertTrue(_has_call(runner, ("wait",)))
        self.assertTrue(_has_call(runner, ("stats", "--no-stream")))
        self.assertTrue(_has_call(runner, ("cp",)))
        self.assertTrue(_has_call(runner, ("rm", "--force", "--volumes")))
        self.assertNotIn(None, runner.timeouts)

    def test_create_reports_policy_and_operation_failures(self) -> None:
        unsupported = DockerContainerBackend(_FakeRunner())
        writable_workspace = ContainerMountDeclaration(
            source=".",
            target="/workspace",
            mount_type=ContainerMountType.WORKSPACE,
            access=ContainerMountAccess.READ,
        )
        object.__setattr__(
            writable_workspace,
            "access",
            ContainerMountAccess.WRITE,
        )

        image = run_async(
            unsupported.resolve_image(
                _run_plan(
                    backend=ContainerBackend.APPLE_CONTAINER,
                    build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                    environment_names=("PATH",),
                    mounts=(
                        ContainerMountDeclaration(
                            target="/input",
                            mount_type=ContainerMountType.INPUT,
                        ),
                        writable_workspace,
                        ContainerMountDeclaration(
                            source=".",
                            target="/secret",
                            mount_type=ContainerMountType.SECRET,
                        ),
                    ),
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("example.test",),
                    ),
                    secret_names=("TOKEN",),
                )
            )
        )
        create_failure_runner = _FakeRunner(
            results={
                ("create",): (
                    DockerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"mount denied",
                    ),
                )
            }
        )
        create_failure = DockerContainerBackend(create_failure_runner)
        image_failure = DockerContainerBackend(
            _FakeRunner(image_digest=f"sha256:{_DIGEST_ALT}")
        )

        messages = tuple(
            diagnostic.message for diagnostic in image.diagnostics
        )
        with self.assertRaises(ContainerBackendError) as invalid_create:
            run_async(
                unsupported.create(
                    _run_plan(backend=ContainerBackend.APPLE_CONTAINER)
                )
            )
        with self.assertRaises(ContainerBackendError) as denied_image:
            run_async(image_failure.create(_run_plan()))
        with self.assertRaises(ContainerBackendError) as raised:
            run_async(create_failure.create(_run_plan()))

        self.assertEqual(
            invalid_create.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            denied_image.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn("plan backend is not docker", messages)
        self.assertIn("Docker backend does not build shell images", messages)
        self.assertIn("network mode allowlist is not supported", messages)
        self.assertIn("network egress allowlists are not supported", messages)
        self.assertIn("environment inheritance is not supported", messages)
        self.assertIn("secret injection is not supported", messages)
        self.assertIn("workspace mount must be read-only", messages)
        self.assertIn("mount type secret is not supported", messages)
        self.assertIn("mount source is required", messages)
        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CREATE_FAILED,
        )
        self.assertIn("mount denied", raised.exception.diagnostic.message)
        self.assertTrue(
            _has_call(
                create_failure_runner,
                (
                    "rm",
                    "--force",
                    "--volumes",
                    _created_container_name(create_failure_runner),
                ),
            )
        )

    def test_create_cleans_generated_name_on_failure_timeout_and_cancel(
        self,
    ) -> None:
        cases: tuple[
            tuple[
                str,
                DockerCommandResult | BaseException,
                ContainerBackendDiagnosticCode | None,
            ],
            ...,
        ] = (
            (
                "failure",
                DockerCommandResult(
                    args=(),
                    return_code=1,
                    stderr=b"create failed after daemon request",
                ),
                ContainerBackendDiagnosticCode.CREATE_FAILED,
            ),
            (
                "timeout",
                TimeoutError("create timed out"),
                ContainerBackendDiagnosticCode.TIMEOUT,
            ),
            ("cancel", CancelledError(), None),
        )

        for name, result, code in cases:
            with self.subTest(name=name):
                runner = _FakeRunner(results={("create",): (result,)})
                backend = DockerContainerBackend(runner)

                if code is None:
                    with self.assertRaises(CancelledError):
                        run_async(backend.create(_run_plan()))
                else:
                    with self.assertRaises(ContainerBackendError) as raised:
                        run_async(backend.create(_run_plan()))
                    self.assertEqual(raised.exception.diagnostic.code, code)

                self.assertTrue(
                    _has_call(
                        runner,
                        (
                            "rm",
                            "--force",
                            "--volumes",
                            _created_container_name(runner),
                        ),
                    )
                )

    def test_create_reports_failed_cleanup_after_daemon_side_create(
        self,
    ) -> None:
        cases: tuple[
            tuple[
                str,
                DockerCommandResult | BaseException,
                ContainerBackendDiagnosticCode,
            ],
            ...,
        ] = (
            (
                "cleanup failed",
                DockerCommandResult(
                    args=(),
                    return_code=1,
                    stderr=b"remove failed",
                ),
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            ),
            (
                "orphan quarantined",
                DockerCommandResult(
                    args=(),
                    return_code=1,
                    stderr=b"orphan permission denied",
                ),
                ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
            ),
            (
                "cleanup timeout",
                TimeoutError("remove timed out"),
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            ),
            (
                "cleanup os error",
                OSError("missing docker"),
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
            ),
        )
        create_result = DockerCommandResult(
            args=(),
            return_code=1,
            stderr=b"create failed after daemon request",
        )

        for name, cleanup_result, code in cases:
            with self.subTest(name=name):
                runner = _FakeRunner(
                    results={
                        ("create",): (create_result,),
                        ("rm",): (cleanup_result,),
                    }
                )

                with self.assertRaises(ContainerBackendError) as raised:
                    run_async(
                        DockerContainerBackend(runner).create(_run_plan())
                    )

                self.assertEqual(raised.exception.diagnostic.code, code)
                self.assertEqual(
                    raised.exception.diagnostic.operation,
                    ContainerBackendOperation.CLEANUP,
                )
                self.assertTrue(
                    _has_call(
                        runner,
                        (
                            "rm",
                            "--force",
                            "--volumes",
                            _created_container_name(runner),
                        ),
                    )
                )

        cancel_cleanup_runner = _FakeRunner(
            results={
                ("create",): (CancelledError(),),
                ("rm",): (
                    DockerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"orphan permission denied",
                    ),
                ),
            }
        )
        with self.assertRaises(ContainerBackendError) as cancel_cleanup:
            run_async(
                DockerContainerBackend(cancel_cleanup_runner).create(
                    _run_plan()
                )
            )
        self.assertEqual(
            cancel_cleanup.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
        )

    def test_failed_create_cleanup_reports_interruption(self) -> None:
        async def run_interrupted_cleanup() -> ContainerBackendDiagnosticCode:
            backend = DockerContainerBackend(
                _FakeRunner(delays={("rm",): 0.05})
            )
            cleanup = create_task(
                backend._cleanup_after_failed_create("avalan-orphan")
            )
            await sleep(0)
            cleanup.cancel()
            diagnostic = await cleanup
            assert diagnostic is not None
            return diagnostic.code

        self.assertEqual(
            run_async(run_interrupted_cleanup()),
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        )

    def test_daemon_failures_after_probe_are_backend_unavailable(self) -> None:
        daemon_failure = DockerCommandResult(
            args=(),
            return_code=1,
            stderr=b"Cannot connect to the Docker daemon",
        )
        image = run_async(
            DockerContainerBackend(
                _FakeRunner(results={("image", "inspect"): (daemon_failure,)})
            ).resolve_image(_run_plan())
        )

        self.assertFalse(image.ok)
        self.assertEqual(
            image.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            image.diagnostics[0].operation,
            ContainerBackendOperation.IMAGE_RESOLUTION,
        )

        create_runner = _FakeRunner(results={("create",): (daemon_failure,)})
        with self.assertRaises(ContainerBackendError) as create_error:
            run_async(
                DockerContainerBackend(create_runner).create(_run_plan())
            )
        self.assertEqual(
            create_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            create_error.exception.diagnostic.operation,
            ContainerBackendOperation.CREATE,
        )
        self.assertTrue(
            _has_call(
                create_runner,
                (
                    "rm",
                    "--force",
                    "--volumes",
                    _created_container_name(create_runner),
                ),
            )
        )

        action_type = Callable[
            [DockerContainerBackend, ContainerBackendContainer],
            Coroutine[object, object, object],
        ]
        cases: tuple[
            tuple[
                str,
                tuple[str, ...],
                ContainerBackendOperation,
                action_type,
            ],
            ...,
        ] = (
            (
                "start",
                ("start",),
                ContainerBackendOperation.START,
                lambda backend, container: backend.start(container),
            ),
            (
                "wait",
                ("wait",),
                ContainerBackendOperation.WAIT,
                lambda backend, container: backend.wait(container),
            ),
            (
                "copy",
                ("cp",),
                ContainerBackendOperation.COPY_OUTPUTS,
                lambda backend, container: backend.copy_outputs(
                    container,
                    _output_contract(),
                ),
            ),
            (
                "stream",
                ("logs",),
                ContainerBackendOperation.STREAM,
                lambda backend, container: backend.stream(container),
            ),
        )

        for name, prefix, operation, action in cases:
            with self.subTest(name=name):
                backend = DockerContainerBackend(
                    _FakeRunner(results={prefix: (daemon_failure,)})
                )
                container = run_async(backend.create(_run_plan()))

                with self.assertRaises(ContainerBackendError) as raised:
                    run_async(action(backend, container))

                self.assertEqual(
                    raised.exception.diagnostic.code,
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                )
                self.assertEqual(
                    raised.exception.diagnostic.operation,
                    operation,
                )

    def test_operation_failures_are_normalized(self) -> None:
        action_type = Callable[
            [DockerContainerBackend, ContainerBackendContainer],
            Coroutine[object, object, object],
        ]
        cases: tuple[
            tuple[
                str,
                tuple[str, ...],
                ContainerBackendOperation,
                ContainerBackendDiagnosticCode,
                action_type,
            ],
            ...,
        ] = (
            (
                "start",
                ("start",),
                ContainerBackendOperation.START,
                ContainerBackendDiagnosticCode.START_FAILED,
                lambda backend, container: backend.start(container),
            ),
            (
                "stream",
                ("logs",),
                ContainerBackendOperation.STREAM,
                ContainerBackendDiagnosticCode.ATTACH_FAILED,
                lambda backend, container: backend.stream(container),
            ),
            (
                "wait",
                ("wait",),
                ContainerBackendOperation.WAIT,
                ContainerBackendDiagnosticCode.WAIT_FAILED,
                lambda backend, container: backend.wait(container),
            ),
            (
                "inspect",
                ("inspect",),
                ContainerBackendOperation.INSPECT,
                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                lambda backend, container: backend.inspect(container),
            ),
        )

        for name, prefix, operation, code, action in cases:
            with self.subTest(name=name):
                backend = DockerContainerBackend(
                    _FakeRunner(
                        results={
                            prefix: (
                                DockerCommandResult(
                                    args=(),
                                    return_code=1,
                                    stderr=f"{name} failed".encode(),
                                ),
                            )
                        }
                    )
                )
                container = run_async(backend.create(_run_plan()))

                with self.assertRaises(ContainerBackendError) as raised:
                    run_async(action(backend, container))

                self.assertEqual(raised.exception.diagnostic.code, code)
                self.assertEqual(
                    raised.exception.diagnostic.operation,
                    operation,
                )

        wait_invalid = DockerContainerBackend(
            _FakeRunner(results={("wait",): (_ok(stdout=b"not an int"),)})
        )
        wait_container = run_async(wait_invalid.create(_run_plan()))
        with self.assertRaises(ContainerBackendError) as wait_error:
            run_async(wait_invalid.wait(wait_container))
        self.assertEqual(
            wait_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.WAIT_FAILED,
        )

    def test_operation_timeouts_and_cli_start_failures_are_normalized(
        self,
    ) -> None:
        action_type = Callable[
            [DockerContainerBackend, ContainerBackendContainer],
            Coroutine[object, object, object],
        ]
        cases: tuple[
            tuple[
                str,
                tuple[str, ...],
                BaseException,
                ContainerBackendOperation,
                ContainerBackendDiagnosticCode,
                action_type,
            ],
            ...,
        ] = (
            (
                "stream timeout",
                ("logs",),
                TimeoutError("slow"),
                ContainerBackendOperation.STREAM,
                ContainerBackendDiagnosticCode.TIMEOUT,
                lambda backend, container: backend.stream(container),
            ),
            (
                "stream os error",
                ("logs",),
                OSError("missing docker"),
                ContainerBackendOperation.STREAM,
                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                lambda backend, container: backend.stream(container),
            ),
            (
                "start os error",
                ("start",),
                OSError("missing docker"),
                ContainerBackendOperation.START,
                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                lambda backend, container: backend.start(container),
            ),
            (
                "wait timeout",
                ("wait",),
                TimeoutError("slow"),
                ContainerBackendOperation.WAIT,
                ContainerBackendDiagnosticCode.TIMEOUT,
                lambda backend, container: backend.wait(container),
            ),
            (
                "copy timeout",
                ("cp",),
                TimeoutError("slow"),
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerBackendDiagnosticCode.TIMEOUT,
                lambda backend, container: backend.copy_outputs(
                    container,
                    _output_contract(),
                ),
            ),
            (
                "copy os error",
                ("cp",),
                OSError("missing docker"),
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                lambda backend, container: backend.copy_outputs(
                    container,
                    _output_contract(),
                ),
            ),
        )

        for name, prefix, error, operation, code, action in cases:
            with self.subTest(name=name):
                backend = DockerContainerBackend(
                    _FakeRunner(results={prefix: (error,)})
                )
                container = run_async(backend.create(_run_plan()))

                with self.assertRaises(ContainerBackendError) as raised:
                    run_async(action(backend, container))

                self.assertEqual(raised.exception.diagnostic.code, code)
                self.assertEqual(
                    raised.exception.diagnostic.operation,
                    operation,
                )

    def test_stats_copy_validation_cleanup_and_orphan_failures(self) -> None:
        stats_backend = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("stats",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"Cannot connect to the Docker daemon",
                        ),
                    )
                }
            )
        )
        stats_non_backend = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("stats",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"container stopped",
                        ),
                    )
                }
            )
        )
        rejected_copy = DockerContainerBackend(
            _FakeRunner(copy_entries={"bad.bin": b"MZunsafe"})
        )
        copy_failure = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("cp",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"copy denied",
                        ),
                    )
                }
            )
        )
        stats_timeout = DockerContainerBackend(
            _FakeRunner(results={("stats",): (TimeoutError("slow"),)})
        )
        stats_os_error = DockerContainerBackend(
            _FakeRunner(results={("stats",): (OSError("missing docker"),)})
        )
        cleanup_failure = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("rm",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"cleanup failed",
                        ),
                    )
                }
            )
        )
        cleanup_timeout = DockerContainerBackend(
            _FakeRunner(results={("stop",): (TimeoutError("slow"),)})
        )
        cleanup_os_error = DockerContainerBackend(
            _FakeRunner(results={("stop",): (OSError("missing docker"),)})
        )
        stop_orphan_marker = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("stop",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"orphan permission denied",
                        ),
                    )
                }
            )
        )
        orphan = DockerContainerBackend(
            _FakeRunner(
                results={
                    ("rm",): (
                        DockerCommandResult(
                            args=(),
                            return_code=1,
                            stderr=b"orphan permission denied",
                        ),
                    )
                }
            )
        )

        stats_backend_container = run_async(stats_backend.create(_run_plan()))
        stats_non_backend_container = run_async(
            stats_non_backend.create(_run_plan())
        )
        rejected_container = run_async(rejected_copy.create(_run_plan()))
        copy_failure_container = run_async(copy_failure.create(_run_plan()))
        stats_timeout_container = run_async(stats_timeout.create(_run_plan()))
        stats_os_error_container = run_async(
            stats_os_error.create(_run_plan())
        )
        cleanup_container = run_async(cleanup_failure.create(_run_plan()))
        cleanup_timeout_container = run_async(
            cleanup_timeout.create(_run_plan())
        )
        cleanup_os_error_container = run_async(
            cleanup_os_error.create(_run_plan())
        )
        stop_orphan_container = run_async(
            stop_orphan_marker.create(_run_plan())
        )
        orphan_container = run_async(orphan.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as stats_error:
            run_async(stats_backend.stats(stats_backend_container))
        stats = run_async(stats_non_backend.stats(stats_non_backend_container))
        rejected = run_async(
            rejected_copy.copy_outputs(rejected_container, _output_contract())
        )
        with self.assertRaises(ContainerBackendError) as copy_error:
            run_async(
                copy_failure.copy_outputs(
                    copy_failure_container,
                    _output_contract(),
                )
            )
        with self.assertRaises(ContainerBackendError) as stats_timeout_error:
            run_async(stats_timeout.stats(stats_timeout_container))
        with self.assertRaises(ContainerBackendError) as stats_os_error_raised:
            run_async(stats_os_error.stats(stats_os_error_container))
        cleanup = run_async(cleanup_failure.remove(cleanup_container))
        cleanup_timed_out = run_async(
            cleanup_timeout.stop(cleanup_timeout_container)
        )
        with self.assertRaises(
            ContainerBackendError
        ) as cleanup_os_error_raised:
            run_async(cleanup_os_error.stop(cleanup_os_error_container))
        stop_orphan = run_async(stop_orphan_marker.stop(stop_orphan_container))
        orphaned = run_async(orphan.remove(orphan_container))

        self.assertEqual(
            stats_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(stats, ())
        self.assertEqual(rejected.decision, ContainerOutputDecisionType.REJECT)
        self.assertEqual(
            copy_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.COPY_FAILED,
        )
        self.assertEqual(
            stats_timeout_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.TIMEOUT,
        )
        self.assertEqual(
            stats_os_error_raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            cleanup.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        )
        self.assertEqual(
            cleanup_timed_out.diagnostics[0].code,
            ContainerBackendDiagnosticCode.TIMEOUT,
        )
        self.assertEqual(
            cleanup_os_error_raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            stop_orphan.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        )
        self.assertEqual(
            orphaned.diagnostics[0].code,
            ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED,
        )

    def test_state_mismatch_and_unknown_state_fail_closed(self) -> None:
        backend = DockerContainerBackend(_FakeRunner())
        apple_container = ContainerBackendContainer(
            container_id="apple-id",
            backend=ContainerBackend.APPLE_CONTAINER,
            plan_fingerprint="fingerprint",
        )
        missing_container = ContainerBackendContainer(
            container_id="missing-id",
            backend=ContainerBackend.DOCKER,
            plan_fingerprint="fingerprint",
        )

        with self.assertRaises(ContainerBackendError) as wrong_backend:
            run_async(backend.attach(apple_container))
        with self.assertRaises(ContainerBackendError) as missing_state:
            run_async(backend.attach(missing_container))

        self.assertEqual(
            wrong_backend.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            missing_state.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_docker_backend_helper_edge_cases(self) -> None:
        empty_failure = DockerCommandResult(args=(), return_code=9)
        stop_orphan = DockerCommandResult(
            args=(),
            return_code=1,
            stderr=b"orphan permission denied",
        )

        self.assertEqual(
            _command_failure_message("failed", empty_failure),
            "failed: exit code 9",
        )
        self.assertEqual(
            _cleanup_failure_code(
                stop_orphan,
                ContainerBackendOperation.STOP,
            ),
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        )
        self.assertEqual(
            _first_json_mapping(b'[{"ok": true}]'),
            {"ok": True},
        )
        self.assertEqual(_first_json_mapping(b"[1]"), {})
        self.assertEqual(_first_json_mapping(b"1"), {})
        self.assertEqual(_first_json_mapping(b"\xff"), {})
        self.assertEqual(_repo_digests({"RepoDigests": "bad"}), ())
        self.assertEqual(
            _repo_digests({"RepoDigests": [123, "bad", "n@sha256:abc"]}),
            ("sha256:abc",),
        )
        self.assertEqual(
            _network_argument(ContainerNetworkMode.FULL), "bridge"
        )
        self.assertEqual(
            _local_image_references(
                "registry.example.test/team/tool:1@sha256:" + _DIGEST
            ),
            (
                "registry.example.test/team/tool:1@sha256:" + _DIGEST,
                "registry.example.test/team/tool:1",
            ),
        )
        self.assertEqual(_memory_usage_bytes(None), None)
        self.assertEqual(_quantity_bytes(""), None)
        self.assertEqual(_quantity_bytes("abc"), None)
        self.assertEqual(_quantity_bytes("1xb"), None)
        self.assertEqual(_wait_exit_code(b""), None)
        self.assertEqual(_wait_exit_code(b"not int"), None)
        self.assertFalse(
            _docker_info_rootless({"SecurityOptions": "rootless"})
        )
        self.assertEqual(_stats_from_json(b"not json"), ())
        self.assertEqual(_stats_from_json(b"[1]"), ())
        self.assertEqual(
            _stats_from_json(
                b'[{"cpu_nanos": "5", "memory_bytes": "6", "pids": "7"}]'
            )[0].memory_bytes,
            6,
        )
        self.assertEqual(
            _stats_from_json(b'{"MemUsage": "bad / 1GiB"}')[0].memory_bytes,
            0,
        )
        self.assertEqual(
            _stats_from_json(b'{"State": {"ExitCode": "8"}}')[0].cpu_nanos,
            0,
        )
        self.assertEqual(
            _int_field({"State": "bad"}, ("State", "ExitCode")), None
        )
        with patch("avalan.container.docker.machine", return_value="x86_64"):
            self.assertEqual(_host_architecture(), "amd64")
        with patch("avalan.container.docker.machine", return_value="aarch64"):
            self.assertEqual(_host_architecture(), "arm64")
        with patch("avalan.container.docker.machine", return_value=""):
            self.assertEqual(_host_architecture(), "unknown")

    def test_managed_lifecycle_success_timeout_cancellation_and_orphan(
        self,
    ) -> None:
        success = run_async(
            run_container_managed_lifecycle(
                DockerContainerBackend(
                    _FakeRunner(
                        logs_stdout=b"ok\n",
                        copy_entries={"result.txt": b"ok"},
                    )
                ),
                _run_plan(
                    resources=ContainerResourceLimits(timeout_seconds=5)
                ),
                lifecycle_resources=ContainerLifecycleResources(),
                output_contract=_output_contract(),
            )
        )
        timeout = run_async(
            run_container_managed_lifecycle(
                DockerContainerBackend(_FakeRunner(delays={("logs",): 0.05})),
                _run_plan(
                    resources=ContainerResourceLimits(timeout_seconds=1)
                ),
                lifecycle_resources=ContainerLifecycleResources(),
                deadlines=ContainerLifecycleDeadlines(
                    execution_seconds=0.001,
                    cleanup_seconds=0.1,
                ),
            )
        )
        orphan = run_async(
            run_container_managed_lifecycle(
                DockerContainerBackend(
                    _FakeRunner(
                        results={
                            ("rm",): (
                                DockerCommandResult(
                                    args=(),
                                    return_code=1,
                                    stderr=b"orphan permission denied",
                                ),
                            )
                        }
                    )
                ),
                _run_plan(
                    resources=ContainerResourceLimits(timeout_seconds=5)
                ),
                lifecycle_resources=ContainerLifecycleResources(),
            )
        )
        cancelled = run_async(_cancelled_lifecycle())

        self.assertEqual(
            success.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(success.stream.stdout_chunks[0].content, b"ok\n")
        self.assertTrue(success.cleanup_completed)
        assert success.output is not None
        self.assertEqual(
            success.output.decision,
            ContainerOutputDecisionType.ACCEPT,
        )
        self.assertEqual(
            timeout.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertTrue(timeout.cleanup_completed)
        self.assertIsNotNone(timeout.timed_out_phase)
        self.assertTrue(orphan.cleanup_uncertain)
        self.assertTrue(orphan.orphan_quarantined)
        self.assertEqual(
            cancelled.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertIsNotNone(cancelled.cancelled_phase)

    def test_concurrent_fake_runs_do_not_block_event_loop(self) -> None:
        async def run_many() -> tuple[float, int]:
            ticker = 0

            async def tick() -> None:
                nonlocal ticker
                for _ in range(5):
                    await sleep(0.01)
                    ticker += 1

            async def run_one(index: int) -> object:
                return await run_container_managed_lifecycle(
                    DockerContainerBackend(
                        _FakeRunner(
                            logs_stdout=f"ok-{index}\n".encode(),
                            delays={("logs",): 0.05},
                        )
                    ),
                    _run_plan(),
                    lifecycle_resources=ContainerLifecycleResources(),
                    deadlines=ContainerLifecycleDeadlines(
                        execution_seconds=0.2,
                        cleanup_seconds=0.2,
                    ),
                )

            started = perf_counter()
            await gather(tick(), *(run_one(index) for index in range(4)))
            return perf_counter() - started, ticker

        duration, ticks = run_async(run_many())

        self.assertLess(duration, 0.18)
        self.assertGreaterEqual(ticks, 4)

    def test_capability_helper_validates_rootless_argument(self) -> None:
        capabilities = docker_container_capabilities(rootless=True)

        self.assertEqual(capabilities.backend, ContainerBackend.DOCKER)
        self.assertTrue(capabilities.rootless)
        self.assertFalse(capabilities.build)

    @skipUnless(
        _LIVE_E2E,
        "set AVALAN_CONTAINER_DOCKER_E2E=1 to run Docker e2e",
    )
    def test_live_docker_backend_e2e(self) -> None:
        if not _LIVE_E2E_IMAGE:
            self.skipTest("set AVALAN_CONTAINER_DOCKER_E2E_IMAGE")
        image_reference = _LIVE_E2E_IMAGE
        assert image_reference is not None
        if "@sha256:" not in image_reference:
            self.skipTest(
                "set AVALAN_CONTAINER_DOCKER_E2E_IMAGE to a digest-pinned "
                "image reference"
            )
        backend = DockerContainerBackend()
        probe = run_async(backend.probe())
        if not probe.ok:
            self.skipTest(probe.diagnostics[0].message)
        plan = _run_plan(
            command="sh",
            argv=("sh", "-c", "printf live-ok"),
            image=ContainerImagePolicy(
                reference=image_reference,
                pull_policy=ContainerPullPolicy.IF_MISSING,
                platform="linux/amd64",
            ),
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                lifecycle_resources=ContainerLifecycleResources(),
                deadlines=ContainerLifecycleDeadlines(
                    pull_seconds=60,
                    create_seconds=10,
                    start_seconds=10,
                    execution_seconds=20,
                    cleanup_seconds=10,
                ),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(result.stream.stdout_chunks[0].content, b"live-ok")


async def _cancelled_lifecycle() -> ContainerManagedLifecycleResult:
    task = create_task(
        run_container_managed_lifecycle(
            DockerContainerBackend(_FakeRunner(delays={("logs",): 0.05})),
            _run_plan(),
            lifecycle_resources=ContainerLifecycleResources(),
            deadlines=ContainerLifecycleDeadlines(
                execution_seconds=1,
                cleanup_seconds=0.2,
            ),
        )
    )
    await sleep(0.01)
    task.cancel()
    return await task


class _FakeRunner:
    def __init__(
        self,
        *,
        available: bool = True,
        results: (
            Mapping[
                tuple[str, ...],
                Sequence[DockerCommandResult | BaseException],
            ]
            | None
        ) = None,
        delays: Mapping[tuple[str, ...], float] | None = None,
        image_digest: str | None = f"sha256:{_DIGEST}",
        image_os: str = "linux",
        image_architecture: str = "amd64",
        image_inspect_stdout: bytes | None = None,
        info_rootless: bool = False,
        info_remote: bool = False,
        logs_stdout: bytes = b"out\n",
        logs_stderr: bytes = b"",
        wait_exit_code: int = 0,
        copy_entries: Mapping[str, bytes] | None = None,
    ) -> None:
        self._available = available
        self._results = {
            prefix: list(items) for prefix, items in (results or {}).items()
        }
        self._delays = dict(delays or {})
        self._image_digest = image_digest
        self._image_os = image_os
        self._image_architecture = image_architecture
        self._image_inspect_stdout = image_inspect_stdout
        self._info_rootless = info_rootless
        self._info_remote = info_remote
        self._logs_stdout = logs_stdout
        self._logs_stderr = logs_stderr
        self._wait_exit_code = wait_exit_code
        self._copy_entries = dict(copy_entries or {"report.txt": b"summary"})
        self.calls: list[tuple[str, ...]] = []
        self.timeouts: list[float | None] = []

    def available(self) -> bool:
        return self._available

    async def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> DockerCommandResult:
        resolved_args = tuple(args)
        self.calls.append(resolved_args)
        self.timeouts.append(timeout_seconds)
        delay = self._delay_for(resolved_args)
        if delay:
            await sleep(delay)
        scripted = self._scripted_result(resolved_args)
        if scripted is not None:
            return scripted
        if resolved_args[0] == "version":
            return _ok(
                args=resolved_args,
                stdout=b'{"Client": {}, "Server": {}}',
            )
        if resolved_args[0] == "info":
            security = b'["name=rootless"]' if self._info_rootless else b"[]"
            operating_system = (
                b"Docker Desktop" if self._info_remote else b"Docker Engine"
            )
            return _ok(
                args=resolved_args,
                stdout=(
                    b'{"SecurityOptions": '
                    + security
                    + b', "OperatingSystem": "'
                    + operating_system
                    + b'"}'
                ),
            )
        if resolved_args[0:2] == ("image", "inspect"):
            return self._image_inspect(resolved_args)
        if resolved_args[0] in {
            "pull",
            "create",
            "start",
            "stop",
            "kill",
            "rm",
        }:
            return _ok(args=resolved_args)
        if resolved_args[0] == "logs":
            return _ok(
                args=resolved_args,
                stdout=self._logs_stdout,
                stderr=self._logs_stderr,
            )
        if resolved_args[0] == "wait":
            return _ok(
                args=resolved_args,
                stdout=f"{self._wait_exit_code}\n".encode(),
            )
        if resolved_args[0] == "inspect":
            return _ok(
                args=resolved_args,
                stdout=(
                    b'{"State": {"Status": "exited", "ExitCode": '
                    + str(self._wait_exit_code).encode()
                    + b"}}"
                ).replace(b'}}"', b"}}"),
            )
        if resolved_args[0] == "stats":
            return _ok(
                args=resolved_args,
                stdout=b'{"MemUsage": "2MiB / 1GiB", "PIDs": "3"}',
            )
        if resolved_args[0] == "cp":
            root = Path(resolved_args[-1])
            for relative_path, content in self._copy_entries.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            return _ok(args=resolved_args)
        return _ok(args=resolved_args)

    def _scripted_result(
        self,
        args: tuple[str, ...],
    ) -> DockerCommandResult | None:
        for prefix in sorted(self._results, key=len, reverse=True):
            if args[: len(prefix)] != prefix:
                continue
            queue = self._results[prefix]
            if not queue:
                return None
            result = queue.pop(0)
            if isinstance(result, BaseException):
                raise result
            return DockerCommandResult(
                args=args,
                return_code=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return None

    def _delay_for(self, args: tuple[str, ...]) -> float:
        for prefix, delay in self._delays.items():
            if args[: len(prefix)] == prefix:
                return delay
        return 0

    def _image_inspect(self, args: tuple[str, ...]) -> DockerCommandResult:
        reference = args[-1]
        if self._image_digest is None:
            return DockerCommandResult(
                args=args,
                return_code=1,
                stderr=b"image not found",
            )
        if self._image_inspect_stdout is not None:
            return _ok(args=args, stdout=self._image_inspect_stdout)
        return _ok(
            args=args,
            stdout=_image_inspect_json(
                reference,
                digest=self._image_digest,
                image_os=self._image_os,
                image_architecture=self._image_architecture,
            ),
        )


def _ok(
    *,
    args: Sequence[str] = (),
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> DockerCommandResult:
    return DockerCommandResult(
        args=tuple(args),
        return_code=0,
        stdout=stdout,
        stderr=stderr,
    )


def _image_inspect_json(
    reference: str,
    *,
    digest: str,
    image_os: str,
    image_architecture: str,
) -> bytes:
    name = reference.split("@", 1)[0]
    return (
        b'{"RepoDigests": ["'
        + name.encode("utf-8")
        + b"@"
        + digest.encode("ascii")
        + b'"], "Os": "'
        + image_os.encode("ascii")
        + b'", "Architecture": "'
        + image_architecture.encode("ascii")
        + b'"}'
    )


def _first_call(runner: _FakeRunner, command: str) -> tuple[str, ...]:
    for call in runner.calls:
        if call and call[0] == command:
            return call
    raise AssertionError(f"missing command {command}")


def _has_call(runner: _FakeRunner, prefix: tuple[str, ...]) -> bool:
    return any(call[: len(prefix)] == prefix for call in runner.calls)


def _created_container_name(runner: _FakeRunner) -> str:
    create_args = _first_call(runner, "create")
    name_flag = create_args.index("--name")
    return create_args[name_flag + 1]


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int,
        stdout: bytes,
        stderr: bytes,
    ) -> None:
        self.returncode = returncode
        self.stdout = _stream_reader(stdout)
        self.stderr = _stream_reader(stderr)

    async def wait(self) -> int:
        return self.returncode


class _CancellingProcess:
    def __init__(self) -> None:
        self.killed = False
        self.waited = False
        self.stdout = _stream_reader(b"")
        self.stderr = _stream_reader(b"")
        self._waits = 0

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self._waits += 1
        if self._waits == 1:
            raise CancelledError
        self.waited = True
        return 1


class _HangingProcess:
    def __init__(self) -> None:
        self.killed = False
        self.waited = False
        self.stdout = _stream_reader(b"")
        self.stderr = _stream_reader(b"")

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        if not self.killed:
            await sleep(1)
        self.waited = True
        return 137


class _RacyCleanupProcess:
    def __init__(self) -> None:
        self.kill_called = False
        self.wait_called = False
        self.stdout = _stream_reader(b"")
        self.stderr = _stream_reader(b"")

    def kill(self) -> None:
        self.kill_called = True
        raise ProcessLookupError

    async def wait(self) -> int:
        self.wait_called = True
        await sleep(1)
        return 137


def _stream_reader(content: bytes) -> StreamReader:
    reader = StreamReader()
    reader.feed_data(content)
    reader.feed_eof()
    return reader


def _run_plan(
    *,
    command: str = "echo",
    argv: Sequence[str] = ("echo", "ok"),
    backend: ContainerBackend = ContainerBackend.DOCKER,
    image: ContainerImagePolicy | None = None,
    pull_policy: ContainerPullPolicy = ContainerPullPolicy.NEVER,
    build_policy: ContainerBuildPolicy = ContainerBuildPolicy.DISABLED,
    environment_names: Sequence[str] = (),
    mounts: Sequence[ContainerMountDeclaration] | None = None,
    network: ContainerNetworkPolicy | None = None,
    resources: ContainerResourceLimits | None = None,
    secret_names: Sequence[str] = (),
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=backend,
        profile_name="docker-profile",
        image=image
        or ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
            platform="linux/amd64",
        ),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command=command,
            argv=argv,
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        mounts=mounts
        or (
            ContainerMountDeclaration(
                source=".",
                target="/workspace",
                mount_type=ContainerMountType.WORKSPACE,
                access=ContainerMountAccess.READ,
            ),
        ),
        environment_names=environment_names,
        secret_names=secret_names,
        network=network or ContainerNetworkPolicy(),
        resources=resources or ContainerResourceLimits(),
        policy_version="phase-docker",
    )


def _output_contract() -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=64,
        max_files=4,
    )


if __name__ == "__main__":
    main()
