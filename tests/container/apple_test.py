from asyncio import CancelledError, StreamReader
from asyncio import run as run_async
from collections.abc import Sequence
from os import environ
from pathlib import Path
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

from avalan.container import (
    AppleContainerBackend,
    AppleContainerCommandResult,
    AppleContainerSubprocessRunner,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendError,
    ContainerBackendOperation,
    ContainerBackendStream,
    ContainerBackendSupportLevel,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerExecutionScope,
    ContainerImagePolicy,
    ContainerLifecycleResources,
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
    ContainerToolRuntimeSettings,
    run_container_managed_lifecycle,
    select_container_backend,
    trusted_container_runtime_from_mapping,
    trusted_container_source,
)
from avalan.entities import ToolCallContext
from avalan.tool import Tool
from avalan.tool.shell import ShellToolSet, ShellToolSettings

_DIGEST = "a" * 64
_DIGEST_ALT = "b" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"
_APPLE_SHARED_WORKSPACE = "/tmp/avalan-apple-workspace"
_LIVE_E2E = environ.get("AVALAN_CONTAINER_APPLE_E2E") == "1"
_LIVE_E2E_IMAGE = environ.get("AVALAN_CONTAINER_APPLE_E2E_IMAGE")


class AppleContainerBackendTest(TestCase):
    def test_probe_reports_opt_in_capabilities(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=0),)
            )
        )

        probe = run_async(backend.probe())
        without_opt_in = select_container_backend(
            _run_plan(),
            (probe,),
        )
        selection = select_container_backend(
            _run_plan(),
            (probe,),
            opt_in_backends=(ContainerBackend.APPLE_CONTAINER,),
        )

        self.assertTrue(probe.ok)
        self.assertFalse(without_opt_in.ok)
        self.assertTrue(selection.ok)
        self.assertEqual(selection.backend, ContainerBackend.APPLE_CONTAINER)
        self.assertIsInstance(probe.capabilities, ContainerBackendCapabilities)
        assert probe.capabilities is not None
        self.assertEqual(
            probe.capabilities.support_level,
            ContainerBackendSupportLevel.OPT_IN,
        )
        self.assertTrue(probe.capabilities.resource_limits)
        self.assertFalse(probe.capabilities.streaming_attach)
        self.assertFalse(probe.capabilities.stats)
        self.assertTrue(probe.capabilities.lifecycle_normalization)

    def test_probe_missing_cli_fails_closed(self) -> None:
        backend = AppleContainerBackend(_FakeRunner(available=False))

        probe = run_async(backend.probe())

        self.assertFalse(probe.ok)
        self.assertFalse(probe.available)
        self.assertIsNone(probe.capabilities)
        self.assertEqual(
            probe.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            probe.diagnostics[0].operation,
            ContainerBackendOperation.PROBE,
        )

    def test_probe_service_failure_fails_closed(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"XPC connection invalid",
                    ),
                )
            )
        )

        probe = run_async(backend.probe())

        self.assertFalse(probe.available)
        self.assertEqual(
            probe.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertIn("XPC connection invalid", probe.diagnostics[0].message)

    def test_probe_file_not_found_fails_closed(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(results=(FileNotFoundError("container"),))
        )

        probe = run_async(backend.probe())

        self.assertFalse(probe.available)
        self.assertEqual(
            probe.diagnostics[0].code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_subprocess_runner_uses_container_cli(self) -> None:
        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _FakeProcess:
            self.assertEqual(args, ("container", "system", "version"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            return _FakeProcess(returncode=3, stdout=b"out", stderr=b"err")

        with patch(
            "avalan.container.apple.which",
            return_value="/usr/local/bin/container",
        ):
            runner = AppleContainerSubprocessRunner()
            self.assertTrue(runner.available())
        with patch(
            "avalan.container.apple.create_subprocess_exec",
            create_process,
        ):
            result = run_async(runner.run(("system", "version")))

        self.assertEqual(result.return_code, 3)
        self.assertEqual(result.stdout, b"out")
        self.assertEqual(result.stderr, b"err")

    def test_subprocess_runner_bounds_captured_output(self) -> None:
        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _FakeProcess:
            self.assertEqual(args, ("container", "run"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            return _FakeProcess(
                returncode=0,
                stdout=b"x" * 1048580,
                stderr=b"",
            )

        runner = AppleContainerSubprocessRunner()

        with patch(
            "avalan.container.apple.create_subprocess_exec",
            create_process,
        ):
            result = run_async(runner.run(("run",)))

        self.assertEqual(len(result.stdout), 1048576)
        self.assertEqual(result.stdout, b"x" * 1048576)

    def test_subprocess_runner_kills_process_on_cancel(self) -> None:
        processes: list[_CancellingProcess] = []

        async def create_process(
            *args: str,
            stdout: object,
            stderr: object,
        ) -> _CancellingProcess:
            self.assertEqual(args, ("container", "run"))
            self.assertIsNotNone(stdout)
            self.assertIsNotNone(stderr)
            process = _CancellingProcess()
            processes.append(process)
            return process

        runner = AppleContainerSubprocessRunner()

        with patch(
            "avalan.container.apple.create_subprocess_exec",
            create_process,
        ):
            with self.assertRaises(CancelledError):
                run_async(runner.run(("run",)))

        process = processes[0]
        self.assertTrue(process.killed)
        self.assertTrue(process.waited)

    def test_create_start_stream_wait_inspect_stats_and_remove(self) -> None:
        runner = _FakeRunner(
            results=(
                AppleContainerCommandResult(args=(), return_code=0),
                AppleContainerCommandResult(
                    args=(),
                    return_code=7,
                    stdout=b"out\n",
                    stderr=b"err\n",
                ),
                AppleContainerCommandResult(
                    args=(),
                    return_code=0,
                    stdout=b'[{"cpuNanos": 1, "memoryBytes": 2, "pids": 3}]',
                ),
                AppleContainerCommandResult(
                    args=(),
                    return_code=0,
                    stdout=b'[{"status": "stopped", "exitCode": 7}]',
                ),
                AppleContainerCommandResult(args=(), return_code=0),
            )
        )
        backend = AppleContainerBackend(runner)
        plan = _run_plan(
            resources=ContainerResourceLimits(
                cpu_count=2,
                memory_bytes=2_097_153,
                pids=64,
                timeout_seconds=30,
            )
        )

        image = run_async(backend.resolve_image(plan))
        container = run_async(backend.create(plan))
        attach = run_async(backend.attach(container))
        start = run_async(backend.start(container))
        chunks = run_async(backend.stream(container))
        stats = run_async(backend.stats(container))
        wait = run_async(backend.wait(container))
        inspection = run_async(backend.inspect(container))
        remove = run_async(backend.remove(container))

        self.assertTrue(image.ok)
        self.assertTrue(attach.ok)
        self.assertTrue(start.ok)
        self.assertEqual(start.metadata["exit_code"], "7")
        self.assertEqual(wait.exit_code, 7)
        self.assertEqual(inspection.status, "stopped")
        self.assertEqual(inspection.exit_code, 7)
        self.assertEqual(stats[0].memory_bytes, 2)
        self.assertTrue(remove.ok)
        self.assertEqual(
            [(chunk.stream, chunk.content) for chunk in chunks],
            [
                (ContainerBackendStream.STDOUT, b"out\n"),
                (ContainerBackendStream.STDERR, b"err\n"),
            ],
        )
        create_args = _first_call(runner, "create")
        self.assertEqual(create_args[0], "create")
        self.assertIn("--network", create_args)
        self.assertIn("none", create_args)
        self.assertIn("--no-dns", create_args)
        self.assertIn("--read-only", create_args)
        self.assertIn("--cap-drop", create_args)
        self.assertIn("--user", create_args)
        self.assertIn("--cpus", create_args)
        self.assertIn("2", create_args)
        self.assertIn("--memory", create_args)
        self.assertIn("3M", create_args)
        self.assertIn("--ulimit", create_args)
        self.assertIn("nproc=64", create_args)
        self.assertIn("--mount", create_args)
        mount_index = create_args.index("--mount") + 1
        self.assertIn("target=/workspace", create_args[mount_index])
        self.assertIn("readonly", create_args[mount_index])
        self.assertTrue(_has_argument_prefix(create_args, "avalan-verified:"))
        self.assertTrue(_has_call(runner, ("start", "--attach")))
        self.assertTrue(_has_call(runner, ("stats", "--format", "json")))
        self.assertTrue(_has_call(runner, ("inspect",)))
        self.assertTrue(_has_call(runner, ("rm",)))
        self.assertTrue(_has_call(runner, ("image", "rm")))

    def test_create_preserves_explicit_local_tag_for_digest_reference(
        self,
    ) -> None:
        runner = _FakeRunner(
            results=(AppleContainerCommandResult(args=(), return_code=0),)
        )
        backend = AppleContainerBackend(runner)

        run_async(
            backend.create(
                _run_plan(
                    image=ContainerImagePolicy(
                        reference=(
                            f"ghcr.io/example/shell-tools:v1@sha256:{_DIGEST}"
                        ),
                        platform="linux/arm64",
                    )
                )
            )
        )

        self.assertTrue(
            _has_call(
                runner,
                ("image", "tag", "ghcr.io/example/shell-tools:v1"),
            )
        )
        self.assertTrue(
            _has_argument_prefix(
                _first_call(runner, "create"), "avalan-verified:"
            )
        )

    def test_create_rejects_local_digest_mismatch(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(image_digest=f"sha256:{_DIGEST_ALT}")
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "digest does not match",
            raised.exception.diagnostic.message,
        )

    def test_create_rejects_missing_local_digest(self) -> None:
        backend = AppleContainerBackend(_FakeRunner(image_digest=None))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "unavailable locally",
            raised.exception.diagnostic.message,
        )

    def test_create_reports_image_inspect_backend_failure(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(image_inspect_error=FileNotFoundError("container"))
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_create_rejects_missing_image_digest_metadata(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(image_inspect_stdout=b'[{"configuration": {}}]')
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "digest is unavailable", raised.exception.diagnostic.message
        )

    def test_create_normalizes_raw_image_id_digest(self) -> None:
        runner = _FakeRunner(
            image_inspect_stdout=(
                b'[{"id": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                b'AAAAAAAAAAAAAAAA"}]'
            )
        )
        backend = AppleContainerBackend(runner)

        container = run_async(backend.create(_run_plan()))

        self.assertEqual(container.backend, ContainerBackend.APPLE_CONTAINER)

    def test_create_rejects_invalid_image_digest_metadata(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(image_inspect_stdout=b'[{"id": "not-a-digest"}]')
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )

    def test_create_reports_verified_tag_inspect_backend_failure(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(tag_image_inspect_error=FileNotFoundError("container"))
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_create_rejects_unavailable_verified_tag(self) -> None:
        backend = AppleContainerBackend(_FakeRunner(tag_image_digest=None))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "unavailable locally",
            raised.exception.diagnostic.message,
        )

    def test_create_rejects_missing_verified_tag_digest_metadata(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(tag_image_inspect_stdout=b'[{"configuration": {}}]')
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "digest is unavailable", raised.exception.diagnostic.message
        )

    def test_create_rejects_verified_tag_digest_mismatch(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(tag_image_digest=f"sha256:{_DIGEST_ALT}")
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )
        self.assertIn(
            "digest does not match",
            raised.exception.diagnostic.message,
        )

    def test_managed_lifecycle_runs_apple_backend_e2e(self) -> None:
        runner = _FakeRunner(
            results=(
                AppleContainerCommandResult(args=(), return_code=0),
                AppleContainerCommandResult(
                    args=(),
                    return_code=0,
                    stdout=b"ok\n",
                ),
                AppleContainerCommandResult(
                    args=(),
                    return_code=0,
                    stdout=b'{"cpuNanos": 1, "memoryBytes": 2, "pids": 3}',
                ),
                AppleContainerCommandResult(
                    args=(),
                    return_code=0,
                    stdout=b'{"status": "stopped", "exitCode": 0}',
                ),
                AppleContainerCommandResult(args=(), return_code=0),
            )
        )
        backend = AppleContainerBackend(runner)

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(
                    resources=ContainerResourceLimits(timeout_seconds=5)
                ),
                lifecycle_resources=ContainerLifecycleResources(),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertTrue(result.cleanup_completed)
        self.assertFalse(result.cleanup_uncertain)
        self.assertEqual(result.stream.stdout_chunks[0].content, b"ok\n")
        self.assertEqual(result.stats[0].pids, 3)
        self.assertTrue(_has_call(runner, ("image", "tag")))
        self.assertTrue(_has_call(runner, ("create",)))
        self.assertTrue(_has_call(runner, ("start", "--attach")))
        self.assertTrue(_has_call(runner, ("stats", "--format", "json")))
        self.assertTrue(_has_call(runner, ("inspect",)))
        self.assertTrue(_has_call(runner, ("rm",)))
        self.assertTrue(_has_call(runner, ("image", "rm")))

    @skipUnless(
        _LIVE_E2E,
        "set AVALAN_CONTAINER_APPLE_E2E=1 to run Apple container e2e",
    )
    def test_live_apple_container_backend_e2e(self) -> None:
        if not _LIVE_E2E_IMAGE:
            self.skipTest("set AVALAN_CONTAINER_APPLE_E2E_IMAGE")
        image_reference = _LIVE_E2E_IMAGE
        assert image_reference is not None
        backend = AppleContainerBackend()
        plan = _run_plan(
            command="sh",
            argv=("sh", "-c", "test -r pyproject.toml && printf live-ok"),
            image=ContainerImagePolicy(
                reference=image_reference,
                platform="linux/arm64",
            ),
            mounts=(
                ContainerMountDeclaration(
                    source=Path.cwd().as_posix(),
                    target="/workspace",
                    mount_type=ContainerMountType.WORKSPACE,
                    access=ContainerMountAccess.READ,
                ),
            ),
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                plan,
                lifecycle_resources=ContainerLifecycleResources(),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(result.stream.stdout_chunks[0].content, b"live-ok")

    @skipUnless(
        _LIVE_E2E,
        "set AVALAN_CONTAINER_APPLE_E2E=1 to run Apple container e2e",
    )
    def test_live_apple_shell_toolset_e2e(self) -> None:
        if not _LIVE_E2E_IMAGE:
            self.skipTest("set AVALAN_CONTAINER_APPLE_E2E_IMAGE")
        image_reference = _LIVE_E2E_IMAGE
        assert image_reference is not None
        runtime = trusted_container_runtime_from_mapping(
            {
                "backend": "apple-container",
                "default_profile": "workspace-readonly",
                "profiles": {
                    "workspace-readonly": {
                        "image": image_reference,
                        "workspace_root": ".",
                        "container_workspace": "/workspace",
                        "pull_policy": "never",
                        "platform": "linux/arm64",
                        "network": {"mode": "none"},
                        "resources": {"timeout_seconds": 30},
                    },
                },
            },
            source=trusted_container_source("sdk"),
        )
        runtime = ContainerToolRuntimeSettings(
            effective_settings=runtime.effective_settings,
            backend=AppleContainerBackend(),
            opt_in_backends=(ContainerBackend.APPLE_CONTAINER,),
        )
        toolset = ShellToolSet(
            settings=ShellToolSettings(
                backend="container",
                workspace_root=".",
                cwd=".",
            ),
            container_runtime=runtime,
        )
        output = run_async(
            _tool_by_name(toolset, "ls")(
                ".",
                context=ToolCallContext(),
            )
        )

        self.assertIn("pyproject.toml", output)

    def test_pull_success_and_pull_never_denial(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=0),)
            )
        )
        plan = _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)
        image = run_async(backend.resolve_image(plan))

        pull = run_async(backend.pull_image(plan, image))
        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.pull_image(_run_plan(), image))

        self.assertTrue(pull.ok)
        self.assertEqual(pull.metadata["return_code"], "0")
        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.PULL_DENIED,
        )

    def test_pull_reports_post_pull_digest_verification_failure(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=0),),
                image_digest=None,
            )
        )
        plan = _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)
        image = run_async(backend.resolve_image(plan))

        pull = run_async(backend.pull_image(plan, image))

        self.assertFalse(pull.ok)
        self.assertEqual(
            pull.diagnostics[0].code,
            ContainerBackendDiagnosticCode.IMAGE_DENIED,
        )

    def test_start_backend_failure_raises_stable_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"XPC connection invalid",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.start(container))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.START_FAILED,
        )
        self.assertEqual(
            raised.exception.diagnostic.operation,
            ContainerBackendOperation.START,
        )

    def test_start_preserves_command_failure_that_mentions_not_found(
        self,
    ) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=127,
                        stderr=b"sh: missing-binary: not found",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        start = run_async(backend.start(container))
        wait = run_async(backend.wait(container))
        chunks = run_async(backend.stream(container))

        self.assertTrue(start.ok)
        self.assertEqual(wait.exit_code, 127)
        self.assertEqual(chunks[0].stream, ContainerBackendStream.STDERR)
        self.assertIn(b"not found", chunks[0].content)

    def test_start_file_not_found_raises_stable_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    FileNotFoundError("container"),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.start(container))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_wait_before_start_raises_stable_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=0),)
            )
        )
        container = run_async(backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.wait(container))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.WAIT_FAILED,
        )

    def test_resolve_image_rejects_unsupported_platform(self) -> None:
        backend = AppleContainerBackend(_FakeRunner())

        image = run_async(
            backend.resolve_image(
                _run_plan(image=ContainerImagePolicy(reference=_IMAGE))
            )
        )

        self.assertFalse(image.ok)
        self.assertEqual(
            image.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertIn("linux/arm64", image.diagnostics[0].message)

    def test_resolve_image_reports_all_policy_restrictions(self) -> None:
        backend = AppleContainerBackend(_FakeRunner())
        workspace_mount = ContainerMountDeclaration(
            source=_APPLE_SHARED_WORKSPACE,
            target="/workspace",
            mount_type=ContainerMountType.WORKSPACE,
            access=ContainerMountAccess.READ,
        )
        object.__setattr__(
            workspace_mount,
            "access",
            ContainerMountAccess.WRITE,
        )

        image = run_async(
            backend.resolve_image(
                _run_plan(
                    backend=ContainerBackend.DOCKER,
                    build_policy=ContainerBuildPolicy.TRUSTED_ONLY,
                    environment_names=("PATH",),
                    mounts=(
                        workspace_mount,
                        ContainerMountDeclaration(
                            source="/Applications",
                            target="/readonly",
                            mount_type=ContainerMountType.WORKSPACE,
                        ),
                        ContainerMountDeclaration(
                            source=_APPLE_SHARED_WORKSPACE,
                            target="/scratch",
                            mount_type=ContainerMountType.SCRATCH,
                            access=ContainerMountAccess.WRITE,
                        ),
                        ContainerMountDeclaration(
                            target="/out",
                            mount_type=ContainerMountType.OUTPUT,
                            access=ContainerMountAccess.WRITE,
                        ),
                    ),
                    network=ContainerNetworkPolicy(
                        mode=ContainerNetworkMode.ALLOWLIST,
                        egress_allowlist=("example.test",),
                    ),
                    resources=ContainerResourceLimits(pids=64),
                    secret_names=("TOKEN",),
                )
            )
        )
        messages = tuple(
            diagnostic.message for diagnostic in image.diagnostics
        )

        self.assertIn("plan backend is not apple", messages)
        self.assertIn(
            "Apple container backend does not build shell images",
            messages,
        )
        self.assertIn("network mode allowlist is not supported", messages)
        self.assertIn("network egress allowlists are not supported", messages)
        self.assertIn("environment inheritance is not supported", messages)
        self.assertIn("secret injection is not supported", messages)
        self.assertIn("workspace mount must be read-only", messages)
        self.assertIn(
            "mount source is outside Apple shared mount prefixes",
            messages,
        )
        self.assertIn("mount type scratch is not supported", messages)
        self.assertIn("mount source is required", messages)

    def test_create_rejects_network_policy(self) -> None:
        backend = AppleContainerBackend(_FakeRunner())

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(
                backend.create(
                    _run_plan(
                        network=ContainerNetworkPolicy(
                            mode=ContainerNetworkMode.FULL,
                        )
                    )
                )
            )

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            raised.exception.diagnostic.operation,
            ContainerBackendOperation.CREATE,
        )

    def test_create_file_not_found_raises_stable_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(results=(FileNotFoundError("container"),))
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_create_command_failure_without_detail_uses_exit_code(
        self,
    ) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=2),)
            )
        )

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.create(_run_plan()))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CREATE_FAILED,
        )
        self.assertIn("exit code 2", raised.exception.diagnostic.message)

    def test_pull_failure_and_build_denial_are_stable(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"registry unavailable",
                    ),
                )
            )
        )
        plan = _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING)
        image = run_async(backend.resolve_image(plan))

        with self.assertRaises(ContainerBackendError) as pull_error:
            run_async(backend.pull_image(plan, image))
        with self.assertRaises(ContainerBackendError) as build_error:
            run_async(backend.build_image(_run_plan()))

        self.assertEqual(
            pull_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.PULL_FAILED,
        )
        self.assertEqual(
            build_error.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BUILD_DENIED,
        )

    def test_copy_outputs_rejects_without_runtime_authority(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(AppleContainerCommandResult(args=(), return_code=0),)
            )
        )
        container = run_async(backend.create(_run_plan()))

        output = run_async(
            backend.copy_outputs(
                container,
                ContainerOutputContract(
                    contract_type=ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                ),
            )
        )

        self.assertEqual(output.decision, ContainerOutputDecisionType.REJECT)
        self.assertTrue(output.diagnostics)

    def test_state_mismatch_and_unknown_state_fail_closed(self) -> None:
        backend = AppleContainerBackend(_FakeRunner())
        docker_container = ContainerBackendContainer(
            container_id="docker-id",
            backend=ContainerBackend.DOCKER,
            plan_fingerprint="fingerprint",
        )
        missing_container = ContainerBackendContainer(
            container_id="missing-id",
            backend=ContainerBackend.APPLE_CONTAINER,
            plan_fingerprint="fingerprint",
        )

        with self.assertRaises(ContainerBackendError) as wrong_backend:
            run_async(backend.attach(docker_container))
        with self.assertRaises(ContainerBackendError) as missing_state:
            run_async(backend.attach(missing_container))
        with self.assertRaises(ContainerBackendError) as wrong_inspect:
            run_async(backend.inspect(docker_container))
        with self.assertRaises(ContainerBackendError) as missing_inspect:
            run_async(backend.inspect(missing_container))

        self.assertEqual(
            wrong_backend.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            missing_state.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            wrong_inspect.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            missing_inspect.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_stats_backend_failure_raises(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"container-apiserver unavailable",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.stats(container))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(
            raised.exception.diagnostic.operation,
            ContainerBackendOperation.STATS,
        )

    def test_stats_file_not_found_and_non_backend_failures(self) -> None:
        file_missing = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    FileNotFoundError("container"),
                )
            )
        )
        non_backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"container is stopped",
                    ),
                )
            )
        )
        file_missing_container = run_async(file_missing.create(_run_plan()))
        non_backend_container = run_async(non_backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(file_missing.stats(file_missing_container))
        stats = run_async(non_backend.stats(non_backend_container))

        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(stats, ())

    def test_stats_handles_json_variants(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=(
                            b'{"cpu": {"nanos": 4}, "memory": {"bytes": 5},'
                            b' "pidCount": 6}'
                        ),
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b"not json",
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b"[1]",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        nested = run_async(backend.stats(container))
        invalid = run_async(backend.stats(container))
        skipped = run_async(backend.stats(container))

        self.assertEqual(nested[0].cpu_nanos, 4)
        self.assertEqual(nested[0].memory_bytes, 5)
        self.assertEqual(nested[0].pids, 6)
        self.assertEqual(invalid, ())
        self.assertEqual(skipped, ())

    def test_inspect_handles_json_variants_and_fallback_exit_code(
        self,
    ) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=5,
                        stdout=b"out",
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b"{}",
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b'{"state": "done", "exit_code": 0}',
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b"not json",
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b"[]",
                    ),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=0,
                        stdout=b'{"State": "bad"}',
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))
        run_async(backend.start(container))

        fallback = run_async(backend.inspect(container))
        mapping = run_async(backend.inspect(container))
        invalid = run_async(backend.inspect(container))
        empty = run_async(backend.inspect(container))
        nested_mismatch = run_async(backend.inspect(container))

        self.assertEqual(fallback.status, "unknown")
        self.assertEqual(fallback.exit_code, 5)
        self.assertEqual(mapping.status, "done")
        self.assertEqual(mapping.exit_code, 0)
        self.assertEqual(invalid.status, "unknown")
        self.assertEqual(empty.status, "unknown")
        self.assertEqual(nested_mismatch.status, "unknown")

    def test_inspect_failure_raises_stable_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"api server down",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        with self.assertRaises(ContainerBackendError) as raised:
            run_async(backend.inspect(container))

        self.assertEqual(
            raised.exception.diagnostic.operation,
            ContainerBackendOperation.INSPECT,
        )

    def test_cleanup_command_backend_failure_is_diagnostic(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(
                        args=(),
                        return_code=1,
                        stderr=b"no such container",
                    ),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))

        result = run_async(backend.remove(container))

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code,
            ContainerBackendDiagnosticCode.CLEANUP_FAILED,
        )

    def test_stop_kill_cleanup_and_cleanup_file_not_found(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(args=(), return_code=0),
                    AppleContainerCommandResult(args=(), return_code=0),
                )
            )
        )
        missing = AppleContainerBackend(
            _FakeRunner(
                results=(
                    AppleContainerCommandResult(args=(), return_code=0),
                    FileNotFoundError("container"),
                )
            )
        )
        container = run_async(backend.create(_run_plan()))
        missing_container = run_async(missing.create(_run_plan()))

        stop = run_async(backend.stop(container))
        kill = run_async(backend.kill(container))
        cleanup = run_async(backend.cleanup(container))
        with self.assertRaises(ContainerBackendError) as raised:
            run_async(missing.remove(missing_container))

        self.assertTrue(stop.ok)
        self.assertTrue(kill.ok)
        self.assertTrue(cleanup.ok)
        self.assertEqual(
            raised.exception.diagnostic.code,
            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_cleanup_after_remove_is_idempotent(self) -> None:
        runner = _FakeRunner(
            results=(
                AppleContainerCommandResult(args=(), return_code=0),
                AppleContainerCommandResult(args=(), return_code=0),
            )
        )
        backend = AppleContainerBackend(runner)
        container = run_async(backend.create(_run_plan()))

        remove = run_async(backend.remove(container))
        cleanup = run_async(backend.cleanup(container))

        self.assertTrue(remove.ok)
        self.assertTrue(cleanup.ok)
        self.assertEqual(
            tuple(call[0] for call in runner.calls),
            ("image", "image", "image", "image", "create", "rm", "image"),
        )

    def test_remove_ignores_image_tag_delete_backend_failure(self) -> None:
        backend = AppleContainerBackend(
            _FakeRunner(image_rm_error=FileNotFoundError("container"))
        )
        container = run_async(backend.create(_run_plan()))

        result = run_async(backend.remove(container))

        self.assertTrue(result.ok)


class _FakeRunner:
    def __init__(
        self,
        *,
        available: bool = True,
        results: Sequence[AppleContainerCommandResult | BaseException] = (),
        image_digest: str | None = f"sha256:{_DIGEST}",
        image_inspect_stdout: bytes | None = None,
        image_inspect_error: BaseException | None = None,
        tag_image_digest: str | None = f"sha256:{_DIGEST}",
        tag_image_inspect_stdout: bytes | None = None,
        tag_image_inspect_error: BaseException | None = None,
        image_rm_error: BaseException | None = None,
        accept_digest_reference: bool = False,
    ) -> None:
        self._available = available
        self._results = list(results)
        self._image_digest = image_digest
        self._image_inspect_stdout = image_inspect_stdout
        self._image_inspect_error = image_inspect_error
        self._tag_image_digest = tag_image_digest
        self._tag_image_inspect_stdout = tag_image_inspect_stdout
        self._tag_image_inspect_error = tag_image_inspect_error
        self._image_rm_error = image_rm_error
        self._accept_digest_reference = accept_digest_reference
        self.calls: list[tuple[str, ...]] = []

    def available(self) -> bool:
        return self._available

    async def run(
        self,
        args: Sequence[str],
    ) -> AppleContainerCommandResult:
        resolved_args = tuple(args)
        self.calls.append(resolved_args)
        if resolved_args[0:2] == ("image", "inspect"):
            if resolved_args[2].startswith("avalan-verified:"):
                return self._image_inspect_result(
                    resolved_args,
                    digest=self._tag_image_digest,
                    stdout=self._tag_image_inspect_stdout,
                    error=self._tag_image_inspect_error,
                )
            if self._image_inspect_error is not None:
                raise self._image_inspect_error
            if (
                "@sha256:" in resolved_args[2]
                and not self._accept_digest_reference
            ):
                return AppleContainerCommandResult(
                    args=resolved_args,
                    return_code=1,
                    stderr=b"image not found",
                )
            if self._image_digest is None:
                return AppleContainerCommandResult(
                    args=resolved_args,
                    return_code=1,
                    stderr=b"image not found",
                )
            if self._image_inspect_stdout is not None:
                return AppleContainerCommandResult(
                    args=resolved_args,
                    return_code=0,
                    stdout=self._image_inspect_stdout,
                )
            return AppleContainerCommandResult(
                args=resolved_args,
                return_code=0,
                stdout=_image_inspect_json(self._image_digest),
            )
        if resolved_args[0:2] in {("image", "tag"), ("image", "rm")}:
            if resolved_args[0:2] == ("image", "rm") and self._image_rm_error:
                raise self._image_rm_error
            return AppleContainerCommandResult(
                args=resolved_args,
                return_code=0,
            )
        if not self._results:
            return AppleContainerCommandResult(
                args=resolved_args,
                return_code=0,
            )
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return AppleContainerCommandResult(
            args=resolved_args,
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _image_inspect_result(
        self,
        args: tuple[str, ...],
        *,
        digest: str | None,
        stdout: bytes | None,
        error: BaseException | None,
    ) -> AppleContainerCommandResult:
        if error is not None:
            raise error
        if digest is None:
            return AppleContainerCommandResult(
                args=args,
                return_code=1,
                stderr=b"image not found",
            )
        if stdout is not None:
            return AppleContainerCommandResult(
                args=args,
                return_code=0,
                stdout=stdout,
            )
        return AppleContainerCommandResult(
            args=args,
            return_code=0,
            stdout=_image_inspect_json(digest),
        )


def _image_inspect_json(digest: str) -> bytes:
    return (
        b'[{"configuration": {"descriptor": {"digest": "'
        + digest.encode("ascii")
        + b'"}}}]'
    )


def _first_call(runner: "_FakeRunner", command: str) -> tuple[str, ...]:
    for call in runner.calls:
        if call and call[0] == command:
            return call
    raise AssertionError(f"missing command {command}")


def _has_call(runner: "_FakeRunner", prefix: tuple[str, ...]) -> bool:
    return any(call[: len(prefix)] == prefix for call in runner.calls)


def _has_argument_prefix(call: tuple[str, ...], prefix: str) -> bool:
    return any(argument.startswith(prefix) for argument in call)


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


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


def _stream_reader(content: bytes) -> StreamReader:
    reader = StreamReader()
    reader.feed_data(content)
    reader.feed_eof()
    return reader


def _run_plan(
    *,
    command: str = "echo",
    argv: Sequence[str] = ("echo", "ok"),
    backend: ContainerBackend = ContainerBackend.APPLE_CONTAINER,
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
        profile_name="apple-profile",
        image=image
        or ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
            platform="linux/arm64",
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
                source=_APPLE_SHARED_WORKSPACE,
                target="/workspace",
                mount_type=ContainerMountType.WORKSPACE,
                access=ContainerMountAccess.READ,
            ),
        ),
        environment_names=environment_names,
        secret_names=secret_names,
        network=network or ContainerNetworkPolicy(),
        resources=resources or ContainerResourceLimits(),
        policy_version="phase-apple",
    )


if __name__ == "__main__":
    main()
