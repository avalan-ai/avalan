from asyncio import CancelledError, create_task
from asyncio import run as run_async
from asyncio import sleep as async_sleep
from builtins import __import__ as builtin_import
from collections.abc import Callable, Mapping, Sequence
from importlib import reload
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import TestCase, main
from unittest.mock import patch

from avalan.isolation import (
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    SandboxEffectiveSettings,
    trusted_isolation_source,
)
from avalan.sandbox import (
    BubblewrapSandboxBackend,
    SandboxBackend,
    SandboxBackendDiagnosticCode,
    SandboxBackendOperation,
    SandboxExecutionPlan,
    SandboxNetworkMode,
    SandboxPlanRequest,
    SandboxPlanRequestKind,
    SandboxResultStatus,
    SandboxSubprocessRequest,
    SandboxSubprocessResult,
    SeatbeltSandboxBackend,
    generate_bubblewrap_arguments,
    generate_seatbelt_profile,
)
from avalan.sandbox import backend as backend_module
from avalan.sandbox.backend import (
    _argument_is_existing_absolute_path,
    _cleanup_with_budget,
    _collect_real_outputs,
    _default_cleanup_handler,
    _default_command_runner,
    _process_limit_preexec,
    _replace_path_prefix,
    _system_prefix_resolves_equivalently,
)


class SandboxPhase6Test(TestCase):
    def test_seatbelt_profile_generation_is_deterministic(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            plan = _plan(
                "seatbelt",
                roots,
                network="none",
                pids=None,
                child_processes="deny",
                inherited_fds="stdio",
            )

            profile = generate_seatbelt_profile(plan)
            equivalent = generate_seatbelt_profile(plan)
            profile_lines = profile.splitlines()

            self.assertEqual(profile, equivalent)
            self.assertIn("(version 1)", profile)
            self.assertIn("(deny default)", profile)
            self.assertIn("(allow sysctl-read)", profile)
            self.assertNotIn("(allow file-read-data)", profile_lines)
            for path in (
                "/System/Library/dyld",
                "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld",
                "/private/var/db/dyld",
                "/usr/lib/dyld",
            ):
                self.assertIn(
                    f'(allow file-read-data (subpath "{path}"))',
                    profile,
                )
            self.assertEqual(
                [
                    line
                    for line in profile_lines
                    if line.startswith("(allow file-read")
                    and line != "(allow file-read-metadata)"
                    and "(subpath " not in line
                ],
                [],
            )
            self.assertIn(
                f'(allow file-read* (subpath "{roots["workspace"]}"))',
                profile,
            )
            self.assertIn(
                f'(allow file-write* (subpath "{roots["output_dir"]}"))',
                profile,
            )
            self.assertIn(
                f'(deny file-read* (subpath "{roots["deny"]}"))',
                profile,
            )
            self.assertIn("(deny network*)", profile)
            self.assertIn("(deny process-fork)", profile)
            with self.assertRaises(AssertionError):
                generate_seatbelt_profile(cast(SandboxExecutionPlan, "raw"))

    def test_seatbelt_profile_adds_safe_system_prefix_aliases(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            roots["output_root"] = "/tmp"
            roots["output_dir"] = "/tmp/avalan-seatbelt-session"

            profile = generate_seatbelt_profile(
                _plan("seatbelt", roots, network="none")
            )

            self.assertIn(
                '(allow file-write* (subpath "/tmp/avalan-seatbelt-session"))',
                profile,
            )
            resolved_output_dir = (
                Path("/tmp/avalan-seatbelt-session").resolve().as_posix()
            )
            if _system_prefix_resolves_equivalently(
                "/tmp/avalan-seatbelt-session",
                resolved_output_dir,
            ):
                self.assertIn(
                    f'(allow file-write* (subpath "{resolved_output_dir}"))',
                    profile,
                )

    def test_bubblewrap_argument_generation_is_policy_derived(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            plan = _plan("bubblewrap", roots, network="none", pids=16)

            argv = generate_bubblewrap_arguments(
                plan,
                sandbox_executable="/usr/bin/bwrap",
                private_runtime_dir="/tmp/avalan-runtime",
            )

            self.assertEqual(argv[0], "/usr/bin/bwrap")
            self.assertIn("--clearenv", argv)
            self.assertIn("--unshare-net", argv)
            self.assertNotIn("--share-net", argv)
            self.assertIn("--proc", argv)
            self.assertIn("--dev", argv)
            self.assertTrue(
                _contains_triplet(argv, "--ro-bind", "/bin", "/bin")
            )
            self.assertTrue(
                _contains_triplet(
                    argv,
                    "--ro-bind",
                    roots["workspace"],
                    roots["workspace"],
                )
            )
            self.assertTrue(
                _contains_triplet(
                    argv,
                    "--bind",
                    roots["output_dir"],
                    roots["output_dir"],
                )
            )
            self.assertTrue(
                _contains_triplet(
                    argv,
                    "--ro-bind",
                    "/tmp/avalan-runtime/deny-empty",
                    roots["deny"],
                )
            )
            self.assertIn("--setenv", argv)
            self.assertEqual(argv[-3:], ("/bin/sh", "-lc", "echo ok"))

    def test_subprocess_request_validation_and_dict(self) -> None:
        with self.assertRaises(AssertionError):
            SandboxSubprocessRequest(
                operation=SandboxBackendOperation.START,
                label="invalid_process_limit",
                argv=("/bin/sh",),
                process_limit=0,
            )
        with self.assertRaises(AssertionError):
            SandboxSubprocessRequest(
                operation=SandboxBackendOperation.START,
                label="invalid_pass_fd_type",
                argv=("/bin/sh",),
                pass_fds=cast(Sequence[int], ("bad",)),
            )
        with self.assertRaises(AssertionError):
            SandboxSubprocessRequest(
                operation=SandboxBackendOperation.START,
                label="invalid_pass_fd_value",
                argv=("/bin/sh",),
                pass_fds=(-1,),
            )

        request = SandboxSubprocessRequest(
            operation=SandboxBackendOperation.START,
            label="dict_request",
            argv=("/bin/sh", "-c", "true"),
            environment={"PATH": "/bin"},
            cwd="/tmp",
            timeout_seconds=1,
            stdout_limit_bytes=10,
            stderr_limit_bytes=11,
            process_limit=2,
            close_fds=False,
            pass_fds=(1,),
        )

        self.assertEqual(
            request.to_dict(),
            {
                "operation": "start",
                "label": "dict_request",
                "argv": ["/bin/sh", "-c", "true"],
                "environment": {"PATH": "/bin"},
                "cwd": "/tmp",
                "timeout_seconds": 1,
                "stdout_limit_bytes": 10,
                "stderr_limit_bytes": 11,
                "process_limit": 2,
                "close_fds": False,
                "pass_fds": [1],
            },
        )

    def test_fake_e2e_seatbelt_executes_outputs_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)

            def write_output(_request: SandboxSubprocessRequest) -> None:
                Path(roots["output_dir"], "result.txt").write_bytes(b"done")

            runner = _RecordingRunner(output_writer=write_output)
            backend = SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                architecture="arm64",
                executable_available=True,
                command_runner=runner,
            )
            plan = _plan(
                "seatbelt",
                roots,
                network="none",
                pids=None,
                child_processes="deny",
                inherited_fds="stdio",
            )

            result = run_async(backend.execute(plan))

            self.assertTrue(result.ok)
            self.assertEqual(result.stdout, b"ok\n")
            self.assertEqual(result.output_artifacts[0].path, "result.txt")
            self.assertFalse(Path(roots["temp_dir"]).exists())
            start_request = runner.request("seatbelt_execute")
            self.assertTrue(start_request.close_fds)
            self.assertEqual(start_request.pass_fds, ())
            self.assertEqual(
                dict(start_request.environment),
                {"LC_ALL": "C.UTF-8", "PATH": "/bin"},
            )
            self.assertEqual(start_request.argv[0], "/fake/sandbox-exec")
            self.assertIn(generate_seatbelt_profile(plan), start_request.argv)

    def test_backend_mismatch_denies_before_probe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            seatbelt_runner = _RecordingRunner()
            bubblewrap_runner = _RecordingRunner()

            seatbelt_result = run_async(
                SeatbeltSandboxBackend(
                    sandbox_executable="/fake/sandbox-exec",
                    host_os="darwin",
                    executable_available=True,
                    command_runner=seatbelt_runner,
                ).execute(_plan("bubblewrap", roots, network="none"))
            )
            bubblewrap_result = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=bubblewrap_runner,
                ).execute(
                    _plan(
                        "seatbelt",
                        roots,
                        network="none",
                        pids=None,
                        child_processes="deny",
                        inherited_fds="stdio",
                    )
                )
            )

            self.assertEqual(
                seatbelt_result.status, SandboxResultStatus.DENIED
            )
            self.assertEqual(
                seatbelt_result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
            )
            self.assertEqual(
                bubblewrap_result.status,
                SandboxResultStatus.DENIED,
            )
            self.assertEqual(
                bubblewrap_result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
            )
            self.assertEqual(seatbelt_runner.requests, [])
            self.assertEqual(bubblewrap_runner.requests, [])

    def test_fake_e2e_bubblewrap_executes_with_env_and_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)

            def write_output(_request: SandboxSubprocessRequest) -> None:
                Path(roots["output_dir"], "artifact.txt").write_bytes(b"x")

            runner = _RecordingRunner(output_writer=write_output)
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                architecture="amd64",
                executable_available=True,
                process_limits_supported=True,
                command_runner=runner,
            )
            plan = _plan("bubblewrap", roots, network="none")

            result = run_async(backend.execute(plan))

            self.assertTrue(result.ok)
            self.assertEqual(result.output_artifacts[0].path, "artifact.txt")
            self.assertFalse(Path(roots["temp_dir"]).exists())
            start_request = runner.request("bubblewrap_execute")
            self.assertIsNone(start_request.process_limit)
            self.assertTrue(start_request.close_fds)
            self.assertEqual(start_request.pass_fds, ())
            self.assertEqual(
                dict(start_request.environment),
                {"LC_ALL": "C.UTF-8", "PATH": "/bin"},
            )
            self.assertIn("--clearenv", start_request.argv)
            self.assertIn("--unshare-net", start_request.argv)

    def test_probe_timeout_error_and_executable_branches(self) -> None:
        with TemporaryDirectory() as tmpdir:
            executable = Path(tmpdir, "sandbox-bin")
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)

            seatbelt_timeout = run_async(
                SeatbeltSandboxBackend(
                    sandbox_executable=str(executable),
                    host_os="darwin",
                    command_runner=_SlowRunner(),
                ).probe(timeout_seconds=0.001)
            )
            bubblewrap_timeout = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable=str(executable),
                    host_os="linux",
                    process_limits_supported=True,
                    command_runner=_SlowRunner(),
                ).probe(timeout_seconds=0.001)
            )
            seatbelt_available = run_async(
                SeatbeltSandboxBackend(
                    sandbox_executable=str(executable),
                    host_os="darwin",
                    command_runner=_RecordingRunner(),
                ).probe()
            )
            bubblewrap_available = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable=str(executable),
                    host_os="linux",
                    process_limits_supported=None,
                    command_runner=_RecordingRunner(),
                ).probe()
            )

            self.assertEqual(
                seatbelt_timeout.diagnostics[0].code,
                SandboxBackendDiagnosticCode.TIMEOUT,
            )
            self.assertEqual(
                bubblewrap_timeout.diagnostics[0].code,
                SandboxBackendDiagnosticCode.TIMEOUT,
            )
            self.assertTrue(seatbelt_available.ok)
            self.assertTrue(bubblewrap_available.ok)

    def test_probe_rejects_unsupported_os_and_missing_binary(self) -> None:
        seatbelt_wrong_os = run_async(
            SeatbeltSandboxBackend(
                host_os="linux",
                executable_available=True,
                command_runner=_RecordingRunner(),
            ).probe()
        )
        bubblewrap_wrong_os = run_async(
            BubblewrapSandboxBackend(
                host_os="darwin",
                executable_available=True,
                command_runner=_RecordingRunner(),
            ).probe()
        )
        seatbelt_missing_binary = run_async(
            SeatbeltSandboxBackend(
                host_os="darwin",
                sandbox_executable="/missing/sandbox-exec",
                executable_available=False,
                command_runner=_RecordingRunner(),
            ).probe()
        )
        missing_binary = run_async(
            BubblewrapSandboxBackend(
                host_os="linux",
                sandbox_executable="/missing/bwrap",
                executable_available=False,
                command_runner=_RecordingRunner(),
            ).probe()
        )

        self.assertFalse(seatbelt_wrong_os.ok)
        self.assertFalse(bubblewrap_wrong_os.ok)
        self.assertFalse(seatbelt_missing_binary.ok)
        self.assertFalse(missing_binary.ok)
        self.assertEqual(
            missing_binary.diagnostics[0].code,
            SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )

    def test_probe_command_failures_are_reported(self) -> None:
        cases: tuple[
            tuple[_RecordingRunner, SandboxBackendDiagnosticCode], ...
        ]
        cases = (
            (
                _RecordingRunner(
                    probe_exceptions={
                        "seatbelt_basic_profile": TimeoutError(
                            "timed out",
                        ),
                    }
                ),
                SandboxBackendDiagnosticCode.TIMEOUT,
            ),
            (
                _RecordingRunner(
                    probe_exceptions={
                        "seatbelt_basic_profile": OSError("cannot exec"),
                    }
                ),
                SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            ),
            (
                _RecordingRunner(
                    failing_probe_labels=("seatbelt_basic_profile",),
                ),
                SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            ),
        )
        for runner, code in cases:
            with self.subTest(code=code.value):
                result = run_async(
                    SeatbeltSandboxBackend(
                        sandbox_executable="/fake/sandbox-exec",
                        host_os="darwin",
                        executable_available=True,
                        command_runner=runner,
                    ).probe()
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)

    def test_bubblewrap_probe_required_control_failures(self) -> None:
        cases = (
            (
                _RecordingRunner(
                    failing_probe_labels=("bubblewrap_user_namespace",),
                ),
                "user namespaces",
            ),
            (
                _RecordingRunner(
                    probe_exceptions={
                        "bubblewrap_mount": OSError("mount denied"),
                    }
                ),
                "bind mounts",
            ),
            (
                _RecordingRunner(
                    failing_probe_labels=("bubblewrap_proc",),
                ),
                "/proc",
            ),
        )
        for runner, message in cases:
            with self.subTest(message=message):
                result = run_async(
                    BubblewrapSandboxBackend(
                        sandbox_executable="/fake/bwrap",
                        host_os="linux",
                        executable_available=True,
                        process_limits_supported=True,
                        command_runner=runner,
                    ).probe()
                )

                self.assertFalse(result.ok)
                self.assertIn(message, result.diagnostics[0].message)

    def test_bubblewrap_fails_closed_on_unsupported_network_namespace(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            runner = _RecordingRunner(
                failing_probe_labels=("bubblewrap_network_namespace",),
            )
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=runner,
            )
            probe = run_async(backend.probe())
            result = run_async(
                backend.execute(_plan("bubblewrap", roots, network="none"))
            )

            self.assertTrue(probe.ok)
            assert probe.capabilities is not None
            self.assertEqual(
                tuple(probe.capabilities.network_modes),
                (SandboxNetworkMode.FULL,),
            )
            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertIn(
                "network mode none is unsupported",
                {diagnostic.message for diagnostic in result.diagnostics},
            )
            self.assertFalse(runner.has_request("bubblewrap_execute"))

    def test_default_runner_bounds_streams_while_reading(self) -> None:
        result = run_async(
            _default_command_runner(
                SandboxSubprocessRequest(
                    operation=SandboxBackendOperation.START,
                    label="bounded_streams",
                    argv=(
                        "/bin/sh",
                        "-c",
                        "printf abcdef; printf ghijkl >&2",
                    ),
                    environment={"PATH": "/bin"},
                    timeout_seconds=2,
                    stdout_limit_bytes=3,
                    stderr_limit_bytes=2,
                )
            )
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, b"abc")
        self.assertEqual(result.stderr, b"gh")
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)

    def test_default_runner_timeout_cancel_and_no_timeout_paths(self) -> None:
        no_timeout = run_async(
            _default_command_runner(
                SandboxSubprocessRequest(
                    operation=SandboxBackendOperation.START,
                    label="no_timeout",
                    argv=("/bin/sh", "-c", "printf done"),
                    environment={"PATH": "/bin"},
                    timeout_seconds=None,
                    stdout_limit_bytes=10,
                    stderr_limit_bytes=10,
                )
            )
        )

        with self.assertRaises(TimeoutError):
            run_async(
                _default_command_runner(
                    SandboxSubprocessRequest(
                        operation=SandboxBackendOperation.START,
                        label="timeout",
                        argv=("/bin/sh", "-c", "sleep 1"),
                        environment={"PATH": "/bin"},
                        timeout_seconds=0.001,
                        stdout_limit_bytes=10,
                        stderr_limit_bytes=10,
                    )
                )
            )

        async def cancel_runner() -> bool:
            task = create_task(
                _default_command_runner(
                    SandboxSubprocessRequest(
                        operation=SandboxBackendOperation.START,
                        label="cancel",
                        argv=("/bin/sh", "-c", "sleep 1"),
                        environment={"PATH": "/bin"},
                        timeout_seconds=None,
                        stdout_limit_bytes=10,
                        stderr_limit_bytes=10,
                    )
                )
            )
            await async_sleep(0.05)
            task.cancel()
            try:
                await task
            except CancelledError:
                return True
            return False

        self.assertEqual(no_timeout.stdout, b"done")
        self.assertTrue(run_async(cancel_runner()))

    def test_profile_generation_network_modes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            seatbelt_loopback = generate_seatbelt_profile(
                _plan(
                    "seatbelt",
                    roots,
                    network="loopback",
                    pids=None,
                    child_processes="deny",
                    inherited_fds="stdio",
                )
            )
            bubblewrap_full = generate_bubblewrap_arguments(
                _plan("bubblewrap", roots, network="full"),
                sandbox_executable="/usr/bin/bwrap",
                private_runtime_dir="/tmp/avalan-runtime",
            )

            self.assertIn('"127.0.0.1:*"', seatbelt_loopback)
            self.assertIn("--share-net", bubblewrap_full)
            with self.assertRaises(AssertionError):
                generate_seatbelt_profile(
                    _plan(
                        "seatbelt",
                        roots,
                        network="allowlist",
                        egress=("api.internal",),
                        pids=None,
                        child_processes="deny",
                        inherited_fds="stdio",
                    )
                )
            with self.assertRaises(AssertionError):
                generate_seatbelt_profile(
                    _plan(
                        "seatbelt",
                        roots,
                        network="full",
                        pids=None,
                        child_processes="deny",
                        inherited_fds="stdio",
                    )
                )
            with self.assertRaises(AssertionError):
                generate_bubblewrap_arguments(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="allowlist",
                        egress=("api.internal",),
                    )
                )

    def test_path_escape_and_symlink_escape_are_denied(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside")
            outside.mkdir()
            outside_file = outside / "secret.txt"
            outside_file.write_text("secret")
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=_RecordingRunner(),
            )
            escaped_arg = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        argv=("/bin/sh", "-lc", str(outside_file)),
                    )
                )
            )

            link = Path(roots["workspace"], "project-link")
            link.symlink_to(outside, target_is_directory=True)
            symlink_cwd = run_async(
                backend.execute(_plan("bubblewrap", roots, cwd=str(link)))
            )

            for result in (escaped_arg, symlink_cwd):
                self.assertEqual(result.status, SandboxResultStatus.DENIED)
                self.assertEqual(
                    result.diagnostics[0].code,
                    SandboxBackendDiagnosticCode.PATH_DENIED,
                )
            self.assertFalse(Path(roots["temp_dir"]).exists())

    def test_denied_root_candidate_and_cleanup_retain_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            denied_file = Path(roots["deny"], "blocked.txt")
            denied_file.write_text("blocked")
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=_RecordingRunner(),
            )

            denied = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        argv=("/bin/sh", "-lc", str(denied_file)),
                        network="none",
                    )
                )
            )
            retained = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="none",
                        cleanup="preserve",
                    )
                )
            )

            self.assertEqual(denied.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                denied.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertTrue(retained.ok)
            self.assertTrue(Path(roots["temp_dir"]).exists())

    def test_seatbelt_path_denial_cleans_prepared_temp_dir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside")
            outside.mkdir()
            outside_file = outside / "secret.txt"
            outside_file.write_text("secret")
            backend = SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                executable_available=True,
                command_runner=_RecordingRunner(),
            )

            result = run_async(
                backend.execute(
                    _plan(
                        "seatbelt",
                        roots,
                        argv=("/bin/sh", "-lc", str(outside_file)),
                        network="none",
                        pids=None,
                        child_processes="deny",
                        inherited_fds="stdio",
                    )
                )
            )

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertFalse(Path(roots["temp_dir"]).exists())

    def test_seatbelt_allows_slash_prefixed_nonpath_arguments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            runner = _RecordingRunner()
            backend = SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                executable_available=True,
                command_runner=runner,
            )

            result = run_async(
                backend.execute(
                    _plan(
                        "seatbelt",
                        roots,
                        argv=("/bin/sh", "-lc", "/tool-skills-file/p"),
                        network="none",
                        pids=None,
                        child_processes="deny",
                        inherited_fds="stdio",
                    )
                )
            )

            self.assertTrue(result.ok)
            self.assertTrue(runner.has_request("seatbelt_execute"))

    def test_absolute_argument_path_exists_error_is_conservative(self) -> None:
        with patch.object(Path, "exists", side_effect=OSError("denied")):
            self.assertTrue(_argument_is_existing_absolute_path("/maybe/path"))

    def test_symlinked_write_root_is_denied_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-write-root")
            outside.mkdir()
            Path(roots["output_dir"]).rmdir()
            Path(roots["output_root"]).rmdir()
            Path(roots["output_root"]).symlink_to(
                outside,
                target_is_directory=True,
            )
            runner = _RecordingRunner()
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=runner,
            )
            plan = _plan("bubblewrap", roots, network="none")

            result = run_async(backend.execute(plan))

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertIn("write_roots", result.diagnostics[0].message)
            self.assertFalse(runner.has_request("bubblewrap_execute"))
            self.assertFalse(Path(roots["temp_dir"]).exists())
            with self.assertRaises(AssertionError):
                generate_bubblewrap_arguments(plan)

    def test_symlinked_output_root_is_denied_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-output")
            outside.mkdir()
            Path(roots["output_dir"]).rmdir()
            Path(roots["output_dir"]).symlink_to(
                outside,
                target_is_directory=True,
            )
            runner = _RecordingRunner()
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=runner,
            )
            plan = _plan("bubblewrap", roots, network="none")

            result = run_async(backend.execute(plan))

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertIn("output_dir", result.diagnostics[0].message)
            self.assertFalse(runner.has_request("bubblewrap_execute"))
            self.assertFalse(Path(roots["temp_dir"]).exists())
            self.assertTrue(outside.exists())

    def test_symlinked_output_parent_is_denied_before_mkdir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-output-parent")
            outside.mkdir()
            link = Path(roots["output_root"], "link")
            link.symlink_to(outside, target_is_directory=True)
            roots["output_dir"] = str(link / "session")
            runner = _RecordingRunner()
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=runner,
            )
            plan = _plan("bubblewrap", roots, network="none")

            result = run_async(backend.execute(plan))

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertIn("output_dir", result.diagnostics[0].message)
            self.assertFalse(runner.has_request("bubblewrap_execute"))
            self.assertFalse((outside / "session").exists())
            self.assertFalse(Path(roots["temp_dir"]).exists())
            with self.assertRaises(AssertionError):
                generate_bubblewrap_arguments(plan)

    def test_symlinked_temp_root_is_denied_and_unlinked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-temp")
            outside.mkdir()
            Path(roots["temp_dir"]).rmdir()
            Path(roots["temp_dir"]).symlink_to(
                outside,
                target_is_directory=True,
            )
            runner = _RecordingRunner()
            backend = SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                executable_available=True,
                command_runner=runner,
            )
            plan = _plan(
                "seatbelt",
                roots,
                network="none",
                pids=None,
                child_processes="deny",
                inherited_fds="stdio",
            )

            with self.assertRaises(AssertionError):
                generate_seatbelt_profile(plan)
            result = run_async(backend.execute(plan))

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertIn("temp_dir", result.diagnostics[0].message)
            self.assertFalse(runner.has_request("seatbelt_execute"))
            self.assertFalse(Path(roots["temp_dir"]).exists())
            self.assertTrue(outside.exists())

    def test_symlinked_temp_parent_is_denied_before_mkdir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-temp-parent")
            outside.mkdir()
            link = Path(roots["scratch_root"], "link")
            link.symlink_to(outside, target_is_directory=True)
            roots["temp_dir"] = str(link / "session")
            runner = _RecordingRunner()
            backend = SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                executable_available=True,
                command_runner=runner,
            )
            plan = _plan(
                "seatbelt",
                roots,
                network="none",
                pids=None,
                child_processes="deny",
                inherited_fds="stdio",
            )

            with self.assertRaises(AssertionError):
                generate_seatbelt_profile(plan)
            result = run_async(backend.execute(plan))

            self.assertEqual(result.status, SandboxResultStatus.DENIED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.PATH_DENIED,
            )
            self.assertIn("temp_dir", result.diagnostics[0].message)
            self.assertFalse(runner.has_request("seatbelt_execute"))
            self.assertFalse((outside / "session").exists())

    def test_execution_timeout_cancel_truncation_and_cleanup_failures(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            timeout = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        execution_exception=TimeoutError("timeout"),
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )
            cancelled = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        execution_exception=CancelledError(),
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )
            truncated = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        execution_stdout=b"x" * 2048,
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )

            async def slow_cleanup(_paths: Sequence[str]) -> bool:
                await async_sleep(1)
                return True

            async def cancelled_cleanup(_paths: Sequence[str]) -> bool:
                raise CancelledError()

            no_paths = run_async(
                _cleanup_with_budget(
                    slow_cleanup,
                    (),
                    0.001,
                    SandboxBackend.BUBBLEWRAP,
                )
            )
            cleanup_timeout = run_async(
                _cleanup_with_budget(
                    slow_cleanup,
                    (roots["temp_dir"],),
                    0.001,
                    SandboxBackend.BUBBLEWRAP,
                )
            )
            cleanup_cancelled = run_async(
                _cleanup_with_budget(
                    cancelled_cleanup,
                    (roots["temp_dir"],),
                    0.5,
                    SandboxBackend.BUBBLEWRAP,
                )
            )

            self.assertEqual(timeout.status, SandboxResultStatus.TIMED_OUT)
            self.assertEqual(cancelled.status, SandboxResultStatus.CANCELLED)
            self.assertTrue(truncated.stream_truncated)
            self.assertEqual(
                truncated.diagnostics[-1].code,
                SandboxBackendDiagnosticCode.STREAM_TRUNCATED,
            )
            self.assertIsNone(no_paths)
            assert cleanup_timeout is not None
            self.assertEqual(
                cleanup_timeout.code,
                SandboxBackendDiagnosticCode.CLEANUP_FAILED,
            )
            assert cleanup_cancelled is not None
            self.assertEqual(
                cleanup_cancelled.code,
                SandboxBackendDiagnosticCode.CANCELLED,
            )

    def test_denied_writes_denied_network_and_fd_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            write_runner = _RecordingRunner(
                execution_exit_code=1,
                execution_stderr=b"Read-only file system\n",
            )
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=write_runner,
            )
            denied_write = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="none",
                        argv=(
                            "/bin/sh",
                            "-lc",
                            f"printf x > {roots['workspace']}/blocked.txt",
                        ),
                    )
                )
            )
            denied_network = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="allowlist",
                        egress=("api.internal",),
                    )
                )
            )
            fd_request = write_runner.request("bubblewrap_execute")

            self.assertEqual(denied_write.status, SandboxResultStatus.FAILED)
            self.assertEqual(
                denied_write.diagnostics[0].code,
                SandboxBackendDiagnosticCode.EXECUTION_FAILED,
            )
            self.assertTrue(
                _contains_triplet(
                    fd_request.argv,
                    "--ro-bind",
                    roots["workspace"],
                    roots["workspace"],
                )
            )
            self.assertTrue(fd_request.close_fds)
            self.assertEqual(fd_request.pass_fds, ())
            self.assertEqual(denied_network.status, SandboxResultStatus.DENIED)
            self.assertIn(
                "network mode allowlist is unsupported",
                {
                    diagnostic.message
                    for diagnostic in denied_network.diagnostics
                },
            )

    def test_process_child_and_inherited_fd_denials(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            unsupported_pids = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=False,
                    command_runner=_RecordingRunner(),
                ).execute(_plan("bubblewrap", roots, pids=2))
            )
            best_effort_pids = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(),
                ).execute(_plan("bubblewrap", roots, pids=2))
            )
            child_runner = _RecordingRunner()
            child_denied = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=child_runner,
                ).execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        child_processes="deny",
                    )
                )
            )
            seatbelt_denied = run_async(
                SeatbeltSandboxBackend(
                    sandbox_executable="/fake/sandbox-exec",
                    host_os="darwin",
                    executable_available=True,
                    command_runner=_RecordingRunner(),
                ).execute(
                    _plan(
                        "seatbelt",
                        roots,
                        network="none",
                        pids=2,
                        inherited_fds="explicit",
                    )
                )
            )

            self.assertEqual(
                unsupported_pids.status, SandboxResultStatus.DENIED
            )
            self.assertIn(
                "pid limits unsupported",
                {
                    diagnostic.message
                    for diagnostic in unsupported_pids.diagnostics
                },
            )
            self.assertEqual(
                best_effort_pids.status,
                SandboxResultStatus.DENIED,
            )
            self.assertIn(
                "pid limits unsupported",
                {
                    diagnostic.message
                    for diagnostic in best_effort_pids.diagnostics
                },
            )
            self.assertEqual(child_denied.status, SandboxResultStatus.DENIED)
            self.assertIn(
                "child process denial unsupported",
                {
                    diagnostic.message
                    for diagnostic in child_denied.diagnostics
                },
            )
            self.assertFalse(child_runner.has_request("bubblewrap_execute"))
            self.assertEqual(
                seatbelt_denied.status, SandboxResultStatus.DENIED
            )
            self.assertIn(
                "pid limits unsupported",
                {
                    diagnostic.message
                    for diagnostic in seatbelt_denied.diagnostics
                },
            )
            self.assertIn(
                "inherited fd policy unsupported",
                {
                    diagnostic.message
                    for diagnostic in seatbelt_denied.diagnostics
                },
            )

    def test_output_rejection_and_cleanup_uncertainty(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            outside = Path(tmpdir, "outside-output.txt")
            outside.write_bytes(b"secret")

            def write_symlink(_request: SandboxSubprocessRequest) -> None:
                Path(roots["output_dir"], "leak").symlink_to(outside)

            rejected = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        output_writer=write_symlink,
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )
            Path(roots["output_dir"], "leak").unlink()

            def write_large(_request: SandboxSubprocessRequest) -> None:
                Path(roots["output_dir"], "large.bin").write_bytes(b"x" * 2048)

            oversized = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        output_writer=write_large,
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )

            async def uncertain_cleanup(_paths: Sequence[str]) -> bool:
                return False

            uncertain = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(),
                    cleanup_handler=uncertain_cleanup,
                ).execute(_plan("bubblewrap", roots, network="none"))
            )

            self.assertEqual(rejected.status, SandboxResultStatus.FAILED)
            self.assertEqual(
                rejected.diagnostics[0].code,
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            )
            self.assertEqual(oversized.status, SandboxResultStatus.FAILED)
            self.assertEqual(
                oversized.diagnostics[0].code,
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            )
            self.assertEqual(uncertain.status, SandboxResultStatus.FAILED)
            self.assertTrue(uncertain.cleanup_uncertain)
            self.assertEqual(
                uncertain.diagnostics[-1].code,
                SandboxBackendDiagnosticCode.CLEANUP_FAILED,
            )

    def test_output_collection_edge_cases(self) -> None:
        with TemporaryDirectory() as tmpdir:
            roots = _roots(tmpdir)
            backend = BubblewrapSandboxBackend(
                sandbox_executable="/fake/bwrap",
                host_os="linux",
                executable_available=True,
                process_limits_supported=True,
                command_runner=_RecordingRunner(),
            )

            not_collected = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="none",
                        collect_outputs=False,
                    )
                )
            )
            artifacts_disabled = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="none",
                        allow_artifacts=False,
                        max_artifact_bytes=0,
                    )
                )
            )
            unmapped_output = run_async(
                backend.execute(
                    _plan(
                        "bubblewrap",
                        roots,
                        network="none",
                        include_output_dir=False,
                    )
                )
            )

            missing_roots = _roots(tmpdir)
            missing_output = Path(missing_roots["output_root"], "missing")
            missing_roots["output_dir"] = str(missing_output)
            missing_plan = _plan(
                "bubblewrap",
                missing_roots,
                network="none",
            )
            missing_artifacts, missing_diagnostic = _collect_real_outputs(
                missing_plan,
                SandboxBackend.BUBBLEWRAP,
            )

            def write_nested(_request: SandboxSubprocessRequest) -> None:
                nested = Path(roots["output_dir"], "nested")
                nested.mkdir()
                Path(nested, "artifact.txt").write_bytes(b"ok")

            nested = run_async(
                BubblewrapSandboxBackend(
                    sandbox_executable="/fake/bwrap",
                    host_os="linux",
                    executable_available=True,
                    process_limits_supported=True,
                    command_runner=_RecordingRunner(
                        output_writer=write_nested,
                    ),
                ).execute(_plan("bubblewrap", roots, network="none"))
            )

            self.assertTrue(not_collected.ok)
            self.assertEqual(not_collected.output_artifacts, ())
            self.assertEqual(
                artifacts_disabled.status,
                SandboxResultStatus.FAILED,
            )
            self.assertEqual(
                artifacts_disabled.diagnostics[0].code,
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            )
            self.assertEqual(
                unmapped_output.status,
                SandboxResultStatus.FAILED,
            )
            self.assertEqual(missing_artifacts, ())
            self.assertIsNone(missing_diagnostic)
            self.assertTrue(nested.ok)
            self.assertEqual(
                nested.output_artifacts[0].path,
                "nested/artifact.txt",
            )

    def test_default_cleanup_handler_reports_os_errors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            stubborn = Path(tmpdir, "stubborn")
            stubborn.mkdir()

            with patch(
                "avalan.sandbox.backend.rmtree",
                side_effect=OSError("cleanup failed"),
            ):
                cleaned = run_async(_default_cleanup_handler((str(stubborn),)))

            self.assertFalse(cleaned)

    def test_process_limit_preexec_uses_resource_limit(self) -> None:
        calls: list[tuple[int, tuple[int, int]]] = []

        def get_limit(resource: int) -> tuple[int, int]:
            self.assertEqual(resource, 7)
            return 100, 50

        def set_limit(resource: int, limits: tuple[int, int]) -> None:
            calls.append((resource, limits))

        with (
            patch("avalan.sandbox.backend._RESOURCE_RLIMIT_NPROC", 7),
            patch("avalan.sandbox.backend._RESOURCE_GETRLIMIT", get_limit),
            patch("avalan.sandbox.backend._RESOURCE_SETRLIMIT", set_limit),
        ):
            apply_limit = _process_limit_preexec(80)
            assert apply_limit is not None
            apply_limit()

        self.assertEqual(calls, [(7, (50, 50))])

    def test_system_prefix_helper_exact_prefix(self) -> None:
        self.assertEqual(
            _replace_path_prefix("/tmp", "/tmp", "/private/tmp"),
            "/private/tmp",
        )

    def test_system_prefix_helper_accepts_nested_equivalent_prefix(
        self,
    ) -> None:
        self.assertEqual(
            _replace_path_prefix("/tmp/work", "/tmp", "/private/tmp"),
            "/private/tmp/work",
        )
        self.assertTrue(
            _system_prefix_resolves_equivalently(
                "/tmp/work",
                "/private/tmp/work",
            )
        )

    def test_start_failures_clean_prepared_temp_dirs(self) -> None:
        cases: tuple[
            tuple[BaseException, SandboxBackendDiagnosticCode],
            ...,
        ] = (
            (
                FileNotFoundError("missing bwrap"),
                SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            ),
            (
                PermissionError("permission denied"),
                SandboxBackendDiagnosticCode.EXECUTION_FAILED,
            ),
            (
                OSError("exec format error"),
                SandboxBackendDiagnosticCode.EXECUTION_FAILED,
            ),
        )
        for exception, code in cases:
            with self.subTest(exception=type(exception).__name__):
                with TemporaryDirectory() as tmpdir:
                    roots = _roots(tmpdir)
                    runner = _RecordingRunner(
                        execution_exception=exception,
                    )
                    backend = BubblewrapSandboxBackend(
                        sandbox_executable="/fake/bwrap",
                        host_os="linux",
                        executable_available=True,
                        process_limits_supported=True,
                        command_runner=runner,
                    )

                    result = run_async(
                        backend.execute(
                            _plan("bubblewrap", roots, network="none")
                        )
                    )

                    self.assertEqual(
                        result.status,
                        SandboxResultStatus.FAILED,
                    )
                    self.assertEqual(result.diagnostics[0].code, code)
                    self.assertFalse(Path(roots["temp_dir"]).exists())

    def test_zz_resource_import_fallback(self) -> None:
        def import_without_resource(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> object:
            if name == "resource":
                raise ImportError("resource unavailable")
            return builtin_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", import_without_resource):
                reload(backend_module)

            self.assertEqual(backend_module._RESOURCE_RLIMIT_NPROC, -1)
            self.assertIsNone(backend_module._RESOURCE_GETRLIMIT)
            self.assertIsNone(backend_module._RESOURCE_SETRLIMIT)
        finally:
            reload(backend_module)


class _RecordingRunner:
    def __init__(
        self,
        *,
        failing_probe_labels: Sequence[str] = (),
        probe_exceptions: Mapping[str, BaseException] | None = None,
        execution_exit_code: int = 0,
        execution_stdout: bytes = b"ok\n",
        execution_stderr: bytes = b"",
        output_writer: (
            Callable[[SandboxSubprocessRequest], None] | None
        ) = None,
        execution_exception: BaseException | None = None,
    ) -> None:
        self.requests: list[SandboxSubprocessRequest] = []
        self._failing_probe_labels = set(failing_probe_labels)
        self._probe_exceptions = dict(probe_exceptions or {})
        self._execution_exit_code = execution_exit_code
        self._execution_stdout = execution_stdout
        self._execution_stderr = execution_stderr
        self._output_writer = output_writer
        self._execution_exception = execution_exception

    async def __call__(
        self,
        request: SandboxSubprocessRequest,
    ) -> SandboxSubprocessResult:
        self.requests.append(request)
        operation = cast(SandboxBackendOperation, request.operation)
        if operation is SandboxBackendOperation.PROBE:
            if request.label in self._probe_exceptions:
                raise self._probe_exceptions[request.label]
            exit_code = 1 if request.label in self._failing_probe_labels else 0
            return SandboxSubprocessResult(exit_code=exit_code)
        if self._execution_exception is not None:
            raise self._execution_exception
        if self._output_writer is not None:
            self._output_writer(request)
        return SandboxSubprocessResult(
            exit_code=self._execution_exit_code,
            stdout=self._execution_stdout,
            stderr=self._execution_stderr,
        )

    def request(self, label: str) -> SandboxSubprocessRequest:
        for request in self.requests:
            if request.label == label:
                return request
        raise AssertionError(f"missing request: {label}")

    def has_request(self, label: str) -> bool:
        return any(request.label == label for request in self.requests)


class _SlowRunner:
    async def __call__(
        self,
        _request: SandboxSubprocessRequest,
    ) -> SandboxSubprocessResult:
        await async_sleep(1)
        return SandboxSubprocessResult(exit_code=0)


def _roots(tmpdir: str) -> dict[str, str]:
    root = Path(tmpdir)
    workspace = root / "workspace"
    project = workspace / "project"
    output_root = workspace / "out"
    output_dir = output_root / "session"
    scratch_root = root / "scratch"
    temp_dir = scratch_root / "session"
    deny = root / "deny"
    for path in (
        project,
        output_dir,
        scratch_root,
        temp_dir,
        deny,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "workspace": str(workspace),
        "project": str(project),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "scratch_root": str(scratch_root),
        "temp_dir": str(temp_dir),
        "deny": str(deny),
    }


def _effective(
    backend: str,
    roots: dict[str, str],
    *,
    network: str = "none",
    egress: tuple[str, ...] = (),
    pids: int | None = None,
    child_processes: str = "allow",
    inherited_fds: str = "explicit",
    cleanup: str = "delete",
    allow_artifacts: bool = True,
    max_artifact_bytes: int = 1024,
) -> SandboxEffectiveSettings:
    settings = IsolationSettings.from_dict(
        _settings_raw(
            backend,
            roots,
            network=network,
            egress=egress,
            pids=pids,
            child_processes=child_processes,
            inherited_fds=inherited_fds,
            cleanup=cleanup,
            allow_artifacts=allow_artifacts,
            max_artifact_bytes=max_artifact_bytes,
        ),
        source=trusted_isolation_source("sdk"),
    ).select_profile(
        IsolationProfileSelection(
            mode=IsolationMode.SANDBOX,
            profile="host-tools",
            required=True,
        )
    )
    assert settings.sandbox is not None
    return settings.sandbox


def _settings_raw(
    backend: str,
    roots: dict[str, str],
    *,
    network: str,
    egress: tuple[str, ...],
    pids: int | None,
    child_processes: str,
    inherited_fds: str,
    cleanup: str,
    allow_artifacts: bool,
    max_artifact_bytes: int,
) -> dict[str, object]:
    return {
        "mode": "sandbox",
        "sandbox": {
            "backend": backend,
            "default_profile": "host-tools",
            "allowed_profiles": ["host-tools"],
            "profiles": {
                "host-tools": {
                    "trusted_executables": ["/bin/sh"],
                    "executable_search_roots": ["/bin"],
                    "read_roots": [roots["workspace"]],
                    "write_roots": [roots["output_root"]],
                    "deny_roots": [roots["deny"]],
                    "scratch_roots": [roots["scratch_root"]],
                    "output_roots": [roots["output_root"]],
                    "environment": {
                        "variables": {"LC_ALL": "C.UTF-8"},
                        "allowlist": ["PATH"],
                    },
                    "network": {
                        "mode": network,
                        "egress_allowlist": list(egress),
                    },
                    "resources": {
                        "timeout_seconds": 2,
                        "pids": pids,
                    },
                    "output": {
                        "max_stdout_bytes": 1024,
                        "max_stderr_bytes": 1024,
                        "allow_artifacts": allow_artifacts,
                        "max_artifact_bytes": max_artifact_bytes,
                    },
                    "child_processes": child_processes,
                    "inherited_fds": inherited_fds,
                    "cleanup": cleanup,
                },
            },
            "profile_registry_id": "default",
            "policy_version": "phase6",
        },
    }


def _plan(
    backend: str,
    roots: dict[str, str],
    *,
    argv: tuple[str, ...] = ("/bin/sh", "-lc", "echo ok"),
    cwd: str | None = None,
    network: str = "none",
    egress: tuple[str, ...] = (),
    pids: int | None = None,
    child_processes: str = "allow",
    inherited_fds: str = "explicit",
    cleanup: str = "delete",
    allow_artifacts: bool = True,
    max_artifact_bytes: int = 1024,
    collect_outputs: bool = True,
    include_temp_dir: bool = True,
    include_output_dir: bool = True,
    cleanup_budget_seconds: float = 0.5,
) -> SandboxExecutionPlan:
    return SandboxExecutionPlan(
        request=SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.TYPED_TOOL,
            logical_name="shell",
            command="/bin/sh",
            argv=argv,
            cwd=cwd or roots["project"],
        ),
        settings=_effective(
            backend,
            roots,
            network=network,
            egress=egress,
            pids=pids,
            child_processes=child_processes,
            inherited_fds=inherited_fds,
            cleanup=cleanup,
            allow_artifacts=allow_artifacts,
            max_artifact_bytes=max_artifact_bytes,
        ),
        environment={"PATH": "/bin", "LC_ALL": "C.UTF-8"},
        temp_dir=roots["temp_dir"] if include_temp_dir else None,
        output_dir=roots["output_dir"] if include_output_dir else None,
        collect_outputs=collect_outputs,
        cleanup_budget_seconds=cleanup_budget_seconds,
        stream_buffer_bytes=4096,
    )


def _contains_triplet(
    argv: Sequence[str],
    option: str,
    source: str,
    target: str,
) -> bool:
    for index in range(len(argv) - 2):
        if argv[index : index + 3] == (option, source, target):
            return True
    return False


if __name__ == "__main__":
    main()
