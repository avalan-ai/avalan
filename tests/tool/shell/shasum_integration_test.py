from dataclasses import replace
from hashlib import sha256
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main, skipUnless

from avalan.container import (
    ContainerBackend,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerProfile,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
)
from avalan.entities import ToolCallContext
from avalan.isolation import (
    SandboxBackend,
    SandboxEffectiveSettings,
    SandboxEnvironmentPolicy,
    SandboxOutputPolicy,
    SandboxProfile,
    SandboxResourceLimits,
    trusted_isolation_source,
)
from avalan.sandbox import (
    SandboxBackendDiagnosticCode,
    SandboxResultStatus,
    SandboxSubprocessRequest,
    SandboxSubprocessResult,
    SeatbeltSandboxBackend,
    generate_bubblewrap_arguments,
    generate_seatbelt_profile,
)
from avalan.tool.shell import (
    SHELL_COMMAND_DEFINITIONS,
    ExecutionPolicy,
    LocalCommandExecutor,
    PathOperand,
    ShasumTool,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellSandboxCommandExecutor,
    ShellToolSettings,
    normalize_shell_execution_request,
)
from avalan.tool.shell.entities import _ShellRuntimeDependency

_DIGEST = "8" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"
_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
_SEATBELT_SMOKE_ENV_NAME = "AVALAN_SHELL_REAL_SEATBELT_SHASUM"


class ShasumLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_local_subprocess_hashes_binary_and_text_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "binary.bin").write_bytes(b"\x00\xff")
            (root / "visible.txt").write_text("hello world", encoding="utf-8")
            executable, marker = _fake_shasum(root)
            tool = _tool(root, executable)

            output = await tool(
                ("binary.bin", "visible.txt"),
                algorithm="256",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.argv,
            (
                "shasum",
                "-a",
                "256",
                "--",
                "binary.bin",
                "visible.txt",
            ),
        )
        self.assertIn(f"{_SHA256}  visible.txt", result.stdout)
        self.assertEqual(
            launched,
            "-a 256 -- binary.bin visible.txt\n",
        )

    async def test_nested_cwd_local_subprocess_preserves_relative_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "-first.bin").write_bytes(b"\x00first")
            (nested / "second file.txt").write_text(
                "hello world",
                encoding="utf-8",
            )
            executable, marker = _fake_shasum(root)
            tool = _tool(root, executable)

            output = await tool(
                ("-first.bin", "second file.txt"),
                algorithm="256",
                cwd="nested",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.argv,
            (
                "shasum",
                "-a",
                "256",
                "--",
                "./-first.bin",
                "second file.txt",
            ),
        )
        self.assertEqual(result.display_argv, result.argv)
        self.assertEqual(result.display_cwd, "nested")
        self.assertEqual(
            launched,
            "-a 256 -- ./-first.bin second file.txt\n",
        )

    async def test_invalid_direct_arguments_do_not_launch(self) -> None:
        cases = (
            ({"paths": (), "algorithm": "256"}, "invalid_option"),
            (
                {"paths": ("visible.txt",), "algorithm": "sha256"},
                "invalid_option",
            ),
            (
                {"paths": ("directory",), "algorithm": "256"},
                "denied_path",
            ),
        )
        for arguments, error_code in cases:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    (root / "visible.txt").write_text(
                        "hello world",
                        encoding="utf-8",
                    )
                    (root / "directory").mkdir()
                    executable, marker = _fake_shasum(root)
                    tool = _tool(root, executable)

                    output = await tool(
                        **arguments,
                        context=ToolCallContext(),
                    )

                    self.assertFalse(marker.exists())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                self.assertIs(
                    output.execution_result.status,
                    ShellExecutionStatus.POLICY_DENIED,
                )
                assert output.execution_result.error_code is not None
                self.assertEqual(
                    output.execution_result.error_code.value,
                    error_code,
                )


class ShasumBackendPlanningIntegrationTest(IsolatedAsyncioTestCase):
    async def test_container_plan_preserves_bounded_shasum_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request("visible.bin", algorithm="512256"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_container_settings(),
            )

        expected = ("shasum", "-a", "512256", "--", "visible.bin")
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, expected)
        self.assertEqual(plan.local_spec.runtime_dependencies, ())
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace",
        )

    async def test_nested_cwd_container_plan_uses_cwd_relative_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "-first.bin").write_bytes(b"\x00first")
            (nested / "second file.bin").write_bytes(b"\x00second")
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(
                    "-first.bin",
                    "second file.bin",
                    algorithm="256",
                    cwd="nested",
                ),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_container_settings(),
            )

        expected = (
            "shasum",
            "-a",
            "256",
            "--",
            "./-first.bin",
            "second file.bin",
        )
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.display_argv, expected)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace/nested",
        )

    async def test_sandbox_plan_preserves_bounded_shasum_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )

            sandbox_settings = _sandbox_settings(root)
            plan = await normalize_shell_execution_request(
                _request("visible.bin", algorithm="384"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=sandbox_settings,
            )
            assert plan.sandbox_plan is not None
            seatbelt_profile = generate_seatbelt_profile(plan.sandbox_plan)
            unwidened_profile = replace(
                plan.sandbox_plan.settings.profile,
                read_roots=sandbox_settings.profile.read_roots,
            )
            unwidened_plan = replace(
                plan.sandbox_plan,
                settings=replace(
                    plan.sandbox_plan.settings,
                    profile=unwidened_profile,
                ),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        expected = ("shasum", "-a", "384", "--", "visible.bin")
        self.assertEqual(plan.local_spec.argv, expected)
        self.assertEqual(
            plan.local_spec.runtime_dependencies,
            (_ShellRuntimeDependency.SYSTEM_PERL,),
        )
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            (
                "/trusted/bin/shasum",
                "-a",
                "384",
                "--",
                "visible.bin",
            ),
        )
        self.assertEqual(plan.sandbox_plan.request.cwd, str(root.resolve()))
        self.assertEqual(
            sandbox_settings.profile.read_roots,
            (str(root.resolve()),),
        )
        self.assertEqual(
            plan.sandbox_plan.settings.profile.read_roots,
            (str(root.resolve()), "/System/Library/Perl"),
        )
        self.assertIn(
            '(allow file-read* (subpath "/System/Library/Perl"))',
            seatbelt_profile,
        )
        self.assertNotIn(
            '(allow file-read* (subpath "/Library/Perl"))',
            seatbelt_profile,
        )
        self.assertNotEqual(
            plan.sandbox_plan.plan_fingerprint,
            unwidened_plan.plan_fingerprint,
        )
        self.assertEqual(
            plan.sandbox_plan.to_dict()["plan_fingerprint"],
            plan.sandbox_plan.plan_fingerprint,
        )

    async def test_nested_cwd_sandbox_plan_uses_cwd_relative_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "-first.bin").write_bytes(b"\x00first")
            (nested / "second file.bin").write_bytes(b"\x00second")
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(
                    "-first.bin",
                    "second file.bin",
                    algorithm="512",
                    cwd="nested",
                ),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(root),
            )

        expected = (
            "/trusted/bin/shasum",
            "-a",
            "512",
            "--",
            "./-first.bin",
            "second file.bin",
        )
        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        assert plan.sandbox_plan is not None
        self.assertEqual(plan.sandbox_plan.request.argv, expected)
        self.assertEqual(plan.sandbox_plan.request.cwd, str(nested.resolve()))

    async def test_bubblewrap_binds_configured_perl_runtime_root(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime_root = root / "system-perl-runtime"
            runtime_root.mkdir()
            (root / "visible.bin").write_bytes(b"\x00visible")
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request("visible.bin", algorithm="256"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(
                    root,
                    backend=SandboxBackend.BUBBLEWRAP,
                    extra_read_roots=(str(runtime_root),),
                ),
            )
            assert plan.sandbox_plan is not None
            bubblewrap_argv = generate_bubblewrap_arguments(plan.sandbox_plan)

        self.assertEqual(
            plan.local_spec.runtime_dependencies,
            (_ShellRuntimeDependency.SYSTEM_PERL,),
        )
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.settings.profile.read_roots,
            (
                str(root.resolve()),
                str(runtime_root),
            ),
        )
        runtime_argument_index = bubblewrap_argv.index(str(runtime_root))
        self.assertEqual(
            bubblewrap_argv[
                runtime_argument_index - 1 : runtime_argument_index + 2
            ],
            ("--ro-bind", str(runtime_root), str(runtime_root)),
        )
        self.assertNotIn("/System/Library/Perl", bubblewrap_argv)

    async def test_replaced_definition_cannot_mint_runtime_root(self) -> None:
        original = SHELL_COMMAND_DEFINITIONS["shasum"]
        with self.assertRaisesRegex(
            AssertionError,
            "runtime_dependencies",
        ):
            replace(
                original,
                runtime_dependencies=("/",),
            )
        forged = replace(original, runtime_dependencies=())
        object.__setattr__(forged, "runtime_dependencies", ("/",))
        replacement = replace(original, runtime_dependencies=())

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )
            SHELL_COMMAND_DEFINITIONS["shasum"] = forged
            try:
                with self.assertRaisesRegex(
                    AssertionError,
                    "runtime_dependencies",
                ):
                    await normalize_shell_execution_request(
                        _request("visible.bin", algorithm="256"),
                        ExecutionPolicy(
                            settings=settings,
                            resolver=_AllResolved(),
                        ),
                        sandbox_settings=_sandbox_settings(root),
                    )
            finally:
                SHELL_COMMAND_DEFINITIONS["shasum"] = original
            SHELL_COMMAND_DEFINITIONS["shasum"] = replacement
            try:
                plan = await normalize_shell_execution_request(
                    _request("visible.bin", algorithm="256"),
                    ExecutionPolicy(
                        settings=settings,
                        resolver=_AllResolved(),
                    ),
                    sandbox_settings=_sandbox_settings(root),
                )
            finally:
                SHELL_COMMAND_DEFINITIONS["shasum"] = original
            assert plan.sandbox_plan is not None
            profile = generate_seatbelt_profile(plan.sandbox_plan)

        self.assertEqual(plan.local_spec.runtime_dependencies, ())
        self.assertEqual(
            plan.sandbox_plan.settings.profile.read_roots,
            (str(root.resolve()),),
        )
        self.assertNotIn(
            '(allow file-read* (subpath "/"))',
            profile,
        )
        self.assertNotIn("/System/Library/Perl", profile)

    async def test_runtime_dependency_cannot_override_deny_root(self) -> None:
        launched: list[str] = []

        async def runner(
            request: SandboxSubprocessRequest,
        ) -> SandboxSubprocessResult:
            launched.append(request.label)
            return SandboxSubprocessResult(exit_code=0)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            executable = root / "shasum"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o700)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )
            plan = await normalize_shell_execution_request(
                _request("visible.bin", algorithm="256"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(executable=str(executable)),
                ),
                sandbox_settings=_sandbox_settings(
                    root,
                    deny_roots=("/System/Library/Perl",),
                    trusted_executable=str(executable),
                    pids=None,
                ),
            )
            assert plan.sandbox_plan is not None
            result = await SeatbeltSandboxBackend(
                sandbox_executable="/fake/sandbox-exec",
                host_os="darwin",
                architecture="arm64",
                executable_available=True,
                command_runner=runner,
            ).execute(plan.sandbox_plan)

        self.assertEqual(result.status.value, SandboxResultStatus.DENIED.value)
        self.assertEqual(
            result.diagnostics[0].code.value,
            SandboxBackendDiagnosticCode.PATH_DENIED.value,
        )
        self.assertIn(
            "/System/Library/Perl",
            result.diagnostics[0].message,
        )
        self.assertNotIn("seatbelt_execute", launched)


@skipUnless(
    environ.get(_SEATBELT_SMOKE_ENV_NAME) == "1",
    f"Set {_SEATBELT_SMOKE_ENV_NAME}=1 to run the real Seatbelt shasum smoke.",
)
class ShasumSeatbeltRuntimeIntegrationTest(IsolatedAsyncioTestCase):
    async def test_real_seatbelt_hashes_with_system_perl_runtime(self) -> None:
        content = b"avalan seatbelt shasum\n"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(content)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
                executable_paths={"shasum": "/usr/bin/shasum"},
            )
            sandbox_settings = _sandbox_settings(
                root,
                trusted_executable="/usr/bin/shasum",
                pids=None,
            )
            tool = ShasumTool(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
                executor=ShellSandboxCommandExecutor(
                    settings=settings,
                    sandbox_settings=sandbox_settings,
                    sandbox_backend=SeatbeltSandboxBackend(),
                ),
            )

            output = await tool(
                ("visible.bin",),
                algorithm="256",
                context=ToolCallContext(),
            )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(
            result.status,
            ShellExecutionStatus.COMPLETED,
            msg=f"{result.error_message}: {result.stderr}",
        )
        self.assertEqual(
            result.stdout,
            f"{sha256(content).hexdigest()}  visible.bin\n",
        )
        self.assertEqual(result.metadata["sandbox_backend"], "seatbelt")


def _tool(root: Path, executable: Path) -> ShasumTool:
    settings = ShellToolSettings(
        workspace_root=str(root),
        executable_paths={"shasum": str(executable)},
    )
    return ShasumTool(
        settings=settings,
        policy=ExecutionPolicy(settings=settings),
        executor=LocalCommandExecutor(settings=settings),
    )


def _request(
    *paths: str,
    algorithm: str,
    cwd: str | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.shasum",
        command="shasum",
        options={"algorithm": algorithm},
        paths=tuple(
            PathOperand(
                name=f"path_{index}",
                path=path,
                kind="file",
                access="read",
            )
            for index, path in enumerate(paths)
        ),
        cwd=cwd,
    )


def _fake_shasum(root: Path) -> tuple[Path, Path]:
    executable = root / "shasum"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        '  "-a 256 -- binary.bin visible.txt")\n'
        '    printf "06eb7d6a69ee19e5fbdfcb34ebd3d2d2'
        'b6b17b2405873719fea5e02afac4b147  binary.bin\\n"\n'
        f'    printf "{_SHA256}  visible.txt\\n" ;;\n'
        '  "-a 256 -- ./-first.bin second file.txt")\n'
        '    printf "7d35fdd25c2a72ad104a3083a200bb13'
        '132165071ff7f5da51e515a73a228c39  ./-first.bin\\n"\n'
        f'    printf "{_SHA256}  second file.txt\\n" ;;\n'
        '  *) printf "unexpected argv\\n" >&2; exit 64 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


def _container_settings() -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="shell-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=False,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=ContainerSettingsSource(
            surface=ContainerSurface.SDK,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        policy_version="shasum-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _sandbox_settings(
    root: Path,
    *,
    backend: SandboxBackend = SandboxBackend.SEATBELT,
    extra_read_roots: tuple[str, ...] = (),
    deny_roots: tuple[str, ...] = (),
    trusted_executable: str = "/trusted/bin/shasum",
    pids: int | None = 16,
) -> SandboxEffectiveSettings:
    resolved_root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=(trusted_executable,),
        read_roots=(str(resolved_root), *extra_read_roots),
        deny_roots=deny_roots,
        scratch_roots=(str(resolved_root / "scratch"),),
        output_roots=(),
        environment=SandboxEnvironmentPolicy(allowlist=("LC_ALL",)),
        resources=SandboxResourceLimits(timeout_seconds=30, pids=pids),
        output=SandboxOutputPolicy(
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
            max_artifact_bytes=0,
            allow_artifacts=False,
        ),
    )
    return SandboxEffectiveSettings(
        backend=backend,
        required=False,
        source=trusted_isolation_source("sdk"),
        policy_version="shasum-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


class _AllResolved:
    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        if self._executable is not None:
            return self._executable
        return f"/trusted/bin/{command.executable_name}"


if __name__ == "__main__":
    main()
