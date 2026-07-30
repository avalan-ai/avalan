from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

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
from avalan.tool.shell import (
    DateTool,
    ExecutionPolicy,
    LocalCommandExecutor,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSettings,
    normalize_shell_execution_request,
)

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"


class DateLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_local_subprocess_preserves_default_date_behavior(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_date(root)
            tool = _tool(root, executable)

            output = await tool(context=ToolCallContext())
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "Thu Jul 30 04:05:06 -03 2026\n")
        self.assertEqual(result.argv, ("date",))
        self.assertEqual(launched, "\n")

    async def test_local_subprocess_uses_fixed_utc_iso8601_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_date(root)
            tool = _tool(root, executable)

            output = await tool(
                utc=True,
                format="iso8601",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "2026-07-30T07:05:06+0000\n")
        self.assertEqual(
            result.argv,
            ("date", "-u", "+%Y-%m-%dT%H:%M:%S%z"),
        )
        self.assertEqual(launched, "-u +%Y-%m-%dT%H:%M:%S%z\n")

    async def test_local_subprocess_uses_validated_custom_format_argv(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_date(root)
            tool = _tool(root, executable)

            output = await tool(
                utc=True,
                custom_format="stamp=%Y-%m-%d %% %H:%M:%S %z",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.stdout,
            "stamp=2026-07-30 % 07:05:06 +0000\n",
        )
        self.assertEqual(
            result.argv,
            (
                "date",
                "-u",
                "+stamp=%Y-%m-%d %% %H:%M:%S %z",
            ),
        )
        self.assertEqual(
            launched,
            "-u +stamp=%Y-%m-%d %% %H:%M:%S %z\n",
        )

    async def test_invalid_direct_arguments_do_not_launch(self) -> None:
        cases = (
            ({"utc": "true"}, "invalid_option"),
            ({"format": "custom"}, "invalid_option"),
            ({"custom_format": 1}, "invalid_option"),
            ({"custom_format": ""}, "invalid_option"),
            ({"custom_format": "%Y\n"}, "invalid_option"),
            ({"custom_format": "café"}, "invalid_option"),
            ({"custom_format": "%_d"}, "invalid_option"),
            ({"custom_format": "x" * 129}, "argument_too_large"),
            (
                {"format": "date", "custom_format": "%Y"},
                "invalid_option",
            ),
        )
        for arguments, error_code in cases:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_date(root)
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
                self.assertEqual(
                    output.execution_result.error_code.value,
                    error_code,
                )


class DateBackendPlanningIntegrationTest(IsolatedAsyncioTestCase):
    async def test_container_plan_preserves_fixed_date_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(utc=True, output_format="iso8601"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_container_settings(),
            )

        expected = ("date", "-u", "+%Y-%m-%dT%H:%M:%S%z")
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace",
        )

    async def test_container_plan_preserves_custom_date_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(
                    utc=True,
                    output_format="default",
                    custom_format="container=%Y-%m-%dT%H:%M:%S%z",
                ),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_container_settings(),
            )

        expected = (
            "date",
            "-u",
            "+container=%Y-%m-%dT%H:%M:%S%z",
        )
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)

    async def test_sandbox_plan_preserves_fixed_date_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(utc=False, output_format="unix"),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        self.assertEqual(plan.local_spec.argv, ("date", "+%s"))
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/date", "+%s"),
        )
        self.assertEqual(plan.sandbox_plan.request.cwd, str(root.resolve()))

    async def test_sandbox_plan_preserves_custom_date_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )

            plan = await normalize_shell_execution_request(
                _request(
                    utc=False,
                    output_format="default",
                    custom_format="sandbox=%Y %% %z",
                ),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        self.assertEqual(
            plan.local_spec.argv,
            ("date", "+sandbox=%Y %% %z"),
        )
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/date", "+sandbox=%Y %% %z"),
        )


def _tool(root: Path, executable: Path) -> DateTool:
    settings = ShellToolSettings(
        workspace_root=str(root),
        executable_paths={"date": str(executable)},
    )
    return DateTool(
        settings=settings,
        policy=ExecutionPolicy(settings=settings),
        executor=LocalCommandExecutor(settings=settings),
    )


def _request(
    *,
    utc: bool,
    output_format: str,
    custom_format: str | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.date",
        command="date",
        options={
            "utc": utc,
            "format": output_format,
            "custom_format": custom_format,
        },
        paths=(),
        cwd=None,
    )


def _fake_date(root: Path) -> tuple[Path, Path]:
    executable = root / "date"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        '  "") printf "Thu Jul 30 04:05:06 -03 2026\\n" ;;\n'
        '  "-u +%Y-%m-%dT%H:%M:%S%z") '
        'printf "2026-07-30T07:05:06+0000\\n" ;;\n'
        '  "-u +stamp=%Y-%m-%d %% %H:%M:%S %z") '
        'printf "stamp=2026-07-30 %% 07:05:06 +0000\\n" ;;\n'
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
        policy_version="date-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _sandbox_settings(root: Path) -> SandboxEffectiveSettings:
    resolved_root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=("/trusted/bin/date",),
        read_roots=(str(resolved_root),),
        scratch_roots=(str(resolved_root / "scratch"),),
        output_roots=(),
        environment=SandboxEnvironmentPolicy(allowlist=("LC_ALL",)),
        resources=SandboxResourceLimits(timeout_seconds=30, pids=16),
        output=SandboxOutputPolicy(
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
            max_artifact_bytes=0,
            allow_artifacts=False,
        ),
    )
    return SandboxEffectiveSettings(
        backend=SandboxBackend.SEATBELT,
        required=False,
        source=trusted_isolation_source("sdk"),
        policy_version="date-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


if __name__ == "__main__":
    main()
