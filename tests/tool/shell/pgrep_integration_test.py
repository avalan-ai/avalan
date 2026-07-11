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
from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
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
from avalan.tool.shell.pgrep import (
    REDACTED_PGREP_PATTERN,
    REDACTED_PGREP_STDERR,
)
from avalan.tool.shell.tools import PgrepTool

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"
PRIVATE_PATTERN = "private-integration-worker"


class PgrepLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_local_subprocess_uses_raw_query_and_returns_safe_result(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_pgrep(root)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_process_tools=True,
                executable_paths={"pgrep": str(executable)},
            )
            tool = PgrepTool(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
                executor=LocalCommandExecutor(settings=settings),
            )
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            output = await tool(
                PRIVATE_PATTERN,
                full=True,
                parent_pid=42,
                context=ToolCallContext(stream_event=record),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "4242\n14242\n")
        self.assertEqual(result.stderr, REDACTED_PGREP_STDERR)
        self.assertEqual(
            result.argv,
            (
                "pgrep",
                "-f",
                "-P",
                "42",
                "--",
                REDACTED_PGREP_PATTERN,
            ),
        )
        self.assertIn(PRIVATE_PATTERN, launched)
        self.assertIn("-f -P 42 --", launched)
        self.assertNotIn(PRIVATE_PATTERN, output)
        self.assertNotIn("worker process name", output)
        self.assertEqual(events, [])

    async def test_local_subprocess_maps_no_match_and_redacts_errors(
        self,
    ) -> None:
        cases = (
            (
                "no-match-process",
                ShellExecutionStatus.NO_MATCHES,
                1,
                "",
            ),
            (
                "tool-error-process",
                ShellExecutionStatus.NONZERO_EXIT,
                2,
                REDACTED_PGREP_STDERR,
            ),
        )
        for pattern, expected_status, expected_exit, expected_stderr in cases:
            with self.subTest(pattern=pattern):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_pgrep(root)
                    settings = ShellToolSettings(
                        workspace_root=str(root),
                        allow_process_tools=True,
                        executable_paths={"pgrep": str(executable)},
                    )
                    tool = PgrepTool(
                        settings=settings,
                        policy=ExecutionPolicy(settings=settings),
                        executor=LocalCommandExecutor(settings=settings),
                    )

                    output = await tool(
                        pattern,
                        context=ToolCallContext(),
                    )

                    self.assertTrue(marker.is_file())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(result.status, expected_status)
                self.assertEqual(result.exit_code, expected_exit)
                self.assertEqual(result.stderr, expected_stderr)
                self.assertNotIn(pattern, output)


class PgrepBackendPlanningIntegrationTest(IsolatedAsyncioTestCase):
    async def test_container_plan_keeps_raw_backend_relative_argv(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request(PRIVATE_PATTERN, full=True),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_container_settings(),
            )

        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(
            plan.local_spec.argv,
            ("pgrep", "-f", "--", PRIVATE_PATTERN),
        )
        self.assertEqual(
            plan.local_spec.display_argv,
            ("pgrep", "-f", "--", REDACTED_PGREP_PATTERN),
        )
        self.assertIsNotNone(plan.container_plan)
        assert plan.container_plan is not None
        self.assertEqual(
            plan.container_plan.run_plan.command.argv,
            ("pgrep", "-f", "--", PRIVATE_PATTERN),
        )
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace",
        )

    async def test_sandbox_plan_keeps_raw_backend_relative_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request(PRIVATE_PATTERN, exact=True),
                ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        self.assertEqual(
            plan.local_spec.argv,
            ("pgrep", "-x", "--", PRIVATE_PATTERN),
        )
        self.assertEqual(
            plan.local_spec.display_argv,
            ("pgrep", "-x", "--", REDACTED_PGREP_PATTERN),
        )
        self.assertIsNotNone(plan.sandbox_plan)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/pgrep", "-x", "--", PRIVATE_PATTERN),
        )
        self.assertEqual(plan.sandbox_plan.request.cwd, str(root.resolve()))


def _request(
    pattern: str,
    *,
    full: bool = False,
    exact: bool = False,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.pgrep",
        command="pgrep",
        options={"pattern": pattern, "full": full, "exact": exact},
        paths=(),
        cwd=None,
    )


def _fake_pgrep(root: Path) -> tuple[Path, Path]:
    executable = root / "pgrep"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        "  *no-match*) exit 1 ;;\n"
        "  *tool-error*)\n"
        "    printf 'private diagnostic %s\\n' \"$*\" >&2\n"
        "    exit 2\n"
        "    ;;\n"
        "esac\n"
        "printf '4242\\nworker process name\\n14242\\n'\n"
        "printf 'private diagnostic %s\\n' \"$*\" >&2\n",
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
        policy_version="pgrep-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _sandbox_settings(root: Path) -> SandboxEffectiveSettings:
    resolved_root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=("/trusted/bin/pgrep",),
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
        policy_version="pgrep-test",
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
