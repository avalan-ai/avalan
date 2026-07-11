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
    PsTool,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSettings,
    normalize_shell_execution_request,
)
from avalan.tool.shell.ps import REDACTED_PS_STDERR


class PsLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_executes_fixed_argv_and_sanitizes_public_result(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_process_tools=True,
                executable_paths={"ps": str(executable)},
            )
            tool = PsTool(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
                executor=LocalCommandExecutor(settings=settings),
            )
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            output = await tool(
                (42,),
                context=ToolCallContext(stream_event=record),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "42 1 S 01:02 /usr/bin/worker\n")
        self.assertEqual(result.stderr, REDACTED_PS_STDERR)
        self.assertEqual(result.metadata, {})
        self.assertEqual(result.generated_files, ())
        self.assertEqual(
            launched,
            "-p 42 -o pid= -o ppid= -o state= -o etime= -o comm=\n",
        )
        self.assertEqual(events, [])

    async def test_executes_fixed_resource_view_and_hides_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                (42,),
                view="resources",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.stdout,
            "42 12.5 3.0 1024 4096 0:00.02 -1\n",
        )
        self.assertEqual(result.stderr, REDACTED_PS_STDERR)
        self.assertEqual(result.metadata, {})
        self.assertEqual(
            launched,
            "-p 42 -o pid= -o pcpu= -o pmem= -o rss= -o vsz= "
            "-o time= -o nice=\n",
        )

    async def test_legacy_positional_cwd_keeps_summary_contract(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "subdir").mkdir()
            executable, marker = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                (42,),
                "subdir",
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        self.assertIs(
            output.execution_result.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertEqual(
            output.execution_result.stdout,
            "42 1 S 01:02 /usr/bin/worker\n",
        )
        self.assertNotIn("pcpu=", launched)

    async def test_maps_exit_statuses_and_redacts_diagnostics(self) -> None:
        cases = (
            (1001, ShellExecutionStatus.NO_MATCHES, 1, None),
            (1002, ShellExecutionStatus.NONZERO_EXIT, 2, "ps exited non-zero"),
        )
        for pid, expected_status, expected_exit, expected_error in cases:
            with self.subTest(pid=pid):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, _ = _fake_ps(root)
                    tool = _tool(root, executable, allow_process_tools=True)
                    output = await tool((pid,), context=ToolCallContext())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(result.status, expected_status)
                self.assertEqual(result.exit_code, expected_exit)
                self.assertEqual(result.error_message, expected_error)
                self.assertNotIn("private", output)
                expected_stderr = "" if pid == 1001 else REDACTED_PS_STDERR
                self.assertEqual(result.stderr, expected_stderr)

    async def test_discards_malformed_and_truncated_final_rows(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                (1003,),
                max_stdout_bytes=30,
                context=ToolCallContext(),
            )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(result.stdout, "1003 1 S 00:01 worker\n")
        self.assertNotIn("secret", output)

    async def test_rejects_unrequested_and_newline_forged_rows(self) -> None:
        for pid in (1004, 1005, 1006):
            with self.subTest(pid=pid):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, _ = _fake_ps(root)
                    output = await _tool(
                        root,
                        executable,
                        allow_process_tools=True,
                    )((pid,), context=ToolCallContext())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                self.assertEqual(output.execution_result.stdout, "")
                self.assertNotIn("forged", output)

    async def test_default_process_gate_denies_without_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=False,
            )((1,), context=ToolCallContext())

            self.assertFalse(marker.exists())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(result.error_message, "ps was denied by policy")
        self.assertEqual(result.metadata, {})

    async def test_multiple_pids_are_denied_without_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )((1, 2), context=ToolCallContext())

            self.assertFalse(marker.exists())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(result.error_message, "ps was denied by policy")

    async def test_invalid_view_is_denied_without_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                (42,),
                view="private",  # type: ignore[arg-type]
                context=ToolCallContext(),
            )

            self.assertFalse(marker.exists())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(result.error_message, "ps was denied by policy")
        self.assertEqual(result.metadata, {})


class PsBackendPlanningIntegrationTest(IsolatedAsyncioTestCase):
    async def test_container_plan_preserves_fixed_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request((42,)),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                container_settings=_container_settings(),
            )

        expected = _expected_argv((42,))
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd,
            "/workspace",
        )

    async def test_sandbox_plan_preserves_fixed_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request((42,)),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        expected = _expected_argv((42,))
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/ps", *expected[1:]),
        )
        self.assertEqual(plan.sandbox_plan.request.cwd, str(root.resolve()))

    async def test_container_resource_plan_preserves_fixed_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request((42,), view="resources"),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                container_settings=_container_settings(),
            )

        expected = _expected_argv((42,), view="resources")
        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.run_plan.command.argv, expected)

    async def test_sandbox_resource_plan_preserves_fixed_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request((42,), view="resources"),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                sandbox_settings=_sandbox_settings(root),
            )

        expected = _expected_argv((42,), view="resources")
        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        self.assertEqual(plan.local_spec.argv, expected)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/ps", *expected[1:]),
        )


def _fake_ps(root: Path) -> tuple[Path, Path]:
    executable = root / "ps"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        "  *pcpu=*)\n"
        "    printf ' 42 12.5 3.0 1024 4096 0:00.02 -1\\n'\n"
        "    printf 'private process diagnostic\\n' >&2\n"
        "    exit 0 ;;\n"
        "esac\n"
        'case "$2" in\n'
        "  1001) exit 1 ;;\n"
        "  1002) printf 'private failure detail\\n' >&2; exit 2 ;;\n"
        "  1003)\n"
        "    printf '1003 1 S 00:01 worker\\n"
        "secret malformed row that is long\\n'\n"
        "    exit 0\n"
        "    ;;\n"
        "  1004)\n"
        "    printf '1004 1 S 00:01 worker\\n99 1 R 00:01 forged\\n'\n"
        "    exit 0 ;;\n"
        "  1005)\n"
        "    printf '1005 1 S 00:01 worker\\n"
        "1005 1 R 00:01 forged\\n'\n"
        "    exit 0 ;;\n"
        "  1006)\n"
        "    printf '1006 1 S 00:01\\n1006 1 R 00:02 forged\\n'\n"
        "    exit 0 ;;\n"
        "esac\n"
        "printf ' 42 1 S 01:02 /usr/bin/worker\\n'\n"
        "printf 'private process diagnostic\\n' >&2\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


def _tool(
    root: Path,
    executable: Path,
    *,
    allow_process_tools: bool,
) -> PsTool:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=allow_process_tools,
        executable_paths={"ps": str(executable)},
    )
    return PsTool(
        settings=settings,
        policy=ExecutionPolicy(settings=settings),
        executor=LocalCommandExecutor(settings=settings),
    )


def _request(
    pids: tuple[int, ...],
    *,
    view: str = "summary",
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.ps",
        command="ps",
        options={"pids": pids, "view": view},
        paths=(),
        cwd=None,
    )


def _expected_argv(
    pids: tuple[int, ...],
    *,
    view: str = "summary",
) -> tuple[str, ...]:
    fields = (
        ("pid", "ppid", "state", "etime", "comm")
        if view == "summary"
        else ("pid", "pcpu", "pmem", "rss", "vsz", "time", "nice")
    )
    argv = [
        "ps",
        "-p",
        ",".join(str(pid) for pid in pids),
    ]
    for field in fields:
        argv.extend(("-o", f"{field}="))
    return tuple(argv)


def _container_settings() -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="shell-readonly",
        image_reference=f"ghcr.io/example/shell-tools@sha256:{'9' * 64}",
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=False,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=ContainerSettingsSource(
            surface=ContainerSurface.SDK,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        policy_version="ps-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _sandbox_settings(root: Path) -> SandboxEffectiveSettings:
    resolved_root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=("/trusted/bin/ps",),
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
        policy_version="ps-test",
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
