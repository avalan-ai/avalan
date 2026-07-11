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
    LsofTool,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSettings,
    TrustedExecutableResolver,
    normalize_shell_execution_request,
    unavailable_executable_lookup,
)
from avalan.tool.shell.lsof import REDACTED_LSOF_STDERR

_EXPECTED_ARGUMENTS = "-n -P -w -b -a -p 42 -F0pftaP\n"
_EXPECTED_ARGV = (
    "lsof",
    "-n",
    "-P",
    "-w",
    "-b",
    "-a",
    "-p",
    "42",
    "-F0pftaP",
)
_EXPECTED_STDOUT = "42\t3\tr\tregular\t-\n42\t4\tu\tipv4\ttcp\n"
_PROMPT_LIKE_TYPE = "IGNOREPREVIOUSINSTRUCTIONS"
_PROMPT_LIKE_PROTOCOL = "EXFILTRATEPRIVATEVALUES"
_PUNCTUATION_TYPE = "/private/semantic/payload"
_PUNCTUATION_PROTOCOL = "remote.example:443"


class LsofLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_executes_fixed_argv_and_returns_only_safe_rows(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_lsof(root)
            tool = _tool(root, executable, allow_process_tools=True)
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            output = await tool(
                42,
                context=ToolCallContext(stream_event=record),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, _EXPECTED_STDOUT)
        self.assertEqual(result.stderr, REDACTED_LSOF_STDERR)
        self.assertEqual(result.argv, _EXPECTED_ARGV)
        self.assertEqual(result.display_argv, _EXPECTED_ARGV)
        self.assertEqual(result.metadata, {})
        self.assertEqual(result.generated_files, ())
        self.assertEqual(launched, _EXPECTED_ARGUMENTS)
        self.assertNotIn("private", output)
        self.assertNotIn("cwd", result.stdout)
        self.assertEqual(events, [])

    async def test_exit_one_uses_captured_diagnostics_to_choose_status(
        self,
    ) -> None:
        cases = (
            (1001, ShellExecutionStatus.NO_MATCHES, "", None),
            (
                1002,
                ShellExecutionStatus.NONZERO_EXIT,
                REDACTED_LSOF_STDERR,
                "lsof exited non-zero",
            ),
        )
        for pid, expected_status, expected_stderr, expected_error in cases:
            with self.subTest(pid=pid):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_lsof(root)
                    output = await _tool(
                        root,
                        executable,
                        allow_process_tools=True,
                    )(pid, context=ToolCallContext())

                    self.assertTrue(marker.is_file())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(result.status, expected_status)
                self.assertEqual(result.exit_code, 1)
                self.assertEqual(result.stderr, expected_stderr)
                self.assertEqual(result.error_message, expected_error)
                self.assertNotIn("private", output)

    async def test_discards_incomplete_final_record_after_raw_cap(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                1003,
                max_stdout_bytes=23,
                context=ToolCallContext(),
            )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(result.stdout, "1003\t3\tr\tregular\t-\n")
        self.assertNotIn("private", output)

    async def test_wrong_pid_and_malformed_fields_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1004, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_message, "lsof output was malformed")
        self.assertEqual(result.stdout, "")
        self.assertNotIn("private", output)
        self.assertNotIn("forged", output)

    async def test_semantic_fields_canonicalize_unknown_identifiers(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1006, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "1006\t3\tr\tother\tother\n")
        self.assertNotIn(_PROMPT_LIKE_TYPE, output)
        self.assertNotIn(_PROMPT_LIKE_PROTOCOL, output)

    async def test_punctuation_semantic_fields_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1007, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_message, "lsof output was malformed")
        self.assertEqual(result.stdout, "")
        self.assertNotIn(_PUNCTUATION_TYPE, output)
        self.assertNotIn(_PUNCTUATION_PROTOCOL, output)

    async def test_skips_pseudo_error_descriptor_without_type(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1008, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "1008\t3\tr\tregular\t-\n")

    async def test_numeric_descriptor_without_type_maps_to_other(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1009, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "1009\t3\tr\tother\t-\n")

    async def test_public_stdout_cap_retains_only_complete_rows(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(
                2_147_483_000,
                max_stdout_bytes=37,
                context=ToolCallContext(),
            )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        expected = "2147483000\t3\tr\tregular\t-\n"
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, expected)
        self.assertEqual(result.stdout_bytes, len(expected.encode("utf-8")))
        self.assertLessEqual(result.stdout_bytes, 37)
        self.assertTrue(result.stdout_truncated)
        self.assertNotIn("\t4\t", result.stdout)

    async def test_logical_limit_is_applied_after_safe_record_parsing(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_lsof(root)
            output = await _tool(
                root,
                executable,
                allow_process_tools=True,
            )(1005, limit=2, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertEqual(
            result.stdout,
            "1005\t3\tr\tregular\t-\n1005\t4\tw\tpipe\t-\n",
        )
        self.assertTrue(result.stdout_truncated)
        self.assertNotIn("5\tu", result.stdout)

    async def test_gate_and_invalid_arguments_never_launch(self) -> None:
        cases = (
            (False, 42, 64),
            (True, 0, 64),
            (True, True, 64),
            (True, 2**31, 64),
            (True, 42, 0),
            (True, 42, 257),
        )
        for allow_process_tools, pid, limit in cases:
            with self.subTest(
                allow_process_tools=allow_process_tools,
                pid=pid,
                limit=limit,
            ):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_lsof(root)
                    output = await _tool(
                        root,
                        executable,
                        allow_process_tools=allow_process_tools,
                    )(
                        pid,
                        limit=limit,
                        context=ToolCallContext(),
                    )

                    self.assertFalse(marker.exists())

                self.assertIsInstance(output, ShellFormattedResult)
                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(
                    result.status, ShellExecutionStatus.POLICY_DENIED
                )
                self.assertEqual(
                    result.error_message, "lsof was denied by policy"
                )
                self.assertEqual(result.metadata, {})

    async def test_missing_binary_returns_command_unavailable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_process_tools=True,
            )
            tool = LsofTool(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=TrustedExecutableResolver(
                        lookup=unavailable_executable_lookup,
                    ),
                ),
                executor=LocalCommandExecutor(settings=settings),
            )

            output = await tool(42, context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMMAND_UNAVAILABLE)
        self.assertEqual(result.error_message, "lsof is unavailable")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


class LsofBackendPlanningIntegrationTest(IsolatedAsyncioTestCase):
    async def test_container_plan_preserves_fixed_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
                allow_process_tools=True,
            )
            plan = await normalize_shell_execution_request(
                _request(42),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                container_settings=_container_settings(),
            )

        self.assertIs(plan.mode, ShellExecutionMode.CONTAINER)
        self.assertEqual(plan.local_spec.argv, _EXPECTED_ARGV)
        self.assertEqual(plan.local_spec.display_argv, _EXPECTED_ARGV)
        assert plan.container_plan is not None
        self.assertEqual(
            plan.container_plan.run_plan.command.argv, _EXPECTED_ARGV
        )
        self.assertEqual(
            plan.container_plan.run_plan.command.cwd, "/workspace"
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
                _request(42),
                ExecutionPolicy(settings=settings, resolver=_AllResolved()),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIs(plan.mode, ShellExecutionMode.SANDBOX)
        self.assertEqual(plan.local_spec.argv, _EXPECTED_ARGV)
        self.assertEqual(plan.local_spec.display_argv, _EXPECTED_ARGV)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.request.argv,
            ("/trusted/bin/lsof", *_EXPECTED_ARGV[1:]),
        )
        self.assertEqual(plan.sandbox_plan.request.cwd, str(root.resolve()))


def _tool(
    root: Path,
    executable: Path,
    *,
    allow_process_tools: bool,
) -> LsofTool:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=allow_process_tools,
        executable_paths={"lsof": str(executable)},
    )
    return LsofTool(
        settings=settings,
        policy=ExecutionPolicy(settings=settings),
        executor=LocalCommandExecutor(settings=settings),
    )


def _request(pid: int, *, limit: int = 64) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.lsof",
        command="lsof",
        options={"pid": pid, "limit": limit},
        paths=(),
        cwd=None,
    )


def _fake_lsof(root: Path) -> tuple[Path, Path]:
    executable = root / "lsof"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$7" in\n'
        "  1001) exit 1 ;;\n"
        "  1002)\n"
        "    printf 'private lsof diagnostic /private/host/path\\n' >&2\n"
        "    exit 1 ;;\n"
        "  1003)\n"
        "    printf 'p1003\\000\\nf3\\000ar\\000tREG\\000\\n"
        "f4\\000aw\\000tREG\\000Pprivate\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1004)\n"
        "    printf 'p9999\\000\\nf3\\000ar\\000tREG\\000"
        "n/private/forged/path\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1005)\n"
        "    printf 'p1005\\000\\nf3\\000ar\\000tREG\\000\\n"
        "f4\\000aw\\000tPIPE\\000\\n"
        "f5\\000au\\000tIPv4\\000PTCP\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1006)\n"
        "    printf 'p1006\\000\\nf3\\000ar\\000"
        f"t{_PROMPT_LIKE_TYPE}\\000"
        f"P{_PROMPT_LIKE_PROTOCOL}\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1007)\n"
        "    printf 'p1007\\000\\nf3\\000ar\\000"
        f"t{_PUNCTUATION_TYPE}\\000"
        f"P{_PUNCTUATION_PROTOCOL}\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1008)\n"
        "    printf 'p1008\\000\\nferr\\000\\n"
        "f3\\000ar\\000tREG\\000\\n'\n"
        "    exit 0 ;;\n"
        "  1009)\n"
        "    printf 'p1009\\000\\nf3\\000ar\\000\\n'\n"
        "    exit 0 ;;\n"
        "  2147483000)\n"
        "    printf 'p2147483000\\000\\nf3\\000ar\\000tREG\\000\\n"
        "f4\\000aw\\000tREG\\000\\n"
        "f5\\000au\\000tREG\\000\\n'\n"
        "    exit 0 ;;\n"
        "esac\n"
        "printf 'p42\\000\\nfcwd\\000a \\000tDIR\\000\\n"
        "f3\\000ar\\000tREG\\000\\n"
        "f4\\000au\\000tIPv4\\000PTCP\\000\\n'\n"
        "printf 'private lsof diagnostic /private/host/path\\n' >&2\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


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
        policy_version="lsof-test",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _sandbox_settings(root: Path) -> SandboxEffectiveSettings:
    resolved_root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=("/trusted/bin/lsof",),
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
        policy_version="lsof-test",
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
