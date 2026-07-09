from asyncio import create_task, gather, sleep, wait_for
from asyncio import run as async_run
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.entities import (
    ToolCallContext,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.isolation import (
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationToolRuntimeSettings,
    SandboxBackend,
    SandboxEffectiveSettings,
    SandboxEnvironmentPolicy,
    SandboxNetworkMode,
    SandboxOutputPolicy,
    SandboxProfile,
    SandboxResourceLimits,
    trusted_isolation_source,
)
from avalan.sandbox import backend as sandbox_backend_module
from avalan.tool import Tool
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionSpec,
    GeneratedOutputPlan,
    LocalCommandExecutor,
    LocalCompositionExecutor,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionMode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellSandboxCommandExecutor,
    ShellToolSet,
    ShellToolSettings,
    lower_shell_execution_spec,
    normalize_shell_execution_request,
)
from avalan.tool.shell import sandbox as shell_sandbox_module
from avalan.tool.shell.entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ShellFormattedCompositionResult,
    ShellFormattedResult,
)
from avalan.tool.shell.sandbox import (
    _diagnostic_summary as _sandbox_diagnostic_summary,
)
from avalan.tool.shell.sandbox import (
    _generated_files as _sandbox_generated_files,
)
from avalan.tool.shell.sandbox import (
    _scrub_capture as _sandbox_scrub_capture,
)


class ShellSandboxPlanningTest(IsolatedAsyncioTestCase):
    async def test_local_and_sandbox_plan_lowering(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            local_settings = ShellToolSettings(workspace_root=str(root))
            sandbox_settings = ShellToolSettings(
                execution_mode="sandbox",
                workspace_root=str(root),
            )
            local_policy = ExecutionPolicy(
                settings=local_settings,
                resolver=_AllResolved(),
            )
            sandbox_policy = ExecutionPolicy(
                settings=sandbox_settings,
                resolver=_AllResolved(),
            )
            request = _cat_request("visible.txt")

            local = await normalize_shell_execution_request(
                request,
                local_policy,
            )
            sandbox = await normalize_shell_execution_request(
                request,
                sandbox_policy,
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertEqual(local.mode, ShellExecutionMode.LOCAL)
        self.assertIsNone(local.sandbox_plan)
        self.assertEqual(sandbox.mode, ShellExecutionMode.SANDBOX)
        self.assertIsNotNone(sandbox.sandbox_plan)
        assert sandbox.sandbox_plan is not None
        self.assertEqual(
            sandbox.sandbox_plan.request.argv,
            ("/trusted/bin/cat", "--", "visible.txt"),
        )
        self.assertEqual(
            sandbox.sandbox_plan.request.cwd,
            str(root.resolve()),
        )
        self.assertIn("sandbox", sandbox.to_dict())

    async def test_required_sandbox_plan_never_lowers_to_local(self) -> None:
        spec = _direct_text_spec(Path.cwd())

        result = await ShellSandboxCommandExecutor(
            sandbox_settings=_sandbox_settings(Path.cwd(), required=True),
            sandbox_backend=None,
        ).execute(spec)

        self.assertEqual(result.backend, "sandbox")
        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIn("no backend", result.error_message or "")

    async def test_sandbox_mode_without_effective_settings_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    execution_mode="sandbox",
                    workspace_root=str(root),
                ),
                resolver=_AllResolved(),
            )

            with self.assertRaises(AssertionError):
                await normalize_shell_execution_request(
                    _cat_request("visible.txt"),
                    policy,
                )

    async def test_sandbox_plan_uses_shell_timeout_without_profile_cap(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = lower_shell_execution_spec(
                _direct_text_spec(root, timeout_seconds=3.4),
                sandbox_settings=_sandbox_settings(root, timeout_seconds=None),
            )

        self.assertIsNotNone(plan.sandbox_plan)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.settings.profile.resources.timeout_seconds,
            3,
        )


class ShellSandboxExecutorTest(IsolatedAsyncioTestCase):
    async def test_sandbox_executes_readonly_command_and_streams(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = sandbox_backend_module.SandboxFakeBackend(
                sandbox_backend_module.SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        sandbox_backend_module.SandboxStreamChunk(
                            stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                            content=b"hello\n",
                            sequence=0,
                        ),
                        sandbox_backend_module.SandboxStreamChunk(
                            stream=sandbox_backend_module.SandboxBackendStream.STDERR,
                            content=b"warn\n",
                            sequence=1,
                        ),
                    ),
                )
            )
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(Path(temporary_directory)),
                sandbox_backend=backend,
            ).execute(_direct_text_spec(root), stream=record)

        self.assertEqual(result.backend, "sandbox")
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.stderr, "warn\n")
        self.assertEqual(result.stdout_bytes, 6)
        self.assertEqual(result.stderr_bytes, 5)
        self.assertEqual(
            [event.kind for event in events],
            [
                ToolExecutionStreamKind.STDOUT,
                ToolExecutionStreamKind.STDERR,
            ],
        )
        self.assertIn(
            sandbox_backend_module.SandboxBackendOperation.PROBE,
            backend.operations,
        )
        self.assertIn(
            sandbox_backend_module.SandboxBackendOperation.START,
            backend.operations,
        )
        self.assertIn("sandbox_plan_fingerprint", result.metadata)

    async def test_sandbox_collects_generated_outputs(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = _PlanAwareGeneratedSandboxBackend(b"value")
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root, output=True),
                sandbox_backend=backend,
            ).execute(
                _direct_generated_spec(
                    root,
                    display_prefix="GENERATED_PREFIX",
                ),
                stream=record,
            )
            runtime_path = backend.runtime_prefix or ""
            output_dir = backend.output_dir

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.output_kind, ShellOutputKind.GENERATED_FILES)
        self.assertEqual(len(result.generated_files), 1)
        generated = result.generated_files[0]
        self.assertEqual(generated.display_path, "GENERATED_PREFIX.txt")
        self.assertEqual(generated.media_type, "text/plain")
        self.assertEqual(generated.bytes, 5)
        self.assertEqual(generated.content_base64, "dmFsdWU=")
        self.assertIn("GENERATED_PREFIX", result.stdout)
        self.assertIn("GENERATED_PREFIX", result.stderr)
        self.assertNotIn(runtime_path, result.stdout)
        self.assertNotIn(runtime_path, result.stderr)
        self.assertIsNotNone(output_dir)
        assert output_dir is not None
        self.assertFalse(output_dir.exists())
        self.assertEqual(output_dir.parent, root.resolve() / "outputs")
        self.assertTrue(output_dir.name.startswith("avalan-shell-"))
        self.assertTrue(events)
        for event in events:
            self.assertNotIn(runtime_path, event.content)

    async def test_generated_outputs_without_output_root_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root, output=False),
                sandbox_backend=None,
            ).execute(_direct_generated_spec(root))

        self.assertEqual(result.backend, "sandbox")
        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertIn("output root", result.error_message or "")

    async def test_generated_output_preparation_failure_fails_closed(
        self,
    ) -> None:
        async def fail_ensure_directory(path: Path) -> None:
            raise OSError(path)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(
                shell_sandbox_module,
                "_ensure_directory",
                fail_ensure_directory,
            ):
                result = await ShellSandboxCommandExecutor(
                    sandbox_settings=_sandbox_settings(root),
                    sandbox_backend=None,
                ).execute(_direct_generated_spec(root))

        self.assertEqual(result.backend, "sandbox")
        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertEqual(
            result.error_message,
            "generated output preparation failed",
        )

    async def test_sandbox_materializes_non_inline_generated_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = _PlanAwareGeneratedSandboxBackend(b"larger")

            result = await ShellSandboxCommandExecutor(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    materialized_input_files_dir="generated-files",
                ),
                sandbox_settings=_sandbox_settings(root, output=True),
                sandbox_backend=backend,
            ).execute(
                _direct_generated_spec(
                    root,
                    display_prefix="GENERATED_PREFIX",
                    max_inline_bytes=1,
                ),
            )

            generated = result.generated_files[0]
            materialized_path = Path(
                str(
                    generated.metadata[
                        GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY
                    ]
                )
            )
            self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
            self.assertIsNone(generated.content_base64)
            self.assertEqual(materialized_path.read_bytes(), b"larger")
            self.assertTrue(
                materialized_path.is_relative_to(
                    (root / "generated-files").resolve(),
                )
            )

    async def test_sandbox_scrubs_truncated_generated_output_path(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cap = len('{"output": "') + len("GENERATED_PREFIX")
            backend = _PlanAwareGeneratedSandboxBackend(
                b"value",
                stderr=False,
            )
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root, output=True),
                sandbox_backend=backend,
            ).execute(
                _direct_generated_spec(
                    root,
                    display_prefix="GENERATED_PREFIX",
                    max_stdout_bytes=cap,
                ),
                stream=record,
            )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(result.stdout_truncated)
        self.assertIn("GENERATED_PREFIX", result.stdout)
        self.assertNotIn(str(root.resolve()), result.stdout)
        self.assertLessEqual(len(result.stdout.encode()), result.stdout_bytes)
        self.assertTrue(events)
        self.assertIn("GENERATED_PREFIX", events[0].content)
        self.assertNotIn(str(root.resolve()), events[0].content)

    async def test_sandbox_stream_caps_are_shell_visible(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root),
                sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        stream_chunks=(
                            sandbox_backend_module.SandboxStreamChunk(
                                stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                                content=b"abcdefghijklmnopqrstuvwxyz",
                                sequence=0,
                            ),
                        ),
                    )
                ),
            ).execute(_direct_text_spec(root, max_stdout_bytes=3))

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "abc")
        self.assertEqual(result.stdout_bytes, 3)
        self.assertTrue(result.stdout_truncated)

    async def test_negative_sandbox_runtime_failures(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases = (
                (
                    "unavailable backend",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        available=False,
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.TOOL_ERROR,
                ),
                (
                    "policy denial",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        denied_paths=(str(root.resolve()),),
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.POLICY_DENIED,
                ),
                (
                    "timeout",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        timeout_operations=(
                            sandbox_backend_module.SandboxBackendOperation.WAIT,
                        ),
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.TIMEOUT,
                ),
                (
                    "cancellation",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        cancel_operations=(
                            sandbox_backend_module.SandboxBackendOperation.WAIT,
                        ),
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.CANCELLED,
                ),
                (
                    "non-zero exit",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        wait_exit_code=7,
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.NONZERO_EXIT,
                ),
                (
                    "mapped missing command",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        wait_exit_code=127,
                    ),
                    _direct_text_spec(
                        root,
                        metadata={
                            "exit_code_statuses": {
                                127: "command_unavailable",
                            }
                        },
                    ),
                    ShellExecutionStatus.COMMAND_UNAVAILABLE,
                ),
                (
                    "mapped timeout status",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        wait_exit_code=42,
                    ),
                    _direct_text_spec(
                        root,
                        metadata={
                            "exit_code_statuses": {
                                42: "timeout",
                            }
                        },
                    ),
                    ShellExecutionStatus.TIMEOUT,
                ),
                (
                    "oversized output",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        stream_chunks=(),
                        output_files={"shell-output.txt": b"x" * 65},
                    ),
                    _direct_generated_spec(root),
                    ShellExecutionStatus.TOO_LARGE,
                ),
                (
                    "unsafe output",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        stream_chunks=(),
                        output_files={"../secret.txt": b"x"},
                    ),
                    _direct_generated_spec(root),
                    ShellExecutionStatus.TOOL_ERROR,
                ),
                (
                    "cleanup failure",
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        cleanup_uncertain=True,
                    ),
                    _direct_text_spec(root),
                    ShellExecutionStatus.TOOL_ERROR,
                ),
            )

            for name, script, spec, expected_status in cases:
                with self.subTest(name=name):
                    result = await ShellSandboxCommandExecutor(
                        sandbox_settings=_sandbox_settings(root),
                        sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                            script
                        ),
                    ).execute(spec)

                    self.assertEqual(result.backend, "sandbox")
                    self.assertEqual(result.status, expected_status)
                    if name == "mapped missing command":
                        self.assertEqual(
                            result.error_code,
                            ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
                        )
                        self.assertEqual(
                            result.error_message,
                            "command is unavailable",
                        )
                    if name == "mapped timeout status":
                        self.assertEqual(
                            result.error_code,
                            ShellExecutionErrorCode.TIMEOUT,
                        )
                        self.assertEqual(
                            result.error_message,
                            "sandbox command exited with timeout",
                        )
                    if name == "unavailable backend":
                        self.assertEqual(
                            result.metadata["isolation_diagnostic_codes"],
                            ("sandbox.provider_unavailable",),
                        )

    async def test_negative_missing_backend_and_executable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = _sandbox_settings(root)

            no_settings = await ShellSandboxCommandExecutor(
                sandbox_settings=None,
                sandbox_backend=None,
            ).execute(_direct_text_spec(root))
            no_backend = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=None,
            ).execute(_direct_text_spec(root))
            missing_executable = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities()
                    )
                ),
            ).execute(_direct_text_spec(root, executable=None))

        self.assertEqual(no_settings.backend, "sandbox")
        self.assertEqual(no_settings.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIn("no settings", no_settings.error_message or "")
        self.assertEqual(no_backend.backend, "sandbox")
        self.assertEqual(no_backend.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIn("no backend", no_backend.error_message or "")
        self.assertEqual(
            missing_executable.status,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
        )

    async def test_negative_backend_exceptions_return_shell_results(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = _sandbox_settings(root)
            script = sandbox_backend_module.SandboxFakeBackendScript(
                capabilities=_capabilities(),
            )

            probe_result = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=cast(
                    sandbox_backend_module.SandboxAsyncBackend,
                    _ProbeRaisesBackend(),
                ),
            ).execute(_direct_text_spec(root))
            execute_result = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=cast(
                    sandbox_backend_module.SandboxAsyncBackend,
                    _ExecuteRaisesBackend(script),
                ),
            ).execute(_direct_text_spec(root))
            diagnostic_probe_result = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=cast(
                    sandbox_backend_module.SandboxAsyncBackend,
                    _ProbeDiagnosticBackend(),
                ),
            ).execute(_direct_text_spec(root))
            diagnostic_execute_result = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=cast(
                    sandbox_backend_module.SandboxAsyncBackend,
                    _ExecuteDiagnosticBackend(script),
                ),
            ).execute(_direct_text_spec(root))
            malformed_diagnostic_result = await ShellSandboxCommandExecutor(
                sandbox_settings=settings,
                sandbox_backend=cast(
                    sandbox_backend_module.SandboxAsyncBackend,
                    _ProbeMalformedDiagnosticBackend(),
                ),
            ).execute(_direct_text_spec(root))

        self.assertEqual(probe_result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            probe_result.error_message,
            "sandbox execution failed: sandbox.backend.unavailable",
        )
        self.assertEqual(
            probe_result.metadata["isolation_diagnostic_codes"],
            ("sandbox.provider_unavailable",),
        )
        self.assertEqual(
            execute_result.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertEqual(
            execute_result.error_message,
            "sandbox execution failed: sandbox.backend.execution_failed",
        )
        self.assertEqual(
            execute_result.metadata["isolation_diagnostic_codes"],
            ("isolation.mode_unavailable",),
        )
        self.assertEqual(
            diagnostic_probe_result.error_message,
            "sandbox execution failed: sandbox.backend.execution_failed",
        )
        self.assertEqual(
            diagnostic_execute_result.metadata["sandbox_diagnostic_codes"],
            ("sandbox.backend.execution_failed",),
        )
        self.assertEqual(
            malformed_diagnostic_result.error_message,
            "sandbox execution failed: sandbox.backend.unavailable",
        )
        self.assertEqual(
            malformed_diagnostic_result.metadata["isolation_diagnostic_codes"],
            ("sandbox.provider_unavailable",),
        )

    async def test_negative_unmapped_cwd_and_unsupported_output_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root.parent
            executor = ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root),
                sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities()
                    )
                ),
            )

            unmapped = await executor.execute(
                _direct_text_spec(root, cwd=str(outside.resolve()))
            )
            unsupported = await executor.execute(
                _direct_generated_spec(root, output_plan=False)
            )

        self.assertEqual(unmapped.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(unsupported.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            unsupported.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )

    async def test_negative_generic_sandbox_backend_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root),
                sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        operation_diagnostics={
                            (
                                sandbox_backend_module.SandboxBackendOperation.WAIT
                            ): (
                                sandbox_backend_module.SandboxBackendDiagnosticCode.EXECUTION_FAILED
                            )
                        },
                    )
                ),
            ).execute(_direct_text_spec(root))

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIn(
            "sandbox.backend.execution_failed",
            result.error_message or "",
        )

    async def test_sandbox_generated_output_validation_failures(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases = (
                (
                    "unsupported suffix",
                    {"shell-output.bin": b"value"},
                    _direct_generated_spec(root),
                    ShellExecutionStatus.TOO_LARGE,
                ),
                (
                    "file count cap",
                    {
                        "shell-output-1.txt": b"one",
                        "shell-output-2.txt": b"two",
                    },
                    _direct_generated_spec(root, max_files=1),
                    ShellExecutionStatus.TOO_LARGE,
                ),
                (
                    "ignored output after file count cap",
                    {
                        "shell-output-1.txt": b"one",
                        "other.txt": b"skip",
                    },
                    _direct_generated_spec(root, max_files=1),
                    ShellExecutionStatus.COMPLETED,
                ),
                (
                    "nested output",
                    {
                        "other.txt": b"skip",
                        "nested/shell-output.txt": b"value",
                    },
                    _direct_generated_spec(root, max_inline_bytes=1),
                    ShellExecutionStatus.COMPLETED,
                ),
                (
                    "bare prefix output",
                    {"shell-output": b"value"},
                    _direct_generated_spec(
                        root,
                        allowed_suffixes=(".out",),
                        suffix_media_types={
                            ".out": "application/octet-stream"
                        },
                    ),
                    ShellExecutionStatus.COMPLETED,
                ),
            )

            for name, output_files, spec, expected_status in cases:
                with self.subTest(name=name):
                    result = await ShellSandboxCommandExecutor(
                        settings=ShellToolSettings(
                            workspace_root=str(root),
                            materialized_input_files_dir="generated-files",
                        ),
                        sandbox_settings=_sandbox_settings(root),
                        sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                            sandbox_backend_module.SandboxFakeBackendScript(
                                capabilities=_capabilities(),
                                stream_chunks=(),
                                output_files=output_files,
                            )
                        ),
                    ).execute(spec)

                    self.assertEqual(result.status, expected_status)
                    if name == "nested output":
                        self.assertEqual(
                            result.generated_files[0].display_path,
                            "nested/shell-output.txt",
                        )
                        self.assertIsNone(
                            result.generated_files[0].content_base64
                        )
                    if name == "bare prefix output":
                        self.assertEqual(
                            result.generated_files[0].display_path,
                            "shell-output",
                        )
                    if name == "ignored output after file count cap":
                        self.assertEqual(len(result.generated_files), 1)
                        self.assertEqual(
                            result.generated_files[0].display_path,
                            "shell-output-1.txt",
                        )

    async def test_duplicate_sandbox_stream_chunks_emit_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            result = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root),
                sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                    sandbox_backend_module.SandboxFakeBackendScript(
                        capabilities=_capabilities(),
                        stream_chunks=(
                            sandbox_backend_module.SandboxStreamChunk(
                                stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                                content=b"first",
                                sequence=0,
                            ),
                            sandbox_backend_module.SandboxStreamChunk(
                                stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                                content=b"second",
                                sequence=1,
                            ),
                            sandbox_backend_module.SandboxStreamChunk(
                                stream=sandbox_backend_module.SandboxBackendStream.STDERR,
                                content=b"warn",
                                sequence=2,
                            ),
                            sandbox_backend_module.SandboxStreamChunk(
                                stream=sandbox_backend_module.SandboxBackendStream.STDERR,
                                content=b"again",
                                sequence=3,
                            ),
                        ),
                    )
                ),
            ).execute(_direct_text_spec(root), stream=record)

        self.assertEqual(result.stdout, "firstsecond")
        self.assertEqual(result.stderr, "warnagain")
        self.assertEqual(
            [(event.kind, event.content) for event in events],
            [
                (ToolExecutionStreamKind.STDOUT, "firstsecond"),
                (ToolExecutionStreamKind.STDERR, "warnagain"),
            ],
        )

    async def test_policy_guards_happen_before_sandbox(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("ok\n", encoding="utf-8")
            (root / ".hidden").write_text("hidden\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            hidden_policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=_AllResolved(),
            )
            sensitive_policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    workspace_root=str(root),
                    allow_hidden=True,
                ),
                resolver=_AllResolved(),
            )
            guarded = (
                (
                    _cat_request("../outside.txt"),
                    hidden_policy,
                    ShellExecutionErrorCode.TRAVERSAL,
                ),
                (
                    _cat_request(".hidden"),
                    hidden_policy,
                    ShellExecutionErrorCode.HIDDEN_PATH,
                ),
                (
                    _cat_request(".env"),
                    sensitive_policy,
                    ShellExecutionErrorCode.SENSITIVE_PATH,
                ),
            )

            for request, policy, error_code in guarded:
                with self.subTest(error_code=error_code.value):
                    with self.assertRaises(ShellPolicyDenied) as context:
                        await normalize_shell_execution_request(
                            request,
                            policy,
                            sandbox_settings=_sandbox_settings(root),
                        )
                    self.assertEqual(context.exception.error_code, error_code)

    async def test_sandbox_backend_concurrency_limit_is_enforced(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = sandbox_backend_module.SandboxFakeBackend(
                sandbox_backend_module.SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        (
                            sandbox_backend_module.SandboxBackendOperation.STREAM
                        ): 0.05,
                    },
                    max_concurrent_executions=1,
                )
            )
            executor = ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(root),
                sandbox_backend=backend,
            )

            first = create_task(executor.execute(_direct_text_spec(root)))
            await sleep(0)
            second = create_task(executor.execute(_direct_text_spec(root)))
            results = await wait_for(gather(first, second), timeout=1)

        statuses = {result.status for result in results}
        self.assertIn(ShellExecutionStatus.COMPLETED, statuses)
        self.assertIn(ShellExecutionStatus.POLICY_DENIED, statuses)
        self.assertEqual(backend.max_observed_concurrent_executions, 1)


class ShellSandboxToolSetTest(IsolatedAsyncioTestCase):
    async def test_sandbox_mode_rejects_custom_executor(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "custom shell executors require",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                executor=LocalCommandExecutor(settings=settings),
            )

    async def test_sandbox_settings_with_local_mode_are_rejected(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(workspace_root=str(fixture_root))

        with self.assertRaisesRegex(
            AssertionError,
            "sandbox settings require",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(fixture_root),
            )

    async def test_sandbox_mode_rejects_custom_composition_executor(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "custom shell composition executors require",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                sandbox_settings=_sandbox_settings(fixture_root),
                composition_executor=LocalCompositionExecutor(),
            )

    async def test_toolset_uses_sandbox_executor_without_schema_exposure(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
        )
        backend = sandbox_backend_module.SandboxFakeBackend(
            sandbox_backend_module.SandboxFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(
                    sandbox_backend_module.SandboxStreamChunk(
                        stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                        content=b"ok\n",
                        sequence=0,
                    ),
                ),
            )
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            sandbox_settings=_sandbox_settings(fixture_root),
            sandbox_backend=backend,
        )
        tool = _tool_by_name(toolset, "cat")
        call = cast(Callable[..., Awaitable[str]], tool)

        output = await call(
            "filesystem/visible.txt",
            context=ToolCallContext(),
        )
        schemas = toolset.json_schemas()

        self.assertIn("status: completed", output)
        self.assertIn("ok", output)
        self.assertIsInstance(output, ShellFormattedResult)
        formatted_output = cast(ShellFormattedResult, output)
        self.assertEqual(formatted_output.execution_result.backend, "sandbox")
        forbidden = {
            "approval",
            "backend",
            "container_backend",
            "container_image",
            "device",
            "image",
            "mount",
            "network",
            "sandbox",
            "sandbox_backend",
            "sandbox_root",
            "secret",
        }
        self.assertTrue(forbidden.isdisjoint(_schema_property_names(schemas)))
        serialized_forbidden = forbidden - {"image"}
        serialized_schema = str(schemas)
        for forbidden_value in serialized_forbidden:
            self.assertNotIn(forbidden_value, serialized_schema)

    async def test_toolset_uses_isolation_runtime_for_sandbox(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
        )
        runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.SANDBOX,
                source=trusted_isolation_source("sdk"),
                sandbox=_sandbox_settings(fixture_root),
            ),
            sandbox_backend=sandbox_backend_module.SandboxFakeBackend(
                sandbox_backend_module.SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        sandbox_backend_module.SandboxStreamChunk(
                            stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                            content=b"ok\n",
                            sequence=0,
                        ),
                    ),
                )
            ),
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            isolation_runtime=runtime,
        )

        call = cast(
            Callable[..., Awaitable[str]], _tool_by_name(toolset, "cat")
        )

        output = await call(
            "filesystem/visible.txt",
            context=ToolCallContext(),
        )

        formatted_output = cast(ShellFormattedResult, output)
        self.assertEqual(formatted_output.execution_result.backend, "sandbox")

    async def test_toolset_pipeline_fails_closed_for_sandbox(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
            allow_pipelines=True,
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            sandbox_settings=_sandbox_settings(fixture_root),
            sandbox_backend=None,
        ).with_enabled_tools(["shell.pipeline"])
        call = cast(
            Callable[..., Awaitable[str]],
            _tool_by_name(toolset, "pipeline"),
        )

        output = await call(
            steps=(
                {
                    "id": "read",
                    "command": "cat",
                    "paths": ("filesystem/visible.txt",),
                },
                {
                    "id": "count",
                    "command": "wc",
                    "stdin_from": {
                        "step_id": "read",
                        "stream": "stdout",
                    },
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        formatted = cast(ShellFormattedCompositionResult, output)
        self.assertEqual(
            formatted.composition_result.status,
            ShellExecutionStatus.POLICY_DENIED,
        )
        self.assertIn("sandbox byte pipelines", output)
        self.assertIn("cannot be lowered to shell evaluation", output)
        self.assertIn(
            "trusted structured runner",
            formatted.composition_result.error_message or "",
        )

    async def test_pipeline_path_guard_precedes_sandbox_denial(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
            allow_pipelines=True,
        )
        toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            sandbox_settings=_sandbox_settings(fixture_root),
            sandbox_backend=None,
        ).with_enabled_tools(["shell.pipeline"])
        call = cast(
            Callable[..., Awaitable[str]],
            _tool_by_name(toolset, "pipeline"),
        )

        output = await call(
            steps=(
                {
                    "id": "read",
                    "command": "cat",
                    "paths": ("../outside.txt",),
                },
            ),
            context=ToolCallContext(),
        )

        formatted = cast(ShellFormattedCompositionResult, output)
        self.assertEqual(
            formatted.composition_result.error_code,
            ShellExecutionErrorCode.TRAVERSAL,
        )
        self.assertNotIn("byte pipelines", output)

    def test_toolset_rejects_unsupported_isolation_runtime_hooks(self) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            execution_mode="sandbox",
            workspace_root=str(fixture_root),
        )
        runtime = IsolationToolRuntimeSettings(
            effective_settings=IsolationEffectiveSettings(
                mode=IsolationMode.SANDBOX,
                source=trusted_isolation_source("sdk"),
                sandbox=_sandbox_settings(fixture_root),
            ),
            authorization_provider=lambda plan: plan,
            audit_listeners=(lambda event: event,),
        )

        with self.assertRaisesRegex(
            AssertionError,
            "shell isolation runtime hooks are not supported",
        ):
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                isolation_runtime=runtime,
            )


class ShellSandboxValueTest(TestCase):
    def test_mixed_sandbox_and_container_lowering_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(AssertionError):
                lower_shell_execution_spec(
                    _direct_text_spec(root),
                    sandbox_settings=_sandbox_settings(root),
                    container_settings=object(),  # type: ignore[arg-type]
                )

    def test_generated_output_validation_defensive_branches(self) -> None:
        plan = _generated_output_plan(
            max_files=3,
            max_file_bytes=4,
            max_total_bytes=6,
        )
        cases = (
            ((), None),
            (
                (
                    sandbox_backend_module.SandboxOutputArtifact(
                        path="shell-output.txt",
                        content=b"12345",
                    ),
                ),
                plan,
            ),
            (
                (
                    sandbox_backend_module.SandboxOutputArtifact(
                        path="shell-output-1.txt",
                        content=b"1234",
                    ),
                    sandbox_backend_module.SandboxOutputArtifact(
                        path="shell-output-2.txt",
                        content=b"1234",
                    ),
                ),
                plan,
            ),
            (
                (
                    sandbox_backend_module.SandboxOutputArtifact(
                        path="/shell-output.txt",
                        content=b"1",
                    ),
                ),
                plan,
            ),
            (
                (
                    sandbox_backend_module.SandboxOutputArtifact(
                        path=".hidden.txt",
                        content=b"1",
                    ),
                ),
                plan,
            ),
        )

        for artifacts, output_plan in cases:
            with self.subTest(artifacts=artifacts, output_plan=output_plan):
                with self.assertRaises(Exception):
                    async_run(_sandbox_generated_files(artifacts, output_plan))

    def test_sandbox_scrub_capture_preserves_byte_cap(self) -> None:
        scrubbed = _sandbox_scrub_capture(
            ("prefix /private/", len("prefix /private/"), True),
            (("/private/tmp/generated", "GENERATED_PREFIX"),),
        )

        self.assertEqual(scrubbed[1], len("prefix /private/"))
        self.assertTrue(scrubbed[2])
        self.assertLessEqual(len(scrubbed[0].encode()), scrubbed[1])
        self.assertNotIn("/private", scrubbed[0])

    def test_sandbox_scrub_capture_leaves_untruncated_prefixes(self) -> None:
        content = "done /private"
        scrubbed = _sandbox_scrub_capture(
            (content, len(content), False),
            (("/private/tmp/generated", "GENERATED_PREFIX"),),
        )

        self.assertEqual(scrubbed, (content, len(content), False))

    def test_empty_sandbox_diagnostic_summary(self) -> None:
        self.assertEqual(
            _sandbox_diagnostic_summary(()),
            "sandbox execution failed",
        )

    def test_generated_sandbox_plan_uses_profile_output_root(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = lower_shell_execution_spec(
                _direct_generated_spec(root),
                sandbox_settings=_sandbox_settings(root),
            )

        self.assertIsNotNone(plan.sandbox_plan)
        assert plan.sandbox_plan is not None
        self.assertEqual(
            plan.sandbox_plan.output_dir,
            str((root / "outputs").resolve()),
        )

    def test_sandbox_directory_helpers_cover_recursive_edges(self) -> None:
        async def fail_root(path: str | Path, *, mode: int = 0o700) -> Path:
            del mode
            raise FileNotFoundError(path)

        async def retry_existing(
            path: str | Path,
            *,
            mode: int = 0o700,
        ) -> Path:
            del mode
            calls.append(Path(path))
            if len(calls) == 1:
                raise FileNotFoundError(path)
            if len(calls) == 3:
                raise FileExistsError(path)
            return Path(path)

        async def retry_tree_existing(
            path: str | Path,
            *,
            mode: int = 0o700,
        ) -> Path:
            del mode
            calls.append(Path(path))
            if len(calls) == 1:
                raise FileNotFoundError(path)
            if len(calls) == 2:
                raise FileExistsError(path)
            return Path(path)

        async def already_exists(
            path: str | Path,
            *,
            mode: int = 0o700,
        ) -> Path:
            del mode
            raise FileExistsError(path)

        with patch.object(
            shell_sandbox_module,
            "_make_directory",
            fail_root,
        ):
            with self.assertRaises(FileNotFoundError):
                async_run(shell_sandbox_module._ensure_directory(Path("/")))

        calls: list[Path] = []
        with patch.object(
            shell_sandbox_module,
            "_make_directory",
            retry_existing,
        ):
            async_run(
                shell_sandbox_module._ensure_directory(
                    Path("/workspace/output")
                )
            )
        self.assertEqual(
            calls,
            [
                Path("/workspace/output"),
                Path("/workspace"),
                Path("/workspace/output"),
            ],
        )

        async_run(
            shell_sandbox_module._make_directory_tree(
                Path("/workspace"),
                stop_at=Path("/workspace"),
            )
        )

        calls = []
        with patch.object(
            shell_sandbox_module,
            "_make_directory",
            retry_tree_existing,
        ):
            async_run(
                shell_sandbox_module._make_directory_tree(
                    Path("/workspace/output"),
                    stop_at=Path("/workspace"),
                )
            )
        self.assertEqual(
            calls,
            [
                Path("/workspace/output"),
                Path("/workspace/output"),
            ],
        )

        with patch.object(
            shell_sandbox_module,
            "_make_directory",
            already_exists,
        ):
            async_run(
                shell_sandbox_module._make_directory_tree(
                    Path("/workspace/output"),
                    stop_at=Path("/workspace"),
                )
            )

        with patch.object(
            shell_sandbox_module,
            "_make_directory",
            fail_root,
        ):
            with self.assertRaises(FileNotFoundError):
                async_run(
                    shell_sandbox_module._make_directory_tree(
                        Path("/other/output"),
                        stop_at=Path("/workspace"),
                    )
                )


def _cat_request(path: str, *, cwd: str | None = None) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.cat",
        command="cat",
        options={},
        paths=(
            PathOperand(
                name="path",
                path=path,
                kind="text_file",
                access="read",
            ),
        ),
        cwd=cwd,
    )


def _direct_text_spec(
    root: Path,
    *,
    executable: str | None = "/trusted/bin/cat",
    cwd: str | None = None,
    timeout_seconds: float = 10,
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
    metadata: dict[str, object] | None = None,
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="sandbox",
        tool_name="shell.cat",
        command="cat",
        executable=executable,
        argv=("cat", "--", "visible.txt"),
        display_argv=("cat", "--", "visible.txt"),
        cwd=str(root.resolve()) if cwd is None else cwd,
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        metadata=metadata,
    )


def _direct_generated_spec(
    root: Path,
    *,
    output_plan: bool = True,
    display_prefix: str = "shell-output",
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
    allowed_suffixes: tuple[str, ...] = (".txt",),
    suffix_media_types: dict[str, str] | None = None,
    max_files: int = 1,
    max_file_bytes: int = 64,
    max_total_bytes: int = 64,
    max_inline_bytes: int = 64,
) -> ExecutionSpec:
    plan = (
        _generated_output_plan(
            display_prefix=display_prefix,
            allowed_suffixes=allowed_suffixes,
            suffix_media_types=suffix_media_types,
            max_files=max_files,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            max_inline_bytes=max_inline_bytes,
        )
        if output_plan
        else None
    )
    return ExecutionPolicy().create_execution_spec(
        backend="sandbox",
        tool_name="shell.pdftoppm",
        command="pdftoppm",
        executable="/trusted/bin/pdftoppm",
        argv=("pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
        display_argv=("pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
        cwd=str(root.resolve()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="application/json",
        output_kind=ShellOutputKind.GENERATED_FILES,
        resource_class="heavy",
        output_plan=plan,
        timeout_seconds=10,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _generated_output_plan(
    *,
    display_prefix: str = "shell-output",
    allowed_suffixes: tuple[str, ...] = (".txt",),
    suffix_media_types: dict[str, str] | None = None,
    max_files: int = 1,
    max_file_bytes: int = 64,
    max_total_bytes: int = 64,
    max_inline_bytes: int = 64,
) -> GeneratedOutputPlan:
    return GeneratedOutputPlan(
        prefix_name="shell-output",
        display_prefix=display_prefix,
        allowed_suffixes=allowed_suffixes,
        suffix_media_types=suffix_media_types or {".txt": "text/plain"},
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_inline_bytes=max_inline_bytes,
    )


def _sandbox_settings(
    root: Path,
    *,
    required: bool = False,
    output: bool = True,
    timeout_seconds: int | None = 30,
) -> SandboxEffectiveSettings:
    root = root.resolve()
    profile = SandboxProfile(
        name="shell-readonly",
        trusted_executables=(
            "/trusted/bin/cat",
            "/trusted/bin/pdftoppm",
        ),
        read_roots=(str(root),),
        scratch_roots=(str(root / "scratch"),),
        output_roots=((str(root / "outputs"),) if output else ()),
        environment=SandboxEnvironmentPolicy(allowlist=("LC_ALL",)),
        resources=SandboxResourceLimits(
            timeout_seconds=timeout_seconds,
            pids=16,
        ),
        output=SandboxOutputPolicy(
            max_stdout_bytes=4096,
            max_stderr_bytes=4096,
            max_artifact_bytes=1024 if output else 0,
            allow_artifacts=output,
        ),
    )
    return SandboxEffectiveSettings(
        backend=SandboxBackend.SEATBELT,
        required=required,
        source=trusted_isolation_source("sdk"),
        policy_version="phase7",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _capabilities() -> sandbox_backend_module.SandboxBackendCapabilities:
    return sandbox_backend_module.SandboxBackendCapabilities(
        backend=SandboxBackend.SEATBELT,
        host_os="darwin",
        architecture="arm64",
        runtime_name="fake-seatbelt",
        sandbox_executable="/usr/bin/sandbox-exec",
        sandbox_executable_available=True,
        filesystem=sandbox_backend_module.SandboxFilesystemControls(
            read_roots=True,
            write_roots=True,
            deny_roots=True,
        ),
        network_modes=(SandboxNetworkMode.NONE,),
        process=sandbox_backend_module.SandboxProcessControls(
            process_limits=True,
            child_processes=True,
            inherited_fds=True,
        ),
        temp_output=sandbox_backend_module.SandboxTempOutputMapping(
            temp_dirs=True,
            output_dirs=True,
            cleanup_budget=True,
        ),
    )


def _schema_property_names(schemas: object) -> set[str]:
    names: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "properties" and isinstance(item, dict):
                    names.update(str(name) for name in item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(schemas)
    return names


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


class _PlanAwareGeneratedSandboxBackend:
    def __init__(self, content: bytes, *, stderr: bool = True) -> None:
        self.content = content
        self.stderr = stderr
        self.runtime_prefix: str | None = None
        self.output_dir: Path | None = None

    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> sandbox_backend_module.SandboxBackendProbeResult:
        return sandbox_backend_module.SandboxBackendProbeResult(
            backend=SandboxBackend.SEATBELT,
            available=True,
            capabilities=_capabilities(),
        )

    async def execute(
        self,
        plan: sandbox_backend_module.SandboxExecutionPlan,
    ) -> sandbox_backend_module.SandboxExecutionResult:
        assert plan.output_dir is not None
        self.output_dir = Path(plan.output_dir)
        self.runtime_prefix = f"{plan.output_dir}/shell-output"
        chunks = [
            sandbox_backend_module.SandboxStreamChunk(
                stream=sandbox_backend_module.SandboxBackendStream.STDOUT,
                content=f'{{"output": "{self.runtime_prefix}"}}\n'.encode(),
                sequence=0,
            )
        ]
        if self.stderr:
            chunks.append(
                sandbox_backend_module.SandboxStreamChunk(
                    stream=sandbox_backend_module.SandboxBackendStream.STDERR,
                    content=f"wrote {self.runtime_prefix}\n".encode(),
                    sequence=1,
                )
            )
        return sandbox_backend_module.SandboxExecutionResult(
            status=sandbox_backend_module.SandboxResultStatus.COMPLETED,
            exit_code=0,
            stream_chunks=tuple(chunks),
            stdout=chunks[0].content,
            stderr=chunks[1].content if len(chunks) > 1 else b"",
            output_artifacts=(
                sandbox_backend_module.SandboxOutputArtifact(
                    path="shell-output.txt",
                    content=self.content,
                ),
            ),
        )


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


class _ProbeRaisesBackend:
    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> "sandbox_backend_module.SandboxBackendProbeResult":
        raise RuntimeError("probe failed")

    async def execute(
        self,
        plan: object,
    ) -> "sandbox_backend_module.SandboxExecutionResult":
        raise AssertionError("execute should not be called")


class _ExecuteRaisesBackend:
    def __init__(
        self,
        script: sandbox_backend_module.SandboxFakeBackendScript,
    ) -> None:
        self._backend = sandbox_backend_module.SandboxFakeBackend(script)

    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> "sandbox_backend_module.SandboxBackendProbeResult":
        return await self._backend.probe(timeout_seconds=timeout_seconds)

    async def execute(
        self,
        plan: object,
    ) -> "sandbox_backend_module.SandboxExecutionResult":
        raise RuntimeError("execute failed")


class _ProbeDiagnosticBackend:
    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> "sandbox_backend_module.SandboxBackendProbeResult":
        raise sandbox_backend_module.SandboxBackendError(_diagnostic())

    async def execute(
        self,
        plan: object,
    ) -> "sandbox_backend_module.SandboxExecutionResult":
        raise AssertionError("execute should not be called")


class _ProbeMalformedDiagnosticBackend:
    async def probe(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> "sandbox_backend_module.SandboxBackendProbeResult":
        error = RuntimeError("malformed diagnostic")
        error.diagnostic = object()  # type: ignore[attr-defined]
        raise error

    async def execute(
        self,
        plan: object,
    ) -> "sandbox_backend_module.SandboxExecutionResult":
        raise AssertionError("execute should not be called")


class _ExecuteDiagnosticBackend(_ExecuteRaisesBackend):
    async def execute(
        self,
        plan: object,
    ) -> "sandbox_backend_module.SandboxExecutionResult":
        raise sandbox_backend_module.SandboxBackendError(_diagnostic())


def _diagnostic() -> sandbox_backend_module.SandboxBackendDiagnostic:
    return sandbox_backend_module.SandboxBackendDiagnostic(
        code=sandbox_backend_module.SandboxBackendDiagnosticCode.EXECUTION_FAILED,
        operation=sandbox_backend_module.SandboxBackendOperation.WAIT,
        message="backend failed",
        backend=SandboxBackend.SEATBELT,
    )


if __name__ == "__main__":
    main()
