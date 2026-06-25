from argparse import Namespace
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.agent.loader import OrchestratorLoader
from avalan.cli.commands import agent as agent_cmds
from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendSupportLevel,
    ContainerDeviceClass,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerNetworkPolicy,
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
    ContainerOutputValidationResult,
    ContainerProfile,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    DockerContainerBackend,
    container_selection_from_mapping,
    trusted_container_runtime_from_mapping,
    trusted_container_source,
)
from avalan.entities import (
    ToolCallContext,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool import Tool
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    GeneratedOutputPlan,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellContainerCommandExecutor,
    ShellExecutionMode,
    ShellExecutionPlan,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
    lower_shell_execution_spec,
    normalize_shell_execution_request,
)
from avalan.tool.shell.container import (
    _diagnostic_summary,
)
from avalan.tool.shell.entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ShellExecutionErrorCode,
)
from avalan.tool.shell.executor import CommandExecutor

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/shell-tools@sha256:{_DIGEST}"


class ShellContainerPlanningTest(IsolatedAsyncioTestCase):
    async def test_local_and_container_plan_lowering(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            local_settings = ShellToolSettings(workspace_root=str(root))
            container_settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )
            local_policy = ExecutionPolicy(
                settings=local_settings,
                resolver=_AllResolved(),
            )
            container_policy = ExecutionPolicy(
                settings=container_settings,
                resolver=_AllResolved(),
            )
            request = _cat_request("visible.txt")
            nested = root / "nested"
            nested.mkdir()
            (nested / "nested.txt").write_text("nested\n", encoding="utf-8")
            nested_request = _cat_request("nested.txt", cwd="nested")

            local = await normalize_shell_execution_request(
                request,
                local_policy,
            )
            container = await normalize_shell_execution_request(
                request,
                container_policy,
                container_settings=_effective_settings(),
            )
            nested_container = await normalize_shell_execution_request(
                nested_request,
                container_policy,
                container_settings=_effective_settings(),
            )

        self.assertEqual(local.mode, ShellExecutionMode.LOCAL)
        self.assertIsNone(local.container_plan)
        self.assertEqual(container.mode, ShellExecutionMode.CONTAINER)
        self.assertIsNotNone(container.container_plan)
        assert container.container_plan is not None
        self.assertEqual(
            container.container_plan.run_plan.command.argv,
            ("cat", "--", "visible.txt"),
        )
        self.assertEqual(
            container.container_plan.run_plan.command.cwd,
            "/workspace",
        )
        self.assertIsNotNone(nested_container.container_plan)
        assert nested_container.container_plan is not None
        self.assertEqual(
            nested_container.container_plan.run_plan.command.cwd,
            "/workspace/nested",
        )
        self.assertIn("container", container.to_dict())

    async def test_required_disabled_container_plan_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    execution_mode="container",
                    workspace_root=str(root),
                ),
                resolver=_AllResolved(),
            )
            spec = await policy.normalize(_cat_request("visible.txt"))

        with self.assertRaises(ShellPolicyDenied):
            lower_shell_execution_spec(
                spec,
                container_settings=_disabled_required_settings(),
            )

    async def test_container_mode_without_effective_settings_fails_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(
                    execution_mode="container",
                    workspace_root=str(root),
                ),
                resolver=_AllResolved(),
            )

            with self.assertRaises(AssertionError):
                await normalize_shell_execution_request(
                    _cat_request("visible.txt"),
                    policy,
                )


class ShellContainerExecutorTest(IsolatedAsyncioTestCase):
    async def test_container_executes_readonly_command_and_streams(
        self,
    ) -> None:
        spec = _direct_text_spec()
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
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
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.PROGRESS,
                        content=b"half",
                        sequence=2,
                    ),
                ),
                stats_samples=(
                    ContainerBackendStats(memory_bytes=4096, pids=2),
                ),
            )
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=backend,
            local_executor=_RecordingLocalExecutor(),
        ).execute(spec, stream=record)

        self.assertEqual(result.backend, "container")
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
                ToolExecutionStreamKind.PROGRESS,
            ],
        )
        self.assertIn(
            ContainerBackendOperation.CREATE,
            backend.operations,
        )
        self.assertIn(
            "container_plan_fingerprint",
            result.metadata,
        )

    async def test_container_enforces_separate_stream_caps(self) -> None:
        cases = (
            (
                10,
                3,
                (
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"wxyz0123456789",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"abcdefghijklmnopqrstuvwxyz",
                        sequence=1,
                    ),
                ),
                "abcdefghij",
                "wxy",
            ),
            (
                3,
                10,
                (
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"abcd0123456789",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"01234567890abcdef",
                        sequence=1,
                    ),
                ),
                "abc",
                "0123456789",
            ),
        )

        for stdout_cap, stderr_cap, chunks, stdout, stderr in cases:
            with self.subTest(stdout_cap=stdout_cap, stderr_cap=stderr_cap):
                result = await ShellContainerCommandExecutor(
                    container_settings=_effective_settings(required=True),
                    container_backend=ContainerFakeBackend(
                        ContainerFakeBackendScript(
                            capabilities=_capabilities(),
                            stream_chunks=chunks,
                        )
                    ),
                ).execute(
                    _direct_text_spec(
                        max_stdout_bytes=stdout_cap,
                        max_stderr_bytes=stderr_cap,
                    )
                )

                self.assertEqual(result.stdout, stdout)
                self.assertEqual(result.stderr, stderr)
                self.assertEqual(result.stdout_bytes, stdout_cap)
                self.assertEqual(result.stderr_bytes, stderr_cap)
                self.assertTrue(result.stdout_truncated)
                self.assertTrue(result.stderr_truncated)
                self.assertNotIn("[container", result.stdout)
                self.assertNotIn("[container", result.stderr)

    async def test_container_progress_does_not_consume_output_budget(
        self,
    ) -> None:
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        chunks = tuple(
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.PROGRESS,
                content=f"progress-{index}".encode(),
                sequence=index,
            )
            for index in range(20)
        ) + (
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=b"abcdefghijklmnopqrstuvwxyz",
                sequence=20,
            ),
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDERR,
                content=b"wxyz0123456789",
                sequence=21,
            ),
        )

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=chunks,
                )
            ),
        ).execute(
            _direct_text_spec(max_stdout_bytes=10, max_stderr_bytes=3),
            stream=record,
        )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "abcdefghij")
        self.assertEqual(result.stderr, "wxy")
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        diagnostic_codes = cast(
            tuple[str, ...],
            result.metadata["container_diagnostic_codes"],
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.EVENT_DROPPED.value,
            diagnostic_codes,
        )
        self.assertIn(
            ToolExecutionStreamKind.STDOUT,
            {event.kind for event in events},
        )
        self.assertIn(
            ToolExecutionStreamKind.STDERR,
            {event.kind for event in events},
        )

    async def test_sdk_cli_and_agent_toml_settings_are_equivalent(
        self,
    ) -> None:
        raw = _runtime_mapping()
        selection = container_selection_from_mapping(
            {"profile": "workspace-readonly", "required": True},
            source=trusted_container_source(ContainerSurface.SDK),
        )
        sdk_runtime = trusted_container_runtime_from_mapping(
            raw,
            source=trusted_container_source(ContainerSurface.SDK),
            selection=selection,
        )
        cli_settings = agent_cmds._agent_tool_settings(
            Namespace(
                tool_shell_backend="container",
                tool_container_backend="docker",
                tool_container_profile="workspace-readonly",
                tool_container_image=_IMAGE,
                tool_container_workspace_root=".",
                tool_container_pull_policy="never",
                tool_container_platform="linux/amd64",
                tool_container_cpu_count=None,
                tool_container_memory_bytes=None,
                tool_container_pids=None,
                tool_container_timeout_seconds=None,
                tool_container_network_mode="none",
                tool_container_review_mode=None,
                tool_shell_container_profile="workspace-readonly",
                tool_shell_container_required=True,
            )
        )
        agent_runtime = (
            OrchestratorLoader._container_runtime_settings_from_config(
                {"agent": {}, "runtime": {}},
                {
                    "container": raw,
                    "shell": {
                        "backend": "container",
                        "container": {"profile": "workspace-readonly"},
                    },
                },
            )
        )
        assert cli_settings.container is not None
        assert agent_runtime is not None
        self.assertIsInstance(
            cli_settings.container.backend,
            DockerContainerBackend,
        )
        runtimes = (
            sdk_runtime,
            cli_settings.container,
            agent_runtime,
        )
        self.assertTrue(
            all(runtime.rootful_authorized for runtime in runtimes)
        )
        canonical = [
            runtime.effective_settings.canonical_policy_input()
            for runtime in runtimes
            if runtime.effective_settings is not None
        ]

        self.assertEqual(canonical[0], canonical[1])
        self.assertEqual(canonical[1], canonical[2])
        outputs: list[str] = []
        for runtime in runtimes:
            assert runtime.effective_settings is not None
            result = await ShellContainerCommandExecutor(
                container_settings=runtime.effective_settings,
                container_backend=ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        stream_chunks=(
                            ContainerBackendStreamChunk(
                                stream=ContainerBackendStream.STDOUT,
                                content=b"ok\n",
                                sequence=0,
                            ),
                        ),
                    )
                ),
            ).execute(_direct_text_spec())
            outputs.append(result.stdout)

        self.assertEqual(outputs, ["ok\n", "ok\n", "ok\n"])

    async def test_container_zero_stream_budgets_do_not_crash(self) -> None:
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDOUT,
                            content=b"a",
                            sequence=0,
                        ),
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDERR,
                            content=b"b",
                            sequence=1,
                        ),
                    ),
                )
            ),
        ).execute(
            _direct_text_spec(max_stdout_bytes=0, max_stderr_bytes=0),
            stream=record,
        )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout_bytes, 0)
        self.assertEqual(result.stderr_bytes, 0)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        self.assertEqual(events, [])

    async def test_container_collects_generated_outputs(self) -> None:
        output_contract = ContainerOutputContract(
            contract_type=ContainerOutputContractType.GENERATED_FILE,
            max_bytes=64,
        )
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                output_result=ContainerOutputValidationResult(
                    decision=ContainerOutputDecisionType.ACCEPT,
                    contract=output_contract,
                    artifacts=(
                        ContainerOutputArtifact(
                            artifact_type=(
                                ContainerOutputContractType.GENERATED_FILE
                            ),
                            path="report.txt",
                            size_bytes=5,
                            media_type="text/plain",
                            digest=f"sha256:{'1' * 64}",
                        ),
                    ),
                    total_bytes=5,
                    file_count=1,
                ),
            )
        )

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=backend,
        ).execute(_direct_generated_spec())

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.output_kind, ShellOutputKind.GENERATED_FILES)
        self.assertEqual(len(result.generated_files), 1)
        self.assertEqual(result.generated_files[0].display_path, "report.txt")
        self.assertEqual(result.generated_files[0].sha256, "1" * 64)
        self.assertIn(
            ContainerBackendOperation.COPY_OUTPUTS,
            backend.operations,
        )

    async def test_local_spec_without_container_settings_uses_local_executor(
        self,
    ) -> None:
        local = _RecordingLocalExecutor()
        result = await ShellContainerCommandExecutor(
            container_settings=None,
            container_backend=None,
            local_executor=local,
        ).execute(_direct_text_spec(backend="local"))

        self.assertEqual(result.backend, "local")
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(local.calls, 1)

    async def test_local_spec_with_container_settings_fails_closed(
        self,
    ) -> None:
        local = _RecordingLocalExecutor()

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_capabilities())
            ),
            local_executor=local,
        ).execute(_direct_text_spec(backend="local"))

        self.assertEqual(result.backend, "container")
        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(local.calls, 0)
        self.assertIn("local shell plans", result.error_message or "")

    async def test_no_silent_host_fallback_when_required_backend_missing(
        self,
    ) -> None:
        local = _RecordingLocalExecutor()

        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=None,
            local_executor=local,
        ).execute(_direct_text_spec())

        self.assertEqual(result.backend, "container")
        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(local.calls, 0)
        self.assertIn("no backend", result.error_message or "")

    async def test_required_disabled_container_does_not_fallback(
        self,
    ) -> None:
        local = _RecordingLocalExecutor()

        result = await ShellContainerCommandExecutor(
            container_settings=_disabled_required_settings(),
            container_backend=None,
            local_executor=local,
        ).execute(_direct_text_spec())

        self.assertEqual(result.backend, "container")
        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.POLICY_DENIED,
        )
        self.assertEqual(local.calls, 0)

    async def test_optional_disabled_container_does_not_fallback(
        self,
    ) -> None:
        local = _RecordingLocalExecutor()

        result = await ShellContainerCommandExecutor(
            container_settings=_disabled_optional_settings(),
            container_backend=None,
            local_executor=local,
        ).execute(_direct_text_spec())

        self.assertEqual(result.backend, "container")
        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(local.calls, 0)
        self.assertIn("no profile", result.error_message or "")

    async def test_negative_container_runtime_failures(self) -> None:
        output_contract = ContainerOutputContract(
            contract_type=ContainerOutputContractType.GENERATED_FILE,
            max_bytes=4,
        )
        cases = (
            (
                "runtime unavailable",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    available=False,
                ),
                _direct_text_spec(),
                ShellExecutionStatus.TOOL_ERROR,
            ),
            (
                "mount failure",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_diagnostics={
                        ContainerBackendOperation.CREATE: (
                            ContainerBackendDiagnosticCode.CREATE_FAILED
                        )
                    },
                ),
                _direct_text_spec(),
                ShellExecutionStatus.TOOL_ERROR,
            ),
            (
                "network denial",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                ),
                _direct_text_spec(),
                ShellExecutionStatus.TOOL_ERROR,
            ),
            (
                "timeout",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_timed_out=True,
                ),
                _direct_text_spec(),
                ShellExecutionStatus.TIMEOUT,
            ),
            (
                "cancellation",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    cancel_operations=(ContainerBackendOperation.WAIT,),
                ),
                _direct_text_spec(),
                ShellExecutionStatus.CANCELLED,
            ),
            (
                "non-zero exit",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_exit_code=7,
                ),
                _direct_text_spec(),
                ShellExecutionStatus.NONZERO_EXIT,
            ),
            (
                "oversized output",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    output_result=ContainerOutputValidationResult(
                        decision=ContainerOutputDecisionType.REJECT,
                        contract=output_contract,
                        diagnostics=(
                            ContainerOutputDiagnostic(
                                code=ContainerOutputDiagnosticCode.TOO_LARGE,
                                path="report.txt",
                                message="too large",
                            ),
                        ),
                    ),
                ),
                _direct_generated_spec(),
                ShellExecutionStatus.TOO_LARGE,
            ),
            (
                "unsafe output",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    output_result=ContainerOutputValidationResult(
                        decision=ContainerOutputDecisionType.REJECT,
                        contract=output_contract,
                        diagnostics=(
                            ContainerOutputDiagnostic(
                                code=(
                                    ContainerOutputDiagnosticCode.UNSAFE_MEDIA
                                ),
                                path="report.txt",
                                message="unsafe media",
                            ),
                        ),
                    ),
                ),
                _direct_generated_spec(),
                ShellExecutionStatus.TOOL_ERROR,
            ),
            (
                "denied image",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    soft_operation_diagnostics={
                        ContainerBackendOperation.IMAGE_RESOLUTION: (
                            ContainerBackendDiagnosticCode.IMAGE_DENIED
                        )
                    },
                ),
                _direct_text_spec(),
                ShellExecutionStatus.POLICY_DENIED,
            ),
            (
                "cleanup failure",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    cleanup_uncertain=True,
                ),
                _direct_text_spec(),
                ShellExecutionStatus.TOOL_ERROR,
            ),
        )

        for name, script, spec, expected_status in cases:
            with self.subTest(name=name):
                settings = (
                    _network_settings()
                    if name == "network denial"
                    else _effective_settings(required=True)
                )
                result = await ShellContainerCommandExecutor(
                    container_settings=settings,
                    container_backend=ContainerFakeBackend(script),
                ).execute(spec)

                self.assertEqual(result.status, expected_status)
                self.assertNotEqual(result.backend, "local")

    async def test_apple_container_requires_runtime_opt_in(self) -> None:
        without_opt_in = await ShellContainerCommandExecutor(
            container_settings=_apple_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_apple_capabilities())
            ),
        ).execute(_direct_text_spec())
        with_opt_in_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_apple_capabilities())
        )
        with_opt_in = await ShellContainerCommandExecutor(
            container_settings=_apple_effective_settings(required=True),
            container_backend=with_opt_in_backend,
            opt_in_backends=(ContainerBackend.APPLE_CONTAINER,),
        ).execute(_direct_text_spec())

        self.assertEqual(
            without_opt_in.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertIn(
            "container.backend.capability_mismatch",
            without_opt_in.error_message or "",
        )
        self.assertEqual(with_opt_in.status, ShellExecutionStatus.COMPLETED)
        self.assertIn(
            ContainerBackendOperation.CREATE,
            with_opt_in_backend.operations,
        )

    async def test_malformed_generated_output_contract_fails_closed(
        self,
    ) -> None:
        result = await ShellContainerCommandExecutor(
            container_settings=_effective_settings(required=True),
            container_backend=ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_capabilities())
            ),
        ).execute(_direct_generated_spec(output_plan=False))

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIn("invalid", result.error_message or "")


class ShellContainerToolSetTest(IsolatedAsyncioTestCase):
    async def test_toolset_uses_container_executor_without_schema_exposure(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("hello\n", encoding="utf-8")
            settings = ShellToolSettings(
                execution_mode="container",
                workspace_root=str(root),
            )
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(capabilities=_capabilities())
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=_AllResolved(),
                ),
                container_settings=_effective_settings(required=True),
                container_backend=backend,
            )
            tool = _tool_by_name(toolset, "cat")
            call = cast(Callable[..., Awaitable[str]], tool)
            output = await call(
                "visible.txt",
                context=ToolCallContext(),
            )
            schemas = toolset.json_schemas()

        serialized_schema = str(schemas)
        self.assertIn("status: completed", output)
        self.assertIn("ok", output)
        for forbidden in (
            "mount",
            "secret",
            "backend",
            "profile",
            "runtime",
            "container_image",
        ):
            self.assertNotIn(forbidden, serialized_schema)

    async def test_policy_denial_and_path_guards_happen_before_container(
        self,
    ) -> None:
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
                (
                    ShellCommandRequest(
                        tool_name="shell.cat",
                        command="cat",
                        options={},
                        paths=(
                            PathOperand(
                                name="path",
                                path="visible.txt",
                                kind="text_file",
                                access="read",
                            ),
                        ),
                        cwd="/outside",
                    ),
                    hidden_policy,
                    ShellExecutionErrorCode.INVALID_CWD,
                ),
            )

            for request, policy, error_code in guarded:
                with self.subTest(error_code=error_code.value):
                    with self.assertRaises(ShellPolicyDenied) as context:
                        await normalize_shell_execution_request(
                            request,
                            policy,
                            container_settings=_effective_settings(
                                required=True
                            ),
                        )
                    self.assertEqual(context.exception.error_code, error_code)


class ShellContainerValueTest(TestCase):
    def test_execution_plan_validates_shape(self) -> None:
        spec = _direct_text_spec()
        string_mode = ShellExecutionPlan(
            mode="local",
            local_spec=spec,
        )

        self.assertEqual(string_mode.mode, ShellExecutionMode.LOCAL)
        self.assertEqual(_diagnostic_summary(()), "container execution failed")
        with self.assertRaises(AssertionError):
            ShellExecutionPlan(
                mode=ShellExecutionMode.CONTAINER,
                local_spec=spec,
            )
        with self.assertRaises(AssertionError):
            ShellExecutionPlan(
                mode=1,  # type: ignore[arg-type]
                local_spec=spec,
            )
        with self.assertRaises(AssertionError):
            lower_shell_execution_spec(
                _direct_text_spec(metadata={"tool_call_id": 1}),
                container_settings=_effective_settings(),
            )

    def test_shell_metadata_becomes_container_request_id(self) -> None:
        plan = lower_shell_execution_spec(
            _direct_text_spec(metadata={"tool_call_id": "call-1"}),
            container_settings=_effective_settings(),
        )

        self.assertIsNotNone(plan.container_plan)
        assert plan.container_plan is not None
        self.assertEqual(plan.container_plan.request.request_id, "call-1")


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
    *,
    backend: Literal["local", "container"] = "container",
    metadata: dict[str, object] | None = None,
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend=backend,
        tool_name="shell.cat",
        command="cat",
        executable="/trusted/bin/cat",
        argv=("/trusted/bin/cat", "visible.txt"),
        display_argv=("cat", "visible.txt"),
        cwd=str(Path.cwd()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=10,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        metadata=metadata,
    )


def _direct_generated_spec(
    *,
    backend: Literal["local", "container"] = "container",
    output_plan: bool = True,
) -> ExecutionSpec:
    plan = (
        GeneratedOutputPlan(
            prefix_name="shell-output",
            display_prefix="shell-output",
            allowed_suffixes=(".txt",),
            suffix_media_types={".txt": "text/plain"},
            max_files=1,
            max_file_bytes=64,
            max_total_bytes=64,
            max_inline_bytes=64,
        )
        if output_plan
        else None
    )
    return ExecutionPolicy().create_execution_spec(
        backend=backend,
        tool_name="shell.pdftoppm",
        command="pdftoppm",
        executable="/trusted/bin/pdftoppm",
        argv=("/trusted/bin/pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
        display_argv=("pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
        cwd=str(Path.cwd()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.GENERATED_FILES,
        resource_class="heavy",
        output_plan=plan,
        timeout_seconds=10,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )


def _effective_settings(
    *,
    required: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="shell-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase10",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _apple_effective_settings(
    *,
    required: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="shell-readonly",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.APPLE_CONTAINER,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase10",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _network_settings() -> ContainerEffectiveSettings:
    profile = ContainerProfile(
        name="shell-network",
        image=ContainerImagePolicy(reference=_IMAGE),
        network=ContainerNetworkPolicy(mode=ContainerNetworkMode.FULL),
    )
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=True,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase10",
        profile_registry_id="shell",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _disabled_optional_settings() -> ContainerEffectiveSettings:
    return ContainerEffectiveSettings(
        backend=ContainerBackend.NONE,
        required=False,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase10",
        profile_registry_id="shell",
    )


def _disabled_required_settings() -> ContainerEffectiveSettings:
    return ContainerEffectiveSettings(
        backend=ContainerBackend.NONE,
        required=True,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_source(),
        policy_version="phase10",
        profile_registry_id="shell",
    )


def _runtime_mapping() -> dict[str, object]:
    return {
        "backend": "docker",
        "default_profile": "workspace-readonly",
        "profiles": {
            "workspace-readonly": {
                "image": _IMAGE,
                "workspace_root": ".",
                "network": "none",
            }
        },
        "policy_version": "phase11",
    }


def _source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.SDK,
        trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
    )


def _capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT),
        device_classes=(ContainerDeviceClass.CPU,),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _apple_capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.APPLE_CONTAINER,
        host_os="darwin",
        guest_os="linux",
        architecture="amd64",
        support_level=ContainerBackendSupportLevel.OPT_IN,
        platform_emulation=False,
        rootless=False,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT),
        device_classes=(ContainerDeviceClass.CPU,),
        resource_limits=True,
        per_container_vm_isolation=True,
        streaming_attach=True,
        stats=True,
        lifecycle_normalization=True,
    )


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


class _RecordingLocalExecutor(CommandExecutor):
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.calls += 1
        if stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="local",
                )
            )
        return ExecutionResult(
            backend="local",
            tool_name=spec.tool_name,
            command=spec.command,
            argv=spec.argv,
            display_argv=spec.display_argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="local",
            stderr="",
            stdout_media_type=spec.stdout_media_type,
            output_kind=spec.output_kind,
            stdout_bytes=5,
            stderr_bytes=0,
        )


if __name__ == "__main__":
    main()
