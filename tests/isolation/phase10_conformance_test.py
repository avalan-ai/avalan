from asyncio import create_task, gather, sleep, wait_for
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendSupportLevel,
    ContainerDeviceClass,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerProfile,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
)
from avalan.isolation import (
    IsolationDiagnosticCode,
    IsolationEffectiveSettings,
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    LocalIsolationPolicy,
    SandboxBackend,
    SandboxEffectiveSettings,
    trusted_isolation_source,
)
from avalan.sandbox import (
    SandboxBackendCapabilities,
    SandboxBackendDiagnosticCode,
    SandboxBackendOperation,
    SandboxBackendStream,
    SandboxFakeBackend,
    SandboxFakeBackendScript,
    SandboxFilesystemControls,
    SandboxNetworkMode,
    SandboxProcessControls,
    SandboxStreamChunk,
    SandboxTempOutputMapping,
)
from avalan.server.container_policy import (
    RemoteContainerRequestPolicy,
    validate_remote_container_arguments,
)
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    LocalCommandExecutor,
    ShellContainerCommandExecutor,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellSandboxCommandExecutor,
)

_DIGEST = "e" * 64
_IMAGE = f"ghcr.io/example/phase10-tools@sha256:{_DIGEST}"
_REPO_ROOT = Path(__file__).parents[2]

_PHASE10_E2E_DIMENSIONS = {
    "shell tools": (
        (
            "tests/tool/shell/sandbox_test.py"
            "::ShellSandboxExecutorTest"
            "::test_sandbox_executes_readonly_command_and_streams"
        ),
        (
            "tests/tool/shell/container_test.py"
            "::ShellContainerExecutorTest"
            "::test_container_executes_readonly_command_and_streams"
        ),
    ),
    "trusted agent runs": (
        (
            "tests/agent/loader_test.py"
            "::LoaderFromFileTestCase"
            "::test_sandbox_toml_sections_load_trusted_settings"
        ),
        (
            "tests/tool/shell/container_test.py"
            "::ShellContainerToolSetTest"
            "::test_toolset_uses_container_executor_without_schema_exposure"
        ),
    ),
    "strict flows": (
        (
            "tests/flow/container_test.py"
            "::FlowContainerPlanTestCase"
            "::test_strict_container_flow_run_streams_lifecycle_events"
        ),
    ),
    "direct tasks": (
        (
            "tests/task/container_execution_test.py"
            "::TaskContainerExecutionTest"
            "::test_direct_container_execution_verifies_and_records_events"
        ),
    ),
    "queued tasks": (
        (
            "tests/task/container_execution_test.py"
            "::TaskContainerExecutionTest"
            "::test_client_and_worker_preserve_direct_queued_equivalence"
        ),
    ),
    "worker execution": (
        (
            "tests/task/container_execution_test.py"
            "::TaskContainerExecutionTest"
            "::test_queued_worker_verifies_container_metadata_before_target"
        ),
    ),
    "server requests": (
        (
            "tests/server/remote_container_test.py"
            "::RemoteContainerProfileSelectionTestCase"
            "::test_allows_exposed_camel_case_profile_selector"
        ),
        (
            "tests/server/responses_test.py"
            "::ResponsesEndpointTestCase"
            "::test_response_endpoint_allows_exposed_container_profile"
        ),
    ),
    "MCP": (
        (
            "tests/server/mcp_router_test.py"
            "::MCPRouterEdgeCaseAsyncTestCase"
            "::test_consume_call_request_allows_exposed_container_profile"
        ),
    ),
    "A2A": (
        (
            "tests/server/a2a_v1_router_test.py"
            "::test_chat_request_allows_exposed_a2a_container_profile"
        ),
    ),
    "runtime envelopes": (
        (
            "tests/container/runtime_envelope_test.py"
            "::ContainerRuntimeEnvelopeTest"
            "::test_fake_e2e_runtime_envelope_lifecycle_and_handoff"
        ),
        (
            "tests/server/container_policy_test.py"
            "::RemoteContainerRequestPolicyTestCase"
            "::test_server_runtime_envelope_status_accepts_available_runtime"
        ),
    ),
}

_PHASE10_PERFORMANCE_DIMENSIONS = {
    "event-loop responsiveness": (
        (
            "tests/isolation/phase10_conformance_test.py"
            "::Phase10StressConformanceTest"
            "::test_concurrent_local_sandbox_and_container_execution_is_bounded"
        ),
    ),
    "stream backpressure": (
        (
            "tests/container/stress_conformance_test.py"
            "::ContainerStressConformanceTest"
            "::test_slow_async_stream_is_drained_after_kept_event_cap"
        ),
    ),
    "bounded output": (
        (
            "tests/tool/shell/sandbox_test.py"
            "::ShellSandboxExecutorTest"
            "::test_sandbox_stream_caps_are_shell_visible"
        ),
        (
            "tests/tool/shell/container_test.py"
            "::ShellContainerExecutorTest"
            "::test_container_enforces_separate_stream_caps"
        ),
    ),
    "cleanup deadlines": (
        (
            "tests/container/watchdog_conformance_test.py"
            "::ContainerWatchdogConformanceTest"
            "::test_cleanup_watchdogs_report_uncertainty"
        ),
    ),
    "bounded backend probe work": (
        (
            "tests/isolation/phase10_conformance_test.py"
            "::Phase10StressConformanceTest"
            "::test_probe_work_is_bounded_for_fake_backend_batches"
        ),
    ),
    "worker throughput": (
        (
            "tests/task/stores/pgsql_benchmark_test.py"
            "::PgsqlBenchmarkCaseTest"
            "::test_plan_issues_detect_claims_without_skip_locked"
        ),
    ),
    "server streaming": (
        (
            "tests/server/create_response_sse_test.py"
            "::CreateResponseSSEEventsTestCase"
            "::test_streaming_response_long_stream_flushes_bounded_deltas"
        ),
    ),
    "concurrent local/sandbox/container execution": (
        (
            "tests/isolation/phase10_conformance_test.py"
            "::Phase10StressConformanceTest"
            "::test_concurrent_local_sandbox_and_container_execution_is_bounded"
        ),
    ),
}

_OPTIONAL_RUNTIME_GATES = {
    "AVALAN_CONTAINER_DOCKER_E2E": (
        "tests/container/docker_test.py",
        "set AVALAN_CONTAINER_DOCKER_E2E=1",
    ),
    "AVALAN_CONTAINER_APPLE_E2E": (
        "tests/container/apple_test.py",
        "set AVALAN_CONTAINER_APPLE_E2E=1",
    ),
    "AVALAN_ISOLATION_SEATBELT_E2E": (
        "tests/sandbox/real_runtime_e2e_test.py",
        "set AVALAN_ISOLATION_SEATBELT_E2E=1",
    ),
    "AVALAN_ISOLATION_BUBBLEWRAP_E2E": (
        "tests/sandbox/real_runtime_e2e_test.py",
        "set AVALAN_ISOLATION_BUBBLEWRAP_E2E=1",
    ),
}


class Phase10IsolationModeConformanceTest(IsolatedAsyncioTestCase):
    async def test_positive_conformance_for_implemented_modes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            local = await LocalCommandExecutor().execute(
                _local_spec(root, "printf local-ok")
            )
            sandbox_results = await gather(
                *(
                    ShellSandboxCommandExecutor(
                        sandbox_settings=_sandbox_settings(root, backend),
                        sandbox_backend=SandboxFakeBackend(
                            SandboxFakeBackendScript(
                                capabilities=_sandbox_capabilities(backend),
                                stream_chunks=(
                                    SandboxStreamChunk(
                                        stream=SandboxBackendStream.STDOUT,
                                        content=(f"{backend.value}-ok").encode(
                                            "ascii"
                                        ),
                                        sequence=0,
                                    ),
                                ),
                            )
                        ),
                    ).execute(_sandbox_spec(root))
                    for backend in (SandboxBackend.SEATBELT,)
                )
            )
            container_results = await gather(
                *(
                    ShellContainerCommandExecutor(
                        container_settings=_container_settings(backend),
                        container_backend=ContainerFakeBackend(
                            ContainerFakeBackendScript(
                                capabilities=_container_capabilities(backend),
                                stream_chunks=(
                                    ContainerBackendStreamChunk(
                                        stream=ContainerBackendStream.STDOUT,
                                        content=(f"{backend.value}-ok").encode(
                                            "ascii"
                                        ),
                                        sequence=0,
                                    ),
                                ),
                            )
                        ),
                        opt_in_backends=(
                            (backend,)
                            if backend is ContainerBackend.APPLE_CONTAINER
                            else ()
                        ),
                    ).execute(_container_spec())
                    for backend in (
                        ContainerBackend.DOCKER,
                        ContainerBackend.APPLE_CONTAINER,
                    )
                )
            )

        self.assertEqual(local.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(local.stdout, "local-ok")
        self.assertEqual(
            [result.status for result in sandbox_results],
            [ShellExecutionStatus.COMPLETED],
        )
        self.assertEqual(
            [result.stdout for result in sandbox_results],
            ["seatbelt-ok"],
        )
        self.assertEqual(
            [result.status for result in container_results],
            [ShellExecutionStatus.COMPLETED, ShellExecutionStatus.COMPLETED],
        )
        self.assertEqual(
            [result.stdout for result in container_results],
            ["docker-ok", "apple-container-ok"],
        )

    async def test_remote_profile_selection_is_validation_only(self) -> None:
        request = validate_remote_container_arguments(
            {
                "containerProfile": "workspace-readonly",
                "messages": [{"role": "user", "content": "hello"}],
            },
            policy=RemoteContainerRequestPolicy(
                exposed_profiles=("workspace-readonly",)
            ),
        )

        self.assertEqual(request.profile, "workspace-readonly")
        self.assertEqual(
            request.arguments,
            {"messages": [{"role": "user", "content": "hello"}]},
        )

    async def test_negative_stable_failure_modes_are_observable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sandbox_no_backend = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(
                    root, SandboxBackend.SEATBELT
                ),
                sandbox_backend=None,
            ).execute(_sandbox_spec(root))
            sandbox_mismatch = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(
                    root, SandboxBackend.SEATBELT
                ),
                sandbox_backend=SandboxFakeBackend(
                    SandboxFakeBackendScript(
                        capabilities=_sandbox_capabilities(
                            SandboxBackend.SEATBELT,
                            write_roots=False,
                        )
                    )
                ),
            ).execute(_sandbox_spec(root))
            container_no_backend = await ShellContainerCommandExecutor(
                container_settings=_container_settings(
                    ContainerBackend.DOCKER,
                    required=True,
                ),
                container_backend=None,
            ).execute(_container_spec())
            container_mismatch = await ShellContainerCommandExecutor(
                container_settings=_container_settings(
                    ContainerBackend.DOCKER
                ),
                container_backend=ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_container_capabilities(
                            ContainerBackend.DOCKER,
                            workspace_mounts=False,
                        )
                    )
                ),
            ).execute(_container_spec())

        self.assertEqual(
            {code.value for code in IsolationDiagnosticCode},
            {
                "isolation.mode_conflict",
                "isolation.unsupported_mode",
                "isolation.unsupported_backend",
                "isolation.mode_unavailable",
                "isolation.capability_mismatch",
                "isolation.elevation_required",
                "isolation.elevation_denied",
                "isolation.fallback_denied",
                "isolation.approval_stale",
                "isolation.policy_drift",
                "isolation.audit_unavailable",
                "sandbox.provider_unavailable",
                "sandbox.profile_generation_failed",
                "sandbox.path_denied",
                "sandbox.network_unenforceable",
                "container.backend.unavailable",
                "container.backend.capability_mismatch",
                "isolation.unsupported_syntax",
            },
        )
        with self.assertRaises(AssertionError):
            IsolationEffectiveSettings(
                mode="sandbox",
                source=trusted_isolation_source("sdk"),
                local=LocalIsolationPolicy(approval_required=False),
                sandbox=_sandbox_settings(root, SandboxBackend.SEATBELT),
            )
        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {"mode": "remote"},
                source=trusted_isolation_source("sdk"),
            )
        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                {
                    "mode": "sandbox",
                    "sandbox": _sandbox_settings_raw("firejail", root),
                },
                source=trusted_isolation_source("sdk"),
            )

        self.assertEqual(
            sandbox_no_backend.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertIn("no backend", sandbox_no_backend.error_message or "")
        self.assertEqual(
            sandbox_mismatch.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertIn(
            SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH.value,
            sandbox_mismatch.error_message or "",
        )
        self.assertEqual(
            container_no_backend.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertIn("no backend", container_no_backend.error_message or "")
        self.assertEqual(
            container_mismatch.status,
            ShellExecutionStatus.TOOL_ERROR,
        )
        self.assertIn(
            ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH.value,
            container_mismatch.error_message or "",
        )


class Phase10StressConformanceTest(IsolatedAsyncioTestCase):
    async def test_concurrent_local_sandbox_and_container_execution_is_bounded(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sandbox_backend = SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_sandbox_capabilities(
                        SandboxBackend.SEATBELT
                    ),
                    operation_delay_seconds={
                        SandboxBackendOperation.STREAM: 0.02,
                    },
                    stream_chunks=(
                        SandboxStreamChunk(
                            stream=SandboxBackendStream.STDOUT,
                            content=b"sandbox",
                            sequence=0,
                        ),
                    ),
                )
            )
            container_backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_container_capabilities(
                        ContainerBackend.DOCKER
                    ),
                    operation_delay_seconds={
                        ContainerBackendOperation.STREAM: 0.02,
                    },
                    stream_chunks=(
                        ContainerBackendStreamChunk(
                            stream=ContainerBackendStream.STDOUT,
                            content=b"container",
                            sequence=0,
                        ),
                    ),
                )
            )
            ticker = _Ticker()

            async def execute_all() -> tuple[ExecutionResult, ...]:
                ticker_task = ticker.start()
                try:
                    return await gather(
                        LocalCommandExecutor().execute(
                            _local_spec(root, "printf local")
                        ),
                        ShellSandboxCommandExecutor(
                            sandbox_settings=_sandbox_settings(
                                root, SandboxBackend.SEATBELT
                            ),
                            sandbox_backend=sandbox_backend,
                        ).execute(_sandbox_spec(root)),
                        ShellContainerCommandExecutor(
                            container_settings=_container_settings(
                                ContainerBackend.DOCKER
                            ),
                            container_backend=container_backend,
                        ).execute(_container_spec()),
                    )
                finally:
                    ticker.stop()
                    await ticker_task

            results = await wait_for(execute_all(), timeout=2)

        self.assertEqual(
            [result.status for result in results],
            [
                ShellExecutionStatus.COMPLETED,
                ShellExecutionStatus.COMPLETED,
                ShellExecutionStatus.COMPLETED,
            ],
        )
        self.assertGreaterEqual(ticker.ticks, 2)
        self.assertEqual(
            sandbox_backend.operations.count(SandboxBackendOperation.PROBE),
            1,
        )
        self.assertEqual(
            container_backend.operations.count(
                ContainerBackendOperation.PROBE
            ),
            1,
        )

    async def test_probe_work_is_bounded_for_fake_backend_batches(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sandbox_backend = SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_sandbox_capabilities(SandboxBackend.SEATBELT)
                )
            )
            container_backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_container_capabilities(
                        ContainerBackend.DOCKER
                    )
                )
            )
            sandbox_executor = ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(
                    root, SandboxBackend.SEATBELT
                ),
                sandbox_backend=sandbox_backend,
            )
            container_executor = ShellContainerCommandExecutor(
                container_settings=_container_settings(
                    ContainerBackend.DOCKER
                ),
                container_backend=container_backend,
            )

            sandbox_results = await gather(
                *(
                    sandbox_executor.execute(_sandbox_spec(root))
                    for _ in range(3)
                )
            )
            container_results = await gather(
                *(
                    container_executor.execute(_container_spec())
                    for _ in range(3)
                )
            )

        self.assertTrue(
            all(
                result.status is ShellExecutionStatus.COMPLETED
                for result in (*sandbox_results, *container_results)
            )
        )
        self.assertEqual(
            sandbox_backend.operations.count(SandboxBackendOperation.PROBE),
            len(sandbox_results),
        )
        self.assertEqual(
            container_backend.operations.count(
                ContainerBackendOperation.PROBE
            ),
            len(container_results),
        )

    async def test_bounded_output_survives_stream_pressure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sandbox = await ShellSandboxCommandExecutor(
                sandbox_settings=_sandbox_settings(
                    root, SandboxBackend.SEATBELT
                ),
                sandbox_backend=SandboxFakeBackend(
                    SandboxFakeBackendScript(
                        capabilities=_sandbox_capabilities(
                            SandboxBackend.SEATBELT
                        ),
                        stream_chunks=tuple(
                            SandboxStreamChunk(
                                stream=SandboxBackendStream.STDOUT,
                                content=b"0123456789",
                                sequence=index,
                            )
                            for index in range(20)
                        ),
                    )
                ),
            ).execute(_sandbox_spec(root, max_stdout_bytes=17))
            container = await ShellContainerCommandExecutor(
                container_settings=_container_settings(
                    ContainerBackend.DOCKER
                ),
                container_backend=ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_container_capabilities(
                            ContainerBackend.DOCKER
                        ),
                        stream_chunks=tuple(
                            ContainerBackendStreamChunk(
                                stream=ContainerBackendStream.STDOUT,
                                content=b"abcdefghij",
                                sequence=index,
                            )
                            for index in range(20)
                        ),
                    )
                ),
            ).execute(_container_spec(max_stdout_bytes=13))

        self.assertEqual(sandbox.stdout_bytes, 17)
        self.assertTrue(sandbox.stdout_truncated)
        self.assertEqual(container.stdout_bytes, 13)
        self.assertTrue(container.stdout_truncated)


class Phase10CoverageInventoryTest(TestCase):
    def test_e2e_inventory_covers_phase10_required_surfaces(self) -> None:
        _assert_complete_inventory(
            _PHASE10_E2E_DIMENSIONS,
            {
                "shell tools",
                "trusted agent runs",
                "strict flows",
                "direct tasks",
                "queued tasks",
                "worker execution",
                "server requests",
                "MCP",
                "A2A",
                "runtime envelopes",
            },
        )
        _assert_node_ids_exist(_PHASE10_E2E_DIMENSIONS)

    def test_performance_inventory_covers_phase10_required_gates(self) -> None:
        _assert_complete_inventory(
            _PHASE10_PERFORMANCE_DIMENSIONS,
            {
                "event-loop responsiveness",
                "stream backpressure",
                "bounded output",
                "cleanup deadlines",
                "bounded backend probe work",
                "worker throughput",
                "server streaming",
                "concurrent local/sandbox/container execution",
            },
        )
        _assert_node_ids_exist(_PHASE10_PERFORMANCE_DIMENSIONS)

    def test_optional_real_runtime_gates_have_clear_skip_diagnostics(
        self,
    ) -> None:
        for env_name, (path, skip_message) in _OPTIONAL_RUNTIME_GATES.items():
            with self.subTest(env_name=env_name):
                source = (_REPO_ROOT / path).read_text(encoding="utf-8")

                self.assertIn(env_name, source)
                self.assertIn(skip_message, source)


class _Ticker:
    def __init__(self) -> None:
        self.ticks = 0
        self._running = True

    def start(self):
        async def tick() -> None:
            while self._running:
                self.ticks += 1
                await sleep(0)

        return create_task(tick())

    def stop(self) -> None:
        self._running = False


def _local_spec(root: Path, command: str) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="local",
        tool_name="shell.phase10",
        command="sh",
        executable="/bin/sh",
        argv=("/bin/sh", "-c", command),
        display_argv=("sh", "-c", command),
        cwd=str(root),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=2,
        max_stdout_bytes=64,
        max_stderr_bytes=64,
    )


def _sandbox_spec(
    root: Path,
    *,
    max_stdout_bytes: int = 64,
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="sandbox",
        tool_name="shell.phase10",
        command="sh",
        executable="/bin/sh",
        argv=("/bin/sh", "-c", "printf sandbox"),
        display_argv=("sh", "-c", "printf sandbox"),
        cwd=str(root),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=2,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=64,
    )


def _container_spec(*, max_stdout_bytes: int = 64) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="container",
        tool_name="shell.phase10",
        command="phase10",
        executable="/trusted/bin/phase10",
        argv=("/trusted/bin/phase10",),
        display_argv=("phase10",),
        cwd=str(Path.cwd()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=2,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=64,
    )


def _sandbox_settings(
    root: Path,
    backend: SandboxBackend,
) -> SandboxEffectiveSettings:
    settings = IsolationSettings.from_dict(
        {
            "mode": "sandbox",
            "sandbox": _sandbox_settings_raw(backend.value, root),
        },
        source=trusted_isolation_source("sdk"),
    )
    effective = settings.select_profile(
        IsolationProfileSelection(
            mode=IsolationMode.SANDBOX,
            profile="phase10",
            required=True,
        )
    )
    assert effective.sandbox is not None
    return effective.sandbox


def _sandbox_settings_raw(backend: str, root: Path) -> dict[str, object]:
    return {
        "backend": backend,
        "default_profile": "phase10",
        "allowed_profiles": ["phase10"],
        "profiles": {
            "phase10": {
                "name": "phase10",
                "trusted_executables": ["/bin/sh"],
                "executable_search_roots": ["/bin"],
                "read_roots": [str(root)],
                "write_roots": [str(root)],
                "deny_roots": [str(root / "denied")],
                "scratch_roots": [str(root)],
                "output_roots": [str(root)],
                "environment": {"variables": {"LC_ALL": "C"}},
                "network": {"mode": "none"},
                "resources": {"timeout_seconds": 2, "pids": 16},
                "output": {"allow_artifacts": False},
                "child_processes": "allow",
            },
        },
        "policy_version": "phase10",
        "profile_registry_id": "phase10",
    }


def _sandbox_capabilities(
    backend: SandboxBackend,
    *,
    write_roots: bool = True,
) -> SandboxBackendCapabilities:
    return SandboxBackendCapabilities(
        backend=backend,
        host_os="linux" if backend is SandboxBackend.BUBBLEWRAP else "darwin",
        architecture="amd64",
        runtime_name=f"fake-{backend.value}",
        sandbox_executable=f"/usr/bin/{backend.value}",
        sandbox_executable_available=True,
        filesystem=SandboxFilesystemControls(
            read_roots=True,
            write_roots=write_roots,
            deny_roots=True,
        ),
        network_modes=(SandboxNetworkMode.NONE,),
        process=SandboxProcessControls(
            process_limits=True,
            child_processes=True,
            inherited_fds=True,
        ),
        temp_output=SandboxTempOutputMapping(
            temp_dirs=True,
            output_dirs=True,
            cleanup_budget=True,
        ),
    )


def _container_settings(
    backend: ContainerBackend,
    *,
    required: bool = True,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name="phase10",
        image_reference=_IMAGE,
    )
    return ContainerEffectiveSettings(
        backend=backend,
        required=required,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=ContainerSettingsSource(
            surface=ContainerSurface.SDK,
            trust_level=ContainerTrustLevel.TRUSTED_OPERATOR,
        ),
        policy_version="phase10",
        profile_registry_id="phase10",
        profile_name=profile.name,
        profile=profile,
        allowed_profiles=(profile.name,),
    )


def _container_capabilities(
    backend: ContainerBackend,
    *,
    workspace_mounts: bool = True,
) -> ContainerBackendCapabilities:
    mount_types = (
        (ContainerMountType.WORKSPACE, ContainerMountType.OUTPUT)
        if workspace_mounts
        else (ContainerMountType.OUTPUT,)
    )
    return ContainerBackendCapabilities(
        backend=backend,
        host_os=(
            "darwin"
            if backend is ContainerBackend.APPLE_CONTAINER
            else "linux"
        ),
        guest_os="linux",
        architecture="amd64",
        support_level=(
            ContainerBackendSupportLevel.OPT_IN
            if backend is ContainerBackend.APPLE_CONTAINER
            else ContainerBackendSupportLevel.SUPPORTED
        ),
        rootless=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=mount_types,
        device_classes=(ContainerDeviceClass.CPU,),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _assert_complete_inventory(
    inventory: dict[str, tuple[str, ...]],
    expected_dimensions: set[str],
) -> None:
    assert set(inventory) == expected_dimensions
    for node_ids in inventory.values():
        assert node_ids


def _assert_node_ids_exist(
    inventory: dict[str, tuple[str, ...]],
) -> None:
    for dimension, node_ids in inventory.items():
        for node_id in node_ids:
            try:
                _assert_node_id_exists(node_id)
            except AssertionError as exc:
                raise AssertionError(f"{dimension}: {node_id}: {exc}") from exc


def _assert_node_id_exists(node_id: str) -> None:
    path, *qualifiers = node_id.split("::")
    assert qualifiers, f"{node_id} must include a test qualifier"
    source_path = _REPO_ROOT / path
    assert source_path.exists(), f"{path} does not exist"
    source = source_path.read_text(encoding="utf-8")
    for qualifier in qualifiers:
        if qualifier.startswith("test_"):
            assert (
                f"def {qualifier}" in source
                or f"async def {qualifier}" in source
            ), f"{node_id} is missing {qualifier}"
        else:
            assert (
                f"class {qualifier}" in source
            ), f"{node_id} is missing {qualifier}"


if __name__ == "__main__":
    main()
