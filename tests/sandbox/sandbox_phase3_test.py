from asyncio import create_task, gather
from asyncio import run as run_async
from asyncio import sleep as async_sleep
from collections.abc import Mapping
from typing import cast
from unittest import TestCase, main

from avalan.isolation import (
    IsolationMode,
    IsolationProfileSelection,
    IsolationSettings,
    SandboxBackend,
    SandboxChildProcessPolicy,
    SandboxCleanupPolicy,
    SandboxEffectiveSettings,
    SandboxInheritedFdPolicy,
    SandboxNetworkMode,
    trusted_isolation_source,
)
from avalan.sandbox import (
    SandboxBackendCapabilities,
    SandboxBackendDiagnostic,
    SandboxBackendDiagnosticCode,
    SandboxBackendOperation,
    SandboxBackendProbeResult,
    SandboxBackendStream,
    SandboxExecutionPlan,
    SandboxExecutionResult,
    SandboxFakeBackend,
    SandboxFakeBackendScript,
    SandboxFilesystemControls,
    SandboxOutputArtifact,
    SandboxPlanRequest,
    SandboxPlanRequestKind,
    SandboxProcessControls,
    SandboxResultStatus,
    SandboxStreamChunk,
    SandboxTempOutputMapping,
    sandbox_backend_capability_profile,
    sandbox_backend_capability_profiles,
    sandbox_backend_probe_from_profile,
    select_sandbox_backend,
)


class SandboxPhase3Test(TestCase):
    def test_capability_catalog_reports_backend_controls(self) -> None:
        seatbelt = sandbox_backend_capability_profile("seatbelt-darwin")
        bubblewrap = sandbox_backend_capability_profile("bubblewrap-linux")
        all_profiles = sandbox_backend_capability_profiles()
        seatbelt_profiles = sandbox_backend_capability_profiles("seatbelt")
        unavailable = seatbelt.probe()
        available = bubblewrap.probe(available=True)
        diagnostic = SandboxBackendDiagnostic(
            code=SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
            operation=SandboxBackendOperation.PROBE,
            backend=SandboxBackend.SEATBELT,
            message="reduced controls",
        )
        diagnostic_probe = sandbox_backend_probe_from_profile(
            "seatbelt-darwin",
            available=False,
            diagnostics=(diagnostic,),
        )

        self.assertEqual(
            {profile.backend for profile in all_profiles},
            {SandboxBackend.SEATBELT, SandboxBackend.BUBBLEWRAP},
        )
        self.assertEqual(seatbelt_profiles, (seatbelt,))
        self.assertFalse(unavailable.ok)
        self.assertIsNone(unavailable.capabilities)
        self.assertEqual(
            unavailable.diagnostics[0].code,
            SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertTrue(available.ok)
        assert available.capabilities is not None
        capability_dict = available.capabilities.to_dict()
        self.assertEqual(capability_dict["host_os"], "linux")
        self.assertEqual(
            capability_dict["filesystem"],
            {
                "read_roots": True,
                "write_roots": True,
                "deny_roots": True,
                "path_normalization": True,
            },
        )
        self.assertIn(
            "network_allowlist",
            cast(list[str], capability_dict["unsupported_controls"]),
        )
        self.assertEqual(
            bubblewrap.to_dict()["profile_id"], "bubblewrap-linux"
        )
        self.assertFalse(diagnostic_probe.ok)
        self.assertEqual(diagnostic_probe.diagnostics, (diagnostic,))
        self.assertEqual(diagnostic.to_dict()["message"], "reduced controls")
        self.assertEqual(available.to_dict()["backend"], "bubblewrap")
        with self.assertRaises(AssertionError):
            sandbox_backend_capability_profile("missing")

    def test_plan_fingerprint_is_deterministic_and_validates_roots(
        self,
    ) -> None:
        effective = _effective("bubblewrap")
        request = SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.TYPED_TOOL,
            logical_name="shell",
            command="/bin/sh",
            argv=("/bin/sh", "-lc", "echo ok"),
            cwd="/workspace/./project",
            request_id="request-1",
        )
        plan = SandboxExecutionPlan(
            request=request,
            settings=effective,
            environment={"PATH": "/bin", "LC_ALL": "C.UTF-8"},
            temp_dir="/tmp/avalan/session",
            output_dir="/workspace/out/session",
            collect_outputs=True,
            cleanup_budget_seconds=1.5,
            stream_buffer_bytes=128,
        )
        equivalent = SandboxExecutionPlan(
            request=SandboxPlanRequest(
                request_kind="typed_tool",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh", "-lc", "echo ok"),
                cwd="/workspace/project",
                request_id="request-1",
            ),
            settings=effective,
            environment={"LC_ALL": "C.UTF-8", "PATH": "/bin"},
            temp_dir="/tmp/avalan/session",
            output_dir="/workspace/out/session",
            collect_outputs=True,
            cleanup_budget_seconds=1.5,
            stream_buffer_bytes=128,
        )

        self.assertEqual(plan.plan_fingerprint, equivalent.plan_fingerprint)
        self.assertEqual(
            plan.to_dict()["plan_fingerprint"], plan.plan_fingerprint
        )
        self.assertEqual(
            plan.canonical_policy_input()["environment"],
            {"LC_ALL": "C.UTF-8", "PATH": "/bin"},
        )
        self.assertEqual(request.to_dict()["cwd"], "/workspace/project")
        self.assertEqual(
            SandboxPlanRequest(
                request_kind="task_attempt",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh",),
                cwd="/workspace",
                attempt_id="attempt-1",
            ).to_dict()["attempt_id"],
            "attempt-1",
        )

        with self.assertRaises(AssertionError):
            SandboxPlanRequest(
                request_kind="unknown",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh",),
                cwd="/workspace",
            )
        with self.assertRaises(AssertionError):
            SandboxPlanRequest(
                request_kind="task_attempt",
                logical_name="shell",
                command="/bin/sh",
                argv=("/bin/sh",),
                cwd="/workspace",
            )
        with self.assertRaises(AssertionError):
            SandboxPlanRequest(
                request_kind="typed_tool",
                logical_name="shell",
                command="sh",
                argv=("/bin/sh",),
                cwd="/workspace",
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=SandboxPlanRequest(
                    request_kind="typed_tool",
                    logical_name="shell",
                    command="/usr/bin/env",
                    argv=("/usr/bin/env",),
                    cwd="/workspace",
                ),
                settings=effective,
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=SandboxPlanRequest(
                    request_kind="typed_tool",
                    logical_name="shell",
                    command="/bin/sh",
                    argv=("/bin/sh",),
                    cwd="/etc/ssh",
                ),
                settings=effective,
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=request,
                settings=effective,
                temp_dir="/tmp/outside",
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=request,
                settings=effective,
                output_dir="/workspace/private",
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=request,
                settings=effective,
                environment={"SECRET_TOKEN": "value"},
            )
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=request,
                settings=effective,
                environment={"LC_ALL": "en_US.UTF-8"},
            )

    def test_backend_selection_accepts_and_rejects_capabilities(self) -> None:
        bubblewrap_plan = _plan("bubblewrap")
        bubblewrap_probe = sandbox_backend_probe_from_profile(
            "bubblewrap-linux",
            available=True,
        )
        selected = select_sandbox_backend(bubblewrap_plan, (bubblewrap_probe,))
        unavailable = select_sandbox_backend(bubblewrap_plan, ())
        unavailable_probe = select_sandbox_backend(
            bubblewrap_plan,
            (sandbox_backend_probe_from_profile("bubblewrap-linux"),),
        )
        seatbelt_selection = select_sandbox_backend(
            _plan("seatbelt", pids=32),
            (
                sandbox_backend_probe_from_profile(
                    "seatbelt-darwin", available=True
                ),
            ),
        )
        reduced = select_sandbox_backend(
            _plan("bubblewrap", pids=32),
            (_probe_with_reduced_controls(),),
        )
        unenforceable_network = select_sandbox_backend(
            _plan("bubblewrap", network="allowlist", egress=("api",)),
            (bubblewrap_probe,),
        )

        self.assertTrue(selected.ok)
        self.assertEqual(selected.backend, SandboxBackend.BUBBLEWRAP)
        assert selected.capabilities is not None
        self.assertEqual(
            selected.capabilities.backend, SandboxBackend.BUBBLEWRAP
        )
        self.assertEqual(
            bubblewrap_plan.settings.profile.child_processes,
            SandboxChildProcessPolicy.ALLOW,
        )
        self.assertEqual(
            bubblewrap_plan.settings.profile.inherited_fds,
            SandboxInheritedFdPolicy.EXPLICIT,
        )
        self.assertEqual(
            bubblewrap_plan.settings.profile.cleanup,
            SandboxCleanupPolicy.DELETE,
        )
        self.assertFalse(unavailable.ok)
        self.assertEqual(
            unavailable.diagnostics[0].code,
            SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertFalse(unavailable_probe.ok)
        self.assertFalse(seatbelt_selection.ok)
        self.assertIn(
            "pid limits unsupported",
            {
                diagnostic.message
                for diagnostic in seatbelt_selection.diagnostics
            },
        )
        reduced_messages = {
            diagnostic.message for diagnostic in reduced.diagnostics
        }
        self.assertIn("sandbox executable is unavailable", reduced_messages)
        self.assertIn("read roots unsupported", reduced_messages)
        self.assertIn("write roots unsupported", reduced_messages)
        self.assertIn("deny roots unsupported", reduced_messages)
        self.assertIn("network mode loopback is unsupported", reduced_messages)
        self.assertIn("pid limits unsupported", reduced_messages)
        self.assertIn("child process policy unsupported", reduced_messages)
        self.assertIn("inherited fd policy unsupported", reduced_messages)
        self.assertIn("temp dirs unsupported", reduced_messages)
        self.assertIn("output dirs unsupported", reduced_messages)
        self.assertIn("cleanup budgets unsupported", reduced_messages)
        self.assertFalse(unenforceable_network.ok)
        self.assertIn(
            "network mode allowlist is unsupported",
            {
                diagnostic.message
                for diagnostic in unenforceable_network.diagnostics
            },
        )

    def test_fake_backend_success_streams_outputs_and_cleanup(self) -> None:
        backend = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(
                    SandboxStreamChunk(
                        stream=SandboxBackendStream.STDOUT,
                        content=b"hello",
                        sequence=0,
                    ),
                    SandboxStreamChunk(
                        stream=SandboxBackendStream.STDERR,
                        content=b"warn",
                        sequence=1,
                    ),
                ),
                stream_delay_seconds=0.001,
                output_files={"result.txt": b"done"},
            )
        )
        probe = SandboxFakeBackend(
            SandboxFakeBackendScript(capabilities=_capabilities())
        )
        no_timeout = SandboxFakeBackend(
            SandboxFakeBackendScript(capabilities=_capabilities())
        )

        result = run_async(backend.execute(_plan("bubblewrap")))
        probe_result = run_async(probe.probe())
        no_timeout_result = run_async(
            no_timeout.execute(_plan("bubblewrap", timeout_seconds=None))
        )

        self.assertTrue(result.ok)
        self.assertTrue(probe_result.ok)
        self.assertEqual(probe_result.to_dict()["backend"], "bubblewrap")
        self.assertTrue(no_timeout_result.ok)
        self.assertEqual(result.stdout, b"hello")
        self.assertEqual(result.stderr, b"warn")
        self.assertEqual(
            result.output_artifacts[0].to_dict(),
            {"path": "result.txt", "size_bytes": 4},
        )
        self.assertEqual(result.stream_chunks[0].to_dict()["content"], "hello")
        self.assertIn(SandboxBackendOperation.CLEANUP, backend.operations)
        self.assertEqual(result.to_dict()["status"], "completed")
        self.assertEqual(
            SandboxOutputArtifact(path="x.txt", content=b"x").size_bytes,
            1,
        )

    def test_fake_backend_failure_denial_timeout_and_cancellation(
        self,
    ) -> None:
        failed = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_exit_code=7,
                )
            ).execute(_plan("bubblewrap"))
        )
        denied = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    denied_paths=("/workspace/project/secret",),
                )
            ).execute(
                _plan(
                    "bubblewrap",
                    argv=("/bin/sh", "/workspace/project/secret/file.txt"),
                )
            )
        )
        timeout = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    timeout_operations=(SandboxBackendOperation.WAIT,),
                )
            ).execute(_plan("bubblewrap"))
        )
        cancelled = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    cancel_operations=(SandboxBackendOperation.START,),
                )
            ).execute(_plan("bubblewrap"))
        )
        start_failed = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_diagnostics={
                        SandboxBackendOperation.START: (
                            SandboxBackendDiagnosticCode.EXECUTABLE_DENIED
                        )
                    },
                )
            ).execute(_plan("bubblewrap"))
        )

        self.assertEqual(failed.status, SandboxResultStatus.FAILED)
        self.assertEqual(failed.exit_code, 7)
        self.assertEqual(
            failed.diagnostics[0].code,
            SandboxBackendDiagnosticCode.EXECUTION_FAILED,
        )
        self.assertEqual(denied.status, SandboxResultStatus.DENIED)
        self.assertEqual(
            denied.diagnostics[0].code,
            SandboxBackendDiagnosticCode.PATH_DENIED,
        )
        self.assertEqual(timeout.status, SandboxResultStatus.TIMED_OUT)
        self.assertEqual(
            timeout.diagnostics[0].code,
            SandboxBackendDiagnosticCode.TIMEOUT,
        )
        self.assertEqual(cancelled.status, SandboxResultStatus.CANCELLED)
        self.assertEqual(start_failed.status, SandboxResultStatus.DENIED)
        self.assertIs(start_failed.diagnostics[0].retryable, False)

    def test_fake_backend_stream_truncates_and_rejects_outputs(self) -> None:
        truncated = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_chunks=(
                        SandboxStreamChunk(
                            stream="stdout",
                            content=b"abc",
                            sequence=0,
                        ),
                        SandboxStreamChunk(
                            stream="stdout",
                            content=b"d",
                            sequence=1,
                        ),
                    ),
                )
            ).execute(_plan("bubblewrap", stream_buffer_bytes=3))
        )
        no_output = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(capabilities=_capabilities())
            ).execute(_plan("bubblewrap", collect_outputs=False))
        )
        disabled_output = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    output_files={"result.txt": b"x"},
                )
            ).execute(_plan("bubblewrap", collect_outputs=False))
        )
        unsafe_output = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    output_files={"../result.txt": b"x"},
                )
            ).execute(_plan("bubblewrap"))
        )
        large_output = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    output_files={"large.txt": b"x" * 2048},
                )
            ).execute(_plan("bubblewrap"))
        )

        self.assertEqual(truncated.stdout, b"abc")
        self.assertTrue(truncated.stream_truncated)
        self.assertEqual(
            truncated.diagnostics[0].code,
            SandboxBackendDiagnosticCode.STREAM_TRUNCATED,
        )
        self.assertTrue(no_output.ok)
        for result in (disabled_output, unsafe_output, large_output):
            self.assertEqual(result.status, SandboxResultStatus.FAILED)
            self.assertEqual(
                result.diagnostics[0].code,
                SandboxBackendDiagnosticCode.OUTPUT_REJECTED,
            )

    def test_fake_backend_probe_and_cleanup_budgets_are_bounded(self) -> None:
        slow_probe = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                probe_delay_seconds=0.02,
            )
        )
        unavailable_probe = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                available=False,
            )
        )
        cleanup_uncertain = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                cleanup_uncertain=True,
            )
        )
        cleanup_diagnostic = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                operation_diagnostics={
                    SandboxBackendOperation.CLEANUP: (
                        SandboxBackendDiagnosticCode.CLEANUP_FAILED
                    )
                },
            )
        )
        cleanup_timeout = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                operation_delay_seconds={
                    SandboxBackendOperation.CLEANUP: 0.03,
                },
            )
        )
        cleanup_cancel = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                cancel_operations=(SandboxBackendOperation.CLEANUP,),
            )
        )
        cleanup_operation_timeout = SandboxFakeBackend(
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                timeout_operations=(SandboxBackendOperation.CLEANUP,),
            )
        )

        probe_result = run_async(slow_probe.probe(timeout_seconds=0.001))
        unavailable_result = run_async(unavailable_probe.probe())
        uncertain_result = run_async(
            cleanup_uncertain.execute(_plan("bubblewrap"))
        )
        diagnostic_result = run_async(
            cleanup_diagnostic.execute(_plan("bubblewrap"))
        )
        timeout_result = run_async(
            cleanup_timeout.execute(
                _plan("bubblewrap", cleanup_budget_seconds=0.001)
            )
        )
        cancel_result = run_async(cleanup_cancel.execute(_plan("bubblewrap")))
        operation_timeout_result = run_async(
            cleanup_operation_timeout.execute(_plan("bubblewrap"))
        )

        self.assertFalse(probe_result.ok)
        self.assertEqual(
            probe_result.diagnostics[0].code,
            SandboxBackendDiagnosticCode.TIMEOUT,
        )
        self.assertFalse(unavailable_result.ok)
        self.assertEqual(
            unavailable_result.diagnostics[0].code,
            SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        for result in (
            uncertain_result,
            diagnostic_result,
            timeout_result,
            cancel_result,
            operation_timeout_result,
        ):
            self.assertEqual(result.status, SandboxResultStatus.FAILED)
            self.assertTrue(result.cleanup_uncertain)
        self.assertEqual(
            cancel_result.diagnostics[-1].code,
            SandboxBackendDiagnosticCode.CANCELLED,
        )

    def test_fake_backend_fails_closed_on_unavailable_or_wrong_backend(
        self,
    ) -> None:
        unavailable = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    available=False,
                )
            ).execute(_plan("bubblewrap"))
        )
        wrong_backend = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(capabilities=_capabilities())
            ).execute(_plan("seatbelt"))
        )
        unsupported_network = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(capabilities=_capabilities())
            ).execute(
                _plan("bubblewrap", network="allowlist", egress=("api",))
            )
        )

        self.assertEqual(unavailable.status, SandboxResultStatus.FAILED)
        self.assertEqual(
            unavailable.diagnostics[0].code,
            SandboxBackendDiagnosticCode.BACKEND_UNAVAILABLE,
        )
        self.assertEqual(wrong_backend.status, SandboxResultStatus.DENIED)
        self.assertEqual(
            wrong_backend.diagnostics[0].code,
            SandboxBackendDiagnosticCode.CAPABILITY_MISMATCH,
        )
        self.assertEqual(
            unsupported_network.status,
            SandboxResultStatus.DENIED,
        )
        self.assertIn(
            "network mode allowlist is unsupported",
            {
                diagnostic.message
                for diagnostic in unsupported_network.diagnostics
            },
        )

    def test_fake_backend_limits_concurrent_executions(self) -> None:
        async def run_two() -> tuple[
            SandboxExecutionResult,
            SandboxExecutionResult,
            int,
        ]:
            backend = SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        SandboxBackendOperation.START: 0.03,
                    },
                    max_concurrent_executions=1,
                )
            )
            first_task = create_task(backend.execute(_plan("bubblewrap")))
            await async_sleep(0)
            second = await backend.execute(_plan("bubblewrap"))
            first = await first_task
            return first, second, backend.max_observed_concurrent_executions

        first_result, second_result, max_observed = run_async(run_two())
        self.assertEqual(first_result.status, SandboxResultStatus.COMPLETED)
        self.assertEqual(second_result.status, SandboxResultStatus.DENIED)
        self.assertEqual(
            second_result.diagnostics[0].code,
            SandboxBackendDiagnosticCode.CONCURRENCY_LIMIT,
        )
        self.assertEqual(max_observed, 1)

    def test_fake_backend_allows_bounded_concurrent_batch(self) -> None:
        async def run_batch() -> tuple[SandboxResultStatus, ...]:
            backend = SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        SandboxBackendOperation.START: 0.005,
                    },
                    max_concurrent_executions=3,
                )
            )
            results = await gather(
                backend.execute(_plan("bubblewrap")),
                backend.execute(_plan("bubblewrap")),
                backend.execute(_plan("bubblewrap")),
            )
            self.assertEqual(backend.max_observed_concurrent_executions, 3)
            return tuple(
                cast(SandboxResultStatus, result.status) for result in results
            )

        self.assertEqual(
            run_async(run_batch()),
            (
                SandboxResultStatus.COMPLETED,
                SandboxResultStatus.COMPLETED,
                SandboxResultStatus.COMPLETED,
            ),
        )

    def test_script_and_result_validation_reject_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                probe_delay_seconds=-1,
            )
        with self.assertRaises(AssertionError):
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                operation_diagnostics={"bad": "sandbox.backend.timeout"},
            )
        with self.assertRaises(AssertionError):
            SandboxFakeBackendScript(
                capabilities=_capabilities(),
                output_files=cast(
                    Mapping[str, bytes],
                    {"x": "not-bytes"},
                ),
            )
        with self.assertRaises(AssertionError):
            SandboxStreamChunk(stream="stdout", content=b"x", sequence=-1)
        with self.assertRaises(AssertionError):
            SandboxOutputArtifact(path="", content=b"x")

    def test_container_only_fields_under_sandbox_policy_are_rejected(
        self,
    ) -> None:
        raw = _settings_raw("bubblewrap")
        sandbox = raw["sandbox"]
        assert isinstance(sandbox, dict)
        profiles = sandbox["profiles"]
        assert isinstance(profiles, dict)
        profile = profiles["host-tools"]
        assert isinstance(profile, dict)
        profile["mounts"] = []

        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                raw,
                source=trusted_isolation_source("sdk"),
            )

    def test_negative_sandbox_policy_matrix_rejects_unsafe_inputs(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                _settings_raw("bubblewrap", network="full", egress=("api",)),
                source=trusted_isolation_source("sdk"),
            )
        raw = _settings_raw("bubblewrap")
        sandbox = raw["sandbox"]
        assert isinstance(sandbox, dict)
        profiles = sandbox["profiles"]
        assert isinstance(profiles, dict)
        profile = profiles["host-tools"]
        assert isinstance(profile, dict)
        environment = profile["environment"]
        assert isinstance(environment, dict)
        environment["inherit_host"] = True
        with self.assertRaises(AssertionError):
            IsolationSettings.from_dict(
                raw,
                source=trusted_isolation_source("sdk"),
            )

        for field in (
            "trusted_executables",
            "executable_search_roots",
            "read_roots",
            "write_roots",
            "deny_roots",
            "scratch_roots",
            "output_roots",
        ):
            raw = _settings_raw("bubblewrap")
            sandbox = raw["sandbox"]
            assert isinstance(sandbox, dict)
            profiles = sandbox["profiles"]
            assert isinstance(profiles, dict)
            profile = profiles["host-tools"]
            assert isinstance(profile, dict)
            profile[field] = ["relative"]
            with self.assertRaises(AssertionError):
                IsolationSettings.from_dict(
                    raw,
                    source=trusted_isolation_source("sdk"),
                )

        raw = _settings_raw("bubblewrap")
        sandbox = raw["sandbox"]
        assert isinstance(sandbox, dict)
        profiles = sandbox["profiles"]
        assert isinstance(profiles, dict)
        profile = profiles["host-tools"]
        assert isinstance(profile, dict)
        profile["output_roots"] = []
        no_output_roots = IsolationSettings.from_dict(
            raw,
            source=trusted_isolation_source("sdk"),
        ).select_profile(
            IsolationProfileSelection(
                mode=IsolationMode.SANDBOX,
                profile="host-tools",
                required=True,
            )
        )
        assert no_output_roots.sandbox is not None
        with self.assertRaises(AssertionError):
            SandboxExecutionPlan(
                request=SandboxPlanRequest(
                    request_kind="typed_tool",
                    logical_name="shell",
                    command="/bin/sh",
                    argv=("/bin/sh",),
                    cwd="/workspace",
                ),
                settings=no_output_roots.sandbox,
                output_dir="/workspace/out/file",
            )

        sensitive = run_async(
            SandboxFakeBackend(
                SandboxFakeBackendScript(
                    capabilities=_capabilities(),
                    denied_paths=(
                        "/workspace/project/.ssh",
                        "/workspace/project/.aws",
                        "/var/run/docker.sock",
                    ),
                )
            ).execute(
                _plan(
                    "bubblewrap",
                    argv=(
                        "/bin/sh",
                        "/workspace/project/.ssh/id_rsa",
                        "/workspace/project/.aws/credentials",
                        "/var/run/docker.sock",
                    ),
                )
            )
        )
        self.assertEqual(sensitive.status, SandboxResultStatus.DENIED)
        self.assertEqual(
            sensitive.diagnostics[0].code,
            SandboxBackendDiagnosticCode.PATH_DENIED,
        )


def _effective(
    backend: str,
    *,
    network: str = "loopback",
    egress: tuple[str, ...] = (),
    pids: int | None = None,
    child_processes: str = "allow",
    inherited_fds: str = "explicit",
    cleanup: str = "delete",
    timeout_seconds: int | None = 1,
) -> SandboxEffectiveSettings:
    settings = IsolationSettings.from_dict(
        _settings_raw(
            backend,
            network=network,
            egress=egress,
            pids=pids,
            child_processes=child_processes,
            inherited_fds=inherited_fds,
            cleanup=cleanup,
            timeout_seconds=timeout_seconds,
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
    *,
    network: str = "loopback",
    egress: tuple[str, ...] = (),
    pids: int | None = None,
    child_processes: str = "allow",
    inherited_fds: str = "explicit",
    cleanup: str = "delete",
    timeout_seconds: int | None = 1,
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
                    "read_roots": ["/workspace"],
                    "write_roots": ["/workspace/out"],
                    "deny_roots": ["/etc/ssh"],
                    "scratch_roots": ["/tmp/avalan"],
                    "output_roots": ["/workspace/out"],
                    "environment": {
                        "variables": {"LC_ALL": "C.UTF-8"},
                        "allowlist": ["PATH"],
                    },
                    "network": {
                        "mode": network,
                        "egress_allowlist": list(egress),
                    },
                    "resources": {
                        "timeout_seconds": timeout_seconds,
                        "pids": pids,
                    },
                    "output": {
                        "max_stdout_bytes": 1024,
                        "max_stderr_bytes": 1024,
                        "allow_artifacts": True,
                        "max_artifact_bytes": 1024,
                    },
                    "child_processes": child_processes,
                    "inherited_fds": inherited_fds,
                    "cleanup": cleanup,
                },
            },
            "profile_registry_id": "default",
            "policy_version": "phase3",
        },
    }


def _plan(
    backend: str,
    *,
    argv: tuple[str, ...] = ("/bin/sh", "-lc", "echo ok"),
    network: str = "loopback",
    egress: tuple[str, ...] = (),
    collect_outputs: bool = True,
    stream_buffer_bytes: int = 4096,
    cleanup_budget_seconds: float = 0.5,
    timeout_seconds: int | None = 1,
    pids: int | None = None,
) -> SandboxExecutionPlan:
    effective = _effective(
        backend,
        network=network,
        egress=egress,
        pids=pids,
        timeout_seconds=timeout_seconds,
    )
    assert effective is not None
    return SandboxExecutionPlan(
        request=SandboxPlanRequest(
            request_kind=SandboxPlanRequestKind.TYPED_TOOL,
            logical_name="shell",
            command="/bin/sh",
            argv=argv,
            cwd="/workspace/project",
        ),
        settings=effective,
        temp_dir="/tmp/avalan/session",
        output_dir="/workspace/out/session",
        collect_outputs=collect_outputs,
        stream_buffer_bytes=stream_buffer_bytes,
        cleanup_budget_seconds=cleanup_budget_seconds,
    )


def _capabilities() -> SandboxBackendCapabilities:
    return sandbox_backend_capability_profile("bubblewrap-linux").capabilities


def _probe_with_reduced_controls() -> SandboxBackendProbeResult:
    capabilities = SandboxBackendCapabilities(
        backend=SandboxBackend.BUBBLEWRAP,
        host_os="linux",
        architecture="amd64",
        runtime_name="reduced bubblewrap",
        sandbox_executable="/usr/bin/bwrap",
        sandbox_executable_available=False,
        filesystem=SandboxFilesystemControls(
            read_roots=False,
            write_roots=False,
            deny_roots=False,
        ),
        network_modes=(SandboxNetworkMode.NONE,),
        process=SandboxProcessControls(
            process_limits=False,
            child_processes=False,
            inherited_fds=False,
        ),
        temp_output=SandboxTempOutputMapping(
            temp_dirs=False,
            output_dirs=False,
            cleanup_budget=False,
        ),
    )
    return SandboxBackendProbeResult(
        backend=SandboxBackend.BUBBLEWRAP,
        available=True,
        capabilities=SandboxFakeBackendScript(
            capabilities=capabilities,
        ).capabilities,
    )


if __name__ == "__main__":
    main()
