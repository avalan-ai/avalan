from asyncio import CancelledError, create_task, sleep
from asyncio import run as run_async
from collections.abc import AsyncIterable, Iterator, Sequence
from typing import cast
from unittest import TestCase, main

from avalan.container import (
    ContainerAsyncBackend,
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendContainer,
    ContainerBackendDiagnosticCode,
    ContainerBackendImageResolution,
    ContainerBackendInspection,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendProbeResult,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendWaitResult,
    ContainerBuildPolicy,
    ContainerCommandPlan,
    ContainerExecutionResult,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerImagePolicy,
    ContainerLifecycleCleanup,
    ContainerLifecycleDeadlines,
    ContainerLifecycleEvent,
    ContainerLifecycleEventPolicy,
    ContainerLifecycleEventStatus,
    ContainerLifecyclePhase,
    ContainerManagedLifecycleResult,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputValidationResult,
    ContainerPullPolicy,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerStreamDrainPolicy,
    drain_container_streams,
    run_container_managed_lifecycle,
)

_DIGEST = "9" * 64
_IMAGE = f"ghcr.io/example/lifecycle-tools@sha256:{_DIGEST}"


class ContainerLifecycleTest(TestCase):
    def test_successful_lifecycle_order_streams_stats_deadlines_cleanup(
        self,
    ) -> None:
        contract = _output_contract()
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(pull=True),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"out-1",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"err-1",
                        sequence=1,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"out-2",
                        sequence=2,
                    ),
                ),
                stats_samples=(
                    ContainerBackendStats(
                        cpu_nanos=5,
                        memory_bytes=1024,
                        pids=2,
                    ),
                ),
                output_result=ContainerOutputValidationResult(
                    decision=ContainerOutputDecisionType.ACCEPT,
                    contract=contract,
                ),
            )
        )
        deadlines = ContainerLifecycleDeadlines(
            pull_seconds=10,
            execution_seconds=5,
            cleanup_seconds=7,
            parent_seconds=3,
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                output_contract=contract,
                deadlines=deadlines,
            )
        )
        started_phases = [
            event.phase
            for event in result.events
            if event.status is ContainerLifecycleEventStatus.STARTED
        ]

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertTrue(result.cleanup_completed)
        self.assertFalse(result.cleanup_uncertain)
        self.assertEqual(
            started_phases,
            [
                ContainerLifecyclePhase.POLICY_NORMALIZATION,
                ContainerLifecyclePhase.BACKEND_SELECTION,
                ContainerLifecyclePhase.IMAGE_RESOLUTION,
                ContainerLifecyclePhase.IMAGE_PULL,
                ContainerLifecyclePhase.CREATE,
                ContainerLifecyclePhase.ATTACH,
                ContainerLifecyclePhase.START,
                ContainerLifecyclePhase.STREAM,
                ContainerLifecyclePhase.STATS,
                ContainerLifecyclePhase.WAIT,
                ContainerLifecyclePhase.INSPECT,
                ContainerLifecyclePhase.COPY_OUTPUTS,
                ContainerLifecyclePhase.REMOVE,
                ContainerLifecyclePhase.CLEANUP,
            ],
        )
        self.assertEqual(
            [chunk.to_dict()["content"] for chunk in result.stream.chunks],
            ["out-1", "err-1", "out-2"],
        )
        self.assertEqual(
            [
                chunk.to_dict()["content"]
                for chunk in result.stream.stdout_chunks
            ],
            ["out-1", "out-2"],
        )
        self.assertEqual(
            [
                chunk.to_dict()["content"]
                for chunk in result.stream.stderr_chunks
            ],
            ["err-1"],
        )
        self.assertEqual(result.stats[0].to_dict()["memory_bytes"], 1024)
        self.assertEqual(deadlines.effective_seconds("image_pull"), 3)
        self.assertEqual(
            result.to_backend_result().execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertEqual(result.to_dict()["cleanup_completed"], True)

    def test_public_backend_run_uses_managed_lifecycle_bounds(self) -> None:
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"x" * 9000,
                        sequence=0,
                    ),
                ),
            )
        )

        result = run_async(backend.run(_run_plan()))

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.COMPLETED,
        )
        self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
        self.assertEqual(len(result.stream_chunks[0].content), 8192)
        self.assertIn(
            ContainerBackendDiagnosticCode.STREAM_TRUNCATED,
            {diagnostic.code for diagnostic in result.diagnostics},
        )

    def test_stream_drain_truncates_and_drops_bounded_output(self) -> None:
        chunks = (
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ",
                sequence=0,
            ),
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDERR,
                content=b"stderr",
                sequence=1,
            ),
            ContainerBackendStreamChunk(
                stream=ContainerBackendStream.STDOUT,
                content=b"dropped",
                sequence=2,
            ),
        )
        result = drain_container_streams(
            chunks,
            ContainerStreamDrainPolicy(
                max_chunks=2,
                max_bytes=46,
                max_chunk_bytes=40,
            ),
        )
        codes = {diagnostic.code for diagnostic in result.diagnostics}

        self.assertEqual(result.truncated_chunks, 1)
        self.assertEqual(result.dropped_chunks, 1)
        self.assertIn(
            b"[container stream truncated]",
            result.chunks[0].content,
        )
        self.assertIn(ContainerBackendDiagnosticCode.STREAM_TRUNCATED, codes)
        self.assertIn(ContainerBackendDiagnosticCode.EVENT_DROPPED, codes)
        self.assertEqual(result.to_dict()["dropped_chunks"], 1)
        filled = drain_container_streams(
            (
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.STDOUT,
                    content=b"12345",
                    sequence=0,
                ),
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.STDOUT,
                    content=b"6",
                    sequence=1,
                ),
            ),
            ContainerStreamDrainPolicy(
                max_chunks=3,
                max_bytes=5,
                max_chunk_bytes=5,
            ),
        )
        marker = drain_container_streams(
            (
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.STDOUT,
                    content=b"abcdef",
                    sequence=0,
                ),
            ),
            ContainerStreamDrainPolicy(
                max_chunks=1,
                max_bytes=4,
                max_chunk_bytes=4,
            ),
        )

        self.assertEqual(filled.dropped_chunks, 1)
        self.assertEqual(marker.chunks[0].content, b"[con")
        with self.assertRaises(AssertionError):
            ContainerStreamDrainPolicy(max_chunks=0)

    def test_incremental_stream_drains_with_idle_timeout_and_bounds(
        self,
    ) -> None:
        idle_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_incremental=True,
                stream_delay_seconds=0.05,
            )
        )
        bounded_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_incremental=True,
                stream_chunks=(
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"abcdef",
                        sequence=0,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDERR,
                        content=b"stderr",
                        sequence=1,
                    ),
                    ContainerBackendStreamChunk(
                        stream=ContainerBackendStream.STDOUT,
                        content=b"dropped",
                        sequence=2,
                    ),
                ),
            )
        )

        idle_result = run_async(
            run_container_managed_lifecycle(
                idle_backend,
                _run_plan(),
                deadlines=ContainerLifecycleDeadlines(
                    execution_seconds=1,
                    idle_seconds=0.001,
                ),
            )
        )
        bounded_result = run_async(
            run_container_managed_lifecycle(
                bounded_backend,
                _run_plan(),
                stream_policy=ContainerStreamDrainPolicy(
                    max_chunks=2,
                    max_bytes=8,
                    max_chunk_bytes=4,
                ),
            )
        )

        self.assertEqual(
            idle_result.timed_out_phase,
            ContainerLifecyclePhase.STREAM,
        )
        self.assertTrue(idle_result.cleanup_completed)
        self.assertEqual(bounded_result.stream.truncated_chunks, 2)
        self.assertEqual(bounded_result.stream.dropped_chunks, 1)
        self.assertEqual(
            [chunk.content for chunk in bounded_result.stream.chunks],
            [b"[con", b"[con"],
        )

    def test_timeouts_are_phase_specific_and_cleanup_runs(self) -> None:
        cases = (
            (
                ContainerBackendOperation.IMAGE_PULL,
                ContainerLifecyclePhase.IMAGE_PULL,
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_BUILD,
                ContainerLifecyclePhase.IMAGE_BUILD,
                _run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY),
                None,
            ),
            (
                ContainerBackendOperation.CREATE,
                ContainerLifecyclePhase.CREATE,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.ATTACH,
                ContainerLifecyclePhase.ATTACH,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.START,
                ContainerLifecyclePhase.START,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.WAIT,
                ContainerLifecyclePhase.WAIT,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerLifecyclePhase.COPY_OUTPUTS,
                _run_plan(),
                _output_contract(),
            ),
        )

        for operation, phase, plan, contract in cases:
            with self.subTest(operation=operation.value):
                backend = ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(pull=True, build=True),
                        timeout_operations=(operation,),
                    )
                )

                result = run_async(
                    run_container_managed_lifecycle(
                        backend,
                        plan,
                        output_contract=contract,
                    )
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.FAILED,
                )
                self.assertEqual(result.timed_out_phase, phase)
                if operation not in {
                    ContainerBackendOperation.IMAGE_PULL,
                    ContainerBackendOperation.IMAGE_BUILD,
                    ContainerBackendOperation.CREATE,
                }:
                    self.assertTrue(result.cleanup_completed)

    def test_cancellations_are_phase_specific_and_cleanup_runs(self) -> None:
        cases = (
            (
                ContainerBackendOperation.IMAGE_PULL,
                ContainerLifecyclePhase.IMAGE_PULL,
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
            ),
            (
                ContainerBackendOperation.IMAGE_BUILD,
                ContainerLifecyclePhase.IMAGE_BUILD,
                _run_plan(build_policy=ContainerBuildPolicy.TRUSTED_ONLY),
                None,
            ),
            (
                ContainerBackendOperation.CREATE,
                ContainerLifecyclePhase.CREATE,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.ATTACH,
                ContainerLifecyclePhase.ATTACH,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.START,
                ContainerLifecyclePhase.START,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.WAIT,
                ContainerLifecyclePhase.WAIT,
                _run_plan(),
                None,
            ),
            (
                ContainerBackendOperation.COPY_OUTPUTS,
                ContainerLifecyclePhase.COPY_OUTPUTS,
                _run_plan(),
                _output_contract(),
            ),
        )

        for operation, phase, plan, contract in cases:
            with self.subTest(operation=operation.value):
                backend = ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(pull=True, build=True),
                        cancel_operations=(operation,),
                    )
                )

                result = run_async(
                    run_container_managed_lifecycle(
                        backend,
                        plan,
                        output_contract=contract,
                    )
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.CANCELLED,
                )
                self.assertEqual(result.cancelled_phase, phase)
                if operation not in {
                    ContainerBackendOperation.IMAGE_PULL,
                    ContainerBackendOperation.IMAGE_BUILD,
                    ContainerBackendOperation.CREATE,
                }:
                    self.assertTrue(result.cleanup_completed)

    def test_deadline_timeout_shutdown_and_event_drop_paths(self) -> None:
        timeout_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                stream_delay_seconds=0.05,
            )
        )
        shutdown_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        dropped_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )

        timeout_result = run_async(
            run_container_managed_lifecycle(
                timeout_backend,
                _run_plan(),
                deadlines=ContainerLifecycleDeadlines(execution_seconds=0.001),
            )
        )
        shutdown_result = run_async(
            run_container_managed_lifecycle(
                shutdown_backend,
                _run_plan(),
                shutdown_requested=True,
            )
        )
        dropped_result = run_async(
            run_container_managed_lifecycle(
                dropped_backend,
                _run_plan(),
                event_policy=ContainerLifecycleEventPolicy(max_events=2),
            )
        )

        self.assertEqual(
            timeout_result.timed_out_phase,
            ContainerLifecyclePhase.STREAM,
        )
        self.assertEqual(
            shutdown_result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertIn(
            ContainerBackendOperation.STOP,
            shutdown_backend.operations,
        )
        self.assertIn(
            ContainerBackendOperation.KILL,
            shutdown_backend.operations,
        )
        self.assertGreater(dropped_result.dropped_events, 0)
        self.assertEqual(
            dropped_result.execution.metadata["dropped_events"],
            str(dropped_result.dropped_events),
        )
        self.assertEqual(
            timeout_result.to_dict()["timed_out_phase"],
            ContainerLifecyclePhase.STREAM.value,
        )
        self.assertEqual(
            shutdown_result.to_dict()["cancelled_phase"],
            ContainerLifecyclePhase.START.value,
        )

    def test_parent_deadline_is_aggregate_lifecycle_budget(self) -> None:
        backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                operation_delay_seconds={
                    ContainerBackendOperation.IMAGE_RESOLUTION: 0.01,
                    ContainerBackendOperation.CREATE: 0.01,
                    ContainerBackendOperation.ATTACH: 0.01,
                },
            )
        )

        result = run_async(
            run_container_managed_lifecycle(
                backend,
                _run_plan(),
                deadlines=ContainerLifecycleDeadlines(
                    create_seconds=1,
                    start_seconds=1,
                    execution_seconds=1,
                    parent_seconds=0.025,
                ),
            )
        )

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertIn(
            result.timed_out_phase,
            {
                ContainerLifecyclePhase.CREATE,
                ContainerLifecyclePhase.ATTACH,
            },
        )

    def test_backend_reported_failures_are_terminal(self) -> None:
        contract = _output_contract()
        cases = (
            (
                "image_denied",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    soft_operation_diagnostics={
                        ContainerBackendOperation.IMAGE_RESOLUTION: (
                            ContainerBackendDiagnosticCode.IMAGE_DENIED
                        ),
                    },
                ),
                _run_plan(),
                None,
                ContainerResultStatus.DENIED,
                None,
            ),
            (
                "pull_failed",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    soft_operation_diagnostics={
                        ContainerBackendOperation.IMAGE_PULL: (
                            ContainerBackendDiagnosticCode.PULL_FAILED
                        ),
                    },
                ),
                _run_plan(pull_policy=ContainerPullPolicy.IF_MISSING),
                None,
                ContainerResultStatus.FAILED,
                None,
            ),
            (
                "start_failed",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    soft_operation_diagnostics={
                        ContainerBackendOperation.START: (
                            ContainerBackendDiagnosticCode.START_FAILED
                        ),
                    },
                ),
                _run_plan(),
                None,
                ContainerResultStatus.FAILED,
                None,
            ),
            (
                "wait_diagnostic",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    soft_operation_diagnostics={
                        ContainerBackendOperation.WAIT: (
                            ContainerBackendDiagnosticCode.WAIT_FAILED
                        ),
                    },
                ),
                _run_plan(),
                None,
                ContainerResultStatus.FAILED,
                None,
            ),
            (
                "wait_timed_out",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_timed_out=True,
                ),
                _run_plan(),
                None,
                ContainerResultStatus.FAILED,
                ContainerLifecyclePhase.WAIT,
            ),
            (
                "nonzero_exit",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    wait_exit_code=7,
                ),
                _run_plan(),
                None,
                ContainerResultStatus.FAILED,
                None,
            ),
            (
                "copy_reject",
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    output_result=ContainerOutputValidationResult(
                        decision=ContainerOutputDecisionType.REJECT,
                        contract=contract,
                    ),
                ),
                _run_plan(),
                contract,
                ContainerResultStatus.FAILED,
                None,
            ),
        )

        for name, script, plan, output_contract, status, timed_out in cases:
            with self.subTest(name=name):
                result = run_async(
                    run_container_managed_lifecycle(
                        ContainerFakeBackend(script),
                        plan,
                        output_contract=output_contract,
                    )
                )

                self.assertEqual(result.execution.status, status)
                self.assertEqual(result.timed_out_phase, timed_out)

    def test_hard_backend_errors_and_stream_interruptions_are_normalized(
        self,
    ) -> None:
        hard_error = run_async(
            run_container_managed_lifecycle(
                ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        operation_diagnostics={
                            ContainerBackendOperation.CREATE: (
                                ContainerBackendDiagnosticCode.CREATE_FAILED
                            ),
                        },
                    )
                ),
                _run_plan(),
            )
        )
        timeout = run_async(
            run_container_managed_lifecycle(
                _ExplodingStreamBackend(TimeoutError()),
                _run_plan(),
            )
        )
        cancelled = run_async(
            run_container_managed_lifecycle(
                _ExplodingStreamBackend(CancelledError()),
                _run_plan(),
            )
        )

        self.assertEqual(
            hard_error.execution.status,
            ContainerResultStatus.FAILED,
        )
        self.assertEqual(
            timeout.timed_out_phase,
            ContainerLifecyclePhase.STREAM,
        )
        self.assertEqual(
            cancelled.cancelled_phase,
            ContainerLifecyclePhase.STREAM,
        )

    def test_cleanup_timeout_cancel_quarantine_and_idempotence(self) -> None:
        cases = (
            (
                ContainerBackendOperation.REMOVE,
                "timeout_operations",
                ContainerBackendDiagnosticCode.TIMEOUT,
            ),
            (
                ContainerBackendOperation.CLEANUP,
                "timeout_operations",
                ContainerBackendDiagnosticCode.TIMEOUT,
            ),
            (
                ContainerBackendOperation.REMOVE,
                "cancel_operations",
                ContainerBackendDiagnosticCode.CANCELLED,
            ),
            (
                ContainerBackendOperation.CLEANUP,
                "cancel_operations",
                ContainerBackendDiagnosticCode.CANCELLED,
            ),
        )

        for operation, script_field, code in cases:
            with self.subTest(operation=operation.value, field=script_field):
                if script_field == "timeout_operations":
                    script = ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        timeout_operations=(operation,),
                    )
                else:
                    script = ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        cancel_operations=(operation,),
                    )
                backend = ContainerFakeBackend(
                    script,
                )

                result = run_async(
                    run_container_managed_lifecycle(backend, _run_plan())
                )

                self.assertEqual(
                    result.execution.status,
                    ContainerResultStatus.FAILED,
                )
                self.assertTrue(result.cleanup_uncertain)
                self.assertIn(
                    code,
                    {diagnostic.code for diagnostic in result.diagnostics},
                )

        quarantine = run_async(
            run_container_managed_lifecycle(
                ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        cleanup_uncertain=True,
                    )
                ),
                _run_plan(),
            )
        )
        self.assertTrue(quarantine.orphan_quarantined)

        retry_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(
                capabilities=_capabilities(),
                cleanup_uncertain=True,
            )
        )
        retry_cleanup = ContainerLifecycleCleanup()
        retry_container = run_async(retry_backend.create(_run_plan()))
        first_retry = run_async(
            retry_cleanup.cleanup(retry_backend, retry_container)
        )
        second_retry = run_async(
            retry_cleanup.cleanup(retry_backend, retry_container)
        )
        success_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        success_retry = run_async(
            retry_cleanup.cleanup(success_backend, retry_container)
        )
        already_cleaned = run_async(
            retry_cleanup.cleanup(success_backend, retry_container)
        )

        forced = run_async(
            run_container_managed_lifecycle(
                ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        wait_timed_out=True,
                        soft_operation_diagnostics={
                            ContainerBackendOperation.STOP: (
                                ContainerBackendDiagnosticCode.CLEANUP_FAILED
                            ),
                            ContainerBackendOperation.KILL: (
                                ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
                            ),
                        },
                    )
                ),
                _run_plan(),
            )
        )
        remove_error = run_async(
            run_container_managed_lifecycle(
                ContainerFakeBackend(
                    ContainerFakeBackendScript(
                        capabilities=_capabilities(),
                        operation_diagnostics={
                            ContainerBackendOperation.REMOVE: (
                                ContainerBackendDiagnosticCode.CLEANUP_FAILED
                            ),
                        },
                    )
                ),
                _run_plan(),
            )
        )
        cleanup_backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        cleanup = ContainerLifecycleCleanup()
        container = run_async(cleanup_backend.create(_run_plan()))
        first = run_async(cleanup.cleanup(cleanup_backend, container))
        second = run_async(cleanup.cleanup(cleanup_backend, container))

        self.assertTrue(forced.cleanup_uncertain)
        self.assertTrue(forced.orphan_quarantined)
        self.assertTrue(remove_error.cleanup_uncertain)
        self.assertFalse(first_retry.already_cleaned)
        self.assertFalse(second_retry.already_cleaned)
        self.assertFalse(success_retry.already_cleaned)
        self.assertTrue(already_cleaned.already_cleaned)
        self.assertEqual(
            retry_backend.operations.count(ContainerBackendOperation.REMOVE),
            2,
        )
        self.assertFalse(first.already_cleaned)
        self.assertTrue(second.already_cleaned)
        self.assertEqual(first.to_dict()["cleanup_uncertain"], False)
        self.assertEqual(
            cleanup_backend.operations.count(ContainerBackendOperation.REMOVE),
            1,
        )

    def test_fake_e2e_caller_cancellation_returns_after_cleanup(self) -> None:
        async def scenario() -> ContainerManagedLifecycleResult:
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    stream_delay_seconds=0.05,
                )
            )
            task = create_task(
                run_container_managed_lifecycle(backend, _run_plan())
            )
            await sleep(0.001)
            task.cancel()
            result = await task
            self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
            return result

        result = run_async(scenario())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertTrue(result.cleanup_completed)

    def test_fake_e2e_cancellation_during_cleanup_is_reported(self) -> None:
        async def scenario() -> ContainerManagedLifecycleResult:
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        ContainerBackendOperation.REMOVE: 0.05,
                    },
                )
            )
            task = create_task(
                run_container_managed_lifecycle(
                    backend,
                    _run_plan(),
                    deadlines=ContainerLifecycleDeadlines(cleanup_seconds=1),
                )
            )
            for _ in range(100):
                if ContainerBackendOperation.REMOVE in backend.operations:
                    break
                await sleep(0.001)
            self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
            task.cancel()
            result = await task
            self.assertIn(
                ContainerBackendOperation.CLEANUP,
                backend.operations,
            )
            return result

        result = run_async(scenario())

        self.assertEqual(
            result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertEqual(
            result.cancelled_phase,
            ContainerLifecyclePhase.CLEANUP,
        )
        self.assertTrue(result.cleanup_completed)

    def test_cleanup_cancellation_reports_bounded_uncertainty(self) -> None:
        async def timeout_scenario() -> ContainerManagedLifecycleResult:
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        ContainerBackendOperation.REMOVE: 0.05,
                    },
                )
            )
            task = create_task(
                run_container_managed_lifecycle(
                    backend,
                    _run_plan(),
                    deadlines=ContainerLifecycleDeadlines(
                        cleanup_seconds=0.001,
                    ),
                )
            )
            for _ in range(100):
                if ContainerBackendOperation.REMOVE in backend.operations:
                    break
                await sleep(0.001)
            task.cancel()
            return await task

        async def multistep_timeout_scenario() -> (
            ContainerManagedLifecycleResult
        ):
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        ContainerBackendOperation.REMOVE: 0.01,
                        ContainerBackendOperation.CLEANUP: 0.05,
                    },
                )
            )
            task = create_task(
                run_container_managed_lifecycle(
                    backend,
                    _run_plan(),
                    deadlines=ContainerLifecycleDeadlines(
                        cleanup_seconds=0.02,
                    ),
                )
            )
            for _ in range(100):
                if ContainerBackendOperation.REMOVE in backend.operations:
                    break
                await sleep(0.001)
            self.assertIn(ContainerBackendOperation.REMOVE, backend.operations)
            task.cancel()
            return await task

        async def repeated_cancel_scenario() -> (
            ContainerManagedLifecycleResult
        ):
            backend = ContainerFakeBackend(
                ContainerFakeBackendScript(
                    capabilities=_capabilities(),
                    operation_delay_seconds={
                        ContainerBackendOperation.REMOVE: 0.05,
                    },
                )
            )
            task = create_task(
                run_container_managed_lifecycle(
                    backend,
                    _run_plan(),
                    deadlines=ContainerLifecycleDeadlines(cleanup_seconds=1),
                )
            )
            for _ in range(100):
                if ContainerBackendOperation.REMOVE in backend.operations:
                    break
                await sleep(0.001)
            task.cancel()
            await sleep(0.001)
            task.cancel()
            return await task

        timeout_result = run_async(timeout_scenario())
        multistep_timeout_result = run_async(multistep_timeout_scenario())
        repeated_cancel_result = run_async(repeated_cancel_scenario())

        self.assertTrue(timeout_result.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.TIMEOUT,
            {diagnostic.code for diagnostic in timeout_result.diagnostics},
        )
        self.assertEqual(
            multistep_timeout_result.execution.status,
            ContainerResultStatus.CANCELLED,
        )
        self.assertEqual(
            multistep_timeout_result.cancelled_phase,
            ContainerLifecyclePhase.CLEANUP,
        )
        self.assertTrue(multistep_timeout_result.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.TIMEOUT,
            {
                diagnostic.code
                for diagnostic in multistep_timeout_result.diagnostics
            },
        )
        self.assertTrue(repeated_cancel_result.cleanup_uncertain)
        self.assertIn(
            ContainerBackendDiagnosticCode.CANCELLED,
            {
                diagnostic.code
                for diagnostic in repeated_cancel_result.diagnostics
            },
        )

    def test_lifecycle_value_objects_validate_and_serialize(self) -> None:
        event = ContainerLifecycleEvent(
            phase="result",
            status="completed",
            sequence=0,
            metadata={"status": "completed"},
        )
        deadlines = ContainerLifecycleDeadlines(parent_seconds=1)
        managed = ContainerManagedLifecycleResult(
            execution=ContainerExecutionResult(
                status=ContainerResultStatus.CANCELLED,
            ),
            timed_out_phase="wait",
            cancelled_phase="stream",
        )

        self.assertEqual(event.to_dict()["phase"], "result")
        self.assertEqual(deadlines.to_dict()["parent_seconds"], 1)
        self.assertEqual(
            ContainerLifecycleDeadlines(pull_seconds=2).effective_seconds(
                ContainerLifecyclePhase.IMAGE_PULL,
            ),
            2,
        )
        self.assertEqual(
            ContainerStreamDrainPolicy(max_chunks=1).to_dict()["max_chunks"],
            1,
        )
        self.assertEqual(
            ContainerLifecycleEventPolicy(max_events=1).to_dict()[
                "max_events"
            ],
            1,
        )
        self.assertEqual(managed.to_dict()["timed_out_phase"], "wait")
        self.assertEqual(managed.to_dict()["cancelled_phase"], "stream")
        self.assertEqual(
            deadlines.effective_seconds(ContainerLifecyclePhase.CLEANUP),
            1,
        )
        with self.assertRaises(AssertionError):
            ContainerLifecycleEvent(
                phase="bad",
                status="completed",
                sequence=0,
            )
        with self.assertRaises(AssertionError):
            ContainerLifecycleEvent(
                phase=ContainerLifecyclePhase.RESULT,
                status=ContainerLifecycleEventStatus.COMPLETED,
                sequence=-1,
            )
        with self.assertRaises(AssertionError):
            ContainerLifecycleDeadlines(pull_seconds=0)
        with self.assertRaises(AssertionError):
            ContainerLifecycleDeadlines(pull_seconds=cast(float, "bad"))
        with self.assertRaises(AssertionError):
            ContainerLifecycleEventPolicy(max_events=0)
        with self.assertRaises(AssertionError):
            ContainerManagedLifecycleResult(
                execution=cast(ContainerExecutionResult, object()),
            )


def _capabilities(
    *,
    pull: bool = True,
    build: bool = True,
) -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=ContainerBackend.DOCKER,
        host_os="linux",
        guest_os="linux",
        architecture="amd64",
        rootless=True,
        build=build,
        pull=pull,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=(ContainerMountType.WORKSPACE,),
        device_classes=(),
        resource_limits=True,
        streaming_attach=True,
        stats=True,
    )


def _run_plan(
    *,
    pull_policy: ContainerPullPolicy = ContainerPullPolicy.NEVER,
    build_policy: ContainerBuildPolicy = ContainerBuildPolicy.DISABLED,
) -> ContainerRunPlan:
    return ContainerRunPlan(
        backend=ContainerBackend.DOCKER,
        profile_name="phase8-profile",
        image=ContainerImagePolicy(
            reference=_IMAGE,
            pull_policy=pull_policy,
            build_policy=build_policy,
        ),
        command=ContainerCommandPlan(
            tool_name="shell.echo",
            command="echo",
            argv=("echo", "ok"),
            cwd="/workspace",
            scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        ),
        policy_version="phase8",
    )


def _output_contract() -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=64,
    )


class _ExplodingChunks:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __iter__(self) -> Iterator[ContainerBackendStreamChunk]:
        raise self._error


class _ExplodingStreamBackend(ContainerAsyncBackend):
    def __init__(self, error: BaseException) -> None:
        self._backend = ContainerFakeBackend(
            ContainerFakeBackendScript(capabilities=_capabilities())
        )
        self._error = error

    async def probe(self) -> ContainerBackendProbeResult:
        return await self._backend.probe()

    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        return await self._backend.resolve_image(plan)

    async def pull_image(
        self,
        plan: ContainerRunPlan,
        image: ContainerBackendImageResolution,
    ) -> ContainerBackendOperationResult:
        return await self._backend.pull_image(plan, image)

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        return await self._backend.build_image(plan)

    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        return await self._backend.create(plan)

    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.start(container)

    async def attach(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.attach(container)

    async def stream(
        self,
        container: ContainerBackendContainer,
    ) -> (
        Sequence[ContainerBackendStreamChunk]
        | AsyncIterable[ContainerBackendStreamChunk]
    ):
        await self._backend.stream(container)
        return cast(
            tuple[ContainerBackendStreamChunk, ...],
            _ExplodingChunks(self._error),
        )

    async def wait(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendWaitResult:
        return await self._backend.wait(container)

    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        return await self._backend.inspect(container)

    async def stats(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStats, ...]:
        return await self._backend.stats(container)

    async def stop(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.stop(container)

    async def kill(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.kill(container)

    async def remove(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.remove(container)

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        return await self._backend.copy_outputs(container, contract)

    async def cleanup(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._backend.cleanup(container)


if __name__ == "__main__":
    main()
