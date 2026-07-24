from asyncio import CancelledError, Event, create_task, sleep
from datetime import UTC, datetime
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, patch

sys_path.append(str(Path(__file__).parent))

from worker_test import (  # type: ignore[import-not-found]
    DurableResumableTarget,
    FakeDurableResumeCoordinator,
    FakeDurableResumeHandle,
    FakeDurableSuspensionCoordinator,
    FakeQueue,
    FakeTarget,
    _definition,
)

from avalan.interaction import (
    ContinuationId,
    InputRequestId,
    InputRequiredResult,
)
from avalan.task import (
    TaskDefinition,
    TaskExecutionRequest,
    TaskQueueClaim,
    TaskQueueConflictError,
    TaskQueueItem,
    TaskQueueItemState,
    TaskRunState,
    TaskStoreConflictError,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskValidationError,
    TaskWorker,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)
from avalan.task import worker as worker_module
from avalan.task.runner import TaskContainerAttemptResult
from avalan.task.stores import InMemoryTaskStore
from avalan.task.worker import (
    TaskWorkerProcessResult,
    _TaskDurableResumeOwner,
    _TaskResumeClaimLeaseBridge,
)


class TaskWorkerContinuationCoverageTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await self._reset()

    async def _reset(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.store = InMemoryTaskStore(clock=lambda: self.now)
        self.definition = _definition()
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                queue="default",
            )
        )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        self.run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        self.queue = FakeQueue(self.store, self.now)
        self.queue.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=self.run.run_id,
            queue_name="default",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=self.now,
            attempts=0,
            created_at=self.now,
            updated_at=self.now,
            run_state=self.run.state,
        )

    def _worker(self, *, target: FakeTarget | None = None) -> TaskWorker:
        return TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=target or FakeTarget(),
            worker_id="worker-1",
            clock=lambda: self.now,
        )

    async def _claim(self) -> TaskQueueClaim:
        claim = await self.queue.claim(
            "default",
            worker_id="worker-1",
            lease_expires_at=self.now.replace(hour=1),
            now=self.now,
            metadata={"worker_id": "worker-1"},
        )
        assert claim is not None
        return claim

    async def _durable_worker(
        self,
    ) -> tuple[TaskWorker, FakeDurableResumeHandle]:
        target = DurableResumableTarget(self.now)
        coordinator = FakeDurableSuspensionCoordinator(self.queue)
        suspending_worker = TaskWorker(
            self.store,
            cast(Any, self.queue),
            target=target,
            worker_id="worker-1",
            durable_suspension_coordinator=coordinator,
            clock=lambda: self.now,
        )
        suspended = await suspending_worker.process_once()
        assert suspended.suspension is not None
        await self.queue.requeue_suspended(
            self.run.run_id,
            request_id=suspended.suspension.request_id,
            continuation_id=suspended.suspension.continuation_id,
            resolution_revision=2,
            now=self.now,
        )
        admission = FakeDurableResumeHandle()
        return (
            TaskWorker(
                self.store,
                cast(Any, self.queue),
                target=target,
                worker_id="worker-1",
                durable_suspension_coordinator=coordinator,
                durable_resume_coordinator=FakeDurableResumeCoordinator(
                    admission
                ),
                clock=lambda: self.now,
            ),
            admission,
        )

    @staticmethod
    def _suspended_outcome() -> TaskTargetOutcome:
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=InputRequestId("request-next"),
                continuation_id=ContinuationId("continuation-next"),
                detached_resumption_available=True,
            )
        )

    @staticmethod
    def _target_context(
        definition: TaskDefinition,
        claim: TaskQueueClaim,
        admission: FakeDurableResumeHandle,
    ) -> TaskTargetContext:
        async def check_cancelled() -> None:
            return None

        return TaskTargetContext(
            definition=definition,
            execution=claim.attempt.context,
            input_value={},
            cancellation_checker=check_cancelled,
            durable_resume=admission,
        )

    async def test_shutdown_after_claim_converges_setup_failure(self) -> None:
        target = FakeTarget()
        worker = self._worker(target=target)

        with patch.object(
            worker,
            "_shutdown_requested",
            side_effect=(False, True),
        ):
            result = await worker.process_once()

        self.assertIsNotNone(result.abandonment)
        self.assertEqual(target.contexts, [])

    async def test_non_durable_start_failure_propagates(self) -> None:
        worker = self._worker()
        error = RuntimeError("start failed before durable admission")

        with (
            patch.object(
                worker,
                "_start_claimed_attempt",
                side_effect=error,
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await worker.process_once()

        self.assertIs(raised.exception, error)

    async def test_durable_execute_convergence_conflict_matrix(self) -> None:
        cases = (
            (
                "shutdown",
                worker_module._TaskWorkerShutdownRequested(),
                True,
            ),
            (
                "queue conflict",
                TaskQueueConflictError("execution lost claim"),
                False,
            ),
            (
                "target failure",
                RuntimeError("durable target failed"),
                False,
            ),
        )
        for label, execution_error, shutdown_requested in cases:
            with self.subTest(label=label):
                await self._reset()
                worker, _admission = await self._durable_worker()
                with (
                    patch.object(
                        worker,
                        "_execute",
                        side_effect=execution_error,
                    ),
                    patch.object(
                        worker,
                        "_converge_durable_failure",
                        AsyncMock(
                            side_effect=TaskStoreConflictError(
                                "convergence lost claim"
                            )
                        ),
                    ),
                ):
                    result = await worker.process_once()

                self.assertTrue(result.lease_lost)
                self.assertEqual(
                    result.shutdown_requested,
                    shutdown_requested,
                )

    async def test_durable_execute_expiry_converges(self) -> None:
        worker, _admission = await self._durable_worker()
        expected = TaskWorkerProcessResult(output="expired convergence")
        expired = worker_module._TaskDurableResumeExpired("resume expired")
        converge = AsyncMock(return_value=expected)

        with (
            patch.object(worker, "_execute", side_effect=expired),
            patch.object(
                worker,
                "_converge_expired_resume",
                converge,
            ),
        ):
            result = await worker.process_once()

        self.assertIs(result, expected)
        converge.assert_awaited_once()

    async def test_durable_suspend_conflict_convergence_matrix(self) -> None:
        for convergence_conflicts in (False, True):
            with self.subTest(convergence_conflicts=convergence_conflicts):
                await self._reset()
                worker, _admission = await self._durable_worker()
                expected = TaskWorkerProcessResult(
                    output="suspension convergence"
                )
                converge = AsyncMock(
                    side_effect=(
                        TaskStoreConflictError("convergence lost claim")
                        if convergence_conflicts
                        else None
                    ),
                    return_value=expected,
                )
                with (
                    patch.object(
                        worker,
                        "_execute",
                        return_value=self._suspended_outcome(),
                    ),
                    patch.object(
                        worker,
                        "_suspend",
                        side_effect=TaskQueueConflictError(
                            "suspension lost claim"
                        ),
                    ),
                    patch.object(
                        worker,
                        "_converge_durable_failure",
                        converge,
                    ),
                ):
                    result = await worker.process_once()

                if convergence_conflicts:
                    self.assertTrue(result.lease_lost)
                else:
                    self.assertIs(result, expected)
                converge.assert_awaited_once()

    async def test_durable_suspend_expiry_converges(self) -> None:
        worker, _admission = await self._durable_worker()
        expected = TaskWorkerProcessResult(output="suspension expired")
        expired = worker_module._TaskDurableResumeExpired("resume expired")
        converge = AsyncMock(return_value=expected)

        with (
            patch.object(
                worker,
                "_execute",
                return_value=self._suspended_outcome(),
            ),
            patch.object(worker, "_suspend", side_effect=expired),
            patch.object(
                worker,
                "_converge_expired_resume",
                converge,
            ),
        ):
            result = await worker.process_once()

        self.assertIs(result, expected)
        converge.assert_awaited_once()

    async def test_durable_suspend_failure_convergence_matrix(self) -> None:
        for convergence_conflicts in (False, True):
            with self.subTest(convergence_conflicts=convergence_conflicts):
                await self._reset()
                worker, _admission = await self._durable_worker()
                expected = TaskWorkerProcessResult(
                    output="suspension failure convergence"
                )
                converge = AsyncMock(
                    side_effect=(
                        TaskStoreConflictError("convergence lost claim")
                        if convergence_conflicts
                        else None
                    ),
                    return_value=expected,
                )
                with (
                    patch.object(
                        worker,
                        "_execute",
                        return_value=self._suspended_outcome(),
                    ),
                    patch.object(
                        worker,
                        "_suspend",
                        side_effect=RuntimeError("suspension failed"),
                    ),
                    patch.object(
                        worker,
                        "_converge_durable_failure",
                        converge,
                    ),
                ):
                    result = await worker.process_once()

                if convergence_conflicts:
                    self.assertTrue(result.lease_lost)
                else:
                    self.assertIs(result, expected)
                converge.assert_awaited_once()

    async def test_non_durable_suspend_failure_retries(self) -> None:
        worker = self._worker()

        with (
            patch.object(
                worker,
                "_execute",
                return_value=self._suspended_outcome(),
            ),
            patch.object(
                worker,
                "_suspend",
                side_effect=RuntimeError("suspension failed"),
            ),
        ):
            result = await worker.process_once()

        self.assertIsNotNone(result.retry)
        self.assertFalse(result.lease_lost)

    async def test_non_durable_suspend_failure_finalize_conflict(self) -> None:
        worker = self._worker()

        with (
            patch.object(
                worker,
                "_execute",
                return_value=self._suspended_outcome(),
            ),
            patch.object(
                worker,
                "_suspend",
                side_effect=RuntimeError("suspension failed"),
            ),
            patch.object(
                worker,
                "_finalize_failure",
                side_effect=TaskQueueConflictError(
                    "failure finalization lost claim"
                ),
            ),
        ):
            result = await worker.process_once()

        self.assertTrue(result.lease_lost)
        self.assertIsNone(result.retry)

    async def test_durable_completion_conflict_convergence_matrix(
        self,
    ) -> None:
        for convergence_conflicts in (False, True):
            with self.subTest(convergence_conflicts=convergence_conflicts):
                await self._reset()
                worker, _admission = await self._durable_worker()
                expected = TaskWorkerProcessResult(
                    output="completion convergence"
                )
                converge = AsyncMock(
                    side_effect=(
                        TaskStoreConflictError("convergence lost claim")
                        if convergence_conflicts
                        else None
                    ),
                    return_value=expected,
                )
                with (
                    patch.object(
                        worker,
                        "_execute",
                        return_value=completed_task_target_outcome(
                            "safe output"
                        ),
                    ),
                    patch.object(
                        worker,
                        "_settle_durable_success",
                        side_effect=TaskQueueConflictError(
                            "completion lost claim"
                        ),
                    ),
                    patch.object(
                        worker,
                        "_converge_durable_failure",
                        converge,
                    ),
                ):
                    result = await worker.process_once()

                if convergence_conflicts:
                    self.assertTrue(result.lease_lost)
                else:
                    self.assertIs(result, expected)
                converge.assert_awaited_once()

    async def test_container_output_validation_failure_raises(self) -> None:
        worker = self._worker()
        claim = await self._claim()
        run, attempt, segment = await worker._start_claimed_attempt(
            claim,
            previous_segment=None,
        )
        sanitizer = worker._sanitizer(self.definition)

        with (
            patch.object(worker, "_input_files", AsyncMock(return_value=())),
            patch.object(
                worker_module,
                "task_container_input_mount_manifest",
                return_value=(),
            ),
            patch.object(
                worker,
                "_run_task_container",
                AsyncMock(
                    return_value=TaskContainerAttemptResult(
                        output={"invalid": "text"}
                    )
                ),
            ),
            patch.object(worker, "_check_cancelled", AsyncMock()),
            self.assertRaises(TaskValidationError),
        ):
            await worker._execute(
                self.definition,
                claim=claim,
                run=run,
                attempt=attempt,
                segment=segment,
                sanitizer=sanitizer,
                durable_resume=None,
                durable_target=None,
                heartbeat_task=None,
            )

    async def test_run_target_preserves_heartbeat_and_interrupt_errors(
        self,
    ) -> None:
        worker = self._worker()
        claim = await self._claim()
        admission = FakeDurableResumeHandle()
        context = self._target_context(
            self.definition,
            claim,
            admission,
        )
        target_blocked = Event()

        async def blocked_target(
            _context: TaskTargetContext,
            *,
            durable_target: object,
        ) -> TaskTargetOutcome:
            del durable_target
            await target_blocked.wait()
            return completed_task_target_outcome("unreachable")

        heartbeat_error = RuntimeError("heartbeat failed")

        async def fail_heartbeat() -> None:
            raise heartbeat_error

        heartbeat_task = create_task(fail_heartbeat())
        await sleep(0)
        interruption_error = RuntimeError("interrupt failed")
        interrupt = AsyncMock(side_effect=interruption_error)

        with (
            patch.object(
                worker,
                "_target_execution",
                new=blocked_target,
            ),
            patch.object(
                worker,
                "_interrupt_durable_target",
                interrupt,
            ),
            self.assertRaises(BaseExceptionGroup) as raised,
        ):
            await worker._run_target(
                context,
                claim=claim,
                timeout=None,
                heartbeat_task=heartbeat_task,
                durable_target=cast(Any, object()),
            )

        self.assertEqual(
            raised.exception.exceptions,
            (heartbeat_error, interruption_error),
        )
        interrupt.assert_awaited_once()

    async def test_admission_wait_failure_cleanup_matrix(self) -> None:
        for cleanup_error in (
            RuntimeError("admission cleanup failed"),
            None,
        ):
            with self.subTest(has_cleanup_error=cleanup_error is not None):
                await self._reset()
                worker = self._worker()
                claim = await self._claim()
                heartbeat_task = create_task(Event().wait())
                wait_error = RuntimeError("admission wait failed")
                cleanup = AsyncMock(return_value=cleanup_error)
                expected_type = (
                    BaseExceptionGroup
                    if cleanup_error is not None
                    else RuntimeError
                )

                with (
                    patch.object(
                        worker,
                        "_admit_durable_resume",
                        AsyncMock(return_value=None),
                    ),
                    patch.object(
                        worker_module,
                        "wait",
                        AsyncMock(side_effect=wait_error),
                    ),
                    patch.object(
                        worker_module,
                        "_cancel_task_error",
                        cleanup,
                    ),
                    self.assertRaises(expected_type) as raised,
                ):
                    await worker._admit_durable_resume_guarded(
                        claim,
                        None,
                        heartbeat_task=heartbeat_task,
                        claim_lease_manager=_TaskResumeClaimLeaseBridge(claim),
                        resume_owner=_TaskDurableResumeOwner(),
                    )

                if cleanup_error is not None:
                    self.assertEqual(
                        raised.exception.exceptions,
                        (wait_error, cleanup_error),
                    )
                else:
                    self.assertIs(raised.exception, wait_error)
                cleanup.assert_awaited_once()
                await sleep(0)
                heartbeat_task.cancel()
                with self.assertRaises(CancelledError):
                    await heartbeat_task


if __name__ == "__main__":
    main()
