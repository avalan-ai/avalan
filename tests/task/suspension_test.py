from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.interaction import (
    ContinuationId,
    InputRequestId,
    InputRequiredResult,
)
from avalan.task import (
    DirectTaskRunner,
    PrivacyAction,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskDefinition,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskQueueCompletion,
    TaskQueueItem,
    TaskQueueItemState,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetCompleted,
    TaskTargetContext,
    TaskTargetOutcomeKind,
    TaskTargetRunner,
    TaskTargetSuspended,
    TaskValidationContext,
    TaskValidationIssue,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)
from avalan.task.stores import InMemoryTaskStore


class _StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        _ = purpose
        return TaskKeyMaterial(
            key_id=key_id or "task-input",
            algorithm="hmac-sha256",
            secret=b"task-suspension-secret",
        )


class _SuspendingTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetSuspended:
        _ = context
        return _suspended_outcome()


class _AttachedSuspendingTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetSuspended:
        _ = context
        return suspended_task_target_outcome(_input_required())


class _MalformedLifecycleCoordinator:
    def __init__(self, store: InMemoryTaskStore) -> None:
        self.store = store

    async def cancel_input_required_task(
        self,
        *,
        task_run_id: str,
        now: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        _ = metadata
        run = await self.store.get_run(task_run_id)
        assert run.last_attempt_id is not None
        attempt = await self.store.get_attempt(run.last_attempt_id)
        result = TaskExecutionResult(error="malformed")
        failed_run = replace(
            run,
            state=TaskRunState.FAILED,
            result=result,
            updated_at=now,
        )
        failed_attempt = replace(
            attempt,
            state=TaskAttemptState.FAILED,
            result=result,
            updated_at=now,
        )
        queue_item = TaskQueueItem(
            queue_item_id="malformed-queue",
            run_id=task_run_id,
            queue_name="durable",
            state=TaskQueueItemState.DEAD,
            priority=0,
            available_at=now,
            attempts=1,
            created_at=now,
            updated_at=now,
            run_state=failed_run.state,
        )
        return TaskQueueCompletion(
            queue_item=queue_item,
            run=failed_run,
            attempt=failed_attempt,
        )


class TaskTargetOutcomeTest(TestCase):
    def test_completed_and_suspended_are_disjoint(self) -> None:
        completed = completed_task_target_outcome({"status": "done"})
        suspended = _suspended_outcome()

        self.assertIsInstance(completed, TaskTargetCompleted)
        self.assertEqual(completed.kind, TaskTargetOutcomeKind.COMPLETED)
        self.assertIsInstance(suspended, TaskTargetSuspended)
        self.assertEqual(suspended.kind, TaskTargetOutcomeKind.SUSPENDED)
        self.assertFalse(hasattr(suspended, "output"))

    def test_suspended_outcome_rejects_invalid_checkpoint(self) -> None:
        with self.assertRaises(AssertionError):
            TaskTargetSuspended(
                input_required=_input_required(),
                checkpoint_id=" ",
            )


class DirectTaskSuspensionTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore()
        self.runner = DirectTaskRunner(
            self.store,
            target=_SuspendingTarget(),
            hmac_provider=_StaticHmacProvider(),
            definition_hash=lambda _: "task-suspension-definition",
        )

    async def test_suspension_is_not_success_or_artifact(self) -> None:
        result = await self.runner.run(
            _definition(),
            input_value="private prompt",
        )

        self.assertTrue(result.suspended)
        self.assertEqual(result.run.state, TaskRunState.INPUT_REQUIRED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUSPENDED)
        assert result.segment is not None
        self.assertEqual(
            result.segment.state,
            TaskAttemptSegmentState.SUSPENDED,
        )
        self.assertEqual(result.segment.checkpoint_id, "checkpoint-1")
        self.assertIsNone(result.output)
        assert result.run.result is not None
        self.assertIsNone(result.run.result.output_summary)
        self.assertEqual(
            await self.store.list_artifacts(result.run.run_id),
            (),
        )
        events = await self.store.list_events(result.run.run_id)
        self.assertEqual(events[-1].event_type, "task_input_required")
        payload = events[-1].payload
        assert isinstance(payload, Mapping)
        self.assertEqual(
            payload["segment_id"],
            result.segment.segment_id,
        )

    async def test_client_wait_observes_input_required_without_success(
        self,
    ) -> None:
        result = await self.runner.run(_definition(), input_value="private")
        client = TaskClient(
            self.store,
            target=_SuspendingTarget(),
            hmac_provider=_StaticHmacProvider(),
        )

        output = await client.wait(
            result.run.run_id,
            timeout_seconds=0,
        )

        self.assertTrue(output.waiting_for_input)
        self.assertFalse(output.ready)
        self.assertIsNone(output.output_summary)
        assert isinstance(output.input_required, Mapping)
        self.assertEqual(output.input_required["kind"], "input_required")

    async def test_durable_cancel_fails_closed_without_coordinator(
        self,
    ) -> None:
        result = await self.runner.run(_definition(), input_value="private")
        client = TaskClient(
            self.store,
            target=_SuspendingTarget(),
            hmac_provider=_StaticHmacProvider(),
        )

        with self.assertRaises(TaskClientUnsupportedOperationError) as error:
            await client.cancel(result.run.run_id)

        self.assertEqual(
            error.exception.code,
            "task.durable_lifecycle_unavailable",
        )
        run = await self.store.get_run(result.run.run_id)
        self.assertEqual(run.state, TaskRunState.INPUT_REQUIRED)
        attempt = await self.store.get_attempt(result.attempt.attempt_id)
        self.assertEqual(attempt.state, TaskAttemptState.SUSPENDED)

    async def test_attached_cancel_preserves_local_transition_path(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=_AttachedSuspendingTarget(),
            hmac_provider=_StaticHmacProvider(),
            definition_hash=lambda _: "task-attached-suspension",
        )
        result = await runner.run(_definition(), input_value="private")
        client = TaskClient(
            self.store,
            target=_AttachedSuspendingTarget(),
            hmac_provider=_StaticHmacProvider(),
        )

        cancelled = await client.cancel(result.run.run_id)

        self.assertEqual(cancelled.state, TaskRunState.CANCELLED)
        attempt = await self.store.get_attempt(result.attempt.attempt_id)
        self.assertEqual(attempt.state, TaskAttemptState.ABANDONED)

    async def test_durable_cancel_rejects_malformed_completion(
        self,
    ) -> None:
        result = await self.runner.run(_definition(), input_value="private")
        client = TaskClient(
            self.store,
            target=_SuspendingTarget(),
            durable_lifecycle_coordinator=_MalformedLifecycleCoordinator(
                self.store
            ),
            hmac_provider=_StaticHmacProvider(),
            clock=lambda: result.run.updated_at + timedelta(seconds=1),
        )

        with self.assertRaises(TaskClientUnsupportedOperationError) as error:
            await client.cancel(result.run.run_id)

        self.assertEqual(
            error.exception.code,
            "task.durable_lifecycle_invalid",
        )
        unchanged = await self.store.get_run(result.run.run_id)
        self.assertEqual(unchanged.state, TaskRunState.INPUT_REQUIRED)


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="suspend", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        privacy=TaskPrivacyPolicy(
            input=PrivacyAction.HASH,
            output=PrivacyAction.REDACT,
        ),
        run=TaskRunPolicy.direct(),
    )


def _input_required() -> InputRequiredResult:
    return InputRequiredResult(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        detached_resumption_available=True,
    )


def _suspended_outcome() -> TaskTargetSuspended:
    return suspended_task_target_outcome(
        _input_required(),
        checkpoint_id="checkpoint-1",
    )


if __name__ == "__main__":
    main()
