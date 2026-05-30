from datetime import UTC, datetime, timedelta
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreNotFoundError,
)
from avalan.task.stores import InMemoryTaskStore


class FixedClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value = self.value + timedelta(seconds=1)
        return self.value


class InMemoryTaskStoreTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore(clock=FixedClock())
        self.definition = TaskDefinition(
            task=TaskMetadata(name="classify", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/classify.toml"),
        )
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-classify",
        )

    async def test_generated_ids_are_non_empty(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        attempt = await self.store.create_attempt(run.run_id)

        self.assertTrue(run.run_id)
        self.assertTrue(attempt.attempt_id)

    async def test_default_clock_creates_timestamped_runs(self) -> None:
        store = InMemoryTaskStore()
        await store.register_definition(
            self.definition,
            definition_hash="hash-default-clock",
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id="hash-default-clock",
                idempotency_key="request-1",
                queue="default",
            )
        )

        self.assertIsNotNone(run.created_at.tzinfo)
        self.assertEqual(run.request.idempotency_key, "request-1")
        self.assertEqual(run.request.queue, "default")

    async def test_lookup_methods_raise_stable_not_found_errors(self) -> None:
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_definition("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_run("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_attempt("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_attempts("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_run_transitions("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_attempt_transitions("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_events("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.get_artifact("missing")
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_artifacts("missing")

    async def test_run_and_attempt_metadata_rejects_unsafe_values(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            await self.store.create_run(
                TaskExecutionRequest(definition_id="hash-classify"),
                metadata={"raw": b"bytes"},
            )

        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        with self.assertRaises(AssertionError):
            await self.store.create_attempt(
                run.run_id,
                metadata={"raw": object()},
            )
        with self.assertRaises(AssertionError):
            await self.store.append_artifact(
                run.run_id,
                ref=TaskArtifactRef(
                    artifact_id="artifact-1",
                    store="local",
                    storage_key="ar/artifact-1",
                ),
                purpose=TaskArtifactPurpose.INPUT,
                metadata={"raw": object()},
            )

    async def test_terminal_run_cannot_create_attempt(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        failed = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.FAILED,
            reason="validation_failed",
        )

        self.assertEqual(failed.state, TaskRunState.FAILED)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(run.run_id)

    async def test_assign_claim_rejects_duplicate_claims(self) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        validated = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        queued = await self.store.transition_run(
            validated.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            queued.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )

        self.assertIsNotNone(claimed.claim)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.assign_claim(
                queued.run_id,
                from_states={TaskRunState.CLAIMED},
                worker_id="worker-2",
                lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
                reason="claimed_again",
            )

    async def test_attempt_transition_rejects_stale_expected_state(
        self,
    ) -> None:
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-classify")
        )
        attempt = await self.store.create_attempt(run.run_id)

        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                attempt.attempt_id,
                from_states={TaskAttemptState.RUNNING},
                to_state=TaskAttemptState.SUCCEEDED,
                reason="stale",
            )


if __name__ == "__main__":
    main()
