from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    TaskArtifactPurpose,
    TaskArtifactRetention,
    TaskArtifactState,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRetentionAction,
    TaskRetentionService,
    TaskRetentionStoreNotFoundError,
    UsageSource,
    UsageTotals,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.stores import InMemoryTaskStore


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


class ManualClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value = self.value + delta


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


class TaskRetentionServiceTest(IsolatedAsyncioTestCase):
    async def test_enforce_run_deletes_expired_input_bytes_only(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            clock = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
            task_store = await self._task_store(clock)
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="hash-a"),
                metadata={"tenant": "safe"},
            )
            artifact_store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
            )
            input_ref = await artifact_store.put(
                b"private input",
                artifact_id="input-1",
                media_type="text/plain",
            )
            output_ref = await artifact_store.put(
                b"private output",
                artifact_id="output-1",
                media_type="text/plain",
            )
            await task_store.append_artifact(
                run.run_id,
                ref=input_ref,
                purpose=TaskArtifactPurpose.INPUT,
                retention=TaskArtifactRetention(delete_after_days=1),
                metadata={"identity": {"sha256": "a" * 64}},
            )
            output_record = await task_store.append_artifact(
                run.run_id,
                ref=output_ref,
                purpose=TaskArtifactPurpose.OUTPUT,
                retention=TaskArtifactRetention(delete_after_days=30),
                metadata={"identity": {"sha256": "b" * 64}},
            )
            usage = await task_store.append_usage(
                run.run_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=2, output_tokens=3),
                metadata={"sink": "safe"},
            )
            event = await task_store.append_event(
                run.run_id,
                event_type="engine_start",
                category=TaskEventCategory.ENGINE,
                payload={"duration_ms": 1},
            )
            clock.advance(timedelta(days=2))
            service = TaskRetentionService(
                task_store,
                {"local": artifact_store},
                clock=clock,
            )

            sweep = await service.enforce_run(
                run.run_id,
                purposes={TaskArtifactPurpose.INPUT},
            )

            self.assertEqual(sweep.enforced_at, clock.value)
            self.assertEqual(len(sweep.results), 1)
            result = sweep.results[0]
            self.assertEqual(result.action, TaskRetentionAction.DELETED)
            self.assertEqual(result.artifact_id, input_ref.artifact_id)
            self.assertEqual(result.record.state, TaskArtifactState.DELETED)
            self.assertFalse(Path(tmp, input_ref.storage_key).exists())
            self.assertTrue(Path(tmp, output_ref.storage_key).exists())
            stored_input = await task_store.get_artifact(input_ref.artifact_id)
            stored_output = await task_store.get_artifact(
                output_ref.artifact_id
            )
            self.assertEqual(stored_input.state, TaskArtifactState.DELETED)
            self.assertEqual(stored_output, output_record)
            input_metadata = cast(Mapping[str, object], stored_input.metadata)
            self.assertIn("identity", input_metadata)
            audit = cast(Mapping[str, object], input_metadata["retention"])
            self.assertEqual(audit["action"], "deleted")
            self.assertEqual(audit["reason"], "retention_expired")
            self.assertEqual(audit["purpose"], "input")
            self.assertNotIn("storage_key", audit)
            self.assertEqual(
                (await task_store.get_run(run.run_id)).metadata["tenant"],
                "safe",
            )
            self.assertEqual(await task_store.list_usage(run.run_id), (usage,))
            self.assertEqual(
                await task_store.list_events(run.run_id), (event,)
            )

    async def test_output_retention_is_independent_from_input_retention(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            clock = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
            task_store = await self._task_store(clock)
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="hash-a")
            )
            artifact_store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
            )
            input_ref = await artifact_store.put(
                b"private input",
                artifact_id="input-1",
            )
            output_ref = await artifact_store.put(
                b"private output",
                artifact_id="output-1",
            )
            await task_store.append_artifact(
                run.run_id,
                ref=input_ref,
                purpose=TaskArtifactPurpose.INPUT,
                retention=TaskArtifactRetention(delete_after_days=1),
            )
            await task_store.append_artifact(
                run.run_id,
                ref=output_ref,
                purpose=TaskArtifactPurpose.OUTPUT,
                retention=TaskArtifactRetention(delete_after_days=1),
            )
            service = TaskRetentionService(
                task_store,
                {"local": artifact_store},
                clock=lambda: datetime(2026, 1, 3, tzinfo=UTC),
            )

            sweep = await service.enforce_run(
                run.run_id,
                purposes={TaskArtifactPurpose.OUTPUT},
            )

            self.assertEqual(
                [result.artifact_id for result in sweep.results],
                [output_ref.artifact_id],
            )
            self.assertTrue(Path(tmp, input_ref.storage_key).exists())
            self.assertFalse(Path(tmp, output_ref.storage_key).exists())
            self.assertEqual(
                (await task_store.get_artifact(input_ref.artifact_id)).state,
                TaskArtifactState.READY,
            )
            self.assertEqual(
                (await task_store.get_artifact(output_ref.artifact_id)).state,
                TaskArtifactState.DELETED,
            )

    async def test_unexpired_and_unconfigured_retention_are_skipped(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            task_store = await self._task_store(
                lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="hash-a")
            )
            artifact_store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
            )
            unconfigured_ref = await artifact_store.put(
                b"private input",
                artifact_id="input-1",
            )
            future_ref = await artifact_store.put(
                b"private output",
                artifact_id="output-1",
            )
            await task_store.append_artifact(
                run.run_id,
                ref=unconfigured_ref,
                purpose=TaskArtifactPurpose.INPUT,
            )
            await task_store.append_artifact(
                run.run_id,
                ref=future_ref,
                purpose=TaskArtifactPurpose.OUTPUT,
                retention=TaskArtifactRetention(delete_after_days=365),
            )
            service = TaskRetentionService(
                task_store,
                {"local": artifact_store},
            )

            sweep = await service.enforce_run(run.run_id)

            self.assertEqual(sweep.results, ())
            self.assertTrue(Path(tmp, unconfigured_ref.storage_key).exists())
            self.assertTrue(Path(tmp, future_ref.storage_key).exists())
            self.assertEqual(
                (
                    await task_store.get_artifact(unconfigured_ref.artifact_id)
                ).state,
                TaskArtifactState.READY,
            )
            self.assertEqual(
                (await task_store.get_artifact(future_ref.artifact_id)).state,
                TaskArtifactState.READY,
            )

    async def test_missing_artifact_bytes_are_marked_lost(self) -> None:
        with TemporaryDirectory() as tmp:
            clock = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
            task_store = await self._task_store(clock)
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="hash-a")
            )
            artifact_store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
            )
            ref = await artifact_store.put(
                b"private bytes",
                artifact_id="artifact-1",
            )
            await artifact_store.delete(ref)
            await task_store.append_artifact(
                run.run_id,
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                retention=TaskArtifactRetention(
                    expires_at=datetime(2026, 1, 2, tzinfo=UTC),
                    retain_metadata=False,
                ),
                metadata={"identity": {"sha256": "a" * 64}},
            )
            service = TaskRetentionService(
                task_store,
                {"local": artifact_store},
                clock=lambda: datetime(2026, 1, 3, tzinfo=UTC),
            )

            sweep = await service.enforce_run(run.run_id)

            self.assertEqual(len(sweep.results), 1)
            result = sweep.results[0]
            self.assertEqual(result.action, TaskRetentionAction.LOST)
            self.assertEqual(result.record.state, TaskArtifactState.LOST)
            record = await task_store.get_artifact(ref.artifact_id)
            self.assertNotIn("identity", record.metadata)
            audit = cast(Mapping[str, object], record.metadata["retention"])
            self.assertEqual(audit["action"], "lost")
            self.assertEqual(audit["reason"], "artifact_bytes_missing")
            self.assertFalse(audit["retain_metadata"])

    async def test_missing_store_does_not_transition_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            task_store = await self._task_store(
                lambda: datetime(2026, 1, 1, tzinfo=UTC)
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="hash-a")
            )
            artifact_store = LocalArtifactStore(
                tmp,
                raw_storage_allowed=True,
            )
            ref = await artifact_store.put(
                b"private bytes",
                artifact_id="artifact-1",
            )
            await task_store.append_artifact(
                run.run_id,
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
                retention=TaskArtifactRetention(delete_after_days=1),
            )
            service = TaskRetentionService(
                task_store,
                {},
                clock=lambda: datetime(2026, 1, 3, tzinfo=UTC),
            )

            with self.assertRaises(TaskRetentionStoreNotFoundError):
                await service.enforce_run(run.run_id)

            self.assertTrue(Path(tmp, ref.storage_key).exists())
            self.assertEqual(
                (await task_store.get_artifact(ref.artifact_id)).state,
                TaskArtifactState.READY,
            )

    async def test_invalid_inputs_fail_fast(self) -> None:
        task_store = await self._task_store(
            lambda: datetime(2026, 1, 1, tzinfo=UTC)
        )
        service = TaskRetentionService(task_store, {})

        with self.assertRaises(AssertionError):
            await service.enforce_run("")
        with self.assertRaises(AssertionError):
            await service.enforce_run(
                "run-1",
                purposes=cast(Collection[TaskArtifactPurpose], ()),
            )
        with self.assertRaises(AssertionError):
            await service.enforce_run(
                "run-1",
                purposes=cast(
                    Collection[TaskArtifactPurpose],
                    ("input",),
                ),
            )

    async def _task_store(
        self,
        clock: Callable[[], datetime],
    ) -> InMemoryTaskStore:
        store = InMemoryTaskStore(
            clock=clock,
            id_factory=SequenceIds(),
        )
        await store.register_definition(definition(), definition_hash="hash-a")
        return store


if __name__ == "__main__":
    main()
