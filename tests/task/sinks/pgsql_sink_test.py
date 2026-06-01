from pathlib import Path
from sys import path as sys_path
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

sys_path.append(str(Path(__file__).parents[1] / "stores"))

from pgsql_contract_test import (  # type: ignore[import-not-found]
    FakeCursor,
    FakePgsqlTaskDatabase,
    SequenceClock,
    SequenceIds,
)

from avalan.task import (
    PgsqlInspectionSink,
    SanitizedTaskEventDraft,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    UsageSource,
    UsageTotals,
    record_observability_event,
    record_observability_usage,
)
from avalan.task.stores import PgsqlTaskStore


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


class PgsqlInspectionSinkTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.database = FakePgsqlTaskDatabase()
        self.store = PgsqlTaskStore(
            self.database,
            clock=SequenceClock(),
            id_factory=SequenceIds(),
        )
        await self.store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        self.run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        self.attempt = await self.store.create_attempt(self.run.run_id)

    async def test_records_event_draft_and_usage_for_inspection(
        self,
    ) -> None:
        sink = PgsqlInspectionSink(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="model_complete",
                category=TaskEventCategory.MODEL,
                payload={"status": "ok"},
            )
        )
        await sink.record_usage(
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=5,
                reasoning_tokens=7,
                total_tokens=8,
            ),
            metadata={"provider": "test"},
        )

        events = await sink.events(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )
        usage = await sink.usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].category, TaskEventCategory.MODEL)
        self.assertEqual(events[0].payload["status"], "ok")
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].source, UsageSource.EXACT)
        self.assertEqual(usage[0].metadata["provider"], "test")
        self.assertEqual(
            await sink.totals(self.run.run_id),
            UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=5,
                reasoning_tokens=7,
                total_tokens=8,
            ),
        )
        self.assertEqual(
            await self.store.list_run_transitions(self.run.run_id), ()
        )
        self.assertEqual(
            await self.store.list_attempt_transitions(self.attempt.attempt_id),
            (),
        )
        self.assertTrue(sink.health().healthy)
        self.assertEqual(sink.health().event_count, 1)
        self.assertEqual(sink.health().usage_count, 1)

    async def test_usage_uses_configured_attempt_when_not_provided(
        self,
    ) -> None:
        sink = PgsqlInspectionSink(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        await sink.record_usage(
            run_id=self.run.run_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=8),
        )

        usage = await sink.usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].attempt_id, self.attempt.attempt_id)

    async def test_recorded_events_are_not_duplicated_by_default(
        self,
    ) -> None:
        stored = await self.store.append_event(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            event_type="model_complete",
            category=TaskEventCategory.MODEL,
            payload={"status": "ok"},
        )
        sink = PgsqlInspectionSink(store=self.store)

        await sink.record_event(stored)

        self.assertEqual(
            await self.store.list_events(self.run.run_id), (stored,)
        )
        self.assertEqual(sink.health().event_count, 1)

    async def test_recorded_events_can_be_copied_when_requested(
        self,
    ) -> None:
        stored = await self.store.append_event(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            event_type="model_complete",
            category=TaskEventCategory.MODEL,
            payload={"status": "ok"},
        )
        sink = PgsqlInspectionSink(
            store=self.store,
            persist_recorded_events=True,
        )

        await sink.record_event(stored)

        events = await self.store.list_events(self.run.run_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], stored)
        self.assertEqual(events[1].sequence, 2)

    async def test_events_support_incremental_fetch(self) -> None:
        await self.store.append_event(
            self.run.run_id,
            event_type="model_start",
            category=TaskEventCategory.MODEL,
            payload={"status": "first"},
        )
        second = await self.store.append_event(
            self.run.run_id,
            event_type="model_complete",
            category=TaskEventCategory.MODEL,
            payload={"status": "second"},
        )
        sink = PgsqlInspectionSink(store=self.store)

        events = await sink.events(self.run.run_id, after_sequence=1)

        self.assertEqual(events, (second,))
        with self.assertRaises(AssertionError):
            await sink.events(self.run.run_id, after_sequence=-1)

    async def test_recorded_usage_is_not_duplicated_by_default(self) -> None:
        stored = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=9),
            metadata={"provider": "test"},
        )
        sink = PgsqlInspectionSink(store=self.store)

        await record_observability_usage(
            sink,
            run_id=stored.run_id,
            attempt_id=stored.attempt_id,
            source=stored.source,
            totals=stored.totals,
            metadata=stored.metadata,
            record=stored,
        )

        self.assertEqual(
            await self.store.list_usage(self.run.run_id), (stored,)
        )
        self.assertEqual(sink.health().usage_count, 1)

    async def test_recorded_usage_can_be_copied_when_requested(self) -> None:
        stored = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=9),
            metadata={"provider": "test"},
        )
        sink = PgsqlInspectionSink(
            store=self.store,
            persist_recorded_usage=True,
        )

        await record_observability_usage(
            sink,
            run_id=stored.run_id,
            attempt_id=stored.attempt_id,
            source=stored.source,
            totals=stored.totals,
            metadata=stored.metadata,
            record=stored,
        )

        usage = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(usage), 2)
        self.assertEqual(usage[0], stored)
        self.assertEqual(usage[1].sequence, 2)

    async def test_recorded_usage_copy_failures_are_counted(self) -> None:
        stored = await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=9),
        )
        sink = PgsqlInspectionSink(
            store=self.store,
            persist_recorded_usage=True,
        )

        with patch.object(
            FakeCursor,
            "_insert_usage",
            side_effect=RuntimeError("private usage failure"),
        ):
            await record_observability_usage(
                sink,
                run_id=stored.run_id,
                attempt_id=stored.attempt_id,
                source=stored.source,
                totals=stored.totals,
                record=stored,
            )

        health = sink.health()
        self.assertFalse(health.healthy)
        self.assertEqual(health.failure_count, 1)
        self.assertEqual(health.last_failure_code, "TaskStoreError")
        self.assertNotIn("private", str(health))
        self.assertEqual(
            await self.store.list_usage(self.run.run_id), (stored,)
        )

    async def test_event_draft_requires_run_id(self) -> None:
        sink = PgsqlInspectionSink(store=self.store)

        with self.assertRaises(ValueError):
            await sink.record_event(
                SanitizedTaskEventDraft(
                    event_type="model_complete",
                    category=TaskEventCategory.MODEL,
                )
            )

        self.assertEqual(sink.health().failure_count, 1)
        self.assertEqual(sink.health().last_failure_code, "ValueError")

    async def test_write_failures_are_counted_and_helper_isolates_them(
        self,
    ) -> None:
        sink = PgsqlInspectionSink(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        with patch.object(
            FakeCursor,
            "_insert_event",
            side_effect=RuntimeError("private event failure"),
        ):
            await record_observability_event(
                sink,
                SanitizedTaskEventDraft(
                    event_type="model_complete",
                    category=TaskEventCategory.MODEL,
                ),
            )

        with patch.object(
            FakeCursor,
            "_insert_usage",
            side_effect=RuntimeError("private usage failure"),
        ):
            await record_observability_usage(
                sink,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(total_tokens=1),
            )

        health = sink.health()
        self.assertFalse(health.healthy)
        self.assertEqual(health.failure_count, 2)
        self.assertEqual(health.last_failure_code, "TaskStoreError")
        self.assertNotIn("private", str(health))
        self.assertEqual(await self.store.list_events(self.run.run_id), ())
        self.assertEqual(await self.store.list_usage(self.run.run_id), ())
        self.assertEqual(
            await self.store.list_run_transitions(self.run.run_id), ()
        )

    async def test_from_database_builds_sink_store(self) -> None:
        sink = PgsqlInspectionSink.from_database(
            self.database,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="model_complete",
                category=TaskEventCategory.MODEL,
            )
        )

        self.assertEqual(len(await sink.events(self.run.run_id)), 1)


if __name__ == "__main__":
    main()
