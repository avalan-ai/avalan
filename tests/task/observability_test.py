from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.event import Event, EventType
from avalan.task import (
    DirectTaskRunner,
    FanoutObservabilitySink,
    HmacProvider,
    NoopObservabilitySink,
    ObservabilitySink,
    ObservabilitySinkHealth,
    ObservabilitySinkType,
    SanitizedTaskEventDraft,
    TaskDefinition,
    TaskDirectTarget,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskObservedEvent,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRunState,
    TaskTargetContext,
    UsageRecord,
    UsageSource,
    UsageTotals,
    record_observability_event,
    record_observability_usage,
)
from avalan.task.stores import InMemoryTaskStore


class StaticHmacProvider(HmacProvider):
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class FakeUsageResponse:
    input_token_count = 3
    output_token_count = 4


class EventAndUsageTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        if context.event_listener is not None:
            result = context.event_listener(
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={
                        "status": "ok",
                        "arguments": {"query": "private query"},
                        "result": "private result",
                    },
                )
            )
            if result is not None:
                await result
        await context.observe_usage(FakeUsageResponse())
        return "done"


@dataclass(slots=True, kw_only=True)
class RecordingSink(ObservabilitySink):
    name: str = "recording"
    events: list[TaskObservedEvent] = field(default_factory=list)
    usages: list[tuple[str, str | None, UsageSource, UsageTotals]] = field(
        default_factory=list
    )

    async def record_event(self, event: TaskObservedEvent) -> None:
        self.events.append(event)

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.usages.append((run_id, attempt_id, source, totals))

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=len(self.events),
            usage_count=len(self.usages),
        )


@dataclass(slots=True, kw_only=True)
class RecordedUsageSink(RecordingSink):
    records: list[UsageRecord] = field(default_factory=list)

    async def record_usage_record(self, record: UsageRecord) -> None:
        self.records.append(record)

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=len(self.events),
            usage_count=len(self.usages) + len(self.records),
        )


class FailingSink(ObservabilitySink):
    async def record_event(self, event: TaskObservedEvent) -> None:
        raise RuntimeError("private event sink failure")

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        raise RuntimeError("private usage sink failure")

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(name="failing")


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
        observability=TaskObservabilityPolicy(
            capture_events=False,
            metrics=True,
            trace=False,
        ),
        privacy=TaskPrivacyPolicy(),
    )


class ObservabilitySinkTest(IsolatedAsyncioTestCase):
    async def test_noop_sink_tracks_safe_health_counters(self) -> None:
        sink = NoopObservabilitySink()

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="model_end",
                category=TaskEventCategory.MODEL,
            )
        )
        await sink.record_usage(
            run_id="run-1",
            attempt_id="attempt-1",
            source=UsageSource.EXACT,
            totals=UsageTotals(input_tokens=1),
            metadata={"safe": True},
        )

        health = sink.health()
        self.assertTrue(health.healthy)
        self.assertEqual(health.event_count, 1)
        self.assertEqual(health.usage_count, 1)
        self.assertEqual(health.failure_count, 0)

    async def test_fanout_isolates_failures_and_keeps_health_counters(
        self,
    ) -> None:
        recording = RecordingSink()
        sink = FanoutObservabilitySink(sinks=(FailingSink(), recording))
        event = SanitizedTaskEventDraft(
            event_type="tool_result",
            category=TaskEventCategory.TOOL,
            payload={"status": "ok"},
        )

        await sink.record_event(event)
        await sink.record_usage(
            run_id="run-1",
            source=UsageSource.ESTIMATED,
            totals=UsageTotals(output_tokens=2),
        )

        health = sink.health()
        self.assertFalse(health.healthy)
        self.assertEqual(health.event_count, 1)
        self.assertEqual(health.usage_count, 1)
        self.assertEqual(health.failure_count, 2)
        self.assertEqual(health.last_failure_code, "RuntimeError")
        self.assertEqual(recording.events, [event])
        self.assertEqual(len(recording.usages), 1)
        self.assertNotIn("private", str(health))

    async def test_safe_record_helpers_drop_sink_failures(self) -> None:
        event = SanitizedTaskEventDraft(
            event_type="tool_result",
            category=TaskEventCategory.TOOL,
            payload={"status": "ok"},
        )

        await record_observability_event(FailingSink(), event)
        await record_observability_usage(
            FailingSink(),
            run_id="run-1",
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=1),
        )

    async def test_usage_helper_prefers_record_hook(self) -> None:
        store = InMemoryTaskStore()
        await store.register_definition(
            definition(),
            definition_hash="hash-recorded-usage",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-recorded-usage")
        )
        record = await store.append_usage(
            run.run_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=1),
        )
        sink = RecordedUsageSink()

        await record_observability_usage(
            sink,
            run_id=record.run_id,
            source=record.source,
            totals=record.totals,
            record=record,
        )

        self.assertEqual(sink.records, [record])
        self.assertEqual(sink.usages, [])

    async def test_runner_success_survives_sink_failures(self) -> None:
        store = InMemoryTaskStore()
        recording = RecordingSink()
        sink = FanoutObservabilitySink(sinks=(FailingSink(), recording))
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventAndUsageTarget()),
            hmac_provider=StaticHmacProvider(),
            observability_sink=sink,
            definition_hash=lambda task: "hash-observability-sinks",
        )

        result = await runner.run(
            definition(),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(recording.events), 1)
        self.assertEqual(len(recording.usages), 1)
        self.assertEqual(sink.health().failure_count, 2)
        self.assertEqual(len(await store.list_usage(result.run.run_id)), 1)
        self.assertNotIn("private query", str(recording.events))
        self.assertNotIn("private result", str(recording.events))

    async def test_runner_success_survives_usage_store_failures(self) -> None:
        store = InMemoryTaskStore()
        recording = RecordingSink()
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventAndUsageTarget()),
            hmac_provider=StaticHmacProvider(),
            observability_sink=recording,
            definition_hash=lambda task: "hash-observability-usage-store",
        )

        with patch.object(
            store,
            "append_usage",
            side_effect=RuntimeError("private usage store failure"),
        ):
            result = await runner.run(
                definition(),
                input_value="private prompt",
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(await store.list_usage(result.run.run_id), ())
        self.assertEqual(len(recording.usages), 1)
        self.assertNotIn("private usage store failure", str(result.run))

    async def test_noop_observability_policy_does_not_emit_to_sink(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        sink = NoopObservabilitySink()
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventAndUsageTarget()),
            hmac_provider=StaticHmacProvider(),
            observability_sink=sink,
            definition_hash=lambda task: "hash-observability-noop",
        )

        result = await runner.run(
            TaskDefinition(
                task=TaskMetadata(name="summarize", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/summarize.toml"),
                observability=TaskObservabilityPolicy.noop(),
                privacy=TaskPrivacyPolicy(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(sink.health().event_count, 0)
        self.assertEqual(sink.health().usage_count, 0)

    async def test_noop_sink_does_not_emit_usage_when_metrics_are_enabled(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        sink = RecordingSink()
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventAndUsageTarget()),
            hmac_provider=StaticHmacProvider(),
            observability_sink=sink,
            definition_hash=lambda task: "hash-observability-noop-sink",
        )

        result = await runner.run(
            TaskDefinition(
                task=TaskMetadata(name="summarize", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/summarize.toml"),
                observability=TaskObservabilityPolicy(
                    sinks=(ObservabilitySinkType.NOOP,),
                    metrics=True,
                    trace=False,
                    capture_events=False,
                ),
                privacy=TaskPrivacyPolicy(),
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(sink.events, [])
        self.assertEqual(sink.usages, [])
        self.assertEqual(len(await store.list_usage(result.run.run_id)), 1)


if __name__ == "__main__":
    main()
