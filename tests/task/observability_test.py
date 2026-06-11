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
    SanitizedTaskUsageEvent,
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
    UsageProviderFamily,
    UsageRecord,
    UsageSource,
    UsageTotals,
    record_observability_event,
    record_observability_usage,
)
from avalan.task.observability import (
    observe_response_usage,
    record_response_usage,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.usage import tag_usage_response, usage_flow_node


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


class ListFailingUsageStore(InMemoryTaskStore):
    async def list_usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        source: UsageSource | None = None,
    ) -> tuple[UsageRecord, ...]:
        raise RuntimeError("private usage list failure")


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

    async def test_usage_helper_fans_out_sanitized_usage_event(self) -> None:
        sink = RecordingSink()

        await record_observability_usage(
            sink,
            run_id="run-private",
            attempt_id="attempt-private",
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=3,
                cached_input_tokens=0,
                output_tokens=5,
                total_tokens=8,
            ),
            metadata={
                "provider_family": UsageProviderFamily.OPENAI,
                "cache_key": "private-cache-key",
                "headers": {"status": "private-header"},
                "raw_model_id": "provider/model-private",
            },
        )

        self.assertEqual(len(sink.events), 1)
        self.assertIsInstance(sink.events[0], SanitizedTaskUsageEvent)
        event = cast(SanitizedTaskUsageEvent, sink.events[0])
        payload = cast(dict[str, object], event.payload)
        self.assertEqual(event.category, TaskEventCategory.USAGE)
        self.assertEqual(payload["source"], "exact")
        self.assertEqual(payload["provider_family"], "openai")
        self.assertEqual(payload["cached_input_tokens"], 0)
        self.assertNotIn("run-private", str(payload))
        self.assertNotIn("attempt-private", str(payload))
        self.assertNotIn("private-cache-key", str(payload))
        self.assertNotIn("private-header", str(payload))
        self.assertNotIn("provider/model-private", str(payload))
        self.assertEqual(len(sink.usages), 1)

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
        self.assertEqual(len(sink.events), 1)
        self.assertEqual(sink.events[0].category, TaskEventCategory.USAGE)

    async def test_response_usage_helper_records_when_dedup_list_fails(
        self,
    ) -> None:
        store = ListFailingUsageStore()
        await store.register_definition(
            definition(),
            definition_hash="hash-list-failure",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-list-failure")
        )
        sink = RecordingSink()

        await record_response_usage(
            sink,
            store=store,
            response=FakeUsageResponse(),
            run_id=run.run_id,
        )

        records = await InMemoryTaskStore.list_usage(store, run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertEqual(len(sink.usages), 1)
        self.assertEqual(len(sink.events), 1)

    async def test_response_usage_helper_deduplicates_mapping_replay(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        await store.register_definition(
            definition(),
            definition_hash="hash-mapping-usage-replay",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-mapping-usage-replay")
        )
        response = {
            "provider_family": "openai",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
                "raw_response_id": "private-response-id",
            },
        }
        equivalent_response = {
            "provider_family": "openai",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
                "raw_response_id": "private-response-id",
            },
        }
        sink = RecordingSink()
        observed: list[TaskObservedEvent] = []

        await record_response_usage(
            sink,
            store=store,
            response=response,
            run_id=run.run_id,
            usage_observer=observed.append,
        )
        await record_response_usage(
            sink,
            store=store,
            response=response,
            run_id=run.run_id,
            usage_observer=observed.append,
        )
        await record_response_usage(
            sink,
            store=store,
            response=equivalent_response,
            run_id=run.run_id,
            usage_observer=observed.append,
        )

        records = await store.list_usage(run.run_id)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].usage_id, records[1].usage_id)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 5)
        self.assertEqual(records[0].totals.total_tokens, 8)
        self.assertEqual(len(sink.usages), 2)
        self.assertEqual(len(sink.events), 2)
        self.assertEqual(len(observed), 2)
        observed_payload = cast(dict[str, object], observed[0].payload)
        self.assertEqual(observed_payload["input_tokens"], 3)
        self.assertNotIn("private-response-id", str(records))
        self.assertNotIn("private-response-id", str(sink.events))

    async def test_response_usage_helper_emits_flow_node_metadata(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        await store.register_definition(
            definition(),
            definition_hash="hash-flow-node-usage",
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id="hash-flow-node-usage")
        )
        response = tag_usage_response(
            FakeUsageResponse(),
            flow_node="analyze_pov_1",
        )
        ignored_response = tag_usage_response(3, flow_node="ignored")
        observed: list[TaskObservedEvent] = []

        await record_response_usage(
            None,
            store=store,
            response=response,
            run_id=run.run_id,
            usage_observer=observed.append,
        )

        records = await store.list_usage(run.run_id)
        payload = cast(dict[str, object], observed[0].payload)
        self.assertNotIn("flow_node", records[0].metadata)
        self.assertEqual(payload["flow_node"], "analyze_pov_1")
        self.assertEqual(usage_flow_node(response), "analyze_pov_1")
        self.assertIsNone(usage_flow_node(ignored_response))

    async def test_observe_response_usage_deduplicates_live_events(
        self,
    ) -> None:
        response = tag_usage_response(
            FakeUsageResponse(),
            flow_node="analyze_pov_1",
        )
        observed: list[TaskObservedEvent] = []
        observed_usage_ids: set[str] = set()

        await observe_response_usage(
            response,
            run_id="run-private",
            attempt_id="attempt-private",
            usage_observer=observed.append,
            observed_usage_ids=observed_usage_ids,
        )
        await observe_response_usage(
            response,
            run_id="run-private",
            attempt_id="attempt-private",
            usage_observer=observed.append,
            observed_usage_ids=observed_usage_ids,
        )

        self.assertEqual(len(observed), 1)
        payload = cast(dict[str, object], observed[0].payload)
        self.assertEqual(payload["input_tokens"], 3)
        self.assertEqual(payload["output_tokens"], 4)
        self.assertEqual(payload["flow_node"], "analyze_pov_1")
        self.assertNotIn("run-private", str(payload))
        self.assertNotIn("attempt-private", str(payload))

    async def test_observe_response_usage_ignores_missing_inputs(
        self,
    ) -> None:
        observed: list[TaskObservedEvent] = []

        await observe_response_usage(
            FakeUsageResponse(),
            run_id="run-private",
            usage_observer=None,
        )
        await observe_response_usage(
            object(),
            run_id="run-private",
            usage_observer=observed.append,
        )
        await observe_response_usage(
            FakeUsageResponse(),
            run_id="run-private",
            usage_observer=lambda event: (_ for _ in ()).throw(
                RuntimeError("private observer failure")
            ),
        )

        self.assertEqual(observed, [])

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
        self.assertEqual(len(recording.events), 2)
        self.assertEqual(len(recording.usages), 1)
        self.assertEqual(sink.health().failure_count, 3)
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
        self.assertEqual(
            [event.category for event in recording.events],
            [TaskEventCategory.TOOL, TaskEventCategory.USAGE],
        )
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
