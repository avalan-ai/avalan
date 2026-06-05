from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.task import (
    PrometheusObservabilitySink,
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventCategory,
    UsageSource,
    UsageTotals,
    record_observability_event,
)


@dataclass(slots=True)
class FakeCounterChild:
    counter: "FakeCounter"
    label_values: tuple[tuple[str, str], ...]

    def inc(self, amount: float = 1.0) -> None:
        if self.counter.fail_on_inc:
            raise RuntimeError("private metric backend failure")
        self.counter.samples[self.label_values] = (
            self.counter.samples.get(self.label_values, 0.0) + amount
        )


@dataclass(slots=True)
class FakeCounter:
    name: str
    labelnames: tuple[str, ...]
    fail_on_inc: bool = False
    samples: dict[tuple[tuple[str, str], ...], float] = field(
        default_factory=dict
    )

    def labels(self, **labels: str) -> FakeCounterChild:
        self.assert_labels(labels)
        return FakeCounterChild(
            counter=self,
            label_values=tuple(sorted(labels.items())),
        )

    def assert_labels(self, labels: Mapping[str, str]) -> None:
        assert tuple(labels) == self.labelnames
        for value in labels.values():
            assert isinstance(value, str)


@dataclass(slots=True)
class FakeCounterFactory:
    fail_on_metric: str | None = None
    counters: dict[str, FakeCounter] = field(default_factory=dict)
    registries: dict[str, object | None] = field(default_factory=dict)

    def __call__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: object | None = None,
    ) -> FakeCounter:
        assert documentation
        self.registries[name] = registry
        counter = FakeCounter(
            name=name,
            labelnames=labelnames,
            fail_on_inc=name == self.fail_on_metric,
        )
        self.counters[name] = counter
        return counter


def metric_samples(
    factory: FakeCounterFactory,
    metric_name: str,
) -> dict[tuple[tuple[str, str], ...], float]:
    return factory.counters[metric_name].samples


class PrometheusObservabilitySinkTest(IsolatedAsyncioTestCase):
    async def test_registry_is_passed_to_counter_factory(self) -> None:
        factory = FakeCounterFactory()
        registry = object()

        PrometheusObservabilitySink(
            counter_factory=factory,
            registry=registry,
        )

        self.assertEqual(
            set(factory.registries.values()),
            {registry},
        )

    async def test_records_sanitized_event_and_numeric_usage_counters(
        self,
    ) -> None:
        factory = FakeCounterFactory()
        sink = PrometheusObservabilitySink(counter_factory=factory)

        await sink.record_event(
            SanitizedTaskEvent(
                event_id="event-1",
                run_id="run-private",
                attempt_id="attempt-private",
                sequence=1,
                event_type="model_complete",
                category=TaskEventCategory.MODEL,
                created_at=datetime.now(timezone.utc),
                payload={
                    "prompt": "private prompt",
                    "model_id": "raw-model-private",
                },
            )
        )
        await sink.record_usage(
            run_id="run-private",
            attempt_id="attempt-private",
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                output_tokens=5,
            ),
            metadata={
                "provider_family": "anthropic",
                "cache_creation_ephemeral_5m_input_tokens": 3,
                "cache_creation_ephemeral_1h_input_tokens": 4,
                "user_id": "private-user",
                "filename": "private.txt",
                "raw_model_id": "provider/model-private",
            },
        )

        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_events",
            ),
            {
                (
                    ("category", "model"),
                    ("event_type", "model_complete"),
                ): 1.0
            },
        )
        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_usage_records",
            ),
            {(("source", "exact"),): 1.0},
        )
        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_usage_tokens",
            ),
            {
                (
                    ("counter", "input_tokens"),
                    ("source", "exact"),
                ): 3.0,
                (
                    ("counter", "cached_input_tokens"),
                    ("source", "exact"),
                ): 1.0,
                (
                    ("counter", "output_tokens"),
                    ("source", "exact"),
                ): 5.0,
            },
        )
        self.assertTrue(sink.health().healthy)
        self.assertEqual(sink.health().event_count, 1)
        self.assertEqual(sink.health().usage_count, 1)
        exported = str(factory.counters)
        self.assertNotIn("run-private", exported)
        self.assertNotIn("attempt-private", exported)
        self.assertNotIn("anthropic", exported)
        self.assertNotIn("ephemeral", exported)
        self.assertNotIn("private prompt", exported)
        self.assertNotIn("raw-model-private", exported)
        self.assertNotIn("private-user", exported)
        self.assertNotIn("private.txt", exported)

    async def test_unknown_event_types_are_collapsed_to_low_cardinality_label(
        self,
    ) -> None:
        factory = FakeCounterFactory()
        sink = PrometheusObservabilitySink(counter_factory=factory)

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="tool_call_user_supplied_private_value",
                category=TaskEventCategory.TOOL,
                payload={"tool_call_id": "call-private"},
            )
        )

        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_events",
            ),
            {
                (
                    ("category", "tool"),
                    ("event_type", "other"),
                ): 1.0
            },
        )
        self.assertNotIn("call-private", str(factory.counters))

    async def test_unknown_category_event_uses_unknown_label(self) -> None:
        factory = FakeCounterFactory()
        sink = PrometheusObservabilitySink(counter_factory=factory)

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="private_custom_event",
                category=TaskEventCategory.UNKNOWN,
            )
        )

        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_events",
            ),
            {
                (
                    ("category", "unknown"),
                    ("event_type", "unknown"),
                ): 1.0
            },
        )

    async def test_failures_are_counted_without_sensitive_details(
        self,
    ) -> None:
        factory = FakeCounterFactory(
            fail_on_metric="avalan_task_observability_events"
        )
        sink = PrometheusObservabilitySink(counter_factory=factory)

        with self.assertRaises(RuntimeError):
            await sink.record_event(
                SanitizedTaskEventDraft(
                    event_type="model_complete",
                    category=TaskEventCategory.MODEL,
                    payload={"error": "private backend payload"},
                )
            )

        health = sink.health()
        self.assertFalse(health.healthy)
        self.assertEqual(health.event_count, 0)
        self.assertEqual(health.failure_count, 1)
        self.assertEqual(health.last_failure_code, "RuntimeError")
        self.assertEqual(
            metric_samples(
                factory,
                "avalan_task_observability_sink_failures",
            ),
            {(("operation", "record_event"),): 1.0},
        )
        self.assertNotIn("private", str(health))

    async def test_usage_failure_is_counted_without_sensitive_details(
        self,
    ) -> None:
        factory = FakeCounterFactory(
            fail_on_metric="avalan_task_observability_usage_records"
        )
        sink = PrometheusObservabilitySink(counter_factory=factory)
        factory.counters[
            "avalan_task_observability_sink_failures"
        ].fail_on_inc = True

        with self.assertRaises(RuntimeError):
            await sink.record_usage(
                run_id="run-private",
                source=UsageSource.EXACT,
                totals=UsageTotals(total_tokens=1),
                metadata={"prompt": "private prompt"},
            )

        health = sink.health()
        self.assertFalse(health.healthy)
        self.assertEqual(health.usage_count, 0)
        self.assertEqual(health.failure_count, 1)
        self.assertEqual(health.last_failure_code, "RuntimeError")
        self.assertNotIn("private", str(health))

    async def test_unsafe_usage_metadata_is_dropped(self) -> None:
        factory = FakeCounterFactory()
        sink = PrometheusObservabilitySink(counter_factory=factory)

        await sink.record_usage(
            run_id="run-private",
            source=UsageSource.EXACT,
            totals=UsageTotals(total_tokens=1),
            metadata={"private": object()},
        )

        health = sink.health()
        self.assertTrue(health.healthy)
        self.assertEqual(health.usage_count, 1)
        self.assertEqual(health.failure_count, 0)
        self.assertNotIn("private", str(health))

    async def test_record_helper_isolates_sink_failures(self) -> None:
        factory = FakeCounterFactory(
            fail_on_metric="avalan_task_observability_events"
        )
        sink = PrometheusObservabilitySink(counter_factory=factory)

        await record_observability_event(
            sink,
            SanitizedTaskEventDraft(
                event_type="model_complete",
                category=TaskEventCategory.MODEL,
            ),
        )

        self.assertEqual(sink.health().failure_count, 1)


class PrometheusDependencyTest(TestCase):
    def test_lazy_prometheus_import_builds_counter_factory(self) -> None:
        factory = FakeCounterFactory()

        class FakePrometheusModule:
            Counter = factory

        def import_module(name: str) -> object:
            self.assertEqual(name, "prometheus_client")
            return FakePrometheusModule()

        sink = PrometheusObservabilitySink(module_importer=import_module)

        self.assertTrue(sink.health().healthy)
        self.assertIn(
            "avalan_task_observability_events",
            factory.counters,
        )

    def test_missing_prometheus_dependency_has_stable_diagnostic(self) -> None:
        def missing_module(name: str) -> object:
            raise ModuleNotFoundError(name)

        with self.assertRaisesRegex(
            RuntimeError,
            "dependency.task_prometheus_missing",
        ):
            PrometheusObservabilitySink(module_importer=missing_module)

    def test_invalid_prometheus_dependency_has_stable_diagnostic(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "dependency.task_prometheus_missing",
        ):
            PrometheusObservabilitySink(module_importer=lambda name: object())


if __name__ == "__main__":
    main()
