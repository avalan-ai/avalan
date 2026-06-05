from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.task import (
    OpenTelemetryObservabilitySink,
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventCategory,
    UsageSource,
    UsageTotals,
    record_observability_event,
)


@dataclass(slots=True)
class FakeSpanContext:
    tracer: "FakeTracer"
    name: str
    attributes: Mapping[str, object] | None

    def __enter__(self) -> object:
        if self.tracer.fail_on_enter:
            raise RuntimeError("private trace backend failure")
        self.tracer.spans.append(
            (
                self.name,
                dict(self.attributes or {}),
            )
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return None


@dataclass(slots=True)
class FakeTracer:
    fail_on_enter: bool = False
    spans: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> FakeSpanContext:
        return FakeSpanContext(
            tracer=self,
            name=name,
            attributes=attributes,
        )


@dataclass(slots=True)
class FakeCounter:
    name: str
    fail_on_add: bool = False
    samples: list[tuple[int | float, dict[str, object]]] = field(
        default_factory=list
    )

    def add(
        self,
        amount: int | float,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        if self.fail_on_add:
            raise RuntimeError("private metric backend failure")
        self.samples.append((amount, dict(attributes or {})))


@dataclass(slots=True)
class FakeMeter:
    fail_on_metric: str | None = None
    counters: dict[str, FakeCounter] = field(default_factory=dict)

    def create_counter(
        self,
        name: str,
        *,
        description: str = "",
        unit: str = "1",
    ) -> FakeCounter:
        assert description
        assert unit
        counter = FakeCounter(
            name=name,
            fail_on_add=name == self.fail_on_metric,
        )
        self.counters[name] = counter
        return counter


def counter_samples(
    meter: FakeMeter,
    name: str,
) -> list[tuple[int | float, dict[str, object]]]:
    return meter.counters[name].samples


class OpenTelemetryObservabilitySinkTest(IsolatedAsyncioTestCase):
    async def test_records_sanitized_event_span_and_metric(self) -> None:
        tracer = FakeTracer()
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

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

        expected_attributes = {
            "task.event.category": "model",
            "task.event.type": "model_complete",
        }
        self.assertEqual(
            tracer.spans,
            [("avalan.task.event", expected_attributes)],
        )
        self.assertEqual(
            counter_samples(meter, "avalan.task.observability.events"),
            [(1, expected_attributes)],
        )
        self.assertTrue(sink.health().healthy)
        self.assertEqual(sink.health().event_count, 1)
        exported = f"{tracer.spans}{meter.counters}"
        self.assertNotIn("run-private", exported)
        self.assertNotIn("attempt-private", exported)
        self.assertNotIn("private prompt", exported)
        self.assertNotIn("raw-model-private", exported)

    async def test_records_usage_with_numeric_span_attributes_only(
        self,
    ) -> None:
        tracer = FakeTracer()
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
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
                "provider_family": "google",
                "cache_read_ephemeral_5m_input_tokens": 3,
                "cache_read_ephemeral_1h_input_tokens": 4,
                "user_id": "private-user",
                "filename": "private.txt",
                "raw_model_id": "provider/model-private",
            },
        )

        self.assertEqual(
            tracer.spans,
            [
                (
                    "avalan.task.usage",
                    {
                        "task.usage.input_tokens": 3,
                        "task.usage.cached_input_tokens": 1,
                        "task.usage.output_tokens": 5,
                    },
                )
            ],
        )
        for _, attributes in tracer.spans:
            for value in attributes.values():
                self.assertIsInstance(value, int)
        self.assertEqual(
            counter_samples(
                meter,
                "avalan.task.observability.usage_records",
            ),
            [(1, {"task.usage.source": "exact"})],
        )
        self.assertEqual(
            counter_samples(meter, "avalan.task.observability.usage_tokens"),
            [
                (
                    3,
                    {
                        "task.usage.source": "exact",
                        "task.usage.counter": "input_tokens",
                    },
                ),
                (
                    1,
                    {
                        "task.usage.source": "exact",
                        "task.usage.counter": "cached_input_tokens",
                    },
                ),
                (
                    5,
                    {
                        "task.usage.source": "exact",
                        "task.usage.counter": "output_tokens",
                    },
                ),
            ],
        )
        self.assertEqual(sink.health().usage_count, 1)
        exported = f"{tracer.spans}{meter.counters}"
        self.assertNotIn("run-private", exported)
        self.assertNotIn("attempt-private", exported)
        self.assertNotIn("google", exported)
        self.assertNotIn("ephemeral", exported)
        self.assertNotIn("private-user", exported)
        self.assertNotIn("private.txt", exported)

    async def test_unknown_event_types_are_collapsed(self) -> None:
        tracer = FakeTracer()
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="tool_call_user_supplied_private_value",
                category=TaskEventCategory.TOOL,
                payload={"tool_call_id": "call-private"},
            )
        )

        self.assertEqual(
            tracer.spans,
            [
                (
                    "avalan.task.event",
                    {
                        "task.event.category": "tool",
                        "task.event.type": "other",
                    },
                )
            ],
        )
        self.assertNotIn("call-private", str(tracer.spans))

    async def test_unknown_category_event_uses_unknown_label(self) -> None:
        tracer = FakeTracer()
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

        await sink.record_event(
            SanitizedTaskEventDraft(
                event_type="private_custom_event",
                category=TaskEventCategory.UNKNOWN,
            )
        )

        self.assertEqual(
            tracer.spans,
            [
                (
                    "avalan.task.event",
                    {
                        "task.event.category": "unknown",
                        "task.event.type": "unknown",
                    },
                )
            ],
        )

    async def test_failures_are_counted_without_sensitive_details(
        self,
    ) -> None:
        tracer = FakeTracer(fail_on_enter=True)
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

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
        self.assertNotIn("private", str(health))

    async def test_usage_failure_is_counted_without_sensitive_details(
        self,
    ) -> None:
        tracer = FakeTracer()
        meter = FakeMeter(
            fail_on_metric="avalan.task.observability.usage_records"
        )
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

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
        tracer = FakeTracer()
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

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
        self.assertEqual(len(tracer.spans), 1)

    async def test_record_helper_isolates_sink_failures(self) -> None:
        tracer = FakeTracer(fail_on_enter=True)
        meter = FakeMeter()
        sink = OpenTelemetryObservabilitySink(
            tracer=tracer,
            meter=meter,
        )

        await record_observability_event(
            sink,
            SanitizedTaskEventDraft(
                event_type="model_complete",
                category=TaskEventCategory.MODEL,
            ),
        )

        self.assertEqual(sink.health().failure_count, 1)


class OpenTelemetryDependencyTest(TestCase):
    def test_lazy_opentelemetry_import_builds_handles(self) -> None:
        tracer = FakeTracer()
        meter = FakeMeter()

        class FakeTraceModule:
            @staticmethod
            def get_tracer(name: str) -> FakeTracer:
                assert name == "avalan.task"
                return tracer

        class FakeMetricsModule:
            @staticmethod
            def get_meter(name: str) -> FakeMeter:
                assert name == "avalan.task"
                return meter

        def import_module(name: str) -> object:
            modules = {
                "opentelemetry.sdk": object(),
                "opentelemetry.trace": FakeTraceModule,
                "opentelemetry.metrics": FakeMetricsModule,
            }
            return modules[name]

        sink = OpenTelemetryObservabilitySink(module_importer=import_module)

        self.assertTrue(sink.health().healthy)
        self.assertIn(
            "avalan.task.observability.events",
            meter.counters,
        )

    def test_missing_opentelemetry_dependency_has_stable_diagnostic(
        self,
    ) -> None:
        def missing_module(name: str) -> object:
            raise ModuleNotFoundError(name)

        with self.assertRaisesRegex(
            RuntimeError,
            "dependency.task_otel_missing",
        ):
            OpenTelemetryObservabilitySink(module_importer=missing_module)

    def test_invalid_opentelemetry_dependency_has_stable_diagnostic(
        self,
    ) -> None:
        def import_module(name: str) -> object:
            modules = {
                "opentelemetry.sdk": object(),
                "opentelemetry.trace": object(),
                "opentelemetry.metrics": object(),
            }
            return modules[name]

        with self.assertRaisesRegex(
            RuntimeError,
            "dependency.task_otel_missing",
        ):
            OpenTelemetryObservabilitySink(module_importer=import_module)

    def test_tracer_and_meter_must_be_provided_together(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "tracer and meter must be provided together",
        ):
            OpenTelemetryObservabilitySink(tracer=FakeTracer())


if __name__ == "__main__":
    main()
