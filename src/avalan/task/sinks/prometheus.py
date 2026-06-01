from ..event import (
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
    TaskEventCategory,
)
from ..feature_gate import TaskFeature, feature_diagnostic
from ..observability import (
    ObservabilitySink,
    ObservabilitySinkHealth,
    TaskObservedEvent,
)
from ..usage import UsageSource, UsageTotals, freeze_usage_metadata

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Protocol, TypeAlias, cast

ModuleImporter: TypeAlias = Callable[[str], object]

_DEFAULT_NAMESPACE = "avalan_task"
_EVENT_METRIC = "observability_events"
_USAGE_RECORD_METRIC = "observability_usage_records"
_USAGE_TOKEN_METRIC = "observability_usage_tokens"
_FAILURE_METRIC = "observability_sink_failures"
_OTHER_EVENT_TYPE = "other"
_UNKNOWN_EVENT_TYPE = "unknown"
_KNOWN_EVENT_TYPES = (
    "call_prepare_after",
    "call_prepare_before",
    "end",
    "engine_error",
    "engine_start",
    "engine_stop",
    "event_sanitization_failed",
    "input_token_count_after",
    "input_token_count_before",
    "memory_lookup",
    "memory_store",
    "model_complete",
    "model_end",
    "model_error",
    "model_start",
    "start",
    "stream_end",
    "token_generated",
    "tool_call",
    "tool_error",
    "tool_result",
    "unknown",
)


class PrometheusCounterChild(Protocol):
    def inc(self, amount: float = 1.0) -> None: ...


class PrometheusCounter(Protocol):
    def labels(self, **labels: str) -> PrometheusCounterChild: ...


class PrometheusCounterFactory(Protocol):
    def __call__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: object | None = None,
    ) -> PrometheusCounter: ...


@dataclass(slots=True, kw_only=True)
class PrometheusObservabilitySink(ObservabilitySink):
    name: str = "prometheus"
    namespace: str = _DEFAULT_NAMESPACE
    registry: object | None = None
    counter_factory: PrometheusCounterFactory | None = None
    module_importer: ModuleImporter = import_module
    known_event_types: tuple[str, ...] = _KNOWN_EVENT_TYPES
    _event_count: int = field(default=0, init=False)
    _usage_count: int = field(default=0, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_code: str | None = field(default=None, init=False)
    _event_counter: PrometheusCounter = field(init=False)
    _usage_record_counter: PrometheusCounter = field(init=False)
    _usage_token_counter: PrometheusCounter = field(init=False)
    _failure_counter: PrometheusCounter = field(init=False)

    def __post_init__(self) -> None:
        _assert_metric_component(self.name, "name")
        _assert_metric_component(self.namespace, "namespace")
        assert callable(self.module_importer)
        assert isinstance(self.known_event_types, tuple)
        for event_type in self.known_event_types:
            _assert_label_value(event_type, "known_event_types")

        factory = self.counter_factory or _prometheus_counter_factory(
            self.module_importer
        )
        self._event_counter = _counter(
            factory,
            self.registry,
            self.namespace,
            _EVENT_METRIC,
            "Count sanitized task events.",
            ("category", "event_type"),
        )
        self._usage_record_counter = _counter(
            factory,
            self.registry,
            self.namespace,
            _USAGE_RECORD_METRIC,
            "Count task usage records.",
            ("source",),
        )
        self._usage_token_counter = _counter(
            factory,
            self.registry,
            self.namespace,
            _USAGE_TOKEN_METRIC,
            "Count provider-reported task usage tokens.",
            ("source", "counter"),
        )
        self._failure_counter = _counter(
            factory,
            self.registry,
            self.namespace,
            _FAILURE_METRIC,
            "Count Prometheus observability sink failures.",
            ("operation",),
        )

    async def record_event(self, event: TaskObservedEvent) -> None:
        assert isinstance(event, SanitizedTaskEvent | SanitizedTaskEventDraft)
        try:
            self._event_counter.labels(
                category=event.category.value,
                event_type=_event_type_label(
                    event.event_type,
                    event.category,
                    self.known_event_types,
                ),
            ).inc()
            self._event_count += 1
        except Exception as error:
            self._record_failure("record_event", error)
            raise

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)
        freeze_usage_metadata(metadata)

        try:
            labels = {"source": source.value}
            self._usage_record_counter.labels(**labels).inc()
            for counter_name, value in _usage_counter_values(totals):
                self._usage_token_counter.labels(
                    source=source.value,
                    counter=counter_name,
                ).inc(float(value))
            self._usage_count += 1
        except Exception as error:
            self._record_failure("record_usage", error)
            raise

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=self._event_count,
            usage_count=self._usage_count,
            failure_count=self._failure_count,
            last_failure_code=self._last_failure_code,
        )

    def _record_failure(self, operation: str, error: Exception) -> None:
        self._failure_count += 1
        self._last_failure_code = type(error).__name__
        try:
            self._failure_counter.labels(operation=operation).inc()
        except Exception:
            return


def _prometheus_counter_factory(
    module_importer: ModuleImporter,
) -> PrometheusCounterFactory:
    try:
        module = module_importer("prometheus_client")
    except Exception as error:
        diagnostic = feature_diagnostic(TaskFeature.PROMETHEUS)
        raise RuntimeError(diagnostic.code) from error
    counter = getattr(module, "Counter", None)
    if not callable(counter):
        diagnostic = feature_diagnostic(TaskFeature.PROMETHEUS)
        raise RuntimeError(diagnostic.code)
    return cast(PrometheusCounterFactory, counter)


def _counter(
    factory: PrometheusCounterFactory,
    registry: object | None,
    namespace: str,
    metric_name: str,
    documentation: str,
    labelnames: tuple[str, ...],
) -> PrometheusCounter:
    if registry is None:
        return factory(
            f"{namespace}_{metric_name}",
            documentation,
            labelnames=labelnames,
        )
    return factory(
        f"{namespace}_{metric_name}",
        documentation,
        labelnames=labelnames,
        registry=registry,
    )


def _event_type_label(
    event_type: str,
    category: TaskEventCategory,
    known_event_types: tuple[str, ...],
) -> str:
    if event_type in known_event_types:
        return event_type
    if category == TaskEventCategory.UNKNOWN:
        return _UNKNOWN_EVENT_TYPE
    return _OTHER_EVENT_TYPE


def _usage_counter_values(
    totals: UsageTotals,
) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    for counter_name in (
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        value = getattr(totals, counter_name)
        if value is not None:
            values.append((counter_name, value))
    return tuple(values)


def _assert_metric_component(value: str | None, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert value is not None
    assert len(value) <= 64, f"{field_name} must be no longer than 64 chars"
    assert value[0].isalpha(), f"{field_name} must start with a letter"
    assert all(
        character.isalnum() or character == "_" for character in value
    ), f"{field_name} must be a Prometheus metric component"


def _assert_label_value(value: str | None, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert value is not None
    assert len(value) <= 64, f"{field_name} must be no longer than 64 chars"
    assert all(
        character.isalnum() or character == "_" for character in value
    ), f"{field_name} must be a safe label value"


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
