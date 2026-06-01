from ..event import (
    SanitizedTaskEvent,
    SanitizedTaskEventDraft,
)
from ..feature_gate import TaskFeature, feature_diagnostic
from ..observability import (
    ObservabilitySink,
    ObservabilitySinkHealth,
    TaskObservedEvent,
)
from ..usage import UsageSource, UsageTotals, freeze_usage_metadata
from ._shared import (
    KNOWN_EVENT_TYPES,
    assert_label_value,
    assert_non_empty_string,
    event_type_label,
    usage_counter_values,
)

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import TracebackType
from typing import Protocol, TypeAlias, cast

ModuleImporter: TypeAlias = Callable[[str], object]

_DEFAULT_INSTRUMENTATION_NAME = "avalan.task"
_EVENT_SPAN_NAME = "avalan.task.event"
_USAGE_SPAN_NAME = "avalan.task.usage"
_EVENT_COUNTER = "avalan.task.observability.events"
_USAGE_RECORD_COUNTER = "avalan.task.observability.usage_records"
_USAGE_TOKEN_COUNTER = "avalan.task.observability.usage_tokens"
_EVENT_CATEGORY_ATTRIBUTE = "task.event.category"
_EVENT_TYPE_ATTRIBUTE = "task.event.type"
_USAGE_SOURCE_ATTRIBUTE = "task.usage.source"
_USAGE_COUNTER_ATTRIBUTE = "task.usage.counter"
_USAGE_ATTRIBUTE_PREFIX = "task.usage."


class OpenTelemetrySpanContext(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class OpenTelemetryTracer(Protocol):
    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
    ) -> OpenTelemetrySpanContext: ...


class OpenTelemetryCounter(Protocol):
    def add(
        self,
        amount: int | float,
        attributes: Mapping[str, object] | None = None,
    ) -> None: ...


class OpenTelemetryMeter(Protocol):
    def create_counter(
        self,
        name: str,
        *,
        description: str = "",
        unit: str = "1",
    ) -> OpenTelemetryCounter: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenTelemetryHandles:
    tracer: OpenTelemetryTracer
    meter: OpenTelemetryMeter

    def __post_init__(self) -> None:
        assert callable(getattr(self.tracer, "start_as_current_span", None))
        assert callable(getattr(self.meter, "create_counter", None))


@dataclass(slots=True, kw_only=True)
class OpenTelemetryObservabilitySink(ObservabilitySink):
    name: str = "otel"
    instrumentation_name: str = _DEFAULT_INSTRUMENTATION_NAME
    tracer: OpenTelemetryTracer | None = None
    meter: OpenTelemetryMeter | None = None
    module_importer: ModuleImporter = import_module
    known_event_types: tuple[str, ...] = KNOWN_EVENT_TYPES
    _event_count: int = field(default=0, init=False)
    _usage_count: int = field(default=0, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_code: str | None = field(default=None, init=False)
    _event_counter: OpenTelemetryCounter = field(init=False)
    _usage_record_counter: OpenTelemetryCounter = field(init=False)
    _usage_token_counter: OpenTelemetryCounter = field(init=False)

    def __post_init__(self) -> None:
        assert_label_value(self.name, "name")
        assert_non_empty_string(
            self.instrumentation_name,
            "instrumentation_name",
        )
        assert callable(self.module_importer)
        assert isinstance(self.known_event_types, tuple)
        for event_type in self.known_event_types:
            assert_label_value(event_type, "known_event_types")

        handles = self._handles()
        self.tracer = handles.tracer
        self.meter = handles.meter
        self._event_counter = handles.meter.create_counter(
            _EVENT_COUNTER,
            description="Count sanitized task events.",
            unit="1",
        )
        self._usage_record_counter = handles.meter.create_counter(
            _USAGE_RECORD_COUNTER,
            description="Count task usage records.",
            unit="1",
        )
        self._usage_token_counter = handles.meter.create_counter(
            _USAGE_TOKEN_COUNTER,
            description="Count provider-reported task usage tokens.",
            unit="token",
        )

    async def record_event(self, event: TaskObservedEvent) -> None:
        assert isinstance(event, SanitizedTaskEvent | SanitizedTaskEventDraft)
        attributes = _event_attributes(event, self.known_event_types)
        assert self.tracer is not None
        try:
            with self.tracer.start_as_current_span(
                _EVENT_SPAN_NAME,
                attributes=attributes,
            ):
                self._event_counter.add(1, attributes=attributes)
            self._event_count += 1
        except Exception as error:
            self._record_failure(error)
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
        assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            assert_non_empty_string(attempt_id, "attempt_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)

        assert self.tracer is not None
        try:
            freeze_usage_metadata(metadata)
            span_attributes = _usage_span_attributes(totals)
            metric_attributes = {_USAGE_SOURCE_ATTRIBUTE: source.value}
            with self.tracer.start_as_current_span(
                _USAGE_SPAN_NAME,
                attributes=span_attributes,
            ):
                self._usage_record_counter.add(1, attributes=metric_attributes)
                for counter_name, value in usage_counter_values(totals):
                    self._usage_token_counter.add(
                        value,
                        attributes={
                            _USAGE_SOURCE_ATTRIBUTE: source.value,
                            _USAGE_COUNTER_ATTRIBUTE: counter_name,
                        },
                    )
            self._usage_count += 1
        except Exception as error:
            self._record_failure(error)
            raise

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=self._event_count,
            usage_count=self._usage_count,
            failure_count=self._failure_count,
            last_failure_code=self._last_failure_code,
        )

    def _handles(self) -> OpenTelemetryHandles:
        if self.tracer is not None and self.meter is not None:
            return OpenTelemetryHandles(
                tracer=self.tracer,
                meter=self.meter,
            )
        if self.tracer is not None or self.meter is not None:
            raise AssertionError("tracer and meter must be provided together")
        return _opentelemetry_handles(
            self.module_importer,
            self.instrumentation_name,
        )

    def _record_failure(self, error: Exception) -> None:
        self._failure_count += 1
        self._last_failure_code = type(error).__name__


def _opentelemetry_handles(
    module_importer: ModuleImporter,
    instrumentation_name: str,
) -> OpenTelemetryHandles:
    try:
        module_importer("opentelemetry.sdk")
        trace_module = module_importer("opentelemetry.trace")
        metrics_module = module_importer("opentelemetry.metrics")
    except Exception as error:
        diagnostic = feature_diagnostic(TaskFeature.OPENTELEMETRY)
        raise RuntimeError(diagnostic.code) from error

    get_tracer = getattr(trace_module, "get_tracer", None)
    get_meter = getattr(metrics_module, "get_meter", None)
    if not callable(get_tracer) or not callable(get_meter):
        diagnostic = feature_diagnostic(TaskFeature.OPENTELEMETRY)
        raise RuntimeError(diagnostic.code)
    return OpenTelemetryHandles(
        tracer=cast(OpenTelemetryTracer, get_tracer(instrumentation_name)),
        meter=cast(OpenTelemetryMeter, get_meter(instrumentation_name)),
    )


def _event_attributes(
    event: TaskObservedEvent,
    known_event_types: tuple[str, ...],
) -> dict[str, object]:
    return {
        _EVENT_CATEGORY_ATTRIBUTE: event.category.value,
        _EVENT_TYPE_ATTRIBUTE: event_type_label(
            event.event_type,
            event.category,
            known_event_types,
        ),
    }


def _usage_span_attributes(totals: UsageTotals) -> dict[str, object]:
    return {
        f"{_USAGE_ATTRIBUTE_PREFIX}{counter_name}": value
        for counter_name, value in usage_counter_values(totals)
    }
