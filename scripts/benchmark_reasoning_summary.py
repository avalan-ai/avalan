#!/usr/bin/env python
"""Run the deterministic credential-free reasoning stream benchmark."""

from argparse import ArgumentParser, Namespace
from asyncio import (
    Event as AsyncEvent,
)
from asyncio import (
    Queue,
    create_task,
    get_running_loop,
    sleep,
    to_thread,
    wait_for,
)
from asyncio import (
    run as asyncio_run,
)
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from gc import collect
from json import dumps
from logging import getLogger
from math import ceil, isfinite
from os import environ
from pathlib import Path
from platform import machine, platform, processor, python_implementation
from statistics import median
from subprocess import run
from sys import executable, stderr, version
from threading import Event as ThreadEvent
from threading import Thread
from time import perf_counter
from tracemalloc import get_traced_memory, start, stop
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from reasoning_summary_json import strict_json_loads

from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import MessageRole
from avalan.event import Event, EventType
from avalan.event.manager import (
    EventDeliveryConfig,
    EventDeliveryPolicy,
    EventHistoryConfig,
    EventListenConfig,
    EventManager,
    EventManagerMode,
    EventSubscriberClass,
)
from avalan.model.nlp.text.generation import (
    _configure_lossless_streamer_handoff,
)
from avalan.model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProjectionState,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    stream_consumer_iterator,
)
from avalan.server.entities import (
    ChatMessage,
    ResponsesRequest,
    ServerOutputRedactionSettings,
)
from avalan.server.routers import responses as responses_router

_MEMORY_SENTINEL = "__REASONING_BENCHMARK_MEMORY__"
_MEMORY_SCOPES = (
    "processing_excluding_source_fixture",
    "retained_total_including_source_fixture",
)
_PHASE9_MEMORY_SENTINEL = "__REASONING_PHASE9_BENCHMARK_MEMORY__"
_PHASE9_ASYNC_SENTINEL = "__REASONING_PHASE9_BENCHMARK_ASYNC__"
_PHASE9_SUMMARY_DELTAS_PER_PART = 64
_PHASE9_WORKLOAD_NAME = "summary_projection_hidden"
_PHASE9_P95_PER_ITEM_MICROSECONDS = 500
_PHASE9_BATCH_BUDGET_MULTIPLIER = 1.25
_PHASE9_WORK_RATIO_LIMIT = 2.5
_PHASE9_HEARTBEAT_INTERVAL_MILLISECONDS = 10
_PHASE9_HEARTBEAT_MAXIMUM_DRIFT_MILLISECONDS = 100
_PHASE9_MAXIMUM_COALESCED_DELTA_CHARACTERS = 4096
_PHASE9_COALESCING_SOURCE_DELTAS = 4097


@dataclass(frozen=True, kw_only=True, slots=True)
class BenchmarkProtocol:
    """Define the locked local reasoning benchmark procedure."""

    warmups: int
    samples: int
    delta_counts: tuple[int, ...]
    delta_text: str
    percentile_method: str
    gc_before_each_sample: bool
    reset_state_before_each_sample: bool
    gc_before_memory_probe: bool
    elapsed_clock: str
    peak_memory_probe: str
    memory_probe_scopes: tuple[str, ...]
    memory_subprocess_timeout_seconds: int
    network_allowed: bool
    network_guard: str

    def __post_init__(self) -> None:
        assert type(self.warmups) is int
        assert type(self.samples) is int
        assert all(type(count) is int for count in self.delta_counts)
        assert type(self.memory_subprocess_timeout_seconds) is int
        assert type(self.gc_before_each_sample) is bool
        assert type(self.reset_state_before_each_sample) is bool
        assert type(self.gc_before_memory_probe) is bool
        assert type(self.network_allowed) is bool
        assert self.warmups == 3
        assert self.samples == 20
        assert self.delta_counts == (4096, 8192)
        assert self.delta_text == "x"
        assert self.percentile_method == "nearest_rank"
        assert self.gc_before_each_sample
        assert self.reset_state_before_each_sample
        assert self.gc_before_memory_probe
        assert self.elapsed_clock == "perf_counter"
        assert self.peak_memory_probe == "tracemalloc"
        assert self.memory_probe_scopes == _MEMORY_SCOPES
        assert self.memory_subprocess_timeout_seconds == 60
        assert not self.network_allowed
        assert self.network_guard == "deny_socket_creation_and_resolution"


@dataclass(frozen=True, kw_only=True, slots=True)
class DeterministicCounts:
    """Record deterministic work and retention counters."""

    source_reads: int
    emitted_items: int
    projected_items: int
    reasoning_deltas: int
    harness_canonical_item_factory_calls: int
    harness_projection_observations: int
    harness_projection_state_factory_calls: int
    production_accumulator_instances: int
    production_accumulator_add_calls: int
    production_reasoning_text_property_reads: int
    retained_canonical_items: int
    retained_reasoning_characters: int
    retained_reasoning_utf8_bytes: int
    max_read_ahead: int

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            assert type(value) is int and value >= 0


@dataclass(frozen=True, kw_only=True, slots=True)
class BenchmarkFixture:
    """Store fixed source items and harness factory-call count."""

    items: tuple[CanonicalStreamItem, ...]
    harness_canonical_item_factory_calls: int

    def __post_init__(self) -> None:
        assert type(self.harness_canonical_item_factory_calls) is int
        assert self.harness_canonical_item_factory_calls >= 0


@dataclass(frozen=True, kw_only=True, slots=True)
class MemoryProbeResult:
    """Record isolated processing and retained-memory measurements."""

    peak_processing_bytes_excluding_source_fixture: int
    current_retained_bytes_including_source_fixture: int
    peak_total_bytes_including_source_fixture: int

    def __post_init__(self) -> None:
        assert type(self.peak_processing_bytes_excluding_source_fixture) is int
        assert (
            type(self.current_retained_bytes_including_source_fixture) is int
        )
        assert type(self.peak_total_bytes_including_source_fixture) is int
        assert self.peak_processing_bytes_excluding_source_fixture > 0
        assert self.current_retained_bytes_including_source_fixture > 0
        assert self.peak_total_bytes_including_source_fixture > 0
        assert (
            self.current_retained_bytes_including_source_fixture
            <= self.peak_total_bytes_including_source_fixture
        )


@dataclass(kw_only=True, slots=True)
class _BenchmarkObserver:
    """Count named harness work and spied production boundaries."""

    harness_canonical_item_factory_calls: int = 0
    harness_projection_observations: int = 0
    harness_projection_state_factory_calls: int = 0
    production_accumulator_instances: int = 0
    production_accumulator_add_calls: int = 0
    production_reasoning_text_property_reads: int = 0

    def create_canonical_item(
        self,
        sequence: int,
        kind: StreamItemKind,
        **kwargs: Any,
    ) -> CanonicalStreamItem:
        """Create and count one fixed canonical source item."""
        item = _item(sequence, kind, **kwargs)
        self.harness_canonical_item_factory_calls += 1
        return item

    def create_projection_state(self) -> StreamProjectionState:
        """Create one state with a production-boundary spy accumulator."""
        self.harness_projection_state_factory_calls += 1
        accumulator = _ObservedCanonicalStreamAccumulator(self)
        return StreamProjectionState(
            stream_session_id="reasoning-benchmark-stream",
            run_id="reasoning-benchmark-run",
            turn_id="reasoning-benchmark-turn",
            accumulator=accumulator,
            accumulate=True,
        )

    def record_projection(self, projection: StreamConsumerProjection) -> None:
        """Record one projection observed after production returned it."""
        assert isinstance(projection, StreamConsumerProjection)
        self.harness_projection_observations += 1

    def materialize_reasoning(
        self,
        accumulator: CanonicalStreamAccumulator,
    ) -> str:
        """Read the production compatibility reasoning-text property."""
        assert isinstance(accumulator, CanonicalStreamAccumulator)
        return accumulator.reasoning_text


class _ObservedCanonicalStreamAccumulator(CanonicalStreamAccumulator):
    """Spy actual production accumulator and materialization boundaries."""

    def __init__(self, observer: _BenchmarkObserver) -> None:
        assert isinstance(observer, _BenchmarkObserver)
        self._benchmark_observer = observer
        observer.production_accumulator_instances += 1
        super().__init__()

    @property
    def reasoning_text(self) -> str:
        self._benchmark_observer.production_reasoning_text_property_reads += 1
        return super().reasoning_text

    def add(self, item: CanonicalStreamItem) -> None:
        self._benchmark_observer.production_accumulator_add_calls += 1
        super().add(item)


@dataclass(frozen=True, kw_only=True, slots=True)
class WorkloadResult:
    """Record one fixed-delta workload result."""

    delta_count: int
    item_count: int
    sample_microseconds: tuple[float, ...]
    median_microseconds: float
    p95_microseconds: float
    median_per_item_microseconds: float
    p95_per_item_microseconds: float
    peak_processing_bytes_excluding_source_fixture: int
    current_retained_bytes_including_source_fixture: int
    peak_total_bytes_including_source_fixture: int
    deterministic: DeterministicCounts

    def __post_init__(self) -> None:
        for integer_value in (
            self.delta_count,
            self.item_count,
            self.peak_processing_bytes_excluding_source_fixture,
            self.current_retained_bytes_including_source_fixture,
            self.peak_total_bytes_including_source_fixture,
        ):
            assert type(integer_value) is int and integer_value > 0
        assert isinstance(self.sample_microseconds, tuple)
        assert self.sample_microseconds
        assert all(
            type(value) is float and isfinite(value) and value >= 0
            for value in self.sample_microseconds
        )
        for float_value in (
            self.median_microseconds,
            self.p95_microseconds,
            self.median_per_item_microseconds,
            self.p95_per_item_microseconds,
        ):
            assert (
                type(float_value) is float
                and isfinite(float_value)
                and float_value >= 0
            )
        assert isinstance(self.deterministic, DeterministicCounts)


@dataclass(frozen=True, kw_only=True, slots=True)
class Phase9WorkloadResult:
    """Record one bounded Phase 9 summary workload result."""

    name: str
    representation: str
    summary_part_count: int
    workload: WorkloadResult

    def __post_init__(self) -> None:
        assert self.name == _PHASE9_WORKLOAD_NAME
        assert (
            self.representation == StreamReasoningRepresentation.SUMMARY.value
        )
        assert type(self.summary_part_count) is int
        assert self.summary_part_count > 0
        assert isinstance(self.workload, WorkloadResult)


@dataclass(frozen=True, kw_only=True, slots=True)
class Phase9HardGateResult:
    """Record the exact Phase 9 hard-gate outcome."""

    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assert type(self.passed) is bool
        assert isinstance(self.failure_reasons, tuple)
        assert all(
            isinstance(reason, str) and reason
            for reason in self.failure_reasons
        )
        assert self.passed is not bool(self.failure_reasons)


class _Phase9ResponsesSummaryStream:
    """Yield a fixed summary stream through the real Responses route."""

    input_token_count = 0
    output_token_count = 0
    usage = None

    def __init__(self) -> None:
        self._items = iter(_phase9_responses_coalescing_fixture())
        self.close_count = 0

    def __aiter__(self) -> "_Phase9ResponsesSummaryStream":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        try:
            return next(self._items)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.close_count += 1


def contract_path() -> Path:
    """Return the machine-readable Phase 0 contract path."""
    return (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "reasoning_summary"
        / "phase0_contract.json"
    )


def benchmark_protocol() -> BenchmarkProtocol:
    """Load the locked benchmark protocol from its single source."""
    payload = strict_json_loads(contract_path().read_text(encoding="utf-8"))
    return _benchmark_protocol_from_payload(payload)


def _benchmark_protocol_from_payload(payload: object) -> BenchmarkProtocol:
    """Validate a contract payload and return its benchmark protocol."""
    assert isinstance(payload, dict)
    schema_version = payload.get("schema_version")
    assert type(schema_version) is int and schema_version == 1
    raw = payload.get("benchmark")
    assert isinstance(raw, dict)
    assert set(raw) == {
        "runner",
        "warmups",
        "samples",
        "delta_counts",
        "delta_text",
        "percentile_method",
        "gc_before_each_sample",
        "reset_state_before_each_sample",
        "gc_before_memory_probe",
        "elapsed_clock",
        "peak_memory_probe",
        "memory_probe_scopes",
        "memory_subprocess_timeout_seconds",
        "network_allowed",
        "network_guard",
    }
    assert raw.get("runner") == "scripts/benchmark_reasoning_summary.py"
    delta_counts = raw.get("delta_counts")
    memory_probe_scopes = raw.get("memory_probe_scopes")
    assert isinstance(delta_counts, list)
    assert isinstance(memory_probe_scopes, list)
    warmups = raw.get("warmups")
    samples = raw.get("samples")
    memory_timeout = raw.get("memory_subprocess_timeout_seconds")
    assert type(warmups) is int
    assert type(samples) is int
    assert type(memory_timeout) is int
    assert all(type(count) is int for count in delta_counts)
    for bool_field in (
        "gc_before_each_sample",
        "reset_state_before_each_sample",
        "gc_before_memory_probe",
        "network_allowed",
    ):
        assert type(raw.get(bool_field)) is bool
    return BenchmarkProtocol(
        warmups=warmups,
        samples=samples,
        delta_counts=tuple(cast(list[int], delta_counts)),
        delta_text=cast(str, raw.get("delta_text")),
        percentile_method=cast(str, raw.get("percentile_method")),
        gc_before_each_sample=cast(bool, raw.get("gc_before_each_sample")),
        reset_state_before_each_sample=cast(
            bool, raw.get("reset_state_before_each_sample")
        ),
        gc_before_memory_probe=cast(bool, raw.get("gc_before_memory_probe")),
        elapsed_clock=cast(str, raw.get("elapsed_clock")),
        peak_memory_probe=cast(str, raw.get("peak_memory_probe")),
        memory_probe_scopes=tuple(cast(list[str], memory_probe_scopes)),
        memory_subprocess_timeout_seconds=memory_timeout,
        network_allowed=cast(bool, raw.get("network_allowed")),
        network_guard=cast(str, raw.get("network_guard")),
    )


@contextmanager
def _network_denied() -> Iterator[None]:
    """Deny socket creation and resolution during benchmark work."""
    message = "network access is prohibited by the benchmark contract"
    with (
        patch("socket.create_connection", side_effect=AssertionError(message)),
        patch("socket.create_server", side_effect=AssertionError(message)),
        patch("socket.getaddrinfo", side_effect=AssertionError(message)),
        patch("socket.socket", side_effect=AssertionError(message)),
    ):
        yield


def run_benchmark() -> dict[str, object]:
    """Run all fixed workloads and return a machine-readable report."""
    protocol = benchmark_protocol()
    with _network_denied():
        workloads = tuple(
            _run_workload(delta_count, protocol)
            for delta_count in protocol.delta_counts
        )
    budget = StreamPerformanceBudget()
    return {
        "schema_version": 1,
        "suite": "reasoning-summary-canonical-stream",
        "generated_at": datetime.now(UTC).isoformat(),
        "python": version,
        "python_implementation": python_implementation(),
        "platform": platform(),
        "machine": machine(),
        "processor": processor(),
        "protocol": asdict(protocol),
        "stream_performance_budget": asdict(budget),
        "workloads": [
            {
                **asdict(workload),
                "sample_microseconds": list(workload.sample_microseconds),
            }
            for workload in workloads
        ],
    }


def run_phase9_benchmark() -> dict[str, object]:
    """Run bounded Phase 9 summary workloads and return their report."""
    protocol = benchmark_protocol()
    with _network_denied():
        workloads = _run_phase9_workloads(protocol)
    asynchronous_metrics = _isolated_phase9_asynchronous_metrics(protocol)
    budget = StreamPerformanceBudget()
    report: dict[str, object] = {
        "schema_version": 1,
        "suite": "reasoning-summary-phase9-performance",
        "generated_at": datetime.now(UTC).isoformat(),
        "python": version,
        "python_implementation": python_implementation(),
        "platform": platform(),
        "machine": machine(),
        "processor": processor(),
        "protocol": asdict(protocol),
        "stream_performance_budget": asdict(budget),
        "phase9_budgets": {
            "p95_per_item_microseconds": (_PHASE9_P95_PER_ITEM_MICROSECONDS),
            "batch_budget_multiplier": _PHASE9_BATCH_BUDGET_MULTIPLIER,
            "work_ratio_8192_over_4096": _PHASE9_WORK_RATIO_LIMIT,
            "heartbeat_interval_milliseconds": (
                _PHASE9_HEARTBEAT_INTERVAL_MILLISECONDS
            ),
            "heartbeat_maximum_drift_milliseconds": (
                _PHASE9_HEARTBEAT_MAXIMUM_DRIFT_MILLISECONDS
            ),
            "maximum_coalesced_delta_characters": (
                _PHASE9_MAXIMUM_COALESCED_DELTA_CHARACTERS
            ),
        },
        "phase9_metrics": asynchronous_metrics,
        "workloads": [
            {
                "name": workload.name,
                "representation": workload.representation,
                "summary_part_count": workload.summary_part_count,
                **asdict(workload.workload),
                "sample_microseconds": list(
                    workload.workload.sample_microseconds
                ),
            }
            for workload in workloads
        ],
    }
    report["hard_gate"] = asdict(evaluate_phase9_hard_gate(report))
    return report


def evaluate_phase9_hard_gate(
    report: dict[str, object],
) -> Phase9HardGateResult:
    """Evaluate every Phase 9 hard metric and return exact failures."""
    failures: list[str] = []
    protocol = cast(dict[str, object], report["protocol"])
    if protocol.get("warmups") != 3:
        failures.append("protocol warmups must equal 3")
    if protocol.get("samples") != 20:
        failures.append("protocol samples must equal 20")
    if protocol.get("delta_counts") not in ([4096, 8192], (4096, 8192)):
        failures.append("protocol delta counts must equal 4096 and 8192")

    stream_budget = cast(
        dict[str, object], report["stream_performance_budget"]
    )
    median_per_item_limit = cast(int, stream_budget["per_item_overhead_us"])
    memory_limit = cast(int, stream_budget["max_memory_bytes"])
    first_item_limit = cast(int, stream_budget["time_to_first_item_ms"])
    workloads = cast(list[dict[str, object]], report["workloads"])
    workload_by_count = {
        cast(int, workload["delta_count"]): workload for workload in workloads
    }
    if set(workload_by_count) != {4096, 8192}:
        failures.append("workloads must contain exactly 4096 and 8192 deltas")
    else:
        for delta_count in (4096, 8192):
            workload = workload_by_count[delta_count]
            prefix = f"workload {delta_count}"
            if workload.get("name") != _PHASE9_WORKLOAD_NAME:
                failures.append(
                    f"{prefix} name must identify the Phase 9 path"
                )
            if workload.get("representation") != "summary":
                failures.append(f"{prefix} representation must be summary")
            if workload.get("summary_part_count") != (
                _phase9_summary_part_count(delta_count)
            ):
                failures.append(f"{prefix} summary part count changed")
            samples = cast(list[object], workload["sample_microseconds"])
            if len(samples) != 20:
                failures.append(f"{prefix} must contain 20 measured samples")
            if (
                cast(float, workload["median_per_item_microseconds"])
                > median_per_item_limit
            ):
                failures.append(
                    f"{prefix} median per-item time exceeded "
                    f"{median_per_item_limit} microseconds"
                )
            if (
                cast(float, workload["p95_per_item_microseconds"])
                > _PHASE9_P95_PER_ITEM_MICROSECONDS
            ):
                failures.append(
                    f"{prefix} p95 per-item time exceeded "
                    f"{_PHASE9_P95_PER_ITEM_MICROSECONDS} microseconds"
                )
            batch_limit = (
                _PHASE9_BATCH_BUDGET_MULTIPLIER
                * delta_count
                * median_per_item_limit
            )
            if cast(float, workload["p95_microseconds"]) > batch_limit:
                failures.append(f"{prefix} p95 batch time exceeded budget")
            for metric_name in (
                "peak_processing_bytes_excluding_source_fixture",
                "current_retained_bytes_including_source_fixture",
                "peak_total_bytes_including_source_fixture",
            ):
                if cast(int, workload[metric_name]) > memory_limit:
                    failures.append(
                        f"{prefix} {metric_name} exceeded {memory_limit} bytes"
                    )
            deterministic = cast(dict[str, object], workload["deterministic"])
            if deterministic.get("max_read_ahead") != 0:
                failures.append(f"{prefix} read-ahead must equal zero")
            if (
                deterministic.get("production_reasoning_text_property_reads")
                != 0
            ):
                failures.append(
                    f"{prefix} hidden reasoning materialization must equal "
                    "zero"
                )

        smaller = workload_by_count[4096]
        larger = workload_by_count[8192]
        median_ratio = cast(float, larger["median_microseconds"]) / cast(
            float, smaller["median_microseconds"]
        )
        if median_ratio > _PHASE9_WORK_RATIO_LIMIT:
            failures.append("median 8192-to-4096 work ratio exceeded 2.5")
        p95_ratio = cast(float, larger["p95_microseconds"]) / cast(
            float, smaller["p95_microseconds"]
        )
        if p95_ratio > _PHASE9_WORK_RATIO_LIMIT:
            failures.append("p95 8192-to-4096 work ratio exceeded 2.5")

    metrics = cast(dict[str, object], report["phase9_metrics"])
    heartbeat = cast(dict[str, object], metrics["heartbeat"])
    for field_name, expected in (
        ("delta_count", 8192),
        ("warmups", 3),
        ("samples", 20),
        ("interval_milliseconds", 10),
    ):
        if heartbeat.get(field_name) != expected:
            failures.append(f"heartbeat {field_name} must equal {expected}")
    if cast(int, heartbeat["tick_count"]) < 23:
        failures.append("heartbeat must observe at least 23 ticks")
    if (
        cast(float, heartbeat["maximum_drift_milliseconds"])
        > _PHASE9_HEARTBEAT_MAXIMUM_DRIFT_MILLISECONDS
    ):
        failures.append("heartbeat maximum drift exceeded 100 milliseconds")

    coalescing = cast(dict[str, object], metrics["responses_coalescing"])
    if coalescing.get("source_delta_count") != 4097:
        failures.append("Responses coalescing source must contain 4097 deltas")
    if cast(int, coalescing["summary_delta_event_count"]) < 2:
        failures.append("Responses coalescing must emit at least two deltas")
    maximum_delta_characters = cast(
        int, coalescing["maximum_delta_characters"]
    )
    if (
        not 0
        < maximum_delta_characters
        <= (_PHASE9_MAXIMUM_COALESCED_DELTA_CHARACTERS)
    ):
        failures.append("Responses coalesced delta exceeded 4096 characters")
    if (
        cast(float, coalescing["first_summary_milliseconds"])
        > first_item_limit
    ):
        failures.append(
            f"Responses first summary exceeded {first_item_limit} milliseconds"
        )
    if coalescing.get("source_close_count") != 1:
        failures.append("Responses coalescing source must close exactly once")

    queue_pressure = cast(dict[str, object], metrics["queue_pressure"])
    expected_queues = {
        "event_lossless_block": ("block", 1),
        "event_critical_block": ("block", 1),
        "event_ui_coalesce": ("coalesce", 64),
        "event_observability_drop": ("drop", 32),
        "event_fail_closed": ("fail_closed", 1),
        "sdk_listen_drop": ("drop", 512),
        "cli_listen_coalesce": ("coalesce", 256),
        "test_listen_drop": ("drop", 1024),
        "local_handoff_block": ("block", 64),
        "orchestrator_staging_fail_closed": ("fail_closed", 4096),
    }
    for name, (expected_policy, expected_limit) in expected_queues.items():
        evidence = cast(dict[str, object], queue_pressure[name])
        if evidence.get("policy") != expected_policy:
            failures.append(f"queue policy {name} changed")
        if evidence.get("queue_limit") != expected_limit:
            failures.append(f"queue limit {name} changed")
        if evidence.get("passed") is not True:
            failures.append(f"queue pressure {name} failed")
    server_listen = cast(
        dict[str, object], queue_pressure["server_listen_disabled"]
    )
    if server_listen.get("enabled") is not False:
        failures.append("server listen queue must remain disabled")
    if server_listen.get("passed") is not True:
        failures.append("queue pressure server_listen_disabled failed")
    return Phase9HardGateResult(
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _run_workload(
    delta_count: int,
    protocol: BenchmarkProtocol,
) -> WorkloadResult:
    fixture = _fixture_items(delta_count, protocol.delta_text)
    for _ in range(protocol.warmups):
        collect()
        _project_fixture(fixture)
    samples: list[float] = []
    deterministic: DeterministicCounts | None = None
    for _ in range(protocol.samples):
        collect()
        started = perf_counter()
        deterministic = _project_fixture(fixture)
        samples.append((perf_counter() - started) * 1_000_000)
    collect()
    memory = _isolated_memory_probe(delta_count, protocol.delta_text, protocol)
    assert deterministic is not None
    median_us = median(samples)
    p95_us = _nearest_rank(samples, 95)
    return WorkloadResult(
        delta_count=delta_count,
        item_count=len(fixture.items),
        sample_microseconds=tuple(samples),
        median_microseconds=median_us,
        p95_microseconds=p95_us,
        median_per_item_microseconds=median_us / len(fixture.items),
        p95_per_item_microseconds=p95_us / len(fixture.items),
        peak_processing_bytes_excluding_source_fixture=(
            memory.peak_processing_bytes_excluding_source_fixture
        ),
        current_retained_bytes_including_source_fixture=(
            memory.current_retained_bytes_including_source_fixture
        ),
        peak_total_bytes_including_source_fixture=(
            memory.peak_total_bytes_including_source_fixture
        ),
        deterministic=deterministic,
    )


def _run_phase9_workloads(
    protocol: BenchmarkProtocol,
) -> tuple[Phase9WorkloadResult, ...]:
    fixtures = {
        delta_count: _phase9_fixture_items(delta_count, protocol.delta_text)
        for delta_count in protocol.delta_counts
    }
    for warmup_index in range(protocol.warmups):
        for delta_count in _phase9_interleaved_order(
            protocol.delta_counts,
            warmup_index,
        ):
            collect()
            _project_phase9_fixture(fixtures[delta_count])

    samples_by_count: dict[int, list[float]] = {
        delta_count: [] for delta_count in protocol.delta_counts
    }
    deterministic_by_count: dict[int, DeterministicCounts] = {}
    for sample_index in range(protocol.samples):
        for delta_count in _phase9_interleaved_order(
            protocol.delta_counts,
            sample_index,
        ):
            collect()
            started = perf_counter()
            deterministic_by_count[delta_count] = _project_phase9_fixture(
                fixtures[delta_count]
            )
            samples_by_count[delta_count].append(
                (perf_counter() - started) * 1_000_000
            )

    results: list[Phase9WorkloadResult] = []
    for delta_count in protocol.delta_counts:
        collect()
        memory = _isolated_phase9_memory_probe(
            delta_count,
            protocol.delta_text,
            protocol,
        )
        fixture = fixtures[delta_count]
        samples = samples_by_count[delta_count]
        deterministic = deterministic_by_count[delta_count]
        median_us = median(samples)
        p95_us = _nearest_rank(samples, 95)
        workload = WorkloadResult(
            delta_count=delta_count,
            item_count=len(fixture.items),
            sample_microseconds=tuple(samples),
            median_microseconds=median_us,
            p95_microseconds=p95_us,
            median_per_item_microseconds=median_us / len(fixture.items),
            p95_per_item_microseconds=p95_us / len(fixture.items),
            peak_processing_bytes_excluding_source_fixture=(
                memory.peak_processing_bytes_excluding_source_fixture
            ),
            current_retained_bytes_including_source_fixture=(
                memory.current_retained_bytes_including_source_fixture
            ),
            peak_total_bytes_including_source_fixture=(
                memory.peak_total_bytes_including_source_fixture
            ),
            deterministic=deterministic,
        )
        results.append(
            Phase9WorkloadResult(
                name=_PHASE9_WORKLOAD_NAME,
                representation=StreamReasoningRepresentation.SUMMARY.value,
                summary_part_count=_phase9_summary_part_count(delta_count),
                workload=workload,
            )
        )
    return tuple(results)


def _phase9_interleaved_order(
    delta_counts: tuple[int, ...],
    iteration: int,
) -> tuple[int, ...]:
    assert delta_counts
    assert type(iteration) is int and iteration >= 0
    return (
        delta_counts if iteration % 2 == 0 else tuple(reversed(delta_counts))
    )


async def _phase9_asynchronous_metrics(
    protocol: BenchmarkProtocol,
) -> dict[str, object]:
    with _network_denied():
        return {
            "heartbeat": await _phase9_heartbeat_metrics(protocol),
            "responses_coalescing": (
                await _phase9_responses_coalescing_metrics()
            ),
            "queue_pressure": await _phase9_queue_pressure_metrics(),
        }


def _isolated_phase9_asynchronous_metrics(
    protocol: BenchmarkProtocol,
) -> dict[str, object]:
    environment = {
        key: value
        for key, value in environ.items()
        if not key.startswith(("COVERAGE_", "COV_CORE_", "PYTEST_"))
    }
    completed = run(
        [
            executable,
            str(Path(__file__).resolve()),
            "--phase9-async-probe",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=protocol.memory_subprocess_timeout_seconds,
    )
    assert type(completed.returncode) is int and completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    payloads = [
        line.removeprefix(_PHASE9_ASYNC_SENTINEL)
        for line in completed.stdout.splitlines()
        if line.startswith(_PHASE9_ASYNC_SENTINEL)
    ]
    assert len(payloads) == 1, completed.stdout + completed.stderr
    payload = strict_json_loads(payloads[0])
    assert isinstance(payload, dict)
    assert set(payload) == {
        "heartbeat",
        "responses_coalescing",
        "queue_pressure",
    }
    return cast(dict[str, object], payload)


async def _phase9_heartbeat_metrics(
    protocol: BenchmarkProtocol,
) -> dict[str, object]:
    delta_count = protocol.delta_counts[-1]
    fixture = _phase9_fixture_items(delta_count, protocol.delta_text)
    interval_seconds = _PHASE9_HEARTBEAT_INTERVAL_MILLISECONDS / 1000
    started = AsyncEvent()
    stopped = AsyncEvent()
    maximum_drift_seconds = 0.0
    tick_count = 0

    async def heartbeat() -> None:
        nonlocal maximum_drift_seconds, tick_count
        previous = perf_counter()
        started.set()
        while not stopped.is_set():
            await sleep(interval_seconds)
            current = perf_counter()
            maximum_drift_seconds = max(
                maximum_drift_seconds,
                current - previous - interval_seconds,
            )
            previous = current
            tick_count += 1

    collect()
    heartbeat_task = create_task(heartbeat())
    await started.wait()
    for _ in range(protocol.warmups):
        await _project_phase9_fixture_incrementally(fixture)
        await sleep(interval_seconds)
    for _ in range(protocol.samples):
        await _project_phase9_fixture_incrementally(fixture)
        await sleep(interval_seconds)
    stopped.set()
    await heartbeat_task
    return {
        "delta_count": delta_count,
        "warmups": protocol.warmups,
        "samples": protocol.samples,
        "interval_milliseconds": _PHASE9_HEARTBEAT_INTERVAL_MILLISECONDS,
        "maximum_drift_milliseconds": maximum_drift_seconds * 1000,
        "tick_count": tick_count,
    }


async def _phase9_responses_coalescing_metrics() -> dict[str, object]:
    stream = _Phase9ResponsesSummaryStream()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.sync_messages = AsyncMock()
    request = ResponsesRequest(
        model="phase9-model",
        input=[ChatMessage(role=MessageRole.USER, content="phase9")],
        stream=True,
    )

    async def orchestrate_stub(
        request_value: ResponsesRequest,
        logger_value: object,
        orchestrator_value: Orchestrator,
    ) -> tuple[_Phase9ResponsesSummaryStream, object, int]:
        assert request_value is request
        assert logger_value is not None
        assert orchestrator_value is orchestrator
        return stream, uuid4(), 0

    first_summary_milliseconds: float | None = None
    maximum_delta_characters = 0
    summary_delta_event_count = 0
    started_at = perf_counter()
    with (
        patch.object(responses_router, "orchestrate", orchestrate_stub),
        patch.object(
            responses_router,
            "resolve_model_id",
            return_value="phase9-model",
        ),
    ):
        response = await responses_router.create_response(
            request,
            getLogger(),
            orchestrator,
            ServerOutputRedactionSettings(),
        )
        async for chunk in cast(AsyncIterator[object], response.body_iterator):
            text = (
                chunk.decode()
                if isinstance(chunk, bytes)
                else cast(str, chunk)
            )
            for payload in _phase9_sse_payloads(text):
                if payload.get("type") != (
                    "response.reasoning_summary_text.delta"
                ):
                    continue
                delta = payload.get("delta")
                assert isinstance(delta, str)
                summary_delta_event_count += 1
                maximum_delta_characters = max(
                    maximum_delta_characters,
                    len(delta),
                )
                if first_summary_milliseconds is None:
                    first_summary_milliseconds = (
                        perf_counter() - started_at
                    ) * 1000
    assert first_summary_milliseconds is not None
    return {
        "source_delta_count": _PHASE9_COALESCING_SOURCE_DELTAS,
        "summary_delta_event_count": summary_delta_event_count,
        "maximum_delta_characters": maximum_delta_characters,
        "first_summary_milliseconds": first_summary_milliseconds,
        "source_close_count": stream.close_count,
    }


def _phase9_sse_payloads(text: str) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = strict_json_loads(line.removeprefix("data: "))
        assert isinstance(payload, dict)
        payloads.append(cast(dict[str, Any], payload))
    return tuple(payloads)


async def _phase9_queue_pressure_metrics() -> dict[str, object]:
    evidence = {
        "event_lossless_block": await _phase9_event_block_evidence(
            EventSubscriberClass.LOSSLESS
        ),
        "event_critical_block": await _phase9_event_block_evidence(
            EventSubscriberClass.CRITICAL
        ),
        "event_ui_coalesce": await _phase9_event_enqueue_evidence(
            EventSubscriberClass.UI
        ),
        "event_observability_drop": await _phase9_event_enqueue_evidence(
            EventSubscriberClass.OBSERVABILITY
        ),
        "event_fail_closed": await _phase9_event_fail_closed_evidence(),
        "sdk_listen_drop": await _phase9_listen_queue_evidence(
            EventManagerMode.SDK
        ),
        "cli_listen_coalesce": await _phase9_listen_queue_evidence(
            EventManagerMode.CLI
        ),
        "test_listen_drop": await _phase9_listen_queue_evidence(
            EventManagerMode.TEST
        ),
        "server_listen_disabled": _phase9_server_listen_evidence(),
        "local_handoff_block": await _phase9_local_handoff_evidence(),
        "orchestrator_staging_fail_closed": (
            _phase9_orchestrator_staging_evidence()
        ),
    }
    return evidence


async def _phase9_event_block_evidence(
    subscriber_class: EventSubscriberClass,
) -> dict[str, object]:
    manager = EventManager(
        history_config=EventHistoryConfig(enabled=False),
        listen_config=EventListenConfig(enabled=False),
    )
    started = AsyncEvent()
    release = AsyncEvent()

    async def listener(event: Event) -> None:
        assert event.type is EventType.START
        started.set()
        await release.wait()

    manager.add_listener(
        listener,
        [EventType.START],
        subscriber_class=subscriber_class,
    )
    trigger_task = create_task(manager.trigger(Event(type=EventType.START)))
    await wait_for(started.wait(), timeout=1)
    blocked = not trigger_task.done()
    release.set()
    await wait_for(trigger_task, timeout=1)
    config = EventManager.default_delivery_config_for_subscriber_class(
        subscriber_class
    )
    await manager.aclose()
    return {
        "policy": config.policy.value,
        "queue_limit": config.queue_limit,
        "blocked": blocked,
        "passed": config.policy is EventDeliveryPolicy.BLOCK and blocked,
    }


async def _phase9_event_enqueue_evidence(
    subscriber_class: EventSubscriberClass,
) -> dict[str, object]:
    manager = EventManager(
        history_config=EventHistoryConfig(enabled=False),
        listen_config=EventListenConfig(enabled=False),
    )

    async def listener(event: Event) -> None:
        assert event.type is EventType.START

    manager.add_listener(
        listener,
        [EventType.START],
        subscriber_class=subscriber_class,
    )
    config = EventManager.default_delivery_config_for_subscriber_class(
        subscriber_class
    )
    for sequence in range(config.queue_limit + 1):
        await manager.trigger(
            Event(type=EventType.START, payload={"sequence": sequence})
        )
    stats = manager.stats
    passed = (
        stats.coalesced > 0
        if config.policy is EventDeliveryPolicy.COALESCE
        else stats.dropped > 0
    )
    result = {
        "policy": config.policy.value,
        "queue_limit": config.queue_limit,
        "maximum_queue_depth": stats.max_queue_depth,
        "coalesced": stats.coalesced,
        "dropped": stats.dropped,
        "passed": passed and stats.max_queue_depth <= config.queue_limit,
    }
    await manager.aclose()
    return result


async def _phase9_event_fail_closed_evidence() -> dict[str, object]:
    manager = EventManager(
        history_config=EventHistoryConfig(enabled=False),
        listen_config=EventListenConfig(enabled=False),
    )

    async def listener(event: Event) -> None:
        assert event.type is EventType.TOKEN_GENERATED

    config = EventDeliveryConfig(
        policy=EventDeliveryPolicy.FAIL_CLOSED,
        queue_limit=1,
    )
    manager.add_listener(
        listener,
        [EventType.TOKEN_GENERATED],
        delivery_config=config,
        include_token_events=True,
    )
    await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
    await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
    stats = manager.stats
    subscriber_closed = not manager.should_emit(EventType.TOKEN_GENERATED)
    result = {
        "policy": config.policy.value,
        "queue_limit": config.queue_limit,
        "failed": stats.failed,
        "subscriber_closed": subscriber_closed,
        "passed": stats.failed == 1 and subscriber_closed,
    }
    await manager.aclose()
    return result


async def _phase9_listen_queue_evidence(
    mode: EventManagerMode,
) -> dict[str, object]:
    manager = EventManager(
        mode=mode,
        history_config=EventHistoryConfig(enabled=False),
    )
    config = manager.listen_config
    for sequence in range(config.queue_limit + 1):
        await manager.trigger(
            Event(type=EventType.START, payload={"sequence": sequence})
        )
    stats = manager.stats
    pressure_count = (
        stats.coalesced
        if config.policy is EventDeliveryPolicy.COALESCE
        else stats.dropped
    )
    result = {
        "policy": config.policy.value,
        "queue_limit": config.queue_limit,
        "maximum_queue_depth": stats.max_queue_depth,
        "pressure_count": pressure_count,
        "passed": pressure_count > 0
        and stats.max_queue_depth <= config.queue_limit,
    }
    await manager.aclose()
    return result


def _phase9_server_listen_evidence() -> dict[str, object]:
    defaults = EventManager.defaults_for_mode(EventManagerMode.SERVER)
    return {
        "enabled": defaults.listen_config.enabled,
        "passed": not defaults.listen_config.enabled,
    }


async def _phase9_local_handoff_evidence() -> dict[str, object]:
    loop = get_running_loop()
    stop_event = ThreadEvent()
    streamer = SimpleNamespace(
        text_queue=Queue(),
        loop=loop,
        stop_signal=object(),
    )
    queue_limit = _configure_lossless_streamer_handoff(
        streamer,
        stop_event,
    )
    assert queue_limit is not None

    def produce() -> None:
        for sequence in range(queue_limit + 1):
            streamer.on_finalized_text(str(sequence))

    producer = Thread(target=produce, daemon=True)
    producer.start()
    for _ in range(1000):
        if streamer.text_queue.qsize() == queue_limit:
            break
        await sleep(0.001)
    blocked = (
        producer.is_alive() and streamer.text_queue.qsize() == queue_limit
    )
    _ = streamer.text_queue.get_nowait()
    await to_thread(producer.join, 1)
    stop_event.set()
    producer_completed = not producer.is_alive()
    return {
        "policy": EventDeliveryPolicy.BLOCK.value,
        "queue_limit": queue_limit,
        "blocked_at_capacity": blocked,
        "producer_completed_after_consume": producer_completed,
        "passed": blocked and producer_completed,
    }


def _phase9_orchestrator_staging_evidence() -> dict[str, object]:
    queue = OrchestratorResponse._make_staging_queue()
    for sequence in range(OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS):
        OrchestratorResponse._put_staging_item(queue, sequence, "phase9")
    overflow_failed_closed = False
    try:
        OrchestratorResponse._put_staging_item(queue, object(), "phase9")
    except RuntimeError:
        overflow_failed_closed = True
    return {
        "policy": EventDeliveryPolicy.FAIL_CLOSED.value,
        "queue_limit": queue.maxsize,
        "overflow_failed_closed": overflow_failed_closed,
        "passed": overflow_failed_closed
        and queue.maxsize == OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
    }


def _fixture_items(
    delta_count: int,
    delta_text: str,
) -> BenchmarkFixture:
    assert type(delta_count) is int and delta_count > 0
    assert isinstance(delta_text, str) and delta_text
    observer = _BenchmarkObserver()

    def allocate(
        sequence: int,
        kind: StreamItemKind,
        **kwargs: Any,
    ) -> CanonicalStreamItem:
        return observer.create_canonical_item(sequence, kind, **kwargs)

    items = [allocate(0, StreamItemKind.STREAM_STARTED)]
    for sequence in range(1, delta_count + 1):
        items.append(
            allocate(
                sequence,
                StreamItemKind.REASONING_DELTA,
                text_delta=delta_text,
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            )
        )
    items.extend(
        (
            allocate(delta_count + 1, StreamItemKind.REASONING_DONE),
            allocate(
                delta_count + 2,
                StreamItemKind.STREAM_COMPLETED,
                usage={"output_tokens": delta_count},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            allocate(delta_count + 3, StreamItemKind.STREAM_CLOSED),
        )
    )
    return BenchmarkFixture(
        items=tuple(items),
        harness_canonical_item_factory_calls=(
            observer.harness_canonical_item_factory_calls
        ),
    )


def _phase9_fixture_items(
    delta_count: int,
    delta_text: str,
) -> BenchmarkFixture:
    assert type(delta_count) is int and delta_count > 0
    assert isinstance(delta_text, str) and delta_text
    observer = _BenchmarkObserver()

    def allocate(
        sequence: int,
        kind: StreamItemKind,
        **kwargs: Any,
    ) -> CanonicalStreamItem:
        return observer.create_canonical_item(sequence, kind, **kwargs)

    items = [allocate(0, StreamItemKind.STREAM_STARTED)]
    for sequence in range(1, delta_count + 1):
        part_ordinal = (sequence - 1) // _PHASE9_SUMMARY_DELTAS_PER_PART
        metadata = (
            {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
            if sequence > 1
            and (sequence - 1) % _PHASE9_SUMMARY_DELTAS_PER_PART == 0
            else None
        )
        items.append(
            allocate(
                sequence,
                StreamItemKind.REASONING_DELTA,
                text_delta=delta_text,
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.SUMMARY
                ),
                segment_instance_ordinal=part_ordinal,
                metadata=metadata,
            )
        )
    items.extend(
        (
            allocate(delta_count + 1, StreamItemKind.REASONING_DONE),
            allocate(
                delta_count + 2,
                StreamItemKind.STREAM_COMPLETED,
                usage={"output_tokens": delta_count},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            allocate(delta_count + 3, StreamItemKind.STREAM_CLOSED),
        )
    )
    return BenchmarkFixture(
        items=tuple(items),
        harness_canonical_item_factory_calls=(
            observer.harness_canonical_item_factory_calls
        ),
    )


def _phase9_responses_coalescing_fixture() -> tuple[CanonicalStreamItem, ...]:
    delta_count = _PHASE9_COALESCING_SOURCE_DELTAS
    items = [_item(0, StreamItemKind.STREAM_STARTED)]
    for sequence in range(1, delta_count + 1):
        items.append(
            _item(
                sequence,
                StreamItemKind.REASONING_DELTA,
                text_delta="x",
                visibility=StreamVisibility.PRIVATE,
                reasoning_representation=(
                    StreamReasoningRepresentation.SUMMARY
                ),
                segment_instance_ordinal=0,
            )
        )
    items.extend(
        (
            _item(delta_count + 1, StreamItemKind.REASONING_DONE),
            _item(
                delta_count + 2,
                StreamItemKind.STREAM_COMPLETED,
                usage={"output_tokens": delta_count},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            _item(delta_count + 3, StreamItemKind.STREAM_CLOSED),
        )
    )
    return tuple(items)


def _item(
    sequence: int,
    kind: StreamItemKind,
    *,
    text_delta: str | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    reasoning_representation: StreamReasoningRepresentation | None = None,
    segment_instance_ordinal: int | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalStreamItem:
    channel = (
        StreamChannel.REASONING
        if kind
        in {StreamItemKind.REASONING_DELTA, StreamItemKind.REASONING_DONE}
        else StreamChannel.CONTROL
    )
    return CanonicalStreamItem(
        stream_session_id="reasoning-benchmark-stream",
        run_id="reasoning-benchmark-run",
        turn_id="reasoning-benchmark-turn",
        sequence=sequence,
        kind=kind,
        channel=channel,
        text_delta=text_delta,
        visibility=visibility,
        usage=cast(Any, usage),
        terminal_outcome=terminal_outcome,
        reasoning_representation=reasoning_representation,
        segment_instance_ordinal=segment_instance_ordinal,
        metadata=cast(Any, metadata or {}),
    )


def expected_deterministic_counts(delta_count: int) -> DeterministicCounts:
    """Return exact deterministic counts for one fixed workload."""
    assert type(delta_count) is int and delta_count > 0
    item_count = delta_count + 4
    return DeterministicCounts(
        source_reads=item_count,
        emitted_items=item_count,
        projected_items=item_count,
        reasoning_deltas=delta_count,
        harness_canonical_item_factory_calls=item_count,
        harness_projection_observations=item_count,
        harness_projection_state_factory_calls=1,
        production_accumulator_instances=1,
        production_accumulator_add_calls=item_count,
        production_reasoning_text_property_reads=1,
        retained_canonical_items=min(item_count, 4096),
        retained_reasoning_characters=delta_count,
        retained_reasoning_utf8_bytes=delta_count,
        max_read_ahead=0,
    )


def expected_phase9_deterministic_counts(
    delta_count: int,
) -> DeterministicCounts:
    """Return exact work counts for one hidden summary workload."""
    assert type(delta_count) is int and delta_count > 0
    item_count = delta_count + 4
    retained_characters = delta_count + 2 * (
        _phase9_summary_part_count(delta_count) - 1
    )
    return DeterministicCounts(
        source_reads=item_count,
        emitted_items=item_count,
        projected_items=item_count,
        reasoning_deltas=delta_count,
        harness_canonical_item_factory_calls=item_count,
        harness_projection_observations=item_count,
        harness_projection_state_factory_calls=1,
        production_accumulator_instances=1,
        production_accumulator_add_calls=item_count,
        production_reasoning_text_property_reads=0,
        retained_canonical_items=min(item_count, 4096),
        retained_reasoning_characters=retained_characters,
        retained_reasoning_utf8_bytes=retained_characters,
        max_read_ahead=0,
    )


def _phase9_summary_part_count(delta_count: int) -> int:
    assert type(delta_count) is int and delta_count > 0
    return ceil(delta_count / _PHASE9_SUMMARY_DELTAS_PER_PART)


def _project_fixture(fixture: BenchmarkFixture) -> DeterministicCounts:
    deterministic, _ = _project_fixture_retained(fixture)
    return deterministic


def _project_fixture_retained(
    fixture: BenchmarkFixture,
) -> tuple[DeterministicCounts, StreamProjectionState]:
    observer = _BenchmarkObserver(
        harness_canonical_item_factory_calls=(
            fixture.harness_canonical_item_factory_calls
        )
    )
    state = observer.create_projection_state()
    reasoning_deltas = 0
    projected_items = 0
    source = _TrackedFixtureSource(fixture.items)
    for item in source:
        projection = state.project(
            item,
            item.sequence,
            unsupported_message="unsupported reasoning benchmark item",
        )
        projected_items += 1
        observer.record_projection(projection)
        if item.kind is StreamItemKind.REASONING_DELTA:
            reasoning_deltas += 1
        source.mark_consumed()
    state.validate_complete()
    reasoning_text = observer.materialize_reasoning(state.accumulator)
    deterministic = DeterministicCounts(
        source_reads=source.read_count,
        emitted_items=source.consumed_count,
        projected_items=projected_items,
        reasoning_deltas=reasoning_deltas,
        harness_canonical_item_factory_calls=(
            observer.harness_canonical_item_factory_calls
        ),
        harness_projection_observations=(
            observer.harness_projection_observations
        ),
        harness_projection_state_factory_calls=(
            observer.harness_projection_state_factory_calls
        ),
        production_accumulator_instances=(
            observer.production_accumulator_instances
        ),
        production_accumulator_add_calls=(
            observer.production_accumulator_add_calls
        ),
        production_reasoning_text_property_reads=(
            observer.production_reasoning_text_property_reads
        ),
        retained_canonical_items=len(state.accumulator.items),
        retained_reasoning_characters=len(reasoning_text),
        retained_reasoning_utf8_bytes=len(reasoning_text.encode("utf-8")),
        max_read_ahead=source.max_read_ahead,
    )
    expected = expected_deterministic_counts(reasoning_deltas)
    assert deterministic == expected, (
        "deterministic benchmark work changed: "
        f"expected={expected!r}, observed={deterministic!r}"
    )
    return deterministic, state


def _project_phase9_fixture(
    fixture: BenchmarkFixture,
) -> DeterministicCounts:
    deterministic, _ = _project_phase9_fixture_retained(fixture)
    return deterministic


async def _project_phase9_fixture_incrementally(
    fixture: BenchmarkFixture,
) -> None:
    """Project a fixture through the incremental runtime consumer path."""

    async def source() -> AsyncIterator[CanonicalStreamItem]:
        for item in fixture.items:
            # Model the await between incrementally delivered stream items.
            await sleep(0)
            yield item

    first = fixture.items[0]
    projected_items = 0
    async for projection in stream_consumer_iterator(
        source(),
        stream_session_id=first.stream_session_id,
        run_id=first.run_id,
        turn_id=first.turn_id,
        unsupported_message="unsupported Phase 9 heartbeat item",
    ):
        assert isinstance(projection, StreamConsumerProjection)
        projected_items += 1
    assert projected_items == len(fixture.items)


def _project_phase9_fixture_retained(
    fixture: BenchmarkFixture,
) -> tuple[DeterministicCounts, StreamProjectionState]:
    observer = _BenchmarkObserver(
        harness_canonical_item_factory_calls=(
            fixture.harness_canonical_item_factory_calls
        )
    )
    state = observer.create_projection_state()
    reasoning_deltas = 0
    projected_items = 0
    source = _TrackedFixtureSource(fixture.items)
    for item in source:
        projection = state.project(
            item,
            item.sequence,
            unsupported_message="unsupported Phase 9 benchmark item",
        )
        projected_items += 1
        observer.record_projection(projection)
        if item.kind is StreamItemKind.REASONING_DELTA:
            reasoning_deltas += 1
        source.mark_consumed()
    state.validate_complete()
    deterministic = DeterministicCounts(
        source_reads=source.read_count,
        emitted_items=source.consumed_count,
        projected_items=projected_items,
        reasoning_deltas=reasoning_deltas,
        harness_canonical_item_factory_calls=(
            observer.harness_canonical_item_factory_calls
        ),
        harness_projection_observations=(
            observer.harness_projection_observations
        ),
        harness_projection_state_factory_calls=(
            observer.harness_projection_state_factory_calls
        ),
        production_accumulator_instances=(
            observer.production_accumulator_instances
        ),
        production_accumulator_add_calls=(
            observer.production_accumulator_add_calls
        ),
        production_reasoning_text_property_reads=(
            observer.production_reasoning_text_property_reads
        ),
        retained_canonical_items=len(state.accumulator.items),
        retained_reasoning_characters=(
            state.accumulator.retained_reasoning_characters
        ),
        retained_reasoning_utf8_bytes=(
            state.accumulator.retained_reasoning_utf8_bytes
        ),
        max_read_ahead=source.max_read_ahead,
    )
    expected = expected_phase9_deterministic_counts(reasoning_deltas)
    assert deterministic == expected, (
        "deterministic Phase 9 benchmark work changed: "
        f"expected={expected!r}, observed={deterministic!r}"
    )
    return deterministic, state


def _isolated_memory_probe(
    delta_count: int,
    delta_text: str,
    protocol: BenchmarkProtocol,
) -> MemoryProbeResult:
    assert type(delta_count) is int and delta_count > 0
    completed = run(
        [
            executable,
            str(Path(__file__).resolve()),
            "--memory-probe-deltas",
            str(delta_count),
            "--memory-probe-text",
            delta_text,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=protocol.memory_subprocess_timeout_seconds,
    )
    assert type(completed.returncode) is int and completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    payloads = [
        line.removeprefix(_MEMORY_SENTINEL)
        for line in completed.stdout.splitlines()
        if line.startswith(_MEMORY_SENTINEL)
    ]
    assert len(payloads) == 1, completed.stdout + completed.stderr
    payload = strict_json_loads(payloads[0])
    assert isinstance(payload, dict)
    assert set(payload) == {
        "peak_processing_bytes_excluding_source_fixture",
        "current_retained_bytes_including_source_fixture",
        "peak_total_bytes_including_source_fixture",
    }
    return MemoryProbeResult(
        peak_processing_bytes_excluding_source_fixture=cast(
            int, payload.get("peak_processing_bytes_excluding_source_fixture")
        ),
        current_retained_bytes_including_source_fixture=cast(
            int, payload.get("current_retained_bytes_including_source_fixture")
        ),
        peak_total_bytes_including_source_fixture=cast(
            int, payload.get("peak_total_bytes_including_source_fixture")
        ),
    )


def _isolated_phase9_memory_probe(
    delta_count: int,
    delta_text: str,
    protocol: BenchmarkProtocol,
) -> MemoryProbeResult:
    assert type(delta_count) is int and delta_count > 0
    completed = run(
        [
            executable,
            str(Path(__file__).resolve()),
            "--phase9-memory-probe-deltas",
            str(delta_count),
            "--memory-probe-text",
            delta_text,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=protocol.memory_subprocess_timeout_seconds,
    )
    assert type(completed.returncode) is int and completed.returncode == 0, (
        completed.stdout + completed.stderr
    )
    payloads = [
        line.removeprefix(_PHASE9_MEMORY_SENTINEL)
        for line in completed.stdout.splitlines()
        if line.startswith(_PHASE9_MEMORY_SENTINEL)
    ]
    assert len(payloads) == 1, completed.stdout + completed.stderr
    payload = strict_json_loads(payloads[0])
    assert isinstance(payload, dict)
    assert set(payload) == {
        "peak_processing_bytes_excluding_source_fixture",
        "current_retained_bytes_including_source_fixture",
        "peak_total_bytes_including_source_fixture",
    }
    return MemoryProbeResult(
        peak_processing_bytes_excluding_source_fixture=cast(
            int, payload.get("peak_processing_bytes_excluding_source_fixture")
        ),
        current_retained_bytes_including_source_fixture=cast(
            int, payload.get("current_retained_bytes_including_source_fixture")
        ),
        peak_total_bytes_including_source_fixture=cast(
            int, payload.get("peak_total_bytes_including_source_fixture")
        ),
    )


def _memory_probe(delta_count: int, delta_text: str) -> MemoryProbeResult:
    fixture = _fixture_items(delta_count, delta_text)
    collect()
    start()
    try:
        _, processing_state = _project_fixture_retained(fixture)
        assert processing_state.accumulator.items
        _, processing_peak = get_traced_memory()
    finally:
        stop()

    collect()
    start()
    try:
        total_fixture = _fixture_items(delta_count, delta_text)
        _, total_state = _project_fixture_retained(total_fixture)
        assert total_fixture.items
        assert total_state.accumulator.items
        total_current, total_peak = get_traced_memory()
    finally:
        stop()
    return MemoryProbeResult(
        peak_processing_bytes_excluding_source_fixture=processing_peak,
        current_retained_bytes_including_source_fixture=total_current,
        peak_total_bytes_including_source_fixture=total_peak,
    )


def _phase9_memory_probe(
    delta_count: int,
    delta_text: str,
) -> MemoryProbeResult:
    fixture = _phase9_fixture_items(delta_count, delta_text)
    collect()
    start()
    try:
        _, processing_state = _project_phase9_fixture_retained(fixture)
        assert processing_state.accumulator.items
        _, processing_peak = get_traced_memory()
    finally:
        stop()

    collect()
    start()
    try:
        total_fixture = _phase9_fixture_items(delta_count, delta_text)
        _, total_state = _project_phase9_fixture_retained(total_fixture)
        assert total_fixture.items
        assert total_state.accumulator.items
        total_current, total_peak = get_traced_memory()
    finally:
        stop()
    return MemoryProbeResult(
        peak_processing_bytes_excluding_source_fixture=processing_peak,
        current_retained_bytes_including_source_fixture=total_current,
        peak_total_bytes_including_source_fixture=total_peak,
    )


class _TrackedFixtureSource:
    """Observe source reads and acknowledgements without read-ahead."""

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = iter(items)
        self.read_count = 0
        self.consumed_count = 0
        self.max_read_ahead = 0

    def __iter__(self) -> "_TrackedFixtureSource":
        return self

    def __next__(self) -> CanonicalStreamItem:
        item = next(self._items)
        self.read_count += 1
        self.max_read_ahead = max(
            self.max_read_ahead,
            self.read_count - self.consumed_count - 1,
        )
        return item

    def mark_consumed(self) -> None:
        """Record that the current pulled item finished projection."""
        assert self.consumed_count < self.read_count
        self.consumed_count += 1


def _nearest_rank(values: list[float], percentile: int) -> float:
    assert values
    assert type(percentile) is int and 0 < percentile <= 100
    ordered = sorted(values)
    rank = ceil((percentile / 100) * len(ordered))
    return ordered[rank - 1]


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Run the locked local reasoning-summary benchmark."
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--phase9", action="store_true")
    parser.add_argument("--phase9-async-probe", action="store_true")
    parser.add_argument("--memory-probe-deltas", type=int, default=None)
    parser.add_argument(
        "--phase9-memory-probe-deltas",
        type=int,
        default=None,
    )
    parser.add_argument("--memory-probe-text", default="x")
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and write or print its JSON report."""
    args = _parse_args()
    if args.phase9_async_probe:
        metrics = asyncio_run(
            _phase9_asynchronous_metrics(benchmark_protocol())
        )
        print(f"{_PHASE9_ASYNC_SENTINEL}" f"{dumps(metrics, sort_keys=True)}")
        return 0
    if args.phase9_memory_probe_deltas is not None:
        with _network_denied():
            memory = _phase9_memory_probe(
                args.phase9_memory_probe_deltas,
                args.memory_probe_text,
            )
        print(
            f"{_PHASE9_MEMORY_SENTINEL}"
            f"{dumps(asdict(memory), sort_keys=True)}"
        )
        return 0
    if args.memory_probe_deltas is not None:
        with _network_denied():
            memory = _memory_probe(
                args.memory_probe_deltas,
                args.memory_probe_text,
            )
        print(f"{_MEMORY_SENTINEL}{dumps(asdict(memory), sort_keys=True)}")
        return 0
    report = run_phase9_benchmark() if args.phase9 else run_benchmark()
    hard_gate: Phase9HardGateResult | None = None
    if args.phase9:
        hard_gate = evaluate_phase9_hard_gate(report)
        report["hard_gate"] = asdict(hard_gate)
    output = dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is None:
        print(output, end="")
    else:
        args.json_out.write_text(output, encoding="utf-8")
    if hard_gate is not None and not hard_gate.passed:
        for failure_reason in hard_gate.failure_reasons:
            print(f"Phase 9 hard gate failed: {failure_reason}", file=stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
