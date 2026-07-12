#!/usr/bin/env python
"""Run the deterministic credential-free reasoning stream benchmark."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from gc import collect
from json import dumps
from math import ceil, isfinite
from pathlib import Path
from platform import machine, platform, processor, python_implementation
from statistics import median
from subprocess import run
from sys import executable, version
from time import perf_counter
from tracemalloc import get_traced_memory, start, stop
from typing import Any, cast
from unittest.mock import patch

from reasoning_summary_json import strict_json_loads

from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProjectionState,
    StreamTerminalOutcome,
    StreamVisibility,
)

_MEMORY_SENTINEL = "__REASONING_BENCHMARK_MEMORY__"
_MEMORY_SCOPES = (
    "processing_excluding_source_fixture",
    "retained_total_including_source_fixture",
)


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


def _item(
    sequence: int,
    kind: StreamItemKind,
    *,
    text_delta: str | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
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
    parser.add_argument("--memory-probe-deltas", type=int, default=None)
    parser.add_argument("--memory-probe-text", default="x")
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and write or print its JSON report."""
    args = _parse_args()
    if args.memory_probe_deltas is not None:
        with _network_denied():
            memory = _memory_probe(
                args.memory_probe_deltas,
                args.memory_probe_text,
            )
        print(f"{_MEMORY_SENTINEL}{dumps(asdict(memory), sort_keys=True)}")
        return 0
    report = run_benchmark()
    output = dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is None:
        print(output, end="")
    else:
        args.json_out.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
