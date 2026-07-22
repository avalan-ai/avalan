from asyncio import (
    CancelledError,
    create_task,
    run,
    sleep,
)
from asyncio import (
    Event as AsyncEvent,
)
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from gc import collect
from statistics import median
from time import perf_counter
from tracemalloc import (
    get_traced_memory,
)
from tracemalloc import (
    start as start_tracing,
)
from tracemalloc import (
    stop as stop_tracing,
)
from typing import Any, cast
from unittest import TestCase

from coverage import Coverage

from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    LocalTextStreamEventParser,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProducerBackend,
    StreamProjectionState,
    StreamProviderCapabilities,
    StreamProviderEvent,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    iter_stream_consumer_projections,
    normalize_provider_stream,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class _BenchmarkSample:
    label: str
    model: str
    time_to_first_token_seconds: float
    total_seconds: float
    estimated_tokens_per_second_total: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _LocalStreamLatencySample:
    tokens_read: int
    items: tuple[CanonicalStreamItem, ...]
    first_item_ms: float
    first_answer_ms: float
    per_item_us: float


class _ImmediateTokens:
    _count: int
    read_count: int

    def __init__(self, count: int) -> None:
        assert count > 0
        self._count = count
        self.read_count = 0

    def __aiter__(self) -> "_ImmediateTokens":
        return self

    async def __anext__(self) -> str:
        self.read_count += 1
        if self.read_count <= self._count:
            return "x"
        raise StopAsyncIteration


@contextmanager
def _coverage_suspended_during_timing() -> Iterator[None]:
    coverage = Coverage.current()
    if coverage is None:
        yield
        return

    coverage.stop()
    try:
        yield
    finally:
        coverage.start()


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="benchmark-stream",
        run_id="benchmark-run",
        turn_id="benchmark-turn",
        sequence=sequence,
        kind=kind,
        channel=_channel(kind),
        text_delta=text_delta,
        usage=cast(Any, usage),
        terminal_outcome=terminal_outcome,
    )


def _channel(kind: StreamItemKind) -> StreamChannel:
    if kind in {StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE}:
        return StreamChannel.ANSWER
    return StreamChannel.CONTROL


def _long_stream_items(count: int) -> Iterable[CanonicalStreamItem]:
    assert count > 0
    yield _item(StreamItemKind.STREAM_STARTED, 0)
    for sequence in range(1, count + 1):
        yield _item(
            StreamItemKind.ANSWER_DELTA,
            sequence,
            text_delta="x",
        )
    yield _item(StreamItemKind.ANSWER_DONE, count + 1)
    yield _item(
        StreamItemKind.STREAM_COMPLETED,
        count + 2,
        usage={"output_tokens": count},
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )
    yield _item(StreamItemKind.STREAM_CLOSED, count + 3)


async def _long_stream(count: int) -> AsyncIterator[CanonicalStreamItem]:
    for item in _long_stream_items(count):
        yield item


async def _local_text_events(
    chunks: AsyncIterator[str],
) -> AsyncIterator[StreamProviderEvent]:
    parser = LocalTextStreamEventParser(parse_tool_calls=False)
    chunks_exhausted = False
    try:
        while True:
            try:
                chunk = await chunks.__anext__()
            except StopAsyncIteration:
                chunks_exhausted = True
                break
            for event in parser.push(chunk):
                yield event
        for event in parser.flush():
            yield event
    finally:
        if not chunks_exhausted:
            await chunks.aclose()


def _local_text_stream(
    chunks: AsyncIterator[str],
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
) -> AsyncIterator[CanonicalStreamItem]:
    return normalize_provider_stream(
        _local_text_events(chunks),
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        capabilities=StreamProviderCapabilities(
            backend=StreamProducerBackend.LOCAL,
            supports_reasoning=True,
            supports_tool_calls=False,
            supports_cancellation=True,
            max_queue_depth=StreamPerformanceBudget().max_queue_depth,
        ),
    )


async def _sample_local_stream_latency(
    count: int,
) -> _LocalStreamLatencySample:
    tokens = _ImmediateTokens(count)
    stream = _local_text_stream(
        tokens,
        stream_session_id="latency-stream",
        run_id="latency-run",
        turn_id="latency-turn",
    )
    started = perf_counter()
    first_item = await stream.__anext__()
    first_item_ms = (perf_counter() - started) * 1000
    first_answer = await stream.__anext__()
    first_answer_ms = (perf_counter() - started) * 1000
    items = [first_item, first_answer]
    async for item in stream:
        items.append(item)
    elapsed_us = (perf_counter() - started) * 1_000_000
    return _LocalStreamLatencySample(
        tokens_read=tokens.read_count,
        items=tuple(items),
        first_item_ms=first_item_ms,
        first_answer_ms=first_answer_ms,
        per_item_us=elapsed_us / len(items),
    )


async def _collect_projections(
    count: int,
) -> tuple[StreamConsumerProjection, ...]:
    return tuple(
        [
            projection
            async for projection in iter_stream_consumer_projections(
                _long_stream(count)
            )
        ]
    )


def _sample(
    label: str,
    model: str,
    time_to_first_token_seconds: float,
    total_seconds: float,
    estimated_tokens_per_second_total: float,
) -> _BenchmarkSample:
    return _BenchmarkSample(
        label=label,
        model=model,
        time_to_first_token_seconds=time_to_first_token_seconds,
        total_seconds=total_seconds,
        estimated_tokens_per_second_total=estimated_tokens_per_second_total,
    )


def _benchmark_samples() -> tuple[_BenchmarkSample, ...]:
    return (
        _sample(
            "main",
            "openai",
            3.407790416997159,
            6.969596957991598,
            38.883166649867945,
        ),
        _sample(
            "main",
            "hermes",
            0.35346041599405,
            11.261452707985882,
            29.303501826721316,
        ),
        _sample(
            "main",
            "gpt-oss",
            0.5907667500141542,
            4.169321375025902,
            88.50361169333165,
        ),
        _sample("phase-1 18:10", "openai", 2.427, 6.317, 37.67),
        _sample("phase-1 18:10", "hermes", 0.367, 11.253, 29.33),
        _sample("phase-1 18:10", "gpt-oss", 0.577, 4.160, 88.70),
        _sample(
            "phase-1 18:11",
            "openai",
            3.0114846249925904,
            6.398630915995454,
            45.79104559188614,
        ),
        _sample(
            "phase-1 18:11",
            "hermes",
            0.35945095799979754,
            11.247989167022752,
            29.338577331449212,
        ),
        _sample(
            "phase-1 18:11",
            "gpt-oss",
            0.5817514579975978,
            4.159713290981017,
            88.70803687361249,
        ),
        _sample(
            "phase-2",
            "openai",
            2.440512667002622,
            4.854216500010807,
            69.8361929261386,
        ),
        _sample(
            "phase-2",
            "hermes",
            0.3528359580086544,
            11.247650292003527,
            29.339461259265164,
        ),
        _sample(
            "phase-2",
            "gpt-oss",
            0.5406505000137258,
            4.117946209007641,
            89.6077756413732,
        ),
        _sample(
            "phase-3",
            "openai",
            2.991492125001969,
            8.43187337499694,
            36.409465174174464,
        ),
        _sample(
            "phase-3",
            "hermes",
            0.36436404098640196,
            11.205010457983008,
            29.45111039721445,
        ),
        _sample(
            "phase-3",
            "gpt-oss",
            0.5647680000110995,
            4.15602233298705,
            88.78681836504698,
        ),
        _sample(
            "phase-4",
            "openai",
            2.355923208000604,
            5.209747332992265,
            60.65553275469553,
        ),
        _sample(
            "phase-4",
            "hermes",
            0.3602247910166625,
            11.27325045800535,
            29.272834949361098,
        ),
        _sample(
            "phase-4",
            "gpt-oss",
            0.6243803329998627,
            4.092844916012837,
            87.22539146383828,
        ),
        _sample("phase-5", "openai", 2.736, 6.860, 51.31),
        _sample("phase-5", "hermes", 0.356, 11.269, 29.28),
        _sample("phase-5", "gpt-oss", 0.547, 4.004, 89.17),
        _sample("phase-6 03:09", "openai", 2.627, 5.218, 53.47),
        _sample("phase-6 03:09", "hermes", 0.358, 11.269, 29.28),
        _sample("phase-6 03:09", "gpt-oss", 0.520, 4.210, 90.49),
        _sample("phase-6 10:08", "openai", 3.344, 6.191, 46.52),
        _sample("phase-6 10:08", "hermes", 0.365, 11.286, 29.24),
        _sample("phase-6 10:08", "gpt-oss", 0.554, 4.261, 89.42),
    )


def _benchmark_final_samples() -> tuple[_BenchmarkSample, ...]:
    return (
        _sample(
            "final",
            "openai",
            5.428630083973985,
            7.73037058400223,
            31.95189639564746,
        ),
        _sample(
            "final",
            "hermes",
            0.3526068750070408,
            11.121671458997298,
            29.671798993220033,
        ),
        _sample(
            "final",
            "gpt-oss",
            0.487567666976247,
            4.1379106669919565,
            92.31712106479425,
        ),
    )


def _sample_for_model(
    samples: tuple[_BenchmarkSample, ...],
    model: str,
) -> _BenchmarkSample:
    for sample in samples:
        if sample.model == model:
            return sample
    raise AssertionError(f"missing benchmark sample for {model}")


def _comparison_samples_for_model(
    samples: tuple[_BenchmarkSample, ...],
    model: str,
) -> tuple[_BenchmarkSample, ...]:
    selected = tuple(sample for sample in samples if sample.model == model)
    assert selected, f"missing benchmark comparison samples for {model}"
    return selected


def _regression_labels(
    final_sample: _BenchmarkSample,
    comparison_samples: tuple[_BenchmarkSample, ...],
    metric: str,
    *,
    lower_is_better: bool,
) -> tuple[str, ...]:
    final_value = getattr(final_sample, metric)
    labels: list[str] = []
    for sample in comparison_samples:
        comparison_value = getattr(sample, metric)
        regressed = (
            final_value > comparison_value
            if lower_is_better
            else final_value < comparison_value
        )
        if regressed:
            labels.append(sample.label)
    return tuple(labels)


_BENCHMARK_MARKDOWN_ROWS = (
    ("main", "2026-06-13T14:36:57.608061+00:00"),
    ("phase-1", "2026-06-13T18:10:16.848632+00:00"),
    ("phase-1", "2026-06-13T18:11:09.177804+00:00"),
    ("phase-2", "2026-06-13T20:46:50.296402+00:00"),
    ("phase-3", "2026-06-14T02:12:54.527172+00:00"),
    ("phase-4", "2026-06-14T15:35:39.917338+00:00"),
    ("phase-5", "2026-06-14T21:03:15.003184+00:00"),
    ("phase-6", "2026-06-15T03:09:19.922841+00:00"),
    ("phase-6", "2026-06-15T10:08:29.700366+00:00"),
    ("canonical-phase-7", "2026-06-16T06:30:51.716877+00:00"),
)
_BENCHMARK_MARKDOWN_COMPARISON_REFERENCES = (
    ("main", ""),
    ("phase-1", "2026-06-13T18:10:16.848632+00:00"),
    ("phase-1", "2026-06-13T18:11:09.177804+00:00"),
    ("phase-2", ""),
    ("phase-3", ""),
    ("phase-4", ""),
    ("phase-5", ""),
    ("phase-6", "2026-06-15T03:09:19.922841+00:00"),
    ("phase-6", "2026-06-15T10:08:29.700366+00:00"),
)
_BENCHMARK_MARKDOWN_MODELS = ("openai", "hermes", "gpt-oss")
_BENCHMARK_MARKDOWN_METRICS = ("TTFT", "Total time", "Estimated tokens/s")


def _benchmark_row_marker(row: str) -> str:
    return f"## {row} baseline - "


def _final_benchmark_markdown_section(markdown: str) -> str:
    start = markdown.find(_benchmark_row_marker("canonical-phase-7"))
    assert start >= 0, "missing final benchmark row"
    next_start = markdown.find("\n## ", start + 1)
    if next_start < 0:
        return markdown[start:]
    return markdown[start:next_start]


def _final_acceptance_markdown_section(markdown: str) -> str:
    start = markdown.find("### Final acceptance comparison")
    assert start >= 0, "missing final acceptance comparison"
    next_start = markdown.find("\n### ", start + 1)
    if next_start < 0:
        return markdown[start:]
    return markdown[start:next_start]


def _model_acceptance_note(section: str, model: str) -> str:
    marker = f"- `{model}`:"
    start = section.find(marker)
    assert start >= 0, f"missing acceptance note for model: {model}"
    next_start = section.find("\n- `", start + 1)
    if next_start < 0:
        return section[start:]
    return section[start:next_start]


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _assert_reference_present(
    section: str,
    label: str,
    timestamp: str,
) -> None:
    assert (
        f"`{label}`" in section
    ), f"missing comparison row reference: {label} {timestamp}"
    if timestamp:
        assert (
            timestamp in section
        ), f"missing comparison row reference: {label} {timestamp}"


def _assert_final_benchmark_markdown_complete(markdown: str) -> None:
    for row, timestamp in _BENCHMARK_MARKDOWN_ROWS:
        marker = _benchmark_row_marker(row)
        assert (
            f"{marker}{timestamp}" in markdown
        ), f"missing benchmark row: {row} {timestamp}"

    section = _final_benchmark_markdown_section(markdown)
    for model in _BENCHMARK_MARKDOWN_MODELS:
        assert (
            f"| `{model}` | ok |" in section
        ), f"missing result row for model: {model}"
        assert (
            f"- `{model}`:" in section
        ), f"missing comparison note for model: {model}"
    for metric in _BENCHMARK_MARKDOWN_METRICS:
        assert metric in section, f"missing comparison metric: {metric}"

    acceptance_section = _final_acceptance_markdown_section(markdown)
    expected_final_result = (
        "Final result: `canonical-phase-7 baseline - "
        "2026-06-16T06:30:51.716877+00:00`"
    )
    assert _normalized_text(expected_final_result) in _normalized_text(
        acceptance_section
    ), "missing final canonical phase-7 result label"
    for label, timestamp in _BENCHMARK_MARKDOWN_COMPARISON_REFERENCES:
        _assert_reference_present(acceptance_section, label, timestamp)

    openai_note = _normalized_text(
        _model_acceptance_note(acceptance_section, "openai")
    )
    assert (
        "accepted hosted regression" in openai_note
    ), "missing OpenAI accepted hosted regression rationale"
    assert (
        "hosted provider/service variance" in openai_note
    ), "missing OpenAI hosted variance rationale"
    for term in ("ttft", "total time", "estimated tokens/s"):
        assert (
            term in openai_note
        ), f"missing OpenAI accepted regression metric: {term}"

    gpt_oss_note = _model_acceptance_note(
        acceptance_section,
        "gpt-oss",
    )
    gpt_oss_note = _normalized_text(gpt_oss_note)
    assert (
        "accepted minor total-time variance" in gpt_oss_note
    ), "missing GPT-OSS accepted total-time rationale"
    assert (
        "output-length" in gpt_oss_note
    ), "missing GPT-OSS output-length rationale"
    assert (
        "single-run variance" in gpt_oss_note
    ), "missing GPT-OSS single-run rationale"
    assert (
        "throughput improved" in gpt_oss_note
    ), "missing GPT-OSS throughput rationale"

    hermes_note = _normalized_text(
        _model_acceptance_note(acceptance_section, "hermes")
    )
    assert (
        "no accepted regression needed" in hermes_note
    ), "missing Hermes no-regression rationale"

    assert "remaining openai hosted and gpt-oss total-time variances" in (
        _normalized_text(acceptance_section)
    ), "missing final per-model variance summary"


def _complete_benchmark_markdown() -> str:
    return """
# Streaming Benchmarks

## main baseline - 2026-06-13T14:36:57.608061+00:00

## phase-1 baseline - 2026-06-13T18:10:16.848632+00:00

## phase-1 baseline - 2026-06-13T18:11:09.177804+00:00

## phase-2 baseline - 2026-06-13T20:46:50.296402+00:00

## phase-3 baseline - 2026-06-14T02:12:54.527172+00:00

## phase-4 baseline - 2026-06-14T15:35:39.917338+00:00

## phase-5 baseline - 2026-06-14T21:03:15.003184+00:00

## phase-6 baseline - 2026-06-15T03:09:19.922841+00:00

## phase-6 baseline - 2026-06-15T10:08:29.700366+00:00

## canonical-phase-7 baseline - 2026-06-16T06:30:51.716877+00:00

| Model | Status | Load s | TTFT s | Total s | Est. tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `openai` | ok | 0.148 | 5.429 | 7.730 | 31.95 |
| `hermes` | ok | 5.963 | 0.353 | 11.122 | 29.67 |
| `gpt-oss` | ok | 3.195 | 0.488 | 4.138 | 92.32 |

Comparison with the `main` baseline from
`2026-06-13T14:36:57.608061+00:00`, the latest `phase-1` row from
`2026-06-13T18:11:09.177804+00:00`, the `phase-2` row from
`2026-06-13T20:46:50.296402+00:00`, the `phase-3` row from
`2026-06-14T02:12:54.527172+00:00`, the `phase-4` row from
`2026-06-14T15:35:39.917338+00:00`, the `phase-5` row from
`2026-06-14T21:03:15.003184+00:00`, and the latest `phase-6` row from
`2026-06-15T10:08:29.700366+00:00`:

- `openai`: TTFT was 5.429s, slower than 3.408s on `main`. Total time
  was 7.730s, slower than `main`. Estimated tokens/s was lower than
  `main`; the slowdown is accepted as hosted provider variance.
- `hermes`: TTFT stayed flat. Total time improved. Estimated tokens/s
  improved.
- `gpt-oss`: TTFT improved. Total time was slightly slower than the
  fastest prior rows. Estimated tokens/s improved, and the total-time
  delta is explained by single-run noise.

No intentional performance regressions were introduced. Local rows stayed
within single-run noise.

### Final acceptance comparison

Final result: `canonical-phase-7 baseline -
2026-06-16T06:30:51.716877+00:00`.

Comparison scope: the final result was compared with the `main` baseline
and every prior recorded phase row in this file: `phase-1` at
`2026-06-13T18:10:16.848632+00:00`, `phase-1` at
`2026-06-13T18:11:09.177804+00:00`, `phase-2`, `phase-3`, `phase-4`,
`phase-5`, `phase-6` at `2026-06-15T03:09:19.922841+00:00`, and
`phase-6` at `2026-06-15T10:08:29.700366+00:00`. The comparison used
TTFT, total wall time, and estimated tokens/s as the acceptance metrics.

- `openai`: accepted hosted regression. Final TTFT was 5.429s, slower
  than `main` at 3.408s and every prior phase row. Final total time was
  7.730s, slower than `main` and every prior phase row except the
  slower `phase-3` hosted row. Final estimated tokens/s was 31.95,
  lower than `main` and every prior phase row. This is explicitly
  accepted as hosted provider/service variance because the local MLX
  rows did not show the same slowdown.
- `hermes`: no accepted regression needed. Final TTFT was at or below
  `main` and every prior phase row, final total time was the fastest
  recorded row, and final estimated tokens/s was the highest recorded
  row.
- `gpt-oss`: accepted minor total-time variance only. Final TTFT was the
  fastest recorded row and final estimated tokens/s was the highest
  recorded row. Final total time improved versus `main`, both
  `phase-1` rows, `phase-3`, and both `phase-6` rows, but was slower
  than the fastest `phase-2`, `phase-4`, and `phase-5` rows. This is
  accepted as output-length and single-run variance because throughput
  improved while the final row emitted more events/output.

Final acceptance: benchmark regressions are resolved by deterministic
budget tests where local code paths are under Avalan control, and the
remaining OpenAI hosted and GPT-OSS total-time variances are explicitly
accepted above.
"""


def _assert_local_stream_latency_shape(
    test_case: TestCase,
    sample: _LocalStreamLatencySample,
    count: int,
) -> None:
    test_case.assertEqual(sample.tokens_read, count + 1)
    test_case.assertEqual(len(sample.items), count + 4)
    test_case.assertIs(sample.items[0].kind, StreamItemKind.STREAM_STARTED)
    test_case.assertIs(sample.items[1].kind, StreamItemKind.ANSWER_DELTA)
    test_case.assertIs(sample.items[-3].kind, StreamItemKind.ANSWER_DONE)
    test_case.assertIs(
        sample.items[-2].kind,
        StreamItemKind.STREAM_COMPLETED,
    )
    test_case.assertIs(sample.items[-1].kind, StreamItemKind.STREAM_CLOSED)


class StreamBenchmarkRegressionTestCase(TestCase):
    def test_final_benchmark_markdown_parser_accepts_complete_rows(
        self,
    ) -> None:
        _assert_final_benchmark_markdown_complete(
            _complete_benchmark_markdown()
        )

    def test_final_benchmark_markdown_parser_rejects_missing_baseline(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "## main baseline - ",
            "## control baseline - ",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing benchmark row: main",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_missing_phase_row(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "## phase-5 baseline - 2026-06-14T21:03:15.003184+00:00",
            "## phase-5 baseline - missing",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing benchmark row: phase-5",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_missing_phase1_reference(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "`2026-06-13T18:10:16.848632+00:00`",
            "`2026-06-13T18:10:00+00:00`",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing comparison row reference: phase-1",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_missing_phase6_reference(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "`2026-06-15T03:09:19.922841+00:00`",
            "`2026-06-15T03:09:00+00:00`",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing comparison row reference: phase-6",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_legacy_phase7_result(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "Final result: `canonical-phase-7 baseline -",
            "Final result: `phase-7 baseline -",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing final canonical phase-7 result label",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_missing_openai_acceptance(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "accepted hosted regression",
            "noted hosted regression",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing OpenAI accepted hosted regression rationale",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_markdown_parser_rejects_missing_gpt_oss_accept(
        self,
    ) -> None:
        markdown = _complete_benchmark_markdown().replace(
            "accepted minor total-time variance",
            "noted minor total-time variance",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "missing GPT-OSS accepted total-time rationale",
        ):
            _assert_final_benchmark_markdown_complete(markdown)

    def test_final_benchmark_comparison_matrix_covers_recorded_rows(
        self,
    ) -> None:
        samples = _benchmark_samples()
        final_samples = _benchmark_final_samples()
        expected_labels = (
            "main",
            "phase-1 18:10",
            "phase-1 18:11",
            "phase-2",
            "phase-3",
            "phase-4",
            "phase-5",
            "phase-6 03:09",
            "phase-6 10:08",
        )

        for model in ("openai", "hermes", "gpt-oss"):
            with self.subTest(model=model):
                comparison_samples = _comparison_samples_for_model(
                    samples,
                    model,
                )
                self.assertEqual(
                    tuple(sample.label for sample in comparison_samples),
                    expected_labels,
                )

        openai = _sample_for_model(final_samples, "openai")
        openai_samples = _comparison_samples_for_model(samples, "openai")
        self.assertEqual(
            _regression_labels(
                openai,
                openai_samples,
                "time_to_first_token_seconds",
                lower_is_better=True,
            ),
            expected_labels,
        )
        self.assertEqual(
            _regression_labels(
                openai,
                openai_samples,
                "total_seconds",
                lower_is_better=True,
            ),
            (
                "main",
                "phase-1 18:10",
                "phase-1 18:11",
                "phase-2",
                "phase-4",
                "phase-5",
                "phase-6 03:09",
                "phase-6 10:08",
            ),
        )
        self.assertEqual(
            _regression_labels(
                openai,
                openai_samples,
                "estimated_tokens_per_second_total",
                lower_is_better=False,
            ),
            expected_labels,
        )

        hermes = _sample_for_model(final_samples, "hermes")
        hermes_samples = _comparison_samples_for_model(samples, "hermes")
        for metric, lower_is_better in (
            ("time_to_first_token_seconds", True),
            ("total_seconds", True),
            ("estimated_tokens_per_second_total", False),
        ):
            with self.subTest(model="hermes", metric=metric):
                self.assertEqual(
                    _regression_labels(
                        hermes,
                        hermes_samples,
                        metric,
                        lower_is_better=lower_is_better,
                    ),
                    (),
                )

        gpt_oss = _sample_for_model(final_samples, "gpt-oss")
        gpt_oss_samples = _comparison_samples_for_model(samples, "gpt-oss")
        self.assertEqual(
            _regression_labels(
                gpt_oss,
                gpt_oss_samples,
                "time_to_first_token_seconds",
                lower_is_better=True,
            ),
            (),
        )
        self.assertEqual(
            _regression_labels(
                gpt_oss,
                gpt_oss_samples,
                "total_seconds",
                lower_is_better=True,
            ),
            ("phase-2", "phase-4", "phase-5"),
        )
        self.assertEqual(
            _regression_labels(
                gpt_oss,
                gpt_oss_samples,
                "estimated_tokens_per_second_total",
                lower_is_better=False,
            ),
            (),
        )

    def test_benchmark_comparison_rejects_missing_model_row(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "missing benchmark sample for missing",
        ):
            _sample_for_model(_benchmark_final_samples(), "missing")

    def test_long_stream_projection_overhead_within_budget(self) -> None:
        count = 8192
        budget = StreamPerformanceBudget()

        started = perf_counter()
        projections = run(_collect_projections(count))
        elapsed_us = (perf_counter() - started) * 1_000_000
        per_item_us = elapsed_us / len(projections)

        print(
            "phase7 benchmark long_stream_projection "
            f"items={len(projections)} per_item_us={per_item_us:.3f}"
        )
        self.assertEqual(len(projections), count + 4)
        self.assertIs(
            projections[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertLessEqual(per_item_us, budget.per_item_overhead_us)

    def test_slow_local_consumer_does_not_read_ahead(self) -> None:
        count = 256

        class PullTrackedTokens:
            def __init__(self) -> None:
                self.read_count = 0

            def __aiter__(self) -> "PullTrackedTokens":
                return self

            async def __anext__(self) -> str:
                self.read_count += 1
                if self.read_count <= count:
                    return "x"
                raise StopAsyncIteration

        async def consume() -> tuple[PullTrackedTokens, int, float]:
            tokens = PullTrackedTokens()
            stream = _local_text_stream(
                tokens,
                stream_session_id="slow-stream",
                run_id="slow-run",
                turn_id="slow-turn",
            )
            answer_count = 0
            max_read_ahead = 0
            started = perf_counter()
            async for item in stream:
                if item.kind is not StreamItemKind.ANSWER_DELTA:
                    continue
                answer_count += 1
                max_read_ahead = max(
                    max_read_ahead,
                    tokens.read_count - answer_count,
                )
                await sleep(0)
            return tokens, max_read_ahead, perf_counter() - started

        tokens, max_read_ahead, elapsed = run(consume())

        print(
            "phase7 benchmark slow_consumer "
            f"tokens={count} elapsed_ms={elapsed * 1000:.3f} "
            f"max_read_ahead={max_read_ahead}"
        )
        self.assertEqual(tokens.read_count, count + 1)
        self.assertEqual(max_read_ahead, 0)

    def test_local_stream_latency_and_overhead_within_budget(self) -> None:
        count = 8192
        budget = StreamPerformanceBudget()
        sample_count = 5

        covered_sample = run(_sample_local_stream_latency(count))
        _assert_local_stream_latency_shape(self, covered_sample, count)

        with _coverage_suspended_during_timing():
            run(_sample_local_stream_latency(count))
            samples = tuple(
                run(_sample_local_stream_latency(count))
                for _ in range(sample_count)
            )

        first_item_ms = median(sample.first_item_ms for sample in samples)
        first_answer_ms = median(sample.first_answer_ms for sample in samples)
        per_item_us = median(sample.per_item_us for sample in samples)
        max_per_item_us = max(sample.per_item_us for sample in samples)

        print(
            "phase7 benchmark local_stream_latency "
            f"tokens={count} samples={sample_count} "
            f"first_item_ms={first_item_ms:.3f} "
            f"first_answer_ms={first_answer_ms:.3f} "
            f"median_per_item_us={per_item_us:.3f} "
            f"max_per_item_us={max_per_item_us:.3f}"
        )
        self.assertLessEqual(first_item_ms, budget.time_to_first_item_ms)
        self.assertLessEqual(first_answer_ms, budget.time_to_first_item_ms)
        self.assertLessEqual(per_item_us, budget.per_item_overhead_us)

    def test_cancellation_latency_within_budget(self) -> None:
        budget = StreamPerformanceBudget()

        class PendingTokens:
            def __init__(self) -> None:
                self.started = AsyncEvent()
                self.cancelled = False
                self.closed = False

            def __aiter__(self) -> "PendingTokens":
                return self

            async def __anext__(self) -> str:
                self.started.set()
                try:
                    await AsyncEvent().wait()
                except CancelledError:
                    self.cancelled = True
                    raise
                return "late"

            async def aclose(self) -> None:
                self.closed = True

        async def cancel_pending_pull() -> tuple[PendingTokens, float]:
            tokens = PendingTokens()
            stream = _local_text_stream(
                tokens,
                stream_session_id="cancel-stream",
                run_id="cancel-run",
                turn_id="cancel-turn",
            )
            started_item = await stream.__anext__()
            self.assertIs(started_item.kind, StreamItemKind.STREAM_STARTED)
            pull = create_task(stream.__anext__())
            await tokens.started.wait()

            started = perf_counter()
            pull.cancel()
            try:
                cancelled_item = await pull
            except CancelledError:
                cancelled_item = None
            if cancelled_item is not None:
                self.assertIs(
                    cancelled_item.kind,
                    StreamItemKind.STREAM_CANCELLED,
                )
            elapsed_ms = (perf_counter() - started) * 1000
            await cast(Any, stream).aclose()
            return tokens, elapsed_ms

        tokens, elapsed_ms = run(cancel_pending_pull())

        print(f"phase7 benchmark cancellation latency_ms={elapsed_ms:.3f}")
        self.assertTrue(tokens.cancelled)
        self.assertTrue(tokens.closed)
        self.assertLessEqual(elapsed_ms, budget.cancellation_latency_ms)

    def test_long_stream_retention_peak_memory_within_budget(self) -> None:
        count = 8192
        retention_policy = StreamRetentionPolicy(
            accumulator_item_limit=128,
            replay_history_item_limit=16,
            ui_buffer_item_limit=16,
            metrics_history_item_limit=16,
            event_history_item_limit=16,
            mcp_resource_item_limit=16,
            a2a_task_record_item_limit=16,
            flow_history_item_limit=16,
        )
        accumulator = CanonicalStreamAccumulator(
            retention_policy=retention_policy
        )
        budget = StreamPerformanceBudget()

        collect()
        start_tracing()
        try:
            started = perf_counter()
            for item in _long_stream_items(count):
                accumulator.add(item)
            elapsed = perf_counter() - started
            current, peak = get_traced_memory()
        finally:
            stop_tracing()

        print(
            "phase7 benchmark retention "
            f"items={count + 4} retained={len(accumulator.items)} "
            f"current_bytes={current} peak_bytes={peak} "
            f"elapsed_ms={elapsed * 1000:.3f}"
        )
        self.assertEqual(
            len(accumulator.items),
            retention_policy.accumulator_item_limit,
        )
        self.assertEqual(accumulator.answer_text, "x" * count)
        self.assertEqual(accumulator.final_usage, {"output_tokens": count})
        self.assertLessEqual(peak, budget.max_memory_bytes)

    def test_canonical_projection_hot_path_overhead_within_budget(
        self,
    ) -> None:
        count = 8192
        budget = StreamPerformanceBudget()
        state = StreamProjectionState(
            stream_session_id="canonical-stream",
            run_id="canonical-run",
            turn_id="canonical-turn",
            accumulate=False,
        )

        started = perf_counter()
        for sequence in range(count):
            item = _item(
                StreamItemKind.ANSWER_DELTA,
                sequence,
                text_delta="x",
            )
            source: CanonicalStreamItem | StreamConsumerProjection = (
                item
                if sequence % 2 == 0
                else StreamConsumerProjection.from_item(item)
            )
            projection = state.project(
                source,
                sequence,
                unsupported_message="unsupported benchmark item",
            )
        elapsed_us = (perf_counter() - started) * 1_000_000
        per_item_us = elapsed_us / count

        print(
            "phase7 benchmark canonical_projection_hot_path "
            f"tokens={count} per_item_us={per_item_us:.3f}"
        )
        self.assertEqual(projection.text_delta, "x")
        self.assertEqual(projection.sequence, count - 1)
        self.assertEqual(state.accumulator.items, ())
        self.assertLessEqual(per_item_us, budget.per_item_overhead_us)

    def test_accumulated_canonical_projection_hot_path_overhead_within_budget(
        self,
    ) -> None:
        count = 8192
        budget = StreamPerformanceBudget()
        state = StreamProjectionState(
            stream_session_id="benchmark-stream",
            run_id="benchmark-run",
            turn_id="benchmark-turn",
            accumulate=True,
        )
        projection: StreamConsumerProjection | None = None

        started = perf_counter()
        for item in _long_stream_items(count):
            projection = state.project(
                item,
                item.sequence,
                unsupported_message="unsupported benchmark item",
            )
        state.validate_complete()
        elapsed_us = (perf_counter() - started) * 1_000_000
        per_item_us = elapsed_us / (count + 4)

        print(
            "phase7 benchmark accumulated_canonical_projection_hot_path "
            f"tokens={count} per_item_us={per_item_us:.3f}"
        )
        self.assertIsNotNone(projection)
        self.assertIs(projection.kind, StreamItemKind.STREAM_CLOSED)
        self.assertEqual(state.accumulator.answer_text, "x" * count)
        self.assertEqual(
            state.accumulator.final_usage,
            {"output_tokens": count},
        )
        self.assertLessEqual(per_item_us, budget.per_item_overhead_us)

    def test_rejects_late_content_during_projection_benchmark(
        self,
    ) -> None:
        async def invalid_stream() -> AsyncIterator[CanonicalStreamItem]:
            yield _item(StreamItemKind.STREAM_STARTED, 0)
            yield _item(
                StreamItemKind.STREAM_COMPLETED,
                1,
                usage={"output_tokens": 0},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late")

        async def collect_invalid() -> None:
            _ = [
                projection
                async for projection in iter_stream_consumer_projections(
                    invalid_stream()
                )
            ]

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            run(collect_invalid())

    def test_projection_benchmark_rejects_unsupported_item(self) -> None:
        state = StreamProjectionState(
            stream_session_id="legacy-stream",
            run_id="legacy-run",
            turn_id="legacy-turn",
            accumulate=False,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported benchmark item",
        ):
            state.project(
                object(),
                0,
                unsupported_message="unsupported benchmark item",
            )

    def test_latency_and_overhead_budgets_reject_non_positive_values(
        self,
    ) -> None:
        cases = (
            {"time_to_first_item_ms": 0},
            {"per_item_overhead_us": 0},
        )

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    StreamPerformanceBudget(**kwargs)
