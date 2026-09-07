from .preparation_fault_test import services
from .records_test import NOW, OWNER, event, occurrence, span
from .registration_test import registration

from asyncio import run
from dataclasses import replace
from datetime import timedelta
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from avalan.trigger.error import TriggerErrorCode
from avalan.trigger.observability import (
    ObservedTriggerScheduler,
    TriggerMetrics,
    replay_events,
)
from avalan.trigger.records import OccurrenceDisposition, occurrence_id
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerDiagnostic,
)
from avalan.trigger.stores.memory import InMemoryTriggerStore


def test_metrics_are_bounded_and_do_not_claim_unknown_counts(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        metrics = TriggerMetrics(tmp_path / "metrics")
        admitted = replace(
            occurrence(),
            disposition=OccurrenceDisposition.ADMITTED,
            run_id="private-run",
            decided_at=NOW + timedelta(seconds=3),
        )
        expired = replace(
            occurrence(),
            disposition=OccurrenceDisposition.EXPIRED,
            run_id=None,
        )
        counted = replace(span(), exact_count=3)
        unknown = replace(span(), exact_count=None)
        await metrics.emit_tick(
            TriggerProcessResult(
                occurrences=(admitted, expired),
                ranges=(counted, unknown),
                conflicts=2,
                admission_retries=1,
                errors=(
                    TriggerSchedulerDiagnostic(
                        trigger_name="private-trigger",
                        code=TriggerErrorCode.ADMISSION_RETRYABLE,
                        detail="not-exported",
                    ),
                ),
                remaining_work=True,
                pending_operations=1,
            )
        )
        output = metrics.path.read_text()
        assert 'disposition="admitted"} 1' in output
        assert 'disposition="expired"} 1' in output
        assert metrics.decisions[counted.disposition] == 3
        assert "dispatch_lag_seconds_sum 3.0" in output
        assert "admission_retries_total 1" in output
        assert "scheduler_healthy 0" in output
        assert "remaining 1" in output and "pending 1" in output
        assert "private" not in output and "not-exported" not in output
        await metrics.emit_tick(TriggerProcessResult())
        assert "scheduler_healthy 1" in metrics.path.read_text()
        assert metrics.dispatch_lag_count == 1

    run(scenario())


def test_event_replay_retains_ids_and_reports_sink_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "events"
        record = event()
        assert (await replay_events((record,), path)).delivered == 1
        assert (await replay_events((record,), path)).delivered == 1
        values = [loads(line) for line in path.read_text().splitlines()]
        assert values[0] == values[1]
        assert values[0]["payload"]["event_id"] == record.event_id
        assert (await replay_events((), path)).failed == 0
        failed = await replay_events(
            (record, record), tmp_path / "missing/events"
        )
        assert failed.delivered == 0 and failed.failed == 2

    run(scenario())


def test_external_sink_failure_cannot_change_settled_tick() -> None:
    async def scenario() -> None:
        with TemporaryDirectory() as directory:
            preparation = await services(Path(directory))
            scheduler = ObservedTriggerScheduler(preparation)
            tick = TriggerProcessResult(conflicts=1)
            with patch(
                "avalan.trigger.observability.TriggerScheduler._process_once",
                AsyncMock(return_value=tick),
            ):
                assert await scheduler.process_once() is tick
                sink = AsyncMock()
                sink.emit_tick.side_effect = OSError("private")
                scheduler.sink = sink
                assert await scheduler.process_once() is tick
                assert scheduler.sink_failures == 1
                sink.emit_tick.side_effect = None
                assert await scheduler.process_once() is tick
                assert scheduler.sink_failures == 1
            assert (await scheduler.shutdown()).settled

    run(scenario())


def test_recent_memory_history_reverses_the_same_bounded_order() -> None:
    async def scenario() -> None:
        from_record = occurrence()
        second = replace(
            from_record,
            occurrence_id=occurrence_id(
                OWNER, "trigger", 1, NOW + timedelta(minutes=1)
            ),
            scheduled_at=NOW + timedelta(minutes=1),
            decided_at=NOW + timedelta(minutes=1),
        )
        store = InMemoryTriggerStore(OWNER, clock=lambda: NOW)
        await store.apply(registration(), expected_generation=None)
        store._occurrences["daily"] = (from_record, second)
        forward = await store.occurrences("daily", limit=1)
        reverse = await store.occurrences("daily", limit=1, newest_first=True)
        assert forward.items == (from_record,)
        assert reverse.items == (second,)
        assert reverse.next_cursor is not None
        assert (
            await store.occurrences(
                "daily", cursor=reverse.next_cursor, newest_first=True
            )
        ).items == (from_record,)

    run(scenario())
