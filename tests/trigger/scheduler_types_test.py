from .records_test import occurrence, span

from dataclasses import replace

from pytest import mark, raises

from avalan.trigger.error import TriggerError
from avalan.trigger.records import OccurrenceDisposition
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
    TriggerShutdownResult,
)


@mark.parametrize(
    "field,maximum",
    [
        ("discovery_limit", 1000),
        ("decisions_per_tick", 1000),
        ("decisions_per_trigger", 100),
        ("admissions_per_tick", 1000),
        ("admissions_per_trigger", 100),
        ("candidate_evaluations", 100000),
        ("search_years", 400),
        ("search_candidates", 100000),
        ("tick_timeout_seconds", 60),
        ("preparation_timeout_seconds", 30),
        ("transaction_timeout_seconds", 30),
        ("poll_interval_seconds", 60),
        ("admission_retry_attempts", 20),
        ("retry_base_seconds", 60),
        ("retry_max_seconds", 3600),
        ("stale_plan_retries", 10),
        ("shutdown_timeout_seconds", 60),
        ("diagnostic_bytes", 4096),
        ("orphan_grace_seconds", 2592000),
    ],
)
def test_host_bounds_reject_unbounded_settings(
    field: str, maximum: int
) -> None:
    settings = TriggerSchedulerSettings()
    assert getattr(replace(settings, **{field: maximum}), field) == maximum
    for invalid in (0, maximum + 1, True):
        with raises(TriggerError):
            replace(settings, **{field: invalid})


def test_result_counts_do_not_invent_unknown_span_cardinality() -> None:
    admitted = occurrence()
    skipped = replace(
        admitted, run_id=None, disposition=OccurrenceDisposition.EXPIRED
    )
    result = TriggerProcessResult(occurrences=(admitted, skipped))
    assert result.admitted == 1 and result.skipped == 1
    assert (
        replace(result, ranges=(replace(span(), exact_count=2),)).skipped == 3
    )
    assert (
        replace(result, ranges=(replace(span(), exact_count=None),)).skipped
        is None
    )
    assert TriggerShutdownResult(pending_operations=0).settled
    assert not TriggerShutdownResult(pending_operations=1).settled


def test_conflicts_include_prior_completion_and_full_discovery_batch() -> None:
    settings = TriggerSchedulerSettings(discovery_limit=1000)
    result = TriggerProcessResult(conflicts=1 + settings.discovery_limit)
    assert result.conflicts == 1001
    with raises(TriggerError):
        replace(result, conflicts=1002)
