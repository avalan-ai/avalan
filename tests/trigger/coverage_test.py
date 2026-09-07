"""Verify range boundaries, slot identity and finite exact-count checking."""

from .records_test import NOW, OWNER, definition, occurrence, span

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from pytest import raises

from avalan.trigger import AtPolicy, AtTrigger, IntervalTrigger
from avalan.trigger.coverage import decisions_overlap, validate_decisions
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import occurrence_id


def test_decision_overlap_uses_half_open_utc_ranges() -> None:
    first = span()
    slot = occurrence()
    assert decisions_overlap(first, slot)
    assert decisions_overlap(slot, first)
    assert decisions_overlap(slot, slot)
    assert decisions_overlap(first, first)
    adjacent = replace(
        first,
        first_at=first.until_at,
        until_at=first.until_at + timedelta(minutes=1),
        decided_at=first.until_at,
    )
    assert not decisions_overlap(first, adjacent)
    assert not decisions_overlap(slot, adjacent)
    assert not decisions_overlap(adjacent, slot)
    assert not decisions_overlap(first, replace(first, revision=2))


def test_validate_checks_the_actual_schedule_and_existing_decisions() -> None:
    registered = replace(definition(), resolved_anchor=NOW)
    validate_decisions(registered, (occurrence(),))
    validate_decisions(registered, (span(),))
    with raises(TriggerError):
        validate_decisions(registered, (replace(span(), revision=2),))
    with raises(TriggerError):
        validate_decisions(definition(), (span(),))
    with raises(TriggerError):
        validate_decisions(registered, (span(),), existing=(occurrence(),))
    with raises(TriggerError):
        validate_decisions(registered, (span(), occurrence()))
    future = replace(span(), until_at=NOW + timedelta(minutes=2))
    with raises(TriggerError):
        validate_decisions(registered, (future,))


def test_exact_counts_are_proved_and_share_a_finite_budget() -> None:
    registered = replace(definition(), resolved_anchor=NOW)
    validate_decisions(registered, (replace(span(), exact_count=1),))
    with raises(TriggerError):
        validate_decisions(registered, (replace(span(), exact_count=2),))
    first = replace(span(), exact_count=1)
    next_time = NOW + timedelta(minutes=1)
    second = replace(
        span(),
        span_id="second",
        first_at=next_time,
        until_at=next_time + timedelta(minutes=1),
        decided_at=next_time,
        exact_count=1,
    )
    with raises(TriggerError) as error:
        validate_decisions(registered, (first, second), count_limit=1)
    assert error.value.code is TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
    validate_decisions(registered, (first, second), count_limit=2)


def test_one_shot_count_stops_at_schedule_exhaustion() -> None:
    registered = replace(
        definition(),
        schedule=AtTrigger(at=NOW),
        policy=AtPolicy(),
        resolved_anchor=None,
    )
    validate_decisions(registered, (replace(span(), exact_count=1),))
    later = NOW + timedelta(microseconds=1)
    invalid = replace(
        occurrence(),
        scheduled_at=later,
        decided_at=later,
        occurrence_id=occurrence_id(OWNER, "trigger", 1, later),
    )
    with raises(TriggerError):
        validate_decisions(registered, (invalid,))


def test_exact_count_does_not_search_beyond_its_known_final_slot() -> None:
    registered = replace(definition(), resolved_anchor=NOW)
    two = replace(
        span(),
        until_at=NOW + timedelta(minutes=2),
        decided_at=NOW + timedelta(minutes=1),
        exact_count=2,
    )
    validate_decisions(registered, (two,))
    end = datetime.max.replace(tzinfo=UTC)
    first = end - timedelta(microseconds=1)
    near_end = replace(
        definition(),
        schedule=IntervalTrigger(every_seconds=1),
        resolved_anchor=first,
    )
    last = replace(
        span(), first_at=first, until_at=end, decided_at=end, exact_count=1
    )
    validate_decisions(near_end, (last,))
