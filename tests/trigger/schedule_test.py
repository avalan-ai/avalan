from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from pytest import mark, raises

from avalan.trigger import (
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    TriggerError,
    TriggerErrorCode,
)
from avalan.trigger.schedule import (
    SearchLimits,
    latest_due,
    next_occurrence,
    preview,
    resolve_anchor,
    schedule_provenance,
    validate_registration_time,
)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


@mark.parametrize(
    ("expression", "base", "zone", "reverse", "expected"),
    [
        (
            "30 2 * * *",
            "2026-03-07T08:00:00+00:00",
            "America/New_York",
            False,
            "2026-03-09T06:30:00+00:00",
        ),
        (
            "* * * * *",
            "2026-11-01T05:59:00+00:00",
            "America/New_York",
            False,
            "2026-11-01T07:00:00+00:00",
        ),
        (
            "* * * * *",
            "2026-11-01T06:15:00+00:00",
            "America/New_York",
            False,
            "2026-11-01T07:00:00+00:00",
        ),
        (
            "* * * * *",
            "2026-11-01T06:15:00+00:00",
            "America/New_York",
            True,
            "2026-11-01T05:59:00+00:00",
        ),
        (
            "* * * * *",
            "2026-09-07T12:00:00+00:00",
            "UTC",
            True,
            "2026-09-07T12:00:00+00:00",
        ),
        (
            "* * * * *",
            "9999-12-31T23:59:59.999999+00:00",
            "UTC",
            True,
            "9999-12-31T23:59:00+00:00",
        ),
        (
            "15 2 * * *",
            "2026-10-03T15:00:00+00:00",
            "Australia/Lord_Howe",
            False,
            "2026-10-04T15:15:00+00:00",
        ),
        (
            "* * * * *",
            "2026-04-04T15:10:00+00:00",
            "Australia/Lord_Howe",
            True,
            "2026-04-04T14:59:00+00:00",
        ),
        (
            "0 12 * * *",
            "2011-12-29T23:00:00+00:00",
            "Pacific/Apia",
            False,
            "2011-12-30T22:00:00+00:00",
        ),
        (
            "0 0 29 2 *",
            "2099-09-07T00:00:00+00:00",
            "UTC",
            True,
            "2096-02-29T00:00:00+00:00",
        ),
        (
            "0 0 31 2 MON",
            "2026-09-08T00:00:00+00:00",
            "UTC",
            False,
            "2027-02-01T00:00:00+00:00",
        ),
        (
            "0 0 31 2 MON",
            "2026-09-08T00:00:00+00:00",
            "UTC",
            True,
            "2026-02-23T00:00:00+00:00",
        ),
        (
            "0 9 */1 * MON",
            "2026-09-08T00:00:00+00:00",
            "UTC",
            False,
            "2026-09-08T09:00:00+00:00",
        ),
        (
            "0 9 * * MON",
            "2026-09-08T00:00:00+00:00",
            "UTC",
            False,
            "2026-09-14T09:00:00+00:00",
        ),
        (
            "0 9 1 * MON",
            "2026-09-08T00:00:00+00:00",
            "UTC",
            False,
            "2026-09-14T09:00:00+00:00",
        ),
        (
            "* * * * *",
            "0001-01-01T00:00:00+00:00",
            "UTC",
            False,
            "0001-01-01T00:01:00+00:00",
        ),
    ],
)
def test_calendar_conformance(
    expression: str, base: str, zone: str, reverse: bool, expected: str
) -> None:
    schedule = CronTrigger(expression=expression, timezone=zone)
    result = (
        latest_due(
            schedule,
            _time(base),
            first_undecided=datetime(1, 1, 1, tzinfo=UTC),
        )
        if reverse
        else next_occurrence(schedule, _time(base))
    )
    assert result == _time(expected)
    assert result is not None
    assert result.astimezone(ZoneInfo(zone)).fold == 0


def test_interval_uses_integer_microsecond_anchor_arithmetic() -> None:
    start = _time("2026-09-07T12:00:00.123456+00:00")
    schedule = IntervalTrigger(every_seconds=10, start_at=start)
    assert (
        next_occurrence(schedule, start - timedelta(microseconds=1)) == start
    )
    assert next_occurrence(schedule, start) == start + timedelta(seconds=10)
    assert latest_due(schedule, start, first_undecided=start) == start
    assert latest_due(
        schedule,
        start + timedelta(seconds=29, microseconds=999999),
        first_undecided=start,
    ) == start + timedelta(seconds=20)
    assert (
        latest_due(
            schedule,
            start - timedelta(seconds=1),
            first_undecided=start - timedelta(seconds=2),
        )
        is None
    )
    assert (
        latest_due(
            schedule, start, first_undecided=start + timedelta(seconds=1)
        )
        is None
    )
    assert (
        latest_due(
            schedule,
            start + timedelta(seconds=9),
            first_undecided=start + timedelta(seconds=1),
        )
        is None
    )
    with raises(TriggerError):
        next_occurrence(schedule, start, anchor=start + timedelta(seconds=1))
    with raises(TriggerError):
        next_occurrence(IntervalTrigger(every_seconds=1), start)


def test_at_exhaustion_and_registration_equality() -> None:
    instant = _time("2026-09-07T12:00:00+00:00")
    schedule = AtTrigger(at=instant)
    assert (
        next_occurrence(schedule, instant - timedelta(microseconds=1))
        == instant
    )
    assert next_occurrence(schedule, instant) is None
    assert (
        latest_due(
            schedule,
            instant - timedelta(microseconds=1),
            first_undecided=instant - timedelta(seconds=1),
        )
        is None
    )
    assert latest_due(schedule, instant, first_undecided=instant) == instant
    validate_registration_time(schedule, instant)
    validate_registration_time(IntervalTrigger(every_seconds=1), instant)
    validate_registration_time(CronTrigger(expression="* * * * *"), instant)
    with raises(TriggerError) as error:
        validate_registration_time(
            schedule, instant + timedelta(microseconds=1)
        )
    assert error.value.code == TriggerErrorCode.PAST_SCHEDULE


def test_preview_labels_implicit_anchor_and_reuses_one_budget() -> None:
    instant = _time("2026-09-07T12:00:00+00:00")
    schedule = IntervalTrigger(every_seconds=10)
    result = preview(schedule, reference_time=instant, count=2)
    assert result.assumed_anchor
    assert result.effective_anchor == instant + timedelta(seconds=10)
    assert result.occurrences == (
        instant + timedelta(seconds=10),
        instant + timedelta(seconds=20),
    )
    assert result.timezone == "UTC"
    assert result.local_times[0].endswith(".000000+00:00")
    assert not preview(
        schedule, reference_time=instant, anchor=instant
    ).assumed_anchor
    at = AtTrigger(at=instant + timedelta(seconds=1))
    assert preview(at, reference_time=instant).occurrences == (at.at,)
    assert resolve_anchor(at, instant) is None
    with raises(TriggerError) as error:
        preview(
            schedule,
            reference_time=instant,
            count=2,
            limits=SearchLimits(candidates=1),
        )
    assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
    cron = preview(
        CronTrigger(expression="0 9 * * *", timezone="America/New_York"),
        reference_time=instant,
        count=1,
    )
    assert cron.local_times == ("2026-09-07T09:00:00.000000-04:00",)


def test_search_and_preview_bounds() -> None:
    for settings in (
        {"years": 0},
        {"years": 401},
        {"candidates": True},
        {"candidates": 100001},
    ):
        with raises(TriggerError):
            SearchLimits(**settings)
    instant = _time("2026-09-07T12:00:00+00:00")
    with raises(TriggerError):
        preview(AtTrigger(at=instant), reference_time=instant, count=101)
    with raises(TriggerError) as error:
        next_occurrence(
            CronTrigger(expression="0 0 29 2 *"),
            instant,
            limits=SearchLimits(years=1),
        )
    assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
    with raises(TriggerError) as error:
        latest_due(
            CronTrigger(expression="* * * * *", timezone="America/New_York"),
            _time("2026-11-01T06:15:00+00:00"),
            first_undecided=instant,
            limits=SearchLimits(candidates=1),
        )
    assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
    assert (
        latest_due(
            CronTrigger(expression="0 0 1 * MON"),
            instant,
            first_undecided=instant - timedelta(seconds=1),
        )
        is None
    )


@mark.parametrize(
    "schedule",
    [
        CronTrigger(expression="* * * * *"),
        IntervalTrigger(
            every_seconds=315360000, start_at=datetime(9999, 1, 1, tzinfo=UTC)
        ),
    ],
)
def test_datetime_overflow_is_not_search_exhaustion(
    schedule: CronTrigger | IntervalTrigger,
) -> None:
    with raises(TriggerError) as error:
        next_occurrence(schedule, datetime.max.replace(tzinfo=UTC))
    assert error.value.code == TriggerErrorCode.DATETIME_OVERFLOW
    with raises(TriggerError):
        resolve_anchor(
            IntervalTrigger(every_seconds=1), datetime.max.replace(tzinfo=UTC)
        )


def test_missing_or_mixed_parser_versions_fail_explicitly() -> None:
    instant = _time("2026-09-07T12:00:00+00:00")
    for result in (PackageNotFoundError("croniter"), "6.0.0"):
        kwargs = (
            {"side_effect": result}
            if isinstance(result, Exception)
            else {"return_value": result}
        )
        with (
            patch("avalan.trigger.schedule.version", **kwargs),
            raises(TriggerError) as error,
        ):
            next_occurrence(CronTrigger(expression="* * * * *"), instant)
        assert error.value.code == TriggerErrorCode.CAPABILITY_UNAVAILABLE


def test_provenance_reports_timezone_source(tmp_path: Path) -> None:
    with (
        patch("avalan.trigger.schedule.TZPATH", ()),
        patch(
            "avalan.trigger.schedule.version", side_effect=["6.2.4", "2026.1"]
        ),
    ):
        assert schedule_provenance().timezone_data == "tzdata:2026.1"
    with (
        patch(
            "avalan.trigger.schedule.version",
            side_effect=["6.2.4", PackageNotFoundError()],
        ),
        patch("avalan.trigger.schedule.TZPATH", (str(tmp_path),)),
    ):
        assert schedule_provenance().timezone_data == "unavailable"
    (tmp_path / "UTC").write_bytes(b"zone fixture")
    with (
        patch("avalan.trigger.schedule.version", return_value="6.2.4"),
        patch("avalan.trigger.schedule.TZPATH", (str(tmp_path),)),
    ):
        value = schedule_provenance().timezone_data
        assert value.startswith("system:sha256:")
        assert len(value.removeprefix("system:sha256:")) == 64
        with (
            patch(
                "avalan.trigger.schedule.Path.read_bytes",
                side_effect=PermissionError("private"),
            ),
            raises(TriggerError),
        ):
            schedule_provenance()


def test_gap_rejection_does_not_reset_absolute_search_horizon() -> None:
    schedule = CronTrigger(
        expression="30 2 8 3 *", timezone="America/New_York"
    )
    with raises(TriggerError) as error:
        next_occurrence(
            schedule,
            _time("2025-03-09T12:00:00+00:00"),
            limits=SearchLimits(years=1),
        )
    assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED


def test_non_interval_anchor_is_rejected_instead_of_ignored() -> None:
    instant = _time("2026-09-07T12:00:00+00:00")
    for schedule in (
        CronTrigger(expression="* * * * *"),
        AtTrigger(at=instant),
    ):
        with raises(TriggerError) as error:
            next_occurrence(schedule, instant, anchor=instant)
        assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE


def test_sparse_dom_branch_does_not_hide_weekday_or_candidate() -> None:
    schedule = CronTrigger(expression="0 0 29 2 MON")
    limits = SearchLimits(years=1)
    assert next_occurrence(
        schedule, _time("2026-09-07T00:00:00+00:00"), limits=limits
    ) == _time("2027-02-01T00:00:00+00:00")
    assert latest_due(
        schedule,
        _time("2027-09-07T00:00:00+00:00"),
        first_undecided=_time("2026-01-01T00:00:00+00:00"),
        limits=limits,
    ) == _time("2027-02-22T00:00:00+00:00")
    # The first branch exhausts the single shared candidate budget. This
    # cannot be turned into absence merely because the other branch is valid.
    with raises(TriggerError) as error:
        next_occurrence(
            schedule,
            _time("2026-09-07T00:00:00+00:00"),
            limits=SearchLimits(years=1, candidates=1),
        )
    assert error.value.path == "search.candidates"


def test_or_merge_keeps_budget_uncertainty_after_a_candidate() -> None:
    schedule = CronTrigger(expression="0 0 1 * MON")
    with raises(TriggerError) as error:
        next_occurrence(
            schedule,
            _time("2026-09-07T00:00:00+00:00"),
            limits=SearchLimits(candidates=1),
        )
    assert error.value.code == TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED
    assert error.value.path == "search.candidates"
