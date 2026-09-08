from .definition import (
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    TriggerSpec,
    integer,
    timestamp,
)
from .dialect import parse_cron
from .error import TriggerError, TriggerErrorCode
from .search_budget import charge_schedule_work

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo

CRONITER_VERSION = "6.2.4"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchLimits:
    years: int = 8
    candidates: int = 10000

    def __post_init__(self) -> None:
        integer(self.years, 1, 400, "search.years")
        integer(self.candidates, 1, 100000, "search.candidates")


@dataclass(slots=True)
class _Budget:
    remaining: int

    def consume(self) -> None:
        if self.remaining == 0:
            raise TriggerError(
                TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "search.candidates"
            )
        charge_schedule_work()
        self.remaining -= 1


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulePreview:
    occurrences: tuple[datetime, ...]
    local_times: tuple[str, ...]
    timezone: str
    effective_anchor: datetime | None
    assumed_anchor: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleProvenance:
    semantics_version: int
    parser_version: str
    timezone_data: str


def schedule_provenance(timezone: str = "UTC") -> ScheduleProvenance:
    """Identify the selected zone data, honoring ZoneInfo search order."""
    _parser()
    CronTrigger(expression="* * * * *", timezone=timezone)
    for root in TZPATH:
        source = Path(root) / timezone
        if source.is_file():
            try:
                zone_version = (
                    "system:sha256:" + sha256(source.read_bytes()).hexdigest()
                )
            except OSError:
                raise TriggerError(
                    TriggerErrorCode.CAPABILITY_UNAVAILABLE,
                    "schedule.timezone_data",
                ) from None
            break
    else:
        try:
            zone_version = "tzdata:" + version("tzdata")
        except PackageNotFoundError:
            zone_version = "unavailable"
    return ScheduleProvenance(
        semantics_version=1,
        parser_version=CRONITER_VERSION,
        timezone_data=zone_version,
    )


def _parser() -> object:
    try:
        if version("croniter") != CRONITER_VERSION:
            raise TriggerError(
                TriggerErrorCode.CAPABILITY_UNAVAILABLE,
                "schedule.parser_version",
            )
        return import_module("croniter")
    except (ImportError, PackageNotFoundError):
        raise TriggerError(
            TriggerErrorCode.CAPABILITY_UNAVAILABLE, "schedule.parser"
        ) from None


def resolve_anchor(
    schedule: TriggerSpec, reference_time: datetime
) -> datetime | None:
    """Resolve an unregistered interval anchor for explicit preview use."""
    reference = timestamp(reference_time, "reference_time")
    if isinstance(schedule, IntervalTrigger):
        return schedule.start_at or _add(
            reference, timedelta(seconds=schedule.every_seconds)
        )
    return None


def validate_registration_time(
    schedule: TriggerSpec, reference_time: datetime
) -> None:
    """Reject a newly registered supplied first firing in the past."""
    reference = timestamp(reference_time, "reference_time")
    supplied = (
        schedule.at
        if isinstance(schedule, AtTrigger)
        else (
            schedule.start_at
            if isinstance(schedule, IntervalTrigger)
            else None
        )
    )
    if supplied is not None and supplied < reference:
        raise TriggerError(TriggerErrorCode.PAST_SCHEDULE, "schedule")


def next_occurrence(
    schedule: TriggerSpec,
    after: datetime,
    *,
    anchor: datetime | None = None,
    limits: SearchLimits = SearchLimits(),
) -> datetime | None:
    """Return the first instant strictly after the UTC cursor."""
    return _calculate(
        schedule,
        timestamp(after, "after"),
        anchor,
        False,
        None,
        limits,
        _Budget(limits.candidates),
    )


def latest_due(
    schedule: TriggerSpec,
    decision_time: datetime,
    *,
    first_undecided: datetime,
    anchor: datetime | None = None,
    limits: SearchLimits = SearchLimits(),
) -> datetime | None:
    """Find the latest due slot without traversing historical backlog."""
    cursor = timestamp(decision_time, "decision_time")
    lower = timestamp(first_undecided, "first_undecided")
    if lower > cursor:
        return None
    return _calculate(
        schedule,
        cursor,
        anchor,
        True,
        lower,
        limits,
        _Budget(limits.candidates),
    )


def preview(
    schedule: TriggerSpec,
    *,
    reference_time: datetime,
    count: int = 5,
    anchor: datetime | None = None,
    limits: SearchLimits = SearchLimits(),
) -> SchedulePreview:
    """Preview bounded future instants and label an assumed anchor."""
    integer(count, 1, 100, "preview.count")
    reference = timestamp(reference_time, "reference_time")
    effective = (
        timestamp(anchor, "anchor")
        if anchor is not None
        else resolve_anchor(schedule, reference)
    )
    budget = _Budget(limits.candidates)
    values: list[datetime] = []
    for _ in range(count):
        value = _calculate(
            schedule, reference, effective, False, None, limits, budget
        )
        if value is None:
            break
        values.append(value)
        reference = value
    timezone = (
        schedule.timezone if isinstance(schedule, CronTrigger) else "UTC"
    )
    return SchedulePreview(
        occurrences=tuple(values),
        local_times=tuple(
            value.astimezone(ZoneInfo(timezone)).isoformat(
                timespec="microseconds"
            )
            for value in values
        ),
        timezone=timezone,
        effective_anchor=effective,
        assumed_anchor=isinstance(schedule, IntervalTrigger)
        and schedule.start_at is None
        and anchor is None,
    )


def _calculate(
    schedule: TriggerSpec,
    cursor: datetime,
    anchor: datetime | None,
    reverse: bool,
    lower: datetime | None,
    limits: SearchLimits,
    budget: _Budget,
) -> datetime | None:
    if anchor is not None and not isinstance(schedule, IntervalTrigger):
        raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "anchor")
    if isinstance(schedule, CronTrigger):
        return _cron(schedule, cursor, reverse, lower, limits, budget)
    budget.consume()
    if isinstance(schedule, AtTrigger):
        result = schedule.at
        if result > cursor if reverse else result <= cursor:
            return None
    else:
        assert isinstance(schedule, IntervalTrigger)
        start = (
            timestamp(anchor, "anchor")
            if anchor is not None
            else schedule.start_at
        )
        if start is None:
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, "schedule.start_at"
            )
        if schedule.start_at is not None and start != schedule.start_at:
            raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "anchor")
        step = timedelta(seconds=schedule.every_seconds)
        ordinal = (cursor - start) // step
        if reverse and ordinal < 0:
            return None
        ordinal = max(0, ordinal + (0 if reverse else 1))
        result = _add(start, ordinal * step)
    return None if lower is not None and result < lower else result


def _add(value: datetime, delta: timedelta) -> datetime:
    try:
        return value + delta
    except OverflowError:
        raise TriggerError(
            TriggerErrorCode.DATETIME_OVERFLOW, "schedule"
        ) from None


def _cron(
    schedule: CronTrigger,
    cursor: datetime,
    reverse: bool,
    lower: datetime | None,
    limits: SearchLimits,
    budget: _Budget,
) -> datetime | None:
    parser = _parser()
    zone = ZoneInfo(schedule.timezone)
    try:
        local = cursor.astimezone(zone)
        wall = local.replace(tzinfo=UTC, second=0, microsecond=0, fold=0)
        if reverse and local.fold:
            first_offset, offset = (
                local.replace(fold=0).utcoffset(),
                local.utcoffset(),
            )
            assert first_offset is not None and offset is not None
            wall += first_offset - offset
        seed = wall + timedelta(seconds=1 if reverse else 0)
        seconds = (seed - _EPOCH) // timedelta(seconds=1)
        candidates: list[datetime] = []
        horizon_exhausted = False
        for expression in parse_cron(schedule.expression).branches():
            iterator = getattr(parser, "croniter")(
                expression, seconds, max_years_between_matches=limits.years
            )
            while True:
                budget.consume()
                try:
                    raw = (
                        iterator.get_prev if reverse else iterator.get_next
                    )(datetime)
                except getattr(parser, "CroniterBadDateError"):
                    # This ordered branch has no match within its year
                    # horizon; another OR branch can still decide the slot.
                    horizon_exhausted = True
                    break
                naive = raw.replace(tzinfo=None)
                if abs(naive.year - local.year) > limits.years:
                    horizon_exhausted = True
                    break
                result = naive.replace(tzinfo=zone, fold=0).astimezone(UTC)
                restored = result.astimezone(zone)
                if restored.replace(tzinfo=None) != naive or restored.fold:
                    continue
                if result > cursor if reverse else result <= cursor:
                    continue
                if lower is None or result >= lower:
                    candidates.append(result)
                break
        if not candidates and horizon_exhausted:
            raise TriggerError(
                TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "search.years"
            )
        return (
            (max(candidates) if reverse else min(candidates))
            if candidates
            else None
        )
    except (OverflowError, ValueError) as error:
        if isinstance(error, TriggerError):
            raise
        raise TriggerError(
            TriggerErrorCode.DATETIME_OVERFLOW, "schedule"
        ) from None
