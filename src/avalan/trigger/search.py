"""Share one finite candidate budget across a transaction decision plan."""

from .definition import TriggerSpec, timestamp
from .schedule import SearchLimits, _Budget, _calculate

from datetime import datetime


class ScheduleSearch:
    """Use the canonical schedule engine with cumulative candidate charging."""

    def __init__(self, limits: SearchLimits = SearchLimits()) -> None:
        self.limits = limits
        self._budget = _Budget(limits.candidates)

    def next(
        self,
        schedule: TriggerSpec,
        after: datetime,
        *,
        anchor: datetime | None,
    ) -> datetime | None:
        return _calculate(
            schedule,
            timestamp(after),
            anchor,
            False,
            None,
            self.limits,
            self._budget,
        )

    def latest(
        self,
        schedule: TriggerSpec,
        through: datetime,
        *,
        first: datetime,
        anchor: datetime | None,
    ) -> datetime | None:
        cursor, lower = timestamp(through), timestamp(first)
        if lower > cursor:
            return None
        return _calculate(
            schedule,
            cursor,
            anchor,
            True,
            lower,
            self.limits,
            self._budget,
        )
