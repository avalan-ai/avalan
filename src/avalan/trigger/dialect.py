from .error import TriggerError, TriggerErrorCode

from calendar import monthrange
from dataclasses import dataclass
from re import fullmatch, split

_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ),
        1,
    )
}
_WEEKDAYS = {
    name: index
    for index, name in enumerate(
        (
            "SUN",
            "MON",
            "TUE",
            "WED",
            "THU",
            "FRI",
            "SAT",
        )
    )
}
_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


@dataclass(frozen=True, slots=True)
class CronFields:
    expression: str
    values: tuple[tuple[int, ...], ...]
    unrestricted_dom: bool
    unrestricted_dow: bool

    def branches(self) -> tuple[str, ...]:
        """Split Unix day OR so an impossible DOM cannot hide weekdays."""
        minute, hour, dom, month, dow = (
            ",".join(str(value) for value in field) for field in self.values
        )
        dom_possible = any(
            day <= monthrange(2000, selected)[1]
            for day in self.values[2]
            for selected in self.values[3]
        )
        if self.unrestricted_dom:
            return (f"{minute} {hour} * {month} {dow}",)
        branches = []
        if dom_possible:
            branches.append(f"{minute} {hour} {dom} {month} *")
        if not self.unrestricted_dow:
            branches.append(f"{minute} {hour} * {month} {dow}")
        if not branches:
            raise TriggerError(
                TriggerErrorCode.IMPOSSIBLE_SCHEDULE, "schedule.expression"
            )
        return tuple(branches)


def parse_cron(expression: str) -> CronFields:
    """Validate the five-field v1 dialect without importing a parser."""
    if (
        not isinstance(expression, str)
        or len(expression) > 1024
        or fullmatch(r"[A-Za-z0-9*,/\- \t\r\n\f\v]+", expression) is None
    ):
        raise TriggerError(
            TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
        )
    fields = split(r"[ \t\r\n\f\v]+", expression.strip())
    if len(fields) != 5:
        raise TriggerError(
            TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
        )
    values = tuple(
        _field(part.upper(), index) for index, part in enumerate(fields)
    )
    result = CronFields(
        " ".join(fields).upper(), values, fields[2] == "*", fields[4] == "*"
    )
    result.branches()
    return result


def _number(value: str, index: int) -> int:
    names = _MONTHS if index == 3 else _WEEKDAYS if index == 4 else {}
    if value in names:
        return names[value]
    if fullmatch(r"[0-9]{1,2}", value) is None:
        raise TriggerError(
            TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
        )
    number = int(value)
    low, high = _BOUNDS[index]
    if not low <= number <= high:
        raise TriggerError(
            TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
        )
    return number


def _field(value: str, index: int) -> tuple[int, ...]:
    low, high = _BOUNDS[index]
    selected: set[int] = set()
    for term in value.split(","):
        pieces = term.split("/")
        if len(pieces) > 2:
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
            )
        atom = pieces[0]
        step = 1
        if len(pieces) == 2:
            if (atom != "*" and "-" not in atom) or fullmatch(
                r"[0-9]{1,2}", pieces[1]
            ) is None:
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
                )
            step = int(pieces[1])
            if not 1 <= step <= (7 if index == 4 else high - low + 1):
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
                )
        if atom == "*":
            start, end = low, high
        elif "-" in atom:
            endpoints = atom.split("-")
            if len(endpoints) != 2:
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
                )
            start, end = (_number(item, index) for item in endpoints)
            if start > end:
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "schedule.expression"
                )
        else:
            start = end = _number(atom, index)
        selected.update(range(start, end + 1, step))
    return tuple(
        sorted({number % 7 for number in selected} if index == 4 else selected)
    )
