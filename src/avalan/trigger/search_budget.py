"""Bound cumulative schedule work across one owned asynchronous operation."""

from .definition import integer
from .error import TriggerError, TriggerErrorCode

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import isfinite
from time import monotonic


class SearchWorkExhausted(TriggerError):
    """Distinguish an operation limit from an invalid schedule."""


@dataclass(slots=True, kw_only=True)
class SearchWorkBudget:
    remaining: int
    deadline: float
    clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        integer(self.remaining, 1, 100000, "search.candidates")
        assert isfinite(self.deadline)
        assert callable(self.clock)

    def consume(self) -> None:
        if self.clock() >= self.deadline:
            raise SearchWorkExhausted(
                TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "search.deadline"
            )
        if self.remaining == 0:
            raise SearchWorkExhausted(
                TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED, "search.cumulative"
            )
        self.remaining -= 1


_active: ContextVar[tuple[SearchWorkBudget, ...]] = ContextVar(
    "trigger_search_work", default=()
)


@contextmanager
def schedule_work(budget: SearchWorkBudget) -> Iterator[None]:
    """Charge this operation and any enclosing operation without resets."""
    assert isinstance(budget, SearchWorkBudget)
    token = _active.set((*_active.get(), budget))
    try:
        yield
    finally:
        _active.reset(token)


def charge_schedule_work() -> None:
    """Charge each active owner before evaluating a schedule candidate."""
    for budget in _active.get():
        budget.consume()
