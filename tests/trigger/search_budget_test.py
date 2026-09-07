"""Prove inherited candidate accounting and operation scope isolation."""

from .records_test import NOW

from asyncio import Event, create_task
from datetime import timedelta
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.trigger.definition import IntervalTrigger
from avalan.trigger.schedule import next_occurrence
from avalan.trigger.search_budget import (
    SearchWorkBudget,
    SearchWorkExhausted,
    schedule_work,
)


class SearchWorkBudgetTest(IsolatedAsyncioTestCase):
    async def test_child_inherits_scope_after_parent_exits(self) -> None:
        ready = Event()
        budget = SearchWorkBudget(remaining=1, deadline=10, clock=lambda: 0)
        schedule = IntervalTrigger(every_seconds=60, start_at=NOW)

        async def child() -> None:
            await ready.wait()
            assert next_occurrence(schedule, NOW) == NOW + timedelta(minutes=1)
            with raises(SearchWorkExhausted, match="search.cumulative"):
                next_occurrence(schedule, NOW)

        with schedule_work(budget):
            running = create_task(child())
        ready.set()
        await running
        assert budget.remaining == 0
        # Another task/caller has no inherited exhausted scope.
        assert next_occurrence(schedule, NOW) == NOW + timedelta(minutes=1)

    async def test_deadline_and_nested_scope_do_not_reset_parent(self) -> None:
        clock = [0.0]
        parent = SearchWorkBudget(
            remaining=2, deadline=10, clock=lambda: clock[0]
        )
        child = SearchWorkBudget(
            remaining=10, deadline=20, clock=lambda: clock[0]
        )
        schedule = IntervalTrigger(every_seconds=60, start_at=NOW)
        with schedule_work(parent):
            with schedule_work(child):
                next_occurrence(schedule, NOW)
            assert parent.remaining == 1 and child.remaining == 9
            clock[0] = 10
            with raises(SearchWorkExhausted, match="search.deadline"):
                next_occurrence(schedule, NOW)
        assert next_occurrence(schedule, NOW) == NOW + timedelta(minutes=1)

    async def test_invalid_budget_and_exception_reset(self) -> None:
        with raises(AssertionError):
            SearchWorkBudget(remaining=1, deadline=float("inf"))
        with raises(ValueError):
            SearchWorkBudget(remaining=0, deadline=10)
        with (
            raises(RuntimeError),
            schedule_work(
                SearchWorkBudget(remaining=1, deadline=10, clock=lambda: 0)
            ),
        ):
            raise RuntimeError("leave scope")
        assert next_occurrence(
            IntervalTrigger(every_seconds=60, start_at=NOW), NOW
        )
