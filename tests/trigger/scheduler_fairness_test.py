"""Prove advancing decision clocks cannot overtake a discovery round."""

from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services
from .records_test import NOW

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from avalan.trigger.admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import (
    IntervalTrigger,
    MisfirePolicy,
    OverlapPolicy,
    RecurringPolicy,
)
from avalan.trigger.preparation import TriggerPreparationService
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_types import TriggerSchedulerSettings


async def register_backlogs(service: TriggerPreparationService) -> None:
    for name in ("first", "second"):
        prepared = await service.prepare_registration(
            replace(
                configuration(),
                name=name,
                schedule=IntervalTrigger(every_seconds=60),
                policy=RecurringPolicy(
                    misfire=MisfirePolicy.ALL, overlap=OverlapPolicy.ALLOW
                ),
            ),
            file_task(),
        )
        await TriggerRegistrationService(service).apply(
            prepared, expected_generation=None
        )


async def prove_advancing_rotation(
    scheduler: TriggerScheduler, advance: Callable[[], Awaitable[datetime]]
) -> None:
    now = await advance()
    blocked = (await scheduler.store.discover(decision_time=now, limit=1))[0]

    async def conflict(
        prepared: PreparedTriggerAdmission,
    ) -> TriggerAdmissionResult:
        assert (
            prepared.plan.snapshot.definition.name == blocked.definition.name
        )
        return TriggerAdmissionResult(
            plan=prepared.plan,
            outcome=TriggerCommitOutcome.NOT_COMMITTED,
            contended=True,
        )

    with patch.object(scheduler.admission, "admit", conflict):
        result = await scheduler.process_once()
    assert result.conflicts == 1 and result.admitted == 0
    assert await scheduler.store.inspect(blocked.definition.name) == blocked
    # Successful admissions advance last_processed_at; the conflicted member
    # must still return within the next round without an artificial write.
    counts = {"first": 0, "second": 0}
    for index in range(6):
        later = await advance()
        assert later > now
        now = later
        result = await scheduler.process_once()
        assert result.admitted == 1 and not result.errors
        for name in counts:
            counts[name] = len((await scheduler.store.occurrences(name)).items)
        if index == 1:
            assert counts[blocked.definition.name] == 1
    assert counts == {"first": 3, "second": 3}


class TriggerSchedulerFairnessTest(IsolatedAsyncioTestCase):
    async def test_advancing_clock_all_backlogs_revisit_initial_conflict(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            service = await services(root)
            await register_backlogs(service)
            scheduler = TriggerScheduler(
                service,
                settings=TriggerSchedulerSettings(
                    discovery_limit=1,
                    decisions_per_trigger=1,
                    admissions_per_trigger=1,
                ),
            )
            now = NOW + timedelta(minutes=10)

            async def advance() -> datetime:
                nonlocal now
                now += timedelta(seconds=1)
                return now

            with patch.object(scheduler.store, "_clock", lambda: now):
                await prove_advancing_rotation(scheduler, advance)
            assert (await scheduler.shutdown()).settled
